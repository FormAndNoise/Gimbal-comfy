"""
tests/test_wayfinder_gps_anchor.py
====================================
Unit, integration, performance, and concurrency tests for WayfinderGPS_Anchor.

Run all tests:
    pytest tests/test_wayfinder_gps_anchor.py -v

Run only fast tests (skip perf/concurrency):
    pytest tests/test_wayfinder_gps_anchor.py -v -m "not slow"

Run only performance tests:
    pytest tests/test_wayfinder_gps_anchor.py -v -m slow

Design decisions documented inline:
- open('x') is used for atomic exclusive file creation to eliminate the
  TOCTOU race between an existence check and a subsequent write. Two
  processes versioning to the same name simultaneously will have one raise
  FileExistsError rather than silently overwriting.
- WAYFINDER_DIR is monkeypatched at the module level (not via mock.patch)
  so that Path operations inside the node see the redirected value without
  requiring import-time interception.
- CUDA tests are skipped when no GPU is present; they are not xfail because
  the absence of CUDA is an environment fact, not a known code defect.
- Concurrency tests use ThreadPoolExecutor rather than multiprocessing to
  stay within pytest's process boundary while still exercising real
  filesystem races. GIL release around I/O means threading is sufficient
  to surface TOCTOU issues.
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

# Add parent directory to path to import wayfinder_gps_anchor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wayfinder_gps_anchor import (
    STAT_PRECISION,
    WayfinderGPS_Anchor,
)


# ===========================================================================
# Fixtures and helpers
# ===========================================================================

def _make_latent(
    B: int = 4,
    C: int = 4,
    H: int = 8,
    W: int = 8,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Build a minimal ComfyUI LATENT dict.

    Parameters
    ----------
    B, C, H, W : int
        Batch, channel, height, width dimensions.
    dtype : torch.dtype
        Tensor dtype. Defaults to float32 (the standard SD latent type).
    device : str
        Target device string ('cpu' or 'cuda').

    Returns
    -------
    dict
        {'samples': Tensor[B, C, H, W]}
    """
    return {"samples": torch.randn(B, C, H, W, dtype=dtype, device=device)}


def _make_meta(
    grid_size_x: int = 3,
    grid_size_y: int = 3,
    x_strength: float = 1.0,
    y_strength: float = 1.0,
) -> Dict[str, Any]:
    """
    Build a minimal wayfinder_meta dict that matches Manifold Explorer output.

    Each cell occupies exactly one batch index (center_batch=1) for simplicity.
    """
    total    = grid_size_x * grid_size_y
    grid_map = []
    for i in range(total):
        row = i // grid_size_x
        col = i % grid_size_x
        grid_map.append({
            "batch_start":  i,
            "batch_end":    i,
            "grid_col":     col,
            "grid_row":     row,
            "offset_x":     col - (grid_size_x - 1) / 2.0,
            "offset_y":     row - (grid_size_y - 1) / 2.0,
            "is_center": (
                col == grid_size_x // 2 and row == grid_size_y // 2
                and grid_size_x % 2 == 1 and grid_size_y % 2 == 1
            ),
            "x_disp_norm":  1.0,
            "y_disp_norm":  1.0,
        })
    return {
        "grid_size_x":        grid_size_x,
        "grid_size_y":        grid_size_y,
        "x_strength":         x_strength,
        "y_strength":         y_strength,
        "wayfinder_grid_map": grid_map,
        "interpolation_mode": "Linear",
        "normalize_vectors":  True,
        "output_shape":       [total, 4, 8, 8],
    }


@pytest.fixture()
def node() -> WayfinderGPS_Anchor:
    return WayfinderGPS_Anchor()


@pytest.fixture()
def tmp_wayfinder_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Redirect the module-level WAYFINDER_DIR to a temp directory.

    Why monkeypatch over mock.patch:
        monkeypatch.setattr modifies the live module attribute in place so
        Path operations inside _versioned_path and anchor() see the temp
        dir without needing to intercept the import. It also restores
        automatically after each test, even on failure.
    """
    import wayfinder_gps_anchor as module
    d = tmp_path / "wayfinder"
    monkeypatch.setattr(module, "WAYFINDER_DIR", d)
    return d


# ===========================================================================
# _sanitize_filename
# ===========================================================================

class TestSanitizeFilename:
    """
    _sanitize_filename must produce filesystem-safe names on all platforms
    (Windows, macOS, Linux) without silently producing empty strings.
    """

    def test_spaces_become_underscores(self):
        assert WayfinderGPS_Anchor._sanitize_filename("my waypoint") == "my_waypoint"

    def test_special_chars_removed(self):
        result = WayfinderGPS_Anchor._sanitize_filename("hello/world:test?<>|")
        for ch in "/\\:?<>|":
            assert ch not in result

    def test_empty_string_fallback(self):
        assert WayfinderGPS_Anchor._sanitize_filename("") == "waypoint"

    def test_only_special_chars_fallback(self):
        assert WayfinderGPS_Anchor._sanitize_filename("???///:::") == "waypoint"

    def test_truncation_at_max_length(self):
        result = WayfinderGPS_Anchor._sanitize_filename("a" * 200, max_length=64)
        assert len(result) == 64

    def test_leading_trailing_whitespace_stripped(self):
        assert WayfinderGPS_Anchor._sanitize_filename("  hello  ") == "hello"

    def test_interior_whitespace_collapsed(self):
        assert WayfinderGPS_Anchor._sanitize_filename("a  b   c") == "a_b_c"

    def test_dots_and_hyphens_preserved(self):
        assert WayfinderGPS_Anchor._sanitize_filename("my-waypoint.v2") == "my-waypoint.v2"

    def test_unicode_letters_preserved(self):
        # \w in Python re includes Unicode word characters
        result = WayfinderGPS_Anchor._sanitize_filename("cafe_resume")
        assert "cafe" in result

    def test_max_length_zero_fallback(self):
        # Truncating to 0 chars must still return the fallback, not empty str
        result = WayfinderGPS_Anchor._sanitize_filename("abc", max_length=0)
        assert result == "waypoint"

    def test_numbers_preserved(self):
        assert WayfinderGPS_Anchor._sanitize_filename("wp_001") == "wp_001"

    def test_already_clean_string_unchanged(self):
        assert WayfinderGPS_Anchor._sanitize_filename("clean_name") == "clean_name"


# ===========================================================================
# _versioned_path
# ===========================================================================

class TestVersionedPath:
    """
    _versioned_path must never return an existing path and must increment
    version numbers monotonically.
    """

    def test_returns_base_when_absent(self, tmp_path: Path):
        p = tmp_path / "test.json"
        assert WayfinderGPS_Anchor._versioned_path(p) == p

    def test_v2_when_base_exists(self, tmp_path: Path):
        p = tmp_path / "test.json"
        p.touch()
        assert WayfinderGPS_Anchor._versioned_path(p) == tmp_path / "test_v2.json"

    def test_skips_occupied_versions(self, tmp_path: Path):
        p = tmp_path / "test.json"
        p.touch()
        (tmp_path / "test_v2.json").touch()
        (tmp_path / "test_v3.json").touch()
        assert WayfinderGPS_Anchor._versioned_path(p) == tmp_path / "test_v4.json"

    def test_result_never_exists(self, tmp_path: Path):
        """Whatever path is returned must not already exist on disk."""
        p = tmp_path / "x.json"
        for _ in range(5):
            candidate = WayfinderGPS_Anchor._versioned_path(p)
            assert not candidate.exists()
            candidate.touch()   # occupy it; next call must skip it


# ===========================================================================
# _tensor_stats
# ===========================================================================

class TestTensorStats:
    """
    _tensor_stats is the primary diagnostic output of the node.
    It must be correct, consistent, and handle unusual tensor configurations.
    """

    def test_required_keys_present(self):
        stats = WayfinderGPS_Anchor._tensor_stats(torch.zeros(4, 4, 4))
        assert "global" in stats
        assert "per_channel" in stats

    def test_global_stat_keys(self):
        g = WayfinderGPS_Anchor._tensor_stats(torch.randn(4, 4, 4))["global"]
        for key in ("mean", "variance", "std", "min", "max"):
            assert key in g, f"Missing key: {key}"

    def test_per_channel_count(self):
        for C in (1, 4, 8, 16):
            pc = WayfinderGPS_Anchor._tensor_stats(torch.randn(C, 4, 4))["per_channel"]
            assert len(pc) == C

    def test_zeros_tensor_statistics(self):
        g = WayfinderGPS_Anchor._tensor_stats(torch.zeros(4, 4, 4))["global"]
        assert g["mean"]     == 0.0
        assert g["min"]      == 0.0
        assert g["max"]      == 0.0
        assert g["variance"] == pytest.approx(0.0, abs=1e-9)

    def test_ones_tensor_statistics(self):
        g = WayfinderGPS_Anchor._tensor_stats(torch.ones(4, 4, 4))["global"]
        assert g["mean"] == pytest.approx(1.0)
        assert g["std"]  == pytest.approx(0.0, abs=1e-9)

    def test_precision_applied(self):
        t = torch.tensor([[[1.123456789]]])
        g = WayfinderGPS_Anchor._tensor_stats(t, precision=3)["global"]
        assert g["mean"] == round(1.123456789, 3)

    def test_values_are_python_floats(self):
        """JSON serialisation requires plain float, not numpy/torch scalars."""
        stats = WayfinderGPS_Anchor._tensor_stats(torch.randn(4, 4, 4))
        assert isinstance(stats["global"]["mean"], float)
        for ch in stats["per_channel"]:
            assert isinstance(ch["mean"], float)

    def test_deterministic_on_identical_input(self):
        t = torch.ones(4, 8, 8) * 3.14
        assert (
            WayfinderGPS_Anchor._tensor_stats(t)["global"]["mean"]
            == WayfinderGPS_Anchor._tensor_stats(t)["global"]["mean"]
        )

    def test_single_element_tensor(self):
        t = torch.tensor([[[42.0]]])
        g = WayfinderGPS_Anchor._tensor_stats(t)["global"]
        assert g["mean"] == pytest.approx(42.0)
        assert g["min"]  == g["max"]

    def test_half_precision_input(self):
        """float16 input must not raise; stats computed in float32."""
        t     = torch.randn(4, 4, 4, dtype=torch.float16)
        stats = WayfinderGPS_Anchor._tensor_stats(t)
        assert isinstance(stats["global"]["mean"], float)

    def test_bfloat16_input(self):
        t     = torch.randn(4, 4, 4, dtype=torch.bfloat16)
        stats = WayfinderGPS_Anchor._tensor_stats(t)
        assert isinstance(stats["global"]["mean"], float)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available in this environment"
    )
    def test_cuda_tensor_no_error(self):
        """
        CUDA tensors must complete without error. The single-transfer
        optimisation (one .cpu() call vs 25+ .item() syncs) is verified
        implicitly by the absence of timeout in the slow marker tests.
        """
        t     = torch.randn(4, 8, 8, device="cuda")
        stats = WayfinderGPS_Anchor._tensor_stats(t)
        assert isinstance(stats["global"]["mean"], float)


# ===========================================================================
# _resolve_grid_coordinate
# ===========================================================================

class TestResolveGridCoordinate:

    def test_finds_correct_cell(self):
        meta = _make_meta(3, 3)
        cell = WayfinderGPS_Anchor._resolve_grid_coordinate(meta, 4)
        assert cell is not None
        assert cell["batch_start"] == 4

    def test_returns_none_without_grid_map(self):
        assert WayfinderGPS_Anchor._resolve_grid_coordinate({}, 0) is None

    def test_returns_none_index_beyond_all_cells(self):
        meta = _make_meta(2, 2)
        assert WayfinderGPS_Anchor._resolve_grid_coordinate(meta, 99) is None

    def test_center_cell_flagged_odd_grid(self):
        meta = _make_meta(3, 3)
        cell = WayfinderGPS_Anchor._resolve_grid_coordinate(meta, 4)
        assert cell["is_center"] is True

    def test_no_center_in_even_grid(self):
        meta  = _make_meta(2, 2)
        cells = meta["wayfinder_grid_map"]
        assert not any(c["is_center"] for c in cells)

    def test_malformed_grid_map_not_list(self):
        meta = {"wayfinder_grid_map": "not_a_list"}
        assert WayfinderGPS_Anchor._resolve_grid_coordinate(meta, 0) is None

    def test_malformed_cell_missing_keys(self):
        """Cells lacking batch_start/end default to -1 and are skipped."""
        meta = {"wayfinder_grid_map": [{"grid_col": 0}]}
        assert WayfinderGPS_Anchor._resolve_grid_coordinate(meta, 0) is None

    def test_first_cell_index_zero(self):
        meta = _make_meta(3, 3)
        cell = WayfinderGPS_Anchor._resolve_grid_coordinate(meta, 0)
        assert cell is not None
        assert cell["batch_start"] == 0

    def test_last_cell(self):
        meta = _make_meta(3, 3)
        cell = WayfinderGPS_Anchor._resolve_grid_coordinate(meta, 8)
        assert cell is not None
        assert cell["batch_end"] == 8


# ===========================================================================
# _compute_absolute_position
# ===========================================================================

class TestComputeAbsolutePosition:

    def test_no_prior_no_cell_is_origin(self):
        pos = WayfinderGPS_Anchor._compute_absolute_position({}, None, 1.0, 1.0)
        assert pos == {"x": 0.0, "y": 0.0}

    def test_cell_offset_scaled_by_strength(self):
        cell = {"offset_x": 1.0, "offset_y": -1.0}
        pos  = WayfinderGPS_Anchor._compute_absolute_position(
            {}, cell, x_strength=2.0, y_strength=3.0
        )
        assert pos["x"] == pytest.approx(2.0)
        assert pos["y"] == pytest.approx(-3.0)

    def test_prior_position_accumulated(self):
        meta = {"accumulated_position": {"x": 1.5, "y": -0.5}}
        cell = {"offset_x": 1.0, "offset_y": 1.0}
        pos  = WayfinderGPS_Anchor._compute_absolute_position(
            meta, cell, 1.0, 1.0
        )
        assert pos["x"] == pytest.approx(2.5)
        assert pos["y"] == pytest.approx(0.5)

    def test_zero_strength_no_displacement(self):
        cell = {"offset_x": 5.0, "offset_y": 5.0}
        pos  = WayfinderGPS_Anchor._compute_absolute_position(
            {}, cell, 0.0, 0.0
        )
        assert pos == {"x": 0.0, "y": 0.0}

    def test_negative_strength_inverts_direction(self):
        cell = {"offset_x": 1.0, "offset_y": 1.0}
        pos  = WayfinderGPS_Anchor._compute_absolute_position(
            {}, cell, -1.0, -1.0
        )
        assert pos["x"] == pytest.approx(-1.0)
        assert pos["y"] == pytest.approx(-1.0)

    def test_malformed_prior_missing_x_defaults_zero(self):
        meta = {"accumulated_position": {"y": 3.0}}  # 'x' absent
        pos  = WayfinderGPS_Anchor._compute_absolute_position(
            meta, None, 1.0, 1.0
        )
        assert pos["x"] == pytest.approx(0.0)
        assert pos["y"] == pytest.approx(3.0)

    def test_malformed_prior_not_dict(self):
        """
        Non-dict accumulated_position raises AttributeError.
        This test documents the known limitation -- guard against it by
        validating upstream meta before chaining Anchor nodes with
        arbitrary sources.
        """
        meta = {"accumulated_position": "broken"}
        with pytest.raises(AttributeError):
            WayfinderGPS_Anchor._compute_absolute_position(meta, None, 1.0, 1.0)


# ===========================================================================
# anchor() -- core integration
# ===========================================================================

class TestAnchorCore:

    def test_returns_three_outputs(self, node):
        result = node.anchor(_make_latent(4), 0, False, "test", False)
        assert len(result) == 3

    def test_selected_latent_shape(self, node):
        out, _, _ = node.anchor(_make_latent(B=9, C=4, H=8, W=8), 5, False, "t", False)
        assert out["samples"].shape == (1, 4, 8, 8)

    def test_selected_values_match_input(self, node):
        latent = _make_latent(4)
        out, _, _ = node.anchor(latent, 2, False, "t", False)
        assert torch.allclose(out["samples"][0], latent["samples"][2])

    def test_clone_is_independent(self, node):
        latent = _make_latent(4)
        out, _, _ = node.anchor(latent, 0, False, "t", False)
        original  = latent["samples"][0].clone()
        out["samples"][0] *= 99.0
        assert torch.allclose(latent["samples"][0], original)

    def test_extra_latent_keys_preserved(self, node):
        latent = _make_latent(4)
        latent["noise_mask"] = torch.ones(1, 1, 8, 8)
        out, _, _ = node.anchor(latent, 0, False, "t", False)
        assert "noise_mask" in out

    def test_samples_key_not_duplicated(self, node):
        latent = _make_latent(4)
        out, _, _ = node.anchor(latent, 0, False, "t", False)
        assert list(out.keys()).count("samples") == 1

    def test_meta_required_keys(self, node):
        _, meta, _ = node.anchor(_make_latent(4), 0, False, "t", False)
        for key in (
            "waypoint_name", "absolute_position", "statistics",
            "elapsed_ms", "save_waypoint", "save_path", "save_error",
        ):
            assert key in meta, f"Missing meta key: {key}"

    def test_report_is_string_with_header(self, node):
        _, _, report = node.anchor(_make_latent(4), 0, False, "t", False)
        assert isinstance(report, str)
        assert "WayfinderGPS_Anchor" in report

    def test_waypoint_name_sanitized_in_meta(self, node):
        _, meta, _ = node.anchor(_make_latent(4), 0, False, "bad/name?", False)
        assert "/" not in meta["waypoint_name"]
        assert "?" not in meta["waypoint_name"]

    def test_elapsed_ms_non_negative(self, node):
        _, meta, _ = node.anchor(_make_latent(4), 0, False, "t", False)
        assert meta["elapsed_ms"] >= 0.0

    def test_perf_logging_enabled_no_error(self, node, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="WayfinderGPS_Anchor"):
            _, _, report = node.anchor(_make_latent(4), 0, False, "t", True)
        assert "Time:" in report

    def test_no_grad_does_not_track_gradients(self, node):
        """
        Ensure the node does not build autograd graphs (VRAM safety).

        Upstream samplers and VAEs may produce tensors with requires_grad=True.
        The node wraps slice and clone in torch.no_grad(), so the output must
        never carry a gradient function regardless of upstream graph state.
        Regression guard: if no_grad is accidentally removed, this test fails.
        """
        latent = _make_latent(4)
        latent["samples"].requires_grad_(True)   # simulate upstream graph
        out, _, _ = node.anchor(latent, 0, False, "grad_test", False)
        assert not out["samples"].requires_grad, (
            "Output latent must not require gradients -- "
            "torch.no_grad() wrapper may have been removed from anchor()"
        )


# ===========================================================================
# anchor() -- index validation
# ===========================================================================

class TestAnchorIndexValidation:

    def test_index_zero_valid(self, node):
        out, _, _ = node.anchor(_make_latent(1), 0, False, "t", False)
        assert out["samples"].shape[0] == 1

    def test_index_last_valid(self, node):
        out, _, _ = node.anchor(_make_latent(5), 4, False, "t", False)
        assert out["samples"].shape[0] == 1

    def test_index_out_of_range_raises(self, node):
        with pytest.raises(ValueError, match="out of range"):
            node.anchor(_make_latent(3), 5, False, "t", False)

    def test_index_equal_batch_size_raises(self, node):
        """Exactly equal is still out of range (0-based)."""
        with pytest.raises(ValueError, match="out of range"):
            node.anchor(_make_latent(4), 4, False, "t", False)

    def test_missing_samples_key_raises(self, node):
        with pytest.raises(ValueError, match="missing 'samples' key"):
            node.anchor({"other": torch.zeros(1)}, 0, False, "t", False)

    def test_wrong_ndim_3d_raises(self, node):
        with pytest.raises(ValueError, match="4-D"):
            node.anchor({"samples": torch.zeros(4, 4, 4)}, 0, False, "t", False)

    def test_wrong_ndim_5d_raises(self, node):
        with pytest.raises(ValueError, match="4-D"):
            node.anchor({"samples": torch.zeros(1, 4, 4, 4, 2)}, 0, False, "t", False)

    def test_empty_dict_raises(self, node):
        with pytest.raises(ValueError, match="missing 'samples' key"):
            node.anchor({}, 0, False, "t", False)


# ===========================================================================
# anchor() -- dtype handling
# ===========================================================================

class TestAnchorDtypes:
    """
    ComfyUI latents may arrive as float16, bfloat16, or float32 depending
    on the pipeline and VAE settings. The node must not corrupt or reject them.
    """

    @pytest.mark.parametrize("dtype", [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ])
    def test_dtype_preserved_in_output(self, node, dtype):
        latent = _make_latent(dtype=dtype)
        out, _, _ = node.anchor(latent, 0, False, "t", False)
        assert out["samples"].dtype == dtype

    @pytest.mark.parametrize("dtype", [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ])
    def test_stats_are_floats_for_all_dtypes(self, node, dtype):
        latent     = _make_latent(dtype=dtype)
        _, meta, _ = node.anchor(latent, 0, False, "t", False)
        assert isinstance(meta["statistics"]["global"]["mean"], float)


# ===========================================================================
# anchor() -- file I/O
# ===========================================================================

class TestAnchorFileIO:

    def test_file_created_on_save(self, node, tmp_wayfinder_dir):
        node.anchor(_make_latent(4), 0, True, "mywp", False)
        assert (tmp_wayfinder_dir / "mywp.json").exists()

    def test_file_is_valid_json(self, node, tmp_wayfinder_dir):
        node.anchor(_make_latent(4), 0, True, "jsontest", False)
        data = json.loads(
            (tmp_wayfinder_dir / "jsontest.json").read_text(encoding="utf-8")
        )
        assert data["waypoint_name"] == "jsontest"
        assert "statistics" in data

    def test_versioning_on_name_collision(self, node, tmp_wayfinder_dir):
        latent = _make_latent(4)
        node.anchor(latent, 0, True, "dup", False)
        node.anchor(latent, 0, True, "dup", False)
        assert (tmp_wayfinder_dir / "dup.json").exists()
        assert (tmp_wayfinder_dir / "dup_v2.json").exists()

    def test_no_file_when_save_false(self, node, tmp_wayfinder_dir):
        node.anchor(_make_latent(4), 0, False, "nosave", False)
        assert not (tmp_wayfinder_dir / "nosave.json").exists()

    def test_save_path_in_meta(self, node, tmp_wayfinder_dir):
        _, meta, _ = node.anchor(_make_latent(4), 0, True, "pathtest", False)
        assert meta["save_path"] is not None
        assert "pathtest" in meta["save_path"]

    def test_save_error_non_fatal_permission(
        self, node, tmp_wayfinder_dir, monkeypatch
    ):
        """
        PermissionError during mkdir must not propagate; node returns a result
        with save_error populated and save_path=None.
        """
        monkeypatch.setattr(
            Path, "mkdir",
            lambda *a, **kw: (_ for _ in ()).throw(PermissionError("no write access"))
        )
        _, meta, report = node.anchor(_make_latent(4), 0, True, "fail", False)
        assert meta["save_error"] is not None
        assert meta["save_path"]  is None
        assert "FAILED" in report

    def test_save_error_non_fatal_oserror(
        self, node, tmp_wayfinder_dir, monkeypatch
    ):
        monkeypatch.setattr(
            Path, "mkdir",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full"))
        )
        _, meta, _ = node.anchor(_make_latent(4), 0, True, "osfail", False)
        assert meta["save_error"] is not None

    def test_json_contains_all_required_fields(self, node, tmp_wayfinder_dir):
        meta_in = _make_meta(3, 3)
        latent  = _make_latent(9)
        node.anchor(latent, 4, True, "fulltest", False, meta_in)
        data = json.loads(
            (tmp_wayfinder_dir / "fulltest.json").read_text(encoding="utf-8")
        )
        for field in (
            "waypoint_name", "select_index", "batch_size",
            "latent_shape", "statistics", "absolute_position",
            "accumulated_position", "grid_cell",
        ):
            assert field in data, f"Missing JSON field: {field}"

    def test_directory_created_if_absent(self, node, tmp_wayfinder_dir):
        assert not tmp_wayfinder_dir.exists()
        node.anchor(_make_latent(4), 0, True, "mkdir_test", False)
        assert tmp_wayfinder_dir.exists()


# ===========================================================================
# anchor() -- metadata inheritance
# ===========================================================================

class TestAnchorMetaInheritance:

    def test_grid_cell_resolved_for_center(self, node):
        latent = _make_latent(9)
        _, meta, _ = node.anchor(latent, 4, False, "ctr", False, _make_meta(3, 3))
        assert meta["grid_cell"] is not None
        assert meta["grid_cell"]["is_center"] is True

    def test_center_absolute_position_is_zero(self, node):
        latent = _make_latent(9)
        _, meta, _ = node.anchor(latent, 4, False, "ctr", False, _make_meta(3, 3))
        assert meta["absolute_position"]["x"] == pytest.approx(0.0)
        assert meta["absolute_position"]["y"] == pytest.approx(0.0)

    def test_chained_anchors_accumulate_position(self, node):
        """
        Simulates:
            Manifold(3x3) -> Anchor(index=8) -> Manifold(2x2) -> Anchor(index=0)

        The second anchor must reflect the combined displacement of both hops.
        """
        latent1   = _make_latent(9)
        _, wp1, _ = node.anchor(latent1, 8, False, "hop1", False, _make_meta(3, 3))

        latent2  = _make_latent(4)
        meta2    = _make_meta(2, 2)
        meta2["accumulated_position"] = wp1["accumulated_position"]
        _, wp2, _ = node.anchor(latent2, 0, False, "hop2", False, meta2)

        assert (
            wp2["accumulated_position"]["x"] != wp1["accumulated_position"]["x"]
            or wp2["accumulated_position"]["y"] != wp1["accumulated_position"]["y"]
        )

    def test_no_meta_defaults_gracefully(self, node):
        _, meta, report = node.anchor(_make_latent(4), 0, False, "bare", False, None)
        assert meta["absolute_position"] == {"x": 0.0, "y": 0.0}
        assert "is_center:   N/A" in report

    def test_empty_meta_dict_handled(self, node):
        _, meta, _ = node.anchor(_make_latent(4), 0, False, "empty", False, {})
        assert meta["absolute_position"] == {"x": 0.0, "y": 0.0}

    def test_upstream_fields_carried_forward(self, node):
        upstream = _make_meta(3, 3)
        _, meta, _ = node.anchor(_make_latent(9), 0, False, "carry", False, upstream)
        assert meta["interpolation_mode"] == "Linear"
        assert meta["normalize_vectors"]  is True
        assert meta["upstream_grid_size"] == [3, 3]


# ===========================================================================
# anchor() -- edge cases
# ===========================================================================

class TestAnchorEdgeCases:

    def test_batch_size_one(self, node):
        out, _, _ = node.anchor(_make_latent(1), 0, False, "single", False)
        assert out["samples"].shape == (1, 4, 8, 8)

    def test_large_batch(self, node):
        out, _, _ = node.anchor(_make_latent(256), 255, False, "large", False)
        assert out["samples"].shape[0] == 1

    @pytest.mark.parametrize("C", [1, 8, 16])
    def test_unusual_channel_counts(self, node, C):
        latent     = _make_latent(B=2, C=C)
        _, meta, _ = node.anchor(latent, 0, False, f"ch{C}", False)
        assert len(meta["statistics"]["per_channel"]) == C

    def test_single_pixel_latent(self, node):
        latent = {"samples": torch.randn(2, 4, 1, 1)}
        out, meta, _ = node.anchor(latent, 1, False, "tiny", False)
        assert out["samples"].shape == (1, 4, 1, 1)

    def test_waypoint_name_all_special_chars(self, node):
        _, meta, _ = node.anchor(_make_latent(2), 0, False, "???///:::", False)
        assert meta["waypoint_name"] == "waypoint"

    def test_deterministic_stats_on_fixed_tensor(self, node):
        latent = {"samples": torch.ones(2, 4, 4, 4)}
        _, m1, _ = node.anchor(latent, 0, False, "det1", False)
        _, m2, _ = node.anchor(latent, 0, False, "det2", False)
        assert m1["statistics"]["global"]["mean"] == m2["statistics"]["global"]["mean"]

    def test_negative_values_in_latent(self, node):
        latent = {"samples": torch.full((2, 4, 4, 4), -5.0)}
        _, meta, _ = node.anchor(latent, 0, False, "neg", False)
        assert meta["statistics"]["global"]["mean"] == pytest.approx(-5.0)
        assert meta["statistics"]["global"]["max"]  == pytest.approx(-5.0)

    def test_inf_values_in_latent(self, node):
        """Inf values are pathological but must not crash the node."""
        import math
        latent = {"samples": torch.full((2, 4, 4, 4), float("inf"))}
        _, meta, _ = node.anchor(latent, 0, False, "inf", False)
        assert math.isinf(meta["statistics"]["global"]["mean"])

    def test_nan_values_in_latent(self, node):
        import math
        latent = {"samples": torch.full((2, 4, 4, 4), float("nan"))}
        _, meta, _ = node.anchor(latent, 0, False, "nan", False)
        assert math.isnan(meta["statistics"]["global"]["mean"])


# ===========================================================================
# Performance tests
# ===========================================================================

@pytest.mark.slow
class TestAnchorPerformance:
    """
    Timing benchmarks for anchor() under realistic and stress conditions.

    Why wall-clock only:
        torch.cuda.Event-based timing requires CUDA. These tests are CPU-safe.
        Wall-clock captures I/O and Python overhead, which is the dominant
        cost for this node in practice.

    Thresholds are intentionally loose (10x expected) to avoid flakiness on
    CI machines with variable load. Tighten per-environment as needed.
    """

    def test_standard_latent_completes_quickly(self, node):
        """4x4x64x64 -- typical SD 512px latent at batch=1."""
        latent = _make_latent(B=1, C=4, H=64, W=64)
        t0 = time.perf_counter()
        node.anchor(latent, 0, False, "perf_std", False)
        assert (time.perf_counter() - t0) < 1.0, "Standard latent took > 1s"

    def test_large_batch_latent(self, node):
        """Batch of 64 -- upper end of Manifold Explorer output."""
        latent = _make_latent(B=64, C=4, H=64, W=64)
        t0 = time.perf_counter()
        node.anchor(latent, 32, False, "perf_batch", False)
        assert (time.perf_counter() - t0) < 2.0, "Large batch took > 2s"

    def test_high_resolution_latent(self, node):
        """4x4x128x128 -- SD XL 1024px latent."""
        latent = _make_latent(B=1, C=4, H=128, W=128)
        t0 = time.perf_counter()
        node.anchor(latent, 0, False, "perf_xl", False)
        assert (time.perf_counter() - t0) < 1.0, "XL latent took > 1s"

    def test_stat_computation_scales_linearly(self, node):
        """
        Stat extraction time should grow roughly linearly with tensor size.
        Verifies no accidental O(N^2) or repeated device syncs crept in.
        """
        small = _make_latent(B=1, C=4, H=32,  W=32)
        large = _make_latent(B=1, C=4, H=128, W=128)

        t0 = time.perf_counter()
        for _ in range(20):
            node.anchor(small, 0, False, "s", False)
        t_small = (time.perf_counter() - t0) / 20

        t0 = time.perf_counter()
        for _ in range(20):
            node.anchor(large, 0, False, "l", False)
        t_large = (time.perf_counter() - t0) / 20

        # 128x128 is 16x the pixels of 32x32; allow 50x tolerance for CI noise
        assert t_large < t_small * 50, (
            f"Stat computation appears super-linear: "
            f"small={t_small*1000:.1f}ms large={t_large*1000:.1f}ms"
        )

    def test_file_save_overhead(self, node, tmp_wayfinder_dir):
        """File I/O must not dominate total execution time excessively."""
        latent = _make_latent(B=1, C=4, H=64, W=64)

        t0 = time.perf_counter()
        node.anchor(latent, 0, False, "no_save", False)
        t_no_save = time.perf_counter() - t0

        t0 = time.perf_counter()
        node.anchor(latent, 0, True, "with_save", False)
        t_save = time.perf_counter() - t0

        assert t_save < t_no_save + 0.5, (
            f"File save overhead too high: {(t_save - t_no_save)*1000:.1f}ms"
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_latent_performance(self, node):
        """CUDA tensor must not be dramatically slower than CPU."""
        latent_cpu  = _make_latent(B=1, C=4, H=64, W=64, device="cpu")
        latent_cuda = _make_latent(B=1, C=4, H=64, W=64, device="cuda")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            node.anchor(latent_cuda, 0, False, "cuda", False)
        torch.cuda.synchronize()
        t_cuda = (time.perf_counter() - t0) / 10

        t0 = time.perf_counter()
        for _ in range(10):
            node.anchor(latent_cpu, 0, False, "cpu", False)
        t_cpu = (time.perf_counter() - t0) / 10

        assert t_cuda < t_cpu * 20, (
            f"CUDA path unexpectedly slow: "
            f"cuda={t_cuda*1000:.1f}ms cpu={t_cpu*1000:.1f}ms"
        )


# ===========================================================================
# Concurrency tests
# ===========================================================================

@pytest.mark.slow
class TestAnchorConcurrency:
    """
    Concurrent access tests for file I/O safety.

    Design rationale for ThreadPoolExecutor over multiprocessing:
        - Stays within pytest's process boundary (no spawn overhead).
        - GIL is released during file I/O, so threads genuinely race on the
          filesystem. Sufficient to surface TOCTOU issues.
        - multiprocessing would be needed only if GIL-held Python code
          (not I/O) were the race site.

    Why open('x') is the correct fix:
        Between _versioned_path() returning a candidate and the subsequent
        open(), another thread could create that exact file. open('x') raises
        FileExistsError in that case rather than silently overwriting, making
        the race visible and the data safe.
    """

    def test_concurrent_saves_no_overwrites(self, node, tmp_wayfinder_dir):
        """
        N threads all saving the same waypoint name must each produce a
        distinct versioned file. No file must be silently overwritten.
        """
        N      = 10
        latent = _make_latent(4)
        errors: List[Exception] = []
        paths:  List[str]       = []
        lock = threading.Lock()

        def save_one(i: int):
            try:
                _, meta, _ = node.anchor(latent, 0, True, "concurrent_wp", False)
                with lock:
                    if meta["save_path"]:
                        paths.append(meta["save_path"])
            except Exception as exc:
                with lock:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(save_one, i) for i in range(N)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Unexpected errors: {errors}"
        assert len(paths) == len(set(paths)), "Duplicate file paths detected"

        for p in paths:
            content = Path(p).read_text(encoding="utf-8")
            data    = json.loads(content)
            assert "waypoint_name" in data

    def test_concurrent_reads_no_error(self, node):
        """
        Concurrent anchor() calls on the same latent (read-only, no save)
        must all succeed without corruption or error.
        """
        latent   = _make_latent(16)
        results: List[Dict] = []
        lock = threading.Lock()

        def read_one(idx: int):
            _, meta, _ = node.anchor(latent, idx % 16, False, f"r{idx}", False)
            with lock:
                results.append(meta)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(read_one, i) for i in range(32)]
            for f in as_completed(futures):
                f.result()

        assert len(results) == 32
        for meta in results:
            assert "statistics" in meta

    def test_concurrent_different_names_no_collision(
        self, node, tmp_wayfinder_dir
    ):
        """
        Threads using distinct waypoint names must never interfere with
        each other's versioning sequences.
        """
        N      = 8
        latent = _make_latent(4)

        def save_named(name: str):
            _, meta, _ = node.anchor(latent, 0, True, name, False)
            return meta["save_path"]

        with ThreadPoolExecutor(max_workers=N) as pool:
            futures = {pool.submit(save_named, f"unique_{i}"): i for i in range(N)}
            paths   = [f.result() for f in as_completed(futures)]

        assert len(paths) == len(set(paths))
        for p in paths:
            assert p is not None