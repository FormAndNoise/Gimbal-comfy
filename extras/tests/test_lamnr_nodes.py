"""
ComfyUI node-level tests for the LAMNr / Disentanglement nodes.

Covers `GimbalLatentStabilizer` (pipeline wrapper), `GimbalLatentMath`
(dispatcher), and `GimbalLatentTelemetry` (advanced OOD metrics), verifying:

  * INPUT_TYPES / FUNCTION / RETURN_TYPES contracts;
  * latent dict input/output shape + dtype preservation;
  * metric FLOAT means and full per-sample DICT telemetry;
  * dispatcher op routing, including error paths (missing target/comparison);
  * graceful degradation (TC/geodesic skipped on B == 1).
"""
import sys
import math
from pathlib import Path

import pytest
import torch

nodes_dir = Path(__file__).resolve().parent.parent.parent / "nodes"
sys.path.insert(0, str(nodes_dir))

from gimbal_latent_stabilizer import GimbalLatentStabilizer   # noqa: E402
from gimbal_latent_math_node import GimbalLatentMath         # noqa: E402
from gimbal_latent_telemetry import GimbalLatentTelemetry    # noqa: E402


def _latent(B=4, C=4, H=8, W=8, seed=None, dtype=torch.float32):
    if seed is not None:
        torch.manual_seed(seed)
    return {"samples": torch.randn(B, C, H, W, dtype=dtype)}


# ===========================================================================
class TestStabilizerNode:
    def test_contract(self):
        cls = GimbalLatentStabilizer
        inp = cls.INPUT_TYPES()
        assert "required" in inp and "latent" in inp["required"]
        assert cls.RETURN_TYPES == ("LATENT", "DICT")
        assert hasattr(cls(), "stabilize")

    def test_output_is_latent_and_telemetry(self):
        node = GimbalLatentStabilizer()
        z = _latent()
        out, tel = node.stabilize(z, truncation_psi=0.9, subspace_rank=-1,
                                  scale_cap=10.0)
        assert out["samples"].shape == z["samples"].shape
        assert torch.isfinite(out["samples"]).all()
        assert tel["instrument"] == "GimbalLatentStabilizer"

    def test_psi_one_rank_zero_collapses_to_mean(self):
        node = GimbalLatentStabilizer()
        z = _latent()
        out, _ = node.stabilize(z, truncation_psi=1.0, subspace_rank=0,
                                scale_cap=1e6)
        mu = z["samples"].float().mean(dim=0)
        assert torch.allclose(out["samples"].float(),
                              mu.unsqueeze(0).expand_as(z["samples"]), atol=1e-3)

    def test_dtype_preservation(self):
        node = GimbalLatentStabilizer()
        z = _latent(dtype=torch.float16)
        out, _ = node.stabilize(z, truncation_psi=0.8, subspace_rank=2,
                                scale_cap=10.0)
        assert out["samples"].dtype == torch.float16


# ===========================================================================
class TestDispatcherNode:
    def test_contract(self):
        cls = GimbalLatentMath
        inp = cls.INPUT_TYPES()
        ops = inp["required"]["op"][0]
        for op in ("channel_diagonal_gaussian", "truncation", "slerp_mu",
                   "bounded_scale", "dequantize", "woodbury_impute",
                   "pipeline", "log_likelihood", "mahalanobis",
                   "total_correlation", "geodesic"):
            assert op in ops
        assert cls.RETURN_TYPES == ("LATENT", "FLOAT", "DICT")

    def test_transform_ops_passthrough_shape(self):
        node = GimbalLatentMath()
        z = _latent()
        for op, kwargs in (
                ("channel_diagonal_gaussian", {}),
                ("truncation", {"psi": 0.7}),
                ("bounded_scale", {"scale_cap": 2.0}),
                ("dequantize", {"jitter_strength": 0.01}),
                ("woodbury_impute", {"subspace_rank": 2}),
                ("pipeline", {"psi": 0.9}),
        ):
            out, mean, tel = node.apply_op(z, op=op, **kwargs)
            assert out["samples"].shape == z["samples"].shape, op
            assert mean == 0.0, op  # transforms report 0.0 mean placeholder
            assert torch.isfinite(out["samples"]).all(), op

    def test_metric_ops_return_float_mean(self):
        node = GimbalLatentMath()
        z = _latent(seed=3)
        for op in ("log_likelihood", "mahalanobis", "total_correlation"):
            out, mean, tel = node.apply_op(z, op=op)
            assert math.isfinite(mean), op

    def test_slerp_endpoints(self):
        node = GimbalLatentMath()
        z = _latent(seed=5)
        tgt = _latent(seed=6)
        out0, _, _ = node.apply_op(z, op="slerp_mu",
                                   t=0.0, additional_latent=tgt)
        out1, _, _ = node.apply_op(z, op="slerp_mu",
                                   t=1.0, additional_latent=tgt)
        assert torch.allclose(out0["samples"], z["samples"], atol=1e-5)
        assert torch.allclose(out1["samples"], tgt["samples"], atol=1e-5)

    def test_slerp_requires_target(self):
        node = GimbalLatentMath()
        z = _latent()
        with pytest.raises(ValueError, match="additional_latent"):
            node.apply_op(z, op="slerp_mu")

    def test_geodesic_requires_target(self):
        node = GimbalLatentMath()
        z = _latent()
        with pytest.raises(ValueError, match="additional_latent"):
            node.apply_op(z, op="geodesic")

    def test_geodesic_self_is_zero(self):
        node = GimbalLatentMath()
        z = _latent(seed=7)
        _, mean, tel = node.apply_op(z, op="geodesic", additional_latent=z)
        assert mean < 1e-2

    def test_dequantize_seed_deterministic(self):
        node = GimbalLatentMath()
        z = _latent(seed=10)
        a, _, _ = node.apply_op(z, op="dequantize",
                                jitter_strength=0.05, seed=123)
        b, _, _ = node.apply_op(z, op="dequantize",
                                jitter_strength=0.05, seed=123)
        assert torch.allclose(a["samples"], b["samples"])


# ===========================================================================
class TestTelemetryNode:
    def test_contract(self):
        cls = GimbalLatentTelemetry
        inp = cls.INPUT_TYPES()
        assert "required" in inp and "latent" in inp["required"]
        assert cls.RETURN_TYPES == ("LATENT", "FLOAT", "FLOAT",
                                    "FLOAT", "FLOAT", "DICT")

    def test_passthrough_and_means(self):
        node = GimbalLatentTelemetry()
        z = _latent(seed=2)
        out, ll, mh, tc, geo, tel = node.diagnose(z)
        assert out is z
        for name, v in (("ll", ll), ("mh", mh), ("tc", tc), ("geo", geo)):
            assert math.isfinite(v), name

    def test_single_sample_degrades_gracefully(self):
        node = GimbalLatentTelemetry()
        z = _latent(B=1, seed=1)
        _, ll, mh, tc, geo, tel = node.diagnose(z)
        assert tc == 0.0 and geo == 0.0
        assert tel["per_sample_total_correlation"] == "skipped (B < 2)"
        assert math.isfinite(ll) and math.isfinite(mh)

    def test_geodesic_with_comparison(self):
        node = GimbalLatentTelemetry()
        z = _latent(seed=8)
        tgt = _latent(seed=9)
        _, _, _, _, geo, tel = node.diagnose(z, comparison_latent=tgt)
        assert geo > 0.0
        assert tel["geodesic_target"] == "comparison_latent"

    def test_geodesic_fallback_to_centroid(self):
        node = GimbalLatentTelemetry()
        z = _latent(B=4, seed=8)
        _, _, _, _, geo, tel = node.diagnose(z)
        assert math.isfinite(geo)
        assert tel["geodesic_target"] == "batch centroid"

    def test_ood_lower_likelihood(self):
        # When stats come from the batch, a single perturbed sample OODs.
        node = GimbalLatentTelemetry()
        torch.manual_seed(12)
        batch = torch.randn(4, 4, 8, 8)
        batch_bad = batch.clone()
        batch_bad[0] += 8.0
        _, _, _, _, _, tel_ok = node.diagnose({"samples": batch})
        _, _, _, _, _, tel_bad = node.diagnose({"samples": batch_bad})
        ll_ok0 = tel_ok["per_sample_log_likelihood"][0]
        ll_bad0 = tel_bad["per_sample_log_likelihood"][0]
        assert ll_bad0 < ll_ok0
