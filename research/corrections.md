# Wayfinder Node Suite Analysis

## Bugs and Issues Found

### Bug 1: WayfinderCompass_Pro.py - Batch Expansion Bypasses `allow_batch_expand` Check

**Location:** Lines ~170-180 in the `navigate` method

**Issue:** The base latent is expanded even when `allow_batch_expand=False`. The expansion line is outside the conditional block.

```python
# BUGGY CODE:
if base.shape[0] == 1 and max_batch > 1:
    if allow_batch_expand:
        warnings.warn(...)  # Warning only shown when True
    base = base.expand(max_batch, *base.shape[1:])  # ALWAYS executes!
```

**Fix:**
```python
# FIXED CODE:
if base.shape[0] == 1 and max_batch > 1:
    if not allow_batch_expand:
        raise ValueError(
            f"WayfinderCompass_Pro: base batch size is 1 but computed delta "
            f"has batch size {max_batch}. Enable 'allow_batch_expand' to permit "
            f"automatic batch expansion."
        )
    warnings.warn(
        f"WayfinderCompass_Pro: expanding base batch 1 -> {max_batch}. "
        f"Memory usage scales linearly with batch size.",
        UserWarning,
        stacklevel=2,
    )
    base = base.expand(max_batch, *base.shape[1:])
```

---

### Bug 2: wayfinder_gps_anchor.py - Non-Dict `accumulated_position` Crashes

**Location:** `_compute_absolute_position` method

**Issue:** If `meta['accumulated_position']` exists but is not a dict (e.g., `None`, string, list), calling `.get()` raises `AttributeError`.

```python
# BUGGY CODE:
prior = meta.get("accumulated_position", {"x": 0.0, "y": 0.0})
# If accumulated_position = None, prior = None
# Then prior.get("x", 0.0) raises AttributeError
```

**Fix:**
```python
# FIXED CODE:
prior_raw = meta.get("accumulated_position", {"x": 0.0, "y": 0.0})
if not isinstance(prior_raw, dict):
    log.warning(
        "WayfinderGPS_Anchor: accumulated_position is not a dict (%s), "
        "defaulting to origin.",
        type(prior_raw).__name__,
    )
    prior = {"x": 0.0, "y": 0.0}
else:
    prior = prior_raw
```

---

### Bug 3: Wayfinder_SemanticSlider.py - `pc_index` UI Max is Static

**Location:** `INPUT_TYPES` classmethod

**Issue:** The slider max is hardcoded to `_DEFAULT_N_COMPONENTS` (10), but actual computable components = `min(10, batch_size)`. Users see a slider to 10 even for batches of 4 samples.

**Fix (Partial - UI limitation):** This is a ComfyUI limitation (dynamic max not supported). The validation in `apply_slider` correctly catches this, but the error message could be more helpful:

```python
# IMPROVED ERROR MESSAGE:
if pc_idx_0 >= effective_n:
    raise ValueError(
        f"pc_index {pc_index} exceeds the number of computable components "
        f"({effective_n}) for a batch of size {B_batch}. "
        f"Each component requires one batch sample. "
        f"Add more samples to your latent_batch or reduce pc_index."
    )
```

---

### Minor Issue: WayfinderManifold_Explorer.py - Redundant Interpolation

**Location:** `_interpolate` method called with `t=1.0`

**Issue:** `lerp(a, b, 1.0)` and `slerp(a, b, 1.0)` both return `b` exactly, making the interpolation path unnecessary overhead. Not a bug, but wasteful.

**Potential Fix:** If interpolation modes other than `t=1.0` are never needed, simplify to direct addition. If future `t` values are planned, leave as-is for extensibility.

---

# Corrected Source Files

## WayfinderCompass_Pro.py (Corrected)

```python
import torch
import torch.nn.functional as F
import logging
import time
import warnings
from typing import Optional, Tuple, Dict, Any

log = logging.getLogger("WayfinderCompass_Pro")


class WayfinderCompass_Pro:
    """
    Latent space vector arithmetic node for directional style/concept navigation.
    Supports Standard, Normalized, and Orthogonal Projection modes with
    optional mask localization, auto-resizing, and batch broadcasting.

    Orthogonal mode: projection computed over flattened (C*H*W) per batch item
    by default, or per-channel (H*W) when ortho_per_channel=True.

    Mask values expected in [0, 1] -- enable clamp_mask_input if sourcing
    from raw tensors outside ComfyUI's native MASK pipeline.
    """

    CATEGORY = "latent/advanced"
    RETURN_TYPES = ("LATENT", "DICT")
    RETURN_NAMES = ("latent_out", "wayfinder_meta")
    FUNCTION = "navigate"

    # Clamp bounds exposed as class-level constants for easy subclass override
    CLAMP_MIN_DEFAULT = -10.0
    CLAMP_MAX_DEFAULT =  10.0

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "base_latent":        ("LATENT",),
                "target_latent":      ("LATENT",),
                "origin_latent":      ("LATENT",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -20.0,
                        "max":  20.0,
                        "step":  0.01,
                        "display": "slider",
                    },
                ),
                "mode": (
                    ["Standard", "Normalized", "Orthogonal_Projection"],
                    {"default": "Standard"},
                ),
                "clamp_output":        ("BOOLEAN", {"default": False}),
                "clamp_min": (
                    "FLOAT",
                    {"default": cls.CLAMP_MIN_DEFAULT, "min": -100.0, "max": 0.0,  "step": 0.5},
                ),
                "clamp_max": (
                    "FLOAT",
                    {"default": cls.CLAMP_MAX_DEFAULT, "min":    0.0, "max": 100.0, "step": 0.5},
                ),
                "allow_batch_expand":  ("BOOLEAN", {"default": False}),
                "ortho_per_channel":   ("BOOLEAN", {"default": False}),
                "clamp_mask_input":    ("BOOLEAN", {"default": False}),
                "enable_perf_logging": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_device(tensor: torch.Tensor, expected_device: torch.device, name: str) -> None:
        """
        Raise a clear RuntimeError if `tensor` is not on `expected_device`.
        Called after explicit .to(device) transfers so a mismatch here
        indicates a genuine inconsistency rather than a missing transfer.
        """
        if tensor.device != expected_device:
            raise RuntimeError(
                f"WayfinderCompass_Pro: device inconsistency on '{name}'. "
                f"Expected {expected_device}, found {tensor.device}. "
                f"This should not occur after the automatic transfer step -- "
                f"check that your tensor was not re-allocated on a different "
                f"device between the transfer and this check."
            )

    # ------------------------------------------------------------------
    # Performance logging context
    # ------------------------------------------------------------------

    class _Timer:
        """Lightweight wall-clock timer used for optional perf logging."""
        def __init__(self, label: str, enabled: bool):
            self.label   = label
            self.enabled = enabled
            self._start  = None

        def __enter__(self):
            if self.enabled:
                self._start = time.perf_counter()
            return self

        def __exit__(self, *_):
            if self.enabled and self._start is not None:
                elapsed = (time.perf_counter() - self._start) * 1000
                log.info(f"[WayfinderCompass_Pro] {self.label}: {elapsed:.2f} ms")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_samples(latent_dict: Dict[str, Any], name: str = "latent") -> torch.Tensor:
        samples = latent_dict.get("samples")
        if samples is None:
            raise ValueError(
                f"WayfinderCompass_Pro: '{name}' dict is missing the 'samples' key. "
                f"Ensure the connected node outputs a valid LATENT with a 'samples' tensor."
            )
        if samples.ndim != 4:
            raise ValueError(
                f"WayfinderCompass_Pro: '{name}.samples' must be 4-D [B, C, H, W], "
                f"got shape {list(samples.shape)}."
            )
        return samples

    @staticmethod
    def _resize_to_base(
        tensor: torch.Tensor,
        base: torch.Tensor,
        name: str,
        perf: bool,
    ) -> torch.Tensor:
        if tensor.shape[-2:] == base.shape[-2:]:
            return tensor
        src_hw  = list(tensor.shape[-2:])
        tgt_hw  = list(base.shape[-2:])
        ratio   = max(src_hw[0] / tgt_hw[0], src_hw[1] / tgt_hw[1])
        if ratio > 4.0 or ratio < 0.25:
            warnings.warn(
                f"WayfinderCompass_Pro: large spatial resize on '{name}' "
                f"({src_hw} -> {tgt_hw}, ratio {ratio:.2f}x). "
                f"Verify that inputs share a compatible latent resolution.",
                UserWarning,
                stacklevel=4,
            )
        t0 = time.perf_counter() if perf else None
        out = F.interpolate(
            tensor.float(),
            size=base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(tensor.dtype)
        if perf:
            elapsed = (time.perf_counter() - t0) * 1000
            log.info(
                f"[WayfinderCompass_Pro] interpolate '{name}' "
                f"{src_hw} -> {tgt_hw}: {elapsed:.2f} ms"
            )
        return out

    @staticmethod
    def _resize_mask(
        mask: torch.Tensor,
        base: torch.Tensor,
        perf: bool,
    ) -> torch.Tensor:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)         # (H, W) -> (1, H, W)
        mask = mask.unsqueeze(1).float()     # (B, H, W) -> (B, 1, H, W)
        if mask.shape[-2:] != base.shape[-2:]:
            t0 = time.perf_counter() if perf else None
            mask = F.interpolate(
                mask,
                size=base.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if perf:
                elapsed = (time.perf_counter() - t0) * 1000
                log.info(f"[WayfinderCompass_Pro] interpolate mask: {elapsed:.2f} ms")
        return mask  # (B, 1, H, W)

    @staticmethod
    def _safe_broadcast(
        base: torch.Tensor,
        other: torch.Tensor,
        name: str,
        allow_expand: bool,
    ) -> torch.Tensor:
        B_b = base.shape[0]
        B_o = other.shape[0]

        if B_b == B_o:
            return other

        # Trivially safe: one side is already size-1
        if B_o == 1:
            return other.expand(B_b, *other.shape[1:])
        if B_b == 1:
            return other  # base will be expanded by the caller

        if not allow_expand:
            raise ValueError(
                f"WayfinderCompass_Pro: batch size mismatch detected between "
                f"base (batch size: {B_b}) and '{name}' (batch size: {B_o}). "
                f"Enable 'allow_batch_expand' to let the node automatically "
                f"expand the smaller batch to match the larger one, or verify "
                f"that all inputs are intended to share the same batch size. "
                f"Note: batch expansion increases memory usage -- review your "
                f"pipeline if this was unexpected."
            )

        # allow_expand=True path
        if B_o > B_b:
            # Caller expands base; return other untouched
            return other

        # B_o < B_b
        if B_b % B_o != 0:
            raise ValueError(
                f"WayfinderCompass_Pro: cannot evenly expand '{name}' "
                f"(batch {B_o}) to match base (batch {B_b}) -- "
                f"{B_b} is not divisible by {B_o}. Adjust your batch sizes."
            )
        warnings.warn(
            f"WayfinderCompass_Pro: expanding '{name}' batch {B_o} -> {B_b} "
            f"via .repeat(). This increases memory usage proportionally. "
            f"If this was unintentional, check upstream batch configuration.",
            UserWarning,
            stacklevel=4,
        )
        repeat = [B_b // B_o] + [1] * (other.ndim - 1)
        return other.repeat(*repeat)

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_standard(
        base: torch.Tensor,
        delta: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        return base + delta * strength

    @staticmethod
    def _apply_normalized(
        base: torch.Tensor,
        delta: torch.Tensor,
        strength: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        flat  = delta.reshape(delta.shape[0], -1)
        norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
        scale = norms.reshape(delta.shape[0], *([1] * (delta.ndim - 1)))
        return base + (delta / scale) * strength

    @staticmethod
    def _apply_orthogonal(
        base: torch.Tensor,
        delta: torch.Tensor,
        strength: float,
        per_channel: bool,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        B = base.shape[0]
        if not per_channel:
            base_flat  = base.reshape(B, -1).float()
            delta_flat = delta.reshape(B, -1).float()
            norms      = delta_flat.norm(dim=1, keepdim=True).clamp(min=eps)
            delta_hat  = delta_flat / norms
            dot        = (base_flat * delta_hat).sum(dim=1, keepdim=True)
            projection = (dot * delta_hat).reshape(base.shape).to(base.dtype)
        else:
            C          = base.shape[1]
            base_flat  = base.reshape(B, C, -1).float()
            delta_flat = delta.reshape(B, C, -1).float()
            norms      = delta_flat.norm(dim=2, keepdim=True).clamp(min=eps)
            delta_hat  = delta_flat / norms
            dot        = (base_flat * delta_hat).sum(dim=2, keepdim=True)
            projection = (dot * delta_hat).reshape(base.shape).to(base.dtype)
        return base + projection * strength

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def navigate(
        self,
        base_latent:        Dict[str, Any],
        target_latent:      Dict[str, Any],
        origin_latent:      Dict[str, Any],
        strength:           float,
        mode:               str,
        clamp_output:       bool,
        clamp_min:          float,
        clamp_max:          float,
        allow_batch_expand: bool,
        ortho_per_channel:  bool,
        clamp_mask_input:   bool,
        enable_perf_logging: bool,
        mask:               Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        if enable_perf_logging:
            logging.basicConfig(level=logging.INFO)

        perf = enable_perf_logging
        T    = self._Timer

        # Validate clamp bounds
        if clamp_output and clamp_min >= clamp_max:
            raise ValueError(
                f"WayfinderCompass_Pro: clamp_min ({clamp_min}) must be "
                f"strictly less than clamp_max ({clamp_max})."
            )

        # 1. Unpack
        with T("unpack latents", perf):
            base   = self._extract_samples(base_latent,   "base_latent")
            target = self._extract_samples(target_latent, "target_latent")
            origin = self._extract_samples(origin_latent, "origin_latent")

        # 2. Device transfer
        device = base.device
        with T("device transfer", perf):
            target = target.to(device)
            origin = origin.to(device)

        # 3. Post-transfer device assertion
        self._check_device(target, device, "target_latent")
        self._check_device(origin, device, "origin_latent")

        # 4. Spatial resize
        with T("spatial resize target/origin", perf):
            target = self._resize_to_base(target, base, "target_latent", perf)
            origin = self._resize_to_base(origin, base, "origin_latent", perf)

        # 5. Batch resolution
        with T("batch broadcast", perf):
            target = self._safe_broadcast(base, target, "target_latent", allow_batch_expand)
            origin = self._safe_broadcast(base, origin, "origin_latent", allow_batch_expand)

            max_batch = max(base.shape[0], target.shape[0], origin.shape[0])
            
            # BUG FIX: Only expand base if allow_batch_expand is True
            if base.shape[0] == 1 and max_batch > 1:
                if not allow_batch_expand:
                    raise ValueError(
                        f"WayfinderCompass_Pro: base batch size is 1 but target/origin "
                        f"have batch size {max_batch}. Enable 'allow_batch_expand' to "
                        f"permit automatic batch expansion, or ensure all inputs have "
                        f"matching batch sizes."
                    )
                warnings.warn(
                    f"WayfinderCompass_Pro: expanding base batch 1 -> {max_batch}. "
                    f"Memory usage scales linearly with batch size.",
                    UserWarning,
                    stacklevel=2,
                )
                base = base.expand(max_batch, *base.shape[1:])

        # 6. Delta
        delta = target - origin

        # 7. Mask
        mask_applied = mask is not None
        if mask_applied:
            with T("mask processing", perf):
                mask_t = mask.to(device)
                self._check_device(mask_t, device, "mask")
                if clamp_mask_input:
                    mask_t = mask_t.clamp(0.0, 1.0)
                mask_t = self._resize_mask(mask_t, base, perf)
                if mask_t.shape[0] == 1 and base.shape[0] > 1:
                    mask_t = mask_t.expand(base.shape[0], *mask_t.shape[1:])
                delta = delta * mask_t

        # 8. Mode dispatch
        with T(f"mode={mode}", perf):
            if mode == "Standard":
                result = self._apply_standard(base, delta, strength)
            elif mode == "Normalized":
                result = self._apply_normalized(base, delta, strength)
            elif mode == "Orthogonal_Projection":
                result = self._apply_orthogonal(base, delta, strength, ortho_per_channel)
            else:
                raise ValueError(
                    f"WayfinderCompass_Pro: unknown mode '{mode}'. "
                    f"Valid options: 'Standard', 'Normalized', 'Orthogonal_Projection'."
                )

        # 9. Clamp
        if clamp_output:
            result = result.clamp(clamp_min, clamp_max)

        # 10. Preserve dtype
        result = result.to(base.dtype)

        # 11. Clean LATENT output -- no injected keys
        out_latent = {k: v for k, v in base_latent.items() if k != "samples"}
        out_latent["samples"] = result

        # 12. Separate DICT metadata output
        meta: Dict[str, Any] = {
            "mode":               mode,
            "strength":           strength,
            "clamp_output":       clamp_output,
            "clamp_range":        [clamp_min, clamp_max] if clamp_output else None,
            "mask_applied":       mask_applied,
            "mask_clamped":       clamp_mask_input if mask_applied else None,
            "allow_batch_expand": allow_batch_expand,
            "ortho_per_channel":  ortho_per_channel if mode == "Orthogonal_Projection" else None,
            "base_shape":         list(base.shape),
            "target_shape":       list(target.shape),
            "origin_shape":       list(origin.shape),
            "result_shape":       list(result.shape),
            "device":             str(device),
            "perf_logging":       perf,
        }

        return (out_latent, meta)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "WayfinderCompass_Pro": WayfinderCompass_Pro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WayfinderCompass_Pro": "Wayfinder Compass Pro (Latent Nav)",
}
```

---

## wayfinder_gps_anchor.py (Corrected)

```python
"""
WayfinderGPS_Anchor
===================
ComfyUI custom node. Extracts a single latent from a batch by index,
computes navigational metadata, and optionally persists a waypoint to disk.

Environment variables
---------------------
WAYFINDER_OUTPUT_DIR : str
    Absolute path to override the default output directory.
    Defaults to <ComfyUI output dir>/wayfinder or ./output/wayfinder.

WAYFINDER_MAX_FILENAME_LENGTH : int
    Maximum character length for sanitized waypoint filenames.
    Defaults to 64.

WAYFINDER_STAT_PRECISION : int
    Decimal places used when rounding tensor statistics.
    Defaults to 6.
"""

import json
import logging
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

log = logging.getLogger("WayfinderGPS_Anchor")

# ---------------------------------------------------------------------------
# Environment-driven configuration
# ---------------------------------------------------------------------------

def _resolve_output_dir() -> Path:
    """
    Resolve the wayfinder output directory using this priority order:

    1. ``WAYFINDER_OUTPUT_DIR`` environment variable (absolute path).
    2. ComfyUI's ``folder_paths.get_output_directory()`` + 'wayfinder'.
    3. Fallback: ``./output/wayfinder`` relative to the working directory.

    Returns
    -------
    Path
        Resolved (not yet created) output directory path.
    """
    env_override = os.environ.get("WAYFINDER_OUTPUT_DIR", "").strip()
    if env_override:
        p = Path(env_override)
        log.debug(f"WayfinderGPS_Anchor: using WAYFINDER_OUTPUT_DIR={p}")
        return p

    try:
        import folder_paths  # type: ignore
        root = Path(folder_paths.get_output_directory())
        log.debug(f"WayfinderGPS_Anchor: using ComfyUI output root={root}")
        return root / "wayfinder"
    except Exception as exc:
        log.error(
            f"WayfinderGPS_Anchor: folder_paths unavailable, "
            f"falling back to ./output/wayfinder. Error: {exc}"
        )
        return Path("output") / "wayfinder"


def _get_int_env(key: str, default: int, minimum: int = 1) -> int:
    """
    Read an integer from an environment variable with a validated default.

    Parameters
    ----------
    key : str
        Environment variable name.
    default : int
        Value to use when the variable is absent or unparseable.
    minimum : int
        Floor value; parsed result is clamped to max(minimum, parsed).

    Returns
    -------
    int
        Resolved integer value.
    """
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        log.warning(
            f"WayfinderGPS_Anchor: {key}={raw!r} is not a valid integer, "
            f"using default={default}."
        )
        return default


WAYFINDER_DIR       = _resolve_output_dir()
MAX_FILENAME_LENGTH = _get_int_env("WAYFINDER_MAX_FILENAME_LENGTH", 64)
STAT_PRECISION      = _get_int_env("WAYFINDER_STAT_PRECISION", 6)

# Convenience alias for unit testing without instantiating the node
_sanitize_filename_standalone = None  # assigned after class definition


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class WayfinderGPS_Anchor:
    """
    ComfyUI node: extract and anchor a single latent from a batch.

    Inputs
    ------
    latent_batch : LATENT
        Any latent batch. Typically the output of WayfinderManifold_Explorer.
    select_index : int
        Zero-based index of the latent to extract. Validated against batch
        size before any tensor work is performed.
    save_waypoint : bool
        When True, serialise the waypoint payload to a JSON file under
        ``WAYFINDER_DIR``. File I/O failures are non-fatal; a warning is
        emitted and execution continues.
    waypoint_name : str
        Human-readable label. Sanitized to a safe filename before use.
        The sanitized form is used in both the payload and file path to
        ensure consistency.
    enable_perf_logging : bool
        When True, emit INFO-level timing logs to the module logger.
    wayfinder_meta : dict, optional
        Metadata dict from an upstream Wayfinder node. Used to inherit
        grid coordinates and accumulated navigational position.

    Outputs
    -------
    anchored_latent : LATENT
        Single-item latent dict containing the selected sample.
    waypoint_meta : DICT
        Full waypoint payload plus I/O status and timing.
    waypoint_report : STRING
        Human-readable summary suitable for a ShowText node.

    Notes
    -----
    - Tensor statistics are computed on a CPU copy to avoid one CUDA device
      sync per ``.item()`` call (25+ syncs for a 4-channel latent otherwise).
    - ``torch.no_grad()`` wraps slice and clone. Stat extraction runs outside
      since it has no gradient path through the CPU copy.
    - File writes use ``open('x')`` (exclusive create) to eliminate the TOCTOU
      race between existence check and write. Two concurrent processes versioning
      to the same name will have one raise FileExistsError rather than
      silently overwriting.
    - ``safe_name`` is derived before payload assembly so every reference --
      payload key, file path, report -- is guaranteed consistent.
    """

    CATEGORY     = "latent/advanced"
    RETURN_TYPES = ("LATENT", "DICT", "STRING")
    RETURN_NAMES = ("anchored_latent", "waypoint_meta", "waypoint_report")
    FUNCTION     = "anchor"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent_batch": ("LATENT",),
                "select_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4095, "step": 1},
                ),
                "save_waypoint":       ("BOOLEAN", {"default": False}),
                "waypoint_name":       ("STRING",  {"default": "waypoint_01"}),
                "enable_perf_logging": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "wayfinder_meta": ("DICT",),
            },
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_samples(d: Dict[str, Any], name: str) -> torch.Tensor:
        """
        Pull the samples tensor from a ComfyUI LATENT dict.

        Parameters
        ----------
        d : dict
            LATENT dictionary as produced by ComfyUI nodes.
        name : str
            Human-readable input name used in error messages.

        Returns
        -------
        torch.Tensor
            4-D tensor of shape [B, C, H, W].

        Raises
        ------
        ValueError
            If 'samples' key is absent or tensor is not 4-D.
        """
        s = d.get("samples")
        if s is None:
            raise ValueError(
                f"WayfinderGPS_Anchor: '{name}' is missing the 'samples' key. "
                f"Ensure the connected node outputs a valid LATENT."
            )
        if s.ndim != 4:
            raise ValueError(
                f"WayfinderGPS_Anchor: '{name}.samples' must be 4-D [B, C, H, W], "
                f"got shape {list(s.shape)}."
            )
        return s

    @staticmethod
    def _sanitize_filename(name: str, max_length: int = MAX_FILENAME_LENGTH) -> str:
        """
        Produce a filesystem-safe filename stem from arbitrary user input.

        Steps applied in order:
        1. Strip leading/trailing whitespace.
        2. Collapse interior whitespace runs to single underscores.
        3. Remove all characters outside ``[A-Za-z0-9._-]``.
        4. Truncate to ``max_length`` characters.
        5. Fall back to ``'waypoint'`` if the result is empty.

        Parameters
        ----------
        name : str
            Raw user-supplied waypoint name.
        max_length : int
            Maximum length of the returned string. Controlled by
            ``WAYFINDER_MAX_FILENAME_LENGTH`` environment variable.

        Returns
        -------
        str
            Safe filename stem, never empty.
        """
        name = name.strip()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^\w.\-]", "", name)
        name = name[:max_length]
        return name or "waypoint"

    @staticmethod
    def _versioned_path(base: Path) -> Path:
        """
        Return a non-existent path derived from ``base``.

        If ``base`` does not exist it is returned unchanged. Otherwise,
        ``_v2``, ``_v3``, ... suffixes are tried until a free name is found.

        Why this is not fully race-safe alone:
            Between this function returning and the caller opening the file,
            another process could create the candidate path. The caller must
            use ``open('x')`` (exclusive create mode) to atomically claim the
            path and raise ``FileExistsError`` if the race is lost rather than
            silently overwriting existing data.

        Parameters
        ----------
        base : Path
            Desired output path (may or may not exist).

        Returns
        -------
        Path
            Non-existent candidate path at the time of the call.
        """
        if not base.exists():
            return base
        stem   = base.stem
        suffix = base.suffix
        parent = base.parent
        v = 2
        while True:
            candidate = parent / f"{stem}_v{v}{suffix}"
            if not candidate.exists():
                return candidate
            v += 1

    # ------------------------------------------------------------------
    # Tensor statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_stats(
        t: torch.Tensor,
        precision: int = STAT_PRECISION,
    ) -> Dict[str, Any]:
        """
        Compute global and per-channel statistics for a single latent tensor.

        Why CPU transfer first:
            On CUDA, each ``.item()`` call triggers a blocking
            ``cudaDeviceSynchronize``. For a 4-channel SD latent with 5 stats
            per channel plus 5 global stats, that is 25 device syncs. A single
            ``.cpu()`` transfer replaces all of them with one PCIe transaction,
            which is substantially faster at any tensor size above ~1KB.

        Parameters
        ----------
        t : torch.Tensor
            Single latent of shape [C, H, W]. No batch dimension.
        precision : int
            Decimal places for rounding. Controlled by ``WAYFINDER_STAT_PRECISION``
            environment variable.

        Returns
        -------
        dict
            Keys: ``'global'`` (dict) and ``'per_channel'`` (list of dicts).
            All numeric values are plain Python floats for JSON serialisability.
        """
        t_cpu = t.float().cpu()
        C     = t_cpu.shape[0]

        per_channel: List[Dict[str, Any]] = []
        for c in range(C):
            ch = t_cpu[c]
            per_channel.append({
                "channel":  c,
                "mean":     round(ch.mean().item(),  precision),
                "variance": round(ch.var().item(),   precision),
                "std":      round(ch.std().item(),   precision),
                "min":      round(ch.min().item(),   precision),
                "max":      round(ch.max().item(),   precision),
            })

        return {
            "global": {
                "mean":     round(t_cpu.mean().item(),  precision),
                "variance": round(t_cpu.var().item(),   precision),
                "std":      round(t_cpu.std().item(),   precision),
                "min":      round(t_cpu.min().item(),   precision),
                "max":      round(t_cpu.max().item(),   precision),
            },
            "per_channel": per_channel,
        }

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_grid_coordinate(
        meta:         Dict[str, Any],
        select_index: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Locate the grid cell in ``wayfinder_grid_map`` that contains
        ``select_index``.

        Searches by ``batch_start`` / ``batch_end`` inclusive range stored
        in each cell entry. Returns ``None`` when no grid map is present or
        no cell covers the requested index (e.g. the upstream node was not a
        Manifold Explorer).

        Parameters
        ----------
        meta : dict
            Upstream wayfinder metadata dict.
        select_index : int
            Batch index of the selected latent.

        Returns
        -------
        dict or None
            Matching grid cell dict, or ``None`` if not found.
        """
        grid_map = meta.get("wayfinder_grid_map")
        if not grid_map or not isinstance(grid_map, list):
            return None
        for cell in grid_map:
            start = cell.get("batch_start", -1)
            end   = cell.get("batch_end",   -1)
            if start <= select_index <= end:
                return cell
        return None

    @staticmethod
    def _compute_absolute_position(
        meta:       Dict[str, Any],
        grid_cell:  Optional[Dict[str, Any]],
        x_strength: float,
        y_strength: float,
    ) -> Dict[str, Any]:
        """
        Compute the cumulative navigational position of the selected latent.

        Adds the cell's scaled displacement (``offset * strength``) to any
        prior accumulated position found in ``meta['accumulated_position']``.
        Chaining Anchor nodes therefore tracks total displacement across
        multiple Manifold Explorer hops.

        Parameters
        ----------
        meta : dict
            Upstream metadata. May contain ``'accumulated_position'`` from a
            previous Anchor node.
        grid_cell : dict or None
            Grid cell for the selected index, as returned by
            ``_resolve_grid_coordinate``.
        x_strength : float
            X-axis strength value from the upstream Manifold Explorer.
        y_strength : float
            Y-axis strength value from the upstream Manifold Explorer.

        Returns
        -------
        dict
            ``{'x': float, 'y': float}`` absolute position.
        """
        # BUG FIX: Handle non-dict accumulated_position gracefully
        prior_raw = meta.get("accumulated_position", {"x": 0.0, "y": 0.0})
        if not isinstance(prior_raw, dict):
            log.warning(
                "WayfinderGPS_Anchor: accumulated_position is not a dict "
                "(got %s), defaulting to origin {x: 0.0, y: 0.0}. "
                "Check upstream metadata for corruption.",
                type(prior_raw).__name__,
            )
            prior = {"x": 0.0, "y": 0.0}
        else:
            prior = prior_raw

        cell_x = grid_cell.get("offset_x", 0.0) * x_strength if grid_cell else 0.0
        cell_y = grid_cell.get("offset_y", 0.0) * y_strength if grid_cell else 0.0

        return {
            "x": round(prior.get("x", 0.0) + cell_x, 8),
            "y": round(prior.get("y", 0.0) + cell_y, 8),
        }

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_report(
        waypoint_name:     str,
        select_index:      int,
        batch_size:        int,
        latent_shape:      List[int],
        stats:             Dict[str, Any],
        grid_cell:         Optional[Dict[str, Any]],
        absolute_position: Dict[str, Any],
        save_path:         Optional[Path],
        save_error:        Optional[str],
        elapsed_ms:        Optional[float],
    ) -> str:
        """
        Build the human-readable waypoint report string.

        Suitable for wiring into a ComfyUI ShowText node. Includes global
        statistics, grid navigation data (when available), absolute position,
        and I/O status.

        Parameters
        ----------
        waypoint_name : str
            Sanitized waypoint label.
        select_index : int
            Index that was extracted.
        batch_size : int
            Total batch size of the input.
        latent_shape : list of int
            Shape of the extracted latent (1-item batch form).
        stats : dict
            Output of ``_tensor_stats``.
        grid_cell : dict or None
            Grid cell metadata or None if unavailable.
        absolute_position : dict
            ``{'x': float, 'y': float}`` cumulative position.
        save_path : Path or None
            Actual file written, or None.
        save_error : str or None
            Error message if save failed, or None.
        elapsed_ms : float or None
            Total node execution time, or None if perf logging disabled.

        Returns
        -------
        str
            Multi-line report string.
        """
        lines = [
            "=== WayfinderGPS_Anchor ===",
            f"Waypoint:      {waypoint_name}",
            f"Selected:      index {select_index} of {batch_size} (0-based)",
            f"Latent shape:  {latent_shape}",
            "",
            "-- Global Statistics --",
            f"  mean:        {stats['global']['mean']}",
            f"  std:         {stats['global']['std']}",
            f"  variance:    {stats['global']['variance']}",
            f"  min:         {stats['global']['min']}",
            f"  max:         {stats['global']['max']}",
            "",
            "-- Navigation --",
        ]

        if grid_cell is not None:
            lines += [
                f"  grid col:    {grid_cell.get('grid_col', 'n/a')}",
                f"  grid row:    {grid_cell.get('grid_row', 'n/a')}",
                f"  offset X:    {grid_cell.get('offset_x', 0.0)}",
                f"  offset Y:    {grid_cell.get('offset_y', 0.0)}",
                f"  is center:   {grid_cell.get('is_center', False)}",
            ]
        else:
            lines.append("  (no grid map found in upstream meta)")

        lines += [
            f"  abs pos X:   {absolute_position['x']}",
            f"  abs pos Y:   {absolute_position['y']}",
            "",
        ]

        if save_path is not None:
            lines.append(f"Saved:         {save_path}")
        elif save_error is not None:
            lines.append(f"Save FAILED:   {save_error}")
        else:
            lines.append("Saved:         (disabled)")

        if elapsed_ms is not None:
            lines.append(f"Time:          {elapsed_ms:.1f} ms")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def anchor(
        self,
        latent_batch:        Dict[str, Any],
        select_index:        int,
        save_waypoint:       bool,
        waypoint_name:       str,
        enable_perf_logging: bool,
        wayfinder_meta:      Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """
        Execute the anchor operation.

        Parameters
        ----------
        latent_batch : dict
            ComfyUI LATENT dict with a 'samples' tensor [B, C, H, W].
        select_index : int
            Zero-based batch index to extract. Validated against batch size.
        save_waypoint : bool
            Write the waypoint payload to disk when True.
        waypoint_name : str
            Raw label; sanitized internally before any use.
        enable_perf_logging : bool
            Emit INFO-level timing messages to the module logger.
        wayfinder_meta : dict, optional
            Upstream Wayfinder metadata for coordinate and position inheritance.

        Returns
        -------
        tuple
            (anchored_latent, waypoint_meta, waypoint_report)
        """
        if enable_perf_logging:
            logging.basicConfig(level=logging.INFO)

        t_total = time.perf_counter()

        # Sanitize before anything else -- safe_name used consistently throughout
        safe_name = self._sanitize_filename(waypoint_name)

        # -- Tensor extraction (no_grad: no gradient graph needed) ---------
        with torch.no_grad():
            samples    = self._extract_samples(latent_batch, "latent_batch")
            batch_size = samples.shape[0]

            if select_index >= batch_size:
                raise ValueError(
                    f"WayfinderGPS_Anchor: select_index ({select_index}) is out of "
                    f"range for a batch of size {batch_size}. "
                    f"Valid range: 0 to {batch_size - 1}."
                )

            # Clone to sever the view-into-batch-storage relationship.
            # Without clone(), downstream mutations to the anchored latent
            # would corrupt the original batch tensor in place.
            selected = samples[select_index : select_index + 1].clone()

        # -- Statistics (one CPU transfer avoids 25+ CUDA device syncs) ----
        stats = self._tensor_stats(selected[0])

        # -- Metadata inheritance ------------------------------------------
        meta         = wayfinder_meta or {}
        x_strength   = float(meta.get("x_strength", 1.0))
        y_strength   = float(meta.get("y_strength", 1.0))
        grid_cell    = self._resolve_grid_coordinate(meta, select_index)
        absolute_pos = self._compute_absolute_position(
            meta, grid_cell, x_strength, y_strength
        )

        # -- Payload assembly (safe_name used for all name fields) ----------
        payload: Dict[str, Any] = {
            "waypoint_name":         safe_name,
            "select_index":          select_index,
            "batch_size":            batch_size,
            "latent_shape":          list(selected.shape),
            "statistics":            stats,
            "grid_cell":             grid_cell,
            "absolute_position":     absolute_pos,
            # Written back so downstream Anchor nodes accumulate correctly
            "accumulated_position":  absolute_pos,
            "interpolation_mode":    meta.get("interpolation_mode"),
            "normalize_vectors":     meta.get("normalize_vectors"),
            "upstream_grid_size": [
                meta.get("grid_size_x"),
                meta.get("grid_size_y"),
            ],
            "upstream_output_shape": meta.get("output_shape"),
        }

        # -- File I/O (non-fatal) ------------------------------------------
        # open('x') is exclusive create: raises FileExistsError if another
        # process claims the same path between _versioned_path() and open(),
        # eliminating the TOCTOU window present in exists() + open('w').
        save_path:  Optional[Path] = None
        save_error: Optional[str]  = None

        if save_waypoint:
            try:
                WAYFINDER_DIR.mkdir(parents=True, exist_ok=True)
                target   = WAYFINDER_DIR / f"{safe_name}.json"
                out_path = self._versioned_path(target)
                with out_path.open("x", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, ensure_ascii=False)
                save_path = out_path
                if enable_perf_logging:
                    log.info(f"[WayfinderGPS_Anchor] saved: {out_path}")
            except FileExistsError as exc:
                save_error = str(exc)
                warnings.warn(
                    f"WayfinderGPS_Anchor: target file was claimed between "
                    f"versioning check and write (race condition). "
                    f"Re-queue to generate a new versioned name. Detail: {exc}",
                    UserWarning, stacklevel=2,
                )
            except PermissionError as exc:
                save_error = str(exc)
                warnings.warn(
                    f"WayfinderGPS_Anchor: permission denied writing to "
                    f"{WAYFINDER_DIR}. Check directory permissions. Detail: {exc}",
                    UserWarning, stacklevel=2,
                )
            except OSError as exc:
                save_error = str(exc)
                warnings.warn(
                    f"WayfinderGPS_Anchor: OS error during save "
                    f"(check path length, disk space, filesystem state). "
                    f"Detail: {exc}",
                    UserWarning, stacklevel=2,
                )
            except Exception as exc:
                save_error = str(exc)
                warnings.warn(
                    f"WayfinderGPS_Anchor: unexpected error during save. "
                    f"Detail: {exc}",
                    UserWarning, stacklevel=2,
                )

        # -- Output assembly -----------------------------------------------
        out_latent = {k: v for k, v in latent_batch.items() if k != "samples"}
        out_latent["samples"] = selected

        elapsed_ms = (time.perf_counter() - t_total) * 1000
        if enable_perf_logging:
            log.info(f"[WayfinderGPS_Anchor] total: {elapsed_ms:.2f} ms")

        waypoint_meta: Dict[str, Any] = {
            **payload,
            "save_path":     str(save_path) if save_path else None,
            "save_error":    save_error,
            "save_waypoint": save_waypoint,
            "elapsed_ms":    round(elapsed_ms, 2),
        }

        report = self._build_report(
            waypoint_name=safe_name,
            select_index=select_index,
            batch_size=batch_size,
            latent_shape=list(selected.shape),
            stats=stats,
            grid_cell=grid_cell,
            absolute_position=absolute_pos,
            save_path=save_path,
            save_error=save_error,
            elapsed_ms=elapsed_ms if enable_perf_logging else None,
        )

        return (out_latent, waypoint_meta, report)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "WayfinderGPS_Anchor": WayfinderGPS_Anchor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WayfinderGPS_Anchor": "Wayfinder GPS Anchor",
}

# Alias for unit testing without node instantiation
_sanitize_filename_standalone = WayfinderGPS_Anchor._sanitize_filename
```

---

# ComfyUI Workflows

## Workflow 1: Basic Compass Pro - Style Transfer

```json
{
  "last_node_id": 12,
  "last_link_id": 11,
  "nodes": [
    {
      "id": 1,
      "type": "CheckpointLoaderSimple",
      "pos": [0, 0],
      "size": [320, 100],
      "widgets_values": ["v1-5-pruned-emaonly.safetensors", "default"],
      "title": "Load Checkpoint"
    },
    {
      "id": 2,
      "type": "CLIPTextEncode",
      "pos": [350, -50],
      "size": [400, 150],
      "widgets_values": ["a photo of a sunny beach, bright, warm colors, high contrast"],
      "title": "Positive Prompt (Target Style)"
    },
    {
      "id": 3,
      "type": "CLIPTextEncode",
      "pos": [350, 150],
      "size": [400, 150],
      "widgets_values": ["a photo of a dark forest, muted, cool colors"],
      "title": "Negative/Origin Prompt"
    },
    {
      "id": 4,
      "type": "EmptyLatentImage",
      "pos": [350, 350],
      "size": [320, 110],
      "widgets_values": [512, 512, 1],
      "title": "Base Latent (Your Image)"
    },
    {
      "id": 5,
      "type": "KSampler",
      "pos": [800, -50],
      "size": [320, 260],
      "widgets_values": [42, "fixed", 20, 8, "euler", "normal", 1.0],
      "title": "Sample Target Style"
    },
    {
      "id": 6,
      "type": "KSampler",
      "pos": [800, 250],
      "size": [320, 260],
      "widgets_values": [42, "fixed", 20, 8, "euler", "normal", 1.0],
      "title": "Sample Origin Style"
    },
    {
      "id": 7,
      "type": "VAEDecode",
      "pos": [1150, -50],
      "size": [210, 50],
      "title": "Decode Target Preview"
    },
    {
      "id": 8,
      "type": "VAEDecode",
      "pos": [1150, 250],
      "size": [210, 50],
      "title": "Decode Origin Preview"
    },
    {
      "id": 9,
      "type": "WayfinderCompass_Pro",
      "pos": [800, 550],
      "size": [400, 500],
      "widgets_values": [
        0.75,
        "Standard",
        false,
        -10.0,
        10.0,
        false,
        false,
        false
      ],
      "title": "Wayfinder Compass Pro"
    },
    {
      "id": 10,
      "type": "VAEDecode",
      "pos": [1250, 550],
      "size": [210, 50],
      "title": "Decode Result"
    },
    {
      "id": 11,
      "type": "PreviewImage",
      "pos": [1500, -50],
      "size": [300, 250],
      "title": "Target Style Preview"
    },
    {
      "id": 12,
      "type": "PreviewImage",
      "pos": [1500, 250],
      "size": [300, 250],
      "title": "Origin Style Preview"
    }
  ],
  "links": [
    [1, 1, 0, 2, 0, "CLIP"],
    [2, 1, 0, 3, 0, "CLIP"],
    [3, 2, 0, 5, 1, "CONDITIONING"],
    [4, 3, 0, 6, 1, "CONDITIONING"],
    [5, 4, 0, 5, 2, "LATENT"],
    [6, 4, 0, 6, 2, "LATENT"],
    [7, 1, 1, 5, 0, "VAE"],
    [8, 1, 1, 6, 0, "VAE"],
    [9, 5, 0, 7, 0, "LATENT"],
    [10, 6, 0, 8, 0, "LATENT"],
    [11, 7, 0, 11, 0, "IMAGE"],
    [12, 8, 0, 12, 0, "IMAGE"]
  ],
  "groups": [
    {
      "title": "Style Direction Definition",
      "bounding": [340, -100, 820, 520],
      "color": "#3f789e"
    },
    {
      "title": "Compass Navigation",
      "bounding": [790, 530, 420, 520],
      "color": "#2a363b"
    }
  ],
  "config": {},
  "extra": {},
  "version": 0.4
}
```

---

## Workflow 2: Manifold Explorer with GPS Anchor

```json
{
  "last_node_id": 18,
  "last_link_id": 20,
  "nodes": [
    {
      "id": 1,
      "type": "CheckpointLoaderSimple",
      "pos": [0, 0],
      "size": [320, 100],
      "widgets_values": ["v1-5-pruned-emaonly.safetensors", "default"],
      "title": "Load Checkpoint"
    },
    {
      "id": 2,
      "type": "CLIPTextEncode",
      "pos": [350, 0],
      "size": [400, 150],
      "widgets_values": ["a portrait photo, neutral lighting"],
      "title": "Prompt"
    },
    {
      "id": 3,
      "type": "EmptyLatentImage",
      "pos": [350, 200],
      "size": [320, 110],
      "widgets_values": [512, 512, 1],
      "title": "Center Latent"
    },
    {
      "id": 4,
      "type": "KSampler",
      "pos": [800, 0],
      "size": [320, 260],
      "widgets_values": [42, "fixed", 20, 8, "euler", "normal", 1.0],
      "title": "Sample Center"
    },
    {
      "id": 5,
      "type": "CLIPTextEncode",
      "pos": [350, -200],
      "size": [400, 150],
      "widgets_values": ["bright, high key, warm lighting"],
      "title": "X-Axis Target (Brightness)"
    },
    {
      "id": 6,
      "type": "CLIPTextEncode",
      "pos": [350, -350],
      "size": [400, 150],
      "widgets_values": ["dark, low key, cool lighting"],
      "title": "X-Axis Origin (Darkness)"
    },
    {
      "id": 7,
      "type": "KSampler",
      "pos": [800, -350],
      "size": [320, 260],
      "widgets_values": [42, "fixed", 20, 8, "euler", "normal", 1.0],
      "title": "Sample X Target"
    },
    {
      "id": 8,
      "type": "KSampler",
      "pos": [800, -600],
      "size": [320, 260],
      "widgets_values": [42, "fixed", 20, 8, "euler", "normal", 1.0],
      "title": "Sample X Origin"
    },
    {
      "id": 9,
      "type": "WayfinderCompass_Pro",
      "pos": [1200, -500],
      "size": [350, 400],
      "widgets_values": [1.0, "Normalized", false, -10.0, 10.0, false, false, false],
      "title": "X Vector (Brightness Direction)"
    },
    {
      "id": 10,
      "type": "CLIPTextEncode",
      "pos": [350, 400],
      "size": [400, 150],
      "widgets_values": ["sharp, detailed, high contrast"],
      "title": "Y-Axis Target (Sharpness)"
    },
    {
      "id": 11,
      "type": "CLIPTextEncode",
      "pos": [350, 550],
      "size": [400, 150],
      "widgets_values": ["soft, blurry, low contrast"],
      "title": "Y-Axis Origin (Softness)"
    },
    {
      "id": 12,
      "type": "KSampler",
      "pos": [800, 400],
      "size": [320, 260],
      "widgets_values": [42, "fixed", 20, 8, "euler", "normal", 1.0],
      "title": "Sample Y Target"
    },
    {
      "id": 13,
      "type": "KSampler",
      "pos": [800, 650],
      "size": [320, 260],
      "widgets_values": [42, "fixed", 20, 8, "euler", "normal", 1.0],
      "title": "Sample Y Origin"
    },
    {
      "id": 14,
      "type": "WayfinderCompass_Pro",
      "pos": [1200, 500],
      "size": [350, 400],
      "widgets_values": [1.0, "Normalized", false, -10.0, 10.0, false, false, false],
      "title": "Y Vector (Sharpness Direction)"
    },
    {
      "id": 15,
      "type": "WayfinderManifold_Explorer",
      "pos": [1600, 0],
      "size": [400, 500],
      "widgets_values": [5, 5, 1.5, 1.5, "Linear", true, false, -10.0, 10.0, false],
      "title": "Manifold Explorer (5x5 Grid)"
    },
    {
      "id": 16,
      "type": "WayfinderGPS_Anchor",
      "pos": [2050, 0],
      "size": [350, 350],
      "widgets_values": [12, true, "bright_sharp_corner", false],
      "title": "GPS Anchor (Select Grid Cell)"
    },
    {
      "id": 17,
      "type": "VAEDecode",
      "pos": [2450, 0],
      "size": [210, 50],
      "title": "Decode Anchored Latent"
    },
    {
      "id": 18,
      "type": "PreviewImage",
      "pos": [2700, 0],
      "size": [400, 400],
      "title": "Selected Grid Cell"
    }
  ],
  "links": [
    [1, 1, 0, 2, 0, "CLIP"],
    [2, 2, 0, 4, 1, "CONDITIONING"],
    [3, 3, 0, 4, 2, "LATENT"],
    [4, 1, 1, 4, 0, "VAE"],
    [5, 1, 0, 5, 0, "CLIP"],
    [6, 1, 0, 6, 0, "CLIP"],
    [7, 5, 0, 7, 1, "CONDITIONING"],
    [8, 6, 0, 8, 1, "CONDITIONING"],
    [9, 1, 1, 7, 0, "VAE"],
    [10, 1, 1, 8, 0, "VAE"],
    [11, 7, 0, 9, 1, "LATENT"],
    [12, 8, 0, 9, 2, "LATENT"],
    [13, 9, 0, 15, 1, "LATENT"],
    [14, 1, 0, 10, 0, "CLIP"],
    [15, 1, 0, 11, 0, "CLIP"],
    [16, 10, 0, 12, 1, "CONDITIONING"],
    [17, 11, 0, 13, 1, "CONDITIONING"],
    [18, 1, 1, 12, 0, "VAE"],
    [19, 1, 1, 13, 0, "VAE"],
    [20, 12, 0, 14, 1, "LATENT"],
    [21, 13, 0, 14, 2, "LATENT"],
    [22, 14, 0, 15, 2, "LATENT"],
    [23, 4, 0, 15, 0, "LATENT"],
    [24, 15, 0, 16, 0, "LATENT"],
    [25, 15, 1, 16, 5, "DICT"],
    [26, 16, 0, 17, 0, "LATENT"],
    [27, 1, 1, 17, 1, "VAE"],
    [28, 17, 0, 18, 0, "IMAGE"]
  ],
  "groups": [
    {
      "title": "X-Axis: Brightness Direction",
      "bounding": [340, -400, 860, 500],
      "color": "#e8a735"
    },
    {
      "title": "Center Point",
      "bounding": [340, -50, 820, 400],
      "color": "#3f789e"
    },
    {
      "title": "Y-Axis: Sharpness Direction",
      "bounding": [340, 350, 860, 600],
      "color": "#35e87a"
    },
    {
      "title": "2D Grid Exploration",
      "bounding": [1590, -50, 420, 550],
      "color": "#2a363b"
    },
    {
      "title": "GPS Anchor & Output",
      "bounding": [2040, -50, 600, 450],
      "color": "#8b35e8"
    }
  ],
  "config": {},
  "extra": {},
  "version": 0.4
}
```

---

## Workflow 3: Semantic Slider with PCA

```json
{
  "last_node_id": 14,
  "last_link_id": 16,
  "nodes": [
    {
      "id": 1,
      "type": "CheckpointLoaderSimple",
      "pos": [0, 0],
      "size": [320, 100],
      "widgets_values": ["v1-5-pruned-emaonly.safetensors", "default"],
      "title": "Load Checkpoint"
    },
    {
      "id": 2,
      "type": "CLIPTextEncode",
      "pos": [350, 0],
      "size": [400, 200],
      "widgets_values": ["various landscape photos: mountains, beaches, forests, deserts, with different lighting conditions and times of day"],
      "title": "Diverse Prompt for PCA Batch"
    },
    {
      "id": 3,
      "type": "EmptyLatentImage",
      "pos": [350, 250],
      "size": [320, 110],
      "widgets_values": [512, 512, 8],
      "title": "Batch of 8 for PCA"
    },
    {
      "id": 4,
      "type": "KSampler",
      "pos": [800, 0],
      "size": [320, 260],
      "widgets_values": [0, "randomize", 20, 8, "euler", "normal", 1.0],
      "title": "Sample Diverse Batch"
    },
    {
      "id": 5,
      "type": "VAEDecode",
      "pos": [1150, 0],
      "size": [210, 50],
      "title": "Decode Batch Preview"
    },
    {
      "id": 6,
      "type": "PreviewImage",
      "pos": [1400, 0],
      "size": [500, 500],
      "title": "PCA Source Batch"
    },
    {
      "id": 7,
      "type": "CLIPTextEncode",
      "pos": [350, 400],
      "size": [400, 150],
      "widgets_values": ["a serene mountain landscape at golden hour"],
      "title": "Base Image Prompt"
    },
    {
      "id": 8,
      "type": "EmptyLatentImage",
      "pos": [350, 580],
      "size": [320, 110],
      "widgets_values": [512, 512, 1],
      "title": "Base Latent"
    },
    {
      "id": 9,
      "type": "KSampler",
      "pos": [800, 400],
      "size": [320, 260],
      "widgets_values": [12345, "fixed", 20, 8, "euler", "normal", 1.0],
      "title": "Sample Base Image"
    },
    {
      "id": 10,
      "type": "Wayfinder_SemanticSlider",
      "pos": [1200, 400],
      "size": [400, 350],
      "widgets_values": [1, 0.5, false],
      "title": "Semantic Slider (PC1)"
    },
    {
      "id": 11,
      "type": "VAEDecode",
      "pos": [1650, 400],
      "size": [210, 50],
      "title": "Decode Modified"
    },
    {
      "id": 12,
      "type": "PreviewImage",
      "pos": [1900, 300],
      "size": [400, 400],
      "title": "PC1 Modified Result"
    },
    {
      "id": 13,
      "type": "VAEDecode",
      "pos": [1650, 500],
      "size": [210, 50],
      "title": "Decode PC Preview"
    },
    {
      "id": 14,
      "type": "PreviewImage",
      "pos": [1900, 700],
      "size": [400, 400],
      "title": "PC Vector Visualization"
    }
  ],
  "links": [
    [1, 1, 0, 2, 0, "CLIP"],
    [2, 2, 0, 4, 1, "CONDITIONING"],
    [3, 3, 0, 4, 2, "LATENT"],
    [4, 1, 1, 4, 0, "VAE"],
    [5, 4, 0, 5, 0, "LATENT"],
    [6, 5, 0, 6, 0, "IMAGE"],
    [7, 1, 0, 7, 0, "CLIP"],
    [8, 7, 0, 9, 1, "CONDITIONING"],
    [9, 8, 0, 9, 2, "LATENT"],
    [10, 1, 1, 9, 0, "VAE"],
    [11, 4, 0, 10, 0, "LATENT"],
    [12, 9, 0, 10, 1, "LATENT"],
    [13, 10, 0, 11, 0, "LATENT"],
    [14, 1, 1, 11, 1, "VAE"],
    [15, 11, 0, 12, 0, "IMAGE"],
    [16, 10, 1, 13, 0, "LATENT"],
    [17, 1, 1, 13, 1, "VAE"],
    [18, 13, 0, 14, 0, "IMAGE"]
  ],
  "groups": [
    {
      "title": "Step 1: Generate Diverse Batch for PCA",
      "bounding": [340, -50, 820, 400],
      "color": "#3f789e"
    },
    {
      "title": "Step 2: Generate Base Image to Modify",
      "bounding": [340, 350, 820, 400],
      "color": "#35e87a"
    },
    {
      "title": "Step 3: Apply PCA Slider",
      "bounding": [1190, 350, 500, 420],
      "color": "#e8a735"
    }
  ],
  "config": {},
  "extra": {},
  "version": 0.4
}
```

---

## Workflow 4: Cross-Modal Bridge with Compass Pro

```json
{
  "last_node_id": 15,
  "last_link_id": 18,
  "nodes": [
    {
      "id": 1,
      "type": "CheckpointLoaderSimple",
      "pos": [0, 0],
      "size": [320, 100],
      "widgets_values": ["v1-5-pruned-emaonly.safetensors", "default"],
      "title": "Load Checkpoint"
    },
    {
      "id": 2,
      "type": "CLIPTextEncode",
      "pos": [350, 0],
      "size": [400, 150],
      "widgets_values": ["a portrait photo"],
      "title": "Base Prompt"
    },
    {
      "id": 3,
      "type": "EmptyLatentImage",
      "pos": [350, 200],
      "size": [320, 110],
      "widgets_values": [512, 512, 1],
      "title": "Base Latent"
    },
    {
      "id": 4,
      "type": "KSampler",
      "pos": [800, 0],
      "size": [320, 260],
      "widgets_values": [42, "fixed", 20, 8, "euler", "normal", 1.0],
      "title": "Sample Base Image"
    },
    {
      "id": 5,
      "type": "VAEDecode",
      "pos": [1150, 0],
      "size": [210, 50],
      "title": "Decode Original"
    },
    {
      "id": 6,
      "type": "PreviewImage",
      "pos": [1400, 0],
      "size": [350, 350],
      "title": "Original Image"
    },
    {
      "id": 7,
      "type": "Wayfinder_CrossModalBridge",
      "pos": [800, 300],
      "size": [500, 250],
      "widgets_values": [
        "bright warm saturated cinematic",
        "Keyword_Heuristics"
      ],
      "title": "Cross-Modal Bridge (Text to Latent Vector)"
    },
    {
      "id": 8,
      "type": "WayfinderCompass_Pro",
      "pos": [1350, 300],
      "size": [400, 450],
      "widgets_values": [1.0, "Standard", false, -10.0, 10.0, false, false, false],
      "title": "Apply Style Direction"
    },
    {
      "id": 9,
      "type": "VAEDecode",
      "pos": [1800, 300],
      "size": [210, 50],
      "title": "Decode Modified"
    },
    {
      "id": 10,
      "type": "PreviewImage",
      "pos": [2050, 200],
      "size": [350, 350],
      "title": "Keyword Heuristics Result"
    },
    {
      "id": 11,
      "type": "Wayfinder_CrossModalBridge",
      "pos": [800, 580],
      "size": [500, 250],
      "widgets_values": [
        "{\"luminance\": 0.3, \"warm_cool\": 0.25, \"detail\": 0.15}",
        "Manual_JSON"
      ],
      "title": "Cross-Modal Bridge (JSON Control)"
    },
    {
      "id": 12,
      "type": "WayfinderCompass_Pro",
      "pos": [1350, 580],
      "size": [400, 450],
      "widgets_values": [1.0, "Standard", false, -10.0, 10.0, false, false, false],
      "title": "Apply JSON Direction"
    },
    {
      "id": 13,
      "type": "VAEDecode",
      "pos": [1800, 580],
      "size": [210, 50],
      "title": "Decode JSON Modified"
    },
    {
      "id": 14,
      "type": "PreviewImage",
      "pos": [2050, 580],
      "size": [350, 350],
      "title": "Manual JSON Result"
    },
    {
      "id": 15,
      "type": "CLIPTextEncode",
      "pos": [350, -200],
      "size": [400, 150],
      "widgets_values": [""],
      "title": "Negative Prompt"
    }
  ],
  "links": [
    [1, 1, 0, 2, 0, "CLIP"],
    [2, 2, 0, 4, 1, "CONDITIONING"],
    [3, 15, 0, 4, 2, "CONDITIONING"],
    [4, 3, 0, 4, 3, "LATENT"],
    [5, 1, 1, 4, 0, "VAE"],
    [6, 4, 0, 5, 0, "LATENT"],
    [7, 5, 0, 6, 0, "IMAGE"],
    [8, 4, 0, 7, 1, "LATENT"],
    [9, 7, 0, 8, 1, "LATENT"],
    [10, 7, 1, 8, 2, "LATENT"],
    [11, 4, 0, 8, 0, "LATENT"],
    [12, 8, 0, 9, 0, "LATENT"],
    [13, 1, 1, 9, 1, "VAE"],
    [14, 9, 0, 10, 0, "IMAGE"],
    [15, 4, 0, 11, 1, "LATENT"],
    [16, 11, 0, 12, 1, "LATENT"],
    [17, 11, 1, 12, 2, "LATENT"],
    [18, 4, 0, 12, 0, "LATENT"],
    [19, 12, 0, 13, 0, "LATENT"],
    [20, 1, 1, 13, 1, "VAE"],
    [21, 13, 0, 14, 0, "IMAGE"]
  ],
  "groups": [
    {
      "title": "Base Image Generation",
      "bounding": [340, -250, 820, 500],
      "color": "#3f789e"
    },
    {
      "title": "Text-to-Vector via Keyword Heuristics",
      "bounding": [790, 250, 600, 350],
      "color": "#e8a735"
    },
    {
      "title": "JSON Precise Control",
      "bounding": [790, 530, 600, 350],
      "color": "#35e87a"
    }
  ],
  "config": {},
  "extra": {},
  "version": 0.4
}
```

---

## Workflow 5: Complete Pipeline - All Nodes Integrated

```json
{
  "last_node_id": 25,
  "last_link_id": 35,
  "nodes": [
    {
      "id": 1,
      "type": "CheckpointLoaderSimple",
      "pos": [0, 0],
      "size": [320, 100],
      "widgets_values": ["v1-5-pruned-emaonly.safetensors", "default"],
      "title": "Load Checkpoint"
    },
    {
      "id": 2,
      "type": "CLIPTextEncode",
      "pos": [0, 150],
      "size": [300, 100],
      "widgets_values": ["a landscape photo"],
      "title": "Base Prompt"
    },
    {
      "id": 3,
      "type": "EmptyLatentImage",
      "pos": [0, 300],
      "size": [300, 80],
      "widgets_values": [512, 512, 1],
      "title": "Base Latent"
    },
    {
      "id": 4,
      "type": "KSampler",
      "pos": [350, 150],
      "size": [280, 230],
      "widgets_values": [42, "fixed", 20, 8, "euler", "normal", 1.0],
      "title": "Sample Base"
    },
    {
      "id": 5,
      "type": "EmptyLatentImage",
      "pos": [0, 450],
      "size": [300, 80],
      "widgets_values": [512, 512, 6],
      "title": "PCA Batch (6 samples)"
    },
    {
      "id": 6,
      "type": "KSampler",
      "pos": [350, 420],
      "size": [280, 230],
      "widgets_values": [0, "randomize", 15, 7, "euler", "normal", 0.8],
      "title": "Sample PCA Batch"
    },
    {
      "id": 7,
      "type": "Wayfinder_SemanticSlider",
      "pos": [680, 150],
      "size": [320, 300],
      "widgets_values": [1, 0.75, false],
      "title": "PCA Slider (PC1)"
    },
    {
      "id": 8,
      "type": "Wayfinder_CrossModalBridge",
      "pos": [680, 480],
      "size": [350, 200],
      "widgets_values": ["bright warm cinematic", "Keyword_Heuristics"],
      "title": "Text Style Vector"
    },
    {
      "id": 9,
      "type": "WayfinderCompass_Pro",
      "pos": [1080, 150],
      "size": [320, 400],
      "widgets_values": [0.5, "Standard", false, -10.0, 10.0, false, false, false],
      "title": "Apply Text Style"
    },
    {
      "id": 10,
      "type": "WayfinderManifold_Explorer",
      "pos": [1080, 580],
      "size": [380, 400],
      "widgets_values": [3, 3, 0.8, 0.8, "Linear", true, false, -10.0, 10.0, false],
      "title": "Create 3x3 Exploration Grid"
    },
    {
      "id": 11,
      "type": "WayfinderGPS_Anchor",
      "pos": [1500, 580],
      "size": [320, 300],
      "widgets_values": [4, true, "selected_point", false],
      "title": "Anchor Center Point"
    },
    {
      "id": 12,
      "type": "VAEDecode",
      "pos": [1450, 150],
      "size": [200, 50],
      "title": "Decode Final"
    },
    {
      "id": 13,
      "type": "PreviewImage",
      "pos": [1700, 100],
      "size": [350, 350],
      "title": "Final Result"
    },
    {
      "id": 14,
      "type": "VAEDecode",
      "pos": [1500, 400],
      "size": [200, 50],
      "title": "Decode Grid Cell"
    },
    {
      "id": 15,
      "type": "PreviewImage",
      "pos": [1700, 480],
      "size": [350, 350],
      "title": "Anchored Grid Cell"
    },
    {
      "id": 16,
      "type": "ShowText",
      "pos": [1500, 850],
      "size": [500, 200],
      "widgets_values": [""],
      "title": "Waypoint Report"
    },
    {
      "id": 17,
      "type": "CLIPTextEncode",
      "pos": [0, 600],
      "size": [300, 100],
      "widgets_values": ["bright high key"],
      "title": "X Target"
    },
    {
      "id": 18,
      "type": "CLIPTextEncode",
      "pos": [0, 750],
      "size": [300, 100],
      "widgets_values": ["dark low key"],
      "title": "X Origin"
    },
    {
      "id": 19,
      "type": "KSampler",
      "pos": [350, 700],
      "size": [280, 230],
      "widgets_values": [42, "fixed", 15, 7, "euler", "normal", 1.0],
      "title": "Sample X Target"
    },
    {
      "id": 20,
      "type": "KSampler",
      "pos": [350, 960],
      "size": [280, 230],
      "widgets_values": [42, "fixed", 15, 7, "euler", "normal", 1.0],
      "title": "Sample X Origin"
    },
    {
      "id": 21,
      "type": "WayfinderCompass_Pro",
      "pos": [680, 720],
      "size": [320, 350],
      "widgets_values": [1.0, "Normalized", false, -10.0, 10.0, false, false, false],
      "title": "X Direction Vector"
    },
    {
      "id": 22,
      "type": "CLIPTextEncode",
      "pos": [0, 1100],
      "size": [300, 100],
      "widgets_values": ["sharp detailed"],
      "title": "Y Target"
    },
    {
      "id": 23,
      "type": "CLIPTextEncode",
      "pos": [0, 1250],
      "size": [300, 100],
      "widgets_values": ["soft blurry"],
      "title": "Y Origin"
    },
    {
      "id": 24,
      "type": "KSampler",
      "pos": [350, 1200],
      "size": [280, 230],
      "widgets_values": [42, "fixed", 15, 7, "euler", "normal", 1.0],
      "title": "Sample Y Target"
    },
    {
      "id": 25,
      "type": "KSampler",
      "pos": [350, 1460],
      "size": [280, 230],
      "widgets_values": [42, "fixed", 15, 7, "euler", "normal", 1.0],
      "title": "Sample Y Origin"
    },
    {
      "id": 26,
      "type": "WayfinderCompass_Pro",
      "pos": [680, 1200],
      "size": [320, 350],
      "widgets_values": [1.0, "Normalized", false, -10.0, 10.0, false, false, false],
      "title": "Y Direction Vector"
    }
  ],
  "links": [
    [1, 1, 0, 2, 0, "CLIP"],
    [2, 2, 0, 4, 1, "CONDITIONING"],
    [3, 3, 0, 4, 3, "LATENT"],
    [4, 1, 1, 4, 0, "VAE"],
    [5, 2, 0, 6, 1, "CONDITIONING"],
    [6, 5, 0, 6, 3, "LATENT"],
    [7, 1, 1, 6, 0, "VAE"],
    [8, 6, 0, 7, 0, "LATENT"],
    [9, 4, 0, 7, 1, "LATENT"],
    [10, 4, 0, 8, 1, "LATENT"],
    [11, 7, 0, 9, 0, "LATENT"],
    [12, 8, 0, 9, 1, "LATENT"],
    [13, 8, 1, 9, 2, "LATENT"],
    [14, 9, 0, 12, 0, "LATENT"],
    [15, 1, 1, 12, 1, "VAE"],
    [16, 12, 0, 13, 0, "IMAGE"],
    [17, 21, 0, 10, 1, "LATENT"],
    [18, 26, 0, 10, 2, "LATENT"],
    [19, 4, 0, 10, 0, "LATENT"],
    [20, 10, 0, 11, 0, "LATENT"],
    [21, 10, 1, 11, 5, "DICT"],
    [22, 11, 0, 14, 0, "LATENT"],
    [23, 1, 1, 14, 1, "VAE"],
    [24, 14, 0, 15, 0, "IMAGE"],
    [25, 11, 2, 16, 0, "STRING"],
    [26, 1, 0, 17, 0, "CLIP"],
    [27, 1, 0, 18, 0, "CLIP"],
    [28, 17, 0, 19, 1, "CONDITIONING"],
    [29, 18, 0, 20, 1, "CONDITIONING"],
    [30, 1, 1, 19, 0, "VAE"],
    [31, 1, 1, 20, 0, "VAE"],
    [32, 19, 0, 21, 1, "LATENT"],
    [33, 20, 0, 21, 2, "LATENT"],
    [34, 1, 0, 22, 0, "CLIP"],
    [35, 1, 0, 23, 0, "CLIP"],
    [36, 22, 0, 24, 1, "CONDITIONING"],
    [37, 23, 0, 25, 1, "CONDITIONING"],
    [38, 1, 1, 24, 0, "VAE"],
    [39, 1, 1, 25, 0, "VAE"],
    [40, 24, 0, 26, 1, "LATENT"],
    [41, 25, 0, 26, 2, "LATENT"]
  ],
  "groups": [
    {
      "title": "1. Base Image + PCA Analysis",
      "bounding": [-20, -50, 1040, 600],
      "color": "#3f789e"
    },
    {
      "title": "2. Text-to-Style + Apply",
      "bounding": [660, 450, 760, 350],
      "color": "#e8a735"
    },
    {
      "title": "3. Define Exploration Axes",
      "bounding": [-20, 680, 1040, 1060],
      "color": "#35e87a"
    },
    {
      "title": "4. Grid Exploration + Anchor",
      "bounding": [1060, 550, 800, 550],
      "color": "#8b35e8"
    }
  ],
  "config": {},
  "extra": {},
  "version": 0.4
}
```