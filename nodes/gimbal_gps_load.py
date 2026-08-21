import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

log = logging.getLogger("GimbalGPS_Load")


def _resolve_waypoint_dir() -> Path:
    """Mirror the resolution logic from GimbalGPS_Anchor for consistency."""
    env_override = os.environ.get("GIMBAL_OUTPUT_DIR") or os.environ.get("WAYFINDER_OUTPUT_DIR", "").strip()
    if env_override:
        return Path(env_override)
    try:
        import folder_paths
        root = Path(folder_paths.get_output_directory())
        return root / "gimbal"
    except Exception as exc:
        log.error(f"folder_paths unavailable, using ./output/gimbal. Error: {exc}")
        return Path("output") / "gimbal"


GIMBAL_DIR    = _resolve_waypoint_dir()
WAYFINDER_DIR = GIMBAL_DIR


def _list_waypoint_files() -> List[str]:
    """Enumerate available .json waypoints for the dropdown widget."""
    if not GIMBAL_DIR.exists():
        return ["<no waypoints found>"]
    files = sorted(
        p.name for p in GIMBAL_DIR.glob("*.json")
        if p.is_file() and p.stat().st_size > 0
    )
    return files if files else ["<no waypoints found>"]


class GimbalGPS_Load:
    """
    [VibeCheck Badge: 🟢 Stabilized]
    
    Companion loader for GimbalGPS_Anchor.

    Reads a previously-saved waypoint .json and reconstructs a usable LATENT
    by combining the saved statistical signature with a provided reference
    latent (for shape/device/dtype) and an optional live generation.

    IMPORTANT: The GPS_Anchor writes STATISTICS about the latent, not the raw
    tensor data itself. True tensor persistence would require a .safetensors
    or .pt companion file. This loader offers three reconstruction modes:

      - Reference_Only:   Returns the reference latent unchanged, but attaches
                          the waypoint's metadata for downstream routing.
                          (Useful when you've regenerated the anchor via a
                          fixed seed and just need the saved grid coordinates.)

      - Statistical_Match: Rescales the reference latent's per-channel mean
                           and std to match the waypoint's recorded statistics.
                           A lossy but reproducible approximation.

      - Coord_Steering:   Returns the reference latent plus a metadata dict
                          containing absolute_position (x, y) — intended to
                          be chained into a downstream Compass for directional
                          steering based on the saved grid coordinates.
    
    What's left:
    - Fine-tune parameter ranges for edge cases.
    """

    CATEGORY     = "Gimbal/Navigation"
    RETURN_TYPES = ("LATENT", "DICT", "STRING")
    RETURN_NAMES = ("reconstructed_latent", "waypoint_meta", "load_report")
    FUNCTION     = "load"


    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "waypoint_file": (_list_waypoint_files(),),
                "reference_latent": ("LATENT",),
                "mode": (
                    ["Reference_Only", "Statistical_Match", "Coord_Steering"],
                    {"default": "Statistical_Match"},
                ),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "display": "slider",
                }),
                "enable_perf_logging": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "manual_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, waypoint_file: str, **kwargs: Any) -> str:
        """Force re-execution when file contents change on disk."""
        try:
            p = GIMBAL_DIR / waypoint_file
            if p.exists():
                return str(p.stat().st_mtime)
        except Exception:
            pass
        return waypoint_file

    @staticmethod
    def _resolve_path(waypoint_file: str, manual_path: str) -> Path:
        if manual_path and manual_path.strip():
            p = Path(manual_path.strip())
            if p.is_file():
                return p
            raise FileNotFoundError(f"manual_path not found: {p}")
        if waypoint_file == "<no waypoints found>":
            raise FileNotFoundError(
                f"No waypoint files in {GIMBAL_DIR}. "
                "Run GimbalGPS_Anchor with save_waypoint=True first, "
                "or specify manual_path."
            )
        p = GIMBAL_DIR / waypoint_file
        if not p.is_file():
            raise FileNotFoundError(f"Waypoint not found: {p}")
        return p

    @staticmethod
    def _load_payload(path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt waypoint JSON at {path}: {exc}")
        if not isinstance(payload, dict):
            raise ValueError(f"Waypoint root must be an object, got {type(payload).__name__}")
        if "statistics" not in payload:
            raise ValueError(f"Waypoint missing 'statistics' key: {path}")
        return payload

    @staticmethod
    def _extract_samples(d: Dict[str, Any], name: str) -> torch.Tensor:
        s = d.get("samples")
        if s is None:
            raise ValueError(f"'{name}' missing 'samples' key.")
        if s.ndim != 4:
            raise ValueError(f"'{name}.samples' must be 4-D [B, C, H, W], got {list(s.shape)}.")
        return s

    @staticmethod
    def _statistical_rescale(
        ref: torch.Tensor,
        target_stats: Dict[str, Any],
        strength: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Per-channel mean/std matching:
            $$ x' = \\frac{(x - \\mu_{ref})}{\\sigma_{ref}} \\cdot \\sigma_{target} + \\mu_{target} $$
        Then blended with the original by `strength`:
            $$ out = (1 - s) \\cdot x + s \\cdot x' $$
        """
        per_channel = target_stats.get("per_channel", [])
        if not per_channel:
            log.warning("Statistical_Match: no per_channel stats; returning reference.")
            return ref

        C = ref.shape[1]
        if len(per_channel) != C:
            log.warning(
                f"Statistical_Match: waypoint has {len(per_channel)} channels "
                f"but reference has {C}. Matching min(C_ref, C_wp) channels."
            )

        result = ref.clone().float()
        n_channels = min(C, len(per_channel))

        for c in range(n_channels):
            ch_stats = per_channel[c]
            tgt_mean = float(ch_stats.get("mean", 0.0))
            tgt_std  = float(ch_stats.get("std",  1.0))

            ch = result[:, c:c+1, :, :]
            ref_mean = ch.mean()
            ref_std  = ch.std().clamp(min=eps)

            normalized = (ch - ref_mean) / ref_std
            rescaled   = normalized * tgt_std + tgt_mean
            result[:, c:c+1, :, :] = torch.lerp(ch, rescaled, strength)

        return result.to(ref.dtype)

    def load(
        self,
        waypoint_file: str,
        reference_latent: Dict[str, Any],
        mode: str,
        strength: float,
        enable_perf_logging: bool,
        manual_path: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:

        if enable_perf_logging:
            log.setLevel(logging.INFO)
        t_start = time.perf_counter()

        path = self._resolve_path(waypoint_file, manual_path)
        payload = self._load_payload(path)

        with torch.no_grad():
            ref_samples = self._extract_samples(reference_latent, "reference_latent")

            if mode == "Reference_Only":
                out_tensor = ref_samples.clone()
            elif mode == "Statistical_Match":
                out_tensor = self._statistical_rescale(
                    ref_samples, payload["statistics"], strength
                )
            elif mode == "Coord_Steering":
                # Returns ref unchanged; downstream Compass uses meta for direction
                out_tensor = ref_samples.clone()
            else:
                raise ValueError(f"Unknown mode: {mode}")

        out_latent = {k: v for k, v in reference_latent.items() if k != "samples"}
        out_latent["samples"] = out_tensor

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        waypoint_meta: Dict[str, Any] = {
            **payload,
            "load_mode":           mode,
            "load_strength":       strength,
            "load_source_path":    str(path),
            "reconstructed_shape": list(out_tensor.shape),
            "load_elapsed_ms":     round(elapsed_ms, 2),
        }

        report_lines = [
            "=== GimbalGPS_Load ===",
            f"Source:        {path.name}",
            f"Mode:          {mode}",
            f"Strength:      {strength:.3f}",
            f"Waypoint:      {payload.get('waypoint_name', '?')}",
            f"Original idx:  {payload.get('select_index', '?')} of {payload.get('batch_size', '?')}",
            f"Abs position:  {payload.get('absolute_position', {})}",
            f"Output shape:  {list(out_tensor.shape)}",
        ]
        if enable_perf_logging:
            report_lines.append(f"Time:          {elapsed_ms:.1f} ms")

        if mode == "Statistical_Match":
            g = payload["statistics"].get("global", {})
            report_lines.append(
                f"Target stats:  mean={g.get('mean')}, std={g.get('std')}"
            )

        return (out_latent, waypoint_meta, "\n".join(report_lines))


# Backward compatibility alias
WayfinderGPS_Load = GimbalGPS_Load

NODE_CLASS_MAPPINGS = {
    "GimbalGPS_Load": GimbalGPS_Load,
    "WayfinderGPS_Load": WayfinderGPS_Load,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GimbalGPS_Load": "📥 Gimbal GPS Load (Recall)",
    "WayfinderGPS_Load": "📥 Gimbal GPS Load (Recall)",
}
