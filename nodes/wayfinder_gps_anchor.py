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


def _resolve_output_dir() -> Path:
    env_override = os.environ.get("WAYFINDER_OUTPUT_DIR", "").strip()
    if env_override:
        return Path(env_override)
    try:
        import folder_paths
        root = Path(folder_paths.get_output_directory())
        return root / "wayfinder"
    except Exception as exc:
        log.error(f"folder_paths unavailable, using ./output/wayfinder. Error: {exc}")
        return Path("output") / "wayfinder"


def _get_int_env(key: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        log.warning(f"{key}={raw!r} invalid, using default={default}.")
        return default


WAYFINDER_DIR       = _resolve_output_dir()
MAX_FILENAME_LENGTH = _get_int_env("WAYFINDER_MAX_FILENAME_LENGTH", 64)
STAT_PRECISION      = _get_int_env("WAYFINDER_STAT_PRECISION", 6)


class WayfinderGPS_Anchor:
    """
    [VibeCheck Badge: 🟢 Stabilized]
    
    Extracts a single latent from a batch by index, computes navigational
    metadata, and optionally persists waypoints to disk.
    
    What's left:
    - Fine-tune parameter ranges for edge cases.
    """

    CATEGORY     = "Wayfinder/Latent"
    RETURN_TYPES = ("LATENT", "DICT", "STRING")
    RETURN_NAMES = ("anchored_latent", "waypoint_meta", "waypoint_report")
    FUNCTION     = "anchor"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent_batch": ("LATENT",),
                "select_index": ("INT", {"default": 0, "min": 0, "max": 4095, "step": 1}),
                "save_waypoint":       ("BOOLEAN", {"default": False}),
                "waypoint_name":       ("STRING",  {"default": "waypoint_01"}),
                "enable_perf_logging": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "wayfinder_meta": ("DICT",),
            },
        }

    @staticmethod
    def _extract_samples(d: Dict[str, Any], name: str) -> torch.Tensor:
        s = d.get("samples")
        if s is None:
            raise ValueError(f"WayfinderGPS_Anchor: '{name}' missing 'samples' key.")
        if s.ndim != 4:
            raise ValueError(f"WayfinderGPS_Anchor: '{name}.samples' must be 4-D [B, C, H, W], got {list(s.shape)}.")
        return s

    @staticmethod
    def _sanitize_filename(name: str, max_length: int = MAX_FILENAME_LENGTH) -> str:
        name = name.strip()
        name = re.sub(r"\s+", "_", name)
        # Remove path traversal and illegal chars
        name = re.sub(r"[^\w.\-]", "", name)
        name = name[:max_length]
        return name or "waypoint"

    @staticmethod
    def _versioned_path(base: Path) -> Path:
        if not base.exists():
            return base
        stem, suffix = base.stem, base.suffix
        parent = base.parent
        v = 2
        while True:
            candidate = parent / f"{stem}_v{v}{suffix}"
            if not candidate.exists():
                return candidate
            v += 1

    @staticmethod
    def _tensor_stats(t: torch.Tensor, precision: int = STAT_PRECISION) -> Dict[str, Any]:
        """
        Compute statistics on CPU to avoid multiple device synchronizations.
        Single .cpu() call is faster than 25+ .item() calls on CUDA.
        """
        t_cpu = t.detach().cpu().float()
        C = t_cpu.shape[0]
        
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

    @staticmethod
    def _resolve_grid_coordinate(meta: Dict[str, Any], select_index: int) -> Optional[Dict[str, Any]]:
        grid_map = meta.get("wayfinder_grid_map")
        if not isinstance(grid_map, list):
            return None
        for cell in grid_map:
            start = cell.get("batch_start", -1)
            end   = cell.get("batch_end",   -1)
            if start <= select_index <= end:
                return cell
        return None

    @staticmethod
    def _compute_absolute_position(
        meta: Dict[str, Any], grid_cell: Optional[Dict[str, Any]], x_strength: float, y_strength: float
    ) -> Dict[str, Any]:
        prior_raw = meta.get("accumulated_position", {"x": 0.0, "y": 0.0}) if isinstance(meta, dict) else {"x": 0.0, "y": 0.0}
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
        
        cell_x = (grid_cell.get("offset_x", 0.0) * x_strength) if grid_cell else 0.0
        cell_y = (grid_cell.get("offset_y", 0.0) * y_strength) if grid_cell else 0.0

        return {
            "x": round(prior.get("x", 0.0) + cell_x, 8),
            "y": round(prior.get("y", 0.0) + cell_y, 8),
        }

    def anchor(
        self,
        latent_batch: Dict[str, Any],
        select_index: int,
        save_waypoint: bool,
        waypoint_name: str,
        enable_perf_logging: bool,
        wayfinder_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:

        if enable_perf_logging:
            log.setLevel(logging.INFO)

        t_total = time.perf_counter()
        safe_name = self._sanitize_filename(waypoint_name)

        # All tensor operations under no_grad
        with torch.no_grad():
            samples    = self._extract_samples(latent_batch, "latent_batch")
            batch_size = samples.shape[0]

            if select_index >= batch_size:
                raise ValueError(f"select_index ({select_index}) out of range [0, {batch_size-1}].")

            # Clone to sever view relationship and prevent upstream corruption
            selected = samples[select_index : select_index + 1].clone()

        # Statistics (CPU-bound)
        stats = self._tensor_stats(selected[0])

        # Metadata resolution
        meta         = wayfinder_meta or {}
        x_strength   = float(meta.get("x_strength", 1.0))
        y_strength   = float(meta.get("y_strength", 1.0))
        grid_cell    = self._resolve_grid_coordinate(meta, select_index)
        absolute_pos = self._compute_absolute_position(meta, grid_cell, x_strength, y_strength)

        payload: Dict[str, Any] = {
            "waypoint_name":        safe_name,
            "select_index":         select_index,
            "batch_size":           batch_size,
            "latent_shape":         list(selected.shape),
            "statistics":           stats,
            "grid_cell":            grid_cell,
            "absolute_position":    absolute_pos,
            "accumulated_position": absolute_pos,
            "interpolation_mode":   meta.get("interpolation_mode"),
            "normalize_vectors":    meta.get("normalize_vectors"),
            "upstream_grid_size":   [meta.get("grid_size_x"), meta.get("grid_size_y")],
            "upstream_output_shape": meta.get("output_shape"),
        }

        # Atomic file I/O
        save_path: Optional[Path] = None
        save_error: Optional[str] = None

        if save_waypoint:
            try:
                WAYFINDER_DIR.mkdir(parents=True, exist_ok=True)
                target   = WAYFINDER_DIR / f"{safe_name}.json"
                out_path = self._versioned_path(target)
                
                # 'x' mode = exclusive creation (atomic, race-safe)
                with out_path.open("x", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, ensure_ascii=False)
                save_path = out_path
                if enable_perf_logging:
                    log.info(f"[WayfinderGPS_Anchor] saved: {out_path}")
            except FileExistsError as exc:
                save_error = str(exc)
                warnings.warn(f"Race condition on file write: {exc}", UserWarning)
            except (PermissionError, OSError) as exc:
                save_error = str(exc)
                warnings.warn(f"File I/O error: {exc}", UserWarning)
            except Exception as exc:
                save_error = str(exc)
                warnings.warn(f"Unexpected save error: {exc}", UserWarning)

        # Reconstruct LATENT dict preserving extra keys (noise_mask, etc.)
        out_latent = {k: v for k, v in latent_batch.items() if k != "samples"}
        out_latent["samples"] = selected

        elapsed_ms = (time.perf_counter() - t_total) * 1000
        
        waypoint_meta = {
            **payload,
            "save_path":  str(save_path) if save_path else None,
            "save_error": save_error,
            "save_waypoint": save_waypoint,
            "elapsed_ms": round(elapsed_ms, 2),
        }

        report_lines = [
            "=== WayfinderGPS_Anchor ===",
            f"Waypoint:      {safe_name}",
            f"Selected:      index {select_index} of {batch_size}",
            f"Latent shape:  {list(selected.shape)}",
            "",
            "-- Global Statistics --",
            f"  mean:        {stats['global']['mean']}",
            f"  std:         {stats['global']['std']}",
            f"  min/max:     {stats['global']['min']} / {stats['global']['max']}",
            "",
            "-- Navigation --",
            f"  is_center:   {grid_cell.get('is_center') if grid_cell else 'N/A'}",
            f"  abs pos:     ({absolute_pos['x']}, {absolute_pos['y']})",
            "",
            f"Saved:         {save_path if save_path else ('FAILED: ' + save_error if save_error else 'disabled')}",
        ]
        if enable_perf_logging:
            report_lines.append(f"Time:          {elapsed_ms:.1f} ms")

        return (out_latent, waypoint_meta, "\n".join(report_lines))


NODE_CLASS_MAPPINGS = {
    "WayfinderGPS_Anchor": WayfinderGPS_Anchor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WayfinderGPS_Anchor": "Wayfinder GPS Anchor",
}