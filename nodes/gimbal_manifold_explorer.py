import torch
import torch.nn.functional as F
import logging
import warnings
import time
from typing import Dict, Any, Tuple, List, Optional
try:
    from .gimbal_slerp import slerp_mu_centered, slerp_origin_centered, compute_batch_centroid
except ImportError:
    from gimbal_slerp import slerp_mu_centered, slerp_origin_centered, compute_batch_centroid

log = logging.getLogger("GimbalManifold_Explorer")


class GimbalManifold_Explorer:
    """
    [VibeCheck Badge: 🟢 Stabilized]
    
    Generates a 2D batch of latents by interpolating a center latent
    along two independent directional vectors (X and Y axes).
    """

    CATEGORY     = "Gimbal/Flight Instruments"
    RETURN_TYPES = ("LATENT", "DICT", "STRING")
    RETURN_NAMES = ("latent_batch", "grid_meta", "grid_report")
    FUNCTION     = "explore"

    MAX_GRID_CELLS = 256

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "center_latent": ("LATENT",),
                "x_vector":      ("LATENT",),
                "y_vector":      ("LATENT",),
                "grid_size_x": ("INT", {"default": 3, "min": 1, "max": 16, "step": 1}),
                "grid_size_y": ("INT", {"default": 3, "min": 1, "max": 16, "step": 1}),
                "x_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "slider",
                }),
                "y_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "slider",
                }),
                "interpolation_mode": (["Linear", "Slerp", "Slerp_Origin"], {"default": "Linear"}),
                "normalize_vectors":   ("BOOLEAN", {"default": True}),
                "clamp_output":        ("BOOLEAN", {"default": False}),
                "clamp_min": ("FLOAT", {"default": -10.0, "min": -100.0, "max": 0.0, "step": 0.5}),
                "clamp_max": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "enable_perf_logging": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mu_override": ("LATENT", {"tooltip": "Population centroid for μ-centered Slerp. If absent, center_latent is used as the anchor."}),
            },
        }

    @staticmethod
    def _extract_samples(d: Dict[str, Any], name: str) -> torch.Tensor:
        s = d.get("samples")
        if s is None:
            raise ValueError(f"GimbalManifold_Explorer: '{name}' missing 'samples' key.")
        if s.ndim != 4:
            raise ValueError(f"GimbalManifold_Explorer: '{name}.samples' must be 4-D [B, C, H, W], got {list(s.shape)}.")
        return s

    @staticmethod
    def _check_channel_compat(center: torch.Tensor, other: torch.Tensor, name: str) -> None:
        if center.shape[1] != other.shape[1]:
            raise ValueError(
                f"Channel mismatch: center_latent (C={center.shape[1]}) vs '{name}' (C={other.shape[1]})."
            )

    @staticmethod
    def _resize_to(tensor: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.shape[-2:] == target.shape[-2:]:
            return tensor
        src_hw = list(tensor.shape[-2:])
        tgt_hw = list(target.shape[-2:])
        ratio  = max(src_hw[0] / tgt_hw[0], src_hw[1] / tgt_hw[1])
        if ratio > 4.0 or ratio < 0.25:
            warnings.warn(
                f"GimbalManifold_Explorer: large spatial resize on '{name}' ({src_hw} -> {tgt_hw}).",
                UserWarning,
                stacklevel=4,
            )
        # Always interpolate in float32 for precision, then cast back
        return F.interpolate(
            tensor.float(),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(tensor.dtype)

    @staticmethod
    def _unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        import math
        B    = v.shape[0]
        flat = v.reshape(B, -1)
        # Use float32 for norm calculation to avoid overflow/underflow
        flat_f = flat.float()
        norm   = flat_f.norm(dim=1, keepdim=True)
        if (norm < eps).any():
            warnings.warn(
                "GimbalManifold_Explorer: zero-norm vector detected and cannot be normalized.",
                UserWarning,
                stacklevel=4,
            )
        norm = norm.clamp(min=eps)
        D = flat_f.shape[1]
        return ((flat_f / norm) * math.sqrt(D)).reshape(v.shape).to(v.dtype)

    @staticmethod
    def _grid_offsets(size: int) -> List[float]:
        half = (size - 1) / 2.0
        return [round(i - half, 8) for i in range(size)]

    @staticmethod
    def _collapse_vector_batch(v: torch.Tensor, name: str) -> torch.Tensor:
        if v.shape[0] == 1:
            return v
        warnings.warn(
            f"GimbalManifold_Explorer: '{name}' has batch size {v.shape[0]}, averaging to single vector.",
            UserWarning,
            stacklevel=3,
        )
        return v.mean(dim=0, keepdim=True)

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
        return torch.lerp(a, b, t)

    @staticmethod
    def _slerp(a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
        """Legacy origin-centered Slerp. Delegates to shared module."""
        return slerp_origin_centered(a, b, t, eps)

    def _interpolate(
        self,
        center: torch.Tensor,
        x_vec: torch.Tensor,
        y_vec: torch.Tensor,
        ox: float,
        oy: float,
        x_strength: float,
        y_strength: float,
        mode: str,
        mu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply displacement to center using specified interpolation mode.

        For Slerp modes, the trajectory follows the high-probability Typical Set
        shell (Gaussian Annulus Theorem) instead of cutting through the hollow
        center of the latent hypersphere.

        Args:
            center:     Center latent tensor [B, C, H, W].
            x_vec:      X-axis direction vector [1, C, H, W].
            y_vec:      Y-axis direction vector [1, C, H, W].
            ox, oy:     Grid offset coordinates.
            x_strength: X-axis strength multiplier.
            y_strength: Y-axis strength multiplier.
            mode:       "Linear", "Slerp" (μ-centered), or "Slerp_Origin".
            mu:         Population centroid for μ-centered Slerp [1, C, H, W].
                        Defaults to center if None.
        """
        if mode == "Linear":
            x_disp = x_vec * (x_strength * ox)
            y_disp = y_vec * (y_strength * oy)
            return center + x_disp + y_disp
        elif mode in ("Slerp", "Slerp_Origin"):
            import math

            # Select the appropriate Slerp function
            if mode == "Slerp":
                # μ-centered: anchor on the population centroid
                anchor = mu if mu is not None else compute_batch_centroid(center)
                def _do_slerp(a, b, t):
                    return slerp_mu_centered(a, b, t, anchor)
            else:
                # Origin-centered (legacy)
                def _do_slerp(a, b, t):
                    return slerp_origin_centered(a, b, t)

            # X axis Slerp
            if ox == 0.0:
                dx = torch.zeros_like(center)
            else:
                sign_x = math.copysign(1.0, ox)
                target_x = center + x_vec * sign_x
                t_x = abs(ox) * x_strength
                slerp_x = _do_slerp(center, target_x, t_x)
                dx = slerp_x - center

            # Y axis Slerp
            if oy == 0.0:
                dy = torch.zeros_like(center)
            else:
                sign_y = math.copysign(1.0, oy)
                target_y = center + y_vec * sign_y
                t_y = abs(oy) * y_strength
                slerp_y = _do_slerp(center, target_y, t_y)
                dy = slerp_y - center

            return center + dx + dy
        else:
            raise ValueError(f"Unknown interpolation_mode '{mode}'.")

    def _build_report(
        self,
        grid_size_x: int,
        grid_size_y: int,
        x_offsets: List[float],
        y_offsets: List[float],
        x_strength: float,
        y_strength: float,
        mode: str,
        normalize: bool,
        center_batch: int,
        output_batch: int,
        has_center: bool,
        clamp_output: bool,
        clamp_min: float,
        clamp_max: float,
        elapsed_ms: float,
        no_grad: bool,
    ) -> str:
        lines = [
            "=== GimbalManifold_Explorer ===",
            f"Grid:          {grid_size_x} x {grid_size_y}  ({grid_size_x * grid_size_y} cells)",
            f"Center batch:  {center_batch}  ->  Output batch: {output_batch}",
            f"X offsets:     {[round(o, 3) for o in x_offsets]}  (strength {x_strength:+.3f})",
            f"Y offsets:     {[round(o, 3) for o in y_offsets]}  (strength {y_strength:+.3f})",
            f"Mode:          {mode}",
            f"Normalize vec: {normalize}",
            f"Exact center:  {has_center}",
            f"Clamp output:  {clamp_output}" + (f"  [{clamp_min}, {clamp_max}]" if clamp_output else ""),
            f"torch.no_grad: {no_grad}",
        ]
        if elapsed_ms:
            lines.append(f"Time:          {elapsed_ms:.1f} ms")

        lines.append("\nGrid map (col x, row y)  * = center:")
        for oy in y_offsets:
            row_str = "  "
            for ox in x_offsets:
                is_c = (ox == 0.0 and oy == 0.0)
                row_str += f"[{'*' if is_c else ' '}{ox:+.1f},{oy:+.1f}] "
            lines.append(row_str)
        return "\n".join(lines)

    def explore(
        self,
        center_latent: Dict[str, Any],
        x_vector: Dict[str, Any],
        y_vector: Dict[str, Any],
        grid_size_x: int,
        grid_size_y: int,
        x_strength: float,
        y_strength: float,
        interpolation_mode: str,
        normalize_vectors: bool,
        clamp_output: bool,
        clamp_min: float,
        clamp_max: float,
        enable_perf_logging: bool,
        mu_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:

        if enable_perf_logging:
            log.setLevel(logging.INFO)

        t_total = time.perf_counter()

        if clamp_output and clamp_min >= clamp_max:
            raise ValueError(f"clamp_min ({clamp_min}) must be < clamp_max ({clamp_max}).")

        total_cells = grid_size_x * grid_size_y
        if total_cells > self.MAX_GRID_CELLS:
            raise ValueError(f"Grid size {total_cells} exceeds cap of {self.MAX_GRID_CELLS}.")

        # All tensor work under no_grad for inference safety
        no_grad_active = True
        with torch.no_grad():
            center = self._extract_samples(center_latent, "center_latent")
            x_vec  = self._extract_samples(x_vector, "x_vector")
            y_vec  = self._extract_samples(y_vector, "y_vector")

            # Device alignment
            device = center.device
            dtype  = center.dtype
            x_vec  = x_vec.to(device=device, dtype=dtype)
            y_vec  = y_vec.to(device=device, dtype=dtype)

            # Channel compatibility
            self._check_channel_compat(center, x_vec, "x_vector")
            self._check_channel_compat(center, y_vec, "y_vector")

            # Spatial resize with dtype preservation
            x_vec = self._resize_to(x_vec, center, "x_vector")
            y_vec = self._resize_to(y_vec, center, "y_vector")

            # Collapse vector batches
            x_vec = self._collapse_vector_batch(x_vec, "x_vector")
            y_vec = self._collapse_vector_batch(y_vec, "y_vector")
            center_col = self._collapse_vector_batch(center, "center_latent")

            # Resolve μ centroid for Slerp modes
            if mu_override is not None:
                mu_s = self._extract_samples(mu_override, "mu_override")
                mu_s = mu_s.to(device=device, dtype=dtype)
                mu_s = self._resize_to(mu_s, center, "mu_override")
                mu_centroid = compute_batch_centroid(mu_s)
                slerp_anchor_source = "mu_override"
            else:
                # Default: empirical channel/batch centroid of center_latent
                mu_centroid = compute_batch_centroid(center)
                slerp_anchor_source = "center_distribution"

            # Convert absolute target positions into directional vectors relative to center
            x_vec = x_vec - center_col
            y_vec = y_vec - center_col

            # Optional normalization
            if normalize_vectors:
                x_vec = self._unit(x_vec)
                y_vec = self._unit(y_vec)

            # Warnings for zero strength
            if x_strength == 0.0:
                warnings.warn("x_strength is 0.0: no X-axis variation.", UserWarning)
            if y_strength == 0.0:
                warnings.warn("y_strength is 0.0: no Y-axis variation.", UserWarning)

            x_offsets    = self._grid_offsets(grid_size_x)
            y_offsets    = self._grid_offsets(grid_size_y)
            center_batch = center.shape[0]
            has_center   = (grid_size_x % 2 == 1 and grid_size_y % 2 == 1)

            cells: List[torch.Tensor] = []
            grid_map: List[Dict[str, Any]] = []
            batch_cursor = 0

            t_grid = time.perf_counter()

            # Pre-allocate displacement buffers to avoid repeated allocations
            for row_idx, oy in enumerate(y_offsets):
                for col_idx, ox in enumerate(x_offsets):
                    # Calculate displacement: strength * offset * unit_vector
                    # Interpolate from center to cell using improved 2D math
                    cell = self._interpolate(
                        center.float(),
                        x_vec.float(),
                        y_vec.float(),
                        ox, oy,
                        x_strength, y_strength,
                        mode=interpolation_mode,
                        mu=mu_centroid.float() if mu_centroid is not None else None,
                    ).to(dtype)

                    if clamp_output:
                        cell = cell.clamp(clamp_min, clamp_max)

                    cells.append(cell)
                    grid_map.append({
                        "batch_start": batch_cursor,
                        "batch_end":   batch_cursor + center_batch - 1,
                        "grid_col":    col_idx,
                        "grid_row":    row_idx,
                        "offset_x":    ox,
                        "offset_y":    oy,
                        "is_center":   (ox == 0.0 and oy == 0.0),
                        "x_disp_norm": float((x_vec * (x_strength * ox)).norm().item()),
                        "y_disp_norm": float((y_vec * (y_strength * oy)).norm().item()),
                    })
                    batch_cursor += center_batch

            if enable_perf_logging:
                log.info(f"[GimbalManifold_Explorer] grid generation: {(time.perf_counter() - t_grid)*1000:.2f} ms")

            # Concatenate all cells along batch dimension
            batch = torch.cat(cells, dim=0)

        # Assemble output latent dict
        out_latent = {k: v for k, v in center_latent.items() if k != "samples"}
        out_latent["samples"] = batch
        
        # CRITICAL: Expand noise_mask to match output batch size
        if "noise_mask" in center_latent:
            mask = center_latent["noise_mask"]
            if mask.shape[0] == 1 and batch.shape[0] > 1:
                out_latent["noise_mask"] = mask.expand(batch.shape[0], *mask.shape[1:])
            elif mask.shape[0] == center_batch and batch.shape[0] > center_batch:
                # If center had its own batch, repeat it to match the grid expansion
                repeats = total_cells
                out_latent["noise_mask"] = mask.repeat(repeats, *([1] * (mask.ndim - 1)))

        elapsed_ms = (time.perf_counter() - t_total) * 1000
        if enable_perf_logging:
            log.info(f"[GimbalManifold_Explorer] total: {elapsed_ms:.2f} ms")

        meta: Dict[str, Any] = {
            "grid_size_x": grid_size_x,
            "grid_size_y": grid_size_y,
            "total_cells": total_cells,
            "center_batch_size": center_batch,
            "output_batch_size": batch.shape[0],
            "x_strength": x_strength,
            "y_strength": y_strength,
            "interpolation_mode": interpolation_mode,
            "slerp_anchor": slerp_anchor_source if interpolation_mode.startswith("Slerp") else None,
            "normalize_vectors": normalize_vectors,
            "clamp_output": clamp_output,
            "clamp_range": [clamp_min, clamp_max] if clamp_output else None,
            "x_offsets": x_offsets,
            "y_offsets": y_offsets,
            "has_exact_center": has_center,
            "output_shape": list(batch.shape),
            "device": str(device),
            "elapsed_ms": round(elapsed_ms, 2),
            "no_grad_active": no_grad_active,
            "gimbal_grid_map": grid_map,
            "wayfinder_grid_map": grid_map,
        }

        report = self._build_report(
            grid_size_x, grid_size_y, x_offsets, y_offsets,
            x_strength, y_strength, interpolation_mode, normalize_vectors,
            center_batch, batch.shape[0], has_center,
            clamp_output, clamp_min, clamp_max,
            elapsed_ms if enable_perf_logging else 0.0,
            no_grad_active,
        )

        return (out_latent, meta, report)


# Backward compatibility alias
WayfinderManifold_Explorer = GimbalManifold_Explorer

NODE_CLASS_MAPPINGS = {
    "GimbalManifold_Explorer": GimbalManifold_Explorer,
    "WayfinderManifold_Explorer": WayfinderManifold_Explorer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GimbalManifold_Explorer": "🗺️ Gimbal Manifold Explorer",
    "WayfinderManifold_Explorer": "🗺️ Gimbal Manifold Explorer",
}