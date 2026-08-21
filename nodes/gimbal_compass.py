import torch
import torch.nn.functional as F
import logging
import time
import warnings
from typing import Optional, Tuple, Dict, Any
try:
    from .gimbal_slerp import slerp_mu_centered, slerp_origin_centered, compute_batch_centroid
except ImportError:
    from gimbal_slerp import slerp_mu_centered, slerp_origin_centered, compute_batch_centroid

log = logging.getLogger("GimbalCompass_Pro")


class GimbalCompass_Pro:
    """
    [VibeCheck Badge: 🟢 Stabilized]
    
    Latent space vector arithmetic with Standard, Normalized, and Orthogonal modes.
    Supports masking and batch broadcasting with full dtype/device safety.
    """

    CATEGORY = "Gimbal/Flight Instruments"
    RETURN_TYPES = ("LATENT", "DICT")
    RETURN_NAMES = ("latent_out", "gimbal_meta")
    FUNCTION = "navigate"

    CLAMP_MIN_DEFAULT = -10.0
    CLAMP_MAX_DEFAULT =  10.0

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "base_latent":   ("LATENT",),
                "target_latent": ("LATENT",),
                "origin_latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "slider"}),
                "mode": (["Standard", "Normalized", "Orthogonal_Projection", "Slerp", "Slerp_Origin", "Blend_Overlay", "Blend_Multiply", "Stochastic_Sample"], {"default": "Standard"}),
                "clamp_output":        ("BOOLEAN", {"default": False}),
                "clamp_min": ("FLOAT", {"default": cls.CLAMP_MIN_DEFAULT, "min": -100.0, "max": 0.0, "step": 0.5}),
                "clamp_max": ("FLOAT", {"default": cls.CLAMP_MAX_DEFAULT, "min": 0.0, "max": 100.0, "step": 0.5}),
                "allow_batch_expand":  ("BOOLEAN", {"default": False}),
                "ortho_per_channel":   ("BOOLEAN", {"default": False}),
                "clamp_mask_input":    ("BOOLEAN", {"default": False}),
                "enable_perf_logging": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mu_centroid": ("LATENT", {"tooltip": "Optional population centroid for μ-centered Slerp. If absent, origin_latent is used as the anchor."}),
            },
        }

    @staticmethod
    def _extract_samples(d: Dict[str, Any], name: str) -> torch.Tensor:
        s = d.get("samples")
        if s is None:
            raise ValueError(f"GimbalCompass_Pro: '{name}' missing 'samples' key.")
        if s.ndim != 4:
            raise ValueError(f"GimbalCompass_Pro: '{name}.samples' must be 4-D [B, C, H, W].")
        return s

    @staticmethod
    def _resize_to_base(tensor: torch.Tensor, base: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.shape[-2:] == base.shape[-2:]:
            return tensor
        src_hw, tgt_hw = tensor.shape[-2:], base.shape[-2:]
        ratio = max(src_hw[0] / tgt_hw[0], src_hw[1] / tgt_hw[1])
        if ratio > 4.0 or ratio < 0.25:
            warnings.warn(f"Large resize on '{name}': {src_hw} -> {tgt_hw}", UserWarning, stacklevel=4)
        
        return F.interpolate(
            tensor.float(),
            size=base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(tensor.dtype)

    @staticmethod
    def _resize_mask(mask: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        # MASK in ComfyUI is [B, H, W] or [H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(1).float()  # [B, 1, H, W]
        if mask.shape[-2:] != base.shape[-2:]:
            mask = F.interpolate(mask, size=base.shape[-2:], mode="bilinear", align_corners=False)
        return mask  # [B, 1, H, W]

    @staticmethod
    def _safe_broadcast(base: torch.Tensor, other: torch.Tensor, name: str, allow_expand: bool) -> torch.Tensor:
        B_b, B_o = base.shape[0], other.shape[0]
        
        if B_b == B_o:
            return other
        if B_o == 1:
            return other.expand(B_b, *other.shape[1:])
        if B_b == 1:
            return other  # Base will be expanded by caller
            
        if not allow_expand:
            raise ValueError(f"Batch mismatch: base={B_b}, '{name}'={B_o}. Enable allow_batch_expand or fix batches.")
        
        if B_o > B_b:
            return other  # Can't shrink, caller must expand base
            
        if B_b % B_o != 0:
            raise ValueError(f"Cannot evenly expand '{name}' batch {B_o} to {B_b}.")
            
        repeat = [B_b // B_o] + [1] * (other.ndim - 1)
        warnings.warn(f"Expanding '{name}' batch {B_o} -> {B_b} via repeat.", UserWarning, stacklevel=4)
        return other.repeat(*repeat)

    @staticmethod
    def _apply_standard(base: torch.Tensor, delta: torch.Tensor, strength: float) -> torch.Tensor:
        return base + delta * strength

    @staticmethod
    def _apply_normalized(base: torch.Tensor, delta: torch.Tensor, strength: float, eps: float = 1e-8) -> torch.Tensor:
        import math
        delta_f = delta.float()
        flat = delta_f.reshape(delta_f.shape[0], -1)
        norms = flat.norm(dim=1, keepdim=True).clamp(min=eps)
        scale = norms.reshape(delta_f.shape[0], *([1] * (delta_f.ndim - 1)))
        D = flat.shape[1]
        return base + ((delta_f / scale) * (strength * math.sqrt(D))).to(base.dtype)

    @staticmethod
    def _apply_orthogonal(
        base: torch.Tensor, delta: torch.Tensor, strength: float, per_channel: bool, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Project base onto delta direction.
        If per_channel: project each channel independently over spatial dims.
        Else: project full vector over C*H*W.
        """
        B = base.shape[0]
        
        if not per_channel:
            # Flatten to [B, D]
            base_flat = base.reshape(B, -1).float()
            delta_flat = delta.reshape(B, -1).float()
            
            delta_norm = delta_flat.norm(dim=1, keepdim=True).clamp(min=eps)
            delta_hat = delta_flat / delta_norm
            
            # Scalar projection: (base · delta_hat)
            dot = (base_flat * delta_hat).sum(dim=1, keepdim=True)
            projection = (dot * delta_hat).reshape(base.shape).to(base.dtype)
        else:
            C = base.shape[1]
            # [B, C, H*W]
            base_flat = base.reshape(B, C, -1).float()
            delta_flat = delta.reshape(B, C, -1).float()
            
            delta_norm = delta_flat.norm(dim=2, keepdim=True).clamp(min=eps)
            delta_hat = delta_flat / delta_norm
            
            dot = (base_flat * delta_hat).sum(dim=2, keepdim=True)  # [B, C, 1]
            projection = (dot * delta_hat).reshape(base.shape).to(base.dtype)
            
        return base + projection * strength

    @staticmethod
    def _apply_slerp(
        base: torch.Tensor, target: torch.Tensor, strength: float,
        mu: torch.Tensor, eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        μ-centered Slerp navigation: interpolates from base toward target
        along the high-probability Typical Set shell, anchored at centroid μ.
        """
        return slerp_mu_centered(base, target, strength, mu, eps)

    @staticmethod
    def _apply_slerp_origin(
        base: torch.Tensor, target: torch.Tensor, strength: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Origin-centered Slerp (legacy). Traverses great-circle arc centered
        at z=0. May cause variance collapse at midpoints in high dimensions.
        """
        return slerp_origin_centered(base, target, strength, eps)

    @staticmethod
    def _apply_overlay(base: torch.Tensor, target: torch.Tensor, strength: float) -> torch.Tensor:
        base_f = base.float()
        target_f = target.float()
        low = 2.0 * base_f * target_f
        high = 1.0 - 2.0 * (1.0 - base_f) * (1.0 - target_f)
        blend = torch.where(base_f < 0.5, low, high)
        return torch.lerp(base_f, blend, strength).to(base.dtype)

    @staticmethod
    def _apply_multiply(base: torch.Tensor, target: torch.Tensor, strength: float) -> torch.Tensor:
        base_f = base.float()
        target_f = target.float()
        blend = base_f * target_f
        return torch.lerp(base_f, blend, strength).to(base.dtype)

    @staticmethod
    def _apply_stochastic_sample(base: torch.Tensor, target: torch.Tensor, strength: float, seed: int) -> torch.Tensor:
        import copy
        rng = torch.Generator(device=base.device)
        rng.manual_seed(seed)
        ratio = max(0.0, min(1.0, strength)) # stochastic sample ratio must be 0-1
        mask = torch.rand(base.shape, generator=rng, device=base.device, dtype=base.dtype) >= ratio
        return torch.where(mask, base, target)

    def navigate(
        self,
        base_latent: Dict[str, Any],
        target_latent: Dict[str, Any],
        origin_latent: Dict[str, Any],
        strength: float,
        mode: str,
        clamp_output: bool,
        clamp_min: float,
        clamp_max: float,
        allow_batch_expand: bool,
        ortho_per_channel: bool,
        clamp_mask_input: bool,
        enable_perf_logging: bool,
        mask: Optional[torch.Tensor] = None,
        seed: int = 0,
        mu_centroid: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        if enable_perf_logging:
            log.setLevel(logging.INFO)

        t_start = time.perf_counter()
        
        if clamp_output and clamp_min >= clamp_max:
            raise ValueError(f"clamp_min ({clamp_min}) must be < clamp_max ({clamp_max}).")

        # CRITICAL: Wrap all tensor operations in no_grad for inference safety
        with torch.no_grad():
            base   = self._extract_samples(base_latent, "base_latent")
            target = self._extract_samples(target_latent, "target_latent")
            origin = self._extract_samples(origin_latent, "origin_latent")
            
            dtype   = base.dtype
            device  = base.device
            
            # Device & Dtype alignment
            target = target.to(device=device, dtype=dtype)
            origin = origin.to(device=device, dtype=dtype)
            
            # Spatial alignment
            target = self._resize_to_base(target, base, "target_latent")
            origin = self._resize_to_base(origin, base, "origin_latent")
            
            # Batch alignment
            target = self._safe_broadcast(base, target, "target_latent", allow_batch_expand)
            origin = self._safe_broadcast(base, origin, "origin_latent", allow_batch_expand)
            
            max_batch = max(base.shape[0], target.shape[0], origin.shape[0])
            if base.shape[0] == 1 and max_batch > 1:
                if not allow_batch_expand:
                    raise ValueError(
                        f"GimbalCompass_Pro: base batch size is 1 but target/origin "
                        f"have batch size {max_batch}. Enable 'allow_batch_expand' to "
                        f"permit automatic batch expansion, or ensure all inputs have "
                        f"matching batch sizes."
                    )
                warnings.warn(
                    f"GimbalCompass_Pro: expanding base batch 1 -> {max_batch}. "
                    f"Memory usage scales linearly with batch size.",
                    UserWarning,
                    stacklevel=2,
                )
                base = base.expand(max_batch, *base.shape[1:])
            
            # Resolve μ centroid for Slerp modes
            if mu_centroid is not None:
                mu_t = self._extract_samples(mu_centroid, "mu_centroid")
                mu_t = mu_t.to(device=device, dtype=dtype)
                mu_t = self._resize_to_base(mu_t, base, "mu_centroid")
                mu_t = self._safe_broadcast(base, mu_t, "mu_centroid", allow_batch_expand)
            else:
                mu_t = origin  # Default: use origin_latent as the population anchor

            # Vector math: delta = target - origin
            delta = target - origin
            
            # Mask application
            mask_applied = mask is not None
            if mask_applied:
                mask_t = mask.to(device=device)
                if clamp_mask_input:
                    mask_t = mask_t.clamp(0.0, 1.0)
                mask_t = self._resize_mask(mask_t, base)
                if mask_t.shape[0] == 1 and base.shape[0] > 1:
                    mask_t = mask_t.expand(base.shape[0], -1, -1, -1)
                delta = delta * mask_t
            
            # Mode dispatch
            if mode == "Standard":
                result = self._apply_standard(base, delta, strength)
            elif mode == "Normalized":
                result = self._apply_normalized(base, delta, strength)
            elif mode == "Orthogonal_Projection":
                result = self._apply_orthogonal(base, delta, strength, ortho_per_channel)
            elif mode == "Slerp":
                result = self._apply_slerp(base, base + delta, strength, mu_t)
            elif mode == "Slerp_Origin":
                result = self._apply_slerp_origin(base, base + delta, strength)
            elif mode == "Blend_Overlay":
                result = self._apply_overlay(base, target, strength)
            elif mode == "Blend_Multiply":
                result = self._apply_multiply(base, target, strength)
            elif mode == "Stochastic_Sample":
                result = self._apply_stochastic_sample(base, target, strength, seed)
            else:
                raise ValueError(f"Unknown mode '{mode}'")
            
            # Output conditioning
            if clamp_output:
                result = result.clamp(clamp_min, clamp_max)
                
            # Preserve input dtype explicitly
            result = result.to(dtype)
            
            # Assemble output latent
            out_latent = {k: v for k, v in base_latent.items() if k != "samples"}
            out_latent["samples"] = result
            
            elapsed = (time.perf_counter() - t_start) * 1000
            
            meta = {
                "mode": mode,
                "strength": strength,
                "clamp_output": clamp_output,
                "clamp_range": [clamp_min, clamp_max] if clamp_output else None,
                "mask_applied": mask_applied,
                "ortho_per_channel": ortho_per_channel if mode == "Orthogonal_Projection" else None,
                "slerp_anchor": "mu_centroid" if (mode.startswith("Slerp") and mu_centroid is not None) else ("origin_latent" if mode == "Slerp" else None),
                "base_shape": list(base.shape),
                "result_shape": list(result.shape),
                "device": str(device),
                "dtype": str(dtype),
                "elapsed_ms": round(elapsed, 2) if enable_perf_logging else None,
            }
            
            return out_latent, meta


# Backward compatibility alias
WayfinderCompass_Pro = GimbalCompass_Pro

NODE_CLASS_MAPPINGS = {
    "GimbalCompass_Pro": GimbalCompass_Pro,
    "WayfinderCompass_Pro": WayfinderCompass_Pro,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GimbalCompass_Pro": "🧭 Gimbal Compass Pro",
    "WayfinderCompass_Pro": "🧭 Gimbal Compass Pro",
}