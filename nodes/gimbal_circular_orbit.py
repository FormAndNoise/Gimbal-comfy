import math
import time
import torch
import torch.nn.functional as F
import warnings
from typing import Dict, Any, Tuple, Optional

class GimbalCircularOrbit:
    """
    [Form & Noise Atelier — Gimbal Node Suite]
    
    Generates a closed-loop circular or harmonic trajectory through latent space.
    Ideal for seamless looping animations and continuous orbital auditing.
    
    Traverses: z(theta) = center + R * (cos(theta) * u + sin(theta) * v)
    where u and v form an orthonormal basis.
    """

    CATEGORY = "Gimbal/Trajectory"
    RETURN_TYPES = ("LATENT", "DICT")
    RETURN_NAMES = ("latent_batch", "flight_telemetry")
    FUNCTION = "generate_orbit"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "center_latent": ("LATENT",),
                "steps": ("INT", {"default": 36, "min": 3, "max": 1024, "step": 1}),
                "radius": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01, "display": "slider"}),
                "orbit_mode": (["Orthogonal_Basis", "Phase_Modulated", "Harmonic_Torus"], {"default": "Orthogonal_Basis"}),
                "preserve_hypersphere_norm": ("BOOLEAN", {"default": True, "tooltip": "Maintains constant L2 radius to prevent density burn"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "direction_x": ("LATENT",),
                "direction_y": ("LATENT",),
                "mask": ("MASK",),
            }
        }

    @staticmethod
    def _extract_samples(d: Dict[str, Any], name: str) -> torch.Tensor:
        s = d.get("samples")
        if s is None:
            raise ValueError(f"GimbalCircularOrbit: '{name}' missing 'samples' tensor.")
        if s.ndim != 4:
            raise ValueError(f"GimbalCircularOrbit: '{name}' must be 4-D [B, C, H, W], got {list(s.shape)}.")
        return s

    @staticmethod
    def _gram_schmidt_orthonormal(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce two orthonormal unit vectors in R^D."""
        B, D = u.shape[0], u.numel() // u.shape[0]
        u_f = u.reshape(B, -1).float()
        v_f = v.reshape(B, -1).float()

        u_norm = u_f.norm(dim=1, keepdim=True).clamp(min=eps)
        u_hat = u_f / u_norm

        # Project v onto u_hat and subtract
        proj = (v_f * u_hat).sum(dim=1, keepdim=True) * u_hat
        v_ortho = v_f - proj
        v_norm = v_ortho.norm(dim=1, keepdim=True).clamp(min=eps)
        v_hat = v_ortho / v_norm

        return u_hat.reshape(u.shape).to(u.dtype), v_hat.reshape(v.shape).to(v.dtype)

    def generate_orbit(
        self,
        center_latent: Dict[str, Any],
        steps: int,
        radius: float,
        orbit_mode: str,
        preserve_hypersphere_norm: bool,
        seed: int,
        direction_x: Optional[Dict[str, Any]] = None,
        direction_y: Optional[Dict[str, Any]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        t_start = time.perf_counter()
        
        with torch.no_grad():
            center_s = self._extract_samples(center_latent, "center_latent")
            B, C, H, W = center_s.shape
            device = center_s.device
            dtype = center_s.dtype
            D = C * H * W

            # 1. Determine or synthesize basis directions u and v
            gen = torch.Generator(device="cpu").manual_seed(seed)
            if direction_x is not None:
                dx = self._extract_samples(direction_x, "direction_x")
                if dx.shape[-2:] != (H, W):
                    dx = F.interpolate(dx.float(), size=(H, W), mode="bilinear", align_corners=False).to(dtype)
            else:
                dx = torch.randn((1, C, H, W), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

            if direction_y is not None:
                dy = self._extract_samples(direction_y, "direction_y")
                if dy.shape[-2:] != (H, W):
                    dy = F.interpolate(dy.float(), size=(H, W), mode="bilinear", align_corners=False).to(dtype)
            else:
                dy = torch.randn((1, C, H, W), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

            # Orthonormalize basis
            u_hat, v_hat = self._gram_schmidt_orthonormal(dx, dy)

            # Target radius scaling (normalized to standard normal scale sqrt(D))
            norm_scale = math.sqrt(D) * radius
            center_f = center_s.float()
            center_norm = center_f.reshape(B, -1).norm(dim=1, keepdim=True).clamp(min=1e-8)

            # 2. Compute orbit trajectory points
            theta = torch.linspace(0, 2 * math.pi, steps + 1, device=device, dtype=torch.float32)[:-1]
            batch_samples = []

            for i, th in enumerate(theta):
                angle = th.item()
                if orbit_mode == "Orthogonal_Basis":
                    coeff_x = math.cos(angle)
                    coeff_y = math.sin(angle)
                elif orbit_mode == "Phase_Modulated":
                    coeff_x = math.cos(angle) * (1.0 + 0.2 * math.sin(3 * angle))
                    coeff_y = math.sin(angle) * (1.0 + 0.2 * math.cos(2 * angle))
                elif orbit_mode == "Harmonic_Torus":
                    coeff_x = math.cos(angle) + 0.3 * math.cos(3 * angle)
                    coeff_y = math.sin(angle) + 0.3 * math.sin(2 * angle)
                else:
                    coeff_x = math.cos(angle)
                    coeff_y = math.sin(angle)

                delta = (u_hat.float() * coeff_x + v_hat.float() * coeff_y) * norm_scale
                sample_t = center_f + delta

                if preserve_hypersphere_norm:
                    # Rescale to maintain initial center radius
                    sample_flat = sample_t.reshape(B, -1)
                    s_norm = sample_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    sample_t = ((sample_flat / s_norm) * center_norm).reshape(sample_t.shape)

                # Mask blending if mask is provided
                if mask is not None:
                    m = mask.unsqueeze(1).float() if mask.ndim == 3 else mask.unsqueeze(0).unsqueeze(1).float()
                    if m.shape[-2:] != (H, W):
                        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
                    sample_t = center_f * (1.0 - m) + sample_t * m

                batch_samples.append(sample_t.to(dtype))

            out_samples = torch.cat(batch_samples, dim=0)

            telemetry = {
                "instrument": "GimbalCircularOrbit",
                "orbit_mode": orbit_mode,
                "total_steps": steps,
                "radius": radius,
                "center_shape": list(center_s.shape),
                "output_batch_shape": list(out_samples.shape),
                "hypersphere_norm_preserved": preserve_hypersphere_norm,
                "execution_time_ms": round((time.perf_counter() - t_start) * 1000, 3),
            }

            out_latent = center_latent.copy()
            out_latent["samples"] = out_samples

            return (out_latent, telemetry)
