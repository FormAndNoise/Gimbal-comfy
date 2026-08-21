"""
[Form & Noise Atelier - Gimbal Node Suite]

Full LAMNr latent quality-improvement stack as a single ComfyUI node.

Wraps the PyTorch primitives in `gimbal_latent_math.run_lamnr_pipeline` and
exposes the four stages (bounded coupling scale, dequantization jitter,
truncation toward the centroid, Woodbury low-rank conditional-mean denoise)
through four floating-point node inputs. No ComfyUI-specific imports beyond
the torch/LATENT contract; all math lives in `gimbal_latent_math`.

Stages, applied in evaluation order:
  E11 bounded_scale    : s_bounded = scale_cap * tanh(s_c / scale_cap)
  E12 dequantize       : z' = z + U(-1,1) * alpha   (skipped if jitter = 0)
  E3  truncation       : z' = mu + psi * (z - mu)
  E7  woodbury_impute  : z_hat = mu + U diag( l/(l+sigma^2) ) U^T (z - mu)

The node preserves the input latent dict, replaces only its `samples` tensor,
and returns per-sample telemetry alongside the transformed latent.
"""

import torch
from typing import Any, Dict, Tuple, Optional

try:
    from .gimbal_latent_math import run_lamnr_pipeline, channel_stats
except ImportError:  # direct top-level import (test harness)
    from gimbal_latent_math import run_lamnr_pipeline, channel_stats


class GimbalLatentStabilizer:
    """Full LAMNr quality-improvement pipeline (E3 + E7 + E11 + E12)."""

    CATEGORY = "Gimbal/Stabilizer"
    RETURN_TYPES = ("LATENT", "DICT")
    RETURN_NAMES = ("stabilized_latent", "stabilizer_telemetry")
    FUNCTION = "stabilize"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent": ("LATENT",),
                "truncation_psi": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 3.0, "step": 0.05,
                    "display": "slider",
                    "tooltip": "1.0 = identity; <1.0 = pull outliers toward the channel-mean centroid (cleaner); >1.0 = exaggerate variance",
                }),
                "subspace_rank": ("INT", {
                    "default": -1, "min": -1, "max": 64,
                    "tooltip": "Woodbury low-rank subspace size; -1 = keep all available SVD components, 0 = Frechet-mean template only",
                }),
                "scale_cap": ("FLOAT", {
                    "default": 10.0, "min": 0.1, "max": 1000.0, "step": 0.1,
                    "tooltip": "Bounded coupling-scale magnitude cap (tanh scale_map). Large -> near identity; small -> strong attenuation",
                }),
            },
            "optional": {
                "jitter_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Uniform dequantization jitter peak amplitude; 0 disables this stage",
                }),
                "residual_variance": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Residual isotropic variance sigma^2 for the Woodbury conditional mean; 0 = estimate from the SVD trailing eigenvalues",
                }),
            },
        }

    def stabilize(
        self,
        latent: Dict[str, Any],
        truncation_psi: float,
        subspace_rank: int,
        scale_cap: float,
        jitter_strength: float = 0.0,
        residual_variance: float = 0.0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with torch.no_grad():
            s = latent.get("samples")
            if s is None or s.ndim != 4:
                raise ValueError(
                    "GimbalLatentStabilizer: latent missing 4-D 'samples' tensor.")

            # sigma2 = 0 means "estimate from the SVD" (None to the primitive).
            sigma2 = residual_variance if residual_variance > 0 else None

            # mu and the low-rank geometry are estimated from the primary
            # latent itself (mu=None), keeping the pipeline self-contained.
            out = run_lamnr_pipeline(
                s,
                psi=float(truncation_psi),
                rank=int(subspace_rank),
                sigma2=sigma2,
                jitter=float(jitter_strength),
                scale_cap=float(scale_cap),
            )

            in_f = s.float()
            out_f = out.float()

            telemetry = {
                "instrument": "GimbalLatentStabilizer",
                "psi": truncation_psi,
                "rank": subspace_rank,
                "scale_cap": scale_cap,
                "jitter_strength": jitter_strength,
                "residual_variance": residual_variance if residual_variance > 0 else "auto (SVD)",
                "input_variance": round(in_f.var().item(), 4),
                "output_variance": round(out_f.var().item(), 4),
                "input_shape": list(s.shape),
            }

            out_latent = latent.copy()
            out_latent["samples"] = out.to(s.dtype)
            return (out_latent, telemetry)
