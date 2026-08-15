import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

class GimbalTruncation:
    """
    [Form & Noise Atelier — Gimbal Node Suite]
    
    StyleGAN2-style latent variance truncation and centroid shrinkage.
    Compresses outliers toward distribution mean (psi < 1.0) to eliminate fried artifacts,
    or exaggerates features away from mean (psi > 1.0).
    
    Formula: z' = mu + psi * (z - mu)
    """

    CATEGORY = "Gimbal/Stabilizer"
    RETURN_TYPES = ("LATENT", "DICT")
    RETURN_NAMES = ("stabilized_latent", "truncation_telemetry")
    FUNCTION = "apply_truncation"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent": ("LATENT",),
                "truncation_psi": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "1.0 = identity; <1.0 = pulls toward core mean (cleaner, less chaotic); >1.0 = exaggerates variance"
                }),
                "channel_adaptive": ("BOOLEAN", {"default": True, "tooltip": "Computes mean per-channel rather than globally"}),
            },
            "optional": {
                "reference_batch": ("LATENT", {"tooltip": "Optional batch to compute empirical centroid mean"}),
            }
        }

    def apply_truncation(
        self,
        latent: Dict[str, Any],
        truncation_psi: float,
        channel_adaptive: bool,
        reference_batch: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with torch.no_grad():
            s = latent.get("samples")
            if s is None or s.ndim != 4:
                raise ValueError("GimbalTruncation: latent missing 4D 'samples' tensor.")

            B, C, H, W = s.shape
            s_f = s.float()

            if reference_batch is not None and reference_batch.get("samples") is not None:
                ref_s = reference_batch["samples"].float()
                if channel_adaptive:
                    # Per-channel mean across reference batch
                    mu = ref_s.mean(dim=(0, 2, 3), keepdim=True)
                else:
                    mu = ref_s.mean(dim=(0, 1, 2, 3), keepdim=True)
            else:
                if channel_adaptive:
                    # Channel mean of current latent
                    mu = s_f.mean(dim=(2, 3), keepdim=True)
                else:
                    mu = s_f.mean(dim=(1, 2, 3), keepdim=True)

            truncated = mu + truncation_psi * (s_f - mu)

            out_latent = latent.copy()
            out_latent["samples"] = truncated.to(s.dtype)

            telemetry = {
                "instrument": "GimbalTruncation",
                "psi": truncation_psi,
                "channel_adaptive": channel_adaptive,
                "initial_variance": round(s_f.var().item(), 4),
                "truncated_variance": round(truncated.var().item(), 4),
                "has_reference_batch": reference_batch is not None,
            }

            return (out_latent, telemetry)
