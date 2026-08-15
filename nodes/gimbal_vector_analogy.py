import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

class GimbalVectorAnalogy:
    """
    [Form & Noise Atelier — Gimbal Node Suite]
    
    Performs classic concept analogy vector arithmetic:
    Target = C + strength * (A - B)
    
    E.g. [Man with Glasses] (A) - [Man] (B) + [Woman] (C) = [Woman with Glasses]
    Supports orthogonal projection and spherical norm preservation.
    """

    CATEGORY = "Gimbal/Arithmetic"
    RETURN_TYPES = ("LATENT", "LATENT", "DICT")
    RETURN_NAMES = ("result_latent", "isolated_delta_vector", "telemetry")
    FUNCTION = "calculate_analogy"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "concept_A": ("LATENT", {"tooltip": "Target concept with desired attribute (e.g. Man with glasses)"}),
                "concept_B": ("LATENT", {"tooltip": "Base concept without attribute (e.g. Man)"}),
                "concept_C": ("LATENT", {"tooltip": "New recipient concept (e.g. Woman)"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05, "display": "slider"}),
                "ortho_project": ("BOOLEAN", {"default": False, "tooltip": "Removes C-parallel component from delta to avoid double-counting"}),
                "preserve_norm": ("BOOLEAN", {"default": True, "tooltip": "Rescales output to match recipient C's spherical radius"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    def calculate_analogy(
        self,
        concept_A: Dict[str, Any],
        concept_B: Dict[str, Any],
        concept_C: Dict[str, Any],
        strength: float,
        ortho_project: bool,
        preserve_norm: bool,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        with torch.no_grad():
            s_A = concept_A.get("samples")
            s_B = concept_B.get("samples")
            s_C = concept_C.get("samples")

            if s_A is None or s_B is None or s_C is None:
                raise ValueError("GimbalVectorAnalogy: concept_A, concept_B, and concept_C must all be valid LATENT dicts.")

            # Align spatial sizes to C
            target_hw = s_C.shape[-2:]
            if s_A.shape[-2:] != target_hw:
                s_A = F.interpolate(s_A.float(), size=target_hw, mode="bilinear", align_corners=False).to(s_C.dtype)
            if s_B.shape[-2:] != target_hw:
                s_B = F.interpolate(s_B.float(), size=target_hw, mode="bilinear", align_corners=False).to(s_C.dtype)

            # Match batches
            B_max = max(s_A.shape[0], s_B.shape[0], s_C.shape[0])
            if s_A.shape[0] == 1 and B_max > 1: s_A = s_A.expand(B_max, *s_A.shape[1:])
            if s_B.shape[0] == 1 and B_max > 1: s_B = s_B.expand(B_max, *s_B.shape[1:])
            if s_C.shape[0] == 1 and B_max > 1: s_C = s_C.expand(B_max, *s_C.shape[1:])

            # Compute delta vector (A - B)
            delta = (s_A.float() - s_B.float())

            C_f = s_C.float()
            C_flat = C_f.reshape(B_max, -1)
            C_norm = C_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

            if ortho_project:
                delta_flat = delta.reshape(B_max, -1)
                C_hat = C_flat / C_norm
                proj = (delta_flat * C_hat).sum(dim=1, keepdim=True) * C_hat
                delta_ortho = delta_flat - proj
                delta = delta_ortho.reshape(delta.shape)

            result = C_f + delta * strength

            if preserve_norm:
                res_flat = result.reshape(B_max, -1)
                res_norm = res_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
                result = ((res_flat / res_norm) * C_norm).reshape(result.shape)

            if mask is not None:
                m = mask.unsqueeze(1).float() if mask.ndim == 3 else mask.unsqueeze(0).unsqueeze(1).float()
                if m.shape[-2:] != target_hw:
                    m = F.interpolate(m, size=target_hw, mode="bilinear", align_corners=False)
                result = C_f * (1.0 - m) + result * m

            out_latent = concept_C.copy()
            out_latent["samples"] = result.to(s_C.dtype)

            delta_latent = concept_A.copy()
            delta_latent["samples"] = delta.to(s_C.dtype)

            telemetry = {
                "instrument": "GimbalVectorAnalogy",
                "strength": strength,
                "ortho_projected": ortho_project,
                "norm_preserved": preserve_norm,
                "delta_l2_norm": round(delta.float().norm().item(), 4),
            }

            return (out_latent, delta_latent, telemetry)
