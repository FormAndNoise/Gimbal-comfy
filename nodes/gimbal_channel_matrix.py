import torch
import torch.nn.functional as F
import warnings
from typing import Dict, Any, Tuple

class GimbalChannelSplit:
    """
    [Form & Noise Atelier — Gimbal Node Suite]
    
    Splits a multi-channel latent tensor into two sub-band groups (A and B).
    For SDXL (4 channels): split into [0:split] and [split:4].
    For FLUX / SD3 (16 channels): split into structural/luminance vs chrominance/high-frequency bands.
    """

    CATEGORY = "Gimbal/Subspace"
    RETURN_TYPES = ("LATENT", "LATENT", "DICT")
    RETURN_NAMES = ("latent_band_A", "latent_band_B", "telemetry")
    FUNCTION = "split_channels"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent": ("LATENT",),
                "split_index": ("INT", {"default": 2, "min": 1, "max": 127, "step": 1, "tooltip": "Channel boundary index"}),
            }
        }

    def split_channels(self, latent: Dict[str, Any], split_index: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        s = latent.get("samples")
        if s is None or s.ndim != 4:
            raise ValueError("GimbalChannelSplit: latent missing 4D 'samples' tensor.")

        B, C, H, W = s.shape
        split_pt = max(1, min(split_index, C - 1))

        band_A_s = s[:, :split_pt, :, :].clone()
        band_B_s = s[:, split_pt:, :, :].clone()

        out_A = latent.copy()
        out_A["samples"] = band_A_s

        out_B = latent.copy()
        out_B["samples"] = band_B_s

        telemetry = {
            "instrument": "GimbalChannelSplit",
            "total_channels": C,
            "split_point": split_pt,
            "band_A_channels": split_pt,
            "band_B_channels": C - split_pt,
            "spatial_dim": [H, W],
        }

        return (out_A, out_B, telemetry)


class GimbalChannelMerge:
    """
    [Form & Noise Atelier — Gimbal Node Suite]
    
    Recombines two sub-band latent groups along the channel dimension.
    Handles spatial dimension alignment if bands were resized or manipulated.
    """

    CATEGORY = "Gimbal/Subspace"
    RETURN_TYPES = ("LATENT", "DICT")
    RETURN_NAMES = ("merged_latent", "telemetry")
    FUNCTION = "merge_channels"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent_band_A": ("LATENT",),
                "latent_band_B": ("LATENT",),
            }
        }

    def merge_channels(self, latent_band_A: Dict[str, Any], latent_band_B: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        s_A = latent_band_A.get("samples")
        s_B = latent_band_B.get("samples")

        if s_A is None or s_B is None:
            raise ValueError("GimbalChannelMerge: both inputs must contain valid LATENT dicts.")

        # Match batch size
        if s_A.shape[0] != s_B.shape[0]:
            if s_A.shape[0] == 1:
                s_A = s_A.expand(s_B.shape[0], *s_A.shape[1:])
            elif s_B.shape[0] == 1:
                s_B = s_B.expand(s_A.shape[0], *s_B.shape[1:])
            else:
                raise ValueError(f"GimbalChannelMerge: batch mismatch ({s_A.shape[0]} vs {s_B.shape[0]}).")

        # Match spatial dimensions if necessary
        if s_A.shape[-2:] != s_B.shape[-2:]:
            s_B = F.interpolate(s_B.float(), size=s_A.shape[-2:], mode="bilinear", align_corners=False).to(s_A.dtype)

        merged_samples = torch.cat([s_A, s_B], dim=1)

        out_latent = latent_band_A.copy()
        out_latent["samples"] = merged_samples

        telemetry = {
            "instrument": "GimbalChannelMerge",
            "band_A_channels": s_A.shape[1],
            "band_B_channels": s_B.shape[1],
            "total_merged_channels": merged_samples.shape[1],
            "output_shape": list(merged_samples.shape),
        }

        return (out_latent, telemetry)


class GimbalChannelScale:
    """
    [Form & Noise Atelier — Gimbal Node Suite]
    
    Independently scales individual channel band gains (e.g. boosting structural variance vs chrominance).
    """

    CATEGORY = "Gimbal/Subspace"
    RETURN_TYPES = ("LATENT", "DICT")
    RETURN_NAMES = ("scaled_latent", "telemetry")
    FUNCTION = "scale_channels"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent": ("LATENT",),
                "ch0_gain": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "ch1_gain": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "ch2_gain": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "ch3_gain": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "remaining_ch_gain": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05, "tooltip": "Applied to channels 4-15 on FLUX/SD3"}),
            }
        }

    def scale_channels(
        self,
        latent: Dict[str, Any],
        ch0_gain: float,
        ch1_gain: float,
        ch2_gain: float,
        ch3_gain: float,
        remaining_ch_gain: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        s = latent.get("samples")
        if s is None or s.ndim != 4:
            raise ValueError("GimbalChannelScale: latent missing 4D 'samples' tensor.")

        B, C, H, W = s.shape
        scaled = s.clone().float()

        gains = [ch0_gain, ch1_gain, ch2_gain, ch3_gain]
        for c in range(min(C, 4)):
            scaled[:, c, :, :] *= gains[c]

        if C > 4:
            scaled[:, 4:, :, :] *= remaining_ch_gain

        out_latent = latent.copy()
        out_latent["samples"] = scaled.to(s.dtype)

        telemetry = {
            "instrument": "GimbalChannelScale",
            "applied_gains": gains + ([remaining_ch_gain] if C > 4 else []),
            "total_channels": C,
        }

        return (out_latent, telemetry)
