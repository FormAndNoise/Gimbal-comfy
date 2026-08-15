import torch
from typing import Dict, Any, Tuple

class GimbalDiagnostics:
    """
    [Form & Noise Atelier — Gimbal Node Suite]
    
    Comprehensive latent space telemetry and health diagnostics.
    Outputs primitive scalar values and a formatted flight log string.
    """

    CATEGORY = "Gimbal/Telemetry"
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("mean", "std_dev", "min_val", "max_val", "l2_norm", "batch_size", "channels", "height", "width", "flight_report")
    FUNCTION = "inspect_latent"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    def inspect_latent(self, latent: Dict[str, Any]) -> Tuple[float, float, float, float, float, int, int, int, int, str]:
        s = latent.get("samples")
        if s is None or s.ndim != 4:
            raise ValueError("GimbalDiagnostics: latent missing 4D 'samples' tensor.")

        B, C, H, W = s.shape
        s_f = s.float()

        mean_val = float(s_f.mean().item())
        std_val = float(s_f.std().item())
        min_val = float(s_f.min().item())
        max_val = float(s_f.max().item())
        l2_val = float(s_f.norm().item())

        # Channel-wise stats
        ch_means = [round(m.item(), 3) for m in s_f.mean(dim=(0, 2, 3))]
        ch_stds = [round(st.item(), 3) for st in s_f.std(dim=(0, 2, 3))]

        report = (
            f"=== GIMBAL LATENT FLIGHT TELEMETRY ===\n"
            f"Dimensions: Batch={B} | Channels={C} | Spatial={H}x{W} (Total Elements: {s.numel():,})\n"
            f"Global Distribution: Mean={mean_val:.4f} | Std={std_val:.4f} | Range=[{min_val:.4f}, {max_val:.4f}]\n"
            f"Total L2 Vector Norm: {l2_val:.4f}\n"
            f"Per-Channel Means: {ch_means}\n"
            f"Per-Channel Stds:  {ch_stds}\n"
            f"Dtype: {s.dtype} | Device: {s.device}\n"
            f"======================================"
        )

        return (mean_val, std_val, min_val, max_val, l2_val, B, C, H, W, report)
