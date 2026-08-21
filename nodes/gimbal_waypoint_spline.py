import math
import time
import torch
import torch.nn.functional as F
import warnings
from typing import Dict, Any, Tuple, List, Optional
try:
    from .gimbal_slerp import slerp_pair_mu_centered, slerp_pair_origin_centered, compute_batch_centroid
except ImportError:
    from gimbal_slerp import slerp_pair_mu_centered, slerp_pair_origin_centered, compute_batch_centroid

class GimbalWaypointSpline:
    """
    [Form & Noise Atelier — Gimbal Node Suite]
    
    Generates a continuous flight trajectory traversing N arbitrary waypoint latents.
    Supports Spherical SLERP, Catmull-Rom Spherical Splines, and Geodesic Constant-Velocity reparameterization.
    """

    CATEGORY = "Gimbal/Trajectory"
    RETURN_TYPES = ("LATENT", "DICT")
    RETURN_NAMES = ("latent_path", "flight_telemetry")
    FUNCTION = "interpolate_waypoints"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "waypoints_batch": ("LATENT", {"tooltip": "Batch of N >= 2 keypoint latents"}),
                "total_steps": ("INT", {"default": 60, "min": 4, "max": 2048, "step": 1}),
                "spline_mode": (["Spherical_SLERP", "Catmull_Rom_Spline", "Normalized_Linear", "Cosine_Ease"], {"default": "Spherical_SLERP"}),
                "loop_trajectory": ("BOOLEAN", {"default": False, "tooltip": "Connects last waypoint back to first for looping"}),
                "tension": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "constant_velocity": ("BOOLEAN", {"default": True, "tooltip": "Reparameterizes by geodesic arc length for uniform speed"}),
            },
            "optional": {
                "mu_anchor": ("LATENT", {"tooltip": "Population centroid for μ-centered Slerp. If absent, the mean of all waypoints is used."}),
            }
        }

    @staticmethod
    def _slerp_pair(a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
        """Legacy origin-centered Slerp for flattened vector pairs. Delegates to shared module."""
        return slerp_pair_origin_centered(a, b, t, eps)

    def interpolate_waypoints(
        self,
        waypoints_batch: Dict[str, Any],
        total_steps: int,
        spline_mode: str,
        loop_trajectory: bool,
        tension: float,
        constant_velocity: bool,
        mu_anchor: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        t_start = time.perf_counter()

        with torch.no_grad():
            s = waypoints_batch.get("samples")
            if s is None or s.ndim != 4:
                raise ValueError("GimbalWaypointSpline: waypoints_batch must contain 4D samples [N, C, H, W].")

            N, C, H, W = s.shape
            if N < 2:
                raise ValueError(f"GimbalWaypointSpline requires at least 2 waypoints in batch, got N={N}.")

            device = s.device
            dtype = s.dtype

            # Flatten waypoints for geometric math
            w_flat = [s[i].reshape(-1).float() for i in range(N)]

            # Compute μ centroid for μ-centered Slerp
            if mu_anchor is not None:
                mu_s = mu_anchor.get("samples")
                if mu_s is not None and mu_s.ndim == 4:
                    mu_flat = mu_s.mean(dim=0).reshape(-1).float().to(device=device)
                else:
                    mu_flat = torch.stack(w_flat).mean(dim=0)
            else:
                # Default: empirical mean of all waypoints
                mu_flat = torch.stack(w_flat).mean(dim=0)

            if loop_trajectory:
                w_flat.append(w_flat[0])
            
            num_segments = len(w_flat) - 1

            # Compute geodesic arc lengths between consecutive waypoints
            seg_lengths = []
            for i in range(num_segments):
                a_norm = (w_flat[i] - mu_flat).norm().clamp(min=1e-7)
                b_norm = (w_flat[i+1] - mu_flat).norm().clamp(min=1e-7)
                dot = (((w_flat[i] - mu_flat) / a_norm) * ((w_flat[i+1] - mu_flat) / b_norm)).sum().clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                seg_lengths.append(math.acos(dot.item()))

            total_arc_length = sum(seg_lengths)
            if total_arc_length < 1e-6:
                total_arc_length = 1.0
                seg_lengths = [1.0 / num_segments] * num_segments

            # Step allocation per segment
            if constant_velocity:
                # Distribute total_steps proportional to geodesic arc length
                steps_per_seg = []
                cum_steps = 0
                for i in range(num_segments):
                    if i == num_segments - 1:
                        n_steps = total_steps - cum_steps
                    else:
                        n_steps = max(1, round(total_steps * (seg_lengths[i] / total_arc_length)))
                        cum_steps += n_steps
                    steps_per_seg.append(max(1, n_steps))
            else:
                base_steps = total_steps // num_segments
                steps_per_seg = [base_steps] * num_segments
                steps_per_seg[-1] += total_steps - (base_steps * num_segments)

            path_tensors = []

            for seg_idx in range(num_segments):
                p0 = w_flat[max(0, seg_idx - 1)]
                p1 = w_flat[seg_idx]
                p2 = w_flat[seg_idx + 1]
                p3 = w_flat[min(len(w_flat) - 1, seg_idx + 2)]

                n_steps = steps_per_seg[seg_idx]
                t_values = [k / float(n_steps) for k in range(n_steps)]

                for t in t_values:
                    if spline_mode == "Spherical_SLERP":
                        pt = slerp_pair_mu_centered(p1, p2, t, mu_flat)
                    elif spline_mode == "Cosine_Ease":
                        t_cos = (1.0 - math.cos(t * math.pi)) / 2.0
                        pt = slerp_pair_mu_centered(p1, p2, t_cos, mu_flat)
                    elif spline_mode == "Catmull_Rom_Spline":
                        # Spherical Catmull-Rom tangent projection
                        t2 = t * t
                        t3 = t2 * t
                        c0 = -tension * t3 + 2 * tension * t2 - tension * t
                        c1 = (2 - tension) * t3 + (tension - 3) * t2 + 1
                        c2 = (tension - 2) * t3 + (3 - 2 * tension) * t2 + tension * t
                        c3 = tension * t3 - tension * t2
                        pt = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3
                    else:  # Normalized_Linear
                        pt = (1.0 - t) * p1 + t * p2
                        # Project onto spherical radius
                        target_r = (1.0 - t) * (p1 - mu_flat).norm() + t * (p2 - mu_flat).norm()
                        pt = mu_flat + ((pt - mu_flat) / (pt - mu_flat).norm().clamp(min=1e-7)) * target_r

                    path_tensors.append(pt.reshape(1, C, H, W).to(dtype=dtype, device=device))

            out_samples = torch.cat(path_tensors, dim=0)

            telemetry = {
                "instrument": "GimbalWaypointSpline",
                "spline_mode": spline_mode,
                "num_waypoints": N,
                "loop_trajectory": loop_trajectory,
                "total_generated_steps": out_samples.shape[0],
                "geodesic_arc_lengths": [round(l, 4) for l in seg_lengths],
                "constant_velocity_engaged": constant_velocity,
                "slerp_anchor": "external_mu" if mu_anchor is not None else "waypoint_mean",
                "execution_time_ms": round((time.perf_counter() - t_start) * 1000, 3),
            }

            out_latent = waypoints_batch.copy()
            out_latent["samples"] = out_samples

            return (out_latent, telemetry)
