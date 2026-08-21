"""
[Form & Noise Atelier - Gimbal Node Suite]

LAMNr advanced latent telemetry: OOD / anomaly / dependence diagnostics.

Extends the primitive `GimbalDiagnostics` node with the research-grade metric
primitives from `gimbal_latent_math`:

  E1/E2  log_likelihood  : per-sample exact log-likelihood under the
                           channel-wise diagonal Gaussian base (OOD score;
                           lower = more out-of-distribution).
  E9     mahalanobis     : per-sample Mahalanobis distance under the
                           low-rank-plus-diagonal cohort model (anomaly score;
                           larger = stronger deviation from the cohort).
  E10    total_correlation: per-sample TC estimate via the density-ratio
                            trick (dependence between latent coordinates;
                            ~0 = statistically disentangled [skipped if B < 2]).
  E5     geodesic        : angular distance on the hypersphere, optionally to a
                            comparison latent; falls back to distance from the
                            batch centroid; 0.0 when B == 1 and no comparison.

Outputs the input LATENT unchanged plus four FLOAT means and a telemetry DICT
carrying the full per-sample vectors.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional

try:
    from .gimbal_latent_math import (
        log_likelihood,
        mahalanobis,
        total_correlation,
        geodesic_angular,
    )
except ImportError:  # direct top-level import (test harness)
    from gimbal_latent_math import (
        log_likelihood,
        mahalanobis,
        total_correlation,
        geodesic_angular,
    )


class GimbalLatentTelemetry:
    """LAMNr OOD/anomaly/dependence metric node (E1, E5, E9, E10)."""

    CATEGORY = "Gimbal/Telemetry"
    RETURN_TYPES = ("LATENT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "DICT")
    RETURN_NAMES = ("latent_passthrough", "mean_log_likelihood",
                    "mean_mahalanobis", "mean_total_correlation",
                    "mean_geodesic", "advanced_telemetry")
    FUNCTION = "diagnose"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "comparison_latent": ("LATENT", {
                    "tooltip": "Optional target for the E5 geodesic comparison. If absent and B>1, distance is measured from the batch centroid.",
                }),
                "subspace_rank": ("INT", {
                    "default": -1, "min": -1, "max": 64,
                    "tooltip": "E9 Mahalanobis low-rank subspace size. -1 = all SVD components; 0 = isotropic-only baseline",
                }),
                "bandwidth": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.001,
                    "tooltip": "E10 Gaussian bandwidth for the TC estimator. 0 = auto (median pairwise distance)",
                }),
            },
        }

    def diagnose(
        self,
        latent: Dict[str, Any],
        comparison_latent: Optional[Dict[str, Any]] = None,
        subspace_rank: int = -1,
        bandwidth: float = 0.0,
    ) -> Tuple[Dict[str, Any], float, float, float, float, Dict[str, Any]]:
        with torch.no_grad():
            s = latent.get("samples")
            if s is None or s.ndim != 4:
                raise ValueError(
                    "GimbalLatentTelemetry: latent missing 4-D 'samples' tensor.")
            B = int(s.shape[0])

            telemetry: Dict[str, Any] = {
                "instrument": "GimbalLatentTelemetry",
                "input_shape": list(s.shape),
            }

            # ----------------------------------------------------------------
            # E1/E2  Exact log-likelihood (channel-diagonal Gaussian)
            # ----------------------------------------------------------------
            ll = log_likelihood(s)
            mean_ll = float(ll.mean().item())
            telemetry["per_sample_log_likelihood"] = ll.tolist()

            # ----------------------------------------------------------------
            # E9  Mahalanobis (low-rank cohort model)
            # ----------------------------------------------------------------
            mh = mahalanobis(s, rank=subspace_rank)
            mean_mh = float(mh.mean().item())
            telemetry["per_sample_mahalanobis"] = mh.tolist()
            telemetry["rank"] = subspace_rank

            # ----------------------------------------------------------------
            # E10  Total Correlation (skipped on B < 2: TC requires a batch)
            # ----------------------------------------------------------------
            if B >= 2:
                bw = bandwidth if bandwidth > 0 else None
                tc = total_correlation(s, bandwidth=bw)
                mean_tc = float(tc.mean().item())
                telemetry["per_sample_total_correlation"] = tc.tolist()
                telemetry["bandwidth"] = bw if bw is not None else "auto (median)"
            else:
                mean_tc = 0.0
                telemetry["per_sample_total_correlation"] = "skipped (B < 2)"
                telemetry["bandwidth"] = "skipped (B < 2)"

            # ----------------------------------------------------------------
            # E5  Geodesic angular distance
            # ----------------------------------------------------------------
            if comparison_latent is not None:
                comp_s = comparison_latent.get("samples")
                if comp_s is None or comp_s.ndim != 4:
                    raise ValueError(
                        "GimbalLatentTelemetry: 'comparison_latent' must "
                        "carry a 4-D 'samples' tensor.")
                H, W = s.shape[2], s.shape[3]
                if comp_s.shape[-2:] != (H, W):
                    comp_s = F.interpolate(comp_s.float(), size=(H, W),
                                           mode="bilinear",
                                           align_corners=False).to(s.dtype)
                geo = geodesic_angular(s, comp_s)
                mean_geo = float(geo.mean().item())
                telemetry["per_sample_geodesic"] = geo.tolist()
                telemetry["geodesic_target"] = "comparison_latent"
            elif B > 1:
                centroid = s.float().mean(dim=0, keepdim=True).expand_as(s)
                geo = geodesic_angular(s, centroid)
                mean_geo = float(geo.mean().item())
                telemetry["per_sample_geodesic"] = geo.tolist()
                telemetry["geodesic_target"] = "batch centroid"
            else:
                mean_geo = 0.0
                telemetry["per_sample_geodesic"] = "skipped (no target, B == 1)"
                telemetry["geodesic_target"] = "none"

            return (latent, mean_ll, mean_mh, mean_tc, mean_geo, telemetry)
