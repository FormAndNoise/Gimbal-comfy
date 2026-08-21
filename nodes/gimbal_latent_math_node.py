"""
[Form & Noise Atelier - Gimbal Node Suite]

ComfyUI dispatcher node exposing every LAMNr / Disentanglement primitive.

The `op` enum selects the equation set; matching parameters are routed from
the node's generic inputs. Transform ops return the transformed latent;
metric ops return the input latent unchanged with the per-sample scalar mean
on the FLOAT output and the full per-sample vector in the telemetry DICT.

Ops and their forwarded parameters:
  channel_diagonal_gaussian -> E2 (none)
  truncation                -> E3 (psi, channel_adaptive, reference_batch)
  slerp_mu                  -> E4 (additional_latent as target, t, reference_batch as mu)
  bounded_scale             -> E11 (scale_cap)
  dequantize                -> E12 (jitter_strength, schedule, step, seed)
  woodbury_impute           -> E7/E8 (subspace_rank, sigma2)
  pipeline                  -> full stack (psi, subspace_rank, scale_cap,
                               jitter_strength, sigma2)
  log_likelihood            -> E1/E2 (none)
  mahalanobis               -> E9 (subspace_rank, sigma2)
  total_correlation         -> E10 (bandwidth = jitter_strength; 0 = auto)
  geodesic                  -> E5 (additional_latent as other vector)

No ComfyUI-specific imports beyond torch. All math dispatched to
`gimbal_latent_math.apply_new_latent_math` / direct primitives for kwargs.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional

try:
    from .gimbal_latent_math import (
        apply_new_latent_math,
        truncation,
        slerp_mu,
        woodbury_impute,
        mahalanobis,
        total_correlation,
        geodesic_angular,
        dequantize,
    )
except ImportError:  # direct top-level import (test harness)
    from gimbal_latent_math import (
        apply_new_latent_math,
        truncation,
        slerp_mu,
        woodbury_impute,
        mahalanobis,
        total_correlation,
        geodesic_angular,
        dequantize,
    )

_TRANSFORM_OPS = (
    "channel_diagonal_gaussian", "truncation", "slerp_mu", "bounded_scale",
    "dequantize", "woodbury_impute", "pipeline",
)
_METRIC_OPS = ("log_likelihood", "mahalanobis", "total_correlation", "geodesic")
_ALL_OPS = _TRANSFORM_OPS + _METRIC_OPS


class GimbalLatentMath:
    """Dispatcher node for LAMNr latent math primitives (E1-E13)."""

    CATEGORY = "Gimbal/Primitives"
    RETURN_TYPES = ("LATENT", "FLOAT", "DICT")
    RETURN_NAMES = ("latent", "mean_metric", "op_telemetry")
    FUNCTION = "apply_op"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent": ("LATENT",),
                "op": (_ALL_OPS, {
                    "default": "pipeline",
                    "tooltip": "Equation set to apply. Transform ops rewrite the latent; metric ops pass it through and emit FLOAT + DICT telemetry.",
                }),
            },
            "optional": {
                "psi": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "E3 truncation coefficient (truncation / pipeline). 1.0 = identity; <1.0 shrinks variance toward centroid; >1.0 exaggerates",
                }),
                "t": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "E4 mu-Slerp interpolation fraction (slerp_mu). 0 = input, 1 = additional_latent target",
                }),
                "subspace_rank": ("INT", {
                    "default": -1, "min": -1, "max": 64,
                    "tooltip": "E7/E8/E9 low-rank subspace size. -1 = all available SVD components; 0 = Frechet-mean / isotropic shrinkage only",
                }),
                "scale_cap": ("FLOAT", {
                    "default": 10.0, "min": 0.1, "max": 1000.0, "step": 0.1,
                    "tooltip": "E11 bounded coupling-scale tanh cap (bounded_scale / pipeline)",
                }),
                "jitter_strength": ("FLOAT", {
                    "default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "E12 dequantization amplitude (dequantize / pipeline). Doubles as E10 Gaussian bandwidth for total_correlation (0 = auto median bandwidth)",
                }),
                "sigma2": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "E7/E8/E9 residual isotropic variance. 0 = estimate from the SVD trailing eigenvalues",
                }),
                "schedule": (["linear", "cosine", "exponential"], {
                    "default": "cosine",
                    "tooltip": "E12 dequantization decay schedule",
                }),
                "step": ("INT", {
                    "default": 0, "min": 0, "max": 1000000,
                    "tooltip": "E12 current dequantization step (out of 1000); decay = 0 at step 1000",
                }),
                "channel_adaptive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "E3 per-channel centroid vs global scalar centroid when no reference_batch is wired (truncation)",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Seed for reproducible E12 jitter",
                }),
                "additional_latent": ("LATENT", {
                    "tooltip": "Target vector for E4 slerp_mu destination or E5 geodesic comparison. Optional; required only for those ops",
                }),
                "reference_batch": ("LATENT", {
                    "tooltip": "Cohort anchor: E4 mu centroid (slerp_mu) or E3 centroid (truncation)",
                }),
            },
        }

    @staticmethod
    def _samples(d: Dict[str, Any], name: str, kind: str = "latent") -> torch.Tensor:
        s = d.get("samples")
        if s is None:
            raise ValueError(f"GimbalLatentMath: '{name}' missing 'samples' tensor.")
        if s.ndim != 4:
            raise ValueError(
                f"GimbalLatentMath: '{name}' must be 4-D [B, C, H, W], "
                f"got {list(s.shape)}.")
        return s

    @staticmethod
    def _maybe_spatial_match(target: torch.Tensor, H: int, W: int,
                              dtype: torch.dtype) -> torch.Tensor:
        """Bilinearly match spatial dims for a secondary latent."""
        if target.shape[-2:] != (H, W):
            return F.interpolate(target.float(), size=(H, W), mode="bilinear",
                                 align_corners=False).to(dtype)
        return target

    def apply_op(
        self,
        latent: Dict[str, Any],
        op: str,
        psi: float = 0.9,
        t: float = 0.5,
        subspace_rank: int = -1,
        scale_cap: float = 10.0,
        jitter_strength: float = 0.001,
        sigma2: float = 0.0,
        schedule: str = "cosine",
        step: int = 0,
        channel_adaptive: bool = True,
        seed: int = 0,
        additional_latent: Optional[Dict[str, Any]] = None,
        reference_batch: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        with torch.no_grad():
            s = self._samples(latent, "latent")
            B, C, H, W = s.shape

            if op not in _ALL_OPS:
                raise ValueError(
                    f"GimbalLatentMath: unknown op '{op}'.")

            # Common sd / sd2 -> None translation.
            sd2 = sigma2 if sigma2 > 0 else None
            telemetry: Dict[str, Any] = {
                "instrument": "GimbalLatentMath", "op": op,
                "input_shape": list(s.shape),
            }

            # Optional inputs (extract once).
            extra_s: Optional[torch.Tensor] = None
            ref_s: Optional[torch.Tensor] = None
            if additional_latent is not None:
                extra_s = self._maybe_spatial_match(
                    self._samples(additional_latent, "additional_latent"),
                    H, W, s.dtype)
            if reference_batch is not None:
                ref_s = reference_batch.get("samples")

            # ----------------------------------------------------------------
            # Full pipeline transform
            # ----------------------------------------------------------------
            if op == "pipeline":
                out_t = apply_new_latent_math(
                    s, "pipeline", psi, subspace_rank, sd2, jitter_strength,
                    scale_cap)
                telemetry.update({
                    "psi": psi, "rank": subspace_rank,
                    "scale_cap": scale_cap, "jitter": jitter_strength,
                    "sigma2": sd2 if sd2 is not None else "auto",
                })
                return (self._pack(latent, out_t), 0.0, telemetry)

            # ----------------------------------------------------------------
            # Transform ops
            # ----------------------------------------------------------------
            if op == "channel_diagonal_gaussian":
                out_t = apply_new_latent_math(s, "channel_diagonal_gaussian")
                return (self._pack(latent, out_t), 0.0, telemetry)

            if op == "truncation":
                ref_mu = ref_s.float().mean(dim=(0, 2, 3), keepdim=True) \
                    if ref_s is not None else None
                out_t = truncation(s, psi=psi, mu=ref_mu,
                                   channel_adaptive=channel_adaptive)
                telemetry.update({"psi": psi, "has_reference": ref_s is not None,
                                  "channel_adaptive": channel_adaptive})
                return (self._pack(latent, out_t), 0.0, telemetry)

            if op == "bounded_scale":
                out_t = apply_new_latent_math(s, "bounded_scale", scale_cap)
                telemetry["scale_cap"] = scale_cap
                return (self._pack(latent, out_t), 0.0, telemetry)

            if op == "dequantize":
                gen = torch.Generator(device="cpu").manual_seed(seed)
                out_t = dequantize(s, strength=jitter_strength,
                                   schedule=schedule, step=step,
                                   total_steps=1000.0, generator=gen)
                telemetry.update({
                    "jitter_strength": jitter_strength,
                    "schedule": schedule, "step": step, "seed": seed,
                })
                return (self._pack(latent, out_t), 0.0, telemetry)

            if op == "woodbury_impute":
                ref_mu = (ref_s.float().reshape(ref_s.shape[0], -1).mean(dim=0)
                          if ref_s is not None else None)
                out_t = woodbury_impute(s, mu=ref_mu, rank=subspace_rank,
                                        sigma2=sd2)
                telemetry.update({"rank": subspace_rank,
                                  "sigma2": sd2 if sd2 is not None else "auto",
                                  "has_reference": ref_s is not None})
                return (self._pack(latent, out_t), 0.0, telemetry)

            if op == "slerp_mu":
                if extra_s is None:
                    raise ValueError(
                        "GimbalLatentMath: 'slerp_mu' op requires "
                        "'additional_latent' as the target.")
                ref_mu = ref_s if ref_s is not None else None
                out_t = slerp_mu(s, extra_s, t=t, mu=ref_mu)
                telemetry.update({"t": t, "has_mu_anchor": ref_s is not None})
                return (self._pack(latent, out_t), 0.0, telemetry)

            # ----------------------------------------------------------------
            # Metric ops (input passthrough; scalars on outputs)
            # ----------------------------------------------------------------
            if op == "log_likelihood":
                m_out = apply_new_latent_math(s, "log_likelihood")
                telemetry["per_sample_log_likelihood"] = m_out.tolist()
                return (latent, float(m_out.mean().item()), telemetry)

            if op == "mahalanobis":
                m_out = mahalanobis(s, rank=subspace_rank, sigma2=sd2)
                telemetry.update({
                    "rank": subspace_rank,
                    "sigma2": sd2 if sd2 is not None else "auto",
                    "per_sample_distance": m_out.tolist(),
                })
                return (latent, float(m_out.mean().item()), telemetry)

            if op == "total_correlation":
                bw = jitter_strength if jitter_strength > 0 else None
                m_out = total_correlation(s, bandwidth=bw)
                telemetry.update({
                    "bandwidth": bw if bw is not None else "auto (median)",
                    "per_sample_tc": m_out.tolist(),
                })
                return (latent, float(m_out.mean().item()), telemetry)

            if op == "geodesic":
                if extra_s is None:
                    raise ValueError(
                        "GimbalLatentMath: 'geodesic' op requires "
                        "'additional_latent' to compare against.")
                ref_mu = ref_s if ref_s is not None else None
                m_out = geodesic_angular(s, extra_s, mu=ref_mu)
                telemetry.update({
                    "has_mu_anchor": ref_s is not None,
                    "per_sample_angle_radians": m_out.tolist(),
                })
                return (latent, float(m_out.mean().item()), telemetry)

            raise ValueError(f"GimbalLatentMath: unhandled op '{op}'.")

    @staticmethod
    def _pack(latent: Dict[str, Any], samples: torch.Tensor) -> Dict[str, Any]:
        out = latent.copy()
        out["samples"] = samples.to(latent["samples"].dtype)
        return out
