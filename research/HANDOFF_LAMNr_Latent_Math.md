# Hand-Off: LAMNr / Disentanglement Latent Math - ComfyUI Integration Guide

> **Status: INTEGRATED.** The math primitives (`nodes/gimbal_latent_math.py`)
> are wrapped by three ComfyUI nodes registered in `__init__.py` (25 node
> classes total):
> * `GimbalLatentStabilizer` -> `nodes/gimbal_latent_stabilizer.py` (full pipeline)
> * `GimbalLatentMath`       -> `nodes/gimbal_latent_math_node.py` (dispatcher)
> * `GimbalLatentTelemetry`  -> `nodes/gimbal_latent_telemetry.py` (OOD metrics)
>
> Tests: `extras/tests/test_apply_new_latent_math.py` (50 math tests) +
> `extras/tests/test_lamnr_nodes.py` (18 node tests). All 68 pass.

> **Audience:** ComfyUI custom-node developer.
> **Source research:** `docs/research/System_Design_Framework_LAMNr.md` and `docs/research/Technical_Synthesis_Disentangled_Representation_Learning.md`.
> **Implementation:** `nodes/gimbal_latent_math.py` (pure PyTorch, no ComfyUI imports).
> **Tests:** `extras/tests/test_apply_new_latent_math.py` + `extras/tests/test_lamnr_nodes.py`.

---

## 1. TL;DR - What was built and what you need to do

A pure-PyTorch math module implementing **13 core equations** extracted from the two research documents was written at `nodes/gimbal_latent_math.py`. It exposes a single dispatcher

```python
from nodes.gimbal_latent_math import apply_new_latent_math
# or, inside the package:
from .nodes.gimbal_latent_math import apply_new_latent_math
```

and one numerically-stabilized primitive per equation. **Your job** is to wrap these primitives as Gimbal ComfyUI nodes (INPUT_TYPES / RETURN_TYPES / FUNCTION / CATEGORY) following the existing house style in `nodes/gimbal_truncation.py`, `nodes/gimbal_slerp.py`, etc. The math is done, tested, and stable; you only add the ComfyUI boilerplate and register the new node classes in `__init__.py`.

You do **not** need to read the two research markdown files to do the integration - everything you need is in section 3 below. Read them only if you want the theoretical motivation.

---

## 2. Research context (one-paragraph summary)

The two research documents describe a latent-space quality framework called **LAMNr** (Latent-Aligned Multiview Normalizing) built on normalizing-flow / disentanglement principles. The core ideas relevant to a ComfyUI latent node suite are:

1. **Channel-wise diagonal Gaussian base** - per-channel `(mu_c, s_c)` statistics, broadcast across spatial locations, prevent "per-voxel scale collapse" and unfold the manifold into a symmetric Gaussian coordinate space.
2. **Gaussian Annulus Theorem / Typical Set** - in high dimensions, probability mass lives on a thin spherical shell, NOT at the origin. Naive Lerp cuts through the hollow center and causes "variance collapse" (faded midpoints).
3. **mu-centered Slerp** - spherical interpolation anchored to the empirical population mean `mu` keeps the trajectory on the high-probability shell.
4. **Low-rank-plus-diagonal covariance via SVD** - `Sigma ~= U Lambda U^T + sigma^2 I` models the cohort covariance without ever forming a `D x D` matrix (D = C*H*W can be ~10^5).
5. **Woodbury matrix identity** - closed-form conditional-mean imputation / denoising in the low-rank subspace, `O(B D r)` instead of `O(D^3)`.
6. **Mahalanobis distance** - anomaly detection benchmarking a latent against the cohort variance.
7. **Total Correlation via the density-ratio trick** - minibatch-weighted estimate of statistical dependence between latent coordinates (the engine of disentanglement).
8. **Exact log-likelihood** - change-of-variables form for OOD detection.
9. **Numerical safeguard stack** - bounded coupling scales (tanh `scale_map`), uniform dequantization jitter with decay, gradient-norm clipping, adaptive-jitter Cholesky, eps-floored division, acos clamping, parallel-vector Lerp fallback.

---

## 3. The API contract (read this first)

### 3.1 Tensor shape and dtype rules

| Rule | Detail |
|---|---|
| **Input shape** | All primitives require a 4-D latent `[B, C, H, W]`. Non-4D inputs raise `ValueError`. |
| **Output shape - transforms** | Same `[B, C, H, W]` as input. |
| **Output shape - metrics** | `[B]` (one scalar per batch sample). |
| **dtype** | Internal math is always float32 for stability. Output is cast back to the **caller's dtype** (incl. float16). |
| **autograd** | Left intact (no forced `torch.no_grad`). Wrap in `no_grad` at the node level for inference, matching `GimbalTruncation`. |

### 3.2 The dispatcher

```python
apply_new_latent_math(latent_tensor, op=None, *op_args) -> torch.Tensor
```

- `op` is the **first positional arg** in `*args` (a string).
- Remaining positional args are forwarded to the primitive.
- `op=None` or omitted -> runs the full `pipeline` (see 3.4).
- Unknown `op` raises `ValueError`.

### 3.3 Primitive reference (one row per equation)

All primitives are importable directly from `nodes.gimbal_latent_math`. Keyword args are supported when calling primitives directly; the dispatcher is positional-only.

| Primitive | Signature (keyword form) | Returns | Eq |
|---|---|---|---|
| `channel_diagonal_gaussian` | `(z, eps=1e-8)` | `[B,C,H,W]` normalized | E2 |
| `channel_stats` | `(z, eps=1e-8)` | `(mu_c [1,C,1,1], s_c [1,C,1,1])` | E2 helper |
| `truncation` | `(z, psi, mu=None, channel_adaptive=True, eps=1e-8)` | `[B,C,H,W]` | E3 |
| `slerp_mu` | `(z, target, t, mu=None, eps=1e-8)` | `[B,C,H,W]` | E4 |
| `geodesic_angular` | `(z, other, mu=None, eps=1e-8)` | `[B]` in `[0, pi]` | E5 |
| `low_rank_covariance_svd` | `(z, rank=-1, eps=1e-8)` | `(mu[D], U[D,r], lam[r], sigma2 scalar, (B,D))` | E6 |
| `woodbury_impute` | `(z, mu=None, rank=-1, sigma2=None, eps=1e-8)` | `[B,C,H,W]` | E7/E8 |
| `mahalanobis` | `(z, mu=None, rank=-1, sigma2=None, eps=1e-8)` | `[B]` | E9 |
| `total_correlation` | `(z, bandwidth=None, eps=1e-8)` | `[B]` | E10 |
| `log_likelihood` | `(z, mu=None, scale=None, eps=1e-8)` | `[B]` | E1/E2 |
| `dequantize` | `(z, strength=1e-3, schedule="linear", step=0.0, total_steps=1.0, eps=1e-8, generator=None)` | `[B,C,H,W]` | E12 |
| `bounded_scale` | `(z, scale_cap=1.0, eps=1e-8)` | `[B,C,H,W]` | E11 |
| `cholesky_with_jitter` | `(mat, max_jitter=1e-3, eps=1e-6)` | lower-triangular factor | E13 utility |
| `run_lamnr_pipeline` | `(z, psi=0.9, rank=-1, sigma2=None, jitter=0.0, scale_cap=10.0, eps=1e-8)` | `[B,C,H,W]` | full stack |

### 3.4 The full pipeline (`run_lamnr_pipeline`)

Applies, in order:

1. `bounded_scale` (E11) - suppress exploding magnitudes.
2. `dequantize` (E12) - only if `jitter > 0`.
3. `truncation` with `psi` (E3) - pull outliers toward the centroid.
4. `woodbury_impute` (E7/E8) - project the residual onto the shared low-rank cohort subspace, removing idiosyncratic noise.

This is what the dispatcher runs by default.

---

## 4. The math (exact formulas, for verification)

Each equation is labeled `E1`-`E13` to match the inline citations in `gimbal_latent_math.py`.

### E1 - Change-of-Variables (exact log-likelihood)
```
log p(x) = log p_Z(f(x)) + log |det J_f|
```

### E2 - Channel-wise Diagonal Gaussian base
Parameters `(mu_c, s_c)` are tied per channel and broadcast across spatial locations:
```
log p_Z(z) = -0.5 * sum_c [ (z_c - mu_c)^2 / (s_c^2 + eps)
                            + log(s_c^2 + eps) + log(2 pi) ]
```
Bijective normalization: `z_norm = (z - mu_c) / (s_c + eps)`, inverse `z = z_norm * s_c + mu_c`.

### E3 - Truncation / variance shrinkage
```
z' = mu + psi * (z - mu)
```
`psi < 1` cleans (pull toward mean), `psi = 1` identity, `psi > 1` exaggerates.

### E4 - mu-centered Slerp
```
a_hat = (a - mu) / ||a - mu||
b_hat = (b - mu) / ||b - mu||
omega = arccos(clamp(a_hat . b_hat, -1+eps, 1-eps))
Slerp_mu(a, b, t) = mu + [ sin((1-t)*omega)/sin(omega) * a_hat
                          + sin(t*omega)/sin(omega)     * b_hat ] * r(t)
r(t) = (1-t)*||a-mu|| + t*||b-mu||
```
Falls back to Lerp when `sin(omega) ~ 0` (parallel/antiparallel). Endpoints: `t=0 -> a`, `t=1 -> b`.

### E5 - Geodesic (angular) distance
```
d_g(a, b) = arccos(clamp( (a-mu).(b-mu) / (||a-mu|| ||b-mu||), -1, 1 ))
```
Returns `[0, pi]`.

### E6 - Low-rank-plus-diagonal covariance via SVD
```
Sigma ~= U Lambda U^T + sigma^2 I,   U^T U = I_r
```
Computed from the economy SVD of the batch-centered data `zc = z_flat - mu`:
```
U_b S Vh^T = svd(zc, full_matrices=False)
U   = Vh[:r].T            # [D, r]  (right singular vectors)
lam = S^2 / B             # [r]     (eigenvalues)
sigma^2 = mean of trailing eigenvalues   (residual isotropic variance)
```
`rank=0` -> diagonal-only model. `rank=-1` -> all available `min(B, D)`.

### E7 - Woodbury matrix identity (push-through)
```
(sigma^2 I + U Lambda U^T)^{-1}
  = (1/sigma^2) [ I - U diag( l_i / (l_i + sigma^2) ) U^T ]
```

### E8 - Cross-modal imputation (conditional mean / MMSE denoiser)
```
z_hat = mu + U diag( l_i / (l_i + sigma^2) ) U^T (z_obs - mu)
```
No `D x D` inversion; work is `O(B D r)`. `rank=0` reduces to `z_hat = mu` (Frechet-mean template).

### E9 - Mahalanobis distance (anomaly detection)
Closed form via E7 (no D x D inversion):
```
d_M(z)^2 = (1/sigma^2) [ ||z-mu||^2
                         - sum_i ( l_i/(l_i+sigma^2) ) (u_i . (z-mu))^2 ]
```

### E10 - Total Correlation via the density-ratio trick
Minibatch-weighted sampling with Gaussian mixture centres = the batch samples themselves:
```
log p_joint(x^m)   = logsumexp_j( -||x^m - x^j||^2 / 2s^2 ) - log M
sum_d log p_d(x^m_d) = sum_d [ logsumexp_j( -(x^m_d - x^j_d)^2 / 2s^2 ) - log M ]
TC_hat_m = log p_joint(x^m) - sum_d log p_d(x^m_d)
```
Constants cancel exactly. `logsumexp` keeps it stable. Default bandwidth `s^2` = median off-diagonal joint distance.

### E11 - Bounded coupling scale (tanh scale_map)
```
s_bounded = scale_cap * tanh(s_c / scale_cap)
z' = mu_c + (z - mu_c) * (s_bounded / (s_c + eps))
```
`scale_cap` large -> identity. `scale_cap` small -> attenuates deviations.

### E12 - Uniform dequantization jitter with decay
```
z' = z + U(-1, 1) * alpha(t)
```
Schedules (`t = step / total_steps` in `[0, 1]`):
- `linear`:      `alpha(t) = a0 * (1 - t)`
- `cosine`:      `alpha(t) = a0 * 0.5 * (1 + cos(pi * t))`
- `exponential`: `alpha(t) = a0 * exp(-t)`

Jitter is symmetric (zero-mean) -> unbiased latent statistics.

### E13 - Numerical safeguard stack
- eps-floored division and `rsqrt`
- `acos` input clamped to `[-1+eps, 1-eps]`
- parallel-vector Lerp fallback in Slerp
- `cholesky_with_jitter`: adaptive diagonal jitter `eps * 10^k` until factorization succeeds or `max_jitter` reached

---

## 5. Numerical stability guarantees (what you do NOT have to worry about)

| Concern | How it is handled |
|---|---|
| Division by zero | All denominators clamped to `eps` (1e-8 float32, 1e-4 float16). |
| `acos` domain errors | Dot products clamped to `[-1+eps, 1-eps]`. |
| Slerp `sin(omega)=0` | Parallel-vector mask -> Lerp fallback. No NaN. |
| SVD failure | Caught (`torch._C._LinAlgError`); retried with tiny Gaussian jitter. |
| `D x D` covariance inversion | Never formed. Woodbury form keeps work in `r`-dim subspace. |
| float16 overflow | All internal math in float32; cast back at the end. |
| Cholesky on singular matrix | `cholesky_with_jitter` adds escalating diagonal jitter. |
| Log-det explosion | `bounded_scale` (E11) caps coupling magnitudes. |

---

## 6. Suggested ComfyUI node wrappers

These are suggestions only - the math module is decoupled from ComfyUI so you can shape the nodes however fits the suite. Each follows the existing `GimbalTruncation` / `GimbalSlerp` pattern.

### 6.1 `GimbalLatentStabilizer` (the full pipeline)
Wraps `run_lamnr_pipeline`. One node, end-to-end quality improvement.
```python
CATEGORY = "Gimbal/Stabilizer"
RETURN_TYPES = ("LATENT", "DICT")
RETURN_NAMES = ("stabilized_latent", "telemetry")
FUNCTION = "stabilize"
INPUT_TYPES = {
    "required": {
        "latent": ("LATENT",),
        "truncation_psi": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 3.0, "step": 0.05}),
        "subspace_rank": ("INT", {"default": -1, "min": -1, "max": 64, "tooltip": "-1 = all available, 0 = mean-only"}),
        "scale_cap": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 1000.0}),
    },
    "optional": {
        "jitter_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
        "residual_variance": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "0 = estimate from SVD"}),
    },
}
```

### 6.2 `GimbalLatentMath` (the dispatcher - expert mode)
A single node exposing `op` as an enum, forwarding the remaining args. Useful for power users.
```python
INPUT_TYPES = {
    "required": {
        "latent": ("LATENT",),
        "op": (["pipeline","truncation","channel_diagonal_gaussian",
                "woodbury_impute","bounded_scale","dequantize","slerp_mu",
                "log_likelihood","mahalanobis","total_correlation","geodesic"],),
    },
    "optional": {
        "target_latent": ("LATENT",),    # for slerp_mu / geodesic
        "psi": ("FLOAT", {"default": 0.9}),
        "t": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
        "rank": ("INT", {"default": -1}),
        "scale_cap": ("FLOAT", {"default": 10.0}),
        "jitter": ("FLOAT", {"default": 0.0}),
    },
}
```
Return `("LATENT", "DICT")` where the DICT carries per-sample metrics for metric ops (or telemetry for transforms).

### 6.3 `GimbalLatentDiagnostics` (metric ops)
A read-only node wrapping `log_likelihood`, `mahalanobis`, `total_correlation`, `geodesic_angular` and returning a telemetry DICT (no latent passthrough, or an identity passthrough). Pairs naturally with the existing `GimbalDiagnostics` node.

### 6.4 Registration
Add to `__init__.py` `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` following the existing entries. Suggested keys:
```python
"GimbalLatentStabilizer": GimbalLatentStabilizer,
"GimbalLatentMath": GimbalLatentMath,
```

---

## 7. How to call the primitives from a node (concrete example)

```python
import torch
from typing import Dict, Any, Tuple, Optional
from .gimbal_latent_math import run_lamnr_pipeline, channel_stats

class GimbalLatentStabilizer:
    CATEGORY = "Gimbal/Stabilizer"
    RETURN_TYPES = ("LATENT", "DICT")
    RETURN_NAMES = ("stabilized_latent", "telemetry")
    FUNCTION = "stabilize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "truncation_psi": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 3.0, "step": 0.05}),
                "subspace_rank": ("INT", {"default": -1, "min": -1, "max": 64}),
                "scale_cap": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 1000.0}),
            },
            "optional": {
                "jitter_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "residual_variance": ("FLOAT", {"default": 0.0, "min": 0.0}),
            },
        }

    def stabilize(self, latent, truncation_psi, subspace_rank, scale_cap,
                  jitter_strength=0.0, residual_variance=0.0):
        with torch.no_grad():
            s = latent["samples"]
            if s is None or s.ndim != 4:
                raise ValueError("GimbalLatentStabilizer: latent missing 4D 'samples' tensor.")
            mu_c, s_c = channel_stats(s.float())
            out = run_lamnr_pipeline(
                s,
                psi=truncation_psi,
                rank=subspace_rank,
                sigma2=residual_variance if residual_variance > 0 else None,
                jitter=jitter_strength,
                scale_cap=scale_cap,
            )
            result = latent.copy()
            result["samples"] = out.to(s.dtype)
            telemetry = {
                "instrument": "GimbalLatentStabilizer",
                "psi": truncation_psi,
                "rank": subspace_rank,
                "input_variance": round(s.float().var().item(), 4),
                "output_variance": round(out.float().var().item(), 4),
            }
            return (result, telemetry)
```

---

## 8. Test coverage and how to run

`extras/tests/test_apply_new_latent_math.py` - **50 tests, all passing**.

Coverage by equation:
- E2: shape/dtype, zero-mean/unit-var per channel, exact invertibility.
- E3: `psi=1` identity, `psi<1` shrinks variance, `psi>1` grows, `psi=0` = mean, dtype.
- E4: `t=0`/`t=1` endpoints exact, midpoint no-variance-collapse, parallel fallback no-NaN, `[1,C,1,1]` mu broadcast, dtype.
- E5: self-distance ~0, orthogonal = pi/2, opposite = pi, range `[0, pi]`.
- E7/E8: `rank=0` = mean template, exact conditional-mean match vs manual SVD, shrinkage reduces radius.
- E9: non-negative, `rank=0` matches isotropic, low-rank matches manual Woodbury form, shape.
- E10: finite + shape, exact logsumexp formula match, independent-dims ~0.
- E1: exact channel-diagonal match, outlier has lower likelihood, shape.
- E11: identity at large cap, bounded magnitude at small cap, dtype.
- E12: zero-strength identity, zero-mean in expectation, schedule decay, dtype.
- E13: well-conditioned exact match, singular matrix recovers finite factor.
- Pipeline: shape/dtype, finite, `psi=1`/`jitter=0`/`rank=0`/large-cap = mean, `psi<1` reduces variance.
- Dispatcher: default = pipeline, unknown op raises, non-4D raises, non-tensor raises, all ops dispatch.

**Run command** (the repo-root `__init__.py` is the ComfyUI package with relative imports, so the `--rootdir` and `--import-mode` flags are required - this is a pre-existing quirk affecting all `extras/tests/` files, not just these):

```bash
python -m pytest extras/tests/test_apply_new_latent_math.py \
    --import-mode=importlib --rootdir=extras/tests -p no:cacheprovider
```

---

## 9. Out of scope for this integration (deliberately deferred)

- Frontend `web/js/gimbal.js` HUD widgets for the new metrics.
- Workflow JSON examples under `extras/example_workflows/`.

The math is frozen, tested, and the three node wrappers above are registered
in `__init__.py`. Node naming and category strings can be adjusted without
touching the math module.

---

*Verified By VibeCheck ✅*
