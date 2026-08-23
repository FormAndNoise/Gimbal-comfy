# 🛡️ Gimbal Latent Stabilizer

> *The safety net you didn't know you needed. Plug it in and stop losing output to artifacts.*

**Class**: `GimbalLatentStabilizer`  
**Category**: `Gimbal/Stabilizer`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT, DICT)` → `stabilized_latent`, `telemetry`

---

## What It Does

When you push latents hard — high CFG, aggressive text steering, long orbital trajectories — values can spike outside the region the VAE was trained on. The result: plastic skin, embossed black wireframes instead of hair, posterized edges with comic-book outlines.

**GimbalLatentStabilizer** is a one-node fix. It runs the full **LAMNr quality pipeline** — variance truncation, bounded scale capping, and optional Woodbury denoising — to pull latent values back into the high-probability zone before your final KSampler sees them.

---

## At a Glance

```
[Any Latent] ──▶ [🛡️ Latent Stabilizer] ──▶ stable_latent ──▶ [KSampler] ──▶ [VAEDecode]
                   psi=0.88                    telemetry_dict
                   scale_cap=8.0
```

---

## Inputs

| Parameter | Type | Default | Range | Description |
|:---|:---|:---|:---|:---|
| `latent` | LATENT | — | — | The latent to stabilize. |
| `truncation_psi` | FLOAT | 0.90 | 0.0 – 3.0 | Variance shrinkage factor. `z' = μ + ψ(z − μ)`. Values <1 pull toward mean; values >1 amplify. |
| `subspace_rank` | INT | -1 | -1 to 64 | Rank for low-rank SVD. `-1` = use all available; `0` = mean-only (template). |
| `scale_cap` | FLOAT | 10.0 | 0.1 – 1000.0 | Bounded scale tanh cap. Smaller values = more aggressive suppression of extreme values. |
| `jitter_strength` | FLOAT | 0.0 | 0.0 – 1.0 | Uniform dequantization jitter `U(−1,1) × strength`. 0 = off. |
| `residual_variance` | FLOAT | 0.0 | 0.0+ | Override for σ² in Woodbury denoising. `0` = estimate from SVD automatically. |

---

## What Each Setting Actually Does

### `truncation_psi` (Most Important)

This is your primary dial. It implements **E3: Truncation Shrinkage**:

```
z' = μ + ψ × (z − μ)
```

- **ψ = 1.0**: Identity — no change.
- **ψ = 0.88**: Pulls every latent vector 12% closer to the batch mean. **Recommended for most workflows.**
- **ψ = 0.85**: More aggressive — good for very high CFG (7.0+).
- **ψ = 0.95**: Gentle — use when you want to preserve extreme values while just clipping the worst spikes.
- **ψ = 0.0**: Everything collapses to the mean — the "Fréchet mean template." Only useful as a baseline.
- **ψ > 1.0**: Amplifies variance. Useful for deliberately pushing toward more extreme stylistic territory.

### `scale_cap` (Anti-Explosion)

Implements **E11: Bounded Scale Map**:

```
s' = scale_cap × tanh(s / scale_cap)
```

Prevents coupling scale blowups and exploding gradients. At `scale_cap = 8.0`, any value above ~8.0 is smoothly capped via the tanh curve. At `scale_cap = 1000.0` it's effectively the identity. Lower = more aggressive capping.

### `subspace_rank`

Controls the Woodbury low-rank denoiser. Implements **E6/E7/E8**:
- The batch covariance is approximated as `Σ ≈ UΛU^T + σ²I` where U is `[D×r]`.
- The Woodbury MMSE denoiser uses this to project latents toward the shared batch structure.
- `-1`: Use all available components (max is `min(B, D)` — usually `B` for typical batch sizes).
- `0`: Collapses to mean-only imputation. Very aggressive.
- `4-16`: Good balance for most workflows.

---

## Recommended Settings by Use Case

| Scenario | ψ | scale_cap | Notes |
|:---|:---|:---|:---|
| General use | 0.88 | 10.0 | Safe default for most workflows. |
| High CFG (6.5–8.0) | 0.85 | 8.0 | More aggressive variance control. |
| Cross-Modal Bridge steering | 0.88 | 8.0 | Verified fix for black-outline artifacts. |
| Orbital animation | 0.90 | 10.0 | Light stabilization to prevent frame-to-frame drift. |
| Material matrix | 0.88 | 8.0 | Verified at denoise 0.65 for silhouette lock. |
| Gentle cleanup only | 0.95 | 20.0 | Minimal intervention. |

---

## The Telemetry Output

The second output `telemetry` is a DICT you can connect to a **Gimbal Diagnostics** node or inspect directly:

```json
{
  "instrument": "GimbalLatentStabilizer",
  "psi": 0.88,
  "rank": -1,
  "input_variance": 2.4531,
  "output_variance": 1.9047
}
```

A well-calibrated stabilization typically reduces variance by 15–25%. If `output_variance / input_variance > 0.95`, your ψ might not be aggressive enough for the amount of noise present.

---

## Pipeline Execution Order

The stabilizer runs these operations in sequence:

1. **E11**: Bounded scale capping — `s' = scale_cap × tanh(s / scale_cap)` — clips extreme magnitudes.
2. **E12**: Dequantization jitter — only if `jitter_strength > 0` — adds uniform noise to smooth quantization boundaries.
3. **E3**: Truncation shrinkage — `z' = μ + ψ(z − μ)` — pulls outliers toward the centroid.
4. **E7/E8**: Woodbury imputation — projects residuals onto the shared low-rank cohort subspace.

---

## Pro Tips

- **Always place Stabilizer after Cross-Modal Bridge** and before your final KSampler. The Bridge generates steering vectors that can push values into extreme regions; the Stabilizer tames them.
- **In the Material Matrix workflow**, Stabilizer between Channel Merge and KSampler is essential. Without it, the high-channel material steering (Ch 2–3 or Ch 8–15) tends to produce shimmering artifacts at denoise 0.65.
- **For orbital animation**, a gentle Stabilizer (ψ=0.90) after the Circular Orbit node prevents subtle variance drift between the first and last frames that would break the seamless loop.
- **jitter_strength**: Leave at 0.0 unless you're working with models that have quantized or quantized-aware latents. Non-zero jitter adds randomness.

---

## Under the Hood (Researchers)

**Numerical stability guarantees:**
| Concern | Handling |
|:---|:---|
| Division by zero | All denominators floored to ε = 1e-8 (float32), 1e-4 (float16) |
| SVD failure | Caught and retried with diagonal Gaussian jitter |
| D×D covariance inversion | Never formed — Woodbury keeps work in r-dim subspace, O(BDr) |
| float16 overflow | All internal math in float32; cast back to input dtype at output |
| Cholesky on singular matrix | `cholesky_with_jitter`: escalating diagonal jitter until factorization succeeds |
| Log-det explosion | E11 bounded_scale prevents coupling scale blowups |

**Input shape requirement**: `[B, C, H, W]` 4D tensor. Raises `ValueError` for non-4D input.  
**autograd**: No forced `torch.no_grad()` at the primitive level — wrap at the node level (matches ComfyUI inference convention).

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
