# ⚙️ The LAMNr Framework: Research-Grade Latent Stabilization

> *"LAMNr is the reason your latents don't catch fire."*

**LAMNr** — *Latent-Aligned Multiview Normalizing* — is the mathematical quality system underpinning the Gimbal Node Suite. Every stabilization, normalization, and anomaly-detection behavior traces back to its 13 equations.

---

## 🧯 The Beginner's Guide: What LAMNr Does for You

### The Problem: Latents Go Haywire

When you push ComfyUI hard — high CFG values (6.5+), aggressive denoise (0.7+), or heavy prompt steering — the numbers inside your latent tensor can spike far outside the range the VAE decoder expects. When the decoder encounters these outlier values, it produces characteristic artifacts:

- **Clipping:** Harsh black or white regions with no detail
- **Posterization:** Color banding, plastic-looking skin
- **Black-outline artifacts:** Wireframe-style embossed edges, especially on hair and foliage
- **Flat fog:** Washed-out areas where the decoder simply gave up

### The Fix: Pull Back to the Safe Zone

LAMNr continuously monitors the statistical health of your latent tensor and applies gentle corrections to keep values in the region where the VAE was trained to decode correctly — without erasing the creative direction you've established.

It's like a flight envelope protection system: you can push the aircraft hard, but it won't let you exceed the structural limits.

### Your Interface: `GimbalLatentStabilizer`

The entire LAMNr pipeline is surfaced through a single node:

```
GimbalLatentStabilizer
├── Input:  LATENT (from any sampler or node)
├── Output: LATENT (stabilized, ready for KSampler or VAE Decode)
│
├── truncation_psi (ψ):  0.88  ← default, works for 90% of cases
├── scale_cap:           8.0   ← maximum coupling scale before tanh
└── jitter_alpha:        0.0   ← enable at 0.01-0.05 for heavy quantization
```

**Recommended placement:** Directly before your final KSampler or VAE Decode node.

**For most workflows:** Just set `truncation_psi = 0.88` and leave everything else at defaults.

---

## 🔬 Power User / Researcher Section: The 13 Equations

The LAMNr framework is defined by 13 equations (E1–E13). Here is each one, its formal statement, and its operational purpose.

---

### E1 — Exact Log-Likelihood (OOD Detection)

```
log p(x) = log p_Z(f(x)) + log|det J_f|
```

**Purpose:** Out-of-distribution (OOD) detection. By computing the exact log-likelihood of a latent under the normalizing flow `f`, LAMNr identifies whether a latent has drifted outside the training distribution. High-CFG sampling is a primary driver of OOD drift.

**In practice:** E1 provides the scalar "health score" that determines whether downstream corrections (E3, E7/E8) are invoked.

---

### E2 — Channel Diagonal Gaussian (Canonical Normalization)

```
log p_Z(z) = -0.5 · Σ_c [ (z_c - μ_c)² / (s_c² + ε) + log(s_c² + ε) + log(2π) ]
```

**Purpose:** Per-channel whitening using a diagonal Gaussian approximation. Each channel `c` is independently normalized to its empirical mean `μ_c` and scale `s_c`. This:

1. Enforces the correct per-channel statistics regardless of fine-tuned encoder drift
2. Positions the Typical Set canonically at radius √D in normalized space
3. Makes μ-centered SLERP (E4) operate on the *true* distribution manifold

**Note:** ε is the numerical safeguard floor (see E13). This equation is the core of the "channel architecture awareness" — per-channel normalization is how Gimbal handles 4-channel SDXL and 16-channel FLUX differently while using the same math.

---

### E3 — Truncation Shrinkage

```
z' = μ + ψ · (z - μ),    ψ ∈ [0.85, 0.95]
```

**Purpose:** Soft truncation toward the distribution mean. Latent values that have drifted toward the Gaussian tails are pulled back proportionally. The parameter ψ (truncation_psi) controls the aggressiveness:

| ψ value | Effect |
|---|---|
| 0.95 | Subtle correction; preserves most outlier energy |
| 0.90 | Moderate; recommended for CFG 5–6 |
| 0.88 | Default; good for CFG 6.5–7.5 |
| 0.85 | Aggressive; for extreme CFG or very high denoise |

**Relationship to StyleGAN truncation trick:** E3 is the latent-space analog of StyleGAN2's ψ truncation in w-space. It trades a small amount of diversity for a large reduction in decoder-visible artifacts.

---

### E4 — μ-Centered SLERP

```
u = (z_A - μ) / ‖z_A - μ‖
v = (z_B - μ) / ‖z_B - μ‖
Ω = arccos(clamp(u · v, -1, 1))

z_slerp(t) = μ + [sin((1-t)Ω) / sin(Ω)] · (z_A - μ)
                + [sin(tΩ) / sin(Ω)] · (z_B - μ)
```

**Purpose:** Geodesic interpolation anchored at the empirical batch centroid μ rather than the geometric origin. Corrects for distribution shift in any encoder that doesn't produce exactly zero-mean latents.

→ Full treatment: [The Gaussian Annulus Theorem](gaussian_annulus.md)

---

### E5 — Geodesic Angular Distance

```
d_g(a, b) = arccos( clamp( (a-μ)·(b-μ) / (‖a-μ‖·‖b-μ‖), -1, 1 ) )
```

**Purpose:** The true distance between two latents on the Typical Set shell, measured in radians. Used for:

- Blend ratio calibration (how much "distance" a SLERP covers)
- Anomaly detection (a latent with d_g > threshold from the batch centroid is flagged OOD)
- Informational telemetry output on `GimbalCompass_Pro`

Clamp is required for numerical safety (see E13).

---

### E6 — Low-Rank + Diagonal SVD Approximation

```
Σ ≈ U Λ Uᵀ + σ²I
```
*Computed via economy SVD on a rank-r approximation, O(BDr) not O(D³)*

**Purpose:** Efficient covariance estimation. The true covariance of a D=65,536 latent space requires a D×D matrix — completely intractable. E6 approximates it as a low-rank component (capturing the top-r principal directions of variance) plus a residual diagonal noise term.

**Parameters:**
- `r` (rank): number of principal components retained. Default: 16 for SDXL, 32 for FLUX
- `σ²` (residual variance): estimated from the (D-r) discarded singular values
- `B` (batch size): used for online updates to the empirical estimate

This approximation enables the Woodbury MMSE denoiser (E7/E8) to run in closed form without any D×D matrix operations.

---

### E7 / E8 — Woodbury MMSE Denoiser

```
ẑ = μ + U · diag( λᵢ/(λᵢ + σ²) ) · Uᵀ · (z_obs - μ)
```

**Purpose:** Closed-form minimum mean-squared error (MMSE) denoising in the low-rank covariance space. Given a noisy observed latent `z_obs`, this returns the posterior mean estimate `ẑ` under the LAMNr Gaussian model.

**Why Woodbury?** Direct MMSE would require inverting a D×D matrix per call. The Woodbury matrix identity reduces this to:
1. Project `(z_obs - μ)` into the r-dimensional subspace via Uᵀ — O(Dr)
2. Apply diagonal scaling `λᵢ/(λᵢ + σ²)` — O(r)
3. Project back via U — O(Dr)

Total: **O(Dr)** instead of O(D³). For D=65,536 and r=16, this is a ~170,000× speedup over naive implementation.

**Interpretation:** The scale factors `λᵢ/(λᵢ+σ²)` are Wiener filter coefficients — they trust the principal components proportionally to their signal-to-noise ratio. Low-variance directions (noise-like) are shrunk toward the mean; high-variance directions (signal-like) are preserved.

---

### E9 — Mahalanobis Distance

```
d_M(z) = √[ (z - μ)ᵀ Σ⁻¹ (z - μ) ]
```
*Computed efficiently in the low-rank basis from E6*

**Purpose:** Anomaly detection metric. Unlike Euclidean distance, Mahalanobis distance accounts for the covariance structure — a latent that deviates along a high-variance principal direction is less anomalous than one that deviates along a low-variance direction.

Values above a threshold (typically d_M > √D + 4) trigger the OOD alert on `GimbalTelemetry` output.

---

### E10 — Total Correlation (Disentanglement Engine)

```
TC(z) = KL[ q(z) ‖ ∏_i q(z_i) ]
```
*Estimated via minibatch weighted samples (Chen et al., 2018)*

**Purpose:** Measures statistical dependence between latent coordinates. Low TC = highly disentangled representation (each dimension carries independent information). High TC = entangled (changing one dimension bleeds into others).

LAMNr uses the minibatch TC estimator for efficiency, avoiding the intractable marginal q(z_i) computation. This score is used to:
- Rank principal components by their independence contribution
- Identify which channels are most entangled (informing Cross-Modal Bridge routing)
- Provide a disentanglement health metric on `GimbalTelemetry`

---

### E11 — Bounded Scale Map

```
s' = scale_cap · tanh(s / scale_cap)
```

**Purpose:** Prevents exploding coupling scales in normalizing flow layers. Without this bound, aggressive CFG can drive coupling scale parameters `s` to very large values, causing the flow's Jacobian determinant to overflow and the log-likelihood (E1) to become meaningless.

`tanh` provides a smooth, differentiable bound that:
- Passes small `s` values through nearly unchanged: `tanh(s/cap) ≈ s/cap` for `s << cap`
- Saturates at ±scale_cap for large values, preventing numerical blowup
- Is everywhere C∞, preserving gradient flow

**Default scale_cap = 8.0.** Reduce to 4.0 for very high CFG (8+); increase to 12.0 only if you're working with deliberately wide distributions.

---

### E12 — Dequantization Jitter

```
z' = z + U(-1, 1) · α(t)
```
*where α(t) is a time-dependent schedule, U(-1,1) is uniform noise*

**Purpose:** Smooths discrete quantization boundaries introduced by 8-bit or 16-bit latent caching, VAE quantization layers, or ComfyUI's internal tensor storage. Without jitter, latents saved and reloaded from disk can have small but systematic discontinuities at quantization grid points.

α(t) is typically a small constant (0.01–0.05) or can be scheduled to decrease as denoise progresses. Set `jitter_alpha = 0.0` (default) to disable if working with float32 throughout.

---

### E13 — Numerical Safeguard Stack

LAMNr applies four numerical safeguards throughout the pipeline:

| Safeguard | Expression | Prevents |
|---|---|---|
| ε-floored division | `s² + ε` in denominators | Division by zero in degenerate channels |
| arccos clamping | `clamp(x, -1+ε, 1-ε)` | NaN from floating-point values slightly outside [-1,1] |
| Parallel-vector LERP fallback | if `sin(Ω) < ε`, use LERP | NaN from SLERP when vectors are nearly identical |
| Cholesky jitter | `Σ + εI` before factorization | Non-positive-definite covariance from numerical drift |

Default `ε = 1e-7`. These safeguards are always active and cannot be disabled — they add negligible computational cost and prevent hard crashes in edge cases.

---

### Full Pipeline Execution Order

When `GimbalLatentStabilizer` processes a latent, the equations execute in this order:

```
Input latent z
     │
     ▼
E11: Bounded Scale Map (scale_cap · tanh)
     │  Clamps coupling scales before anything else touches the flow
     ▼
E12: Dequantization Jitter (if jitter_alpha > 0)
     │  Smooths quantization artifacts before statistical estimation
     ▼
E6:  Low-Rank SVD / Covariance Estimation
     │  Builds the statistical model of the current latent batch
     ▼
E2:  Channel Diagonal Gaussian (per-channel normalization)
     │  Canonically positions the Typical Set
     ▼
E9:  Mahalanobis Distance (anomaly scoring)
     │  Flags OOD latents for telemetry
     ▼
E3:  Truncation Shrinkage (ψ)
     │  Pulls tail values toward the Typical Set
     ▼
E7/E8: Woodbury MMSE Denoiser
     │  Final closed-form denoising in low-rank subspace
     ▼
Output stabilized latent z'
```

---

### Parameter Quick Reference

| Parameter | Node | Recommended | Notes |
|---|---|---|---|
| `truncation_psi` ψ | GimbalLatentStabilizer | 0.88 | Lower = more aggressive truncation |
| `scale_cap` | GimbalLatentStabilizer | 8.0 | Reduce to 4.0 for CFG > 8 |
| `jitter_alpha` | GimbalLatentStabilizer | 0.0 | Enable at 0.01–0.05 for quantized latents |
| `rank` r | GimbalLatentStabilizer | 16 / 32 | Auto-set by architecture detection |
| `ε` | (internal) | 1e-7 | Not user-adjustable |

---

### References

- Chen, R.T.Q. et al. (2018). *Isolating Sources of Disentanglement in VAEs.* NeurIPS.
- Kingma, D.P. & Glow, P. (2018). *Glow: Generative Flow with Invertible 1×1 Convolutions.* NeurIPS.
- Woodbury, M.A. (1950). *Inverting Modified Matrices.* Memorandum Report 42.
- Karras, T. et al. (2020). *Analyzing and Improving the Image Quality of StyleGAN.* CVPR. (truncation trick)

---

*Page maintained by Form & Noise Atelier · Gimbal Node Suite documentation*
