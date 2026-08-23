# 🏔️ SLERP vs LERP: Why the Arc Matters

*How to move between two latent coordinates without falling through the floor.*

---

## The Ball Analogy

Picture a hollow rubber ball. Two points on the surface — one on the left side, one on the right. You want to get from A to B.

**LERP** says: drill straight through the inside. Shortest Euclidean path. But the interior of the ball is hollow — there's nothing there. You pass through a void, and when you emerge on the other side, you're not quite on the surface anymore.

**SLERP** says: walk along the surface of the ball. Slightly longer path, but every step of the way you're on the actual material of the sphere — never cutting through dead space.

Now replace the rubber ball with a 65,536-dimensional hypersphere. The surface is the **Typical Set** — the thin shell where every valid, sharp, coherent image lives. The hollow interior is the low-probability dead zone where images are foggy, washed out, and confused.

> **LERP cuts through the dead zone. SLERP walks the surface.**

---

## The Failure: What Mid-Denoise SLERP Looks Like

Not all SLERPs are equal. The *geometry* of the interpolation matters, but so does *when in the denoising process* you apply it.

### What went wrong

![Mid-Denoise SLERP Failure: Rock Spires](../../assets/test_runs/failures_and_collisions/01_slerp_35pct_00001_.png)

*Above: SLERP applied at Step 8 of 20. Concept A was "cyber spire cityscape", Concept B was "redwood forest". The tree branches became rock spires. Moss turned to boulders. The shapes are crystallized from Concept A's geometry, but the surface finish is Concept B's material — a geometric re-skinning collision.*

This failure was documented during Gimbal development and is what motivated the "Step 0 only" guidance in `GimbalCompass_Pro`. Here is what happened, step by step:

1. The UNet began denoising from random noise toward Concept A.
2. By Step 8 of 20, the attention heads had already **committed to object boundaries** — the branch angles, the spire heights, the silhouette topology were crystallized in the latent structure.
3. Applying SLERP at Step 8 forcibly blended in the geometric structure of Concept B.
4. But the spatial structure (Channels 0–1) and texture/color channels (Channels 2–3) were blended at the same ratio — the geometry from one concept and the surface material from the other were incompatible.
5. Result: rock-shaped objects with forest surface normals. Black outlining from conflicting edge definitions.

---

## The Fix: Step 0 Noise SLERP

The correct approach is to apply SLERP **before any denoising has begun** — at Step 0, when the tensor is still pure Gaussian noise with no spatial structure at all.

| t = 0.35 | t = 0.50 | t = 0.65 |
|:--------:|:--------:|:--------:|
| ![SLERP t=0.35](../../assets/test_runs/fresh_run6_exploration/01_slerp_noise_35pct.png) | ![SLERP t=0.50](../../assets/test_runs/fresh_run6_exploration/01_slerp_noise_50pct.png) | ![SLERP t=0.65](../../assets/test_runs/fresh_run6_exploration/01_slerp_noise_65pct.png) |
| Closer to Concept A | Balanced blend | Closer to Concept B |

*These are all clean, sharp, coherent images. The geodesic arc at Step 0 — before the UNet has committed to any geometry — produces smooth, valid transitions at every t value.*

The key insight: at Step 0, the UNet's attention hasn't crystallized any spatial structure yet. The SLERP blend simply produces a new valid starting-noise tensor that is a geodesic combination of the two source noise vectors. The UNet then denoises this combined noise from scratch, and it lands on a coherent blend because both noise sources were valid members of the Typical Set — and their arc-midpoint is also a valid member.

---

## ⚙️ Power User Section: The Mathematics

### LERP and Its Norm Collapse

The standard linear interpolation formula is:

```
z(t) = (1 - t) · z₁ + t · z₂
```

At the midpoint `t = 0.5`, the Euclidean norm of the result is:

```
‖z_lerp(0.5)‖ = √( (0.5·‖z₁‖)² + (0.5·‖z₂‖)² )
              = (√2 / 2) · ‖z₁‖
              ≈ 0.707 · r
```

(Assuming `‖z₁‖ ≈ ‖z₂‖ ≈ r`, and that `z₁ · z₂ ≈ 0`, which is approximately true for high-dimensional random vectors.)

This means the midpoint of a LERP has only **70.7% of the norm** of either endpoint. It has drifted from radius `r` (on the Typical Set shell) to radius `0.707r` (inside the dead zone). The diffusion model interprets this low-norm region as **low-variance**, producing foggy, over-smoothed, low-contrast outputs.

### μ-Centered SLERP (Equation E4)

Gimbal's SLERP formula centers everything on the empirical mean `μ` of the latent batch, not the origin. This matters because the latent space in a trained model is not centered at zero — the distribution has a learned mean offset. Operating in the translated coordinate frame `(z - μ)` ensures you are walking on the correct hypersphere shell:

```
z_slerp(t) = μ + r(t) · [ sin((1-t)·ω) / sin(ω) · û  +  sin(t·ω) / sin(ω) · v̂ ]
```

Where:
- `û = (z₁ - μ) / ‖z₁ - μ‖`  — unit direction from μ to z₁
- `v̂ = (z₂ - μ) / ‖z₂ - μ‖`  — unit direction from μ to z₂
- `ω = arccos(û · v̂)`           — angle between the two directions
- `r(t) = (1 - t)·‖z₁ - μ‖ + t·‖z₂ - μ‖` — linearly interpolated radius (preserves both endpoint norms)

This formula guarantees:
1. At `t=0`: returns `z₁` exactly.
2. At `t=1`: returns `z₂` exactly.
3. For all `t ∈ [0,1]`: the norm relative to `μ` smoothly varies between `‖z₁-μ‖` and `‖z₂-μ‖` — **no norm collapse**.
4. The path is a great-circle geodesic on the hypersphere shell centered at `μ`.

### Why Step 0 vs Step 8: The Crystallization Problem

The UNet is a hierarchical attention network with multiple spatial resolution levels. During early denoising steps:

- **Steps 0–4**: Coarse global layout is being established. Attention maps are diffuse and uncommitted.
- **Steps 5–10**: Object boundaries begin to crystallize. Attention heads commit to edge locations, occlusion relationships, and large-scale geometry. The **spatial frequency channels** (Channels 0–1 in SDXL) become the first to lock in.
- **Steps 11+**: Fine texture and surface detail are being refined. Channels 2–3 (chroma, specular) are actively driven.

Applying SLERP at Step 8 forces a blend **after** Channels 0–1 have already crystallized geometry for Concept A, but **before** Channels 2–3 have committed surface material. The result is a geometric structure from Concept A with a surface finish blended from Concept B — incoherent at the physical level (a tree branch cross-section cannot naturally resolve the surface normals of a granite boulder).

At **Step 0**, neither channel set has committed to anything. The tensor is pure independent Gaussian noise: `z ~ N(μ, σ²I)`. A SLERP on this noise vector produces a new noise vector that is itself a valid sample from the same distribution (because the Typical Set is preserved). The UNet then denoises this combined noise as a unified starting point, and the geometry and material co-evolve coherently from a single starting condition.

### Numerical Safeguards

The SLERP formula has two known failure modes that must be handled:

| Condition | Problem | Gimbal's Fix |
|:----------|:--------|:-------------|
| `ω ≈ 0` (nearly identical vectors) | `sin(ω) ≈ 0`, division by zero | Automatic fallback to LERP (result is the same at this scale) |
| `ω ≈ π` (nearly antipodal vectors) | `sin(ω) ≈ 0`, numerical instability | Automatic LERP fallback with warning |
| `arccos` argument outside `[-1, 1]` | Domain error, NaN | Argument is clamped to `[-1 + ε, 1 - ε]` before calling `arccos` |
| Near-zero norm `‖z₁ - μ‖ ≈ 0` | Division by zero in unit vector | Norm is floored to `ε = 1e-8` before division |

These are implemented in `gimbal_latent_math.py` as the **E13 Numerical Safeguard Stack** and apply to all interpolation paths in Gimbal.

---

## 🎛️ The Three SLERP Modes in Compass Pro

`GimbalCompass_Pro` exposes three interpolation modes:

| Mode | Internal Name | What It Does |
|:-----|:-------------|:-------------|
| **Slerp** | `Slerp` | μ-centered SLERP (Equation E4). Recommended default. Preserves Typical Set geometry. |
| **Slerp Origin** | `Slerp_Origin` | Legacy zero-centered SLERP. Equivalent to E4 when `μ = 0`. Kept for backward compatibility. |
| **Normalized** | `Normalized` | Projects the blend onto the unit hypersphere. Useful when you want uniform norm across all t. |

**When to use each:**

- Use `Slerp` (μ-centered) for almost all blending tasks — it is geometrically correct and norm-preserving.
- Use `Slerp_Origin` only when working with latents from models known to have `μ ≈ 0` (some early SD 1.5 VAE checkpoints).
- Use `Normalized` when you are chaining multiple SLERP operations and need to ensure uniform variance across the chain.

---

## 🔗 Related Concepts

- [What Is Latent Space?](./latent_space.md) — The manifold your model lives in
- [The Gaussian Annulus](./gaussian_annulus.md) — Why the dead zone exists
- [μ-Centered Geometry](./gaussian_annulus.md) — Why we translate to the mean before interpolating
- [LAMNr Mathematical System](./lamnr_framework.md) — Full E1–E13 equation reference including E4

---

*Part of the **Gimbal Node Suite** documentation · Form & Noise Atelier*
