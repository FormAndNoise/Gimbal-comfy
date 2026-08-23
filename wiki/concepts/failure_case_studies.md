# 💥 Failure Case Studies: What We Broke and How We Fixed It

> *"Every artifact is a lesson, every crash has a telemetry log, and every black box recorder tells a story. We kept detailed notes."*

This page documents real failures from Gimbal development runs — not polished post-mortems, but honest breakdowns of what went wrong, why it went wrong at a mechanistic level, and exactly what changes fixed it. If you're pushing Gimbal to its limits, this is required reading.

---

## Overview

| # | Failure | Root Cause | Severity |
|---|---|---|---|
| 1 | Mid-denoise spatial SLERP → rock spires, comic outlines | Attention crystallization | 🔴 Major artifact |
| 2 | High CFG variance frying → plastic skin, wireframe hair | Gaussian tail overflow | 🔴 Major artifact |
| 3 | Vector analogy → double-exposure face ghost | Spatial residual carries full face layout | 🟡 Architectural misuse |
| 4 | Unregistered keyword → zero change at low denoise | Token miss → Δ=0 | 🟡 Silent failure |

---

## Case 1: Mid-Denoise Spatial SLERP (Run 4)

### The Failure

During Run 4 testing of `GimbalCompass_Pro`, we attempted to inject a SLERP blend at **Step 8 of 25** — roughly 32% through the denoising process — to redirect a forest scene toward a mountain scene.

**Results:** Tree branches became vertical rock spires. Moss patches transformed into boulders with hard black comic-book outlines and severe posterization. The image looked like two concepts had been forcibly welded together with a soldering iron.

![Mid-denoise SLERP failure: tree branches became rock spires with black outlines](../../assets/test_runs/failures_and_collisions/01_slerp_35pct_00001_.png)

### Root Cause

By Step 8 of 25, the UNet's attention heads have already **crystallized spatial feature maps**. Each attention layer has committed to a spatial layout: "this region is a vertical woody object," "this region is ground cover," etc.

When you force a vector blend at this point, the model receives a fundamentally contradictory signal:
1. The *content* of the latent now points toward mountain geometry
2. The *attention heads* are still optimized for forest spatial organization

The attention heads must rapidly re-skin every high-frequency boundary using the new content signal but the old spatial scaffolding. The result is exactly what you'd expect from that mismatch: geometry at the wrong scale, edges that don't match the underlying object structure, and the telltale black-outline posterization where the decoder is receiving conflicting high-contrast gradients.

**Mathematical framing:** At step 8, the latent is no longer on the isotropic Gaussian Typical Set — it has been partially denoised into a structured, low-entropy state. SLERP geometry assumes isotropic Gaussian statistics (the Typical Set). Applying SLERP to a partially structured latent violates this assumption, producing off-manifold blends.

### The Fix: Step 0 Noise Blending

**Perform SLERP on Step 0 initial Gaussian noise, before denoising begins.** At Step 0, the latent is still an unstructured sample from the Typical Set. SLERP's geometric assumptions hold perfectly. The UNet then denoises from the blended starting point, naturally synthesizing a coherent blend without any attention-head conflict.

The results across blend ratios:

| Blend ratio | Result |
|---|---|
| 35% | Predominantly forest with mountain influence |
| 50% | Balanced hybrid — coherent, no artifacts |
| 65% | Predominantly mountain, forest traces present |

![Step 0 SLERP at 35% — clean blend, no artifacts](../../assets/test_runs/fresh_run6_exploration/01_slerp_noise_35pct.png)

![Step 0 SLERP at 50% — balanced coherent hybrid](../../assets/test_runs/fresh_run6_exploration/01_slerp_noise_50pct.png)

![Step 0 SLERP at 65% — mountain-dominant, clean](../../assets/test_runs/fresh_run6_exploration/01_slerp_noise_65pct.png)

### Rule Derived

> **Never apply SLERP after Step 0 on spatial latents.** `GimbalCompass_Pro` enforces this by design — its SLERP injection operates exclusively on the initial noise tensor. The mid-denoise blend option was removed after this run.

---

## Case 2: High CFG Variance Frying (Run 4)

### The Failure

Simultaneously with Case 1 testing, we ran a portrait steering workflow at **CFG 6.5–7.5** with **denoise 0.72–0.80**. Subject was a portrait; target was a dramatically lit version of the same subject.

**Results:** Skin became plastic and posterized, completely losing pore and texture detail. Hair became an embossed black wireframe — a characteristic "engraving" artifact where each strand was outlined in hard black with no internal color information.

![High CFG variance frying: plastic skin and wireframe hair artifacts](../../assets/test_runs/failures_and_collisions/02_steered_denoise_072_00001_.png)

### Root Cause

At CFG 6.5+, the guidance mechanism pushes the denoised prediction strongly away from the unconditional output toward the conditional target. This works well in the bulk of latent space, but it systematically drives **activations toward the tails of the Gaussian distribution** — values at 4σ, 5σ, or beyond.

The VAE decoder was trained on latents with normal-range activations (~±3σ). When it receives values at 5σ+, it has two failure modes:

1. **Hard clipping:** values beyond the decoder's effective dynamic range get clipped to maximum contrast, creating the black-boundary wireframe effect
2. **Posterization:** in the mid-range, the decoder produces color-band artifacts because its learned transfer function becomes poorly conditioned in the high-activation regime

This is a VAE decoder generalization failure — not a prompt issue, not a sampler issue. The prompt and sampler are doing exactly what they should. The problem is that the resulting latent values fall outside the distribution the decoder was trained on.

**Gaussian tail quantification:** At denoise 0.75 + CFG 7.0, we measured peak channel activations of 4.8σ in Ch 2–3 (chroma channels). The VAE's training data had <0.01% probability mass at those values. The decoder is extrapolating, not interpolating.

### The Fix

Two-part correction:

**1. Insert `GimbalLatentStabilizer` with ψ=0.88, scale_cap=8.0**

This applies truncation shrinkage (E3) before the latent reaches the decoder:
```
z' = μ + 0.88 · (z - μ)
```
Tail values at 4.8σ are pulled back to approximately 4.2σ — still expressive, but within the decoder's trained dynamic range.

**2. Drop steering CFG to 3.8, reduce denoise to 0.45–0.60**

High CFG is only needed to push the guidance signal through a large latent modification. With `GimbalCrossModalBridge` handling the material-channel steering precisely, the global CFG can be much lower. The Bridge applies targeted channel-specific deltas; the KSampler CFG only needs to be high enough to overcome noise, not to overcome an off-target latent.

The combination: precise steering at low CFG + Stabilizer truncation = dramatic lighting without any posterization or wireframe artifacts.

### Rule Derived

> **CFG above 6.0 requires `GimbalLatentStabilizer` in the pipeline.** There are no exceptions. The Stabilizer's computational cost is negligible; the artifact prevention is significant. Set ψ=0.88 as the default; dial down to ψ=0.85 only for CFG 8+.

---

## Case 3: Vector Analogy Double-Exposure Collision

### The Failure

We tested the vector analogy technique for **attribute transfer** — specifically, glasses transfer:

```
z_out = z_C + (z_A - z_B)
```

Where:
- `z_A` = Person A wearing glasses
- `z_B` = Person A without glasses (same identity, same pose)
- `z_C` = Person C without glasses (different identity)

**Hypothesis:** Δ = z_A - z_B should isolate "glassesness" as a residual vector.

**Results:** Severe **phantom double-exposure ghosting** — Person A's face was visibly superimposed over Person C's face. The glasses were present, but so was Person A's entire facial geometry.

![Vector analogy failure: double-exposure ghosting — two faces superimposed](../../assets/test_runs/failures_and_collisions/04_analogy_spatial_direct.png)

### Root Cause

This is a fundamental architectural constraint of **spatial latent grids**, not a parameter-tuning issue.

In a 2D spatial latent grid (128×128 for SDXL), every location in the grid has spatial correspondence to the image. Channel values at position (h, w) encode information about the *image region* at that spatial location.

The residual Δ = z_A - z_B is not a pure "glasses attribute vector." It is the **pixel-wise difference** between two spatially grounded latent grids. At every spatial location, Δ encodes the *full difference* between the two images at that location — which includes:

- The glasses shape and position ✓ (intended)
- Person A's jawline difference ✗ (unintended)
- Person A's eye geometry difference ✗ (unintended)
- Person A's forehead structure difference ✗ (unintended)

Adding Δ to z_C literally stamps Person A's face layout differences onto Person C at every spatial location. The "attribute isolation" assumption holds only in **spatially global** representations (like CLIP embeddings or w-space in StyleGAN) — not in spatial latent grids.

### The Fix

Localized attribute transfer on spatial latents requires one of three approaches:

**Option A: `GimbalCrossModalBridge` (semantic guidance)**
Instead of arithmetic residuals, use keyword-driven cross-modal guidance. The Bridge applies semantically grounded material/attribute changes without spatial face-layout bleeding.

**Option B: Step 0 blending via `GimbalCompass_Pro`**
Blend the initial noise of two Step-0 generations rather than performing arithmetic on denoised latents.

**Option C: Spatial mask bounding**
If arithmetic residuals must be used, mask the delta application to a tight bounding region around the glasses area using a spatial mask node. Outside the mask, Δ is zeroed — limiting the bleed to the intended region.

The corrected result using orthonormal-locked analogy (Option B approach):

![Fixed analogy with orthonormal locking — clean glasses transfer, no face bleed](../../assets/test_runs/fresh_run6_exploration/05_analogy_ortho_norm_locked.png)

### Rule Derived

> **Vector analogy arithmetic on spatial latent grids transfers spatial face layout, not just semantic attributes.** Use `GimbalCrossModalBridge` for attribute transfer. Reserve latent arithmetic for global concepts (style, lighting class, color temperature) that are spatially uniform across the image.

---

## Case 4: Unregistered Keyword Delta Zero Collapse

### The Failure

A user testing `GimbalCrossModalBridge` in `Keyword_Heuristics` mode passed the keyword phrase:

```
'mirror polished liquid chrome'
```

to steer a chair product render toward a chrome finish, at **denoise 0.38**.

**Results:** Three consecutive generations with **zero visible change** — the chair looked identical to the unsteered baseline. No chrome, no metallic quality, no response whatsoever.

### Root Cause

`Keyword_Heuristics` mode works by tokenizing the input phrase and looking up each token against the internal `LATENT_SIGNATURES` dictionary:

```python
LATENT_SIGNATURES = {
    'metallic': [...],
    'chrome':   [...],   # ← NOT in the dict
    'warm': [...],
    'cool': [...],
    # ... etc.
}
```

At the time of this run, `'chrome'` was **not a registered token** in `LATENT_SIGNATURES`. Neither were `'mirror'`, `'polished'`, or `'liquid'` as standalone tokens. The tokenizer found no matches and returned Δ = 0 — a zero delta vector.

With Δ = 0, no latent modification occurs. At denoise 0.38, the UNet operates on an unmodified latent with a conditioned prompt that doesn't include chrome keywords. The output is simply the baseline generation. **The node silently did nothing.**

This is a silent failure: no error, no warning, no indication to the user that the keyword was unrecognized.

### The Fix

**Use registered keyword signatures.** The `LATENT_SIGNATURES` dictionary maps semantic intent through token clusters, not raw English words. To achieve chrome:

```
'cool sharp crisp monochrome bright cold'  →  Chrome signature
```

To achieve the oxblood velvet alternative tested in the same run:

```
'warm saturated vivid dark moody fire'     →  Oxblood Velvet signature
```

Both produce strong, unambiguous material transformations:

![Chrome steering result using registered keywords at denoise 0.65](../../assets/test_runs/fresh_run6_exploration/09_chair_steered_liquid_chrome.png)

![Oxblood velvet steering result using registered keywords at denoise 0.65](../../assets/test_runs/fresh_run6_exploration/09_chair_steered_oxblood_velvet.png)

**Additionally:** Increase denoise to **0.65** when using the Bridge. At 0.38, the UNet's latent is already substantially denoised and structured — even a correct Δ may not be large enough to overcome the existing latent's inertia. Pair with `GimbalLatentStabilizer` ψ=0.88 to prevent variance frying at the higher denoise.

**Planned fix (backlog):** Add a warning log output when zero tokens match `LATENT_SIGNATURES`, so users see an explicit message rather than silent no-op behavior.

### Rule Derived

> **`GimbalCrossModalBridge` uses token matching, not natural language understanding.** Always use registered keyword clusters from the [keyword signature reference](../nodes/crossmodal_bridge.md#keyword-signatures). Set denoise ≥ 0.60 when using the Bridge for material transformation. At denoise < 0.45, Bridge influence is weak even with correct keywords.

---

## Summary Table

| Case | What looked wrong | Why it happened | What fixed it |
|---|---|---|---|
| 1 · Mid-denoise SLERP | Geometry corruption, comic outlines | Attention crystallization at step 8 | Move SLERP to Step 0 initial noise |
| 2 · CFG Frying | Plastic skin, wireframe hair | Tail activations (4.8σ) exceed VAE range | GimbalLatentStabilizer ψ=0.88 + lower CFG |
| 3 · Analogy bleed | Double-exposure face ghosting | Spatial Δ carries full face layout | CrossModalBridge or spatial mask |
| 4 · Zero delta | No change, silent failure | Unregistered token → Δ=0 | Use registered keyword clusters + denoise ≥ 0.65 |

---

*Page maintained by Form & Noise Atelier · Gimbal Node Suite documentation*
