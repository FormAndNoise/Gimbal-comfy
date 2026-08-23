# ➕ Gimbal Vector Analogy · 🎭 Gimbal Likeness Isolator · 📉 Gimbal Truncation

---

## ➕ Gimbal Vector Analogy

**Class**: `GimbalVectorAnalogy`  
**Category**: `Gimbal/Arithmetic`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT, DICT)` → `result_latent`, `analogy_meta`

### What It Does

Implements **concept arithmetic** in latent space:

```
result = C + (A − B) × strength
```

The classic example from word embedding research: `King − Man + Woman = Queen`. In latent space: `Portrait − Formal + Casual = Casual Portrait`.

Or more practically: `Architecture − Daytime + Night = Night Architecture`.

### Inputs

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `latent_A` | LATENT | — | The concept to extract from. |
| `latent_B` | LATENT | — | The reference to subtract (what you don't want). |
| `latent_C` | LATENT | — | The target to modify. |
| `strength` | FLOAT | 1.0 | Scale of the analogy transfer. |
| `mode` | enum | Standard | Standard, Normalized, or Orthogonal_Projection |
| `ortho_project` | BOOL | False | Project A−B perpendicular to C before adding (prevents double-exposure). |

### The Spatial Ghosting Problem — And the Fix

**Critical warning**: Direct vector analogy (`C + (A − B)`) on spatial diffusion latents (`[B, 4, 128, 128]`) almost always produces **phantom double-exposure ghosting**. The residual `A − B` carries Person A's entire spatial face layout — adding it to Person C stamps A's face topology onto C.

**See it fail**: `../assets/test_runs/failures_and_collisions/04_analogy_spatial_direct.png`  
**See the fix**: `../assets/test_runs/fresh_run6_exploration/05_analogy_ortho_norm_locked.png`

**The fix**: Use `mode = Orthogonal_Projection` with `ortho_project = True`. This projects the delta `A−B` perpendicular to the C vector before adding it, removing the spatial identity component.

### When It Actually Works Well

- **Global attribute arithmetic** (mood, lighting temperature, scene type) rather than localized facial or structural attributes.
- Small strength values (0.3–0.7) to prevent the delta from overwhelming C.
- Orthogonal_Projection mode for any kind of identity-sensitive transfer.

---

## 🎭 Gimbal Likeness Isolator

**Class**: `GimbalLikenessIsolator`  
**Category**: `Gimbal/Conditioning`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(MODEL, CLIP)` → `patched_model`, `patched_clip`

### What It Does

A dynamic probe for **LoRA identity vectors**. Instead of applying a LoRA at a fixed weight, the Likeness Isolator exposes three independent control axes:

| Control | Range | What It Does |
|:---|:---|:---|
| `strength` | 0.0–2.0 | Overall LoRA patch magnitude (same as normal LoRA strength). |
| `alpha` | 0.0–1.0 | Network override ratio — how much the LoRA replaces vs. blends with base weights. |
| `likeness_mask` | 0.0–1.0 | Ratio of identity-specific tokens to keep vs. suppress in the CLIP encoder. |

The key innovation is the `likeness_mask` parameter. LoRA weights mix **identity tokens** (specific to the subject) with **style tokens** (lighting, clothing, etc.). The Likeness Isolator allows you to decouple these:
- `likeness_mask = 1.0`: Full identity — subject looks exactly like the training target.
- `likeness_mask = 0.5`: Partial — keeps some identity features, loses others.
- `likeness_mask = 0.0`: Style-only — lighting and clothing style from LoRA, no identity imprint.

### Inputs

| Parameter | Type | Description |
|:---|:---|:---|
| `model` | MODEL | Base diffusion model. |
| `clip` | CLIP | CLIP encoder. |
| `lora_name` | STRING | Filename of the LoRA to load (from ComfyUI's LoRA directory). |
| `strength` | FLOAT | Overall LoRA magnitude. |
| `alpha` | FLOAT | Network override ratio. |
| `likeness_mask` | FLOAT | Identity token isolation ratio. |

### Quick Wiring

```
[Checkpoint] ─▶ MODEL, CLIP ─▶ [🎭 Likeness Isolator]
                                 lora_name = 'character_v2.safetensors'
                                 strength = 1.0
                                 alpha = 1.0
                                 likeness_mask = 0.8
                                        │
                              patched_model, patched_clip
                                        │
                               [CLIPTextEncode] ─▶ [KSampler]
```

---

## 📉 Gimbal Truncation

**Class**: `GimbalTruncation`  
**Category**: `Gimbal/Quality`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT,)` → `truncated_latent`

### What It Does

A lightweight, surgical version of the Stabilizer's E3 equation:

```
z' = μ + ψ × (z − μ)
```

Where ψ < 1 shrinks variance toward the centroid. Think of it as the "safety valve" — insert it anywhere to tame a latent without the full LAMNr pipeline overhead.

### Inputs

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `latent` | LATENT | — | Input latent. |
| `truncation_psi` | FLOAT | 0.9 | Shrinkage factor. 1.0 = identity; <1.0 = pull toward mean. |
| `channel_adaptive` | BOOL | True | Compute μ per-channel instead of global. More accurate. |
| `mu_override` | LATENT (optional) | — | Use a different centroid than the batch mean. |

### When to Use Truncation vs. Stabilizer

| Use `GimbalTruncation` when... | Use `GimbalLatentStabilizer` when... |
|:---|:---|
| You want a lightweight single operation | You need the full quality stack (scale cap + truncation + Woodbury) |
| Adding to a tight, low-VRAM workflow | You're dealing with serious artifacts (black outlines, posterization) |
| ψ=0.9 gentle cleanup only | CFG was high and you're seeing extreme values |
| You already have a Stabilizer downstream | You don't have a Stabilizer in the pipeline |

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
