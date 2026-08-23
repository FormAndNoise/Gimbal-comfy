<!-- Gimbal Node Suite — Form & Noise Atelier -->
<!-- Navigate latent space with precision flight instruments, not lottery prompts. -->

# 🌉 Workflow 02: Text-Steered Lighting

![Text-Steered Lighting Hero](../../assets/test_runs/02_text_steered/02_text_steered_portrait.jpg)

> **VibeCheck** 🟢 Stabilized &nbsp;|&nbsp; Category: `Gimbal/Navigation` + `Gimbal/Stabilizer` &nbsp;|&nbsp; Primary Nodes: **GimbalCrossModalBridge · GimbalCompass_Pro · GimbalLatentStabilizer**

---

## Overview

Text-Steered Lighting lets you describe a lighting environment in plain English — "golden hour warm cinematic soft" — and *surgically* apply that lighting direction to an existing latent without touching the foreground geometry. The **Cross-Modal Bridge** converts your keywords into a latent-space direction vector. The **Compass Pro** projects that vector onto the image latent using Orthogonal Projection mode, which pushes the lighting while preserving the image's own structure. The **Latent Stabilizer** cleans up any variance spikes before the final KSampler commits the result.

This workflow is ideal for product photography, portrait retouching, and architectural visualization — anywhere you need precise environmental control without re-generating the subject.

---

## 🖼️ What You'll Create

- A re-lit version of an input image that preserves subject silhouette, proportions, and material identity
- Controllable atmospheric style through plain-language keyword presets
- A stable, artifact-free output ready for final sampling at low denoise (0.55–0.65)

---

## 🗺️ Node Chain Diagram

```
[Load Image / VAE Encode] ─────────────────────── BASE_LATENT
                                                        │
[🌉 GimbalCrossModalBridge]                             │
  ├── instruction  = "golden hour warm cinematic soft"  │
  └── mode         = Keyword_Heuristics                 │
  ↓ direction_latent (the lighting vector)              │
                                                        │
[🧭 GimbalCompass_Pro]                                  │
  ├── base_latent   = BASE_LATENT ◄──────────────────────┘
  ├── target_latent = direction_latent
  ├── origin_latent = [Empty Latent]  ← zero-delta reference
  ├── strength      = 1.5
  └── mode          = Orthogonal_Projection
  ↓ steered_latent

[🛡️ GimbalLatentStabilizer]
  ├── truncation_psi = 0.88
  └── scale_cap      = 8.0
  ↓ stable_latent

[KSampler]  (CFG=3.8, denoise=0.60, steps=20)
  ↓
[VAE Decode] → [Save Image]
```

---

## ⚙️ Settings That Work

Verified with FLUX.1-dev, SDXL 1.0 Base, and SD 1.5.

| Parameter | Optimal | Notes |
|---|---|---|
| Compass `strength` | `1.5` | Orthogonal Projection is gentler than Standard — 1.5 gives visible impact |
| Compass `mode` | `Orthogonal_Projection` | Projects the lighting vector *orthogonal* to the existing image structure |
| Stabilizer `truncation_psi` | `0.88` | Pulls 12% of variance toward mean — eliminates fringe artifacts and noise spikes |
| Stabilizer `scale_cap` | `8.0` | Hard cap prevents any single channel from blowing out |
| KSampler `cfg` | `3.8` | Low — the Bridge already did the semantic steering; high CFG re-introduces prompt dominance |
| KSampler `denoise` | `0.55–0.65` | Enough to commit the new lighting without re-synthesizing geometry |
| KSampler `steps` | `20` | 20 is sufficient at this denoise level |

---

## 💡 Verified Lighting Presets

Use these keyword strings in the Cross-Modal Bridge `instruction` field:

| Style | Keywords | Sample Output |
|---|---|---|
| Golden Hour | `warm golden cinematic soft` | ![Golden Sunset](../../assets/test_runs/02_text_steered/03_golden_sunset.png) |
| Cyberpunk Midnight | `dark cool neon moody cinematic` | ![Cyberpunk](../../assets/test_runs/02_text_steered/02_TextSteered_02_cyberpunk_midnight_00002_.png) |
| Daylight Coastal | `bright crisp cool ethereal` | ![Daylight](../../assets/test_runs/02_text_steered/01_daylight_coastal.png) |
| Bioluminescent | `cool neon ethereal underwater vivid` | ![Bioluminescent](../../assets/test_runs/fresh_run6_exploration/02_watch_steered_bioluminescent.png) |
| Volcanic Forge | `warm fire gritty dark punchy` | ![Volcanic Forge](../../assets/test_runs/fresh_run6_exploration/02_watch_steered_volcanic_forge.png) |

---

## 🔧 Step-by-Step Wiring Instructions

1. **Load your input image.** Use a Load Image node, then wire it through a VAE Encode node (using the checkpoint's VAE) to produce `BASE_LATENT`.
2. **Add GimbalCrossModalBridge.**
   - Set `instruction` to one of the keyword presets above (or write your own descriptive phrase).
   - Set `mode` to `Keyword_Heuristics`.
   - The output `direction_latent` is your lighting vector.
3. **Add GimbalCompass_Pro.**
   - `base_latent` → `BASE_LATENT` (your encoded input image)
   - `target_latent` → `direction_latent` (from Cross-Modal Bridge)
   - `origin_latent` → connect an **Empty Latent Image** node (same resolution as your image). This is the zero-delta reference that grounds the projection.
   - `strength` → `1.5`
   - `mode` → `Orthogonal_Projection`
4. **Add GimbalLatentStabilizer.**
   - Wire `steered_latent` from Compass Pro into the Stabilizer input.
   - Set `truncation_psi=0.88`, `scale_cap=8.0`.
5. **Add KSampler.**
   - `latent_image` → `stable_latent` from Stabilizer
   - `cfg=3.8`, `denoise=0.60`, `steps=20`
   - Use the same positive conditioning as your original image description (or a neutral prompt).
6. **Add VAE Decode → Save Image.** Wire `VAE` from the checkpoint.
7. **Queue the prompt.** Your re-lit image appears in the output folder with the original silhouette intact.

> **Tip:** For even tighter geometry preservation, reduce `denoise` to `0.50`. At `0.50`, the KSampler corrects noise statistics without any structural hallucination.

---

## 🖼️ Gallery

### Baseline vs. Steered (Watch Photography)

| Baseline (Daylight) | Bioluminescent | Volcanic Forge |
|---|---|---|
| ![Baseline](../../assets/test_runs/fresh_run6_exploration/02_watch_daylight_baseline.png) | ![Bioluminescent](../../assets/test_runs/fresh_run6_exploration/02_watch_steered_bioluminescent.png) | ![Volcanic Forge](../../assets/test_runs/fresh_run6_exploration/02_watch_steered_volcanic_forge.png) |

### Automotive Use Case

| Studio Control | Cyberpunk Steered |
|---|---|
| ![Car Studio](../../assets/test_runs/architectural_showcases/02_v2_car_studio_ctrl.png) | ![Car Cyberpunk](../../assets/test_runs/architectural_showcases/02_v2_car_steered_cyberpunk.png) |

---

## 🛠️ Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Lighting change is too subtle | `strength` too low | Raise Compass strength to `2.0`–`2.5` (watch for artifacts above 3.0) |
| Subject geometry is distorted / morphed | `denoise` too high or wrong Compass mode | Lower `denoise` to `0.50`; confirm mode is `Orthogonal_Projection` not `Standard` |
| Black outlines or fringe artifacts appear | Variance spike post-steering | Ensure GimbalLatentStabilizer is inserted between Compass and KSampler |
| Lighting looks generic / not matching keyword | Bridge mode on wrong setting | Confirm `mode = Keyword_Heuristics`; try rephrasing keywords (simpler is better) |
| Output has extreme color shift (looks inverted) | `strength` too high | Cap strength at `2.0`; the Orthogonal mode amplifies the vector — values above 3.0 can flip sign |
| Subject skin tone or material changes | `denoise` too high | Target `0.55`–`0.65`; above `0.75` the KSampler begins re-synthesizing materials |

---

## 🔬 Power User Notes

### Why Orthogonal Projection Preserves Geometry

Standard Compass mode computes a weighted sum of the base and target latents — it physically moves the latent toward the target. **Orthogonal Projection** instead decomposes the target vector into a component *parallel* to the base latent and a component *perpendicular* to it, then applies only the perpendicular (orthogonal) component. Because the image's own structure lives in the parallel direction, it is untouched. Only the directions the base image does not already occupy — i.e., the lighting information — are added.

Mathematically:
```
v_orthogonal = direction - (direction · base / ||base||²) × base
steered = base + strength × v_orthogonal
```

### Stacking Multiple Lighting Vectors

Run the Cross-Modal Bridge twice with different keyword strings, then average the two `direction_latent` outputs using a Latent Composite or simple math node before feeding into Compass Pro. This allows blended lighting environments (e.g., 60% golden hour + 40% neon).

### Keyword Writing Tips

The Bridge's `Keyword_Heuristics` mode maps individual words to pre-learned latent directions. The most reliable words are **temperature descriptors** (warm/cool), **tonal descriptors** (dark/bright/muted), **cinematic references** (cinematic/punchy/flat), and **energy descriptors** (crisp/soft/gritty). Avoid full sentences — three to five single words outperform a long phrase.

---

## 📁 Workflow Files

| Format | Path |
|---|---|
| ComfyUI drag-and-drop | [`workflows/ui/Gimbal_02_TextSteered.json`](../../workflows/ui/Gimbal_02_TextSteered.json) |
| API / FLUX variant | [`workflows/api_flux/API_FLUX_Gimbal_02_TextSteered.json`](../../workflows/api_flux/API_FLUX_Gimbal_02_TextSteered.json) |
| API / SDXL variant | [`workflows/api_sdxl/API_Gimbal_02_TextSteered.json`](../../workflows/api_sdxl/API_Gimbal_02_TextSteered.json) |

---

*Gimbal Node Suite — Form & Noise Atelier*
*Navigate latent space with precision flight instruments, not lottery prompts.*
