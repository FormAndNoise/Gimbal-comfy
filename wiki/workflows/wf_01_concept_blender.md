<!-- Gimbal Node Suite — Form & Noise Atelier -->
<!-- Navigate latent space with precision flight instruments, not lottery prompts. -->

# 🧭 Workflow 01: Concept Blender

![Concept Blender Hero](../../assets/test_runs/01_concept_blender/01_concept_blend_forest_mountain.jpg)

> **VibeCheck** 🟢 Stabilized &nbsp;|&nbsp; Category: `Gimbal/Navigation` &nbsp;|&nbsp; Primary Node: **GimbalCompass_Pro**

---

## Overview

The Concept Blender workflow is your simplest entry point into Gimbal's latent navigation system. It takes two independently sampled latents — each representing a fully-formed concept — and moves along the **geodesic path** between them on the latent sphere. The result is a synthesized third image that genuinely borrows character from both parents, rather than a naive pixel-blend.

This is not prompt averaging. It is **spherical linear interpolation (Slerp)** across the noise manifold, which means the midpoint preserves image variance and stays in a high-probability region of the diffusion model's prior.

---

## 🖼️ What You'll Create

- A blend image combining two semantically distinct concepts (e.g. cyberpunk spire + redwood forest)
- A controllable midpoint on the geodesic arc between Concept A and Concept B
- Optionally: a sweep series at multiple strengths (0.25 → 0.50 → 0.75) for animation or selection

---

## 🗺️ Node Chain Diagram

```
[Checkpoint Loader] ─────────────────────────── MODEL, CLIP, VAE
                                                     │
              ┌──────────────────────────────────────┤
              │                                      │
[CLIP Text Encode A: "cyberpunk spire"] → COND_A    │
[CLIP Text Encode B: "redwood forest"]  → COND_B    │
[Empty Latent Image]                    → LATENT_SEED│
              │                                      │
[KSampler A]  (seed_A, denoise=1.0, COND_A) ────────┼──→ LATENT_A
[KSampler B]  (seed_B, denoise=1.0, COND_B) ────────┘──→ LATENT_B
              │
              ▼
[🧭 GimbalCompass_Pro]
  ├── base_latent   = LATENT_A
  ├── target_latent = LATENT_B
  ├── origin_latent = LATENT_A   ← (or Empty Latent for directional push)
  ├── strength      = 0.50       ← blend ratio
  └── mode          = Slerp
              │
              ▼  latent_out
[KSampler]  (denoise=0.90, CFG=5.5, steps=25)
              │
              ▼
[VAE Decode] → [Save Image]
```

---

## ⚙️ Settings That Work

Verified on FLUX.1-dev and SDXL 1.0 Base.

| Parameter | Optimal | Range | Notes |
|---|---|---|---|
| `strength` | `0.50` | 0.0 – 1.0 | 0 = pure A · 0.5 = midpoint · 1.0 = pure B |
| `mode` | `Slerp` | Slerp, Normalized | Slerp preserves variance at midpoint — use this |
| KSampler `denoise` | `0.90` | 0.85 – 1.0 | Higher gives the model more latitude to synthesize the blend |
| KSampler `cfg` | `5.5` | 4.5 – 7.0 | Keep moderate — high CFG fries mid-blend latents |
| KSampler `steps` | `25` | 20 – 30 | Standard quality range |

---

## 🔧 Step-by-Step Wiring Instructions

1. **Load a checkpoint.** Connect `MODEL`, `CLIP`, and `VAE` outputs to downstream nodes.
2. **Add two CLIP Text Encode nodes.** Wire each to `CLIP`. Enter your two concept prompts — keep them semantically distinct for best results.
3. **Add an Empty Latent Image node.** Set your desired resolution (e.g. 1024×1024 for SDXL).
4. **Add KSampler A.** Wire `MODEL`, `COND_A` (positive), and the Empty Latent. Set `denoise=1.0`. Use a unique seed. Connect output to `LATENT_A`.
5. **Duplicate KSampler A → KSampler B.** Switch positive conditioning to `COND_B`. Use a *different* seed. Connect output to `LATENT_B`.
6. **Add GimbalCompass_Pro node.**
   - `base_latent` → `LATENT_A`
   - `target_latent` → `LATENT_B`
   - `origin_latent` → `LATENT_A` (same as base for pure blending)
   - `strength` → `0.50`
   - `mode` → `Slerp`
7. **Add final KSampler.** Wire `MODEL`, a conditioning (either A, B, or a third neutral prompt), and `latent_out` from Compass. Set `denoise=0.90`, `cfg=5.5`, `steps=25`.
8. **Add VAE Decode → Save Image.** Wire `VAE` from the checkpoint and `SAMPLES` from the final KSampler.
9. **Queue the prompt.** Your blend image will appear in the output folder.

> **Tip:** To generate a blend series, duplicate the Compass Pro node and create variants at `strength=0.25`, `0.50`, and `0.75`. Connect all three to separate KSamplers sharing the same seed for a comparable sweep.

---

## 🖼️ Gallery

### Control Concepts

| Concept A | Concept B |
|---|---|
| ![Cyberpunk Spire](../../assets/test_runs/fresh_run6_exploration/01_ctrl_A_cyber_spire.png) | ![Redwood Forest](../../assets/test_runs/fresh_run6_exploration/01_ctrl_B_redwood_forest.png) |

### Blend Series (35% → 50% → 65%)

| Strength 0.35 | Strength 0.50 | Strength 0.65 |
|---|---|---|
| ![35%](../../assets/test_runs/fresh_run6_exploration/01_slerp_noise_35pct.png) | ![50%](../../assets/test_runs/fresh_run6_exploration/01_slerp_noise_50pct.png) | ![65%](../../assets/test_runs/fresh_run6_exploration/01_slerp_noise_65pct.png) |

### Composite Output

![Concept Blender Output](../../assets/test_runs/01_concept_blender/01_ConceptBlender_blend_00001_.png)

---

## 🛠️ Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Blend looks foggy or washed-out at 50% | Using Linear instead of Slerp | Switch `mode` to `Slerp` |
| No visible blend — result looks like pure A or pure B | `origin_latent` is identical to `target_latent` | Ensure `origin_latent ≠ target_latent`; check both KSampler seeds are different |
| Posterized edges / black outlines in blend result | CFG too high for a blended latent | Reduce `cfg` to `4.5`, or insert a **GimbalLatentStabilizer** node with `ψ=0.88` before the final KSampler |
| Blend looks like an average, not a synthesis | `denoise` too low on the final KSampler | Raise final KSampler `denoise` to `0.90`–`1.0` |
| Artifacts at blend midpoint (noise explosions) | Concept latents are too far apart on the sphere | Try `strength=0.40` or `0.60` rather than exactly `0.50` |

---

## 🔬 Power User Notes

### `origin_latent` and the Navigation Coordinate System

`origin_latent` defines the **zero-point** of the navigation coordinate system for the Compass Pro node. Its role changes depending on your intent:

- **Pure concept blending:** Set `origin_latent = base_latent`. This causes the blend to navigate from A toward B along the geodesic arc, with no external reference frame.
- **Directional push from A toward B:** Set `origin_latent` to a **third neutral reference** — an empty latent, or a 'plain white studio' latent sampled from a neutral prompt. This reframes the navigation so that the direction vector is measured relative to a baseline, enabling more controlled steering.

### Why Slerp Outperforms Linear at Midpoint

At the 50% linear interpolation point, the L2 norm of the interpolated latent drops by approximately √2 relative to the endpoints. This pushes the midpoint *off* the Typical Set shell — into a lower-probability density region where the diffusion model hallucinates instead of synthesizing. Slerp maintains constant L2 radius throughout the arc, keeping every intermediate point on the shell.

### Extending to Animation

Set `strength` as a ComfyUI primitive connected to a float sequence node. Each queued prompt at a different strength value produces one frame of a smooth morph animation. At 25 frames (strength 0.00 → 1.00 in steps of 0.04), you get a 1-second clip at 25fps.

---

## 📁 Workflow Files

| Format | Path |
|---|---|
| ComfyUI drag-and-drop | [`workflows/ui/Gimbal_01_ConceptBlender.json`](../../workflows/ui/Gimbal_01_ConceptBlender.json) |
| API / FLUX variant | [`workflows/api_flux/API_FLUX_Gimbal_01_ConceptBlender.json`](../../workflows/api_flux/API_FLUX_Gimbal_01_ConceptBlender.json) |

---

*Gimbal Node Suite — Form & Noise Atelier*
*Navigate latent space with precision flight instruments, not lottery prompts.*
