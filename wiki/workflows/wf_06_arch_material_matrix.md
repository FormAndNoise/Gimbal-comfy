# 🏛️ Workflow 06: Architecture Material Matrix

> *Sweep a 2D parameter manifold across architectural layouts, generating multi-material elevations and structural finishes in a single pass.*

**Primary Flight Instruments**: `GimbalCrossModalBridge` + `GimbalManifold_Explorer` + `GimbalLatentStabilizer`  
**Architecture**: SDXL 4-channel (`128×128×4`) & FLUX.1 16-channel (`64×64×16`)  
**Workflow Files**: `workflows/ui/Gimbal_06_ArchitectureMaterialMatrix.json` | `workflows/api_flux/API_FLUX_Gimbal_06_ArchitectureMaterialMatrix.json`

---

## Overview

Architectural visualization requires rapid iteration over surface materials (e.g. brutalist concrete vs. golden travertine vs. obsidian glazing) across consistent building elevations. Rather than prompting separate images with differing random geometries, **Workflow 06** maps two material vectors onto the X and Y axes of a 2D latent topography.

---

## Visual Showcase

| Hero Baseline | Obsidian Facade | Gold Travertine |
| :---: | :---: | :---: |
| ![Hero Base](../../assets/test_runs/architectural_showcases/08_v2_hero_base.png) | ![Obsidian](../../assets/test_runs/architectural_showcases/09_v2_obsidian.png) | ![Travertine](../../assets/test_runs/architectural_showcases/09_v2_gold_travertine.png) |
| *Original structural elevation* | *Monochrome reflective glazing* | *Warm textured mineral stone* |

---

## Node Chain Wiring Architecture

```
[Checkpoint Loader: SDXL / FLUX] ──▶ MODEL, CLIP, VAE
[Prompt: "Modern brutalist pavilion"] ──▶ CONDITIONING
[Empty Latent / Initial Noise] ──▶ [KSampler: Seed 42, Denoise 1.0] ──▶ BASE_ARCH_LATENT

[🌉 Cross-Modal Bridge A] "brutalist concrete monochrome crisp" ──▶ X_MATERIAL_VECTOR
[🌉 Cross-Modal Bridge B] "warm travertine gold limestone textured" ──▶ Y_MATERIAL_VECTOR

[🗺️ Manifold Explorer]
  center_latent = BASE_ARCH_LATENT
  x_vector = X_MATERIAL_VECTOR
  y_vector = Y_MATERIAL_VECTOR
  grid_size_x = 3, grid_size_y = 3 (9 elevation variations)
  x_strength = 1.4, y_strength = 1.4
  interpolation_mode = "Slerp"
  normalize_vectors = True
        │
  latent_batch (9 latents)
        │
[🛡️ Latent Stabilizer] psi = 0.88, scale_cap = 8.0
        │
[KSampler: Refine] denoise = 0.48, CFG = 4.0, steps = 22
        │
[VAEDecode] ──▶ [🪡 Grid Stitch: 3 cols] ──▶ [SaveImage]
```

---

## Optimal Configuration Specifications

| Parameter | Optimal Value | Permissible Range | Operational Role |
| :--- | :---: | :---: | :--- |
| **Grid Size** | `3 × 3` | `2 × 2` – `4 × 4` | Synthesizes 9 material permutation quadrants. |
| **X / Y Strength** | `1.40` | `1.0 – 2.0` | Controls the material saturation depth. |
| **Interpolation Mode** | `Slerp` | `Slerp` | Maintains constant L2 radius on Gaussian Typical Set. |
| **Stabilizer $\psi$** | `0.88` | `0.85 – 0.92` | Reins in high-frequency texture noise spikes. |
| **Refinement Denoise** | `0.48` | `0.42 – 0.55` | Preserves architectural elevation geometry. |
| **Refinement CFG** | `4.0` | `3.5 – 5.0` | Prevents over-guidance edge burning. |

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
