# 🗺️ Workflow 03: Manifold Grid

> *Survey a concept's 2D latent neighborhood in a single multi-sample batch. 9 images, 1 batch, zero seed gambling.*

**Primary Flight Instrument**: `GimbalManifold_Explorer`  
**Architecture**: SDXL 4-channel (`128×128×4`) & FLUX.1 16-channel (`64×64×16`)  
**Workflow Files**: `workflows/ui/Gimbal_03_ManifoldGrid.json` | `workflows/api_flux/API_FLUX_Gimbal_03_ManifoldGrid.json`

---

## Overview

Instead of guessing random seeds to find a good composition, **Workflow 03** generates a structured 2D grid around a center concept latent $\mathbf{z}_0$ along two independent orthogonal direction vectors $\mathbf{u}$ and $\mathbf{v}$.

$$\mathbf{z}(i, j) = \text{SLERP}_\mu\left(\mathbf{z}_0, \mathbf{z}_0 + \alpha_i \mathbf{u} + \beta_j \mathbf{v}, t_{ij}\right)$$

---

## Visual Gallery: 3×3 Quadrant Slices

| Quadrant 1 (Top-Left) | Quadrant 2 (Top-Right) |
| :---: | :---: |
| ![Q1](../../assets/test_runs/03_manifold_grid/03_manifold_slice_q1.jpg) | ![Q2](../../assets/test_runs/03_manifold_grid/03_manifold_slice_q2.jpg) |
| *High +Y, -X Axis Exploration* | *High +Y, +X Axis Exploration* |

| Quadrant 3 (Bottom-Left) | Quadrant 4 (Bottom-Right) |
| :---: | :---: |
| ![Q3](../../assets/test_runs/03_manifold_grid/03_manifold_slice_q3.jpg) | ![Q4](../../assets/test_runs/03_manifold_grid/03_manifold_slice_q4.jpg) |
| *High -Y, -X Axis Exploration* | *High -Y, +X Axis Exploration* |

---

## Node Chain Wiring Architecture

```
[Checkpoint Loader: SDXL / FLUX] ──▶ MODEL, CLIP, VAE
[Prompt: "Ceramic tea bowl handcrafted"] ──▶ CONDITIONING
[EmptyLatentImage: 1024x1024] ──▶ [KSampler A: Seed 101, Denoise 1.0] ──▶ CENTER_LATENT

[🌉 Cross-Modal Bridge: "warm golden earthy crackle"] ──▶ X_VECTOR
[🌉 Cross-Modal Bridge: "cool obsidian dark glossy"] ──▶ Y_VECTOR

[🗺️ Manifold Explorer]
  center_latent = CENTER_LATENT
  x_vector = X_VECTOR
  y_vector = Y_VECTOR
  grid_size_x = 3
  grid_size_y = 3
  x_strength = 1.50
  y_strength = 1.50
  interpolation_mode = "Slerp"
  normalize_vectors = True
        │
  latent_batch (9 latents)
        │
[🛡️ Latent Stabilizer] psi = 0.88, scale_cap = 8.0
        │
[KSampler B: Refine] denoise = 0.50, CFG = 4.5, steps = 20
        │
[VAEDecode] ──▶ [🪡 Grid Stitch: columns=3] ──▶ [SaveImage]
```

---

## Detailed Parameter Matrix

| Parameter | Recommended | Range | Technical Function |
| :--- | :---: | :---: | :--- |
| `grid_size_x` / `grid_size_y` | `3` | `1 – 16` | Grid dimensions ($3 \times 3 = 9$ frames; max 256 cells). |
| `x_strength` / `y_strength` | `1.50` | `0.5 – 3.0` | Scalar span along respective axes. |
| `interpolation_mode` | `"Slerp"` | `Linear`, `Slerp`, `Slerp_Origin` | $\mu$-centered SLERP avoids high-dimensional variance collapse. |
| `normalize_vectors` | `True` | `True/False` | Unit-normalizes axes before scaling to prevent signal overdrive. |
| `denoise` (Refinement) | `0.50` | `0.45 – 0.60` | Balances topological variation against structural coherence. |

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
