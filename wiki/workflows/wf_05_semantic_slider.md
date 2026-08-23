# 🎚️ Workflow 05: Semantic Slider

> *Unsupervised attribute isolation via real-time SVD/PCA batch covariance decomposition.*

**Primary Flight Instrument**: `GimbalSemanticSlider`  
**Architecture**: SDXL 4-channel & FLUX.1 16-channel  
**Workflow Files**: `workflows/ui/Gimbal_05_SemanticSlider.json` | `workflows/api_flux/API_FLUX_Gimbal_05_SemanticSlider.json`

---

## Overview

**Workflow 05** extracts the principal directions of variance across a batch of sample latents $\mathbf{X} \in \mathbb{R}^{B \times D}$ using singular value decomposition:
$$\mathbf{X}_c = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
The right singular vectors $\mathbf{v}_k$ define orthogonal semantic axes (e.g. key lighting, facial expression, material roughness). The user can then slide along $\mathbf{v}_k$ on an individual base latent without altering other attributes.

---

## Visual Showcase

![Semantic Slider Portrait](../../assets/test_runs/05_semantic_slider/05_semantic_slider_portrait_fullres.png)  
*Portrait illumination and age modulation along principal component PC0 with orthogonal geometry locking.*

---

## Node Chain Wiring Architecture

```
[Varied Prompt Batch (N >= 4)] ──▶ [KSampler A: Batch 8] ──▶ LATENT_BATCH (8 samples)
[Target Image to Modulate] ──▶ [VAE Encode / KSampler] ──▶ BASE_LATENT

[🎚️ Semantic Slider]
  latent_batch = LATENT_BATCH
  base_latent = BASE_LATENT
  n_components = 8
  pc_index = 0             (0 = Primary variance axis)
  slider_value = 1.50       (Adjust ±1.5 standard deviations)
  orthogonalize = True      (Subtracts parallel components from base)
  normalize_direction = True
        │
  modulated_latent
        │
[KSampler B: Refine] denoise = 0.45, CFG = 4.5, steps = 20
        │
[VAEDecode] ──▶ [SaveImage]
```

---

## Principal Component Interpretation Guide

| Component Index | Typical Semantic Axis | Variance Explained |
| :---: | :--- | :---: |
| **PC0** | Key illumination / Global luminance energy | `35% – 50%` |
| **PC1** | Color temperature (Warm Amber vs Cold Cyan) | `15% – 25%` |
| **PC2** | Macro structural volume / Object massing | `8% – 15%` |
| **PC3+** | Fine surface texture, specular micro-highlights | `< 8%` |

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
