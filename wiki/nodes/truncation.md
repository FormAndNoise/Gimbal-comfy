# 📉 Gimbal Truncation

> *Surgical variance shrinkage towards the empirical population centroid, reining in high-CFG outlier spikes without full pipeline overhead.*

**Class**: `GimbalTruncation`  
**Category**: `Gimbal/Quality`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT,)` → `truncated_latent`

---

## What It Does

`GimbalTruncation` implements **Equation E3** from the LAMNr theoretical framework:
$$\mathbf{z}' = \mu + \psi(\mathbf{z} - \mu)$$

Where:
- $\mu$ is the empirical population mean (calculated across channels or supplied via reference).
- $\psi$ is the truncation shrinkage coefficient.

In high-dimensional diffusion spaces, aggressive text guidance (CFG > 6.0) forces latent activations into the extreme Gaussian tails ($> 4\sigma$). In these low-probability outlier regions, the VAE decoder produces high-contrast black boundary clipping, plastic textures, and harsh edge ringing. `GimbalTruncation` cleanly contracts the latent envelope back to the optimal generative sweet spot.

---

## Inputs & Parameters

| Parameter | Type | Default | Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | — | Latent tensor $[B, C, H, W]$. |
| `truncation_psi` | FLOAT | 0.90 | 0.0 – 3.0 | Shrinkage factor $\psi$. ($1.0 = \text{identity}$; $<1.0 = \text{contract}$; $>1.0 = \text{amplify}$). |
| `channel_adaptive` | BOOLEAN | True | True/False | Computes per-channel mean $\mu_c$ independently rather than a single global scalar mean. |
| `mu_override` | LATENT (opt) | — | — | External centroid tensor anchor. If omitted, computes batch mean. |

---

## Truncation Factor ($\psi$) Calibration Guide

| $\psi$ Value | Visual Impact | Ideal Use Case |
| :---: | :--- | :--- |
| **0.00** | Complete collapse to Fréchet mean template ($\mathbf{z}' = \mu$). | Baseline diagnostic template extraction. |
| **0.80 – 0.85** | Strong variance contraction. Smooths skin, removes all high-contrast artifacts. | High CFG ($7.0 - 8.5$), extreme prompt pushes. |
| **0.88 – 0.92** | **Recommended Standard.** Retains fine textures while eliminating boundary ringing. | General production workflows, Cross-Modal steering. |
| **1.00** | Exact mathematical identity ($\mathbf{z}' = \mathbf{z}$). | Passthrough / baseline comparisons. |
| **1.10 – 1.30** | Variance expansion. Boosts contrast, exaggerates textural micro-geometry. | Stylized illustrative or gritty photographic styles. |

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
