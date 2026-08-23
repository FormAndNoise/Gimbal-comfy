# ⚖️ Gimbal Channel Scale

> *Independent per-channel frequency gain and amplitude control across 4-channel and 16-channel diffusion models.*

**Class**: `GimbalChannelScale`  
**Category**: `Gimbal/Subspace`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT,)` → `scaled_latent`

---

## What It Does

`GimbalChannelScale` allows fine-grained amplification or attenuation of individual latent channels. By scaling specific channels, users can increase high-frequency surface detail, damp color saturation, or boost directional lighting contrast before sampling.

---

## Inputs

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | Input latent tensor $[B, C, H, W]$. |
| `global_scale` | FLOAT | 1.0 | Global multiplier applied across all channels simultaneously. |
| `scale_ch0` .. `scale_ch15` | FLOAT | 1.0 | Independent per-channel multipliers (channels $> C$ are safely ignored). |

---

## SDXL Channel Frequency Mapping

- `scale_ch0` (Luminance Macro): Controls global key illumination.
- `scale_ch1` (Color Balance): Controls warm/cool temperature separation.
- `scale_ch2` (Chroma Saturation): Controls vividness vs monochrome attenuation.
- `scale_ch3` (High-Frequency Texture): Controls fine surface detail sharpness.

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
