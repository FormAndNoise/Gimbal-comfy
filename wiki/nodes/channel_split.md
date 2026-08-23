# 🔀 Channel Split · 🔁 Channel Merge · ⚖️ Channel Scale

> *The surgery suite. Freeze structure, mutate material. 100% silhouette lock.*

---

## Overview: The Subspace Toolkit

These three nodes work as a team. They let you **decompose a latent into frequency bands**, operate on each band independently, and **recompose** the result — all while preserving the spatial structure of the original image.

The core use case: **material transformation without silhouette change**.

| Node | Function |
|:---|:---|
| 🔀 **GimbalChannelSplit** | Splits a latent into low-channel and high-channel bands at a split index. |
| 🔁 **GimbalChannelMerge** | Recompose the two bands into a single latent. |
| ⚖️ **GimbalChannelScale** | Apply independent gain to each band (or individual channel) before merging. |

---

## The Channel Frequency Map

Different channel ranges carry different kinds of visual information:

| Architecture | Split Index | Low Channels | High Channels |
|:---|:---|:---|:---|
| SD 1.5 / SDXL (4ch) | **2** | Ch 0–1: Structure, contours, geometry | Ch 2–3: Chroma, speculars, texture |
| FLUX.1 / SD3 (16ch) | **8** | Ch 0–7: Macro spatial, composition | Ch 8–15: Micro texture, surface, color gamut |

---

## 🔀 GimbalChannelSplit

**Class**: `GimbalChannelSplit`  
**Category**: `Gimbal/Subspace`  
**Returns**: `(LATENT, LATENT)` → `low_channels`, `high_channels`

### Inputs

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `latent` | LATENT | — | Input latent to split. |
| `split_index` | INT | 2 | Channel index where the split occurs. Channels [0, split_index) go to `low_channels`; channels [split_index, C) go to `high_channels`. |

### How It Works

Simple tensor slicing:
```python
low  = samples[:, :split_index, :, :]   # [B, split_index, H, W]
high = samples[:, split_index:, :, :]   # [B, C-split_index, H, W]
```

Each output is a valid `LATENT` dict that can be processed independently by other Gimbal nodes.

---

## 🔁 GimbalChannelMerge

**Class**: `GimbalChannelMerge`  
**Category**: `Gimbal/Subspace`  
**Returns**: `(LATENT,)` → `merged_latent`

### Inputs

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `low_channels` | LATENT | — | Low-channel band (structure/geometry). |
| `high_channels` | LATENT | — | High-channel band (material/texture). |

### How It Works

Concatenates along the channel dimension:
```python
merged = torch.cat([low_samples, high_samples], dim=1)
```

The output latent has the same total channel count as the original. If you only modified `high_channels`, the `low_channels` remain exactly as they were — guaranteeing zero silhouette change.

---

## ⚖️ GimbalChannelScale

**Class**: `GimbalChannelScale`  
**Category**: `Gimbal/Subspace`  
**Returns**: `(LATENT,)` → `scaled_latent`

An optional step between Split and Merge that applies **independent gain factors** to each channel or band. Useful for finely controlling how aggressively each frequency range is modified.

### Inputs

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `latent` | LATENT | — | A latent or split band to scale. |
| `scale_ch0` through `scale_ch15` | FLOAT | 1.0 | Per-channel scale factors. 1.0 = no change; >1.0 = amplify; <1.0 = attenuate. |
| `global_scale` | FLOAT | 1.0 | Multiplied by all channels simultaneously. |

---

## The Full Subspace Material Pipeline

Here's the complete wiring for the **Subspace Material Matrix** workflow:

```
[Base Latent (architecture)] 
        │
[🔀 Channel Split]  split_index=2 (SDXL) or 8 (FLUX)
  │             │
  low_ch       high_ch
  (FREEZE)     (STEER)
               │
   [🌉 Cross-Modal Bridge] 'cold monochrome crisp sharp'
               │ direction_latent
   [🧭 Compass Pro] Orthogonal_Projection, strength=1.5
               │
   [🛡️ Latent Stabilizer] psi=0.88
               │
               high_ch_modified
               │
[🔁 Channel Merge]
  low = low_ch (FROZEN — silhouette intact)
  high = high_ch_modified (material transformed)
        │
   [KSampler] (CFG=4.5, denoise=0.65)
        │
   [VAEDecode] ─▶ Output
```

---

## Material Preset Results

| Baseline | Liquid Chrome | Oxblood Velvet |
|:---:|:---:|:---:|
| ![Ctrl](../../assets/test_runs/architectural_showcases/09_mat_ctrl_concrete.png) | ![Chrome](../../assets/test_runs/fresh_run6_exploration/09_chair_steered_liquid_chrome.png) | ![Velvet](../../assets/test_runs/fresh_run6_exploration/09_chair_steered_oxblood_velvet.png) |
| Oat linen / concrete baseline | cold monochrome crisp bright sharp | warm saturated vivid dark moody fire |

| Gold Travertine | Obsidian Glass |
|:---:|:---:|
| ![Travertine](../../assets/test_runs/architectural_showcases/09_v2_gold_travertine.png) | ![Obsidian](../../assets/test_runs/architectural_showcases/09_mat_obsidian.png) |
| warm golden punchy crisp | dark cold monochrome sharp contrast |

---

## Pro Tips

- **Always Stabilize after high-channel steering**: The high-channel modification with Cross-Modal Bridge can push values into extreme regions. `GimbalLatentStabilizer` (ψ=0.88) between the Compass output and the Channel Merge is essential.
- **Only freeze when you mean to**: If you want both structure AND material to change, don't use the split at all — just pipe the full latent through the Compass Pro directly.
- **FLUX.1 users**: Always use split_index=8 for FLUX.1 models. Using split_index=2 on a 16-channel latent will give you incorrect channel groupings.
- **Per-channel scale for fine control**: Use `GimbalChannelScale` on the high channels to attenuate overly aggressive material changes before merging. Scale 0.7 on modified high channels = a gentler transition.

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
