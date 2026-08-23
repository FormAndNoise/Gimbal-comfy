# 🔁 Gimbal Channel Merge

> *Reconstruct multi-channel latent tensors by recombining split structural and textural frequency bands.*

**Class**: `GimbalChannelMerge`  
**Category**: `Gimbal/Subspace`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT,)` → `merged_latent`

---

## What It Does

`GimbalChannelMerge` takes two split latent sub-bands (`low_channels` and `high_channels`) and concatenates them along the channel dimension (dimension 1), restoring the complete 4-channel (SDXL) or 16-channel (FLUX.1) tensor for decoding or downstream sampling.

---

## Inputs

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `low_channels` | LATENT | Structural / contour frequency band tensor $[B, C_{\text{low}}, H, W]$. |
| `high_channels` | LATENT | Texture / chroma / lighting band tensor $[B, C_{\text{high}}, H, W]$. |

---

## Wiring Architecture

```
[🔀 Channel Split] ──┬──▶ low_channels (Ch 0..1) ──[FROZEN SILHOUETTE]─────────┐
                     └──▶ high_channels (Ch 2..3) ──▶ [🧭 Compass] ──▶ [🛡️] ──┼──▶ [🔁 Channel Merge] ──▶ [KSampler]
```

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
