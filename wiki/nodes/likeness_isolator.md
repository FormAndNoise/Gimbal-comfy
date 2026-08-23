# 🎭 Gimbal Likeness Isolator

> *Dynamic LoRA parameter probe decoupling facial/subject identity tokens from background lighting and stylistic biases.*

**Class**: `GimbalLikenessIsolator`  
**Category**: `Gimbal/Conditioning`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(MODEL, CLIP)` → `patched_model`, `patched_clip`

---

## What It Does

Standard ComfyUI LoRA loaders treat weights as an indivisible black box: changing strength alters the character's facial structure, clothing, background lighting, and render style simultaneously.

`GimbalLikenessIsolator` intercepts the LoRA weight patching process and exposes **three independent steering axes**:

1. **`strength`**: Overall LoRA scaling multiplier.
2. **`alpha`**: Dynamic network override ratio (can be driven by external flight instruments like `GimbalManifold_Explorer` or `GimbalSemanticSlider`).
3. **`likeness_mask`**: Text-encoder token isolation ratio, separating identity-specific tokens from style/atmosphere tokens.

---

## Inputs & Parameters

| Parameter | Type | Default | Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `model` | MODEL | — | — | Base diffusion model from checkpoint loader. |
| `clip` | CLIP | — | — | Base CLIP text encoder. |
| `lora_name` | STRING | `character.safetensors` | — | LoRA weights file in the ComfyUI models directory. |
| `strength` | FLOAT | 1.0 | 0.0 – 2.0 | Overall magnitude applied to UNet weight deltas. |
| `alpha` | FLOAT | 1.0 | 0.0 – 2.0 | Dynamic scaling factor (supports modulation). |
| `likeness_mask` | FLOAT | 0.80 | 0.0 – 1.0 | Ratio of identity tokens preserved in the text encoder. |

---

## Likeness Mask Calibration

```
likeness_mask = 1.0  ──▶ Full Identity + Training Style (Character + original background/lighting)
likeness_mask = 0.8  ──▶ Pure Character Identity (Transfers cleanly to new environments)
likeness_mask = 0.3  ──▶ Soft Likeness Resemblance (Subtle facial similarity on different ethnicity/age)
likeness_mask = 0.0  ──▶ Style-Only Transfer (Lighting/brushwork only, zero character resemblance)
```

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
