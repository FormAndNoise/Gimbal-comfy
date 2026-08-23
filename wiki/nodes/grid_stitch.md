# 🪡 Gimbal Grid Stitch

> *Stitch batches of latent variations or manifold grids into high-resolution composite image contact sheets.*

**Class**: `GimbalGridStitch`  
**Category**: `Gimbal/Utility`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(IMAGE,)` → `stitched_image`

---

## What It Does

`GimbalGridStitch` takes a batch of decoded images (such as the 9 output images from `GimbalManifold_Explorer` or a 36-frame sequence from `GimbalCircularOrbit`) and arranges them into a clean, border-aligned composite contact sheet image.

---

## Inputs

| Parameter | Type | Default | Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `images` | IMAGE | — | — | Decoded image batch $[B, H, W, C]$. |
| `columns` | INT | 3 | 1 – 64 | Number of columns in the output composite grid. |
| `pad_value` | FLOAT | 0.0 | 0.0 – 1.0 | Background fill color for empty cells (0.0 = black, 1.0 = white). |

---

## Example Contact Sheet Layout

For a batch of 9 images with `columns = 3`:
```
┌─────────┬─────────┬─────────┐
│ Cell 0  │ Cell 1  │ Cell 2  │  (Top row: y = +strength)
├─────────┼─────────┼─────────┤
│ Cell 3  │ Cell 4  │ Cell 5  │  (Middle row: y = 0, center)
├─────────┼─────────┼─────────┤
│ Cell 6  │ Cell 7  │ Cell 8  │  (Bottom row: y = -strength)
└─────────┴─────────┴─────────┘
```

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
