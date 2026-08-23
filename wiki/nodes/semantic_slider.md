# 🎚️ Gimbal Semantic Slider

> *Dial one attribute without touching the rest of the image. Real PCA-based feature isolation.*

**Class**: `GimbalSemanticSlider`  
**Category**: `Gimbal/Decomposition`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT, DICT, STRING)` → `modulated_latent`, `pca_meta`, `pca_report`

---

## What It Does

The Semantic Slider performs **Principal Component Analysis (PCA)** on a batch of latents to discover the *directions of maximum variance* — the fundamental axes along which your images differ most.

Once those axes are found, you can slide a specific one up or down. Want more of the dominant lighting difference between your batch samples? Slide PC0. Want the secondary variation (maybe geometry vs. texture)? Slide PC1.

The result: **controlled, targeted attribute modification** without the guesswork of text prompting.

---

## The Key Insight (Beginner-Friendly)

Imagine you generated 8 images of the same room with slightly different prompts. Some are brighter, some have warmer colors, some have different furniture arrangements. PCA finds the "directions" that account for most of those differences:

- **PC0** might be "overall brightness"
- **PC1** might be "warm vs. cool color temperature"  
- **PC2** might be "amount of furniture detail"

The Semantic Slider lets you grab any one of those directions and push your image along it — independently of all the others.

---

## Quick Wiring

```
[KSampler × 8 images] ─▶ latent_batch ─▶ [🎚️ Semantic Slider]
[Single Reference] ────▶ base_latent ──▶      │
                          pc_index = 0         │
                          slider_value = 1.5   │
                          orthogonalize = True  │
                                               ▼
                                       modulated_latent ─▶ [KSampler (denoise 0.45)] ─▶ [VAEDecode]
```

---

## Inputs

| Parameter | Type | Default | Range | Description |
|:---|:---|:---|:---|:---|
| `latent_batch` | LATENT | — | — | A batch of latents to perform PCA on. Minimum 4. |
| `base_latent` | LATENT | — | — | The single latent to apply the PC direction to. |
| `n_components` | INT | 10 | 1–10 | How many principal components to compute. |
| `pc_index` | INT | 0 | 0–9 | Which component to slide. PC0 = largest variance. |
| `slider_value` | FLOAT | 0.0 | -5.0 – 5.0 | How far to push along the PC direction. ±1.0 = one standard deviation. |
| `orthogonalize` | BOOL | True | — | Orthogonalize PC direction against base before applying. Recommended. |
| `normalize_direction` | BOOL | True | — | Normalize PC to unit length before scaling. |
| `enable_perf_logging` | BOOL | False | — | Console timing output. |

---

## How to Read the `pca_report` Output

The text report tells you what each component captured:

```
PCA Report — GimbalSemanticSlider
Batch size: 8 | Components computed: 8
PC0 variance explained: 42.3%
PC1 variance explained: 18.7%
PC2 variance explained: 11.2%
...
Applied: PC0 | slider_value: 1.50 | base_shape: [1, 4, 128, 128]
```

A component explaining >30% of variance is usually a dominant, interpretable attribute (lighting, color temperature). Components <5% are usually noise or idiosyncratic detail.

---

## Optimal Settings

| Scenario | batch_size | n_components | pc_index | slider_value |
|:---|:---|:---|:---|:---|
| Find the dominant variation | ≥4 | 5 | 0 | 1.0 → explore |
| Fine-grained attribute isolation | ≥8 | 10 | 0,1,2 | ±1.5 |
| Extreme push | ≥8 | 10 | 0 | ±3.0 |
| Conservative refinement | ≥4 | 5 | 0 | ±0.5 |

---

## Pro Tips

- **Batch size = quality**: More samples → better-estimated principal components. Use at least 4; 8+ gives much cleaner PC directions.
- **PC0 first**: Always start by exploring PC0. It captures the most variance and is most likely to be a human-interpretable attribute like brightness or color temperature.
- **`orthogonalize=True`**: This removes any component of the PC direction that's already present in your base latent. Without it, the slider can push in directions that reinforce existing structure rather than revealing new variation.
- **Feed with varied prompts**: The PCA is only as good as the variance in your batch. Use prompts that genuinely differ (different lighting, different moods, different materials) to get meaningful components.
- **Don't confuse batch size with pc_index**: If `pc_index > n_effective_components` (where `n_effective = min(n_components, batch_size)`), you'll get an error. Always use a batch at least as large as the highest `pc_index` you want to explore.

---

## Under the Hood (Researchers)

**PCA implementation:**
1. Flatten each latent from `[B, C, H, W]` to `[B, D]` where `D = C × H × W`.
2. Compute batch centroid μ.
3. Center: `X_c = X - μ`.
4. Compute economy SVD: `U, S, Vh = svd(X_c, full_matrices=False)`.
5. Principal components = rows of `Vh` (right singular vectors). These are the directions of maximum variance in R^D.
6. Eigenvalues = `S² / B` (variance explained per component).

**Direction application:**
- `pc_dir = Vh[pc_index]` — the PC direction vector in R^D.
- If `orthogonalize`: project out any component of `pc_dir` parallel to `base_flat`, ensuring the slider moves perpendicular to the base's current position.
- `result = base_flat + slider_value × pc_dir × scale`.
- Reshape back to `[1, C, H, W]`.

**Known limitation (UI):** ComfyUI doesn't support dynamic slider maximums, so `pc_index` displays a static max of `n_components` (default 10) even if your batch only supports fewer components. The validation inside `apply_slider` catches this with an informative error message.

---

![Semantic Slider Portrait](../../assets/test_runs/05_semantic_slider/05_semantic_slider_portrait.jpg)  
*PCA-isolated attribute control on a portrait. PC0 controlled overall luminance direction.*

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
