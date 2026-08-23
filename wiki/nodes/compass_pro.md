# 🧭 Gimbal Compass Pro

> **VibeCheck:** 🟢 Stabilized &nbsp;|&nbsp; **Category:** `Gimbal/Flight Instruments` &nbsp;|&nbsp; **Class:** `GimbalCompass_Pro`
>
> *Navigate latent space with precision flight instruments, not lottery prompts.*

---

The steering engine of Gimbal. Performs vector arithmetic between two latent points and applies the resulting direction to a base latent with strength control.

```
Returns: (LATENT, DICT)  →  latent_out, gimbal_meta
```

---

## 🗂️ When to Use It

| Use Case | Mode to Reach For |
|---|---|
| **Concept blending** — morph a forest into a mountain mid-denoise | `Slerp` at strength 0.5 |
| **Text-steered atmosphere** — push mood without changing geometry | `Orthogonal_Projection` at strength 1.0–1.5 |
| **Brand lighting transfer** — how much of a reference lighting style is already present? | `Orthogonal_Projection` at strength 1.0 |
| **Concept arithmetic** — `king - man + woman = queen` in latent space | `Standard` or `Normalized` |
| **Negative steering** — push *away* from a concept (e.g. remove warmth) | Any mode with `strength < 0` |

---

## ⚡ Quick Wiring

```
[KSampler A (concept A)]──► base_latent ──┐
[KSampler B (concept B)]──► target_latent ─┤
[Empty Latent]────────────► origin_latent ─┴──► 🧭 Compass Pro ──► latent_out ──► [KSampler] ──► [VAEDecode] ──► Output
                                                                  └──► gimbal_meta (telemetry)
```

> **Tip:** For basic concept blending, set `origin_latent` to an **Empty Latent** (all zeros). The delta becomes `target - 0 = target`, and `strength` becomes a pure blend weight.

---

## 📋 Inputs

### Required

| Parameter | Type | Range / Options | Default | Description |
|---|---|---|---|---|
| `base_latent` | `LATENT` | — | *required* | The starting point. Your base image latent or empty noise. This is the canvas being steered. |
| `target_latent` | `LATENT` | — | *required* | The destination concept. The node computes a direction *toward* this. |
| `origin_latent` | `LATENT` | — | *required* | The reference anchor. Delta = `target − origin` is computed here, then applied to `base`. Use an Empty Latent to treat `target` as an absolute direction. |
| `strength` | `FLOAT` slider | −10.0 → 10.0 | `1.0` | How hard to push. `0` = no change. `1` = full delta applied. Negative = push *away* from target. Values > 1 extrapolate beyond the destination. |
| `mode` | enum | See **Modes** | `Standard` | The mathematical model for how the delta is applied. |
| `clamp_output` | `BOOL` | — | `False` | If enabled, clamps the result tensor to [`clamp_min`, `clamp_max`] after steering. Useful to prevent extreme values from breaking the VAE. |
| `clamp_min` | `FLOAT` | −100.0 → 0.0 | `−10.0` | Lower bound of the output clamp range. Only active when `clamp_output = True`. |
| `clamp_max` | `FLOAT` | 0.0 → 100.0 | `10.0` | Upper bound of the output clamp range. Only active when `clamp_output = True`. |
| `allow_batch_expand` | `BOOL` | — | `False` | Allow broadcasting when batch sizes mismatch. When enabled, single-sample tensors expand to match via `.expand()`. Raises an error when disabled. |
| `ortho_per_channel` | `BOOL` | — | `False` | Only used by `Orthogonal_Projection` mode. When `True`, projects each channel independently over its spatial dimensions instead of the full C×H×W vector. |
| `clamp_mask_input` | `BOOL` | — | `False` | Clamps the incoming mask values to [0, 1] before applying. Recommended if your mask source may produce values outside that range. |
| `enable_perf_logging` | `BOOL` | — | `False` | Logs per-run timing in milliseconds to the ComfyUI console. Useful for benchmarking complex chains. |

### Optional

| Parameter | Type | Description |
|---|---|---|
| `mask` | `MASK` | Spatially restricts navigation to a region. The mask multiplies the delta before it is applied — white (1.0) = full steering, black (0.0) = no change. Bilinear-resized to match base resolution automatically. |
| `seed` | `INT` | Deterministic seed used **only** by `Stochastic_Sample` mode. Same seed = same pixel selection pattern. |
| `mu_centroid` | `LATENT` | Override the population centroid μ for μ-centered `Slerp` mode. When absent, `origin_latent` is used as the anchor. Provide a large batch of representative latents for best results. |

---

## 🎛️ Modes In Depth

### `Standard`

```
result = base + (target − origin) × strength
```

The simplest mode. Raw delta added at full scale. Good when your delta is already well-scaled (e.g., Cross-Modal Bridge outputs). No normalization — large deltas will cause proportionally large shifts.

---

### `Normalized`

```python
delta_hat = delta / ‖delta‖₂
result = base + delta_hat × (strength × √D)
```

Unit-normalizes the delta direction to the latent's L2 norm, then scales by `strength × √D` (where D = total elements). This **prevents over-driving** when different inputs produce wildly different delta magnitudes. Recommended when chaining multiple Compass nodes.

---

### `Orthogonal_Projection`

```
delta_hat = delta / ‖delta‖₂
result = base + (base · delta_hat) × delta_hat × strength
```

Projects the `base` tensor onto the delta direction vector. This answers: *"how much of this concept/style is already present in base?"* — then amplifies or attenuates that component.

Best for **brand lighting transfer** and **text-steered atmosphere**: it modifies only the component of `base` that is *already aligned* with the target direction, leaving perpendicular (geometric/structural) content untouched.

Enable `ortho_per_channel = True` to project each channel independently over its spatial dimensions (C×[H×W] instead of [C×H×W]).

---

### `Slerp` *(μ-centered, recommended)*

Interpolates along the **geodesic arc** on the Typical Set shell, anchored at the empirical centroid μ. Instead of cutting through the hollow center of the latent hypersphere (which causes variance collapse), it travels along the high-probability shell where real image latents live.

- Best quality for concept blending
- `strength ∈ [0, 1]` for smooth interpolation (values outside this range extrapolate)
- Provide `mu_centroid` from a large dataset for maximum accuracy; defaults to `origin_latent`

---

### `Slerp_Origin` *(legacy, zero-centered)*

Origin-centered SLERP. Traverses the great-circle arc centered at z=0. Kept for backward compatibility with older workflows.

> ⚠️ **Warning:** May cause **variance collapse** at midpoints in high dimensions (SDXL, FLUX). Prefer `Slerp` for new work.

---

### `Blend_Overlay`

Photoshop-style overlay blend formula:
- Where `base < 0.5`: `2 × base × target`
- Where `base ≥ 0.5`: `1 − 2 × (1−base) × (1−target)`

Then linearly interpolated with `strength`. Good for **texture overlay effects** where the darker/lighter areas respond differently to the blend.

---

### `Blend_Multiply`

```
blend = base × target
result = lerp(base, blend, strength)
```

Multiplicative blend. Darkens and suppresses where `target` has low values, preserves where target is bright. Useful for applying shadow/occlusion masks from one concept onto another.

---

### `Stochastic_Sample`

Per-pixel random selection between `base` and `target` at ratio = `strength` (clamped to [0, 1]). Uses a seeded `torch.Generator` — **fully deterministic** given the same `seed`. Useful for noise injection experiments and controlled randomization.

---

## 📡 `gimbal_meta` Output

The second output is a telemetry dictionary providing full transparency into what the node did:

| Key | Type | Description |
|---|---|---|
| `mode` | `str` | The mode that was executed. |
| `strength` | `float` | The strength value used. |
| `clamp_output` | `bool` | Whether output clamping was active. |
| `clamp_range` | `[float, float]` or `null` | The [min, max] clamp range, or `null` if clamping was off. |
| `mask_applied` | `bool` | Whether a mask was connected and used. |
| `ortho_per_channel` | `bool` or `null` | Per-channel mode state (Orthogonal_Projection only). |
| `slerp_anchor` | `str` or `null` | Which tensor was used as μ anchor (`"mu_centroid"` or `"origin_latent"`). |
| `base_shape` | `[B, C, H, W]` | Shape of the base latent (after batch expansion). |
| `result_shape` | `[B, C, H, W]` | Shape of the output latent. |
| `device` | `str` | PyTorch device string (e.g., `"cuda:0"`). |
| `dtype` | `str` | Tensor dtype (e.g., `"torch.float16"`). |
| `elapsed_ms` | `float` or `null` | Execution time in ms. Only populated when `enable_perf_logging = True`. |

> Connect `gimbal_meta` to a **Gimbal Latent Telemetry** node to display this as formatted text in the UI.

---

## 🖼️ Example Outputs

![Orthogonal Projection mode: Cross-Modal Bridge steering with 100% foreground geometry lock](../../assets/test_runs/02_text_steered/02_text_steered_portrait.jpg)

*Orthogonal Projection mode: Cross-Modal Bridge steering with 100% foreground geometry lock*

---

![Slerp mode at Step 0: geodesic midpoint between two concept prompts](../../assets/test_runs/fresh_run6_exploration/01_slerp_noise_50pct.png)

*Slerp mode at Step 0: geodesic midpoint between two concept prompts*

---

## 💡 Pro Tips

- **SLERP blending at Step 0:** Set `origin_latent` to an Empty Latent and `target_latent` to your second concept's KSampler output. `strength = 0.5` places you exactly at the geodesic midpoint before any denoising begins.

- **Text-steered atmosphere:** Use `Orthogonal_Projection` mode with `strength 1.0–1.5`, and connect a **Cross-Modal Bridge** output as `target_latent`. The Cross-Modal Bridge converts text keywords into latent directions; Orthogonal Projection then applies *only the component of that direction already present in your base* — geometry lock is free.

- **Multi-step navigation:** Chain Compass outputs. Send `latent_out` from one Compass into `base_latent` of the next. Each hop steers in a new direction — useful for compound concept arithmetic (warm + cinematic + dark = moody noir).

- **Foreground-preserving style transfer:** Paint a mask that covers only the background. Connect it to the `mask` input. The steering delta is zeroed in masked-out (black) areas, leaving your foreground untouched.

- **Strength calibration:** Start at `strength = 1.0` (Standard) or `strength = 0.5` (Slerp), then nudge by 0.1 steps. Values above 2.0 in Standard mode often produce abstract/broken results unless your delta is very small.

---

## 🔬 Under the Hood *(Power User)*

### Delta Computation

```python
delta = target - origin          # Raw direction vector
# Then mode-specific transform applied to (base, delta, strength)
```

The `target_latent` and `origin_latent` are automatically resized (bilinear interpolation in float32, then cast back to original dtype) and device/dtype-aligned to match `base_latent` before the subtraction.

### Mask Application

```python
delta = delta * mask_tensor      # [B, 1, H, W] — spatially gates the direction
```

The mask is bilinear-upsampled to match `base` spatial dimensions and broadcast along the batch and channel dimensions. With `clamp_mask_input = True`, mask values are clamped to [0, 1] before multiplication — prevents negative mask values from reversing the steering direction.

### Batch Broadcasting

When `allow_batch_expand = True`, single-sample tensors (B=1) are expanded to the maximum batch size via `.expand()` (zero-copy view). If `B_target % B_base ≠ 0`, an error is raised — partial repeats are not allowed.

### Device / Dtype Safety

All tensors are coerced to `base_latent`'s device and dtype before any arithmetic:

```python
target = target.to(device=device, dtype=dtype)
origin = origin.to(device=device, dtype=dtype)
```

The final result is explicitly cast back to `dtype` after all operations to prevent silent fp32 upcasting from leaking through.

### VRAM Safety

`torch.no_grad()` wraps **all** tensor operations. No autograd graph is built. For SDXL latents (B=1, C=4, H=128, W=128), a single Compass forward pass uses approximately **0.5 MB** of VRAM overhead beyond the input tensors.

---

## ⚙️ Technical Reference

| Property | Value |
|---|---|
| ComfyUI class name | `GimbalCompass_Pro` |
| Legacy alias | `WayfinderCompass_Pro` |
| Function | `navigate()` |
| Return types | `("LATENT", "DICT")` |
| Return names | `("latent_out", "gimbal_meta")` |
| Category | `Gimbal/Flight Instruments` |
| Default clamp range | [−10.0, +10.0] |
| VRAM mode | `torch.no_grad()` |

---

*Form & Noise Atelier — Gimbal Node Suite*
