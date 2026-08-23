# 🗺️ Gimbal Manifold Explorer

> **VibeCheck:** 🟢 Stabilized &nbsp;|&nbsp; **Category:** `Gimbal/Flight Instruments` &nbsp;|&nbsp; **Class:** `GimbalManifold_Explorer`
>
> *Navigate latent space with precision flight instruments, not lottery prompts.*

---

Generates an N×M batch of latents by interpolating a center point along two independent directional vectors — a topological map of the local neighborhood.

```
Returns: (LATENT, DICT, STRING)  →  latent_batch, grid_meta, grid_report
```

Think of it as a **2D concept map**: your center point is home base, the X-axis is one dimension of variation (e.g. warm → cool), and the Y-axis is another (e.g. sharp → soft). The node generates every intersection in one shot as a batch.

---

## 🗂️ When to Use It

| Use Case | Setup |
|---|---|
| **Rapid concept exploration** — see 9 or 16 variations of a scene at once | 3×3 or 4×4 grid, low denoise (0.45–0.55) |
| **Material × mood matrix** — oak/steel/concrete × warm/cool/neutral | X = material vector, Y = temperature vector |
| **Architectural layout variations** — two orthogonal spatial concepts as axes | X = open/enclosed, Y = modern/organic |
| **Understanding concept interaction** — does warm+dark = noir? | Inspect corner cells of the matrix |
| **KSampler seed sweeps** — fix vectors, vary center seed for texture variety | Same grid, multiple center latents batched |

---

## ⚡ Quick Wiring

```
[Center Latent] ──────────► center_latent ─┐
[Cross-Modal Bridge A] ───► x_vector ──────┤
[Cross-Modal Bridge B] ───► y_vector ──────┴──► 🗺️ Manifold Explorer
                                                │
                                latent_batch ──► [KSampler (denoise 0.45–0.60)]
                                grid_meta ────► [Gimbal Telemetry / Show Text]
                                grid_report ──► [Show Text]
                                                │
                                           [VAEDecode] ──► [Image Grid / Save Images]
```

> ⚠️ **Critical:** The `latent_batch` output is pure latent math — it has not been denoised. Always pass it through a KSampler at **denoise 0.45–0.60** before decoding. Skipping this step gives you blurry averaged noise, not coherent images.

---

## 📋 Inputs

### Required

| Parameter | Type | Range / Options | Default | Description |
|---|---|---|---|---|
| `center_latent` | `LATENT` | — | *required* | The center of the exploration grid. All cells are computed as displacements from this point. Use a KSampler output or a meaningful reference latent. |
| `x_vector` | `LATENT` | — | *required* | Directional vector defining the **horizontal axis**. The node computes `x_vector − center` as the raw direction, then optionally normalizes it. |
| `y_vector` | `LATENT` | — | *required* | Directional vector defining the **vertical axis**. Same treatment as `x_vector`. X and Y are independent — they do not need to be orthogonal, though that produces the cleanest grids. |
| `grid_size_x` | `INT` | 1–16 | `3` | Number of columns. 3×3 = 9 images. 4×4 = 16. Maximum total cells is **256** (e.g. 16×16). |
| `grid_size_y` | `INT` | 1–16 | `3` | Number of rows. See `grid_size_x`. |
| `x_strength` | `FLOAT` slider | −10.0 → 10.0 | `1.0` | Scales how far each step moves along the X axis. `1.0` = one unit vector step per grid cell offset. Increase for more dramatic X variation. |
| `y_strength` | `FLOAT` slider | −10.0 → 10.0 | `1.0` | Same as `x_strength`, for the Y axis. |
| `interpolation_mode` | enum | `Linear`, `Slerp`, `Slerp_Origin` | `Linear` | How each cell is computed from the center. See **Interpolation Modes** below. |
| `normalize_vectors` | `BOOL` | — | `True` | Normalize X and Y to unit length (scaled by √D) before applying strength. **Strongly recommended** — prevents one axis from dominating due to magnitude differences. |
| `clamp_output` | `BOOL` | — | `False` | Clamp each cell's tensor to [`clamp_min`, `clamp_max`] after interpolation. |
| `clamp_min` | `FLOAT` | −100.0 → 0.0 | `−10.0` | Lower clamp bound (active only when `clamp_output = True`). |
| `clamp_max` | `FLOAT` | 0.0 → 100.0 | `10.0` | Upper clamp bound (active only when `clamp_output = True`). |
| `enable_perf_logging` | `BOOL` | — | `False` | Logs grid generation and total time to the ComfyUI console. |

### Optional

| Parameter | Type | Description |
|---|---|---|
| `mu_override` | `LATENT` | Population centroid override for μ-centered `Slerp`. When absent, `center_latent` is used as the geodesic anchor. Provide a large batch of representative latents for maximum accuracy. |

---

## 🎛️ Interpolation Modes

### `Linear`

```
cell[i,j] = center + (x_vec × x_strength × ox) + (y_vec × y_strength × oy)
```

Where `ox` and `oy` are the signed grid offsets (e.g. for a 3×3 grid: −1, 0, +1).

Fast and deterministic. Works well close to center. At large offsets, can drift off the Typical Set (the high-probability shell where real image latents live), producing blurry or artifact-prone images. Good starting point for exploration.

---

### `Slerp` *(μ-centered, recommended)*

Applies geodesic arc interpolation anchored at the population centroid μ for **each axis independently**, then adds the displacements:

```
dx = Slerp_μ(center, center + x_vec, t_x) − center
dy = Slerp_μ(center, center + y_vec, t_y) − center
cell = center + dx + dy
```

Stays on the Typical Set shell throughout the grid — corner cells are as well-formed as center cells. Best visual quality, especially for large grids or extreme strengths.

---

### `Slerp_Origin` *(legacy)*

Zero-centered SLERP. Kept for backward compatibility.

> ⚠️ **Warning:** May cause variance collapse at cells far from center in high-dimensional spaces (SDXL 4ch, FLUX 16ch). Prefer `Slerp` for new work.

---

## 📡 `grid_meta` Output

The `grid_meta` dictionary provides full grid telemetry:

| Key | Description |
|---|---|
| `grid_size_x` / `grid_size_y` | Grid dimensions as specified. |
| `total_cells` | Total images in the batch (X × Y). |
| `center_batch_size` | Batch size of the input `center_latent`. |
| `output_batch_size` | Total output batch size (total_cells × center_batch). |
| `x_strength` / `y_strength` | Axis strength values used. |
| `interpolation_mode` | Mode string. |
| `slerp_anchor` | Which tensor was used as μ anchor (Slerp modes only). |
| `normalize_vectors` | Whether normalization was applied. |
| `clamp_output` / `clamp_range` | Clamping state. |
| `x_offsets` / `y_offsets` | List of signed offset coordinates for each axis. |
| `has_exact_center` | `True` when grid dims are both odd (center cell is exact copy of `center_latent`). |
| `output_shape` | Full output batch tensor shape `[B_total, C, H, W]`. |
| `device` / `elapsed_ms` | Hardware and timing. |
| `gimbal_grid_map` | Per-cell list: batch start/end index, grid col/row, offset coords, is_center, X/Y displacement norms. |

---

## 📝 `grid_report` Output

A human-readable plain-text report showing grid structure, perfect for logging and documentation:

```
=== GimbalManifold_Explorer ===
Grid:          3 x 3  (9 cells)
Center batch:  1  ->  Output batch: 9
X offsets:     [-1.0, 0.0, 1.0]  (strength +1.000)
Y offsets:     [-1.0, 0.0, 1.0]  (strength +1.000)
Mode:          Slerp
Normalize vec: True
Exact center:  True

Grid map (col x, row y)  * = center:
  [ -1.0,-1.0] [ +0.0,-1.0] [ +1.0,-1.0]
  [ -1.0,+0.0] [* +0.0,+0.0] [ +1.0,+0.0]
  [ -1.0,+1.0] [ +0.0,+1.0] [ +1.0,+1.0]
```

Connect this to a **Show Text** node to see your grid coordinates live in the workflow.

---

## 🖼️ Example Output

![Two quadrants of a 3×3 manifold grid — material/mood matrix](../../assets/test_runs/03_manifold_grid/03_manifold_slice_q1.jpg)

*Quadrant 1 of a 3×3 manifold grid: X = material direction, Y = temperature*

---

![Two quadrants of a 3×3 manifold grid — material/mood matrix](../../assets/test_runs/03_manifold_grid/03_manifold_slice_q2.jpg)

*Quadrant 2: opposite corner — note the compound concept at the intersection*

---

## 💡 Pro Tips

- **2D mood matrix:** Connect `GimbalCrossModalBridge` with prompt `"warm cinematic"` to `x_vector`, and `"cool ethereal"` to `y_vector`. Use a neutral scene latent as `center_latent`. The result is a 3×3 map of all combinations, decoded in one pass.

- **KSampler denoise is not optional:** The output batch is pure latent math — no denoising has occurred. Feed it into a KSampler at **denoise 0.45–0.60**. This is the sweet spot: enough denoising to crisp up the math artifacts, not enough to override the directional content.

- **Axis strength tuning:** Start both `x_strength` and `y_strength` at `1.0`. If corner cells look broken or too abstract, reduce to `0.75`. If center and corners look nearly identical, increase to `1.5`. The goal is perceptible but coherent variation.

- **Slerp for quality:** Switch from `Linear` to `Slerp` when your grid is 4×4 or larger, or when you're using high strength values. Linear mode drifts off the manifold at extreme offsets — Slerp stays on the Typical Set shell throughout.

- **Even vs. odd grid sizes:** Odd sizes (3×3, 5×5) include an **exact center cell** (a copy of `center_latent` with zero displacement). Even sizes (2×2, 4×4) do not — the center falls between cells. Use odd sizes when you need a reference anchor in the batch.

- **Batch indexing:** Cells are laid out row-by-row (Y=0 first, then Y=1, etc.) within the batch dimension. Use `gimbal_grid_map` from `grid_meta` to look up which batch index corresponds to which grid position.

---

## 🔬 Under the Hood *(Power User)*

### Grid Construction

For a grid of size X×Y, offset coordinates are generated as:

```python
x_offsets = [i - (X-1)/2  for i in range(X)]   # e.g. 3×3: [-1.0, 0.0, +1.0]
y_offsets = [j - (Y-1)/2  for j in range(Y)]
```

For each cell `(ox, oy)`, the actual displacement is:
```
Linear:  cell = center + x_vec×(x_strength×ox) + y_vec×(y_strength×oy)
Slerp:   dx = Slerp_μ(center, center+x_vec, |ox|×x_strength) − center
         dy = Slerp_μ(center, center+y_vec, |oy|×y_strength) − center
         cell = center + dx + dy
```

### Input Processing Pipeline

1. **Extract samples** — validates 4-D `[B, C, H, W]` shape.
2. **Device alignment** — coerces x/y vectors to `center_latent`'s device and dtype.
3. **Channel check** — raises `ValueError` if C mismatches between any inputs.
4. **Spatial resize** — bilinear interpolation in float32 if H/W differ, cast back to original dtype.
5. **Batch collapse** — if x/y vectors have B>1, they are averaged to a single vector (with a `UserWarning`).
6. **Direction conversion** — `x_vec = x_vec − center_col` (converts absolute latent positions to relative direction vectors).
7. **Normalization** — if enabled, unit-normalizes each direction vector: `v_hat = (v/‖v‖) × √D`.

### VRAM Ceiling

Max cells = **256** (hard-coded `MAX_GRID_CELLS`). A 16×16 grid at SDXL resolution (1×4×128×128) produces a batch of 256 tensors totaling ~256 MB in fp16. The node raises `ValueError` before allocating if the limit would be exceeded.

### `torch.no_grad()` Scope

All tensor operations run inside `with torch.no_grad()`. The output `dict` is assembled after the context manager exits, copying non-`samples` keys from `center_latent` to preserve ComfyUI metadata (e.g. `noise_mask`).

### `noise_mask` Propagation

If `center_latent` contains a `noise_mask`, it is automatically expanded or repeated to match the full output batch size — preventing downstream KSampler crashes when a masked latent is explored.

---

## ⚙️ Technical Reference

| Property | Value |
|---|---|
| ComfyUI class name | `GimbalManifold_Explorer` |
| Legacy alias | `WayfinderManifold_Explorer` |
| Function | `explore()` |
| Return types | `("LATENT", "DICT", "STRING")` |
| Return names | `("latent_batch", "grid_meta", "grid_report")` |
| Category | `Gimbal/Flight Instruments` |
| Max grid cells | 256 |
| VRAM mode | `torch.no_grad()` |

---

*Form & Noise Atelier — Gimbal Node Suite*
