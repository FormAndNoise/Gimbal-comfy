# 📍 Gimbal GPS Anchor (Save) & GPS Load

> *Found the perfect image? Save its exact mathematical coordinates. Come back tomorrow and it'll still be there.*

**Classes**: `GimbalGPS_Anchor` / `GimbalGPS_Load`  
**Category**: `Gimbal/Navigation`  
**VibeCheck**: 🟢 Stabilized  
**Anchor Returns**: `(LATENT, DICT, STRING)` → `anchored_latent`, `waypoint_meta`, `waypoint_report`

---

## What It Does

**GPS Anchor** extracts a single latent from a batch and saves its mathematical coordinates to a `.json` waypoint file on disk. **GPS Load** reads that file back and reconstructs the latent tensor, letting you resume exactly where you left off — in a different session, on a different machine, or when running batch production.

Think of it as a **save point for your latent space position**. Found the perfect lighting on image 4 of 9 in your manifold grid? Anchor it. Come back and load it. Or share the `.json` file with a collaborator.

---

## Anchor Inputs

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `latent_batch` | LATENT | — | A batch of latents. The node extracts one. |
| `select_index` | INT | 0 | Which image from the batch to anchor (0-indexed). |
| `save_waypoint` | BOOL | False | Whether to write the `.json` file to disk. |
| `waypoint_name` | STRING | `waypoint_01` | Filename stem. Saved as `{name}.json` in the output directory. |
| `enable_perf_logging` | BOOL | False | Console timing output. |
| `gimbal_meta` | DICT (optional) | — | Metadata from a previous Compass Pro or other Gimbal node. |
| `wayfinder_meta` | DICT (optional) | — | Legacy Wayfinder metadata (backward compat). |

---

## Anchor Outputs

| Output | Type | Description |
|:---|:---|:---|
| `anchored_latent` | LATENT | Single extracted latent from the batch at `select_index`. |
| `waypoint_meta` | DICT | Full navigational metadata including statistics, coordinates, and hash. |
| `waypoint_report` | STRING | Human-readable text report of the waypoint (great for logging nodes). |

---

## What Gets Saved in the Waypoint `.json`

```json
{
  "waypoint_name": "golden_arch_v3",
  "timestamp": "2026-08-23T00:00:00",
  "shape": [1, 4, 128, 128],
  "dtype": "float32",
  "device": "cpu",
  "statistics": {
    "min": -4.231,
    "max": 4.187,
    "mean": 0.0021,
    "std": 1.0034,
    "l2_norm": 512.4
  },
  "coordinate_hash": "a7f3c2d1...",
  "samples": [[[[...]]]]
}
```

The `coordinate_hash` is a cryptographic fingerprint of the latent tensor — useful for verifying that a loaded waypoint matches what was originally saved.

---

## Quick Wiring: Extracting a Winner from a Manifold Grid

```
[🗺️ Manifold Explorer] ─▶ latent_batch (9 latents)
                                    │
                         [📍 GPS Anchor]
                          select_index = 3      ← Pick the winner
                          save_waypoint = True
                          waypoint_name = "arch_golden_q2"
                                    │
                         anchored_latent ─▶ [KSampler] ─▶ [VAEDecode]
```

---

## The GPS Load Node

**GPS Load** is the companion to GPS Anchor. It reads a saved `.json` waypoint and outputs a latent you can wire into any Gimbal node.

```
[📂 GPS Load]
  waypoint_path = "output/gimbal/arch_golden_q2.json"
        │
  latent ─▶ [🧭 Compass Pro] (base_latent)
```

This is the foundation of the **Brand-Locked Lighting** workflow (Workflow 04): anchor the brand's photometric lighting signature once, then load it in every future session to ensure consistent brand aesthetics across all product shoots.

---

## Output Directory

The GPS Anchor writes to:
1. `$GIMBAL_OUTPUT_DIR` — if this environment variable is set.
2. `$WAYFINDER_OUTPUT_DIR` — legacy override.
3. `ComfyUI/output/gimbal/` — default (resolved via `folder_paths.get_output_directory()`).

---

## Pro Tips

- **Use with Manifold Explorer**: Generate a 3×3 grid (9 latents), use `select_index` 0–8 to preview each quadrant, then anchor the best one.
- **Brand lighting**: Anchor your approved brand photometric latent once. Load it in every product photography workflow. The entire team can share the `.json` file.
- **Seed recovery**: If you forget what seed produced a result, the `statistics` in the waypoint give you enough fingerprinting data to narrow it down.
- **Cross-session**: The anchored latent is saved as raw tensor data — it works across ComfyUI restarts, machine migrations, and model swaps (as long as the latent spatial dimensions match the target model).

---

## Under the Hood (Researchers)

- **Extraction**: Uses tensor slicing `samples[select_index:select_index+1]` to produce a `[1, C, H, W]` latent.
- **Hashing**: SHA-256 hash computed over the tensor's byte representation (`tensor.numpy().tobytes()`). Deterministic across sessions for identical tensors.
- **Storage format**: JSON with tensor serialized as a nested Python list (`tensor.tolist()`). Human-readable and version-control-friendly.
- **Metadata chain**: The optional `gimbal_meta` input allows chaining metadata from upstream nodes (Compass Pro, etc.) into the waypoint's provenance record.
- **Non-dict meta guard**: If `accumulated_position` in meta is not a dict (e.g., None, string), the node logs a warning and defaults to `{"x": 0.0, "y": 0.0}` — preventing `AttributeError` crashes.
- **Filename sanitization**: Waypoint names are regex-sanitized to `[a-zA-Z0-9_\-]` with a max length of 64 characters (configurable via `GIMBAL_MAX_FILENAME_LENGTH` env var).

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
