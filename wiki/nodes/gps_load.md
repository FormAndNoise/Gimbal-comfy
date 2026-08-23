# 📂 Gimbal GPS Load (Recall Waypoint)

> *Instantly recall saved cryptographic latent coordinates and navigational provenance metadata from disk.*

**Class**: `GimbalGPS_Load`  
**Category**: `Gimbal/Navigation`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT, DICT, STRING)` → `loaded_latent`, `waypoint_meta`, `waypoint_report`

---

## What It Does

`GimbalGPS_Load` is the inverse of `GimbalGPS_Anchor`. It reads a `.json` waypoint file created during an earlier session (or shared by a team member) and reconstructs the exact multi-channel latent tensor along with its historical flight log and coordinate hash.

---

## Inputs

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `waypoint_file` | STRING | `waypoint_01.json` | Filename or relative path to the saved waypoint JSON. |
| `device` | enum | `auto` | Target device (`auto`, `cuda`, `cpu`). |
| `dtype` | enum | `auto` | Target precision (`auto`, `float32`, `float16`, `bfloat16`). |

---

## Operational Workflow

```
[📂 GPS Load]
  waypoint_file = "brand_photometric_v1.json"
        │
  loaded_latent ──▶ [🧭 Compass Pro] (target_latent)
  waypoint_meta ──▶ [📊 Diagnostics] (audit provenance)
```

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
