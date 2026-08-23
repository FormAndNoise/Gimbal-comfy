# 📍 Workflow 04: Brand-Locked Lighting

> *Extract a brand lighting grammar into a cryptographic GPS waypoint and project it orthogonally across product catalogs.*

**Primary Flight Instruments**: `GimbalGPS_Anchor` + `GimbalGPS_Load` + `GimbalCompass_Pro` (Orthogonal mode) + `GimbalLatentStabilizer`  
**Architecture**: SDXL 4-channel & FLUX.1 16-channel  
**Workflow Files**: `workflows/ui/Gimbal_04_BrandLocked.json` | `workflows/api_flux/API_FLUX_Gimbal_04_BrandLocked.json`

---

## Overview

Maintaining visual consistency across e-commerce catalogs or marketing campaigns typically requires rigid prompt repetition that degrades with diverse subjects. **Workflow 04** decouples the lighting and tonal atmosphere of an approved brand hero image and serializes it to a `.json` waypoint. That waypoint is then recalled and orthogonally projected onto completely unrelated products (e.g. perfume → chair → audio hardware) with zero geometry interference.

---

## Visual Showcase

| Approved Brand Hero (Perfume) | Transferred: Speaker Hardware | Transferred: Lounge Chair |
| :---: | :---: | :---: |
| ![Hero](../../assets/test_runs/04_brand_locked/01_anchor_perfume_crimson.png) | ![Speaker](../../assets/test_runs/04_brand_locked/02_speaker_crimson_slit.png) | ![Chair](../../assets/test_runs/04_brand_locked/04_chair_black_leather_chrome.png) |
| *Crimson key lighting anchor* | *100% lighting & contrast match* | *Consistent studio speculars* |

---

## Node Chain Wiring Architecture

### Stage 1: Anchor the Brand Waypoint
```
[Brand Hero Image] ──▶ [VAE Encode] ──▶ BRAND_LATENT
                                            │
                                  [📍 GPS Anchor]
                                    select_index = 0
                                    save_waypoint = True
                                    waypoint_name = "atelier_crimson_lighting_v1"
                                            │
                                     Saved to disk: output/gimbal/atelier_crimson_lighting_v1.json
```

### Stage 2: Orthogonal Transfer to New Products
```
[📂 GPS Load] "atelier_crimson_lighting_v1.json" ──▶ BRAND_WAYPOINT_LATENT

[New Product Latent: Chair / Speaker] ──▶ PRODUCT_LATENT

[🧭 Compass Pro]
  base_latent = PRODUCT_LATENT
  target_latent = BRAND_WAYPOINT_LATENT
  origin_latent = Empty Latent (Zero Reference)
  mode = "Orthogonal_Projection"
  strength = 1.50
        │
  STEERED_PRODUCT_LATENT
        │
[🛡️ Latent Stabilizer] psi = 0.88, scale_cap = 8.0
        │
[KSampler: Refine] denoise = 0.50, CFG = 3.8, steps = 20
        │
[VAEDecode] ──▶ [SaveImage]
```

---

## Technical Calibration Specifications

| Parameter | Setting | Rationale |
| :--- | :---: | :--- |
| **Compass Mode** | `Orthogonal_Projection` | Projects lighting vectors without corrupting product contours. |
| **Compass Strength** | `1.50` | Full stylistic projection amplitude. |
| **Stabilizer $\psi$** | `0.88` | Prevents high-CFG boundary clipping. |
| **Refine Denoise** | `0.50` | Locks product silhouette while baking in lighting tones. |

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
