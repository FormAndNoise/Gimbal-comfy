# 🎭 Workflow 07: Likeness Isolator

> *Isolate and steer character and subject identity tokens independently from environmental style, lighting, and camera angle.*

**Primary Flight Instruments**: `GimbalLikenessIsolator` + `GimbalCompass_Pro` (Orthogonal mode)  
**Architecture**: SDXL 4-channel & FLUX.1 16-channel  
**Workflow Files**: `workflows/ui/Gimbal_07_LikenessIsolator.json`

---

## Overview

When training character LoRAs, style and lighting from training photos often bleed into the weights. **Workflow 07** isolates facial identity tokens into an orthogonal latent vector, allowing character likeness to be applied to completely unrelated scenes without dragging along lighting or clothing artifacts.

---

## Visual Showcase

| Concept A: Source Identity | Concept B: Clean Baseline | Concept C: Target Recipient | Locked Likeness Result |
| :---: | :---: | :---: | :---: |
| ![Source](../../assets/test_runs/fresh_run6_exploration/01_concept_A_glasses_amber.png) | ![Clean](../../assets/test_runs/fresh_run6_exploration/02_concept_B_clean_baseline.png) | ![Target](../../assets/test_runs/fresh_run6_exploration/03_concept_C_recipient_clean.png) | ![Result](../../assets/test_runs/fresh_run6_exploration/05_analogy_ortho_norm_locked.png) |
| *Amber glasses identity* | *Neutral control baseline* | *Recipient model* | *Orthogonal transfer* |

---

## Node Chain Wiring Architecture

```
[Checkpoint Loader] ──▶ MODEL, CLIP

[🎭 Gimbal Likeness Isolator]
  lora_name = "character_identity_v1.safetensors"
  strength = 1.0
  alpha = 1.0
  likeness_mask = 0.80  (Preserves facial geometry, strips training lighting)
        │
  PATCHED_MODEL, PATCHED_CLIP
        │
[CLIPTextEncode: "Cinematic portrait in neon cyberpunk alley"] ──▶ CONDITIONING
[EmptyLatentImage] ──▶ [KSampler: denoise=1.0, CFG=5.0] ──▶ IDENTITY_LATENT

[🧭 Compass Pro] (Optional Orthogonal Locking)
  base_latent = IDENTITY_LATENT
  mode = "Orthogonal_Projection"
  strength = 1.2
        │
[🛡️ Latent Stabilizer] psi = 0.90
        │
[VAEDecode] ──▶ [SaveImage]
```

---

## Parameter Guidelines

| Parameter | Setting | Impact |
| :--- | :---: | :--- |
| `likeness_mask` | `0.80` | High identity transfer with background decoupling. |
| `alpha` | `1.00` | Full network injection without weight decay. |
| `Stabilizer psi` | `0.90` | Eliminates skin posterization and specular clipping. |

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
