<!-- Gimbal Node Suite — Form & Noise Atelier -->
<!-- Navigate latent space with precision flight instruments, not lottery prompts. -->

# 🔀 Workflow 09: Subspace Material Matrix

![Subspace Material Matrix Showcase](../../assets/brand/gimbal_09_subspace_material_showcase.png)

> **VibeCheck** 🟢 Stabilized &nbsp;|&nbsp; Category: `Gimbal/Decomposition` + `Gimbal/Subspace` &nbsp;|&nbsp; Primary Nodes: **GimbalChannelSplit · GimbalCrossModalBridge · GimbalChannelMerge · GimbalLatentStabilizer**

---

## Overview

The Subspace Material Matrix is Gimbal's most surgical workflow. It **decomposes** a latent into frequency bands — separating the structural geometry channels from the material/texture channels — steers only the material channels with a semantic direction vector, then **recomposes** the latent with the frozen geometry intact. The result is 100% silhouette lock with complete material transformation: the same building, chair, or product in obsidian, velvet, liquid chrome, or any material you can describe in five words.

This works because latent channels are not all equal. In SDXL's 4-channel latent space, channels 0–1 encode low-frequency structural information (edges, form, spatial layout) while channels 2–3 carry high-frequency material and texture information. The Channel Split node isolates these bands so you can operate on them independently.

---

## 🖼️ What You'll Create

- Material variants of an existing render with zero structural deviation
- A library of material studies from a single base latent (concrete → obsidian → gold travertine → liquid chrome)
- Product visualization, architectural material boards, and furniture specification sheets — all from one geometry capture

---

## 🗺️ Node Chain Diagram (SDXL 4-Channel)

```
[Load Image / VAE Encode] ──────────────────── BASE_LATENT
                                                    │
[🔀 GimbalChannelSplit]  ◄──────────────────────────┘
  └── split_index = 2
  │
  ├── low_channels  (Ch 0–1: geometry) ─────────── FREEZE ──────────────────────┐
  └── high_channels (Ch 2–3: material) → STEER                                  │
              │                                                                  │
[🌉 GimbalCrossModalBridge]                                                      │
  ├── instruction = "cold monochrome crisp sharp"                                │
  └── mode        = Keyword_Heuristics                                           │
  ↓ delta_vector                                                                 │
                                                                                 │
[🧭 GimbalCompass_Pro]                                                           │
  ├── base          = high_channels (Ch 2–3)                                     │
  ├── target        = delta_vector                                                │
  ├── mode          = Orthogonal_Projection                                      │
  └── strength      = 1.5                                                        │
  ↓ steered_material                                                             │
                                                                                 │
[🛡️ GimbalLatentStabilizer]                                                      │
  ├── truncation_psi = 0.88                                                      │
  └── scale_cap      = 8.0                                                       │
  ↓ stable_material                                                              │
                                                                                 │
[🔁 GimbalChannelMerge]  ◄──────────────────── FROZEN low_channels ─────────────┘
  ├── low_channels  = low_channels (frozen Ch 0–1)
  └── high_channels = stable_material (steered Ch 2–3)
  ↓ merged_latent

[KSampler]  (CFG=4.5, denoise=0.65, steps=20)
  ↓
[VAE Decode] → [Save Image]
```

---

## ⚙️ Settings That Work

Verified on SDXL 1.0 Base and FLUX.1-dev.

| Parameter | Value | Notes |
|---|---|---|
| Channel Split `split_index` (SDXL) | `2` | Separates Ch0–1 (structure) from Ch2–3 (material) |
| Channel Split `split_index` (FLUX / SD3) | `8` | FLUX uses 16-channel latent; structure lives in Ch0–7 |
| Compass `mode` | `Orthogonal_Projection` | Steers material without contaminating structure |
| Compass `strength` | `1.5` | Orthogonal is gentle — 1.5 gives visible material shift |
| Stabilizer `truncation_psi` | `0.88` | **Critical** — prevents black-outline artifacts at this denoise level |
| Stabilizer `scale_cap` | `8.0` | Clamps channel blow-out |
| KSampler `denoise` | `0.65` | High enough to commit material, low enough to preserve silhouette |
| KSampler `cfg` | `4.5` | Balanced — preserves both material and structural cues |

### Split Index Guide

| Architecture | Split Index | Structure Channels | Material Channels |
|---|---|---|---|
| SD 1.5 / SDXL | `2` | Ch 0–1 | Ch 2–3 |
| FLUX.1 / SD3 | `8` | Ch 0–7 | Ch 8–15 |

---

## 🎨 Material Presets

Enter these keyword strings in the Cross-Modal Bridge `instruction` field:

| Material | Keywords | Notes |
|---|---|---|
| Liquid Chrome | `cold monochrome crisp sharp bright` | Pushes toward reflective metallic neutrals |
| Oxblood Velvet | `warm dark saturated moody fire` | Rich dark reds and deep absorption |
| Gold Travertine | `warm golden punchy crisp` | Natural stone with warm mineral undertones |
| Obsidian Glass | `dark cold monochrome sharp contrast` | High contrast, deep blacks, glass-like reflection |
| Oat Linen (neutral) | `soft muted flat desaturated` | Textured neutral — good reset/baseline material |

---

## 🔧 Step-by-Step Wiring Instructions

1. **Encode your base image.** Use Load Image → VAE Encode (with the checkpoint's VAE) to produce `BASE_LATENT`.
2. **Add GimbalChannelSplit.**
   - Wire `BASE_LATENT` into the input.
   - Set `split_index = 2` for SDXL, `8` for FLUX.
   - Outputs: `low_channels` (geometry — do not modify) and `high_channels` (material — to be steered).
3. **Add GimbalCrossModalBridge.**
   - Set `instruction` to one of the material preset keyword strings above.
   - Set `mode = Keyword_Heuristics`.
   - Output: `delta_vector` — the material direction.
4. **Add GimbalCompass_Pro.**
   - `base` → `high_channels` from Channel Split
   - `target` → `delta_vector` from Cross-Modal Bridge
   - `mode` → `Orthogonal_Projection`
   - `strength` → `1.5`
   - Output: `steered_material`
5. **Add GimbalLatentStabilizer.**
   - Wire `steered_material` into the input.
   - `truncation_psi = 0.88`, `scale_cap = 8.0`
   - Output: `stable_material`
6. **Add GimbalChannelMerge.**
   - `low_channels` → the **frozen** `low_channels` from Step 2 (geometry untouched)
   - `high_channels` → `stable_material` from Step 5
   - Output: `merged_latent` — your complete, geometry-preserved, material-steered latent.
7. **Add KSampler.**
   - `latent_image` → `merged_latent`
   - `cfg = 4.5`, `denoise = 0.65`, `steps = 20`
   - Use the same prompt as the original image.
8. **Add VAE Decode → Save Image.**
9. **To generate material variants:** Duplicate the Cross-Modal Bridge and Compass Pro nodes, swap the keyword string, and queue again. The geometry is re-used from the same split — only the material steering changes.

---

## 🖼️ Gallery

### Architecture — Material Studies

| Baseline (Concrete) | Obsidian | Travertine |
|---|---|---|
| ![Concrete](../../assets/test_runs/architectural_showcases/09_mat_ctrl_concrete.png) | ![Obsidian](../../assets/test_runs/architectural_showcases/09_mat_obsidian.png) | ![Travertine](../../assets/test_runs/architectural_showcases/09_mat_travertine.png) |

| Gold Travertine V2 | Obsidian V2 |
|---|---|
| ![Gold Travertine V2](../../assets/test_runs/architectural_showcases/09_v2_gold_travertine.png) | ![Obsidian V2](../../assets/test_runs/architectural_showcases/09_v2_obsidian.png) |

### Furniture — Material Studies

| Baseline (Oat Linen) | Liquid Chrome | Oxblood Velvet |
|---|---|---|
| ![Oat Linen](../../assets/test_runs/fresh_run6_exploration/09_chair_ctrl_oat_linen.png) | ![Liquid Chrome](../../assets/test_runs/fresh_run6_exploration/09_chair_steered_liquid_chrome.png) | ![Oxblood Velvet](../../assets/test_runs/fresh_run6_exploration/09_chair_steered_oxblood_velvet.png) |

---

## 🛠️ Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Silhouette / geometry is distorted | `denoise` too high OR wrong Compass mode | Lower `denoise` to `0.55`; confirm mode is `Orthogonal_Projection` |
| Material change is invisible | `strength` too low | Raise Compass `strength` to `2.0`; also check Bridge mode is `Keyword_Heuristics` |
| Black outlines / fringe artifacts | Stabilizer not inserted | Add GimbalLatentStabilizer between Compass and Channel Merge |
| Wrong architecture split (structure looks melted) | `split_index` set for wrong model | SDXL = `2`, FLUX/SD3 = `8` — confirm your checkpoint architecture |
| All material variants look the same | Keywords not contrasting enough | Use more extreme keyword strings; contrast warm/dark with cold/bright |
| Channel Merge output looks like two images overlaid | Channels from different-resolution latents | Ensure Split and Merge are using the same `BASE_LATENT` source — no resize in between |

---

## 🔬 Power User Notes

### Why Channels 0–1 Carry Structure in SDXL

The VAE encoder in SDXL applies a KL-divergence regularized compression that, empirically, distributes information by spatial frequency. Channels 0 and 1 capture the dominant low-frequency gradients — the coarse structure. Channels 2 and 3 carry higher-frequency residuals — fine texture, material response to light, surface grain. This is not a strict architectural choice but an emergent property of the training. It is consistent enough across models that `split_index=2` is reliable for SDXL-family VAEs.

For FLUX.1 (16-channel latent), the same principle applies but the split point shifts to `8` due to the expanded channel count.

### Operating on Only One Channel

You are not limited to a binary split. For extreme precision, use `split_index=1` to isolate Channel 0 alone (primary low-frequency structure), or `split_index=3` to isolate just Channel 3 (highest-frequency material residual). Experiment with partial steers at different split points to find the exact boundary for your source image.

### Chaining Multiple Material Steers

You can stack material steers sequentially:
1. Run the full workflow with `"warm golden punchy crisp"` → save the merged latent.
2. Re-encode the saved merged latent with VAE Encode.
3. Run the workflow again with `"crisp sharp contrast"`.

Each pass refines the material further. Use low `strength` values (0.8–1.0) per pass to avoid accumulation of artifacts.

### Building a Material Board

Connect the Channel Merge output to a **Batch Latent** node (if available in your ComfyUI setup) and attach multiple Compass Pro nodes with different material vectors. Process the entire material library in one queue pass, producing a complete material board in a single run.

---

## 📁 Workflow Files

| Format | Path |
|---|---|
| ComfyUI drag-and-drop | [`workflows/ui/Gimbal_09_SubspaceMaterialMatrix.json`](../../workflows/ui/Gimbal_09_SubspaceMaterialMatrix.json) |
| API / FLUX variant | [`workflows/api_flux/API_FLUX_Gimbal_09_SubspaceMaterialMatrix.json`](../../workflows/api_flux/API_FLUX_Gimbal_09_SubspaceMaterialMatrix.json) |
| API / SDXL variant | [`workflows/api_sdxl/API_Gimbal_09_SubspaceMaterialMatrix.json`](../../workflows/api_sdxl/API_Gimbal_09_SubspaceMaterialMatrix.json) |

---

*Gimbal Node Suite — Form & Noise Atelier*
*Navigate latent space with precision flight instruments, not lottery prompts.*
