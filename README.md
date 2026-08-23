<div align="center">

<img src="assets/brand/gimbal_avatar_512.png" width="96" alt="Gimbal Flight Instruments" />

# Gimbal Node Suite

### *Navigate latent space with precision flight instruments, not lottery prompts.*

**A ComfyUI custom node suite by [Form & Noise](https://github.com/form-and-noise)** &nbsp;|&nbsp; 🟢 Stabilized

[![Tests](https://img.shields.io/badge/tests-220%20passed%2C%200%20failed-brightgreen?style=flat-square)](./wiki/concepts/lamnr_framework.md)
[![GPU Certified](https://img.shields.io/badge/GPU-RTX%203060%20live%20run-blue?style=flat-square)](./wiki/visual_asset_index.md)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange?style=flat-square)](https://github.com/comfyanonymous/ComfyUI)
[![SDXL](https://img.shields.io/badge/SDXL-4ch-teal?style=flat-square)](./wiki/concepts/channel_architecture.md)
[![FLUX.1](https://img.shields.io/badge/FLUX.1-16ch-purple?style=flat-square)](./wiki/concepts/channel_architecture.md)

> 🎙️ **Flight Deck Announcement**: *"Ladies and gentlemen, please stow your random seed generators in the upright position. We have reached cruising altitude in $\mathbb{R}^{65,536}$ and the slot machine is officially out of service."*

</div>

---

## 🧭 Master Navigation Index

- **[What Is Gimbal?](#what-is-gimbal)**
- **[Visual Showcase & Galleries](#-visual-showcase--galleries)**
- **[Installation](#-installation)**
- **[🎯 Getting Started & How to Use](#-getting-started--how-to-use-guide)**
  - [The Core Mental Model](#the-core-mental-model)
  - [Tutorial 1: Your First Concept Blend (2-Minute Quick Start)](#tutorial-1-your-first-concept-blend-2-minute-quick-start)
  - [Tutorial 2: Natural Language Lighting Steering](#tutorial-2-natural-language-lighting-steering)
  - [Tutorial 3: 2D Neighborhood Mapping (Manifold Grid)](#tutorial-3-2d-neighborhood-mapping-manifold-grid)
  - [Tutorial 4: Bookmarking Latents with GPS Waypoints](#tutorial-4-bookmarking-latents-with-gps-waypoints)
  - [Golden Rules of Latent Flight](#the-5-golden-rules-of-latent-flight)
- **[🛠️ Complete Flight Instrument Catalog (Node Reference)](#-complete-flight-instrument-catalog-nodes)**
- **[📋 Canonical Workflow Guides](#-canonical-workflow-guides)**
- **[🔬 Mathematical & Theoretical Deep Dives](#-mathematical--theoretical-deep-dives)**
- **[📖 Project History](#-project-history)**
- **[📚 Documentation Archives & Research Papers](./docs/README.md)**
- **[🏷️ Brand Standards](#-brand-standards)**

---

## What Is Gimbal?

Every diffusion artist knows the feeling: you have a vivid creative vision, but your tools offer a high-stakes slot machine. Change the seed, roll again, pray to the VAE gods, and hope something vaguely close emerges from the noise. Gimbal exists to officially decommission that loop.

**Gimbal is a suite of latent-space flight instruments** — 19 ComfyUI custom nodes that let you navigate the high-dimensional manifold of a diffusion model the way a pilot navigates the sky: with exact coordinates, heading, altitude, and intent. Instead of re-rolling seeds and hoping a random walk lands somewhere interesting, you plot a flight plan, hold your orientation, and land right on your target runway.

> [!TIP]
> **Co-Pilot's Log**: *Why spend 45 minutes re-rolling seeds like a casino regular when you can vector-steer straight to your target coordinates and still have time for a coffee break?*

### Core Capabilities at a Glance

| Creative Goal                                                              | Primary Flight Instrument                                                              | Underlying Mathematical Engine                        |
| :------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- | :---------------------------------------------------- |
| **Blend two concepts** (e.g. Redwood Forest $\leftrightarrow$ Alpine Peak) | 🧭 [Compass Pro](./wiki/nodes/compass_pro.md)                                               | Geodesic Spherical Linear Interpolation ($\mu$-SLERP) |
| **Steer atmosphere with words** ("golden hour soft cinematic")             | 🌉 [Cross-Modal Bridge](./wiki/nodes/crossmodal_bridge.md)                                  | Calibrated keyword-to-subspace signature projection   |
| **Survey 9 variations in a 2D grid**                                       | 🗺️ [Manifold Explorer](./wiki/nodes/manifold_explorer.md)                                  | Orthogonal 2D topological surface mapping             |
| **Generate seamless 360° animation loops**                                 | 🔄 [Circular Orbit](./wiki/nodes/circular_orbit.md)                                         | Constant-radius closed-loop geodesic trajectories     |
| **Lock brand lighting across 1,000 products**                              | 📍 [GPS Anchor](./wiki/nodes/gps_anchor.md) + 📂 [GPS Load](./wiki/nodes/gps_load.md)            | Cryptographic coordinate hashing & disk caching       |
| **Dial one attribute without changing others**                             | 🎚️ [Semantic Slider](./wiki/nodes/semantic_slider.md)                                      | Real-time SVD/PCA batch covariance decomposition      |
| **Swap material, lock 100% silhouette**                                    | 🔀 [Channel Split](./wiki/nodes/channel_split.md) + 🔁 [Merge](./wiki/nodes/channel_merge.md)    | Subspace frequency band decoupling (4ch / 16ch)       |
| **Eliminate posterization & black outlines**                               | 🛡️ [Latent Stabilizer](./wiki/nodes/latent_stabilizer.md)                                  | Low-rank Woodbury MMSE denoiser & variance truncation |
| **Audit latent probability & OOD distance**                                | 📊 [Diagnostics](./wiki/nodes/diagnostics.md) + 📡 [Telemetry](./wiki/nodes/latent_telemetry.md) | Exact log-likelihood & Mahalanobis distance scoring   |

---

## 🖼️ Visual Showcase & Galleries

<p align="center">
  <img src="assets/brand/gimbal_social_preview.png" alt="Gimbal Social Preview" width="800"/>
</p>

<table>
<tr>
<td align="center"><img src="assets/brand/gimbal_02_cinematic_steering_showcase.png" width="380"/><br/><strong>Workflow 02: Cinematic Steering</strong><br/><em>100% vehicle silhouette lock with text-steered cyberpunk lighting.</em></td>
<td align="center"><img src="assets/brand/gimbal_08_harmonic_orbiter_showcase.png" width="380"/><br/><strong>Workflow 08: Harmonic Orbiter</strong><br/><em>Constant-radius closed-loop geodesic 360° tour.</em></td>
</tr>
<tr>
<td align="center"><img src="assets/brand/gimbal_09_subspace_material_showcase.png" width="380"/><br/><strong>Workflow 09: Subspace Material</strong><br/><em>Subspace channel decoupling (Concrete → Chrome → Velvet).</em></td>
<td align="center"><img src="assets/brand/gimbal_04_semantic_slider_showcase.png" width="380"/><br/><strong>Workflow 04: Brand Lighting</strong><br/><em>Photometric lighting transfer across luxury product categories.</em></td>
</tr>
</table>

### Sample Output Comparisons

| Concept Blender | Text Steered Portrait | Manifold Grid Slice | Semantic Slider |
|:---:|:---:|:---:|:---:|
| ![Forest ↔ Mountain Blend](assets/test_runs/01_concept_blender/01_concept_blend_forest_mountain.jpg) | ![Text Steered Portrait](assets/test_runs/02_text_steered/02_text_steered_portrait.jpg) | ![Manifold Grid Slice Q1](assets/test_runs/03_manifold_grid/03_manifold_slice_q1.jpg) | ![Semantic Slider Portrait](assets/test_runs/05_semantic_slider/05_semantic_slider_portrait.jpg) |
| `GimbalCompass_Pro` SLERP $t=0.50$ | `GimbalCrossModalBridge` + Orthogonal | `GimbalManifold_Explorer` 3×3 grid | `GimbalSemanticSlider` PC-0 $\pm 1.5$ |

> 📁 **Explore all 206 test renders and 3 interactive HTML HUD galleries in the [Visual Asset & Gallery Index](./wiki/visual_asset_index.md)**.

---

## 🚀 Installation

### System Requirements
- **ComfyUI** (any recent build)
- **Python**: 3.10+
- **PyTorch**: 2.0+ with CUDA (RTX 3060 / 40-series tested) or CPU
- **Supported Architectures**: SD 1.5, SDXL 1.0 (4-channel), SD 3.5, FLUX.1 (16-channel)

### Step-by-Step Installation

```bash
# 1. Navigate to your ComfyUI custom_nodes folder
cd ComfyUI/custom_nodes

# 2. Clone the repository
git clone https://github.com/form-and-noise/ComfyUI-Gimbal.git

# 3. Install requirements
pip install -r ComfyUI-Gimbal/requirements.txt

# 4. Restart ComfyUI
```

All nodes will appear in the ComfyUI **Add Node → Gimbal/*** menu, organized into 7 logical flight categories.

---

## 🎯 Getting Started & How to Use Guide

### The Core Mental Model

Traditional diffusion treats latent tensors as random noise that the UNet denoises into an image. Gimbal treats the latent tensor $\mathbf{z} \in \mathbb{R}^{C \times H \times W}$ as a **geometric point in a high-dimensional vector space** ($D = 65,536$ dimensions).

```
   ┌─────────────────────────────────────────────────────────────┐
   │                    THE GIMBAL FLIGHT DECK                   │
   │                                                             │
   │   🧭 COMPASS PRO        🗺️ MANIFOLD EXPLORER   📍 GPS ANCHOR│
   │   [Vector Steering]     [2D Topology Grids]    [Save Point] │
   │          ▲                       ▲                   ▲      │
   │          └───────────────────────┼───────────────────┘      │
   │                                  │                          │
   │                    🌉 CROSS-MODAL BRIDGE                    │
   │                    [Natural Language Input]                 │
   │                                  │                          │
   │                    🛡️ LATENT STABILIZER                     │
   │                    [LAMNr Quality Filter]                   │
   └─────────────────────────────────────────────────────────────┘
```

---

### Tutorial 1: Your First Concept Blend (2-Minute Quick Start)

**Goal**: Blend a Redwood Forest with a Mountain Summit at a precise 50% geometric midpoint without muddy colors.

```
┌─────────────────────────────────────────────────────────────────┐
│              CONCEPT BLENDER — WIRING DIAGRAM                    │
│                                                                   │
│  [Prompt A: "dense redwood forest"]                               │
│       └──► KSampler A (denoise=1.0) ──► latent_A ──┐             │
│                                                     ▼             │
│  [Prompt B: "rocky mountain summit"] ──► 🧭 Gimbal Compass Pro   │
│       └──► KSampler B (denoise=1.0) ──► latent_B ──┘             │
│                                           (mode: Slerp, t=0.50)   │
│                                                │                  │
│                                                ▼                  │
│                                         KSampler (final)         │
│                                         (denoise=0.90, CFG=5.5)   │
│                                                │                  │
│                                                ▼                  │
│                                           VAEDecode ──► 🖼️ Image  │
└─────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step Instructions:
1. Create two standard text prompts: Prompt A (*Redwood Forest*) and Prompt B (*Mountain Summit*).
2. Wire each prompt into its own `KSampler` running `denoise = 1.0` to generate two raw concept latents (`latent_A` and `latent_B`).
3. Add a **🧭 Gimbal Compass Pro** node (`Add Node → Gimbal/Flight Instruments → Gimbal Compass Pro`).
4. Connect `latent_A` to `base_latent` and `origin_latent`. Connect `latent_B` to `target_latent`.
5. Set `mode = "Slerp"` and `strength = 0.50`.
6. Pass `latent_out` into a final `KSampler` set to `denoise = 0.90`, `CFG = 5.5`, `steps = 25`.
7. Decode with `VAEDecode` and save. You now have a photorealistic, crisp 50% hybrid environment.

> 📖 **Full Workflow Guide**: [Workflow 01: Concept Blender](./wiki/workflows/wf_01_concept_blender.md)

---

### Tutorial 2: Natural Language Lighting Steering

**Goal**: Take an existing car or portrait and change the atmosphere to "Cyberpunk Midnight" while locking the vehicle silhouette 100%.

```
[Base Latent: Studio Car Render] ──┐
                                   ▼
[🌉 Cross-Modal Bridge] ──► 🧭 Compass Pro (Orthogonal_Projection, strength=1.5)
  instruction = "dark cool neon"   │
                                   ▼
                        [🛡️ Latent Stabilizer] (psi=0.88, scale_cap=8.0)
                                   │
                                   ▼
                        [KSampler: Refine] (denoise=0.55, CFG=3.8) ──► [VAEDecode]
```

1. Generate or VAE-encode your base subject into `BASE_LATENT`.
2. Add a **🌉 Gimbal Cross-Modal Bridge** node. Type `"dark cool neon cyberpunk cinematic"` into `instruction`.
3. Connect `direction_latent` from the Bridge to `target_latent` of a **🧭 Compass Pro**. Connect `BASE_LATENT` to `base_latent`.
4. Set Compass mode to `Orthogonal_Projection` and `strength = 1.50`.
5. Wire `latent_out` through a **🛡️ Gimbal Latent Stabilizer** (`truncation_psi = 0.88`, `scale_cap = 8.0`).
6. Pass into a refinement `KSampler` with `denoise = 0.55` and `CFG = 3.8`. The car's body panels and reflections remain identical, but all lighting and background atmosphere shift to neon cyberpunk.

> 📖 **Full Workflow Guide**: [Workflow 02: Text-Steered Lighting](./wiki/workflows/wf_02_text_steered.md)

---

### Tutorial 3: 2D Neighborhood Mapping (Manifold Grid)

**Goal**: Explore 9 variations of a concept across two independent visual axes in a single render batch.

1. Connect your center starting latent to `center_latent` of **🗺️ Gimbal Manifold Explorer**.
2. Connect a "Warm Earthy" vector to `x_vector` and a "Cool Obsidian" vector to `y_vector` (from Cross-Modal Bridge).
3. Set `grid_size_x = 3`, `grid_size_y = 3`, `x_strength = 1.5`, `y_strength = 1.5`, `interpolation_mode = "Slerp"`.
4. Connect the output `latent_batch` (9 latents) to a refinement `KSampler` with `denoise = 0.50`.
5. Pass decoded images to **🪡 Gimbal Grid Stitch** (`columns = 3`) to view all 9 variations in a clean contact sheet.

> 📖 **Full Workflow Guide**: [Workflow 03: Manifold Grid](./wiki/workflows/wf_03_manifold_grid.md)

---

### Tutorial 4: Bookmarking Latents with GPS Waypoints

**Goal**: Extract the best image from a 9-image manifold grid, save its coordinates to disk, and reload it in a future session.

1. Add a **📍 Gimbal GPS Anchor** node. Connect `latent_batch` from your Manifold Explorer.
2. Set `select_index = 4` (picks the center cell) and `save_waypoint = True`.
3. Give it a name: `waypoint_name = "golden_arch_hero_v1"`.
4. When executed, Gimbal writes `output/gimbal/golden_arch_hero_v1.json` with full tensor coordinates, statistics, and cryptographic SHA-256 hash.
5. In any future workflow or session, add a **📂 Gimbal GPS Load** node, point to that JSON file, and immediately resume navigation from that exact location.

> 📖 **Full Workflow Guide**: [Workflow 04: Brand-Locked Lighting](./wiki/workflows/wf_04_brand_locked.md)

---

### The 5 Golden Rules of Latent Flight

> 🛫 **Pre-Flight Safety Checklist**: *Disregarding these rules may result in mid-air spatial collisions, deep-fried pixel turbulence, or unexpected character re-skinning.*

1. **Step-0 Noise vs. Mid-Denoise (The Early Bird Rule)**: Always perform concept blends (SLERP) on **Step-0 Gaussian initial noise** before spatial feature maps crystallize into stubborn real estate. Intercepting latents at Step 8 is like trying to redesign an airplane while it's landing—it forces grotesque boundary re-skinning (tree trunks morphing violently into rock spires).
2. **Refinement Denoise Sweet Spot ($0.45 – 0.60$)**: When refining latents modified by Compass Pro or Manifold Explorer, keep denoise between $0.45$ and $0.60$. Higher denoise ($>0.70$) completely overwrites your carefully calculated flight path; lower denoise ($<0.35$) leaves behind raw, un-denoised math artifacts that look like abstract mathematical soup.
3. **Drop CFG During Refinement ($3.5 – 4.5$) (Don't Yell at the UNet)**: Since Gimbal has already injected clear semantic direction into the latent tensor, high guidance ($>6.5$) will over-drive the signal and deep-fry your images with crispy black wireframe outlines.
4. **Use Orthogonal Projection for Geometry Lock**: When steering lighting or atmosphere on an existing image, always set `Orthogonal_Projection` mode. This decomposes the change vector perpendicular to your subject—allowing you to repaint the sky neon cyberpunk without accidentally altering the shape of your sports car.
5. **Always Deploy Latent Stabilizer After Steering**: When applying high text-steering gains, insert `GimbalLatentStabilizer` ($\psi = 0.88$, $\text{cap} = 8.0$) before your final KSampler. Think of it as your automatic flight stabilizer—reining in outlier noise spikes before they cause visual turbulence.

---

## 🛠️ Complete Flight Instrument Catalog (Nodes)

| Instrument Name | ComfyUI Display Name | Category | Primary Function & Mathematical Core | Documentation |
| :--- | :--- | :--- | :--- | :--- |
| `GimbalCompass_Pro` | 🧭 Gimbal Compass Pro | `Flight Instruments` | Vector arithmetic, $\mu$-SLERP, and orthogonal projection steering. | [compass_pro.md](./wiki/nodes/compass_pro.md) |
| `GimbalManifold_Explorer` | 🗺️ Gimbal Manifold Explorer | `Flight Instruments` | 2D $\mu$-centered orthogonal latent topography grid synthesis. | [manifold_explorer.md](./wiki/nodes/manifold_explorer.md) |
| `GimbalCrossModalBridge` | 🌉 Gimbal Cross-Modal Bridge | `Conditioning` | Calibrated keyword-to-subspace signature projection. | [crossmodal_bridge.md](./wiki/nodes/crossmodal_bridge.md) |
| `GimbalCircularOrbit` | 🔄 Gimbal Circular Orbit | `Trajectory` | Constant-radius closed-loop geodesic orbits ($z(\theta) = \mu + r(\cos\theta\mathbf{u} + \sin\theta\mathbf{v})$). | [circular_orbit.md](./wiki/nodes/circular_orbit.md) |
| `GimbalWaypointSpline` | 〰️ Gimbal Waypoint Spline | `Trajectory` | Spherical Catmull-Rom geodesic spline multi-stop flight path. | [waypoint_spline.md](./wiki/nodes/waypoint_spline.md) |
| `GimbalSemanticSlider` | 🎚️ Gimbal Semantic Slider | `Decomposition` | Real-time SVD/PCA batch covariance decomposition attribute isolation. | [semantic_slider.md](./wiki/nodes/semantic_slider.md) |
| `GimbalGPS_Anchor` | 📍 Gimbal GPS Anchor (Save) | `Navigation` | Extracts single latents, computes hashes, and saves JSON waypoints. | [gps_anchor.md](./wiki/nodes/gps_anchor.md) |
| `GimbalGPS_Load` | 📂 Gimbal GPS Load | `Navigation` | Recalls saved cryptographic waypoint tensors and provenance metadata. | [gps_load.md](./wiki/nodes/gps_load.md) |
| `GimbalChannelSplit` | 🔀 Gimbal Channel Split | `Subspace` | Decouples 4-ch (SDXL) or 16-ch (FLUX) into frequency sub-bands. | [channel_split.md](./wiki/nodes/channel_split.md) |
| `GimbalChannelMerge` | 🔁 Gimbal Channel Merge | `Subspace` | Lossless concatenation and recomposition of split latent bands. | [channel_merge.md](./wiki/nodes/channel_merge.md) |
| `GimbalChannelScale` | ⚖️ Gimbal Channel Scale | `Subspace` | Independent per-channel frequency gain and amplitude control. | [channel_scale.md](./wiki/nodes/channel_scale.md) |
| `GimbalLatentStabilizer` | 🛡️ Gimbal Latent Stabilizer | `Stabilizer` | Full LAMNr pipeline: Woodbury denoise, scale cap, and $\psi$ shrinkage. | [latent_stabilizer.md](./wiki/nodes/latent_stabilizer.md) |
| `GimbalTruncation` | 📉 Gimbal Truncation | `Quality` | Surgical variance shrinkage toward centroid ($z' = \mu + \psi(z-\mu)$). | [truncation.md](./wiki/nodes/truncation.md) |
| `GimbalLatentMath` | 🔢 Gimbal Latent Math | `Primitives` | Full dispatcher exposing all 13 LAMNr primitives in a single node. | [latent_math.md](./wiki/nodes/latent_math.md) |
| `GimbalDiagnostics` | 📊 Gimbal Diagnostics | `Telemetry` | Real-time tensor readout: min, max, mean, std, L2 norm, channel variance. | [diagnostics.md](./wiki/nodes/diagnostics.md) |
| `GimbalLatentTelemetry` | 📡 Gimbal Latent Telemetry | `Telemetry` | Research-grade OOD metrics: Exact Log-Likelihood, Mahalanobis, TC. | [latent_telemetry.md](./wiki/nodes/latent_telemetry.md) |
| `GimbalVectorAnalogy` | ➕ Gimbal Vector Analogy | `Arithmetic` | Concept arithmetic ($A - B + C$) with orthogonal projection safeguards. | [vector_analogy.md](./wiki/nodes/vector_analogy.md) |
| `GimbalLikenessIsolator`| 🎭 Gimbal Likeness Isolator | `Conditioning` | Differential LoRA probe decoupling identity tokens from scene style. | [likeness_isolator.md](./wiki/nodes/likeness_isolator.md) |
| `GimbalGridStitch` | 🪡 Gimbal Grid Stitch | `Utility` | Stitches multi-sample batches into composite image contact sheets. | [grid_stitch.md](./wiki/nodes/grid_stitch.md) |

---

## 📋 Canonical Workflow Guides

| # | Workflow Name | Primary Instruments | Key Use Case | Documentation |
| :---: | :--- | :--- | :--- | :--- |
| **01** | **Concept Blender** | Compass Pro (Slerp) | Geodesic blend between two prompts at Step 0. | [wf_01_concept_blender.md](./wiki/workflows/wf_01_concept_blender.md) |
| **02** | **Text-Steered Lighting** | Cross-Modal + Compass (Ortho) | Project natural language lighting onto locked geometry. | [wf_02_text_steered.md](./wiki/workflows/wf_02_text_steered.md) |
| **03** | **Manifold Grid** | Manifold Explorer | 2D topological surface variation grid. | [wf_03_manifold_grid.md](./wiki/workflows/wf_03_manifold_grid.md) |
| **04** | **Brand-Locked Lighting** | GPS Anchor + Compass (Ortho) | Capture brand lighting grammar and project across products. | [wf_04_brand_locked.md](./wiki/workflows/wf_04_brand_locked.md) |
| **05** | **Semantic Slider** | Semantic Slider (PCA/SVD) | Real-time covariance attribute modulation. | [wf_05_semantic_slider.md](./wiki/workflows/wf_05_semantic_slider.md) |
| **06** | **Architecture Material Matrix** | Cross-Modal + Manifold | 2D material $\times$ elevation architectural sweep. | [wf_06_arch_material_matrix.md](./wiki/workflows/wf_06_arch_material_matrix.md) |
| **07** | **Likeness Isolator** | Likeness Isolator + Compass | Differential LoRA identity token isolation. | [wf_07_likeness_isolator.md](./wiki/workflows/wf_07_likeness_isolator.md) |
| **08** | **Harmonic Orbiter** | Circular Orbit (Geodesic) | Constant-radius closed-loop 360° architectural/product tour. | [wf_08_harmonic_orbiter.md](./wiki/workflows/wf_08_harmonic_orbiter.md) |
| **09** | **Subspace Material Matrix** | Channel Split + Merge + Stabilizer | Subspace frequency band decoupling (100% silhouette lock). | [wf_09_subspace_material.md](./wiki/workflows/wf_09_subspace_material.md) |
| **10** | **Pro Multi-Instrument Pipeline**| Full Chained Flight Deck | End-to-end multi-instrument production validation. | [wf_10_pro_pipeline.md](./wiki/workflows/wf_10_pro_pipeline.md) |

---

## 🔬 Mathematical & Theoretical Deep Dives

For power users, machine learning engineers, and researchers seeking the mathematical formulations underlying Gimbal:

- **[What is Latent Space?](./wiki/concepts/latent_space.md)** — Intuitive guide to high-dimensional manifold topology and VAE compression.
- **[SLERP vs. LERP: Why the Arc Matters](./wiki/concepts/slerp_vs_lerp.md)** — Mathematical derivation of midpoint variance collapse and empirical $\mu$-centered SLERP (Equation E4).
- **[The Gaussian Annulus Theorem](./wiki/concepts/gaussian_annulus.md)** — Proof of why high-dimensional mass concentrates on a thin hypersphere shell in $\mathbb{R}^{65,536}$.
- **[Channel Architecture: 4ch (SDXL) vs. 16ch (FLUX.1)](./wiki/concepts/channel_architecture.md)** — Subspace frequency band decoupling and broad-spectrum cluster mapping.
- **[The LAMNr Framework (Equations E1–E13)](./wiki/concepts/lamnr_framework.md)** — Full technical dossier covering normalizing flows, Woodbury matrix MMSE inversion, Mahalanobis OOD metrics, and Total Correlation.
- **[Failure Case Studies & Remediations](./wiki/concepts/failure_case_studies.md)** — Audited post-mortems of Step-8 SLERP collisions, CFG variance frying, analogy double-exposures, and delta-zero collapse.

---

## 📖 Project History

- **[Creation Timeline & Development History](./wiki/history/project_history.md)** — The evolutionary story from the original pre-alpha "Wayfinder" prototype, through the "Latent Explorer" LAMNr research era and Run 4 failure remediations, to the certified "Gimbal" node suite.

---




</div>
