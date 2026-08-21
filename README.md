# 🧭 Gimbal — Navigate latent space with precision flight instruments, not lottery prompts.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/lockup-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="brand/lockup-light.svg">
    <img alt="Gimbal — Navigate latent space with precision flight instruments, not lottery prompts." src="brand/gimbal_lockup_dark.png" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/FormAndNoise"><img src="https://img.shields.io/badge/Form%20%26%20Noise-Atelier-0E8A8A?style=flat-square" alt="Form & Noise Atelier"></a>
  <a href="https://github.com/comfyanonymous/ComfyUI"><img src="https://img.shields.io/badge/ComfyUI-Custom%20Nodes-141414?style=flat-square&logo=comfyui" alt="ComfyUI Node Suite"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python 3.10+"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch 2.0+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-teal?style=flat-square" alt="License: MIT"></a>
</p>

---

**Gimbal** (formerly *Wayfinder* / *Latent Explorer*) is an artisan-grade latent space navigation suite for visual generative artists, creative technologists, and AI researchers built on **ComfyUI**.

Generative latent spaces are not lottery slots—they are high-dimensional topological manifolds. **Gimbal** treats latent space as a navigable, reproducible coordinate geography: steer along concept axes, map 2D manifold slices with spherical linear interpolation ($\mu$-centered SLERP), execute closed-loop orbits, spline through keypoints with constant geodesic velocity, and stabilize distributions using normalizing flow math.

---

## 🏛️ Studio Ecosystem & Atelier Brand Standards

**Gimbal** is designed and maintained by **[Form & Noise Atelier](https://github.com/FormAndNoise)** as part of the **Loose Endorsed Family** of precision creative engineering tools:

| Tool | Focus & Architecture | Stack / Target |
| :--- | :--- | :--- |
| **🧭 Gimbal** | Latent space flight instruments, vector navigation, and manifold mapping | ComfyUI / PyTorch / CUDA |
| **🃏 Cartouche** | Local-first workspace for managing deliverable card grids | Tauri 2 / React 19 / SQLite |
| **📐 Pantograph** | High-performance batch image vectorization engine (`vpipe`) | Rust / WebAssembly |
| **📖 Colophon** | Autonomous LLM-driven publishing and editorial book layout suite | ComfyUI / Typst / Python |
| **📑 Quire** | Structural PDF signature imposition for Coptic and codex bookbinding | Python / CLI |
| **🎞️ Dredge** | Resilient MP4/MOV stream recovery and Annex B/ADTS stream reconstruction | Rust / Native CLI |

---

## 🎨 Brand Identity & Design Tokens

Gimbal adheres to the strict **Form & Noise Atelier Brand Standards** (documented in full at [`brand/BRAND.md`](brand/BRAND.md)):

### 1. Invariant Hero Lockup & Job Line
All studio releases share the unified Loose Endorsed Family lockup formula:
```markdown
[Symbol] [Product Name] — [One-Sentence Job Line]
🧭 Gimbal — Navigate latent space with precision flight instruments, not lottery prompts.
```

### 2. Visual Mark & Geometry
- **Metaphor**: An aeronautical artificial horizon / attitude indicator dial consisting of a circular bezel enclosing horizontal datum bars, pitch ladder index lines, and **one solid center heading pip**.
- **Canvas Geometry**: $24 \times 24$ unit square canvas (bounded to $20 \times 20$ optical area) with $2\text{u}$ corner radii.
- **Stroke & Line Quality**: Exact $1.75\text{u}$ stroke width (`stroke-width="1.75"`), `fill="none"`, with `stroke-linecap="round"` and `stroke-linejoin="round"`.
- **The Family Fingerprint**: Exactly **one solid circular state/heading pip** ($r = 1.3\text{u}$ / $2.6\text{u}$ diameter, `fill="currentColor"`, `stroke="none"`) centered at the instrument origin `(12, 12)`.

<p align="center">
  <img src="brand/symbol.svg" alt="Gimbal Attitude Indicator Mark" width="120" height="120">
</p>

### 3. Atelier Color Law
Gimbal's palette is grounded in the contrast between deep instrument voids and crisp paper schematics, accented with high-visibility instrument teal and the signature Form & Noise house metal:

| Token Name | Light Ground (`#F6F1EA` Paper) | Dark Ground (`#0B0B0B` Void) | Purpose |
| :--- | :--- | :--- | :--- |
| **Instrument Accent** | `#0E8A8A` (Instrument Teal) | `#35B8B8` (Instrument Teal Dark) | Active vectors, telemetry highlights, heading markers |
| **House Metal** | `#D45500` (Safety Rust) | `#D45500` (Safety Rust) | Universal Form & Noise Atelier signature metal accent |
| **Ink Foreground** | `#141414` (Deep Charcoal) | `#F2EEE8` (Warm Off-White) | High-contrast typography, primary iconography, dial rims |
| **Ground / Canvas** | `#F6F1EA` (Paper Ground) | `#0B0B0B` (Void Ground) | Canvas backdrop, node body background, HUD surfaces |
| **Muted Hairline** | `#8A8680` | `#6B6763` | Subordinate pitch rungs, coordinate grids, border hairlines |

### 4. Typography Stack
- **Wordmarks & Display Headers**: **Space Grotesk** (Medium / SemiBold, tracking $-1\%$ to $-2\%$)
- **Interface & Body Text**: **Inter** (Regular / Medium)
- **Code, Telemetry & Math**: **IBM Plex Mono** / **JetBrains Mono** (Regular)

### 5. ComfyUI Frontend HUD Styling
The frontend extension in [`web/js/gimbal.js`](web/js/gimbal.js) themes all Gimbal nodes in ComfyUI with the **Atelier Void** (`#0B0B0B` / `#0E8A8A` / `#35B8B8`) aesthetic, providing real-time vector diagnostics, custom compass dials, and interactive batch telemetry.

---

## 🛠️ Flight Instrument Catalog

### 1. Flight Instruments & Trajectory Navigation
| Node | Class Name | Description |
| :--- | :--- | :--- |
| **🧭 Gimbal Compass Pro** | `GimbalCompass_Pro` | Vector arithmetic ($Target - Origin$) with **Standard**, **Normalized**, **Orthogonal Projection**, and **Blend Overlay** modes plus localized mask guidance. |
| **🗺️ Gimbal Manifold Explorer** | `GimbalManifold_Explorer` | 2D topological manifold slice generator across independent X/Y vectors with true $\mu$-centered SLERP following the high-probability Gaussian Annulus shell. |
| **🔄 Gimbal Circular Orbit** | `GimbalCircularOrbit` | Constant-radius closed-loop spherical orbits for seamless looping animations and harmonic latent tours. |
| **🛤️ Gimbal Waypoint Spline** | `GimbalWaypointSpline` | Continuous Catmull-Rom spherical spline flight paths through $N$ arbitrary latent keypoints with constant geodesic velocity. |
| **🎚️ Gimbal Semantic Slider** | `Gimbal_SemanticSlider` | Real-time PCA/SVD decomposition extracting orthogonal variance directions for independent attribute modulation. |
| **🌉 Gimbal Cross-Modal Bridge** | `Gimbal_CrossModalBridge` | Multimodal text-to-latent projection steering diffusion representations via keyword heuristics or CLIP pooled embeddings. |
| **🧬 Gimbal Likeness Isolator** | `LikenessVectorIsolator` | Dynamic LoRA probe isolating identity tokens from stylistic tokens in the CLIP text-encoder. |
| **🗺️ Gimbal Grid Stitch** | `LatentSpaceGridStitch` | Contact-sheet assembly for latent grids and manifold sweeps. |

### 2. Subspace & Channel Manipulation
| Node | Class Name | Description |
| :--- | :--- | :--- |
| **🔀 Gimbal Channel Split** | `GimbalChannelSplit` | Deconstructs latent tensors into independent frequency/structural channel bands (SDXL 4-ch, FLUX/SD3 16-ch). |
| **🔁 Gimbal Channel Merge** | `GimbalChannelMerge` | Reconstructs full multidimensional latent tensors from decoupled channel subspaces. |
| **🎛️ Gimbal Channel Band Scaler** | `GimbalChannelScale` | Independent per-channel and cluster gain control for fine-grained color, luminance, and texture balance. |
| **🎯 Gimbal Latent Truncation** | `GimbalTruncation` | Latent variance shrinkage toward the distribution centroid ($z' = \mu + \psi(z - \mu)$) to rein in noisy outlier artifacts. |
| **⚖️ Gimbal Vector Analogy** | `GimbalVectorAnalogy` | Classic $A - B + C$ concept analogy arithmetic with orthogonal projection and hypersphere norm preservation. |
| **📍 Gimbal GPS Anchor (Save)** | `GimbalGPS_Anchor` | Extracts waypoints, logs cryptographic hashes, computes statistical moments, and serializes coordinates to disk. |
| **📥 Gimbal GPS Load (Recall)** | `GimbalGPS_Load` | Restores saved waypoints across sessions and checkpoints with automatic statistical rescaling. |
| **📊 Gimbal Latent Diagnostics** | `GimbalDiagnostics` | Live telemetry HUD reporting min, max, mean, standard deviation, L2 norm, and channel variance. |

### 3. LAMNr & Disentanglement Research Math
| Node | Class Name | Description |
| :--- | :--- | :--- |
| **🛠️ Gimbal Latent Stabilizer** | `GimbalLatentStabilizer` | Full quality pipeline: bounded coupling scale, dequantization jitter, truncation, and Woodbury low-rank conditional-mean denoise. |
| **🔣 Gimbal Latent Math** | `GimbalLatentMath` | Generic dispatcher routing 13 core normalizing-flow and disentanglement equations (E1–E12). |
| **📟 Gimbal Latent Telemetry** | `GimbalLatentTelemetry` | Research-grade out-of-distribution (OOD) metrics: exact log-likelihood, Mahalanobis distance, Total Correlation, and geodesic angular distance. |

---

## 🧬 Architecture-Aware Latent Mechanics

Gimbal automatically recognizes the channel topology of active models:

- **4-Channel Latents (SD1.5, SD2.1, SDXL)**: Mapped to standard Luminance, Chroma, and High-Frequency Texture axes.
- **16-Channel Latents (FLUX.1, SD3.5)**: Implements **Broad-Spectrum Cluster Mapping**. 16-channel latents represent abstract multimodal features; Gimbal operates across semantic channel clusters to prevent color-space collapse and manifold tearing.

---

## 🚀 Installation

### Option 1: ComfyUI Manager (Recommended)
Search for `Gimbal` or `ComfyUI-Gimbal` in the ComfyUI Manager and click **Install**.

### Option 2: Git Clone
1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/FormAndNoise/gimbal-comfy.git ComfyUI-Gimbal
   ```
3. Install dependencies:
   ```bash
   pip install -r ComfyUI-Gimbal/requirements.txt
   ```
4. Restart ComfyUI.

---

## 🧪 Testing

Gimbal includes an audited test suite covering all mathematical primitives, SLERP geometry, GPS persistence, and node contracts:

```bash
pytest
```

---

## 📂 Repository Structure

```
gimbal-comfy/
├── __init__.py                # ComfyUI node registration & display names
├── AGENTS.md                  # Atelier agent protocol & specifications
├── pytest.ini                 # Test discovery configuration
├── requirements.txt           # Minimal PyTorch & ComfyUI dependencies
├── brand/                     # Form & Noise Atelier visual marks & lockups
│   ├── BRAND.md               # Atelier design standards & tokens
│   ├── symbol.svg             # 24x24u Attitude indicator dial symbol
│   ├── symbol-micro.svg       # Micro variant icon
│   ├── lockup-dark.svg        # Dark mode vector lockup
│   ├── lockup-light.svg       # Light mode vector lockup
│   ├── avatar-512.svg         # 512x512 avatar SVG
│   ├── gimbal_avatar_512.png  # Rasterized 512px avatar
│   ├── gimbal_lockup_dark.png # High-res hero lockup (dark)
│   ├── gimbal_lockup_light.png# High-res hero lockup (light)
│   └── gimbal_social_preview.png # Social share card
├── docs/                      # Research documentation & technical specs
│   ├── Gimbal_Starter_Guide.md# Quick start & flight guide
│   └── research/              # LAMNr & Disentangled representation papers
├── nodes/                     # Core PyTorch flight instruments & math
│   ├── gimbal_compass.py
│   ├── gimbal_manifold_explorer.py
│   ├── gimbal_circular_orbit.py
│   ├── gimbal_waypoint_spline.py
│   ├── gimbal_channel_matrix.py
│   ├── gimbal_truncation.py
│   ├── gimbal_vector_analogy.py
│   ├── gimbal_slerp.py
│   ├── gimbal_latent_math.py
│   ├── gimbal_latent_math_node.py
│   ├── gimbal_latent_stabilizer.py
│   ├── gimbal_latent_telemetry.py
│   ├── gimbal_gps_anchor.py
│   ├── gimbal_gps_load.py
│   ├── gimbal_crossmodal_bridge.py
│   ├── gimbal_semanticslider.py
│   ├── gimbal_likeness_isolator.py
│   └── gimbal_grid_stitch.py
├── web/                       # ComfyUI Frontend extension
│   └── js/gimbal.js           # Atelier Dark HUD styling & widgets
└── extras/
    ├── example_workflows/     # Ready-to-use ComfyUI workflow JSONs
    ├── tests/                 # Comprehensive pytest test suite
    └── workflows/             # Exploration & testing lab graphs
```

---

## 📜 License & Studio Attribution

Released under the **MIT License**. Created with precision by **[Form & Noise Atelier](https://github.com/FormAndNoise)**.
