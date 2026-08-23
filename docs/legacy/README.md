# 🧭 Wayfinder: Latent Space Navigation Suite

**Wayfinder** is a high-precision ComfyUI node suite designed for artists, developers, and industrial creatives to treat **Latent Space** as a navigable, reproducible geography. By moving beyond random seeds, Wayfinder provides a "flight instrument" approach to generative AI—allowing users to steer, map, and anchor their creative journey through high-dimensional model space.

-----

## 📖 Table of Contents

  * [The Philosophy](#-the-philosophy)
  * [Node Documentation](#-node-documentation)
  * [Architecture-Awareness (SDXL vs. FLUX)](#-architecture-awareness)
  * [Industry Use Cases](#-industry-use-cases)
  * [Technical Specs & Contributions](#-technical-specs--contributions)
  * [Installation](#-installation)
  * [Example Usage](#-example-usage)

-----

## 🧩 The Philosophy

Most generative workflows treat the "latent" as a black box. **Wayfinder** changes this by introducing three core metaphors:

1.  **Search (Compass)**: Define a direction (e.g., "More Industrial," "Less Abstract") and move toward it.
2.  **Map (Explorer)**: Generate a 2D topographic grid to see every "neighboring" variation of an idea.
3.  **Pivot (GPS)**: Bookmark a specific mathematical coordinate and make it your new "Home".

-----

## 🛠 Node Documentation

### 🧭 Wayfinder Compass Pro
The steering engine of the suite. It performs vector arithmetic between two points ($Target - Origin$) and applies that difference to your $Base$ latent.
  * **Modes**:
      * **Standard**: Linear vector addition.
      * **Normalized**: Unit-normalizes the direction to prevent over-driving the signal.
      * **Orthogonal**: Projects the base onto the target vector for maximum stylistic alignment.
  * **Pro Feature**: Includes an optional **Mask** input to localize stylistic shifts to specific parts of an image.

### 🗺️ Wayfinder Manifold Explorer
Generates a batch of latents representing a 2D grid.
  * **Function**: Set an X-axis vector and a Y-axis vector to see how two concepts interact.
  * **Interpolation**: Supports both **Linear** and **Slerp** (Spherical Linear Interpolation) to maintain visual fidelity across high-dimensional curves.

### 📍 Wayfinder GPS Anchor
The "Logbook" of your journey.
  * **Function**: Extracts a single image from a batch, logs its precise coordinates, and saves them to a `.json` waypoint.
  * **Persistence**: Allows you to reload a "Waypoint" later to resume a session or share a specific "latent location" with a team.

### 🌉 Wayfinder Cross-Modal Bridge
Translates natural language or LLM JSON outputs into latent vectors.
  * **Intelligence**: Features "Keyword Heuristics" that map words like *'Neon'*, *'Cinematic'*, or *'Moody'* to specific latent channel offsets.

### 🎚️ Wayfinder Semantic Slider
A precision laboratory tool using **Principal Component Analysis (PCA)**.
  * **Logic**: Analyzes a batch of images and isolates the "Principal Components" (the directions of most significant change).
  * **Use**: Slide individual components to adjust specific features (e.g., "Lighting" or "Geometry") without affecting other traits.

### 🧬 Likeness Vector Isolator
A dynamic probe for LoRA influence to isolate and modulate specific identity vectors.
  * **Function**: Loads a LoRA and exposes internal parameters so they can be driven by input vectors.
  * **Control**: Allows independent modulation of `strength` (overall patch), `alpha` (network override), and `likeness_mask` (isolation of identity tokens from style tokens in the CLIP text-encoder).

-----

## 🧬 Architecture-Awareness

Wayfinder is built for the 2026 generative landscape. It automatically detects and scales its math based on the model architecture:

  * **4-Channel Models (SD1.5, SDXL)**: Standard mapping to Luminance, Chroma, and Texture axes.
  * **16-Channel Models (FLUX.1, SD3.5)**: Implements **Broad-Spectrum Cluster Mapping**. Because 16-channel latents are highly abstracted, Wayfinder keywords influence "clusters" of channels (e.g., channels 4–9 for saturation) to prevent color-space collapse.

-----

## 💼 Industry Use Cases

### 🎨 Concept Art & Illustration
**Goal**: Iterate on a character's "Mood" while keeping the geometry identical.
**Workflow**: Use the **Compass** to add a "Melancholic" vector to a character sketch. Use the **Explorer** to find the perfect 0.35 strength "sweet spot."

### 🏠 Architecture & Construction
**Goal**: Rapidly explore material finishes for a single floor plan.
**Workflow**: Input the floor plan as the $Base$. Use the **Manifold Explorer** with "Wood/Warm" on the X-axis and "Concrete/Cold" on the Y-axis. Generate 25 structural variations in one click.

### 📊 Marketing & Branding
**Goal**: Ensure 1,000 product variations share the exact same "Brand Lighting."
**Workflow**: Use the **GPS Anchor** to save the "Brand Waypoint." Use the **Compass** in every workflow to "Force Project" all generations onto that specific stylistic coordinate.

-----

## ⚙️ Technical Specs & Contributions

Wayfinder is designed for high-performance production environments:
  * **No gradient tracking**: All operations use `torch.no_grad()` and explicit memory de-allocation to prevent VRAM leaks.
  * **Device/dtype safety**: Automatic alignment of tensors (float16, float32, bfloat16, CPU/CUDA).
  * **Audit-Hardened**: Code has been audited for numerical stability, especially for parallel-vector SLERP cases (preventing NaN errors).
  * **Testing**: Includes a comprehensive `pytest` suite for GPS coordinate persistence and I/O safety.

**Call for Contributors**:
We welcome code improvements and real-world bug reports. If you are interested in expanding the **Cross-Modal Bridge** keyword library or optimizing the **PCA** logic for larger batches, please submit a Pull Request or issue.

-----

## 🚀 Installation

1.  Navigate to your `ComfyUI/custom_nodes/` folder: `cd ComfyUI/custom_nodes/`
2.  Clone this repository: `git clone https://github.com/your-username/ComfyUI-Wayfinder.git`
3.  Install requirements: `pip install -r requirements.txt`
4.  Restart ComfyUI to load the new nodes.

-----

## Example Usage

### Basic Latent Navigation
![Compass Steering](../../assets/brand/gimbal_02_cinematic_steering_showcase.png)
```python
# Move from origin to target with strength control
base → WayfinderCompass_Pro → modified_latent
  ├─ target_latent
  ├─ origin_latent
  └─ strength: 0.5
```

### 2D Latent Grid Exploration
![Manifold Grid](../../assets/brand/gimbal_08_harmonic_orbiter_showcase.png)
```python
# Create 3x3 grid of variations
center → WayfinderManifold_Explorer → latent_batch[9]
  ├─ x_vector (horizontal variations)
  ├─ y_vector (vertical variations)
  └─ grid_size: 3x3
```

### Text-Guided Modifications
![Cross-Modal Style Transfer](../../assets/brand/gimbal_04_semantic_slider_showcase.png)
```python
# Apply text-based changes
latent → Wayfinder_CrossModalBridge → modified_latent
  └─ instruction: "make it brighter and more vivid"
```

### Identity and Likeness Isolation
![Likeness Control](../../assets/brand/gimbal_09_subspace_material_showcase.png)
```python
# Probe LoRA parameters and isolate the identity tokens
model, clip → LikenessVectorIsolator → patched_model, patched_clip
  ├─ lora_name: "character_lora.safetensors"
  ├─ strength: 1.0
  ├─ alpha: 1.0 (Driven by external inputs like Manifold Explorer)
  └─ likeness_mask: 0.8
```

-----
*Developed for artists who prefer steering to gambling.*
