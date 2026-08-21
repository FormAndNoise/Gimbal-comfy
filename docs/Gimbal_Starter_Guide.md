# Welcome to Gimbal: Latent Space Flight Instruments for ComfyUI 🧭

> *"Navigate latent space with precision flight instruments, not lottery prompts."*

Generative AI isn't just magic—it's math. The "Latent Space" where diffusion models operate is a high-dimensional topological manifold. 

The **Gimbal Node Suite** gives you the flight instruments needed to deliberately navigate that landscape. Instead of rolling the dice with random seeds and text prompts, Gimbal lets you extract coordinates, steer using vector math, and chart new territories using interpolation grids.

---

## 🚀 The Core Nodes

### 1. 🧭 `GimbalCompass_Pro` (The Steering Wheel)
This node performs high-precision vector arithmetic on your latent images.

*   **What it does:** Allows you to take a `base_latent`, and push it towards a `target_latent` relative to an `origin_latent`.
*   **How to wire it:** 
    1. Hook up an `Empty Latent Image` to `base_latent`.
    2. Generate a latent from prompt A ("a forest") and pipe it into `base_latent`.
    3. Generate a latent from prompt B ("snow") and pipe it into `target_latent`.
    4. Adjust the `strength` slider. At 0.5, you will get a perfect mathematical hybrid of the two concepts.

### 2. 🗺️ `GimbalManifold_Explorer` (The Map Maker)
Generates a 2D batch grid of latent variations with true $\mu$-centered SLERP.

*   **What it does:** Taking a center point, it creates a grid of latent variations expanding outward along an X-axis and a Y-axis.
*   **How to wire it:**
    1. Hook your starting latent to `center_latent`.
    2. Hook up directional vectors (e.g., from the Cross-Modal bridge) to `x_vector` and `y_vector`.
    3. Set your grid size (e.g., 3x3). It will output a batch of 9 latents interpolating across those concepts.
    4. **CRITICAL:** Pass this batch into a *second* `KSampler` with a lower denoise (e.g. 0.45 - 0.6) before `VAEDecode` to turn the raw math into sharp, realistic variations.

### 3. 🌉 `GimbalCrossModalBridge` (The Translator)
Converts text instructions directly into latent directional vectors.

*   **What it does:** Maps words like "bright" or "vivid" to heuristic latent channel changes or projects CLIP text embeddings into latent subspaces.
*   **How to wire it:** Type "make it brighter" and plug its output into the `target_latent` of a Compass, or the `x_vector` of a Manifold Explorer.

### 4. 🎚️ `GimbalSemanticSlider` (The Fine-Tuner)
PCA-based feature isolation.

*   **What it does:** Analyzes a batch of latents, finds the principal components (the fundamental visual differences like lighting, age, or color), and gives you a slider to control that specific feature without altering the rest of the image.
*   **How to wire it:** Pass a batch of varied latents into `latent_batch`, and the original latent into `base_latent`. Use the `pc_index` to select which "feature" to slide, and `slider_value` to push it.

### 5. 📍 `GimbalGPS_Anchor` & `GimbalGPS_Load` (The Save Point & Recall)
Saves specific latent coordinates to disk so you can return to them later.
*   **What it does:** Found the perfect image in a batch? The GPS anchor extracts it, saves its exact mathematical state, and lets you reload it in future sessions.

---

*Happy exploring! Latent space is vast, but you are no longer lost.*