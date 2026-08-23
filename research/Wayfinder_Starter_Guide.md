# Welcome to Wayfinder: Latent Geography for ComfyUI 🧭

Generative AI isn't just magic—it's math. The "Latent Space" where Stable Diffusion operates is an unimaginably vast, multi-dimensional geographic landscape. 

The **Wayfinder Node Suite** gives you the flight instruments needed to deliberately navigate that landscape. Instead of rolling the dice with random seeds and text prompts, Wayfinder lets you extract coordinates, steer using vector math, and chart new territories using interpolation grids.

![Hero Image: A beautiful 3x3 latent space interpolation grid or an abstract representation of navigating latent space](images/hero_banner.jpg)

---

## 🚀 The Core Nodes

### 1. 🧭 `WayfinderCompass_Pro` (The Steering Wheel)
This node performs high-precision vector arithmetic on your latent images.

![Screenshot of the Compass node wired up with a before/after visual showing Image A + Image B = Hybrid Image C](images/compass_example.jpg)

*   **What it does:** Allows you to take a `base_latent`, and push it towards a `target_latent`.
*   **How to wire it:** 
    1. Hook up an `Empty Latent Image` to `base_latent`.
    2. Generate a latent from prompt A ("a forest") and pipe it into `base_latent`.
    3. Generate a latent from prompt B ("snow") and pipe it into `target_latent`.
    4. Adjust the `strength` slider. At 0.5, you will get a perfect mathematical hybrid of the two concepts.

### 2. 🗺️ `WayfinderManifold_Explorer` (The Map Maker)
Generates a 2D batch grid of latent variations.

![A 3x3 grid of images output by the Manifold Explorer, showing smooth transitions from one concept on the X axis to another on the Y axis](images/manifold_grid.jpg)

*   **What it does:** Taking a center point, it creates a grid of latent variations expanding outward along an X-axis and a Y-axis.
*   **How to wire it:**
    1. Hook your starting latent to `center_latent`.
    2. Hook up directional vectors (e.g., from the Cross-Modal bridge) to the `x_vector` and `y_vector`.
    3. Set your grid size (e.g., 3x3). It will output a batch of 9 latents interpolating across those concepts.
    4. **CRITICAL:** Pass this batch into a *second* `KSampler` with a lower denoise (e.g. 0.45 - 0.6) before `VAEDecode` to turn the raw math into sharp, realistic variations.

### 3. 🌉 `Wayfinder_CrossModalBridge` (The Translator)
Converts text instructions directly into latent directional vectors.

![Screenshot of the CrossModalBridge node translating the text 'make it brighter' into a vector output](images/crossmodal_bridge.jpg)

*   **What it does:** Rather than running a full CLIP encode, it maps words like "bright" or "vivid" to heuristic latent channel changes.
*   **How to wire it:** Type "make it brighter" and plug its output into the `target_latent` of a Compass, or the `x_vector` of a Manifold Explorer.

### 4. 🎚️ `Wayfinder_SemanticSlider` (The Fine-Tuner)
PCA-based feature isolation.

![A strip of 3 images showing the same base image, but with the Semantic Slider at -1, 0, and +1 isolating a specific trait like age or lighting without changing the core image](images/semantic_slider_example.jpg)

*   **What it does:** Analyzes a batch of latents, finds the principal components (the fundamental visual differences like lighting, age, or color), and gives you a slider to control that specific feature without altering the rest of the image.
*   **How to wire it:** Pass a batch of varied latents into `latent_batch`, and the original latent into `base_latent`. Use the `pc_index` to select which "feature" to slide, and `slider_value` to push it.

### 5. 📍 `WayfinderGPS_Anchor` (The Save Point)
Saves specific latent coordinates to disk so you can return to them later.
*   **What it does:** Found the perfect image in a batch of 9? The GPS anchor extracts it, saves its exact mathematical state, and lets you reload it in future sessions.

---

## 🛠️ Getting Started with the Workflows

We have provided two starter workflows to help you understand the wiring:

### 1. `Starter_Compass_Steering.json`
*   **Concept:** This workflow demonstrates how to take an empty latent space and mathematically steer it halfway between two concepts before running it through the KSampler.
*   **To use:** Drag and drop the JSON into ComfyUI. Click "Queue Prompt". You will see it sample a perfect hybrid latent.

![Full ComfyUI Canvas screenshot of Starter_Compass_Steering.json in action](images/workflow_compass.jpg)

### 2. `Starter_Manifold_Grid.json`
*   **Concept:** This workflow uses the Manifold Explorer to generate a 3x3 grid (a batch of 9 images). The X-axis pushes the latent toward "red", and the Y-axis pushes it toward "futuristic".
*   **To use:** Drag and drop the JSON into ComfyUI. Click "Queue Prompt". It will generate a batch of 9 variations.

![Full ComfyUI Canvas screenshot of Starter_Manifold_Grid.json in action](images/workflow_manifold.jpg)

---
*Happy exploring! Latent space is vast, but you are no longer lost.* Latent space is vast, but you are no longer lost.*