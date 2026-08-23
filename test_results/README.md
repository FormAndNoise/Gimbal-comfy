# Gimbal Flight Instruments — Audited Remediation Test Suite (Run 5)

All generation benchmarks comparing mathematical latent space steering against unsteered baselines (SDXL Base 1.0 @ 1024×1024) are centralized below, highlighting the audited remediation of Run 4 spatial/latent failures.

## Summary of Run 5 Remediation Benchmarks

### 1. Concept Blender (Step 0 Noise-Space SLERP vs Step 8 Spatial Re-skinning)
- **The Issue in Run 4**: Slerping spatial latents at Step 8 forced the UNet to re-skin tree branches into vertical cliff spires and moss mounds into rocks, producing heavy black comic-book outlines and posterization (`01_slerp_35pct_00001_.png`).
- **The Remediation in Run 5**: Performing SLERP on initial Gaussian noise at Step 0 ($t=0$) before spatial coordinates crystallize allows the UNet to synthesize a coherent, photorealistic alpine landscape:
  - `01_ctrl_A_forest_v5.png` — Dense Forest Baseline ($t=0.0$).
  - `01_slerp_noise_35pct_v5.png` — Alpine Valley with dense evergreen forest ($t=0.35$).
  - `01_slerp_noise_50pct_v5.png` — Balanced alpine valley with snow-covered pine slopes & towering glacier peak ($t=0.50$).
  - `01_slerp_noise_65pct_v5.png` — Mountain dominant alpine tundra ($t=0.65$).
  - `01_ctrl_B_mountain_v5.png` — Snowy Mountain Peak Baseline ($t=1.0$).

### 2. Text-Steered Diffusion (Stabilized & De-Posterized)
- **The Issue in Run 4**: High CFG (6.5) and high denoise (0.72–0.80) caused severe latent variance frying, creating embossed hair, heavy black contour linework, and posterized skin (`02_steered_denoise_072_00001_.png`).
- **The Remediation in Run 5**: Inserting `GimbalLatentStabilizer` ($\psi = 0.88$) and dropping second-stage CFG to 3.8 at denoise 0.42 restores natural photorealistic skin and hair while delivering rich amber rim lighting:
  - `02_ctrl_baseline_portrait_v5.png` — Daylight portrait baseline (Seed 789).
  - `02_steered_denoise_042_v5.png` — Clean, soft, photorealistic skin with warm directional rim lighting.
  - `02_steered_denoise_055_v5.png` — Stronger amber glow without black contour linework.

### 3. Brand-Locked Lighting (Hallucination & Watermark Removal)
- **The Issue in Run 4**: High denoise (0.72) caused the UNet to hallucinate floating black wireframe geometry on the left and a fake watermark logo in the upper corner (`04_brand_denoise_072_00001_.png`).
- **The Remediation in Run 5**: Orthogonal steering stabilized with `GimbalLatentStabilizer` and calibrated at denoise 0.50 / CFG 3.8 eliminates floating wireframe hallucinations and watermarks while preserving crisp leather and gold speculars:
  - `04_source_anchor_watch_v5.png` — Gold watch on dark velvet lighting anchor.
  - `04_ctrl_baseline_handbag_v5.png` — Flat daylight handbag control.
  - `04_brand_denoise_050_v5.png` — Flawless commercial product render with zero artifacts.
  - `04_brand_denoise_060_v5.png` — Deep shadow falloff with clean hardware reflections.

### 4. Concept Vector Analogy (Spatial Alignment Diagnostics)
- **Core Finding**: Latent tensors in diffusion models have a 1-to-1 spatial coordinate grid. Direct spatial elementwise subtraction $(A - B)$ from Person A stamps their eyes and jawline onto Person C, creating a phantom double-exposure (`03_analogy_smile_plus_1_00001_.png`).
- **Architectural Rule**: Facial and anatomical attributes should be steered via **Cross-Modal Bridge / Prompt Conditioning** (`GimbalCrossModalBridge`) or **Mask-Guided Arithmetic** rather than global elementwise spatial addition.

---

## Interactive Gallery
Open [`test_results/GALLERY.html`](GALLERY.html) in any browser to inspect the interactive dark HUD Before-vs-After remediation gallery.
