<!-- Gimbal Node Suite — Form & Noise Atelier -->
<!-- Navigate latent space with precision flight instruments, not lottery prompts. -->

# 🔄 Workflow 08: Harmonic Orbiter

![Harmonic Orbiter Showcase](../../assets/brand/gimbal_08_harmonic_orbiter_showcase.png)

> **VibeCheck** 🟢 Stabilized &nbsp;|&nbsp; Category: `Gimbal/Trajectory` &nbsp;|&nbsp; Primary Node: **GimbalCircularOrbit**

---

## Overview

The Harmonic Orbiter generates a **closed-loop 360° orbit** in latent space around a single anchor concept. It produces a batch of frames — each at a fixed radius from the center latent — that, when played back sequentially, create a seamlessly looping animation tour of stylistic variations on a single subject. Think of it as rotating a camera around the concept itself rather than around a 3D object: every frame is a distinct but related interpretation, and the final frame connects back to the first.

The math is clean: the orbit traces a perfect circle in the two-dimensional subspace spanned by two orthonormal basis vectors `u` and `v`, computed via Gram-Schmidt orthonormalization. Every point on that circle lives at the same L2 radius from the origin as the center latent — squarely on the Typical Set shell.

---

## 🖼️ What You'll Create

- A batch of **N sequential latent frames** (default: 36) forming a complete 360° orbit
- A **seamlessly looping animation** when rendered to PNG and assembled into a GIF or video
- Stylistic variation within a controlled perceptual neighborhood — consistent subject, varying atmosphere

---

## 🗺️ Node Chain Diagram

```
[Checkpoint Loader] ───────────────────── MODEL, CLIP, VAE
[CLIP Text Encode: 'architectural exterior'] → CONDITIONING

[KSampler]
  ├── seed    = 42
  ├── denoise = 1.0      ← full synthesis — this is the orbit center
  └── cfg     = 7.0
  ↓ CENTER_LATENT

[🔄 GimbalCircularOrbit]
  ├── center_latent           = CENTER_LATENT
  ├── steps                   = 36        ← 36 frames = 10°/step = 1 full revolution
  ├── radius                  = 0.96      ← near Typical Set shell
  ├── orbit_mode              = Orthogonal_Basis
  ├── preserve_hypersphere_norm = True
  └── seed                    = 0
  ↓ latent_batch (36 frames stacked)

[KSampler]  (denoise=0.45, CFG=4.5, steps=20)
  ↓ image_batch
[VAE Decode] → [Save Image] → 36 PNG frames
```

---

## ⚙️ Settings That Work

Verified with FLUX.1-dev and SDXL 1.0 Base.

| Parameter | Optimal | Range | Notes |
|---|---|---|---|
| `steps` | `36` | 12 – 72 | 36 = smooth 360° (10°/frame). Use 72 for ultra-smooth 5°/frame. |
| `radius` | `0.96` | 0.5 – 1.5 | Near 1.0 = on the Typical Set shell. Values >1.5 may produce artifacts at orbit poles. |
| `orbit_mode` | `Orthogonal_Basis` | Orthogonal_Basis, Phase_Modulated, Harmonic_Torus | Most stable and diverse; recommended for new users. |
| `preserve_hypersphere_norm` | `True` | True / False | **Always True for animation.** Prevents density drift at frame transitions. |
| `seed` | `0` | any | Controls the random orientation of the orbital plane. Same seed = same orbit angle. |
| KSampler `denoise` | `0.45` | 0.35 – 0.60 | Low — preserves the geometric structure from the orbit math. |
| KSampler `cfg` | `4.5` | 3.5 – 6.0 | Moderate — frame-to-frame consistency degrades above 7.0 at this denoise. |

---

## 🔧 Step-by-Step Wiring Instructions

1. **Load a checkpoint.** Connect `MODEL`, `CLIP`, and `VAE` to downstream nodes.
2. **Add a CLIP Text Encode node.** Write your subject prompt (e.g. `"architectural exterior, brutalist concrete, golden hour"`). This defines the center concept.
3. **Add KSampler (Center).** Wire `MODEL` and your conditioning. Set `seed=42`, `denoise=1.0`. This fully synthesizes the **center latent** — the anchor point of the orbit.
4. **Add GimbalCircularOrbit.**
   - `center_latent` → output of the center KSampler
   - `steps` → `36`
   - `radius` → `0.96`
   - `orbit_mode` → `Orthogonal_Basis`
   - `preserve_hypersphere_norm` → `True`
   - `seed` → `0`
   - Output: `latent_batch` (a stacked batch of 36 latents)
5. **Add KSampler (Orbit).** Wire `MODEL` and conditioning. Set `latent_image` to `latent_batch`. Set `denoise=0.45`, `cfg=4.5`, `steps=20`. This KSampler processes all 36 frames in one batch pass.
6. **Add VAE Decode.** Wire `VAE` from the checkpoint and `SAMPLES` from the orbit KSampler. Output is a batch of 36 decoded images.
7. **Add Save Image.** All 36 frames save with sequential filenames (e.g. `00001_` through `00036_`).
8. **Assemble animation.** Use ffmpeg, Photoshop Timeline, or DaVinci Resolve to compile the PNGs into a looping GIF or MP4.

> **Tip:** To preview the orbit before committing 36 frames, set `steps=8` for a quick 8-frame test pass. If the orbit looks good, raise to 36.

---

## 🖼️ Gallery

### Architecture Orbit — Base & Frames

| Base Building | Frame 00 | Frame 01 |
|---|---|---|
| ![Base](../../assets/test_runs/architectural_showcases/08_arch_base_hero.png) | ![Frame 00](../../assets/test_runs/architectural_showcases/08_arch_orbit_frame_00.png) | ![Frame 01](../../assets/test_runs/architectural_showcases/08_arch_orbit_frame_01.png) |

| Frame 02 | Frame 03 |
|---|---|
| ![Frame 02](../../assets/test_runs/architectural_showcases/08_arch_orbit_frame_02.png) | ![Frame 03](../../assets/test_runs/architectural_showcases/08_arch_orbit_frame_03.png) |

### Desert Orbit Tour

| Frame 00 | Frame 01 |
|---|---|
| ![Desert 00](../../assets/test_runs/fresh_run6_exploration/08_desert_orbit_tour_frame_00.png) | ![Desert 01](../../assets/test_runs/fresh_run6_exploration/08_desert_orbit_tour_frame_01.png) |

---

## 🛠️ Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Frames look too similar — orbit feels "stuck" | Radius too small | Raise `radius` to `1.2`–`1.5` for more variation |
| Flickering between frames (non-smooth) | `preserve_hypersphere_norm = False` | Enable `preserve_hypersphere_norm` — this is the primary cause of density jumps |
| Subject identity lost by Frame 10+ | `denoise` too high on orbit KSampler | Lower orbit KSampler `denoise` to `0.35` |
| Last frame doesn't match first frame (no seamless loop) | `steps` not set to a full-revolution divisor | Use `steps=36` (10°/step) or `steps=72` (5°/step) — both close the loop exactly |
| Extreme artifacts at specific frames | `radius > 1.5` on models with tight priors | Reduce `radius` to `0.96`–`1.2` |
| Orbit plane is boring / all frames look like variations of the same angle | Low-entropy seed | Change `seed` — different seeds pick different `u, v` basis pairs = different orbital perspectives |

---

## 🔬 Power User Notes

### The Orbit Equation

Every frame position `z(θ)` is computed as:

```
z(θ) = center + r · (cos(θ) · u + sin(θ) · v)
```

Where:
- `center` — the anchor latent (your fully-synthesized center KSampler output)
- `r` — the orbit radius (default `0.96`)
- `θ` — the current angle, stepping uniformly from `0` to `2π` across `steps` frames
- `u`, `v` — two orthonormal basis vectors in latent space, computed via **Gram-Schmidt orthonormalization**

The Gram-Schmidt process guarantees `u ⊥ v` in R^D (D = latent dimension, typically 4×H/8×W/8 for SDXL). The orbit is therefore a **perfect circle** in the 2D subspace spanned by `{u, v}` — not an approximation.

### `preserve_hypersphere_norm = True`

When enabled, each orbit frame is scaled so its L2 norm equals the center latent's L2 norm. This projects all frames onto the same **hyperspherical shell** as the center, ensuring consistent probability density across the entire animation. Without this, frames at the "poles" of the orbit (where cos(θ) and sin(θ) partially cancel) would drift inward on the hypersphere — into lower-probability regions — producing subtle quality degradation at those positions.

### Orbit Modes

| Mode | Trajectory Shape | Best For |
|---|---|---|
| `Orthogonal_Basis` | Perfect circle in a random 2D subspace | Smooth 360° tours, general animation |
| `Phase_Modulated` | Figure-8 / Lissajous curve | More complex variation, non-repeating paths |
| `Harmonic_Torus` | Torus surface (two independent frequencies) | Ultra-diverse long loops, music video style |

### Semantic Orbit Control

Connect `direction_x` and `direction_y` inputs from two independent **Cross-Modal Bridge** nodes to lock the orbital plane to semantic directions. Example: `direction_x = "warm"`, `direction_y = "dark"`. The orbit then sweeps from warm-bright → dark-bright → dark-cool → warm-cool → back to warm-bright, giving you a perceptually meaningful semantic loop rather than a random one.

### ffmpeg Assembly Command

```bash
ffmpeg -framerate 24 -pattern_type glob -i "output/*.png" \
       -vf "scale=1024:1024,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
       -loop 0 orbit_animation.gif
```

For MP4 (smaller file, better quality):
```bash
ffmpeg -framerate 24 -pattern_type glob -i "output/*.png" \
       -c:v libx264 -pix_fmt yuv420p -crf 18 orbit_animation.mp4
```

---

## 📁 Workflow Files

| Format | Path |
|---|---|
| ComfyUI drag-and-drop | [`workflows/ui/Gimbal_08_HarmonicOrbiter.json`](../../workflows/ui/Gimbal_08_HarmonicOrbiter.json) |
| API / FLUX variant | [`workflows/api_flux/API_FLUX_Gimbal_08_HarmonicOrbiter.json`](../../workflows/api_flux/API_FLUX_Gimbal_08_HarmonicOrbiter.json) |

---

*Gimbal Node Suite — Form & Noise Atelier*
*Navigate latent space with precision flight instruments, not lottery prompts.*
