# 🔄 Gimbal Circular Orbit

> *Set it to 36 steps and walk around your subject. Seamlessly. Forever.*

**Class**: `GimbalCircularOrbit`  
**Category**: `Gimbal/Trajectory`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT, DICT)` → `latent_batch`, `flight_telemetry`

---

## What It Does

The Circular Orbit generates a **closed-loop batch of latent frames** by traversing a perfect circle through latent space around a center point. Feed the resulting batch through a KSampler at low denoise and you get **seamless, variance-preserved frames** that loop back to the start without a visual jump.

Use it for:
- 360° architectural tours
- Infinite looping product animations
- Continuous orbital auditing of a concept's latent neighborhood
- Generating a diverse set of views from a single reference

---

## The Math (Simple Version)

The orbit traces a circle in the 2D subspace spanned by two orthonormal vectors **u** and **v**:

```
z(θ) = center + r × (cos(θ)·u + sin(θ)·v)
```

Where:
- `center` is your reference latent
- `r` is the orbital radius
- `θ` advances from 0° to 360° across all steps
- `u, v` are orthonormal basis vectors (perpendicular to each other in R^D)

Because the path is a **closed circle** (θ=0 and θ=360° are the same point), the animation loops perfectly with no discontinuity.

---

## Quick Wiring

```
[KSampler → center concept] ─▶ center_latent ─▶ [🔄 Circular Orbit]
                                                    steps = 36
                                                    radius = 0.96
                                                    orbit_mode = Orthogonal_Basis
                                                    preserve_hypersphere_norm = True
                                                           │
                                                   latent_batch (36 frames)
                                                           │
                                               [KSampler] (denoise=0.45, CFG=4.5)
                                                           │
                                               [VAEDecode] ─▶ 36 PNG frames
```

---

## Inputs

| Parameter | Type | Default | Range | Description |
|:---|:---|:---|:---|:---|
| `center_latent` | LATENT | — | — | The orbital center — the concept you'll circle around. |
| `steps` | INT | 36 | 3–1024 | Number of frames. 36 = 10°/frame. 72 = 5°/frame (smoother). |
| `radius` | FLOAT | 1.0 | 0.01–20.0 | Orbital radius. Near 1.0 = near Typical Set shell. |
| `orbit_mode` | enum | Orthogonal_Basis | 3 modes | How u and v basis vectors are generated. |
| `preserve_hypersphere_norm` | BOOL | True | — | Project each frame to center's L2 radius. Prevents density burn. |
| `seed` | INT | 0 | 0–2^64 | Random seed for basis generation. Change to explore different orbital planes. |
| `direction_x` | LATENT (optional) | — | — | Lock the u-axis to this semantic direction. |
| `direction_y` | LATENT (optional) | — | — | Lock the v-axis to this semantic direction. |
| `mask` | MASK (optional) | — | — | Spatially restrict orbital motion to masked region. |
| `mu_centroid` | LATENT (optional) | — | — | Population centroid for norm projection. Defaults to center_latent's radius. |

---

## Orbit Modes

### `Orthogonal_Basis` (Recommended)

Generates a random orthonormal pair **u, v** using Gram-Schmidt orthonormalization. The result is a perfect circle in a randomly-oriented 2D subspace of R^D. Changing `seed` explores a different orbital plane through the same center.

This is the most stable mode and the one used in all certification runs.

### `Phase_Modulated`

Applies a harmonic phase offset between the u and v components:
```
z(θ) = center + r × (cos(θ)·u + sin(θ+φ)·v)
```
Where φ is the phase offset. Creates figure-8, elliptical, or Lissajous trajectories instead of a perfect circle. Good for more complex looping paths.

### `Harmonic_Torus`

Combines two independent orbital frequencies: one on the primary circle, one on a smaller secondary orbit. The result is a toroidal path through latent space. Useful for generating more varied batches with a secondary "wobble" around the main orbital path.

---

## Choosing the Right Radius

| Radius | Effect |
|:---|:---|
| 0.50 | Very close to center. Subtle, coherent variations. |
| 0.96 | **Recommended.** Near the Typical Set shell. Rich variation, consistent quality. |
| 1.5 | Wider orbit. More dramatic variation, possible artifacts at extreme points. |
| 2.0+ | Risk of low-probability latent regions. May produce abstract or incoherent frames. |

The `preserve_hypersphere_norm=True` flag partially mitigates high-radius issues by projecting each frame back to the center's L2 norm — but it's not a complete fix for very large radii.

---

## Gallery: Architectural Orbit

![Base Building](../../assets/test_runs/architectural_showcases/08_arch_base_hero.png)  
*The center latent: a photorealistic architectural exterior.*

| Frame 0 | Frame 1 | Frame 2 |
|:---:|:---:|:---:|
| ![Frame 0](../../assets/test_runs/architectural_showcases/08_arch_orbit_frame_00.png) | ![Frame 1](../../assets/test_runs/architectural_showcases/08_arch_orbit_frame_01.png) | ![Frame 2](../../assets/test_runs/architectural_showcases/08_arch_orbit_frame_02.png) |

| Frame 3 | Frame 4 | Frame 5 |
|:---:|:---:|:---:|
| ![Frame 3](../../assets/test_runs/architectural_showcases/08_arch_orbit_frame_03.png) | ![Frame 4](../../assets/test_runs/architectural_showcases/08_arch_orbit_frame_04.png) | ![Frame 5](../../assets/test_runs/architectural_showcases/08_arch_orbit_frame_05.png) |

*Six frames from a 36-step orbit. The building's silhouette and massing remain consistent while lighting and material perspective shift around it.*

---

## Gallery: Desert Orbit Tour

| Frame 0 | Frame 1 | Frame 2 |
|:---:|:---:|:---:|
| ![D0](../../assets/test_runs/fresh_run6_exploration/08_desert_orbit_tour_frame_00.png) | ![D1](../../assets/test_runs/fresh_run6_exploration/08_desert_orbit_tour_frame_01.png) | ![D2](../../assets/test_runs/fresh_run6_exploration/08_desert_orbit_tour_frame_02.png) |

*Desert landscape orbital tour — different orbital plane via different seed.*

---

## Pro Tips

- **36 steps for 10°/frame**: This is the minimum for smooth motion. Use 72 for ultra-smooth, or 18 for rough storyboarding.
- **Low denoise on the output KSampler**: Use 0.40–0.50. Higher denoise gives the UNet too much latitude to diverge from the orbital geometry.
- **Semantic axis lock**: Connect a Cross-Modal Bridge output to `direction_x` to constrain one axis to a semantic direction (e.g., 'warm' vs 'cool'). The orbit then sweeps a semantically meaningful plane.
- **Different seeds = different character**: The orbital plane is random — different seeds produce orbits with different aesthetic character around the same center.
- **Seamless loops**: Because θ is evenly spaced from 0 to 2π (exclusive), frames 0 and N are mathematically adjacent. No crossfade needed for a loop.

---

## Under the Hood (Researchers)

**Gram-Schmidt orthonormalization** (Orthogonal_Basis mode):
```python
u_hat = u / ||u||
v_ortho = v - (v·u_hat)·u_hat
v_hat = v_ortho / ||v_ortho||
```
Guarantees u_hat ⊥ v_hat in R^D.

**`preserve_hypersphere_norm`**: After computing each frame z(θ), re-scales it so that `||z(θ) - μ|| = ||center - μ||`. This keeps every frame at the same L2 radius as the center, maintaining constant density on the Typical Set shell.

**Batch output shape**: `[steps, C, H, W]`. For SDXL with steps=36: `[36, 4, 128, 128]`.

**Memory**: Each step is computed sequentially via a Python loop over θ values; frames are stacked into a single batch tensor. For 36 steps × SDXL: ~36 × 4 × 128 × 128 × 4 bytes ≈ 9.4 MB in float32.

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
