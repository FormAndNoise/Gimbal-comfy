# 〰️ Gimbal Waypoint Spline

> *Plan multi-stop geodesic flight paths through latent space with smooth Catmull-Rom or SLERP trajectories.*

**Class**: `GimbalWaypointSpline`  
**Category**: `Gimbal/Trajectory`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT, DICT)` → `latent_path`, `flight_telemetry`

---

## What It Does

While **Compass Pro** flies between two points and **Circular Orbit** loops in a circle, **Gimbal Waypoint Spline** allows you to define an **arbitrary sequence of $N \ge 2$ waypoint latents** (e.g. Concept A → Concept B → Concept C → Concept D) and computes a smooth, continuous path connecting them all.

It supports true **Spherical SLERP**, **Catmull-Rom Spherical Splines**, and **Geodesic Constant-Velocity Reparameterization**, guaranteeing that camera, concept, or lighting transitions never accelerate or stutter between waypoints.

---

## At a Glance

```
[Batch of N Waypoint Latents] ──▶ [〰️ Waypoint Spline] ──▶ latent_path (Batch of M frames)
                                   total_steps = 60         flight_telemetry
                                   spline_mode = "Spherical_SLERP"
                                   loop_trajectory = False
                                   constant_velocity = True
```

---

## Inputs

| Parameter | Type | Default | Range | Description |
|:---|:---|:---|:---|:---|
| `waypoints_batch` | LATENT | — | $N \ge 2$ | Batch of keypoint latents representing the path stops. |
| `total_steps` | INT | 60 | 4 – 2048 | Total number of interpolated frames to generate along the entire flight path. |
| `spline_mode` | enum | `Spherical_SLERP` | 4 modes | Interpolation curve algorithm (see below). |
| `loop_trajectory` | BOOLEAN | False | True/False | Connects the final waypoint back to the first waypoint for a continuous closed loop. |
| `tension` | FLOAT | 0.5 | 0.0 – 1.0 | Spline curve tension (Catmull-Rom mode only). Higher = tighter turns. |
| `constant_velocity` | BOOLEAN | True | True/False | Reparameterizes sample distribution by geodesic arc length for uniform travel speed across long and short segments. |
| `mu_anchor` | LATENT (opt) | — | — | Population centroid for $\mu$-centered SLERP. Defaults to empirical mean of all waypoints. |

---

## Spline Modes in Depth

### 1. `Spherical_SLERP` (Piecewise Geodesic)
Connects consecutive waypoints using great-circle arcs on the high-dimensional hypersphere shell:
$$\mathbf{z}(t) = \text{SLERP}_\mu(\mathbf{w}_i, \mathbf{w}_{i+1}, t)$$
Provides exact waypoint passing and zero variance collapse, with sharp direction changes at each waypoint node.

### 2. `Catmull_Rom_Spline` (Smooth Spherical Spline)
Computes smooth $C^1$-continuous cubic tangents across 4-waypoint windows:
$$\mathbf{p}(u) = \frac{1}{2} \begin{bmatrix} 1 & u & u^2 & u^3 \end{bmatrix} \begin{bmatrix} 0 & 2 & 0 & 0 \\ -\tau & 0 & \tau & 0 \\ 2\tau & \tau-6 & 6-2\tau & -\tau \\ -\tau & 4-\tau & \tau-4 & \tau \end{bmatrix} \begin{bmatrix} \mathbf{w}_{i-1} \\ \mathbf{w}_i \\ \mathbf{w}_{i+1} \\ \mathbf{w}_{i+2} \end{bmatrix}$$
Followed by normalization back to the hypersphere shell. Ensures seamless, organic curve transitions with zero sharp corners.

### 3. `Normalized_Linear`
Fast piecewise linear interpolation with post-hoc L2 unit normalization. Useful for rapid draft sweeps.

### 4. `Cosine_Ease`
SLERP interpolation modulated with an S-curve cosine easing schedule ($t' = \frac{1 - \cos(\pi t)}{2}$), providing smooth deceleration into each waypoint and acceleration out.

---

## Geodesic Constant-Velocity Reparameterization

When waypoints are spaced unevenly (e.g. Waypoint 1 to 2 is distant, while 2 to 3 is subtle), naive step allocation causes the flight to "rush" through large transitions and "crawl" through small ones.

When `constant_velocity = True`, the node calculates exact geodesic arc lengths:
$$d_g(\mathbf{w}_i, \mathbf{w}_{i+1}) = \arccos\left(\frac{(\mathbf{w}_i - \mu) \cdot (\mathbf{w}_{i+1} - \mu)}{\|\mathbf{w}_i - \mu\| \|\mathbf{w}_{i+1} - \mu\|}\right)$$
And allocates steps proportionally to segment length:
$$S_i = \max\left(1, \text{round}\left(S_{\text{total}} \cdot \frac{d_g(\mathbf{w}_i, \mathbf{w}_{i+1})}{D_{\text{total}}}\right)\right)$$

---

## Pro Tips

- **Batch Construction**: Combine multiple latent anchors using standard ComfyUI `Latent Batch` nodes or multiple `GPS Load` outputs stacked together.
- **Looping Flythroughs**: Set `loop_trajectory = True` and feed to video assembly scripts (e.g. ffmpeg) for seamless looping concept journeys.
- **Tension Tuning**: In `Catmull_Rom_Spline` mode, set `tension = 0.75` if you notice overshoot loops on tightly clustered waypoints.

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
