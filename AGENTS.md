# AGENTS.md — Gimbal Node Suite

Latent space flight instruments for visual generative artists (ComfyUI Custom Node Suite).

**Brand Identity**:
- Name: **Gimbal** (Repo: `gimbal-comfy`, Package: `ComfyUI-Gimbal`)
- Accent: Instrument Teal (`#0E8A8A` / Dark: `#35B8B8`)
- Job Line: *"Navigate latent space with precision flight instruments, not lottery prompts."*
- Symbol: 24×24u artificial horizon / attitude indicator dial with a pitch ladder and 1 solid center heading pip.
- Suite Model: Loose Endorsed Family (Form & Noise Atelier). Standard canvas: Accent + Ink (#141414) + Ground (Paper #F6F1EA / Void #0B0B0B). Space Grotesk / Inter / IBM Plex Mono.

## Key Capabilities & Nodes

### Flight Instruments & Trajectory Navigation
1. **`GimbalCompass_Pro`**: Standard, Normalized, Orthogonal, and SLERP vector directional steering with mask guidance.
2. **`GimbalManifold_Explorer`**: 2D topological manifold slice generator across independent X/Y latent vectors with contact-sheet grid stitching.
3. **`GimbalCircularOrbit`**: Constant-radius closed-loop spherical orbits for seamless looping animations and harmonic tours.
4. **`GimbalWaypointSpline`**: Continuous Catmull-Rom spherical spline flight paths through N arbitrary latent keypoints with constant geodesic velocity.
5. **`Gimbal_SemanticSlider`**: PCA/SVD decomposition extracting orthogonal variance directions for independent attribute steering.
6. **`Gimbal_CrossModalBridge`**: Multimodal CLIP text-to-latent projection steering diffusion without re-prompting.
7. **`GimbalChannelSplit` / `Merge` / `Scale`**: Subspace channel matrix for SDXL (4-ch) and FLUX/SD3 (16-ch) independent frequency/structural steering.
8. **`GimbalTruncation`**: Latent variance shrinkage toward distribution centroid ($z' = \mu + \psi(z - \mu)$) to rein in noisy outliers.
9. **`GimbalVectorAnalogy`**: Classic $A - B + C$ GAN concept analogy arithmetic with orthogonal projection and norm preservation.
10. **`GimbalGPS_Anchor` & `GPS_Load`**: Cryptographic coordinate hashing, disk caching, and statistical rescaling across checkpoints.
11. **`GimbalDiagnostics`**: Live telemetry reporting min, max, mean, std, L2 norm, and channel variance.
12. **`GimbalLatentStabilizer`** (LAMNr): Full quality pipeline - bounded coupling scale, dequantization jitter, truncation, Woodbury low-rank conditional-mean denoise.
13. **`GimbalLatentMath`** (LAMNr): Dispatcher node routing every equation (E1-E12) through generic inputs.
14. **`GimbalLatentTelemetry`** (LAMNr): Research-grade OOD metrics - exact log-likelihood, Mahalanobis, Total Correlation, geodesic angular distance.

### ComfyUI Frontend Extension
- `web/js/gimbal.js`: Atelier Dark `#0B0B0B` / `#0E8A8A` HUD styling, live tensor diagnostics widgets, and dynamic grid sample calculations.
