# 📡 Channel Architecture: 4-Channel vs 16-Channel Models

> *"More channels isn't just 'more quality' — it's a fundamentally different language for describing an image."*

---

## 🎛️ The Beginner's Guide: Signal Buses

Every image the AI generates lives as a **latent tensor** — a compact grid of numbers that the VAE decoder eventually unfolds into pixels. The number of **channels** in that tensor determines how richly the latent space can describe visual information.

Think of channels like buses on a mixing desk. Older models have 4 buses. Newer ones have 16.

### SD 1.5 / SDXL — 4 Channels

With 4 channels, the latent must pack everything about an image into four signals. A rough analogy:

| Channel | What it roughly carries |
|---|---|
| 0 | Overall structure — where things are |
| 1 | Large-scale shapes and object boundaries |
| 2 | Color information, warm/cool balance |
| 3 | Fine detail, texture, and surface quality |

**Practical implication:** Structure and material are entangled. Changing color often nudges shapes. Changing texture bleeds into outlines.

### FLUX.1 / SD3 — 16 Channels

Sixteen channels give the model 4× as much latent bandwidth. Visual signals that were previously packed together get their own dedicated buses:

| Channel group | What it roughly carries |
|---|---|
| 0–7 | Spatial composition, geometry, object massing |
| 8–15 | Surface finish, color gamut, micro-texture |

**Practical implication:** Structure and material are much more separable. You can often mutate material channels without touching structure channels — and Gimbal knows exactly how to exploit this.

### Gimbal Auto-Detection 🤖

You never need to tell Gimbal which architecture you're using. Every Gimbal node inspects the tensor shape at runtime:

- `[B, 4, H, W]` → SDXL mode
- `[B, 16, H, W]` → FLUX.1 / SD3 mode

Math, defaults, and channel split indices all adjust automatically.

---

## 🔬 Technical Deep-Dive

### SDXL Latent Anatomy — `[B, 4, 128, 128]`

SDXL's VAE encodes 1024×1024 images to a 128×128 spatial grid with 4 channels. Post-training analysis of activation statistics across ~50k samples reveals this functional split:

| Channel | Role | Spatial Frequency |
|---|---|---|
| Ch 0 | Low-freq structural massing, large object contours | Low |
| Ch 1 | Boundary geometry, negative-space definition | Low–Mid |
| Ch 2 | Chroma signal, color harmony, lighting speculars | Mid–High |
| Ch 3 | High-freq surface texture, micro-detail, material grain | High |

**Critical insight:** Channels 0–1 carry the *where* (composition, silhouette), while channels 2–3 carry the *what* (material, finish). This clean split is the basis for the **Channel Split trick**.

---

### FLUX.1 Latent Anatomy — `[B, 16, 64, 64]`

FLUX.1's DiT backbone works with a denser 16-channel representation on a smaller 64×64 grid. The effective dimensionality is identical (D = 65,536), but the channel roles are more nuanced:

| Channel cluster | Role | Dominant signal type |
|---|---|---|
| Ch 0–3 | Primary spatial geometry, large-form composition | Structural |
| Ch 4–7 | Secondary geometry, mid-range shape articulation | Structural–transitional |
| Ch 8–11 | Color gamut bands, material reflectance classes | Material |
| Ch 12–15 | Micro-texture, surface finish, high-freq chroma | Material–textural |

FLUX channels exhibit more **cross-channel correlation** than SDXL. Raw per-channel indexing is less meaningful; Gimbal's FLUX mode operates on **cluster projections** rather than discrete channel offsets.

---

### The Channel Split Trick ✂️

`GimbalChannelSplit` lets you surgically divide a latent into structure and material halves, process them independently, then recombine.

**SDXL recipe:**
```
Split index: 2
→ Structure half: Ch 0–1  (frozen or gently guided)
→ Material half:  Ch 2–3  (aggressively steered via GimbalCrossModalBridge)
```

**FLUX.1 recipe:**
```
Split index: 8
→ Structure half: Ch 0–7   (frozen or gently guided)
→ Material half:  Ch 8–15  (aggressively steered via GimbalCrossModalBridge)
```

**What this achieves:**
- **100% silhouette lock** — the structure channels are never modified, so object contours, poses, and spatial composition are pixel-identical to the reference.
- **Complete material transformation** — the material channels can be driven to an entirely different finish (e.g., stone → liquid chrome → brushed copper) without bleeding into shape.

> 🛠️ **Power User Tip:** Chain two `GimbalCrossModalBridge` nodes on the material half — one for macro material class (e.g., `metallic`), one for micro finish (e.g., `brushed directional grain`) — before recombining.

---

### Cross-Modal Bridge: SDXL vs FLUX Mode

`GimbalCrossModalBridge` operates differently depending on detected architecture:

**SDXL mode (4-channel):**
- Keyword signatures map to per-channel scalar offsets
- Example: `'warm saturated vivid'` → `[Δ_ch0, Δ_ch1, Δ_ch2, Δ_ch3]`
- Precise per-channel control; tight correspondence between keyword and channel target

**FLUX mode (16-channel):**
- Keyword signatures map to **broad-spectrum cluster projections** across all 16 channels
- Cluster membership is soft (weighted, not binary)
- Example: `'saturation'` → weighted influence on channels 4–9 (color gamut cluster)
- This prevents keyword guidance from becoming overly localized in a 16-channel space

**Architecture detection toggle:**
Both modes are selected automatically. The `mode` parameter on `GimbalCrossModalBridge` is only needed for manual override or debugging.

---

### Channel Architecture × Gaussian Annulus

A subtlety worth noting: both architectures share **D = 65,536** effective dimensions.

```
SDXL:  4 ch × 128 × 128 = 65,536
FLUX:  16 ch × 64 × 64  = 65,536
```

The Gaussian Annulus theorem applies identically to both. The shell radius, SLERP geometry, and LAMNr normalization math are the same — which is why Gimbal's core navigation logic is architecture-agnostic at the math level, even when channel handling differs.

→ See [The Gaussian Annulus Theorem](gaussian_annulus.md) for the full treatment.

---

### Channel Map Reference Table

| Property | SDXL | FLUX.1 / SD3 |
|---|---|---|
| Latent shape | `[B, 4, 128, 128]` | `[B, 16, 64, 64]` |
| Total dimensions D | 65,536 | 65,536 |
| Structure channels | 0–1 | 0–7 |
| Material channels | 2–3 | 8–15 |
| Split index | 2 | 8 |
| Bridge guidance mode | Per-channel offset | Cluster projection |
| Typical Set radius | ≈ 256σ | ≈ 256σ |

---

*Page maintained by Form & Noise Atelier · Gimbal Node Suite documentation*
