# 📊 Gimbal Diagnostics · 📡 Gimbal Latent Telemetry

> *Know what you're looking at. Real-time readouts for your latent tensors.*

---

## 📊 Gimbal Diagnostics (Live Stats)

**Class**: `GimbalDiagnostics`  
**Category**: `Gimbal/Telemetry`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT, DICT)` → passthrough `latent`, `stats_dict`

### What It Does

A non-destructive inspection node. Passes the latent through unchanged while reporting live statistics to the console and the output dict. Wire it anywhere in your pipeline to peek at what's happening inside your latents.

### Quick Wiring

```
[Compass Pro] ─▶ latent_out ─▶ [📊 Diagnostics] ─▶ latent (passthrough) ─▶ [KSampler]
                                                        │
                                                   stats_dict ─▶ [any display node]
```

### What It Measures

| Metric | Description |
|:---|:---|
| `min` | Minimum value in the tensor |
| `max` | Maximum value in the tensor |
| `mean` | Global mean across all elements |
| `std` | Global standard deviation |
| `l2_norm` | L2 (Euclidean) norm of the full tensor |
| `channel_variance` | Per-channel variance `[C]` — see which channels are being driven hardest |
| `shape` | Tensor shape `[B, C, H, W]` |
| `dtype` | float16, float32, bfloat16 |
| `device` | cpu / cuda:0 |

### Interpreting the Numbers

**Healthy SDXL latent (before KSampler):**
- `mean` ≈ 0.0 (centered Gaussian)
- `std` ≈ 0.9–1.2 (near unit variance)
- `l2_norm` ≈ 256–300 (√D × σ ≈ √65536 × 1.0 = 256)
- `max` typically < 4.0 (within ~4σ)

**Warning signs:**
- `max > 8.0` or `min < -8.0`: Latent values in extreme tail — activate Latent Stabilizer.
- `std > 3.0`: Variance explosion — reduce CFG or add Stabilizer.
- `channel_variance` wildly uneven (e.g., Ch3 = 5.0 when Ch0 = 0.8): A specific channel is being heavily driven.

---

## 📡 Gimbal Latent Telemetry (Research-Grade OOD Metrics)

**Class**: `GimbalLatentTelemetry`  
**Category**: `Gimbal/Telemetry`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT, DICT)` → passthrough `latent`, `telemetry_dict`

### What It Does

An advanced metrics node implementing **research-grade out-of-distribution (OOD) detection** and **disentanglement measurement**. For power users who want to deeply understand their latent space.

### Metrics Computed

| Metric | Equation | What It Tells You |
|:---|:---|:---|
| **Exact Log-Likelihood** | E1: `log p(x) = log p_Z(f(x)) + log|det J_f|` | How probable is this latent under the model's prior? Lower = more "out there." |
| **Channel Diagonal Gaussian** | E2 | Per-channel normalized log-likelihood |
| **Mahalanobis Distance** | E9 | How many "standard deviations" is this latent from the batch cohort? |
| **Total Correlation** | E10 | Are the latent's coordinate dimensions statistically independent? Lower TC = better disentanglement. |
| **Geodesic Angular Distance** | E5 | Angular distance from the centroid on the Typical Set shell (in radians) |

### Interpreting Telemetry

```json
{
  "log_likelihood": -12450.3,     ← Negative; more negative = less probable
  "mahalanobis_distance": 1.82,   ← <2.0 is well within cohort; >4.0 is anomalous
  "total_correlation": 0.023,     ← Near 0 = good disentanglement; >0.5 = entangled
  "geodesic_distance_rad": 0.31,  ← Angular distance from centroid on Typical Set
  "shape": [1, 4, 128, 128]
}
```

**Mahalanobis benchmarks:**
- `< 2.0`: Well within cohort distribution — safe zone.
- `2.0–4.0`: Moderate outlier — creative territory.
- `> 4.0`: Strong anomaly — may produce artifacts. Consider Stabilizer.

**Total Correlation benchmarks:**
- `0.0–0.05`: Excellent disentanglement.
- `0.05–0.2`: Mild correlation between channels — normal.
- `> 0.5`: Significant entanglement — attributes are "leaking" into each other.

### Pro Use Cases

- **OOD Auditing**: Feed generated latents through Telemetry before decode. Flag any with Mahalanobis > 4.0 for automatic re-roll.
- **Disentanglement Research**: Compare Total Correlation between different model architectures or conditioning methods.
- **Navigation Verification**: After a Compass Pro operation, verify the geodesic distance moved as expected.
- **Quality Gates**: Build automated pipelines that only pass latents with `mahalanobis < 3.0` to the VAE decoder.

---

## Combined Usage: The Telemetry Stack

The recommended diagnostics setup for a professional pipeline:

```
[Latent Input]
      │
[📊 Diagnostics] ─▶ quick health check → console
      │ (passthrough)
[🛡️ Stabilizer] ─▶ psi=0.88 if issues found
      │
[📡 Telemetry] ─▶ OOD metrics → quality gate check
      │ (passthrough)
[KSampler] ─▶ VAEDecode ─▶ Output
```

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
