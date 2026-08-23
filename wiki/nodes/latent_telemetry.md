# 📡 Gimbal Latent Telemetry

> *Comprehensive out-of-distribution (OOD) diagnostic suite computing exact log-likelihood, Mahalanobis distance, and Total Correlation statistical independence.*

**Class**: `GimbalLatentTelemetry`  
**Category**: `Gimbal/Telemetry`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT, DICT, STRING)` → `latent`, `telemetry_data`, `telemetry_report`

---

## What It Does

`GimbalLatentTelemetry` provides flight instruments for mathematical quality assurance. It computes non-destructive diagnostic metrics on any latent tensor, evaluating how closely the distribution conforms to high-probability manifold boundaries.

---

## Computed Telemetry Metrics

### 1. Exact Log-Likelihood (Eq. E1 / E2)
$$\log p_Z(\mathbf{z}) = -\frac{1}{2} \sum_c \left[ \frac{(z_c - \mu_c)^2}{s_c^2 + \epsilon} + \log(s_c^2 + \epsilon) + \log(2\pi) \right]$$
Measures the overall generative probability of the latent. A sharp drop indicates an out-of-distribution anomaly.

### 2. Low-Rank Mahalanobis Distance (Eq. E9)
$$d_M(\mathbf{z})^2 = \frac{1}{\sigma^2} \left[ \|\mathbf{z} - \mu\|^2 - \sum_{i=1}^r \frac{\lambda_i}{\lambda_i + \sigma^2} (\mathbf{u}_i \cdot (\mathbf{z} - \mu))^2 \right]$$
Benchmarks the latent against cohort variance without computing expensive $D \times D$ covariance matrices ($O(B D r)$ via Woodbury push-through).

### 3. Total Correlation / Disentanglement Score (Eq. E10)
$$\hat{\text{TC}}_m = \log p_{\text{joint}}(\mathbf{z}^m) - \sum_d \log p_d(z_d^m)$$
Evaluates mutual dependence between coordinate channels. Lower TC values indicate superior semantic disentanglement.

---

## Telemetry Report Format (`telemetry_report`)

```text
============================================================
GIMBAL TELEMETRY REPORT — STABILIZED FLIGHT INSTRUMENTS
============================================================
Tensor Shape        : [1, 4, 128, 128] (D = 65,536)
Mean Log-Likelihood : -12,410.84 nats
Mahalanobis Dist    : 1.84 (🟢 NOMINAL COHORT)
Total Correlation   : 0.021 nats (🟢 HIGH DISENTANGLEMENT)
Geodesic Arc Radius : 256.12 σ (🟢 ON TYPICAL SET SHELL)
Channel Variance    : [Ch0: 1.02, Ch1: 0.98, Ch2: 1.05, Ch3: 0.96]
============================================================
```

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
