# 🔢 Gimbal Latent Math (Dispatcher & Primitive Lab)

> *Expert-grade mathematical laboratory exposing all 13 LAMNr normalizing-flow and disentanglement primitives in a single configurable dispatcher node.*

**Class**: `GimbalLatentMath`  
**Category**: `Gimbal/Primitives`  
**VibeCheck**: 🟢 Stabilized  
**Returns**: `(LATENT, FLOAT, DICT)` → `latent`, `mean_metric`, `op_telemetry`

---

## What It Does

`GimbalLatentMath` is the power user's swiss army knife. Instead of wiring multiple discrete nodes, this single node exposes the full mathematical catalog of the **LAMNr (Latent-Aligned Multiview Normalizing) framework**.

- **Transform Operations**: Apply coordinate normalization, Woodbury matrix imputation, bounded scale capping, or dequantization jitter directly to the tensor.
- **Metric Operations**: Pass the latent tensor through unmodified while calculating research-grade telemetry (Exact Log-Likelihood, Mahalanobis OOD distance, Total Correlation coordinate dependence, and Geodesic angular distance).

---

## Operations Catalog (`op`)

| Op Category | Operation Name | Eq Ref | Description | Key Parameters |
| :--- | :--- | :---: | :--- | :--- |
| **Transform** | `pipeline` | Full | Runs full LAMNr stack (Cap → Jitter → Truncation → Woodbury). | `psi`, `subspace_rank`, `scale_cap` |
| **Transform** | `channel_diagonal_gaussian` | **E2** | Per-channel zero-mean/unit-variance normalization. | — |
| **Transform** | `truncation` | **E3** | Variance shrinkage towards empirical population mean $\mu$. | `psi`, `channel_adaptive` |
| **Transform** | `slerp_mu` | **E4** | Spherical geodesic interpolation towards target on Typical Set shell. | `t`, `additional_latent` |
| **Transform** | `bounded_scale` | **E11** | Bounded scale map suppressing extreme activation spikes ($s \tanh$). | `scale_cap` |
| **Transform** | `dequantize` | **E12** | Uniform stochastic noise jitter smoothing discrete quantization boundaries. | `jitter_strength`, `schedule` |
| **Transform** | `woodbury_impute` | **E7/E8** | Low-rank MMSE conditional mean subspace denoiser ($O(B D r)$). | `subspace_rank`, `sigma2` |
| **Metric** | `log_likelihood` | **E1/E2** | Exact log-likelihood under diagonal Gaussian base prior. | — |
| **Metric** | `mahalanobis` | **E9** | Anomaly detection score against cohort covariance. | `subspace_rank`, `sigma2` |
| **Metric** | `total_correlation` | **E10** | Minibatch statistical dependence between latent dimensions. | `jitter_strength` (bandwidth) |
| **Metric** | `geodesic` | **E5** | Angular distance in $[0, \pi]$ relative to target latent. | `additional_latent` |

---

## Inputs & Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | Primary input latent tensor $[B, C, H, W]$. |
| `op` | enum | `pipeline` | Mathematical primitive to execute. |
| `psi` | FLOAT | 0.90 | Truncation shrinkage coefficient ($z' = \mu + \psi(z - \mu)$). |
| `t` | FLOAT | 0.50 | Interpolation fraction for `slerp_mu` ($0.0 \rightarrow \text{base}, 1.0 \rightarrow \text{target}$). |
| `subspace_rank` | INT | -1 | Rank $r$ for Woodbury SVD decomposition (-1 = full available rank, 0 = isotropic mean). |
| `scale_cap` | FLOAT | 10.0 | Maximum scale bound ($s' = \text{cap} \cdot \tanh(s / \text{cap})$). |
| `jitter_strength` | FLOAT | 0.001 | Dequantization noise magnitude $\mathcal{U}(-1, 1) \cdot \alpha$. |
| `schedule` | enum | `linear` | Decay schedule for dequantization jitter (`linear`, `cosine`, `exponential`). |
| `step` / `total_steps` | INT | 0 / 1 | Denoising step timeline for jitter decay calculations. |
| `channel_adaptive` | BOOLEAN | True | Compute mean and variance statistics per channel independently. |
| `additional_latent` | LATENT (opt) | — | Secondary latent required for `slerp_mu` and `geodesic` distance operations. |
| `reference_batch` | LATENT (opt) | — | Cohort reference batch for establishing empirical centroid $\mu$ and covariance $\Sigma$. |

---

## Telemetry Output Schema (`op_telemetry`)

When executing metric or transform operations, `op_telemetry` emits structured diagnostic JSON:
```json
{
  "op": "mahalanobis",
  "per_sample_metric": [1.42, 1.88, 3.12, 1.15],
  "mean_metric": 1.8925,
  "shape": [4, 4, 128, 128],
  "dtype": "torch.float32",
  "device": "cuda:0"
}
```

---

*Part of the Gimbal Node Suite — Form & Noise Atelier*
