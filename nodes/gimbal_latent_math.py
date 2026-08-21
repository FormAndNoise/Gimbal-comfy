"""
[Form & Noise Atelier - Gimbal Node Suite]
LAMNr / Disentangled-Representation latent-space math primitives.

Pure, highly-optimized PyTorch implementation of the core equations from:

  * "System Design Framework: Latent-Aligned Multiview Normalizing Flows (LAMNr)"
  * "Technical Synthesis: Disentangled Representation Learning in
     High-Dimensional Generative Modeling"

This module contains NO ComfyUI boilerplate. It exposes a single dispatcher

    apply_new_latent_math(latent_tensor, op, *args)

plus a set of numerically-stabilized primitives, one per core equation in the
research. All transforms accept and return a 4-D latent tensor of shape
``[B, C, H, W]``. Metric ops return a ``[B]`` tensor (one scalar per batch
sample). Internal arithmetic is performed in float32 for stability; the output
is cast back to the caller's dtype and the spatial shape is preserved.

---------------------------------------------------------------------------
Core equations implemented (cited inline on each primitive)
---------------------------------------------------------------------------
  E1  Change-of-Variables (exact log-likelihood):
        log p(x) = log p_Z(f(x)) + log|det J_f|
  E2  Channel-wise Diagonal Gaussian base  (mu_c, s_c per channel):
        log p_Z(z) = -0.5 * sum_c [ (z_c-mu_c)^2 / (s_c^2+eps) + log(s_c^2+eps) + log(2 pi) ]
  E3  Truncation / variance shrinkage toward the centroid:
        z' = mu + psi * (z - mu)
  E4  mu-centered Slerp (great-circle arc on the hypersphere centred at mu):
        Slerp_mu(a,b,t) = mu + [ sin((1-t)w)/sin(w) * a_hat
                                + sin(t w)/sin(w) * b_hat ] * r(t)
        a_hat = (a-mu)/||a-mu||,  b_hat = (b-mu)/||b-mu||,
        w = arccos(clamp(a_hat . b_hat)),  r(t) = (1-t)||a-mu|| + t||b-mu||
  E5  Geodesic (angular) distance on the mu-centred hypersphere:
        d_g(a,b) = arccos(clamp((a-mu).(b-mu) / (||a-mu|| ||b-mu||), -1, 1))
  E6  Low-rank-plus-diagonal covariance via SVD:
        Sigma ~= U Lambda U^T + sigma^2 I,   U^T U = I_r,   Lambda = diag(l_1..l_r)
  E7  Woodbury matrix identity (push-through) for the low-rank inverse:
        (sigma^2 I + U Lambda U^T)^{-1}
          = (1/sigma^2) [ I - U diag( l_i / (l_i + sigma^2) ) U^T ]
  E8  Cross-modal imputation (conditional mean, MMSE / Ridge denoising):
        z_hat = mu + U diag( l_i/(l_i+sigma^2) ) U^T (z_obs - mu)
  E9  Mahalanobis distance (anomaly detection), closed-form via E7:
        d_M(z)^2 = (1/sigma^2)[ ||z-mu||^2
                               - sum_i (l_i/(l_i+sigma^2)) (u_i . (z-mu))^2 ]
  E10 Total Correlation via the density-ratio trick / minibatch-weighted
       sampling (logsumexp-stabilized):
        TC_hat_m = log p_joint(z^m) - sum_d log p_d(z^m_d)
        where each batch sample indexes an isotropic Gaussian mixture of
        bandwidth sigma^2; the constant terms cancel exactly.
  E11 Bounded coupling scale (tanh scale_map, RealNVP/Glow safeguard):
        s' = scale_cap * tanh(s / scale_cap)
  E12 Uniform dequantization jitter with a decay schedule alpha(t):
        z' = z + U(-1,1) * alpha(t) ;  linear | cosine | exponential decay
  E13 Numerical safeguard stack: eps-floored division, acos clamping,
       parallel-vector Lerp fallback, adaptive-jitter Cholesky utility.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

__all__ = [
    "apply_new_latent_math",
    "channel_stats",
    "channel_diagonal_gaussian",
    "truncation",
    "slerp_mu",
    "geodesic_angular",
    "low_rank_covariance_svd",
    "woodbury_impute",
    "mahalanobis",
    "total_correlation",
    "log_likelihood",
    "dequantize",
    "bounded_scale",
    "run_lamnr_pipeline",
]

_EPS = 1e-8
_OPS = (
    "channel_diagonal_gaussian", "truncation", "slerp_mu", "geodesic",
    "mahalanobis", "woodbury_impute", "total_correlation", "log_likelihood",
    "dequantize", "bounded_scale", "pipeline",
)


# ---------------------------------------------------------------------------
#  Small numerics utilities (E13)
# ---------------------------------------------------------------------------

def _eps(dtype: torch.dtype) -> float:
    """Return a scale-appropriate epsilon for the working dtype."""
    if dtype == torch.float16:
        return 1e-4
    if dtype == torch.bfloat16:
        return 1e-3
    return 1e-8


def _to_f32(t: torch.Tensor) -> torch.Tensor:
    return t.float() if t.dtype != torch.float32 else t


def _rsqrt(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Reciprocal-square-root with an additive epsilon floor (no NaN/Inf)."""
    return torch.rsqrt(x.clamp_min(eps))


def cholesky_with_jitter(mat: torch.Tensor, max_jitter: float = 1e-3,
                        eps: float = 1e-6) -> torch.Tensor:
    """
    Cholesky factorization with an *adaptive jitter* (E13).

    Adds ``eps * 10**k`` to the diagonal until the factorization succeeds or
    ``max_jitter`` is reached. Used only as a robust fallback for full-covariance
    paths; the default low-rank paths avoid Cholesky entirely via E7.
    """
    d = mat.shape[-1]
    eye = torch.eye(d, dtype=mat.dtype, device=mat.device)
    jitter = 0.0
    for _ in range(12):
        try:
            return torch.linalg.cholesky(mat + jitter * eye)
        except torch._C._LinAlgError:  # noqa: BLE001 - controlled retry
            jitter = max(eps, jitter * 10.0) if jitter else eps
            if jitter > max_jitter:
                # Last resort: regularize heavily and return whatever we get.
                return torch.linalg.cholesky(
                    mat + max_jitter * eye + eps * eye)
    return torch.linalg.cholesky(mat + max_jitter * eye)


# ---------------------------------------------------------------------------
#  E2  Channel-wise Diagonal Gaussian base
# ---------------------------------------------------------------------------

def channel_stats(z: torch.Tensor, eps: float = _EPS
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Channel-wise diagonal Gaussian sufficient statistics (mu_c, s_c).

    Parameters (mu_c, s_c) are *tied within each channel* and broadcast across
    spatial locations, yielding tensors of shape ``[1, C, 1, 1]``. This is the
    LAMNr design choice that prevents "per-voxel scale collapse" and keeps the
    flow's discrete transitions a smooth diffeomorphic mapping of the manifold.

    Args:
        z: latent tensor ``[B, C, H, W]``.
        eps: additive epsilon before the square-root (numerical floor).

    Returns:
        (mu_c [1, C, 1, 1], s_c [1, C, 1, 1]) in float32.
    """
    if z.ndim != 4:
        raise ValueError(
            f"channel_stats: expected 4-D tensor [B, C, H, W], got {tuple(z.shape)}.")
    zf = _to_f32(z)
    mu_c = zf.mean(dim=(0, 2, 3), keepdim=True)
    var_c = zf.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    s_c = torch.sqrt(var_c + eps)
    return mu_c, s_c


def channel_diagonal_gaussian(z: torch.Tensor, eps: float = _EPS
                              ) -> torch.Tensor:
    """
    Bijective map into the channel-diagonal standard-Gaussian coordinates (E2).

        z_norm = (z - mu_c) / (s_c + eps)

    This is the "topological unfolding" of the manifold into a symmetric
    Gaussian base space: per-channel it centers and scales so that each channel
    becomes approximately N(0, 1). Invertibility is exact: ``z = z_norm * s_c + mu_c``.

    Returns a tensor of the same shape and dtype as ``z``.
    """
    if z.ndim != 4:
        raise ValueError(
            f"channel_diagonal_gaussian: expected 4-D tensor [B, C, H, W], "
            f"got {tuple(z.shape)}.")
    orig_dtype = z.dtype
    zf = _to_f32(z)
    mu_c, s_c = channel_stats(zf, eps=eps)
    z_norm = (zf - mu_c) / (s_c + eps)
    return z_norm.to(orig_dtype)


# ---------------------------------------------------------------------------
#  E3  Truncation / variance shrinkage
# ---------------------------------------------------------------------------

def truncation(z: torch.Tensor, psi: float,
               mu: Optional[torch.Tensor] = None,
               channel_adaptive: bool = True,
               eps: float = _EPS) -> torch.Tensor:
    """
    StyleGAN2-style latent variance truncation toward the distribution centroid.

        z' = mu + psi * (z - mu)

    ``psi < 1``  compresses outliers toward the core (cleaner, less chaotic).
    ``psi = 1``  identity.
    ``psi > 1``  exaggerates variance away from the mean.

    Args:
        z: latent ``[B, C, H, W]``.
        psi: truncation coefficient.
        mu: optional external centroid; if None it is estimated from ``z``.
        channel_adaptive: when estimating ``mu`` from ``z``, use a per-channel
            spatial mean ``[1, C, 1, 1]`` (True) or a global scalar (False).
    """
    if z.ndim != 4:
        raise ValueError(
            f"truncation: expected 4-D tensor [B, C, H, W], got {tuple(z.shape)}.")
    orig_dtype = z.dtype
    zf = _to_f32(z)
    if mu is None:
        if channel_adaptive:
            mu = zf.mean(dim=(0, 2, 3), keepdim=True)
        else:
            mu = zf.mean(dim=(0, 1, 2, 3), keepdim=True)
    else:
        mu = mu.to(zf.dtype)
    out = mu + float(psi) * (zf - mu)
    return out.to(orig_dtype)


# ---------------------------------------------------------------------------
#  E4  mu-centered Slerp
# ---------------------------------------------------------------------------

def slerp_mu(z: torch.Tensor, target: torch.Tensor, t: float,
             mu: Optional[torch.Tensor] = None,
             eps: float = _EPS) -> torch.Tensor:
    """
    mu-centered Spherical Linear Interpolation on the high-probability Typical
    Set (E4, Gaussian Annulus Theorem).

    Naive Lerp cuts through the hollow interior of the hypersphere and suffers
    "variance collapse"; mu-centered Slerp keeps the trajectory on the
    great-circle arc centred at the empirical population centroid ``mu``.

        z_t = mu + [ sin((1-t)w)/sin(w) * a_hat + sin(t w)/sin(w) * b_hat ] * r(t)
        a_hat = (z-mu)/||z-mu||,  b_hat = (target-mu)/||target-mu||
        w = arccos(clamp(a_hat . b_hat, -1+e, 1-e)),  r(t) = (1-t)||z-mu|| + t||target-mu||

    Parallel / antiparallel directions (``sin(w) ~ 0``) fall back to Lerp so the
    output is always finite. Endpoints ``t=0`` -> z, ``t=1`` -> target.

    Args:
        z: source latent ``[B, C, H, W]``.
        target: destination latent, same shape as ``z``.
        t: interpolation parameter in ``[0, 1]``.
        mu: population centroid; if None, the channel-spatial centroid of ``z``.
    """
    if z.ndim != 4:
        raise ValueError(
            f"slerp_mu: expected 4-D tensor [B, C, H, W], got {tuple(z.shape)}.")
    if target.shape != z.shape:
        raise ValueError(
            f"slerp_mu: 'target' must match 'z' shape {tuple(z.shape)}, "
            f"got {tuple(target.shape)}.")
    orig_dtype = z.dtype
    B = z.shape[0]
    zf, tf = _to_f32(z), _to_f32(target)
    if mu is None:
        mu = zf.mean(dim=(0, 2, 3), keepdim=True)
    else:
        mu = mu.to(zf.dtype)

    a = zf.reshape(B, -1)
    b = tf.reshape(B, -1)
    # Broadcast mu (possibly [1, C, 1, 1] or [B, C, H, W]) to the batch, then
    # flatten to [B, D] for vector math.
    m = mu.expand(z.shape).reshape(B, -1)

    a_c = a - m
    b_c = b - m
    a_n = a_c.norm(dim=1, keepdim=True).clamp_min(eps)
    b_n = b_c.norm(dim=1, keepdim=True).clamp_min(eps)
    a_hat = a_c / a_n
    b_hat = b_c / b_n
    dot = (a_hat * b_hat).sum(dim=1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    parallel = sin_omega.abs() < 1e-4
    sin_safe = sin_omega.clamp_min(1e-5)

    ca = torch.sin((1.0 - t) * omega) / sin_safe
    cb = torch.sin(t * omega) / sin_safe
    direction = ca * a_hat + cb * b_hat
    radius = (1.0 - t) * a_n + t * b_n
    slerp = m + direction * radius
    lerp = (1.0 - t) * a + t * b
    out = torch.where(parallel, lerp, slerp)
    return out.reshape(z.shape).to(orig_dtype)


# ---------------------------------------------------------------------------
#  E5  Geodesic (angular) distance
# ---------------------------------------------------------------------------

def geodesic_angular(z: torch.Tensor, other: torch.Tensor,
                    mu: Optional[torch.Tensor] = None,
                    eps: float = _EPS) -> torch.Tensor:
    """
    mu-centered geodesic (angular) distance between two latents (E5).

        d_g = arccos(clamp( (a-mu).(b-mu) / (||a-mu|| ||b-mu||), -1, 1 ))

    Used for semantic / directional similarity on the hypersphere. Returns a
    ``[B]`` tensor of per-sample angular distances in ``[0, pi]``.
    """
    if z.ndim != 4:
        raise ValueError(
            f"geodesic_angular: expected 4-D tensor [B, C, H, W], got {tuple(z.shape)}.")
    if other.shape != z.shape:
        raise ValueError(
            f"geodesic_angular: 'other' must match 'z' shape {tuple(z.shape)}.")
    B = z.shape[0]
    zf, of = _to_f32(z), _to_f32(other)
    if mu is None:
        mu = zf.mean(dim=(0, 2, 3), keepdim=True)
    else:
        mu = mu.to(zf.dtype)
    m = mu.expand(z.shape).reshape(B, -1)
    a = (zf.reshape(B, -1) - m)
    b = (of.reshape(B, -1) - m)
    dot = (a * b).sum(dim=1)
    denom = a.norm(dim=1) * b.norm(dim=1)
    cos = (dot / denom.clamp_min(eps)).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)


# ---------------------------------------------------------------------------
#  E6  Low-rank-plus-diagonal covariance via SVD
# ---------------------------------------------------------------------------

def low_rank_covariance_svd(z: torch.Tensor, rank: int = 0,
                            eps: float = _EPS
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor]:
    """
    Low-rank-plus-diagonal covariance factorization (E6).

        Sigma ~= U Lambda U^T + sigma^2 I,   U^T U = I_r

    Computed from the *economy SVD* of the batch-centred data so that no D x D
    matrix is ever formed (D = C*H*W can be ~10^5 while B is small):

        zc = z_flat - mu                              [B, D]
        U_b S Vh^T = svd(zc, full_matrices=False)
        U   = Vh[:r].T                                [D, r]   (right singular vectors)
        lam = S^2 / B                                 [r]     (eigenvalues)
        sigma^2 = mean of the *trailing* eigenvalues  (residual isotropic variance)

    Args:
        z: latent ``[B, C, H, W]``.
        rank: number of retained principal directions. ``0`` selects the
            *diagonal-only* model (no low-rank term); ``-1`` keeps all available
            components (``min(B-1, D)``). ``>0`` keeps the top-``rank``.
        eps: numerical floor on the residual variance.

    Returns:
        (mu [D], U [D, r], lam [r], sigma2 scalar tensor, (B, D)).
    """
    if z.ndim != 4:
        raise ValueError(
            f"low_rank_covariance_svd: expected 4-D tensor [B, C, H, W], "
            f"got {tuple(z.shape)}.")
    zf = _to_f32(z)
    B, D = zf.shape[0], zf[0].numel()
    zc_flat = zf.reshape(B, D)
    mu = zc_flat.mean(dim=0)
    zc = zc_flat - mu

    # Economy SVD: U_b [B, k], S [k], Vh [k, D] with k = min(B, D).
    # The right singular vectors Vh are the principal directions in latent space.
    try:
        _, S, Vh = torch.linalg.svd(zc, full_matrices=False)
    except torch._C._LinAlgError:  # noqa: BLE001 - fall back to a tiny jitter
        _, S, Vh = torch.linalg.svd(zc + eps * torch.randn_like(zc),
                                    full_matrices=False)

    eig = (S ** 2) / float(B)               # eigenvalues of the covariance
    k = min(B, D)
    if rank == 0:
        r = 0
    elif rank == -1:
        r = k
    else:
        r = min(max(rank, 0), k)

    if r > 0:
        U = Vh[:r].t().contiguous()         # [D, r]
        lam = eig[:r].clamp_min(0.0)         # [r]
        # Residual isotropic variance = mean of the trailing eigenvalues.
        if k > r:
            sigma2 = eig[r:].mean()
        else:
            sigma2 = eig.new_tensor(eps)
    else:
        U = zc.new_zeros(D, 0)
        lam = eig.new_zeros(0)
        sigma2 = eig.mean() if k > 0 else eig.new_tensor(eps)

    sigma2 = sigma2.clamp_min(eps)
    return mu, U, lam, sigma2, (B, D)


# ---------------------------------------------------------------------------
#  E7 + E8  Woodbury push-through -> conditional-mean imputation
# ---------------------------------------------------------------------------

def woodbury_impute(z: torch.Tensor, mu: Optional[torch.Tensor] = None,
                    rank: int = -1, sigma2: Optional[float] = None,
                    eps: float = _EPS) -> torch.Tensor:
    """
    Cross-modal conditional-mean imputation via the Woodbury push-through (E7, E8).

    Given the low-rank-plus-diagonal model ``Sigma = U Lambda U^T + sigma^2 I``
    of the cohort, the closed-form conditional mean (MMSE / Ridge denoiser) of
    the structured signal given a (possibly noisy / partially observed) latent
    ``z_obs`` is

        z_hat = mu + U diag( l_i / (l_i + sigma^2) ) U^T (z_obs - mu)

    which is exactly the Woodbury inverse applied to the residual
    ``(z_obs - mu)``. No D x D matrix is inverted; the work is ``O(B D r)`` with
    ``r`` the retained subspace dimension. Setting ``rank=0`` reduces to the
    pure shrinkage ``z_hat = mu`` (the Frechet-mean template).

    Args:
        z: observed latent ``[B, C, H, W]``.
        mu: cohort centroid; if None, estimated from ``z``.
        rank: subspace size (``0`` -> mean template, ``-1`` -> all available).
        sigma2: residual isotropic variance; if None, estimated from the SVD.
    """
    if z.ndim != 4:
        raise ValueError(
            f"woodbury_impute: expected 4-D tensor [B, C, H, W], got {tuple(z.shape)}.")
    orig_dtype = z.dtype
    zf = _to_f32(z)
    mu_est, U, lam, s2_est, (B, D) = low_rank_covariance_svd(zf, rank=rank, eps=eps)
    if mu is None:
        mu_vec = mu_est
    else:
        mu_vec = _to_f32(mu).reshape(D)

    s2 = float(sigma2) if sigma2 is not None else float(s2_est)
    s2 = max(s2, eps)

    zc = zf.reshape(B, D) - mu_vec.unsqueeze(0)            # [B, D]
    if U.shape[1] > 0 and lam.numel() > 0:
        # diag( l_i / (l_i + sigma^2) ) U^T (z - mu),  all in the r-dim subspace
        w = U.t() @ zc.t()                                # [r, B]
        shrink = lam / (lam + s2)                          # [r]
        proj = (shrink.unsqueeze(1) * w).t() @ U.t()       # [B, r] @ [r, D] -> [B, D]
        z_hat = mu_vec.unsqueeze(0) + proj
    else:
        z_hat = mu_vec.unsqueeze(0).expand(B, D)
    return z_hat.reshape(z.shape).to(orig_dtype)


# ---------------------------------------------------------------------------
#  E9  Mahalanobis distance (anomaly detection)
# ---------------------------------------------------------------------------

def mahalanobis(z: torch.Tensor, mu: Optional[torch.Tensor] = None,
                rank: int = -1, sigma2: Optional[float] = None,
                eps: float = _EPS) -> torch.Tensor:
    """
    Mahalanobis distance under the low-rank-plus-diagonal model (E9).

        d_M(z)^2 = (z-mu)^T Sigma^{-1} (z-mu)
                 = (1/sigma^2) [ ||z-mu||^2
                                 - sum_i ( l_i/(l_i+sigma^2) ) (u_i . (z-mu))^2 ]

    The inner sum is evaluated in the r-dim subspace via the Woodbury form (E7),
    so no D x D inversion is required. Returns a ``[B]`` tensor of distances
    (one per batch sample); larger values indicate stronger anomalies.
    """
    if z.ndim != 4:
        raise ValueError(
            f"mahalanobis: expected 4-D tensor [B, C, H, W], got {tuple(z.shape)}.")
    zf = _to_f32(z)
    mu_est, U, lam, s2_est, (B, D) = low_rank_covariance_svd(zf, rank=rank, eps=eps)
    mu_vec = mu_est if mu is None else _to_f32(mu).reshape(D)
    s2 = max(float(sigma2) if sigma2 is not None else float(s2_est), eps)

    zc = zf.reshape(B, D) - mu_vec.unsqueeze(0)          # [B, D]
    sq_norm = (zc ** 2).sum(dim=1)                       # ||z-mu||^2  [B]
    if U.shape[1] > 0 and lam.numel() > 0:
        w = U.t() @ zc.t()                               # [r, B]
        shrink = lam / (lam + s2)                        # [r]
        proj_sq = (w ** 2) * shrink.unsqueeze(1)         # [r, B], sum over r
        low_term = proj_sq.sum(dim=0)                    # [B]
    else:
        low_term = torch.zeros_like(sq_norm)
    d2 = (sq_norm - low_term) / s2
    d2 = d2.clamp_min(0.0)                                # guard tiny negatives
    return torch.sqrt(d2)                                # 0 when d2 <= 0; stable


# ---------------------------------------------------------------------------
#  E10  Total Correlation via the density-ratio trick
# ---------------------------------------------------------------------------

def _gauss_log_density_pairwise(diff_sq: torch.Tensor, sigma2: float,
                                M: int) -> torch.Tensor:
    """
    logsumexp-stabilized log of an isotropic Gaussian mixture evaluated at each
    sample, where the *mixture centres are the batch samples themselves*.

        log (1/M) sum_j N(x; x_j, sigma^2 I)
          = -0.5 * dim * log(2 pi sigma^2)
            + logsumexp_j( -||x - x_j||^2 / (2 sigma^2) ) - log M

    Args:
        diff_sq: squared distances ``[M, M]`` (sample i vs centre j).
        sigma2: isotropic bandwidth.
        M: number of mixture components.

    Returns:
        ``[M]`` tensor of log-densities (constant term NOT included).
    """
    log_kernel = -diff_sq / (2.0 * sigma2)
    return torch.logsumexp(log_kernel, dim=1) - math.log(M)


def total_correlation(z: torch.Tensor, bandwidth: Optional[float] = None,
                     eps: float = _EPS) -> torch.Tensor:
    """
    Total Correlation estimate via the density-ratio trick (E10).

    TC = KL( joint(z) || prod_d marginal(z_d) ) measures statistical
    dependence between the latent coordinates; it is the engine of
    disentanglement. The density-ratio trick approximates the intractable joint
    and product-of-marginals by Gaussian mixtures whose *centres are the batch
    samples themselves* (minibatch-weighted sampling):

        log p_joint(x^m)   = logsumexp_j( -||x^m - x^j||^2 / 2s^2 ) - log M
        sum_d log p_d(x^m_d) = sum_d [ logsumexp_j( -(x^m_d - x^j_d)^2 / 2s^2 ) - log M ]
        TC_hat_m = log p_joint(x^m) - sum_d log p_d(x^m_d)

    The ``-0.5 dim log(2 pi s^2)`` and ``-log M`` constants cancel between the
    joint and the summed marginals, so the estimate reduces exactly to the
    difference of logsumexp terms shown. ``logsumexp`` keeps the computation
    numerically stable. Returns a ``[B]`` tensor of per-sample TC estimates;
    the scalar TC is ``.mean()``.

    Args:
        z: latent ``[B, C, H, W]``.
        bandwidth: Gaussian bandwidth ``sigma``. If None, a robust default of
            ``sqrt(median pairwise distance)`` is used (a standard heuristic).
    """
    if z.ndim != 4:
        raise ValueError(
            f"total_correlation: expected 4-D tensor [B, C, H, W], got {tuple(z.shape)}.")
    if z.shape[0] < 2:
        raise ValueError(
            "total_correlation: requires a batch of at least 2 samples.")
    zf = _to_f32(z)
    M, D = zf.shape[0], zf[0].numel()
    zc = zf.reshape(M, D)

    # Pairwise squared distances [M, M].
    sq = (zc.unsqueeze(0) - zc.unsqueeze(1)).pow(2)      # [M, M, D]
    diff_sq_full = sq.sum(dim=2)                          # [M, M]  joint distances
    diff_sq_diag = sq                                     # [M, M, D] per-dim distances

    if bandwidth is None:
        with torch.no_grad():
            # Robust bandwidth: sqrt of the median off-diagonal joint distance.
            off = diff_sq_full[~torch.eye(M, dtype=torch.bool,
                                          device=zc.device)]
            med = off.median().clamp_min(eps)
            sigma2 = med.item() if med.numel() else 1.0
            sigma2 = max(sigma2, eps)
    else:
        sigma2 = max(float(bandwidth) ** 2, eps)

    # Joint log-density (isotropic over all D dims).
    log_joint = _gauss_log_density_pairwise(diff_sq_full, sigma2, M)   # [M]
    # Per-dim log-densities, summed over D.
    per_dim_logk = torch.logsumexp(-diff_sq_diag / (2.0 * sigma2), dim=1)   # [M, D]
    log_marginals = per_dim_logk.sum(dim=1) - D * math.log(M)               # [M]

    tc = log_joint - log_marginals
    return tc


# ---------------------------------------------------------------------------
#  E1 / E2  Exact log-likelihood under the channel-diagonal Gaussian base
# ---------------------------------------------------------------------------

def log_likelihood(z: torch.Tensor, mu: Optional[torch.Tensor] = None,
                   scale: Optional[torch.Tensor] = None,
                   eps: float = _EPS) -> torch.Tensor:
    """
    Exact log-likelihood under the channel-wise diagonal Gaussian base (E1, E2).

        log p_Z(z) = -0.5 * sum_c [ (z_c - mu_c)^2 / (s_c^2 + eps)
                                    + log(s_c^2 + eps) + log(2 pi) ]

    This is the change-of-variables form ``log p(x) = log p_Z(f(x)) +
    log|det J_f|`` specialised to the *channel-diagonal* affine map
    ``f(z) = (z - mu_c)/s_c`` (so the log-Jacobian is ``-sum_c log s_c`` and is
    already absorbed into the per-channel ``log(s_c^2)`` term above). Returns a
    ``[B]`` tensor of per-sample log-likelihoods (lower = more anomalous / OOD).

    Args:
        z: latent ``[B, C, H, W]``.
        mu: external centroid ``[1, C, 1, 1]``; if None, estimated from ``z``.
        scale: external per-channel std ``[1, C, 1, 1]``; if None, estimated.
    """
    if z.ndim != 4:
        raise ValueError(
            f"log_likelihood: expected 4-D tensor [B, C, H, W], got {tuple(z.shape)}.")
    zf = _to_f32(z)
    if mu is None or scale is None:
        mu_c, s_c = channel_stats(zf, eps=eps)
        if mu is not None:
            mu_c = mu.to(zf.dtype)
        if scale is not None:
            s_c = scale.to(zf.dtype).clamp_min(eps)
    else:
        mu_c = mu.to(zf.dtype)
        s_c = scale.to(zf.dtype).clamp_min(eps)

    var = (s_c ** 2) + eps
    log2pi = math.log(2.0 * math.pi)
    # Per-channel contributions, summed over H, W (and implicitly over C via the
    # broadcast of [1, C, 1, 1]); then the [B] batch dimension is retained.
    quad = ((zf - mu_c) ** 2) / var                        # [B, C, H, W]
    n_hw = zf.shape[2] * zf.shape[3]
    log_det = torch.log(var).sum() * n_hw                  # scalar (sum_c log s_c^2) * H*W
    quad_sum = quad.sum(dim=(1, 2, 3))                     # [B]
    return -0.5 * (quad_sum + log_det + float(zf[0].numel()) * log2pi)


# ---------------------------------------------------------------------------
#  E11  Bounded coupling scale (tanh scale_map)
# ---------------------------------------------------------------------------

def bounded_scale(z: torch.Tensor, scale_cap: float = 1.0,
                  eps: float = _EPS) -> torch.Tensor:
    """
    Bounded coupling scale safeguard (E11, RealNVP/Glow scale_map).

    The per-channel scale ``s_c`` is passed through a tanh ``scale_map`` that
    bounds its effective magnitude to ``[-scale_cap, scale_cap]`` while leaving
    the sign and the zero point intact:

        s_bounded = scale_cap * tanh(s_c / scale_cap)
        z' = mu_c + (z - mu_c) * (s_bounded / (s_c + eps))

    For ``s_c << scale_cap`` the gain tends to 1 (identity); for ``s_c >>
    scale_cap`` the gain saturates, suppressing exploding magnitudes. Returns a
    tensor of the same shape and dtype as ``z``.
    """
    if z.ndim != 4:
        raise ValueError(
            f"bounded_scale: expected 4-D tensor [B, C, H, W], got {tuple(z.shape)}.")
    if scale_cap <= 0:
        raise ValueError("bounded_scale: 'scale_cap' must be > 0.")
    orig_dtype = z.dtype
    zf = _to_f32(z)
    mu_c, s_c = channel_stats(zf, eps=eps)
    s_bounded = float(scale_cap) * torch.tanh(s_c / float(scale_cap))
    gain = s_bounded / (s_c + eps)
    out = mu_c + (zf - mu_c) * gain
    return out.to(orig_dtype)


# ---------------------------------------------------------------------------
#  E12  Uniform dequantization jitter with a decay schedule
# ---------------------------------------------------------------------------

def dequantize(z: torch.Tensor, strength: float = 1e-3,
               schedule: str = "linear", step: float = 0.0,
               total_steps: float = 1.0, eps: float = _EPS,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Uniform dequantization jitter with a decay schedule (E12).

        z' = z + U(-1, 1) * alpha(t)

    The scalar schedule ``alpha(t)`` prevents the model from collapsing onto
    "spiky" discrete modes and is annealed across training:

        linear      alpha(t) = a0 * (1 - t)
        cosine      alpha(t) = a0 * 0.5 * (1 + cos(pi * t))
        exponential alpha(t) = a0 * exp(-k * t)

    where ``t = step / total_steps`` is a progress fraction in ``[0, 1]``. The
    jitter is symmetric (zero-mean) so the latent statistics are unbiased.

    Args:
        z: latent ``[B, C, H, W]``.
        strength: peak jitter amplitude ``a0``.
        schedule: one of {"linear", "cosine", "exponential"}.
        step: current step (or progress fraction if ``total_steps`` == 1).
        total_steps: total number of steps; the progress is ``step/total_steps``.
        generator: optional torch.Generator for reproducible jitter.
    """
    if z.ndim != 4:
        raise ValueError(
            f"dequantize: expected 4-D tensor [B, C, H, W], got {tuple(z.shape)}.")
    if strength <= 0:
        return z.clone()
    schedule = schedule.lower()
    if schedule not in ("linear", "cosine", "exponential"):
        raise ValueError(
            f"dequantize: 'schedule' must be linear|cosine|exponential, "
            f"got {schedule!r}.")
    ts = max(float(total_steps), 1.0)
    t = max(0.0, min(1.0, float(step) / ts))

    if schedule == "linear":
        alpha = strength * (1.0 - t)
    elif schedule == "cosine":
        alpha = strength * 0.5 * (1.0 + math.cos(math.pi * t))
    else:  # exponential
        alpha = strength * math.exp(-t)

    if alpha <= 0.0:
        return z.clone()
    orig_dtype = z.dtype
    zf = _to_f32(z)
    noise = torch.rand(zf.shape, dtype=torch.float32, device=zf.device,
                       generator=generator) * 2.0 - 1.0
    out = zf + noise * alpha
    return out.to(orig_dtype)


# ---------------------------------------------------------------------------
#  Full LAMNr quality-improvement pipeline
# ---------------------------------------------------------------------------

def run_lamnr_pipeline(z: torch.Tensor, psi: float = 0.9,
                       rank: int = -1, sigma2: Optional[float] = None,
                       jitter: float = 0.0, scale_cap: float = 10.0,
                       eps: float = _EPS) -> torch.Tensor:
    """
    Full LAMNr latent quality-improvement stack, in evaluation order:

    1. Channel-diagonal Gaussian base (E2): centre & scale per channel.
    2. Bounded coupling scale (E11): suppress exploding magnitudes.
    3. Uniform dequantization jitter (E12): ``alpha = jitter`` (skipped if 0).
    4. Truncation toward the centroid (E3): ``psi < 1`` reins in outliers.
    5. Woodbury conditional-mean denoising (E7/E8): project the residual onto
       the shared low-rank cohort subspace, removing idiosyncratic noise while
       preserving global structure.

    Each stage preserves ``[B, C, H, W]`` shape and the input dtype. The output
    is a stabilized latent whose per-channel statistics are bounded, whose
    outliers are pulled toward the Frechet-mean template, and whose high-frequency
    idiosyncrasies are projected out in the shared subspace.

    Args:
        z: latent ``[B, C, H, W]``.
        psi: truncation coefficient (``< 1`` cleans, ``= 1`` identity, ``> 1`` exaggerates).
        rank: Woodbury subspace size (``-1`` keeps all available components).
        sigma2: residual isotropic variance; if None, estimated from the SVD.
        jitter: dequantization peak amplitude (``0`` disables the stage).
        scale_cap: bounded-scale magnitude cap.
    """
    if z.ndim != 4:
        raise ValueError(
            f"run_lamnr_pipeline: expected 4-D tensor [B, C, H, W], "
            f"got {tuple(z.shape)}.")
    # Work in float32 throughout, cast back at the very end.
    zf = _to_f32(z)
    mu_c, _ = channel_stats(zf, eps=eps)                      # E2 stats (for reporting)
    z1 = bounded_scale(zf, scale_cap=scale_cap, eps=eps)    # E11 safeguard
    if jitter > 0.0:
        z1 = dequantize(z1, strength=jitter, schedule="cosine",
                        step=0.0, total_steps=1.0, eps=eps)  # E12
    z2 = truncation(z1, psi=psi, mu=mu_c, channel_adaptive=True, eps=eps)  # E3
    # Woodbury conditional mean: estimate the cohort centroid (mu) and subspace
    # internally from z2 so the residual lives in the low-rank geometry of z2.
    z3 = woodbury_impute(z2, mu=None, rank=rank, sigma2=sigma2, eps=eps)   # E7/E8
    return z3.to(z.dtype)


# ---------------------------------------------------------------------------
#  Dispatcher
# ---------------------------------------------------------------------------

def apply_new_latent_math(latent_tensor: torch.Tensor, *args) -> torch.Tensor:
    """
    Dispatcher for the LAMNr / disentanglement latent math primitives.

    Signature:
        apply_new_latent_math(latent_tensor, op=None, *op_args)

    The first positional argument in ``*args`` selects the operation; the
    remaining positional arguments are forwarded to the primitive. If no ``op``
    is given, the full LAMNr quality-improvement pipeline (``run_lamnr_pipeline``)
    is applied.

    Return contract:
        * Transform ops -> tensor of the same ``[B, C, H, W]`` shape and dtype.
        * Metric ops    -> ``[B]`` tensor (one scalar per batch sample).

    Supported ``op`` values and their positional arguments:

        channel_diagonal_gaussian (eps=1e-8)
            -> normalized latent [B, C, H, W]                      (E2)
        truncation (psi, mu=None, channel_adaptive=True, eps=1e-8)
            -> truncated latent [B, C, H, W]                        (E3)
        slerp_mu (target, t, mu=None, eps=1e-8)
            -> interpolated latent [B, C, H, W]                     (E4)
        geodesic (other, mu=None, eps=1e-8)
            -> angular distance [B] in [0, pi]                       (E5)
        mahalanobis (mu=None, rank=-1, sigma2=None, eps=1e-8)
            -> Mahalanobis distance [B]                            (E9)
        woodbury_impute (mu=None, rank=-1, sigma2=None, eps=1e-8)
            -> conditional-mean latent [B, C, H, W]                (E7/E8)
        total_correlation (bandwidth=None, eps=1e-8)
            -> per-sample Total Correlation [B]                    (E10)
        log_likelihood (mu=None, scale=None, eps=1e-8)
            -> per-sample log-likelihood [B]                       (E1/E2)
        dequantize (strength=1e-3, schedule="linear", step=0.0,
                    total_steps=1.0, eps=1e-8)
            -> jittered latent [B, C, H, W]                        (E12)
        bounded_scale (scale_cap=1.0, eps=1e-8)
            -> scale-bounded latent [B, C, H, W]                   (E11)
        pipeline (psi=0.9, rank=-1, sigma2=None, jitter=0.0,
                  scale_cap=10.0, eps=1e-8)
            -> stabilized latent [B, C, H, W]                       (full stack)
    """
    if not torch.is_tensor(latent_tensor):
        raise TypeError(
            f"apply_new_latent_math: 'latent_tensor' must be a torch.Tensor, "
            f"got {type(latent_tensor).__name__}.")
    if latent_tensor.ndim != 4:
        raise ValueError(
            f"apply_new_latent_math: expected 4-D tensor [B, C, H, W], "
            f"got {tuple(latent_tensor.shape)}.")

    op = args[0] if args else "pipeline"
    rest = args[1:] if len(args) > 1 else ()

    if op == "channel_diagonal_gaussian":
        return channel_diagonal_gaussian(latent_tensor, *(rest or (_EPS,)))
    if op == "truncation":
        return truncation(latent_tensor, *rest)
    if op == "slerp_mu":
        return slerp_mu(latent_tensor, *rest)
    if op == "geodesic":
        return geodesic_angular(latent_tensor, *rest)
    if op == "mahalanobis":
        return mahalanobis(latent_tensor, *rest)
    if op == "woodbury_impute":
        return woodbury_impute(latent_tensor, *rest)
    if op == "total_correlation":
        return total_correlation(latent_tensor, *rest)
    if op == "log_likelihood":
        return log_likelihood(latent_tensor, *rest)
    if op == "dequantize":
        return dequantize(latent_tensor, *rest)
    if op == "bounded_scale":
        return bounded_scale(latent_tensor, *rest)
    if op == "pipeline" or op is None:
        return run_lamnr_pipeline(latent_tensor, *rest)

    raise ValueError(
        f"apply_new_latent_math: unknown op {op!r}. "
        f"Expected one of {_OPS + ('pipeline',)}.")
