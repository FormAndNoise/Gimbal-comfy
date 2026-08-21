"""
[Form & Noise Atelier — Gimbal Node Suite]

Shared Spherical Linear Interpolation (Slerp) module.

Provides μ-centered and origin-centered Slerp functions for navigating
high-dimensional latent spaces along the Typical Set (the high-probability
hyperspherical shell described by the Gaussian Annulus Theorem).

Mathematical Foundation:
    In high dimensions (D > 100), probability mass concentrates in a thin
    spherical shell of radius ≈ √D, NOT at the origin. Naive Lerp cuts
    through the hollow center, causing "variance collapse" (faded, blurry
    midpoints). μ-centered Slerp anchors the trajectory to an empirical
    population centroid μ, keeping the interpolation path on the
    high-probability manifold.

    Slerp_μ(a, b, t) = μ + [sin((1-t)ω)/sin(ω) · â + sin(tω)/sin(ω) · b̂] · r(t)

    where â = (a - μ) / ‖a - μ‖,  b̂ = (b - μ) / ‖b - μ‖,
          ω = arccos(â · b̂),  r(t) = (1-t)‖a - μ‖ + t‖b - μ‖.
"""

import torch
from typing import Optional


def compute_batch_centroid(samples: torch.Tensor) -> torch.Tensor:
    """
    Compute the empirical centroid μ from a batch of latents.

    - For a multi-sample cohort (B > 2), computes the mean template across the batch:
      mean(dim=0, keepdim=True).
    - For small batches or single samples (B <= 2), computes the channel-wise spatial mean:
      mean(dim=(0, -2, -1), keepdim=True), yielding a [1, C, 1, 1] baseline anchor that
      preserves individual spatial variance and prevents degenerate (z - μ = 0) collapse.

    Args:
        samples: Latent tensor of shape [B, C, H, W].

    Returns:
        Centroid tensor of shape [1, C, H, W] or [1, C, 1, 1].
    """
    if samples.ndim != 4:
        raise ValueError(
            f"compute_batch_centroid: expected 4-D tensor [B, C, H, W], "
            f"got {list(samples.shape)}."
        )
    if samples.shape[0] > 2:
        return samples.float().mean(dim=0, keepdim=True)
    else:
        return samples.float().mean(dim=(0, -2, -1), keepdim=True)


def slerp_mu_centered(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float,
    mu: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    μ-centered Spherical Linear Interpolation.

    Interpolates between tensors a and b along a great-circle arc on the
    hypersphere centered at the empirical population centroid μ, keeping
    the trajectory strictly within the high-probability Typical Set shell.

    Math:
        ã = a - μ,  b̃ = b - μ
        â = ã / ‖ã‖,  b̂ = b̃ / ‖b̃‖
        ω = arccos(clamp(â · b̂, -1+ε, 1-ε))
        Slerp_μ(a, b, t) = μ + [sin((1-t)ω)/sin(ω) · â + sin(tω)/sin(ω) · b̂] · r(t)
        where r(t) = (1-t)‖ã‖ + t‖b̃‖

    Falls back to Lerp when vectors are nearly parallel (ω ≈ 0).

    Args:
        a:   Source tensor [B, C, H, W] or [B, D].
        b:   Target tensor, same shape as a.
        t:   Interpolation parameter in [0, 1]. 0 → a, 1 → b.
        mu:  Population centroid tensor, broadcastable to a's shape.
        eps: Numerical stability floor.

    Returns:
        Interpolated tensor, same shape and dtype as a.
    """
    orig_shape = a.shape
    orig_dtype = a.dtype
    B = a.shape[0]

    # Flatten to [B, D] for vector math, always in float32
    a_flat = a.reshape(B, -1).float()
    b_flat = b.reshape(B, -1).float()

    # Ensure mu matches spatial shape and batch size of a before flattening
    if mu.shape != a.shape:
        if mu.ndim == a.ndim:
            mu_t = mu.expand_as(a)
        elif mu.ndim == 2 and a.ndim == 4:
            mu_t = mu.reshape(mu.shape[0], a.shape[1], 1, 1).expand_as(a)
        elif mu.numel() == 1:
            mu_t = mu.expand_as(a)
        else:
            # Try general broadcast
            mu_t = torch.broadcast_to(mu, a.shape)
    else:
        mu_t = mu

    mu_flat = mu_t.reshape(B, -1).float()

    # Center on μ
    a_centered = a_flat - mu_flat
    b_centered = b_flat - mu_flat

    # Compute norms of centered vectors
    a_norm = a_centered.norm(dim=1, keepdim=True).clamp(min=eps)
    b_norm = b_centered.norm(dim=1, keepdim=True).clamp(min=eps)

    # Unit direction vectors
    a_hat = a_centered / a_norm
    b_hat = b_centered / b_norm

    # Cosine of angle between centered directions
    dot = (a_hat * b_hat).sum(dim=1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)

    # Angle between directions
    omega = torch.acos(dot)

    # Parallel-vector check: if ω ≈ 0 or ω ≈ π, fall back to Lerp
    sin_omega = torch.sin(omega)
    is_parallel = (sin_omega.abs() < 1e-4)

    # Safe denominators
    sin_omega_safe = sin_omega.clamp(min=1e-5)

    # Slerp direction coefficients
    coeff_a = torch.sin((1.0 - t) * omega) / sin_omega_safe
    coeff_b = torch.sin(t * omega) / sin_omega_safe

    # Interpolated direction (on unit sphere centered at μ)
    slerp_direction = coeff_a * a_hat + coeff_b * b_hat

    # Smoothly interpolate the radius (distance from μ)
    interp_radius = (1.0 - t) * a_norm + t * b_norm

    # Slerp result: μ + direction * radius
    slerp_result = mu_flat + slerp_direction * interp_radius

    # Lerp fallback for parallel vectors
    lerp_result = (1.0 - t) * a_flat + t * b_flat

    # Select based on parallel mask
    out_flat = torch.where(is_parallel, lerp_result, slerp_result)

    return out_flat.reshape(orig_shape).to(orig_dtype)


def slerp_origin_centered(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Origin-centered Spherical Linear Interpolation (legacy behavior).

    Equivalent to μ-centered Slerp with μ = 0. Preserved for backwards
    compatibility. This traverses a great-circle arc on the hypersphere
    centered at the coordinate origin.

    Note: In high dimensions, this path cuts inward toward the
    low-probability origin, causing variance collapse at midpoints.
    Prefer slerp_mu_centered() for production workflows.

    Args:
        a:   Source tensor [B, C, H, W] or [B, D].
        b:   Target tensor, same shape as a.
        t:   Interpolation parameter in [0, 1].
        eps: Numerical stability floor.

    Returns:
        Interpolated tensor, same shape and dtype as a.
    """
    mu_zero = torch.zeros(1, *a.shape[1:], device=a.device, dtype=a.dtype)
    return slerp_mu_centered(a, b, t, mu_zero, eps)


def slerp_pair_mu_centered(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float,
    mu: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    μ-centered Slerp for flattened 1-D vector pairs (no batch dimension).

    Used by GimbalWaypointSpline where waypoints are already flattened
    to [D] tensors. Handles radius interpolation and parallel fallback.

    Args:
        a:   Source vector [D].
        b:   Target vector [D].
        t:   Interpolation parameter in [0, 1].
        mu:  Population centroid vector [D].
        eps: Numerical stability floor.

    Returns:
        Interpolated vector [D], same dtype as a.
    """
    import math

    a_f = a.float()
    b_f = b.float()
    mu_f = mu.float()

    if mu_f.numel() != a_f.numel():
        if mu_f.numel() == 1:
            mu_f = mu_f.expand_as(a_f)
        elif a_f.numel() % mu_f.numel() == 0:
            repeat = a_f.numel() // mu_f.numel()
            mu_f = mu_f.repeat_interleave(repeat)
        else:
            mu_f = torch.broadcast_to(mu_f, a_f.shape)

    # Center on μ
    a_centered = a_f - mu_f
    b_centered = b_f - mu_f

    a_norm = a_centered.norm().clamp(min=eps)
    b_norm = b_centered.norm().clamp(min=eps)
    a_hat = a_centered / a_norm
    b_hat = b_centered / b_norm

    dot = (a_hat * b_hat).sum().clamp(-1.0 + eps, 1.0 - eps)
    omega = math.acos(dot.item())

    if omega < 1e-4:
        # Parallel fallback — Lerp in original space
        return ((1.0 - t) * a_f + t * b_f).to(a.dtype)

    sin_omega = math.sin(omega)
    w_a = math.sin((1.0 - t) * omega) / sin_omega
    w_b = math.sin(t * omega) / sin_omega

    # Interpolated direction on the μ-centered sphere
    interp_dir = w_a * a_hat + w_b * b_hat

    # Smoothly interpolate radius
    target_radius = (1.0 - t) * a_norm + t * b_norm

    result = mu_f + interp_dir * target_radius
    return result.to(a.dtype)


def slerp_pair_origin_centered(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Origin-centered Slerp for flattened 1-D vector pairs (legacy behavior).

    Args:
        a:   Source vector [D].
        b:   Target vector [D].
        t:   Interpolation parameter in [0, 1].
        eps: Numerical stability floor.

    Returns:
        Interpolated vector [D], same dtype as a.
    """
    mu_zero = torch.zeros_like(a)
    return slerp_pair_mu_centered(a, b, t, mu_zero, eps)
