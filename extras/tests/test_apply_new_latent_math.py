"""
Numerical-correctness tests for the LAMNr / Disentanglement latent math in
``nodes/gimbal_latent_math.py`` (the pure-PyTorch ``apply_new_latent_math``).

These tests verify, for every core equation in the research:

  * tensor shape  [B, C, H, W] preserved on transform ops;
  * dtype preserved (incl. float16);
  * numerical stability (no NaN / Inf, parallel-vector fallbacks);
  * exact mathematical identities (Slerp endpoints, truncation psi=1,
    Woodbury conditional mean, Mahalanobis closed form, log-likelihood,
    channel-diagonal normalization, bounded scale, dequantization jitter);
  * the dispatcher contract (default op, unknown op raises, non-4D raises).
"""
import sys
import math
from pathlib import Path

import pytest
import torch

# Ensure the nodes directory is importable as a top-level package.
nodes_dir = Path(__file__).resolve().parent.parent.parent / "nodes"
sys.path.insert(0, str(nodes_dir))

import gimbal_latent_math as L          # noqa: E402
from gimbal_latent_math import (        # noqa: E402
    apply_new_latent_math,
    channel_stats,
    channel_diagonal_gaussian,
    truncation,
    slerp_mu,
    geodesic_angular,
    low_rank_covariance_svd,
    woodbury_impute,
    mahalanobis,
    total_correlation,
    log_likelihood,
    dequantize,
    bounded_scale,
    run_lamnr_pipeline,
    cholesky_with_jitter,
)


# --- helpers ---------------------------------------------------------------

def _latent(B=4, C=4, H=8, W=8, seed=0, dtype=torch.float32):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(B, C, H, W, dtype=dtype)


def _assert_no_nan_inf(t, name):
    assert not torch.isnan(t).any().item(), f"{name}: produced NaN"
    assert not torch.isinf(t).any().item(), f"{name}: produced Inf"


# ===========================================================================
# Dispatcher contract
# ===========================================================================

class TestDispatcher:
    def test_default_op_is_pipeline(self):
        z = _latent()
        default = apply_new_latent_math(z)
        explicit = apply_new_latent_math(z, "pipeline")
        assert torch.allclose(default, explicit, atol=1e-6)

    def test_unknown_op_raises(self):
        z = _latent()
        with pytest.raises(ValueError, match="unknown op"):
            apply_new_latent_math(z, "no_such_op")

    def test_non_4d_raises(self):
        z = torch.randn(4, 8, 8)
        with pytest.raises(ValueError, match="4-D tensor"):
            apply_new_latent_math(z, "truncation", 0.5)

    def test_non_tensor_raises(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            apply_new_latent_math([1, 2, 3], "truncation", 0.5)

    def test_all_ops_dispatch(self):
        z, tgt = _latent(), _latent()
        for op, args in [
            ("channel_diagonal_gaussian", ()),
            ("truncation", (0.7,)),
            ("slerp_mu", (tgt, 0.5)),
            ("geodesic", (tgt,)),
            ("mahalanobis", ()),
            ("woodbury_impute", ()),
            ("total_correlation", ()),
            ("log_likelihood", ()),
            ("dequantize", (1e-3, "cosine")),
            ("bounded_scale", (2.0,)),
            ("pipeline", ()),
        ]:
            out = apply_new_latent_math(z, op, *args)
            assert torch.is_tensor(out), f"{op}: must return a tensor"


# ===========================================================================
# E2  Channel-diagonal Gaussian base
# ===========================================================================

class TestChannelDiagonalGaussian:
    def test_shape_and_dtype(self):
        z = _latent(dtype=torch.float16)
        out = channel_diagonal_gaussian(z)
        assert out.shape == z.shape
        assert out.dtype == torch.float16

    def test_zero_mean_unit_var_per_channel(self):
        # A latent drawn from a per-channel Gaussian should normalize to ~N(0,1)
        # per channel (mean ~0, std ~1) up to finite-sample tolerance.
        torch.manual_seed(7)
        z = torch.randn(64, 4, 16, 16) * torch.tensor([1.0, 2.0, 0.5, 3.0]).view(1, 4, 1, 1) \
            + torch.tensor([0.0, 1.0, -2.0, 0.5]).view(1, 4, 1, 1)
        out = channel_diagonal_gaussian(z)
        mu = out.mean(dim=(0, 2, 3))
        sd = out.std(dim=(0, 2, 3))
        assert torch.allclose(mu, torch.zeros_like(mu), atol=1e-4)
        assert torch.allclose(sd, torch.ones_like(sd), atol=1e-4)

    def test_invertibility(self):
        z = _latent()
        mu_c, s_c = channel_stats(z)
        out = channel_diagonal_gaussian(z)
        recon = out * (s_c + 1e-8) + mu_c
        assert torch.allclose(recon, z, atol=1e-4)


# ===========================================================================
# E3  Truncation / variance shrinkage
# ===========================================================================

class TestTruncation:
    def test_psi_one_is_identity(self):
        z = _latent()
        mu = z.mean(dim=(0, 2, 3), keepdim=True)
        out = truncation(z, 1.0, mu=mu)
        assert torch.allclose(out, z, atol=1e-5)

    def test_psi_lt_one_shrinks_variance(self):
        z = _latent()
        out = truncation(z, 0.5)
        assert out.var() < z.var()

    def test_psi_gt_one_grows_variance(self):
        z = _latent()
        out = truncation(z, 1.5)
        assert out.var() > z.var()

    def test_psi_zero_is_mean(self):
        z = _latent()
        mu = z.mean(dim=(0, 2, 3), keepdim=True)
        out = truncation(z, 0.0, mu=mu)
        assert torch.allclose(out, mu.expand_as(z), atol=1e-5)

    def test_shape_dtype(self):
        z = _latent(dtype=torch.float16)
        out = truncation(z, 0.8)
        assert out.shape == z.shape and out.dtype == torch.float16


# ===========================================================================
# E4  mu-centered Slerp
# ===========================================================================

class TestSlerpMu:
    def test_t0_returns_source(self):
        a, b = _latent(), _latent()
        out = slerp_mu(a, b, 0.0)
        assert torch.allclose(out, a, atol=1e-5)

    def test_t1_returns_target(self):
        a, b = _latent(), _latent()
        out = slerp_mu(a, b, 1.0)
        assert torch.allclose(out, b, atol=1e-5)

    def test_midpoint_no_variance_collapse(self):
        # The mu-centered radius at t=0.5 must stay within [min, max] of the
        # endpoint radii (no collapse toward the hollow origin).
        torch.manual_seed(42)
        mu = torch.randn(1, 4, 16, 16) * 0.05
        a = mu + torch.randn(1, 4, 16, 16)
        b = mu + torch.randn(1, 4, 16, 16)
        out = slerp_mu(a, b, 0.5, mu=mu)
        r_a = (a - mu).norm().item()
        r_b = (b - mu).norm().item()
        r_m = (out - mu).norm().item()
        assert r_m >= min(r_a, r_b) * 0.95
        assert r_m <= max(r_a, r_b) * 1.05

    def test_parallel_vectors_fallback_no_nan(self):
        a = torch.ones(1, 4, 8, 8)
        b = a + 1e-6
        mu = torch.zeros(1, 4, 8, 8)
        out = slerp_mu(a, b, 0.5, mu=mu)
        _assert_no_nan_inf(out, "slerp_mu(parallel)")
        assert torch.allclose(out, (a + b) / 2, atol=1e-3)

    def test_mu_broadcast_from_1c11(self):
        # mu of shape [1, C, 1, 1] must broadcast correctly (regression guard).
        a, b = _latent(), _latent()
        mu = a.mean(dim=(0, 2, 3), keepdim=True)   # [1, C, 1, 1]
        out = slerp_mu(a, b, 0.5, mu=mu)
        assert out.shape == a.shape
        _assert_no_nan_inf(out, "slerp_mu([1,C,1,1])")

    def test_dtype_preserved(self):
        a = _latent(dtype=torch.float16)
        b = _latent(dtype=torch.float16)
        mu = torch.zeros(1, 4, 8, 8, dtype=torch.float16)
        out = slerp_mu(a, b, 0.5, mu=mu)
        assert out.dtype == torch.float16


# ===========================================================================
# E5  Geodesic (angular) distance
# ===========================================================================

class TestGeodesic:
    def test_self_distance_near_zero(self):
        z = _latent()
        d = geodesic_angular(z, z)
        assert d.max().item() < 1e-2      # eps-clamped floor, not exactly 0
        assert d.shape == (z.shape[0],)

    def test_orthogonal_vectors_quarter_turn(self):
        # Two orthogonal unit offsets from mu -> angular distance = pi/2.
        mu = torch.zeros(1, 1, 1, 4)
        a = mu.clone(); a[..., 0] = 1.0
        b = mu.clone(); b[..., 1] = 1.0
        d = geodesic_angular(a, b, mu=mu)
        assert abs(d.item() - math.pi / 2) < 1e-4

    def test_opposite_vectors_half_turn(self):
        mu = torch.zeros(1, 1, 1, 4)
        a = mu.clone(); a[..., 0] = 1.0
        b = mu.clone(); b[..., 0] = -1.0
        d = geodesic_angular(a, b, mu=mu)
        assert abs(d.item() - math.pi) < 1e-3

    def test_range_in_unit_pi(self):
        z, other = _latent(), _latent()
        d = geodesic_angular(z, other)
        assert (d >= -1e-5).all() and (d <= math.pi + 1e-5).all()


# ===========================================================================
# E6 / E7 / E8  Low-rank SVD + Woodbury conditional mean
# ===========================================================================

class TestWoodburyImpute:
    def test_rank_zero_is_mean_template(self):
        # With no retained subspace, the conditional mean collapses to mu.
        z = _latent()
        out = woodbury_impute(z, rank=0)
        mu = z.float().mean(dim=0)           # per-batch spatial mean [C, H, W]
        assert torch.allclose(out.float(), mu.unsqueeze(0).expand_as(z), atol=1e-4)

    def test_conditional_mean_exact(self):
        # z_hat = mu + U diag(l/(l+s^2)) U^T (z - mu)  must hold exactly.
        torch.manual_seed(11)
        z = torch.randn(6, 4, 4, 4)
        B, C, H, W = z.shape
        D = C * H * W
        zf = z.float()
        mu = zf.mean(dim=0).reshape(D)
        zc = zf.reshape(B, D) - mu.unsqueeze(0)
        _, S, Vh = torch.linalg.svd(zc, full_matrices=False)
        r = 3
        U = Vh[:r].t().contiguous()
        lam = (S[:r] ** 2) / B
        s2 = 0.25
        shrink = lam / (lam + s2)
        proj = (shrink.unsqueeze(1) * (U.t() @ zc.t())).t() @ U.t()
        manual = mu.unsqueeze(0) + proj
        out = woodbury_impute(z, rank=r, sigma2=s2)
        assert torch.allclose(out, manual.reshape(B, C, H, W), atol=1e-5)

    def test_shrinkage_reduces_radius(self):
        # For a non-degenerate subspace the conditional mean is strictly
        # closer to mu than the observation (shrink factors in [0,1)).
        z = _latent()
        mu = z.float().mean(dim=0)
        out = woodbury_impute(z, rank=2, sigma2=1.0)
        r_in = (z.float() - mu).reshape(z.shape[0], -1).norm(dim=1)
        r_out = (out - mu).reshape(z.shape[0], -1).norm(dim=1)
        assert (r_out <= r_in + 1e-5).all()


# ===========================================================================
# E9  Mahalanobis distance
# ===========================================================================

class TestMahalanobis:
    def test_non_negative(self):
        z = _latent()
        d = mahalanobis(z)
        assert (d >= -1e-5).all()

    def test_rank_zero_matches_isotropic(self):
        # rank=0 -> Sigma = sigma^2 I with sigma^2 = mean eigenvalue.
        torch.manual_seed(5)
        z = torch.randn(6, 4, 4, 4)
        B, D = z.shape[0], z[0].numel()
        mu = z.mean(dim=0)
        zc = (z - mu).reshape(B, D)                       # must flatten to [B, D]
        _, S, _ = torch.linalg.svd(zc, full_matrices=False)
        sigma2 = (S ** 2 / B).mean().item()
        manual = (zc ** 2).sum(1) / sigma2
        out = mahalanobis(z, rank=0)
        assert torch.allclose(out ** 2, manual.clamp_min(0), atol=1e-3)

    def test_low_rank_matches_manual(self):
        torch.manual_seed(9)
        z = torch.randn(6, 3, 4, 4)
        B, D = z.shape[0], z[0].numel()
        mu = z.mean(dim=0)
        zc = (z - mu).reshape(B, D)
        _, S, Vh = torch.linalg.svd(zc - zc.mean(0), full_matrices=False)
        r = 3
        U = Vh[:r].t()
        lam = (S[:r] ** 2) / B
        s2 = 0.4
        shrink = lam / (lam + s2)
        w = U.t() @ zc.t()
        d2 = ((zc ** 2).sum(1) - (shrink.unsqueeze(1) * w ** 2).sum(0)) / s2
        out = mahalanobis(z, rank=r, sigma2=s2)
        assert torch.allclose(out, torch.sqrt(d2.clamp_min(0)), atol=1e-4)

    def test_shape(self):
        z = _latent()
        assert mahalanobis(z).shape == (z.shape[0],)


# ===========================================================================
# E10  Total Correlation (density-ratio trick)
# ===========================================================================

class TestTotalCorrelation:
    def test_finite_and_shape(self):
        z = _latent(B=8)
        tc = total_correlation(z)
        _assert_no_nan_inf(tc, "total_correlation")
        assert tc.shape == (8,)

    def test_matches_manual_logsumexp(self):
        # Exact recomputation of the density-ratio TC estimate (E10):
        #   TC_m = logsumexp_j(-||z_m - z_j||^2 / 2s^2) - log M
        #          - sum_d [ logsumexp_j(-(z_m_d - z_j_d)^2 / 2s^2) ] + D log M
        # with the default bandwidth s^2 = median off-diagonal joint distance.
        torch.manual_seed(1)
        z = torch.randn(5, 2, 3, 3)
        M, D = z.shape[0], z[0].numel()
        zc = z.reshape(M, D)
        sq = (zc.unsqueeze(0) - zc.unsqueeze(1)).pow(2)       # [M, M, D]
        dff = sq.sum(2)                                        # [M, M]
        off = dff[~torch.eye(M, dtype=torch.bool)]
        sigma2 = off.median().clamp_min(1e-8).item()
        log_joint = torch.logsumexp(-dff / (2 * sigma2), dim=1) - math.log(M)
        log_marg = torch.logsumexp(-sq / (2 * sigma2), dim=1).sum(1) \
            - D * math.log(M)
        manual_tc = log_joint - log_marg
        out = total_correlation(z)
        _assert_no_nan_inf(out, "total_correlation")
        assert torch.allclose(out, manual_tc, atol=1e-4)

    def test_independent_dims_near_zero(self):
        # A large iid batch: the minibatch TC estimate is ~0 on average
        # (joint density ~ product of marginals for independent dims).
        torch.manual_seed(4)
        z = torch.randn(64, 4, 8, 8)
        tc = total_correlation(z)
        _assert_no_nan_inf(tc, "total_correlation")
        assert abs(tc.mean().item()) < 1.0            # not claimed unbiased, just bounded

    def test_min_batch_size(self):
        with pytest.raises(ValueError, match="at least 2"):
            total_correlation(torch.randn(1, 4, 8, 8))


# ===========================================================================
# E1 / E2  Exact log-likelihood (channel-diagonal Gaussian)
# ===========================================================================

class TestLogLikelihood:
    def test_matches_manual_channel_diagonal(self):
        torch.manual_seed(2)
        z = torch.randn(5, 4, 6, 6)
        mu_c = z.mean(dim=(0, 2, 3), keepdim=True)
        s_c = z.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
        var = s_c ** 2 + 1e-8
        quad = ((z - mu_c) ** 2 / var).sum()
        log_det = torch.log(var).sum() * (z.shape[2] * z.shape[3])
        D = z[0].numel()
        manual = -0.5 * (quad + log_det + D * math.log(2 * math.pi))
        out = log_likelihood(z)
        # log_likelihood returns per-sample; compare to the manual per-sample sum.
        quad_b = ((z - mu_c) ** 2 / var).sum(dim=(1, 2, 3))
        manual_b = -0.5 * (quad_b + log_det + D * math.log(2 * math.pi))
        assert torch.allclose(out, manual_b, atol=1e-3)

    def test_outlier_has_lower_likelihood(self):
        # An outlier (far from the channel mean) should have a lower log-p.
        z = _latent(B=4)
        mu_c, s_c = channel_stats(z)
        out = log_likelihood(z)
        # Build an in-distribution sample and an outlier.
        inlier = z[0:1]
        outlier = (mu_c + 8.0 * s_c).expand_as(inlier)
        ll_in = log_likelihood(inlier, mu=mu_c, scale=s_c)
        ll_out = log_likelihood(outlier, mu=mu_c, scale=s_c)
        assert ll_out.item() < ll_in.item()

    def test_shape(self):
        z = _latent()
        assert log_likelihood(z).shape == (z.shape[0],)


# ===========================================================================
# E11  Bounded coupling scale
# ===========================================================================

class TestBoundedScale:
    def test_identity_when_cap_large(self):
        # With a very large cap, tanh(s/s_cap) ~ s/s_cap -> gain ~ 1.
        z = _latent()
        out = bounded_scale(z, scale_cap=1e6)
        assert torch.allclose(out, z, atol=1e-3)

    def test_bounded_magnitude(self):
        # With a tiny cap, the per-channel deviation is heavily attenuated.
        z = _latent()
        mu_c, s_c = channel_stats(z)
        out = bounded_scale(z, scale_cap=1e-3)
        assert (out - mu_c).abs().max() <= (z - mu_c).abs().max()

    def test_shape_dtype(self):
        z = _latent(dtype=torch.float16)
        out = bounded_scale(z, scale_cap=2.0)
        assert out.shape == z.shape and out.dtype == torch.float16


# ===========================================================================
# E12  Dequantization jitter
# ===========================================================================

class TestDequantize:
    def test_zero_strength_is_identity(self):
        z = _latent()
        out = dequantize(z, strength=0.0)
        assert torch.allclose(out, z)

    def test_jitter_is_zero_mean_in_expectation(self):
        # Symmetric U(-1,1) jitter -> mean difference ~ 0 over many draws.
        z = torch.zeros(1, 4, 64, 64)
        diffs = torch.stack([dequantize(z, strength=1.0, schedule="linear",
                                        generator=torch.Generator().manual_seed(i))
                            for i in range(200)])
        assert abs(diffs.mean().item()) < 1e-2

    def test_schedule_decay(self):
        # alpha(0) > alpha(t>0) for a decaying schedule.
        z = _latent()
        g = torch.Generator().manual_seed(0)
        a0 = dequantize(z, strength=1.0, schedule="linear", step=0.0, total_steps=1.0,
                        generator=torch.Generator().manual_seed(7))
        a1 = dequantize(z, strength=1.0, schedule="linear", step=1.0, total_steps=1.0,
                        generator=torch.Generator().manual_seed(7))
        assert a0.abs().sum() > a1.abs().sum()

    def test_shape_dtype(self):
        z = _latent(dtype=torch.float16)
        out = dequantize(z, strength=1e-3, schedule="cosine")
        assert out.shape == z.shape and out.dtype == torch.float16


# ===========================================================================
# E13  Cholesky with adaptive jitter (safeguard utility)
# ===========================================================================

class TestCholeskyJitter:
    def test_well_conditioned_matches_exact(self):
        A = torch.tensor([[4.0, 0.0], [0.0, 9.0]])
        L = cholesky_with_jitter(A)
        assert torch.allclose(L, torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
                              atol=1e-4)

    def test_singular_recovers_finite_factor(self):
        # A rank-1 (singular PSD) matrix must not raise; the jitter recovers a
        # finite (heavily regularized) lower-triangular factor.
        A = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        L = cholesky_with_jitter(A)
        assert torch.isfinite(L).all()
        assert L.shape == (2, 2)


# ===========================================================================
# Full LAMNr pipeline
# ===========================================================================

class TestPipeline:
    def test_shape_and_dtype(self):
        z = _latent(dtype=torch.float16)
        out = run_lamnr_pipeline(z)
        assert out.shape == z.shape and out.dtype == torch.float16
        _assert_no_nan_inf(out, "pipeline")

    def test_finite_for_float32(self):
        z = _latent()
        out = run_lamnr_pipeline(z)
        _assert_no_nan_inf(out, "pipeline")

    def test_psi_one_jitter_zero_rank_zero_is_mean(self):
        # With psi=1 (no truncation), jitter=0, rank=0 (no subspace), and a
        # very large scale_cap (so bounded_scale is identity), the pipeline
        # collapses to the batch mean template.
        z = _latent()
        out = run_lamnr_pipeline(z, psi=1.0, jitter=0.0, rank=0, scale_cap=1e6)
        mu = z.float().mean(dim=0)
        assert torch.allclose(out.float(), mu.unsqueeze(0).expand_as(z), atol=1e-3)

    def test_psi_lt_one_reduces_variance(self):
        z = _latent()
        out = run_lamnr_pipeline(z, psi=0.5, jitter=0.0)
        assert out.var() <= z.var() + 1e-4
