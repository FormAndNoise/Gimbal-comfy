"""
Tests for μ-centered Slerp integration across the Gimbal Node Suite.

Validates the shared gimbal_slerp module and the upgraded interpolation
behavior in Manifold Explorer, Compass Pro, Waypoint Spline, and Circular Orbit.
"""
import sys
import math
from pathlib import Path

import pytest
import torch

# Ensure nodes directory is in sys.path for direct imports
nodes_dir = Path(__file__).resolve().parent.parent.parent / "nodes"
sys.path.insert(0, str(nodes_dir))

from gimbal_slerp import (
    slerp_mu_centered,
    slerp_origin_centered,
    slerp_pair_mu_centered,
    slerp_pair_origin_centered,
    compute_batch_centroid,
)
from gimbal_manifold_explorer import GimbalManifold_Explorer, WayfinderManifold_Explorer
from gimbal_compass import GimbalCompass_Pro, WayfinderCompass_Pro
from gimbal_waypoint_spline import GimbalWaypointSpline
from gimbal_circular_orbit import GimbalCircularOrbit


# ─── Helpers ──────────────────────────────────────────────────────────

def _make_latent(B=1, C=4, H=8, W=8, seed=None):
    """Create a deterministic random LATENT dict."""
    if seed is not None:
        torch.manual_seed(seed)
    return {"samples": torch.randn(B, C, H, W)}


def _flat_norm(t: torch.Tensor, mu: torch.Tensor = None) -> float:
    """L2 norm of (t - mu) flattened, as a Python float."""
    if mu is not None:
        diff = (t - mu).float().reshape(t.shape[0], -1)
    else:
        diff = t.float().reshape(t.shape[0], -1)
    return diff.norm(dim=1).mean().item()


# ═══════════════════════════════════════════════════════════════════════
# 1. SHARED SLERP MODULE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestSlerpMuCentered:
    """Tests for the core slerp_mu_centered() function."""

    def test_endpoint_t0_returns_a(self):
        """slerp(a, b, 0.0, μ) must return a exactly."""
        a = torch.randn(1, 4, 8, 8)
        b = torch.randn(1, 4, 8, 8)
        mu = torch.randn(1, 4, 8, 8) * 0.1
        result = slerp_mu_centered(a, b, 0.0, mu)
        assert torch.allclose(result, a, atol=1e-5), \
            f"t=0.0 should return a. Max diff: {(result - a).abs().max():.6f}"

    def test_endpoint_t1_returns_b(self):
        """slerp(a, b, 1.0, μ) must return b exactly."""
        a = torch.randn(1, 4, 8, 8)
        b = torch.randn(1, 4, 8, 8)
        mu = torch.randn(1, 4, 8, 8) * 0.1
        result = slerp_mu_centered(a, b, 1.0, mu)
        assert torch.allclose(result, b, atol=1e-5), \
            f"t=1.0 should return b. Max diff: {(result - b).abs().max():.6f}"

    def test_midpoint_norm_preserved(self):
        """
        At t=0.5, the μ-centered radius should be between ‖a-μ‖ and ‖b-μ‖.
        This is the critical test: no variance collapse at the midpoint.
        """
        torch.manual_seed(42)
        mu = torch.randn(1, 4, 16, 16) * 0.05  # near-zero centroid
        a = mu + torch.randn(1, 4, 16, 16)      # unit-ish offset
        b = mu + torch.randn(1, 4, 16, 16)

        result = slerp_mu_centered(a, b, 0.5, mu)

        # Centered norms
        r_a = _flat_norm(a, mu)
        r_b = _flat_norm(b, mu)
        r_mid = _flat_norm(result, mu)

        r_min = min(r_a, r_b) * 0.95  # small tolerance
        r_max = max(r_a, r_b) * 1.05

        assert r_mid >= r_min, \
            f"Midpoint radius {r_mid:.4f} collapsed below min({r_a:.4f}, {r_b:.4f})"
        assert r_mid <= r_max, \
            f"Midpoint radius {r_mid:.4f} exceeded max({r_a:.4f}, {r_b:.4f})"

    def test_mu_zero_matches_origin_centered(self):
        """μ=0 should produce identical output to origin-centered Slerp."""
        torch.manual_seed(123)
        a = torch.randn(2, 4, 8, 8)
        b = torch.randn(2, 4, 8, 8)
        mu_zero = torch.zeros(1, 4, 8, 8)

        result_mu = slerp_mu_centered(a, b, 0.3, mu_zero)
        result_origin = slerp_origin_centered(a, b, 0.3)

        assert torch.allclose(result_mu, result_origin, atol=1e-5), \
            "μ=0 Slerp should match origin-centered Slerp."

    def test_parallel_vectors_fallback_to_lerp(self):
        """When a ≈ b (relative to μ), should fall back to Lerp without NaN."""
        mu = torch.zeros(1, 4, 8, 8)
        a = torch.ones(1, 4, 8, 8)
        b = a + 1e-6  # near-identical

        result = slerp_mu_centered(a, b, 0.5, mu)
        assert not torch.isnan(result).any(), "Parallel vectors produced NaN"
        expected_lerp = (a + b) / 2
        assert torch.allclose(result, expected_lerp, atol=1e-3), \
            "Parallel fallback should approximate Lerp"

    def test_batch_dimension_preserved(self):
        """Output batch dimension should match input."""
        a = torch.randn(4, 4, 8, 8)
        b = torch.randn(4, 4, 8, 8)
        mu = torch.randn(1, 4, 8, 8)  # broadcast mu
        result = slerp_mu_centered(a, b, 0.5, mu)
        assert result.shape == a.shape

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        a = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        b = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        mu = torch.zeros(1, 4, 8, 8, dtype=torch.float16)
        result = slerp_mu_centered(a, b, 0.5, mu)
        assert result.dtype == torch.float16


class TestSlerpPairMuCentered:
    """Tests for the 1D pair slerp (used by WaypointSpline)."""

    def test_endpoint_t0(self):
        a = torch.randn(256)
        b = torch.randn(256)
        mu = torch.randn(256) * 0.1
        result = slerp_pair_mu_centered(a, b, 0.0, mu)
        assert torch.allclose(result, a, atol=1e-4)

    def test_endpoint_t1(self):
        a = torch.randn(256)
        b = torch.randn(256)
        mu = torch.randn(256) * 0.1
        result = slerp_pair_mu_centered(a, b, 1.0, mu)
        assert torch.allclose(result, b, atol=1e-4)

    def test_midpoint_norm_preserved(self):
        torch.manual_seed(99)
        mu = torch.randn(1024) * 0.05
        a = mu + torch.randn(1024)
        b = mu + torch.randn(1024)

        result = slerp_pair_mu_centered(a, b, 0.5, mu)

        r_a = (a - mu).norm().item()
        r_b = (b - mu).norm().item()
        r_mid = (result - mu).norm().item()

        assert r_mid >= min(r_a, r_b) * 0.95
        assert r_mid <= max(r_a, r_b) * 1.05

    def test_origin_wrapper_matches(self):
        torch.manual_seed(77)
        a = torch.randn(512)
        b = torch.randn(512)
        mu_zero = torch.zeros(512)

        r1 = slerp_pair_mu_centered(a, b, 0.4, mu_zero)
        r2 = slerp_pair_origin_centered(a, b, 0.4)
        assert torch.allclose(r1, r2, atol=1e-5)


class TestComputeBatchCentroid:
    """Tests for compute_batch_centroid()."""

    def test_single_sample_returns_channel_spatial_mean(self):
        s = torch.randn(1, 4, 8, 8)
        mu = compute_batch_centroid(s)
        expected = s.float().mean(dim=(-2, -1), keepdim=True)
        assert torch.allclose(mu, expected, atol=1e-6)
        assert mu.shape == (1, 4, 1, 1)

    def test_batch_mean(self):
        s = torch.randn(10, 4, 8, 8)
        mu = compute_batch_centroid(s)
        expected = s.float().mean(dim=0, keepdim=True)
        assert torch.allclose(mu, expected, atol=1e-6)
        assert mu.shape == (1, 4, 8, 8)


# ═══════════════════════════════════════════════════════════════════════
# 2. MANIFOLD EXPLORER INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestManifoldExplorerSlerp:
    """Integration tests for Slerp modes in the Manifold Explorer."""

    def _run_explore(self, mode, mu_override=None, x_strength=1.0, y_strength=1.0):
        node = WayfinderManifold_Explorer()
        torch.manual_seed(42)
        center = _make_latent(seed=0)
        x_vec  = _make_latent(seed=1)
        y_vec  = _make_latent(seed=2)

        kwargs = dict(
            center_latent=center,
            x_vector=x_vec,
            y_vector=y_vec,
            grid_size_x=3,
            grid_size_y=3,
            x_strength=x_strength,
            y_strength=y_strength,
            interpolation_mode=mode,
            normalize_vectors=True,
            clamp_output=False,
            clamp_min=-10.0,
            clamp_max=10.0,
            enable_perf_logging=False,
        )
        if mu_override is not None:
            kwargs["mu_override"] = mu_override

        return node.explore(**kwargs)

    def test_slerp_mode_runs(self):
        """Slerp (μ-centered) mode should run without error."""
        out_latent, meta, report = self._run_explore("Slerp")
        assert out_latent["samples"].shape[0] == 9  # 3x3 grid
        assert meta["interpolation_mode"] == "Slerp"
        assert meta["slerp_anchor"] == "center_distribution"

    def test_slerp_origin_mode_runs(self):
        """Slerp_Origin mode should run without error."""
        out_latent, meta, report = self._run_explore("Slerp_Origin")
        assert out_latent["samples"].shape[0] == 9

    def test_linear_mode_still_works(self):
        """Linear mode should be unaffected by the changes."""
        out_latent, meta, report = self._run_explore("Linear")
        assert out_latent["samples"].shape[0] == 9
        assert meta["slerp_anchor"] is None

    def test_slerp_with_external_mu(self):
        """Slerp with mu_override should report 'mu_override' as anchor."""
        mu = _make_latent(seed=99)
        out_latent, meta, report = self._run_explore("Slerp", mu_override=mu)
        assert meta["slerp_anchor"] == "mu_override"
        assert out_latent["samples"].shape[0] == 9

    def test_slerp_preserves_center_cell_exactly(self):
        """The center cell (offset 0,0) should equal the center latent exactly."""
        out_latent, meta, report = self._run_explore("Slerp")
        grid_map = meta["wayfinder_grid_map"]
        center_cell = [c for c in grid_map if c["is_center"]]
        assert len(center_cell) == 1
        idx = center_cell[0]["batch_start"]
        center_orig = _make_latent(seed=0)["samples"]
        cell = out_latent["samples"][idx:idx+1]
        assert torch.allclose(cell, center_orig, atol=1e-4), \
            "Center cell should match original center_latent"

    def test_slerp_vs_origin_differ_at_offcenter(self):
        """
        μ-centered Slerp (with external centroid) and origin-centered Slerp
        should produce different results for off-center grid cells at intermediate strengths.
        """
        mu = _make_latent(seed=88)
        out_mu, _, _ = self._run_explore("Slerp", mu_override=mu, x_strength=0.5, y_strength=0.5)
        out_origin, _, _ = self._run_explore("Slerp_Origin", x_strength=0.5, y_strength=0.5)

        # Compare first cell (not center)
        mu_cell = out_mu["samples"][0]
        origin_cell = out_origin["samples"][0]

        # They should differ
        diff = (mu_cell - origin_cell).abs().max().item()
        assert diff > 1e-4, \
            f"μ-centered and origin-centered Slerp should differ, but max diff is {diff:.6f}"


# ═══════════════════════════════════════════════════════════════════════
# 3. COMPASS PRO INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestCompassProSlerp:
    """Integration tests for the new Slerp modes in Compass Pro."""

    def test_slerp_mode_runs(self):
        node = WayfinderCompass_Pro()
        base = _make_latent(seed=10)
        target = _make_latent(seed=11)
        origin = _make_latent(seed=12)

        out_latent, meta = node.navigate(
            base_latent=base,
            target_latent=target,
            origin_latent=origin,
            strength=0.5,
            mode="Slerp",
            clamp_output=False,
            clamp_min=-10.0,
            clamp_max=10.0,
            allow_batch_expand=False,
            ortho_per_channel=False,
            clamp_mask_input=False,
            enable_perf_logging=False,
        )
        assert out_latent["samples"].shape == base["samples"].shape
        assert meta["mode"] == "Slerp"

    def test_slerp_origin_mode_runs(self):
        node = WayfinderCompass_Pro()
        base = _make_latent(seed=10)
        target = _make_latent(seed=11)
        origin = _make_latent(seed=12)

        out_latent, meta = node.navigate(
            base_latent=base,
            target_latent=target,
            origin_latent=origin,
            strength=0.5,
            mode="Slerp_Origin",
            clamp_output=False,
            clamp_min=-10.0,
            clamp_max=10.0,
            allow_batch_expand=False,
            ortho_per_channel=False,
            clamp_mask_input=False,
            enable_perf_logging=False,
        )
        assert out_latent["samples"].shape == base["samples"].shape

    def test_slerp_with_mu_centroid(self):
        node = WayfinderCompass_Pro()
        base = _make_latent(seed=10)
        target = _make_latent(seed=11)
        origin = _make_latent(seed=12)
        mu = _make_latent(seed=99)

        out_latent, meta = node.navigate(
            base_latent=base,
            target_latent=target,
            origin_latent=origin,
            strength=0.5,
            mode="Slerp",
            clamp_output=False,
            clamp_min=-10.0,
            clamp_max=10.0,
            allow_batch_expand=False,
            ortho_per_channel=False,
            clamp_mask_input=False,
            enable_perf_logging=False,
            mu_centroid=mu,
        )
        assert out_latent["samples"].shape == base["samples"].shape

    def test_standard_mode_unaffected(self):
        """Existing Standard mode should be completely unaffected."""
        node = WayfinderCompass_Pro()
        base = _make_latent(seed=10)
        target = _make_latent(seed=11)
        origin = _make_latent(seed=12)

        out_latent, meta = node.navigate(
            base_latent=base,
            target_latent=target,
            origin_latent=origin,
            strength=1.0,
            mode="Standard",
            clamp_output=False,
            clamp_min=-10.0,
            clamp_max=10.0,
            allow_batch_expand=False,
            ortho_per_channel=False,
            clamp_mask_input=False,
            enable_perf_logging=False,
        )
        # Standard: result = base + delta * strength
        delta = target["samples"] - origin["samples"]
        expected = base["samples"] + delta * 1.0
        assert torch.allclose(out_latent["samples"], expected, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# 4. WAYPOINT SPLINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestWaypointSplineMuSlerp:
    """Integration tests for μ-centered Slerp in Waypoint Spline."""

    def test_slerp_mode_runs(self):
        node = GimbalWaypointSpline()
        torch.manual_seed(42)
        w1 = torch.randn(1, 4, 8, 8)
        w2 = torch.randn(1, 4, 8, 8)
        w3 = torch.randn(1, 4, 8, 8)
        waypoints = {"samples": torch.cat([w1, w2, w3], dim=0)}

        out_latent, telemetry = node.interpolate_waypoints(
            waypoints_batch=waypoints,
            total_steps=12,
            spline_mode="Spherical_SLERP",
            loop_trajectory=False,
            tension=0.5,
            constant_velocity=True,
        )
        assert out_latent["samples"].shape[0] == 12
        assert telemetry["slerp_anchor"] == "waypoint_mean"

    def test_slerp_with_external_mu_anchor(self):
        node = GimbalWaypointSpline()
        torch.manual_seed(42)
        w1 = torch.randn(1, 4, 8, 8)
        w2 = torch.randn(1, 4, 8, 8)
        waypoints = {"samples": torch.cat([w1, w2], dim=0)}
        mu = _make_latent(seed=77)

        out_latent, telemetry = node.interpolate_waypoints(
            waypoints_batch=waypoints,
            total_steps=8,
            spline_mode="Spherical_SLERP",
            loop_trajectory=False,
            tension=0.5,
            constant_velocity=True,
            mu_anchor=mu,
        )
        assert out_latent["samples"].shape[0] == 8
        assert telemetry["slerp_anchor"] == "external_mu"

    def test_cosine_ease_uses_mu_centered(self):
        node = GimbalWaypointSpline()
        torch.manual_seed(42)
        w1 = torch.randn(1, 4, 8, 8)
        w2 = torch.randn(1, 4, 8, 8)
        waypoints = {"samples": torch.cat([w1, w2], dim=0)}

        out_latent, telemetry = node.interpolate_waypoints(
            waypoints_batch=waypoints,
            total_steps=8,
            spline_mode="Cosine_Ease",
            loop_trajectory=False,
            tension=0.5,
            constant_velocity=True,
        )
        assert out_latent["samples"].shape[0] == 8

    def test_path_norm_preservation(self):
        """
        Along a μ-centered Slerp path, all intermediate points should have
        μ-centered norm bounded by the waypoints' norms.
        """
        torch.manual_seed(42)
        w1 = torch.randn(1, 4, 16, 16)
        w2 = torch.randn(1, 4, 16, 16)
        w3 = torch.randn(1, 4, 16, 16)
        waypoints = {"samples": torch.cat([w1, w2, w3], dim=0)}
        mu = compute_batch_centroid(waypoints["samples"])

        node = GimbalWaypointSpline()
        out_latent, _ = node.interpolate_waypoints(
            waypoints_batch=waypoints,
            total_steps=30,
            spline_mode="Spherical_SLERP",
            loop_trajectory=False,
            tension=0.5,
            constant_velocity=True,
        )

        samples = out_latent["samples"]
        mu_flat = mu.reshape(-1).float()
        r_w1 = (w1.reshape(-1).float() - mu_flat).norm().item()
        r_w2 = (w2.reshape(-1).float() - mu_flat).norm().item()
        r_w3 = (w3.reshape(-1).float() - mu_flat).norm().item()
        r_min = min(r_w1, r_w2, r_w3) * 0.90
        r_max = max(r_w1, r_w2, r_w3) * 1.10

        for i in range(samples.shape[0]):
            r_i = (samples[i].reshape(-1).float() - mu_flat).norm().item()
            assert r_i >= r_min, \
                f"Step {i}: radius {r_i:.2f} below min {r_min:.2f}"
            assert r_i <= r_max, \
                f"Step {i}: radius {r_i:.2f} above max {r_max:.2f}"


# ═══════════════════════════════════════════════════════════════════════
# 5. CIRCULAR ORBIT INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestCircularOrbitMuCentroid:
    """Integration tests for mu_centroid in Circular Orbit."""

    def test_orbit_with_mu_centroid(self):
        node = GimbalCircularOrbit()
        center = _make_latent(seed=50)
        mu = _make_latent(seed=51)

        out_latent, telemetry = node.generate_orbit(
            center_latent=center,
            steps=12,
            radius=1.0,
            orbit_mode="Orthogonal_Basis",
            preserve_hypersphere_norm=True,
            seed=42,
            mu_centroid=mu,
        )
        assert out_latent["samples"].shape == (12, 4, 8, 8)
        assert telemetry["norm_anchor"] == "mu_centroid"

    def test_orbit_without_mu_centroid(self):
        node = GimbalCircularOrbit()
        center = _make_latent(seed=50)

        out_latent, telemetry = node.generate_orbit(
            center_latent=center,
            steps=12,
            radius=1.0,
            orbit_mode="Orthogonal_Basis",
            preserve_hypersphere_norm=True,
            seed=42,
        )
        assert out_latent["samples"].shape == (12, 4, 8, 8)
        assert telemetry["norm_anchor"] == "center_latent"

    def test_orbit_norm_uses_mu_radius(self):
        """
        When mu_centroid is provided and preserve_hypersphere_norm=True,
        orbit points should be projected to the mu norm, not the center norm.
        """
        torch.manual_seed(42)
        center = {"samples": torch.randn(1, 4, 8, 8) * 0.5}
        mu = {"samples": torch.randn(1, 4, 8, 8) * 2.0}

        node = GimbalCircularOrbit()
        out_latent, _ = node.generate_orbit(
            center_latent=center,
            steps=8,
            radius=0.5,
            orbit_mode="Orthogonal_Basis",
            preserve_hypersphere_norm=True,
            seed=42,
            mu_centroid=mu,
        )

        mu_norm = mu["samples"].float().reshape(1, -1).norm().item()
        for i in range(out_latent["samples"].shape[0]):
            sample_norm = out_latent["samples"][i].float().reshape(-1).norm().item()
            # Should be close to mu_norm, not center_norm
            assert abs(sample_norm - mu_norm) / mu_norm < 0.05, \
                f"Step {i}: norm {sample_norm:.2f} should be near mu_norm {mu_norm:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
