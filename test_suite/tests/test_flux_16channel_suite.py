"""
FLUX.1 (16-Channel) Tensor Test Suite for Gimbal Latent Flight Instruments.
Verifies that all nodes in the suite flawlessly process, manipulate, stabilize,
and serialize 16-channel FLUX latent tensors [B, 16, H, W].
"""

import os
import pytest
import torch

from nodes.gimbal_compass import GimbalCompass_Pro
from nodes.gimbal_manifold_explorer import GimbalManifold_Explorer
from nodes.gimbal_circular_orbit import GimbalCircularOrbit
from nodes.gimbal_waypoint_spline import GimbalWaypointSpline
from nodes.gimbal_semanticslider import GimbalSemanticSlider
from nodes.gimbal_crossmodal_bridge import GimbalCrossModalBridge
from nodes.gimbal_channel_matrix import GimbalChannelSplit, GimbalChannelMerge, GimbalChannelScale
from nodes.gimbal_truncation import GimbalTruncation
from nodes.gimbal_vector_analogy import GimbalVectorAnalogy
from nodes.gimbal_gps_anchor import GimbalGPS_Anchor
from nodes.gimbal_diagnostics import GimbalDiagnostics
from nodes.gimbal_latent_stabilizer import GimbalLatentStabilizer
from nodes.gimbal_latent_math import apply_new_latent_math
from nodes.gimbal_latent_telemetry import GimbalLatentTelemetry


@pytest.fixture
def flux_latent_b1():
    """Generates a 1-batch 16-channel FLUX latent tensor [1, 16, 64, 64]."""
    torch.manual_seed(42)
    return {"samples": torch.randn(1, 16, 64, 64, dtype=torch.float32)}


@pytest.fixture
def flux_latent_b4():
    """Generates a 4-batch 16-channel FLUX latent tensor [4, 16, 64, 64]."""
    torch.manual_seed(101)
    return {"samples": torch.randn(4, 16, 64, 64, dtype=torch.float32)}


# ---------------------------------------------------------------------------
# 1. GIMBAL COMPASS PRO (16-CH FLUX SLERP, NORMALIZED, ORTHOGONAL)
# ---------------------------------------------------------------------------
def test_flux_compass_modes(flux_latent_b1):
    compass = GimbalCompass_Pro()
    target = {"samples": torch.randn(1, 16, 64, 64, dtype=torch.float32)}
    origin = {"samples": torch.randn(1, 16, 64, 64, dtype=torch.float32)}

    for mode in ["Standard", "Normalized", "Orthogonal_Projection", "Slerp", "Slerp_Origin", "Blend_Overlay", "Blend_Multiply", "Stochastic_Sample"]:
        out, telem = compass.navigate(
            base_latent=flux_latent_b1,
            target_latent=target,
            origin_latent=origin,
            strength=0.75,
            mode=mode,
            clamp_output=True,
            clamp_min=-8.0,
            clamp_max=8.0,
            allow_batch_expand=False,
            ortho_per_channel=True,
            clamp_mask_input=False,
            enable_perf_logging=False,
        )
        assert out["samples"].shape == (1, 16, 64, 64)
        assert not torch.isnan(out["samples"]).any()
        assert not torch.isinf(out["samples"]).any()


# ---------------------------------------------------------------------------
# 2. GIMBAL MANIFOLD EXPLORER (16-CH FLUX 2D TOPOLOGY)
# ---------------------------------------------------------------------------
def test_flux_manifold_explorer(flux_latent_b1):
    explorer = GimbalManifold_Explorer()
    vec_x = {"samples": torch.randn(1, 16, 64, 64, dtype=torch.float32)}
    vec_y = {"samples": torch.randn(1, 16, 64, 64, dtype=torch.float32)}

    out, meta, report = explorer.explore(
        center_latent=flux_latent_b1,
        x_vector=vec_x,
        y_vector=vec_y,
        grid_size_x=3,
        grid_size_y=3,
        x_strength=1.5,
        y_strength=1.5,
        interpolation_mode="Slerp",
        normalize_vectors=True,
        clamp_output=True,
        clamp_min=-8.0,
        clamp_max=8.0,
        enable_perf_logging=False,
    )
    assert out["samples"].shape == (9, 16, 64, 64)
    assert not torch.isnan(out["samples"]).any()


# ---------------------------------------------------------------------------
# 3. GIMBAL CIRCULAR ORBIT (16-CH FLUX GEODESIC LOOP)
# ---------------------------------------------------------------------------
def test_flux_circular_orbit(flux_latent_b1):
    orbiter = GimbalCircularOrbit()
    out, telem = orbiter.generate_orbit(
        steps=8,
        radius=1.2,
        orbit_mode="Orthogonal_Basis",
        preserve_hypersphere_norm=True,
        seed=42,
        center_latent=flux_latent_b1,
    )
    assert out["samples"].shape == (8, 16, 64, 64)
    assert telem["total_steps"] == 8
    assert not torch.isnan(out["samples"]).any()


# ---------------------------------------------------------------------------
# 4. GIMBAL WAYPOINT SPLINE (16-CH FLUX CATMULL-ROM SPHERICAL PATH)
# ---------------------------------------------------------------------------
def test_flux_waypoint_spline(flux_latent_b4):
    spline = GimbalWaypointSpline()
    out, telem = spline.interpolate_waypoints(
        waypoints_batch=flux_latent_b4,
        total_steps=16,
        spline_mode="Spherical_SLERP",
        loop_trajectory=True,
        tension=0.5,
        constant_velocity=True,
    )
    assert out["samples"].shape == (16, 16, 64, 64)
    assert not torch.isnan(out["samples"]).any()


# ---------------------------------------------------------------------------
# 5. GIMBAL SEMANTIC SLIDER (16-CH FLUX SVD/PCA DECOMPOSITION)
# ---------------------------------------------------------------------------
def test_flux_semantic_slider(flux_latent_b4, flux_latent_b1):
    slider = GimbalSemanticSlider()
    out, preview = slider.apply_slider(
        latent_batch=flux_latent_b4,
        base_latent=flux_latent_b1,
        pc_index=1,
        slider_value=1.5,
        orthogonalize=True,
    )
    assert out["samples"].shape == (1, 16, 64, 64)
    assert not torch.isnan(out["samples"]).any()


# ---------------------------------------------------------------------------
# 6. GIMBAL CROSS-MODAL BRIDGE (16-CH FLUX PADDED SIGNATURES)
# ---------------------------------------------------------------------------
def test_flux_crossmodal_bridge(flux_latent_b1):
    bridge = GimbalCrossModalBridge()
    target, origin = bridge.translate(
        llm_instruction="cool sharp crisp monochrome bright cold",
        base_latent=flux_latent_b1,
        mapping_mode="Keyword_Heuristics",
    )
    assert target["samples"].shape == (1, 16, 64, 64)
    assert not torch.isnan(target["samples"]).any()


# ---------------------------------------------------------------------------
# 7. GIMBAL CHANNEL MATRIX (16-CH FLUX SPLIT, MERGE, SCALE)
# ---------------------------------------------------------------------------
def test_flux_channel_matrix(flux_latent_b1):
    splitter = GimbalChannelSplit()
    merger = GimbalChannelMerge()
    scaler = GimbalChannelScale()

    # Split 16 channels at index 8 -> [1, 8, 64, 64] and [1, 8, 64, 64]
    band_a, band_b, tel_split = splitter.split_channels(latent=flux_latent_b1, split_index=8)
    assert band_a["samples"].shape == (1, 8, 64, 64)
    assert band_b["samples"].shape == (1, 8, 64, 64)

    # Scale band B
    scaled_b, tel_scale = scaler.scale_channels(
        latent=band_b,
        ch0_gain=1.0, ch1_gain=1.0, ch2_gain=1.0, ch3_gain=1.0,
        remaining_ch_gain=1.5
    )

    # Merge back into 16-channel latent
    merged, tel_merge = merger.merge_channels(latent_band_A=band_a, latent_band_B=scaled_b)
    assert merged["samples"].shape == (1, 16, 64, 64)


# ---------------------------------------------------------------------------
# 8. GIMBAL TRUNCATION & LATENT STABILIZER (16-CH FLUX)
# ---------------------------------------------------------------------------
def test_flux_truncation_and_stabilizer(flux_latent_b1):
    trunc = GimbalTruncation()
    out_trunc, tel = trunc.apply_truncation(latent=flux_latent_b1, truncation_psi=0.85, channel_adaptive=True)
    assert out_trunc["samples"].shape == (1, 16, 64, 64)

    stab = GimbalLatentStabilizer()
    out_stab = stab.stabilize(
        latent=flux_latent_b1,
        truncation_psi=0.88,
        subspace_rank=8,
        scale_cap=8.0,
        jitter_strength=0.01,
    )[0]
    assert out_stab["samples"].shape == (1, 16, 64, 64)
    assert not torch.isnan(out_stab["samples"]).any()


# ---------------------------------------------------------------------------
# 9. GIMBAL GPS ANCHOR (16-CH FLUX PERSISTENCE & STATS)
# ---------------------------------------------------------------------------
def test_flux_gps_anchor_persistence(flux_latent_b1):
    anchor = GimbalGPS_Anchor()

    out_latent, meta, report = anchor.anchor(
        latent_batch=flux_latent_b1,
        select_index=0,
        waypoint_name="flux_unit_test",
        save_waypoint=False,
        enable_perf_logging=False,
    )
    assert out_latent["samples"].shape == (1, 16, 64, 64)
    assert meta["waypoint_name"] == "flux_unit_test"
    assert meta["latent_shape"] == [1, 16, 64, 64]
    assert len(meta["statistics"]["per_channel"]) == 16


# ---------------------------------------------------------------------------
# 10. GIMBAL DIAGNOSTICS & TELEMETRY (16-CH FLUX)
# ---------------------------------------------------------------------------
def test_flux_diagnostics_and_telemetry(flux_latent_b1):
    diag = GimbalDiagnostics()
    telem_node = GimbalLatentTelemetry()

    mean_val, std_val, min_val, max_val, l2_val, B, C, H, W, report = diag.inspect_latent(latent=flux_latent_b1)
    assert B == 1
    assert C == 16
    assert H == 64
    assert W == 64

    lat_out, m_ll, m_mah, m_tc, m_geo, tel_dict = telem_node.diagnose(latent=flux_latent_b1)
    assert lat_out["samples"].shape == (1, 16, 64, 64)
    assert isinstance(m_ll, float)


# ---------------------------------------------------------------------------
# 11. GIMBAL LATENT MATH DISPATCHER (16-CH FLUX E1-E12 EQUATIONS)
# ---------------------------------------------------------------------------
def test_flux_latent_math_dispatcher(flux_latent_b1):
    z = flux_latent_b1["samples"]
    
    # E3: Truncation
    z_trunc = apply_new_latent_math(z, "truncation", 0.8)
    assert z_trunc.shape == (1, 16, 64, 64)

    # E4: Slerp_mu
    z_slerp = apply_new_latent_math(z, "slerp_mu", torch.randn_like(z), 0.5)
    assert z_slerp.shape == (1, 16, 64, 64)

    # E5: Geodesic Distance
    d_geo = apply_new_latent_math(z, "geodesic", z)
    assert d_geo.item() == pytest.approx(0.0, abs=1e-4)

    # E11: Bounded Scale
    z_scale = apply_new_latent_math(z, "bounded_scale", 5.0)
    assert z_scale.shape == (1, 16, 64, 64)

    # E12: Dequantization Jitter
    z_jitter = apply_new_latent_math(z, "dequantize", 0.05)
    assert z_jitter.shape == (1, 16, 64, 64)
