import sys
import os
from pathlib import Path
import pytest
import torch

# Ensure nodes directory is in sys.path
nodes_dir = Path(__file__).resolve().parent.parent.parent / "nodes"
sys.path.insert(0, str(nodes_dir))

from gimbal_circular_orbit import GimbalCircularOrbit
from gimbal_waypoint_spline import GimbalWaypointSpline
from gimbal_channel_matrix import GimbalChannelSplit, GimbalChannelMerge, GimbalChannelScale
from gimbal_truncation import GimbalTruncation
from gimbal_vector_analogy import GimbalVectorAnalogy
from gimbal_diagnostics import GimbalDiagnostics
from Wayfinder_compass import WayfinderCompass_Pro
from wayfindermanifold_explorer import WayfinderManifold_Explorer

def test_circular_orbit():
    orbit_node = GimbalCircularOrbit()
    center = {"samples": torch.randn((1, 4, 32, 32))}
    out_latent, telemetry = orbit_node.generate_orbit(
        center_latent=center,
        steps=12,
        radius=1.5,
        orbit_mode="Orthogonal_Basis",
        preserve_hypersphere_norm=True,
        seed=42,
    )
    assert out_latent["samples"].shape == (12, 4, 32, 32)
    assert telemetry["total_steps"] == 12
    assert telemetry["hypersphere_norm_preserved"] is True
    print("test_circular_orbit passed!")

def test_waypoint_spline():
    spline_node = GimbalWaypointSpline()
    w1 = torch.randn((1, 4, 32, 32))
    w2 = torch.randn((1, 4, 32, 32))
    w3 = torch.randn((1, 4, 32, 32))
    waypoints = {"samples": torch.cat([w1, w2, w3], dim=0)}
    
    out_latent, telemetry = spline_node.interpolate_waypoints(
        waypoints_batch=waypoints,
        total_steps=24,
        spline_mode="Spherical_SLERP",
        loop_trajectory=False,
        tension=0.5,
        constant_velocity=True,
    )
    assert out_latent["samples"].shape[0] == 24
    assert out_latent["samples"].shape[1:] == (4, 32, 32)
    assert telemetry["num_waypoints"] == 3
    print("test_waypoint_spline passed!")

def test_channel_split_and_merge():
    split_node = GimbalChannelSplit()
    merge_node = GimbalChannelMerge()
    scale_node = GimbalChannelScale()
    
    # 16-channel FLUX tensor
    flux_latent = {"samples": torch.randn((1, 16, 16, 16))}
    band_A, band_B, tel_split = split_node.split_channels(flux_latent, split_index=8)
    assert band_A["samples"].shape == (1, 8, 16, 16)
    assert band_B["samples"].shape == (1, 8, 16, 16)
    
    merged, tel_merge = merge_node.merge_channels(band_A, band_B)
    assert merged["samples"].shape == (1, 16, 16, 16)
    
    scaled, tel_scale = scale_node.scale_channels(flux_latent, 1.2, 0.8, 1.0, 1.0, remaining_ch_gain=0.5)
    assert scaled["samples"].shape == (1, 16, 16, 16)
    print("test_channel_split_and_merge passed!")

def test_truncation():
    trunc_node = GimbalTruncation()
    noisy_latent = {"samples": torch.randn((1, 4, 32, 32)) * 3.0}
    stabilized, telemetry = trunc_node.apply_truncation(noisy_latent, truncation_psi=0.5, channel_adaptive=True)
    assert stabilized["samples"].shape == (1, 4, 32, 32)
    assert telemetry["truncated_variance"] < telemetry["initial_variance"]
    print("test_truncation passed!")

def test_vector_analogy():
    analogy_node = GimbalVectorAnalogy()
    cA = {"samples": torch.randn((1, 4, 32, 32))}
    cB = {"samples": torch.randn((1, 4, 32, 32))}
    cC = {"samples": torch.randn((1, 4, 32, 32))}
    
    res, delta, tel = analogy_node.calculate_analogy(cA, cB, cC, strength=1.0, ortho_project=True, preserve_norm=True)
    assert res["samples"].shape == (1, 4, 32, 32)
    assert delta["samples"].shape == (1, 4, 32, 32)
    print("test_vector_analogy passed!")

def test_diagnostics():
    diag_node = GimbalDiagnostics()
    latent = {"samples": torch.randn((2, 4, 32, 32))}
    mean_val, std_val, min_val, max_val, l2_val, B, C, H, W, report = diag_node.inspect_latent(latent)
    assert B == 2
    assert C == 4
    assert H == 32
    assert W == 32
    assert "GIMBAL LATENT FLIGHT TELEMETRY" in report
    print("test_diagnostics passed!")

if __name__ == "__main__":
    test_circular_orbit()
    test_waypoint_spline()
    test_channel_split_and_merge()
    test_truncation()
    test_vector_analogy()
    test_diagnostics()
    print("\nALL GIMBAL SUITE TESTS PASSED PERFECTLY!")
