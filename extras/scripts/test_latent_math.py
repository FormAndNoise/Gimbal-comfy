import sys
import os
import torch

# Add Wayfinder root to path so we can import the nodes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wayfindermanifold_explorer import WayfinderManifold_Explorer
from Wayfinder_compass import WayfinderCompass_Pro

def print_stats(name, t):
    if torch.isnan(t).any():
        print(f"[{name}] HAS NaNs!")
    else:
        print(f"[{name}] min: {t.min().item():.3f}, max: {t.max().item():.3f}, mean: {t.mean().item():.3f}, std: {t.std().item():.3f}")

def run_tests():
    torch.manual_seed(42)
    B, C, H, W = 1, 4, 32, 32
    
    # Create mock latents
    base_tensor = torch.randn(B, C, H, W)
    target_tensor = torch.randn(B, C, H, W) * 2.0 + 1.0  # different variance
    origin_tensor = torch.randn(B, C, H, W)
    
    # Also test identical vectors and parallel vectors
    parallel_tensor = base_tensor * 2.5
    opposing_tensor = base_tensor * -1.5
    
    print("=== MOCK TENSOR STATS ===")
    print_stats("Base", base_tensor)
    print_stats("Target", target_tensor)
    
    print("\n=== TESTING WAYFINDER COMPASS ===")
    compass = WayfinderCompass_Pro()
    
    base_latent = {"samples": base_tensor.clone()}
    target_latent = {"samples": target_tensor.clone()}
    origin_latent = {"samples": origin_tensor.clone()}
    
    # Test 1: Normalized Mode
    print("\n--- Compass: Normalized Mode ---")
    out, meta = compass.navigate(
        base_latent, target_latent, origin_latent,
        strength=1.0, mode="Normalized", clamp_output=False,
        clamp_min=-10.0, clamp_max=10.0, allow_batch_expand=False,
        ortho_per_channel=False, clamp_mask_input=False, enable_perf_logging=False
    )
    print_stats("Normalized Out", out["samples"])
    
    # Test 2: Orthogonal Mode
    print("\n--- Compass: Orthogonal_Projection ---")
    out, meta = compass.navigate(
        base_latent, target_latent, origin_latent,
        strength=1.0, mode="Orthogonal_Projection", clamp_output=False,
        clamp_min=-10.0, clamp_max=10.0, allow_batch_expand=False,
        ortho_per_channel=False, clamp_mask_input=False, enable_perf_logging=False
    )
    print_stats("Orthogonal Out (global)", out["samples"])
    
    # Test 3: Orthogonal Mode (Per Channel)
    out, meta = compass.navigate(
        base_latent, target_latent, origin_latent,
        strength=1.0, mode="Orthogonal_Projection", clamp_output=False,
        clamp_min=-10.0, clamp_max=10.0, allow_batch_expand=False,
        ortho_per_channel=True, clamp_mask_input=False, enable_perf_logging=False
    )
    print_stats("Orthogonal Out (per-channel)", out["samples"])
    
    # Test 3b: Blend_Overlay
    out, meta = compass.navigate(
        base_latent, target_latent, origin_latent,
        strength=0.5, mode="Blend_Overlay", clamp_output=False,
        clamp_min=-10.0, clamp_max=10.0, allow_batch_expand=False,
        ortho_per_channel=False, clamp_mask_input=False, enable_perf_logging=False
    )
    print_stats("Blend_Overlay Out", out["samples"])


    print("\n=== TESTING MANIFOLD EXPLORER ===")
    manifold = WayfinderManifold_Explorer()
    
    # Test 4: Slerp with normal vectors
    print("\n--- Manifold: Slerp ---")
    out, meta, report = manifold.explore(
        center_latent=base_latent,
        x_vector={"samples": target_tensor.clone()},
        y_vector={"samples": origin_tensor.clone()},
        grid_size_x=3, grid_size_y=3,
        x_strength=1.0, y_strength=1.0,
        interpolation_mode="Slerp",
        normalize_vectors=False, clamp_output=False,
        clamp_min=-10.0, clamp_max=10.0, enable_perf_logging=False
    )
    print_stats("Manifold Slerp Out", out["samples"])
    
    # Test 5: Slerp with Parallel Vectors (Should trigger parallel mask)
    print("\n--- Manifold: Slerp (Parallel X Vector) ---")
    out, meta, report = manifold.explore(
        center_latent=base_latent,
        x_vector={"samples": parallel_tensor.clone()},
        y_vector={"samples": origin_tensor.clone()},
        grid_size_x=3, grid_size_y=3,
        x_strength=1.0, y_strength=1.0,
        interpolation_mode="Slerp",
        normalize_vectors=False, clamp_output=False,
        clamp_min=-10.0, clamp_max=10.0, enable_perf_logging=False
    )
    print_stats("Manifold Slerp (Parallel) Out", out["samples"])
    
    # Test 6: Slerp with Opposing Vectors
    print("\n--- Manifold: Slerp (Opposing X Vector) ---")
    out, meta, report = manifold.explore(
        center_latent=base_latent,
        x_vector={"samples": opposing_tensor.clone()},
        y_vector={"samples": origin_tensor.clone()},
        grid_size_x=3, grid_size_y=3,
        x_strength=1.0, y_strength=1.0,
        interpolation_mode="Slerp",
        normalize_vectors=False, clamp_output=False,
        clamp_min=-10.0, clamp_max=10.0, enable_perf_logging=False
    )
    print_stats("Manifold Slerp (Opposing) Out", out["samples"])

if __name__ == "__main__":
    run_tests()
