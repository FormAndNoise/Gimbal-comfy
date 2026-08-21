import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch

log = logging.getLogger(__name__)

MAX_CACHE_ENTRIES: int = 8
_pca_cache: OrderedDict[str, "PCAResult"] = OrderedDict()
_DEFAULT_N_COMPONENTS = 10


class PCAResult:
    """
    [VibeCheck Badge: 🟢 Stabilized]
    Immutable container for PCA decomposition results.
    """
    __slots__ = ("components", "singular_values", "mean", "spatial_shape", "explained_variance_ratio")
    
    def __init__(
        self,
        components: torch.Tensor,
        singular_values: torch.Tensor,
        mean: torch.Tensor,
        spatial_shape: Tuple[int, int, int],
    ):
        self.components = components
        self.singular_values = singular_values
        self.mean = mean
        self.spatial_shape = spatial_shape
        
        var = singular_values ** 2
        total = var.sum().clamp(min=1e-12)
        self.explained_variance_ratio = var / total


def _batch_hash(samples: torch.Tensor) -> str:
    """Hash based on float16 content for speed with acceptable collision risk."""
    data = samples.detach().cpu().to(torch.float16).numpy().tobytes()
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _cache_get(key: str) -> Optional[PCAResult]:
    if key in _pca_cache:
        _pca_cache.move_to_end(key)
        return _pca_cache[key]
    return None


def _cache_put(key: str, result: PCAResult) -> None:
    if key in _pca_cache:
        _pca_cache.move_to_end(key)
    else:
        if len(_pca_cache) >= MAX_CACHE_ENTRIES:
            evicted_key, _ = _pca_cache.popitem(last=False)
            log.debug(f"PCA cache evicted: {evicted_key[:8]}")
        _pca_cache[key] = result


def _validate_latent(d: Any, label: str) -> torch.Tensor:
    if not isinstance(d, dict):
        raise ValueError(f"'{label}' must be LATENT dict.")
    samples = d.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise ValueError(f"'{label}['samples']' must be torch.Tensor.")
    if samples.ndim != 4:
        raise ValueError(f"'{label}' must be 4-D [B, C, H, W], got {tuple(samples.shape)}.")
    return samples


def _run_pca(flat: torch.Tensor, n_components: int) -> PCAResult:
    """
    Compute PCA using randomized SVD via torch.pca_lowrank.
    Runs on CPU to avoid GPU memory pressure and for better float32 precision.
    """
    B, D = flat.shape
    n_components = min(n_components, B, D)
    
    mean = flat.mean(dim=0)
    centered = flat - mean
    
    # q=n_components for exact computation, niter for stability
    U, S, V = torch.pca_lowrank(centered, q=n_components, center=False, niter=4)
    
    # V: [D, n_components], columns are eigenvectors
    components = V.T.contiguous()  # [n_components, D]
    
    return PCAResult(components, S, mean, spatial_shape=None)  # shape set by caller


class GimbalSemanticSlider:
    """
    [VibeCheck Badge: 🟢 Stabilized]
    
    PCA-based latent feature controller. Isolates variance directions
    for independent semantic control.
    """

    CATEGORY = "Gimbal/Flight Instruments"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("modified_latent", "pc_preview")
    FUNCTION = "apply_slider"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent_batch": ("LATENT",),
                "base_latent": ("LATENT",),
                "pc_index": ("INT", {"default": 1, "min": 1, "max": _DEFAULT_N_COMPONENTS, "step": 1}),
                "slider_value": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05, "display": "slider"}),
                "orthogonalize": ("BOOLEAN", {"default": False}),
            },
        }

    def apply_slider(
        self,
        latent_batch: Dict[str, Any],
        base_latent: Dict[str, Any],
        pc_index: int,
        slider_value: float,
        orthogonalize: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        t_total = time.perf_counter()

        # CRITICAL: Ensure no gradient tracking through PCA manipulation
        with torch.no_grad():
            batch_samples = _validate_latent(latent_batch, "latent_batch")
            base_samples  = _validate_latent(base_latent, "base_latent")
            
            B_batch = batch_samples.shape[0]
            if B_batch < 2:
                raise ValueError(f"latent_batch must have >= 2 samples for PCA, got {B_batch}.")
            
            pc_idx_0 = pc_index - 1  # Convert to 0-based
            effective_n = min(_DEFAULT_N_COMPONENTS, B_batch)
            
            if pc_idx_0 >= effective_n:
                raise ValueError(
                    f"pc_index {pc_index} exceeds the number of computable components "
                    f"({effective_n}) for a batch of size {B_batch}. "
                    f"Each component requires one batch sample. "
                    f"Add more samples to your latent_batch or reduce pc_index."
                )
            
            # PCA computation (cached)
            key = _batch_hash(batch_samples)
            cached = _cache_get(key)
            
            if cached is None:
                log.info(f"Computing PCA for batch {tuple(batch_samples.shape)}...")
                t_pca = time.perf_counter()
                
                # Move to CPU for SVD stability
                flat_batch = batch_samples.detach().cpu().float().reshape(B_batch, -1)
                cached = _run_pca(flat_batch, effective_n)
                cached.spatial_shape = (
                    batch_samples.shape[1],
                    batch_samples.shape[2],
                    batch_samples.shape[3],
                )
                
                _cache_put(key, cached)
                log.info(f"PCA computed in {(time.perf_counter() - t_pca)*1000:.1f}ms")
            else:
                log.debug(f"PCA cache hit: {key[:8]}")
            
            # Extract PC vector
            pc_vector = cached.components[pc_idx_0].clone()
            
            # Optional Gram-Schmidt re-orthogonalization
            if orthogonalize:
                for i, other in enumerate(cached.components):
                    if i == pc_idx_0:
                        continue
                    # Subtract projection
                    proj = torch.dot(pc_vector, other) * other
                    pc_vector -= proj
                norm = pc_vector.norm()
                if norm < 1e-10:
                    log.warning(f"PC {pc_index} collapsed during orthogonalization.")
                else:
                    pc_vector = pc_vector / norm
            
            # Apply displacement to base_latent
            B_base, C, H, W = base_samples.shape
            D = C * H * W
            
            # Flatten base and move to CPU for arithmetic (matching PC device)
            base_flat = base_samples.detach().cpu().float().reshape(B_base, D)

            # Delta is slider * PC direction
            import math
            std_dev = cached.singular_values[pc_idx_0] / max(1.0, math.sqrt(B_batch - 1))
            delta_flat = slider_value * std_dev * pc_vector.unsqueeze(0)  # [1, D]
            modified_flat = base_flat + delta_flat  # Broadcasting over batch            
            # Reshape and restore device/dtype
            modified_4d = modified_flat.reshape(B_base, C, H, W)
            modified_4d = modified_4d.to(device=base_samples.device, dtype=base_samples.dtype)
            modified_latent = {"samples": modified_4d}
            
            # PC preview (unit vector reshaped)
            pc_4d = pc_vector.reshape(1, C, H, W).to(device=base_samples.device, dtype=base_samples.dtype)
            pc_preview = {"samples": pc_4d}
            
            log.debug(
                f"apply_slider: PC{pc_index} (var={cached.explained_variance_ratio[pc_idx_0]:.2%}), "
                f"slider={slider_value:.3f}, time={(time.perf_counter()-t_total)*1000:.2f}ms"
            )
            
            return modified_latent, pc_preview


# Backward compatibility aliases
Wayfinder_SemanticSlider = GimbalSemanticSlider
Gimbal_SemanticSlider = GimbalSemanticSlider

NODE_CLASS_MAPPINGS = {
    "GimbalSemanticSlider": GimbalSemanticSlider,
    "Gimbal_SemanticSlider": GimbalSemanticSlider,
    "Wayfinder_SemanticSlider": Wayfinder_SemanticSlider,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GimbalSemanticSlider": "🎚️ Gimbal Semantic Slider (PCA)",
    "Gimbal_SemanticSlider": "🎚️ Gimbal Semantic Slider (PCA)",
    "Wayfinder_SemanticSlider": "🎚️ Gimbal Semantic Slider (PCA)",
}