import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

log = logging.getLogger(__name__)

LATENT_SIGNATURES: Dict[str, List[float]] = {
    "bright":       [ 0.25,  0.05,  0.05,  0.00],
    "dark":         [-0.25, -0.05, -0.05,  0.00],
    "overexposed":  [ 0.50,  0.10,  0.10,  0.00],
    "underexposed": [-0.40, -0.08, -0.08,  0.00],
    "contrast":     [ 0.15,  0.00,  0.00,  0.15],
    "flat":         [-0.10,  0.00,  0.00, -0.10],
    "punchy":       [ 0.20,  0.05,  0.05,  0.20],
    "sharp":        [ 0.00,  0.00,  0.00,  0.35],
    "soft":         [ 0.00,  0.00,  0.00, -0.30],
    "blurry":       [ 0.00,  0.00,  0.00, -0.45],
    "crisp":        [ 0.05,  0.00,  0.00,  0.30],
    "saturated":    [ 0.05,  0.25,  0.20,  0.05],
    "vivid":        [ 0.08,  0.30,  0.25,  0.05],
    "desaturated":  [-0.05, -0.25, -0.20, -0.05],
    "muted":        [-0.05, -0.18, -0.15, -0.05],
    "monochrome":   [ 0.00, -0.40, -0.40,  0.00],
    "warm":         [ 0.05,  0.20, -0.10,  0.00],
    "cool":         [ 0.00, -0.20,  0.10,  0.00],
    "golden":       [ 0.10,  0.25, -0.15,  0.05],
    "cold":         [-0.05, -0.25,  0.15,  0.00],
    "dreamy":       [ 0.10, -0.05, -0.05, -0.20],
    "gritty":       [-0.05,  0.00,  0.05,  0.25],
    "cinematic":    [-0.10,  0.05, -0.05,  0.20],
    "faded":        [-0.15, -0.10, -0.10, -0.15],
    "neon":         [ 0.05,  0.35,  0.30,  0.10],
    "pastel":       [ 0.20, -0.15, -0.10, -0.20],
    "moody":        [-0.20,  0.05,  0.00,  0.15],
    "ethereal":     [ 0.15, -0.10, -0.05, -0.25],
    "underwater":   [-0.15, -0.10,  0.25, -0.05],
    "fire":         [ 0.20,  0.30, -0.15,  0.10],
}

KEYWORD_ALIASES: Dict[str, str] = {
    "bright": "bright", "brighter": "bright", "brighten": "bright",
    "darker": "dark", "darken": "dark",
    "sharpened": "sharp", "sharpen": "sharp",
    "soften": "soft", "softer": "soft",
    "saturate": "saturated", "desaturate": "desaturated",
    "warmth": "warm", "warmer": "warm",
    "cooler": "cool", "cooling": "cool",
    "hazy": "dreamy", "foggy": "dreamy",
    "ocean": "underwater", "aquatic": "underwater",
    "flame": "fire", "burning": "fire", "hot": "fire",
}

SEMANTIC_CHANNEL_MAP: Dict[str, int] = {
    "luminance": 0, "brightness": 0,
    "warm_cool": 1, "temperature": 1,
    "green_mag": 2, "saturation": 2,
    "detail": 3, "sharpness": 3,
}


def _coerce(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Cast tensor to match device and dtype of reference."""
    return tensor.to(device=reference.device, dtype=reference.dtype)


def _validate_latent(d: Any, label: str) -> torch.Tensor:
    if not isinstance(d, dict):
        raise ValueError(f"'{label}' must be LATENT dict, got {type(d).__name__}.")
    samples = d.get("samples")
    if samples is None:
        raise ValueError(f"'{label}' has no 'samples' key.")
    if not isinstance(samples, torch.Tensor):
        raise ValueError(f"'{label}['samples']' must be torch.Tensor.")
    if samples.ndim != 4:
        raise ValueError(f"'{label}['samples']' must be 4-D [B, C, H, W], got {tuple(samples.shape)}.")
    return samples


def _resolve_keyword(word: str) -> Optional[str]:
    word = word.lower()
    return KEYWORD_ALIASES.get(word) if word in KEYWORD_ALIASES else (word if word in LATENT_SIGNATURES else None)


def _resolve_channel_index(key: str, channel_count: int) -> Optional[int]:
    key_lower = key.lower()
    if key_lower in SEMANTIC_CHANNEL_MAP:
        idx = SEMANTIC_CHANNEL_MAP[key_lower]
        return idx if idx < channel_count else None
    for prefix in ("channel_", "ch_"):
        if key_lower.startswith(prefix):
            suffix = key_lower[len(prefix):]
            if suffix.isdigit():
                idx = int(suffix)
                if 0 <= idx < channel_count:
                    return idx
                log.warning(f"Channel index {idx} out of range [0, {channel_count})")
                return None
    return None


class Wayfinder_CrossModalBridge:
    """
    [VibeCheck Badge: 🟢 Stabilized]
    Translates text instructions into latent vector perturbations.
    What's left:
    - Fine-tune parameter ranges for edge cases.
    """

    CATEGORY = "Wayfinder/Latent"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("target_vector", "origin_vector")
    FUNCTION = "translate"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "llm_instruction": ("STRING", {"default": "", "multiline": True}),
                "base_latent": ("LATENT",),
                "mapping_mode": (["Keyword_Heuristics", "Embedding_Projection", "Manual_JSON"], {"default": "Keyword_Heuristics"}),
            },
            "optional": {
                "conditioning": ("CONDITIONING",),
            }
        }

    def translate(
        self,
        llm_instruction: str,
        base_latent: Dict[str, Any],
        mapping_mode: str,
        conditioning=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        # CRITICAL: No gradients needed for inference
        with torch.no_grad():
            samples = _validate_latent(base_latent, "base_latent")
            
            # Clone dict to preserve metadata like noise_mask
            origin_vector = base_latent.copy()
            origin_vector["samples"] = samples.clone()

            dispatch = {
                "Keyword_Heuristics":   self._keyword_heuristics,
                "Embedding_Projection": self._embedding_projection,
                "Manual_JSON":          self._manual_json,
            }

            if mapping_mode not in dispatch:
                raise ValueError(f"Unknown mapping_mode '{mapping_mode}'. Valid: {list(dispatch.keys())}")

            if mapping_mode == "Embedding_Projection":
                delta = self._embedding_projection(llm_instruction, samples, conditioning)
            else:
                delta = dispatch[mapping_mode](llm_instruction, samples)
            
            # Ensure dtype/device match to prevent runtime crashes
            delta = _coerce(delta, samples)
            
            target_vector = base_latent.copy()
            target_vector["samples"] = samples + delta

        return target_vector, origin_vector

    def _keyword_heuristics(self, instruction: str, samples: torch.Tensor) -> torch.Tensor:
        import re
        tokens = re.split(r"[\s,.\-!?;:]+", instruction.lower())
        channel_offsets = torch.zeros(samples.shape[1], dtype=torch.float32)
        matched: List[str] = []

        for token in tokens:
            if not token:
                continue
            key = _resolve_keyword(token)
            if key:
                sig = LATENT_SIGNATURES[key]
                # Pad signature to match actual channel count (e.g. 16 for FLUX)
                sig_tensor = torch.tensor(sig, dtype=torch.float32)
                if samples.shape[1] > len(sig):
                    padded_sig = torch.zeros(samples.shape[1], dtype=torch.float32)
                    padded_sig[:len(sig)] = sig_tensor
                    channel_offsets += padded_sig
                else:
                    channel_offsets += sig_tensor[:samples.shape[1]]
                matched.append(key)

        if not matched:
            log.warning(f"Keyword_Heuristics: no keywords found in: {instruction[:120]}")

        B, C, H, W = samples.shape
        # Ensure contiguous memory layout for expansion
        delta = channel_offsets.view(1, C, 1, 1).expand(B, -1, H, W).contiguous()
        return delta

    def _embedding_projection(self, instruction: str, samples: torch.Tensor, conditioning=None) -> torch.Tensor:
        """
        Projects LLM/CLIP hidden-state into latent channel dimensions using a learned projection module.
        """
        B, C, H, W = samples.shape
        
        if conditioning is not None and len(conditioning) > 0:
            cond_data = conditioning[0]
            cond_dict = cond_data[1] if len(cond_data) > 1 else {}
            pooled = cond_dict.get("pooled_output")
            
            if pooled is not None:
                log.info(f"Embedding_Projection: projecting {pooled.shape[-1]}-dim CLIP pooled output to {C} latent channels.")
                pooled = pooled.to(device=samples.device, dtype=torch.float32)
                
                # Initialize a proper projection layer
                import os
                from pathlib import Path
                
                in_dim = pooled.shape[-1]
                
                # Define a small MLP for the projection mapping
                class CrossModalProjector(torch.nn.Module):
                    def __init__(self, in_features: int, out_features: int):
                        super().__init__()
                        self.net = torch.nn.Sequential(
                            torch.nn.Linear(in_features, 512),
                            torch.nn.SiLU(),
                            torch.nn.Linear(512, out_features)
                        )
                    def forward(self, x):
                        return self.net(x)
                
                projector = CrossModalProjector(in_dim, C).to(device=samples.device)
                
                # Attempt to load trained weights if they exist
                model_dir = Path(__file__).parent.parent / "models"
                weight_path = model_dir / f"crossmodal_proj_{in_dim}_to_{C}.pt"
                if weight_path.exists():
                    try:
                        projector.load_state_dict(torch.load(weight_path, map_location=samples.device))
                        log.info(f"Loaded trained projection weights from {weight_path}")
                    except Exception as e:
                        log.warning(f"Failed to load projection weights: {e}")
                else:
                    log.info("No trained weights found. Using initialized (untrained) projection network.")
                
                # We do not want gradients inside inference nodes
                with torch.no_grad():
                    channel_offsets = projector(pooled)
                
                if channel_offsets.shape[0] == 1 and B > 1:
                    channel_offsets = channel_offsets.expand(B, -1)
                elif channel_offsets.shape[0] != B:
                    # truncate or pad
                    channel_offsets = channel_offsets[:B]
                
                delta = channel_offsets.view(channel_offsets.shape[0], C, 1, 1).expand(-1, -1, H, W).contiguous()
                return delta
                
        # Fallback
        log.info("Embedding_Projection: No conditioning provided, falling back to zeros.")
        return torch.zeros(B, C, H, W, dtype=torch.float32, device=samples.device)

    def _manual_json(self, instruction: str, samples: torch.Tensor) -> torch.Tensor:
        """Parse JSON for per-channel control."""
        B, C, H, W = samples.shape
        if not instruction.strip():
            log.warning("Manual_JSON: empty instruction.")
            return torch.zeros(B, C, H, W, dtype=torch.float32, device=samples.device)

        # Strip markdown fences
        cleaned = instruction.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            log.error(f"Manual_JSON parse error: {exc}")
            return torch.zeros(B, C, H, W, dtype=torch.float32, device=samples.device)

        if not isinstance(payload, dict):
            log.error(f"Manual_JSON expected dict, got {type(payload).__name__}")
            return torch.zeros(B, C, H, W, dtype=torch.float32, device=samples.device)

        channel_offsets = torch.zeros(C, dtype=torch.float32)

        for raw_key, raw_value in payload.items():
            try:
                offset = float(raw_value)
            except (TypeError, ValueError):
                log.warning(f"Manual_JSON: key '{raw_key}' non-numeric value {raw_value}; skipping.")
                continue

            ch_idx = _resolve_channel_index(raw_key, C)
            if ch_idx is not None and ch_idx < C:
                channel_offsets[ch_idx] += offset
            else:
                # Try keyword signature scaling
                key = _resolve_keyword(raw_key)
                if key:
                    sig = LATENT_SIGNATURES[key]
                    sig_tensor = torch.tensor(sig, dtype=torch.float32)
                    if C > len(sig):
                        padded_sig = torch.zeros(C, dtype=torch.float32)
                        padded_sig[:len(sig)] = sig_tensor
                        channel_offsets += padded_sig * offset
                    else:
                        channel_offsets += sig_tensor[:C] * offset
                else:
                    log.warning(f"Manual_JSON: unrecognized key '{raw_key}'")

        delta = channel_offsets.view(1, C, 1, 1).expand(B, -1, H, W).contiguous()
        return delta


NODE_CLASS_MAPPINGS = {
    "Wayfinder_CrossModalBridge": Wayfinder_CrossModalBridge,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wayfinder_CrossModalBridge": "Wayfinder Cross-Modal Bridge",
}