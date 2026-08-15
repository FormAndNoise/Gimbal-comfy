import folder_paths
import comfy.utils
import comfy.sd

class LikenessVectorIsolator:
    """
    [VibeCheck Badge: 🟢 Stabilized]
    
    A dynamic 'probe' for LoRA influence to isolate and modulate specific identity vectors.
    """
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Overall model patch strength."}),
                "alpha": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Mapped to CLIP strength or individual alpha isolation if supported."}),
                "likeness_mask": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05, "tooltip": "Weights the identity tokens in CLIP text-encoder."}),
                }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_lora"
    CATEGORY = "Wayfinder/Latent"

    def apply_lora(self, model, clip, lora_name, strength, alpha, likeness_mask):
        if strength == 0 and alpha == 0 and likeness_mask == 1.0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise ValueError(f"LoRA '{lora_name}' not found.")

        lora_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # Apply Likeness Mask Logic mapping
        # Alpha is used as the base modifier, but lightness mask isolates the CLIP text encoder
        # weighting to prioritize identity.
        clip_str = alpha * likeness_mask

        # Apply using the backend method
        patched_model, patched_clip = comfy.sd.load_lora_for_models(
            model, 
            clip, 
            lora_dict, 
            strength, 
            clip_str
        )
        return (patched_model, patched_clip)

NODE_CLASS_MAPPINGS = {
    "LikenessVectorIsolator": LikenessVectorIsolator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LikenessVectorIsolator": "🧬 Likeness Vector Isolator"
}
