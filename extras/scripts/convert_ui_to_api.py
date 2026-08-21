import json
import glob
import os
import sys

widget_mappings = {
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "CLIPTextEncode": ["text"],
    "CheckpointLoaderSimple": ["ckpt_name"],
    "SaveImage": ["filename_prefix"],
    "VAEDecode": [],
    "Gimbal_CrossModalBridge": ["llm_instruction", "mapping_mode"],
    "GimbalManifold_Explorer": ["grid_size_x", "grid_size_y", "x_strength", "y_strength", "interpolation_mode", "normalize_vectors", "clamp_output", "clamp_min", "clamp_max", "enable_perf_logging"],
    "GimbalCompass_Pro": ["strength", "mode", "clamp_output", "clamp_min", "clamp_max", "allow_batch_expand", "ortho_per_channel", "clamp_mask_input", "enable_perf_logging"],
    "GimbalGPS_Anchor": ["blend_factor", "anchor_type"],
    "Gimbal_SemanticSlider": ["attribute", "strength", "interpolation", "clamp_output"],
    "GimbalConceptBlender": ["blend_ratio", "mode"],
    "Gimbal_ConceptBlender": ["blend_ratio", "mode"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"]
}

def convert_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if "last_node_id" not in data:
        return False
    
    api_format = {}
    links = {link[0]: link for link in data.get('links', [])}

    for node in data.get('nodes', []):
        node_id = str(node['id'])
        node_type = node['type']
        inputs = {}
        
        # Add widgets
        widgets_values = node.get('widgets_values', [])
        widget_names = widget_mappings.get(node_type, [])
        for i, val in enumerate(widgets_values):
            if i < len(widget_names):
                inputs[widget_names[i]] = val
            else:
                inputs[f"widget_{i}"] = val

        # Add links
        for in_link in node.get('inputs', []):
            link_id = in_link.get('link')
            if link_id is not None and link_id in links:
                l = links[link_id]
                origin_node = str(l[1])
                origin_slot = l[2]
                inputs[in_link['name']] = [origin_node, origin_slot]

        api_format[node_id] = {
            "class_type": node_type,
            "inputs": inputs
        }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(api_format, f, indent=2)
    return True

files = glob.glob('example_workflows/*.json')
converted = 0
for f in files:
    try:
        if convert_file(f):
            print(f"Converted {f} to API format")
            converted += 1
    except Exception as e:
        print(f"Error converting {f}: {e}")
print(f"Converted {converted} files total.")
