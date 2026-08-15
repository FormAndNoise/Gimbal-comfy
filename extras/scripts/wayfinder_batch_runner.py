import json
import glob
import os
import requests
import copy

widget_mappings = {
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "CLIPTextEncode": ["text"],
    "CheckpointLoaderSimple": ["ckpt_name"],
    "SaveImage": ["filename_prefix"],
    "VAEDecode": [],
    "VAEEncode": [],
    "LoadImage": ["image", "upload"],
    "ShowText|pysssss": ["text"],
    "Wayfinder_CrossModalBridge": ["llm_instruction", "mapping_mode"],
    "WayfinderManifold_Explorer": ["grid_size_x", "grid_size_y", "x_strength", "y_strength", "interpolation_mode", "normalize_vectors", "clamp_output", "clamp_min", "clamp_max", "enable_perf_logging"],
    "WayfinderCompass_Pro": ["strength", "mode", "clamp_output", "clamp_min", "clamp_max", "allow_batch_expand", "ortho_per_channel", "clamp_mask_input", "enable_perf_logging"],
    "WayfinderGPS_Anchor": ["select_index", "save_waypoint", "waypoint_name", "enable_perf_logging"],
    "Wayfinder_SemanticSlider": ["pc_index", "slider_value", "orthogonalize"],
    "WayfinderConceptBlender": ["blend_ratio", "mode"],
    "Wayfinder_ConceptBlender": ["blend_ratio", "mode"],
    "WayfinderGPS_Load": ["waypoint_name", "restore_mode"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "LatentFromBatch": ["batch_index", "length"]
}

def convert_to_api(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # If already API format
    if "nodes" not in data:
        return data
        
    api_format = {}
    links = {link[0]: link for link in data.get('links', [])}

    for node in data.get('nodes', []):
        node_id = str(node['id'])
        node_type = node['type']
        inputs = {}
        
    def resolve_link(l_id, links_d, nodes_d):
        if l_id is None or l_id not in links_d:
            return None, None
        l = links_d[l_id]
        src_id, src_slot = str(l[1]), l[2]
        
        src_node = nodes_d.get(src_id)
        if src_node and src_node.get("type") == "Reroute":
            # Find the input to this Reroute
            if "inputs" in src_node and len(src_node["inputs"]) > 0:
                return resolve_link(src_node["inputs"][0].get("link"), links_d, nodes_d)
        return src_id, src_slot

    for node in data.get('nodes', []):
        if node['type'] == 'Reroute':
            continue
            
        node_id = str(node['id'])
        node_type = node['type']
        inputs = {}
        
        widgets_values = node.get('widgets_values', [])
        widget_names = widget_mappings.get(node_type, [])
        for i, val in enumerate(widgets_values):
            if i < len(widget_names):
                inputs[widget_names[i]] = val
            else:
                inputs[f"widget_{i}"] = val

        for in_link in node.get('inputs', []):
            link_id = in_link.get('link')
            origin_node, origin_slot = resolve_link(link_id, links, {str(n['id']): n for n in data.get('nodes', [])})
            if origin_node is not None:
                inputs[in_link['name']] = [origin_node, origin_slot]

        api_format[node_id] = {
            "class_type": node_type,
            "inputs": inputs
        }
    return api_format

def convert_to_flux(api_data):
    flux_data = copy.deepcopy(api_data)
    
    # 1. Add Flux specific loaders
    flux_data["200"] = {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": "flux1-dev-Q4_0.gguf"}
    }
    flux_data["201"] = {
        "class_type": "DualCLIPLoader",
        "inputs": {
            "clip_name1": "t5\\t5xxl_fp8_e4m3fn.safetensors",
            "clip_name2": "clip_l.safetensors",
            "type": "flux"
        }
    }
    flux_data["202"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "ae.safetensors"}
    }
    
    # 2. Find Checkpoint loader
    ckpt_id = None
    for k, v in list(flux_data.items()):
        if v.get("class_type") == "CheckpointLoaderSimple":
            ckpt_id = str(k)
            del flux_data[k]
            break
            
    if not ckpt_id:
        # If no checkpoint loader, we can't do flux substitution easily
        return flux_data
        
    # 3. Reroute inputs
    for node_id, node in flux_data.items():
        inputs = node.get("inputs", {})
        for in_name, in_val in list(inputs.items()):
            if isinstance(in_val, list) and len(in_val) == 2 and str(in_val[0]) == ckpt_id:
                slot = int(in_val[1])
                if slot == 0: # MODEL
                    inputs[in_name] = ["200", 0]
                elif slot == 1: # CLIP
                    inputs[in_name] = ["201", 0]
                elif slot == 2: # VAE
                    inputs[in_name] = ["202", 0]
                    
    return flux_data

def run_tests():
    # Ping ComfyUI
    COMFY_URL = "http://127.0.0.1:8188/prompt"
    try:
        requests.get("http://127.0.0.1:8188/system_stats", timeout=2)
    except:
        try:
            requests.get("http://127.0.0.1:8000/system_stats", timeout=2)
            COMFY_URL = "http://127.0.0.1:8000/prompt"
        except:
            print("WARNING: ComfyUI is not reachable on 8188 or 8000. Will only save json files.")
            COMFY_URL = None

    workflows = glob.glob("../example_workflows/Wayfinder_0*.json")
    
    os.makedirs("../example_workflows/api", exist_ok=True)
    
    for wf in sorted(workflows):
        basename = os.path.basename(wf)
        print(f"Processing {basename}...")
        api_data_sdxl = convert_to_api(wf)
        
        out_sdxl = f"../example_workflows/api/API_{basename}"
        with open(out_sdxl, "w", encoding="utf-8") as f:
            json.dump(api_data_sdxl, f, indent=2)
            
        flux_data = convert_to_flux(api_data_sdxl)
        out_flux = f"../example_workflows/api/API_FLUX_{basename}"
        with open(out_flux, "w", encoding="utf-8") as f:
            json.dump(flux_data, f, indent=2)
            
        if COMFY_URL:
            # Dispatch SDXL
            r = requests.post(COMFY_URL, json={"prompt": api_data_sdxl})
            if r.status_code == 200:
                print(f"  [>] Queued SDXL run for {basename}")
            else:
                print(f"  [X] Failed SDXL run for {basename}: {r.text}")
                
            # Dispatch Flux
            r2 = requests.post(COMFY_URL, json={"prompt": flux_data})
            if r2.status_code == 200:
                print(f"  [>] Queued FLUX run for {basename}")
            else:
                print(f"  [X] Failed FLUX run for {basename}: {r2.text}")

if __name__ == "__main__":
    run_tests()
