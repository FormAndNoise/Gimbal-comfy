import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/architectural_showcase"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT = "sd_xl_base_1.0.safetensors"

def queue_and_wait(prompt_dict, tag, timeout=360):
    payload = {"prompt": prompt_dict}
    r = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"[{tag}] Queue failed ({r.status_code}): {r.text}")
    prompt_id = r.json()["prompt_id"]
    print(f"[{tag}] Queued prompt_id: {prompt_id}. Waiting for completion...")
    
    start_t = time.time()
    while time.time() - start_t < timeout:
        time.sleep(2)
        hr = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
        if hr.status_code == 200:
            hdata = hr.json()
            if prompt_id in hdata:
                item = hdata[prompt_id]
                status = item.get("status", {})
                if status.get("completed", False):
                    elapsed = round(time.time() - start_t, 2)
                    print(f"[{tag}] Completed in {elapsed}s")
                    return item.get("outputs", {})
                elif status.get("status_str") == "error":
                    raise RuntimeError(f"[{tag}] Execution error: {status.get('messages')}")
    raise TimeoutError(f"[{tag}] Timed out after {timeout}s")

def save_output_image(outputs, save_node_id, target_filename):
    out_info = outputs.get(str(save_node_id), {}).get("images", [])
    if not out_info:
        raise RuntimeError(f"No image outputs found for node {save_node_id}: {outputs}")
    
    saved_paths = []
    for idx, img_meta in enumerate(out_info):
        fname = img_meta["filename"]
        subfolder = img_meta.get("subfolder", "")
        img_type = img_meta.get("type", "output")
        
        view_url = f"{COMFY_URL}/view?filename={fname}&subfolder={subfolder}&type={img_type}"
        r = requests.get(view_url, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch image {fname} from ComfyUI view API: {r.status_code}")
        
        if len(out_info) > 1:
            name_parts = os.path.splitext(target_filename)
            cur_filename = f"{name_parts[0]}_frame_{idx:02d}{name_parts[1]}"
        else:
            cur_filename = target_filename
            
        dest_path = os.path.join(OUTPUT_DIR, cur_filename)
        with open(dest_path, "wb") as f:
            f.write(r.content)
        print(f"  -> Saved {cur_filename} ({len(r.content)} bytes)")
        saved_paths.append(dest_path)
    return saved_paths

# ==============================================================================
# WORKFLOW 2: GIMBAL_09_SUBSPACE_MATERIAL_MATRIX (FACADE MUTATION)
# ==============================================================================
def run_workflow_09_material_matrix():
    print("\n" + "=" * 70)
    print("RUNNING WORKFLOW 2 (GIMBAL 09): SUBSPACE MATERIAL MATRIX — FACADE MUTATION")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # Base Architecture Prompt: Modernist Minimalist Villa
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "architectural elevation shot of a modernist villa, clean geometric lines, floor to ceiling glass, soft overcast daylight, architectural digest, sharp focus", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, distorted geometry, watermark", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # 1. Base Geometry Anchor (Seed 777)
        "10": {"class_type": "KSampler", "inputs": {"seed": 777, "control_after_generate": "fixed", "steps": 24, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Arch/09_mat_ctrl_concrete", "images": ["11", 0]}},

        # Material 1 Target: Fluted Dark Obsidian & Blackened Steel
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "architectural elevation shot of a modernist villa with fluted dark polished obsidian stone facade, blackened steel louvers, bronze tinted glass, luxury commercial architecture", "clip": ["1", 1]}},
        
        # Material 2 Target: Warm Travertine Stone & Brushed Gold Accents
        "15": {"class_type": "CLIPTextEncode", "inputs": {"text": "architectural elevation shot of a modernist villa with warm ivory travertine stone facade, brushed gold metal window trims, warm glowing interior lights", "clip": ["1", 1]}},

        # Subspace Split (SDXL: 4 channels -> 2 structural channels + 2 material/chroma channels)
        "20": {"class_type": "GimbalChannelSplit", "inputs": {"split_index": 2, "latent": ["10", 0]}},

        # Material 1: Steer Chroma Band with CrossModal & Re-merge
        "21": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "dark polished obsidian glass, blackened steel, bronze reflection", "mapping_mode": "Keyword_Heuristics", "base_latent": ["20", 1]}},
        "22": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.85, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 1], "target_latent": ["21", 0], "origin_latent": ["21", 1]}},
        "23": {"class_type": "GimbalChannelMerge", "inputs": {"latent_band_A": ["20", 0], "latent_band_B": ["22", 0]}},
        "24": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["23", 0], "truncation_psi": 0.90, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},
        
        # Render Material 1 (Obsidian) at low denoise (0.38) - Locks Geometry 100%
        "25": {"class_type": "KSampler", "inputs": {"seed": 777, "control_after_generate": "fixed", "steps": 20, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.38, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["24", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Arch/09_mat_obsidian", "images": ["26", 0]}},

        # Material 2: Travertine & Gold
        "31": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "warm ivory travertine stone, brushed gold metal accents, warm sunlight", "mapping_mode": "Keyword_Heuristics", "base_latent": ["20", 1]}},
        "32": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.85, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 1], "target_latent": ["31", 0], "origin_latent": ["31", 1]}},
        "33": {"class_type": "GimbalChannelMerge", "inputs": {"latent_band_A": ["20", 0], "latent_band_B": ["32", 0]}},
        "34": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["33", 0], "truncation_psi": 0.90, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},
        
        # Render Material 2 (Travertine) at low denoise (0.38)
        "35": {"class_type": "KSampler", "inputs": {"seed": 777, "control_after_generate": "fixed", "steps": 20, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.38, "model": ["1", 0], "positive": ["15", 0], "negative": ["3", 0], "latent_image": ["34", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Arch/09_mat_travertine", "images": ["36", 0]}},
    }

    outputs = queue_and_wait(wf, "Workflow 09: Subspace Material Matrix")
    save_output_image(outputs, "12", "09_mat_ctrl_concrete.png")
    save_output_image(outputs, "27", "09_mat_obsidian.png")
    save_output_image(outputs, "37", "09_mat_travertine.png")

if __name__ == "__main__":
    run_workflow_09_material_matrix()
