import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/fresh_sections_3_and_4"
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
    
    dest_path = os.path.join(OUTPUT_DIR, target_filename)
    img_meta = out_info[0]
    fname = img_meta["filename"]
    subfolder = img_meta.get("subfolder", "")
    img_type = img_meta.get("type", "output")
    
    view_url = f"{COMFY_URL}/view?filename={fname}&subfolder={subfolder}&type={img_type}"
    r = requests.get(view_url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch image {fname}: {r.status_code}")
    
    with open(dest_path, "wb") as f:
        f.write(r.content)
    print(f"  -> Saved {target_filename} ({len(r.content)} bytes)")
    return dest_path

def test_perfume_brand_lighting():
    print("\n" + "=" * 70)
    print("TESTING LUXURY PERFUME BRAND-LOCKED LIGHTING TRANSFER (CROSS-MODAL ORTHOGONAL)")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # 1. Base Perfume in Daylight Studio (Seed 1001)
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "luxury minimalist geometric glass perfume bottle on white marble pedestal, clean clear soft daylight, neutral studio lighting, high-end cosmetic photography, sharp focus, 8k", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        "10": {"class_type": "KSampler", "inputs": {"seed": 1001, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Brand_V7/01_perfume_daylight_ctrl", "images": ["11", 0]}},

        # 2. Steered 1: Crimson Ruby & Gold Slit Light
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "luxury minimalist geometric glass perfume bottle on dark black pedestal, razor-sharp glowing ruby red and amber gold vertical slit lighting, dark moody shadow falloff, high-end editorial commercial photography, 8k", "clip": ["1", 1]}},

        "20": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "ruby red and amber gold slit lighting, dark black pedestal, moody shadow falloff", "mapping_mode": "Keyword_Heuristics", "base_latent": ["10", 0]}},
        "21": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["10", 0], "target_latent": ["20", 0], "origin_latent": ["20", 1]}},
        "22": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["21", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        "25": {"class_type": "KSampler", "inputs": {"seed": 1001, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["22", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Brand_V7/02_perfume_ruby_slit", "images": ["26", 0]}},

        # 3. Steered 2: Emerald Cyan & Obsidian Night Lighting
        "30": {"class_type": "CLIPTextEncode", "inputs": {"text": "luxury minimalist geometric glass perfume bottle on obsidian pedestal, glowing emerald green and electric cyan edge lighting, midnight darkroom reflections, luxury editorial photography, 8k", "clip": ["1", 1]}},

        "31": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "glowing emerald green and electric cyan edge lighting, obsidian pedestal, midnight darkroom reflections", "mapping_mode": "Keyword_Heuristics", "base_latent": ["10", 0]}},
        "32": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["10", 0], "target_latent": ["31", 0], "origin_latent": ["31", 1]}},
        "33": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["32", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        "35": {"class_type": "KSampler", "inputs": {"seed": 1001, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["30", 0], "negative": ["3", 0], "latent_image": ["33", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Brand_V7/03_perfume_emerald_cyan", "images": ["36", 0]}},
    }

    outputs = queue_and_wait(wf, "Perfume Brand Lighting")
    save_output_image(outputs, "12", "01_sec3_perfume_daylight_ctrl.png")
    save_output_image(outputs, "27", "02_sec3_perfume_ruby_slit.png")
    save_output_image(outputs, "37", "03_sec3_perfume_emerald_cyan.png")

if __name__ == "__main__":
    test_perfume_brand_lighting()
