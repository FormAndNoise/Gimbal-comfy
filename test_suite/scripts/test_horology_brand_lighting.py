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

def test_luxury_horology_lighting():
    print("\n" + "=" * 70)
    print("TESTING LUXURY HOROLOGY BRAND-LOCKED LIGHTING TRANSFER")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # 1. Anchor Source: Luxury matte black chronograph watch under dramatic warm amber tungsten slit lighting on dark slate
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "luxury minimalist matte black automatic chronograph watch on black slate pedestal, razor-sharp warm amber gold vertical slit lighting, dark moody shadow falloff, commercial macro product photography, sharp focus, 8k", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        "10": {"class_type": "KSampler", "inputs": {"seed": 4402, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Brand_V6/01_anchor_watch_amber_slit", "images": ["11", 0]}},

        # 2. Recipient Baseline: Same luxury watch under flat neutral daylight
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "luxury minimalist matte black automatic chronograph watch on white marble pedestal, flat neutral overcast daylight, clean catalog product photography, sharp focus, 8k", "clip": ["1", 1]}},
        
        "20": {"class_type": "KSampler", "inputs": {"seed": 4402, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["1", 2]}},
        "22": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Brand_V6/02_ctrl_watch_daylight", "images": ["21", 0]}},

        # 3. GPS Anchor Lighting Transfer
        "24": {"class_type": "CLIPTextEncode", "inputs": {"text": "luxury minimalist matte black automatic chronograph watch on black slate pedestal, razor-sharp warm amber gold vertical slit lighting, dark moody shadow falloff, commercial macro product photography, sharp focus, 8k", "clip": ["1", 1]}},

        "30": {"class_type": "GimbalGPS_Anchor", "inputs": {"select_index": 0, "save_waypoint": False, "waypoint_name": "brand_horology_v6", "enable_perf_logging": False, "latent_batch": ["10", 0]}},
        "31": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.5, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["30", 0], "origin_latent": ["4", 0]}},
        "32": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["31", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        "35": {"class_type": "KSampler", "inputs": {"seed": 4402, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.58, "model": ["1", 0], "positive": ["24", 0], "negative": ["3", 0], "latent_image": ["32", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Brand_V6/03_steered_watch_amber_slit", "images": ["36", 0]}},
    }

    outputs = queue_and_wait(wf, "Luxury Horology Lighting")
    save_output_image(outputs, "12", "01_sec3_anchor_amber_slit.png")
    save_output_image(outputs, "22", "02_sec3_ctrl_daylight_watch.png")
    save_output_image(outputs, "37", "03_sec3_steered_amber_watch.png")

if __name__ == "__main__":
    test_luxury_horology_lighting()
