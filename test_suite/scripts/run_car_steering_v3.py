import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/car_showcase_v3"
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

def run_bold_car_steering_suite():
    print("\n" + "=" * 70)
    print("RUNNING BOLD AUTOMOTIVE STEERING SUITE (HIGH DENOISE & DRAMATIC ENVIRONMENTS)")
    print("=" * 70)

    # 1. Base Studio Car (Seed 4412)
    # 2. Cyberpunk Neon Rain (Denoise 0.65, 0.72, 0.78)
    # 3. Desert Golden Hour Sunset (Denoise 0.72)
    
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # Clean Studio Baseline
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "sleek futuristic concept hypercar in minimalist studio, studio photography, clean reflections, sharp focus, 8k, automotive design", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Base Studio Car (Seed 4412)
        "10": {"class_type": "KSampler", "inputs": {"seed": 4412, "control_after_generate": "fixed", "steps": 24, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Car_V3/01_car_studio_ctrl", "images": ["11", 0]}},

        # Prompt Cyberpunk Night
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "sleek futuristic concept hypercar on rain-soaked city street at midnight, glowing neon signs, wet asphalt puddles reflecting neon lights, volumetric cyan and magenta rim lighting, cinematic film still, masterpiece, 8k", "clip": ["1", 1]}},

        # Prompt Desert Sunset
        "15": {"class_type": "CLIPTextEncode", "inputs": {"text": "sleek futuristic concept hypercar on desert highway at dramatic golden hour sunset, glowing orange sun on horizon, heat haze, warm volumetric dust lighting, cinematic wide shot, masterpiece, 8k", "clip": ["1", 1]}},

        # CrossModal Cyberpunk Steering
        "20": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "rain-soaked midnight city, neon cyan and magenta reflections, wet asphalt, dark atmosphere", "mapping_mode": "Keyword_Heuristics", "base_latent": ["10", 0]}},
        "21": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.8, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["10", 0], "target_latent": ["20", 0], "origin_latent": ["20", 1]}},
        "22": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["21", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        # Cyberpunk Denoise 0.65
        "25": {"class_type": "KSampler", "inputs": {"seed": 4412, "control_after_generate": "fixed", "steps": 22, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["22", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Car_V3/02_cyberpunk_denoise_065", "images": ["26", 0]}},

        # Cyberpunk Denoise 0.72 (Bold dramatic environment)
        "30": {"class_type": "KSampler", "inputs": {"seed": 4412, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.72, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["22", 0]}},
        "31": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["1", 2]}},
        "32": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Car_V3/03_cyberpunk_denoise_072", "images": ["31", 0]}},

        # Desert Sunset Steering
        "40": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "desert highway at sunset, glowing golden orange sun, warm dust haze, lens flare", "mapping_mode": "Keyword_Heuristics", "base_latent": ["10", 0]}},
        "41": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.8, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["10", 0], "target_latent": ["40", 0], "origin_latent": ["40", 1]}},
        "42": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["41", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        # Desert Sunset Denoise 0.72
        "45": {"class_type": "KSampler", "inputs": {"seed": 4412, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.72, "model": ["1", 0], "positive": ["15", 0], "negative": ["3", 0], "latent_image": ["42", 0]}},
        "46": {"class_type": "VAEDecode", "inputs": {"samples": ["45", 0], "vae": ["1", 2]}},
        "47": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Car_V3/04_desert_sunset_denoise_072", "images": ["46", 0]}},
    }

    outputs = queue_and_wait(wf, "Bold Car Steering Suite")
    save_output_image(outputs, "12", "01_car_studio_ctrl.png")
    save_output_image(outputs, "27", "02_cyberpunk_denoise_065.png")
    save_output_image(outputs, "32", "03_cyberpunk_denoise_072.png")
    save_output_image(outputs, "47", "04_desert_sunset_denoise_072.png")

if __name__ == "__main__":
    run_bold_car_steering_suite()
