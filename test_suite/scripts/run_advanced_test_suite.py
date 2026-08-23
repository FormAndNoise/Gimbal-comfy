import os
import sys
import json
import time
import requests
import copy

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT = "sd_xl_base_1.0.safetensors"

def queue_and_wait(prompt_dict, tag, timeout=300):
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
    img_meta = out_info[0]
    fname = img_meta["filename"]
    subfolder = img_meta.get("subfolder", "")
    img_type = img_meta.get("type", "output")
    
    view_url = f"{COMFY_URL}/view?filename={fname}&subfolder={subfolder}&type={img_type}"
    r = requests.get(view_url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch image {fname} from ComfyUI view API: {r.status_code}")
    
    dest_path = os.path.join(OUTPUT_DIR, target_filename)
    with open(dest_path, "wb") as f:
        f.write(r.content)
    print(f"  -> Saved {target_filename} ({len(r.content)} bytes)")
    return dest_path

# ==============================================================================
# TEST 1: CONCEPT BLENDER (Forward + Reverse + Pushed Past 50%)
# ==============================================================================
def run_test_1_concept_blender():
    print("\n" + "=" * 70)
    print("RUNNING TEST 1: CONCEPT BLENDER (SWEEP & INVERSION)")
    print("=" * 70)
    
    # 1. Base workflow generating Control A (Forest), Control B (Mountain)
    # and multiple forward/reverse blend points
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a dense misty forest, ancient trees, dappled light, masterpiece", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "a snowy mountain peak, frozen tundra, crisp winter air, masterpiece", "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, watermark", "clip": ["1", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Base Latents
        "6": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
        "7": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
        
        # Decode Controls
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/01_ctrl_A_forest", "images": ["8", 0]}},
        
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/01_ctrl_B_mountain", "images": ["10", 0]}},
        
        # Prompt for blend reconstruction
        "12": {"class_type": "CLIPTextEncode", "inputs": {"text": "misty landscape blending ancient forest and snowy mountain peaks, masterpiece, photorealistic", "clip": ["1", 1]}},
        
        # --- FORWARD BLENDS (Forest -> Mountain) ---
        # 50% Forward
        "20": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.50, "mode": "Normalized", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["6", 0], "target_latent": ["7", 0], "origin_latent": ["6", 0]}},
        "21": {"class_type": "KSampler", "inputs": {"seed": 123, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["12", 0], "negative": ["4", 0], "latent_image": ["20", 0]}},
        "22": {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["1", 2]}},
        "23": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/01_blend_FtoM_50", "images": ["22", 0]}},

        # 75% Forward (Push past 50%)
        "24": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.75, "mode": "Normalized", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["6", 0], "target_latent": ["7", 0], "origin_latent": ["6", 0]}},
        "25": {"class_type": "KSampler", "inputs": {"seed": 123, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["12", 0], "negative": ["4", 0], "latent_image": ["24", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/01_blend_FtoM_75", "images": ["26", 0]}},

        # 90% Forward (Strong mountain push)
        "28": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.90, "mode": "Normalized", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["6", 0], "target_latent": ["7", 0], "origin_latent": ["6", 0]}},
        "29": {"class_type": "KSampler", "inputs": {"seed": 123, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["12", 0], "negative": ["4", 0], "latent_image": ["28", 0]}},
        "30": {"class_type": "VAEDecode", "inputs": {"samples": ["29", 0], "vae": ["1", 2]}},
        "31": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/01_blend_FtoM_90", "images": ["30", 0]}},

        # --- REVERSE BLENDS (Mountain Base -> Forest Target) ---
        # 50% Reverse
        "40": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.50, "mode": "Normalized", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["7", 0], "target_latent": ["6", 0], "origin_latent": ["7", 0]}},
        "41": {"class_type": "KSampler", "inputs": {"seed": 123, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["12", 0], "negative": ["4", 0], "latent_image": ["40", 0]}},
        "42": {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["1", 2]}},
        "43": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/01_blend_MtoF_50", "images": ["42", 0]}},

        # 75% Reverse (Mountain Base + 75% Forest)
        "44": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.75, "mode": "Normalized", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["7", 0], "target_latent": ["6", 0], "origin_latent": ["7", 0]}},
        "45": {"class_type": "KSampler", "inputs": {"seed": 123, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["12", 0], "negative": ["4", 0], "latent_image": ["44", 0]}},
        "46": {"class_type": "VAEDecode", "inputs": {"samples": ["45", 0], "vae": ["1", 2]}},
        "47": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/01_blend_MtoF_75", "images": ["46", 0]}},
    }
    
    outputs = queue_and_wait(wf, "Test 1: Concept Blender Sweep")
    save_output_image(outputs, "9", "01_control_A_forest_00001_.png")
    save_output_image(outputs, "11", "01_control_B_mountain_00001_.png")
    save_output_image(outputs, "23", "01_blend_forest_to_mountain_50pct_00001_.png")
    save_output_image(outputs, "27", "01_blend_forest_to_mountain_75pct_00001_.png")
    save_output_image(outputs, "31", "01_blend_forest_to_mountain_90pct_00001_.png")
    save_output_image(outputs, "43", "01_blend_mountain_to_forest_50pct_00001_.png")
    save_output_image(outputs, "47", "01_blend_mountain_to_forest_75pct_00001_.png")

# ==============================================================================
# TEST 2: TEXT-STEERED DIFFUSION (Progression of Pushes)
# ==============================================================================
def run_test_2_text_steered():
    print("\n" + "=" * 70)
    print("RUNNING TEST 2: TEXT-STEERED DIFFUSION (PROGRESSIVE PUSH)")
    print("=" * 70)
    
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman, soft daylight, simple background, photograph", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, deformed, low quality", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Baseline unsteered portrait
        "5": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/02_ctrl_baseline", "images": ["6", 0]}},
        
        # CrossModal Bridge for Neon/Amber Lighting Vector
        "8": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "cinematic moody warm golden glowing amber rim lighting, intense shadows", "mapping_mode": "Keyword_Heuristics", "base_latent": ["5", 0]}},
        
        # Post-steer text encode
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman, dramatic cinematic golden hour lighting, amber rim lighting, photograph", "clip": ["1", 1]}},
        
        # 1. Moderate Push: Strength 1.2, Denoise 0.48
        "20": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.2, "mode": "Standard", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["5", 0], "target_latent": ["8", 0], "origin_latent": ["8", 1]}},
        "21": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.48, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["20", 0]}},
        "22": {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["1", 2]}},
        "23": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/02_steered_push_1_2", "images": ["22", 0]}},

        # 2. Strong Push: Strength 1.6, Denoise 0.54
        "24": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Standard", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["5", 0], "target_latent": ["8", 0], "origin_latent": ["8", 1]}},
        "25": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.54, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["24", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/02_steered_push_1_6", "images": ["26", 0]}},

        # 3. Heavy Push / Cyberpunk Glow: Strength 2.0, Denoise 0.60
        "28": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 2.0, "mode": "Standard", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["5", 0], "target_latent": ["8", 0], "origin_latent": ["8", 1]}},
        "29": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.60, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["28", 0]}},
        "30": {"class_type": "VAEDecode", "inputs": {"samples": ["29", 0], "vae": ["1", 2]}},
        "31": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/02_steered_push_2_0", "images": ["30", 0]}},
    }
    
    outputs = queue_and_wait(wf, "Test 2: Text Steered Push")
    save_output_image(outputs, "7", "02_control_baseline_portrait_00001_.png")
    save_output_image(outputs, "23", "02_gimbal_steered_push_1_2_00001_.png")
    save_output_image(outputs, "27", "02_gimbal_steered_push_1_6_00001_.png")
    save_output_image(outputs, "31", "02_gimbal_steered_push_2_0_00001_.png")

# ==============================================================================
# TEST 3: BRAND LOCKED LIGHTING (Push Progression + Modes)
# ==============================================================================
def run_test_3_brand_locked():
    print("\n" + "=" * 70)
    print("RUNNING TEST 3: BRAND LOCKED LIGHTING (STRONGER PUSH)")
    print("=" * 70)
    
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        # Anchor Prompt (Gold luxury watch on dark velvet)
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "brand hero shot: luxury gold watch on dark black velvet, volumetric rim lighting, deep shadows, amber highlights, commercial photography", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, amateur", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Generate Anchor
        "5": {"class_type": "KSampler", "inputs": {"seed": 1337, "control_after_generate": "fixed", "steps": 24, "cfg": 7.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "GimbalGPS_Anchor", "inputs": {"select_index": 0, "save_waypoint": True, "waypoint_name": "brand_hero_lighting_v2", "enable_perf_logging": False, "latent_batch": ["5", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/04_source_anchor_watch", "images": ["7", 0]}},
        
        # Product Prompt (Handbag)
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "elegant luxury handbag on marble surface, product photography, studio catalog", "clip": ["1", 1]}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality", "clip": ["1", 1]}},
        
        # Baseline Handbag (Neutral Daylight)
        "11": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["4", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/04_ctrl_handbag_daylight", "images": ["12", 0]}},
        
        # Handbag Target Refinement Prompt
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "elegant luxury handbag, dark moody background, gold volumetric rim lighting, luxury commercial photography", "clip": ["1", 1]}},

        # 1. Moderate Orthogonal Push: Strength 0.70, Denoise 0.44
        "20": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.70, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["11", 0], "target_latent": ["6", 0], "origin_latent": ["11", 0]}},
        "21": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.44, "model": ["1", 0], "positive": ["14", 0], "negative": ["10", 0], "latent_image": ["20", 0]}},
        "22": {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["1", 2]}},
        "23": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/04_brand_locked_push_0_70", "images": ["22", 0]}},

        # 2. Strong Orthogonal Push: Strength 1.05, Denoise 0.52
        "24": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.05, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["11", 0], "target_latent": ["6", 0], "origin_latent": ["11", 0]}},
        "25": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.52, "model": ["1", 0], "positive": ["14", 0], "negative": ["10", 0], "latent_image": ["24", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/04_brand_locked_push_1_05", "images": ["26", 0]}},

        # 3. High Transfer / Full Contrast Push: Strength 1.40, Denoise 0.58
        "28": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.40, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["11", 0], "target_latent": ["6", 0], "origin_latent": ["11", 0]}},
        "29": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.58, "model": ["1", 0], "positive": ["14", 0], "negative": ["10", 0], "latent_image": ["28", 0]}},
        "30": {"class_type": "VAEDecode", "inputs": {"samples": ["29", 0], "vae": ["1", 2]}},
        "31": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/04_brand_locked_push_1_40", "images": ["30", 0]}},
    }
    
    outputs = queue_and_wait(wf, "Test 3: Brand Locked Lighting Push")
    save_output_image(outputs, "8", "04_source_anchor_watch_00001_.png")
    save_output_image(outputs, "13", "04_control_baseline_handbag_00001_.png")
    save_output_image(outputs, "23", "04_gimbal_steered_handbag_push_0_70_00001_.png")
    save_output_image(outputs, "27", "04_gimbal_steered_handbag_push_1_05_00001_.png")
    save_output_image(outputs, "31", "04_gimbal_steered_handbag_push_1_40_00001_.png")

# ==============================================================================
# TEST 4: SEMANTIC FEATURE STEERING (Multi-Example Progression & Components)
# ==============================================================================
def run_test_4_semantic_slider():
    print("\n" + "=" * 70)
    print("RUNNING TEST 4: SEMANTIC SLIDER (MULTI-EXAMPLE & SPECTRUM)")
    print("=" * 70)
    
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a person, neutral background, studio lighting, highly detailed photograph", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, deformed, low quality", "clip": ["1", 1]}},
        
        # Batch of 4 diverse latents to compute PCA covariance subspace
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 4}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        
        # Select single base image from batch (Index 0)
        "6": {"class_type": "LatentFromBatch", "inputs": {"batch_index": 0, "length": 1, "samples": ["5", 0]}},
        
        # 1. Baseline Reference (Shift 0.0)
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/03_slider_baseline", "images": ["7", 0]}},
        
        # 2. PC1 = -4.0 (Strong Negative)
        "20": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 1, "slider_value": -4.0, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "21": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.50, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["20", 0]}},
        "22": {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["1", 2]}},
        "23": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/03_slider_pc1_minus_4", "images": ["22", 0]}},

        # 3. PC1 = -2.0 (Moderate Negative)
        "24": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 1, "slider_value": -2.0, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "25": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.50, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["24", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/03_slider_pc1_minus_2", "images": ["26", 0]}},

        # 4. PC1 = +2.0 (Moderate Positive)
        "28": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 1, "slider_value": 2.0, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "29": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.50, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["28", 0]}},
        "30": {"class_type": "VAEDecode", "inputs": {"samples": ["29", 0], "vae": ["1", 2]}},
        "31": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/03_slider_pc1_plus_2", "images": ["30", 0]}},

        # 5. PC1 = +4.0 (Strong Positive)
        "32": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 1, "slider_value": 4.0, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "33": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.50, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["32", 0]}},
        "34": {"class_type": "VAEDecode", "inputs": {"samples": ["33", 0], "vae": ["1", 2]}},
        "35": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/03_slider_pc1_plus_4", "images": ["34", 0]}},

        # 6. PC2 = +3.0 (Orthogonal Secondary Axis)
        "36": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 2, "slider_value": 3.0, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "37": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.50, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["36", 0]}},
        "38": {"class_type": "VAEDecode", "inputs": {"samples": ["37", 0], "vae": ["1", 2]}},
        "39": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test/03_slider_pc2_plus_3", "images": ["38", 0]}},
    }
    
    outputs = queue_and_wait(wf, "Test 4: Semantic Slider Sweep")
    save_output_image(outputs, "8", "03_control_baseline_pca_00001_.png")
    save_output_image(outputs, "23", "03_gimbal_slider_pc1_minus_4_00001_.png")
    save_output_image(outputs, "27", "03_gimbal_slider_pc1_minus_2_00001_.png")
    save_output_image(outputs, "31", "03_gimbal_slider_pc1_plus_2_00001_.png")
    save_output_image(outputs, "35", "03_gimbal_slider_pc1_plus_4_00001_.png")
    save_output_image(outputs, "39", "03_gimbal_slider_pc2_plus_3_00001_.png")

def main():
    print("Starting full test run suite with requested user adjustments...")
    run_test_1_concept_blender()
    run_test_2_text_steered()
    run_test_3_brand_locked()
    run_test_4_semantic_slider()
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
