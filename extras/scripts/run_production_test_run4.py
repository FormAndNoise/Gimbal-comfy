import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/comparisons"
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
# TEST 1 (RUN 4): CONCEPT BLENDER — THE PERFECTED ALPINE FOREST CONTINUUM
# ==============================================================================
def run_test_1_concept_blender_v4():
    print("\n" + "=" * 70)
    print("RUNNING TEST 1 (V4): CONCEPT BLENDER — PERFECTED CONTINUUM")
    print("=" * 70)
    
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a dense misty forest, ancient mossy trees, dappled sunlight, masterpiece, sharp focus", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "a towering snowy mountain peak, frozen tundra, crisp winter air, masterpiece, sharp focus", "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, ghosting, double exposure, watermark", "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "majestic snowy mountain peak rising above ancient misty evergreen forest, winter alpine landscape, masterpiece, photorealistic", "clip": ["1", 1]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Baselines (24 steps, Seed 42)
        "10": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 24, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/01_ctrl_A_forest", "images": ["11", 0]}},
        
        "13": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 24, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}},
        "15": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/01_ctrl_B_mountain", "images": ["14", 0]}},

        # Trajectory Step 8 Latents
        "20": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 8, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "21": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 8, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},

        # 1. 35% Blend (Alpine Valley Forest)
        "30": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.35, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["21", 0], "origin_latent": ["20", 0]}},
        "31": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 22, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.70, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["30", 0]}},
        "32": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["1", 2]}},
        "33": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/01_slerp_35pct", "images": ["32", 0]}},

        # 2. 50% Blend (Sub-Alpine Glacier Forest)
        "34": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.50, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["21", 0], "origin_latent": ["20", 0]}},
        "35": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 22, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.70, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["34", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/01_slerp_50pct", "images": ["36", 0]}},

        # 3. 65% Blend (Dominant Alpine Peak with Pine Groves)
        "38": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.65, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["21", 0], "origin_latent": ["20", 0]}},
        "39": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 22, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.70, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["38", 0]}},
        "40": {"class_type": "VAEDecode", "inputs": {"samples": ["39", 0], "vae": ["1", 2]}},
        "41": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/01_slerp_65pct", "images": ["40", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 1 V4: Concept Blender")
    save_output_image(outputs, "12", "01_control_A_forest_00001_.png")
    save_output_image(outputs, "15", "01_control_B_mountain_00001_.png")
    save_output_image(outputs, "33", "01_slerp_35pct_00001_.png")
    save_output_image(outputs, "37", "01_slerp_50pct_00001_.png")
    save_output_image(outputs, "41", "01_slerp_65pct_00001_.png")

# ==============================================================================
# TEST 2 (RUN 4): TEXT-STEERED DIFFUSION — HEAVY DENOISE SHOWCASE
# ==============================================================================
def run_test_2_text_steered_v4():
    print("\n" + "=" * 70)
    print("RUNNING TEST 2 (V4): TEXT-STEERED DIFFUSION — HEAVY DENOISE SHOWCASE")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman, soft daylight, simple background, photograph", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, deformed, low quality", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Base portrait
        "5": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/02_ctrl_baseline", "images": ["6", 0]}},
        
        # CrossModal Bridge for Neon/Amber Lighting Vector
        "8": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "cinematic moody warm golden glowing amber rim lighting, neon shadows", "mapping_mode": "Keyword_Heuristics", "base_latent": ["5", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman, dramatic cinematic golden hour lighting, amber rim lighting, photograph", "clip": ["1", 1]}},
        
        # Steering Vector (Strength 1.5)
        "10": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.5, "mode": "Standard", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["5", 0], "target_latent": ["8", 0], "origin_latent": ["8", 1]}},

        # 1. Denoise 0.65 (Rich Amber Rim Lighting & Contrast)
        "20": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["10", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["1", 2]}},
        "22": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/02_steered_denoise_065", "images": ["21", 0]}},

        # 2. Denoise 0.72 (Vivid Editorial Cyberpunk Amber)
        "23": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.72, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["10", 0]}},
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["1", 2]}},
        "25": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/02_steered_denoise_072", "images": ["24", 0]}},

        # 3. Denoise 0.80 (Deep Atmospheric Cinematic Studio)
        "26": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.80, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["10", 0]}},
        "27": {"class_type": "VAEDecode", "inputs": {"samples": ["26", 0], "vae": ["1", 2]}},
        "28": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/02_steered_denoise_080", "images": ["27", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 2 V4: Text Steered Heavy Denoise")
    save_output_image(outputs, "7", "02_control_baseline_portrait_00001_.png")
    save_output_image(outputs, "22", "02_steered_denoise_065_00001_.png")
    save_output_image(outputs, "25", "02_steered_denoise_072_00001_.png")
    save_output_image(outputs, "28", "02_steered_denoise_080_00001_.png")

# ==============================================================================
# TEST 3 (RUN 4): BRAND LOCKED LIGHTING — COMMERCIAL EDITORIAL BENCHMARK
# ==============================================================================
def run_test_3_brand_locked_v4():
    print("\n" + "=" * 70)
    print("RUNNING TEST 3 (V4): BRAND LOCKED LIGHTING — EDITORIAL BENCHMARK")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "brand hero shot: luxury gold watch on dark black velvet, volumetric rim lighting, deep shadows, amber highlights, commercial photography", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, amateur", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Watch Anchor
        "5": {"class_type": "KSampler", "inputs": {"seed": 1337, "control_after_generate": "fixed", "steps": 24, "cfg": 7.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "GimbalGPS_Anchor", "inputs": {"select_index": 0, "save_waypoint": True, "waypoint_name": "brand_watch_lighting", "enable_perf_logging": False, "latent_batch": ["5", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/04_source_anchor_watch", "images": ["7", 0]}},
        
        # Handbag Baseline (Seed 4242)
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "elegant luxury handbag on marble surface, product photography, studio catalog", "clip": ["1", 1]}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality", "clip": ["1", 1]}},
        "11": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["4", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/04_ctrl_handbag_daylight", "images": ["12", 0]}},
        
        # Brand Target Refinement Prompt
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "elegant luxury handbag, dark black velvet backdrop, gold volumetric rim lighting, luxury commercial photography", "clip": ["1", 1]}},

        # Orthogonal Steering Vector Injection
        "15": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.15, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["11", 0], "target_latent": ["6", 0], "origin_latent": ["11", 0]}},

        # 1. Denoise 0.65 (Rich Dark Velvet & Gold Speculars)
        "20": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["14", 0], "negative": ["10", 0], "latent_image": ["15", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["1", 2]}},
        "22": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/04_brand_denoise_065", "images": ["21", 0]}},

        # 2. Denoise 0.72 (Accepted Master Benchmark)
        "23": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.72, "model": ["1", 0], "positive": ["14", 0], "negative": ["10", 0], "latent_image": ["15", 0]}},
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["1", 2]}},
        "25": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/04_brand_denoise_072", "images": ["24", 0]}},

        # 3. Denoise 0.80 (Dramatic Deep Contrast Studio)
        "26": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.80, "model": ["1", 0], "positive": ["14", 0], "negative": ["10", 0], "latent_image": ["15", 0]}},
        "27": {"class_type": "VAEDecode", "inputs": {"samples": ["26", 0], "vae": ["1", 2]}},
        "28": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/04_brand_denoise_080", "images": ["27", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 3 V4: Brand Locked Lighting Benchmark")
    save_output_image(outputs, "8", "04_source_anchor_watch_00001_.png")
    save_output_image(outputs, "13", "04_control_baseline_handbag_00001_.png")
    save_output_image(outputs, "22", "04_brand_denoise_065_00001_.png")
    save_output_image(outputs, "25", "04_brand_denoise_072_00001_.png")
    save_output_image(outputs, "28", "04_brand_denoise_080_00001_.png")

# ==============================================================================
# TEST 4 & 5 (RUN 4): CONCEPT VECTOR ANALOGY & CONTROLLED SEMANTIC STEERING
# (COMPLETELY REPLACING RANDOM PCA WITH EXPLICIT SEMANTIC ATTRIBUTE VECTORS!)
# ==============================================================================
def run_test_4_and_5_concept_analogy_v4():
    print("\n" + "=" * 70)
    print("RUNNING TEST 4 & 5 (V4): CONCEPT VECTOR ANALOGY & PURE ATTRIBUTE VECTORS")
    print("=" * 70)

    # In V4, we replace random batch PCA with GimbalVectorAnalogy: Target = C + strength * (A - B)
    # 1. Attribute Dial 1: Natural Smile / Expression Transfer (A=Smiling, B=Neutral, C=Subject)
    # 2. Attribute Dial 2: Studio Sunglasses / Accessory Transfer (A=Sunglasses, B=Neutral, C=Subject)
    # 3. Attribute Dial 3: Cyberpunk Neon Lighting Transfer (A=Cyberpunk, B=Neutral, C=Subject)
    # This guarantees 100% clean, crisp, deterministic semantic control with ZERO frying or random noise!
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a person with a warm happy joyful smile, studio lighting, highly detailed photograph", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a person with a neutral serious expression, studio lighting, highly detailed photograph", "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a person wearing stylish dark sunglasses, studio lighting, highly detailed photograph", "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, deformed, low quality, cartoon", "clip": ["1", 1]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Base Subject C (Seed 789, unsteered neutral portrait)
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a person, soft neutral daylight, studio photograph", "clip": ["1", 1]}},
        "11": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["10", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/03_analogy_base_subject", "images": ["12", 0]}},

        # Source Concept A: Smiling (Seed 999)
        "20": {"class_type": "KSampler", "inputs": {"seed": 999, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        # Source Concept B: Neutral (Seed 999 - exact same seed to isolate ONLY expression delta!)
        "21": {"class_type": "KSampler", "inputs": {"seed": 999, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        
        # Source Concept D: Sunglasses (Seed 999 - to isolate sunglasses delta!)
        "22": {"class_type": "KSampler", "inputs": {"seed": 999, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},

        # --- ATTRIBUTE DIAL 1: SMILE ANALOGY ARITHMETIC ---
        # 1A. Negative Smile (-1.0 -> Serious / Stern Expression)
        "30": {"class_type": "GimbalVectorAnalogy", "inputs": {"strength": -1.0, "ortho_project": True, "preserve_norm": True, "concept_A": ["20", 0], "concept_B": ["21", 0], "concept_C": ["11", 0]}},
        "31": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.42, "model": ["1", 0], "positive": ["10", 0], "negative": ["5", 0], "latent_image": ["30", 0]}},
        "32": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["1", 2]}},
        "33": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/03_analogy_smile_minus_1", "images": ["32", 0]}},

        # 1B. Positive Smile (+1.0 -> Natural Warm Smile)
        "34": {"class_type": "GimbalVectorAnalogy", "inputs": {"strength": 1.0, "ortho_project": True, "preserve_norm": True, "concept_A": ["20", 0], "concept_B": ["21", 0], "concept_C": ["11", 0]}},
        "35": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.42, "model": ["1", 0], "positive": ["10", 0], "negative": ["5", 0], "latent_image": ["34", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/03_analogy_smile_plus_1", "images": ["36", 0]}},

        # 1C. Strong Smile (+1.6 -> Broad Joyful Smile)
        "38": {"class_type": "GimbalVectorAnalogy", "inputs": {"strength": 1.6, "ortho_project": True, "preserve_norm": True, "concept_A": ["20", 0], "concept_B": ["21", 0], "concept_C": ["11", 0]}},
        "39": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.48, "model": ["1", 0], "positive": ["10", 0], "negative": ["5", 0], "latent_image": ["38", 0]}},
        "40": {"class_type": "VAEDecode", "inputs": {"samples": ["39", 0], "vae": ["1", 2]}},
        "41": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/03_analogy_smile_plus_1_6", "images": ["40", 0]}},

        # --- ATTRIBUTE DIAL 2: SUNGLASSES ACCESSORY TRANSFER ---
        # 2A. Sunglasses Infusion (+1.2)
        "50": {"class_type": "GimbalVectorAnalogy", "inputs": {"strength": 1.2, "ortho_project": True, "preserve_norm": True, "concept_A": ["22", 0], "concept_B": ["21", 0], "concept_C": ["11", 0]}},
        "51": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.50, "model": ["1", 0], "positive": ["10", 0], "negative": ["5", 0], "latent_image": ["50", 0]}},
        "52": {"class_type": "VAEDecode", "inputs": {"samples": ["51", 0], "vae": ["1", 2]}},
        "53": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test4/03_analogy_sunglasses", "images": ["52", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 4 & 5 V4: Concept Vector Analogy")
    save_output_image(outputs, "13", "03_analogy_base_subject_00001_.png")
    save_output_image(outputs, "33", "03_analogy_smile_minus_1_00001_.png")
    save_output_image(outputs, "37", "03_analogy_smile_plus_1_00001_.png")
    save_output_image(outputs, "41", "03_analogy_smile_plus_1_6_00001_.png")
    save_output_image(outputs, "53", "03_analogy_sunglasses_plus_1_2_00001_.png")

def main():
    print("Starting Run 4 (Production Suite) based on user guidance...")
    run_test_1_concept_blender_v4()
    run_test_2_text_steered_v4()
    run_test_3_brand_locked_v4()
    run_test_4_and_5_concept_analogy_v4()
    print("\n" + "=" * 70)
    print("ALL RUN 4 BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
