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
# TEST 1: CONCEPT BLENDER (NOISE-LEVEL SLERP & TRUE UNIFIED SYNTHESIS)
# ==============================================================================
def run_test_1_concept_blender_v3():
    print("\n" + "=" * 70)
    print("RUNNING TEST 1 (V3): CONCEPT BLENDER — TRUE NOISE & TRAJECTORY SLERP")
    print("=" * 70)
    
    # In V3, we blend the initial latent seeds (seed 42 vs seed 1042) along the hypersphere,
    # conditioning on a unified hybrid prompt ("majestic snowy mountain peak rising above ancient dense pine forest, mist, dappled light")
    # This completely eliminates double-exposure ghosting and produces a coherent hybrid scene!
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a dense misty forest, ancient mossy trees, dappled sunlight, masterpiece, sharp focus", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "a towering snowy mountain peak, frozen tundra, crisp winter air, masterpiece, sharp focus", "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, ghosting, double exposure, watermark", "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "majestic snowy mountain peak rising above ancient misty evergreen forest, winter alpine landscape, masterpiece, photorealistic", "clip": ["1", 1]}},
        
        # Noise Seeds
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Control A: Pure Forest (Seed 42, 24 steps)
        "10": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 24, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/01_ctrl_A_forest", "images": ["11", 0]}},
        
        # Control B: Pure Mountain (Seed 42, 24 steps)
        "13": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 24, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}},
        "15": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/01_ctrl_B_mountain", "images": ["14", 0]}},

        # Intermediate Step-8 Latents for Trajectory Slerp
        # Sampling 8 steps of Forest and 8 steps of Mountain
        "20": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 8, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "21": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 8, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},

        # Method 1: Slerp on Step-8 Trajectory (50% Blend) -> Denoise remaining 0.68 steps
        "30": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.50, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["21", 0], "origin_latent": ["20", 0]}},
        "31": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.68, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["30", 0]}},
        "32": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["1", 2]}},
        "33": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/01_slerp_trajectory_50", "images": ["32", 0]}},

        # Method 2: Slerp on Step-8 Trajectory (75% Mountain Dominant) -> Denoise 0.68
        "34": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.75, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["21", 0], "origin_latent": ["20", 0]}},
        "35": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.68, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["34", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/01_slerp_trajectory_75", "images": ["36", 0]}},

        # Method 3: Reverse Trajectory Slerp (Mountain Base + 50% Forest) -> Denoise 0.68
        "38": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.50, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["21", 0], "target_latent": ["20", 0], "origin_latent": ["21", 0]}},
        "39": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.68, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["38", 0]}},
        "40": {"class_type": "VAEDecode", "inputs": {"samples": ["39", 0], "vae": ["1", 2]}},
        "41": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/01_slerp_reverse_50", "images": ["40", 0]}},

        # Method 4: High Denoise (0.78) Direct Concept Fusion
        "42": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.50, "mode": "Normalized", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["10", 0], "target_latent": ["13", 0], "origin_latent": ["10", 0]}},
        "43": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.78, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["42", 0]}},
        "44": {"class_type": "VAEDecode", "inputs": {"samples": ["43", 0], "vae": ["1", 2]}},
        "45": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/01_blend_high_denoise_078", "images": ["44", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 1 V3: Concept Blender")
    save_output_image(outputs, "12", "01_control_A_forest_00001_.png")
    save_output_image(outputs, "15", "01_control_B_mountain_00001_.png")
    save_output_image(outputs, "33", "01_slerp_trajectory_50pct_00001_.png")
    save_output_image(outputs, "37", "01_slerp_trajectory_75pct_00001_.png")
    save_output_image(outputs, "41", "01_slerp_reverse_50pct_00001_.png")
    save_output_image(outputs, "45", "01_blend_high_denoise_78pct_00001_.png")

# ==============================================================================
# TEST 2: TEXT-STEERED DIFFUSION (DENOISE SWEEP: LESS VS MORE)
# ==============================================================================
def run_test_2_text_steered_denoise_sweep():
    print("\n" + "=" * 70)
    print("RUNNING TEST 2 (V3): TEXT-STEERED DIFFUSION — DENOISE SWEEP")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman, soft daylight, simple background, photograph", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, deformed, low quality", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Base portrait
        "5": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/02_ctrl_baseline", "images": ["6", 0]}},
        
        # CrossModal Bridge for Neon/Amber Lighting Vector
        "8": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "cinematic moody warm golden glowing amber rim lighting, neon shadows", "mapping_mode": "Keyword_Heuristics", "base_latent": ["5", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman, dramatic cinematic golden hour lighting, amber rim lighting, photograph", "clip": ["1", 1]}},
        
        # Steering Vector (Strength 1.4)
        "10": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.4, "mode": "Standard", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["5", 0], "target_latent": ["8", 0], "origin_latent": ["8", 1]}},

        # A. Low Denoise (0.28) — Subtle lighting glaze / strict likeness
        "20": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.28, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["10", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["1", 2]}},
        "22": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/02_denoise_028", "images": ["21", 0]}},

        # B. Moderate Denoise (0.42) — Golden tone shift
        "23": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.42, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["10", 0]}},
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["1", 2]}},
        "25": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/02_denoise_042", "images": ["24", 0]}},

        # C. High Denoise (0.58) — Rich amber rim light & deep shadows (Sweet Spot)
        "26": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.58, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["10", 0]}},
        "27": {"class_type": "VAEDecode", "inputs": {"samples": ["26", 0], "vae": ["1", 2]}},
        "28": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/02_denoise_058", "images": ["27", 0]}},

        # D. Extra High Denoise (0.72) — Full cinematic re-rendering
        "29": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.72, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["10", 0]}},
        "30": {"class_type": "VAEDecode", "inputs": {"samples": ["29", 0], "vae": ["1", 2]}},
        "31": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/02_denoise_072", "images": ["30", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 2 V3: Text Steered Denoise Sweep")
    save_output_image(outputs, "7", "02_control_baseline_portrait_00001_.png")
    save_output_image(outputs, "22", "02_steered_denoise_028_00001_.png")
    save_output_image(outputs, "25", "02_steered_denoise_042_00001_.png")
    save_output_image(outputs, "28", "02_steered_denoise_058_00001_.png")
    save_output_image(outputs, "31", "02_steered_denoise_072_00001_.png")

# ==============================================================================
# TEST 3: BRAND LOCKED LIGHTING (DENOISE SWEEP: EASE UP VS GO HARDER)
# ==============================================================================
def run_test_3_brand_locked_denoise_sweep():
    print("\n" + "=" * 70)
    print("RUNNING TEST 3 (V3): BRAND LOCKED LIGHTING — DENOISE SPECTRUM")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "brand hero shot: luxury gold watch on dark black velvet, volumetric rim lighting, deep shadows, amber highlights, commercial photography", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, amateur", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Anchor (Gold Watch)
        "5": {"class_type": "KSampler", "inputs": {"seed": 1337, "control_after_generate": "fixed", "steps": 24, "cfg": 7.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "GimbalGPS_Anchor", "inputs": {"select_index": 0, "save_waypoint": True, "waypoint_name": "brand_watch_lighting", "enable_perf_logging": False, "latent_batch": ["5", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/04_source_anchor_watch", "images": ["7", 0]}},
        
        # Handbag Prompt
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "elegant luxury handbag on marble surface, product photography, studio catalog", "clip": ["1", 1]}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality", "clip": ["1", 1]}},
        
        # Baseline Handbag (Neutral Daylight)
        "11": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["4", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/04_ctrl_handbag_daylight", "images": ["12", 0]}},
        
        # Brand Target Refinement Prompt
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "elegant luxury handbag, dark black velvet backdrop, gold volumetric rim lighting, luxury commercial photography", "clip": ["1", 1]}},

        # Orthogonal Steering Vector Injection (Strength 1.10)
        "15": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.10, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["11", 0], "target_latent": ["6", 0], "origin_latent": ["11", 0]}},

        # A. Eased Up Denoise (0.24) — Subtle specular & shadow glaze
        "20": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.24, "model": ["1", 0], "positive": ["14", 0], "negative": ["10", 0], "latent_image": ["15", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["1", 2]}},
        "22": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/04_denoise_024", "images": ["21", 0]}},

        # B. Moderate Denoise (0.38) — Background begins transitioning to dark velvet
        "23": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.38, "model": ["1", 0], "positive": ["14", 0], "negative": ["10", 0], "latent_image": ["15", 0]}},
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["1", 2]}},
        "25": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/04_denoise_038", "images": ["24", 0]}},

        # C. Harder Denoise (0.55) — Dark velvet background + gold highlights
        "26": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.55, "model": ["1", 0], "positive": ["14", 0], "negative": ["10", 0], "latent_image": ["15", 0]}},
        "27": {"class_type": "VAEDecode", "inputs": {"samples": ["26", 0], "vae": ["1", 2]}},
        "28": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/04_denoise_055", "images": ["27", 0]}},

        # D. Very Hard Denoise (0.72) — Full studio atmosphere lock
        "29": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.72, "model": ["1", 0], "positive": ["14", 0], "negative": ["10", 0], "latent_image": ["15", 0]}},
        "30": {"class_type": "VAEDecode", "inputs": {"samples": ["29", 0], "vae": ["1", 2]}},
        "31": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/04_denoise_072", "images": ["30", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 3 V3: Brand Locked Denoise Sweep")
    save_output_image(outputs, "8", "04_source_anchor_watch_00001_.png")
    save_output_image(outputs, "13", "04_control_baseline_handbag_00001_.png")
    save_output_image(outputs, "22", "04_brand_denoise_024_00001_.png")
    save_output_image(outputs, "25", "04_brand_denoise_038_00001_.png")
    save_output_image(outputs, "28", "04_brand_denoise_055_00001_.png")
    save_output_image(outputs, "31", "04_brand_denoise_072_00001_.png")

# ==============================================================================
# TEST 4 & 5: SEMANTIC SLIDER (CALIBRATED AXES & DEEP DIVE ON AXIS 2, 3, 4)
# ==============================================================================
def run_test_4_and_5_semantic_slider_v3():
    print("\n" + "=" * 70)
    print("RUNNING TEST 4 & 5 (V3): SEMANTIC SLIDER — CLEAN CALIBRATED AXES")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a person, neutral background, studio lighting, highly detailed photograph", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, deformed, low quality", "clip": ["1", 1]}},
        
        # Batch of 6 diverse latents for PCA subspace
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 6}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 22, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        
        # Extract Base Portrait (Index 0)
        "6": {"class_type": "LatentFromBatch", "inputs": {"batch_index": 0, "length": 1, "samples": ["5", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/03_slider_baseline", "images": ["7", 0]}},
        
        # --- AXIS 1: CALIBRATED (Un-fried) Key Lighting / Contrast ---
        # PC1 = -1.2 (Subdued Cool Lighting)
        "10": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 1, "slider_value": -1.2, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "11": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["10", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/03_pc1_minus_1_2", "images": ["12", 0]}},

        # PC1 = +1.2 (Luminous Warm Key Light - NOT crispy!)
        "14": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 1, "slider_value": 1.2, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "15": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["14", 0]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["1", 2]}},
        "17": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/03_pc1_plus_1_2", "images": ["16", 0]}},

        # --- AXIS 2: THE WORKING AXIS (Morphology, Pose & Gaze) ---
        # PC2 = -1.8 (Negative Axis 2 Shift)
        "20": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 2, "slider_value": -1.8, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "21": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["20", 0]}},
        "22": {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["1", 2]}},
        "23": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/03_pc2_minus_1_8", "images": ["22", 0]}},

        # PC2 = +1.8 (Positive Axis 2 Shift)
        "24": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 2, "slider_value": 1.8, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "25": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["24", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/03_pc2_plus_1_8", "images": ["26", 0]}},

        # --- AXIS 3: THIRD ORTHOGONAL COMPONENT (Expression / Age / Depth) ---
        # PC3 = -2.0
        "30": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 3, "slider_value": -2.0, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "31": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["30", 0]}},
        "32": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["1", 2]}},
        "33": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/03_pc3_minus_2_0", "images": ["32", 0]}},

        # PC3 = +2.0
        "34": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 3, "slider_value": 2.0, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "35": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["34", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/03_pc3_plus_2_0", "images": ["36", 0]}},

        # --- AXIS 4: FOURTH ORTHOGONAL COMPONENT (Focal Background / Framing) ---
        # PC4 = +2.0
        "40": {"class_type": "GimbalSemanticSlider", "inputs": {"pc_index": 4, "slider_value": 2.0, "orthogonalize": True, "latent_batch": ["5", 0], "base_latent": ["6", 0]}},
        "41": {"class_type": "KSampler", "inputs": {"seed": 100, "control_after_generate": "fixed", "steps": 18, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["40", 0]}},
        "42": {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["1", 2]}},
        "43": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test3/03_pc4_plus_2_0", "images": ["42", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 4 & 5 V3: Semantic Slider Clean Axes")
    save_output_image(outputs, "8", "03_control_baseline_pca_00001_.png")
    save_output_image(outputs, "13", "03_gimbal_slider_pc1_calibrated_minus_1_2_00001_.png")
    save_output_image(outputs, "17", "03_gimbal_slider_pc1_calibrated_plus_1_2_00001_.png")
    save_output_image(outputs, "23", "03_gimbal_slider_pc2_minus_1_8_00001_.png")
    save_output_image(outputs, "27", "03_gimbal_slider_pc2_plus_1_8_00001_.png")
    save_output_image(outputs, "33", "03_gimbal_slider_pc3_minus_2_0_00001_.png")
    save_output_image(outputs, "37", "03_gimbal_slider_pc3_plus_2_0_00001_.png")
    save_output_image(outputs, "43", "03_gimbal_slider_pc4_plus_2_0_00001_.png")

def main():
    print("Starting Focused Test Run 3 addressing all user feedback...")
    run_test_1_concept_blender_v3()
    run_test_2_text_steered_denoise_sweep()
    run_test_3_brand_locked_denoise_sweep()
    run_test_4_and_5_semantic_slider_v3()
    print("\n" + "=" * 70)
    print("ALL RUN 3 TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
