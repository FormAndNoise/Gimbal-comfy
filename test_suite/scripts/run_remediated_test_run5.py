import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/comparisons_v5"
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
# TEST 1 (V5): CONCEPT BLENDER — STEP 0 NOISE-SPACE SLERP
# ==============================================================================
def run_test_1_noise_slerp_v5():
    print("\n" + "=" * 70)
    print("RUNNING TEST 1 (V5): CONCEPT BLENDER — STEP 0 NOISE-SPACE SLERP")
    print("=" * 70)
    
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a dense misty forest, ancient mossy trees, dappled sunlight, masterpiece, photorealistic, sharp focus", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "a towering snowy mountain peak, frozen tundra, crisp winter air, masterpiece, photorealistic, sharp focus", "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy", "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "majestic snowy mountain peak rising above ancient misty evergreen forest, winter alpine landscape, masterpiece, photorealistic", "clip": ["1", 1]}},
        
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Control A (Forest, Seed 42, 24 steps)
        "10": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 24, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/01_ctrl_A_forest", "images": ["11", 0]}},
        
        # Control B (Mountain, Seed 1042, 24 steps)
        "13": {"class_type": "KSampler", "inputs": {"seed": 1042, "control_after_generate": "fixed", "steps": 24, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}},
        "15": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/01_ctrl_B_mountain", "images": ["14", 0]}},

        # Noise generation for Seed 42 and Seed 1042
        "20": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 24, "cfg": 1.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "21": {"class_type": "KSampler", "inputs": {"seed": 1042, "control_after_generate": "fixed", "steps": 24, "cfg": 1.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},

        # 1. 35% Noise Blend (Alpine Valley Forest)
        "30": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.35, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["21", 0], "origin_latent": ["20", 0]}},
        "31": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 24, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.90, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["30", 0]}},
        "32": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["1", 2]}},
        "33": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/01_slerp_noise_35pct", "images": ["32", 0]}},

        # 2. 50% Noise Blend (Balanced Alpine Scene)
        "34": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.50, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["21", 0], "origin_latent": ["20", 0]}},
        "35": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 24, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.90, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["34", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/01_slerp_noise_50pct", "images": ["36", 0]}},

        # 3. 65% Noise Blend (Mountain Dominant Alpine Scene)
        "38": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.65, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["21", 0], "origin_latent": ["20", 0]}},
        "39": {"class_type": "KSampler", "inputs": {"seed": 42, "control_after_generate": "fixed", "steps": 24, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.90, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["38", 0]}},
        "40": {"class_type": "VAEDecode", "inputs": {"samples": ["39", 0], "vae": ["1", 2]}},
        "41": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/01_slerp_noise_65pct", "images": ["40", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 1 V5: Concept Blender (Noise-Space SLERP)")
    save_output_image(outputs, "12", "01_ctrl_A_forest_v5.png")
    save_output_image(outputs, "15", "01_ctrl_B_mountain_v5.png")
    save_output_image(outputs, "33", "01_slerp_noise_35pct_v5.png")
    save_output_image(outputs, "37", "01_slerp_noise_50pct_v5.png")
    save_output_image(outputs, "41", "01_slerp_noise_65pct_v5.png")

# ==============================================================================
# TEST 2 (V5): TEXT-STEERED DIFFUSION — STABILIZED & DE-POSTERIZED
# ==============================================================================
def run_test_2_text_steered_v5():
    print("\n" + "=" * 70)
    print("RUNNING TEST 2 (V5): TEXT-STEERED DIFFUSION — STABILIZED & LOW-CFG")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman, soft natural daylight, simple clean background, photorealistic portrait photography, fine details", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, deformed, low quality, cartoon, posterized, heavy contours, dark outlines", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Base portrait (Seed 789)
        "5": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 22, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/02_ctrl_baseline", "images": ["6", 0]}},
        
        # CrossModal Bridge for Neon/Amber Lighting Vector
        "8": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "cinematic moody warm golden glowing amber rim lighting, neon shadows", "mapping_mode": "Keyword_Heuristics", "base_latent": ["5", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman, dramatic cinematic golden hour lighting, amber rim lighting, photorealistic photograph, 8k", "clip": ["1", 1]}},
        
        # Steering Vector (Strength 0.90)
        "10": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.90, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["5", 0], "target_latent": ["8", 0], "origin_latent": ["8", 1]}},

        # Stabilizer
        "15": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["10", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},

        # 1. Denoise 0.42 (Controlled, Natural Skin, Rich Amber Rim Light, CFG 3.8)
        "20": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 20, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.42, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["15", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["1", 2]}},
        "22": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/02_steered_denoise_042", "images": ["21", 0]}},

        # 2. Denoise 0.55 (Rich Cinematic Amber Glow with Clean Skin)
        "23": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 20, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.55, "model": ["1", 0], "positive": ["9", 0], "negative": ["3", 0], "latent_image": ["15", 0]}},
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["1", 2]}},
        "25": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/02_steered_denoise_055", "images": ["24", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 2 V5: Text Steered Stabilized")
    save_output_image(outputs, "7", "02_ctrl_baseline_portrait_v5.png")
    save_output_image(outputs, "22", "02_steered_denoise_042_v5.png")
    save_output_image(outputs, "25", "02_steered_denoise_055_v5.png")

# ==============================================================================
# TEST 3 (V5): BRAND LOCKED LIGHTING — CLEAN VELVET TRANSFER
# ==============================================================================
def run_test_3_brand_locked_v5():
    print("\n" + "=" * 70)
    print("RUNNING TEST 3 (V5): BRAND LOCKED LIGHTING — CLEAN VELVET TRANSFER")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "brand hero shot: luxury gold watch on dark black velvet, volumetric rim lighting, deep shadows, amber highlights, commercial photography, sharp focus", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, amateur, cartoon, white marble, cracks, logo", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Watch Anchor (Seed 1337)
        "5": {"class_type": "KSampler", "inputs": {"seed": 1337, "control_after_generate": "fixed", "steps": 24, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "GimbalGPS_Anchor", "inputs": {"select_index": 0, "save_waypoint": True, "waypoint_name": "brand_watch_lighting_v5", "enable_perf_logging": False, "latent_batch": ["5", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/04_source_anchor_watch", "images": ["7", 0]}},
        
        # Handbag Baseline (Seed 4242)
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "elegant luxury handbag, product photography, studio catalog, sharp focus", "clip": ["1", 1]}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark", "clip": ["1", 1]}},
        "11": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 22, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["4", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/04_ctrl_handbag_daylight", "images": ["12", 0]}},
        
        # Target Prompt: Dark Velvet Backdrop & Gold Rim Light
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "elegant luxury handbag resting on dark black velvet cloth, volumetric gold studio rim lighting, deep shadows, commercial product photography", "clip": ["1", 1]}},

        # Orthogonal Steering Vector Injection
        "15": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.85, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["11", 0], "target_latent": ["6", 0], "origin_latent": ["11", 0]}},

        # Stabilizer
        "16": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["15", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},

        # Denoise 0.50 (CFG 3.8 to eliminate posterization and marble cracks)
        "20": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 20, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.50, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["16", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["1", 2]}},
        "22": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/04_brand_denoise_050", "images": ["21", 0]}},

        # Denoise 0.60 (Deeper Black Velvet Atmosphere)
        "23": {"class_type": "KSampler", "inputs": {"seed": 4242, "control_after_generate": "fixed", "steps": 20, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.60, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["16", 0]}},
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["1", 2]}},
        "25": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/04_brand_denoise_060", "images": ["24", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 3 V5: Brand Locked Lighting")
    save_output_image(outputs, "8", "04_source_anchor_watch_v5.png")
    save_output_image(outputs, "13", "04_ctrl_baseline_handbag_v5.png")
    save_output_image(outputs, "22", "04_brand_denoise_050_v5.png")
    save_output_image(outputs, "25", "04_brand_denoise_060_v5.png")

# ==============================================================================
# TEST 4 (V5): CONCEPT VECTOR ANALOGY — CHANNEL-MEAN DE-GHOSTING
# ==============================================================================
def run_test_4_concept_analogy_v5():
    print("\n" + "=" * 70)
    print("RUNNING TEST 4 (V5): CONCEPT VECTOR ANALOGY — CHANNEL-MEAN DE-GHOSTING")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman with a warm happy joyful smile, studio lighting, highly detailed photograph", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman with a neutral serious expression, studio lighting, highly detailed photograph", "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman wearing stylish dark sunglasses, studio lighting, highly detailed photograph", "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, deformed, low quality, cartoon, phantom, double face, extra eyes", "clip": ["1", 1]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Base Subject C (Seed 789, unsteered neutral portrait)
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman, soft neutral studio lighting, photograph, natural skin texture", "clip": ["1", 1]}},
        "11": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 22, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["10", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/03_analogy_base_subject", "images": ["12", 0]}},

        # Source Concept A: Smiling (Seed 999)
        "20": {"class_type": "KSampler", "inputs": {"seed": 999, "control_after_generate": "fixed", "steps": 22, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        # Source Concept B: Neutral (Seed 999)
        "21": {"class_type": "KSampler", "inputs": {"seed": 999, "control_after_generate": "fixed", "steps": 22, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        # Source Concept D: Sunglasses (Seed 999)
        "22": {"class_type": "KSampler", "inputs": {"seed": 999, "control_after_generate": "fixed", "steps": 22, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},

        # Positive Target Prompt for Refinement
        "25": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman with a warm happy joyful smile, studio lighting, natural skin texture, sharp focus", "clip": ["1", 1]}},
        "26": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait of a woman wearing stylish dark sunglasses, studio lighting, natural skin texture, sharp focus", "clip": ["1", 1]}},

        # 1. Smile Analogy with Channel_Mean
        "30": {"class_type": "GimbalVectorAnalogy", "inputs": {"strength": 1.0, "spatial_mode": "Channel_Mean", "ortho_project": True, "preserve_norm": True, "concept_A": ["20", 0], "concept_B": ["21", 0], "concept_C": ["11", 0]}},
        "31": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 20, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.40, "model": ["1", 0], "positive": ["25", 0], "negative": ["5", 0], "latent_image": ["30", 0]}},
        "32": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["1", 2]}},
        "33": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/03_analogy_smile_channel_mean", "images": ["32", 0]}},

        # 2. Sunglasses Analogy with Channel_Mean
        "40": {"class_type": "GimbalVectorAnalogy", "inputs": {"strength": 1.2, "spatial_mode": "Channel_Mean", "ortho_project": True, "preserve_norm": True, "concept_A": ["22", 0], "concept_B": ["21", 0], "concept_C": ["11", 0]}},
        "41": {"class_type": "KSampler", "inputs": {"seed": 789, "control_after_generate": "fixed", "steps": 20, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.45, "model": ["1", 0], "positive": ["26", 0], "negative": ["5", 0], "latent_image": ["40", 0]}},
        "42": {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["1", 2]}},
        "43": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_Test5/03_analogy_sunglasses_channel_mean", "images": ["42", 0]}},
    }

    outputs = queue_and_wait(wf, "Test 4 V5: Concept Vector Analogy Channel-Mean")
    save_output_image(outputs, "13", "03_analogy_base_subject_v5.png")
    save_output_image(outputs, "33", "03_analogy_smile_channel_mean_v5.png")
    save_output_image(outputs, "43", "03_analogy_sunglasses_channel_mean_v5.png")

def main():
    print("Starting Run 5 (Remediated Production Suite)...")
    # Test 1 already completed, run Tests 2, 3, 4
    run_test_2_text_steered_v5()
    run_test_3_brand_locked_v5()
    run_test_4_concept_analogy_v5()
    print("\n" + "=" * 70)
    print("ALL RUN 5 REMEDIATION BENCHMARKS COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
