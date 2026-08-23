import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/architectural_showcase_v2"
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
# 1. UPGRADED GIMBAL 08: NOISE-SPACE & CALIBRATED HARMONIC ORBITER (DRAMATIC TOUR)
# ==============================================================================
def test_upgraded_08_harmonic_orbiter():
    print("\n" + "=" * 70)
    print("TESTING UPGRADED GIMBAL 08: DRAMATIC HARMONIC ORBITER (4 DISTINCT PHASES)")
    print("=" * 70)
    
    # We test Step-0 Noise Spherical Orbit with GimbalCircularOrbit across 4 key phases:
    # Golden Hour Sunset (Phase 0) -> Midnight Glowing Pavilion (Phase 1) -> Misty Blue Dawn (Phase 2) -> Return Loop (Phase 3)
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "monolithic brutalist architectural pavilion on rugged coastal cliff, cantilevered board-formed concrete, glowing warm interior lighting, architectural photography, masterpiece, 8k, sharp focus", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy, deformed", "clip": ["1", 1]}},
        
        # Step 0 Initial Noise Latent
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Base Hero Render
        "10": {"class_type": "KSampler", "inputs": {"seed": 9901, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_V2/08_base_hero", "images": ["11", 0]}},

        # Spherical Orbit on Latent with Strong Geodesic Sweep (radius 1.85, 4 keyframe tour)
        "20": {"class_type": "GimbalCircularOrbit", "inputs": {"steps": 4, "radius": 1.85, "orbit_mode": "Orthogonal_Basis", "preserve_hypersphere_norm": True, "seed": 77, "center_latent": ["10", 0]}},
        "25": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["20", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},

        # Resample at calibrated denoise (0.60) to allow real atmospheric and lighting evolution
        "30": {"class_type": "KSampler", "inputs": {"seed": 9901, "control_after_generate": "fixed", "steps": 22, "cfg": 4.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.60, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["25", 0]}},
        "31": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["1", 2]}},
        "32": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_V2/08_orbit_tour", "images": ["31", 0]}},
    }

    outputs = queue_and_wait(wf, "Upgraded 08: Harmonic Orbiter")
    save_output_image(outputs, "12", "08_v2_hero_base.png")
    save_output_image(outputs, "32", "08_v2_orbit.png")

# ==============================================================================
# 2. UPGRADED GIMBAL 09: BOLD HIGH-CONTRAST MATERIAL MATRIX (LOCKED GEOMETRY)
# ==============================================================================
def test_upgraded_09_material_matrix():
    print("\n" + "=" * 70)
    print("TESTING UPGRADED GIMBAL 09: BOLD HIGH-CONTRAST MATERIAL MATRIX")
    print("=" * 70)

    # Base: Clean Sculptural Brutalist Villa (Seed 3301)
    # Material 1: Midnight Obsidian Stone, Cyan/Neon Glass, Dark Atmosphere
    # Material 2: Gilded Byzantine Gold Leaf, Raw Ochre Travertine, Emerald Accents
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "architectural elevation photograph of a sculptural modernist museum building, white matte concrete facade, geometric cantilevered volumes, clear glass windows, bright studio daylight, architectural digest, sharp focus", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, distorted geometry, watermark", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # 1. Base Geometry Anchor (White Concrete Museum)
        "10": {"class_type": "KSampler", "inputs": {"seed": 3301, "control_after_generate": "fixed", "steps": 24, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_V2/09_mat_01_white_concrete", "images": ["11", 0]}},

        # Prompt 1: Obsidian & Dark Chrome
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "architectural elevation photograph of a sculptural modernist museum building, polished black obsidian stone facade, dark smoked glass, dark moody cinematic lighting, masterpiece", "clip": ["1", 1]}},
        
        # Prompt 2: Byzantine Gold & Travertine
        "15": {"class_type": "CLIPTextEncode", "inputs": {"text": "architectural elevation photograph of a sculptural modernist museum building, shimmering gold leaf panels, warm honey travertine stone facade, warm sunset lighting, luxury architectural monument", "clip": ["1", 1]}},

        # Channel Split (Lock Band A: Geometry channels 0-1; Steer Band B: Chroma channels 2-3)
        "20": {"class_type": "GimbalChannelSplit", "inputs": {"split_index": 2, "latent": ["10", 0]}},

        # Material 1: Steer Chroma Band with CrossModal & Re-merge
        "21": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "polished black obsidian, dark smoked glass, deep shadows, midnight blue", "mapping_mode": "Keyword_Heuristics", "base_latent": ["20", 1]}},
        "22": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.5, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 1], "target_latent": ["21", 0], "origin_latent": ["21", 1]}},
        "23": {"class_type": "GimbalChannelMerge", "inputs": {"latent_band_A": ["20", 0], "latent_band_B": ["22", 0]}},
        "24": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["23", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},
        
        # Render Material 1 (Obsidian) at denoise 0.55
        "25": {"class_type": "KSampler", "inputs": {"seed": 3301, "control_after_generate": "fixed", "steps": 22, "cfg": 4.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.55, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["24", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_V2/09_mat_02_obsidian", "images": ["26", 0]}},

        # Material 2: Gold & Travertine
        "31": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "shimmering gold leaf facade, warm honey travertine stone, golden amber highlights", "mapping_mode": "Keyword_Heuristics", "base_latent": ["20", 1]}},
        "32": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.5, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 1], "target_latent": ["31", 0], "origin_latent": ["31", 1]}},
        "33": {"class_type": "GimbalChannelMerge", "inputs": {"latent_band_A": ["20", 0], "latent_band_B": ["32", 0]}},
        "34": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["33", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},
        
        # Render Material 2 (Gold/Travertine) at denoise 0.55
        "35": {"class_type": "KSampler", "inputs": {"seed": 3301, "control_after_generate": "fixed", "steps": 22, "cfg": 4.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.55, "model": ["1", 0], "positive": ["15", 0], "negative": ["3", 0], "latent_image": ["34", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_V2/09_mat_03_gold_travertine", "images": ["36", 0]}},
    }

    outputs = queue_and_wait(wf, "Upgraded 09: Material Matrix")
    save_output_image(outputs, "12", "09_v2_white_concrete.png")
    save_output_image(outputs, "27", "09_v2_obsidian.png")
    save_output_image(outputs, "37", "09_v2_gold_travertine.png")

# ==============================================================================
# 3. UPGRADED GIMBAL 02: HIGH-AESTHETIC CINEMATIC LIGHTING (NOT DRAB PORTRAIT)
# ==============================================================================
def test_upgraded_02_cinematic_steering():
    print("\n" + "=" * 70)
    print("TESTING UPGRADED GIMBAL 02: HIGH-AESTHETIC DUAL-TONE CYBERPUNK LIGHTING")
    print("=" * 70)

    # Sleek High-End Concept Hypercar or Cyberpunk Hero Portrait with distinct lighting
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # Clean Studio Baseline
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "sleek futuristic concept hypercar in minimalist studio, studio photography, clean reflections, sharp focus, 8k, automotive design", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Base Studio Car (Seed 4412)
        "10": {"class_type": "KSampler", "inputs": {"seed": 4412, "control_after_generate": "fixed", "steps": 24, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_V2/02_car_studio_ctrl", "images": ["11", 0]}},

        # Target 1: Cyberpunk Neon Noir (Dual Cyan & Magenta Volumetric Lighting)
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "sleek futuristic concept hypercar in rain-slicked cyberpunk night city, vibrant volumetric cyan and neon magenta rim lighting, wet asphalt reflections, cinematic atmosphere", "clip": ["1", 1]}},

        "20": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "cyberpunk neon cyan and magenta rim lighting, dark city night, wet street reflections", "mapping_mode": "Keyword_Heuristics", "base_latent": ["10", 0]}},
        "21": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.4, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["10", 0], "target_latent": ["20", 0], "origin_latent": ["20", 1]}},
        "22": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["21", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},
        
        "25": {"class_type": "KSampler", "inputs": {"seed": 4412, "control_after_generate": "fixed", "steps": 22, "cfg": 4.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.52, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["22", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_V2/02_car_steered_cyberpunk", "images": ["26", 0]}},
    }

    outputs = queue_and_wait(wf, "Upgraded 02: Cinematic Car Steering")
    save_output_image(outputs, "12", "02_v2_car_studio_ctrl.png")
    save_output_image(outputs, "27", "02_v2_car_steered_cyberpunk.png")

def main():
    print("Executing Bold Upgraded Visual Benchmarks...")
    test_upgraded_08_harmonic_orbiter()
    test_upgraded_09_material_matrix()
    test_upgraded_02_cinematic_steering()
    print("\n" + "=" * 70)
    print("ALL UPGRADED VISUAL BENCHMARKS COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
