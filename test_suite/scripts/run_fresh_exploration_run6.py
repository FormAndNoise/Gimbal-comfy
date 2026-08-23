"""
Gimbal Latent Flight Instruments — Fresh Exploration Test Suite (Run 6)
Executes bold, newly designed test domains across concept blending, cross-modal
atmospheric steering, closed-loop geodesic tours, and dual-band material matrices.
"""

import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/fresh_run6"
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
                    return item.get("outputs", {}), elapsed
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
            raise RuntimeError(f"Failed to fetch image {fname}: {r.status_code}")
        
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
# SUITE 1: CONCEPT BLENDER — STEP 0 NOISE-SPACE SLERP (SOLARPUNK BIOME)
# ==============================================================================
def run_suite_1_solarpunk_slerp():
    print("\n" + "=" * 75)
    print("SUITE 1: CONCEPT BLENDER — STEP 0 NOISE SLERP (CYBERPUNK SPIRE <-> REDWOOD BIOME)")
    print("=" * 75)
    
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # Prompt A: Cyberpunk Megacity Spire at twilight
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "towering cyberpunk megacity skyscraper spire, complex neon geometry, glowing holographic signage, rain-slicked futuristic architecture, twilight, masterpiece, 8k, photorealistic", "clip": ["1", 1]}},
        
        # Prompt B: Ancient Redwood Biome Forest
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "dense ancient giant redwood forest, towering moss-covered sequoia trees, lush ferns, dappled morning sunlight filtering through canopy, pristine nature, masterpiece, 8k, photorealistic", "clip": ["1", 1]}},
        
        # Negative
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy, distorted", "clip": ["1", 1]}},
        
        # Hybrid Prompt for Slerp reconstruction
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "solarpunk architectural fusion of towering organic skyscraper spire seamlessly integrated into ancient redwood biome, glowing warm windows among mossy branches, masterpiece, photorealistic, 8k", "clip": ["1", 1]}},
        
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Baseline A: Cyberpunk Spire (Seed 5544)
        "10": {"class_type": "KSampler", "inputs": {"seed": 5544, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/01_ctrl_A_cyber_spire", "images": ["11", 0]}},
        
        # Baseline B: Redwood Forest (Seed 6655)
        "13": {"class_type": "KSampler", "inputs": {"seed": 6655, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}},
        "15": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/01_ctrl_B_redwood_forest", "images": ["14", 0]}},

        # Noise Generators for Seeds
        "20": {"class_type": "KSampler", "inputs": {"seed": 5544, "control_after_generate": "fixed", "steps": 24, "cfg": 1.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "21": {"class_type": "KSampler", "inputs": {"seed": 6655, "control_after_generate": "fixed", "steps": 24, "cfg": 1.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},

        # Slerp Noise 35%
        "30": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.35, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["21", 0], "origin_latent": ["20", 0]}},
        "31": {"class_type": "KSampler", "inputs": {"seed": 5544, "control_after_generate": "fixed", "steps": 24, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.90, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["30", 0]}},
        "32": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["1", 2]}},
        "33": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/01_slerp_noise_35pct", "images": ["32", 0]}},

        # Slerp Noise 50% (Balanced Solarpunk Habitat)
        "34": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.50, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["21", 0], "origin_latent": ["20", 0]}},
        "35": {"class_type": "KSampler", "inputs": {"seed": 5544, "control_after_generate": "fixed", "steps": 24, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.90, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["34", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/01_slerp_noise_50pct", "images": ["36", 0]}},

        # Slerp Noise 65%
        "38": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.65, "mode": "Slerp", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": False, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["21", 0], "origin_latent": ["20", 0]}},
        "39": {"class_type": "KSampler", "inputs": {"seed": 5544, "control_after_generate": "fixed", "steps": 24, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.90, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["38", 0]}},
        "40": {"class_type": "VAEDecode", "inputs": {"samples": ["39", 0], "vae": ["1", 2]}},
        "41": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/01_slerp_noise_65pct", "images": ["40", 0]}},
    }
    
    outputs, elapsed = queue_and_wait(wf, "Suite1_Solarpunk_SLERP")
    save_output_image(outputs, "12", "01_ctrl_A_cyber_spire.png")
    save_output_image(outputs, "15", "01_ctrl_B_redwood_forest.png")
    save_output_image(outputs, "33", "01_slerp_noise_35pct.png")
    save_output_image(outputs, "37", "01_slerp_noise_50pct.png")
    save_output_image(outputs, "41", "01_slerp_noise_65pct.png")

# ==============================================================================
# SUITE 2: CROSS-MODAL STEERING — LUXURY CHRONOGRAPH ATMOSPHERIC DUALITY
# ==============================================================================
def run_suite_2_horology_steering():
    print("\n" + "=" * 75)
    print("SUITE 2: CROSS-MODAL STEERING — LUXURY CHRONOGRAPH (OCEANIC & FORGE STEERING)")
    print("=" * 75)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # Baseline: Minimalist matte titanium luxury chronograph watch in clean daylight studio
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "hero product macro shot of a sleek matte titanium luxury chronograph wristwatch on stone pedestal, neutral daylight studio lighting, clean industrial design, crisp typography, sharp focus, 8k", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy, deformed", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # 1. Base Control Watch (Seed 8821)
        "10": {"class_type": "KSampler", "inputs": {"seed": 8821, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/02_watch_daylight_baseline", "images": ["11", 0]}},

        # Target 1: Deep Oceanic Bioluminescence
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "hero product macro shot of a sleek matte titanium luxury chronograph wristwatch on stone pedestal, deep underwater abyssal darkness, glowing cyan and electric sapphire bioluminescent particles, soft ethereal aquatic caustics, luxury commercial render, 8k", "clip": ["1", 1]}},
        "20": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "deep abyssal underwater darkness, glowing cyan and electric sapphire bioluminescence, ethereal aquatic caustics", "mapping_mode": "Keyword_Heuristics", "base_latent": ["10", 0]}},
        "21": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.5, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["10", 0], "target_latent": ["20", 0], "origin_latent": ["20", 1]}},
        "22": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["21", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        "25": {"class_type": "KSampler", "inputs": {"seed": 8821, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.60, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["22", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/02_watch_steered_bioluminescent", "images": ["26", 0]}},

        # Target 2: Volcanic Obsidian Forge Lighting
        "30": {"class_type": "CLIPTextEncode", "inputs": {"text": "hero product macro shot of a sleek matte titanium luxury chronograph wristwatch on stone pedestal, darkroom forge, intense glowing molten amber magma reflections, razor-sharp orange rim lighting, pitch black background with floating embers, commercial photography, 8k", "clip": ["1", 1]}},
        "31": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "darkroom forge, glowing molten amber magma reflections, intense orange rim lighting, floating embers, pitch black background", "mapping_mode": "Keyword_Heuristics", "base_latent": ["10", 0]}},
        "32": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.5, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["10", 0], "target_latent": ["31", 0], "origin_latent": ["31", 1]}},
        "33": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["32", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        "35": {"class_type": "KSampler", "inputs": {"seed": 8821, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.60, "model": ["1", 0], "positive": ["30", 0], "negative": ["3", 0], "latent_image": ["33", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/02_watch_steered_volcanic_forge", "images": ["36", 0]}},
    }
    
    outputs, elapsed = queue_and_wait(wf, "Suite2_Horology_Steering")
    save_output_image(outputs, "12", "02_watch_daylight_baseline.png")
    save_output_image(outputs, "27", "02_watch_steered_bioluminescent.png")
    save_output_image(outputs, "37", "02_watch_steered_volcanic_forge.png")

# ==============================================================================
# SUITE 3: HARMONIC ORBITER — 6-POINT PARAMETRIC DESERT PAVILION TOUR
# ==============================================================================
def run_suite_3_parametric_desert_orbit():
    print("\n" + "=" * 75)
    print("SUITE 3: HARMONIC ORBITER — 6-KEYFRAME PARAMETRIC DESERT PAVILION TOUR")
    print("=" * 75)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "futuristic parametric sculptural glass and sandstone desert pavilion emerging from rolling sand dunes, dramatic curvilinear architecture, warm golden hour sunlight, architectural photography, masterpiece, sharp focus, 8k", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy, deformed", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # 6-step Geodesic Orbit (0, 60, 120, 180, 240, 300 deg)
        "20": {"class_type": "GimbalCircularOrbit", "inputs": {"steps": 6, "radius": 0.96, "orbit_mode": "Orthogonal_Basis", "preserve_hypersphere_norm": True, "seed": 3141, "center_latent": ["4", 0]}},
        "25": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["20", 0], "truncation_psi": 0.92, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},
        
        "30": {"class_type": "KSampler", "inputs": {"seed": 9191, "control_after_generate": "fixed", "steps": 24, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["25", 0]}},
        "31": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["1", 2]}},
        "32": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/08_desert_orbit_tour", "images": ["31", 0]}},
    }
    
    outputs, elapsed = queue_and_wait(wf, "Suite3_Parametric_Desert_Orbit")
    save_output_image(outputs, "32", "08_desert_orbit_tour.png")

# ==============================================================================
# SUITE 4: SUBSPACE MATERIAL MATRIX — ICONIC SCULPTURAL ARMCHAIR MUTATIONS
# ==============================================================================
def run_suite_4_armchair_material_matrix():
    print("\n" + "=" * 75)
    print("SUITE 4: SUBSPACE MATERIAL MATRIX — SCULPTURAL ARMCHAIR METAMORPHOSIS")
    print("=" * 75)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # 1. Base Control: Sculptural curved Bauhaus armchair in raw natural oat linen and light ash wood
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "minimalist architectural sculptural curved lounge armchair in raw natural oat linen and blonde ash wood frame, concrete architectural interior, soft neutral daylight, catalog design photography, sharp focus, 8k", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, distorted geometry, watermark, noisy", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        "10": {"class_type": "KSampler", "inputs": {"seed": 9201, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/09_chair_ctrl_oat_linen", "images": ["11", 0]}},

        # Split SDXL Latent (Channels 0..1 = Geometry Silhouette, Channels 2..3 = Material Chroma)
        "20": {"class_type": "GimbalChannelSplit", "inputs": {"split_index": 2, "latent": ["10", 0]}},

        # Target 1: Mirror-Polished Liquid Chrome & Smoked Glass
        "21": {"class_type": "CLIPTextEncode", "inputs": {"text": "minimalist architectural sculptural curved lounge armchair made entirely of mirror polished liquid chrome and dark smoked glass, concrete architectural interior, glossy studio reflections, luxury industrial design render, 8k", "clip": ["1", 1]}},
        "22": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "mirror polished liquid chrome, dark smoked glass, mirror reflections, glossy metallic surface", "mapping_mode": "Keyword_Heuristics", "base_latent": ["20", 1]}},
        "23": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.85, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 1], "target_latent": ["22", 0], "origin_latent": ["22", 1]}},
        "24": {"class_type": "GimbalChannelMerge", "inputs": {"latent_band_A": ["20", 0], "latent_band_B": ["23", 0]}},
        "25": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["24", 0], "truncation_psi": 0.90, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},
        
        "26": {"class_type": "KSampler", "inputs": {"seed": 9201, "control_after_generate": "fixed", "steps": 20, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.38, "model": ["1", 0], "positive": ["21", 0], "negative": ["3", 0], "latent_image": ["25", 0]}},
        "27": {"class_type": "VAEDecode", "inputs": {"samples": ["26", 0], "vae": ["1", 2]}},
        "28": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/09_chair_steered_liquid_chrome", "images": ["27", 0]}},

        # Target 2: High-Gloss Oxblood Velvet & Brushed Rose Gold Trim
        "31": {"class_type": "CLIPTextEncode", "inputs": {"text": "minimalist architectural sculptural curved lounge armchair in rich deep oxblood red velvet and brushed rose gold metal frame, concrete architectural interior, warm ambient specular highlights, luxury bespoke furniture, 8k", "clip": ["1", 1]}},
        "32": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "rich deep oxblood red velvet, brushed rose gold metal trim, warm velvet sheen, luxury bespoke upholstery", "mapping_mode": "Keyword_Heuristics", "base_latent": ["20", 1]}},
        "33": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 0.85, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 1], "target_latent": ["32", 0], "origin_latent": ["32", 1]}},
        "34": {"class_type": "GimbalChannelMerge", "inputs": {"latent_band_A": ["20", 0], "latent_band_B": ["33", 0]}},
        "35": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["34", 0], "truncation_psi": 0.90, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},
        
        "36": {"class_type": "KSampler", "inputs": {"seed": 9201, "control_after_generate": "fixed", "steps": 20, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.38, "model": ["1", 0], "positive": ["31", 0], "negative": ["3", 0], "latent_image": ["35", 0]}},
        "37": {"class_type": "VAEDecode", "inputs": {"samples": ["36", 0], "vae": ["1", 2]}},
        "38": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6/09_chair_steered_oxblood_velvet", "images": ["37", 0]}},
    }
    
    outputs, elapsed = queue_and_wait(wf, "Suite4_Armchair_Material_Matrix")
    save_output_image(outputs, "12", "09_chair_ctrl_oat_linen.png")
    save_output_image(outputs, "28", "09_chair_steered_liquid_chrome.png")
    save_output_image(outputs, "38", "09_chair_steered_oxblood_velvet.png")

def main():
    print("=" * 80)
    print("    GIMBAL LATENT FLIGHT INSTRUMENTS — FRESH TEST SUITE (RUN 6)")
    print("=" * 80)
    
    run_suite_1_solarpunk_slerp()
    run_suite_2_horology_steering()
    run_suite_3_parametric_desert_orbit()
    run_suite_4_armchair_material_matrix()
    
    print("\n" + "=" * 80)
    print("    ALL RUN 6 SUITES EXECUTED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
