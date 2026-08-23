import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/brand_and_material_v4"
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

# ==============================================================================
# 1. BRAND-LOCKED LIGHTING: LUXURY CRIMSON SLIT-LIGHT TRANSFER
# ==============================================================================
def run_brand_locked_perfume_to_headphones():
    print("\n" + "=" * 70)
    print("RUNNING BRAND-LOCKED LIGHTING: CRIMSON SLIT-LIGHT TRANSFER (PERFUME -> HEADPHONES)")
    print("=" * 70)

    # 1. Anchor Source: Luxury Matte Black Perfume Flask with Razor-Sharp Crimson/Amber Slit Rim Lighting (Seed 9001)
    # 2. Recipient Control: Futuristic Matte Grey Headphones in Flat Overcast Daylight (Seed 6002)
    # 3. GPS Anchor extraction + Orthogonal Steering + Stabilizer (denoise 0.58, CFG 3.8)
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # Source Prompt: Perfume under dramatic crimson slit light
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "luxury minimalist matte black perfume flacon on dark velvet pedestal, sharp dramatic crimson and amber vertical slit rim lighting, deep black shadows, high-end editorial product photography, 8k, sharp focus", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # 1. Render Anchor Source (Perfume)
        "10": {"class_type": "KSampler", "inputs": {"seed": 9001, "control_after_generate": "fixed", "steps": 24, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Brand_V4/01_anchor_perfume_crimson", "images": ["11", 0]}},

        # Recipient Prompt: Headphones in flat daylight
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "sleek sculptural matte grey wireless over-ear headphones sitting on minimalist marble table, flat overcast daylight, product photography, clean white catalog shot, 8k", "clip": ["1", 1]}},
        "15": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # 2. Render Recipient Baseline (Headphones Daylight)
        "20": {"class_type": "KSampler", "inputs": {"seed": 6002, "control_after_generate": "fixed", "steps": 24, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["15", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["1", 2]}},
        "22": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Brand_V4/02_ctrl_headphones_daylight", "images": ["21", 0]}},

        # 3. GPS Anchor extraction from Perfume Anchor & Steering onto Headphones
        "24": {"class_type": "CLIPTextEncode", "inputs": {"text": "sleek sculptural matte grey wireless over-ear headphones, dramatic crimson and amber vertical slit rim lighting, pitch black darkroom background, deep shadows, high-end editorial commercial render, 8k", "clip": ["1", 1]}},
        
        "30": {"class_type": "GimbalGPS_Anchor", "inputs": {"select_index": 0, "save_waypoint": False, "waypoint_name": "brand_crimson_slit_v4", "enable_perf_logging": False, "latent_batch": ["10", 0]}},
        "31": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.5, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["30", 0], "origin_latent": ["4", 0]}},
        "32": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["31", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        # Render Steered Headphones at denoise 0.58
        "35": {"class_type": "KSampler", "inputs": {"seed": 6002, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.58, "model": ["1", 0], "positive": ["24", 0], "negative": ["3", 0], "latent_image": ["32", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Brand_V4/03_steered_headphones_crimson", "images": ["36", 0]}},
    }

    outputs = queue_and_wait(wf, "Brand-Locked Crimson Slit Lighting")
    save_output_image(outputs, "12", "01_anchor_perfume_crimson.png")
    save_output_image(outputs, "22", "02_ctrl_headphones_daylight.png")
    save_output_image(outputs, "37", "03_steered_headphones_crimson.png")

# ==============================================================================
# 2. SUBSPACE MATERIAL MATRIX: HIGH-CONTRAST SCULPTURAL MONOLITH MUTATION
# ==============================================================================
def run_subspace_material_sculpture():
    print("\n" + "=" * 70)
    print("RUNNING SUBSPACE MATERIAL MATRIX: HIGH-CONTRAST SCULPTURAL MONOLITH")
    print("=" * 70)

    # Base: Clean Sculptural Geometric Modernist Monolith (Matte White Plaster)
    # Variant 1: Polished Midnight Obsidian Glass & Cyan Inner Neon Glow
    # Variant 2: Polished 24k Byzantine Gold Leaf & Rich Emerald Malachite
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # Base Prompt: Matte White Plaster Sculpture
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "sculptural geometric modern art monolith, smooth matte white plaster and raw limestone, elegant curves and sharp facet edges, museum pedestal, clean neutral studio lighting, 8k, sharp focus", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, distorted geometry, watermark, noisy", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # 1. Base Sculpture Anchor (Seed 4501)
        "10": {"class_type": "KSampler", "inputs": {"seed": 4501, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Material_V4/01_sculpture_white_plaster", "images": ["11", 0]}},

        # Prompt 1: Jet-Black Polished Obsidian & Cyan Neon Glass
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "sculptural geometric modern art monolith made of polished jet-black obsidian glass with glowing electric cyan neon crystal core, mirror specular reflections, dark luxury studio, 8k, masterpiece", "clip": ["1", 1]}},
        
        # Prompt 2: 24k Byzantine Gold Leaf & Emerald Malachite
        "15": {"class_type": "CLIPTextEncode", "inputs": {"text": "sculptural geometric modern art monolith made of mirror-polished 24k gold leaf and banded deep emerald green malachite stone, opulent gold reflections, warm museum lighting, 8k, masterpiece", "clip": ["1", 1]}},

        # Channel Split (Lock Band A: Geometry channels 0-1; Steer Band B: Chroma/Material channels 2-3)
        "20": {"class_type": "GimbalChannelSplit", "inputs": {"split_index": 2, "latent": ["10", 0]}},

        # Material 1: Steer Chroma Band with CrossModal & Re-merge
        "21": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "polished jet-black obsidian glass, electric cyan neon core, dark midnight reflections", "mapping_mode": "Keyword_Heuristics", "base_latent": ["20", 1]}},
        "22": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 1], "target_latent": ["21", 0], "origin_latent": ["21", 1]}},
        "23": {"class_type": "GimbalChannelMerge", "inputs": {"latent_band_A": ["20", 0], "latent_band_B": ["22", 0]}},
        "24": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["23", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        # Render Material 1 (Obsidian) at denoise 0.55 - 100% Geometry Lock
        "25": {"class_type": "KSampler", "inputs": {"seed": 4501, "control_after_generate": "fixed", "steps": 22, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.55, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["24", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Material_V4/02_sculpture_obsidian_cyan", "images": ["26", 0]}},

        # Material 2: Steer Chroma Band to 24k Gold & Emerald Malachite
        "31": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "mirror-polished 24k gold leaf, banded deep emerald malachite stone, rich golden amber reflections", "mapping_mode": "Keyword_Heuristics", "base_latent": ["20", 1]}},
        "32": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 1], "target_latent": ["31", 0], "origin_latent": ["31", 1]}},
        "33": {"class_type": "GimbalChannelMerge", "inputs": {"latent_band_A": ["20", 0], "latent_band_B": ["32", 0]}},
        "34": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["33", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        # Render Material 2 (Gold / Malachite) at denoise 0.55
        "35": {"class_type": "KSampler", "inputs": {"seed": 4501, "control_after_generate": "fixed", "steps": 22, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.55, "model": ["1", 0], "positive": ["15", 0], "negative": ["3", 0], "latent_image": ["34", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Material_V4/03_sculpture_gold_malachite", "images": ["36", 0]}},
    }

    outputs = queue_and_wait(wf, "Subspace Material Sculpture")
    save_output_image(outputs, "12", "01_sculpture_white_plaster.png")
    save_output_image(outputs, "27", "02_sculpture_obsidian_cyan.png")
    save_output_image(outputs, "37", "03_sculpture_gold_malachite.png")

def main():
    print("Executing High-Contrast Brand Lighting and Subspace Material Tests (Run V4)...")
    run_brand_locked_perfume_to_headphones()
    run_subspace_material_sculpture()
    print("\n" + "=" * 70)
    print("ALL V4 TESTS COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
