import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/bold_brand_material_v5"
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

def test_bold_suites():
    print("\n" + "=" * 70)
    print("TESTING BOLD BRAND LIGHTING & MATERIAL MATRIX TRANSFORMATIONS")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # SUITE 1: BRAND LIGHTING TRANSFER (LUXURY AUDIO MONITOR / SPEAKER)
    # --------------------------------------------------------------------------
    # 1. Daylight Baseline: Minimalist modern architectural bookshelf speaker in flat daylight studio
    # 2. Steered Crimson Slit Noir: Same speaker with dramatic vertical crimson and amber slit rim light, pitch black background
    wf_brand = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # Daylight Baseline Prompt
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "sleek minimalist matte black architectural wireless bookshelf speaker on concrete table, flat neutral daylight, modern industrial design, catalog product photography, sharp focus, 8k", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # 1. Base Daylight Speaker (Seed 3301)
        "10": {"class_type": "KSampler", "inputs": {"seed": 3301, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Brand_V5/01_speaker_daylight_ctrl", "images": ["11", 0]}},

        # Prompt for Crimson Slit Noir Lighting
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "sleek minimalist matte black architectural wireless bookshelf speaker on concrete table, dramatic razor-sharp crimson red and amber gold vertical slit rim lighting, pitch black darkroom background, deep velvety shadows, luxury editorial commercial render, 8k", "clip": ["1", 1]}},

        # Cross-Modal Steering for Slit Lighting
        "20": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "dramatic crimson red and amber gold slit rim lighting, pitch black darkroom background, deep velvet shadows", "mapping_mode": "Keyword_Heuristics", "base_latent": ["10", 0]}},
        "21": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["10", 0], "target_latent": ["20", 0], "origin_latent": ["20", 1]}},
        "22": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["21", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        # Render Steered Speaker at denoise 0.65
        "25": {"class_type": "KSampler", "inputs": {"seed": 3301, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["22", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Brand_V5/02_speaker_crimson_slit", "images": ["26", 0]}},

        # ----------------------------------------------------------------------
        # SUITE 2: BOLD SUBSPACE MATERIAL TRANSFORMATION (ICONIC ARMCHAIR)
        # ----------------------------------------------------------------------
        # 1. Base Control: Minimalist sculptural architectural lounge armchair in raw natural light oat linen / white plaster (Seed 7712)
        # 2. Material 1: Polished jet-black glossy obsidian leather & polished chrome frame
        # 3. Material 2: Gilded 24k gold leaf & royal sapphire blue velvet
        "30": {"class_type": "CLIPTextEncode", "inputs": {"text": "iconic modernist architectural lounge armchair, clean sculptural geometry, raw natural light oat linen fabric and matte birch frame, minimalist studio, architectural digest, sharp focus, 8k", "clip": ["1", 1]}},
        "31": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Render Base Armchair
        "40": {"class_type": "KSampler", "inputs": {"seed": 7712, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["30", 0], "negative": ["3", 0], "latent_image": ["31", 0]}},
        "41": {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["1", 2]}},
        "42": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Material_V5/01_chair_oat_linen_ctrl", "images": ["41", 0]}},

        # Prompt Material 1: Black Obsidian Leather & Chrome
        "44": {"class_type": "CLIPTextEncode", "inputs": {"text": "iconic modernist architectural lounge armchair, clean sculptural geometry, polished jet-black glossy obsidian leather and mirror-polished chrome steel frame, luxury dark studio, sharp focus, 8k", "clip": ["1", 1]}},
        
        # Prompt Material 2: 24k Gold & Sapphire Blue Velvet
        "45": {"class_type": "CLIPTextEncode", "inputs": {"text": "iconic modernist architectural lounge armchair, clean sculptural geometry, rich royal sapphire blue velvet upholstery and mirror-polished 24k solid gold frame, opulent lighting, sharp focus, 8k", "clip": ["1", 1]}},

        # Material 1: CrossModal + Stabilizer (denoise 0.65)
        "50": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "jet-black glossy obsidian leather, mirror-polished chrome frame, dark studio reflections", "mapping_mode": "Keyword_Heuristics", "base_latent": ["40", 0]}},
        "51": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["40", 0], "target_latent": ["50", 0], "origin_latent": ["50", 1]}},
        "52": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["51", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        "55": {"class_type": "KSampler", "inputs": {"seed": 7712, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["44", 0], "negative": ["3", 0], "latent_image": ["52", 0]}},
        "56": {"class_type": "VAEDecode", "inputs": {"samples": ["55", 0], "vae": ["1", 2]}},
        "57": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Material_V5/02_chair_black_leather_chrome", "images": ["56", 0]}},

        # Material 2: Sapphire Velvet & 24k Gold (denoise 0.65)
        "60": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "rich royal sapphire blue velvet upholstery, mirror-polished 24k solid gold frame, golden reflections", "mapping_mode": "Keyword_Heuristics", "base_latent": ["40", 0]}},
        "61": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["40", 0], "target_latent": ["60", 0], "origin_latent": ["60", 1]}},
        "62": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["61", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        "65": {"class_type": "KSampler", "inputs": {"seed": 7712, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["45", 0], "negative": ["3", 0], "latent_image": ["62", 0]}},
        "66": {"class_type": "VAEDecode", "inputs": {"samples": ["65", 0], "vae": ["1", 2]}},
        "67": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Material_V5/03_chair_sapphire_gold", "images": ["66", 0]}},
    }

    outputs = queue_and_wait(wf_brand, "Bold Brand & Material Suite")
    save_output_image(outputs, "12", "01_speaker_daylight_ctrl.png")
    save_output_image(outputs, "27", "02_speaker_crimson_slit.png")
    save_output_image(outputs, "42", "03_chair_oat_linen_ctrl.png")
    save_output_image(outputs, "57", "04_chair_black_leather_chrome.png")
    save_output_image(outputs, "67", "05_chair_sapphire_gold.png")

if __name__ == "__main__":
    test_bold_suites()
