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

# ==============================================================================
# SECTION 3: BRAND-LOCKED / GPS LIGHTING TRANSFER
# Japanese Zen Pavilion Amber Lantern Glow -> Modern Penthouse Lounge
# ==============================================================================
def run_section_3_lighting_transfer():
    print("\n" + "=" * 70)
    print("RUNNING SECTION 3: GPS LIGHTING TRANSFER (ZEN LANTERN GLOW -> PENTHOUSE)")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # 1. Source Lighting Anchor: Traditional Kyoto Zen Pavilion at twilight with glowing amber shoji lanterns, dark polished cedar, reflections
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "luxurious Kyoto zen pavilion interior at twilight, warm glowing amber shoji paper lanterns, dark polished cedar wood floor reflections, moody atmospheric lighting, architectural digest, masterpiece, 8k", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        "10": {"class_type": "KSampler", "inputs": {"seed": 8110, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Sec3/01_anchor_zen_lighting", "images": ["11", 0]}},

        # 2. Recipient Baseline: Modern minimalist concrete penthouse lounge overlooking city, flat overcast daylight
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "sleek minimalist architectural penthouse living room with floor to ceiling glass windows, smooth cast concrete walls, low modern sofa and marble coffee table, flat neutral overcast daylight, architectural photography, sharp focus, 8k", "clip": ["1", 1]}},
        "15": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        "20": {"class_type": "KSampler", "inputs": {"seed": 5110, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["15", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["1", 2]}},
        "22": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Sec3/02_ctrl_penthouse_daylight", "images": ["21", 0]}},

        # 3. GPS Anchor & Cross-Modal Steering Transfer
        "24": {"class_type": "CLIPTextEncode", "inputs": {"text": "sleek minimalist architectural penthouse living room with floor to ceiling glass windows, smooth cast concrete walls, low modern sofa and marble coffee table, dramatic warm glowing amber lantern lighting, dark polished floor reflections, twilight moody atmospheric interior, 8k", "clip": ["1", 1]}},

        "30": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "warm glowing amber lantern lighting, dark polished cedar reflections, twilight moody atmosphere", "mapping_mode": "Keyword_Heuristics", "base_latent": ["20", 0]}},
        "31": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 0], "target_latent": ["30", 0], "origin_latent": ["30", 1]}},
        "32": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["31", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        # Render Steered Penthouse at denoise 0.65
        "35": {"class_type": "KSampler", "inputs": {"seed": 5110, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["24", 0], "negative": ["3", 0], "latent_image": ["32", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Sec3/03_steered_penthouse_amber_lantern", "images": ["36", 0]}},
    }

    outputs = queue_and_wait(wf, "Section 3 Lighting Transfer")
    save_output_image(outputs, "12", "01_sec3_anchor_zen_lighting.png")
    save_output_image(outputs, "22", "02_sec3_ctrl_penthouse_daylight.png")
    save_output_image(outputs, "37", "03_sec3_steered_penthouse_amber_lantern.png")

# ==============================================================================
# SECTION 4: SEMANTIC SLIDER / CONCEPT AXIS STEERING
# Cantilevered Modernist Villa -> Biophilic Solarpunk (+1.5) vs Nordic Frost (-1.5)
# ==============================================================================
def run_section_4_semantic_slider():
    print("\n" + "=" * 70)
    print("RUNNING SECTION 4: SEMANTIC SLIDER (BIOPHILIC LIVING VILLA vs NORDIC FROST)")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # 1. Base Villa Baseline (Seed 6220): Cantilevered concrete and glass villa over water in neutral clear daylight
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "monolithic modernist cantilevered glass villa over reflective water pond, smooth white concrete slabs, floor to ceiling glass, clean clear daylight, architectural photography, sharp focus, 8k", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        "10": {"class_type": "KSampler", "inputs": {"seed": 6220, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Sec4/01_villa_baseline_ctrl", "images": ["11", 0]}},

        # 2. Axis Direction Positive: Lush Biophilic Solarpunk (Cascading vertical gardens, hanging vines, tropical palms)
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "monolithic modernist cantilevered glass villa over reflective water pond, lush cascading vertical gardens, hanging green ivy vines, tropical ferns and palm trees wrapping the white concrete slabs, vibrant solarpunk ecosystem, warm golden sunlight, 8k", "clip": ["1", 1]}},
        
        # 3. Axis Direction Negative: Nordic Frost & Alpine Winter (Snow dusted roof slabs, frozen iced pond, winter twilight)
        "15": {"class_type": "CLIPTextEncode", "inputs": {"text": "monolithic modernist cantilevered glass villa over frozen iced pond, snow dusted white concrete roof slabs, frost glazed glass, cold blue alpine winter twilight, pine trees with snow, warm cozy interior glow, 8k", "clip": ["1", 1]}},

        # Steering Positive: Biophilic Gardens
        "20": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "lush cascading vertical gardens, hanging green ivy vines, tropical ferns, vibrant solarpunk sunlight", "mapping_mode": "Keyword_Heuristics", "base_latent": ["10", 0]}},
        "21": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["10", 0], "target_latent": ["20", 0], "origin_latent": ["20", 1]}},
        "22": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["21", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        "25": {"class_type": "KSampler", "inputs": {"seed": 6220, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["14", 0], "negative": ["3", 0], "latent_image": ["22", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["1", 2]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Sec4/02_villa_steered_biophilic_green", "images": ["26", 0]}},

        # Steering Negative: Nordic Frost Winter
        "30": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "frozen iced pond, snow dusted roof slabs, frost glazed glass, cold blue alpine winter twilight", "mapping_mode": "Keyword_Heuristics", "base_latent": ["10", 0]}},
        "31": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["10", 0], "target_latent": ["30", 0], "origin_latent": ["30", 1]}},
        "32": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["31", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        "35": {"class_type": "KSampler", "inputs": {"seed": 6220, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["15", 0], "negative": ["3", 0], "latent_image": ["32", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["1", 2]}},
        "37": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Sec4/03_villa_steered_nordic_frost", "images": ["36", 0]}},
    }

    outputs = queue_and_wait(wf, "Section 4 Semantic Slider")
    save_output_image(outputs, "12", "01_sec4_villa_baseline_ctrl.png")
    save_output_image(outputs, "27", "02_sec4_villa_biophilic_green.png")
    save_output_image(outputs, "37", "03_sec4_villa_nordic_frost.png")

def main():
    print("Generating Fresh Sections 3 & 4 with New Controls & Directions...")
    run_section_3_lighting_transfer()
    run_section_4_semantic_slider()
    print("\n" + "=" * 70)
    print("SECTIONS 3 & 4 GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
