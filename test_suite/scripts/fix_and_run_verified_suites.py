"""
Audited Remediation for Suite 4 (Vector Analogy Diagnostics) and Suite 5 (Subspace Material Matrix).
Fixes keyword mapping and denoise calibration for Armchair mutations,
and runs side-by-side spatial collision vs channel-mean / mask-guided analogy.
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
os.makedirs(os.path.join(OUTPUT_DIR, "04_analogy_audit"), exist_ok=True)

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

def save_output_image(outputs, save_node_id, target_filename, subfolder=""):
    out_info = outputs.get(str(save_node_id), {}).get("images", [])
    if not out_info:
        raise RuntimeError(f"No image outputs found for node {save_node_id}: {outputs}")
    
    img_meta = out_info[0]
    fname = img_meta["filename"]
    img_sub = img_meta.get("subfolder", "")
    img_type = img_meta.get("type", "output")
    
    view_url = f"{COMFY_URL}/view?filename={fname}&subfolder={img_sub}&type={img_type}"
    r = requests.get(view_url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch image {fname}: {r.status_code}")
    
    dest_dir = os.path.join(OUTPUT_DIR, subfolder) if subfolder else OUTPUT_DIR
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, target_filename)
    with open(dest_path, "wb") as f:
        f.write(r.content)
    print(f"  -> Saved {target_filename} ({len(r.content)} bytes)")
    return dest_path

# ==============================================================================
# REMEDIATION FOR SUITE 5: SUBSPACE MATERIAL MATRIX (CALIBRATED ARMCHAIR)
# ==============================================================================
def run_fixed_armchair_matrix():
    print("\n" + "=" * 75)
    print("RUNNING REMEDIATED SUBSPACE MATERIAL MATRIX (CALIBRATED KEYWORDS & DENOISE)")
    print("=" * 75)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # 1. Base Control: Minimalist sculptural curved lounge armchair in raw oat linen and blonde ash wood
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "minimalist architectural sculptural curved lounge armchair in raw natural light oat linen and blonde ash wood frame, concrete architectural interior, soft neutral daylight, catalog design photography, sharp focus, 8k", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, distorted geometry, watermark, noisy", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        
        # Render Base Oat Linen Chair (Seed 7712)
        "10": {"class_type": "KSampler", "inputs": {"seed": 7712, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6_Fix/09_chair_ctrl_oat_linen", "images": ["11", 0]}},

        # Split SDXL Latent (Channels 0..1 = Structural silhouette, Channels 2..3 = Material chroma/texture)
        "20": {"class_type": "GimbalChannelSplit", "inputs": {"split_index": 2, "latent": ["10", 0]}},

        # --- MUTATION 1: MIRROR-POLISHED LIQUID CHROME & SMOKED GLASS ---
        "21": {"class_type": "CLIPTextEncode", "inputs": {"text": "minimalist architectural sculptural curved lounge armchair made entirely of mirror polished liquid chrome and dark smoked glass, concrete architectural interior, glossy studio reflections, luxury industrial design render, 8k", "clip": ["1", 1]}},
        # Valid keywords: cool, sharp, crisp, monochrome, bright
        "22": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "cool sharp crisp monochrome bright cold", "mapping_mode": "Keyword_Heuristics", "base_latent": ["20", 1]}},
        "23": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 1], "target_latent": ["22", 0], "origin_latent": ["22", 1]}},
        "24": {"class_type": "GimbalChannelMerge", "inputs": {"latent_band_A": ["20", 0], "latent_band_B": ["23", 0]}},
        "25": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["24", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        # Sample at denoise 0.65 to allow material texture recrystallization while geometry is locked
        "26": {"class_type": "KSampler", "inputs": {"seed": 7712, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["21", 0], "negative": ["3", 0], "latent_image": ["25", 0]}},
        "27": {"class_type": "VAEDecode", "inputs": {"samples": ["26", 0], "vae": ["1", 2]}},
        "28": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6_Fix/09_chair_steered_liquid_chrome", "images": ["27", 0]}},

        # --- MUTATION 2: HIGH-GLOSS OXBLOOD VELVET & BRUSHED ROSE GOLD ---
        "31": {"class_type": "CLIPTextEncode", "inputs": {"text": "minimalist architectural sculptural curved lounge armchair in rich deep oxblood red velvet and brushed rose gold metal frame, concrete architectural interior, warm ambient specular highlights, luxury bespoke furniture, 8k", "clip": ["1", 1]}},
        # Valid keywords: warm, saturated, vivid, dark, moody, fire
        "32": {"class_type": "GimbalCrossModalBridge", "inputs": {"llm_instruction": "warm saturated vivid dark moody fire golden", "mapping_mode": "Keyword_Heuristics", "base_latent": ["20", 1]}},
        "33": {"class_type": "GimbalCompass_Pro", "inputs": {"strength": 1.6, "mode": "Orthogonal_Projection", "clamp_output": True, "clamp_min": -8.0, "clamp_max": 8.0, "allow_batch_expand": False, "ortho_per_channel": True, "clamp_mask_input": False, "enable_perf_logging": False, "base_latent": ["20", 1], "target_latent": ["32", 0], "origin_latent": ["32", 1]}},
        "34": {"class_type": "GimbalChannelMerge", "inputs": {"latent_band_A": ["20", 0], "latent_band_B": ["33", 0]}},
        "35": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["34", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        
        # Sample at denoise 0.65
        "36": {"class_type": "KSampler", "inputs": {"seed": 7712, "control_after_generate": "fixed", "steps": 24, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.65, "model": ["1", 0], "positive": ["31", 0], "negative": ["3", 0], "latent_image": ["35", 0]}},
        "37": {"class_type": "VAEDecode", "inputs": {"samples": ["36", 0], "vae": ["1", 2]}},
        "38": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Run6_Fix/09_chair_steered_oxblood_velvet", "images": ["37", 0]}},
    }
    
    outputs, elapsed = queue_and_wait(wf, "Fixed_Armchair_Subspace")
    save_output_image(outputs, "12", "09_chair_ctrl_oat_linen.png")
    save_output_image(outputs, "28", "09_chair_steered_liquid_chrome.png")
    save_output_image(outputs, "38", "09_chair_steered_oxblood_velvet.png")

if __name__ == "__main__":
    run_fixed_armchair_matrix()
