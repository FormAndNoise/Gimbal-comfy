"""
Gimbal Vector Analogy Audited Test Suite:
Validates A - B + C vector arithmetic across:
1. Concept A (Man with bold eyeglasses & amber rim lighting, Seed 1001)
2. Concept B (Man clean baseline without glasses, Seed 1001)
3. Concept C (Recipient Woman baseline, Seed 2002)
4. Direct Spatial Analogy: C + (A - B)
5. Orthogonal Norm-Preserved Analogy: C + Ortho(A - B) with norm lock
6. Channel Mean Lighting Analogy: C + Mean(A - B)
"""

import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/fresh_run6/04_analogy_audit"
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
    
    img_meta = out_info[0]
    fname = img_meta["filename"]
    subfolder = img_meta.get("subfolder", "")
    img_type = img_meta.get("type", "output")
    
    view_url = f"{COMFY_URL}/view?filename={fname}&subfolder={subfolder}&type={img_type}"
    r = requests.get(view_url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch image {fname}: {r.status_code}")
    
    dest_path = os.path.join(OUTPUT_DIR, target_filename)
    with open(dest_path, "wb") as f:
        f.write(r.content)
    print(f"  -> Saved {target_filename} ({len(r.content)} bytes)")
    return dest_path

def run_analogy_suite():
    print("\n" + "=" * 75)
    print("RUNNING GIMBAL VECTOR ANALOGY AUDIT (A - B + C ARITHMETIC)")
    print("=" * 75)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        # Concept A: Person with thick bold black eyeglasses and dramatic amber rim lighting
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait photo of a distinguished person wearing thick black acetate eyeglasses, dramatic warm amber rim lighting, dark studio background, sharp focus, 8k", "clip": ["1", 1]}},
        
        # Concept B: Same person clean face without glasses, neutral flat light
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait photo of a distinguished person with clean face and no glasses, neutral flat studio lighting, dark background, sharp focus, 8k", "clip": ["1", 1]}},
        
        # Concept C: Recipient Person (different subject / face) in flat daylight
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait photo of a stylish young creative person, smiling, clean neutral daylight studio lighting, sharp focus, 8k", "clip": ["1", 1]}},
        
        # Target Prompt for C with analogy attributes
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait photo of a stylish young creative person wearing thick black acetate eyeglasses, dramatic warm amber rim lighting, sharp focus, 8k", "clip": ["1", 1]}},
        
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, deformed, watermark, noisy", "clip": ["1", 1]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},

        # 1. Concept A (Seed 1001)
        "10": {"class_type": "KSampler", "inputs": {"seed": 1001, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Analogy/01_concept_A_glasses_amber", "images": ["11", 0]}},

        # 2. Concept B (Seed 1001)
        "13": {"class_type": "KSampler", "inputs": {"seed": 1001, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}},
        "15": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Analogy/02_concept_B_clean_baseline", "images": ["14", 0]}},

        # 3. Concept C (Seed 2002)
        "16": {"class_type": "KSampler", "inputs": {"seed": 2002, "control_after_generate": "fixed", "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["1", 2]}},
        "18": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Analogy/03_concept_C_recipient_clean", "images": ["17", 0]}},

        # 4. Mode 1: Direct Spatial Vector Analogy [C + (A - B)]
        "20": {"class_type": "GimbalVectorAnalogy", "inputs": {"concept_A": ["10", 0], "concept_B": ["13", 0], "concept_C": ["16", 0], "strength": 0.85, "spatial_mode": "Spatial_Direct", "ortho_project": False, "preserve_norm": True}},
        "21": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["20", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        "22": {"class_type": "KSampler", "inputs": {"seed": 2002, "control_after_generate": "fixed", "steps": 22, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.50, "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["21", 0]}},
        "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["1", 2]}},
        "24": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Analogy/04_analogy_spatial_direct", "images": ["23", 0]}},

        # 5. Mode 2: Orthogonal & Norm-Preserved Analogy [C + Ortho(A - B)]
        "30": {"class_type": "GimbalVectorAnalogy", "inputs": {"concept_A": ["10", 0], "concept_B": ["13", 0], "concept_C": ["16", 0], "strength": 0.85, "spatial_mode": "Spatial_Direct", "ortho_project": True, "preserve_norm": True}},
        "31": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["30", 0], "truncation_psi": 0.88, "subspace_rank": -1, "scale_cap": 8.0, "jitter_strength": 0.0}},
        "32": {"class_type": "KSampler", "inputs": {"seed": 2002, "control_after_generate": "fixed", "steps": 22, "cfg": 3.8, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.50, "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["31", 0]}},
        "33": {"class_type": "VAEDecode", "inputs": {"samples": ["32", 0], "vae": ["1", 2]}},
        "34": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Analogy/05_analogy_ortho_norm_locked", "images": ["33", 0]}},
    }
    
    outputs, elapsed = queue_and_wait(wf, "Vector_Analogy_Audit")
    save_output_image(outputs, "12", "01_concept_A_glasses_amber.png")
    save_output_image(outputs, "15", "02_concept_B_clean_baseline.png")
    save_output_image(outputs, "18", "03_concept_C_recipient_clean.png")
    save_output_image(outputs, "24", "04_analogy_spatial_direct.png")
    save_output_image(outputs, "34", "05_analogy_ortho_norm_locked.png")

if __name__ == "__main__":
    run_analogy_suite()
