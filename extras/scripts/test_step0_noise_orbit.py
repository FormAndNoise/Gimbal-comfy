import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/architectural_noise_orbit"
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

def test_step0_noise_orbit():
    print("\n" + "=" * 70)
    print("TESTING STEP-0 NOISE SPHERICAL ORBIT (CLEAN PHOTOREALISTIC ARCHITECTURAL TOUR)")
    print("=" * 70)

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "monolithic brutalist architectural pavilion on rugged coastal cliff, cantilevered board-formed concrete, floor to ceiling glass, warm interior light, architectural photography, masterpiece, sharp focus", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, cartoon, watermark, noisy", "clip": ["1", 1]}},
        
        # Step 0 Empty Latent Seed
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},

        # GimbalCircularOrbit applied directly at Step 0 on Gaussian Noise!
        "20": {"class_type": "GimbalCircularOrbit", "inputs": {"steps": 4, "radius": 0.95, "orbit_mode": "Orthogonal_Basis", "preserve_hypersphere_norm": True, "seed": 105, "center_latent": ["4", 0]}},

        # Stabilizer to ensure exact standard normal distribution
        "25": {"class_type": "GimbalLatentStabilizer", "inputs": {"latent": ["20", 0], "truncation_psi": 0.92, "subspace_rank": -1, "scale_cap": 10.0, "jitter_strength": 0.0}},

        # Full Denoise Synthesis on the Orbiting Noise Field
        "30": {"class_type": "KSampler", "inputs": {"seed": 8802, "control_after_generate": "fixed", "steps": 24, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["25", 0]}},
        "31": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["1", 2]}},
        "32": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Gimbal_NoiseOrbit/08_arch_tour", "images": ["31", 0]}},
    }

    outputs = queue_and_wait(wf, "Step-0 Noise Orbit")
    save_output_image(outputs, "32", "08_step0_orbit.png")

if __name__ == "__main__":
    test_step0_noise_orbit()
