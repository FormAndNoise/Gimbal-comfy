"""
Comprehensive Live Workflow Test Runner for Gimbal Node Suite.
Executes all 9 official workflows + chained Pro pipeline against running ComfyUI.
"""

import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "H:/New folder/Gimbal-comfy/test_results/live_workflow_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT = "sd_xl_base_1.0.safetensors"

def queue_and_wait(prompt_dict, tag, timeout=300):
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
                    print(f"[{tag}] Completed successfully in {elapsed}s")
                    return item.get("outputs", {}), elapsed
                elif status.get("status_str") == "error":
                    raise RuntimeError(f"[{tag}] Execution error: {status.get('messages')}")
    raise TimeoutError(f"[{tag}] Timed out after {timeout}s")

def save_output_images(outputs, save_node_ids, prefix):
    saved_files = []
    if isinstance(save_node_ids, (int, str)):
        save_node_ids = [save_node_ids]
        
    for node_id in save_node_ids:
        out_info = outputs.get(str(node_id), {}).get("images", [])
        for idx, img_meta in enumerate(out_info):
            fname = img_meta["filename"]
            subfolder = img_meta.get("subfolder", "")
            img_type = img_meta.get("type", "output")
            
            view_url = f"{COMFY_URL}/view?filename={fname}&subfolder={subfolder}&type={img_type}"
            r = requests.get(view_url, timeout=30)
            if r.status_code != 200:
                print(f"  [Warning] Failed to fetch image {fname}: {r.status_code}")
                continue
                
            out_name = f"{prefix}_{idx:02d}_{fname}" if len(out_info) > 1 else f"{prefix}_{fname}"
            dest_path = os.path.join(OUTPUT_DIR, out_name)
            with open(dest_path, "wb") as f:
                f.write(r.content)
            print(f"  -> Saved {out_name} ({len(r.content)} bytes)")
            saved_files.append(dest_path)
    return saved_files

def main():
    print("=" * 80)
    print("     GIMBAL NODE SUITE — AUTOMATED WORKFLOW TEST SUITE")
    print("=" * 80)
    
    # Check ComfyUI connectivity
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
        stats = r.json()
        print(f"[ComfyUI] Connected: {COMFY_URL}")
        devices = stats.get("devices", [{}])
        if devices:
            d = devices[0]
            print(f"[Hardware] GPU: {d.get('name')} | VRAM: {d.get('vram_total', 0) // (1024*1024)} MB")
    except Exception as e:
        print(f"[Error] Cannot connect to ComfyUI at {COMFY_URL}: {e}")
        return

    results = []

    # --------------------------------------------------------------------------
    # TEST 1: Gimbal 01 — Concept Blender
    # --------------------------------------------------------------------------
    try:
        tag = "Workflow_01_ConceptBlender"
        print(f"\n[{tag}] Starting test...")
        with open("H:/New folder/Gimbal-comfy/extras/example_workflows/api/API_Gimbal_01_ConceptBlender.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        outputs, elapsed = queue_and_wait(wf, tag)
        saved = save_output_images(outputs, ["13"], "01_ConceptBlender")
        results.append({"name": "01_ConceptBlender", "status": "PASS", "time": elapsed, "images": len(saved)})
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append({"name": "01_ConceptBlender", "status": "FAIL", "error": str(e)})

    # --------------------------------------------------------------------------
    # TEST 2: Gimbal 02 — Text-Steered Diffusion
    # --------------------------------------------------------------------------
    try:
        tag = "Workflow_02_TextSteered"
        print(f"\n[{tag}] Starting test...")
        with open("H:/New folder/Gimbal-comfy/extras/example_workflows/api/API_Gimbal_02_TextSteered.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        outputs, elapsed = queue_and_wait(wf, tag)
        saved = save_output_images(outputs, ["12", "27"], "02_TextSteered")
        results.append({"name": "02_TextSteered", "status": "PASS", "time": elapsed, "images": len(saved)})
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append({"name": "02_TextSteered", "status": "FAIL", "error": str(e)})

    # --------------------------------------------------------------------------
    # TEST 3: Gimbal 03 — Manifold Grid Explorer (2x2 fast grid test)
    # --------------------------------------------------------------------------
    try:
        tag = "Workflow_03_ManifoldGrid"
        print(f"\n[{tag}] Starting test...")
        with open("H:/New folder/Gimbal-comfy/extras/example_workflows/api/API_Gimbal_03_ManifoldGrid.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        wf["8"]["inputs"]["grid_size_x"] = 2
        wf["8"]["inputs"]["grid_size_y"] = 2
        outputs, elapsed = queue_and_wait(wf, tag)
        saved = save_output_images(outputs, ["11"], "03_ManifoldGrid")
        results.append({"name": "03_ManifoldGrid", "status": "PASS", "time": elapsed, "images": len(saved)})
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append({"name": "03_ManifoldGrid", "status": "FAIL", "error": str(e)})

    # --------------------------------------------------------------------------
    # TEST 4: Gimbal 04 — Brand-Locked Steering
    # --------------------------------------------------------------------------
    try:
        tag = "Workflow_04_BrandLocked"
        print(f"\n[{tag}] Starting test...")
        with open("H:/New folder/Gimbal-comfy/extras/example_workflows/api/API_Gimbal_04_BrandLocked.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        outputs, elapsed = queue_and_wait(wf, tag)
        saved = save_output_images(outputs, ["8", "15"], "04_BrandLocked")
        results.append({"name": "04_BrandLocked", "status": "PASS", "time": elapsed, "images": len(saved)})
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append({"name": "04_BrandLocked", "status": "FAIL", "error": str(e)})

    # --------------------------------------------------------------------------
    # TEST 5: Gimbal 05 — Semantic Slider (PCA Decomposition)
    # --------------------------------------------------------------------------
    try:
        tag = "Workflow_05_SemanticSlider"
        print(f"\n[{tag}] Starting test...")
        with open("H:/New folder/Gimbal-comfy/extras/example_workflows/api/API_Gimbal_05_SemanticSlider.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        wf["4"]["inputs"]["batch_size"] = 4
        outputs, elapsed = queue_and_wait(wf, tag)
        saved = save_output_images(outputs, ["10"], "05_SemanticSlider")
        results.append({"name": "05_SemanticSlider", "status": "PASS", "time": elapsed, "images": len(saved)})
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append({"name": "05_SemanticSlider", "status": "FAIL", "error": str(e)})

    # --------------------------------------------------------------------------
    # TEST 6: Gimbal 06 — Architecture Material Matrix
    # --------------------------------------------------------------------------
    try:
        tag = "Workflow_06_ArchitectureMaterialMatrix"
        print(f"\n[{tag}] Starting test...")
        with open("H:/New folder/Gimbal-comfy/extras/example_workflows/api/API_Gimbal_06_ArchitectureMaterialMatrix.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        wf["8"]["inputs"]["grid_size_x"] = 2
        wf["8"]["inputs"]["grid_size_y"] = 2
        wf["12"]["inputs"]["select_index"] = 0
        outputs, elapsed = queue_and_wait(wf, tag)
        saved = save_output_images(outputs, ["11"], "06_ArchMaterialMatrix")
        results.append({"name": "06_ArchMaterialMatrix", "status": "PASS", "time": elapsed, "images": len(saved)})
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append({"name": "06_ArchMaterialMatrix", "status": "FAIL", "error": str(e)})

    # --------------------------------------------------------------------------
    # TEST 7: Gimbal 07 — Likeness Isolator
    # --------------------------------------------------------------------------
    try:
        tag = "Workflow_07_LikenessIsolator"
        print(f"\n[{tag}] Starting test...")
        with open("H:/New folder/Gimbal-comfy/extras/example_workflows/api/API_Gimbal_07_LikenessIsolator.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        outputs, elapsed = queue_and_wait(wf, tag)
        saved = save_output_images(outputs, ["8"], "07_LikenessIsolator")
        results.append({"name": "07_LikenessIsolator", "status": "PASS", "time": elapsed, "images": len(saved)})
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append({"name": "07_LikenessIsolator", "status": "FAIL", "error": str(e)})

    # --------------------------------------------------------------------------
    # TEST 8: Gimbal 08 — Harmonic Orbiter
    # --------------------------------------------------------------------------
    try:
        tag = "Workflow_08_HarmonicOrbiter"
        print(f"\n[{tag}] Starting test...")
        with open("H:/New folder/Gimbal-comfy/extras/example_workflows/api/API_Gimbal_08_HarmonicOrbiter.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        wf["20"]["inputs"]["steps"] = 3
        outputs, elapsed = queue_and_wait(wf, tag)
        saved = save_output_images(outputs, ["32"], "08_HarmonicOrbiter")
        results.append({"name": "08_HarmonicOrbiter", "status": "PASS", "time": elapsed, "images": len(saved)})
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append({"name": "08_HarmonicOrbiter", "status": "FAIL", "error": str(e)})

    # --------------------------------------------------------------------------
    # TEST 9: Gimbal 09 — Subspace Material Matrix
    # --------------------------------------------------------------------------
    try:
        tag = "Workflow_09_SubspaceMaterialMatrix"
        print(f"\n[{tag}] Starting test...")
        with open("H:/New folder/Gimbal-comfy/extras/example_workflows/api/API_Gimbal_09_SubspaceMaterialMatrix.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        outputs, elapsed = queue_and_wait(wf, tag)
        saved = save_output_images(outputs, ["12", "27"], "09_SubspaceMaterialMatrix")
        results.append({"name": "09_SubspaceMaterialMatrix", "status": "PASS", "time": elapsed, "images": len(saved)})
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append({"name": "09_SubspaceMaterialMatrix", "status": "FAIL", "error": str(e)})

    # --------------------------------------------------------------------------
    # TEST 10: Chained Pro Pipeline (Compass + CrossModal + Manifold + GPS + Slider)
    # --------------------------------------------------------------------------
    try:
        tag = "Workflow_10_ProPipeline"
        print(f"\n[{tag}] Starting test...")
        with open("H:/New folder/Gimbal-comfy/extras/example_workflows/Pro_Compass_Manifold_SemanticSlider_Pipeline.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        wf["7"]["inputs"]["grid_size_x"] = 2
        wf["7"]["inputs"]["grid_size_y"] = 2
        wf["8"]["inputs"]["select_index"] = 0
        outputs, elapsed = queue_and_wait(wf, tag)
        saved = save_output_images(outputs, ["12"], "10_ProPipeline")
        results.append({"name": "10_ProPipeline", "status": "PASS", "time": elapsed, "images": len(saved)})
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append({"name": "10_ProPipeline", "status": "FAIL", "error": str(e)})

    # Summary table
    print("\n" + "=" * 80)
    print("                    TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Workflow Name':<35} | {'Status':<8} | {'Time (s)':<10} | {'Images':<8}")
    print("-" * 80)
    for res in results:
        t_str = f"{res.get('time', 0.0):.2f}" if "time" in res else "-"
        img_str = str(res.get("images", 0)) if "images" in res else "-"
        print(f"{res['name']:<35} | {res['status']:<8} | {t_str:<10} | {img_str:<8}")
    print("=" * 80)

    # Save summary json
    summary_path = os.path.join(OUTPUT_DIR, "test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved test summary to {summary_path}")

if __name__ == "__main__":
    main()
