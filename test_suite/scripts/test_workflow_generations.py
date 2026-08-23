"""
Automated live workflow execution test suite for Gimbal ComfyUI Suite.
Tests all workflows against the running ComfyUI server.
"""

import json
import glob
import os
import sys
import time
import requests
import copy

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"

node_alias_map = {
    "LatentGANArithmetic": "GimbalVectorAnalogy",
    "LatentTruncation": "GimbalTruncation",
    "LatentChannelSplit": "GimbalChannelSplit",
    "LatentChannelMerge": "GimbalChannelMerge",
    "LatentPCA": "GimbalSemanticSlider",
    "LatentApplyDirection": "GimbalCompass_Pro",
    "LatentPerturbBatch": "GimbalCompass_Pro",
    "LatentSlerp": "GimbalCompass_Pro",
    "LatentSlerpBatch": "GimbalCompass_Pro",
    "LatentDirectionStore": "GimbalGPS_Anchor",
    "LatentCircularWalk": "GimbalCircularOrbit",
    "LatentMultiWaypointInterpolation": "GimbalWaypointSpline",
    "LatentGrid2D": "GimbalManifold_Explorer",
    "LatentStatistics": "GimbalDiagnostics",
    "LatentInfo": "GimbalDiagnostics",
    "LatentBlend": "GimbalCompass_Pro",
    "LatentNormalizedInterpolation": "GimbalCompass_Pro",
    "LatentWeightedAverage": "GimbalCompass_Pro",
    "LatentScalarMultiply": "GimbalCompass_Pro",
    "LatentNormalize": "GimbalTruncation",
    "LatentAddNoise": "GimbalLatentStabilizer",
    "LatentNoisePerturbation": "GimbalCompass_Pro",
    "LatentRandomWalk": "GimbalCircularOrbit",
    "LatentRandomDirection": "GimbalCompass_Pro",
    "LatentSelectFromBatch": "GimbalGPS_Anchor",
}

widget_mappings = {
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "CLIPTextEncode": ["text"],
    "CheckpointLoaderSimple": ["ckpt_name"],
    "SaveImage": ["filename_prefix"],
    "VAEDecode": [],
    "VAEEncode": [],
    "LoadImage": ["image", "upload"],
    "ShowText|pysssss": ["text"],
    "ShowText": ["text"],
    "GimbalCrossModalBridge": ["llm_instruction", "mapping_mode"],
    "Gimbal_CrossModalBridge": ["llm_instruction", "mapping_mode"],
    "Wayfinder_CrossModalBridge": ["llm_instruction", "mapping_mode"],
    "GimbalManifold_Explorer": ["grid_size_x", "grid_size_y", "x_strength", "y_strength", "interpolation_mode", "normalize_vectors", "clamp_output", "clamp_min", "clamp_max", "enable_perf_logging"],
    "WayfinderManifold_Explorer": ["grid_size_x", "grid_size_y", "x_strength", "y_strength", "interpolation_mode", "normalize_vectors", "clamp_output", "clamp_min", "clamp_max", "enable_perf_logging"],
    "GimbalCompass_Pro": ["strength", "mode", "clamp_output", "clamp_min", "clamp_max", "allow_batch_expand", "ortho_per_channel", "clamp_mask_input", "enable_perf_logging"],
    "WayfinderCompass_Pro": ["strength", "mode", "clamp_output", "clamp_min", "clamp_max", "allow_batch_expand", "ortho_per_channel", "clamp_mask_input", "enable_perf_logging"],
    "GimbalGPS_Anchor": ["select_index", "save_waypoint", "waypoint_name", "enable_perf_logging"],
    "WayfinderGPS_Anchor": ["select_index", "save_waypoint", "waypoint_name", "enable_perf_logging"],
    "GimbalSemanticSlider": ["pc_index", "slider_value", "orthogonalize"],
    "Gimbal_SemanticSlider": ["pc_index", "slider_value", "orthogonalize"],
    "Wayfinder_SemanticSlider": ["pc_index", "slider_value", "orthogonalize"],
    "GimbalLikenessIsolator": ["lora_name", "strength", "alpha", "likeness_mask"],
    "LikenessVectorIsolator": ["lora_name", "strength", "alpha", "likeness_mask"],
    "GimbalGPS_Load": ["waypoint_name", "restore_mode"],
    "WayfinderGPS_Load": ["waypoint_name", "restore_mode"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "LatentFromBatch": ["batch_index", "length"],
    "GimbalVectorAnalogy": ["strength", "orthogonalize", "preserve_norm"],
    "GimbalTruncation": ["psi", "rescale_norm", "channel_adaptive"],
    "GimbalCircularOrbit": ["num_frames", "radius", "elevation", "closed_loop"],
    "GimbalWaypointSpline": ["num_frames", "tension", "closed_loop"],
    "GimbalChannelScale": ["scale_c0", "scale_c1", "scale_c2", "scale_c3"],
    "GimbalDiagnostics": ["tag", "print_to_console"],
    "GimbalLatentStabilizer": ["coupling_scale", "jitter_eps", "psi", "denoise_rank", "enable_stabilization"],
    "GimbalLatentMath": ["operation", "scalar_a", "scalar_b", "clamp_output", "clamp_min", "clamp_max"],
    "GimbalLatentTelemetry": ["enable_mahalanobis", "enable_tc", "enable_geodesic", "tag", "print_to_console"],
}

def get_comfy_info():
    r = requests.get(f"{COMFY_URL}/object_info", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI object_info failed: {r.status_code}")
    info = r.json()
    ckpts = info.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
    loras = info.get("LoraLoader", {}).get("input", {}).get("required", {}).get("lora_name", [[]])[0]
    return info, ckpts, loras

def resolve_link(l_id, links_d, nodes_d):
    if l_id is None or l_id not in links_d:
        return None, None
    l = links_d[l_id]
    src_id, src_slot = str(l[1]), l[2]
    
    src_node = nodes_d.get(src_id)
    if src_node and src_node.get("type") in ["Reroute", "PrimitiveNode"]:
        if "inputs" in src_node and len(src_node["inputs"]) > 0:
            return resolve_link(src_node["inputs"][0].get("link"), links_d, nodes_d)
    return src_id, src_slot

def convert_ui_to_api(wf_data, available_ckpts, available_loras, fast_mode=True):
    if "nodes" not in wf_data:
        # Already API format
        api_data = copy.deepcopy(wf_data)
    else:
        api_data = {}
        links = {link[0]: link for link in wf_data.get("links", [])}
        nodes_by_id = {str(n["id"]): n for n in wf_data.get("nodes", [])}

        for node in wf_data.get("nodes", []):
            if node.get("type") in ["Reroute", "Note", "PrimitiveNode"]:
                continue
                
            node_id = str(node["id"])
            raw_type = node.get("type")
            node_type = node_alias_map.get(raw_type, raw_type)
            
            # Map legacy display types
            if node_type == "ShowText":
                node_type = "ShowText|pysssss"
                
            inputs = {}
            
            widgets_values = node.get("widgets_values", [])
            widget_names = widget_mappings.get(node_type, [])
            for i, val in enumerate(widgets_values):
                if i < len(widget_names):
                    inputs[widget_names[i]] = val
                else:
                    inputs[f"widget_{i}"] = val

            for in_link in node.get("inputs", []):
                link_id = in_link.get("link")
                origin_node, origin_slot = resolve_link(link_id, links, nodes_by_id)
                if origin_node is not None:
                    inputs[in_link["name"]] = [origin_node, origin_slot]

            api_data[node_id] = {
                "class_type": node_type,
                "inputs": inputs
            }

        # Resolve unmapped links for nodes like PreviewImage/SaveImage
        for l in wf_data.get("links", []):
            tgt_id = str(l[3])
            if tgt_id in api_data:
                in_type = l[5]
                src_node, src_slot = resolve_link(l[0], links, nodes_by_id)
                if src_node is not None:
                    name = "images" if in_type == "IMAGE" else "latent_image" if in_type == "LATENT" else "text" if in_type == "STRING" else f"input_{l[4]}"
                    if name not in api_data[tgt_id]["inputs"]:
                        api_data[tgt_id]["inputs"][name] = [src_node, src_slot]

    # Adapt checkpoints, LoRAs, and sampling steps for test execution
    default_sdxl = next((c for c in available_ckpts if "sd_xl_base" in c.lower()), available_ckpts[0] if available_ckpts else None)
    default_sd15 = next((c for c in available_ckpts if "ghostmix" in c.lower() or "v1-5" in c.lower()), available_ckpts[0] if available_ckpts else None)
    default_lora = next((l for l in available_loras if "sdxl" in l.lower()), available_loras[0] if available_loras else None)

    for node_id, node in api_data.items():
        raw_ctype = node.get("class_type")
        ctype = node_alias_map.get(raw_ctype, raw_ctype)
        node["class_type"] = ctype
        inp = node.get("inputs", {})
        
        if ctype == "CheckpointLoaderSimple":
            current_ckpt = inp.get("ckpt_name", "")
            if current_ckpt not in available_ckpts:
                if "sd_xl" in current_ckpt.lower() or "xl" in current_ckpt.lower():
                    inp["ckpt_name"] = default_sdxl
                else:
                    inp["ckpt_name"] = default_sd15
                    
        elif ctype in ["LoraLoader", "GimbalLikenessIsolator", "LikenessVectorIsolator"]:
            current_lora = inp.get("lora_name", "")
            if current_lora not in available_loras:
                inp["lora_name"] = default_lora
                
        elif ctype in ["KSampler", "KSamplerAdvanced"]:
            if fast_mode and "steps" in inp:
                inp["steps"] = min(inp["steps"], 8)
            if "negative" not in inp or inp["negative"] is None:
                if "positive" in inp:
                    inp["negative"] = inp["positive"]
                    
        elif ctype == "LoadImage":
            inp["image"] = "floor_plan_reference.png"

        elif ctype in ["EmptyLatentImage"] and fast_mode:
            # Reduce batch size for manifold/batch tests if large
            if "batch_size" in inp and inp["batch_size"] > 4:
                inp["batch_size"] = 4

    return api_data

def test_workflow(wf_path, available_ckpts, available_loras, timeout_seconds=180):
    with open(wf_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
        
    api_prompt = convert_ui_to_api(raw, available_ckpts, available_loras, fast_mode=True)
    
    # Send to ComfyUI
    payload = {"prompt": api_prompt}
    try:
        r = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=10)
        if r.status_code != 200:
            return False, f"Validation/Queue Failed ({r.status_code}): {r.text}", None
            
        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            return False, f"No prompt_id returned: {r.json()}", None
            
        # Poll for execution
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            time.sleep(2)
            hr = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=5)
            if hr.status_code == 200:
                hdata = hr.json()
                if prompt_id in hdata:
                    item = hdata[prompt_id]
                    status = item.get("status", {})
                    if status.get("completed", False):
                        outputs = item.get("outputs", {})
                        elapsed = round(time.time() - start_time, 2)
                        return True, f"Completed in {elapsed}s", outputs
                    elif status.get("status_str") == "error":
                        messages = status.get("messages", [])
                        return False, f"Execution Error: {messages}", None
                        
        return False, f"Timeout after {timeout_seconds}s", None
    except Exception as e:
        return False, f"Exception: {e}", None

def run_all_workflow_tests():
    print("=" * 70)
    print(" GIMBAL NODE SUITE — AUTOMATED WORKFLOW GENERATION TEST RUNNER")
    print("=" * 70)
    
    info, ckpts, loras = get_comfy_info()
    print(f"[OK] Connected to ComfyUI on {COMFY_URL}")
    print(f"[OK] Available Checkpoints: {len(ckpts)}, Available LoRAs: {len(loras)}\n")

    workflow_categories = {
        "Gimbal Example Pipelines": sorted(glob.glob("extras/example_workflows/Gimbal_*.json")),
        "Interactive UI Workflows": sorted(glob.glob("extras/workflows/*.json")),
        "Standard API Workflows": sorted(glob.glob("extras/example_workflows/api/API_Gimbal_*.json")),
    }

    results = {}
    
    for cat_name, wfs in workflow_categories.items():
        print(f"\n--- Category: {cat_name} ({len(wfs)} files) ---")
        results[cat_name] = []
        
        for wf in wfs:
            basename = os.path.basename(wf)
            print(f"Testing {basename}...", end="", flush=True)
            success, msg, outputs = test_workflow(wf, ckpts, loras, timeout_seconds=180)
            if success:
                print(f" -> [PASSED] ({msg})")
                results[cat_name].append({"file": basename, "status": "PASSED", "details": msg})
            else:
                print(f" -> [FAILED] ({msg})")
                results[cat_name].append({"file": basename, "status": "FAILED", "details": msg})

    print("\n" + "=" * 70)
    print(" TEST RUN SUMMARY REPORT")
    print("=" * 70)
    
    total_passed = 0
    total_failed = 0
    
    for cat_name, items in results.items():
        passed = sum(1 for x in items if x["status"] == "PASSED")
        failed = sum(1 for x in items if x["status"] == "FAILED")
        total_passed += passed
        total_failed += failed
        print(f"\n[{cat_name}] ({passed}/{len(items)} passed):")
        for it in items:
            mark = "[+] PASS" if it["status"] == "PASSED" else "[-] FAIL"
            print(f"  {mark} {it['file']}: {it['details']}")

    print("\n" + "-" * 70)
    print(f"TOTAL: {total_passed} Passed, {total_failed} Failed out of {total_passed + total_failed} Workflows.")
    print("-" * 70)

if __name__ == "__main__":
    run_all_workflow_tests()
