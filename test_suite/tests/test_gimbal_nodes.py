"""
Test script for Gimbal Node Suite
This script verifies that all Gimbal nodes are properly loaded and functional in ComfyUI
"""

import requests
import json
import time
import sys

COMFYUI_URL = "http://localhost:8188"

def check_comfyui_connection():
    """Check if ComfyUI is running and accessible"""
    global COMFYUI_URL
    try:
        response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=2)
        if response.status_code == 200:
            print("[OK] ComfyUI is running and accessible at 8188")
            return True
    except requests.exceptions.ConnectionError:
        try:
            COMFYUI_URL = "http://localhost:8000"
            response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=2)
            if response.status_code == 200:
                print("[OK] ComfyUI is running and accessible at 8000")
                return True
        except requests.exceptions.ConnectionError:
            print("[FAIL] Cannot connect to ComfyUI at 8188 or 8000")
            return False
    return False

def get_available_nodes():
    """Fetch list of available nodes from ComfyUI"""
    try:
        response = requests.get(f"{COMFYUI_URL}/object_info")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching node list: {e}")
    return {}

def check_gimbal_nodes():
    """Check if all Gimbal nodes are loaded"""
    required_nodes = [
        "GimbalCompass_Pro",
        "GimbalManifold_Explorer", 
        "GimbalGPS_Anchor",
        "GimbalCrossModalBridge",
        "GimbalSemanticSlider"
    ]
    
    nodes = get_available_nodes()
    if not nodes:
        print("[FAIL] Could not fetch node list from ComfyUI")
        return False
    
    all_found = True
    print("\nChecking Gimbal nodes:")
    for node_name in required_nodes:
        if node_name in nodes:
            print(f"[OK] {node_name} - LOADED")
            # Print node info
            node_info = nodes[node_name]
            if 'input' in node_info and 'required' in node_info['input']:
                print(f"  Inputs: {', '.join(node_info['input']['required'].keys())}")
        else:
            print(f"[FAIL] {node_name} - NOT FOUND")
            all_found = False
    
    return all_found

import pytest

def test_simple_workflow():
    """Test a simple workflow using Wayfinder nodes"""
    if not check_comfyui_connection():
        pytest.skip("ComfyUI server not running at :8188 or :8000")

    # Minimal test workflow
    workflow = {
        "1": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            }
        },
        "2": {
            "class_type": "WayfinderCompass_Pro",
            "inputs": {
                "base_latent": ["1", 0],
                "target_latent": ["1", 0],
                "origin_latent": ["1", 0],
                "strength": 1.0,
                "mode": "Standard",
                "clamp_output": False,
                "clamp_min": -10.0,
                "clamp_max": 10.0,
                "allow_batch_expand": False,
                "ortho_per_channel": False,
                "clamp_mask_input": False,
                "enable_perf_logging": True
            }
        },
        "3": {
            "class_type": "SaveLatent",
            "inputs": {
                "samples": ["2", 0],
                "filename_prefix": "wayfinder_test"
            }
        }
    }
    
    validate_payload = {
        "prompt": workflow
    }
    
    try:
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json=validate_payload
        )
        assert response.status_code == 200, f"Workflow validation failed: {response.text}"
        result = response.json()
        assert 'prompt_id' in result, f"Workflow validation failed: {result}"
    except Exception as e:
        pytest.fail(f"Error testing workflow: {e}")


def main():
    print("=== Gimbal Node Suite Test ===\n")
    
    # Check ComfyUI connection
    if not check_comfyui_connection():
        print("\nPlease ensure ComfyUI is running and restart it to load the Gimbal nodes.")
        return 1
    
    # Check if Gimbal nodes are loaded
    nodes_loaded = check_gimbal_nodes()
    
    if not nodes_loaded:
        print("\n[WARN] Gimbal nodes are not loaded!")
        print("Please restart ComfyUI to load the custom nodes from:")
        print("  ComfyUI/custom_nodes/ComfyUI-Gimbal")
        return 1
    
    # Test a simple workflow
    if test_simple_workflow():
        print("\n[OK] All tests passed! Gimbal nodes are functional.")
        return 0
    else:
        print("\n[FAIL] Workflow test failed. Check the node implementations.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
