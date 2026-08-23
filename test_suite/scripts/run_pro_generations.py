import requests
import json

import sys

# Attempt dynamic check of ComfyUI port
COMFYUI_URL = "http://localhost:8188/prompt"
try:
    requests.get("http://localhost:8188/system_stats", timeout=2)
except requests.exceptions.ConnectionError:
    try:
        requests.get("http://localhost:8000/system_stats", timeout=2)
        COMFYUI_URL = "http://localhost:8000/prompt"
    except requests.exceptions.ConnectionError:
        pass # use 8188 as default fallback

def run_workflow(filepath, name):
    try:
        with open(filepath, 'r') as f:
            workflow_data = json.load(f)
        r = requests.post(COMFYUI_URL, json={'prompt': workflow_data}, timeout=5)
        print(f"[{name}] Status: {r.status_code}, Response: {r.text}")
    except Exception as e:
        print(f"[{name}] Error: {e}")

print("=== Queuing Pro Master Workflows ===")
run_workflow('../example_workflows/Pro_Directors_Cut_Grid.json', 'Director\'s Cut (Perfume)')
run_workflow('../example_workflows/Pro_Orthogonal_Style_Transfer.json', 'Orthogonal Transfer (Fashion)')
print("=== Done! ===")
