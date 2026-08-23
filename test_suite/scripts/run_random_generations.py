import requests
import json
import time

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

def run_workflow(workflow_data):
    try:
        r = requests.post(COMFYUI_URL, json={'prompt': workflow_data}, timeout=5)
        print(f"Status: {r.status_code}, Response: {r.text}")
    except Exception as e:
        print(f"Error: {e}")

print("=== Queuing Random Gimbal Generations ===")

# ---------------------------------------------------------
# 1. Compass Steering
# ---------------------------------------------------------
with open('../example_workflows/Starter_Compass_Steering.json', 'r') as f:
    wf_compass = json.load(f)

# Run 1: Medieval to Sci-Fi
wf_compass["6"]["inputs"]["text"] = "a bustling medieval marketplace, highly detailed, fantasy"
wf_compass["7"]["inputs"]["text"] = "a high-tech alien space station, sci-fi, glowing lights"
wf_compass["14"]["inputs"]["filename_prefix"] = "Gimbal_Compass_Medieval_Alien"
print("Queuing Compass Steering 1: Medieval -> Alien...")
run_workflow(wf_compass)

# Run 2: Terrifying to Cute
wf_compass["6"]["inputs"]["text"] = "a close up of a terrifying giant tarantula spider, hairy"
wf_compass["7"]["inputs"]["text"] = "a fluffy cute stuffed teddy bear toy, soft"
wf_compass["14"]["inputs"]["filename_prefix"] = "Gimbal_Compass_Spider_Bear"
print("Queuing Compass Steering 2: Tarantula -> Teddy Bear...")
run_workflow(wf_compass)

# ---------------------------------------------------------
# 2. Manifold Grid
# ---------------------------------------------------------
with open('../example_workflows/Starter_Manifold_Grid.json', 'r') as f:
    wf_grid = json.load(f)

# Run 1: Piano + Jello + Fire
wf_grid["6"]["inputs"]["text"] = "a grand piano in a concert hall"
wf_grid["12"]["inputs"]["llm_instruction"] = "make it look like it is made of green jello"
wf_grid["13"]["inputs"]["llm_instruction"] = "make it engulfed in roaring flames"
wf_grid["16"]["inputs"]["filename_prefix"] = "Gimbal_Grid_Piano_Jello_Fire"
print("Queuing Manifold Grid 1: Piano (X=Jello, Y=Fire)...")
run_workflow(wf_grid)

# Run 2: Coffee + Gothic + Underwater
wf_grid["6"]["inputs"]["text"] = "a simple ceramic cup of hot coffee on a table"
wf_grid["12"]["inputs"]["llm_instruction"] = "make it gothic architecture style with gargoyles"
wf_grid["13"]["inputs"]["llm_instruction"] = "make it completely underwater with bubbles and fish"
wf_grid["16"]["inputs"]["filename_prefix"] = "Gimbal_Grid_Coffee_Gothic_Water"
print("Queuing Manifold Grid 2: Coffee (X=Gothic, Y=Underwater)...")
run_workflow(wf_grid)

# ---------------------------------------------------------
# 3. Style Transfer
# ---------------------------------------------------------
with open('../example_workflows/Starter_Style_Transfer.json', 'r') as f:
    wf_style = json.load(f)

# Run 1: Haunted Toaster
wf_style["3"]["inputs"]["text"] = "a modern clean minimalist living room"
wf_style["4"]["inputs"]["text"] = "a haunted Victorian mansion interior with cobwebs and ghosts and dark shadows"
wf_style["5"]["inputs"]["text"] = "a shiny new stainless steel toaster appliance"
wf_style["12"]["inputs"]["filename_prefix"] = "Gimbal_Style_Haunted_Toaster"
print("Queuing Style Transfer 1: Haunted Victorian Toaster...")
run_workflow(wf_style)

# Run 2: Nebula Puppy
wf_style["3"]["inputs"]["text"] = "a clear blue sunny sky, plain"
wf_style["4"]["inputs"]["text"] = "a swirling chaotic purple and gold nebula in deep space, cosmic"
wf_style["5"]["inputs"]["text"] = "a happy golden retriever puppy dog"
wf_style["12"]["inputs"]["filename_prefix"] = "Gimbal_Style_Nebula_Puppy"
print("Queuing Style Transfer 2: Cosmic Nebula Puppy...")
run_workflow(wf_style)

print("=== All jobs queued successfully! ===")
