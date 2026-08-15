import json
import glob
import os

workflows = glob.glob('../example_workflows/Starter_*.json') + glob.glob('../example_workflows/Pro_*.json')

for wf in workflows:
    with open(wf, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Check if this is UI format, skip if so
    if "nodes" in data or "links" in data or "last_node_id" in data:
        continue
        
    modified = False
    
    vae_decode_nodes = {k: v for k, v in data.items() if v.get("class_type") == "VAEDecode"}
    ksamplers = {k: v for k, v in data.items() if v.get("class_type") == "KSampler"}
    
    if not ksamplers:
        continue
        
    # Heuristic: the last KSampler id is usually the base or the main one
    last_ks_id = str(max([int(k) for k in ksamplers.keys() if k.isdigit()]))
    base_ks = ksamplers[last_ks_id]
    
    for vae_id, vae_node in vae_decode_nodes.items():
        samples_input = vae_node.get("inputs", {}).get("samples")
        if not samples_input:
            continue
            
        parent_id = str(samples_input[0])
        parent_node = data.get(parent_id)
        
        if not parent_node:
            continue
            
        parent_class = parent_node.get("class_type")
        
        # If VAEDecode is already decoding a KSampler, we skip it
        if parent_class in ["KSampler", "KSamplerAdvanced"]:
            continue
            
        # Generate new integer id
        new_ks_id = str(max([int(k) for k in data.keys() if k.isdigit()] + [0]) + 1)
        
        new_ks = {
            "class_type": "KSampler",
            "inputs": {
                "seed": base_ks["inputs"].get("seed", 123456) + 123,
                "steps": base_ks["inputs"].get("steps", 25),
                "cfg": base_ks["inputs"].get("cfg", 7.0),
                "sampler_name": base_ks["inputs"].get("sampler_name", "euler"),
                "scheduler": base_ks["inputs"].get("scheduler", "normal"),
                "denoise": 0.45,
                "model": base_ks["inputs"]["model"],
                "positive": base_ks["inputs"]["positive"],
                "negative": base_ks["inputs"]["negative"],
                "latent_image": samples_input
            }
        }
        
        data[new_ks_id] = new_ks
        vae_node["inputs"]["samples"] = [new_ks_id, 0]
        modified = True
        
    if modified:
        with open(wf, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Retrofitted {os.path.basename(wf)}")

print("Done.")
