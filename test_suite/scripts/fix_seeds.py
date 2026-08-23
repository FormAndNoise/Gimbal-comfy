import json
import glob
import os

workflows = glob.glob('../example_workflows/*.json')

for wf in workflows:
    with open(wf, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified = False
    first_seed = None
    
    for node_id, node in data.items():
        if isinstance(node, dict) and node.get("class_type") == "KSampler":
            inputs = node.get("inputs", {})
            if "seed" in inputs:
                if inputs.get("denoise") == 1.0:
                    if first_seed is None:
                        first_seed = inputs["seed"]
                    else:
                        if inputs["seed"] != first_seed:
                            inputs["seed"] = first_seed
                            modified = True

    if modified:
        print(f"Fixed seeds in {os.path.basename(wf)}")
        with open(wf, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

print("Done fixing seeds.")
