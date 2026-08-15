import json
import urllib.request
import urllib.error
from wayfinder_batch_runner import convert_to_api

COMFY_URL = "http://127.0.0.1:8000"

def get_checkpoints():
    try:
        r = urllib.request.urlopen(f"{COMFY_URL}/object_info/CheckpointLoaderSimple")
        d = json.loads(r.read())
        return d['CheckpointLoaderSimple']['input']['required']['ckpt_name'][0]
    except Exception as e:
        print(f"Failed to fetch checkpoints: {e}")
        return []

def run_all_checkpoints():
    workflow_path = "../example_workflows/Wayfinder_01_ConceptBlender.json"
    api_data = convert_to_api(workflow_path)
    
    # find checkpoint node
    ckpt_node_id = None
    for k, v in api_data.items():
        if v.get("class_type") == "CheckpointLoaderSimple":
            ckpt_node_id = k
            break
            
    if not ckpt_node_id:
        print("Couldn't find CheckpointLoaderSimple node in the workflow.")
        return
        
    checkpoints = get_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints.")
    
    for ckpt in checkpoints:
        print(f"Queueing generation for {ckpt}...")
        api_data[ckpt_node_id]["inputs"]["ckpt_name"] = ckpt
        
        req = urllib.request.Request(f"{COMFY_URL}/prompt", 
                                     data=json.dumps({"prompt": api_data}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req)
            resp = json.loads(r.read())
            if "prompt_id" in resp:
                print(f"  [OK] Successfully queued with prompt_id: {resp['prompt_id']}")
            else:
                print(f"  [WARN] Queued but no prompt_id? Response: {resp}")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            print(f"  [ERROR] Failed to queue {ckpt}. HTTP {e.code}: {error_body}")
        except Exception as e:
            print(f"  [ERROR] Failed to queue {ckpt}: {e}")

if __name__ == "__main__":
    run_all_checkpoints()
