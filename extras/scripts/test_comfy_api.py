import urllib.request
import json
import time

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data)
    try:
        response = urllib.request.urlopen(req)
        return json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        print("Details:", e.read().decode('utf-8'))
        return None
    except Exception as e:
        print(f"Error queueing prompt: {e}")
        return None

def main():
    # Load the API workflow
    try:
        with open("example_workflows/api/API_Wayfinder_03_ManifoldGrid.json", "r") as f:
            prompt = json.load(f)
    except Exception as e:
        print(f"Error reading workflow: {e}")
        return

    # Strip nodes that might be missing on a fresh install
    nodes_to_delete = []
    for node_id, node_data in prompt.items():
        if node_data.get("class_type") == "ShowText|pysssss":
            nodes_to_delete.append(node_id)
            
    for node_id in nodes_to_delete:
        del prompt[node_id]

    print("Queueing Wayfinder_03_ManifoldGrid.json via API...")
    result = queue_prompt(prompt)
    if result:
        print(f"Successfully queued! Prompt ID: {result['prompt_id']}")
        print("Check the ComfyUI terminal or UI for generation progress.")
    else:
        print("Failed to queue. Make sure ComfyUI is running and Wayfinder nodes are loaded.")

if __name__ == "__main__":
    main()
