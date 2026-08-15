import json
import glob
import os
import requests

widget_mappings = {
    'EmptyLatentImage': ['width', 'height', 'batch_size'],
    'KSampler': ['seed', 'control_after_generate', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise'],
    'CLIPTextEncode': ['text'],
    'CheckpointLoaderSimple': ['ckpt_name'],
    'SaveImage': ['filename_prefix'],
    'VAEDecode': [],
    'VAEEncode': [],
    'LoadImage': ['image', 'upload'],
    'ShowText|pysssss': ['text'],
    'ShowText': ['text'],
    'Wayfinder_CrossModalBridge': ['llm_instruction', 'mapping_mode'],
    'WayfinderManifold_Explorer': ['grid_size_x', 'grid_size_y', 'x_strength', 'y_strength', 'interpolation_mode', 'normalize_vectors', 'clamp_output', 'clamp_min', 'clamp_max', 'enable_perf_logging'],
    'WayfinderCompass_Pro': ['strength', 'mode', 'clamp_output', 'clamp_min', 'clamp_max', 'allow_batch_expand', 'ortho_per_channel', 'clamp_mask_input', 'enable_perf_logging'],
    'WayfinderGPS_Anchor': ['select_index', 'save_waypoint', 'waypoint_name', 'enable_perf_logging'],
    'Wayfinder_SemanticSlider': ['pc_index', 'slider_value', 'orthogonalize'],
    'WayfinderConceptBlender': ['blend_ratio', 'mode'],
    'WayfinderGPS_Load': ['waypoint_name', 'restore_mode'],
    'LoraLoader': ['lora_name', 'strength_model', 'strength_clip'],
    'PreviewImage': [],
}

def convert_to_api(wf_path):
    with open(wf_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    api_format = {}
    nodes_by_id = {str(n['id']): n for n in data.get('nodes', [])}
    links = data.get('links', [])
    
    inputs_for_node = {str(n['id']): {} for n in data.get('nodes', [])}
    
    for n in data.get('nodes', []):
        nid = str(n['id'])
        if 'inputs' in n:
            for inp in n['inputs']:
                if inp.get('link') is not None:
                    for l in links:
                        if l[0] == inp['link']:
                            inputs_for_node[nid][inp['name']] = [str(l[1]), l[2]]
                            break
                            
    # For nodes missing 'inputs' like PreviewImage
    for l in links:
        tgt_id = str(l[3])
        if tgt_id in nodes_by_id:
            tgt_node = nodes_by_id[tgt_id]
            if 'inputs' not in tgt_node or len(tgt_node['inputs']) == 0:
                in_type = l[5]
                name = 'images' if in_type == 'IMAGE' else 'latent_image' if in_type == 'LATENT' else 'text' if in_type == 'STRING' else f'input_{l[4]}'
                inputs_for_node[tgt_id][name] = [str(l[1]), l[2]]

    for node in data.get('nodes', []):
        node_id = str(node['id'])
        node_type = node['type']
        
        if node_type == 'ShowText':
            node_type = 'ShowText|pysssss'
            
        inputs = {}
        
        widgets_values = node.get('widgets_values', [])
        widget_names = widget_mappings.get(node_type, [])
        for i, val in enumerate(widgets_values):
            if i < len(widget_names):
                inputs[widget_names[i]] = val
            else:
                inputs[f'widget_{i}'] = val

        for k, v in inputs_for_node[node_id].items():
            inputs[k] = v

        api_format[node_id] = {
            'class_type': node_type,
            'inputs': inputs
        }
    return api_format

COMFY_URL = 'http://127.0.0.1:8188/prompt'
try:
    requests.get('http://127.0.0.1:8188/system_stats', timeout=2)
except:
    COMFY_URL = 'http://127.0.0.1:8000/prompt'

workflows = glob.glob(r'H:\Wayfinder\extras\workflows\*.json')
for wf in workflows:
    basename = os.path.basename(wf)
    print(f'Queueing {basename}...')
    try:
        data = convert_to_api(wf)
        r = requests.post(COMFY_URL, json={'prompt': data})
        if r.status_code == 200:
            print(f'  [>] Queued {basename} - Prompt ID: {r.json().get("prompt_id")}')
        else:
            print(f'  [X] Failed {basename}: {r.text}')
    except Exception as e:
        print(f'  [!] Error: {e}')
