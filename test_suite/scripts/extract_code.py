"""
Extract Python code from documentation files and create proper node files
"""

import re
import os

def extract_node_code(content, node_number):
    """Extract code for a specific node from the documentation"""
    
    # Find the node section
    pattern = rf'<summary><b>Node {node_number}:.*?</b>.*?</summary>\s*```python(.*?)```\s*</details>'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def process_file(filepath):
    """Process a documentation file and extract all node codes"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dictionary to store extracted codes
    extracted = {}
    
    # Extract each node (1-5)
    for i in range(1, 6):
        code = extract_node_code(content, i)
        if code:
            # Determine which file this code belongs to
            if 'WayfinderManifold_Explorer' in code:
                extracted['wayfindermanifold_explorer.py'] = code
            elif 'Wayfinder_CrossModalBridge' in code:
                extracted['Wayfinder_crossmodal_bridge.py'] = code
            elif 'WayfinderGPS_Anchor' in code:
                extracted['wayfinder_gps_anchor.py'] = code
            elif 'Wayfinder_SemanticSlider' in code:
                extracted['wayfinder_semanticslider.py'] = code
            elif 'WayfinderCompass_Pro' in code:
                extracted['Wayfinder_compass.py'] = code
    
    return extracted

def main():
    """Main extraction process"""
    
    # Files to process
    doc_files = [
        'Wayfinder_compass.py',
        'wayfindermanifold_explorer.py',
        'Wayfinder_crossmodal_bridge.py',
        'wayfinder_gps_anchor.py',
        'wayfinder_semanticslider.py'
    ]
    
    all_extracted = {}
    
    # Process each documentation file
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            print(f"Processing {doc_file}...")
            extracted = process_file(doc_file)
            all_extracted.update(extracted)
    
    # Write the extracted code to proper Python files
    for filename, code in all_extracted.items():
        output_path = f"extracted_{filename}"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"✓ Extracted code to {output_path}")
    
    # Also update the original files with clean code
    print("\nUpdating original files with clean code...")
    for filename, code in all_extracted.items():
        if os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"✓ Updated {filename}")
    
    # Update the copied files in ComfyUI custom_nodes
    custom_nodes_path = r"C:\Users\Aarik\AppData\Local\Programs\ComfyUI\resources\ComfyUI\custom_nodes\Wayfinder"
    if os.path.exists(custom_nodes_path):
        print(f"\nUpdating files in ComfyUI custom_nodes...")
        for filename, code in all_extracted.items():
            dest_path = os.path.join(custom_nodes_path, filename)
            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"✓ Updated {dest_path}")

if __name__ == "__main__":
    main()
