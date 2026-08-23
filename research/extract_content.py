import os
import re
import traceback

TEXT_FILE = r"C:\Users\Aarik\Wayfinder\example_workflows\Let me build the workflow JSON file.txt"

os.chdir(r"C:\Users\Aarik\Wayfinder")

try:
    with open(TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    py_matches = re.finditer(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    py_content = None
    for match in py_matches:
        if "import json" in match.group(1) and "WayfinderGPS_Load" in match.group(1):
            py_content = match.group(1).strip()
            break
            
    if py_content:
        with open("wayfinder_gps_load.py", "w", encoding='utf-8') as f:
            f.write(py_content + "\n")

    json_matches = re.finditer(r"📦 Workflow \d+: `(.*?)`.*?```json\s*(.*?)\s*```", text, re.DOTALL)
    for match in json_matches:
        filename = match.group(1)
        content = match.group(2).strip()
        filepath = os.path.join("example_workflows", filename)
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(content + "\n")
    
    # Also extract deliverable 1
    json_dlv = re.finditer(r"📦 Deliverable 1: `(.*?)`.*?```json\s*(.*?)\s*```", text, re.DOTALL)
    for match in json_dlv:
        filename = match.group(1)
        content = match.group(2).strip()
        filepath = os.path.join("example_workflows", filename)
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(content + "\n")

except Exception as e:
    with open("error.log", "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
