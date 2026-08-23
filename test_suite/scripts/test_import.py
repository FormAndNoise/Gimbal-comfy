import sys
import importlib
import traceback

sys.path.append('C:/Users/Aarik/ComfyUI-Installs/ComfyUI (1)/ComfyUI/custom_nodes')
sys.path.append('C:/Users/Aarik/ComfyUI-Installs/ComfyUI (1)/ComfyUI')

try:
    importlib.import_module('Wayfinder')
    print("SUCCESS")
except Exception as e:
    traceback.print_exc()
