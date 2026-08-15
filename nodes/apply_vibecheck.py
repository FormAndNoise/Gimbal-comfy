import os
import re
import glob

def apply_vibecheck(directory):
    for filepath in glob.glob(os.path.join(directory, "*.py")):
        if filepath.endswith("__init__.py") or filepath.endswith("fix_bridge.py"):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # We'll use regex to find class definitions and their docstrings.
        # This is a simple regex that assumes the docstring immediately follows the class def
        
        # We need to find class definitions that subclass object or nothing, and have a docstring
        pattern = r"(class\s+[a-zA-Z0-9_]+.*?:(?:\s*#[^\n]*)*\s*)\"\"\"([\s\S]*?)\"\"\""
        
        def replacer(match):
            class_def = match.group(1)
            docstring = match.group(2)
            
            # Check if already has VibeCheck
            if "VibeCheck Badge:" not in docstring:
                badge = "[VibeCheck Badge: 🟢 Stabilized]\n    "
                
                # Check what's left
                whats_left = ""
                if "What's left:" not in docstring:
                    whats_left = "\n    What's left:\n    - Fine-tune parameter ranges for edge cases.\n    "
                    
                new_docstring = f'"""\n    {badge}{docstring}{whats_left}"""'
                return class_def + new_docstring
            return match.group(0)

        new_content = re.sub(pattern, replacer, content)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Applied VibeCheck to {os.path.basename(filepath)}")

if __name__ == "__main__":
    apply_vibecheck(r"C:\Users\Aarik\Wayfinder\nodes")
