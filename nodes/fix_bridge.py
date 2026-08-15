import os
import re

def modify_bridge(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update INPUT_TYPES to take optional CONDITIONING
    input_hook = r"(\"mapping_mode\": \(\[\"Keyword_Heuristics\", \"Embedding_Projection\", \"Manual_JSON\"\], \{\"default\": \"Keyword_Heuristics\"\}\),\n\s*\},)"
    input_repl = """\\1
            "optional": {
                "conditioning": ("CONDITIONING",),
            }"""
    content = re.sub(input_hook, input_repl, content)

    # 2. Update translate signature
    sig_hook = r"(def translate\(\n\s*self,\n\s*llm_instruction: str,\n\s*base_latent: Dict\[str, Any\],\n\s*mapping_mode: str,)\n\s*\) -> Tuple\[Dict\[str, Any\], Dict\[str, Any\]\]:"
    sig_repl = """\\1
        conditioning=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:"""
    content = re.sub(sig_hook, sig_repl, content)

    # 3. Update dispatch logic
    disp_hook = r"(delta = dispatch\[mapping_mode\]\(llm_instruction, samples\))"
    disp_repl = """if mapping_mode == "Embedding_Projection":
                delta = self._embedding_projection(llm_instruction, samples, conditioning)
            else:
                delta = dispatch[mapping_mode](llm_instruction, samples)"""
    content = re.sub(disp_hook, disp_repl, content)

    # 4. Update _embedding_projection
    proj_hook = r"(def _embedding_projection\(self, instruction: str, samples: torch\.Tensor\) -> torch\.Tensor:[\s\S]*?return delta)"
    proj_repl = """def _embedding_projection(self, instruction: str, samples: torch.Tensor, conditioning=None) -> torch.Tensor:
        \"\"\"
        Projects LLM/CLIP hidden-state into latent channel dimensions.
        \"\"\"
        B, C, H, W = samples.shape
        
        if conditioning is not None and len(conditioning) > 0:
            cond_data = conditioning[0]
            cond_dict = cond_data[1] if len(cond_data) > 1 else {}
            pooled = cond_dict.get("pooled_output")
            
            if pooled is not None:
                log.info(f"Embedding_Projection: projecting {pooled.shape[-1]}-dim CLIP pooled output to {C} latent channels.")
                pooled = pooled.to(device=samples.device, dtype=torch.float32)
                
                # Deterministic projection matrix based on text hash so it's stable per prompt
                seed = hash(instruction) & 0x7FFFFFFF
                gen = torch.Generator(device=samples.device).manual_seed(seed)
                proj = torch.randn(pooled.shape[-1], C, device=samples.device, generator=gen) * 0.05
                
                channel_offsets = torch.matmul(pooled, proj)
                
                if channel_offsets.shape[0] == 1 and B > 1:
                    channel_offsets = channel_offsets.expand(B, -1)
                elif channel_offsets.shape[0] != B:
                    # truncate or pad
                    channel_offsets = channel_offsets[:B]
                
                delta = channel_offsets.view(channel_offsets.shape[0], C, 1, 1).expand(-1, -1, H, W).contiguous()
                return delta
                
        # Fallback
        log.info(f"Embedding_Projection: No conditioning provided, falling back to hash-based noise.")
        seed = hash(instruction) & 0x7FFFFFFF
        generator = torch.Generator(device=samples.device).manual_seed(seed)
        delta = torch.randn(B, C, H, W, generator=generator, device=samples.device) * 0.02
        return delta"""
    content = re.sub(proj_hook, proj_repl, content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Modified crossmodal bridge successfully!")

if __name__ == "__main__":
    modify_bridge(r"C:\Users\Aarik\Wayfinder\nodes\Wayfinder_crossmodal_bridge.py")
