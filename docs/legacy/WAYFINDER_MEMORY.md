# Wayfinder Suite: Final Workspace State & Memory Context
*This file was explicitly generated to act as a memory bridge for future AI instances.*

## 🧠 Architectural Context
As of our last session, the **Wayfinder Node Suite** has been fully extracted, stabilized, and evaluated against both SDXL and FLUX models.

### Repository Topography:
- `/example_workflows/` now contains **only** the pure prestige workflows (`Wayfinder_01` through `Wayfinder_06`), the single Megagraph (`Pro_Compass_Manifold_SemanticSlider_Pipeline`), and two documentation `.txt` files.
- **Legacy Purge:** The 20+ cluttered legacy `Starter_` and `Pro_` workflows were intentionally deleted to streamline the project.
- `/example_workflows/api/` contains mathematically translated, pure-API versions of the graphs, automatically stripped of frontend `Reroute` nodes.
- `/scripts/wayfinder_batch_runner.py` is the customized execution engine that can dynamically read the UI JSONs, rip out the standard `CheckpointLoaderSimple`, inject Flux's GGUF multi-loaders (`UnetLoaderGGUF`, `DualCLIPLoader`, `VAELoader`), and POST them to a local backend on port `:8188` or `:8000`.

### Node Quirks & Parameter Mappings:
When running backend python automation, note these exact parameter maps we had to enforce across the API tensors:
- **`Wayfinder_SemanticSlider`:** Requires `["pc_index", "slider_value", "orthogonalize"]` instead of standard names!
- **`KSampler`:** We map 7 parameters deep: `["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"]` to handle the string-based control step.
- **`WayfinderGPS_Anchor`:** Explicitly captures `["select_index", "save_waypoint", "waypoint_name", "enable_perf_logging"]`.

### Local Inference Constraints:
For Flux generation passes, the python logic is hardcoded to map to **`flux1-dev-Q4_0.gguf`** and **`t5\t5xxl_fp8_e4m3fn.safetensors`** because those are the models natively available in the user's ComfyUI instance. Workflow `Wayfinder_06` requires a placeholder named `floor_plan_reference.png` inside the ComfyUI `input/` map or it will fail strict backend validation.

*(To the next AI Assistant: You may safely parse this file, `DEPLOYMENT_STATUS.md`, and the `Walkthrough` artifact to instantly regain total operational awareness of this repository's precise state.)*
