# 🌉 Gimbal Cross-Modal Bridge

> **VibeCheck:** 🟢 Stabilized &nbsp;|&nbsp; **Category:** `Gimbal/Flight Instruments` &nbsp;|&nbsp; **Class:** `GimbalCrossModalBridge`
>
> *Navigate latent space with precision flight instruments, not lottery prompts.*

---

Translates natural language instructions into latent vector directions by mapping keywords to calibrated channel signatures.

```
Returns: (LATENT, LATENT)  →  target_vector, origin_vector
```

The Cross-Modal Bridge is the **translation layer** between human language and latent space geometry. You write words like `"warm cinematic sharp"` and it produces a `target_vector` that points toward that concept in the latent space — ready to be fed into a Compass Pro or Manifold Explorer.

---

## 🗂️ When to Use It

| Use Case | Setup |
|---|---|
| **Text-steered style transfer** | Bridge → `target_latent` of Compass Pro (Orthogonal_Projection mode) |
| **2D mood matrix axes** | Two Bridges → `x_vector` + `y_vector` of Manifold Explorer |
| **Concept arithmetic** | Bridge A + Bridge B combined via Compass chain |
| **Semantic slider axis** | Bridge output defines a named direction (e.g. "warm") for a Semantic Slider |
| **Zero-shot material design** | Describe material properties; steer a base product latent |

---

## ⚡ Quick Wiring

```
[Empty Latent or KSampler output] ──► base_latent ──┐
[Text Input: "warm cinematic dark"] ─────────────────┤
                                              mapping_mode: Keyword_Heuristics
                                                     └──► 🌉 Cross-Modal Bridge
                                                          │
                                    target_vector ──────► target_latent of 🧭 Compass Pro
                                    origin_vector ──────► origin_latent of 🧭 Compass Pro
```

---

## 🎛️ Mapping Modes

### `Keyword_Heuristics` *(fast, offline, recommended for most use)*

Matches tokens from the text input against the built-in **`LATENT_SIGNATURES`** dictionary. Each matched keyword contributes a pre-calibrated 4-channel offset (SDXL) or padded N-channel offset (FLUX) to the delta.

- ✅ No CLIP model required
- ✅ Fully deterministic and reproducible
- ✅ Fast (CPU arithmetic, < 1ms)
- ⚠️ Limited to registered vocabulary — unknown words are silently skipped
- ⚠️ Treats all matched keywords as additive with equal weight

**How it works:**

```python
tokens = split(instruction)       # tokenize on whitespace/punctuation
for token in tokens:
    key = resolve_alias(token)    # normalize aliases (e.g. "warmer" → "warm")
    if key in LATENT_SIGNATURES:
        channel_offsets += LATENT_SIGNATURES[key]
delta = channel_offsets.broadcast_to(B, C, H, W)
```

---

### `Embedding_Projection` *(richer, requires CLIP)*

Projects full CLIP text embeddings (pooled output) into latent channel space using a small learned MLP:

```
CLIP pooled output [D_clip] → Linear(512) → SiLU → Linear(C) → channel_offsets [C]
```

Connect a CLIP conditioning output to the optional `conditioning` input. If a trained projection checkpoint exists at `models/crossmodal_proj_{D}_{C}.pt`, it is loaded automatically. Otherwise, an **untrained** (randomly initialized) projector is used — useful for experimentation, not production.

- ✅ Richer semantic coverage than keyword heuristics
- ✅ Handles novel phrasing not in the keyword dictionary
- ⚠️ Requires CLIP connection
- ⚠️ Untrained projector = random directions (still useful for exploration)

---

### `Manual_JSON`

Direct per-channel control via JSON payload in the text field:

```json
{
  "luminance": 0.3,
  "warm_cool": -0.2,
  "detail": 0.4,
  "channel_7": 0.15
}
```

Keys can be semantic names (`"luminance"`, `"warm_cool"`, `"green_mag"`, `"detail"`), channel indices (`"channel_0"` through `"channel_N"`), or keyword names (scales the full keyword signature by the value). Markdown code fences are stripped automatically.

---

## 📚 The LATENT_SIGNATURES Dictionary

These are the 30 calibrated keyword signatures built into the node. Each signature is a 4-element array `[Ch0, Ch1, Ch2, Ch3]` representing offsets in the SDXL 4-channel latent space:

| Keyword | Ch0 Luminance | Ch1 Warm/Cool | Ch2 Green/Mag | Ch3 Detail |
|---|---|---|---|---|
| `bright` | +0.25 | +0.05 | +0.05 | 0.00 |
| `dark` | −0.25 | −0.05 | −0.05 | 0.00 |
| `overexposed` | +0.50 | +0.10 | +0.10 | 0.00 |
| `underexposed` | −0.40 | −0.08 | −0.08 | 0.00 |
| `contrast` | +0.15 | 0.00 | 0.00 | +0.15 |
| `flat` | −0.10 | 0.00 | 0.00 | −0.10 |
| `punchy` | +0.20 | +0.05 | +0.05 | +0.20 |
| `sharp` | 0.00 | 0.00 | 0.00 | +0.35 |
| `soft` | 0.00 | 0.00 | 0.00 | −0.30 |
| `blurry` | 0.00 | 0.00 | 0.00 | −0.45 |
| `crisp` | +0.05 | 0.00 | 0.00 | +0.30 |
| `saturated` | +0.05 | +0.25 | +0.20 | +0.05 |
| `vivid` | +0.08 | +0.30 | +0.25 | +0.05 |
| `desaturated` | −0.05 | −0.25 | −0.20 | −0.05 |
| `muted` | −0.05 | −0.18 | −0.15 | −0.05 |
| `monochrome` | 0.00 | −0.40 | −0.40 | 0.00 |
| `warm` | +0.05 | +0.20 | −0.10 | 0.00 |
| `cool` | 0.00 | −0.20 | +0.10 | 0.00 |
| `golden` | +0.10 | +0.25 | −0.15 | +0.05 |
| `cold` | −0.05 | −0.25 | +0.15 | 0.00 |
| `dreamy` | +0.10 | −0.05 | −0.05 | −0.20 |
| `gritty` | −0.05 | 0.00 | +0.05 | +0.25 |
| `cinematic` | −0.10 | +0.05 | −0.05 | +0.20 |
| `faded` | −0.15 | −0.10 | −0.10 | −0.15 |
| `neon` | +0.05 | +0.35 | +0.30 | +0.10 |
| `pastel` | +0.20 | −0.15 | −0.10 | −0.20 |
| `moody` | −0.20 | +0.05 | 0.00 | +0.15 |
| `ethereal` | +0.15 | −0.10 | −0.05 | −0.25 |
| `underwater` | −0.15 | −0.10 | +0.25 | −0.05 |
| `fire` | +0.20 | +0.30 | −0.15 | +0.10 |

> **Reading the table:** Ch0 maps to overall luminance/exposure. Ch1 maps to warm/cool temperature. Ch2 maps to green/magenta saturation axis. Ch3 maps to high-frequency detail/sharpness. Keywords primarily affecting a single dimension have near-zero values in the other channels.

---

## 🔤 Keyword Aliases

You don't need to use exact dictionary spellings — common natural language variants are resolved automatically:

| What you type | Resolves to |
|---|---|
| `brighter`, `brighten` | `bright` |
| `darker`, `darken` | `dark` |
| `sharpened`, `sharpen` | `sharp` |
| `soften`, `softer` | `soft` |
| `saturate` | `saturated` |
| `desaturate` | `desaturated` |
| `warmth`, `warmer` | `warm` |
| `cooler`, `cooling` | `cool` |
| `hazy`, `foggy` | `dreamy` |
| `ocean`, `aquatic` | `underwater` |
| `flame`, `burning`, `hot` | `fire` |

Aliases are resolved before signature lookup — the alias table in code is `KEYWORD_ALIASES`.

---

## 🌊 FLUX.1 Behavior (16-Channel Latents)

SDXL uses 4 latent channels. FLUX.1 uses **16 channels**. When the connected `base_latent` has more than 4 channels, signatures are zero-padded to fill the extra channels:

```python
if C > len(signature):        # e.g. 16 > 4
    padded = zeros(C)
    padded[:4] = signature    # fill first 4 channels
    channel_offsets += padded
```

This means on FLUX, the primary 4 channels still receive their calibrated values, while channels 4–15 receive zero offset from keyword heuristics. The broad-spectrum weighting behavior:

- **Saturation-family keywords** (`saturated`, `vivid`, `neon`, etc.) — primary effect in channels 1–2, propagated loosely to channels 4–9 via zero-pad (i.e., no propagation in current version — treat these as Ch0–3 steering only on FLUX until a FLUX-calibrated signature set ships).
- **Structure keywords** (`contrast`, `sharp`, `crisp`) — primary effect in channels 0 and 3.
- **Texture keywords** (`gritty`, `soft`, `blurry`) — primary effect in Ch3; channels 8–15 unaffected.

> **Roadmap note:** FLUX-native 16-channel signatures are planned for a future release. For FLUX workflows today, use `Manual_JSON` mode to directly set per-channel offsets, or use `Embedding_Projection` with a FLUX-compatible CLIP model.

---

## 🖼️ Example Outputs

### Mirror polished liquid chrome — keywords: `cool sharp crisp monochrome bright cold`

![Mirror polished liquid chrome via Cross-Modal Bridge keyword steering](../../assets/test_runs/fresh_run6_exploration/09_chair_steered_liquid_chrome.png)

*`cool sharp crisp monochrome bright cold` → 6 keywords registered → additive channel offsets drive Ch1 negative (cool), Ch3 high positive (sharp/crisp), Ch1+2 heavily negative (monochrome)*

---

### Oxblood velvet — keywords: `warm saturated vivid dark moody fire`

![Oxblood velvet via Cross-Modal Bridge keyword steering](../../assets/test_runs/fresh_run6_exploration/09_chair_steered_oxblood_velvet.png)

*`warm saturated vivid dark moody fire` → Ch0 dark (−0.20 moody, −0.25 dark), Ch1 strong warm (+0.30 fire, +0.25 vivid), Ch3 texture detail (+0.15 moody)*

---

## 💡 Pro Tips

- **Combine keywords freely.** All matched signatures are summed. `"warm cinematic dark"` produces `[−0.10+0.05+0.05, 0.05+0.20+0.05, −0.05+0.00+0.05−0.10, 0.00+0.20+0.00]` = a rich combined offset with warm color temperature, slight underexposure, and cinematic detail boost.

- **For Compass Pro Orthogonal_Projection:** Connect `target_vector` → `target_latent` and `origin_vector` → `origin_latent` of Compass Pro. Set Compass to `Orthogonal_Projection` mode at `strength = 1.0–1.5`. This projects only the component of your base image that already aligns with the text-described direction — geometry lock is implicit.

- **For Manifold Explorer axes:** Run two Bridge nodes with different keyword prompts. Connect one's `target_vector` to `x_vector`, the other to `y_vector`. The grid maps the 2D intersection of both concepts.

- **Debugging zero output:** If nothing seems to change in your output, use **Show Text** on the `llm_instruction` field and verify your keywords appear in the LATENT_SIGNATURES table or aliases list above. Unknown tokens are silently dropped — no error is raised. This is intentional (allows free-form LLM output without crashing), but can be confusing.

- **Scaling strength via Compass:** The Bridge produces a fixed-magnitude delta. To scale it up or down, connect through Compass Pro with `mode = Standard` and adjust `strength`. Bridge strength itself is baked into the signature values — there is no strength parameter on the Bridge directly.

---

## ⚠️ Failure Cases

| Symptom | Cause | Fix |
|---|---|---|
| No change in output | All tokens unregistered | Check keywords against the LATENT_SIGNATURES table |
| Subtle/weak effect | Keywords have small signatures; or high-CFG KSampler overrides the delta | Increase Compass strength, or reduce KSampler CFG |
| VAE decode artifacts | Delta too large (many strong keywords summed) | Wrap in Compass with `clamp_output = True`, or reduce keywords |
| FLUX: effects only in first 4 channels | Expected behavior — FLUX 16ch signatures not yet available | Use `Manual_JSON` for full 16-channel control |
| `Embedding_Projection` produces garbage | No trained weights found; untrained projector is random | Acceptable for exploration — provide trained `.pt` checkpoint for production |

---

## 🔬 Under the Hood *(Power User)*

### Keyword Heuristics Pipeline

```
instruction string
    → re.split(r"[\s,.\-!?;:]+", lower)        # tokenize
    → _resolve_keyword(token)                    # alias + dict lookup
    → LATENT_SIGNATURES[key]                     # 4-element float list
    → sig_tensor padded to C channels            # broadcast to [B, C, H, W]
    → channel_offsets accumulated (additive)
→ delta = channel_offsets.view(1, C, 1, 1).expand(B, C, H, W).contiguous()
```

### Output Tensor Structure

- `target_vector["samples"]` = `base_samples + delta` — the steered latent
- `origin_vector["samples"]` = `base_samples.clone()` — an unmodified copy of the input

Both outputs share the same `base_latent` metadata (e.g. `noise_mask`), copied via `dict.copy()`.

### Why Return Both Vectors?

This interface mirrors the Compass Pro input contract directly:

```
Bridge.target_vector → Compass.target_latent
Bridge.origin_vector → Compass.origin_latent
```

The delta recomputed inside Compass is then `target_vector − origin_vector = delta` — the exact same delta computed in the Bridge. This round-trips cleanly and allows Compass to apply mode-specific transforms (Normalized, Slerp, etc.) to a Bridge-generated direction without re-specifying the origin.

### `torch.no_grad()` Scope

All tensor operations run inside `with torch.no_grad()`. Safe to run on CPU (keyword heuristics) or GPU (embedding projection). The embedding projection MLP is moved to `samples.device` automatically.

---

## ⚙️ Technical Reference

| Property | Value |
|---|---|
| ComfyUI class name | `GimbalCrossModalBridge` |
| Legacy aliases | `Gimbal_CrossModalBridge`, `Wayfinder_CrossModalBridge` |
| Function | `translate()` |
| Return types | `("LATENT", "LATENT")` |
| Return names | `("target_vector", "origin_vector")` |
| Category | `Gimbal/Flight Instruments` |
| Mapping modes | `Keyword_Heuristics`, `Embedding_Projection`, `Manual_JSON` |
| Registered keywords | 30 (SDXL-calibrated) |
| Alias entries | 13 |
| VRAM mode | `torch.no_grad()` |

---

*Form & Noise Atelier — Gimbal Node Suite*
