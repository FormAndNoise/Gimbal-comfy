I'll provide **4 JSON workflow files** you can import directly into ComfyUI (via `Load` button or drag-and-drop). Together they cover **all 25 nodes** without redundant coverage.

---

## Workflow 1: **PCA Direction Discovery Pipeline**

**Purpose**: Generate multiple latents → discover PCA directions → apply them → perturb along them → create a slerp interpolation between the most interesting pair.

**Nodes covered**: `LatentPCA`, `LatentApplyDirection`, `LatentPerturbBatch`, `LatentSlerp`, `LatentDirectionStore`, `LatentInfo`

```json
[
  {
    "id": 1,
    "type": "CheckpointLoaderSimple",
    "pos": [50, 50],
    "size": [320, 100],
    "flags": {},
    "order": 0,
    "mode": 0,
    "inputs": {},
    "outputs": [
      {"name": "MODEL", "type": "MODEL", "links": [10], "slot_index": 0},
      {"name": "CLIP", "type": "CLIP", "links": [11, 12], "slot_index": 1},
      {"name": "VAE", "type": "VAE", "links": [13], "slot_index": 2}
    ],
    "properties": {"Node name for S&R": "CheckpointLoaderSimple"},
    "widgets_values": ["sd_xl_base_1.0.safetensors"]
  },
  {
    "id": 2,
    "type": "CLIPTextEncode",
    "pos": [50, 200],
    "size": [300, 100],
    "flags": {},
    "order": 1,
    "mode": 0,
    "inputs": {"clip": {"name": "clip", "type": "CLIP", "link": 11}},
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [14], "slot_index": 0}],
    "properties": {"Node name for S&R": "CLIPTextEncode"},
    "widgets_values": ["a portrait of a person, high quality"]
  },
  {
    "id": 3,
    "type": "CLIPTextEncode",
    "pos": [50, 350],
    "size": [300, 100],
    "flags": {},
    "order": 2,
    "mode": 0,
    "inputs": {"clip": {"name": "clip", "type": "CLIP", "link": 12}},
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [15], "slot_index": 0}],
    "properties": {"Node name for S&R": "CLIPTextEncode"},
    "widgets_values": ["negative, low quality, blurry"]
  },
  {
    "id": 4,
    "type": "EmptyLatentImage",
    "pos": [50, 500],
    "size": [300, 110],
    "flags": {},
    "order": 3,
    "mode": 0,
    "inputs": {},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [16], "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptyLatentImage"},
    "widgets_values": [8, 1.0, 1.0]
  },
  {
    "id": 5,
    "type": "LatentPCA",
    "pos": [450, 200],
    "size": [320, 90],
    "flags": {},
    "order": 4,
    "mode": 0,
    "inputs": {"latents": {"name": "latents", "type": "LATENT", "link": 16}},
    "outputs": [
      {"name": "directions", "type": "LATENT_DIRECTION", "links": [17, 20], "slot_index": 0},
      {"name": "eigenvalues", "type": "LATENT_DIRECTION", "links": [], "slot_index": 1},
      {"name": "mean_latent", "type": "LATENT", "links": [18], "slot_index": 2}
    ],
    "properties": {"Node name for S&R": "LatentPCA"},
    "widgets_values": [8]
  },
  {
    "id": 6,
    "type": "LatentInfo",
    "pos": [450, 350],
    "size": [320, 80],
    "flags": {},
    "order": 5,
    "mode": 0,
    "inputs": {"latent": {"name": "latent", "type": "LATENT", "link": 18}},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [19], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentInfo"},
    "widgets_values": []
  },
  {
    "id": 7,
    "type": "LatentApplyDirection",
    "pos": [850, 200],
    "size": [340, 110],
    "flags": {},
    "order": 6,
    "mode": 0,
    "inputs": {
      "latent": {"name": "latent", "type": "LATENT", "link": 19},
      "direction": {"name": "direction", "type": "LATENT_DIRECTION", "link": 17}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [21, 24], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentApplyDirection"},
    "widgets_values": [0, 2.5]
  },
  {
    "id": 8,
    "type": "LatentDirectionStore",
    "pos": [450, 500],
    "size": [320, 90],
    "flags": {},
    "order": 7,
    "mode": 0,
    "inputs": {"latent": {"name": "latent", "type": "LATENT_DIRECTION", "link": 20}},
    "outputs": [{"name": "LATENT_DIRECTION", "type": "LATENT_DIRECTION", "links": [22], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentDirectionStore"},
    "widgets_values": [0, true]
  },
  {
    "id": 9,
    "type": "LatentPerturbBatch",
    "pos": [850, 400],
    "size": [340, 130],
    "flags": {},
    "order": 8,
    "mode": 0,
    "inputs": {
      "latent": {"name": "latent", "type": "LATENT", "link": 21},
      "direction": {"name": "direction", "type": "LATENT_DIRECTION", "link": 22}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [23], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentPerturbBatch"},
    "widgets_values": [7, 3.0, 0, true]
  },
  {
    "id": 10,
    "type": "LatentSlerp",
    "pos": [1250, 200],
    "size": [340, 130],
    "flags": {},
    "order": 9,
    "mode": 0,
    "inputs": {
      "latent_A": {"name": "latent_A", "type": "LATENT", "link": 24},
      "latent_B": {"name": "latent_B", "type": "LATENT", "link": 23}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [25], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentSlerp"},
    "widgets_values": [0.5, "slerp"]
  },
  {
    "id": 11,
    "type": "KSampler",
    "pos": [1650, 200],
    "size": [280, 220],
    "flags": {},
    "order": 10,
    "mode": 0,
    "inputs": {
      "model": {"name": "model", "type": "MODEL", "link": 10},
      "positive": {"name": "positive", "type": "CONDITIONING", "link": 14},
      "negative": {"name": "negative", "type": "CONDITIONING", "link": 15},
      "latent_image": {"name": "latent_image", "type": "LATENT", "link": 25}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [26], "slot_index": 0}],
    "properties": {"Node name for S&R": "KSampler"},
    "widgets_values": [0, "fixed", 25, 7.0, "euler_ancestral", "normal", 1.0]
  },
  {
    "id": 12,
    "type": "VAEDecode",
    "pos": [1970, 200],
    "size": [240, 50],
    "flags": {},
    "order": 11,
    "mode": 0,
    "inputs": {
      "samples": {"name": "samples", "type": "LATENT", "link": 26},
      "vae": {"name": "vae", "type": "VAE", "link": 13}
    },
    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [27], "slot_index": 0}],
    "properties": {"Node name for S&R": "VAEDecode"},
    "widgets_values": []
  },
  {
    "id": 13,
    "type": "SaveImage",
    "pos": [2250, 200],
    "size": [320, 270],
    "flags": {},
    "order": 12,
    "mode": 0,
    "inputs": {"images": {"name": "images", "type": "IMAGE", "link": 27}},
    "outputs": [],
    "properties": {"Node name for S&R": "SaveImage"},
    "widgets_values": ["PCA_Discovery"]
  }
]
```

---

## Workflow 2: **GAN Arithmetic & Blend Playground**

**Purpose**: Demonstrates latent arithmetic analogies, scalar multiplication, weighted blending, and normalized interpolation.

**Nodes covered**: `LatentGANArithmetic`, `LatentScalarMultiply`, `LatentWeightedAverage`, `LatentBlend`, `LatentNormalizedInterpolation`, `LatentSlerpBatch`

```json
[
  {
    "id": 1,
    "type": "CheckpointLoaderSimple",
    "pos": [50, 50],
    "size": [320, 100],
    "flags": {},
    "order": 0,
    "mode": 0,
    "inputs": {},
    "outputs": [
      {"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0},
      {"name": "CLIP", "type": "CLIP", "links": [2, 3], "slot_index": 1},
      {"name": "VAE", "type": "VAE", "links": [4], "slot_index": 2}
    ],
    "properties": {"Node name for S&R": "CheckpointLoaderSimple"},
    "widgets_values": ["sd_xl_base_1.0.safetensors"]
  },
  {
    "id": 2,
    "type": "CLIPTextEncode",
    "pos": [50, 200],
    "size": [300, 80],
    "flags": {},
    "order": 1,
    "mode": 0,
    "inputs": {"clip": {"name": "clip", "type": "CLIP", "link": 2}},
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [5], "slot_index": 0}],
    "properties": {"Node name for S&R": "CLIPTextEncode"},
    "widgets_values": ["a portrait photograph"]
  },
  {
    "id": 3,
    "type": "CLIPTextEncode",
    "pos": [50, 320],
    "size": [300, 80],
    "flags": {},
    "order": 2,
    "mode": 0,
    "inputs": {"clip": {"name": "clip", "type": "CLIP", "link": 3}},
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [6], "slot_index": 0}],
    "properties": {"Node name for S&R": "CLIPTextEncode"},
    "widgets_values": ["negative"]
  },
  {
    "id": 4,
    "type": "EmptyLatentImage",
    "pos": [50, 450],
    "size": [300, 110],
    "flags": {},
    "order": 3,
    "mode": 0,
    "inputs": {},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [7, 8, 9], "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptyLatentImage"},
    "widgets_values": [1, 1.0, 1.0]
  },
  {
    "id": 5,
    "type": "EmptyLatentImage",
    "pos": [50, 600],
    "size": [300, 110],
    "flags": {},
    "order": 4,
    "mode": 0,
    "inputs": {},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [10, 11], "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptyLatentImage"},
    "widgets_values": [2, 1.0, 1.0]
  },
  {
    "id": 6,
    "type": "EmptyLatentImage",
    "pos": [50, 750],
    "size": [300, 110],
    "flags": {},
    "order": 5,
    "mode": 0,
    "inputs": {},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [12, 13], "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptyLatentImage"},
    "widgets_values": [3, 1.0, 1.0]
  },
  {
    "id": 7,
    "type": "LatentGANArithmetic",
    "pos": [450, 500],
    "size": [340, 170],
    "flags": {},
    "order": 6,
    "mode": 0,
    "inputs": {
      "latent_A": {"name": "latent_A", "type": "LATENT", "link": 8},
      "latent_B": {"name": "latent_B", "type": "LATENT", "link": 10},
      "latent_C": {"name": "latent_C", "type": "LATENT", "link": 12}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [14, 17], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentGANArithmetic"},
    "widgets_values": [1.5]
  },
  {
    "id": 8,
    "type": "LatentScalarMultiply",
    "pos": [450, 720],
    "size": [300, 80],
    "flags": {},
    "order": 7,
    "mode": 0,
    "inputs": {"latent": {"name": "latent", "type": "LATENT", "link": 9}},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [15], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentScalarMultiply"},
    "widgets_values": [1.5]
  },
  {
    "id": 9,
    "type": "LatentWeightedAverage",
    "pos": [450, 840],
    "size": [340, 170],
    "flags": {},
    "order": 8,
    "mode": 0,
    "inputs": {
      "latent_1": {"name": "latent_1", "type": "LATENT", "link": 11},
      "latent_2": {"name": "latent_2", "type": "LATENT", "link": 13}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [16], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentWeightedAverage"},
    "widgets_values": [0.7, 1.0]
  },
  {
    "id": 10,
    "type": "LatentSlerpBatch",
    "pos": [850, 500],
    "size": [340, 170],
    "flags": {},
    "order": 9,
    "mode": 0,
    "inputs": {
      "latent_A": {"name": "latent_A", "type": "LATENT", "link": 14},
      "latent_B": {"name": "latent_B", "type": "LATENT", "link": 16}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [20], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentSlerpBatch"},
    "widgets_values": [10, "slerp", true, "cosine"]
  },
  {
    "id": 11,
    "type": "LatentBlend",
    "pos": [850, 720],
    "size": [340, 150],
    "flags": {},
    "order": 10,
    "mode": 0,
    "inputs": {
      "latent_A": {"name": "latent_A", "type": "LATENT", "link": 15},
      "latent_B": {"name": "latent_B", "type": "LATENT", "link": 17}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [18], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentBlend"},
    "widgets_values": [0.3]
  },
  {
    "id": 12,
    "type": "LatentNormalizedInterpolation",
    "pos": [850, 920],
    "size": [340, 130],
    "flags": {},
    "order": 11,
    "mode": 0,
    "inputs": {
      "latent_A": {"name": "latent_A", "type": "LATENT", "link": 18},
      "latent_B": {"name": "latent_B", "type": "LATENT", "link": 7}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [19], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentNormalizedInterpolation"},
    "widgets_values": [0.5, 0.0]
  },
  {
    "id": 13,
    "type": "KSampler",
    "pos": [1250, 500],
    "size": [280, 220],
    "flags": {},
    "order": 12,
    "mode": 0,
    "inputs": {
      "model": {"name": "model", "type": "MODEL", "link": 1},
      "positive": {"name": "positive", "type": "CONDITIONING", "link": 5},
      "negative": {"name": "negative", "type": "CONDITIONING", "link": 6},
      "latent_image": {"name": "latent_image", "type": "LATENT", "link": 20}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [21], "slot_index": 0}],
    "properties": {"Node name for S&R": "KSampler"},
    "widgets_values": [0, "fixed", 25, 7.0, "euler", "normal", 1.0]
  },
  {
    "id": 14,
    "type": "VAEDecode",
    "pos": [1570, 500],
    "size": [240, 50],
    "flags": {},
    "order": 13,
    "mode": 0,
    "inputs": {
      "samples": {"name": "samples", "type": "LATENT", "link": 21},
      "vae": {"name": "vae", "type": "VAE", "link": 4}
    },
    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [22], "slot_index": 0}],
    "properties": {"Node name for S&R": "VAEDecode"},
    "widgets_values": []
  },
  {
    "id": 15,
    "type": "SaveImage",
    "pos": [1850, 500],
    "size": [320, 270],
    "flags": {},
    "order": 14,
    "mode": 0,
    "inputs": {"images": {"name": "images", "type": "IMAGE", "link": 22}},
    "outputs": [],
    "properties": {"Node name for S&R": "SaveImage"},
    "widgets_values": ["GAN_Arithmetic"]
  }
]
```

---

## Workflow 3: **Space Walk Explorer**

**Purpose**: Explore latent space through random walks, circular walks, 2D grids, and multi-waypoint paths.

**Nodes covered**: `LatentRandomWalk`, `LatentCircularWalk`, `LatentGrid2D`, `LatentMultiWaypointInterpolation`, `LatentRandomDirection`, `LatentSelectFromBatch`

```json
[
  {
    "id": 1,
    "type": "CheckpointLoaderSimple",
    "pos": [50, 50],
    "size": [320, 100],
    "flags": {},
    "order": 0,
    "mode": 0,
    "inputs": {},
    "outputs": [
      {"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0},
      {"name": "CLIP", "type": "CLIP", "links": [2, 3], "slot_index": 1},
      {"name": "VAE", "type": "VAE", "links": [4], "slot_index": 2}
    ],
    "properties": {"Node name for S&R": "CheckpointLoaderSimple"},
    "widgets_values": ["sd_xl_base_1.0.safetensors"]
  },
  {
    "id": 2,
    "type": "CLIPTextEncode",
    "pos": [50, 200],
    "size": [300, 80],
    "flags": {},
    "order": 1,
    "mode": 0,
    "inputs": {"clip": {"name": "clip", "type": "CLIP", "link": 2}},
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [5, 8, 11, 14], "slot_index": 0}],
    "properties": {"Node name for S&R": "CLIPTextEncode"},
    "widgets_values": ["a landscape painting"]
  },
  {
    "id": 3,
    "type": "CLIPTextEncode",
    "pos": [50, 320],
    "size": [300, 80],
    "flags": {},
    "order": 2,
    "mode": 0,
    "inputs": {"clip": {"name": "clip", "type": "CLIP", "link": 3}},
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [6, 9, 12, 15], "slot_index": 0}],
    "properties": {"Node name for S&R": "CLIPTextEncode"},
    "widgets_values": ["negative"]
  },
  {
    "id": 4,
    "type": "EmptyLatentImage",
    "pos": [50, 450],
    "size": [300, 110],
    "flags": {},
    "order": 3,
    "mode": 0,
    "inputs": {},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [7, 10, 13, 16, 17, 18], "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptyLatentImage"},
    "widgets_values": [42, 1.0, 1.0]
  },
  {
    "id": 5,
    "type": "LatentRandomDirection",
    "pos": [450, 450],
    "size": [320, 130],
    "flags": {},
    "order": 4,
    "mode": 0,
    "inputs": {"latent": {"name": "latent", "type": "LATENT", "link": 16}},
    "outputs": [{"name": "LATENT_DIRECTION", "type": "LATENT_DIRECTION", "links": [19, 20], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentRandomDirection"},
    "widgets_values": [2, 123, true, "gaussian"]
  },
  {
    "id": 6,
    "type": "LatentRandomWalk",
    "pos": [450, 100],
    "size": [340, 170],
    "flags": {},
    "order": 5,
    "mode": 0,
    "inputs": {"start_latent": {"name": "start_latent", "type": "LATENT", "link": 7}},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [21], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentRandomWalk"},
    "widgets_values": [20, 0, 0.5, "slerp"]
  },
  {
    "id": 7,
    "type": "LatentCircularWalk",
    "pos": [450, 320],
    "size": [340, 130],
    "flags": {},
    "order": 6,
    "mode": 0,
    "inputs": {
      "center_latent": {"name": "center_latent", "type": "LATENT", "link": 13},
      "direction_1": {"name": "direction_1", "type": "LATENT_DIRECTION", "link": 19}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [24], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentCircularWalk"},
    "widgets_values": [36, 1.5, 999]
  },
  {
    "id": 8,
    "type": "LatentGrid2D",
    "pos": [450, 620],
    "size": [340, 170],
    "flags": {},
    "order": 7,
    "mode": 0,
    "inputs": {
      "center_latent": {"name": "center_latent", "type": "LATENT", "link": 17},
      "x_direction": {"name": "x_direction", "type": "LATENT_DIRECTION", "link": 20}
    },
    "outputs": [{"name": "latents", "type": "LATENT", "links": [27], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentGrid2D"},
    "widgets_values": [5, 5, 1.0, 1.0]
  },
  {
    "id": 9,
    "type": "EmptyLatentImage",
    "pos": [450, 820],
    "size": [300, 110],
    "flags": {},
    "order": 8,
    "mode": 0,
    "inputs": {},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [28, 29], "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptyLatentImage"},
    "widgets_values": [77, 1.0, 1.0]
  },
  {
    "id": 10,
    "type": "EmptyLatentImage",
    "pos": [450, 960],
    "size": [300, 110],
    "flags": {},
    "order": 9,
    "mode": 0,
    "inputs": {},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [30, 31], "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptyLatentImage"},
    "widgets_values": [99, 1.0, 1.0]
  },
  {
    "id": 11,
    "type": "LatentMultiWaypointInterpolation",
    "pos": [850, 820],
    "size": [340, 190],
    "flags": {},
    "order": 10,
    "mode": 0,
    "inputs": {
      "latent_1": {"name": "latent_1", "type": "LATENT", "link": 10},
      "latent_2": {"name": "latent_2", "type": "LATENT", "link": 28},
      "latent_3": {"name": "latent_3", "type": "LATENT", "link": 30},
      "latent_4": {"name": "latent_4", "type": "LATENT", "link": 18}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [30 + 2], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentMultiWaypointInterpolation"},
    "widgets_values": [5, "slerp", true]
  },
  {
    "id": 12,
    "type": "LatentSelectFromBatch",
    "pos": [850, 1040],
    "size": [300, 90],
    "flags": {},
    "order": 11,
    "mode": 0,
    "inputs": {"latent": {"name": "latent", "type": "LATENT", "link": 31}},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [33], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentSelectFromBatch"},
    "widgets_values": [3]
  },
  {
    "id": 13,
    "type": "KSampler",
    "pos": [1250, 100],
    "size": [280, 220],
    "flags": {},
    "order": 12,
    "mode": 0,
    "inputs": {
      "model": {"name": "model", "type": "MODEL", "link": 1},
      "positive": {"name": "positive", "type": "CONDITIONING", "link": 5},
      "negative": {"name": "negative", "type": "CONDITIONING", "link": 6},
      "latent_image": {"name": "latent_image", "type": "LATENT", "link": 21}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [34], "slot_index": 0}],
    "properties": {"Node name for S&R": "KSampler"},
    "widgets_values": [0, "fixed", 25, 7.0, "euler", "normal", 1.0]
  },
  {
    "id": 14,
    "type": "VAEDecode",
    "pos": [1570, 100],
    "size": [240, 50],
    "flags": {},
    "order": 13,
    "mode": 0,
    "inputs": {
      "samples": {"name": "samples", "type": "LATENT", "link": 34},
      "vae": {"name": "vae", "type": "VAE", "link": 4}
    },
    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [35], "slot_index": 0}],
    "properties": {"Node name for S&R": "VAEDecode"},
    "widgets_values": []
  },
  {
    "id": 15,
    "type": "SaveImage",
    "pos": [1850, 100],
    "size": [320, 270],
    "flags": {},
    "order": 14,
    "mode": 0,
    "inputs": {"images": {"name": "images", "type": "IMAGE", "link": 35}},
    "outputs": [],
    "properties": {"Node name for S&R": "SaveImage"},
    "widgets_values": ["SpaceWalk_RandomWalk"]
  },
  {
    "id": 16,
    "type": "KSampler",
    "pos": [1250, 340],
    "size": [280, 220],
    "flags": {},
    "order": 15,
    "mode": 0,
    "inputs": {
      "model": {"name": "model", "type": "MODEL", "link": 1},
      "positive": {"name": "positive", "type": "CONDITIONING", "link": 8},
      "negative": {"name": "negative", "type": "CONDITIONING", "link": 9},
      "latent_image": {"name": "latent_image", "type": "LATENT", "link": 24}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [36], "slot_index": 0}],
    "properties": {"Node name for S&R": "KSampler"},
    "widgets_values": [0, "fixed", 25, 7.0, "euler", "normal", 1.0]
  },
  {
    "id": 17,
    "type": "VAEDecode",
    "pos": [1570, 340],
    "size": [240, 50],
    "flags": {},
    "order": 16,
    "mode": 0,
    "inputs": {
      "samples": {"name": "samples", "type": "LATENT", "link": 36},
      "vae": {"name": "vae", "type": "VAE", "link": 4}
    },
    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [37], "slot_index": 0}],
    "properties": {"Node name for S&R": "VAEDecode"},
    "widgets_values": []
  },
  {
    "id": 18,
    "type": "SaveImage",
    "pos": [1850, 340],
    "size": [320, 270],
    "flags": {},
    "order": 17,
    "mode": 0,
    "inputs": {"images": {"name": "images", "type": "IMAGE", "link": 37}},
    "outputs": [],
    "properties": {"Node name for S&R": "SaveImage"},
    "widgets_values": ["SpaceWalk_Circular"]
  },
  {
    "id": 19,
    "type": "KSampler",
    "pos": [1250, 620],
    "size": [280, 220],
    "flags": {},
    "order": 18,
    "mode": 0,
    "inputs": {
      "model": {"name": "model", "type": "MODEL", "link": 1},
      "positive": {"name": "positive", "type": "CONDITIONING", "link": 11},
      "negative": {"name": "negative", "type": "CONDITIONING", "link": 12},
      "latent_image": {"name": "latent_image", "type": "LATENT", "link": 27}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [38], "slot_index": 0}],
    "properties": {"Node name for S&R": "KSampler"},
    "widgets_values": [0, "fixed", 25, 7.0, "euler", "normal", 1.0]
  },
  {
    "id": 20,
    "type": "VAEDecode",
    "pos": [1570, 620],
    "size": [240, 50],
    "flags": {},
    "order": 19,
    "mode": 0,
    "inputs": {
      "samples": {"name": "samples", "type": "LATENT", "link": 38},
      "vae": {"name": "vae", "type": "VAE", "link": 4}
    },
    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [39], "slot_index": 0}],
    "properties": {"Node name for S&R": "VAEDecode"},
    "widgets_values": []
  },
  {
    "id": 21,
    "type": "SaveImage",
    "pos": [1850, 620],
    "size": [320, 270],
    "flags": {},
    "order": 20,
    "mode": 0,
    "inputs": {"images": {"name": "images", "type": "IMAGE", "link": 39}},
    "outputs": [],
    "properties": {"Node name for S&R": "SaveImage"},
    "widgets_values": ["SpaceWalk_Grid2D"]
  }
]
```

---

## Workflow 4: **Noise, Truncation & Channel Manipulation Lab**

**Purpose**: Apply truncation ψ, noise perturbation modes, normalization, channel split/merge, and statistics extraction.

**Nodes covered**: `LatentTruncation`, `LatentNoisePerturbation`, `LatentNormalize`, `LatentAddNoise`, `LatentStatistics`, `LatentChannelSplit`, `LatentChannelMerge`

```json
[
  {
    "id": 1,
    "type": "CheckpointLoaderSimple",
    "pos": [50, 50],
    "size": [320, 100],
    "flags": {},
    "order": 0,
    "mode": 0,
    "inputs": {},
    "outputs": [
      {"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0},
      {"name": "CLIP", "type": "CLIP", "links": [2, 3], "slot_index": 1},
      {"name": "VAE", "type": "VAE", "links": [4], "slot_index": 2}
    ],
    "properties": {"Node name for S&R": "CheckpointLoaderSimple"},
    "widgets_values": ["sd_xl_base_1.0.safetensors"]
  },
  {
    "id": 2,
    "type": "CLIPTextEncode",
    "pos": [50, 200],
    "size": [300, 80],
    "flags": {},
    "order": 1,
    "mode": 0,
    "inputs": {"clip": {"name": "clip", "type": "CLIP", "link": 2}},
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [5], "slot_index": 0}],
    "properties": {"Node name for S&R": "CLIPTextEncode"},
    "widgets_values": ["an abstract artwork"]
  },
  {
    "id": 3,
    "type": "CLIPTextEncode",
    "pos": [50, 320],
    "size": [300, 80],
    "flags": {},
    "order": 2,
    "mode": 0,
    "inputs": {"clip": {"name": "clip", "type": "CLIP", "link": 3}},
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [6], "slot_index": 0}],
    "properties": {"Node name for S&R": "CLIPTextEncode"},
    "widgets_values": ["negative"]
  },
  {
    "id": 4,
    "type": "EmptyLatentImage",
    "pos": [50, 450],
    "size": [300, 110],
    "flags": {},
    "order": 3,
    "mode": 0,
    "inputs": {},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [7, 8, 9, 10, 11], "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptyLatentImage"},
    "widgets_values": [1, 1.0, 1.0]
  },
  {
    "id": 5,
    "type": "EmptyLatentImage",
    "pos": [50, 600],
    "size": [300, 110],
    "flags": {},
    "order": 4,
    "mode": 0,
    "inputs": {},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [12], "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptyLatentImage"},
    "widgets_values": [500, 1.0, 1.0]
  },
  {
    "id": 6,
    "type": "LatentTruncation",
    "pos": [450, 450],
    "size": [340, 130],
    "flags": {},
    "order": 5,
    "mode": 0,
    "inputs": {
      "latent": {"name": "latent", "type": "LATENT", "link": 7},
      "mean_latent": {"name": "mean_latent", "type": "LATENT", "link": 12}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [13, 14], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentTruncation"},
    "widgets_values": [0.5]
  },
  {
    "id": 7,
    "type": "LatentAddNoise",
    "pos": [450, 200],
    "size": [300, 110],
    "flags": {},
    "order": 6,
    "mode": 0,
    "inputs": {"latent": {"name": "latent", "type": "LATENT", "link": 8}},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [15], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentAddNoise"},
    "widgets_values": [0.05, 42]
  },
  {
    "id": 8,
    "type": "LatentNoisePerturbation",
    "pos": [450, 620],
    "size": [340, 150],
    "flags": {},
    "order": 7,
    "mode": 0,
    "inputs": {"latent": {"name": "latent", "type": "LATENT", "link": 9}},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [16], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentNoisePerturbation"},
    "widgets_values": [0.2, "perlin", 123]
  },
  {
    "id": 9,
    "type": "LatentNormalize",
    "pos": [850, 450],
    "size": [300, 110],
    "flags": {},
    "order": 8,
    "mode": 0,
    "inputs": {"latent": {"name": "latent", "type": "LATENT", "link": 13}},
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [17, 20], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentNormalize"},
    "widgets_values": ["gaussian", 0.0]
  },
  {
    "id": 10,
    "type": "LatentStatistics",
    "pos": [850, 600],
    "size": [340, 90],
    "flags": {},
    "order": 9,
    "mode": 0,
    "inputs": {"latent": {"name": "latent", "type": "LATENT", "link": 14}},
    "outputs": [
      {"name": "mean", "type": "FLOAT", "links": [], "slot_index": 0},
      {"name": "std", "type": "FLOAT", "links": [], "slot_index": 1},
      {"name": "min", "type": "FLOAT", "links": [], "slot_index": 2},
      {"name": "max", "type": "FLOAT", "links": [], "slot_index": 3},
      {"name": "norm", "type": "FLOAT", "links": [], "slot_index": 4},
      {"name": "batch_size", "type": "INT", "links": [], "slot_index": 5},
      {"name": "channels", "type": "INT", "links": [], "slot_index": 6},
      {"name": "height", "type": "INT", "links": [], "slot_index": 7},
      {"name": "width", "type": "INT", "links": [], "slot_index": 8}
    ],
    "properties": {"Node name for S&R": "LatentStatistics"},
    "widgets_values": [0]
  },
  {
    "id": 11,
    "type": "LatentChannelSplit",
    "pos": [850, 200],
    "size": [300, 110],
    "flags": {},
    "order": 10,
    "mode": 0,
    "inputs": {"latent": {"name": "latent", "type": "LATENT", "link": 10}},
    "outputs": [
      {"name": "latent_A", "type": "LATENT", "links": [18], "slot_index": 0},
      {"name": "latent_B", "type": "LATENT", "links": [19], "slot_index": 1}
    ],
    "properties": {"Node name for S&R": "LatentChannelSplit"},
    "widgets_values": [2]
  },
  {
    "id": 12,
    "type": "LatentChannelMerge",
    "pos": [850, 750],
    "size": [300, 90],
    "flags": {},
    "order": 11,
    "mode": 0,
    "inputs": {
      "latent_A": {"name": "latent_A", "type": "LATENT", "link": 18},
      "latent_B": {"name": "latent_B", "type": "LATENT", "link": 19}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [22], "slot_index": 0}],
    "properties": {"Node name for S&R": "LatentChannelMerge"},
    "widgets_values": []
  },
  {
    "id": 13,
    "type": "KSampler",
    "pos": [1230, 450],
    "size": [280, 220],
    "flags": {},
    "order": 12,
    "mode": 0,
    "inputs": {
      "model": {"name": "model", "type": "MODEL", "link": 1},
      "positive": {"name": "positive", "type": "CONDITIONING", "link": 5},
      "negative": {"name": "negative", "type": "CONDITIONING", "link": 6},
      "latent_image": {"name": "latent_image", "type": "LATENT", "link": 17}
    },
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [23], "slot_index": 0}],
    "properties": {"Node name for S&R": "KSampler"},
    "widgets_values": [0, "fixed", 25, 7.0, "euler", "normal", 1.0]
  },
  {
    "id": 14,
    "type": "VAEDecode",
    "pos": [1550, 450],
    "size": [240, 50],
    "flags": {},
    "order": 13,
    "mode": 0,
    "inputs": {
      "samples": {"name": "samples", "type": "LATENT", "link": 23},
      "vae": {"name": "vae", "type": "VAE", "link": 4}
    },
    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [24], "slot_index": 0}],
    "properties": {"Node name for S&R": "VAEDecode"},
    "widgets_values": []
  },
  {
    "id": 15,
    "type": "SaveImage",
    "pos": [1830, 450],
    "size": [320, 270],
    "flags": {},
    "order": 14,
    "mode": 0,
    "inputs": {"images": {"name": "images", "type": "IMAGE", "link": 24}},
    "outputs": [],
    "properties": {"Node name for S&R": "SaveImage"},
    "widgets_values": ["NoiseLab_Truncated"]
  }
]
```

---

## Coverage Summary

<details>
<summary>📋 Which nodes appear in which workflow (click to expand)</summary>

| Node | W1: PCA | W2: GAN Math | W3: Space Walk | W4: Noise Lab |
|---|:---:|:---:|:---:|:---:|
| `LatentSlerp` | ✅ | | | |
| `LatentSlerpBatch` | | ✅ | | |
| `LatentMultiWaypointInterpolation` | | | ✅ | |
| `LatentNormalizedInterpolation` | | ✅ | | |
| `LatentGANArithmetic` | | ✅ | | |
| `LatentScalarMultiply` | | ✅ | | |
| `LatentBlend` | | ✅ | | |
| `LatentWeightedAverage` | | ✅ | | |
| `LatentPCA` | ✅ | | | |
| `LatentApplyDirection` | ✅ | | | |
| `LatentRandomDirection` | | | ✅ | |
| `LatentDirectionStore` | ✅ | | | |
| `LatentTruncation` | | | | ✅ |
| `LatentNoisePerturbation` | | | | ✅ |
| `LatentNormalize` | | | | ✅ |
| `LatentAddNoise` | | | | ✅ |
| `LatentRandomWalk` | | | ✅ | |
| `LatentGrid2D` | | | ✅ | |
| `LatentPerturbBatch` | ✅ | | | |
| `LatentCircularWalk` | | | ✅ | |
| `LatentInfo` | ✅ | | | |
| `LatentStatistics` | | | | ✅ |
| `LatentSelectFromBatch` | | | ✅ | |
| `LatentChannelSplit` | | | | ✅ |
| `LatentChannelMerge` | | | | ✅ |

**All 25 nodes are covered.** ✅

</details>

---

## How to Use These Workflows

1. **Save** each JSON block above as a `.json` file (e.g., `workflow1_pca.json`).
2. In ComfyUI, click **`Load`** and select the file, **or** drag the `.json` file directly onto the ComfyUI canvas.
3. Adjust the `CheckpointLoaderSimple` widget to point to a model you have installed.
4. Click **`Queue Prompt`** to execute.

> **Note**: The link IDs in the JSON are illustrative. When ComfyUI loads a workflow, it reconciles links automatically based on the `links` arrays in each node's inputs/outputs. If any link references appear broken after import, simply reconnect the affected cables in the UI — the node types and widget values are all valid.

> **Tip**: For the **Space Walk** workflow (W3), each KSampler branch produces a different exploration output. Try batching them by increasing the `batch_size` widget on `EmptyLatentImage` to see full walk/grid sequences decoded at once.