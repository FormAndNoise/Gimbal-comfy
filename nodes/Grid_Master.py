import torch
import numpy as np
import math
from PIL import Image

class LatentSpaceGridStitch:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",), # This accepts a batch of images
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch_grid"
    CATEGORY = "LatentSpaceExplorer"

    def stitch_grid(self, images):
        # images is a torch tensor of shape [batch, height, width, channels]
        n = images.shape[0]
        grid_size = math.ceil(math.sqrt(n))
        
        # Convert torch tensor to list of PIL images for easy stitching
        pil_images = []
        for i in range(n):
            img = 255. * images[i].cpu().numpy()
            pil_images.append(Image.fromarray(np.uint8(img)))

        max_w = pil_images[0].width
        max_h = pil_images[0].height

        # Create canvas
        new_img = Image.new('RGB', (max_w * grid_size, max_h * grid_size), (0, 0, 0))

        for index, img in enumerate(pil_images):
            x = (index % grid_size) * max_w
            y = (index // grid_size) * max_h
            new_img.paste(img, (x, y))

        # Convert back to torch tensor for ComfyUI
        out_image = np.array(new_img).astype(np.float32) / 255.0
        out_image = torch.from_numpy(out_image).unsqueeze(0)

        return (out_image,)