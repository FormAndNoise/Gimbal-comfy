import unittest
import torch
import sys
import os

# Add parent directory to path to import Wayfinder_crossmodal_bridge
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Wayfinder_crossmodal_bridge import Wayfinder_CrossModalBridge

class TestCrossModalBridge(unittest.TestCase):
    def test_metadata_preservation(self):
        """
        Verifies that the `base_latent` metadata (like noise_mask) is preserved 
        in both `target_vector` and `origin_vector` when translated.
        """
        bridge = Wayfinder_CrossModalBridge()

        # Create a mock base latent with samples and extra metadata
        samples = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        noise_mask = torch.ones((1, 1, 8, 8), dtype=torch.float32)

        base_latent = {
            "samples": samples,
            "noise_mask": noise_mask,
            "extra_field": "some_value"
        }

        # Run translation
        target_vector, origin_vector = bridge.translate(
            llm_instruction="make it brighter",
            base_latent=base_latent,
            mapping_mode="Keyword_Heuristics"
        )

        # Check origin_vector
        self.assertIn("noise_mask", origin_vector)
        self.assertEqual(origin_vector["noise_mask"].shape, noise_mask.shape)
        self.assertIn("extra_field", origin_vector)
        self.assertEqual(origin_vector["extra_field"], "some_value")
        self.assertTrue(torch.equal(origin_vector["samples"], samples))

        # Check target_vector
        self.assertIn("noise_mask", target_vector)
        self.assertEqual(target_vector["noise_mask"].shape, noise_mask.shape)
        self.assertIn("extra_field", target_vector)
        self.assertEqual(target_vector["extra_field"], "some_value")
        
        # Target samples should have changed because of instruction
        self.assertFalse(torch.equal(target_vector["samples"], samples))

if __name__ == '__main__':
    unittest.main()
