import sys
import unittest.mock as mock

# Mock ComfyUI specific modules so that tests can run outside the ComfyUI environment
sys.modules['folder_paths'] = mock.MagicMock()
sys.modules['comfy'] = mock.MagicMock()
sys.modules['comfy.utils'] = mock.MagicMock()
sys.modules['comfy.sd'] = mock.MagicMock()
