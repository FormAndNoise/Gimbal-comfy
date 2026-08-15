"""
[Form & Noise Atelier — Gimbal Node Suite]
Precision Latent Flight Instruments for ComfyUI.
"""

import os
from .nodes.Wayfinder_compass import WayfinderCompass_Pro
from .nodes.wayfindermanifold_explorer import WayfinderManifold_Explorer
from .nodes.wayfinder_gps_anchor import WayfinderGPS_Anchor
from .nodes.Wayfinder_crossmodal_bridge import Wayfinder_CrossModalBridge
from .nodes.wayfinder_semanticslider import Wayfinder_SemanticSlider
from .nodes.wayfinder_likeness_isolator import LikenessVectorIsolator
from .nodes.Grid_Master import LatentSpaceGridStitch
from .nodes.wayfinder_gps_load import WayfinderGPS_Load

from .nodes.gimbal_circular_orbit import GimbalCircularOrbit
from .nodes.gimbal_waypoint_spline import GimbalWaypointSpline
from .nodes.gimbal_channel_matrix import GimbalChannelSplit, GimbalChannelMerge, GimbalChannelScale
from .nodes.gimbal_truncation import GimbalTruncation
from .nodes.gimbal_vector_analogy import GimbalVectorAnalogy
from .nodes.gimbal_diagnostics import GimbalDiagnostics

NODE_CLASS_MAPPINGS = {
    # Core Flight Instruments
    "WayfinderCompass_Pro": WayfinderCompass_Pro,
    "GimbalCompass_Pro": WayfinderCompass_Pro,
    
    "WayfinderManifold_Explorer": WayfinderManifold_Explorer,
    "GimbalManifold_Explorer": WayfinderManifold_Explorer,
    
    "WayfinderGPS_Anchor": WayfinderGPS_Anchor,
    "GimbalGPS_Anchor": WayfinderGPS_Anchor,
    
    "WayfinderGPS_Load": WayfinderGPS_Load,
    "GimbalGPS_Load": WayfinderGPS_Load,
    
    "Wayfinder_CrossModalBridge": Wayfinder_CrossModalBridge,
    "Gimbal_CrossModalBridge": Wayfinder_CrossModalBridge,
    
    "Wayfinder_SemanticSlider": Wayfinder_SemanticSlider,
    "Gimbal_SemanticSlider": Wayfinder_SemanticSlider,
    
    "LikenessVectorIsolator": LikenessVectorIsolator,
    "LatentSpaceGridStitch": LatentSpaceGridStitch,

    # Mathematical & Trajectory Primitives
    "GimbalCircularOrbit": GimbalCircularOrbit,
    "GimbalWaypointSpline": GimbalWaypointSpline,
    "GimbalChannelSplit": GimbalChannelSplit,
    "GimbalChannelMerge": GimbalChannelMerge,
    "GimbalChannelScale": GimbalChannelScale,
    "GimbalTruncation": GimbalTruncation,
    "GimbalVectorAnalogy": GimbalVectorAnalogy,
    "GimbalDiagnostics": GimbalDiagnostics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WayfinderCompass_Pro": "🧭 Gimbal Compass Pro",
    "GimbalCompass_Pro": "🧭 Gimbal Compass Pro",
    
    "WayfinderManifold_Explorer": "🗺️ Gimbal Manifold Explorer",
    "GimbalManifold_Explorer": "🗺️ Gimbal Manifold Explorer",
    
    "WayfinderGPS_Anchor": "📍 Gimbal GPS Anchor (Save)",
    "GimbalGPS_Anchor": "📍 Gimbal GPS Anchor (Save)",
    
    "WayfinderGPS_Load": "📥 Gimbal GPS Load (Recall)",
    "GimbalGPS_Load": "📥 Gimbal GPS Load (Recall)",
    
    "Wayfinder_CrossModalBridge": "🌉 Gimbal Cross-Modal Bridge (Text-to-Latent)",
    "Gimbal_CrossModalBridge": "🌉 Gimbal Cross-Modal Bridge (Text-to-Latent)",
    
    "Wayfinder_SemanticSlider": "🎚️ Gimbal Semantic Slider (PCA)",
    "Gimbal_SemanticSlider": "🎚️ Gimbal Semantic Slider (PCA)",
    
    "LikenessVectorIsolator": "🧬 Gimbal Likeness Isolator",
    "LatentSpaceGridStitch": "🗺️ Gimbal Grid Stitch",

    "GimbalCircularOrbit": "🔄 Gimbal Circular Orbit",
    "GimbalWaypointSpline": "🛤️ Gimbal Waypoint Spline",
    "GimbalChannelSplit": "🔀 Gimbal Channel Split",
    "GimbalChannelMerge": "🔁 Gimbal Channel Merge",
    "GimbalChannelScale": "🎛️ Gimbal Channel Band Scaler",
    "GimbalTruncation": "🎯 Gimbal Latent Truncation",
    "GimbalVectorAnalogy": "⚖️ Gimbal Vector Analogy (GAN Math)",
    "GimbalDiagnostics": "📊 Gimbal Latent Diagnostics",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]