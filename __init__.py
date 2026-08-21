"""
[Form & Noise Atelier — Gimbal Node Suite]
Precision Latent Flight Instruments for ComfyUI.
"""

import os
try:
    from .nodes.gimbal_compass import GimbalCompass_Pro, WayfinderCompass_Pro
    from .nodes.gimbal_manifold_explorer import GimbalManifold_Explorer, WayfinderManifold_Explorer
    from .nodes.gimbal_gps_anchor import GimbalGPS_Anchor, WayfinderGPS_Anchor
    from .nodes.gimbal_gps_load import GimbalGPS_Load, WayfinderGPS_Load
    from .nodes.gimbal_crossmodal_bridge import GimbalCrossModalBridge, Wayfinder_CrossModalBridge, Gimbal_CrossModalBridge
    from .nodes.gimbal_semanticslider import GimbalSemanticSlider, Wayfinder_SemanticSlider, Gimbal_SemanticSlider
    from .nodes.gimbal_likeness_isolator import GimbalLikenessIsolator, LikenessVectorIsolator
    from .nodes.gimbal_grid_stitch import GimbalGridStitch, LatentSpaceGridStitch

    from .nodes.gimbal_circular_orbit import GimbalCircularOrbit
    from .nodes.gimbal_waypoint_spline import GimbalWaypointSpline
    from .nodes.gimbal_channel_matrix import GimbalChannelSplit, GimbalChannelMerge, GimbalChannelScale
    from .nodes.gimbal_truncation import GimbalTruncation
    from .nodes.gimbal_vector_analogy import GimbalVectorAnalogy
    from .nodes.gimbal_diagnostics import GimbalDiagnostics
    from .nodes.gimbal_latent_stabilizer import GimbalLatentStabilizer
    from .nodes.gimbal_latent_math_node import GimbalLatentMath
    from .nodes.gimbal_latent_telemetry import GimbalLatentTelemetry
except (ImportError, ValueError):
    from nodes.gimbal_compass import GimbalCompass_Pro, WayfinderCompass_Pro
    from nodes.gimbal_manifold_explorer import GimbalManifold_Explorer, WayfinderManifold_Explorer
    from nodes.gimbal_gps_anchor import GimbalGPS_Anchor, WayfinderGPS_Anchor
    from nodes.gimbal_gps_load import GimbalGPS_Load, WayfinderGPS_Load
    from nodes.gimbal_crossmodal_bridge import GimbalCrossModalBridge, Wayfinder_CrossModalBridge, Gimbal_CrossModalBridge
    from nodes.gimbal_semanticslider import GimbalSemanticSlider, Wayfinder_SemanticSlider, Gimbal_SemanticSlider
    from nodes.gimbal_likeness_isolator import GimbalLikenessIsolator, LikenessVectorIsolator
    from nodes.gimbal_grid_stitch import GimbalGridStitch, LatentSpaceGridStitch

    from nodes.gimbal_circular_orbit import GimbalCircularOrbit
    from nodes.gimbal_waypoint_spline import GimbalWaypointSpline
    from nodes.gimbal_channel_matrix import GimbalChannelSplit, GimbalChannelMerge, GimbalChannelScale
    from nodes.gimbal_truncation import GimbalTruncation
    from nodes.gimbal_vector_analogy import GimbalVectorAnalogy
    from nodes.gimbal_diagnostics import GimbalDiagnostics
    from nodes.gimbal_latent_stabilizer import GimbalLatentStabilizer
    from nodes.gimbal_latent_math_node import GimbalLatentMath
    from nodes.gimbal_latent_telemetry import GimbalLatentTelemetry

NODE_CLASS_MAPPINGS = {
    # Core Flight Instruments
    "GimbalCompass_Pro": GimbalCompass_Pro,
    "WayfinderCompass_Pro": WayfinderCompass_Pro,
    
    "GimbalManifold_Explorer": GimbalManifold_Explorer,
    "WayfinderManifold_Explorer": WayfinderManifold_Explorer,
    
    "GimbalGPS_Anchor": GimbalGPS_Anchor,
    "WayfinderGPS_Anchor": WayfinderGPS_Anchor,
    
    "GimbalGPS_Load": GimbalGPS_Load,
    "WayfinderGPS_Load": WayfinderGPS_Load,
    
    "GimbalCrossModalBridge": GimbalCrossModalBridge,
    "Gimbal_CrossModalBridge": GimbalCrossModalBridge,
    "Wayfinder_CrossModalBridge": Wayfinder_CrossModalBridge,
    
    "GimbalSemanticSlider": GimbalSemanticSlider,
    "Gimbal_SemanticSlider": GimbalSemanticSlider,
    "Wayfinder_SemanticSlider": Wayfinder_SemanticSlider,
    
    "GimbalLikenessIsolator": GimbalLikenessIsolator,
    "LikenessVectorIsolator": LikenessVectorIsolator,

    "GimbalGridStitch": GimbalGridStitch,
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

    # LAMNr / Disentanglement research math
    "GimbalLatentStabilizer": GimbalLatentStabilizer,
    "GimbalLatentMath": GimbalLatentMath,
    "GimbalLatentTelemetry": GimbalLatentTelemetry,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GimbalCompass_Pro": "🧭 Gimbal Compass Pro",
    "WayfinderCompass_Pro": "🧭 Gimbal Compass Pro",
    
    "GimbalManifold_Explorer": "🗺️ Gimbal Manifold Explorer",
    "WayfinderManifold_Explorer": "🗺️ Gimbal Manifold Explorer",
    
    "GimbalGPS_Anchor": "📍 Gimbal GPS Anchor (Save)",
    "WayfinderGPS_Anchor": "📍 Gimbal GPS Anchor (Save)",
    
    "GimbalGPS_Load": "📥 Gimbal GPS Load (Recall)",
    "WayfinderGPS_Load": "📥 Gimbal GPS Load (Recall)",
    
    "GimbalCrossModalBridge": "🌉 Gimbal Cross-Modal Bridge (Text-to-Latent)",
    "Gimbal_CrossModalBridge": "🌉 Gimbal Cross-Modal Bridge (Text-to-Latent)",
    "Wayfinder_CrossModalBridge": "🌉 Gimbal Cross-Modal Bridge (Text-to-Latent)",
    
    "GimbalSemanticSlider": "🎚️ Gimbal Semantic Slider (PCA)",
    "Gimbal_SemanticSlider": "🎚️ Gimbal Semantic Slider (PCA)",
    "Wayfinder_SemanticSlider": "🎚️ Gimbal Semantic Slider (PCA)",
    
    "GimbalLikenessIsolator": "🧬 Gimbal Likeness Isolator",
    "LikenessVectorIsolator": "🧬 Gimbal Likeness Isolator",

    "GimbalGridStitch": "🗺️ Gimbal Grid Stitch",
    "LatentSpaceGridStitch": "🗺️ Gimbal Grid Stitch",

    "GimbalCircularOrbit": "🔄 Gimbal Circular Orbit",
    "GimbalWaypointSpline": "🛤️ Gimbal Waypoint Spline",
    "GimbalChannelSplit": "🔀 Gimbal Channel Split",
    "GimbalChannelMerge": "🔁 Gimbal Channel Merge",
    "GimbalChannelScale": "🎛️ Gimbal Channel Band Scaler",
    "GimbalTruncation": "🎯 Gimbal Latent Truncation",
    "GimbalVectorAnalogy": "⚖️ Gimbal Vector Analogy (GAN Math)",
    "GimbalDiagnostics": "📊 Gimbal Latent Diagnostics",

    "GimbalLatentStabilizer": "🛠️ Gimbal Latent Stabilizer (LAMNr)",
    "GimbalLatentMath": "🔣 Gimbal Latent Math (Dispatcher)",
    "GimbalLatentTelemetry": "📟 Gimbal Latent Telemetry (LAMNr OOD)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]