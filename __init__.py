# H2_SamViT_Gizmo - SAM3 + ViTMatte Gizmo for Nuke
# Copyright (C) 2026
# 
# A comprehensive AI-powered segmentation and matting gizmo combining
# SAM3 (Segment Anything Model 3) with ViTMatte for high-quality alpha mattes.

__version__ = "1.0.0"
__author__ = "H2"

import nuke
import os

# Get the directory where this package is installed
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
GIZMO_DIR = os.path.join(PACKAGE_DIR, "gizmos")
ICONS_DIR = os.path.join(PACKAGE_DIR, "icons")

def register():
    """Register the gizmo with Nuke."""
    # Add gizmo path
    nuke.pluginAddPath(GIZMO_DIR)
    
    # Add icons path
    if os.path.exists(ICONS_DIR):
        nuke.pluginAddPath(ICONS_DIR)
    
    # Import callbacks to register knob changed handlers
    from . import callbacks
    
    print(f"[H2_SamViT_Gizmo] Registered v{__version__}")

def unregister():
    """Unregister the gizmo from Nuke."""
    print("[H2_SamViT_Gizmo] Unregistered")
