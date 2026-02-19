# menu.py - Nuke menu registration for H2_SamViT_Gizmo
# Add this to your menu.py or init.py to register the gizmo

import nuke
import os

def add_h2_samvit_menu():
    """Add H2 SamViT Gizmo to Nuke's menu."""
    toolbar = nuke.menu("Nodes")
    
    # Create H2 submenu if it doesn't exist
    h2_menu = toolbar.findItem("H2")
    if not h2_menu:
        h2_menu = toolbar.addMenu("H2", icon="H2_icon.png")
    
    # Add the gizmo
    h2_menu.addCommand(
        "H2 SamViT",
        "nuke.createNode('H2_SamViT')",
        icon="H2_SamViT_icon.png"
    )
    
    # Also add to Image menu for convenience
    image_menu = toolbar.findItem("Image")
    if image_menu:
        image_menu.addCommand(
            "H2 SamViT",
            "nuke.createNode('H2_SamViT')"
        )

# Auto-register on import
try:
    add_h2_samvit_menu()
except Exception as e:
    print(f"[H2_SamViT_Gizmo] Menu registration failed: {e}")
