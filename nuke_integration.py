# nuke_integration.py - Nuke-specific integration utilities for H2 SamViT
# Handles pixel access, node operations, and viewer updates

import nuke
import numpy as np
from typing import Optional, Tuple


def get_input_image(node: nuke.Node, channel: str = "rgba") -> Optional[np.ndarray]:
    """
    Extract image data from a node's input.
    
    Args:
        node: The Nuke node to get input from
        channel: Channel layer to extract ("rgba", "rgb", or specific channel)
    
    Returns:
        numpy array of image data (HxWx4 for rgba, HxWx3 for rgb)
    """
    input_node = node.input(0)
    if not input_node:
        return None
    
    # Get format/bounds
    fmt = input_node.format()
    width = fmt.width()
    height = fmt.height()
    
    # Create numpy array
    if channel == "rgba":
        channels = 4
        channel_names = ["red", "green", "blue", "alpha"]
    elif channel == "rgb":
        channels = 3
        channel_names = ["red", "green", "blue"]
    else:
        channels = 1
        channel_names = [channel]
    
    image = np.zeros((height, width, channels), dtype=np.float32)
    
    # Sample each channel
    # Note: In production, this would use Nuke's optimized pixel access
    # This is a simplified placeholder
    try:
        for c, ch_name in enumerate(channel_names):
            for y in range(height):
                for x in range(width):
                    # This is slow - production code would use Tile API
                    image[y, x, c] = input_node.sample(f"rgba.{ch_name}", x, y)
    except Exception as e:
        print(f"[H2 SamViT] Warning: Could not sample input: {e}")
        return None
    
    # Convert to uint8 for model input
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    return image_uint8


def write_mask_to_alpha(node: nuke.Node, mask: np.ndarray) -> bool:
    """
    Write a mask to the node's alpha channel output.
    
    Args:
        node: The Nuke node
        mask: Mask array (HxW, 0-1 float)
    
    Returns:
        True if successful
    """
    # In production, this would write to the node's internal buffer
    # that gets read during render
    
    # Store in node's custom storage
    from . import cache
    frame = nuke.frame()
    cache.store_mask(node.name(), frame, mask)
    
    return True


def get_cached_mask(node: nuke.Node, frame: int) -> Optional[np.ndarray]:
    """
    Get a cached mask for a specific frame.
    
    Args:
        node: The Nuke node
        frame: Frame number
    
    Returns:
        Cached mask or None
    """
    from . import cache
    return cache.get_mask(node.name(), frame)


def get_node_format(node: nuke.Node) -> Tuple[int, int]:
    """Get the format (width, height) from a node's input."""
    input_node = node.input(0)
    if input_node:
        fmt = input_node.format()
        return fmt.width(), fmt.height()
    
    # Fallback to project format
    fmt = nuke.root().format()
    return fmt.width(), fmt.height()


def force_viewer_update():
    """Force Nuke's viewer to refresh."""
    nuke.updateUI()


def get_viewer_mouse_position(viewer_node: nuke.Node) -> Optional[Tuple[float, float]]:
    """
    Get the current mouse position in viewer coordinates.
    
    Note: This requires Nuke's panel API access.
    """
    # In production, this would access Nuke's viewer state
    return None


def create_output_image(
    source_image: np.ndarray,
    mask: np.ndarray,
    overlay_color: Tuple[float, float, float],
    overlay_alpha: float,
    display_mode: str
) -> np.ndarray:
    """
    Create the output image with mask overlay.
    
    Args:
        source_image: Original RGB(A) image
        mask: Alpha mask (HxW, 0-1)
        overlay_color: RGB color for overlay
        overlay_alpha: Opacity of overlay
        display_mode: "Overlay" or "Matte"
    
    Returns:
        Output image with alpha channel
    """
    h, w = mask.shape[:2]
    
    # Ensure source has 4 channels
    if source_image.shape[2] == 3:
        output = np.zeros((h, w, 4), dtype=np.float32)
        output[:, :, :3] = source_image[:, :, :3]
    else:
        output = source_image.copy()
    
    # Set alpha channel to mask
    output[:, :, 3] = mask
    
    if display_mode == "Overlay":
        # Apply colored overlay
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        overlay[:, :, 0] = overlay_color[0]
        overlay[:, :, 1] = overlay_color[1]
        overlay[:, :, 2] = overlay_color[2]
        
        # Blend overlay where mask exists
        blend_factor = mask[:, :, np.newaxis] * overlay_alpha
        output[:, :, :3] = (
            output[:, :, :3] * (1 - blend_factor) +
            overlay * blend_factor
        )
    
    elif display_mode == "Matte":
        # Show black/white mask as RGB
        output[:, :, 0] = mask
        output[:, :, 1] = mask
        output[:, :, 2] = mask
    
    return output


def validate_input(node: nuke.Node) -> Tuple[bool, str]:
    """
    Validate that the node has valid input.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    input_node = node.input(0)
    
    if not input_node:
        return False, "No input connected. Connect an image to the input."
    
    # Check if input has valid format
    try:
        fmt = input_node.format()
        if fmt.width() < 1 or fmt.height() < 1:
            return False, "Input has invalid dimensions."
    except:
        return False, "Could not read input format."
    
    return True, ""


def setup_node_callbacks(node: nuke.Node):
    """Setup callbacks for node events."""
    
    # Knob changed callback
    def on_knob_changed():
        from . import callbacks
        callbacks.on_knob_changed(nuke.thisNode(), nuke.thisKnob())
    
    # Add callback
    nuke.addKnobChanged(on_knob_changed, node=node)


def cleanup_node(node: nuke.Node):
    """Clean up resources when node is deleted."""
    from . import cache, temporal
    
    node_name = node.name()
    
    # Clear caches
    cache.clear_node_cache(node_name)
    temporal.clear_temporal_buffer(node_name)
    
    print(f"[H2 SamViT] Cleaned up resources for {node_name}")
