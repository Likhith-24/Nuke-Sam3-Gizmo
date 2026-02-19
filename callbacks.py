# callbacks.py - Nuke callback handlers for H2 SamViT gizmo
# Handles interactive point/bbox editing, inference triggers, and UI updates

import nuke
import math
from typing import List, Tuple, Optional, Dict, Any

# Maximum number of points supported
MAX_POINTS = 32


def get_enabled_points(node: nuke.Node) -> List[Dict[str, Any]]:
    """Get all enabled points with their properties."""
    points = []
    for i in range(1, MAX_POINTS + 1):
        idx = f"{i:02d}"
        enabled_knob = node.knob(f"point_{idx}_enabled")
        if enabled_knob and enabled_knob.value():
            pos_knob = node.knob(f"point_{idx}")
            fg_knob = node.knob(f"point_{idx}_fg")
            if pos_knob:
                x, y = pos_knob.value()
                is_fg = fg_knob.value() if fg_knob else True
                points.append({
                    "index": i,
                    "x": x,
                    "y": y,
                    "is_foreground": is_fg,
                    "label": 1 if is_fg else 0
                })
    return points


def get_next_available_point_index(node: nuke.Node) -> Optional[int]:
    """Find the next available (disabled) point slot."""
    for i in range(1, MAX_POINTS + 1):
        idx = f"{i:02d}"
        enabled_knob = node.knob(f"point_{idx}_enabled")
        if enabled_knob and not enabled_knob.value():
            return i
    return None


def add_point(node: nuke.Node, x: float, y: float, is_foreground: bool = True) -> bool:
    """Add a new point at the specified coordinates."""
    point_idx = get_next_available_point_index(node)
    if point_idx is None:
        nuke.message("Maximum number of points (32) reached.")
        return False
    
    idx = f"{point_idx:02d}"
    pos_knob = node.knob(f"point_{idx}")
    fg_knob = node.knob(f"point_{idx}_fg")
    enabled_knob = node.knob(f"point_{idx}_enabled")
    
    if pos_knob and fg_knob and enabled_knob:
        pos_knob.setValue([x, y])
        fg_knob.setValue(is_foreground)
        enabled_knob.setValue(True)
        return True
    return False


def delete_point(node: nuke.Node, point_index: int) -> bool:
    """Delete a point by index."""
    if point_index < 1 or point_index > MAX_POINTS:
        return False
    
    idx = f"{point_index:02d}"
    pos_knob = node.knob(f"point_{idx}")
    fg_knob = node.knob(f"point_{idx}_fg")
    enabled_knob = node.knob(f"point_{idx}_enabled")
    
    if pos_knob and fg_knob and enabled_knob:
        pos_knob.setValue([0, 0])
        fg_knob.setValue(True)
        enabled_knob.setValue(False)
        return True
    return False


def clear_all_points(node: nuke.Node) -> None:
    """Clear all points. Bounding box is not affected."""
    for i in range(1, MAX_POINTS + 1):
        delete_point(node, i)
    print("[H2 SamViT] All points cleared.")


def clear_box(node: nuke.Node) -> None:
    """Clear the bounding box to default. Points are not affected."""
    tl_knob = node.knob("bbox_top_left")
    br_knob = node.knob("bbox_bottom_right")
    
    if tl_knob:
        tl_knob.setValue([0, 0])
    if br_knob:
        br_knob.setValue([0, 0])
    
    print("[H2 SamViT] Bounding box cleared.")


def get_bbox(node: nuke.Node) -> Optional[Tuple[float, float, float, float]]:
    """Get bounding box as (x1, y1, x2, y2) or None if disabled/empty."""
    enabled_knob = node.knob("bbox_enabled")
    if not enabled_knob or not enabled_knob.value():
        return None
    
    tl_knob = node.knob("bbox_top_left")
    br_knob = node.knob("bbox_bottom_right")
    
    if not tl_knob or not br_knob:
        return None
    
    x1, y1 = tl_knob.value()
    x2, y2 = br_knob.value()
    
    # Check if bbox is empty
    if x1 == x2 == y1 == y2 == 0:
        return None
    
    # Normalize coordinates (ensure x1 < x2, y1 < y2)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    return (x1, y1, x2, y2)


def set_bbox(node: nuke.Node, x1: float, y1: float, x2: float, y2: float) -> None:
    """Set the bounding box coordinates."""
    tl_knob = node.knob("bbox_top_left")
    br_knob = node.knob("bbox_bottom_right")
    
    if tl_knob and br_knob:
        tl_knob.setValue([min(x1, x2), min(y1, y2)])
        br_knob.setValue([max(x1, x2), max(y1, y2)])


def find_closest_point(node: nuke.Node, x: float, y: float, threshold: float = 20.0) -> Optional[int]:
    """Find the closest enabled point to the given coordinates."""
    points = get_enabled_points(node)
    closest_idx = None
    closest_dist = float('inf')
    
    for p in points:
        dist = math.sqrt((p["x"] - x) ** 2 + (p["y"] - y) ** 2)
        if dist < closest_dist and dist < threshold:
            closest_dist = dist
            closest_idx = p["index"]
    
    return closest_idx


def move_point(node: nuke.Node, point_index: int, x: float, y: float) -> bool:
    """Move a point to new coordinates."""
    if point_index < 1 or point_index > MAX_POINTS:
        return False
    
    idx = f"{point_index:02d}"
    pos_knob = node.knob(f"point_{idx}")
    enabled_knob = node.knob(f"point_{idx}_enabled")
    
    if pos_knob and enabled_knob and enabled_knob.value():
        pos_knob.setValue([x, y])
        return True
    return False


def reset_all_parameters(node: nuke.Node) -> None:
    """Reset all parameters to their default values."""
    # Clear all points
    clear_all_points(node)
    
    # Clear bounding box
    clear_box(node)
    
    # Reset tool states
    defaults = {
        "enable_edit": True,
        "draw_box": True,
        "add_fg_point": False,
        "add_bg_point": False,
        "bbox_enabled": 0,
        "input_threshold": 90,
        "fill_holes": True,
        "fill_holes_area": 16,
        "crop_padding": 20,
        "display_mode": 1,  # Overlay
        "show_mask_overlay": True,
        "overlay_color": [0.5, 0, 0, 0.5],
        "mask_shrink_grow": 0,
        "edge_feather": 0,
        "offset_mask_x": 0,
        "offset_mask_y": 0,
        "refiner_trimap_width": 10,
        "trimap_erode_radius": 2,
        "trimap_dilate_radius": 2,
        "final_binary_sharp": False,
        "enable_temporal_consistency": True,
        "temporal_weight": 50,
        "suppression_threshold": 30,
        "enable_temporal": False,
        "temporal_smoothing": 50,
        "smooth_edges_only": False,
        "edge_width": 8,
        "show_ui_overlays": True,
        "show_point_labels": True,
        "ui_color": [0, 0.652, 1, 0.8],
        "overlay_scale": 1.0,
        "cache_memory_percent": 25,
        "pipeline_mode": 0,  # Point / Bbox
        "text_prompt": "",
    }
    
    for knob_name, default_value in defaults.items():
        knob = node.knob(knob_name)
        if knob:
            try:
                knob.setValue(default_value)
            except:
                pass
    
    print("[H2 SamViT] All parameters reset to defaults.")


def on_knob_changed(node: nuke.Node, knob: nuke.Knob) -> None:
    """Handle knob changes for interactive editing."""
    knob_name = knob.name()
    
    # Handle tool mutual exclusivity
    tool_knobs = ["draw_box", "add_fg_point", "add_bg_point"]
    if knob_name in tool_knobs and knob.value():
        for tool in tool_knobs:
            if tool != knob_name:
                other_knob = node.knob(tool)
                if other_knob:
                    other_knob.setValue(False)
    
    # Handle pipeline mode changes
    if knob_name == "pipeline_mode":
        text_prompt_knob = node.knob("text_prompt")
        if text_prompt_knob:
            # Show/hide text prompt based on mode
            if knob.value() == 1:  # Text Prompt mode
                text_prompt_knob.setVisible(True)
            else:
                text_prompt_knob.setVisible(False)
    
    # Update cache limit display
    if knob_name == "cache_memory_percent":
        update_cache_limit_display(node)


def update_cache_limit_display(node: nuke.Node) -> None:
    """Update the cache limit display based on memory percentage."""
    cache_pct = node.knob("cache_memory_percent").value()
    
    # Estimate available memory (rough calculation)
    import psutil
    try:
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        cache_gb = available_gb * (cache_pct / 100)
        
        # Estimate frames based on 4K resolution (assume ~9MB per frame)
        frame_size_mb = 9  # 3840 * 2160 * 4 bytes / 1024^2
        estimated_frames = int(cache_gb * 1024 / frame_size_mb)
        
        cache_limit_knob = node.knob("cache_limit")
        if cache_limit_knob:
            cache_limit_knob.setValue(f"{cache_gb:.1f} GB (~{estimated_frames} frames @ 3840x2160)")
    except:
        pass


def run_inference(node: nuke.Node) -> None:
    """Trigger SAM3 + ViTMatte inference."""
    from . import inference
    
    # Check if masking is enabled
    if not node.knob("enable_edit").value():
        nuke.message("Enable Edit is disabled. Enable it to run inference.")
        return
    
    # Get pipeline mode
    pipeline_mode = node.knob("pipeline_mode").value()
    
    if pipeline_mode == 0:  # Point / Bbox mode
        points = get_enabled_points(node)
        bbox = get_bbox(node)
        
        if not points and not bbox:
            nuke.message("No points or bounding box defined. Add prompts before running inference.")
            return
        
        inference.run_point_bbox_inference(node, points, bbox)
    
    else:  # Text Prompt mode
        text_prompt = node.knob("text_prompt").value()
        if not text_prompt.strip():
            nuke.message("Text prompt is empty. Enter a description before running inference.")
            return
        
        points = get_enabled_points(node)  # Optional point for instance selection
        inference.run_text_prompt_inference(node, text_prompt, points)


def get_inference_params(node: nuke.Node) -> Dict[str, Any]:
    """Get all relevant parameters for inference."""
    params = {
        # Pre-processing
        "input_threshold": node.knob("input_threshold").value(),
        "fill_holes": node.knob("fill_holes").value(),
        "fill_holes_area": node.knob("fill_holes_area").value(),
        
        # Trimap / Refiner
        "refiner_trimap_width": node.knob("refiner_trimap_width").value(),
        "trimap_erode_radius": node.knob("trimap_erode_radius").value(),
        "trimap_dilate_radius": node.knob("trimap_dilate_radius").value(),
        "final_binary_sharp": node.knob("final_binary_sharp").value(),
        
        # Temporal
        "enable_temporal": node.knob("enable_temporal").value(),
        "temporal_smoothing": node.knob("temporal_smoothing").value(),
        "smooth_edges_only": node.knob("smooth_edges_only").value(),
        "edge_width": node.knob("edge_width").value(),
        "enable_temporal_consistency": node.knob("enable_temporal_consistency").value(),
        "temporal_weight": node.knob("temporal_weight").value(),
        "suppression_threshold": node.knob("suppression_threshold").value(),
        
        # Output
        "display_mode": node.knob("display_mode").value(),
        "mask_shrink_grow": node.knob("mask_shrink_grow").value(),
        "edge_feather": node.knob("edge_feather").value(),
        "offset_mask_x": node.knob("offset_mask_x").value(),
        "offset_mask_y": node.knob("offset_mask_y").value(),
        
        # Performance
        "crop_padding": node.knob("crop_padding").value(),
        "cache_memory_percent": node.knob("cache_memory_percent").value(),
    }
    return params


# Register knob callback for the gizmo
def register_callbacks():
    """Register global knob changed callback."""
    nuke.addKnobChanged(
        lambda: on_knob_changed(nuke.thisNode(), nuke.thisKnob()),
        nodeClass="H2_SamViT"
    )


# Auto-register on import
try:
    register_callbacks()
except:
    pass
