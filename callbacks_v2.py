# callbacks_v2.py - Nuke callback handlers for H2 SamViT v2
# Uses native Tracker4 and CornerPin2D nodes for viewer handles.
#
# Architecture:
# - FG Points: Internal Tracker4 nodes named FG_Point_01..FG_Point_16
# - BG Points: Internal Tracker4 nodes named BG_Point_01..BG_Point_16
# - Pos Bbox: Internal CornerPin2D named POS_BBOX (axis-aligned rectangle)
# - Neg Bbox: Internal CornerPin2D named NEG_BBOX (SAM3 exclude region)
#
# Python reads positions from these nodes for inference.
# Users interact with native Nuke handles in the Viewer.

import nuke
import math
import os
from typing import List, Tuple, Optional, Dict, Any

# Maximum number of FG/BG points each
MAX_FG_POINTS = 16
MAX_BG_POINTS = 16

# Re-entrancy guard
_in_knob_changed = False


# ─────────────────────────────────────────────────────────────────────
#  Internal node access
# ─────────────────────────────────────────────────────────────────────

def _get_internal_node(gizmo, internal_name: str):
    """
    Get an internal node from inside the gizmo group.
    
    Args:
        gizmo: The H2_SamViT gizmo node
        internal_name: Name of the internal node (e.g., "FG_01")
    
    Returns:
        The internal node or None if not found
    """
    try:
        # Enter the group context
        gizmo.begin()
        node = nuke.toNode(internal_name)
        gizmo.end()
        return node
    except Exception as e:
        print(f"[H2 SamViT] Error accessing internal node {internal_name}: {e}")
        try:
            gizmo.end()
        except:
            pass
        return None


def _get_transform_center(transform_node) -> Optional[Tuple[float, float]]:
    """
    Get the center position from a Transform node.
    
    Transform's "center" knob draws a visible crosshair in the Viewer.
    
    Returns:
        (x, y) tuple or None if not available
    """
    if transform_node is None:
        return None
    
    try:
        center_knob = transform_node.knob('center')
        if center_knob:
            x, y = center_knob.value()
            return (x, y)
        return None
    except Exception as e:
        print(f"[H2 SamViT] Error reading transform center: {e}")
        return None


def _set_transform_center(transform_node, x: float, y: float):
    """Set the center position of a Transform node."""
    if transform_node is None:
        return False
    
    try:
        center_knob = transform_node.knob('center')
        if center_knob:
            center_knob.setValue([x, y])
            return True
        return False
    except Exception as e:
        print(f"[H2 SamViT] Error setting transform center: {e}")
        return False


def _get_cornerpin_bbox(cornerpin_node) -> Optional[Tuple[float, float, float, float]]:
    """
    Get bounding box from a CornerPin2D node as (x_min, y_min, x_max, y_max).
    
    CornerPin has 4 corners: to1 (bottom-left), to2 (bottom-right), 
    to3 (top-right), to4 (top-left). We derive an axis-aligned bbox.
    """
    if cornerpin_node is None:
        return None
    
    try:
        to1 = cornerpin_node.knob('to1').value()  # bottom-left
        to2 = cornerpin_node.knob('to2').value()  # bottom-right
        to3 = cornerpin_node.knob('to3').value()  # top-right
        to4 = cornerpin_node.knob('to4').value()  # top-left
        
        # Extract axis-aligned bounding box
        all_x = [to1[0], to2[0], to3[0], to4[0]]
        all_y = [to1[1], to2[1], to3[1], to4[1]]
        
        x_min = min(all_x)
        x_max = max(all_x)
        y_min = min(all_y)
        y_max = max(all_y)
        
        # Check if bbox is valid (non-zero area)
        if x_max - x_min < 1 or y_max - y_min < 1:
            return None
        
        return (x_min, y_min, x_max, y_max)
    except Exception as e:
        print(f"[H2 SamViT] Error reading CornerPin bbox: {e}")
        return None


def _set_cornerpin_bbox(cornerpin_node, x1: float, y1: float, x2: float, y2: float):
    """
    Set axis-aligned bounding box on a CornerPin2D node.
    
    Args:
        x1, y1: Bottom-left corner
        x2, y2: Top-right corner
    """
    if cornerpin_node is None:
        return False
    
    try:
        # Ensure proper ordering
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Set all four corners for axis-aligned rectangle
        # to1 = bottom-left, to2 = bottom-right, to3 = top-right, to4 = top-left
        cornerpin_node.knob('to1').setValue([x_min, y_min])
        cornerpin_node.knob('to2').setValue([x_max, y_min])
        cornerpin_node.knob('to3').setValue([x_max, y_max])
        cornerpin_node.knob('to4').setValue([x_min, y_max])
        
        # Also set 'from' to match (identity transform)
        cornerpin_node.knob('from1').setValue([x_min, y_min])
        cornerpin_node.knob('from2').setValue([x_max, y_min])
        cornerpin_node.knob('from3').setValue([x_max, y_max])
        cornerpin_node.knob('from4').setValue([x_min, y_max])
        
        return True
    except Exception as e:
        print(f"[H2 SamViT] Error setting CornerPin bbox: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
#  Point management
# ─────────────────────────────────────────────────────────────────────

def get_fg_points(node) -> List[Dict[str, Any]]:
    """
    Get all enabled foreground points from internal Transform nodes.
    
    Returns:
        List of dicts with keys: index, x, y, is_foreground=True, label=1
    """
    points = []
    for i in range(1, MAX_FG_POINTS + 1):
        idx = f"{i:02d}"
        enabled_knob = node.knob(f"fg_{idx}_enabled")
        
        if enabled_knob and enabled_knob.value():
            transform = _get_internal_node(node, f"FG_{idx}")
            pos = _get_transform_center(transform)
            
            if pos is not None:
                points.append({
                    "index": i,
                    "x": pos[0],
                    "y": pos[1],
                    "is_foreground": True,
                    "label": 1
                })
    
    return points


def get_bg_points(node) -> List[Dict[str, Any]]:
    """
    Get all enabled background points from internal Transform nodes.
    
    Returns:
        List of dicts with keys: index, x, y, is_foreground=False, label=0
    """
    points = []
    for i in range(1, MAX_BG_POINTS + 1):
        idx = f"{i:02d}"
        enabled_knob = node.knob(f"bg_{idx}_enabled")
        
        if enabled_knob and enabled_knob.value():
            transform = _get_internal_node(node, f"BG_{idx}")
            pos = _get_transform_center(transform)
            
            if pos is not None:
                points.append({
                    "index": i,
                    "x": pos[0],
                    "y": pos[1],
                    "is_foreground": False,
                    "label": 0
                })
    
    return points


def get_all_points(node) -> List[Dict[str, Any]]:
    """Get all enabled points (FG + BG combined)."""
    return get_fg_points(node) + get_bg_points(node)


def add_fg_point(node) -> bool:
    """
    Enable the next available FG point slot.
    The user should then drag the Transform's center handle in the Viewer.
    """
    # Find first disabled FG slot
    for i in range(1, MAX_FG_POINTS + 1):
        idx = f"{i:02d}"
        enabled_knob = node.knob(f"fg_{idx}_enabled")
        
        if enabled_knob and not enabled_knob.value():
            # Enable this slot
            enabled_knob.setValue(True)
            
            # Get input format to center the new point
            input_node = node.input(0)
            if input_node:
                fmt = input_node.format()
                center_x = fmt.width() / 2
                center_y = fmt.height() / 2
            else:
                center_x, center_y = 960, 540  # Default HD center
            
            # Position the transform center
            transform = _get_internal_node(node, f"FG_{idx}")
            if transform:
                _set_transform_center(transform, center_x, center_y)
            
            print(f"[H2 SamViT] Added FG point {i} at ({center_x}, {center_y})")
            print(f"[H2 SamViT] Double-click gizmo, select FG_{idx}, drag center in Viewer")
            return True
    
    nuke.message("Maximum FG points (16) reached.")
    return False


def add_bg_point(node) -> bool:
    """
    Enable the next available BG point slot.
    The user should then drag the Transform's center handle in the Viewer.
    """
    # Find first disabled BG slot
    for i in range(1, MAX_BG_POINTS + 1):
        idx = f"{i:02d}"
        enabled_knob = node.knob(f"bg_{idx}_enabled")
        
        if enabled_knob and not enabled_knob.value():
            # Enable this slot
            enabled_knob.setValue(True)
            
            # Get input format
            input_node = node.input(0)
            if input_node:
                fmt = input_node.format()
                center_x = fmt.width() / 2
                center_y = fmt.height() / 2
            else:
                center_x, center_y = 960, 540
            
            # Position slightly offset from center
            transform = _get_internal_node(node, f"BG_{idx}")
            if transform:
                _set_transform_center(transform, center_x + 100, center_y + 100)
            
            print(f"[H2 SamViT] Added BG point {i}")
            print(f"[H2 SamViT] Double-click gizmo, select BG_{idx}, drag center in Viewer")
            return True
    
    nuke.message("Maximum BG points (16) reached.")
    return False


def clear_all_points(node) -> None:
    """Disable all FG and BG points. Bounding boxes are not affected."""
    # Disable all FG points
    for i in range(1, MAX_FG_POINTS + 1):
        idx = f"{i:02d}"
        enabled_knob = node.knob(f"fg_{idx}_enabled")
        if enabled_knob:
            enabled_knob.setValue(False)
    
    # Disable all BG points
    for i in range(1, MAX_BG_POINTS + 1):
        idx = f"{i:02d}"
        enabled_knob = node.knob(f"bg_{idx}_enabled")
        if enabled_knob:
            enabled_knob.setValue(False)
    
    print("[H2 SamViT] All points cleared.")


# ─────────────────────────────────────────────────────────────────────
#  Bounding box management
# ─────────────────────────────────────────────────────────────────────

def get_pos_bbox(node) -> Optional[Tuple[float, float, float, float]]:
    """
    Get the positive bounding box as (x_min, y_min, x_max, y_max).
    
    Returns None if bbox is disabled or invalid.
    """
    enabled_knob = node.knob("bbox_enabled")
    if not enabled_knob or not enabled_knob.value():
        return None
    
    cornerpin = _get_internal_node(node, "POS_BBOX")
    return _get_cornerpin_bbox(cornerpin)


def get_neg_bbox(node) -> Optional[Tuple[float, float, float, float]]:
    """
    Get the negative bounding box as (x_min, y_min, x_max, y_max).
    SAM3 feature for exclude regions.
    
    Returns None if bbox is disabled or invalid.
    """
    enabled_knob = node.knob("neg_bbox_enabled")
    if not enabled_knob or not enabled_knob.value():
        return None
    
    cornerpin = _get_internal_node(node, "NEG_BBOX")
    return _get_cornerpin_bbox(cornerpin)


def reset_bbox(node) -> None:
    """Reset the positive bounding box to default position."""
    enabled_knob = node.knob("bbox_enabled")
    if enabled_knob:
        enabled_knob.setValue(False)
    
    # Set to default rectangle in center
    input_node = node.input(0)
    if input_node:
        fmt = input_node.format()
        w, h = fmt.width(), fmt.height()
        x1, y1 = w * 0.25, h * 0.25
        x2, y2 = w * 0.75, h * 0.75
    else:
        x1, y1 = 100, 100
        x2, y2 = 200, 200
    
    cornerpin = _get_internal_node(node, "POS_BBOX")
    _set_cornerpin_bbox(cornerpin, x1, y1, x2, y2)
    
    print("[H2 SamViT] Positive bbox reset.")


def reset_neg_bbox(node) -> None:
    """Reset the negative bounding box to default position."""
    enabled_knob = node.knob("neg_bbox_enabled")
    if enabled_knob:
        enabled_knob.setValue(False)
    
    # Set to small default rectangle
    cornerpin = _get_internal_node(node, "NEG_BBOX")
    _set_cornerpin_bbox(cornerpin, 50, 50, 150, 150)
    
    print("[H2 SamViT] Negative bbox reset.")


# ─────────────────────────────────────────────────────────────────────
#  Inference data collection
# ─────────────────────────────────────────────────────────────────────

def collect_prompt_data(node) -> Dict[str, Any]:
    """
    Collect all prompt data (points + bboxes) for inference.
    
    Returns:
        Dict containing:
        - points: List of {x, y, label} dicts
        - pos_bbox: (x1, y1, x2, y2) or None
        - neg_bbox: (x1, y1, x2, y2) or None (SAM3 only)
        - text_prompt: str or None (Text Prompt mode)
        - pipeline_mode: "Point / Bbox" or "Text Prompt"
    """
    pipeline_mode = node.knob("pipeline_mode").value()
    
    data = {
        "pipeline_mode": pipeline_mode,
        "points": [],
        "pos_bbox": None,
        "neg_bbox": None,
        "text_prompt": None,
    }
    
    if pipeline_mode == "Text Prompt":
        text_knob = node.knob("text_prompt")
        if text_knob:
            data["text_prompt"] = text_knob.value()
    
    # Always collect points (even in Text Prompt mode for instance selection)
    all_points = get_all_points(node)
    data["points"] = [{"x": p["x"], "y": p["y"], "label": p["label"]} for p in all_points]
    
    # Collect bounding boxes
    data["pos_bbox"] = get_pos_bbox(node)
    data["neg_bbox"] = get_neg_bbox(node)
    
    return data


# ─────────────────────────────────────────────────────────────────────
#  onCreate / knobChanged callbacks
# ─────────────────────────────────────────────────────────────────────

def on_create(node) -> None:
    """
    Called when the gizmo is created.
    Initialize internal state and cache.
    """
    print("[H2 SamViT] Node created: " + node.fullName())
    
    # Update cache limit display
    try:
        _update_cache_limit_display(node)
    except:
        pass


def on_knob_changed(node, knob) -> None:
    """
    Handle knob value changes.
    """
    global _in_knob_changed
    if _in_knob_changed:
        return
    
    _in_knob_changed = True
    try:
        knob_name = knob.name()
        
        # Pipeline mode changed - show/hide text prompt
        if knob_name == "pipeline_mode":
            mode = knob.value()
            text_knob = node.knob("text_prompt")
            if text_knob:
                if mode == "Text Prompt":
                    text_knob.setVisible(True)
                else:
                    text_knob.setVisible(False)
        
        # Cache memory % changed
        elif knob_name == "cache_memory_percent":
            _update_cache_limit_display(node)
        
        # Model family changed - show/hide SAM3 features
        elif knob_name == "model_family":
            family = knob.value()
            # Neg bbox is SAM3 only
            neg_bbox_knob = node.knob("neg_bbox_enabled")
            if neg_bbox_knob:
                if family != "SAM3":
                    neg_bbox_knob.setValue(False)
    
    finally:
        _in_knob_changed = False


def _update_cache_limit_display(node) -> None:
    """Update the cache limit label based on current settings."""
    import psutil
    
    try:
        cache_percent = node.knob("cache_memory_percent").value()
        available_ram = psutil.virtual_memory().available
        cache_bytes = int(available_ram * (cache_percent / 100.0))
        cache_gb = cache_bytes / (1024 ** 3)
        
        cache_label = node.knob("cache_limit")
        if cache_label:
            cache_label.setValue(f"{cache_gb:.1f} GB")
    except ImportError:
        # psutil not available
        pass
    except Exception as e:
        print(f"[H2 SamViT] Cache calculation error: {e}")


# ─────────────────────────────────────────────────────────────────────
#  Inference execution
# ─────────────────────────────────────────────────────────────────────

def run_inference(node) -> None:
    """
    Run SAM3 + ViTMatte inference on the current frame.
    """
    print("[H2 SamViT] Running inference...")
    
    # Collect prompt data
    prompt_data = collect_prompt_data(node)
    print(f"[H2 SamViT] Points: {len(prompt_data['points'])}")
    print(f"[H2 SamViT] Pos Bbox: {prompt_data['pos_bbox']}")
    print(f"[H2 SamViT] Neg Bbox: {prompt_data['neg_bbox']}")
    print(f"[H2 SamViT] Pipeline: {prompt_data['pipeline_mode']}")
    
    if prompt_data["pipeline_mode"] == "Text Prompt":
        print(f"[H2 SamViT] Text Prompt: {prompt_data['text_prompt']}")
    
    # Get input image
    input_node = node.input(0)
    if not input_node:
        nuke.message("No input connected.")
        return
    
    # Get current frame
    frame = nuke.frame()
    
    try:
        from . import inference
        
        # Get model settings
        model_family = node.knob("model_family").value()
        model_size = node.knob("model_size").value()
        model_precision = node.knob("model_precision").value()
        
        # Get processing settings
        settings = {
            "use_vitmatte": node.knob("use_vitmatte").value(),
            "trimap_erode_radius": int(node.knob("trimap_erode_radius").value()),
            "trimap_dilate_radius": int(node.knob("trimap_dilate_radius").value()),
            "fill_holes": node.knob("fill_holes").value(),
            "fill_holes_area": int(node.knob("fill_holes_area").value()),
            "crop_padding": node.knob("crop_padding").value() / 100.0,
            "mask_shrink_grow": int(node.knob("mask_shrink_grow").value()),
            "edge_feather": int(node.knob("edge_feather").value()),
            "offset_mask_x": int(node.knob("offset_mask_x").value()),
            "offset_mask_y": int(node.knob("offset_mask_y").value()),
            "final_binary_sharp": node.knob("final_binary_sharp").value(),
        }
        
        # Run inference
        result = inference.run_sam_inference(
            node=node,
            frame=frame,
            prompt_data=prompt_data,
            model_family=model_family,
            model_size=model_size,
            precision=model_precision,
            settings=settings
        )
        
        if result is not None:
            print(f"[H2 SamViT] Inference complete for frame {frame}")
        else:
            print("[H2 SamViT] Inference returned no result")
    
    except ImportError as e:
        print(f"[H2 SamViT] Inference module not available: {e}")
        nuke.message("Inference module not available. Install dependencies first.")
    except Exception as e:
        print(f"[H2 SamViT] Inference error: {e}")
        import traceback
        traceback.print_exc()
        nuke.message(f"Inference error: {e}")


def process_sequence(node) -> None:
    """
    Process the entire frame range with temporal consistency.
    """
    print("[H2 SamViT] Processing sequence...")
    
    # Get frame range
    first_frame = nuke.root().firstFrame()
    last_frame = nuke.root().lastFrame()
    
    nuke.message(f"Processing frames {first_frame} to {last_frame}...\n"
                 "This is a placeholder - full sequence processing not yet implemented.")


# ─────────────────────────────────────────────────────────────────────
#  Model management
# ─────────────────────────────────────────────────────────────────────

def download_model_action(node) -> None:
    """Download the selected model weights."""
    model_family = node.knob("model_family").value()
    model_size = node.knob("model_size").value()
    
    print(f"[H2 SamViT] Downloading {model_family} {model_size}...")
    
    try:
        from . import model_manager
        success = model_manager.download_model(model_family, model_size)
        
        status_knob = node.knob("model_status")
        if success:
            status_knob.setValue(f"{model_family} {model_size} ready")
            print("[H2 SamViT] Model downloaded successfully.")
        else:
            status_knob.setValue("Download failed")
    except ImportError:
        nuke.message("Model manager not available. Install dependencies first.")
    except Exception as e:
        print(f"[H2 SamViT] Model download error: {e}")
        nuke.message(f"Download error: {e}")


# ─────────────────────────────────────────────────────────────────────
#  Parameter reset
# ─────────────────────────────────────────────────────────────────────

_DEFAULT_VALUES = {
    "enable_edit": True,
    "show_ui_overlays": True,
    "show_point_labels": True,
    "fg_point_color": [0, 1, 0],
    "bg_point_color": [1, 0, 0],
    "bbox_color": [0, 0.65, 1],
    "neg_bbox_color": [1, 0.3, 0],
    "overlay_scale": 1.0,
    "input_threshold": 100,
    "fill_holes": True,
    "fill_holes_area": 16,
    "crop_padding": 20,
    "use_vitmatte": False,
    "trimap_erode_radius": 5,
    "trimap_dilate_radius": 15,
    "show_trimap_overlay": False,
    "output_alpha_mode": "Straight",
    "display_mode": "Overlay",
    "show_mask_overlay": True,
    "overlay_color": [0.5, 0, 0, 0.5],
    "mask_shrink_grow": 0,
    "edge_feather": 0,
    "offset_mask_x": 0,
    "offset_mask_y": 0,
    "final_binary_sharp": True,
    "enable_temporal_consistency": True,
    "temporal_weight": 50,
    "suppression_threshold": 30,
    "enable_temporal": False,
    "temporal_smoothing": 50,
    "smooth_edges_only": False,
    "edge_width": 8,
    "cache_memory_percent": 25,
    "bbox_enabled": False,
    "neg_bbox_enabled": False,
}


def reset_all_parameters(node) -> None:
    """Reset all parameters to default values."""
    for knob_name, default_value in _DEFAULT_VALUES.items():
        knob = node.knob(knob_name)
        if knob:
            try:
                if isinstance(default_value, list):
                    knob.setValue(default_value)
                else:
                    knob.setValue(default_value)
            except Exception as e:
                print(f"[H2 SamViT] Could not reset {knob_name}: {e}")
    
    # Clear all points
    clear_all_points(node)
    
    # Reset bboxes
    reset_bbox(node)
    reset_neg_bbox(node)
    
    print("[H2 SamViT] All parameters reset to defaults.")


# ─────────────────────────────────────────────────────────────────────
#  Utility functions for external access
# ─────────────────────────────────────────────────────────────────────

def get_input_image(node, frame: int = None):
    """
    Get the input image as a numpy array.
    
    Args:
        node: The H2_SamViT gizmo
        frame: Frame number (default: current frame)
    
    Returns:
        numpy array (H, W, C) in RGB format, or None
    """
    if frame is None:
        frame = nuke.frame()
    
    input_node = node.input(0)
    if not input_node:
        return None
    
    try:
        import numpy as np
        
        # Get format info
        fmt = input_node.format()
        width, height = fmt.width(), fmt.height()
        
        # Sample pixels
        # Note: This is a simplified approach. For production,
        # you'd want to use nuke.executeInMainThread() or
        # render to a temp file.
        
        # For now, return placeholder
        print(f"[H2 SamViT] Would read {width}x{height} frame {frame}")
        return None
        
    except Exception as e:
        print(f"[H2 SamViT] Error reading input image: {e}")
        return None
