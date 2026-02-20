# callbacks.py - Nuke callback handlers for H2 SamViT gizmo
# Handles interactive point/bbox editing, inference triggers, and UI updates

import nuke
import math
from typing import List, Tuple, Optional, Dict, Any

# Maximum number of points supported
MAX_POINTS = 32

# Re-entrancy guard — prevents recursive on_knob_changed when we
# programmatically set knob values (e.g. unticking add_fg_point).
_in_knob_changed = False


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


def download_model_action(node: nuke.Node) -> None:
    """Download the selected SAM model checkpoint."""
    from . import model_manager

    family = node.knob("model_family").value()

    # SAM3 ignores version/size knobs
    if family == "SAM3":
        version, size = "3.0", "Default"   # dummy — get_info returns _SAM3
    else:
        version = node.knob("sam_version").value()
        size = node.knob("model_size").value()

    if model_manager.is_downloaded(family, version, size):
        nuke.message(model_manager.status_text(family, version, size))
        return

    info = model_manager.get_info(family, version, size)
    url = info.get("url", "")
    if not url:
        nuke.message(
            "No download URL configured.\n\n"
            f"Place the checkpoint manually in:\n{model_manager.MODELS_DIR}"
        )
        return

    label = f"{family} v{version} {size}" if family == "SAM2" else "SAM3"
    if not nuke.ask(
        f"Download {label}?\n"
        f"Estimated size: ~{info['mb']} MB\n\n"
        "Nuke will pause during download.\n"
        "Check Script Editor for progress."
    ):
        return

    try:
        model_manager.download(family, version, size)
        node.knob("model_status").setValue(
            model_manager.status_text(family, version, size)
        )
        nuke.message("Download complete!")
    except Exception as e:
        node.knob("model_status").setValue("Download failed")
        nuke.message(f"Download failed:\n{e}")


def update_model_status(node: nuke.Node) -> None:
    """Refresh the model status label based on current knob selections."""
    from . import model_manager

    family = node.knob("model_family").value()

    if family == "SAM3":
        version, size = "3.0", "Default"
    else:
        version = node.knob("sam_version").value()
        size = node.knob("model_size").value()

    status_knob = node.knob("model_status")
    if status_knob:
        status_knob.setValue(model_manager.status_text(family, version, size))


def reset_all_parameters(node: nuke.Node) -> None:
    """Reset all parameters to their default values."""
    # Clear all points
    clear_all_points(node)
    
    # Clear bounding box
    clear_box(node)
    
    # Reset tool states
    defaults = {
        "model_family": "SAM2",
        "sam_version": "2.1",
        "model_size": "Large",
        "model_precision": "fp16",
        "enable_edit": True,
        "draw_box": False,
        "add_fg_point": False,
        "add_bg_point": False,
        "bbox_enabled": 0,
        "input_threshold": 100,
        "black_point": 0.0,
        "white_point": 1.0,
        "fill_holes": True,
        "fill_holes_area": 16,
        "crop_padding": 20,
        "show_trimap_overlay": False,
        "trimap_overlay_opacity": 0.6,
        "trimap_erode_radius": 3,
        "trimap_dilate_radius": 10,
        "output_alpha_mode": "Straight",
        "display_mode": "Overlay",
        "show_mask_overlay": True,
        "overlay_color": [0.5, 0, 0, 0.5],
        "mask_shrink_grow": 0,
        "edge_feather": 0,
        "offset_mask_x": 0,
        "offset_mask_y": 0,
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
        "fg_point_color": [0, 1, 0],
        "bg_point_color": [1, 0, 0],
        "bbox_color": [0, 0.65, 1],
        "overlay_scale": 1.0,
        "cache_memory_percent": 25,
        "pipeline_mode": "Point / Bbox",
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


def _apply_visibility(node: nuke.Node) -> None:
    """Set all conditional knob visibility based on current selections.

    Called on node creation / panel open and whenever the relevant
    knobs change, so the UI always reflects the correct state.
    """
    # SAM2 vs SAM3 — hide version/size when SAM3 is active
    family_knob = node.knob("model_family")
    if family_knob:
        is_sam2 = (family_knob.value() == "SAM2")
        for k_name in ("sam_version", "model_size"):
            kn = node.knob(k_name)
            if kn:
                kn.setVisible(is_sam2)

    # Pipeline mode — show text_prompt only in Text Prompt mode
    pm_knob = node.knob("pipeline_mode")
    tp_knob = node.knob("text_prompt")
    if pm_knob and tp_knob:
        tp_knob.setVisible(pm_knob.value() == "Text Prompt")


def on_create(node: nuke.Node) -> None:
    """onCreate handler — set initial knob visibility."""
    _apply_visibility(node)
    update_model_status(node)
    # Ensure the viewer overlay is installed
    try:
        from . import ui_overlay
        ui_overlay.ensure_installed()
    except Exception:
        pass


def on_knob_changed(node: nuke.Node, knob: nuke.Knob) -> None:
    """Handle knob changes for interactive editing (ONYX-style).

    Uses a module-level ``_in_knob_changed`` flag to prevent recursive
    re-entry when we programmatically set knob values (e.g. unticking
    ``add_fg_point`` after placing a point).
    """
    global _in_knob_changed
    if _in_knob_changed:
        return
    _in_knob_changed = True
    try:
        _on_knob_changed_inner(node, knob)
    finally:
        _in_knob_changed = False


def _on_knob_changed_inner(node: nuke.Node, knob: nuke.Knob) -> None:
    """Actual handler logic — called only from on_knob_changed."""
    knob_name = knob.name()

    # ── showPanel: re-apply visibility when user opens the properties ──
    if knob_name == "showPanel":
        _apply_visibility(node)
        return

    # ── Pipeline mode: show/hide text_prompt ──
    if knob_name == "pipeline_mode":
        _apply_visibility(node)

    # ── Model selection: show/hide SAM2-specific knobs + refresh status ──
    if knob_name in ("model_family", "sam_version", "model_size"):
        _apply_visibility(node)
        update_model_status(node)

    # ── Tool mutual exclusivity: draw_box / add_fg_point / add_bg_point ──
    tool_knobs = ["draw_box", "add_fg_point", "add_bg_point"]
    if knob_name in tool_knobs and knob.value():
        for other in tool_knobs:
            if other != knob_name:
                k = node.knob(other)
                if k:
                    k.setValue(False)

    # ── Draw Box → sync bbox_enabled ──
    if knob_name == "draw_box":
        bbox_knob = node.knob("bbox_enabled")
        if bbox_knob:
            bbox_knob.setValue(1 if knob.value() else 0)

    # ── Add FG Point checkbox: place a point, then untick ──
    if knob_name == "add_fg_point" and knob.value():
        _place_point(node, is_foreground=True)
        knob.setValue(False)

    # ── Add BG Point checkbox: place a point, then untick ──
    if knob_name == "add_bg_point" and knob.value():
        _place_point(node, is_foreground=False)
        knob.setValue(False)

    # ── Cache display ──
    if knob_name == "cache_memory_percent":
        update_cache_limit_display(node)


def _place_point(node: nuke.Node, is_foreground: bool) -> None:
    """Place a new point at the centre of the input frame.

    After placement the user drags the XY crosshair in the Viewer
    to move it.  Expand Points List to see all active handles.
    """
    input_node = node.input(0)
    if input_node:
        fmt = input_node.format()
        cx, cy = fmt.width() / 2.0, fmt.height() / 2.0
    else:
        cx, cy = 960.0, 540.0

    kind = "FG" if is_foreground else "BG"
    if add_point(node, cx, cy, is_foreground=is_foreground):
        print(
            f"[H2 SamViT] {kind} point placed at ({int(cx)}, {int(cy)}).  "
            "Drag its crosshair in the Viewer to reposition."
        )


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
    
    if pipeline_mode == "Point / Bbox":
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
    family = node.knob("model_family").value()

    # SAM3 ignores the version/size knobs (they're hidden)
    if family == "SAM3":
        sam_version = "3.0"
        model_size = "Default"
    else:
        sam_version = node.knob("sam_version").value()
        model_size = node.knob("model_size").value()

    params = {
        # Model selection
        "model_family": family,
        "sam_version": sam_version,
        "model_size": model_size,
        "model_precision": node.knob("model_precision").value(),

        # Pre-processing
        "input_threshold": node.knob("input_threshold").value(),
        "black_point": node.knob("black_point").value(),
        "white_point": node.knob("white_point").value(),
        "fill_holes": node.knob("fill_holes").value(),
        "fill_holes_area": node.knob("fill_holes_area").value(),
        
        # Trimap / Refiner
        "trimap_erode_radius": node.knob("trimap_erode_radius").value(),
        "trimap_dilate_radius": node.knob("trimap_dilate_radius").value(),
        "show_trimap_overlay": node.knob("show_trimap_overlay").value(),
        "trimap_overlay_opacity": node.knob("trimap_overlay_opacity").value(),
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
        "output_alpha_mode": node.knob("output_alpha_mode").value(),
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
    """Register global knob-changed and onCreate callbacks.

    NOTE: The gizmo file already embeds ``onCreate`` and
    ``knobChanged`` scripts that call these same functions.
    The global registration here serves only as a safety-net
    for nodes that were somehow created without the embedded
    scripts (e.g. via ``nuke.createNode`` before the gizmo
    was sourced).
    """
    nuke.addKnobChanged(
        lambda: on_knob_changed(nuke.thisNode(), nuke.thisKnob()),
        nodeClass="H2_SamViT"
    )
    nuke.addOnCreate(
        lambda: on_create(nuke.thisNode()),
        nodeClass="H2_SamViT"
    )


# Auto-register on import
try:
    register_callbacks()
    # Install the viewer overlay (Ctrl+click, Shift+drag)
    from . import ui_overlay
    ui_overlay.ensure_installed()
except:
    pass
