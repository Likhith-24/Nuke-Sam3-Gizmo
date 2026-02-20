# callbacks.py - Nuke callback handlers for H2 SamViT gizmo
# Handles interactive point/bbox editing, inference triggers, and UI updates
#
# IMPORTANT: Never call nuke.execute(), nuke.updateUI(), or modify the
# internal node graph (node.begin()/end()) inside a knobChanged callback.
# Nuke is already executing while processing callbacks and will show
# "I'm already executing something else" if you try.

import nuke
import math
from typing import List, Tuple, Optional, Dict, Any

# Maximum number of points supported
MAX_POINTS = 32

# Re-entrancy guard — prevents recursive on_knob_changed when we
# programmatically set knob values (e.g. unticking add_fg_point).
_in_knob_changed = False


# ─────────────────────────────────────────────────────────────────────
#  Edit-mode management  (used by viewer_events.py + gizmo buttons)
# ─────────────────────────────────────────────────────────────────────

_edit_modes = {}  # {node_fullname: mode_string}

_MODE_LABELS = {
    "": "<span style='color:#888'>Ready</span>",
    "fg": ("<span style='color:#00ff00'><b>\u25cf Placing FG points</b></span>"
           " \u2014 click in Viewer, click button again to stop"),
    "bg": ("<span style='color:#ff4444'><b>\u25cf Placing BG points</b></span>"
           " \u2014 click in Viewer, click button again to stop"),
    "bbox": ("<span style='color:#4488ff'><b>\u25a1 Draw Bbox</b></span>"
             " \u2014 click corner 1 in Viewer"),
    "bbox_c2": ("<span style='color:#4488ff'><b>\u25a1 Draw Bbox</b></span>"
                " \u2014 now click corner 2"),
    "neg_bbox": ("<span style='color:#ff6600'><b>\u25a1 Neg Bbox</b></span>"
                 " \u2014 click corner 1 in Viewer"),
    "neg_bbox_c2": ("<span style='color:#ff6600'><b>\u25a1 Neg Bbox</b></span>"
                    " \u2014 now click corner 2"),
    "delete": ("<span style='color:#ffaa00'><b>\u2715 Delete mode</b></span>"
               " \u2014 click near a point to remove it"),
}


def get_edit_mode(node) -> str:
    """Return the current edit mode for *node*, or '' if none."""
    return _edit_modes.get(node.fullName(), "")


def set_edit_mode(node, mode: str) -> None:
    """Set the edit mode for *node* ('' to clear)."""
    if mode:
        _edit_modes[node.fullName()] = mode
    else:
        _edit_modes.pop(node.fullName(), None)
    update_mode_status(node, mode)


def toggle_mode(node, mode: str) -> None:
    """Toggle *mode* on/off for *node*.  Called by gizmo buttons.

    Also ensures the overlay is initialised (Constant → Read)
    the first time any mode is activated, since this is a button
    callback (safe for node.begin/end).
    """
    current = get_edit_mode(node)
    if current == mode:
        exit_edit_mode(node)
    else:
        # Ensure overlay is ready (safe here — button context)
        try:
            from . import ui_overlay
            ui_overlay.initialize_overlay(node)
        except Exception:
            pass
        set_edit_mode(node, mode)
        print(f"[H2 SamViT] Entered {mode.upper()} mode \u2014 click in viewer")


def exit_edit_mode(node) -> None:
    """Exit any active edit mode for *node*."""
    was = get_edit_mode(node)
    set_edit_mode(node, "")
    if was:
        print("[H2 SamViT] Edit mode exited")


def update_mode_status(node, mode: str) -> None:
    """Update the mode_status label on the gizmo (if the knob exists)."""
    k = node.knob("mode_status")
    if k is None:
        return
    k.setValue(_MODE_LABELS.get(mode, _MODE_LABELS[""]))


# ─────────────────────────────────────────────────────────────────────
#  Point handle visibility
# ─────────────────────────────────────────────────────────────────────

def _update_point_visibility(node) -> None:
    """Show/hide each XY crosshair handle based on its enabled flag.

    This makes the crosshair handles appear/disappear in the Viewer
    when points are added or removed — providing the green/red dot
    feedback the user expects.
    """
    for i in range(1, MAX_POINTS + 1):
        idx = f"{i:02d}"
        pos_knob = node.knob(f"point_{idx}")
        enabled_knob = node.knob(f"point_{idx}_enabled")
        if pos_knob and enabled_knob:
            pos_knob.setVisible(bool(enabled_knob.value()))


def _update_bbox_visibility(node) -> None:
    """Show/hide bounding box XY handles based on bbox_enabled."""
    bbox_enabled = node.knob("bbox_enabled")
    show = bool(bbox_enabled.value()) if bbox_enabled else False
    for knob_name in ("bbox_top_left", "bbox_bottom_right"):
        k = node.knob(knob_name)
        if k:
            k.setVisible(show)


# ─────────────────────────────────────────────────────────────────────
#  Point / bbox data access
# ─────────────────────────────────────────────────────────────────────

def get_enabled_points(node) -> List[Dict[str, Any]]:
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


def get_next_available_point_index(node) -> Optional[int]:
    """Find the next available (disabled) point slot."""
    for i in range(1, MAX_POINTS + 1):
        idx = f"{i:02d}"
        enabled_knob = node.knob(f"point_{idx}_enabled")
        if enabled_knob and not enabled_knob.value():
            return i
    return None


def add_point(node, x: float, y: float, is_foreground: bool = True) -> bool:
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
        _update_point_visibility(node)
        return True
    return False


def delete_point(node, point_index: int) -> bool:
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
        _update_point_visibility(node)
        return True
    return False


def clear_all_points(node) -> None:
    """Clear all points. Bounding box is not affected."""
    exit_edit_mode(node)
    for i in range(1, MAX_POINTS + 1):
        idx = f"{i:02d}"
        pos_knob = node.knob(f"point_{idx}")
        fg_knob = node.knob(f"point_{idx}_fg")
        enabled_knob = node.knob(f"point_{idx}_enabled")
        if pos_knob and fg_knob and enabled_knob:
            pos_knob.setValue([0, 0])
            fg_knob.setValue(True)
            enabled_knob.setValue(False)
    _update_point_visibility(node)
    # Full overlay render — safe here (button callback context).
    try:
        from . import ui_overlay
        ui_overlay.render_overlay(node)
    except Exception:
        pass
    print("[H2 SamViT] All points cleared.")


def clear_box(node) -> None:
    """Clear all bounding boxes (positive + negative). Points are not affected."""
    exit_edit_mode(node)
    # Clear positive bbox
    tl_knob = node.knob("bbox_top_left")
    br_knob = node.knob("bbox_bottom_right")
    if tl_knob:
        tl_knob.setValue([0, 0])
    if br_knob:
        br_knob.setValue([0, 0])
    bbox_knob = node.knob("bbox_enabled")
    if bbox_knob:
        bbox_knob.setValue(0)
    _update_bbox_visibility(node)

    # Clear negative bbox
    for kn in ("neg_bbox_top_left", "neg_bbox_bottom_right"):
        k = node.knob(kn)
        if k:
            k.setValue([0, 0])
    neg_en = node.knob("neg_bbox_enabled")
    if neg_en:
        neg_en.setValue(0)
    _update_neg_bbox_visibility(node)

    # Full overlay render — safe here (button callback context).
    try:
        from . import ui_overlay
        ui_overlay.render_overlay(node)
    except Exception:
        pass
    print("[H2 SamViT] All bounding boxes cleared.")


def get_bbox(node) -> Optional[Tuple[float, float, float, float]]:
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


def set_bbox(node, x1: float, y1: float, x2: float, y2: float) -> None:
    """Set the bounding box coordinates."""
    tl_knob = node.knob("bbox_top_left")
    br_knob = node.knob("bbox_bottom_right")

    if tl_knob and br_knob:
        tl_knob.setValue([min(x1, x2), min(y1, y2)])
        br_knob.setValue([max(x1, x2), max(y1, y2)])

    bbox_knob = node.knob("bbox_enabled")
    if bbox_knob:
        bbox_knob.setValue(1)

    _update_bbox_visibility(node)


def get_neg_bbox(node) -> Optional[Tuple[float, float, float, float]]:
    """Get negative bounding box as (x1, y1, x2, y2) or None."""
    enabled_knob = node.knob("neg_bbox_enabled")
    if not enabled_knob or not enabled_knob.value():
        return None

    tl_knob = node.knob("neg_bbox_top_left")
    br_knob = node.knob("neg_bbox_bottom_right")
    if not tl_knob or not br_knob:
        return None

    x1, y1 = tl_knob.value()
    x2, y2 = br_knob.value()
    if x1 == x2 == y1 == y2 == 0:
        return None

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return (x1, y1, x2, y2)


def set_neg_bbox(node, x1: float, y1: float, x2: float, y2: float) -> None:
    """Set the negative bounding box coordinates (SAM3 exclude region)."""
    tl_knob = node.knob("neg_bbox_top_left")
    br_knob = node.knob("neg_bbox_bottom_right")

    if tl_knob and br_knob:
        tl_knob.setValue([min(x1, x2), min(y1, y2)])
        br_knob.setValue([max(x1, x2), max(y1, y2)])

    en_knob = node.knob("neg_bbox_enabled")
    if en_knob:
        en_knob.setValue(1)

    _update_neg_bbox_visibility(node)


def clear_neg_box(node) -> None:
    """Clear the negative bounding box."""
    for kn in ("neg_bbox_top_left", "neg_bbox_bottom_right"):
        k = node.knob(kn)
        if k:
            k.setValue([0, 0])
    en = node.knob("neg_bbox_enabled")
    if en:
        en.setValue(0)
    _update_neg_bbox_visibility(node)
    try:
        from . import ui_overlay
        ui_overlay.render_overlay(node)
    except Exception:
        pass
    print("[H2 SamViT] Negative bbox cleared.")


def _update_neg_bbox_visibility(node) -> None:
    """Show/hide negative bbox handles."""
    en = node.knob("neg_bbox_enabled")
    show = bool(en.value()) if en else False
    for kn in ("neg_bbox_top_left", "neg_bbox_bottom_right"):
        k = node.knob(kn)
        if k:
            k.setVisible(show)


def find_closest_point(node, x: float, y: float, threshold: float = 20.0) -> Optional[int]:
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


def move_point(node, point_index: int, x: float, y: float) -> bool:
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


# ─────────────────────────────────────────────────────────────────────
#  Model management
# ─────────────────────────────────────────────────────────────────────

def download_model_action(node) -> None:
    """Download the selected SAM model checkpoint."""
    from . import model_manager

    family = node.knob("model_family").value()

    if family == "SAM3":
        version, size = "3.0", "Default"
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


def update_model_status(node) -> None:
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


# ─────────────────────────────────────────────────────────────────────
#  Reset
# ─────────────────────────────────────────────────────────────────────

def reset_all_parameters(node) -> None:
    """Reset all parameters to their default values."""
    clear_all_points(node)
    clear_box(node)

    defaults = {
        "model_family": "SAM2",
        "sam_version": "2.1",
        "model_size": "Large",
        "model_precision": "bf16",
        "enable_edit": True,
        "draw_box": False,
        "bbox_enabled": 0,
        "use_vitmatte": False,
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
        "final_binary_sharp": True,
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
            except Exception:
                pass

    print("[H2 SamViT] All parameters reset to defaults.")


# ─────────────────────────────────────────────────────────────────────
#  Visibility
# ─────────────────────────────────────────────────────────────────────

def _apply_visibility(node) -> None:
    """Set all conditional knob visibility based on current selections."""
    # SAM2 vs SAM3 vs SEC-4B
    family_knob = node.knob("model_family")
    family = family_knob.value() if family_knob else "SAM2"
    is_sam2 = (family == "SAM2")
    is_sam3 = (family == "SAM3")
    is_sec = (family == "SEC-4B")

    if family_knob:
        # SAM2-specific knobs (version, size dropdowns)
        for k_name in ("sam_version", "model_size"):
            kn = node.knob(k_name)
            if kn:
                kn.setVisible(is_sam2)

    # Negative bbox — SAM3 only (SAM2/SEC don't support it)
    neg_bbox_btn = node.knob("draw_neg_bbox_btn")
    if neg_bbox_btn:
        neg_bbox_btn.setVisible(is_sam3)
    for k_name in ("neg_bbox_top_left", "neg_bbox_bottom_right",
                    "neg_bbox_enabled", "neg_bbox_color",
                    "bbox_neg_label"):
        kn = node.knob(k_name)
        if kn:
            kn.setVisible(is_sam3)

    # Pipeline mode
    pm_knob = node.knob("pipeline_mode")
    tp_knob = node.knob("text_prompt")
    if pm_knob and tp_knob:
        tp_knob.setVisible(pm_knob.value() == "Text Prompt")

    # ViTMatte trimap knobs — only relevant when ViTMatte is enabled
    vitmatte_knob = node.knob("use_vitmatte")
    if vitmatte_knob:
        vitmatte_on = bool(vitmatte_knob.value())
        for k_name in ("vitmatte_info", "trimap_erode_radius",
                        "trimap_dilate_radius", "show_trimap_overlay",
                        "trimap_overlay_opacity"):
            kn = node.knob(k_name)
            if kn:
                kn.setVisible(vitmatte_on)

    # Point handle visibility
    _update_point_visibility(node)

    # Bbox handle visibility (pos + neg)
    _update_bbox_visibility(node)
    _update_neg_bbox_visibility(node)


# ─────────────────────────────────────────────────────────────────────
#  Callbacks  (onCreate / knobChanged)
# ─────────────────────────────────────────────────────────────────────

def on_create(node) -> None:
    """onCreate handler — set initial knob visibility and overlay."""
    _apply_visibility(node)
    update_model_status(node)
    # Pre-initialise OverlaySource as a Read node so that
    # refresh_overlay_safe() works from knobChanged later.
    try:
        from . import ui_overlay
        ui_overlay.initialize_overlay(node)
    except Exception:
        pass


def on_knob_changed(node, knob) -> None:
    """Handle knob changes for interactive editing (ONYX-style).

    IMPORTANT: This runs inside Nuke's execution context.
    Do NOT call nuke.execute(), nuke.updateUI(), or modify
    the internal node graph (node.begin/end) from here.
    """
    global _in_knob_changed
    if _in_knob_changed:
        return
    _in_knob_changed = True
    try:
        _on_knob_changed_inner(node, knob)
    finally:
        _in_knob_changed = False


def _on_knob_changed_inner(node, knob) -> None:
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

    # ── ViTMatte toggle: show/hide trimap knobs ──
    if knob_name == "use_vitmatte":
        _apply_visibility(node)

    # ── Draw Box → sync bbox_enabled + update handles ──
    if knob_name == "draw_box":
        bbox_knob = node.knob("bbox_enabled")
        if knob.value():
            if bbox_knob:
                bbox_knob.setValue(1)
            # If the box is empty, initialise the handles at a sensible
            # default (centre-quarter of the frame) so the user can
            # immediately see and drag them in the Viewer.
            tl = node.knob("bbox_top_left")
            br = node.knob("bbox_bottom_right")
            if tl and br:
                tlx, tly = tl.value()
                brx, bry = br.value()
                if tlx == 0 and tly == 0 and brx == 0 and bry == 0:
                    inp = node.input(0)
                    if inp:
                        fmt = inp.format()
                        w, h = fmt.width(), fmt.height()
                    else:
                        w, h = 1920, 1080
                    tl.setValue([w * 0.25, h * 0.25])
                    br.setValue([w * 0.75, h * 0.75])
        else:
            if bbox_knob:
                bbox_knob.setValue(0)
        _update_bbox_visibility(node)
        _try_refresh_overlay(node)

    # ── bbox_enabled changed directly → update handles ──
    if knob_name == "bbox_enabled":
        _update_bbox_visibility(node)
        _try_refresh_overlay(node)

    # ── Point enabled toggle → update XY handle visibility ──
    if knob_name.startswith("point_") and knob_name.endswith("_enabled"):
        _update_point_visibility(node)
        _try_refresh_overlay(node)

    # ── Point position dragged in viewer → refresh overlay dots ──
    if (knob_name.startswith("point_")
            and not knob_name.endswith(("_fg", "_enabled", "_grp"))):
        _try_refresh_overlay(node)

    # ── Bbox handles dragged in viewer → refresh overlay box ──
    if knob_name in ("bbox_top_left", "bbox_bottom_right",
                      "neg_bbox_top_left", "neg_bbox_bottom_right"):
        _try_refresh_overlay(node)

    # ── neg_bbox_enabled changed → update handles ──
    if knob_name == "neg_bbox_enabled":
        _update_neg_bbox_visibility(node)
        _try_refresh_overlay(node)

    # ── Appearance knobs changed → refresh overlay ──
    if knob_name in ("fg_point_color", "bg_point_color", "bbox_color",
                      "neg_bbox_color", "overlay_scale", "show_point_labels"):
        _try_refresh_overlay(node)

    # ── Backward-compat: old type-6 checkbox knobs for Add FG/BG Point ──
    # If the user has a node created before the type-22 button change,
    # clicking the checkbox triggers knobChanged. Handle it here.
    if knob_name == "add_fg_point" and knob.value():
        _place_point(node, is_foreground=True)
        knob.setValue(False)  # reset checkbox
        _try_refresh_overlay(node)

    if knob_name == "add_bg_point" and knob.value():
        _place_point(node, is_foreground=False)
        knob.setValue(False)  # reset checkbox
        _try_refresh_overlay(node)

    # ── Cache display ──
    if knob_name == "cache_memory_percent":
        update_cache_limit_display(node)


def _try_refresh_overlay(node) -> None:
    """Re-render the overlay after a point/bbox drag.

    Safe to call from ``knobChanged`` because it does NOT use
    ``node.begin()``/``node.end()``.  Only works after
    ``initialize_overlay`` (in ``on_create``) has converted
    OverlaySource from a Constant to a Read node.
    """
    try:
        from . import ui_overlay
        ui_overlay.refresh_overlay_safe(node)
    except Exception:
        pass


def add_fg_point_action(node) -> None:
    """Button callback — add a FG point and render the overlay.

    Called from the gizmo's 'Add FG Point' button (type 22 script knob).
    Since this is a button callback (not knobChanged), it is safe to
    modify the internal node graph via ``render_overlay``.
    """
    _place_point(node, is_foreground=True)
    try:
        from . import ui_overlay
        ui_overlay.render_overlay(node)
    except Exception:
        pass


def add_bg_point_action(node) -> None:
    """Button callback — add a BG point and render the overlay.

    Called from the gizmo's 'Add BG Point' button (type 22 script knob).
    Since this is a button callback (not knobChanged), it is safe to
    modify the internal node graph via ``render_overlay``.
    """
    _place_point(node, is_foreground=False)
    try:
        from . import ui_overlay
        ui_overlay.render_overlay(node)
    except Exception:
        pass


def _place_point(node, is_foreground: bool) -> None:
    """Place a new point at the centre of the input frame."""
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


def update_cache_limit_display(node) -> None:
    """Update the cache limit display based on memory percentage."""
    cache_pct = node.knob("cache_memory_percent").value()
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        cache_gb = available_gb * (cache_pct / 100)
        frame_size_mb = 9
        estimated_frames = int(cache_gb * 1024 / frame_size_mb)
        cache_limit_knob = node.knob("cache_limit")
        if cache_limit_knob:
            cache_limit_knob.setValue(
                f"{cache_gb:.1f} GB (~{estimated_frames} frames @ 3840x2160)"
            )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────

def run_inference(node) -> None:
    """Trigger SAM + ViTMatte inference on the current frame."""
    from . import inference

    if not node.knob("enable_edit").value():
        nuke.message("Enable Edit is disabled. Enable it to run inference.")
        return

    pipeline_mode = node.knob("pipeline_mode").value()

    if pipeline_mode == "Point / Bbox":
        points = get_enabled_points(node)
        bbox = get_bbox(node)
        neg_bbox = get_neg_bbox(node)

        if not points and not bbox:
            nuke.message(
                "No points or bounding box defined.\n"
                "Add prompts before running inference."
            )
            return

        inference.run_point_bbox_inference(node, points, bbox, neg_bbox)

    else:  # Text Prompt mode
        text_prompt = node.knob("text_prompt").value()
        if not text_prompt.strip():
            nuke.message("Text prompt is empty.")
            return

        points = get_enabled_points(node)
        inference.run_text_prompt_inference(node, text_prompt, points)


def process_sequence(node) -> None:
    """Run inference on every frame of the input's frame range.

    Uses the same prompts (points / bbox / text) for every frame.
    A progress bar is shown in the Nuke UI.
    """
    from . import inference

    if not node.knob("enable_edit").value():
        nuke.message("Enable Edit is disabled. Enable it to process.")
        return

    input_node = node.input(0)
    if not input_node:
        nuke.message("No input connected.")
        return

    first = int(input_node.firstFrame())
    last  = int(input_node.lastFrame())
    if first == last:
        run_inference(node)
        return

    total = last - first + 1
    if not nuke.ask(
        f"Process frames {first} – {last}?\n"
        f"({total} frames)\n\n"
        "This may take a while. Progress will be shown.\n"
        "Click Yes to start."
    ):
        return

    pipeline_mode = node.knob("pipeline_mode").value()

    if pipeline_mode == "Point / Bbox":
        points = get_enabled_points(node)
        bbox   = get_bbox(node)
        neg_bbox = get_neg_bbox(node)
        if not points and not bbox:
            nuke.message("No points or bounding box defined.")
            return
    else:
        text_prompt = node.knob("text_prompt").value()
        if not text_prompt.strip():
            nuke.message("Text prompt is empty.")
            return
        points = get_enabled_points(node)

    # ── Process each frame ──
    task = nuke.ProgressTask("H2 SamViT — Processing Sequence")
    try:
        for i, frame in enumerate(range(first, last + 1)):
            if task.isCancelled():
                nuke.message(f"Cancelled at frame {frame}.")
                break

            pct = int(100.0 * i / total)
            task.setProgress(pct)
            task.setMessage(f"Frame {frame}  ({i + 1}/{total})")

            # Set current frame so image_from_nuke_node captures the right one
            nuke.root()['frame'].setValue(frame)

            try:
                if pipeline_mode == "Point / Bbox":
                    inference.run_point_bbox_inference(
                        node, points, bbox, neg_bbox)
                else:
                    inference.run_text_prompt_inference(
                        node, text_prompt, points,
                    )
            except Exception as exc:
                print(f"[H2 SamViT] Frame {frame} failed: {exc}")
                continue

        task.setProgress(100)
        task.setMessage("Done")
    finally:
        del task

    nuke.message(f"Sequence complete — {total} frames processed.")


# ─────────────────────────────────────────────────────────────────────
#  Inference parameters
# ─────────────────────────────────────────────────────────────────────

def get_inference_params(node) -> Dict[str, Any]:
    """Get all relevant parameters for inference."""
    family = node.knob("model_family").value()

    if family == "SAM3":
        sam_version = "3.0"
        model_size = "Default"
    else:
        sam_version = node.knob("sam_version").value()
        model_size = node.knob("model_size").value()

    params = {
        "model_family": family,
        "sam_version": sam_version,
        "model_size": model_size,
        "model_precision": node.knob("model_precision").value(),
        "use_vitmatte": node.knob("use_vitmatte").value() if node.knob("use_vitmatte") else False,
        "input_threshold": node.knob("input_threshold").value(),
        "black_point": node.knob("black_point").value(),
        "white_point": node.knob("white_point").value(),
        "fill_holes": node.knob("fill_holes").value(),
        "fill_holes_area": node.knob("fill_holes_area").value(),
        "trimap_erode_radius": node.knob("trimap_erode_radius").value(),
        "trimap_dilate_radius": node.knob("trimap_dilate_radius").value(),
        "show_trimap_overlay": node.knob("show_trimap_overlay").value(),
        "trimap_overlay_opacity": node.knob("trimap_overlay_opacity").value(),
        "final_binary_sharp": node.knob("final_binary_sharp").value(),
        "enable_temporal": node.knob("enable_temporal").value(),
        "temporal_smoothing": node.knob("temporal_smoothing").value(),
        "smooth_edges_only": node.knob("smooth_edges_only").value(),
        "edge_width": node.knob("edge_width").value(),
        "enable_temporal_consistency": node.knob("enable_temporal_consistency").value(),
        "temporal_weight": node.knob("temporal_weight").value(),
        "suppression_threshold": node.knob("suppression_threshold").value(),
        "display_mode": node.knob("display_mode").value(),
        "output_alpha_mode": node.knob("output_alpha_mode").value(),
        "mask_shrink_grow": node.knob("mask_shrink_grow").value(),
        "edge_feather": node.knob("edge_feather").value(),
        "offset_mask_x": node.knob("offset_mask_x").value(),
        "offset_mask_y": node.knob("offset_mask_y").value(),
        "crop_padding": node.knob("crop_padding").value(),
        "cache_memory_percent": node.knob("cache_memory_percent").value(),
    }
    return params


# ─────────────────────────────────────────────────────────────────────
#  Global registration
# ─────────────────────────────────────────────────────────────────────

def register_callbacks():
    """Register global knob-changed and onCreate callbacks."""
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
except Exception:
    pass
