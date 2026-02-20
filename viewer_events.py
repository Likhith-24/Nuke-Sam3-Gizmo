# viewer_events.py — Mode-based viewer click handler for H2 SamViT
#
# Installs a lightweight Qt event filter that intercepts plain
# left-clicks in the Nuke Viewer ONLY when an edit mode is active.
# When no mode is active, ALL events pass through — zero conflict
# with Nuke's native shortcuts.
#
# Reserved by Nuke (we NEVER touch these):
#   Left-Click (no mod)    → select / interact with knob handles
#   Middle-Click/Drag      → pan viewer
#   Right-Click            → viewer context menu
#   Scroll                 → zoom
#   Ctrl+Click             → colour picker
#   Ctrl+Shift+Drag        → pixel region sample
#   Alt+Middle-Drag        → frame scrub / 3D rotate
#   Shift+Middle-Drag      → constrained pan
#
# Our interaction (mode-based, NO modifiers):
#   1. User clicks a button on the H2 SamViT panel to enter a mode
#   2. Plain left-click in Viewer places points/bbox corners
#   3. Click the same button again to exit the mode
#   4. Right-click, middle-click, modifier+clicks all pass through
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import nuke

try:
    from PySide2 import QtWidgets, QtCore
    _HAS_QT = True
except ImportError:
    _HAS_QT = False

_handler = None  # singleton


# ─────────────────────────────────────────────────────────────────────
#  Event filter
# ─────────────────────────────────────────────────────────────────────

class _ViewerClickHandler(QtCore.QObject):
    """App-level event filter — mode-based click-to-place.

    Only intercepts events when an edit mode is active on a selected
    H2_SamViT node.  Otherwise every event passes straight through.
    """

    def __init__(self):
        super().__init__()
        self._bbox_corner1 = None  # (x, y) of first corner, or None

    def eventFilter(self, obj, event):  # noqa: N802
        try:
            return self._handle_event(obj, event)
        except Exception:
            return False  # Never crash Nuke

    def _handle_event(self, obj, event):
        etype = event.type()

        # ── Only mouse-button press ──
        if etype != QtCore.QEvent.MouseButtonPress:
            return False

        # ── Only plain left-click (NO modifiers) ──
        button = event.button()
        if button != QtCore.Qt.LeftButton:
            return False  # right-click → context menu, middle → pan

        mods = event.modifiers()
        if mods != QtCore.Qt.NoModifier:
            return False  # ANY modifier held → pass to Nuke

        # ── Only on the Viewer GL surface ──
        if not _is_viewer_gl(obj):
            return False

        # ── Must have an active H2 node with a mode ──
        node = _get_active_h2_node()
        if node is None:
            return False

        from H2_SamViT_Gizmo import callbacks
        mode = callbacks.get_edit_mode(node)
        if not mode:
            return False  # No mode → Nuke gets the click

        # ── Convert widget pos → image coords ──
        coords = _widget_to_image(obj, event.pos(), node)
        if coords is None:
            return False

        img_x, img_y = coords

        # ── Dispatch by mode ──
        if mode == "fg":
            self._on_add_point(node, img_x, img_y, is_fg=True)
            return True

        if mode == "bg":
            self._on_add_point(node, img_x, img_y, is_fg=False)
            return True

        if mode in ("bbox", "bbox_c2"):
            self._on_bbox_click(node, img_x, img_y)
            return True

        if mode in ("neg_bbox", "neg_bbox_c2"):
            self._on_neg_bbox_click(node, img_x, img_y)
            return True

        if mode == "delete":
            self._on_delete_nearest(node, img_x, img_y)
            return True

        return False

    # ── Actions ─────────────────────────────────────────────────

    def _on_add_point(self, node, x, y, is_fg):
        from H2_SamViT_Gizmo import callbacks
        kind = "FG" if is_fg else "BG"
        if callbacks.add_point(node, x, y, is_foreground=is_fg):
            print(f"[H2 SamViT] {kind} point at ({int(x)}, {int(y)})")
            _safe_overlay_refresh(node)

    def _on_bbox_click(self, node, x, y):
        from H2_SamViT_Gizmo import callbacks
        if self._bbox_corner1 is None:
            self._bbox_corner1 = (x, y)
            callbacks.update_mode_status(node, "bbox_c2")
            print(f"[H2 SamViT] Pos bbox corner 1: ({int(x)}, {int(y)})"
                  " — click corner 2")
        else:
            x1, y1 = self._bbox_corner1
            callbacks.set_bbox(node, x1, y1, x, y)
            self._bbox_corner1 = None
            db = node.knob("draw_box")
            if db:
                db.setValue(True)
            callbacks.exit_edit_mode(node)
            print(f"[H2 SamViT] Pos bbox set: "
                  f"({int(min(x1, x))},{int(min(y1, y))}) → "
                  f"({int(max(x1, x))},{int(max(y1, y))})")
            _safe_overlay_refresh(node)

    def _on_neg_bbox_click(self, node, x, y):
        from H2_SamViT_Gizmo import callbacks
        if self._bbox_corner1 is None:
            self._bbox_corner1 = (x, y)
            callbacks.update_mode_status(node, "neg_bbox_c2")
            print(f"[H2 SamViT] Neg bbox corner 1: ({int(x)}, {int(y)})"
                  " — click corner 2")
        else:
            x1, y1 = self._bbox_corner1
            callbacks.set_neg_bbox(node, x1, y1, x, y)
            self._bbox_corner1 = None
            callbacks.exit_edit_mode(node)
            print(f"[H2 SamViT] Neg bbox set: "
                  f"({int(min(x1, x))},{int(min(y1, y))}) → "
                  f"({int(max(x1, x))},{int(max(y1, y))})")
            _safe_overlay_refresh(node)

    def _on_delete_nearest(self, node, x, y):
        from H2_SamViT_Gizmo import callbacks
        idx = callbacks.find_closest_point(node, x, y, threshold=40.0)
        if idx is not None:
            callbacks.delete_point(node, idx)
            print(f"[H2 SamViT] Deleted point {idx}")
            _safe_overlay_refresh(node)
        else:
            print(f"[H2 SamViT] No point near ({int(x)}, {int(y)})")


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _safe_overlay_refresh(node):
    """Overlay refresh — uses full render (safe from Qt event context)."""
    try:
        from H2_SamViT_Gizmo import ui_overlay
        ui_overlay.render_overlay(node)
    except Exception as e:
        print(f"[H2 SamViT] Overlay refresh error: {e}")


def _get_active_h2_node():
    """Return the first selected H2_SamViT node with Enable Edit on."""
    try:
        for n in nuke.selectedNodes():
            ee = n.knob("enable_edit")
            mf = n.knob("model_family")
            if ee and mf and ee.value():
                return n
    except Exception:
        pass
    return None


def _is_viewer_gl(widget):
    """Return True if *widget* is a Nuke Viewer's GL surface."""
    cn = type(widget).__name__
    if "GL" not in cn and "Viewport" not in cn:
        return False

    parent = widget.parent()
    depth = 0
    while parent is not None and depth < 15:
        pn = type(parent).__name__
        on = ""
        try:
            on = parent.objectName() or ""
        except Exception:
            pass
        lp = pn.lower()
        lo = on.lower()
        if "viewer" in lp or "viewer" in lo:
            return True
        if "dag" in lp or "nodegraph" in lo:
            return False
        try:
            parent = parent.parent()
        except Exception:
            break
        depth += 1
    return widget.width() > 200 and widget.height() > 200


def _widget_to_image(widget, local_pos, node):
    """Convert viewer mouse pos → Nuke image coords (Y-up, bot-left).

    Assumes centred-fit display mode.
    """
    input_node = node.input(0)
    if input_node is None:
        return None

    fmt = input_node.format()
    img_w = fmt.width()
    img_h = fmt.height()
    par = fmt.pixelAspect()
    if img_w < 1 or img_h < 1:
        return None

    ww = float(widget.width())
    wh = float(widget.height())
    if ww < 1 or wh < 1:
        return None

    display_w = img_w * par
    display_h = float(img_h)

    zoom = min(ww / display_w, wh / display_h)
    rendered_w = display_w * zoom
    rendered_h = display_h * zoom

    off_x = (ww - rendered_w) / 2.0
    off_y = (wh - rendered_h) / 2.0

    mx = float(local_pos.x())
    my = float(local_pos.y())

    img_x = (mx - off_x) / zoom / par
    img_y = img_h - (my - off_y) / zoom

    return (img_x, img_y)


# ─────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────

def install():
    """Install the viewer click handler."""
    global _handler
    if not _HAS_QT:
        print("[H2 SamViT] PySide2 not available — click handler disabled.")
        return
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    if _handler is None:
        _handler = _ViewerClickHandler()
    app.installEventFilter(_handler)
    print("[H2 SamViT] Viewer click handler active (mode-based)")


def uninstall():
    """Remove the viewer click handler."""
    global _handler
    if _handler is not None and _HAS_QT:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.removeEventFilter(_handler)
        _handler = None
