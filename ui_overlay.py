# ui_overlay.py — Colour overlay for H2 SamViT
#
# Draws lightweight colour indicators that supplement Nuke's native
# XY-knob crosshair handles:
#   • Coloured rings around point positions  (green = FG, red = BG)
#   • Coloured rectangle outline for bbox / neg-bbox
#
# The actual point/bbox interaction (dragging, positioning) is handled
# entirely by Nuke's built-in type-12 XY_Knob crosshairs.
# This module only adds colour so the user can tell FG from BG.
#
# OverlaySource is a Read node baked into the gizmo.  All functions
# access it via nuke.toNode(fullName) — NO node.begin()/end() ever.
# Safe from knobChanged, Qt event filter, and button callbacks.

from __future__ import annotations
import os, tempfile
from typing import Optional, Tuple
import nuke

_OVERLAY_DIR = os.path.join(tempfile.gettempdir(), "h2_samvit_overlays")
os.makedirs(_OVERLAY_DIR, exist_ok=True)


def _knob_rgb(node, name: str, default: Tuple[int, int, int] = (0, 255, 0)):
    k = node.knob(name)
    if not k:
        return default
    v = k.value()
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return tuple(min(255, max(0, int(c * 255))) for c in v[:3])
    return default


# ─── Public API ─────────────────────────────────────────────────────

def render_overlay(node) -> None:
    """Render colour overlay and push it to the gizmo's Read node."""
    path = render_overlay_image(node)
    if path is None:
        clear_overlay(node)
        return
    _push_to_read(node, path)


def refresh_overlay_safe(node) -> None:
    """Same as render_overlay — safe from any context."""
    path = render_overlay_image(node)
    if path:
        _push_to_read(node, path)


def render_overlay_image(node) -> Optional[str]:
    """Draw colour rings + bbox outlines to a transparent PNG."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[H2 SamViT] PIL not available — colour overlay disabled")
        return None

    inp = node.input(0)
    if not inp:
        return None
    fmt = inp.format()
    w, h = fmt.width(), fmt.height()
    if w < 1 or h < 1:
        return None

    from . import callbacks
    points   = callbacks.get_enabled_points(node)
    bbox     = callbacks.get_bbox(node)
    neg_bbox = callbacks.get_neg_bbox(node)
    if not points and not bbox and not neg_bbox:
        return None

    fg_rgb      = _knob_rgb(node, "fg_point_color",  (0, 255, 0))
    bg_rgb      = _knob_rgb(node, "bg_point_color",  (255, 0, 0))
    box_rgb     = _knob_rgb(node, "bbox_color",      (0, 167, 255))
    neg_box_rgb = _knob_rgb(node, "neg_bbox_color",  (255, 77, 0))

    scale = 1.0
    sk = node.knob("overlay_scale")
    if sk:
        try:
            scale = max(0.25, float(sk.value()))
        except Exception:
            pass

    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── Coloured rings around each point ────────────────────────
    # Ring sits around the native XY crosshair — colour = FG/BG.
    ring_r = max(10, int(min(w, h) * 0.012 * scale))
    ring_w = max(3, int(ring_r * 0.35))

    for pt in points:
        px = int(pt["x"])
        py = int(h - pt["y"])          # Nuke Y-up → image Y-down
        col = fg_rgb if pt["is_foreground"] else bg_rgb
        draw.ellipse(
            [px - ring_r, py - ring_r, px + ring_r, py + ring_r],
            outline=(*col, 220), width=ring_w,
        )

    # ── Positive bbox outline ───────────────────────────────────
    if bbox:
        x1, y1, x2, y2 = bbox
        iy_top, iy_bot = int(h - y2), int(h - y1)
        bw = max(2, int(3 * scale))
        draw.rectangle(
            [int(x1), iy_top, int(x2), iy_bot],
            outline=(*box_rgb, 220), width=bw,
        )

    # ── Negative bbox outline (SAM3) ────────────────────────────
    if neg_bbox:
        x1, y1, x2, y2 = neg_bbox
        iy_top, iy_bot = int(h - y2), int(h - y1)
        bw = max(2, int(3 * scale))
        ix1, ix2 = int(x1), int(x2)
        draw.rectangle([ix1, iy_top, ix2, iy_bot],
                       outline=(*neg_box_rgb, 220), width=bw)
        draw.line([(ix1, iy_top), (ix2, iy_bot)],
                  fill=(*neg_box_rgb, 120), width=2)
        draw.line([(ix2, iy_top), (ix1, iy_bot)],
                  fill=(*neg_box_rgb, 120), width=2)

    path = os.path.join(_OVERLAY_DIR, f"{node.name()}_overlay.png")
    img.save(path, "PNG", compress_level=1)
    return path


def clear_overlay(node) -> None:
    """Write a transparent PNG to blank the overlay."""
    try:
        from PIL import Image
    except ImportError:
        return
    inp = node.input(0)
    if inp:
        fmt = inp.format()
        w, h = fmt.width(), fmt.height()
    else:
        w, h = 4, 4
    path = os.path.join(_OVERLAY_DIR, f"{node.name()}_overlay.png")
    Image.new("RGBA", (w, h), (0, 0, 0, 0)).save(path, "PNG", compress_level=1)
    _push_to_read(node, path)


def initialize_overlay(node) -> None:
    """No-op — OverlaySource is a Read node from gizmo creation."""
    pass


# ─── Internal ───────────────────────────────────────────────────────

def _push_to_read(node, path: str) -> None:
    """Set the file on the gizmo's OverlaySource Read node."""
    try:
        src = nuke.toNode(node.fullName() + ".OverlaySource")
        if src is None:
            return
        src["file"].setValue(path)
        try:
            src["raw"].setValue(True)
        except Exception:
            pass
        try:
            src["premultiplied"].setValue(False)
        except Exception:
            pass
        try:
            src["first"].setValue(1)
            src["last"].setValue(1)
            src["before"].setValue("hold")
            src["after"].setValue("hold")
        except Exception:
            pass
        try:
            src["reload"].execute()
        except Exception:
            pass
    except Exception as e:
        print(f"[H2 SamViT] Overlay update: {e}")


# ─── Stubs ──────────────────────────────────────────────────────────

def install() -> bool:
    return True

def uninstall() -> None:
    pass

def ensure_installed() -> None:
    pass
