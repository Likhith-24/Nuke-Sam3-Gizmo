# ui_overlay.py  —  Viewer overlay for H2 SamViT
#
# Renders coloured point and bounding-box annotations into a transparent
# PNG image and feeds it into the gizmo's internal **OverlaySource** node.
# The gizmo's **OverlayMerge** (Merge2 / over / RGB-only) composites it
# over the main image so the user sees green/red dots and a cyan bbox
# directly in the Nuke Viewer — no fragile Qt overlay needed.
#
# Call ``render_overlay(node)`` whenever points, bbox, or appearance
# knobs change.  Call ``clear_overlay(node)`` to remove the overlay.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

import nuke


# Temp directory for overlay PNGs
_OVERLAY_DIR = os.path.join(tempfile.gettempdir(), "h2_samvit_overlays")
os.makedirs(_OVERLAY_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
#  Colour helpers
# ─────────────────────────────────────────────────────────────────────

def _knob_rgb(node, name: str, default: Tuple[int, int, int] = (0, 255, 0)):
    """Read a Color_Knob (type 18) and return an (R, G, B) tuple 0-255."""
    k = node.knob(name)
    if not k:
        return default
    v = k.value()
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return tuple(min(255, max(0, int(c * 255))) for c in v[:3])
    return default


# ─────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────

def render_overlay(node) -> None:
    """(Re-)generate the coloured overlay image and update the gizmo.

    Called from ``callbacks.py`` whenever points, bbox, or appearance
    knobs change.  Writes a transparent RGBA PNG and injects it into
    the gizmo's internal OverlaySource Read node.  The OverlayMerge
    node composites it over the main image (RGB only — alpha is
    preserved from CopyAlpha).
    """
    path = render_overlay_image(node)
    if path is None:
        clear_overlay(node)
        return
    _update_overlay_node(node, path)


def render_overlay_image(node) -> Optional[str]:
    """Render the coloured overlay PNG and return its file path.

    Returns ``None`` if there are no points or bbox to draw, or if
    Pillow is not available.  This function does NOT touch the Nuke
    node graph and is therefore safe to call from ``knobChanged``.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None  # Pillow not available — skip overlay

    # ── Input dimensions ────────────────────────────────────────
    input_node = node.input(0)
    if not input_node:
        return None
    fmt = input_node.format()
    w, h = fmt.width(), fmt.height()
    if w < 1 or h < 1:
        return None

    # ── Collect prompts ─────────────────────────────────────────
    from . import callbacks
    points = callbacks.get_enabled_points(node)
    bbox = callbacks.get_bbox(node)
    neg_bbox = callbacks.get_neg_bbox(node)

    if not points and not bbox and not neg_bbox:
        return None

    # ── Appearance knobs ────────────────────────────────────────
    fg_rgb  = _knob_rgb(node, "fg_point_color", (0, 255, 0))
    bg_rgb  = _knob_rgb(node, "bg_point_color", (255, 0, 0))
    box_rgb = _knob_rgb(node, "bbox_color",     (0, 167, 255))
    neg_box_rgb = _knob_rgb(node, "neg_bbox_color", (255, 77, 0))

    scale = 1.0
    sk = node.knob("overlay_scale")
    if sk:
        try:
            scale = max(0.25, float(sk.value()))
        except Exception:
            pass

    show_labels = True
    lk = node.knob("show_point_labels")
    if lk:
        show_labels = bool(lk.value())

    # ── Create transparent RGBA image ───────────────────────────
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    radius = max(4, int(10 * scale))
    half   = max(2, int(radius * 0.5))
    lw     = max(1, int(2 * scale))

    # ── Draw bounding box ───────────────────────────────────────
    if bbox:
        bx1, by1, bx2, by2 = bbox
        # Nuke Y-up → image Y-down
        iy_top = int(h - by2)
        iy_bot = int(h - by1)
        ix_left, ix_right = int(bx1), int(bx2)

        # Semi-transparent fill
        draw.rectangle([ix_left, iy_top, ix_right, iy_bot],
                       fill=(*box_rgb, 25))
        # Solid border
        draw.rectangle([ix_left, iy_top, ix_right, iy_bot],
                       outline=(*box_rgb, 200),
                       width=max(1, int(2 * scale)))
        # Corner handles
        hs = max(3, int(6 * scale))
        for cx, cy in [(ix_left, iy_top), (ix_right, iy_top),
                       (ix_right, iy_bot), (ix_left, iy_bot)]:
            draw.rectangle([cx - hs, cy - hs, cx + hs, cy + hs],
                           fill=(*box_rgb, 230))

    # ── Draw negative bounding box (SAM3 exclude region) ────────
    if neg_bbox:
        bx1, by1, bx2, by2 = neg_bbox
        iy_top = int(h - by2)
        iy_bot = int(h - by1)
        ix_left, ix_right = int(bx1), int(bx2)

        # Semi-transparent fill (red-ish tint)
        draw.rectangle([ix_left, iy_top, ix_right, iy_bot],
                       fill=(*neg_box_rgb, 30))
        # Dashed-style border (solid for now — PIL has no dashes)
        draw.rectangle([ix_left, iy_top, ix_right, iy_bot],
                       outline=(*neg_box_rgb, 200),
                       width=max(1, int(2 * scale)))
        # Corner handles
        hs = max(3, int(6 * scale))
        for cx, cy in [(ix_left, iy_top), (ix_right, iy_top),
                       (ix_right, iy_bot), (ix_left, iy_bot)]:
            draw.rectangle([cx - hs, cy - hs, cx + hs, cy + hs],
                           fill=(*neg_box_rgb, 230))
        # Draw an × across the box to indicate "exclude"
        draw.line([(ix_left, iy_top), (ix_right, iy_bot)],
                  fill=(*neg_box_rgb, 120), width=max(1, int(1.5 * scale)))
        draw.line([(ix_right, iy_top), (ix_left, iy_bot)],
                  fill=(*neg_box_rgb, 120), width=max(1, int(1.5 * scale)))

    # ── Draw points ─────────────────────────────────────────────
    # Try to load a font once for labels
    _font = None
    if show_labels:
        try:
            fsize = max(10, int(13 * scale))
            for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                         "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                         "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
                if os.path.isfile(path):
                    _font = ImageFont.truetype(path, fsize)
                    break
            if _font is None:
                _font = ImageFont.load_default()
        except Exception:
            _font = None

    for pt in points:
        px = int(pt["x"])
        py = int(h - pt["y"])  # Nuke Y-up → image Y-down
        col = fg_rgb if pt["is_foreground"] else bg_rgb

        # Filled circle with white outline
        draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                     fill=(*col, 200),
                     outline=(255, 255, 255, 230),
                     width=max(1, int(1.5 * scale)))

        # +  (FG) or –  (BG) symbol
        draw.line([(px - half, py), (px + half, py)],
                  fill=(255, 255, 255, 255), width=lw)
        if pt["is_foreground"]:
            draw.line([(px, py - half), (px, py + half)],
                      fill=(255, 255, 255, 255), width=lw)

        # Numeric index label
        if show_labels and _font is not None:
            try:
                draw.text((px + radius + 4, py - radius - 2),
                          str(pt["index"]),
                          fill=(255, 255, 255, 220), font=_font)
            except Exception:
                pass

    # ── Save PNG (fast, minimal compression) ────────────────────
    overlay_path = os.path.join(_OVERLAY_DIR, f"{node.name()}_overlay.png")
    img.save(overlay_path, "PNG", compress_level=1)

    return overlay_path


def refresh_overlay_safe(node) -> None:
    """Re-render overlay and update an existing Read node.

    Safe to call from ``knobChanged`` because it does NOT use
    ``node.begin()``/``node.end()`` — it accesses the internal
    OverlaySource node via its full hierarchical name.

    Only works after the first ``render_overlay()`` call has already
    converted OverlaySource from a Constant to a Read node.
    """
    path = render_overlay_image(node)
    if path is None:
        return

    # Access the internal node via its full path — no begin/end needed.
    try:
        src = nuke.toNode(node.fullName() + ".OverlaySource")
        if src and src.Class() != "Constant":
            src["file"].setValue(path)
            try:
                src["reload"].execute()
            except Exception:
                pass
    except Exception:
        pass


def initialize_overlay(node) -> None:
    """Pre-create OverlaySource as a Read node for live overlay updates.

    Called from ``on_create`` so that ``refresh_overlay_safe()`` can
    update the overlay from ``knobChanged`` without needing
    ``node.begin()``/``node.end()``.

    Creates a minimal transparent PNG and converts the default Constant
    into a Read node pointing to it.  Subsequent calls to
    ``render_overlay_image`` + ``refresh_overlay_safe`` will overwrite
    that PNG with the real coloured overlay.
    """
    try:
        from PIL import Image
    except ImportError:
        return  # Pillow not installed yet — skip

    # Create a small transparent placeholder PNG
    path = os.path.join(_OVERLAY_DIR, f"{node.name()}_overlay.png")
    if not os.path.exists(path):
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        img.save(path, "PNG")

    node.begin()
    try:
        src = nuke.toNode("OverlaySource")
        if src and src.Class() == "Constant":
            xp, yp = src.xpos(), src.ypos()
            merge = nuke.toNode("OverlayMerge")
            copy_alpha = nuke.toNode("CopyAlpha")

            nuke.delete(src)

            r = nuke.nodes.Read()
            r.setName("OverlaySource")
            r["file"].setValue(path)
            r["raw"].setValue(True)
            # Ensure overlay is visible at ANY timeline frame
            r["first"].setValue(1)
            r["last"].setValue(1)
            r["before"].setValue("hold")
            r["after"].setValue("hold")
            r["on_error"].setValue("nearest frame")
            r.setXpos(xp)
            r.setYpos(yp)

            if merge and copy_alpha:
                merge.setInput(0, copy_alpha)
                merge.setInput(1, r)
    finally:
        node.end()


def clear_overlay(node) -> None:
    """Remove any overlay — write a transparent PNG and update the Read.

    Keeps OverlaySource as a Read node (does NOT revert to Constant)
    so that ``refresh_overlay_safe()`` continues to work from
    ``knobChanged`` callbacks.
    """
    try:
        from PIL import Image
    except ImportError:
        return

    # Determine dimensions from input
    input_node = node.input(0)
    if input_node:
        fmt = input_node.format()
        w, h = fmt.width(), fmt.height()
    else:
        w, h = 4, 4

    # Write transparent PNG
    path = os.path.join(_OVERLAY_DIR, f"{node.name()}_overlay.png")
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    img.save(path, "PNG", compress_level=1)

    # Update the Read node (or create it from Constant)
    _update_overlay_node(node, path)


# ─────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _update_overlay_node(node, path: str) -> None:
    """Swap the OverlaySource Constant → Read (first call) or reload
    the existing Read node (subsequent calls)."""
    node.begin()
    try:
        src        = nuke.toNode("OverlaySource")
        merge      = nuke.toNode("OverlayMerge")
        copy_alpha = nuke.toNode("CopyAlpha")

        if src is None:
            return

        if src.Class() == "Constant":
            # First overlay — replace Constant with a Read node
            xp, yp = src.xpos(), src.ypos()
            nuke.delete(src)

            r = nuke.nodes.Read()
            r.setName("OverlaySource")
            r["file"].setValue(path)
            r["raw"].setValue(True)  # Overlay is pre-rendered data — no colorspace
            # Ensure overlay is visible at ANY timeline frame
            r["first"].setValue(1)
            r["last"].setValue(1)
            r["before"].setValue("hold")
            r["after"].setValue("hold")
            r["on_error"].setValue("nearest frame")
            r.setXpos(xp)
            r.setYpos(yp)

            # Re-wire Merge inputs: B = main image, A = overlay
            if merge and copy_alpha:
                merge.setInput(0, copy_alpha)
                merge.setInput(1, r)
        else:
            # Existing Read — just update file path and reload
            src["file"].setValue(path)
            try:
                src["reload"].execute()
            except Exception:
                pass
    finally:
        node.end()

    try:
        nuke.updateUI()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
#  Backward-compat stubs
# ─────────────────────────────────────────────────────────────────────
# Old code called install() / ensure_installed() for a Qt overlay.
# These are now harmless no-ops — the overlay is rendered into the
# gizmo's internal node graph instead.

def install() -> bool:
    """No-op — overlay is now rendered via internal gizmo nodes."""
    return True

def uninstall() -> None:
    """No-op."""
    pass

def ensure_installed() -> None:
    """No-op."""
    pass
