# viewer_overlay.py — Qt overlay for ONYX-style interactive handles
#
# Draws coloured crosshairs, circles, bbox rectangles and mode badges
# directly on top of the Nuke Viewer using a transparent QWidget that
# sits over the GL surface.  This gives the same interactive visual
# experience as an OFX overlay-interact, without needing compiled C++.
#
# The overlay reads point / bbox data from the gizmo's knobs and
# converts image coordinates (Y-up, bottom-left origin) to widget
# coordinates (Y-down, top-left origin) assuming auto-fit display.
#
# Press Home / F in the Viewer to fit the image — this gives the
# most accurate handle placement.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import nuke

try:
    from PySide2 import QtWidgets, QtCore, QtGui

    _HAS_QT = True
except ImportError:
    try:
        from PySide6 import QtWidgets, QtCore, QtGui

        _HAS_QT = True
    except ImportError:
        _HAS_QT = False

_overlay = None  # singleton


# ─────────────────────────────────────────────────────────────────────
#  Overlay Widget
# ─────────────────────────────────────────────────────────────────────

class ViewerOverlay(QtWidgets.QWidget):
    """Transparent widget painted on top of the Viewer GL surface."""

    _CROSS = 15  # crosshair arm length  (pixels, before scale)
    _RING = 10  # circle radius
    _HNDL = 5  # bbox corner handle half-size

    def __init__(self, viewer_widget):
        super().__init__(viewer_widget)
        self._viewer = viewer_widget
        self._node = None
        self._last_sz = (0, 0)

        # ── Make transparent ──
        for flag in (
            QtCore.Qt.WA_TranslucentBackground,
            QtCore.Qt.WA_NoSystemBackground,
            QtCore.Qt.WA_TransparentForMouseEvents,
        ):
            self.setAttribute(flag, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        self._sync_geo()

        # ── Timer: ~12 fps keeps the overlay in sync ──
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(83)

        self.show()
        self.raise_()

    # ── public ────────────────────────────────────────────────

    def set_node(self, node):
        self._node = node
        self.update()

    # ── tick / geometry ───────────────────────────────────────

    def _tick(self):
        self._sync_geo()
        self._auto_node()
        self.raise_()
        self.update()

    def _sync_geo(self):
        p = self.parent()
        if p:
            pw, ph = p.width(), p.height()
            if (pw, ph) != self._last_sz:
                self.setGeometry(0, 0, pw, ph)
                self._last_sz = (pw, ph)

    def _auto_node(self):
        """Auto-detect the first selected H2_SamViT node."""
        if self._node is not None:
            try:
                _ = self._node.name()  # verify alive
            except Exception:
                self._node = None
        if self._node is None:
            try:
                for n in nuke.selectedNodes():
                    if n.knob("model_family") and n.knob("enable_edit"):
                        self._node = n
                        return
            except Exception:
                pass

    # ── painting ──────────────────────────────────────────────

    def paintEvent(self, event):  # noqa: N802
        node = self._node
        if node is None:
            return
        show_k = node.knob("show_ui_overlays")
        if show_k and not show_k.value():
            return

        try:
            from H2_SamViT_Gizmo import callbacks
        except ImportError:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        sc = self._scale(node)
        fg_c = self._col(node, "fg_point_color", (0, 255, 0))
        bg_c = self._col(node, "bg_point_color", (255, 0, 0))
        bx_c = self._col(node, "bbox_color", (0, 167, 255))
        nb_c = self._col(node, "neg_bbox_color", (255, 77, 0))

        labels_on = True
        lk = node.knob("show_point_labels")
        if lk:
            labels_on = bool(lk.value())

        # ── Points ──
        for pt in callbacks.get_enabled_points(node):
            wpos = self._i2w(node, pt["x"], pt["y"])
            if wpos is None:
                continue
            c = QtGui.QColor(*fg_c) if pt["is_foreground"] else QtGui.QColor(*bg_c)
            self._paint_cross(painter, *wpos, c, sc)
            if labels_on:
                tag = f"{'FG' if pt['is_foreground'] else 'BG'}{pt['index']}"
                self._paint_tag(painter, *wpos, tag, c, sc)

        # ── Positive bbox ──
        bbox = callbacks.get_bbox(node)
        if bbox:
            self._paint_box(painter, node, bbox, QtGui.QColor(*bx_c), sc)

        # ── Negative bbox ──
        neg = callbacks.get_neg_bbox(node)
        if neg:
            self._paint_box(painter, node, neg, QtGui.QColor(*nb_c), sc, diag=True)

        # ── Mode badge ──
        mode = callbacks.get_edit_mode(node)
        if mode:
            self._paint_badge(painter, mode)

        painter.end()

    # ── draw helpers ──────────────────────────────────────────

    def _paint_cross(self, p, wx, wy, color, sc):
        cl = int(self._CROSS * sc)
        cr = int(self._RING * sc)
        lw = max(1, int(2 * sc))
        pen = QtGui.QPen(color, lw)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        iwx, iwy = int(wx), int(wy)
        p.drawLine(iwx - cl, iwy, iwx + cl, iwy)
        p.drawLine(iwx, iwy - cl, iwx, iwy + cl)
        p.drawEllipse(QtCore.QPointF(wx, wy), cr, cr)

    def _paint_tag(self, p, wx, wy, text, color, sc):
        cr = int(self._RING * sc) + 4
        font = p.font()
        font.setPointSize(max(8, int(9 * sc)))
        font.setBold(True)
        p.setFont(font)
        # shadow
        p.setPen(QtGui.QColor(0, 0, 0, 200))
        p.drawText(int(wx + cr + 1), int(wy - 1), text)
        # foreground
        p.setPen(color)
        p.drawText(int(wx + cr), int(wy - 2), text)

    def _paint_box(self, p, node, bbox, color, sc, diag=False):
        x1, y1, x2, y2 = bbox
        tl = self._i2w(node, x1, y2)   # image top → widget top
        br = self._i2w(node, x2, y1)   # image bottom → widget bottom
        if tl is None or br is None:
            return
        lw = max(1, int(2 * sc))
        pen = QtGui.QPen(color, lw, QtCore.Qt.DashLine)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        rect = QtCore.QRectF(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
        p.drawRect(rect)

        # corner handles
        hs = max(3, int(self._HNDL * sc))
        for cx, cy in [
            (tl[0], tl[1]), (br[0], tl[1]),
            (tl[0], br[1]), (br[0], br[1]),
        ]:
            p.fillRect(int(cx - hs), int(cy - hs), hs * 2, hs * 2, color)

        if diag:
            pen_d = QtGui.QPen(color, 1)
            p.setPen(pen_d)
            p.drawLine(int(tl[0]), int(tl[1]), int(br[0]), int(br[1]))
            p.drawLine(int(br[0]), int(tl[1]), int(tl[0]), int(br[1]))

    def _paint_badge(self, p, mode):
        info = {
            "fg":          ("● Place FG Point",             (0, 255, 0)),
            "bg":          ("● Place BG Point",             (255, 68, 68)),
            "bbox":        ("□ Draw Bbox – click corner 1", (68, 136, 255)),
            "bbox_c2":     ("□ Draw Bbox – click corner 2", (68, 136, 255)),
            "neg_bbox":    ("□ Neg Bbox – click corner 1",  (255, 102, 0)),
            "neg_bbox_c2": ("□ Neg Bbox – click corner 2",  (255, 102, 0)),
            "delete":      ("✕ Delete Point – click near",  (255, 170, 0)),
        }
        text, rgb = info.get(mode, (mode, (200, 200, 200)))

        font = QtGui.QFont("Verdana", 11, QtGui.QFont.Bold)
        p.setFont(font)
        fm = QtGui.QFontMetrics(font)
        tw = fm.horizontalAdvance(text) if hasattr(fm, "horizontalAdvance") else fm.width(text)
        th = fm.height()
        pad = 10
        bg_r = QtCore.QRectF(14, 14, tw + pad * 2, th + pad)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(0, 0, 0, 190))
        p.drawRoundedRect(bg_r, 6, 6)
        p.setPen(QtGui.QColor(*rgb))
        p.drawText(int(bg_r.x() + pad), int(bg_r.y() + th + 1), text)

    # ── coordinate conversion ─────────────────────────────────

    def _i2w(self, node, ix, iy):
        """Image coords (Y-up, bot-left) → widget coords (Y-down, top-left).

        Accurate when the Viewer is in auto-fit mode (Home / F key).
        """
        inp = node.input(0) if node else None
        if inp is None:
            return None
        try:
            fmt = inp.format()
            iw = fmt.width()
            ih = fmt.height()
            par = fmt.pixelAspect()
        except Exception:
            return None
        if iw < 1 or ih < 1:
            return None

        vw = float(self._viewer.width())
        vh = float(self._viewer.height())
        if vw < 1 or vh < 1:
            return None

        dw = iw * par
        dh = float(ih)
        z = min(vw / dw, vh / dh)
        ox = (vw - dw * z) / 2.0
        oy = (vh - dh * z) / 2.0

        return (ix * par * z + ox, (ih - iy) * z + oy)

    # ── utilities ─────────────────────────────────────────────

    @staticmethod
    def _scale(node):
        sk = node.knob("overlay_scale")
        if sk:
            try:
                return max(0.25, float(sk.value()))
            except Exception:
                pass
        return 1.0

    @staticmethod
    def _col(node, name, default):
        k = node.knob(name)
        if not k:
            return default
        try:
            v = k.value()
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                return tuple(min(255, max(0, int(c * 255))) for c in v[:3])
        except Exception:
            pass
        return default


# ─────────────────────────────────────────────────────────────────────
#  Module API
# ─────────────────────────────────────────────────────────────────────

def install():
    """Try to install the overlay on the Viewer GL surface.

    Returns True on success.  If the Viewer isn't ready yet
    (common at startup), returns False — call ``ensure_installed()``
    later from the event handler.
    """
    global _overlay
    if not _HAS_QT:
        return False

    widget = _find_viewer_gl()
    if widget is None:
        return False

    _kill_existing()
    _overlay = ViewerOverlay(widget)
    print("[H2 SamViT] Qt viewer overlay active")
    return True


def ensure_installed():
    """Lazy install — safe to call on every viewer event."""
    if _overlay is not None:
        # Check parent is still valid
        try:
            p = _overlay.parent()
            if p is None or not p.isVisible():
                _kill_existing()
                return install()
        except Exception:
            _kill_existing()
            return install()
        return True
    return install()


def uninstall():
    _kill_existing()


def repaint():
    if _overlay is not None:
        _overlay.update()


def set_node(node):
    if _overlay is not None:
        _overlay.set_node(node)


def _kill_existing():
    global _overlay
    if _overlay is not None:
        try:
            _overlay._timer.stop()
            _overlay.close()
            _overlay.deleteLater()
        except Exception:
            pass
        _overlay = None


def _find_viewer_gl():
    """Locate the Nuke Viewer's GL surface widget."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return None

    for w in app.allWidgets():
        cn = type(w).__name__
        if "GL" not in cn and "Viewport" not in cn:
            continue
        parent = w.parent()
        depth = 0
        while parent is not None and depth < 15:
            pn = type(parent).__name__
            try:
                on = parent.objectName() or ""
            except Exception:
                on = ""
            lp = pn.lower()
            lo = on.lower()
            if "viewer" in lp or "viewer" in lo:
                if w.width() > 200 and w.height() > 200:
                    return w
            if "dag" in lp or "nodegraph" in lo:
                break
            try:
                parent = parent.parent()
            except Exception:
                break
            depth += 1
    return None
