# ui_overlay.py  —  Interactive viewer overlay for H2 SamViT
#
# Provides two capabilities:
#
#   1.  **Viewer interaction** via a Qt event-filter on the GL viewport:
#         • Ctrl + Left Click   → place FG point  (positive / green)
#         • Ctrl + Right Click  → place BG point  (negative / red)
#         • Shift + Left Drag   → draw bounding box
#
#   2.  **Coloured annotations** painted on a transparent overlay widget:
#         • Green filled circles for FG points  (with + symbol)
#         • Red   filled circles for BG points  (with – symbol)
#         • Cyan  dashed rectangle for bounding box
#         • Numeric labels next to each point
#
# The overlay is installed once (via ``install()``) and from then on
# repaints automatically whenever the viewer refreshes.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from typing import Optional, Tuple

try:
    from PySide6 import QtWidgets, QtCore, QtGui   # Nuke 16+
except ImportError:
    from PySide2 import QtWidgets, QtCore, QtGui   # Nuke 14-15

# ── Enum compatibility (PySide6 strict-enum vs PySide2 flat enum) ──
try:
    from PySide6.QtCore import Qt
    _ControlMod      = Qt.KeyboardModifier.ControlModifier
    _ShiftMod         = Qt.KeyboardModifier.ShiftModifier
    _LeftBtn          = Qt.MouseButton.LeftButton
    _RightBtn         = Qt.MouseButton.RightButton
    _NoPen            = Qt.PenStyle.NoPen
    _DashLine         = Qt.PenStyle.DashLine
    _DashDotLine      = Qt.PenStyle.DashDotLine
    _WA_TransMouse    = Qt.WidgetAttribute.WA_TransparentForMouseEvents
    _WA_TransBG       = Qt.WidgetAttribute.WA_TranslucentBackground
    _Antialiasing     = QtGui.QPainter.RenderHint.Antialiasing
    _MousePress       = QtCore.QEvent.Type.MouseButtonPress
    _MouseRelease     = QtCore.QEvent.Type.MouseButtonRelease
    _MouseMove        = QtCore.QEvent.Type.MouseMove
except (ImportError, AttributeError):
    from PySide2.QtCore import Qt                       # type: ignore[assignment]
    _ControlMod      = Qt.ControlModifier               # type: ignore[attr-defined]
    _ShiftMod         = Qt.ShiftModifier                 # type: ignore[attr-defined]
    _LeftBtn          = Qt.LeftButton                    # type: ignore[attr-defined]
    _RightBtn         = Qt.RightButton                   # type: ignore[attr-defined]
    _NoPen            = Qt.NoPen                         # type: ignore[attr-defined]
    _DashLine         = Qt.DashLine                      # type: ignore[attr-defined]
    _DashDotLine      = Qt.DashDotLine                   # type: ignore[attr-defined]
    _WA_TransMouse    = Qt.WA_TransparentForMouseEvents  # type: ignore[attr-defined]
    _WA_TransBG       = Qt.WA_TranslucentBackground      # type: ignore[attr-defined]
    _Antialiasing     = QtGui.QPainter.Antialiasing      # type: ignore[attr-defined]
    _MousePress       = QtCore.QEvent.MouseButtonPress   # type: ignore[attr-defined]
    _MouseRelease     = QtCore.QEvent.MouseButtonRelease # type: ignore[attr-defined]
    _MouseMove        = QtCore.QEvent.MouseMove          # type: ignore[attr-defined]

import nuke

# ── Module-level state ──────────────────────────────────────────────
_installed: bool = False
_event_filter: Optional["_ViewerEventFilter"] = None
_overlay_widget: Optional["_OverlayWidget"] = None
_viewer_widget: Optional[QtWidgets.QWidget] = None
_retry_timer: Optional[QtCore.QTimer] = None
_retry_count: int = 0


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _event_xy(event) -> Tuple[float, float]:
    """Get (x, y) from a QMouseEvent — works in both PySide2 & 6."""
    try:
        p = event.position()          # PySide6 / Qt 6
    except AttributeError:
        p = event.pos()               # PySide2 / Qt 5
    return float(p.x()), float(p.y())


def _widget_to_image(widget: QtWidgets.QWidget,
                     wx: float, wy: float
                     ) -> Tuple[Optional[float], Optional[float]]:
    """Convert widget pixel coords → Nuke image coords.

    Uses ``nuke.zoom()`` (pixels-per-unit) and ``nuke.center()``
    (image-space centre of the viewport).  Y is flipped because
    Qt  has Y-down, Nuke has Y-up.
    """
    try:
        z = nuke.zoom()
        cx, cy = nuke.center()
        w, h = widget.width(), widget.height()
        ix = cx + (wx - w / 2.0) / z
        iy = cy + (h / 2.0 - wy) / z
        return ix, iy
    except Exception:
        return None, None


def _image_to_widget(overlay: QtWidgets.QWidget,
                     ix: float, iy: float
                     ) -> Tuple[float, float]:
    """Image coords → overlay widget pixel coords."""
    try:
        z = nuke.zoom()
        cx, cy = nuke.center()
        w, h = overlay.width(), overlay.height()
        wx = (ix - cx) * z + w / 2.0
        wy = h / 2.0 - (iy - cy) * z
        return wx, wy
    except Exception:
        return 0.0, 0.0


def _get_h2_node() -> Optional[nuke.Node]:
    """Return the first selected H2_SamViT node, or *None*."""
    try:
        for n in nuke.selectedNodes():
            if n.knob("model_family") and n.knob("enable_edit"):
                return n
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────
#  Event filter (installed on the viewer widget)
# ─────────────────────────────────────────────────────────────────────

class _ViewerEventFilter(QtCore.QObject):
    """Intercepts Ctrl+Click (points) and Shift+Drag (bbox) on the
    Nuke viewer, forwarding everything else untouched.
    """

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._dragging: bool = False
        self._drag_start: Optional[Tuple[float, float]] = None

    # ── core ──────────────────────────────────────────────────────

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        node = _get_h2_node()
        if node is None:
            return False

        ek = node.knob("enable_edit")
        if not ek or not ek.value():
            return False

        etype = event.type()

        # ── Mouse Press ───────────────────────────────────────────
        if etype == _MousePress:
            mods = event.modifiers()
            btn  = event.button()
            ex, ey = _event_xy(event)

            # Ctrl + Left → FG point
            if mods == _ControlMod and btn == _LeftBtn:
                ix, iy = _widget_to_image(obj, ex, ey)
                if ix is not None:
                    from H2_SamViT_Gizmo import callbacks
                    callbacks.add_point(node, ix, iy, is_foreground=True)
                    _refresh()
                    print(f"[H2 SamViT] FG point at ({ix:.0f}, {iy:.0f})")
                return True

            # Ctrl + Right → BG point
            if mods == _ControlMod and btn == _RightBtn:
                ix, iy = _widget_to_image(obj, ex, ey)
                if ix is not None:
                    from H2_SamViT_Gizmo import callbacks
                    callbacks.add_point(node, ix, iy, is_foreground=False)
                    _refresh()
                    print(f"[H2 SamViT] BG point at ({ix:.0f}, {iy:.0f})")
                return True

            # Shift + Left → start bbox drag
            if mods == _ShiftMod and btn == _LeftBtn:
                self._dragging = True
                self._drag_start = (ex, ey)
                return True

        # ── Mouse Move (only during Shift-drag) ──────────────────
        if etype == _MouseMove and self._dragging:
            ex, ey = _event_xy(event)
            if _overlay_widget and self._drag_start:
                _overlay_widget.set_temp_bbox(self._drag_start, (ex, ey))
                _overlay_widget.update()
            return True

        # ── Mouse Release (finish bbox) ──────────────────────────
        if etype == _MouseRelease and self._dragging:
            ex, ey = _event_xy(event)
            if self._drag_start:
                ix1, iy1 = _widget_to_image(obj, *self._drag_start)
                ix2, iy2 = _widget_to_image(obj, ex, ey)
                if ix1 is not None and ix2 is not None:
                    from H2_SamViT_Gizmo import callbacks
                    callbacks.set_bbox(node, ix1, iy1, ix2, iy2)
                    bk = node.knob("bbox_enabled")
                    if bk:
                        bk.setValue(1)
                    print(f"[H2 SamViT] Bbox ({ix1:.0f},{iy1:.0f})"
                          f" \u2192 ({ix2:.0f},{iy2:.0f})")
            self._dragging = False
            self._drag_start = None
            if _overlay_widget:
                _overlay_widget.clear_temp_bbox()
                _overlay_widget.update()
            return True

        return False                       # pass all other events through


# ─────────────────────────────────────────────────────────────────────
#  Overlay widget (transparent, drawn on top of the viewer)
# ─────────────────────────────────────────────────────────────────────

class _OverlayWidget(QtWidgets.QWidget):
    """Transparent painting surface for coloured point / bbox annotations.

    Mouse events pass straight through to the viewer underneath
    (``WA_TransparentForMouseEvents``).
    """

    def __init__(self, viewer: QtWidgets.QWidget) -> None:
        super().__init__(viewer)
        self._viewer = viewer
        self._temp_bbox: Optional[Tuple[Tuple[float, float],
                                        Tuple[float, float]]] = None

        self.setAttribute(_WA_TransMouse, True)
        self.setAttribute(_WA_TransBG, True)
        self.setStyleSheet("background: transparent;")
        self.setGeometry(0, 0, viewer.width(), viewer.height())

        # Default colours (overridden from node knobs in paint)
        self._fg_col  = QtGui.QColor(0, 255, 0, 200)
        self._bg_col  = QtGui.QColor(255, 0, 0, 200)
        self._box_col = QtGui.QColor(0, 167, 255, 200)

        # Periodic sync (geometry + repaint)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(80)           # ~12 fps

        self.show()
        self.raise_()

    # ── public helpers ────────────────────────────────────────────

    def set_temp_bbox(self,
                      start: Tuple[float, float],
                      current: Tuple[float, float]) -> None:
        self._temp_bbox = (start, current)

    def clear_temp_bbox(self) -> None:
        self._temp_bbox = None

    # ── internal ──────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._viewer:
            vw, vh = self._viewer.width(), self._viewer.height()
            if self.width() != vw or self.height() != vh:
                self.setGeometry(0, 0, vw, vh)
        self.raise_()
        self.update()

    def _read_colours(self, node: nuke.Node) -> None:
        """Update internal colour cache from the node's knobs."""
        def _kc(name: str, default: QtGui.QColor) -> QtGui.QColor:
            k = node.knob(name)
            if not k:
                return default
            v = k.value()
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                r, g, b = [min(255, max(0, int(c * 255))) for c in v[:3]]
                a = int(v[3] * 255) if len(v) >= 4 else 200
                return QtGui.QColor(r, g, b, a)
            return default

        self._fg_col  = _kc("fg_point_color", QtGui.QColor(0, 255, 0, 200))
        self._bg_col  = _kc("bg_point_color", QtGui.QColor(255, 0, 0, 200))
        self._box_col = _kc("bbox_color",     QtGui.QColor(0, 167, 255, 200))

    # ── painting ──────────────────────────────────────────────────

    def paintEvent(self, event) -> None:   # noqa: N802
        node = _get_h2_node()
        if node is None:
            return

        sk = node.knob("show_ui_overlays")
        if sk and not sk.value():
            return

        self._read_colours(node)

        scale = 1.0
        sk2 = node.knob("overlay_scale")
        if sk2:
            try:
                scale = float(sk2.value())
            except Exception:
                pass

        painter = QtGui.QPainter(self)
        painter.setRenderHint(_Antialiasing)

        self._paint_bbox(painter, node, scale)
        self._paint_temp_bbox(painter, scale)
        self._paint_points(painter, node, scale)

        painter.end()

    # ── points ────────────────────────────────────────────────────

    def _paint_points(self, p: QtGui.QPainter,
                      node: nuke.Node, scale: float) -> None:
        from H2_SamViT_Gizmo import callbacks
        points = callbacks.get_enabled_points(node)

        r = 7 * scale
        show_labels = True
        lk = node.knob("show_point_labels")
        if lk:
            show_labels = bool(lk.value())

        for pt in points:
            wx, wy = _image_to_widget(self, pt["x"], pt["y"])
            colour = self._fg_col if pt["is_foreground"] else self._bg_col

            # Filled circle with white border
            p.setBrush(QtGui.QBrush(colour))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220),
                                1.5 * scale))
            p.drawEllipse(QtCore.QPointF(wx, wy), r, r)

            # +/\u2013 symbol
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 255),
                                max(1.0, 2 * scale)))
            half = r * 0.5
            p.drawLine(QtCore.QPointF(wx - half, wy),
                       QtCore.QPointF(wx + half, wy))
            if pt["is_foreground"]:
                p.drawLine(QtCore.QPointF(wx, wy - half),
                           QtCore.QPointF(wx, wy + half))

            # Label
            if show_labels:
                font = p.font()
                font.setPixelSize(max(10, int(11 * scale)))
                font.setBold(True)
                p.setFont(font)
                p.setPen(QtGui.QColor(255, 255, 255, 220))
                p.drawText(int(wx + r + 3), int(wy - r + 3),
                           str(pt["index"]))

    # ── bounding box ──────────────────────────────────────────────

    def _paint_bbox(self, p: QtGui.QPainter,
                    node: nuke.Node, scale: float) -> None:
        from H2_SamViT_Gizmo import callbacks
        bbox = callbacks.get_bbox(node)
        if not bbox:
            return

        x1, y1, x2, y2 = bbox
        wx1, wy1 = _image_to_widget(self, x1, y1)
        wx2, wy2 = _image_to_widget(self, x2, y2)

        pen = QtGui.QPen(self._box_col, 2 * scale)
        pen.setStyle(_DashLine)
        p.setPen(pen)
        fill = QtGui.QColor(self._box_col)
        fill.setAlpha(30)
        p.setBrush(QtGui.QBrush(fill))

        rect = QtCore.QRectF(
            min(wx1, wx2), min(wy1, wy2),
            abs(wx2 - wx1), abs(wy2 - wy1),
        )
        p.drawRect(rect)

        # Corner handles
        hs = 5 * scale
        p.setBrush(QtGui.QBrush(self._box_col))
        p.setPen(_NoPen)
        for cx, cy in [(wx1, wy1), (wx2, wy1), (wx2, wy2), (wx1, wy2)]:
            p.drawRect(QtCore.QRectF(cx - hs, cy - hs, hs * 2, hs * 2))

    # ── temp bbox (while Shift-dragging) ──────────────────────────

    def _paint_temp_bbox(self, p: QtGui.QPainter, scale: float) -> None:
        if self._temp_bbox is None:
            return
        (sx, sy), (cx, cy) = self._temp_bbox

        pen = QtGui.QPen(self._box_col, 2 * scale)
        pen.setStyle(_DashDotLine)
        p.setPen(pen)
        fill = QtGui.QColor(self._box_col)
        fill.setAlpha(40)
        p.setBrush(QtGui.QBrush(fill))

        rect = QtCore.QRectF(
            min(sx, cx), min(sy, cy),
            abs(cx - sx), abs(cy - sy),
        )
        p.drawRect(rect)


# ─────────────────────────────────────────────────────────────────────
#  Viewer-widget finder
# ─────────────────────────────────────────────────────────────────────

def _find_viewer_widget() -> Optional[QtWidgets.QWidget]:
    """Heuristically locate Nuke's main image-viewer GL widget.

    Searches the Qt widget hierarchy for a large, visible widget whose
    class name suggests it is an OpenGL viewport (common names across
    Nuke versions: ``GLViewer``, ``QOpenGLWidget``, ``Viewport``,
    various ``Foundry::`` prefixed classes).
    """
    app = QtWidgets.QApplication.instance()
    if not app:
        return None

    best: Optional[QtWidgets.QWidget] = None
    best_score = 0

    for w in app.allWidgets():
        if not isinstance(w, QtWidgets.QWidget):
            continue
        if not w.isVisible() or w.width() < 200 or w.height() < 200:
            continue

        cn = w.metaObject().className()
        on = w.objectName().lower()

        # Skip standard container widgets
        if isinstance(w, (QtWidgets.QMainWindow, QtWidgets.QMenuBar,
                          QtWidgets.QStatusBar, QtWidgets.QToolBar,
                          QtWidgets.QDockWidget, QtWidgets.QSplitter,
                          QtWidgets.QStackedWidget, QtWidgets.QTabWidget,
                          QtWidgets.QTabBar, QtWidgets.QScrollArea)):
            continue

        # Skip DAG / properties / script editor
        if any(k in on for k in ('dag', 'node', 'script', 'propert',
                                  'curve', 'dope')):
            continue
        if any(k in cn for k in ('DAG', 'NodeGraph', 'ScriptEditor',
                                  'CurveEditor', 'DopeSheet')):
            continue

        area = w.width() * w.height()
        score = area

        # Boost score for class-name hints
        if any(k in cn for k in ('GL', 'OpenGL', 'Viewport', 'Viewer')):
            score *= 100

        if score > best_score:
            best_score = score
            best = w

    return best


# ─────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────

def _refresh() -> None:
    """Trigger an immediate repaint of the overlay."""
    if _overlay_widget:
        _overlay_widget.update()


def install() -> bool:
    """Install the viewer event-filter and overlay widget.

    Safe to call multiple times — subsequent calls are no-ops.
    Returns *True* on success.
    """
    global _installed, _event_filter, _overlay_widget, _viewer_widget

    if _installed:
        return True

    viewer = _find_viewer_widget()
    if not viewer:
        _schedule_retry()
        return False

    _event_filter  = _ViewerEventFilter()
    _overlay_widget = _OverlayWidget(viewer)
    viewer.installEventFilter(_event_filter)

    _viewer_widget = viewer
    _installed = True
    print("[H2 SamViT] Viewer overlay installed  "
          "(Ctrl+LClick = FG,  Ctrl+RClick = BG,  Shift+Drag = Bbox)")
    return True


def _schedule_retry() -> None:
    """Retry ``install()`` after a short delay (viewer may not exist yet)."""
    global _retry_timer, _retry_count
    if _retry_count >= 10:
        return                      # give up after 10 attempts
    if _retry_timer is not None:
        return                      # already scheduled
    _retry_timer = QtCore.QTimer()
    _retry_timer.setSingleShot(True)
    _retry_timer.timeout.connect(_do_retry)
    _retry_timer.start(2000)


def _do_retry() -> None:
    global _retry_timer, _retry_count
    _retry_timer = None
    _retry_count += 1
    if not install():
        _schedule_retry()


def uninstall() -> None:
    """Remove the overlay and event-filter."""
    global _installed, _event_filter, _overlay_widget, _viewer_widget
    if not _installed:
        return
    try:
        if _viewer_widget and _event_filter:
            _viewer_widget.removeEventFilter(_event_filter)
    except Exception:
        pass
    try:
        if _overlay_widget:
            _overlay_widget.hide()
            _overlay_widget.deleteLater()
    except Exception:
        pass
    _event_filter   = None
    _overlay_widget = None
    _viewer_widget  = None
    _installed      = False


def ensure_installed() -> None:
    """Try to install if not already done (called from callbacks)."""
    if not _installed:
        install()
