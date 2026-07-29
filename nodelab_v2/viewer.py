"""Viewer panel (G4 + Phase-6 navigation) — renders the pulled Dataset of the viewed
node as a multi-channel colour composite, with smooth M/T/Z sliders, per-axis
play/pause + fps, and per-channel toggle buttons tinted by emission colour.

Rendering stays in the runner's worker thread (:func:`nodelab_v2.runner.render_plane`
per channel); this panel only *composites* the ready per-channel float planes into an
RGB ``QImage`` — each active channel scaled by percentile auto-contrast and multiplied
by the colour closest to its emission spectrum (:func:`nodelab_v2.theme.emission_qcolor`),
additively blended. Playback advances one axis frame-by-frame, self-throttled to the
target fps and reporting the achieved fps.

**Overlays** (Points / Labels / Tracks / Mesh, and a reserved tab for Voxels) are owned by
:mod:`nodelab_v2.overlays`: this panel extracts the geometry for the viewed ``(m,t,z,c)``
and hands it, plus the backend's ``plane_to_widget`` mapping, to a single
:class:`~nodelab_v2.overlays.OverlayRenderer`. Both image backends paint them the same
way — **in widget space, sized in screen pixels** — so an outline keeps its thickness
while you zoom into a label instead of being magnified with the image. The look is
configured in the *Overlays* popup (:mod:`nodelab_v2.overlay_dialog`).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor, QImage, QPainter, QPen, QPixmap, QPolygonF, QTransform)
from PySide6.QtWidgets import (
    QDoubleSpinBox, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QToolButton,
    QVBoxLayout, QWidget,
)

from nodegraph.domains import Domain
from nodegraph.mesh import mesh_part as MESH_PART
from nodelab_v2 import overlays as OV
from nodelab_v2 import theme as T
from nodelab_v2.minimap import ElidedLabel

_AXES = ("m", "t", "z")           # the playable/sliderable spatial-series axes (C = toggles)

#: full ↔ mini-map text for the Overlays button: in the mini-map the whole control strip
#: has to fit in ~200 px, so it shrinks to the glyph alone (the tooltip carries the name).
_OVL_LABEL = ("◈ Overlays", "◈")


def _autocontrast(plane: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    """A float plane percentile-normalized to [0, 1] (hot-pixel-safe)."""
    a = np.asarray(plane, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0.0, 1.0)


def plane_to_qimage(plane: np.ndarray, *, lo_pct: float = 1.0,
                    hi_pct: float = 99.5) -> QImage:
    """Auto-contrast grayscale-8 QImage from a float plane (kept for callers/tests)."""
    a = np.asarray(plane, dtype=float)
    if a.size == 0:
        return QImage(1, 1, QImage.Format_Grayscale8)
    u8 = (_autocontrast(a, lo_pct, hi_pct) * 255.0).astype(np.uint8)
    u8 = np.ascontiguousarray(u8)
    h, w = u8.shape
    return QImage(u8.data, w, h, w, QImage.Format_Grayscale8).copy()


def composite_to_qimage(planes: Dict[int, np.ndarray], colors: Dict[int, Tuple[int, int, int]],
                        *, lo_pct: float = 1.0, hi_pct: float = 99.5) -> QImage:
    """Additively blend the active per-channel planes into an RGB QImage — each channel
    auto-contrasted then multiplied by its (r,g,b) emission colour."""
    if not planes:
        return QImage(1, 1, QImage.Format_RGB888)
    shape = next(iter(planes.values())).shape
    rgb = np.zeros((shape[0], shape[1], 3), dtype=float)
    for ch, plane in planes.items():
        if plane.shape != shape:
            continue
        norm = _autocontrast(plane, lo_pct, hi_pct)
        r, g, b = colors.get(ch, (255, 255, 255))
        rgb[..., 0] += norm * (r / 255.0)
        rgb[..., 1] += norm * (g / 255.0)
        rgb[..., 2] += norm * (b / 255.0)
    u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    u8 = np.ascontiguousarray(u8)
    h, w, _ = u8.shape
    return QImage(u8.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def composite_with_clim(planes: Dict[int, np.ndarray],
                        colors: Dict[int, Tuple[int, int, int]],
                        clims: Dict[int, Tuple[float, float]],
                        gammas: Optional[Dict[int, float]] = None) -> QImage:
    """Like :func:`composite_to_qimage` but uses **precomputed** ``(lo, hi)`` intensity
    bounds per channel (contrast computed once per volume and cached, the biggest
    per-frame CPU saving on the fallback path) plus an optional per-channel ``gamma``
    (transfer ``norm**gamma``) — the CPU mirror of the GPU shader's LUT."""
    if not planes:
        return QImage(1, 1, QImage.Format_RGB888)
    gammas = gammas or {}
    shape = next(iter(planes.values())).shape
    rgb = np.zeros((shape[0], shape[1], 3), dtype=float)
    for ch, plane in planes.items():
        if plane.shape != shape:
            continue
        a = np.asarray(plane, dtype=float)
        lohi = clims.get(ch)
        if lohi is None:
            norm = _autocontrast(a, 1.0, 99.5)
        else:
            lo, hi = lohi
            if hi <= lo:
                hi = lo + 1.0
            norm = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
        gm = float(gammas.get(ch, 1.0))
        if abs(gm - 1.0) > 1e-3:
            norm = np.power(norm, max(gm, 1e-3))
        r, g, b = colors.get(ch, (255, 255, 255))
        rgb[..., 0] += norm * (r / 255.0)
        rgb[..., 1] += norm * (g / 255.0)
        rgb[..., 2] += norm * (b / 255.0)
    u8 = np.ascontiguousarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    h, w, _ = u8.shape
    return QImage(u8.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def _text_on(col) -> str:
    """Black or white, whichever contrasts with ``col`` (relative luminance)."""
    lum = 0.299 * col.red() + 0.587 * col.green() + 0.114 * col.blue()
    return "#0b0e12" if lum > 150 else "#f0f3f7"


class _ImageView(QGraphicsView):
    """The image surface — a single pixmap item on a graphics scene. Scroll-wheel
    zooms toward the cursor; left-drag pans (``ScrollHandDrag``, on by default, exactly
    like the node canvas); scrollbars stay hidden (the drag is the way to move, and they
    would eat scarce room in the mini-map).

    Framing: the view auto-fits (keeping aspect ratio) when a NEW image SIZE arrives and
    whenever the widget is resized — so re-homing the panel into the mini-map re-frames
    the image instead of cropping it. Once the user has zoomed or panned, that framing is
    theirs: scrubbing and resizing both leave it alone until a double-click re-fits.

    Overlays are painted through :attr:`overlay_cb` in :meth:`drawForeground` with the
    world transform reset, i.e. **in viewport pixels** — mirroring
    :class:`~nodelab_v2.glview.GLImageView` exactly, so one renderer serves both backends
    and overlay line widths / glyph sizes no longer scale with the zoom. That also means
    the whole viewport must repaint on a pan (``FullViewportUpdate``): a minimal update
    would only refresh the scrolled band while the overlay covers everything."""

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)          # pan on by default
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setMinimumSize(200, 200)
        self._last_size = None
        self._need_fit = False
        self._user_view = False          # the user zoomed/panned → stop auto-framing
        #: the panel sets this to paint overlays over the image: cb(painter)
        self.overlay_cb = None

    # ── overlay surface (mirrors GLImageView's API) ─────────────────────────────
    def plane_to_widget(self, px: float, py: float) -> QPointF:
        """Map a plane-space pixel to viewport coords. The pixmap item sits at the scene
        origin at 1:1, so scene units *are* displayed-plane pixels."""
        return self.viewportTransform().map(QPointF(px, py))

    def refresh(self) -> None:
        self.viewport().update()

    def drawForeground(self, p: QPainter, rect: QRectF) -> None:
        super().drawForeground(p, rect)
        if self.overlay_cb is None:
            return
        p.save()
        p.setTransform(QTransform())          # → viewport pixels, like the GL path
        try:
            self.overlay_cb(p)
        except Exception:                     # noqa: BLE001 — overlays are non-fatal
            pass
        p.restore()

    def set_pixmap(self, pix: QPixmap) -> None:
        changed = self._last_size != pix.size()
        self._item.setPixmap(pix)
        if not pix.isNull():
            self._scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        self._last_size = pix.size()
        if changed:
            self._need_fit = True
            self._user_view = False      # a differently sized image re-frames anyway
            self._maybe_fit()

    def fit(self) -> None:
        self._need_fit = True
        self._user_view = False
        self._maybe_fit()

    def _maybe_fit(self) -> None:
        if (self._need_fit and self.width() > 2 and self.height() > 2
                and not self._item.pixmap().isNull()):
            self.fitInView(self._item, Qt.KeepAspectRatio)
            self._need_fit = False

    def wheelEvent(self, e) -> None:
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self._user_view = True
        self.scale(f, f)

    def mouseMoveEvent(self, e) -> None:
        if e.buttons() & Qt.LeftButton:
            self._user_view = True       # a hand-drag pan is the user's framing now
        super().mouseMoveEvent(e)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if not self._user_view:
            self._need_fit = True        # docking ↔ mini-map re-frames, never crops
        self._maybe_fit()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._maybe_fit()

    def mouseDoubleClickEvent(self, e) -> None:
        self.fit()                         # double-click resets the framing
        e.accept()


class HistogramLUT(QWidget):
    """A compact intensity-histogram with two draggable handles — the black point (lo)
    and white point (hi) of the LUT window. Dragging a handle (or the shaded band between
    them) emits :data:`window_changed`; the transfer-function ramp is drawn across the
    window so the mapping is legible. The histogram is log-scaled (microscopy is heavily
    skewed toward the low end).

    Three ranges, kept distinct: the **data range** (``_rmin/_rmax``, the full bit-depth
    extent — handles clamp here), the **view range** (``_vmin/_vmax``, what's drawn — the
    mouse wheel zooms it, and :meth:`fit_view_to_window` snaps it to the handles), and the
    **window** (``_lo/_hi``, the LUT). Zooming the view lets you place the handles precisely
    inside a dim part of the histogram without changing the LUT."""

    window_changed = Signal(float, float)      # (lo, hi) in data units
    gamma_changed = Signal(float)              # transfer-function gamma

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(56)
        self.setMinimumWidth(160)
        self.setCursor(Qt.SizeHorCursor)
        self.setToolTip("Drag the handles to set the LUT · drag the middle dot up/down "
                        "for gamma · wheel to zoom · double-click to reset the zoom")
        self._rmin = 0.0
        self._rmax = 1.0
        self._vmin = 0.0                            # view (zoom) range
        self._vmax = 1.0
        self._lo = 0.0
        self._hi = 1.0
        self._gamma = 1.0
        self._vals: Optional[np.ndarray] = None     # cached samples (re-binned on zoom)
        self._hist: Optional[np.ndarray] = None     # log-normalized bar heights [0,1]
        self._tint = (120, 170, 255)
        self._drag: Optional[str] = None            # 'lo' | 'hi' | 'band'
        self._drag_x0 = 0.0
        self._drag_lo0 = 0.0
        self._drag_hi0 = 0.0

    def set_range(self, rmin: float, rmax: float) -> None:
        """Set the full data range. A genuinely new range resets the zoom to full; the
        same range (a new frame of the same channel) keeps the user's current zoom."""
        rmin = float(rmin)
        rmax = float(rmax) if rmax > rmin else rmin + 1.0
        if (rmin, rmax) != (self._rmin, self._rmax):
            self._rmin, self._rmax = rmin, rmax
            self._vmin, self._vmax = rmin, rmax
        self.update()

    def set_window(self, lo: float, hi: float) -> None:
        self._lo, self._hi = float(lo), float(hi)
        self.update()

    def set_gamma(self, gamma: float) -> None:
        self._gamma = max(1e-2, float(gamma))
        self.update()

    def set_tint(self, rgb: Tuple[int, int, int]) -> None:
        self._tint = rgb
        self.update()

    def set_values(self, values: np.ndarray) -> None:
        """Cache a channel's samples (caller subsamples) and (re)bin over the view range."""
        a = np.asarray(values, dtype=float).ravel()
        self._vals = a[np.isfinite(a)]
        self._recompute_hist()

    def _recompute_hist(self, bins: int = 256) -> None:
        if self._vals is None or self._vals.size == 0:
            self._hist = None
        else:
            counts, _ = np.histogram(self._vals, bins=bins, range=(self._vmin, self._vmax))
            h = np.log1p(counts.astype(float))
            m = float(h.max())
            self._hist = (h / m) if m > 0 else None
        self.update()

    def fit_view_to_window(self) -> None:
        """Zoom the histogram to the current handles (with a little padding) so you can
        fine-tune the window inside a narrow region."""
        lo, hi = min(self._lo, self._hi), max(self._lo, self._hi)
        pad = max((hi - lo) * 0.15, (self._rmax - self._rmin) * 1e-3)
        self._vmin = max(self._rmin, lo - pad)
        self._vmax = min(self._rmax, hi + pad)
        if self._vmax - self._vmin < 1e-9:
            self._vmin, self._vmax = self._rmin, self._rmax
        self._recompute_hist()

    def reset_view(self) -> None:
        self._vmin, self._vmax = self._rmin, self._rmax
        self._recompute_hist()

    # ── value ↔ pixel (in the current view range) ────────────────────────────────
    def _v2x(self, v: float) -> float:
        return (v - self._vmin) / (self._vmax - self._vmin) * self.width()

    def _x2v(self, x: float) -> float:
        return self._vmin + max(0.0, min(1.0, x / max(1, self.width()))) * (
            self._vmax - self._vmin)

    def wheelEvent(self, e) -> None:
        """Zoom the view range around the value under the cursor."""
        factor = 0.8 if e.angleDelta().y() > 0 else 1.25
        center = self._x2v(e.position().x())
        span = (self._vmax - self._vmin) * factor
        full = self._rmax - self._rmin
        span = max(min(span, full), full * 1e-3)   # clamp: never below 0.1% of full
        frac = (center - self._vmin) / max(self._vmax - self._vmin, 1e-9)
        vmin = center - frac * span
        vmax = vmin + span
        if vmin < self._rmin:
            vmin, vmax = self._rmin, self._rmin + span
        if vmax > self._rmax:
            vmax, vmin = self._rmax, self._rmax - span
        self._vmin, self._vmax = max(self._rmin, vmin), min(self._rmax, vmax)
        self._recompute_hist()
        e.accept()

    # ── paint ────────────────────────────────────────────────────────────────────
    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, T.BODY)
        r, g, b = self._tint
        # histogram bars
        if self._hist is not None and self._hist.size:
            n = self._hist.size
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(r, g, b, 130))
            bw = w / n
            for i, hv in enumerate(self._hist.tolist()):
                bh = hv * (h - 2)
                p.drawRect(QRectF(i * bw, h - bh, bw + 0.6, bh))
        # window band
        xlo, xhi = self._v2x(self._lo), self._v2x(self._hi)
        p.fillRect(QRectF(xlo, 0, max(1.0, xhi - xlo), h), QColor(255, 255, 255, 22))
        # transfer function t = clamp(...)**gamma across the window (a curve when gamma≠1)
        top, bot = 1.0, h - 1.0
        p.setPen(QPen(QColor(r, g, b), 1.6))
        curve = []
        steps = 40
        for k in range(steps + 1):
            tt = k / steps
            out = tt ** self._gamma
            curve.append(QPointF(xlo + tt * (xhi - xlo), bot - out * (bot - top)))
        p.drawPolyline(QPolygonF(curve))
        # handle lines
        p.setPen(QPen(T.INK, 1.5))
        p.drawLine(QPointF(xlo, 0), QPointF(xlo, h))
        p.drawLine(QPointF(xhi, 0), QPointF(xhi, h))
        p.setBrush(T.INK)
        p.drawEllipse(QPointF(xlo, h - 3), 3.0, 3.0)
        p.drawEllipse(QPointF(xhi, 3), 3.0, 3.0)
        # midpoint (gamma) dot — drag up/down to reshape the curve
        mx, my = self._gamma_dot()
        p.setPen(QPen(T.INK, 1.0))
        p.setBrush(QColor(r, g, b))
        p.drawEllipse(QPointF(mx, my), 4.5, 4.5)
        p.end()

    def _gamma_dot(self) -> Tuple[float, float]:
        """Widget-space (x, y) of the midpoint dot: at the window centre, height = the
        transfer output there (0.5**gamma)."""
        xlo, xhi = self._v2x(self._lo), self._v2x(self._hi)
        bot, top = self.height() - 1.0, 1.0
        return (xlo + 0.5 * (xhi - xlo), bot - (0.5 ** self._gamma) * (bot - top))

    # ── interaction ──────────────────────────────────────────────────────────────
    def mousePressEvent(self, e) -> None:
        x, y = e.position().x(), e.position().y()
        xlo, xhi = self._v2x(self._lo), self._v2x(self._hi)
        mx, my = self._gamma_dot()
        if (x - mx) ** 2 + (y - my) ** 2 <= 8 ** 2:   # near the gamma dot
            self._drag = "gamma"
        elif abs(x - xlo) <= 6:
            self._drag = "lo"
        elif abs(x - xhi) <= 6:
            self._drag = "hi"
        elif xlo < x < xhi:
            self._drag = "band"
        else:
            self._drag = "lo" if abs(x - xlo) < abs(x - xhi) else "hi"
        self._drag_x0 = x
        self._drag_lo0, self._drag_hi0 = self._lo, self._hi
        self._apply_drag(x, y)

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None:
            self._apply_drag(e.position().x(), e.position().y())

    def mouseReleaseEvent(self, e) -> None:
        self._drag = None

    def mouseDoubleClickEvent(self, e) -> None:
        self.reset_view()                          # double-click resets the zoom to full
        e.accept()

    def _apply_drag(self, x: float, y: float) -> None:
        eps = (self._vmax - self._vmin) * 1e-3     # finer when zoomed in
        if self._drag == "gamma":
            # midpoint output m = 0.5**gamma → gamma = log(m)/log(0.5)
            bot = self.height() - 1.0
            m = min(0.98, max(0.02, (bot - y) / max(bot, 1.0)))
            self._gamma = min(10.0, max(0.1, np.log(m) / np.log(0.5)))
            self.update()
            self.gamma_changed.emit(self._gamma)
            return
        if self._drag == "band":
            dv = self._x2v(x) - self._x2v(self._drag_x0)
            span = self._drag_hi0 - self._drag_lo0
            lo = min(max(self._rmin, self._drag_lo0 + dv), self._rmax - span)
            self._lo, self._hi = lo, lo + span
        elif self._drag == "lo":
            self._lo = min(self._x2v(x), self._hi - eps)
        elif self._drag == "hi":
            self._hi = max(self._x2v(x), self._lo + eps)
        else:
            return
        self.update()
        self.window_changed.emit(self._lo, self._hi)


class ViewerPanel(QWidget):
    """Shows the viewed node's colour composite; emits ``request_changed`` when the
    coords (m,t,z) or the active channel set move, so the window re-pulls."""

    request_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 4)
        v.setSpacing(4)

        # No header row — the image fills the top of the panel; the viewed-node name +
        # readout live in the status strip, and the overlay controls collapsed into ONE
        # button on the controls strip (built below) that opens the Overlays popup.
        #
        # Overlay state: the settings (loaded from the built-in defaults, then the project
        # file, then this machine's), the renderer that paints them, and a revision that
        # invalidates the per-frame geometry cache whenever they change.
        self.overlays, self._ovl_sources = OV.load_defaults()
        self._renderer = OV.OverlayRenderer()
        self._ovl_dialog = None
        self._ovl_rev = 0
        self._geo_key: Optional[tuple] = None
        self._geo_points: List[OV.PointMark] = []
        self._geo_tracks: List[OV.TrackPath] = []
        self._geo_lab: Optional[np.ndarray] = None
        self._geo_mesh: List[OV.MeshSection] = []

        # ── image (dominant: scroll-zoom + drag-pan; now fills the reclaimed header) ─
        # Prefer the GPU backend (contrast/colour/compositing in a shader, upload-once);
        # fall back to the CPU QGraphicsView path headless or on a GL failure.
        self._compact = False          # mini-map layout (set_compact) — read on rebuild
        self._gl = None
        self._img_layout = v
        self._view = self._make_view()
        v.addWidget(self._view, 1)

        # ── compact controls, BELOW the image ───────────────────────────────────
        controls = QWidget()
        controls.setProperty("role", "controls")
        self._controls = controls
        cv = QVBoxLayout(controls)
        cv.setContentsMargins(0, 2, 0, 0)
        cv.setSpacing(1)

        self._sliders: Dict[str, QSlider] = {}
        self._val_lbls: Dict[str, QLabel] = {}
        self._play_btns: Dict[str, QToolButton] = {}
        self._fps_spins: Dict[str, QDoubleSpinBox] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)
        grid.setColumnStretch(1, 1)
        for r, ax in enumerate(_AXES):
            name = QLabel(ax.upper())
            name.setProperty("role", "axis")
            sld = QSlider(Qt.Horizontal)
            sld.setRange(0, 0)
            sld.valueChanged.connect(lambda _v, a=ax: self._on_slider(a))
            val = QLabel("0/0")
            val.setProperty("role", "muted")
            val.setMinimumWidth(84)
            play = QToolButton()
            play.setText("▶")
            play.setCheckable(True)
            play.setAutoRaise(True)
            play.setToolTip(f"Play / pause the {ax.upper()} axis")
            play.toggled.connect(lambda on, a=ax: self._on_play(a, on))
            fps = QDoubleSpinBox()
            fps.setRange(0.5, 60.0)
            fps.setValue(8.0)
            fps.setDecimals(0)
            fps.setSingleStep(1.0)
            fps.setSuffix("fps")
            fps.setToolTip("Target playback rate")
            fps.valueChanged.connect(lambda _v, a=ax: self._retarget_fps(a))
            grid.addWidget(name, r, 0)
            grid.addWidget(sld, r, 1)
            grid.addWidget(val, r, 2)
            grid.addWidget(play, r, 3)
            grid.addWidget(fps, r, 4)
            self._sliders[ax] = sld
            self._val_lbls[ax] = val
            self._play_btns[ax] = play
            self._fps_spins[ax] = fps
        cv.addLayout(grid)

        # ── display tools: per-frame Auto contrast, Fit-zoom, overlay toggles ────
        tools = QHBoxLayout()
        tools.setSpacing(6)
        tools.setContentsMargins(0, 0, 0, 0)
        self._lut_auto = QPushButton("Auto")
        self._lut_auto.setCheckable(True)
        self._lut_auto.setCursor(Qt.PointingHandCursor)
        self._lut_auto.setToolTip("Auto-contrast (percentile). When ON it re-applies to "
                                  "EVERY frame while scrubbing or playing.")
        self._lut_auto.toggled.connect(self._on_auto_toggled)
        self._lut_fit = QPushButton("Fit")
        self._lut_fit.setCursor(Qt.PointingHandCursor)
        self._lut_fit.setToolTip("Zoom every channel histogram to its window")
        self._lut_fit.clicked.connect(self._fit_all)
        tools.addWidget(self._lut_auto)
        tools.addWidget(self._lut_fit)
        tools.addStretch(1)
        # ONE overlay control: the popup owns every domain's look (and its on/off), so the
        # strip no longer grows a checkbox per domain as domains are added.
        self._ovl_btn = QPushButton(_OVL_LABEL[0])
        self._ovl_btn.setCursor(Qt.PointingHandCursor)
        self._ovl_btn.setToolTip("Configure the Point / Label / Track / Mesh (and reserved "
                                 "Voxel) overlays — size, opacity, look, colour")
        self._ovl_btn.clicked.connect(self.open_overlay_dialog)
        tools.addWidget(self._ovl_btn)
        self._tools_row = QWidget()
        self._tools_row.setLayout(tools)
        cv.addWidget(self._tools_row)

        # ── per-channel LUT strip: each channel's toggle over its own histogram,
        #    all channels side by side (built in _rebuild_channels). ──────────────
        self._lut_strip = QHBoxLayout()
        self._lut_strip.setSpacing(8)
        self._lut_strip.setContentsMargins(0, 0, 0, 0)
        self._lut_strip_w = QWidget()
        self._lut_strip_w.setLayout(self._lut_strip)
        cv.addWidget(self._lut_strip_w)
        v.addWidget(controls)

        self._chan_btns: Dict[int, QPushButton] = {}
        self._hists: Dict[int, HistogramLUT] = {}
        self._lut_edits: Dict[int, Tuple[QLineEdit, QLineEdit]] = {}
        self._lut_cols: List[QWidget] = []
        self._chan_colors: Dict[int, Tuple[int, int, int]] = {}
        self._active_channels: List[int] = [0]
        self._auto_on = False

        # elided: a QLabel's minimum width is its whole string, which would stop the
        # mini-map from ever shrinking below the length of this status line.
        self._status = ElidedLabel("")
        self._status.setProperty("role", "muted")
        v.addWidget(self._status)

        # state
        self._axes = None
        self._axes_key: Optional[Tuple[int, int, int, int]] = None
        self._planes: Dict[int, np.ndarray] = {}
        self._ref_plane: Optional[np.ndarray] = None
        self._dataset = None
        self._base_pix: Optional[QPixmap] = None
        self._chan_names: List[str] = []
        # contrast: (node_id, channel) → (lo, hi) in native intensity units, computed
        # once per volume (not per frame) — the biggest per-frame CPU saving.
        self._clim: Dict[Tuple[str, int], Tuple[float, float]] = {}
        self._drange: Dict[Tuple[str, int], Tuple[float, float]] = {}   # LUT slider extent
        self._gammas: Dict[Tuple[str, int], float] = {}   # per-channel transfer gamma
        self._bit_depth: Optional[int] = None     # significant bit depth (metadata)
        self._clim_node: Optional[str] = None
        self._node_id: Optional[str] = None
        # playback — a wall-clock QTimer that draws whatever frame is ready and drops
        # frames to hold the target fps (decoupled from decode; napari's frame budget).
        self._playing_axis: Optional[str] = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick_play)
        self._last_frame_t: Optional[float] = None
        self._fps_ema: Optional[float] = None
        self.restyle()

    # ── backend ─────────────────────────────────────────────────────────────────
    def _make_view(self):
        """The image surface. The GPU :class:`~nodelab_v2.glview.GLImageView` — raw 16-bit
        upload once + LUT window / colour / compositing in a fragment shader, so contrast
        (LUT) changes are instantaneous — is the default on a real windowed session. Set
        ``NODELAB_GL=0`` to force the CPU :class:`_ImageView`. Headless platforms
        (probe/CI) and a runtime ``gl_failed`` fall back to the CPU path automatically.

        Both surfaces expose the same overlay contract — ``overlay_cb`` (a QPainter in
        widget pixels) + ``plane_to_widget`` + ``refresh`` — so :meth:`_paint_overlays`
        is backend-agnostic."""
        import os
        view = None
        if os.environ.get("NODELAB_GL", "1") not in ("0", "false", "no"):
            try:
                from nodelab_v2.glview import GLImageView, probe_gl_available
                if probe_gl_available():
                    view = GLImageView()
                    view.gl_failed.connect(self._fallback_to_cpu)
                    self._gl = view
            except Exception:                    # noqa: BLE001 — any import/ctor issue → CPU
                self._gl = None
                view = None
        if view is None:
            view = _ImageView()
        view.overlay_cb = self._paint_overlays
        return view

    def _fallback_to_cpu(self) -> None:
        """Runtime GL failure → replace the GL surface with the CPU view. Deferred to the
        next event-loop turn: ``gl_failed`` fires from inside the GL widget's own
        ``paintGL``, and tearing the widget down mid-paint segfaults."""
        if self._gl is None:
            return
        QTimer.singleShot(0, self._do_fallback_to_cpu)

    def _do_fallback_to_cpu(self) -> None:
        if self._gl is None:
            return
        self._gl = None
        old = self._view
        self._view = _ImageView()
        self._view.overlay_cb = self._paint_overlays
        self._img_layout.replaceWidget(old, self._view)
        old.setParent(None)
        old.deleteLater()
        self._apply_image_minimum()          # the fresh surface must honour compact mode
        if self._planes:
            self._display(self._node_id or "", self._planes, self._axes)

    # ── compact (mini-map) mode ────────────────────────────────────────────────
    def set_compact(self, on: bool, *, force: bool = False) -> None:
        """Trim the panel to mini-map size (:mod:`nodelab_v2.minimap`) — or restore the
        docked layout.

        Compact keeps everything that says *what you are looking at* (image, M/T/Z
        cursor + play, channel toggles, the Overlays button, status) and drops what needs
        room to be usable: the LUT histograms, their tool buttons and the per-axis fps
        spinners. It is a pure layout change — no image, contrast, channel or playback
        state is touched, so docking back mid-playback just makes the controls reappear.

        ``force`` re-applies the current state, which the window does after every pull:
        :meth:`_rebuild_channels` builds fresh LUT columns, and fresh widgets are
        visible — without this the histograms would creep back into the mini-map."""
        on = bool(on)
        if on == self._compact and not force:
            return
        self._compact = on
        # compact keeps the channel toggles (they head each LUT column) but drops the
        # histograms + the LUT tool buttons + the fps spinners.
        self._lut_auto.setVisible(not on)
        self._lut_fit.setVisible(not on)
        for hst in self._hists.values():
            hst.setVisible(not on)
        for lo_e, hi_e in self._lut_edits.values():
            lo_e.setVisible(not on)
            hi_e.setVisible(not on)
        for sp in self._fps_spins.values():
            sp.setVisible(not on)
        for lbl in self._val_lbls.values():
            lbl.setMinimumWidth(44 if on else 84)
        self._ovl_btn.setText(_OVL_LABEL[1] if on else _OVL_LABEL[0])
        lay = self.layout()
        lay.setContentsMargins(*((3, 2, 3, 2) if on else (6, 6, 6, 4)))
        lay.setSpacing(2 if on else 4)
        self._apply_image_minimum()

    def _apply_image_minimum(self) -> None:
        """The image surface's floor. Both backends ship a 200×200 minimum, which alone
        would keep the mini-map from getting small; compact drops it to 120×90."""
        w, h = (120, 90) if self._compact else (200, 200)
        self._view.setMinimumSize(w, h)

    @property
    def compact(self) -> bool:
        return self._compact

    # ── API used by the window ─────────────────────────────────────────────────
    def coords(self) -> Tuple[int, int, int, int]:
        c = self._active_channels[0] if self._active_channels else 0
        return (self._sliders["m"].value(), self._sliders["t"].value(),
                self._sliders["z"].value(), c)

    def channels(self) -> Tuple[int, ...]:
        return tuple(sorted(self._active_channels)) or (0,)

    # ── result / error ─────────────────────────────────────────────────────────
    def show_result(self, node_id: str, planes, axes, seconds: float,
                    dataset=None) -> None:
        """Full-pull delivery: refresh dataset/axes/channels, then display. Contrast is
        (re)computed once for a new node/volume and cached across all subsequent frames."""
        self._dataset = dataset
        if node_id != self._clim_node:
            self._clim.clear()                    # a new node/volume → recompute contrast
            self._clim_node = node_id
        self._node_id = node_id
        key = (axes.m, axes.t, axes.z, axes.c) if axes is not None else None
        if axes is not None and key != self._axes_key:
            self._apply_axes(axes, dataset)
            self._axes_key = key
        self._axes = axes if axes is not None else self._axes

        if not planes:
            self._planes = {}
            self._ref_plane = self._base_pix = None
            if self._gl is not None:
                self._gl.clear()
            else:
                self._view.set_pixmap(QPixmap())
            self._status.setText(f"{node_id} · no image on this output · "
                                 f"pulled in {seconds:.2f}s")
            return
        npts = self._display(node_id, planes, self._axes)
        h, w = self._ref_plane.shape[:2]
        extra = f" · {npts} points" if npts else ""
        chans = "+".join(self._chan_names[c] if c < len(self._chan_names) else f"Ch{c}"
                         for c in self.channels())
        self._status.setText(
            f"{node_id} · {w}×{h} px · {chans}{extra} · pulled in {seconds:.2f}s")
        self._measure_fps()

    def show_planes(self, node_id: str, planes, axes, seconds: float) -> None:
        """Fast-path delivery (scrub/play): the dataset and axes are unchanged, so only
        the displayed frame moves — no axes/channel rebuild, no spreadsheet refresh."""
        if axes is not None:
            self._axes = axes
        if not planes:
            return
        self._node_id = node_id
        self._display(node_id, planes, self._axes)
        self._measure_fps()

    def _clim_for(self, node_id: str, ch: int, plane: np.ndarray) -> Tuple[float, float]:
        """Percentile contrast bounds for a channel, computed once (native units) and
        cached — this is what keeps ``np.percentile`` off the per-frame hot path."""
        ckey = (node_id, ch)
        lohi = self._clim.get(ckey)
        if lohi is None:
            a = np.asarray(plane, dtype=float)
            finite = a[np.isfinite(a)]
            if finite.size == 0:
                lohi = (0.0, 1.0)
                self._drange[ckey] = (0.0, 1.0)
            else:
                lo = float(np.percentile(finite, 1.0))
                hi = float(np.percentile(finite, 99.5))
                if hi <= lo:
                    hi = lo + 1.0
                lohi = (lo, hi)
                self._drange[ckey] = self._display_range(plane)
            self._clim[ckey] = lohi
        return lohi

    def _display_range(self, plane: np.ndarray) -> Tuple[float, float]:
        """The LUT histogram extent. Prefer the file's **significant bit depth** from
        metadata (``bit_depth`` → ``0 .. 2**bits-1``) — the pixel values alone can't
        reveal it (a dim 12-bit frame maxes out below 1024, which is why inferring from
        the data under-capped the slider at 1023). Without metadata, an integer image
        falls back to its dtype's full range (never under-caps); a float (computed) image
        uses its observed range."""
        a = np.asarray(plane)
        if self._bit_depth and np.issubdtype(a.dtype, np.integer):
            return 0.0, float((1 << int(self._bit_depth)) - 1)
        if np.issubdtype(a.dtype, np.integer):
            return 0.0, float(np.iinfo(a.dtype).max)
        f = np.asarray(a, dtype=float)
        finite = f[np.isfinite(f)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = float(finite.min()), float(finite.max())
        return lo, (hi if hi > lo else lo + 1.0)

    def _display(self, node_id: str, planes, axes) -> int:
        """Render the given per-channel planes on the active backend. Returns the number
        of overlay points (for the status line). When Auto is ON, the percentile contrast
        is recomputed for THIS frame (so it tracks brightness while scrubbing/playing)."""
        self._planes = dict(planes)
        self._ref_plane = next(iter(self._planes.values()))
        self._geo_key = None                  # a new frame → re-extract overlay geometry
        if self._auto_on:
            clims = {}
            for ch, pl in self._planes.items():
                lohi = self._auto_clim(pl)
                self._clim[(node_id, ch)] = lohi           # reflect on the histogram
                self._drange.setdefault((node_id, ch), self._display_range(pl))
                clims[ch] = lohi
        else:
            clims = {ch: self._clim_for(node_id, ch, pl) for ch, pl in self._planes.items()}
        gammas = {ch: self._gammas.get((node_id, ch), 1.0) for ch in self._planes}
        self._sync_luts()
        if self._gl is not None:
            for ch, pl in self._planes.items():
                lo, hi = clims[ch]
                self._gl.set_channel(ch, lo, hi,
                                     self._chan_colors.get(ch, (255, 255, 255)), gammas[ch])
            self._gl.set_planes(self._planes)     # uploads + repaints (overlays via cb)
            return self._point_count()
        self._base_pix = QPixmap.fromImage(
            composite_with_clim(self._planes, self._chan_colors, clims, gammas))
        return self._repaint()

    # ── LUT / contrast (instantaneous on GPU: a shader-uniform change) ───────────
    def _lut_channel(self) -> int:
        return self._active_channels[0] if self._active_channels else 0

    def _auto_clim(self, plane: np.ndarray) -> Tuple[float, float]:
        """Percentile (1 / 99.5) contrast from a cheap subsample — used per frame when
        Auto is toggled on."""
        a = np.asarray(plane)
        step = max(1, int(np.sqrt(a.size / 200_000)))
        s = a[::step, ::step] if a.ndim == 2 else a
        s = np.asarray(s, dtype=float)
        finite = s[np.isfinite(s)]
        if finite.size == 0:
            return (0.0, 1.0)
        lo = float(np.percentile(finite, 1.0))
        hi = float(np.percentile(finite, 99.5))
        return (lo, hi if hi > lo else lo + 1.0)

    def _hist_tip(self, ch: int) -> str:
        ckey = (self._node_id, ch)
        lo, hi = self._clim.get(ckey, (0.0, 1.0))
        name = self._chan_names[ch] if ch < len(self._chan_names) else f"Ch{ch}"
        return (f"{name}: {lo:.0f}–{hi:.0f}  γ={self._gammas.get(ckey, 1.0):.2f}\n"
                "drag handles=window · middle dot=gamma · wheel=zoom · dbl-click=reset")

    @staticmethod
    def _fmt(v: float) -> str:
        """Compact numeric text for a LUT bound (int-like for sensor counts, ``%.4g``
        otherwise so small float ranges keep their precision)."""
        return f"{v:.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.4g}"

    def _update_lut_edits(self, ch: int, force: bool = False) -> None:
        """Refresh channel ``ch``'s black/white text fields from its window. ``force``
        overwrites even a focused field — used when the value changed from a DRAG (the
        histogram doesn't steal keyboard focus, so without this a focused field would show
        a stale number during/after a drag). Periodic syncs pass ``force=False`` so they
        never stomp a value being typed."""
        edits = self._lut_edits.get(ch)
        if edits is None:
            return
        lo, hi = self._clim.get((self._node_id, ch), (0.0, 1.0))
        for e, val in zip(edits, (lo, hi)):
            if force or not e.hasFocus():
                e.blockSignals(True)
                e.setText(self._fmt(val))
                e.blockSignals(False)

    def _on_lut_edit(self, ch: int) -> None:
        """Enter/blur on a black/white field → set that exact window, drop Auto, and fit
        the histogram view to the new window."""
        if self._node_id is None:
            return
        lo_e, hi_e = self._lut_edits[ch]
        try:
            lo, hi = float(lo_e.text()), float(hi_e.text())
        except ValueError:
            self._update_lut_edits(ch, force=True)   # bad input → revert to current
            return
        if hi <= lo:
            hi = lo + 1.0
        # No change (e.g. blur right after a drag already set this window) → nothing to do,
        # and in particular don't re-fit the view.
        cur = self._clim.get((self._node_id, ch))
        if cur is not None and abs(cur[0] - lo) < 1e-9 and abs(cur[1] - hi) < 1e-9:
            return
        if self._lut_auto.isChecked():
            self._lut_auto.setChecked(False)  # a manual value leaves per-frame Auto
        self._clim[(self._node_id, ch)] = (lo, hi)
        # widen the histogram's data range if the typed value exceeds it, so the handle
        # can actually sit there, then fit the view to the new window.
        ckey = (self._node_id, ch)
        dmin, dmax = self._drange.get(ckey, (lo, hi))
        self._drange[ckey] = (min(dmin, lo), max(dmax, hi))
        hst = self._hists.get(ch)
        if hst is not None:
            hst.set_range(*self._drange[ckey])
            hst.set_window(lo, hi)
            hst.fit_view_to_window()          # auto-fit the new manual value
        self._apply_lut(ch)

    def _sync_luts(self) -> None:
        """Refresh EVERY channel's histogram — range, window, gamma, tint, and the
        (subsampled) distribution of its shown plane. Cheap: sub-sampled per channel."""
        if self._node_id is None:
            return
        for ch, hst in self._hists.items():
            ckey = (self._node_id, ch)
            pl = self._planes.get(ch)
            if ckey not in self._drange and pl is not None:
                self._drange[ckey] = self._display_range(pl)
            dmin, dmax = self._drange.get(ckey, (0.0, 1.0))
            lo, hi = self._clim.get(ckey, (dmin, dmax))
            hst.set_tint(self._chan_colors.get(ch, (120, 170, 255)))
            hst.set_range(dmin, dmax)
            hst.set_window(lo, hi)
            hst.set_gamma(self._gammas.get(ckey, 1.0))
            if pl is not None:
                a = np.asarray(pl)
                step = max(1, int(np.sqrt(a.size / 200_000)))
                hst.set_values(a[::step, ::step] if a.ndim == 2 else a)
            hst.setToolTip(self._hist_tip(ch))
            self._update_lut_edits(ch)

    def _on_lut_window(self, ch: int, lo: float, hi: float) -> None:
        """A drag on channel ``ch``'s handles → update its window and push it (instant)."""
        if self._node_id is None:
            return
        if self._lut_auto.isChecked():
            self._lut_auto.setChecked(False)      # a manual edit leaves per-frame Auto
        self._clim[(self._node_id, ch)] = (float(lo), float(hi))
        self._apply_lut(ch)

    def _on_lut_gamma(self, ch: int, gamma: float) -> None:
        """A drag on channel ``ch``'s midpoint dot → update its gamma (instant)."""
        if self._node_id is None:
            return
        self._gammas[(self._node_id, ch)] = float(gamma)
        self._apply_lut(ch)

    def _on_auto_toggled(self, on: bool) -> None:
        """Auto is a TOGGLE: while on, contrast is recomputed for every frame. Flipping it
        on re-contrasts the current frame immediately."""
        self._auto_on = bool(on)
        if self._auto_on and self._planes and self._node_id is not None:
            self._display(self._node_id, self._planes, self._axes)

    def _fit_all(self) -> None:
        for hst in self._hists.values():
            hst.fit_view_to_window()

    def _apply_lut(self, ch: int) -> None:
        """Push the current window + gamma for ``ch`` to the display. On GPU this is a
        uniform change + repaint — no decode, no re-upload — so it is instantaneous."""
        lo, hi = self._clim.get((self._node_id, ch), (0.0, 1.0))
        gm = self._gammas.get((self._node_id, ch), 1.0)
        hst = self._hists.get(ch)
        if hst is not None:
            hst.setToolTip(self._hist_tip(ch))
        # force: _apply_lut is only called from user LUT actions (drag/gamma/typed value);
        # the field must follow a drag even while it (still) holds keyboard focus.
        self._update_lut_edits(ch, force=True)
        if self._gl is not None:
            self._gl.set_channel(ch, lo, hi,
                                 self._chan_colors.get(ch, (255, 255, 255)), gm)
            self._gl.refresh()
        elif self._planes:
            clims = {c: self._clim.get((self._node_id, c),
                                       self._clim_for(self._node_id, c, pl))
                     for c, pl in self._planes.items()}
            gammas = {c: self._gammas.get((self._node_id, c), 1.0) for c in self._planes}
            self._base_pix = QPixmap.fromImage(
                composite_with_clim(self._planes, self._chan_colors, clims, gammas))
            self._repaint()

    def show_error(self, node_id: str, trace: str) -> None:
        self._stop_play()
        last = [ln for ln in trace.strip().splitlines() if ln.strip()][-1]
        self._status.setText(f"{node_id} FAILED — {last}")
        self._view.setToolTip(trace)

    def show_running(self, node_id: str) -> None:
        self._status.setText(f"pulling {node_id}…")

    # ── axes / channels rebuild ────────────────────────────────────────────────
    def _apply_axes(self, axes, dataset) -> None:
        for ax in _AXES:
            n = getattr(axes, ax)
            sld = self._sliders[ax]
            sld.blockSignals(True)
            sld.setMaximum(max(0, n - 1))
            sld.setEnabled(n > 1)
            sld.blockSignals(False)
            self._val_lbls[ax].setText(f"{sld.value()}/{max(0, n - 1)}")
            playable = n > 1
            self._play_btns[ax].setEnabled(playable)
            self._fps_spins[ax].setEnabled(playable)
            if not playable and self._play_btns[ax].isChecked():
                self._play_btns[ax].setChecked(False)
        self._rebuild_channels(axes, dataset)

    def _rebuild_channels(self, axes, dataset) -> None:
        nc = getattr(axes, "c", 1)
        md = getattr(dataset, "metadata", {}) or {}
        bd = md.get("bit_depth")
        self._bit_depth = int(bd) if bd else None
        names = md.get("channel_names") or []
        emis = md.get("channel_emission_nm") or []
        self._chan_names = [str(names[i]) if i < len(names) and names[i] else f"Ch{i + 1}"
                            for i in range(nc)]
        self._chan_colors = {}
        for i in range(nc):
            nm = emis[i] if i < len(emis) else None
            col = T.emission_qcolor(nm)
            self._chan_colors[i] = (col.red(), col.green(), col.blue())
        # keep only still-valid active channels; default to channel 0 if none
        self._active_channels = [c for c in self._active_channels if c < nc] or [0]

        # rebuild the per-channel LUT columns: [channel toggle] over [its histogram]
        for col in self._lut_cols:
            col.setParent(None)
            col.deleteLater()
        self._lut_cols = []
        self._chan_btns = {}
        self._hists = {}
        self._lut_edits = {}
        for i in range(nc):
            btn = QPushButton(self._chan_names[i])
            btn.setCheckable(True)
            btn.setChecked(i in self._active_channels)
            btn.setCursor(Qt.PointingHandCursor)
            em = emis[i] if i < len(emis) else None
            btn.setToolTip(f"{self._chan_names[i]}"
                           + (f" · {em:.0f} nm" if isinstance(em, (int, float)) else "")
                           + " — toggle in the composite")
            btn.clicked.connect(lambda _c, idx=i: self._on_channel_toggle(idx))
            self._style_channel_btn(btn, i)
            hst = HistogramLUT()
            hst.set_tint(self._chan_colors.get(i, (120, 170, 255)))
            hst.setVisible(not self._compact)
            hst.window_changed.connect(lambda lo, hi, c=i: self._on_lut_window(c, lo, hi))
            hst.gamma_changed.connect(lambda gm, c=i: self._on_lut_gamma(c, gm))
            # editable black/white readouts — click to type an exact value
            lo_e, hi_e = QLineEdit(), QLineEdit()
            for e, which in ((lo_e, "black point"), (hi_e, "white point")):
                e.setProperty("role", "lutedit")
                e.setAlignment(Qt.AlignCenter)
                e.setToolTip(f"{which} — type a value + Enter")
                e.setVisible(not self._compact)
                e.editingFinished.connect(lambda c=i: self._on_lut_edit(c))
            erow = QHBoxLayout()
            erow.setContentsMargins(0, 0, 0, 0)
            erow.setSpacing(3)
            erow.addWidget(lo_e)
            erow.addWidget(hi_e)
            colw = QWidget()
            cl = QVBoxLayout(colw)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            cl.addWidget(btn)
            cl.addWidget(hst, 1)
            cl.addLayout(erow)
            self._lut_strip.addWidget(colw, 1)
            self._lut_cols.append(colw)
            self._chan_btns[i] = btn
            self._hists[i] = hst
            self._lut_edits[i] = (lo_e, hi_e)

    def _style_channel_btn(self, btn: QPushButton, idx: int) -> None:
        r, g, b = self._chan_colors.get(idx, (196, 200, 208))
        from PySide6.QtGui import QColor
        col = QColor(r, g, b)
        on_text = _text_on(col)
        btn.setStyleSheet(f"""
            QPushButton {{ border:1px solid rgb({r},{g},{b}); border-radius:5px;
                padding:3px 9px; font-weight:600; color:rgb({r},{g},{b});
                background:transparent; }}
            QPushButton:checked {{ background:rgb({r},{g},{b}); color:{on_text}; }}
        """)

    # ── interaction ────────────────────────────────────────────────────────────
    def _on_slider(self, ax: str) -> None:
        sld = self._sliders[ax]
        self._val_lbls[ax].setText(f"{sld.value()}/{sld.maximum()}")
        self.request_changed.emit()

    def _on_channel_toggle(self, idx: int) -> None:
        active = set(self._active_channels)
        if idx in active:
            active.discard(idx)
        else:
            active.add(idx)
        if not active:                       # never leave zero channels shown
            active = {idx}
            self._chan_btns[idx].setChecked(True)
        self._active_channels = sorted(active)
        self.request_changed.emit()          # re-pull → _display re-syncs the histograms

    # ── playback (wall-clock timer, frame-dropping) ─────────────────────────────
    def _interval_ms(self, ax: str) -> int:
        return int(max(1.0, 1000.0 / max(0.5, self._fps_spins[ax].value())))

    def _on_play(self, ax: str, on: bool) -> None:
        if on:
            if self._sliders[ax].maximum() < 1:
                self._play_btns[ax].setChecked(False)
                return
            # one axis at a time
            for other in _AXES:
                if other != ax and self._play_btns[other].isChecked():
                    self._play_btns[other].blockSignals(True)
                    self._play_btns[other].setChecked(False)
                    self._play_btns[other].setText("▶")
                    self._play_btns[other].blockSignals(False)
            self._playing_axis = ax
            self._play_btns[ax].setText("⏸")
            self._last_frame_t = None
            self._fps_ema = None
            self._play_timer.start(self._interval_ms(ax))
        else:
            if self._playing_axis == ax:
                self._stop_play()
            self._play_btns[ax].setText("▶")

    def _stop_play(self) -> None:
        ax = self._playing_axis
        self._playing_axis = None
        self._play_timer.stop()
        if ax is not None:
            btn = self._play_btns[ax]
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.setText("▶")
            btn.blockSignals(False)
            self._val_lbls[ax].setText(f"{self._sliders[ax].value()}/"
                                       f"{self._sliders[ax].maximum()}")

    def _retarget_fps(self, ax: str) -> None:
        if self._playing_axis == ax and self._play_timer.isActive():
            self._play_timer.setInterval(self._interval_ms(ax))

    def _tick_play(self) -> None:
        """Fire on the wall clock: advance the cursor and request the frame. The plane
        request is served from the warm cache (a miss decodes one small plane); if a tick
        arrives while the previous is still working, Qt coalesces the timeout — i.e. the
        frame is dropped — so playback keeps real-time rather than lagging behind."""
        ax = self._playing_axis
        if ax is None:
            return
        sld = self._sliders[ax]
        nxt = sld.value() + 1
        if nxt > sld.maximum():
            nxt = 0
        sld.setValue(nxt)          # → _on_slider → request_changed → request_plane (fast)

    def _measure_fps(self) -> None:
        if self._playing_axis is None:
            return
        now = time.perf_counter()
        if self._last_frame_t is not None:
            dt = now - self._last_frame_t
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema is None else \
                    0.6 * self._fps_ema + 0.4 * inst
        self._last_frame_t = now
        ax = self._playing_axis
        sld = self._sliders[ax]
        fps = f" · {self._fps_ema:.1f} fps" if self._fps_ema else ""
        self._val_lbls[ax].setText(f"{sld.value()}/{sld.maximum()}{fps}")

    # ── overlays ───────────────────────────────────────────────────────────────
    # The panel's job is to turn the viewed Dataset + (m,t,z,c) into geometry; the LOOK
    # lives in nodelab_v2.overlays (settings + renderer) and is edited in the Overlays
    # popup. Extraction is cached per (dataset, coords, plane size, settings revision)
    # because a pan/zoom repaints constantly and must not re-walk the attribute tables.
    def open_overlay_dialog(self) -> None:
        """Open (or raise) the Overlays popup — the single entry point to every overlay's
        configuration. Non-modal, so edits are seen against the live image."""
        from nodelab_v2.overlay_dialog import OverlayDialog
        if self._ovl_dialog is None:
            dlg = OverlayDialog(self.overlays, self._renderer, self)
            dlg.changed.connect(self._overlays_changed)
            self._ovl_dialog = dlg
            if self._ovl_sources:
                dlg._status.setText(" · ".join(self._ovl_sources))
            btn = self._ovl_btn
            if btn.isVisible():                  # drop it just under the button
                dlg.move(btn.mapToGlobal(QPoint(0, btn.height() + 6)))
        dlg = self._ovl_dialog
        dlg.reload()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _overlays_changed(self) -> None:
        """A settings edit: drop the caches and redraw (no re-pull, no re-composite —
        overlays are painted over the finished image)."""
        self._ovl_rev += 1
        self._renderer.invalidate()
        self._geo_key = None
        self._view.refresh()

    def overlay_enabled(self, tab: str) -> bool:
        return bool(self.overlays.group(tab).enabled)

    def set_overlay_enabled(self, tab: str, on: bool) -> None:
        """Turn one domain's overlay on/off (what the old per-domain checkboxes did)."""
        grp = self.overlays.group(tab)
        if bool(grp.enabled) != bool(on):
            grp.enabled = bool(on)
            if self._ovl_dialog is not None:
                self._ovl_dialog.reload()
            self._overlays_changed()

    def _ensure_geometry(self) -> None:
        """Extract the overlay geometry for the viewed frame if the cache is stale."""
        key = (id(self._dataset), self.coords(),
               None if self._ref_plane is None else self._ref_plane.shape[:2],
               self._ovl_rev)
        if key == self._geo_key:
            return
        self._geo_key = key
        s = self.overlays
        self._geo_points = self._point_marks() if s.points.enabled else []
        self._geo_tracks = self._track_paths() if s.tracks.enabled else []
        self._geo_lab = self._label_plane() if s.labels.enabled else None
        self._geo_mesh = self._mesh_sections() if s.mesh.enabled else []

    def _paint_overlays(self, p: QPainter) -> None:
        """Paint every enabled overlay in **widget** coordinates.

        Called back by whichever image surface is live (GL after its draw, the CPU view
        from ``drawForeground`` with the world transform reset). Because the geometry is
        mapped through the backend's ``plane_to_widget`` at paint time and every size in
        the settings is a screen size, zooming moves the overlay with the image without
        changing how thick or how big it is.
        """
        if self._ref_plane is None:
            return
        mp = getattr(self._view, "plane_to_widget", None)
        if mp is None:                            # a surface without the overlay contract
            return
        H, W = self._ref_plane.shape[:2]
        ax = self._axes
        self._ensure_geometry()
        frame = OV.OverlayFrame(
            map_pt=mp, plane_wh=(W, H),
            sy=H / max(1, ax.y) if ax is not None else 1.0,
            sx=W / max(1, ax.x) if ax is not None else 1.0,
            points=self._geo_points, tracks=self._geo_tracks, mesh=self._geo_mesh,
            label_plane=self._geo_lab,
            current_t=self._sliders["t"].value(),
        )
        self._renderer.paint(p, self.overlays, frame)

    def _point_count(self) -> int:
        """Points drawn on the viewed plane (the status line's ``· N points``)."""
        if not (self.overlays.points.enabled and self._axes is not None):
            return 0
        return len(self._points_here())

    def _point_marks(self) -> List[OV.PointMark]:
        """Every Point-domain row that belongs on the viewed frame, as draw-ready marks.

        Positions stay in *structure* coordinates (full-resolution image pixels); the
        renderer scales them into the displayed plane. ``id`` (when the layer carries one)
        is the per-point colour key so a detection keeps its colour across frames, and the
        layer index is the per-layer key. With ``z_project`` on, off-plane detections come
        along flagged ``on_plane=False`` and the renderer dims them.
        """
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes"):
            return []
        m, t, z, _c = self.coords()
        project = bool(self.overlays.points.z_project)
        by_layer: Dict[Any, Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.POINT:
                by_layer.setdefault(layer, {})[name] = attr.values
        out: List[OV.PointMark] = []
        # sorted: the per-layer colour must not depend on dict insertion order
        for li, layer in enumerate(sorted(by_layer, key=lambda v: str(v))):
            cols = by_layer[layer]
            if "y" not in cols or "x" not in cols:
                continue
            ys, xs = cols["y"], cols["x"]
            zs = cols.get("z"); ms = cols.get("m"); ts = cols.get("t")
            ids = cols.get("id")
            for i in range(len(ys)):
                if ms is not None and int(ms[i]) != m:
                    continue
                if ts is not None and int(ts[i]) != t:
                    continue
                on_plane = zs is None or round(float(zs[i])) == z
                if not on_plane and not project:
                    continue
                key = int(ids[i]) if ids is not None else i + 1
                out.append(OV.PointMark(float(ys[i]), float(xs[i]), key, li, on_plane))
        return out

    def _points_here(self):
        """On-plane point positions ``(y, x)`` — kept for callers/tests and the count."""
        if self.overlays.points.enabled:
            self._ensure_geometry()
            marks = self._geo_points
        else:
            marks = self._point_marks()
        return [(mk.y, mk.x) for mk in marks if mk.on_plane]

    def _label_plane(self) -> Optional[np.ndarray]:
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes") or self._ref_plane is None:
            return None
        m, t, z, c = self.coords()
        best = None
        best_regions = 1
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is not Domain.VOXEL:
                continue
            vals = attr.values
            if not np.issubdtype(vals.dtype, np.integer) or vals.ndim != 6:
                continue
            try:
                plane = vals[m, t, z, c]
            except IndexError:
                continue
            regions = int(plane.max())
            if regions >= 1 and regions >= best_regions:
                best, best_regions = plane, regions
        if best is None:
            return None
        ay, ax_ = best.shape
        ty = max(1, int(np.ceil(ay / self._ref_plane.shape[0])))
        tx = max(1, int(np.ceil(ax_ / self._ref_plane.shape[1])))
        return best[::ty, ::tx]

    def _mesh_sections(self) -> List[OV.MeshSection]:
        """Every Mesh element on the viewed frame, cut by the viewed Z plane.

        The Mesh domain stores three flat CSR strata per mesh under one domain, keyed by a
        layer sub-name (``L`` element rows / ``L/vert`` / ``L/face``) — see
        :mod:`nodegraph.mesh`. This walks the store directly rather than going through
        ``read_mesh`` so a partially-written or foreign mesh can never raise inside a paint
        call: anything that does not look like a mesh is simply skipped.

        Coordinates stay in *structure* pixels (voxel ``z,y,x``, the domain's storage
        convention); the renderer scales them into the displayed plane, exactly as for
        points and tracks.
        """
        ds = self._dataset
        out: List[OV.MeshSection] = []
        if ds is None or not hasattr(ds, "attributes"):
            return out
        m, t, z, c = self.coords()
        near = max(0.0, float(self.overlays.mesh.near_z))
        vertex_style = self.overlays.mesh.style == "points"
        by_layer: Dict[Any, Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.MESH and layer is not None:
                by_layer.setdefault(layer, {})[name] = attr.values
        for layer in sorted(by_layer, key=lambda v: str(v)):
            base, part = MESH_PART(str(layer))
            if part is not None:                       # only iterate ELEMENT buckets
                continue
            el = by_layer[layer]
            vt = by_layer.get(f"{base}/vert")
            fc = by_layer.get(f"{base}/face")
            if vt is None or fc is None:
                continue
            if not ({"id", "m", "t", "c", "vert_start", "vert_count",
                     "face_start", "face_count"} <= set(el)
                    and {"z", "y", "x"} <= set(vt) and {"v0", "v1", "v2"} <= set(fc)):
                continue
            vz, vy, vx = (np.asarray(vt[k], dtype=float) for k in ("z", "y", "x"))
            tri = np.column_stack([np.asarray(fc[f"v{i}"], dtype=np.int64)
                                   for i in range(3)])
            verts = np.column_stack([vz, vy, vx])
            for row in range(len(np.asarray(el["id"]))):
                if (int(np.asarray(el["m"])[row]) != m
                        or int(np.asarray(el["t"])[row]) != t
                        or int(np.asarray(el["c"])[row]) != c):
                    continue
                vs = int(np.asarray(el["vert_start"])[row])
                vc = int(np.asarray(el["vert_count"])[row])
                fs = int(np.asarray(el["face_start"])[row])
                nf = int(np.asarray(el["face_count"])[row])
                oid = int(np.asarray(el["id"])[row])
                sub_v = verts[vs:vs + vc]
                sub_f = tri[fs:fs + nf] - vs           # v0..v2 are GLOBAL indices
                sec = OV.MeshSection(oid)
                if vertex_style:
                    keep = np.abs(sub_v[:, 0] - float(z)) <= near
                    sec.verts = [(float(a), float(b))
                                 for a, b in zip(sub_v[keep, 1], sub_v[keep, 2])]
                else:
                    sec.loops, sec.closed = OV.mesh_section(sub_v, sub_f, float(z))
                if sec.loops or sec.verts:
                    out.append(sec)
        return out

    def _member_layers(self):
        """``{(domain, layer): {col: values}}`` for every Point/Label layer that carries
        a joinable ``id,y,x`` triple — the member position source for Track trajectories."""
        ds = self._dataset
        out: Dict[tuple, Dict[str, np.ndarray]] = {}
        if ds is None or not hasattr(ds, "attributes"):
            return out
        for (dom, layer, name), attr in ds.attributes.items():
            if dom in (Domain.POINT, Domain.LABEL):
                out.setdefault((dom, layer), {})[name] = attr.values
        return {k: v for k, v in out.items()
                if {"id", "y", "x"} <= set(v)}

    def _track_paths(self) -> List[OV.TrackPath]:
        """Join every Track layer to its members' positions and return, for the current
        M, one :class:`~nodelab_v2.overlays.TrackPath` per track — its positions ordered
        by ``t``, the index of the vertex at the viewed T, and each vertex's timepoint
        (which is what lets the *trail* modes draw only part of a trajectory).

        A Track layer only stores ``track_id, t, member_id`` — the geometry lives on the
        member (Point/Label) domain, keyed by ``id``. Point and Label ids share an
        id-space (both start at 1), so per Track layer we pick the member layer whose id
        set best covers the members (resolves the ambiguity, avoids cross-space matches).
        Positions stay projected over z and filtered to the viewed M."""
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes") or self._axes is None:
            return []
        m, t, _z, _c = self.coords()
        track_layers: Dict[Optional[str], Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.TRACK:
                track_layers.setdefault(layer, {})[name] = attr.values
        if not track_layers:
            return []
        members = self._member_layers()
        trajectories = []
        for cols in track_layers.values():
            if not {"track_id", "t", "member_id"} <= set(cols):
                continue
            tid = np.asarray(cols["track_id"]).astype(np.int64)
            tt = np.asarray(cols["t"]).astype(np.int64)
            mem = np.asarray(cols["member_id"]).astype(np.int64)
            want = set(mem.tolist())
            # pick the member layer covering the most of this table's member ids
            best_cols, best_cov = None, 0
            for mcols in members.values():
                cov = len(want & set(np.asarray(mcols["id"]).astype(np.int64).tolist()))
                if cov > best_cov:
                    best_cols, best_cov = mcols, cov
            if best_cols is None:
                continue
            ids = np.asarray(best_cols["id"]).astype(np.int64)
            ys = np.asarray(best_cols["y"], dtype=float)
            xs = np.asarray(best_cols["x"], dtype=float)
            ms = np.asarray(best_cols["m"]).astype(np.int64) if "m" in best_cols else None
            id_to_i = {int(v): i for i, v in enumerate(ids.tolist())}
            for track in np.unique(tid).tolist():
                sel = tid == track
                order = np.argsort(tt[sel], kind="stable")
                ts_ord = tt[sel][order].tolist()
                mem_ord = mem[sel][order].tolist()
                path, cur, ts_kept = [], None, []
                for tv, mv in zip(ts_ord, mem_ord):
                    i = id_to_i.get(int(mv))
                    if i is None or (ms is not None and int(ms[i]) != m):
                        continue
                    path.append((float(ys[i]), float(xs[i])))
                    ts_kept.append(int(tv))
                    if int(tv) == t:
                        cur = len(path) - 1
                if path:
                    trajectories.append(OV.TrackPath(int(track), path, cur, ts_kept))
        return trajectories

    def _tracks_here(self):
        """``(track_id, path, current_index, timepoints)`` per track — the tuple form kept
        for callers/tests around :meth:`_track_paths`."""
        if self.overlays.tracks.enabled:
            self._ensure_geometry()
            paths = self._geo_tracks
        else:
            paths = self._track_paths()
        return [(tr.track_id, tr.path, tr.current, tr.times) for tr in paths]

    def _track_color(self, track_id: int) -> QColor:
        """The colour track ``track_id`` is drawn in under the current settings (a
        deterministic golden-angle hue in ``per_track`` mode, stable across every T)."""
        return self._renderer.track_color(self.overlays.tracks, int(track_id))

    def _repaint(self) -> int:
        """Show the current composite and ask the surface to redraw its overlays.

        The overlays are **not** baked into the pixmap any more (that is precisely what
        made them scale with the zoom on the CPU path) — they are painted on top in widget
        space by :meth:`_paint_overlays`. So this is now a cheap "push the image, ask for a
        repaint", and toggling an overlay never re-composites the channels."""
        if self._ref_plane is None:
            return 0
        if self._gl is not None:
            self._gl.refresh()
            return self._point_count()
        if self._base_pix is None:
            return 0
        self._view.set_pixmap(self._base_pix)
        self._view.refresh()
        return self._point_count()

    # ── styling ────────────────────────────────────────────────────────────────
    def restyle(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{ background:{T.PANEL.name()}; color:{T.INK.name()}; }}
            QLabel[role="muted"] {{ color:{T.MUTED.name()}; font-size:11px; }}
            QLabel[role="title"] {{ color:{T.INK.name()}; font-weight:600; }}
            QLabel[role="axis"] {{ color:{T.INK_2.name()}; font-family:{T.MONO};
                font-weight:700; min-width:12px; }}
            QGraphicsView {{ border:1px solid {T.BORDER.name()}; border-radius:6px;
                background:{T.BG.name()}; }}
            QCheckBox {{ color:{T.INK_2.name()}; font-size:11px; }}
            QDoubleSpinBox {{ background:{T.BODY.name()}; color:{T.INK.name()};
                border:1px solid {T.BORDER.name()}; border-radius:4px; padding:0px 2px;
                font-family:{T.MONO}; font-size:11px; max-width:50px; max-height:18px; }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width:0; }}
            QLineEdit[role="lutedit"] {{ background:{T.BODY.name()}; color:{T.INK.name()};
                border:1px solid {T.BORDER.name()}; border-radius:4px; padding:1px 2px;
                font-family:{T.MONO}; font-size:10px; max-height:17px; min-width:34px; }}
            QLineEdit[role="lutedit"]:focus {{ border:1px solid {T.ACCENT.name()}; }}
            QToolButton {{ background:transparent; color:{T.INK.name()};
                border:0; padding:0px 4px; font-size:13px; max-height:18px; }}
            QToolButton:checked {{ color:{T.ACCENT.name()}; }}
            QToolButton:disabled {{ color:{T.MUTED.name()}; }}
            QSlider {{ max-height:14px; }}
            QSlider::groove:horizontal {{ height:4px; border-radius:2px;
                background:{T.BODY.name()}; }}
            QSlider::sub-page:horizontal {{ background:{T.ACCENT_DIM.name()};
                border-radius:2px; }}
            QSlider::handle:horizontal {{ width:11px; margin:-5px 0;
                border-radius:6px; background:{T.ACCENT.name()};
                border:1px solid {T.BG.name()}; }}
            QSlider::handle:horizontal:disabled {{ background:{T.MUTED.name()}; }}
        """ + T.controls_qss() + f"""
            QCheckBox {{ color:{T.INK_2.name()}; font-size:11px; spacing:5px; }}
        """)
        self._status.set_color(T.MUTED)      # self-painted (elided) → QSS can't reach it
        for idx, b in self._chan_btns.items():
            self._style_channel_btn(b, idx)
        dlg = getattr(self, "_ovl_dialog", None)   # restyle() also runs from __init__
        if dlg is not None:
            dlg.restyle()


__all__ = ["ViewerPanel", "plane_to_qimage", "composite_to_qimage"]
