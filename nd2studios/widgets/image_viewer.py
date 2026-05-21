"""
Fast QPixmap-based image viewer with multi-channel RGB compositing,
T slider, per-channel contrast, and overlay callback support.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QSizePolicy, QPushButton,
)
from PySide6.QtCore import Qt, Signal, QPoint, QRectF, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QPolygonF

from nd2studios.core.settings import Settings


# Standard LUT colors available for channel assignment
CHANNEL_COLORS = {
    "gray": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (255, 0, 0),
    "blue": (0, 100, 255),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "white": (255, 255, 255),
}


def frame_to_uint8(frame: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Convert any-dtype 2D array to uint8 using percentile contrast."""
    f = frame.astype(np.float32)
    lo = np.percentile(f, p_low)
    hi = np.percentile(f, p_high)
    f = np.clip((f - lo) / (hi - lo + 1e-10), 0, 1)
    return (f * 255).astype(np.uint8)


def apply_lut_color(gray_uint8: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Apply a monotone color LUT to a grayscale uint8 image. Returns (H, W, 3) float32 0-255."""
    r, g, b = color
    f = gray_uint8.astype(np.float32)
    rgb = np.zeros((*gray_uint8.shape, 3), dtype=np.float32)
    rgb[..., 0] = f * (r / 255.0)
    rgb[..., 1] = f * (g / 255.0)
    rgb[..., 2] = f * (b / 255.0)
    return rgb


def composite_channels(
    channel_frames: Dict[str, np.ndarray],
    channel_colors: Dict[str, Tuple[int, int, int]],
    channel_enabled: Dict[str, bool],
    auto_contrast: bool = True,
) -> np.ndarray:
    """
    Composite multiple channels into a single RGB image.

    Parameters
    ----------
    channel_frames : dict mapping channel_name -> (H, W) array (any dtype)
    channel_colors : dict mapping channel_name -> (R, G, B) tuple
    channel_enabled : dict mapping channel_name -> bool
    auto_contrast : bool, use per-frame percentile stretching

    Returns
    -------
    (H, W, 3) uint8 RGB image
    """
    # Get shape from first channel
    sample = next(iter(channel_frames.values()))
    H, W = sample.shape
    composite = np.zeros((H, W, 3), dtype=np.float32)

    for ch_name, frame in channel_frames.items():
        if not channel_enabled.get(ch_name, True):
            continue
        color = channel_colors.get(ch_name, (255, 255, 255))

        if auto_contrast:
            gray = frame_to_uint8(frame)
        else:
            if np.issubdtype(frame.dtype, np.integer):
                mx = np.iinfo(frame.dtype).max
            else:
                mx = frame.max() if frame.max() > 0 else 1.0
            gray = (frame.astype(np.float32) / mx * 255).astype(np.uint8)

        composite += apply_lut_color(gray, color)

    return np.clip(composite, 0, 255).astype(np.uint8)


class ZoomToolbar(QWidget):
    """Home / Zoom-in / Zoom-out button row bound to an ImageCanvas,
    with a live zoom-level readout."""

    def __init__(self, canvas: "ImageCanvas", parent=None):
        super().__init__(parent)
        self._canvas = canvas
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Sized so the labels render legibly on every platform — the old
        # 28×default buttons were too small and the ⌂ glyph didn't render
        # on some systems.
        BTN_W, BTN_H = 56, 26

        self.btn_home = QPushButton("Home")
        self.btn_home.setToolTip("Reset view (fit image to window)")
        self.btn_home.setFixedSize(BTN_W, BTN_H)
        self.btn_home.clicked.connect(self._on_home)
        layout.addWidget(self.btn_home)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setToolTip("Zoom in")
        self.btn_zoom_in.setFixedSize(BTN_H, BTN_H)
        self.btn_zoom_in.clicked.connect(self._on_zoom_in)
        layout.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("\u2212")  # − (minus sign)
        self.btn_zoom_out.setToolTip("Zoom out")
        self.btn_zoom_out.setFixedSize(BTN_H, BTN_H)
        self.btn_zoom_out.clicked.connect(self._on_zoom_out)
        layout.addWidget(self.btn_zoom_out)

        # Pan toggle — left-click-drag to pan when active.
        self.btn_pan = QPushButton("Pan")
        self.btn_pan.setToolTip(
            "Toggle pan tool. When on, left-click and drag to move the image.\n"
            "When off, click reports pixel coordinates."
        )
        self.btn_pan.setCheckable(True)
        self.btn_pan.setFixedSize(BTN_W, BTN_H)
        self.btn_pan.toggled.connect(canvas.set_pan_mode)
        layout.addWidget(self.btn_pan)

        # Keep the button in sync if pan_mode is changed elsewhere.
        canvas.pan_mode_changed.connect(self._on_pan_mode_changed)

        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        self.lbl_zoom.setMinimumWidth(48)
        layout.addWidget(self.lbl_zoom)

        # Update label whenever the canvas updates its zoom.
        canvas.zoom_changed.connect(self._on_zoom_changed)

    def _on_home(self):
        self._canvas.reset_zoom()

    def _on_zoom_in(self):
        self._canvas.zoom_in()

    def _on_zoom_out(self):
        self._canvas.zoom_out()

    def _on_zoom_changed(self, zoom: float):
        self.lbl_zoom.setText(f"{zoom * 100:.0f}%")

    def _on_pan_mode_changed(self, enabled: bool):
        # Block signals to avoid feedback if canvas drove the change.
        if self.btn_pan.isChecked() != enabled:
            self.btn_pan.blockSignals(True)
            self.btn_pan.setChecked(enabled)
            self.btn_pan.blockSignals(False)


class ImageCanvas(QLabel):
    """QLabel that displays a scaled QPixmap with overlay painting and click reporting."""

    clicked = Signal(float, float)            # image-space y, x
    zoom_changed = Signal(float)              # current zoom multiplier (1.0 = fit)
    pan_mode_changed = Signal(bool)           # True if pan tool is active
    crop_rect_selected = Signal(int, int, int, int)  # x, y, w, h (image pixels)
    # Manual-mask drawing — emitted on mouse release.
    # (mode, vertices) where vertices is List[Tuple[float, float]] of (iy, ix)
    # in image pixel space. For "rect" and "ellipse" the list has the two
    # bounding-box corners; for "polygon" it is the recorded freehand path
    # (the consumer closes the loop).
    shape_drawn = Signal(str, list)
    # Vertex drag in edit mode — (vertex_index, iy, ix). Emitted while the
    # mouse is held; on release the canvas emits ``edit_committed()`` so the
    # consumer can persist or coalesce.
    vertex_moved = Signal(int, float, float)
    edit_committed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background-color: {Settings.BG_SECONDARY};")

        self._source_pixmap: Optional[QPixmap] = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._overlay_fn: Optional[Callable] = None
        self._img_w = 0
        self._img_h = 0

        # Zoom/pan state
        self._zoom = 1.0
        self._pan_x = 0.0  # pan offset in image pixels
        self._pan_y = 0.0
        self._panning = False
        self._pan_start = None
        self._pan_start_offset = None
        self.pan_mode = False  # instance attribute, not class
        self.setMouseTracking(True)

        # Crop mode state
        self._crop_mode: bool = False
        self._crop_drag_start: Optional[Tuple[float, float]] = None  # (iy, ix)
        self._crop_drag_end: Optional[Tuple[float, float]] = None    # (iy, ix)
        self._crop_dragging: bool = False

        # Draw mode state (manual-mask shape drawing).
        # _draw_mode is one of: None, "rect", "ellipse", "polygon".
        self._draw_mode: Optional[str] = None
        self._draw_drag_start: Optional[Tuple[float, float]] = None  # (iy, ix)
        self._draw_drag_end: Optional[Tuple[float, float]] = None    # (iy, ix)
        self._draw_dragging: bool = False
        self._poly_vertices: List[Tuple[float, float]] = []  # [(iy, ix), ...]

        # Vertex-edit mode — when ``_edit_vertices`` is not None, draggable
        # handles are painted at each vertex and the user can move them.
        self._edit_vertices: Optional[List[Tuple[float, float]]] = None
        self._edit_drag_idx: Optional[int] = None

    def set_image(self, rgb_array: np.ndarray):
        """Set image from (H, W, 3) uint8 array."""
        h, w = rgb_array.shape[:2]
        self._img_h, self._img_w = h, w
        rgb = np.ascontiguousarray(rgb_array)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._source_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_grayscale(self, gray_uint8: np.ndarray):
        """Set image from (H, W) uint8 array."""
        h, w = gray_uint8.shape
        self._img_h, self._img_w = h, w
        gray = np.ascontiguousarray(gray_uint8)
        qimg = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
        self._source_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_overlay(self, fn: Optional[Callable]):
        self._overlay_fn = fn
        self.update()

    def reset_zoom(self):
        """Reset zoom and pan to fit-to-window."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_in(self):
        self._zoom = min(self._zoom * 1.3, 50.0)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_out(self):
        self._zoom = max(self._zoom / 1.3, 0.1)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def set_pan_mode(self, enabled: bool):
        """Toggle pan tool. When enabled, left-click-drag pans the image
        instead of emitting a pixel click. Updates the cursor."""
        enabled = bool(enabled)
        if enabled == self.pan_mode:
            return
        self.pan_mode = enabled
        self.setCursor(Qt.CursorShape.OpenHandCursor if enabled
                       else Qt.CursorShape.ArrowCursor)
        if not enabled:
            # Cancel any in-flight pan drag
            self._panning = False
            self._pan_start = None
            self._pan_start_offset = None
        self.pan_mode_changed.emit(enabled)

    def set_crop_mode(self, enabled: bool) -> None:
        """Toggle crop selection tool. Deactivates pan, draw, and edit modes."""
        self._crop_mode = bool(enabled)
        self._crop_drag_start = None
        self._crop_drag_end = None
        self._crop_dragging = False
        if enabled:
            self.set_pan_mode(False)
            self._clear_draw_state()
            self._draw_mode = None
            self._edit_vertices = None
            self._edit_drag_idx = None
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_draw_mode(self, mode: Optional[str]) -> None:
        """Set the active shape-drawing tool, or None to disable.

        Valid modes: "rect", "ellipse", "polygon".  Activating a draw mode
        deactivates pan, crop, and vertex-edit so the tools remain mutually
        exclusive.
        """
        if mode not in (None, "rect", "ellipse", "polygon"):
            return
        self._clear_draw_state()
        self._draw_mode = mode
        if mode is not None:
            self.set_pan_mode(False)
            # Clear any in-flight crop without re-entering set_crop_mode (which
            # would zero out the draw cursor we are about to apply).
            self._crop_mode = False
            self._crop_drag_start = None
            self._crop_drag_end = None
            self._crop_dragging = False
            # Drawing a new shape and editing existing vertices are mutually
            # exclusive — clear edit state.
            self._edit_vertices = None
            self._edit_drag_idx = None
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def _clear_draw_state(self) -> None:
        """Reset all draw-tool in-flight state without changing the active mode."""
        self._draw_drag_start = None
        self._draw_drag_end = None
        self._draw_dragging = False
        self._poly_vertices = []

    def set_edit_vertices(self, vertices: Optional[List[Tuple[float, float]]]) -> None:
        """Enter vertex-edit mode showing draggable handles at ``vertices``.

        ``vertices`` is a list of ``(iy, ix)`` pairs in image-pixel space.
        Pass ``None`` to leave edit mode.  Activating edit mode deactivates
        pan, crop, and the draw tools so the modes remain mutually exclusive.
        """
        if vertices is None:
            self._edit_vertices = None
            self._edit_drag_idx = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            return
        # Store a copy so external mutation never leaks back through the
        # canvas state.
        self._edit_vertices = [(float(y), float(x)) for y, x in vertices]
        self._edit_drag_idx = None
        # Mutex with the other tools.
        self.set_pan_mode(False)
        self._crop_mode = False
        self._crop_drag_start = None
        self._crop_drag_end = None
        self._crop_dragging = False
        self._draw_mode = None
        self._clear_draw_state()
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.update()

    def _hit_test_vertex(self, wx: float, wy: float) -> Optional[int]:
        """Return the index of the vertex nearest ``(wx, wy)`` in widget
        coords, if within ``HANDLE_HIT_PX`` widget pixels. Otherwise None."""
        if not self._edit_vertices:
            return None
        # Hit radius scales with the device — 10 widget pixels works on both
        # zoomed-in and zoomed-out views.
        hit_radius_sq = 10.0 * 10.0
        best_idx: Optional[int] = None
        best_d2 = hit_radius_sq
        for i, (iy, ix) in enumerate(self._edit_vertices):
            vx, vy = self.image_to_widget(iy, ix)
            d2 = (vx - wx) ** 2 + (vy - wy) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best_idx = i
        return best_idx

    def center_on(self, img_y: float, img_x: float):
        """Pan so the given image coordinate is at the center of the widget."""
        self._pan_x = img_x - self._img_w / 2
        self._pan_y = img_y - self._img_h / 2
        self.update()

    def paintEvent(self, event):
        if self._source_pixmap is None:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        pw, ph = self._source_pixmap.width(), self._source_pixmap.height()
        ww, wh = self.width(), self.height()
        base_scale = min(ww / pw, wh / ph)
        self._scale = base_scale * self._zoom
        sw, sh = int(pw * self._scale), int(ph * self._scale)
        ox = int((ww - sw) / 2 - self._pan_x * self._scale)
        oy = int((wh - sh) / 2 - self._pan_y * self._scale)
        self._offset = QPoint(ox, oy)
        painter.drawPixmap(ox, oy, sw, sh, self._source_pixmap)
        if self._overlay_fn:
            self._overlay_fn(painter, self._scale, ox, oy, pw, ph)
        if self._crop_drag_start and self._crop_drag_end:
            start_iy, start_ix = self._crop_drag_start
            end_iy, end_ix = self._crop_drag_end
            wx0, wy0 = self.image_to_widget(start_iy, start_ix)
            wx1, wy1 = self.image_to_widget(end_iy, end_ix)
            pen = QPen(QColor(255, 220, 0), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 220, 0, 40)))
            painter.drawRect(QRectF(
                min(wx0, wx1), min(wy0, wy1),
                abs(wx1 - wx0), abs(wy1 - wy0),
            ))
        # In-progress draw-mode preview — cyan dashed to distinguish from crop.
        if self._draw_mode is not None and self._draw_dragging:
            pen = QPen(QColor(0, 220, 255), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 220, 255, 40)))
            if (self._draw_mode in ("rect", "ellipse")
                    and self._draw_drag_start is not None
                    and self._draw_drag_end is not None):
                sy, sx = self._draw_drag_start
                ey, ex = self._draw_drag_end
                wx0, wy0 = self.image_to_widget(sy, sx)
                wx1, wy1 = self.image_to_widget(ey, ex)
                rect = QRectF(
                    min(wx0, wx1), min(wy0, wy1),
                    abs(wx1 - wx0), abs(wy1 - wy0),
                )
                if self._draw_mode == "rect":
                    painter.drawRect(rect)
                else:
                    painter.drawEllipse(rect)
            elif self._draw_mode == "polygon" and len(self._poly_vertices) >= 2:
                poly = QPolygonF()
                for vy, vx in self._poly_vertices:
                    wx, wy = self.image_to_widget(vy, vx)
                    poly.append(QPointF(wx, wy))
                # Draw the path open during the drag — the closing edge appears
                # on release, when the consumer rasterizes the closed polygon.
                painter.drawPolyline(poly)
        # Vertex-edit handles — draw the polygon outline + circles at each
        # vertex so the user can grab any handle to deform the mask locally.
        if self._edit_vertices:
            outline = QPolygonF()
            widget_pts: List[Tuple[float, float]] = []
            for iy, ix in self._edit_vertices:
                wx, wy = self.image_to_widget(iy, ix)
                widget_pts.append((wx, wy))
                outline.append(QPointF(wx, wy))
            # Closed outline so the user sees the shape they're editing.
            pen = QPen(QColor(0, 220, 255), 1.5, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(outline)
            # Handles.
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 140, 200), 1.5))
            for i, (wx, wy) in enumerate(widget_pts):
                r = 5.5 if i == self._edit_drag_idx else 4.0
                painter.drawEllipse(QPointF(wx, wy), r, r)
        painter.end()

    def wheelEvent(self, event):
        """Wheel/pinch zoom is disabled — too sensitive on trackpads.
        Use the toolbar (+ / − / Home) or the Pan toggle instead."""
        event.ignore()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._edit_vertices is not None:
                idx = self._hit_test_vertex(event.pos().x(), event.pos().y())
                if idx is not None:
                    self._edit_drag_idx = idx
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    self.update()
                    event.accept()
                    return
                # Click in empty space inside edit mode is a no-op — neither
                # a pixel click nor a pan start, since that would be confusing.
                event.accept()
                return
            if self._draw_mode is not None:
                iy, ix = self.widget_to_image(event.pos().x(), event.pos().y())
                if self._draw_mode in ("rect", "ellipse"):
                    self._draw_drag_start = (iy, ix)
                    self._draw_drag_end = (iy, ix)
                    self._draw_dragging = True
                else:  # polygon
                    self._poly_vertices = [(iy, ix)]
                    self._draw_dragging = True
                self.update()
                event.accept()
                return
            if self._crop_mode:
                iy, ix = self.widget_to_image(event.pos().x(), event.pos().y())
                self._crop_drag_start = (iy, ix)
                self._crop_drag_end = (iy, ix)
                self._crop_dragging = True
                event.accept()
                return
            if self.pan_mode:
                # Pan mode: start dragging
                self._panning = True
                self._pan_start = event.pos()
                self._pan_start_offset = (self._pan_x, self._pan_y)
                event.accept()
                return
            else:
                # Select mode: emit click
                iy, ix = self.widget_to_image(event.pos().x(), event.pos().y())
                if 0 <= ix < self._img_w and 0 <= iy < self._img_h:
                    self.clicked.emit(iy, ix)
        elif event.button() == Qt.MouseButton.MiddleButton:
            # Middle always pans
            self._panning = True
            self._pan_start = event.pos()
            self._pan_start_offset = (self._pan_x, self._pan_y)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self._edit_vertices is not None
                and self._edit_drag_idx is not None):
            iy, ix = self.widget_to_image(event.pos().x(), event.pos().y())
            # Clamp inside the image so dragged vertices don't escape the frame.
            iy = max(0.0, min(float(self._img_h - 1), iy))
            ix = max(0.0, min(float(self._img_w - 1), ix))
            self._edit_vertices[self._edit_drag_idx] = (iy, ix)
            self.vertex_moved.emit(self._edit_drag_idx, iy, ix)
            self.update()
            event.accept()
            return
        if self._draw_mode is not None and self._draw_dragging:
            iy, ix = self.widget_to_image(event.pos().x(), event.pos().y())
            if self._draw_mode in ("rect", "ellipse"):
                self._draw_drag_end = (iy, ix)
            else:  # polygon — append vertex only when the cursor has moved far
                # enough to keep the vertex count bounded (~ every 2 image pixels).
                if self._poly_vertices:
                    py, px = self._poly_vertices[-1]
                    if (iy - py) * (iy - py) + (ix - px) * (ix - px) >= 4.0:
                        self._poly_vertices.append((iy, ix))
                else:
                    self._poly_vertices.append((iy, ix))
            self.update()
            event.accept()
            return
        if self._crop_mode and self._crop_dragging:
            iy, ix = self.widget_to_image(event.pos().x(), event.pos().y())
            self._crop_drag_end = (iy, ix)
            self.update()
            event.accept()
            return
        if self._panning and self._pan_start is not None:
            dx = event.pos().x() - self._pan_start.x()
            dy = event.pos().y() - self._pan_start.y()
            self._pan_x = self._pan_start_offset[0] - dx / max(self._scale, 0.01)
            self._pan_y = self._pan_start_offset[1] - dy / max(self._scale, 0.01)
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton
                and self._edit_drag_idx is not None):
            self._edit_drag_idx = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.edit_committed.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._draw_dragging \
                and self._draw_mode is not None:
            mode = self._draw_mode
            vertices: List[Tuple[float, float]] = []
            if mode in ("rect", "ellipse"):
                if self._draw_drag_start is not None and self._draw_drag_end is not None:
                    sy, sx = self._draw_drag_start
                    ey, ex = self._draw_drag_end
                    # Require a minimum drag distance — a single click is not a
                    # shape, and emitting a zero-size bounding box would create
                    # an empty mask label.
                    if abs(ex - sx) > 3 or abs(ey - sy) > 3:
                        vertices = [(sy, sx), (ey, ex)]
            else:  # polygon
                if len(self._poly_vertices) >= 3:
                    vertices = list(self._poly_vertices)
            self._clear_draw_state()
            self.update()
            if vertices:
                self.shape_drawn.emit(mode, vertices)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._crop_dragging:
            self._crop_dragging = False
            if self._crop_drag_start and self._crop_drag_end:
                start_iy, start_ix = self._crop_drag_start
                end_iy, end_ix = self._crop_drag_end
                dx = abs(end_ix - start_ix)
                dy = abs(end_iy - start_iy)
                if dx > 3 or dy > 3:
                    # Significant drag — emit crop rect
                    x = max(0, int(min(start_ix, end_ix)))
                    y = max(0, int(min(start_iy, end_iy)))
                    w = max(1, int(abs(end_ix - start_ix)))
                    h = max(1, int(abs(end_iy - start_iy)))
                    w = min(w, self._img_w - x)
                    h = min(h, self._img_h - y)
                    self._crop_drag_start = None
                    self._crop_drag_end = None
                    self.update()
                    self.crop_rect_selected.emit(x, y, w, h)
                else:
                    # Single click — emit clicked as normal
                    iy, ix = self._crop_drag_start
                    self._crop_drag_start = None
                    self._crop_drag_end = None
                    self.update()
                    if 0 <= ix < self._img_w and 0 <= iy < self._img_h:
                        self.clicked.emit(iy, ix)
            event.accept()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = False
        super().mouseReleaseEvent(event)

    def widget_to_image(self, wx: float, wy: float) -> Tuple[float, float]:
        ix = (wx - self._offset.x()) / self._scale
        iy = (wy - self._offset.y()) / self._scale
        return iy, ix

    def image_to_widget(self, iy: float, ix: float) -> Tuple[float, float]:
        return ix * self._scale + self._offset.x(), iy * self._scale + self._offset.y()


class ImageViewer(QWidget):
    """
    Image viewer supporting:
    - Single-channel mode: set_data(array) with color combo
    - Multi-channel mode: set_channels(dict) with per-channel colors + toggles
    - T (time) slider, auto-contrast, overlay callback
    """

    frame_changed = Signal(int)

    def __init__(self, parent=None, show_controls=True):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, stretch=1)

        self._show_controls = show_controls

        # Zoom toolbar (Home / + / -) — always present on every viewer
        zoom_row = QHBoxLayout()
        zoom_row.setContentsMargins(4, 0, 4, 0)
        self.zoom_toolbar = ZoomToolbar(self.canvas)
        zoom_row.addWidget(self.zoom_toolbar)
        zoom_row.addStretch(1)
        layout.addLayout(zoom_row)

        if show_controls:
            ctrl = QHBoxLayout()
            ctrl.setContentsMargins(4, 0, 4, 4)

            ctrl.addWidget(QLabel("T:"))
            self.t_slider = QSlider(Qt.Orientation.Horizontal)
            self.t_slider.setRange(0, 0)
            self.t_slider.valueChanged.connect(self._on_t_changed)
            ctrl.addWidget(self.t_slider, stretch=1)
            self.t_label = QLabel("0/0")
            self.t_label.setMinimumWidth(55)
            ctrl.addWidget(self.t_label)

            self.auto_contrast_cb = QCheckBox("Auto")
            self.auto_contrast_cb.setChecked(True)
            self.auto_contrast_cb.stateChanged.connect(self._refresh)
            ctrl.addWidget(self.auto_contrast_cb)

            layout.addLayout(ctrl)
        else:
            self.t_slider = None
            self.t_label = None
            self.auto_contrast_cb = None

        # Data — supports single-channel or multi-channel
        self._data: Optional[np.ndarray] = None              # single: (T, H, W)
        self._channels: Optional[Dict[str, np.ndarray]] = None  # multi: {name: (T,H,W)}
        self._channel_colors: Dict[str, Tuple[int, int, int]] = {}
        self._channel_enabled: Dict[str, bool] = {}
        self._current_t = 0
        self._overlay_fn: Optional[Callable] = None

    # ── Single-channel API (backward compatible) ──

    def set_data(self, data: Optional[np.ndarray], color: Tuple[int, int, int] = (255, 255, 255)):
        """Set single-channel timeseries as (T, H, W)."""
        self._data = data
        self._channels = None
        self._channel_colors = {"ch0": color}
        self._channel_enabled = {"ch0": True}

        if data is None:
            self.canvas._source_pixmap = None
            self.canvas.update()
            if self.t_slider:
                self.t_slider.setRange(0, 0)
                self.t_label.setText("0/0")
            return

        T = data.shape[0]
        if self.t_slider:
            self.t_slider.setRange(0, max(0, T - 1))
            self.t_slider.setValue(0)
        self._current_t = 0
        self._refresh()

    # ── Multi-channel API ──

    def set_channels(
        self,
        channels: Dict[str, np.ndarray],
        colors: Dict[str, Tuple[int, int, int]],
        enabled: Optional[Dict[str, bool]] = None,
    ):
        """
        Set multi-channel timeseries for RGB compositing.

        Parameters
        ----------
        channels : dict mapping name -> (T, H, W) array
        colors : dict mapping name -> (R, G, B) tuple
        enabled : dict mapping name -> bool (default all True)
        """
        self._data = None
        self._channels = channels
        self._channel_colors = dict(colors)
        self._channel_enabled = enabled or {k: True for k in channels}

        if not channels:
            self.canvas._source_pixmap = None
            self.canvas.update()
            return

        sample = next(iter(channels.values()))
        T = sample.shape[0]
        if self.t_slider:
            self.t_slider.setRange(0, max(0, T - 1))
            self.t_slider.setValue(0)
        self._current_t = 0
        self._refresh()

    def update_channel_color(self, ch_name: str, color: Tuple[int, int, int]):
        self._channel_colors[ch_name] = color
        self._refresh()

    def update_channel_enabled(self, ch_name: str, enabled: bool):
        self._channel_enabled[ch_name] = enabled
        self._refresh()

    # ── Common API ──

    def set_frame_index(self, t: int):
        n = self.n_frames
        if n == 0:
            return
        t = max(0, min(t, n - 1))
        if self.t_slider:
            self.t_slider.blockSignals(True)
            self.t_slider.setValue(t)
            self.t_slider.blockSignals(False)
        self._current_t = t
        self._refresh()

    def set_overlay(self, fn: Optional[Callable]):
        self._overlay_fn = fn
        self.canvas.set_overlay(fn)

    def get_current_frame(self) -> Optional[np.ndarray]:
        if self._data is not None:
            return self._data[self._current_t]
        return None

    @property
    def current_t(self) -> int:
        return self._current_t

    @property
    def n_frames(self) -> int:
        if self._data is not None:
            return self._data.shape[0]
        if self._channels:
            return next(iter(self._channels.values())).shape[0]
        return 0

    def _on_t_changed(self, val: int):
        self._current_t = val
        if self.t_label:
            self.t_label.setText(f"{val}/{self.t_slider.maximum()}")
        self._refresh()
        self.frame_changed.emit(val)

    def _refresh(self):
        auto = self.auto_contrast_cb.isChecked() if self.auto_contrast_cb else True
        t = self._current_t

        if self._channels:
            # Multi-channel composite
            frames = {}
            for ch_name, ch_data in self._channels.items():
                if t < ch_data.shape[0]:
                    frames[ch_name] = ch_data[t]
            if frames:
                rgb = composite_channels(frames, self._channel_colors,
                                         self._channel_enabled, auto)
                self.canvas.set_image(rgb)
            return

        if self._data is not None:
            frame = self._data[t]
            if auto:
                gray = frame_to_uint8(frame)
            else:
                if np.issubdtype(frame.dtype, np.integer):
                    mx = np.iinfo(frame.dtype).max
                else:
                    mx = frame.max() if frame.max() > 0 else 1.0
                gray = (frame.astype(np.float32) / mx * 255).astype(np.uint8)

            color = self._channel_colors.get("ch0", (255, 255, 255))
            if color == (255, 255, 255):
                rgb = np.stack([gray, gray, gray], axis=-1)
            else:
                rgb = np.clip(apply_lut_color(gray, color), 0, 255).astype(np.uint8)
            self.canvas.set_image(rgb)
