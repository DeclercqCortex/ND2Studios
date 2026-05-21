"""
``TilePreviewWidget`` — corner thumbnail showing how the M tiles would
arrange themselves if stitched.

Painted directly with ``QPainter`` so the redraw on every M-slider
move stays cheap. Reuses :func:`compute_tile_layout` from the
stitch exporter so what the user sees here is exactly what they'd
get from "Stitch M…".

Behaviour
---------
- Auto-positioned by :class:`MultiAxisViewer` in the bottom-right
  corner of the image canvas. Hidden when there is only one tile or
  the file lacks usable stage XY positions.
- Clicking emits ``stitch_requested`` so the Import page can launch
  the stitch dialog without the user having to hunt for the button.
- The currently displayed M is highlighted in
  ``Settings.ACCENT_PURPLE``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QPainter, QPen,
)
from PySide6.QtWidgets import QFrame, QToolTip

from nd2studios.backend.exporters.stitch_exporter import (
    StitchLayout, compute_tile_layout,
)
from nd2studios.core.settings import Settings


class TilePreviewWidget(QFrame):
    """Tiny thumbnail of the M-tile arrangement for click-to-stitch."""

    stitch_requested = Signal()

    def __init__(self, parent=None,
                 width: int = 160, height: int = 120,
                 margin_px: int = 8):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.margin_px = margin_px
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"background-color: rgba(33, 37, 43, 220);"
            f"border: 1px solid {Settings.BORDER_COLOR};"
            f"border-radius: 4px;"
        )
        self.setToolTip(
            "Tile layout preview — click to open the stitch dialog.\n"
            "Highlighted rectangle is the currently displayed M position."
        )

        self._stage_xy_um: List[Tuple[float, float]] = []
        self._pixel_size_um: float = 1.0
        self._tile_h: int = 0
        self._tile_w: int = 0
        self._m_indices: List[int] = []
        self._current_m: int = 0
        self._layout: Optional[StitchLayout] = None

    # ── Public API ──
    def update_layout(self,
                      stage_xy_um: List[Tuple[float, float]],
                      pixel_size_um: float,
                      tile_h: int, tile_w: int,
                      m_indices: Optional[List[int]] = None,
                      current_m: int = 0) -> None:
        """Recompute the tile layout and repaint."""
        self._stage_xy_um = list(stage_xy_um or [])
        self._pixel_size_um = max(float(pixel_size_um), 1e-6)
        self._tile_h = int(max(1, tile_h))
        self._tile_w = int(max(1, tile_w))
        if m_indices is None:
            m_indices = list(range(len(self._stage_xy_um)))
        self._m_indices = list(m_indices)
        self._current_m = int(current_m)

        # Compute the layout once; we rescale it to the widget rect on paint.
        self._layout = compute_tile_layout(
            stage_xy_um=self._stage_xy_um,
            pixel_size_um=self._pixel_size_um,
            tile_h=self._tile_h,
            tile_w=self._tile_w,
            m_indices=self._m_indices,
        )
        self.update()

    def set_current_m(self, m: int) -> None:
        """Highlight a different tile without recomputing the layout."""
        if int(m) == self._current_m:
            return
        self._current_m = int(m)
        self.update()

    def is_renderable(self) -> bool:
        """``True`` if there's a meaningful preview to show.

        We hide the widget when there's only one tile or zero stage
        positions — there's nothing to preview otherwise.
        """
        return bool(self._layout) and len(self._m_indices) > 1

    # ── Painting ──
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().paintEvent(event)
        if self._layout is None or not self._m_indices:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Inset for the title strip at top.
        title = "Tile layout"
        font = QFont("Helvetica Neue", 8)
        painter.setFont(font)
        fm = QFontMetrics(font)
        title_h = fm.height() + 4
        painter.setPen(QPen(QColor(Settings.FG_SECONDARY)))
        painter.drawText(
            QPoint(self.margin_px, title_h - 2),
            f"{title}  ({len(self._m_indices)} tiles · click to stitch)",
        )

        # Map the layout canvas into the remaining widget rect with margin.
        avail = QRect(
            self.margin_px,
            title_h + 2,
            self.width() - 2 * self.margin_px,
            self.height() - title_h - 2 * self.margin_px,
        )
        if avail.width() <= 4 or avail.height() <= 4:
            painter.end()
            return

        canvas_w = max(1, self._layout.canvas_w)
        canvas_h = max(1, self._layout.canvas_h)
        sx = avail.width() / canvas_w
        sy = avail.height() / canvas_h
        s = min(sx, sy)
        # Centre the tile bbox inside the available rect.
        ox = avail.x() + (avail.width() - canvas_w * s) / 2
        oy = avail.y() + (avail.height() - canvas_h * s) / 2

        # Draw tiles.
        for offset, m in zip(self._layout.offsets, self._m_indices):
            y, x = offset
            rx = ox + x * s
            ry = oy + y * s
            rw = self._layout.tile_w * s
            rh = self._layout.tile_h * s
            highlight = (m == self._current_m)
            pen_color = (Settings.ACCENT_PURPLE
                         if highlight else Settings.FG_SECONDARY)
            fill_color = QColor(Settings.ACCENT_PURPLE)
            fill_color.setAlpha(70 if highlight else 30)
            pen = QPen(QColor(pen_color))
            pen.setWidth(2 if highlight else 1)
            painter.setPen(pen)
            painter.setBrush(fill_color)
            painter.drawRect(int(rx), int(ry),
                              max(1, int(rw)), max(1, int(rh)))

        # Tiny tag for the current M index in the corner of the highlight.
        try:
            cur_offset = self._layout.offsets[
                self._m_indices.index(self._current_m)]
            cy, cx = cur_offset
            label_x = ox + cx * s + 2
            label_y = oy + cy * s + fm.ascent() + 1
            painter.setPen(QPen(QColor(Settings.ACCENT_PURPLE)))
            painter.drawText(QPoint(int(label_x), int(label_y)),
                              f"M{self._current_m}")
        except ValueError:
            pass

        painter.end()

    # ── Click ──
    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.LeftButton and self.is_renderable():
            self.stitch_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)
