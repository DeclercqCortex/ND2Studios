"""
``TileLayoutWidget`` and ``TileLayoutDialog`` — interactive multipoint
layout for navigation (sidebar) and stitch selection (dialog).

The widget paints the M-tile arrangement using
:func:`compute_tile_layout` from the stitch exporter, so what's drawn
is exactly what the stitcher produces. Two interaction modes:

- ``"navigate"``: click a tile → emit ``navigate_requested(m: int)``.
  The viewer subscribes and sets the M slider to ``m``.
- ``"select"``: click a tile → toggle membership in
  ``selected_indices``; emit ``selection_changed(set[int])``. Used by
  ``StitchDialog`` to pick which tiles go into the stitch.

Both modes always show the *current* M as a highlighted ring so the
user can see where they are even while selecting.

A small ``⤢`` button at the top-right emits ``expand_requested``;
the host opens a :class:`TileLayoutDialog` (a modal containing the
same widget at a much larger size) so users with hundreds of tiles
can find what they're looking for.
"""
from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QPainter, QPen,
)
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from nd2studios.backend.exporters.stitch_exporter import (
    StitchLayout, compute_tile_layout,
)
from nd2studios.core.settings import Settings


_MARGIN = 8
_HEADER_H = 22
_EXPAND_BTN_SIZE = 22
# Target on-screen pixels per tile when sizing the widget. Comfortable
# for 2-digit M labels; the user can hit `⤢` for an even bigger view.
_TARGET_TILE_PX = 50


class TileLayoutWidget(QFrame):
    """Interactive M-tile layout widget."""

    navigate_requested = Signal(int)               # m index clicked (navigate mode)
    selection_changed = Signal(set)                # set[int] of selected M indices (select mode)
    expand_requested = Signal()                    # ⤢ button clicked

    def __init__(self,
                 mode: str = "navigate",
                 parent: Optional[QWidget] = None,
                 show_expand_button: bool = True,
                 minimum_size: Tuple[int, int] = (240, 200)):
        super().__init__(parent)
        if mode not in ("navigate", "select"):
            raise ValueError(f"Unknown mode {mode!r}")
        self._mode = mode
        self._show_expand_button = show_expand_button

        self.setMinimumSize(*minimum_size)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"background-color: rgba(33, 37, 43, 220);"
            f"border: 1px solid {Settings.BORDER_COLOR};"
            f"border-radius: 4px;"
        )

        # Layout state.
        self._stage_xy_um: List[Tuple[float, float]] = []
        self._pixel_size_um: float = 1.0
        self._tile_h: int = 0
        self._tile_w: int = 0
        self._m_indices: List[int] = []
        self._current_m: int = 0
        self._selected_indices: Set[int] = set()
        self._layout: Optional[StitchLayout] = None

        # Cached widget-pixel rects per tile (recomputed on every paint).
        self._tile_rects: List[Tuple[int, QRect]] = []  # [(m, rect), ...]
        self._hovered_m: Optional[int] = None

        # Optional expand button (a child QPushButton positioned in the
        # top-right corner; cheaper than painting + custom hit-testing).
        if self._show_expand_button:
            self._btn_expand = QPushButton("⤢", self)
            self._btn_expand.setFixedSize(_EXPAND_BTN_SIZE, _EXPAND_BTN_SIZE)
            self._btn_expand.setToolTip("Expand tile layout")
            self._btn_expand.setStyleSheet(
                "QPushButton { background: transparent; border: none; "
                f"color: {Settings.FG_SECONDARY}; "
                "padding: 0; font: 11pt; }"
                "QPushButton:hover { color: " + Settings.ACCENT_PURPLE + "; }"
            )
            self._btn_expand.clicked.connect(self.expand_requested.emit)
        else:
            self._btn_expand = None

    # ── Mode ──
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("navigate", "select"):
            raise ValueError(f"Unknown mode {mode!r}")
        self._mode = mode
        self.update()

    # ── Tile data ──
    def set_tile_layout(self,
                        stage_xy_um: List[Tuple[float, float]],
                        pixel_size_um: float,
                        tile_h: int, tile_w: int,
                        m_indices: Optional[List[int]] = None,
                        current_m: int = 0,
                        selected_indices: Optional[Set[int]] = None) -> None:
        """Rebuild the layout from stage XY (or grid fallback)."""
        self._stage_xy_um = list(stage_xy_um or [])
        self._pixel_size_um = max(float(pixel_size_um), 1e-6)
        self._tile_h = int(max(1, tile_h))
        self._tile_w = int(max(1, tile_w))
        if m_indices is None:
            m_indices = list(range(len(self._stage_xy_um)))
        self._m_indices = list(m_indices)
        self._current_m = int(current_m)
        if selected_indices is not None:
            self._selected_indices = set(selected_indices)
        self._layout = compute_tile_layout(
            stage_xy_um=self._stage_xy_um,
            pixel_size_um=self._pixel_size_um,
            tile_h=self._tile_h, tile_w=self._tile_w,
            m_indices=self._m_indices,
        )
        # Auto-size to a readable per-tile pixel target. The host
        # (LutSidebar / TileLayoutDialog) wraps us in a QScrollArea, so
        # exceeding the visible viewport just engages scrollbars.
        self._adapt_minimum_size()
        self.update()

    def _adapt_minimum_size(self) -> None:
        if self._layout is None or self._layout.tile_w <= 0:
            return
        # Size the widget based on the actual number of tiles, not the canvas
        # bounding box. For physical layouts, the bounding box can be many
        # times larger than the tile count (e.g. 2× tile spacing gives a
        # canvas 13 tiles wide for a 7-tile-wide row), which would produce an
        # absurdly wide minimum size that overflows the sidebar.
        n = max(1, len(self._m_indices))
        n_cols = max(1, math.ceil(math.sqrt(n)))
        n_rows = max(1, math.ceil(n / n_cols))
        natural_w = n_cols * _TARGET_TILE_PX + 2 * _MARGIN
        natural_h = _HEADER_H + 4 + n_rows * _TARGET_TILE_PX + _MARGIN + 2
        self.setMinimumSize(max(240, natural_w), max(180, natural_h))

    def set_current_m(self, m: int) -> None:
        if int(m) == self._current_m:
            return
        self._current_m = int(m)
        self.update()

    def selected_indices(self) -> Set[int]:
        return set(self._selected_indices)

    def set_selected_indices(self, indices: Set[int]) -> None:
        self._selected_indices = set(indices)
        self.update()
        if self._mode == "select":
            self.selection_changed.emit(self.selected_indices())

    def is_renderable(self) -> bool:
        return bool(self._layout) and len(self._m_indices) > 0

    # ── Painting ──
    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        # Reposition the corner expand button.
        if self._btn_expand is not None:
            self._btn_expand.move(
                self.width() - _EXPAND_BTN_SIZE - 4,
                4,
            )

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().paintEvent(event)
        if self._layout is None or not self._m_indices:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Header: mode-aware title.
        title_font = QFont("Helvetica Neue", 8)
        painter.setFont(title_font)
        fm = QFontMetrics(title_font)
        title_text = (
            f"{len(self._m_indices)} tiles · click to jump"
            if self._mode == "navigate" else
            f"{len(self._selected_indices)} / {len(self._m_indices)}"
            f" tiles selected · click to toggle"
        )
        painter.setPen(QPen(QColor(Settings.FG_SECONDARY)))
        painter.drawText(
            QPoint(_MARGIN, _HEADER_H - 6), title_text,
        )

        # Available rect for tiles.
        avail = QRect(
            _MARGIN,
            _HEADER_H + 4,
            self.width() - 2 * _MARGIN,
            self.height() - _HEADER_H - 2 * _MARGIN,
        )
        if avail.width() <= 4 or avail.height() <= 4:
            painter.end()
            return

        canvas_w = max(1, self._layout.canvas_w)
        canvas_h = max(1, self._layout.canvas_h)
        s = min(avail.width() / canvas_w, avail.height() / canvas_h)
        ox = avail.x() + (avail.width() - canvas_w * s) / 2
        oy = avail.y() + (avail.height() - canvas_h * s) / 2

        # Recompute and cache widget-pixel rects.
        self._tile_rects.clear()
        idx_font = QFont("Helvetica Neue", 7)
        idx_fm = QFontMetrics(idx_font)
        for offset, m in zip(self._layout.offsets, self._m_indices):
            ty, tx = offset
            rx = int(ox + tx * s)
            ry = int(oy + ty * s)
            rw = max(2, int(self._layout.tile_w * s))
            rh = max(2, int(self._layout.tile_h * s))
            rect = QRect(rx, ry, rw, rh)
            self._tile_rects.append((m, rect))

            is_selected = (self._mode == "select"
                           and m in self._selected_indices)
            is_current = (m == self._current_m)
            is_hovered = (m == self._hovered_m)

            # Fill colour.
            if is_selected:
                fill = QColor(Settings.ACCENT_GREEN)
                fill.setAlpha(95)
            elif is_current:
                fill = QColor(Settings.ACCENT_PURPLE)
                fill.setAlpha(70)
            elif is_hovered:
                fill = QColor(Settings.ACCENT_CYAN)
                fill.setAlpha(40)
            else:
                fill = QColor(Settings.BG_HOVER)
                fill.setAlpha(60)

            # Outline.
            if is_selected:
                pen_color = Settings.ACCENT_GREEN
                pen_width = 2
            elif is_current:
                pen_color = Settings.ACCENT_PURPLE
                pen_width = 2
            elif is_hovered:
                pen_color = Settings.ACCENT_CYAN
                pen_width = 1
            else:
                pen_color = Settings.FG_SECONDARY
                pen_width = 1

            painter.setBrush(fill)
            pen = QPen(QColor(pen_color))
            pen.setWidth(pen_width)
            painter.setPen(pen)
            painter.drawRect(rect)

            # Tile label inside the rect when there's room (≥18 px).
            if rw >= 18 and rh >= idx_fm.height():
                painter.setFont(idx_font)
                label = str(m)
                tw = idx_fm.horizontalAdvance(label)
                if tw <= rw - 4:
                    painter.setPen(QPen(QColor(Settings.FG_PRIMARY)))
                    painter.drawText(
                        QPoint(rx + (rw - tw) // 2,
                               ry + (rh + idx_fm.ascent()) // 2 - 1),
                        label,
                    )

        painter.end()

    # ── Mouse interaction ──
    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() != Qt.LeftButton or not self.is_renderable():
            super().mousePressEvent(event)
            return
        m = self._tile_at(event.position().toPoint())
        if m is None:
            super().mousePressEvent(event)
            return
        if self._mode == "navigate":
            self.navigate_requested.emit(int(m))
        else:  # select
            if m in self._selected_indices:
                self._selected_indices.remove(m)
            else:
                self._selected_indices.add(m)
            self.update()
            self.selection_changed.emit(self.selected_indices())
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        m = self._tile_at(event.position().toPoint())
        if m != self._hovered_m:
            self._hovered_m = m
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._hovered_m is not None:
            self._hovered_m = None
            self.update()

    def _tile_at(self, pt: QPoint) -> Optional[int]:
        # Reverse iteration so the topmost-painted tile wins on overlap.
        for m, rect in reversed(self._tile_rects):
            if rect.contains(pt):
                return m
        return None


class TileLayoutDialog(QDialog):
    """Modal "expand" view of a TileLayoutWidget.

    In ``navigate`` mode the dialog auto-closes after the user clicks
    a tile (one-shot "find that tile"). In ``select`` mode it stays
    open so the user can build up a selection iteratively; it forwards
    ``selection_changed`` continuously.
    """

    navigate_requested = Signal(int)
    selection_changed = Signal(set)

    def __init__(self,
                 mode: str = "navigate",
                 stage_xy_um: Optional[List[Tuple[float, float]]] = None,
                 pixel_size_um: float = 1.0,
                 tile_h: int = 0, tile_w: int = 0,
                 m_indices: Optional[List[int]] = None,
                 current_m: int = 0,
                 selected_indices: Optional[Set[int]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Tile layout")
        self.setMinimumSize(720, 600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.widget = TileLayoutWidget(
            mode=mode, parent=None,
            show_expand_button=False,
            minimum_size=(680, 520),
        )
        self.widget.set_tile_layout(
            stage_xy_um=stage_xy_um or [],
            pixel_size_um=pixel_size_um,
            tile_h=tile_h, tile_w=tile_w,
            m_indices=m_indices,
            current_m=current_m,
            selected_indices=selected_indices,
        )
        # Wrap the widget in a scroll area so layouts bigger than the
        # modal don't get squashed — the V1.6 widget auto-sizes to a
        # readable per-tile pixel target which can easily exceed
        # 720 × 600 for mosaics with many tiles.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(self.widget)
        layout.addWidget(scroll, stretch=1)

        # Wire signals.
        self.widget.navigate_requested.connect(self._on_navigate)
        self.widget.selection_changed.connect(self.selection_changed.emit)

        # Footer.
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_navigate(self, m: int) -> None:
        self.navigate_requested.emit(int(m))
        # In navigate mode, close right after — one-shot UX.
        if self.widget.mode == "navigate":
            self.accept()

    def selected_indices(self) -> Set[int]:
        return self.widget.selected_indices()
