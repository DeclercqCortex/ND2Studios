"""
``LutSidebar`` — collapsible right-side panel that hosts:

- a **Tiles** section with an interactive :class:`TileLayoutWidget`
  (V1.3 addition; click a tile to navigate to that M),
- a **LUTs** section with one :class:`LutHistogramWidget` per channel
  (V1.1).

Both sections have their own collapse toggle so users with many
channels or many tiles can independently hide whichever they aren't
using. The whole sidebar also collapses 280 px → 28 px via
``QPropertyAnimation`` (matches the main app sidebar).

Signals
-------
- ``channel_contrast_changed(name, lo, hi, gamma)`` — forwarded from
  the per-channel LUT widgets.
- ``tile_navigate_requested(m)`` — forwarded from the tile widget.
- ``collapse_changed(bool)`` — sidebar expand/collapse toggle.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import (
    QEasingCurve, QPropertyAnimation, Qt, Signal,
)
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from nd2studios.core.settings import Settings
from nd2studios.widgets.image_viewer import CHANNEL_COLORS
from nd2studios.widgets.lut_histogram import LutHistogramWidget
from nd2studios.widgets.tile_layout import TileLayoutDialog, TileLayoutWidget


class _SectionHeader(QFrame):
    """Reusable collapsible section header (title + ▶/◀ toggle)."""

    toggled = Signal(bool)

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(24)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 4, 0)
        layout.setSpacing(4)
        self._lbl = QLabel(title)
        self._lbl.setStyleSheet(
            f"color: {Settings.ACCENT_CYAN}; font: bold 9pt 'Helvetica Neue';")
        layout.addWidget(self._lbl)
        layout.addStretch(1)
        self._btn = QPushButton("▾")
        self._btn.setFixedSize(20, 20)
        self._btn.setToolTip(f"Collapse / expand the {title} section")
        self._btn.setStyleSheet(
            "QPushButton { background: transparent; "
            f"color: {Settings.FG_SECONDARY}; "
            "border: none; padding: 0; font: 10pt; }"
            "QPushButton:hover { color: " + Settings.FG_PRIMARY + "; }"
        )
        self._btn.clicked.connect(self._on_clicked)
        layout.addWidget(self._btn)
        self._expanded = True

    def _on_clicked(self) -> None:
        self._expanded = not self._expanded
        self._btn.setText("▾" if self._expanded else "▸")
        self.toggled.emit(self._expanded)

    @property
    def is_expanded(self) -> bool:
        return self._expanded


class LutSidebar(QFrame):
    """Collapsible right-side panel: Tiles + LUTs."""

    EXPANDED_WIDTH = 300
    COLLAPSED_WIDTH = 28
    ANIMATION_MS = 260
    TILE_WIDGET_HEIGHT = 240   # default vertical room for the tile preview

    channel_contrast_changed = Signal(str, float, float, float)
    tile_navigate_requested = Signal(int)
    collapse_changed = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("contentArea")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumWidth(self.EXPANDED_WIDTH)
        self.setMaximumWidth(self.EXPANDED_WIDTH)

        self._expanded = True
        self._lut_widgets: Dict[str, LutHistogramWidget] = {}
        self._anim_min: Optional[QPropertyAnimation] = None
        self._anim_max: Optional[QPropertyAnimation] = None

        # Tile state cached so an "expand" modal can mirror it.
        self._tile_state: Dict[str, Any] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top bar with whole-sidebar collapse toggle.
        top_bar = QFrame()
        top_bar.setFixedHeight(28)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 0, 4, 0)
        top_layout.setSpacing(4)
        self._title = QLabel("Panel")
        self._title.setStyleSheet(
            f"color: {Settings.ACCENT_PURPLE}; font: bold 9pt 'Helvetica Neue';")
        top_layout.addWidget(self._title)
        top_layout.addStretch(1)
        self._collapse_btn = QPushButton("▶")
        self._collapse_btn.setFixedSize(20, 20)
        self._collapse_btn.setToolTip("Collapse / expand the side panel")
        self._collapse_btn.setStyleSheet(
            "QPushButton { background: transparent; "
            f"color: {Settings.FG_SECONDARY}; "
            "border: none; padding: 0; font: 10pt; }"
            "QPushButton:hover { color: " + Settings.FG_PRIMARY + "; }"
        )
        self._collapse_btn.clicked.connect(self.toggle)
        top_layout.addWidget(self._collapse_btn)
        outer.addWidget(top_bar)

        # Scroll area — both Tiles and LUTs sections live inside.
        self._body = QScrollArea()
        self._body.setWidgetResizable(True)
        # V1.6: allow horizontal scrolling so wide tile mosaics can
        # extend past the sidebar width without getting squashed.
        self._body.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._inner = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(6, 4, 6, 4)
        self._inner_layout.setSpacing(8)

        # Tiles section.
        self._tiles_header = _SectionHeader("Tiles")
        self._tiles_header.toggled.connect(self._on_tiles_toggled)
        self._inner_layout.addWidget(self._tiles_header)
        self.tile_widget = TileLayoutWidget(
            mode="navigate",
            show_expand_button=True,
            minimum_size=(240, 200),
        )
        # V1.6: drop the fixed height — the widget now sets its own
        # natural minimum size from the tile count, and the sidebar's
        # QScrollArea handles overflow.
        self.tile_widget.navigate_requested.connect(
            self.tile_navigate_requested.emit)
        self.tile_widget.expand_requested.connect(self._open_tile_dialog)
        self.tile_widget.hide()
        self._inner_layout.addWidget(self.tile_widget)

        # LUTs section.
        self._lut_header = _SectionHeader("LUTs")
        self._lut_header.toggled.connect(self._on_luts_toggled)
        self._inner_layout.addWidget(self._lut_header)
        self._lut_container = QWidget()
        self._lut_container_layout = QVBoxLayout(self._lut_container)
        self._lut_container_layout.setContentsMargins(0, 0, 0, 0)
        self._lut_container_layout.setSpacing(8)
        self._inner_layout.addWidget(self._lut_container)

        self._inner_layout.addStretch(1)
        self._body.setWidget(self._inner)
        outer.addWidget(self._body, stretch=1)

        self._update_collapse_glyph()

    # ── Whole-sidebar collapse ──
    @property
    def is_expanded(self) -> bool:
        return self._expanded

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, value: bool) -> None:
        if value == self._expanded:
            return
        self._expanded = bool(value)
        target = self.EXPANDED_WIDTH if self._expanded else self.COLLAPSED_WIDTH
        for prop, attr in (("minimumWidth", "_anim_min"),
                            ("maximumWidth", "_anim_max")):
            anim = QPropertyAnimation(self, prop.encode("utf-8"))
            anim.setDuration(self.ANIMATION_MS)
            anim.setStartValue(self.width())
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.Type.InOutQuart)
            anim.start()
            setattr(self, attr, anim)
        self._body.setVisible(self._expanded)
        self._title.setVisible(self._expanded)
        self._update_collapse_glyph()
        self.collapse_changed.emit(self._expanded)

    def _update_collapse_glyph(self) -> None:
        self._collapse_btn.setText("▶" if self._expanded else "◀")

    # ── Section toggles ──
    def _on_tiles_toggled(self, expanded: bool) -> None:
        # Hide just the tile widget; the header stays visible so the
        # user can re-open the section.
        self.tile_widget.setVisible(expanded and bool(self._tile_state))

    def _on_luts_toggled(self, expanded: bool) -> None:
        self._lut_container.setVisible(expanded)

    # ── Tile data ──
    def set_tile_layout(self,
                        stage_xy_um: List[Tuple[float, float]],
                        pixel_size_um: float,
                        tile_h: int, tile_w: int,
                        n_multipoints: int,
                        current_m: int = 0) -> None:
        """Configure the embedded tile widget. Hide the section when there
        is nothing useful to show (≤ 1 tile)."""
        if n_multipoints <= 1:
            self._tile_state = {}
            self.tile_widget.hide()
            self._tiles_header.hide()
            return

        m_indices = list(range(n_multipoints))
        self._tile_state = {
            "stage_xy_um": list(stage_xy_um or []),
            "pixel_size_um": float(pixel_size_um),
            "tile_h": int(tile_h),
            "tile_w": int(tile_w),
            "m_indices": m_indices,
            "current_m": int(current_m),
        }
        self.tile_widget.set_tile_layout(
            stage_xy_um=stage_xy_um or [],
            pixel_size_um=pixel_size_um,
            tile_h=tile_h, tile_w=tile_w,
            m_indices=m_indices, current_m=current_m,
        )
        self._tiles_header.show()
        self.tile_widget.setVisible(self._tiles_header.is_expanded)

    def set_current_m(self, m: int) -> None:
        if not self._tile_state:
            return
        self._tile_state["current_m"] = int(m)
        self.tile_widget.set_current_m(int(m))

    def _open_tile_dialog(self) -> None:
        if not self._tile_state:
            return
        dlg = TileLayoutDialog(
            mode="navigate",
            stage_xy_um=self._tile_state["stage_xy_um"],
            pixel_size_um=self._tile_state["pixel_size_um"],
            tile_h=self._tile_state["tile_h"],
            tile_w=self._tile_state["tile_w"],
            m_indices=self._tile_state["m_indices"],
            current_m=self._tile_state["current_m"],
            parent=self.window(),
        )
        dlg.navigate_requested.connect(self.tile_navigate_requested.emit)
        dlg.exec()

    # ── LUT population ──
    def rebuild(self,
                channel_names: List[str],
                channel_display: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Replace the LUT widgets with a fresh set, one per channel."""
        for w in list(self._lut_widgets.values()):
            w.setParent(None)
            w.deleteLater()
        self._lut_widgets.clear()
        # Drop everything in the container.
        while self._lut_container_layout.count():
            item = self._lut_container_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        cycle = ["green", "red", "cyan", "magenta", "yellow", "blue", "orange"]
        for i, name in enumerate(channel_names):
            cd = (channel_display or {}).get(name, {})
            color_name = cd.get("color", cycle[i % len(cycle)])
            rgb = CHANNEL_COLORS.get(color_name, (255, 255, 255))

            block = QWidget()
            block_layout = QVBoxLayout(block)
            block_layout.setContentsMargins(0, 0, 0, 0)
            block_layout.setSpacing(2)

            head = QFrame()
            head_layout = QHBoxLayout(head)
            head_layout.setContentsMargins(0, 0, 0, 0)
            head_layout.setSpacing(6)
            swatch = QFrame()
            swatch.setFixedSize(10, 10)
            swatch.setStyleSheet(
                f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});"
                f"border: 1px solid {Settings.BORDER_COLOR};"
                f"border-radius: 2px;"
            )
            head_layout.addWidget(swatch)
            label = QLabel(name)
            label.setStyleSheet(
                f"color: {Settings.FG_PRIMARY}; font: 9pt 'Helvetica Neue';")
            head_layout.addWidget(label, stretch=1)
            block_layout.addWidget(head)

            lut = LutHistogramWidget(name)
            lut.contrast_changed.connect(
                lambda lo, hi, g, n=name: self.channel_contrast_changed.emit(n, lo, hi, g))
            block_layout.addWidget(lut)
            self._lut_widgets[name] = lut

            self._lut_container_layout.addWidget(block)

    def lut_for(self, name: str) -> Optional[LutHistogramWidget]:
        return self._lut_widgets.get(name)

    def update_swatch(self, name: str, rgb_tuple) -> None:
        """When a channel's color combo changes, repaint its sidebar swatch."""
        for i in range(self._lut_container_layout.count()):
            w = self._lut_container_layout.itemAt(i).widget()
            if w is None:
                continue
            label_widgets = w.findChildren(QLabel)
            for lbl in label_widgets:
                if lbl.text() == name:
                    swatches = w.findChildren(QFrame)
                    for s in swatches:
                        if s.width() == 10 and s.height() == 10:
                            s.setStyleSheet(
                                f"background-color: rgb({rgb_tuple[0]},"
                                f" {rgb_tuple[1]}, {rgb_tuple[2]});"
                                f"border: 1px solid {Settings.BORDER_COLOR};"
                                f"border-radius: 2px;"
                            )
                            return
