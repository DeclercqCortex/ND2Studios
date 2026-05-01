"""
``StitchDialog`` — opened from the Import page to stitch multipoint
tiles into a single TIFF stack without modifying the original file.

Layout:

    ┌──────────────────────────────────────────────┐
    │  Stitch M positions                          │
    │  ┌──────── layout preview ──────────┐        │
    │  │ [MplCanvas with tile rectangles] │        │
    │  └──────────────────────────────────┘        │
    │                                              │
    │  Tile selection:  [☑0 ☑1 ☑2 …]  [All] [None] │
    │  Channels:         [☑DAPI ☑GFP ☐TRITC] [colors] │
    │  Z mode:           [max ▾]   Z slice: [- 0 -] │
    │  Output:           [/path/to/out.tif] [Browse]│
    │                                              │
    │             [Cancel]   [Export]              │
    └──────────────────────────────────────────────┘
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
    QWidget,
)

from nd2studios.backend.exporters.stitch_exporter import (
    StitchLayout, compute_tile_layout,
)
from nd2studios.backend.nd2_volume import LazyND2Volume
from nd2studios.core.settings import Settings
from nd2studios.widgets.common import MplCanvas
from nd2studios.widgets.image_viewer import CHANNEL_COLORS
from nd2studios.workers.stitch_worker import StitchRequest, StitchWorker


_DEFAULT_COLOR_CYCLE = [
    "green", "red", "cyan", "magenta", "yellow", "blue", "orange",
]


class StitchDialog(QDialog):
    """Configure and launch a multipoint stitch + TIFF export."""

    def __init__(self,
                 volume: LazyND2Volume,
                 stage_xy_um: List[Tuple[float, float]],
                 channel_display: Optional[Dict[str, Dict[str, str]]] = None,
                 main_window=None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Stitch M positions")
        self.setMinimumSize(620, 620)

        self.volume = volume
        self.stage_xy_um = stage_xy_um
        self.main_window = main_window
        self._channel_display = channel_display or {}
        self._tile_checkboxes: List[QCheckBox] = []
        self._channel_checkboxes: List[QCheckBox] = []
        self._channel_colors: List[QComboBox] = []
        self._worker: Optional[StitchWorker] = None
        self._output_path: str = ""

        self._build_ui()
        self._refresh_layout_preview()

    # ── UI ──
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Layout preview.
        preview_group = QGroupBox("Layout preview")
        preview_layout = QVBoxLayout(preview_group)
        self.canvas = MplCanvas(self, width=5, height=3.4, dpi=90)
        self.ax = self.canvas.add_subplot(111)
        preview_layout.addWidget(self.canvas)
        self.lbl_layout_source = QLabel("")
        self.lbl_layout_source.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        preview_layout.addWidget(self.lbl_layout_source)
        layout.addWidget(preview_group)

        # Tile selection.
        tile_group = QGroupBox(f"Tile selection ({self.volume.n_multipoints} positions)")
        tile_outer = QVBoxLayout(tile_group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(120)
        host = QWidget()
        grid = QHBoxLayout(host)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(4)
        for m in range(self.volume.n_multipoints):
            cb = QCheckBox(f"M{m}")
            cb.setChecked(True)
            cb.stateChanged.connect(lambda *_: self._refresh_layout_preview())
            grid.addWidget(cb)
            self._tile_checkboxes.append(cb)
        grid.addStretch(1)
        scroll.setWidget(host)
        tile_outer.addWidget(scroll)
        bulk = QHBoxLayout()
        btn_all = QPushButton("Select all")
        btn_all.clicked.connect(lambda: self._set_all_tiles(True))
        bulk.addWidget(btn_all)
        btn_none = QPushButton("Select none")
        btn_none.clicked.connect(lambda: self._set_all_tiles(False))
        bulk.addWidget(btn_none)
        bulk.addStretch(1)
        tile_outer.addLayout(bulk)
        layout.addWidget(tile_group)

        # Channels.
        ch_group = QGroupBox("Channels")
        ch_outer = QVBoxLayout(ch_group)
        for c, name in enumerate(self.volume.channel_names):
            row = QHBoxLayout()
            cb = QCheckBox(name)
            cb.setChecked(self._channel_display.get(name, {}).get("enabled", True))
            row.addWidget(cb)
            self._channel_checkboxes.append(cb)
            combo = QComboBox()
            combo.addItems(list(CHANNEL_COLORS.keys()))
            current_color = (
                self._channel_display.get(name, {}).get(
                    "color", _DEFAULT_COLOR_CYCLE[c % len(_DEFAULT_COLOR_CYCLE)])
            )
            if current_color in CHANNEL_COLORS:
                combo.setCurrentText(current_color)
            combo.setFixedWidth(96)
            row.addWidget(combo)
            self._channel_colors.append(combo)
            row.addStretch(1)
            ch_outer.addLayout(row)
        layout.addWidget(ch_group)

        # Z mode + Z slice.
        z_group = QGroupBox("Z handling")
        z_form = QFormLayout(z_group)
        self.combo_z = QComboBox()
        self.combo_z.addItems(["max", "mean", "min", "none"])
        z_form.addRow("Z mode", self.combo_z)
        self.spin_z = QSpinBox()
        self.spin_z.setRange(0, max(0, self.volume.n_zslices - 1))
        z_form.addRow("Z slice (when 'none')", self.spin_z)
        layout.addWidget(z_group)

        # Output.
        out_group = QGroupBox("Output")
        out_form = QHBoxLayout(out_group)
        self.lbl_out = QLabel("(no path chosen)")
        self.lbl_out.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        out_form.addWidget(self.lbl_out, stretch=1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        out_form.addWidget(btn_browse)
        layout.addWidget(out_group)

        # Buttons.
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        bb.button(QDialogButtonBox.StandardButton.Apply).setText("Export")
        bb.button(QDialogButtonBox.StandardButton.Apply).setObjectName("primaryBtn")
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_export)
        layout.addWidget(bb)

    # ── Helpers ──
    def _set_all_tiles(self, value: bool) -> None:
        for cb in self._tile_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(value)
            cb.blockSignals(False)
        self._refresh_layout_preview()

    def _selected_m_indices(self) -> List[int]:
        return [i for i, cb in enumerate(self._tile_checkboxes) if cb.isChecked()]

    def _selected_channels(self) -> List[int]:
        return [i for i, cb in enumerate(self._channel_checkboxes) if cb.isChecked()]

    def _channel_color_map(self) -> Dict[int, Tuple[int, int, int]]:
        out: Dict[int, Tuple[int, int, int]] = {}
        for i, combo in enumerate(self._channel_colors):
            out[i] = CHANNEL_COLORS.get(combo.currentText(), (255, 255, 255))
        return out

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Stitched output", "",
            "TIFF (*.tif *.tiff);;All files (*)",
        )
        if path:
            if not path.lower().endswith((".tif", ".tiff")):
                path += ".tif"
            self._output_path = path
            self.lbl_out.setText(path)

    def _refresh_layout_preview(self) -> None:
        m_indices = self._selected_m_indices()
        layout = compute_tile_layout(
            stage_xy_um=self.stage_xy_um,
            pixel_size_um=self.volume.pixel_size_um,
            tile_h=self.volume.height,
            tile_w=self.volume.width,
            m_indices=m_indices,
        )
        self._draw_preview(layout, m_indices)
        self.lbl_layout_source.setText(
            f"Source: {layout.source}   Canvas: "
            f"{layout.canvas_w} × {layout.canvas_h} px"
        )

    def _draw_preview(self, layout: StitchLayout, m_indices: List[int]) -> None:
        ax = self.ax
        ax.clear()
        ax.set_facecolor(Settings.BG_SECONDARY)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        # Draw tile rectangles.
        for offset, m in zip(layout.offsets, m_indices):
            y, x = offset
            rect_x = [x, x + layout.tile_w, x + layout.tile_w, x, x]
            rect_y = [y, y, y + layout.tile_h, y + layout.tile_h, y]
            ax.plot(rect_x, rect_y, color=Settings.ACCENT_PURPLE, linewidth=1)
            ax.text(x + layout.tile_w / 2, y + layout.tile_h / 2, f"{m}",
                    color=Settings.FG_PRIMARY, fontsize=8,
                    ha="center", va="center")
        ax.set_xlim(-layout.tile_w * 0.1, layout.canvas_w + layout.tile_w * 0.1)
        ax.set_ylim(layout.canvas_h + layout.tile_h * 0.1, -layout.tile_h * 0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(Settings.BORDER_COLOR)
        self.canvas.draw_idle()

    # ── Export ──
    def _on_export(self) -> None:
        m_indices = self._selected_m_indices()
        if not m_indices:
            QMessageBox.information(self, "Pick tiles", "Select at least one M position.")
            return
        channel_indices = self._selected_channels()
        if not channel_indices:
            QMessageBox.information(self, "Pick channels", "Enable at least one channel.")
            return
        if not self._output_path:
            self._browse()
            if not self._output_path:
                return

        layout = compute_tile_layout(
            stage_xy_um=self.stage_xy_um,
            pixel_size_um=self.volume.pixel_size_um,
            tile_h=self.volume.height,
            tile_w=self.volume.width,
            m_indices=m_indices,
        )
        request = StitchRequest(
            volume=self.volume,
            layout=layout,
            m_indices=m_indices,
            channel_indices=channel_indices,
            channel_colors=self._channel_color_map(),
            filepath=self._output_path,
            z_mode=self.combo_z.currentText(),
            z_index=int(self.spin_z.value()),
            rgb=len(channel_indices) > 1,
            pixel_size_um=self.volume.pixel_size_um,
        )
        self._worker = StitchWorker(request)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        if self.main_window is not None:
            self.main_window.set_status_text("Stitching…")
        self._worker.start()

    def _on_progress(self, p: int) -> None:
        if self.main_window is not None:
            self.main_window.set_progress(p)

    def _on_status(self, msg: str) -> None:
        if self.main_window is not None:
            self.main_window.set_status_text(msg)

    def _on_done(self, result: str) -> None:
        if self.main_window is not None:
            self.main_window.set_progress(0)
            self.main_window.set_status_text(f"Stitched: {result}")
        QMessageBox.information(self, "Stitch complete", f"Wrote:\n{result}")
        self.accept()

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Stitch failed", msg)
        if self.main_window is not None:
            self.main_window.set_progress(0)
