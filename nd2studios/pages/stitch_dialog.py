"""
``StitchDialog`` (V1.3) — opened from the Import page to stitch
multipoint tiles into a single TIFF stack.

Layout
------

    ┌──────────────────────────────────────────────┐
    │  Stitch M positions                          │
    │  ┌──────── interactive layout ───────┐        │
    │  │  TileLayoutWidget(mode=select)    │        │
    │  │  click tiles to toggle inclusion  │        │
    │  └───────────────────────────────────┘        │
    │  [Select all]  [Select none]   N/M selected   │
    │                                              │
    │  Channels:        [☑DAPI ☑GFP ☐TRITC] [colors]│
    │  Z mode:          [max ▾]   Z slice: [- 0 -]  │
    │  Output:          [/path/to/out.tif]  [Browse]│
    │                                              │
    │             [Cancel]   [Export]              │
    └──────────────────────────────────────────────┘

V1.3 changes vs V1.2: replaced the matplotlib `MplCanvas` preview *and*
the row of per-tile checkboxes with a single :class:`TileLayoutWidget`
in ``"select"`` mode. Click any tile in the canvas to toggle inclusion;
selection state is read on Export. Same backend job, simpler UI.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from nd2studios.backend.exporters.stitch_exporter import (
    compute_tile_layout,
)
from nd2studios.backend.nd2_volume import LazyND2Volume
from nd2studios.core.settings import Settings
from nd2studios.widgets.image_viewer import CHANNEL_COLORS
from nd2studios.widgets.tile_layout import TileLayoutWidget
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
        self.setMinimumSize(680, 720)

        self.volume = volume
        self.stage_xy_um = list(stage_xy_um or [])
        self.main_window = main_window
        self._channel_display = channel_display or {}
        self._channel_checkboxes: List[QCheckBox] = []
        self._channel_colors: List[QComboBox] = []
        self._worker: Optional[StitchWorker] = None
        self._output_path: str = ""

        # Default selection: every tile is in.
        self._selected: Set[int] = set(range(volume.n_multipoints))

        self._build_ui()
        self._refresh_summary()

    # ── UI ──
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Interactive layout in select mode.
        layout_group = QGroupBox("Tile selection")
        lg_layout = QVBoxLayout(layout_group)
        self.tile_widget = TileLayoutWidget(
            mode="select",
            show_expand_button=True,
            minimum_size=(620, 280),
        )
        self.tile_widget.set_tile_layout(
            stage_xy_um=self.stage_xy_um,
            pixel_size_um=self.volume.pixel_size_um,
            tile_h=self.volume.height,
            tile_w=self.volume.width,
            m_indices=list(range(self.volume.n_multipoints)),
            current_m=0,
            selected_indices=self._selected,
        )
        self.tile_widget.selection_changed.connect(self._on_selection_changed)
        self.tile_widget.expand_requested.connect(self._open_expanded_dialog)
        lg_layout.addWidget(self.tile_widget, stretch=1)

        # Bulk + summary row.
        bulk_row = QHBoxLayout()
        btn_all = QPushButton("Select all")
        btn_all.clicked.connect(lambda: self._set_all_tiles(True))
        bulk_row.addWidget(btn_all)
        btn_none = QPushButton("Select none")
        btn_none.clicked.connect(lambda: self._set_all_tiles(False))
        bulk_row.addWidget(btn_none)
        bulk_row.addStretch(1)
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        bulk_row.addWidget(self.lbl_summary)
        lg_layout.addLayout(bulk_row)
        layout.addWidget(layout_group)

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

        # Z mode. 'none' preserves every Z plane in the stitched
        # hyperstack; max/mean/min collapse Z to 1.
        z_group = QGroupBox("Z handling")
        z_outer = QVBoxLayout(z_group)
        z_form = QFormLayout()
        self.combo_z = QComboBox()
        self.combo_z.addItems(["max", "mean", "min", "none"])
        z_form.addRow("Z mode", self.combo_z)
        z_outer.addLayout(z_form)
        self.lbl_z_hint = QLabel("")
        self.lbl_z_hint.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        self.lbl_z_hint.setWordWrap(True)
        z_outer.addWidget(self.lbl_z_hint)
        layout.addWidget(z_group)
        self.combo_z.currentTextChanged.connect(self._on_z_mode_changed)
        self._on_z_mode_changed(self.combo_z.currentText())

        # Output path.
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

    # ── Selection handling ──
    def _on_selection_changed(self, indices: Set[int]) -> None:
        self._selected = set(indices)
        self._refresh_summary()

    def _set_all_tiles(self, value: bool) -> None:
        n = self.volume.n_multipoints
        new_set = set(range(n)) if value else set()
        self.tile_widget.set_selected_indices(new_set)
        # Widget emits the change; just update local state and summary.
        self._selected = new_set
        self._refresh_summary()

    def _open_expanded_dialog(self) -> None:
        """Pop a much larger view of the layout for huge tile sets."""
        from nd2studios.widgets.tile_layout import TileLayoutDialog
        dlg = TileLayoutDialog(
            mode="select",
            stage_xy_um=self.stage_xy_um,
            pixel_size_um=self.volume.pixel_size_um,
            tile_h=self.volume.height, tile_w=self.volume.width,
            m_indices=list(range(self.volume.n_multipoints)),
            current_m=0,
            selected_indices=self._selected,
            parent=self,
        )
        # Keep the small widget mirrored as the user picks tiles in the modal.
        dlg.selection_changed.connect(self._sync_from_dialog)
        dlg.exec()

    def _sync_from_dialog(self, indices: Set[int]) -> None:
        self._selected = set(indices)
        self.tile_widget.set_selected_indices(self._selected)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        n_total = self.volume.n_multipoints
        n_sel = len(self._selected)
        layout = compute_tile_layout(
            stage_xy_um=self.stage_xy_um,
            pixel_size_um=self.volume.pixel_size_um,
            tile_h=self.volume.height, tile_w=self.volume.width,
            m_indices=sorted(self._selected),
        ) if self._selected else None
        if layout is not None:
            n_ch = sum(
                1 for cb in self._channel_checkboxes if cb.isChecked()
            ) or 1
            n_z_out = (self.volume.n_zslices
                       if self.combo_z.currentText() == "none"
                          and self.volume.n_zslices > 1
                       else 1)
            mb = (layout.canvas_h * layout.canvas_w
                  * n_ch * self.volume.dtype.itemsize
                  * self.volume.n_timepoints * n_z_out) / 1_048_576
            z_note = f" · {n_z_out} Z" if n_z_out > 1 else ""
            self.lbl_summary.setText(
                f"{n_sel} / {n_total} tiles · "
                f"{layout.canvas_w}×{layout.canvas_h} px{z_note} · "
                f"~{mb:.0f} MB"
            )
        else:
            self.lbl_summary.setText(f"0 / {n_total} tiles selected")

    def _on_z_mode_changed(self, mode: str) -> None:
        if self.volume.n_zslices <= 1:
            self.lbl_z_hint.setText("Single-Z file — Z mode has no effect.")
        elif mode == "none":
            self.lbl_z_hint.setText(
                f"All {self.volume.n_zslices} Z planes preserved "
                "(TZCYX hyperstack)."
            )
        else:
            self.lbl_z_hint.setText(
                f"{self.volume.n_zslices} Z planes collapsed via "
                f"{mode}-projection."
            )
        self._refresh_summary()

    # ── Channels ──
    def _selected_channels(self) -> List[int]:
        return [i for i, cb in enumerate(self._channel_checkboxes) if cb.isChecked()]

    def _channel_color_map(self) -> Dict[int, Tuple[int, int, int]]:
        out: Dict[int, Tuple[int, int, int]] = {}
        for i, combo in enumerate(self._channel_colors):
            out[i] = CHANNEL_COLORS.get(combo.currentText(), (255, 255, 255))
        return out

    # ── Output / Export ──
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

    def _on_export(self) -> None:
        m_indices = sorted(self._selected)
        if not m_indices:
            QMessageBox.information(self, "Pick tiles",
                                    "Click at least one tile to include it.")
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
            tile_h=self.volume.height, tile_w=self.volume.width,
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
            z_index=0,
            rgb=False,
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
        # Wait for the worker's OS thread to finish before the dialog
        # closes — otherwise Python may GC the QThread wrapper while
        # the underlying thread is still in its tail-end shutdown,
        # producing "QThread: Destroyed while thread is still running".
        if self._worker is not None:
            self._worker.wait(5000)
        self.accept()

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Stitch failed", msg)
        if self.main_window is not None:
            self.main_window.set_progress(0)
        if self._worker is not None:
            self._worker.wait(5000)
