"""
Export page — three tabs covering V1.0 outputs:

1. **Z-Projection TIFF** — single-channel multi-page TIFF, one file per
   enabled channel. Bit-depth selector. Source: raw or processed.
2. **RGB Composite TIFF** — multi-channel additive composite TIFF.
   Source: raw or processed.
3. **Movie** — MP4 / GIF time-lapse with optional scale bar, timestamp,
   and channel-label overlays.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QRadioButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from nd2studios.backend.exporters.movie_exporter import MovieOptions
from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.core.settings import Settings
from nd2studios.widgets.image_viewer import CHANNEL_COLORS
from nd2studios.workers.export_worker import ExportRequest, ExportWorker


class ExportPage(QWidget):
    """Page 3: export TIFF stacks, RGB composites, and movies."""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._worker: Optional[ExportWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # Source selector — applies to every tab.
        src_group = QGroupBox("Source")
        sl = QHBoxLayout(src_group)
        self.rb_raw = QRadioButton("Raw")
        self.rb_proc = QRadioButton("Processed (recipe applied)")
        self.rb_proc.setChecked(True)
        sg = QButtonGroup(self)
        sg.addButton(self.rb_raw)
        sg.addButton(self.rb_proc)
        sl.addWidget(self.rb_raw)
        sl.addWidget(self.rb_proc)
        sl.addStretch(1)
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        sl.addWidget(self.lbl_summary)
        outer.addWidget(src_group)

        # Tabs.
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_tiff(), "Z-Projection TIFF")
        self.tabs.addTab(self._build_tab_composite(), "RGB Composite")
        self.tabs.addTab(self._build_tab_movie(), "Movie")
        outer.addWidget(self.tabs, stretch=1)

    # ── Tab 1: TIFF stack ──
    def _build_tab_tiff(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        form = QFormLayout()
        self.combo_tiff_bitdepth = QComboBox()
        self.combo_tiff_bitdepth.addItems(["passthrough", "uint16", "uint8"])
        form.addRow("Bit depth", self.combo_tiff_bitdepth)
        layout.addLayout(form)

        info = QLabel(
            "Writes one multi-page TIFF per enabled channel. Filenames are\n"
            "<basename>_<channel>.tif (or just <basename>.tif if a single\n"
            "channel is enabled). Pixel size is written into the TIFF tags."
        )
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.btn_export_tiff = QPushButton("Export TIFF Stack…")
        self.btn_export_tiff.setObjectName("primaryBtn")
        self.btn_export_tiff.clicked.connect(self._on_export_tiff)
        layout.addWidget(self.btn_export_tiff)
        layout.addStretch(1)
        return w

    # ── Tab 2: RGB composite ──
    def _build_tab_composite(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info = QLabel(
            "Writes a single multi-page RGB TIFF (uint8) where each enabled\n"
            "channel is mapped to its assigned color and additively blended."
        )
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.btn_export_composite = QPushButton("Export RGB Composite TIFF…")
        self.btn_export_composite.setObjectName("primaryBtn")
        self.btn_export_composite.clicked.connect(self._on_export_composite)
        layout.addWidget(self.btn_export_composite)
        layout.addStretch(1)
        return w

    # ── Tab 3: Movie ──
    def _build_tab_movie(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Format / FPS
        basics = QGroupBox("Basics")
        bf = QFormLayout(basics)
        self.combo_movie_fmt = QComboBox()
        self.combo_movie_fmt.addItems(["mp4", "gif"])
        bf.addRow("Format", self.combo_movie_fmt)
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(0.5, 60.0)
        self.spin_fps.setValue(10.0)
        self.spin_fps.setSingleStep(0.5)
        bf.addRow("FPS", self.spin_fps)
        layout.addWidget(basics)

        # Scale bar
        sb = QGroupBox("Scale bar")
        sf = QFormLayout(sb)
        self.cb_scalebar = QCheckBox("Show scale bar")
        self.cb_scalebar.setChecked(True)
        sf.addRow(self.cb_scalebar)
        self.spin_scalebar_um = QDoubleSpinBox()
        self.spin_scalebar_um.setRange(0.1, 10000.0)
        self.spin_scalebar_um.setValue(50.0)
        self.spin_scalebar_um.setSuffix(" µm")
        sf.addRow("Length", self.spin_scalebar_um)
        self.combo_scalebar_pos = QComboBox()
        self.combo_scalebar_pos.addItems(
            ["bottom-right", "bottom-left", "top-right", "top-left"])
        sf.addRow("Position", self.combo_scalebar_pos)
        self.combo_scalebar_color = QComboBox()
        self.combo_scalebar_color.addItems(["white", "yellow", "black", "magenta"])
        sf.addRow("Color", self.combo_scalebar_color)
        layout.addWidget(sb)

        # Timestamp
        ts = QGroupBox("Timestamp")
        tf = QFormLayout(ts)
        self.cb_timestamp = QCheckBox("Show timestamp")
        self.cb_timestamp.setChecked(True)
        tf.addRow(self.cb_timestamp)
        self.combo_ts_pos = QComboBox()
        self.combo_ts_pos.addItems(
            ["bottom-left", "bottom-right", "top-left", "top-right"])
        tf.addRow("Position", self.combo_ts_pos)
        self.combo_ts_color = QComboBox()
        self.combo_ts_color.addItems(["white", "yellow", "black"])
        tf.addRow("Color", self.combo_ts_color)
        self.cb_ts_use_synth = QCheckBox(
            "Use synthetic dt instead of ND2 timestamps")
        tf.addRow(self.cb_ts_use_synth)
        self.spin_ts_dt = QDoubleSpinBox()
        self.spin_ts_dt.setRange(0.001, 10000.0)
        self.spin_ts_dt.setValue(1.0)
        self.spin_ts_dt.setSuffix(" s")
        tf.addRow("dt (synthetic)", self.spin_ts_dt)
        layout.addWidget(ts)

        # Channel labels
        cl = QGroupBox("Channel labels")
        cf = QFormLayout(cl)
        self.cb_channel_labels = QCheckBox("Show channel labels")
        self.cb_channel_labels.setChecked(True)
        cf.addRow(self.cb_channel_labels)
        self.combo_chl_pos = QComboBox()
        self.combo_chl_pos.addItems(
            ["top-left", "top-right", "bottom-left", "bottom-right"])
        cf.addRow("Position", self.combo_chl_pos)
        layout.addWidget(cl)

        self.btn_export_movie = QPushButton("Export Movie…")
        self.btn_export_movie.setObjectName("primaryBtn")
        self.btn_export_movie.clicked.connect(self._on_export_movie)
        layout.addWidget(self.btn_export_movie)
        layout.addStretch(1)
        return w

    # ── Page lifecycle ──
    def on_activated(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        # Default to processed if available, raw otherwise.
        if exp._processed_channels:
            self.rb_proc.setChecked(True)
        else:
            self.rb_raw.setChecked(True)
            self.rb_proc.setEnabled(False)
        # Show a quick summary.
        n = exp.n_frames or 0
        ch = len(exp._raw_channels or {})
        self.lbl_summary.setText(
            f"{n} frames · {ch} channels · {exp.frame_width}×{exp.frame_height} px"
            f" · {exp.pixel_size_um:.3f} µm/px"
        )

    def load_from_experiment(self, exp: ND2StudiosRecord) -> None:
        if "fps" in exp.export_config:
            self.spin_fps.setValue(float(exp.export_config["fps"]))
        if "movie_format" in exp.export_config:
            self.combo_movie_fmt.setCurrentText(str(exp.export_config["movie_format"]))

    def save_to_experiment(self, exp: ND2StudiosRecord) -> None:
        exp.export_config = {
            "fps": float(self.spin_fps.value()),
            "movie_format": self.combo_movie_fmt.currentText(),
            "tiff_bit_depth": self.combo_tiff_bitdepth.currentText(),
        }
        exp.fps = float(self.spin_fps.value())

    # ── Source picking ──
    def _channels_and_state(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int, int]], Dict[str, bool]]:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return {}, {}, {}
        exp = self.main_window.exp_manager.active
        if self.rb_proc.isChecked() and exp._processed_channels:
            channels = exp._processed_channels
        else:
            # Materialize lazy proxies for export.
            channels = {}
            for name, data in (exp._raw_channels or {}).items():
                m = getattr(data, "materialize", None)
                channels[name] = m() if callable(m) else np.asarray(data)
        colors: Dict[str, Tuple[int, int, int]] = {}
        enabled: Dict[str, bool] = {}
        for name in channels:
            cd = exp.channel_display.get(name, {})
            colors[name] = CHANNEL_COLORS.get(cd.get("color", "gray"), (255, 255, 255))
            enabled[name] = bool(cd.get("enabled", True))
        return channels, colors, enabled

    # ── Export handlers ──
    def _on_export_tiff(self) -> None:
        channels, colors, enabled = self._channels_and_state()
        if not channels:
            QMessageBox.information(self, "Nothing to export", "Import a file first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export TIFF Stack", "",
            "TIFF (*.tif *.tiff);;All files (*)",
        )
        if not path:
            return
        req = ExportRequest(
            mode="tiff_stack",
            filepath=path,
            channels=channels,
            colors=colors,
            enabled=enabled,
            pixel_size_um=self._pixel_size_um(),
            bit_depth=self.combo_tiff_bitdepth.currentText(),
        )
        self._run_export(req, "Writing TIFF…")

    def _on_export_composite(self) -> None:
        channels, colors, enabled = self._channels_and_state()
        if not channels:
            QMessageBox.information(self, "Nothing to export", "Import a file first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export RGB Composite TIFF", "",
            "TIFF (*.tif *.tiff);;All files (*)",
        )
        if not path:
            return
        req = ExportRequest(
            mode="rgb_composite",
            filepath=path,
            channels=channels,
            colors=colors,
            enabled=enabled,
            pixel_size_um=self._pixel_size_um(),
        )
        self._run_export(req, "Writing RGB composite…")

    def _on_export_movie(self) -> None:
        channels, colors, enabled = self._channels_and_state()
        if not channels:
            QMessageBox.information(self, "Nothing to export", "Import a file first.")
            return
        fmt = self.combo_movie_fmt.currentText()
        ext = ".mp4" if fmt == "mp4" else ".gif"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Movie", "",
            f"{fmt.upper()} (*{ext});;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext

        opts = MovieOptions(
            fps=float(self.spin_fps.value()),
            codec=fmt,
            show_scale_bar=self.cb_scalebar.isChecked(),
            scale_bar_um=float(self.spin_scalebar_um.value()),
            scale_bar_color=self.combo_scalebar_color.currentText(),
            scale_bar_position=self.combo_scalebar_pos.currentText(),
            show_timestamp=self.cb_timestamp.isChecked(),
            timestamp_position=self.combo_ts_pos.currentText(),
            timestamp_color=self.combo_ts_color.currentText(),
            timestamp_dt_seconds=(float(self.spin_ts_dt.value())
                                  if self.cb_ts_use_synth.isChecked() else None),
            show_channel_labels=self.cb_channel_labels.isChecked(),
            channel_label_position=self.combo_chl_pos.currentText(),
        )

        # Frame timestamps: only used if the synthetic dt checkbox is OFF.
        ts = None
        if (self.main_window is not None
                and self.main_window.exp_manager.active is not None
                and self.main_window.exp_manager.active._frame_timestamps is not None):
            ts = np.asarray(self.main_window.exp_manager.active._frame_timestamps)

        req = ExportRequest(
            mode="movie",
            filepath=path,
            channels=channels,
            colors=colors,
            enabled=enabled,
            pixel_size_um=self._pixel_size_um(),
            frame_timestamps_s=ts,
            movie_options=opts,
        )
        self._run_export(req, "Rendering movie…")

    # ── Worker plumbing ──
    def _run_export(self, request: ExportRequest, label: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy",
                                    "An export is already running.")
            return
        self._worker = ExportWorker(request, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        if self.main_window is not None:
            self.main_window.set_status_text(label)
        self._worker.start()

    def _pixel_size_um(self) -> float:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return 1.0
        return float(self.main_window.exp_manager.active.pixel_size_um or 1.0)

    def _on_progress(self, p: int) -> None:
        if self.main_window is not None:
            self.main_window.set_progress(p)

    def _on_status(self, msg: str) -> None:
        if self.main_window is not None:
            self.main_window.set_status_text(msg)

    def _on_done(self, result: Any) -> None:
        if self.main_window is not None:
            self.main_window.set_progress(0)
            self.main_window.set_status_text(f"Saved: {result}")
            self.main_window.exp_manager.set_status("ready_to_export")
        QMessageBox.information(self, "Export complete",
                                f"Wrote:\n{result}")

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Export failed", msg)
        if self.main_window is not None:
            self.main_window.set_progress(0)
