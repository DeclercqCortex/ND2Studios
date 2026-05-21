"""
Import page (V1.1).

Browse for an ND2 (or TIFF) file. Surface the full extended metadata
(T/Z/C/P, pixel size, z step, channel names + colors + emission/excitation
+ exposures, frame timestamps, stage XY/Z, objective, binning, camera).

The right side hosts a :class:`MultiAxisViewer` with M, T, Z sliders and
per-channel toggle / color / LUT histogram controls so the user can
explore the file live before committing to import. The left side shows
file metadata and per-channel intrinsic info (exposure, emission,
excitation) — these are display-only; channel show/hide and color are
controlled by the viewer.

A "Stitch M positions" button opens :class:`StitchDialog` for users who
want to combine multipoint tiles into a single TIFF stack without
modifying the original ND2.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.core.settings import Settings
from nd2studios.pages.reconstruct_dialog import ReconstructDialog
from nd2studios.pages.stitch_dialog import StitchDialog
from nd2studios.widgets.multi_axis_viewer import MultiAxisViewer
from nd2studios.workers.load_worker import LoadWorker


class ChannelInfoRow(QWidget):
    """Read-only metadata row: ``Cn: name  exposure  ex  em``.

    Channel show/hide and color are controlled by the viewer; this
    widget exists only to surface acquisition parameters that aren't
    part of the visualization choice.
    """

    def __init__(self, idx: int, name: str,
                 exposure_ms: Optional[float] = None,
                 emission_nm: Optional[float] = None,
                 excitation_nm: Optional[float] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.idx = idx
        self.name = name
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)

        lbl = QLabel(f"C{idx}: {name}")
        lbl.setMinimumWidth(160)
        layout.addWidget(lbl)

        info_bits: List[str] = []
        if exposure_ms is not None:
            info_bits.append(f"{exposure_ms:.0f} ms")
        if excitation_nm is not None:
            info_bits.append(f"ex {excitation_nm:.0f}")
        if emission_nm is not None:
            info_bits.append(f"em {emission_nm:.0f}")
        info = QLabel("  ".join(info_bits) or "—")
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        layout.addWidget(info, stretch=1)


class ImportPage(QWidget):
    """Page 1: file browse + metadata + multi-axis preview + stitch."""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._filepath: Optional[str] = None
        self._meta_dict: Dict[str, Any] = {}
        self._info_rows: List[ChannelInfoRow] = []
        self._loader: Optional[LoadWorker] = None

        self._build_ui()

    # ── UI ──
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(0)

        _splitter = QSplitter(Qt.Horizontal)
        _splitter.setChildrenCollapsible(False)

        # Left column: file + metadata + intrinsic channel info.
        left = QWidget()
        left.setMinimumWidth(200)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        # File group.
        file_group = QGroupBox("File")
        fl = QVBoxLayout(file_group)
        self.btn_browse = QPushButton("Browse ND2 / TIFF…")
        self.btn_browse.setObjectName("primaryBtn")
        self.btn_browse.clicked.connect(self._on_browse)
        fl.addWidget(self.btn_browse)
        self.btn_reconstruct = QPushButton("Reconstruct from multiple files…")
        self.btn_reconstruct.setToolTip(
            "Pick several ND2 or TIFF files and chain them along an axis "
            "(T / M / Z / C) into a single virtual volume.")
        self.btn_reconstruct.clicked.connect(self._on_reconstruct)
        fl.addWidget(self.btn_reconstruct)
        self.lbl_filepath = QLabel("No file loaded.")
        self.lbl_filepath.setWordWrap(True)
        self.lbl_filepath.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        fl.addWidget(self.lbl_filepath)
        ll.addWidget(file_group)

        # Z mode + frame stride.
        proj_group = QGroupBox("Z mode & range")
        pl = QFormLayout(proj_group)
        self.combo_zproj = QComboBox()
        self.combo_zproj.addItems(["max", "mean", "min", "none"])
        self.combo_zproj.setCurrentText("max")
        self.combo_zproj.currentTextChanged.connect(self._on_z_mode_changed)
        pl.addRow("Z mode", self.combo_zproj)
        self.spin_t_stride = QSpinBox()
        self.spin_t_stride.setRange(1, 1000)
        self.spin_t_stride.setValue(1)
        pl.addRow("Frame stride", self.spin_t_stride)
        ll.addWidget(proj_group)

        # Metadata table.
        meta_group = QGroupBox("Metadata")
        ml = QVBoxLayout(meta_group)
        self.meta_table = QTableWidget(0, 2)
        self.meta_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.meta_table.horizontalHeader().setStretchLastSection(True)
        self.meta_table.verticalHeader().setVisible(False)
        self.meta_table.setMaximumHeight(280)
        ml.addWidget(self.meta_table)
        ll.addWidget(meta_group)

        # Per-channel intrinsic info (read-only).
        info_group = QGroupBox("Channel info")
        cl = QVBoxLayout(info_group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(140)
        self._info_container = QWidget()
        self._info_layout = QVBoxLayout(self._info_container)
        self._info_layout.setContentsMargins(2, 2, 2, 2)
        self._info_layout.addStretch(1)
        scroll.setWidget(self._info_container)
        cl.addWidget(scroll)
        ll.addWidget(info_group)

        # Buttons (Confirm + Stitch).
        btn_row = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm Import")
        self.btn_confirm.setObjectName("successBtn")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.clicked.connect(self._on_confirm)
        btn_row.addWidget(self.btn_confirm)
        self.btn_stitch = QPushButton("Stitch M…")
        self.btn_stitch.setEnabled(False)
        self.btn_stitch.setToolTip(
            "Stitch multipoint (M) tiles into a single TIFF stack.\n"
            "Original file is not modified.")
        self.btn_stitch.clicked.connect(self._on_stitch)
        btn_row.addWidget(self.btn_stitch)
        ll.addLayout(btn_row)

        ll.addStretch(1)
        _splitter.addWidget(left)

        # Right column: multi-axis viewer.
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        rl.addWidget(QLabel("Preview", objectName="sectionHeader"))
        self.viewer = MultiAxisViewer(self, show_tile_preview=True)
        self.viewer.coords_changed.connect(self._on_coords_changed)
        self.viewer.channels_changed.connect(self._on_channels_changed)
        # Click on the corner tile preview opens the stitch dialog.
        self.viewer.stitch_requested.connect(self._on_stitch)
        rl.addWidget(self.viewer, stretch=1)
        _splitter.addWidget(right)
        _splitter.setStretchFactor(0, 0)
        _splitter.setStretchFactor(1, 1)
        _splitter.setSizes([420, 900])
        outer.addWidget(_splitter, stretch=1)

    # ── Page lifecycle ──
    def on_activated(self) -> None:
        pass

    def load_from_experiment(self, exp: ND2StudiosRecord) -> None:
        """Restore UI state from a session that was just loaded."""
        if not exp.import_config:
            return
        cfg = exp.import_config
        if "z_projection" in cfg:
            self.combo_zproj.setCurrentText(cfg["z_projection"])
        if "t_stride" in cfg:
            self.spin_t_stride.setValue(int(cfg["t_stride"]))
        if exp.nd2_metadata:
            self._meta_dict = exp.nd2_metadata
            self._populate_metadata_table(self._meta_dict)
            self._populate_info_rows(self._meta_dict)
        if cfg.get("filepath"):
            self._filepath = cfg["filepath"]
            self.lbl_filepath.setText(os.path.basename(self._filepath))

    def save_to_experiment(self, exp: ND2StudiosRecord) -> None:
        exp.import_config = {
            "filepath": self._filepath,
            "z_projection": self.combo_zproj.currentText(),
            "t_stride": int(self.spin_t_stride.value()),
        }
        # Pull channel state from the viewer (single source of truth in V1.1).
        viewer_state = self.viewer.channel_state()
        if viewer_state:
            exp.channel_display = viewer_state
        m, t, z = self.viewer.coords()
        exp.m_index = int(m)
        exp.z_view_index = int(z)
        exp.z_view_mode = self.combo_zproj.currentText()
        # Rebuild lazy channels so any Z-mode or M-position change made after
        # "Confirm Import" (e.g. user tweaks the combo then switches tabs) is
        # always reflected in exp._raw_channels before Recipe/Export sees it.
        if exp._raw_volume is not None:
            channels = exp._raw_volume.all_channels_as_lazy(
                m=exp.m_index,
                z_mode=exp.z_view_mode,
                z_index=exp.z_view_index,
            )
            enabled_names = [
                name for name, cfg in (exp.channel_display or {}).items()
                if cfg.get("enabled", True)
            ]
            if enabled_names:
                channels = {n: channels[n] for n in enabled_names if n in channels}
            exp._raw_channels = channels

    # ── Browse (single file) ──
    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ND2 or TIFF", "",
            "Microscopy files (*.nd2 *.tif *.tiff);;All files (*)",
        )
        if not path:
            return
        self._filepath = path
        self.lbl_filepath.setText(path)
        self._start_load_worker(filepaths=[path], chain_axis="Z")

    # ── Reconstruct (multi-file with chosen chain axis, V1.28+) ──
    def _on_reconstruct(self) -> None:
        dialog = ReconstructDialog(parent=self)
        if dialog.exec() != ReconstructDialog.Accepted:
            return
        paths = dialog.filepaths()
        axis = dialog.chain_axis()
        mapping = dialog.chain_mapping()
        if len(paths) < 2:
            return
        self._filepath = paths[0]
        first = os.path.basename(paths[0])
        self.lbl_filepath.setText(
            f"{len(paths)} files chained on {axis}: {first} … "
            f"+{len(paths) - 1} more"
        )
        self._start_load_worker(
            filepaths=paths, chain_axis=axis, chain_mapping=mapping,
        )

    def _start_load_worker(self, filepaths: List[str], chain_axis: str,
                            chain_mapping=None) -> None:
        self.btn_confirm.setEnabled(False)
        self.btn_stitch.setEnabled(False)
        self._loader = LoadWorker(
            filepaths=filepaths,
            chain_axis=chain_axis,
            chain_mapping=chain_mapping,
            z_projection=self.combo_zproj.currentText(),
            t_stride=int(self.spin_t_stride.value()),
        )
        self._loader.progress.connect(self._on_progress)
        self._loader.status.connect(self._on_status)
        self._loader.finished.connect(self._on_loaded)
        self._loader.error.connect(self._on_error)
        self._loader.start()

    def _on_progress(self, p: int) -> None:
        if self.main_window is not None:
            self.main_window.set_progress(p)

    def _on_status(self, msg: str) -> None:
        if self.main_window is not None:
            self.main_window.set_status_text(msg)

    def _on_loaded(self, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        meta = payload.get("metadata", {})
        self._meta_dict = meta
        self._populate_metadata_table(meta)
        self._populate_info_rows(meta)

        if self.main_window is not None and self.main_window.exp_manager.active is not None:
            exp = self.main_window.exp_manager.active
            exp._raw_channels = payload.get("channels", {})
            exp._raw_volume = payload.get("volume")
            exp._frame_timestamps = payload.get("frame_timestamps_s")
            exp.nd2_metadata = meta
            exp.n_frames = int(meta.get("n_timepoints", 0))
            exp.frame_height = int(meta.get("height", 0))
            exp.frame_width = int(meta.get("width", 0))
            exp.n_multipoints = int(meta.get("n_multipoints", 1))
            exp.n_zslices = int(meta.get("n_zslices", 1))
            exp.pixel_size_um = float(meta.get("pixel_size_um", 1.0))

            # Wire the viewer to the volume so M/T/Z scrolling works.
            volume = payload.get("volume")
            if volume is not None:
                # Pass stage XY positions so the corner tile-preview
                # overlay can render.
                self.viewer.set_volume(
                    volume,
                    channel_display=exp.channel_display,
                    z_mode=self.combo_zproj.currentText(),
                    z_index=exp.z_view_index,
                    m=exp.m_index, t=0, z=exp.z_view_index,
                    stage_xy_um=list(meta.get("stage_xy_um") or []),
                )
            else:
                # TIFF path: no volume, just channels.
                self.viewer.set_channels(
                    exp._raw_channels or {},
                    channel_display=exp.channel_display,
                )

        self.btn_confirm.setEnabled(True)
        # Stitch button only useful when there's more than one M.
        n_m = int(meta.get("n_multipoints", 1))
        self.btn_stitch.setEnabled(n_m > 1)
        if self.main_window is not None:
            self.main_window.set_progress(0)
            self.main_window.set_status_text(
                "File loaded. Scroll M / T / Z, tweak channels, then Confirm.")

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Load failed", msg)
        if self.main_window is not None:
            self.main_window.set_progress(0)
            self.main_window.set_status_text("Load failed.")

    # ── Population ──
    def _populate_metadata_table(self, meta: Dict[str, Any]) -> None:
        rows: List[Tuple[str, str]] = []
        rows.append(("File", os.path.basename(meta.get("filepath", ""))))
        rows.append(("Dimensions",
                     f"T={meta.get('n_timepoints')}, Z={meta.get('n_zslices')}, "
                     f"C={meta.get('n_channels')}, P={meta.get('n_multipoints')}"))
        rows.append(("Frame size",
                     f"{meta.get('width')} × {meta.get('height')} px"))
        rows.append(("Dtype", str(meta.get("dtype", ""))))
        rows.append(("Pixel size", f"{meta.get('pixel_size_um', 1.0):.4f} µm"))
        rows.append(("Z step", f"{meta.get('z_step_um', 1.0):.3f} µm"))

        objm = meta.get("objective_magnification")
        objna = meta.get("objective_na")
        objname = meta.get("objective_name", "")
        if any([objname, objm, objna]):
            obj_text = objname or ""
            if objm is not None:
                obj_text += f"  {objm:g}×"
            if objna is not None:
                obj_text += f"  NA {objna:g}"
            rows.append(("Objective", obj_text.strip()))
        if meta.get("camera_name"):
            rows.append(("Camera", meta["camera_name"]))
        if meta.get("microscope_name"):
            rows.append(("Microscope", meta["microscope_name"]))
        if meta.get("binning_x") is not None:
            rows.append(("Binning",
                         f"{meta['binning_x']}×{meta.get('binning_y', meta['binning_x'])}"))

        ts = meta.get("frame_timestamps_s") or []
        if ts:
            rows.append(("Acquisition span",
                         f"{ts[-1] - ts[0]:.2f} s ({len(ts)} frames)"))
            if len(ts) > 1:
                dt = (ts[-1] - ts[0]) / (len(ts) - 1)
                rows.append(("Mean dt", f"{dt:.3f} s"))

        stage = meta.get("stage_xy_um") or []
        n_m = int(meta.get("n_multipoints", 1))
        if stage:
            xs = [p[0] for p in stage]
            ys = [p[1] for p in stage]
            rows.append(("Stage XY range",
                         f"X {min(xs):.1f}…{max(xs):.1f}  "
                         f"Y {min(ys):.1f}…{max(ys):.1f} µm"))
        # Tile source diagnostic — combines the metadata reader's
        # status with the layout widget's actual decision (V1.4 + V1.7).
        layout_source = meta.get("stage_layout_source") or ""
        if n_m > 1 or layout_source:
            tile_msg = self._tile_source_summary(meta, stage, n_m, layout_source)
            if tile_msg:
                rows.append(("Tile source", tile_msg))

        loops = meta.get("loops") or []
        if loops:
            rows.append(("Loops",
                         ", ".join(f"{lp.get('type','?')}×{lp.get('count','?')}"
                                   for lp in loops)))

        self.meta_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.meta_table.setItem(i, 0, QTableWidgetItem(k))
            self.meta_table.setItem(i, 1, QTableWidgetItem(str(v)))

    def _tile_source_summary(self, meta: Dict[str, Any],
                              stage: List[Tuple[float, float]],
                              n_m: int, layout_source: str) -> str:
        """One-line summary for the metadata-table 'Tile source' row.

        Combines the metadata reader's status (V1.4) with the layout
        widget's expansion decision (V1.7) so the user sees both at a
        glance, e.g.:

        - "68 of 68 from stage XY"
        - "68 multipoints → 17 unique cells × 4 visits → 8 × 17 layout"
        - "grid fallback (no stage XY metadata)"
        """
        if layout_source == "missing":
            return f"grid fallback ({n_m} tiles, no stage XY metadata)"
        if layout_source.startswith("partial:"):
            return (f"{len(stage)} of {n_m} from stage XY · "
                    f"grid fallback for the rest")
        if not stage or not layout_source.startswith("stage_xy"):
            return ""

        # Have valid stage XY — ask the layout algorithm what it decided.
        try:
            from nd2studios.backend.exporters.stitch_exporter import (
                compute_tile_layout,
            )
            tile_h = int(meta.get("height", 0)) or 1
            tile_w = int(meta.get("width", 0)) or 1
            px_um = float(meta.get("pixel_size_um", 1.0))
            layout = compute_tile_layout(
                list(stage),
                pixel_size_um=px_um,
                tile_h=tile_h, tile_w=tile_w,
            )
        except Exception:
            return f"{len(stage)} of {n_m} from stage XY"

        if layout.source == "physical":
            cols = layout.n_phys_cols
            rows = layout.n_phys_rows
            partial = f"{len(stage)} of {n_m}" if len(stage) < n_m else str(len(stage))
            method = layout_source.split(":")[-1] if ":" in layout_source else ""
            method_label = f" via {method}" if method else ""
            return (f"{partial} multipoints · physical layout "
                    f"{cols} cols × {rows} rows "
                    f"({layout.canvas_w} × {layout.canvas_h} px{method_label})")
        return f"grid fallback ({n_m} tiles)"

    def _populate_info_rows(self, meta: Dict[str, Any]) -> None:
        for row in self._info_rows:
            row.setParent(None)
            row.deleteLater()
        self._info_rows.clear()

        names: List[str] = list(meta.get("channel_names")
                                or [f"Ch{i}" for i in range(int(meta.get("n_channels", 1)))])
        exposures = meta.get("channel_exposure_ms") or []
        emissions = meta.get("channel_emission_nm") or []
        excitations = meta.get("channel_excitation_nm") or []

        for i, name in enumerate(names):
            row = ChannelInfoRow(
                idx=i, name=name,
                exposure_ms=(exposures[i] if i < len(exposures) else None),
                emission_nm=(emissions[i] if i < len(emissions) else None),
                excitation_nm=(excitations[i] if i < len(excitations) else None),
            )
            self._info_layout.insertWidget(self._info_layout.count() - 1, row)
            self._info_rows.append(row)

    # ── Viewer hooks ──
    def _on_coords_changed(self, m: int, t: int, z: int) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        exp.m_index = int(m)
        exp.z_view_index = int(z)

    def _on_channels_changed(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        exp.channel_display = self.viewer.channel_state()

    def _on_z_mode_changed(self, _mode: str) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if exp._raw_volume is None:
            return
        m, t, z = self.viewer.coords()
        self.viewer.set_volume(
            exp._raw_volume,
            channel_display=exp.channel_display,
            z_mode=self.combo_zproj.currentText(),
            z_index=int(z), m=int(m), t=int(t), z=int(z),
            stage_xy_um=list((exp.nd2_metadata or {}).get("stage_xy_um") or []),
        )

    # ── Stitch ──
    def _on_stitch(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if exp._raw_volume is None:
            QMessageBox.information(
                self, "No volume",
                "Stitching needs a multi-position ND2 file. "
                "Browse for one first.")
            return
        stage_xy = (exp.nd2_metadata.get("stage_xy_um") or [])
        dialog = StitchDialog(
            volume=exp._raw_volume,
            stage_xy_um=list(stage_xy),
            channel_display=exp.channel_display,
            main_window=self.main_window,
            parent=self,
        )
        dialog.exec()

    # ── Confirm ──
    def _on_confirm(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if not exp._raw_channels and exp._raw_volume is None:
            QMessageBox.information(self, "Nothing to import",
                                    "Browse for an ND2 / TIFF file first.")
            return

        # save_to_experiment rebuilds channels from volume with current z_mode.
        self.save_to_experiment(exp)

        if not exp._raw_channels:
            QMessageBox.information(self, "Nothing to import",
                                    "Browse for an ND2 / TIFF file first.")
            return

        self.main_window.exp_manager.set_status("imported")
        self.main_window.set_status_text("Imported.")
