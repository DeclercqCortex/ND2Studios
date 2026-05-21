"""
Results page (V1.22).

Displays extended per-object measurements computed from the AnalysisResult
label masks produced by the Analysis page.  Measurements are shown in a
sortable table alongside a summary panel and can be exported to CSV.
Images with the label overlay can be exported as TIFF or JPG.

Prerequisite: at least one analysis pipeline must have been run for the
active session (exp.analysis_results must be non-empty).
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QSortFilterProxyModel, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QAbstractItemView,
)

from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.core.settings import Settings


# ── Minimal read-only table model ────────────────────────────────────────────

class _MeasurementsModel(QAbstractTableModel):
    """Flat list-of-dicts → sortable QAbstractTableModel."""

    def __init__(self, rows: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self._rows = rows
        self._headers: List[str] = list(rows[0].keys()) if rows else []

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return len(self._headers)

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):  # noqa: N802
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._headers[section]
            return str(section + 1)
        return None

    def data(self, index: QModelIndex, role=Qt.DisplayRole):  # noqa: N802
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.UserRole):
            return None
        val = self._rows[index.row()].get(self._headers[index.column()])
        if role == Qt.UserRole:
            return val
        if val is None:
            return "—"
        if isinstance(val, float):
            return f"{val:.4g}"
        return str(val)

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        if column >= len(self._headers):
            return
        key = self._headers[column]
        reverse = order == Qt.DescendingOrder
        self.layoutAboutToBeChanged.emit()
        self._rows.sort(
            key=lambda r: (r.get(key) is None, r.get(key) or 0),
            reverse=reverse,
        )
        self.layoutChanged.emit()


# ── Results page ──────────────────────────────────────────────────────────────

class ResultsPage(QWidget):
    """Page 5: Results — measurements from analysis binaries."""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._measurements: List[Dict[str, Any]] = []
        self._current_pipeline: str = ""
        self._exp: Optional[ND2StudiosRecord] = None
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Top controls row ──
        ctrl_row = QHBoxLayout()

        ctrl_row.addWidget(QLabel("Pipeline:"))
        self._combo_pipeline = QComboBox()
        self._combo_pipeline.setMinimumWidth(220)
        self._combo_pipeline.currentTextChanged.connect(self._on_pipeline_changed)
        ctrl_row.addWidget(self._combo_pipeline)

        self._btn_compute = QPushButton("Compute Measurements")
        self._btn_compute.setObjectName("primaryBtn")
        self._btn_compute.clicked.connect(self._on_compute)
        ctrl_row.addWidget(self._btn_compute)

        ctrl_row.addStretch(1)

        self._lbl_count = QLabel("")
        self._lbl_count.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        ctrl_row.addWidget(self._lbl_count)

        root.addLayout(ctrl_row)

        # ── Splitter: table + summary ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Measurements table
        table_widget = QWidget()
        tl = QVBoxLayout(table_widget)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(QLabel("Measurements", objectName="sectionHeader"))
        self._table = QTableView()
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        tl.addWidget(self._table, stretch=1)
        splitter.addWidget(table_widget)

        # Summary panel
        summary_group = QGroupBox("Summary")
        sl = QVBoxLayout(summary_group)
        self._summary_labels: Dict[str, QLabel] = {}
        for key in (
            "n_objects",
            "n_frames",
            "mean_area_um2",
            "std_area_um2",
        ):
            row_w = QWidget()
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl_key = QLabel(key.replace("_", " ").title() + ":")
            lbl_key.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
            lbl_key.setMinimumWidth(130)
            lbl_val = QLabel("—")
            rl.addWidget(lbl_key)
            rl.addWidget(lbl_val, stretch=1)
            self._summary_labels[key] = lbl_val
            sl.addWidget(row_w)

        # Dynamic channel intensity summary labels added at compute time
        self._dynamic_summary_container = QVBoxLayout()
        sl.addLayout(self._dynamic_summary_container)
        sl.addStretch(1)
        summary_group.setMinimumWidth(240)
        summary_group.setMaximumWidth(340)
        splitter.addWidget(summary_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, stretch=1)

        # ── Export row ──
        export_row = QHBoxLayout()
        self._btn_export_csv = QPushButton("Export CSV")
        self._btn_export_csv.setEnabled(False)
        self._btn_export_csv.clicked.connect(self._on_export_csv)
        export_row.addWidget(self._btn_export_csv)

        self._btn_export_masks = QPushButton("Export Label Masks (TIFF)")
        self._btn_export_masks.setEnabled(False)
        self._btn_export_masks.clicked.connect(self._on_export_masks)
        export_row.addWidget(self._btn_export_masks)

        self._combo_img_fmt = QComboBox()
        self._combo_img_fmt.addItems(["TIFF", "JPG"])
        export_row.addWidget(self._combo_img_fmt)
        self._btn_export_images = QPushButton("Export Overlay Images")
        self._btn_export_images.setEnabled(False)
        self._btn_export_images.clicked.connect(self._on_export_images)
        export_row.addWidget(self._btn_export_images)

        export_row.addStretch(1)
        root.addLayout(export_row)

        # ── Empty-state label ──
        self._lbl_empty = QLabel(
            "Run an analysis pipeline on the Analysis page first,\n"
            "then click 'Compute Measurements'."
        )
        self._lbl_empty.setAlignment(Qt.AlignCenter)
        self._lbl_empty.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 11pt;"
        )
        root.addWidget(self._lbl_empty)
        self._lbl_empty.setVisible(False)

    # ── Page lifecycle ────────────────────────────────────────────────────────

    def on_activated(self) -> None:
        exp = self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        self._refresh_pipeline_combo(exp)

    def load_from_experiment(self, exp: ND2StudiosRecord) -> None:
        self._exp = exp
        cfg = exp.results_config
        if cfg.get("pipeline"):
            idx = self._combo_pipeline.findText(cfg["pipeline"])
            if idx >= 0:
                self._combo_pipeline.setCurrentIndex(idx)
        self._refresh_pipeline_combo(exp)

    def save_to_experiment(self, exp: ND2StudiosRecord) -> None:
        exp.results_config = {
            "pipeline": self._combo_pipeline.currentText(),
            "image_format": self._combo_img_fmt.currentText().lower(),
        }

    # ── Pipeline combo ────────────────────────────────────────────────────────

    def _refresh_pipeline_combo(self, exp: Optional[ND2StudiosRecord]) -> None:
        self._combo_pipeline.blockSignals(True)
        prev = self._combo_pipeline.currentText()
        self._combo_pipeline.clear()
        if exp is not None and exp.analysis_results:
            for key in exp.analysis_results:
                self._combo_pipeline.addItem(key)
            idx = self._combo_pipeline.findText(prev)
            if idx >= 0:
                self._combo_pipeline.setCurrentIndex(idx)
        self._combo_pipeline.blockSignals(False)
        has_results = bool(exp is not None and exp.analysis_results)
        self._btn_compute.setEnabled(has_results)
        self._lbl_empty.setVisible(not has_results)

    def _on_pipeline_changed(self, name: str) -> None:
        self._current_pipeline = name
        # Clear stale measurements when the pipeline changes.
        self._measurements = []
        self._update_table([])
        self._update_summary([])
        self._btn_export_csv.setEnabled(False)
        self._btn_export_masks.setEnabled(False)
        self._btn_export_images.setEnabled(False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_result(self, exp: ND2StudiosRecord, pipeline_name: str):
        """Return the AnalysisResult for the current m_index.

        analysis_results[pipeline_name] is stored as Dict[int, AnalysisResult]
        by AnalysisPage._on_finished.  Unwrap it here so the rest of the page
        never has to know about the nesting.
        """
        from nd2studios.core.analysis_registry import AnalysisResult
        stored = exp.analysis_results.get(pipeline_name)
        if stored is None:
            return None
        if isinstance(stored, AnalysisResult):
            return stored
        if isinstance(stored, dict):
            m = int(exp.m_index)
            result = stored.get(m)
            if result is None:
                result = next(iter(stored.values()), None)
            return result
        return None

    def _channels_for_m(self, exp: ND2StudiosRecord, m: int) -> Dict[str, Any]:
        """Return materialised (T, H, W) channel arrays for a single M position."""
        vol = getattr(exp, "_raw_volume", None)
        if vol is not None and exp._processed_channels is None:
            z_mode = getattr(exp, "z_view_mode", None) or "max"
            z_index = int(getattr(exp, "z_view_index", None) or 0)
            return {
                ch_name: vol.to_lazy_channel(
                    c_idx, m=m, z_mode=z_mode, z_index=z_index
                ).materialize()
                for c_idx, ch_name in enumerate(vol.channel_names)
            }
        channels = exp._processed_channels or exp._raw_channels or {}
        result: Dict[str, Any] = {}
        for k, v in channels.items():
            if hasattr(v, "materialize") and callable(v.materialize):
                result[k] = v.materialize()
            else:
                result[k] = np.asarray(v)
        return result

    # ── Compute ───────────────────────────────────────────────────────────────

    def _on_compute(self) -> None:
        from nd2studios.backend.results_engine import compute_measurements
        from nd2studios.core.analysis_registry import AnalysisResult

        exp = self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        if exp is None:
            return

        pipeline_name = self._combo_pipeline.currentText()
        if not pipeline_name or pipeline_name not in exp.analysis_results:
            QMessageBox.information(
                self, "No Analysis Result",
                "Run an analysis pipeline on the Analysis page first."
            )
            return

        stored = exp.analysis_results.get(pipeline_name)
        if isinstance(stored, AnalysisResult):
            results_by_m = {0: stored}
        elif isinstance(stored, dict):
            results_by_m = {m: r for m, r in stored.items() if isinstance(r, AnalysisResult)}
        else:
            results_by_m = {}

        if not results_by_m:
            QMessageBox.information(
                self, "No Analysis Result",
                "No result found for the selected pipeline."
            )
            return

        metadata = dict(exp.nd2_metadata or {})
        metadata.setdefault("pixel_size_um", exp.pixel_size_um)

        multi_m = len(results_by_m) > 1
        all_rows: List[Dict[str, Any]] = []

        for m, result in sorted(results_by_m.items()):
            channels_for_m = self._channels_for_m(exp, m)
            try:
                rows = compute_measurements(
                    result.label_masks,
                    channels_for_m,
                    metadata,
                    m_index=m,
                    volumetric_voxel_counts=result.volumetric_voxel_counts,
                )
            except Exception as exc:
                QMessageBox.warning(self, "Compute Error", str(exc))
                return
            if multi_m:
                for row in rows:
                    row["m_position"] = m
            all_rows.extend(rows)

        self._measurements = all_rows
        self._update_table(all_rows)
        self._update_summary(all_rows)

        first_result = next(iter(results_by_m.values()))
        has_data = bool(all_rows)
        self._btn_export_csv.setEnabled(has_data)
        self._btn_export_masks.setEnabled(bool(first_result.label_masks))
        self._btn_export_images.setEnabled(has_data)
        self._lbl_count.setText(f"{len(all_rows)} objects")

        if self.main_window is not None:
            self.main_window.set_status_text(
                f"Results: {len(all_rows)} objects from '{pipeline_name}'."
            )

    # ── Table update ─────────────────────────────────────────────────────────

    def _update_table(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            self._table.setModel(None)
            return
        model = _MeasurementsModel(rows, parent=self)
        proxy = QSortFilterProxyModel(self)
        proxy.setSourceModel(model)
        self._table.setModel(proxy)
        self._table.resizeColumnsToContents()
        # Cap column width so wide columns don't dominate.
        for c in range(proxy.columnCount()):
            w = self._table.columnWidth(c)
            self._table.setColumnWidth(c, min(w, 160))

    # ── Summary update ────────────────────────────────────────────────────────

    def _update_summary(self, rows: List[Dict[str, Any]]) -> None:
        # Clear dynamic channel labels
        while self._dynamic_summary_container.count():
            item = self._dynamic_summary_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not rows:
            for lbl in self._summary_labels.values():
                lbl.setText("—")
            return

        areas = [r.get("area_um2") for r in rows if r.get("area_um2") is not None]
        frames = {r.get("frame") for r in rows}

        self._summary_labels["n_objects"].setText(str(len(rows)))
        self._summary_labels["n_frames"].setText(str(len(frames)))
        if areas:
            arr = np.asarray(areas, dtype=float)
            self._summary_labels["mean_area_um2"].setText(f"{arr.mean():.3f} µm²")
            self._summary_labels["std_area_um2"].setText(f"{arr.std():.3f} µm²")
        else:
            self._summary_labels["mean_area_um2"].setText("—")
            self._summary_labels["std_area_um2"].setText("—")

        # Dynamic: mean intensity per channel found in rows
        intensity_keys = sorted({
            k for r in rows for k in r if k.startswith("mean_intensity_")
        })
        for key in intensity_keys:
            ch_label = key[len("mean_intensity_"):].replace("_", " ")
            vals = [r[key] for r in rows if r.get(key) is not None]
            if not vals:
                continue
            mean_v = float(np.mean(vals))
            row_w = QWidget()
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl_key = QLabel(f"Mean intensity ({ch_label}):")
            lbl_key.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
            lbl_val = QLabel(f"{mean_v:.3f}")
            rl.addWidget(lbl_key)
            rl.addWidget(lbl_val, stretch=1)
            self._dynamic_summary_container.addWidget(row_w)

    # ── Export: CSV ───────────────────────────────────────────────────────────

    def _on_export_csv(self) -> None:
        if not self._measurements:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Measurements CSV", "measurements.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        fieldnames = list(self._measurements[0].keys())
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self._measurements)
            if self.main_window is not None:
                self.main_window.set_status_text(
                    f"CSV saved: {os.path.basename(path)}"
                )
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))

    # ── Export: label masks ───────────────────────────────────────────────────

    def _on_export_masks(self) -> None:
        from nd2studios.backend.results_engine import export_label_masks_tiff
        from nd2studios.core.analysis_registry import AnalysisResult

        exp = self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        if exp is None:
            return
        pipeline_name = self._combo_pipeline.currentText()
        stored = exp.analysis_results.get(pipeline_name) if exp else None
        if stored is None:
            return

        if isinstance(stored, AnalysisResult):
            results_by_m = {0: stored}
        elif isinstance(stored, dict):
            results_by_m = {m: r for m, r in stored.items() if isinstance(r, AnalysisResult)}
        else:
            return
        if not results_by_m:
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out_dir:
            return

        try:
            multi_m = len(results_by_m) > 1
            total_paths: List[str] = []
            for m, result in sorted(results_by_m.items()):
                if not result.label_masks:
                    continue
                target = os.path.join(out_dir, f"M{m:02d}") if multi_m else out_dir
                if multi_m:
                    os.makedirs(target, exist_ok=True)
                paths = export_label_masks_tiff(result.label_masks, target)
                total_paths.extend(paths)
            if self.main_window is not None:
                self.main_window.set_status_text(
                    f"Masks saved: {len(total_paths)} file(s) → {out_dir}"
                )
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))

    # ── Export: overlay images ────────────────────────────────────────────────

    def _on_export_images(self) -> None:
        from nd2studios.backend.results_engine import export_overlay_frames
        from nd2studios.core.analysis_registry import AnalysisResult

        exp = self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        if exp is None:
            return
        pipeline_name = self._combo_pipeline.currentText()
        stored = exp.analysis_results.get(pipeline_name) if exp else None
        if stored is None:
            return

        if isinstance(stored, AnalysisResult):
            results_by_m = {0: stored}
        elif isinstance(stored, dict):
            results_by_m = {m: r for m, r in stored.items() if isinstance(r, AnalysisResult)}
        else:
            return
        if not results_by_m:
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out_dir:
            return

        fmt = self._combo_img_fmt.currentText().lower()
        metadata = dict(exp.nd2_metadata or {})

        try:
            multi_m = len(results_by_m) > 1
            total_paths: List[str] = []
            for m, result in sorted(results_by_m.items()):
                if not result.label_masks:
                    continue
                channels_for_m = self._channels_for_m(exp, m)
                target = os.path.join(out_dir, f"M{m:02d}") if multi_m else out_dir
                if multi_m:
                    os.makedirs(target, exist_ok=True)
                paths = export_overlay_frames(
                    channels=channels_for_m,
                    label_masks=result.label_masks,
                    metadata=metadata,
                    output_dir=target,
                    fmt=fmt,
                    channel_display=exp.channel_display or {},
                )
                total_paths.extend(paths)
            if self.main_window is not None:
                self.main_window.set_status_text(
                    f"Images saved: {len(total_paths)} frame(s) → {out_dir}"
                )
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))
