"""
Batch page (V1.23).

Load a saved .nd2st.json pipeline template, select a folder or individual
files, and run every file through the full pipeline (load → recipe →
analysis → measurements).  All per-file result rows are aggregated into a
single CSV in the chosen output directory.

Optionally exports per-file overlay images (TIFF or JPG) alongside the CSV.
The "Save as Template" button captures the current session's pipeline into a
new .nd2st.json file so the user can reuse it here or share it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.core.settings import Settings


class BatchPage(QWidget):
    """Page 6: Batch — pipeline template runner over multiple files."""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._template: Optional[Dict[str, Any]] = None
        self._template_path: str = ""
        self._worker = None
        self._exp: Optional[ND2StudiosRecord] = None
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Template section ──
        tmpl_group = QGroupBox("Pipeline Template")
        tl = QVBoxLayout(tmpl_group)

        path_row = QHBoxLayout()
        self._edit_template = QLineEdit()
        self._edit_template.setPlaceholderText("No template loaded…")
        self._edit_template.setReadOnly(True)
        path_row.addWidget(self._edit_template, stretch=1)
        btn_browse_tmpl = QPushButton("Browse…")
        btn_browse_tmpl.clicked.connect(self._on_browse_template)
        path_row.addWidget(btn_browse_tmpl)
        tl.addLayout(path_row)

        tmpl_info_row = QHBoxLayout()
        self._lbl_tmpl_name = QLabel("—")
        self._lbl_tmpl_name.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        )
        tmpl_info_row.addWidget(self._lbl_tmpl_name, stretch=1)
        self._btn_save_tmpl = QPushButton("Save current session as template…")
        self._btn_save_tmpl.clicked.connect(self._on_save_template)
        tmpl_info_row.addWidget(self._btn_save_tmpl)
        tl.addLayout(tmpl_info_row)

        root.addWidget(tmpl_group)

        # ── Files section ──
        files_group = QGroupBox("Files")
        fl = QVBoxLayout(files_group)

        file_btns = QHBoxLayout()
        btn_add_files = QPushButton("Add Files…")
        btn_add_files.clicked.connect(self._on_add_files)
        file_btns.addWidget(btn_add_files)
        btn_add_folder = QPushButton("Add Folder…")
        btn_add_folder.clicked.connect(self._on_add_folder)
        file_btns.addWidget(btn_add_folder)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear_files)
        file_btns.addWidget(btn_clear)
        file_btns.addStretch(1)
        self._lbl_file_count = QLabel("0 files")
        self._lbl_file_count.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        )
        file_btns.addWidget(self._lbl_file_count)
        fl.addLayout(file_btns)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._file_list.setMaximumHeight(180)
        fl.addWidget(self._file_list)

        remove_row = QHBoxLayout()
        btn_remove_sel = QPushButton("Remove Selected")
        btn_remove_sel.clicked.connect(self._on_remove_selected)
        remove_row.addWidget(btn_remove_sel)
        remove_row.addStretch(1)
        fl.addLayout(remove_row)

        root.addWidget(files_group)

        # ── Output section ──
        out_group = QGroupBox("Output")
        ol = QVBoxLayout(out_group)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Directory:"))
        self._edit_output = QLineEdit()
        self._edit_output.setPlaceholderText("Choose output directory…")
        out_row.addWidget(self._edit_output, stretch=1)
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.clicked.connect(self._on_browse_output)
        out_row.addWidget(btn_browse_out)
        ol.addLayout(out_row)

        img_row = QHBoxLayout()
        self._chk_export_images = QCheckBox("Export per-file overlay images")
        img_row.addWidget(self._chk_export_images)
        img_row.addWidget(QLabel("Format:"))
        self._combo_fmt = QComboBox()
        self._combo_fmt.addItems(["TIFF", "JPG"])
        img_row.addWidget(self._combo_fmt)
        img_row.addStretch(1)
        ol.addLayout(img_row)

        root.addWidget(out_group)

        # ── Run section ──
        run_group = QGroupBox("Run")
        rl = QVBoxLayout(run_group)

        run_btns = QHBoxLayout()
        self._btn_run = QPushButton("Run Batch")
        self._btn_run.setObjectName("successBtn")
        self._btn_run.clicked.connect(self._on_run)
        run_btns.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        run_btns.addWidget(self._btn_cancel)
        run_btns.addStretch(1)
        rl.addLayout(run_btns)

        self._progress_bar = QProgressBar()
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        rl.addWidget(self._progress_bar)

        self._lbl_status = QLabel("Ready.")
        self._lbl_status.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;"
        )
        rl.addWidget(self._lbl_status)

        root.addWidget(run_group)
        root.addStretch(1)

    # ── Page lifecycle ────────────────────────────────────────────────────────

    def on_activated(self) -> None:
        pass

    def load_from_experiment(self, exp: ND2StudiosRecord) -> None:
        self._exp = exp
        cfg = exp.batch_config
        if cfg.get("template_path") and os.path.isfile(cfg["template_path"]):
            self._load_template_from_path(cfg["template_path"])
        if cfg.get("output_dir"):
            self._edit_output.setText(cfg["output_dir"])
        if cfg.get("image_format"):
            idx = self._combo_fmt.findText(cfg["image_format"].upper())
            if idx >= 0:
                self._combo_fmt.setCurrentIndex(idx)
        self._chk_export_images.setChecked(bool(cfg.get("export_images", False)))

    def save_to_experiment(self, exp: ND2StudiosRecord) -> None:
        exp.batch_config = {
            "template_path": self._template_path,
            "output_dir": self._edit_output.text().strip(),
            "image_format": self._combo_fmt.currentText().lower(),
            "export_images": self._chk_export_images.isChecked(),
        }

    # ── Template I/O ──────────────────────────────────────────────────────────

    def _on_browse_template(self) -> None:
        from nd2studios.backend.template import TEMPLATE_EXTENSION
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline Template", "",
            f"ND2Studios Template (*{TEMPLATE_EXTENSION});;All files (*)",
        )
        if path:
            self._load_template_from_path(path)

    def _load_template_from_path(self, path: str) -> None:
        from nd2studios.backend.template import load_template
        try:
            self._template = load_template(path)
            self._template_path = path
            self._edit_template.setText(path)
            name = self._template.get("name", os.path.basename(path))
            created = self._template.get("created", "")
            self._lbl_tmpl_name.setText(
                f"{name}  ·  created {created[:10]}" if created else name
            )
        except Exception as exc:
            QMessageBox.warning(self, "Template Load Failed", str(exc))

    def _on_save_template(self) -> None:
        from nd2studios.backend.template import (
            save_template, write_template, TEMPLATE_EXTENSION,
        )
        exp = self._exp
        if exp is None and self.main_window is not None:
            exp = self.main_window.exp_manager.active
        if exp is None:
            QMessageBox.information(
                self, "No Session", "No active session to save as template."
            )
            return

        name, ok = _simple_text_input(
            self, "Template Name", "Name for this template:",
            default=exp.name or "My Pipeline",
        )
        if not ok or not name.strip():
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Template", name.replace(" ", "_"),
            f"ND2Studios Template (*{TEMPLATE_EXTENSION})",
        )
        if not path:
            return

        try:
            tmpl = save_template(
                name=name.strip(),
                import_config=dict(exp.import_config or {}),
                recipe=list(exp.recipe or []),
                recipe_normalized=bool(exp.recipe_normalized),
                analysis_config=dict(exp.analysis_config or {}),
                results_config=dict(exp.results_config or {}),
            )
            final_path = write_template(tmpl, path)
            self._load_template_from_path(final_path)
            if self.main_window is not None:
                self.main_window.set_status_text(
                    f"Template saved: {os.path.basename(final_path)}"
                )
        except Exception as exc:
            QMessageBox.warning(self, "Save Failed", str(exc))

    # ── File queue ────────────────────────────────────────────────────────────

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Files", "",
            "Microscopy files (*.nd2 *.tif *.tiff);;All files (*)",
        )
        self._add_paths(paths)

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add Folder")
        if not folder:
            return
        paths = []
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".nd2", ".tif", ".tiff")):
                paths.append(os.path.join(folder, fname))
        if not paths:
            QMessageBox.information(
                self, "No Files Found",
                "No ND2 or TIFF files found in the selected folder."
            )
            return
        self._add_paths(paths)

    def _add_paths(self, paths: List[str]) -> None:
        existing = {
            self._file_list.item(i).data(Qt.UserRole)
            for i in range(self._file_list.count())
        }
        for p in paths:
            if p not in existing:
                item = QListWidgetItem(os.path.basename(p))
                item.setData(Qt.UserRole, p)
                item.setToolTip(p)
                self._file_list.addItem(item)
                existing.add(p)
        self._update_file_count()

    def _on_remove_selected(self) -> None:
        for item in self._file_list.selectedItems():
            self._file_list.takeItem(self._file_list.row(item))
        self._update_file_count()

    def _on_clear_files(self) -> None:
        self._file_list.clear()
        self._update_file_count()

    def _update_file_count(self) -> None:
        n = self._file_list.count()
        self._lbl_file_count.setText(f"{n} file{'s' if n != 1 else ''}")

    # ── Output directory ──────────────────────────────────────────────────────

    def _on_browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self._edit_output.setText(folder)

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        from nd2studios.workers.batch_worker import BatchWorker

        # Validate inputs
        if self._template is None:
            QMessageBox.information(
                self, "No Template",
                "Load or save a pipeline template first."
            )
            return

        filepaths = [
            self._file_list.item(i).data(Qt.UserRole)
            for i in range(self._file_list.count())
        ]
        if not filepaths:
            QMessageBox.information(
                self, "No Files",
                "Add at least one file to process."
            )
            return

        output_dir = self._edit_output.text().strip()
        if not output_dir:
            QMessageBox.information(
                self, "No Output Directory",
                "Choose an output directory."
            )
            return

        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._lbl_status.setText("Starting…")

        self._worker = BatchWorker(
            filepaths=filepaths,
            template=self._template,
            output_dir=output_dir,
            export_images=self._chk_export_images.isChecked(),
            image_format=self._combo_fmt.currentText().lower(),
        )
        self._worker.progress.connect(self._progress_bar.setValue)
        self._worker.status.connect(self._lbl_status.setText)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._lbl_status.setText("Cancelling…")
            self._btn_cancel.setEnabled(False)

    def _on_file_done(self, done: int, total: int) -> None:
        self._lbl_status.setText(f"Processed {done} / {total} files…")
        if self.main_window is not None:
            self.main_window.set_progress(
                int(done / max(total, 1) * 100),
                f"Batch: {done}/{total}",
            )

    def _on_finished(self, csv_path: str) -> None:
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress_bar.setVisible(False)
        if self.main_window is not None:
            self.main_window.set_progress(0)

        if csv_path and os.path.isfile(csv_path):
            self._lbl_status.setText(
                f"Done. Results → {os.path.basename(csv_path)}"
            )
            reply = QMessageBox.question(
                self, "Batch Complete",
                f"Batch finished.\n\nResults CSV:\n{csv_path}\n\n"
                "Open the output folder?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                _open_folder(os.path.dirname(csv_path))
        else:
            self._lbl_status.setText("Done. (No objects detected or no analysis pipeline set.)")

    def _on_error(self, msg: str) -> None:
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress_bar.setVisible(False)
        self._lbl_status.setText(f"Error: {msg[:80]}")
        QMessageBox.warning(self, "Batch Error", msg)
        if self.main_window is not None:
            self.main_window.set_progress(0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simple_text_input(parent, title: str, label: str, default: str = "") -> tuple:
    """Minimal text input dialog — avoids importing QInputDialog everywhere."""
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    layout = QFormLayout(dlg)
    edit = QLineEdit(default)
    layout.addRow(label, edit)
    btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)
    layout.addRow(btns)
    ok = dlg.exec() == QDialog.Accepted
    return edit.text(), ok


def _open_folder(path: str) -> None:
    """Open a folder in the system file manager."""
    import subprocess
    import sys
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass
