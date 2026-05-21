"""
Recipe page — build a linear pipeline of enhancement steps with the
trial / accept / reject pattern adapted from CellTracker.

Workflow:
1. User picks a plugin from the combo box.
2. The `ParamEditor` shows that plugin's parameters.
3. "Trial" runs the *full current pipeline + the trial step* on the raw
   channels. The result lands on the experiment as
   `_processed_channels` and is shown in the preview.
4. "Accept" appends the trial step to the recipe and locks the trial
   result as the new committed state.
5. "Reject" discards the trial; the committed state is unchanged.
6. "Remove Last" pops the last step from the recipe and reprocesses.
7. "Save Recipe" / "Load Recipe" persist the recipe (without dataset)
   as `.nd2s_recipe.json` so it can be applied to other ND2 files.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.core.plugin_registry import PluginBase
from nd2studios.core.settings import Settings
from nd2studios.widgets.common import ParamEditor
from nd2studios.widgets.image_viewer import CHANNEL_COLORS
from nd2studios.widgets.multi_axis_viewer import MultiAxisViewer
from nd2studios.workers.recipe_worker import RecipeWorker
from nd2studios.backend.recipes import (
    save_recipe as save_recipe_json,
    load_recipe as load_recipe_json,
    RECIPE_EXTENSION,
)


class RecipePage(QWidget):
    """Page 2: build a processing recipe."""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        # Committed recipe (list of (plugin_name, params) pairs).
        self._recipe: List[Tuple[str, Dict[str, Any]]] = []
        self._normalized: bool = False
        # Trial state.
        self._trial_step: Optional[Tuple[str, Dict[str, Any]]] = None
        self._worker: Optional[RecipeWorker] = None
        # Crop state — in-session only, not persisted.
        self._crop_rect: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h

        self._build_ui()
        self._populate_plugin_list()

    # ── UI ──
    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # Left column: plugin picker + params + buttons.
        left = QWidget()
        left.setFixedWidth(420)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        # Plugin picker
        pick_group = QGroupBox("Add a step")
        pl = QVBoxLayout(pick_group)
        self.combo_plugin = QComboBox()
        self.combo_plugin.currentTextChanged.connect(self._on_plugin_changed)
        pl.addWidget(self.combo_plugin)
        self.lbl_plugin_desc = QLabel("")
        self.lbl_plugin_desc.setWordWrap(True)
        self.lbl_plugin_desc.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        pl.addWidget(self.lbl_plugin_desc)
        self.param_editor = ParamEditor()
        pl.addWidget(self.param_editor)
        ll.addWidget(pick_group)

        # Trial / accept / reject row
        trial_row = QHBoxLayout()
        self.btn_trial = QPushButton("Trial")
        self.btn_trial.setObjectName("primaryBtn")
        self.btn_trial.clicked.connect(self._on_trial)
        trial_row.addWidget(self.btn_trial)
        self.btn_accept = QPushButton("Accept")
        self.btn_accept.setObjectName("successBtn")
        self.btn_accept.setEnabled(False)
        self.btn_accept.clicked.connect(self._on_accept)
        trial_row.addWidget(self.btn_accept)
        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setObjectName("dangerBtn")
        self.btn_reject.setEnabled(False)
        self.btn_reject.clicked.connect(self._on_reject)
        trial_row.addWidget(self.btn_reject)
        ll.addLayout(trial_row)

        # Crop group
        crop_group = QGroupBox("Crop")
        cl = QVBoxLayout(crop_group)
        self.btn_crop_mode = QPushButton("Crop Mode")
        self.btn_crop_mode.setCheckable(True)
        self.btn_crop_mode.setToolTip(
            "Enable crop tool on the Raw viewer.\n"
            "Click anywhere to enter the corner and size manually.\n"
            "Click-drag to draw the crop rectangle — values are pre-filled."
        )
        cl.addWidget(self.btn_crop_mode)
        self.lbl_crop_status = QLabel("No crop applied")
        self.lbl_crop_status.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        cl.addWidget(self.lbl_crop_status)
        self.btn_reset_crop = QPushButton("Reset Crop")
        self.btn_reset_crop.setEnabled(False)
        cl.addWidget(self.btn_reset_crop)
        ll.addWidget(crop_group)

        # Recipe list
        rec_group = QGroupBox("Recipe")
        rl = QVBoxLayout(rec_group)
        self.cb_normalized = QCheckBox("Frame-mean normalization (applied first)")
        self.cb_normalized.stateChanged.connect(self._on_normalized_changed)
        rl.addWidget(self.cb_normalized)
        self.list_recipe = QListWidget()
        rl.addWidget(self.list_recipe)
        rec_buttons = QHBoxLayout()
        self.btn_remove = QPushButton("Remove Last")
        self.btn_remove.clicked.connect(self._on_remove_last)
        rec_buttons.addWidget(self.btn_remove)
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._on_clear)
        rec_buttons.addWidget(self.btn_clear)
        rl.addLayout(rec_buttons)
        save_load = QHBoxLayout()
        self.btn_save_recipe = QPushButton("Save Recipe…")
        self.btn_save_recipe.clicked.connect(self._on_save_recipe)
        save_load.addWidget(self.btn_save_recipe)
        self.btn_load_recipe = QPushButton("Load Recipe…")
        self.btn_load_recipe.clicked.connect(self._on_load_recipe)
        save_load.addWidget(self.btn_load_recipe)
        rl.addLayout(save_load)
        ll.addWidget(rec_group)
        ll.addStretch(1)
        outer.addWidget(left)

        # Right column: before / after preview.
        right = QSplitter(Qt.Orientation.Horizontal)
        before_w = QWidget()
        bl = QVBoxLayout(before_w)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(QLabel("Raw", objectName="sectionHeader"))
        self.viewer_raw = MultiAxisViewer(before_w)
        bl.addWidget(self.viewer_raw, stretch=1)
        right.addWidget(before_w)

        after_w = QWidget()
        al = QVBoxLayout(after_w)
        al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(QLabel("Processed (current state + trial)",
                            objectName="sectionHeader"))
        self.viewer_proc = MultiAxisViewer(after_w)
        al.addWidget(self.viewer_proc, stretch=1)
        right.addWidget(after_w)

        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 1)
        outer.addWidget(right, stretch=1)

        # Crop signal wiring (viewers exist now)
        self.btn_crop_mode.toggled.connect(self._on_crop_mode_toggled)
        self.btn_reset_crop.clicked.connect(self._reset_crop)
        self.viewer_raw.canvas.clicked.connect(self._on_canvas_click)
        self.viewer_raw.crop_rect_selected.connect(self._on_crop_drag)

    # ── Crop ──
    def _on_crop_mode_toggled(self, enabled: bool) -> None:
        self.viewer_raw.set_crop_mode(enabled)
        if enabled:
            self.viewer_raw.zoom_toolbar.btn_pan.setChecked(False)

    def _on_canvas_click(self, iy: float, ix: float) -> None:
        if not self.btn_crop_mode.isChecked():
            return
        result = self._show_crop_dialog(x=int(ix), y=int(iy), w=0, h=0)
        if result is not None:
            self._apply_crop(*result)

    def _on_crop_drag(self, x: int, y: int, w: int, h: int) -> None:
        result = self._show_crop_dialog(x=x, y=y, w=w, h=h)
        if result is not None:
            self._apply_crop(*result)

    def _show_crop_dialog(
        self, x: int, y: int, w: int, h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Open a dialog for the user to confirm/edit the crop rectangle.

        Returns (x, y, w, h) or None if cancelled.
        """
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return None
        exp = self.main_window.exp_manager.active
        source = exp._original_raw_channels or exp._raw_channels
        if not source:
            return None

        # Determine current image dimensions from the *original* (pre-crop) source.
        sample = next(iter(source.values()))
        if hasattr(sample, "shape"):
            arr_shape = sample.shape
        else:
            arr_shape = np.asarray(sample).shape
        img_h = int(arr_shape[-2])
        img_w = int(arr_shape[-1])

        dlg = QDialog(self)
        dlg.setWindowTitle("Crop Image")
        layout = QVBoxLayout(dlg)

        info = QLabel(f"Image: {img_w} × {img_h} px  "
                      f"(X = columns from left, Y = rows from top)")
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        layout.addWidget(info)

        form = QFormLayout()
        sp_x = QSpinBox()
        sp_x.setRange(0, img_w - 1)
        sp_x.setValue(max(0, min(x, img_w - 1)))
        sp_x.setToolTip("Left edge of crop (pixels from image left)")

        sp_y = QSpinBox()
        sp_y.setRange(0, img_h - 1)
        sp_y.setValue(max(0, min(y, img_h - 1)))
        sp_y.setToolTip("Top edge of crop (pixels from image top)")

        sp_w = QSpinBox()
        sp_w.setRange(1, img_w)
        sp_w.setValue(w if w > 0 else max(1, img_w - x))
        sp_w.setToolTip("Width of crop in pixels")

        sp_h = QSpinBox()
        sp_h.setRange(1, img_h)
        sp_h.setValue(h if h > 0 else max(1, img_h - y))
        sp_h.setToolTip("Height of crop in pixels")

        form.addRow("X (left corner):", sp_x)
        form.addRow("Y (top corner):", sp_y)
        form.addRow("Width (px):", sp_w)
        form.addRow("Height (px):", sp_h)
        layout.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        cx, cy, cw, ch = sp_x.value(), sp_y.value(), sp_w.value(), sp_h.value()
        if cx + cw > img_w or cy + ch > img_h:
            QMessageBox.warning(
                self, "Invalid crop",
                f"Crop region ({cx}+{cw}={cx+cw}, {cy}+{ch}={cy+ch}) "
                f"exceeds image bounds ({img_w}, {img_h}).\n"
                "Clamping to image boundary."
            )
            cw = min(cw, img_w - cx)
            ch = min(ch, img_h - cy)
        return (cx, cy, cw, ch)

    def _apply_crop(self, x: int, y: int, w: int, h: int) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if not exp._raw_channels:
            return

        # Always crop from original so the user can re-crop without nesting.
        if exp._original_raw_channels is None:
            exp._original_raw_channels = dict(exp._raw_channels)
        source = exp._original_raw_channels

        cropped: Dict[str, Any] = {}
        for ch_name, ch_data in source.items():
            crop_fn = getattr(ch_data, "crop", None)
            if callable(crop_fn):
                cropped[ch_name] = crop_fn(y, y + h, x, x + w)
            else:
                arr = np.asarray(ch_data)
                cropped[ch_name] = arr[:, y:y + h, x:x + w]

        self._crop_rect = (x, y, w, h)
        exp.crop_rect = (x, y, w, h)
        exp._raw_channels = cropped
        exp._processed_channels = None

        self.btn_crop_mode.setChecked(False)
        self.btn_reset_crop.setEnabled(True)
        self.lbl_crop_status.setText(f"Crop: x={x}, y={y}, {w}×{h} px")

        # After crop, always show the flat cropped channels (no Z volume path).
        self.viewer_raw.set_channels(cropped, channel_display=exp.channel_display)
        if self._recipe:
            self._run_recipe(exp, list(self._recipe),
                             label="Re-running recipe on cropped data…",
                             on_done=self._on_revert_done)
        else:
            self.viewer_proc.set_channels(cropped, channel_display=exp.channel_display)

    def _reset_crop(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if exp._original_raw_channels is None:
            return

        exp._raw_channels = exp._original_raw_channels
        exp._original_raw_channels = None
        exp._processed_channels = None
        exp.crop_rect = None
        self._crop_rect = None
        self.btn_reset_crop.setEnabled(False)
        self.lbl_crop_status.setText("No crop applied")

        self._refresh_raw_viewer(exp)
        if self._recipe:
            self._run_recipe(exp, list(self._recipe),
                             label="Re-running recipe after crop reset…",
                             on_done=self._on_revert_done)
        else:
            self.viewer_proc.set_channels(exp._raw_channels, channel_display=exp.channel_display)

    def _populate_plugin_list(self) -> None:
        # Force-import to ensure registration.
        import nd2studios.plugins.enhancement.builtin  # noqa: F401
        plugins = PluginBase.get_plugins("enhancement")
        names = sorted([p.name for p in plugins])
        self.combo_plugin.clear()
        self.combo_plugin.addItems(names)
        if names:
            self._on_plugin_changed(names[0])

    # ── Page lifecycle ──
    def on_activated(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        # Restore crop UI from the experiment record (survives tab navigation).
        # exp._original_raw_channels persists on the record for all file types,
        # so no reconstruction from _raw_volume is needed.
        self.btn_crop_mode.setChecked(False)
        if exp.crop_rect is not None:
            x, y, w, h = exp.crop_rect
            self._crop_rect = exp.crop_rect
            self.btn_reset_crop.setEnabled(True)
            self.lbl_crop_status.setText(f"Crop: x={x}, y={y}, {w}×{h} px")
        else:
            self._crop_rect = None
            self.btn_reset_crop.setEnabled(False)
            self.lbl_crop_status.setText("No crop applied")
        # Refresh the raw preview from the active experiment.
        if exp._raw_channels:
            self._refresh_raw_viewer(exp)
            # If a processed cache already exists, mirror it on the right.
            if exp._processed_channels:
                self.viewer_proc.set_channels(exp._processed_channels,
                                              channel_display=exp.channel_display)

    def _refresh_raw_viewer(self, exp: ND2StudiosRecord) -> None:
        """Wire the raw viewer to the volume (Z-scrollable) or flat channels.

        Volume path: z_mode="none", multi-Z volume, no crop active.
        Flat path: everything else (projection, TIFF source, or crop active).
        """
        if (exp.crop_rect is None
                and exp._raw_volume is not None
                and exp.n_zslices > 1
                and exp.z_view_mode == "none"):
            self.viewer_raw.set_volume(
                exp._raw_volume,
                channel_display=exp.channel_display,
                z_mode="none",
                z_index=exp.z_view_index,
                m=exp.m_index, t=0, z=exp.z_view_index,
            )
        else:
            self.viewer_raw.set_channels(exp._raw_channels,
                                         channel_display=exp.channel_display)

    def load_from_experiment(self, exp: ND2StudiosRecord) -> None:
        self._recipe = list(exp.recipe)
        self._normalized = bool(exp.recipe_normalized)
        self.cb_normalized.blockSignals(True)
        self.cb_normalized.setChecked(self._normalized)
        self.cb_normalized.blockSignals(False)
        self._refresh_recipe_list()

    def save_to_experiment(self, exp: ND2StudiosRecord) -> None:
        exp.recipe = list(self._recipe)
        exp.recipe_normalized = bool(self._normalized)

    # ── Plugin combo ──
    def _on_plugin_changed(self, name: str) -> None:
        cls = PluginBase.get_plugin("enhancement", name)
        if cls is None:
            return
        plugin = cls()
        self.lbl_plugin_desc.setText(getattr(plugin, "description", ""))
        self.param_editor.set_params(plugin.get_params())

    def _on_normalized_changed(self, _state: int) -> None:
        self._normalized = self.cb_normalized.isChecked()

    # ── Trial / Accept / Reject ──
    def _on_trial(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if not exp._raw_channels:
            QMessageBox.information(self, "Nothing to process",
                                    "Import a file first.")
            return

        plugin_name = self.combo_plugin.currentText()
        params = self.param_editor.get_values()
        # Run the full committed recipe + the trial step, on raw channels.
        full_recipe = list(self._recipe) + [(plugin_name, params)]
        self._trial_step = (plugin_name, params)

        self._run_recipe(exp, full_recipe, label="Trial running…",
                         on_done=self._on_trial_done)

    def _on_trial_done(self, processed: Dict[str, Any]) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        # Show in the right viewer; do NOT commit to the recipe yet.
        self.viewer_proc.set_channels(processed,
                                      channel_display=exp.channel_display)
        # Stash on the experiment so Accept can promote it cheaply.
        exp._processed_channels = processed
        self.btn_accept.setEnabled(True)
        self.btn_reject.setEnabled(True)

    def _on_accept(self) -> None:
        if self._trial_step is None:
            return
        self._recipe.append(self._trial_step)
        self._trial_step = None
        self.btn_accept.setEnabled(False)
        self.btn_reject.setEnabled(False)
        self._refresh_recipe_list()
        if self.main_window is not None:
            self.main_window.set_status_text(f"Step accepted ({len(self._recipe)} total).")
            self.main_window.exp_manager.set_status("preprocessed")

    def _on_reject(self) -> None:
        self._trial_step = None
        self.btn_accept.setEnabled(False)
        self.btn_reject.setEnabled(False)
        # Re-run committed recipe to revert the right-side preview.
        if self.main_window is not None and self.main_window.exp_manager.active is not None:
            exp = self.main_window.exp_manager.active
            if self._recipe:
                self._run_recipe(exp, list(self._recipe),
                                 label="Reverting trial…",
                                 on_done=self._on_revert_done)
            else:
                # No committed recipe → mirror raw on the right.
                exp._processed_channels = None
                if exp._raw_channels:
                    self.viewer_proc.set_channels(exp._raw_channels,
                                                  channel_display=exp.channel_display)

    def _on_revert_done(self, processed: Dict[str, Any]) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        exp._processed_channels = processed
        self.viewer_proc.set_channels(processed,
                                      channel_display=exp.channel_display)

    # ── Remove / Clear ──
    def _on_remove_last(self) -> None:
        if not self._recipe:
            return
        self._recipe.pop()
        self._refresh_recipe_list()
        # Reprocess.
        if self.main_window is not None and self.main_window.exp_manager.active is not None:
            exp = self.main_window.exp_manager.active
            if self._recipe:
                self._run_recipe(exp, list(self._recipe),
                                 label="Reprocessing…",
                                 on_done=self._on_revert_done)
            else:
                exp._processed_channels = None
                if exp._raw_channels:
                    self.viewer_proc.set_channels(exp._raw_channels,
                                                  channel_display=exp.channel_display)

    def _on_clear(self) -> None:
        self._recipe.clear()
        self._trial_step = None
        self.btn_accept.setEnabled(False)
        self.btn_reject.setEnabled(False)
        self._refresh_recipe_list()
        if self.main_window is not None and self.main_window.exp_manager.active is not None:
            exp = self.main_window.exp_manager.active
            exp._processed_channels = None
            if exp._raw_channels:
                self.viewer_proc.set_channels(exp._raw_channels,
                                              channel_display=exp.channel_display)

    # ── Save / Load recipe ──
    def _on_save_recipe(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Recipe", "",
            f"ND2Studios Recipe (*{RECIPE_EXTENSION})",
        )
        if not path:
            return
        if not path.endswith(RECIPE_EXTENSION):
            path += RECIPE_EXTENSION
        try:
            save_recipe_json(path, self._recipe, self._normalized,
                             name=os.path.basename(path),
                             notes="Saved from ND2Studios")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        if self.main_window is not None:
            self.main_window.set_status_text(f"Saved recipe: {os.path.basename(path)}")

    def _on_load_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Recipe", "",
            f"ND2Studios Recipe (*{RECIPE_EXTENSION});;All files (*)",
        )
        if not path:
            return
        try:
            data = load_recipe_json(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self._recipe = [(s["name"], dict(s.get("params", {}))) for s in data.get("pipeline", [])]
        self._normalized = bool(data.get("normalized", False))
        self.cb_normalized.blockSignals(True)
        self.cb_normalized.setChecked(self._normalized)
        self.cb_normalized.blockSignals(False)
        self._refresh_recipe_list()
        # Re-apply against current data.
        if self.main_window is not None and self.main_window.exp_manager.active is not None:
            exp = self.main_window.exp_manager.active
            if exp._raw_channels and self._recipe:
                self._run_recipe(exp, list(self._recipe),
                                 label="Applying loaded recipe…",
                                 on_done=self._on_revert_done)

    # ── Helpers ──
    def _refresh_recipe_list(self) -> None:
        self.list_recipe.clear()
        for i, (name, params) in enumerate(self._recipe):
            line = f"{i + 1}. {name}"
            if params:
                line += f"   ({', '.join(f'{k}={v}' for k, v in params.items())})"
            self.list_recipe.addItem(QListWidgetItem(line))

    def _run_recipe(self, exp: ND2StudiosRecord,
                    recipe: List[Tuple[str, Dict[str, Any]]],
                    label: str,
                    on_done) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(500)
        self._worker = RecipeWorker(
            channels=exp._raw_channels or {},
            recipe=recipe,
            normalized=self._normalized,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(on_done)
        self._worker.error.connect(self._on_error)
        if self.main_window is not None:
            self.main_window.set_status_text(label)
        self._worker.start()

    def _on_progress(self, p: int) -> None:
        if self.main_window is not None:
            self.main_window.set_progress(p)

    def _on_status(self, msg: str) -> None:
        if self.main_window is not None:
            self.main_window.set_status_text(msg)

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Recipe failed", msg)
        if self.main_window is not None:
            self.main_window.set_progress(0)

