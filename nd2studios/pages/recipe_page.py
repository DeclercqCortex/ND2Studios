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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
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
        # Refresh the raw preview from the active experiment.
        if exp._raw_channels:
            self.viewer_raw.set_channels(exp._raw_channels,
                                         channel_display=exp.channel_display)
            # If a processed cache already exists, mirror it on the right.
            if exp._processed_channels:
                self.viewer_proc.set_channels(exp._processed_channels,
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

