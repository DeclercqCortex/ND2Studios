"""The **Overlays** popup — one tab per attribute domain, live-applied.

The Viewer's three overlay checkboxes became a single *Overlays* button that opens this
panel. Every control here is built from :data:`nodelab_v2.overlays.FIELDS`, so a new
setting is one :class:`~nodelab_v2.overlays.FieldSpec` away from having a widget, a
tooltip, a range, a dependency and a place in the spread rules — there is no per-tab
hand-written form to keep in sync.

Four things the layout is deliberate about:

* **Live.** The dialog is non-modal and every edit emits :data:`changed` immediately, so
  you tune a look against the real image instead of against a description of it.
* **A preview that is the real renderer.** The strip at the top of each tab calls
  :func:`nodelab_v2.overlays.render_preview`, which drives the same
  :class:`~nodelab_v2.overlays.OverlayRenderer` the Viewer uses. If the preview and the
  image ever disagreed, one of them would be lying.
* **Reserved tabs say so.** Voxels and Mesh keep their settings (they serialize and they
  spread), but their controls are disabled behind a banner rather than presented as live
  knobs that nothing reads — the same rule the node catalog holds itself to.
* **Spread reports.** "Spread to other tabs" names every value it moved, and says so when
  a role had nowhere applicable to go.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from nodelab_v2 import overlays as OV
from nodelab_v2 import theme as T


class _PreviewStrip(QWidget):
    """A synthetic sample of one tab's overlay, painted by the real renderer."""

    def __init__(self, settings: OV.OverlaySettings, tab: str,
                 renderer: OV.OverlayRenderer) -> None:
        super().__init__()
        self._settings = settings
        self._tab = tab
        self._renderer = renderer
        self.setMinimumHeight(96)
        self.setToolTip("A live sample drawn by the same renderer as the image")

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        rect = QRectF(0, 0, self.width(), self.height())
        OV.render_preview(p, rect, self._settings, self._tab, self._renderer)
        p.setPen(T.BORDER)
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
        p.end()


class _ColorButton(QPushButton):
    """A swatch that opens a colour picker and reports the chosen ``#rrggbb``."""

    picked = Signal(str)

    def __init__(self, value: str) -> None:
        super().__init__()
        self._value = value
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(22)
        self.clicked.connect(self._choose)
        self.set_value(value)

    def set_value(self, value: str) -> None:
        self._value = str(value)
        col = QColor(self._value)
        ink = "#0b0e12" if col.lightness() > 140 else "#f0f3f7"
        self.setText(self._value)
        # the :disabled rule matters — a swatch keeps its own stylesheet, so without it a
        # colour the chosen mode ignores would still look like a live control
        self.setStyleSheet(
            f"QPushButton {{ background:{col.name()}; color:{ink}; font-family:{T.MONO};"
            f" font-size:11px; border:1px solid {T.BORDER.name()}; border-radius:5px;"
            f" padding:2px 8px; }}"
            f"QPushButton:disabled {{ background:{T.BG.name()};"
            f" color:{T.MUTED.name()}; border:1px dashed {T.BORDER.name()}; }}")

    def value(self) -> str:
        return self._value

    def _choose(self) -> None:
        col = QColorDialog.getColor(QColor(self._value), self, "Overlay colour")
        if col.isValid():
            self.set_value(col.name())
            self.picked.emit(col.name())


class OverlayDialog(QDialog):
    """The tabbed overlay editor. Mutates the caller's :class:`OverlaySettings` in place
    and emits :data:`changed` after every edit."""

    changed = Signal()

    def __init__(self, settings: OV.OverlaySettings,
                 renderer: Optional[OV.OverlayRenderer] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Overlays")
        self.setWindowFlag(Qt.Window)
        self.setMinimumWidth(400)
        self._settings = settings
        self._renderer = renderer or OV.OverlayRenderer()
        self._widgets: Dict[Tuple[str, str], QWidget] = {}
        self._rows: Dict[Tuple[str, str], Tuple[QWidget, QWidget]] = {}
        self._previews: Dict[str, _PreviewStrip] = {}
        self._enables: Dict[str, QCheckBox] = {}
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.tabs = QTabWidget()
        for info in OV.TAB_INFO:
            self.tabs.addTab(self._build_tab(info), info.title)
        root.addWidget(self.tabs, 1)

        # ── spread ──────────────────────────────────────────────────────────────
        spread = QPushButton("Spread this tab's look to the others")
        spread.setCursor(Qt.PointingHandCursor)
        spread.setToolTip(
            "Copy the settings that make sense everywhere — "
            + ", ".join(OV.SPREAD_ROLE_LABEL[r] for r in OV.SPREAD_ROLES)
            + " — from the current tab onto the other tabs. Domain-specific settings "
              "(arm spread, trail mode, fill) are left alone.")
        spread.clicked.connect(self._spread)
        root.addWidget(spread)

        # ── files ───────────────────────────────────────────────────────────────
        files = QHBoxLayout()
        files.setSpacing(6)
        for text, tip, slot in (
            ("Load…", "Load overlay settings from a file", self._load_file),
            ("Save…", "Save these overlay settings to a file you choose",
             self._save_file),
            ("Make default", f"Make this the permanent overlay on this machine "
                             f"({OV.USER_FILE})", self._save_user),
            ("Save to project", f"Write the project default that ships with the repo "
                                f"({OV.PROJECT_FILE.name}) — commit it to share the "
                                f"look", self._save_project),
        ):
            b = QPushButton(text)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            files.addWidget(b)
        root.addLayout(files)

        tail = QHBoxLayout()
        tail.setSpacing(6)
        reset_tab = QPushButton("Reset tab")
        reset_tab.setToolTip("Restore the built-in defaults for the current tab")
        reset_tab.clicked.connect(self._reset_tab)
        reset_all = QPushButton("Reset all")
        reset_all.setToolTip("Restore the built-in defaults for every tab")
        reset_all.clicked.connect(self._reset_all)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        tail.addWidget(reset_tab)
        tail.addWidget(reset_all)
        tail.addStretch(1)
        tail.addWidget(close)
        root.addLayout(tail)

        self._status = QLabel("")
        self._status.setProperty("role", "muted")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self.tabs.currentChanged.connect(lambda _i: self._refresh_preview())
        self.restyle()
        self.reload()

    # ── construction ───────────────────────────────────────────────────────────
    def _build_tab(self, info: OV.TabInfo) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(2, 6, 2, 2)
        lay.setSpacing(7)

        blurb = QLabel(info.blurb)
        blurb.setWordWrap(True)
        blurb.setProperty("role", "muted")
        lay.addWidget(blurb)

        preview = _PreviewStrip(self._settings, info.key, self._renderer)
        self._previews[info.key] = preview
        lay.addWidget(preview)

        enable = QCheckBox(f"Show the {info.title} overlay")
        enable.setProperty("role", "enable")
        enable.setCursor(Qt.PointingHandCursor)
        enable.toggled.connect(lambda on, tab=info.key: self._set_enabled(tab, on))
        self._enables[info.key] = enable
        lay.addWidget(enable)

        if not info.implemented:
            note = QLabel("⚠ Not implemented yet — nothing is drawn for this domain. "
                          "The settings below are stored, spread and saved, so the look "
                          "is ready the moment the renderer lands.")
            note.setWordWrap(True)
            note.setProperty("role", "warn")
            lay.addWidget(note)
            enable.setEnabled(False)

        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(5)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        for spec in OV.FIELDS[info.key]:
            label, editor = self._build_row(info.key, spec)
            form.addRow(label, editor)
            self._rows[(info.key, spec.key)] = (label, editor)
            if not info.implemented:
                editor.setEnabled(False)
                label.setEnabled(False)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(form_host)
        lay.addWidget(scroll, 1)
        return page

    def _build_row(self, tab: str, spec: OV.FieldSpec) -> Tuple[QLabel, QWidget]:
        label = QLabel(spec.label)
        if spec.tip:
            label.setToolTip(spec.tip)
        editor: QWidget
        if spec.kind == "bool":
            editor = QCheckBox()
            editor.toggled.connect(
                lambda v, t=tab, k=spec.key: self._set(t, k, bool(v)))
        elif spec.kind == "int":
            editor = QSpinBox()
            editor.setRange(int(spec.lo), int(spec.hi))
            editor.setSingleStep(max(1, int(spec.step)))
            editor.setSuffix(spec.unit)
            editor.valueChanged.connect(
                lambda v, t=tab, k=spec.key: self._set(t, k, int(v)))
        elif spec.kind == "float":
            editor = QDoubleSpinBox()
            editor.setRange(float(spec.lo), float(spec.hi))
            editor.setSingleStep(float(spec.step))
            editor.setDecimals(int(spec.decimals))
            editor.setSuffix(spec.unit)
            editor.valueChanged.connect(
                lambda v, t=tab, k=spec.key: self._set(t, k, float(v)))
        elif spec.kind == "choice":
            editor = QComboBox()
            for value, text in spec.choices:
                editor.addItem(text, value)
            editor.currentIndexChanged.connect(
                lambda _i, t=tab, k=spec.key, w=editor:
                self._set(t, k, w.currentData()))
        else:                                        # color
            editor = _ColorButton(getattr(self._settings.group(tab), spec.key))
            editor.picked.connect(
                lambda v, t=tab, k=spec.key: self._set(t, k, v))
        if spec.tip:
            editor.setToolTip(spec.tip)
        self._widgets[(tab, spec.key)] = editor
        return label, editor

    # ── state sync ─────────────────────────────────────────────────────────────
    def reload(self) -> None:
        """Push the settings into every control (after a load / reset / external edit)."""
        self._loading = True
        try:
            for tab in OV.TABS:
                grp = self._settings.group(tab)
                self._enables[tab].setChecked(bool(grp.enabled))
                for spec in OV.FIELDS[tab]:
                    w = self._widgets[(tab, spec.key)]
                    val = getattr(grp, spec.key)
                    if isinstance(w, QCheckBox):
                        w.setChecked(bool(val))
                    elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                        w.setValue(val)
                    elif isinstance(w, QComboBox):
                        idx = w.findData(val)
                        w.setCurrentIndex(idx if idx >= 0 else 0)
                    elif isinstance(w, _ColorButton):
                        w.set_value(val)
        finally:
            self._loading = False
        self._sync_dependencies()
        self._refresh_preview()

    def _sync_dependencies(self) -> None:
        """Grey out the rows whose ``enable_if`` is not satisfied — a control that the
        chosen mode ignores must not look live."""
        for tab in OV.TABS:
            implemented = OV.TAB_BY_KEY[tab].implemented
            grp = self._settings.group(tab)
            for spec in OV.FIELDS[tab]:
                on = implemented and OV.field_enabled(grp, spec)
                label, editor = self._rows[(tab, spec.key)]
                label.setEnabled(on)
                editor.setEnabled(on)

    def _set(self, tab: str, key: str, value: Any) -> None:
        if self._loading:
            return
        grp = self._settings.group(tab)
        if getattr(grp, key) == value:
            return
        setattr(grp, key, value)
        self._sync_dependencies()
        self._apply(f"{OV.TAB_BY_KEY[tab].title}: {key} = {value}")

    def _set_enabled(self, tab: str, on: bool) -> None:
        if self._loading:
            return
        grp = self._settings.group(tab)
        if bool(grp.enabled) == bool(on):
            return
        grp.enabled = bool(on)
        self._apply(f"{OV.TAB_BY_KEY[tab].title} overlay {'on' if on else 'off'}")

    def _apply(self, note: str = "") -> None:
        self._renderer.invalidate()
        self._refresh_preview()
        if note:
            self._status.setText(note)
        self.changed.emit()

    def _refresh_preview(self) -> None:
        for strip in self._previews.values():
            strip.update()

    def current_tab(self) -> str:
        idx = max(0, self.tabs.currentIndex())
        return OV.TABS[min(idx, len(OV.TABS) - 1)]

    # ── actions ────────────────────────────────────────────────────────────────
    def _spread(self) -> None:
        tab = self.current_tab()
        notes = OV.spread_settings(self._settings, tab)
        self.reload()
        self._apply()
        if notes:
            self._status.setText(f"Spread from {OV.TAB_BY_KEY[tab].title} — "
                                 + "; ".join(notes))
        else:
            self._status.setText(f"Nothing to spread from "
                                 f"{OV.TAB_BY_KEY[tab].title} — the other tabs already "
                                 f"match on every shared setting.")

    def _load_file(self) -> None:
        path, _f = QFileDialog.getOpenFileName(self, "Load overlay settings", "",
                                               OV.FILE_FILTER)
        if not path:
            return
        try:
            changed = self._settings.update_from_dict(OV.read_json(Path(path)))
        except Exception as exc:                     # noqa: BLE001 — report, never crash
            QMessageBox.warning(self, "Overlays", f"Could not load:\n{exc}")
            return
        self.reload()
        self._apply()
        self._status.setText(f"Loaded {Path(path).name} — {len(changed)} setting(s) "
                             f"changed")

    def _save_file(self) -> None:
        path, _f = QFileDialog.getSaveFileName(
            self, "Save overlay settings", f"overlays{OV.FILE_SUFFIX}", OV.FILE_FILTER)
        if path:
            self._write(Path(path), f"Saved to {Path(path).name}")

    def _save_user(self) -> None:
        self._write(OV.USER_FILE,
                    f"These overlays are now this machine's default ({OV.USER_FILE})")

    def _save_project(self) -> None:
        self._write(OV.PROJECT_FILE,
                    f"Wrote the project default {OV.PROJECT_FILE.name} — commit it to "
                    f"share this look with the repo")

    def _write(self, path: Path, note: str) -> None:
        try:
            OV.write_json(path, self._settings)
        except Exception as exc:                     # noqa: BLE001
            QMessageBox.warning(self, "Overlays", f"Could not save:\n{exc}")
            return
        self._status.setText(note)

    def _reset_tab(self) -> None:
        tab = self.current_tab()
        self._settings.update_from_dict({"overlays": {tab: OV.defaults_for(tab)}})
        self.reload()
        self._apply(f"{OV.TAB_BY_KEY[tab].title} reset to the built-in defaults")

    def _reset_all(self) -> None:
        self._settings.update_from_dict(OV.OverlaySettings().to_dict())
        self.reload()
        self._apply("All overlays reset to the built-in defaults")

    # ── styling ────────────────────────────────────────────────────────────────
    def restyle(self) -> None:
        self.setStyleSheet(f"""
            QDialog, QWidget {{ background:{T.PANEL.name()}; color:{T.INK.name()}; }}
            QLabel[role="muted"] {{ color:{T.MUTED.name()}; font-size:11px; }}
            QLabel[role="warn"] {{ color:{T.DIM2D.name()}; font-size:11px;
                border:1px solid {T.DIM2D.name()}; border-radius:6px; padding:5px 7px; }}
            QCheckBox[role="enable"] {{ color:{T.INK.name()}; font-weight:700;
                font-size:12px; }}
            QTabWidget::pane {{ border:1px solid {T.BORDER.name()};
                border-radius:8px; top:-1px; }}
            QTabBar::tab {{ background:{T.BODY.name()}; color:{T.INK_2.name()};
                border:1px solid {T.BORDER.name()}; border-bottom:0;
                border-top-left-radius:7px; border-top-right-radius:7px;
                padding:5px 12px; margin-right:2px; }}
            QTabBar::tab:selected {{ background:{T.PANEL.name()}; color:{T.INK.name()};
                border-color:{T.ACCENT.name()}; }}
            QTabBar::tab:disabled {{ color:{T.MUTED.name()}; }}
            QScrollArea {{ background:transparent; }}
        """ + T.controls_qss())
        for strip in self._previews.values():
            strip.update()


__all__ = ["OverlayDialog"]
