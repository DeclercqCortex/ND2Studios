"""Properties inspector (NodeLab v2) — the click-to-edit mirror of a selected node.

Binds to a :class:`~nodelab_v2.node_item.NodeItem` and builds an editable form from its
:class:`~nodegraph.registry.NodeSpec`: the 2D/3D switch, the resolved footprint, and one
row per active parameter (spin boxes, mode dropdowns, unit labels). Metadata-derived
params (``derive``) show an **auto / pinned** toggle — the sticky ``__locked__`` override:
auto shows the metadata-derived value; editing or pinning fixes it; unpinning reverts.
Edits write back to the node's ``params``/``locked`` and relayout the card live.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QScrollArea, QSpinBox,
    QToolButton, QVBoxLayout, QWidget,
)

from nodegraph.sockets import SocketType
from nodelab_v2 import theme as T
from nodelab_v2.node_item import NodeItem

_UNIT = {"um": "µm", "um_axial": "µm↕", "um2": "µm²", "um3": "µm³",
         "nm": "nm", "s": "s", "px": "px"}


def _clean_path(s: str) -> str:
    """Normalize a hand-entered path: trim whitespace and surrounding quotes (a
    Windows "Copy as path" paste wraps the path in double quotes)."""
    return (s or "").strip().strip('"').strip("'").strip()


class _NoWheelCombo(QComboBox):
    """A combo that ignores the wheel unless it has focus.

    Not cosmetic — a fix for a live data-loss bug. The inspector is a fixed-height
    QScrollArea, so scrolling to a lower row drags the pointer across every combo above
    it, and PySide6 delivers each notch to the combo under the cursor: measured, ONE
    wheel event over an editable combo fires ``textActivated`` and ``accept()``s the
    event. The user silently rewrites (and pins) a param they never touched, and the
    panel does not scroll because the event was eaten. Ignoring it defers to the scroll
    area. A layer name is not an ordinal the wheel should walk, so even a focused combo
    gains nothing from wheel-stepping — but StrongFocus + the focus test keeps the
    conventional behaviour available after a deliberate click."""

    def wheelEvent(self, e):                      # noqa: N802 - Qt naming
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


def _derived_value(node: NodeItem, s) -> float:
    """The LIVE metadata-derived value from the node's propagated envelope (G8 —
    replaces the old hard-coded preview table)."""
    v = node.resolved(s)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(s.default) if s.default is not None else 0.0


def _h(col) -> str:
    return col.name()


def _inspector_qss() -> str:
    return f"""
QWidget#inspRoot {{ background:{_h(T.PANEL)}; }}
QScrollArea {{ border:0; background:{_h(T.PANEL)}; }}
QLabel {{ color:{_h(T.INK)}; }}
QLabel[role="eyebrow"] {{ color:{_h(T.INK_2)}; font-weight:600; letter-spacing:1.4px; }}
QLabel[role="muted"] {{ color:{_h(T.MUTED)}; }}
QLabel[role="op"] {{ color:{_h(T.MUTED)}; font-family:{T.MONO}; }}
QFrame[role="sep"] {{ background:{_h(T.BORDER)}; max-height:1px; min-height:1px; border:0; }}
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
  background:{_h(T.BODY)}; color:{_h(T.INK)}; border:1px solid {_h(T.BORDER)};
  border-radius:6px; padding:4px 7px; font-family:{T.MONO}; min-height:18px;
}}
QDoubleSpinBox:disabled, QSpinBox:disabled {{ color:{_h(T.MUTED)}; }}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
  border-color:{_h(T.ACCENT)}; }}
QComboBox::drop-down {{ border:0; width:16px; }}
QToolButton {{
  background:{_h(T.BODY)}; color:{_h(T.MUTED)}; border:1px solid {_h(T.BORDER)};
  border-radius:6px; padding:4px 7px; font-size:10px; font-weight:600;
}}
QToolButton:checked {{ color:{_h(T.MUTED)}; }}
QToolButton[state="auto"] {{ color:{_h(T.ACCENT)}; border-color:{_h(T.ACCENT_DIM)}; }}
""" + T.controls_qss()


class SwitchWidget(QWidget):
    """The 2D/3D on-off switch as a QWidget (amber 2D / cyan 3D)."""

    toggled = Signal(str)

    def __init__(self, dim: str = "2D") -> None:
        super().__init__()
        self.dim = dim
        self.setFixedSize(84, 24)
        self.setCursor(Qt.PointingHandCursor)

    def set_dim(self, dim: str) -> None:
        if dim != self.dim:
            self.dim = dim
            self.update()

    def mousePressEvent(self, e) -> None:
        self.dim = "3D" if self.dim == "2D" else "2D"
        self.update()
        self.toggled.emit(self.dim)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        on = self.dim == "3D"
        lblw, tw, th, knob = 18, 40, 22, 18
        f = QFont(T.MONO, 8); f.setBold(True); p.setFont(f)
        p.setPen(T.INK if not on else T.MUTED)
        p.drawText(0, 0, lblw, 24, Qt.AlignCenter, "2D")
        p.setPen(T.INK if on else T.MUTED)
        p.drawText(self.width() - lblw, 0, lblw, 24, Qt.AlignCenter, "3D")
        tx = lblw + 4
        col = T.ACCENT if on else T.DIM2D
        p.setPen(QPen(col, 1)); p.setBrush(col)
        p.drawRoundedRect(tx, 1, tw, th, th / 2, th / 2)
        kx = tx + (tw - knob - 1) if on else tx + 1
        p.setPen(Qt.NoPen); p.setBrush(T.ACCENT_INK if on else T.DIM2D_INK)
        p.drawEllipse(kx, 2, knob, knob)


class InspectorPanel(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        # wide enough for the richest param row (label + spinbox + unit + ƒ-auto button)
        # PLUS the vertical scrollbar, so the right edge (the ƒ-auto buttons) never clips.
        self.setFixedWidth(376)
        self.setStyleSheet(_inspector_qss())
        self._node: Optional[NodeItem] = None
        self._auto_boxes = []              # (node, socket, box) — live ƒmd refresh (G8)
        self._host = QWidget(); self._host.setObjectName("inspRoot")
        self.setWidget(self._host)
        self._v = QVBoxLayout(self._host)
        self._v.setContentsMargins(0, 0, 0, 0)
        self._v.setSpacing(0)
        self._rebuild()

    # ── binding ─────────────────────────────────────────────────────────────
    def set_node(self, node: Optional[NodeItem]) -> None:
        if self._node is not None:
            try:
                self._node.changed.disconnect(self._on_changed)
            except (RuntimeError, TypeError):
                pass
        self._node = node
        if node is not None:
            node.changed.connect(self._on_changed)
        self._rebuild()

    def _on_changed(self, *_a) -> None:
        self._rebuild()

    # ── build ───────────────────────────────────────────────────────────────
    def restyle(self) -> None:
        """Re-apply the stylesheet from the current theme tokens (G9) + rebuild so
        the per-widget inline colors (chips/dots) re-read too."""
        self.setStyleSheet(_inspector_qss())
        self._rebuild()

    def refresh_derived(self) -> None:
        """Re-seed the disabled (auto) value boxes from the live envelopes (G8) —
        called on every document change; never touches an editable/focused box."""
        for node, s, box in self._auto_boxes:
            try:
                box.blockSignals(True)
                box.setValue(_derived_value(node, s))
            except RuntimeError:              # widget already deleted mid-rebuild
                continue
            finally:
                try:
                    box.blockSignals(False)
                except RuntimeError:
                    pass

    def _clear(self) -> None:
        self._auto_boxes = []
        while self._v.count():
            item = self._v.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)     # detach now so it can't paint before deleteLater runs
                w.deleteLater()

    def _eyebrow(self, text: str, color=None) -> QLabel:
        lab = QLabel(text.upper()); lab.setProperty("role", "eyebrow")
        if color is not None:
            lab.setStyleSheet(f"color:{_h(color)};")
        f = lab.font(); f.setPointSize(8); lab.setFont(f)
        return lab

    def _sep(self) -> QFrame:
        fr = QFrame(); fr.setProperty("role", "sep"); return fr

    def _section(self, title: str, extra: str = "") -> QWidget:
        sec = QWidget()
        lay = QVBoxLayout(sec); lay.setContentsMargins(16, 13, 16, 13); lay.setSpacing(8)
        head = self._eyebrow(title)
        if extra:
            row = QHBoxLayout(); row.addWidget(head); row.addStretch(1)
            m = QLabel(extra); m.setProperty("role", "muted")
            fm = m.font(); fm.setPointSize(9); m.setFont(fm)
            row.addWidget(m)
            lay.addLayout(row)
        else:
            lay.addWidget(head)
        sec._lay = lay  # type: ignore[attr-defined]
        return sec

    def _rebuild(self) -> None:
        self._clear()
        node = self._node
        if node is None or node.spec is None:
            if node is not None and getattr(node, "_is_group", False):
                msg = (f"Group “{node._group_name}”.\n\nA reusable subgraph collapsed into "
                       f"one node. Select it and use Graph → Ungroup (Ctrl+Shift+G) to edit "
                       f"its contents, then re-group. Pulling it runs the whole subgraph.")
            elif node is not None and node.spec is None:
                msg = (f"Unrecognized node type “{node.op_key}”.\n\nThis op is not in "
                       f"the registry — the file may come from a newer build or a "
                       f"plugin that isn't loaded.")
            else:
                msg = "Select a node to edit its parameters."
            ph = QLabel(msg)
            ph.setProperty("role", "muted"); ph.setAlignment(Qt.AlignCenter)
            ph.setWordWrap(True); ph.setContentsMargins(30, 40, 30, 40)
            self._v.addWidget(ph); self._v.addStretch(1)
            return
        spec = node.spec
        cat = T.category_color(spec.category)

        # header
        hd = QWidget()
        hl = QVBoxLayout(hd); hl.setContentsMargins(16, 14, 16, 12); hl.setSpacing(4)
        hl.addWidget(self._eyebrow(spec.category, cat))
        row = QHBoxLayout()
        title = QLabel(spec.label); tf = title.font(); tf.setPointSize(13); tf.setBold(True)
        title.setFont(tf); row.addWidget(title); row.addStretch(1)
        if spec.has_dim_lever():
            sw = SwitchWidget(node.dim)
            sw.toggled.connect(node.set_dim)
            row.addWidget(sw)
        hl.addLayout(row)
        op = QLabel(spec.op_key); op.setProperty("role", "op")
        of = op.font(); of.setPointSize(9); op.setFont(of); hl.addWidget(op)
        self._v.addWidget(hd)
        self._v.addWidget(self._sep())

        # footprint
        fp = self._section("Footprint")
        chip = QLabel(node.granularity().replace("_", " ").upper())
        gcol = T.gran_color(node.granularity())
        chip.setStyleSheet(
            f"color:{_h(gcol)}; border:1px solid {_h(T.alpha(gcol,120))};"
            f"background:{_h(T.alpha(gcol,36))}; border-radius:4px; padding:3px 7px;"
            f"font-family:{T.MONO}; font-weight:700;")
        cf = chip.font(); cf.setPointSize(8); chip.setFont(cf)
        frow = QHBoxLayout(); frow.addWidget(chip); frow.addStretch(1)
        fp._lay.addLayout(frow)  # type: ignore[attr-defined]
        note = QLabel("The 2D / 3D switch resolves this footprint and the active sockets; "
                      "it folds into the memo key, so 2D and 3D cache separately."
                      if spec.has_dim_lever() else "Dimension-agnostic — no 2D/3D switch.")
        note.setProperty("role", "muted"); note.setWordWrap(True)
        nf = note.font(); nf.setPointSize(9); note.setFont(nf)
        fp._lay.addWidget(note)  # type: ignore[attr-defined]
        self._v.addWidget(fp)
        self._v.addWidget(self._sep())

        # parameters
        params = [s for s in node._active_inputs() if s.type is not SocketType.DATASET]
        if params:
            sec = self._section("Parameters", str(len(params)))
            for s in params:
                sec._lay.addWidget(self._param_row(node, s))  # type: ignore[attr-defined]
            self._v.addWidget(sec)
            self._v.addWidget(self._sep())

        # in-body modes (non-dim)
        modes = [m for m in spec.modes if not m.is_dim_lever]
        if modes:
            sec = self._section("Mode")
            for m in modes:
                sec._lay.addWidget(self._mode_row(node, m))  # type: ignore[attr-defined]
            self._v.addWidget(sec)
            self._v.addWidget(self._sep())

        # connections
        sec = self._section("Connections")
        for s in spec.inputs:
            if s.type is SocketType.DATASET:
                sec._lay.addWidget(self._conn_label(f"in · {s.name}", T.SOCKET[s.type]))  # type: ignore[attr-defined]
        for s in spec.outputs:
            sec._lay.addWidget(self._conn_label(f"out · {s.name}", T.SOCKET[s.type]))  # type: ignore[attr-defined]
        self._v.addWidget(sec)
        self._v.addStretch(1)

    # ── rows ────────────────────────────────────────────────────────────────
    def _param_row(self, node: NodeItem, s) -> QWidget:
        row = QWidget(); lay = QHBoxLayout(row); lay.setContentsMargins(0, 3, 0, 3); lay.setSpacing(8)
        lab = QLabel(s.name); lab.setMinimumWidth(78); lay.addWidget(lab)
        lay.addStretch(1)
        derived = bool(s.derive)
        pinned = s.name in node.params or s.name in node.locked
        auto = derived and not pinned

        if s.type is SocketType.INT:
            box = QSpinBox(); box.setRange(0, 100000)
            box.setValue(int(node.params.get(s.name, s.default if s.default is not None else 0)))
        elif s.type is SocketType.STRING and (s.layer_in or s.layer_in_mode):
            lay.addWidget(self._layer_box(node, s))
            return row
        elif s.type is SocketType.STRING:
            from PySide6.QtWidgets import QLineEdit
            is_path = s.name == "path"
            box = QLineEdit(str(node.params.get(s.name, s.default or "")))
            if is_path:
                box.setPlaceholderText("empty = synthetic demo · or Browse… for an .nd2")
            # commit on editing-finished (not per keystroke — a path edit mid-typing
            # must not spam the document/propagation)
            box.editingFinished.connect(
                lambda b=box, nm=s.name, p=is_path: self._commit_text(node, nm, b, p))
            lay.addWidget(box)
            if is_path:
                browse = QToolButton(); browse.setText("Browse…")
                browse.setToolTip("Choose a file — fills the path (no manual typing)")
                browse.clicked.connect(
                    lambda _c, b=box, nm=s.name: self._browse_path(node, nm, b))
                lay.addWidget(browse)
            return row
        else:
            box = QDoubleSpinBox(); box.setRange(0.0, 1e6); box.setDecimals(3); box.setSingleStep(0.05)
            base = node.params.get(s.name, _derived_value(node, s) if derived
                                   else (s.default if s.default is not None else 0.0))
            try:
                box.setValue(float(base))
            except (TypeError, ValueError):
                box.setValue(0.0)
        box.setEnabled(not auto)
        box.valueChanged.connect(lambda v, nm=s.name: self._set_param(node, nm, v))
        if auto and isinstance(box, QDoubleSpinBox):
            self._auto_boxes.append((node, s, box))
        lay.addWidget(box)

        unit = _UNIT.get(s.unit, s.unit)
        if unit:
            u = QLabel(unit); u.setProperty("role", "muted"); u.setFixedWidth(24)
            uf = u.font(); uf.setPointSize(9); u.setFont(uf); lay.addWidget(u)

        if derived:
            btn = QToolButton(); btn.setCheckable(True); btn.setChecked(not auto)
            btn.setText("pinned" if pinned else "ƒ auto")
            btn.setProperty("state", "pinned" if pinned else "auto")
            btn.setToolTip("Metadata-derived (auto). Pin to fix the value; unpin to revert."
                           if auto else "Pinned. Click to revert to the metadata-derived value.")
            btn.clicked.connect(lambda _c, nm=s.name: self._toggle_pin(node, nm))
            lay.addWidget(btn)
        return row

    def _layer_box(self, node: NodeItem, s):
        """An EDITABLE combo for a ``layer_in`` socket: the layers actually present on
        the incoming edge, plus free text.

        Editable rather than a closed dropdown because the suggestion list is honest but
        incomplete — a couple of producers name layers the edit-time pass cannot predict
        (``transform.rasterize_field`` invents one Voxel layer per field component from a
        prefix), and a node can be wired up after the consumer is configured. A closed
        list would make those layers unreachable; free text with suggestions strictly
        improves on the plain line edit it replaces and can never block a valid name.

        Signal choice matters. ``currentTextChanged``/``currentIndexChanged`` also fire
        while the list is being REPOPULATED and while the user types, so they would
        commit half-typed names and fight the rebuild. Only ``activated`` (a real user
        pick, never programmatic) and the line edit's ``editingFinished`` commit."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QCompleter

        box = _NoWheelCombo()
        box.setEditable(True)
        box.setInsertPolicy(QComboBox.NoInsert)      # typing must not grow the list
        box.setFocusPolicy(Qt.StrongFocus)
        box.setDuplicatesEnabled(False)

        try:
            choices = list(node.doc.layer_choices(node.node_id, s))
        except Exception:                            # never let a picker break the panel
            choices = []
        current = str(node.params.get(s.name, s.default or ""))
        box.blockSignals(True)
        box.addItems(choices)
        box.setEditText(current)
        box.blockSignals(False)

        cp = QCompleter(choices, box)
        cp.setCaseSensitivity(Qt.CaseInsensitive)
        # PopupCompletion, not InlineCompletion: inline would type-ahead-fill the edit
        # with a suggestion, so tabbing away would COMMIT a name the user never chose.
        cp.setCompletionMode(QCompleter.PopupCompletion)
        cp.popup().setStyleSheet(T.controls_qss())   # else the popup ignores the theme
        box.setCompleter(cp)

        if choices:
            box.setToolTip("Layers on the incoming edge: " + ", ".join(choices)
                           + "\n(free text is allowed — some layer names cannot be "
                             "predicted before the graph runs)")
        else:
            box.setToolTip("No layers detected upstream yet — connect a producer, or "
                           "type the name.")
        box.activated.connect(
            lambda _i, nm=s.name, b=box: self._set_param(node, nm, b.currentText()))
        box.lineEdit().editingFinished.connect(
            lambda nm=s.name, b=box: self._set_param(node, nm, b.currentText()))
        return box

    def _mode_row(self, node: NodeItem, m) -> QWidget:
        row = QWidget(); lay = QHBoxLayout(row); lay.setContentsMargins(0, 3, 0, 3); lay.setSpacing(8)
        lab = QLabel(m.name); lay.addWidget(lab); lay.addStretch(1)
        # _NoWheelCombo: a wheel over a Mode dropdown used to silently change the
        # method (and trigger the deferred rebuild) while the user was only
        # scrolling the panel — the same hazard documented on _NoWheelCombo.
        combo = _NoWheelCombo(); combo.addItems(list(m.choices))
        cur = node.rec.modes.get(m.name, m.resolved_default())
        if cur in m.choices:
            combo.setCurrentText(cur)
        combo.currentTextChanged.connect(lambda t, nm=m.name: self._set_mode(node, nm, t))
        lay.addWidget(combo)
        return row

    def _conn_label(self, text: str, col) -> QWidget:
        row = QWidget(); lay = QHBoxLayout(row); lay.setContentsMargins(0, 2, 0, 2); lay.setSpacing(8)
        dot = QLabel(); dot.setFixedSize(11, 11)
        dot.setStyleSheet(f"background:{_h(col)}; border-radius:5px;")
        lay.addWidget(dot)
        lab = QLabel(text); lab.setProperty("role", "muted")
        lf = lab.font(); lf.setPointSize(10); lab.setFont(lf)
        lay.addWidget(lab); lay.addStretch(1)
        return row

    def _commit_text(self, node: NodeItem, name: str, box, is_path: bool) -> None:
        """Commit a QLineEdit string param. For a ``path`` field, normalize first —
        strip whitespace and surrounding quotes (a Windows "Copy as path" paste wraps
        the path in double quotes) — and reflect the cleaned value back in the box."""
        text = _clean_path(box.text()) if is_path else box.text()
        if is_path and text != box.text():
            box.blockSignals(True); box.setText(text); box.blockSignals(False)
        self._set_param(node, name, text)

    def _browse_path(self, node: NodeItem, name: str, box) -> None:
        from PySide6.QtWidgets import QFileDialog
        start = _clean_path(box.text())
        path, _f = QFileDialog.getOpenFileName(
            self, "Select an ND2 file", start, "ND2 (*.nd2);;All files (*)")
        if path:
            box.setText(path)
            self._set_param(node, name, path)

    # ── edits (all through the document — G8 re-propagates envelopes) ────────
    def _set_param(self, node: NodeItem, name: str, value) -> None:
        rec = node.rec
        rec.params[name] = value
        rec.set_locked(rec.locked | {name})     # editing pins (sticky __locked__)
        node.doc.touch()
        # don't full-rebuild (keeps focus in the box); the card refreshes via sync

    def _set_mode(self, node: NodeItem, name: str, value: str) -> None:
        node.rec.modes[name] = value            # modes are NOT params (serialize split)
        node.doc.touch()
        # A mode change can reconfigure the ACTIVE socket set (`available_in`), so the
        # form has to be rebuilt: `doc.touch()` only re-seeds the auto boxes
        # (refresh_derived) and re-lays-out the card (scene.sync → item.refresh), which
        # would leave the inspector showing the *previous* method's params. Deferred to
        # the next event-loop turn so we never tear down the QComboBox that is mid-emit.
        QTimer.singleShot(0, self._rebuild)

    def _toggle_pin(self, node: NodeItem, name: str) -> None:
        rec = node.rec
        if name in rec.params or name in rec.locked:
            rec.params.pop(name, None)
            rec.set_locked(rec.locked - {name})                 # revert to auto
        else:
            spec_sock = node.spec.input(name) if node.spec else None
            rec.params[name] = (_derived_value(node, spec_sock)
                                if spec_sock is not None else 0.0)
            rec.set_locked(rec.locked | {name})                 # pin the derived value
        node.doc.touch()
        self._rebuild()


__all__ = ["InspectorPanel", "SwitchWidget"]
