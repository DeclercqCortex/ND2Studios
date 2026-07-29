"""The empty-canvas welcome card (2026-07-27).

NodeLab v2 opens on a **blank canvas** — no demo graph — with this card centred on it:
a short invitation to place the first node, the gestures that do it, and three buttons
for the usual first moves (load an image, browse the node palette, open the example
graph). It hides itself the moment the document has a node and comes back on File → New.

It is chrome only: it owns no document state, and it forwards a palette **drag-and-drop
that lands on it** to the canvas underneath (:data:`op_dropped`) — otherwise the card
would swallow the very drop it is asking for, sitting where you would aim it.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from nodelab_v2 import theme as T

#: the palette's drag payload (mirrors :class:`~nodelab_v2.scene.GraphView`)
OP_MIME = "application/x-nd2studios-op"


class WelcomeCard(QWidget):
    """Centred invitation shown while the canvas holds no nodes."""

    SIZE = (452, 248)
    GLYPH_H = 66                  # painted "drop a node here" square above the text

    load_image_requested = Signal()
    browse_nodes_requested = Signal()
    example_requested = Signal()
    op_dropped = Signal(str, QPointF)     # (op_key, scene position) — drop passthrough

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("welcome")
        self.setAcceptDrops(True)
        self.setAutoFillBackground(False)
        self.resize(*self.SIZE)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(26, self.GLYPH_H, 26, 20)
        lay.setSpacing(3)

        self._title = QLabel("Start your graph")
        self._title.setAlignment(Qt.AlignCenter)
        self._sub = QLabel("The canvas is empty — place a node to begin.")
        self._sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._title)
        lay.addWidget(self._sub)
        lay.addSpacing(9)

        self._hints = []
        for text in (
                "Double-click a node in the <b>Nodes</b> palette — or drag it onto the canvas",
                "<b>Ctrl+L</b> loads an ND2/TIFF and drops a source node with its channels",
                "Double-click any node to preview it · <b>Ctrl+Space</b> maximizes the canvas"):
            lab = QLabel(text)
            lab.setAlignment(Qt.AlignCenter)
            lab.setTextFormat(Qt.RichText)
            lay.addWidget(lab)
            self._hints.append(lab)
        lay.addStretch(1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._btn_load = QPushButton("Load image…")
        self._btn_load.setProperty("role", "primary")
        self._btn_load.setToolTip("File → Load ND2/TIFF file… (Ctrl+L)")
        self._btn_browse = QPushButton("Browse nodes")
        self._btn_browse.setToolTip("Jump to the Nodes palette search")
        self._btn_example = QPushButton("Example graph")
        self._btn_example.setToolTip("Fill the canvas with a small demo chain to explore")
        for b in (self._btn_load, self._btn_browse, self._btn_example):
            b.setCursor(Qt.PointingHandCursor)
            row.addWidget(b)
        lay.addLayout(row)

        self._btn_load.clicked.connect(self.load_image_requested.emit)
        self._btn_browse.clicked.connect(self.browse_nodes_requested.emit)
        self._btn_example.clicked.connect(self.example_requested.emit)

        parent.installEventFilter(self)
        self.hide()
        self.restyle()

    # ── placement ──────────────────────────────────────────────────────────────
    def eventFilter(self, obj, ev) -> bool:        # noqa: N802 — Qt override
        if obj is self.parentWidget() and ev.type() == QEvent.Resize:
            self.recenter()
        return False

    def recenter(self) -> None:
        """Sit in the middle of the canvas, shrinking if the canvas is smaller."""
        par = self.parentWidget()
        if par is None:
            return
        w = min(self.SIZE[0], max(240, par.width() - 24))
        h = min(self.SIZE[1], max(180, par.height() - 24))
        if (w, h) != (self.width(), self.height()):
            self.resize(w, h)
        self.move(max(0, (par.width() - w) // 2), max(0, (par.height() - h) // 2))

    def setVisible(self, on: bool) -> None:        # noqa: N802 — Qt override
        if on:
            self.recenter()
        super().setVisible(on)

    # ── palette drop passthrough ───────────────────────────────────────────────
    def dragEnterEvent(self, e) -> None:           # noqa: N802 — Qt override
        e.acceptProposedAction() if e.mimeData().hasFormat(OP_MIME) else e.ignore()

    def dragMoveEvent(self, e) -> None:            # noqa: N802 — Qt override
        e.acceptProposedAction() if e.mimeData().hasFormat(OP_MIME) else e.ignore()

    def dropEvent(self, e) -> None:                # noqa: N802 — Qt override
        """A node dropped ON the card lands on the canvas beneath it, at that point."""
        view = self.parentWidget()
        if not e.mimeData().hasFormat(OP_MIME) or view is None:
            e.ignore()
            return
        op = bytes(e.mimeData().data(OP_MIME)).decode("utf-8")
        pt = self.mapTo(view, e.position().toPoint())
        self.op_dropped.emit(op, view.mapToScene(pt))
        e.acceptProposedAction()

    # ── paint ──────────────────────────────────────────────────────────────────
    def paintEvent(self, _e) -> None:              # noqa: N802 — Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        p.setPen(Qt.NoPen)
        p.setBrush(T.PANEL)
        p.drawRoundedRect(r, 13, 13)
        p.setPen(QPen(T.BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r, 13, 13)

        # the "place a node here" affordance: a dashed accent square with a plus
        side = 40.0
        box = QRectF(r.center().x() - side / 2, r.top() + 16, side, side)
        pen = QPen(T.alpha(T.ACCENT, 170), 1.6, Qt.DashLine)
        pen.setDashPattern([3.0, 2.6])
        p.setPen(pen)
        p.drawRoundedRect(box, 7, 7)
        p.setPen(QPen(T.ACCENT, 2.0, Qt.SolidLine, Qt.RoundCap))
        c = box.center()
        p.drawLine(QPointF(c.x() - 8, c.y()), QPointF(c.x() + 8, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - 8), QPointF(c.x(), c.y() + 8))
        p.end()

    # ── styling ────────────────────────────────────────────────────────────────
    def restyle(self) -> None:
        tf = QFont(T.SANS, 13)
        tf.setBold(True)
        self._title.setFont(tf)
        self._sub.setFont(QFont(T.SANS, 9))
        for lab in self._hints:
            lab.setFont(QFont(T.SANS, 8))
        self.setStyleSheet(f"""
            QWidget#welcome {{ background:transparent; }}
            QLabel {{ background:transparent; color:{T.INK_2.name()}; }}
        """ + T.controls_qss() + f"""
            QPushButton[role="primary"] {{ background:{T.ACCENT.name()};
                color:{T.ACCENT_INK.name()}; border:1px solid {T.ACCENT.name()}; }}
            QPushButton[role="primary"]:hover {{
                background:{T.mix(T.ACCENT, T.INK, 0.12).name()}; }}
        """)
        self._title.setStyleSheet(f"color:{T.INK.name()}; background:transparent;")
        for lab in self._hints:
            lab.setStyleSheet(f"color:{T.MUTED.name()}; background:transparent;")
        self.update()


__all__ = ["WelcomeCard", "OP_MIME"]
