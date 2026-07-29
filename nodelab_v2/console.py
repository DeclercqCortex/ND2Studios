"""Console panel — a copyable log of run activity and, above all, node failures.

Node failures used to surface only as a Viewer tooltip (uncopyable). This dockable
panel keeps a running, **selectable** monospace log: ``started`` / ``finished`` info
lines plus the *full* traceback of any failed pull, with a one-click *Copy all* so the
error can be pasted verbatim.

Rich text (per-line color) via ``QTextEdit`` + ``QTextCharFormat`` — deliberately NOT
HTML, so ``toPlainText()`` / the clipboard return the traceback with its exact
whitespace (regular spaces, indentation intact), which is the whole point. Qt-only; the
window feeds it from the runner signals.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from nodelab_v2 import theme as T

_MAX_LINES = 4000        # bound memory; oldest lines drop past this


class ConsolePanel(QWidget):
    """A read-only, selectable log. Call :meth:`log` (or :meth:`error`) to append."""

    def __init__(self) -> None:
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 8)
        v.setSpacing(5)

        # No title label / dock title (both were redundant "Console" chrome) — just a
        # slim right-aligned Copy/Clear strip floating above the flush log.
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        head.addStretch(1)
        self._copy = QPushButton("Copy all")
        self._copy.setProperty("role", "mini")
        self._copy.setCursor(Qt.PointingHandCursor)
        self._copy.clicked.connect(self.copy_all)
        self._clear = QPushButton("Clear")
        self._clear.setProperty("role", "mini")
        self._clear.setCursor(Qt.PointingHandCursor)
        self._clear.clicked.connect(self.clear)
        head.addWidget(self._copy)
        head.addWidget(self._clear)
        v.addLayout(head)

        self._out = QTextEdit()
        self._out.setReadOnly(True)
        self._out.setLineWrapMode(QTextEdit.NoWrap)
        v.addWidget(self._out, 1)

        self.restyle()

    # ── API ───────────────────────────────────────────────────────────────────
    def log(self, text: str, *, level: str = "info") -> None:
        """Append ``text`` as one timestamped entry (multi-line kept intact, whitespace
        preserved). ``level`` is ``"info"`` | ``"ok"`` | ``"error"`` and only tints the
        leading ``HH:MM:SS ·`` marker — the body stays in the readable ink color."""
        stamp = time.strftime("%H:%M:%S")
        mark = {"error": "✗", "ok": "✓"}.get(level, "·")
        mark_col = {"error": T.ERROR, "ok": T.WIRE}.get(level, T.MUTED)

        head = QTextCharFormat()
        head.setForeground(mark_col)
        body = QTextCharFormat()
        body.setForeground(T.INK)

        cur = self._out.textCursor()
        cur.movePosition(QTextCursor.End)
        if not self._out.document().isEmpty():
            cur.insertText("\n")
        cur.insertText(f"{stamp} {mark} ", head)
        first, *rest = text.rstrip("\n").splitlines() or [""]
        cur.insertText(first, body)
        for ln in rest:
            cur.insertText("\n" + ln, body)
        self._trim()
        self._out.moveCursor(QTextCursor.End)
        self._out.ensureCursorVisible()

    def error(self, text: str) -> None:
        self.log(text, level="error")

    def copy_all(self) -> None:
        QGuiApplication.clipboard().setText(self._out.toPlainText())

    def clear(self) -> None:
        self._out.clear()

    def _trim(self) -> None:
        """Drop oldest lines once the log exceeds :data:`_MAX_LINES` blocks."""
        doc = self._out.document()
        extra = doc.blockCount() - _MAX_LINES
        if extra <= 0:
            return
        cur = QTextCursor(doc)
        cur.movePosition(QTextCursor.Start)
        cur.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor, extra)
        cur.removeSelectedText()

    def restyle(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{ background:{T.PANEL.name()}; color:{T.INK.name()}; }}
            QTextEdit {{ background:{T.BG.name()}; color:{T.INK.name()};
                border:1px solid {T.BORDER.name()}; border-radius:8px;
                selection-background-color:{T.ACCENT_DIM.name()};
                font-family:{T.MONO}; font-size:12px; padding:6px 8px; }}
            QPushButton[role="mini"] {{ background:transparent; color:{T.MUTED.name()};
                border:1px solid {T.BORDER.name()}; border-radius:6px;
                padding:2px 10px; font-size:11px; font-weight:600; }}
            QPushButton[role="mini"]:hover {{ background:{T.PANEL_HI.name()};
                color:{T.INK.name()}; border-color:{T.BORDER_HI.name()}; }}
            QPushButton[role="mini"]:pressed {{ background:{T.ACCENT_DIM.name()}; }}
        """ + T.controls_qss())


__all__ = ["ConsolePanel"]
