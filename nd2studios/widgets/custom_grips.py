"""
Edge-resize grip for a frameless window.

A trimmed-down rewrite of `Modern_GUI_PyDracula`'s `CustomGrip`. Each
instance covers one edge of the parent window and resizes it on drag.
Cursor shape is set per-edge so the user gets the standard resize hint.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QCursor, QMouseEvent
from PySide6.QtWidgets import QWidget


_EDGE_CURSOR = {
    Qt.LeftEdge: Qt.SizeHorCursor,
    Qt.RightEdge: Qt.SizeHorCursor,
    Qt.TopEdge: Qt.SizeVerCursor,
    Qt.BottomEdge: Qt.SizeVerCursor,
}


class CustomGrip(QWidget):
    """Transparent, drag-to-resize grip on one edge of a frameless window."""

    def __init__(self, parent: QWidget, edge: Qt.Edge):
        super().__init__(parent)
        self._parent = parent
        self._edge = edge
        self.setCursor(QCursor(_EDGE_CURSOR.get(edge, Qt.ArrowCursor)))
        # Transparent — the grip is invisible chrome; the visible edge is
        # the parent's `#bgApp` border.
        self.setStyleSheet("background: transparent;")
        self._mouse_pos: QPoint | None = None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._mouse_pos = event.globalPosition().toPoint()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._mouse_pos is None or event.buttons() != Qt.LeftButton:
            return
        gp = event.globalPosition().toPoint()
        delta = gp - self._mouse_pos
        self._mouse_pos = gp
        self._resize_parent(delta)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._mouse_pos = None
        event.accept()

    def _resize_parent(self, delta: QPoint) -> None:
        p = self._parent
        geo: QRect = p.geometry()
        min_w = max(p.minimumWidth(), 200)
        min_h = max(p.minimumHeight(), 200)

        if self._edge == Qt.LeftEdge:
            new_left = min(geo.left() + delta.x(), geo.right() - min_w)
            geo.setLeft(new_left)
        elif self._edge == Qt.RightEdge:
            geo.setRight(max(geo.right() + delta.x(), geo.left() + min_w))
        elif self._edge == Qt.TopEdge:
            new_top = min(geo.top() + delta.y(), geo.bottom() - min_h)
            geo.setTop(new_top)
        elif self._edge == Qt.BottomEdge:
            geo.setBottom(max(geo.bottom() + delta.y(), geo.top() + min_h))

        p.setGeometry(geo)
