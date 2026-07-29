"""Labelled frame graphics item (NodeLab v2) — a Blender-style frame that visually
groups a set of nodes on the canvas.

A frame is **GUI-only**: it never enters the run graph (its record rides in the
document's ``ui`` extras, like node positions). It paints BEHIND the nodes, auto-sizes
to enclose its member :class:`~nodelab_v2.node_item.NodeItem`\\ s (recomputed on every
node move via :meth:`GraphScene.reroute`), and dragging the frame moves all its members
together. A frame always has ≥1 member (an emptied frame is removed by the document), so
it needs no stored geometry — the member bounds ARE its geometry.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

from nodelab_v2 import theme as T


class FrameItem(QGraphicsObject):
    """A rounded, tinted rectangle with a title bar, drawn behind its member nodes."""

    PAD = 22.0            # margin between the member bounds and the frame edge
    TITLE_H = 24.0        # title-bar height (above the members)
    Z = -5.0              # behind nodes (0) and wires

    def __init__(self, frame_rec, scene) -> None:
        super().__init__()
        self.rec = frame_rec
        self._scene = scene
        self._rect = QRectF(0, 0, 160, 90)
        self.setZValue(self.Z)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.SizeAllCursor)
        self._drag_last: Optional[QPointF] = None
        self.reflow()

    @property
    def frame_id(self) -> str:
        return self.rec.id

    # ── geometry (follows the members) ─────────────────────────────────────────
    def _member_items(self) -> List["QGraphicsObject"]:
        out = []
        for nid in self.rec.members:
            it = self._scene.node_items.get(nid)
            if it is not None and it.scene() is not None:
                out.append(it)
        return out

    def reflow(self) -> None:
        """Recompute position + size to enclose the member nodes (+ padding + a title
        bar). Cheap; called from :meth:`GraphScene.sync` and on every node move."""
        items = self._member_items()
        self.prepareGeometryChange()
        if not items:
            self.update()                       # document removes emptied frames; be safe
            return
        bounds = None
        for it in items:
            # card_rect, not boundingRect: the latter carries the glow repaint margin, so
            # a frame would breathe in/out as its members get selected.
            r = it.mapToScene(it.card_rect()).boundingRect()
            bounds = r if bounds is None else bounds.united(r)
        bounds = bounds.adjusted(-self.PAD, -self.PAD - self.TITLE_H,
                                 self.PAD, self.PAD)
        self.setPos(bounds.topLeft())
        self._rect = QRectF(0, 0, bounds.width(), bounds.height())
        self.update()

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-2, -2, 2, 2)   # the 2 px selected border + AA

    def _color(self) -> QColor:
        return QColor(*self.rec.color) if self.rec.color else QColor(T.ACCENT)

    # ── painting ────────────────────────────────────────────────────────────────
    def paint(self, p: QPainter, *_a) -> None:
        p.setRenderHint(QPainter.Antialiasing, True)
        col = self._color()
        sel = self.isSelected()
        p.setPen(QPen(col if sel else T.alpha(col, 150), 2.0 if sel else 1.4))
        p.setBrush(T.alpha(col, 30))
        p.drawRoundedRect(self._rect, 9, 9)
        # title bar (a tinted band across the top; rounded top via an overlapping rect)
        band = QRectF(0, 0, self._rect.width(), self.TITLE_H + 9)
        p.setPen(Qt.NoPen)
        p.setBrush(T.alpha(col, 60))
        p.drawRoundedRect(band, 9, 9)
        p.setBrush(T.alpha(col, 60))
        p.drawRect(QRectF(0, self.TITLE_H - 1, self._rect.width(), 10))
        f = QFont(T.SANS, 8)
        f.setBold(True)
        p.setFont(f)
        p.setPen(T.INK)
        p.drawText(QRectF(11, 0, self._rect.width() - 22, self.TITLE_H),
                   Qt.AlignVCenter | Qt.AlignLeft, self.rec.title)

    # ── drag = move all members together ────────────────────────────────────────
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.setSelected(True)
            self._drag_last = e.scenePos()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if self._drag_last is not None:
            delta = e.scenePos() - self._drag_last
            self._drag_last = e.scenePos()
            for it in self._member_items():
                # moving each NodeItem fires its itemChange → doc.set_pos + scene.reroute
                # (which reflows this frame to follow) — no direct frame move needed.
                it.moveBy(delta.x(), delta.y())
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if self._drag_last is not None:
            self._drag_last = None
            e.accept()
            return
        super().mouseReleaseEvent(e)


__all__ = ["FrameItem"]
