"""Bezier wire between two sockets (NodeLab v2, Phase 5).

An :class:`EdgeItem` mirrors ONE document edge tuple ``(src, src_socket, dst,
dst_socket)`` — the scene rebuilds these from the document, so the canvas can never
drift from the model. Edges are selectable (fat hit path) and deletable; a selected
wire paints in the accent color.

**Flow (2026-07-28).** While a pull is in flight, a wire whose source has already
produced carries animated accent dashes over its normal tint (:meth:`set_flow`, phase
driven by the scene's shared progress timer). The dashes are an *overlay*: the wire keeps
its domain / channel coloring, which is what it means, so the run state never overwrites
the data semantics.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QBrush, QLinearGradient, QPainterPath, QPainterPathStroker, QPen,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem

from nodegraph.sockets import SocketType
from nodelab_v2 import theme as T
from nodelab_v2.node_item import SocketItem


def _avg_color(cols: list):
    """The component-wise mean of a list of QColors (the aggregate channel tint for a
    wire carrying a multi-channel subset)."""
    n = len(cols)
    from PySide6.QtGui import QColor
    return QColor(round(sum(c.red() for c in cols) / n),
                  round(sum(c.green() for c in cols) / n),
                  round(sum(c.blue() for c in cols) / n))


def wire_path(a, b) -> QPainterPath:
    """The canonical cubic-bezier wire path from scene point ``a`` to ``b``."""
    dx = max(48.0, (b.x() - a.x()) * 0.5)
    path = QPainterPath(a)
    path.cubicTo(a.x() + dx, a.y(), b.x() - dx, b.y(), b.x(), b.y())
    return path


class EdgeItem(QGraphicsPathItem):
    """A Dataset/value wire; carries its model edge tuple for delete/sync."""

    def __init__(self, src: SocketItem, dst: SocketItem, model_edge: tuple) -> None:
        super().__init__()
        self.src = src
        self.dst = dst
        self.model_edge = model_edge      # (src_id, src_socket, dst_id, dst_socket)
        self.flow = False                # data is moving along this wire right now
        self._phase = 0.0                # dash travel, advanced by the scene's timer
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.update_path()

    def set_flow(self, on: bool) -> None:
        on = bool(on)
        if on != self.flow:
            self.flow = on
            self._phase = 0.0
            self.update()

    def advance_phase(self, step: float = 0.08) -> None:
        if not self.flow:
            return
        self._phase = (self._phase + step) % 1.0
        self.update()

    def update_path(self) -> None:
        self.setPath(wire_path(self.src.anchor(), self.dst.anchor()))

    def boundingRect(self) -> QRectF:
        """``QGraphicsPathItem`` sizes this from ``pen()`` — which this item never sets,
        so it defaults to width 1 while :meth:`paint` strokes up to 5.5 (the flow halo).
        Without the margin, every re-path (i.e. every node drag) leaves the fat parts of
        the old wire on the canvas."""
        return super().boundingRect().adjusted(-3.5, -3.5, 3.5, 3.5)

    def shape(self) -> QPainterPath:          # fat hit area for click-select
        stroker = QPainterPathStroker()
        stroker.setWidth(12.0)
        return stroker.createStroke(self.path())

    def _channel_colors(self) -> list:
        """The per-channel colors this Dataset wire carries **when it carries a strict
        subset** of the source file's channels — else ``[]`` (a full multi-channel bundle
        is not channel-tinted). A synthetic ``chK`` output carries exactly channel K; any
        other Dataset output is judged by its propagated envelope (``c`` channels, tinted
        by emission) against the upstream file total."""
        ni = self.src.node_item
        idx = ni.output_channel_index(self.src.spec.name)
        if idx is not None:
            return [ni.channel_qcolor(idx)]
        env = ni.env()
        c = env.axes.c
        if c <= 0 or "c" in env.unknown_axes:
            return []
        total = ni.doc.source_channel_total(ni.rec.id)
        if not (1 <= c < total):
            return []
        emis = env.metadata.get("channel_emission_nm")
        return [T.emission_qcolor(emis[i] if isinstance(emis, (list, tuple))
                                  and i < len(emis) else None)
                for i in range(c)]

    def _wire_colors(self) -> list:
        """The colors this wire is tinted by. A Dataset wire → one color per domain in
        the accumulated set flowing out of the source node (striped when several), each
        **blended toward the channel color** when the wire carries a strict channel
        subset (domain × channel mix). A value wire → its socket-type color. Read live so
        an upstream edit re-tints."""
        spec = self.src.spec
        if spec.type is not SocketType.DATASET:
            return [T.SOCKET.get(spec.type, T.WIRE)]
        domain_cols = [T.domain_qcolor(d) for d in self.src.node_item.out_domains()]
        domain_cols = domain_cols or [T.WIRE]
        chan_cols = self._channel_colors()
        if not chan_cols:
            return domain_cols
        agg = chan_cols[0] if len(chan_cols) == 1 else _avg_color(chan_cols)
        return [T.mix(dc, agg, 0.5) for dc in domain_cols]

    def paint(self, p, option, widget=None) -> None:
        if self.isSelected():
            pen = QPen(T.ACCENT, 3.0)
        else:
            cols = self._wire_colors()
            if len(cols) == 1:
                pen = QPen(cols[0], 2.5)
            else:                             # hard domain bands along the wire
                a, b = self.src.anchor(), self.dst.anchor()
                grad = QLinearGradient(a, b)
                n = len(cols)
                for i, col in enumerate(cols):
                    grad.setColorAt(i / n, col)
                    grad.setColorAt(min(1.0, (i + 1) / n - 1e-4), col)
                pen = QPen(QBrush(grad), 2.5)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawPath(self.path())
        if self.flow:
            # halo + travelling dashes ON TOP of the tint (never instead of it)
            glow = QPen(T.alpha(T.ACCENT, 52), 5.5)
            glow.setCapStyle(Qt.RoundCap)
            p.setPen(glow)
            p.drawPath(self.path())
            # slightly WIDER than the tinted base so the gaps read as gaps (a narrower
            # dash pen just fringes the wire instead of breaking it up)
            dash = QPen(T.ACCENT, 2.8)
            dash.setCapStyle(Qt.FlatCap)
            dash.setDashPattern([2.5, 1.8])          # → 7 px on, 5 px off at width 2.8
            dash.setDashOffset(-self._phase * 12.0 / 2.8)
            p.setPen(dash)
            p.drawPath(self.path())


__all__ = ["EdgeItem", "wire_path"]
