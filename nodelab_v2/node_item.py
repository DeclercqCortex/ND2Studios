"""Node / socket / 2D-3D-switch graphics items (NodeLab v2, Phase 5).

A :class:`NodeItem` is a *view* of a :class:`~nodelab_v2.document.NodeRecord`: it reads
params/modes from the record's live dicts (shared with the inspector) and its
metadata-derived ``ƒmd`` pills from the document's propagated
:class:`~nodegraph.metadata.MetaEnvelope` (the G8 live re-seed — edit an upstream node
and every downstream auto pill updates). Field sockets draw as a diamond, single-value
as a circle, the Dataset main wire as a larger dot — colors from
:data:`nodegraph.sockets.SOCKET_COLOR`.

Directive guards (V2.03 H11 / G8): the 2D/3D switch is DISABLED when the incoming
envelope's z is known to be 1 (unknown ≠ 1 — an unresolved source never greys it), and
a node locked to 3D while z==1 paints a red validation badge. Muted nodes (G3) dim and
tag; pressing a socket starts a wire drag (G1, handled by the scene).

**Run state (2026-07-28).** Every card wears a 2 px accent rail on its header's bottom
edge plus a status dot at the header's right, fed by
:class:`~nodelab_v2.runner.EngineRunner`'s per-node events (:meth:`set_run_state`):
*queued* = a hollow dot, *running* = a pulsing dot over a rail that fills when the
compute reports ``ctx.progress`` and sweeps when it does not, *done* = a solid glowing
dot over a full rail, *cached* = a hollow ring (nothing was recomputed), *error* = red.
A working card also takes an accent border + outer glow, and its outgoing wires flow.
The exact percentage / wall time lives in the card's **tooltip** and the status bar, so a
busy canvas reads as light rather than as a wall of tiny text. A lazy node finishes in
microseconds by design — its rail flashes and the real cost appears on whichever node
actually reads the planes, which is the truth, not a bug.

**Delete affordance.** Hovering a card reveals a ✕ badge on its top-right corner
(:class:`CloseItem`); clicking it asks the scene to delete that node. The keyboard
``Del`` and the right-click menu do the same thing — the badge is the discoverable one.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

from nodegraph.domains import domain_abbr
from nodegraph.groups import group_name_of
from nodegraph.metadata import envelope_symbols, eval_derive
from nodegraph.registry import NODES
from nodegraph.sockets import SocketType
from nodelab_v2 import theme as T
from nodelab_v2.document import GraphDocument, NodeRecord, TITLE_KEY

#: a synthetic per-channel output socket name — ``ch0``, ``ch1``, …
_CH_SOCKET_RE = re.compile(r"^ch(\d+)$")


# ── socket ────────────────────────────────────────────────────────────────────

class SocketItem(QGraphicsItem):
    """A single socket dot; its ``scenePos()`` is the wire anchor. Pressing it begins
    a wire drag (delegated to the scene); ``highlight`` paints the drop-target ring."""

    R = 9.0

    def __init__(self, parent: "NodeItem", spec, io: str) -> None:
        super().__init__(parent)
        self.spec = spec
        self.io = io                       # "in" | "out"
        self.node_item = parent
        self.channel_color: Optional[QColor] = None   # per-channel output tint (chK)
        self.highlight: Optional[bool] = None   # None | True (valid) | False (invalid)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CrossCursor)
        kind = ("Dataset" if spec.type is SocketType.DATASET
                else (f"field · {spec.type.value}" if spec.is_field else spec.type.value))
        tip = f"{spec.name} — {kind}"
        if spec.unit:
            tip += f" · {spec.unit}"
        if getattr(spec, "multi", False):
            tip += " · multi"
        self._base_tip = tip
        self.setToolTip(tip)

    def set_domain_tip(self, domains, missing=()) -> None:
        """Append the domain-set this Dataset socket carries/requires (the rail's
        hover detail). No-op tail for value sockets (``domains`` empty)."""
        tip = self._base_tip
        if domains:
            names = ", ".join(d.value for d in domains)
            tip += ("\ncarries: " if self.io == "out" else "\nrequires: ") + names
        if missing:
            tip += "\n⚠ missing upstream: " + ", ".join(d.value for d in missing)
        self.setToolTip(tip)

    def boundingRect(self) -> QRectF:
        # +1 for the antialiased outer edge of the highlight ring (drawn at radius
        # ``R - 1`` with a 2 px pen, i.e. exactly out to R): without the slack, moving a
        # socket leaves a hairline of the ring behind.
        g = self.R + 1.0
        return QRectF(-g, -g, 2 * g, 2 * g)

    def paint(self, p: QPainter, *_a) -> None:
        p.setRenderHint(QPainter.Antialiasing, True)
        col = T.SOCKET[self.spec.type]
        if self.spec.type is SocketType.DATASET and self.channel_color is not None:
            col = self.channel_color               # per-channel output dot tint
        if self.highlight is not None:
            ring = T.WIRE if self.highlight else T.ERROR
            p.setPen(QPen(ring, 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(0, 0), self.R - 1, self.R - 1)
        p.setPen(QPen(T.BG, 2))
        if self.spec.type is SocketType.DATASET:
            p.setBrush(col)
            p.drawEllipse(QPointF(0, 0), 6.5, 6.5)
        elif self.spec.is_field:
            path = QPainterPath()
            r = 5.5
            path.moveTo(0, -r); path.lineTo(r, 0); path.lineTo(0, r); path.lineTo(-r, 0)
            path.closeSubpath()
            p.setBrush(col)
            p.drawPath(path)
        else:
            p.setBrush(col)
            p.drawEllipse(QPointF(0, 0), 5.0, 5.0)

    def anchor(self) -> QPointF:
        return self.scenePos()

    def set_highlight(self, state: Optional[bool]) -> None:
        if state != self.highlight:
            self.highlight = state
            self.update()

    def mousePressEvent(self, e) -> None:
        sc = self.scene()
        if e.button() == Qt.LeftButton and sc is not None and hasattr(sc, "begin_wire"):
            sc.begin_wire(self, e.scenePos())
            e.accept()
            return
        super().mousePressEvent(e)


# ── 2D / 3D switch ──────────────────────────────────────────────────────────────

class SwitchItem(QGraphicsObject):
    """The two-color on/off switch — amber for 2D, cyan for 3D (both active states).
    ``allow_3d=False`` (incoming z known == 1, H11) greys the 3D side and ignores
    clicks toward 3D."""

    toggled = Signal(str)
    TRACK_W, TRACK_H, KNOB = 34.0, 18.0, 14.0
    LBL_W = 17.0
    WIDTH = LBL_W + 5 + TRACK_W + 5 + LBL_W

    def __init__(self, parent: "NodeItem", dim: str) -> None:
        super().__init__(parent)
        self.dim = dim
        self.allow_3d = True
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.WIDTH, 20)

    def set_allow_3d(self, allow: bool) -> None:
        if allow != self.allow_3d:
            self.allow_3d = allow
            self.setToolTip("" if allow else
                            "3D is unavailable: the incoming data has z == 1 (H11)")
            self.update()

    def paint(self, p: QPainter, *_a) -> None:
        p.setRenderHint(QPainter.Antialiasing, True)
        on = self.dim == "3D"
        f = QFont(T.MONO, 7); f.setBold(True)
        p.setFont(f)
        p.setPen(T.INK if not on else T.MUTED)
        p.drawText(QRectF(0, 0, self.LBL_W, 20), Qt.AlignCenter, "2D")
        p.setPen((T.INK if on else T.MUTED) if self.allow_3d else T.alpha(T.MUTED, 110))
        p.drawText(QRectF(self.WIDTH - self.LBL_W, 0, self.LBL_W, 20),
                   Qt.AlignCenter, "3D")
        tx = self.LBL_W + 5
        track = QRectF(tx, 1, self.TRACK_W, self.TRACK_H)
        col = T.ACCENT if on else T.DIM2D
        if not self.allow_3d and not on:
            col = T.mix(col, T.PANEL, 0.35)
        p.setPen(QPen(col, 1))
        p.setBrush(col)
        p.drawRoundedRect(track, self.TRACK_H / 2, self.TRACK_H / 2)
        kx = tx + (self.TRACK_W - self.KNOB - 1) if on else tx + 1
        p.setPen(Qt.NoPen)
        p.setBrush(T.ACCENT_INK if on else T.DIM2D_INK)
        p.drawEllipse(QRectF(kx, 2, self.KNOB, self.KNOB))

    def set_dim(self, dim: str) -> None:
        if dim != self.dim:
            self.dim = dim
            self.update()
            self.toggled.emit(dim)

    def mousePressEvent(self, e) -> None:
        target = "3D" if self.dim == "2D" else "2D"
        if target == "3D" and not self.allow_3d:
            e.accept()
            return
        self.set_dim(target)
        e.accept()


# ── hover ✕ delete badge ──────────────────────────────────────────────────────

class CloseItem(QGraphicsObject):
    """The delete badge on a card's top-right corner: visible only while the card is
    hovered (or the badge itself is), so it never competes with the node's content. It
    only *asks* — :class:`NodeItem` re-emits and the scene owns the actual removal, so
    every delete path (badge, ``Del``, context menu) runs the same code."""

    clicked = Signal()

    def __init__(self, parent: "NodeItem") -> None:
        super().__init__(parent)
        self._hot = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.ArrowCursor)
        self.setToolTip("Delete this node (Del)")
        self.setVisible(False)
        self.setZValue(4)          # above the header gradient and the 2D/3D switch

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, T.CLOSE_BTN, T.CLOSE_BTN)

    def paint(self, p: QPainter, *_a) -> None:
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.boundingRect().adjusted(0.5, 0.5, -0.5, -0.5)
        col = T.ERROR if self._hot else T.MUTED
        p.setPen(QPen(col, 1))
        p.setBrush(T.mix(T.PANEL, col, 0.30) if self._hot else T.PANEL)
        p.drawEllipse(r)
        p.setPen(QPen(T.INK if self._hot else T.INK_2, 1.4))
        d = 3.6
        cx, cy = r.center().x(), r.center().y()
        p.drawLine(QPointF(cx - d, cy - d), QPointF(cx + d, cy + d))
        p.drawLine(QPointF(cx - d, cy + d), QPointF(cx + d, cy - d))

    def hoverEnterEvent(self, e) -> None:
        self._hot = True
        self.update()
        super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e) -> None:
        self._hot = False
        self.update()
        # the badge overhangs the card's corner, so leaving it can mean leaving the card
        # too — let the card re-decide (it checks whether anything is still hovered)
        parent = self.parentItem()
        if isinstance(parent, NodeItem):
            parent.hide_close_if_away()
        super().hoverLeaveEvent(e)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            e.accept()               # swallow the press so the card doesn't start moving
            return
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.boundingRect().contains(e.pos()):
            self.clicked.emit()
            e.accept()
            return
        super().mouseReleaseEvent(e)


# ── node card ────────────────────────────────────────────────────────────────

class NodeItem(QGraphicsObject):
    """A node card bound to a document :class:`NodeRecord`."""

    changed = Signal(object)          # emitted on dim/param/relayout (self)
    delete_requested = Signal(str)    # node_id — the hover ✕ badge was clicked

    def __init__(self, rec: NodeRecord, doc: GraphDocument) -> None:
        super().__init__()
        self.rec = rec
        self.doc = doc
        self.spec = NODES.get(rec.op_key)
        self._is_reroute = rec.op_key == "rr.reroute"
        self._group_name = group_name_of(rec.op_key)   # a group instance? → its name
        self._is_group = self._group_name is not None
        self._sockets: Dict[Tuple[str, str], SocketItem] = {}
        self._rows: List[tuple] = []
        self._viewed = False              # the Viewer / mini-map is showing this node
        self._width = float(T.NODE_W)
        self._height = float(T.HEADER_H)
        self._switch: Optional[SwitchItem] = None
        # run state (fed by the runner's per-node events; see set_run_state)
        self._run = ""                    # "" | queued | running | decoding | cached
        self._run_frac: Optional[float] = None    # None → indeterminate
        self._run_note = ""
        self._run_secs: Optional[float] = None
        self._phase = 0.0                 # marquee position for the indeterminate sweep
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        if self.spec is not None and self.spec.has_dim_lever():
            self._switch = SwitchItem(self, self.dim)
            self._switch.setPos(T.NODE_W - SwitchItem.WIDTH - 10, 9)
            self._switch.toggled.connect(self.set_dim)
        self._close = CloseItem(self)
        self._close.clicked.connect(lambda: self.delete_requested.emit(self.rec.id))
        self._shown_collapsed = rec.collapsed
        self.setPos(rec.x, rec.y)
        self._layout()

    # ── model passthroughs ────────────────────────────────────────────────────
    @property
    def node_id(self) -> str:
        return self.rec.id

    @property
    def op_key(self) -> str:
        return self.rec.op_key

    @property
    def params(self) -> dict:
        return self.rec.params

    @property
    def dim(self) -> str:
        st = self.rec.state()
        return st.get("dim", "2D")

    @property
    def locked(self) -> set:
        return self.rec.locked

    def env(self):
        return self.doc.env(self.rec.id)

    def state(self) -> dict:
        return self.rec.state()

    def _active_inputs(self):
        return self.doc.input_specs(self.rec.id)

    def _active_outputs(self):
        # instance-aware: a source/split card grows one synthetic ``chK`` output per
        # channel (the document owns the resolution + the channel descriptors).
        return self.doc.output_specs(self.rec.id)

    # ── per-channel outputs (chK) ─────────────────────────────────────────────
    @staticmethod
    def output_channel_index(socket_name: str) -> Optional[int]:
        """The channel index K if ``socket_name`` is a synthetic ``chK`` output, else
        ``None``. Used by the edge painter to tint the wire by that channel's color."""
        m = _CH_SOCKET_RE.match(socket_name)
        return int(m.group(1)) if m else None

    def channel_qcolor(self, index: int) -> QColor:
        """The display color of channel ``index``: its native color if the descriptor
        carries one, else a color derived from its emission wavelength (neutral grey
        when unknown — e.g. a TIFF or a transmitted-light channel)."""
        descs = self.doc.channel_descriptors(self.rec.id)
        if 0 <= index < len(descs):
            ch = descs[index]
            col = ch.get("color")
            if col:
                return QColor(int(col[0]), int(col[1]), int(col[2]))
            return T.emission_qcolor(ch.get("emission_nm"))
        return T.SOCKET[SocketType.DATASET]

    def set_viewed(self, on: bool) -> None:
        """Flag this card as the one the Viewer is showing (accent spine + live dot)."""
        on = bool(on)
        if on != self._viewed:
            self._viewed = on
            self.update()

    def granularity(self) -> str:
        g = self.spec.resolve_granularity(self.state()) if self.spec else None
        return g.value if g is not None else "tileable"

    # H11: the incoming z is KNOWN to be 1 (unknown never counts as 1)
    def z_is_one(self) -> bool:
        env = self.env()
        return env.axes.z == 1 and "z" not in env.unknown_axes

    def dim_invalid(self) -> bool:
        """A 3D lever on known z==1 data — the H11 red-badge validation error."""
        return (self.spec is not None and self.spec.has_dim_lever()
                and self.dim == "3D" and self.z_is_one())

    # ── domain interface (socket rail + wire tint) ────────────────────────────
    def _dsorted(self, domains) -> tuple:
        return tuple(sorted(domains, key=lambda d: d.value))

    def reads_domains(self) -> tuple:
        """Domains this node requires on its Dataset input (the input rail)."""
        return self._dsorted(self.spec.reads_domains) if self.spec else ()

    def out_domains(self) -> tuple:
        """The accumulated domain-set flowing out of this node (the output rail +
        the tint of every wire leaving it)."""
        return self._dsorted(self.env().domains)

    def missing_domains(self) -> frozenset:
        """Required domains absent upstream — the red validation chips (H-domains)."""
        return self.doc.missing_domains(self.rec.id)

    def resolved(self, s) -> object:
        """The pill value: an explicit param, else the LIVE metadata-derived value
        from this node's propagated envelope (G8), else the static default."""
        if s.name in self.rec.params:
            return self.rec.params[s.name]
        if s.derive:
            try:
                v = eval_derive(s.derive, envelope_symbols(self.env()))
                if isinstance(v, float):
                    return round(v, 4)
                return v
            except Exception:  # noqa: BLE001 — missing symbols → placeholder
                return "auto"
        return s.default

    def is_derived(self, s) -> bool:
        return bool(s.derive) and s.name not in self.locked and s.name not in self.rec.params

    # ── layout ──────────────────────────────────────────────────────────────────
    def _apply_domain_tip(self, sock: "SocketItem", s, io: str) -> None:
        if s.type is not SocketType.DATASET:
            return
        if io == "in":
            sock.set_domain_tip(self.reads_domains(), self.missing_domains())
        else:
            sock.set_domain_tip(self.out_domains())

    def _tint_channel_socket(self, sock: "SocketItem", s) -> None:
        idx = self.output_channel_index(s.name)
        if idx is not None:
            sock.channel_color = self.channel_qcolor(idx)

    def _clear_sockets(self) -> None:
        for sock in self._sockets.values():
            sock.setParentItem(None)
            if self.scene() is not None:
                self.scene().removeItem(sock)
        self._sockets.clear()

    def _layout(self) -> None:
        self.prepareGeometryChange()
        self._clear_sockets()
        self._rows = []
        self._width = float(T.NODE_W)
        self._place_close()
        if self._is_reroute:
            self._layout_reroute()
            return
        if self.rec.collapsed:
            self._layout_collapsed()
            return
        y = T.HEADER_H + T.GRAN_H
        for s in self._active_inputs():
            sock = SocketItem(self, s, "in")
            sock.setPos(0, y + T.ROW_H / 2)
            self._apply_domain_tip(sock, s, "in")
            self._sockets[("in", s.name)] = sock
            self._rows.append(("in", s, y))
            y += T.ROW_H
        for m in (self.spec.modes if self.spec else ()):
            if m.is_dim_lever:
                continue
            self._rows.append(("mode", m, y))
            y += T.ROW_H
        for s in self._active_outputs():
            sock = SocketItem(self, s, "out")
            sock.setPos(T.NODE_W, y + T.ROW_H / 2)
            self._apply_domain_tip(sock, s, "out")
            self._tint_channel_socket(sock, s)
            self._sockets[("out", s.name)] = sock
            self._rows.append(("out", s, y))
            y += T.ROW_H
        self._height = y + T.PAD_BOTTOM
        if self._switch is not None:
            self._switch.set_allow_3d(not self.z_is_one())
        self.update()

    def _layout_reroute(self) -> None:
        """A reroute renders as a compact dot: one Dataset input on the left edge, one
        output on the right, both at mid-height; no header/rows/switch."""
        r = T.RR_SIZE / 2.0
        for s in self._active_inputs():
            sock = SocketItem(self, s, "in")
            sock.setPos(0, r)
            self._apply_domain_tip(sock, s, "in")
            self._sockets[("in", s.name)] = sock
        for s in self._active_outputs():
            sock = SocketItem(self, s, "out")
            sock.setPos(T.RR_SIZE, r)
            self._apply_domain_tip(sock, s, "out")
            self._sockets[("out", s.name)] = sock
        self._width = float(T.RR_SIZE)
        self._height = float(T.RR_SIZE)
        self.update()

    def _layout_collapsed(self) -> None:
        """Compact: header only, ALL active sockets kept (so wires stay valid) but
        stacked at the card edges (Blender-style collapse)."""
        ins, outs = list(self._active_inputs()), list(self._active_outputs())
        band = max(len(ins), len(outs))
        y0 = T.HEADER_H + 8
        for i, s in enumerate(ins):
            sock = SocketItem(self, s, "in")
            sock.setPos(0, y0 + i * 12)
            self._apply_domain_tip(sock, s, "in")
            self._sockets[("in", s.name)] = sock
        for i, s in enumerate(outs):
            sock = SocketItem(self, s, "out")
            sock.setPos(T.NODE_W, y0 + i * 12)
            self._apply_domain_tip(sock, s, "out")
            self._tint_channel_socket(sock, s)
            self._sockets[("out", s.name)] = sock
        self._height = T.HEADER_H + max(0, band) * 12 + 12
        if self._switch is not None:
            self._switch.set_allow_3d(not self.z_is_one())
        self.update()

    def refresh(self) -> None:
        """Re-read model state (envelope/derives/mute/guards) — cheap, no relayout
        unless the active socket set OR the collapsed state changed."""
        want = ([s.name for s in self._active_inputs()],
                [s.name for s in self._active_outputs()])
        have = ([k[1] for k in self._sockets if k[0] == "in"],
                [k[1] for k in self._sockets if k[0] == "out"])
        collapse_changed = getattr(self, "_shown_collapsed", None) != self.rec.collapsed
        if want != have or collapse_changed:
            self._shown_collapsed = self.rec.collapsed
            self._layout()
        elif self._switch is not None:
            self._switch.set_allow_3d(not self.z_is_one())
            if self._switch.dim != self.dim:
                self._switch.dim = self.dim
                self._switch.update()
        self.update()

    def set_dim(self, dim: str) -> None:
        if dim == self.dim:
            return
        self.rec.modes["dim"] = dim
        if self._switch is not None and self._switch.dim != dim:
            self._switch.dim = dim
            self._switch.update()
        self._layout()
        if self.scene() is not None and hasattr(self.scene(), "reroute"):
            self.scene().reroute()
        self.doc.touch()
        self.changed.emit(self)

    def socket(self, io: str, name: str) -> Optional[SocketItem]:
        return self._sockets.get((io, name))

    def sockets(self) -> List[SocketItem]:
        return list(self._sockets.values())

    # ── painting ────────────────────────────────────────────────────────────────
    #: How far outside the card :meth:`_paint_card_glow` / the reroute ring reach — the
    #: outermost pass is grown by 3.0 with a 1.6 px pen, so ~4 px plus antialiasing.
    GLOW_M = 5.0

    def card_rect(self) -> QRectF:
        """The card's *geometry* — what the node visually occupies, excluding the glow.
        Hit-testing, frame bounds and layout use this; only :meth:`boundingRect` (the
        repaint region) carries the glow margin."""
        return QRectF(0, 0, self._width, self._height)

    def boundingRect(self) -> QRectF:
        # MUST cover every pixel paint() touches, glow included. Qt only repaints the
        # boundingRect it is told about, so anything painted outside it is left behind as
        # a smear when the card moves (the trailing outlines while dragging).
        m = self.GLOW_M
        return self.card_rect().adjusted(-m, -m, m, m)

    def shape(self) -> QPainterPath:
        """Clicks and rubber-band selection follow the card, not its glow margin."""
        path = QPainterPath()
        if self._is_reroute:
            path.addEllipse(self.card_rect())
        else:
            path.addRoundedRect(self.card_rect(), T.RADIUS, T.RADIUS)
        return path

    def _paint_reroute_progress(self, p: QPainter) -> None:
        """A reroute is too small for a rail — its run state reads as a glowing ring
        (pulsing while it works), the same language as the cards' header dot."""
        if not self._run or self._run == "queued":
            return
        col = self._run_color()
        if self._run in ("running", "decoding"):
            col = T.alpha(col, 96 + int(159 * abs(1.0 - 2.0 * ((self._phase * 1.7) % 1.0))))
        p.setBrush(Qt.NoBrush)
        for grow, a in ((2.0, 40),):
            p.setPen(QPen(T.alpha(col, a), 2))
            p.drawEllipse(QRectF(1 - grow, 1 - grow, T.RR_SIZE - 2 + 2 * grow,
                                 T.RR_SIZE - 2 + 2 * grow))
        p.setPen(QPen(col, 2))
        p.drawEllipse(QRectF(1, 1, T.RR_SIZE - 2, T.RR_SIZE - 2))

    def _paint_reroute(self, p: QPainter) -> None:
        """A small rounded dot in the Dataset-socket colour; the in/out SocketItems sit
        on its left/right edges. Selection/mute read the same as a full card."""
        p.setRenderHint(QPainter.Antialiasing, True)
        if self.rec.muted:
            p.setOpacity(0.45)
        r = T.RR_SIZE / 2.0
        p.setPen(QPen(T.ACCENT if self.isSelected() else T.BORDER,
                      2 if self.isSelected() else 1))
        p.setBrush(T.PANEL)
        p.drawEllipse(QRectF(1, 1, T.RR_SIZE - 2, T.RR_SIZE - 2))
        p.setPen(Qt.NoPen)
        p.setBrush(T.SOCKET[SocketType.DATASET])
        p.drawEllipse(QPointF(r, r), 4.5, 4.5)

    def paint(self, p: QPainter, *_a) -> None:
        if self._is_reroute:
            self._paint_reroute(p)
            self._paint_reroute_progress(p)
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        if self.rec.muted:
            p.setOpacity(0.45)
        spec = self.spec
        cat = "group" if self._is_group else (spec.category if spec else "general")
        hdr = T.category_color(cat)
        rect = QRectF(0, 0, T.NODE_W, self._height)

        failed = self._run == "error"          # this node is where the pull broke
        working = self._run in ("running", "decoding")
        border = T.ERROR if (self.dim_invalid() or failed) else (
            T.ACCENT if self.isSelected() else
            (T.alpha(T.ACCENT, 170) if working else T.BORDER))
        body = rect.adjusted(0.5, 0.5, -0.5, -0.5)
        if failed or working or self.isSelected():
            self._paint_card_glow(p, body, T.ERROR if failed else T.ACCENT)
        p.setPen(QPen(border, 2 if (self.isSelected() or self.dim_invalid() or failed)
                      else 1))
        p.setBrush(T.PANEL)
        p.drawRoundedRect(body, T.RADIUS, T.RADIUS)

        hpath = QPainterPath()
        hpath.addRoundedRect(QRectF(1, 1, T.NODE_W - 2, T.HEADER_H), T.RADIUS, T.RADIUS)
        hpath.addRect(QRectF(1, T.HEADER_H - T.RADIUS, T.NODE_W - 2, T.RADIUS))
        grad = QLinearGradient(0, 0, 0, T.HEADER_H)
        grad.setColorAt(0, T.mix(T.PANEL, hdr, 0.26))
        grad.setColorAt(1, T.PANEL)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawPath(hpath)
        # the category spine — accent and full-height while this card is the one the
        # Viewer (or the mini-map) is showing
        p.setBrush(T.ACCENT if self._viewed else hdr)
        p.drawRoundedRect(
            QRectF(1, 1, 3, (self._height - 2) if self._viewed else T.HEADER_H), 1.5, 1.5)
        p.setPen(QPen(T.BORDER, 1))
        p.drawLine(QPointF(1, T.HEADER_H), QPointF(T.NODE_W - 1, T.HEADER_H))
        # run overlay: the rail rides the header edge and the dot sits in the header, so
        # both read the same on a full card and on a collapsed one.
        self._paint_header_rail(p)
        self._paint_status_dot(p)

        # header text — ELIDED so long titles never run under the switch (G10)
        title_w = (T.NODE_W - 90 if self._switch is None
                   else T.NODE_W - SwitchItem.WIDTH - 34)
        cat_f = QFont(T.SANS, 6); cat_f.setBold(True)
        cat_f.setCapitalization(QFont.AllUppercase)
        cat_f.setLetterSpacing(QFont.PercentageSpacing, 112)
        p.setFont(cat_f)
        p.setPen(T.ACCENT if self._viewed else hdr)
        tag = ("● " if self._viewed else "") + cat + \
              ("  ·  MUTED" if self.rec.muted else "")
        p.drawText(QRectF(12, 5, title_w, 11), Qt.AlignVCenter | Qt.AlignLeft, tag)
        tf = QFont(T.SANS, 9); tf.setBold(True)
        p.setFont(tf)
        p.setPen(T.ERROR if self.dim_invalid() else T.INK)
        # a source node titled with its loaded file name (a group with its group name,
        # else the node-type label)
        label = (self._group_name or self.rec.params.get(TITLE_KEY)
                 or (spec.label if spec else self.rec.op_key))
        label = QFontMetricsF(tf).elidedText(label, Qt.ElideRight, title_w)
        p.drawText(QRectF(12, 15, title_w, 16), Qt.AlignVCenter | Qt.AlignLeft, label)

        if self.rec.collapsed:
            return                     # header-only compact card (sockets at edges)

        # granularity chip (a group instance shows a GROUP badge instead — it is opaque)
        if self._is_group:
            gcol = T.category_color("group")
            foot_txt, chip_txt = "subgraph", "GROUP"
        else:
            gname = self.granularity()
            gcol = T.gran_color(gname)
            foot_txt, chip_txt = "footprint", gname.replace("_", " ").upper()
        p.setFont(QFont(T.SANS, 7))
        p.setPen(T.MUTED)
        p.drawText(QRectF(12, T.HEADER_H, 56, T.GRAN_H - 8), Qt.AlignVCenter, foot_txt)
        cf = QFont(T.MONO, 6); cf.setBold(True)
        p.setFont(cf)
        cw = QFontMetricsF(cf).horizontalAdvance(chip_txt) + 12
        chip = QRectF(64, T.HEADER_H + 3, cw, T.GRAN_H - 14)
        p.setPen(QPen(T.alpha(gcol, 120), 1))
        p.setBrush(T.alpha(gcol, 36))
        p.drawRoundedRect(chip, 4, 4)
        p.setPen(gcol)
        p.drawText(chip, Qt.AlignCenter, chip_txt)
        p.setPen(QPen(T.BORDER, 1, Qt.DashLine))
        p.drawLine(QPointF(12, T.HEADER_H + T.GRAN_H - 4),
                   QPointF(T.NODE_W - 12, T.HEADER_H + T.GRAN_H - 4))

        # rows
        lf = QFont(T.SANS, 8.5)
        for kind, obj, y in self._rows:
            if kind == "in":
                p.setFont(lf); p.setPen(T.INK)
                p.drawText(QRectF(14, y, 120, T.ROW_H), Qt.AlignVCenter | Qt.AlignLeft,
                           obj.name)
                if obj.type is not SocketType.DATASET:
                    self._paint_pill(p, obj, y)
                else:
                    nw = QFontMetricsF(lf).horizontalAdvance(obj.name)
                    self._paint_domain_chips(p, 14 + nw + 10, y, self.reads_domains(),
                                             align_left=True,
                                             missing=self.missing_domains())
            elif kind == "mode":
                p.setFont(lf); p.setPen(T.INK_2)
                p.drawText(QRectF(14, y, 90, T.ROW_H), Qt.AlignVCenter | Qt.AlignLeft,
                           obj.name)
                val = self.rec.modes.get(obj.name, obj.resolved_default())
                self._paint_value_pill(p, y, f"{val} ▾", None, False)
            elif kind == "out":
                p.setFont(lf); p.setPen(T.INK)
                text = obj.label or obj.name          # chK sockets show "K · name"
                p.drawText(QRectF(T.NODE_W - 134, y, 120, T.ROW_H),
                           Qt.AlignVCenter | Qt.AlignRight, text)
                # domain chips only on the combined output; per-channel rows read
                # cleaner with just the channel-tinted socket dot + its name.
                if obj.type is SocketType.DATASET and self.output_channel_index(
                        obj.name) is None:
                    nw = QFontMetricsF(lf).horizontalAdvance(text)
                    self._paint_domain_chips(p, T.NODE_W - 14 - nw - 10, y,
                                             self.out_domains(), align_left=False)

    def _paint_domain_chips(self, p: QPainter, x: float, y: float, domains,
                            *, align_left: bool, missing=frozenset()) -> None:
        """A rail of small colored abbreviation chips (VOX/LBL/PT/…) beside a Dataset
        socket. ``align_left`` flows right from ``x`` (inputs); otherwise the block
        ends at ``x`` (outputs). A missing required domain paints in the error color."""
        if not domains:
            return
        cf = QFont(T.MONO, 6); cf.setBold(True)
        p.setFont(cf)
        fm = QFontMetricsF(cf)
        ch = T.ROW_H - 12
        top = y + 6
        gap = 3.0
        items = [(d, domain_abbr(d), fm.horizontalAdvance(domain_abbr(d)) + 10.0)
                 for d in domains]
        if not align_left:
            total = sum(w for _, _, w in items) + gap * (len(items) - 1)
            x = max(14.0, x - total)
        for d, txt, w in items:
            rect = QRectF(x, top, w, ch)
            miss = d in missing
            col = T.ERROR if miss else T.domain_qcolor(d)
            p.setPen(QPen(col, 1))
            p.setBrush(T.alpha(col, 60 if miss else 46))
            p.drawRoundedRect(rect, 3, 3)
            p.setPen(col)
            p.drawText(rect, Qt.AlignCenter, txt)
            x += w + gap

    def _paint_pill(self, p: QPainter, s, y: float) -> None:
        derived = self.is_derived(s)
        val = self.resolved(s)
        txt = "" if val is None else str(val)
        unit = {"um": "µm", "um_axial": "µm↕", "nm": "nm", "s": "s"}.get(s.unit, s.unit)
        self._paint_value_pill(p, y, txt, unit, derived)

    def _paint_value_pill(self, p: QPainter, y: float, txt: str, unit,
                          derived: bool) -> None:
        vf = QFont(T.MONO, 8)
        fm = QFontMetricsF(vf)
        badge_w = 22 if derived else 0
        uw = (fm.horizontalAdvance(unit) + 5) if unit else 0
        w = fm.horizontalAdvance(txt) + 14 + badge_w + uw
        w = max(w, 34)
        pill = QRectF(T.NODE_W - 12 - w, y + 4, w, T.ROW_H - 8)
        p.setPen(QPen(T.ACCENT_DIM if derived else T.BORDER, 1))
        p.setBrush(T.BODY)
        p.drawRoundedRect(pill, 5, 5)
        x = pill.left() + 6
        if derived:
            bf = QFont(T.MONO, 6); bf.setBold(True)
            p.setFont(bf)
            brect = QRectF(x, pill.top() + 3, 18, pill.height() - 6)
            p.setPen(Qt.NoPen); p.setBrush(T.alpha(T.ACCENT, 30))
            p.drawRoundedRect(brect, 3, 3)
            p.setPen(T.ACCENT)
            p.drawText(brect, Qt.AlignCenter, "ƒmd")
            x += 20
        p.setFont(vf)
        p.setPen(T.INK)
        p.drawText(QRectF(x, pill.top(), w, pill.height()),
                   Qt.AlignVCenter | Qt.AlignLeft, txt)
        if unit:
            p.setPen(T.MUTED)
            p.drawText(pill.adjusted(0, 0, -6, 0), Qt.AlignVCenter | Qt.AlignRight, unit)

    # ── run state (per-node progress) ─────────────────────────────────────────
    def set_run_state(self, state: str, *, fraction: Optional[float] = None,
                      note: str = "", seconds: Optional[float] = None) -> None:
        """Set what this node is doing in the current pull.

        ``state`` is one of ``""`` (idle — nothing painted), ``"queued"``,
        ``"running"``, ``"decoding"`` (its planes are being read — where a lazy chain's
        cost actually lands), ``"cached"`` (validated memo hit), ``"done"`` or
        ``"error"``. ``fraction`` in ``[0,1]`` makes the bar determinate; ``None``
        leaves it an indeterminate sweep. Repaints only when something changed, so the
        runner may call this as often as it likes."""
        frac = None if fraction is None else max(0.0, min(1.0, float(fraction)))
        if (state, frac, note, seconds) == (self._run, self._run_frac, self._run_note,
                                            self._run_secs):
            return
        if state != self._run:
            self._phase = 0.0
        self._run, self._run_frac = state, frac
        self._run_note, self._run_secs = note, seconds
        # the card itself stays graphic (rail + dot); the numbers live in the tooltip and
        # the status bar, so a busy canvas doesn't turn into a wall of tiny text.
        txt = self._run_text()
        label = self._group_name or (self.spec.label if self.spec else self.rec.op_key)
        self.setToolTip(f"{self.rec.id} · {label}\n{txt}" if txt else "")
        self.update()

    def run_state(self) -> str:
        return self._run

    def is_running(self) -> bool:
        """True while this card wants animation frames — the dot pulses whenever the node
        is working, so a determinate rail animates too."""
        return self._run in ("running", "decoding")

    def advance_phase(self, step: float = 0.06) -> None:
        """Move the indeterminate sweep one animation frame (driven by the scene's
        single shared timer — one timer for the canvas, not one per card)."""
        if not self.is_running():
            return
        self._phase = (self._phase + step) % 1.0
        self.update()

    def _run_color(self) -> QColor:
        """One accent for every kind of activity (red only for a failure) — the run
        overlay reads as a single system rather than a traffic light."""
        if self._run == "error":
            return T.ERROR
        if self._run == "queued":
            return T.MUTED
        return T.ACCENT

    def _run_text(self) -> str:
        if self._run == "queued":
            return "queued"
        if self._run == "cached":
            return "cached"
        if self._run == "error":
            return "error"
        if self._run == "decoding":
            return "reading"
        if self._run == "running":
            # the percentage only — the note ("t=3 z=1") is long and would collide with
            # the footprint chip; the status bar carries it instead.
            if self._run_frac is not None:
                return f"{int(round(self._run_frac * 100))}%"
            return "running"
        if self._run == "done" and self._run_secs is not None:
            s = self._run_secs
            return f"{s * 1000:.0f} ms" if s < 1.0 else f"{s:.2f} s"
        return ""

    @staticmethod
    def _glow_pill(p: QPainter, rect: QRectF, col: QColor) -> None:
        """A rounded bar with a soft halo. QPainter has no ``box-shadow``, so the glow is
        two translucent passes grown around the bar — the same read as the mockup's
        ``0 0 8px accent`` without an (expensive, blurry) graphics effect."""
        p.setPen(Qt.NoPen)
        for grow, a in ((2.4, 26), (1.1, 58)):
            r = rect.adjusted(-grow, -grow, grow, grow)
            p.setBrush(T.alpha(col, a))
            p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setBrush(col)
        p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

    def _paint_card_glow(self, p: QPainter, rect: QRectF, col: QColor) -> None:
        """The card's outer glow while it is selected / running / failed."""
        p.setBrush(Qt.NoBrush)
        for grow, a in ((3.0, 20), (1.4, 46)):
            p.setPen(QPen(T.alpha(col, a), 1.6))
            p.drawRoundedRect(rect.adjusted(-grow, -grow, grow, grow),
                              T.RADIUS + grow, T.RADIUS + grow)

    def _paint_header_rail(self, p: QPainter) -> None:
        """The progress rail riding the header's bottom edge — **live work only**: a
        determinate fill while the compute reports ``ctx.progress``, a sweeping segment
        while it does not. Terminal states (queued / done / cached / error) carry no rail;
        they read from the dot, so a finished graph stays calm instead of being striped
        with full bars."""
        if self._run not in ("running", "decoding"):
            return
        w = T.NODE_W - 2.0
        y = T.HEADER_H - T.PROG_H
        col = self._run_color()
        if self._run_frac is not None:
            if self._run_frac <= 0:
                return
            self._glow_pill(p, QRectF(1.0, y, max(2.0, w * self._run_frac), T.PROG_H), col)
            return
        # indeterminate: a segment ping-ponging inside the rail. It stays FULLY inside
        # (never clipped to nothing at the turnaround), so a card that is working always
        # looks like it is working — even in a single frame, e.g. a screenshot.
        seg = max(22.0, w * 0.28)
        tri = self._phase * 2.0
        f = tri if tri <= 1.0 else 2.0 - tri
        self._glow_pill(p, QRectF(1.0 + (w - seg) * f, y, seg, T.PROG_H), col)

    def _dot_center(self) -> QPointF:
        """Where the header status dot sits: the header's right edge, stepped left of the
        2D/3D switch on the nodes that carry one."""
        x = (self._switch.pos().x() - 9.0 if self._switch is not None
             else T.NODE_W - 13.0)
        return QPointF(x, T.HEADER_H / 2.0)

    def _paint_status_dot(self, p: QPainter) -> None:
        """The header dot: a pulsing accent bead while the node works, a solid bead once
        it produced, a hollow ring when it was *reused* (cached) or merely enlisted
        (queued), red when it raised."""
        if not self._run:
            return
        c, r, col = self._dot_center(), T.DOT_R, self._run_color()
        if self._run in ("running", "decoding"):
            # pulse off the shared animation phase (the mockup's ~0.7 s blink)
            col = T.alpha(col, 96 + int(159 * abs(1.0 - 2.0 * ((self._phase * 1.7) % 1.0))))
        if self._run in ("queued", "cached"):        # nothing was computed here
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, 1.4))
            p.drawEllipse(c, r, r)
            return
        p.setPen(Qt.NoPen)
        for grow, a in ((3.4, 28), (1.8, 62)):
            p.setBrush(T.alpha(col, a))
            p.drawEllipse(c, r + grow, r + grow)
        p.setBrush(col)
        p.drawEllipse(c, r, r)

    # ── events ────────────────────────────────────────────────────────────────
    def _place_close(self) -> None:
        """Pin the ✕ badge to the card's top-right corner — mostly inside it (so the
        pointer stays over the card on the way to the badge), just clear of the 2D/3D
        switch and the elided title."""
        w = T.RR_SIZE if self._is_reroute else T.NODE_W
        self._close.setPos(w - T.CLOSE_BTN + 4, -4)

    def hide_close_if_away(self) -> None:
        """Hide the ✕ badge once the pointer is over neither the card nor the badge.
        Deferred by a beat because Qt delivers the card's leave and the badge's enter in
        an unspecified order — hiding eagerly would yank the badge out from under a
        pointer that is on its way to click it."""
        def check() -> None:
            if self.scene() is None:
                return
            if not (self.isUnderMouse() or self._close.isUnderMouse()):
                self._close.setVisible(False)
        QTimer.singleShot(80, check)

    def hoverEnterEvent(self, e) -> None:
        self._close.setVisible(True)
        super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e) -> None:
        self.hide_close_if_away()
        super().hoverLeaveEvent(e)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.doc.set_pos(self.rec.id, self.pos().x(), self.pos().y())
            if self.scene() is not None and hasattr(self.scene(), "reroute"):
                self.scene().reroute()
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.update()
        return super().itemChange(change, value)


__all__ = ["SocketItem", "SwitchItem", "CloseItem", "NodeItem"]
