"""The GraphDocument — NodeLab v2's Qt-free editing model (Phase 5).

The canvas is a *view*; this is the source of truth: node records (op_key + params +
modes, the same dicts the graphics items and inspector mutate in place), edges, and
GUI-only extras (canvas position, mute). It builds a real
:class:`nodegraph.graph.Graph` on demand, runs the **edit-time MetaEnvelope pass**
(:func:`nodegraph.metadata.propagate_meta`) after every structural edit — the G8 "live
widget re-seed" — and round-trips ``*.nd2graph.json`` through
:mod:`nodegraph.serialize` (GUI extras ride in a top-level ``ui`` object the headless
loader ignores).

Wiring rules (G1): a connection must pass :func:`nodegraph.sockets.can_connect` on the
two sockets' *active* specs, must not create a cycle, and a second wire into a
non-multi input **replaces** the existing one (Blender behavior). Mute (G3) is resolved
at graph-build time for runs: a muted node with a Dataset input is bypassed
(pass-through), so the engine never sees it.

Qt-free; standard library + nodegraph only (testable headless).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from nodegraph.domains import AXIS_ORDER, Domain
from nodegraph.graph import Edge, Graph, NodeInstance
from nodegraph.groups import (
    GROUP_INPUT, GROUP_OUTPUT, Group, expand as _group_expand, group_name_of,
    inst_id as _inst_id,
)
from nodegraph.metadata import MetaEnvelope, propagate_meta
from nodegraph.registry import InDataset, NODES, OutDataset
from nodegraph.serialize import from_dict as _ng_from_dict, to_dict as _ng_to_dict
from nodegraph.sockets import Direction, SocketType, can_connect as _can_connect
from nodegraph.zones import Zone, unroll as _unroll
from nodelab_v2.ops import materialize_channel_taps

#: params key holding the sticky pinned-override list (serialized per V2.03; stripped
#: from the params handed to the ENGINE — it is a UI annotation, not a compute input).
LOCKED_KEY = "__locked__"

#: per-instance UI annotations (serialized, stripped from the params handed to the ENGINE
#: exactly like LOCKED_KEY): a source node's file-name display title, and the captured
#: per-channel descriptor list ``[{name, emission_nm, color}, …]`` that drives its
#: synthetic per-channel output sockets and their wire tints.
TITLE_KEY = "__title__"
CHANNELS_KEY = "__channels__"

#: op_keys whose GUI card grows one synthetic per-channel output socket (``ch0…``) per
#: channel — materialized into ``channel.select`` taps at graph-build (see nodelab_v2.ops).
CHANNEL_TAP_OPS = ("io.load", "channel.split")

#: params keys that are UI-only and must be stripped before the graph runs.
_UI_PARAM_KEYS = (LOCKED_KEY, TITLE_KEY, CHANNELS_KEY)


class NodeRecord:
    """One placed node. ``params``/``modes`` are the LIVE dicts the GUI mutates."""

    __slots__ = ("id", "op_key", "params", "modes", "x", "y", "muted", "collapsed")

    def __init__(self, node_id: str, op_key: str, *,
                 params: Optional[dict] = None, modes: Optional[dict] = None,
                 x: float = 0.0, y: float = 0.0, muted: bool = False,
                 collapsed: bool = False) -> None:
        self.id = node_id
        self.op_key = op_key
        self.params: Dict[str, Any] = dict(params or {})
        self.modes: Dict[str, str] = dict(modes or {})
        self.x, self.y = float(x), float(y)
        self.muted = bool(muted)
        self.collapsed = bool(collapsed)

    def spec(self):
        return NODES.get(self.op_key)

    def state(self) -> Dict[str, str]:
        spec = self.spec()
        base = spec.default_state() if spec is not None else {}
        base.update(self.modes)
        return base

    @property
    def locked(self) -> set:
        return set(self.params.get(LOCKED_KEY, ()))

    def set_locked(self, names: set) -> None:
        if names:
            self.params[LOCKED_KEY] = sorted(names)
        else:
            self.params.pop(LOCKED_KEY, None)


class FrameRecord:
    """A labelled frame grouping a set of nodes on the canvas — **GUI-only** (it never
    enters the run graph; it rides in the ``ui`` extras like node positions). A frame
    auto-sizes to enclose its member nodes; it always has ≥1 member (an emptied frame is
    removed), so it needs no stored geometry."""

    __slots__ = ("id", "title", "members", "color")

    def __init__(self, frame_id: str, title: str = "Frame",
                 members=(), color=None) -> None:
        self.id = frame_id
        self.title = str(title)
        self.members: List[str] = list(members)
        self.color = tuple(int(v) for v in color) if color else None


EdgeTuple = Tuple[str, str, str, str]        # (src, src_socket, dst, dst_socket)


class GraphDocument:
    """The editable model + envelope cache. ``on_change`` callbacks fire after every
    structural edit or re-propagation (the canvas/inspector re-seed from them)."""

    def __init__(self) -> None:
        self.nodes: Dict[str, NodeRecord] = {}
        self.edges: List[EdgeTuple] = []
        self.meta_seeds: Dict[str, MetaEnvelope] = {}
        self.envs: Dict[str, MetaEnvelope] = {}
        self.path: Optional[str] = None           # last save/load file
        self.revision = 0                          # bumped on every edit
        # zones/groups + zone back-edges aren't GUI-editable yet, but a loaded file's
        # are carried through VERBATIM so re-saving never silently destroys them
        # (review 2026-07-22 BLOCKER). The GUI reads/writes only the FORWARD nodes+
        # edges layer; `self.edges` holds forward 4-tuples the canvas draws, while a
        # loaded back-edge (Out→In zone feedback) rides in `_back_edges` untouched.
        self._zones: list = []
        self._groups: list = []
        self._back_edges: List[Edge] = []
        # GUI-only labelled frames (canvas organization; never enter the run graph —
        # they ride in the `ui` extras exactly like node positions).
        self.frames: Dict[str, FrameRecord] = {}
        self._listeners: List[Callable[[], None]] = []

    # ── listeners ────────────────────────────────────────────────────────────
    def on_change(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def _notify(self) -> None:
        self.revision += 1
        self.propagate()
        for fn in list(self._listeners):
            fn()

    # ── node ops ─────────────────────────────────────────────────────────────
    def new_id(self, prefix: str = "n") -> str:
        i = len(self.nodes) + 1
        while f"{prefix}{i}" in self.nodes:
            i += 1
        return f"{prefix}{i}"

    def add_node(self, op_key: str, *, x: float = 0.0, y: float = 0.0,
                 node_id: Optional[str] = None, params: Optional[dict] = None,
                 modes: Optional[dict] = None) -> NodeRecord:
        nid = node_id or self.new_id()
        if nid in self.nodes:
            raise ValueError(f"duplicate node id {nid!r}")
        rec = NodeRecord(nid, op_key, params=params, modes=modes, x=x, y=y)
        self.nodes[nid] = rec
        self._notify()
        return rec

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            return
        op_key = self.nodes[node_id].op_key
        self.edges = [e for e in self.edges if e[0] != node_id and e[2] != node_id]
        del self.nodes[node_id]
        self.meta_seeds.pop(node_id, None)
        self._prune_frames(node_id)
        self._drop_orphan_group(op_key)          # a removed instance drops its unused def
        self._notify()

    def _drop_orphan_group(self, op_key: str) -> None:
        """If ``op_key`` is a group-instance op_key with no remaining instances, drop the
        group definition (kept while any instance still references it)."""
        name = group_name_of(op_key)
        if name and not any(group_name_of(r.op_key) == name for r in self.nodes.values()):
            self._groups = [g for g in self._groups if g.name != name]

    def touch(self) -> None:
        """Signal a param/mode edit (values live in shared dicts — no copy needed)."""
        self._notify()

    def clear(self) -> None:
        """Empty the document (File → New)."""
        self.nodes.clear()
        self.edges = []
        self.meta_seeds.clear()
        self._zones = []
        self._groups = []
        self._back_edges = []
        self.frames = {}
        self.path = None
        self._notify()

    # ── frames (GUI-only canvas grouping) ──────────────────────────────────────
    def new_frame_id(self) -> str:
        i = len(self.frames) + 1
        while f"f{i}" in self.frames:
            i += 1
        return f"f{i}"

    def add_frame(self, title: str = "Frame", members=(), color=None,
                  frame_id: Optional[str] = None) -> FrameRecord:
        """Create a labelled frame around ``members`` (only the ids that exist are
        kept). A frame must enclose ≥1 node — an empty selection raises."""
        mem = [n for n in members if n in self.nodes]
        if not mem:
            raise ValueError("select one or more nodes to frame")
        fid = frame_id or self.new_frame_id()
        if fid in self.frames:
            raise ValueError(f"duplicate frame id {fid!r}")
        self.frames[fid] = FrameRecord(fid, title, mem, color)
        self._notify()
        return self.frames[fid]

    def remove_frame(self, frame_id: str) -> None:
        if frame_id in self.frames:
            del self.frames[frame_id]
            self._notify()

    def rename_frame(self, frame_id: str, title: str) -> None:
        fr = self.frames.get(frame_id)
        if fr is not None:
            fr.title = str(title)
            self._notify()

    def _prune_frames(self, node_id: str) -> None:
        """Drop a removed node from every frame; a frame left with no members is
        removed (frames never persist empty). Does NOT notify (the caller does)."""
        for fid in list(self.frames):
            fr = self.frames[fid]
            if node_id in fr.members:
                fr.members = [n for n in fr.members if n != node_id]
                if not fr.members:
                    del self.frames[fid]

    def set_pos(self, node_id: str, x: float, y: float) -> None:
        rec = self.nodes.get(node_id)
        if rec is not None:
            rec.x, rec.y = float(x), float(y)   # position is not a model edit: no notify

    def set_muted(self, node_id: str, muted: bool) -> None:
        rec = self.nodes.get(node_id)
        if rec is not None and rec.muted != muted:
            rec.muted = muted
            self._notify()

    def set_collapsed(self, node_id: str, collapsed: bool) -> None:
        rec = self.nodes.get(node_id)
        if rec is not None and rec.collapsed != collapsed:
            rec.collapsed = collapsed
            self._notify()

    # ── instance-aware socket resolution (per-channel outputs) ─────────────────
    def output_specs(self, node_id: str) -> list:
        """The live OUTPUT socket specs of one node INSTANCE: the type's active outputs
        plus, for a ``io.load``/``channel.split`` node with ≥2 channels, one synthetic
        ``ch{K}`` :class:`OutDataset` per channel (labelled by the channel name). These
        synthetic sockets are materialized into ``channel.select`` taps at graph-build —
        here they only need to exist so the canvas can lay them out and validate wires."""
        rec = self.nodes.get(node_id)
        if rec is not None and group_name_of(rec.op_key):
            return [OutDataset()]                 # a group instance: one Dataset output
        spec = rec.spec() if rec else None
        if spec is None:
            return []
        base = list(spec.active_outputs(rec.state()))
        if rec.op_key in CHANNEL_TAP_OPS:
            descs = self.channel_descriptors(node_id)
            if len(descs) >= 2:
                for i, ch in enumerate(descs):
                    base.append(OutDataset(f"ch{i}",
                                           label=f"{i} · {ch.get('name') or f'Ch{i}'}"))
        return base

    def input_specs(self, node_id: str) -> list:
        """The live INPUT socket specs of one node instance (no per-channel expansion —
        provided for symmetry with :meth:`output_specs`)."""
        rec = self.nodes.get(node_id)
        if rec is not None and group_name_of(rec.op_key):
            return [InDataset()]                  # a group instance: one Dataset input
        spec = rec.spec() if rec else None
        return list(spec.active_inputs(rec.state())) if spec else []

    def channel_descriptors(self, node_id: str) -> list:
        """The per-channel descriptors ``[{name, emission_nm, color}, …]`` for a source /
        split node: the captured ``__channels__`` list (Load, richest — real names +
        native colors) if present, else derived from the node's own propagated envelope
        (Split passes its input env through unchanged, so its channel count/emission are
        already correct)."""
        rec = self.nodes.get(node_id)
        if rec is None:
            return []
        chans = rec.params.get(CHANNELS_KEY)
        if isinstance(chans, list) and chans:
            return chans
        return self._env_channel_descriptors(self.env(node_id))

    @staticmethod
    def _env_channel_descriptors(env: MetaEnvelope) -> list:
        c = env.axes.c
        if c <= 0 or "c" in env.unknown_axes:
            return []                               # channel count not yet known
        emis = env.metadata.get("channel_emission_nm")
        names = env.metadata.get("channel_names")
        out = []
        for i in range(c):
            out.append({
                "name": (names[i] if isinstance(names, (list, tuple)) and i < len(names)
                         else f"Ch{i}"),
                "emission_nm": (emis[i] if isinstance(emis, (list, tuple))
                                and i < len(emis) else None),
                "color": None,
            })
        return out

    def source_channel_total(self, node_id: str) -> int:
        """The total channel count of the source file(s) feeding ``node_id`` — used by
        the wire tint to decide whether a wire carries a strict channel subset. Walks up
        to the source roots and takes the max (a root ``io.load`` reports its captured
        ``__channels__`` length, else its seeded envelope's ``c``)."""
        seen: set = set()
        stack = [node_id]
        totals = []
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            preds = [e for e in self.edges if e[2] == nid]
            if preds:
                stack.extend(e[0] for e in preds)
            else:
                rec = self.nodes.get(nid)
                chans = rec.params.get(CHANNELS_KEY) if rec else None
                totals.append(len(chans) if isinstance(chans, list) and chans
                              else self.env(nid).axes.c)
        return max(totals) if totals else self.env(node_id).axes.c

    # ── wiring (G1) ──────────────────────────────────────────────────────────
    def _socket_spec(self, node_id: str, io: str, name: str):
        rec = self.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None:
            return None
        pool = self.input_specs(node_id) if io == "in" else self.output_specs(node_id)
        return next((s for s in pool if s.name == name), None)

    def can_connect(self, src: str, src_socket: str, dst: str, dst_socket: str
                    ) -> Tuple[bool, str]:
        """(ok, reason). Validates direction/type via ``sockets.can_connect`` on the
        ACTIVE socket specs, self-loops, and cycles."""
        if src == dst:
            return False, "self-loop"
        a = self._socket_spec(src, "out", src_socket)
        b = self._socket_spec(dst, "in", dst_socket)
        if a is None or b is None:
            return False, "unknown or inactive socket"
        sa, sb = a.instantiate(), b.instantiate()
        if sa.direction is not Direction.OUT or sb.direction is not Direction.IN:
            return False, "direction"
        if not _can_connect(sa, sb):
            return False, f"{sa.type.value} → {sb.type.value} is not connectable"
        if self._creates_cycle(src, dst):
            return False, "would create a cycle (rejected outside zones)"
        return True, ""

    def _creates_cycle(self, src: str, dst: str) -> bool:
        """True if adding dst←src closes a cycle (src reachable FROM dst)."""
        stack, seen = [dst], set()
        while stack:
            nid = stack.pop()
            if nid == src:
                return True
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(e[2] for e in self.edges if e[0] == nid)
        return False

    def connect(self, src: str, src_socket: str, dst: str, dst_socket: str
                ) -> List[EdgeTuple]:
        """Add the wire (validated). A non-multi input's existing wire is REPLACED.
        Returns the list of edges removed by the replacement."""
        ok, reason = self.can_connect(src, src_socket, dst, dst_socket)
        if not ok:
            raise ValueError(f"cannot connect: {reason}")
        removed: List[EdgeTuple] = []
        b = self._socket_spec(dst, "in", dst_socket)
        if b is not None and not b.multi:
            removed = [e for e in self.edges if e[2] == dst and e[3] == dst_socket]
            for e in removed:
                self.edges.remove(e)
        edge = (src, src_socket, dst, dst_socket)
        if edge not in self.edges:
            self.edges.append(edge)
        self._notify()
        return removed

    def disconnect(self, src: str, src_socket: str, dst: str, dst_socket: str) -> None:
        e = (src, src_socket, dst, dst_socket)
        if e in self.edges:
            self.edges.remove(e)
            self._notify()

    def edge_into(self, dst: str, dst_socket: str) -> Optional[EdgeTuple]:
        return next((e for e in self.edges if e[2] == dst and e[3] == dst_socket), None)

    # ── graph / envelopes ────────────────────────────────────────────────────
    def to_graph(self, *, for_run: bool = False, materialize: bool = False) -> Graph:
        """The headless :class:`Graph`. ``for_run`` strips the UI-only param annotations
        (``__locked__``/``__title__``/``__channels__`` — they must not re-key the memo)
        and resolves MUTED nodes by bypassing them (first Dataset input → their
        consumers). ``materialize`` rewrites each synthetic per-channel output edge
        (``chK``) into a real ``channel.select`` tap AND inlines every group instance
        (``group:<name>``) into its body via :func:`nodegraph.groups.expand` — used for
        the run graph and the edit-time envelope pass, but NOT for saving (the file keeps
        the instance nodes + separate group definitions)."""
        g = Graph()
        for rec in self.nodes.values():
            params = dict(rec.params)
            if for_run:
                for k in _UI_PARAM_KEYS:
                    params.pop(k, None)
            g.add(NodeInstance(rec.id, rec.op_key, params=params,
                               modes=dict(rec.modes)))
        edges = list(self.edges)
        if for_run:
            edges = self._bypass_muted(edges)
        for (s, ss, d, ds) in edges:
            g.connect(s, d, src_socket=ss, dst_socket=ds)
        for e in self._back_edges:            # preserved zone-feedback edges verbatim
            g.connect(e.src, e.dst, src_socket=e.src_socket,
                      dst_socket=e.dst_socket, kind=e.kind)
        if materialize and self._groups:      # inline group instances into their bodies
            g = _group_expand(g, self._groups)
        return materialize_channel_taps(g) if materialize else g

    def _bypass_muted(self, edges: List[EdgeTuple]) -> List[EdgeTuple]:
        """Mute-with-passthrough (G3): rewire around each muted node via its first
        connected Dataset input; a muted SOURCE simply drops its out-edges."""
        for rec in self.nodes.values():
            if not rec.muted:
                continue
            ds_ins = [s.name for s in self.input_specs(rec.id)
                      if s.type is SocketType.DATASET]
            feed = next((e for e in edges if e[2] == rec.id and e[3] in ds_ins), None)
            outs = [e for e in edges if e[0] == rec.id]
            edges = [e for e in edges if e[0] != rec.id and e[2] != rec.id]
            if feed is not None:
                for (_, _, d, ds) in outs:
                    edges.append((feed[0], feed[1], d, ds))
        return edges

    # ── zone creation (Repeat) ─────────────────────────────────────────────────
    def _dataset_inputs(self, node_id: str) -> set:
        # via input_specs so a group instance's synthetic Dataset input resolves too
        return {s.name for s in self.input_specs(node_id)
                if s.type is SocketType.DATASET}

    def wrap_repeat_zone(self, node_ids, iterations: int = 3) -> str:
        """Wrap a selected **linear** sub-chain in a Repeat zone: insert paired
        ``zone.repeat_in``/``zone.repeat_out`` boundary nodes, rewire the single
        dataset frontier in/out through them, add the ``Out→In`` back-edge, and record
        a :class:`~nodegraph.zones.Zone`. The result is validated by
        :func:`nodegraph.zones.unroll` before commit — a malformed selection raises
        ``ValueError`` and leaves the document untouched. Group creation and Sim/nested
        zones are a later phase (they need a distinct UX / eval design)."""
        sel = set(node_ids)
        if not sel:
            raise ValueError("select the nodes to wrap in a Repeat zone")
        if not sel <= set(self.nodes):
            raise ValueError("selection includes unknown nodes")
        if any(nid in z.members for z in self._zones for nid in sel):
            raise ValueError("a selected node is already in a zone (nested zones are a "
                             "later phase)")
        if any(self.nodes[nid].op_key.startswith(("zone.", "group."))
               for nid in sel):
            raise ValueError("don't wrap zone/group boundary nodes")
        # the dataset frontier: edges crossing the selection boundary
        ext_in = [e for e in self.edges
                  if e[2] in sel and e[0] not in sel and e[3] in self._dataset_inputs(e[2])]
        ext_out = [e for e in self.edges if e[0] in sel and e[2] not in sel]
        if len(ext_in) != 1 or len(ext_out) != 1:
            raise ValueError(
                "select a single linear sub-chain with exactly one dataset input and "
                f"one output (found {len(ext_in)} in, {len(ext_out)} out)")

        snapshot = (dict(self.nodes), list(self.edges), list(self._back_edges),
                    list(self._zones))
        try:
            (es, ess, ed, eds) = ext_in[0]
            (xs, xss, xd, xds) = ext_out[0]
            rin, rout = self.new_id("rin"), self.new_id("rout")
            # position the boundary nodes just outside the chain's span
            xs_min = min(self.nodes[n].x for n in sel)
            xs_max = max(self.nodes[n].x for n in sel)
            ymid = sum(self.nodes[n].y for n in sel) / len(sel)
            self.nodes[rin] = NodeRecord(rin, "zone.repeat_in", x=xs_min - 150, y=ymid)
            self.nodes[rout] = NodeRecord(rout, "zone.repeat_out", x=xs_max + 230, y=ymid)
            self.edges.remove(ext_in[0])
            self.edges.remove(ext_out[0])
            self.edges += [
                (es, ess, rin, "data"), (rin, "out", ed, eds),
                (xs, xss, rout, "data"), (rout, "out", xd, xds)]
            self._back_edges.append(Edge(rout, rin, "out", "data", "back"))
            zid = self.new_id("z")
            self._zones.append(Zone(zid, "repeat", rin, rout,
                                    body=frozenset(sel), iterations=max(1, int(iterations))))
            _unroll(self.to_graph(for_run=False), self._zones)   # validate — raises if bad
        except Exception:
            (self.nodes, self.edges, self._back_edges, self._zones) = (
                dict(snapshot[0]), list(snapshot[1]), list(snapshot[2]), list(snapshot[3]))
            raise
        self._notify()
        return zid

    # ── group creation / ungrouping ────────────────────────────────────────────
    def _unique_group_name(self, base: str) -> str:
        existing = {g.name for g in self._groups}
        name = (base or "Group").strip() or "Group"
        if name not in existing:
            return name
        i = 2
        while f"{name} {i}" in existing:
            i += 1
        return f"{name} {i}"

    def make_group(self, node_ids, name: str = "Group") -> str:
        """Collapse a selected **linear** sub-chain into one reusable group instance node
        (Blender's 'Make Group'): the selection becomes a :class:`~nodegraph.groups.Group`
        DEFINITION (its body, bounded by ``group.input``/``group.output`` boundaries) and
        is replaced in the parent by a single ``group:<name>`` instance node wired to the
        same single dataset frontier. It runs / propagates by inlining the body
        (``to_graph(materialize)`` → :func:`nodegraph.groups.expand`). Validated by a trial
        expand before commit — a malformed selection raises and leaves the document
        untouched. Reverse with :meth:`ungroup`."""
        sel = set(node_ids)
        if not sel:
            raise ValueError("select the nodes to group")
        if not sel <= set(self.nodes):
            raise ValueError("selection includes unknown nodes")
        if any(self.nodes[nid].op_key.startswith(("zone.", "group."))
               or group_name_of(self.nodes[nid].op_key) for nid in sel):
            raise ValueError("don't group zone/group boundary or group-instance nodes")
        if any(nid in z.members for z in self._zones for nid in sel):
            raise ValueError("a selected node is in a zone (group-in-zone is a later phase)")
        if any(not any(e[2] == nid for e in self.edges) for nid in sel):
            raise ValueError("don't group a source node (its seed would be buried); leave "
                             "sources outside the group")
        ext_in = [e for e in self.edges if e[2] in sel and e[0] not in sel
                  and e[3] in self._dataset_inputs(e[2])]
        ext_out = [e for e in self.edges if e[0] in sel and e[2] not in sel]
        if len(ext_in) != 1 or len(ext_out) != 1:
            raise ValueError(
                "select a single linear sub-chain with exactly one dataset input and one "
                f"output (found {len(ext_in)} in, {len(ext_out)} out)")

        snapshot = (dict(self.nodes), list(self.edges), list(self._groups))
        try:
            gname = self._unique_group_name(name)
            (es, ess, ed, eds) = ext_in[0]         # external src → interior dst(eds)
            (xs, xss, xd, xds) = ext_out[0]        # interior src(xss) → external dst
            body = Graph()
            for nid in sel:
                r = self.nodes[nid]
                body.add(NodeInstance(nid, r.op_key, params=dict(r.params),
                                      modes=dict(r.modes)))
            gin, gout = "grp.in", "grp.out"        # boundary ids (unique within the body)
            body.add(NodeInstance(gin, GROUP_INPUT))
            body.add(NodeInstance(gout, GROUP_OUTPUT))
            for (s, ss, d, ds) in self.edges:      # interior edges (both ends inside)
                if s in sel and d in sel:
                    body.connect(s, d, src_socket=ss, dst_socket=ds)
            body.connect(gin, ed, src_socket="out", dst_socket=eds)    # input → interior
            body.connect(xs, gout, src_socket=xss, dst_socket="data")  # interior → output
            grp = Group(gname, body, gin, gout)
            inst = self.new_id("grp")
            cx = sum(self.nodes[n].x for n in sel) / len(sel)
            cy = sum(self.nodes[n].y for n in sel) / len(sel)
            for nid in sel:
                del self.nodes[nid]
                self.meta_seeds.pop(nid, None)
                self._prune_frames(nid)
            self.edges = [e for e in self.edges if e[0] not in sel and e[2] not in sel]
            self.nodes[inst] = NodeRecord(inst, grp.op_key, x=cx, y=cy)
            self.edges.append((es, ess, inst, "data"))
            self.edges.append((inst, "out", xd, xds))
            self._groups.append(grp)
            _group_expand(self.to_graph(for_run=False), self._groups)   # validate — raises
        except Exception:
            (self.nodes, self.edges, self._groups) = (
                dict(snapshot[0]), list(snapshot[1]), list(snapshot[2]))
            raise
        self._notify()
        return inst

    def ungroup(self, inst_id: str) -> bool:
        """Inline a group instance back into the parent as real nodes (the reverse of
        :meth:`make_group`): the body interior is restored with FRESH collision-free ids,
        its internal edges + the single dataset frontier are reconnected, and the instance
        + its now-unused group definition are removed. Returns ``False`` (no-op) if
        ``inst_id`` is not a group instance."""
        rec = self.nodes.get(inst_id)
        name = group_name_of(rec.op_key) if rec is not None else None
        grp = self._group_by_name(name) if name else None
        if grp is None:
            return False
        in_edge = next((e for e in self.edges if e[2] == inst_id), None)   # ext → inst.data
        out_edges = [e for e in self.edges if e[0] == inst_id]             # inst.out → ext
        idmap: Dict[str, str] = {}
        for k, bnid in enumerate(sorted(grp.interior)):
            bnode = grp.body.nodes[bnid]
            nid = self.new_id()
            idmap[bnid] = nid
            self.nodes[nid] = NodeRecord(nid, bnode.op_key, params=dict(bnode.params),
                                         modes=dict(bnode.modes),
                                         x=rec.x + k * 40, y=rec.y + k * 30)
        for e in grp.body.edges:                   # interior edges (no boundary endpoint)
            if e.src in (grp.input_id, grp.output_id) or \
                    e.dst in (grp.input_id, grp.output_id):
                continue
            si, di = idmap.get(e.src), idmap.get(e.dst)
            if si and di:
                self.edges.append((si, e.src_socket, di, e.dst_socket))
        for e in grp.body.edges:                   # boundary edges → external frontier
            if e.src == grp.input_id and in_edge is not None:
                di = idmap.get(e.dst)
                if di:
                    self.edges.append((in_edge[0], in_edge[1], di, e.dst_socket))
            if e.dst == grp.output_id:
                si = idmap.get(e.src)
                if si:
                    for (_s, _ss, xd, xds) in out_edges:
                        self.edges.append((si, e.src_socket, xd, xds))
        self.edges = [e for e in self.edges if e[0] != inst_id and e[2] != inst_id]
        del self.nodes[inst_id]
        self._prune_frames(inst_id)
        self._groups = [g for g in self._groups if g.name != name]
        self._notify()
        return True

    def propagate(self) -> None:
        """Re-run the edit-time MetaEnvelope pass (G8 live re-seed). Sources without
        a seed get an ALL-UNKNOWN envelope — unknown is not z==1, so the H11 lever
        guard never greys 3D just because a source hasn't resolved yet."""
        try:
            # to_graph(materialize) inlines group instances; a malformed group (e.g. a
            # loaded file referencing a missing definition) raises here too — swallow it
            # like a mid-edit cycle so a single bad state never crashes the live re-seed.
            g = self.to_graph(materialize=True)
            seeds = dict(self.meta_seeds)
            unknown = MetaEnvelope(unknown_axes=frozenset(AXIS_ORDER))
            for nid in g.roots():
                seeds.setdefault(nid, unknown)
            self.envs = propagate_meta(g, seeds)
        except ValueError:
            self.envs = {}                        # a mid-edit cycle / bad group: no envelopes
        # a group instance is inlined in the expanded graph, so it has no envelope of its
        # own — give it its body's OUTPUT envelope so its card + every downstream node's
        # domain rail read correctly THROUGH the opaque instance.
        for nid, rec in self.nodes.items():
            gname = group_name_of(rec.op_key)
            if not gname:
                continue
            grp = self._group_by_name(gname)
            if grp is not None:
                out_env = self.envs.get(_inst_id(grp.output_id, nid))
                if out_env is not None:
                    self.envs[nid] = out_env

    def _group_by_name(self, name: str) -> Optional[Group]:
        return next((g for g in self._groups if g.name == name), None)

    def env(self, node_id: str) -> MetaEnvelope:
        return self.envs.get(node_id, MetaEnvelope())

    # ── layer picker (V2.11) ───────────────────────────────────────────────────
    def layer_choices(self, node_id: str, sock) -> list:
        """The layer names a ``layer_in`` socket should offer — those present on the
        edge feeding this node, in the domain the socket declares.

        Deliberately NOT modelled on :meth:`input_domains`, which UNIONS every Dataset
        predecessor. Three nodes take a second Dataset input whose layers must never be
        offered here (``analysis.measure``'s ``raw``, and the ``reference`` of
        ``dvc_field``/``dic_correlate``): the payload flows from the PRIMARY input, and
        ``propagate_meta`` agrees — it builds the envelope from ``dataset_preds[0]``
        alone. So follow the primary edge only.

        Never falls back to this node's OWN envelope: that already contains the layers
        this node writes, so a source picker would offer the node its own output."""
        if sock is None:
            return []
        domain = sock.layer_in
        if not domain and sock.layer_in_mode:
            rec = self.nodes.get(node_id)
            spec = rec.spec() if rec else None
            value = rec.state().get(sock.layer_in_mode) if (rec and spec) else None
            try:
                domain = Domain(value) if value else None
            except ValueError:                      # a mode value that is not a Domain
                domain = None
        if domain is None:
            return []
        rec = self.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None:
            return []
        primary = next((s.name for s in spec.inputs
                        if s.type is SocketType.DATASET), None)
        if primary is None:
            return []
        edge = self.edge_into(node_id, primary)
        if edge is None:
            return []
        return list(self.env(edge[0]).layers_in(domain))

    # ── domain interface (socket rail + wire tint + validation) ────────────────
    def input_domains(self, node_id: str) -> frozenset:
        """The accumulated domain-set arriving on ``node_id``'s Dataset input(s) —
        the union of every Dataset-predecessor's output domain-set. Empty for a
        source (its domains come from its own ``adds_domains``)."""
        rec = self.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None:
            return frozenset()
        ds_ins = {s.name for s in spec.inputs if s.type is SocketType.DATASET}
        doms: frozenset = frozenset()
        for src, _ssock, dst, dsock in self.edges:
            if dst == node_id and dsock in ds_ins:
                doms = doms | self.env(src).domains
        return doms

    def missing_domains(self, node_id: str) -> frozenset:
        """Required domains (``reads_domains``) absent from the upstream set — the
        GUI's red validation chips (e.g. a Measure node with no Label upstream)."""
        rec = self.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None:
            return frozenset()
        return spec.missing_domains(self.input_domains(node_id))

    def set_meta_seed(self, node_id: str, env: MetaEnvelope) -> None:
        self.meta_seeds[node_id] = env
        self._notify()

    # ── save / load (G6) ─────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        # carry any loaded zones/groups through unchanged — the GUI edits only the
        # nodes+edges layer, but must never DROP structure it can't yet edit.
        d = _ng_to_dict(self.to_graph(), zones=self._zones, groups=self._groups)
        d["ui"] = {
            "nodes": {rec.id: {"x": rec.x, "y": rec.y, "muted": rec.muted,
                               "collapsed": rec.collapsed}
                      for rec in self.nodes.values()},
            "frames": {fr.id: {"title": fr.title, "members": list(fr.members),
                               "color": list(fr.color) if fr.color else None}
                       for fr in self.frames.values()},
        }
        return d

    def load_dict(self, d: Dict[str, Any]) -> None:
        graph, zones, groups = _ng_from_dict(d)
        ui = d.get("ui", {}) if isinstance(d.get("ui", {}), dict) else {}
        ui_nodes = ui.get("nodes", {}) if isinstance(ui.get("nodes", {}), dict) else {}
        self.nodes.clear()
        self.edges = []
        self.meta_seeds.clear()
        self._zones = list(zones)                    # preserved verbatim (not yet edited)
        self._groups = list(groups)
        self._back_edges = [e for e in graph.edges if e.kind != "forward"]
        for nid, inst in graph.nodes.items():
            extra = ui_nodes.get(nid, {})
            self.nodes[nid] = NodeRecord(
                nid, inst.op_key, params=dict(inst.params), modes=dict(inst.modes),
                x=float(extra.get("x", 0.0)), y=float(extra.get("y", 0.0)),
                muted=bool(extra.get("muted", False)),
                collapsed=bool(extra.get("collapsed", False)))
        for e in graph.edges:
            if e.kind == "forward":                  # back-edges ride in _back_edges
                self.edges.append((e.src, e.src_socket, e.dst, e.dst_socket))
        # GUI frames (only members that survived the load are kept; empty ⇒ dropped)
        self.frames = {}
        ui_frames = ui.get("frames", {}) if isinstance(ui.get("frames", {}), dict) else {}
        for fid, fd in ui_frames.items():
            if not isinstance(fd, dict):
                continue
            mem = [n for n in fd.get("members", []) if n in self.nodes]
            if mem:
                col = fd.get("color")
                self.frames[fid] = FrameRecord(fid, fd.get("title", "Frame"), mem,
                                               tuple(col) if col else None)
        self._notify()

    @property
    def has_unedited_structure(self) -> bool:
        """True if this file carries zones/back-edges the GUI can't fully edit yet (all
        preserved on save; surfaced so the window can warn the user). Groups are now
        GUI-manageable (make/ungroup), so they no longer count."""
        return bool(self._zones or self._back_edges)

    def save_file(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        self.path = path

    def load_file(self, path: str) -> None:
        import json
        with open(path, "r", encoding="utf-8") as f:
            self.load_dict(json.load(f))
        self.path = path


__all__ = ["GraphDocument", "NodeRecord", "FrameRecord", "LOCKED_KEY"]
