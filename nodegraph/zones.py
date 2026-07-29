"""Repeat / Simulation zones — the unroll model (nodegraph v2, Phase 4a).

**Locked decision (2026-07-22, user) — the Simulation-zone × memo resolution
(V2.00 §16):** *unroll + revision-fold*, with *debug-verify + an impure escape hatch*.

A zone bounds a body subgraph between a paired ``In`` and ``Out`` node, with a **back-
edge** ``Out → In`` (:attr:`~nodegraph.graph.Edge.kind` ``"back"``) carrying the
iteration feedback. :func:`unroll` expands the zone into a **flat per-iteration chain**:
iteration ``i`` is a distinct copy of every body/boundary node, and the back-edge
becomes a *forward* edge ``Out@(i-1) → In@i``. Because the result is an ordinary acyclic
graph, the existing :class:`~nodegraph.engine.Engine` + two-hash memo run **unchanged**,
and iteration ``i``'s ``recipe_hash`` naturally folds iteration ``i-1``'s ``Out``
**revision** (the engine already folds upstream revisions) → correct **incremental
invalidation**: re-pulling with an unchanged graph is fully memo-cached; editing the
seed/a body param invalidates the chain from that point on; a downstream edit leaves the
zone cached.

* **Iteration 0** re-initialises: the external dataset wired into ``In`` feeds only
  iteration 0 (V2.00 §8 "re-initialised at t₀"); ``In@i>0`` gets the feedback instead.
  A loop-invariant external input wired straight to a *body* node feeds every iteration.
* **Purity** is assumed in the hot path (like every nodegraph compute). :func:`assert_zone_pure`
  double-computes and compares fingerprints (debug-verify). A zone flagged ``impure``
  folds a per-unroll ``epoch`` salt into its copies so it is **non-cacheable** across
  pulls (the escape hatch for genuinely stateful/random bodies).

Repeat and Simulation zones share this feedback-chain mechanism; they differ only in
intent + GUI (Repeat = run N times to convergence; Sim = temporal state across frames).

**Per-frame-T (Simulation specialization).** A ``zone.frame`` node (op_key
:data:`FRAME_OP`) placed in a zone body is stamped ``__frame__ = <iteration index>`` by
:func:`unroll`, and its compute slices that frame of its (T-stacked) input. So iteration
*t* processes frame *t* while the Sim ``In``/``Out`` feedback carries the cross-frame
STATE — a scan-over-frames-with-carried-state, exactly the Blender split (the sim sockets
carry state; per-frame data enters the body through a separate frame source). No ``Zone``
schema change: the specialization is entirely the ``zone.frame`` node + this one bake, so
serialization and the rest of the model are untouched. The caller sets ``iterations = T``.

Nesting, field-driven N, and an in-engine per-iteration verify are later refinements. Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence

from nodegraph.graph import Edge, Graph, NodeInstance

#: op_key of the per-frame slice node (Simulation-zone per-frame-T specialization). A
#: ``zone.frame`` node inside a zone body is stamped ``__frame__ = <iteration index>`` by
#: :func:`unroll`, so iteration *t* processes frame *t* of its (T-stacked) input while the
#: Sim ``In``/``Out`` feedback carries the cross-frame STATE (Blender sim-socket semantics).
#: The node's compute (in :mod:`nodegraph.nodes`) slices that frame; here we only bake the
#: index. For a per-frame Sim zone the caller sets ``iterations = T`` (the input's timepoints).
FRAME_OP = "zone.frame"


@dataclass(frozen=True)
class Zone:
    """One placed zone: its paired boundary nodes, its body (node ids strictly between
    ``In`` and ``Out``), its iteration count, and whether its body is ``impure``
    (non-cacheable across pulls)."""

    id: str
    kind: str                                   # "repeat" | "sim"
    in_id: str
    out_id: str
    body: FrozenSet[str] = field(default_factory=frozenset)
    iterations: int = 1
    impure: bool = False

    @property
    def members(self) -> FrozenSet[str]:
        return frozenset({self.in_id, self.out_id}) | frozenset(self.body)

    @property
    def n(self) -> int:
        return max(1, int(self.iterations))


def iter_id(node_id: str, zone_id: str, i: int) -> str:
    """The unrolled id of ``node_id`` in iteration ``i`` of zone ``zone_id``."""
    return f"{node_id}#{zone_id}@{i}"


def unroll(graph: Graph, zones: Sequence[Zone], *, epoch: int = 0) -> Graph:
    """Expand ``zones`` into a flat per-iteration DAG the stock engine can run.

    ``epoch`` salts the copies of any ``impure`` zone so a new epoch rekeys them
    (the non-cacheable escape hatch); pure zones ignore it (stable keys → cached).
    Phase 4a rejects nested / overlapping zones and cross-zone body edges.
    """
    zlist = list(zones)
    member_of: Dict[str, Zone] = {}
    for z in zlist:
        for nid in z.members:
            if nid in member_of:
                raise ValueError(
                    f"node {nid!r} is in two zones (nested/overlapping zones are a "
                    f"Phase-4b refinement)")
            if nid not in graph.nodes:
                raise ValueError(f"zone {z.id!r} references unknown node {nid!r}")
            member_of[nid] = z

    nodes: Dict[str, NodeInstance] = {}
    edges: List[Edge] = []

    # 1) non-zone nodes pass through unchanged
    for nid, node in graph.nodes.items():
        if nid not in member_of:
            nodes[nid] = node

    # 2) each zone member is copied once per iteration (impure → epoch-salted;
    #    a `zone.frame` node → stamped with this iteration's frame index for per-frame-T)
    for z in zlist:
        for i in range(z.n):
            for nid in z.members:
                node = graph.nodes[nid]
                params = dict(node.params)
                if z.impure:
                    params["__epoch__"] = epoch
                if node.op_key == FRAME_OP:
                    params["__frame__"] = i          # iteration t reads frame t (per-frame-T)
                cid = iter_id(nid, z.id, i)
                nodes[cid] = NodeInstance(cid, node.op_key, params=params,
                                          modes=dict(node.modes))

    # 3) edges
    for e in graph.edges:
        src_z, dst_z = member_of.get(e.src), member_of.get(e.dst)
        if e.kind == "back":
            z = dst_z or src_z
            if z is None or src_z is not dst_z or e.src != z.out_id or e.dst != z.in_id:
                raise ValueError(
                    f"a back-edge must be {z.out_id if z else '<Out>'} → "
                    f"{z.in_id if z else '<In>'} within one zone (got {e.src}→{e.dst})")
            for i in range(1, z.n):                       # Out@(i-1) → In@i (feedback)
                edges.append(Edge(iter_id(e.src, z.id, i - 1), iter_id(e.dst, z.id, i),
                                  e.src_socket, e.dst_socket))
            continue
        if src_z is None and dst_z is None:
            edges.append(e)                               # fully external
        elif src_z is None:                               # external → zone member
            if e.dst == dst_z.in_id:                      # seed: iteration 0 only (re-init t₀)
                edges.append(Edge(e.src, iter_id(e.dst, dst_z.id, 0),
                                  e.src_socket, e.dst_socket))
            else:                                          # loop-invariant: every iteration
                for i in range(dst_z.n):
                    edges.append(Edge(e.src, iter_id(e.dst, dst_z.id, i),
                                      e.src_socket, e.dst_socket))
        elif dst_z is None:                               # zone member → external (last iter)
            edges.append(Edge(iter_id(e.src, src_z.id, src_z.n - 1), e.dst,
                              e.src_socket, e.dst_socket))
        else:                                              # both in a zone → replicate per iter
            if src_z is not dst_z:
                raise ValueError("a forward edge between two different zones is a "
                                 "Phase-4b refinement (no cross-zone wiring yet)")
            for i in range(src_z.n):
                edges.append(Edge(iter_id(e.src, src_z.id, i), iter_id(e.dst, src_z.id, i),
                                  e.src_socket, e.dst_socket))

    return Graph(nodes, edges)


def zone_output(zone: Zone) -> str:
    """The unrolled node id that carries a zone's final output (its last ``Out``)."""
    return iter_id(zone.out_id, zone.id, zone.n - 1)


def assert_zone_pure(make_engine: Callable[[], "object"], node_id: str) -> object:
    """Debug-verify (the locked determinism policy): pull ``node_id`` twice on FRESH
    engines/memos and assert byte-identical outputs, catching a non-deterministic zone
    body (RNG / wall-clock / order-dependent reductions) that would otherwise poison the
    revision-fold cache. Returns the (first, realized) payload. ``make_engine`` is a
    thunk building a fresh :class:`~nodegraph.engine.Engine` over the unrolled graph.

    **C1 (V2.04 §3):** a lazy streaming output's fingerprint is *structural* — equal
    across pulls by construction, regardless of what the closures would compute — so
    purity MUST compare **bytes**: both payloads are :func:`~nodegraph.streaming.realize`\\ d
    before fingerprinting (an impure lazy body is caught; the debug-only cost is one
    full gather per pull)."""
    from nodegraph.memo import output_fingerprint
    from nodegraph.streaming import realize
    first = realize(make_engine().pull(node_id))
    second = realize(make_engine().pull(node_id))
    fa, fb = output_fingerprint(first), output_fingerprint(second)
    if fa != fb:
        raise AssertionError(
            f"zone output {node_id!r} is non-deterministic ({fa} != {fb}) — a body node "
            f"is impure; mark the zone impure (non-cacheable) or make the body pure")
    return first


__all__ = ["Zone", "iter_id", "unroll", "zone_output", "assert_zone_pure"]
