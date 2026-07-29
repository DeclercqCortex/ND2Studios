"""GUI-facing node ops that must be **Qt-free** so a headless consumer can load and
run a ``*.nd2graph.json`` saved by NodeLab v2 (review 2026-07-22).

Two ops are introduced by the GUI layer rather than the core catalog:

* ``io.load`` — the pipeline source. It has **no compute**: the engine gets its pixels
  from a seed :class:`~nodegraph.dataset.Dataset` (the GUI runner resolves the ``path``
  param to a provider; a headless consumer supplies its own seed for each ``io.load``
  node — see :func:`headless_engine`).
* ``view.viewer`` — an inspection tap; a pure pass-through compute registered into
  ``COMPUTES`` **here** (not in the Qt runner) so ``nodegraph.engine.Engine`` can run a
  GUI-authored graph without importing PySide6.

**Per-channel output taps.** A ``io.load`` / ``channel.split`` node exposes one *synthetic*
per-channel output socket ``ch0…chN-1`` in the GUI (each a single channel). The engine is
one-payload-per-node, so these can't be distinct engine outputs; instead
:func:`materialize_channel_taps` rewrites every ``chK`` output edge into a real
``channel.select`` tap (``params={"channels":[K]}``) at graph-build time — reusing the
tested select compute + its lockstep ``channel_select`` meta_transform. This runs for the
run graph, the edit-time envelope pass, and any headless consumer alike.

This module imports only ``nodegraph`` — no PySide6 — so ``import nodelab_v2.ops`` is
safe in a batch/CI context.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from nodegraph.engine import Engine
from nodegraph.graph import Edge, Graph, NodeInstance
from nodegraph.nodes import COMPUTES, register_node
from nodegraph.domains import Domain
from nodegraph.registry import InDataset, InString, NODES, OutDataset, define_node
from nodegraph.sockets import SocketType

#: a synthetic per-channel output socket name — ``ch0``, ``ch1``, … (GUI-only; the
#: materialization pass turns each wired one into a real ``channel.select`` tap).
CH_SOCKET_RE = re.compile(r"^ch(\d+)$")


def ensure_ops() -> None:
    """Idempotently register ``io.load`` (source, no compute) + ``view.viewer``
    (pass-through). Safe to call repeatedly and from any thread (pure registry
    writes)."""
    spec = NODES.get("io.load")
    if spec is None or spec.input("path") is None:
        define_node(
            "io.load", "Load ND2/TIFF file", category="io",
            inputs=[InString("path", "Path", field=False, default="")],
            outputs=[OutDataset("image")],
            adds_domains=frozenset({Domain.VOXEL}),   # the source of the image domain
            description="Open an ND2 or TIFF as the pipeline source (the GUI ingests it "
                        "once to a b2nd store next to the file; empty path = synthetic "
                        "demo). Exposes one output per channel + a combined 'All "
                        "channels' output.")
    if NODES.get("view.viewer") is None:
        register_node(
            lambda ctx: ctx.inputs[0],
            op_key="view.viewer", label="Viewer", category="io",
            inputs=[InDataset()], outputs=[OutDataset()],
            description="Inspection tap — the Viewer panel renders whatever Dataset "
                        "flows through it (V2.00 §10).")


def _real_dataset_out(op_key: str) -> str:
    """The name of a node type's first real Dataset output socket (``image`` for
    ``io.load``, ``out`` for ``channel.split`` / most nodes) — the socket a channel tap
    feeds from."""
    spec = NODES.get(op_key)
    if spec is not None:
        for s in spec.outputs:
            if s.type is SocketType.DATASET:
                return s.name
    return "out"


def materialize_channel_taps(graph: Graph) -> Graph:
    """Return a runnable graph in which every GUI-synthetic per-channel output edge
    (``src_socket`` matching ``chK``) is rewired through a real ``channel.select`` tap.

    One tap node is inserted per ``(source_node, channel_index)`` and shared by every
    edge leaving that ``chK`` socket. Full-bundle edges (``image`` / ``out`` / value
    sockets) pass through unchanged. The input graph is not mutated; if there are no
    channel taps the same graph object is returned.
    """
    taps: dict = {}                              # (src, k) -> tap node id
    extra_nodes: dict = {}
    new_edges = []
    for e in graph.edges:
        m = CH_SOCKET_RE.match(e.src_socket) if e.kind == "forward" else None
        if m is None:
            new_edges.append(e)
            continue
        k = int(m.group(1))
        key = (e.src, k)
        tap_id = taps.get(key)
        if tap_id is None:
            tap_id = f"__tap__{e.src}__c{k}"
            taps[key] = tap_id
            extra_nodes[tap_id] = NodeInstance(
                tap_id, "channel.select", params={"channels": [k]})
            real_out = _real_dataset_out(graph.nodes[e.src].op_key)
            new_edges.append(Edge(e.src, tap_id, real_out, "data", "forward"))
        new_edges.append(Edge(tap_id, e.dst, "out", e.dst_socket, e.kind))
    if not extra_nodes:
        return graph
    nodes = dict(graph.nodes)
    nodes.update(extra_nodes)
    return Graph(nodes=nodes, edges=new_edges)


def headless_engine(graph: Graph, *, seeds: Mapping[str, Any],
                    meta_seeds: Optional[Mapping[str, Any]] = None,
                    **engine_kwargs: Any) -> Engine:
    """Build a runnable :class:`~nodegraph.engine.Engine` for a GUI-authored graph
    **without** any Qt import. The caller supplies a seed Dataset per ``io.load`` node
    (headless has no file dialog / ingest runner); every other op resolves through the
    shared ``COMPUTES`` — including ``view.viewer``, registered by :func:`ensure_ops`.
    Per-channel output edges are materialized into ``channel.select`` taps first.
    """
    ensure_ops()
    graph = materialize_channel_taps(graph)
    return Engine(graph, computes=COMPUTES, seeds=dict(seeds),
                  meta_seeds=dict(meta_seeds or {}), **engine_kwargs)


__all__ = ["ensure_ops", "headless_engine", "materialize_channel_taps",
           "CH_SOCKET_RE"]
