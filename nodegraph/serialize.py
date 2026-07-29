"""Save / load a nodegraph v2 model to a JSON-native dict — the headless core of the
``*.nd2graph.json`` file format (V2.00 §12). Qt-free; standard library only.

This module round-trips the **structural** model only — exactly what lives in the
headless dataclasses:

* :class:`~nodegraph.graph.Graph` / :class:`~nodegraph.graph.NodeInstance` /
  :class:`~nodegraph.graph.Edge` — node ids, op_keys, params, modes, and every edge
  (INCLUDING ``kind="back"`` zone-feedback edges, which are load-bearing).
* :class:`~nodegraph.zones.Zone` — the Repeat/Sim unroll markers.
* :class:`~nodegraph.groups.Group` — a reusable subgraph whose ``body`` is itself a
  nested :class:`Graph`, serialized recursively via the same node/edge helpers.

Design notes
------------
* **JSON-native values only.** ``params`` / ``modes`` are assumed to already hold plain
  Python scalars / containers (numbers, strings, bools, lists — incl. the optional
  ``__locked__`` sticky-override list). No numpy: the saved model is pure structure, so
  this module never imports numpy, Qt, or ``nodegraph.nodes`` (no compute is needed).
* **Deterministic output** so a saved file diffs cleanly: node/edge/zone/group lists are
  emitted in a stable sorted order, and :func:`to_json` passes ``sort_keys=True`` so
  every nested mapping (params/modes included) is key-sorted too.
* **Validated on load.** An unknown/absent ``format_version`` or malformed structure
  raises a clear :class:`ValueError`.
* **Extension point.** GUI-only per-node state (canvas position, mute/collapse, frame
  membership) is *not* in the headless :class:`NodeInstance` yet, so it is intentionally
  NOT serialized here. When those fields land, extend :func:`_node_to_dict` /
  :func:`_node_from_dict` (and bump :data:`FORMAT_VERSION`); unknown extra keys in a node
  object are tolerated on load so a forward-written file still reads back.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from nodegraph.graph import Edge, Graph, NodeInstance
from nodegraph.zones import Zone
from nodegraph.groups import Group

#: The on-disk format version (``*.nd2graph.json``). A file whose ``format_version`` is
#: not in :data:`SUPPORTED_VERSIONS` is rejected by :func:`from_dict`.
FORMAT_VERSION: str = "2.0"

#: Versions :func:`from_dict` is willing to read (currently just the one).
SUPPORTED_VERSIONS = frozenset({FORMAT_VERSION})


# ── low-level element (de)serialization ──────────────────────────────────────

def _node_to_dict(node: NodeInstance) -> Dict[str, Any]:
    """One :class:`NodeInstance` → a JSON-native dict (headless fields only)."""
    return {
        "id": node.id,
        "op_key": node.op_key,
        "params": dict(node.params),
        "modes": dict(node.modes),
    }


def _node_from_dict(d: Mapping[str, Any]) -> NodeInstance:
    _require(isinstance(d, Mapping), "node must be an object")
    nid = d.get("id")
    op_key = d.get("op_key")
    _require(isinstance(nid, str), "node.id must be a string")
    _require(isinstance(op_key, str), f"node {nid!r}.op_key must be a string")
    params = d.get("params", {})
    modes = d.get("modes", {})
    _require(isinstance(params, Mapping), f"node {nid!r}.params must be an object")
    _require(isinstance(modes, Mapping), f"node {nid!r}.modes must be an object")
    return NodeInstance(nid, op_key, params=dict(params), modes=dict(modes))


def _edge_to_dict(e: Edge) -> Dict[str, Any]:
    return {
        "src": e.src,
        "dst": e.dst,
        "src_socket": e.src_socket,
        "dst_socket": e.dst_socket,
        "kind": e.kind,
    }


def _edge_from_dict(d: Mapping[str, Any]) -> Edge:
    _require(isinstance(d, Mapping), "edge must be an object")
    src, dst = d.get("src"), d.get("dst")
    _require(isinstance(src, str), "edge.src must be a string")
    _require(isinstance(dst, str), "edge.dst must be a string")
    src_socket = d.get("src_socket", "out")
    dst_socket = d.get("dst_socket", "data")
    kind = d.get("kind", "forward")
    _require(isinstance(src_socket, str), f"edge {src}->{dst}.src_socket must be a string")
    _require(isinstance(dst_socket, str), f"edge {src}->{dst}.dst_socket must be a string")
    _require(isinstance(kind, str), f"edge {src}->{dst}.kind must be a string")
    return Edge(src, dst, src_socket, dst_socket, kind)


def _edge_sort_key(d: Mapping[str, Any]) -> Tuple[str, str, str, str, str]:
    return (d["src"], d["dst"], d["src_socket"], d["dst_socket"], d["kind"])


def _graph_to_dict(graph: Graph) -> Dict[str, Any]:
    """A :class:`Graph` → ``{"nodes": [...], "edges": [...]}`` with stable ordering."""
    nodes = sorted((_node_to_dict(n) for n in graph.nodes.values()),
                   key=lambda n: n["id"])
    edges = sorted((_edge_to_dict(e) for e in graph.edges), key=_edge_sort_key)
    return {"nodes": nodes, "edges": edges}


def _graph_from_dict(d: Mapping[str, Any]) -> Graph:
    _require(isinstance(d, Mapping), "graph must be an object")
    raw_nodes = d.get("nodes", [])
    raw_edges = d.get("edges", [])
    _require(isinstance(raw_nodes, list), "graph.nodes must be a list")
    _require(isinstance(raw_edges, list), "graph.edges must be a list")
    nodes: Dict[str, NodeInstance] = {}
    for nd in raw_nodes:
        node = _node_from_dict(nd)
        if node.id in nodes:
            raise ValueError(f"duplicate node id {node.id!r}")
        nodes[node.id] = node
    edges: List[Edge] = [_edge_from_dict(ed) for ed in raw_edges]
    return Graph(nodes, edges)


def _zone_to_dict(z: Zone) -> Dict[str, Any]:
    return {
        "id": z.id,
        "kind": z.kind,
        "in_id": z.in_id,
        "out_id": z.out_id,
        "body": sorted(z.body),          # frozenset → stable list
        "iterations": z.iterations,
        "impure": z.impure,
    }


def _zone_from_dict(d: Mapping[str, Any]) -> Zone:
    _require(isinstance(d, Mapping), "zone must be an object")
    zid = d.get("id")
    _require(isinstance(zid, str), "zone.id must be a string")
    for k in ("kind", "in_id", "out_id"):
        _require(isinstance(d.get(k), str), f"zone {zid!r}.{k} must be a string")
    body = d.get("body", [])
    _require(isinstance(body, list) and all(isinstance(b, str) for b in body),
             f"zone {zid!r}.body must be a list of strings")
    iterations = d.get("iterations", 1)
    impure = d.get("impure", False)
    _require(isinstance(iterations, int) and not isinstance(iterations, bool),
             f"zone {zid!r}.iterations must be an int")
    _require(isinstance(impure, bool), f"zone {zid!r}.impure must be a bool")
    return Zone(zid, d["kind"], d["in_id"], d["out_id"],
                body=frozenset(body), iterations=iterations, impure=impure)


def _group_to_dict(g: Group) -> Dict[str, Any]:
    return {
        "name": g.name,
        "input_id": g.input_id,
        "output_id": g.output_id,
        "body": _graph_to_dict(g.body),   # nested Graph → recurse
    }


def _group_from_dict(d: Mapping[str, Any]) -> Group:
    _require(isinstance(d, Mapping), "group must be an object")
    name = d.get("name")
    _require(isinstance(name, str), "group.name must be a string")
    for k in ("input_id", "output_id"):
        _require(isinstance(d.get(k), str), f"group {name!r}.{k} must be a string")
    _require("body" in d, f"group {name!r} is missing its body graph")
    body = _graph_from_dict(d["body"])
    return Group(name, body, d["input_id"], d["output_id"])


# ── public API ────────────────────────────────────────────────────────────────

def to_dict(graph: Graph, *, zones: Iterable[Zone] = (),
            groups: Iterable[Group] = ()) -> Dict[str, Any]:
    """Serialize a graph and its zones/groups to a JSON-native dict.

    Lists are emitted in a stable order (nodes by id, edges by their 5-tuple, zones by
    id, groups by name) so the result diffs cleanly.
    """
    return {
        "format_version": FORMAT_VERSION,
        "graph": _graph_to_dict(graph),
        "zones": sorted((_zone_to_dict(z) for z in zones), key=lambda z: z["id"]),
        "groups": sorted((_group_to_dict(g) for g in groups), key=lambda g: g["name"]),
    }


def from_dict(d: Mapping[str, Any]) -> Tuple[Graph, List[Zone], List[Group]]:
    """Rebuild ``(graph, zones, groups)`` from a dict produced by :func:`to_dict`.

    Raises :class:`ValueError` on an unknown/absent ``format_version`` or malformed
    structure.
    """
    _require(isinstance(d, Mapping), "top-level document must be an object")
    version = d.get("format_version")
    if version is None:
        raise ValueError("missing 'format_version' (not a nd2graph document)")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported format_version {version!r} "
            f"(this build reads {sorted(SUPPORTED_VERSIONS)})")

    _require("graph" in d, "document is missing its 'graph'")
    graph = _graph_from_dict(d["graph"])

    raw_zones = d.get("zones", [])
    raw_groups = d.get("groups", [])
    _require(isinstance(raw_zones, list), "'zones' must be a list")
    _require(isinstance(raw_groups, list), "'groups' must be a list")
    zones = [_zone_from_dict(z) for z in raw_zones]
    groups = [_group_from_dict(g) for g in raw_groups]
    return graph, zones, groups


def to_json(graph: Graph, *, zones: Iterable[Zone] = (),
            groups: Iterable[Group] = (), indent: int = 2) -> str:
    """:func:`to_dict` → a deterministic JSON string (``sort_keys=True`` so nested
    param/mode maps are key-sorted too)."""
    return json.dumps(to_dict(graph, zones=zones, groups=groups),
                      indent=indent, sort_keys=True)


def from_json(s: str) -> Tuple[Graph, List[Zone], List[Group]]:
    """Parse a JSON string and rebuild ``(graph, zones, groups)`` via :func:`from_dict`."""
    try:
        d = json.loads(s)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    return from_dict(d)


# ── helpers ───────────────────────────────────────────────────────────────────

def _require(cond: bool, msg: str) -> None:
    """Raise a clear :class:`ValueError` when a load-time structural invariant fails."""
    if not cond:
        raise ValueError(f"malformed nd2graph document: {msg}")


__all__ = [
    "FORMAT_VERSION", "SUPPORTED_VERSIONS",
    "to_dict", "from_dict", "to_json", "from_json",
]
