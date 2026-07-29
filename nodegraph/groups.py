"""Node groups — the inline-expand model (nodegraph v2, Phase 4b).

A **node group** collapses a subgraph into one reusable node, exactly like Blender's
geometry-node groups. A :class:`Group` is a reusable subgraph *definition*: a body
:class:`~nodegraph.graph.Graph` bounded by a paired ``Group Input`` and ``Group
Output`` boundary node (op_keys :data:`GROUP_INPUT` / :data:`GROUP_OUTPUT`). A group
*instance* is a single placeholder node in a parent graph whose ``op_key`` is
``"group:<name>"`` (see :func:`group_name_of`); it stands in for the whole body.

:func:`expand` inlines every instance into its parent — the mirror of
:func:`nodegraph.zones.unroll`. Each instance is replaced by a **fresh unique-id copy**
of the group body (a body node ``N`` becomes ``f"{N}%{instance_id}"`` via
:func:`inst_id`), and the interface is stitched up:

* the parent edges **into** the instance node are rewired onto the copied ``Group
  Input`` boundary (external producer → ``input%inst`` → the body's real consumers),
* the copied ``Group Output`` boundary → the parent edges **out of** the instance
  (the body's real producer → ``output%inst`` → external consumers),
* an instance-to-instance edge joins the src group's copied ``Group Output`` to the
  dst group's copied ``Group Input``.

Like a zone's ``In``/``Out`` markers, the boundary nodes **persist** in the flat graph
as pure identity pass-throughs (so :func:`group_output` — the analog of
:func:`nodegraph.zones.zone_output` — can name a group's external output as
``inst_id(output_id, instance_id)``); the intervening pass-through carries the data,
so the interface flow ``external in → body`` / ``body → external out`` holds through it.

**Nestable / fixed point.** A group body may itself contain group instances, so
:func:`expand` recurses: a body is fully expanded *before* it is copied, so the result
contains no residual ``group:*`` nodes (a fixed point). Recursion is guarded by an
expansion-path stack — a group that (directly or transitively) contains itself raises a
clear :class:`ValueError` instead of looping forever. An unknown group reference or a
malformed group definition is likewise rejected with a clear :class:`ValueError`. Nodes
outside any group pass through untouched. Qt-free; pure standard library.

**Boundary-node contract** (registered elsewhere, mirroring the ``zone.*`` block in
:mod:`nodegraph.nodes`; this module does *not* register them): ``group.input`` and
``group.output`` are pure pass-throughs — one DATASET in, one DATASET out, and their
compute is ``lambda ctx: ctx.inputs[0]`` (output = the single input unchanged). They
carry no iteration/feedback semantics (unlike a zone boundary); they exist only to mark
the group's single interface input / output for :func:`expand` to stitch through.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from nodegraph.graph import Edge, Graph, NodeInstance


#: op_keys of the paired boundary markers (registered as pass-throughs elsewhere).
GROUP_INPUT = "group.input"
GROUP_OUTPUT = "group.output"

#: A group *instance* node carries this op_key prefix: ``"group:<name>"``.
GROUP_PREFIX = "group:"

#: The boundary nodes' live socket names (InDataset "data" in, OutDataset "out" out).
_IN_SOCKET = "data"
_OUT_SOCKET = "out"


@dataclass(frozen=True)
class Group:
    """One reusable subgraph definition: a body :class:`~nodegraph.graph.Graph` bounded
    by a ``Group Input`` (:attr:`input_id`) and ``Group Output`` (:attr:`output_id`)
    boundary node. Instantiated in a parent graph by a placeholder node whose op_key is
    :attr:`op_key` (``"group:<name>"``)."""

    name: str
    body: Graph
    input_id: str
    output_id: str

    @property
    def op_key(self) -> str:
        """The op_key an instance node uses to reference this group."""
        return f"{GROUP_PREFIX}{self.name}"

    @property
    def members(self) -> FrozenSet[str]:
        """Every body node id (interface boundaries + interior)."""
        return frozenset(self.body.nodes)

    @property
    def interior(self) -> FrozenSet[str]:
        """Body node ids strictly between the two boundary markers."""
        return frozenset(self.body.nodes) - {self.input_id, self.output_id}


def inst_id(node_id: str, instance_id: str) -> str:
    """The inlined id of body node ``node_id`` under group instance ``instance_id``
    (the analog of :func:`nodegraph.zones.iter_id`). Nesting composes: an inner copy
    ``N%MID`` inlined again under ``G`` becomes ``N%MID%G`` — unique + traceable."""
    return f"{node_id}%{instance_id}"


def group_name_of(op_key: str) -> Optional[str]:
    """The group name an op_key references (``"group:foo"`` → ``"foo"``), or ``None``
    if ``op_key`` is not a group-instance op_key."""
    if op_key.startswith(GROUP_PREFIX):
        return op_key[len(GROUP_PREFIX):]
    return None


def group_input(group: Group, instance_id: str) -> str:
    """The flat-graph node id of ``group``'s external INPUT boundary for a given
    instance (a pass-through the parent's inbound edge lands on)."""
    return inst_id(group.input_id, instance_id)


def group_output(group: Group, instance_id: str) -> str:
    """The flat-graph node id that carries a group instance's external OUTPUT (the
    copied ``Group Output`` pass-through) — the analog of
    :func:`nodegraph.zones.zone_output`."""
    return inst_id(group.output_id, instance_id)


def _as_group_map(groups: Union[Mapping[str, Group], Iterable[Group]]) -> Dict[str, Group]:
    """Normalise the ``groups`` arg (a name→Group map or an iterable of Groups) into a
    ``{name: Group}`` dict, rejecting duplicates / wrong types."""
    grps = list(groups.values()) if isinstance(groups, Mapping) else list(groups)
    gmap: Dict[str, Group] = {}
    for grp in grps:
        if not isinstance(grp, Group):
            raise ValueError(f"expected a Group, got {type(grp).__name__}")
        if grp.name in gmap:
            raise ValueError(f"duplicate group name {grp.name!r}")
        gmap[grp.name] = grp
    return gmap


def _validate_group(grp: Group) -> None:
    """Reject a malformed group definition (boundary nodes missing / mis-typed)."""
    for role, nid, want in (("input", grp.input_id, GROUP_INPUT),
                            ("output", grp.output_id, GROUP_OUTPUT)):
        node = grp.body.nodes.get(nid)
        if node is None:
            raise ValueError(
                f"group {grp.name!r} names {role} boundary {nid!r} but it is not in "
                f"the body ({sorted(grp.body.nodes)})")
        if node.op_key != want:
            raise ValueError(
                f"group {grp.name!r} {role} boundary {nid!r} must have op_key {want!r} "
                f"(got {node.op_key!r})")


def expand(graph: Graph,
           groups: Union[Mapping[str, Group], Iterable[Group]]) -> Graph:
    """Inline every group instance in ``graph`` into a flat :class:`Graph` the stock
    engine can run (the mirror of :func:`nodegraph.zones.unroll`).

    ``groups`` is a ``{name: Group}`` map or any iterable of :class:`Group`. Expansion
    is recursive / to a fixed point: a group body may itself contain group instances,
    which are expanded before the body is copied, so the result has no residual
    ``group:*`` nodes. A recursive (self-containing) group, an unknown group reference,
    or a malformed group definition each raise a clear :class:`ValueError`.
    """
    gmap = _as_group_map(groups)
    for grp in gmap.values():
        _validate_group(grp)
    return _expand_graph(graph, gmap, ())


def _expand_graph(graph: Graph, gmap: Mapping[str, Group],
                  stack: Tuple[str, ...]) -> Graph:
    """One expansion level: replace this graph's group-instance nodes with copies of
    their (recursively pre-expanded) bodies. ``stack`` is the group-name expansion path,
    used to detect recursive nesting."""
    # 1) classify instance nodes (op_key "group:<name>"), rejecting unknown refs
    instances: Dict[str, Group] = {}
    for nid, node in graph.nodes.items():
        name = group_name_of(node.op_key)
        if name is None:
            continue
        grp = gmap.get(name)
        if grp is None:
            raise ValueError(
                f"node {nid!r} references unknown group {name!r} "
                f"(known: {sorted(gmap)})")
        instances[nid] = grp

    nodes: Dict[str, NodeInstance] = {}
    edges: List[Edge] = []

    # 2) nodes outside any group pass through unchanged
    for nid, node in graph.nodes.items():
        if nid not in instances:
            nodes[nid] = node

    # 3) inline each instance: copy its fully-expanded body under a unique id prefix
    for inst, grp in instances.items():
        if grp.name in stack:
            path = " -> ".join(stack + (grp.name,))
            raise ValueError(
                f"group {grp.name!r} is recursively nested (expansion path {path}); "
                f"a group cannot contain itself")
        expanded_body = _expand_graph(grp.body, gmap, stack + (grp.name,))
        for bnid, bnode in expanded_body.nodes.items():
            cid = inst_id(bnid, inst)
            if cid in nodes:
                raise ValueError(
                    f"id collision {cid!r} inlining group instance {inst!r} "
                    f"(instance ids must be unique)")
            nodes[cid] = NodeInstance(cid, bnode.op_key,
                                      params=dict(bnode.params),
                                      modes=dict(bnode.modes))
        for e in expanded_body.edges:
            edges.append(Edge(inst_id(e.src, inst), inst_id(e.dst, inst),
                              e.src_socket, e.dst_socket, e.kind))

    # 4) parent edges — rewire those touching an instance onto the copied boundaries
    for e in graph.edges:
        s_grp = instances.get(e.src)
        d_grp = instances.get(e.dst)
        if s_grp is None and d_grp is None:                     # fully external
            edges.append(e)
        elif s_grp is None:                                     # external -> instance
            edges.append(Edge(e.src, inst_id(d_grp.input_id, e.dst),
                              e.src_socket, _IN_SOCKET, e.kind))
        elif d_grp is None:                                     # instance -> external
            edges.append(Edge(inst_id(s_grp.output_id, e.src), e.dst,
                              _OUT_SOCKET, e.dst_socket, e.kind))
        else:                                                   # instance -> instance
            edges.append(Edge(inst_id(s_grp.output_id, e.src),
                              inst_id(d_grp.input_id, e.dst),
                              _OUT_SOCKET, _IN_SOCKET, e.kind))

    return Graph(nodes, edges)


__all__ = [
    "Group", "GROUP_INPUT", "GROUP_OUTPUT", "GROUP_PREFIX",
    "inst_id", "group_name_of", "group_input", "group_output", "expand",
]
