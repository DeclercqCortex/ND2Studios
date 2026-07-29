"""nodegraph — the Blender-faithful node-graph core for ND2Studios (v2).

Greenfield, Qt-free. Phase 1 delivers the model core: the eleven attribute
:mod:`~nodegraph.domains`, the lattice :mod:`~nodegraph.transfer` generator (with
real execution + structure/channel bridge routing), the :mod:`~nodegraph.sockets`
type system with implicit conversion, the :class:`~nodegraph.dataset.Dataset`
payload, and a thin node :mod:`~nodegraph.registry`.

See ``CodeLog/ClaudesPlan/V2.00_nodegraph_blender_revamp.md`` for the full design.
"""
from __future__ import annotations

from nodegraph.domains import (
    AXIS_ORDER, Domain, LATTICE_DOMAINS, STRUCTURE_DOMAINS,
    axes_of, comparable, is_finer, is_lattice, is_structure, join, meet,
)
from nodegraph.dataset import (
    AttributeLayer, AxisSizes, CALIBRATION_KEYS, Dataset,
)
from nodegraph.revision import next_revision, peek_revision
from nodegraph.reducers import (
    REDUCERS, TILEABLE_REDUCERS, PartialReducer, is_tileable, partial_reducer,
    reduce, tree_reduce,
)
from nodegraph.transfer import (
    BridgeStep, LatticeStep, TransferPlan, bridges, execute_transfer,
    lattice_transfer, plan_transfer, register_bridge,
)
from nodegraph.sockets import (
    Direction, Socket, SocketType, can_connect, can_convert,
)
from nodegraph.registry import (
    NODES, NodeSpec, SocketSpec, ModeSpec, Granularity, DIM_MODE, define_node,
    InDataset, OutDataset, OutValue, Mode, DimMode,
    InFloat, InInt, InBool, InVector, InColor, InString,
)
from nodegraph.graph import Edge, Graph, NodeInstance
from nodegraph.metadata import (
    MetaEnvelope, META_TRANSFORMS, envelope_symbols, eval_derive,
    named_meta_transform, propagate_meta, resolve_dim_default,
)
from nodegraph.provider import (
    ArrayProvider, B2ndProvider, SyntheticProvider, TileProvider,
)
from nodegraph.memo import (
    Entry, Memo, OutputHeader, digest, leaf_recipe_hash, node_recipe_hash,
    output_fingerprint, value_digest,
)
from nodegraph.engine import Compute, Engine, EvalContext, ReadContext
from nodegraph.structure import (
    COORD_COLUMNS, StructureTable, TrackMembership, connectivity_offsets,
    label_components, point_table, seeded_watershed,
)
from nodegraph.mesh import (
    MESH_STRATA, MeshElement, MeshTables, build_mesh_tables, enclosed_volume_um3,
    faces_are_closed, mesh_element, mesh_layer, mesh_names, mesh_part,
    mesh_provenance, read_mesh, surface_area_um2, with_mesh,
)
from nodegraph.bridges import (
    BRIDGE_FUNCS, broadcast_track, containing_label, gather_by_track,
    label_to_voxel, point_to_voxel, points_in_label, timepoint_to_members,
    tracks_per_timepoint, voxel_to_label, voxel_to_point,
)
from nodegraph.field import (
    Attr, BinOp, Const, Field, FieldCache, FieldContext, Input, UnaryOp,
    VirtualArray, Where, evaluate, field_expr_hash, field_key,
)
from nodegraph.boundary import boundary_dim, extract_boundary, fill_boundary
from nodegraph.streaming import (
    MapComputeProvider, PlaneRealizeProvider, StreamProvider, TileCache, TReduceProvider,
    VolumeComputeProvider, WindowView, ZReduceProvider, realize, recursion_headroom,
    stream_fp,
)
from nodegraph.nodes import (
    COMPUTES, diffraction_sigmas, gaussian_psf, register_node, to_pixels_v2,
)

__all__ = [
    # domains
    "AXIS_ORDER", "Domain", "LATTICE_DOMAINS", "STRUCTURE_DOMAINS",
    "axes_of", "comparable", "is_finer", "is_lattice", "is_structure",
    "join", "meet",
    # dataset
    "AttributeLayer", "AxisSizes", "CALIBRATION_KEYS", "Dataset",
    # revision (memo identity)
    "next_revision", "peek_revision",
    # reducers
    "REDUCERS", "TILEABLE_REDUCERS", "PartialReducer", "is_tileable",
    "partial_reducer", "reduce", "tree_reduce",
    # transfer
    "BridgeStep", "LatticeStep", "TransferPlan", "bridges", "execute_transfer",
    "lattice_transfer", "plan_transfer", "register_bridge",
    # sockets
    "Direction", "Socket", "SocketType", "can_connect", "can_convert",
    # registry
    "NODES", "NodeSpec", "SocketSpec", "ModeSpec", "Granularity", "DIM_MODE",
    "define_node", "InDataset", "OutDataset", "OutValue", "Mode", "DimMode",
    "InFloat", "InInt", "InBool", "InVector", "InColor", "InString",
    # graph + metadata pass (V2.03)
    "Edge", "Graph", "NodeInstance",
    "MetaEnvelope", "META_TRANSFORMS", "envelope_symbols", "eval_derive",
    "named_meta_transform", "propagate_meta", "resolve_dim_default",
    # provider (Phase 2a)
    "TileProvider", "SyntheticProvider", "B2ndProvider", "ArrayProvider",
    # memo + engine (Phase 2a)
    "Memo", "Entry", "OutputHeader", "digest", "value_digest", "leaf_recipe_hash",
    "node_recipe_hash", "output_fingerprint",
    "Engine", "EvalContext", "ReadContext", "Compute",
    # structure producers (Phase 2a)
    "StructureTable", "COORD_COLUMNS", "TrackMembership", "connectivity_offsets",
    "label_components", "seeded_watershed", "point_table",
    # the Mesh domain (V2.08)
    "MESH_STRATA", "MeshElement", "MeshTables", "build_mesh_tables", "with_mesh",
    "read_mesh", "mesh_element", "mesh_layer", "mesh_part", "mesh_names",
    "mesh_provenance", "faces_are_closed", "surface_area_um2", "enclosed_volume_um3",
    # structure-bridge execution (Phase 2)
    "voxel_to_label", "label_to_voxel", "voxel_to_point", "point_to_voxel",
    "containing_label", "points_in_label",
    "gather_by_track", "broadcast_track", "tracks_per_timepoint",
    "timepoint_to_members", "BRIDGE_FUNCS",
    # fields (Phase 2)
    "Field", "Const", "Attr", "Input", "BinOp", "UnaryOp", "Where", "VirtualArray",
    "FieldContext", "evaluate", "field_expr_hash", "field_key", "FieldCache",
    # constructive Point↔Label boundary (Phase 2)
    "boundary_dim", "fill_boundary", "extract_boundary",
    # streaming eval (C1 / V2.04)
    "TileCache", "StreamProvider", "MapComputeProvider", "VolumeComputeProvider",
    "ZReduceProvider", "TReduceProvider", "PlaneRealizeProvider",
    "WindowView", "stream_fp", "recursion_headroom", "realize",
    # node port (Phase 3)
    "COMPUTES", "register_node", "to_pixels_v2", "diffraction_sigmas", "gaussian_psf",
]
