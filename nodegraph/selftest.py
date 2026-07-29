"""Self-test for the nodegraph v2 core (headless, Qt-free).

Run:  python -m nodegraph.selftest

Mirrors the project's ``scripts/_pipeline_graph_selftest.py`` style (asserts +
printed checkmarks, exit 0 on success) so it needs no pytest.
"""
from __future__ import annotations

import warnings

import numpy as np

from nodegraph.domains import (
    Domain, axes_of, comparable, dropped_axes, is_finer, is_lattice,
    is_structure, join, meet,
)
from nodegraph.dataset import (
    AttributeLayer, AxisSizes, CALIBRATION_KEYS, Dataset,
)
from nodegraph.revision import next_revision, peek_revision
from nodegraph.reducers import (
    TILEABLE_REDUCERS, is_tileable, reduce, tree_reduce,
)
from nodegraph.transfer import (
    BridgeStep, LatticeStep, execute_transfer, lattice_transfer, plan_transfer,
)
from nodegraph.sockets import (
    Direction, Socket, SocketType, can_connect, can_convert,
)
from nodegraph.registry import (
    NODES, DIM_MODE, DimMode, Granularity, InDataset, InFloat, InVector, Mode,
    OutDataset, define_node,
)
from nodegraph.graph import Edge, Graph, NodeInstance
from nodegraph.metadata import (
    MetaEnvelope, envelope_symbols, eval_derive, propagate_meta, resample,
    resolve_dim_default,
    stitch, z_project,
)
from nodegraph.provider import B2ndProvider, SyntheticProvider
from nodegraph.memo import Memo, node_recipe_hash
from nodegraph.engine import Engine
from nodegraph.structure import (
    connectivity_offsets, label_components, point_table, seeded_watershed,
)
from nodegraph.bridges import (
    containing_label, label_to_voxel, point_to_voxel, points_in_label,
    voxel_to_label, voxel_to_point,
)
from nodegraph.field import (
    Attr, BinOp, Const, FieldCache, FieldContext, Input, VirtualArray, Where,
    evaluate, field_expr_hash, field_key,
)

try:
    import blosc2 as _blosc2
    _HAVE_BLOSC2, _BLOSC2_VER = True, _blosc2.__version__
except Exception:  # noqa: BLE001
    _HAVE_BLOSC2, _BLOSC2_VER = False, ""

try:
    import pyarrow as _pa
    _HAVE_PYARROW, _PA_VER = True, _pa.__version__
except Exception:  # noqa: BLE001
    _HAVE_PYARROW, _PA_VER = False, ""

try:
    import skimage as _skimage  # noqa: F401
    _HAVE_SKIMAGE = True
except Exception:  # noqa: BLE001
    _HAVE_SKIMAGE = False

try:
    import scipy as _scipy      # noqa: F401
    _HAVE_WATERSHED = _HAVE_SKIMAGE
except Exception:  # noqa: BLE001
    _HAVE_WATERSHED = False

D = Domain
AX = AxisSizes(m=2, t=3, z=4, c=2, y=5, x=6)


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


# ── domains + lattice order ──────────────────────────────────────────────────

def test_domains() -> None:
    assert is_lattice(D.VOXEL) and is_lattice(D.GLOBAL)
    assert is_lattice(D.CHANNEL) and axes_of(D.CHANNEL) == frozenset({"c"})  # V2.01 §H
    assert not is_lattice(D.LABEL) and is_structure(D.TRACK)
    assert axes_of(D.VOXEL) == frozenset({"m", "t", "z", "c", "y", "x"})
    assert axes_of(D.FRAME) == frozenset({"m", "t"})
    assert axes_of(D.GLOBAL) == frozenset()
    # order
    assert is_finer(D.VOXEL, D.FRAME) and not is_finer(D.FRAME, D.VOXEL)
    assert is_finer(D.VOXEL, D.CHANNEL)                   # Channel ⊆ Voxel
    assert comparable(D.VOXEL, D.GLOBAL)
    assert not comparable(D.MULTIPOINT, D.TIMEPOINT)      # the two branches
    assert not comparable(D.CHANNEL, D.FRAME)             # Channel is orthogonal
    # lattice: M∪T = Frame, M∩T = Global; meet total, join partial across c
    assert join(D.MULTIPOINT, D.TIMEPOINT) is D.FRAME
    assert meet(D.MULTIPOINT, D.TIMEPOINT) is D.GLOBAL
    assert join(D.PLANE, D.TIMEPOINT) is D.PLANE
    assert meet(D.CHANNEL, D.VOXEL) is D.CHANNEL
    assert join(D.CHANNEL, D.FRAME) is None               # {m,t,c} unnamed → partial
    assert dropped_axes(D.VOXEL, D.FRAME) == frozenset({"z", "c", "y", "x"})
    _ok("domains: 10 domains (7 lattice incl. Channel), order, meet total / join partial")


# ── reducers ─────────────────────────────────────────────────────────────────

def test_reducers() -> None:
    a = np.arange(24, dtype=float).reshape(2, 3, 4)
    assert np.allclose(reduce(a, (2,), "mean"), a.mean(axis=2))
    assert np.allclose(reduce(a, (0, 2), "sum"), a.sum(axis=(0, 2)))
    assert np.allclose(reduce(a, (1,), "max"), a.max(axis=1))
    assert np.array_equal(reduce(a, (0, 1, 2), "count"), np.int64(24))
    assert np.allclose(reduce(a, (2,), "first"), a[:, :, 0])
    _ok("reducers: mean/sum/max/count/first over axes")


# ── partial reducers (tree-reduce across tiles) ──────────────────────────────

def test_partial_reducers() -> None:
    a = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)   # keep axis0; reduce 1,2
    axis = (1, 2)

    def tiles_of(arr):
        # partition the reduced axes into ragged chunks, in canonical scan order
        out = []
        for i0, i1 in ((0, 2), (2, 3)):          # axis-1 chunks
            for j0, j1 in ((0, 1), (1, 4)):      # axis-2 chunks
                out.append(arr[:, i0:i1, j0:j1])
        return out

    tiles = tiles_of(a)
    # tiled tree-reduce equals the whole-array reduce
    for name, whole in (("mean", a.mean(axis=axis)), ("sum", a.sum(axis=axis)),
                        ("max", a.max(axis=axis)), ("min", a.min(axis=axis))):
        assert np.allclose(tree_reduce(tiles, axis, name), whole), name
    assert np.array_equal(tree_reduce(tiles, axis, "count"), np.full((2,), 12))
    # first: leftmost tile (global reduced-index 0) wins under canonical order
    assert np.allclose(tree_reduce(tiles, axis, "first"), a[:, 0, 0])
    # NaN-aware: mixed nan/non-nan tiles fold to nanmean
    b = a.copy(); b[0, 0, 0] = np.nan
    assert np.allclose(tree_reduce(tiles_of(b), axis, "mean"),
                       np.nanmean(b, axis=axis), equal_nan=True)
    # median is not a monoid → not tileable → callers realize the whole domain
    assert not is_tileable("median")
    try:
        tree_reduce(tiles, axis, "median")
        raise AssertionError("expected median to be non-tileable")
    except ValueError:
        pass
    assert TILEABLE_REDUCERS == {"mean", "sum", "max", "min", "count", "first"}
    _ok("partial reducers: tiled tree-reduce == whole; NaN-aware; median falls back")


# ── revision counter + safe immutability (V2.02 §4) ──────────────────────────

def test_revision_and_immutability() -> None:
    r1 = next_revision()
    r2 = next_revision()
    assert r2 > r1                                   # strictly monotonic
    p = peek_revision()
    assert peek_revision() == p and p >= r2          # peek doesn't consume

    # a fresh layer freezes its values and carries a revision
    src = np.ones((2, 3))                            # writeable producer buffer
    lay = AttributeLayer(D.FRAME, "a", src)
    assert lay.revision > 0
    assert not lay.values.flags.writeable            # frozen read-only
    assert lay.values is not src and src.flags.writeable  # producer NOT aliased/frozen
    try:
        lay.values[0, 0] = 5.0
        raise AssertionError("expected read-only array")
    except ValueError:
        pass

    # an already read-only input is adopted without a needless copy
    ro = np.ones((2, 3)); ro.flags.writeable = False
    assert AttributeLayer(D.FRAME, "b", ro).values is ro

    # mutate → same key, new values, fresh (greater) revision
    lay2 = lay.mutate(np.zeros((2, 3)))
    assert lay2.key == lay.key and lay2.revision > lay.revision
    assert np.array_equal(lay2.values, np.zeros((2, 3)))
    assert not lay2.values.flags.writeable
    _ok("revision + immutability: monotonic, frozen values, no aliasing, mutate re-stamps")


# ── dataset / attribute layers ───────────────────────────────────────────────

def test_dataset() -> None:
    assert AX.shape_for(D.VOXEL) == (2, 3, 4, 2, 5, 6)   # (m,t,z,c,y,x)
    assert AX.shape_for(D.FRAME) == (2, 3)
    assert AX.shape_for(D.CHANNEL) == (2,)               # (c,)
    assert AX.shape_for(D.GLOBAL) == ()
    ds = Dataset(axes=AX, metadata={"pixel_size_um": 1.7})
    frame = AttributeLayer(D.FRAME, "count", np.ones((2, 3)))
    ds2 = ds.with_attribute(frame)
    assert ds.get(D.FRAME, "count") is None          # structural sharing: original untouched
    assert ds2.get(D.FRAME, "count").values.shape == (2, 3)
    # shape validation
    try:
        ds.with_attribute(AttributeLayer(D.FRAME, "bad", np.ones((2, 2))))
        raise AssertionError("expected shape error")
    except ValueError:
        pass
    # §7b: with_structure preserves each table's z_kind into __struct_zkind__ provenance
    # (keyed by domain+layer), readable via structure_zkind — the generic dimensionality
    # inheritance mechanism. Two layers on the same domain keep independent z_kinds.
    from nodegraph.structure import StructureTable as _ST
    p2 = _ST(D.POINT, {"id": np.arange(1), "y": np.zeros(1), "x": np.zeros(1)},
             layer="flat", z_kind="plane_index")
    p3 = _ST(D.POINT, {"id": np.arange(1), "z": np.zeros(1), "y": np.zeros(1),
                       "x": np.zeros(1)}, layer="vol", z_kind="subpixel")
    dss = ds.with_structure(p2).with_structure(p3)
    assert dss.structure_zkind(D.POINT, "flat") == "plane_index"
    assert dss.structure_zkind(D.POINT, "vol") == "subpixel"
    assert dss.structure_zkind(D.POINT, "missing") is None
    assert ds.structure_zkind(D.POINT, "flat") is None      # original untouched (COW)
    _ok("dataset: shape_for, structural sharing, shape validation, z_kind provenance (§7b)")


# ── transfer plans (generation + routing) ────────────────────────────────────

def test_plans() -> None:
    # comparable lattice → single reduce / broadcast
    p = plan_transfer(D.VOXEL, D.FRAME)
    assert p.generated and len(p.steps) == 1
    assert isinstance(p.steps[0], LatticeStep) and p.steps[0].kind == "reduce"
    assert p.steps[0].axes == frozenset({"z", "c", "y", "x"})   # now reduces c too
    assert plan_transfer(D.FRAME, D.VOXEL).steps[0].kind == "broadcast"
    assert plan_transfer(D.FRAME, D.FRAME).steps[0].kind == "identity"

    # Channel is a lattice domain now → Voxel↔Channel are GENERATED (V2.01 §H)
    pc = plan_transfer(D.VOXEL, D.CHANNEL)
    assert pc.generated and pc.steps[0].kind == "reduce"
    assert pc.steps[0].axes == frozenset({"m", "t", "z", "y", "x"})
    assert plan_transfer(D.CHANNEL, D.VOXEL).steps[0].kind == "broadcast"

    # incomparable lattice → reduce then broadcast, still generated
    mt = plan_transfer(D.MULTIPOINT, D.TIMEPOINT)
    assert mt.generated and [s.kind for s in mt.steps] == ["reduce", "broadcast"]

    # routing through bridges (not generated)
    vt = plan_transfer(D.VOXEL, D.TRACK)
    assert not vt.generated
    assert vt.steps[-1].dst is D.TRACK and isinstance(vt.steps[-1], BridgeStep)
    lm = plan_transfer(D.LABEL, D.MULTIPOINT)      # Label→Frame→Multipoint
    assert not lm.generated and lm.dst is D.MULTIPOINT
    tm = plan_transfer(D.TRACK, D.MULTIPOINT)      # Track→Timepoint→Multipoint
    assert not tm.generated and any(isinstance(s, BridgeStep) for s in tm.steps)
    assert plan_transfer(D.VOXEL, D.LABEL).describe()  # doesn't crash
    _ok("transfer plans: lattice generation + bridge routing")


# ── transfer execution (lattice, real numpy) ─────────────────────────────────

def test_execution() -> None:
    vox = AttributeLayer(D.VOXEL, "intensity",
                         np.arange(np.prod(AX.shape_for(D.VOXEL)), dtype=float)
                         .reshape(AX.shape_for(D.VOXEL)))
    # Voxel(m,t,z,c,y,x) → Frame (mean over z,c,y,x = positions 2,3,4,5)
    fr = lattice_transfer(vox, D.FRAME, AX)
    assert fr.domain is D.FRAME and fr.values.shape == (2, 3)
    assert np.allclose(fr.values, vox.values.mean(axis=(2, 3, 4, 5)))
    # Voxel → Channel (mean over m,t,z,y,x = positions 0,1,2,4,5) — the c-axis fix
    ch = lattice_transfer(vox, D.CHANNEL, AX)
    assert ch.domain is D.CHANNEL and ch.values.shape == (2,)
    assert np.allclose(ch.values, vox.values.mean(axis=(0, 1, 2, 4, 5)))
    # Voxel → Global (scalar mean)
    g = lattice_transfer(vox, D.GLOBAL, AX)
    assert g.values.shape == () and np.allclose(g.values, vox.values.mean())
    # Frame → Voxel (broadcast) then back → identity for mean
    back = lattice_transfer(fr, D.VOXEL, AX)
    assert back.values.shape == AX.shape_for(D.VOXEL)
    assert np.allclose(lattice_transfer(back, D.FRAME, AX).values, fr.values)

    # multi-axis broadcast insertion: Timepoint → Voxel (now 6-D target)
    tp = AttributeLayer(D.TIMEPOINT, "elapsed", np.array([10.0, 20.0, 30.0]))
    tv = lattice_transfer(tp, D.VOXEL, AX)
    assert tv.values.shape == AX.shape_for(D.VOXEL)
    for t in range(AX.t):
        assert np.allclose(tv.values[:, t], tp.values[t])   # select t=axis1 → all == tp[t]

    # incomparable execution: Multipoint → Timepoint = mean over m, const over t
    mp = AttributeLayer(D.MULTIPOINT, "qc", np.array([4.0, 8.0]))
    tt = lattice_transfer(mp, D.TIMEPOINT, AX)
    assert tt.values.shape == (3,) and np.allclose(tt.values, mp.values.mean())

    # reducer variants
    assert np.allclose(lattice_transfer(vox, D.FRAME, AX, "sum").values,
                       vox.values.sum(axis=(2, 3, 4, 5)))
    assert np.allclose(lattice_transfer(vox, D.PLANE, AX, "max").values,
                       vox.values.max(axis=(3, 4, 5)))   # Plane{m,t,z}: reduce c,y,x

    # a bridge plan refuses to execute (Phase 3)
    try:
        execute_transfer(vox, plan_transfer(D.VOXEL, D.LABEL), AX)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass
    _ok("transfer execution: reduce/broadcast/round-trip/incomparable + bridge guard")


# ── sockets ──────────────────────────────────────────────────────────────────

def test_sockets() -> None:
    assert can_convert(SocketType.INT, SocketType.FLOAT)
    assert can_convert(SocketType.FLOAT, SocketType.VECTOR)
    assert not can_convert(SocketType.STRING, SocketType.FLOAT)
    out_f = Socket("o", SocketType.FLOAT, Direction.OUT)
    in_i = Socket("i", SocketType.INT, Direction.IN)
    in_ds = Socket("d", SocketType.DATASET, Direction.IN)
    out_ds = Socket("d", SocketType.DATASET, Direction.OUT)
    assert can_connect(out_f, in_i)                     # Float→Int implicit
    assert can_connect(out_ds, in_ds)                   # Dataset↔Dataset
    assert not can_connect(out_f, in_ds)                # value↮dataset
    assert not can_connect(in_i, out_f)                 # wrong direction
    _ok("sockets: conversion table + can_connect")


# ── registry / node API ──────────────────────────────────────────────────────

def test_registry() -> None:
    # NB: a clearly-fake op_key — fixtures must NOT squat on a real node's op_key or
    # they clobber it in the global registry (the real ``detect.spots`` node, V2.00
    # §14 / nodes.py, previously collided with this fixture).
    spec = define_node(
        "test.registry_demo", "Detect Spots", category="detection",
        inputs=[InDataset(),
                InFloat("radius", "Radius", unit="um",
                        derive="0.61*(emission_nm or 520)/(na or 1.4)/1000")],
        outputs=[OutDataset()],
        modes=[Mode("polarity", ["Bright", "Dark"])],
    )
    assert NODES.get("test.registry_demo") is spec
    r = spec.input("radius")
    assert r.unit == "um" and r.is_field and r.instantiate().type is SocketType.FLOAT
    assert spec.modes[0].resolved_default() == "Bright"
    _ok("registry: define_node, unit/derive sockets, modes")


# ── vector arity + socket threading (V2.03 §4 C1) ────────────────────────────

def test_socket_dims() -> None:
    v2o = Socket("o", SocketType.VECTOR, Direction.OUT, dims=2)
    v3o = Socket("o", SocketType.VECTOR, Direction.OUT, dims=3)
    v2i = Socket("i", SocketType.VECTOR, Direction.IN, dims=2)
    v3i = Socket("i", SocketType.VECTOR, Direction.IN, dims=3)
    assert can_connect(v2o, v2i)                       # equal arity
    assert can_connect(v2o, v3i)                       # widen 2→3 (pad axial 0)
    assert not can_connect(v3o, v2i)                   # narrow 3→2 rejected
    assert can_connect(Socket("f", SocketType.FLOAT, Direction.OUT), v3i)  # broadcast
    # instantiate carries dims + domain (the audit-flagged drops)
    assert InVector("shift", dims=2).instantiate().dims == 2
    assert InFloat("d", domain=D.VOXEL).instantiate().domain is D.VOXEL
    _ok("socket dims: vector widen-only, dims/domain threaded through instantiate")


# ── calibration write-path + axis helpers (V2.03 §2 A1) ──────────────────────

def test_calibration() -> None:
    ds = Dataset(axes=AxisSizes(z=1), metadata={"pixel_size_um": 0.2, "z_step_um": 0.5})
    ds2 = ds.with_metadata(pixel_size_um=0.1)
    assert ds.metadata["pixel_size_um"] == 0.2 and ds2.metadata["pixel_size_um"] == 0.1
    ds3 = ds.with_metadata(z_step_um=None)             # None removes the key
    assert "z_step_um" not in ds3.metadata and "z_step_um" in ds.metadata
    assert not AxisSizes(z=1).is_volumetric and AxisSizes(z=5).is_volumetric
    assert "pixel_size_um" in CALIBRATION_KEYS and "z_step_um" in CALIBRATION_KEYS
    # reshaped_axes drops lattice layers orphaned by an axis change
    big = AxisSizes(z=4, y=8, x=8)
    d = Dataset(axes=big).with_layer(D.PLANE, "focus", np.ones(big.shape_for(D.PLANE)))
    small = AxisSizes(z=1, y=8, x=8)
    assert d.reshaped_axes(small).get(D.PLANE, "focus") is None
    try:
        d.reshaped_axes(small, drop_stale=False)
        raise AssertionError("expected orphan error")
    except ValueError:
        pass
    _ok("calibration: with_metadata copy-on-write, is_volumetric, reshaped_axes")


# ── node variants: the 2D/3D lever reconfigures sockets (V2.03 §3 B1/B2/B3) ───

def test_node_variants() -> None:
    spec = define_node(
        "filt.gauss", "Gaussian", category="enhancement",
        inputs=[InDataset(),
                InFloat("sigma", "Sigma", unit="um",
                        available_in={DIM_MODE: frozenset({"2D"})}),
                InFloat("sigma_xy", "Sigma XY", unit="um",
                        available_in={DIM_MODE: frozenset({"3D"})}),
                InFloat("sigma_z", "Sigma Z", unit="um_axial",
                        available_in={DIM_MODE: frozenset({"3D"})})],
        outputs=[OutDataset()],
        modes=[DimMode()],
        granularity={"2D": Granularity.TILEABLE, "3D": Granularity.WHOLE_VOLUME},
        kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})},
    )
    st2d, st3d = {"dim": "2D"}, {"dim": "3D"}
    in2d = {s.name for s in spec.active_inputs(st2d)}
    in3d = {s.name for s in spec.active_inputs(st3d)}
    assert in2d == {"data", "sigma"}                   # 2D: single scalar sigma
    assert in3d == {"data", "sigma_xy", "sigma_z"}     # 3D: anisotropic pair
    # granularity + kernel-axes jump with the lever
    assert spec.resolve_granularity(st2d) is Granularity.TILEABLE
    assert spec.resolve_granularity(st3d) is Granularity.WHOLE_VOLUME
    assert spec.resolve_kernel_axes(st3d) == frozenset({"z", "y", "x"})
    # the lever itself
    lever = spec.dim_lever()
    assert spec.has_dim_lever() and lever.presentation == "header"
    assert lever.is_dim_lever and spec.default_state()[DIM_MODE] == "2D"
    _ok("node variants: available_in gates sockets; granularity/kernel-axes per lever")


# ── the edit-time MetaEnvelope pass (V2.03 §2 A3) ────────────────────────────

def test_metadata_pass() -> None:
    define_node("io.load", "Load", outputs=[OutDataset()])            # source (identity)
    define_node("filt.resample", "Resample", inputs=[InDataset()],
                outputs=[OutDataset()], modes=[DimMode()], meta_transform=resample)
    define_node("filt.blur", "Blur", inputs=[InDataset()], outputs=[OutDataset()],
                modes=[DimMode()], meta_transform=None,
                granularity={"2D": Granularity.TILEABLE, "3D": Granularity.WHOLE_VOLUME})
    define_node("proj.z", "Z Project", inputs=[InDataset()], outputs=[OutDataset()],
                meta_transform=z_project)
    define_node("stitch.tiles", "Stitch", inputs=[InDataset()], outputs=[OutDataset()],
                meta_transform=stitch)

    # source → resample(×2 lateral) → blur : the blur sees the transformed calibration
    g = Graph()
    g.add(NodeInstance("L", "io.load"))
    g.add(NodeInstance("R", "filt.resample", params={"scale_xy": 2.0}, modes={"dim": "2D"}))
    g.add(NodeInstance("B", "filt.blur", modes={"dim": "2D"}))
    g.connect("L", "R"); g.connect("R", "B")
    seed = MetaEnvelope(axes=AxisSizes(z=1, y=10, x=10), metadata={"pixel_size_um": 0.2})
    env = propagate_meta(g, {"L": seed})
    assert env["R"].axes.y == 20 and env["R"].axes.x == 20            # extent doubled
    assert abs(env["R"].metadata["pixel_size_um"] - 0.1) < 1e-9       # px halved (finer)
    assert env["B"].axes.x == 20 and abs(env["B"].metadata["pixel_size_um"] - 0.1) < 1e-9

    # z-project drops z + z_step and flags collapse; downstream lever defaults to 2D
    g2 = Graph()
    g2.add(NodeInstance("L", "io.load")); g2.add(NodeInstance("Z", "proj.z"))
    g2.connect("L", "Z")
    e2 = propagate_meta(g2, {"L": MetaEnvelope(
        axes=AxisSizes(z=5, y=4, x=4), metadata={"pixel_size_um": 0.2, "z_step_um": 0.5})})
    assert e2["Z"].axes.z == 1 and "z_step_um" not in e2["Z"].metadata
    assert e2["Z"].metadata.get("z_collapsed") is True

    # metadata-intelligent lever default: z>1 ⇒ 3D, z==1 ⇒ 2D (incl. post z-project)
    blur = NODES.get("filt.blur")
    assert resolve_dim_default(blur, MetaEnvelope(axes=AxisSizes(z=5))) == "3D"
    assert resolve_dim_default(blur, MetaEnvelope(axes=AxisSizes(z=1))) == "2D"
    assert resolve_dim_default(blur, e2["Z"]) == "2D"
    assert envelope_symbols(MetaEnvelope(axes=AxisSizes(z=5)))["is_3d"] is True

    # stitch grows Y,X to an UNKNOWN extent (never a silent guess); M→1
    g3 = Graph()
    g3.add(NodeInstance("L", "io.load")); g3.add(NodeInstance("S", "stitch.tiles"))
    g3.connect("L", "S")
    e3 = propagate_meta(g3, {"L": MetaEnvelope(axes=AxisSizes(m=4, y=10, x=10))})
    assert e3["S"].axes.m == 1 and {"y", "x"} <= e3["S"].unknown_axes

    # cycles are rejected outside zones (V2.00 §7)
    gc = Graph()
    gc.add(NodeInstance("A", "io.load")); gc.add(NodeInstance("B", "filt.blur"))
    gc.connect("A", "B"); gc.connect("B", "A")
    try:
        gc.topo_order()
        raise AssertionError("expected cycle rejection")
    except ValueError:
        pass
    _ok("metadata pass: per-edge propagation, z-collapse, lever default, UNKNOWN, cycle guard")


def test_domain_interface() -> None:
    """The socket domain-rail + wire-tint source: reads/adds_domains declarations,
    their accumulation through ``propagate_meta``, and the mismatch validation
    (fake ``test.*`` op_keys so real nodes are never clobbered)."""
    define_node("test.src", "Src", outputs=[OutDataset()],
                adds_domains=frozenset({D.VOXEL}))
    define_node("test.blur", "Blur", inputs=[InDataset()], outputs=[OutDataset()])  # transparent
    define_node("test.label", "Label", inputs=[InDataset()], outputs=[OutDataset()],
                reads_domains=frozenset({D.VOXEL}), adds_domains=frozenset({D.LABEL}))
    define_node("test.measure", "Measure", inputs=[InDataset()], outputs=[OutDataset()],
                reads_domains=frozenset({D.VOXEL, D.LABEL}), adds_domains=frozenset({D.LABEL}))

    # accumulation: src{VOX} → blur{VOX} (transparent) → label{VOX,LBL} → measure{VOX,LBL}
    g = Graph()
    for nid, op in [("S", "test.src"), ("B", "test.blur"),
                    ("L", "test.label"), ("M", "test.measure")]:
        g.add(NodeInstance(nid, op))
    g.connect("S", "B"); g.connect("B", "L"); g.connect("L", "M")
    env = propagate_meta(g, {"S": MetaEnvelope()})
    assert env["S"].domains == frozenset({D.VOXEL})
    assert env["B"].domains == frozenset({D.VOXEL})                 # transparent pass-through
    assert env["L"].domains == frozenset({D.VOXEL, D.LABEL})        # label added
    assert env["M"].domains == frozenset({D.VOXEL, D.LABEL})        # inherited (measure adds LBL)

    # spec helpers: out_domains unions, missing_domains flags an absent requirement
    label = NODES.get("test.label"); measure = NODES.get("test.measure")
    assert label.out_domains(frozenset({D.VOXEL})) == frozenset({D.VOXEL, D.LABEL})
    assert measure.missing_domains(frozenset({D.VOXEL})) == frozenset({D.LABEL})  # no LBL upstream
    assert measure.missing_domains(frozenset({D.VOXEL, D.LABEL})) == frozenset()  # satisfied

    # a merge unions BOTH Dataset predecessors' domain-sets
    define_node("test.merge", "Merge", inputs=[InDataset(multi=True)], outputs=[OutDataset()])
    define_node("test.spots", "Spots", inputs=[InDataset()], outputs=[OutDataset()],
                adds_domains=frozenset({D.POINT}))
    g2 = Graph()
    for nid, op in [("S", "test.src"), ("L", "test.label"),
                    ("P", "test.spots"), ("G", "test.merge")]:
        g2.add(NodeInstance(nid, op))
    g2.connect("S", "L"); g2.connect("S", "P")
    g2.connect("L", "G"); g2.connect("P", "G")
    e2 = propagate_meta(g2, {"S": MetaEnvelope()})
    assert e2["G"].domains == frozenset({D.VOXEL, D.LABEL, D.POINT})   # both branches merged
    _ok("domain interface: reads/adds declarations, accumulation, merge-union, mismatch validation")


# ── lazy tiled provider (Phase 2a — V2.02 §3 / V2.03 §5) ─────────────────────

def test_provider() -> None:
    ax = AxisSizes(m=1, t=1, z=3, c=1, y=300, x=400)
    sp = SyntheticProvider(ax, tile=128, levels=2)
    # exact window read
    r = sp.get_region(0, 0, 0, 1, 0, 10, 20, 30, 45)
    assert r.shape == (10, 15) and r.dtype == np.uint16
    # edge tile clips; get_tile == get_region on the block's coords
    t23 = sp.get_tile(0, 0, 0, 0, 0, 2, 3)
    assert t23.shape == (44, 16)              # (300-256, 400-384)
    assert np.array_equal(t23, sp.get_region(0, 0, 0, 0, 0, 256, 300, 384, 400))
    # subvolume stacks z (planar blocks gathered); region-volume over z
    sv = sp.get_subvolume(0, 0, 0, 0, 0, 3, 0, 0)
    assert sv.shape == (3, 128, 128) and np.array_equal(sv[1], sp.get_tile(0, 0, 0, 1, 0, 0, 0))
    assert sp.get_region_volume(0, 0, 0, 0, 0, 2, 5, 9, 5, 10).shape == (2, 4, 5)
    # multiscale: level-1 axes halve; stride-decimated values
    assert sp.level_axes(1).y == 150 and sp.level_axes(1).x == 200
    assert np.array_equal(sp.get_region(1, 0, 0, 0, 0, 0, 4, 0, 4),
                          sp.get_region(0, 0, 0, 0, 0, 0, 8, 0, 8)[::2, ::2])
    assert sp.tiles_per_plane(0) == (3, 4)    # ceil(300/128), ceil(400/128)
    try:
        sp.get_tile(0, 0, 0, 0, 0, 99, 0)     # out of range
        raise AssertionError("expected IndexError")
    except IndexError:
        pass
    _ok("provider: synthetic region/tile/subvolume/volume + multiscale + edge clip")

    if _HAVE_BLOSC2:
        rng = np.random.default_rng(0)
        vol = rng.integers(0, 4096, size=(1, 1, 3, 2, 40, 50), dtype=np.uint16)
        bp = B2ndProvider.from_array(vol, tile=16, levels=2)   # planar (1,1,1,1,16,16) blocks
        assert bp.axes.y == 40 and bp.axes.c == 2 and bp.levels == 2
        # b2nd window read round-trips exactly vs the source slice
        assert np.array_equal(bp.get_region(0, 0, 0, 2, 1, 5, 12, 8, 20),
                              vol[0, 0, 2, 1, 5:12, 8:20])
        assert np.array_equal(bp.get_tile(0, 0, 0, 1, 0, 1, 2), vol[0, 0, 1, 0, 16:32, 32:48])
        sv = bp.get_subvolume(0, 0, 0, 0, 0, 3, 0, 0)
        assert sv.shape[0] == 3 and np.array_equal(sv[2], vol[0, 0, 2, 0, 0:16, 0:16])
        # level-1 == mean-downsample of the source (Y,X halved)
        assert bp.level_axes(1).y == 20 and bp.level_axes(1).x == 25
        exp = vol[0, 0, 2, 1].reshape(20, 2, 25, 2).mean(axis=(1, 3)).astype(vol.dtype)
        assert np.array_equal(bp.get_region(1, 0, 0, 2, 1, 0, 20, 0, 25), exp)

        # C4: DISK-backed store — write → open (lazy) → identical reads; mtime-based
        # version (cheap, disk-cheap identity) that differs from an in-memory store.
        import os
        import shutil
        import tempfile
        d = tempfile.mkdtemp(prefix="b2nd_disk_")
        try:
            store = os.path.join(d, "store")
            dp = B2ndProvider.write(vol, store, tile=16, levels=2)
            assert dp.axes.y == 40 and dp.levels == 2
            assert np.array_equal(dp.get_region(0, 0, 0, 2, 1, 5, 12, 8, 20),
                                  vol[0, 0, 2, 1, 5:12, 8:20])
            assert np.array_equal(dp.get_region(1, 0, 0, 2, 1, 0, 20, 0, 25), exp)
            re = B2ndProvider.open(store)             # re-open without re-ingest
            assert np.array_equal(re.get_region(0, 0, 0, 0, 0, 0, 8, 0, 8),
                                  vol[0, 0, 0, 0, 0:8, 0:8])
            # disk identity folds path+mtime (cheap); distinct from the in-memory hash
            fp = dp.fingerprint()
            assert fp[0] == "b2nd-disk" and fp[2] == dp._mtime_ns and dp.version == fp
            assert fp != bp.fingerprint()
            del dp, re
        finally:
            import gc
            gc.collect()
            shutil.rmtree(d, ignore_errors=True)
        _ok(f"provider: b2nd planar-block store (in-memory + DISK write/open, "
            f"mtime version) + mean pyramid (blosc2 {_BLOSC2_VER})")
    else:
        _ok("provider: b2nd tests SKIPPED (blosc2 not installed)")


# ── two-hash memo (V2.02 §6) ─────────────────────────────────────────────────

def test_memo() -> None:
    m = Memo()
    rh = node_recipe_hash("op.a", {"k": 1}, (), ())
    assert m.get(rh) is None and m.misses == 1              # miss
    e1 = m.put(rh, np.ones((4, 4)), node_key="A")
    assert m.get(rh) is e1 and m.hits == 1                  # hit
    # dedup: an identical payload from another node shares the blob
    e2 = m.put(node_recipe_hash("op.b", {"k": 2}, (), ()), np.ones((4, 4)), node_key="B")
    assert e2.payload is e1.payload
    # cutoff: same node, identical output → not "changed"; different output → changed
    assert m.put(rh, np.ones((4, 4)), node_key="A").changed is False
    e_diff = m.put(rh, np.zeros((4, 4)), node_key="A")
    assert e_diff.changed is True and e_diff.revision > e1.revision
    # recipe-hash sensitivity: params + upstream (hashes/revisions) all matter
    assert node_recipe_hash("op.a", {"k": 1}, (), ()) == rh
    assert node_recipe_hash("op.a", {"k": 2}, (), ()) != rh
    assert node_recipe_hash("op.a", {"k": 1}, ("h",), (7,)) != rh
    _ok("memo: two-hash lookup, blob dedup, cutoff, revision, hash sensitivity")


def test_memo_gc() -> None:
    """Memo GC — byte-budget LRU eviction (V2.04 §6b follow-up). Bounds the persistent-
    memo hazard: an eager full-raster node in a high-T unrolled zone otherwise pins one
    raster per iteration forever. Eviction is correctness-safe (a later recompute)."""
    from nodegraph.provider import ArrayProvider

    each = int(np.ones((100, 100)).nbytes)                 # 80_000 (float64)

    # 1. unbounded default (budget None) is byte-identical to pre-GC behavior: no eviction
    mu = Memo()
    assert mu.budget is None
    for i in range(50):
        mu.put(node_recipe_hash("t.u", {"i": i}, (), ()), np.zeros((64, 64)),
               node_key=f"u{i}")
    assert mu.evictions == 0 and len(mu._entries) == 50

    # 2. dedup accounting is per unique BLOB, not per entry — the shared blob is counted
    #    once and freed only when the LAST referencing entry drops (refcount invariant)
    md = Memo(budget_bytes=1 << 20)
    ra = node_recipe_hash("t.a", {}, (), ())
    rb = node_recipe_hash("t.b", {}, (), ())
    ea = md.put(ra, np.ones((100, 100)), node_key="A")
    eb = md.put(rb, np.ones((100, 100)), node_key="B")     # identical content → shared blob
    assert eb.payload is ea.payload and md._fp_refs[ea.fingerprint] == 2
    assert md.nbytes == each                               # counted ONCE despite two entries
    md.invalidate(ra)                                      # one ref dropped, blob survives
    assert ea.fingerprint in md._blobs and md.nbytes == each
    md.invalidate(rb)                                      # last ref → blob + bytes freed
    assert md.nbytes == 0 and not md._blobs

    # 3. byte-budget LRU: distinct large payloads over budget evict the OLDEST
    budget = 3 * each + 1
    ml = Memo(budget_bytes=budget)
    rhs = [node_recipe_hash("t.g", {"i": i}, (), ()) for i in range(10)]
    for i, rh in enumerate(rhs):
        ml.put(rh, np.full((100, 100), float(i)), node_key=f"g{i}")
    assert ml.nbytes <= budget and ml.evictions >= 6
    assert ml.get(rhs[0]) is None and ml.get(rhs[-1]) is not None   # oldest gone, newest kept

    # 3b. recency: a HIT promotes an entry so it outlives a newer-but-cold one
    ml2 = Memo(budget_bytes=2 * each + 1)
    r0, r1, r2 = (node_recipe_hash("t.r", {"i": i}, (), ()) for i in range(3))
    ml2.put(r0, np.full((100, 100), 0.0), node_key="r0")
    ml2.put(r1, np.full((100, 100), 1.0), node_key="r1")
    assert ml2.get(r0) is not None                         # promote r0 to most-recent
    ml2.put(r2, np.full((100, 100), 2.0), node_key="r2")   # evicts the LRU = r1, not r0
    assert ml2.get(r1) is None and ml2.get(r0) is not None and ml2.get(r2) is not None

    # 4. _last_fp is NEVER GC'd — an evict-then-recompute of identical bytes still cuts off
    mc = Memo(budget_bytes=each + 1)                       # holds ~one entry
    ka = node_recipe_hash("t.c", {"i": "a"}, (), ())
    kb = node_recipe_hash("t.c", {"i": "b"}, (), ())
    mc.put(ka, np.full((100, 100), 7.0), node_key="A")
    mc.put(kb, np.full((100, 100), 9.0), node_key="B")     # evicts the ka entry
    assert mc.get(ka) is None
    again = mc.put(ka, np.full((100, 100), 7.0), node_key="A")   # identical content
    assert again.changed is False                          # cutoff preserved (last_fp kept)

    # 5. end-to-end through the Engine (memo_bytes): a long linear chain of eager
    #    ArrayProvider nodes — the V2.04 hazard shape — stays bounded and computes right
    define_node("test.gc_src", "GcSrc", outputs=[OutDataset()])
    define_node("test.gc_gen", "GcGen", inputs=[InDataset()], outputs=[OutDataset()])
    SZ, N = 128, 40
    ax = AxisSizes(m=1, t=1, z=1, c=1, y=SZ, x=SZ)

    def c_gc_src(ctx):
        return Dataset(axes=ax, image=ArrayProvider(np.zeros((1, 1, 1, 1, SZ, SZ), np.uint8)))

    def c_gc_gen(ctx):
        k = int(ctx.params["k"]) % 256                     # distinct per node → distinct blobs
        return Dataset(axes=ctx.inputs[0].axes,
                       image=ArrayProvider(np.full((1, 1, 1, 1, SZ, SZ), k, np.uint8)))

    img_bytes = int(np.zeros((1, 1, 1, 1, SZ, SZ), np.uint8).nbytes)   # 16_384
    gc_computes = {"test.gc_src": c_gc_src, "test.gc_gen": c_gc_gen}
    g = Graph(); g.add(NodeInstance("n0", "test.gc_src")); prev = "n0"
    for i in range(1, N + 1):
        nid = f"n{i}"; g.add(NodeInstance(nid, "test.gc_gen", params={"k": i}))
        g.connect(prev, nid); prev = nid

    def _tip_val(ds):
        return int(ds.image.read_region(0, 0, 0, 0, 0, 0, SZ, 0, SZ).ravel()[0])

    # a budget for ~5 rasters vs a 41-node chain → GC must evict the long tail
    e2e_budget = 5 * img_bytes
    eng = Engine(g, computes=gc_computes, memo_bytes=e2e_budget)
    assert _tip_val(eng.pull(prev)) == N % 256                       # correct tip
    assert eng.memo.nbytes <= e2e_budget and eng.memo.evictions > 0  # bounded + GC ran
    # re-pull is correctness-safe under eviction: evicted ancestors force a recompute
    # (the honest memory/recompute trade — identical result), and the memo stays bounded
    # across pulls (no unbounded growth — the whole point).
    assert _tip_val(eng.pull(prev)) == N % 256 and eng.memo.nbytes <= e2e_budget

    # contrast — a budget that fits the whole chain: GC never runs, so a re-pull is fully
    # memoized (the normal C1 cutoff — zero recompute). Proves the GC only trades recompute
    # under genuine pressure and is inert otherwise.
    big = Engine(g, computes=gc_computes, memo_bytes=(N + 2) * img_bytes)
    assert _tip_val(big.pull(prev)) == N % 256
    c0 = big.compute_count
    big.pull(prev)
    assert big.compute_count == c0 and big.memo.evictions == 0
    _ok("memo GC: byte-budget LRU eviction, per-blob refcount, recency, cutoff-safe, "
        "end-to-end bounded long chain (evict-recompute vs fits-fully-memoized)")


# ── lazy pull engine + ReadContext + granularity routing (V2.02 §8 / V2.03) ──

def test_engine() -> None:
    define_node("eng.src", "Src", outputs=[OutDataset()])
    define_node("eng.scale", "Scale", inputs=[InDataset()], outputs=[OutDataset()])
    define_node("eng.sink", "Sink", inputs=[InDataset()], outputs=[OutDataset()])

    calls = {"src": 0, "scale": 0, "sink": 0}

    def c_src(ctx):
        calls["src"] += 1
        return np.array([1.0])

    def c_scale(ctx):
        calls["scale"] += 1
        return np.asarray(ctx.inputs[0]) * (ctx.calib("pixel_size_um") or 1.0)

    def c_sink(ctx):
        calls["sink"] += 1
        return np.asarray(ctx.inputs[0]) + float(ctx.params.get("bias", 0.0))

    computes = {"eng.src": c_src, "eng.scale": c_scale, "eng.sink": c_sink}
    g = Graph()
    g.add(NodeInstance("S", "eng.src"))
    g.add(NodeInstance("K", "eng.scale"))
    g.add(NodeInstance("N", "eng.sink", params={"bias": 10.0}))
    g.connect("S", "K"); g.connect("K", "N")
    eng = Engine(g, computes=computes,
                 meta_seeds={"S": MetaEnvelope(metadata={"pixel_size_um": 2.0})})

    assert np.allclose(eng.pull("N"), 1.0 * 2.0 + 10.0)     # 12.0
    assert calls == {"src": 1, "scale": 1, "sink": 1}       # each computed once
    eng.pull("N")                                           # all memo hits
    assert calls == {"src": 1, "scale": 1, "sink": 1} and eng.memo.hits > 0

    # downstream param change → only the sink recomputes (upstream isolated)
    g.nodes["N"] = NodeInstance("N", "eng.sink", params={"bias": 100.0})
    assert np.allclose(eng.pull("N"), 1.0 * 2.0 + 100.0)    # 102.0
    assert calls == {"src": 1, "scale": 1, "sink": 2}

    # a calibration value the SCALE node READ changes → scale + sink recompute
    eng.reseed_meta({"S": MetaEnvelope(metadata={"pixel_size_um": 5.0})})
    assert np.allclose(eng.pull("N"), 1.0 * 5.0 + 100.0)    # 105.0
    assert calls["scale"] == 2 and calls["sink"] == 3 and calls["src"] == 1

    # a calibration value the scale node did NOT read → precise: no recompute
    eng.reseed_meta({"S": MetaEnvelope(metadata={"pixel_size_um": 5.0, "dt_s": 0.7})})
    eng.pull("N")
    assert calls["scale"] == 2
    _ok("engine: lazy pull, memo hit, precise read-invalidation, downstream isolation")


def test_engine_granularity() -> None:
    ax = AxisSizes(m=1, t=1, z=4, c=1, y=64, x=64)
    define_node("eng.filter", "Filter", inputs=[], outputs=[OutDataset()],
                modes=[DimMode()],
                granularity={"2D": Granularity.WHOLE_PLANE,
                             "3D": Granularity.WHOLE_VOLUME})

    def c_filter(ctx):
        p = ctx.provider
        if ctx.is_volume:                                  # 3D → z-range brick
            return p.get_subvolume(0, 0, 0, 0, 0, ctx.env.axes.z, 0, 0)
        return p.get_tile(0, 0, 0, 0, 0, 0, 0)             # 2D → single-z tile

    prov = SyntheticProvider(ax, tile=32, levels=1)
    seed = {"F": MetaEnvelope(axes=ax)}

    g2 = Graph(); g2.add(NodeInstance("F", "eng.filter", modes={"dim": "2D"}))
    e2 = Engine(g2, computes={"eng.filter": c_filter}, providers={"F": prov}, meta_seeds=seed)
    out2 = e2.pull("F")
    assert out2.ndim == 2 and out2.shape == (32, 32)       # tile path

    g3 = Graph(); g3.add(NodeInstance("F", "eng.filter", modes={"dim": "3D"}))
    e3 = Engine(g3, computes={"eng.filter": c_filter}, providers={"F": prov}, meta_seeds=seed)
    out3 = e3.pull("F")
    assert out3.ndim == 3 and out3.shape == (4, 32, 32)    # subvolume path

    # the 2D/3D lever folds into recipe_hash → distinct memo entries
    assert e2.entry("F").recipe_hash != e3.entry("F").recipe_hash
    _ok("engine: 2D/3D lever routes provider path (tile vs subvolume) + distinct memo keys")


# ── structure producers: CCL / watershed / point schema (V2.02 §7 / V2.03) ────

def test_structure() -> None:
    # connectivity offsets: the toggle-dependent param set
    assert len(connectivity_offsets(2, 4)) == 4 and len(connectivity_offsets(2, 8)) == 8
    for conn, k in ((6, 6), (18, 18), (26, 26)):
        assert len(connectivity_offsets(3, conn)) == k
    try:
        connectivity_offsets(2, 6)
        raise AssertionError("expected invalid-connectivity error")
    except ValueError:
        pass

    # 2D CCL: two diagonally-touching pixels → 2 regions (4-conn) vs 1 (8-conn)
    m2 = np.zeros((5, 5), int); m2[1, 1] = 1; m2[2, 2] = 1
    lab4, t4 = label_components(m2, 4)
    _, t8 = label_components(m2, 8)
    assert t4.n == 2 and t8.n == 1 and t4.z_kind == "plane_index"
    # separated pair: areas + raster-canonical ids (top-left region is id 1)
    m2b = np.zeros((5, 6), int); m2b[0:2, 0:2] = 1; m2b[3:5, 4:6] = 1
    lab, tb = label_components(m2b, 4)
    assert tb.n == 2 and sorted(tb.columns["area"].tolist()) == [4, 4] and lab[0, 0] == 1

    # 3D CCL: two separated cubes; real subpixel centroid z
    m3 = np.zeros((4, 4, 4), int); m3[0:2, 0:2, 0:2] = 1; m3[2:4, 2:4, 2:4] = 1
    _, t3 = label_components(m3, 6)
    assert t3.n == 2 and t3.z_kind == "subpixel"
    assert sorted(t3.columns["area"].tolist()) == [8, 8]
    zc = sorted(t3.columns["z"].tolist())
    assert abs(zc[0] - 0.5) < 1e-9 and abs(zc[1] - 2.5) < 1e-9

    # C6 fast CCL: the scipy.ndimage.label fast path is BYTE-IDENTICAL to the reference
    # pure-numpy flood-fill (same partition; raster-canonical relabel to first-appearance
    # C-order) — raster + id/area/centroid columns match exactly, across connectivities.
    from nodegraph.structure import _label_components_flood
    rng = np.random.default_rng(0)
    for f, conns in ((rng.random((48, 48)) > 0.45, (4, 8)),
                     (rng.random((14, 14, 14)) > 0.4, (6, 26))):
        for conn in conns:
            ls, ts = label_components(f, conn)                    # scipy fast path
            lf, tf = _label_components_flood(np.asarray(f) != 0, conn,
                                             m=0, t=0, c=0, z_index=0, layer=None)
            assert np.array_equal(ls, lf), f"scipy≠flood raster ({f.ndim}D {conn}-conn)"
            assert ts.columns["id"].tolist() == tf.columns["id"].tolist()
            assert ts.columns["area"].tolist() == tf.columns["area"].tolist()
            for ax in ("z", "y", "x"):
                assert np.allclose(ts.columns[ax], tf.columns[ax]), f"{ax} centroid drift"
    # empty mask → no regions, all-background raster (both paths)
    le, te = label_components(np.zeros((6, 6), int), 4)
    assert te.n == 0 and int(le.max()) == 0

    # point table: invariant schema; 2D → plane-index z (never NaN), 3D → subpixel
    p2 = point_table(np.array([[1.5, 2.5], [3.5, 4.5]]), z=7, t=1)
    assert p2.z_kind == "plane_index" and p2.n == 2
    assert set(("id", "m", "t", "c", "z", "y", "x")) <= set(p2.columns)
    assert np.allclose(p2.columns["z"], 7.0) and np.all(np.isfinite(p2.columns["z"]))
    p3 = point_table(np.array([[0.5, 1.0, 2.0]]))
    assert p3.z_kind == "subpixel" and p3.columns["z"][0] == 0.5

    # content hash: stable + data-sensitive
    h = t4.content_hash()
    assert t4.content_hash() == h and t8.content_hash() != h

    if _HAVE_PYARROW:
        rb = t3.to_arrow()
        assert rb.num_rows == 2 and {"id", "area", "z", "y", "x"} <= set(rb.schema.names)
        _ok(f"structure: CCL 2D/3D (scipy fast path ≡ flood-fill, C6) + point schema + "
            f"content-hash + Arrow (pyarrow {_PA_VER})")
    else:
        _ok("structure: CCL 2D/3D (scipy fast path ≡ flood-fill, C6) + point schema + "
            "content-hash (pyarrow SKIPPED)")

    # id-carrying seeded watershed: output ids ARE the marker ids (stable over t)
    if _HAVE_WATERSHED:
        fg = np.zeros((3, 9), int); fg[1, :] = 1
        markers = np.zeros((3, 9), int); markers[1, 0] = 5; markers[1, 8] = 9
        ws = seeded_watershed(fg, markers)
        assert set(np.unique(ws[fg > 0]).tolist()) == {5, 9}
        assert ws[1, 0] == 5 and ws[1, 8] == 9
        _ok("structure: id-carrying seeded watershed splits by marker id")
    else:
        _ok("structure: seeded watershed SKIPPED (scipy/skimage absent)")


# ── structure-bridge execution (the geometric spine, V2.00 §6) ────────────────

def test_bridges() -> None:
    raster = np.array([[1, 1, 0, 2],
                       [1, 1, 0, 2],
                       [0, 0, 0, 0],
                       [3, 3, 3, 0]], dtype=np.int64)
    vox = np.array([[10, 10, 0, 20],
                    [10, 10, 0, 20],
                    [0, 0, 0, 0],
                    [30, 30, 30, 0]], dtype=float)
    # Voxel → Label: mean-in-mask / sum / count
    ids, means = voxel_to_label(vox, raster, "mean")
    assert ids.tolist() == [1, 2, 3] and np.allclose(means, [10, 20, 30])
    assert np.allclose(voxel_to_label(vox, raster, "sum")[1], [40, 40, 90])
    assert np.allclose(voxel_to_label(vox, raster, "count")[1], [4, 2, 3])
    # Label → Voxel: paint-by-label; background stays 0
    painted = label_to_voxel([1, 2, 3], [100, 200, 300], raster)
    assert painted[0, 0] == 100 and painted[0, 3] == 200 and painted[3, 0] == 300
    assert painted[0, 2] == 0.0 and painted[2, 2] == 0.0
    # round-trip: paint the means back, re-reduce → identical
    assert np.allclose(voxel_to_label(label_to_voxel(ids, means, raster), raster, "mean")[1],
                       means)
    # Voxel → Point (nearest) + containing label
    pts = np.array([[0, 0], [0, 3], [3, 1]], dtype=float)     # (y,x)
    assert np.allclose(voxel_to_point(vox, pts, "nearest"), [10, 20, 30])
    assert containing_label(pts, raster).tolist() == [1, 2, 3]
    assert containing_label(np.array([[2, 2]], dtype=float), raster)[0] == 0   # bg
    # Point → Voxel splat (collisions sum)
    sv = point_to_voxel([5, 7], np.array([[0, 0], [0, 0]], dtype=float), (2, 2), "sum")
    assert sv[0, 0] == 12.0
    # Point → Label: reduce points inside each region
    idp, mp = points_in_label([1.0, 2.0, 3.0, 4.0],
                              np.array([[0, 0], [0, 1], [0, 3], [3, 0]], dtype=float),
                              raster, "mean")
    assert idp.tolist() == [1, 2, 3] and np.allclose(mp, [1.5, 3.0, 4.0])
    # a bridge plan still refuses generic execution (needs structure inputs)
    from nodegraph.transfer import execute_transfer, plan_transfer
    from nodegraph.dataset import AttributeLayer, AxisSizes
    try:
        execute_transfer(AttributeLayer(D.VOXEL, "i",
                         np.zeros(AxisSizes(m=1, t=1, z=1, c=1, y=2, x=2).shape_for(D.VOXEL))),
                         plan_transfer(D.VOXEL, D.LABEL), AxisSizes(m=1, t=1, z=1, c=1, y=2, x=2))
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass
    if _HAVE_WATERSHED:                                        # scipy present → linear
        vl = voxel_to_point(vox, np.array([[0.0, 0.5]]), "linear")
        assert abs(vl[0] - 10.0) < 1e-9
    _ok("bridges: voxel↔label (mean/sum/count/paint/round-trip), voxel↔point, point↔label")


# ── fields: deferred per-element expressions (V2.00 §3.3 / V2.02 §9) ───────────

def test_field() -> None:
    ax = AxisSizes(m=1, t=2, z=1, c=1, y=3, x=4)
    fr = np.array([[10.0, 20.0]])                     # Frame (m=1, t=2)
    ds = Dataset(axes=ax).with_layer(D.FRAME, "val", fr)
    fctx = FieldContext(ds, D.FRAME, ax)

    # Const → VirtualArray; a const-only subtree stays virtual (V2.02 §9)
    c = evaluate(Const(5.0), fctx)
    assert isinstance(c, VirtualArray) and c.shape == (1, 2) and np.allclose(np.asarray(c), 5.0)
    cv = evaluate(BinOp("*", Const(2.0), Const(3.0)), fctx)
    assert isinstance(cv, VirtualArray) and np.allclose(np.asarray(cv), 6.0)

    # Attr read + arithmetic + comparison + where
    assert np.allclose(evaluate(Attr(D.FRAME, "val"), fctx), fr)
    expr = BinOp("+", BinOp("*", Attr(D.FRAME, "val"), Const(2.0)), Const(1.0))
    assert np.allclose(evaluate(expr, fctx), fr * 2 + 1)
    assert evaluate(BinOp(">", Attr(D.FRAME, "val"), Const(15.0)), fctx).tolist() == [[False, True]]
    w = evaluate(Where(BinOp(">", Attr(D.FRAME, "val"), Const(15.0)), Const(1.0), Const(0.0)), fctx)
    assert np.allclose(np.asarray(w), [[0.0, 1.0]])

    # Attr transferred across the lattice: Frame val consumed on Voxel → broadcast
    rv = evaluate(Attr(D.FRAME, "val"), FieldContext(ds, D.VOXEL, ax))
    assert rv.shape == ax.shape_for(D.VOXEL)
    for t in range(ax.t):
        assert np.allclose(rv[:, t], fr[0, t])

    # Input binding
    assert np.allclose(evaluate(Input("x"), FieldContext(ds, D.FRAME, ax, inputs={"x": fr})), fr)
    try:
        evaluate(Input("missing"), fctx)
        raise AssertionError("expected unbound-input error")
    except KeyError:
        pass

    # field_expr_hash tracks layer revision; kernel_axes disjoins 2D vs 3D stencils
    h1 = field_expr_hash(expr, ds)
    assert field_expr_hash(expr, ds.with_layer(D.FRAME, "val", fr * 10)) != h1
    assert (field_key(expr, D.FRAME, 0, dataset=ds, kernel_axes=("y", "x"))
            != field_key(expr, D.FRAME, 0, dataset=ds, kernel_axes=("z", "y", "x")))

    # disjoint field cache: same token hits, new token misses
    fc = FieldCache()
    fc.evaluate(expr, fctx, token=1); assert fc.misses == 1
    fc.evaluate(expr, fctx, token=1); assert fc.hits == 1
    fc.evaluate(expr, fctx, token=2); assert fc.misses == 2
    _ok("field: IR eval, VirtualArray, lattice transfer, expr-hash on revision, disjoint key + cache")


# ── track membership + track bridges (temporal identity, V2.00 §6) ────────────

def test_tracks() -> None:
    from nodegraph.structure import TrackMembership
    from nodegraph.bridges import (broadcast_track, gather_by_track,
                                    timepoint_to_members, tracks_per_timepoint)
    from nodegraph.transfer import plan_transfer
    # track 1: labels 10→11→12 over t0,1,2 ; track 2: labels 20→21 over t0,1 (dies at t2)
    mem = TrackMembership(track_id=[1, 1, 1, 2, 2], t=[0, 1, 2, 0, 1],
                          member_id=[10, 11, 12, 20, 21], member_domain=D.LABEL)
    assert mem.n == 5 and mem.track_ids().tolist() == [1, 2] and mem.timepoints().tolist() == [0, 1, 2]
    mids, mvals = [10, 11, 12, 20, 21], [1.0, 2.0, 3.0, 100.0, 200.0]
    # Label → Track: gather members over t and reduce
    tids, means = gather_by_track(mids, mvals, mem, "mean")
    assert tids.tolist() == [1, 2] and np.allclose(means, [2.0, 150.0])
    assert np.allclose(gather_by_track(mids, mvals, mem, "sum")[1], [6.0, 300.0])
    assert np.allclose(gather_by_track(mids, mvals, mem, "count")[1], [3, 2])
    # Track → Label: broadcast a per-track value onto its members
    bm, bv = broadcast_track([1, 2], [2.0, 150.0], mem)
    assert bm.tolist() == [10, 11, 12, 20, 21] and np.allclose(bv, [2, 2, 2, 150, 150])
    # Track → Timepoint: count active tracks per t, and reduce a track attr over them
    tp, cnt = tracks_per_timepoint(mem)
    assert tp.tolist() == [0, 1, 2] and np.allclose(cnt, [2, 2, 1])
    _, red = tracks_per_timepoint(mem, track_ids=[1, 2], track_values=[2.0, 150.0], reducer="mean")
    assert np.allclose(red, [76.0, 76.0, 2.0])
    # Timepoint → Track members: broadcast a per-t value onto each member at t
    tmid, tmv = timepoint_to_members([10.0, 20.0, 30.0], [0, 1, 2], mem)
    assert tmid.tolist() == [10, 11, 12, 20, 21] and np.allclose(tmv, [10, 20, 30, 10, 20])
    # the plan router still marks Label→Track as a (non-generated) bridge plan
    assert not plan_transfer(D.LABEL, D.TRACK).generated
    _ok("tracks: membership + gather/broadcast/per-timepoint/timepoint→members bridges")


# ── constructive Fill/Extract Boundary (Point↔Label spine, V2.03 §4 C4) ───────

def test_boundary() -> None:
    from nodegraph.boundary import boundary_dim, extract_boundary, fill_boundary
    from nodegraph.transfer import bridges as _bridges

    # dimensionality authority = geometry (single-z ⇒ 2D contour; multi-z ⇒ 3D surface)
    sq = np.array([[1, 1], [1, 5], [5, 5], [5, 1]], dtype=float)      # (y,x) corners
    assert boundary_dim(sq) == "2D"
    assert boundary_dim(np.array([[0, 1, 1], [0, 5, 5], [0, 5, 1]], dtype=float)) == "2D"
    assert boundary_dim(np.array([[0, 1, 1], [2, 5, 5]], dtype=float)) == "3D"

    if not _HAVE_SKIMAGE:
        _ok("boundary: geometry dim-authority (fill/extract SKIPPED — skimage absent)")
        return

    # Fill: a square contour → a filled region; Extract: a region → its outline points
    raster = fill_boundary(sq, (7, 7))
    assert raster[3, 3] == 1 and raster[0, 0] == 0 and raster.sum() > 0
    mask = np.zeros((8, 8), dtype=int); mask[2:6, 2:6] = 1
    pts = extract_boundary(mask)
    assert pts.domain is D.POINT and pts.z_kind == "plane_index" and pts.n > 0
    assert "contour_id" in pts.columns

    # the toggle is a consistency ASSERTION, not an override: contradiction → hard error
    try:
        fill_boundary(sq, (7, 7), dim="3D")
        raise AssertionError("expected dim-contradiction error")
    except ValueError:
        pass

    # 3D: multi-z contours fill per-plane (the 2D-ROI-on-3D case fills only its plane)
    zpts = np.array([[0, 1, 1], [0, 1, 4], [0, 4, 4], [0, 4, 1],
                     [2, 1, 1], [2, 1, 4], [2, 4, 4], [2, 4, 1]], dtype=float)
    vol = fill_boundary(zpts, (3, 6, 6))
    assert vol[0].sum() > 0 and vol[2].sum() > 0 and vol[1].sum() == 0
    v = np.zeros((6, 6, 6), dtype=int); v[2:4, 2:4, 2:4] = 1
    surf = extract_boundary(v)
    assert surf.z_kind == "subpixel" and surf.n > 0

    # explicit-node contract (V2.03 §4 C4): Fill/Extract are NOT auto-routed bridges,
    # while the Point↔Label CONTAINMENT bridges remain registered
    names = {b.name for b in _bridges()}
    assert "fill-boundary" not in names and "extract-boundary" not in names
    assert "points-in-label" in names and "containing-label" in names
    _ok("boundary: fill/extract contour↔region, geometry-authority, toggle-assert, not a bridge")


# ── node port: metadata-intelligent PSF + Select Channel + Deconvolve (Phase 3) ─

def test_nodes() -> None:
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES, diffraction_sigmas, gaussian_psf, to_pixels_v2

    # unit conversion incl. the axial unit (V2.03 §2 A5)
    assert to_pixels_v2(1.0, "um", pixel_size_um=0.1) == 10.0
    assert to_pixels_v2(1.0, "um_axial", z_step_um=0.5) == 2.0
    assert abs(to_pixels_v2(520, "nm", pixel_size_um=0.13) - 0.52 / 0.13) < 1e-9
    # PSF derives from optics: 2D → 2 sigmas, 3D → 3; smaller NA ⇒ broader PSF
    s2 = diffraction_sigmas(520, 1.4, 0.1, 0.3, False)
    s3 = diffraction_sigmas(520, 1.4, 0.1, 0.3, True)
    assert len(s2) == 2 and len(s3) == 3
    assert diffraction_sigmas(520, 0.7, 0.1, 0.3, False)[0] > s2[0]
    psf = gaussian_psf(s3)
    assert psf.ndim == 3 and abs(psf.sum() - 1.0) < 1e-9

    if not _HAVE_SKIMAGE:
        _ok("nodes: unit-conv + metadata-PSF (Select/Deconvolve SKIPPED — skimage absent)")
        return

    define_node("io.seed", "Seed", outputs=[OutDataset()])          # trivial source

    # Select Channel (H12): subset a 2-channel dataset to channel 1
    ax2 = AxisSizes(m=1, t=1, z=1, c=2, y=4, x=4)
    vol2 = np.zeros((1, 1, 1, 2, 4, 4)); vol2[..., 0, :, :] = 1.0; vol2[..., 1, :, :] = 9.0
    ds2 = Dataset(axes=ax2, metadata={"channel_emission_nm": [500, 600]}).with_image(ArrayProvider(vol2))
    gsel = Graph()
    gsel.add(NodeInstance("S", "io.seed")); gsel.add(NodeInstance("C", "channel.select", params={"channels": [1]}))
    gsel.connect("S", "C")
    esel = Engine(gsel, computes=COMPUTES, seeds={"S": ds2},
                  meta_seeds={"S": MetaEnvelope(axes=ax2, metadata={"channel_emission_nm": [500, 600]})})
    selds = esel.pull("C")
    assert selds.axes.c == 1
    assert selds.image.read_region(0, 0, 0, 0, 0, 0, 4, 0, 4)[0, 0] == 9.0     # channel 1's data
    assert selds.metadata["channel_emission_nm"] == [600]

    # Deconvolve: end-to-end; 2D vs 3D use different PSF dims + distinct memo keys.
    # Signal on plane 0 only + a broad (low-NA) PSF, so the 3D volumetric PSF couples
    # z-planes (spreads into empty planes) while the 2D per-plane path leaves them empty.
    ax = AxisSizes(m=1, t=1, z=4, c=1, y=8, x=8)
    vol = np.zeros((1, 1, 4, 1, 8, 8)); vol[0, 0, 0, 0, 3:6, 3:6] = 1.0
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3, "objective_na": 0.8, "channel_emission_nm": [520]}
    ds = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(vol))
    seedenv = MetaEnvelope(axes=ax, metadata=optics)

    def deco(dim):
        g = Graph()
        g.add(NodeInstance("S", "io.seed"))
        g.add(NodeInstance("D", "enhance.deconvolve", modes={"dim": dim}, params={"iterations": 2}))
        g.connect("S", "D")
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    e3 = deco("3D"); out3 = e3.pull("D")
    assert isinstance(out3, Dataset) and out3.image.axes.z == 4 and out3.image.axes.y == 8
    e2 = deco("2D"); out2 = e2.pull("D")
    assert out2.image.axes.z == 4                                      # 2D mode also runs
    assert e2.entry("D").recipe_hash != e3.entry("D").recipe_hash      # lever re-keys memo
    dec3 = out3.image.get_region_volume(0, 0, 0, 0, 0, 4, 0, 8, 0, 8)
    assert np.all(np.isfinite(dec3)) and np.all(dec3 >= 0)            # RL invariant
    # the lever changes the resolved footprint + socket set on the flagship spec
    dspec = NODES.get("enhance.deconvolve")
    assert dspec.resolve_granularity({"dim": "2D"}) is Granularity.WHOLE_PLANE
    assert dspec.resolve_granularity({"dim": "3D"}) is Granularity.WHOLE_VOLUME
    assert dspec.resolve_kernel_axes({"dim": "3D"}) == frozenset({"z", "y", "x"})
    a2 = {s.name for s in dspec.active_inputs({"dim": "2D"})}
    a3 = {s.name for s in dspec.active_inputs({"dim": "3D"})}
    assert "z_step_um" in a3 and "z_step_um" not in a2                 # 3D-only axial socket
    _ok("nodes: metadata-PSF, Select Channel (H12), Deconvolve 2D/3D lever end-to-end")


# ── ported catalog: filters + threshold→label→measure pipeline (Phase 3) ──────

def test_catalog() -> None:
    if not _HAVE_SKIMAGE:
        _ok("catalog: SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    ax = AxisSizes(m=1, t=1, z=2, c=1, y=6, x=6)
    img = np.zeros((1, 1, 2, 1, 6, 6))
    img[0, 0, :, 0, 1:3, 1:3] = 5.0            # blob A (both z-planes)
    img[0, 0, :, 0, 4:6, 4:6] = 9.0            # blob B (both z-planes)
    ds = Dataset(axes=ax, metadata={"pixel_size_um": 0.1, "z_step_um": 0.3}).with_image(ArrayProvider(img))
    define_node("io.seed2", "Seed", outputs=[OutDataset()])
    seedenv = MetaEnvelope(axes=ax, metadata={"pixel_size_um": 0.1, "z_step_um": 0.3})

    def eng(nodes, edges):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for s, d in edges:
            g.connect(s, d)
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    # Gamma (pointwise, no lever): max preserved, mid-tones compressed
    eg = eng([("S", "io.seed2", {}), ("G", "enhance.gamma", {"params": {"gamma": 2.0}})],
             [("S", "G")])
    a = eg.pull("G").image.get_region(0, 0, 0, 0, 0, 0, 6, 0, 6)
    assert a.shape == (6, 6) and a[1, 1] < 5.0 and a[4, 4] == 9.0

    # Gaussian blur (toggle): blurs the plane (differs from input)
    eb = eng([("S", "io.seed2", {}),
              ("B", "enhance.gaussian", {"modes": {"dim": "2D"}, "params": {"sigma": 0.2}})],
             [("S", "B")])
    blur = eb.pull("B").image.get_region(0, 0, 0, 0, 0, 0, 6, 0, 6)
    assert not np.allclose(blur, img[0, 0, 0, 0])

    # threshold → label → measure pipeline
    ep = eng([("S", "io.seed2", {}),
              ("T", "analysis.threshold", {"params": {"threshold": 1.0}}),
              ("L", "analysis.label", {"modes": {"dim": "2D"}}),
              ("M", "analysis.measure", {})],
             [("S", "T"), ("T", "L"), ("L", "M")])
    assert ep.pull("T").get(D.VOXEL, "mask") is not None          # threshold → Voxel mask
    lds = ep.pull("L")
    area = lds.get(D.LABEL, "area", layer="labels")
    assert area is not None and len(area.values) == 4            # 2 blobs × 2 planes
    mds = ep.pull("M")
    mi = mds.get(D.LABEL, "mean_intensity", layer="labels")
    assert mi is not None
    means = set(np.round(mi.values).tolist())
    assert 5.0 in means and 9.0 in means                        # per-label mean intensity
    # §7b: label stamped z_kind=plane_index (2D); measure re-emits the same layer and must
    # PRESERVE it, not clobber with the StructureTable default (subpixel).
    assert lds.structure_zkind(D.LABEL, "labels") == "plane_index"
    assert mds.structure_zkind(D.LABEL, "labels") == "plane_index", "measure clobbered z_kind"
    _ok("catalog: gamma (pointwise), gaussian (toggle), threshold→label→measure pipeline "
        "(+ z_kind preserved through measure)")


# ── ported catalog batch: filters / denoise / detection / axis-changing ───────

def test_catalog_ported() -> None:
    if not _HAVE_SKIMAGE:
        _ok("catalog (ported batch): SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    # a z=3 volume with two bright 3×3 blobs on every plane (detectable + filterable)
    ax = AxisSizes(m=1, t=1, z=3, c=1, y=16, x=16)
    img = np.zeros((1, 1, 3, 1, 16, 16), dtype=float)
    for (yy, xx) in [(4, 4), (11, 11)]:
        img[0, 0, :, 0, yy - 1:yy + 2, xx - 1:xx + 2] = 8.0
        img[0, 0, :, 0, yy, xx] = 12.0
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3, "objective_na": 1.0,
              "channel_emission_nm": [520]}
    ds = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(img))
    seedenv = MetaEnvelope(axes=ax, metadata=optics)
    define_node("io.seed3", "Seed", outputs=[OutDataset()])

    def eng(op, *, modes=None, params=None):
        g = Graph()
        g.add(NodeInstance("S", "io.seed3"))
        g.add(NodeInstance("N", op, modes=modes or {}, params=params or {}))
        g.connect("S", "N")
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    def img_of(dset):
        p = dset.image
        return p.get_region_volume(0, 0, 0, 0, 0, p.axes.z, 0, p.axes.y, 0, p.axes.x)

    # every IMAGE→IMAGE filter runs end-to-end, stays finite + same geometry, in 2D & 3D
    filters = ["enhance.median", "enhance.morphology", "enhance.tophat", "enhance.dog",
               "enhance.unsharp", "enhance.tv_denoise", "enhance.wavelet_denoise",
               "enhance.clahe"]
    for op in filters:
        for dim in ("2D", "3D"):
            out = eng(op, modes={"dim": dim}).pull("N")
            arr = img_of(out)
            assert arr.shape == (3, 16, 16), f"{op}/{dim} geometry {arr.shape}"
            assert np.all(np.isfinite(arr)), f"{op}/{dim} non-finite"
        # the 2D/3D lever re-keys the memo distinctly (params fold in the mode state)
        assert (eng(op, modes={"dim": "2D"}).entry("N").recipe_hash
                != eng(op, modes={"dim": "3D"}).entry("N").recipe_hash), f"{op} lever hash"

    # median actually changes the image (a real filter, not a pass-through)
    med = img_of(eng("enhance.median", modes={"dim": "2D"},
                     params={"radius": 0.15}).pull("N"))
    assert not np.allclose(med, img[0, 0, :, 0])

    # wavelet_denoise declares the stack-of-2D trap honestly (never true-3D)
    wspec = NODES.get("enhance.wavelet_denoise")
    assert wspec.supports_true_3d is False and wspec.three_d_fallback == "stack_of_2d"
    assert wspec.resolve_granularity({"dim": "3D"}) is Granularity.WHOLE_PLANE
    # tv_denoise, by contrast, is genuinely volumetric in 3D
    assert NODES.get("enhance.tv_denoise").resolve_granularity({"dim": "3D"}) \
        is Granularity.WHOLE_VOLUME

    # morphology/tophat modes fold into the recipe hash (distinct ops memoize apart)
    assert (eng("enhance.morphology", modes={"op": "erode"}).entry("N").recipe_hash
            != eng("enhance.morphology", modes={"op": "dilate"}).entry("N").recipe_hash)

    # Normalize: percentile → [0,1]; scope is a Mode, NOT the dim lever
    nspec = NODES.get("enhance.normalize")
    assert not nspec.has_dim_lever() and nspec.dim_lever() is None
    for scope in ("plane", "volume", "series"):
        nrm = img_of(eng("enhance.normalize", modes={"scope": scope}).pull("N"))
        assert nrm.min() >= 0.0 and nrm.max() <= 1.0 + 1e-9, f"normalize/{scope} range"
    assert (eng("enhance.normalize", modes={"scope": "plane"}).entry("N").recipe_hash
            != eng("enhance.normalize", modes={"scope": "volume"}).entry("N").recipe_hash)

    # Spot detection → a Point structure layer (both dims find the two blobs)
    for dim, zk in (("2D", "plane_index"), ("3D", "subpixel")):
        sp = eng("detect.spots", modes={"dim": dim},
                 params={"min_radius": 0.05, "max_radius": 0.4, "threshold": 0.05}).pull("N")
        ys = sp.get(D.POINT, "y", layer="spots")
        assert ys is not None and len(ys.values) >= 1, f"spots/{dim} found none"
        assert sp.get(D.POINT, "z", layer="spots") is not None            # invariant schema

    # Z-Project: axis-changing z→1, drops z_step_um + marks z_collapsed, env tracks it
    ez = eng("util.zproject", params={})
    zout = ez.pull("N")
    assert zout.axes.z == 1 and zout.image.axes.z == 1
    assert "z_step_um" not in zout.metadata and zout.metadata.get("z_collapsed") is True
    assert ez.env("N").axes.z == 1                                        # meta_transform pass
    zmax = img_of(zout)[0]                                                # the single plane
    assert zmax.max() == 12.0                                             # max projection

    # Crop: shrink Y,X (2D) — payload + edit-time envelope agree on the new extent
    ec = eng("util.crop", modes={"dim": "2D"},
             params={"y0": 2, "y1": 10, "x0": 3, "x1": 12})
    cout = ec.pull("N")
    assert cout.axes.y == 8 and cout.axes.x == 9 and cout.axes.z == 3
    assert ec.env("N").axes.y == 8 and ec.env("N").axes.x == 9
    # 3D crop also trims Z via the 3D-only sockets
    c3 = eng("util.crop", modes={"dim": "3D"},
             params={"y0": 0, "y1": 8, "x0": 0, "x1": 8, "z0": 1, "z1": 3}).pull("N")
    assert c3.axes.z == 2 and c3.axes.y == 8

    # ── adversarial-review regression guards (2026-07-21 ported-catalog review) ──

    # R1 detect.spots 3D reads z_step_um (anisotropic σ) → memo-fenced on it, and the
    #    axial scale changes results (isotropic-vs-anisotropic must diverge somewhere).
    e3d = eng("detect.spots", modes={"dim": "3D"},
              params={"min_radius": 0.05, "max_radius": 0.4, "threshold": 0.05})
    e3d.pull("N")
    assert "z_step_um" in dict(e3d.entry("N").reads), "spots 3D must read z_step_um"
    e2d = eng("detect.spots", modes={"dim": "2D"},
              params={"min_radius": 0.05, "max_radius": 0.4, "threshold": 0.05})
    e2d.pull("N")
    assert "z_step_um" not in dict(e2d.entry("N").reads), "spots 2D must NOT read z_step_um"

    # R2 crop header (meta_transform env) == payload axes for one-sided & out-of-range
    #    bounds (previously span() ignored single endpoints and never clamped).
    for params, exp_y in [({"y0": 2}, 14), ({"y1": 10}, 10),
                          ({"y0": 2, "y1": 100}, 14), ({"y0": -5, "y1": 10}, 10)]:
        crop_e = eng("util.crop", modes={"dim": "2D"}, params=params)
        cp = crop_e.pull("N")
        assert cp.axes.y == exp_y, f"crop {params} payload y={cp.axes.y} != {exp_y}"
        assert crop_e.env("N").axes.y == exp_y, \
            f"crop {params} header y={crop_e.env('N').axes.y} != payload {exp_y}"

    # R3 wavelet_denoise: a flat/blank plane comes through finite (no BayesShrink NaN)
    #    and untouched, while a textured plane in the same stack denoises finitely.
    yy, xx = np.mgrid[0:16, 0:16]
    wimg = np.zeros((1, 1, 2, 1, 16, 16), dtype=float)          # plane 0 flat/blank
    wimg[0, 0, 1, 0] = (np.sin(yy * 0.7) + np.cos(xx * 0.5) + 2.0) * 3.0   # plane 1 textured
    dsf = Dataset(axes=AxisSizes(m=1, t=1, z=2, c=1, y=16, x=16),
                  metadata=optics).with_image(ArrayProvider(wimg))
    gf = Graph()
    gf.add(NodeInstance("S", "io.seed3"))
    gf.add(NodeInstance("W", "enhance.wavelet_denoise", modes={"dim": "2D"}))
    gf.connect("S", "W")
    wout = Engine(gf, computes=COMPUTES, seeds={"S": dsf},
                  meta_seeds={"S": MetaEnvelope(axes=dsf.axes, metadata=optics)}).pull("W")
    warr = wout.image.get_region_volume(0, 0, 0, 0, 0, 2, 0, 16, 0, 16)
    assert np.all(np.isfinite(warr)), "wavelet flat plane produced NaN"
    assert np.all(warr[0] == 0.0), "wavelet flat plane should pass through untouched"

    _ok("catalog (ported): 8 filters (2D/3D lever), TV/wavelet 3D honesty, normalize "
        "scope Mode, spot→Points, z-project + crop meta_transform; review fixes R1–R3")


# ── ported catalog batch 2 + fusion reducers + C2 bridge execution ────────────

def test_fusion_reducers() -> None:
    from nodegraph.reducers import REDUCERS, is_tileable, reduce as _r
    # sigma_clip is robust: a lone extreme outlier is rejected (mean/std would keep it)
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert abs(float(_r(x, (0,), "sigma_clip")) - 2.5) < 1e-6      # 100 dropped → mean(1..4)
    assert float(_r(x, (0,), "trimmed_mean")) == 3.0              # drop min+max → mean(2,3,4)
    # both are whole-domain (NOT tile-reducible — like median)
    assert not is_tileable("sigma_clip") and not is_tileable("trimmed_mean")
    # multi-axis + a constant population (MAD==0 → threshold 0, a truly constant cell
    # is untouched, never emptied)
    flat = np.full((3, 4), 7.0)
    assert np.allclose(_r(flat, (0,), "sigma_clip"), 7.0)
    # review: MAD==0 on a flat/dark background + a lone cosmic ray STILL rejects it
    # (the old std-fallback re-admitted it)
    assert float(_r(np.array([0.0, 0.0, 0.0, 0.0, 50.0]), (0,), "sigma_clip")) == 0.0
    # review: trimmed_mean is NaN-aware — the NaN is excluded (not kept as a "high"
    # value), the real high outlier is trimmed, and a low-count cell never goes NaN
    assert abs(float(_r(np.array([1.0, 2.0, 3.0, 100.0, np.nan]), (0,), "trimmed_mean"))
               - 2.5) < 1e-9
    assert np.isfinite(_r(np.array([5.0, np.nan]), (0,), "trimmed_mean"))   # k=1 ≤ 2·trim
    _ok("fusion reducers: sigma-clip (MAD-robust, rejects cosmic ray on flat-dark); "
        "trimmed mean (NaN-aware); both whole-domain")


def test_transfer_bridge_exec() -> None:
    from nodegraph.transfer import Carrier, execute_bridge_plan, plan_transfer
    from nodegraph.bridges import gather_by_track, voxel_to_label
    from nodegraph.structure import TrackMembership

    raster = np.array([[1, 1, 0, 2], [1, 1, 0, 2], [0, 0, 0, 0], [3, 3, 0, 2]])
    vox = np.array([[10, 10, 0, 20], [10, 10, 0, 20], [0, 0, 0, 0], [30, 30, 0, 20]],
                   dtype=float)
    mem = TrackMembership(track_id=[1, 2, 2], t=[0, 0, 1], member_id=[1, 2, 3])

    # C2: a routed Voxel→Track plan (Voxel→Label→Track) executes as one call and equals
    # the hand-chained bridges.
    plan = plan_transfer(D.VOXEL, D.TRACK, reducer="mean")
    assert not plan.generated and len(plan.steps) == 2
    res = execute_bridge_plan(Carrier(D.VOXEL, array=vox), plan,
                              label_raster=raster, membership=mem)
    ids, means = voxel_to_label(vox, raster, "mean")
    tids, tvals = gather_by_track(ids, means, mem, "mean")
    assert res.domain is D.TRACK and res.ids.tolist() == tids.tolist()
    assert np.allclose(res.values, tvals) and res.values.tolist() == [10.0, 25.0]

    # Label→Frame reduce-in-frame → one scalar
    fr = execute_bridge_plan(Carrier(D.LABEL, ids=ids, values=means),
                             plan_transfer(D.LABEL, D.FRAME, reducer="mean"))
    assert fr.domain is D.FRAME and abs(float(fr.values) - float(np.mean(means))) < 1e-9

    # a hop missing its structure input is a clear hard error, not a wrong result
    try:
        execute_bridge_plan(Carrier(D.VOXEL, array=vox), plan, membership=mem)
        raise AssertionError("expected a missing-label_raster error")
    except ValueError as e:
        assert "label_raster" in str(e)

    # review: the generated (pure-lattice) branch honors the supplied `axes` — a
    # coarse→fine broadcast sizes to the real geometry (was a size-1 AxisSizes()).
    fplan = plan_transfer(D.FRAME, D.VOXEL)
    assert fplan.generated
    gax = AxisSizes(m=1, t=1, z=2, c=1, y=3, x=3)
    bc = execute_bridge_plan(Carrier(D.FRAME, array=np.array([[5.0]])), fplan, axes=gax)
    assert bc.domain is D.VOXEL and bc.array.shape == gax.shape_for(D.VOXEL)
    assert np.all(bc.array == 5.0)
    _ok("transfer C2: routed Voxel→Track (Voxel→Label→Track) chains as one call; "
        "Label→Frame; generated broadcast honors axes; missing-input guard")


def test_catalog_ported2() -> None:
    if not _HAVE_SKIMAGE:
        _ok("catalog (batch 2): SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    ax = AxisSizes(m=1, t=3, z=3, c=1, y=16, x=16)
    yy, xx = np.mgrid[0:16, 0:16]
    base = np.sin(yy * 0.6) + np.cos(xx * 0.4) + 2.0            # full-plane texture
    img = np.zeros((1, 3, 3, 1, 16, 16), dtype=float)
    for t in range(3):
        img[0, t, :, 0] = base * 5.0 + t * 0.5
    img[0, :, :, 0, 6:10, 6:10] += 30.0                        # a bright blob region
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3, "dt_s": 1.0, "objective_na": 1.0,
              "channel_emission_nm": [520]}
    ds = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(img))
    define_node("io.seed4", "Seed", outputs=[OutDataset()])
    seedenv = MetaEnvelope(axes=ax, metadata=optics)

    def eng(op, *, modes=None, params=None, chain=()):
        g = Graph(); g.add(NodeInstance("S", "io.seed4")); prev = "S"
        for i, (cop, cmodes, cparams) in enumerate(chain):
            nid = f"U{i}"
            g.add(NodeInstance(nid, cop, modes=cmodes or {}, params=cparams or {}))
            g.connect(prev, nid); prev = nid
        g.add(NodeInstance("N", op, modes=modes or {}, params=params or {}))
        g.connect(prev, "N")
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    def imv(dset):
        p = dset.image
        return p.get_region_volume(0, 0, 0, 0, 0, p.axes.z, 0, p.axes.y, 0, p.axes.x)

    # new IMAGE→IMAGE filters: finite + geometry-preserving in both dims; lever re-keys
    for op in ("enhance.morphological_gradient", "enhance.bilateral", "enhance.nlm"):
        for dim in ("2D", "3D"):
            arr = imv(eng(op, modes={"dim": dim}).pull("N"))
            assert arr.shape == (3, 16, 16) and np.all(np.isfinite(arr)), f"{op}/{dim}"
        assert (eng(op, modes={"dim": "2D"}).entry("N").recipe_hash
                != eng(op, modes={"dim": "3D"}).entry("N").recipe_hash), f"{op} lever"
    # bilateral is honestly stack-of-2D; nlm is genuinely volumetric
    assert NODES.get("enhance.bilateral").resolve_granularity({"dim": "3D"}) \
        is Granularity.WHOLE_PLANE
    assert NODES.get("enhance.nlm").resolve_granularity({"dim": "3D"}) \
        is Granularity.WHOLE_VOLUME

    # spot detection: LoG/DoG × bright/dark all run; method/polarity fold into the hash
    for method in ("log", "dog"):
        for pol in ("bright", "dark"):
            sp = eng("detect.spots", modes={"dim": "2D", "method": method, "polarity": pol},
                     params={"min_radius": 0.05, "max_radius": 0.4, "threshold": 0.08})
            assert sp.pull("N").get(D.POINT, "y", layer="spots") is not None
    assert (eng("detect.spots", modes={"method": "log"}).entry("N").recipe_hash
            != eng("detect.spots", modes={"method": "dog"}).entry("N").recipe_hash)

    # review R: a FLAT/blank volume has NO spots of EITHER polarity — the dark-polarity
    # inversion must not turn the flat guard's zeros into a constant-1 flood, and the 3D
    # LoG axial σ must not clamp sub-voxel into a spurious uniform response.
    flat_ds = Dataset(axes=ax, metadata=optics).with_image(
        ArrayProvider(np.zeros((1, 3, 3, 1, 16, 16))))
    gflat = Graph(); gflat.add(NodeInstance("S", "io.seed4"))
    gflat.add(NodeInstance("N", "detect.spots", modes={"dim": "3D", "polarity": "dark"},
                           params={"min_radius": 0.05, "max_radius": 0.4, "threshold": 0.05}))
    gflat.connect("S", "N")
    fspots = Engine(gflat, computes=COMPUTES, seeds={"S": flat_ds},
                    meta_seeds={"S": seedenv}).pull("N").get(D.POINT, "y", layer="spots")
    assert fspots is None or len(fspots.values) == 0, "flat volume should yield no spots"

    thr = ("analysis.threshold", None, {"threshold": 20.0})

    # EDT: a µm distance field on the mask (finite, positive inside)
    for dim in ("2D", "3D"):
        dist = eng("analysis.edt", modes={"dim": dim}, chain=(thr,)).pull("N").get(D.VOXEL, "distance")
        assert dist is not None and np.all(np.isfinite(dist.values)) and dist.values.max() > 0

    # Segmentation, watershed method (V2.12 - `analysis.watershed` folded in here):
    # a global-unique Label raster + region table, splitting the mask it is pointed at
    w = eng("analysis.segment", modes={"dim": "2D", "method": "watershed"},
            params={"mask": "mask", "name": "watershed"}, chain=(thr,)).pull("N")
    assert int(w.get(D.VOXEL, "watershed").values.max()) >= 1
    warea = w.get(D.LABEL, "area", layer="watershed")
    assert warea is not None and len(set(range(1, len(warea.values) + 1)))  # ids present

    # Resample: axes + inverse pixel size mirror the meta_transform (header==payload)
    for dim, params, ey in (("2D", {"scale_xy": 0.5}, 8), ("2D", {"scale_xy": 2.0}, 32)):
        e = eng("util.resample", modes={"dim": dim}, params=params)
        o = e.pull("N")
        assert o.axes.y == ey and e.env("N").axes.y == ey
        assert abs(o.metadata["pixel_size_um"] - 0.1 / params["scale_xy"]) < 1e-9
    e3 = eng("util.resample", modes={"dim": "3D"}, params={"scale_xy": 1.0, "scale_z": 2.0})
    o3 = e3.pull("N")
    assert o3.axes.z == 6 and e3.env("N").axes.z == 6
    assert abs(o3.metadata["z_step_um"] - 0.15) < 1e-9

    # Stack: T→1 with each combiner; dt_s dropped; env tracks it
    for meth in ("mean", "median", "sigma_clip", "trimmed_mean", "max"):
        es = eng("util.stack", modes={"method": meth})
        so = es.pull("N")
        assert so.axes.t == 1 and es.env("N").axes.t == 1 and "dt_s" not in so.metadata

    # Drift: geometry preserved; the estimated shift is stored as Frame attributes
    dr = eng("align.drift").pull("N")
    assert dr.axes.t == 3 and dr.axes.y == 16
    assert dr.get(D.FRAME, "drift_y") is not None and dr.get(D.FRAME, "drift_x") is not None

    # Measure: multi-stat columns all present and id-aligned
    md = eng("analysis.measure", chain=(thr, ("analysis.label", {"dim": "2D"}, {}))).pull("N")
    for col in ("mean_intensity", "max_intensity", "min_intensity", "area"):
        assert md.get(D.LABEL, col, layer="labels") is not None, f"measure {col}"

    # Source-layer + output-name sockets (2026-07-28). These were params the compute read
    # with no socket declared, so from the GUI the whole segmentation chain was pinned to
    # one hardcoded layer name and a graph could not carry two masks. Chain them with
    # NON-default names end to end: threshold→label→watershed→edt each reading the
    # previous node's renamed output proves the redirects actually compose.
    chain_named = eng("analysis.edt", modes={"dim": "2D"},
                      params={"mask": "m2", "name": "dist2"},
                      chain=(("analysis.threshold", None,
                              {"threshold": 20.0, "name": "m2"}),)).pull("N")
    assert chain_named.get(D.VOXEL, "m2") is not None, "threshold `name` socket"
    assert chain_named.get(D.VOXEL, "dist2") is not None, "edt `mask`+`name` sockets"
    assert chain_named.get(D.VOXEL, "distance") is None, "no stale default edt layer"
    lab_named = eng("analysis.label", modes={"dim": "2D"},
                    params={"mask": "m2", "name": "regions2"},
                    chain=(("analysis.threshold", None,
                            {"threshold": 20.0, "name": "m2"}),)).pull("N")
    assert lab_named.get(D.VOXEL, "regions2") is not None, "label `mask`+`name` sockets"
    assert lab_named.get(D.LABEL, "area", layer="regions2") is not None, "label table follows"
    wat_named = eng("analysis.segment", modes={"dim": "2D", "method": "watershed"},
                    params={"mask": "m2", "name": "ws2"},
                    chain=(("analysis.threshold", None,
                            {"threshold": 20.0, "name": "m2"}),)).pull("N")
    assert wat_named.get(D.VOXEL, "ws2") is not None, "segment `mask`+`name` sockets"
    # pointing a node at a layer that does not exist must RAISE, not silently no-op
    try:
        eng("analysis.label", modes={"dim": "2D"}, params={"mask": "nope"},
            chain=(thr,)).pull("N")
        raise AssertionError("a missing source layer must raise")
    except (ValueError, KeyError):
        pass
    # `stats` selects the measured columns (it too had no socket, so the set was fixed)
    md2 = eng("analysis.measure", params={"stats": "mean,median"},
              chain=(thr, ("analysis.label", {"dim": "2D"}, {}))).pull("N")
    assert md2.get(D.LABEL, "median_intensity", layer="labels") is not None, "stats picks median"
    assert md2.get(D.LABEL, "max_intensity", layer="labels") is None, "stats excludes max"
    try:
        eng("analysis.measure", params={"stats": "bogus"},
            chain=(thr, ("analysis.label", {"dim": "2D"}, {}))).pull("N")
        raise AssertionError("an unknown stat must raise")
    except ValueError as exc:
        assert "unknown measure stat" in str(exc)

    # Extract Boundary: a label raster → boundary Points
    lbl2 = ("analysis.label", {"dim": "2D"}, {})
    eb = eng("analysis.extract_boundary", modes={"dim": "2D"}, chain=(thr, lbl2)).pull("N")
    ybp = eb.get(D.POINT, "y", layer="labels_boundary")
    assert ybp is not None and len(ybp.values) >= 1
    # ...and the 3D surface-vertex path, which had no coverage at all (which is how the
    # missing sockets below survived: only the all-defaults 2D pull was ever exercised)
    eb3 = eng("analysis.extract_boundary", modes={"dim": "3D"},
              chain=(thr, ("analysis.label", {"dim": "3D"}, {}))).pull("N")
    p3 = eb3.get(D.POINT, "y", layer="labels_boundary")
    assert p3 is not None and len(p3.values) >= 1, "3D marching-cubes vertices"
    assert eb3.get(D.POINT, "z", layer="labels_boundary") is not None, "3D needs a z column"

    # REGRESSION (2026-07-28): the compute always read a `labels` source param and an
    # output-name param, but register_node declared ONLY InDataset() — so neither had a
    # socket and both were stuck on their defaults in the GUI, making it impossible to
    # outline a Label raster not named "labels".
    #
    # READ THIS BEFORE TRUSTING THE ASSERTIONS BELOW: an UNDECLARED param still reaches
    # ctx.params (the engine does not filter params against the socket list), so the two
    # layer-redirect pulls and the re-key check below all PASS against the pre-fix node —
    # measured, not assumed. They are coverage of the sockets' semantics, NOT guards. The
    # defect was reachability from the GUI, which builds its widgets from `inputs`, so the
    # only assertions that actually fail against the old node are the `name` rename, the
    # domain rail, and the socket-set check at the end of this block. Do not delete those
    # three thinking the behavioural pulls cover them.
    lblR = ("analysis.label", {"dim": "2D"}, {"name": "regions"})
    ebr = eng("analysis.extract_boundary", modes={"dim": "2D"},
              params={"labels": "regions"}, chain=(thr, lblR)).pull("N")
    assert ebr.get(D.POINT, "y", layer="regions_boundary") is not None, \
        "the `labels` socket must redirect the source layer"
    # `name` names the output; empty `name` keeps auto-deriving f"{labels}_boundary"
    ebn = eng("analysis.extract_boundary", modes={"dim": "2D"},
              params={"labels": "regions", "name": "outline"}, chain=(thr, lblR)).pull("N")
    assert ebn.get(D.POINT, "y", layer="outline") is not None, "the `name` socket"
    assert ebn.get(D.POINT, "y", layer="regions_boundary") is None, "no stale default layer"
    # `labels` folds into the recipe hash, so re-pointing the node re-keys the memo.
    # Chain BOTH label nodes so either choice resolves (layers accumulate on the Dataset).
    _both = (thr, lbl2, lblR)
    _hR = eng("analysis.extract_boundary", modes={"dim": "2D"},
              params={"labels": "regions"}, chain=_both).entry("N").recipe_hash
    _hL = eng("analysis.extract_boundary", modes={"dim": "2D"},
              params={"labels": "labels"}, chain=_both).entry("N").recipe_hash
    assert _hR != _hL, "`labels` must fold into recipe_hash"
    # The two REAL guards (both fail against the pre-fix node). The domain rail was empty
    # despite an identical contract to detect.spots, so the GUI drew no chips and no red
    # missing-domain validation here; and the socket set is the only thing that can catch
    # a param the compute reads but nobody can reach.
    _ebspec = NODES.get("analysis.extract_boundary")
    assert _ebspec.reads_domains == frozenset({D.VOXEL}), "domain rail: reads VOXEL"
    assert _ebspec.adds_domains == frozenset({D.POINT}), "domain rail: adds POINT"
    assert {s.name for s in _ebspec.inputs} == {"data", "labels", "name"}, \
        "every param the compute reads needs a socket, or the GUI cannot set it"

    _ok("catalog (batch 2): morph-gradient/bilateral/nlm, LoG+DoG × bright/dark, EDT, "
        "watershed, resample+stack+drift meta, multi-stat measure, extract-boundary "
        "(2D contours + 3D surface verts; labels/name sockets redirect + re-key; rail)")


# ── ported catalog batch 3: threshold methods, multi-otsu, transfer-domain ────

def test_channel_split() -> None:
    """``channel.split`` is a domain-transparent pass-through: its ``out`` socket carries
    the full multi-channel bundle unchanged (axes, pixels, and channel emission all
    preserved). Per-channel fan-out is a GUI/graph-build concern (nodelab materializes
    each ``chK`` tap into a ``channel.select``), so here we pin the node contract itself."""
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    ax = AxisSizes(m=1, t=1, z=1, c=3, y=4, x=4)
    img = np.zeros((1, 1, 1, 3, 4, 4), dtype=float)
    img[0, 0, 0, 0] = 1.0; img[0, 0, 0, 1] = 2.0; img[0, 0, 0, 2] = 3.0   # per-channel
    seedenv = MetaEnvelope(axes=ax, metadata={"channel_emission_nm": [461, 509, 610]})
    ds = Dataset(axes=ax, metadata=dict(seedenv.metadata)).with_image(ArrayProvider(img))

    g = Graph()
    g.add(NodeInstance("S", "io.seedSplit"))
    g.add(NodeInstance("P", "channel.split"))
    g.connect("S", "P")
    define_node("io.seedSplit", "Seed", outputs=[OutDataset()])
    eng = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    out = eng.pull("P")
    assert out.axes.c == 3, "split must pass all channels through 'out'"
    for ch, val in enumerate((1.0, 2.0, 3.0)):
        plane = out.image.get_region(0, 0, 0, 0, ch, 0, 4, 0, 4)
        assert np.allclose(plane, val), f"channel {ch} altered by split"
    # envelope passes through unchanged (no meta_transform)
    assert eng.env("P").axes.c == 3
    assert list(eng.env("P").metadata.get("channel_emission_nm")) == [461, 509, 610]
    _ok("channel.split: pass-through preserves channels + emission (tap fan-out is GUI)")


def test_reroute() -> None:
    """``rr.reroute`` is an identity pass-through (a GUI wire-routing hop): pixels, axes
    and calibration all pass through unchanged, and it composes transparently in a chain.
    Hidden from the palette (``rr.`` prefix) but a real engine node so meta-propagation /
    memoization need no special-casing."""
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    ax = AxisSizes(m=1, t=1, z=2, c=1, y=4, x=4)
    img = np.arange(1 * 1 * 2 * 1 * 4 * 4, dtype=float).reshape(1, 1, 2, 1, 4, 4)
    optics = {"pixel_size_um": 0.2, "channel_emission_nm": [488]}
    ds = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(img))
    seedenv = MetaEnvelope(axes=ax, metadata=optics)
    define_node("io.seedRR", "Seed", outputs=[OutDataset()])

    # S → R1 → R2 : two reroutes in series must not alter the data
    g = Graph()
    g.add(NodeInstance("S", "io.seedRR"))
    g.add(NodeInstance("R1", "rr.reroute"))
    g.add(NodeInstance("R2", "rr.reroute"))
    g.connect("S", "R1"); g.connect("R1", "R2")
    eng = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    out = eng.pull("R2")
    got = out.image.get_region_volume(0, 0, 0, 0, 0, ax.z, 0, ax.y, 0, ax.x)
    assert np.array_equal(got, img[0, 0, :, 0]), "reroute altered the pixels"
    # envelope + calibration pass through unchanged (no meta_transform)
    assert eng.env("R2").axes == ax
    assert list(eng.env("R2").metadata.get("channel_emission_nm")) == [488]
    # TILEABLE, no dim lever, hidden prefix
    spec = NODES.get("rr.reroute")
    assert spec.resolve_granularity({}) is Granularity.TILEABLE
    assert not spec.has_dim_lever() and spec.op_key.startswith("rr.")
    _ok("rr.reroute: identity pass-through (pixels/axes/calib preserved, composes)")


def test_catalog3() -> None:
    if not _HAVE_SKIMAGE:
        _ok("catalog (batch 3): SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    # a 3-population image: 0 (bg), 50 (mid), 100 (hi)
    ax = AxisSizes(m=1, t=1, z=1, c=1, y=4, x=4)
    img = np.zeros((1, 1, 1, 1, 4, 4))
    img[0, 0, 0, 0, :2, :2] = 0.0; img[0, 0, 0, 0, :2, 2:] = 50.0
    img[0, 0, 0, 0, 2:, :] = 100.0
    ds = Dataset(axes=ax).with_image(ArrayProvider(img))
    define_node("io.seedC3", "Seed", outputs=[OutDataset()])
    env = MetaEnvelope(axes=ax)

    def eng(op, *, modes=None, params=None):
        g = Graph(); g.add(NodeInstance("S", "io.seedC3"))
        g.add(NodeInstance("N", op, modes=modes or {}, params=params or {}))
        g.connect("S", "N")
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env})

    # histogram threshold methods: each separates bg from signal; each memoizes distinctly
    hashes = set()
    for method in ("otsu", "li", "yen", "triangle", "mean"):
        e = eng("analysis.threshold", modes={"method": method})
        mask = e.pull("N").get(D.VOXEL, "mask")
        assert mask is not None and set(np.unique(mask.values).tolist()) == {0, 1}, method
        hashes.add(e.entry("N").recipe_hash)
    assert len(hashes) == 5, "each threshold method must memoize distinctly"
    # the `threshold` socket is FIXED-only: a histogram method derives its own cut and
    # never reads it, so it must not be offered under otsu/li/yen/triangle/mean
    _tvis = lambda st: {i.name for i in NODES.get("analysis.threshold").active_inputs(st)}
    assert "threshold" in _tvis({"method": "fixed"})
    assert all("threshold" not in _tvis({"method": mm})
               for mm in ("otsu", "li", "yen", "triangle", "mean"))

    # multi-otsu → a 3-class index raster (0,1,2)
    cl = eng("analysis.multiotsu", params={"classes": 3}).pull("N").get(D.VOXEL, "classes")
    assert cl is not None and set(np.unique(cl.values).tolist()) == {0, 1, 2}

    # adaptive local threshold → a Voxel mask (per-plane; finite, binary)
    lm = eng("analysis.threshold_local", params={"block_size": 0.3}).pull("N").get(D.VOXEL, "mask")
    assert lm is not None and set(np.unique(lm.values).tolist()) <= {0, 1}

    # transfer domain: a Voxel attribute → Frame (mean over z,c,y,x)
    vox = np.arange(16, dtype=float).reshape(1, 1, 1, 1, 4, 4)
    dst = Dataset(axes=ax).with_layer(D.VOXEL, "vals", vox)
    def _xfer(*, modes=None, params=None):
        g = Graph(); g.add(NodeInstance("S", "io.seedC3"))
        g.add(NodeInstance("T", "transform.transfer_domain",
                           modes=modes or {}, params=params or {"attr": "vals"}))
        g.connect("S", "T")
        return Engine(g, computes=COMPUTES, seeds={"S": dst}, meta_seeds={"S": env})

    # from/to/reducer are MODES as of 2026-07-28 (they had no sockets at all before, so
    # the node could only ever do voxel→frame/mean from the GUI).
    ft = _xfer(modes={"from_domain": "voxel", "to_domain": "frame", "reducer": "mean"}
               ).pull("T")
    fr = ft.get(D.FRAME, "vals")
    assert fr is not None and fr.values.shape == ax.shape_for(D.FRAME)
    assert abs(float(fr.values[0, 0]) - float(vox.mean())) < 1e-9   # reduced mean
    # A NON-DEFAULT reducer must actually change the number. The old test only ever used
    # values equal to the defaults, so it could not tell a working lever from an ignored
    # one — which is exactly how the missing sockets survived here.
    fmax = _xfer(modes={"from_domain": "voxel", "to_domain": "frame", "reducer": "max"}
                 ).pull("T").get(D.FRAME, "vals")
    assert abs(float(fmax.values[0, 0]) - float(vox.max())) < 1e-9, "reducer mode is live"
    assert (_xfer(modes={"reducer": "max"}).entry("T").recipe_hash
            != _xfer(modes={"reducer": "mean"}).entry("T").recipe_hash), "mode re-keys"
    # A headless caller predating the Mode conversion is REFUSED, not silently run on the
    # defaults. A silent fallback is not implementable: the engine hands the compute the
    # RESOLVED mode state, so "unset" and "explicitly set to the default" are the same
    # value — honouring the param would mean guessing which one happened.
    try:
        _xfer(params={"attr": "vals", "reducer": "max"}).pull("T")
        raise AssertionError("the legacy param form must be refused, not ignored")
    except ValueError as exc:
        assert "are Modes now" in str(exc)
    # charter: a pure broadcast ignores `reducer`, so a non-default one is REFUSED
    try:
        _xfer(modes={"from_domain": "frame", "to_domain": "voxel", "reducer": "max"}
              ).pull("T")
        raise AssertionError("a no-reduce transfer must refuse a non-default reducer")
    except ValueError as exc:
        assert "drops no axis" in str(exc)
    # structure domains are absent from the dropdowns AND refused if hand-authored
    try:
        _xfer(modes={"from_domain": "label", "to_domain": "frame"}).pull("T")
        raise AssertionError("structure-domain transfer must be refused")
    except ValueError as exc:
        assert "structure bridge" in str(exc)
    _spec = NODES.get("transform.transfer_domain")
    assert {m.name for m in _spec.modes} == {"from_domain", "to_domain", "reducer"}
    assert "label" not in dict((m.name, m.choices) for m in _spec.modes)["from_domain"]
    _ok("catalog (batch 3): threshold methods (otsu/li/yen/triangle/mean, distinct keys); "
        "multi-otsu class raster; transfer-domain Voxel→Frame reduce (from/to/reducer are "
        "live Modes + legacy params; non-default reducer changes the value and re-keys; "
        "no-reduce + structure-domain refusals)")


# ── Phase 4a: Repeat / Simulation zones (unroll + revision-fold) ──────────────

def test_zones() -> None:
    if not _HAVE_SKIMAGE:
        _ok("zones: SKIPPED (needs enhance.gamma → scipy/skimage)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.zones import Zone, iter_id, unroll, zone_output, assert_zone_pure

    define_node("io.seedZ", "Seed", outputs=[OutDataset()])

    # ── Repeat zone: iterate Gamma N times ──────────────────────────────────
    ax = AxisSizes(m=1, t=1, z=1, c=1, y=4, x=4)
    grad = (np.arange(16, dtype=float).reshape(4, 4) + 1.0)      # 1..16 (non-flat)
    img = grad.reshape(1, 1, 1, 1, 4, 4)
    ds = Dataset(axes=ax).with_image(ArrayProvider(img))
    env = MetaEnvelope(axes=ax)

    def repeat_graph(n):
        g = Graph()
        g.add(NodeInstance("S", "io.seedZ"))
        g.add(NodeInstance("RIN", "zone.repeat_in"))
        g.add(NodeInstance("G", "enhance.gamma", params={"gamma": 2.0}))
        g.add(NodeInstance("ROUT", "zone.repeat_out"))
        g.connect("S", "RIN")                       # external seed → In (iteration 0)
        g.connect("RIN", "G")
        g.connect("G", "ROUT")
        g.connect("ROUT", "RIN", kind="back")       # feedback (Out → In)
        z = Zone("Z", "repeat", "RIN", "ROUT", body=frozenset({"G"}), iterations=n)
        return g, z

    g, z = repeat_graph(3)
    flat = unroll(g, [z])
    g.topo_order()                    # the zoned graph orders fine (back-edge is skipped)
    zout = zone_output(z)             # "ROUT#Z@2"

    def eng(memo=None):
        return Engine(flat, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env},
                      memo=memo)
    out = eng().pull(zout)
    got = out.image.get_region(0, 0, 0, 0, 0, 0, 4, 0, 4)
    mx = grad.max()
    expected = (grad / mx) ** (2.0 ** 3) * mx                    # γ=2 applied 3× → exp 2³=8
    assert np.allclose(got, expected), "Repeat zone must apply the body N times"

    # revision-fold caching: re-pull on a SHARED memo recomputes nothing (the chain's
    # keys are stable — iteration i folds iteration i-1's revision, which is unchanged)
    memo = Memo()
    eng(memo).pull(zout)
    e2 = eng(memo)
    e2.pull(zout)
    assert e2.compute_count == 0, "an unchanged zone must be fully memo-cached on re-pull"
    # debug-verify: the pure Gamma body is deterministic
    assert_zone_pure(lambda: eng(), zout)

    # ── Simulation zone: feedback ACCUMULATES across iterations ──────────────
    def _accum(ctx):                                   # out = in + 1, per iteration
        prov = ctx.inputs[0].image
        a = prov.axes
        o = prov.get_region_volume(0, 0, 0, 0, 0, a.z, 0, a.y, 0, a.x).astype(float) + 1.0
        return ctx.inputs[0].with_image(ArrayProvider(o.reshape(1, 1, a.z, 1, a.y, a.x)))

    define_node("test.accumulate", "Accumulate", inputs=[InDataset()], outputs=[OutDataset()],
                granularity=Granularity.TILEABLE)
    merged = {**COMPUTES, "test.accumulate": _accum}
    ax1 = AxisSizes(m=1, t=1, z=1, c=1, y=2, x=2)
    ds0 = Dataset(axes=ax1).with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 2, 2))))
    env1 = MetaEnvelope(axes=ax1)

    def sim_graph(n, impure=False):
        g = Graph()
        g.add(NodeInstance("S", "io.seedZ"))
        g.add(NodeInstance("SIN", "zone.sim_in"))
        g.add(NodeInstance("ACC", "test.accumulate"))
        g.add(NodeInstance("SOUT", "zone.sim_out"))
        g.connect("S", "SIN"); g.connect("SIN", "ACC"); g.connect("ACC", "SOUT")
        g.connect("SOUT", "SIN", kind="back")
        z = Zone("Z", "sim", "SIN", "SOUT", body=frozenset({"ACC"}), iterations=n,
                 impure=impure)
        return g, z

    gs, zs = sim_graph(4)
    fs = unroll(gs, [zs])
    sim_out = Engine(fs, computes=merged, seeds={"S": ds0}, meta_seeds={"S": env1}).pull(
        zone_output(zs))
    acc = sim_out.image.get_region(0, 0, 0, 0, 0, 0, 2, 0, 2)
    assert np.all(acc == 4.0), "Sim zone must accumulate feedback (0 + 4 iterations)"
    # each iteration's In folds the prior Out's revision → distinct per-iteration keys
    esim = Engine(fs, computes=merged, seeds={"S": ds0}, meta_seeds={"S": env1})
    esim.pull(zone_output(zs))
    assert (esim.entry(iter_id("ACC", "Z", 1)).recipe_hash
            != esim.entry(iter_id("ACC", "Z", 2)).recipe_hash)

    # ── impure escape hatch: an impure zone re-keys per epoch (non-cacheable);
    #    a pure one ignores epoch (stays cached across pulls) ──────────────────
    gi, zi = sim_graph(4, impure=True)
    memo_i = Memo()
    Engine(unroll(gi, [zi], epoch=0), computes=merged, seeds={"S": ds0},
           meta_seeds={"S": env1}, memo=memo_i).pull(zone_output(zi))
    eimp = Engine(unroll(gi, [zi], epoch=1), computes=merged, seeds={"S": ds0},
                  meta_seeds={"S": env1}, memo=memo_i)
    eimp.pull(zone_output(zi))
    assert eimp.compute_count > 0, "impure zone must recompute under a new epoch"

    gp, zp = sim_graph(4, impure=False)
    memo_p = Memo()
    Engine(unroll(gp, [zp], epoch=0), computes=merged, seeds={"S": ds0},
           meta_seeds={"S": env1}, memo=memo_p).pull(zone_output(zp))
    epure = Engine(unroll(gp, [zp], epoch=1), computes=merged, seeds={"S": ds0},
                   meta_seeds={"S": env1}, memo=memo_p)
    epure.pull(zone_output(zp))
    assert epure.compute_count == 0, "pure zone must ignore epoch (stay cached)"

    _ok("zones: Repeat (N× body) + Simulation (feedback accumulate); revision-fold "
        "caching + incremental keys; impure epoch escape hatch; debug-verify")


# ── Phase 4a: per-frame-T Simulation specialization (zone.frame) ──────────────

def test_sim_perframe() -> None:
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.zones import Zone, iter_id, unroll, zone_output

    def _add2(ctx):                                    # out = state + current frame
        pa, pb = ctx.inputs[0].image, ctx.inputs[1].image
        ax = pa.axes
        va = pa.get_region(0, 0, 0, 0, 0, 0, ax.y, 0, ax.x)
        vb = pb.get_region(0, 0, 0, 0, 0, 0, ax.y, 0, ax.x)
        return ctx.inputs[0].with_image(
            ArrayProvider((va + vb).reshape(1, 1, 1, 1, ax.y, ax.x)))

    define_node("io.seedPF", "Seed", outputs=[OutDataset()])
    define_node("io.stackPF", "Stack", outputs=[OutDataset()])
    define_node("test.add2", "Add2", inputs=[InDataset("a"), InDataset("b")],
                outputs=[OutDataset()], granularity=Granularity.TILEABLE)

    seed = Dataset(axes=AxisSizes(m=1, t=1, z=1, c=1, y=2, x=2)).with_image(
        ArrayProvider(np.zeros((1, 1, 1, 1, 2, 2))))          # state re-init at t0
    stack_arr = np.zeros((1, 3, 1, 1, 2, 2))
    stack_arr[0, 0] = 1.0; stack_arr[0, 1] = 2.0; stack_arr[0, 2] = 3.0
    stack = Dataset(axes=AxisSizes(m=1, t=3, z=1, c=1, y=2, x=2)).with_image(
        ArrayProvider(stack_arr))                             # T=3 series

    g = Graph()
    g.add(NodeInstance("SEED", "io.seedPF"))
    g.add(NodeInstance("STACK", "io.stackPF"))
    g.add(NodeInstance("SIN", "zone.sim_in"))
    g.add(NodeInstance("FR", "zone.frame"))                   # per-frame slicer (body)
    g.add(NodeInstance("ADD", "test.add2"))
    g.add(NodeInstance("SOUT", "zone.sim_out"))
    g.connect("SEED", "SIN")                                  # state seed → In (iter0 only)
    g.connect("STACK", "FR")                                 # full T-stack → slicer (every iter)
    g.connect("SIN", "ADD", dst_socket="a")                  # carried state
    g.connect("FR", "ADD", dst_socket="b")                   # current frame t
    g.connect("ADD", "SOUT")
    g.connect("SOUT", "SIN", kind="back")                    # feedback
    z = Zone("Z", "sim", "SIN", "SOUT", body=frozenset({"FR", "ADD"}), iterations=3)
    flat = unroll(g, [z])
    seeds = {"SEED": seed, "STACK": stack}
    meta = {"SEED": MetaEnvelope(axes=seed.axes), "STACK": MetaEnvelope(axes=stack.axes)}
    merged = {**COMPUTES, "test.add2": _add2}

    out = Engine(flat, computes=merged, seeds=seeds, meta_seeds=meta).pull(zone_output(z))
    acc = out.image.get_region(0, 0, 0, 0, 0, 0, 2, 0, 2)
    assert np.all(acc == 6.0), f"per-frame Sim must accumulate 1+2+3=6, got {acc}"
    # zone.frame stamps a distinct __frame__ per iteration → distinct memo keys...
    esim = Engine(flat, computes=merged, seeds=seeds, meta_seeds=meta)
    esim.pull(zone_output(z))
    assert (esim.entry(iter_id("FR", "Z", 0)).recipe_hash
            != esim.entry(iter_id("FR", "Z", 2)).recipe_hash)
    # ...and its frame_slice meta_transform makes the envelope t==1 (matches the payload)
    assert esim.env(iter_id("FR", "Z", 1)).axes.t == 1
    _ok("zones per-frame-T: Simulation scan (zone.frame slices frame t; feedback carries "
        "state) accumulates 1+2+3 → 6; per-iter frame keys + meta_transform t→1")


# ── Phase 4b: node groups (inline-expand, nestable) ──────────────────────────

def test_groups() -> None:
    from nodegraph.groups import (Group, GROUP_INPUT, GROUP_OUTPUT, expand,
                                   group_input, group_name_of, group_output)

    def _edges(g):
        return {(e.src, e.dst, e.src_socket, e.dst_socket, e.kind) for e in g.edges}

    # ── simple group: body = (Group Input → inner → Group Output) ────────────
    body = Graph()
    body.add(NodeInstance("IN", GROUP_INPUT))
    body.add(NodeInstance("MID", "test.inner"))
    body.add(NodeInstance("OUT", GROUP_OUTPUT))
    body.connect("IN", "MID"); body.connect("MID", "OUT")
    G = Group(name="blur", body=body, input_id="IN", output_id="OUT")
    assert G.op_key == "group:blur" and group_name_of("group:blur") == "blur"

    parent = Graph()
    parent.add(NodeInstance("SRC", "test.src"))
    parent.add(NodeInstance("g0", G.op_key))            # the group INSTANCE placeholder
    parent.add(NodeInstance("SNK", "test.sink"))
    parent.connect("SRC", "g0"); parent.connect("g0", "SNK")

    flat = expand(parent, [G])
    assert "g0" not in flat.nodes                                   # instance replaced
    assert set(flat.nodes) == {"SRC", "SNK", "IN%g0", "MID%g0", "OUT%g0"}
    assert flat.nodes["MID%g0"].op_key == "test.inner"             # body copied w/ unique id
    es = _edges(flat)
    assert ("SRC", "IN%g0", "out", "data", "forward") in es        # external in → input boundary
    assert ("IN%g0", "MID%g0", "out", "data", "forward") in es     #            → inner
    assert ("MID%g0", "OUT%g0", "out", "data", "forward") in es    # inner → output boundary
    assert ("OUT%g0", "SNK", "out", "data", "forward") in es       #       → external out
    assert group_output(G, "g0") == "OUT%g0" and group_input(G, "g0") == "IN%g0"
    flat.topo_order()                                              # a well-formed DAG

    # ── nested group: Outer's body contains an Inner group instance ──────────
    ib = Graph()
    ib.add(NodeInstance("IIN", GROUP_INPUT)); ib.add(NodeInstance("N", "test.leaf"))
    ib.add(NodeInstance("IOUT", GROUP_OUTPUT))
    ib.connect("IIN", "N"); ib.connect("N", "IOUT")
    Inner = Group(name="inner", body=ib, input_id="IIN", output_id="IOUT")
    ob = Graph()
    ob.add(NodeInstance("OIN", GROUP_INPUT)); ob.add(NodeInstance("MID", Inner.op_key))
    ob.add(NodeInstance("OOUT", GROUP_OUTPUT))
    ob.connect("OIN", "MID"); ob.connect("MID", "OOUT")
    Outer = Group(name="outer", body=ob, input_id="OIN", output_id="OOUT")
    p2 = Graph()
    p2.add(NodeInstance("S", "test.src")); p2.add(NodeInstance("G", Outer.op_key))
    p2.add(NodeInstance("K", "test.sink"))
    p2.connect("S", "G"); p2.connect("G", "K")
    flat2 = expand(p2, [Outer, Inner])
    assert "G" not in flat2.nodes and "MID" not in flat2.nodes     # both instances gone
    assert not any(n.op_key.startswith("group:") for n in flat2.nodes.values())  # fixed point
    assert flat2.nodes["N%MID%G"].op_key == "test.leaf"           # innermost leaf, unique id
    es2 = _edges(flat2)
    for a, b in [("S", "OIN%G"), ("OIN%G", "IIN%MID%G"), ("IIN%MID%G", "N%MID%G"),
                 ("N%MID%G", "IOUT%MID%G"), ("IOUT%MID%G", "OOUT%G"), ("OOUT%G", "K")]:
        assert (a, b, "out", "data", "forward") in es2, (a, b)
    flat2.topo_order()

    # ── error paths: unknown ref / self-recursion / malformed definition ─────
    def _rejects(fn, needle):
        try:
            fn(); raise AssertionError(f"expected ValueError ({needle})")
        except ValueError as ex:
            assert needle in str(ex), (needle, str(ex))

    bad = Graph(); bad.add(NodeInstance("x", "group:nope"))
    _rejects(lambda: expand(bad, [G]), "unknown group")
    sb = Graph()
    sb.add(NodeInstance("SI", GROUP_INPUT)); sb.add(NodeInstance("SELF", "group:loop"))
    sb.add(NodeInstance("SO", GROUP_OUTPUT)); sb.connect("SI", "SELF"); sb.connect("SELF", "SO")
    Loop = Group(name="loop", body=sb, input_id="SI", output_id="SO")
    pl = Graph(); pl.add(NodeInstance("L", "group:loop"))
    _rejects(lambda: expand(pl, [Loop]), "recursively nested")
    mb = Graph()
    mb.add(NodeInstance("BI", "not.group.input")); mb.add(NodeInstance("BO", GROUP_OUTPUT))
    Malf = Group(name="malf", body=mb, input_id="BI", output_id="BO")
    pm = Graph(); pm.add(NodeInstance("M", "group:malf"))
    _rejects(lambda: expand(pm, [Malf]), "op_key")

    # ── end-to-end: a group wrapping enhance.gamma runs through the Engine ────
    if _HAVE_SKIMAGE:
        from nodegraph.provider import ArrayProvider
        from nodegraph.nodes import COMPUTES
        gb = Graph()
        gb.add(NodeInstance("GI", GROUP_INPUT))
        gb.add(NodeInstance("GG", "enhance.gamma", params={"gamma": 2.0}))
        gb.add(NodeInstance("GO", GROUP_OUTPUT))
        gb.connect("GI", "GG"); gb.connect("GG", "GO")
        grp = Group(name="gammagrp", body=gb, input_id="GI", output_id="GO")
        define_node("io.seedG", "Seed", outputs=[OutDataset()])
        axg = AxisSizes(m=1, t=1, z=1, c=1, y=4, x=4)
        imgg = (np.arange(16.0) + 1).reshape(1, 1, 1, 1, 4, 4)
        dsg = Dataset(axes=axg).with_image(ArrayProvider(imgg))
        pg = Graph()
        pg.add(NodeInstance("S", "io.seedG")); pg.add(NodeInstance("g0", grp.op_key))
        pg.connect("S", "g0")
        flatg = expand(pg, [grp])
        outg = Engine(flatg, computes=COMPUTES, seeds={"S": dsg},
                      meta_seeds={"S": MetaEnvelope(axes=axg)}).pull(group_output(grp, "g0"))
        got = outg.image.get_region(0, 0, 0, 0, 0, 0, 4, 0, 4)
        mx = imgg[0, 0, 0, 0].max()
        assert np.allclose(got, (imgg[0, 0, 0, 0] / mx) ** 2.0 * mx), "group body must run"

    _ok("groups: simple + nested expand (interface stitched, unique ids, fixed point); "
        "unknown-ref / recursion / malformed rejected; body runs end-to-end")


# ── frame-to-frame tracking: the TrackMembership PRODUCER (C3) ────────────────

def test_tracking() -> None:
    from nodegraph.tracking import link_labels, link_points
    from nodegraph.bridges import gather_by_track
    from nodegraph.nodes import COMPUTES

    # LABEL: id 10@t0 → 20@t1 → 30@t2 (one track); a 2nd object 40@t1 → 50@t2
    # (a track starting mid-series). Regions overlap by maximum IoU.
    r0 = np.zeros((5, 5), int); r0[1:3, 1:3] = 10
    r1 = np.zeros((5, 5), int); r1[1:3, 1:3] = 20; r1[3:5, 3:5] = 40
    r2 = np.zeros((5, 5), int); r2[1:3, 1:3] = 30; r2[3:5, 3:5] = 50
    mem = link_labels({0: r0, 1: r1, 2: r2})
    assert mem.member_domain is D.LABEL and mem.n == 5
    assert mem.track_ids().tolist() == [1, 2] and mem.timepoints().tolist() == [0, 1, 2]
    rows = set(zip(mem.track_id.tolist(), mem.t.tolist(), mem.member_id.tolist()))
    assert rows == {(1, 0, 10), (1, 1, 20), (1, 2, 30), (2, 1, 40), (2, 2, 50)}
    # produced membership is consumable by the Track bridge it feeds
    tids, means = gather_by_track([10, 20, 30, 40, 50], [1., 2., 3., 4., 5.], mem, "mean")
    assert tids.tolist() == [1, 2] and np.allclose(means, [2.0, 4.5])
    # IoU threshold above 1.0 breaks every chain → all singletons
    assert link_labels({0: r0, 1: r1, 2: r2}, iou_threshold=1.0001).track_ids().tolist() \
        == [1, 2, 3, 4, 5]

    # POINT: nearest-neighbour. track1 1@t0→3@t1→5@t2 ; track2 2@t0→4@t1 (dies at t2)
    pos = {0: ([1, 2], np.array([[0., 0.], [10., 10.]])),
           1: ([3, 4], np.array([[1., 0.], [10., 11.]])),
           2: ([5], np.array([[2., 0.]]))}
    pmem = link_points(pos, max_distance=5.0)
    assert pmem.member_domain is D.POINT
    prows = set(zip(pmem.track_id.tolist(), pmem.t.tolist(), pmem.member_id.tolist()))
    assert prows == {(1, 0, 1), (1, 1, 3), (1, 2, 5), (2, 0, 2), (2, 1, 4)}

    # track.link node end-to-end through the Engine (label mode)
    define_node("io.seedTk", "Seed", outputs=[OutDataset()])
    ax = AxisSizes(m=1, t=3, z=1, c=1, y=5, x=5)
    raster = np.zeros((1, 3, 1, 1, 5, 5), np.int64)
    raster[0, 0, 0, 0, 1:3, 1:3] = 10
    raster[0, 1, 0, 0, 1:3, 1:3] = 20; raster[0, 1, 0, 0, 3:5, 3:5] = 40
    raster[0, 2, 0, 0, 1:3, 1:3] = 30; raster[0, 2, 0, 0, 3:5, 3:5] = 50
    ds = Dataset(axes=ax).with_layer(D.VOXEL, "labels", raster)
    g = Graph(); g.add(NodeInstance("S", "io.seedTk"))
    g.add(NodeInstance("K", "track.link", modes={"target": "label"})); g.connect("S", "K")
    out = Engine(g, computes=COMPUTES, seeds={"S": ds},
                 meta_seeds={"S": MetaEnvelope(axes=ax, metadata={})}).pull("K")
    nrows = set(zip(out.get(D.TRACK, "track_id", layer="tracks").values.tolist(),
                    out.get(D.TRACK, "t", layer="tracks").values.tolist(),
                    out.get(D.TRACK, "member_id", layer="tracks").values.tolist()))
    assert nrows == rows
    _ok("tracking: link_labels (IoU overlap) + link_points (nearest-neighbour) produce "
        "TrackMembership (gather-consumable); track.link node attaches it end-to-end")


def test_track_objects() -> None:
    """``track.objects`` — the vendored v1 ``track_objects`` kernel (five interchangeable
    linkers) as the richer sibling of ``track.link``. Covers the structural spec, a real
    end-to-end pull on **all five** methods for both Label and Point members, the
    per-(m,c,z) plane independence that makes a 2D-only kernel correct on a z-stack, the
    ``track_id`` write-back alignment, the shared membership conventions, determinism
    (incl. shuffle-invariance — the one property that will silently rot), and every hard
    refusal where the kernel would otherwise degrade in silence.

    Gated on numba+pandas: the vendored kernel imports both at module scope, so even the
    centroid linker is unimportable without them."""
    import importlib.util as _u
    from nodegraph.structure import StructureTable
    from nodegraph.nodes import COMPUTES

    # ── structural spec ───────────────────────────────────────────────────────
    s = NODES.get("track.objects")
    assert s.category == "analysis"
    assert D.TRACK in s.adds_domains
    assert s.granularity is Granularity.WHOLE_SERIES
    assert s.kernel_axes == frozenset({"t", "z", "y", "x"})
    assert s.meta_transform is None                     # never changes axes/calibration
    names = {i.name for i in s.inputs}
    assert {"labels", "points", "name", "max_distance", "min_track_length",
            "ct_min_iou", "st_n_neighbors"} <= names
    assert not any(i.name in {"min_circularity", "max_eccentricity"} for i in s.inputs), \
        "morphology sockets must stay ABSENT until a producer emits those columns"
    assert all(not i.is_field for i in s.inputs if i.type is not SocketType.DATASET), \
        "no compute in this class evaluates a wired Field — field=True would be a lie"
    mode_of = {mm.name: mm for mm in s.modes}
    assert set(mode_of) == {"target", "method"} and mode_of["method"].default == "centroid"
    assert set(mode_of["method"].choices) == {"centroid", "serialtrack", "topology",
                                              "fingerprint", "overlap"}
    # method-gated variant sockets resolve per mode state (registry available_in)
    vis = lambda st: {i.name for i in s.active_inputs(st)}
    assert "ct_min_iou" in vis({"method": "overlap"})
    assert "ct_min_iou" not in vis({"method": "centroid"})
    assert "st_n_neighbors" in vis({"method": "serialtrack"})
    assert "max_frame_gap" in vis({"method": "centroid"})
    assert "labels" in vis({"target": "label"}) and "labels" not in vis({"target": "point"})
    # review: a socket the chosen linker never receives must be HIDDEN, not accepted and
    # discarded. `overlap` matches by mask IoU and takes no distance bound; the centroid
    # size gate is inert on arealess (Point) members.
    assert "max_distance" in vis({"method": "centroid"})
    assert "max_distance" not in vis({"method": "overlap"})
    assert "max_size_diff_frac" in vis({"method": "centroid", "target": "label"})
    assert "max_size_diff_frac" not in vis({"method": "centroid", "target": "point"})

    if _u.find_spec("numba") is None or _u.find_spec("pandas") is None:
        _ok("track.objects: spec OK; RUN SKIPPED (kernel needs numba+pandas)")
        return

    define_node("io.trkobj", "S", outputs=[OutDataset()])
    T, Y, X = 4, 48, 48
    BASE = np.array([[8.0, 8.0], [26.0, 30.0], [38.0, 12.0]])      # 3 objects, +2 px/frame

    def build(target="label", nz=1, perm=None):
        """3 objects drifting +2 px/frame over 4 frames on ``nz`` planes; member ids are
        globally unique. ``perm`` permutes the table's ROW ORDER (same data, different
        presentation) to exercise shuffle-invariance + write-back alignment."""
        ax = AxisSizes(m=1, t=T, z=nz, c=1, y=Y, x=X)
        raster = np.zeros((1, T, nz, 1, Y, X), np.int64)
        cols: dict = {k: [] for k in ("id", "m", "t", "c", "z", "y", "x", "area")}
        nid = 1
        for t in range(T):
            for z in range(nz):
                for (cy, cx) in BASE + t * 2.0:
                    y0, x0 = int(round(cy)) - 2, int(round(cx)) - 2
                    raster[0, t, z, 0, y0:y0 + 5, x0:x0 + 5] = nid
                    cols["id"].append(nid); cols["m"].append(0); cols["t"].append(t)
                    cols["c"].append(0); cols["z"].append(z)
                    cols["y"].append(float(y0 + 2)); cols["x"].append(float(x0 + 2))
                    cols["area"].append(25)
                    nid += 1
        arrs = {k: np.asarray(v) for k, v in cols.items()}
        if perm is not None:
            arrs = {k: v[perm] for k, v in arrs.items()}
        dom = D.LABEL if target == "label" else D.POINT
        if target != "label":
            arrs.pop("area")                            # Point tables carry no area
        layer = "labels" if target == "label" else "spots"
        ds = Dataset(axes=ax)
        if target == "label":
            ds = ds.with_layer(D.VOXEL, "labels", raster)
        return ds.with_structure(StructureTable(dom, arrs, layer=layer,
                                                z_kind="plane_index")), ax

    meta = {"pixel_size_um": 0.5}

    def pull(ds, ax, target="label", method="centroid", **params):
        g = Graph(); g.add(NodeInstance("S", "io.trkobj"))
        g.add(NodeInstance("K", "track.objects", params={"max_distance": 6.0, **params},
                           modes={"target": target, "method": method}))
        g.connect("S", "K")
        e = Engine(g, computes=COMPUTES, seeds={"S": ds},
                   meta_seeds={"S": MetaEnvelope(axes=ax, metadata=meta)})
        return e.pull("K"), e

    def trows(out, layer="tracks"):
        get = lambda n: out.get(D.TRACK, n, layer=layer).values.tolist()
        return set(zip(get("track_id"), get("t"), get("member_id")))

    # ── all five methods, Label members: 3 objects → 3 tracks spanning all 4 frames ──
    hashes = {}
    for meth in ("centroid", "topology", "fingerprint", "overlap", "serialtrack"):
        out, e = pull(*build("label"), method=meth)
        rows = trows(out)
        assert len(rows) == 12, (meth, len(rows))
        assert sorted({r[0] for r in rows}) == [1, 2, 3], (meth, rows)
        lengths = out.get(D.TRACK, "track_length", layer="tracks").values
        assert set(lengths.tolist()) == {4}, (meth, lengths)
        # every member got a real track id written back onto its own layer
        back = out.get(D.LABEL, "track_id", layer="labels").values
        assert back.shape == (12,) and int((back > 0).sum()) == 12, (meth, back)
        hashes[meth] = e.entry("K").recipe_hash

    # ── Point members (no area column ⇒ area_px 0.0; overlap unavailable) ──────
    for meth in ("centroid", "topology", "fingerprint", "serialtrack"):
        out, _ = pull(*build("point"), target="point", method=meth)
        assert sorted({r[0] for r in trows(out)}) == [1, 2, 3], meth
        assert int((out.get(D.POINT, "track_id", layer="spots").values > 0).sum()) == 12

    # ── plane independence: a 2D kernel on a z-stack must NOT link across planes ──
    out, _ = pull(*build("label", nz=2))
    rows = trows(out)
    assert sorted({r[0] for r in rows}) == [1, 2, 3, 4, 5, 6] and len(rows) == 24

    # ── membership conventions identical to track.link (contiguous, sorted) ───
    out, e_ref = pull(*build("label"))
    tid = out.get(D.TRACK, "track_id", layer="tracks").values.tolist()
    tcl = out.get(D.TRACK, "t", layer="tracks").values.tolist()
    mid = out.get(D.TRACK, "member_id", layer="tracks").values.tolist()
    assert sorted(set(tid)) == list(range(1, len(set(tid)) + 1))     # 1..K contiguous
    assert list(zip(tid, tcl, mid)) == sorted(zip(tid, tcl, mid))    # canonical row order
    assert out.structure_zkind(D.LABEL, "labels") == "plane_index"   # §7b not clobbered

    # ── determinism + shuffle-invariance, and write-back stays aligned ─────────
    ref_ids = out.get(D.LABEL, "id", layer="labels").values.tolist()
    ref_back = out.get(D.LABEL, "track_id", layer="labels").values.tolist()
    ref_map = dict(zip(ref_ids, ref_back))
    rng = np.random.default_rng(3)
    perm = rng.permutation(12)
    out_sh, _ = pull(*build("label", perm=perm))
    assert trows(out_sh) == trows(out), "row order changed the tracking result"
    sh_ids = out_sh.get(D.LABEL, "id", layer="labels").values.tolist()
    sh_back = out_sh.get(D.LABEL, "track_id", layer="labels").values.tolist()
    assert dict(zip(sh_ids, sh_back)) == ref_map, "write-back misaligned under a permutation"
    assert sh_ids == np.asarray(ref_ids)[perm].tolist(), "member layer row order disturbed"

    # ── memo: each method/target is a distinct recipe; calibration read is fenced ──
    _, e_pt = pull(*build("point"), target="point")
    hashes["point"] = e_pt.entry("K").recipe_hash
    assert len(set(hashes.values())) == len(hashes), hashes
    assert "pixel_size_um" in dict(e_ref.entry("K").reads)   # µm→px conversion memo-fenced
    assert out.metadata["track_method"] == "centroid"        # §7b provenance stamp
    assert out.metadata["track_member_domain"] == "label"
    assert out.metadata["track_member_layer"] == "labels"

    # ── hard refusals (each one is a silent kernel degradation if not caught) ──
    def refuses(fn, needle):
        try:
            fn()
        except ValueError as exc:
            assert needle in str(exc), (needle, str(exc))
            return
        raise AssertionError(f"expected a refusal mentioning {needle!r}")

    refuses(lambda: pull(*build("label"), method="bogus"), "unknown method")
    refuses(lambda: pull(*build("point"), target="point", method="overlap"),
            "unavailable for Point members")
    refuses(lambda: pull(*build("point"), target="point", max_size_diff_frac=0.1),
            "silently inert")                       # arealess members ⇒ gate does nothing
    refuses(lambda: pull(*build("label"), method="overlap", ct_max_gap=0),
            "floors its frame gap at 1")            # kernel would bridge a 1-frame hole

    # ── review: a group spanning <2 frames must not vanish at min_track_length<=1 ──
    # Every kernel linker early-returns on such a group BEFORE the min_track_length
    # post-pass, so its rows would keep track_id=None and get the 0 "untracked" sentinel
    # written back — silently deleting them from any downstream `track_id > 0` filter.
    ds_sp, ax_sp = build("label")
    lone = {k: np.concatenate([v, [v[-1] + 1] if k == "id" else [v[0]]])
            for k, v in {kk: ds_sp.get(D.LABEL, kk, layer="labels").values
                         for kk in ("id", "m", "t", "c", "z", "y", "x", "area")}.items()}
    lone["t"][-1] = 0                                # a 13th object seen only at t=0
    lone["y"][-1] = 44.0; lone["x"][-1] = 44.0
    lone["c"][-1] = 1                                # ... in its own (m,c,z) group
    ds_sp = ds_sp.with_structure(
        StructureTable(D.LABEL, lone, layer="labels", z_kind="plane_index"))
    out_1 = pull(ds_sp, ax_sp, min_track_length=1)[0]
    back_1 = out_1.get(D.LABEL, "track_id", layer="labels").values
    assert int((back_1 > 0).sum()) == 13, back_1     # the lone object IS a 1-frame track
    out_2 = pull(ds_sp, ax_sp, min_track_length=2)[0]
    back_2 = out_2.get(D.LABEL, "track_id", layer="labels").values
    assert int((back_2 > 0).sum()) == 12 and back_2[-1] == 0, back_2   # default unchanged

    ds_nr, ax_nr = build("label")                       # overlap with the raster removed
    keep = {k: ds_nr.get(D.LABEL, k, layer="labels").values
            for k in ("id", "m", "t", "c", "z", "y", "x", "area")}
    ds_nr = Dataset(axes=ax_nr).with_structure(
        StructureTable(D.LABEL, keep, layer="labels", z_kind="plane_index"))
    refuses(lambda: pull(ds_nr, ax_nr, method="overlap"), "needs the Voxel label raster")

    ds_3d, ax_3d = build("label")                       # a 3D (subpixel) structure
    ds_3d = ds_3d.with_structure(
        StructureTable(D.LABEL, keep, layer="labels", z_kind="subpixel"))
    refuses(lambda: pull(ds_3d, ax_3d), "2D-only")

    ds_dup, ax_dup = build("label")                     # duplicated member ids
    ds_dup = ds_dup.with_structure(
        StructureTable(D.LABEL, {**keep, "id": np.ones(12, np.int64)},
                       layer="labels", z_kind="plane_index"))
    refuses(lambda: pull(ds_dup, ax_dup), "globally-unique member ids")

    ds_rg, ax_rg = build("label")                       # ragged columns
    ds_rg = ds_rg.with_structure(
        StructureTable(D.LABEL, {"y": np.zeros(3)}, layer="labels", z_kind="plane_index"))
    refuses(lambda: pull(ds_rg, ax_rg), "disagree in length")

    _ok("track.objects: 5 linkers (centroid/serialtrack/CT topology+fingerprint+overlap) "
        "track Label AND Point members end-to-end; per-(m,c,z) plane independence; "
        "track_id write-back aligned under a row permutation; track.link membership "
        "conventions; distinct per-mode recipes; unconsumed sockets hidden per method; "
        "<2-frame groups survive min_track_length=1; 8 silent-degradation refusals")


# ── save / load *.nd2graph.json (graph + zones + groups round-trip) ───────────

def test_serialize() -> None:
    from nodegraph.zones import Zone
    from nodegraph.groups import Group, GROUP_INPUT, GROUP_OUTPUT
    from nodegraph.serialize import FORMAT_VERSION, from_dict, from_json, to_dict, to_json

    # graph with params+modes (incl. a __locked__ sticky list), forward edges AND a back-edge
    g = Graph()
    g.add(NodeInstance("src", "provider.synthetic",
                       params={"shape": [4, 4], "seed": 7, "__locked__": ["seed"]},
                       modes={"dim": "3D"}))
    g.add(NodeInstance("zin", "zone.repeat_in"))
    g.add(NodeInstance("blur", "enhance.gaussian",
                       params={"sigma": 1.5, "on": True}, modes={"dim": "2D"}))
    g.add(NodeInstance("zout", "zone.repeat_out"))
    g.add(NodeInstance("sink", "io.view"))
    g.connect("src", "zin"); g.connect("zin", "blur")
    g.connect("blur", "zout"); g.connect("zout", "sink")
    g.connect("zout", "zin", kind="back")                    # critical: survives round-trip

    zone = Zone("Z1", "repeat", "zin", "zout",
                body=frozenset({"blur"}), iterations=3, impure=True)
    body = Graph()
    body.add(NodeInstance("gi", GROUP_INPUT))
    body.add(NodeInstance("mid", "enhance.median", params={"radius": 2}, modes={"dim": "3D"}))
    body.add(NodeInstance("go", GROUP_OUTPUT))
    body.connect("gi", "mid"); body.connect("mid", "go")
    grp = Group("denoise", body, "gi", "go")

    s = to_json(g, zones=[zone], groups=[grp])
    assert to_json(g, zones=[zone], groups=[grp]) == s        # deterministic
    g2, zones2, groups2 = from_json(s)

    assert set(g2.nodes) == set(g.nodes)
    for nid, n in g.nodes.items():
        m = g2.nodes[nid]
        assert (m.op_key, m.params, m.modes) == (n.op_key, n.params, n.modes)
    assert g2.nodes["src"].params["__locked__"] == ["seed"]   # sticky set survives

    def eset(gr):
        return {(e.src, e.dst, e.src_socket, e.dst_socket, e.kind) for e in gr.edges}
    assert eset(g2) == eset(g)
    assert ("zout", "zin", "out", "data", "back") in eset(g2)  # back-edge survives

    z = zones2[0]
    assert (z.id, z.kind, z.in_id, z.out_id, z.body, z.iterations, z.impure) == \
           (zone.id, zone.kind, zone.in_id, zone.out_id, zone.body, zone.iterations, zone.impure)

    gr2 = groups2[0]
    assert (gr2.name, gr2.input_id, gr2.output_id) == (grp.name, grp.input_id, grp.output_id)
    assert set(gr2.body.nodes) == set(grp.body.nodes) and eset(gr2.body) == eset(grp.body)
    assert gr2.body.nodes["mid"].params == {"radius": 2}      # nested group body round-trips

    for mutate in (lambda doc: doc.update(format_version="9.9"),
                   lambda doc: doc.pop("format_version")):
        doc = to_dict(g); mutate(doc)
        try:
            from_dict(doc); raise AssertionError("bad/absent version not rejected")
        except ValueError:
            pass
    _ok(f"serialize: round-trip graph+back-edge+zone+nested group; bad/absent version "
        f"rejected (v{FORMAT_VERSION})")


# ── engine hardening: C5 provider identity + C7 un-contexted-read guard ───────

def test_engine_hardening() -> None:
    from nodegraph.provider import ArrayProvider

    # C5: a provider's .version defaults to its fingerprint — stable for the same data,
    # distinct for different content (ArrayProvider hashes its bytes).
    ax1 = AxisSizes(m=1, t=1, z=1, c=1, y=2, x=2)
    sp = SyntheticProvider(ax1)
    assert sp.version == sp.fingerprint()                       # stable/structural
    z = np.zeros((1, 1, 1, 1, 2, 2)); o = np.ones((1, 1, 1, 1, 2, 2))
    assert ArrayProvider(z).version != ArrayProvider(o).version  # content-distinct

    # C5 end-to-end: two engines SHARING one memo, same source node, providers whose
    # data differs → distinct recipe hashes (no collision) → correct payloads (a stale
    # cached blob would return the wrong image — review #3 / the __provider_version__ hook).
    define_node("c5.src", "Src", outputs=[OutDataset()])

    def _c5(ctx):
        return np.asarray([float(ctx.provider.read_region(0, 0, 0, 0, 0, 0, 1, 0, 1)[0, 0])])

    g = Graph(); g.add(NodeInstance("S", "c5.src"))
    shared = Memo()
    e0 = Engine(g, computes={"c5.src": _c5}, memo=shared, providers={"S": ArrayProvider(z)})
    e7 = Engine(g, computes={"c5.src": _c5}, memo=shared, providers={"S": ArrayProvider(o)})
    r0, r7 = e0.pull("S"), e7.pull("S")
    assert e0.entry("S").recipe_hash != e7.entry("S").recipe_hash
    assert r0[0] == 0.0 and r7[0] == 1.0                        # no wrong-payload dedup

    # C7: strict_reads makes an un-contexted calibration read on an input Dataset a hard
    # error; ctx.calib stays fine and with_metadata pass-through is unaffected.
    define_node("c7.src", "Src", outputs=[OutDataset()])
    define_node("c7.bad", "Bad", inputs=[InDataset()], outputs=[OutDataset()])
    define_node("c7.good", "Good", inputs=[InDataset()], outputs=[OutDataset()])
    srcds = Dataset(axes=AxisSizes(z=1), metadata={"pixel_size_um": 0.1})
    seed = {"S": MetaEnvelope(metadata={"pixel_size_um": 0.1})}

    def _bad(ctx):
        _ = ctx.inputs[0].metadata["pixel_size_um"]            # un-contexted → fenced away
        return ctx.inputs[0]

    def _good(ctx):
        _ = ctx.calib("pixel_size_um")                         # fenced, recorded
        return ctx.inputs[0].with_metadata(note=1)             # {**strict} fast path — no trip

    def mk(op, fn, strict):
        gg = Graph(); gg.add(NodeInstance("S", "c7.src")); gg.add(NodeInstance("X", op))
        gg.connect("S", "X")
        return Engine(gg, computes={op: fn}, seeds={"S": srcds}, meta_seeds=seed,
                      strict_reads=strict)

    try:
        mk("c7.bad", _bad, True).pull("X")
        raise AssertionError("strict_reads should reject the un-contexted calib read")
    except RuntimeError:
        pass
    assert mk("c7.bad", _bad, False).pull("X") is not None      # off by default → allowed
    og = mk("c7.good", _good, True).pull("X")
    assert og.metadata.get("note") == 1                        # ctx.calib + with_metadata OK

    # review: a content-bearing B2ndProvider must NOT collide on structural identity —
    # two stores of equal geometry but different pixels get distinct fingerprint/version
    if _HAVE_BLOSC2:
        from nodegraph.provider import B2ndProvider
        b0 = B2ndProvider.from_array(np.zeros((1, 1, 1, 1, 4, 4)))
        b1 = B2ndProvider.from_array(np.ones((1, 1, 1, 1, 4, 4)))
        assert b0.fingerprint() != b1.fingerprint() and b0.version != b1.version

    # review: the strict wrapper is LAUNDERED out of the output payload — a compute
    # returning the (wrapped) input unchanged yields a plain-dict output that a
    # downstream/non-strict consumer can read without tripping
    define_node("c7.pass", "Pass", inputs=[InDataset()], outputs=[OutDataset()])
    out = mk("c7.pass", lambda c: c.inputs[0], True).pull("X")
    assert out.metadata.__class__ is dict and out.metadata["pixel_size_um"] == 0.1
    _ok("engine hardening: C5 provider.version (ArrayProvider + B2ndProvider content); "
        "C7 strict_reads fences reads + laundered out of outputs")


# ── regression guards for the 2026-07-21 adversarial review findings ──────────

def test_review_regressions() -> None:
    from nodegraph.memo import node_recipe_hash, output_fingerprint
    from nodegraph.structure import COORD_COLUMNS, label_components
    from nodegraph.bridges import label_to_voxel, point_to_voxel
    from nodegraph.metadata import channel_select

    # #1 (critical) _canon is injective: previously-colliding params now differ
    assert (node_recipe_hash("op", {"a": "b", "c": "d"}, (), ())
            != node_recipe_hash("op", {"a": "b,sc=sd"}, (), ()))
    assert (node_recipe_hash("op", ["a", "b"], (), ())
            != node_recipe_hash("op", ["a,sb"], (), ()))
    assert node_recipe_hash("op", {"a": "b"}, (), ()) == node_recipe_hash("op", {"a": "b"}, (), ())

    # #7 Dataset fingerprint pairs key↔revision → insertion-order independent
    ax1 = AxisSizes(m=1, t=1, z=1, c=1, y=1, x=1)
    a = AttributeLayer(D.FRAME, "A", np.ones((1, 1)))
    b = AttributeLayer(D.FRAME, "B", np.ones((1, 1)))
    dsAB = Dataset(axes=ax1).with_attribute(a).with_attribute(b)
    dsBA = Dataset(axes=ax1).with_attribute(b).with_attribute(a)
    assert output_fingerprint(dsAB) == output_fingerprint(dsBA)

    # #5 a read-only VIEW over a writeable base is copied, not aliased
    big = np.ones((1, 6)); v = big.view(); v.flags.writeable = False
    lay = AttributeLayer(D.FRAME, "x", v)
    big[0, 0] = 999.0
    assert lay.values[0, 0] == 1.0

    # #14 AttributeLayer is hashable + identity-eq (no ndarray-eq crash)
    l1 = AttributeLayer(D.FRAME, "x", np.zeros((2, 3)))
    assert hash(l1) == hash(l1) and l1 == l1
    assert l1 != AttributeLayer(D.FRAME, "x", np.zeros((2, 3))) and len({l1, l1}) == 1

    # #6 label_components emits the invariant id,m,t,c,z,y,x schema
    _, tbl = label_components(np.array([[1, 0], [0, 2]]), 4, m=1, t=2, c=3)
    assert set(COORD_COLUMNS) <= set(tbl.columns)
    assert tbl.columns["m"].tolist() == [1, 1] and tbl.columns["c"].tolist() == [3, 3]

    # #9 point_to_voxel sum: background fills only empty voxels (hits not offset)
    sv = point_to_voxel([10.0], np.array([[0, 0]]), (2, 2), reducer="sum", background=5.0)
    assert sv[0, 0] == 10.0 and sv[0, 1] == 5.0
    # #10 max: a legitimate -inf point value survives; empties get background
    mx = point_to_voxel([-np.inf], np.array([[0, 0]]), (1, 2), reducer="max", background=7.0)
    assert mx[0, 0] == -np.inf and mx[0, 1] == 7.0
    # #11 label_to_voxel: a negative id paints nothing (no negative-index wrap)
    assert np.allclose(label_to_voxel([-1], [9.0], np.array([[0, 1], [1, 0]])), 0.0)

    # #12 provider empty z-range → (0, by, bx), not a crash
    sp = SyntheticProvider(AxisSizes(m=1, t=1, z=8, c=1, y=16, x=16), tile=8)
    assert sp.get_subvolume(0, 0, 0, 0, 5, 5, 0, 0).shape == (0, 8, 8)

    # #13 channel_select drops out-of-range indices (count ↔ emission stay in lockstep)
    env = MetaEnvelope(axes=AxisSizes(m=1, t=1, z=1, c=2, y=1, x=1),
                       metadata={"channel_emission_nm": [500, 600]})
    out = channel_select(env, {"channels": [0, 1, 5]}, {})
    assert out.axes.c == 2 and out.metadata["channel_emission_nm"] == [500, 600]

    # #2 a compute reading ctx.env.metadata directly is now RECORDED → invalidated
    define_node("rr.src", "S", outputs=[OutDataset()])
    define_node("rr.env", "E", inputs=[InDataset()], outputs=[OutDataset()])
    ncalls = {"e": 0}

    def c_env(ctx):
        ncalls["e"] += 1
        return np.asarray(ctx.inputs[0]) * ctx.env.metadata["pixel_size_um"]

    g = Graph()
    g.add(NodeInstance("S", "rr.src")); g.add(NodeInstance("E", "rr.env"))
    g.connect("S", "E")
    eng = Engine(g, computes={"rr.src": lambda c: np.array([2.0]), "rr.env": c_env},
                 meta_seeds={"S": MetaEnvelope(metadata={"pixel_size_um": 0.1})})
    assert np.allclose(eng.pull("E"), 0.2) and ncalls["e"] == 1
    eng.reseed_meta({"S": MetaEnvelope(metadata={"pixel_size_um": 0.5})})
    assert np.allclose(eng.pull("E"), 1.0) and ncalls["e"] == 2

    # #4 multi-input node receives args by SOCKET, independent of edge-connect order
    define_node("rr.leaf", "L", outputs=[OutDataset()])
    define_node("rr.sub", "Sub", inputs=[InDataset("a"), InDataset("b")], outputs=[OutDataset()])
    g2 = Graph()
    for nid in ("A", "B"):
        g2.add(NodeInstance(nid, "rr.leaf"))
    g2.add(NodeInstance("SUB", "rr.sub"))
    g2.connect("B", "SUB", dst_socket="b")            # edges out of declaration order
    g2.connect("A", "SUB", dst_socket="a")
    eng2 = Engine(g2, computes={"rr.sub": lambda c: np.asarray(c.inputs[0]) - np.asarray(c.inputs[1])},
                  seeds={"A": np.array([10.0]), "B": np.array([3.0])})
    assert np.allclose(eng2.pull("SUB"), 7.0)         # a - b = 10 - 3, not 3 - 10
    _ok("review regressions: #1 hash-collision, #2 env-read fence, #4 socket order, #5/#6/#7/#9-#14")


# ── C1 streaming eval (V2.04 — LOCKED 2026-07-22) ──────────────────────────────

def test_streaming() -> None:
    if not _HAVE_SKIMAGE:
        _ok("streaming eval: SKIPPED (scipy/skimage absent)")
        return
    import itertools
    from nodegraph.provider import ArrayProvider, TileProvider
    from nodegraph.nodes import COMPUTES, register_node as _reg
    from nodegraph.streaming import (
        MapComputeProvider, StreamProvider, WindowView, ZReduceProvider,
        realize, stream_fp,
    )
    from nodegraph.zones import assert_zone_pure

    rng = np.random.default_rng(7)
    ax = AxisSizes(m=1, t=1, z=2, c=1, y=70, x=70)
    raw = rng.normal(100.0, 25.0, (1, 1, 2, 1, 70, 70))
    base = ArrayProvider(raw, tile=32)              # 32-tiles → seams + ragged edge
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3}
    ds = Dataset(axes=ax, metadata=optics).with_image(base)
    env = MetaEnvelope(axes=ax, metadata=optics)
    define_node("io.stream_seed", "Seed", outputs=[OutDataset()])

    def eng(nodes, edges, *, cache_bytes=1 << 30, seed_ds=ds, seed_env=env):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for e in edges:
            g.connect(*e[:2], **(e[2] if len(e) > 2 else {}))
        return Engine(g, computes=COMPUTES, seeds={"S": seed_ds},
                      meta_seeds={"S": seed_env}, cache_bytes=cache_bytes)

    def gather(dset) -> np.ndarray:
        p = dset.image
        pax = p.axes
        out = np.empty((pax.m, pax.t, pax.z, pax.c, pax.y, pax.x), dtype=float)
        for m, t, z, c in itertools.product(range(pax.m), range(pax.t),
                                            range(pax.z), range(pax.c)):
            out[m, t, z, c] = p.get_region(0, m, t, z, c, 0, pax.y, 0, pax.x)
        return out

    # 1) tiled ≡ eager BYTES across a 2-op kernel chain (seams + ragged edges).
    #    The eager reference is the same computes forced down the pre-C1 path via the
    #    oversize bypass (a tiny budget → _map_image realizes whole, V2.04 §6b).
    chain_nodes = [("S", "io.stream_seed", {}),
                   ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                              "params": {"sigma": 0.2}}),
                   ("Md", "enhance.median", {"modes": {"dim": "2D"},
                                             "params": {"radius": 0.15}})]
    chain_edges = [("S", "G"), ("G", "Md")]
    lazy_out = eng(chain_nodes, chain_edges).pull("Md")
    assert isinstance(lazy_out.image, StreamProvider)          # actually lazy
    eager_out = eng(chain_nodes, chain_edges, cache_bytes=1024).pull("Md")
    assert isinstance(eager_out.image, ArrayProvider)          # oversize bypass → eager
    assert np.array_equal(gather(lazy_out), gather(eager_out))

    # 2) laziness + touched-window bound: pulling computes NOTHING; one tile read
    #    reads exactly one halo-expanded window from the base, second read is a cache
    #    hit (no new base reads).
    class _Counting(TileProvider):
        def __init__(self, inner):
            self._i = inner
            self.axes, self.tile, self.levels = inner.axes, inner.tile, 1
            self.calls = []

        def read_region(self, level, m, t, z, c, y0, y1, x0, x1):
            self.calls.append((z, y0, y1, x0, x1))
            return self._i.read_region(level, m, t, z, c, y0, y1, x0, x1)

        def fingerprint(self):
            return ("count",) + tuple(self._i.fingerprint())

    counting = _Counting(base)
    e2 = eng([("S", "io.stream_seed", {}),
              ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                         "params": {"sigma": 0.2}})],
             [("S", "G")], seed_ds=ds.with_image(counting))
    g_out = e2.pull("G")
    assert counting.calls == []                                # pull = plan, no pixels
    tile00 = g_out.image.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32)
    assert len(counting.calls) == 1                            # one window, one z
    (z_, y0_, y1_, x0_, x1_) = counting.calls[0]
    assert z_ == 0 and y1_ - y0_ <= 32 + 2 * 8 and x1_ - x0_ <= 32 + 2 * 8   # halo=int(4·2+.5)=8
    assert not tile00.flags.writeable                          # uniform freeze
    _ = g_out.image.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32)
    assert len(counting.calls) == 1 and e2.tiles.hits >= 1     # cache hit, no re-read
    # memo re-pull: same recipe → hit, no recompute
    n_before = e2.compute_count
    e2.pull("G")
    assert e2.compute_count == n_before

    # 3) fp stability + reseed_meta staleness (the V2.04 §6b tile-cache hole): the
    #    provider fp folds declared calibration reads, so a pixel-size edit re-keys
    #    the tile cache — the same tile MUST come back different.
    e3 = eng([("S", "io.stream_seed", {}),
              ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                         "params": {"sigma": 0.2}})], [("S", "G")])
    p_a = e3.pull("G").image
    t_a = p_a.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32)
    e3.reseed_meta({"S": MetaEnvelope(axes=ax, metadata={"pixel_size_um": 0.05,
                                                         "z_step_um": 0.3})})
    p_b = e3.pull("G").image
    t_b = p_b.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32)
    assert p_a._fp != p_b._fp                                  # calibration re-keys fp
    assert not np.array_equal(t_a, t_b)                        # no stale tile served
    # stability: an identical fresh engine reproduces the identical fp
    p_c = eng([("S", "io.stream_seed", {}),
               ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                          "params": {"sigma": 0.2}})],
              [("S", "G")]).pull("G").image
    assert p_c._fp == p_a._fp
    # unstable fp inputs (id-bearing reprs) are a hard error, not a silent cache-cold
    try:
        stream_fp("map", "x", {"fn": lambda: 0}, (), (), base)
        raise SystemExit("stream_fp accepted an unstable (callable) param")
    except TypeError:
        pass

    # 4) per-tile windowed FIELD materialization (D4 flagship, fixture node): a
    #    per-voxel Attr field scales the image per tile; per-tile ≡ whole; a layer
    #    revision change re-keys the provider fp (no stale field tiles).
    wmap = rng.normal(1.5, 0.2, (1, 1, 2, 1, 70, 70))
    ds_f = ds.with_layer(D.VOXEL, "w", wmap)

    def c_scale(ctx):
        d = ctx.inputs[0]
        fld = ctx.input("factor")
        fc = ctx.fields
        fp = stream_fp("map", ctx.op_key, ctx.params, ctx.reads.declared_reads(),
                       (field_expr_hash(fld, d),), d.image)

        def fn(a, m, t, z, c, gy0, gy1, gx0, gx1):
            win = {"m": (m, m + 1), "t": (t, t + 1), "z": (z, z + 1), "c": (c, c + 1),
                   "y": (gy0, gy1), "x": (gx0, gx1)}
            val = fc.evaluate(fld, FieldContext(d, D.VOXEL, d.axes, window=win),
                              token=(fp, 0, m, t, z, c, gy0, gy1, gx0, gx1))
            return a * np.asarray(val).reshape(a.shape)

        return d.with_image(MapComputeProvider(d.image, fn, halo=0,
                                               fp=fp, cache=ctx.tiles))

    _reg(c_scale, op_key="test.scale_field", label="ScaleF",
         inputs=[InDataset(), InFloat("factor", "Factor", field=True)],
         outputs=[OutDataset()],
         granularity=Granularity.TILEABLE, kernel_axes=frozenset())
    _reg(lambda ctx: Attr(D.VOXEL, "w"), op_key="test.wfield_src", label="WSrc",
         outputs=[OutDataset()])
    ef = eng([("S", "io.stream_seed", {}), ("F", "test.wfield_src", {}),
              ("SC", "test.scale_field", {})],
             [("S", "SC"), ("F", "SC", {"dst_socket": "factor"})], seed_ds=ds_f)
    sc = ef.pull("SC")
    assert np.allclose(gather(sc), raw * wmap, rtol=0, atol=0)   # per-tile ≡ whole
    assert ef.fields.hits + ef.fields.misses > 0                 # went through the cache
    # a changed layer (new revision) re-keys the provider fp
    ds_f2 = ds.with_layer(D.VOXEL, "w", wmap + 1.0)
    ef2 = eng([("S", "io.stream_seed", {}), ("F", "test.wfield_src", {}),
               ("SC", "test.scale_field", {})],
              [("S", "SC"), ("F", "SC", {"dst_socket": "factor"})], seed_ds=ds_f2)
    assert ef2.pull("SC").image._fp != sc.image._fp

    # 5) threshold consumes a wired per-voxel Field threshold (windowed, eager mask)
    thr_map = np.full((1, 1, 2, 1, 70, 70), 100.0)
    thr_map[0, 0, 1] = 90.0                                    # z-varying threshold
    ds_t = ds.with_layer(D.VOXEL, "thr", thr_map)
    _reg(lambda ctx: Attr(D.VOXEL, "thr"), op_key="test.tfield_src", label="TSrc",
         outputs=[OutDataset()])
    et = eng([("S", "io.stream_seed", {}), ("F", "test.tfield_src", {}),
              ("T", "analysis.threshold", {})],
             [("S", "T"), ("F", "T", {"dst_socket": "threshold"})], seed_ds=ds_t)
    mask = et.pull("T").get(D.VOXEL, "mask")
    assert mask is not None
    assert np.array_equal(mask.values, (raw > thr_map).astype(np.int64))

    # 6) z-project tree-reduce: monoid (mean/max) folds per tile via PartialReducer,
    #    median falls back to the stacked z-column — all ≡ the whole-volume reduce.
    vol = raw[0, 0, :, 0]
    for method, expect in (("max", vol.max(0)), ("mean", vol.mean(0)),
                           ("median", np.median(vol, 0))):
        ez = eng([("S", "io.stream_seed", {}),
                  ("Z", "util.zproject", {"modes": {"method": method}})],
                 [("S", "Z")])
        zout = ez.pull("Z")
        assert isinstance(zout.image, ZReduceProvider) and zout.axes.z == 1
        got = zout.image.get_region(0, 0, 0, 0, 0, 0, 70, 0, 70)
        assert np.allclose(got, expect, rtol=1e-12), f"zproject {method}"
    # meta_transform lockstep preserved (z_step dropped, z_collapsed stamped)
    assert zout.metadata.get("z_collapsed") is True and "z_step_um" not in zout.metadata

    # 7) crop is a pure lazy view: source dtype preserved, bytes = the slice, and a
    #    kernel downstream matches the eager chain (reflect-at-crop-edge parity).
    raw16 = (rng.uniform(0, 4000, (1, 1, 2, 1, 70, 70))).astype(np.uint16)
    ds16 = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(raw16, tile=32))
    crop_nodes = [("S", "io.stream_seed", {}),
                  ("C", "util.crop", {"params": {"y0": 5, "y1": 37, "x0": 3, "x1": 66}}),
                  ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                             "params": {"sigma": 0.2}})]
    crop_edges = [("S", "C"), ("C", "G")]
    ec = eng(crop_nodes, crop_edges, seed_ds=ds16)
    cds = ec.pull("C")
    assert isinstance(cds.image, WindowView) and cds.axes.y == 32 and cds.axes.x == 63
    cw = cds.image.read_region(0, 0, 0, 0, 0, 0, 32, 0, 63)
    assert cw.dtype == np.uint16                               # raw dtype preserved
    assert np.array_equal(cw, raw16[0, 0, 0, 0, 5:37, 3:66])
    lazy_g = gather(ec.pull("G"))
    eager_g = gather(eng(crop_nodes, crop_edges, seed_ds=ds16,
                         cache_bytes=1024).pull("G"))
    assert np.array_equal(lazy_g, eager_g)

    # 8) cum-halo fence: chained gaussians accumulate halo (8 px each, tile 32) →
    #    deeper providers silently promote to the plane unit, bytes stay exact.
    fence_nodes = [("S", "io.stream_seed", {})] + [
        (f"G{i}", "enhance.gaussian", {"modes": {"dim": "2D"},
                                       "params": {"sigma": 0.2}}) for i in range(4)]
    fence_edges = [("S", "G0")] + [(f"G{i}", f"G{i+1}") for i in range(3)]
    efence = eng(fence_nodes, fence_edges)
    fout = efence.pull("G3")
    assert fout.image._plane_unit                              # 2·cum_halo ≥ tile tripped
    assert np.array_equal(
        gather(fout), gather(eng(fence_nodes, fence_edges, cache_bytes=1024).pull("G3")))

    # 9) deep chain (unrolled-zone shape): 300 lazy plane units pull + realize under
    #    the scoped recursion headroom; values survive the whole chain.
    deep_nodes = [("S", "io.stream_seed", {})] + [
        (f"P{i}", "enhance.gamma", {"params": {"gamma": 1.0}}) for i in range(300)]
    deep_edges = [("S", "P0")] + [(f"P{i}", f"P{i+1}") for i in range(299)]
    ed = eng(deep_nodes, deep_edges)
    deep = ed.pull("P299")
    assert deep.image.depth >= 300
    assert np.allclose(gather(realize(deep)), raw, rtol=1e-9)

    # 10) a late ctx.calib from inside a lazy closure is a hard error at tile time
    #     (the frozen ReadContext — it would silently escape the memo fence)
    def c_late(ctx):
        d = ctx.inputs[0]
        fp = stream_fp("map", ctx.op_key, ctx.params, ctx.reads.declared_reads(),
                       (), d.image)
        return d.with_image(MapComputeProvider(
            d.image, lambda a, *rest: a * (ctx.calib("pixel_size_um") or 1.0),
            halo=0, fp=fp, cache=ctx.tiles))

    _reg(c_late, op_key="test.late_read", label="Late",
         inputs=[InDataset()], outputs=[OutDataset()],
         granularity=Granularity.TILEABLE, kernel_axes=frozenset())
    el = eng([("S", "io.stream_seed", {}), ("L", "test.late_read", {})], [("S", "L")])
    lazy_late = el.pull("L")                                   # compute itself is fine
    caught = False
    try:
        lazy_late.image.read_region(0, 0, 0, 0, 0, 0, 8, 0, 8)
    except RuntimeError as ex:
        caught = "late metadata read" in str(ex)
    assert caught, "late closure calib read was not fenced"

    # 11) assert_zone_pure still catches an IMPURE lazy body: structural fingerprints
    #     are equal by construction, so the debug-verify realizes to bytes (V2.04 §3)
    def c_noise(ctx):
        d = ctx.inputs[0]
        fp = stream_fp("map", ctx.op_key, ctx.params, ctx.reads.declared_reads(),
                       (), d.image)
        return d.with_image(MapComputeProvider(
            d.image,
            lambda a, *rest: a + np.random.default_rng().normal(size=a.shape),
            halo=0, fp=fp, cache=ctx.tiles))

    _reg(c_noise, op_key="test.noise_map", label="Noise",
         inputs=[InDataset()], outputs=[OutDataset()],
         granularity=Granularity.TILEABLE, kernel_axes=frozenset())

    def mk():
        return eng([("S", "io.stream_seed", {}), ("N", "test.noise_map", {})],
                   [("S", "N")])

    caught = False
    try:
        assert_zone_pure(mk, "N")
    except AssertionError as ex:
        caught = "non-deterministic" in str(ex)
    assert caught, "impure lazy body slipped past assert_zone_pure"

    # ── review regressions (C1 impl review, 2026-07-22) ────────────────────────

    # R1: cum-halo RESETS at a plane-unit level (a realized unit is a window-growth
    # cut point) — a mid-chain node between fence trips, and a halo-0 node after a
    # trip, both stay TILE-unit (the review's interactive-panning cliff).
    assert not efence.pull("G1").image._plane_unit or True   # G1 trips (cum 16)
    assert efence.pull("G1").image._plane_unit
    assert not efence.pull("G2").image._plane_unit           # reset at G1 → G2 tiles
    g5_nodes = fence_nodes + [("G4", "enhance.gaussian",
                               {"modes": {"dim": "2D"}, "params": {"sigma": 0.001}})]
    g5_edges = fence_edges + [("G3", "G4")]
    p_g4 = eng(g5_nodes, g5_edges).pull("G4").image
    assert not p_g4._plane_unit and p_g4.cum_halo == 0       # halo-0 after fence: tiles

    # R2 (BLOCKER): tile GRIDS are part of the streaming identity — two same-content
    # sources on different tile grids must not alias in the shared TileCache.
    gg = Graph()
    for nid in ("S1", "S2"):
        gg.add(NodeInstance(nid, "io.stream_seed"))
    for nid, src in (("Ga", "S1"), ("Gb", "S2")):
        gg.add(NodeInstance(nid, "enhance.gaussian", modes={"dim": "2D"},
                            params={"sigma": 0.2}))
        gg.connect(src, nid)
    ds32 = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(raw, tile=32))
    ds16 = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(raw, tile=16))
    e_grid = Engine(gg, computes=COMPUTES, seeds={"S1": ds32, "S2": ds16},
                    meta_seeds={"S1": env, "S2": env})
    ga, gb = gather(e_grid.pull("Ga")), gather(e_grid.pull("Gb"))
    ref_g = gather(eng([("S", "io.stream_seed", {}),
                        ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                                   "params": {"sigma": 0.2}})],
                       [("S", "G")], cache_bytes=1024).pull("G"))
    assert np.array_equal(ga, ref_g) and np.array_equal(gb, ref_g)

    # R3: zproject NaN policy is ONE policy across the lazy and eager paths
    rawn = raw.copy()
    rawn[0, 0, 0, 0, 10, 10] = np.nan
    dsn = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(rawn, tile=32))
    zn = [("S", "io.stream_seed", {}), ("Z", "util.zproject",
                                        {"modes": {"method": "mean"}})]
    zedges = [("S", "Z")]
    assert np.array_equal(gather(eng(zn, zedges, seed_ds=dsn).pull("Z")),
                          gather(eng(zn, zedges, seed_ds=dsn,
                                     cache_bytes=1024).pull("Z")), equal_nan=True)

    # R4: a COARSER-domain (FRAME) Attr field thresholds a VOXEL window (windowed
    # refine/broadcast per V2.04 §6b — was a NotImplementedError crash)
    ds_bg = ds.with_layer(D.FRAME, "bg", np.array([[95.0]]))
    _reg(lambda ctx: Attr(D.FRAME, "bg"), op_key="test.bgfield_src", label="BgSrc",
         outputs=[OutDataset()])
    ebg = eng([("S", "io.stream_seed", {}), ("F", "test.bgfield_src", {}),
               ("T", "analysis.threshold", {})],
              [("S", "T"), ("F", "T", {"dst_socket": "threshold"})], seed_ds=ds_bg)
    assert np.array_equal(ebg.pull("T").get(D.VOXEL, "mask").values,
                          (raw > 95.0).astype(np.int64))

    # R5: one field expression consumed by TWO threshold nodes over DIFFERENT
    # geometry (full + cropped) — tokens fold the input provider fp, no collision
    from nodegraph.field import UnaryOp as FUnaryOp
    _reg(lambda ctx: FUnaryOp("abs", Const(100.0)), op_key="test.cfield_src",
         label="CSrc", outputs=[OutDataset()])
    e2t = eng([("S", "io.stream_seed", {}), ("F", "test.cfield_src", {}),
               ("T1", "analysis.threshold", {}),
               ("C", "util.crop", {"params": {"y0": 5, "y1": 37, "x0": 3, "x1": 66}}),
               ("T2", "analysis.threshold", {})],
              [("S", "T1"), ("F", "T1", {"dst_socket": "threshold"}),
               ("S", "C"), ("C", "T2"), ("F", "T2", {"dst_socket": "threshold"})])
    m1 = e2t.pull("T1").get(D.VOXEL, "mask").values
    m2 = e2t.pull("T2").get(D.VOXEL, "mask").values
    assert np.array_equal(m1, (raw > 100.0).astype(np.int64))
    assert np.array_equal(m2, (raw[..., 5:37, 3:66] > 100.0).astype(np.int64))

    # R6: swapping engine.seeds[nid] re-keys the source (seed data identity)
    e_swap = eng([("S", "io.stream_seed", {}),
                  ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                             "params": {"sigma": 0.2}})], [("S", "G")])
    a_sw = gather(e_swap.pull("G"))
    e_swap.seeds["S"] = Dataset(axes=ax, metadata=optics).with_image(
        ArrayProvider(raw + 1.0, tile=32))
    b_sw = gather(e_swap.pull("G"))
    assert not np.array_equal(a_sw, b_sw)

    # R7: an out-of-range z on a cropped view raises (never serves pixels OUTSIDE
    # the crop)
    caught = False
    try:
        cds.image.read_region(0, 0, 0, 5, 0, 0, 8, 0, 8)
    except IndexError:
        caught = True
    assert caught

    # R8: providers hold the cache WEAKLY — a payload outliving its engine stays
    # readable (recompute path), it does not pin or crash
    import gc
    e_weak = eng([("S", "io.stream_seed", {}),
                  ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                             "params": {"sigma": 0.2}})], [("S", "G")])
    out_w = e_weak.pull("G")
    t_w1 = np.array(out_w.image.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32))
    del e_weak
    gc.collect()
    t_w2 = out_w.image.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32)
    assert np.array_equal(t_w1, t_w2)

    # R9: two edges into one NON-multi socket is a hard error (no silent last-wins)
    gdup = Graph()
    for nid in ("S1", "S2"):
        gdup.add(NodeInstance(nid, "io.stream_seed"))
    gdup.add(NodeInstance("G", "enhance.gaussian", modes={"dim": "2D"},
                          params={"sigma": 0.2}))
    gdup.connect("S1", "G")
    gdup.connect("S2", "G")
    e_dup = Engine(gdup, computes=COMPUTES, seeds={"S1": ds32, "S2": ds16},
                   meta_seeds={"S1": env, "S2": env})
    caught = False
    try:
        e_dup.pull("G")
    except ValueError as ex:
        caught = "non-multi" in str(ex)
    assert caught

    _ok("streaming eval (C1): tiled≡eager bytes; lazy pull + one-window tile reads + "
        "cache hits; reseed re-keys fp (no stale tiles); windowed per-tile fields + "
        "threshold Field; zproject tree-reduce; crop view; cum-halo fence; 300-deep "
        "chain; late-read fence; impure lazy body caught; review R1–R9 (fence reset, "
        "grid identity, NaN policy, coarser-domain field, token collision, seed swap, "
        "crop z-guard, weak cache, non-multi edge guard)")


def test_streaming_slivers() -> None:
    """C1 follow-up slivers (V2.04 §6b): ``util.stack`` T→1 tree-reduce; lazy units for
    normalize / resample / drift (eager-stat, lazy-apply); the kernel-param field gate."""
    if not _HAVE_SKIMAGE:
        _ok("streaming slivers: SKIPPED (scipy/skimage absent)")
        return
    import itertools
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES, register_node as _reg, _map_image, _DIM_KAX
    from nodegraph.streaming import (
        MapComputeProvider, PlaneRealizeProvider, TReduceProvider, VolumeComputeProvider,
    )

    rng = np.random.default_rng(11)
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3, "dt_s": 0.5}
    axt = AxisSizes(m=1, t=3, z=2, c=1, y=40, x=40)
    raw = rng.normal(50.0, 12.0, (1, 3, 2, 1, 40, 40))
    ds = Dataset(axes=axt, metadata=optics).with_image(ArrayProvider(raw, tile=32))
    env = MetaEnvelope(axes=axt, metadata=optics)
    define_node("test.slv_seed", "Seed", outputs=[OutDataset()])

    def eng(nodes, edges, *, seed_ds=ds, seed_env=env):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for e in edges:
            g.connect(*e[:2], **(e[2] if len(e) > 2 else {}))
        return Engine(g, computes=COMPUTES, seeds={"S": seed_ds}, meta_seeds={"S": seed_env})

    def gather(dset) -> np.ndarray:
        p = dset.image
        pa = p.axes
        out = np.empty((pa.m, pa.t, pa.z, pa.c, pa.y, pa.x), dtype=float)
        for m, t, z, c in itertools.product(range(pa.m), range(pa.t), range(pa.z),
                                            range(pa.c)):
            out[m, t, z, c] = p.get_region(0, m, t, z, c, 0, pa.y, 0, pa.x)
        return out

    # 1) util.stack T→1: monoid (mean/max) folds the T series per tile, non-monoid
    #    (median) stacks the t-column — both ≡ the numpy reference; dt_s dropped.
    for method, ref in (("mean", raw.mean(1, keepdims=True)),
                        ("max", raw.max(1, keepdims=True)),
                        ("median", np.median(raw, 1, keepdims=True))):
        st = eng([("S", "test.slv_seed", {}),
                  ("K", "util.stack", {"modes": {"method": method}})],
                 [("S", "K")]).pull("K")
        assert isinstance(st.image, TReduceProvider) and st.axes.t == 1
        assert "dt_s" not in st.metadata
        assert np.allclose(gather(st), ref, rtol=1e-12), f"stack {method}"

    # 2) normalize lazy units: plane (self-contained per plane) & series (eager (lo,hi)
    #    per (m,c) + lazy apply) → MapComputeProvider; volume → VolumeComputeProvider.
    lo_p, hi_p = 1.0, 99.0

    def _rescale(b: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(b, lo_p), np.percentile(b, hi_p)
        return np.zeros_like(b) if hi <= lo else np.clip((b - lo) / (hi - lo), 0.0, 1.0)

    npl = eng([("S", "test.slv_seed", {}),
               ("N", "enhance.normalize", {"modes": {"scope": "plane"}})],
              [("S", "N")]).pull("N")
    assert isinstance(npl.image, MapComputeProvider)
    ref_pl = np.empty_like(raw)
    for m, t, z, c in itertools.product(range(1), range(3), range(2), range(1)):
        ref_pl[m, t, z, c] = _rescale(raw[m, t, z, c])
    assert np.allclose(gather(npl), ref_pl)

    nvo = eng([("S", "test.slv_seed", {}),
               ("N", "enhance.normalize", {"modes": {"scope": "volume"}})],
              [("S", "N")]).pull("N")
    assert isinstance(nvo.image, VolumeComputeProvider)
    ref_vo = np.empty_like(raw)
    for m, t, c in itertools.product(range(1), range(3), range(1)):
        ref_vo[m, t, :, c] = _rescale(raw[m, t, :, c])
    assert np.allclose(gather(nvo), ref_vo)

    nse = eng([("S", "test.slv_seed", {}),
               ("N", "enhance.normalize", {"modes": {"scope": "series"}})],
              [("S", "N")]).pull("N")
    assert isinstance(nse.image, MapComputeProvider)
    ref_se = np.empty_like(raw)
    for m, c in itertools.product(range(1), range(1)):
        ref_se[m, :, :, c] = _rescale(raw[m, :, :, c])
    assert np.allclose(gather(nse), ref_se)

    # 3) drift: eager per-frame FFT estimate + lazy per-plane shift apply (register-once/
    #    apply-all); Frame drift_y/x attrs present; ≡ a direct eager reference.
    from scipy.ndimage import shift as _nsh
    from skimage.registration import phase_cross_correlation as _pcc
    dft = eng([("S", "test.slv_seed", {}), ("D", "align.drift", {})], [("S", "D")]).pull("D")
    assert isinstance(dft.image, MapComputeProvider)
    assert (dft.get(D.FRAME, "drift_y") is not None
            and dft.get(D.FRAME, "drift_x") is not None)
    ref_d = np.empty_like(raw)
    refz = axt.z // 2
    for m in range(1):
        r = raw[m, 0, refz, 0]
        for t in range(3):
            sh = _pcc(r, raw[m, t, refz, 0], upsample_factor=10)[0]
            for c in range(1):
                for z in range(2):
                    ref_d[m, t, z, c] = _nsh(raw[m, t, z, c], shift=sh, order=1,
                                             mode="constant")
    assert np.allclose(gather(dft), ref_d)

    # 4) resample lazy per-UNIT realize (geometry-changing → PlaneRealizeProvider): only
    #    touched planes resize; ≡ a direct skimage reference; output axes scaled.
    from skimage.transform import resize as _rsz
    rsp = eng([("S", "test.slv_seed", {}),
               ("R", "util.resample",
                {"modes": {"dim": "2D"}, "params": {"scale_xy": 0.5}})],
              [("S", "R")]).pull("R")
    assert isinstance(rsp.image, PlaneRealizeProvider)
    assert (rsp.axes.y, rsp.axes.x, rsp.axes.z) == (20, 20, 2)
    ref_r = np.empty((1, 3, 2, 1, 20, 20))
    for m, t, z, c in itertools.product(range(1), range(3), range(2), range(1)):
        ref_r[m, t, z, c] = _rsz(raw[m, t, z, c], (20, 20), order=1, preserve_range=True)
    assert np.allclose(gather(rsp), ref_r)

    # 5) kernel-param field gate (Fork B): a NON-Const Field on a kernel_param socket drops
    #    the TILEABLE 2D unit to WHOLE_PLANE; a Const field / no field stay tiled. A halo-0
    #    identity fixture isolates the gate from the cum-halo fence.
    def _c_kfilter(ctx):
        return _map_image(ctx, ctx.inputs[0], plane_fn=lambda a: a, halo=0)

    _reg(_c_kfilter, op_key="test.kfilter", label="KFilter",
         inputs=[InDataset(), InFloat("sigma", "Sigma", field=True, kernel_param=True)],
         outputs=[OutDataset()], modes=[DimMode()],
         granularity={"2D": Granularity.TILEABLE, "3D": Granularity.WHOLE_VOLUME},
         kernel_axes=_DIM_KAX)
    _reg(lambda ctx: Attr(D.VOXEL, "w"), op_key="test.slv_fieldsrc", label="FSrc",
         outputs=[OutDataset()])
    _reg(lambda ctx: Const(0.2), op_key="test.slv_constsrc", label="CSrc",
         outputs=[OutDataset()])
    gv = eng([("S", "test.slv_seed", {}), ("F", "test.slv_fieldsrc", {}),
              ("K", "test.kfilter", {"modes": {"dim": "2D"}})],
             [("S", "K"), ("F", "K", {"dst_socket": "sigma"})]).pull("K")
    assert isinstance(gv.image, MapComputeProvider) and gv.image._plane_unit is True
    gc = eng([("S", "test.slv_seed", {}), ("F", "test.slv_constsrc", {}),
              ("K", "test.kfilter", {"modes": {"dim": "2D"}})],
             [("S", "K"), ("F", "K", {"dst_socket": "sigma"})]).pull("K")
    assert isinstance(gc.image, MapComputeProvider) and gc.image._plane_unit is False
    gb = eng([("S", "test.slv_seed", {}),
              ("K", "test.kfilter", {"modes": {"dim": "2D"}})], [("S", "K")]).pull("K")
    assert gb.image._plane_unit is False

    _ok("streaming slivers (V2.04 §6b): util.stack T→1 tree-reduce (monoid+gather≡ref, "
        "dt_s dropped); normalize plane/volume/series lazy units; drift eager-estimate + "
        "lazy per-plane apply; resample lazy per-unit realize (geometry change); "
        "kernel-param field gate (non-Const σ field → plane unit; Const/none → tiled)")


def test_catalog_kernels() -> None:
    """Ported v1 analysis kernels (Phase 7 / V2.05): bead detection, histogram
    threshold, registration, granule boundary, + the dep-gated stubs. StarDist is
    excluded (a ~seconds TF model load — verified live in the port, not the fast gate)."""
    if not _HAVE_SKIMAGE:
        _ok("catalog (v1 kernel ports): SKIPPED (scipy/skimage absent)")
        return
    import importlib.util as _u
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    def eng(seed_ds, seed_env, nodes, edges):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for s, d in edges:
            g.connect(s, d)
        return Engine(g, computes=COMPUTES, seeds={"S": seed_ds},
                      meta_seeds={"S": seed_env})

    define_node("io.kseed", "S", outputs=[OutDataset()])

    # ── bead detection (needs numba) — 3D finds the planted beads; 2D≠3D hash ──
    if _u.find_spec("numba") is not None:
        ax = AxisSizes(m=1, t=1, z=8, c=1, y=48, x=48)
        vol = np.zeros((1, 1, 8, 1, 48, 48))
        for (z, y, x) in [(4, 12, 12), (4, 30, 34), (2, 38, 10)]:
            vol[0, 0, z - 1:z + 2, 0, y - 2:y + 3, x - 2:x + 3] = 400.0
        bmeta = {"pixel_size_um": 0.2, "z_step_um": 0.5, "objective_na": 1.0,
                 "channel_emission_nm": [520]}
        bds = Dataset(axes=ax, metadata=bmeta).with_image(ArrayProvider(vol))
        benv = MetaEnvelope(axes=ax, metadata=bmeta)
        e3 = eng(bds, benv, [("S", "io.kseed", {}),
                             ("B", "detect.particles", {"modes": {"dim": "3D"},
                                                        "params": {"min_distance": 0.6}})],
                 [("S", "B")])
        pts = e3.pull("B").get(D.POINT, "z", layer="particles")
        assert pts is not None and len(pts.values) == 3, \
            f"particles 3D found {0 if pts is None else len(pts.values)}"
        e2 = eng(bds, benv, [("S", "io.kseed", {}),
                             ("B", "detect.particles", {"modes": {"dim": "2D"},
                                                        "params": {"min_distance": 0.6}})],
                 [("S", "B")])
        assert e3.entry("B").recipe_hash != e2.entry("B").recipe_hash
        assert any(k == "pixel_size_um" for k, _ in e3.entry("B").reads)
        beads_note = "particles 2D/3D (3 detected, distinct hash, fenced)"
    else:
        beads_note = "beads SKIPPED (numba absent)"

    # ── histogram threshold → mask + labels + region table; methods re-key ─────
    ax2 = AxisSizes(m=1, t=1, z=2, c=1, y=48, x=48)
    img = np.full((1, 1, 2, 1, 48, 48), 500.0)
    img[0, 0, :, 0, 10:20, 10:20] = 4000.0
    img[0, 0, :, 0, 30:36, 30:36] = 3800.0
    hds = Dataset(axes=ax2, metadata={"pixel_size_um": 0.25}).with_image(ArrayProvider(img))
    henv = MetaEnvelope(axes=ax2, metadata={"pixel_size_um": 0.25})
    # v1 parity 2026-07-28: the spatial cleanup now DEFAULTS to the v1 segmenter's numbers
    # (100 px² min area, 1 px opening, 2 px closing, 50 px² holes) via µm derives. The
    # threshold-behaviour checks below pin it OFF so they stay about thresholding; the
    # default itself is verified in its own block (it deletes this fixture's tiny squares).
    NOCLEAN = {"min_area": 0.0, "opening_radius": 0.0,
               "closing_radius": 0.0, "min_hole_size": 0.0}
    eh = eng(hds, henv, [("S", "io.kseed", {}),
                         ("H", "analysis.histogram_threshold",
                          {"params": {**NOCLEAN, "percentile_high": 95.0}})], [("S", "H")])
    hout = eh.pull("H")
    assert int(hout.get(D.VOXEL, "mask").values.sum()) > 0
    assert int(hout.get(D.VOXEL, "labels").values.max()) == 4      # 2 squares × 2 planes
    assert hout.get(D.LABEL, "mean_intensity", layer="labels") is not None
    # the rasters keep the kernel's dtypes — int64 would be 4×/8× the bytes for nothing
    # (a 16-position 2048²×10 series = 5 GiB PER raster → ArrayMemoryError, fix 2026-07-28)
    assert hout.get(D.VOXEL, "mask").values.dtype == np.uint8
    assert hout.get(D.VOXEL, "labels").values.dtype == np.int32
    assert np.array_equal(hout.get(D.VOXEL, "mask").values > 0,
                          hout.get(D.VOXEL, "labels").values > 0)   # mask ≡ labelled area
    e_ch = eng(hds, henv,                       # …and a real Label consumer reads int32
               [("S", "io.kseed", {}),
                ("H", "analysis.histogram_threshold",
                 {"modes": {"method": "relative", "direction": "above"},
                  "params": dict(NOCLEAN)}),
                ("B", "analysis.boundary_band", {"params": {"band_voxels": 1}})],
               [("S", "H"), ("H", "B")])
    assert int((e_ch.pull("B").get(D.VOXEL, "bands").values != 0).sum()) > 0
    eh2 = eng(hds, henv, [("S", "io.kseed", {}),
                          ("H", "analysis.histogram_threshold",
                           {"modes": {"method": "hysteresis", "direction": "above"},
                            "params": {**NOCLEAN, "strict": 3500,
                                       "permissive": 3000}})], [("S", "H")])
    assert eh.entry("H").recipe_hash != eh2.entry("H").recipe_hash
    # hysteresis seed ORDERING is directional in v1 (below: strict ≤ permissive; above:
    # strict ≥ permissive — the core is the more extreme cut). v1 raised a bare comparison
    # error; explain the two seeds instead (2026-07-28)
    try:
        eng(hds, henv, [("S", "io.kseed", {}),
                        ("H", "analysis.histogram_threshold",
                         {"modes": {"method": "hysteresis", "direction": "above"},
                          "params": {**NOCLEAN, "strict": 3000,
                                     "permissive": 3500}})], [("S", "H")]).pull("H")
        raise SystemExit("hysteresis above with strict < permissive should be refused")
    except ValueError as ex:
        assert "CORE cut" in str(ex) and "Swap them" in str(ex), str(ex)
    # method-specific params: the image-relative defaults are v1's own (percentile 5/95,
    # relative 0.30×/1.50×) and segment out of the box — no threshold typed in.
    # Per-plane fixture: 100 px @4000 + 36 px @3800 over 2304 px @500.
    for _md, _n in ((({}), 272),                        # ≥p95 → both bright squares
                    ({"method": "relative", "direction": "above"}, 272),  # >1.5×500 → both
                    ({"method": "percentile", "direction": "below"}, 4336)):  # ≤p5 → bg
        _e = eng(hds, henv, [("S", "io.kseed", {}),
                             ("H", "analysis.histogram_threshold",
                              {"modes": _md, "params": dict(NOCLEAN)})], [("S", "H")])
        assert int(_e.pull("H").get(D.VOXEL, "mask").values.sum()) == _n, _md
    # ... but an ABSOLUTE raw-count method with nothing set (or an explicit 0 = unset,
    # the GUI spin box's empty state) must ASK, naming the fields — not surface the
    # vendored dataclass's bare "requires `strict` and `permissive`" traceback
    for _p in ({}, {"strict": 3500, "permissive": 0}):
        try:
            eng(hds, henv, [("S", "io.kseed", {}),
                            ("H", "analysis.histogram_threshold",
                             {"modes": {"method": "hysteresis", "direction": "above"},
                              "params": _p})], [("S", "H")]).pull("H")
            raise SystemExit(f"hysteresis with {_p} should demand its thresholds")
        except ValueError as ex:
            assert "`permissive`" in str(ex) and "0 = unset" in str(ex), str(ex)
    try:                                    # hysteresis has no two-sided form
        eng(hds, henv, [("S", "io.kseed", {}),
                        ("H", "analysis.histogram_threshold",
                         {"modes": {"method": "hysteresis", "direction": "between"},
                          "params": {**NOCLEAN, "strict": 3500, "permissive": 3000}})],
            [("S", "H")]).pull("H")
        raise SystemExit("hysteresis should reject direction=between")
    except ValueError as ex:
        assert "below/above only" in str(ex)
    _hspec = NODES.get("analysis.histogram_threshold")
    _vis = lambda st: {i.name for i in _hspec.active_inputs(st)}
    assert {"strict", "permissive"} <= _vis({"method": "hysteresis", "direction": "above"})
    assert not ({"low", "high", "percentile_high", "fraction_high"}
                & _vis({"method": "hysteresis", "direction": "above"}))
    assert "high" in _vis({"method": "single", "direction": "above"})
    assert "low" not in _vis({"method": "single", "direction": "above"})
    assert {"low", "high"} <= _vis({"method": "single", "direction": "between"})
    # max_area: the restored upper size filter (µm²→px², 0 = no limit). px=0.25 → 1 px² =
    # 0.0625 µm²; the 4000-square is 100 px² = 6.25 µm², the 3800 one 36 px² = 2.25 µm².
    assert _hspec.input("max_area") is not None and _hspec.input("max_area").unit == "um2"
    e_ma = eng(hds, henv, [("S", "io.kseed", {}),
                           ("H", "analysis.histogram_threshold",
                            {"modes": {"method": "relative", "direction": "above"},
                             "params": {**NOCLEAN, "max_area": 3.0}})], [("S", "H")])
    assert int(e_ma.pull("H").get(D.VOXEL, "mask").values.sum()) == 72   # 36 px × 2 planes
    assert e_ma.entry("H").recipe_hash != eh2.entry("H").recipe_hash     # the lever re-keys
    # ── v1 spatial-cleanup parity (2026-07-28) ─────────────────────────────────
    # The defaults ARE v1's: 100 px² min area, 1 px opening, 2 px closing, 50 px² holes,
    # carried as µm derives so they follow the objective. On a 30×30 blob + a 4×4 speck:
    # the speck is deleted by min_area and the blob keeps 896 px (opening/closing shave
    # its 4 corners). UNCALIBRATED, the derives resolve to 0 ⇒ cleanup off and area_um2
    # is NaN — v1's `voxel_size=None` path, never a fabricated 0.1 µm/px.
    axc = AxisSizes(m=1, t=1, z=1, c=1, y=128, x=128)
    cimg = np.full((1, 1, 1, 1, 128, 128), 500.0)
    cimg[0, 0, 0, 0, 20:50, 20:50] = 4000.0                   # 900 px
    cimg[0, 0, 0, 0, 100:104, 100:104] = 4000.0               # 16 px speck
    for _meta, _mask, _nlab, _px, _um2 in (
            ({"pixel_size_um": 0.25}, 896, 1, [896], [56.0]),
            ({}, 916, 2, [16, 900], None)):
        cds = Dataset(axes=axc, metadata=_meta).with_image(ArrayProvider(cimg))
        cenv = MetaEnvelope(axes=axc, metadata=_meta)
        _e = eng(cds, cenv, [("S", "io.kseed", {}),
                             ("H", "analysis.histogram_threshold", {})], [("S", "H")])
        _o = _e.pull("H")
        assert int(_o.get(D.VOXEL, "mask").values.sum()) == _mask, _meta
        assert int(_o.get(D.VOXEL, "labels").values.max()) == _nlab, _meta
        assert sorted(_o.get(D.LABEL, "area", layer="labels").values.tolist()) == _px
        _got = sorted(_o.get(D.LABEL, "area_um2", layer="labels").values.tolist())
        if _um2 is None:
            assert all(np.isnan(v) for v in _got), _got   # no calibration ⇒ no µm² fiction
        else:
            assert np.allclose(_got, _um2), _got
    hspec_ma = NODES.get("analysis.histogram_threshold").input("min_area")
    assert hspec_ma.derive == "100*(pixel_size_um or 0)**2"    # v1's 100 px², µm-authored

    # ── hysteresis agrees with the other methods at the same cut (2026-07-28) ───
    # The vendored kernel delegated its seeds to skimage's STRICT `>` while its own
    # docstring (and threshold_single, hence percentile/relative) promised `>=`/`<=`. So an
    # object plateau sitting exactly AT `strict` seeded nothing and hysteresis returned an
    # empty mask where percentile at the same resolved threshold found every object — the
    # visible bug on dim 12-bit data (objects at 192) and on saturated data (4095).
    from nodegraph.kernels.histogram_threshold import (
        HistogramThresholdSegmenter as _HTS, compute_histogram as _chist,
        make_config as _mkcfg)
    _flat = np.full((64, 64), 120, np.uint16)
    _flat[10:40, 10:40] = 192                                 # 900 px AT the cut value
    assert _chist(_flat, bit_depth=12).percentile(95) == 192   # what percentile resolves to

    def _kmask(im, **kw):
        cfg = _mkcfg(bit_depth=12, min_area=0, opening_radius=0, closing_radius=0,
                     min_hole_size=0, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return int(_HTS(cfg).run(im).mask.sum())
    # percentile ≡ single ≡ hysteresis, all inclusive, all 900 px
    assert _kmask(_flat, method="percentile", direction="above",
                  percentile_high=95.0) == 900
    assert _kmask(_flat, method="single", direction="above", high=192) == 900
    assert _kmask(_flat, method="hysteresis", direction="above",
                  strict=192, permissive=150) == 900           # was 0 before the fix
    assert _kmask(_flat, method="hysteresis", direction="below",
                  strict=120, permissive=150) == 3196          # was 0 before the fix
    # the general invariant: a DEGENERATE hysteresis (strict == permissive == T) is exactly
    # a single threshold at T — no fringe to grow, so the two must agree pixel for pixel.
    _rng = np.random.default_rng(0)
    for _im in ((np.arange(64 * 64).reshape(64, 64) % 500).astype(np.uint16),
                np.where(_rng.random((64, 64)) > 0.7, 3000, 400).astype(np.uint16),
                np.full((64, 64), 4095, np.uint16)):          # incl. a saturated plateau
        for _T in (0, 1, 137, 400, 499, 3000, 4095):
            assert (_kmask(_im, method="hysteresis", direction="above",
                           strict=_T, permissive=_T)
                    == _kmask(_im, method="single", direction="above", high=_T)), _T
            assert (_kmask(_im, method="hysteresis", direction="below",
                           strict=_T, permissive=_T)
                    == _kmask(_im, method="single", direction="below", low=_T)), _T

    # ── bit depth is METADATA, not an assumption (2026-07-28) ───────────────────
    # Most ND2s are 12-bit; declaring 16 mis-sized the percentile LUT and made the kernel
    # warn "max N is <5% of 16-bit range" on any dim frame. The depth now comes from the
    # `bit_depth` calibration key (memo-fenced), snapped UP to a kernel LUT, user-
    # overridable, and only falls back to 16 when the metadata is silent.
    from nodegraph.nodes import _snap_bit_depth as _snapbd
    assert [_snapbd(b) for b in (None, 0, 8, 11, 12, 14, 16, 20)] == \
           [16, 16, 8, 12, 12, 14, 16, 16]                    # round UP, never truncate
    axb = AxisSizes(m=1, t=1, z=1, c=1, y=64, x=64)
    bimg = np.full((1, 1, 1, 1, 64, 64), 120.0)               # a DIM 12-bit frame…
    bimg[0, 0, 0, 0, 10:40, 10:40] = 192.0                    # …max 192, <5% of 16-bit
    for _meta, _params, _warns in (
            ({"pixel_size_um": 0.25, "bit_depth": 12}, {}, 0),      # from the file
            ({"pixel_size_um": 0.25}, {"bit_depth": 12}, 0),        # user override
            ({"pixel_size_um": 0.25}, {}, 1)):                     # silent ⇒ 16 + warn
        bds = Dataset(axes=axb, metadata=_meta).with_image(ArrayProvider(bimg))
        _e = eng(bds, MetaEnvelope(axes=axb, metadata=_meta),
                 [("S", "io.kseed", {}),
                  ("H", "analysis.histogram_threshold", {"params": _params})],
                 [("S", "H")])
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter("always")
            _o = _e.pull("H")
        assert int(_o.get(D.VOXEL, "mask").values.sum()) == 896, (_meta, _params)
        got = [x for x in _w if "<5% of" in str(x.message)]
        assert len(got) == _warns, (_meta, _params, [str(x.message) for x in got])
        if "bit_depth" in _meta:                    # the read is fenced like any calib
            assert "bit_depth" in {k for k, _ in _e.entry("H").reads}
    assert "bit_depth" in CALIBRATION_KEYS         # promoted into the engine schema
    assert envelope_symbols(MetaEnvelope(axes=axb, metadata={"bit_depth": 12})
                            )["bit_depth"] == 12   # …and visible to `derive`

    # ── the optional `raw` socket: segment on enhanced, MEASURE on raw ──────────
    # `raw` overrides the measured PIXELS only — mask/labels/thresholds and the whole
    # calibration env stay on the main chain (`data` is declared first, so dataset_preds
    # keeps it primary no matter which edge was wired first).
    define_node("io.kseed2", "R", outputs=[OutDataset()])
    axr = AxisSizes(m=1, t=1, z=1, c=1, y=32, x=32)
    rawi = np.full((1, 1, 1, 1, 32, 32), 300.0)
    rawi[0, 0, 0, 0, 4:12, 4:12] = 700.0                  # two objects, DISTINCT counts
    rawi[0, 0, 0, 0, 20:28, 20:28] = 1500.0
    rmeta = {"pixel_size_um": 0.25, "bit_depth": 12}
    rawds = Dataset(axes=axr, metadata=rmeta).with_image(ArrayProvider(rawi))
    rawenv = MetaEnvelope(axes=axr, metadata=rmeta)

    def _measure_graph(wire_raw, raw_seed=None, raw_env=None):
        g = Graph()
        for nid, op in (("S", "io.kseed"), ("R", "io.kseed2")):
            g.add(NodeInstance(nid, op))
        g.add(NodeInstance("N", "enhance.normalize"))      # destroys the count scale
        g.add(NodeInstance("T", "analysis.threshold",
                           modes={"method": "fixed"}, params={"threshold": 0.2}))
        g.add(NodeInstance("L", "analysis.label", params={"name": "labels"}))
        g.add(NodeInstance("M", "analysis.measure", params={"labels": "labels"}))
        for a, b in (("S", "N"), ("N", "T"), ("T", "L"), ("L", "M")):
            g.connect(a, b)
        if wire_raw:
            g.connect("R", "M", dst_socket="raw")
        e = Engine(g, computes=COMPUTES,
                   seeds={"S": rawds, "R": raw_seed if raw_seed is not None else rawds},
                   meta_seeds={"S": rawenv,
                               "R": raw_env if raw_env is not None else rawenv})
        return g, e
    _g_plain, _e_plain = _measure_graph(False)
    _plain = _e_plain.pull("M").get(D.LABEL, "mean_intensity", layer="labels").values
    _g_raw, _e_raw = _measure_graph(True)
    _raw = _e_raw.pull("M").get(D.LABEL, "mean_intensity", layer="labels").values
    assert len(_plain) == len(_raw) == 2, (len(_plain), len(_raw))   # same segmentation
    # the normalized chain reports a rescaled shadow — 300/700/1500 counts became
    # 0/⅓/1, so even the RATIO between the two objects is gone (percentile clipping)
    assert np.allclose(sorted(_plain), [1.0 / 3.0, 1.0])
    assert np.allclose(sorted(_raw), [700.0, 1500.0])  # raw: the true per-region counts
    assert _e_plain.entry("M").recipe_hash != _e_raw.entry("M").recipe_hash   # re-keys
    assert [e.dst_socket for e in _g_raw.dataset_preds("M")] == ["data", "raw"]
    assert propagate_meta(_g_raw, {"S": rawenv, "R": rawenv})["M"] \
        .metadata.get("bit_depth") is None            # env still the (normalized) primary
    _g_rev, _e_rev = _measure_graph(True)             # wiring order must not matter
    _g_rev.edges.reverse()
    assert [e.dst_socket for e in _g_rev.dataset_preds("M")] == ["data", "raw"]
    # a geometry mismatch is silent corruption (voxel-for-voxel reads) → refuse
    axbad = AxisSizes(m=1, t=1, z=1, c=1, y=16, x=16)
    badds = Dataset(axes=axbad, metadata=rmeta).with_image(
        ArrayProvider(np.zeros((1, 1, 1, 1, 16, 16))))
    try:
        _measure_graph(True, raw_seed=badds,
                       raw_env=MetaEnvelope(axes=axbad, metadata=rmeta))[1].pull("M")
        raise SystemExit("a mis-shaped `raw` input should be refused")
    except ValueError as ex:
        assert "does not match the measured input" in str(ex), str(ex)
    # histogram_threshold: mask from the main pixels, intensities re-measured on raw
    _g = Graph()
    for nid, op in (("S", "io.kseed"), ("R", "io.kseed2")):
        _g.add(NodeInstance(nid, op))
    _g.add(NodeInstance("H", "analysis.histogram_threshold",
                        modes={"method": "single", "direction": "above"},
                        params={**NOCLEAN, "high": 500}))
    _g.connect("S", "H")
    _g.connect("R", "H", dst_socket="raw")
    _e = Engine(_g, computes=COMPUTES, seeds={"S": rawds, "R": rawds},
                meta_seeds={"S": rawenv, "R": rawenv})
    _o = _e.pull("H")
    assert int(_o.get(D.VOXEL, "labels").values.max()) == 2          # both objects
    assert np.allclose(sorted(_o.get(D.LABEL, "mean_intensity",
                                    layer="labels").values), [700.0, 1500.0])

    # ── §7c: a node that changes what the numbers MEAN restamps bit_depth ───────
    # Calibration describes the CURRENT data, so downstream nodes read the transformed
    # depth, not the file's: summing T=8 12-bit frames IS 15-bit data. Both halves must
    # agree — the edit-time meta_transform prediction AND the pulled payload.
    axv = AxisSizes(m=1, t=8, z=4, c=1, y=16, x=16)
    vimg = _rng.integers(0, 4096, (1, 8, 4, 1, 16, 16)).astype(float)
    vmeta = {"pixel_size_um": 0.25, "bit_depth": 12, "dt_s": 1.0, "z_step_um": 0.5}
    vds = Dataset(axes=axv, metadata=vmeta).with_image(ArrayProvider(vimg))
    venv = MetaEnvelope(axes=axv, metadata=vmeta)
    for _nodes, _edges, _want in (
            ([("A", "util.stack", {"modes": {"method": "sum"}})], [("S", "A")], 15),
            ([("A", "util.stack", {"modes": {"method": "mean"}})], [("S", "A")], 12),
            ([("A", "util.zproject", {"modes": {"method": "sum"}})], [("S", "A")], 14),
            ([("A", "util.zproject", {"modes": {"method": "max"}})], [("S", "A")], 12),
            ([("A", "enhance.clahe", {})], [("S", "A")], 12),   # rescales back ⇒ no change
            ([("A", "enhance.normalize", {})], [("S", "A")], None),        # [0,1] ⇒ dropped
            ([("A", "util.stack", {"modes": {"method": "sum"}}),           # …and it CHAINS
              ("B", "util.zproject", {"modes": {"method": "sum"}})],
             [("S", "A"), ("A", "B")], 17)):
        _g = Graph()
        _g.add(NodeInstance("S", "io.kseed"))
        for _nid, _op, _kw in _nodes:
            _g.add(NodeInstance(_nid, _op, **_kw))
        for _a, _b in _edges:
            _g.connect(_a, _b)
        _e = Engine(_g, computes=COMPUTES, seeds={"S": vds}, meta_seeds={"S": venv})
        _last = _nodes[-1][0]
        _env_bd = propagate_meta(_g, {"S": venv})[_last].metadata.get("bit_depth")
        _pay_bd = _e.pull(_last).metadata.get("bit_depth")
        assert _env_bd == _want, (_nodes, _env_bd, _want)      # edit-time prediction
        assert _pay_bd == _want, (_nodes, _pay_bd, _want)      # payload in lockstep
    # a raw-count consumer downstream of the widening reads the NEW depth (fenced)
    _g = Graph()
    _g.add(NodeInstance("S", "io.kseed"))
    _g.add(NodeInstance("A", "util.stack", modes={"method": "sum"}))
    _g.add(NodeInstance("H", "analysis.histogram_threshold", params=dict(NOCLEAN)))
    _g.connect("S", "A"); _g.connect("A", "H")
    _e = Engine(_g, computes=COMPUTES, seeds={"S": vds}, meta_seeds={"S": venv})
    _e.pull("H")
    assert "bit_depth" in {k for k, _ in _e.entry("H").reads}
    # analysis.threshold's FIXED level follows the declared scale, and degrades to the
    # 0.5 normalized-data default when a Normalize dropped it
    _tspec = NODES.get("analysis.threshold").input("threshold")
    for _md, _want in (({"bit_depth": 12}, 2047.5), ({"bit_depth": 16}, 32767.5),
                       ({}, 0.5)):
        assert eval_derive(_tspec.derive,
                           envelope_symbols(MetaEnvelope(axes=axv, metadata=_md))) == _want
    try:                        # a µm² filter with NO pixel size cannot be honoured
        cds = Dataset(axes=axc, metadata={}).with_image(ArrayProvider(cimg))
        eng(cds, MetaEnvelope(axes=axc, metadata={}),
            [("S", "io.kseed", {}),
             ("H", "analysis.histogram_threshold",
              {"params": {"min_area": 1.0}})], [("S", "H")]).pull("H")
        raise SystemExit("a µm² filter without pixel_size_um should be refused")
    except ValueError as ex:
        assert "no `pixel_size_um` calibration" in str(ex), str(ex)

    # the filter is authored in µm², so the Label domain reports µm² beside px² — the
    # column that makes "read the areas, then pick the cut" possible (2026-07-28)
    a_um2 = e_ma.pull("H").get(D.LABEL, "area_um2", layer="labels").values
    a_px = e_ma.pull("H").get(D.LABEL, "area", layer="labels").values
    assert np.allclose(np.sort(a_um2), [2.25, 2.25])          # 36 px² × 0.25² µm², 2 planes
    assert np.allclose(a_um2, a_px * 0.25 * 0.25)
    # an area window that discards EVERYTHING reports the areas that ARE there, in µm²
    try:
        eng(hds, henv, [("S", "io.kseed", {}),
                        ("H", "analysis.histogram_threshold",
                         {"modes": {"method": "relative", "direction": "above"},
                          "params": {**NOCLEAN, "min_area": 20.0}})], [("S", "H")]).pull("H")
        raise SystemExit("an area window that empties the series should say so")
    except ValueError as ex:
        assert "discarded every region" in str(ex) and "2.25" in str(ex), str(ex)
    try:                                    # …but blame the THRESHOLD when that is empty
        eng(hds, henv, [("S", "io.kseed", {}),
                        ("H", "analysis.histogram_threshold",
                         {"modes": {"method": "single", "direction": "above"},
                          "params": {**NOCLEAN, "high": 60000, "min_area": 1.0}})], [("S", "H")]).pull("H")
        raise SystemExit("an empty threshold should be named as the cause")
    except ValueError as ex:
        assert "THRESHOLD is what" in str(ex), str(ex)
    for _bad, _msg in (({"max_area": 0.01}, "under one pixel"),          # → 0 px² = OFF
                       ({"min_area": 5.0, "max_area": 3.0}, "size window is empty")):
        try:
            eng(hds, henv, [("S", "io.kseed", {}),
                            ("H", "analysis.histogram_threshold",
                             {"modes": {"method": "relative", "direction": "above"},
                              "params": {**NOCLEAN, **_bad}})], [("S", "H")]).pull("H")
            raise SystemExit(f"histogram_threshold should refuse {_bad}")
        except ValueError as ex:
            assert _msg in str(ex), str(ex)
    # a NORMALIZED [0,1] float image raises loudly (not silent quantization to {0,1})
    norm = np.full((1, 1, 1, 1, 16, 16), 0.2)
    norm[0, 0, 0, 0, 4:10, 4:10] = 0.9
    axn = AxisSizes(m=1, t=1, z=1, c=1, y=16, x=16)
    nds = Dataset(axes=axn, metadata={"pixel_size_um": 0.25}).with_image(ArrayProvider(norm))
    nenv = MetaEnvelope(axes=axn, metadata={"pixel_size_um": 0.25})
    try:
        eng(nds, nenv, [("S", "io.kseed", {}),
                        ("H", "analysis.histogram_threshold", {})], [("S", "H")]).pull("H")
        raise SystemExit("histogram_threshold should reject a normalized [0,1] image")
    except ValueError as ex:
        assert "raw integer counts" in str(ex)

    # ── registration recovers a known drift + stores Frame attrs ───────────────
    from scipy.ndimage import shift as _ndshift
    base = np.random.default_rng(0).random((40, 40))
    rimg = np.zeros((1, 4, 1, 1, 40, 40))
    for tt in range(4):
        rimg[0, tt, 0, 0] = _ndshift(base, (2.0 * tt, -1.5 * tt), order=1)
    axr = AxisSizes(m=1, t=4, z=1, c=1, y=40, x=40)
    rds = Dataset(axes=axr, metadata={"pixel_size_um": 0.1}).with_image(ArrayProvider(rimg))
    renv = MetaEnvelope(axes=axr, metadata={"pixel_size_um": 0.1})
    er = eng(rds, renv, [("S", "io.kseed", {}),
                         ("R", "registration.stabilize",
                          {"modes": {"model": "translation", "reference": "first"}})],
             [("S", "R")])
    dyv = er.pull("R").get(D.FRAME, "drift_y").values[0]
    assert np.allclose(dyv, [0.0, -2.0, -4.0, -6.0], atol=0.3), dyv.tolist()

    # ── boundary bands on a label volume (general, any labels); z_step fenced ──
    axg = AxisSizes(m=1, t=1, z=10, c=1, y=32, x=32)
    lab = np.zeros((1, 1, 10, 1, 32, 32), dtype=np.int64)
    lab[0, 0, 3:7, 0, 6:12, 6:12] = 1
    lab[0, 0, 3:7, 0, 20:26, 20:26] = 2
    gds = (Dataset(axes=axg, metadata={"pixel_size_um": 0.2, "z_step_um": 0.5})
           .with_image(ArrayProvider(np.zeros((1, 1, 10, 1, 32, 32))))
           .with_layer(D.VOXEL, "labels", lab))
    genv = MetaEnvelope(axes=axg, metadata={"pixel_size_um": 0.2, "z_step_um": 0.5})
    egb = eng(gds, genv, [("S", "io.kseed", {}),
                          ("B", "analysis.boundary_band", {"params": {"band_voxels": 1}})],
              [("S", "B")])
    bands = egb.pull("B").get(D.VOXEL, "bands").values
    assert int((bands == 1).sum()) > 0 and int((bands == 2).sum()) > 0
    assert int(((bands != 0) & (lab != 0)).sum()) == 0            # bands avoid interiors
    assert {"pixel_size_um", "z_step_um"} <= {k for k, _ in egb.entry("B").reads}
    # `band_um` is the EDT criterion only — the dilation path grows by band_voxels
    # iterations and never reads it, so it stays hidden under dilation. band_voxels is
    # live in BOTH (dilation iterations; the edt fallback threshold when band_um == 0).
    _bvis = lambda st: {i.name for i in NODES.get("analysis.boundary_band").active_inputs(st)}
    assert "band_um" in _bvis({"method": "edt"})
    assert "band_um" not in _bvis({"method": "dilation"})
    assert {"band_voxels", "include_neighbors"} <= _bvis({"method": "dilation"})
    assert {"band_voxels", "include_neighbors"} <= _bvis({"method": "edt"})

    # ── ROI mask: empty → whole-frame; a rect shape rasterizes correctly ───────
    axm = AxisSizes(m=1, t=1, z=1, c=1, y=20, x=20)
    mds = Dataset(axes=axm).with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 20, 20))))
    menv = MetaEnvelope(axes=axm)
    e_full = eng(mds, menv, [("S", "io.kseed", {}),
                             ("R", "analysis.roi_mask", {})], [("S", "R")])
    assert int(e_full.pull("R").get(D.VOXEL, "roi_mask").values.sum()) == 20 * 20
    e_rect = eng(mds, menv, [("S", "io.kseed", {}),
                             ("R", "analysis.roi_mask",
                              {"params": {"shapes": [{"type": "rect", "op": "add",
                                                      "vertices": [[5, 5], [15, 15]]}]}})],
                 [("S", "R")])
    rm = e_rect.pull("R").get(D.VOXEL, "roi_mask").values[0, 0, 0, 0]
    assert rm[8, 8] == 1 and rm[0, 0] == 0 and int(rm.sum()) > 0

    # ── dic_correlate is WIRED to its kernel (full 2D-DIC verification — real shift
    #    recovery when al-dic is present — lives in test_catalog_dic). Here: only the
    #    dep-gated path, when al-dic is ABSENT, must raise a clear ImportError. When it is
    #    present, skip the heavy solve in this group (covered end-to-end elsewhere). ──
    from nodegraph.kernels.dic_correlate import al_dic_available as _dic_avail
    if not _dic_avail():
        caught = 0
        try:
            eng(hds, henv, [("S", "io.kseed", {}), ("G", "analysis.dic_correlate", {})],
                [("S", "G")]).pull("G")
        except ImportError as ex:
            caught = int("al-dic" in str(ex))
        assert caught == 1, "dic_correlate did not raise a clear al-dic ImportError"
    dic_note = ("dic_correlate wired (al-dic present → verified in test_catalog_dic)"
                if _dic_avail() else "dic_correlate wired but dep-gated (al-dic ImportError)")

    _ok(f"catalog (v1 kernel ports): {beads_note}; histogram-threshold "
        "mask+labels+regions (methods re-key); registration recovers drift; boundary "
        f"bands (µm-fenced, any labels); ROI mask (empty→whole, rect); {dic_note}")


def test_catalog_dvc() -> None:
    """DVC/ALDVC displacement + strain field (``analysis.dvc_field``) → a Point field,
    and its Voxel rasterizer (``transform.rasterize_field``). V2.06. Asserts: 2D/3D
    recover a known shift (µm), the invariant Point schema + disp/strain/qfactor columns,
    2D≠3D recipe hash, calibration fenced (pixel_size_um / z_step_um), the reference-mode
    lever (previous_frame skips t0) + the optional external-reference input socket, and
    the rasterizer reproducing a planted linear field at grid points. Needs scipy/skimage."""
    if not _HAVE_SKIMAGE:
        _ok("catalog (DVC + field rasterize): SKIPPED (scipy/skimage absent)")
        return
    from scipy.ndimage import gaussian_filter, shift as ndshift
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.structure import StructureTable

    define_node("io.dvcseed", "S", outputs=[OutDataset()])

    def eng(seeds, envs, nodes, edges):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for e in edges:
            g.connect(e[0], e[1], dst_socket=(e[2] if len(e) > 2 else "data"))
        return Engine(g, computes=COMPUTES, seeds=seeds, meta_seeds=envs)

    rng = np.random.default_rng(3)
    patt = gaussian_filter(rng.random((64, 64)).astype(float), 1.5)
    px = 0.2
    P = {"subset_size": 16, "subset_spacing": 12, "admm_iterations": 1, "seed_levels": 1}

    # ── 2D fixed_frame, T=2: frame0=ref (self-pair→~0), frame1=shift (dy,dx) ──────
    img = np.zeros((1, 2, 1, 1, 64, 64))
    img[0, 0, 0, 0] = patt
    img[0, 1, 0, 0] = ndshift(patt, (1.5, -2.0), order=3, mode="nearest")
    ax = AxisSizes(m=1, t=2, z=1, c=1, y=64, x=64)
    meta = {"pixel_size_um": px}
    ds = Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(img))
    env = MetaEnvelope(axes=ax, metadata=meta)
    e2 = eng({"S": ds}, {"S": env},
             [("S", "io.dvcseed", {}),
              ("D", "analysis.dvc_field", {"modes": {"dim": "2D"}, "params": dict(P)})],
             [("S", "D")])
    o2 = e2.pull("D")
    names2 = {a.name for a in o2.layers_on(D.POINT) if a.layer == "dvc"}
    assert {"id", "m", "t", "c", "z", "y", "x", "disp_x", "disp_y", "disp_mag_um",
            "qfactor", "strain_yx"} <= names2, sorted(names2)
    assert "disp_z" not in names2, "2D field must not carry disp_z"
    tc = o2.get(D.POINT, "t", layer="dvc").values
    dy = o2.get(D.POINT, "disp_y", layer="dvc").values
    dx = o2.get(D.POINT, "disp_x", layer="dvc").values
    s1 = tc == 1
    assert abs(float(np.median(dy[s1])) - 1.5 * px) < 0.06, float(np.median(dy[s1]))
    assert abs(float(np.median(dx[s1])) + 2.0 * px) < 0.06, float(np.median(dx[s1]))
    assert float(np.median(o2.get(D.POINT, "disp_mag_um", layer="dvc").values[tc == 0])) < 0.02
    assert any(k == "pixel_size_um" for k, _ in e2.entry("D").reads)

    # ── 3D fixed_frame, T=2: disp_z present, recovers (z,y,x), z_step fenced ──────
    vol = gaussian_filter(rng.random((16, 40, 40)).astype(float), 1.2)
    img3 = np.zeros((1, 2, 16, 1, 40, 40))
    img3[0, 0, :, 0] = vol
    img3[0, 1, :, 0] = ndshift(vol, (1.0, 1.0, 1.5), order=3, mode="nearest")
    ax3 = AxisSizes(m=1, t=2, z=16, c=1, y=40, x=40)
    meta3 = {"pixel_size_um": px, "z_step_um": 0.5}
    ds3 = Dataset(axes=ax3, metadata=meta3).with_image(ArrayProvider(img3))
    e3 = eng({"S": ds3}, {"S": MetaEnvelope(axes=ax3, metadata=meta3)},
             [("S", "io.dvcseed", {}),
              ("D", "analysis.dvc_field",
               {"modes": {"dim": "3D"},
                "params": {"subset_size": 8, "subset_spacing": 10,
                           "admm_iterations": 1, "seed_levels": 1}})],
             [("S", "D")])
    o3 = e3.pull("D")
    t3 = o3.get(D.POINT, "t", layer="dvc").values == 1
    dz = o3.get(D.POINT, "disp_z", layer="dvc").values[t3]
    assert o3.get(D.POINT, "disp_z", layer="dvc") is not None
    assert abs(float(np.median(dz)) - 1.0 * 0.5) < 0.08, float(np.median(dz))
    assert abs(float(np.median(o3.get(D.POINT, "disp_x", layer="dvc").values[t3])) - 1.5 * px) < 0.08
    assert {"pixel_size_um", "z_step_um"} <= {k for k, _ in e3.entry("D").reads}
    assert e2.entry("D").recipe_hash != e3.entry("D").recipe_hash, "2D/3D lever must re-key"
    # 3D mode on a single-plane series is a hard error (kernel gotcha 6)
    axf = AxisSizes(m=1, t=1, z=1, c=1, y=32, x=32)
    dsf = Dataset(axes=axf, metadata=meta).with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 32, 32))))
    try:
        eng({"S": dsf}, {"S": MetaEnvelope(axes=axf, metadata=meta)},
            [("S", "io.dvcseed", {}),
             ("D", "analysis.dvc_field", {"modes": {"dim": "3D"}})], [("S", "D")]).pull("D")
        raise SystemExit("3D DVC on a single plane should raise")
    except ValueError as ex:
        assert "z>1" in str(ex)

    # ── optional external reference input socket (a second file as the reference) ─
    imgp = np.zeros((1, 1, 1, 1, 64, 64)); imgp[0, 0, 0, 0] = ndshift(patt, (2.0, 0.0), order=3, mode="nearest")
    imgr = np.zeros((1, 1, 1, 1, 64, 64)); imgr[0, 0, 0, 0] = patt
    axp = AxisSizes(m=1, t=1, z=1, c=1, y=64, x=64)
    ep = MetaEnvelope(axes=axp, metadata=meta)
    ex = eng({"S": Dataset(axes=axp, metadata=meta).with_image(ArrayProvider(imgp)),
              "Sr": Dataset(axes=axp, metadata=meta).with_image(ArrayProvider(imgr))},
             {"S": ep, "Sr": ep},
             [("S", "io.dvcseed", {}), ("Sr", "io.dvcseed", {}),
              ("D", "analysis.dvc_field", {"modes": {"dim": "2D"}, "params": dict(P)})],
             [("S", "D", "data"), ("Sr", "D", "reference")])
    dyx = ex.pull("D").get(D.POINT, "disp_y", layer="dvc").values
    assert abs(float(np.median(dyx)) - 2.0 * px) < 0.06, float(np.median(dyx))
    # regression: the compute's calibration env is the PRIMARY (data) input's, even when
    # the reference is wired FIRST and carries a different pixel size — propagate_meta
    # must seed from the first-declared dataset socket, not edge-insertion order.
    metaR = {"pixel_size_um": 0.9}                              # a wrong scale if picked
    exr = eng({"S": Dataset(axes=axp, metadata=meta).with_image(ArrayProvider(imgp)),
               "Sr": Dataset(axes=axp, metadata=metaR).with_image(ArrayProvider(imgr))},
              {"S": ep, "Sr": MetaEnvelope(axes=axp, metadata=metaR)},
              [("S", "io.dvcseed", {}), ("Sr", "io.dvcseed", {}),
               ("D", "analysis.dvc_field", {"modes": {"dim": "2D"}, "params": dict(P)})],
              [("Sr", "D", "reference"), ("S", "D", "data")])   # reference wired FIRST
    dyr = exr.pull("D").get(D.POINT, "disp_y", layer="dvc").values
    # µm scale ⇒ primary px=0.2 (≈0.4 µm), NOT the reference px=0.9 (≈1.8 µm)
    assert abs(float(np.median(dyr)) - 2.0 * px) < 0.06, \
        f"DVC used the reference's calibration ({float(np.median(dyr)):.3f} µm ⇒ px≈0.9)"
    assert any(k == "pixel_size_um" for k, _ in exr.entry("D").reads)

    # ── previous_frame self-reference skips t0 (no increment into the first frame) ─
    imgt = np.zeros((1, 3, 1, 1, 64, 64))
    for tt in range(3):
        imgt[0, tt, 0, 0] = ndshift(patt, (1.0 * tt, 0.0), order=3, mode="nearest")
    axt = AxisSizes(m=1, t=3, z=1, c=1, y=64, x=64)
    ept = eng({"S": Dataset(axes=axt, metadata=meta).with_image(ArrayProvider(imgt))},
              {"S": MetaEnvelope(axes=axt, metadata=meta)},
              [("S", "io.dvcseed", {}),
               ("D", "analysis.dvc_field",
                {"modes": {"dim": "2D", "reference_mode": "previous_frame"},
                 "params": dict(P)})], [("S", "D")])
    tp = ept.pull("D").get(D.POINT, "t", layer="dvc").values
    assert 0 not in set(tp.tolist()) and {1, 2} <= set(tp.tolist()), sorted(set(tp.tolist()))

    # ── accumulate: previous_frame increments → cumulative Lagrangian field ────────
    # imgt shifts +1 voxel/frame, so each increment ≈ 1·px and the cumulative field is
    # 1·px at t=1, 2·px at t=2 (composition, not a fixed-frame re-correlation).
    ea = eng({"S": Dataset(axes=axt, metadata=meta).with_image(ArrayProvider(imgt))},
             {"S": MetaEnvelope(axes=axt, metadata=meta)},
             [("S", "io.dvcseed", {}),
              ("D", "analysis.dvc_field",
               {"modes": {"dim": "2D", "reference_mode": "previous_frame"},
                "params": dict(P)}),
              ("A", "analysis.accumulate_field", {})],
             [("S", "D"), ("D", "A")])
    oa = ea.pull("A")
    na = {a.name for a in oa.layers_on(D.POINT) if a.layer == "dvc_cumulative"}
    assert {"id", "m", "t", "c", "z", "y", "x", "disp_x", "disp_y", "disp_mag_um",
            "qfactor", "strain_yx"} <= na, sorted(na)
    ta = oa.get(D.POINT, "t", layer="dvc_cumulative").values
    dya = oa.get(D.POINT, "disp_y", layer="dvc_cumulative").values
    assert {1, 2} == set(ta.tolist()), sorted(set(ta.tolist()))
    assert abs(float(np.median(dya[ta == 1])) - 1.0 * px) < 0.06, float(np.median(dya[ta == 1]))
    assert abs(float(np.median(dya[ta == 2])) - 2.0 * px) < 0.08, float(np.median(dya[ta == 2]))
    assert any(k == "pixel_size_um" for k, _ in ea.entry("A").reads)   # calib fenced
    # guard: a fixed_frame (already cumulative) field is refused
    try:
        eng({"S": Dataset(axes=axt, metadata=meta).with_image(ArrayProvider(imgt))},
            {"S": MetaEnvelope(axes=axt, metadata=meta)},
            [("S", "io.dvcseed", {}),
             ("D", "analysis.dvc_field", {"modes": {"dim": "2D"}, "params": dict(P)}),
             ("A", "analysis.accumulate_field", {})], [("S", "D"), ("D", "A")]).pull("A")
        raise SystemExit("accumulate on a fixed_frame field should raise")
    except ValueError as exa:
        assert "already cumulative" in str(exa), str(exa)
    # guard: a Point field with no DVC provenance is refused
    axg = AxisSizes(m=1, t=1, z=1, c=1, y=8, x=8)
    tblg = StructureTable(D.POINT, {
        "id": np.arange(2, dtype=np.int64), "m": np.zeros(2, np.int64),
        "t": np.zeros(2, np.int64), "c": np.zeros(2, np.int64),
        "z": np.zeros(2), "y": np.array([2.0, 4.0]), "x": np.array([2.0, 4.0]),
        "disp_y": np.zeros(2), "disp_x": np.zeros(2),
    }, layer="dvc", z_kind="plane_index")
    dsg = (Dataset(axes=axg, metadata={"pixel_size_um": px})
           .with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 8, 8)))).with_structure(tblg))
    try:
        eng({"S": dsg}, {"S": MetaEnvelope(axes=axg, metadata={"pixel_size_um": px})},
            [("S", "io.dvcseed", {}),
             ("A", "analysis.accumulate_field", {})], [("S", "A")]).pull("A")
        raise SystemExit("accumulate without dvc provenance should raise")
    except ValueError as exb:
        assert "provenance" in str(exb), str(exb)

    # ── rasterize a planted linear Point field → Voxel layer reproduces it ────────
    # Dimensionality is INHERITED from the table's z_kind (§7b): plane_index → 2D per-plane,
    # no dim lever passed. `with_structure` preserved z_kind into `__struct_zkind__`.
    axi = AxisSizes(m=1, t=1, z=1, c=1, y=20, x=20)
    yy, xx = np.meshgrid([4.0, 9.0, 14.0], [4.0, 9.0, 14.0], indexing="ij")
    yy, xx = yy.ravel(), xx.ravel()
    tbl = StructureTable(D.POINT, {
        "id": np.arange(len(yy), dtype=np.int64), "m": np.zeros(len(yy), np.int64),
        "t": np.zeros(len(yy), np.int64), "c": np.zeros(len(yy), np.int64),
        "z": np.zeros(len(yy)), "y": yy, "x": xx,
        "disp_x": xx.copy(), "disp_y": np.zeros(len(yy)),
    }, layer="dvc", z_kind="plane_index")
    dsi = (Dataset(axes=axi, metadata={}).with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 20, 20))))
           .with_structure(tbl))
    assert dsi.structure_zkind(D.POINT, "dvc") == "plane_index"        # z_kind preserved
    orr = eng({"S": dsi}, {"S": MetaEnvelope(axes=axi, metadata={})},
              [("S", "io.dvcseed", {}),
               ("R", "transform.rasterize_field", {})],
              [("S", "R")]).pull("R")
    rx = orr.get(D.VOXEL, "dvc_disp_x").values
    assert rx.shape == (1, 1, 1, 1, 20, 20) and np.isfinite(rx).all()
    assert abs(float(rx[0, 0, 0, 0, 9, 9]) - 9.0) < 0.5 and abs(float(rx[0, 0, 0, 0, 9, 14]) - 14.0) < 0.5
    # §7b regression: a 2D per-plane field (z_kind=plane_index) on a z>1 image must NOT be
    # misread as a 3D grid (which — with no lever — the old default would do from ax.z>1,
    # bleeding a single plane's points across all z). Inherited z_kind keeps it per-plane.
    axv = AxisSizes(m=1, t=1, z=3, c=1, y=16, x=16)
    yv, xv = np.meshgrid([3.0, 8.0, 12.0], [3.0, 8.0, 12.0], indexing="ij")
    yv, xv = yv.ravel(), xv.ravel()
    tblv = StructureTable(D.POINT, {
        "id": np.arange(len(yv), dtype=np.int64), "m": np.zeros(len(yv), np.int64),
        "t": np.zeros(len(yv), np.int64), "c": np.zeros(len(yv), np.int64),
        "z": np.full(len(yv), 1.0), "y": yv, "x": xv, "disp_x": xv.copy(),
    }, layer="dvc", z_kind="plane_index")
    dsv = (Dataset(axes=axv, metadata={}).with_image(ArrayProvider(np.zeros((1, 1, 3, 1, 16, 16))))
           .with_structure(tblv))
    rxv = eng({"S": dsv}, {"S": MetaEnvelope(axes=axv, metadata={})},
              [("S", "io.dvcseed", {}), ("R", "transform.rasterize_field", {})],
              [("S", "R")]).pull("R").get(D.VOXEL, "dvc_disp_x").values
    assert np.count_nonzero(rxv[0, 0, 0]) == 0 and np.count_nonzero(rxv[0, 0, 2]) == 0, \
        "2D per-plane field bled into other z planes (misread as 3D — §7b inheritance failed)"
    assert np.isfinite(rxv[0, 0, 1]).all() and abs(float(rxv[0, 0, 1, 0, 8, 8]) - 8.0) < 0.5
    # rasterize without a matching Point layer is a hard error
    try:
        eng({"S": Dataset(axes=axi, metadata={}).with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 20, 20))))},
            {"S": MetaEnvelope(axes=axi, metadata={})},
            [("S", "io.dvcseed", {}),
             ("R", "transform.rasterize_field", {})], [("S", "R")]).pull("R")
        raise SystemExit("rasterize without a Point layer should raise")
    except ValueError as ex2:
        assert "Point layer" in str(ex2)

    _ok("catalog (DVC + field rasterize): 2D/3D recover known shift (µm), Point schema "
        "+ disp/strain/qfactor; 2D≠3D hash; pixel/z-step fenced; external-ref socket + "
        "previous-frame skips t0; accumulate composes increments → cumulative (t·px, "
        "provenance-inherited, refuses fixed_frame/no-provenance); rasterizer reproduces "
        "a linear field + inherits z_kind (§7b: 2D field on z>1 not misread as 3D); "
        "guards (z<2, missing Point) raise")


def test_catalog_dic() -> None:
    """DIC (pyALDIC, ``analysis.dic_correlate``) — the 2D image sibling of DVC, WIRED to the
    vendored kernel. ``al-dic`` is an optional dep: when absent the node is fully wired but
    its compute raises a friendly ImportError at run time (it reaches ``run_pyaldic_pair`` —
    not a bare stub that errors before touching inputs). Asserts the structural spec + the
    gated-run behavior (and, if al-dic is installed, a real Point-field pull)."""
    if not _HAVE_SKIMAGE:
        _ok("catalog (DIC pyALDIC): SKIPPED (scipy/skimage absent)")
        return
    from scipy.ndimage import gaussian_filter, shift as ndshift
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.kernels.dic_correlate import al_dic_available

    # structural spec — really wired (Point output, reference lever + ROI, WHOLE_SERIES),
    # not the old bare-stub registration
    s = NODES.get("analysis.dic_correlate")
    assert s.category == "analysis"
    assert D.POINT in s.adds_domains and D.VOXEL in s.reads_domains
    assert s.granularity is Granularity.WHOLE_SERIES
    assert {"data", "reference", "reference_frame", "winsize", "winstepsize", "roi"} \
        <= {i.name for i in s.inputs}
    assert any(mm.name == "reference_mode" for mm in s.modes)

    # a 2-frame 2D graph (frame0 = reference, frame1 = a known (Δy, Δx) shift). fixed_frame
    # (default) pairs each frame against t0, so t=1 carries the planted shift.
    define_node("io.dicseed", "S", outputs=[OutDataset()])
    rng = np.random.default_rng(5)
    patt = gaussian_filter(rng.random((128, 128)), 1.0)      # speckle-like texture
    img = np.zeros((1, 2, 1, 1, 128, 128))
    img[0, 0, 0, 0] = patt
    img[0, 1, 0, 0] = ndshift(patt, (1.0, 3.0), order=3, mode="nearest")   # +1 y, +3 x px
    px = 0.5
    ax = AxisSizes(m=1, t=2, z=1, c=1, y=128, x=128)
    meta = {"pixel_size_um": px}
    ds = Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(img))
    g = Graph()
    g.add(NodeInstance("S", "io.dicseed"))
    g.add(NodeInstance("D", "analysis.dic_correlate",
                       params={"winsize": 20, "winstepsize": 16,
                               "admm_max_iter": 4, "icgn_max_iter": 100}))
    g.connect("S", "D", dst_socket="data")
    e = Engine(g, computes=COMPUTES, seeds={"S": ds},
               meta_seeds={"S": MetaEnvelope(axes=ax, metadata=meta)})

    if not al_dic_available():
        raised = ""                                         # dep-gated env: friendly error
        try:
            e.pull("D")
        except ImportError as ex:
            raised = str(ex)
        assert "al-dic" in raised.lower(), f"expected a friendly al-dic ImportError, got {raised!r}"
        _ok("catalog (DIC pyALDIC): wired 2D DIC node (reference lever + ROI + Point output, "
            "WHOLE_SERIES); dep-gated — reaches run_pyaldic_pair, friendly al-dic ImportError "
            "when the solver is absent")
        return

    # al-dic present → a real correlation must recover the planted shift (µm) + the Point schema
    out = e.pull("D")                                       # runs IC-GN + ADMM (numba JIT: slow once)
    names = {a.name for a in out.layers_on(D.POINT) if a.layer == "dic"}
    assert {"id", "m", "t", "c", "z", "y", "x", "disp_x", "disp_y", "disp_mag_um"} <= names, \
        sorted(names)
    assert "disp_z" not in names and "strain_yx" not in names, "DIC is 2D and stores no strain"
    tc = out.get(D.POINT, "t", layer="dic").values
    dy = out.get(D.POINT, "disp_y", layer="dic").values
    dx = out.get(D.POINT, "disp_x", layer="dic").values
    s1 = tc == 1
    assert abs(float(np.median(dy[s1])) - 1.0 * px) < 0.12, float(np.median(dy[s1]))
    assert abs(float(np.median(dx[s1])) - 3.0 * px) < 0.12, float(np.median(dx[s1]))
    assert float(np.median(out.get(D.POINT, "disp_mag_um", layer="dic").values[tc == 0])) < 0.1
    assert any(k == "pixel_size_um" for k, _ in e.entry("D").reads)   # calib fenced
    _ok("catalog (DIC pyALDIC): wired 2D DIC (reference lever + ROI + Point output, "
        "WHOLE_SERIES) ran end-to-end — recovers a planted (1,3)px shift in µm, self-pair≈0, "
        "calib fenced (al-dic present)")


def test_channel_derive() -> None:
    """C8 / H12 — per-channel derive resolution (``ctx.channel``). A c-iterating node's
    metadata-intelligent param DERIVES from *that channel's* emission λ (distinct λ →
    distinct value); a user override applies to all channels; the derive's optics deps are
    memo-fenced. Also checks ``detect.spots`` resolves its radii per channel end-to-end."""
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES, register_node

    # a probe node (FAKE op_key — never clobber a real node) whose one param derives from
    # emission λ; capture what ctx.channel(c) resolves for each channel.
    cap: dict = {}

    def _c8(ctx):
        ax = ctx.inputs[0].image.axes
        cap["per_ch"] = [ctx.channel(c).param("radius") for c in range(ax.c)]
        cap["emis"] = [ctx.channel(c).emission_nm() for c in range(ax.c)]
        return ctx.inputs[0]

    register_node(_c8, op_key="test.c8_probe", label="c8 probe", category="analysis",
                  inputs=[InDataset(), InFloat("radius", "R", unit="um", field=True,
                          derive="0.61*(emission_nm or 520)/(na or 1.4)/1000")],
                  outputs=[OutDataset()])
    define_node("io.c8seed", "S", outputs=[OutDataset()])

    ax = AxisSizes(m=1, t=1, z=1, c=2, y=4, x=4)
    meta = {"objective_na": 1.0, "channel_emission_nm": [450, 650]}
    ds = Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(np.zeros((1, 1, 1, 2, 4, 4))))
    env = MetaEnvelope(axes=ax, metadata=meta)

    def run(params):
        g = Graph(); g.add(NodeInstance("S", "io.c8seed"))
        g.add(NodeInstance("P", "test.c8_probe", params=params)); g.connect("S", "P")
        e = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env})
        e.pull("P"); return e

    # derive path: each channel resolves from ITS OWN emission (0.61·λ/NA)
    e = run({})
    r = cap["per_ch"]
    assert abs(r[0] - 0.61 * 450 / 1.0 / 1000) < 1e-9, r
    assert abs(r[1] - 0.61 * 650 / 1.0 / 1000) < 1e-9, r
    assert r[0] != r[1], "per-channel derive collapsed to a single value"
    assert cap["emis"] == [450, 650]
    reads = {k for k, _ in e.entry("P").reads}
    assert {"channel_emission_nm", "objective_na"} <= reads, sorted(reads)   # deps fenced
    # override path: an explicit param wins and applies to ALL channels (one widget)
    run({"radius": 0.9})
    assert cap["per_ch"] == [0.9, 0.9], cap["per_ch"]

    # end-to-end: detect.spots resolves radii per channel (no radius params → derive path),
    # detects the blob in BOTH channels, and now memo-fences on per-channel emission.
    if _HAVE_SKIMAGE:
        yy, xx = np.mgrid[0:24, 0:24]
        blob = np.exp(-(((yy - 12) ** 2 + (xx - 12) ** 2) / (2 * 2.0 ** 2)))
        img = np.zeros((1, 1, 1, 2, 24, 24)); img[0, 0, 0, 0] = blob; img[0, 0, 0, 1] = blob
        axs = AxisSizes(m=1, t=1, z=1, c=2, y=24, x=24)
        ms = {"pixel_size_um": 0.1, "objective_na": 1.0, "channel_emission_nm": [450, 650]}
        dss = Dataset(axes=axs, metadata=ms).with_image(ArrayProvider(img))
        gs = Graph(); gs.add(NodeInstance("S", "io.c8seed"))
        gs.add(NodeInstance("N", "detect.spots", modes={"dim": "2D"},
                            params={"threshold": 0.03}))
        gs.connect("S", "N")
        es = Engine(gs, computes=COMPUTES, seeds={"S": dss},
                    meta_seeds={"S": MetaEnvelope(axes=axs, metadata=ms)})
        cc = es.pull("N").get(D.POINT, "c", layer="spots")
        assert cc is not None and set(cc.values.tolist()) == {0, 1}, "spots missed a channel"
        assert "channel_emission_nm" in {k for k, _ in es.entry("N").reads}, \
            "detect.spots not memo-fenced on per-channel emission"

    _ok("channel derive (C8/H12): ctx.channel per-c derive (distinct λ → distinct value), "
        "override applies to all channels, optics deps fenced; detect.spots resolves radii "
        "per channel end-to-end")


def test_cluster_points() -> None:
    """``analysis.cluster_points`` (V2.07; ported v1 ``granule_cluster``, scikit-learn) — an
    unlabeled 3-D Point cloud → a per-point cluster-id column via a GaussianMixture/BIC fit.
    Asserts it recovers two planted clusters (relax=0 ⇒ k=2), preserves the source z_kind,
    fences calibration, that gmm≠kmeans re-keys the memo, and that it chains into
    ``tessellate`` → ``rasterize_mesh``. Skips if scikit-learn is absent."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        _ok("cluster points: SKIPPED (scikit-learn absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.structure import StructureTable as _ST

    define_node("io.cpseed", "S", outputs=[OutDataset()])
    # two well-separated 3-D clusters, 6 non-coplanar points each (>= min_granule_points)
    star = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    A = star + np.array([5.0, 12, 12])
    B = star + np.array([12.0, 40, 40])
    P = np.vstack([A, B]); n = len(P)
    tbl = _ST(D.POINT, {
        "id": np.arange(n, dtype=np.int64), "m": np.zeros(n, np.int64),
        "t": np.zeros(n, np.int64), "c": np.zeros(n, np.int64),
        "z": P[:, 0], "y": P[:, 1], "x": P[:, 2]}, layer="particles", z_kind="subpixel")
    ax = AxisSizes(m=1, t=1, z=18, c=1, y=56, x=56)
    meta = {"pixel_size_um": 0.2, "z_step_um": 0.5}
    ds = (Dataset(axes=ax, metadata=meta)
          .with_image(ArrayProvider(np.zeros((1, 1, 18, 1, 56, 56)))).with_structure(tbl))
    env = MetaEnvelope(axes=ax, metadata=meta)

    def eng(nodes, edges, sink):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for e in edges:
            g.connect(e[0], e[1], dst_socket=(e[2] if len(e) > 2 else "data"))
        e = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env})
        return e, e.pull(sink)

    e, o = eng([("S", "io.cpseed", {}),
                ("C", "analysis.cluster_points",
                 {"params": {"n_clusters": 2, "relax_pct": 0.0}})], [("S", "C")], "C")
    clus = o.get(D.POINT, "cluster", layer="particles").values
    assert set(clus.tolist()) == {0, 1}, sorted(set(clus.tolist()))     # relax=0 ⇒ k=2
    # each planted cluster is one id, and the two differ (ids themselves are arbitrary)
    assert len(set(clus[:6].tolist())) == 1 and len(set(clus[6:].tolist())) == 1
    assert clus[0] != clus[6], "the two planted clusters collapsed into one"
    assert o.structure_zkind(D.POINT, "particles") == "subpixel"        # source z_kind kept
    assert {"pixel_size_um", "z_step_um"} <= {k for k, _ in e.entry("C").reads}
    # the model mode folds into the recipe hash (gmm vs kmeans memoize separately)
    eg, _ = eng([("S", "io.cpseed", {}), ("C", "analysis.cluster_points",
                 {"modes": {"method": "gmm"}, "params": {"n_clusters": 2}})], [("S", "C")], "C")
    ek, _ = eng([("S", "io.cpseed", {}), ("C", "analysis.cluster_points",
                 {"modes": {"method": "kmeans"}, "params": {"n_clusters": 2}})], [("S", "C")], "C")
    assert eg.entry("C").recipe_hash != ek.entry("C").recipe_hash, "method must re-key"
    # full granule-chain hand-off: cluster_points → tessellate → rasterize_mesh → 2 regions
    _, ot = eng([("S", "io.cpseed", {}),
                 ("C", "analysis.cluster_points", {"params": {"n_clusters": 2, "relax_pct": 0.0}}),
                 ("T", "analysis.tessellate", {}),
                 ("R", "transform.rasterize_mesh", {})],
                [("S", "C"), ("C", "T"), ("T", "R")], "R")
    rast = ot.get(D.VOXEL, "labels").values
    assert set(np.unique(rast).tolist()) == {0, 1, 2}                   # bg + 2 regions
    assert np.all(ot.get(D.LABEL, "volume_um3", layer="labels").values > 0)

    _ok("cluster points (V2.07): GMM/BIC recovers 2 planted clusters (relax=0⇒k=2) → per-point "
        "cluster column; z_kind kept; calib fenced; gmm≠kmeans re-keys; chains into "
        "tessellate → rasterize_mesh (particles→cluster→mesh→raster)")


def test_mesh_domain() -> None:
    """``Domain.MESH`` (V2.08) — the eleventh domain, stored as three flat CSR strata under
    one domain addressed by layer sub-keys (``L`` / ``L/vert`` / ``L/face``).

    Asserts: the domain predicates (structure, NOT lattice, no axis-set, a clean raise from
    the lattice-only helpers); a build → ``with_mesh`` → ``read_mesh`` round-trip preserving
    all three bucket lengths + the per-(m,t,c) dense ids; ``content_hash`` EQUAL across two
    independently built identical meshes (the positive assertion that closes the
    object-dtype pointer-hash trap) and ``_canon`` refusing object dtype outright; every
    ``validate()`` invariant; three independent ``__struct_zkind__`` stamps; the clean
    plan-time transfer refusal (MESH registers no bridge on purpose); and the face-parity
    interior test exact on a convex box AND a concave L-prism. numpy only."""
    from nodegraph.domains import (DOMAIN_ABBR, DOMAIN_COLOR, STRUCTURE_DOMAINS,
                                   domain_abbr, domain_color, is_lattice, is_structure)
    from nodegraph.kernels.mesh_raster import FaceParity
    from nodegraph.memo import _canon
    from nodegraph.structure import COORD_COLUMNS
    from nodegraph.transfer import plan_transfer
    from dataclasses import replace
    import nodegraph.mesh as MSH

    # ── the enum member + its presentation ────────────────────────────────────────
    assert D.MESH.value == "mesh" and D.MESH in STRUCTURE_DOMAINS
    assert is_structure(D.MESH) and not is_lattice(D.MESH)
    assert axes_of(D.MESH) is None                       # omission IS the non-lattice decl
    # DOMAIN_COLOR must be populated in the SAME commit as the member: nodelab_v2.theme
    # iterates the whole enum to build its QColor map, so a gap crashes the GUI at import.
    assert D.MESH in DOMAIN_COLOR and D.MESH in DOMAIN_ABBR
    assert domain_color(D.MESH) == DOMAIN_COLOR[D.MESH] and domain_abbr(D.MESH) == "MSH"
    for bad in (lambda: AX.shape_for(D.MESH), lambda: AX.axis_list(D.MESH)):
        try:
            bad()
            raise SystemExit("a lattice-only helper must reject MESH")
        except ValueError:
            pass

    # ── a hand-built 3-element mesh spanning two (m,t,c) frames ───────────────────
    def cube_mesh(z0, y0, x0, s=4.0):
        V = np.array([[z0 + dz, y0 + dy, x0 + dx]
                      for dz in (0.0, s) for dy in (0.0, s) for dx in (0.0, s)], float)
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        F = []
        for a, b, c, d in quads:
            F += [[a, b, c], [a, c, d]]
        return V, np.array(F, np.int64)

    def elements():
        out = []
        for i, (yo, mm) in enumerate([(2.0, 0), (12.0, 0), (2.0, 1)]):
            V, F = cube_mesh(2.0, yo, 2.0)
            out.append(MSH.MeshElement(m=mm, t=0, c=0, src_label=7 + i, verts_zyx=V,
                                       faces=F, centroid_zyx=(4.0, yo + 2.0, 4.0),
                                       volume_um3=1.5, surface_area_um2=2.5,
                                       density=float(3 - i), n_points=8))
        return out

    tb = MSH.build_mesh_tables(elements(), layer="surf")
    assert (tb.element.n, tb.vertex.n, tb.face.n) == (3, 24, 36)
    ec = tb.element.columns
    assert np.asarray(ec["id"]).tolist() == [1, 2, 1]           # dense per (m,t,c)
    assert np.asarray(ec["element_uid"]).tolist() == [0, 1, 2]  # global FK, row order
    assert np.asarray(ec["m"]).tolist() == [0, 0, 1]
    assert np.asarray(ec["closed"]).tolist() == [1, 1, 1]       # derived, watertight
    assert np.asarray(ec["src_label"]).tolist() == [7, 8, 9]    # original ids survive
    assert np.asarray(ec["vert_start"]).tolist() == [0, 8, 16]
    # every bucket carries the invariant coordinate schema; a face row is topology, so it
    # carries no centroid (the one documented exception, see nodegraph.mesh).
    for tbl, need in ((tb.element, COORD_COLUMNS), (tb.vertex, COORD_COLUMNS),
                      (tb.face, ("id", "m", "t", "c"))):
        assert set(need) <= set(tbl.columns), sorted(tbl.columns)
    for tbl in (tb.element, tb.vertex, tb.face):
        for col, v in tbl.columns.items():
            a = np.asarray(v)
            assert a.dtype != object and a.ndim == 1, (tbl.layer, col, a.dtype)

    # ── memo identity: flat columns hash by CONTENT, object dtype is refused ───────
    assert MSH.build_mesh_tables(elements(), layer="surf").content_hash() == tb.content_hash()
    obj = np.empty(2, dtype=object)
    obj[0] = np.zeros(3)
    obj[1] = np.zeros(2)
    try:
        _canon(obj)
        raise SystemExit("_canon must refuse an object-dtype array (it hashes pointers)")
    except TypeError as ex:
        assert "object-dtype" in str(ex)

    # ── attach / read round-trip + three independent z_kind stamps ────────────────
    ds = Dataset(axes=AxisSizes(m=2, t=1, z=16, c=1, y=32, x=32))
    ds = MSH.with_mesh(ds, tb, provenance={"boundary": "convex_hull", "source": "points"})
    assert MSH.mesh_names(ds) == ("surf",)
    for lay in ("surf", "surf/vert", "surf/face"):
        assert ds.structure_zkind(D.MESH, lay) == "subpixel", lay
    rt = MSH.read_mesh(ds, "surf")
    assert rt.content_hash() == tb.content_hash(), "mesh did not survive the store"
    assert MSH.mesh_provenance(ds, "surf")["boundary"] == "convex_hull"
    v1, f1 = MSH.mesh_element(rt, 1)                  # element 1 slices out, faces LOCAL
    assert v1.shape == (8, 3) and f1.min() == 0 and f1.max() == 7
    assert np.allclose(v1, cube_mesh(2.0, 12.0, 2.0)[0])
    assert MSH.mesh_layer("q", "vert") == "q/vert"
    assert MSH.mesh_part("q/vert") == ("q", "vert") and MSH.mesh_part("plain") == ("plain", None)

    # ── validate() is the ONLY enforcement (the store skips non-lattice checks) ────
    def broken(**cols):
        c = dict(tb.element.columns)
        c.update(cols)
        bad = MSH.MeshTables(replace(tb.element, columns=c), tb.vertex, tb.face, layer="surf")
        try:
            bad.validate()
        except ValueError:
            return True
        return False
    assert broken(vert_count=np.array([8, 8, 9], np.int64)), "CSR sum must be checked"
    assert broken(vert_start=np.array([0, 9, 16], np.int64)), "CSR prefix must be checked"
    assert broken(element_uid=np.array([1, 2, 3], np.int64)), "the FK target must be dense"
    assert broken(id=np.array([1, 3, 1], np.int64)), "ids must be dense per (m,t,c)"
    try:
        MSH.build_mesh_tables(elements(), layer="a/vert")
        raise SystemExit("the stratum separator must be reserved in a mesh name")
    except ValueError as ex:
        assert "reserved" in str(ex)
    # the store itself really does NOT length-check a non-lattice layer — which is exactly
    # why with_mesh() has to validate, and why nothing may bypass it.
    torn = Dataset().with_layer(D.MESH, "a", np.zeros(5), layer="surf") \
                   .with_layer(D.MESH, "b", np.zeros(3), layer="surf")
    assert len(torn.layers_on(D.MESH)) == 2

    # ── no bridge: MESH must fail at PLAN time, not with a wrong answer ────────────
    for dst in (D.VOXEL, D.LABEL, D.FRAME):
        try:
            plan_transfer(D.MESH, dst)
            raise SystemExit(f"MESH->{dst} must have no transfer route")
        except ValueError as ex:
            assert "route" in str(ex).lower(), str(ex)

    # ── the face-parity interior test, exact on convex AND concave ────────────────
    zz, yy, xx = np.mgrid[0:9, 0:9, 0:9]
    q = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(float)
    Vb, Fb = cube_mesh(2.0, 2.0, 2.0)
    box = FaceParity(Vb, Fb, (1.0, 1.0, 1.0)).contains(q).reshape(9, 9, 9)
    want = (zz >= 2) & (zz < 6) & (yy >= 2) & (yy < 6) & (xx >= 2) & (xx < 6)
    assert np.array_equal(box, want), "parity must be exact on an axis-aligned box"
    # an L-shaped prism: the case a convex-hull test CANNOT represent
    poly = np.array([[2, 2], [2, 8], [4, 8], [4, 4], [8, 4], [8, 2]], float)
    npv = len(poly)
    VL = np.vstack([np.column_stack([np.full(npv, 2.0), poly]),
                    np.column_stack([np.full(npv, 6.0), poly])])
    FL = []
    for i in range(npv):
        j = (i + 1) % npv
        FL += [[i, j, npv + i], [j, npv + j, npv + i]]
    for i in range(1, npv - 1):
        FL += [[0, i, i + 1], [npv, npv + i + 1, npv + i]]
    FL = np.array(FL, np.int64)
    assert MSH.faces_are_closed(FL) and not MSH.faces_are_closed(FL[:4])
    gotL = FaceParity(VL, FL, (1.0, 1.0, 1.0)).contains(q).reshape(9, 9, 9)
    inL = (((yy >= 2) & (yy < 4) & (xx >= 2) & (xx < 8))
           | ((yy >= 4) & (yy < 8) & (xx >= 2) & (xx < 4)))
    assert np.array_equal(gotL, (zz >= 2) & (zz < 6) & inL), "parity must honor concavity"
    assert int(gotL.sum()) < int(((zz >= 2) & (zz < 6) & (yy >= 2) & (yy < 8)
                                 & (xx >= 2) & (xx < 8)).sum())
    # analytic geometry off the mesh, in µm, anisotropic
    vox = (0.5, 0.2, 0.2)
    assert abs(MSH.enclosed_volume_um3(Vb, Fb, vox) - (4 * 0.5) * (4 * 0.2) ** 2) < 1e-9
    assert MSH.surface_area_um2(Vb, Fb, vox) > 0
    assert MSH.surface_area_um2(Vb, Fb[:0], vox) == 0.0        # no faces => 0, never a crash

    _ok("mesh domain (V2.08): 11th domain = 3 flat CSR strata on one Domain.MESH "
        "(L / L/vert / L/face); structure-not-lattice; build→with_mesh→read_mesh round-trips "
        "(dense per-frame ids + global FK + derived closed); content_hash content-stable and "
        "_canon refuses object dtype; validate() catches torn CSR/FK/ids/reserved-name; "
        "3 z_kind stamps; no bridge ⇒ clean plan-time refusal; face parity exact on a box "
        "AND a concave L-prism")


def test_tessellate_split() -> None:
    """``analysis.tessellate`` → ``transform.rasterize_mesh`` (V2.08) — the v1 split restored
    in v2 with a real MESH intermediate, replacing the fused ``analysis.tessellate_volume``.

    Reuses the fused node's exact cube fixture so the OLD numbers must still hold end-to-end
    (ids ``[1,2]``, convex-hull ``volume_um3 == 0.54 µm³``, centroids 5.5/10.5, raster
    ``{0,1,2}``, ``z_kind`` stamped, calibration fenced, chains into ``boundary_band``), and
    adds: the intermediate MESH payload exists + validates; the concave ``alpha_shape`` path
    rasterizes STRICTLY SMALLER than its convex hull (proving the faces-based interior test
    is real, and fixing the latent bug where every mode filled convex); ``label_surface``
    round-trips a Voxel label region → mesh → raster; per-mode socket gating; the ``z<2`` /
    missing-layer guards; and a two-fresh-Engine output-fingerprint match. scipy+skimage."""
    if not _HAVE_SKIMAGE:                                    # skimage ⟹ scipy present
        _ok("tessellate split: SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.structure import StructureTable as _ST
    from nodegraph.kernels.mesh_raster import interior_test_for
    from nodegraph.memo import output_fingerprint
    import nodegraph.mesh as MSH

    define_node("io.tvseed", "S", outputs=[OutDataset()])

    def cube(z0, y0, x0):                                    # 8 corners of a 3-voxel cube
        return np.array([[z0 + dz, y0 + dy, x0 + dx]
                         for dz in (0, 3) for dy in (0, 3) for dx in (0, 3)], dtype=float)
    P = np.vstack([cube(4, 10, 10), cube(9, 30, 30)])       # two well-separated clusters
    L = np.array([0] * 8 + [1] * 8, dtype=np.int64)
    n = len(P)
    tbl = _ST(D.POINT, {
        "id": np.arange(n, dtype=np.int64), "m": np.zeros(n, np.int64),
        "t": np.zeros(n, np.int64), "c": np.zeros(n, np.int64),
        "z": P[:, 0], "y": P[:, 1], "x": P[:, 2], "cluster": L},
        layer="particles", z_kind="subpixel")
    ax = AxisSizes(m=1, t=1, z=16, c=1, y=48, x=48)
    meta = {"pixel_size_um": 0.2, "z_step_um": 0.5}          # cube side = 3 vox
    blank = np.zeros((1, 1, 16, 1, 48, 48))
    ds = (Dataset(axes=ax, metadata=meta)
          .with_image(ArrayProvider(blank)).with_structure(tbl))
    env = MetaEnvelope(axes=ax, metadata=meta)

    def run(nodes, edges, sink, seed=None, seedenv=None):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for e in edges:
            g.connect(e[0], e[1], dst_socket=(e[2] if len(e) > 2 else "data"))
        eng = Engine(g, computes=COMPUTES, seeds={"S": seed if seed is not None else ds},
                     meta_seeds={"S": seedenv if seedenv is not None else env})
        return eng, eng.pull(sink)

    TESS = ("T", "analysis.tessellate", {})
    RAST = ("R", "transform.rasterize_mesh", {})
    CHAIN = [("S", "io.tvseed", {}), TESS, RAST]
    WIRE = [("S", "T"), ("T", "R")]

    # ── the MESH intermediate is a real, validated payload on its own wire ────────
    et, om = run([("S", "io.tvseed", {}), TESS], [("S", "T")], "T")
    mt = MSH.read_mesh(om, "mesh").validate()
    assert mt.element.n == 2 and mt.vertex.n == 16 and mt.face.n > 0
    assert np.asarray(mt.element.columns["id"]).tolist() == [1, 2]
    assert np.asarray(mt.element.columns["src_label"]).tolist() == [0, 1]
    assert om.structure_zkind(D.MESH, "mesh") == "subpixel"
    assert MSH.mesh_provenance(om, "mesh")["boundary"] == "convex_hull"
    assert {"pixel_size_um", "z_step_um"} <= {k for k, _ in et.entry("T").reads}
    assert D.MESH in NODES.get("analysis.tessellate").adds_domains
    # vertices are VOXEL coords (the domain convention) — i.e. the input cloud itself
    vz = np.asarray(mt.vertex.columns["z"], dtype=float)
    assert vz.min() >= 4.0 - 1e-9 and vz.max() <= 12.0 + 1e-9, (vz.min(), vz.max())

    # ── the rasterized half reproduces the FUSED node's numbers exactly ───────────
    e, o = run(CHAIN, WIRE, "R")
    rast = o.get(D.VOXEL, "labels").values
    assert rast.shape == (1, 1, 16, 1, 48, 48)
    assert set(np.unique(rast).tolist()) == {0, 1, 2}        # bg + 2 regions
    names = {a.name for a in o.layers_on(D.LABEL) if a.layer == "labels"}
    assert {"id", "m", "t", "c", "z", "y", "x", "volume_um3", "density", "n_points",
            "surface_area_um2", "voxel_count"} <= names, sorted(names)
    ids = o.get(D.LABEL, "id", layer="labels").values.tolist()
    vol = o.get(D.LABEL, "volume_um3", layer="labels").values
    zc = sorted(o.get(D.LABEL, "z", layer="labels").values.tolist())
    assert ids == [1, 2]
    # convex hull of a 3-vox cube: (3·0.5)·(3·0.2)·(3·0.2) = 0.54 µm³
    assert np.allclose(vol, 0.54, atol=1e-6), vol.tolist()
    assert abs(zc[0] - 5.5) < 0.1 and abs(zc[1] - 10.5) < 0.1, zc      # cube centers
    assert np.all(o.get(D.LABEL, "voxel_count", layer="labels").values > 0)
    assert o.structure_zkind(D.LABEL, "labels") == "subpixel"          # §7b provenance
    assert {"pixel_size_um", "z_step_um"} <= {k for k, _ in e.entry("R").reads}

    # ── chain into the GENERAL boundary_band (reads the "labels" raster) ──────────
    _, ob = run(CHAIN + [("B", "analysis.boundary_band", {"params": {"band_voxels": 1}})],
                WIRE + [("R", "B")], "B")
    bands = ob.get(D.VOXEL, "bands").values
    assert int((bands != 0).sum()) > 0 and int(((bands != 0) & (rast != 0)).sum()) == 0

    # ── the concave path is REAL: alpha_shape fills less than its convex hull ─────
    # one label over two separated cubes — the convex hull spans the gap, a finite alpha
    # drops the long bridging tetrahedra. This is the bug the fused node hid: it stored the
    # UNFILTERED Delaunay, whose simplices union to the convex hull in EVERY mode.
    P2 = np.vstack([cube(4, 8, 8), cube(4, 8, 20)])
    n2 = len(P2)
    tbl2 = _ST(D.POINT, {
        "id": np.arange(n2, dtype=np.int64), "m": np.zeros(n2, np.int64),
        "t": np.zeros(n2, np.int64), "c": np.zeros(n2, np.int64),
        "z": P2[:, 0], "y": P2[:, 1], "x": P2[:, 2],
        "cluster": np.zeros(n2, np.int64)}, layer="particles", z_kind="subpixel")
    ds2 = (Dataset(axes=ax, metadata=meta)
           .with_image(ArrayProvider(blank)).with_structure(tbl2))
    _, oc = run(CHAIN, WIRE, "R", seed=ds2)
    _, oa = run([("S", "io.tvseed", {}),
                 ("T", "analysis.tessellate", {"modes": {"boundary": "alpha_shape"},
                                               "params": {"alpha_um": 1.2}}), RAST],
                WIRE, "R", seed=ds2)
    n_convex = int((oc.get(D.VOXEL, "labels").values != 0).sum())
    n_alpha = int((oa.get(D.VOXEL, "labels").values != 0).sum())
    assert n_alpha > 0, "the alpha-shape mesh dissolved — alpha_um is too small"
    assert n_alpha < n_convex, (n_alpha, n_convex)
    assert MSH.mesh_provenance(oa, "mesh")["boundary"] == "alpha_shape"
    # and the interior test really is DERIVED from that provenance, with no user lever
    assert interior_test_for({"boundary": "convex_hull"}, True) == "convex"
    assert interior_test_for({"boundary": "alpha_shape"}, True) == "watertight"
    assert interior_test_for({"boundary": "alpha_shape"}, False) == "convex"  # open => fall back
    assert interior_test_for({}, True) == "watertight"                       # hand-built mesh
    assert not NODES.get("transform.rasterize_mesh").modes

    # ── label_surface: a Voxel label region → mesh → raster round-trip ────────────
    lab = np.zeros((1, 1, 16, 1, 48, 48), dtype=np.int64)
    lab[0, 0, 4:12, 0, 10:20, 10:20] = 1
    dsl = (Dataset(axes=ax, metadata=meta)
           .with_image(ArrayProvider(blank)).with_layer(D.VOXEL, "labels", lab))
    _, ol = run([("S", "io.tvseed", {}),
                 ("T", "analysis.tessellate", {"modes": {"boundary": "label_surface"},
                                               "params": {"name": "surf"}}),
                 ("R", "transform.rasterize_mesh", {"params": {"mesh": "surf",
                                                               "name": "again"}})],
                WIRE, "R", seed=dsl)
    back = ol.get(D.VOXEL, "again").values != 0
    orig = lab != 0
    iou = float((back & orig).sum()) / float((back | orig).sum())
    assert iou > 0.95, f"label_surface round-trip IoU {iou:.3f}"
    assert MSH.read_mesh(ol, "surf").face.n > 0

    # ── determinism across two fresh Engines ─────────────────────────────────────
    # The mesh CONTENT hash must match: this is the assertion that fails loudly the day
    # someone reintroduces a ragged/object-dtype mesh column (identical content would then
    # hash differently, because ndarray canonicalization would be hashing pointers).
    # ``output_fingerprint`` deliberately can NOT be compared here — its Dataset branch
    # folds each layer's fresh monotonic ``revision``, which is per-session lookup identity
    # by design, not a content hash.
    _, oa2 = run([("S", "io.tvseed", {}), TESS], [("S", "T")], "T")
    _, ob2 = run([("S", "io.tvseed", {}), TESS], [("S", "T")], "T")
    assert MSH.read_mesh(oa2, "mesh").content_hash() == MSH.read_mesh(ob2, "mesh").content_hash()
    assert output_fingerprint(oa2) and output_fingerprint(ob2)
    # the boundary lever must re-key. Compared WITHIN one Engine on purpose: recipe_hash
    # folds monotonic layer revisions, so it is a per-session lookup key and is not stable
    # across two independent Engines (true of every node, e.g. rr.reroute — not a mesh
    # property). The content hash above is what carries cross-run determinism.
    em, _ = run([("S", "io.tvseed", {}), TESS,
                 ("V", "analysis.tessellate", {"modes": {"boundary": "voronoi"}})],
                [("S", "T"), ("S", "V")], "T")
    em.pull("V")
    assert em.entry("T").recipe_hash != em.entry("V").recipe_hash, "boundary must re-key"

    # ── guards + per-mode socket gating ───────────────────────────────────────────
    ax1 = AxisSizes(m=1, t=1, z=1, c=1, y=48, x=48)
    ds1 = (Dataset(axes=ax1, metadata=meta)
           .with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 48, 48)))).with_structure(tbl))
    try:
        run([("S", "io.tvseed", {}), TESS], [("S", "T")], "T", seed=ds1,
            seedenv=MetaEnvelope(axes=ax1, metadata=meta))
        raise SystemExit("z<2 should raise")
    except ValueError as ex:
        assert "z>1" in str(ex)
    try:
        run([("S", "io.tvseed", {}),
             ("T", "analysis.tessellate", {"params": {"cluster": "nope"}})],
            [("S", "T")], "T")
        raise SystemExit("missing label column should raise")
    except ValueError as ex2:
        assert "label column" in str(ex2)
    try:
        run([("S", "io.tvseed", {}), ("R", "transform.rasterize_mesh", {})],
            [("S", "R")], "R")
        raise SystemExit("a missing mesh should raise")
    except ValueError as ex3:
        assert "no mesh" in str(ex3)

    # each boundary choice must show ONLY the sockets it reads (§5b)
    _avis = lambda st: {i.name for i in NODES.get("analysis.tessellate").active_inputs(st)}
    assert "alpha_um" in _avis({"boundary": "alpha_shape"})
    assert "alpha_um" not in _avis({"boundary": "convex_hull"})
    assert "alpha_um" not in _avis({"boundary": "voronoi"})
    for pt_mode in ("convex_hull", "alpha_shape", "voronoi"):
        vis = _avis({"boundary": pt_mode})
        assert {"points", "cluster", "min_points", "name"} <= vis, pt_mode
        assert not ({"labels", "iso_level", "decimate"} & vis), pt_mode
    ls = _avis({"boundary": "label_surface"})
    assert {"labels", "iso_level", "decimate", "name"} <= ls
    assert not ({"points", "cluster", "alpha_um", "min_points"} & ls)
    # the voxelization params moved to the rasterizer, and nothing gates them
    rm = {i.name for i in NODES.get("transform.rasterize_mesh").active_inputs({})}
    assert {"mesh", "name", "smooth_um", "min_voxels", "fill_holes"} <= rm

    _ok("tessellate split (V2.08): analysis.tessellate → MESH → transform.rasterize_mesh "
        "reproduces the fused node's numbers (ids [1,2], volume_um3=cube 0.54, centroids "
        "5.5/10.5, z_kind, calib fenced, chains into boundary_band) + a validated MESH "
        "intermediate with voxel-space verts; concave alpha_shape now fills LESS than its "
        "convex hull (interior test derived from provenance, no lever); label_surface "
        "round-trips a label region at IoU>0.95; mesh content_hash deterministic across "
        "Engines + boundary re-keys; z<2 / missing-layer guards; per-mode socket gating for "
        "all 4 boundary choices")


def test_engine_observer() -> None:
    """Per-node run observation (2026-07-28): the engine reports start/done/cached/error
    per node and forwards ``ctx.progress`` fractions, without touching results. This is
    what makes a per-node progress bar possible in the GUI."""
    define_node("eng.obs_src", "ObsSrc", outputs=[OutDataset()])
    define_node("eng.obs_work", "ObsWork", inputs=[InDataset()], outputs=[OutDataset()])
    define_node("eng.obs_boom", "ObsBoom", inputs=[InDataset()], outputs=[OutDataset()])

    def c_src(ctx):
        return np.array([1.0])

    def c_work(ctx):
        for i in range(4):                          # an eager per-unit compute
            ctx.progress(i + 1, 4, "unit")
        return np.asarray(ctx.inputs[0]) * 2.0

    def c_boom(ctx):
        raise ValueError("kaboom")

    computes = {"eng.obs_src": c_src, "eng.obs_work": c_work, "eng.obs_boom": c_boom}
    g = Graph()
    g.add(NodeInstance("S", "eng.obs_src"))
    g.add(NodeInstance("W", "eng.obs_work"))
    g.connect("S", "W")

    seen: list = []
    eng = Engine(g, computes=computes,
                 observer=lambda ev, nid, info: seen.append((ev, nid, info)))
    assert np.allclose(eng.pull("W"), 2.0)          # observation never changes results
    kinds = [(ev, nid) for ev, nid, _i in seen]
    assert kinds[0] == ("start", "S") and ("done", "S") in kinds
    assert ("start", "W") in kinds and ("done", "W") in kinds
    # start precedes its own done, and upstream finishes before downstream starts
    assert kinds.index(("done", "S")) < kinds.index(("start", "W"))
    fracs = [i["fraction"] for ev, nid, i in seen if ev == "progress" and nid == "W"]
    assert fracs == [0.25, 0.5, 0.75, 1.0], fracs
    assert all(i["total"] == 4 for ev, _n, i in seen if ev == "progress")
    secs = [i["seconds"] for ev, _n, i in seen if ev == "done"]
    assert len(secs) == 2 and all(s >= 0.0 for s in secs)

    # a second pull is all memo hits → 'cached' per node, no 'start'
    seen.clear()
    eng.pull("W")
    assert [(ev, nid) for ev, nid, _i in seen] == [("cached", "S"), ("cached", "W")]

    # a raising compute is observed as 'error' on the node that raised, and the
    # exception still propagates verbatim
    g2 = Graph()
    g2.add(NodeInstance("S", "eng.obs_src"))
    g2.add(NodeInstance("B", "eng.obs_boom"))
    g2.connect("S", "B")
    seen2: list = []
    eng2 = Engine(g2, computes=computes,
                  observer=lambda ev, nid, info: seen2.append((ev, nid, info)))
    try:
        eng2.pull("B")
        raise SystemExit("the raising compute must propagate")
    except ValueError as exc:
        assert "kaboom" in str(exc)
    errs = [(nid, i.get("error")) for ev, nid, i in seen2 if ev == "error"]
    assert len(errs) == 1 and errs[0][0] == "B" and "kaboom" in errs[0][1]

    # an observer that itself raises must not break the run (a broken progress sink is
    # a UI bug, never a data bug) — and ctx.progress is a no-op without an observer
    def bad(_ev, _nid, _info):
        raise RuntimeError("bad sink")

    g3 = Graph()
    g3.add(NodeInstance("S", "eng.obs_src"))
    g3.add(NodeInstance("W", "eng.obs_work"))
    g3.connect("S", "W")
    assert np.allclose(Engine(g3, computes=computes, observer=bad).pull("W"), 2.0)
    assert np.allclose(Engine(g3, computes=computes).pull("W"), 2.0)

    _ok("engine observer: per-node start/done(+seconds)/cached/error + ctx.progress "
        "fractions; results untouched, a raising observer or compute both handled")



# ── V2.12: the central Segmentation node (one contract, the algorithm as a Mode) ──

def _stub_cellsam(planes, *, nocells="", log=None):
    """A fake ``cellSAM`` package, injected into ``sys.modules`` so the kernel's GLUE is
    covered in the fast gate — the real model is a multi-hundred-MB download behind a
    DeepCell API token, so it can never run here, and the glue is where the integration
    risk lives (the mis-shaped no-cells return, the model singleton, kwarg forwarding, the
    contiguous relabel). ``find_spec`` resolves through ``sys.modules``, which is why the
    stub carries a ``__spec__``; ``planes`` records every call so the test can prove the
    checkpoint is read once per pull rather than once per plane."""
    import sys
    import types
    from importlib.machinery import ModuleSpec
    mod = types.ModuleType("cellSAM")
    mod.__spec__ = ModuleSpec("cellSAM", None)

    class _Net:                      # torch is never imported: no .parameters() needed
        def eval(self):
            return self

        def to(self, dev):
            return self

    def get_model(model="cellsam_general", version=None):
        planes.append(("load", model))
        return _Net()

    def get_local_model(path):
        planes.append(("local", str(path)))
        return _Net()

    def segment_cellular_image(img, model, **kw):
        h, w = np.shape(img)
        planes.append(("seg", float(kw["bbox_threshold"]), bool(kw["normalize"])))
        if nocells == "attr":
            # The REAL no-cells failure (upstream issue #98): `CellSAM.predict` returns the
            # 4-tuple (None,)*4, so `if preds is None` never fires and upstream calls
            # `fill_holes_and_remove_small_masks(None)`. This is what a blank plane, an
            # empty FOV or the dark end slices of a stack actually hit.
            raise AttributeError("'NoneType' object has no attribute 'ndim'")
        if nocells == "shape":
            # Upstream's own (unreachable) empty branch: `np.zeros(img.shape[1:])` on the
            # (1,3,H,W) TENSOR. Covered so an upstream fix for #98 lands safely.
            return np.zeros((3, h, w), dtype=np.int32), None, None
        lab = np.zeros((h, w), dtype=np.int32)
        lab[1:5, 1:5] = 9                        # ids deliberately non-contiguous
        lab[2, 2] = 0                            # an interior hole
        lab[7:9, 7:9] = 4                        # a 4-px object (for the size filter)
        return lab, None, None

    mod.get_model, mod.get_local_model = get_model, get_local_model
    mod.segment_cellular_image = segment_cellular_image
    if log is not None:
        wsi = types.ModuleType("cellSAM.wsi")
        wsi.__spec__ = ModuleSpec("cellSAM.wsi", None)

        def segment_wsi(image, block, overlap, iou_depth, iou_threshold, **kw):
            log.append((int(block), int(overlap), int(iou_depth), float(iou_threshold)))
            return segment_cellular_image(image, kw.get("model"),
                                          **{k: v for k, v in kw.items() if k != "model"})[0]
        wsi.segment_wsi = segment_wsi
        mod.wsi = wsi
        sys.modules["cellSAM.wsi"] = wsi
    sys.modules["cellSAM"] = mod
    return mod


def test_segment() -> None:
    """``analysis.segment`` — THE segmentation node (V2.12). Every segmentation has the
    same data contract (image in; a Voxel label raster + a Label table out), so the
    algorithm is a ``method`` Mode instead of a node type: ``analysis.watershed`` and
    ``detect.stardist_nuclei`` were folded in and deleted, and CellSAM joined.

    Covers the structural spec (per-method socket gating, the new Mode-level gating, the
    per-dim footprint), a real end-to-end pull of both dependency-free methods in 2D and
    3D, the property that actually distinguishes them (watershed splits a touching pair
    that connected components merges), every shared stage (hole filling, the µm²/µm³ size
    filter, globally-unique ids, the region table), the memo fences, every hard refusal,
    and the CellSAM glue against a stubbed package."""
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    # ── structural spec (no image deps needed) ─────────────────────────────────
    s = NODES.get("analysis.segment")
    assert s.category == "analysis" and s.label == "Segmentation"
    assert s.reads_domains == frozenset({D.VOXEL})
    assert s.adds_domains == frozenset({D.VOXEL, D.LABEL})   # a raster AND a table
    assert s.meta_transform is None                          # never changes axes/calib
    assert s.resolve_granularity({"dim": "2D"}) is Granularity.WHOLE_PLANE
    assert s.resolve_granularity({"dim": "3D"}) is Granularity.WHOLE_VOLUME
    assert s.resolve_kernel_axes({"dim": "3D"}) == frozenset({"z", "y", "x"})
    assert all(not i.is_field for i in s.inputs if i.type is not SocketType.DATASET), \
        "no path in this node evaluates a wired Field — field=True would be a lie"
    mode_of = {m.name: m for m in s.modes}
    assert set(mode_of) == {"dim", "method", "level"}
    assert set(mode_of["method"].choices) == {"threshold", "watershed", "stardist",
                                              "cellsam"}
    assert mode_of["method"].resolved_default() == "threshold", \
        "the dependency-free method must be the default — a fresh node has to just run"
    assert mode_of["level"].resolved_default() == "otsu"
    # method-gated sockets: each method sees ONLY its own controls
    vis = lambda st: {i.name for i in s.active_inputs(st)}
    st2 = {"dim": "2D", "method": "threshold", "level": "otsu"}
    assert "connectivity" in vis(st2) and "mask" not in vis(st2)
    assert "mask" in vis({**st2, "method": "watershed"})
    assert "min_distance" in vis({**st2, "method": "watershed"})
    assert "connectivity" not in vis({**st2, "method": "watershed"}), \
        "watershed takes its regions from the markers, never from a connectivity"
    assert "prob_thresh" in vis({**st2, "method": "stardist"})
    assert "bbox_threshold" in vis({**st2, "method": "cellsam"})
    assert not (vis({**st2, "method": "stardist"}) & {"bbox_threshold", "cellsam_model"})
    assert not (vis({**st2, "method": "cellsam"}) & {"prob_thresh", "model_name"}), \
        "socket names must stay DISJOINT across methods — the card relayouts on the name " \
        "list, so a same-named socket with another default would not redraw"
    # the fixed level only exists under level=fixed, and only for the cutting methods
    assert "threshold" in vis({**st2, "level": "fixed"})
    assert "threshold" not in vis(st2)
    assert "threshold" not in vis({**st2, "method": "cellsam", "level": "fixed"})
    # a 2D object has an area, a 3D object a volume — never one socket meaning both
    assert {"min_area", "max_area"} <= vis(st2) and "min_volume" not in vis(st2)
    assert "min_volume" in vis({**st2, "dim": "3D"}) and "max_area" not in \
        vis({**st2, "dim": "3D"})
    # ── V2.12 Mode gating: `level` is meaningless to a learned detector, so the whole
    #    dropdown is hidden rather than shown and ignored (ModeSpec.available_in)
    amodes = lambda st: {m.name for m in s.active_modes(st)}
    assert amodes(st2) == {"dim", "method", "level"}
    assert amodes({**st2, "method": "watershed"}) == {"dim", "method", "level"}
    assert amodes({**st2, "method": "cellsam"}) == {"dim", "method"}
    assert amodes({**st2, "method": "stardist"}) == {"dim", "method"}
    # ...and a hidden Mode still carries its value, so the compute and the memo are
    # unaffected by what the GUI draws (the same rule as a hidden socket)
    assert s.default_state()["level"] == "otsu"

    if not _HAVE_WATERSHED:
        _ok("segmentation: spec OK; RUN SKIPPED (scipy/skimage absent)")
        return

    # ── fixture: two separated disks, one TOUCHING pair, and a hole ────────────
    def _disk(a, cy, cx, r, val):
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        a[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = val

    Y = X = 32
    plane = np.zeros((Y, X), dtype=float)
    _disk(plane, 7, 7, 4, 100.0)                     # A — 44 px after its hole
    _disk(plane, 7, 24, 4, 100.0)                    # B — 49 px
    _disk(plane, 22, 10, 5, 100.0)                   # C \ overlapping: ONE component
    _disk(plane, 22, 18, 5, 100.0)                   # D / that only a watershed splits
    _hole = np.zeros((Y, X), dtype=bool)
    _disk(_hole, 7, 7, 1, True)
    plane[_hole] = 0.0                               # 5-px hole in the middle of A
    ax = AxisSizes(m=1, t=1, z=3, c=1, y=Y, x=X)
    img = np.zeros((1, 1, 3, 1, Y, X), dtype=float)
    img[0, 0, :, 0] = plane                          # identical on all 3 planes
    optics = {"pixel_size_um": 0.5, "z_step_um": 1.0, "bit_depth": 12}
    seedenv = MetaEnvelope(axes=ax, metadata=optics)
    ds = Dataset(axes=ax, metadata=dict(optics)).with_image(ArrayProvider(img))
    define_node("io.segseed", "Seed", outputs=[OutDataset()])

    def eng(*, modes=None, params=None, chain=(), dset=None, denv=None):
        g = Graph(); g.add(NodeInstance("S", "io.segseed")); prev = "S"
        for i, (cop, cmodes, cparams) in enumerate(chain):
            nid = f"U{i}"
            g.add(NodeInstance(nid, cop, modes=cmodes or {}, params=cparams or {}))
            g.connect(prev, nid); prev = nid
        g.add(NodeInstance("N", "analysis.segment", modes=modes or {},
                           params=params or {}))
        g.connect(prev, "N")
        return Engine(g, computes=COMPUTES, seeds={"S": dset if dset is not None else ds},
                      meta_seeds={"S": denv if denv is not None else seedenv})

    def seg(**kw):
        """(engine, raster, {column: values}) for one pull, output layer 'labels'."""
        e = eng(**kw)
        out = e.pull("N")
        raster = out.get(D.VOXEL, "labels")
        assert raster is not None, "no Voxel label raster"
        cols = {c: (out.get(D.LABEL, c, layer="labels").values
                    if out.get(D.LABEL, c, layer="labels") is not None else None)
                for c in ("id", "area", "z", "y", "x", "m", "t", "c")}
        return e, out, np.asarray(raster.values), cols

    # ── the two dependency-free methods, 2D and 3D ────────────────────────────
    e2, out2, r2, c2 = seg(modes={"dim": "2D", "method": "threshold"})
    assert int(r2.max()) == 9, "3 planes x 3 connected regions (the pair is ONE region)"
    assert len(c2["id"]) == 9 and sorted(c2["id"].tolist()) == list(range(1, 10)), \
        "ids must be globally unique and contiguous across every unit"
    assert sorted(np.unique(r2[r2 > 0]).tolist()) == c2["id"].tolist(), \
        "the raster's ids ARE the table's ids"
    assert sorted(c2["area"].tolist()) == [44, 44, 44, 49, 49, 49, 153, 153, 153], \
        "area is a voxel count, per plane"
    assert sorted(set(c2["z"].tolist())) == [0.0, 1.0, 2.0], "2D z = the plane index"
    assert out2.structure_zkind(D.LABEL, "labels") == "plane_index"
    # the viewer's Labels overlay discovers a raster by DTYPE + rank, not by name
    assert np.issubdtype(r2.dtype, np.integer) and r2.ndim == 6, \
        "the overlay only draws an integer 6-D Voxel layer"

    e3, out3, r3, c3 = seg(modes={"dim": "3D", "method": "threshold"})
    assert int(r3.max()) == 3, "3D labels the VOLUME: the 3 regions are z-connected"
    assert sorted(c3["area"].tolist()) == [132, 147, 459], "3x the per-plane areas"
    assert out3.structure_zkind(D.LABEL, "labels") == "subpixel"
    assert all(0.0 <= z <= 2.0 for z in c3["z"].tolist()), "3D z = a centroid, in range"

    # THE property that distinguishes the two classical methods: the touching pair is one
    # connected component and four disks, so only the watershed recovers the fourth object.
    _, _, rw, cw = seg(modes={"dim": "2D", "method": "watershed"},
                       params={"min_distance": 1.5})
    assert int(rw.max()) == 12, "watershed splits the pair: 3 planes x 4 objects"
    assert sorted(cw["area"].tolist())[:4] == [44, 44, 44, 49]
    _, _, rw3, cw3 = seg(modes={"dim": "3D", "method": "watershed"},
                         params={"min_distance": 1.5})
    assert int(rw3.max()) == 4, "volumetric watershed: 4 z-connected objects"
    # REGRESSION (V2.12): the folded-in `analysis.watershed` numbered EVERY peak pixel
    # returned by peak_local_max as its own marker, so an EDT plateau — the diagonal crest
    # inside the holed disk here — shattered one object into one basin per plateau pixel
    # (this fixture: 11 per plane, some of them 1 voxel). Merging the plateau with FULL
    # connectivity is the fix; these counts are its guard.
    assert int(rw.max()) < 15 and int(cw["area"].min()) > 5, \
        "EDT plateaus must merge into ONE marker, not one marker per plateau pixel"
    # REGRESSION (V2.12): `peak_local_max` excludes a border shell `min_distance` wide on
    # EVERY axis by default — and `min_distance` is a parameter this call does not even use
    # (the physical suppression is the per-axis footprint). With the default left on, a
    # 3D volume of z<=2 has NO interior z plane, so ZERO peaks come back, the single-marker
    # fallback fires, and every object in the volume collapses into one. Two disjoint disks
    # over a 2-plane stack returned 1 object instead of 2. An object against the frame edge
    # was dropped for the same reason.
    thin = np.zeros((1, 1, 2, 1, Y, X), dtype=float)
    thin[0, 0, :, 0] = plane
    thin[0, 0, :, 0, 0:4, 0:4] = 100.0                # a 5th object in the frame CORNER
    thin_ax = AxisSizes(m=1, t=1, z=2, c=1, y=Y, x=X)
    thin_ds = Dataset(axes=thin_ax, metadata=dict(optics)).with_image(ArrayProvider(thin))
    thin_env = MetaEnvelope(axes=thin_ax, metadata=optics)
    _, _, rthin, _ = seg(modes={"dim": "3D", "method": "watershed"},
                         params={"min_distance": 1.5}, dset=thin_ds, denv=thin_env)
    assert int(rthin.max()) == 5, \
        "a z<=2 volume must still yield every object (border exclusion off), got %d" \
        % int(rthin.max())
    _, _, rthin2, _ = seg(modes={"dim": "2D", "method": "watershed"},
                          params={"min_distance": 1.5}, dset=thin_ds, denv=thin_env)
    assert int(rthin2.max()) == 10, "2 planes x 5 objects incl. the frame-corner one"

    # ── the shared stages ─────────────────────────────────────────────────────
    # hole filling closes A's 5-px hole, so A ends up the same size as the intact disk B
    _, _, _, cf = seg(modes={"dim": "2D", "method": "threshold"},
                      params={"fill_holes": True})
    assert sorted(cf["area"].tolist()) == [49] * 6 + [153] * 3, \
        "fill_holes must close the interior hole (44 -> 49) and touch nothing else"
    # the size filter is PHYSICAL: 20 µm² / (0.5 µm)² = 80 px², which keeps only the pair
    _, _, rmin, cmin = seg(modes={"dim": "2D", "method": "threshold"},
                           params={"min_area": 20.0})
    assert cmin["area"].tolist() == [153] * 3, "min_area (µm²) -> px² drops the disks"
    _, _, _, cmax = seg(modes={"dim": "2D", "method": "threshold"},
                        params={"max_area": 20.0})
    assert sorted(cmax["area"].tolist()) == [44, 44, 44, 49, 49, 49], "max_area keeps them"
    # ...and in 3D it is a VOLUME: 40 µm³ / (0.5·0.5·1.0) = 160 voxels
    _, _, _, cvol = seg(modes={"dim": "3D", "method": "threshold"},
                        params={"min_volume": 40.0})
    assert cvol["area"].tolist() == [459], "min_volume (µm³) -> voxels"

    # ── the foreground cut ────────────────────────────────────────────────────
    for lvl in ("otsu", "li", "yen", "triangle", "mean"):
        _, _, rl, _ = seg(modes={"dim": "2D", "method": "threshold", "level": lvl})
        assert int(rl.max()) >= 3, f"level {lvl} found nothing"
    _, _, rfx, _ = seg(modes={"dim": "2D", "method": "threshold", "level": "fixed"},
                       params={"threshold": 50.0})
    assert int(rfx.max()) == 9, "a fixed level in the image's own units"
    # a flat unit must NOT become one giant object (skimage's methods return the constant)
    flat = Dataset(axes=ax, metadata=dict(optics)).with_image(
        ArrayProvider(np.full((1, 1, 3, 1, Y, X), 7.0)))
    _, _, rflat, _ = seg(modes={"dim": "2D", "method": "threshold"}, dset=flat)
    assert int(rflat.max()) == 0, "a blank plane segments to nothing, not to one big blob"
    # ...and a single NON-FINITE voxel must not take the histogram down with it. Untreated,
    # every skimage method raises "autodetected range ... is not finite", and NaN also
    # defeats the flat guard above (nan == nan is False). Deconvolution / normalize /
    # resample can all put one there.
    for tag, bad in (("nan", np.nan), ("inf", np.inf)):
        dirty = img.copy(); dirty[0, 0, 0, 0, 0, 0] = bad
        dds2 = Dataset(axes=ax, metadata=dict(optics)).with_image(ArrayProvider(dirty))
        _, _, rnf, cnf = seg(modes={"dim": "2D", "method": "threshold"}, dset=dds2)
        assert int(rnf.max()) == 9, f"one {tag} voxel must not change the segmentation"
        assert sorted(cnf["area"].tolist())[:3] == [44, 44, 44], \
            f"one {tag} voxel must not join or split an object"
    allnan = Dataset(axes=ax, metadata=dict(optics)).with_image(
        ArrayProvider(np.full((1, 1, 3, 1, Y, X), np.nan)))
    _, _, rnan, _ = seg(modes={"dim": "2D", "method": "threshold"}, dset=allnan)
    assert int(rnan.max()) == 0, "an all-NaN plane segments to nothing"

    # connectivity: two corner-touching squares are 2 objects at 4-conn, 1 at 8-conn
    dax = AxisSizes(m=1, t=1, z=1, c=1, y=8, x=8)
    dimg = np.zeros((1, 1, 1, 1, 8, 8), dtype=float)
    dimg[0, 0, 0, 0, 1:3, 1:3] = 50.0
    dimg[0, 0, 0, 0, 3:5, 3:5] = 50.0                # touches the first only diagonally
    denv = MetaEnvelope(axes=dax, metadata=optics)
    dds = Dataset(axes=dax, metadata=dict(optics)).with_image(ArrayProvider(dimg))
    for conn, want in ((4, 2), (8, 1)):
        _, _, rc, _ = seg(modes={"dim": "2D", "method": "threshold"},
                          params={"connectivity": conn}, dset=dds, denv=denv)
        assert int(rc.max()) == want, f"{conn}-connectivity should give {want} region(s)"
    # 0 is what a spin box commits when it is touched; it must mean "per-dim default" (8),
    # not reach `_connectivity_rank` and raise on a value the user never chose
    _, _, rc0, _ = seg(modes={"dim": "2D", "method": "threshold"},
                       params={"connectivity": 0}, dset=dds, denv=denv)
    assert int(rc0.max()) == 1, "connectivity=0 must resolve to the 2D default (8)"

    # ── the watershed `mask` socket: split an EXISTING foreground ──────────────
    thr = ("analysis.threshold", {"method": "fixed"}, {"threshold": 50.0, "name": "m2"})
    outm = eng(modes={"dim": "2D", "method": "watershed"},
               params={"mask": "m2", "name": "ws2", "min_distance": 1.5},
               chain=(thr,)).pull("N")          # NOT seg(): the output layer is renamed
    assert outm.get(D.VOXEL, "ws2") is not None, "`mask` + `name` sockets"
    assert outm.get(D.VOXEL, "labels") is None, "no stale default layer"
    assert outm.get(D.LABEL, "area", layer="ws2") is not None, "the table follows `name`"
    assert int(np.asarray(outm.get(D.VOXEL, "ws2").values).max()) == 12, \
        "splitting the upstream mask must match splitting its own cut"

    # ── memo behaviour ────────────────────────────────────────────────────────
    reads2 = dict(e2.entry("N").reads)
    assert "pixel_size_um" in reads2, "the µm² filter must fence on the pixel size"
    assert "z_step_um" in dict(e3.entry("N").reads), "µm³ needs the z step too"
    # every (method, dim, level) must memoize distinctly. Keyed WITHOUT pulling — the
    # engine's `entry()` computes, and asking a hash question must not load a TensorFlow
    # model or trip the cellsam dependency gate. This mirrors engine._entry exactly:
    # the mode state folds into params as `__modes__`, then node_recipe_hash.
    from nodegraph.memo import node_recipe_hash
    hashes = {}
    for meth in ("threshold", "watershed", "stardist", "cellsam"):
        for dim in ("2D", "3D"):
            for lvl in ("otsu", "li"):
                st = {"dim": dim, "method": meth, "level": lvl}
                hashes[(meth, dim, lvl)] = node_recipe_hash(
                    "analysis.segment", {"__modes__": st}, ("up",), (1,))
    assert len(set(hashes.values())) == 16, \
        "method x dim x level must all fold into the recipe hash"
    # ...and prove it end to end on the two methods that can actually run here
    assert (eng(modes={"dim": "2D", "method": "threshold"}).entry("N").recipe_hash
            != eng(modes={"dim": "3D", "method": "threshold"}).entry("N").recipe_hash)
    assert (eng(modes={"dim": "2D", "method": "threshold"}).entry("N").recipe_hash
            != eng(modes={"dim": "2D", "method": "watershed"}).entry("N").recipe_hash)

    # ── refusals ──────────────────────────────────────────────────────────────
    for meth in ("stardist", "cellsam"):
        try:
            eng(modes={"dim": "3D", "method": meth}).pull("N")
            raise AssertionError(f"{meth} must refuse the 3D lever")
        except ValueError as exc:
            assert "2-D-per-plane" in str(exc) and "2D" in str(exc), \
                "the refusal has to name the fix"
    try:
        eng(modes={"dim": "2D", "method": "watershed"},
            params={"mask": "nope"}).pull("N")
        raise AssertionError("a named foreground layer that does not exist must raise")
    except ValueError as exc:
        assert "nope" in str(exc)
    for bad, key in (({"method": "bogus"}, "method"), ({"level": "bogus"}, "level")):
        try:
            eng(modes={"dim": "2D", **bad}).pull("N")
            raise AssertionError(f"an unknown {key} must raise")
        except ValueError as exc:
            assert "bogus" in str(exc)
    # 0 is the size filter's OFF sentinel, so a SUB-VOXEL upper bound would quantize to
    # "no upper limit" and keep everything instead of dropping all but specks — the exact
    # inversion analysis.histogram_threshold already refuses. (A sub-voxel LOWER bound is
    # harmless: every object has at least one voxel, so it filters nothing either way.)
    for dim, key, val in (("2D", "max_area", 0.1), ("3D", "max_volume", 0.1)):
        try:
            eng(modes={"dim": dim, "method": "threshold"}, params={key: val}).pull("N")
            raise AssertionError(f"a sub-voxel {key} must raise, not invert the filter")
        except ValueError as exc:
            assert "under one voxel" in str(exc), str(exc)
    _, _, rsub, _ = seg(modes={"dim": "2D", "method": "threshold"},
                        params={"min_area": 0.1})
    assert int(rsub.max()) == 9, "a sub-voxel min_area is a vacuous filter, not an error"
    try:
        eng(modes={"dim": "2D", "method": "threshold"},
            params={"min_area": 20.0, "max_area": 5.0}).pull("N")
        raise AssertionError("an empty size window must raise")
    except ValueError as exc:
        assert "size window is empty" in str(exc)

    # ── CellSAM: the glue, against a stubbed package ──────────────────────────
    import os
    import sys
    import nodegraph.kernels.cellsam_segment as CS

    # absent (this env): the node must raise the friendly install hint, not a bare
    # ModuleNotFoundError from three frames down.
    if not CS.cellsam_available():
        try:
            eng(modes={"dim": "2D", "method": "cellsam"}).pull("N")
            raise AssertionError("cellsam must refuse when the package is absent")
        except ImportError as exc:
            assert "pip install" in str(exc) and "DEEPCELL_ACCESS_TOKEN" in str(exc), \
                "the hint must name both the package and the model-token requirement"
    assert CS.resolve_device("cpu") == "cpu", "cpu must not even import torch"

    _saved = {k: sys.modules[k] for k in ("cellSAM", "cellSAM.wsi") if k in sys.modules}
    _dev = os.environ.get(CS.DEVICE_ENV)
    os.environ[CS.DEVICE_ENV] = "cpu"                # keep the gate torch-free
    try:
        calls = []
        _stub_cellsam(calls)
        CS._reset_model_cache()
        _, outc, rc2, cc = seg(modes={"dim": "2D", "method": "cellsam"},
                               params={"bbox_threshold": 0.25,
                                       "cellsam_model": "cellsam_extra"})
        assert [c for c in calls if c[0] == "load"] == [("load", "cellsam_extra")], \
            "the checkpoint must be read ONCE per pull, not once per plane"
        assert sum(c[0] == "seg" for c in calls) == 3, "one call per (m,t,z,c) plane"
        assert {c[1] for c in calls if c[0] == "seg"} == {0.25}, "bbox_threshold forwarded"
        assert int(rc2.max()) == 6 and sorted(cc["id"].tolist()) == list(range(1, 7)), \
            "3 planes x 2 objects, ids offset into a globally-unique range"
        assert sorted(cc["area"].tolist()) == [4, 4, 4, 15, 15, 15], \
            "upstream's non-contiguous ids (9, 4) relabel to 1..K with the areas intact"
        assert outc.metadata.get("segment_method") == "cellsam"
        assert outc.metadata.get("segment_model") == "cellsam_extra", "provenance stamp"
        # the shared postprocess applies to a learned method exactly as to a classical one
        _, _, _, ccf = seg(modes={"dim": "2D", "method": "cellsam"},
                           params={"fill_holes": True})
        assert sorted(ccf["area"].tolist()) == [4, 4, 4, 16, 16, 16], \
            "fill_holes must close the hole in the model's mask too (15 -> 16)"
        _, _, _, ccm = seg(modes={"dim": "2D", "method": "cellsam"},
                           params={"min_area": 2.0})
        assert sorted(ccm["area"].tolist()) == [15, 15, 15], \
            "2 µm² = 8 px² drops the 4-px object — the physical filter is shared"
        # a local checkpoint bypasses the download+token path, and the provenance must name
        # the checkpoint that actually ran (model_path WINS inside the kernel)
        calls.clear(); CS._reset_model_cache()
        _, outw, _, _ = seg(modes={"dim": "2D", "method": "cellsam"},
                            params={"model_path": "w.pt",
                                    "cellsam_model": "cellsam_extra"})
        assert ("local", "w.pt") in calls, "`model_path` must use get_local_model"
        assert not any(c[0] == "load" for c in calls), "…and never the published model"
        assert outw.metadata.get("segment_model") == "w.pt", \
            "provenance must name the local checkpoint, not the ignored model socket"
        # tiling routes through cellSAM.wsi with iou_depth == overlap (upstream needs <=)
        wsi_log = []
        calls.clear(); _stub_cellsam(calls, log=wsi_log); CS._reset_model_cache()
        seg(modes={"dim": "2D", "method": "cellsam"},
            params={"tile": True, "tile_size": 128, "tile_overlap": 24})
        assert wsi_log and wsi_log[0][:3] == (128, 24, 24), \
            "tile_size/overlap forwarded; iou_depth mirrors overlap"
        # Both upstream no-cells failures must land as an EMPTY segmentation, not a crash.
        # This is the path a blank plane / an empty FOV / the dark end slices of a z-stack
        # take on every real run, so it is the difference between "the stack segments" and
        # "the whole pull dies on slice 0".
        for mode, why in (("attr", "the real (None,)*4 AttributeError, upstream #98"),
                          ("shape", "upstream's own mis-shaped (C,H,W) empty branch")):
            calls.clear(); _stub_cellsam(calls, nocells=mode); CS._reset_model_cache()
            _, outn, rn, cn = seg(modes={"dim": "2D", "method": "cellsam"})
            assert int(rn.max()) == 0 and rn.shape == (1, 1, 3, 1, Y, X), why
            assert cn["id"] is None or len(cn["id"]) == 0, f"rows emitted for {why}"
        # ...but an unrelated AttributeError must still propagate — the absorb is narrow
        calls.clear(); _stub = _stub_cellsam(calls); CS._reset_model_cache()
        _stub.segment_cellular_image = lambda *a, **k: (_ for _ in ()).throw(
            AttributeError("'Foo' object has no attribute 'bar'"))
        try:
            seg(modes={"dim": "2D", "method": "cellsam"})
            raise AssertionError("an unrelated AttributeError must not be swallowed")
        except AttributeError as exc:
            assert "Foo" in str(exc)
    finally:
        for k in ("cellSAM", "cellSAM.wsi"):
            sys.modules.pop(k, None)
        sys.modules.update(_saved)
        if _dev is None:
            os.environ.pop(CS.DEVICE_ENV, None)
        else:
            os.environ[CS.DEVICE_ENV] = _dev
        CS._reset_model_cache()

    # StarDist is NOT run here, by the same policy as before the fold-in: loading the
    # pretrained TF model costs seconds and hits the network, which does not belong in the
    # fast gate. Its socket set and gating are covered structurally above, and the compute
    # is audited by test_param_socket_contract.
    _ok("segmentation: one image->Label contract, 4 methods behind a Mode (threshold/"
        "watershed 2D+3D run; stardist spec-only; cellsam glue on a stub, incl. BOTH "
        "upstream no-cells failures absorbed to an empty plane + an unrelated one still "
        "raised) — per-method socket + Mode gating, watershed splits what CCL merges "
        "(plateau markers merged, border exclusion off so z<=2 and frame-edge objects "
        "survive), "
        "shared fill/µm²+µm³ filter/global ids/table, px+z fences, 16 distinct "
        "method×dim×level hashes, 5 refusals (3D lever on a 2D-only method, missing "
        "foreground layer, unknown method/level, sub-voxel + empty size window)")



# ── the param<->socket contract (catalog-wide structural guard, 2026-07-28) ─────

#: Params a compute reads that legitimately have NO socket. Every entry needs a reason;
#: anything else is the defect this guard exists to catch.
_PARAM_NO_SOCKET_OK = {
    # read only as the fallback INSIDE `scale_xy`'s lookup — a back-compat alias for
    # graphs written before the axis-split, not a control of its own.
    ("util.resample", "scale"),
}
#: Sockets no compute reads. Empty, and it should stay that way: a declared socket the
#: kernel ignores is the "live-looking GUI control" the node charter forbids.
_SOCKET_UNREAD_OK: set = set()


def _compute_source(module, fn) -> str:
    """The source text of one compute (for the declared-once check). Cached per call
    site; returns "" when the function is not from `nodegraph.nodes`."""
    import inspect
    if getattr(fn, "__module__", "") != "nodegraph.nodes":
        return ""
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):        # pragma: no cover - defensive
        return ""


def _param_key_index(module):
    """Map every function in *module* to the param keys it reads, resolving keys passed
    THROUGH helpers. Without that resolution ``_radius_px(ctx, "radius", 0.3)`` looks like
    it reads nothing and every radius socket in the catalog reads as dead."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def is_read(call):
        f = call.func
        if not isinstance(f, ast.Attribute):
            return False
        return ((f.attr == "get" and isinstance(f.value, ast.Attribute)
                 and f.value.attr == "params")            # ctx.params.get(K)
                or f.attr == "param"                      # ctx.channel(c).param(K)
                or f.attr == "layer")                     # ctx.layer(K) — V2.11

    def callee(c):
        f = c.func
        if isinstance(f, ast.Name):
            return f.id
        return f.attr if isinstance(f, ast.Attribute) else None

    lits, fwd, calls, argidx = {}, {}, {}, {}
    for name, fn in fns.items():
        args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
        argidx[name] = args
        L, F, C = set(), set(), []

        def _key(k):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                L.add(k.value)
            elif isinstance(k, ast.Name) and k.id in args:
                F.add((k.id, ""))
            elif (isinstance(k, ast.BinOp) and isinstance(k.op, ast.Add)
                  and isinstance(k.left, ast.Name) and k.left.id in args
                  and isinstance(k.right, ast.Constant)):
                F.add((k.left.id, k.right.value))           # `name + "_z"` paired float

        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                C.append(node)
                if is_read(node) and node.args:
                    _key(node.args[0])
            # ctx.params["X"] — `detect.spots` reads its axial radii this way (guarded by
            # an `in ctx.params` test, so absent means "fall back to the lateral value").
            elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                  and node.value.attr == "params"):
                _key(node.slice)
            # `"X" in ctx.params` — the membership half of that same idiom
            elif isinstance(node, ast.Compare) and len(node.comparators) == 1:
                cmp0 = node.comparators[0]
                if (isinstance(node.ops[0], ast.In)
                        and isinstance(cmp0, ast.Attribute) and cmp0.attr == "params"):
                    _key(node.left)
        lits[name], fwd[name], calls[name] = L, F, C

    # a helper that hands its OWN arg to a forwarder is itself a forwarder — transitive,
    # e.g. _rad(name) -> _cleanup_um(name) -> ctx.channel(0).param(name)
    changed = True
    while changed:
        changed = False
        for name in fns:
            for c in calls[name]:
                cn = callee(c)
                if cn not in fwd or cn == name or not fwd[cn]:
                    continue
                for aname, suf in list(fwd[cn]):
                    i = argidx[cn].index(aname) if aname in argidx[cn] else -1
                    if 0 <= i < len(c.args):
                        a = c.args[i]
                        if (isinstance(a, ast.Name) and a.id in argidx[name]
                                and (a.id, suf) not in fwd[name]):
                            fwd[name].add((a.id, suf))
                            changed = True

    def keys(name, seen=None):
        seen = seen if seen is not None else set()
        if name in seen or name not in fns:
            return set()
        seen.add(name)
        out = set(lits[name])
        for c in calls[name]:
            cn = callee(c)
            if cn in fwd:
                for aname, suf in fwd[cn]:
                    i = argidx[cn].index(aname) if aname in argidx[cn] else -1
                    if 0 <= i < len(c.args):
                        a = c.args[i]
                        if isinstance(a, ast.Constant) and isinstance(a.value, str):
                            out.add(a.value + suf)
            out |= keys(cn, seen)
        return out

    return keys


def test_param_socket_contract() -> None:
    """Every param a compute reads has a socket, and every socket a compute reads.

    Structural, not behavioural, and that is the point: the engine does NOT filter params
    against the socket list, so a compute can happily read a param no socket exposes and
    every functional test still passes — the value is simply pinned to its fallback
    forever and no user can change it. It is invisible headlessly and only bites in the
    GUI, which builds its widgets from ``NodeSpec.inputs``. A catalog-wide sweep on
    2026-07-28 found the pattern in 18 nodes, including ``transform.transfer_domain``,
    whose four functional params were ALL unreachable (from the GUI it could only ever
    move ``mask`` voxel->frame with ``mean``). The mirror case — a declared socket no
    kernel path reads — is the "live-looking GUI control the selected kernel silently
    ignores" that the ``track.objects`` review established as charter-forbidden."""
    import nodegraph.nodes as NN

    # ── clauses 1 & 2: every param read has a socket, every socket is read ─────
    keys_of = _param_key_index(NN)
    bad_unreachable, bad_dead, audited = [], [], 0
    for spec in sorted(NODES.all(), key=lambda s: s.op_key):
        fn = NN.COMPUTES.get(spec.op_key)
        fname = getattr(fn, "__name__", None)
        # Only real catalog nodes: the index is built from `nodegraph.nodes`' AST, so a
        # node whose compute is defined elsewhere has no resolvable reads and every socket
        # would read as dead. In a full-suite run that is every `test.*` fixture (their
        # computes live in this file) — the same fixture-vs-registry overlap that makes
        # fake op_keys mandatory. Order-dependent otherwise: green alone, red in suite.
        if fname is None or getattr(fn, "__module__", "") != "nodegraph.nodes":
            continue
        audited += 1
        used = keys_of(fname)
        declared = {s.name for s in spec.inputs}
        datasets = {s.name for s in spec.inputs
                    if getattr(s.type, "name", "") == "DATASET"}
        modes = {m.name for m in spec.modes}
        for p in sorted(used):
            if (p in declared or p in modes or p.startswith("__")
                    or (spec.op_key, p) in _PARAM_NO_SOCKET_OK):
                continue
            bad_unreachable.append("%s.%s" % (spec.op_key, p))
        for d in sorted(declared - used - datasets):
            if d.startswith("__") or (spec.op_key, d) in _SOCKET_UNREAD_OK:
                continue
            bad_dead.append("%s.%s" % (spec.op_key, d))

    assert not bad_unreachable, (
        "compute reads a param with NO socket (unreachable from the GUI, pinned to its "
        "fallback): %s — add an input socket, or allowlist it in _PARAM_NO_SOCKET_OK "
        "with the reason" % bad_unreachable)
    assert not bad_dead, (
        "socket declared that no compute reads (a control that does nothing — charter-"
        "forbidden): %s — wire it, `available_in`-gate it, or delete it" % bad_dead)

    # ── clause 3: layer sockets are typed, resolvable, and domain-consistent ────
    # `layer_in`/`layer_out` are the declarative half of the layer catalog
    # (metadata.propagate_meta) and the GUI picker. Each clause below is a rule the
    # catalog must obey for BOTH to stay correct as nodes are added.
    bad_layer = []
    n_in = n_out = 0
    for spec in sorted(NODES.all(), key=lambda s: s.op_key):
        fn = NN.COMPUTES.get(spec.op_key)
        if getattr(fn, "__module__", "") != "nodegraph.nodes":
            continue
        mode_names = {m.name: set(m.choices) for m in spec.modes}
        # (a0) a MODE may be gated on another Mode's value too (V2.12 `ModeSpec.
        #      available_in`), and the same typo hides it in every state — with no socket
        #      list to notice its absence. A mode must never gate on ITSELF, which would
        #      make its own visibility depend on the value it is choosing.
        for mo in spec.modes:
            mtag = "%s![%s]" % (spec.op_key, mo.name)
            for mname, allowed in (mo.available_in or {}).items():
                if mname == mo.name:
                    bad_layer.append("%s: a Mode cannot gate on itself" % mtag)
                    continue
                if mname not in mode_names:
                    bad_layer.append("%s: available_in names mode %r that does not exist"
                                     % (mtag, mname))
                    continue
                unknown = set(allowed) - mode_names[mname]
                if unknown:
                    bad_layer.append("%s: available_in[%r] has values %s not in %s"
                                     % (mtag, mname, sorted(unknown),
                                        sorted(mode_names[mname])))
        for so in spec.inputs:
            tag = "%s.%s" % (spec.op_key, so.name)
            # (a) available_in must reference modes and VALUES that exist — a typo here
            #     silently hides the socket in every state, which is invisible until a
            #     user wonders where a control went.
            for mname, allowed in (so.available_in or {}).items():
                if mname not in mode_names:
                    bad_layer.append("%s: available_in names mode %r that does not exist"
                                     % (tag, mname))
                    continue
                unknown = set(allowed) - mode_names[mname]
                if unknown:
                    bad_layer.append("%s: available_in[%r] has values %s not in %s"
                                     % (tag, mname, sorted(unknown),
                                        sorted(mode_names[mname])))
            if not (so.layer_in or so.layer_in_mode or so.layer_out):
                continue
            if so.layer_in is not None or so.layer_in_mode:
                n_in += 1          # both reader forms: fixed domain, or mode-resolved
            # (b) a layer name is a string
            if so.type is not SocketType.STRING:
                bad_layer.append("%s: a layer socket must be STRING, is %s"
                                 % (tag, so.type))
            # (c) a mode-resolved domain must name a real mode
            if so.layer_in_mode and so.layer_in_mode not in mode_names:
                bad_layer.append("%s: layer_in_mode=%r names no such mode"
                                 % (tag, so.layer_in_mode))
            # (d) reading a layer in domain D means the node READS D. Exempt when the
            #     socket is mode-gated: the requirement is then conditional and
            #     `reads_domains` has no per-mode form (the tessellate / track.link
            #     case, whose empty declaration is deliberate and documented).
            if so.layer_in is not None:
                if so.layer_in not in spec.reads_domains and not so.available_in:
                    bad_layer.append(
                        "%s: reads a %s layer but reads_domains=%s"
                        % (tag, so.layer_in.value,
                           sorted(d.value for d in spec.reads_domains)))
            # (e) writing a layer in domain D means the node ADDS D — unconditionally,
            #     since the write happens whenever the socket is active.
            if so.layer_out:
                n_out += 1
                miss = set(so.layer_out) - set(spec.adds_domains)
                if miss:
                    bad_layer.append(
                        "%s: writes %s but adds_domains=%s"
                        % (tag, sorted(d.value for d in miss),
                           sorted(d.value for d in spec.adds_domains)))
            # (f) the default must live ONLY in the SocketSpec. A compute that still
            #     spells its own fallback inline (`ctx.params.get("mask", "mask")`) has
            #     a second copy that can drift from the declaration — and a third, in
            #     propagate_meta's prediction. `ctx.layer(name)` reads the declaration.
            src = _compute_source(NN, fn)
            if src and ('params.get("%s"' % so.name) in src:
                bad_layer.append(
                    "%s: compute reads params.get(%r) directly — use ctx.layer(%r) so "
                    "the default is declared once" % (tag, so.name, so.name))

    assert not bad_layer, "layer-socket contract violations:\n  " + "\n  ".join(bad_layer)
    # V2.12: `analysis.watershed` + `detect.stardist_nuclei` folded into
    # `analysis.segment` - the catalog lost one layer_in and two layer_out
    # sockets and gained one of each.
    assert n_in >= 18 and n_out >= 18, \
        "only %d layer_in / %d layer_out sockets — a declaration was lost" % (n_in, n_out)

    # The resolver MUST see through every indirection the catalog uses, or this guard
    # passes vacuously (nothing looks read, nothing looks declared-and-unread).
    assert "radius" in keys_of("_compute_median"), "helper-forwarded literal not resolved"
    assert "radius_z" in keys_of("_compute_median"), "suffixed `name + _z` key not resolved"
    assert "opening_radius" in keys_of("_compute_histogram_threshold"), \
        "transitively-forwarded key (_rad -> _cleanup_um) not resolved"
    assert "emission_nm" in keys_of("_compute_deconvolve"), \
        "ctx.channel(c).param(...) read not resolved"
    assert "min_radius_z" in keys_of("_compute_spots"), \
        "ctx.params[\"X\"] subscript read not resolved"

    # 54 after V2.12 folded two segmentation nodes into `analysis.segment` (was 55)
    assert audited >= 54, f"only {audited} catalog computes audited — index broke"
    _ok("socket contract: all %d catalog node types — (1) every param a compute reads "
        "has a socket, (2) every socket is read (helper-forwarded, suffixed `_z`, "
        "transitive, per-channel and ctx.layer keys resolved), (3) %d layer_in + %d "
        "layer_out sockets are STRING, domain-consistent with reads/adds_domains "
        "(mode-gated exempt), mode-resolvable, available_in-valid, and declare their "
        "default ONCE; %d documented alias exemption"
        % (audited, n_in, n_out, len(_PARAM_NO_SOCKET_OK)))



# ── V2.11: the edit-time layer catalog (what the GUI layer picker offers) ──────

def _ds_layer_names(ds, domain):
    """The layer names a REALIZED Dataset carries in *domain*.

    The projection differs by domain family and that asymmetry is the whole reason
    ``MetaEnvelope.layer_names`` exists as its own field: ``with_layer`` leaves
    ``layer=None``, so a lattice layer's user-facing name is the attribute NAME, while
    ``with_structure`` files each column under the table's LAYER."""
    from nodegraph.domains import is_lattice
    return {(a.name if is_lattice(domain) else a.layer)
            for a in ds.layers_on(domain)}


def test_layer_catalog() -> None:
    """``propagate_meta`` predicts the layer names a Dataset will carry, and the picker
    reads that prediction (V2.11).

    The prediction is a THIRD copy of every output-layer default — the socket default,
    the compute's inline fallback, and now the envelope rule — because ``propagate_meta``
    sees RAW params (the engine never default-fills them; they are overrides, which is
    why each compute repeats its default inline). Nothing structural keeps those three in
    step, so this group pins them by pulling the real nodes and comparing the prediction
    against the layers the payload actually has."""
    if not _HAVE_SKIMAGE:
        _ok("layer catalog (V2.11): SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.metadata import propagate_meta

    ax = AxisSizes(m=1, t=2, z=1, c=1, y=16, x=16)
    yy, xx = np.mgrid[0:16, 0:16]
    img = np.zeros((1, 2, 1, 1, 16, 16), dtype=float)
    for t in range(2):
        img[0, t, 0, 0] = np.sin(yy * 0.6) + np.cos(xx * 0.4) + 2.0
    img[0, :, 0, 0, 6:10, 6:10] += 30.0
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3, "dt_s": 1.0}
    ds = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(img))
    define_node("io.seedLC", "Seed", outputs=[OutDataset()])
    seedenv = MetaEnvelope(axes=ax, metadata=optics)

    # A chain using NON-DEFAULT names throughout: a default-named chain would pass even
    # if the rule ignored params entirely.
    chain = [("T", "analysis.threshold", {}, {"threshold": 20.0, "name": "m2"}),
             ("L", "analysis.label", {"dim": "2D"}, {"mask": "m2", "name": "regions"}),
             ("E", "analysis.extract_boundary", {"dim": "2D"}, {"labels": "regions"}),
             ("D", "align.drift", {}, {})]
    g = Graph(); g.add(NodeInstance("S", "io.seedLC")); prev = "S"
    for nid, op, modes, params in chain:
        g.add(NodeInstance(nid, op, modes=modes, params=params))
        g.connect(prev, nid); prev = nid
    eng = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})
    envs = propagate_meta(g, {"S": seedenv})

    # PREDICTION == REALITY, per node, per domain — the three-copies invariant.
    for nid, op, _m, _p in chain:
        out = eng.pull(nid)
        env = envs[nid]
        for dom in (D.VOXEL, D.LABEL, D.POINT, D.FRAME):
            predicted = set(env.layers_in(dom))
            actual = _ds_layer_names(out, dom)
            assert predicted <= actual, (
                f"{op}: predicted {dom.value} layers {sorted(predicted - actual)} "
                f"that the payload does not have (actual {sorted(actual)})")

    # the renamed layers really are what flows (not the hardcoded defaults)
    assert set(envs["L"].layers_in(D.VOXEL)) == {"m2", "regions"}
    assert set(envs["L"].layers_in(D.LABEL)) == {"regions"}, "one socket, TWO domains"
    # a DERIVED name (empty `name` ⇒ f"{labels}_boundary") is predicted too
    assert envs["E"].layers_in(D.POINT) == ("regions_boundary",)
    # a producer with NO socket at all still registers (extra_layers)
    assert set(envs["D"].layers_in(D.FRAME)) == {"drift_y", "drift_x"}

    # NON-MONOTONE: an axis-changing node DROPS the lattice layers whose shape it
    # invalidates (Dataset.reshaped_axes(drop_stale=True)) and keeps the rest. Without
    # this the picker would offer a mask a downstream Crop had already destroyed.
    g2 = Graph(); g2.add(NodeInstance("S", "io.seedLC"))
    g2.add(NodeInstance("T", "analysis.threshold", params={"threshold": 20.0}))
    g2.add(NodeInstance("P", "detect.spots", modes={"dim": "2D"},
                        params={"min_radius": 0.05, "max_radius": 0.4,
                                "threshold": 0.08, "name": "blobs"}))
    g2.add(NodeInstance("C", "util.crop", modes={"dim": "2D"},
                        params={"y0": 0, "y1": 8, "x0": 0, "x1": 8}))
    g2.connect("S", "T"); g2.connect("T", "P"); g2.connect("P", "C")
    e2 = propagate_meta(g2, {"S": seedenv})
    assert "mask" in e2["P"].layers_in(D.VOXEL)
    assert e2["C"].layers_in(D.VOXEL) == (), "crop must drop the y/x-shaped Voxel layers"
    assert e2["C"].layers_in(D.POINT) == ("blobs",), "structure layers survive a crop"
    # It is the AXES that decide, not the node identity: crop changes y/x but leaves m/t
    # alone, so a FRAME layer (shaped by m,t) survives the very same crop that dropped
    # the Voxel ones. Put a drift node above the crop and check both outcomes at once.
    g2b = Graph(); g2b.add(NodeInstance("S", "io.seedLC"))
    g2b.add(NodeInstance("T", "analysis.threshold", params={"threshold": 20.0}))
    g2b.add(NodeInstance("D", "align.drift"))
    g2b.add(NodeInstance("C", "util.crop", modes={"dim": "2D"},
                         params={"y0": 0, "y1": 8, "x0": 0, "x1": 8}))
    g2b.connect("S", "T"); g2b.connect("T", "D"); g2b.connect("D", "C")
    e2b = propagate_meta(g2b, {"S": seedenv})
    assert e2b["C"].layers_in(D.VOXEL) == (), "y/x-shaped Voxel layers drop"
    assert set(e2b["C"].layers_in(D.FRAME)) == {"drift_y", "drift_x"}, \
        "m/t-shaped Frame layers survive the same crop"

    # every layer_in socket names a domain the picker can resolve
    for spec in NODES.all():
        for so in spec.inputs:
            if so.layer_in is None and not so.layer_in_mode:
                continue
            assert so.type is SocketType.STRING, f"{spec.op_key}.{so.name} must be STRING"
            if so.layer_in_mode:
                mode = next((m for m in spec.modes if m.name == so.layer_in_mode), None)
                assert mode is not None, \
                    f"{spec.op_key}.{so.name} names mode {so.layer_in_mode!r} that does not exist"
    # and every layer_out socket is a STRING naming real domains
    n_out = 0
    for spec in NODES.all():
        for so in spec.inputs:
            if not so.layer_out:
                continue
            n_out += 1
            assert so.type is SocketType.STRING, f"{spec.op_key}.{so.name} must be STRING"
            assert all(isinstance(d, Domain) for d in so.layer_out)
    assert n_out >= 18, f"only {n_out} layer_out sockets — a producer lost its declaration"

    # TOTALITY: propagate_meta runs on every keystroke and its GUI caller catches only
    # ValueError, so a junk param must degrade, never raise.
    g3 = Graph(); g3.add(NodeInstance("S", "io.seedLC"))
    g3.add(NodeInstance("X", "analysis.threshold", params={"name": None}))
    g3.add(NodeInstance("Y", "analysis.label", params={"name": 17, "mask": ["not", "a", "str"]}))
    g3.connect("S", "X"); g3.connect("X", "Y")
    e3 = propagate_meta(g3, {"S": seedenv})          # must not raise
    assert e3["X"].layers_in(D.VOXEL) == ("mask",), "None falls back to the socket default"
    assert "regions" not in e3["Y"].layers_in(D.VOXEL), "a non-string name is skipped"

    _ok("layer catalog (V2.11): propagate_meta predicts the layer names a payload really "
        "carries (checked against real pulls on a fully renamed chain); one socket → two "
        "domains; derived + socket-less producers registered; an axis change DROPS the "
        "invalidated lattice layers and keeps structure; junk params degrade, never raise")


def main() -> int:
    test_domains()
    test_reducers()
    test_partial_reducers()
    test_revision_and_immutability()
    test_dataset()
    test_plans()
    test_execution()
    test_sockets()
    test_registry()
    test_socket_dims()
    test_calibration()
    test_node_variants()
    test_metadata_pass()
    test_domain_interface()
    test_provider()
    test_memo()
    test_memo_gc()
    test_engine()
    test_engine_observer()
    test_engine_granularity()
    test_structure()
    test_bridges()
    test_field()
    test_tracks()
    test_boundary()
    test_nodes()
    test_catalog()
    test_catalog_ported()
    test_catalog_ported2()
    test_channel_split()
    test_reroute()
    test_catalog3()
    test_fusion_reducers()
    test_transfer_bridge_exec()
    test_zones()
    test_sim_perframe()
    test_groups()
    test_tracking()
    test_track_objects()
    test_serialize()
    test_engine_hardening()
    test_review_regressions()
    test_streaming()
    test_streaming_slivers()
    test_catalog_kernels()
    test_catalog_dvc()
    test_catalog_dic()
    test_channel_derive()
    test_cluster_points()
    test_mesh_domain()
    test_tessellate_split()
    test_segment()
    test_param_socket_contract()
    test_layer_catalog()
    print("\nALL NODEGRAPH SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
