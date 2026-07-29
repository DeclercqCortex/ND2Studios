"""Domain transfer — the generator + the explicit bridge registry (nodegraph v2).

The heart of the model. When a field/attribute defined on domain **A** is consumed
on domain **B**, ``plan_transfer(A, B)`` returns a :class:`TransferPlan`:

* **Between acquisition-lattice domains** the plan is *generated* — coarsening
  **reduces** over the dropped axes, refining **broadcasts** — so no per-pair rule
  is written. Lattice plans **execute for real** on numpy arrays here.
* **Detected structures** (Label/Point/Track) use **explicit bridges** (Voxel↔Label,
  Label↔Point, →Track, Track↔Timepoint, Label/Point↔Frame). These are registered
  with metadata; execution is a later phase (this phase nails the lattice + routing).
  (``Channel`` is a lattice domain since V2.01 §H, so its transfers are generated.)
* **Indirect pairs** (e.g. Voxel→Track, Label→Multipoint) are **routed** by BFS
  over the domain graph (the lattice mega-edge + registered bridges), producing a
  multi-hop plan whose per-hop default is shown.

Qt-free; numpy only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np

from nodegraph.dataset import AttributeLayer, AxisSizes
from nodegraph.domains import (
    AXIS_ORDER, Domain, LATTICE_DOMAINS, axes_of, is_lattice, is_structure,
)
from nodegraph.reducers import DEFAULT_REDUCER, reduce as _reduce


# ── plan representation ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class LatticeStep:
    """A generated hop between lattice domains."""

    kind: str                     # "reduce" | "broadcast" | "identity"
    axes: FrozenSet[str]          # axes reduced away / broadcast over
    reducer: str = DEFAULT_REDUCER

    def describe(self) -> str:
        if self.kind == "identity":
            return "identity"
        axs = "".join(a for a in AXIS_ORDER if a in self.axes)
        return (f"reduce[{axs}] ({self.reducer})" if self.kind == "reduce"
                else f"broadcast[{axs}]")


@dataclass(frozen=True)
class BridgeStep:
    """A registered explicit bridge hop (structure / channel)."""

    src: Domain
    dst: Domain
    name: str
    kind: str                     # "reduce" | "broadcast" | "map"
    reducer: str = DEFAULT_REDUCER

    def describe(self) -> str:
        return f"{self.src.value}→{self.dst.value} ({self.name})"


Step = Union[LatticeStep, BridgeStep]


@dataclass(frozen=True)
class TransferPlan:
    src: Domain
    dst: Domain
    steps: Tuple[Step, ...]
    generated: bool               # True ⇒ fully lattice-generated (executable now)

    def describe(self) -> str:
        body = " · ".join(s.describe() for s in self.steps) or "identity"
        return f"{self.src.value} → {self.dst.value}: {body}"


# ── explicit bridge registry ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Bridge:
    src: Domain
    dst: Domain
    name: str
    kind: str                     # "reduce" | "broadcast" | "map"
    reducer: str = DEFAULT_REDUCER
    note: str = ""


_BRIDGES: Dict[Tuple[Domain, Domain], Bridge] = {}


def register_bridge(src: Domain, dst: Domain, name: str, kind: str,
                    reducer: str = DEFAULT_REDUCER, note: str = "") -> Bridge:
    b = Bridge(src, dst, name, kind, reducer, note)
    _BRIDGES[(src, dst)] = b
    return b


def bridges() -> Tuple[Bridge, ...]:
    return tuple(_BRIDGES.values())


def _register_default_bridges() -> None:
    D = Domain
    # Voxel ↔ Label / Point (the geometric spine)
    register_bridge(D.VOXEL, D.LABEL, "mean-in-mask", "reduce",
                    note="group voxels by label id and reduce")
    register_bridge(D.LABEL, D.VOXEL, "paint-by-label", "broadcast",
                    note="broadcast a label's value onto its region's voxels")
    register_bridge(D.VOXEL, D.POINT, "sample-at-position", "map",
                    note="interpolate the voxel field at the point")
    register_bridge(D.POINT, D.VOXEL, "splat", "broadcast",
                    note="deposit a point value onto the nearest voxel(s)")
    # Label ↔ Point
    register_bridge(D.POINT, D.LABEL, "points-in-label", "reduce",
                    note="reduce the points inside each region")
    register_bridge(D.LABEL, D.POINT, "containing-label", "map",
                    note="the region a point falls in")
    # Label / Point → Track and back
    register_bridge(D.LABEL, D.TRACK, "gather-by-track", "reduce",
                    note="a track's member labels over t, then reduce")
    register_bridge(D.POINT, D.TRACK, "gather-by-track", "reduce",
                    note="a track's member points over t, then reduce")
    register_bridge(D.TRACK, D.LABEL, "broadcast-track", "broadcast")
    register_bridge(D.TRACK, D.POINT, "broadcast-track", "broadcast")
    # Track ↔ Timepoint (a track runs over the time axis)
    register_bridge(D.TRACK, D.TIMEPOINT, "active-tracks", "reduce",
                    note="reduce over the tracks active at each t")
    register_bridge(D.TIMEPOINT, D.TRACK, "broadcast-t", "broadcast")
    # structure ↔ Frame (reduce into / broadcast out of the containing frame)
    register_bridge(D.LABEL, D.FRAME, "reduce-in-frame", "reduce")
    register_bridge(D.POINT, D.FRAME, "reduce-in-frame", "reduce")
    register_bridge(D.FRAME, D.LABEL, "broadcast-to-labels", "broadcast")
    register_bridge(D.FRAME, D.POINT, "broadcast-to-points", "broadcast")
    # Channel is a lattice domain (V2.01 §H), so Channel↔lattice transfers are
    # GENERATED (reduce/broadcast over the ``c`` axis) — no hand-registered bridge.


_register_default_bridges()


# ── routing ──────────────────────────────────────────────────────────────────

def _neighbours(d: Domain) -> List[Domain]:
    outs = set()
    if is_lattice(d):
        outs |= (LATTICE_DOMAINS - {d})      # the lattice mega-edge (any→any)
    for (s, t) in _BRIDGES:
        if s == d:
            outs.add(t)
    return sorted(outs, key=lambda x: x.value)  # deterministic routes


def _route(src: Domain, dst: Domain) -> Optional[List[Domain]]:
    """Shortest domain path src→dst over lattice edges + bridges (BFS)."""
    if src == dst:
        return [src]
    seen = {src}
    q: deque = deque([[src]])
    while q:
        path = q.popleft()
        for nb in _neighbours(path[-1]):
            if nb in seen:
                continue
            new = path + [nb]
            if nb == dst:
                return new
            seen.add(nb)
            q.append(new)
    return None


def _lattice_steps(src: Domain, dst: Domain, reducer: str) -> List[LatticeStep]:
    a_src, a_dst = axes_of(src), axes_of(dst)
    if a_src == a_dst:
        return [LatticeStep("identity", frozenset(), reducer)]
    steps: List[LatticeStep] = []
    red = a_src - a_dst
    bro = a_dst - a_src
    if red:
        steps.append(LatticeStep("reduce", frozenset(red), reducer))
    if bro:
        steps.append(LatticeStep("broadcast", frozenset(bro), reducer))
    return steps


def plan_transfer(src: Domain, dst: Domain,
                  reducer: str = DEFAULT_REDUCER) -> TransferPlan:
    """Plan the transfer of an attribute from domain ``src`` to ``dst``."""
    if src == dst:
        return TransferPlan(src, dst, (LatticeStep("identity", frozenset(), reducer),), True)
    if is_lattice(src) and is_lattice(dst):
        return TransferPlan(src, dst, tuple(_lattice_steps(src, dst, reducer)), True)
    path = _route(src, dst)
    if path is None:
        raise ValueError(f"no transfer route from {src.value} to {dst.value}")
    steps: List[Step] = []
    for a, b in zip(path, path[1:]):
        if is_lattice(a) and is_lattice(b):
            steps.extend(_lattice_steps(a, b, reducer))
        else:
            br = _BRIDGES[(a, b)]
            steps.append(BridgeStep(a, b, br.name, br.kind, br.reducer))
    generated = all(isinstance(s, LatticeStep) for s in steps)
    return TransferPlan(src, dst, tuple(steps), generated)


# ── execution (lattice steps only, for now) ──────────────────────────────────

def _apply_lattice_step(values: np.ndarray, cur_axes: List[str],
                        step: LatticeStep, axes: AxisSizes
                        ) -> Tuple[np.ndarray, List[str]]:
    if step.kind == "identity":
        return values, cur_axes
    if step.kind == "reduce":
        pos = tuple(i for i, ax in enumerate(cur_axes) if ax in step.axes)
        out = _reduce(values, pos, step.reducer)
        return out, [ax for ax in cur_axes if ax not in step.axes]
    # broadcast: insert the new axes at their canonical positions, then broadcast
    target = [a for a in AXIS_ORDER if a in (set(cur_axes) | step.axes)]
    present = list(cur_axes)
    out = values
    for i, ax in enumerate(target):
        if ax not in present:
            out = np.expand_dims(out, i)
            present.insert(i, ax)
    shape = tuple(axes.size(a) for a in target)
    out = np.broadcast_to(out, shape)
    return out, target


def execute_transfer(layer: AttributeLayer, plan: TransferPlan,
                     axes: AxisSizes) -> AttributeLayer:
    """Execute a **generated** (lattice-only) plan on ``layer``'s array.

    Raises ``NotImplementedError`` if the plan contains a structure/channel
    bridge (those execute in a later phase); use ``plan.generated`` to check.
    """
    if layer.domain is not plan.src:
        raise ValueError(f"layer is on {layer.domain.value}, plan starts at "
                         f"{plan.src.value}")
    values = np.asarray(layer.values)
    cur_axes = list(axes.axis_list(plan.src)) if is_lattice(plan.src) else []
    for step in plan.steps:
        if isinstance(step, BridgeStep):
            raise NotImplementedError(
                f"bridge '{step.name}' ({step.describe()}) needs structure inputs "
                f"(label raster / point positions) the plan doesn't carry; execute it "
                f"via nodegraph.bridges (voxel_to_label / label_to_voxel / "
                f"voxel_to_point / point_to_voxel / points_in_label / containing_label)")
        values, cur_axes = _apply_lattice_step(values, cur_axes, step, axes)
    return AttributeLayer(plan.dst, layer.name, np.asarray(values), layer.layer)


def lattice_transfer(layer: AttributeLayer, dst: Domain, axes: AxisSizes,
                     reducer: str = DEFAULT_REDUCER) -> AttributeLayer:
    """Convenience: plan + execute a lattice transfer of ``layer`` to ``dst``."""
    return execute_transfer(layer, plan_transfer(layer.domain, dst, reducer), axes)


# ── multi-hop bridge execution (C2 — chain lattice + bridge hops) ─────────────
#
# ``execute_transfer`` runs lattice-only plans; the individual structure bridges live
# in :mod:`nodegraph.bridges`. What was missing (C2) is a single call that walks a
# *routed* :class:`TransferPlan` — e.g. Voxel→Track = Voxel→Label→Track, or
# Voxel→Point→Label — chaining each hop with the structure inputs (a label raster,
# point positions, a track membership). This executor operates at the granularity the
# bridges do: one plane/volume raster + one point set (not the full 6-D lattice); a
# node loops it over (m,t,c). It carries a domain-tagged payload (`Carrier`) and
# dispatches each hop; a hop needing a structure input that was not supplied, or one
# outside the geometric spine (the temporal Track↔Timepoint hops, which are member×t,
# and Frame→structure, which needs the target id set), raises with a clear message.

@dataclass
class Carrier:
    """The payload flowing through a bridge chain: a lattice ``array`` for lattice
    domains, or ``ids``+``values`` for a structure/point domain (Point ids are the row
    order 0..N-1; Label/Track ids are the id set)."""

    domain: Domain
    array: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None


def _need(value, what: str, hop: str):
    if value is None:
        raise ValueError(f"bridge hop {hop} needs {what} — pass it to execute_bridge_plan")
    return value


def _bridge_hop(cur: Carrier, step: "BridgeStep", *, label_raster, points,
                membership, shape) -> Carrier:
    from nodegraph.bridges import (
        broadcast_track, containing_label, gather_by_track, label_to_voxel,
        point_to_voxel, points_in_label, voxel_to_label, voxel_to_point,
    )
    D = Domain
    s, d, hop, red = step.src, step.dst, step.describe(), step.reducer
    if (s, d) == (D.VOXEL, D.LABEL):
        ids, vals = voxel_to_label(_need(cur.array, "the source Voxel array", hop),
                                   _need(label_raster, "a label_raster", hop), red)
        return Carrier(D.LABEL, ids=ids, values=vals)
    if (s, d) == (D.LABEL, D.VOXEL):
        arr = label_to_voxel(cur.ids, cur.values, _need(label_raster, "a label_raster", hop))
        return Carrier(D.VOXEL, array=arr)
    if (s, d) == (D.VOXEL, D.POINT):
        pts = _need(points, "point positions", hop)
        vals = voxel_to_point(_need(cur.array, "the source Voxel array", hop), pts, "linear")
        return Carrier(D.POINT, ids=np.arange(len(vals)), values=vals)
    if (s, d) == (D.POINT, D.VOXEL):
        arr = point_to_voxel(cur.values, _need(points, "point positions", hop),
                             _need(shape, "the target voxel shape", hop), red)
        return Carrier(D.VOXEL, array=arr)
    if (s, d) == (D.POINT, D.LABEL):
        ids, vals = points_in_label(cur.values, _need(points, "point positions", hop),
                                    _need(label_raster, "a label_raster", hop), red)
        return Carrier(D.LABEL, ids=ids, values=vals)
    if (s, d) == (D.LABEL, D.POINT):
        pts = _need(points, "point positions", hop)
        per_pt = containing_label(pts, _need(label_raster, "a label_raster", hop))
        lut = {int(i): float(v) for i, v in zip(cur.ids, cur.values)}
        vals = np.array([lut.get(int(lab), 0.0) for lab in per_pt], dtype=float)
        return Carrier(D.POINT, ids=np.arange(len(vals)), values=vals)
    if (s, d) in ((D.LABEL, D.TRACK), (D.POINT, D.TRACK)):
        mem = _need(membership, "a track membership", hop)
        mids = cur.ids if cur.ids is not None else np.arange(len(cur.values))
        tids, vals = gather_by_track(mids, cur.values, mem, red)
        return Carrier(D.TRACK, ids=tids, values=vals)
    if (s, d) in ((D.TRACK, D.LABEL), (D.TRACK, D.POINT)):
        mem = _need(membership, "a track membership", hop)
        mids, vals = broadcast_track(cur.ids, cur.values, mem)
        return Carrier(d, ids=mids, values=vals)
    if d is D.FRAME and is_structure(s):
        # reduce-in-frame: collapse the structure's values to one frame scalar. Keyed on
        # the predicate, not a hardcoded member list, so it stays correct as the structure
        # family grows — this hop needs nothing domain-specific beyond "has values".
        vals = np.asarray(cur.values, dtype=float)
        out = _reduce(vals, (0,), red) if vals.size else np.asarray(np.nan)
        return Carrier(D.FRAME, values=np.asarray(out).reshape(()))
    raise NotImplementedError(
        f"bridge hop {hop} is outside the chained executor's geometric spine "
        f"(Voxel/Label/Point/Track/→Frame). Temporal Track↔Timepoint hops and "
        f"Frame→structure broadcasts need the extra representation/id-set — call the "
        f"nodegraph.bridges function directly for those.")


def execute_bridge_plan(carrier: Carrier, plan: TransferPlan, *,
                        axes: Optional[AxisSizes] = None,
                        label_raster: Optional[np.ndarray] = None,
                        points: Optional[np.ndarray] = None,
                        membership: Optional[object] = None,
                        shape: Optional[Tuple[int, ...]] = None) -> Carrier:
    """Walk a routed :class:`TransferPlan` for ONE frame/volume, applying each hop.

    A pure-lattice plan delegates to :func:`execute_transfer` (so a coarse→fine
    **broadcast** step is sized to the dataset's real axes — pass ``axes``; a
    reduce/identity-only plan needs none). A routed (bridge) plan chains its hops via
    :mod:`nodegraph.bridges`, threading the supplied structure inputs — closing the C2
    gap where an indirect route (Voxel→Track, Voxel→Point→Label, Label→Frame) was
    *planned* but had no single executor. A structure hop lacking its input raises."""
    grid = axes or AxisSizes()
    if plan.generated:
        layer = AttributeLayer(plan.src, "carrier", np.asarray(carrier.array))
        return Carrier(plan.dst, array=np.asarray(execute_transfer(layer, plan, grid).values))
    cur = carrier
    for step in plan.steps:
        if isinstance(step, LatticeStep):
            if cur.array is None:
                raise NotImplementedError(
                    f"lattice hop {step.describe()} mid-route needs a lattice array "
                    f"carrier; this executor chains the structure spine (C2 scope)")
            cur_axes = [a for a in AXIS_ORDER if a in (axes_of(cur.domain) or set())]
            arr, _ = _apply_lattice_step(cur.array, cur_axes, step, grid)
            cur = Carrier(cur.domain, array=arr)
        else:
            cur = _bridge_hop(cur, step, label_raster=label_raster, points=points,
                              membership=membership, shape=shape)
    return cur


__all__ = [
    "LatticeStep", "BridgeStep", "Step", "TransferPlan", "Bridge",
    "register_bridge", "bridges", "plan_transfer", "execute_transfer",
    "lattice_transfer", "Carrier", "execute_bridge_plan",
]
