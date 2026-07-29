"""Fields — deferred, per-element value expressions (nodegraph v2, Phase 2 — V2.00 §3.3).

A **field** is a value socket carrying a *deferred function* rather than a concrete
value: it is evaluated per-element on the **consuming node's domain**, only where
consumed (V2.00 §3.3). This module is the minimal field IR + evaluator + a disjoint
field memo namespace (V2.02 §9) — deliberately small per V2.00 §16 ("start minimal,
grow"): arithmetic, comparisons, attribute reads, and select.

IR nodes (all frozen):

* :class:`Const` — a constant → a non-materializing :class:`VirtualArray` (V2.02 §9).
* :class:`Attr` — read a named ``AttributeLayer``; if it lives on another **lattice**
  domain it is transferred to the consuming domain by the default rule (the reducer,
  shown on the wire — V2.00 §5/§6). Structure-domain field transfer is deferred.
* :class:`Input` — an anonymous input field already resolved on the domain.
* :class:`BinOp` / :class:`UnaryOp` / :class:`Where` — arithmetic, comparison, select.

The field memo key (:func:`field_key`) is ``("field", field_expr_hash, domain,
kernel_axes, token)`` — a namespace **disjoint** from the image tile key (V2.02 §9);
``field_expr_hash`` folds operator identity + params + referenced layer **revisions**
(so a changed layer invalidates) + ``kernel_axes`` (V2.03 §4 C2, so a 2D vs 3D stencil
field never collides). Qt-free; numpy + stdlib.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field as _dc_field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import Domain, is_lattice
from nodegraph.memo import digest
from nodegraph.transfer import lattice_transfer


# ── VirtualArray: a non-materializing constant/broadcast field (V2.02 §9) ──────

class VirtualArray:
    """A constant field of a given shape that does not allocate until forced. Stays
    virtual under scalar / VirtualArray ops; materializes (via ``__array__``) the
    moment it meets a real ndarray."""

    __slots__ = ("fill", "shape")

    def __init__(self, fill: float, shape: Tuple[int, ...]) -> None:
        self.fill = fill
        self.shape = tuple(shape)

    def materialize(self) -> np.ndarray:
        return np.full(self.shape, self.fill)

    def __array__(self, dtype=None) -> np.ndarray:
        a = self.materialize()
        return a.astype(dtype) if dtype is not None else a

    def _combine(self, other: Any, op: Callable, *, reflected: bool = False) -> Any:
        if isinstance(other, VirtualArray) and other.shape == self.shape:
            return VirtualArray(op(other.fill, self.fill) if reflected
                                else op(self.fill, other.fill), self.shape)
        if np.isscalar(other):
            return VirtualArray(op(other, self.fill) if reflected
                                else op(self.fill, other), self.shape)
        return NotImplemented          # a real array → numpy materializes via __array__

    def __add__(self, o): return self._combine(o, operator.add)
    def __radd__(self, o): return self._combine(o, operator.add, reflected=True)
    def __sub__(self, o): return self._combine(o, operator.sub)
    def __rsub__(self, o): return self._combine(o, operator.sub, reflected=True)
    def __mul__(self, o): return self._combine(o, operator.mul)
    def __rmul__(self, o): return self._combine(o, operator.mul, reflected=True)
    def __truediv__(self, o): return self._combine(o, operator.truediv)
    def __rtruediv__(self, o): return self._combine(o, operator.truediv, reflected=True)
    def __neg__(self): return VirtualArray(-self.fill, self.shape)

    def __repr__(self) -> str:
        return f"VirtualArray(fill={self.fill!r}, shape={self.shape})"


# ── field IR ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class Attr:
    domain: Domain
    name: str
    layer: Optional[str] = None
    reducer: str = "mean"          # rule applied if transferred to the consuming domain


@dataclass(frozen=True)
class Input:
    name: str


@dataclass(frozen=True)
class BinOp:
    op: str
    a: "Field"
    b: "Field"


@dataclass(frozen=True)
class UnaryOp:
    op: str
    a: "Field"


@dataclass(frozen=True)
class Where:
    cond: "Field"
    a: "Field"
    b: "Field"


Field = Union[Const, Attr, Input, BinOp, UnaryOp, Where]

_BINOPS: Dict[str, Callable] = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "==": operator.eq, "!=": operator.ne, "min": np.minimum, "max": np.maximum,
}
_UNARYOPS: Dict[str, Callable] = {
    "neg": operator.neg, "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
    "exp": np.exp, "not": np.logical_not,
}


# ── evaluation ──────────────────────────────────────────────────────────────────

@dataclass
class FieldContext:
    """What a field is evaluated against: the source ``dataset`` (for Attr reads),
    the consuming ``domain`` + ``axes`` (the target shape), and any anonymous
    ``inputs`` already resolved on the domain.

    ``window`` (C1 / V2.04 §4) restricts evaluation to a sub-extent: a mapping
    ``axis → (start, stop)`` over the domain's axes (missing axes = full extent).
    With a window set, ``Attr`` layers on the consuming domain are **sliced** to it,
    ``Const`` broadcasts to the window's shape, and ``inputs`` must already be
    window-shaped — so a per-tile consumer materializes only its window. Windowed
    **cross-domain** transfer is deferred (raises)."""

    dataset: Dataset
    domain: Domain
    axes: AxisSizes
    inputs: Dict[str, Any] = _dc_field(default_factory=dict)
    window: Optional[Dict[str, Tuple[int, int]]] = None

    def _bounds(self, axis: str) -> Tuple[int, int]:
        full = (0, self.axes.size(axis))
        return tuple(self.window.get(axis, full)) if self.window else full

    def slicer(self, domain: Domain) -> Tuple[slice, ...]:
        """Per-axis slices of ``window`` for a lattice ``domain``'s canonical shape."""
        return tuple(slice(*self._bounds(a)) for a in self.axes.axis_list(domain))

    def target_shape(self) -> Tuple[int, ...]:
        if not is_lattice(self.domain):
            return ()
        if self.window is None:
            return self.axes.shape_for(self.domain)
        return tuple(b - a for a, b in
                     (self._bounds(ax) for ax in self.axes.axis_list(self.domain)))


def evaluate(field: Field, ctx: FieldContext) -> Any:
    """Evaluate ``field`` on ``ctx.domain``. Returns an ndarray, or a
    :class:`VirtualArray` for a constant/broadcast subtree (V2.02 §9)."""
    if isinstance(field, Const):
        return VirtualArray(field.value, ctx.target_shape())
    if isinstance(field, Input):
        if field.name not in ctx.inputs:
            raise KeyError(f"unbound field input {field.name!r}")
        return ctx.inputs[field.name]
    if isinstance(field, Attr):
        layer = ctx.dataset.get(field.domain, field.name, field.layer)
        if layer is None:
            raise KeyError(f"no attribute {field.name!r} on {field.domain.value}"
                           + (f" layer {field.layer!r}" if field.layer else ""))
        if field.domain is ctx.domain:
            if ctx.window is not None:
                return layer.values[ctx.slicer(field.domain)]     # windowed slice (C1)
            return layer.values
        if is_lattice(field.domain) and is_lattice(ctx.domain):
            if ctx.window is not None:
                src_axes = ctx.axes.axis_list(field.domain)
                dst_axes = ctx.axes.axis_list(ctx.domain)
                if set(src_axes) <= set(dst_axes):
                    # REFINE under a window (V2.04 §6b: a coarser-domain Attr indexes
                    # the window's coordinates on ITS axes and broadcasts) — e.g. a
                    # FRAME-domain background layer thresholding a VOXEL window.
                    v = layer.values[tuple(slice(*ctx._bounds(a)) for a in src_axes)]
                    src = set(src_axes)
                    shape = tuple(
                        (ctx._bounds(a)[1] - ctx._bounds(a)[0]) if a in src else 1
                        for a in dst_axes)
                    return np.broadcast_to(v.reshape(shape), ctx.target_shape())
                raise NotImplementedError(
                    "windowed COARSENING field transfer (a reduce over out-of-window "
                    "axes) is deferred (C1 / V2.04 §4); refine/broadcast is supported")
            return lattice_transfer(layer, ctx.domain, ctx.axes, field.reducer).values
        raise NotImplementedError(
            f"field transfer {field.domain.value}→{ctx.domain.value} needs a structure "
            f"bridge (use nodegraph.bridges); deferred for fields")
    if isinstance(field, UnaryOp):
        return _UNARYOPS[field.op](evaluate(field.a, ctx))
    if isinstance(field, BinOp):
        return _BINOPS[field.op](evaluate(field.a, ctx), evaluate(field.b, ctx))
    if isinstance(field, Where):
        cond = np.asarray(evaluate(field.cond, ctx))
        return np.where(cond, np.asarray(evaluate(field.a, ctx)),
                        np.asarray(evaluate(field.b, ctx)))
    raise TypeError(f"not a field node: {field!r}")


# ── field memo namespace (V2.02 §9) ────────────────────────────────────────────

def field_expr_hash(field: Field, dataset: Optional[Dataset] = None) -> str:
    """Hash of the field expression: operator identity + params + referenced layer
    **revisions** (a changed layer invalidates). A missing layer hashes a sentinel."""
    if isinstance(field, Const):
        return digest("const", field.value)
    if isinstance(field, Input):
        return digest("input", field.name)
    if isinstance(field, Attr):
        rev = None
        if dataset is not None:
            layer = dataset.get(field.domain, field.name, field.layer)
            rev = layer.revision if layer is not None else None
        return digest("attr", field.domain, field.name, field.layer, field.reducer, rev)
    if isinstance(field, UnaryOp):
        return digest("unary", field.op, field_expr_hash(field.a, dataset))
    if isinstance(field, BinOp):
        return digest("bin", field.op, field_expr_hash(field.a, dataset),
                      field_expr_hash(field.b, dataset))
    if isinstance(field, Where):
        return digest("where", field_expr_hash(field.cond, dataset),
                      field_expr_hash(field.a, dataset), field_expr_hash(field.b, dataset))
    raise TypeError(f"not a field node: {field!r}")


def field_key(field: Field, domain: Domain, token: Any, *,
              dataset: Optional[Dataset] = None,
              kernel_axes: Tuple[str, ...] = ()) -> str:
    """The disjoint field memo key (V2.02 §9 / V2.03 §4 C2): ``("field", expr_hash,
    domain, kernel_axes, token)`` — ``token`` = a tile id (image-coupled) or a
    domain_version (element-domain). Never collides with the image tile key."""
    return digest("field", field_expr_hash(field, dataset), domain,
                  tuple(sorted(kernel_axes)), token)


class FieldCache:
    """A materialization cache in the field namespace (one site per key, V2.02 §9).

    With a ``store`` (the engine's byte-budget :class:`~nodegraph.streaming.TileCache`,
    C1 / V2.04 §6b) ndarray materializations live under the disjoint ``("f", key)``
    namespace and share the budget; :class:`VirtualArray` results are recreated (free)
    rather than cached. Without a store it is the original unbounded dict (tests)."""

    def __init__(self, store: Any = None) -> None:
        self._by_key: Dict[str, Any] = {}
        # weak: a compute closure capturing this FieldCache must not pin an orphaned
        # engine's whole TileCache (V2.04 §3 weak/late-bound; review 2026-07-22)
        import weakref
        self._store_ref = weakref.ref(store) if store is not None else None
        self.hits = 0
        self.misses = 0

    def evaluate(self, field: Field, ctx: FieldContext, *, token: Any = 0,
                 kernel_axes: Tuple[str, ...] = ()) -> Any:
        key = field_key(field, ctx.domain, token, dataset=ctx.dataset,
                        kernel_axes=kernel_axes)
        if self._store_ref is not None:
            store = self._store_ref()
            if store is None:                    # engine gone: evaluate uncached
                self.misses += 1
                return evaluate(field, ctx)
            hit = store.get(("f", key))
            if hit is not None:
                self.hits += 1
                return hit
            self.misses += 1
            val = evaluate(field, ctx)
            if isinstance(val, np.ndarray):
                return store.put(("f", key), val)
            return val
        if key in self._by_key:
            self.hits += 1
            return self._by_key[key]
        self.misses += 1
        val = evaluate(field, ctx)
        self._by_key[key] = val
        return val


__all__ = [
    "VirtualArray", "Const", "Attr", "Input", "BinOp", "UnaryOp", "Where", "Field",
    "FieldContext", "evaluate", "field_expr_hash", "field_key", "FieldCache",
]
