"""Reducers — the aggregation rules used when a transfer coarsens a domain.

Each reducer collapses a numpy array over a set of axis positions, returning the
array with those axes removed. These are the ``mean · median · max · min · sum ·
count · first`` options a domain transfer exposes (``nearest`` is a spatial/
structure-bridge concept handled in :mod:`nodegraph.transfer`, not here).

Qt-free; numpy only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Tuple

import numpy as np

# A reducer: (array, axis-position-tuple) -> reduced array.
Reducer = Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]


def _mean(a: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    return np.nanmean(a, axis=axis) if _has_nan(a) else np.mean(a, axis=axis)


def _sum(a: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    return np.nansum(a, axis=axis) if _has_nan(a) else np.sum(a, axis=axis)


def _max(a: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    return np.nanmax(a, axis=axis) if _has_nan(a) else np.max(a, axis=axis)


def _min(a: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    return np.nanmin(a, axis=axis) if _has_nan(a) else np.min(a, axis=axis)


def _median(a: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    return np.nanmedian(a, axis=axis) if _has_nan(a) else np.median(a, axis=axis)


def _count(a: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    """Number of contributing elements along the reduced axes (NaN-aware)."""
    if _has_nan(a):
        return np.sum(~np.isnan(a), axis=axis)
    counted = int(np.prod([a.shape[ax] for ax in axis])) if axis else 1
    keep = tuple(s for i, s in enumerate(a.shape) if i not in set(axis))
    return np.full(keep, counted, dtype=np.int64)


def _first(a: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    """Take index 0 along each reduced axis (in descending order so positions
    stay valid as dimensions are removed)."""
    out = a
    for ax in sorted(axis, reverse=True):
        out = np.take(out, 0, axis=ax)
    return out


def _has_nan(a: np.ndarray) -> bool:
    return np.issubdtype(a.dtype, np.floating) and bool(np.isnan(a).any())


# ── fusion reducers (robust stacking — SNR ↑ while rejecting outliers) ─────────
#
# Registration/stacking fuses a temporal (or multi-view) axis into one frame; a
# plain mean is optimal only for Gaussian noise, so stacking wants **robust**
# combiners that reject cosmic rays / hot pixels / registration ghosts (open
# decision §7 / node_backend_reference.md "SNR stacking"). Like ``median`` these
# need the whole population along the reduced axis (a global mean/σ, or a sort), so
# they are **not** monoids — absent from ``_PARTIAL`` → ``tree_reduce`` raises and
# the caller realizes the whole domain (consistent with ``median``, V2.02 §7b).

def _sigma_clip(a: np.ndarray, axis: Tuple[int, ...], *,
                k: float = 3.0, iters: int = 3) -> np.ndarray:
    """Sigma-clipped (robust) mean: iteratively drop samples more than ``k`` robust
    standard deviations from the **median** along ``axis``, then mean the survivors.

    Centres on the median and estimates the spread from the MAD (median absolute
    deviation, scaled ×1.4826) rather than the mean/std — a single extreme outlier
    inflates the std enough to survive its own 3σ test (the mean/std form fails to
    reject a cosmic ray), whereas the MAD is unmoved by it. Falls back to the std where
    the MAD is 0 (a near-constant population). NaN-aware; a rejected sample → NaN."""
    x = np.asarray(a, dtype=float).copy()
    for _ in range(max(1, iters)):
        med = np.nanmedian(x, axis=axis, keepdims=True)
        mad = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
        # threshold = k·(scaled MAD); where MAD==0 (a majority-equal / quantized-dark
        # population) the median IS the consensus, so any deviation is an outlier —
        # use a 0 threshold there, NOT the ordinary std (which the outlier itself
        # inflates enough to re-admit the very cosmic ray we must reject).
        thresh = np.where(mad > 0, k * 1.4826 * mad, 0.0)
        with np.errstate(invalid="ignore"):
            reject = np.abs(x - med) > thresh          # a truly constant column: 0 > 0 is False
        if not reject.any():
            break
        x = np.where(reject, np.nan, x)
    return np.nanmean(x, axis=axis)


def _trimmed_mean(a: np.ndarray, axis: Tuple[int, ...], *, trim: int = 1) -> np.ndarray:
    """Winsorized/trimmed mean: sort the samples along the reduced axes and drop the
    ``trim`` lowest and highest before averaging (a rank-robust fusion combiner).

    **NaN-aware.** ``np.sort`` sends NaNs to the high end, so a fixed-width high trim
    would drop NaNs and *keep* the real high outliers (and could empty a cell). Instead
    the trim is taken over each cell's **valid** (non-NaN) population: keep ranks
    ``[trim, k-trim)`` where ``k`` is the per-cell valid count, and if ``k ≤ 2·trim``
    keep all valid samples (never empties a cell that has any valid sample)."""
    x = np.moveaxis(np.asarray(a, dtype=float), axis, tuple(range(-len(axis), 0)))
    keep = x.shape[:x.ndim - len(axis)]
    flat = np.sort(x.reshape(*keep, -1), axis=-1)          # NaNs sort to the high end
    n = flat.shape[-1]
    k = np.sum(~np.isnan(flat), axis=-1, keepdims=True)    # per-cell valid count
    idx = np.arange(n)
    trimmable = k > 2 * trim
    kept = np.where(trimmable, (idx >= trim) & (idx < k - trim), idx < k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(kept, flat, np.nan), axis=-1)


REDUCERS: Dict[str, Reducer] = {
    "mean": _mean,
    "sum": _sum,
    "max": _max,
    "min": _min,
    "median": _median,
    "count": _count,
    "first": _first,
    "sigma_clip": _sigma_clip,
    "trimmed_mean": _trimmed_mean,
}

DEFAULT_REDUCER = "mean"


def reduce(array: np.ndarray, axis: Tuple[int, ...], reducer: str = DEFAULT_REDUCER
           ) -> np.ndarray:
    """Collapse ``array`` over ``axis`` with the named reducer."""
    fn = REDUCERS.get(reducer)
    if fn is None:
        raise ValueError(f"unknown reducer {reducer!r}; "
                         f"choose from {sorted(REDUCERS)}")
    if not axis:
        return array
    return fn(array, tuple(axis))


# ── partial reducers: associative monoids for tree-reduce across tiles ─────────
#
# A lazy provider yields the reduced axes in TILES, so a coarsening (e.g.
# Voxel→Frame reducing z,c,y,x) must fold tile-by-tile instead of realizing the
# whole array. A :class:`PartialReducer` expresses each reducer as a monoid
# ``(lift, combine, lower)`` (V2.02 §5): ``lift`` reduces one tile to a *partial*
# carrying the kept-axis shape; ``combine`` merges two partials **associatively**
# (``None`` is the identity — an empty accumulator or a cancelled tile drops out
# cleanly); ``lower`` finalizes. Tiles partition only the *reduced* axes, so
# partials for one output element share a shape and combine element-wise.
#
# ``mean`` carries a ``(sum, count)`` pair; ``sum/count`` add; ``max/min`` fold
# with NaN-ignoring ``fmax/fmin``; ``first`` keeps the leftmost non-empty partial
# (order-dependent but associative — fold tiles in canonical scan order so the
# tile holding global index 0 wins, V2.02 §J). ``median`` is **not** a monoid, so
# it is absent here and falls back to a whole-domain realization (V2.02 §7b).

Partial = Any   # an array, or a (sum, count) pair for mean, or None (identity)


class PartialReducer(Protocol):
    name: str
    associative: bool
    def lift(self, tile: np.ndarray, axis: Tuple[int, ...]) -> Partial: ...
    def combine(self, p: Partial, q: Partial) -> Partial: ...
    def lower(self, p: Partial) -> np.ndarray: ...


class _Sum:
    name, associative = "sum", True
    def lift(self, tile, axis): return _sum(tile, axis)
    def combine(self, p, q): return q if p is None else (p if q is None else p + q)
    def lower(self, p): return p


class _Count:
    name, associative = "count", True
    def lift(self, tile, axis): return _count(tile, axis)
    def combine(self, p, q): return q if p is None else (p if q is None else p + q)
    def lower(self, p): return p


class _Max:
    name, associative = "max", True
    def lift(self, tile, axis): return _max(tile, axis)
    def combine(self, p, q): return q if p is None else (p if q is None else np.fmax(p, q))
    def lower(self, p): return p


class _Min:
    name, associative = "min", True
    def lift(self, tile, axis): return _min(tile, axis)
    def combine(self, p, q): return q if p is None else (p if q is None else np.fmin(p, q))
    def lower(self, p): return p


class _Mean:
    name, associative = "mean", True
    def lift(self, tile, axis): return (_sum(tile, axis), _count(tile, axis))
    def combine(self, p, q):
        if p is None:
            return q
        if q is None:
            return p
        return (p[0] + q[0], p[1] + q[1])
    def lower(self, p):
        s, c = p
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.asarray(s, dtype=float) / np.asarray(c, dtype=float)


class _First:
    # Order-dependent: keep the leftmost non-empty partial. Associative (a "keep
    # left" monoid), so the fold MUST visit tiles in canonical scan order for the
    # global-first (reduced index 0) to win (V2.02 §J).
    name, associative = "first", True
    def lift(self, tile, axis): return _first(tile, axis)
    def combine(self, p, q): return p if p is not None else q
    def lower(self, p): return p


_PARTIAL: Dict[str, PartialReducer] = {
    r.name: r for r in (_Mean(), _Sum(), _Max(), _Min(), _Count(), _First())
}

#: Reducers that tree-reduce across tiles (``median`` is excluded — see above).
TILEABLE_REDUCERS = frozenset(_PARTIAL)


def partial_reducer(name: str) -> Optional[PartialReducer]:
    """The :class:`PartialReducer` for ``name``, or ``None`` if not tileable."""
    return _PARTIAL.get(name)


def is_tileable(name: str) -> bool:
    """True if ``name`` folds across tiles (False for ``median``)."""
    return name in _PARTIAL


def tree_reduce(tiles: Iterable[np.ndarray], axis: Tuple[int, ...],
                reducer: str = DEFAULT_REDUCER) -> np.ndarray:
    """Fold ``tiles`` (each covering a slab of the reduced ``axis``) into the
    reduced result — equal to :func:`reduce` on the concatenation, without
    realizing it. Raises for a non-tileable reducer (realize the whole domain
    instead) or an empty ``tiles`` sequence."""
    r = partial_reducer(reducer)
    if r is None:
        raise ValueError(
            f"{reducer!r} is not tile-reducible; realize the whole domain "
            f"(V2.02 §5/§7b). Tileable: {sorted(TILEABLE_REDUCERS)}")
    acc: Partial = None
    seen = False
    for tile in tiles:
        acc = r.combine(acc, r.lift(np.asarray(tile), tuple(axis)))
        seen = True
    if not seen:
        raise ValueError("tree_reduce needs at least one tile")
    return r.lower(acc)


__all__ = [
    "Reducer", "REDUCERS", "DEFAULT_REDUCER", "reduce",
    "PartialReducer", "TILEABLE_REDUCERS", "partial_reducer", "is_tileable",
    "tree_reduce",
]
