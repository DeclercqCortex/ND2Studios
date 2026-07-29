"""Per-tile lazy / streaming evaluation (nodegraph v2, C1 — V2.04, LOCKED 2026-07-22).

The C1 execution model is **provider-chaining**: a ``TILEABLE`` / ``WHOLE_PLANE`` /
``WHOLE_VOLUME`` node returns a Dataset whose image is a *computing* lazy provider
instead of a realized :class:`~nodegraph.provider.ArrayProvider`. A
:class:`MapComputeProvider` serves ``read_region`` on demand: it decomposes the request
into **canonical tiles of its own grid**, computes each missing tile by reading the
tile extent **+ halo** from its base provider (itself possibly lazy — the chain is the
dataflow), applying the node's kernel, and cropping; only canonical tiles enter the
shared :class:`TileCache` (V2.04 §6b) so every chain level gets cache reuse. Plane /
volume units cache whole planes (volumes as per-z planar slabs — the keystone verdict).

Correctness pillars (V2.04 §3/§6b):

* **Flat fingerprints** — every streaming provider's identity is a single digest string
  computed **once at construction** from ``(op_key, params, declared calibration reads,
  field expression hashes, base fingerprint)``. Folding the declared reads closes the
  ``reseed_meta`` staleness hole (the node-level reads fence does not protect the tile
  cache); folding field expression hashes (which embed layer *revisions*) closes the
  attribute-layer hole; flatness avoids the nested-tuple ``_canon`` recursion blow-up
  on unrolled zone chains.
* **Halo = overlap-recompute** (V2.04 §2, probe-verified): windows are clipped at the
  *immediate base's* true extents, reproducing scipy ``mode='reflect'`` edge behavior.
* **cum-halo fence** — when ``2·cum_halo ≥ tile`` the accumulated windows make tiling
  pointless; the provider silently switches to the plane unit (V2.04 §6b).
* **Uniform freeze** — every array a streaming provider returns is read-only; an
  in-place kernel fails loudly and deterministically.

Qt-free; numpy + stdlib.
"""
from __future__ import annotations

import sys
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.memo import digest
from nodegraph.provider import ArrayProvider, TileProvider
from nodegraph.reducers import partial_reducer, reduce as _reduce_whole


# ── fingerprint stability guard (V2.04 §6b: no id-bearing reprs in a fp) ────────

_STABLE_SCALARS = (type(None), bool, int, float, str, bytes, np.integer, np.floating,
                   Enum)


def _assert_stable(o: Any, path: str = "value") -> None:
    """Reject values whose canonical encoding would fall into ``_canon``'s repr
    fallback (id-bearing reprs make the fingerprint unstable across re-pulls —
    silently cache-cold + ``changed`` always true)."""
    if isinstance(o, _STABLE_SCALARS) or isinstance(o, np.ndarray):
        return
    if isinstance(o, (tuple, list, set, frozenset)):
        for i, x in enumerate(o):
            _assert_stable(x, f"{path}[{i}]")
        return
    if isinstance(o, dict):
        for k, v in o.items():
            _assert_stable(k, f"{path} key {k!r}")
            _assert_stable(v, f"{path}[{k!r}]")
        return
    raise TypeError(
        f"unstable fingerprint input at {path}: {type(o).__name__!r} would hash via "
        f"repr() (id-bearing) — bake only plain scalars/containers into a streaming "
        f"provider fingerprint (V2.04 §6b)")


def stream_fp(kind: str, op_key: str, params: Mapping[str, Any],
              reads: Tuple[Tuple[str, str], ...], field_hashes: Tuple[str, ...],
              base: TileProvider) -> str:
    """The flat streaming-provider fingerprint digest (V2.04 §3/§6b): op + params +
    **declared calibration reads** (everything the closure could have baked from
    ``ctx.calib``) + **field expression hashes** (embed layer revisions) + the base
    provider's fingerprint. Computed once at construction; O(1) per chain level
    (a streaming base's own fingerprint is already a flat digest)."""
    p = dict(params)
    _assert_stable(p, "params")
    base_fp = base.fingerprint()
    _assert_stable(base_fp, "base fingerprint")
    # the base's tile size is folded in because the TileCache key's (iy, ix) address is
    # only meaningful ON a grid — two same-content providers on different grids must
    # never share tile entries (review 2026-07-22 BLOCKER: cross-grid collisions)
    return digest("stream", kind, op_key, p, tuple(reads), tuple(field_hashes),
                  base_fp, getattr(base, "tile", None))


# ── recursion headroom (deep unrolled chains; V2.04 §6b fence notes) ────────────

def _stack_depth() -> int:
    f = sys._getframe()
    d = 0
    while f is not None:
        d += 1
        f = f.f_back
    return d


@contextmanager
def recursion_headroom(extra_frames: int):
    """Scoped ``sys.setrecursionlimit`` raise — the stopgap for deep unrolled-zone
    chains (a recursive read/pull nests a few frames per chain level; the iterative
    ``_entry`` rewrite is a follow-up). Headroom is granted **relative to the live
    stack depth** — entering from an already-deep stack must not silently grant less
    than requested (review 2026-07-22)."""
    need = _stack_depth() + int(extra_frames) + 200
    old = sys.getrecursionlimit()
    if need <= old:
        yield
        return
    sys.setrecursionlimit(need)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


# ── the shared byte-budget LRU tile/unit cache (V2.04 §3) ───────────────────────

class TileCache:
    """Engine-owned LRU over frozen ndarrays, keyed by namespaced tuples that embed a
    streaming provider's fingerprint digest — the concrete V2.02 §9 "image tile key".
    Field materializations share the budget under a disjoint ``("f", ...)`` namespace.
    An entry larger than the whole budget is served but never stored; eviction is
    plain LRU. Eviction never invalidates handed-out arrays (they are immutable)."""

    def __init__(self, budget_bytes: int = 1 << 30) -> None:
        self.budget = int(budget_bytes)
        self._lru: "OrderedDict[Any, np.ndarray]" = OrderedDict()
        self.nbytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[np.ndarray]:
        a = self._lru.get(key)
        if a is None:
            self.misses += 1
            return None
        self._lru.move_to_end(key)
        self.hits += 1
        return a

    def put(self, key: Any, arr: np.ndarray) -> np.ndarray:
        """Freeze (owning the buffer first — a frozen *view* still reflects base
        writes) and store; returns the stored (read-only) array."""
        a = np.asarray(arr)
        if a.base is not None:
            a = a.copy()
        a.flags.writeable = False
        if a.nbytes > self.budget:               # oversize: serve, never store
            return a
        old = self._lru.pop(key, None)
        if old is not None:
            self.nbytes -= old.nbytes
        self._lru[key] = a
        self.nbytes += a.nbytes
        while self.nbytes > self.budget and len(self._lru) > 1:
            _, v = self._lru.popitem(last=False)
            self.nbytes -= v.nbytes
        return a

    def __len__(self) -> int:
        return len(self._lru)

    def clear(self) -> None:
        self._lru.clear()
        self.nbytes = 0
        self.hits = self.misses = 0


# ── streaming provider base ─────────────────────────────────────────────────────

def _freeze(a: np.ndarray) -> np.ndarray:
    if a.base is not None:
        a = a.copy()
    if a.flags.writeable:
        a.flags.writeable = False
    return a


class _NoCache:
    """Fallback when a provider's (weakly-held) engine cache has been collected:
    every read recomputes, results still come back frozen. Correctness-preserving —
    a memoized lazy Dataset stays readable after its engine is gone (V2.04 §3:
    providers hold a weak/late-bound cache reference; review 2026-07-22)."""

    budget = 0

    def get(self, key: Any) -> Optional[np.ndarray]:
        return None

    def put(self, key: Any, arr: np.ndarray) -> np.ndarray:
        return _freeze(np.asarray(arr))


_NO_CACHE = _NoCache()


class StreamProvider(TileProvider):
    """Base for computing lazy providers: flat cached fingerprint, chain ``depth``,
    accumulated halo, and canonical-tile request decomposition. ``levels = 1``:
    pyramid semantics for *computed* results are a Viewer decision (V2.04 §5-1).
    The cache reference is **weak** — a memoized Dataset must not pin an orphaned
    engine's whole TileCache across engine rebuilds (V2.04 §3)."""

    levels = 1

    def __init__(self, base: TileProvider, *, fp: str, cache: TileCache) -> None:
        self._base = base
        self._fp = fp
        self._cache_ref = weakref.ref(cache)
        self.axes = base.axes
        self.tile = base.tile
        self.depth = getattr(base, "depth", 0) + 1
        self.cum_halo = getattr(base, "cum_halo", 0)

    @property
    def _cache(self):
        c = self._cache_ref()
        return c if c is not None else _NO_CACHE

    def fingerprint(self) -> tuple:
        return ("stream", self._fp)              # flat — O(1) at any chain depth

    def level_axes(self, level: int) -> AxisSizes:
        if level != 0:
            raise ValueError("streaming providers have no pyramid (level 0 only)")
        return self.axes

    # ── request assembly ──────────────────────────────────────────────────────
    def _assemble(self, y0: int, y1: int, x0: int, x1: int,
                  tile_of: Callable[[int, int], np.ndarray]) -> np.ndarray:
        """Serve window ``[y0:y1, x0:x1]`` from canonical tiles. A request that IS one
        whole canonical tile returns the cached array itself (no copy); any other
        window is assembled into a fresh (frozen) buffer — copy-on-serve, so a held
        window never pins a whole cached plane (V2.04 §6b)."""
        T = self.tile
        ay, ax_ = self.axes.y, self.axes.x
        y0, y1 = max(0, y0), min(y1, ay)
        x0, x1 = max(0, x0), min(x1, ax_)
        if y1 <= y0 or x1 <= x0:
            return _freeze(np.empty((max(0, y1 - y0), max(0, x1 - x0)), dtype=float))
        if (y0 % T == 0 and x0 % T == 0
                and y1 == min(y0 + T, ay) and x1 == min(x0 + T, ax_)):
            return tile_of(y0 // T, x0 // T)     # exactly one canonical tile
        out: Optional[np.ndarray] = None
        for iy in range(y0 // T, (y1 - 1) // T + 1):
            for ix in range(x0 // T, (x1 - 1) // T + 1):
                tl = tile_of(iy, ix)
                ty0, tx0 = iy * T, ix * T
                sy0, sy1 = max(y0, ty0), min(y1, ty0 + tl.shape[0])
                sx0, sx1 = max(x0, tx0), min(x1, tx0 + tl.shape[1])
                if out is None:
                    out = np.empty((y1 - y0, x1 - x0), dtype=tl.dtype)
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                    tl[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
        return _freeze(out)

    @staticmethod
    def _window_of(plane: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """A window of a cached plane — copy-on-serve unless it is the whole plane."""
        y1 = min(y1, plane.shape[0])
        x1 = min(x1, plane.shape[1])
        if y0 == 0 and x0 == 0 and (y1, x1) == plane.shape:
            return plane
        return _freeze(plane[y0:y1, x0:x1].copy())


# ── the map providers (TILEABLE tile+halo / WHOLE_PLANE plane / WHOLE_VOLUME) ───

#: internal kernel contract — receives the float window and its address
#: ``(m, t, z, c, gy0, gy1, gx0, gx1)`` (the actual extent incl. halo), so a
#: field-consuming kernel can evaluate its Field on exactly that window.
MapFn = Callable[..., np.ndarray]


class MapComputeProvider(StreamProvider):
    """The TILEABLE / WHOLE_PLANE lazy unit (V2.04 §1). ``unit="tile"`` computes
    canonical tiles from tile+halo base windows (overlap-recompute, halo clipped at
    the base's true extents); ``unit="plane"`` computes whole planes on first touch.
    The cum-halo fence silently promotes a tile unit to the plane unit when
    ``2·cum_halo ≥ tile`` (window blow-up makes tiling pointless, V2.04 §6b)."""

    def __init__(self, base: TileProvider, fn: MapFn, *, halo: int = 0,
                 unit: str = "tile", fp: str, cache: TileCache) -> None:
        super().__init__(base, fp=fp, cache=cache)
        self._fn = fn
        self.halo = int(max(0, halo))
        self.cum_halo = getattr(base, "cum_halo", 0) + self.halo
        self._plane_unit = (unit == "plane") or (2 * self.cum_halo >= self.tile)
        if self._plane_unit:
            # a plane-unit level computes + caches its WHOLE unit — a window-growth
            # cut point: downstream halo accumulation restarts here, so halo-0 chains
            # after a tripped fence stay tile-unit (review 2026-07-22)
            self.cum_halo = 0

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if level != 0:
            raise ValueError("streaming providers have no pyramid (level 0 only)")
        if self._plane_unit:
            return self._window_of(self._plane(m, t, z, c), y0, y1, x0, x1)
        return self._assemble(y0, y1, x0, x1,
                              lambda iy, ix: self._tile(m, t, z, c, iy, ix))

    def _tile(self, m: int, t: int, z: int, c: int, iy: int, ix: int) -> np.ndarray:
        key = ("t", self._fp, m, t, z, c, iy, ix)
        a = self._cache.get(key)
        if a is not None:
            return a
        T, h = self.tile, self.halo
        ay, ax_ = self.axes.y, self.axes.x
        ty0, tx0 = iy * T, ix * T
        ty1, tx1 = min(ty0 + T, ay), min(tx0 + T, ax_)
        gy0, gx0 = max(0, ty0 - h), max(0, tx0 - h)
        gy1, gx1 = min(ay, ty1 + h), min(ax_, tx1 + h)
        win = self._base.get_region(0, m, t, z, c, gy0, gy1, gx0, gx1)
        res = np.asarray(
            self._fn(np.asarray(win, dtype=float), m, t, z, c, gy0, gy1, gx0, gx1),
            dtype=float)
        res = res[ty0 - gy0:ty1 - gy0, tx0 - gx0:tx1 - gx0]
        return self._cache.put(key, res)

    def _plane(self, m: int, t: int, z: int, c: int) -> np.ndarray:
        key = ("p", self._fp, m, t, z, c)
        a = self._cache.get(key)
        if a is not None:
            return a
        ay, ax_ = self.axes.y, self.axes.x
        win = self._base.get_region(0, m, t, z, c, 0, ay, 0, ax_)
        res = np.asarray(self._fn(np.asarray(win, dtype=float), m, t, z, c,
                                  0, ay, 0, ax_), dtype=float)
        return self._cache.put(key, res)


#: volume kernel contract — ``(vol_float, m, t, c) -> (Z, Y, X)``.
VolumeFn = Callable[[np.ndarray, int, int, int], np.ndarray]


class VolumeComputeProvider(StreamProvider):
    """The WHOLE_VOLUME lazy unit: first touch of any window in ``(m, t, c)`` computes
    the whole ``(Z, Y, X)`` volume once and caches it as **per-z planar slabs** (the
    keystone verdict — never one monolith), then serves windows from the z-plane.
    Strictly better than today's realize-ALL-of-6D: only touched volumes compute."""

    def __init__(self, base: TileProvider, vfn: VolumeFn, *, fp: str,
                 cache: TileCache) -> None:
        super().__init__(base, fp=fp, cache=cache)
        self._vfn = vfn
        self.cum_halo = 0        # a realized-unit level is a window-growth cut point

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if level != 0:
            raise ValueError("streaming providers have no pyramid (level 0 only)")
        if not (0 <= z < self.axes.z):
            raise IndexError(f"z={z} out of range [0, {self.axes.z}) — an out-of-range "
                             f"read must not trigger a whole-volume compute")
        key = ("p", self._fp, m, t, z, c)
        plane = self._cache.get(key)
        if plane is None:
            ax = self.axes
            vol = self._base.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
            res = np.asarray(self._vfn(np.asarray(vol, dtype=float), m, t, c),
                             dtype=float)
            if res.shape != (ax.z, ax.y, ax.x):
                raise ValueError(
                    f"volume kernel returned {res.shape}, expected {(ax.z, ax.y, ax.x)}")
            for zi in range(ax.z):               # per-z planar slabs (best-effort cache)
                stored = self._cache.put(("p", self._fp, m, t, zi, c), res[zi])
                if zi == z:
                    plane = stored
        return self._window_of(plane, y0, y1, x0, x1)


class _AxisReduceProvider(StreamProvider):
    """Engine-driven tree-reduce over ONE acquisition axis (``_axis`` = ``"z"`` or
    ``"t"``, set by a subclass): output tile ``(iy, ix)`` folds the base's slices along
    that axis **incrementally** via the matching :class:`~nodegraph.reducers.PartialReducer`
    monoid (memory O(window)); a non-monoid reducer (median / sigma_clip / trimmed_mean)
    stacks the reduced-axis column (memory O(window·N)) — both exact, neither realizes a
    whole plane. Tiles are cast to float at lift so the tiled result is byte-identical to
    the eager ``astype(float)`` reduce."""

    _axis = "z"

    def __init__(self, base: TileProvider, reducer: str, *, fp: str,
                 cache: TileCache) -> None:
        super().__init__(base, fp=fp, cache=cache)
        self._reducer = reducer
        self.axes = replace(base.axes, **{self._axis: 1})

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if level != 0:
            raise ValueError("streaming providers have no pyramid (level 0 only)")
        return self._assemble(y0, y1, x0, x1,
                              lambda iy, ix: self._tile(m, t, z, c, iy, ix))

    def _tile(self, m: int, t: int, z: int, c: int, iy: int, ix: int) -> np.ndarray:
        # the reduced axis is size-1 in the output, so the caller's coordinate for it is
        # 0 (z-reduce) / 0 (t-reduce); it is included in the key harmlessly and the base
        # loop below overwrites it with the real slice index.
        key = ("t", self._fp, m, t, z, c, iy, ix)
        a = self._cache.get(key)
        if a is not None:
            return a
        T = self.tile
        n = getattr(self._base.axes, self._axis)
        ty0, tx0 = iy * T, ix * T
        ty1, tx1 = min(ty0 + T, self.axes.y), min(tx0 + T, self.axes.x)
        coords = {"m": m, "t": t, "z": z, "c": c}

        def slab(i: int) -> np.ndarray:
            coords[self._axis] = i                # vary only the reduced axis
            return self._base.get_region(0, coords["m"], coords["t"], coords["z"],
                                         coords["c"], ty0, ty1, tx0, tx1)

        pr = partial_reducer(self._reducer)
        if pr is not None:
            acc = None
            for i in range(n):
                acc = pr.combine(acc, pr.lift(np.asarray(slab(i), dtype=float)[None], (0,)))
            res = np.asarray(pr.lower(acc), dtype=float)
        else:                                     # non-monoid: stack the reduced-axis column
            col = np.stack([np.asarray(slab(i), dtype=float) for i in range(n)], axis=0)
            res = np.asarray(_reduce_whole(col, (0,), self._reducer), dtype=float)
        return self._cache.put(key, res)


class ZReduceProvider(_AxisReduceProvider):
    """Reduce over Z → a z==1 provider (``util.zproject``; V2.04 §1)."""

    _axis = "z"


class TReduceProvider(_AxisReduceProvider):
    """Reduce over Timepoint → a t==1 provider (``util.stack`` T→1 SNR fusion). The
    tree-reduce that closes the V2.04 §6b `util.stack` sliver — the whole-series
    reduce streams per tile, folding all T incrementally, instead of eagerly stacking
    the T series in memory per (m, z, c)."""

    _axis = "t"


#: 2D per-plane op — ``(plane_float, m, t, z, c) -> (ny, nx)`` (output geometry may differ).
PlaneFn = Callable[..., np.ndarray]


class PlaneRealizeProvider(StreamProvider):
    """A per-unit lazy realize whose OUTPUT geometry may differ from the base — the
    "lazy units" provider for a non-tileable / geometry-changing op (``util.resample``;
    V2.04 §6b). No tiling, no halo: each output UNIT (a whole plane in 2D, a whole
    ``(Z, Y, X)`` volume in 3D) is computed **once on first touch** from the whole input
    unit, cached (volumes as per-z planar slabs), then windows are served from it.
    Strictly better than realizing ALL of 6-D up front — a Viewer scrubbing one
    ``(m, t, z, c)`` computes only that unit."""

    def __init__(self, base: TileProvider, out_axes: AxisSizes, *,
                 plane_fn: Optional[PlaneFn] = None,
                 volume_fn: Optional[VolumeFn] = None, is_volume: bool = False,
                 fp: str, cache: TileCache) -> None:
        super().__init__(base, fp=fp, cache=cache)
        self.axes = out_axes
        self._plane_fn = plane_fn
        self._volume_fn = volume_fn
        self._is_volume = bool(is_volume)
        self.cum_halo = 0        # a realized-unit level is a window-growth cut point
        if self._is_volume and volume_fn is None:
            raise ValueError("PlaneRealizeProvider(is_volume=True) needs a volume_fn")
        if not self._is_volume and plane_fn is None:
            raise ValueError("PlaneRealizeProvider needs a plane_fn")

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if level != 0:
            raise ValueError("streaming providers have no pyramid (level 0 only)")
        if not (0 <= z < self.axes.z):
            raise IndexError(f"z={z} out of range [0, {self.axes.z}) — an out-of-range "
                             f"read must not trigger a whole-unit compute")
        key = ("p", self._fp, m, t, z, c)
        plane = self._cache.get(key)
        if plane is None:
            plane = self._unit(m, t, z, c)
        return self._window_of(plane, y0, y1, x0, x1)

    def _unit(self, m: int, t: int, z: int, c: int) -> np.ndarray:
        out, bax = self.axes, self._base.axes
        if self._is_volume:
            vin = self._base.get_region_volume(0, m, t, c, 0, bax.z, 0, bax.y, 0, bax.x)
            res = np.asarray(self._volume_fn(np.asarray(vin, dtype=float), m, t, c),
                             dtype=float)
            if res.shape != (out.z, out.y, out.x):
                raise ValueError(f"volume op returned {res.shape}, expected "
                                 f"{(out.z, out.y, out.x)}")
            plane = None
            for zi in range(out.z):                # cache per-z planar slabs
                stored = self._cache.put(("p", self._fp, m, t, zi, c), res[zi])
                if zi == z:
                    plane = stored
            return plane
        # 2D: the output plane at (m,t,z,c) is a function of the whole input plane at the
        # SAME (m,t,z,c) — z is unchanged in a 2D op (resample keeps z; normalize/drift too).
        pin = self._base.get_region(0, m, t, z, c, 0, bax.y, 0, bax.x)
        res = np.asarray(self._plane_fn(np.asarray(pin, dtype=float), m, t, z, c),
                         dtype=float)
        if res.shape != (out.y, out.x):
            raise ValueError(f"plane op returned {res.shape}, expected {(out.y, out.x)}")
        return self._cache.put(("p", self._fp, m, t, z, c), res)


class WindowView(TileProvider):
    """A lazy crop — a pure translated sub-extent view (the ``_ChannelView`` pattern
    with offsets, V2.04 §6b). No compute, no cache; kernel ops downstream clip their
    halos at THIS view's extents, which reproduces the eager reflect-at-crop-edge
    behavior exactly (the eager path also filtered the already-cropped array)."""

    levels = 1

    def __init__(self, base: TileProvider, *, z0: int = 0, y0: int = 0, x0: int = 0,
                 axes: AxisSizes) -> None:
        self._base = base
        self._z0, self._y0, self._x0 = int(z0), int(y0), int(x0)
        self.axes = axes
        self.tile = base.tile
        self.depth = getattr(base, "depth", 0) + 1
        self.cum_halo = getattr(base, "cum_halo", 0)
        self._fp = digest("window", base.fingerprint(), (self._z0, self._y0, self._x0),
                          (axes.m, axes.t, axes.z, axes.c, axes.y, axes.x))

    def fingerprint(self) -> tuple:
        return ("window", self._fp)

    def level_axes(self, level: int) -> AxisSizes:
        if level != 0:
            raise ValueError("WindowView has no pyramid (level 0 only)")
        return self.axes

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        if level != 0:
            raise ValueError("WindowView has no pyramid (level 0 only)")
        if not (0 <= z < self.axes.z):
            raise IndexError(f"z={z} out of range for cropped view [0, {self.axes.z}) "
                             f"— translating it would serve pixels OUTSIDE the crop")
        return self._base.read_region(
            0, m, t, z + self._z0, c,
            y0 + self._y0, y1 + self._y0, x0 + self._x0, x1 + self._x0)


# ── realization (sinks: export, debug-verify, oversize fallback) ────────────────

def realize(payload: Any) -> Any:
    """Force a Dataset's lazy image to bytes → an :class:`ArrayProvider`-backed
    Dataset. Non-Datasets, image-less Datasets, and already-realized images pass
    through. Used by ``assert_zone_pure`` (structural fingerprints are equal by
    construction — purity must compare BYTES, V2.04 §3) and export."""
    if not isinstance(payload, Dataset) or payload.image is None:
        return payload
    prov = payload.image
    if isinstance(prov, ArrayProvider):
        return payload
    ax = prov.axes
    with recursion_headroom(8 * (getattr(prov, "depth", 0) + 2)):
        probe = prov.get_region(0, 0, 0, 0, 0, 0, min(1, ax.y), 0, min(1, ax.x))
        out = np.empty((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=probe.dtype)
        for m in range(ax.m):
            for t in range(ax.t):
                for z in range(ax.z):
                    for c in range(ax.c):
                        out[m, t, z, c] = prov.get_region(0, m, t, z, c,
                                                          0, ax.y, 0, ax.x)
    return payload.with_image(ArrayProvider(out, tile=prov.tile))


__all__ = [
    "TileCache", "StreamProvider", "MapComputeProvider", "VolumeComputeProvider",
    "ZReduceProvider", "TReduceProvider", "PlaneRealizeProvider",
    "WindowView", "stream_fp", "recursion_headroom", "realize",
]
