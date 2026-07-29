"""Lazy tiled image provider (nodegraph v2, Phase 2a — V2.02 §3 + V2.03 §5 D1).

``Dataset.image`` holds a :class:`TileProvider`: the lazy voxel source the pull
engine reads through. The keystone benchmark (2026-07-21, real 6554² ND2 plane)
settled the read granularity: a 512² block ROI costs **1.5–4.8%** of a whole-plane
read → **TILED**; and for 3D, **planar `(1,512,512)` blocks win** — a z-range
subvolume reads in ~11% of whole-volume with per-z blocks vs ~47% with a fat
z-spanning block (blosc2 decompresses whole blocks), so **block = one 2D tile per
z**, and a subvolume is *gathered* from planar blocks across the z-range.

The contract (V2.03 §5 D1): ``get_tile``/``get_region`` serve TILEABLE / WHOLE_PLANE
pulls at a single z; ``get_subvolume``/``get_region_volume`` serve the WHOLE_VOLUME
(and WHOLE_SERIES z-range) pulls the 2D/3D toggle's 3D mode needs — realized whole
once and memoized as one payload upstream (V2.02 §7b), then sliced per display tile.

Concrete providers implement :meth:`TileProvider.read_region` (a single-z 2D window)
and report ``axes``/``levels``/``tile``; the base derives every other read from it.

* :class:`SyntheticProvider` — a deterministic formula (numpy only; for tests).
* :class:`B2ndProvider` — a Blosc2 b2nd store with planar blocks (``blosc2`` is
  **lazily imported**, so the core and the synthetic provider need no blosc2). ND2
  ingest (ND2 → 6-D numpy) is an app/ingest-layer concern that feeds
  :meth:`B2ndProvider.from_array`; ``nodegraph`` itself stays nd2-free.

Qt-free; numpy at the core, blosc2 optional.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, List, Optional

import numpy as np

from nodegraph.dataset import AxisSizes


class TileProvider(ABC):
    """The lazy voxel-source contract. Subclasses set ``axes``/``levels``/``tile``
    and implement :meth:`read_region`; the base derives tile/volume reads."""

    axes: AxisSizes
    levels: int = 1
    tile: int = 512

    # ── the one primitive concrete providers implement ────────────────────────
    @abstractmethod
    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """A single-z 2D window ``[y0:y1, x0:x1]`` at ``level`` for ``(m,t,z,c)``."""

    # ── multiscale geometry ────────────────────────────────────────────────────
    def level_axes(self, level: int) -> AxisSizes:
        """Axis sizes at ``level`` (spatial pyramid — y,x halve per level)."""
        f = 1 << level
        ax = self.axes
        return replace(ax, y=max(1, ax.y // f), x=max(1, ax.x // f))

    def tiles_per_plane(self, level: int = 0) -> tuple:
        ax = self.level_axes(level)
        return (-(-ax.y // self.tile), -(-ax.x // self.tile))   # ceil-div (ny, nx)

    def fingerprint(self) -> tuple:
        """A memo identity for this provider's content, folded into a Dataset's
        output-fingerprint (so datasets differing only by image don't collide/dedup).
        The base is **structural** (type + geometry) — correct for a deterministic
        provider like :class:`SyntheticProvider`; a content-backed provider overrides
        it (see :class:`ArrayProvider`). A distinct on-disk source that is only
        structurally identical is the deferred provider-identity item (review #3)."""
        return ("provider", type(self).__name__,
                (self.axes.m, self.axes.t, self.axes.z, self.axes.c,
                 self.axes.y, self.axes.x), self.levels, self.tile)

    @property
    def version(self) -> Any:
        """The provider's data identity folded into a **source** node's recipe key
        (C5 / review #3): the engine reads ``prov.version`` into ``__provider_version__``
        so two providers with different data on the same source node do not collide on
        one cached payload. Defaults to :meth:`fingerprint` — structural for a
        deterministic provider (:class:`SyntheticProvider`), content for
        :class:`ArrayProvider` (it overrides ``fingerprint``). A disk-backed provider
        (the future on-disk :class:`B2ndProvider`) overrides this to fold in the file's
        ``mtime_ns``/id so an in-place file change invalidates the memo."""
        return self.fingerprint()

    # ── derived reads (V2.03 §5 D1) ────────────────────────────────────────────
    def get_region(self, level: int, m: int, t: int, z: int, c: int,
                   y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        ax = self.level_axes(level)
        return self.read_region(level, m, t, z, c,
                                max(0, y0), min(y1, ax.y), max(0, x0), min(x1, ax.x))

    def get_tile(self, level: int, m: int, t: int, z: int, c: int,
                 iy: int, ix: int) -> np.ndarray:
        """One block at grid position ``(iy, ix)`` — clipped at the plane edge."""
        ax = self.level_axes(level)
        y0, x0 = iy * self.tile, ix * self.tile
        if y0 >= ax.y or x0 >= ax.x or iy < 0 or ix < 0:
            raise IndexError(f"tile ({iy},{ix}) out of range for level {level} "
                             f"{(ax.y, ax.x)} tile={self.tile}")
        return self.read_region(level, m, t, z, c, y0, min(y0 + self.tile, ax.y),
                                x0, min(x0 + self.tile, ax.x))

    def get_subvolume(self, level: int, m: int, t: int, c: int,
                      z0: int, z1: int, iy: int, ix: int) -> np.ndarray:
        """A ``(z1-z0, block_y, block_x)`` brick — planar blocks gathered across the
        z-range (the benchmark-preferred path; not a MIP). An empty range returns a
        ``(0, block_y, block_x)`` array (review #12: ``np.stack([])`` would crash)."""
        if z1 <= z0:
            probe = self.get_tile(level, m, t, 0, c, iy, ix)   # z=0 always in range
            return np.empty((0,) + probe.shape, dtype=probe.dtype)
        return np.stack([self.get_tile(level, m, t, z, c, iy, ix)
                         for z in range(z0, z1)], axis=0)

    def get_region_volume(self, level: int, m: int, t: int, c: int,
                          z0: int, z1: int, y0: int, y1: int, x0: int, x1: int
                          ) -> np.ndarray:
        """An arbitrary ``(Z,Y,X)`` ROI (composes single-z region reads); an empty
        range returns a ``(0, ...)`` array (review #12)."""
        if z1 <= z0:
            probe = self.get_region(level, m, t, 0, c, y0, y1, x0, x1)
            return np.empty((0,) + probe.shape, dtype=probe.dtype)
        return np.stack([self.get_region(level, m, t, z, c, y0, y1, x0, x1)
                         for z in range(z0, z1)], axis=0)


# ── synthetic provider (numpy only; deterministic — for tests) ────────────────

class SyntheticProvider(TileProvider):
    """A provider whose voxels are a cheap deterministic function of their address,
    so a window read is exact and O(window) — ideal for testing the read contract.
    Levels are **stride-decimated** (level ``l`` samples every ``2**l``)."""

    def __init__(self, axes: AxisSizes, *, tile: int = 512, levels: int = 1) -> None:
        self.axes = axes
        self.tile = tile
        self.levels = levels

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        s = 1 << level
        yy = (np.arange(y0, y1, dtype=np.int64) * s)[:, None]
        xx = (np.arange(x0, x1, dtype=np.int64) * s)[None, :]
        val = (yy * 7 + xx * 3 + z * 131 + c * 17 + t * 19 + m * 23) & 0xFFF
        return np.broadcast_to(val, (y1 - y0, x1 - x0)).astype(np.uint16)


# ── Blosc2 b2nd provider (planar blocks; blosc2 lazily imported) ──────────────

def _mean_downsample_2x(a: np.ndarray) -> np.ndarray:
    """Mean-pool the trailing (Y,X) by 2× (intensity pyramid). Returns ``a`` if a
    spatial axis is < 2 (cannot halve further)."""
    y, x = a.shape[-2], a.shape[-1]
    if y < 2 or x < 2:
        return a
    a = a[..., :y // 2 * 2, :x // 2 * 2]
    r = a.reshape(*a.shape[:-2], y // 2, 2, x // 2, 2).mean(axis=(-3, -1))
    return r.astype(a.dtype)


class B2ndProvider(TileProvider):
    """A Blosc2 b2nd-backed provider. Each pyramid level is a 6-D ``(M,T,Z,C,Y,X)``
    b2nd array with **planar blocks** ``(1,1,1,1,tile,tile)`` (the benchmark verdict);
    a window read decompresses only the touched planar blocks."""

    def __init__(self, arrays: List, axes: AxisSizes, *, tile: int = 512,
                 urlpath: Optional[str] = None, mtime_ns: Optional[int] = None) -> None:
        self._arrays = arrays
        self.axes = axes
        self.tile = tile
        self.levels = len(arrays)
        self._urlpath = urlpath          # set for a disk-backed store (else in-memory)
        self._mtime_ns = mtime_ns        # store mtime → cheap identity for a disk store

    def level_axes(self, level: int) -> AxisSizes:
        m, t, z, c, y, x = self._arrays[level].shape       # exact stored geometry
        return AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x)

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        return np.asarray(self._arrays[level][m, t, z, c, y0:y1, x0:x1])

    def fingerprint(self) -> tuple:
        """Content/data identity (C5 / review). A b2nd store is content-bearing, so the
        base *structural* identity would collide two stores of equal geometry but
        different pixels (a wrong memo hit + blob-dedup aliasing). A **disk** store folds
        in its path + ``mtime_ns`` (cheap — an in-place file edit re-stamps mtime → new
        identity; C4/C5 disk path). An **in-memory** store hashes the level-0 payload
        (O(level0) decompress) — **memoized on the instance** (C1: streaming chains
        re-read the base fingerprint per provider construction; the store is immutable
        for this provider's lifetime, so hash once)."""
        if self._urlpath is not None:
            m, t, z, c, y, x = self._arrays[0].shape
            return ("b2nd-disk", str(self._urlpath), self._mtime_ns,
                    (m, t, z, c, y, x), str(self._arrays[0].dtype))
        fp = getattr(self, "_fp", None)
        if fp is not None:
            return fp
        import hashlib
        a0 = np.ascontiguousarray(self._arrays[0][...])
        h = hashlib.blake2b(a0.tobytes(), digest_size=16).hexdigest()
        self._fp = ("b2nd", a0.shape, str(a0.dtype), self.levels, self.tile, h)
        return self._fp

    # ── ingest / persistence (blosc2 lazily imported; nd2-free) ─────────────────
    @staticmethod
    def _build_levels(vol6d: np.ndarray, *, tile: int, levels: int,
                      cparams: Optional[dict], urlpath: Optional[str] = None) -> List:
        """Build ``levels`` planar-block b2nd pyramid arrays from a 6-D volume. If
        ``urlpath`` is a directory, each level persists to ``level_<l>.b2nd`` there;
        otherwise the arrays are in-memory."""
        import blosc2
        import os
        if vol6d.ndim != 6:
            raise ValueError(f"expected a 6-D (M,T,Z,C,Y,X) array, got {vol6d.shape}")
        cparams = cparams or {"codec": blosc2.Codec.ZSTD,
                              "filters": [blosc2.Filter.BITSHUFFLE]}

        def store(vol: np.ndarray, level: int):
            _, _, _, _, y, x = vol.shape
            by, bx = min(tile, y), min(tile, x)
            cy, cx = min(2 * tile, y), min(2 * tile, x)
            kw: dict = {"chunks": (1, 1, 1, 1, cy, cx), "blocks": (1, 1, 1, 1, by, bx),
                        "cparams": cparams}
            if urlpath is not None:
                kw["urlpath"] = os.path.join(urlpath, f"level_{level}.b2nd")
                kw["mode"] = "w"
            return blosc2.asarray(np.ascontiguousarray(vol), **kw)

        arrays, cur = [], vol6d
        for lvl in range(max(1, levels)):
            arrays.append(store(cur, lvl))
            if lvl < levels - 1:
                cur = _mean_downsample_2x(cur)
        return arrays

    @classmethod
    def from_array(cls, vol6d: np.ndarray, *, tile: int = 512, levels: int = 1,
                   cparams: Optional[dict] = None) -> "B2ndProvider":
        """Ingest a ``(M,T,Z,C,Y,X)`` numpy volume into an **in-memory** planar-block
        b2nd store with ``levels`` mean-downsampled pyramid levels. (An ND2 → 6-D numpy
        reader is an app/ingest-layer concern; ``nodegraph`` stays nd2-free.)"""
        arrays = cls._build_levels(vol6d, tile=tile, levels=levels, cparams=cparams)
        m, t, z, c, y, x = vol6d.shape
        return cls(arrays, AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x), tile=tile)

    @classmethod
    def write(cls, vol6d: np.ndarray, urlpath: str, *, tile: int = 512, levels: int = 1,
              cparams: Optional[dict] = None) -> "B2ndProvider":
        """Persist a 6-D volume to an **on-disk** b2nd store at directory ``urlpath``
        (one ``level_<l>.b2nd`` per pyramid level) and return a provider opened over it
        (the C4 disk path — ingest once, then a lazy disk-backed provider)."""
        import os
        os.makedirs(urlpath, exist_ok=True)
        cls._build_levels(np.asarray(vol6d), tile=tile, levels=levels, cparams=cparams,
                          urlpath=urlpath)
        return cls.open(urlpath)

    @classmethod
    def open(cls, urlpath: str) -> "B2ndProvider":
        """Open an on-disk b2nd store written by :meth:`write` — lazy (blocks decompress
        on read). ``version``/``fingerprint`` fold in the store's ``mtime_ns`` so an
        in-place file change invalidates the memo (C5)."""
        import blosc2
        import glob
        import os
        files = sorted(
            glob.glob(os.path.join(urlpath, "level_*.b2nd")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]))
        if not files:
            raise FileNotFoundError(f"no level_*.b2nd store in {urlpath!r}")
        arrays = [blosc2.open(f) for f in files]
        m, t, z, c, y, x = arrays[0].shape
        blocks = getattr(arrays[0], "blocks", None)
        tile = int(blocks[-1]) if blocks else 512
        mtime_ns = max(os.stat(f).st_mtime_ns for f in files)
        return cls(arrays, AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x), tile=tile,
                   urlpath=urlpath, mtime_ns=mtime_ns)


class ArrayProvider(TileProvider):
    """A provider backed by an in-memory ``(M,T,Z,C,Y,X)`` array — how a *realizing*
    node (e.g. deconvolve) wraps its computed volume back into ``Dataset.image``.
    Single-level (no pyramid); numpy only."""

    def __init__(self, array: np.ndarray, *, tile: int = 512) -> None:
        a = np.asarray(array)
        if a.ndim != 6:
            raise ValueError(f"expected a 6-D (M,T,Z,C,Y,X) array, got {a.shape}")
        self._a = a
        m, t, z, c, y, x = a.shape
        self.axes = AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x)
        self.tile = tile
        self.levels = 1

    @property
    def nbytes(self) -> int:
        """The realized raster's in-memory bytes — what a memo eviction of the Dataset
        holding this provider actually frees (Memo GC sizing, :func:`~nodegraph.memo.payload_bytes`)."""
        return int(self._a.nbytes)

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        if level != 0:
            raise ValueError("ArrayProvider has no pyramid (level 0 only)")
        return np.asarray(self._a[m, t, z, c, y0:y1, x0:x1])

    def fingerprint(self) -> tuple:
        """Content fingerprint — the realized array's bytes (this provider IS its
        content, so two ArrayProviders differ iff their arrays differ). **Memoized on
        the instance** (C1): the array is frozen by convention once memoized, and
        streaming chains re-read base fingerprints per construction — hash once."""
        fp = getattr(self, "_fp", None)
        if fp is not None:
            return fp
        import hashlib
        digest = hashlib.blake2b(np.ascontiguousarray(self._a).tobytes(),
                                 digest_size=16).hexdigest()
        # `tile` is part of the identity: the streaming TileCache addresses tiles by
        # (iy, ix) ON this provider's grid, so two same-content providers on different
        # grids must never alias (C1 review 2026-07-22).
        self._fp = ("array", self._a.shape, str(self._a.dtype), self.tile, digest)
        return self._fp


__all__ = ["TileProvider", "SyntheticProvider", "B2ndProvider", "ArrayProvider"]
