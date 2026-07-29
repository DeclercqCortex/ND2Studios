"""Structure-domain producers — whole-domain CCL / seeded watershed → columnar tables
(nodegraph v2, Phase 2a — V2.02 §7 + V2.03 §3 B3 / §4 C3).

Detected-structure domains (Label / Point / Track) are computed **whole** (a whole
plane in 2D, a whole volume in 3D — never per-tile CCL, V2.02 §7/§I.c) and stored as
**columnar tables** (Arrow-target). The 2D/3D lever drives the compute:

* **2D** — a plane mask, ``connectivity`` ∈ {4, 8}, ``px²`` size filters, ``Granularity``
  WHOLE_PLANE. Point ``z_kind = "plane_index"`` (z is the integer plane, never NaN).
* **3D** — a volume mask, ``connectivity`` ∈ {6, 18, 26}, ``px³`` filters, WHOLE_VOLUME.
  Point ``z_kind = "subpixel"`` (interpolated z).

The connectivity/size params ride the node's recipe hash as ordinary params (V2.02 §6),
so 2D-CCL and 3D-CCL of the same input memoize distinctly (V2.03 §3 B3). The
:class:`StructureTable` schema is **invariant** — ``id,m,t,c,z,y,x`` always present, ``z``
never NaN (V2.03 §4 C3) — so downstream bridges never branch on mode.

Core is numpy-only (a deterministic union-free flood-fill CCL) so the module imports
and tests without scipy/skimage/pyarrow; ``to_arrow`` (pyarrow) and
:func:`seeded_watershed` (scipy EDT + skimage watershed) are **lazily imported**.
The flood-fill CCL is the correctness baseline — a scipy/cc3d fast path is a later
optimization the whole-frame-cost benchmark (V2.02 §7) will gate. Qt-free.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nodegraph.domains import Domain
from nodegraph.memo import digest


# ── the columnar structure table (Arrow-target) ──────────────────────────────

# The invariant coordinate columns every structure table carries (V2.03 §4 C3).
COORD_COLUMNS: Tuple[str, ...] = ("id", "m", "t", "c", "z", "y", "x")


@dataclass(frozen=True)
class StructureTable:
    """A Label / Point / Track table: named numpy columns + set-level metadata. The
    concrete Arrow RecordBatch (V2.02 §7) is produced on demand by :meth:`to_arrow`
    (pyarrow lazily imported); this numpy form is the headless source of truth."""

    domain: Domain
    columns: Mapping[str, np.ndarray]
    layer: Optional[str] = None
    z_kind: str = "subpixel"          # "plane_index" (2D) | "subpixel" (3D)
    channel_kind: str = "single"      # "single" | "per_point"

    @property
    def n(self) -> int:
        for v in self.columns.values():
            return int(len(v))
        return 0

    def content_hash(self) -> str:
        """Cheap FULL content hash (tables are KB–MB, V2.02 §H.4)."""
        parts: List[Any] = ["struct", self.domain, self.layer, self.z_kind,
                            self.channel_kind]
        for k in sorted(self.columns):
            parts.append(k)
            parts.append(np.ascontiguousarray(self.columns[k]))
        return digest(*parts)

    def to_arrow(self):
        """A pyarrow ``RecordBatch`` (lazily imported) with metadata for z_kind/
        channel_kind. Arrow is optional — the numpy columns are authoritative."""
        import pyarrow as pa
        batch = pa.record_batch({k: pa.array(np.asarray(v))
                                 for k, v in self.columns.items()})
        return batch.replace_schema_metadata({
            b"domain": self.domain.value.encode(),
            b"z_kind": self.z_kind.encode(),
            b"channel_kind": self.channel_kind.encode(),
            b"layer": (self.layer or "").encode(),
        })


# ── connectivity (the 2D/3D-dependent param set, V2.03 §3 B3) ─────────────────

# connectivity value → max number of nonzero offset components (the neighbourhood
# "rank"): 2D {4→1, 8→2}; 3D {6→1, 18→2, 26→3}.
_CONNECTIVITY: Dict[int, Dict[int, int]] = {2: {4: 1, 8: 2}, 3: {6: 1, 18: 2, 26: 3}}


def _connectivity_rank(ndim: int, connectivity: int) -> int:
    """Validate ``connectivity`` for ``ndim`` and return its neighbourhood **rank** (the max
    number of nonzero offset components): 2D {4→1, 8→2}; 3D {6→1, 18→2, 26→3}. The rank is
    exactly scipy's ``generate_binary_structure(ndim, rank)`` connectivity argument, so the
    pure-numpy offsets and the scipy CCL fast path share one source of truth."""
    valid = _CONNECTIVITY.get(ndim)
    if valid is None or connectivity not in valid:
        raise ValueError(
            f"connectivity {connectivity} invalid for {ndim}D "
            f"(use {sorted(valid) if valid else 'nD=2 or 3'})")
    return valid[connectivity]


def connectivity_offsets(ndim: int, connectivity: int) -> List[Tuple[int, ...]]:
    """Neighbour offsets for ``connectivity`` in ``ndim`` (both directions)."""
    rank = _connectivity_rank(ndim, connectivity)
    return [o for o in itertools.product((-1, 0, 1), repeat=ndim)
            if 1 <= sum(1 for v in o if v) <= rank]


# ── connected-components labelling (WHOLE_PLANE / WHOLE_VOLUME) ────────────────

def _canonical_relabel(raw: np.ndarray, num: int) -> Tuple[np.ndarray, int]:
    """Relabel a components raster ``raw`` (labels ``1..num``, 0 = background) so ids are
    ``1..K`` in **raster-canonical first-appearance C-order** — deterministic and
    tile-independent, identical to the pure-numpy flood-fill's numbering (scipy's own label
    VALUES are scan-order arbitrary). Returns ``(relabeled int64 raster, K)``."""
    if num <= 0:
        return raw.astype(np.int64), 0
    flat = raw.ravel()                                    # C-order (raster canonical)
    nz = flat[flat != 0]
    if nz.size == 0:
        return np.zeros_like(raw, dtype=np.int64), 0
    uniq, first = np.unique(nz, return_index=True)        # uniq sorted; first = first-occ idx
    order = uniq[np.argsort(first)]                       # labels ordered by first appearance
    remap = np.zeros(int(num) + 1, dtype=np.int64)
    remap[order] = np.arange(1, order.size + 1, dtype=np.int64)
    return remap[raw], int(order.size)


def _label_table(labels: np.ndarray, k: int, areas: np.ndarray, cz: np.ndarray,
                 cy: np.ndarray, cx: np.ndarray, *, ndim: int, m: int, t: int, c: int,
                 layer: Optional[str]) -> StructureTable:
    """Assemble the invariant ``id,m,t,c,area,z,y,x`` Label table (review #6)."""
    cols = {
        "id": np.arange(1, k + 1, dtype=np.int64),
        "m": np.full(k, m, dtype=np.int64),
        "t": np.full(k, t, dtype=np.int64),
        "c": np.full(k, c, dtype=np.int64),
        "area": np.asarray(areas, dtype=np.int64),        # px² (2D) / px³ (3D) — voxels
        "z": np.asarray(cz, dtype=float),
        "y": np.asarray(cy, dtype=float),
        "x": np.asarray(cx, dtype=float),
    }
    return StructureTable(Domain.LABEL, cols, layer=layer,
                          z_kind=("plane_index" if ndim == 2 else "subpixel"))


def label_components(mask: np.ndarray, connectivity: int, *, m: int = 0, t: int = 0,
                     c: int = 0, z_index: int = 0, layer: Optional[str] = None
                     ) -> Tuple[np.ndarray, StructureTable]:
    """Label a 2D plane or 3D volume ``mask`` (nonzero = foreground) into connected
    regions and return ``(label_raster, table)``. Ids are **raster-canonical**
    (1..K in first-appearance order — deterministic, tile-independent). The table
    carries the invariant ``id,m,t,c,z,y,x`` schema (review #6): ``m,t,c`` locate the
    plane/volume; ``z``/``y``/``x`` are region centroids (a 2D mask records
    ``z = z_index``, ``z_kind="plane_index"``; a 3D mask the true centroid z).

    **Fast path:** ``scipy.ndimage.label`` (C-level union-find) then a raster-canonical
    relabel, because the pure-numpy flood-fill is ~1200× slower at 512² and extrapolates
    to ~198 s at 6554² (C6 benchmark). scipy's labels are relabelled to first-appearance
    C-order so the raster + table are **byte-identical** to the flood-fill (same
    connectivity ⇒ same partition; only the numbering is normalized). Falls back to the
    flood-fill if scipy is absent (the core stays importable without scipy).
    """
    fg = np.asarray(mask) != 0                       # boolean foreground (m is the M-index)
    if fg.ndim not in (2, 3):
        raise ValueError(f"mask must be 2D or 3D, got {fg.ndim}D")
    rank = _connectivity_rank(fg.ndim, connectivity)     # validate (shared with the offsets)
    try:
        from scipy import ndimage as ndi
    except ImportError:                                   # no scipy → the reference flood-fill
        return _label_components_flood(fg, connectivity, m=m, t=t, c=c,
                                       z_index=z_index, layer=layer)
    raw, num = ndi.label(fg, structure=ndi.generate_binary_structure(fg.ndim, rank))
    labels, k = _canonical_relabel(raw, num)
    areas = np.bincount(labels.ravel(), minlength=k + 1)[1:k + 1].astype(np.int64)
    if k:
        com = np.atleast_2d(np.asarray(
            ndi.center_of_mass(fg, labels, np.arange(1, k + 1)), dtype=float))
    else:
        com = np.zeros((0, fg.ndim), dtype=float)
    if fg.ndim == 2:
        cy, cx = com[:, 0], com[:, 1]
        cz = np.full(k, float(z_index))                  # 2D: z is the plane index
    else:
        cz, cy, cx = com[:, 0], com[:, 1], com[:, 2]     # 3D: true subpixel centroid z
    table = _label_table(labels, k, areas, cz, cy, cx, ndim=fg.ndim, m=m, t=t, c=c,
                         layer=layer)
    return labels, table


def _label_components_flood(fg: np.ndarray, connectivity: int, *, m: int, t: int, c: int,
                            z_index: int, layer: Optional[str]
                            ) -> Tuple[np.ndarray, StructureTable]:
    """Pure-numpy flood-fill CCL — the **reference** implementation (and the scipy-absent
    fallback) for :func:`label_components`, producing identical raster-canonical output.
    O(n) but Python-level per voxel, so ~1200× slower than the scipy fast path (C6); kept
    for correctness parity (the selftest asserts scipy≡flood) and no-scipy environments.
    Takes an already-boolean ``fg``."""
    offsets = connectivity_offsets(fg.ndim, connectivity)
    shape = fg.shape
    labels = np.zeros(shape, dtype=np.int64)
    areas: List[int] = []
    cz: List[float] = []
    cy: List[float] = []
    cx: List[float] = []
    cur = 0
    for start in map(tuple, np.argwhere(fg)):        # raster order → canonical ids
        if labels[start]:
            continue
        cur += 1
        labels[start] = cur
        stack = [start]
        n = 0
        sz = sy = sx = 0.0
        while stack:
            p = stack.pop()
            if fg.ndim == 2:
                y, x = p
                sz += z_index
            else:
                z, y, x = p
                sz += z
            sy += y
            sx += x
            n += 1
            for o in offsets:
                q = tuple(pi + oi for pi, oi in zip(p, o))
                if all(0 <= qi < si for qi, si in zip(q, shape)) and fg[q] and not labels[q]:
                    labels[q] = cur
                    stack.append(q)
        areas.append(n)
        cz.append(sz / n)
        cy.append(sy / n)
        cx.append(sx / n)
    table = _label_table(labels, cur, np.asarray(areas, dtype=np.int64),
                         np.asarray(cz, float), np.asarray(cy, float), np.asarray(cx, float),
                         ndim=fg.ndim, m=m, t=t, c=c, layer=layer)
    return labels, table


# ── id-carrying seeded watershed (stable ids over time, V2.02 §7) ─────────────

def seeded_watershed(fg_mask: np.ndarray, markers: np.ndarray, *,
                     sampling: Optional[Sequence[float]] = None) -> np.ndarray:
    """Split ``fg_mask`` by ``watershed(-EDT, markers, mask=fg)`` so each output label
    id **is** its marker id — stable across the Timepoint axis, never tile-local
    renumbered (V2.02 §7). ``sampling`` gives anisotropic voxel spacing (e.g.
    ``(z_step_um, pixel_size_um, pixel_size_um)``) for the EDT. Lazily imports scipy +
    skimage (no GPU watershed exists — cuCIM #89 — so this stays CPU)."""
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    fg = np.asarray(fg_mask) != 0
    edt = ndi.distance_transform_edt(fg, sampling=sampling)
    return watershed(-edt, markers=np.asarray(markers), mask=fg)


# ── point table (invariant schema, V2.03 §4 C3) ───────────────────────────────

def point_table(positions: np.ndarray, *, z: Any = None, m: int = 0, t: int = 0,
                c: int = 0, ids: Optional[np.ndarray] = None,
                z_kind: Optional[str] = None, channel_kind: str = "single",
                layer: Optional[str] = None) -> StructureTable:
    """A Point table with the invariant ``id,m,t,c,z,y,x`` schema. ``positions`` is
    ``(N,3)`` ``(z,y,x)`` (3D-mode, ``z_kind="subpixel"``) or ``(N,2)`` ``(y,x)``
    (2D-mode — ``z`` is the integer plane index, ``z_kind="plane_index"``, never NaN)."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError(f"positions must be (N,2) or (N,3), got {pos.shape}")
    n = pos.shape[0]
    if pos.shape[1] == 3:
        zc, yc, xc = pos[:, 0], pos[:, 1], pos[:, 2]
        zk = z_kind or "subpixel"
    else:
        yc, xc = pos[:, 0], pos[:, 1]
        zc = (np.asarray(z, dtype=float) if z is not None and not np.isscalar(z)
              else np.full(n, float(z) if z is not None else 0.0))
        zk = z_kind or "plane_index"
    cols = {
        "id": (np.arange(n, dtype=np.int64) if ids is None
               else np.asarray(ids, dtype=np.int64)),
        "m": np.full(n, m, dtype=np.int64),
        "t": np.full(n, t, dtype=np.int64),
        "c": np.full(n, c, dtype=np.int64),
        "z": np.asarray(zc, dtype=float),
        "y": np.asarray(yc, dtype=float),
        "x": np.asarray(xc, dtype=float),
    }
    return StructureTable(Domain.POINT, cols, layer=layer, z_kind=zk,
                          channel_kind=channel_kind)


# ── track membership (defines the Track domain, V2.00 §3.2) ───────────────────

@dataclass(frozen=True)
class TrackMembership:
    """Defines the Track domain: which **member** (a Label or Point id) each track
    occupies at each timepoint. Three parallel columns — ``track_id``, ``t``,
    ``member_id`` — one row per (track, timepoint) occupancy (the temporal identity
    ``t → member``, V2.00 §3.2). ``member_domain`` is LABEL or POINT. The tracking
    *algorithm* that builds this (frame-to-frame linking) is a node/Simulation-zone
    concern; the bridges consume a given membership."""

    track_id: np.ndarray
    t: np.ndarray
    member_id: np.ndarray
    member_domain: Domain = Domain.LABEL

    def __post_init__(self) -> None:
        for name in ("track_id", "t", "member_id"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.int64))
        n = len(self.track_id)
        if not (len(self.t) == n == len(self.member_id)):
            raise ValueError("track_id, t, member_id must be equal-length")

    @property
    def n(self) -> int:
        return int(len(self.track_id))

    def track_ids(self) -> np.ndarray:
        return np.unique(self.track_id)

    def timepoints(self) -> np.ndarray:
        return np.unique(self.t)

    def to_table(self, *, layer: Optional[str] = None) -> StructureTable:
        """A columnar view of the membership rows (Track domain)."""
        return StructureTable(Domain.TRACK, {
            "track_id": self.track_id, "t": self.t, "member_id": self.member_id,
        }, layer=layer, z_kind="plane_index")


__all__ = [
    "COORD_COLUMNS", "StructureTable", "connectivity_offsets",
    "label_components", "seeded_watershed", "point_table", "TrackMembership",
]
