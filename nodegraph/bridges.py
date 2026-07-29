"""Structure-bridge EXECUTION — the geometric-spine transfers (nodegraph v2, Phase 2).

:mod:`nodegraph.transfer` *plans* and *routes* domain transfers; lattice hops execute
there, but the **detected-structure bridges** (Voxel↔Label/Point, Point↔Label) were
deferred with a ``NotImplementedError`` because they need structure inputs the plan
doesn't carry — the **label raster** that defines the Label domain and the **point
positions** that define the Point domain (V2.00 §6, V2.02 §7). This module supplies
those executions as plain numpy operations, so the node port can move an attribute
across the spine with the default rule shown on the wire.

Implemented (the constructive spine, V2.00 §3.2 / §6):

* ``voxel_to_label`` — group voxels by label id, **mean-in-mask** (or sum/count/max/min/median).
* ``label_to_voxel`` — **paint-by-label**: broadcast a per-label value onto its region.
* ``voxel_to_point`` — **sample-at-position**: nearest (numpy) or linear (scipy, lazy).
* ``point_to_voxel`` — **splat**: deposit point values onto the nearest voxel (sum/mean/max).
* ``containing_label`` — the label a point falls in; ``points_in_label`` reduces the points
  inside each region.

**Deferred:** the Track bridges (gather-by-track / broadcast-track) — they need the Track
membership table (a track's member per timepoint), which the structure model grows next.

Core is numpy-only; linear sampling lazily imports scipy. Qt-free.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from nodegraph.domains import Domain
from nodegraph.structure import TrackMembership


# ── grouping helper (Voxel/Point → Label reductions) ──────────────────────────

_GROUP_REDUCERS = ("mean", "sum", "count", "max", "min", "median")


def _group_reduce(labels: np.ndarray, values: np.ndarray, reducer: str, *,
                  drop_nonpositive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce ``values`` grouped by ``labels``. With ``drop_nonpositive`` (default),
    non-positive keys are dropped as background (Label/Track ids are ≥1); pass False
    for keys where 0 is meaningful (e.g. timepoints). Returns ``(ids, reduced)`` with
    ``ids`` the sorted keys."""
    if reducer not in _GROUP_REDUCERS:
        raise ValueError(f"unknown reducer {reducer!r}; choose {list(_GROUP_REDUCERS)}")
    labels = np.asarray(labels).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if drop_nonpositive:
        fg = labels > 0
        labels, values = labels[fg], values[fg]
    ids = np.unique(labels)
    if ids.size == 0:
        return ids.astype(np.int64), np.array([], dtype=float)
    idx = np.searchsorted(ids, labels)                      # ids sorted → exact
    if reducer in ("sum", "mean", "count"):
        counts = np.bincount(idx, minlength=ids.size)
        sums = np.bincount(idx, weights=values, minlength=ids.size)
        if reducer == "sum":
            out = sums
        elif reducer == "count":
            out = counts.astype(float)
        else:
            out = sums / np.maximum(counts, 1)
    else:
        fn = {"max": np.max, "min": np.min, "median": np.median}[reducer]
        out = np.array([fn(values[idx == k]) if np.any(idx == k) else np.nan
                        for k in range(ids.size)], dtype=float)
    return ids.astype(np.int64), out


# ── Voxel ↔ Label ──────────────────────────────────────────────────────────────

def voxel_to_label(voxel_values: np.ndarray, label_raster: np.ndarray,
                   reducer: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
    """Group a Voxel-domain field by the label mask and reduce per region
    (mean-in-mask by default). ``voxel_values`` and ``label_raster`` share a shape."""
    voxel_values = np.asarray(voxel_values)
    label_raster = np.asarray(label_raster)
    if voxel_values.shape != label_raster.shape:
        raise ValueError(f"shape mismatch {voxel_values.shape} vs {label_raster.shape}")
    return _group_reduce(label_raster, voxel_values, reducer)


def label_to_voxel(label_ids: Sequence[int], label_values: Sequence[float],
                   label_raster: np.ndarray, *, background: float = 0.0) -> np.ndarray:
    """Paint each label's value onto its region's voxels (broadcast); background
    voxels (raster == 0) and ids absent from ``label_ids`` get ``background``."""
    raster = np.asarray(label_raster)
    ids = np.asarray(label_ids, dtype=np.int64)
    vals = np.asarray(label_values, dtype=float)
    hi = int(raster.max()) if raster.size else 0
    lut = np.full(hi + 1, background, dtype=float)
    keep = (ids >= 1) & (ids <= hi)     # drop bg/negative ids (review #11: -1 wrapped)
    lut[ids[keep]] = vals[keep]
    out = lut[np.clip(raster, 0, hi)]
    out[raster == 0] = background
    return out


# ── Voxel ↔ Point ──────────────────────────────────────────────────────────────

def voxel_to_point(voxel_values: np.ndarray, points: np.ndarray,
                   method: str = "nearest") -> np.ndarray:
    """Sample a Voxel-domain field at each point. ``points`` is ``(N, ndim)`` in the
    voxel array's axis order. ``nearest`` (numpy) rounds+clips; ``linear`` lazily
    imports ``scipy.ndimage.map_coordinates``."""
    v = np.asarray(voxel_values)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != v.ndim:
        raise ValueError(f"points {pts.shape} incompatible with {v.ndim}D voxels")
    if method == "nearest":
        idx = tuple(np.clip(np.round(pts[:, d]).astype(np.int64), 0, v.shape[d] - 1)
                    for d in range(v.ndim))
        return v[idx]
    if method == "linear":
        from scipy.ndimage import map_coordinates
        return map_coordinates(v.astype(float), pts.T, order=1, mode="nearest")
    raise ValueError(f"unknown method {method!r} (nearest | linear)")


def point_to_voxel(point_values: Sequence[float], points: np.ndarray,
                   shape: Tuple[int, ...], reducer: str = "sum", *,
                   background: float = 0.0) -> np.ndarray:
    """Splat point values onto their nearest voxels in an array of ``shape``;
    collisions reduce by ``sum`` | ``mean`` | ``max``."""
    pts = np.asarray(points, dtype=float)
    vals = np.asarray(point_values, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != len(shape):
        raise ValueError(f"points {pts.shape} incompatible with shape {shape}")
    idx = tuple(np.clip(np.round(pts[:, d]).astype(np.int64), 0, shape[d] - 1)
                for d in range(len(shape)))
    flat = np.ravel_multi_index(idx, shape)
    size = int(np.prod(shape))
    # explicit coverage so ``background`` fills ONLY empty voxels — never offsets a hit
    # voxel (review #9) and never collides with a legitimate -inf value (review #10).
    covered = np.zeros(size, dtype=bool)
    covered[np.unique(flat)] = True
    if reducer == "sum":
        out = np.zeros(size, dtype=float)
        np.add.at(out, flat, vals)
    elif reducer == "mean":
        out = np.zeros(size, dtype=float)
        cnt = np.zeros(size, dtype=float)
        np.add.at(out, flat, vals)
        np.add.at(cnt, flat, 1.0)
        out[covered] /= cnt[covered]
    elif reducer == "max":
        out = np.full(size, -np.inf, dtype=float)
        np.maximum.at(out, flat, vals)
    else:
        raise ValueError(f"unknown reducer {reducer!r} (sum | mean | max)")
    out[~covered] = background
    return out.reshape(shape)


# ── Point ↔ Label ────────────────────────────────────────────────────────────

def containing_label(points: np.ndarray, label_raster: np.ndarray) -> np.ndarray:
    """The label id each point falls in (0 = background) — nearest-voxel lookup."""
    return voxel_to_point(np.asarray(label_raster), points, "nearest").astype(np.int64)


def points_in_label(point_values: Sequence[float], points: np.ndarray,
                    label_raster: np.ndarray, reducer: str = "mean"
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce point values grouped by the label region that contains each point."""
    labels = containing_label(points, label_raster)
    return _group_reduce(labels, np.asarray(point_values, dtype=float), reducer)


# ── Track bridges (temporal identity over the Timepoint axis, V2.00 §6) ───────

def _member_row_values(membership: TrackMembership, member_ids: Sequence[int],
                       member_values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Value of each membership row's member (NaN + mask where the member is absent)."""
    mids = np.asarray(member_ids, dtype=np.int64)
    mvals = np.asarray(member_values, dtype=float)
    order = np.argsort(mids)
    mids_s, mvals_s = mids[order], mvals[order]
    idx = np.clip(np.searchsorted(mids_s, membership.member_id), 0, len(mids_s) - 1)
    ok = len(mids_s) > 0
    found = (mids_s[idx] == membership.member_id) if ok else np.zeros(membership.n, bool)
    row = np.where(found, mvals_s[idx] if ok else np.nan, np.nan)
    return row, found


def gather_by_track(member_ids: Sequence[int], member_values: Sequence[float],
                    membership: TrackMembership, reducer: str = "mean"
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Label/Point → Track: gather a track's member values over ``t`` and reduce
    (mean/sum/count/max/min/median). Returns ``(track_ids, values)``."""
    row, found = _member_row_values(membership, member_ids, member_values)
    return _group_reduce(membership.track_id[found], row[found], reducer)


def broadcast_track(track_ids: Sequence[int], track_values: Sequence[float],
                    membership: TrackMembership) -> Tuple[np.ndarray, np.ndarray]:
    """Track → Label/Point: broadcast each track's value onto its per-timepoint
    members. Returns ``(member_ids, values)`` for the members that belong to a track."""
    tids = np.asarray(track_ids, dtype=np.int64)
    tvals = np.asarray(track_values, dtype=float)
    order = np.argsort(tids)
    tids_s, tvals_s = tids[order], tvals[order]
    idx = np.clip(np.searchsorted(tids_s, membership.track_id), 0, len(tids_s) - 1)
    found = (tids_s[idx] == membership.track_id) if len(tids_s) else np.zeros(membership.n, bool)
    row = np.where(found, tvals_s[idx] if len(tids_s) else np.nan, np.nan)
    members, first = np.unique(membership.member_id[found], return_index=True)
    return members, row[found][first]


def tracks_per_timepoint(membership: TrackMembership, *,
                         track_ids: Optional[Sequence[int]] = None,
                         track_values: Optional[Sequence[float]] = None,
                         reducer: str = "count") -> Tuple[np.ndarray, np.ndarray]:
    """Track → Timepoint: reduce over the tracks active at each ``t``. Default
    ``count`` = number of distinct active tracks per timepoint; otherwise reduce
    ``track_values`` (keyed by ``track_ids``) over each t's active tracks."""
    ts = membership.t
    if reducer == "count":
        uniq = np.unique(ts)
        counts = np.array([len(np.unique(membership.track_id[ts == tt])) for tt in uniq],
                          dtype=float)
        return uniq.astype(np.int64), counts
    tids = np.asarray(track_ids, dtype=np.int64)
    tvals = np.asarray(track_values, dtype=float)
    order = np.argsort(tids)
    tids_s, tvals_s = tids[order], tvals[order]
    idx = np.clip(np.searchsorted(tids_s, membership.track_id), 0, len(tids_s) - 1)
    row = tvals_s[idx]
    return _group_reduce(ts, row, reducer, drop_nonpositive=False)   # t=0 is valid


def timepoint_to_members(timepoint_values: Sequence[float], timepoints: Sequence[int],
                         membership: TrackMembership) -> Tuple[np.ndarray, np.ndarray]:
    """Timepoint → Track members: broadcast a per-``t`` value onto each track's member
    at ``t``. Returns per-membership-row ``(member_id, value)`` (member×t granularity)."""
    tpts = np.asarray(timepoints, dtype=np.int64)
    tvals = np.asarray(timepoint_values, dtype=float)
    order = np.argsort(tpts)
    tpts_s, tvals_s = tpts[order], tvals[order]
    idx = np.clip(np.searchsorted(tpts_s, membership.t), 0, len(tpts_s) - 1)
    found = (tpts_s[idx] == membership.t) if len(tpts_s) else np.zeros(membership.n, bool)
    row = np.where(found, tvals_s[idx] if len(tpts_s) else np.nan, np.nan)
    return membership.member_id, row


#: Which (src, dst) bridges execute here (signatures differ — see each function).
BRIDGE_FUNCS = {
    (Domain.VOXEL, Domain.LABEL): voxel_to_label,
    (Domain.LABEL, Domain.VOXEL): label_to_voxel,
    (Domain.VOXEL, Domain.POINT): voxel_to_point,
    (Domain.POINT, Domain.VOXEL): point_to_voxel,
    (Domain.POINT, Domain.LABEL): points_in_label,
    (Domain.LABEL, Domain.POINT): containing_label,
    (Domain.LABEL, Domain.TRACK): gather_by_track,
    (Domain.POINT, Domain.TRACK): gather_by_track,
    (Domain.TRACK, Domain.LABEL): broadcast_track,
    (Domain.TRACK, Domain.POINT): broadcast_track,
    (Domain.TRACK, Domain.TIMEPOINT): tracks_per_timepoint,
    (Domain.TIMEPOINT, Domain.TRACK): timepoint_to_members,
}


__all__ = [
    "voxel_to_label", "label_to_voxel", "voxel_to_point", "point_to_voxel",
    "containing_label", "points_in_label",
    "gather_by_track", "broadcast_track", "tracks_per_timepoint",
    "timepoint_to_members", "BRIDGE_FUNCS",
]
