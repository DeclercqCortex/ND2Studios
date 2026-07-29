"""Frame-to-frame tracking — the algorithm that PRODUCES a ``TrackMembership``
(nodegraph v2, Phase 2b — V2.00 §3.2 / §6, roadmap item C3).

The Track domain is *defined* by a :class:`~nodegraph.structure.TrackMembership`
(three parallel columns ``track_id``, ``t``, ``member_id`` — one row per
(track, timepoint) occupancy). The four Track bridges in
:mod:`nodegraph.bridges` (``gather_by_track`` / ``broadcast_track`` /
``tracks_per_timepoint`` / ``timepoint_to_members``) all *consume* a membership;
until now nothing *built* one. This module is that missing producer.

Two linkers, both **pure numpy, deterministic** (no RNG, no wall-clock — the
tie-breaks are the only source of order and they are fully specified):

* :func:`link_labels` — links Label regions across consecutive timepoints by
  **maximum overlap (IoU)**. For each ``t → t+1`` it scores every pair
  ``(label a at t, label b at t+1)`` by intersection-over-union of their voxels
  and matches greedily 1-to-1 (highest IoU first) above ``iou_threshold``. A
  chain of matched ids across time is one track. Appearance/disappearance falls
  out for free (an unmatched id is a 1-frame track); merges/splits are resolved
  greedily to 1-1 (the loser starts/ends a track).

* :func:`link_points` — greedy **nearest-neighbour** linking of Point detections
  across consecutive timepoints, within ``max_distance`` (in the point-coordinate
  units the caller supplies — the ``track.link`` node feeds micron-scaled
  coordinates so the threshold reads in µm).

Both linkers assume the member ids are **globally unique** across timepoints —
exactly what :func:`~nodegraph.nodes._compute_label` (``analysis.label``) and
:func:`~nodegraph.nodes._compute_spots` (``detect.spots``) emit (an id appears at
exactly one ``t``). That is also what the Track bridges require, since they key a
member's value by its id. The produced ``member_id`` column therefore matches the
Label/Point ``id`` column the bridges will be handed.

Track ids are assigned ``1..K`` deterministically by **first appearance**: a
track's key is ``(earliest t, smallest member id at that t)``, tracks are sorted
by that key, and ids are handed out in order.

Qt-free; numpy core (scipy/skimage are never needed here).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from nodegraph.domains import Domain
from nodegraph.structure import TrackMembership


# ── union-find over member ids (chain the per-frame links into tracks) ────────

class _UnionFind:
    """A tiny deterministic union-find keyed by member id (a plain ``int``). The
    root of a merged set is always the **smaller** id, so grouping is independent
    of the order unions arrive in."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:                 # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        lo, hi = (ra, rb) if ra < rb else (rb, ra)    # smaller id becomes the root
        self.parent[hi] = lo


def _build_membership(t_of: Mapping[int, int], links: Sequence[Tuple[int, int]],
                      member_domain: Domain) -> TrackMembership:
    """Assemble a :class:`TrackMembership` from the per-member timepoints ``t_of``
    (``member_id -> t``) and the frame-to-frame ``links`` (``(a, b)`` pairs). Members
    connected through the links share a track; track ids are ``1..K`` in
    first-appearance order; rows are emitted sorted by ``(track_id, t, member_id)``."""
    uf = _UnionFind()
    for mid in t_of:
        uf.find(mid)                                  # every member is at least a singleton
    for a, b in links:
        uf.union(int(a), int(b))

    groups: Dict[int, List[int]] = {}
    for mid in t_of:
        groups.setdefault(uf.find(mid), []).append(mid)

    def first_appearance(members: List[int]) -> Tuple[int, int]:
        return min((t_of[m], m) for m in members)     # (earliest t, smallest id there)

    ordered = sorted(groups.values(), key=first_appearance)
    track_id: List[int] = []
    t_col: List[int] = []
    member_id: List[int] = []
    for tid, members in enumerate(ordered, start=1):
        for m in sorted(members, key=lambda mm: (t_of[mm], mm)):
            track_id.append(tid)
            t_col.append(t_of[m])
            member_id.append(m)
    return TrackMembership(track_id=track_id, t=t_col, member_id=member_id,
                           member_domain=member_domain)


# ── label linking (maximum IoU overlap) ───────────────────────────────────────

def _areas(raster: np.ndarray) -> Dict[int, int]:
    """Voxel count of every positive label id in ``raster`` (background 0 dropped)."""
    vals, counts = np.unique(raster[raster > 0], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())}


def _match_overlap(a_raster: np.ndarray, b_raster: np.ndarray,
                   iou_threshold: float) -> List[Tuple[int, int]]:
    """Greedy 1-to-1 match of the label ids in ``a_raster`` to those in ``b_raster``
    by descending IoU (ties broken by ``(id_a, id_b)`` ascending — deterministic).
    Only pairs with ``iou >= iou_threshold`` and a positive voxel overlap match."""
    a = np.asarray(a_raster).ravel()
    b = np.asarray(b_raster).ravel()
    both = (a > 0) & (b > 0)
    a, b = a[both], b[both]
    if a.size == 0:
        return []
    pairs = np.stack([a, b], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)   # co-occurrence counts
    area_a = _areas(np.asarray(a_raster))
    area_b = _areas(np.asarray(b_raster))
    cands: List[Tuple[float, int, int]] = []
    for (ai, bi), overlap in zip(uniq.tolist(), counts.tolist()):
        ai, bi, overlap = int(ai), int(bi), int(overlap)
        union = area_a[ai] + area_b[bi] - overlap
        iou = overlap / union if union > 0 else 0.0
        if overlap > 0 and iou >= iou_threshold:
            cands.append((iou, ai, bi))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))     # highest IoU first, then ids
    used_a: set = set()
    used_b: set = set()
    matched: List[Tuple[int, int]] = []
    for _iou, ai, bi in cands:
        if ai in used_a or bi in used_b:
            continue
        used_a.add(ai)
        used_b.add(bi)
        matched.append((ai, bi))
    return matched


def link_labels(rasters_by_t: Mapping[int, np.ndarray], *,
                iou_threshold: float = 0.0) -> TrackMembership:
    """Link Label regions across time by maximum-overlap (IoU) and return a
    :class:`TrackMembership` (``member_domain = Domain.LABEL``).

    Parameters
    ----------
    rasters_by_t
        ``{t: label_raster}`` — one labelled integer array per timepoint (0 =
        background), exactly as ``analysis.label`` emits: ids are **globally unique**
        (an id appears at one ``t`` only). Every raster must share the same shape (a
        ``(Y, X)`` plane in 2D mode, a ``(Z, Y, X)`` volume in 3D mode); overlap is
        the count of voxels where two ids coincide. Links are formed between
        **adjacent timepoints in sorted key order** (a gap in ``t`` is bridged as
        adjacency).
    iou_threshold
        Minimum intersection-over-union for a match (default ``0.0`` — any positive
        voxel overlap links; raise it to reject grazing touches).

    Notes
    -----
    Deterministic: candidate pairs are ranked by ``(-IoU, id_a, id_b)`` and matched
    greedily 1-to-1. A label with no accepted match on either side is a single-frame
    track (appearance / disappearance). Merges & splits resolve to 1-1 greedily — the
    highest-IoU partner wins, the runner-up begins or ends its own track.
    """
    ts = sorted(rasters_by_t)
    t_of: Dict[int, int] = {}
    for t in ts:
        raster = np.asarray(rasters_by_t[t])
        for lbl in np.unique(raster).tolist():
            if int(lbl) > 0:
                t_of[int(lbl)] = int(t)
    links: List[Tuple[int, int]] = []
    for t0, t1 in zip(ts, ts[1:]):
        a_raster = np.asarray(rasters_by_t[t0])
        b_raster = np.asarray(rasters_by_t[t1])
        if a_raster.shape != b_raster.shape:
            raise ValueError(
                f"label rasters at t={t0} {a_raster.shape} and t={t1} "
                f"{b_raster.shape} must share a shape to compute voxel overlap")
        links.extend(_match_overlap(a_raster, b_raster, iou_threshold))
    return _build_membership(t_of, links, Domain.LABEL)


# ── point linking (greedy nearest neighbour) ───────────────────────────────────

def _match_nn(ids_a: np.ndarray, pos_a: np.ndarray, ids_b: np.ndarray,
              pos_b: np.ndarray, max_distance: float) -> List[Tuple[int, int]]:
    """Greedy 1-to-1 nearest-neighbour match between two point sets within
    ``max_distance`` (Euclidean, in the coordinate units supplied). Candidate pairs
    are ranked by ``(distance, id_a, id_b)`` ascending — deterministic."""
    ids_a = np.asarray(ids_a)
    ids_b = np.asarray(ids_b)
    p_a = np.asarray(pos_a, dtype=float)
    p_b = np.asarray(pos_b, dtype=float)
    if p_a.size == 0 or p_b.size == 0:
        return []
    if p_a.ndim != 2 or p_b.ndim != 2 or p_a.shape[1] != p_b.shape[1]:
        raise ValueError(
            f"positions must be (N, D) with matching D, got {p_a.shape} / {p_b.shape}")
    dists = np.sqrt(((p_a[:, None, :] - p_b[None, :, :]) ** 2).sum(axis=-1))
    cands: List[Tuple[float, int, int]] = []
    for i in range(len(ids_a)):
        for j in range(len(ids_b)):
            dist = float(dists[i, j])
            if dist <= max_distance:
                cands.append((dist, int(ids_a[i]), int(ids_b[j])))
    cands.sort(key=lambda c: (c[0], c[1], c[2]))      # nearest first, then ids
    used_a: set = set()
    used_b: set = set()
    matched: List[Tuple[int, int]] = []
    for _dist, ai, bi in cands:
        if ai in used_a or bi in used_b:
            continue
        used_a.add(ai)
        used_b.add(bi)
        matched.append((ai, bi))
    return matched


def link_points(positions_by_t: Mapping[int, Tuple[Sequence[int], np.ndarray]], *,
                max_distance: float = float("inf")) -> TrackMembership:
    """Link Point detections across time by greedy nearest-neighbour and return a
    :class:`TrackMembership` (``member_domain = Domain.POINT``).

    Parameters
    ----------
    positions_by_t
        ``{t: (ids, positions)}`` — per timepoint, the point ids (**globally unique**,
        as ``detect.spots`` emits) and their ``(N, D)`` coordinates. ``D`` must match
        across timepoints; the ``track.link`` node passes micron-scaled ``(z, y, x)``
        so ``max_distance`` reads in µm, but any consistent coordinate space works.
        Links are formed between **adjacent timepoints in sorted key order**.
    max_distance
        Maximum link distance in the coordinate units (default ``inf`` — always link
        the nearest available partner).

    Notes
    -----
    Deterministic: candidate pairs are ranked by ``(distance, id_a, id_b)`` and matched
    greedily 1-to-1. An unmatched point starts / ends a track (appearance /
    disappearance).
    """
    ts = sorted(positions_by_t)
    t_of: Dict[int, int] = {}
    for t in ts:
        ids, _pos = positions_by_t[t]
        for pid in np.asarray(ids).tolist():
            t_of[int(pid)] = int(t)
    links: List[Tuple[int, int]] = []
    for t0, t1 in zip(ts, ts[1:]):
        ids0, pos0 = positions_by_t[t0]
        ids1, pos1 = positions_by_t[t1]
        links.extend(_match_nn(ids0, pos0, ids1, pos1, max_distance))
    return _build_membership(t_of, links, Domain.POINT)


#: Public alias of :func:`_build_membership`. A tracker that computes its links some
#: other way (e.g. ``track.objects``, which delegates to the vendored ``track_objects``
#: kernel) assembles its :class:`TrackMembership` through this so it inherits the exact
#: conventions the built-in linkers use — contiguous ``1..K`` track ids in
#: first-appearance order, rows sorted ``(track_id, t, member_id)`` — and the two
#: trackers' outputs stay directly comparable.
build_membership = _build_membership


__all__ = ["link_labels", "link_points", "build_membership"]
