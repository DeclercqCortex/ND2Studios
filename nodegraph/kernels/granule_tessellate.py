"""Granule tessellation + density-merge kernel (self-contained, portable).

PURPOSE
    Turn a labelled 3-D point cloud of bead centroids into per-granule boundary
    meshes and a FINAL (post-merge) per-point granule assignment. Given an initial
    label per point (e.g. a GMM clustering output), each granule's points are turned
    into a boundary (alpha-shape / concave hull, or a Voronoi-cell union), and an
    iterative region-adjacency density-merge folds adjacent granules with
    similar-enough density together (union-find, higher id -> lower id).

WHERE THE REAL MATH LIVES
    In-repo, in THIS file. The geometry is plain ``scipy.spatial`` (ConvexHull,
    Delaunay, Voronoi, cKDTree) plus hand-written tetrahedron circumradius/volume
    and boundary-face extraction. No external analysis package is involved; there is
    no heavy learned model. numpy + scipy are the only compute deps.

PROVENANCE (branch Version-1.45)
    Vendored verbatim from:
      * nd2studios/backend/analysis/granule_tessellate.py  (the whole kernel)
      * nd2studios/backend/analysis/granule_types.py       (only NOISE_LABEL and the
        GranuleBoundary + GranuleTessellation dataclasses, copied in below)

    Vendored verbatim; imports nothing from nd2studios; caller owns all prep.

EDITS MADE (per the vendoring rules)
    * Removed ``from nd2studios.backend.analysis.granule_types import (...)`` and
      copied the referenced NOISE_LABEL constant + GranuleBoundary + GranuleTessellation
      dataclasses directly into this file (byte-for-byte from granule_types.py).
    * Added ``from dataclasses import dataclass, field`` (required by the copied
      dataclasses; was implicit via the granule_types module before).
    * DROPPED UI/registry-only members: NONE. The source backend module has no
      get_params()/Registry decorators/ParamSpec/ParamEditor hooks; every symbol
      here is on the compute path.

CALLER OWNS ALL PREP
    The kernel takes an already-prepared point cloud + initial labels. It does no
    file I/O, no per-multipoint / per-timepoint looping, no bead detection, no
    crop/downsample/registration/exclusion. Those are the caller's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi, cKDTree
from scipy.spatial import QhullError

NOISE_LABEL = -1


@dataclass
class GranuleBoundary:
    """One granule's boundary + the density the merge compares.

    ``vertices_um``/``faces`` are a triangle mesh in world ``(x, y, z)`` µm (the
    ``viz3d`` convention). ``delaunay`` is retained (when built by the alpha-shape
    path) so P4's voxel inside-test is a cheap ``find_simplex`` with no
    re-triangulation; it is ``None`` for Voronoi-mode boundaries (P4 rebuilds one).
    """

    granule_id: int
    vertices_um: np.ndarray                      # (Nv, 3) world (x, y, z) µm
    faces: np.ndarray                            # (Nf, 3) int triangle indices
    enclosed_volume_um3: float
    n_points: int
    density: float                               # n_points / enclosed_volume_um3 (or 1/cell_vol)
    delaunay: Optional[Any] = None               # scipy.spatial.Delaunay | None


@dataclass
class GranuleTessellation:
    """P3 output → P4 input. Carries the FINAL (post-merge) granule assignment.

    ``point_labels`` is the final granule id per INPUT point (index-aligned with the
    ``(N,3)`` cloud in ``record._granule_points_by_m[m][t]``); ``merged_from`` maps
    each final id to the list of original GMM ids folded into it.
    """

    mode: str                                    # "alpha_shape" | "voronoi"
    boundaries: Dict[int, GranuleBoundary]       # final granule_id -> boundary
    point_labels: np.ndarray                     # (N,) int final granule id per point
    voxel_size_um: Tuple[float, float, float]    # (dz, dy, dx) the boundaries are in
    merged_from: Dict[int, List[int]] = field(default_factory=dict)


# Public mode strings (P6 param choices).
TESS_MODE_ALPHA = "alpha_shape"
TESS_MODE_VORONOI = "voronoi"

# Parameter defaults (P6 supplies real values; these keep the function standalone).
_DEFAULTS: Dict[str, Any] = {
    "tess_mode": TESS_MODE_ALPHA,
    "alpha": float("inf"),   # circumradius threshold in µm; <=0 or None => convex hull
    "merge_tol": 0.15,       # density-ratio tolerance in [0, 1]
    "adj_dist_um": 5.0,      # alpha-shape adjacency distance (µm)
    "min_granule_points": 4, # granules below this are dropped to noise
}

# Geometric floor: a 3-D Delaunay / ConvexHull needs >= 4 non-coplanar points.
_MIN_HULL_POINTS = 4
_EPS = 1e-12


# ── parameter parsing ─────────────────────────────────────────────────────────
def _get(params: Optional[Dict[str, Any]], key: str) -> Any:
    if isinstance(params, dict) and params.get(key) is not None:
        return params[key]
    return _DEFAULTS[key]


# ── coordinate conversion ───────────────────────────────────────────────────────
def _points_to_xyz_um(points_zyx: np.ndarray,
                      voxel_size_um: Tuple[float, float, float]) -> np.ndarray:
    """``(N,3)`` ``(z,y,x)`` voxels → ``(N,3)`` world ``(x,y,z)`` µm."""
    pts = np.asarray(points_zyx, dtype=float).reshape(-1, 3)
    dz, dy, dx = (float(voxel_size_um[0]), float(voxel_size_um[1]),
                  float(voxel_size_um[2]))
    xyz = np.empty_like(pts)
    xyz[:, 0] = pts[:, 2] * dx       # x
    xyz[:, 1] = pts[:, 1] * dy       # y
    xyz[:, 2] = pts[:, 0] * dz       # z
    return xyz


# ── tetrahedron geometry (alpha-shape core) ─────────────────────────────────────
def _tetra_geometry(tetra_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Circumradius and volume of each tetrahedron.

    ``tetra_pts`` is ``(M, 4, 3)``. Returns ``(circumradii, volumes)`` each length
    ``M``; degenerate (near-coplanar) tetrahedra get ``circumradius = inf`` and
    ``volume = 0`` so they are always dropped.
    """
    tetra_pts = np.asarray(tetra_pts, dtype=float)
    if tetra_pts.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    p0 = tetra_pts[:, 0]
    a = tetra_pts[:, 1] - p0
    b = tetra_pts[:, 2] - p0
    c = tetra_pts[:, 3] - p0
    bc = np.cross(b, c)
    ca = np.cross(c, a)
    ab = np.cross(a, b)
    denom = np.einsum("ij,ij->i", a, bc)          # a·(b×c) = 6·signed volume
    volumes = np.abs(denom) / 6.0
    a2 = np.einsum("ij,ij->i", a, a)
    b2 = np.einsum("ij,ij->i", b, b)
    c2 = np.einsum("ij,ij->i", c, c)
    num = a2[:, None] * bc + b2[:, None] * ca + c2[:, None] * ab
    circum = np.full(denom.shape[0], np.inf)
    safe = np.abs(denom) > _EPS
    circum[safe] = np.linalg.norm(num[safe], axis=1) / (2.0 * np.abs(denom[safe]))
    return circum, volumes


def tetra_circumradii(delaunay: Delaunay) -> np.ndarray:
    """Per-simplex circumradius of a ``scipy.spatial.Delaunay`` (index-aligned with
    ``delaunay.simplices`` / ``find_simplex``). Degenerate tetra return ``inf``.
    """
    pts = np.asarray(delaunay.points, dtype=float)
    circum, _ = _tetra_geometry(pts[delaunay.simplices])
    return circum


def _boundary_faces(kept_simplices: np.ndarray) -> np.ndarray:
    """Faces owned by exactly one surviving tetrahedron → the alpha-shape surface.

    ``kept_simplices`` is ``(K, 4)`` vertex indices. Returns ``(Nf, 3)`` int face
    indices (into the same point array the simplices index).
    """
    seen: Dict[Tuple[int, int, int], List[Any]] = {}
    for s in kept_simplices:
        v = (int(s[0]), int(s[1]), int(s[2]), int(s[3]))
        for combo in ((v[0], v[1], v[2]), (v[0], v[1], v[3]),
                      (v[0], v[2], v[3]), (v[1], v[2], v[3])):
            key = tuple(sorted(combo))
            entry = seen.get(key)
            if entry is None:
                seen[key] = [1, combo]
            else:
                entry[0] += 1
    faces = [entry[1] for entry in seen.values() if entry[0] == 1]
    if not faces:
        return np.zeros((0, 3), dtype=int)
    return np.asarray(faces, dtype=int).reshape(-1, 3)


# ── per-granule boundary builders ────────────────────────────────────────────────
def _alpha_shape_boundary(pts: np.ndarray, gid: int,
                          alpha: float) -> Optional[GranuleBoundary]:
    """Alpha-shape (concave hull) of one granule's points; ``None`` if unbuildable.

    ``pts`` is ``(n, 3)`` world ``(x, y, z)`` µm. Finite ``alpha`` drops tetra with
    circumradius above it; a non-finite / non-positive ``alpha`` uses the convex hull.
    """
    n = int(pts.shape[0])
    if n < _MIN_HULL_POINTS:
        return None

    use_convex = (not np.isfinite(alpha)) or (alpha <= 0.0)
    if not use_convex:
        try:
            tri = Delaunay(pts)
        except (QhullError, ValueError):
            tri = None
        if tri is not None:
            circum, volumes = _tetra_geometry(tri.points[tri.simplices])
            keep = np.isfinite(circum) & (circum <= alpha) & (volumes > 0.0)
            if keep.any():
                volume = float(volumes[keep].sum())
                faces = _boundary_faces(tri.simplices[keep])
                density = (n / volume) if volume > _EPS else 0.0
                return GranuleBoundary(
                    granule_id=int(gid), vertices_um=pts, faces=faces,
                    enclosed_volume_um3=volume, n_points=n, density=density,
                    delaunay=tri,
                )
            # alpha too small: no tetra survive → granule dissolves.
            return None
        # Delaunay failed → fall through to the convex-hull path.

    # Convex-hull path (alpha → ∞, or Delaunay unavailable).
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return None
    volume = float(hull.volume)
    faces = np.asarray(hull.simplices, dtype=int).reshape(-1, 3)
    try:
        tri2: Optional[Delaunay] = Delaunay(pts)
    except (QhullError, ValueError):
        tri2 = None
    density = (n / volume) if volume > _EPS else 0.0
    return GranuleBoundary(
        granule_id=int(gid), vertices_um=pts, faces=faces,
        enclosed_volume_um3=volume, n_points=n, density=density, delaunay=tri2,
    )


def _voronoi_cell_volumes(pts: np.ndarray) -> Tuple[Optional[Voronoi], np.ndarray]:
    """Global Voronoi of ``pts`` ``(N,3)`` µm → ``(vor, cell_volumes)``.

    ``cell_volumes[i]`` is the ``ConvexHull`` volume of point ``i``'s finite
    Voronoi-cell vertices, or ``nan`` for an unbounded (outer) cell.
    """
    n = int(pts.shape[0])
    vols = np.full(n, np.nan)
    if n < 5:
        return None, vols
    try:
        vor = Voronoi(pts)
    except (QhullError, ValueError):
        return None, vols
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        if len(region) == 0 or -1 in region:
            continue                              # unbounded outer cell → skip
        try:
            vols[i] = float(ConvexHull(vor.vertices[region]).volume)
        except (QhullError, ValueError):
            vols[i] = np.nan
    return vor, vols


def _voronoi_boundary(pts: np.ndarray, gid: int,
                      cell_vols: np.ndarray) -> Optional[GranuleBoundary]:
    """Union boundary + density for one granule in Voronoi mode; ``None`` if empty.

    The union boundary is a first-cut ``ConvexHull`` of the member points (the
    documented approximation to "outer faces of the union of member cells").
    """
    n = int(pts.shape[0])
    finite = np.isfinite(cell_vols) & (cell_vols > 0.0)
    volume = float(cell_vols[finite].sum()) if finite.any() else 0.0
    if n < _MIN_HULL_POINTS or volume <= _EPS:
        return None
    density = n / volume
    try:
        faces = np.asarray(ConvexHull(pts).simplices, dtype=int).reshape(-1, 3)
    except (QhullError, ValueError):
        faces = np.zeros((0, 3), dtype=int)
    return GranuleBoundary(
        granule_id=int(gid), vertices_um=pts, faces=faces,
        enclosed_volume_um3=volume, n_points=n, density=density, delaunay=None,
    )


# ── region adjacency ──────────────────────────────────────────────────────────
def _adjacency_alpha(pts_xyz: np.ndarray, labels: np.ndarray,
                     active: Set[int], adj_dist_um: float) -> Set[frozenset]:
    """Alpha-shape adjacency: centroid distance ``< adj_dist_um`` OR any cross-granule
    point pair within ``adj_dist_um`` (``cKDTree``, the app's idiom → touching hulls).
    """
    pairs: Set[frozenset] = set()
    ids = sorted(active)
    if len(ids) < 2:
        return pairs
    centroids = {g: pts_xyz[labels == g].mean(axis=0) for g in ids}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            gi, gj = ids[i], ids[j]
            if np.linalg.norm(centroids[gi] - centroids[gj]) < adj_dist_um:
                pairs.add(frozenset((gi, gj)))
    idx = np.where(np.isin(labels, ids))[0]
    if idx.size >= 2 and adj_dist_um > 0.0:
        tree = cKDTree(pts_xyz[idx])
        for a, b in tree.query_pairs(r=float(adj_dist_um)):
            la, lb = int(labels[idx[a]]), int(labels[idx[b]])
            if la != lb:
                pairs.add(frozenset((la, lb)))
    return pairs


def _adjacency_voronoi(labels: np.ndarray, ridge_points: np.ndarray,
                       active: Set[int]) -> Set[frozenset]:
    """Voronoi adjacency: a shared ridge between cells of two different granules."""
    pairs: Set[frozenset] = set()
    for p, q in ridge_points:
        lp, lq = int(labels[p]), int(labels[q])
        if lp != lq and lp in active and lq in active:
            pairs.add(frozenset((lp, lq)))
    return pairs


# ── iterative RAG density-merge ──────────────────────────────────────────────────
def _iterative_merge(
    labels: np.ndarray,
    build_fn: Callable[[int, np.ndarray], Optional[GranuleBoundary]],
    adjacency_fn: Callable[[np.ndarray, Set[int]], Set[frozenset]],
    merge_tol: float,
    merged_from: Dict[int, List[int]],
) -> Tuple[Dict[int, GranuleBoundary], np.ndarray, Dict[int, List[int]]]:
    """Drive the merge until no adjacent pair has similar-enough density.

    Mutates ``labels`` / ``merged_from`` in place and returns the final
    ``(boundaries, labels, merged_from)``. Union-find is implicit: on each merge the
    higher granule id is relabelled to the lower one.
    """
    active: Set[int] = {int(g) for g in np.unique(labels) if int(g) != NOISE_LABEL}

    while True:
        # Build every active granule's boundary; self-heal by dropping any that
        # fail to build (degenerate) to noise, then restart the pass cleanly.
        boundaries: Dict[int, GranuleBoundary] = {}
        dropped = False
        for gid in sorted(active):
            boundary = build_fn(gid, labels)
            if boundary is None:
                labels[labels == gid] = NOISE_LABEL
                merged_from.pop(gid, None)
                dropped = True
            else:
                boundaries[gid] = boundary
        if dropped:
            active = {int(g) for g in np.unique(labels) if int(g) != NOISE_LABEL}
            continue
        active = set(boundaries.keys())

        adjacency = adjacency_fn(labels, active)
        best: Optional[Tuple[float, int, int]] = None
        for pair in sorted(adjacency, key=lambda fs: tuple(sorted(fs))):
            gi, gj = sorted(pair)
            di, dj = boundaries[gi].density, boundaries[gj].density
            hi = max(di, dj)
            if not (np.isfinite(di) and np.isfinite(dj)) or hi <= _EPS:
                continue
            ratio = abs(di - dj) / hi
            if ratio <= merge_tol and (best is None or ratio < best[0]):
                best = (ratio, gi, gj)
        if best is None:
            return boundaries, labels, merged_from

        _, keep, drop = best
        labels[labels == drop] = keep
        merged_from[keep] = merged_from.get(keep, [keep]) + merged_from.pop(drop, [drop])
        active = {int(g) for g in np.unique(labels) if int(g) != NOISE_LABEL}


# ── public entry point ────────────────────────────────────────────────────────
def tessellate_granules(points_zyx: np.ndarray,
                        labels: np.ndarray,
                        voxel_size_um: Tuple[float, float, float],
                        params: Optional[Dict[str, Any]]) -> GranuleTessellation:
    """Tessellate a labelled point cloud into per-granule boundaries + merged labels.

    Parameters
    ----------
    points_zyx : (N, 3) float
        Bead centroids in ``(z, y, x)`` **voxel** order (P0 §5).
    labels : (N,) int
        *Initial* per-point granule id (P2 GMM output); ``-1`` = noise.
    voxel_size_um : (dz, dy, dx)
        Voxel size in µm.
    params : dict | None
        ``tess_mode`` (``"alpha_shape"``/``"voronoi"``), ``alpha`` (µm circumradius;
        alpha-shape only), ``merge_tol`` (0–1 density-ratio tolerance),
        ``adj_dist_um`` (alpha-shape adjacency), ``min_granule_points`` (drop tiny
        granules). Missing keys fall back to module defaults.

    Returns
    -------
    GranuleTessellation
        ``point_labels`` are the FINAL merged granule id per input point (index-aligned
        with ``points_zyx``); ``boundaries`` are the final per-granule boundaries with
        vertices in world ``(x, y, z)`` µm; ``merged_from`` maps each final id to the
        original GMM ids folded into it.
    """
    mode = str(_get(params, "tess_mode"))
    if mode not in (TESS_MODE_ALPHA, TESS_MODE_VORONOI):
        mode = TESS_MODE_ALPHA
    alpha = float(_get(params, "alpha"))
    merge_tol = float(_get(params, "merge_tol"))
    adj_dist_um = float(_get(params, "adj_dist_um"))
    min_pts = int(_get(params, "min_granule_points"))

    dz, dy, dx = (float(voxel_size_um[0]), float(voxel_size_um[1]),
                  float(voxel_size_um[2]))
    vox = (dz, dy, dx)

    pts_xyz = _points_to_xyz_um(points_zyx, vox)
    n = int(pts_xyz.shape[0])

    if n == 0:
        return GranuleTessellation(mode=mode, boundaries={},
                                   point_labels=np.zeros(0, dtype=int),
                                   voxel_size_um=vox, merged_from={})

    cur = np.asarray(labels, dtype=int).reshape(-1).copy()
    if cur.shape[0] != n:
        raise ValueError(
            f"labels length {cur.shape[0]} != number of points {n}")

    # Drop granules below the user's minimum-point threshold to noise up front.
    for gid in [int(g) for g in np.unique(cur) if int(g) != NOISE_LABEL]:
        if int((cur == gid).sum()) < min_pts:
            cur[cur == gid] = NOISE_LABEL

    merged_from: Dict[int, List[int]] = {
        int(g): [int(g)] for g in np.unique(cur) if int(g) != NOISE_LABEL
    }

    if mode == TESS_MODE_VORONOI:
        vor, cell_vols = _voronoi_cell_volumes(pts_xyz)
        if vor is None:
            # Degenerate cloud: nothing tessellable, return labels unchanged.
            return GranuleTessellation(mode=mode, boundaries={},
                                       point_labels=cur, voxel_size_um=vox,
                                       merged_from=merged_from)
        ridge_points = np.asarray(vor.ridge_points)

        def build_fn(gid: int, lab: np.ndarray) -> Optional[GranuleBoundary]:
            member = np.where(lab == gid)[0]
            return _voronoi_boundary(pts_xyz[member], gid, cell_vols[member])

        def adjacency_fn(lab: np.ndarray, active: Set[int]) -> Set[frozenset]:
            return _adjacency_voronoi(lab, ridge_points, active)
    else:
        def build_fn(gid: int, lab: np.ndarray) -> Optional[GranuleBoundary]:
            return _alpha_shape_boundary(pts_xyz[lab == gid], gid, alpha)

        def adjacency_fn(lab: np.ndarray, active: Set[int]) -> Set[frozenset]:
            return _adjacency_alpha(pts_xyz, lab, active, adj_dist_um)

    boundaries, cur, merged_from = _iterative_merge(
        cur, build_fn, adjacency_fn, merge_tol, merged_from)

    merged_from = {int(k): sorted(set(int(x) for x in v))
                   for k, v in merged_from.items()}

    return GranuleTessellation(mode=mode, boundaries=boundaries,
                               point_labels=cur, voxel_size_um=vox,
                               merged_from=merged_from)
