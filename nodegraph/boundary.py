"""Constructive Point↔Label spine — Fill / Extract Boundary (nodegraph v2, Phase 2).

The primary Point↔Label relationship is **constructive** (V2.00 §3.2, §16): an ordered
point set is a **boundary** — a contour (2D) or surface (3D) — that **fills to a Label
region** (:func:`fill_boundary`), and a Label's outline **extracts back to points**
(:func:`extract_boundary`). This is the granule pipeline.

Per V2.03 §4 C4 (the resolved conflicts-locked hole), these are **explicit data-layer
operations, NOT auto-routed transfer bridges** — they are deliberately absent from
``transfer._BRIDGES`` so a constructive fill never becomes an implicit routing edge
(locked decision 6). The **dimensionality authority is the input geometry**: a point
set on a single z fills as a planar contour on that plane (never extruded — e.g. a 2D
ROI drawn on one slice of a z>1 volume); points spanning multiple z fill per-plane. The
header 2D/3D toggle is a **consistency assertion only** — it hard-errors on contradiction
with the derived geometry, it never overrides it.

Attribute *transfer* between an independent point set and a mask still uses containment
(:func:`nodegraph.bridges.containing_label` / :func:`~nodegraph.bridges.points_in_label`),
which is unchanged. skimage (draw / measure) is **lazily imported**; the core stays
numpy-only. Qt-free.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from nodegraph.structure import StructureTable, point_table


def boundary_dim(points: np.ndarray) -> str:
    """The dimensionality implied by a point set's geometry (V2.03 §4 C4): ``"2D"``
    for ``(N,2)`` points or ``(N,3)`` points sharing one z-plane (a planar contour);
    ``"3D"`` for points spanning multiple z (a surface)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"points must be (N,2) or (N,3), got {pts.shape}")
    if pts.shape[1] == 2:
        return "2D"
    return "2D" if np.unique(np.round(pts[:, 0]).astype(np.int64)).size <= 1 else "3D"


def _assert_dim(dim: Optional[str], derived: str) -> None:
    if dim is not None and dim != derived:
        raise ValueError(
            f"2D/3D toggle {dim!r} contradicts the input geometry ({derived}); the "
            f"toggle is a consistency assertion, not an override (V2.03 §4 C4)")


def fill_boundary(points: np.ndarray, shape: Tuple[int, ...], *, label_id: int = 1,
                  dim: Optional[str] = None) -> np.ndarray:
    """Fill a boundary point set into a Label raster of ``shape`` (``label_id`` inside,
    0 outside). ``(N,2)`` / single-z ``(N,3)`` points fill a planar polygon on their
    plane; multi-z ``(N,3)`` points fill each z-plane's polygon (a per-slice surface).
    ``dim`` (if given) must match the derived geometry (V2.03 §4 C4)."""
    from skimage.draw import polygon
    pts = np.asarray(points, dtype=float)
    derived = boundary_dim(pts)
    _assert_dim(dim, derived)
    raster = np.zeros(shape, dtype=np.int64)
    if pts.shape[1] == 2:
        if len(shape) != 2:
            raise ValueError(f"2D points need a 2D shape, got {shape}")
        rr, cc = polygon(pts[:, 0], pts[:, 1], shape)
        raster[rr, cc] = label_id
        return raster
    # (N,3): fill each z-plane's contour (single-z → one plane; the 2D-ROI-on-3D case)
    if len(shape) != 3:
        raise ValueError(f"3D points need a 3D shape, got {shape}")
    zr = np.round(pts[:, 0]).astype(np.int64)
    for z in np.unique(zr):
        if not (0 <= z < shape[0]):
            continue
        sel = zr == z
        rr, cc = polygon(pts[sel, 1], pts[sel, 2], shape[1:])
        raster[z, rr, cc] = label_id
    return raster


def extract_boundary(label_raster: np.ndarray, *, level: float = 0.5,
                     dim: Optional[str] = None, layer: Optional[str] = None
                     ) -> StructureTable:
    """Extract a Label region's outline as boundary Points: a 2D raster → ordered
    contour points (``skimage.measure.find_contours``, ``z_kind="plane_index"``, with a
    ``contour_id`` column); a 3D raster → surface vertices (``marching_cubes``,
    ``z_kind="subpixel"``). Inverts :func:`fill_boundary`. ``dim`` asserts consistency."""
    r = np.asarray(label_raster)
    if r.ndim == 2:
        _assert_dim(dim, "2D")
        from skimage.measure import find_contours
        contours = find_contours(r.astype(float), level)
        if not contours:
            return point_table(np.zeros((0, 2)), z_kind="plane_index", layer=layer)
        allpts = np.vstack(contours)                                   # (N,2) y,x
        cids = np.concatenate([np.full(len(c), i, dtype=np.int64)
                               for i, c in enumerate(contours)])
        tbl = point_table(allpts, z_kind="plane_index", layer=layer)
        return replace(tbl, columns={**tbl.columns, "contour_id": cids})
    if r.ndim == 3:
        _assert_dim(dim, "3D")
        if not (r.min() < level < r.max()):                           # no surface present
            return point_table(np.zeros((0, 3)), z_kind="subpixel", layer=layer)
        from skimage.measure import marching_cubes
        # ``_faces`` is discarded because this function's contract is a Point table, and
        # keeping it here would duplicate a better node rather than fill a gap: this call
        # meshes the UNION of every non-zero label as one surface (a label raster at level
        # 0.5 has no crossing at a 1|2 interface, so two touching regions yield a single
        # shell), while ``analysis.tessellate``'s ``label_surface`` mode marching-cubes each
        # region separately and keeps its faces. label_surface is the MESH route for a label
        # raster; a union iso-surface's only MESH consumer, ``transform.rasterize_mesh``,
        # would hand back the filled union mask you can get from ``labels > 0`` directly.
        verts, _faces, _normals, _values = marching_cubes(r.astype(float), level)
        return point_table(verts, z_kind="subpixel", layer=layer)     # (N,3) z,y,x
    raise ValueError(f"label_raster must be 2D or 3D, got {r.ndim}D")


__all__ = ["boundary_dim", "fill_boundary", "extract_boundary"]
