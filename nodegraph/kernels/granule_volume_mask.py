# ============================================================================
# granule_volume_mask.py — VENDORED, SELF-CONTAINED kernel for the
# "Granule Volume Mask" node.
#
# PURPOSE
#   Voxelize tessellated granule boundaries onto a confocal (Z,H,W) voxel grid,
#   producing a per-granule {label: (Z,H,W) bool} dict PLUS a combined
#   (Z,H,W) int32 label volume (0 = background). Pure numpy / scipy, Qt-free.
#
# WHERE THE REAL MATH LIVES
#   IN-REPO. The entire algorithm is defined below (point-in-Delaunay
#   rasterization, signed-distance-Gaussian non-shrinking smoothing, connected-
#   component speck removal, dense 1-based relabel, density/-id overlap
#   tie-break). Third-party heavy lifting is scipy only (spatial.Delaunay for
#   the inside test; ndimage EDT / gaussian_filter / label / binary_fill_holes).
#
# PROVENANCE (branch: Version-1.45)
#   Vendored VERBATIM from:
#     nd2studios/backend/analysis/granule_mask.py
#   Extraction blueprint (not vendored): tests/granule/test_granule_mask.py
#
# NOTE: vendored verbatim; imports nothing from nd2studios; caller owns all prep
#   (no file I/O, no per-multipoint/per-timepoint loop, no crop/downsample/
#   registration/exclusion — the caller supplies one already-prepared
#   tessellation + grid shape + voxel size).
#
# DROPPED nd2studios imports / members (only permitted edits applied):
#   (2a) Removed `from nd2studios.backend.analysis.granule_types import
#        (COMBINED_LABELS_KEY, GranuleBoundary, GranuleTessellation)`.
#        - COMBINED_LABELS_KEY was imported but UNUSED on the compute path
#          (referenced only in prose/tests) -> dropped, nothing else changed.
#        - GranuleBoundary / GranuleTessellation appear ONLY in type
#          annotations. Under `from __future__ import annotations` these are
#          never evaluated at runtime, so the granule_types dependency is a
#          type-hint-only dep and is dropped. `tess` and `boundary` are
#          DUCK-TYPED at runtime (getattr .boundaries / .vertices_um /
#          .delaunay / .density), so any object exposing those attributes works.
#   No private helper was renamed (no name collision: single source module).
#   No compute-path code was altered.
#
# CALLER CONVENIENCE
#   A ~10-line adapter `tessellation_from_list(...)` is provided at the BOTTOM
#   so callers can pass a plain list of (vertices_um, density, delaunay) tuples
#   instead of constructing a GranuleTessellation dataclass. It is NOT part of
#   the vendored math — it only builds a duck-typed stand-in object.
# ============================================================================
"""granule_mask — voxelize tessellated granule boundaries onto the confocal grid
(pure numpy / scipy, **Qt-free**).

Backs the V1.70 **Volume Mask** node (``special:granule_mask``), the
object-producing node of the granule-separation chain. It consumes a P3
:class:`~nd2studios.backend.analysis.granule_types.GranuleTessellation` (one
:class:`~nd2studios.backend.analysis.granule_types.GranuleBoundary` per granule)
and rasterizes each boundary into a dense ``(Z, H, W)`` boolean volume on the
same confocal grid as the raw stack, then applies an optional signed-distance
Gaussian smoothing so the surface reads as a *continuous, smooth* volume rather
than a stair-stepped hull.

Output mirrors ``record._mask3d_by_m[m][t]`` for a single object — a
``{granule_id: (Z, H, W) bool}`` dict — so the whole V1.65–V1.68 back half
(``object_scope.iter_objects`` → per-object DVC → ``viz3d`` surfaces) consumes it
unchanged. A companion combined ``(Z, H, W) int32`` label volume (``0`` =
background) is returned alongside for convenience (see the overlap tie-break
below).

Coordinate convention (obeyed exactly, matching ``granule_types`` / ``viz3d``):

* Grid dimensions ``shape_zhw = (Z, H, W)``; ``voxel_size_um = (dz, dy, dx)`` µm.
* A voxel center at index ``(z, y, x)`` maps to **world** ``(x*dx, y*dy, z*dz)`` µm
  (mesh vertices live in world ``(x, y, z)`` µm — the ``viz3d`` convention). The
  world origin of plane ``0`` is ``z0 = 0``; the plane a world-``z`` sample falls
  on is ``floor((z_um - z0) / dz)`` (P0 decision 8).
* Inside-test reuses ``GranuleBoundary.delaunay.find_simplex(centers) >= 0`` when
  present (P3's triangulation, cheap); a Voronoi-mode boundary without one gets a
  fresh ``scipy.spatial.Delaunay(vertices_um)`` built here (convex-hull inside
  test) — no re-triangulation cost is paid twice.

The SDF-Gaussian smoothing reuses the idiom of
:func:`nd2studios.backend.analysis.mask3d._signed_distance`
(``distance_transform_edt``): ``sdf = edt(mask) - edt(~mask)`` with physical
``sampling=(dz, dy, dx)``, Gaussian-blur ``sdf`` with a per-axis sigma (the
µm smoothing scale divided by each axis' voxel size), then re-threshold at
``sdf > 0``. Blurring the SDF rather than the binary mask rounds convex corners
without eroding volume — the standard non-shrinking morphological smoother.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# --- VENDOR EDIT (rule 2a): the original imported three symbols from
#     nd2studios.backend.analysis.granule_types:
#         COMBINED_LABELS_KEY, GranuleBoundary, GranuleTessellation
#     COMBINED_LABELS_KEY is unused on the compute path. GranuleBoundary and
#     GranuleTessellation are used ONLY in type annotations, which are never
#     evaluated at runtime (from __future__ import annotations). Both `tess`
#     and `boundary` are duck-typed at runtime. We alias the type names to Any
#     purely so static tooling / get_type_hints() do not raise on the strings.
GranuleBoundary = Any        # type-hint-only stand-in (duck-typed at runtime)
GranuleTessellation = Any    # type-hint-only stand-in (duck-typed at runtime)

# ── default parameters (P6 supplies the real values from the node's editor) ────
DEFAULT_SMOOTH_SIGMA_UM = 0.0        # 0 → smoothing off
DEFAULT_FILL_HOLES = False
DEFAULT_MIN_OBJECT_VOXELS = 1        # 1 → keep everything

# World-z of plane 0 (the (z,y,x)->(x*dx,y*dy,z*dz) mapping puts it at the origin).
_Z0_UM = 0.0


def _param(params: Any, key: str, default: Any) -> Any:
    """Read ``key`` from a dict-like *or* attribute-bearing ``params`` (or None)."""
    if params is None:
        return default
    if isinstance(params, dict):
        return params.get(key, default)
    return getattr(params, key, default)


def _build_delaunay(points_xyz: np.ndarray) -> Optional[Any]:
    """Build a ``scipy.spatial.Delaunay`` over world ``(x, y, z)`` vertices.

    Returns ``None`` for a degenerate / coplanar cloud (``QhullError``) so the
    caller can skip that granule instead of crashing (P0 §7).
    """
    from scipy.spatial import Delaunay, QhullError

    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 4:
        return None
    try:
        return Delaunay(pts)
    except (QhullError, ValueError):
        return None


def _voxelize_boundary(boundary: GranuleBoundary,
                       shape_zhw: Tuple[int, int, int],
                       voxel_size_um: Tuple[float, float, float]) -> np.ndarray:
    """Rasterize one boundary to a ``(Z, H, W)`` bool volume via a point-in-hull test.

    Only voxels inside the boundary vertices' world bounding box are tested (mapped
    back to voxel indices with the ``floor`` rule); each z-plane in that box is
    tested in one ``find_simplex`` call, so peak memory stays ~one plane of points.
    """
    Z, H, W = int(shape_zhw[0]), int(shape_zhw[1]), int(shape_zhw[2])
    dz, dy, dx = (float(voxel_size_um[0]), float(voxel_size_um[1]),
                  float(voxel_size_um[2]))
    mask = np.zeros((Z, H, W), dtype=bool)
    if Z <= 0 or H <= 0 or W <= 0 or dz <= 0 or dy <= 0 or dx <= 0:
        return mask

    verts = np.asarray(boundary.vertices_um, dtype=float).reshape(-1, 3)
    if verts.shape[0] == 0:
        return mask

    tri = boundary.delaunay
    if tri is None:
        tri = _build_delaunay(verts)
    if tri is None:
        return mask   # degenerate boundary — nothing to voxelize

    # Vertex world bbox (columns are X, Y, Z) → voxel-index bbox via the floor
    # rule, padded by one voxel and clamped to the grid.
    x_um, y_um, z_um = verts[:, 0], verts[:, 1], verts[:, 2]
    xi0 = max(0, int(np.floor(x_um.min() / dx)) - 1)
    xi1 = min(W, int(np.floor(x_um.max() / dx)) + 2)
    yi0 = max(0, int(np.floor(y_um.min() / dy)) - 1)
    yi1 = min(H, int(np.floor(y_um.max() / dy)) + 2)
    zi0 = max(0, int(np.floor((z_um.min() - _Z0_UM) / dz)) - 1)
    zi1 = min(Z, int(np.floor((z_um.max() - _Z0_UM) / dz)) + 2)
    if xi1 <= xi0 or yi1 <= yi0 or zi1 <= zi0:
        return mask

    ys = np.arange(yi0, yi1)
    xs = np.arange(xi0, xi1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    xw = xx.ravel() * dx          # world X for this plane's grid
    yw = yy.ravel() * dy          # world Y
    n_plane = xw.size
    for zi in range(zi0, zi1):
        zw = float(zi) * dz + _Z0_UM
        centers = np.empty((n_plane, 3), dtype=float)
        centers[:, 0] = xw
        centers[:, 1] = yw
        centers[:, 2] = zw
        inside = tri.find_simplex(centers) >= 0
        if inside.any():
            mask[zi, yi0:yi1, xi0:xi1] = inside.reshape(yy.shape)
    return mask


def _sdf_smooth(mask: np.ndarray,
                voxel_size_um: Tuple[float, float, float],
                smooth_sigma_um: float) -> np.ndarray:
    """Signed-distance Gaussian smoothing (non-shrinking corner rounding).

    ``sdf = edt(mask) - edt(~mask)`` in physical µm (``sampling=(dz, dy, dx)``),
    blurred with a per-axis sigma of ``smooth_sigma_um`` converted to voxels
    (``sigma/dz, sigma/dy, sigma/dx``), then re-thresholded at ``sdf > 0``. Fully
    empty / full masks are returned unchanged.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    dz, dy, dx = (float(voxel_size_um[0]), float(voxel_size_um[1]),
                  float(voxel_size_um[2]))
    m = np.asarray(mask, dtype=bool)
    if smooth_sigma_um <= 0 or not m.any() or m.all():
        return m
    sampling = (dz, dy, dx)
    inside = distance_transform_edt(m, sampling=sampling)
    outside = distance_transform_edt(~m, sampling=sampling)
    sdf = inside - outside
    sigma_vox = (smooth_sigma_um / dz if dz > 0 else 0.0,
                 smooth_sigma_um / dy if dy > 0 else 0.0,
                 smooth_sigma_um / dx if dx > 0 else 0.0)
    sdf = gaussian_filter(sdf, sigma=sigma_vox)
    return sdf > 0.0


def _drop_small_components(mask: np.ndarray, min_object_voxels: int) -> np.ndarray:
    """Remove 26-connected components smaller than ``min_object_voxels`` voxels."""
    if min_object_voxels <= 1 or not mask.any():
        return mask
    from scipy.ndimage import label

    structure = np.ones((3, 3, 3), dtype=bool)   # 26-connectivity
    labeled, n = label(mask, structure=structure)
    if n == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    keep = np.zeros(counts.shape[0], dtype=bool)
    keep[1:] = counts[1:] >= int(min_object_voxels)   # index 0 is background
    return keep[labeled]


def build_granule_masks(
    tess: GranuleTessellation,
    shape_zhw: Tuple[int, int, int],
    voxel_size_um: Tuple[float, float, float],
    params: Any,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Voxelize every granule boundary onto the confocal grid.

    Returns ``({label: (Z, H, W) bool}, combined (Z, H, W) int32)`` where ``label``
    is a **dense 1-based** granule label (``0`` = background, reserved) shared by the
    per-granule dict and the combined volume. The per-granule dict has the same value
    type as a single ``record._mask3d_by_m[m][t]`` object, so
    ``object_scope.iter_objects`` / the scope lever / DVC-on-object read it with no
    special-casing.

    Parameters (from ``params``, a dict or attribute object; missing → defaults):

    * ``smooth_sigma`` (float µm, default ``0`` = off) — SDF-Gaussian smoothing
      scale; see :func:`_sdf_smooth`. Non-shrinking corner rounding.
    * ``fill_holes`` (bool, default ``False``) — ``scipy.ndimage.binary_fill_holes``
      on each granule volume.
    * ``min_object_voxels`` (int, default ``1``) — drop connected components smaller
      than this (speck removal). A granule that ends up empty is omitted from the
      returned dict (and paints nothing into the combined volume).

    **Overlap tie-break** (combined int32): where two granules claim the same voxel,
    the **higher-density** granule wins; on equal density, the **lower granule id**
    wins. Implemented by painting granules in ascending ``(density, -id)`` order so
    the most-preferred one is written last and overwrites. The per-granule bool
    masks are left untouched by this (they may overlap); only the combined label
    volume is disambiguated.
    """
    from scipy.ndimage import binary_fill_holes

    Z, H, W = int(shape_zhw[0]), int(shape_zhw[1]), int(shape_zhw[2])
    voxel_size_um = (float(voxel_size_um[0]), float(voxel_size_um[1]),
                     float(voxel_size_um[2]))
    smooth_sigma = float(_param(params, "smooth_sigma", DEFAULT_SMOOTH_SIGMA_UM))
    fill_holes = bool(_param(params, "fill_holes", DEFAULT_FILL_HOLES))
    min_object_voxels = int(_param(params, "min_object_voxels",
                                   DEFAULT_MIN_OBJECT_VOXELS))

    masks_raw: Dict[int, np.ndarray] = {}
    boundaries = getattr(tess, "boundaries", {}) or {}
    for gid, boundary in boundaries.items():
        vol = _voxelize_boundary(boundary, (Z, H, W), voxel_size_um)
        if smooth_sigma > 0:
            vol = _sdf_smooth(vol, voxel_size_um, smooth_sigma)
        if fill_holes and vol.any():
            vol = binary_fill_holes(vol)
        vol = _drop_small_components(vol, min_object_voxels)
        if vol.any():
            masks_raw[int(gid)] = np.ascontiguousarray(vol)

    # Re-key to DENSE 1-BASED labels. ``0`` is reserved for background in the
    # combined volume, but granule ids from the GMM can legitimately be ``0`` —
    # which would then be indistinguishable from background (granule 0 would
    # vanish). Mapping the surviving ids to ``1..K`` (in ascending id order) fixes
    # that; for already-1-based ids it is the identity map. The per-granule dict and
    # the combined volume share the SAME labels, so downstream (P5 boundary's
    # ``combined != gid`` self-exclusion, DVC-on-object) stay in agreement.
    relabel = {gid: i + 1 for i, gid in enumerate(sorted(masks_raw.keys()))}
    masks: Dict[int, np.ndarray] = {relabel[gid]: v for gid, v in masks_raw.items()}

    # Combined int32 label volume with the documented overlap tie-break.
    combined = np.zeros((Z, H, W), dtype=np.int32)

    def _preference(gid: int) -> Tuple[float, int]:
        density = float(getattr(boundaries.get(gid), "density", 0.0) or 0.0)
        return (density, -int(gid))   # ascending → last painted = winner

    for gid in sorted(masks_raw.keys(), key=_preference):
        combined[masks_raw[gid]] = int(relabel[gid])

    return masks, combined


# ============================================================================
# CALLER-CONVENIENCE ADAPTER (NOT vendored math — builds a duck-typed object)
# ============================================================================
class _SimpleBoundary:
    """Duck-typed stand-in for a GranuleBoundary (only the attributes the
    kernel reads: ``vertices_um``, ``density``, ``delaunay``)."""

    __slots__ = ("granule_id", "vertices_um", "density", "delaunay")

    def __init__(self, granule_id, vertices_um, density=0.0, delaunay=None):
        self.granule_id = int(granule_id)
        self.vertices_um = np.asarray(vertices_um, dtype=float).reshape(-1, 3)
        self.density = float(density)
        self.delaunay = delaunay   # optional prebuilt scipy.spatial.Delaunay


class _SimpleTessellation:
    """Duck-typed stand-in for a GranuleTessellation (only ``boundaries``)."""

    __slots__ = ("boundaries",)

    def __init__(self, boundaries):
        self.boundaries = dict(boundaries)


def tessellation_from_list(granules):
    """Build a duck-typed tessellation from a plain list, for callers that do
    not want to construct the app's GranuleTessellation dataclass.

    ``granules`` is an iterable of tuples, each one of:
        (vertices_um,)                        -> density 0.0, no prebuilt tri
        (vertices_um, density)                -> no prebuilt tri (convex hull)
        (vertices_um, density, delaunay)      -> reuse prebuilt scipy.Delaunay
    where ``vertices_um`` is an (Nv, 3) array of world (x, y, z) µm coordinates.

    Granule ids are assigned by position (0, 1, 2, ...); the kernel then
    relabels survivors to dense 1-based ids, so the ids you get back in the
    returned dict are 1..K regardless.
    """
    boundaries = {}
    for gid, g in enumerate(granules):
        g = tuple(g)
        verts = g[0]
        density = g[1] if len(g) > 1 else 0.0
        delaunay = g[2] if len(g) > 2 else None
        boundaries[gid] = _SimpleBoundary(gid, verts, density, delaunay)
    return _SimpleTessellation(boundaries)
