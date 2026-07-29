"""Mesh → voxel rasterization (nodegraph v2, V2.08).

Two interior tests, one code path. Both go through the **vendored**
:func:`nodegraph.kernels.granule_volume_mask.build_granule_masks`, which is duck-typed on
``.boundaries`` / ``.vertices_um`` / ``.delaunay`` / ``.density`` and whose whole inside
test is ``tri.find_simplex(centers) >= 0``. So a one-method shim exposing ``find_simplex``
drops straight in and inherits the vendored bbox walk, the µm smoothing, hole filling and
small-component removal **with zero kernel edits** — and, critically, without re-deriving
the kernel's index rule (voxel ``(z,y,x)`` has world centre ``(x*dx, y*dy, z*dz)``, no
half-voxel offset), which is the single easiest place to ship a half-voxel shift.

* ``mode="convex"`` — pass ``delaunay=None`` and let the kernel build its own
  ``Delaunay(verts)``. This is **bit-for-bit what the fused node already did**: the
  alpha-shape producer stores the *unfiltered* ``Delaunay``, and the union of a Delaunay
  triangulation's simplices is exactly the convex hull of its points, so
  ``find_simplex >= 0`` has always been a convex-hull membership test in every mode.
* ``mode="watertight"`` — pass :class:`FaceParity`, an even-odd ray-cast over the mesh's
  **faces**. This is the only test that respects concavity, so it is what finally makes
  the voxel mask agree with the analytic ``volume_um3`` / ``surface_area_um2`` the
  tessellation already reports. Even-odd parity is **orientation-independent**, which is
  required here: the vendored producers key ``_boundary_faces`` on a sorted vertex triple
  but store the raw unsorted one, and ``ConvexHull.equations`` is discarded, so winding is
  arbitrary — a signed / generalized winding number would silently mis-fill without an
  orientation-repair pass first.

Vertices arrive in **voxel ``(z,y,x)``** (the Mesh domain's storage convention), so the
parity test runs entirely in voxel space with no unit conversion at all. Only the convex
path converts to the kernel's world ``(x,y,z)`` µm.

numpy for the parity test; scipy only via the vendored kernel (lazily imported there).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# Exact hits are the COMMON case here, not a corner case: mesh vertices routinely sit on
# integer grid lines (marching cubes emits them there, and a synthetic axis-aligned test
# mesh is all of them), while the sample points ARE the integer lattice. Two failure modes
# follow, and both need the ray origin nudged off the lattice:
#   * in (z,y): a ray through a shared edge/vertex is counted by BOTH triangles, which
#     flips parity for the whole rest of the row;
#   * along x: a sample lying exactly ON a face is decided by float noise — barycentric
#     interpolation of a face at x=6 returns 6.000000000000001, so a strict comparison
#     silently reverses.
# Three mutually incommensurate offsets make every one of those unreachable. The cost is
# sampling ~1e-4 voxel off centre, and the resulting convention is HALF-OPEN at surfaces
# (a voxel centre exactly on the boundary reads as outside). That is inherent to parity —
# it cannot include both the near and far face of the same solid — and differs from the
# convex path's closed ``find_simplex >= 0`` by at most one voxel layer, on faces that lie
# exactly on the sampling lattice.
_EPS_Z = 1.0e-4
_EPS_Y = 1.7e-4
_EPS_X = 2.3e-4

# |denominator| below this means the triangle projects edge-on to the (z,y) plane, i.e.
# it is parallel to the +x ray. Such a triangle contributes zero-measure crossings and is
# skipped — dropping it is what keeps parity correct, not an approximation.
_DEGENERATE = 1.0e-12


class FaceParity:
    """Even-odd ray-cast point-in-mesh test, shaped as a ``find_simplex`` drop-in.

    Constructed over voxel-space vertices + faces; ``find_simplex`` accepts the vendored
    kernel's world ``(x, y, z)`` µm sample points, converts them back to voxels, and
    returns ``0`` where inside / ``-1`` where outside (the kernel only ever tests
    ``>= 0``).

    Rays travel along **+x**, so the work is grouped per ``(z, y)`` ray: one pass over the
    faces per distinct ``z`` to pick the candidates whose ``z``-span the plane crosses,
    then a small vectorized barycentric test per ``y``. That matches how the vendored
    voxelizer calls in — one plane at a time — so the candidate pass is amortized.
    """

    __slots__ = ("_v", "_f", "_vox", "_a", "_b", "_c", "_den", "_zmin", "_zmax",
                 "_ymin", "_ymax")

    def __init__(self, verts_zyx: np.ndarray, faces: np.ndarray,
                 voxel_size_um: Sequence[float]) -> None:
        v = np.asarray(verts_zyx, dtype=float).reshape(-1, 3)
        f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        self._v = v
        self._f = f
        self._vox = tuple(float(s) for s in voxel_size_um)
        if len(f) == 0 or len(v) == 0:
            self._a = self._b = self._c = np.zeros((0, 3))
            self._den = self._zmin = self._zmax = self._ymin = self._ymax = np.zeros(0)
            return
        self._a, self._b, self._c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        az, ay = self._a[:, 0], self._a[:, 1]
        bz, by = self._b[:, 0], self._b[:, 1]
        cz, cy = self._c[:, 0], self._c[:, 1]
        # barycentric denominator in the (z,y) projection (u=z, v=y)
        self._den = (by - cy) * (az - cz) + (cz - bz) * (ay - cy)
        tri_z = np.stack([az, bz, cz], axis=1)
        tri_y = np.stack([ay, by, cy], axis=1)
        self._zmin, self._zmax = tri_z.min(axis=1), tri_z.max(axis=1)
        self._ymin, self._ymax = tri_y.min(axis=1), tri_y.max(axis=1)

    # ── the ``Delaunay``-compatible surface ────────────────────────────────────────
    def find_simplex(self, centers_xyz: np.ndarray) -> np.ndarray:
        """``0`` inside / ``-1`` outside for each world ``(x,y,z)`` µm sample point."""
        pts = np.asarray(centers_xyz, dtype=float).reshape(-1, 3)
        dz, dy, dx = self._vox
        q = np.column_stack([pts[:, 2] / dz, pts[:, 1] / dy, pts[:, 0] / dx])
        return np.where(self.contains(q), 0, -1).astype(np.int64)

    def contains(self, query_zyx: np.ndarray) -> np.ndarray:
        """Boolean inside-mask for voxel-space ``(N,3)`` ``(z,y,x)`` query points."""
        q = np.asarray(query_zyx, dtype=float).reshape(-1, 3)
        out = np.zeros(len(q), dtype=bool)
        if len(self._f) == 0 or len(q) == 0:
            return out
        qz = q[:, 0] + _EPS_Z
        qy = q[:, 1] + _EPS_Y
        qx = q[:, 2] + _EPS_X
        good = np.abs(self._den) > _DEGENERATE

        for z in np.unique(qz):
            at_z = qz == z
            cand = good & (self._zmin <= z) & (self._zmax >= z)
            if not cand.any():
                continue
            ci = np.flatnonzero(cand)
            az, ay, ax = self._a[ci, 0], self._a[ci, 1], self._a[ci, 2]
            by_, bx = self._b[ci, 1], self._b[ci, 2]
            bz = self._b[ci, 0]
            cz, cy, cx = self._c[ci, 0], self._c[ci, 1], self._c[ci, 2]
            den = self._den[ci]
            ymin, ymax = self._ymin[ci], self._ymax[ci]

            for y in np.unique(qy[at_z]):
                sel = at_z & (qy == y)
                keep = (ymin <= y) & (ymax >= y)
                if not keep.any():
                    continue
                l1 = ((by_[keep] - cy[keep]) * (z - cz[keep])
                      + (cz[keep] - bz[keep]) * (y - cy[keep])) / den[keep]
                l2 = ((cy[keep] - ay[keep]) * (z - cz[keep])
                      + (az[keep] - cz[keep]) * (y - cy[keep])) / den[keep]
                l3 = 1.0 - l1 - l2
                hit = (l1 >= 0.0) & (l2 >= 0.0) & (l3 >= 0.0)
                if not hit.any():
                    continue
                xc = np.sort(l1[hit] * ax[keep][hit] + l2[hit] * bx[keep][hit]
                             + l3[hit] * cx[keep][hit])
                # a +x ray is inside iff an ODD number of crossings lie ahead of it
                ahead = xc.size - np.searchsorted(xc, qx[sel], side="right")
                out[sel] = (ahead % 2) == 1
        return out


def mesh_to_verts_um(verts_zyx: np.ndarray,
                     voxel_size_um: Sequence[float]) -> np.ndarray:
    """Voxel ``(z,y,x)`` → the vendored kernel's world ``(x,y,z)`` µm, matching
    ``granule_tessellate._points_to_xyz_um`` exactly (``_Z0_UM = 0``, no half-voxel)."""
    v = np.asarray(verts_zyx, dtype=float).reshape(-1, 3)
    dz, dy, dx = (float(s) for s in voxel_size_um)
    return np.column_stack([v[:, 2] * dx, v[:, 1] * dy, v[:, 0] * dz])


def verts_um_to_zyx(verts_um: np.ndarray,
                    voxel_size_um: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`mesh_to_verts_um` — world ``(x,y,z)`` µm → voxel ``(z,y,x)``."""
    v = np.asarray(verts_um, dtype=float).reshape(-1, 3)
    dz, dy, dx = (float(s) for s in voxel_size_um)
    return np.column_stack([v[:, 2] / dz, v[:, 1] / dy, v[:, 0] / dx])


def voxelize_mesh(verts_zyx: np.ndarray, faces: np.ndarray,
                  shape_zyx: Tuple[int, int, int],
                  voxel_size_um: Sequence[float], *,
                  mode: str = "convex",
                  density: float = 0.0,
                  smooth_um: float = 0.0,
                  fill_holes: bool = False,
                  min_voxels: int = 1) -> np.ndarray:
    """Rasterize ONE mesh element to a ``(Z,Y,X)`` bool volume.

    ``mode="convex"`` reproduces the pre-split behaviour exactly (kernel-built Delaunay =
    convex hull of the vertices); ``mode="watertight"`` uses :class:`FaceParity` so
    concavity survives. Post-processing (µm SDF smoothing → hole fill → small-component
    removal) is the vendored kernel's, identical for both modes.

    Returns an all-``False`` volume for a mesh the interior test cannot resolve (fewer
    than 4 vertices, a Qhull failure in convex mode, or no faces in watertight mode).
    Callers should treat an empty result as "this element did not rasterize" — the
    vendored kernel drops such elements silently, so the node layer reports it instead.
    """
    from nodegraph.kernels.granule_volume_mask import (build_granule_masks,
                                                       tessellation_from_list)
    Z, Y, X = (int(s) for s in shape_zyx)
    verts_um = mesh_to_verts_um(verts_zyx, voxel_size_um)
    if mode == "watertight":
        tri: Optional[Any] = FaceParity(verts_zyx, faces, voxel_size_um)
        if len(np.asarray(faces).reshape(-1, 3)) == 0:
            return np.zeros((Z, Y, X), dtype=bool)
    elif mode == "convex":
        tri = None                       # the kernel builds Delaunay(verts) itself
    else:
        raise ValueError(f"unknown interior test {mode!r} (expected convex|watertight)")

    # NOTE the tuple: ``tessellation_from_list`` does ``tuple(g)`` per entry, so a bare
    # ndarray would be tuple-ified into its ROWS and silently become a 1-vertex mesh.
    tess = tessellation_from_list([(verts_um, float(density), tri)])
    masks, _combined = build_granule_masks(
        tess, (Z, Y, X), tuple(float(s) for s in voxel_size_um),
        {"smooth_sigma": max(0.0, float(smooth_um)),
         "fill_holes": bool(fill_holes),
         "min_object_voxels": max(1, int(min_voxels))})
    if not masks:                        # empty after voxelize/smooth/fill/speck-removal
        return np.zeros((Z, Y, X), dtype=bool)
    return np.asarray(masks[1], dtype=bool)


def interior_test_for(provenance: Dict[str, Any], closed: bool) -> str:
    """Pick the interior test for one element from the mesh's stamped provenance
    (``wire-node-v2`` §7b — derive, don't ask), falling back to the mesh's own derived
    ``closed`` topology when a hand-built mesh carries no provenance.

    ``convex_hull`` keeps the cheap, exactly-correct ``find_simplex`` path. Anything that
    can be concave (``alpha_shape`` with a finite alpha, ``label_surface``) uses the
    faces-based parity test — but only if the surface is actually watertight, since parity
    on an open surface is meaningless.
    """
    boundary = str(provenance.get("boundary", "")) or None
    if boundary == "convex_hull":
        return "convex"
    if boundary in ("alpha_shape", "label_surface"):
        return "watertight" if closed else "convex"
    return "watertight" if closed else "convex"


__all__ = ["FaceParity", "voxelize_mesh", "interior_test_for",
           "mesh_to_verts_um", "verts_um_to_zyx"]
