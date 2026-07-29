# Granule Tessellation — Integration Contract

## 1. Purpose & where the real math lives

Turns a labelled 3-D point cloud of bead centroids into **per-granule boundary
meshes** plus a **FINAL (post-merge) per-point granule label**. Each granule's
points become a boundary (alpha-shape / concave hull, or Voronoi-cell union), then
an iterative region-adjacency **density-merge** folds adjacent granules of
similar-enough density together (union-find, higher id → lower id).

**Real math is in-repo, in `pure_analysis/granule_tessellate.py`.** It is plain
`scipy.spatial` (`ConvexHull`, `Delaunay`, `Voronoi`, `cKDTree`) plus hand-written
tetrahedron circumradius/volume and boundary-face extraction. **No external analysis
package, no learned model.** Only `numpy` + `scipy` do the work.

## 2. Entry point

```python
tessellate_granules(
    points_zyx: np.ndarray,                       # (N, 3) float, (z, y, x) VOXELS
    labels: np.ndarray,                           # (N,) int, initial per-point label (-1 = noise)
    voxel_size_um: Tuple[float, float, float],    # (dz, dy, dx) micrometers
    params: Optional[Dict[str, Any]],             # see Parameters; None => all defaults
) -> GranuleTessellation
```

Supporting dataclasses (also defined in the module, importable):
`GranuleBoundary`, `GranuleTessellation`. Constant: `NOISE_LABEL = -1`.
Public helpers on the compute path but callable standalone: `tetra_circumradii(delaunay)`.

## 3. Inputs

| name | type | shape | dtype | axis order | units | required? / default | meaning & constraints |
|------|------|-------|-------|-----------|-------|---------------------|-----------------------|
| `points_zyx` | `np.ndarray` | `(N, 3)` | float (cast to `float`) | **(z, y, x)** | **VOXELS** | required | Bead centroids. Reshaped to `(-1, 3)`. `N == 0` → empty result. |
| `labels` | `np.ndarray` | `(N,)` | int (cast to `int`) | — | — | required | Initial granule id per point; index-aligned with `points_zyx`. `-1` = noise (excluded from tessellation). Length MUST equal `N` or `ValueError`. |
| `voxel_size_um` | tuple/seq of 3 floats | `(3,)` | float | **(dz, dy, dx)** | µm | required | Voxel size. Drives the voxel→micron scaling AND the (z,y,x)→(x,y,z) flip. |
| `params` | `dict` or `None` | — | — | — | — | optional; `None` → all defaults | See table below. Per-key: missing OR `None` value → module default. |

## 4. Parameters (`params` dict)

| name | type | default | valid range / choices | semantics |
|------|------|---------|-----------------------|-----------|
| `tess_mode` | str | `"alpha_shape"` | `"alpha_shape"` \| `"voronoi"` | Boundary construction mode. Unrecognized value silently coerced to `"alpha_shape"`. |
| `alpha` | float | `inf` | any float | Alpha-shape circumradius threshold in **µm**: tetrahedra with circumradius > `alpha` are dropped (concave hull). `alpha <= 0` OR non-finite (`inf`/`nan`) → **ConvexHull fallback** (there is NO `"auto"`). Ignored in voronoi mode. |
| `merge_tol` | float | `0.15` | `[0, 1]` | Density-ratio tolerance. A pair merges only if `abs(dᵢ−dⱼ)/max(dᵢ,dⱼ) <= merge_tol`. `0` ⇒ effectively no merging; `1` ⇒ any adjacent pair may merge. |
| `adj_dist_um` | float | `5.0` | `>= 0` µm | Alpha-shape adjacency distance. Two granules are adjacent if centroid distance `< adj_dist_um` OR any cross-granule point pair is within `adj_dist_um` (`cKDTree`). **`adj_dist_um == 0` ⇒ proximity adjacency effectively OFF** (centroid test `dist < 0` never fires, and the `cKDTree` branch is gated by `> 0.0` so it is skipped). Ignored in voronoi mode. |
| `min_granule_points` | int | `4` | `>= 1` | Granules with fewer points are dropped to noise up front, before tessellation. |

## 5. Output — `GranuleTessellation`

| field | type | shape | dtype | axis order | units | meaning |
|-------|------|-------|-------|-----------|-------|---------|
| `mode` | str | — | — | — | — | The resolved mode (`"alpha_shape"` / `"voronoi"`). |
| `boundaries` | `Dict[int, GranuleBoundary]` | — | — | — | — | Final granule id → boundary. Keys are the surviving (post-merge) ids. |
| `point_labels` | `np.ndarray` | `(N,)` | int | — | — | **FINAL** merged granule id per INPUT point, index-aligned with `points_zyx`. `-1` = noise. |
| `voxel_size_um` | `Tuple[float,float,float]` | `(3,)` | float | (dz, dy, dx) | µm | Echo of the voxel size the boundaries were built in. |
| `merged_from` | `Dict[int, List[int]]` | — | — | — | — | Final id → sorted list of original input ids folded into it. |

Each `GranuleBoundary`:

| field | type | shape | dtype | axis order | units | meaning |
|-------|------|-------|-------|-----------|-------|---------|
| `granule_id` | int | — | — | — | — | Final granule id. |
| `vertices_um` | `np.ndarray` | `(Nv, 3)` | float | **(x, y, z)** | **µm** | The granule's member points, in WORLD coordinates. (This is the whole point set; `faces` index into it.) |
| `faces` | `np.ndarray` | `(Nf, 3)` | int | — | — | Triangle vertex indices into `vertices_um`. Alpha-shape: surface faces owned by exactly one surviving tetra. Convex/Voronoi: hull simplices. May be `(0, 3)` if none. |
| `enclosed_volume_um3` | float | — | — | — | µm³ | Alpha-shape: Σ surviving tetra volumes. Convex: hull volume. Voronoi: Σ finite member cell volumes. |
| `n_points` | int | — | — | — | — | Number of member points. |
| `density` | float | — | — | — | pts/µm³ | `n_points / enclosed_volume_um3` (0 if volume ~0). This is what the merge compares. |
| `delaunay` | `scipy.spatial.Delaunay` or `None` | — | — | (x,y,z) µm | — | **LIVE Delaunay object** for alpha-shape / convex-hull mode (used by the downstream Volume-Mask inside-test via `find_simplex`, no re-triangulation). `None` in voronoi mode. |

## 6. Conventions & GOTCHAS (the real integration risk)

- **AXIS FLIP IS INTRINSIC.** Input `points_zyx` is `(z, y, x)` in **voxels**; all
  geometry and every output `vertices_um` is world **`(x, y, z)` in µm**. The flip
  is done by `_points_to_xyz_um` driven by `voxel_size_um=(dz, dy, dx)`:
  `x = x_vox*dx`, `y = y_vox*dy`, `z = z_vox*dz`. Do NOT pre-flip the input and do
  NOT expect z-first output — downstream consumers read `(x, y, z)` µm.
- **`alpha` has NO `"auto"`.** `alpha <= 0` OR non-finite ⇒ ConvexHull fallback.
  A finite positive `alpha` is a **µm circumradius** threshold — it is scale-sensitive
  to `voxel_size_um`. Too small ⇒ all tetra dropped ⇒ granule dissolves to noise.
- **`adj_dist_um == 0` turns proximity adjacency OFF** in alpha-shape mode (see
  Parameters). In that case granules never merge via proximity; only voronoi mode
  merges by shared ridge.
- **`delaunay` must stay populated for alpha-shape mode** — the downstream Volume-Mask
  node relies on it for its inside-test. Do not strip it. It is `None` for voronoi.
- **Both modes route through `_iterative_merge`** (union-find; on each merge the higher
  id is relabelled to the lower id; density/boundary/adjacency recomputed each pass;
  loop until no pair satisfies `merge_tol`).
- **Voronoi is computed ONCE globally** (cells + ridge adjacency), then sliced per
  granule each merge pass. **Alpha-shapes are rebuilt per granule per merge pass.**
- **Self-healing merge pass:** if any active granule fails to build a boundary
  (degenerate/coplanar), it is dropped to noise and the whole pass restarts.
- **Voronoi cell volume** skips unbounded (outer) cells (`nan`); a granule whose members
  are all on the outer hull can end with volume 0 → dropped to noise.
- **Min-hull floor:** a granule needs `>= 4` points to build (3-D Delaunay/ConvexHull);
  Voronoi additionally needs `>= 5` total points to compute at all.
- `merged_from` values are de-duplicated and sorted at the end.

## 7. Dependencies

| pip package | why | import timing |
|-------------|-----|---------------|
| `numpy` | arrays, linear algebra (cross/einsum/norm) | **import-time** (top-level) |
| `scipy` | `scipy.spatial`: `ConvexHull`, `Delaunay`, `Voronoi`, `cKDTree`, `QhullError` | **import-time** (top-level) |

No lazy/optional deps. No `numba`/`tensorflow`/`cv2`/`sklearn`/`al_dic`. If `scipy`
is absent the module fails at import.

## 8. Failure modes / edge cases

- **`N == 0`:** returns an empty `GranuleTessellation` (`point_labels` shape `(0,)`,
  `boundaries={}`).
- **`len(labels) != N`:** raises `ValueError`.
- **Degenerate/coplanar cloud (voronoi):** `Voronoi` fails or all cells unbounded →
  returns labels unchanged with `boundaries={}`.
- **Degenerate granule (alpha/convex):** `QhullError`/`ValueError` caught → granule
  fails to build → dropped to noise inside the merge loop.
- **`alpha` too small:** no tetra survive → granule dissolves to noise.
- **Zero/near-zero volume:** `density` set to `0.0`; such granules never win a merge
  (guarded by `hi <= _EPS`).
- **All points labelled `-1`:** nothing to tessellate; empty `boundaries`, labels
  returned as-is.

## 9. Minimal runnable example

```python
import numpy as np
from granule_tessellate import tessellate_granules

rng = np.random.default_rng(0)
a = rng.normal([10, 20, 20], 2, size=(40, 3))   # blob 1, (z,y,x) voxels
b = rng.normal([10, 40, 40], 2, size=(40, 3))   # blob 2
points_zyx = np.vstack([a, b])                   # (80, 3)
labels = np.array([1] * 40 + [2] * 40)           # (80,)
voxel_size_um = (0.5, 0.2, 0.2)                  # (dz, dy, dx)

res = tessellate_granules(
    points_zyx, labels, voxel_size_um,
    {"tess_mode": "alpha_shape", "alpha": 8.0, "merge_tol": 0.15, "adj_dist_um": 5.0},
)

# res.point_labels.shape == (80,)          final label per input point
# res.boundaries is Dict[int, GranuleBoundary]  (here 2 granules)
# each boundary: vertices_um (Nv,3) in (x,y,z) µm; faces (Nf,3) int; delaunay is a
#   live scipy.spatial.Delaunay
b1 = next(iter(res.boundaries.values()))
assert b1.vertices_um.shape[1] == 3
assert b1.faces.shape[1] == 3
```

## 10. Pipeline wiring (original ND2Studios pipeline)

- **Upstream (feeds this kernel):** a bead-detection node produces the `(N,3)` `(z,y,x)`
  voxel cloud; a GMM-clustering node produces the initial per-point `labels`. The
  caller supplies `voxel_size_um` from the image metadata. All per-multipoint /
  per-timepoint looping, cropping, exclusion, etc. happen **before** this kernel.
- **Downstream (this kernel feeds):** the `GranuleTessellation` goes to the
  **Volume-Mask** node, which rasterizes each `GranuleBoundary` to a voxel mask using
  the retained `delaunay` (`find_simplex` inside-test) for alpha-shape mode (rebuilds
  one for voronoi). `point_labels` / `merged_from` propagate the final granule identity
  to viewers and per-granule downstream analysis (DVC surfaces, etc.).

## 11. Provenance

Branch **`Version-1.45`**. Vendored verbatim from:

- `nd2studios/backend/analysis/granule_tessellate.py` — the entire kernel.
- `nd2studios/backend/analysis/granule_types.py` — only `NOISE_LABEL` and the
  `GranuleBoundary` + `GranuleTessellation` dataclasses (copied in byte-for-byte).

Edits: removed the `nd2studios...granule_types` import (definitions copied inline);
added `from dataclasses import dataclass, field`. No compute-path change. No
UI/registry-only members existed to drop.
