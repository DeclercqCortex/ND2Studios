# Granule Volume Mask — integration contract

## 1. Purpose + where the real math lives

Voxelize tessellated granule boundaries (one closed surface / point-cloud per
granule) onto a confocal `(Z, H, W)` voxel grid, producing:

- a **per-granule** `{label: (Z,H,W) bool}` dict, and
- a **combined** `(Z,H,W) int32` label volume (`0` = background).

**Where the real math lives: IN-REPO.** The whole algorithm is defined in the
vendored `granule_volume_mask.py` (point-in-Delaunay rasterization, signed-
distance-Gaussian non-shrinking smoothing, connected-component speck removal,
dense 1-based relabel, density/-id overlap tie-break). The only external heavy
lifting is `scipy` (`scipy.spatial.Delaunay` for the inside test;
`scipy.ndimage` for EDT / gaussian / label / fill-holes). No external analysis
package — this is not a thin wrapper over anything.

## 2. Entry point

```python
def build_granule_masks(
    tess,                         # duck-typed tessellation (see Inputs)
    shape_zhw: tuple[int,int,int],       # (Z, H, W)
    voxel_size_um: tuple[float,float,float],  # (dz, dy, dx) µm
    params,                       # dict OR attribute-object OR None
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    # -> ({dense-1-based-label: (Z,H,W) bool}, combined (Z,H,W) int32)
```

Caller-convenience adapter (build the `tess` argument without the app dataclass):

```python
def tessellation_from_list(granules) -> _SimpleTessellation
# granules: iterable of tuples, each:
#   (vertices_um,)                    # density 0.0, convex-hull inside test
#   (vertices_um, density)            # convex-hull inside test
#   (vertices_um, density, delaunay)  # reuse a prebuilt scipy.spatial.Delaunay
# vertices_um: (Nv, 3) world (x, y, z) µm
```

## 3. Inputs

| name | Python type | array shape | dtype | axis order | units | required? / default | meaning & constraints |
|------|-------------|-------------|-------|-----------|-------|---------------------|------------------------|
| `tess` | duck-typed object | — | — | — | — | required | Must expose `.boundaries` → `{gid: boundary}`. Each `boundary` must expose `.vertices_um`, `.density`, `.delaunay`. Missing `.boundaries` → treated as empty → empty result. |
| `tess.boundaries` | `dict[int, boundary]` | — | — | — | — | required | Keys are original granule ids (may be `0`; ids need not be contiguous). |
| `boundary.vertices_um` | `np.ndarray` | `(Nv, 3)` | float | columns = (X, Y, Z) | µm (world) | required | World coords of the boundary point cloud / mesh vertices. `Nv==0` → granule skipped. `<4` non-degenerate pts → no Delaunay → skipped. |
| `boundary.delaunay` | `scipy.spatial.Delaunay` or `None` | — | — | (x,y,z) | µm | required attr (may be `None`) | Prebuilt triangulation for the inside test (alpha-shape concavity preserved). If `None`, a fresh convex-hull `Delaunay(vertices_um)` is built here. |
| `boundary.density` | `float` | scalar | float | — | arbitrary | required attr | Only used for the combined-volume overlap tie-break (higher wins). |
| `shape_zhw` | `tuple[int,int,int]` | `(3,)` | int | (Z, H, W) | voxels | required | Output grid dims. Any non-positive dim → all-background result. |
| `voxel_size_um` | `tuple[float,float,float]` | `(3,)` | float | (dz, dy, dx) | µm | required | Physical voxel spacing. Any non-positive spacing → that granule voxelizes to empty. |
| `params` | `dict` / attr-object / `None` | — | — | — | — | `None`/`{}` → all defaults | See Parameters. |

## 4. Parameters (read from `params` via dict-get or getattr)

| name | type | default | valid range / choices | semantics |
|------|------|---------|-----------------------|-----------|
| `smooth_sigma` | float (µm) | `0.0` | `>= 0`; `0` = off | SDF-Gaussian smoothing scale. Per-axis sigma = `smooth_sigma / (dz,dy,dx)` voxels. Rounds convex corners **without shrinking** volume (blurs the signed distance field, re-thresholds `> 0`). |
| `fill_holes` | bool | `False` | — | `scipy.ndimage.binary_fill_holes` on each granule volume (after smoothing). |
| `min_object_voxels` | int | `1` | `>= 1`; `1` = keep all | Drop 26-connected components smaller than this (speck removal). A granule that ends up empty is omitted from the dict and painted nowhere. |

Order of ops per granule: voxelize → (smooth if `smooth_sigma>0`) → (fill_holes if set and non-empty) → drop small components → keep if non-empty.

## 5. Output

`(masks, combined)`:

**`masks`** — `dict[int, np.ndarray]`
- key: `int` — **dense 1-based** granule label (`1..K`). NOT the original id.
- value: `(Z, H, W)` `bool` array, C-contiguous. Axis order `(z, y, x)`. `True` = inside granule.
- Per-granule masks may **overlap** each other (intentionally not disambiguated).

**`combined`** — `np.ndarray`, shape `(Z, H, W)`, dtype `int32`, axis order `(z,y,x)`.
- Value `0` = background; value `L` (1-based) = the granule that owns that voxel after the overlap tie-break.
- Shares the SAME relabeled ids as `masks` (`combined == L` ⇔ that granule's label).

## 6. Conventions & GOTCHAS (the real integration risk)

- **Coordinate mapping is exact.** Vertices are world `(x, y, z)` µm (column order X,Y,Z). `voxel_size_um = (dz, dy, dx)`. A voxel at index `(z, y, x)` has world center `(x*dx, y*dy, z*dz)`. World-z of plane 0 is `z0 = 0`. Plane of a world-z sample = `floor((z_um - z0)/dz)`, `z0 = 0`. **Feed vertices in world µm with this origin, or masks land on the wrong planes.** (Caller owns any offset/registration to make `z0 = 0` hold.)
- **Axis-order flip between vertices and arrays:** vertices are `(x,y,z)`; output arrays are `(z,y,x)`. Do not transpose one to match the other — the kernel handles it.
- **Relabeling:** returned dict keys are dense `1..K` in ascending-original-id order, NOT the original granule ids. This exists because original ids can be `0`, which would collide with background. `masks` and `combined` share these relabeled ids — downstream self-exclusion (`combined != label`) and DVC-on-object rely on this agreement.
- **Overlap tie-break (combined only):** granules are painted in ascending `(density, -id)` order (last write wins), so **higher density wins**; on equal density, **lower original id wins**. Per-granule bool masks are left overlapping — only `combined` is disambiguated.
- **Inside test:** uses `boundary.delaunay.find_simplex(centers) >= 0` when present (preserves alpha-shape concavity). If absent, rebuilds a convex-hull Delaunay → a Voronoi/no-triangulation boundary **degrades to its convex hull** (concavities are filled).
- **SDF smoothing is non-shrinking** (blurs `edt(mask)-edt(~mask)` in µm, re-thresholds `>0`); empty or full masks pass through unchanged.
- **Degenerate boundaries** (`<4` pts, coplanar → QhullError) are silently skipped, not errors.
- **Bounding-box optimization:** only voxels inside each boundary's world bbox (padded 1 voxel, clamped) are tested — voxels outside the vertex extent are never inside.

## 7. Dependencies

| package | why | import timing |
|---------|-----|---------------|
| `numpy` | arrays throughout | **import-time** (top level) |
| `scipy` | `spatial.Delaunay`/`QhullError` (inside test); `ndimage.distance_transform_edt`, `gaussian_filter`, `label`, `binary_fill_holes` | **lazy** — imported inside functions, only when `build_granule_masks` actually runs. The module imports fine without scipy installed. |

No nd2studios imports; no `granule_types` runtime dependency (type-hint only, dropped).

## 8. Failure modes / edge cases

- Empty / missing `tess.boundaries` → `({}, zeros((Z,H,W), int32))`.
- Non-positive `Z/H/W` or `dz/dy/dx` → that granule (or the whole grid) voxelizes to empty; combined is all-zero.
- `Nv < 4` or degenerate/coplanar cloud with no prebuilt Delaunay → granule skipped.
- Granule emptied by `min_object_voxels` → omitted from dict, absent from combined.
- `scipy` not installed → import succeeds, but calling `build_granule_masks` raises `ImportError` at first scipy use.
- Vertices in the wrong origin/scale → masks silently land on wrong planes (no error). Verify the coordinate convention (§6).

## 9. Minimal runnable example

```python
import sys; sys.path.insert(0, "pure_analysis")
import numpy as np, granule_volume_mask as gvm

def fib_sphere(n, r, center_xyz):
    i = np.arange(n, dtype=float); g = np.pi*(3-np.sqrt(5.0))
    z = 1 - 2*(i+0.5)/n; rho = np.sqrt(np.clip(1-z*z, 0, 1)); th = g*i
    u = np.column_stack([np.cos(th)*rho, np.sin(th)*rho, z])
    return u*r + np.asarray(center_xyz, float)

DZ, DY, DX = 2.0, 0.5, 0.5
verts = fib_sphere(600, 6.0, (10.0, 10.0, 12.0))   # world (x,y,z) µm

tess = gvm.tessellation_from_list([(verts, 1.0)])   # (vertices_um, density)
masks, combined = gvm.build_granule_masks(tess, (16, 48, 48), (DZ, DY, DX), {})

# masks -> {1: (16,48,48) bool};  combined -> (16,48,48) int32, unique {0,1}
print(sorted(masks.keys()))            # [1]
print(masks[1].shape, masks[1].dtype)  # (16, 48, 48) bool
print(int(masks[1].sum()))             # ~1721  (analytic (4/3)πr³/(dz·dy·dx) ≈ 1810)
print(combined.dtype, sorted(np.unique(combined).tolist()))  # int32 [0, 1]
```

Two-granule overlap: pass `[(v1, 1.0), (v2, 2.0)]` — combined `unique` is `{0,1,2}`
and the density-2 granule owns every contested voxel.

## 10. Pipeline wiring (original)

- **Upstream (feeds this kernel):** the granule-separation chain produces a
  `GranuleTessellation` (P3) — a GMM/clustering step assigns points to granules,
  then per-granule boundaries (alpha-shape or Voronoi mode) carry `vertices_um`,
  `density`, and an optional prebuilt `delaunay`. The caller also supplies the
  confocal grid `shape_zhw` and `voxel_size_um` from the raw stack's metadata.
  All prep (file load, per-multipoint/per-timepoint looping, crop, downsample,
  registration, exclusion) happens **before** this kernel — the caller owns it.
- **Downstream (this kernel feeds):** the per-granule `{label: (Z,H,W) bool}`
  dict has the exact shape of `record._mask3d_by_m[m][t]` for one object, so
  `object_scope.iter_objects` → per-object DVC → `viz3d` surface rendering
  consume it unchanged. The combined int32 volume drives self-exclusion
  (`combined != label`) and DVC-on-object label lookups.

## 11. Provenance

- Branch: **Version-1.45**
- Vendored verbatim: `nd2studios/backend/analysis/granule_mask.py`
- Extraction blueprint (not vendored): `tests/granule/test_granule_mask.py`
- Dropped (rule 2a): `from nd2studios.backend.analysis.granule_types import (COMBINED_LABELS_KEY, GranuleBoundary, GranuleTessellation)` — `COMBINED_LABELS_KEY` unused on compute path; the two type names are annotation-only (aliased to `Any`). No compute-path code altered; no helper renamed.
