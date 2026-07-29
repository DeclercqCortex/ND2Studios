# Bead Detection — integration contract

## 1. Purpose + where the real math lives

Detect bead / particle centroids in one **already-prepared** single-channel 3-D
volume and return them as a point cloud. This is the *first* granule-separation
stage in the original pipeline.

**The real math is IN-REPO** (vendored into `bead_detect.py`), not an external
package. It is a hand-written particle detector (`ParticleDetector`):

- **`log` mode** — Laplacian-of-Gaussian → local-maximum filter → 3-point parabola
  sub-voxel refinement (numba kernels `_subpixel_poly_2d/3d`).
- **`components` mode** — threshold → connected-component labeling → intensity-weighted
  centroid, optionally refined with radial-symmetry sub-voxel localization
  (numba kernel `_radial_symmetry_3d`, Liu et al. 2013; 3-D only).

Heavy lifting uses `scipy.ndimage` (LoG, labeling, max-filter, center-of-mass) and
`numba` (JIT sub-voxel kernels). No tracking/DVC package is required.

## 2. Entry point

```python
def detect_beads(volume_zhw, voxel_size_um, params) -> tuple[np.ndarray, list[dict]]
```

Everything else in the module (`ParticleDetector`, `DetectionConfig`,
`DetectionMethod`, `make_point_rows`, the numba kernels, the `_*` helpers) is
support code called by `detect_beads`. Call **only** `detect_beads`.

## 3. Inputs

| name | Python type | array shape | dtype | axis order | units | required? / default | meaning & constraints |
|------|-------------|-------------|-------|-----------|-------|---------------------|-----------------------|
| `volume_zhw` | `np.ndarray` | `(Z, H, W)` or `(H, W)` | any real (int/float); cast to float64 internally | `(z, y, x)` | raw intensity | required | One channel's raw volume. `Z == 1` or a 2-D `(H, W)` array triggers the 2-D fallback. `ndim` other than 2 or 3 → `ValueError`. |
| `voxel_size_um` | `tuple[float, float, float]` | `(3,)` | float | `(dz, dy, dx)` | micrometers | required | Physical voxel spacing. **Order is `(dz, dy, dx)`** (matches `viz3d.prep.Spacing`), NOT `(dx,dy,dz)`. Used for anisotropy factors and for the µm columns in the rows. Non-positive entries are ignored when computing the anisotropy minimum. |
| `params` | `dict` | — | — | — | — | required (may be `{}`) | Read with `.get` defaults — see Parameters. |

## 4. Parameters (keys inside `params`)

| name | type | default | valid range / choices | semantics |
|------|------|---------|-----------------------|-----------|
| `detect_mode` | str | `"log"` | `"log"` \| `"components"` | `log` = LoG local-maxima (→ `DetectionMethod.TRACTRAC`, `bead_radius = max(1, min_distance_px/2)`); `components` = connected-component centroids (`bead_radius = 0`; `TPT` if `subpixel` else `TRACTRAC`). Case-insensitive; anything not `"components"` falls to `log`. |
| `min_distance_px` | int/float | `5` | `> 0` | Minimum peak separation in **voxels**. Doubles as the LoG scale (`sigma = min_distance_px/2`) AND the greedy NMS radius applied after detection. `<= 0` disables the NMS post-filter. |
| `threshold` | float | `0.0` | `[0, 1]` normalized | Foreground threshold on the **max-normalized** image (`img/img.max()`). `0` → auto Otsu (`_otsu_threshold_norm`, computed on the max-normalized histogram). |
| `min_intensity` | float | `0.0` | `>= 0`, raw units | Post-filter: drop peaks whose **raw** intensity (sampled at the rounded voxel) is below this. `0` disables it. |
| `subpixel` | bool | `True` | — | Keep sub-voxel refinement. `False` rounds final centroids to integer voxels (`np.rint`) and, in `components` mode, skips radial-symmetry. |
| `min_size` | int | `1` | `>= 1` voxels | Min blob volume (3-D) / area (2-D); blobs outside `[min_size, max_size]` are discarded before centroid/LoG masking. |
| `max_size` | int | `2**31 - 1` | — | Max blob volume/area (effectively unbounded by default). |
| `m_position` / `m` | int | `0` | — | Multipoint index written onto every DATA row (metadata only; no looping here). |
| `frame` / `t` | int | `0` | — | Timepoint index written onto every DATA row (metadata only). |
| `all_multipoints` | bool | — | — | **Ignored by this kernel** (consumed by the app page handler; harmless if present). |

## 5. Output

Returns a 2-tuple `(points_zyx, rows)`.

**`points_zyx`** — `np.ndarray`

| aspect | value |
|--------|-------|
| shape | `(N, 3)` (always 2-D, even when `N == 0` → `(0, 3)`) |
| dtype | `float64` (C-contiguous) |
| axis order | `(z, y, x)` — **voxel units**, NOT µm |
| meaning | Sub-voxel centroids. In the 2-D fallback the z-column is forced to `0.0`. |

**`rows`** — `list[dict]`, length `N`, each dict has exactly these keys:

| key | type | units | meaning |
|-----|------|-------|---------|
| `bead_id` | int | — | Row index `0..N-1` (positional; not a persistent id). |
| `m_position` | int | — | From `params`. |
| `frame` | int | — | From `params`. |
| `centroid_z_px`, `centroid_y_px`, `centroid_x_px` | float | voxels | Same coords as `points_zyx`. |
| `centroid_z_um`, `centroid_y_um`, `centroid_x_um` | float | micrometers | `px * dz/dy/dx` respectively (the **only** place voxel→µm happens). |
| `granule_id` | `None` | — | Always `None` here (pre-clustering; a later stage fills it). |

`points_zyx` and `rows` are index-aligned (`rows[i]` ↔ `points_zyx[i]`).

## 6. Conventions & GOTCHAS (the real integration risk)

- **`(z,y,x)` ⇄ `(x,y,z)` transpose/flip is LOAD-BEARING.** The vendored
  `ParticleDetector` works natively in `(x, y, z)` axis order. `_detector_coords`
  transposes the input `(Z,H,W)`→`(W,H,Z)` before `detect()` and reverses the
  returned columns back to `(z,y,x)`. This is the **only** place the axis order
  changes; every value this kernel returns is `(z,y,x)`. Do not "simplify" it away.
- **`voxel_size_um` is `(dz, dy, dx)`** — z first. Feeding `(dx,dy,dz)` silently
  corrupts anisotropy and the µm columns.
- **Anisotropy vs. units.** Internally `cfg.abc = (dx,dy,dz)/min(dx,dy,dz)` in the
  detector's `(x,y,z)` order (dimensionless aspect ratio), while `cfg.dccd = (1,1,1)`.
  This keeps the detector's sub-voxel shifts in **voxel** units — so the returned
  cloud is voxel coords and the µm conversion happens exactly once, in
  `make_point_rows`. Don't set `dccd` to the physical spacing.
- **`threshold == 0` means auto-Otsu**, not "no threshold". Otsu is computed on the
  **max-normalized** image to match the detector's internal `img/img.max()`
  normalization. A nonzero threshold is interpreted in that same `[0,1]` space.
- **2-D fallback** (`Z == 1` or a plain `(H,W)` input): output stays `(N,3)` with the
  z-column forced to `0.0`; radial-symmetry refinement is unavailable (2-D uses the
  parabola path). Downstream stays uniform.
- **Post-filters run in the wrapper, AFTER the detector**, in this order:
  (1) `min_intensity` floor on raw intensity sampled at rounded/clipped voxels,
  then (2) greedy min-distance NMS keeping the brightest point per `min_distance_px`
  (Euclidean, voxel) neighborhood.
- **`min_distance_px` overloads two roles**: LoG sigma (`/2`) and NMS radius.
  Changing it changes both blob scale and dedup radius.
- **LoG detection trims a border** of `max((sigma+2)/2, 1)` voxels and rejects
  peaks whose parabola shift `|d| >= 0.5`. Beads within that border are not returned.
- **`richardson_lucy` (skimage) is lazy** and only fires if `cfg.psf` is set — this
  wrapper never sets it, so skimage is not exercised on the default path.
- **Determinism:** the LoG path adds a fixed-seed (`rng(42)`) tie-break noise of
  `1e-5` before the max-filter; `components`+`subpixel` adds fixed-seed (`rng(0)`)
  `rand_noise` (`1e-7`) before radial symmetry. Results are reproducible.

## 7. Dependencies

| pip package | why | import time |
|-------------|-----|-------------|
| `numpy` | arrays, Otsu, NMS, row building | **import-time** (top level) |
| `scipy` | `scipy.ndimage`: gaussian_laplace, label, sum_labels, maximum_filter, center_of_mass | **import-time** (top level — vendored verbatim from `detection.py`) |
| `numba` | JIT sub-voxel kernels (`@nb.njit`) | **import-time** (top level — vendored verbatim from `detection.py`) |
| `scikit-image` | `richardson_lucy` PSF deconvolution | **lazy** — imported inside `ParticleDetector.detect` only when `cfg.psf is not None` (never on this wrapper's path). Not required. |

`_require_backends` re-checks `numpy`/`scipy`/`numba` via `importlib.util.find_spec`
and raises a friendly `ImportError` naming the missing package. Because scipy/numba
are also top-level imports, a missing one actually fails at **module import**.

## 8. Failure modes / edge cases

- **Wrong ndim:** input that is neither 2-D nor 3-D → `ValueError`.
- **Empty / all-zero volume:** detector returns `(0,3)`; `rows == []`. No crash.
- **No beads pass filters:** `(0,3)` + `[]`.
- **`vmax <= 0`** anywhere: Otsu returns `0.0`; detector's normalization returns an
  empty `(0, ndim)` result.
- **`components` mode in 2-D:** radial symmetry is skipped (3-D only); plain centroids.
- **Missing numba/scipy:** module import fails (heavy dep) — install them. `skimage`
  absence is fine unless a PSF is supplied (this wrapper never does).
- **Degenerate blobs** filtered out by `[min_size, max_size]` never appear.

## 9. Minimal runnable example

```python
import sys; sys.path.insert(0, "pure_analysis")
import numpy as np
from bead_detect import detect_beads

rng = np.random.default_rng(0)
vol = (rng.random((12, 64, 64)) * 20).astype(np.float32)   # (Z, H, W)
for z, y, x in [(6, 20, 20), (6, 40, 44), (3, 50, 15)]:
    vol[z-1:z+2, y-2:y+3, x-2:x+3] += 400.0                # plant 3 bright blobs

pts, rows = detect_beads(
    vol,
    voxel_size_um=(2.0, 0.5, 0.5),          # (dz, dy, dx) µm
    params={"detect_mode": "log", "min_distance_px": 4, "threshold": 0.0},
)
# pts.shape == (3, 3)  float64, axis order (z, y, x) in voxels
# len(rows) == 3 ; rows[i] has centroid_*_px, centroid_*_um, granule_id is None

# 2-D fallback: pass a single (H, W) plane
pts2, rows2 = detect_beads(vol[6], (2.0, 0.5, 0.5), {})
# pts2.shape == (N, 3) with pts2[:, 0] all == 0.0
```

Expected: `pts.shape == (3, 3)`, `pts.dtype == float64`, `len(rows) == 3`,
`rows[0]["granule_id"] is None`; 2-D call returns `(N, 3)` with a zero z-column.

## 10. Pipeline wiring (original pipeline)

- **Upstream (caller-owned prep, NOT vendored):** an ND2 plane runner / handler
  loads one channel, selects the multipoint `m` and timepoint `t`, applies any
  crop / downsample / registration / exclusion, and Z-collapses/keeps the stack into
  a single raw `(Z, H, W)` volume plus the `(dz,dy,dx)` voxel size. That prepared
  volume + voxel size + the node's params dict are what feed `detect_beads`.
- **Downstream:** `points_zyx` `(N,3)` `(z,y,x)` voxel cloud is the fast path into the
  next granule stages (clustering / GMM → tessellation → masks). The `rows`
  `List[Dict]` ride the app's `PortType.DATA` convention; a later clustering node
  fills each row's `granule_id`. µm columns are read by the measurement/track layer.

## 11. Provenance

Branch **`Version-1.45`**. Vendored verbatim by byte-copy from:

- `nd2studios/backend/analysis/bead_detect.py` — `detect_beads` + `_otsu_threshold_norm`,
  `_sample_intensity`, `_suppress_close`, `_detector_coords`, `_require_backends`, defaults.
- `nd2studios/backend/serialtrack/detection.py` — `ParticleDetector` + numba kernels
  `_subpixel_poly_2d`, `_subpixel_poly_3d`, `_radial_symmetry_3d`.
- `nd2studios/backend/serialtrack/config.py` — `DetectionConfig`, `DetectionMethod`.
- `nd2studios/backend/analysis/granule_types.py` — `make_point_rows`, `NOISE_LABEL`.

Dropped as off-compute-path (see module header): `config.py` tracking enums/dataclasses
(`GlobalSolver`, `LocalSolver`, `TrackingMode`, `TrajectoryConfig`, `TrackingConfig`);
`granule_types.py` `GranuleBoundary`/`GranuleTessellation`, `*_ATTR` constants,
`points_from_rows`, `GRANULE_PALETTE`/`granule_color`. The single edit to compute code
was removing nd2studios/relative imports (now defined in-file).
