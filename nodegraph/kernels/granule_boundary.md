# Granule Boundary Extraction — Integration Contract

## 1. Purpose

Grow an **outward boundary band** of thickness `N` voxels around each labeled
granule: the voxels within `N` of a granule's surface whose label is **not** that
granule (i.e. background **and** neighbouring granules — "the direction that there
is not the same value"). Produces one bool band per granule plus a **combined
band-label volume** painting each band voxel with its source granule id (overlaps
resolved nearest-surface, then lowest-id).

**Where the real math lives:** fully **in-repo** — pure NumPy + `scipy.ndimage`
(`binary_dilation`, `distance_transform_edt`, `generate_binary_structure`). No
external analysis package.

## 2. Entry point

```python
def extract_boundary_bands(
    masks_by_id: Mapping[Any, np.ndarray],
    combined_labels_zhw: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    params: Mapping[str, Any],
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    ...
```

Helper (also vendored): `_iter_granule_masks(masks_by_id) -> Dict[int, np.ndarray]`
— filters the input dict to int-gid bool masks, skipping the reserved `'_labels'`
key and any non-integer key.

## 3. Inputs

| name | Python type | array shape | dtype | axis order | units | required? | meaning & constraints |
|------|-------------|-------------|-------|------------|-------|-----------|-----------------------|
| `masks_by_id` | `Mapping[Any, np.ndarray]` | each mask `(Z,H,W)` | mask cast to `bool` | `(z, y, x)` | — | required | `{int gid -> bool mask}`. **May also contain a reserved `'_labels'` int volume and other non-int keys — these are skipped.** Masks whose shape ≠ `combined_labels_zhw` or that are all-False yield an all-False band. |
| `combined_labels_zhw` | `np.ndarray` | `(Z,H,W)` | integer | `(z, y, x)` | label id | required | Authoritative label source. Voxel value = owning granule id, `0` = background. May include neighbour ids **not** present in `masks_by_id`. Defines output shape. |
| `voxel_size_um` | `Tuple[float,float,float]` | `(3,)` | float | `(dz, dy, dx)` | µm | required | Feeds `distance_transform_edt(sampling=...)`; anisotropy-correct EDT band + tie-break. |
| `params` | `Mapping[str, Any]` | — | — | — | — | required | See §4. Missing keys fall back to defaults. |

## 4. Parameters (inside `params`)

| name | type | default | valid range / choices | semantics |
|------|------|---------|-----------------------|-----------|
| `band_voxels` | int | `1` | `>= 0` (clamped via `max(0, n)`) | `N`. Dilation: number of 6-connectivity iterations (grow by taxicab distance ≤ N). EDT: sets threshold when `band_um` not given (`N * min(dz,dy,dx)`). `N=0` in dilation → empty band. |
| `band_method` | str | `"dilation"` | `"dilation"` \| `"edt"` (lowercased) | `"dilation"`: `binary_dilation(mask, 6-conn, iterations=N) & ~mask & not_self`. `"edt"`: `(0 < edt <= threshold) & not_self & ~mask`. Any value other than `"edt"` falls through to dilation. |
| `band_um` | float or None/`""` | `0.0` | `> 0` to take effect | EDT threshold in µm; overrides `N * min(voxel)` when `> 0`. Ignored by the dilation method. |
| `include_neighbors` | bool | `True` | `True` \| `False` | `True`: `not_self = (labels != gid)` — band reaches into neighbouring granules **and** background. `False`: `not_self = (labels == 0)` — band restricted to background only. |

## 5. Output

`Tuple[bands_by_id, combined_band_labels]`

| field | structure | shape | dtype | axis order | units | meaning |
|-------|-----------|-------|-------|------------|-------|---------|
| `bands_by_id` | `Dict[int, np.ndarray]` | each `(Z,H,W)` | `bool` | `(z,y,x)` | — | Per-granule outward band. Present for every int gid in the (filtered) input, even if all-False. Overlaps NOT resolved here — bands may share voxels. |
| `combined_band_labels` | `np.ndarray` | `(Z,H,W)` | `int32` | `(z,y,x)` | granule id | Each band voxel painted with its owning granule id; `0` = no band. Overlaps resolved nearest-surface (smaller EDT dist), ties → lowest gid. |

If the (filtered) `masks_by_id` is empty, returns `({}, zeros((Z,H,W), int32))`.

## 6. Conventions & GOTCHAS

- **`'_labels'` reserved key.** `masks_by_id` deliberately mixes int-gid bool
  masks with a reserved `'_labels'` int volume (and possibly other non-int keys).
  A faithful caller can hand the whole unfiltered dict; `_iter_granule_masks`
  skips `'_labels'` and any key that fails `int(key)`. Do **not** pre-strip it
  in a way that changes semantics — just pass it through.
- **Axis order is `(z, y, x)` throughout** for masks and label volume; voxel size
  is `(dz, dy, dx)`. No x/y/z flips happen in this kernel.
- **`combined_labels_zhw` is authoritative**, independent of `masks_by_id`. A
  granule's band with `include_neighbors=True` reaches into any voxel whose label
  ≠ gid, which can include neighbour ids that were never handed in as masks.
- **Tie-break is order-dependent by construction but deterministic.** Granules are
  processed in **ascending gid** order; combined-volume claim uses strict
  `edt < best_dist`, so on an exact distance tie the earlier (lower-id) granule
  keeps the voxel — a later, equal-distance granule never displaces it.
- **Dilation vs EDT agreement.** The 6-connectivity structuring element
  (`generate_binary_structure(3, 1)`) is deliberate: iterating N times grows by
  taxicab distance ≤ N and matches the EDT band at `N=1` on isotropic voxels.
- **`band_um` only affects EDT.** In dilation mode the band thickness is purely
  `band_voxels`; `band_um` is ignored.
- **`~mask` excludes the granule interior** from its own band in both methods.
- **Shape/empty guard.** Masks with the wrong shape or no True voxels get an
  all-False band and are skipped in the combined paint (no error raised).

## 7. Dependencies

| package | pip name | why | import time |
|---------|----------|-----|-------------|
| numpy | `numpy` | arrays, dtype casts, `best_dist` bookkeeping | **import-time** (top level) |
| scipy | `scipy` | `scipy.ndimage.binary_dilation`, `distance_transform_edt`, `generate_binary_structure` | **lazy** — imported inside `extract_boundary_bands` (only when a non-empty mask set is processed). Verbatim from source. |

The module **imports at top level only `numpy`**; scipy is imported inside the
function exactly as the original does, so `import granule_boundary` succeeds even
without scipy installed — scipy is required only to actually run the kernel.

## 8. Failure modes / edge cases

- **Empty / all-non-int `masks_by_id`** → `({}, zeros int32)`, no scipy import.
- **All-False or wrong-shape mask** → all-False band for that gid; skipped in
  combined paint.
- **`band_voxels=0` + dilation** → `grown = mask`, so band = `mask & ~mask & ...`
  = all-False.
- **scipy missing** → `ImportError` at call time (not import time), only if there
  is at least one valid mask.
- **`band_um` non-numeric (not None/"")** → `float(band_um_param)` may raise
  `ValueError`; pass a number, `None`, or `""`.
- **Degenerate geometry** (single-voxel granule, granule filling the whole volume)
  is handled: EDT of a full-True complement is all-inf → no band voxels ≤ threshold.

## 9. Minimal runnable example

```python
import numpy as np
import sys; sys.path.insert(0, "pure_analysis")
from granule_boundary import extract_boundary_bands

Z, H, W = 6, 20, 20
labels = np.zeros((Z, H, W), dtype=np.int32)
labels[2:4, 5:9, 5:9] = 1      # granule 1
labels[2:4, 12:16, 12:16] = 2  # granule 2

masks_by_id = {
    1: labels == 1,
    2: labels == 2,
    "_labels": labels,          # reserved key — skipped by the kernel
}

bands, combined = extract_boundary_bands(
    masks_by_id,
    labels,
    voxel_size_um=(1.0, 0.5, 0.5),
    params={"band_voxels": 2, "band_method": "dilation", "include_neighbors": True},
)

# bands[1].shape == (6, 20, 20), dtype bool
# combined.shape == (6, 20, 20), dtype int32; values in {0, 1, 2}
assert bands[1].shape == (Z, H, W) and bands[1].dtype == bool
assert combined.shape == (Z, H, W) and combined.dtype == np.int32
```

## 10. Pipeline wiring (original context)

- **Feeds INTO this kernel** (P4 granule tessellation → mask node output):
  `record._granule_masks_by_m[m][t]` — the per-`(m,t)` dict of `{gid: bool mask,
  "_labels": int32 label volume}`, plus the voxel size `(dz,dy,dx)`. The caller
  (page / plane_runner) selects the multipoint/timepoint and hands the whole dict
  plus its `'_labels'` entry as `combined_labels_zhw`.
- **This kernel feeds** `record._granule_bands_by_m[m][t]`, stored in the same
  shape convention (`{gid: bool band, "_labels": combined int32 band}`). Downstream
  the band masks the raw / DVC volume so the matrix surrounding each granule can be
  recovered and correlated (per-granule DVC surface selection, V1.74).

## 11. Provenance

- Branch: **Version-1.45**
- `nd2studios/backend/analysis/granule_boundary.py` — vendored **verbatim**
  (entire module).
- `nd2studios/backend/analysis/granule_types.py` — only `COMBINED_LABELS_KEY =
  "_labels"` inlined as a module constant.
- Edits: removed the one `from nd2studios...granule_types import
  COMBINED_LABELS_KEY` line and inlined the constant. No renames, no dropped
  compute-path or UI/registry members (the source had none).
