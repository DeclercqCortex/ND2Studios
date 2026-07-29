# Histogram Threshold Segmenter — Integration Contract

## 1. Purpose + where the real math lives

Single-frame, histogram-driven intensity-threshold segmentation of a **2-D**
image. Given an already-prepared 2-D integer frame, the kernel thresholds it
(one of **4 methods × 4 directions**), cleans the resulting binary mask with
morphology (opening → closing → hole-fill → small/large-object removal), labels
the connected components, and measures per-region properties. It returns a
`SegmentationResult` bundling the boolean mask, the integer label image, a list
of per-region dicts, the histogram (when computed), the resolved thresholds, the
config used, and a provenance dict.

**Where the real math lives: IN-REPO.** There is **no external analysis package**
behind this. The algorithm is native ND2Studios code built on stock
`scikit-image` (`skimage.filters` / `.morphology` / `.measure` / `.segmentation`)
and `scipy.ndimage` primitives. The orchestration, the four thresholding
directions, the hysteresis inversion trick, the percentile→intensity mapping, and
the region measurement are all repo code vendored verbatim.

## 2. Entry point

```python
from histogram_threshold import (
    HistogramThresholdSegmenter,
    ThresholdConfig,
    SegmentationResult,
    make_config,   # convenience: forces bit_depth_strict=False
)

cfg: ThresholdConfig = make_config(method="hysteresis", direction="above",
                                   strict=3500, permissive=3000, min_area=100)

segmenter = HistogramThresholdSegmenter(cfg)

result: SegmentationResult = segmenter.run(
    image,                     # np.ndarray, 2-D (H, W), integer dtype
    reference_mask=None,       # optional np.ndarray[bool] (H, W); only used by method="relative"
    voxel_size=None,           # optional tuple[float, ...]; e.g. (dy, dx) in um
)
```

Exact signatures:

```python
class ThresholdConfig:  # dataclass — see §4 for all fields
    ...

class HistogramThresholdSegmenter:
    def __init__(self, config: ThresholdConfig) -> None: ...
    def run(self, image: np.ndarray, *,
            reference_mask: np.ndarray | None = None,
            voxel_size: tuple[float, ...] | None = None) -> SegmentationResult: ...

def make_config(*, bit_depth_strict: bool = False, **kwargs) -> ThresholdConfig: ...
```

`make_config` is the recommended constructor for headless/node use: it flips the
`ThresholdConfig` `bit_depth_strict` default from `True` to `False` (see §6).

## 3. Inputs

| name | Python type | array shape | dtype | axis order | units | required? / default | meaning & constraints |
|------|-------------|-------------|-------|-----------|-------|---------------------|-----------------------|
| `image` | `np.ndarray` | `(H, W)` — **2-D only** | **integer** (`uint8/uint16/int*`) | `(row=y, col=x)` | raw integer counts in `[0, 2**bit_depth − 1]` | **required** | The single already-prepared frame to segment. Must be integer dtype — float raises `BitDepthError`. Caller has already done any crop/downsample/registration/exclusion and any per-T/per-Z/per-multipoint looping. |
| `reference_mask` | `np.ndarray[bool]` or `None` | `(H, W)` (same as `image`) | bool | `(y, x)` | — | optional / `None` | **Only consulted when `method="relative"`.** Selects the pixels whose **median** intensity anchors the relative threshold. `None` → the whole frame is the reference. Ignored by all other methods. |
| `voxel_size` | `tuple[float, ...]` or `None` | e.g. `(dy, dx)` | float | matches image axes | µm/px | optional / `None` | If given, each region gets an extra `area_um2 = area_px * prod(voxel_size)`. `None` → no `area_um2` key. NOTE: kernel multiplies **all** elements of the tuple, so pass exactly the in-plane spacings for a 2-D frame (`(dy, dx)`); a 3-element voxel size would wrongly fold in Z. |

## 4. Parameters (`ThresholdConfig` fields)

| name | type | default | valid range / choices | semantics |
|------|------|---------|-----------------------|-----------|
| `method` | str | `"hysteresis"` | `single` \| `hysteresis` \| `percentile` \| `relative` | Which threshold engine to run. |
| `direction` | str | `"below"` | `below` \| `above` \| `between` \| `outside` | Sense of the cut (see §6). `hysteresis` supports **only** `below`/`above`. |
| `low` | int \| None | `None` | `[0, 2**bit_depth−1]` | `single`: lower cutoff. `below`→`image<=low`; `between`/`outside` use both. |
| `high` | int \| None | `None` | same | `single`: upper cutoff. `above`→`image>=high`. |
| `strict` | int \| None | `None` | same | `hysteresis`: seed level (must be connected to). `below`: `strict<=permissive`; `above`: `strict>=permissive`. |
| `permissive` | int \| None | `None` | same | `hysteresis`: grow level. |
| `percentile_low` | float \| None | `None` | `[0, 100]` | `percentile`: maps to an intensity via the frame CDF, then used as `low`. |
| `percentile_high` | float \| None | `None` | `[0, 100]` | `percentile`: maps to an intensity, used as `high`. |
| `sanity_floor` | int \| None | `None` | intensity | `percentile`: clamps resolved `low` via `low = min(low, sanity_floor)` (guards against calling bright pixels "dark"). |
| `sanity_ceiling` | int \| None | `None` | intensity | `percentile`: clamps resolved `high` via `high = max(high, sanity_ceiling)`. |
| `fraction_low` | float \| None | `None` | ≥0 | `relative`: `low = round(reference_median * fraction_low)`. |
| `fraction_high` | float \| None | `None` | ≥0 | `relative`: `high = round(reference_median * fraction_high)`. |
| `bit_depth` | int | `12` | `{8,10,12,14,16}` | LUT ceiling; sets histogram length (`2**bit_depth` bins) and the validation max. |
| `bit_depth_strict` | bool | `True` (dataclass) / **`False`** via `make_config` | — | `True` → raise `BitDepthError` if `image.max()` exceeds ceiling; `False` → warn only. **Use `False` for rescaled/normalized TIFFs.** |
| `min_area` | int | `100` | ≥0 px | Remove connected objects **smaller than** `min_area` (implemented as skimage `min_size` semantics via `max_size=min_area−1`; see §6). `0` disables. |
| `max_area` | int | `0` | ≥0 px | Remove objects **larger than** `max_area` (8-connected labeling). `0` disables. |
| `opening_radius` | int | `1` | ≥0 px | Morphological opening footprint radius (disk). `0` skips. |
| `closing_radius` | int | `2` | ≥0 px | Morphological closing footprint radius (disk). `0` skips. |
| `min_hole_size` | int | `50` | ≥0 px | Fill interior holes **smaller than** `min_hole_size` (`max_size=min_hole_size−1`). `0` skips. |
| `homogeneity_gate` | bool | `False` | — | If `True`, AND the mask with a local-std gate (expensive). **Disabled by default.** |
| `homogeneity_window` | int | `7` | odd px | Window for the homogeneity gate (only if enabled). |
| `homogeneity_std_max` | float | `20.0` | ≥0 | Max local std to keep a pixel (only if enabled). |

`ThresholdConfig.__post_init__` validates required-parameter combinations at
construction time and raises `ValueError` for missing params (e.g.
`method="single", direction="below"` requires `low`).

## 5. Output — `SegmentationResult`

Dataclass with these fields:

| field | structure/type | shape | dtype | axis order | units | meaning |
|-------|----------------|-------|-------|-----------|-------|---------|
| `mask` | `np.ndarray` | `(H, W)` | `bool` | `(y, x)` | — | Post-morphology binary foreground. |
| `labels` | `np.ndarray` | `(H, W)` | `int32` | `(y, x)` | — | Connected-component labels (8-connectivity); `0` = background, `1..N` = objects. |
| `regions` | `list[dict]` | length = N objects | — | — | mixed | Per-region measurements (see below). Empty list `[]` if no objects. |
| `histogram` | `Histogram` or `None` | — | — | — | — | Full-LUT histogram — **only populated when `method="percentile"`**; `None` for all other methods (see §6). |
| `threshold_used` | `dict[str, Any]` | — | — | — | intensity | The resolved thresholds actually applied (keys vary by method: e.g. `low_resolved`, `high_resolved`, `reference_median`, `strict`, `permissive`). |
| `config` | `ThresholdConfig` | — | — | — | — | Echo of the config used. |
| `provenance` | `dict[str, Any]` | — | — | — | — | `timestamp` (UTC ISO), `input_shape`, `input_dtype`, `input_hash` (16-hex sampled hash). |

Each entry in `regions` is a dict with:

| key | type | units | meaning |
|-----|------|-------|---------|
| `label_id` | int | — | Matches the value in `labels`. |
| `area_px` | int | px | Pixel count. |
| `centroid_y` | float | px (row) | Centroid row (y). **y first.** |
| `centroid_x` | float | px (col) | Centroid col (x). |
| `mean_intensity` | float | counts | Mean of `image` inside the region. |
| `min_intensity` | float | counts | Min intensity. |
| `max_intensity` | float | counts | Max intensity. |
| `area_um2` | float | µm² | **Present only if `voxel_size` was passed**; `area_px * prod(voxel_size)`. |

## 6. Conventions & GOTCHAS (the real integration risk)

- **2-D ONLY.** `run()` hard-wires `is_3d=False` everywhere. Do not pass a 3-D
  volume — footprints are `disk`, labeling is 2-D. Loop over planes in the caller.
- **Axis order is `(y, x)` = `(row, col)`.** Centroids are reported `centroid_y`
  then `centroid_x`. If the target node system is x-first, **flip them**.
- **`voxel_size` multiplies ALL tuple elements.** For a 2-D frame pass `(dy, dx)`.
  Passing a 3-tuple silently folds a Z spacing into `area_um2`.
- **Integer input required, in raw counts.** Float dtype raises `BitDepthError`.
  Values must live in `[0, 2**bit_depth − 1]`; thresholds are specified in those
  same raw counts.
- **`bit_depth_strict` default is `True` on the raw dataclass** and raises
  `BitDepthError` when `image.max()` exceeds the bit-depth ceiling (common on
  rescaled/normalized TIFFs that fill their `uint16` container). **The headless
  adapter/`make_config` forces `bit_depth_strict=False`** so it warns instead.
  Prefer `make_config`, or set the field explicitly.
- **Histogram is computed ONLY for `method="percentile"`.** For
  `single`/`hysteresis`/`relative`, `result.histogram is None` (a deliberate
  optimization — the full bincount is a multi-GB int64 intermediate on large
  stitched frames). Don't assume `result.histogram` is always present.
- **Morphology size semantics: `max_size=value−1` reproduces a strict "< value".**
  `min_area`/`min_hole_size` remove objects/holes *strictly smaller than* the
  configured value. An object of exactly `min_area` px is **kept**.
- **Labeling & large-object removal use `connectivity=2` (8-connectivity).**
- **`clear_border` is imported but UNUSED (dead code).** No border clearing
  happens — objects touching the frame edge are **retained**. Kept verbatim per
  vendoring rules.
- **`homogeneity_gate` exists but is disabled by default** (`homogeneity_gate=False`).
  When enabled it ANDs an O(N·window²) local-std mask into the result.
- **Morphology order is fixed:** opening → closing → hole-fill → small-object
  removal → large-object removal. Opening runs before closing so isolated noise
  is killed before closing could bridge it.
- **Hysteresis `below` inverts the image** (`image.max() − image`) and remaps the
  low/high bounds — a correctness-critical trick; don't "simplify" it.
- **`__post_init__` raises `ValueError` at config construction** for missing
  required params, before `run()` is ever called.

## 7. Dependencies

All **import-time / top-level** (no lazy/gated imports in this kernel):

| pip package | import | why |
|-------------|--------|-----|
| `numpy` | `numpy` | arrays, bincount, median, searchsorted (universally safe top-level dep). |
| `scikit-image` | `skimage.filters`, `.morphology`, `.measure`, `.segmentation` | Otsu/triangle/minimum/multi-Otsu suggestions, `apply_hysteresis_threshold`, opening/closing/disk/ball, `remove_small_objects/holes`, `label`, `regionprops_table`, `clear_border` (unused). |
| `scipy` | `scipy.ndimage.generic_filter` | homogeneity-gate local std (only executed if the gate is enabled). |

No lazy imports, no TensorFlow/StarDist/csbdeep/al_dic. Nothing from
`nd2studios`. If import fails it is because `scikit-image` or `scipy` is not
installed.

## 8. Failure modes / edge cases

- **Float / non-integer `image`** → `BitDepthError` (from `validate_bit_depth`).
- **`image.max()` > bit-depth ceiling** → `BitDepthError` if `bit_depth_strict`,
  else a `warnings.warn`. Also warns if max < 5% of range (bit-depth mismatch).
- **Missing required params for the chosen method/direction** → `ValueError` at
  `ThresholdConfig(...)` construction (`__post_init__`).
- **`hysteresis` with `direction` other than `below`/`above`** → `ValueError`.
- **`between`/`outside` with `high < low`** → `ValueError`.
- **No objects survive morphology** → `mask` all-`False`, `labels` all-`0`,
  `regions == []` (empty list, not an error).
- **Empty / degenerate input** → a `(0, H)`-style empty array will typically
  bincount/label to zero regions; there is no explicit empty-frame guard, so pass
  a real 2-D frame.
- **Missing `scikit-image`/`scipy`** → `ImportError` at module import.
- **`method="relative"` with an all-zero `reference_mask`** → `np.median` of an
  empty selection yields `nan`; resolved thresholds become `nan`-derived. Ensure
  the reference mask selects pixels.

## 9. Minimal runnable example

```python
import sys; sys.path.insert(0, "pure_analysis")
import numpy as np
import histogram_threshold as ht

# synthetic 12-bit frame with a bright square
img = np.random.default_rng(0).integers(0, 4096, size=(64, 64), dtype=np.uint16)
img[20:40, 20:40] = 4000

cfg = ht.make_config(method="hysteresis", direction="above",
                     strict=3500, permissive=3000, min_area=5)
res = ht.HistogramThresholdSegmenter(cfg).run(img, voxel_size=(0.5, 0.5))

print(res.mask.shape, res.mask.dtype)   # (64, 64) bool
print(res.labels.max())                 # >= 1  (int32 label image)
print(len(res.regions))                 # number of objects
print(sorted(res.regions[0]))           # ['area_px','area_um2','centroid_x','centroid_y',
                                         #  'label_id','max_intensity','mean_intensity','min_intensity']
print(res.histogram)                    # None  (hysteresis does not build a histogram)
```

Expected: `mask` is `(64, 64) bool`; `labels` is `(64, 64) int32`; `regions` is a
list of dicts each carrying `area_um2` (because `voxel_size` was passed);
`histogram is None` for the non-percentile method.

For percentile (the one path that populates `histogram`):

```python
cfg = ht.make_config(method="percentile", direction="above",
                     percentile_high=90.0, min_area=5)
res = ht.HistogramThresholdSegmenter(cfg).run(img)
assert res.histogram is not None
```

## 10. Pipeline wiring (in the ORIGINAL ND2Studios pipeline)

- **Upstream (feeds this kernel):** the not-vendored `plane_runner.py` /
  `source_utils.py` / `histogram_threshold_pipeline.py` registry node. Those
  perform all prep the kernel does **not** do: file/ND2 I/O, channel selection,
  per-multipoint / per-timepoint / per-Z **looping**, cropping, downsampling,
  registration, and exclusion-mask application. They hand the kernel one
  already-prepared 2-D integer frame at a time (plus an optional
  `reference_mask` and `voxel_size` derived from `ND2Metadata.pixel_size_um`).
- **This kernel:** frame → `SegmentationResult` (mask + labels + regions +
  provenance) for that single plane.
- **Downstream (consumes the result):** the pipeline aggregates per-plane
  `regions` into tables and stacks the per-plane `labels`/`mask` back into the
  T/Z/multipoint volume the caller was iterating; region tables flow to
  measurement/export nodes. The target node system should replicate this
  fan-out/fan-in around the kernel — the kernel itself is strictly single-frame.

## 11. Provenance

Vendored verbatim (byte-concatenated) from branch **`Version-1.45`**:

- `nd2studios/backend/analysis/histothresh/config.py` → `ThresholdConfig`
- `nd2studios/backend/analysis/histothresh/histogram.py` → `Histogram`, `compute_histogram`, Otsu/triangle/minimum/multi-Otsu suggesters
- `nd2studios/backend/analysis/histothresh/thresholds.py` → `threshold_single/hysteresis/percentile/relative`
- `nd2studios/backend/analysis/histothresh/morphology.py` → `apply_spatial_constraints`, `homogeneity_gate`
- `nd2studios/backend/analysis/histothresh/validation.py` → `BIT_DEPTH_MAX`, `BitDepthError`, `validate_bit_depth`, `infer_bit_depth`
- `nd2studios/backend/analysis/histothresh/identifier.py` → `HistogramThresholdSegmenter`, `SegmentationResult`, `_measure_regions`, `_frame_hash`

**Not vendored:** the package `__init__.py` re-export shim; and the
registry/prep layer (`histogram_threshold_pipeline.py`, `plane_runner.py`,
`source_utils.py`).

**Edits applied** (only what the vendoring rules permit): stripped all
intra-package `from .` relative imports; collapsed the six per-file
`from __future__ import annotations` into a single top-of-file one (a
future-import must be the module's first statement — duplicates deeper are a
`SyntaxError`). No private helpers renamed (no cross-file name collisions). No
compute-path member altered or removed. Added one non-vendored convenience,
`make_config(...)`, which defaults `bit_depth_strict=False`.
