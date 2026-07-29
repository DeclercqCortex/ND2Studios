# Registration — integration contract

Kernel module: `pure_analysis/registration.py`
Node name: **Registration**

---

## 1. Purpose & where the real math lives

Estimate per-frame spatial transforms that align every frame of a `(T, H, W)`
single-channel time series onto a common anchor (drift correction /
stabilization), then resample **any** channel through those transforms. The
design is *register-once, apply-to-all*: estimate on one reference channel, then
apply the identical transform bundle to every other channel so colocalization is
preserved.

**The real math is in-repo** — plain Python over numpy plus standard library
primitives. There is no external math package. It delegates to:

- `scipy.ndimage` — sub-pixel resampling (`shift`) and band-pass whitening (`gaussian_filter`)
- `skimage.registration.phase_cross_correlation` — sub-pixel translation
- `skimage.feature.ORB`, `skimage.measure.ransac`, `skimage.transform.*` — feature model
- `cv2` (OpenCV) — `findTransformECC`, `warpAffine` / `warpPerspective`

Three transform families: **translation** (phase correlation, the workhorse),
**euclidean/affine** (ECC, seeded by a phase-correlation translation), and
**feature** (ORB + RANSAC, for large motion / rotation / scale).

---

## 2. Entry points

Primary (series-level):

```python
estimate_series(
    series: np.ndarray,               # (T, H, W)
    model: str = "translation",       # "translation" | "euclidean" | "affine" | "feature"
    reference: str = "previous",      # "first" | "previous" | "mean" | "template"
    upsample: int = 20,
    highpass_sigma: float = 2.0,
    min_confidence: float = 0.0,
    normalize: str = "none",          # "none" | "zscore"
    roi: Optional[Dict[str, Any]] = None,
    feature_transform: str = "affine",  # "euclidean" | "similarity" | "affine"
    min_inliers: int = 8,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]                   # the transform bundle (see §5)

apply_series(
    series: np.ndarray,               # (T, H, W) — any channel
    transforms: Dict[str, Any],       # bundle from estimate_series
    interp_order: int = 1,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> np.ndarray                       # (T, H, W), dtype preserved

apply_frame(
    frame: np.ndarray,                # (H, W) single plane
    transforms: Dict[str, Any],
    t: int,                           # which frame's transform to apply
    interp_order: int = 1,
) -> np.ndarray                       # (H, W), dtype preserved
```

Leaf functions (usable standalone, pairwise):

```python
estimate_translation(reference, moving, upsample=20, highpass_sigma=2.0,
                     window=True, mask=None, bbox=None) -> (shift (2,) float64, ncc float)
ecc_align(reference, moving, model="euclidean", init_shift=None, iters=200,
          eps=1e-6, gauss=5, interp_order=1, mask=None) -> (warp (2,3) float32, cc float, aligned)
estimate_features(reference, moving, transform="affine", mask=None, bbox=None,
                  n_keypoints=800, min_inliers=8, residual_threshold=2.0) -> (warp (2,3) float32, conf float)
apply_shift(image, shift, order=1) -> np.ndarray           # (H,W), dtype preserved
apply_warp(image, warp_matrix, motion=None, output_shape=None, interp_order=1) -> np.ndarray
common_translation_crop(shifts, shape, inset_edges=True) -> (y0,y1,x0,x1) | None
```

The typical node wiring is: `estimate_series` on the reference channel →
`apply_series` on every channel with the returned bundle.

---

## 3. Inputs

### `estimate_series`

| name | Python type | array shape | dtype | axis order | units | required? / default | meaning & constraints |
|---|---|---|---|---|---|---|---|
| `series` | `np.ndarray` | `(T, H, W)` | any (int or float; computed in float, dtype irrelevant here) | T, row(y), col(x) | pixels | **required** | Single-channel time series. Must be 3-D or raises `ValueError`. Caller must have already done all prep (I/O, per-M/per-Z extraction, crop, downsample, exclusion). |
| `roi` | `dict` \| `None` | — | — | — | pixels | default `None` (whole frame) | See §4 (ROI spec). Restricts *estimation* to a region; the transform is still applied full-frame. |
| `progress_cb` | callable | — | — | — | — | default `None` | Called with an int 0–100. |
| `cancelled_cb` | callable | — | — | — | — | default `None` | Returns `True` to stop early; remaining frames keep identity transforms. |

### `apply_series` / `apply_frame`

| name | Python type | array shape | dtype | axis order | units | required? / default | meaning & constraints |
|---|---|---|---|---|---|---|---|
| `series` | `np.ndarray` | `(T, H, W)` | any | T, y, x | pixels | **required** (apply_series) | Any channel; the same `T` as the bundle. Output dtype = input dtype. |
| `frame` | `np.ndarray` | `(H, W)` | any | y, x | pixels | **required** (apply_frame) | One plane. |
| `transforms` | `dict` | — | — | — | — | **required** | The exact bundle returned by `estimate_series` (uses its `shifts` / `warps`). |
| `t` | `int` | — | — | — | — | **required** (apply_frame) | Frame index; out-of-range → frame returned unchanged. |

### leaf `reference` / `moving` (estimate_translation, ecc_align, estimate_features)

| name | Python type | array shape | dtype | axis order | units | required? | meaning |
|---|---|---|---|---|---|---|---|
| `reference` | `np.ndarray` | `(H, W)` | any | y, x | pixels | required | The fixed image. |
| `moving` | `np.ndarray` | `(H, W)` | any | y, x | pixels | required | Aligned onto `reference`. |
| `mask` | `np.ndarray` bool \| `None` | `(H, W)` | bool | y, x | — | optional | Freeform ROI restriction (full-frame coords, not cropped). |
| `bbox` | tuple `(y0,y1,x0,x1)` \| `None` | — | int | y, x | pixels | optional | Rectangular ROI crop (half-open, row/col). |

---

## 4. Parameters

| name | type | default | valid range / choices | semantics |
|---|---|---|---|---|
| `model` | str | `"translation"` | `translation`, `euclidean`, `affine`, `feature` | DOF of the transform. `translation` → phase correlation (returns `shifts`, `warps=None`). `euclidean`/`affine` → ECC seeded by phase correlation (returns 2×3 `warps`). `feature` → ORB+RANSAC (returns 2×3 `warps`). |
| `reference` | str | `"previous"` | `first`, `previous`, `mean`, `template` | Anchor selection. `first` = every frame → frame 0. `previous` = cumulative (each frame → prior), composed to absolute. `mean` = every frame (incl. 0) → series mean. `template` = two-pass (rough-stabilize → average) anchor, robust to bleaching/atypical anchor. |
| `upsample` | int | `20` | ≥ 1 | Phase-correlation sub-pixel factor; accuracy ≈ 1/`upsample` px. |
| `highpass_sigma` | float | `2.0` | ≥ 0 (0 = off) | Gaussian band-pass whitening sigma before correlation; suppresses DC / uneven illumination. |
| `min_confidence` | float | `0.0` | typically 0.0–1.0 | Frames whose confidence (NCC or ECC cc or RANSAC inlier fraction) falls below this are **gated**. 0 = never gate. See §6 for gating behavior. |
| `normalize` | str | `"none"` | `none`, `zscore` | `zscore` per-frame normalizes for *estimation only* (counters photobleaching); geometry unaffected. |
| `roi` | dict \| None | `None` | see below | Restricts estimation to a region; applied full-frame. |
| `feature_transform` | str | `"affine"` | `euclidean`, `similarity`, `affine` | Model class fit by RANSAC (only when `model="feature"`). |
| `min_inliers` | int | `8` | ≥ 3 | Minimum RANSAC inliers (and pre-filter match count) for a `feature` fit; below → identity + conf 0. |
| `interp_order` | int | `1` | 0 nearest, 1 bilinear, 3 cubic | Resampling order in `apply_series`/`apply_frame`/`apply_shift` (cv2 warps use LINEAR for ≥1 else NEAREST). |

**ROI spec** (`roi` argument → `roi_to_mask`):

- `{"kind":"rect", "x":int, "y":int, "w":int, "h":int}` → yields both a box mask
  **and** a bbox. Translation uses the **bbox** (sub-pixel crop); ECC/feature use the **mask**.
- `{"kind":"shapes", "shapes":[{"type":"rect"|"ellipse"|"polygon", "vertices":[[y,x], ...]}, ...]}`
  → rasterized freeform mask, **no bbox**. Freeform paths are the masked /
  `inputMask` correlation routes. `rect`/`ellipse` use 2 vertices (opposite
  corners / bounding box); `polygon` uses ≥3 `[y,x]` vertices.
- falsy / unknown `kind` → whole frame.

---

## 5. Output

`estimate_series` returns a **dict bundle** (the "transforms"):

| field | type | shape | dtype | axis order | units | meaning |
|---|---|---|---|---|---|---|
| `model` | str | — | — | — | — | Echo of the `model` argument. |
| `reference` | str | — | — | — | — | Echo of the `reference` argument. |
| `shifts` | `np.ndarray` | `(T, 2)` | float64 | (row, col) = (y, x) | pixels | **Absolute** per-frame translation. For non-translation models this is the warp's translation part `(warp[1,2], warp[0,2])` (useful for plotting). |
| `warps` | `np.ndarray` \| `None` | `(T, 2, 3)` | float32 | — | — | `None` for `model="translation"`. Otherwise absolute per-frame 2×3 affine mapping **reference → moving** (see §6). |
| `confidence` | `np.ndarray` | `(T,)` | float64 | — | — | Per-frame score: NCC (translation), ECC cc (euclidean/affine), or RANSAC inlier fraction (feature). Frame 0 (anchor) = 1.0 for anchored modes. |
| `gated` | `np.ndarray` | `(T,)` | bool | — | — | `True` where `confidence < min_confidence` and the transform was held/skipped. |

`apply_series` → `(T, H, W)` array, **same dtype** as its input series.
`apply_frame` → `(H, W)` array, same dtype as input frame.
`common_translation_crop` → `(y0, y1, x0, x1)` int tuple (half-open, row/col) or
`None` if drift leaves no common overlap.

---

## 6. Conventions & GOTCHAS (the real integration risk)

1. **Shifts are `(row, col)` = `(y, x)`** — numpy axis order, matching skimage.
   NOT `(x, y)`. `shifts[t]` is the vector applied to the moving frame to align
   it onto the reference (`reference ≈ apply_shift(moving, shift)`).

2. **ECC / feature warps map reference → moving** and MUST be applied with
   `cv2.WARP_INVERSE_MAP`. `apply_warp` (and hence `apply_series`) already does
   this. Do not re-invert the matrix yourself.

3. **`estimate_series` returns ABSOLUTE per-frame transforms.** In `previous`
   mode the incremental frame-to-frame transforms are composed
   (translations summed via `cum_shift`; warps composed via homogeneous
   3×3 multiply `_homog(warp) @ h_abs`) into absolute transforms. This is what
   lets the **same bundle apply to every channel** (register-once/apply-to-all).
   Do not attempt to re-accumulate in the caller.

4. **Confidence gating differs by mode.** With `min_confidence > 0`:
   - **Absolute modes** (`first`/`mean`/`template`): a gated frame **holds the
     last-good transform** (`last_shift` / `last_warp`), so one bad frame does
     not snap to identity or poison neighbors.
   - **`previous` (cumulative)**: a gated frame **skips the increment** (keeps
     the running `cum_shift` / `h_abs` unchanged), so drift is not corrupted by a
     bad pairwise estimate.
   `gated[t]` records which frames were gated.

5. **Anchor / start frame.** `first` and `previous` leave frame 0 as the fixed
   anchor (identity, `start=1`). `mean` and `template` register **every** frame
   including frame 0 (`start=0`), because they have no natural anchor frame.

6. **ROI: rect vs shapes.** A rect ROI drives translation via a **sub-pixel
   bbox crop** but drives ECC/feature via a **mask** (keeps rotation center
   correct). Freeform `shapes` produce a mask only, and **masked phase
   correlation is integer-pixel only** (not sub-pixel) — a documented accuracy
   tradeoff for freeform regions.

7. **`normalize="zscore"` affects estimation only** — the geometry (and thus the
   applied resampling) is identical; it only changes which correlation peak is
   found on bleaching series.

8. **`common_translation_crop` is translation-only and optional.** It returns
   the largest axis-aligned rectangle that is real (non-padded) data in every
   frame after `apply_shift`. It is meaningless for rotation/affine footprints
   (their valid region is not an axis-aligned rectangle) — do not apply it to
   warp-model output. `inset_edges=True` trims 1 px off each padded side to avoid
   bilinear edge fuzz. Pass every channel's shifts stacked as `(N,2)` if you want
   the intersection across channels.

9. **Dtype is preserved end-to-end.** All math runs in float; results are clipped
   to the integer range and cast back (`_cast_like`), so `apply_series` on a
   uint16 channel returns uint16 (no wrap-around from interpolation overshoot).

10. **Border handling.** Resampling zero-pads vacated borders
    (`mode="constant"`, `cval=0` / `BORDER_CONSTANT`). Aligned frames therefore
    have black margins where content moved in — use `common_translation_crop`
    (translation) to trim them.

11. **`apply_series` / `apply_frame` short-circuit identity.** A zero shift or an
    `eye(2,3)` warp returns the frame untouched (no resampling, no interpolation
    blur) — so unregistered frames are bit-exact.

---

## 7. Dependencies

| pip package | import name | why | when needed |
|---|---|---|---|
| numpy | `numpy` | arrays, linear algebra | **import-time** (top level) |
| scikit-image | `skimage` | `phase_cross_correlation`, ORB/RANSAC, `draw.ellipse`/`draw.polygon` (ROI raster) | **import-time** — `skimage.draw` is imported at top of this module (matches the manual_mask source); other skimage submodules are imported lazily inside functions. |
| scipy | `scipy` | `ndimage.shift` (resample), `ndimage.gaussian_filter` (high-pass) | **lazy** — imported inside the functions that use them. |
| opencv-python | `cv2` | ECC (`findTransformECC`) + affine/perspective warping | **lazy** — imported inside ECC/warp functions; only needed for `model` in {euclidean, affine} and `apply_warp`. |

All four are top-level project deps of ND2Studios. Only numpy and scikit-image
are needed at *import* time; scipy and cv2 are pulled in only when the relevant
code path runs (per the verbatim import strategy of the sources).

---

## 8. Failure modes / edge cases

- **Non-3-D `series`** → `ValueError("estimate_series expects (T,H,W) ...")`.
  Likewise `apply_series`. Caller must reduce Z / select a channel first.
- **Unknown `model` / `reference`** → `ValueError` listing valid choices.
- **Blank / no-texture frame** (`std < 1e-6` after windowing) →
  `estimate_translation` returns `(zeros(2), 0.0)`; combined with
  `min_confidence` this gates the frame rather than jumping.
- **ECC non-convergence** (`cv2.error`) → `ecc_align` returns the seed warp,
  `cc=0.0`, and the *unaligned* moving frame; with `min_confidence>0` the frame
  is gated. (Observed on pure-noise synthetic input — a graceful fallback, not a
  crash.)
- **Feature model, too few keypoints/matches/inliers** → `estimate_features`
  returns `(eye(2,3), 0.0)` (identity + zero confidence).
- **ROI degenerate** (empty rect, or shapes that rasterize to nothing) →
  `roi_to_mask` returns `(None, None)` → whole-frame estimation.
- **`common_translation_crop` with drift exceeding the frame** → returns `None`.
- **Out-of-range `t` in `apply_frame`** → returns the frame unchanged.
- **Missing optional dep**: if cv2 is absent, only the ECC/affine + `apply_warp`
  paths fail (at call time); translation and feature-detection paths still work.

---

## 9. Minimal runnable example

```python
import sys; sys.path.insert(0, "pure_analysis")
import numpy as np
from scipy.ndimage import shift as nd_shift
import registration as reg

# Synthetic (T,H,W) series with known drift (2 px down, 1.5 px left per frame)
H = W = 64; T = 4
base = np.random.default_rng(0).random((H, W)).astype(np.float32)
series = np.stack([nd_shift(base, (2.0 * t, -1.5 * t), order=1) for t in range(T)])

# Estimate on the reference channel, apply to (this or any other) channel
tf = reg.estimate_series(series, model="translation", reference="first")
# tf["shifts"].shape == (4, 2);  tf["warps"] is None;  tf["confidence"].shape == (4,)
print(np.round(tf["shifts"][1], 2))   # -> [-2.  1.5]   (row, col) that undoes frame-1 drift

aligned = reg.apply_series(series, tf)        # (4, 64, 64), dtype preserved
one     = reg.apply_frame(series[2], tf, 2)   # (64, 64)
crop    = reg.common_translation_crop(tf["shifts"], (H, W))  # (y0,y1,x0,x1) or None
```

Expected shapes: `aligned` → `(4, 64, 64)`; `one` → `(64, 64)`;
`tf["shifts"]` → `(4, 2)`; `tf["confidence"]`/`tf["gated"]` → `(4,)`.

---

## 10. Pipeline wiring (original ND2Studios pipeline)

- **Upstream (feeds this kernel):** the caller supplies an already-prepared
  `(T, H, W)` single-channel array — after ND2 load, per-multipoint (M) and
  per-Z selection/projection, optional crop/downsample, and any exclusion. In
  the app this prep is done by the pipelines page / plane_runner / source_utils
  (NOT vendored here). Registration is estimated on one chosen **reference
  channel**.
- **This kernel:** `estimate_series` (reference channel) → transform bundle →
  `apply_series` on **every** channel with that one bundle. `apply_frame` powers
  the paused single-plane preview so a cropped preview shows drift-corrected
  data.
- **Downstream (what it feeds):** the aligned channel dict flows to the rest of
  the recipe / analysis (projection, export, tracking, DVC, etc.). For the
  translation model, `common_translation_crop` optionally trims the shared
  border so all frames are equal-size, recentred, and padding-free before export.

---

## 11. Provenance

Branch: **Version-1.45**. Vendored verbatim (byte-copied) from:

- `nd2studios/backend/registration/estimate.py` — entire module body
  (`estimate_series`, `apply_series`, `apply_frame`, `estimate_translation`,
  `ecc_align`, `estimate_features`, `apply_shift`, `apply_warp`,
  `common_translation_crop`, `stabilize`, `roi_to_mask`, `_build_template`,
  `_normalize_series`, `_cast_like`, `_cv_motion`, `_homog`, `REFERENCE_MODES`,
  `MODELS`).
- `nd2studios/backend/stitch/register.py` — helpers `_highpass`, `_hann2d`,
  `_ncc` only (the stitch module itself was not imported: its top level pulls in
  `StitchConfig` / `Dataset`).
- `nd2studios/backend/analysis/manual_mask.py` — `rasterize_shapes` + its
  private helper `_rasterize` only (for the freeform `shapes` ROI branch).

Edits: removed the two `nd2studios` imports (symbols now defined in-file); no
renames (no collisions); dropped UI/registry-only members from manual_mask
(`ManualMaskPipeline`, `get_params`, `@AnalysisPipeline.register`, tiled-mask /
polygon-edit helpers) and stitcher-only members from register
(`_overlap_boxes`, `_pair_shift`, `_candidate_pairs`, `_solve_axis`,
`refine_positions`). `method.py` / `RegistrationResult` were not vendored.
