# Track Objects — integration contract

Vendored kernel: `pure_analysis/track_objects.py`
Node name: **Track Objects**
Source branch: `Version-1.45`

---

## 1. Purpose & where the real math lives

Frame-to-frame **object linking**: consumes per-frame object detections
(measurement row-dicts) and assigns a stable `track_id` to each detection across
the time axis, plus `track_length` and a `track_validation` flag. Five
interchangeable linking methods are exposed through one entry point.

All math is **in-repo Python** — there is no external tracking pip package. Three
bodies, all byte-copied into the single vendored `.py`:

| Body | Source | Provides |
|---|---|---|
| Dispatch + centroid linker | `nd2studios/backend/object_tracker.py` | `link_objects`, grouping, centroid Hungarian |
| Cell-Tracker linkers | `nd2studios/backend/celltracker/tracking.py` | `solve_lap`, `link_frames`, `compute_topology_features`, `track_timeseries`, `track_fingerprint`, `track_overlap` |
| SerialTrack PTV | `nd2studios/backend/serialtrack/*.py` | `SerialTracker` (ADMM topology particle tracker) |

SerialTrack here is a from-scratch NumPy/SciPy/Numba port (not MATLAB
SerialTrack, not a pip package).

---

## 2. Entry point — signatures

Primary entry (mutates `rows` in place **and** returns the same list):

```python
def link_objects(
    rows: list[dict],
    max_displacement_px: float = 100.0,
    min_track_length: int = 2,
    min_circularity: float = 0.0,
    max_eccentricity: float = 1.0,
    max_size_diff_frac: float = 1.0,
    max_frame_gap: int = 0,
    method: str = METHOD_CENTROID,
    # --- SerialTrack only ---
    st_mode: str = "Incremental",
    st_n_neighbors: int = 25,
    st_solver: str = "Regularization",
    st_loc_solver: str = "Topology",
    st_n_neighbors_min: int = 1,
    st_smoothness: float = 0.1,
    st_outlier_threshold: float = 5.0,
    st_max_iter: int = 20,
    st_iter_stop_threshold: float = 1e-2,
    st_dist_missing: float = 5.0,
    st_use_prev_results: bool = False,
    # --- Cell-Tracker only ---
    ct_n_neighbors: int = 5,
    ct_topo_weight: float = 0.3,
    ct_area_weight: float = 0.3,
    ct_max_gap: int = 3,
    ct_min_iou: float = 0.1,
    label_masks: dict[tuple[str, int], np.ndarray] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[dict]
```

Convenience wrapper (translates a node param-dict; also converts a µm distance to
pixels via `pixel_size_um`):

```python
def link_objects_with_params(
    rows: list[dict],
    params: dict,
    pixel_size_um: float | None = None,
    label_masks: dict[tuple[str, int], np.ndarray] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[dict]
```

Method constants (pass one to `method=`):

```python
METHOD_CENTROID       = "Centroid (nearest-neighbor)"
METHOD_SERIALTRACK    = "SerialTrack (topology PTV)"
METHOD_CT_TOPOLOGY    = "Cell-Tracker: Topology (Hungarian)"
METHOD_CT_FINGERPRINT = "Cell-Tracker: Spatial Fingerprint"
METHOD_CT_OVERLAP     = "Cell-Tracker: Mask Overlap (IoU)"
TRACKING_METHODS      = [ ...all five... ]
```

Pure leaf math (also exported, callable standalone):

```python
solve_lap(cost: np.ndarray, no_match_cost: float)
    -> (matches: list[(prev,curr)], unmatched_prev: list[int], unmatched_curr: list[int])
link_frames(centroids_prev, centroids_curr, max_dist=30.0,
            topo_prev=None, topo_curr=None, topo_weight=0.3, no_match_cost=None)
    -> (matches, unmatched_prev, unmatched_curr)
compute_topology_features(centroids: (N,2), n_neighbors=5) -> (N, 2*n_neighbors)
track_timeseries(df, max_dist=30.0, n_neighbors=5, use_topology=True,
                 topo_weight=0.3, no_match_cost=None, progress_cb=None) -> DataFrame(+track_id)
track_fingerprint(df, max_dist=30.0, area_weight=0.3, max_gap=3,
                  no_match_cost=None, progress_cb=None) -> DataFrame(+track_id)
track_overlap(df, masks: (T,H,W) int, min_iou=0.1, max_gap=1,
              no_match_cost=None, progress_cb=None) -> DataFrame(+track_id)
```

---

## 3. Inputs

### `rows` — list of measurement dicts (string-keyed)

| Key | Python type | required? | meaning & constraints |
|---|---|---|---|
| `segmentation_channel` | str | required* | groups detections; coerced via `str(...)`, default `""` if absent |
| `frame` | int | required | 0-based time index; coerced via `int(...)`, default `0` |
| `centroid_y_px` | float | required | object centroid **row (y)**, pixels; `None`→`0.0` |
| `centroid_x_px` | float | required | object centroid **col (x)**, pixels; `None`→`0.0` |
| `area_px` | float | required | object area in pixels; `None`→`0.0` (size gate) |
| `m_position` | int | optional | multipoint index; groups detections; default `0` |
| `label_id` | int | optional | per-frame integer mask id; **required** for CT_TOPOLOGY / CT_FINGERPRINT / CT_OVERLAP; default `0` |
| `circularity` | float | optional | `4π·area/perimeter²`, 0–1; morphology filter input |
| `eccentricity` | float | optional | 0–1; morphology filter input |

\* `segmentation_channel` is not hard-required but is one of the two grouping
keys; omitting it lumps everything into group `("", 0)`.

Rows may carry any number of extra keys — they are preserved untouched.

### `label_masks` — only for `METHOD_CT_OVERLAP`

| Item | type | shape | dtype | axis order | meaning |
|---|---|---|---|---|---|
| key | `tuple[str, int]` | — | — | — | `(segmentation_channel, m_position)` — must match a group |
| value | `np.ndarray` | `(T, H, W)` | integer label image | `T` = frame, then `(y, x)` | pixel value = the object's `label_id` in that frame |

For a single-group run, a mis-keyed dict with exactly one entry is tolerated (the
sole value is used). When a group's masks are missing, CT_OVERLAP silently
**falls back** to the fingerprint linker.

### `progress_cb`

`Callable[[float, str], None]` — fraction in `[0,1]` + status message. Optional.

---

## 4. Parameters

| Name | type | default | range / choices | effect |
|---|---|---|---|---|
| `method` | str | `METHOD_CENTROID` | the 5 `METHOD_*` constants | selects the linker |
| `max_displacement_px` | float | 100.0 | > 0, pixels | hard max centroid link distance / SerialTrack field-of-search / CT max link dist |
| `min_track_length` | int | 2 | ≥ 1 | tracks shorter than this get `track_id=None` |
| `min_circularity` | float | 0.0 | 0–1 | rows below are excluded from tracking (0 disables) |
| `max_eccentricity` | float | 1.0 | 0–1 | rows above are excluded (1.0 disables) |
| `max_size_diff_frac` | float | 1.0 | 0–1 | **centroid only**: `|Δarea|/max(area)` gate (1.0 disables) |
| `max_frame_gap` | int | 0 | ≥ 0 | **centroid only**: missed frames a track may bridge (0 = must re-detect next frame) |
| `st_mode` | str | `"Incremental"` | `"Incremental"` \| `"Cumulative"` | SerialTrack link-to-previous vs link-to-first |
| `st_n_neighbors` | int | 25 | ≥ 2 | SerialTrack topology descriptor neighbor count (`n_neighbors_max`) |
| `st_solver` | str | `"Regularization"` | `"MLS"` \| `"Regularization"` \| `"ADMM"` | SerialTrack global-step solver |
| `st_loc_solver` | str | `"Topology"` | `"Topology"` \| `"Histogram then Topology"` | SerialTrack local matcher |
| `st_n_neighbors_min` | int | 1 | ≥ 1 | floor of the exponential neighbor decay |
| `st_smoothness` | float | 0.1 | ≥ 0 | global smoothing strength |
| `st_outlier_threshold` | float | 5.0 | ≥ 0 | Westerweel normalized-median-residual cutoff (0 disables) |
| `st_max_iter` | int | 20 | ≥ 1 | max ADMM iterations per frame pair |
| `st_iter_stop_threshold` | float | 1e-2 | ≥ 0 | ADMM convergence threshold on the disp-update norm |
| `st_dist_missing` | float | 5.0 | ≥ 0 | ghost-particle cull distance `ε_d` (px), late iterations |
| `st_use_prev_results` | bool | False | — | warm-start predictor; POD-GPR stage (7+ frames) **needs scikit-learn** |
| `ct_n_neighbors` | int | 5 | ≥ 1 | **CT_TOPOLOGY**: rotation-invariant descriptor neighbors |
| `ct_topo_weight` | float | 0.3 | 0–1 | **CT_TOPOLOGY**: topology cost weight vs raw distance |
| `ct_area_weight` | float | 0.3 | 0–1 | **CT_FINGERPRINT**: area-similarity weight vs distance |
| `ct_max_gap` | int | 3 | ≥ 0 | **CT_FINGERPRINT / CT_OVERLAP**: frames a track may vanish and re-link |
| `ct_min_iou` | float | 0.1 | 0–1 | **CT_OVERLAP**: min mask IoU to link |

Params that don't apply to the chosen method are accepted and ignored.

---

## 5. Output

`link_objects` returns the **same** `list[dict]` object it was given (identity
preserved), with three keys **added/overwritten on every row**:

| Key | dtype | meaning |
|---|---|---|
| `track_id` | `int` or `None` | stable track identity; `None` if the row was excluded (morphology filter) or its final track spans fewer than `min_track_length` frames |
| `track_length` | `int` | number of frames the row's track spans (`1` if untracked/excluded) |
| `track_validation` | `str` or `None` | `"unvalidated"` when `track_id is not None`, else `None` |

`track_id`s are **1-based and globally unique across all `(channel, m_position)`
groups** (a shared counter continues across groups).

The `track_*` leaf functions (`track_timeseries`, `track_fingerprint`,
`track_overlap`) instead return a **copy** of the input DataFrame with a
`track_id` column (`int`, `-1` = unassigned). `solve_lap` / `link_frames` return
plain Python index lists/tuples; `compute_topology_features` returns an
`(N, 2*n_neighbors)` float array (first block = sorted neighbor distances, second
= sorted angular gaps).

---

## 6. Conventions & GOTCHAS (the real integration risk)

1. **Mutate-in-place contract.** `link_objects` edits the dicts you pass and
   returns the *same list*. Do not assume a fresh copy. If you need to preserve
   the originals, deep-copy the rows before calling.
2. **Axis order is `(y, x)` = `(row, col)`.** Centroids are read from
   `centroid_y_px` / `centroid_x_px` and stacked as `[y, x]` internally. Feed
   `compute_topology_features` / `link_frames` an `(N, 2)` array in `[y, x]`
   order to match.
3. **Units are pixels.** All distances (`max_displacement_px`, SerialTrack
   `f_o_s`, CT `max_dist`) are pixels. The GUI wrapper `link_objects_with_params`
   is the *only* place a µm→px conversion happens (`max_distance / pixel_size_um`
   when `distance_unit == "µm"`).
4. **Grouping happens inside `link_objects`.** Detections are grouped by
   `(str(segmentation_channel), int(m_position))`; linking runs independently per
   group; `track_id`s never collide across groups (shared mutable `next_id`
   counter). Do **not** pre-split by channel/position — pass all rows at once.
5. **`None`→`0.0` coercion.** `centroid_y_px`, `centroid_x_px`, `area_px` are read
   through helpers that map `None`/missing to `0.0`. A row with a real centroid at
   the origin and a row with missing centroids are indistinguishable.
6. **CT_OVERLAP `label` must equal the mask id.** For every row,
   `label_id` must equal the integer value painted for that object in
   `masks[frame]`. Objects present in the mask but absent from `rows` (filtered
   out upstream) are ignored so they can't steal a link. `masks` must be `(T,H,W)`
   and integer; `masks.ndim != 3` raises `ValueError`.
7. **CT_OVERLAP mask fallback.** If `label_masks` has no entry for a group (and
   isn't a single-entry dict), that group silently falls back to
   `track_fingerprint` — you still get tracks, but not IoU-based ones.
8. **Two-frame minimum.** Every linker early-returns (no links) when a group spans
   fewer than 2 frames with detections; those rows stay `track_id=None`.
9. **Morphology pre-filter.** Rows failing `min_circularity` / `max_eccentricity`
   are dropped from tracking entirely (stay `track_id=None`) — they are not merely
   down-weighted.
10. **Frame index doubles as a time axis AND (for overlap) a mask index.** For
    CT_OVERLAP, `frame` values index into `masks` (`masks[frame]`); frames `>=
    masks.shape[0]` are skipped.
11. **`min_track_length` is applied last.** All methods link first; the caller-side
    post-pass then nulls `track_id` for tracks shorter than `min_track_length` (but
    still records their `track_length`).
12. **Logging collision (cosmetic).** All vendored modules share one rebound `log`
    logger — log records emit under a single logger name. No effect on results.

---

## 7. Dependencies

| Package | pip name | when needed | why |
|---|---|---|---|
| numpy | `numpy` | **import-time** | arrays everywhere |
| scipy | `scipy` | **import-time** | `linear_sum_assignment`, `cKDTree`, sparse solve, interpolators, `ndimage` |
| numba | `numba` | **import-time** | JIT kernels in SerialTrack `detection.py` / `matching.py` are `@nb.njit`-decorated at module load |
| pandas | `pandas` | **import-time** | `celltracker/tracking.py` imports it at top level; the CT/overlap linkers build DataFrames |
| scikit-image | `scikit-image` | lazy (runtime) | only inside SerialTrack `ParticleDetector` PSF deconvolution path — **not reached** by the coordinate/row-dict tracking path |
| scikit-learn | `scikit-learn` | **lazy** (runtime) | only SerialTrack POD-GPR warm start (`st_use_prev_results=True` on 7+ frames); a clear `RuntimeError` is raised if missing |

Import-time set to run *any* method: `numpy scipy numba pandas`. (numba is
import-time purely because the SerialTrack detection/matching kernels decorate at
load — even the non-SerialTrack methods can't import the module without it.)

---

## 8. Failure modes / edge cases

- **Empty `rows`** → returned unchanged (each row would have gotten the three keys,
  but there are none).
- **Single frame / single detection per group** → no links; all rows `track_id=None`.
- **`min_track_length` too high** → everything nulled to `track_id=None`.
- **CT_OVERLAP with wrong mask shape** → `ValueError("track_overlap needs (T,H,W) masks…")`.
- **CT_OVERLAP masks missing for a group** → silent fallback to fingerprint linker.
- **SerialTrack `st_use_prev_results=True`, 7+ frames, no scikit-learn** →
  `RuntimeError` telling the user to install scikit-learn or turn the option off.
- **numba / pandas not installed** → `ImportError` at *module import* (they are
  top-level in the vendored sources), before any method runs.
- **Degenerate geometry** (fewer points than needed for a simplex / topology
  descriptor) → interpolators fall back to nearest-neighbour and topology features
  return zeros; no crash.

---

## 9. Minimal runnable example

```python
import numpy as np
import track_objects as T   # from pure_analysis/

# 3 objects drifting +2 px/frame across 4 frames, one channel, one position
base = np.array([[10., 10.], [50., 50.], [90., 20.]])
rows = []
for fr in range(4):
    for lbl, (cy, cx) in enumerate(base + fr * 2.0, start=1):
        rows.append(dict(
            segmentation_channel="GFP", m_position=0, frame=fr, label_id=lbl,
            centroid_y_px=float(cy), centroid_x_px=float(cx), area_px=100.0,
            circularity=0.9, eccentricity=0.2,
        ))

T.link_objects(rows, max_displacement_px=20.0, min_track_length=2,
               method=T.METHOD_CENTROID)

ids = sorted({r["track_id"] for r in rows if r["track_id"] is not None})
print(ids)                     # -> [1, 2, 3]   (3 tracks, each length 4)
print(rows[0]["track_length"]) # -> 4
print(rows[0]["track_validation"])  # -> "unvalidated"

# CT_OVERLAP needs an (T,H,W) integer label image whose ids == label_id:
masks = np.zeros((4, 120, 120), np.int32)
for fr in range(4):
    for lbl, (cy, cx) in enumerate(base + fr * 2.0, start=1):
        y0, x0 = int(cy), int(cx)
        masks[fr, y0:y0+8, x0:x0+8] = lbl
rows2 = [dict(r) for r in rows]      # link_objects mutates — copy first
T.link_objects(rows2, max_displacement_px=20.0, min_track_length=2,
               method=T.METHOD_CT_OVERLAP, ct_min_iou=0.1,
               label_masks={("GFP", 0): masks})
```

Expected: all five `METHOD_*` values yield `track_id`s `[1, 2, 3]` on this input.

---

## 10. Pipeline wiring (original ND2Studios flow)

- **Upstream (feeds this kernel):** a segmentation node (StarDist / Nuclei /
  granule) produces per-frame integer label masks, then a **measurement** node
  (`compute_measurements`) turns each labeled object into a row-dict with
  `segmentation_channel`, `frame`, `m_position`, `label_id`, `centroid_y_px`,
  `centroid_x_px`, `area_px`, `circularity`, `eccentricity`. Those row-dicts are
  exactly the `rows` list. For `METHOD_CT_OVERLAP`, the same segmentation masks are
  passed through as `label_masks[(channel, m_position)] = (T,H,W)`.
- **This node:** groups rows by `(channel, m_position)`, links per group, writes
  `track_id` / `track_length` / `track_validation` back onto each row in place.
- **Downstream (consumes the output):** Review / if-else pipeline steps and track
  visualisation read `track_id` to draw trajectories and to gate per-track logic.
  The caller owns *all* prep — file I/O, per-multipoint/per-timepoint iteration,
  cropping, downsampling, registration, exclusion — none of that is in this kernel.

---

## 11. Provenance

Byte-copied verbatim from branch **`Version-1.45`**, in this file order:

```
nd2studios/backend/serialtrack/config.py
nd2studios/backend/serialtrack/outliers.py
nd2studios/backend/serialtrack/matching.py
nd2studios/backend/serialtrack/detection.py
nd2studios/backend/serialtrack/regularization.py
nd2studios/backend/serialtrack/fields.py
nd2studios/backend/serialtrack/prediction.py
nd2studios/backend/serialtrack/trajectories.py
nd2studios/backend/serialtrack/tracking.py
nd2studios/backend/celltracker/tracking.py
nd2studios/backend/object_tracker.py
```

`regularization.py`, `fields.py`, and `prediction.py` were **not** in the
original "vendor these" list but are transitively imported by SerialTrack's
`tracking.py` (the ADMM global-step solvers, strain fields, and POD-GPR warm
start), so they were vendored too to satisfy the zero-relative-import rule.

Edits (import-only): eleven `from __future__ import annotations` collapsed to one
at top; all `from .` relative imports and the two `from nd2studios.backend.*`
lazy imports removed (every referenced symbol is defined in-file). Third-party
import strategy kept verbatim: numba/scipy/pandas top-level, scikit-learn lazy.
No name collisions required renaming; no UI/registry/compute-path members dropped.
```
