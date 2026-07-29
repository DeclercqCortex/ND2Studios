# stardist_segment — integration contract

StarDist 2-D-per-frame nuclear segmentation kernel, vendored from ND2Studios.

---

## 1. Purpose + where the real math lives

Segment nuclei in a **single, already-prepared 2-D frame** with StarDist
(Schmidt et al., MICCAI 2018), returning an integer label image plus StarDist's
detection `details`. An optional area filter / relabel helper is included.

**The real math is external.** The star-convex-polygon detection and
non-maximum suppression live entirely inside the `stardist` + `tensorflow` +
`csbdeep` packages (the StarDist2D CNN and its trained weights). This is NOT
in the ND2Studios repo. The in-repo (vendored) math is thin glue:

1. csbdeep percentile normalization to the `(1, 99.8)` percentile window;
2. auto-tiling heuristic: tile only if the larger image dimension `> 1024`,
   using `(H//512, W//512)` tiles (each clamped to `>= 1`);
3. one `StarDist2D.predict_instances(...)` call;
4. `filter_and_relabel`: a `np.bincount` + lookup-table area filter and
   contiguous relabel (a separate helper, NOT called inside `segment_frame`).

This kernel is **2-D-per-frame**. All looping (T / Z / multipoint), file I/O,
crop/downsample/registration/exclusion, and measurement extraction are the
caller's responsibility.

---

## 2. Entry points

```python
def segment_frame(
    image: np.ndarray,
    model=None,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.3,
    scale: float | None = None,
    model_name: str = "2D_versatile_fluo",
    disable_gpu: bool = False,
) -> tuple[np.ndarray, dict]: ...

def get_stardist_model(
    model_name: str = "2D_versatile_fluo",
    disable_gpu: bool = False,
):  # -> stardist.models.StarDist2D
    ...

def filter_and_relabel(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
) -> np.ndarray: ...
```

`get_stardist_model` returns a cached module-global `StarDist2D` singleton.
`filter_and_relabel` is a standalone post-processing step on any label image.

---

## 3. Inputs

### `segment_frame`

| name  | Python type      | shape  | dtype               | axis order | units  | required? / default | meaning & constraints |
|-------|------------------|--------|---------------------|------------|--------|---------------------|-----------------------|
| image | `np.ndarray`     | `(H,W)`| any numeric (int/float; internally normalized to float) | `(row=Y, col=X)` | intensity (a.u.) | **required** | Single 2-D frame — one channel, one Z, one T. Must be exactly 2-D (`img_norm.shape` is unpacked as `h, w`). |
| model | `StarDist2D` or `None` | — | — | — | — | default `None` | Pass a pre-loaded model to reuse it; `None` loads/uses the module singleton. |

### `get_stardist_model`

| name        | type   | default              | meaning |
|-------------|--------|----------------------|---------|
| model_name  | `str`  | `"2D_versatile_fluo"`| Pretrained StarDist2D bundle name (downloaded/cached by stardist). |
| disable_gpu | `bool` | `False`              | If `True`, clears `CUDA_VISIBLE_DEVICES` **before** the first TF import. |

### `filter_and_relabel`

| name     | Python type  | shape   | dtype               | units | required? | meaning & constraints |
|----------|--------------|---------|---------------------|-------|-----------|-----------------------|
| mask     | `np.ndarray` | `(H,W)` | integer label image | label ids | required | Labels `0`=background, `1..N`=objects. Non-contiguous ids are fine. |
| min_area | `int`        | scalar  | int                 | pixels | required | Keep objects with pixel count `>= min_area`. |
| max_area | `int`        | scalar  | int                 | pixels | required | Keep objects with pixel count `<= max_area`. |

---

## 4. Parameters

| name        | type            | default              | valid range / choices | semantics (effect on output) |
|-------------|-----------------|----------------------|-----------------------|------------------------------|
| prob_thresh | `float`         | `0.5`                | `[0, 1]`              | Object-probability threshold. Higher → fewer, higher-confidence detections. Passed straight to `predict_instances`. |
| nms_thresh  | `float`         | `0.3`                | `[0, 1]`              | Non-maximum-suppression IoU threshold. Higher → allows more overlap between kept objects. |
| scale       | `float \| None` | `None`               | `> 0` or `None`       | Rescale factor applied by StarDist before inference (upsample small nuclei / downsample large). `None` = no rescale. Output labels are returned at the ORIGINAL `(H,W)`. |
| model_name  | `str`           | `"2D_versatile_fluo"`| any stardist bundle   | Which pretrained net. `2D_versatile_fluo` targets fluorescent nuclei. |
| disable_gpu | `bool`          | `False`              | —                     | Force CPU inference (see gotchas — only effective before first TF import). |

---

## 5. Output

### `segment_frame` → `(labels, details)`

| field   | structure | shape  | dtype   | axis order | units | meaning |
|---------|-----------|--------|---------|------------|-------|---------|
| labels  | `np.ndarray` | `(H,W)` | `int32` (as returned by StarDist) | `(Y, X)` | label ids | `0`=background; each nucleus gets a unique id `1..N`. Ids are contiguous as produced by StarDist. Same `(H,W)` as input even when `scale` is set. |
| details | `dict`    | —      | —       | —          | —     | StarDist detection metadata. Keys observed: `prob` (per-object probability, `(N,)`), `points` (per-object centroid `(N, 2)` in `(y, x)`), `coord` (per-object polygon vertices `(N, 2, n_rays)` in `(y, x)`). Exact keys depend on the stardist version. |

### `filter_and_relabel` → `np.ndarray`

| field | shape | dtype | meaning |
|-------|-------|-------|---------|
| out   | same as input `mask` | `int32` | Objects whose pixel area is outside `[min_area, max_area]` are set to `0`; survivors are relabeled `1..K` in ascending original-label order. |

---

## 6. Conventions & GOTCHAS (the real integration risk)

- **Axis order is `(H, W)` = `(row=Y, col=X)`** everywhere. `details['points']`
  and `details['coord']` are in **`(y, x)` = (row, col)** order — the opposite
  of `(x, y)`. Flip if your target expects `(x, y)`.
- **Strictly 2-D input.** `segment_frame` does `h, w = img_norm.shape`; a 3-D
  array (a stack, an `(H,W,1)`, or an RGB `(H,W,3)`) raises `ValueError`. The
  caller must have already reduced to one plane.
- **Normalization is fixed at `(1, 99.8)` percentiles** inside `segment_frame`
  — the caller's own scaling does not change relative contrast much but extreme
  outliers will move the window. There is no way to change the percentiles
  without editing the kernel.
- **Auto-tiling triggers when `max(H, W) > 1024`**, using `(H//512, W//512)`
  tiles. Results are numerically equivalent to untiled but this is a memory,
  not a correctness, knob.
- **`scale` rescales for inference only** — output `labels` come back at the
  original `(H, W)`.
- **Model singleton is module-global and NOT thread-safe.** `get_stardist_model`
  caches `_stardist_model` / `_stardist_model_name` at module scope and reuses
  it whenever `model_name` matches. If two threads request different
  `model_name`s concurrently they will race on the globals. The original app
  parallelizes by **process** (each process loads its own model), never by
  thread. If the target node system multithreads, either serialize model access
  or load a per-call `model` and pass it in explicitly.
- **`disable_gpu` and threading config are ONE-SHOT and global.** TF inter/intra-op
  parallelism is pinned to `1` thread on the first model load
  (`_ensure_tf_threading`), and `CUDA_VISIBLE_DEVICES` is cleared *before* the
  first TF import. Once TF is initialized, later `disable_gpu`/threading changes
  have no effect (`_tf_configured` guard returns early). Whatever the first call
  requests wins for the life of the process.
- **Windows `os.symlink` monkeypatch.** `get_stardist_model` calls
  `_patch_windows_symlink()`, which on `win32` replaces `os.symlink` process-wide
  with a copy-tree fallback (csbdeep model loading uses symlinks that need
  elevation / Developer Mode on Windows). This is a global side effect on the
  `os` module — benign but worth knowing.
- **`filter_and_relabel` area is raw pixel count** (`np.bincount` over labels),
  NOT physical area. Convert µm² thresholds to pixels using the caller's pixel
  size before calling. Bounds are **inclusive** on both ends.

---

## 7. Dependencies

| package        | why                                         | when imported |
|----------------|---------------------------------------------|---------------|
| `numpy`        | arrays, bincount/LUT relabel                | **import-time** (top level) |
| `tensorflow`   | StarDist2D CNN backend                      | **lazy** — inside `_ensure_tf_threading`, only when a model is loaded |
| `stardist`     | `StarDist2D`, `predict_instances`, weights  | **lazy** — inside `get_stardist_model` |
| `csbdeep`      | `normalize` percentile normalization        | **lazy** — inside `segment_frame` |

`pip install stardist tensorflow csbdeep` (plus `numpy`). The three heavy deps
are required only when you actually segment — `import stardist_segment` and
`filter_and_relabel` work with numpy alone.

**Note:** scikit-image is a nominal dependency listed for this node in the app,
but neither vendored function actually imports or uses it. It is not required.

---

## 8. Failure modes / edge cases

- **Non-2-D `image`** → `ValueError` on the `h, w = img_norm.shape` unpack.
- **Missing heavy dep** → `ModuleNotFoundError` / `ImportError` at the first
  `segment_frame` / `get_stardist_model` call (not at import).
- **Model download fails** (no network / bad `model_name`) → error from
  `StarDist2D.from_pretrained`.
- **Windows without Developer Mode** → handled: the symlink monkeypatch copies
  the model tree instead of failing.
- **Empty / all-background mask in `filter_and_relabel`** → returns the input
  cast to `int32` unchanged (guarded via `mask.size == 0 or mask.max() == 0`).
- **`min_area > max_area`** → every object is dropped; result is all zeros.
- **Blank / featureless `image`** → valid empty result: `labels.max() == 0`,
  `details` arrays have length 0.

---

## 9. Minimal runnable example

```python
import sys; sys.path.insert(0, "pure_analysis")
import numpy as np
import stardist_segment as s

# --- pure-numpy path (no heavy deps needed) ---
mask = np.array([[0, 1, 1, 2],
                 [0, 1, 3, 3],
                 [4, 4, 4, 4]], dtype=np.int32)
out = s.filter_and_relabel(mask, min_area=2, max_area=3)
# label 1 area=3 (kept), 2 area=1 (dropped), 3 area=2 (kept), 4 area=4 (dropped)
# -> [[0 1 1 0],
#     [0 1 2 2],
#     [0 0 0 0]]   shape (3,4) int32
assert out.shape == (3, 4) and out.dtype == np.int32

# --- full StarDist path (needs stardist + tensorflow + csbdeep) ---
rng = np.random.default_rng(0)
img = rng.random((64, 64)).astype(np.float32)
img[20:30, 20:30] += 5; img[40:50, 45:55] += 5   # two bright blobs
labels, details = s.segment_frame(img)            # labels (64,64) int32
# labels.max() -> number of detected objects (2 for this synthetic input)
# details keys -> ['coord', 'points', 'prob']
```

---

## 10. Pipeline wiring (original ND2Studios flow)

Upstream of this kernel, the app does all prep:
- `nd2_loader` / source_utils read one channel as a `(T,H,W)` frame source;
- `plane_runner` iterates timepoints (and multipoints), pulling **one 2-D
  `(H,W)` frame** per call — that frame is exactly `segment_frame`'s `image`;
- crop / downsample / registration / exclusion are applied before this point.

`segment_frame` produces `(labels, details)` for that frame. Downstream:
- `filter_and_relabel` cleans the label image by area;
- the app then re-stacks per-frame labels into a `(T,H,W)` int32 label stack
  (`segment_timeseries` did this in one shot — **dropped** here) and feeds it to
  measurement / tracking nodes.

For the target node system: feed this node ONE prepared 2-D frame per invocation;
loop upstream. `filter_and_relabel` is a natural separate downstream node.

---

## 11. Provenance

Branch: **Version-1.45**.

| symbol | vendored from |
|--------|---------------|
| `segment_frame`, `get_stardist_model`, `_ensure_tf_threading`, `_patch_windows_symlink`, model-singleton globals | `nd2studios/backend/celltracker/segmentation.py` |
| `filter_and_relabel` | `nd2studios/backend/analysis/source_utils.py` |

Vendored verbatim (byte-copied); imports nothing from `nd2studios`; caller owns
all prep.

**Dropped (not on the compute path):** `segment_timeseries` (T-loop wrapper),
`source_shape` / `read_plane` (caller-side frame-source shape helpers). No
private-helper renames were needed (no name collision between the two source
modules).
