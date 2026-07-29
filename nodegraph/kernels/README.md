# `pure_analysis/` — portable analysis math-kernels

Fifteen self-contained analysis **math-kernels**. Fourteen were vendored (byte-verbatim)
out of the ND2Studios app (branch `Version-1.45`) so they can be ported into a *different*
software's new node system without dragging along the ND2Studios GUI, its plugin
registry, or its pipeline runtime; the fifteenth (`cellsam_segment`, V2.12) is a new
adapter of the same shape around a third-party package that has no v1 counterpart.
(`mesh_raster.py`, V2.08, is new in-repo code rather than a kernel of this kind and is
deliberately not indexed here — its contract lives in its module docstring.)

Each kernel is a `<module>.py` + `<module>.md` pair:

- **`<module>.py`** — the compute code. Imports **nothing** from `nd2studios`.
  Top-level third-party deps only (numpy/scipy/etc.). Every symbol a kernel
  references is defined inside its own file (the vendor pass byte-concatenated the
  source files deps-first, stripped the intra-package/relative imports, and dropped
  only UI/registry members that were off the compute path).
- **`<module>.md`** — the **integration contract**. An 11-section document with the
  entry-point signature, input/param/output tables, the load-bearing coordinate and
  unit conventions, failure modes, a runnable example, and pipeline-wiring notes.
  **Read the `.md` before calling the kernel.** The per-kernel `.md` is authoritative;
  this README only indexes them and calls out the conventions that recur across kernels.

## What these kernels are NOT

They are **pure compute over already-prepared numpy arrays**. The **caller owns all
data preparation**. None of these responsibilities live inside the kernels:

- **crop** / ROI extraction of the working region
- **downsample** / resampling (and the matching voxel-size rescale — see conventions)
- **per-multipoint (m-position) and per-timepoint (T) looping** — kernels act on one
  frame / one volume / one ref+def pair; you drive the loops
- **image registration / drift correction** as a pre-step (the `registration` kernel
  estimates and applies transforms, but you decide when to run it)
- **exclusion / masking** of unwanted regions
- **file I/O**, metadata parsing, ND2 loading

Feed a kernel an already-cropped, already-downsampled, already-looped numpy array and
it returns a numpy result. That is the whole contract.

## How to use one

1. Copy the `<module>.py` (and its `.md`) into your target project.
2. `pip install` the kernel's third-party deps (see the **dependency matrix** below).
3. `import <module>`; read `<module>.md`; call its entry point with prepared arrays.

---

## Kernel index

| module | node name | entry point | real math | key deps |
|--------|-----------|-------------|-----------|----------|
| [`histogram_threshold`](histogram_threshold.md) | Histogram Threshold Segmenter | `HistogramThresholdSegmenter(cfg).run(image, reference_mask, voxel_size)` — `make_config(**kw)` helper | in-repo (on skimage/scipy) | numpy, scikit-image, scipy |
| [`aldvc_field`](aldvc_field.md) | DVC / ALDVC | `run_aldvc(ref_vol, def_vol, voxel_size_um, params, ...)` | in-repo (clean-room ALDVC port) | numpy, scipy, scikit-image, cupy* |
| [`track_objects`](track_objects.md) | Track Objects | `link_objects(rows, ...)` — `link_objects_with_params(...)` µm-wrapper | in-repo (no tracking pkg) | numpy, scipy, numba, pandas, sklearn*, skimage* |
| [`registration`](registration.md) | Registration | `estimate_series(...)`, `apply_series(...)`, `apply_frame(...)` | in-repo (on scipy/skimage/cv2) | numpy, scikit-image, scipy, opencv-python |
| [`bead_detect`](bead_detect.md) | Bead Detection | `detect_beads(volume_zhw, voxel_size_um, params)` | in-repo (ParticleDetector) | numpy, scipy, numba, scikit-image* |
| [`granule_cluster`](granule_cluster.md) | Granule Clustering | `cluster_granules(points_zyx, voxel_size_um, params)` | in-repo (fits via sklearn) | numpy, scikit-learn* |
| [`granule_tessellate`](granule_tessellate.md) | Granule Tessellation | `tessellate_granules(points_zyx, labels, voxel_size_um, params)` | in-repo (scipy.spatial) | numpy, scipy |
| [`granule_volume_mask`](granule_volume_mask.md) | Granule Volume Mask | `build_granule_masks(tess, shape_zhw, voxel_size_um, params)` — `tessellation_from_list(...)` adapter | in-repo | numpy, scipy |
| [`granule_boundary`](granule_boundary.md) | Granule Boundary Extraction | `extract_boundary_bands(masks_by_id, combined_labels_zhw, voxel_size_um, params)` | in-repo | numpy, scipy |
| [`dic_mesh_region`](dic_mesh_region.md) | DIC Mesh Region | `build_roi_mask(shapes, H, W)` (+ `has_region`) | in-repo (skimage.draw) | numpy, scikit-image |
| [`dic_mesh_refinement`](dic_mesh_refinement.md) | DIC Mesh Refinement | `build_roi_mask(...)`, `_refinement_policy(refinement, half_win)` | mask in-repo / refinement policy **external** | numpy, scikit-image, al-dic* |
| [`stardist_segment`](stardist_segment.md) | Segmentation (method=stardist) | `segment_frame(image, ...)` (+ `get_stardist_model`, `filter_and_relabel`) | **external** (stardist CNN + NMS) | numpy, tensorflow*, stardist*, csbdeep* |
| [`cellsam_segment`](cellsam_segment.md) | Segmentation (method=cellsam) | `segment_plane(image, ...)` (+ `get_cellsam_model`, `relabel_contiguous`, `cellsam_available`) | **external** (SAM ViT + CellFinder Anchor-DETR) | numpy, cellSAM*, torch*, dask-image* |
| [`dic_correlate`](dic_correlate.md) | DIC (pyALDIC) | `run_pyaldic_pair(...)`, `run_pyaldic_series(...)` | **external** (al-dic IC-GN + ADMM) | numpy, scipy, scikit-image, al-dic* |
| [`checkpoint`](checkpoint.md) | Checkpoint | `save_checkpoints(...)`, `load_checkpoints(...)`, `checkpoints_dir_for(...)` | **none — serialization only** | numpy |

`*` = lazy / optional dependency (see matrix). "in-repo" = the algorithm is native
ND2Studios code vendored here verbatim; "external" = this kernel is a thin adapter
around a third-party package that holds the real math.

---

## Shared conventions (the main integration risk)

These recur across kernels and are where a port most easily goes wrong. When a kernel's
own `.md` contradicts anything here, **the kernel `.md` wins** — but the defaults below
are what almost every kernel assumes.

### 1. Array axis order

Numpy is row-major and every kernel follows **slowest-axis-first** image layout. There
is no `(x, y)` array anywhere on the array side — `x` is always the **last** axis.

| context | array layout | notes |
|---------|--------------|-------|
| single 2-D frame | `(H, W)` = `(y, x)` | histogram_threshold, stardist_segment, DIC image pair |
| 3-D volume | `(Z, H, W)` = `(z, y, x)` | bead_detect, granule masks/boundary, DVC |
| time series | `(T, H, W)` | registration input; the T loop is otherwise the caller's |
| point cloud | `(N, 3)` rows = `(z, y, x)` **voxels** | bead_detect output, granule_cluster/tessellate input |

Kernel-specific twists that bite:

- **DVC / `aldvc_field`** — the displacement field is stored **slowest-axis-first**:
  component `0` = `y` in 2-D, component `0` = `z` in 3-D. Strain uses
  `F[i, j] = du_i / dx_j`. 2-D vs 3-D is inferred from `ndim`; a singleton-Z volume
  must be **squeezed to 2-D** by the caller before calling.
- **`bead_detect`** — the internal `ParticleDetector` works in native `(x, y, z)`
  detector coordinates and flips to/from the wrapper's `(z, y, x)` inside
  `_detector_coords`. The wrapper's **input volume and output cloud are both `(z, y, x)`**;
  do not pass detector-order arrays.
- **`granule_tessellate`** — **input flips to output**: `points_zyx` are `(z, y, x)`
  **voxels**, but the returned boundary `vertices_um` are world `(x, y, z)` **microns**.
  This `(z,y,x) voxel → (x,y,z) micron` flip is intrinsic to the kernel.
- **`granule_cluster`** — input rows `(z, y, x)` voxels; `info["means"]` come back
  `(z, y, x)` but in **µm-scaled fit space**, not voxels (divide by `voxel_size_um` to
  return to voxels).
- **`granule_volume_mask`** — the `tess` boundary `vertices_um` are world `(x, y, z)`
  µm (matching the tessellation output), while the output grid is `(Z, H, W)`.
- **`registration`** — shifts/translations are `(row, col) = (dy, dx)`. Warps use
  OpenCV `WARP_INVERSE_MAP` (reference→moving); register-once/apply-to-all composes
  absolute transforms.
- **DIC (`dic_correlate`)** — pyALDIC works internally in **`(x, y)`**: it interleaves
  `U = [u(=x), v(=y), ...]` on `[x, y]` coordinates. The vendored output adapter applies
  the load-bearing **`[x,y]→[y,x]` / `[u,v]→[dy,dx]` swap** so the returned `DVCResult`
  matches the `(y, x)` field convention shared with `aldvc_field`. Do not double-swap.
- **ROI / mask shapes (`dic_mesh_region`, `dic_mesh_refinement`, and registration's
  freeform ROI)** — shape vertices are **`[row, col] = [y, x]`**, `H` = rows (`y`),
  `W` = cols (`x`). Masks are boolean `(H, W)`; shapes are replayed **in order**
  (`add`/`cut`/`invert`/`clear` are stateful); an empty mask is valid.

### 2. Units — `voxel_size_um` and displacements

- **`voxel_size_um` is always `(dz, dy, dx)`** — slowest axis first, **not** `(dx, dy, dz)`.
  For a 2-D frame it is `(dy, dx)`; for the DIC nodes it is `(y, x)`. This ordering matches
  the array axis order above.
- **Displacements and shifts are computed in voxels / pixels.** Convert to microns by
  multiplying each component by the matching `voxel_size_um` entry — the kernels store
  the raw voxel/pixel field and hand you `voxel_size_um` so *you* apply the scale.
- **Downsample ↔ voxel-size contract.** Kernels never downsample. If you downsample a
  frame/volume before calling, you **must** rescale `voxel_size_um` by the same factor
  (a 2× downsample → 2× larger `dy/dx`), or every micron-space result (cluster means,
  tessellation vertices, strain, µm columns) will be wrong by that factor.

### 3. Labels, ids, and the params dict

- **Label images**: `0` = background; combined label volumes are `int32`.
- **`NOISE_LABEL = -1`** is the noise/unassigned sentinel in the granule cloud kernels.
- **Reserved `"_labels"` key**: granule mask dicts (`masks_by_id`) may carry a reserved
  `"_labels"` volume and other non-int keys; kernels skip anything that is not an int gid.
  Overlap tie-breaks resolve nearest-surface then **lowest gid**.
- **Params are passed as a plain `dict`** and read with `.get(key, default)`; missing keys
  fall back to documented defaults. Several kernels (e.g. `granule_cluster`) treat *falsy*
  values (`0`, `""`, `None`) as "use default" via the `value or default` idiom — check the
  kernel `.md` before relying on `0`/empty as a real value.
- **`track_objects` mutates in place**: `link_objects` adds `track_id`/`track_length`/
  `track_validation` keys to the input row-dicts **and returns the same list object**.

### 4. Where the real math lives (thin wrappers vs. in-repo)

Know which kernels are only glue, because those carry a heavy install and their behavior
is defined by the external package's version:

- **External-package wrappers** (real math NOT in these files):
  - `stardist_segment` → `stardist` + `tensorflow` + `csbdeep` (the star-convex CNN + NMS).
    The in-repo glue is only percentile-normalize `(1, 99.8)`, a `>1024` auto-tiling
    heuristic, one `predict_instances`, and a bincount+LUT area filter.
  - `dic_correlate` → `al-dic` / pyALDIC (`run_aldic`: IC-GN subset + ADMM over an
    adaptive quadtree FE mesh). In-repo glue is the float01/DICPara input adapter and the
    griddata-resample + axis-swap output adapter.
  - `dic_mesh_refinement`'s `_refinement_policy` → `al-dic` (`build_refinement_policy`);
    returns `None` (fail-soft) when al-dic is absent. Its `build_roi_mask` half is in-repo.
  - `cellsam_segment` → `cellSAM` + `segment_anything` + `torch` (a SAM ViT-B decoder
    prompted by CellFinder box detections). The in-repo glue is the model singleton, the
    device knob, four upstream-quirk normalizations and a contiguous relabel.
- **Pure in-repo math** (native ND2Studios code; installs are just numpy/scipy-class):
  `histogram_threshold`, `aldvc_field` (a clean-room numpy/scipy port of FranckLab ALDVC —
  *not* the same thing as the `al-dic`-backed `dic_correlate`), `track_objects`,
  `registration`, `bead_detect`, `granule_cluster` (fits delegated to sklearn),
  `granule_tessellate`, `granule_volume_mask`, `granule_boundary`, `dic_mesh_region`.
- **No analysis at all**: `checkpoint` is pure serialization (compressed NPZ + JSON
  manifest freeze/resume cache); it holds zero math.

### 5. Lazy vs. import-time dependencies

Some heavy deps are **lazy** — `import <module>` succeeds without them and only the specific
call path that needs them raises (a friendly `ImportError`, usually `find_spec`-gated).
Others are **import-time** — the module will not even import without them. This changes what
you must install just to load a kernel. See the matrix; the headline cases:

- **Lazy / optional**: `cellSAM`/`torch` (cellsam_segment — `find_spec`-gated, and
  `resolve_device("cpu")` avoids the torch import entirely),
  `tensorflow`/`stardist`/`csbdeep` (stardist_segment), `al-dic`
  (dic_correlate, dic_mesh_refinement), `cupy` (aldvc_field — GPU FFT seed only, CPU
  fallback), `scikit-learn` (granule_cluster; track_objects warm-start).
- **Import-time (must be installed to `import`)**: `numpy` (all), `scipy` +
  `numba` (bead_detect; and via `track_objects`), `pandas` (track_objects — pulled in
  because CellTracker's tracking imports it at top level), `scikit-image` (histogram_threshold;
  and registration via `skimage.draw` at module top).

---

## Dependency matrix

`IT` = required at **import** time (module won't load without it). `L` = **lazy** (module
imports fine; needed only when a specific path runs). `opt` = optional feature only (kernel
degrades gracefully / falls back if absent). Blank = not used.

| module | numpy | scipy | scikit-image | opencv | numba | pandas | scikit-learn | tensorflow | stardist | csbdeep | cupy | al-dic |
|--------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| histogram_threshold | IT | IT | IT | | | | | | | | | |
| aldvc_field | IT | IT | IT | | | | | | | | L·opt | |
| track_objects | IT | IT | L | | IT | IT | L | | | | | |
| registration | IT | L | IT | L | | | | | | | | |
| bead_detect | IT | IT | L·opt | | IT | | | | | | | |
| granule_cluster | IT | | | | | | L | | | | | |
| granule_tessellate | IT | IT | | | | | | | | | | |
| granule_volume_mask | IT | L | | | | | | | | | | |
| granule_boundary | IT | L | | | | | | | | | | |
| dic_mesh_region | IT | | L | | | | | | | | | |
| dic_mesh_refinement | IT | | L | | | | | | | | | L·opt |
| stardist_segment | IT | | | | | | | L | L | L | | |
| cellsam_segment | IT | | | | | | L·opt | | | | | |
| dic_correlate | IT | L | L | | | | | | | | | L·opt |
| checkpoint | IT | | | | | | | | | | | |

Notes:
- `stardist_segment` lists scikit-image nominally in the app, but **neither vendored
  function imports it** — only numpy is actually import-time; TF/stardist/csbdeep are lazy.
- `track_objects` is heavier to *import* than the original app code was: because
  CellTracker's `tracking.py` imports `pandas` at top level and SerialTrack's numba
  kernels decorate at module load, **`import track_objects` needs numba + pandas present**,
  even though the app imported those helpers lazily. sklearn stays lazy (POD-GPR warm start).
- `dic_correlate`'s `al-dic` transitively pulls `numba` and `PySide6` when actually run.
- `checkpoint` also uses stdlib `json`; `dic_mesh_refinement` added stdlib `importlib`.

---

## Faithfulness & smoke-test status

**Vendored byte-verbatim.** The only edits the vendor pass made were: (a) rewriting
imports — stripping every `from nd2studios…` and relative `from .` import, collapsing the
per-file `from __future__ import annotations` into one at the top, and rewiring in-file
cross-references; (b) renaming on genuine name collisions (none were needed in practice);
(c) dropping members that were strictly **off the compute path** — Qt/GUI code, plugin
`get_params()`/`ParamSpec` lists, `@Registry.register` decorators, and ABC registry bases.
**No compute-path code was altered** — the algorithms are byte-identical to `Version-1.45`.
A few kernels add a clearly-labelled *non-vendored convenience* (e.g. `histogram_threshold.make_config`
forcing `bit_depth_strict=False`, `granule_volume_mask.tessellation_from_list`); these are
documented in the respective `.md` and are additive only.

Each kernel was smoke-tested: `import` check plus a synthetic call where all deps were present.

| module | import | synthetic run | note |
|--------|:------:|:-------------:|------|
| histogram_threshold | ok | ran | all 4 threshold methods on a 64×64 uint16 frame |
| aldvc_field | ok | ran | serial 2-D + 3-D with `n_workers=2` (process-pool / shared-memory) |
| track_objects | ok | ran | all 5 linking methods; in-place mutation confirmed |
| registration | ok | ran | every entry point; translation recovered injected drift exactly |
| bead_detect | ok | ran | 3-D detection + 2-D fallback |
| granule_cluster | ok | ran | 2-granule cloud; BIC sweep; empty-cloud early return |
| granule_tessellate | ok | ran | alpha-shape / convex-hull fallback / voronoi / empty |
| granule_volume_mask | ok | ran | 600-pt sphere voxelized (~1721 vs ~1810 analytic) |
| granule_boundary | ok | ran | dilation + edt band methods |
| dic_mesh_region | ok | ran | rect + circle-cut → `(10,10)` bool mask |
| dic_mesh_refinement | ok | **partial** | mask paths ran; `_refinement_policy` → `None` (al-dic absent, correct fail-soft) |
| stardist_segment | ok | ran | 64×64 two-blob frame end-to-end (TF/stardist/csbdeep present) |
| cellsam_segment | ok | **partial** | glue ran against a stubbed `cellSAM` (singleton reuse, kwarg forwarding, the `(3,H,W)` no-cells quirk, contiguous relabel); the real SAM+CellFinder net **not run** (package not installed, weights need a DeepCell token) |
| dic_correlate | ok | **partial** | in-repo adapters ran (axis swap verified); external al-dic solver **not run** (not installed) |
| checkpoint | ok | ran | save/load round-trip; store rebuild verified |

The two **partial** rows are dependency-availability gaps, not extraction defects: both the
`al-dic`-backed correlation path (`dic_correlate`) and the al-dic refinement policy
(`dic_mesh_refinement`) are lazily imported and raise a friendly `ImportError` when the
optional package is absent, which is the intended behavior. All in-repo code on those paths
was exercised. Install `al-dic` in the target to light them up.
