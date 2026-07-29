# `dic_correlate` — DIC (pyALDIC) integration contract

## 1. Purpose + where the real math lives

Runs **2D Augmented-Lagrangian Digital Image Correlation** on an already-prepared
reference/deformed image pair (or ordered series) and returns a dense
displacement field resampled onto a regular grid as a `DVCResult`.

**The real correlation math is EXTERNAL.** Local IC-GN subset matching + the
global ADMM solve over an *adaptive quadtree finite-element mesh* live entirely
inside the third-party **`al-dic` (pyALDIC)** package
(`al_dic.core.pipeline.run_aldic`), which this module imports **lazily**.

The **in-repo math** vendored here is only the thin shell around it:
- **input adapter** — normalize image to float64 `[0,1]`; map param dict → an
  `al_dic` `DICPara` with pow2/even snapping; rasterize an ROI mask.
- **output adapter** — de-interleave pyALDIC's `U=[u,v,...]` and
  `scipy.griddata`-resample the scattered FE-mesh result onto a regular grid,
  applying the load-bearing `[x,y]→[y,x]` / `[u,v]→[dy,dx]` axis swap.

## 2. Entry points

```python
run_pyaldic_pair(
    ref_img: np.ndarray,              # (H, W)
    def_img: np.ndarray,              # (H, W)
    voxel_size_um: Tuple[float, ...], # (y, x)
    params: Dict[str, Any],
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    *,
    roi_mask: Optional[np.ndarray] = None,      # (H, W)
    refinement: Optional[Dict[str, Any]] = None,
) -> DVCResult                                   # cumulative field for def vs ref

run_pyaldic_series(
    images: Sequence[np.ndarray],               # [ref, def1, def2, ...] each (H,W)
    masks: Optional[Sequence[np.ndarray]],      # per-image (H,W) or None
    params: Dict[str, Any],
    voxel_size_um: Tuple[float, ...],           # (y, x)
    *,
    reference_mode: str = "accumulative",       # "accumulative" | "incremental"
    refinement: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> List[Dict[str, DVCResult]]                  # len == len(images) - 1

# Helper (optional): build a boolean ROI mask from serializable vector shapes.
build_roi_mask(shapes: Optional[Sequence[Dict]], H: int, W: int) -> np.ndarray  # (H,W) bool

# Availability probe (no import side effects):
al_dic_available() -> bool
```

## 3. Inputs

| name | Python type | shape | dtype | axis order | units | required? | meaning & constraints |
|------|-------------|-------|-------|------------|-------|-----------|-----------------------|
| `ref_img` / `def_img` | `np.ndarray` | `(H, W)` | any numeric (uint16/float…) | `[y, x]` | intensity | required | Reference & deformed grayscale frames, same shape. A `(C,H,W)` array is auto-collapsed (max if ≤4 chans, else mid slice). Normalized internally to float64 `[0,1]`. |
| `images` | `Sequence[np.ndarray]` | each `(H, W)` | any numeric | `[y, x]` | intensity | required | `images[0]` is the reference; ≥2 required. |
| `masks` | `Sequence[np.ndarray]` or `None` | each `(H, W)` | numeric/bool | `[y, x]` | — | optional | Per-image ROI. `None` → all-ones (no masking). Cast to **float64** at the al_dic boundary. |
| `roi_mask` (pair) | `np.ndarray` or `None` | `(H, W)` | numeric/bool | `[y, x]` | — | optional | Single mask applied to BOTH ref & def. |
| `voxel_size_um` | `Tuple[float, ...]` | len 2 | float | `(y, x)` | µm/px | required | Physical pixel size; stored on the result, applied by `DVCResult.displacement_um()`. **Not** applied to the raw field. |
| `refinement` | `Dict` or `None` | — | — | — | optional | Adaptive-mesh refinement spec (see §4); best-effort, silently ignored if al_dic lacks the module. |

## 4. Parameters (`params` dict)

| name | type | default | valid range / choices | semantics |
|------|------|---------|-----------------------|-----------|
| `winsize` | int | 40 | ≥2, forced **even** | IC-GN subset size (px). |
| `winstepsize` | int | 16 | snapped to nearest **power of 2**, ≥2 | Grid pitch between subset centers (px). Becomes the output grid `step`. |
| `winsize_min` | int | 8 | snapped **pow2**, clamped ≤ `winstepsize` | Min adaptive element size. |
| `init_guess_mode` | str | `"auto"` | al_dic-defined (e.g. `auto`) | Initial-guess strategy for IC-GN. |
| `mu` | float | 1e-3 | >0 | ADMM penalty weight. |
| `tol` | float | 1e-2 | >0 | Convergence tolerance. |
| `admm_max_iter` | int | 3 | ≥1 | ADMM outer iterations. |
| `icgn_max_iter` | int | 100 | ≥1 | IC-GN inner iterations. |
| `disp_smoothness` | float | 5e-4 | ≥0 | Displacement regularization. |
| `strain_smoothness` | float | 1e-5 | ≥0 | Strain regularization. |
| `compute_strain` | bool | True | — | Passed to `run_aldic` (this adapter does NOT store strain; `strain_field=None`). |
| `reference_mode` (series arg) | str | `"accumulative"` | `accumulative` \| `incremental` | Cumulative-vs-per-step referencing (native to pyALDIC). |

`refinement` dict shape: `{"criteria": {"mask_boundary": bool, "roi_edge": bool, "brush": bool}, "brush": <(H,W) array>, "min_element_size": int}`.

## 5. Output — `DVCResult` (dataclass)

For a pair: one `DVCResult`. For a series: `List[Dict[str, DVCResult]]`, one dict
per deformed frame, keys `"primary"` (cumulative — from pyALDIC `U_accum`, falling
back to `U`) and `"increment"` (per-step `U`).

| field | shape | dtype | axis order | units | meaning |
|-------|-------|-------|------------|-------|---------|
| `dim` | scalar | int | — | — | always `2`. |
| `grid_coords` | `(Gy, Gx, 2)` | float64 | `[y, x]` | pixels (downsampled) | Regular-grid node coordinates; `[...,0]=y`, `[...,1]=x`. |
| `displacement_field` | `(Gy, Gx, 2)` | float64 | `[y, x]` | pixels | `[...,0]=dy`, `[...,1]=dx`. Convert to µm via `.displacement_um()`. |
| `strain_field` | — | — | — | — | `None` (downstream derives strain from displacement). |
| `voxel_size_um` | len-2 tuple | float | `(y, x)` | µm/px | as supplied. |
| `converged` | scalar | bool | — | — | always `True` (adapter does not read per-frame convergence). |
| `iterations`,`mu` | scalar | int/float | — | — | echoed from params. |
| `method` | str | — | — | — | `"pyALDIC"`. |
| `notes` | str | — | — | — | `"cumulative"` or `"increment"`. |
| `diagnostics` | dict | — | — | — | `engine`, `n_nodes`, `grid_shape=(Gy,Gx)`, `winsize`, `winstepsize`, `reference_mode`. |

Convenience: `.magnitude` `(Gy,Gx)` in px; `.displacement_um()` scales by
`voxel_size_um`; `.magnitude_um()` in µm.

## 6. Conventions & GOTCHAS (real integration risk)

- **Axis swap is LOAD-BEARING.** pyALDIC uses node coords `[x, y]` and
  interleaved `U=[u0,v0,u1,v1,...]` with `u = x-displacement`, `v = y-displacement`.
  `_frame_to_dvcresult` de-interleaves and **swaps to `DVCResult` `[y,x]`/`[dy,dx]`**.
  If you bypass this adapter and read `U` directly, remember it is x-major.
- **Masks are float64 at the al_dic boundary, NOT bool.** They are cast via
  `.astype(np.float64)`. `use_masks` is **auto-detected** as
  `any(mask.min() < 1.0)` — an all-ones mask disables masking even if passed.
- **`um2px` is forced to `1.0`.** Displacement stays in (downsampled) **pixels**;
  the µm conversion is deferred to `DVCResult.displacement_um()` via
  `voxel_size_um`. Do not double-apply pixel size.
- **Grid ≠ image.** The output grid spans only the FE-mesh node bounding box
  (ROI-limited), with pitch = the snapped `winstepsize`. `grid_coords` are in
  the (possibly downsampled) pixel space the caller fed in.
- **Snapping.** `winstepsize`/`winsize_min` are snapped to powers of two;
  `winsize` is forced even. Your requested values may be silently adjusted —
  read them back from `diagnostics`.
- **Scattered→regular resample.** `griddata` linear interpolation (needs ≥4
  nodes) with nearest-neighbor hole-fill. Fewer than 4 nodes → nearest only.
- **`run_pyaldic_series` returns `len(images) - 1` entries** (no self-pair for
  the reference).
- **CALLER OWNS ALL PREP** — no file I/O, no multipoint/timepoint looping, no
  crop/downsample/registration/exclusion here. Feed final `(H,W)` arrays.

## 7. Dependencies

| package | import-time or lazy | why |
|---------|---------------------|-----|
| `numpy` | import-time | arrays throughout. |
| `scipy` | **lazy** (inside `_interp_to_grid`) | `scipy.interpolate.griddata` scattered→regular resample. Needed whenever a result is adapted. |
| `scikit-image` | **lazy** (inside `_rasterize_region`) | `skimage.draw` disk/ellipse/polygon — only if you use `build_roi_mask`. |
| `al-dic` (pyALDIC) | **lazy** (inside `_require_al_dic` / `_refinement_policy`) | the actual DIC solver. **Transitively pulls `numba` and `PySide6>=6.6`.** Absent → friendly `ImportError` only when you call `run_pyaldic_*`; module import still succeeds. |

Install the solver: `pip install al-dic`.

## 8. Failure modes / edge cases

- **`al-dic` not installed** → `ImportError` with install hint, raised only on
  `run_pyaldic_pair`/`run_pyaldic_series` call (import & adapters still work).
- **< 2 images** → `ValueError("AL-DIC needs at least 2 images …")`.
- **User cancel** (`cancelled_cb()` truthy) → surfaces as `RuntimeError` /
  empty series; pair convenience raises `RuntimeError` if no result produced.
- **Flat image** (`max <= min`) → normalized to all-zeros (solver may not track).
- **Degenerate `U`** (`u.size < 2*n`) → per-node displacement defaults to 0.
- **Empty/missing ROI shapes** → `build_roi_mask` returns all-False; caller
  decides whether "no ROI" means full frame.
- **Bad `refinement` policy** → swallowed, treated as no refinement.

## 9. Minimal runnable example

```python
import sys; sys.path.insert(0, "pure_analysis")
import numpy as np, dic_correlate as d

H = W = 256
ref = (np.random.rand(H, W) * 4095).astype(np.uint16)
# a synthetic 3-px x-shift as the "deformed" frame:
defo = np.roll(ref, 3, axis=1)

# requires al-dic installed; otherwise raises a friendly ImportError:
res = d.run_pyaldic_pair(ref, defo, voxel_size_um=(0.5, 0.5),
                         params={"winsize": 40, "winstepsize": 16})
print(res.grid_coords.shape)          # (Gy, Gx, 2)
print(res.displacement_field.shape)   # (Gy, Gx, 2)  -> [dy, dx] in px
print(res.displacement_um().shape)    # same, scaled by voxel_size_um
```

Adapter-only check (no al-dic needed) — verifies the axis swap:

```python
class M: coordinates_fem = np.array([[0,0],[10,0],[0,10],[10,10],[5,5]], float)
U = np.zeros(10); U[0::2] = 1.0; U[1::2] = 2.0   # u(x)=1, v(y)=2
r = d._frame_to_dvcresult(M(), U, step=5, voxel_size_um=(1,1),
                          method="pyALDIC", params={}, reference_mode="accumulative")
# r.displacement_field[...,0] ~= 2 (dy),  [...,1] ~= 1 (dx)
```

## 10. Pipeline wiring (original ND2Studios)

- **Feeds in:** the caller (page/plane-runner) supplies final `(H,W)` frames
  after channel selection, optional downsample, crop, and registration; an
  optional ROI mask (built from drawn vector shapes via `build_roi_mask`); and
  the ND2 `pixel_size_um` as `voxel_size_um`. A "refine" node may supply the
  `refinement` dict.
- **Feeds out:** the `DVCResult` (`(Gy,Gx,2)` displacement in px + `voxel_size_um`)
  goes to the DVC panel / downstream strain & visualization, which derive strain
  from the displacement field and convert to µm on demand.

## 11. Provenance (branch `Version-1.45`)

- `nd2studios/backend/dic/engine.py` — `run_pyaldic_pair`, `run_pyaldic_series`,
  `_frame_to_dvcresult`, `_interp_to_grid`, `_regular_grid`, `_mesh_coords`,
  `_build_dicpara`, `_to_float01`, `_snap_pow2`, `_refinement_policy`,
  `_require_al_dic`, `al_dic_available`, `_INSTALL_HINT` (verbatim).
- `nd2studios/backend/dic/roi.py` — `build_roi_mask`, `_rasterize_region`,
  `has_region`, `OP_ADD`, `OP_CUT` (verbatim).
- `nd2studios/core/dvc_registry.py` — `DVCResult` dataclass only (verbatim).

**Dropped (UI/registry-only, not on the compute path):** `DVCMethod` ABC,
`DVCParams`, the `@DVCMethod.register` registry, and the `ParamSpec` import from
`dvc_registry`. **No helper renames** were needed (no name collisions).

**DEVIATION from verbatim (2026-07-27, al-dic ≥0.7 compat):** `_build_dicpara` now sets
`gridxy_roi_range` to the full image extent `(gridx=(0, W), gridy=(0, H))`. al-dic 0.7.x
requires this ROI range EXPLICITLY when `run_aldic()` is called directly — it defaults to a
zero-size box and otherwise raises `ValueError("No grid points generated … ROI is empty")`.
The solver insets the range by `winsize//2` (`integer_search`: `min_x = max(gridx[0], w//2)`,
`max_x = min(gridx[1], W-1-w//2)`) and applies `use_masks` itself, so the full-image default is
correct and the mask still restricts correlation. A future ROI-bbox tightening (grid limited to
the mask's bounding box) is an optional optimization. Verified: recovers a planted (1,3)px shift
on a 128² speckle pair (`nodegraph.selftest:test_catalog_dic`).
