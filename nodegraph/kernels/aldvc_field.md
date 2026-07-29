# `aldvc_field` — DVC (ALDVC) kernel integration contract

Interface contract for wiring the vendored Augmented Lagrangian Digital
Volume/Image Correlation kernel into a node system. Read this before calling
`run_aldvc`.

---

## 1. Purpose + where the real math lives

Given **two same-shape volumes** — a *reference* and a *deformed* one — the
kernel computes a **dense displacement field** (in voxels) on a regular subset
grid plus a **strain tensor field**. Works for 2D images `(H, W)` (DIC, 6-DOF
affine subset warp) and 3D volumes `(Z, H, W)` (DVC, 12-DOF); dimensionality is
inferred from `ndim`.

**The real math is IN-REPO**, not in a third-party package. `aldvc_field.py` is
a clean-room numpy/scipy/scikit-image port of FranckLab's MATLAB ALDVC (Yang,
Hazlett, Landauer & Franck, *Exp. Mech.* 2020), implemented natively in
ND2Studios' `nd2studios/backend/dvc/` package and vendored here verbatim.
Optional CuPy accelerates only the seed FFT (lazy import, CPU fallback).

Pipeline stages inside `run_aldvc`: **Stage 0** normalize + spline-prefilter →
**Stage 1** FFT integer seed (multigrid) → **Stage 2** outlier clean + inpaint →
**Stages 3–6** local IC-GN + ADMM global-compatibility loop → **Stage 7** strain.

---

## 2. Entry point

Primary:

```python
def run_aldvc(
    ref_vol: np.ndarray,
    def_vol: np.ndarray,
    voxel_size_um: tuple[float, ...],
    params: dict,
    progress_cb: Callable[[int], None] | None = None,
    cancelled_cb: Callable[[], bool] | None = None,
    u0_seed: np.ndarray | None = None,
    use_fft_seed: bool = True,
) -> DVCResult
```

Thin OO wrapper (identical behavior; forwards `**kwargs` such as `u0_seed`/
`use_fft_seed`):

```python
ALDVCMethod().run(ref_vol, def_vol, voxel_size_um, params,
                  progress_cb=None, cancelled_cb=None, **kwargs) -> DVCResult
```

Lower-level entry points also vendored (call these only if you are re-implementing
the orchestration; **they do NOT do Stage-0 prep** — see gotchas):

- `run_admm(ref, defm_pref, grid, u0, subset_size, *, mu, admm_iterations, ...) -> ADMMResult`
- `local_icgn(ref, defm_pref, grid, u0, subset_size, ...) -> (u_grid, F_grid, zncc_grid, iters_grid)`
- `build_grid(shape, subset_size, subset_spacing) -> Grid`
- `accumulate_incremental(grid, increments) -> [(t, u_accum_grid), ...]`
- `build_accumulated_results(grid, increment_results, voxel_size_um, *, strain_type, strain_smooth) -> {t: DVCResult}`

---

## 3. Inputs

| name | Python type | array shape | dtype | axis order | units | required? / default | meaning & constraints |
|---|---|---|---|---|---|---|---|
| `ref_vol` | `np.ndarray` | `(H,W)` or `(Z,H,W)` | any real (cast internally to f32) | `(y,x)` / `(z,y,x)`, slowest-first | intensity | required | Reference (undeformed) image/volume. Pass **RAW** — `run_aldvc` normalizes it. |
| `def_vol` | `np.ndarray` | same as `ref_vol` | any real | same as `ref_vol` | intensity | required | Deformed image/volume; **must match `ref_vol.shape` exactly**. Pass RAW. |
| `voxel_size_um` | `tuple[float,...]` | len = `ndim` | float | `(y,x)` / `(z,y,x)` | µm/voxel | required (falls back to all-1.0 if `len != ndim`) | Physical voxel size; used only for the strain cross-axis rescaling and stored on the result. Displacements stay in **voxels** regardless. |
| `params` | `dict` | — | — | — | — | required (may be `{}`; all keys defaulted) | Tunables — see §4. Unknown keys ignored. |
| `progress_cb` | `Callable[[int],None]` | — | — | — | 0–100 | optional | Progress ticks. |
| `cancelled_cb` | `Callable[[],bool]` | — | — | — | — | optional | Return `True` to abort → raises `InterruptedError`. |
| `u0_seed` | `np.ndarray` | `(ndim, *grid_shape)` | float | mesh order, component-first | voxels | optional / `None` | Warm-start displacement (a prior frame's field). Used **only** when `use_fft_seed=False` **and** its shape equals `(ndim, *grid)`; otherwise silently ignored and the FFT seed runs. |
| `use_fft_seed` | `bool` | — | — | — | — | optional / `True` | `True` = always run the FFT multigrid seed. `False` = warm-start from `u0_seed` (skips the FFT search). |

---

## 4. Parameters (`params` dict keys)

| name | type | default | valid range / choices | semantics |
|---|---|---|---|---|
| `subset_size` | int | 16 | 4–128 (even) | Edge length (voxels) of each correlation subset window. Larger = smoother/more robust, less local. |
| `subset_spacing` | int | 10 | ≥1 | Spacing (voxels) between subset centers → sets grid density (measurement resolution). |
| `search_radius` | int | 0 | ≥0 | Max residual seed displacement per axis for the FFT integer search. `0` → auto `max(4, subset_size)`. |
| `seed_levels` | int | 3 | ≥1 | Coarse-to-fine multigrid pyramid levels for the integer seed. Higher brackets larger motion; `1` = single-scale. |
| `correlation` | str | `"zncc"` | `"zncc"`, `"phase"` | Seed correlation metric. ZNCC = FFT normalized cross-correlation (robust to brightness/contrast); `phase` = phase cross-correlation. |
| `mu` | float | 1e-3 | 1e-6–1.0 | ADMM `u`-coupling penalty (augmented-Lagrangian). |
| `admm_iterations` | int | 4 | ≥0 | Outer ADMM iterations. `0` ⇒ pass-0 only = conventional local DVC + one global solve. |
| `strain_type` | str | `"infinitesimal"` | `infinitesimal`, `green-lagrange`, `almansi`, `hencky` | Strain measure derived from the displacement gradient. |
| `strain_smooth` | float | 0.0 | ≥0 | Gaussian σ applied to `û` before `∂u/∂x` (0 = off). |
| `use_gpu` | bool | False | — | Route the **seed FFT** through CuPy if importable (CPU fallback). IC-GN always runs on CPU. |
| `n_workers` | int | 0 | ≥0 | IC-GN process-pool workers. `0` = auto (`min(cores-1, 8)`); `1` = serial (cached SD/Hessian). Forced to `1` when `grid.n_nodes < 64`. |
| `icgn_tol` | float | 1e-2 | >0 | IC-GN convergence tol (radius-weighted parameter step). |
| `icgn_max_iter` | int | 100 | ≥1 | Max IC-GN iterations per subset. |
| `admm_tol` | float | 1e-2 | >0 | ADMM stop: `‖Δû‖₂/√N < admm_tol`. |
| `cc_thresh` | float | 0.5 | [-1,1] | Drop subsets whose correlation confidence is below this (→ NaN → inpaint). |
| `median_thresh` | float | 2.0 | >0 | Normalized-median-test threshold (Westerweel–Scarano) for outlier rejection. |

> `DVCParams` also carries `tracking_mode` and `newFFTSearch`, but `run_aldvc`
> itself does **not** read them — they are series-level knobs handled by the
> caller (cumulative-vs-incremental accumulation via `build_accumulated_results`,
> and per-frame warm-start via the `u0_seed`/`use_fft_seed` arguments).

---

## 5. Output — `DVCResult` (dataclass)

`d = ndim` (2 or 3); `grid = (Gy,Gx)` 2D or `(Gz,Gy,Gx)` 3D.

| field | type | shape | dtype | axis order | units | meaning |
|---|---|---|---|---|---|---|
| `dim` | int | — | — | — | — | 2 or 3. |
| `grid_coords` | ndarray | `(*grid, d)` | f64 | mesh order; last axis = component | voxels | Subset-center coordinates. `[...,0]` = slowest axis (y in 2D, z in 3D). |
| `displacement_field` | ndarray | `(*grid, d)` | f64 | mesh order; last axis = component | **voxels** | Dense displacement. `[...,0]`=y/z, `[...,1]`=x/y, `[...,2]`=x. |
| `voxel_size_um` | tuple | len `d` | float | `(y,x)`/`(z,y,x)` | µm | Stored voxel size (as passed / defaulted). |
| `strain_field` | ndarray | `(*grid, d, d)` | f64 | mesh order; last two axes = tensor `[i,j]` | dimensionless | Strain tensor of the requested measure. |
| `strain_type` | str | — | — | — | — | Echo of the `strain_type` used. |
| `qfactor` | ndarray | `(*grid,)` | f64 | mesh order | — | Per-subset final ZNCC correlation confidence in `[-1,1]` (NaN = failed subset). |
| `converged` | bool | — | — | — | — | ADMM convergence flag. |
| `iterations` | int | — | — | — | — | ADMM iterations actually run. |
| `mu`, `beta` | float | — | — | — | — | Penalty `mu` used and the L-curve-selected `beta`. |
| `method`, `notes` | str | — | — | — | — | Human-readable provenance/summary. |
| `diagnostics` | dict | — | — | — | — | `grid_shape`, `n_subsets`, `beta`, `admm_residuals`, `median_zncc`, `search_radius`, `n_workers`, `use_gpu`. |

Convenience methods: `.magnitude` (voxels), `.displacement_um()` (per-axis
`× voxel_size_um`), `.magnitude_um()`.

---

## 6. Conventions & GOTCHAS (the real integration risk)

1. **Axis order is slowest-axis-first, matching numpy array axes.** 2D = `(y,x)`,
   3D = `(z,y,x)`. Every grid coordinate and displacement **component** uses this
   same order: component `0` is `y` (2D) / `z` (3D). Do **not** feed `(x,y,...)`
   ordered data or you will silently transpose the field.
2. **`F[i,j] = ∂u_i/∂x_j`** (component `i`, derivative axis `j`), mesh order.
   Strain last-two-axes are `[i,j]` in the same convention.
3. **DOF pack/unpack layout is load-bearing.** `pack_u`/`unpack_u`
   (`u_vec[ndim*p + c]`) and `pack_F`/`unpack_F`
   (`F_vec[ndim²*p + (j*ndim + i)]`, component `i` fastest, then derivative axis
   `j`, node `p` in C-order) **must** match the sparse finite-difference operator
   `D` built in `_build_fd_operator` so that `D @ u_vec ≈ F_vec`. Do not reorder
   either side independently — they are a matched pair.
4. **Displacements come out in VOXELS**, always — never µm — regardless of
   `voxel_size_um`. To get µm, use `.displacement_um()` (multiplies each component
   by `voxel_size_um[c]`).
5. **Voxel ↔ µm contract under downsampling (CALLER'S JOB).** If the caller
   downsampled the volumes by factor `s` before calling, the returned voxels are
   *downsampled* voxels. Scale `voxel_size_um` by `s` (`voxel_size_um_effective =
   original_um_per_voxel * s`) so `.displacement_um()` / strain stay physically
   correct. The kernel has no knowledge of any caller-side downsample.
6. **2D vs 3D is inferred from `ndim`.** A volume with a singleton Z (`(1,H,W)`)
   is treated as **3D** and will misbehave — the **caller must `np.squeeze` a
   singleton-Z volume to 2D** before calling.
7. **Stage-0 prep is internal to `run_aldvc` ONLY.** `run_aldvc` normalizes both
   volumes (min/max → [0,1] f32) and spline-prefilters the deformed one. So pass
   `run_aldvc` **RAW** arrays. But `run_admm` / `local_icgn` expect
   `defm_pref` = the deformed volume **already** normalized AND
   `scipy.ndimage.spline_filter`ed (order 3); their per-iteration sampling is
   `map_coordinates(order=3, prefilter=False)`. Do not pass a raw deformed volume
   to those lower-level functions.
8. **Strain non-cubic-voxel rescaling** multiplies each `G[i,j]` by
   `voxel_i/voxel_j` — this is applied inside `compute_strain` when
   `voxel_size_um` is given; strain is therefore dimensionless and physically
   correct for anisotropic (confocal z-step ≠ xy) voxels.
9. **`n_workers > 1` uses a `ProcessPoolExecutor` (spawn on Windows)** with
   volumes in `multiprocessing.shared_memory`. The worker function
   (`_icgn_block`) is pickled by qualified name, so **the module must be
   importable in the child** — i.e. the caller's `sys.path` (and any wrapper
   `__main__` guard) must let a spawned process `import aldvc_field`. The
   `n_workers = 1` path is a plain in-process loop with no such requirement and is
   the safe default when embedding.
10. **Failed/low-confidence subsets are NaN'd then inpainted** (nearest finite
    value via EDT) before the global solve, so the returned field is dense/finite;
    `qfactor` still marks the originally-failed subsets (NaN there).
11. **Grid insetting & minimum size.** Centers are inset by `subset_size//2` from
    every border; tiny axes fall back to a midpoint, and an axis that could hold
    ≥2 centers is forced to ≥2 (the FD operator and `np.gradient` need ≥2 nodes).

---

## 7. Dependencies (pip names)

| package | why | import-time or lazy |
|---|---|---|
| `numpy` | arrays throughout | **import-time** (module top) |
| `scipy` | `ndimage` (spline_filter, map_coordinates, gradient/median filters, EDT), `sparse` + `sparse.linalg` (FD operator + factorized solve), `interpolate` (RegularGridInterpolator), `signal` (fftconvolve NCC) | **import-time** |
| `scikit-image` (`skimage`) | `registration.phase_cross_correlation` (the `correlation="phase"` seed) | **import-time** |
| `cupy` (+ `cupyx`) | GPU seed FFT when `use_gpu=True` | **lazy / optional** — imported inside `_get_backends`/`_to_host`, guarded by `importlib.util.find_spec`; absent → CPU. Not required to import or run. |

Standard-library only otherwise (`multiprocessing.shared_memory`, `concurrent.futures`,
`dataclasses`, `typing`, `os`, `contextlib`, `importlib`).

---

## 8. Failure modes / edge cases

- **Shape mismatch** `ref_vol.shape != def_vol.shape` → `ValueError`.
- **Wrong ndim** (`ndim not in {2,3}`) → `ValueError`.
- **Constant/degenerate volume** → normalized to all-zeros; featureless subsets
  are skipped (their `qfactor` stays NaN, displacement inpainted from neighbors).
- **Singleton-Z `(1,H,W)`** → treated as 3D, will produce a degenerate z-grid;
  squeeze to 2D first (gotcha 6).
- **`cancelled_cb()` returns True** → raises `InterruptedError("DVC cancelled")`.
- **Missing CuPy with `use_gpu=True`** → silent CPU fallback (no error).
- **Empty/too-small grid** (`n_nodes < 64`) → forces `n_workers=1`.
- **All-NaN local field** (every subset failed) → `inpaint_nans` returns zeros;
  result is finite but zero displacement.
- **Spawned-worker import failure** (`n_workers>1`, module not importable in
  child) → a `ProcessPoolExecutor` error; use `n_workers=1` if the embedding host
  cannot make `aldvc_field` importable in child processes.

---

## 9. Minimal runnable example

```python
import sys; sys.path.insert(0, "pure_analysis")
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift
import aldvc_field as A

rng = np.random.default_rng(0)
ref = gaussian_filter(rng.random((72, 72)).astype(np.float32), 1.5)   # (y, x)
defm = ndshift(ref, shift=(1.0, 2.0), order=3, mode="nearest")        # dy=1, dx=2

res = A.run_aldvc(
    ref, defm, voxel_size_um=(1.0, 1.0),
    params=dict(subset_size=16, subset_spacing=12,
                admm_iterations=2, seed_levels=1, n_workers=1),
)

print(res.displacement_field.shape)   # (5, 5, 2)   -> (*grid, ndim)
print(res.strain_field.shape)         # (5, 5, 2, 2)
d = res.displacement_field
print(np.nanmedian(d[..., 0]), np.nanmedian(d[..., 1]))  # ~0.999 (y), ~1.998 (x)
```

3D is identical with `(Z,H,W)` inputs and `voxel_size_um=(z,y,x)`; output
displacement is `(*grid, 3)` with component order `(z, y, x)`.

---

## 10. Pipeline wiring (in the original ND2Studios pipeline)

**Upstream (caller owns all of it — NOT vendored):**
the DVC node handler / plane_runner selects a reference frame and a deformed
frame from an ND2 multipoint/timeseries, applies any crop / downsample /
registration / exclusion mask, and squeezes singleton-Z. It then calls the kernel
**once per reference→deformed pair**. Voxel size is taken from `ND2Metadata`
(scaled by any downsample factor per gotcha 5).

**This kernel** = the `DVC (ALDVC)` node body: two prepared volumes in → one
`DVCResult` (dense voxel displacement + strain) out.

**Downstream:** for **incremental** tracking mode the per-step `DVCResult`s are
composed into cumulative fields by `build_accumulated_results` (Lagrangian
point-tracking through the increments) — also vendored here. The `DVCResult`
(displacement/strain/qfactor grids) then feeds visualization (voxel/surface
render, quiver, strain heatmaps) and the `.nd2dvc` export bundle. None of that
downstream rendering/export code is vendored — only the accumulation helper.

---

## 11. Provenance

Branch **`Version-1.45`**. Byte-copied (deps-first) from:

```
nd2studios/compute/parallel/shared_array.py   (shared_ndarray, attach_shared)
nd2studios/core/dvc_registry.py               (DVCResult, DVCParams)
nd2studios/backend/dvc/mesh.py
nd2studios/backend/dvc/outliers.py
nd2studios/backend/dvc/strain.py
nd2studios/backend/dvc/integer_search.py
nd2studios/backend/dvc/global_step.py
nd2studios/backend/dvc/icgn.py
nd2studios/backend/dvc/parallel.py
nd2studios/backend/dvc/admm.py
nd2studios/backend/dvc/engine.py
nd2studios/backend/dvc/tracking.py
nd2studios/backend/dvc/method.py
```

Dropped (UI/registry-only, not on the compute path): the `DVCMethod` ABC registry
base + `@register` decorator + `get_methods`/`get_method`, `ALDVCMethod.get_params()`
(ParamSpec UI list), and the `ParamSpec` import. No compute-path code altered; no
helper renamed (no cross-module collisions).
```
