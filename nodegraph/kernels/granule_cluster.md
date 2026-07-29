# Granule clustering — integration contract

## 1. Purpose + where the real math lives

Assign every bead centroid in a 3-D point cloud to a **granule** cluster.
The user seeds a granule count `n_granules`; the kernel relaxes it by ±`relax_pct`%
into a range of candidate model orders `k`, fits a mixture model at each `k`, and
selects the best `k` by **BIC** (lower is better).

- **Real math location: in-repo.** The BIC sweep, µm scaling, X-means-style
  KMeans BIC, the `reg_covar` bump-ladder retry, and all fallback logic are
  implemented in ND2Studios itself (`nd2studios/backend/analysis/granule_cluster.py`).
- The underlying model fits are delegated to **scikit-learn**
  (`sklearn.mixture.GaussianMixture`, `sklearn.cluster.KMeans`), lazily imported
  inside the fit helpers.

## 2. Entry point

```python
def cluster_granules(
    points_zyx: np.ndarray,                       # (N, 3) float, voxel units, (z, y, x)
    voxel_size_um: Tuple[float, float, float],    # (dz, dy, dx) µm, or None -> (1,1,1)
    params: Dict[str, Any],                       # see Parameters
) -> Tuple[np.ndarray, Dict[str, Any]]:           # (labels (N,) int, info dict)
```

Single free function. No class, no state. Deterministic (fixed random seed).

## 3. Inputs

| name | Python type | array shape | dtype | axis order | units | required? / default | meaning & constraints |
|------|-------------|-------------|-------|-----------|-------|---------------------|-----------------------|
| `points_zyx` | `np.ndarray` (or array-like) | `(N, 3)` | any numeric; cast to `float64` internally | **`(z, y, x)`** per row | **voxels** | required | Bead centroids. Reshaped via `.reshape(-1, 3)` — must be exactly 3 columns. `N == 0` is allowed (early return). |
| `voxel_size_um` | `tuple[float, float, float]` or `None` | `(3,)` | float | `(dz, dy, dx)` | µm/voxel | required (pass `None` for isotropic) | Physical voxel size; used to scale voxel coords to µm before fitting so anisotropic Z does not bias the model. `None` ⇒ `(1.0, 1.0, 1.0)`. |
| `params` | `dict` | — | — | — | — | required (may be `{}`/`None`) | See Parameters table. Missing keys fall back to defaults. |

## 4. Parameters (keys of `params`)

| name | type | default | valid range / choices | semantics |
|------|------|---------|-----------------------|-----------|
| `n_granules` | int | `1` | `>= 1` (coerced to `max(1, int(...))`) | Seed / expected granule count; center of the BIC search range. |
| `relax_pct` | float | `0.0` | `>= 0.0` (coerced to `max(0.0, ...)`), interpreted as a percent | ±p% relaxation around `n_granules`. Builds `k_lo = round(n_granules*(1-p))`, `k_hi = round(n_granules*(1+p))` with `p = relax_pct/100`. `0` ⇒ single `k = n_granules`. |
| `method` | str | `"gmm"` | `"gmm"` or `"kmeans"` (anything not `"kmeans"` ⇒ GMM); lowercased+stripped | `"gmm"` = full-covariance GaussianMixture (keeps anisotropic granules whole). `"kmeans"` = spherical KMeans with an X-means-style BIC. |
| `n_init` | int | `1` | `>= 1` (coerced to `max(1, int(...))`) | Number of model restarts passed to sklearn (`n_init`). |

Note: falsy values (`0`, `""`, `None`) for any key fall back to the default via `... or default`.

## 5. Output

Returns `(labels, info)`.

| field | structure | shape | dtype | axis order | units | meaning |
|-------|-----------|-------|-------|-----------|-------|---------|
| `labels` | `np.ndarray` | `(N,)` | `int` (`int64`) | matches input row order | — | Per-point granule id in `0 .. k-1`. Never `-1`; every point is assigned. |
| `info["k"]` | `int` | scalar | int | — | — | Selected number of granules (best BIC). |
| `info["bic_by_k"]` | `dict[int, float]` | — | — | — | — | BIC score for each successfully-fit candidate `k`. May be empty on total fit failure. |
| `info["means"]` | `np.ndarray` | `(k, 3)` | `float` | `(z, y, x)` | **µm** (fitted space) | Cluster centers **in the scaled µm space**, NOT voxels. |

## 6. Conventions & GOTCHAS

- **Input rows are `(z, y, x)` in VOXELS.** The caller must supply this exact
  axis order and unit.
- **`means` are in FITTED µm SPACE, not voxels.** They equal
  `voxel_coord * (dz, dy, dx)`. To map back to voxels, divide each column by the
  corresponding `voxel_size_um` component. Do not assume voxels.
- **µm scaling is per-axis multiply**, `x_um = pts * [dz, dy, dx]` — applied
  before every fit so anisotropic Z does not dominate the Gaussians / distances.
- **`voxel_size_um=None` ⇒ isotropic `(1,1,1)`**; then µm space == voxel space
  and `means` are effectively voxels.
- **Fixed seed.** `_RANDOM_STATE = 0` is hard-coded; there is **no per-run seed
  parameter** by design. Output is deterministic for identical input.
- **k is clamped to `<= N`.** `k_hi = min(k_hi, N)`, `k_lo` clamped into
  `[1, k_hi]`. You cannot request more clusters than points.
- **BIC is lower-is-better** for both methods. The KMeans path uses a custom
  X-means-style `_kmeans_bic` deliberately built to be directly comparable to
  `GaussianMixture.bic`.
- **GMM singular-covariance retry ladder.** On `ValueError`/`FloatingPointError`
  or non-finite BIC, `reg_covar` is bumped `1e-6 -> ×100`, up to 5 tries, before
  that `k` is skipped.
- **`NOISE_LABEL = -1`** is inlined but unused by this stage; noise labelling is
  a downstream concern.

## 7. Dependencies

| pip package | import time | why |
|-------------|-------------|-----|
| `numpy` | import-time (top level) | array math, scaling, BIC arithmetic. |
| `scikit-learn` | **lazy** — imported inside `_fit_gmm` / `_fit_kmeans`, gated by `importlib.util.find_spec("sklearn")` in `_require_sklearn()` | `GaussianMixture` and `KMeans` fits. Only needed when `cluster_granules` actually runs. A friendly `ImportError` is raised if absent. |

Module import itself needs only numpy. `scikit-learn` is required at call time.

## 8. Failure modes / edge cases

- **Empty cloud (`N == 0`)**: returns `(np.zeros((0,), int), {"k": 0,
  "bic_by_k": {}, "means": (0,3) float})`. No fitting attempted.
- **All fits failed** (every candidate `k` degenerate): collapses to a single
  cluster — `labels = zeros(N)`, `info = {"k": 1, "bic_by_k": {}, "means":
  centroid}` where `means` is the `(1,3)` µm-space centroid of the cloud.
- **scikit-learn missing**: `_require_sklearn()` raises
  `ImportError("Granule clustering needs scikit-learn — pip install scikit-learn")`.
- **Wrong column count**: `.reshape(-1, 3)` raises `ValueError` if input isn't
  divisible by 3.
- **`k > N`**: cannot happen — clamped (see gotchas).

## 9. Minimal runnable example

```python
import sys; sys.path.insert(0, "pure_analysis")
import numpy as np
from granule_cluster import cluster_granules

rng = np.random.default_rng(0)
a = rng.normal([5, 5, 5],   0.5, (30, 3))   # granule near voxel (5,5,5)
b = rng.normal([5, 20, 20], 0.5, (30, 3))   # granule near voxel (5,20,20)
points_zyx = np.vstack([a, b])              # (60, 3) voxels, (z,y,x)

labels, info = cluster_granules(
    points_zyx,
    voxel_size_um=(2.0, 0.3, 0.3),          # (dz, dy, dx) µm
    params={"n_granules": 2, "relax_pct": 50, "method": "gmm", "n_init": 2},
)

# labels.shape == (60,)  dtype int
# info["k"] == 2
# info["means"].shape == (2, 3)   # in µm space
# info["bic_by_k"] keys == {1, 2, 3}  (n_granules=2 ±50% -> k in 1..3, clamped <=N)
```

## 10. Pipeline wiring (original ND2Studios flow)

- **Upstream (feeds this kernel):** a bead / spot detection stage produces the
  `(N, 3)` voxel centroid cloud (`points_zyx`); the app supplies `voxel_size_um`
  from the ND2 metadata. The caller handles all per-multipoint / per-timepoint
  looping, cropping, exclusion, and detection — none of that is inside this kernel.
- **Downstream (this kernel feeds):** the per-point `labels` and `info["means"]`
  drive the granule-separation / per-granule feature nodes (V1.70+), where
  `NOISE_LABEL = -1` may later mark points as noise. Per-granule DVC surface
  selection and "All granules" composites consume these cluster ids.

## 11. Provenance

- Vendored from `nd2studios/backend/analysis/granule_cluster.py` — branch **Version-1.45**.
- Inlined dependency: `NOISE_LABEL = -1` (originally imported from
  `nd2studios/backend/analysis/granule_types.py`; unused on the compute path).
- Vendored verbatim. Imports nothing from `nd2studios`. No UI/registry members
  existed to drop; no helpers renamed.
