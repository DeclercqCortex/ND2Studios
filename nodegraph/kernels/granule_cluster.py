# ============================================================================
# VENDORED KERNEL — Granule clustering
# ============================================================================
# Purpose: assign each bead centroid in a 3-D point cloud to a granule cluster,
#   picking the model order (number of granules k) by BIC over a relaxed range
#   around a user-seeded count.
#
# WHERE THE REAL MATH LIVES: in-repo. The clustering (BIC sweep, GMM/KMeans fit,
#   X-means-style KMeans BIC, µm scaling, degeneracy fallbacks) is implemented
#   here in ND2Studios itself. The heavy lifting of the actual GMM/KMeans fits is
#   delegated to scikit-learn (sklearn.mixture.GaussianMixture / sklearn.cluster
#   .KMeans), lazily imported inside the fit helpers.
#
# PROVENANCE: vendored from
#   nd2studios/backend/analysis/granule_cluster.py  (branch: Version-1.45)
#   The single cross-module symbol it referenced — NOISE_LABEL from
#   nd2studios/backend/analysis/granule_types.py — is inlined below (= -1); it is
#   unused on the compute path.
#
# Vendored verbatim; imports nothing from nd2studios; caller owns all prep
#   (no file I/O, no per-multipoint/per-timepoint looping, no bead detection —
#   the caller supplies an already-extracted (N,3) voxel point cloud).
#
# DROPPED UI/registry-only members: NONE (source is already a pure math module).
# RENAMED for collision: NONE.
# ============================================================================
"""Granule clustering (V1.70 · P2) — pure, Qt-free.

Answers *"which granule does each bead belong to"* for a 3-D cloud of bead
centroids. The user seeds a granule count ``n_granules``; we relax it by ±``p``%
and let **BIC** pick the best model order over the resulting ``k`` range. The
default model is a full-covariance **Gaussian Mixture** (so an elongated /
anisotropic granule stays a single cluster instead of being split); a spherical
**KMeans** fallback is exposed for sparse / unstable clouds.

Optional dependency: ``scikit-learn`` is lazily imported and gated by
``importlib.util.find_spec`` (mirrors ``backend/serialtrack/prediction.py`` and
the cellpose/stardist precedent). It is deliberately *not* in
``requirements.txt``; a friendly :class:`ImportError` is raised when absent.

Coordinate / unit convention (P0 contract):

* ``points_zyx`` is ``(N, 3)`` in **voxel** units, ordered ``(z, y, x)``.
* ``voxel_size_um`` is ``(dz, dy, dx)`` µm. Coordinates are scaled to **µm**
  *before* fitting so anisotropic Z does not bias the Gaussians / distances.
* Returned ``means`` are in the fitted **µm** space.
* ``-1`` (:data:`granule_types.NOISE_LABEL`) is reserved for noise by downstream
  stages; this GMM/KMeans stage never emits it (every point is assigned).
"""
from __future__ import annotations

import importlib.util
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Inlined from nd2studios.backend.analysis.granule_types (import removed for
# self-containment). Reserved by downstream stages for noise; UNUSED by the math
# in this module (kept only to preserve provenance / the original '# noqa').
NOISE_LABEL: int = -1  # noqa: F401

# Fixed for reproducibility — the node exposes no per-run seed (P2 §Params).
_RANDOM_STATE: int = 0

_SKLEARN_MISSING_MSG = (
    "Granule clustering needs scikit-learn — `pip install scikit-learn`"
)

# reg_covar bump ladder for degenerate / ill-defined covariance retries.
_REG_COVAR_START = 1e-6
_REG_COVAR_FACTOR = 100.0
_REG_COVAR_TRIES = 5


def _require_sklearn() -> None:
    """Raise a friendly :class:`ImportError` if scikit-learn is unavailable."""
    if importlib.util.find_spec("sklearn") is None:
        raise ImportError(_SKLEARN_MISSING_MSG)


def _fit_gmm(
    x_um: np.ndarray, k: int, n_init: int, random_state: int
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Fit a full-covariance GMM with ``k`` components.

    Retries with a progressively larger ``reg_covar`` when the fit hits an
    ill-defined (singular / collapsed) covariance. Returns ``(bic, labels,
    means)`` in the µm space, or ``None`` if every retry failed.
    """
    from sklearn.mixture import GaussianMixture

    reg = _REG_COVAR_START
    for _ in range(_REG_COVAR_TRIES):
        try:
            gm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=n_init,
                reg_covar=reg,
                random_state=random_state,
            )
            gm.fit(x_um)
            bic = float(gm.bic(x_um))
            if not np.isfinite(bic):
                reg *= _REG_COVAR_FACTOR
                continue
            labels = gm.predict(x_um).astype(int)
            return bic, labels, np.asarray(gm.means_, dtype=float)
        except (ValueError, FloatingPointError):
            reg *= _REG_COVAR_FACTOR
    return None


def _kmeans_bic(
    x_um: np.ndarray, labels: np.ndarray, centers: np.ndarray, inertia: float
) -> float:
    """Lower-is-better BIC for a hard KMeans partition (X-means style).

    Models the partition as an equal-variance spherical Gaussian mixture with
    hard assignments, so the score is directly comparable to
    :meth:`GaussianMixture.bic` (also lower-is-better). Free parameters:
    ``k`` d-dim means + one shared variance + ``k-1`` mixing weights.
    """
    n, d = x_um.shape
    k = centers.shape[0]
    counts = np.bincount(labels, minlength=k).astype(float)

    denom = max(n - k, 1) * d
    variance = max(inertia / denom, 1e-9)

    # LL = Σ_k[ n_k log(n_k/N) - (n_k d / 2) log(2πσ²) ] - (N-k) d / 2
    nonzero = counts > 0
    log_lik = float(
        np.sum(counts[nonzero] * np.log(counts[nonzero] / n))
        - 0.5 * d * np.log(2.0 * np.pi * variance) * np.sum(counts)
        - 0.5 * max(n - k, 0) * d
    )
    n_params = k * d + 1 + (k - 1)
    return -2.0 * log_lik + n_params * np.log(max(n, 1))


def _fit_kmeans(
    x_um: np.ndarray, k: int, n_init: int, random_state: int
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Fit spherical KMeans with ``k`` clusters; returns ``(bic, labels, means)``."""
    from sklearn.cluster import KMeans

    try:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(x_um).astype(int)
        centers = np.asarray(km.cluster_centers_, dtype=float)
        bic = _kmeans_bic(x_um, labels, centers, float(km.inertia_))
        if not np.isfinite(bic):
            return None
        return bic, labels, centers
    except (ValueError, FloatingPointError):
        return None


def cluster_granules(
    points_zyx: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cluster a bead point cloud into granules by a BIC-selected mixture model.

    Parameters
    ----------
    points_zyx : (N, 3) float
        Bead centroids in **voxel** units, ordered ``(z, y, x)`` (P0 convention).
    voxel_size_um : (dz, dy, dx)
        Physical voxel size in µm. Coordinates are scaled to µm before fitting so
        anisotropic Z does not bias the model. ``None`` ⇒ isotropic ``(1, 1, 1)``.
    params : dict
        ``n_granules`` (int, seed count), ``relax_pct`` (float 0–100, the ±p%),
        ``method`` (``"gmm"`` | ``"kmeans"``), ``n_init`` (int, restarts).

    Returns
    -------
    labels : (N,) int
        Per-point granule id in ``0 .. k-1``.
    info : dict
        ``{"k": int, "bic_by_k": {k: bic}, "means": (k, 3) µm array}``.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """
    _require_sklearn()

    pts = np.asarray(points_zyx, dtype=float).reshape(-1, 3)
    n = int(pts.shape[0])

    if voxel_size_um is None:
        dz = dy = dx = 1.0
    else:
        dz, dy, dx = (float(voxel_size_um[0]),
                      float(voxel_size_um[1]),
                      float(voxel_size_um[2]))
    x_um = pts * np.array([dz, dy, dx], dtype=float)

    params = params or {}
    n_granules = max(1, int(params.get("n_granules", 1) or 1))
    relax_pct = max(0.0, float(params.get("relax_pct", 0.0) or 0.0))
    method = str(params.get("method", "gmm") or "gmm").strip().lower()
    n_init = max(1, int(params.get("n_init", 1) or 1))

    # Empty cloud → nothing to cluster.
    if n == 0:
        return (np.zeros((0,), dtype=int),
                {"k": 0, "bic_by_k": {}, "means": np.zeros((0, 3), dtype=float)})

    # BIC sweep bounds, clamped so k never exceeds the number of points.
    p = relax_pct / 100.0
    k_lo = max(1, int(round(n_granules * (1.0 - p))))
    k_hi = max(k_lo, int(round(n_granules * (1.0 + p))))
    k_hi = min(k_hi, n)
    k_lo = max(1, min(k_lo, k_hi))

    fit_one = _fit_kmeans if method == "kmeans" else _fit_gmm

    bic_by_k: Dict[int, float] = {}
    fitted: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for k in range(k_lo, k_hi + 1):
        result = fit_one(x_um, k, n_init, _RANDOM_STATE)
        if result is None:
            continue
        bic, labels, means = result
        bic_by_k[k] = bic
        fitted[k] = (labels, means)

    # Every fit failed (extreme degeneracy) → collapse to a single cluster.
    if not fitted:
        return (np.zeros((n,), dtype=int),
                {"k": 1, "bic_by_k": {},
                 "means": x_um.mean(axis=0, keepdims=True)})

    best_k = min(bic_by_k, key=lambda kk: bic_by_k[kk])
    labels, means = fitted[best_k]
    info: Dict[str, Any] = {
        "k": int(best_k),
        "bic_by_k": bic_by_k,
        "means": means,
    }
    return labels, info
