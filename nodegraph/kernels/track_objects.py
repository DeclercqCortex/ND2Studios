"""
track_objects.py — vendored "Track Objects" math kernel (ND2Studios)
====================================================================

PURPOSE
-------
Frame-to-frame object *linking*: takes per-frame object detections (measurement
row-dicts) and assigns stable ``track_id`` / ``track_length`` /
``track_validation`` fields across the time axis. Five interchangeable linking
methods are exposed through the single entry point ``link_objects``:

  * Centroid nearest-neighbour   (scipy ``linear_sum_assignment``)
  * SerialTrack topology PTV      (scale/rotation-invariant particle tracking)
  * Cell-Tracker Topology         (rotation-invariant descriptor + Jaqaman LAP)
  * Cell-Tracker Spatial Fingerprint (position + area + gap filling)
  * Cell-Tracker Mask Overlap     (IoU LAP on integer label images)

WHERE THE REAL MATH LIVES
-------------------------
All of it is IN-REPO Python (numpy/scipy/numba) — there is no external tracking
package. The three math bodies are:
  * ``nd2studios/backend/object_tracker.py``  — the group/frame dispatch + the
    centroid Hungarian linker (the top-level ``link_objects`` entry).
  * ``nd2studios/backend/celltracker/tracking.py`` — the Jaqaman birth/death LAP
    (``solve_lap``), rotation-invariant topology features, and the three
    Cell-Tracker linkers (``track_timeseries``, ``track_fingerprint``,
    ``track_overlap``).
  * ``nd2studios/backend/serialtrack/*.py`` — the full SerialTrack ADMM particle
    tracker (config, detection, matching, outliers, regularization, fields,
    prediction, trajectories, tracking) used by ``METHOD_SERIALTRACK``. Note:
    SerialTrack here is a from-scratch NumPy/SciPy/Numba port (not the MATLAB
    SerialTrack or any pip package).

PROVENANCE (branch: Version-1.45)
---------------------------------
Vendored verbatim (byte-copied) from, in file order:
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

Vendored verbatim; imports nothing from nd2studios; caller owns all prep
(no file I/O, no per-multipoint / per-timepoint looping, no crop / downsample /
registration / exclusion — the caller does those and hands over row-dicts).

EDITS MADE (see rules; every deviation documented here and in the .md)
---------------------------------------------------------------------
(a) Import rewrites only:
    * Collapsed the eleven per-file ``from __future__ import annotations`` down
      to the single one at the very top of this file (required — future imports
      must precede all other statements).
    * Removed all intra-package relative imports (``from .config import ...``,
      ``from .outliers import ...``, ``from .regularization import ...``, etc.)
      and the two ``from nd2studios.backend.serialtrack...`` /
      ``from nd2studios.backend.celltracker.tracking import ...`` lazy imports
      inside ``object_tracker``'s method-dispatch helpers. Every symbol they
      referenced is now defined earlier in this same file.
    * Third-party imports are left EXACTLY as each source had them: numba and
      scipy stay top-level (numba is imported at module top by detection.py /
      matching.py, so it is an IMPORT-TIME dependency); scikit-learn stays LAZY
      (imported inside InitialGuessPredictor, only when SerialTrack's POD-GPR
      warm start runs). pandas is imported at top level by celltracker/tracking
      (so it too is import-time here) and additionally, verbatim, lazily inside
      object_tracker's CT helpers.
(b) Name collisions resolved: NONE required. The only duplicated module-level
    names across the concatenated sources are the ``log`` logger and the
    ``ProgressCB = Callable[[float, str], None]`` alias; both simply re-bind to
    equivalent values and neither is on the compute path, so nothing was
    renamed. (Consequence: all log records now emit under the last-bound logger
    name — cosmetic only.)
(c) Dropped UI/registry-only members: NONE. These backend modules carry no
    ParamSpec/get_params/@Registry decorators or ParamEditor hooks — nothing was
    on a non-compute path to drop.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# The bodies below are byte-copied from the sources listed above. Their original
# per-file import lines remain inline (re-imports are harmless in Python); only
# the __future__ / relative / nd2studios imports were stripped as noted.
# ─────────────────────────────────────────────────────────────────────────────


# ======================================================================
# VENDORED FROM: nd2studios/backend/serialtrack/config.py
# ======================================================================

"""
SerialTrack Python — Configuration dataclasses
================================================
    serialtrack/config.py

Replaces MATLAB structs: BeadPara, MPTPara, and trajectory-merge params.

Changes from v1
----------------
- Added TrackingMode.DOUBLE_FRAME  (was missing — MATLAB 'dbf' mode)
- Added TrackingConfig.loc_solver   (was missing — MATLAB locSolver 1 or 2)
- Added TrajectoryConfig dataclass  (was entirely missing — trajectory stitching params)
- Added TrackingConfig.trajectory   (embeds TrajectoryConfig)
- Removed duplicate GlobalSolver from regularization.py — single source of truth here
- Fixed roi_slices() for the case where roi_x/roi_y are None before init
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════

class DetectionMethod(IntEnum):
    """Particle detection strategy."""
    TPT = 1       # Blob → centroid → radial symmetry sub-pixel
    TRACTRAC = 2  # LoG blob → local max → 2nd-order poly sub-pixel


class GlobalSolver(IntEnum):
    """Global step solver for ADMM iterations."""
    MLS = 1              # Moving least-squares fitting
    REGULARIZATION = 2   # Scatter → grid regularization
    ADMM = 3             # Augmented Lagrangian with L-curve


class LocalSolver(IntEnum):
    """Local step solver for particle matching."""
    TOPOLOGY = 1         # Topology-based feature matching
    HISTOGRAM_THEN_TOPOLOGY = 2  # Histogram first, then topology


class TrackingMode(IntEnum):
    """Frame-to-frame vs. all-to-reference tracking."""
    INCREMENTAL = 1
    CUMULATIVE = 2
    DOUBLE_FRAME = 3     # Independent frame-pair mode (was missing)


# ═══════════════════════════════════════════════════════════════
#  Detection config
# ═══════════════════════════════════════════════════════════════

@dataclass
class DetectionConfig:
    """Particle detection / localization parameters.

    Replaces MATLAB ``BeadPara`` struct.  All lengths are in pixels.
    Works for both 2-D and 3-D images; dimension is inferred at runtime.
    """
    method: DetectionMethod = DetectionMethod.TRACTRAC
    threshold: float = 0.4
    bead_radius: float = 3.0   # 0 → use regionprops centroid only
    min_size: int = 2          # min blob volume (3D) or area (2D) [px^d]
    max_size: int = 1000
    color: str = "white"       # foreground colour: "white" | "black"
    # Optional PSF deconvolution (Richardson-Lucy)
    psf: Optional[np.ndarray] = None
    deconv_iters: int = 6
    # TPT / radial-symmetry params (3-D only)
    win_size: Tuple[int, ...] = (5, 5, 5)
    dccd: Tuple[float, ...] = (1.0, 1.0, 1.0)
    abc: Tuple[float, ...] = (1.0, 1.0, 1.0)
    rand_noise: float = 1e-7


# ═══════════════════════════════════════════════════════════════
#  Trajectory merge config  (was entirely missing)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrajectoryConfig:
    """Parameters for post-processing trajectory segment merging.

    Replaces the MATLAB variables: distThres, extrapMethod,
    minTrajSegLength, maxGapTrajSeqLength.

    Used primarily in incremental tracking mode to stitch together
    short trajectory segments that were split due to detection gaps.
    """
    dist_threshold: float = 1.0
    """Distance threshold to connect split trajectory segments [px]."""

    extrap_method: str = "pchip"
    """Extrapolation scheme: 'pchip' for smooth motion,
    'nearest' for Brownian motion."""

    min_segment_length: int = 10
    """Minimum trajectory segment length (in frames) to attempt
    extrapolation and merging."""

    max_gap_length: int = 0
    """Maximum frame gap allowed between connected segments.
    0 means segments must be adjacent."""

    merge_passes: int = 4
    """Number of merge passes (MATLAB default: 4)."""


# ═══════════════════════════════════════════════════════════════
#  Tracking config
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrackingConfig:
    """Particle linking / tracking parameters.

    Replaces MATLAB ``MPTPara`` struct.
    """
    # --- search & matching ---
    f_o_s: float = 60.0
    n_neighbors_max: int = 25
    n_neighbors_min: int = 1

    # --- local solver (was missing) ---
    loc_solver: LocalSolver = LocalSolver.TOPOLOGY

    # --- global solver ---
    solver: GlobalSolver = GlobalSolver.REGULARIZATION
    smoothness: float = 0.1

    # --- outlier removal (Westerweel universal test) ---
    outlier_threshold: float = 5.0

    # --- ADMM iteration control ---
    max_iter: int = 20
    iter_stop_threshold: float = 1e-2

    # --- strain gauge ---
    strain_n_neighbors: int = 20
    strain_f_o_s: float = 60.0

    # --- prediction / initialisation ---
    use_prev_results: bool = False
    dist_missing: float = 5.0

    # --- mode ---
    mode: TrackingMode = TrackingMode.INCREMENTAL

    # --- trajectory merging (was missing) ---
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    # --- physical scales ---
    xstep: float = 1.0   # length-unit per pixel
    ystep: float = 1.0
    zstep: float = 1.0
    tstep: float = 1.0   # time-unit per frame

    # --- ROI (auto-set from first image) ---
    roi_x: Optional[Tuple[int, int]] = None
    roi_y: Optional[Tuple[int, int]] = None
    roi_z: Optional[Tuple[int, int]] = None   # None ⇒ 2-D
    mask: Optional[np.ndarray] = None

    # ── derived helpers ──

    @property
    def ndim(self) -> int:
        return 2 if self.roi_z is None else 3

    @property
    def steps(self) -> np.ndarray:
        """Physical pixel sizes as array [xstep, ystep(, zstep)]."""
        if self.ndim == 2:
            return np.array([self.xstep, self.ystep])
        return np.array([self.xstep, self.ystep, self.zstep])

    def init_roi_from_image(self, img: np.ndarray) -> None:
        """Set ROI to full image extent and default mask."""
        self.roi_x = (0, img.shape[0])
        self.roi_y = (0, img.shape[1])
        if img.ndim >= 3:
            self.roi_z = (0, img.shape[2])
        else:
            self.roi_z = None
        if self.mask is None:
            self.mask = np.ones(img.shape, dtype=bool)

    def roi_slices(self) -> Tuple[slice, ...]:
        """ROI as a tuple of slices for direct array indexing."""
        sx = slice(*(self.roi_x or (0, None)))
        sy = slice(*(self.roi_y or (0, None)))
        if self.roi_z is not None:
            return (sx, sy, slice(*self.roi_z))
        return (sx, sy)

# ======================================================================
# VENDORED FROM: nd2studios/backend/serialtrack/outliers.py
# ======================================================================

"""
SerialTrack Python — Outlier detection & missing-particle culling
==================================================================
    serialtrack/outliers.py

Replaces: removeOutlierTPT.m  (~60 lines)

Implements Westerweel & Scarano (2005) universal outlier detection,
missing-particle detection, and adaptive f_o_s update.

Fixes from v1
-------------
- REMOVED circular import: ``from .matching import ...`` was unused
  and created a circular dependency (matching → outliers → matching).
- Fixed logger name: was "serialtrack.matching", now "serialtrack.outliers".
"""

from typing import Tuple
import numpy as np
from scipy.spatial import cKDTree
import logging

log = logging.getLogger("serialtrack.outliers")


# ═══════════════════════════════════════════════════════════════
#  Westerweel universal outlier detection
# ═══════════════════════════════════════════════════════════════

def remove_outliers(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    track_a2b: np.ndarray,
    threshold: float = 5.0,
    n_neighbors: int = 27,
) -> np.ndarray:
    """Universal outlier detection for PTV data.

    Implements the normalised median residual test from:
        Westerweel & Scarano, "Universal outlier detection for PIV data",
        Exp. Fluids 39(6), 2005.

    Replaces ``removeOutlierTPT.m``.

    Parameters
    ----------
    coords_a    : (Na, D) — reference positions
    coords_b    : (Nb, D) — deformed positions
    track_a2b   : (Na,) int64 — index map (-1 = untracked)
    threshold   : float — normalised residual cutoff (typ. 2–5)
    n_neighbors : int — neighbors for median computation (typ. 27)

    Returns
    -------
    track_a2b : (Na,) int64 — updated with outliers set to -1
    """
    track = track_a2b.copy()
    tracked_mask = track >= 0
    tracked_idx = np.where(tracked_mask)[0]

    if len(tracked_idx) < n_neighbors + 1:
        return track  # too few points for statistics

    # Positions and displacements of tracked particles
    x0 = coords_a[tracked_idx]
    x1 = coords_b[track[tracked_idx]]
    u = x1 - x0  # (M, D)
    ndim = u.shape[1]

    # KNN among tracked particles
    tree = cKDTree(x0)
    K = min(n_neighbors, len(x0) - 1)
    _, knn_idx = tree.query(x0, k=K + 1)  # includes self at col 0

    # Fluctuation floor (Westerweel: epsilon ≈ 0.1 px)
    eps = 0.075

    is_outlier = np.zeros(len(tracked_idx), dtype=np.bool_)

    for d in range(ndim):
        u_d = u[:, d]
        # Gather neighbor displacements: (M, K+1)
        u_neigh = u_d[knn_idx]

        # Median displacement from all neighbors (including self)
        u_med = np.median(u_neigh, axis=1)

        # Median absolute deviation of neighbors from their median
        resid_neigh = np.abs(u_neigh - u_med[:, np.newaxis])
        r_med = np.median(resid_neigh, axis=1) + eps

        # Normalised residual for each particle
        rn = np.abs(u_d - u_med) / r_med

        is_outlier |= rn > threshold

    # Mark outliers as untracked
    track[tracked_idx[is_outlier]] = -1

    n_removed = int(np.sum(is_outlier))
    if n_removed > 0:
        log.info("Outlier removal: %d / %d particles flagged",
                 n_removed, len(tracked_idx))

    return track


# ═══════════════════════════════════════════════════════════════
#  Missing-particle detection (used late in ADMM iterations)
# ═══════════════════════════════════════════════════════════════

def find_not_missing(
    coords_a: np.ndarray,
    coords_b_warped: np.ndarray,
    dist_threshold: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify particles that have plausible matches after warping.

    Replaces the missing-particle culling logic in the MATLAB ADMM loop
    that fires when n_neighbors < 4.

    Parameters
    ----------
    coords_a      : reference particle coords
    coords_b_warped : deformed coords after global-step warp
    dist_threshold  : max allowed nearest-neighbor distance [px]

    Returns
    -------
    not_missing_a : indices of A particles with nearby B partners
    not_missing_b : indices of B particles with nearby A partners
    """
    dist_thresh = max(2.0, dist_threshold)

    tree_b = cKDTree(coords_b_warped)
    tree_a = cKDTree(coords_a)

    # A → B: for each A particle, nearest B
    dist_ab, _ = tree_b.query(coords_a, k=1)
    not_missing_a = np.where(dist_ab < dist_thresh)[0]

    # B → A: for each B particle, nearest A
    dist_ba, _ = tree_a.query(coords_b_warped, k=1)
    not_missing_b = np.where(dist_ba < dist_thresh)[0]

    return not_missing_a, not_missing_b


# ═══════════════════════════════════════════════════════════════
#  Adaptive f_o_s update (used between ADMM iterations)
# ═══════════════════════════════════════════════════════════════

def update_f_o_s(
    disp_update: np.ndarray,
    f_o_s_current: float = 60.0,
) -> float:
    """Shrink field-of-search based on displacement quantiles.

    Replaces the MATLAB quantile-based f_o_s update in the ADMM loop.
    Uses median + 0.5*IQR per component; takes the max across components.

    The floor is the larger of 2 px and 10% of the current f_o_s,
    preventing the search window from collapsing to zero while still
    allowing meaningful shrinkage.

    Parameters
    ----------
    disp_update : (N, D) displacement update from the global step
    f_o_s_current : current field of search value

    Returns
    -------
    new_f_o_s : float
    """
    if len(disp_update) == 0:
        return f_o_s_current

    ndim = disp_update.shape[1]
    vals = []
    for d in range(ndim):
        q25, q50, q75 = np.percentile(disp_update[:, d], [25, 50, 75])
        vals.append(q50 + 0.5 * (q75 - q25))

    # Floor: max(2 px, 10% of current f_o_s) — prevents collapse
    floor = max(2.0, 0.1 * f_o_s_current)
    return max(floor, *vals)


# ======================================================================
# VENDORED FROM: nd2studios/backend/serialtrack/matching.py
# ======================================================================

"""
SerialTrack Python — Chunk 2
=============================
Split this file into two modules:
    serialtrack/matching.py
    serialtrack/outliers.py

Dependencies (same as Chunk 1):
    pip install numpy scipy numba
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  FILE 1: serialtrack/matching.py — Topology matching & linking  ║
# ╚══════════════════════════════════════════════════════════════════╝

from typing import Tuple, Optional
import numpy as np
import numba as nb
from scipy.spatial import cKDTree
import logging


log = logging.getLogger("serialtrack.matching")


# ═══════════════════════════════════════════════════════════════
#  Numba kernels — rotation-invariant topology features
# ═══════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def _cross3(a, b):
    """Cross product of two 3-vectors."""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ])


@nb.njit(cache=True)
def _norm3(v):
    return np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


@nb.njit(cache=True)
def _dot3(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@nb.njit(parallel=True, cache=True)
def _build_features_3d(coords, neighbor_idx, n_neighbors):
    """Build rotation-invariant topology features for 3-D particles.

    For each particle, we:
      1. Get the K nearest neighbors (excluding self) from neighbor_idx.
      2. Build a rotation-invariant (RI) local frame:
         - ex = direction to nearest neighbor
         - ez = cross(r1, r2), oriented so dot(ez, r3) > 0
         - ey = cross(ez, ex)
      3. Transform neighbor offsets into RI frame.
      4. Compute spherical coords {r, phi, theta}.
      5. Reorder by phi starting from the nearest neighbor.
      6. Store features: r (distances), phi_diff (angular gaps), theta.

    Parameters
    ----------
    coords : (N, 3) float64 — particle positions
    neighbor_idx : (N, K+1) int64 — KNN indices (col 0 = self)
    n_neighbors : int — K

    Returns
    -------
    feat_r     : (N, K) float64 — reordered distances
    feat_phi   : (N, K) float64 — reordered angular differences in xy-plane
    feat_theta : (N, K) float64 — reordered polar angles
    """
    N = coords.shape[0]
    K = n_neighbors
    feat_r = np.zeros((N, K), dtype=np.float64)
    feat_phi = np.zeros((N, K), dtype=np.float64)
    feat_theta = np.zeros((N, K), dtype=np.float64)

    for i in nb.prange(N):
        # ----- 1. Neighbor offsets -----
        dx = np.empty((K, 3), dtype=np.float64)
        for k in range(K):
            j = neighbor_idx[i, k + 1]  # skip self at col 0
            dx[k, 0] = coords[j, 0] - coords[i, 0]
            dx[k, 1] = coords[j, 1] - coords[i, 1]
            dx[k, 2] = coords[j, 2] - coords[i, 2]

        # ----- 2. Build rotation-invariant frame -----
        r1 = dx[0]
        nr1 = _norm3(r1)
        if nr1 < 1e-30:
            continue
        ex = r1 / nr1

        # ez = cross(r1, r2), normalised
        if K >= 2:
            ez = _cross3(dx[0], dx[1])
        else:
            # Fallback: pick arbitrary orthogonal
            if abs(ex[0]) < 0.9:
                ez = _cross3(ex, np.array([1.0, 0.0, 0.0]))
            else:
                ez = _cross3(ex, np.array([0.0, 1.0, 0.0]))
        nez = _norm3(ez)
        if nez < 1e-30:
            # r1 ∥ r2 — use fallback
            if abs(ex[0]) < 0.9:
                ez = _cross3(ex, np.array([1.0, 0.0, 0.0]))
            else:
                ez = _cross3(ex, np.array([0.0, 1.0, 0.0]))
            nez = _norm3(ez)
            if nez < 1e-30:
                continue
        ez = ez / nez

        # Orient ez so 3rd neighbor is in +ez hemisphere
        if K >= 3 and _dot3(ez, dx[2]) < 0.0:
            ez = -ez

        # ey = cross(ez, ex)
        ey = _cross3(ez, ex)
        ney = _norm3(ey)
        if ney < 1e-30:
            continue
        ey = ey / ney

        # ----- 3. Transform into RI frame -----
        dx_ri = np.empty((K, 3), dtype=np.float64)
        for k in range(K):
            dx_ri[k, 0] = _dot3(dx[k], ex)
            dx_ri[k, 1] = _dot3(dx[k], ey)
            dx_ri[k, 2] = _dot3(dx[k], ez)

        # ----- 4. Spherical coordinates -----
        r_arr = np.empty(K, dtype=np.float64)
        phi_arr = np.empty(K, dtype=np.float64)
        theta_arr = np.empty(K, dtype=np.float64)
        for k in range(K):
            r_arr[k] = np.sqrt(
                dx_ri[k,0]**2 + dx_ri[k,1]**2 + dx_ri[k,2]**2
            )
            phi_arr[k] = np.arctan2(dx_ri[k, 1], dx_ri[k, 0])
            rxy = np.sqrt(dx_ri[k,0]**2 + dx_ri[k,1]**2)
            theta_arr[k] = np.arctan2(dx_ri[k, 2], rxy)

        # ----- 5. Reorder by phi, starting from nearest (idx 0) -----
        # Sort phi to get order
        order = np.argsort(phi_arr)
        # Find where the nearest neighbor (original index 0) lands
        start = 0
        for k in range(K):
            if order[k] == 0:
                start = k
                break
        # Build reordered arrays starting from nearest neighbor
        for k in range(K):
            idx = order[(start + k) % K]
            feat_r[i, k] = r_arr[idx]
            theta_arr_val = theta_arr[idx]
            feat_theta[i, k] = theta_arr_val

        # phi_diff: consecutive angle differences (circular)
        phi_reord = np.empty(K, dtype=np.float64)
        for k in range(K):
            idx = order[(start + k) % K]
            phi_reord[k] = phi_arr[idx]
        for k in range(K):
            diff = phi_reord[(k + 1) % K] - phi_reord[k]
            if diff < 0.0:
                diff += 2.0 * np.pi
            feat_phi[i, k] = diff

    return feat_r, feat_phi, feat_theta


@nb.njit(parallel=True, cache=True)
def _build_features_2d(coords, neighbor_idx, n_neighbors):
    """Build topology features for 2-D particles.

    Same logic as 3-D but without theta and without the RI frame
    (2-D only needs r and phi_diff — phi_diff is already rotation-invariant
    because it measures relative angles between neighbors).

    Returns
    -------
    feat_r   : (N, K) float64 — reordered distances
    feat_phi : (N, K) float64 — reordered angular differences
    """
    N = coords.shape[0]
    K = n_neighbors
    feat_r = np.zeros((N, K), dtype=np.float64)
    feat_phi = np.zeros((N, K), dtype=np.float64)

    for i in nb.prange(N):
        dx = np.empty((K, 2), dtype=np.float64)
        for k in range(K):
            j = neighbor_idx[i, k + 1]
            dx[k, 0] = coords[j, 0] - coords[i, 0]
            dx[k, 1] = coords[j, 1] - coords[i, 1]

        r_arr = np.empty(K, dtype=np.float64)
        phi_arr = np.empty(K, dtype=np.float64)
        for k in range(K):
            r_arr[k] = np.sqrt(dx[k, 0]**2 + dx[k, 1]**2)
            phi_arr[k] = np.arctan2(dx[k, 1], dx[k, 0])

        order = np.argsort(phi_arr)
        start = 0
        for k in range(K):
            if order[k] == 0:
                start = k
                break

        for k in range(K):
            idx = order[(start + k) % K]
            feat_r[i, k] = r_arr[idx]

        phi_reord = np.empty(K, dtype=np.float64)
        for k in range(K):
            idx = order[(start + k) % K]
            phi_reord[k] = phi_arr[idx]
        for k in range(K):
            diff = phi_reord[(k + 1) % K] - phi_reord[k]
            if diff < 0.0:
                diff += 2.0 * np.pi
            feat_phi[i, k] = diff

    return feat_r, feat_phi


# ═══════════════════════════════════════════════════════════════
#  Numba kernel — brute-force topology match (parallelised)
# ═══════════════════════════════════════════════════════════════

@nb.njit(parallel=True, cache=True)
def _match_features_3d(
    coords_a, feat_r_a, feat_phi_a, feat_theta_a,
    coords_b, feat_r_b, feat_phi_b, feat_theta_b,
    cand_idx,       # (Na, max_cands) int64, -1 padded
    cand_counts,    # (Na,) int64, how many valid candidates per A particle
    f_o_s,
):
    """Find topology matches: A → B.

    For each particle in A, search its candidate list in B.
    A match is accepted when all three SSE channels (r, phi, theta)
    minimise at the same candidate AND displacement < f_o_s.

    Returns
    -------
    match_a : (M,) int64 — indices into A
    match_b : (M,) int64 — indices into B
    """
    Na = coords_a.shape[0]
    # Pre-allocate max possible (one match per A particle)
    out_a = np.full(Na, -1, dtype=np.int64)
    out_b = np.full(Na, -1, dtype=np.int64)

    for i in nb.prange(Na):
        nc = cand_counts[i]
        if nc == 0:
            continue

        # SSE for each candidate
        best_r = np.int64(-1);   min_r = np.inf
        best_p = np.int64(-1);   min_p = np.inf
        best_t = np.int64(-1);   min_t = np.inf

        for ci in range(nc):
            j = cand_idx[i, ci]
            sse_r = 0.0; sse_p = 0.0; sse_t = 0.0
            for k in range(feat_r_a.shape[1]):
                dr = feat_r_a[i, k] - feat_r_b[j, k]
                dp = feat_phi_a[i, k] - feat_phi_b[j, k]
                dt = feat_theta_a[i, k] - feat_theta_b[j, k]
                sse_r += dr * dr
                sse_p += dp * dp
                sse_t += dt * dt
            if sse_r < min_r:
                min_r = sse_r; best_r = j
            if sse_p < min_p:
                min_p = sse_p; best_p = j
            if sse_t < min_t:
                min_t = sse_t; best_t = j

        # All three must agree
        if best_r == best_p and best_p == best_t and best_r >= 0:
            # Check displacement < f_o_s
            d2 = 0.0
            for d in range(3):
                dd = coords_a[i, d] - coords_b[best_r, d]
                d2 += dd * dd
            if np.sqrt(d2) < f_o_s:
                out_a[i] = i
                out_b[i] = best_r

    return out_a, out_b


@nb.njit(parallel=True, cache=True)
def _match_features_2d(
    coords_a, feat_r_a, feat_phi_a,
    coords_b, feat_r_b, feat_phi_b,
    cand_idx, cand_counts, f_o_s,
):
    """2-D topology match: only r and phi channels.

    Match accepted when both r and phi minimise at same candidate.
    """
    Na = coords_a.shape[0]
    out_a = np.full(Na, -1, dtype=np.int64)
    out_b = np.full(Na, -1, dtype=np.int64)

    for i in nb.prange(Na):
        nc = cand_counts[i]
        if nc == 0:
            continue

        best_r = np.int64(-1);  min_r = np.inf
        best_p = np.int64(-1);  min_p = np.inf

        for ci in range(nc):
            j = cand_idx[i, ci]
            sse_r = 0.0; sse_p = 0.0
            for k in range(feat_r_a.shape[1]):
                dr = feat_r_a[i, k] - feat_r_b[j, k]
                dp = feat_phi_a[i, k] - feat_phi_b[j, k]
                sse_r += dr * dr
                sse_p += dp * dp
            if sse_r < min_r:
                min_r = sse_r; best_r = j
            if sse_p < min_p:
                min_p = sse_p; best_p = j

        if best_r == best_p and best_r >= 0:
            d2 = 0.0
            for d in range(2):
                dd = coords_a[i, d] - coords_b[best_r, d]
                d2 += dd * dd
            if np.sqrt(d2) < f_o_s:
                out_a[i] = i
                out_b[i] = best_r

    return out_a, out_b


# ═══════════════════════════════════════════════════════════════
#  High-level matcher classes
# ═══════════════════════════════════════════════════════════════

class TopologyMatcher:
    """Scale & rotation invariant particle matcher.

    This is the core of SerialTrack — it replaces:
        - ``f_track_neightopo_match3.m``   (3-D)
        - ``f_track_neightopo_match.m``    (2-D)

    Algorithm
    ---------
    1. Build a cKDTree for each point set (vectorised bulk KNN).
    2. Extract rotation-invariant topology features via Numba JIT.
    3. Build candidate lists (B particles near each A particle).
    4. Compare feature vectors in parallel via Numba.
    5. Accept matches where ALL feature channels agree.

    Examples
    --------
    >>> matcher = TopologyMatcher(n_neighbors=20, f_o_s=60.0)
    >>> matches = matcher.match(coords_a, coords_b)   # (M, 2) array
    """

    def __init__(self, n_neighbors: int = 20, f_o_s: float = 60.0):
        self.n_neighbors = n_neighbors
        self.f_o_s = f_o_s

    def match(
        self,
        coords_a: np.ndarray,
        coords_b: np.ndarray,
    ) -> np.ndarray:
        """Find topology-based matches from A → B.

        Parameters
        ----------
        coords_a : (Na, D) float64 — particle coords in image A
        coords_b : (Nb, D) float64 — particle coords in image B

        Returns
        -------
        matches : (M, 2) int64 — column 0 = index into A, column 1 = index into B
        """
        ndim = coords_a.shape[1]
        K = self.n_neighbors
        Na, Nb = len(coords_a), len(coords_b)

        if Na == 0 or Nb == 0:
            return np.empty((0, 2), dtype=np.int64)

        # Clamp K to available particles (minus self)
        K = min(K, Na - 1, Nb - 1)
        if K < 1:
            return np.empty((0, 2), dtype=np.int64)

        # 1. KNN for feature extraction (self-neighbors)
        tree_a = cKDTree(coords_a)
        tree_b = cKDTree(coords_b)
        _, knn_a = tree_a.query(coords_a, k=K + 1)  # includes self
        _, knn_b = tree_b.query(coords_b, k=K + 1)

        knn_a = np.ascontiguousarray(knn_a.astype(np.int64))
        knn_b = np.ascontiguousarray(knn_b.astype(np.int64))

        # 2. Build features
        if ndim == 3:
            feat_r_a, feat_phi_a, feat_theta_a = _build_features_3d(
                coords_a, knn_a, K
            )
            feat_r_b, feat_phi_b, feat_theta_b = _build_features_3d(
                coords_b, knn_b, K
            )
        else:
            feat_r_a, feat_phi_a = _build_features_2d(coords_a, knn_a, K)
            feat_r_b, feat_phi_b = _build_features_2d(coords_b, knn_b, K)

        # 3. Build candidate lists: for each A particle, which B particles
        #    are within sqrt(ndim)*f_o_s ?
        cand_idx, cand_counts = self._build_candidates(
            coords_a, tree_b, K, ndim
        )

        # 4. Match
        if ndim == 3:
            out_a, out_b = _match_features_3d(
                coords_a, feat_r_a, feat_phi_a, feat_theta_a,
                coords_b, feat_r_b, feat_phi_b, feat_theta_b,
                cand_idx, cand_counts, self.f_o_s,
            )
        else:
            out_a, out_b = _match_features_2d(
                coords_a, feat_r_a, feat_phi_a,
                coords_b, feat_r_b, feat_phi_b,
                cand_idx, cand_counts, self.f_o_s,
            )

        # 5. Collect valid matches
        valid = out_a >= 0
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int64)
        return np.column_stack((out_a[valid], out_b[valid]))

    def _build_candidates(
        self, coords_a, tree_b, K, ndim
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build padded candidate index array using ball query.

        Falls back to KNN with distance filter when f_o_s is finite.
        When f_o_s is inf, all B particles are candidates.
        """
        Na = len(coords_a)
        Nb = tree_b.n

        if self.f_o_s == np.inf or self.f_o_s <= 0:
            # All B particles are candidates
            max_c = Nb
            cand_idx = np.empty((Na, max_c), dtype=np.int64)
            cand_counts = np.full(Na, Nb, dtype=np.int64)
            row = np.arange(Nb, dtype=np.int64)
            for i in range(Na):
                cand_idx[i, :] = row
            return cand_idx, cand_counts

        # Use ball_point query for radius search
        radius = np.sqrt(ndim) * self.f_o_s
        ball_results = tree_b.query_ball_point(coords_a, r=radius)

        # Determine max candidate count for padding
        max_c = max(len(r) for r in ball_results) if len(ball_results) > 0 else 0
        max_c = max(max_c, 1)  # at least 1 column
        cand_idx = np.full((Na, max_c), -1, dtype=np.int64)
        cand_counts = np.zeros(Na, dtype=np.int64)

        for i, indices in enumerate(ball_results):
            n = len(indices)
            cand_counts[i] = n
            for ci in range(n):
                cand_idx[i, ci] = indices[ci]

        return cand_idx, cand_counts


class NearestNeighborMatcher:
    """Simple nearest-neighbor fallback matcher.

    Used when n_neighbors ≤ 2 (insufficient for topology matching).
    Replaces ``f_track_nearest_neighbour3.m``.
    """

    def __init__(self, f_o_s: float = 60.0):
        self.f_o_s = f_o_s

    def match(
        self,
        coords_a: np.ndarray,
        coords_b: np.ndarray,
    ) -> np.ndarray:
        """One-to-one nearest neighbor matching.

        Returns (M, 2) int64 array of (idx_a, idx_b) pairs.
        """
        if len(coords_a) == 0 or len(coords_b) == 0:
            return np.empty((0, 2), dtype=np.int64)

        tree_b = cKDTree(coords_b)
        dists, idx_b = tree_b.query(coords_a, k=1)

        # Filter by f_o_s
        mask = dists < self.f_o_s
        idx_a = np.where(mask)[0]
        return np.column_stack((idx_a, idx_b[mask]))


# ═══════════════════════════════════════════════════════════════
#  Displacement computation from matches
# ═══════════════════════════════════════════════════════════════

def compute_displacement(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    matches: np.ndarray,
    outlier_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build track_A2B index array and compute displacements.

    Replaces ``funCompDisp3.m``.

    Parameters
    ----------
    coords_a : (Na, D) — reference particle positions
    coords_b : (Nb, D) — deformed particle positions
    matches  : (M, 2) int — (index_a, index_b) pairs
    outlier_threshold : float — if > 0, apply Westerweel outlier removal

    Returns
    -------
    track_a2b : (Na,) int64 — track_a2b[i] = index into B, or -1
    disp_a2b  : (M', D) float64 — displacements for tracked particles
    """
    Na = len(coords_a)
    ndim = coords_a.shape[1]
    track_a2b = np.full(Na, -1, dtype=np.int64)

    if len(matches) == 0:
        return track_a2b, np.empty((0, ndim))

    for ia, ib in matches:
        track_a2b[ia] = ib

    # Outlier removal
    if outlier_threshold > 0:
        track_a2b = remove_outliers(
            coords_a, coords_b, track_a2b, outlier_threshold
        )

    # Compute clean displacements
    tracked = track_a2b >= 0
    idx_a = np.where(tracked)[0]
    idx_b = track_a2b[idx_a]
    disp = coords_b[idx_b] - coords_a[idx_a]

    return track_a2b, disp


# ═══════════════════════════════════════════════════════════════
#  Convenience: adaptive matcher selection
# ═══════════════════════════════════════════════════════════════

def match_particles(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    n_neighbors: int,
    f_o_s: float,
    outlier_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-call convenience function: detect matches + compute disp.

    Automatically selects topology matching (n_neighbors > 2) or
    nearest-neighbor fallback (n_neighbors ≤ 2), matching the
    behaviour of the MATLAB ADMM inner loop.

    Parameters
    ----------
    coords_a, coords_b : particle coordinates
    n_neighbors : current ADMM neighbor count
    f_o_s : field of search
    outlier_threshold : Westerweel threshold (0 = skip)

    Returns
    -------
    matches   : (M, 2) int64
    track_a2b : (Na,) int64
    disp_a2b  : (M', D) float64
    """
    if n_neighbors > 2:
        matcher = TopologyMatcher(n_neighbors=n_neighbors, f_o_s=f_o_s)
    else:
        matcher = NearestNeighborMatcher(f_o_s=f_o_s)

    matches = matcher.match(coords_a, coords_b)

    track_a2b, disp_a2b = compute_displacement(
        coords_a, coords_b, matches, outlier_threshold
    )

    return matches, track_a2b, disp_a2b


# ======================================================================
# VENDORED FROM: nd2studios/backend/serialtrack/detection.py
# ======================================================================

"""
SerialTrack Python — Chunk 1
=============================
Split this file into three modules:
    serialtrack/config.py
    serialtrack/io.py
    serialtrack/detection.py

Dependencies:
    pip install numpy scipy scikit-image numba tifffile
"""
# ╔══════════════════════════════════════════════════════════════╗
# ║  FILE 3: serialtrack/detection.py — Particle detection      ║
# ╚══════════════════════════════════════════════════════════════╝
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, Union
import numpy as np
from pathlib import Path
from typing import List
import logging
from scipy import ndimage
import numba as nb
# ─────────────────────────────────────────────────────────────
#  Numba-accelerated sub-pixel localization kernels
# ─────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _subpixel_poly_2d(log_img, xs, ys):
    """3-point parabola sub-pixel refinement for 2-D peaks.

    For each peak at integer coords (xs[i], ys[i]) in *log_img*,
    fit y = a + bx + cx² along each axis and return the shift to
    the parabola vertex.

    Returns (dx, dy) arrays — shifts to add to integer coords.
    """
    n = len(xs)
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    h, w = log_img.shape  # note: (x-dim, y-dim) in our convention

    for i in range(n):
        xi, yi = xs[i], ys[i]
        # x-direction
        if 0 < xi < h - 1:
            a = log_img[xi - 1, yi]
            b = log_img[xi, yi]
            c = log_img[xi + 1, yi]
            d = 2.0 * (a - 2.0 * b + c)
            dx[i] = -(c - a) / d if abs(d) > 1e-12 else 0.0
        else:
            dx[i] = 0.0
        # y-direction
        if 0 < yi < w - 1:
            a = log_img[xi, yi - 1]
            b = log_img[xi, yi]
            c = log_img[xi, yi + 1]
            d = 2.0 * (a - 2.0 * b + c)
            dy[i] = -(c - a) / d if abs(d) > 1e-12 else 0.0
        else:
            dy[i] = 0.0
    return dx, dy


@nb.njit(cache=True)
def _subpixel_poly_3d(log_img, xs, ys, zs):
    """3-point parabola sub-pixel refinement for 3-D peaks.

    Returns (dx, dy, dz) arrays.
    """
    n = len(xs)
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    dz = np.empty(n, dtype=np.float64)
    sx, sy, sz = log_img.shape

    for i in range(n):
        xi, yi, zi = xs[i], ys[i], zs[i]
        # x
        if 0 < xi < sx - 1:
            a, b, c = log_img[xi-1,yi,zi], log_img[xi,yi,zi], log_img[xi+1,yi,zi]
            d = 2.0*(a - 2.0*b + c)
            dx[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dx[i] = 0.0
        # y
        if 0 < yi < sy - 1:
            a, b, c = log_img[xi,yi-1,zi], log_img[xi,yi,zi], log_img[xi,yi+1,zi]
            d = 2.0*(a - 2.0*b + c)
            dy[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dy[i] = 0.0
        # z
        if 0 < zi < sz - 1:
            a, b, c = log_img[xi,yi,zi-1], log_img[xi,yi,zi], log_img[xi,yi,zi+1]
            d = 2.0*(a - 2.0*b + c)
            dz[i] = -(c - a)/d if abs(d) > 1e-12 else 0.0
        else:
            dz[i] = 0.0
    return dx, dy, dz


@nb.njit(parallel=True, cache=True)
def _radial_symmetry_3d(patches, half_win, dccd, abc):
    """Radial-symmetry sub-voxel localization (Liu et al. 2013).

    Parameters
    ----------
    patches : (N, wx, wy, wz)  float64 array of image patches
    half_win : (3,) int array   half window sizes
    dccd : (3,) float array     pixel spacings
    abc  : (3,) float array     anisotropy factors

    Returns
    -------
    dx, dy, dz : (N,) sub-pixel shifts
    """
    N = patches.shape[0]
    wx, wy, wz = patches.shape[1], patches.shape[2], patches.shape[3]
    dx = np.zeros(N, dtype=np.float64)
    dy = np.zeros(N, dtype=np.float64)
    dz = np.zeros(N, dtype=np.float64)
    a, b, c = abc[0], abc[1], abc[2]
    dxc, dyc, dzc = dccd[0], dccd[1], dccd[2]

    for pi in nb.prange(N):
        # --- intensity-weighted centroid ---
        sx_ = 0.0; sy_ = 0.0; sz_ = 0.0; tot = 0.0
        for ix in range(wx):
            for iy in range(wy):
                for iz in range(wz):
                    v = patches[pi, ix, iy, iz]
                    sx_ += v * (ix - (wx-1)*0.5) * dxc
                    sy_ += v * (iy - (wy-1)*0.5) * dyc
                    sz_ += v * (iz - (wz-1)*0.5) * dzc
                    tot += v
        if tot < 1e-30:
            continue
        xm = sx_ / tot;  ym = sy_ / tot;  zm = sz_ / tot

        # --- build 3×3 normal system from gradient votes ---
        A00=0.;A01=0.;A02=0.;A11=0.;A12=0.;A22=0.
        B0=0.;B1=0.;B2=0.

        for ix in range(1, wx-1):
            for iy in range(1, wy-1):
                for iz in range(1, wz-1):
                    gu = (patches[pi,ix+1,iy,iz] - patches[pi,ix-1,iy,iz])/(2*dxc)
                    gv = (patches[pi,ix,iy+1,iz] - patches[pi,ix,iy-1,iz])/(2*dyc)
                    gw = (patches[pi,ix,iy,iz+1] - patches[pi,ix,iy,iz-1])/(2*dzc)
                    gm = np.sqrt(gu*gu + gv*gv + gw*gw)
                    if gm < 1e-10:
                        continue
                    gu /= gm; gv /= gm; gw /= gm

                    xp = (ix-(wx-1)*0.5)*dxc/a - xm/a
                    yp = (iy-(wy-1)*0.5)*dyc/b - ym/b
                    zp = (iz-(wz-1)*0.5)*dzc/c - zm/c
                    dd = np.sqrt(xp*xp + yp*yp + zp*zp)
                    if dd < 1e-10:
                        continue
                    q = gm*gm / dd

                    A00 += q*(1-gu*gu); A01 += q*(-gu*gv); A02 += q*(-gu*gw)
                    A11 += q*(1-gv*gv); A12 += q*(-gv*gw); A22 += q*(1-gw*gw)
                    dot = gu*xp + gv*yp + gw*zp
                    B0 += q*(xp - gu*dot)
                    B1 += q*(yp - gv*dot)
                    B2 += q*(zp - gw*dot)

        # --- solve 3×3 symmetric system via Cramer ---
        det = (A00*(A11*A22 - A12*A12)
             - A01*(A01*A22 - A02*A12)
             + A02*(A01*A12 - A02*A11))
        if abs(det) < 1e-30:
            continue
        inv = 1.0 / det
        dx[pi] = ((A11*A22-A12*A12)*B0 + (A02*A12-A01*A22)*B1 + (A01*A12-A02*A11)*B2)*inv*a
        dy[pi] = ((A02*A12-A01*A22)*B0 + (A00*A22-A02*A02)*B1 + (A01*A02-A00*A12)*B2)*inv*b
        dz[pi] = ((A01*A12-A02*A11)*B0 + (A01*A02-A00*A12)*B1 + (A00*A11-A01*A01)*B2)*inv*c

    return dx, dy, dz


# ─────────────────────────────────────────────────────────────
#  Main detector class
# ─────────────────────────────────────────────────────────────

class ParticleDetector:
    """Detect and localise particles in 2-D or 3-D images.

    Methods
    -------
    detect(img)
        Full pipeline: threshold → filter → detect → sub-pixel.
        Returns ``(N, ndim)`` coordinate array.

    Examples
    --------
    >>> cfg = DetectionConfig(threshold=0.3, bead_radius=4)
    >>> det = ParticleDetector(cfg)
    >>> coords = det.detect(image_3d)       # shape (N, 3)
    >>> coords = det.detect(image_2d)       # shape (N, 2)
    """

    def __init__(self, config: DetectionConfig):
        self.cfg = config

    # ── public API ──────────────────────────────────────────

    def detect(
        self,
        img: np.ndarray,
        roi_slices: Optional[Tuple[slice, ...]] = None,
    ) -> np.ndarray:
        """Run full detection pipeline. Returns (N, ndim) coords."""
        ndim = img.ndim

        # ROI crop
        offset = np.zeros(ndim, dtype=np.float64)
        if roi_slices is not None:
            offset = np.array([s.start or 0 for s in roi_slices], dtype=np.float64)
            img = img[roi_slices].copy()

        # Deconvolution
        if self.cfg.psf is not None:
            from skimage.restoration import richardson_lucy
            img = richardson_lucy(
                img.astype(np.float64), self.cfg.psf,
                num_iter=self.cfg.deconv_iters, clip=False,
            )

        # Invert for dark particles
        if self.cfg.color == "black":
            img = img.max() - img

        # Normalise to [0, 1]
        img = img.astype(np.float64)
        vmax = img.max()
        if vmax > 0:
            img_n = img / vmax
        else:
            return np.empty((0, ndim))

        # Dispatch
        if self.cfg.method == DetectionMethod.TRACTRAC:
            coords = self._detect_tractrac(img_n)
        else:
            coords = self._detect_tpt(img_n, img)

        # Offset back to full-image coords & clip
        if coords.size:
            coords += offset
        return coords

    # ── TracTrac method ─────────────────────────────────────

    def _detect_tractrac(self, img_n: np.ndarray) -> np.ndarray:
        """LoG → local-maximum → sub-pixel polynomial fit."""
        ndim = img_n.ndim
        bw = self._size_filtered_mask(img_n)
        img_m = img_n * bw  # masked image

        if self.cfg.bead_radius > 0:
            return self._log_detect(img_m, img_n, ndim)
        else:
            return self._centroid_detect(img_n, ndim)

    def _log_detect(self, img_m, img_n, ndim):
        sigma = self.cfg.bead_radius

        # Laplacian of Gaussian
        log_img = -ndimage.gaussian_laplace(img_m, sigma=sigma)

        # Local-maximum filter
        fp_size = int(2 * sigma * 2) + 1
        fp = np.ones((fp_size,) * ndim)
        rng = np.random.default_rng(42)
        noise = rng.random(log_img.shape) * 1e-5
        dilated = ndimage.maximum_filter(log_img + noise, footprint=fp)
        peaks = ((log_img + noise) == dilated) & (img_n > self.cfg.threshold)

        coords_int = np.asarray(np.nonzero(peaks), dtype=np.int64).T  # (N, ndim)
        if len(coords_int) == 0:
            return np.empty((0, ndim))

        # Trim border
        nb_ = max(int((sigma + 2) / 2), 1)
        mask = np.ones(len(coords_int), dtype=np.bool_)
        for d in range(ndim):
            mask &= (coords_int[:, d] >= nb_) & (coords_int[:, d] < img_n.shape[d] - nb_)
        coords_int = coords_int[mask]
        if len(coords_int) == 0:
            return np.empty((0, ndim))

        # Sub-pixel via parabola on log(LoG)
        log_safe = np.log(np.clip(log_img - log_img.min() + 1e-8, 1e-12, None))

        if ndim == 2:
            dx, dy = _subpixel_poly_2d(log_safe, coords_int[:, 0], coords_int[:, 1])
            valid = (np.abs(dx) < 0.5) & (np.abs(dy) < 0.5)
            out = coords_int[valid].astype(np.float64)
            out[:, 0] += dx[valid]
            out[:, 1] += dy[valid]
        else:
            dx, dy, dz = _subpixel_poly_3d(
                log_safe, coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
            )
            valid = (np.abs(dx) < 0.5) & (np.abs(dy) < 0.5) & (np.abs(dz) < 0.5)
            out = coords_int[valid].astype(np.float64)
            out[:, 0] += dx[valid]
            out[:, 1] += dy[valid]
            out[:, 2] += dz[valid]
        return out

    # ── TPT method ──────────────────────────────────────────

    def _detect_tpt(self, img_n: np.ndarray, img_raw: np.ndarray) -> np.ndarray:
        """Blob centroid → radial-symmetry sub-voxel refinement."""
        ndim = img_n.ndim
        coords = self._centroid_detect(img_n, ndim)
        if len(coords) == 0 or ndim != 3:
            return coords  # radial symmetry only for 3-D

        # Extract patches for radial-symmetry
        ws = np.array(self.cfg.win_size[:3], dtype=np.int64)
        half = ws // 2
        ci = np.round(coords).astype(np.int64)

        # Pad with reflected noise
        img_f = img_raw.astype(np.float64)
        img_f += self.cfg.rand_noise * np.random.default_rng(0).random(img_f.shape)
        img_p = np.pad(img_f, [(h, h) for h in half], mode="reflect")

        patches = np.empty((len(ci), ws[0], ws[1], ws[2]), dtype=np.float64)
        for i, c in enumerate(ci):
            cp = c + half  # padded coords
            patches[i] = img_p[
                cp[0]-half[0]:cp[0]+half[0]+1,
                cp[1]-half[1]:cp[1]+half[1]+1,
                cp[2]-half[2]:cp[2]+half[2]+1,
            ]

        dccd = np.array(self.cfg.dccd[:3], dtype=np.float64)
        abc = np.array(self.cfg.abc[:3], dtype=np.float64)

        dx, dy, dz = _radial_symmetry_3d(patches, half, dccd, abc)

        # Apply only well-behaved shifts
        ok = (np.abs(dx) < half[0]) & (np.abs(dy) < half[1]) & (np.abs(dz) < half[2])
        out = coords.copy()
        out[ok, 0] += dx[ok]
        out[ok, 1] += dy[ok]
        out[ok, 2] += dz[ok]
        return out

    # ── shared helpers ──────────────────────────────────────

    def _size_filtered_mask(self, img_n: np.ndarray) -> np.ndarray:
        """Threshold → label → keep blobs within [min_size, max_size]."""
        bw = img_n > self.cfg.threshold
        labeled, n = ndimage.label(bw)
        if n == 0:
            return bw
        sizes = ndimage.sum_labels(bw, labeled, range(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        for i, s in enumerate(sizes, 1):
            if self.cfg.min_size <= s <= self.cfg.max_size:
                keep[i] = True
        return keep[labeled]

    def _centroid_detect(self, img_n: np.ndarray, ndim: int) -> np.ndarray:
        """Connected-component centroids, filtered by size."""
        bw = img_n > self.cfg.threshold
        labeled, n = ndimage.label(bw)
        if n == 0:
            return np.empty((0, ndim))
        sizes = ndimage.sum_labels(bw, labeled, range(1, n + 1))
        centroids = ndimage.center_of_mass(img_n, labeled, range(1, n + 1))
        out = np.array([
            c for c, s in zip(centroids, sizes)
            if self.cfg.min_size <= s <= self.cfg.max_size
        ])
        return out if out.size else np.empty((0, ndim))

    @staticmethod
    def clip_to_bounds(coords: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Remove coords outside [0, shape) for each dimension."""
        if coords.size == 0:
            return coords
        mask = np.ones(len(coords), dtype=bool)
        for d in range(coords.shape[1]):
            mask &= (coords[:, d] >= 0) & (coords[:, d] < shape[d])
        return coords[mask]

# ======================================================================
# VENDORED FROM: nd2studios/backend/serialtrack/regularization.py
# ======================================================================

"""
SerialTrack Python — Chunk 3
=============================
Split this file into two modules:
    serialtrack/regularization.py
    serialtrack/fields.py

Dependencies:
    pip install numpy scipy numba
"""

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FILE 1: serialtrack/regularization.py                              ║
# ║  Global-step solvers: MLS, grid regularization, ADMM-AL             ║
# ║  Replaces: funCompDefGrad3.m, funScatter2Grid3D.m, regularizeNd.m,  ║
# ║            funDerivativeOp3.m, and the gbSolver branches in          ║
# ║            f_track_serial_match3D.m                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

from typing import Tuple, Optional, Dict, Any
from enum import IntEnum
import numpy as np
from scipy.spatial import cKDTree, QhullError
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.sparse import eye as speye, diags as spdiags, kron as spkron, csc_matrix
from scipy.sparse.linalg import spsolve
import logging


log = logging.getLogger("serialtrack.regularization")


# ═══════════════════════════════════════════════════════════════
#  Scatter → Grid interpolation  (replaces funScatter2Grid3D.m
#  + the 600-line regularizeNd.m)
# ═══════════════════════════════════════════════════════════════

def scatter_to_grid(
    coords: np.ndarray,
    values: np.ndarray,
    grid_step: np.ndarray,
    smoothness: float = 0.0,
    grid_coords: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Interpolate scattered data onto a regular grid.

    Replaces ``funScatter2Grid3D.m`` + ``regularizeNd.m`` (~650 lines).

    Parameters
    ----------
    coords : (N, D) — scattered point positions
    values : (N,)   — scalar field values at those points
    grid_step : (D,) — grid spacing per dimension
    smoothness : float — RBF smoothing parameter (0 = pure interpolation)
    grid_coords : optional pre-built meshgrid arrays

    Returns
    -------
    grids : tuple of D arrays, each with shape of the output grid
    f_grid : array with same shape — interpolated values
    """
    ndim = coords.shape[1]
    gs = np.asarray(grid_step, dtype=np.float64)

    # Build grid axes
    if grid_coords is not None:
        grids = grid_coords
        axes = tuple(np.unique(g) for g in grids)
    else:
        axes = []
        for d in range(ndim):
            lo, hi = coords[:, d].min(), coords[:, d].max()
            axes.append(np.arange(lo, hi + gs[d], gs[d]))
        axes = tuple(axes)
        grids = np.meshgrid(*axes, indexing="ij")
        grids = tuple(grids)

    query_pts = np.column_stack([g.ravel() for g in grids])

    # Minimum point requirements:
    #   LinearNDInterpolator needs ndim+1 for a simplex
    #   RBFInterpolator(degree=1) needs ndim+1 points
    min_pts = ndim + 1
    if len(coords) < min_pts:
        log.warning("scatter_to_grid: only %d points (need %d) — returning zeros",
                     len(coords), min_pts)
        f_grid = np.zeros(grids[0].shape, dtype=np.float64)
        return grids, f_grid

    if smoothness <= 0:
        # Pure linear interpolation (fast)
        f_flat = _linear_or_nearest_interpolate(coords, values, query_pts)
    else:
        # RBF with smoothing — replaces the entire regularizeNd.m
        interp = RBFInterpolator(
            coords, values,
            smoothing=smoothness,
            kernel="thin_plate_spline",
            degree=1,
        )
        f_flat = interp(query_pts)

    f_grid = f_flat.reshape(grids[0].shape)
    return grids, f_grid


def _linear_or_nearest_interpolate(
    coords: np.ndarray,
    values: np.ndarray,
    query_pts: np.ndarray,
) -> np.ndarray:
    """Try LinearNDInterpolator, fallback to nearest neighbour on degenerate input."""
    ndim = coords.shape[1]
    if len(coords) < ndim + 1:
        log.warning(
            "linear interpolation: only %d points for %dd data — using nearest neighbour",
            len(coords), ndim,
        )
        interp = NearestNDInterpolator(coords, values)
        return interp(query_pts)

    try:
        interp = LinearNDInterpolator(coords, values, fill_value=0.0)
        return interp(query_pts)
    except QhullError as exc:
        log.warning(
            "LinearNDInterpolator failed (%s); falling back to nearest neighbour",
            exc,
        )
        interp = NearestNDInterpolator(coords, values)
        return interp(query_pts)


def scatter_to_grid_multi(
    coords: np.ndarray,
    disp: np.ndarray,
    grid_step: np.ndarray,
    smoothness: float = 0.0,
    grid_coords: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Interpolate a multi-component displacement field to a grid.

    Parameters
    ----------
    coords : (N, D) — particle positions
    disp   : (N, D) — displacement vectors at those positions
    grid_step : (D,) — grid spacing
    smoothness : float

    Returns
    -------
    grids : tuple of D meshgrid arrays
    disp_grid : (D, *grid_shape) — gridded displacement components
    """
    ndim = coords.shape[1]
    grids = grid_coords          # reuse grid if provided (critical for ADMM)
    components = []
    for d in range(ndim):
        grids, fg = scatter_to_grid(
            coords, disp[:, d], grid_step, smoothness, grids
        )
        components.append(fg)
    return grids, np.array(components)  # shape (D, *grid_shape)


# ═══════════════════════════════════════════════════════════════
#  Bounded scatter → Grid  (Cell-Tracker model, no extrapolation)
#  Mirrors CellTracker/backend/fields.py: linear inside the convex
#  hull, ZERO outside it, then an optional Gaussian blur.  Used for
#  the PTV / post-processing field products, where the grid IS the
#  output and is evaluated everywhere — including the empty corners
#  and gaps that the RBF thin-plate-spline path blows up on.
# ═══════════════════════════════════════════════════════════════

def scatter_to_grid_bounded(
    coords: np.ndarray,
    values: np.ndarray,
    grid_step: np.ndarray,
    smoothing_sigma: float = 0.0,
    grid_coords: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Interpolate scattered data onto a grid **without extrapolating**.

    This is the original Cell-Tracker approach (``CellTracker/backend/fields.py``):
    piecewise-linear interpolation inside the convex hull of the points, filled
    with ``0`` outside it (``griddata`` / ``LinearNDInterpolator`` return NaN
    there), followed by an optional Gaussian blur of ``smoothing_sigma`` grid
    cells.  Unlike :func:`scatter_to_grid`'s ``thin_plate_spline`` RBF branch,
    the output can never exceed the range of ``values`` — so it does not blow up
    in the empty corners / inter-cluster gaps of the bounding-box grid.

    Parameters
    ----------
    coords : (N, D) — scattered point positions
    values : (N,)   — scalar field values at those points
    grid_step : (D,) — grid spacing per dimension
    smoothing_sigma : float — Gaussian blur sigma in *grid cells* (0 = none)
    grid_coords : optional pre-built meshgrid arrays
    """
    ndim = coords.shape[1]
    gs = np.asarray(grid_step, dtype=np.float64)

    if grid_coords is not None:
        grids = grid_coords
    else:
        axes = []
        for d in range(ndim):
            lo, hi = coords[:, d].min(), coords[:, d].max()
            axes.append(np.arange(lo, hi + gs[d], gs[d]))
        grids = tuple(np.meshgrid(*axes, indexing="ij"))

    query_pts = np.column_stack([g.ravel() for g in grids])

    if len(coords) < ndim + 1:
        log.warning("scatter_to_grid_bounded: only %d points (need %d) — returning zeros",
                     len(coords), ndim + 1)
        return grids, np.zeros(grids[0].shape, dtype=np.float64)

    # Linear interpolation inside the hull; 0 outside it (fill_value=0.0 in
    # _linear_or_nearest_interpolate). No extrapolation, ever.
    f_grid = _linear_or_nearest_interpolate(coords, values, query_pts).reshape(
        grids[0].shape)

    if smoothing_sigma and smoothing_sigma > 0:
        f_grid = gaussian_filter(f_grid, sigma=float(smoothing_sigma))
    return grids, f_grid


def scatter_to_grid_multi_bounded(
    coords: np.ndarray,
    disp: np.ndarray,
    grid_step: np.ndarray,
    smoothing_sigma: float = 0.0,
    grid_coords: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Multi-component :func:`scatter_to_grid_bounded` (Cell-Tracker model).

    Returns ``(grids, disp_grid)`` with ``disp_grid`` of shape
    ``(D, *grid_shape)`` — bounded, never extrapolated.
    """
    ndim = coords.shape[1]
    grids = grid_coords
    components = []
    for d in range(ndim):
        grids, fg = scatter_to_grid_bounded(
            coords, disp[:, d], grid_step, smoothing_sigma, grids
        )
        components.append(fg)
    return grids, np.array(components)  # shape (D, *grid_shape)


# ═══════════════════════════════════════════════════════════════
#  Global Solver 1: Moving Least Squares (MLS)
#  Replaces funCompDefGrad3.m
# ═══════════════════════════════════════════════════════════════

def solve_mls(
    disp: np.ndarray,
    coords: np.ndarray,
    query_coords: np.ndarray,
    f_o_s: float,
    n_neighbors: int,
) -> np.ndarray:
    """Moving least-squares displacement interpolation.

    For each particle, fits a local affine model:
        u(x) = u0 + du/dx * (x - x0)
    using its K nearest neighbors within f_o_s, then evaluates
    the fitted displacement at the query point.

    Replaces the gbSolver==1 branch + ``funCompDefGrad3.m``.

    Parameters
    ----------
    disp : (M, D)  — displacement at matched particles
    coords : (M, D) — positions of those matched particles
    query_coords : (Nq, D) — positions at which to evaluate
    f_o_s : float — field of search radius
    n_neighbors : int — max neighbors to use per point

    Returns
    -------
    disp_at_query : (Nq, D) — interpolated displacements
    """
    ndim = coords.shape[1]
    Nq = len(query_coords)
    result = np.zeros((Nq, ndim), dtype=np.float64)

    if len(coords) < ndim + 1:
        return result

    tree = cKDTree(coords)
    K = min(n_neighbors, len(coords) - 1)
    radius = np.sqrt(ndim) * f_o_s

    # Bulk KNN query
    dd, ii = tree.query(query_coords, k=K + 1)

    for qi in range(Nq):
        # Filter to within radius
        mask = dd[qi] < radius
        idx = ii[qi][mask]
        if len(idx) < ndim + 1:
            # Fall back to nearest value
            if len(idx) > 0:
                result[qi] = disp[idx[0]]
            continue

        # Local coords relative to query point
        dx = coords[idx] - query_coords[qi]  # (k, D)
        # Build [1, dx1, dx2, (dx3)] design matrix
        A = np.column_stack([np.ones(len(idx)), dx])  # (k, D+1)

        # Solve for each displacement component
        for d in range(ndim):
            try:
                params, _, _, _ = np.linalg.lstsq(A, disp[idx, d], rcond=None)
                result[qi, d] = params[0]  # constant term = value at query
            except np.linalg.LinAlgError:
                result[qi, d] = np.mean(disp[idx, d])

    return result


# ═══════════════════════════════════════════════════════════════
#  Global Solver 2: Grid Regularization
#  Replaces the gbSolver==2 branch
# ═══════════════════════════════════════════════════════════════

def solve_regularization(
    disp: np.ndarray,
    coords: np.ndarray,
    query_coords: np.ndarray,
    grid_step: np.ndarray,
    smoothness: float,
    grid_coords: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], np.ndarray]:
    """Scatter → regularised grid → interpolate back to particles.

    Replaces the gbSolver==2 branch in f_track_serial_match3D.m.

    Returns
    -------
    disp_at_query : (Nq, D)
    grids : grid coordinate arrays (cached for next iteration)
    disp_grid : (D, *grid_shape)
    """
    ndim = coords.shape[1]

    # Scatter to grid with smoothing
    grids, disp_grid = scatter_to_grid_multi(
        coords, disp, grid_step, smoothness, grid_coords
    )

    # Interpolate grid back to scattered query points
    disp_at_query = _interp_grid_to_points(grids, disp_grid, query_coords)

    return disp_at_query, grids, disp_grid


def _interp_grid_to_points(
    grids: Tuple[np.ndarray, ...],
    field: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Interpolate a gridded D-component field to scattered points.

    Uses LinearNDInterpolator for speed.
    """
    ndim = points.shape[1]
    grid_pts = np.column_stack([g.ravel() for g in grids])
    result = np.zeros((len(points), ndim), dtype=np.float64)
    for d in range(ndim):
        result[:, d] = _linear_or_nearest_interpolate(
            grid_pts, field[d].ravel(), points
        )
    return result


# ═══════════════════════════════════════════════════════════════
#  Global Solver 3: ADMM / Augmented Lagrangian
#  Replaces the gbSolver==3 branch  (most complex)
# ═══════════════════════════════════════════════════════════════

class ADMMLSolver:
    """Augmented Lagrangian solver for displacement regularisation.

    On the first ADMM outer iteration it:
      1. Scatter-interpolates to a regular grid.
      2. Builds a sparse gradient operator D (central finite differences).
      3. Tunes regularisation alpha via L-curve.
      4. Initialises dual variable v.

    On subsequent iterations it reuses the grid and dual variable,
    performing one ADMM update:
        u_hat = (alpha*D'D + I)^{-1} (u - v)
        v     = v + u_hat - u

    Replaces the gbSolver==3 branch in ``f_track_serial_match3D.m``
    plus ``funDerivativeOp3.m`` (~200 lines of sparse index logic).
    """

    def __init__(self):
        self.grids: Optional[Tuple[np.ndarray, ...]] = None
        self.v_dual: Optional[np.ndarray] = None
        self.alpha: float = 1.0
        self.D: Optional[csc_matrix] = None
        self._DtD: Optional[csc_matrix] = None
        self._n_grid: int = 0

    def solve(
        self,
        disp: np.ndarray,
        coords: np.ndarray,
        query_coords: np.ndarray,
        grid_step: np.ndarray,
        smoothness: float,
        is_first_iter: bool,
        roi_ranges: Optional[Tuple[Tuple[float,float], ...]] = None,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], np.ndarray]:
        """One ADMM global-step update.

        Parameters
        ----------
        disp, coords : matched displacement and positions
        query_coords : all current B particle positions
        grid_step : grid spacing
        smoothness : regularisation weight (used in initial scatter)
        is_first_iter : True on first ADMM iteration (triggers L-curve)
        roi_ranges : optional ((xmin,xmax), (ymin,ymax), ...) for grid

        Returns
        -------
        disp_at_query, grids, disp_grid — same interface as solve_regularization
        """
        ndim = coords.shape[1]

        # 1. Scatter to grid
        grids, disp_grid = scatter_to_grid_multi(
            coords, disp, grid_step, smoothness, self.grids
        )
        self.grids = grids
        grid_shape = grids[0].shape
        n_pts = int(np.prod(grid_shape))

        # 2. Interleave components: [u0_x, u0_y, u0_z, u1_x, ...]
        u_vec = np.zeros(ndim * n_pts, dtype=np.float64)
        for d in range(ndim):
            u_vec[d::ndim] = disp_grid[d].ravel()

        # 3. Build gradient operator on first iteration (or if grid changed)
        need_rebuild = is_first_iter
        if (not is_first_iter
                and self._DtD is not None
                and self._DtD.shape[0] != ndim * n_pts):
            log.warning("ADMM grid shape changed (%d → %d) — rebuilding operators",
                        self._DtD.shape[0], ndim * n_pts)
            need_rebuild = True

        if need_rebuild:
            self.D = _build_gradient_operator(grid_shape, grid_step, ndim)
            self._DtD = self.D.T @ self.D
            self.v_dual = np.zeros_like(u_vec)
            self.alpha = self._tune_alpha(u_vec, ndim * n_pts)

        # 4. ADMM update
        A = self.alpha * self._DtD + speye(ndim * n_pts, format="csc")
        rhs = u_vec - self.v_dual
        u_hat = spsolve(A, rhs)

        # 5. Update dual
        self.v_dual = self.v_dual + u_hat - u_vec

        # 6. Reshape back to grid
        disp_grid_out = np.zeros_like(disp_grid)
        for d in range(ndim):
            disp_grid_out[d] = u_hat[d::ndim].reshape(grid_shape)

        # 7. Interpolate to query points
        disp_at_query = _interp_grid_to_points(grids, disp_grid_out, query_coords)

        return disp_at_query, grids, disp_grid_out

    def _tune_alpha(self, u_vec: np.ndarray, n: int) -> float:
        """L-curve method to find best regularisation alpha.

        Tests a log-spaced list and fits a parabola to find the elbow.
        """
        alpha_list = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
        err_fid = np.zeros(len(alpha_list))
        err_smooth = np.zeros(len(alpha_list))
        I_n = speye(n, format="csc")

        for i, alpha in enumerate(alpha_list):
            A = alpha * self._DtD + I_n
            u_hat = spsolve(A, u_vec)  # v_dual is 0 on first call
            diff = u_hat - u_vec
            err_fid[i] = np.sqrt(diff @ diff)
            Du = self.D @ u_hat
            err_smooth[i] = np.sqrt(Du @ Du)

        # Normalise and sum
        ef = err_fid / (err_fid.max() + 1e-30)
        es = err_smooth / (err_smooth.max() + 1e-30)
        total = ef + es
        idx_best = int(np.argmin(total))

        # Parabolic refinement around minimum
        if 0 < idx_best < len(alpha_list) - 1:
            log_a = np.log10(alpha_list[idx_best - 1:idx_best + 2])
            y = total[idx_best - 1:idx_best + 2]
            try:
                p = np.polyfit(log_a, y, 2)
                if abs(p[0]) > 1e-15:
                    best = 10 ** (-p[1] / (2 * p[0]))
                    log.info("ADMM alpha tuned to %.4g", best)
                    return float(best)
            except Exception:
                pass

        best = float(alpha_list[idx_best])
        log.info("ADMM alpha selected: %.4g", best)
        return best


# ═══════════════════════════════════════════════════════════════
#  Sparse gradient operator (central finite differences)
#  Replaces funDerivativeOp3.m  (~200 lines → ~40 lines)
# ═══════════════════════════════════════════════════════════════

def _build_gradient_operator(
    grid_shape: Tuple[int, ...],
    grid_step: np.ndarray,
    ndim: int,
) -> csc_matrix:
    """Build a sparse central-difference gradient operator D.

    Such that F_vec = D @ u_vec, where:
      u_vec has ndim components interleaved per grid point
      F_vec has ndim² components interleaved per grid point

    For 3-D: output layout per point is
        [F11, F21, F31, F12, F22, F32, F13, F23, F33]
    matching the MATLAB convention in funDerivativeOp3.m.

    Uses np.gradient internally for the actual stencil logic,
    but builds a sparse matrix for use in the ADMM linear system.
    """
    n_pts = int(np.prod(grid_shape))
    n_u = ndim * n_pts          # size of u_vec
    n_f = ndim * ndim * n_pts   # size of F_vec

    # For each spatial derivative direction and each displacement component,
    # build one row-block of D using sparse difference stencils.
    rows, cols, vals = [], [], []

    strides = _compute_strides(grid_shape)  # linear index strides per dim

    for deriv_dim in range(ndim):           # which spatial direction (∂/∂x_d)
        h = float(grid_step[deriv_dim])
        stride = strides[deriv_dim]
        sz = grid_shape[deriv_dim]

        for comp in range(ndim):            # which displacement component
            # F row index within the ndim² block:
            #   MATLAB layout: F_{comp+1, deriv_dim+1}
            #   row offset = deriv_dim * ndim + comp
            f_offset = deriv_dim * ndim + comp

            for p in range(n_pts):
                f_row = ndim * ndim * p + f_offset

                # Multi-index of this grid point
                mi = _linear_to_multi(p, grid_shape)
                idx_along = mi[deriv_dim]

                # Central difference where possible, one-sided at borders
                if 0 < idx_along < sz - 1:
                    # central: (u[i+1] - u[i-1]) / (2h)
                    p_plus = p + stride
                    p_minus = p - stride
                    rows.append(f_row); cols.append(ndim * p_minus + comp); vals.append(-1.0 / (2 * h))
                    rows.append(f_row); cols.append(ndim * p_plus  + comp); vals.append( 1.0 / (2 * h))
                elif idx_along == 0:
                    # forward: (-u[i] + u[i+1]) / h  (first-order)
                    p_plus = p + stride
                    rows.append(f_row); cols.append(ndim * p       + comp); vals.append(-1.0 / h)
                    rows.append(f_row); cols.append(ndim * p_plus  + comp); vals.append( 1.0 / h)
                else:  # idx_along == sz - 1
                    # backward: (-u[i-1] + u[i]) / h
                    p_minus = p - stride
                    rows.append(f_row); cols.append(ndim * p_minus + comp); vals.append(-1.0 / h)
                    rows.append(f_row); cols.append(ndim * p       + comp); vals.append( 1.0 / h)

    D = csc_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(n_f, n_u),
    )
    return D


def _compute_strides(shape: Tuple[int, ...]) -> list:
    """Linear-index stride for each dimension (C-order)."""
    nd = len(shape)
    strides = [1] * nd
    for d in range(nd - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return strides


def _linear_to_multi(idx: int, shape: Tuple[int, ...]) -> list:
    """Convert a linear index to a multi-index (C-order)."""
    nd = len(shape)
    mi = [0] * nd
    for d in range(nd - 1, -1, -1):
        mi[d] = idx % shape[d]
        idx //= shape[d]
    return mi


# ═══════════════════════════════════════════════════════════════
#  Unified global-step dispatcher
# ═══════════════════════════════════════════════════════════════

class DisplacementRegularizer:
    """Unified interface to all three global-step solvers.

    Maintains state for the ADMM solver across iterations.

    Usage (inside the ADMM tracking loop)::

        reg = DisplacementRegularizer(solver=GlobalSolver.ADMM)
        for iter_num in range(max_iter):
            ...  # local step: get matched_disp, matched_coords
            smooth_disp = reg.solve(
                matched_disp, matched_coords, all_b_coords,
                grid_step, smoothness, f_o_s, n_neighbors,
                is_first_iter=(iter_num == 0),
            )
    """

    def __init__(self, solver: GlobalSolver = GlobalSolver.REGULARIZATION):
        self.solver_type = solver
        self._admm = ADMMLSolver() if solver == GlobalSolver.ADMM else None
        self.grids: Optional[Tuple[np.ndarray, ...]] = None
        self.disp_grid: Optional[np.ndarray] = None

    def solve(
        self,
        disp: np.ndarray,
        coords: np.ndarray,
        query_coords: np.ndarray,
        grid_step: np.ndarray,
        smoothness: float,
        f_o_s: float = 60.0,
        n_neighbors: int = 20,
        is_first_iter: bool = False,
    ) -> np.ndarray:
        """Run one global-step solve.

        Returns (Nq, D) displacement at query_coords.
        """
        if len(coords) == 0:
            return np.zeros_like(query_coords)

        gs = np.asarray(grid_step, dtype=np.float64)

        if self.solver_type == GlobalSolver.MLS:
            return solve_mls(disp, coords, query_coords, f_o_s, n_neighbors)

        elif self.solver_type == GlobalSolver.REGULARIZATION:
            result, self.grids, self.disp_grid = solve_regularization(
                disp, coords, query_coords, gs, smoothness, self.grids
            )
            return result

        elif self.solver_type == GlobalSolver.ADMM:
            result, self.grids, self.disp_grid = self._admm.solve(
                disp, coords, query_coords, gs, smoothness, is_first_iter
            )
            return result

        raise ValueError(f"Unknown solver: {self.solver_type}")



# ======================================================================
# VENDORED FROM: nd2studios/backend/serialtrack/fields.py
# ======================================================================

"""
SerialTrack Python — Displacement & strain field computation
=============================================================
    serialtrack/fields.py

Replaces: funDerivativeOp3 (post-processing usage),
          funCompDefGrad3 (strain gauge), postprocessing sections.

Fixes from v1
-------------
- Fixed logger name: was "serialtrack.regularization", now "serialtrack.fields".
- Fixed ``DisplacementField.gradient()``: was fragile inference of grid
  spacing from meshgrid diff.  Now uses the actual grid axis spacing
  directly from the meshgrid arrays (simple ``np.gradient`` with axis spacing).
- Fixed ``DisplacementField.velocity``: broadcasting was fragile for 2D
  (hardcoded 4 trailing ``None`` dimensions).  Now uses generic reshaping.
- Fixed ``StrainField``: added ``pixel_steps`` attribute for serialization
  compatibility.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass, field as dc_field
import logging


log = logging.getLogger("serialtrack.fields")


# ═══════════════════════════════════════════════════════════════
#  Displacement field
# ═══════════════════════════════════════════════════════════════

@dataclass
class DisplacementField:
    """Gridded displacement field with metadata.

    Stores displacement components on a regular grid and provides
    gradient (strain) computation via ``np.gradient``.

    Attributes
    ----------
    grids : tuple of D ndarray
        Meshgrid coordinate arrays (one per spatial dimension).
    components : (D, *grid_shape) ndarray
        Displacement components on the grid.
    pixel_steps : (D,) ndarray
        Physical size per pixel in each dimension.
    time_step : float
        Time between frames (default 1.0).
    """
    grids: Tuple[np.ndarray, ...]
    components: np.ndarray
    pixel_steps: np.ndarray
    time_step: float = 1.0

    @property
    def ndim(self) -> int:
        return len(self.grids)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.grids[0].shape

    @property
    def velocity(self) -> np.ndarray:
        """Displacement / time_step → velocity field, in physical units."""
        ndim = self.ndim
        # Build a shape like (D, 1, 1, ...) for broadcasting
        ps_shape = [ndim] + [1] * ndim
        ps = self.pixel_steps.reshape(ps_shape)
        return self.components * ps / self.time_step

    def _axis_spacings(self) -> list:
        """Extract grid spacing along each axis from the meshgrid arrays.

        For axis ``d``, we extract the unique 1-D coordinates from
        ``self.grids[d]`` and compute the step.  Falls back to
        ``pixel_steps[d]`` if the grid has only one point along that axis.
        """
        spacings = []
        for d in range(self.ndim):
            # Extract the unique coordinates along axis d
            # (meshgrid arrays repeat; taking a 1-D slice is cheapest)
            idx = [0] * self.ndim
            idx[d] = slice(None)
            axis_coords = self.grids[d][tuple(idx)]
            if len(axis_coords) > 1:
                step = float(np.median(np.diff(axis_coords)))
                spacings.append(step)
            else:
                spacings.append(float(self.pixel_steps[d]))
        return spacings

    def gradient(self) -> np.ndarray:
        """Compute deformation gradient tensor F = ∂u/∂x.

        Returns
        -------
        F : (D, D, *grid_shape) — F[i, j] = ∂u_i / ∂x_j
        """
        ndim = self.ndim
        spacings = self._axis_spacings()
        F = np.zeros((ndim, ndim, *self.shape), dtype=np.float64)

        for i in range(ndim):
            # np.gradient returns a list of arrays (one per axis) for ndim > 1
            grads = np.gradient(self.components[i], *spacings)
            if ndim == 1:
                grads = [grads]
            for j in range(ndim):
                F[i, j] = grads[j]
        return F

    def strain(self) -> np.ndarray:
        """Infinitesimal strain tensor ε = 0.5*(F + F^T).

        Returns (D, D, *grid_shape).
        """
        F = self.gradient()
        return 0.5 * (F + F.transpose(1, 0, *range(2, 2 + self.ndim)))

    def to_physical(self) -> 'DisplacementField':
        """Return a new field with displacements in physical units."""
        ndim = self.ndim
        phys = self.components.copy()
        for d in range(ndim):
            phys[d] *= self.pixel_steps[d]
        return DisplacementField(
            grids=self.grids,
            components=phys,
            pixel_steps=self.pixel_steps,
            time_step=self.time_step,
        )


# ═══════════════════════════════════════════════════════════════
#  Strain field
# ═══════════════════════════════════════════════════════════════

@dataclass
class StrainField:
    """Pre-computed strain field with components.

    Attributes
    ----------
    grids : tuple of D ndarray
        Meshgrid coordinate arrays.
    F_tensor : (D, D, *grid_shape) ndarray
        Deformation gradient tensor.
    eps_tensor : (D, D, *grid_shape) ndarray
        Infinitesimal strain tensor.
    pixel_steps : (D,) ndarray
        Physical size per pixel (for serialization round-trip).
    """
    grids: Tuple[np.ndarray, ...]
    F_tensor: np.ndarray
    eps_tensor: np.ndarray
    pixel_steps: np.ndarray = dc_field(default_factory=lambda: np.ones(3))


# ═══════════════════════════════════════════════════════════════
#  Scattered MLS strain gauge
# ═══════════════════════════════════════════════════════════════

def compute_strain_mls(
    disp: np.ndarray,
    coords: np.ndarray,
    f_o_s: float,
    n_neighbors: int,
    pixel_steps: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving least-squares strain gauge at each particle.

    Replaces ``funCompDefGrad3.m`` for post-processing strain
    computation.  Fits u(x) = u0 + F·(x - x0) at each particle
    using its neighbors.

    Parameters
    ----------
    disp : (N, D) — displacement at each particle
    coords : (N, D) — particle positions
    f_o_s, n_neighbors : search parameters
    pixel_steps : optional physical scaling

    Returns
    -------
    U : (N, D) — fitted displacement (smoothed)
    F_tensor : (N, D, D) — deformation gradient at each particle
    valid : (N,) bool — which particles had a valid fit
    """
    ndim = coords.shape[1]
    N = len(coords)
    U = np.full((N, ndim), np.nan)
    F_tensor = np.full((N, ndim, ndim), np.nan)
    valid = np.zeros(N, dtype=bool)

    if N < ndim + 1:
        return U, F_tensor, valid

    tree = cKDTree(coords)
    K = min(n_neighbors, N - 1)
    radius = np.sqrt(ndim) * f_o_s
    dd, ii = tree.query(coords, k=K + 1)

    for p in range(N):
        # Neighbors within radius
        mask = dd[p] < radius
        idx = ii[p][mask]
        if len(idx) < ndim + 1:
            U[p] = disp[p]
            continue

        # Design matrix: [1, (x-x0), (y-y0), (z-z0)]
        dx = coords[idx] - coords[p]
        A = np.column_stack([np.ones(len(idx)), dx])

        try:
            for d in range(ndim):
                params, _, _, _ = np.linalg.lstsq(A, disp[idx, d], rcond=None)
                U[p, d] = params[0]
                F_tensor[p, d, :] = params[1:]  # ∂u_d/∂x_j
            valid[p] = True
        except np.linalg.LinAlgError:
            U[p] = disp[p]

    # Apply physical scaling if provided
    if pixel_steps is not None:
        ps = np.asarray(pixel_steps)
        for i in range(ndim):
            for j in range(ndim):
                F_tensor[valid, i, j] *= ps[i] / ps[j]

    return U, F_tensor, valid


# ═══════════════════════════════════════════════════════════════
#  Full post-processing pipeline
# ═══════════════════════════════════════════════════════════════

def compute_gridded_strain(
    coords: np.ndarray,
    disp: np.ndarray,
    grid_step: np.ndarray,
    smoothness: float = 1e-3,
    pixel_steps: Optional[np.ndarray] = None,
) -> Tuple[DisplacementField, StrainField]:
    """Full post-processing: scatter → grid → gradient → strain.

    Replaces the postprocessing sections of
    ``run_Serial_MPT_3D_hardpar_accum.m``.

    Uses the **bounded** (Cell-Tracker) scatter-to-grid: linear interpolation
    inside the convex hull of the tracked particles, zero outside it, then an
    optional Gaussian blur. This is the field-*output* path (the grid is what we
    plot), so it must never extrapolate — the RBF ``thin_plate_spline`` path used
    by the tracking global-step solvers grows unbounded (``r²·log r``) in the
    empty corners / inter-cluster gaps of the bounding-box grid, producing giant
    field vectors from tiny per-particle displacements. ``smoothness`` is
    therefore interpreted here as a **Gaussian smoothing sigma in grid cells**
    (matching Cell-Tracker's ``sigma``), not an RBF regularisation weight.

    Returns both the gridded displacement field and strain field.
    """
    ndim = coords.shape[1]
    ps = np.ones(ndim) if pixel_steps is None else np.asarray(pixel_steps)

    grids, disp_grid = scatter_to_grid_multi_bounded(
        coords, disp, grid_step, smoothness
    )

    dfield = DisplacementField(
        grids=grids,
        components=disp_grid,
        pixel_steps=ps,
    )

    F = dfield.gradient()
    eps = 0.5 * (F + F.transpose(1, 0, *range(2, 2 + ndim)))

    sfield = StrainField(grids=grids, F_tensor=F, eps_tensor=eps,
                         pixel_steps=ps)

    return dfield, sfield


# ======================================================================
# VENDORED FROM: nd2studios/backend/serialtrack/prediction.py
# ======================================================================

"""
SerialTrack Python — Chunk 4a
==============================
    serialtrack/prediction.py

Initial guess predictor for ADMM warm-starting.
Replaces: funInitGuess3.m  (~80 lines)
         funPOR_GPR.m      (~70 lines)

Uses sklearn PCA + GaussianProcessRegressor instead of custom POD + fitrgp.

Dependencies:
    pip install numpy scipy scikit-learn
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import logging

log = logging.getLogger("serialtrack.prediction")


def _interp_prev_to_current(
    prev_coords: np.ndarray,
    prev_disp: np.ndarray,
    current_coords: np.ndarray,
) -> np.ndarray:
    """Interpolate a previous displacement field to current particle positions.

    Replaces the repeated scatteredInterpolant calls in funInitGuess3.m.
    Returns (N_current, D) displacement array.
    """
    ndim = prev_coords.shape[1]
    result = np.zeros((len(current_coords), ndim), dtype=np.float64)

    for d in range(ndim):
        interp = LinearNDInterpolator(
            prev_coords, prev_disp[:, d], fill_value=0.0
        )
        result[:, d] = interp(current_coords)

    return result


class InitialGuessPredictor:
    """Predict initial displacement for the next frame.

    Implements three strategies matching the MATLAB funInitGuess3.m logic:

    1. **Frame 3** (1 history frame): Linear extrapolation → 2× previous.
    2. **Frames 4–6** (2 history frames): Linear extrapolation from 2 prior.
    3. **Frame 7+** (5+ history frames): POD-GPR prediction using sklearn.

    The frame numbering convention: ``frame_idx`` is 1-based to match MATLAB,
    where frame 1 is the reference and frame 2 is the first deformed image.

    Usage
    -----
    >>> predictor = InitialGuessPredictor()
    >>> init_disp = predictor.predict(
    ...     frame_idx=5,
    ...     current_coords=coords_b,
    ...     prev_coords_list=prev_coords,   # list of (N,D) arrays
    ...     prev_disp_list=prev_disp,        # list of (N,D) arrays
    ... )
    """

    def __init__(self, n_pod_modes: int = 3, n_history: int = 5):
        """
        Parameters
        ----------
        n_pod_modes : int
            Number of POD basis vectors for the GPR predictor.
        n_history : int
            Number of past frames to use for POD-GPR (default 5).
        """
        self.n_pod_modes = n_pod_modes
        self.n_history = n_history

    def predict(
        self,
        frame_idx: int,
        current_coords: np.ndarray,
        prev_coords_list: List[np.ndarray],
        prev_disp_list: List[np.ndarray],
    ) -> np.ndarray:
        """Compute initial displacement guess for ``current_coords``.

        Parameters
        ----------
        frame_idx : int
            1-based frame index (≥ 3 required for prediction).
        current_coords : (N, D) float64
            Particle positions in the current deformed frame.
        prev_coords_list : list of (Ni, D) arrays
            Particle positions for previous frames.
            Index ``i`` corresponds to frame ``i+2`` (0-based list).
        prev_disp_list : list of (Ni, D) arrays
            Tracked displacements for previous frames (same indexing).

        Returns
        -------
        init_disp : (N, D) float64
            Predicted displacement at ``current_coords``.
        """
        ndim = current_coords.shape[1]
        N = len(current_coords)

        # Not enough history
        n_avail = len(prev_disp_list)
        if n_avail == 0 or frame_idx < 3:
            return np.zeros((N, ndim), dtype=np.float64)

        if frame_idx == 3 and n_avail >= 1:
            return self._extrapolate_linear_1(
                current_coords, prev_coords_list, prev_disp_list, frame_idx
            )

        if frame_idx <= 6 and n_avail >= 2:
            return self._extrapolate_linear_2(
                current_coords, prev_coords_list, prev_disp_list, frame_idx
            )

        if frame_idx > 6 and n_avail >= self.n_history:
            return self._predict_pod_gpr(
                current_coords, prev_coords_list, prev_disp_list, frame_idx
            )

        # Fallback: simple linear extrapolation from most recent
        if n_avail >= 2:
            return self._extrapolate_linear_2(
                current_coords, prev_coords_list, prev_disp_list, frame_idx
            )
        return self._extrapolate_linear_1(
            current_coords, prev_coords_list, prev_disp_list, frame_idx
        )

    # ── Strategy 1: single-frame extrapolation (frame 3) ──────

    def _extrapolate_linear_1(self, current, prev_c, prev_d, fi):
        """u_init = 2 * u_{n-1} interpolated to current positions."""
        idx = fi - 3  # index into 0-based list
        idx = min(idx, len(prev_d) - 1)
        u_prev = _interp_prev_to_current(prev_c[idx], prev_d[idx], current)
        return 2.0 * u_prev

    # ── Strategy 2: two-frame linear extrapolation (frames 4-6) ──

    def _extrapolate_linear_2(self, current, prev_c, prev_d, fi):
        """u_init = 2*u_{n-1} - u_{n-2}, each interpolated to current."""
        i2 = min(fi - 3, len(prev_d) - 1)  # most recent
        i3 = min(fi - 4, len(prev_d) - 1)  # one before
        if i3 < 0:
            return self._extrapolate_linear_1(current, prev_c, prev_d, fi)

        u2 = _interp_prev_to_current(prev_c[i2], prev_d[i2], current)
        u3 = _interp_prev_to_current(prev_c[i3], prev_d[i3], current)
        return 2.0 * u2 - u3

    # ── Strategy 3: POD-GPR (frame 7+) ───────────────────────

    def _predict_pod_gpr(self, current, prev_c, prev_d, fi):
        """POD + Gaussian Process Regression prediction.

        Replaces funPOR_GPR.m + the ImgSeqNum > 6 branch.

        1. Interpolate the last ``n_history`` displacement fields to
           the current particle positions → snapshot matrix.
        2. Apply POD (PCA) to extract dominant modes.
        3. Fit a GP to each mode's temporal coefficient.
        4. Predict next time step and reconstruct.
        """
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        ndim = current.shape[1]
        N = len(current)
        nT = self.n_history
        nB = min(self.n_pod_modes, nT - 1)

        # Determine which previous frames to use
        # MATLAB: ImgSeqNum+[-5:1:-1] mapped to 0-based list indices
        start = max(0, len(prev_d) - nT)
        end_ = len(prev_d)
        indices = list(range(start, end_))
        nT_actual = len(indices)
        if nT_actual < 2:
            return self._extrapolate_linear_2(current, prev_c, prev_d, fi)

        # Build snapshot matrices: (nT, N) per displacement component
        snapshots = [np.zeros((nT_actual, N)) for _ in range(ndim)]
        t_train = np.arange(nT_actual, dtype=np.float64).reshape(-1, 1)

        for ti, idx in enumerate(indices):
            u_interp = _interp_prev_to_current(prev_c[idx], prev_d[idx], current)
            for d in range(ndim):
                snapshots[d][ti, :] = u_interp[:, d]

        # Predict next time for each component
        t_predict = np.array([[float(nT_actual)]]) 
        result = np.zeros((N, ndim), dtype=np.float64)

        for d in range(ndim):
            result[:, d] = self._pod_gpr_1d(
                snapshots[d], t_train, t_predict, nB
            )

        log.debug("POD-GPR prediction for frame %d using %d snapshots, %d modes",
                   fi, nT_actual, nB)
        return result

    @staticmethod
    def _pod_gpr_1d(
        T_snap: np.ndarray,
        t_train: np.ndarray,
        t_predict: np.ndarray,
        n_modes: int,
    ) -> np.ndarray:
        """POD-GPR for a single displacement component.

        Parameters
        ----------
        T_snap : (nT, N) — snapshot matrix
        t_train : (nT, 1) — training times
        t_predict : (1, 1) — prediction time
        n_modes : int — number of POD modes

        Returns
        -------
        u_pred : (N,) — predicted field at t_predict
        """
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, WhiteKernel, ConstantKernel
        )

        nT, N = T_snap.shape
        n_modes = min(n_modes, nT - 1, N)
        if n_modes < 1:
            return T_snap[-1]  # fallback: return last snapshot

        # --- POD via sklearn PCA ---
        # PCA centres the data (subtracts mean) automatically
        pca = PCA(n_components=n_modes)
        # a_train: (nT, n_modes) — temporal coefficients
        a_train = pca.fit_transform(T_snap)

        # --- GP regression on each mode's temporal coefficient ---
        a_pred = np.zeros((1, n_modes))

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(
            noise_level=1e-4, noise_level_bounds=(1e-8, 1e0)
        )

        for k in range(n_modes):
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=2,
                alpha=1e-6,
            )
            gpr.fit(t_train, a_train[:, k])
            a_pred[0, k] = gpr.predict(t_predict)[0]

        # --- Reconstruct ---
        u_pred = pca.inverse_transform(a_pred)  # (1, N)
        return u_pred[0]

# ======================================================================
# VENDORED FROM: nd2studios/backend/serialtrack/trajectories.py
# ======================================================================

"""
SerialTrack Python — Trajectory stitching & merging
=====================================================
    serialtrack/trajectories.py

Replaces the ~200-line trajectory merge/stitch section from:
    - run_Serial_MPT_3D_hardpar_inc.m  (incremental postprocessing)
    - run_Serial_MPT_3D_hardpar_accum.m (cumulative trajectory collection)

The MATLAB code:
  1. Builds trajectory segments from frame-to-frame track_A2B links
  2. Extrapolates each segment forward/backward using pchip/nearest
  3. Searches for other segments whose endpoints are near the
     extrapolated predictions
  4. Merges matching segments and fills gaps via interpolation
  5. Repeats for multiple merge passes

This Python version uses scipy.interpolate.PchipInterpolator and
vectorised numpy operations for the core logic.

Dependencies:
    pip install numpy scipy
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.interpolate import PchipInterpolator
import logging


log = logging.getLogger("serialtrack.trajectories")


# ═══════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrajectorySegment:
    """A single trajectory segment (possibly with NaN gaps).

    Attributes
    ----------
    coords : (n_frames, D) float64
        Particle positions per frame. NaN where not tracked.
    start_frame : int
        First non-NaN frame index (0-based).
    length : int
        Number of consecutive non-NaN frames.
    active : bool
        False if this segment has been merged into another.
    """
    coords: np.ndarray
    start_frame: int = 0
    length: int = 0
    active: bool = True

    def recompute_bounds(self):
        """Recompute start_frame and length from coords."""
        valid = ~np.isnan(self.coords[:, 0])
        indices = np.where(valid)[0]
        if len(indices) == 0:
            self.start_frame = 0
            self.length = 0
            self.active = False
        else:
            self.start_frame = int(indices[0])
            self.length = int(np.sum(valid))

    @property
    def end_frame(self) -> int:
        """Last non-NaN frame (exclusive)."""
        return self.start_frame + self.length

    def valid_coords(self) -> np.ndarray:
        """Return only non-NaN rows."""
        valid = ~np.isnan(self.coords[:, 0])
        return self.coords[valid]


# ═══════════════════════════════════════════════════════════════
#  Build trajectory segments from incremental tracking results
# ═══════════════════════════════════════════════════════════════

def build_segments_incremental(
    coords_per_frame: List[np.ndarray],
    track_a2b_per_frame: List[np.ndarray],
) -> List[TrajectorySegment]:
    """Build trajectory segments from incremental frame-to-frame links.

    Replaces the MATLAB "Compute and collect all trajectory segments"
    section in run_Serial_MPT_3D_hardpar_inc.m.

    Parameters
    ----------
    coords_per_frame : list of (Ni, D) arrays
        Detected particle coordinates per frame (0-indexed).
        coords_per_frame[0] = reference frame particles.
    track_a2b_per_frame : list of (Ni,) int arrays
        track_a2b_per_frame[i] maps frame i → frame i+1.
        Value -1 means untracked.

    Returns
    -------
    segments : list of TrajectorySegment
    """
    n_frames = len(coords_per_frame)
    ndim = coords_per_frame[0].shape[1]
    segments: List[TrajectorySegment] = []

    # For each starting frame, trace forward through the links
    for start_f in range(n_frames):
        n_particles = len(coords_per_frame[start_f])

        for p_idx in range(n_particles):
            coords = np.full((n_frames, ndim), np.nan)
            coords[start_f] = coords_per_frame[start_f][p_idx]

            current_idx = p_idx
            for f in range(start_f, n_frames - 1):
                track = track_a2b_per_frame[f]
                if current_idx < len(track) and track[current_idx] >= 0:
                    next_idx = track[current_idx]
                    coords[f + 1] = coords_per_frame[f + 1][next_idx]
                    current_idx = next_idx
                else:
                    break  # chain broken

            # Only keep if we have at least 2 valid frames
            valid_count = int(np.sum(~np.isnan(coords[:, 0])))
            if valid_count >= 2:
                seg = TrajectorySegment(coords=coords)
                seg.recompute_bounds()
                segments.append(seg)

    # Deduplicate: segments sharing the same (frame, position) are redundant
    segments = _deduplicate_segments(segments)

    log.info("Built %d trajectory segments from %d frames",
             len(segments), n_frames)
    return segments


def build_segments_cumulative(
    coords_ref: np.ndarray,
    coords_per_frame: List[np.ndarray],
    track_a2b_per_frame: List[np.ndarray],
) -> List[TrajectorySegment]:
    """Build trajectory segments from cumulative tracking results.

    In cumulative mode, every track_a2b maps reference (frame 0) → frame i.
    This is simpler — each reference particle has one trajectory.

    Parameters
    ----------
    coords_ref : (Na, D) — reference frame particles
    coords_per_frame : list of (Ni, D) — detected per deformed frame
    track_a2b_per_frame : list of (Na,) int — ref→frame_i links

    Returns
    -------
    segments : list of TrajectorySegment
    """
    n_frames = len(track_a2b_per_frame) + 1  # +1 for reference
    ndim = coords_ref.shape[1]
    Na = len(coords_ref)
    segments: List[TrajectorySegment] = []

    for p_idx in range(Na):
        coords = np.full((n_frames, ndim), np.nan)
        coords[0] = coords_ref[p_idx]

        for fi, track in enumerate(track_a2b_per_frame):
            if track[p_idx] >= 0:
                coords[fi + 1] = coords_per_frame[fi][track[p_idx]]

        valid_count = int(np.sum(~np.isnan(coords[:, 0])))
        if valid_count >= 1:
            seg = TrajectorySegment(coords=coords)
            seg.recompute_bounds()
            segments.append(seg)

    log.info("Built %d cumulative trajectories", len(segments))
    return segments


# ═══════════════════════════════════════════════════════════════
#  Trajectory segment merging
# ═══════════════════════════════════════════════════════════════

def merge_segments(
    segments: List[TrajectorySegment],
    config: TrajectoryConfig,
) -> List[TrajectorySegment]:
    """Merge trajectory segments by extrapolation and proximity matching.

    Replaces the ~150-line "Merge trajectory segments" section from
    run_Serial_MPT_3D_hardpar_inc.m.

    Algorithm (per merge pass):
      For each gap size g in [0, max_gap_length]:
        For segment lengths from longest to min_segment_length:
          For each segment S of that length:
            1. Extrapolate S forward/backward using pchip/nearest.
            2. Find shorter segments whose start/end is near the
               extrapolated prediction (within dist_threshold).
            3. Merge the best candidate into S, fill the gap.

    Parameters
    ----------
    segments : list of TrajectorySegment
    config : TrajectoryConfig

    Returns
    -------
    merged : list of TrajectorySegment (only active ones)
    """
    if not segments:
        return segments

    n_frames = segments[0].coords.shape[0]
    ndim = segments[0].coords.shape[1]

    for merge_pass in range(config.merge_passes):
        n_merged_this_pass = 0

        for gap in range(config.max_gap_length + 1):

            # Process from longest to shortest segments
            max_len = n_frames - 1
            for seg_len in range(max_len, config.min_segment_length - 1, -1):

                # Collect segments of this length
                for i, seg_i in enumerate(segments):
                    if not seg_i.active or seg_i.length != seg_len:
                        continue

                    # Try to extend in the forward direction
                    target_frame = seg_i.end_frame + gap
                    if 0 <= target_frame < n_frames:
                        pred_pos = _extrapolate_position(
                            seg_i, target_frame, config.extrap_method
                        )
                        if pred_pos is not None:
                            best_j = _find_best_candidate(
                                segments, i, target_frame,
                                pred_pos, config.dist_threshold,
                                direction="forward",
                            )
                            if best_j >= 0:
                                _merge_into(segments[i], segments[best_j],
                                            config.extrap_method)
                                n_merged_this_pass += 1

                    # Try to extend in the backward direction
                    target_frame = seg_i.start_frame - 1 - gap
                    if 0 <= target_frame < n_frames:
                        pred_pos = _extrapolate_position(
                            seg_i, target_frame, config.extrap_method
                        )
                        if pred_pos is not None:
                            best_j = _find_best_candidate(
                                segments, i, target_frame,
                                pred_pos, config.dist_threshold,
                                direction="backward",
                            )
                            if best_j >= 0:
                                _merge_into(segments[i], segments[best_j],
                                            config.extrap_method)
                                n_merged_this_pass += 1

        log.debug("Merge pass %d: merged %d segments",
                  merge_pass + 1, n_merged_this_pass)
        if n_merged_this_pass == 0:
            break  # No more merges possible

    active = [s for s in segments if s.active and s.length >= 1]
    log.info("After merging: %d active trajectories", len(active))
    return active


# ═══════════════════════════════════════════════════════════════
#  Convert to trajectory matrix
# ═══════════════════════════════════════════════════════════════

def segments_to_matrix(
    segments: List[TrajectorySegment],
) -> np.ndarray:
    """Convert segment list to (N_traj, n_frames, D) array.

    NaN entries indicate frames where the particle was not tracked.
    """
    if not segments:
        return np.empty((0, 0, 0))
    n_frames = segments[0].coords.shape[0]
    ndim = segments[0].coords.shape[1]
    active = [s for s in segments if s.active]
    mat = np.full((len(active), n_frames, ndim), np.nan)
    for i, seg in enumerate(active):
        mat[i] = seg.coords
    return mat


# ═══════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════

def _extrapolate_position(
    seg: TrajectorySegment,
    target_frame: int,
    method: str,
) -> Optional[np.ndarray]:
    """Extrapolate a segment's trajectory to a target frame.

    Uses scipy PchipInterpolator for 'pchip' or nearest-neighbor
    for 'nearest' (suitable for Brownian motion).

    Returns None if extrapolation is not possible.
    """
    valid_mask = ~np.isnan(seg.coords[:, 0])
    valid_frames = np.where(valid_mask)[0]
    if len(valid_frames) < 2:
        # Can't extrapolate with fewer than 2 points
        if len(valid_frames) == 1:
            return seg.coords[valid_frames[0]].copy()
        return None

    ndim = seg.coords.shape[1]
    result = np.zeros(ndim)
    t = valid_frames.astype(np.float64)

    for d in range(ndim):
        values = seg.coords[valid_frames, d]
        if method == "pchip" and len(valid_frames) >= 2:
            try:
                interp = PchipInterpolator(t, values, extrapolate=True)
                result[d] = interp(float(target_frame))
            except Exception:
                result[d] = values[-1] if target_frame > t[-1] else values[0]
        else:
            # Nearest: use the closest endpoint
            if target_frame >= t[-1]:
                result[d] = values[-1]
            else:
                result[d] = values[0]

    return result


def _find_best_candidate(
    segments: List[TrajectorySegment],
    exclude_idx: int,
    target_frame: int,
    pred_pos: np.ndarray,
    dist_threshold: float,
    direction: str,
) -> int:
    """Find the best segment to merge at the target frame.

    For 'forward' direction: looks for segments starting at target_frame.
    For 'backward' direction: looks for segments ending at target_frame+1.

    Returns index into segments list, or -1 if none found.
    """
    best_idx = -1
    best_dist = np.inf

    for j, seg_j in enumerate(segments):
        if j == exclude_idx or not seg_j.active or seg_j.length == 0:
            continue

        if direction == "forward":
            # Candidate must start at or near target_frame
            if seg_j.start_frame != target_frame:
                continue
            cand_pos = seg_j.coords[target_frame]
        else:
            # Candidate must end at or near target_frame + 1
            if seg_j.end_frame != target_frame + 1:
                continue
            cand_pos = seg_j.coords[target_frame]

        if np.any(np.isnan(cand_pos)):
            continue

        dist = np.linalg.norm(pred_pos - cand_pos)
        if dist < dist_threshold and dist < best_dist:
            best_dist = dist
            best_idx = j

    return best_idx


def _merge_into(
    target: TrajectorySegment,
    source: TrajectorySegment,
    fill_method: str,
):
    """Merge source segment into target, then fill NaN gaps.

    Copies all non-NaN entries from source into target,
    deactivates source, then interpolates any internal gaps.
    """
    # Copy non-NaN entries from source
    valid_src = ~np.isnan(source.coords[:, 0])
    target.coords[valid_src] = source.coords[valid_src]

    # Deactivate source
    source.active = False
    source.coords[:] = np.nan
    source.length = 0

    # Fill internal gaps in the merged trajectory
    _fill_gaps(target, fill_method)

    # Recompute bounds
    target.recompute_bounds()


def _fill_gaps(seg: TrajectorySegment, method: str):
    """Interpolate internal NaN gaps in a trajectory segment.

    Replaces MATLAB's fillmissing(). Only fills gaps *between*
    the first and last valid frame (no extrapolation beyond endpoints).
    """
    valid_mask = ~np.isnan(seg.coords[:, 0])
    valid_frames = np.where(valid_mask)[0]
    if len(valid_frames) < 2:
        return

    first, last = valid_frames[0], valid_frames[-1]
    interior = np.arange(first, last + 1)
    ndim = seg.coords.shape[1]

    t = valid_frames.astype(np.float64)
    for d in range(ndim):
        values = seg.coords[valid_frames, d]
        if method == "pchip" and len(valid_frames) >= 2:
            try:
                interp = PchipInterpolator(t, values)
                seg.coords[interior, d] = interp(interior.astype(np.float64))
            except Exception:
                # Fallback: linear
                seg.coords[interior, d] = np.interp(
                    interior, valid_frames, values
                )
        else:
            seg.coords[interior, d] = np.interp(
                interior, valid_frames, values
            )


def _deduplicate_segments(
    segments: List[TrajectorySegment],
) -> List[TrajectorySegment]:
    """Remove duplicate segments that share identical trajectories.

    Two segments are duplicates if they have identical non-NaN entries
    at the same frames. Keep the longer one.
    """
    if len(segments) <= 1:
        return segments

    # Sort by length descending so we keep longer ones
    segments.sort(key=lambda s: s.length, reverse=True)

    # Use a set of (frame, rounded_coords) tuples as fingerprints
    seen_fingerprints = set()
    unique = []

    for seg in segments:
        valid = ~np.isnan(seg.coords[:, 0])
        frames = np.where(valid)[0]
        if len(frames) == 0:
            continue
        # Fingerprint: (start_frame, end_frame, first_coord, last_coord)
        fp = (
            int(frames[0]), int(frames[-1]),
            tuple(np.round(seg.coords[frames[0]], 4)),
            tuple(np.round(seg.coords[frames[-1]], 4)),
        )
        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            unique.append(seg)

    return unique

# ======================================================================
# VENDORED FROM: nd2studios/backend/serialtrack/tracking.py
# ======================================================================

"""
SerialTrack Python — Main tracking engine (v3)
================================================
    serialtrack/tracking.py

Fixes from v2
-------------
- Fixed ADMM convergence order: MATLAB checks convergence BEFORE warping;
  Python was warping BEFORE checking.  Now matches MATLAB: on convergence,
  the last global-step displacement is NOT applied.
- Fixed ``_local_step`` retry loop: the retry was a no-op because
  ``min(n_neighbors, n_max_local)`` never changes when n_max_local starts
  at n_neighbors_max.  Now correctly increases the neighbor count on retry,
  matching the MATLAB ``n_neighborsMax = round(n_neighborsMax + 5)`` logic.
- Fixed f_o_s update: now passes the current f_o_s as context so the floor
  can adapt, instead of hardcoding 60.0.
- Double-frame mode (TrackingMode.DOUBLE_FRAME) implemented.
- Trajectory building integrated into TrackingSession.
"""

from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Tuple, Callable
import time
import numpy as np
import logging


log = logging.getLogger("serialtrack.tracking")


# ═══════════════════════════════════════════════════════════════
#  Result containers
# ═══════════════════════════════════════════════════════════════

@dataclass
class FrameResult:
    """Tracking output for a single frame pair."""
    frame_idx: int
    coords_b: np.ndarray
    disp_b2a: np.ndarray
    track_a2b: np.ndarray
    track_b2a: np.ndarray
    match_ratio: float
    n_iterations: int
    wall_time: float
    # Post-processing (deformed config)
    disp_field: Optional[DisplacementField] = None
    strain_field: Optional[StrainField] = None
    # Post-processing (reference config)
    disp_field_ref: Optional[DisplacementField] = None
    strain_field_ref: Optional[StrainField] = None


@dataclass
class TrackingSession:
    """Full tracking session results across all frames."""
    detection_config: DetectionConfig
    tracking_config: TrackingConfig
    coords_ref: np.ndarray
    frame_results: List[FrameResult] = dc_field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frame_results) + 1

    @property
    def tracking_ratios(self) -> np.ndarray:
        return np.array([r.match_ratio for r in self.frame_results])

    def get_trajectories(self) -> np.ndarray:
        """Build (N_ref, n_frames, D) trajectory matrix via naive chaining.

        For full trajectory stitching (with gap filling), use
        ``build_stitched_trajectories()`` instead.
        """
        ndim = self.coords_ref.shape[1]
        Na = len(self.coords_ref)
        nf = self.n_frames
        traj = np.full((Na, nf, ndim), np.nan)
        traj[:, 0, :] = self.coords_ref

        for fi, res in enumerate(self.frame_results, 1):
            tracked = res.track_a2b >= 0
            idx_a = np.where(tracked)[0]
            idx_b = res.track_a2b[idx_a]
            traj[idx_a, fi, :] = res.coords_b[idx_b]

        return traj

    def build_stitched_trajectories(self) -> np.ndarray:
        """Build trajectories with segment merging and gap filling.

        Uses the trajectory stitching algorithm from the MATLAB code.
        Returns (N_traj, n_frames, D) array with NaN for missing frames.
        """
        cfg = self.tracking_config

        if cfg.mode == TrackingMode.CUMULATIVE:
            coords_list = [res.coords_b for res in self.frame_results]
            track_list = [res.track_a2b for res in self.frame_results]
            segments = build_segments_cumulative(
                self.coords_ref, coords_list, track_list
            )
        else:
            # Incremental or double-frame
            coords_list = [self.coords_ref] + [
                res.coords_b for res in self.frame_results
            ]
            track_list = [res.track_a2b for res in self.frame_results]
            segments = build_segments_incremental(coords_list, track_list)

        # Merge segments
        merged = merge_segments(segments, cfg.trajectory)

        return segments_to_matrix(merged)


# ═══════════════════════════════════════════════════════════════
#  Single-frame ADMM tracker
# ═══════════════════════════════════════════════════════════════

class _ADMMFrameTracker:
    """ADMM iteration loop for a single frame pair.

    Follows the MATLAB ``f_track_serial_match3D.m`` structure exactly:
      1. Local step: topology or nearest-neighbor matching
      2. Global step: regularised displacement interpolation
      3. Convergence check (BEFORE warping — matching MATLAB)
      4. Warp B coordinates and update f_o_s
      5. Cull missing particles when n_neighbors < 4
    """

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.regularizer = DisplacementRegularizer(solver=cfg.solver)

    def run(
        self,
        coords_a: np.ndarray,
        coords_b: np.ndarray,
        init_disp: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        """Execute ADMM tracking loop.

        Returns: disp_b2a, track_a2b, track_b2a, match_ratio, n_iters
        """
        cfg = self.cfg
        ndim = coords_a.shape[1]
        Na, Nb = len(coords_a), len(coords_b)

        coords_b_curr = coords_b.copy()
        disp_b2a = np.zeros((Nb, ndim))
        not_missing_a = np.arange(Na)
        not_missing_b = np.arange(Nb)
        match_ratio_eq1_count = 0
        match_ratio = 0.0

        grid_step = np.full(ndim, min(round(0.5 * cfg.f_o_s), 20), dtype=np.float64)

        track_a2b = np.full(Na, -1, dtype=np.int64)
        track_b2a = np.full(Nb, -1, dtype=np.int64)

        # Working copy of f_o_s that updates per iteration
        working_f_o_s = float(cfg.f_o_s)

        if init_disp is not None and init_disp.shape == disp_b2a.shape:
            disp_b2a += init_disp
            coords_b_curr = coords_b + init_disp

        for iter_num in range(cfg.max_iter):
            n_neighbors = round(
                cfg.n_neighbors_min
                + np.exp(-0.5 * iter_num) * (cfg.n_neighbors_max - cfg.n_neighbors_min)
            )
            log.info("  Iter %d | n_neighbors=%d | f_o_s=%.1f",
                     iter_num + 1, n_neighbors, working_f_o_s)

            # ── LOCAL STEP ──
            matches, local_track, local_disp = self._local_step(
                coords_a, coords_b_curr,
                not_missing_a, not_missing_b,
                n_neighbors, working_f_o_s, cfg.outlier_threshold,
            )

            if len(matches) == 0:
                log.warning("  No matches at iter %d", iter_num + 1)
                break

            track_a2b = local_track
            match_ratio = np.sum(track_a2b >= 0) / max(len(not_missing_a), 1)
            log.info("  Tracking ratio: %d/%d = %.4f",
                     np.sum(track_a2b >= 0), len(not_missing_a), match_ratio)

            # ── GLOBAL STEP ──
            tracked_mask = track_a2b >= 0
            idx_a = np.where(tracked_mask)[0]
            idx_b = track_a2b[idx_a]
            matched_disp_b2a = -(coords_b_curr[idx_b] - coords_a[idx_a])

            temp_disp = self.regularizer.solve(
                disp=matched_disp_b2a,
                coords=coords_b_curr[idx_b],
                query_coords=coords_b_curr,
                grid_step=grid_step,
                smoothness=cfg.smoothness,
                f_o_s=working_f_o_s,
                n_neighbors=n_neighbors,
                is_first_iter=(iter_num == 0),
            )

            # ── Convergence check (BEFORE warping — matches MATLAB) ──
            update_norm = np.sqrt(np.sum(temp_disp**2) / max(len(temp_disp), 1))
            log.info("  Disp update norm: %.6f", update_norm)

            if match_ratio > 0.999:
                match_ratio_eq1_count += 1

            threshold = np.sqrt(ndim) * cfg.iter_stop_threshold
            if update_norm < threshold or match_ratio_eq1_count > 5:
                log.info("  Converged at iter %d", iter_num + 1)
                break

            # ── Warp B (only if NOT converged — matches MATLAB) ──
            disp_b2a += temp_disp
            coords_b_curr = coords_b + disp_b2a

            # ── Update f_o_s for next iteration ──
            if len(temp_disp) > 0:
                working_f_o_s = update_f_o_s(temp_disp, working_f_o_s)

            # ── Cull missing particles (late iterations) ──
            if n_neighbors < 4:
                not_missing_a, not_missing_b = find_not_missing(
                    coords_a, coords_b_curr, cfg.dist_missing
                )

        # Build track_b2a
        track_b2a = np.full(Nb, -1, dtype=np.int64)
        for ia in range(Na):
            ib = track_a2b[ia]
            if ib >= 0:
                track_b2a[ib] = ia

        return disp_b2a, track_a2b, track_b2a, match_ratio, iter_num + 1

    def _local_step(self, coords_a, coords_b_curr, not_missing_a, not_missing_b,
                    n_neighbors, f_o_s, outlier_threshold):
        """Local matching step with retry logic.

        On failure, increases the neighbor count (matching MATLAB:
        ``n_neighborsMax = round(n_neighborsMax + 5)``), which allows
        the topology matcher to use more neighbors for a richer
        feature vector.

        FIX: The v2 code used ``min(n_neighbors, n_max_local)`` which
        was a no-op.  Now correctly increases n_neighbors on retry.
        """
        ca = coords_a[not_missing_a]
        cb = coords_b_curr[not_missing_b]
        retry_n_neighbors = n_neighbors

        matches_raw = np.empty((0, 2), dtype=np.int64)

        max_attempts = 5
        for attempt in range(max_attempts):
            if retry_n_neighbors > 2:
                matcher = TopologyMatcher(n_neighbors=retry_n_neighbors, f_o_s=f_o_s)
            else:
                matcher = NearestNeighborMatcher(f_o_s=f_o_s)

            matches_raw = matcher.match(ca, cb)

            if len(matches_raw) == 0:
                # Increase neighbor count for richer topology features
                retry_n_neighbors = min(retry_n_neighbors + 5, len(ca) - 1)
                if retry_n_neighbors < 3:
                    break  # Can't do topology matching with so few particles
                log.debug("  Local step retry: n_neighbors → %d", retry_n_neighbors)
            else:
                break

        if len(matches_raw) == 0:
            Na = len(coords_a)
            ndim = coords_a.shape[1]
            return (
                np.empty((0, 2), dtype=np.int64),
                np.full(Na, -1, dtype=np.int64),
                np.empty((0, ndim)),
            )

        # Map local indices back to full arrays
        matches_full = np.column_stack([
            not_missing_a[matches_raw[:, 0]],
            not_missing_b[matches_raw[:, 1]],
        ])
        track_a2b, disp_a2b = compute_displacement(
            coords_a, coords_b_curr, matches_full, outlier_threshold
        )
        return matches_full, track_a2b, disp_a2b


# ═══════════════════════════════════════════════════════════════
#  Main public tracker
# ═══════════════════════════════════════════════════════════════

class SerialTracker:
    """Top-level SerialTrack particle tracking engine.

    Supports INCREMENTAL, CUMULATIVE, and DOUBLE_FRAME modes.
    """

    def __init__(self, detection_config: DetectionConfig, tracking_config: TrackingConfig):
        self.det_cfg = detection_config
        self.trk_cfg = tracking_config
        self.detector = ParticleDetector(detection_config)
        self.predictor = InitialGuessPredictor()

    def track_images(
        self,
        images: List[np.ndarray],
        progress_cb: Optional[Callable] = None,
    ) -> TrackingSession:
        """Track particles across a sequence of images."""
        if len(images) < 2:
            raise ValueError("Need at least 2 images")

        cfg = self.trk_cfg
        cfg.init_roi_from_image(images[0])

        coords_ref = self.detector.detect(images[0], cfg.roi_slices())
        coords_ref = ParticleDetector.clip_to_bounds(coords_ref, images[0].shape)
        log.info("Detected %d particles in reference image", len(coords_ref))

        all_coords = [coords_ref]
        for i in range(1, len(images)):
            c = self.detector.detect(images[i], cfg.roi_slices())
            c = ParticleDetector.clip_to_bounds(c, images[i].shape)
            all_coords.append(c)
            log.info("Detected %d particles in frame %d", len(c), i + 1)

        return self._run_tracking(all_coords, progress_cb)

    def track_coordinates(
        self,
        coords_list: List[np.ndarray],
        progress_cb: Optional[Callable] = None,
    ) -> TrackingSession:
        """Track from pre-detected particle coordinates."""
        if len(coords_list) < 2:
            raise ValueError("Need at least 2 coordinate sets")
        return self._run_tracking(coords_list, progress_cb)

    def _run_tracking(
        self,
        all_coords: List[np.ndarray],
        progress_cb: Optional[Callable],
    ) -> TrackingSession:
        cfg = self.trk_cfg
        coords_ref = all_coords[0]
        n_frames = len(all_coords)

        session = TrackingSession(
            detection_config=self.det_cfg,
            tracking_config=cfg,
            coords_ref=coords_ref,
        )

        prev_coords: List[np.ndarray] = []
        prev_disp: List[np.ndarray] = []

        for fi in range(1, n_frames):
            log.info("====== Frame %d / %d ======", fi + 1, n_frames)
            t0 = time.perf_counter()

            coords_b = all_coords[fi]

            # Decide reference for this frame based on mode
            if cfg.mode == TrackingMode.CUMULATIVE:
                coords_a = coords_ref
            elif cfg.mode == TrackingMode.DOUBLE_FRAME:
                coords_a = all_coords[fi - 1]
            else:  # INCREMENTAL
                if fi == 1:
                    coords_a = coords_ref
                else:
                    coords_a = prev_coords[-1] if prev_coords else coords_ref

            # Initial guess (not used for double-frame mode)
            init_disp = None
            if (cfg.mode != TrackingMode.DOUBLE_FRAME
                    and cfg.use_prev_results
                    and fi >= 2 and len(prev_disp) > 0):
                init_disp = self.predictor.predict(
                    frame_idx=fi + 1,
                    current_coords=coords_b,
                    prev_coords_list=prev_coords,
                    prev_disp_list=prev_disp,
                )

            # Run ADMM tracker
            admm = _ADMMFrameTracker(cfg)
            disp_b2a, track_a2b, track_b2a, ratio, n_iters = admm.run(
                coords_a, coords_b, init_disp
            )

            wall = time.perf_counter() - t0
            log.info("  Frame %d done: ratio=%.3f, iters=%d, time=%.2fs",
                     fi + 1, ratio, n_iters, wall)

            result = FrameResult(
                frame_idx=fi + 1,
                coords_b=coords_b,
                disp_b2a=disp_b2a,
                track_a2b=track_a2b,
                track_b2a=track_b2a,
                match_ratio=ratio,
                n_iterations=n_iters,
                wall_time=wall,
            )

            # Strain in both configs
            if cfg.strain_n_neighbors > 0:
                self._compute_strain(result, coords_a, cfg)

            session.frame_results.append(result)

            prev_coords.append(coords_b)
            prev_disp.append(disp_b2a)

            if progress_cb is not None:
                progress_cb(fi + 1, n_frames, result)

        return session

    def _compute_strain(self, result: FrameResult, coords_a: np.ndarray,
                        cfg: TrackingConfig):
        """Compute displacement & strain in BOTH deformed and reference configs."""
        disp_a2b = -result.disp_b2a
        coords_b = result.coords_b

        tracked = result.track_b2a >= 0
        if np.sum(tracked) < cfg.strain_n_neighbors:
            return

        ndim = coords_b.shape[1]
        grid_step = np.full(ndim, min(round(0.5 * cfg.strain_f_o_s), 20),
                            dtype=np.float64)

        # 1. Strain in deformed configuration (on B particles)
        try:
            dfield, sfield = compute_gridded_strain(
                coords=coords_b[tracked],
                disp=disp_a2b[tracked],
                grid_step=grid_step,
                smoothness=1e-3,
                pixel_steps=cfg.steps,
            )
            result.disp_field = dfield
            result.strain_field = sfield
        except Exception as e:
            log.warning("Strain (deformed config) failed: %s", e)

        # 2. Strain in reference configuration (on A particles = B - disp)
        try:
            coords_ref_config = coords_b[tracked] - disp_a2b[tracked]
            dfield_ref, sfield_ref = compute_gridded_strain(
                coords=coords_ref_config,
                disp=disp_a2b[tracked],
                grid_step=grid_step,
                smoothness=1e-3,
                pixel_steps=cfg.steps,
            )
            result.disp_field_ref = dfield_ref
            result.strain_field_ref = sfield_ref
        except Exception as e:
            log.warning("Strain (reference config) failed: %s", e)


# ═══════════════════════════════════════════════════════════════
#  Convenience function
# ═══════════════════════════════════════════════════════════════

def track(
    images: List[np.ndarray],
    detection_config: Optional[DetectionConfig] = None,
    tracking_config: Optional[TrackingConfig] = None,
    progress_cb: Optional[Callable] = None,
) -> TrackingSession:
    """One-call convenience function for tracking."""
    det_cfg = detection_config or DetectionConfig()
    trk_cfg = tracking_config or TrackingConfig()
    tracker = SerialTracker(det_cfg, trk_cfg)
    return tracker.track_images(images, progress_cb)


# ======================================================================
# VENDORED FROM: nd2studios/backend/celltracker/tracking.py
# ======================================================================

"""Cell tracking: Hungarian LAP (with birth/death), topology, and mask overlap.

Vendored from CellTracker ``backend/tracking.py`` (topology features inspired by
SerialTrack's matching.py). Only the headless ``track_timeseries`` (topology
Hungarian) and ``track_fingerprint`` (position + area, gap filling) linkers and
their helpers are copied — CellTracker's ``track_serialtrack`` is omitted (it
carries a hardcoded path and is already covered by
:mod:`nd2studios.backend.serialtrack`).

Operates on pandas DataFrames with columns ``frame``, ``label``, ``centroid_y``,
``centroid_x`` (and ``area`` for the fingerprint linker); returns a copy with a
``track_id`` column added.

V1.58 — the per-frame assignment now uses a Jaqaman-style LAP with birth/death
"no-match" nodes (:func:`solve_lap`) rather than a bare ``linear_sum_assignment``.
A bare complete matching is forced to link the smaller side in full, which (a)
manufactures spurious long links on count imbalance (high ``max_dist``) and (b)
shuffles a whole neighborhood onto each other's targets when a true partner is
just out of range ("tracking currents"; low ``max_dist``). Letting a detection
stay unmatched at a fixed cost removes both. V1.58 also adds :func:`track_overlap`
— a mask-IoU linker that consumes the StarDist label masks directly, the robust
default for dense, slow-moving nuclei where centroid distance is ambiguous.
"""

import logging
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

log = logging.getLogger(__name__)

# A progress callback reports a fraction in [0, 1] and a short status message.
# Kept Qt-free so the backend stays importable without PySide6.
ProgressCB = Callable[[float, str], None]

# A per-frame cost matrix larger than this (N × M entries) makes the O(n³)
# Hungarian assignment the dominant cost; we log a one-time warning so a
# dense-field slowdown is explained rather than mysterious.
_BIG_COST_MATRIX = 4_000_000

# Finite stand-in for a forbidden pair inside the augmented cost matrix. The
# optimum never selects one (a birth+death pair is always cheaper), so its only
# job is to dominate any achievable real assignment total without overflowing.
_BIG = 1.0e12


# ═══════════════════════════════════════════════════════════════
#  Assignment: Hungarian LAP with birth/death (no-match) nodes
# ═══════════════════════════════════════════════════════════════

def _solve_lap_block(cost: np.ndarray, no_match_cost: float) -> List[Tuple[int, int]]:
    """Optimal assignment of one dense cost block, births/deaths allowed.

    Builds the Jaqaman (2008) augmented matrix and returns the accepted
    ``(row, col)`` links. ``cost`` may contain ``np.inf`` for forbidden pairs;
    those are never returned. Intended for a single connected component, so the
    block stays small even on a dense field (see :func:`solve_lap`).
    """
    N, M = cost.shape
    d = float(no_match_cost)
    size = N + M
    aug = np.full((size, size), _BIG, dtype=np.float64)
    # Q1 — real linking costs (forbidden → _BIG so they're never chosen).
    aug[:N, :M] = np.where(np.isfinite(cost), cost, _BIG)
    # Q2 — each prev may "die" on its own dummy column (diagonal = d).
    death = np.full((N, N), _BIG, dtype=np.float64)
    np.fill_diagonal(death, d)
    aug[:N, M:] = death
    # Q3 — each curr may be "born" on its own dummy row (diagonal = d).
    birth = np.full((M, M), _BIG, dtype=np.float64)
    np.fill_diagonal(birth, d)
    aug[N:, :M] = birth
    # Q4 — dummy↔dummy pairings are free; they only absorb the leftover
    # birth-rows / death-cols left over once real links are chosen.
    aug[N:, M:] = 0.0

    row_ind, col_ind = linear_sum_assignment(aug)
    out: List[Tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if r < N and c < M and np.isfinite(cost[r, c]):
            out.append((int(r), int(c)))
    return out


def solve_lap(
    cost: np.ndarray,
    no_match_cost: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve one frame-to-frame assignment allowing births and deaths.

    The Jaqaman et al. (2008) linear-assignment formulation used by u-track /
    TrackMate: the ``N×M`` linking-cost block is augmented with dummy "no-match"
    nodes so a detection may stay **unmatched at a fixed cost** (``no_match_cost``)
    instead of being force-linked. Without them a bare ``linear_sum_assignment``
    must return a complete matching of the smaller side, which forces spurious
    long links on count imbalance and shuffles neighborhoods onto each other's
    targets ("tracking currents") when a true partner is just out of range. An
    honest birth/death is cheaper than either, so the dummies remove both.

    For speed on dense fields the gated cost matrix (most pairs ``inf``) is split
    into connected components of finite-cost edges and each small component is
    solved independently — exact, because a node's only non-linking option is its
    own birth/death dummy, so components never couple.

    Parameters
    ----------
    cost : (N, M) float array
        Linking cost per prev→curr pair; forbidden pairs marked ``np.inf`` are
        never accepted.
    no_match_cost : float
        Cost ``d`` of leaving a detection unmatched (a birth or a death). A link
        is preferred only when cheaper than ``d``, so ``d`` is also a soft gate
        on top of the hard ``inf`` gate.

    Returns
    -------
    matches : list of (prev_idx, curr_idx)
    unmatched_prev, unmatched_curr : list of indices
    """
    N, M = cost.shape
    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    unmatched_prev = set(range(N))
    unmatched_curr = set(range(M))
    matches: List[Tuple[int, int]] = []

    finite = np.isfinite(cost)
    if not finite.any():
        return [], list(range(N)), list(range(M))

    # Connected components over the bipartite graph of finite (allowed) edges.
    # Prev node i → graph node i; curr node j → graph node N + j. Nodes with no
    # allowed edge fall out as singletons and stay unmatched (birth / death).
    pr, cu = np.nonzero(finite)
    n_nodes = N + M
    data = np.ones(pr.size, dtype=np.int8)
    adj = coo_matrix((data, (pr, N + cu)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(adj, directed=False)

    for comp in range(n_comp):
        node_ids = np.nonzero(labels == comp)[0]
        rows = node_ids[node_ids < N]
        cols = node_ids[node_ids >= N] - N
        if rows.size == 0 or cols.size == 0:
            continue  # a lone prev (death) or lone curr (birth)
        sub = cost[np.ix_(rows, cols)]
        for r, c in _solve_lap_block(sub, no_match_cost):
            pi, ci = int(rows[r]), int(cols[c])
            matches.append((pi, ci))
            unmatched_prev.discard(pi)
            unmatched_curr.discard(ci)

    return matches, sorted(unmatched_prev), sorted(unmatched_curr)


# ═══════════════════════════════════════════════════════════════
#  Topology features
# ═══════════════════════════════════════════════════════════════

def compute_topology_features(
    centroids: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    For each cell, compute rotation-invariant topology features:
    sorted distances and angular gaps to K nearest neighbors.

    Parameters
    ----------
    centroids : (N, 2) array of [y, x] positions
    n_neighbors : int, number of neighbors to use

    Returns
    -------
    features : (N, 2*K) array — first K columns are sorted distances,
               next K columns are sorted angular gaps
    """
    N = len(centroids)
    if N < 2:
        return np.zeros((N, 2 * n_neighbors))

    K = min(n_neighbors, N - 1)
    tree = cKDTree(centroids)
    dists, indices = tree.query(centroids, k=K + 1)  # +1 for self

    # Remove self (first column)
    dists = dists[:, 1:]       # (N, K)
    indices = indices[:, 1:]   # (N, K)

    features = np.zeros((N, 2 * n_neighbors))

    # Vectorized equivalent of the original per-cell loop. cKDTree returns a
    # uniform K neighbors for every cell (k == K here), so the whole thing is
    # array ops. The ``k > 1`` guard preserves the original behavior of leaving
    # the angular-gap block as zeros when there is only a single neighbor.
    k = K
    # Sorted neighbor distances (cKDTree already returns them ascending).
    features[:, :k] = dists[:, :k]

    if k > 1:
        # Neighbor offset vectors, (N, k, 2) as [y, x].
        neighbors = centroids[indices[:, :k]] - centroids[:, None, :]
        angles = np.arctan2(neighbors[:, :, 0], neighbors[:, :, 1])  # (N, k)
        angles_sorted = np.sort(angles, axis=1)
        gaps = np.diff(angles_sorted, axis=1)                        # (N, k-1)
        wrap = (2 * np.pi + angles_sorted[:, 0] - angles_sorted[:, -1])[:, None]
        gaps = np.concatenate([gaps, wrap], axis=1)                  # (N, k)
        gaps_sorted = np.sort(gaps, axis=1)
        features[:, n_neighbors:n_neighbors + k] = gaps_sorted

    return features


# ═══════════════════════════════════════════════════════════════
#  Frame-to-frame linking
# ═══════════════════════════════════════════════════════════════

def link_frames(
    centroids_prev: np.ndarray,
    centroids_curr: np.ndarray,
    max_dist: float = 30.0,
    topo_prev: Optional[np.ndarray] = None,
    topo_curr: Optional[np.ndarray] = None,
    topo_weight: float = 0.3,
    no_match_cost: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Link detections between two consecutive frames with a birth/death LAP.

    Parameters
    ----------
    centroids_prev, centroids_curr : (N, 2) and (M, 2) arrays
    max_dist : float, maximum linking distance in pixels (hard gate)
    topo_prev, topo_curr : topology feature arrays (optional)
    topo_weight : float 0-1, weight of topology cost vs distance cost
    no_match_cost : float, optional
        Cost of leaving a detection unmatched (birth/death). Defaults to
        ``max_dist``. See :func:`solve_lap` — this is what stops a cell whose
        true partner is out of range from being force-linked onto a neighbor
        (the "tracking currents" failure mode).

    Returns
    -------
    matches : list of (prev_idx, curr_idx)
    unmatched_prev : list of indices
    unmatched_curr : list of indices
    """
    N = len(centroids_prev)
    M = len(centroids_curr)

    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    # Distance cost
    dist_cost = cdist(centroids_prev, centroids_curr)

    # Topology cost (if available)
    if topo_prev is not None and topo_curr is not None and topo_weight > 0:
        topo_cost = cdist(topo_prev, topo_curr, metric="euclidean")
        # Normalize topology cost to same scale as distance
        topo_scale = np.median(dist_cost[dist_cost < max_dist]) if np.any(dist_cost < max_dist) else 1.0
        topo_norm = np.median(topo_cost) if topo_cost.size > 0 else 1.0
        if topo_norm > 0:
            topo_cost = topo_cost * (topo_scale / topo_norm)
        cost = (1 - topo_weight) * dist_cost + topo_weight * topo_cost
    else:
        cost = dist_cost.copy()

    # Hard gate: forbid links beyond max_dist (marked inf → never linked). The
    # gate is on raw distance, not the topology-blended cost, so topology only
    # ranks the *reachable* candidates.
    cost[dist_cost > max_dist] = np.inf

    # LAP with birth/death: an unmatched cell costs `no_match_cost` (default
    # max_dist) instead of being force-linked onto a neighbor.
    d = float(max_dist if no_match_cost is None else no_match_cost)
    return solve_lap(cost, d)


# ═══════════════════════════════════════════════════════════════
#  Full timeseries tracking
# ═══════════════════════════════════════════════════════════════

def track_timeseries(
    df: pd.DataFrame,
    max_dist: float = 30.0,
    n_neighbors: int = 5,
    use_topology: bool = True,
    topo_weight: float = 0.3,
    no_match_cost: Optional[float] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> pd.DataFrame:
    """
    Track cells across all frames using nearest-neighbor linking.

    Parameters
    ----------
    df : DataFrame with columns: frame, label, centroid_y, centroid_x
    max_dist : float, max linking distance
    n_neighbors : int, neighbors for topology features
    use_topology : bool, whether to use topology-augmented cost
    topo_weight : float, weight of topology in cost matrix
    progress_cb : callable(fraction_0_1, message), reports linking progress

    Returns
    -------
    df : same DataFrame with added 'track_id' column
    """
    # Group by frame ONCE (dict of per-frame sub-frames) instead of re-scanning
    # the whole DataFrame with a boolean mask on every iteration — that repeated
    # mask was O(T² · cells) and a major slice of the wall-clock on dense fields.
    by_frame = {int(f): sub for f, sub in df.groupby("frame", sort=True)}
    frames = sorted(by_frame)
    T = len(frames)
    n_det = len(df)
    log.info(
        "CT topology tracking: %d detections across %d frames (~%.0f/frame), "
        "topology=%s, max_dist=%.1f",
        n_det, T, (n_det / T if T else 0), use_topology, max_dist,
    )

    next_id = 1
    track_ids: dict = {}
    t_link = 0.0
    t_topo = 0.0
    max_cost_cells = 0

    # First frame: every cell gets a new track.
    first = by_frame[frames[0]]
    for lbl in first["label"].to_numpy():
        track_ids[(frames[0], lbl)] = next_id
        next_id += 1

    for i in range(1, T):
        pf, cf = frames[i - 1], frames[i]
        prev = by_frame[pf]
        curr = by_frame[cf]

        c_prev = prev[["centroid_y", "centroid_x"]].values
        c_curr = curr[["centroid_y", "centroid_x"]].values
        l_prev = prev["label"].values
        l_curr = curr["label"].values

        max_cost_cells = max(max_cost_cells, len(c_prev) * len(c_curr))

        # Topology features
        topo_prev = topo_curr = None
        if use_topology and len(c_prev) > n_neighbors and len(c_curr) > n_neighbors:
            _t0 = time.perf_counter()
            topo_prev = compute_topology_features(c_prev, n_neighbors)
            topo_curr = compute_topology_features(c_curr, n_neighbors)
            t_topo += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        matches, _, unmatched_curr = link_frames(
            c_prev, c_curr, max_dist,
            topo_prev, topo_curr, topo_weight,
            no_match_cost=no_match_cost,
        )
        t_link += time.perf_counter() - _t0

        for prev_idx, curr_idx in matches:
            prev_key = (pf, l_prev[prev_idx])
            curr_key = (cf, l_curr[curr_idx])
            track_ids[curr_key] = track_ids[prev_key]

        for curr_idx in unmatched_curr:
            track_ids[(cf, l_curr[curr_idx])] = next_id
            next_id += 1

        if progress_cb:
            progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")

    if max_cost_cells > _BIG_COST_MATRIX:
        log.warning(
            "CT topology: largest per-frame cost matrix is %d entries — the "
            "Hungarian assignment is O(n³), so this dense field is inherently "
            "slow. Consider a smaller max_dist or fewer detections.",
            max_cost_cells,
        )

    # Assign track_id column. A vectorized dict lookup over zipped numpy arrays
    # replaces the old ``df.apply(..., axis=1)`` (a Python call + Series build
    # per row, which alone cost tens of seconds at 100k+ detections).
    _t0 = time.perf_counter()
    df = df.copy()
    frame_arr = df["frame"].to_numpy()
    label_arr = df["label"].to_numpy()
    df["track_id"] = [
        track_ids.get((int(f), int(lbl)), -1)
        for f, lbl in zip(frame_arr, label_arr)
    ]
    t_assign = time.perf_counter() - _t0

    log.info(
        "CT topology done: %d tracks; link %.2fs, topology %.2fs, assign %.2fs",
        next_id - 1, t_link, t_topo, t_assign,
    )
    if progress_cb:
        progress_cb(1.0, "Tracking done")

    return df


# ═══════════════════════════════════════════════════════════════
#  Spatial fingerprint tracker with gap filling
# ═══════════════════════════════════════════════════════════════

def fingerprint_cost_matrix(
    prev_centroids: np.ndarray,
    curr_centroids: np.ndarray,
    prev_areas: np.ndarray,
    curr_areas: np.ndarray,
    max_dist: float = 30.0,
    area_weight: float = 0.3,
) -> np.ndarray:
    """
    Build a cost matrix combining spatial distance and area similarity.

    Cost = (1 - area_weight) * spatial_distance + area_weight * area_mismatch

    area_mismatch is scaled so that a 2x area difference roughly equals
    max_dist in cost.
    """
    N, M = len(prev_centroids), len(curr_centroids)
    if N == 0 or M == 0:
        return np.zeros((N, M))

    dist = cdist(prev_centroids, curr_centroids)

    # Area mismatch: |log(a1/a2)| scaled to distance units
    pa = prev_areas[:, None].astype(np.float64)
    ca = curr_areas[None, :].astype(np.float64)
    pa = np.clip(pa, 1, None)
    ca = np.clip(ca, 1, None)
    area_ratio = np.abs(np.log(pa / ca))  # 0 = same size, ln(2)=0.69 = 2x mismatch
    area_cost = area_ratio * (max_dist / 0.7)  # normalize so 2x area ~ max_dist

    cost = (1 - area_weight) * dist + area_weight * area_cost
    cost[dist > max_dist] = np.inf  # hard gate; solve_lap treats inf as forbidden

    return cost


def track_fingerprint(
    df: pd.DataFrame,
    max_dist: float = 30.0,
    area_weight: float = 0.3,
    max_gap: int = 3,
    no_match_cost: Optional[float] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> pd.DataFrame:
    """
    Track cells using spatial position + size fingerprinting with gap filling.

    Designed for cells that:
    - Don't move far between frames
    - May disappear for a few frames (missed detections)
    - Maintain similar size across frames

    Parameters
    ----------
    df : DataFrame with columns: frame, label, centroid_y, centroid_x, area
    max_dist : float, max linking distance in pixels
    area_weight : float 0-1, weight of area similarity vs distance
    max_gap : int, max frames a cell can disappear and still be re-linked
    progress_cb : callable(fraction_0_1, message), reports linking progress

    Returns
    -------
    df with 'track_id' column
    """
    # Group by frame once (see track_timeseries — avoids the O(T² · cells)
    # repeated boolean mask).
    by_frame = {int(f): sub for f, sub in df.groupby("frame", sort=True)}
    frames = sorted(by_frame)
    T = len(frames)
    n_det = len(df)
    log.info(
        "CT fingerprint tracking: %d detections across %d frames (~%.0f/frame), "
        "max_dist=%.1f, area_weight=%.2f, max_gap=%d",
        n_det, T, (n_det / T if T else 0), max_dist, area_weight, max_gap,
    )
    _t_start = time.perf_counter()
    next_id = 1
    track_ids = {}  # (frame, label) -> track_id

    # Active tracks: track_id -> {last_frame, last_y, last_x, last_area}
    active_tracks = {}

    # First frame
    first = by_frame[frames[0]]
    for lbl, cy, cx, ar in zip(
        first["label"].to_numpy(), first["centroid_y"].to_numpy(),
        first["centroid_x"].to_numpy(), first["area"].to_numpy(),
    ):
        tid = next_id
        next_id += 1
        track_ids[(frames[0], lbl)] = tid
        active_tracks[tid] = {"last_frame": frames[0], "y": cy, "x": cx, "area": ar}

    for i in range(1, T):
        cf = frames[i]
        curr = by_frame[cf]

        if curr.empty:
            if progress_cb:
                progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
            continue

        c_curr = curr[["centroid_y", "centroid_x"]].values
        a_curr = curr["area"].values
        l_curr = curr["label"].values

        # Gather all active tracks (including those with gaps)
        alive_tids = []
        alive_centroids = []
        alive_areas = []

        for tid, info in active_tracks.items():
            gap = cf - info["last_frame"]
            if gap <= max_gap + 1:  # +1 because consecutive frames have gap=1
                alive_tids.append(tid)
                alive_centroids.append([info["y"], info["x"]])
                alive_areas.append(info["area"])

        if not alive_tids:
            # No active tracks — all cells start new tracks
            for ci in range(len(l_curr)):
                tid = next_id
                next_id += 1
                track_ids[(cf, l_curr[ci])] = tid
                active_tracks[tid] = {
                    "last_frame": cf,
                    "y": c_curr[ci, 0],
                    "x": c_curr[ci, 1],
                    "area": a_curr[ci],
                }
            if progress_cb:
                progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
            continue

        prev_centroids = np.array(alive_centroids)
        prev_areas = np.array(alive_areas)

        # Build cost matrix with spatial + area fingerprint
        cost = fingerprint_cost_matrix(
            prev_centroids, c_curr, prev_areas, a_curr,
            max_dist=max_dist, area_weight=area_weight,
        )

        # LAP with birth/death (see solve_lap). Unmatched alive tracks are left
        # in `active_tracks` for gap re-linking; unmatched detections below start
        # new tracks — no forced link onto a neighbor.
        d = float(max_dist if no_match_cost is None else no_match_cost)
        matches, _, _ = solve_lap(cost, d)

        matched_curr = set()
        for r, c in matches:
            tid = alive_tids[r]
            track_ids[(cf, l_curr[c])] = tid
            active_tracks[tid] = {
                "last_frame": cf,
                "y": c_curr[c, 0],
                "x": c_curr[c, 1],
                "area": a_curr[c],
            }
            matched_curr.add(c)

        # Unmatched detections start new tracks
        for ci in range(len(l_curr)):
            if ci not in matched_curr:
                tid = next_id
                next_id += 1
                track_ids[(cf, l_curr[ci])] = tid
                active_tracks[tid] = {
                    "last_frame": cf,
                    "y": c_curr[ci, 0],
                    "x": c_curr[ci, 1],
                    "area": a_curr[ci],
                }

        # Prune dead tracks (gap exceeded)
        dead = [tid for tid, info in active_tracks.items()
                if cf - info["last_frame"] > max_gap + 1]
        for tid in dead:
            del active_tracks[tid]

        if progress_cb:
            progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")

    # Assign track_id column (vectorized dict lookup — see track_timeseries).
    df = df.copy()
    frame_arr = df["frame"].to_numpy()
    label_arr = df["label"].to_numpy()
    df["track_id"] = [
        track_ids.get((int(f), int(lbl)), -1)
        for f, lbl in zip(frame_arr, label_arr)
    ]

    log.info("CT fingerprint done: %d tracks in %.2fs",
             next_id - 1, time.perf_counter() - _t_start)
    if progress_cb:
        progress_cb(1.0, "Tracking done")

    return df


# ═══════════════════════════════════════════════════════════════
#  Mask-overlap (IoU) tracking — the robust default for segmentation
# ═══════════════════════════════════════════════════════════════

def _frame_footprints(
    mask_frame: np.ndarray,
    keep: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Per-label flat pixel indices for one label image.

    Returns ``(labels, areas, flats)`` where object ``labels[i]`` covers
    ``areas[i]`` pixels at flat indices ``flats[i]`` (into the raveled frame).
    One stable argsort over the foreground pixels, so cost is ∝ foreground area
    rather than ``n_labels × frame_pixels``. ``keep`` restricts output to those
    label ids — used to ignore mask objects the measurement stage filtered out
    of the DataFrame (so they can't steal an overlap link).
    """
    flat = np.asarray(mask_frame).reshape(-1)
    fg = np.flatnonzero(flat)
    if fg.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), []
    labs = flat[fg]
    order = np.argsort(labs, kind="stable")
    fg = fg[order]
    labs = labs[order]
    uniq, starts, counts = np.unique(labs, return_index=True, return_counts=True)
    labels: List[int] = []
    areas: List[int] = []
    flats: List[np.ndarray] = []
    for i in range(len(uniq)):
        lab = int(uniq[i])
        if keep is not None and lab not in keep:
            continue
        s = int(starts[i]); n = int(counts[i])
        labels.append(lab)
        areas.append(n)
        flats.append(fg[s:s + n])
    return np.array(labels, np.int64), np.array(areas, np.int64), flats


def track_overlap(
    df: pd.DataFrame,
    masks: np.ndarray,
    min_iou: float = 0.1,
    max_gap: int = 1,
    no_match_cost: Optional[float] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> pd.DataFrame:
    """Track segmented objects by **mask overlap (IoU)** with a birth/death LAP.

    For dense, slowly-moving nuclei (e.g. StarDist H2B masks) overlap is far more
    discriminative than centroid distance: an object overlaps its own previous
    mask heavily and a neighbor's barely at all, so neither a wide gate (spurious
    long links) nor a tight one (neighbor "currents") is needed. Each consecutive
    frame pair is linked on ``cost = 1 - IoU`` (pairs below ``min_iou`` forbidden)
    via :func:`solve_lap`, so an object with no real overlap starts/ends a track
    instead of being force-linked. A track keeps its last footprint for up to
    ``max_gap`` missed frames so a dropped detection can still re-link by overlap.

    Parameters
    ----------
    df : DataFrame with columns ``frame``, ``label`` (``label`` must equal the
        integer id in ``masks`` for that frame; extra columns are preserved).
    masks : (T, H, W) int label image for this group's segmentation channel.
    min_iou : float, minimum intersection-over-union to allow a link.
    max_gap : int, frames a track may vanish and still re-link by overlap.
    no_match_cost : float, optional; birth/death cost. Defaults to 1.0 (the cost
        of zero overlap), so any above-``min_iou`` overlap beats starting anew.
    progress_cb : callable(fraction_0_1, message).

    Returns
    -------
    df : copy with a ``track_id`` column (``-1`` where unassigned).
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(f"track_overlap needs (T,H,W) masks, got {masks.shape!r}")
    n_masks, H, W = masks.shape

    by_frame = {int(f): sub for f, sub in df.groupby("frame", sort=True)}
    frames = sorted(by_frame)
    T = len(frames)
    d = float(1.0 if no_match_cost is None else no_match_cost)

    log.info(
        "CT overlap tracking: %d detections across %d frames, min_iou=%.2f, "
        "max_gap=%d", len(df), T, min_iou, max_gap,
    )
    _t0 = time.perf_counter()

    next_id = 1
    track_ids: dict = {}
    active: dict = {}                       # tid -> {last_frame, flat, area}
    carry = np.zeros(H * W, dtype=np.int32)  # reused prev-footprint paint buffer

    if T == 0:
        df = df.copy()
        df["track_id"] = pd.Series([], dtype=int)
        return df

    def _seed(frame: int, labels, areas, flats) -> None:
        nonlocal next_id
        for lab, ar, fl in zip(labels, areas, flats):
            tid = next_id
            next_id += 1
            track_ids[(frame, int(lab))] = tid
            active[tid] = {"last_frame": frame, "flat": fl, "area": int(ar)}

    # First frame: every measured object starts a track.
    f0 = frames[0]
    keep0 = {int(x) for x in by_frame[f0]["label"].to_numpy()}
    if f0 < n_masks:
        _seed(f0, *_frame_footprints(masks[f0], keep0))
    if progress_cb:
        progress_cb(1.0 / T, f"Linking frame 1/{T}")

    for i in range(1, T):
        cf = frames[i]
        if cf >= n_masks:
            if progress_cb:
                progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
            continue
        keep = {int(x) for x in by_frame[cf]["label"].to_numpy()}
        cur_labels, cur_areas, cur_flats = _frame_footprints(masks[cf], keep)
        C = len(cur_labels)
        if C == 0:
            if progress_cb:
                progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
            continue

        # Tracks still within the gap budget are candidates for re-link.
        alive = [(tid, info) for tid, info in active.items()
                 if cf - info["last_frame"] <= max_gap + 1]
        if not alive:
            _seed(cf, cur_labels, cur_areas, cur_flats)
            if progress_cb:
                progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
            continue

        # Paint each alive track's last footprint into the carry buffer, older
        # first so a more-recent footprint wins any pixel contested after motion.
        alive.sort(key=lambda kv: kv[1]["last_frame"])
        K = len(alive)
        prev_area = np.empty(K, np.int64)
        idx2tid = [0] * K
        painted: List[np.ndarray] = []
        for k, (tid, info) in enumerate(alive):
            fl = info["flat"]
            carry[fl] = k + 1
            painted.append(fl)
            prev_area[k] = info["area"]
            idx2tid[k] = tid

        # Overlap crosstab: for each current object, tally shared pixels against
        # every alive footprint it touches, then convert to IoU cost.
        cost = np.full((K, C), np.inf, dtype=np.float64)
        for c in range(C):
            ka = carry[cur_flats[c]]
            hit = ka > 0
            if not hit.any():
                continue
            kk, ov = np.unique(ka[hit], return_counts=True)
            for k1, o in zip(kk, ov):
                k = int(k1) - 1
                union = prev_area[k] + int(cur_areas[c]) - int(o)
                iou = (o / union) if union > 0 else 0.0
                if iou >= min_iou:
                    cost[k, c] = 1.0 - iou

        for fl in painted:              # reset only touched pixels (cheap)
            carry[fl] = 0

        matches, _, _ = solve_lap(cost, d)
        matched_c = set()
        for k, c in matches:
            tid = idx2tid[k]
            track_ids[(cf, int(cur_labels[c]))] = tid
            active[tid] = {"last_frame": cf, "flat": cur_flats[c],
                           "area": int(cur_areas[c])}
            matched_c.add(c)

        for c in range(C):              # unmatched detections start new tracks
            if c not in matched_c:
                tid = next_id
                next_id += 1
                track_ids[(cf, int(cur_labels[c]))] = tid
                active[tid] = {"last_frame": cf, "flat": cur_flats[c],
                               "area": int(cur_areas[c])}

        dead = [tid for tid, info in active.items()
                if cf - info["last_frame"] > max_gap + 1]
        for tid in dead:
            del active[tid]

        if progress_cb:
            progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")

    df = df.copy()
    frame_arr = df["frame"].to_numpy()
    label_arr = df["label"].to_numpy()
    df["track_id"] = [
        track_ids.get((int(f), int(lbl)), -1)
        for f, lbl in zip(frame_arr, label_arr)
    ]
    log.info("CT overlap done: %d tracks in %.2fs",
             next_id - 1, time.perf_counter() - _t0)
    if progress_cb:
        progress_cb(1.0, "Tracking done")

    return df


# ======================================================================
# VENDORED FROM: nd2studios/backend/object_tracker.py
# ======================================================================

"""
Object tracker for the Pipelines / Results tabs.

Implements a frame-to-frame Hungarian centroid linker that assigns stable
track IDs to objects across consecutive T frames.  The per-track reference
centroid (and area) **updates every frame** — it follows the object as it
moves rather than anchoring to the first detection — and a track can survive a
short detection gap (occlusion / missed segmentation) before it is retired.
Pure NumPy / SciPy — no Qt imports.

Driven by the "Track Objects" pipeline node (see
``link_objects_with_params``) and called implicitly after
``compute_measurements`` so downstream Review / if-else steps have track ids.
"""

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)

# Progress callback: fraction in [0, 1] plus a short status message. Kept
# Qt-free so the backend stays importable without PySide6; the GUI's
# ``_TrackJob`` adapts it onto its ``ProgressReporter``.
ProgressCB = Callable[[float, str], None]


# Tracking methods exposed by the "Track Objects" node.  The linker dispatches on
# this so methods slot in as new entries without changing call sites.
#   * METHOD_CENTROID    — Hungarian nearest-neighbor on object centroids.
#   * METHOD_SERIALTRACK — SerialTrack topology PTV (scale/rotation invariant);
#     vendored under ``nd2studios.backend.serialtrack`` and driven via its
#     ``track_coordinates`` path (the object centroids are the pre-detected
#     particles, so no image re-detection happens).
#   * METHOD_CT_TOPOLOGY / METHOD_CT_FINGERPRINT — CellTracker's topology-Hungarian
#     and spatial-fingerprint linkers, vendored under
#     ``nd2studios.backend.celltracker`` (pandas-DataFrame trackers; the per-group
#     ``_link_group_celltracker`` bridges the row-dicts to/from that shape).
METHOD_CENTROID = "Centroid (nearest-neighbor)"
METHOD_SERIALTRACK = "SerialTrack (topology PTV)"
METHOD_CT_TOPOLOGY = "Cell-Tracker: Topology (Hungarian)"
METHOD_CT_FINGERPRINT = "Cell-Tracker: Spatial Fingerprint"
#   * METHOD_CT_OVERLAP — links segmented objects by mask IoU (Jaqaman LAP with
#     birth/death), the robust default for dense, slowly-moving nuclei; consumes
#     the per-(channel, m) StarDist label masks passed via ``label_masks``.
METHOD_CT_OVERLAP = "Cell-Tracker: Mask Overlap (IoU)"
TRACKING_METHODS: List[str] = [
    METHOD_CENTROID, METHOD_SERIALTRACK, METHOD_CT_TOPOLOGY, METHOD_CT_FINGERPRINT,
    METHOD_CT_OVERLAP,
]


def link_objects(
    rows: List[Dict[str, Any]],
    max_displacement_px: float = 100.0,
    min_track_length: int = 2,
    min_circularity: float = 0.0,
    max_eccentricity: float = 1.0,
    max_size_diff_frac: float = 1.0,
    max_frame_gap: int = 0,
    method: str = METHOD_CENTROID,
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
    ct_n_neighbors: int = 5,
    ct_topo_weight: float = 0.3,
    ct_area_weight: float = 0.3,
    ct_max_gap: int = 3,
    ct_min_iou: float = 0.1,
    label_masks: Optional[Dict[Tuple[str, int], "np.ndarray"]] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> List[Dict[str, Any]]:
    """Assign track_id, track_length, and track_validation to every row.

    Mutates *rows* in-place and returns the same list.

    Parameters
    ----------
    rows:
        List of measurement dicts produced by compute_measurements().
        Each dict must have: segmentation_channel, frame,
        centroid_y_px, centroid_x_px, area_px.
        m_position is optional (defaults to 0 when absent).
    max_displacement_px:
        Maximum centroid displacement (Euclidean, pixels) between linked
        detections.  Objects farther apart than this are not the same track.
    min_track_length:
        Minimum number of frames a track must span to be kept.  Shorter tracks
        receive track_id = None and track_validation = None.
    min_circularity:
        Objects whose circularity (4π·area/perimeter²) is below this value are
        excluded from tracking.  Range 0–1; default 0.0 disables the filter.
    max_eccentricity:
        Objects whose eccentricity exceeds this value are excluded from
        tracking.  Range 0–1; default 1.0 disables the filter.
    max_size_diff_frac:
        Maximum fractional change in object area between linked detections,
        measured as ``|area_a - area_b| / max(area_a, area_b)`` (range 0–1).
        Detections whose size changes by more than this are not linked.
        1.0 disables the size gate.
    max_frame_gap:
        Number of consecutive missed frames a track may bridge before it is
        retired.  0 = the track must be re-detected in the very next frame;
        2 = it may skip up to two frames and re-link afterwards.
    method:
        Tracking method — ``METHOD_CENTROID`` (default) or ``METHOD_SERIALTRACK``.
    st_mode:
        SerialTrack mode, ``"Incremental"`` (link each frame to the previous) or
        ``"Cumulative"`` (link every frame to the first).  Ignored for the
        centroid method.
    st_n_neighbors:
        SerialTrack topology-descriptor neighbor count (``n_neighbors_max``).
        Ignored for the centroid method.
    st_solver:
        SerialTrack global-step solver: ``"MLS"`` (mesh-free moving least
        squares), ``"Regularization"`` (scatter→grid smoothing; default), or
        ``"ADMM"`` (augmented-Lagrangian with automatic L-curve α — most faithful
        to the paper, costlier).  SerialTrack only.
    st_loc_solver:
        SerialTrack local matcher: ``"Topology"`` or
        ``"Histogram then Topology"``.  SerialTrack only.
    st_n_neighbors_min:
        Floor for the exponential neighbor-count decay across iterations
        (``n_neighbors_min``).  SerialTrack only.
    st_smoothness:
        Global smoothing strength (the ``α/µ`` knob; used by Regularization and
        ADMM).  SerialTrack only.
    st_outlier_threshold:
        Westerweel normalized-median-residual cutoff (``0`` disables).
        SerialTrack only.
    st_max_iter:
        Max ADMM iterations per frame pair.  SerialTrack only.
    st_iter_stop_threshold:
        ADMM convergence threshold on the displacement-update norm.  SerialTrack
        only.
    st_dist_missing:
        Ghost-particle cull distance ``ε_d`` (px), active in late iterations.
        SerialTrack only.
    st_use_prev_results:
        Enable the data-driven initial-guess predictor (warm start) for frames
        ≥3.  The POD-GPR stage (frames ≥7) needs scikit-learn.  SerialTrack only.
    ct_n_neighbors:
        CellTracker topology-Hungarian neighbor count for the rotation-invariant
        descriptor.  Used only by ``METHOD_CT_TOPOLOGY``.
    ct_topo_weight:
        CellTracker topology cost weight (0–1) blended with raw distance.  Used
        only by ``METHOD_CT_TOPOLOGY``.
    ct_area_weight:
        CellTracker fingerprint area-similarity weight (0–1) vs. distance.  Used
        only by ``METHOD_CT_FINGERPRINT``.
    ct_max_gap:
        CellTracker fingerprint gap-filling budget — frames a track may vanish
        and still re-link.  Used by ``METHOD_CT_FINGERPRINT`` and
        ``METHOD_CT_OVERLAP``.
    ct_min_iou:
        Minimum mask intersection-over-union to link two detections.  Used only
        by ``METHOD_CT_OVERLAP``.
    label_masks:
        Optional ``{(segmentation_channel, m_position): (T, H, W) int label
        image}`` consumed by ``METHOD_CT_OVERLAP`` for mask-IoU linking.  When a
        group's masks are absent the overlap method falls back to the fingerprint
        linker so it still produces tracks.

    Returns
    -------
    The same list, with three new keys added to every dict:

    track_id          int | None  — None if excluded from tracking or
                                    track_length < min_track_length
    track_length      int         — frames the track spans
    track_validation  str | None  — "unvalidated" if track_id is not None,
                                    else None
    """
    min_track_length = max(1, int(min_track_length))
    max_frame_gap = max(0, int(max_frame_gap))

    # ── Initialise all rows ───────────────────────────────────────────────────
    for r in rows:
        r["track_id"] = None
        r["track_length"] = 1
        r["track_validation"] = None

    if not rows:
        return rows

    # ── Morphology filter — ineligible rows keep track_id = None ─────────────
    def _passes(r: Dict[str, Any]) -> bool:
        circ = r.get("circularity")
        if circ is not None and float(circ) < min_circularity:
            return False
        ecc = r.get("eccentricity")
        if ecc is not None and float(ecc) > max_eccentricity:
            return False
        return True

    # ── Group eligible rows by (channel, m_position) ─────────────────────────
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not _passes(r):
            continue
        key = (
            str(r.get("segmentation_channel", "")),
            int(r.get("m_position", 0)),
        )
        groups[key].append(r)

    _next_track_id = [1]  # mutable counter shared across groups

    n_groups = max(1, len(groups))
    log.info(
        "link_objects: method=%r, %d eligible detections in %d (channel, m) "
        "group(s)", method, sum(len(g) for g in groups.values()), n_groups,
    )
    _t_start = time.perf_counter()

    for gi, (group_key, group_rows) in enumerate(groups.items()):
        # Map each linker's own 0..1 progress into this group's slice of the
        # overall bar, so multi-group runs still advance smoothly and a
        # single-group run (the common case) passes progress straight through.
        def _group_cb(frac: float, msg: str, _gi: int = gi) -> None:
            if progress_cb is not None:
                progress_cb((_gi + max(0.0, min(1.0, frac))) / n_groups, msg)

        if method == METHOD_CT_OVERLAP:
            _link_group_overlap(
                group_rows, group_key, max_displacement_px,
                ct_min_iou, ct_max_gap, label_masks,
                _next_track_id, progress_cb=_group_cb,
            )
        elif method == METHOD_SERIALTRACK:
            _link_group_serialtrack(
                group_rows, max_displacement_px, st_mode, st_n_neighbors,
                _next_track_id,
                solver_str=st_solver, loc_solver_str=st_loc_solver,
                n_neighbors_min=st_n_neighbors_min, smoothness=st_smoothness,
                outlier_threshold=st_outlier_threshold, max_iter=st_max_iter,
                iter_stop_threshold=st_iter_stop_threshold,
                dist_missing=st_dist_missing, use_prev_results=st_use_prev_results,
                progress_cb=_group_cb,
            )
        elif method in (METHOD_CT_TOPOLOGY, METHOD_CT_FINGERPRINT):
            _link_group_celltracker(
                group_rows, method, max_displacement_px,
                ct_n_neighbors, ct_topo_weight, ct_area_weight, ct_max_gap,
                _next_track_id, progress_cb=_group_cb,
            )
        else:
            _link_group(
                group_rows, max_displacement_px, max_size_diff_frac,
                max_frame_gap, _next_track_id, progress_cb=_group_cb,
            )

    log.info("link_objects: linking finished in %.2fs (%d tracks assigned)",
             time.perf_counter() - _t_start, _next_track_id[0] - 1)
    if progress_cb is not None:
        progress_cb(1.0, "Tracking done")

    # ── Promote track_length and track_validation ─────────────────────────────
    track_frames: Dict[int, int] = defaultdict(int)
    for r in rows:
        tid = r["track_id"]
        if tid is not None:
            track_frames[tid] += 1

    for r in rows:
        tid = r["track_id"]
        if tid is None:
            continue
        n = track_frames[tid]
        if n < min_track_length:
            r["track_id"] = None
            r["track_length"] = n
        else:
            r["track_length"] = n
            r["track_validation"] = "unvalidated"

    return rows


def link_objects_with_params(
    rows: List[Dict[str, Any]],
    params: Dict[str, Any],
    pixel_size_um: Optional[float] = None,
    label_masks: Optional[Dict[Tuple[str, int], "np.ndarray"]] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> List[Dict[str, Any]]:
    """Run :func:`link_objects` from a "Track Objects" node param dict.

    Translates the node's user-facing knobs into :func:`link_objects` arguments,
    converting a µm distance threshold to pixels via *pixel_size_um* when
    available.  Knobs:

    * shared — ``method``, ``max_distance`` with ``distance_unit`` of pixels/µm
      (becomes ``max_displacement_px``; SerialTrack and the CellTracker linkers
      use it as the field of search / max link distance), ``min_track_length``.
    * centroid only — ``max_size_diff``, ``max_frame_gap``.
    * SerialTrack only — ``st_mode``, ``st_n_neighbors``, ``st_solver``,
      ``st_loc_solver``, ``st_n_neighbors_min``, ``st_smoothness``,
      ``st_outlier_threshold``, ``st_max_iter``, ``st_iter_stop_threshold``,
      ``st_dist_missing``, ``st_use_prev_results``.
    * Cell-Tracker topology only — ``ct_n_neighbors``, ``ct_topo_weight``.
    * Cell-Tracker fingerprint only — ``ct_area_weight``, ``ct_max_gap``.

    Params that don't apply to the chosen method are passed through to
    :func:`link_objects` but ignored by the active code path.
    """
    params = params or {}
    max_distance = float(params.get("max_distance", 100.0))
    unit = str(params.get("distance_unit", "pixels"))
    if unit == "µm" and pixel_size_um:
        max_distance = max_distance / float(pixel_size_um)
    return link_objects(
        rows,
        max_displacement_px=max_distance,
        min_track_length=int(params.get("min_track_length", 2)),
        max_size_diff_frac=float(params.get("max_size_diff", 1.0)),
        max_frame_gap=int(params.get("max_frame_gap", 0)),
        method=str(params.get("method", METHOD_CENTROID)),
        st_mode=str(params.get("st_mode", "Incremental")),
        st_n_neighbors=int(params.get("st_n_neighbors", 25)),
        st_solver=str(params.get("st_solver", "Regularization")),
        st_loc_solver=str(params.get("st_loc_solver", "Topology")),
        st_n_neighbors_min=int(params.get("st_n_neighbors_min", 1)),
        st_smoothness=float(params.get("st_smoothness", 0.1)),
        st_outlier_threshold=float(params.get("st_outlier_threshold", 5.0)),
        st_max_iter=int(params.get("st_max_iter", 20)),
        st_iter_stop_threshold=float(params.get("st_iter_stop_threshold", 1e-2)),
        st_dist_missing=float(params.get("st_dist_missing", 5.0)),
        st_use_prev_results=bool(params.get("st_use_prev_results", False)),
        ct_n_neighbors=int(params.get("ct_n_neighbors", 5)),
        ct_topo_weight=float(params.get("ct_topo_weight", 0.3)),
        ct_area_weight=float(params.get("ct_area_weight", 0.3)),
        ct_max_gap=int(params.get("ct_max_gap", 3)),
        ct_min_iou=float(params.get("ct_min_iou", 0.1)),
        label_masks=label_masks,
        progress_cb=progress_cb,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

# active[track_id] layout: (cy, cx, area, last_frame)
_Active = Tuple[float, float, float, int]


def _link_group(
    group_rows: List[Dict[str, Any]],
    max_displacement_px: float,
    max_size_diff_frac: float,
    max_frame_gap: int,
    next_id: List[int],
    progress_cb: Optional[ProgressCB] = None,
) -> None:
    """Link objects within a single (channel, m_position) group.

    Walks the detected frames in order.  Each track keeps its *last matched*
    centroid + area + frame, so the reference moves with the object.  A track
    not re-detected this frame is carried forward (up to ``max_frame_gap``
    missed frames) instead of being dropped, which lets it re-link across a gap.
    """
    frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in group_rows:
        frames[int(r.get("frame", 0))].append(r)

    sorted_frames = sorted(frames.keys())
    T = len(sorted_frames)
    if T < 2:
        return

    # Seed first frame with fresh track IDs.
    active: Dict[int, _Active] = {}
    for r in frames[sorted_frames[0]]:
        r["track_id"] = next_id[0]
        active[next_id[0]] = (_cy(r), _cx(r), _area(r), sorted_frames[0])
        next_id[0] += 1

    for i, fr in enumerate(sorted_frames[1:], start=1):
        if progress_cb is not None:
            progress_cb((i + 1) / T, f"Linking frame {i + 1}/{T}")
        curr_rows = frames[fr]
        if not curr_rows:
            continue

        # Retire tracks that have now exceeded the allowed gap.
        active = {
            tid: v for tid, v in active.items()
            if (fr - v[3] - 1) <= max_frame_gap
        }

        prev_track_ids = list(active.keys())
        if not prev_track_ids:
            for r in curr_rows:
                r["track_id"] = next_id[0]
                active[next_id[0]] = (_cy(r), _cx(r), _area(r), fr)
                next_id[0] += 1
            continue

        # Cost = Euclidean centroid distance, gated by distance + size change.
        prev_vals = [active[tid] for tid in prev_track_ids]
        prev_centroids = np.array([[v[0], v[1]] for v in prev_vals], dtype=np.float64)
        prev_areas = np.array([v[2] for v in prev_vals], dtype=np.float64)
        curr_centroids = np.array([(_cy(r), _cx(r)) for r in curr_rows], dtype=np.float64)
        curr_areas = np.array([_area(r) for r in curr_rows], dtype=np.float64)

        diff = prev_centroids[:, None, :] - curr_centroids[None, :, :]  # (P, C, 2)
        cost = np.sqrt((diff ** 2).sum(axis=2))                          # (P, C)

        # Size-difference gate: |Δarea| / max(area) must stay within the
        # threshold, else the pair cannot be the same object.
        area_diff = np.abs(prev_areas[:, None] - curr_areas[None, :])
        area_max = np.maximum.outer(prev_areas, curr_areas)
        area_max[area_max <= 0.0] = 1.0
        size_diff = area_diff / area_max
        blocked = size_diff > max_size_diff_frac
        cost[blocked] = max_displacement_px + 1.0

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_curr: set = set()
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] <= max_displacement_px:
                tid = prev_track_ids[ri]
                curr_rows[ci]["track_id"] = tid
                active[tid] = (_cy(curr_rows[ci]), _cx(curr_rows[ci]),
                               _area(curr_rows[ci]), fr)
                matched_curr.add(ci)

        # Unmatched current objects start new tracks.  Unmatched *previous*
        # tracks stay in `active` with their old last_frame, so they remain
        # eligible to re-link until the gap budget runs out.
        for ci, r in enumerate(curr_rows):
            if ci not in matched_curr:
                r["track_id"] = next_id[0]
                active[next_id[0]] = (_cy(r), _cx(r), _area(r), fr)
                next_id[0] += 1


def _link_group_serialtrack(
    group_rows: List[Dict[str, Any]],
    f_o_s: float,
    mode_str: str,
    n_neighbors_max: int,
    next_id: List[int],
    *,
    solver_str: str = "Regularization",
    loc_solver_str: str = "Topology",
    n_neighbors_min: int = 1,
    smoothness: float = 0.1,
    outlier_threshold: float = 5.0,
    max_iter: int = 20,
    iter_stop_threshold: float = 1e-2,
    dist_missing: float = 5.0,
    use_prev_results: bool = False,
    progress_cb: Optional[ProgressCB] = None,
) -> None:
    """Link objects within one (channel, m_position) group via SerialTrack.

    The object centroids are fed to SerialTrack's ``track_coordinates`` path as
    pre-detected particles (no image re-detection).  ``f_o_s`` is the field of
    search (max neighbor radius, px) — the node's "Max distance" knob.  The
    remaining keyword args are SerialTrack's own tunables (global/local solver,
    neighbor-count decay, smoothing, outlier + ghost-cull thresholds, ADMM
    iteration budget, warm-start predictor).  Each detection's track id is chained
    from the per-frame ``track_b2a`` index maps; ``min_track_length`` filtering
    happens in the caller's post-pass.

    The SerialTrack library (numba JIT) is imported lazily so a missing optional
    dependency surfaces only for this method and is caught by the node handlers.
    """

    # Group by frame, keep only frames that actually have detections, in order.
    frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in group_rows:
        frames[int(r.get("frame", 0))].append(r)
    sorted_frames = [f for f in sorted(frames.keys()) if frames[f]]
    if len(sorted_frames) < 2:
        return  # nothing to link (mirrors _link_group's early-out)

    # coords_list[i] aligns row-for-row with row_refs[i] (same object order).
    coords_list: List[np.ndarray] = []
    row_refs: List[List[Dict[str, Any]]] = []
    for fr in sorted_frames:
        rows_f = frames[fr]
        coords_list.append(
            np.array([[_cy(r), _cx(r)] for r in rows_f], dtype=np.float64)
        )
        row_refs.append(rows_f)

    mode = (TrackingMode.CUMULATIVE if mode_str == "Cumulative"
            else TrackingMode.INCREMENTAL)
    global_solver = {
        "MLS": GlobalSolver.MLS,
        "Regularization": GlobalSolver.REGULARIZATION,
        "ADMM": GlobalSolver.ADMM,
    }.get(solver_str, GlobalSolver.REGULARIZATION)
    local_solver = {
        "Topology": LocalSolver.TOPOLOGY,
        "Histogram then Topology": LocalSolver.HISTOGRAM_THEN_TOPOLOGY,
    }.get(loc_solver_str, LocalSolver.TOPOLOGY)
    n_max = max(2, int(n_neighbors_max))
    trk = TrackingConfig(
        mode=mode,
        f_o_s=float(f_o_s),
        n_neighbors_max=n_max,
        n_neighbors_min=min(n_max, max(1, int(n_neighbors_min))),
        loc_solver=local_solver,
        solver=global_solver,
        smoothness=max(0.0, float(smoothness)),
        outlier_threshold=max(0.0, float(outlier_threshold)),
        max_iter=max(1, int(max_iter)),
        iter_stop_threshold=max(0.0, float(iter_stop_threshold)),
        dist_missing=max(0.0, float(dist_missing)),
        use_prev_results=bool(use_prev_results),
        strain_n_neighbors=0,      # skip per-frame strain (not needed for ids)
    )
    # SerialTrack runs as one opaque (numba-JIT) call — no intra-call progress
    # hook — so we mark the start; per-frame updates follow in the chaining loop.
    if progress_cb is not None:
        progress_cb(0.0, f"SerialTrack linking {len(sorted_frames)} frames…")
    try:
        session = SerialTracker(
            DetectionConfig(), trk).track_coordinates(coords_list)
    except ImportError as exc:
        # The only optional import on this path is scikit-learn, pulled in by the
        # POD-GPR warm start (frames ≥7) when use_prev_results is on.
        if use_prev_results:
            raise RuntimeError(
                "SerialTrack 'Use previous results' (POD-GPR warm start) needs "
                "scikit-learn for sequences of 7+ frames — install it "
                "(pip install scikit-learn) or turn the option off."
            ) from exc
        raise

    # Seed the reference (first) frame with fresh ids, then chain forward.
    ids_per_frame: List[List[int]] = [[] for _ in row_refs]
    for r in row_refs[0]:
        r["track_id"] = next_id[0]
        ids_per_frame[0].append(next_id[0])
        next_id[0] += 1

    n_pairs = max(1, len(session.frame_results))
    for k, res in enumerate(session.frame_results):
        if progress_cb is not None:
            progress_cb((k + 1) / n_pairs, f"Chaining frame {k + 2}/{len(row_refs)}")
        # res is the pair whose B is coords_list[k + 1].
        prev_ids = ids_per_frame[0] if mode == TrackingMode.CUMULATIVE \
            else ids_per_frame[k]
        t_b2a = np.asarray(res.track_b2a)
        b_rows = row_refs[k + 1]
        b_ids: List[int] = []
        for j, rrow in enumerate(b_rows):
            a = int(t_b2a[j]) if j < len(t_b2a) else -1
            if 0 <= a < len(prev_ids):
                tid = prev_ids[a]
            else:                       # appeared / untracked → new track
                tid = next_id[0]
                next_id[0] += 1
            rrow["track_id"] = tid
            b_ids.append(tid)
        ids_per_frame[k + 1] = b_ids


def _link_group_celltracker(
    group_rows: List[Dict[str, Any]],
    method: str,
    max_displacement_px: float,
    ct_n_neighbors: int,
    ct_topo_weight: float,
    ct_area_weight: float,
    ct_max_gap: int,
    next_id: List[int],
    progress_cb: Optional[ProgressCB] = None,
) -> None:
    """Link objects within one (channel, m_position) group via CellTracker.

    Bridges ND2Studios' row-dicts to CellTracker's DataFrame convention: each row
    becomes a (``frame``, ``label``, ``centroid_y``, ``centroid_x``, ``area``)
    record (``label`` = the per-frame ``label_id``), the chosen vendored linker
    runs, and the resulting per-call ``track_id`` (1-based) is remapped onto the
    shared global ``next_id`` counter so ids never collide across groups.  The
    caller's post-pass handles ``track_length`` / ``min_track_length``.

    pandas / CellTracker are imported lazily so a missing optional dependency
    surfaces only for these methods and is caught by the node handlers.
    """
    import pandas as pd


    # Build the DataFrame; keep a parallel handle from (frame, label) back to the
    # originating row so the assigned id can be written in place.
    records: List[Dict[str, Any]] = []
    row_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in group_rows:
        fr = int(r.get("frame", 0))
        lbl = int(r.get("label_id", 0))
        records.append({
            "frame": fr,
            "label": lbl,
            "centroid_y": _cy(r),
            "centroid_x": _cx(r),
            "area": _area(r),
        })
        row_by_key[(fr, lbl)] = r

    df = pd.DataFrame(records)
    if df.empty or df["frame"].nunique() < 2:
        return  # nothing to link (mirrors _link_group's early-out)

    if method == METHOD_CT_FINGERPRINT:
        tracked = track_fingerprint(
            df, max_dist=float(max_displacement_px),
            area_weight=float(ct_area_weight), max_gap=max(0, int(ct_max_gap)),
            progress_cb=progress_cb,
        )
    else:  # METHOD_CT_TOPOLOGY
        tracked = track_timeseries(
            df, max_dist=float(max_displacement_px),
            n_neighbors=max(1, int(ct_n_neighbors)),
            use_topology=True, topo_weight=float(ct_topo_weight),
            progress_cb=progress_cb,
        )

    # Remap local (per-call) track ids to the shared global counter, skipping the
    # -1 "unassigned" sentinel, then write onto the originating rows.
    local_to_global: Dict[int, int] = {}
    for rec in tracked.itertuples(index=False):
        local = int(getattr(rec, "track_id"))
        if local < 0:
            continue
        if local not in local_to_global:
            local_to_global[local] = next_id[0]
            next_id[0] += 1
        row = row_by_key.get((int(rec.frame), int(rec.label)))
        if row is not None:
            row["track_id"] = local_to_global[local]


def _link_group_overlap(
    group_rows: List[Dict[str, Any]],
    group_key: Tuple[str, int],
    max_displacement_px: float,
    ct_min_iou: float,
    ct_max_gap: int,
    label_masks: Optional[Dict[Tuple[str, int], Any]],
    next_id: List[int],
    progress_cb: Optional[ProgressCB] = None,
) -> None:
    """Link one (channel, m_position) group by StarDist mask overlap (IoU).

    Bridges ND2Studios' row-dicts to CellTracker's DataFrame convention (as
    ``_link_group_celltracker``), then runs ``track_overlap`` on this group's
    ``(T, H, W)`` label image, looked up from ``label_masks[group_key]``. When
    the masks are unavailable the linker falls back to the fingerprint linker
    (position + area + birth/death) so the method still yields tracks rather than
    erroring. Local (per-call) ids are remapped onto the shared ``next_id``
    counter so ids never collide across groups.

    pandas / CellTracker are imported lazily so a missing optional dependency
    surfaces only for this method and is caught by the node handlers.
    """
    import pandas as pd


    records: List[Dict[str, Any]] = []
    row_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in group_rows:
        fr = int(r.get("frame", 0))
        lbl = int(r.get("label_id", 0))
        records.append({
            "frame": fr, "label": lbl,
            "centroid_y": _cy(r), "centroid_x": _cx(r), "area": _area(r),
        })
        row_by_key[(fr, lbl)] = r

    df = pd.DataFrame(records)
    if df.empty or df["frame"].nunique() < 2:
        return

    masks = None
    if label_masks:
        masks = label_masks.get(group_key)
        if masks is None and len(label_masks) == 1:
            # Single-group run keyed slightly differently — use the only masks.
            masks = next(iter(label_masks.values()))

    if masks is None:
        log.warning(
            "CT overlap: no label masks for group %r — falling back to the "
            "Spatial Fingerprint linker.", group_key,
        )
        tracked = track_fingerprint(
            df, max_dist=float(max_displacement_px), area_weight=0.3,
            max_gap=max(0, int(ct_max_gap)), progress_cb=progress_cb,
        )
    else:
        tracked = track_overlap(
            df, np.asarray(masks), min_iou=float(ct_min_iou),
            max_gap=max(1, int(ct_max_gap)), progress_cb=progress_cb,
        )

    local_to_global: Dict[int, int] = {}
    for rec in tracked.itertuples(index=False):
        local = int(getattr(rec, "track_id"))
        if local < 0:
            continue
        if local not in local_to_global:
            local_to_global[local] = next_id[0]
            next_id[0] += 1
        row = row_by_key.get((int(rec.frame), int(rec.label)))
        if row is not None:
            row["track_id"] = local_to_global[local]


def _cy(row: Dict[str, Any]) -> float:
    return float(row.get("centroid_y_px") or 0.0)


def _cx(row: Dict[str, Any]) -> float:
    return float(row.get("centroid_x_px") or 0.0)


def _area(row: Dict[str, Any]) -> float:
    """Object area in pixels, used by the size-difference gate (0 if absent)."""
    return float(row.get("area_px") or 0.0)
