"""
aldvc_field — self-contained Augmented Lagrangian Digital Volume/Image
Correlation (ALDVC) math kernel, vendored out of ND2Studios for reuse in a
different node system.

WHAT THIS DOES
--------------
Given two same-shape volumes (a reference and a deformed one), 2D ``(H, W)`` or
3D ``(Z, H, W)``, it computes a DENSE displacement field (in voxels) on a regular
subset grid plus a strain tensor field. Entry point: ``run_aldvc(...)`` -> a
``DVCResult`` dataclass. The pipeline is: Stage 0 normalize + spline-prefilter,
Stage 1 FFT integer seed (multigrid), Stage 2 outlier clean + inpaint, Stages 3-6
local IC-GN + ADMM global compatibility loop, Stage 7 strain.

WHERE THE REAL MATH LIVES
-------------------------
IN-REPO. This is NOT a thin wrapper around a third-party DVC package. The
algorithm is a clean-room Python/numpy/scipy port of FranckLab's MATLAB ALDVC
(Yang, Hazlett, Landauer & Franck, Exp. Mech. 2020) written natively in the
ND2Studios ``nd2studios/backend/dvc/`` package. Every stage (FFT NCC seed,
inverse-compositional Gauss-Newton subset registration, sparse finite-difference
global compatibility solve, ADMM outer loop, strain tensor) is implemented here
in pure numpy/scipy/scikit-image. Optional CuPy accelerates only the seed FFT and
is imported lazily with a CPU fallback.

PROVENANCE (branch: Version-1.45)
---------------------------------
Vendored verbatim (byte-copied, deps-first) from:
  - nd2studios/compute/parallel/shared_array.py   (shared_ndarray, attach_shared)
  - nd2studios/core/dvc_registry.py               (DVCResult, DVCParams)
  - nd2studios/backend/dvc/mesh.py
  - nd2studios/backend/dvc/outliers.py
  - nd2studios/backend/dvc/strain.py
  - nd2studios/backend/dvc/integer_search.py
  - nd2studios/backend/dvc/global_step.py
  - nd2studios/backend/dvc/icgn.py
  - nd2studios/backend/dvc/parallel.py
  - nd2studios/backend/dvc/admm.py
  - nd2studios/backend/dvc/engine.py
  - nd2studios/backend/dvc/tracking.py
  - nd2studios/backend/dvc/method.py

Vendored verbatim; imports nothing from nd2studios; caller owns all prep
(no file I/O, no per-multipoint/per-timepoint looping, no crop/downsample/
registration/exclusion, no singleton-Z squeeze — those are the caller's job).

EDITS MADE DURING VENDORING (only the permitted kinds)
------------------------------------------------------
(a) Removed all ``from nd2studios...`` and duplicate ``from __future__`` imports;
    the internal module references now resolve within this single file:
      * global_step's ``import ... mesh as _mesh`` -> the ``_mesh.`` prefix was
        stripped so it calls the in-file ``pack_u``/``pack_F``/``unpack_u``/
        ``unpack_F`` directly.
      * icgn's lazy ``from ...parallel import parallel_local_icgn`` and engine's
        lazy ``from ...parallel import default_workers`` now reference the in-file
        definitions.
      * parallel's worker-side lazy ``from ...icgn import (...)`` removed; those
        helpers are module-level here.
(c) DROPPED UI/registry-only members (NOT on the compute path):
      * ``DVCMethod`` ABC registry base + ``@DVCMethod.register`` decorator +
        ``get_methods``/``get_method``/``register`` (from dvc_registry.py).
      * ``ALDVCMethod.get_params()`` (returned a ParamSpec UI list) and the
        ``@DVCMethod.register`` decorator on it; ``ALDVCMethod`` is now a plain
        class whose ``run`` still calls ``run_aldvc`` verbatim.
      * The ``ParamSpec`` import (only ``get_params`` needed it) — no ParamSpec
        stub was required because ``DVCResult``/``DVCParams`` do not use it.
No compute-path code was altered. No private helper needed renaming (there were
no cross-module name collisions).
"""
from __future__ import annotations


# ==== vendored from nd2studios/compute/parallel/shared_array.py ====
"""``multiprocessing.shared_memory`` glue for large numpy arrays.

Addendum Phase 5, Pattern 2. The default :class:`ProcessPoolExecutor`
behavior is to pickle every argument and reconstruct it inside the
worker, which for a 100 MB ND2 channel stack costs hundreds of ms per
task — enough to erase the parallel speedup. ``shared_memory`` lets
the parent allocate one OS-backed buffer, copy the array in once, and
hand workers a tiny ``(name, shape, dtype)`` triple they can attach
to lock-free.

Two surfaces:

- :func:`shared_ndarray` — context manager used by the parent. Allocates,
  copies, yields ``(name, shape, dtype_str)`` for transport into
  worker tasks, then cleans the block up on exit (``close`` + ``unlink``).
- :func:`attach_shared` — used inside the worker. Returns the
  ``ndarray`` view plus the underlying handle so the worker can
  ``close()`` it on exit (workers must **not** ``unlink``; that's the
  parent's job).
"""

from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Iterator, Tuple

import numpy as np


@contextmanager
def shared_ndarray(arr: np.ndarray) -> Iterator[Tuple[str, Tuple[int, ...], str]]:
    """Expose *arr* to worker processes via shared memory.

    Yields ``(name, shape, dtype_str)``. Worker functions reconstruct
    the array with :func:`attach_shared`. On exit the block is closed
    and unlinked so the OS frees the backing memory.

    Use as::

        with shared_ndarray(stack) as (name, shape, dtype_str):
            args = [(name, shape, dtype_str, t, params) for t in range(T)]
            with ProcessPoolExecutor(...) as ex:
                results = list(ex.map(worker_fn, args))

    The yielded ``shape`` is a tuple and ``dtype_str`` is a string
    (e.g. ``'float32'``) so workers can rebuild ``np.dtype`` without
    needing the parent's exact ``np.dtype`` instance.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.ascontiguousarray(arr)
    if not arr.flags.c_contiguous:
        # ``shared_memory`` wants a flat byte image. Materialize a
        # contiguous copy when the caller passed in a strided view.
        arr = np.ascontiguousarray(arr)

    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    try:
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[:] = arr
        yield shm.name, tuple(arr.shape), str(arr.dtype)
    finally:
        try:
            shm.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            shm.unlink()
        except FileNotFoundError:
            # Already unlinked — another part of the parent or a worker
            # race lost; ignore.
            pass
        except Exception:  # noqa: BLE001
            pass


def attach_shared(
    name: str,
    shape: Tuple[int, ...],
    dtype_str: str,
) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
    """Attach to an existing shared block from inside a worker.

    Returns ``(view, shm_handle)``. The caller **must** keep
    ``shm_handle`` alive while using ``view`` and call
    ``shm_handle.close()`` when done — failing to close leaks a file
    descriptor per task on POSIX. ``unlink`` is the parent's job
    (handled by :func:`shared_ndarray`'s context manager); calling it
    in a worker will tear the block out from under sibling workers.
    """
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return arr, shm


# ==== vendored from nd2studios/core/dvc_registry.py ====
"""
DVC Method Registry for ND2Studios.

Digital Volume/Image Correlation (DVC/DIC) methods measure a dense
displacement field — and the strain tensor derived from it — between a
*reference* volume and a *deformed* volume. This is distinct from both
enhancement plugins (``(T,H,W) -> (T,H,W)`` recipe steps) and analysis
pipelines (``AnalysisResult`` = label masks + per-object measurements):
DVC consumes full ``(Z,H,W)`` volumes (or ``(H,W)`` images in the 2D
DIC case) and produces a dense vector/tensor field that neither of the
other two result types can represent.

So DVC gets its own registry, modeled byte-for-byte on
:class:`~nd2studios.core.analysis_registry.AnalysisPipeline`'s idiom and
reusing :class:`~nd2studios.core.plugin_registry.ParamSpec`, with its own
:class:`DVCResult` container.

Usage::

    class ALDVCMethod(DVCMethod):
        name = "ALDVC"
        description = "Augmented Lagrangian Digital Volume Correlation"
        def get_params(self): ...
        def run(self, ref_vol, def_vol, voxel_size_um, params,
                progress_cb, cancelled_cb): ...

This module imports only the standard library + numpy + ``ParamSpec``; it
deliberately has **no PySide6 dependency** so the backend engine can call
it (and run headless) without pulling in Qt.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np



@dataclass
class DVCResult:
    """Structured output from a :meth:`DVCMethod.run` call.

    Shapes (``d`` = 2 for 2D DIC, 3 for 3D DVC):

    - ``grid_coords``: ``(*grid, d)`` subset-center coordinates **in voxels**,
      where ``grid`` is ``(Gy, Gx)`` (2D) or ``(Gz, Gy, Gx)`` (3D).
    - ``displacement_field``: ``(*grid, d)`` displacement **in voxels**, axis
      order matching ``grid_coords`` (i.e. ``[..., 0]`` is the slowest spatial
      axis: y in 2D, z in 3D).
    - ``strain_field``: ``(*grid, n_components)`` or ``None`` until computed.

    ``voxel_size_um`` is ``(y, x)`` (2D) or ``(z, y, x)`` (3D); use
    :meth:`displacement_um` to convert the displacement field to micrometers.
    """
    dim: int
    grid_coords: np.ndarray
    displacement_field: np.ndarray
    voxel_size_um: Tuple[float, ...] = ()

    strain_field: Optional[np.ndarray] = None
    strain_type: str = ""

    qfactor: Optional[np.ndarray] = None        # (*grid,) correlation confidence
    converged: bool = False
    iterations: int = 0
    mu: float = 0.0
    beta: float = 0.0

    method: str = ""
    notes: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ── convenience accessors ──
    @property
    def magnitude(self) -> np.ndarray:
        """``(*grid,)`` displacement magnitude in voxels."""
        return np.sqrt(np.sum(np.square(self.displacement_field), axis=-1))

    def displacement_um(self) -> np.ndarray:
        """Displacement field converted to micrometers (per-axis scaling)."""
        if not self.voxel_size_um or len(self.voxel_size_um) != self.dim:
            return self.displacement_field
        scale = np.asarray(self.voxel_size_um, dtype=np.float64)
        return self.displacement_field * scale

    def magnitude_um(self) -> np.ndarray:
        return np.sqrt(np.sum(np.square(self.displacement_um()), axis=-1))


@dataclass
class DVCParams:
    """Convenience typed view over the param dict a DVCMethod receives.

    Kept thin (and optional) on purpose — methods may continue to read the
    raw ``params`` dict produced by ``ParamEditor.get_values()``. This exists
    so callers building params programmatically (headless scripts, batch) get
    sane defaults and IDE help.
    """
    subset_size: int = 16
    subset_spacing: int = 10
    correlation: str = "zncc"          # "zncc" | "phase"
    tracking_mode: str = "cumulative"  # "cumulative" | "incremental"
    strain_type: str = "infinitesimal"  # "infinitesimal" | "green-lagrange"
    search_radius: int = 0             # 0 → auto (= subset_size)
    seed_levels: int = 3               # multigrid FFT-seed pyramid levels (1 = single)
    admm_iterations: int = 4
    mu: float = 1e-3
    strain_smooth: float = 0.0         # Gaussian σ on û before ∂u/∂x (0 = off)
    n_workers: int = 0                 # IC-GN process-pool workers (0 = auto)
    newFFTSearch: bool = False         # re-seed every frame vs cross-frame warm-start
    use_gpu: bool = False              # route the FFT seed through CuPy if present

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DVCParams":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# ==== vendored from nd2studios/backend/dvc/mesh.py ====
"""
DVC mesh + DOF bookkeeping — the single source of truth for coordinate and
degree-of-freedom conventions in the ALDVC port.

Why this module is isolated
---------------------------
The single largest correctness risk when porting FranckLab's MATLAB ALDVC is
index/order scrambling (1-based vs 0-based, column- vs row-major flattening,
``ndgrid`` axis order, the swapped ``(v, u)`` interpolation argument order). We
sidestep the MATLAB conventions entirely and adopt clean, internally-consistent
**numpy** conventions, centralized here so every other DVC module inherits them:

* **Axis order == array axis order.** A 2D image is ``(y, x)``; a 3D volume is
  ``(z, y, x)``. Grid coordinates and displacement components use the *same*
  slowest-first order — component ``0`` is ``z`` in 3D / ``y`` in 2D. This is the
  order the ND2Studios executor's ``get_frame(z_mode="none")`` yields and the
  order :mod:`nd2studios.backend.serialtrack.fields` already uses
  (``F[i, j] = ∂u_i/∂x_j``).
* **Regular grid of subset centers**, spaced ``subset_spacing`` voxels apart and
  inset by ``subset_size // 2`` from every border so each subset window fits.
* **Flat DOF layout matches** :func:`nd2studios.backend.serialtrack.regularization._build_gradient_operator`
  so we can reuse its sparse finite-difference operator ``D`` verbatim in the
  global step:

  - displacement vector ``u_vec``: ``u_vec[ndim*p + c] = u[c]`` at grid node ``p``
    (``p`` is the C-order linear index of the grid node, ``c`` the component).
  - deformation-gradient vector ``F_vec``:
    ``F_vec[ndim² * p + (j*ndim + i)] = ∂u_i/∂x_j`` — i.e. component ``i`` varies
    fastest within a node, then derivative axis ``j``. This is exactly the layout
    ``_build_gradient_operator`` produces, so ``D @ u_vec ≈ F_vec``.

Pure numpy — no PySide6 (backend-purity rule).
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Grid:
    """A regular grid of subset centers over an image/volume.

    Attributes
    ----------
    axes : list of ``ndim`` 1-D arrays
        Per-axis center coordinates (in voxels), slowest axis first.
    coords : (*grid_shape, ndim) float64
        Center coordinate of every node, ``coords[..., c]`` the ``c``-th axis
        (``ij`` meshgrid → axis order matches ``grid_shape``).
    grid_shape : tuple[int, ...]
        Number of centers along each axis.
    step : (ndim,) float64
        Spacing between adjacent centers, per axis (voxels). Equals
        ``subset_spacing`` except on axes too small to hold >1 center.
    ndim : int
    """
    axes: List[np.ndarray]
    coords: np.ndarray
    grid_shape: Tuple[int, ...]
    step: np.ndarray
    ndim: int

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.grid_shape)) if self.grid_shape else 0

    def coords_flat(self) -> np.ndarray:
        """``(n_nodes, ndim)`` C-order flattened center coordinates."""
        return self.coords.reshape(-1, self.ndim)


def build_grid(shape: Tuple[int, ...], subset_size: int, subset_spacing: int) -> Grid:
    """Build a :class:`Grid` of subset centers for a volume of ``shape``.

    Centers are ``subset_spacing`` apart, inset by ``subset_size // 2`` so every
    subset window lies fully inside the volume. Tiny axes fall back to a single
    center at the axis midpoint.
    """
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    half = max(1, int(subset_size) // 2)
    step = max(1, int(subset_spacing))
    axes: List[np.ndarray] = []
    steps: List[float] = []
    for n in shape:
        half_a = min(half, max(0, (n - 1) // 2))   # can't inset past the axis
        start = half_a
        stop = max(start + 1, n - half_a)          # exclusive upper bound
        c = np.arange(start, stop, step, dtype=np.float64)
        if c.size == 0:
            c = np.asarray([n / 2.0], dtype=np.float64)
        elif c.size == 1 and (stop - 1) > start:
            # Force >=2 centers where the extent allows: the finite-difference
            # gradient operator + np.gradient (strain) are undefined on a size-1
            # axis, so a single grid node along an axis would break the global
            # solve. Two evenly-placed centers keep a thin (e.g. shallow-Z) grid
            # well-formed.
            c = np.round(np.linspace(start, stop - 1, 2)).astype(np.float64)
        axes.append(c)
        steps.append(float(np.median(np.diff(c))) if c.size > 1 else float(step))
    mesh = np.meshgrid(*axes, indexing="ij")
    coords = np.stack(mesh, axis=-1).astype(np.float64)
    return Grid(
        axes=axes,
        coords=coords,
        grid_shape=tuple(a.size for a in axes),
        step=np.asarray(steps, dtype=np.float64),
        ndim=ndim,
    )


# ─────────────────────────────────────────────────────────────────────────
#  DOF pack / unpack  (layout matches serialtrack _build_gradient_operator)
# ─────────────────────────────────────────────────────────────────────────

def pack_u(disp_grid: np.ndarray) -> np.ndarray:
    """``(ndim, *grid) → (ndim*n_nodes,)`` — ``u_vec[ndim*p + c] = disp_grid[c].flat[p]``."""
    ndim = disp_grid.shape[0]
    n = int(np.prod(disp_grid.shape[1:]))
    out = np.empty(ndim * n, dtype=np.float64)
    for c in range(ndim):
        out[c::ndim] = np.asarray(disp_grid[c], dtype=np.float64).ravel(order="C")
    return out


def unpack_u(u_vec: np.ndarray, grid_shape: Tuple[int, ...], ndim: int) -> np.ndarray:
    """``(ndim*n,) → (ndim, *grid)`` — inverse of :func:`pack_u`."""
    out = np.empty((ndim, *grid_shape), dtype=np.float64)
    for c in range(ndim):
        out[c] = u_vec[c::ndim].reshape(grid_shape)
    return out


def pack_F(F_grid: np.ndarray) -> np.ndarray:
    """``(ndim, ndim, *grid) → (ndim²*n,)``.

    ``F_grid[i, j]`` is ``∂u_i/∂x_j``; packed at per-node offset ``j*ndim + i``
    (component ``i`` fastest), matching ``_build_gradient_operator``'s ``F_vec``.
    """
    ndim = F_grid.shape[0]
    n = int(np.prod(F_grid.shape[2:]))
    out = np.empty(ndim * ndim * n, dtype=np.float64)
    for j in range(ndim):          # derivative axis
        for i in range(ndim):      # displacement component
            out[(j * ndim + i)::(ndim * ndim)] = \
                np.asarray(F_grid[i, j], dtype=np.float64).ravel(order="C")
    return out


def unpack_F(F_vec: np.ndarray, grid_shape: Tuple[int, ...], ndim: int) -> np.ndarray:
    """``(ndim²*n,) → (ndim, ndim, *grid)`` — inverse of :func:`pack_F` (``[i, j]``)."""
    out = np.empty((ndim, ndim, *grid_shape), dtype=np.float64)
    for j in range(ndim):
        for i in range(ndim):
            out[i, j] = F_vec[(j * ndim + i)::(ndim * ndim)].reshape(grid_shape)
    return out


# ==== vendored from nd2studios/backend/dvc/outliers.py ====
"""
Stage 2 — outlier detection + NaN inpainting for displacement fields.

Two guards, both standard in the DIC/DVC literature and in FranckLab's
``RemoveOutliers3`` / ``inpaint_nans3``:

* **cc threshold** — drop subsets whose correlation confidence is too low.
* **normalized median test** (Westerweel & Scarano, *Exp. Fluids* 2005) — drop
  vectors that disagree with their neighborhood median beyond a robust,
  self-scaling threshold.

Flagged nodes become NaN, then :func:`inpaint_nans` fills them (nearest-value via
Euclidean distance transform — always converges as long as one finite value
exists), so a handful of bad subsets never poison the field or the downstream
global solve. Grid axis order follows :mod:`nd2studios.backend.dvc.mesh`.

Pure numpy/scipy — no PySide6.
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter


def normalized_median_flags(
    u_grid: np.ndarray, *, eps: float = 0.1, threshold: float = 2.0,
    size: int = 3,
) -> np.ndarray:
    """Boolean mask of nodes failing the normalized median test (any component).

    ``u_grid`` is ``(ndim, *grid)``. ``eps`` is the noise floor (voxels) that
    keeps the test from firing on uniform fields; ``threshold`` ~2 is typical.
    """
    ndim = u_grid.shape[0]
    flags = np.zeros(u_grid.shape[1:], dtype=bool)
    for c in range(ndim):
        comp = np.asarray(u_grid[c], dtype=np.float64)
        finite = np.nan_to_num(comp, nan=0.0)
        med = median_filter(finite, size=size, mode="nearest")
        res = np.abs(finite - med)
        res_med = median_filter(res, size=size, mode="nearest")
        norm_res = res / (res_med + eps)
        flags |= norm_res > threshold
    return flags


def remove_outliers(
    u_grid: np.ndarray, cc: np.ndarray, *,
    cc_thresh: float = 0.5, median_thresh: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Set low-confidence / median-failing / non-finite nodes to NaN.

    Returns ``(u_clean, bad_mask)`` where ``u_clean`` is ``u_grid`` with flagged
    nodes NaN'd (a copy) and ``bad_mask`` is ``(*grid,)`` bool.
    """
    ndim = u_grid.shape[0]
    bad = ~np.all(np.isfinite(u_grid), axis=0)
    if cc is not None:
        bad |= np.asarray(cc) < float(cc_thresh)
    bad |= normalized_median_flags(u_grid, threshold=median_thresh)
    out = np.array(u_grid, dtype=np.float64, copy=True)
    out[:, bad] = np.nan
    return out, bad


def inpaint_nans(field: np.ndarray) -> np.ndarray:
    """Fill NaNs in a scalar ``(*grid,)`` array by nearest finite value.

    Uses the Euclidean distance transform to index the nearest valid sample —
    guaranteed to fill every NaN provided at least one finite value exists.
    """
    arr = np.asarray(field, dtype=np.float64)
    nan_mask = ~np.isfinite(arr)
    if not nan_mask.any():
        return arr
    if nan_mask.all():
        return np.zeros_like(arr)
    idx = distance_transform_edt(nan_mask, return_distances=False,
                                 return_indices=True)
    return arr[tuple(idx)]


def inpaint_vector(u_grid: np.ndarray) -> np.ndarray:
    """Apply :func:`inpaint_nans` to each component of a ``(ndim, *grid)`` field."""
    out = np.array(u_grid, dtype=np.float64, copy=True)
    for c in range(out.shape[0]):
        out[c] = inpaint_nans(out[c])
    return out


# ==== vendored from nd2studios/backend/dvc/strain.py ====
"""
Stage 7 — strain tensor from the compatible displacement field.

Builds the displacement gradient ``G = ∇û`` (finite differences on the subset
grid), the deformation gradient ``F = I + G``, and one of four strain measures.
Mirrors FranckLab ``ComputeStrain3`` and parallels
:meth:`nd2studios.backend.serialtrack.fields.DisplacementField.gradient`
(``G[i, j] = ∂u_i/∂x_j``, mesh axis order).

Physical (anisotropic-voxel) scaling: converting both displacement components and
spatial axes to micrometers rescales each gradient entry by
``voxel_i / voxel_j`` — the cross-axis rescaling required for non-cubic voxels
(confocal ``z`` step ≠ ``xy`` pixel).

Pure numpy/scipy — no PySide6.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def displacement_gradient(
    u_grid: np.ndarray, grid_step: np.ndarray,
    voxel_size: Optional[np.ndarray] = None, smooth_sigma: float = 0.0,
) -> np.ndarray:
    """``(ndim, *grid) → (ndim, ndim, *grid)`` displacement gradient ``∂u_i/∂x_j``.

    ``grid_step`` is the node spacing (voxels) per axis. If ``voxel_size`` is
    given, the result is in physical (dimensionless-strain) units with the
    ``voxel_i/voxel_j`` cross-axis rescaling applied.
    """
    ndim = u_grid.shape[0]
    step = np.asarray(grid_step, dtype=np.float64)
    u = np.asarray(u_grid, dtype=np.float64)
    if smooth_sigma and smooth_sigma > 0:
        u = np.stack([gaussian_filter(u[i], sigma=float(smooth_sigma))
                      for i in range(ndim)])
    G = np.zeros((ndim, ndim, *u.shape[1:]), dtype=np.float64)
    gshape = u.shape[1:]
    for i in range(ndim):
        for j in range(ndim):
            # np.gradient needs >=2 samples along the axis; a singleton axis
            # (e.g. a 1-node-deep Z grid) contributes zero gradient there.
            if gshape[j] < 2:
                continue
            G[i, j] = np.gradient(u[i], float(step[j]), axis=j)
    if voxel_size is not None:
        v = np.asarray(voxel_size, dtype=np.float64)
        for i in range(ndim):
            for j in range(ndim):
                G[i, j] *= v[i] / v[j]
    return G


def strain_from_gradient(G: np.ndarray, strain_type: str = "infinitesimal") -> np.ndarray:
    """Strain tensor ``(ndim, ndim, *grid)`` from the displacement gradient ``G``.

    ``strain_type`` ∈ {infinitesimal, green-lagrange, almansi, hencky}.
    """
    ndim = G.shape[0]

    def _t(A):  # transpose the two tensor axes, keep grid axes
        return A.transpose(1, 0, *range(2, A.ndim))

    st = str(strain_type).lower().replace("_", "-")
    if st in ("infinitesimal", "small", "engineering"):
        return 0.5 * (G + _t(G))

    eye = np.eye(ndim).reshape(ndim, ndim, *([1] * (G.ndim - 2)))
    F = G + eye                                       # deformation gradient
    if st in ("green-lagrange", "green", "lagrange"):
        FtF = np.einsum("ki...,kj...->ij...", F, F)   # Fᵀ F
        return 0.5 * (FtF - eye)
    if st in ("almansi", "euler-almansi", "eulerian-almansi"):
        Finv = _inv_tensor_field(F)
        FiTFi = np.einsum("ki...,kj...->ij...", Finv, Finv)   # F⁻ᵀ F⁻¹
        return 0.5 * (eye - FiTFi)
    if st in ("hencky", "log", "logarithmic"):
        return _hencky(F)
    raise ValueError(f"unknown strain_type: {strain_type!r}")


def _inv_tensor_field(F: np.ndarray) -> np.ndarray:
    moved = np.moveaxis(F, (0, 1), (-2, -1))
    inv = np.linalg.inv(moved)
    return np.moveaxis(inv, (-2, -1), (0, 1))


def _hencky(F: np.ndarray) -> np.ndarray:
    """Hencky (logarithmic) strain ``½ ln(FᵀF)`` via eigen-decomposition."""
    ndim = F.shape[0]
    C = np.einsum("ki...,kj...->ij...", F, F)         # right Cauchy-Green
    Cm = np.moveaxis(C, (0, 1), (-2, -1))
    w, V = np.linalg.eigh(Cm)
    w = np.clip(w, 1e-12, None)
    logw = 0.5 * np.log(w)
    E = np.einsum("...ik,...k,...jk->...ij", V, logw, V)
    return np.moveaxis(E, (-2, -1), (0, 1))


def compute_strain(
    u_grid: np.ndarray, grid_step: np.ndarray,
    voxel_size: Optional[np.ndarray] = None, *,
    strain_type: str = "infinitesimal", smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(F_def, strain)`` — deformation gradient ``I+∇u`` and the strain
    tensor of the requested measure, both ``(ndim, ndim, *grid)``."""
    G = displacement_gradient(u_grid, grid_step, voxel_size, smooth_sigma)
    ndim = G.shape[0]
    eye = np.eye(ndim).reshape(ndim, ndim, *([1] * (G.ndim - 2)))
    strain = strain_from_gradient(G, strain_type)
    return G + eye, strain


# ==== vendored from nd2studios/backend/dvc/integer_search.py ====
"""
Stage 1 — integer displacement seed via windowed FFT cross-correlation.

Per-subset normalized cross-correlation (the FIDVC/DIC "bigxcorr" seed): for
each subset center we extract the reference subset window and correlate it,
via FFT, against a slightly larger search window in the deformed volume,
locating the best integer offset and refining it to sub-voxel by a separable
parabolic peak fit. The peak correlation value doubles as the ``cc`` /
q-factor confidence that :mod:`nd2studios.backend.dvc.outliers` thresholds on.

This seed is deliberately coarse — it only has to land the subsequent IC-GN
(:mod:`nd2studios.backend.dvc.icgn`) inside its convergence basin. Works for 2D
images and 3D volumes; axis order follows :mod:`nd2studios.backend.dvc.mesh`
(``(y, x)`` / ``(z, y, x)``, slowest first).

Pure numpy/scipy/scikit-image — no PySide6.
"""

import importlib.util
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.registration import phase_cross_correlation



def _get_backends(use_gpu: bool):
    """Return ``(xp, signal_module, on_device)``.

    ``(cupy, cupyx.scipy.signal, True)`` when ``use_gpu`` and CuPy is importable;
    otherwise ``(numpy, scipy.signal, False)``. The same :func:`_ncc_fft` runs on
    either backend, so the GPU path keeps the deformed/reference volumes resident
    on the device and does the per-subset FFT cross-correlation there.
    """
    if use_gpu:
        try:
            if importlib.util.find_spec("cupy") is not None:
                import cupy as cp
                import cupyx.scipy.signal as csignal
                return cp, csignal, True
        except Exception:  # noqa: BLE001 — any CuPy import/init issue → CPU
            pass
    import scipy.signal as ssignal
    return np, ssignal, False


def _to_host(a) -> np.ndarray:
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(a)


def _ncc_fft(search, template, xp, signal):
    """FFT normalized cross-correlation of ``template`` within ``search`` (valid
    positions) — the Lewis (1995) NCC that ``skimage.feature.match_template``
    computes, written against a backend module (numpy or cupy) so it runs on CPU
    or GPU. Returns the NCC map (backend array) or ``None`` for a flat template.
    """
    t0 = template - template.mean()
    tnorm = float(xp.sqrt(xp.sum(t0 * t0)))
    if tnorm < 1e-12:
        return None
    n = float(template.size)
    rev = tuple(slice(None, None, -1) for _ in range(template.ndim))
    ones = xp.ones_like(template)
    num = signal.fftconvolve(search, t0[rev], mode="valid")           # Σ I·T0
    sum_i = signal.fftconvolve(search, ones[rev], mode="valid")       # Σ I
    sum_i2 = signal.fftconvolve(search * search, ones[rev], mode="valid")  # Σ I²
    var = xp.clip(sum_i2 - (sum_i * sum_i) / n, 0.0, None)
    denom = tnorm * xp.sqrt(var)
    return xp.where(denom > 1e-12, num / denom, 0.0)


def _parabolic_subpixel(corr: np.ndarray, peak: Tuple[int, ...]) -> np.ndarray:
    """Separable 3-point parabolic peak refinement.

    Returns a per-axis fractional offset in ``[-0.5, 0.5]`` added to the integer
    ``peak`` index. Falls back to 0 on any axis where the peak is on a border or
    the curvature is degenerate.
    """
    ndim = corr.ndim
    delta = np.zeros(ndim, dtype=np.float64)
    for ax in range(ndim):
        i = peak[ax]
        if i <= 0 or i >= corr.shape[ax] - 1:
            continue
        sl_m = list(peak); sl_m[ax] = i - 1
        sl_p = list(peak); sl_p[ax] = i + 1
        cm = float(corr[tuple(sl_m)])
        c0 = float(corr[tuple(peak)])
        cp = float(corr[tuple(sl_p)])
        denom = (cm - 2.0 * c0 + cp)
        if abs(denom) > 1e-12:
            d = 0.5 * (cm - cp) / denom
            if -1.0 < d < 1.0:
                delta[ax] = d
    return delta


def _clamp_window(ctr: np.ndarray, half: np.ndarray, shape: Tuple[int, ...]):
    """Return ``(lo, hi)`` integer slice bounds for a window of half-width
    ``half`` around integer center ``ctr``, clamped to ``[0, shape)``."""
    lo = np.maximum(ctr - half, 0)
    hi = np.minimum(ctr + half + 1, np.asarray(shape))
    return lo.astype(np.int64), hi.astype(np.int64)


def integer_search(
    ref: np.ndarray,
    defm: np.ndarray,
    grid: Grid,
    subset_size: int,
    search_radius: int,
    *,
    correlation: str = "zncc",
    use_gpu: bool = False,
    u0_center: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    progress_lo: int = 0,
    progress_hi: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integer + sub-voxel displacement seed on the subset grid.

    Parameters
    ----------
    ref, defm : normalized (float) volumes of identical shape (2D or 3D).
    grid : the subset-center grid from :func:`mesh.build_grid`.
    subset_size : edge length of the correlation window (voxels).
    search_radius : how far (voxels) the deformed match may sit from the
        (offset) reference position — bounds the *residual* seed displacement.
    correlation : ``"zncc"`` (masked FFT normalized cross-correlation) or
        ``"phase"`` (phase cross-correlation).
    u0_center : optional ``(ndim, *grid)`` per-subset displacement guess. The
        deformed search window is centered at ``ctr + round(u0_center)`` and the
        returned displacement carries that offset — the multigrid refinement hook.

    Returns
    -------
    u0 : (ndim, *grid_shape) sub-voxel displacement seed (voxels, mesh axis order).
    cc : (*grid_shape) peak correlation confidence in ``[-1, 1]`` (ZNCC) — the
        q-factor proxy; ``0`` marks a subset that could not be correlated.
    """
    ndim = grid.ndim
    half = np.full(ndim, max(1, int(subset_size) // 2), dtype=np.int64)
    rad = np.full(ndim, max(1, int(search_radius)), dtype=np.int64)
    gshape = grid.grid_shape
    u0 = np.zeros((ndim, *gshape), dtype=np.float64)
    cc = np.zeros(gshape, dtype=np.float64)

    ref = np.asarray(ref, dtype=np.float32)
    defm = np.asarray(defm, dtype=np.float32)
    # Backend: CPU numpy/scipy, or CuPy (volumes resident on the device for the
    # per-subset FFT NCC) when use_gpu and CuPy is present. Same _ncc_fft either way.
    xp, signal, on_dev = _get_backends(use_gpu)
    ref_x = xp.asarray(ref) if on_dev else ref
    defm_x = xp.asarray(defm) if on_dev else defm
    flat_idx = list(np.ndindex(*gshape))
    n = len(flat_idx)
    for k, idx in enumerate(flat_idx):
        if cancelled_cb is not None and (k & 63) == 0 and cancelled_cb():
            raise InterruptedError("DVC cancelled")
        ctr = np.rint(grid.coords[idx]).astype(np.int64)   # (ndim,) center voxel
        off = (np.rint(u0_center[(slice(None), *idx)]).astype(np.int64)
               if u0_center is not None else np.zeros(ndim, dtype=np.int64))
        dctr = ctr + off                                    # deformed-window center
        rlo, rhi = _clamp_window(ctr, half, ref.shape)
        rsl = tuple(slice(int(a), int(b)) for a, b in zip(rlo, rhi))
        template = ref[rsl]
        if template.size == 0 or float(template.std()) < 1e-6:
            continue                                        # flat/empty → no info

        if correlation == "phase":
            # Compare equal-size windows (ref @ ctr vs deformed @ ctr+off).
            dlo, dhi = _clamp_window(dctr, half, defm.shape)
            dsl = tuple(slice(int(a), int(b)) for a, b in zip(dlo, dhi))
            moving = defm[dsl]
            if moving.shape != template.shape or float(moving.std()) < 1e-6:
                continue
            try:
                shift, _err, _phase = phase_cross_correlation(
                    template, moving, upsample_factor=10, normalization=None)
                # residual shift maps moving→ref; ref→deformed disp = off − shift.
                u0[(slice(None), *idx)] = (off.astype(np.float64)
                                           - np.asarray(shift, dtype=np.float64))
                cc[idx] = 1.0
            except Exception:                               # noqa: BLE001
                continue
            _emit(progress_cb, progress_lo, progress_hi, k, n)
            continue

        # ZNCC: FFT normalized cross-correlation of the template within an
        # expanded deformed search window (on the GPU when on_dev), centered at
        # ctr+off so the returned displacement includes the coarse-level offset.
        slo, shi = _clamp_window(dctr, half + rad, defm.shape)
        ssl = tuple(slice(int(a), int(b)) for a, b in zip(slo, shi))
        if any((int(shi[d]) - int(slo[d])) < template.shape[d]
               for d in range(ndim)):
            continue
        corr_x = _ncc_fft(defm_x[ssl], ref_x[rsl], xp, signal)
        if corr_x is None:
            continue
        corr = _to_host(corr_x)
        if not np.isfinite(corr).any():
            continue
        peak = np.unravel_index(int(np.nanargmax(corr)), corr.shape)
        # Best-match origin of the template inside defm = slo + peak (+subvoxel).
        sub = _parabolic_subpixel(corr, peak)
        match_origin = slo.astype(np.float64) + np.asarray(peak, np.float64) + sub
        u0[(slice(None), *idx)] = match_origin - rlo.astype(np.float64)
        cc[idx] = float(corr[peak])
        _emit(progress_cb, progress_lo, progress_hi, k, n)

    return u0, cc


def _emit(cb: Optional[Callable[[int], None]], lo: int, hi: int, k: int, n: int) -> None:
    if cb is not None and n > 0 and (k & 63) == 0:
        cb(int(lo + (hi - lo) * k / n))


# ─────────────────────────────────────────────────────────────────────────
#  Multigrid (coarse-to-fine) seed  (ALDVC IntegerSearch3Multigrid)
# ─────────────────────────────────────────────────────────────────────────

def _downsample_facs(shape, f: int, subset_size: int):
    """Per-axis downsample factor: ``f`` where the axis stays usable after, else 1
    (keeps a thin Z axis from being crushed)."""
    return tuple(int(f) if (s // f) >= max(6, subset_size // 2) and s >= f * 4 else 1
                 for s in shape)


def _downsample(vol: np.ndarray, facs) -> np.ndarray:
    """Block-mean downsample per axis (crops to a multiple of the factor)."""
    a = np.asarray(vol, dtype=np.float32)
    sl = tuple(slice(0, s - (s % f)) for s, f in zip(a.shape, facs))
    a = a[sl]
    newshape = []
    for s, f in zip(a.shape, facs):
        newshape += [s // f, f]
    a = a.reshape(newshape)
    return a.mean(axis=tuple(range(1, a.ndim, 2)))


def _interp_u_to_grid(src_grid: Grid, u_src: np.ndarray, dst_grid: Grid,
                      facs) -> np.ndarray:
    """Interpolate a coarse-image displacement field ``u_src`` (``(ndim,*src)``, in
    coarse-image voxels) onto ``dst_grid`` at full resolution — evaluate at the
    dst node coords mapped into the coarse image (``/facs``) and scale each
    component back up (``×facs``)."""
    ndim = src_grid.ndim
    facs = np.asarray(facs, dtype=np.float64)
    pts = dst_grid.coords_flat() / facs
    out = np.zeros((ndim, dst_grid.n_nodes), dtype=np.float64)
    for c in range(ndim):
        interp = RegularGridInterpolator(
            [np.asarray(a, float) for a in src_grid.axes], u_src[c],
            method="linear", bounds_error=False, fill_value=None)
        out[c] = np.nan_to_num(interp(pts)) * facs[c]
    return out.reshape(ndim, *dst_grid.grid_shape)


def integer_search_multigrid(
    ref: np.ndarray, defm: np.ndarray, grid: Grid, subset_size: int,
    search_radius: int, *, levels: int = 3, correlation: str = "zncc",
    use_gpu: bool = False,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    progress_lo: int = 0, progress_hi: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coarse-to-fine integer seed (ALDVC ``IntegerSearch3Multigrid``).

    Seeds on a ``2^(levels-1)``-downsampled volume with a large (cheap)
    search radius to bracket **large** displacement, upsamples the estimate onto
    the full grid as a per-subset ``u0_center``, then refines at full resolution
    with a small residual radius. ``levels=1`` is the single-scale path.
    """
    levels = max(1, int(levels))
    if levels == 1:
        return integer_search(
            ref, defm, grid, subset_size, search_radius, correlation=correlation,
            use_gpu=use_gpu, progress_cb=progress_cb, cancelled_cb=cancelled_cb,
            progress_lo=progress_lo, progress_hi=progress_hi)
    f = 2 ** (levels - 1)
    facs = _downsample_facs(ref.shape, f, subset_size)
    if all(x == 1 for x in facs):
        return integer_search(
            ref, defm, grid, subset_size, search_radius, correlation=correlation,
            use_gpu=use_gpu, progress_cb=progress_cb, cancelled_cb=cancelled_cb,
            progress_lo=progress_lo, progress_hi=progress_hi)
    ref_c = _downsample(ref, facs)
    defm_c = _downsample(defm, facs)
    fmean = float(np.mean([x for x in facs if x > 1])) or 1.0
    coarse_spacing = max(2, int(round(float(np.mean(grid.step)) / fmean)))
    coarse_grid = build_grid(ref_c.shape, subset_size, coarse_spacing)
    coarse_rad = int(max(np.ceil(search_radius / fmean),
                         0.3 * min(ref_c.shape)))
    mid = int(progress_lo + 0.4 * (progress_hi - progress_lo))
    u0_c, _cc_c = integer_search(
        ref_c, defm_c, coarse_grid, subset_size, coarse_rad,
        correlation=correlation, use_gpu=use_gpu, progress_cb=progress_cb,
        cancelled_cb=cancelled_cb, progress_lo=progress_lo, progress_hi=mid)
    u0_center = _interp_u_to_grid(coarse_grid, u0_c, grid, facs)
    fine_rad = max(4, int(subset_size) // 2)
    return integer_search(
        ref, defm, grid, subset_size, fine_rad, correlation=correlation,
        use_gpu=use_gpu, u0_center=u0_center, progress_cb=progress_cb,
        cancelled_cb=cancelled_cb, progress_lo=mid, progress_hi=progress_hi)


# ==== vendored from nd2studios/backend/dvc/global_step.py ====
"""
Stage 4 — global augmented-Lagrangian compatibility solve (ALDVC Subpb2, FD).

The local IC-GN field ``(u, F)`` is noisy and kinematically incompatible
(``F`` need not equal ``∇u``). This step projects it onto a smooth, compatible
displacement ``û`` (with ``F̂ = ∇û``) by solving the augmented-Lagrangian normal
equations

    (β·DᵀD + μ·I) û = β·Dᵀ(F − w_F) + μ·(u − w_u)

where ``D`` is the sparse finite-difference gradient operator and ``w_u`` / ``w_F``
are the ADMM scaled duals. This is exactly the FD variant of FranckLab's Subpb2.

**Operator ``D``** (:func:`_build_fd_operator`) is a hardened re-derivation of
SerialTrack's ``funDerivativeOp3`` port (``regularization._build_gradient_operator``)
— same finite-difference stencil and DOF layout (matching
:mod:`nd2studios.backend.dvc.mesh` ``pack_u`` / ``pack_F``), but it **guards
singleton grid axes** so a thin (shallow-Z) DVC grid can't overflow the operator
(SerialTrack's version assumes ≥2 nodes per axis). SerialTrack's own ``ADMMLSolver``
solves the ``μ``-only variant ``(α·DᵀD + I)û = u − v``; DVC adds the **F-coupling**
term ``β·Dᵀ(F − w_F)`` that carries the local deformation gradient into the solve.

Robustness over the MATLAB reference (which runs a near-singular solve): a small
Tikhonov term and a guarded ``poly2`` L-curve fit. The matrix is constant across
the β-sweep and the ADMM iterations (only the RHS changes), so its factorization
is cached.

Pure numpy/scipy — no PySide6.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csc_matrix, eye as speye
from scipy.sparse.linalg import factorized



def _build_fd_operator(grid_shape, grid_step, ndim: int) -> csc_matrix:
    """Sparse finite-difference gradient operator ``D`` with ``F_vec = D @ u_vec``.

    Same DOF layout as :func:`mesh.pack_u` / :func:`mesh.pack_F` — ``u`` at
    ``ndim*p + comp`` and ``∂u_comp/∂x_deriv`` at ``ndim²*p + deriv*ndim + comp``
    (``p`` C-order). Central differences interior, one-sided at borders. This is a
    hardened re-derivation of SerialTrack's ``funDerivativeOp3`` port that
    additionally **guards singleton axes** (a grid axis with <2 nodes contributes
    a zero derivative instead of indexing an out-of-range neighbour), so thin
    (e.g. shallow-Z) DVC grids never overflow the operator.
    """
    grid_shape = tuple(int(s) for s in grid_shape)
    n = int(np.prod(grid_shape)) if grid_shape else 0
    n_u = ndim * n
    n_f = ndim * ndim * n
    # C-order strides.
    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * grid_shape[d + 1]

    def _multi(idx):
        mi = [0] * ndim
        for d in range(ndim - 1, -1, -1):
            mi[d] = idx % grid_shape[d]
            idx //= grid_shape[d]
        return mi

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for deriv in range(ndim):
        h = float(grid_step[deriv]) or 1.0
        stride = strides[deriv]
        sz = grid_shape[deriv]
        for comp in range(ndim):
            f_off = deriv * ndim + comp
            for p in range(n):
                f_row = ndim * ndim * p + f_off
                if sz < 2:
                    continue                    # singleton axis → zero derivative
                idx_along = _multi(p)[deriv]
                if 0 < idx_along < sz - 1:       # central
                    rows += [f_row, f_row]
                    cols += [ndim * (p - stride) + comp, ndim * (p + stride) + comp]
                    vals += [-1.0 / (2 * h), 1.0 / (2 * h)]
                elif idx_along == 0:             # forward
                    rows += [f_row, f_row]
                    cols += [ndim * p + comp, ndim * (p + stride) + comp]
                    vals += [-1.0 / h, 1.0 / h]
                else:                            # backward
                    rows += [f_row, f_row]
                    cols += [ndim * (p - stride) + comp, ndim * p + comp]
                    vals += [-1.0 / h, 1.0 / h]
    return csc_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n_f, n_u))


class AugLagGlobalStep:
    """Cached augmented-Lagrangian global solver for one grid.

    Build once per (grid, β, μ); reuse :meth:`solve` across ADMM iterations
    (only the RHS — i.e. the duals — changes).
    """

    def __init__(self, grid: Grid, *, tikhonov: float = 1e-6):
        self.grid = grid
        self.ndim = grid.ndim
        self.n = grid.n_nodes
        self.tikhonov = float(tikhonov)
        self.D: csc_matrix = _build_fd_operator(
            grid.grid_shape, grid.step, grid.ndim)
        self.DtD: csc_matrix = (self.D.T @ self.D).tocsc()
        self._I = speye(self.ndim * self.n, format="csc")
        self._mu: Optional[float] = None
        self._beta: Optional[float] = None
        self._factor = None

    # ── matrix / factorization cache ────────────────────────────────────
    def _ensure_factor(self, mu: float, beta: float) -> None:
        if self._factor is not None and self._mu == mu and self._beta == beta:
            return
        A = (beta * self.DtD + (mu + self.tikhonov) * self._I).tocsc()
        self._factor = factorized(A)      # splu-backed solve(rhs)
        self._mu, self._beta = mu, beta

    def _rhs(self, u_vec, F_vec, wu_vec, wF_vec, mu, beta) -> np.ndarray:
        # Standard scaled-ADMM z-update RHS: μ(u + w_u) + β·Dᵀ(F + w_F).
        # Paired in admm.py with targets (ẑ − w) and dual update (w += x − z).
        return beta * (self.D.T @ (F_vec + wF_vec)) + mu * (u_vec + wu_vec)

    # ── public solve ────────────────────────────────────────────────────
    def solve(
        self, u_grid: np.ndarray, F_grid: np.ndarray,
        wu_grid: Optional[np.ndarray], wF_grid: Optional[np.ndarray],
        mu: float, beta: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(u_hat_grid (ndim,*grid), F_hat_grid (ndim,ndim,*grid))``.

        ``F_hat = ∇û`` (the compatible gradient, ``unpack_F(D @ û)``). Duals may be
        ``None`` (treated as zero).
        """
        u_vec = pack_u(u_grid)
        F_vec = pack_F(F_grid)
        wu_vec = pack_u(wu_grid) if wu_grid is not None else 0.0
        wF_vec = pack_F(wF_grid) if wF_grid is not None else 0.0
        self._ensure_factor(mu, beta)
        rhs = self._rhs(u_vec, F_vec, wu_vec, wF_vec, mu, beta)
        uhat = self._factor(rhs)
        Fhat = self.D @ uhat
        return (unpack_u(uhat, self.grid.grid_shape, self.ndim),
                unpack_F(Fhat, self.grid.grid_shape, self.ndim))

    # ── β selection (L-curve, MATLAB ErrSum criterion) ──────────────────
    def tune_beta(
        self, u_grid: np.ndarray, F_grid: np.ndarray, mu: float,
        beta_list: Optional[List[float]] = None,
    ) -> float:
        """Pick β by the FranckLab ``ErrSum = ‖u−û‖ + ‖F−∇û‖·mean(step)²`` L-curve,
        with a guarded parabolic refine. Duals are zero at selection time."""
        if beta_list is None:
            base = float(np.mean(self.grid.step) ** 2) * mu
            factors = np.array([np.sqrt(1e-5), 1e-2, np.sqrt(1e-3),
                                1e-1, np.sqrt(1e-1)])
            beta_list = list(np.maximum(factors * base, 1e-12))
        u_vec = pack_u(u_grid)
        F_vec = pack_F(F_grid)
        w2 = float(np.mean(self.grid.step) ** 2)
        errs = np.empty(len(beta_list))
        for i, b in enumerate(beta_list):
            self._ensure_factor(mu, float(b))
            uhat = self._factor(mu * u_vec)          # duals zero → rhs = μ·u
            fid = float(np.linalg.norm(u_vec - uhat))
            smooth = float(np.linalg.norm(F_vec - (self.D @ uhat)))
            errs[i] = fid + smooth * w2
        i0 = int(np.argmin(errs))
        beta = float(beta_list[i0])
        if 0 < i0 < len(beta_list) - 1:
            lb = np.log10(np.asarray(beta_list[i0 - 1:i0 + 2]))
            y = errs[i0 - 1:i0 + 2]
            try:
                p = np.polyfit(lb, y, 2)
                if abs(p[0]) > 1e-15:
                    cand = 10 ** (-p[1] / (2 * p[0]))
                    if beta_list[i0 - 1] <= cand <= beta_list[i0 + 1]:
                        beta = float(cand)
            except Exception:                         # noqa: BLE001
                pass
        # Reset the cached factor so the next solve rebuilds at the chosen β.
        self._factor = None
        self._mu = self._beta = None
        return beta


# ==== vendored from nd2studios/backend/dvc/icgn.py ====
"""
Stages 3 & 5 — local subset registration by Inverse-Compositional Gauss-Newton.

This is the numerical heart of DVC and the one piece SerialTrack has no analogue
for (SerialTrack links particle centroids; it never registers image subsets).
For each subset we refine the integer seed to a sub-voxel first-order (affine)
warp

    x'(Δx) = x0 + Δx + u + G·Δx          G[i, j] = ∂u_i/∂x_j

by IC-GN with a zero-normalized SSD (ZNSSD ≡ ZNCC) objective — robust to linear
brightness/contrast change. Following Baker–Matthews / Blaber (Ncorr):

* the steepest-descent images ``∇f · ∂W/∂p`` and the Hessian ``H = SDᵀSD`` are
  built **once per subset from the reference** and reused across iterations
  (the deformed image is never differentiated);
* the warp is updated by **inverse composition** ``W ← W ∘ ΔW⁻¹`` (a homogeneous
  ``(d+1)×(d+1)`` matrix), never additively;
* the deformed volume is **spline-prefiltered once** by the caller, so per-iter
  sampling is ``map_coordinates(order=3, prefilter=False)``.

An optional quadratic **penalty** pulls the warp toward an ADMM target
``(u→a, F→B)`` with weights ``(μ, β)`` — this is Subpb1 of ALDVC. With
``μ=β=0`` it is plain local (conventional) DVC.

Pure numpy/scipy — no PySide6.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates



@dataclass
class _RefSubset:
    """Cached reference-side quantities for one subset (constant across IC-GN
    iterations and across ADMM outer iterations)."""
    c0: np.ndarray          # (ndim,) integer subset center
    dx: np.ndarray          # (ndim, n_pix) local offsets from c0
    f0: np.ndarray          # (n_pix,) zero-mean reference intensities
    f_norm: float           # ||f - mean(f)||
    sd: np.ndarray          # (n_pix, n_params) steepest-descent images
    H_img: np.ndarray       # (n_params, n_params) image Hessian SDᵀSD
    ndim: int
    n_params: int


def _subset_offsets(subset_size: int, ndim: int) -> np.ndarray:
    """``(ndim, n_pix)`` integer local offsets spanning a centered subset."""
    half = max(1, int(subset_size) // 2)
    axis = np.arange(-half, half + 1, dtype=np.float64)
    mesh = np.meshgrid(*([axis] * ndim), indexing="ij")
    return np.stack([m.ravel(order="C") for m in mesh], axis=0)


def _prepare_reference(ref: np.ndarray, c0: np.ndarray, dx: np.ndarray,
                       subset_shape: Tuple[int, ...]) -> Optional[_RefSubset]:
    """Extract the reference window at ``c0`` and precompute SD images + Hessian.

    Parameter order is ``[u_0..u_{d-1}, G_00, G_01, ..., G_{d-1,d-1}]`` (row-major
    ``G``). SD column for ``u_i`` is ``∂f/∂x_i``; for ``G_ij`` it is
    ``(∂f/∂x_i)·Δx_j``.
    """
    ndim = c0.size
    half = np.asarray(subset_shape, dtype=np.int64) // 2
    lo = c0 - half
    hi = c0 + half + 1
    if np.any(lo < 0) or np.any(hi > np.asarray(ref.shape)):
        return None                                   # window off the edge
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    f = np.asarray(ref[sl], dtype=np.float64)
    if f.std() < 1e-8:
        return None                                   # featureless subset
    grads = np.gradient(f)                            # list[ndim] (ndim>1) or array
    if ndim == 1:
        grads = [grads]
    fg = [g.ravel(order="C") for g in grads]          # ∂f/∂x_k, each (n_pix,)
    f_flat = f.ravel(order="C")
    f0 = f_flat - f_flat.mean()
    f_norm = float(np.sqrt(np.sum(f0 * f0)))
    if f_norm < 1e-8:
        return None

    n_pix = f_flat.size
    n_params = ndim + ndim * ndim
    sd = np.empty((n_pix, n_params), dtype=np.float64)
    for i in range(ndim):                             # displacement params
        sd[:, i] = fg[i]
    col = ndim
    for i in range(ndim):                             # gradient params G_ij
        for j in range(ndim):
            sd[:, col] = fg[i] * dx[j]
            col += 1
    H_img = sd.T @ sd
    return _RefSubset(c0=c0, dx=dx, f0=f0, f_norm=f_norm, sd=sd,
                      H_img=H_img, ndim=ndim, n_params=n_params)


def _warp_matrix(u: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Homogeneous ``(d+1)×(d+1)`` warp ``[[I+G, u], [0, 1]]``."""
    ndim = u.size
    W = np.eye(ndim + 1, dtype=np.float64)
    W[:ndim, :ndim] = np.eye(ndim) + G
    W[:ndim, ndim] = u
    return W


def _icgn_subset(
    ref_sub: _RefSubset,
    defm_pref: np.ndarray,
    u_init: np.ndarray,
    G_init: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    mu: float = 0.0,
    beta: float = 0.0,
    u_target: Optional[np.ndarray] = None,
    F_target: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Run IC-GN for one subset. Returns ``(u, G, zncc, iters)``.

    ``u`` is the center displacement (voxels, mesh axis order); ``G`` the
    displacement gradient ``∂u_i/∂x_j``. ``zncc`` in ``[-1, 1]`` (NaN on failure).
    """
    ndim = ref_sub.ndim
    dx = ref_sub.dx                                    # (ndim, n_pix)
    c0 = ref_sub.c0.astype(np.float64)
    f0, f_norm, sd = ref_sub.f0, ref_sub.f_norm, ref_sub.sd

    # Penalty Hessian (diagonal): μ on the u params, β on the G params.
    pen_diag = np.zeros(ref_sub.n_params)
    if mu > 0.0:
        pen_diag[:ndim] = mu
    if beta > 0.0:
        pen_diag[ndim:] = beta
    B_vec = None
    if beta > 0.0 and F_target is not None:
        B_vec = np.asarray(F_target, dtype=np.float64).reshape(-1)   # row-major G

    W = _warp_matrix(np.asarray(u_init, float), np.asarray(G_init, float))
    shape_arr = np.asarray(defm_pref.shape)
    zncc = np.nan
    it = 0
    for it in range(1, int(max_iter) + 1):
        u = W[:ndim, ndim]
        G = W[:ndim, :ndim] - np.eye(ndim)
        # Warp reference-subset pixels into the deformed volume and sample.
        pos = c0[:, None] + dx + u[:, None] + G @ dx        # (ndim, n_pix)
        in_lo = np.all(pos >= 0, axis=0)
        in_hi = np.all(pos <= (shape_arr[:, None] - 1), axis=0)
        valid = in_lo & in_hi
        if valid.mean() < 0.8:
            return (np.full(ndim, np.nan), np.full((ndim, ndim), np.nan),
                    np.nan, it)
        g = map_coordinates(defm_pref, pos, order=3, prefilter=False,
                            mode="nearest").astype(np.float64)
        g0 = g - g.mean()
        g_norm = float(np.sqrt(np.sum(g0 * g0)))
        if g_norm < 1e-8:
            return (np.full(ndim, np.nan), np.full((ndim, ndim), np.nan),
                    np.nan, it)
        fn = f0 / f_norm
        gn = g0 / g_norm
        zncc = float(np.dot(fn, gn))
        evec = gn - fn                                       # (n_pix,)

        rhs = f_norm * (sd.T @ evec)
        H = ref_sub.H_img
        if pen_diag.any():
            H = H + np.diag(pen_diag)
            g_pen = np.zeros(ref_sub.n_params)
            if mu > 0.0 and u_target is not None:
                g_pen[:ndim] = mu * (np.asarray(u_target, float) - u)
            if beta > 0.0 and B_vec is not None:
                g_pen[ndim:] = beta * (B_vec - G.reshape(-1))
            rhs = rhs + g_pen
        try:
            dp = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            return (np.full(ndim, np.nan), np.full((ndim, ndim), np.nan),
                    np.nan, it)

        du = dp[:ndim]
        dG = dp[ndim:].reshape(ndim, ndim)
        dW = _warp_matrix(du, dG)
        try:
            W = W @ np.linalg.inv(dW)
        except np.linalg.LinAlgError:
            return (np.full(ndim, np.nan), np.full((ndim, ndim), np.nan),
                    np.nan, it)

        # Convergence: radius-weighted parameter step (Ncorr criterion).
        half = float(np.max(np.abs(dx))) or 1.0
        step = np.sqrt(np.sum(du ** 2) + (half ** 2) * np.sum(dG ** 2))
        if step < tol:
            break

    u = W[:ndim, ndim].copy()
    G = (W[:ndim, :ndim] - np.eye(ndim)).copy()
    return u, G, zncc, it


def local_icgn(
    ref: np.ndarray,
    defm_pref: np.ndarray,
    grid: Grid,
    u0: np.ndarray,
    subset_size: int,
    *,
    tol: float = 1e-2,
    max_iter: int = 100,
    mu: float = 0.0,
    beta: float = 0.0,
    u_target: Optional[np.ndarray] = None,
    F_target: Optional[np.ndarray] = None,
    ref_cache: Optional[dict] = None,
    n_workers: int = 1,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    progress_lo: int = 0,
    progress_hi: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """IC-GN over every subset on ``grid``.

    ``defm_pref`` must be the deformed volume **already** spline-prefiltered
    (``scipy.ndimage.spline_filter``). ``u0`` is the ``(ndim, *grid)`` seed.
    When ``mu``/``beta`` > 0, ``u_target`` / ``F_target`` (``(ndim, *grid)`` /
    ``(ndim, ndim, *grid)``) supply the per-node ADMM penalty targets (Subpb1).
    ``ref_cache`` (a dict keyed by node index) memoizes the per-subset reference
    SD/Hessian across ADMM outer iterations (serial path only).

    ``n_workers > 1`` fans the sweep across processes
    (:func:`nd2studios.backend.dvc.parallel.parallel_local_icgn`) instead of the
    cached serial loop below — the multicore win dominates the lost SD cache on
    large grids.

    Returns ``(u_grid, F_grid, zncc_grid, iters_grid)``; failed subsets are NaN.
    """
    if n_workers and int(n_workers) > 1:
        return parallel_local_icgn(
            ref, defm_pref, grid, u0, subset_size, tol=tol, max_iter=max_iter,
            mu=mu, beta=beta, u_target=u_target, F_target=F_target,
            n_workers=int(n_workers), progress_cb=progress_cb,
            cancelled_cb=cancelled_cb, progress_lo=progress_lo,
            progress_hi=progress_hi)
    ndim = grid.ndim
    gshape = grid.grid_shape
    u_out = np.full((ndim, *gshape), np.nan)
    F_out = np.full((ndim, ndim, *gshape), np.nan)
    zncc_out = np.full(gshape, np.nan)
    iters_out = np.zeros(gshape, dtype=np.int32)
    dx = _subset_offsets(subset_size, ndim)
    subset_shape = tuple([max(1, int(subset_size) // 2) * 2 + 1] * ndim)
    if ref_cache is None:
        ref_cache = {}

    idx_list = list(np.ndindex(*gshape))
    n = len(idx_list)
    for k, idx in enumerate(idx_list):
        if cancelled_cb is not None and (k & 31) == 0 and cancelled_cb():
            raise InterruptedError("DVC cancelled")
        seed = u0[(slice(None), *idx)]
        if not np.all(np.isfinite(seed)):
            continue
        rs = ref_cache.get(idx)
        if rs is None:
            c0 = np.rint(grid.coords[idx]).astype(np.int64)
            rs = _prepare_reference(ref, c0, dx, subset_shape)
            ref_cache[idx] = rs
        if rs is None:
            continue
        ut = u_target[(slice(None), *idx)] if u_target is not None else None
        Ft = (F_target[(slice(None), slice(None), *idx)]
              if F_target is not None else None)
        u, G, zncc, it = _icgn_subset(
            rs, defm_pref, seed, np.zeros((ndim, ndim)),
            tol=tol, max_iter=max_iter, mu=mu, beta=beta,
            u_target=ut, F_target=Ft)
        u_out[(slice(None), *idx)] = u
        F_out[(slice(None), slice(None), *idx)] = G
        zncc_out[idx] = zncc
        iters_out[idx] = it
        if progress_cb is not None and (k & 31) == 0 and n > 0:
            progress_cb(int(progress_lo + (progress_hi - progress_lo) * k / n))

    return u_out, F_out, zncc_out, iters_out


# ==== vendored from nd2studios/backend/dvc/parallel.py ====
"""
Process-pool fan-out for the DVC local step (Stages 3 & 5).

The IC-GN sweep is embarrassingly parallel across subsets and is the dominant
cost of DVC on a real 3D stack. This module fans the sweep across all physical
cores with a :class:`concurrent.futures.ProcessPoolExecutor`, putting the
reference and (prefiltered) deformed volumes into ``multiprocessing.shared_memory``
**once** (via :mod:`nd2studios.compute.parallel.shared_array`) so workers attach
lock-free instead of re-pickling hundreds of MB per task. The per-subset kernel is
the module-level :func:`nd2studios.backend.dvc.icgn._icgn_subset` (closures can't
pickle into spawned workers — Windows uses spawn).

Trade-off vs. the serial path: workers recompute each subset's reference
SD/Hessian (the serial path caches them across ADMM iterations), but the multicore
win dominates on large grids. ``n_workers <= 1`` keeps the cached serial path.

Pure numpy/scipy — no PySide6.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional, Tuple

import numpy as np



def default_workers() -> int:
    """A sensible default worker count: physical cores minus one, capped at 8."""
    return max(1, min((os.cpu_count() or 2) - 1, 8))


def _icgn_block(args):
    """Worker: IC-GN over a block of subsets. Attaches the shared volumes,
    solves each subset, returns ``[(u, G, zncc, iters) | None, ...]``."""
    # Import inside the worker so the spawned process resolves them fresh.
    (ref_meta, def_meta, centers, seeds, u_targets, F_targets,
     subset_size, tol, max_iter, mu, beta) = args
    ref, ref_shm = attach_shared(*ref_meta)
    defm, def_shm = attach_shared(*def_meta)
    try:
        ndim = len(ref_meta[1])
        dx = _subset_offsets(subset_size, ndim)
        half = max(1, int(subset_size) // 2)
        subset_shape = tuple([half * 2 + 1] * ndim)
        out = []
        for k in range(len(centers)):
            seed = seeds[k]
            if not np.all(np.isfinite(seed)):
                out.append(None)
                continue
            c0 = np.rint(centers[k]).astype(np.int64)
            rs = _prepare_reference(ref, c0, dx, subset_shape)
            if rs is None:
                out.append(None)
                continue
            ut = u_targets[k] if u_targets is not None else None
            Ft = F_targets[k] if F_targets is not None else None
            res = _icgn_subset(rs, defm, seed, np.zeros((ndim, ndim)),
                               tol=tol, max_iter=max_iter, mu=mu, beta=beta,
                               u_target=ut, F_target=Ft)
            out.append(res)
        return out
    finally:
        ref_shm.close()
        def_shm.close()


def parallel_local_icgn(
    ref: np.ndarray, defm_pref: np.ndarray, grid: Grid, u0: np.ndarray,
    subset_size: int, *, tol: float, max_iter: int, mu: float, beta: float,
    u_target: Optional[np.ndarray], F_target: Optional[np.ndarray],
    n_workers: int,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    progress_lo: int = 0, progress_hi: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """IC-GN over every subset, fanned across ``n_workers`` processes.

    Same return contract as :func:`nd2studios.backend.dvc.icgn.local_icgn`
    (``u_grid, F_grid, zncc_grid, iters_grid``, NaN for failed subsets).
    """
    ndim = grid.ndim
    gshape = grid.grid_shape
    idx_list = list(np.ndindex(*gshape))
    n = len(idx_list)
    centers = [grid.coords[idx] for idx in idx_list]
    seeds = [u0[(slice(None), *idx)] for idx in idx_list]
    uts = ([u_target[(slice(None), *idx)] for idx in idx_list]
           if u_target is not None else None)
    fts = ([F_target[(slice(None), slice(None), *idx)] for idx in idx_list]
           if F_target is not None else None)

    # Blocks: a few per worker for load balance.
    nb = max(1, int(n_workers) * 4)
    bounds = np.linspace(0, n, nb + 1).astype(int)
    blocks = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

    u_out = np.full((ndim, *gshape), np.nan)
    F_out = np.full((ndim, ndim, *gshape), np.nan)
    zncc_out = np.full(gshape, np.nan)
    iters_out = np.zeros(gshape, dtype=np.int32)

    ref = np.ascontiguousarray(ref, dtype=np.float32)
    defm_pref = np.ascontiguousarray(defm_pref, dtype=np.float32)
    with shared_ndarray(ref) as ref_meta, shared_ndarray(defm_pref) as def_meta:
        tasks = []
        for (a, b) in blocks:
            tasks.append((
                ref_meta, def_meta, centers[a:b], seeds[a:b],
                (uts[a:b] if uts is not None else None),
                (fts[a:b] if fts is not None else None),
                int(subset_size), float(tol), int(max_iter), float(mu), float(beta)))
        done = 0
        with ProcessPoolExecutor(max_workers=int(n_workers)) as ex:
            for (a, b), block_res in zip(blocks, ex.map(_icgn_block, tasks)):
                for j, res in enumerate(block_res):
                    if res is None:
                        continue
                    idx = idx_list[a + j]
                    u, G, zncc, it = res
                    u_out[(slice(None), *idx)] = u
                    F_out[(slice(None), slice(None), *idx)] = G
                    zncc_out[idx] = zncc
                    iters_out[idx] = it
                done += (b - a)
                if progress_cb is not None and n > 0:
                    progress_cb(int(progress_lo + (progress_hi - progress_lo) * done / n))
                if cancelled_cb is not None and cancelled_cb():
                    raise InterruptedError("DVC cancelled")
    return u_out, F_out, zncc_out, iters_out


# ==== vendored from nd2studios/backend/dvc/admm.py ====
"""
Stage 6 — the ALDVC ADMM outer loop (the "augmented Lagrangian" in ALDVC).

Ties the local image-correlation solve (Subpb1, :mod:`icgn`) to the global
compatibility projection (Subpb2, :mod:`global_step`) and alternates them with
scaled-dual updates until the compatible displacement stops moving. This is what
buys global-DVC accuracy at near-local-DVC cost.

Standard scaled ADMM for ``min f(u,F) s.t. u=û, F=∇û`` (``f`` = the ZNSSD image
residual solved by IC-GN):

* **Subpb1** — penalized local IC-GN, warp pulled toward ``(û − s_u, F̂ − s_F)``.
* **Subpb2** — global solve ``(μI + βDᵀD)û = μ(u+s_u) + βDᵀ(F+s_F)``.
* **duals** — ``s_u += (u − û)``, ``s_F += (F − F̂)``.
* **converge** — ``‖û_new − û_old‖₂/√N < ADMMtol``, capped at ``admm_iterations``.

Pass 0 (before the loop) is plain local IC-GN + one global solve ⇒ conventional
DVC; each subsequent iteration tightens compatibility.

Pure numpy/scipy — no PySide6.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np



@dataclass
class ADMMResult:
    u: np.ndarray           # (ndim, *grid) compatible displacement (voxels)
    F: np.ndarray           # (ndim, ndim, *grid) compatible gradient ∇û
    zncc: np.ndarray        # (*grid) final local correlation confidence
    converged: bool
    iterations: int
    beta: float
    residuals: list         # per-iteration ‖Δû‖₂/√N


def _inpaint_tensor(F: np.ndarray) -> np.ndarray:
    out = np.array(F, dtype=np.float64, copy=True)
    ndim = out.shape[0]
    for i in range(ndim):
        for j in range(ndim):
            out[i, j] = inpaint_nans(out[i, j])
    return out


def run_admm(
    ref: np.ndarray,
    defm_pref: np.ndarray,
    grid: Grid,
    u0: np.ndarray,
    subset_size: int,
    *,
    mu: float = 1e-3,
    admm_iterations: int = 4,
    icgn_tol: float = 1e-2,
    icgn_max_iter: int = 100,
    admm_tol: float = 1e-2,
    cc_thresh: float = 0.5,
    median_thresh: float = 2.0,
    n_workers: int = 1,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    progress_lo: int = 0,
    progress_hi: int = 100,
) -> ADMMResult:
    """Run conventional local DVC (pass 0) then the ADMM refinement loop.

    ``ref`` is the (normalized) reference; ``defm_pref`` the (normalized) deformed
    volume **already spline-prefiltered**. ``u0`` is the ``(ndim,*grid)`` integer
    seed. Returns an :class:`ADMMResult` with the compatible field.
    """
    ndim = grid.ndim
    ref_cache: dict = {}                       # per-subset reference SD/H reuse
    n_iters = max(0, int(admm_iterations))

    def _span(lo_frac: float, hi_frac: float):
        lo = progress_lo + (progress_hi - progress_lo) * lo_frac
        hi = progress_lo + (progress_hi - progress_lo) * hi_frac
        return int(lo), int(hi)

    # ── Pass 0: conventional local IC-GN ───────────────────────────────
    p_lo, p_hi = _span(0.0, 0.5 if n_iters else 0.9)
    u_L, F_L, zncc, _iters = local_icgn(
        ref, defm_pref, grid, u0, subset_size,
        tol=icgn_tol, max_iter=icgn_max_iter, ref_cache=ref_cache,
        n_workers=n_workers,
        progress_cb=progress_cb, cancelled_cb=cancelled_cb,
        progress_lo=p_lo, progress_hi=p_hi)

    # Clean the noisy local field before the compatibility solve.
    u_L, _bad = remove_outliers(u_L, zncc, cc_thresh=cc_thresh,
                                median_thresh=median_thresh)
    u_L = inpaint_vector(u_L)
    F_L = _inpaint_tensor(F_L)

    # ── First global solve (β via L-curve) ─────────────────────────────
    gstep = AugLagGlobalStep(grid)
    beta = gstep.tune_beta(u_L, F_L, mu)
    uhat, Fhat = gstep.solve(u_L, F_L, None, None, mu, beta)

    residuals: list = []
    converged = (n_iters == 0)
    if n_iters == 0:
        return ADMMResult(u=uhat, F=Fhat, zncc=zncc, converged=True,
                          iterations=0, beta=beta, residuals=residuals)

    # ── ADMM loop ──────────────────────────────────────────────────────
    wu = np.zeros_like(uhat)                   # scaled dual for u constraint
    wF = np.zeros_like(Fhat)                   # scaled dual for F constraint
    n_nodes = float(grid.n_nodes)
    it = 0
    for it in range(1, n_iters + 1):
        if cancelled_cb is not None and cancelled_cb():
            raise InterruptedError("DVC cancelled")
        a = uhat - wu                          # Subpb1 targets (z − dual)
        B = Fhat - wF
        li_lo, li_hi = _span(0.5 + 0.5 * (it - 1) / n_iters,
                             0.5 + 0.5 * it / n_iters)
        u1, F1, zncc, _it2 = local_icgn(
            ref, defm_pref, grid, uhat, subset_size,
            tol=icgn_tol, max_iter=icgn_max_iter,
            mu=mu, beta=beta, u_target=a, F_target=B, ref_cache=ref_cache,
            n_workers=n_workers,
            progress_cb=progress_cb, cancelled_cb=cancelled_cb,
            progress_lo=li_lo, progress_hi=li_hi)
        u1 = inpaint_vector(u1)
        F1 = _inpaint_tensor(F1)

        prev = uhat
        uhat, Fhat = gstep.solve(u1, F1, wu, wF, mu, beta)
        wu = wu + (u1 - uhat)                   # dual update (x − z)
        wF = wF + (F1 - Fhat)

        change = float(np.linalg.norm((uhat - prev).ravel()) / np.sqrt(n_nodes))
        residuals.append(change)
        if change < admm_tol:
            converged = True
            break

    return ADMMResult(u=uhat, F=Fhat, zncc=zncc, converged=converged,
                      iterations=it, beta=beta, residuals=residuals)


# ==== vendored from nd2studios/backend/dvc/engine.py ====
"""
Top-level ALDVC orchestration.

Runs the full Augmented Lagrangian Digital Volume Correlation pipeline on one
reference→deformed pair and returns a :class:`DVCResult`:

    Stage 0  normalize both volumes, spline-prefilter the deformed one once
    Stage 1  per-subset FFT integer displacement seed        (integer_search)
    Stage 2  outlier clean + inpaint the seed                 (outliers)
    Stage 3–6  local IC-GN + ADMM global compatibility loop   (admm → icgn+global_step)
    Stage 7  strain tensor from the compatible field          (strain)

Works for 2D images (DIC, 6-DOF) and 3D volumes (DVC, 12-DOF); dimensionality is
inferred from the input. Axis order throughout is mesh order (``(y,x)`` /
``(z,y,x)``, slowest first). Pure numpy/scipy — NO PySide6 (backend-purity rule).

This replaces the ``Add-DVC`` Phase-0 global-shift stub; ``normalize_volume`` is
retained (Stage 0). See ``CodeLog/ClaudesPlan/V1.51_dvc_aldvc_node.md``.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import spline_filter



# ──────────────────────────────────────────────────────────────────────
# Stage 0 — volume preparation
# ──────────────────────────────────────────────────────────────────────
def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Affine min/max rescale to float32 in ``[0, 1]`` over the whole volume.

    Mirrors ALDVC ``funNormalizeImg3``. A degenerate (constant) volume maps to
    all-zeros rather than dividing by zero.
    """
    arr = np.asarray(vol, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _get(raw: Dict[str, Any], key: str, default):
    v = raw.get(key, default)
    return default if v is None else v


def run_aldvc(
    ref_vol: np.ndarray,
    def_vol: np.ndarray,
    voxel_size_um: Tuple[float, ...],
    params: Dict[str, Any],
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    u0_seed: Optional[np.ndarray] = None,
    use_fft_seed: bool = True,
) -> DVCResult:
    """Run ALDVC on a single reference/deformed pair.

    ``ref_vol``/``def_vol`` are ``(H,W)`` (2D DIC) or ``(Z,H,W)`` (3D DVC) and
    must share a shape. ``params`` is the ParamEditor dict (see
    :class:`~nd2studios.core.dvc_registry.DVCParams` plus optional
    ``search_radius``, ``seed_levels``, ``icgn_tol``, ``icgn_max_iter``,
    ``admm_tol``, ``cc_thresh``, ``median_thresh``, ``strain_smooth``).

    ``u0_seed`` (``(ndim,*grid)``) + ``use_fft_seed=False`` warm-start the seed
    from a prior frame's field (ALDVC's cross-frame ``U0``), skipping the FFT
    search; on any shape mismatch the FFT multigrid seed is used instead.
    """
    ref = np.asarray(ref_vol)
    mov = np.asarray(def_vol)
    if ref.shape != mov.shape:
        raise ValueError(f"reference and deformed volumes must match: "
                         f"{ref.shape} vs {mov.shape}")
    if ref.ndim not in (2, 3):
        raise ValueError(f"expected a 2D image or 3D volume, got ndim={ref.ndim}")

    p = DVCParams.from_dict(params)
    raw = params or {}
    dim = ref.ndim
    subset_size = int(p.subset_size)
    subset_spacing = int(p.subset_spacing)
    search_radius = int(_get(raw, "search_radius", 0)) or max(4, subset_size)
    seed_levels = max(1, int(_get(raw, "seed_levels", 3)))
    correlation = str(p.correlation)
    mu = float(p.mu)
    admm_iterations = int(p.admm_iterations)
    strain_type = str(p.strain_type)
    strain_smooth = float(_get(raw, "strain_smooth", 0.0))
    use_gpu = bool(p.use_gpu)
    n_workers = int(_get(raw, "n_workers", 0))
    icgn_tol = float(_get(raw, "icgn_tol", 1e-2))
    icgn_max_iter = int(_get(raw, "icgn_max_iter", 100))
    admm_tol = float(_get(raw, "admm_tol", 1e-2))
    cc_thresh = float(_get(raw, "cc_thresh", 0.5))
    median_thresh = float(_get(raw, "median_thresh", 2.0))

    def _tick(v: int) -> None:
        if progress_cb is not None:
            progress_cb(int(max(0, min(100, v))))

    def _cancelled() -> bool:
        return bool(cancelled_cb()) if cancelled_cb is not None else False

    _tick(2)
    ref_n = normalize_volume(ref)
    def_n = normalize_volume(mov)
    if _cancelled():
        raise InterruptedError("DVC cancelled")
    # Prefilter the deformed volume ONCE so per-iteration warping is a pure
    # cubic-B-spline evaluation (map_coordinates prefilter=False).
    defm_pref = spline_filter(def_n.astype(np.float32), order=3,
                              mode="nearest").astype(np.float32)
    grid = build_grid(ref.shape, subset_size, subset_spacing)
    # Fan the IC-GN sweep across cores by default; skip the process-pool overhead
    # on tiny grids. GPU (seed FFTs) and CPU multiprocessing (IC-GN) compose.
    if n_workers <= 0:
        n_workers = default_workers()
    if grid.n_nodes < 64:
        n_workers = 1
    _tick(8)

    # Stage 1 — integer seed. Warm-start from a prior frame's field when given
    # (ALDVC's cross-frame U0), else the multigrid FFT search; clean the FFT seed
    # (the warm-start field is already smooth).
    want_warm = (u0_seed is not None and not use_fft_seed
                 and tuple(np.shape(u0_seed)) == (dim, *grid.grid_shape))
    if want_warm:
        u0 = inpaint_vector(np.asarray(u0_seed, dtype=np.float64).copy())
        _tick(32)
    else:
        u0, cc = integer_search_multigrid(
            ref_n, def_n, grid, subset_size, search_radius, levels=seed_levels,
            correlation=correlation, use_gpu=use_gpu,
            progress_cb=progress_cb, cancelled_cb=cancelled_cb,
            progress_lo=8, progress_hi=32)
        u0, _bad = remove_outliers(u0, cc, cc_thresh=cc_thresh,
                                   median_thresh=median_thresh)
        u0 = inpaint_vector(u0)
    _tick(34)

    # Stages 3–6 — local IC-GN + ADMM compatibility loop.
    res = run_admm(
        ref_n, defm_pref, grid, u0, subset_size,
        mu=mu, admm_iterations=admm_iterations,
        icgn_tol=icgn_tol, icgn_max_iter=icgn_max_iter, admm_tol=admm_tol,
        cc_thresh=cc_thresh, median_thresh=median_thresh, n_workers=n_workers,
        progress_cb=progress_cb, cancelled_cb=cancelled_cb,
        progress_lo=34, progress_hi=90)
    _tick(92)

    # Stage 7 — strain.
    if not voxel_size_um or len(voxel_size_um) != dim:
        voxel_size_um = tuple([1.0] * dim)
    voxel = np.asarray(voxel_size_um, dtype=np.float64)
    _F_def, strain = compute_strain(
        res.u, grid.step, voxel_size=voxel,
        strain_type=strain_type, smooth_sigma=strain_smooth)
    _tick(98)

    disp = np.moveaxis(res.u, 0, -1)                  # (*grid, ndim), voxels
    strain_field = np.moveaxis(strain, (0, 1), (-2, -1))   # (*grid, ndim, ndim)

    _tick(100)
    return DVCResult(
        dim=dim,
        grid_coords=grid.coords,
        displacement_field=disp,
        voxel_size_um=tuple(float(v) for v in voxel_size_um),
        strain_field=strain_field,
        strain_type=strain_type,
        qfactor=np.asarray(res.zncc),
        converged=bool(res.converged),
        iterations=int(res.iterations),
        mu=mu,
        beta=float(res.beta),
        method="ALDVC",
        notes=(
            f"ALDVC · {dim}D · subset={subset_size} spacing={subset_spacing} · "
            f"ADMM {res.iterations}/{admm_iterations} iters "
            f"(converged={res.converged}) · β={res.beta:.3g}"
        ),
        diagnostics={
            "grid_shape": list(grid.grid_shape),
            "n_subsets": int(grid.n_nodes),
            "beta": float(res.beta),
            "admm_residuals": [float(r) for r in res.residuals],
            "median_zncc": float(np.nanmedian(res.zncc)) if res.zncc.size else 0.0,
            "search_radius": int(search_radius),
            "n_workers": int(n_workers),
            "use_gpu": bool(use_gpu),
        },
    )


# ==== vendored from nd2studios/backend/dvc/tracking.py ====
"""
Incremental → cumulative accumulation (ALDVC Section, main_ALDVC.m lines 631–706).

In **incremental** tracking mode ALDVC correlates each consecutive frame pair
(N-1 → N), which keeps every correlated step small (robust for large total motion),
then **composes** the per-step increments into a cumulative displacement field by
Lagrangian point-tracking: the reference grid points are advanced through the
sequence — at each step the increment is interpolated at the points' *current*
(drifted) positions and added — and the cumulative displacement is
``U_accum = coordCurr − coord``. Strain is then computed from ``U_accum``.

This module ports that accumulation. It replaces the MATLAB ``interp3(...,'makima')``
+ median±σ clip + ``inpaint_nans3`` with a :class:`RegularGridInterpolator` +
finite-value guard (the grids are regular, so a grid interpolator is exact and
cheaper than scattered interpolation).

Pure numpy/scipy — no PySide6.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator



def accumulate_incremental(
    grid: Grid, increments: List[Tuple[int, np.ndarray]],
) -> List[Tuple[int, np.ndarray]]:
    """Compose ordered per-step increment fields into cumulative displacements.

    Parameters
    ----------
    grid : the (shared) subset grid the increments live on.
    increments : ``[(t, u_grid), ...]`` in ascending frame order, each ``u_grid``
        an ``(ndim, *grid)`` incremental displacement (frame ``t-1`` → ``t``), in
        voxels.

    Returns ``[(t, u_accum_grid), ...]`` — the cumulative displacement from the
    reference frame at each ``t`` (``(ndim, *grid)``), by tracking the reference
    grid points through the increments.
    """
    ndim = grid.ndim
    axes = [np.asarray(a, dtype=np.float64) for a in grid.axes]
    coords0 = grid.coords_flat().astype(np.float64)     # (N, ndim), reference
    cur = coords0.copy()
    out: List[Tuple[int, np.ndarray]] = []
    for t, u_grid in increments:
        u_grid = np.asarray(u_grid, dtype=np.float64)
        disp = np.zeros_like(cur)
        for c in range(ndim):
            interp = RegularGridInterpolator(
                axes, u_grid[c], method="linear",
                bounds_error=False, fill_value=None)   # extrapolate at borders
            disp[:, c] = np.nan_to_num(interp(cur), nan=0.0,
                                       posinf=0.0, neginf=0.0)
        cur = cur + disp
        u_accum = (cur - coords0).reshape(*grid.grid_shape, ndim)
        out.append((int(t), np.moveaxis(u_accum, -1, 0)))   # (ndim, *grid)
    return out


def build_accumulated_results(
    grid: Grid, increment_results: List[Tuple[int, DVCResult]],
    voxel_size_um: Tuple[float, ...], *, strain_type: str = "infinitesimal",
    strain_smooth: float = 0.0,
) -> Dict[int, DVCResult]:
    """Turn ordered incremental :class:`DVCResult`s into cumulative ones.

    Accumulates the increments (:func:`accumulate_incremental`), then rebuilds a
    :class:`DVCResult` per frame with the cumulative displacement + strain
    recomputed from it — the field ALDVC's incremental mode actually reports.
    """
    ndim = grid.ndim
    incr = [(t, np.moveaxis(np.asarray(r.displacement_field), -1, 0))
            for t, r in increment_results]     # (ndim,*grid) each
    accum = accumulate_incremental(grid, incr)
    voxel = (np.asarray(voxel_size_um, dtype=np.float64)
             if voxel_size_um and len(voxel_size_um) == ndim else np.ones(ndim))
    out: Dict[int, DVCResult] = {}
    base_by_t = dict(increment_results)
    for t, u_grid in accum:
        base = base_by_t[t]
        _F, strain = compute_strain(u_grid, grid.step, voxel_size=voxel,
                                    strain_type=strain_type,
                                    smooth_sigma=strain_smooth)
        disp = np.moveaxis(u_grid, 0, -1)                    # (*grid, ndim)
        strain_field = np.moveaxis(strain, (0, 1), (-2, -1))
        diag = dict(base.diagnostics)
        diag["accumulated_from_incremental"] = True
        out[t] = DVCResult(
            dim=base.dim, grid_coords=base.grid_coords, displacement_field=disp,
            voxel_size_um=base.voxel_size_um, strain_field=strain_field,
            strain_type=strain_type, qfactor=base.qfactor,
            converged=base.converged, iterations=base.iterations,
            mu=base.mu, beta=base.beta, method="ALDVC (cumulative from incremental)",
            notes=base.notes + " · accumulated to cumulative", diagnostics=diag)
    return out


# ==== vendored from nd2studios/backend/dvc/method.py ====
"""
Registered DVC methods.

Ships :class:`ALDVCMethod`, whose ``run`` calls the full ALDVC pipeline in
:func:`nd2studios.backend.dvc.engine.run_aldvc`. Force-imported in
``__main__.py`` so its ``@DVCMethod.register`` decorator fires at startup and the
method is discoverable by the DVC node (its ``get_params`` also supplies the
node's ParamSpec list, via ``registry_adapter.param_specs_for``).

Pure backend module — no PySide6.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np



class ALDVCMethod:
    name = "ALDVC"
    description = (
        "Augmented Lagrangian Digital Volume Correlation — hybrid local "
        "(IC-GN subset) + global (compatibility) DVC. 3D volumes or 2D "
        "images (DIC)."
    )

    def run(
        self,
        ref_vol: np.ndarray,
        def_vol: np.ndarray,
        voxel_size_um: Tuple[float, ...],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[[int], None]] = None,
        cancelled_cb: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> DVCResult:
        # ``kwargs`` forwards the series-level extras (``u0_seed`` / ``use_fft_seed``
        # for cross-frame warm-start) that the DVC worker passes per frame.
        return run_aldvc(
            ref_vol, def_vol, voxel_size_um, params,
            progress_cb=progress_cb, cancelled_cb=cancelled_cb, **kwargs,
        )
