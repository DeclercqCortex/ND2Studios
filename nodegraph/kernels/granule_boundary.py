# =============================================================================
# VENDORED KERNEL — Granule Boundary Extraction
# -----------------------------------------------------------------------------
# Purpose: grow an outward boundary band around each labeled granule and paint a
#          combined band-label volume (overlaps resolved nearest-surface/lowest-id).
#
# WHERE THE REAL MATH LIVES: fully in-repo. The algorithm is pure NumPy +
#          scipy.ndimage (binary_dilation / distance_transform_edt /
#          generate_binary_structure). No external analysis package is involved.
#
# Provenance (branch Version-1.45):
#   - nd2studios/backend/analysis/granule_boundary.py  (entire module, verbatim)
#   - nd2studios/backend/analysis/granule_types.py     (only COMBINED_LABELS_KEY,
#                                                        inlined as a constant)
#
# Vendored verbatim; imports nothing from nd2studios; caller owns all prep
# (file I/O, per-multipoint/per-timepoint looping, crop/downsample/registration/
# exclusion are NOT done here — the caller supplies already-prepared masks +
# label volume + voxel size).
#
# Dropped UI/registry-only members: NONE. granule_boundary.py contained no
# get_params/@register/ParamEditor/registry base — the whole module is compute path.
#
# Edits applied (rules): (a) removed the `from nd2studios...granule_types import
# COMBINED_LABELS_KEY` and inlined the constant `COMBINED_LABELS_KEY = "_labels"`.
# No renames (b) and no dropped members (c) were needed.
# =============================================================================
"""Granule boundary-band extraction (V1.70 · P5) — pure, Qt-free.

From the labeled granule voxels (P4), grow an **outward** boundary band of
thickness ``N`` around each granule: the voxels within ``N`` of a granule's
surface whose label is **not** that granule — i.e. *"the direction that there is
not the same value"*, which is background **and** neighbouring granules (P0
decision 9). The band masks the raw / DVC volume so the matrix immediately
surrounding each granule can be recovered and correlated.

Two band constructions are offered (``params["band_method"]``):

* ``"dilation"`` (default) — a **voxel-count** band. ``binary_dilation`` of the
  granule mask by ``N`` iterations with a 6-connectivity (face) structuring
  element, minus the granule itself, intersected with "not this granule". The
  6-connectivity element is deliberate: iterating it ``N`` times grows the mask
  by taxicab distance ``≤ N`` and makes the band agree with the ``edt`` method at
  ``N = 1`` on isotropic voxels.
* ``"edt"`` — a **metric** band honouring anisotropic voxel size. The Euclidean
  distance transform of the granule's complement (with ``sampling=(dz,dy,dx)``)
  gives each voxel's µm distance to the surface; the band keeps voxels with
  ``0 < edt <= threshold`` where ``threshold`` is ``band_um`` when set, else
  ``N * min(dz, dy, dx)``.

Combined band-label volume: every band voxel is painted with its source granule
id. When a voxel falls in **two** bands, the tie-break is *nearest surface* — the
granule whose surface is closer (smaller EDT distance) wins; on an exact tie (or
equal distance), the **lowest granule id** wins (granules are processed in
ascending id order with a strict ``<`` comparison, so an equal-distance later id
never displaces an already-claimed voxel).

Pure backend — **no PySide6** (backend-purity rule). ``scipy.ndimage`` is a hard
dependency here as elsewhere in this package (see ``mask3d`` /
``distance_transform_edt``).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np

# Inlined from nd2studios.backend.analysis.granule_types (rule 2: no nd2studios
# imports). This is the reserved dict key inside the masks/bands per-(m,t) dict
# for the combined integer-label volume, sitting beside the per-granule bool masks.
COMBINED_LABELS_KEY = "_labels"

BAND_METHOD_DILATION = "dilation"
BAND_METHOD_EDT = "edt"

# Param keys / defaults (P6 wires these from the node editor).
_DEFAULT_BAND_VOXELS = 1
_DEFAULT_BAND_METHOD = BAND_METHOD_DILATION
_DEFAULT_INCLUDE_NEIGHBORS = True


def _iter_granule_masks(
    masks_by_id: Mapping[Any, np.ndarray],
) -> Dict[int, np.ndarray]:
    """Return ``{int gid -> (Z,H,W) bool}`` from a masks dict.

    The stored per-``(m, t)`` dict (``record._granule_masks_by_m``) also carries
    the reserved :data:`COMBINED_LABELS_KEY` integer-label volume beside the
    per-granule bool masks; skip it (and any non-integer key) so the caller may
    hand us the whole dict unfiltered.
    """
    out: Dict[int, np.ndarray] = {}
    for key, mask in (masks_by_id or {}).items():
        if key == COMBINED_LABELS_KEY:
            continue
        try:
            gid = int(key)
        except (TypeError, ValueError):
            continue
        out[gid] = np.asarray(mask, dtype=bool)
    return out


def extract_boundary_bands(
    masks_by_id: Mapping[Any, np.ndarray],
    combined_labels_zhw: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    params: Mapping[str, Any],
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Extract an outward boundary band per granule.

    Parameters
    ----------
    masks_by_id:
        ``{granule_id:int -> (Z,H,W) bool}`` per-granule masks (P4 output). A
        reserved :data:`COMBINED_LABELS_KEY` entry, if present, is ignored.
    combined_labels_zhw:
        ``(Z,H,W)`` int label volume — voxel value is the granule id owning it,
        ``0`` for background. Used both for the "not this granule" mask and for
        the ``include_neighbors=False`` background-only restriction.
    voxel_size_um:
        ``(dz, dy, dx)`` µm. Feeds ``distance_transform_edt``'s ``sampling`` so
        the EDT band and the overlap tie-break are anisotropy-correct.
    params:
        ``band_voxels`` (int ``N``, default 1), ``band_method``
        (``"dilation"`` | ``"edt"``, default ``"dilation"``), ``band_um`` (float;
        overrides ``N * min(voxel)`` for the EDT threshold when ``> 0``),
        ``include_neighbors`` (bool, default True — when False the band is
        restricted to background, ``combined_labels == 0``).

    Returns
    -------
    (bands_by_id, combined_band_labels):
        ``bands_by_id`` is ``{granule_id:int -> (Z,H,W) bool}``; empty / missing
        granules yield an all-False band. ``combined_band_labels`` is a
        ``(Z,H,W)`` int32 volume painting each band voxel with its source
        granule id (0 = no band), overlaps resolved nearest-surface / lowest-id.
    """
    labels = np.asarray(combined_labels_zhw)
    shape = tuple(int(s) for s in labels.shape)

    dz, dy, dx = (float(voxel_size_um[0]), float(voxel_size_um[1]),
                  float(voxel_size_um[2]))
    sampling = (dz, dy, dx)
    min_voxel = min(dz, dy, dx)

    n = int(params.get("band_voxels", _DEFAULT_BAND_VOXELS) or 0)
    n = max(0, n)
    method = str(params.get("band_method", _DEFAULT_BAND_METHOD)).lower()
    include_neighbors = bool(
        params.get("include_neighbors", _DEFAULT_INCLUDE_NEIGHBORS)
    )
    band_um_param = params.get("band_um", None)
    band_um = float(band_um_param) if band_um_param not in (None, "") else 0.0

    # EDT threshold (µm): explicit band_um wins, else N voxels of the finest axis.
    edt_threshold = band_um if band_um > 0.0 else float(n) * min_voxel

    masks = _iter_granule_masks(masks_by_id)

    bands_by_id: Dict[int, np.ndarray] = {}
    combined_band = np.zeros(shape, dtype=np.int32)
    if not masks:
        return bands_by_id, combined_band

    # "not this granule" varies by include_neighbors: background + neighbours
    # (label != gid) by default, or background only (label == 0) when False.
    from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure

    # 6-connectivity (face-adjacent) 3-D structuring element. Iterating it N times
    # grows the mask by taxicab distance ≤ N and matches the EDT band at N=1.
    structure = generate_binary_structure(3, 1)

    # Nearest-surface distance of the granule currently owning each voxel; used to
    # break band overlaps. Ascending-id processing + strict `<` ⇒ lowest id wins ties.
    best_dist = np.full(shape, np.inf, dtype=np.float64)

    for gid in sorted(masks):
        mask = masks[gid]
        band = np.zeros(shape, dtype=bool)
        if mask.shape != shape or not mask.any():
            # Empty / one-voxel-that-fills-nothing / shape-mismatch → empty band.
            bands_by_id[gid] = band
            continue

        # Distance (µm) from every voxel to this granule's surface. Zero inside
        # the granule; >0 outside. Also the EDT band criterion and the tie-break.
        edt = distance_transform_edt(~mask, sampling=sampling)

        # Voxels that are "not this granule".
        if include_neighbors:
            not_self = labels != gid
        else:
            not_self = labels == 0

        if method == BAND_METHOD_EDT:
            band = (edt > 0.0) & (edt <= edt_threshold) & not_self & ~mask
        else:  # BAND_METHOD_DILATION (default)
            if n <= 0:
                grown = mask
            else:
                grown = binary_dilation(mask, structure=structure, iterations=n)
            band = grown & ~mask & not_self

        bands_by_id[gid] = band
        if not band.any():
            continue

        # Paint the combined volume: take voxels this band claims that are
        # unclaimed or strictly closer to this granule's surface than the current
        # owner. Equal distance keeps the earlier (lower-id) owner.
        take = band & (edt < best_dist)
        combined_band[take] = np.int32(gid)
        best_dist[take] = edt[take]

    return bands_by_id, combined_band
