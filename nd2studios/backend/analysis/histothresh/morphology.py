from __future__ import annotations

import numpy as np
from skimage.morphology import (
    opening,
    closing,
    disk,
    ball,
    remove_small_objects,
    remove_small_holes,
)
from scipy.ndimage import generic_filter


def apply_spatial_constraints(
    mask: np.ndarray,
    *,
    min_area: int = 100,
    opening_radius: int = 1,
    closing_radius: int = 2,
    min_hole_size: int = 50,
    is_3d: bool = False,
) -> np.ndarray:
    """Apply opening → closing → hole-fill → small-object removal in that order.

    Opening first kills isolated noise pixels before closing would bridge them.
    Closing bridges legitimate small gaps. Hole filling repairs interior pockets.
    Small-object removal applies the size threshold last.
    """
    footprint_fn = ball if is_3d else disk
    out = mask.copy()
    if opening_radius > 0:
        out = opening(out, footprint=footprint_fn(opening_radius))
    if closing_radius > 0:
        out = closing(out, footprint=footprint_fn(closing_radius))
    if min_hole_size > 0:
        # max_size removes holes <= value; subtract 1 to match old "< value" semantics
        out = remove_small_holes(out, max_size=max(0, min_hole_size - 1))
    if min_area > 0:
        out = remove_small_objects(out, max_size=max(0, min_area - 1))
    return out


def homogeneity_gate(
    image: np.ndarray,
    *,
    window: int = 7,
    std_max: float = 20.0,
    is_3d: bool = False,
) -> np.ndarray:
    """Return a boolean mask of pixels whose local standard deviation is below `std_max`.

    Use to suppress noisy/textured regions that pass intensity thresholds on average
    but aren't truly homogeneous. O(N * window^d) — disabled by default.
    """
    size = (window,) * (3 if is_3d else 2)
    local_std = generic_filter(image.astype(np.float32), np.std, size=size)
    return local_std < std_max
