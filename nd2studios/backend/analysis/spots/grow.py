from __future__ import annotations

from typing import Literal

import numpy as np
from skimage.morphology import dilation, disk
from skimage.segmentation import watershed


def grow_seeds_dilation(seed_mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Grow a boolean seed mask by morphological dilation using a disk footprint."""
    if radius_px <= 0:
        return seed_mask.astype(bool)
    return dilation(seed_mask, footprint=disk(radius_px))


def grow_seeds_watershed(
    image: np.ndarray,
    seed_labels: np.ndarray,
    *,
    polarity: Literal["bright", "dark"],
    boundary_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Expand seeds to the full bright/dark region using watershed-from-marker.

    For bright spots the image is negated so seeds sit at local minima of the
    surface (= local maxima of the original image). `boundary_mask` (bool array)
    limits expansion — pass the intensity gate mask to prevent runaway fill.
    Returns int32 label array.
    """
    img_f = image.astype(np.float32)
    surface = -img_f if polarity == "bright" else img_f
    result = watershed(surface, markers=seed_labels, mask=boundary_mask)
    return result.astype(np.int32)
