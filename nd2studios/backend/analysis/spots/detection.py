from __future__ import annotations

from typing import Literal

import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima, h_minima


def find_extrema(
    response: np.ndarray,
    *,
    polarity: Literal["bright", "dark"],
    min_distance: int,
    contrast: float,
    intensity_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return (N, 2) array of (row, col) extrema in the normalised response map.

    `response` must be normalised to [-1, 1] before calling so that `contrast`
    is bit-depth-independent. `intensity_mask` (bool, same shape) restricts the
    search to pixels that pass the LUT-histogram intensity gate.
    """
    if polarity == "bright":
        search = response.astype(np.float32, copy=True)
    else:
        search = -response.astype(np.float32)

    # Suppress regions that fail the intensity gate by setting them below any
    # possible peak threshold so peak_local_max never selects them.
    if intensity_mask is not None:
        floor = float(search.min()) - 1.0
        search[~intensity_mask] = floor

    coords = peak_local_max(search, min_distance=max(1, min_distance))

    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.intp)

    # Contrast gate: keep only peaks with sufficient normalised response magnitude.
    scores = np.abs(response[coords[:, 0], coords[:, 1]])
    keep = scores >= contrast
    return coords[keep]


def h_transform_seeds(
    image: np.ndarray,
    *,
    polarity: Literal["bright", "dark"],
    h: float,
) -> np.ndarray:
    """Return a boolean seed mask from h-maxima (bright) or h-minima (dark).

    Each connected component in the result is one spot seed. Used as an
    alternative to DoG + peak_local_max for cross-validation against classical
    morphological detectors (Vincent 1993).
    """
    img_f = image.astype(np.float32)
    if polarity == "bright":
        return h_maxima(img_f, h=h) > 0
    return h_minima(img_f, h=h) > 0
