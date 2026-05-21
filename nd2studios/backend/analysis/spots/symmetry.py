from __future__ import annotations

import math
from typing import Literal

import numpy as np
from skimage.measure import label as sk_label, regionprops

SYMMETRY_FLOOR: dict[str, float] = {
    "all": 0.0,
    "more": 0.45,
    "medium": 0.65,
    "less": 0.80,
}


def circularity(area: float, perimeter: float) -> float:
    """Return 4*pi*area / perimeter^2, clipped to [0, 1].

    Returns 0.0 when perimeter is zero to avoid division by zero.
    """
    if perimeter == 0.0:
        return 0.0
    return min(1.0, 4.0 * math.pi * area / (perimeter ** 2))


def filter_by_symmetry(
    candidates_mask: np.ndarray,
    *,
    setting: Literal["all", "more", "medium", "less"],
    floor: dict[str, float] | None = None,
) -> np.ndarray:
    """Drop connected components whose circularity falls below the threshold.

    `floor` defaults to SYMMETRY_FLOOR; an override is accepted for
    cross-validation experiments. Returns a cleaned boolean mask.
    """
    thresholds = floor if floor is not None else SYMMETRY_FLOOR
    threshold = thresholds[setting]

    bool_mask = candidates_mask.astype(bool)
    if threshold == 0.0:
        return bool_mask

    labels = sk_label(bool_mask, connectivity=2)
    out = np.zeros_like(bool_mask)

    for prop in regionprops(labels):
        if circularity(prop.area, prop.perimeter) >= threshold:
            out[labels == prop.label] = True

    return out
