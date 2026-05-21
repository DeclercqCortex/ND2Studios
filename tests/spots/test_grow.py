from __future__ import annotations

import numpy as np
import pytest
from skimage.draw import disk

from nd2studios.backend.analysis.spots.grow import (
    grow_seeds_dilation,
    grow_seeds_watershed,
)


def _point_mask(H: int, W: int, r: int, c: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[r, c] = True
    return mask


def _seed_labels(H: int, W: int, r: int, c: int) -> np.ndarray:
    labels = np.zeros((H, W), dtype=np.int32)
    labels[r, c] = 1
    return labels


# ---------------------------------------------------------------------------
# grow_seeds_dilation
# ---------------------------------------------------------------------------

def test_dilation_zero_radius_unchanged():
    mask = _point_mask(64, 64, 32, 32)
    out = grow_seeds_dilation(mask, radius_px=0)
    assert out.sum() == 1


def test_dilation_expands_point_to_disk():
    mask = _point_mask(64, 64, 32, 32)
    radius = 5
    out = grow_seeds_dilation(mask, radius_px=radius)
    # Dilating a point by disk(r) gives a disk-shaped region.
    # Area should be within 20% of the ideal circle area pi*r^2.
    import math
    ideal = math.pi * radius ** 2
    assert ideal * 0.80 <= out.sum() <= ideal * 1.30, (
        f"Dilated area {out.sum()} far from ideal circle area {ideal:.0f}"
    )


def test_dilation_returns_bool_or_compatible():
    mask = _point_mask(64, 64, 32, 32)
    out = grow_seeds_dilation(mask, radius_px=3)
    assert out.dtype == bool or np.issubdtype(out.dtype, np.integer)


# ---------------------------------------------------------------------------
# grow_seeds_watershed
# ---------------------------------------------------------------------------

def test_watershed_bright_expands_from_seed():
    """Watershed from a bright-spot seed should expand into the bright region."""
    H, W = 64, 64
    img = np.zeros((H, W), dtype=np.uint16)
    # Create a bright disk
    rr, cc = disk((32, 32), 10, shape=(H, W))
    img[rr, cc] = 3000

    seed = _seed_labels(H, W, 32, 32)
    boundary = img > 500  # restrict expansion to bright area
    result = grow_seeds_watershed(img, seed, polarity="bright", boundary_mask=boundary)

    assert result.max() >= 1
    # Watershed region should overlap with the bright disk
    assert (result[rr, cc] > 0).sum() > 0


def test_watershed_dark_expands_from_seed():
    """Watershed from a dark-spot seed should expand into the dark region."""
    H, W = 64, 64
    img = np.full((H, W), 3000, dtype=np.uint16)
    rr, cc = disk((32, 32), 10, shape=(H, W))
    img[rr, cc] = 100

    seed = _seed_labels(H, W, 32, 32)
    boundary = img < 500
    result = grow_seeds_watershed(img, seed, polarity="dark", boundary_mask=boundary)

    assert result.max() >= 1
    assert (result[rr, cc] > 0).sum() > 0


def test_watershed_returns_int32():
    H, W = 32, 32
    img = np.ones((H, W), dtype=np.uint16) * 1000
    seed = _seed_labels(H, W, 16, 16)
    result = grow_seeds_watershed(img, seed, polarity="bright")
    assert result.dtype == np.int32
