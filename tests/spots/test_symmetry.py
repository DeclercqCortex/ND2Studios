from __future__ import annotations

import math

import numpy as np
import pytest
from skimage.draw import disk

from nd2studios.backend.analysis.spots.symmetry import (
    SYMMETRY_FLOOR,
    circularity,
    filter_by_symmetry,
)
from nd2studios.backend.analysis.spots.config import SpotsConfig
from nd2studios.backend.analysis.spots.identifier import BrightDarkSpotsSegmenter


# ---------------------------------------------------------------------------
# circularity()
# ---------------------------------------------------------------------------

def test_circularity_perfect_circle():
    # For a true circle: area = pi*r^2, perimeter = 2*pi*r
    # circularity = 4*pi*(pi*r^2) / (2*pi*r)^2 = 4*pi^2*r^2 / (4*pi^2*r^2) = 1
    area = math.pi * 10 ** 2
    perim = 2 * math.pi * 10
    assert circularity(area, perim) == pytest.approx(1.0, rel=1e-6)


def test_circularity_zero_perimeter():
    assert circularity(100.0, 0.0) == 0.0


def test_circularity_clipped_to_one():
    # Extremely small perimeter relative to area → formula > 1, should clip
    assert circularity(1000.0, 1.0) == 1.0


def test_circularity_elongated_shape():
    # A very elongated rectangle has low circularity
    # area = 2*100 = 200, perimeter ≈ 2*(2+100) = 204
    c = circularity(200.0, 204.0)
    assert c < 0.5


# ---------------------------------------------------------------------------
# SYMMETRY_FLOOR ordering
# ---------------------------------------------------------------------------

def test_symmetry_floor_ordering():
    """The floor values must be non-decreasing from 'all' to 'less'."""
    settings = ["all", "more", "medium", "less"]
    floors = [SYMMETRY_FLOOR[s] for s in settings]
    assert floors == sorted(floors)


# ---------------------------------------------------------------------------
# filter_by_symmetry()
# ---------------------------------------------------------------------------

def _circle_mask(H: int, W: int, r: int, c: int, radius: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    rr, cc = disk((r, c), radius, shape=(H, W))
    mask[rr, cc] = True
    return mask


def test_filter_all_keeps_everything():
    mask = _circle_mask(128, 128, 64, 64, 20)
    out = filter_by_symmetry(mask, setting="all")
    assert out.sum() == mask.sum()


def test_filter_medium_keeps_circle():
    mask = _circle_mask(128, 128, 64, 64, 20)
    out = filter_by_symmetry(mask, setting="medium")
    # A rasterised disk is sufficiently circular to pass the medium threshold
    assert out.any()


def test_filter_less_rejects_elongated_rect():
    # Create a very elongated rectangle (1×50 pixels) — low circularity
    mask = np.zeros((128, 128), dtype=bool)
    mask[60, 30:80] = True  # 50-pixel long, 1-pixel tall strip
    out = filter_by_symmetry(mask, setting="less")
    assert not out.any(), "Elongated strip should be rejected by 'less' symmetry gate"


# ---------------------------------------------------------------------------
# Monotonicity: more restrictive setting → fewer objects
# ---------------------------------------------------------------------------

def test_symmetry_monotonicity(gaussian_blobs_12bit):
    """Tighter symmetry gates must yield monotonically non-increasing object counts."""
    img, _ = gaussian_blobs_12bit
    cfg_base = dict(
        typical_diameter_um=10.0,
        contrast=0.01,
        intensity_percentile=None,
        bit_depth=12,
        kernel="dog",
        output_mode="circular",
        grow_method="none",
        grow_radius_um=0.0,
        polarity="bright",
        bit_depth_strict=False,
    )
    counts = []
    for sym in ("all", "more", "medium", "less"):
        cfg = SpotsConfig(**{**cfg_base, "symmetry": sym})
        result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
        counts.append(len(result.regions))
    assert counts == sorted(counts, reverse=True), (
        f"Counts not monotonically non-increasing: {counts}"
    )
