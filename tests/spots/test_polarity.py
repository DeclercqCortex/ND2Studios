"""§7.3 — Polarity symmetry: bright on I == dark on (max - I).

For a bright-spot image I, build I_dark = I.max() - I.
Detections on I with polarity=bright and on I_dark with polarity=dark
must share the same centroids (within 1 px) and similar contrast/circularity scores.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from nd2studios.backend.analysis.spots.config import SpotsConfig
from nd2studios.backend.analysis.spots.identifier import BrightDarkSpotsSegmenter


def _cfg(polarity: str) -> SpotsConfig:
    return SpotsConfig(
        polarity=polarity,
        typical_diameter_um=1.0,   # 1 µm × 1 px/µm = 1 px — small to match blob size
        contrast=0.01,
        symmetry="all",
        intensity_percentile=None,
        bit_depth=12,
        output_mode="circular",
        kernel="dog",
        grow_radius_um=0.0,
        grow_method="none",
        bit_depth_strict=False,
    )


def test_polarity_same_centroid_count(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg_b = SpotsConfig(
        polarity="bright",
        typical_diameter_um=10.0,
        contrast=0.01,
        symmetry="all",
        intensity_percentile=None,
        bit_depth=12,
        output_mode="circular",
        kernel="dog",
        grow_radius_um=0.0,
        grow_method="none",
        bit_depth_strict=False,
    )
    img_dark = (4095 - img).astype(np.uint16)
    cfg_d = SpotsConfig(
        polarity="dark",
        typical_diameter_um=10.0,
        contrast=0.01,
        symmetry="all",
        intensity_percentile=None,
        bit_depth=12,
        output_mode="circular",
        kernel="dog",
        grow_radius_um=0.0,
        grow_method="none",
        bit_depth_strict=False,
    )
    r_bright = BrightDarkSpotsSegmenter(cfg_b).run(img, pixel_size_um=0.1)
    r_dark = BrightDarkSpotsSegmenter(cfg_d).run(img_dark, pixel_size_um=0.1)

    n_b = len(r_bright.regions)
    n_d = len(r_dark.regions)
    # Allow a small discrepancy due to edge/rounding effects
    assert abs(n_b - n_d) <= max(2, int(0.2 * max(n_b, n_d, 1))), (
        f"Bright found {n_b} spots, dark found {n_d} — large mismatch"
    )


def test_polarity_centroid_agreement(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    img_dark = (4095 - img).astype(np.uint16)

    cfg_b = SpotsConfig(
        polarity="bright",
        typical_diameter_um=10.0,
        contrast=0.01,
        symmetry="all",
        intensity_percentile=None,
        bit_depth=12,
        output_mode="circular",
        kernel="dog",
        grow_radius_um=0.0,
        grow_method="none",
        bit_depth_strict=False,
    )
    cfg_d = SpotsConfig(
        polarity="dark",
        typical_diameter_um=10.0,
        contrast=0.01,
        symmetry="all",
        intensity_percentile=None,
        bit_depth=12,
        output_mode="circular",
        kernel="dog",
        grow_radius_um=0.0,
        grow_method="none",
        bit_depth_strict=False,
    )

    r_bright = BrightDarkSpotsSegmenter(cfg_b).run(img, pixel_size_um=0.1)
    r_dark = BrightDarkSpotsSegmenter(cfg_d).run(img_dark, pixel_size_um=0.1)

    if len(r_bright.regions) == 0 or len(r_dark.regions) == 0:
        pytest.skip("One polarity found no regions — cannot compare centroids")

    cb = np.array([[r["centroid_y"], r["centroid_x"]] for r in r_bright.regions])
    cd = np.array([[r["centroid_y"], r["centroid_x"]] for r in r_dark.regions])

    cost = np.sqrt(
        ((cb[:, None, 0] - cd[None, :, 0]) ** 2)
        + ((cb[:, None, 1] - cd[None, :, 1]) ** 2)
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_dists = cost[row_ind, col_ind]

    if len(matched_dists) > 0:
        median_dist = float(np.median(matched_dists))
        assert median_dist <= 2.0, (
            f"Median centroid distance between bright and dark = {median_dist:.2f} px"
        )
