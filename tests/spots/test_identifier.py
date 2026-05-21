from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.analysis.spots.config import SpotsConfig
from nd2studios.backend.analysis.spots.identifier import BrightDarkSpotsSegmenter, SpotsResult


def _default_cfg(**overrides) -> SpotsConfig:
    defaults = dict(
        polarity="bright",
        typical_diameter_um=10.0,
        contrast=0.01,
        symmetry="all",
        intensity_percentile=None,
        bit_depth=12,
        grow_radius_um=0.0,
        grow_method="none",
        output_mode="circular",
        kernel="dog",
        bit_depth_strict=False,
    )
    defaults.update(overrides)
    return SpotsConfig(**defaults)


# ---------------------------------------------------------------------------
# SpotsResult contract
# ---------------------------------------------------------------------------

def test_result_type(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    assert isinstance(result, SpotsResult)


def test_result_label_shape(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    assert result.labels.shape == img.shape
    assert result.labels.dtype == np.int32


def test_result_mask_bool(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    assert result.mask.dtype == bool
    assert result.mask.shape == img.shape


def test_result_mask_label_consistent(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    assert np.array_equal(result.mask, result.labels > 0)


def test_result_regions_keys(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    required = {"label_id", "area_px", "area_um2", "centroid_y", "centroid_x",
                "mean_intensity", "diameter_px", "contrast_score", "circularity"}
    for row in result.regions:
        assert required <= set(row.keys()), f"Missing keys: {required - set(row.keys())}"


def test_result_detects_known_blobs(gaussian_blobs_12bit):
    img, gt = gaussian_blobs_12bit
    cfg = _default_cfg(contrast=0.01, symmetry="all")
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    assert len(result.regions) >= 1


def test_result_provenance_keys(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    for key in ("timestamp", "input_shape", "input_dtype", "input_hash"):
        assert key in result.provenance, f"Missing provenance key: {key}"


def test_result_provenance_hash_length(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    assert len(result.provenance["input_hash"]) == 16


def test_empty_image_returns_zero_regions():
    """A flat uniform image has no spots."""
    img = np.full((64, 64), 2048, dtype=np.uint16)
    cfg = _default_cfg(contrast=0.5)  # high contrast gate
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    assert len(result.regions) == 0
    assert result.labels.max() == 0


def test_region_output_mode(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg(output_mode="region", contrast=0.01, symmetry="all")
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    # Region mode should still produce a valid label array
    assert result.labels.shape == img.shape
    assert result.labels.dtype == np.int32


def test_area_um2_nonzero_with_pixel_size(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.065)
    for row in result.regions:
        assert row["area_um2"] > 0


def test_area_um2_zero_without_pixel_size(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=None)
    for row in result.regions:
        assert row["area_um2"] == 0.0
