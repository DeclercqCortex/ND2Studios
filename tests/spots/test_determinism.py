"""§7.6 — Determinism and provenance integrity.

Two consecutive run() calls on the same image must return identical labels,
centroids, and provenance.input_hash. The hash must change when the image changes.
"""
from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.analysis.spots.config import SpotsConfig
from nd2studios.backend.analysis.spots.identifier import BrightDarkSpotsSegmenter


def _default_cfg() -> SpotsConfig:
    return SpotsConfig(
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


def test_identical_labels_on_repeat(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    seg = BrightDarkSpotsSegmenter(cfg)
    r1 = seg.run(img, pixel_size_um=0.1)
    r2 = seg.run(img, pixel_size_um=0.1)
    assert np.array_equal(r1.labels, r2.labels)


def test_identical_centroids_on_repeat(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    seg = BrightDarkSpotsSegmenter(cfg)
    r1 = seg.run(img, pixel_size_um=0.1)
    r2 = seg.run(img, pixel_size_um=0.1)
    assert np.array_equal(r1.centroids, r2.centroids)


def test_identical_hash_on_repeat(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    seg = BrightDarkSpotsSegmenter(cfg)
    r1 = seg.run(img, pixel_size_um=0.1)
    r2 = seg.run(img, pixel_size_um=0.1)
    assert r1.provenance["input_hash"] == r2.provenance["input_hash"]


def test_hash_changes_on_perturbation(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    seg = BrightDarkSpotsSegmenter(cfg)

    r1 = seg.run(img, pixel_size_um=0.1)

    img2 = img.copy()
    img2[0, 0] = (img2[0, 0] + 1) % 4096  # single-pixel perturbation
    r2 = seg.run(img2, pixel_size_um=0.1)

    assert r1.provenance["input_hash"] != r2.provenance["input_hash"]


def test_provenance_timestamp_present(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    ts = result.provenance["timestamp"]
    assert isinstance(ts, str)
    assert "T" in ts  # ISO 8601 format


def test_provenance_input_shape(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    cfg = _default_cfg()
    result = BrightDarkSpotsSegmenter(cfg).run(img, pixel_size_um=0.1)
    assert result.provenance["input_shape"] == img.shape
