from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.analysis.spots.detection import find_extrema, h_transform_seeds
from nd2studios.backend.analysis.spots.scale_space import dog_response, resolve_sigma


def _normalise(resp: np.ndarray) -> np.ndarray:
    m = float(np.abs(resp).max())
    return resp / max(m, 1e-8)


def test_find_extrema_detects_bright_blobs(gaussian_blobs_12bit):
    img, gt = gaussian_blobs_12bit
    d_px = 10.0
    si, so = resolve_sigma(d_px)
    resp = _normalise(dog_response(img, si, so))
    coords = find_extrema(resp, polarity="bright", min_distance=5, contrast=0.01)
    assert coords.shape[1] == 2
    # Each known blob should have at least one detected centroid within d/2 pixels
    for blob in gt["blobs"]:
        br, bc = blob["row"], blob["col"]
        if len(coords) > 0:
            dists = np.hypot(coords[:, 0] - br, coords[:, 1] - bc)
            assert dists.min() <= d_px / 2, f"No detection near blob at ({br},{bc})"


def test_find_extrema_contrast_gate():
    """Very high contrast threshold should reject all peaks."""
    rng = np.random.default_rng(0)
    resp = _normalise(rng.standard_normal((64, 64)).astype(np.float32))
    coords = find_extrema(resp, polarity="bright", min_distance=3, contrast=2.0)
    assert len(coords) == 0


def test_find_extrema_dark_polarity(gaussian_blobs_dark_12bit):
    img, gt = gaussian_blobs_dark_12bit
    d_px = 10.0
    si, so = resolve_sigma(d_px)
    resp = _normalise(dog_response(img, si, so))
    coords = find_extrema(resp, polarity="dark", min_distance=5, contrast=0.01)
    assert len(coords) >= 1


def test_find_extrema_returns_array_type(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    si, so = resolve_sigma(10.0)
    resp = _normalise(dog_response(img, si, so))
    coords = find_extrema(resp, polarity="bright", min_distance=5, contrast=0.01)
    assert coords.ndim == 2
    assert coords.shape[1] == 2


def test_find_extrema_intensity_mask_restricts(gaussian_blobs_12bit):
    """Blocking all pixels with intensity_mask should yield zero detections."""
    img, _ = gaussian_blobs_12bit
    si, so = resolve_sigma(10.0)
    resp = _normalise(dog_response(img, si, so))
    all_false = np.zeros(img.shape, dtype=bool)
    coords = find_extrema(
        resp, polarity="bright", min_distance=5, contrast=0.0,
        intensity_mask=all_false,
    )
    assert len(coords) == 0


def test_h_transform_seeds_bright(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    seeds = h_transform_seeds(img, polarity="bright", h=500.0)
    assert seeds.dtype == bool or seeds.dtype == np.bool_
    assert seeds.shape == img.shape
    assert seeds.any(), "h_maxima should detect at least one bright seed"


def test_h_transform_seeds_dark(gaussian_blobs_dark_12bit):
    img, _ = gaussian_blobs_dark_12bit
    seeds = h_transform_seeds(img, polarity="dark", h=500.0)
    assert seeds.shape == img.shape
    assert seeds.any(), "h_minima should detect at least one dark seed"
