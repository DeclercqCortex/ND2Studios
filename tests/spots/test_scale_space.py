from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.analysis.spots.scale_space import (
    dog_response,
    log_response,
    resolve_sigma,
)


def test_resolve_sigma_ratio():
    si, so = resolve_sigma(10.0)
    assert so / si == pytest.approx(1.6, rel=1e-6)


def test_resolve_sigma_fwhm_relationship():
    # FWHM of a Gaussian with std sigma is 2*sqrt(2*ln2)*sigma
    # For typical_diameter_px = FWHM, sigma_in = FWHM / (2*sqrt(2*ln2))
    d = 8.0
    si, _ = resolve_sigma(d)
    fwhm_recovered = si * 2.0 * np.sqrt(2.0 * np.log(2.0))
    assert fwhm_recovered == pytest.approx(d, rel=1e-6)


def test_log_response_shape_and_dtype(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    resp = log_response(img, sigma=4.0)
    assert resp.shape == img.shape
    assert resp.dtype == np.float32


def test_dog_response_shape_and_dtype(gaussian_blobs_12bit):
    img, _ = gaussian_blobs_12bit
    si, so = resolve_sigma(10.0)
    resp = dog_response(img, si, so)
    assert resp.shape == img.shape
    assert resp.dtype == np.float32


def test_log_bright_spot_positive_peak(gaussian_blobs_12bit):
    """A bright Gaussian blob should produce a positive LoG peak near its centre."""
    img, gt = gaussian_blobs_12bit
    blob = gt["blobs"][0]
    r, c = blob["row"], blob["col"]
    d = blob["diameter_px"]
    si, _ = resolve_sigma(float(d))
    resp = log_response(img, si)
    # Peak should be positive and near the blob centre
    assert resp[r, c] > 0
    # The pixel at the known centre should be the local max in a neighbourhood
    neighbourhood = resp[r - 3: r + 4, c - 3: c + 4]
    assert resp[r, c] == pytest.approx(neighbourhood.max(), rel=0.05)


def test_dog_bright_spot_positive_peak(gaussian_blobs_12bit):
    img, gt = gaussian_blobs_12bit
    blob = gt["blobs"][0]
    r, c = blob["row"], blob["col"]
    d = blob["diameter_px"]
    si, so = resolve_sigma(float(d))
    resp = dog_response(img, si, so)
    assert resp[r, c] > 0


def test_dog_and_log_sign_agreement(gaussian_blobs_12bit):
    """DoG and LoG should agree on the sign at every blob centre."""
    img, gt = gaussian_blobs_12bit
    for blob in gt["blobs"]:
        r, c = blob["row"], blob["col"]
        d = float(blob["diameter_px"])
        si, so = resolve_sigma(d)
        log_val = log_response(img, si)[r, c]
        dog_val = dog_response(img, si, so)[r, c]
        assert (log_val > 0) == (dog_val > 0), f"Sign disagreement at blob {blob}"
