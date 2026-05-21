from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.analysis.spots.validation import (
    BitDepthError,
    validate_bit_depth,
    validate_diameter,
)


def test_validate_diameter_positive():
    validate_diameter(5.0)  # should not raise


def test_validate_diameter_zero_raises():
    with pytest.raises(ValueError, match="must be > 0"):
        validate_diameter(0.0)


def test_validate_diameter_negative_raises():
    with pytest.raises(ValueError, match="must be > 0"):
        validate_diameter(-1.0)


def test_validate_bit_depth_valid_12bit():
    img = np.zeros((64, 64), dtype=np.uint16)
    validate_bit_depth(img, 12, strict=True)  # max=0 ≤ 4095 — valid


def test_validate_bit_depth_float_raises():
    img = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(BitDepthError, match="integer dtype"):
        validate_bit_depth(img, 12, strict=True)


def test_validate_bit_depth_overflow_strict():
    img = np.full((64, 64), 5000, dtype=np.uint16)
    with pytest.raises(BitDepthError, match="exceeds expected max"):
        validate_bit_depth(img, 12, strict=True)


def test_validate_bit_depth_overflow_warn():
    img = np.full((64, 64), 5000, dtype=np.uint16)
    with pytest.warns(UserWarning):
        validate_bit_depth(img, 12, strict=False)
