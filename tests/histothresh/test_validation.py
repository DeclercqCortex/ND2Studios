from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.analysis.histothresh.validation import (
    BitDepthError,
    infer_bit_depth,
    validate_bit_depth,
)


def test_valid_12bit(synthetic_12bit_image: np.ndarray) -> None:
    validate_bit_depth(synthetic_12bit_image, bit_depth=12)


def test_rejects_overflow() -> None:
    img = np.array([[0, 5000]], dtype=np.uint16)  # > 4095
    with pytest.raises(BitDepthError):
        validate_bit_depth(img, bit_depth=12, strict=True)


def test_warns_on_overflow_when_not_strict() -> None:
    img = np.array([[0, 5000]], dtype=np.uint16)
    with pytest.warns(UserWarning):
        validate_bit_depth(img, bit_depth=12, strict=False)


def test_rejects_float() -> None:
    img = np.array([[0.0, 1.0]], dtype=np.float32)
    with pytest.raises(BitDepthError):
        validate_bit_depth(img, bit_depth=12)


def test_rejects_unsupported_bit_depth() -> None:
    img = np.array([[0, 100]], dtype=np.uint16)
    with pytest.raises(BitDepthError):
        validate_bit_depth(img, bit_depth=11)


def test_infer() -> None:
    img8 = np.array([[0, 200]], dtype=np.uint8)
    assert infer_bit_depth(img8) == 8

    img12 = np.array([[0, 4000]], dtype=np.uint16)
    assert infer_bit_depth(img12) == 12
