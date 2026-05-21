from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_12bit_image() -> np.ndarray:
    """A 256×256 12-bit image with two known dark blobs."""
    rng = np.random.default_rng(42)
    img = rng.integers(800, 2400, size=(256, 256), dtype=np.uint16)
    img[40:80, 40:80] = rng.integers(5, 30, size=(40, 40), dtype=np.uint16)
    img[150:170, 200:230] = rng.integers(10, 40, size=(20, 30), dtype=np.uint16)
    return img


@pytest.fixture
def synthetic_12bit_image_3d() -> np.ndarray:
    """An 8×128×128 12-bit volume with one embedded dark blob."""
    rng = np.random.default_rng(42)
    img = rng.integers(800, 2400, size=(8, 128, 128), dtype=np.uint16)
    img[2:6, 30:60, 30:60] = rng.integers(5, 30, size=(4, 30, 30), dtype=np.uint16)
    return img
