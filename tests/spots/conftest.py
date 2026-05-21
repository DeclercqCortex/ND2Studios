from __future__ import annotations

import numpy as np
import pytest
from skimage.draw import disk


_RNG = np.random.default_rng(42)

_BLOBS = [
    # (row, col, diameter_px, peak_intensity)
    (64, 64, 10, 3500),
    (128, 180, 8, 3000),
    (192, 96, 12, 3800),
]


def _make_bright_image(H: int = 256, W: int = 256) -> tuple[np.ndarray, list[dict]]:
    """Create a 12-bit image with Gaussian bright blobs on a low-level background."""
    rng = np.random.default_rng(42)
    img = rng.integers(100, 500, (H, W), dtype=np.int32)
    Y, X = np.ogrid[:H, :W]
    gt: list[dict] = []
    for r, c, d, peak in _BLOBS:
        sigma = d / 2.355
        blob = np.exp(-((Y - r) ** 2 + (X - c) ** 2) / (2.0 * sigma ** 2))
        img = img + (blob * peak).astype(np.int32)
        gt.append({"row": r, "col": c, "diameter_px": d, "peak": int(peak)})
    img = np.clip(img, 0, 4095).astype(np.uint16)
    return img, gt


@pytest.fixture(scope="session")
def gaussian_blobs_12bit() -> tuple[np.ndarray, dict]:
    """12-bit image with three known Gaussian bright blobs + ground-truth dict."""
    img, gt = _make_bright_image()
    return img, {"blobs": gt}


@pytest.fixture(scope="session")
def gaussian_blobs_dark_12bit(gaussian_blobs_12bit) -> tuple[np.ndarray, dict]:
    """Inverted-polarity counterpart: bright blobs become dark pits."""
    img, gt = gaussian_blobs_12bit
    return (4095 - img).astype(np.uint16), gt
