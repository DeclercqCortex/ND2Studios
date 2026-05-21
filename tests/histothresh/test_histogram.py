from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.analysis.histothresh.histogram import compute_histogram


def test_full_lut_range_12bit() -> None:
    img = np.array([[0, 100, 4095]], dtype=np.uint16)
    hist = compute_histogram(img, bit_depth=12)
    assert len(hist.counts) == 4096
    assert hist.counts[0] == 1
    assert hist.counts[100] == 1
    assert hist.counts[4095] == 1
    assert hist.total_pixels == 3


def test_percentile() -> None:
    img = np.tile(np.arange(100, dtype=np.uint16), (1, 1))
    hist = compute_histogram(img, bit_depth=12)
    p50 = hist.percentile(50)
    assert 49 <= p50 <= 50


def test_cumulative_monotonic(synthetic_12bit_image: np.ndarray) -> None:
    hist = compute_histogram(synthetic_12bit_image, bit_depth=12)
    cdf = hist.cumulative()
    assert np.all(np.diff(cdf) >= 0)
    assert cdf[-1] == pytest.approx(1.0)
