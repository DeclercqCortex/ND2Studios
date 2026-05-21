from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.analysis.histothresh.config import ThresholdConfig
from nd2studios.backend.analysis.histothresh.identifier import HistogramThresholdSegmenter


def test_end_to_end_2d(synthetic_12bit_image: np.ndarray) -> None:
    cfg = ThresholdConfig(
        method="hysteresis",
        direction="below",
        strict=50,
        permissive=100,
        bit_depth=12,
        min_area=20,
    )
    result = HistogramThresholdSegmenter(cfg).run(synthetic_12bit_image)
    assert len(result.regions) >= 2   # the two injected dark blobs
    assert result.mask.dtype == bool
    assert result.labels.max() >= 2
    assert "input_hash" in result.provenance


def test_end_to_end_single_method(synthetic_12bit_image: np.ndarray) -> None:
    cfg = ThresholdConfig(
        method="single",
        direction="below",
        low=80,
        bit_depth=12,
        min_area=0,
    )
    result = HistogramThresholdSegmenter(cfg).run(synthetic_12bit_image)
    assert result.mask.dtype == bool
    assert result.mask.any()


def test_percentile_method(synthetic_12bit_image: np.ndarray) -> None:
    cfg = ThresholdConfig(
        method="percentile",
        direction="below",
        percentile_low=5.0,
        bit_depth=12,
        min_area=0,
        opening_radius=0,
        closing_radius=0,
    )
    result = HistogramThresholdSegmenter(cfg).run(synthetic_12bit_image)
    # Roughly 5% of pixels should be masked
    assert result.mask.mean() == pytest.approx(0.05, abs=0.02)


def test_determinism(synthetic_12bit_image: np.ndarray) -> None:
    cfg = ThresholdConfig(
        method="hysteresis",
        direction="below",
        strict=50,
        permissive=100,
        bit_depth=12,
        min_area=20,
    )
    seg = HistogramThresholdSegmenter(cfg)
    r1 = seg.run(synthetic_12bit_image)
    r2 = seg.run(synthetic_12bit_image)
    assert np.array_equal(r1.mask, r2.mask)
    assert np.array_equal(r1.labels, r2.labels)


def test_provenance_recorded(synthetic_12bit_image: np.ndarray) -> None:
    cfg = ThresholdConfig(
        method="single",
        direction="below",
        low=80,
        bit_depth=12,
    )
    result = HistogramThresholdSegmenter(cfg).run(synthetic_12bit_image)
    assert result.threshold_used["method"] == "single"
    assert result.threshold_used["low"] == 80
    assert "timestamp" in result.provenance
