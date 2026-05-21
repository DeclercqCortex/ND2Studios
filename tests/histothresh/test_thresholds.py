from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.analysis.histothresh.thresholds import (
    threshold_hysteresis,
    threshold_percentile,
    threshold_single,
)


def test_below() -> None:
    img = np.array([[10, 50, 100]], dtype=np.uint16)
    mask = threshold_single(img, "below", low=50)
    assert mask.tolist() == [[True, True, False]]


def test_above() -> None:
    img = np.array([[10, 50, 100]], dtype=np.uint16)
    mask = threshold_single(img, "above", high=50)
    assert mask.tolist() == [[False, True, True]]


def test_between() -> None:
    img = np.array([[10, 50, 100, 200]], dtype=np.uint16)
    mask = threshold_single(img, "between", low=40, high=120)
    assert mask.tolist() == [[False, True, True, False]]


def test_outside() -> None:
    img = np.array([[10, 50, 100, 200]], dtype=np.uint16)
    mask = threshold_single(img, "outside", low=40, high=120)
    assert mask.tolist() == [[True, False, False, True]]


def test_between_rejects_inverted_bounds() -> None:
    img = np.array([[10, 50]], dtype=np.uint16)
    with pytest.raises(ValueError):
        threshold_single(img, "between", low=100, high=10)


def test_hysteresis_below_basic() -> None:
    img = np.array(
        [
            [200, 200, 200, 200, 200],
            [200,  20,  60,  60, 200],
            [200,  60,  60,  60, 200],
            [200, 200, 200,  60, 200],
            [200, 200, 200, 200, 200],
        ],
        dtype=np.uint16,
    )
    mask = threshold_hysteresis(img, "below", strict=30, permissive=80)
    assert mask[1, 1]   # core pixel
    assert mask[1, 2]   # connected permissive pixel
    assert mask[3, 3]   # connected via chain
    assert not mask[0, 0]


def test_hysteresis_below_isolated_perm_excluded() -> None:
    """A pixel below permissive but with no core connection is excluded."""
    img = np.full((5, 5), 200, dtype=np.uint16)
    img[1, 1] = 20   # core
    img[1, 2] = 60   # connected permissive
    img[4, 4] = 60   # isolated permissive — must NOT be included
    mask = threshold_hysteresis(img, "below", strict=30, permissive=80)
    assert mask[1, 1]
    assert mask[1, 2]
    assert not mask[4, 4]


def test_hysteresis_above_symmetric() -> None:
    img = np.full((5, 5), 50, dtype=np.uint16)
    img[2, 2] = 4000
    img[2, 3] = 2000
    mask = threshold_hysteresis(img, "above", strict=3000, permissive=1500)
    assert mask[2, 2]
    assert mask[2, 3]
    assert not mask[0, 0]


def test_hysteresis_rejects_wrong_direction() -> None:
    img = np.zeros((3, 3), dtype=np.uint16)
    with pytest.raises(ValueError):
        threshold_hysteresis(img, "between", strict=10, permissive=20)  # type: ignore[arg-type]


def test_percentile_below_with_sanity_floor() -> None:
    img = np.full((10, 10), 1000, dtype=np.uint16)
    img[0, 0] = 800
    # P1 resolves near 800; sanity_floor=100 clips it, leaving nothing dark
    mask = threshold_percentile(
        img, "below", percentile_low=1.0, sanity_floor=100
    )
    assert not mask.any()
