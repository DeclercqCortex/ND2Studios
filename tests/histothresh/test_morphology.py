from __future__ import annotations

import numpy as np

from nd2studios.backend.analysis.histothresh.morphology import apply_spatial_constraints


def test_min_area_filter() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5, 5] = True           # single pixel — should be removed
    mask[10:18, 10:18] = True   # 64 px — should survive
    out = apply_spatial_constraints(
        mask, min_area=10, opening_radius=0, closing_radius=0, min_hole_size=0
    )
    assert not out[5, 5]
    assert out[12, 12]


def test_hole_fill() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    mask[4:6, 4:6] = False  # 4-pixel interior hole
    out = apply_spatial_constraints(
        mask, min_area=0, opening_radius=0, closing_radius=0, min_hole_size=10
    )
    assert out[5, 5]
