from __future__ import annotations

from nd2studios.backend.analysis.histothresh.validation import (
    BIT_DEPTH_MAX,
    BitDepthError,
    infer_bit_depth,
    validate_bit_depth,
)

__all__ = [
    "validate_bit_depth",
    "infer_bit_depth",
    "BIT_DEPTH_MAX",
    "BitDepthError",
    "validate_diameter",
]


def validate_diameter(typical_diameter_um: float) -> None:
    """Raise ValueError if typical_diameter_um is not strictly positive."""
    if typical_diameter_um <= 0:
        raise ValueError(f"typical_diameter_um must be > 0, got {typical_diameter_um}")
