from __future__ import annotations

import warnings
import numpy as np

BIT_DEPTH_MAX: dict[int, int] = {8: 255, 10: 1023, 12: 4095, 14: 16383, 16: 65535}


class BitDepthError(ValueError):
    """Raised when image data does not match the expected bit depth."""


def validate_bit_depth(
    image: np.ndarray,
    bit_depth: int = 12,
    *,
    strict: bool = True,
) -> None:
    """Validate that `image` contains values consistent with `bit_depth`.

    Raises BitDepthError for non-integer dtype, unsupported bit depth, or
    (when strict=True) values exceeding the expected maximum.
    """
    if bit_depth not in BIT_DEPTH_MAX:
        raise BitDepthError(
            f"Unsupported bit depth {bit_depth}. Supported: {sorted(BIT_DEPTH_MAX)}"
        )
    if not np.issubdtype(image.dtype, np.integer):
        raise BitDepthError(
            f"Expected integer dtype, got {image.dtype}. "
            "Float inputs must be explicitly converted; this tool operates on raw counts."
        )

    expected_max = BIT_DEPTH_MAX[bit_depth]
    actual_max = int(image.max())

    if actual_max > expected_max:
        msg = (
            f"Image max value {actual_max} exceeds expected max {expected_max} "
            f"for {bit_depth}-bit data. Data may have been rescaled to fill its "
            f"container ({image.dtype}). Thresholds set in {bit_depth}-bit units "
            "will be incorrect on this input."
        )
        if strict:
            raise BitDepthError(msg)
        warnings.warn(msg, stacklevel=2)

    if actual_max < expected_max * 0.05:
        warnings.warn(
            f"Image max {actual_max} is <5% of {bit_depth}-bit range. "
            "Verify the bit depth setting; image may be lower-bit than declared.",
            stacklevel=2,
        )


def infer_bit_depth(image: np.ndarray) -> int:
    """Best-effort bit-depth inference from observed value range.

    Use only as a fallback; explicit configuration is always preferred.
    """
    actual_max = int(image.max())
    for bd in sorted(BIT_DEPTH_MAX):
        if actual_max <= BIT_DEPTH_MAX[bd]:
            return bd
    raise BitDepthError(f"Image max {actual_max} exceeds all supported bit depths.")
