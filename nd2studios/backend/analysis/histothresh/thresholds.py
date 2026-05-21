from __future__ import annotations

from typing import Literal
import numpy as np
from skimage.filters import apply_hysteresis_threshold

from .histogram import Histogram, compute_histogram

Direction = Literal["below", "above", "between", "outside"]


def threshold_single(
    image: np.ndarray,
    direction: Direction,
    *,
    low: int | None = None,
    high: int | None = None,
) -> np.ndarray:
    """Apply a single (non-hysteretic) threshold.

    Parameter requirements by direction:
      below   : low      → mask = image <= low
      above   : high     → mask = image >= high
      between : low,high → mask = low <= image <= high
      outside : low,high → mask = (image < low) | (image > high)
    """
    if direction == "below":
        if low is None:
            raise ValueError("`below` requires `low`")
        return image <= low
    if direction == "above":
        if high is None:
            raise ValueError("`above` requires `high`")
        return image >= high
    if direction == "between":
        if low is None or high is None:
            raise ValueError("`between` requires both `low` and `high`")
        if high < low:
            raise ValueError(f"`high` ({high}) must be >= `low` ({low})")
        return (image >= low) & (image <= high)
    if direction == "outside":
        if low is None or high is None:
            raise ValueError("`outside` requires both `low` and `high`")
        if high < low:
            raise ValueError(f"`high` ({high}) must be >= `low` ({low})")
        return (image < low) | (image > high)
    raise ValueError(f"Unknown direction: {direction!r}")


def threshold_hysteresis(
    image: np.ndarray,
    direction: Literal["below", "above"],
    *,
    strict: int,
    permissive: int,
) -> np.ndarray:
    """Hysteresis threshold for unidirectional cuts.

    For `below`: strict <= permissive. Pixels are included if they are
    <= permissive AND connected to a region of pixels <= strict.
    For `above`: strict >= permissive. Pixels are included if they are
    >= permissive AND connected to a region of pixels >= strict.

    scikit-image's apply_hysteresis_threshold uses (low, high) where
    low <= high and selects pixels above `low` connected to pixels above `high`.
    We adapt for both directions.
    """
    if direction == "below":
        if strict > permissive:
            raise ValueError(
                f"For `below`, strict ({strict}) must be <= permissive ({permissive})"
            )
        inverted = image.max() - image
        return apply_hysteresis_threshold(
            inverted,
            low=image.max() - permissive,
            high=image.max() - strict,
        )
    if direction == "above":
        if strict < permissive:
            raise ValueError(
                f"For `above`, strict ({strict}) must be >= permissive ({permissive})"
            )
        return apply_hysteresis_threshold(image, low=permissive, high=strict)
    raise ValueError(f"Hysteresis only supports 'below' or 'above', got {direction!r}")


def threshold_percentile(
    image: np.ndarray,
    direction: Direction,
    *,
    bit_depth: int = 12,
    percentile_low: float | None = None,
    percentile_high: float | None = None,
    sanity_floor: int | None = None,
    sanity_ceiling: int | None = None,
) -> np.ndarray:
    """Threshold by percentile of the image's histogram.

    Maps percentiles to integer intensity values, then dispatches to
    `threshold_single`. `sanity_floor` clips the resolved `low` value to
    prevent calling bright pixels "dark" on images without genuine dark regions.
    `sanity_ceiling` is the symmetric guard for the bright end.
    """
    hist = compute_histogram(image, bit_depth=bit_depth)
    low = hist.percentile(percentile_low) if percentile_low is not None else None
    high = hist.percentile(percentile_high) if percentile_high is not None else None

    if low is not None and sanity_floor is not None:
        low = min(low, sanity_floor)
    if high is not None and sanity_ceiling is not None:
        high = max(high, sanity_ceiling)

    return threshold_single(image, direction, low=low, high=high)


def threshold_relative(
    image: np.ndarray,
    direction: Direction,
    *,
    reference_mask: np.ndarray | None = None,
    fraction_low: float | None = None,
    fraction_high: float | None = None,
) -> np.ndarray:
    """Threshold relative to the median intensity inside `reference_mask`.

    Resolves `low = median * fraction_low` and/or `high = median * fraction_high`,
    then dispatches to `threshold_single`.
    """
    if reference_mask is None:
        reference = image
    else:
        reference = image[reference_mask]
    median = float(np.median(reference))
    low = int(round(median * fraction_low)) if fraction_low is not None else None
    high = int(round(median * fraction_high)) if fraction_high is not None else None
    return threshold_single(image, direction, low=low, high=high)
