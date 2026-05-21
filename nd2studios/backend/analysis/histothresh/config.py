from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ThresholdConfig:
    """Configuration for a single segmentation run.

    All intensity values are in raw integer counts in [0, 2**bit_depth - 1].
    """

    method: Literal["single", "hysteresis", "percentile", "relative"] = "hysteresis"
    direction: Literal["below", "above", "between", "outside"] = "below"

    # Single / hysteresis (raw integer counts)
    low: int | None = None
    high: int | None = None
    strict: int | None = None
    permissive: int | None = None

    # Percentile (0–100)
    percentile_low: float | None = None
    percentile_high: float | None = None
    sanity_floor: int | None = None
    sanity_ceiling: int | None = None

    # Relative (fraction of reference median)
    fraction_low: float | None = None
    fraction_high: float | None = None

    # Bit depth
    bit_depth: int = 12
    bit_depth_strict: bool = True

    # Spatial constraints
    min_area: int = 100
    opening_radius: int = 1
    closing_radius: int = 2
    min_hole_size: int = 50

    # Optional homogeneity gate
    homogeneity_gate: bool = False
    homogeneity_window: int = 7
    homogeneity_std_max: float = 20.0

    def __post_init__(self) -> None:
        if self.method == "single":
            if self.direction == "below" and self.low is None:
                raise ValueError("method=single, direction=below requires `low`")
            if self.direction == "above" and self.high is None:
                raise ValueError("method=single, direction=above requires `high`")
            if self.direction in ("between", "outside"):
                if self.low is None or self.high is None:
                    raise ValueError(
                        f"method=single, direction={self.direction} requires both `low` and `high`"
                    )
        elif self.method == "hysteresis":
            if self.direction not in ("below", "above"):
                raise ValueError(
                    f"hysteresis only supports below/above, got {self.direction}"
                )
            if self.strict is None or self.permissive is None:
                raise ValueError("method=hysteresis requires `strict` and `permissive`")
        elif self.method == "percentile":
            if self.direction == "below" and self.percentile_low is None:
                raise ValueError(
                    "method=percentile, direction=below requires `percentile_low`"
                )
            if self.direction == "above" and self.percentile_high is None:
                raise ValueError(
                    "method=percentile, direction=above requires `percentile_high`"
                )
            if self.direction in ("between", "outside"):
                if self.percentile_low is None or self.percentile_high is None:
                    raise ValueError(
                        f"method=percentile, direction={self.direction} requires both percentiles"
                    )
        elif self.method == "relative":
            if self.direction == "below" and self.fraction_low is None:
                raise ValueError(
                    "method=relative, direction=below requires `fraction_low`"
                )
            if self.direction == "above" and self.fraction_high is None:
                raise ValueError(
                    "method=relative, direction=above requires `fraction_high`"
                )
