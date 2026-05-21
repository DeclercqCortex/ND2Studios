from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class SpotsConfig:
    """Configuration for a single BrightDarkSpotsSegmenter run.

    All spatial values are in micrometres. Intensity thresholds are
    bit-depth-independent (normalised response units in [0, 1]).
    """

    polarity: Literal["bright", "dark"] = "bright"
    typical_diameter_um: float = 5.0
    contrast: float = 0.05            # normalised LoG/DoG response threshold
    symmetry: Literal["all", "more", "medium", "less"] = "medium"

    # LUT-histogram-based intensity gate (delegates to histothresh.histogram)
    intensity_percentile: float | None = 50.0   # bright: floor; dark: ceiling
    bit_depth: int = 12

    # Grow objects
    grow_radius_um: float = 0.0
    grow_method: Literal["dilation", "watershed", "none"] = "none"

    # Output mode
    output_mode: Literal["circular", "region"] = "region"

    # Background mask
    background_mask: bool = False

    # Engine
    kernel: Literal["log", "dog"] = "dog"
    bit_depth_strict: bool = False    # match histothresh default

    def __post_init__(self) -> None:
        if self.typical_diameter_um <= 0:
            raise ValueError("typical_diameter_um must be > 0")
        if not (0.0 <= self.contrast <= 1.0):
            raise ValueError("contrast must be in [0, 1] (normalised units)")
        if self.symmetry not in ("all", "more", "medium", "less"):
            raise ValueError(f"unknown symmetry setting {self.symmetry!r}")
        if self.intensity_percentile is not None and not (
            0.0 <= self.intensity_percentile <= 100.0
        ):
            raise ValueError("intensity_percentile must be in [0, 100]")
        if self.grow_method != "none" and self.grow_radius_um < 0:
            raise ValueError("grow_radius_um must be >= 0")
