from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
from skimage.measure import label, regionprops_table
from skimage.segmentation import clear_border

from .config import ThresholdConfig
from .histogram import Histogram, compute_histogram
from .morphology import apply_spatial_constraints, homogeneity_gate
from .thresholds import (
    threshold_hysteresis,
    threshold_percentile,
    threshold_relative,
    threshold_single,
)
from .validation import validate_bit_depth


@dataclass
class SegmentationResult:
    """Output from a single-frame HistogramThresholdSegmenter.run() call."""

    mask: np.ndarray
    labels: np.ndarray
    regions: list[dict[str, Any]]
    histogram: Histogram
    threshold_used: dict[str, Any]
    config: ThresholdConfig
    provenance: dict[str, Any] = field(default_factory=dict)


class HistogramThresholdSegmenter:
    """Orchestrates histogram-driven threshold segmentation on a single 2-D frame."""

    def __init__(self, config: ThresholdConfig) -> None:
        self.config = config

    def run(
        self,
        image: np.ndarray,
        *,
        reference_mask: np.ndarray | None = None,
        voxel_size: tuple[float, ...] | None = None,
    ) -> SegmentationResult:
        cfg = self.config
        validate_bit_depth(image, cfg.bit_depth, strict=cfg.bit_depth_strict)

        hist = compute_histogram(image, bit_depth=cfg.bit_depth)

        mask = self._threshold_one(image, hist, reference_mask)

        if cfg.homogeneity_gate:
            gate = homogeneity_gate(
                image,
                window=cfg.homogeneity_window,
                std_max=cfg.homogeneity_std_max,
                is_3d=False,
            )
            mask = mask & gate

        mask = apply_spatial_constraints(
            mask,
            min_area=cfg.min_area,
            opening_radius=cfg.opening_radius,
            closing_radius=cfg.closing_radius,
            min_hole_size=cfg.min_hole_size,
            is_3d=False,
        )

        labels = label(mask, connectivity=2).astype(np.int32)
        regions = _measure_regions(labels, image, voxel_size)

        return SegmentationResult(
            mask=mask,
            labels=labels,
            regions=regions,
            histogram=hist,
            threshold_used=self._resolved_thresholds(image, hist, reference_mask),
            config=cfg,
            provenance={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_shape": image.shape,
                "input_dtype": str(image.dtype),
                "input_hash": hashlib.sha256(image.tobytes()).hexdigest()[:16],
            },
        )

    def _threshold_one(
        self,
        image: np.ndarray,
        hist: Histogram,
        reference_mask: np.ndarray | None,
    ) -> np.ndarray:
        cfg = self.config
        if cfg.method == "single":
            return threshold_single(image, cfg.direction, low=cfg.low, high=cfg.high)
        if cfg.method == "hysteresis":
            assert cfg.direction in ("below", "above")
            assert cfg.strict is not None and cfg.permissive is not None
            return threshold_hysteresis(
                image,
                cfg.direction,
                strict=cfg.strict,
                permissive=cfg.permissive,
            )
        if cfg.method == "percentile":
            return threshold_percentile(
                image,
                cfg.direction,
                bit_depth=cfg.bit_depth,
                percentile_low=cfg.percentile_low,
                percentile_high=cfg.percentile_high,
                sanity_floor=cfg.sanity_floor,
                sanity_ceiling=cfg.sanity_ceiling,
            )
        if cfg.method == "relative":
            return threshold_relative(
                image,
                cfg.direction,
                reference_mask=reference_mask,
                fraction_low=cfg.fraction_low,
                fraction_high=cfg.fraction_high,
            )
        raise ValueError(f"Unknown method {cfg.method}")

    def _resolved_thresholds(
        self,
        image: np.ndarray,
        hist: Histogram,
        reference_mask: np.ndarray | None,
    ) -> dict[str, Any]:
        cfg = self.config
        if cfg.method == "percentile":
            return {
                "method": "percentile",
                "low_resolved": hist.percentile(cfg.percentile_low) if cfg.percentile_low is not None else None,
                "high_resolved": hist.percentile(cfg.percentile_high) if cfg.percentile_high is not None else None,
                "percentile_low": cfg.percentile_low,
                "percentile_high": cfg.percentile_high,
            }
        if cfg.method == "relative":
            ref = image if reference_mask is None else image[reference_mask]
            median = float(np.median(ref))
            return {
                "method": "relative",
                "reference_median": median,
                "low_resolved": int(median * cfg.fraction_low) if cfg.fraction_low else None,
                "high_resolved": int(median * cfg.fraction_high) if cfg.fraction_high else None,
            }
        if cfg.method == "hysteresis":
            return {
                "method": "hysteresis",
                "strict": cfg.strict,
                "permissive": cfg.permissive,
            }
        return {"method": "single", "low": cfg.low, "high": cfg.high}


def _measure_regions(
    labels: np.ndarray,
    image: np.ndarray,
    voxel_size: tuple[float, ...] | None,
) -> list[dict[str, Any]]:
    if labels.max() == 0:
        return []

    properties = (
        "label",
        "area",
        "centroid",
        "mean_intensity",
        "min_intensity",
        "max_intensity",
    )
    table = regionprops_table(labels, intensity_image=image, properties=properties)

    n = len(table["label"])
    voxel_area = float(np.prod(voxel_size)) if voxel_size is not None else None

    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {
            "label_id": int(table["label"][i]),
            "area_px": int(table["area"][i]),
            "centroid_y": float(table["centroid-0"][i]),
            "centroid_x": float(table["centroid-1"][i]),
            "mean_intensity": float(table["mean_intensity"][i]),
            "min_intensity": float(table["min_intensity"][i]),
            "max_intensity": float(table["max_intensity"][i]),
        }
        if voxel_area is not None:
            row["area_um2"] = row["area_px"] * voxel_area
        rows.append(row)

    return rows
