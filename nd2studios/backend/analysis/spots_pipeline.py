"""Bright / Dark Spots — AnalysisPipeline adapter.

Wraps the spots backend into the ND2Studios analysis registry so it appears
in the General Analysis tab alongside Tear Detection, Nuclei Segmentation,
and Histogram Threshold Segmenter.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from nd2studios.core.analysis_registry import AnalysisPipeline, AnalysisResult
from nd2studios.core.plugin_registry import ParamSpec

from .spots.config import SpotsConfig
from .spots.identifier import BrightDarkSpotsSegmenter


@AnalysisPipeline.register
class BrightDarkSpotsPipeline(AnalysisPipeline):
    """GA3-style scale-space spot detector for bright and dark features.

    Combines a Laplacian-of-Gaussian / DoG band-pass response, a contrast gate,
    a circularity-based symmetry filter (all/more/medium/less objects), an
    LUT-histogram intensity gate, optional grow-objects step, and circular-disk
    or true-region output. No external dependencies beyond scipy and scikit-image.
    """

    name = "Bright / Dark Spots"
    description = (
        "GA3-style scale-space spot detector for bright (foci, vesicles, nuclei) "
        "or dark (granules, pits) features. Combines a LoG/DoG band-pass response, "
        "contrast gate, circularity symmetry filter, LUT-histogram intensity gate, "
        "optional grow-objects step, and circular-disk or true-region output."
    )

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                "channel_name", "Channel", "choice", "",
                choices=[],
                tooltip="Channel to analyse.",
            ),
            ParamSpec(
                "polarity", "Polarity", "choice", "bright",
                choices=["bright", "dark"],
                tooltip="bright: detect local maxima (foci, vesicles). dark: detect local minima (pits, granules).",
            ),
            ParamSpec(
                "typical_diameter_um", "Typical diameter (µm)", "float", 5.0,
                min_val=0.1, max_val=1000.0, step=0.5,
                tooltip="Expected spot diameter in micrometres. Sets the LoG/DoG scale.",
            ),
            ParamSpec(
                "contrast", "Contrast threshold", "float", 0.05,
                min_val=0.0, max_val=1.0, step=0.01,
                tooltip=(
                    "Minimum normalised DoG/LoG response (0–1) for a peak to be accepted. "
                    "Higher values reject low-contrast spots."
                ),
            ),
            ParamSpec(
                "symmetry", "Symmetry", "choice", "medium",
                choices=["all", "more", "medium", "less"],
                tooltip=(
                    "Circularity gate: all = no gate; more ≈ 0.45; medium ≈ 0.65; less ≈ 0.80. "
                    "Maps to the GA3 Bright/Dark Spots symmetry setting."
                ),
            ),
            ParamSpec(
                "intensity_percentile", "Intensity gate (percentile)", "float", 50.0,
                min_val=0.0, max_val=100.0, step=1.0,
                tooltip=(
                    "Global intensity gate from the LUT histogram. "
                    "Bright: centroid must exceed this percentile. "
                    "Dark: centroid must be below it. "
                    "Set to 0 to disable."
                ),
            ),
            ParamSpec(
                "bit_depth", "Bit depth", "int", 12,
                min_val=8, max_val=16, step=2,
                tooltip="Declared acquisition bit depth used for the LUT histogram.",
            ),
            ParamSpec(
                "grow_radius_um", "Grow radius (µm)", "float", 0.0,
                min_val=0.0, max_val=50.0, step=0.5,
                tooltip="Post-detection mask expansion radius in micrometres. 0 = disabled.",
            ),
            ParamSpec(
                "grow_method", "Grow method", "choice", "none",
                choices=["none", "dilation", "watershed"],
                tooltip="none: no grow. dilation: disk dilation. watershed: basin expansion.",
            ),
            ParamSpec(
                "output_mode", "Output mode", "choice", "region",
                choices=["region", "circular"],
                tooltip=(
                    "region: true connected-component mask via watershed. "
                    "circular: synthesised disk of radius = typical_diameter/2."
                ),
            ),
            ParamSpec(
                "kernel", "Kernel", "choice", "dog",
                choices=["dog", "log"],
                tooltip="dog: Difference of Gaussians (fast). log: Laplacian of Gaussian (exact).",
            ),
            ParamSpec(
                "background_mask", "Background mask", "bool", False,
                tooltip=(
                    "Show a secondary overlay on all pixels NOT covered by detected spots. "
                    "Useful for evaluating background signal in each frame."
                ),
            ),
        ]

    def run(
        self,
        channels: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[[int], None]] = None,
        cancelled_cb: Optional[Callable[[], bool]] = None,
    ) -> AnalysisResult:

        ch = params.get("channel_name", "")
        if not ch or ch not in channels:
            available = list(channels.keys())
            ch = available[0] if available else ""
        if not ch:
            raise ValueError("No channel available to analyse.")

        stack: np.ndarray = np.asarray(channels[ch])
        if stack.ndim == 2:
            stack = stack[np.newaxis]

        n_frames = stack.shape[0]
        pixel_size_um: float = float(metadata.get("pixel_size_um", 0.0))

        # intensity_percentile=0.0 means "gate disabled"
        raw_pct = float(params.get("intensity_percentile", 50.0))
        intensity_pct: float | None = raw_pct if raw_pct > 0.0 else None

        use_bg_mask: bool = bool(params.get("background_mask", False))

        cfg = SpotsConfig(
            polarity=str(params.get("polarity", "bright")),
            typical_diameter_um=float(params.get("typical_diameter_um", 5.0)),
            contrast=float(params.get("contrast", 0.05)),
            symmetry=str(params.get("symmetry", "medium")),
            intensity_percentile=intensity_pct,
            bit_depth=int(params.get("bit_depth", 12)),
            grow_radius_um=float(params.get("grow_radius_um", 0.0)),
            grow_method=str(params.get("grow_method", "none")),
            output_mode=str(params.get("output_mode", "region")),
            background_mask=use_bg_mask,
            kernel=str(params.get("kernel", "dog")),
            bit_depth_strict=False,
        )

        seg = BrightDarkSpotsSegmenter(cfg)
        label_stack = np.zeros(stack.shape, dtype=np.int32)
        bg_stack: np.ndarray | None = (
            np.zeros(stack.shape, dtype=np.int32) if use_bg_mask else None
        )
        measurements: list[dict[str, Any]] = []

        for t in range(n_frames):
            if cancelled_cb is not None and cancelled_cb():
                break

            frame = stack[t]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = seg.run(frame, pixel_size_um=pixel_size_um if pixel_size_um > 0 else None)

            label_stack[t] = result.labels
            if bg_stack is not None and result.background_labels is not None:
                bg_stack[t] = result.background_labels

            # Per-frame background intensity (mean ± std over non-spot pixels)
            bg_mean: float = 0.0
            bg_std: float = 0.0
            if use_bg_mask and result.background_labels is not None:
                bg_pixels = frame[result.background_labels > 0].astype(np.float64)
                if bg_pixels.size > 0:
                    bg_mean = float(np.mean(bg_pixels))
                    bg_std = float(np.std(bg_pixels))

            for row in result.regions:
                entry: dict[str, Any] = {
                    "frame": t,
                    "label_id": row["label_id"],
                    "area_px": row["area_px"],
                    "area_um2": row["area_um2"],
                    "centroid_y": row["centroid_y"],
                    "centroid_x": row["centroid_x"],
                    "mean_intensity": row["mean_intensity"],
                    "diameter_px": row["diameter_px"],
                    "contrast_score": row["contrast_score"],
                    "circularity": row["circularity"],
                    "polarity": cfg.polarity,
                    "background_mean_intensity": bg_mean,
                    "background_std_intensity": bg_std,
                }
                measurements.append(entry)

            if progress_cb is not None:
                progress_cb(int((t + 1) / n_frames * 100))

        secondary_masks: dict[str, np.ndarray] = {}
        if bg_stack is not None:
            secondary_masks[f"{ch}__background"] = bg_stack

        summary = _build_summary(measurements)
        return AnalysisResult(
            label_masks={ch: label_stack},
            measurements=measurements,
            summary=summary,
            overlay_color=(0, 100, 255),
            overlay_alpha=0.80,
            secondary_label_masks=secondary_masks,
            secondary_overlay_color=(160, 160, 160),
            secondary_overlay_alpha=0.25,
        )


def _build_summary(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    if not measurements:
        return {
            "total_objects": 0,
            "mean_area_px": 0.0,
            "std_area_px": 0.0,
            "mean_area_um2": 0.0,
            "std_area_um2": 0.0,
            "n_frames_with_objects": 0,
        }
    areas_px = np.array([m["area_px"] for m in measurements], dtype=float)
    areas_um2 = np.array([m["area_um2"] for m in measurements], dtype=float)
    frames_with = len({m["frame"] for m in measurements})
    return {
        "total_objects": len(measurements),
        "mean_area_px": float(np.mean(areas_px)),
        "std_area_px": float(np.std(areas_px)),
        "mean_area_um2": float(np.mean(areas_um2)),
        "std_area_um2": float(np.std(areas_um2)),
        "n_frames_with_objects": frames_with,
    }
