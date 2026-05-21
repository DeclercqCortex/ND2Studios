"""
Nuclei segmentation pipeline using Cellpose 3.

Cellpose is an optional dependency. The pipeline class is always importable;
calling run() without cellpose installed raises ImportError with install
instructions. The page catches this and shows a friendly dialog.

Requires (if running):
    pip install cellpose
"""
from __future__ import annotations

import importlib.util
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from nd2studios.core.analysis_registry import AnalysisPipeline, AnalysisResult
from nd2studios.core.plugin_registry import ParamSpec

CELLPOSE_AVAILABLE: bool = importlib.util.find_spec("cellpose") is not None


@AnalysisPipeline.register
class NucleiSegmentationPipeline(AnalysisPipeline):
    """Per-frame 2D nuclei instance segmentation via Cellpose 3.

    Segments each time-point independently using the ``nuclei`` or ``cyto3``
    Cellpose model. Produces integer label masks and per-object measurements
    (area, centroid, mean channel intensity).

    Optional dependency: ``pip install cellpose``
    """

    name = "Nuclei Segmentation"
    description = (
        "Per-frame 2D instance segmentation of nuclei using Cellpose 3. "
        "Outputs integer label masks and per-object measurements. "
        "Requires: pip install cellpose"
    )

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                "channel_name", "Channel", "choice", "",
                choices=[],
                tooltip="Channel to segment (e.g. DAPI). Populated from loaded data.",
            ),
            ParamSpec(
                "model_type", "Model", "choice", "nuclei",
                choices=["nuclei", "cyto3"],
                tooltip="Cellpose model: 'nuclei' for DAPI/Hoechst, 'cyto3' for whole-cell.",
            ),
            ParamSpec(
                "diameter", "Diameter (px)", "float", 30.0,
                min_val=0.0, max_val=500.0, step=1.0,
                tooltip="Expected nucleus diameter in pixels. Set to 0 for auto-detection.",
            ),
            ParamSpec(
                "min_area", "Min area (px²)", "int", 50,
                min_val=1, max_val=100_000, step=10,
                tooltip="Discard objects smaller than this area in pixels².",
            ),
            ParamSpec(
                "max_area", "Max area (px²)", "int", 50_000,
                min_val=100, max_val=10_000_000, step=100,
                tooltip="Discard objects larger than this area in pixels².",
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
        if not CELLPOSE_AVAILABLE:
            raise ImportError(
                "Cellpose is not installed.\n\n"
                "Install with:\n    pip install cellpose\n\n"
                "Then restart ND2Studios."
            )

        from cellpose import models  # type: ignore[import]
        from skimage.measure import regionprops  # already a project dependency

        # ── Resolve channel ──────────────────────────────────────────────────
        channel_name = params.get("channel_name", "")
        if channel_name not in channels:
            channel_name = next(iter(channels))

        volume: np.ndarray = channels[channel_name]
        if volume.ndim == 2:
            volume = volume[np.newaxis]          # treat single frame as T=1
        T, H, W = volume.shape

        pixel_size_um: float = float(metadata.get("pixel_size_um", 1.0))

        # ── Cellpose parameters ───────────────────────────────────────────────
        model_type: str = params.get("model_type", "nuclei")
        raw_diam = float(params.get("diameter", 30.0))
        diameter: Optional[float] = raw_diam if raw_diam > 0 else None
        min_area: int = int(params.get("min_area", 50))
        max_area: int = int(params.get("max_area", 50_000))

        if progress_cb:
            progress_cb(0)

        model = models.Cellpose(model_type=model_type, gpu=False)

        label_stack = np.zeros((T, H, W), dtype=np.int32)
        measurements: List[Dict[str, Any]] = []

        for t in range(T):
            if cancelled_cb and cancelled_cb():
                break

            frame = volume[t].astype(np.float32)

            # Percentile normalization to [0, 1] — standard for fluorescence
            lo = float(np.percentile(frame, 1.0))
            hi = float(np.percentile(frame, 99.9))
            denom = hi - lo
            if denom < 1e-10:
                denom = 1.0
            frame_norm = np.clip((frame - lo) / denom, 0.0, 1.0)

            masks_list, _, _, _ = model.eval(
                [frame_norm],
                diameter=diameter,
                channels=[0, 0],  # grayscale
                progress=False,
            )
            mask: np.ndarray = masks_list[0].astype(np.int32)

            # Area filtering
            intensity_img = volume[t].astype(np.float32)
            for region in regionprops(mask, intensity_image=intensity_img):
                if region.area < min_area or region.area > max_area:
                    mask[mask == region.label] = 0
            mask = _relabel_contiguous(mask)
            label_stack[t] = mask

            # Per-object measurements
            for region in regionprops(mask, intensity_image=intensity_img):
                cy, cx = region.centroid
                measurements.append({
                    "frame": t,
                    "label_id": int(region.label),
                    "area_px": float(region.area),
                    "area_um2": float(region.area) * pixel_size_um ** 2,
                    "centroid_y": float(cy),
                    "centroid_x": float(cx),
                    "mean_intensity": float(region.mean_intensity),
                })

            if progress_cb:
                progress_cb(int((t + 1) / T * 100))

        areas = [m["area_px"] for m in measurements]
        summary: Dict[str, Any] = {
            "total_objects": len(measurements),
            "n_frames_with_objects": len({m["frame"] for m in measurements}),
            "mean_area_px": float(np.mean(areas)) if areas else 0.0,
            "std_area_px": float(np.std(areas)) if areas else 0.0,
            "mean_area_um2": float(np.mean(areas)) * pixel_size_um ** 2 if areas else 0.0,
            "std_area_um2": float(np.std(areas)) * pixel_size_um ** 2 if areas else 0.0,
        }

        return AnalysisResult(
            label_masks={channel_name: label_stack},
            measurements=measurements,
            summary=summary,
        )


def _relabel_contiguous(mask: np.ndarray) -> np.ndarray:
    """Re-index integer mask so label IDs are contiguous from 1."""
    out = np.zeros_like(mask)
    new_id = 1
    for old_id in np.unique(mask):
        if old_id == 0:
            continue
        out[mask == old_id] = new_id
        new_id += 1
    return out
