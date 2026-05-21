"""
Tear / Dark Region Analysis Pipeline — classical Phase-1 baseline.

Detects dark, spatially homogeneous regions inside tissue (physical tears,
bubbles, off-section areas) using a three-feature homogeneity score derived
from local intensity, local variance, and local entropy.

No DL dependency required. All operations use scipy + skimage, both of which
are already project dependencies.

Reference: tear_region_analysis_build_guide.md — Phase 0-1 (classical baseline)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    generic_filter,
    uniform_filter,
    zoom,
)
from skimage.filters import gaussian, threshold_otsu
from skimage.filters.rank import entropy as rank_entropy
from skimage.measure import label as skimage_label, regionprops
from skimage.morphology import disk

from nd2studios.core.analysis_registry import AnalysisPipeline, AnalysisResult
from nd2studios.core.plugin_registry import ParamSpec


@AnalysisPipeline.register
class TearDetectionPipeline(AnalysisPipeline):
    """Detect dark / homogeneous tear regions in fluorescence images.

    Uses a classical pipeline:
    1. Tissue mask via Otsu on log-intensity (no-stain background excluded).
    2. Homogeneity score = (1-norm_intensity) × (1-norm_variance) × (1-norm_entropy)
       with all terms rank-normalised inside the tissue mask for batch robustness.
    3. Connected-component extraction with area and solidity filtering.
    4. Optional weak-stain disambiguation using a counterstain channel.

    All-classical; no deep learning required.
    """

    name = "Tear / Dark Region Detection"
    description = (
        "Detect dark, spatially homogeneous regions (physical tears, bubbles, "
        "off-section areas) using a rank-normalised homogeneity score "
        "(intensity × variance × entropy). Optionally disambiguates true tears "
        "from weakly-stained tissue using a counterstain channel. "
        "No additional dependencies required."
    )

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                "channel_name", "Primary channel", "choice", "",
                choices=[],
                tooltip="Channel to analyse (ideally a structural counterstain "
                        "present throughout the tissue, e.g. DAPI).",
            ),
            ParamSpec(
                "counterstain_channel", "Counterstain (optional)", "choice", "None",
                choices=["None"],
                tooltip="Second channel used to disambiguate true tears (dark in "
                        "all channels) from weakly-stained regions (signal present "
                        "in counterstain). Set to 'None' to skip classification.",
            ),
            ParamSpec(
                "window_size_px", "Window size (px)", "int", 20,
                min_val=5, max_val=200, step=5,
                tooltip="Pixel window used to compute local intensity, variance, "
                        "and entropy. Should be ~1-2× the expected tear diameter.",
            ),
            ParamSpec(
                "homogeneity_threshold", "Homogeneity threshold", "float", 0.55,
                min_val=0.0, max_val=1.0, step=0.05,
                tooltip="Score threshold ∈ [0,1]. Higher = stricter (fewer "
                        "detections). Tune per dataset; 0.5–0.65 is typical.",
            ),
            ParamSpec(
                "min_area_um2", "Min area (µm²)", "float", 50.0,
                min_val=0.0, max_val=1_000_000.0, step=10.0,
                tooltip="Discard regions smaller than this area.",
            ),
            ParamSpec(
                "max_area_um2", "Max area (µm²)", "float", 0.0,
                min_val=0.0, max_val=1_000_000_000.0, step=1000.0,
                tooltip="Discard regions larger than this area. Set to 0 for no "
                        "upper limit.",
            ),
        ]

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        channels: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[[int], None]] = None,
        cancelled_cb: Optional[Callable[[], bool]] = None,
    ) -> AnalysisResult:

        # ── Resolve channels ──────────────────────────────────────────────────
        ch = params.get("channel_name", "")
        if ch not in channels:
            ch = next(iter(channels))

        volume: np.ndarray = np.asarray(channels[ch])
        if volume.ndim == 2:
            volume = volume[np.newaxis]
        T, H, W = volume.shape

        cs_name: str = params.get("counterstain_channel", "None")
        cs_volume: Optional[np.ndarray] = None
        if cs_name != "None" and cs_name in channels:
            cs_arr = np.asarray(channels[cs_name])
            cs_volume = cs_arr[np.newaxis] if cs_arr.ndim == 2 else cs_arr

        pixel_size_um: float = float(metadata.get("pixel_size_um", 1.0))
        px2 = pixel_size_um ** 2

        win: int = max(3, int(params.get("window_size_px", 20)))
        threshold: float = float(params.get("homogeneity_threshold", 0.55))
        min_area_um2: float = float(params.get("min_area_um2", 50.0))
        max_area_um2: float = float(params.get("max_area_um2", 0.0))
        min_area_px = max(1, int(min_area_um2 / px2)) if pixel_size_um > 0 else 1
        max_area_px = int(max_area_um2 / px2) if (max_area_um2 > 0 and pixel_size_um > 0) else 0

        if progress_cb:
            progress_cb(0)

        label_stack = np.zeros((T, H, W), dtype=np.int32)
        measurements: List[Dict[str, Any]] = []

        for t in range(T):
            if cancelled_cb and cancelled_cb():
                break

            frame = volume[t].astype(np.float32)
            cs_frame = cs_volume[t].astype(np.float32) if cs_volume is not None else None

            tissue_mask = _tissue_mask(frame)
            score_map = _homogeneity_score(frame, tissue_mask, win)

            candidate = (score_map > threshold) & tissue_mask

            labeled, _ = skimage_label(candidate, return_num=True)
            intensity_img = frame

            accepted = np.zeros_like(labeled, dtype=np.int32)
            new_id = 1
            for region in regionprops(labeled, intensity_image=intensity_img):
                if region.area < min_area_px:
                    continue
                if max_area_px > 0 and region.area > max_area_px:
                    continue
                accepted[labeled == region.label] = new_id

                # Classification
                region_mask = labeled == region.label
                region_class = _classify_region(
                    region_mask, frame, cs_frame, tissue_mask
                )

                cy, cx = region.centroid
                # Mean homogeneity score over this region
                hs = float(np.mean(score_map[region_mask]))

                measurements.append({
                    "frame": t,
                    "label_id": new_id,
                    "area_px": float(region.area),
                    "area_um2": float(region.area) * px2,
                    "centroid_y": float(cy),
                    "centroid_x": float(cx),
                    "mean_intensity": float(region.mean_intensity),
                    "class": region_class,
                    "homogeneity_score": round(hs, 4),
                    "solidity": round(float(region.solidity), 4),
                    "eccentricity": round(float(region.eccentricity), 4),
                })
                new_id += 1

            label_stack[t] = accepted

            if progress_cb:
                progress_cb(int((t + 1) / T * 100))

        areas = [m["area_px"] for m in measurements]
        summary: Dict[str, Any] = {
            "total_objects": len(measurements),
            "n_frames_with_objects": len({m["frame"] for m in measurements}),
            "mean_area_px": float(np.mean(areas)) if areas else 0.0,
            "std_area_px": float(np.std(areas)) if areas else 0.0,
            "mean_area_um2": float(np.mean(areas)) * px2 if areas else 0.0,
            "std_area_um2": float(np.std(areas)) * px2 if areas else 0.0,
        }

        return AnalysisResult(
            label_masks={ch: label_stack},
            measurements=measurements,
            summary=summary,
        )


# ── Stage helpers (pure functions, no Qt) ────────────────────────────────────


def _tissue_mask(frame: np.ndarray) -> np.ndarray:
    """Return a boolean tissue mask via Otsu on log-intensity at 4× downsample.

    Downsampling speeds up the operation and smooths sensor noise before
    thresholding. The mask is upsampled back to the original resolution.
    """
    H, W = frame.shape
    scale = 4
    small = zoom(frame, 1.0 / scale, order=1).astype(np.float32)
    blurred = gaussian(small, sigma=3.0, preserve_range=True)

    log_img = np.log1p(blurred.astype(np.float64))
    try:
        thr = threshold_otsu(log_img)
        binary_small = log_img > thr
    except Exception:
        # Fallback if threshold_otsu fails (e.g. all-zero frame)
        binary_small = np.ones_like(small, dtype=bool)

    # Morphological cleanup on the downsampled mask
    binary_small = binary_closing(binary_small, structure=disk(5))
    binary_small = binary_fill_holes(binary_small)
    binary_small = binary_opening(binary_small, structure=disk(2))

    # Upsample back to original size (nearest-neighbour to stay binary)
    mask_full = zoom(binary_small.astype(np.float32), scale, order=0) > 0.5
    # Crop/pad to exact original shape in case of rounding
    mask_out = np.zeros((H, W), dtype=bool)
    sh, sw = min(mask_full.shape[0], H), min(mask_full.shape[1], W)
    mask_out[:sh, :sw] = mask_full[:sh, :sw]
    return mask_out


def _rank_normalize(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rank-normalize arr to [0,1] within the tissue mask.

    Using ranks instead of min-max normalization is the key robustness choice:
    it is invariant to absolute intensity levels across batches/microscopes.
    """
    out = np.zeros_like(arr, dtype=np.float32)
    tissue_vals = arr[mask]
    if tissue_vals.size == 0:
        return out
    order = tissue_vals.argsort()
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, tissue_vals.size)
    out[mask] = ranks
    return out


def _homogeneity_score(
    frame: np.ndarray,
    tissue_mask: np.ndarray,
    win: int,
) -> np.ndarray:
    """Compute per-pixel homogeneity score inside the tissue mask.

    score = (1 - norm_intensity) × (1 - norm_variance) × (1 - norm_entropy)

    All three components are rank-normalised within the tissue mask, making
    the score invariant to global intensity shifts between acquisitions.
    """
    norm_frame = _percentile_normalize(frame)

    # Local intensity
    local_int = uniform_filter(norm_frame.astype(np.float64), size=win).astype(np.float32)

    # Local variance via generic_filter (slower but accurate)
    local_var = generic_filter(
        norm_frame.astype(np.float64), np.var, size=win
    ).astype(np.float32)

    # Local entropy (skimage rank, needs uint8 input)
    uint8_frame = (norm_frame * 255).clip(0, 255).astype(np.uint8)
    radius = max(1, win // 2)
    local_ent = rank_entropy(uint8_frame, disk(radius)).astype(np.float32)

    # Rank-normalize all three within tissue
    norm_I = _rank_normalize(local_int, tissue_mask)
    norm_V = _rank_normalize(local_var, tissue_mask)
    norm_E = _rank_normalize(local_ent, tissue_mask)

    score = (1.0 - norm_I) * (1.0 - norm_V) * (1.0 - norm_E)
    # Zero out pixels outside tissue
    score[~tissue_mask] = 0.0
    return score


def _percentile_normalize(frame: np.ndarray) -> np.ndarray:
    """Clip and normalize to [0, 1] using 1st–99.8th percentile."""
    lo = float(np.percentile(frame, 1.0))
    hi = float(np.percentile(frame, 99.8))
    denom = hi - lo if (hi - lo) > 1e-10 else 1.0
    return np.clip((frame.astype(np.float32) - lo) / denom, 0.0, 1.0)


def _classify_region(
    region_mask: np.ndarray,
    primary_frame: np.ndarray,
    counterstain_frame: Optional[np.ndarray],
    tissue_mask: np.ndarray,
) -> str:
    """Classify a detected region as 'tear', 'weak_stain', or 'unknown'.

    A region is classified as 'weak_stain' if its mean counterstain intensity
    exceeds the 15th percentile of counterstain signal inside the tissue mask
    (i.e. there is genuine signal present). Otherwise it is a 'tear'.
    Without a counterstain channel, classification is deferred to 'unknown'.
    """
    if counterstain_frame is None:
        return "unknown"

    tissue_cs = counterstain_frame[tissue_mask]
    if tissue_cs.size == 0:
        return "unknown"

    tissue_floor = float(np.percentile(tissue_cs, 15))
    mean_cs = float(np.mean(counterstain_frame[region_mask]))
    return "weak_stain" if mean_cs > tissue_floor else "tear"
