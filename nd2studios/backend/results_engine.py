"""
Pure-backend measurement engine for the Results tab (V1.22).

Computes extended per-object measurements from AnalysisResult label masks
plus the source channel arrays.  No Qt imports — callable from workers and
headless scripts.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.measure import regionprops


# ── Core measurement computation ─────────────────────────────────────────────

def compute_measurements(
    label_masks: Dict[str, np.ndarray],
    channels: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    m_index: int = 0,
    volumetric_voxel_counts: Optional[Dict[str, Dict[Tuple[int, int], int]]] = None,
) -> List[Dict[str, Any]]:
    """Compute extended per-object measurements from label masks.

    Args:
        label_masks: {seg_channel_name: (T, H, W) int32}  from AnalysisResult
        channels:    {channel_name: (T, H, W) array}       processed or raw
        metadata:    nd2_metadata dict  (pixel_size_um, z_step_um, n_zslices,
                     stage_xy_um, …)
        m_index:     multipoint index used for absolute stage coordinates
        volumetric_voxel_counts: optional per-channel per-(frame, label_id)
            voxel count from the pipeline. When supplied, ``volume_um3``
            comes from the true 3D voxel count; otherwise it falls back to
            the uniform-Z assumption ``area_um2 x n_zslices x z_step_um``.

    Returns:
        List of measurement dicts, one dict per detected object per frame.
        Columns: segmentation_channel, frame, label_id, area_px, area_um2,
        delta_area_px, delta_area_um2, volume_um3, delta_volume_um3,
        centroid_y/x_px, centroid_y/x_um, [centroid_y/x_stage_um],
        perimeter, eccentricity, solidity, bbox_*, mean/std_intensity_{ch}.
    """
    pixel_size: float = float(metadata.get("pixel_size_um") or 1.0)
    z_step: float = float(metadata.get("z_step_um") or 1.0)
    n_zslices: int = max(1, int(metadata.get("n_zslices") or 1))
    stage_xy = metadata.get("stage_xy_um") or []
    stage_pos: Optional[Tuple[float, float]] = (
        (float(stage_xy[m_index][0]), float(stage_xy[m_index][1]))
        if (stage_xy and m_index < len(stage_xy))
        else None
    )
    voxel_counts = volumetric_voxel_counts or {}

    # Materialise any lazy proxies so we can index freely.
    mat_channels: Dict[str, Optional[np.ndarray]] = {
        k: _materialise(v) for k, v in channels.items()
    }

    rows: List[Dict[str, Any]] = []

    for seg_channel, masks in label_masks.items():
        if masks.ndim != 3:
            continue
        T, H, W = masks.shape

        for t in range(T):
            mask_frame = np.asarray(masks[t], dtype=np.int32)
            if mask_frame.max() == 0:
                continue

            primary_img = _get_frame(mat_channels, seg_channel, t)
            props = regionprops(mask_frame, intensity_image=primary_img)

            for prop in props:
                cy_px, cx_px = prop.centroid

                # Volume from real voxel count if the pipeline supplied one
                # (e.g., Manual Mask with per-Z shapes); else apply the
                # uniform-Z assumption: area × n_z × z_step. Z step is in µm
                # so the result is in µm³.
                ch_voxels = voxel_counts.get(seg_channel) or {}
                voxel_count = ch_voxels.get((int(t), int(prop.label)))
                if voxel_count is not None:
                    volume_um3 = float(voxel_count) * (pixel_size ** 2) * z_step
                else:
                    volume_um3 = float(prop.area) * (pixel_size ** 2) * n_zslices * z_step

                row: Dict[str, Any] = {
                    "segmentation_channel": seg_channel,
                    "frame": int(t),
                    "label_id": int(prop.label),
                    "area_px": int(prop.area),
                    "area_um2": round(prop.area * pixel_size ** 2, 4),
                    "volume_um3": round(volume_um3, 4),
                    "centroid_y_px": round(cy_px, 3),
                    "centroid_x_px": round(cx_px, 3),
                    "centroid_y_um": round(cy_px * pixel_size, 4),
                    "centroid_x_um": round(cx_px * pixel_size, 4),
                    "perimeter": round(float(prop.perimeter), 3),
                    "eccentricity": round(float(prop.eccentricity), 4),
                    "solidity": (
                        round(float(prop.solidity), 4)
                        if prop.solidity is not None else None
                    ),
                    "bbox_min_row": int(prop.bbox[0]),
                    "bbox_min_col": int(prop.bbox[1]),
                    "bbox_max_row": int(prop.bbox[2]),
                    "bbox_max_col": int(prop.bbox[3]),
                }

                # Absolute stage coordinates (offset from frame centre)
                if stage_pos is not None:
                    sx, sy = stage_pos
                    row["centroid_x_stage_um"] = round(
                        sx + (cx_px - W / 2.0) * pixel_size, 4
                    )
                    row["centroid_y_stage_um"] = round(
                        sy + (cy_px - H / 2.0) * pixel_size, 4
                    )

                # Per-channel intensity stats
                for ch_name, ch_arr in mat_channels.items():
                    frame_data = _get_frame(mat_channels, ch_name, t)
                    if frame_data is None or frame_data.shape != mask_frame.shape:
                        continue
                    pixels = frame_data[mask_frame == prop.label]
                    if pixels.size:
                        safe = ch_name.replace(" ", "_")
                        row[f"mean_intensity_{safe}"] = round(float(pixels.mean()), 4)
                        row[f"std_intensity_{safe}"] = round(float(pixels.std()), 4)

                rows.append(row)

    # Per-track per-frame ΔArea — applied to every pipeline so any mask source
    # (manual, threshold, nuclei, spots, …) gets it for free. Sort by
    # (segmentation_channel, label_id, frame) so consecutive rows belong to
    # the same "track" and we can diff in one pass.
    #
    # Caveat: for instance-segmentation pipelines (nuclei, spots) label_id is
    # not stable across frames — regionprops assigns fresh ids per frame.
    # ΔArea for those reflects "label k between consecutive frames" which is
    # only meaningful after tracking. For single-object pipelines and the
    # manual mask (where label_id is user-controlled) the value is exact.
    # TODO V1.27+: add delta_volume_um3 when Z-stacks land.
    rows.sort(key=lambda r: (r["segmentation_channel"], int(r["label_id"]), int(r["frame"])))
    prev_key = None
    prev_area_px: Optional[int] = None
    prev_area_um2: Optional[float] = None
    prev_volume_um3: Optional[float] = None
    for r in rows:
        key = (r["segmentation_channel"], int(r["label_id"]))
        if key != prev_key or prev_area_px is None:
            r["delta_area_px"] = None
            r["delta_area_um2"] = None
            r["delta_volume_um3"] = None
        else:
            r["delta_area_px"] = int(r["area_px"]) - prev_area_px
            r["delta_area_um2"] = round(float(r["area_um2"]) - prev_area_um2, 4)
            r["delta_volume_um3"] = round(
                float(r["volume_um3"]) - (prev_volume_um3 or 0.0), 4
            )
        prev_key = key
        prev_area_px = int(r["area_px"])
        prev_area_um2 = float(r["area_um2"])
        prev_volume_um3 = float(r["volume_um3"])

    return rows


# ── Image export with label overlay ──────────────────────────────────────────

def export_overlay_frames(
    channels: Dict[str, np.ndarray],
    label_masks: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    output_dir: str,
    fmt: str = "tiff",
    channel_display: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Export per-frame composite images with label mask burned in.

    For each T frame, builds a uint8 RGB composite from *channels* using
    *channel_display* colours (or a default palette), then overlays the
    combined label mask in semi-transparent cyan.

    Args:
        channels:        {name: (T, H, W)} arrays
        label_masks:     {seg_channel: (T, H, W) int32}
        metadata:        nd2_metadata dict
        output_dir:      directory to write files into (must exist)
        fmt:             "tiff" or "jpg"
        channel_display: {name: {color, lut_lo, lut_hi}} — uses viewer state

    Returns:
        List of written file paths.
    """
    import imageio

    mat_channels = {k: _materialise(v) for k, v in channels.items()}

    # Determine T from the first channel
    T = 1
    for arr in mat_channels.values():
        if arr is not None and arr.ndim == 3:
            T = arr.shape[0]
            break

    # Build combined binary mask across all segmentation channels
    H = W = 0
    for arr in mat_channels.values():
        if arr is not None and arr.ndim == 3:
            _, H, W = arr.shape
            break

    ext = "tiff" if fmt.lower() in ("tif", "tiff") else "jpg"

    written: List[str] = []
    for t in range(T):
        # RGB composite
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        ch_list = list(mat_channels.keys())
        default_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
        ]
        for ci, ch_name in enumerate(ch_list):
            arr = _get_frame(mat_channels, ch_name, t)
            if arr is None:
                continue
            disp = (channel_display or {}).get(ch_name, {})
            if not disp.get("enabled", True):
                continue
            color_hex: str = disp.get("color") or ""
            color = _hex_to_rgb(color_hex) if color_hex else default_colors[ci % len(default_colors)]
            lut_lo = float(disp.get("lut_lo") or 0.0)
            lut_hi = float(disp.get("lut_hi") or 1.0)
            gray = _normalise_frame(arr, lut_lo, lut_hi)  # float32 0-1
            for c, weight in enumerate(color):
                rgb[..., c] = np.clip(
                    rgb[..., c].astype(np.float32) + gray * weight, 0, 255
                ).astype(np.uint8)

        # Overlay binary mask in semi-transparent cyan
        combined_mask = np.zeros((H, W), dtype=bool)
        for masks in label_masks.values():
            if masks.ndim == 3 and t < masks.shape[0]:
                combined_mask |= masks[t] > 0
        if combined_mask.any():
            overlay = rgb.astype(np.float32).copy()
            overlay[combined_mask, 0] = overlay[combined_mask, 0] * 0.5
            overlay[combined_mask, 1] = np.clip(
                overlay[combined_mask, 1] * 0.5 + 127, 0, 255
            )
            overlay[combined_mask, 2] = np.clip(
                overlay[combined_mask, 2] * 0.5 + 127, 0, 255
            )
            rgb = overlay.astype(np.uint8)

        fname = os.path.join(output_dir, f"frame_{t:04d}.{ext}")
        if ext == "tiff":
            imageio.imwrite(fname, rgb)
        else:
            imageio.imwrite(fname, rgb, quality=92)
        written.append(fname)

    return written


def export_label_masks_tiff(
    label_masks: Dict[str, np.ndarray],
    output_dir: str,
) -> List[str]:
    """Export each label mask stack as an int32 TIFF (one file per channel)."""
    import tifffile

    written: List[str] = []
    for ch_name, masks in label_masks.items():
        safe = ch_name.replace(" ", "_").replace("/", "_")
        fpath = os.path.join(output_dir, f"labels_{safe}.tif")
        arr = np.asarray(masks, dtype=np.int32)
        tifffile.imwrite(fpath, arr, imagej=True)
        written.append(fpath)
    return written


# ── Helpers ───────────────────────────────────────────────────────────────────

def _materialise(data: Any) -> Optional[np.ndarray]:
    if data is None:
        return None
    if hasattr(data, "materialize") and callable(data.materialize):
        return data.materialize()
    try:
        return np.asarray(data)
    except Exception:
        return None


def _get_frame(
    channels: Dict[str, Optional[np.ndarray]],
    name: str,
    t: int,
) -> Optional[np.ndarray]:
    arr = channels.get(name)
    if arr is None:
        return None
    try:
        if arr.ndim == 3:
            return arr[t].astype(np.float32, copy=False)
        if arr.ndim == 2:
            return arr.astype(np.float32, copy=False)
    except (IndexError, TypeError):
        pass
    return None


def _normalise_frame(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale frame to [0, 255] float32 using percentile LUT bounds."""
    f = frame.astype(np.float32)
    vmax = float(f.max())
    if vmax == 0:
        return np.zeros_like(f)
    lo_v = lo * vmax
    hi_v = hi * vmax if hi > 0 else vmax
    span = hi_v - lo_v
    if span <= 0:
        return np.zeros_like(f)
    return np.clip((f - lo_v) / span * 255.0, 0, 255)


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (200, 200, 200)
