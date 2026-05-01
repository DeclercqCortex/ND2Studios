"""
RGB composite TIFF exporter.

Takes a `Dict[channel_name, (T,H,W)]` plus per-channel colors/enable
flags, and writes a multi-page RGB TIFF (T, H, W, 3) uint8 with each
channel mapped to its assigned color and additively blended.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import tifffile


CHANNEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "gray": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (255, 0, 0),
    "blue": (0, 100, 255),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "white": (255, 255, 255),
}


def _percentile_uint8(frame: np.ndarray, p_low: float = 0.5,
                       p_high: float = 99.5) -> np.ndarray:
    f = frame.astype(np.float32)
    lo = np.percentile(f, p_low)
    hi = np.percentile(f, p_high)
    f = np.clip((f - lo) / (hi - lo + 1e-10), 0, 1)
    return (f * 255).astype(np.uint8)


def _composite_frame(
    frames: Dict[str, np.ndarray],
    colors: Dict[str, Tuple[int, int, int]],
    enabled: Dict[str, bool],
) -> np.ndarray:
    sample = next(iter(frames.values()))
    h, w = sample.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for name, frame in frames.items():
        if not enabled.get(name, True):
            continue
        gray = _percentile_uint8(frame).astype(np.float32)
        r, g, b = colors.get(name, (255, 255, 255))
        out[..., 0] += gray * (r / 255.0)
        out[..., 1] += gray * (g / 255.0)
        out[..., 2] += gray * (b / 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def export_rgb_composite_tiff(
    channels: Dict[str, np.ndarray],
    colors: Dict[str, Tuple[int, int, int]],
    enabled: Dict[str, bool],
    filepath: str,
    pixel_size_um: Optional[float] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    """Write an RGB composite TIFF stack.

    Parameters
    ----------
    channels : {channel_name: (T, H, W) array}
    colors : {channel_name: (R, G, B) uint8 tuple}
    enabled : {channel_name: bool}
    filepath : output path. `.tif` is appended if missing.
    pixel_size_um : optional scale tag.
    progress_cb : 0–100.
    """
    if not channels:
        raise ValueError("export_rgb_composite_tiff: no channels to export")

    if not filepath.lower().endswith((".tif", ".tiff")):
        filepath += ".tif"

    sample = next(iter(channels.values()))
    if sample.ndim != 3:
        raise ValueError(f"channels must be (T,H,W) — got {sample.shape}")
    n = sample.shape[0]

    resolution = None
    metadata = {}
    if pixel_size_um is not None and pixel_size_um > 0:
        resolution = (1.0 / pixel_size_um, 1.0 / pixel_size_um)
        metadata["unit"] = "um"

    bigtiff = n * sample.shape[1] * sample.shape[2] * 3 > 3_900_000_000

    with tifffile.TiffWriter(filepath, bigtiff=bigtiff) as writer:
        for t in range(n):
            frame_dict = {name: arr[t] for name, arr in channels.items()}
            rgb = _composite_frame(frame_dict, colors, enabled)
            writer.write(
                rgb,
                photometric="rgb",
                resolution=resolution,
                resolutionunit="MICROMETER" if resolution else None,
                metadata=metadata if t == 0 else None,
            )
            if progress_cb is not None and (t % 5 == 0 or t == n - 1):
                progress_cb(int((t + 1) / n * 100))
