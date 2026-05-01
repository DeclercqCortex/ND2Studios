"""
TIFF stack exporter.

Writes a (T, H, W) timeseries — already Z-projected — as a multi-page
TIFF. Optionally bigtiff if the file would exceed 4 GB.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import tifffile


def _normalize_to_uint8(stack: np.ndarray, p_low: float = 0.5,
                       p_high: float = 99.5) -> np.ndarray:
    f = stack.astype(np.float32)
    lo = np.percentile(f, p_low)
    hi = np.percentile(f, p_high)
    f = np.clip((f - lo) / (hi - lo + 1e-10), 0, 1)
    return (f * 255).astype(np.uint8)


def _normalize_to_uint16(stack: np.ndarray, p_low: float = 0.5,
                         p_high: float = 99.5) -> np.ndarray:
    f = stack.astype(np.float32)
    lo = np.percentile(f, p_low)
    hi = np.percentile(f, p_high)
    f = np.clip((f - lo) / (hi - lo + 1e-10), 0, 1)
    return (f * 65535).astype(np.uint16)


def export_tiff_stack(
    stack: np.ndarray,
    filepath: str,
    bit_depth: str = "passthrough",
    pixel_size_um: Optional[float] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    """Write `stack` (T, H, W) to a multi-page TIFF.

    Parameters
    ----------
    stack : (T, H, W) array of any numeric dtype.
    filepath : output path. `.tif` is appended if missing.
    bit_depth : one of {"passthrough", "uint8", "uint16"}.
        - "passthrough" keeps the source dtype.
        - "uint8" / "uint16" rescale to that range using the 0.5–99.5
          percentile bounds of the whole stack (consistent across frames
          so contrast doesn't drift between pages).
    pixel_size_um : if given, written into the TIFF resolution tags so
        downstream tools (Fiji, napari) get scale right.
    progress_cb : 0–100 progress.
    """
    if stack.ndim != 3:
        raise ValueError(f"export_tiff_stack expects (T,H,W) — got {stack.shape}")
    if not filepath.lower().endswith((".tif", ".tiff")):
        filepath += ".tif"

    if bit_depth == "uint8":
        stack = _normalize_to_uint8(stack)
    elif bit_depth == "uint16":
        stack = _normalize_to_uint16(stack)
    elif bit_depth != "passthrough":
        raise ValueError(f"Unknown bit_depth: {bit_depth}")

    n = stack.shape[0]
    nbytes = stack.nbytes
    bigtiff = nbytes > 3_900_000_000  # 3.9 GB threshold

    # tifffile resolution tag is (px/cm or px/inch, depending on unit). We
    # write px/μm-equivalent via the `resolutionunit=NONE` trick: store
    # 1/pixel_size_um as resolution. Most tools read it correctly.
    resolution: Optional[tuple] = None
    metadata = {}
    if pixel_size_um is not None and pixel_size_um > 0:
        # `tifffile` accepts (xres, yres) in samples-per-resolution-unit.
        # Use resolutionunit='MICROMETER' to make Fiji happy.
        resolution = (1.0 / pixel_size_um, 1.0 / pixel_size_um)
        metadata["unit"] = "um"
        metadata["spacing"] = pixel_size_um

    with tifffile.TiffWriter(filepath, bigtiff=bigtiff) as writer:
        for t in range(n):
            writer.write(
                stack[t],
                photometric="minisblack",
                resolution=resolution,
                resolutionunit="MICROMETER" if resolution else None,
                metadata=metadata if t == 0 else None,
            )
            if progress_cb is not None and (t % 10 == 0 or t == n - 1):
                progress_cb(int((t + 1) / n * 100))
