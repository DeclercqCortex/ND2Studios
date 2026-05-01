"""
Stitch multipoint (M) tiles into a single time-lapse and write it as
a TIFF stack.

V1.1 design:

* Tile placement uses ND2 stage XY positions (in micrometres) divided
  by the file's pixel size to get pixel offsets. Tiles are placed
  on a canvas whose dimensions are chosen to fit every tile.
* No overlap blending — the second tile to be drawn at a given pixel
  simply overwrites the first. Most Nikon tile scans are designed to
  have small overlap; for the common case this looks fine.
* Falls back to a row-major grid layout if stage XY positions are
  empty / degenerate / all identical.

The pipeline:

    1. ``compute_tile_layout()`` → :class:`StitchLayout` (canvas size +
       per-tile (y, x) corner pixel offsets).
    2. ``stitch_timepoint()`` for one (t, channel) → 2D canvas array.
    3. ``export_stitched_tiff()`` walks (T, channels) and writes a
       multi-page TIFF — single channel as gray, multi as RGB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile

from nd2studios.backend.nd2_volume import LazyND2Volume
from nd2studios.backend.exporters.composite_exporter import (
    CHANNEL_COLORS, _percentile_uint8,
)


@dataclass
class StitchLayout:
    """Result of placing M tiles on a canvas."""
    canvas_h: int
    canvas_w: int
    # offsets[m] = (y, x) corner pixel for tile m on the canvas.
    offsets: List[Tuple[int, int]] = field(default_factory=list)
    # Tile size (uniform — ND2 tiles are always the same H/W).
    tile_h: int = 0
    tile_w: int = 0
    # Source flag, useful for diagnostics in the GUI.
    source: str = "stage_xy"   # 'stage_xy' or 'grid_fallback'


def compute_tile_layout(stage_xy_um: List[Tuple[float, float]],
                         pixel_size_um: float,
                         tile_h: int, tile_w: int,
                         m_indices: Optional[List[int]] = None) -> StitchLayout:
    """Place tiles on a canvas using physical stage XY positions.

    Parameters
    ----------
    stage_xy_um : per-multipoint (x, y) stage position in µm.
    pixel_size_um : ND2 pixel size; used to convert µm → px.
    tile_h, tile_w : single-tile pixel dimensions.
    m_indices : which M positions to include (defaults to all).

    Falls back to a row-major grid if stage_xy is empty or all positions
    are identical (within 0.1 µm).
    """
    if pixel_size_um <= 0:
        pixel_size_um = 1.0
    if m_indices is None:
        m_indices = list(range(len(stage_xy_um)))
    if not m_indices:
        return StitchLayout(canvas_h=tile_h, canvas_w=tile_w,
                            offsets=[(0, 0)], tile_h=tile_h, tile_w=tile_w,
                            source="empty")

    # Stage-XY path.
    if len(stage_xy_um) >= len(m_indices):
        try:
            xs = np.array([stage_xy_um[m][0] for m in m_indices], dtype=np.float64)
            ys = np.array([stage_xy_um[m][1] for m in m_indices], dtype=np.float64)
            spread_x = xs.max() - xs.min()
            spread_y = ys.max() - ys.min()
            if spread_x > 0.1 or spread_y > 0.1:
                # Convert to pixels relative to (min_x, min_y), flip Y so
                # acquisitions in physical "up" land at the top of the
                # canvas (microscope stages are usually right-handed
                # whereas image arrays grow downward).
                px = ((xs - xs.min()) / pixel_size_um).round().astype(int)
                py = ((ys.max() - ys) / pixel_size_um).round().astype(int)
                canvas_w = int(px.max() + tile_w)
                canvas_h = int(py.max() + tile_h)
                offsets = [(int(py[i]), int(px[i])) for i in range(len(m_indices))]
                return StitchLayout(canvas_h=canvas_h, canvas_w=canvas_w,
                                    offsets=offsets, tile_h=tile_h, tile_w=tile_w,
                                    source="stage_xy")
        except Exception:
            pass

    # Grid fallback.
    n = len(m_indices)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    offsets = [((m // cols) * tile_h, (m % cols) * tile_w) for m in range(n)]
    return StitchLayout(
        canvas_h=rows * tile_h, canvas_w=cols * tile_w,
        offsets=offsets, tile_h=tile_h, tile_w=tile_w,
        source="grid_fallback",
    )


def stitch_one_frame(tile_frames: List[np.ndarray],
                     layout: StitchLayout,
                     dtype: np.dtype) -> np.ndarray:
    """Place a list of tile frames onto a single canvas.

    Tiles whose offset would push them past the canvas are clipped.
    """
    canvas = np.zeros((layout.canvas_h, layout.canvas_w), dtype=dtype)
    for (y, x), frame in zip(layout.offsets, tile_frames):
        h, w = frame.shape
        y1 = min(y + h, layout.canvas_h)
        x1 = min(x + w, layout.canvas_w)
        canvas[y:y1, x:x1] = frame[:y1 - y, :x1 - x]
    return canvas


def export_stitched_tiff(
    volume: LazyND2Volume,
    layout: StitchLayout,
    m_indices: List[int],
    channel_indices: List[int],
    channel_colors: Dict[int, Tuple[int, int, int]],
    filepath: str,
    z_mode: str = "max",
    z_index: int = 0,
    rgb: bool = True,
    pixel_size_um: Optional[float] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    """Walk T, stitch tiles for each enabled channel, write multi-page TIFF.

    Parameters
    ----------
    rgb : if True and len(channel_indices) > 1, output is an RGB
        composite using ``channel_colors`` and percentile contrast. If
        False, only the first channel is written (single-channel TIFF).
    """
    if not filepath.lower().endswith((".tif", ".tiff")):
        filepath += ".tif"

    n_t = volume.n_timepoints
    nbytes_per_frame = layout.canvas_h * layout.canvas_w * (3 if rgb else 1) * 2
    bigtiff = nbytes_per_frame * n_t > 3_900_000_000

    resolution = None
    metadata = {}
    if pixel_size_um is not None and pixel_size_um > 0:
        resolution = (1.0 / pixel_size_um, 1.0 / pixel_size_um)
        metadata["unit"] = "um"

    with tifffile.TiffWriter(filepath, bigtiff=bigtiff) as writer:
        for t in range(n_t):
            # Stitch each enabled channel separately, then either output
            # the gray canvas (single-channel) or compose RGB.
            per_channel_canvas: Dict[int, np.ndarray] = {}
            for c in channel_indices:
                tiles = []
                for m in m_indices:
                    try:
                        f = volume.get_frame(c=c, m=m, t=t, z=z_index,
                                              z_mode=z_mode)
                    except Exception:
                        f = np.zeros((layout.tile_h, layout.tile_w),
                                     dtype=volume.dtype)
                    tiles.append(f)
                per_channel_canvas[c] = stitch_one_frame(tiles, layout, volume.dtype)

            if not rgb or len(channel_indices) == 1:
                page = per_channel_canvas[channel_indices[0]]
                writer.write(page, photometric="minisblack",
                              resolution=resolution,
                              resolutionunit="MICROMETER" if resolution else None,
                              metadata=metadata if t == 0 else None)
            else:
                # Additive RGB composite using percentile contrast per channel.
                h, w = layout.canvas_h, layout.canvas_w
                rgb_canvas = np.zeros((h, w, 3), dtype=np.float32)
                for c in channel_indices:
                    gray = _percentile_uint8(per_channel_canvas[c]).astype(np.float32)
                    color = channel_colors.get(c, (255, 255, 255))
                    rgb_canvas[..., 0] += gray * (color[0] / 255.0)
                    rgb_canvas[..., 1] += gray * (color[1] / 255.0)
                    rgb_canvas[..., 2] += gray * (color[2] / 255.0)
                writer.write(np.clip(rgb_canvas, 0, 255).astype(np.uint8),
                              photometric="rgb",
                              resolution=resolution,
                              resolutionunit="MICROMETER" if resolution else None,
                              metadata=metadata if t == 0 else None)

            if progress_cb is not None and (t % 4 == 0 or t == n_t - 1):
                progress_cb(int((t + 1) / max(1, n_t) * 100))
