"""
Stitch multipoint (M) tiles into a single time-lapse and write it as
a TIFF stack.

V1.14 design (pure physical layout):

* Tile placement converts stage XY metadata directly to pixel offsets:
  ``offset_x = round((stage_x - min_stage_x) / pixel_size_um)``
  ``offset_y = round((max_stage_y - stage_y) / pixel_size_um)``
  Y is flipped so the highest stage Y maps to row 0 (top of image).
* Physical gaps between non-adjacent tiles appear as empty (black)
  regions on the canvas, faithfully representing the acquisition
  footprint as if the file were one reconstituted image.
* No overlap blending — the second tile drawn at a given pixel simply
  overwrites the first.
* Falls back to a row-major sqrt-grid layout when no stage XY data is
  available or all positions are identical.

The pipeline:

    1. ``compute_tile_layout()`` → :class:`StitchLayout` (canvas size +
       per-tile (y, x) corner pixel offsets).
    2. ``stitch_one_frame()`` for one (t, channel) → 2D canvas array.
    3. ``export_stitched_tiff()`` walks (T, channels) and writes a
       multi-page TIFF — single channel as gray, multi as RGB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile

from nd2studios.backend.nd2_volume import LazyND2Volume
from nd2studios.backend.exporters.composite_exporter import CHANNEL_COLORS
from nd2studios.backend.exporters.tiff_exporter import (
    _silence_bigtiff_imagej_warning,
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
    source: str = "stage_xy"   # 'stage_xy', 'serpentine', 'grid_fallback', 'empty'
    # V1.7: number of unique physical (col,row) cells before expansion
    # and the maximum number of multipoints any one cell received.
    n_phys_cells: int = 0
    n_visits_per_cell: int = 1
    n_phys_cols: int = 0
    n_phys_rows: int = 0
    # V1.10: when source == "serpentine", explain why we got here so
    # the GUI can show a meaningful diagnostic. Values:
    #   "revisits" — V1.8 trigger (max_dup > 1)
    #   "sparse"   — V1.10 trigger (fill_ratio < SPARSE_THRESHOLD)
    serpentine_reason: str = ""
    # Fill ratio of the physical grid (populated cells / phys_cols * phys_rows).
    # 1.0 = every cell visited; 0.4 = a 40 %-filled ROI scan.
    fill_ratio: float = 1.0


def _cluster_axis(values: np.ndarray,
                   max_tolerance: Optional[float] = None) -> List[float]:
    """Cluster nearby 1D values into ascending centroids.

    The tolerance is **derived from the data**, not passed in:

    - Compute all positive-pair gaps between sorted unique values.
    - If they form a bimodal distribution (small "jitter" gaps and
      larger "step" gaps with a ≥ 3× ratio between them), tolerance =
      midpoint between the largest jitter gap and the smallest step gap.
    - Otherwise (uniform gaps → regular grid, no jitter) tolerance =
      half the smallest gap.

    ``max_tolerance`` is an optional sanity cap (typically half the tile
    size in µm) — protects degenerate scans where a tiny rounding
    difference shouldn't produce a cluster boundary.

    Empty input → empty list. Single value → one cluster.
    """
    if len(values) == 0:
        return []
    sorted_vals = np.sort(np.asarray(values, dtype=np.float64))
    if len(sorted_vals) == 1:
        return [float(sorted_vals[0])]

    diffs = np.diff(sorted_vals)
    nonzero = diffs[diffs > 1e-6]
    if len(nonzero) == 0:
        # All values are effectively identical.
        return [float(np.mean(sorted_vals))]

    sorted_gaps = np.sort(nonzero)
    if len(sorted_gaps) == 1:
        tol = float(sorted_gaps[0]) * 0.5
    else:
        # Detect bimodal "jitter vs step" jump.
        ratios = sorted_gaps[1:] / np.maximum(sorted_gaps[:-1], 1e-9)
        i_max = int(np.argmax(ratios))
        if ratios[i_max] > 3.0:
            # Bimodal: split at the jump.
            tol = (float(sorted_gaps[i_max]) +
                    float(sorted_gaps[i_max + 1])) / 2.0
        else:
            # Unimodal: regular grid, no jitter — half the smallest gap.
            tol = float(sorted_gaps[0]) * 0.5

    if max_tolerance is not None and max_tolerance > 0:
        tol = min(tol, float(max_tolerance))
    tol = max(tol, 1e-6)

    centroids: List[float] = []
    current = [float(sorted_vals[0])]
    for v in sorted_vals[1:]:
        if float(v) - current[-1] <= tol:
            current.append(float(v))
        else:
            centroids.append(float(np.mean(current)))
            current = [float(v)]
    centroids.append(float(np.mean(current)))
    return centroids


def _assign_index(value: float, centroids: List[float]) -> int:
    """Return the index of the centroid closest to ``value``."""
    best_idx = 0
    best_diff = abs(value - centroids[0])
    for i in range(1, len(centroids)):
        d = abs(value - centroids[i])
        if d < best_diff:
            best_diff = d
            best_idx = i
    return best_idx


def _serpentine_layout(n_tiles: int, tile_h: int, tile_w: int,
                       n_rows_hint: int) -> "StitchLayout":
    """Place ``n_tiles`` cells in serpentine (boustrophedon) order.

    M=0 → row 0, col 0 (top-left). Even rows go left-to-right; odd
    rows go right-to-left. Used when an ND2 file's M axis encodes
    "visits" of the same physical positions (V1.7 detection); in that
    case the actual scan order is what the user wants to see, not the
    degenerate spatial X axis.

    ``n_rows_hint`` is typically the number of unique stage Y values
    from clustering. ``n_cols`` is derived as ``ceil(n_tiles / rows)``.
    """
    n_rows = max(1, int(n_rows_hint))
    n_cols = (n_tiles + n_rows - 1) // n_rows  # ceil
    offsets: List[Tuple[int, int]] = []
    for m in range(n_tiles):
        row = m // n_cols
        col_in_row = m % n_cols
        if row % 2 == 0:
            col = col_in_row
        else:
            col = n_cols - 1 - col_in_row
        offsets.append((row * tile_h, col * tile_w))
    return StitchLayout(
        canvas_h=n_rows * tile_h,
        canvas_w=n_cols * tile_w,
        offsets=offsets,
        tile_h=tile_h, tile_w=tile_w,
        source="serpentine",
        n_phys_cells=n_tiles,
        n_visits_per_cell=1,
        n_phys_cols=n_cols,
        n_phys_rows=n_rows,
    )


def compute_tile_layout(stage_xy_um: List[Tuple[float, float]],
                         pixel_size_um: float,
                         tile_h: int, tile_w: int,
                         m_indices: Optional[List[int]] = None,
                         cluster_tolerance_frac: float = 0.5) -> StitchLayout:
    """Place tiles on a canvas using pure physical coordinates.

    V1.14 algorithm:

    1. Convert each tile's stage XY (µm) to pixel offsets:
       ``offset_x = round((stage_x - min_stage_x) / pixel_size_um)``
       ``offset_y = round((max_stage_y - stage_y) / pixel_size_um)``
    2. Y is flipped: highest stage Y → row 0 (top of image).
    3. Canvas = bounding box of all tile corners.
    4. Physical gaps between non-adjacent tiles appear as empty (black)
       pixels, faithfully representing the scan footprint.

    Falls back to a row-major sqrt-grid only when there is no usable
    stage XY data (empty list, or all positions identical).
    """
    if pixel_size_um <= 0:
        pixel_size_um = 1.0
    if m_indices is None:
        m_indices = list(range(len(stage_xy_um)))
    if not m_indices:
        return StitchLayout(canvas_h=tile_h, canvas_w=tile_w,
                            offsets=[(0, 0)], tile_h=tile_h, tile_w=tile_w,
                            source="empty")

    # When acquisition stopped early, stage_xy_um may be shorter than
    # m_indices — truncate to the M's that have data.
    if len(stage_xy_um) < len(m_indices):
        m_indices = list(m_indices[:len(stage_xy_um)])

    # Physical layout (V1.14): stage µm → pixel offsets directly.
    if len(stage_xy_um) >= len(m_indices) and len(m_indices) > 0:
        try:
            xs = np.array([stage_xy_um[m][0] for m in m_indices], dtype=np.float64)
            ys = np.array([stage_xy_um[m][1] for m in m_indices], dtype=np.float64)
            spread_x = float(xs.max() - xs.min())
            spread_y = float(ys.max() - ys.min())
            if spread_x > 0.1 or spread_y > 0.1:
                min_x = float(xs.min())
                max_y = float(ys.max())

                offsets: List[Tuple[int, int]] = []
                for i in range(len(m_indices)):
                    m = m_indices[i]
                    sx, sy = stage_xy_um[m]
                    px_x = int(round((sx - min_x) / pixel_size_um))
                    px_y = int(round((max_y - sy) / pixel_size_um))
                    offsets.append((px_y, px_x))
                # Reverse so M=0 occupies the canvas slot that M=last had
                # and M=last occupies M=0's slot. Tile image content is
                # unchanged; only canvas positions are swapped.
                offsets = list(reversed(offsets))

                canvas_w = max(off[1] for off in offsets) + tile_w
                canvas_h = max(off[0] for off in offsets) + tile_h

                n_unique = len(set(offsets))
                n_phys_cols = len(set(off[1] // max(tile_w, 1) for off in offsets))
                n_phys_rows = len(set(off[0] // max(tile_h, 1) for off in offsets))

                return StitchLayout(
                    canvas_h=canvas_h,
                    canvas_w=canvas_w,
                    offsets=offsets,
                    tile_h=tile_h, tile_w=tile_w,
                    source="physical",
                    n_phys_cells=n_unique,
                    n_visits_per_cell=max(1, len(m_indices) // max(n_unique, 1)),
                    n_phys_cols=n_phys_cols,
                    n_phys_rows=n_phys_rows,
                    fill_ratio=1.0,
                )
        except Exception:
            pass

    # Grid fallback (sqrt row-major). Handles n == 0 gracefully.
    n = len(m_indices)
    if n <= 0:
        return StitchLayout(canvas_h=tile_h, canvas_w=tile_w,
                            offsets=[], tile_h=tile_h, tile_w=tile_w,
                            source="grid_fallback")
    cols = max(1, int(np.ceil(np.sqrt(n))))
    rows = max(1, int(np.ceil(n / cols)))
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
    n_c = len(channel_indices)

    # z_mode='none' on a multi-Z volume keeps every Z plane so the stitched
    # output is a true TZCYX hyperstack; projection modes (and single-Z
    # volumes) collapse to Z=1 as before.
    preserve_z = (z_mode == "none" and volume.n_zslices > 1)
    n_z_out = volume.n_zslices if preserve_z else 1

    nbytes_total = (layout.canvas_h * layout.canvas_w
                    * n_c * n_t * n_z_out * volume.dtype.itemsize)
    bigtiff = nbytes_total > 3_900_000_000

    # Disk-backed output (V1.32). ``tifffile.memmap`` allocates an
    # ImageJ-formatted TIFF on disk and returns an ndarray view into it.
    # Writing through that view never holds more than one canvas frame
    # in RAM, while still producing a valid Fiji-readable hyperstack —
    # a 90 GiB (T, Z, C, H, W) output that previously OOM-killed on
    # `np.zeros(...)` now goes straight to disk.
    ch_names = [volume.channel_names[c] for c in channel_indices]
    ij_meta: Dict[str, object] = {"Labels": ch_names}
    resolution = None
    if pixel_size_um is not None and pixel_size_um > 0:
        resolution = (1.0 / pixel_size_um, 1.0 / pixel_size_um)
        ij_meta["unit"] = "um"
        ij_meta["spacing"] = float(pixel_size_um)

    n_pages_total = max(1, n_t * n_z_out * n_c)
    pages_written = 0
    with _silence_bigtiff_imagej_warning():
        mm = tifffile.memmap(
            filepath,
            shape=(n_t, n_z_out, n_c, layout.canvas_h, layout.canvas_w),
            dtype=volume.dtype,
            imagej=True,
            bigtiff=bigtiff,
            photometric="minisblack",
            resolution=resolution,
            resolutionunit="MICROMETER" if resolution else None,
            metadata=ij_meta,
        )
        try:
            for t in range(n_t):
                for z_out in range(n_z_out):
                    z_request = z_out if preserve_z else z_index
                    for c_idx, c in enumerate(channel_indices):
                        tiles: List[np.ndarray] = []
                        for m in m_indices:
                            try:
                                f = volume.get_frame(
                                    c=c, m=m, t=t, z=z_request,
                                    z_mode=z_mode,
                                )
                            except Exception:
                                f = np.zeros(
                                    (layout.tile_h, layout.tile_w),
                                    dtype=volume.dtype,
                                )
                            tiles.append(f)
                        canvas = stitch_one_frame(
                            tiles, layout, volume.dtype,
                        )
                        mm[t, z_out, c_idx] = canvas
                        pages_written += 1
                        if progress_cb is not None and (
                            pages_written % 4 == 0
                            or pages_written == n_pages_total
                        ):
                            progress_cb(
                                int(pages_written / n_pages_total * 95)
                            )
            # Force OS to push any cached writes before we hand the
            # file back to the user.
            try:
                mm.flush()
            except Exception:
                pass
        finally:
            del mm

    if progress_cb is not None:
        progress_cb(100)
# end of stitch_exporter.py
