"""
ExportWorker — write a deliverable to disk in a background thread.

Supports five modes:
- "tiff_stack": single multi-channel ImageJ TZCYX hyperstack
  (`export_tiff_hyperstack`). One file with all enabled channels.
- "tiff_zstack": same hyperstack shape, but Z > 1 — pulled from the raw
  volume per (M, T, Z) request.
- "rgb_composite": multi-channel RGB composite TIFF.
- "movie": MP4 / GIF time-lapse with optional overlays.
- "image_sequence": one PNG per frame, naming includes only axes with
  more than one position (``name_T01_M02_Z03.png``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.workers.base_worker import BaseWorker
from nd2studios.backend.exporters.composite_exporter import ImageAdjustments
from nd2studios.backend.exporters.movie_exporter import MovieOptions


@dataclass
class ExportRequest:
    """Description of one export job."""
    mode: str                       # "tiff_stack" | "tiff_zstack" | "rgb_composite" | "movie" | "image_sequence"
    filepath: str                   # output file (or directory, for image_sequence)
    channels: Dict[str, np.ndarray] = field(default_factory=dict)
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    enabled: Dict[str, bool] = field(default_factory=dict)
    pixel_size_um: float = 1.0
    frame_timestamps_s: Optional[np.ndarray] = None
    # Per-channel LUT settings: {name: (lo, hi, gamma)}. When populated, the
    # exporter uses these bounds instead of the auto percentile stretch so the
    # output matches what the user sees in the viewer.
    lut_settings: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    # Post-composite brightness / contrast / saturation / hue / fade. Shared
    # by movie, image_sequence, and rgb_composite when populated.
    image_adjustments: Optional[ImageAdjustments] = None
    # Mode-specific:
    bit_depth: str = "passthrough"          # tiff_stack / tiff_zstack
    movie_options: Optional[MovieOptions] = None  # movie / image_sequence (overlays)
    # tiff_zstack / image_sequence: raw volume + which M position + optional crop.
    raw_volume: Optional[Any] = None
    m_index: int = 0
    crop_rect: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    # image_sequence only.
    basename: str = "frame"
    iterate_volume: bool = False      # if True and raw_volume set, iterate (M, T, Z)
    z_mode: str = "none"
    z_view_index: int = 0


class ExportWorker(BaseWorker):
    """Run one ExportRequest in a background thread."""

    def __init__(self, request: ExportRequest, parent=None):
        super().__init__(parent)
        self.request = request

    def run_task(self) -> str:
        req = self.request
        if req.mode == "tiff_stack":
            return self._export_tiff_stack(req)
        if req.mode == "tiff_zstack":
            return self._export_tiff_zstack(req)
        if req.mode == "rgb_composite":
            return self._export_rgb_composite(req)
        if req.mode == "movie":
            return self._export_movie(req)
        if req.mode == "image_sequence":
            return self._export_image_sequence(req)
        raise ValueError(f"Unknown export mode: {req.mode}")

    def _export_tiff_stack(self, req: ExportRequest) -> str:
        """Write all enabled channels as ONE ImageJ TZCYX hyperstack.

        Matches the file construction produced by ``export_stitched_tiff``:
        single multi-channel ``.tif`` with ``Labels=[...]``, ``unit=um``,
        ``spacing``, and resolution tags.
        """
        from nd2studios.backend.exporters.tiff_exporter import (
            export_tiff_hyperstack,
        )
        import os

        path = req.filepath
        if not path.lower().endswith((".tif", ".tiff")):
            path += ".tif"

        self.set_status(f"Writing {os.path.basename(path)}…")
        return export_tiff_hyperstack(
            req.channels,
            req.enabled,
            path,
            bit_depth=req.bit_depth,
            pixel_size_um=req.pixel_size_um,
            progress_cb=self.set_progress,
        )

    def _export_tiff_zstack(self, req: ExportRequest) -> str:
        """Build a single ``(T, Z, C, H, W)`` ImageJ hyperstack from the raw volume."""
        from nd2studios.backend.exporters.tiff_exporter import (
            export_tiff_hyperstack,
        )
        import os

        vol = req.raw_volume
        path = req.filepath
        if not path.lower().endswith((".tif", ".tiff")):
            path += ".tif"

        enabled_channels = [
            (c, name) for c, name in enumerate(vol.channel_names)
            if req.enabled.get(name, True)
        ]
        if not enabled_channels:
            raise ValueError("Z-stack export: no enabled channels")

        n_ch = len(enabled_channels)
        n_t = vol.n_timepoints
        cx, cy, cw, ch_px = req.crop_rect if req.crop_rect else (0, 0, 0, 0)

        channels: Dict[str, np.ndarray] = {}
        enabled: Dict[str, bool] = {}
        for ch_i, (c, name) in enumerate(enabled_channels):
            if self.cancelled:
                break
            self.set_status(f"Reading Z stack: {name}…")
            frames: List[np.ndarray] = []
            for t in range(n_t):
                z_frames = []
                for z in range(vol.n_zslices):
                    frame = vol.get_frame(c=c, m=req.m_index, t=t, z=z, z_mode="none")
                    if req.crop_rect is not None:
                        frame = frame[cy:cy + ch_px, cx:cx + cw]
                    z_frames.append(frame)
                frames.append(np.stack(z_frames, axis=0))  # (Z, H, W)
                self.set_progress(
                    int((ch_i / n_ch + (t + 1) / n_t / n_ch) * 80)
                )
            channels[name] = np.stack(frames, axis=0)  # (T, Z, H, W)
            enabled[name] = True

        self.set_status(f"Writing {os.path.basename(path)}…")
        return export_tiff_hyperstack(
            channels,
            enabled,
            path,
            bit_depth=req.bit_depth,
            pixel_size_um=req.pixel_size_um,
            progress_cb=lambda p: self.set_progress(80 + int(p * 0.2)),
        )

    def _export_rgb_composite(self, req: ExportRequest) -> str:
        from nd2studios.backend.exporters.composite_exporter import export_rgb_composite_tiff

        self.set_status("Writing RGB composite TIFF…")
        export_rgb_composite_tiff(
            req.channels,
            req.colors,
            req.enabled,
            req.filepath,
            pixel_size_um=req.pixel_size_um,
            lut_settings=req.lut_settings or None,
            image_adjustments=req.image_adjustments,
            progress_cb=self.set_progress,
        )
        return req.filepath

    def _export_movie(self, req: ExportRequest) -> str:
        from nd2studios.backend.exporters.movie_exporter import export_movie

        self.set_status("Rendering movie…")
        export_movie(
            req.channels,
            req.colors,
            req.enabled,
            req.filepath,
            options=req.movie_options or MovieOptions(),
            pixel_size_um=req.pixel_size_um,
            frame_timestamps_s=req.frame_timestamps_s,
            lut_settings=req.lut_settings or None,
            image_adjustments=req.image_adjustments,
            progress_cb=self.set_progress,
            status_cb=self.set_status,
        )
        return req.filepath

    def _export_image_sequence(self, req: ExportRequest) -> str:
        from nd2studios.backend.exporters.image_sequence_exporter import (
            ImageSequenceRequest, export_image_sequence,
        )

        seq_req = ImageSequenceRequest(
            output_dir=req.filepath,
            basename=req.basename or "frame",
            channels=req.channels if not req.iterate_volume else {},
            raw_volume=req.raw_volume if req.iterate_volume else None,
            z_mode=req.z_mode,
            z_view_index=req.z_view_index,
            crop_rect=req.crop_rect,
            colors=req.colors,
            enabled=req.enabled,
            lut_settings=req.lut_settings,
            image_adjustments=req.image_adjustments,
            movie_options=req.movie_options or MovieOptions(),
            pixel_size_um=req.pixel_size_um,
            frame_timestamps_s=req.frame_timestamps_s,
        )
        self.set_status("Writing image sequence…")
        return export_image_sequence(
            seq_req,
            progress_cb=self.set_progress,
            status_cb=self.set_status,
        )
