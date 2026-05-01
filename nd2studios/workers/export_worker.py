"""
ExportWorker — write a deliverable to disk in a background thread.

Supports three modes:
- "tiff_stack": single-channel multi-page TIFF (one file per channel,
  named with a suffix).
- "rgb_composite": multi-channel RGB composite TIFF.
- "movie": MP4 / GIF time-lapse with optional overlays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.workers.base_worker import BaseWorker
from nd2studios.backend.exporters.movie_exporter import MovieOptions


@dataclass
class ExportRequest:
    """Description of one export job."""
    mode: str                       # "tiff_stack" | "rgb_composite" | "movie"
    filepath: str                   # output file (suffix appended for multi-channel TIFF)
    channels: Dict[str, np.ndarray] = field(default_factory=dict)
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    enabled: Dict[str, bool] = field(default_factory=dict)
    pixel_size_um: float = 1.0
    frame_timestamps_s: Optional[np.ndarray] = None
    # Mode-specific:
    bit_depth: str = "passthrough"          # tiff_stack
    movie_options: Optional[MovieOptions] = None  # movie


class ExportWorker(BaseWorker):
    """Run one ExportRequest in a background thread."""

    def __init__(self, request: ExportRequest, parent=None):
        super().__init__(parent)
        self.request = request

    def run_task(self) -> str:
        req = self.request
        if req.mode == "tiff_stack":
            return self._export_tiff_stack(req)
        if req.mode == "rgb_composite":
            return self._export_rgb_composite(req)
        if req.mode == "movie":
            return self._export_movie(req)
        raise ValueError(f"Unknown export mode: {req.mode}")

    def _export_tiff_stack(self, req: ExportRequest) -> str:
        from nd2studios.backend.exporters.tiff_exporter import export_tiff_stack
        import os

        # One file per enabled channel, with the channel name as a suffix.
        base, ext = os.path.splitext(req.filepath)
        if ext.lower() not in (".tif", ".tiff"):
            ext = ".tif"
        written: List[str] = []

        enabled_names = [n for n in req.channels if req.enabled.get(n, True)]
        n = max(1, len(enabled_names))
        for i, name in enumerate(enabled_names):
            if self.cancelled:
                break
            path = f"{base}_{_sanitize(name)}{ext}" if len(enabled_names) > 1 else f"{base}{ext}"
            self.set_status(f"Writing {os.path.basename(path)}…")
            export_tiff_stack(
                req.channels[name],
                path,
                bit_depth=req.bit_depth,
                pixel_size_um=req.pixel_size_um,
                progress_cb=lambda p, k=i, total=n: self.set_progress(
                    int((k + p / 100) / total * 100)
                ),
            )
            written.append(path)

        return "; ".join(written)

    def _export_rgb_composite(self, req: ExportRequest) -> str:
        from nd2studios.backend.exporters.composite_exporter import export_rgb_composite_tiff

        self.set_status("Writing RGB composite TIFF…")
        export_rgb_composite_tiff(
            req.channels,
            req.colors,
            req.enabled,
            req.filepath,
            pixel_size_um=req.pixel_size_um,
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
            progress_cb=self.set_progress,
        )
        return req.filepath


def _sanitize(name: str) -> str:
    """Make a channel name filesystem-safe."""
    keep = "-_.()"
    return "".join(c if c.isalnum() or c in keep else "_" for c in name)
