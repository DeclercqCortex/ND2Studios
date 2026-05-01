"""
Background worker for stitched-TIFF export.

Wraps :func:`export_stitched_tiff` so the GUI can show progress and
status messages while stitching. The worker accepts a
:class:`LazyND2Volume`, a layout, the chosen M and channel indices,
and an output path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nd2studios.backend.exporters.stitch_exporter import (
    StitchLayout, export_stitched_tiff,
)
from nd2studios.backend.nd2_volume import LazyND2Volume
from nd2studios.workers.base_worker import BaseWorker


@dataclass
class StitchRequest:
    volume: LazyND2Volume
    layout: StitchLayout
    m_indices: List[int]
    channel_indices: List[int]
    channel_colors: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    filepath: str = ""
    z_mode: str = "max"
    z_index: int = 0
    rgb: bool = True
    pixel_size_um: Optional[float] = None


class StitchWorker(BaseWorker):
    """Run a stitch+export job in the background."""

    def __init__(self, request: StitchRequest, parent=None):
        super().__init__(parent)
        self.request = request

    def run_task(self) -> str:
        req = self.request
        self.set_status("Stitching tiles…")
        export_stitched_tiff(
            volume=req.volume,
            layout=req.layout,
            m_indices=req.m_indices,
            channel_indices=req.channel_indices,
            channel_colors=req.channel_colors,
            filepath=req.filepath,
            z_mode=req.z_mode,
            z_index=req.z_index,
            rgb=req.rgb,
            pixel_size_um=req.pixel_size_um,
            progress_cb=self.set_progress,
        )
        return req.filepath
