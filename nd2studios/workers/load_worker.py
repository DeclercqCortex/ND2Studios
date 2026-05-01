"""
LoadWorker — open an ND2 (or TIFF) file in a background thread.

Returns:
    {
        "metadata": dict (extended ND2 metadata or TIFF-derived equivalent),
        "channels": {channel_name: LazyND2Channel | np.ndarray},
        "channel_names": list[str],
        "frame_timestamps_s": np.ndarray | None,
    }
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from nd2studios.workers.base_worker import BaseWorker


class LoadWorker(BaseWorker):
    """Open a file and emit a dict of raw channel data plus rich metadata."""

    def __init__(
        self,
        filepath: str,
        z_projection: str = "max",
        z_start: int = 0,
        z_end: Optional[int] = None,
        t_start: int = 0,
        t_end: Optional[int] = None,
        t_stride: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.filepath = filepath
        self.z_projection = z_projection
        self.z_start = z_start
        self.z_end = z_end
        self.t_start = t_start
        self.t_end = t_end
        self.t_stride = t_stride

    def run_task(self) -> Dict[str, Any]:
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext == ".nd2":
            return self._load_nd2()
        elif ext in (".tif", ".tiff"):
            return self._load_tiff()
        raise ValueError(f"Unsupported file extension: {ext}")

    # ── ND2 path ──
    def _load_nd2(self) -> Dict[str, Any]:
        from nd2studios.backend.nd2_loader import (
            read_nd2_metadata, read_nd2_metadata_extended,
            load_nd2_timeseries_lazy,
        )
        from nd2studios.backend.nd2_volume import LazyND2Volume

        self.set_status("Reading metadata…")
        meta = read_nd2_metadata(self.filepath)
        ext_meta = read_nd2_metadata_extended(self.filepath)
        self.set_progress(15)

        # V1.1: build the M/T/Z/C volume the viewer scrolls through.
        self.set_status("Opening lazy volume…")
        volume = LazyND2Volume(self.filepath)
        self.set_progress(25)

        n_channels = meta.n_channels
        channel_names: List[str] = list(ext_meta.get("channel_names")
                                        or meta.channel_names
                                        or [f"Ch{i}" for i in range(n_channels)])

        # Build per-channel (T, H, W) lazy proxies for the recipe pipeline.
        # We use the CURRENT z_projection setting and m_index = 0; the user
        # can change M from the viewer and we will rebuild these on Confirm.
        self.set_status(f"Building {n_channels} lazy channel proxies…")
        channels: Dict[str, Any] = {}
        for i, name in enumerate(channel_names):
            if self.cancelled:
                return {}
            channels[name] = load_nd2_timeseries_lazy(
                self.filepath,
                channel_index=i,
                t_start=self.t_start,
                t_end=self.t_end,
                t_stride=self.t_stride,
                z_start=self.z_start,
                z_end=self.z_end,
                z_projection=self.z_projection,
            )
            self.set_progress(25 + int(65 * (i + 1) / max(1, n_channels)))

        ts = ext_meta.get("frame_timestamps_s") or []
        ts_array = np.asarray(ts, dtype=np.float64) if ts else None

        self.set_progress(100)
        self.set_status("Done.")
        return {
            "metadata": ext_meta,
            "channels": channels,
            "channel_names": channel_names,
            "frame_timestamps_s": ts_array,
            "volume": volume,
            "source_type": "nd2",
        }

    # ── TIFF path ──
    def _load_tiff(self) -> Dict[str, Any]:
        from nd2studios.backend.tiff_loader import (
            get_tiff_info, load_tiff_stack_lazy,
        )

        self.set_status("Inspecting TIFF…")
        info = get_tiff_info(self.filepath)
        self.set_progress(30)

        # V1.0: assume TYX (single channel timeseries). Multi-channel TIFFs
        # are an extension point — we'll wire up channel-aware loading once
        # users surface real-world inputs that need it.
        self.set_status("Building lazy TIFF view…")
        lazy = load_tiff_stack_lazy(self.filepath)
        channels = {"Ch0": lazy}
        self.set_progress(90)

        sample = next(iter(channels.values()))
        meta = {
            "filepath": self.filepath,
            "height": sample.shape[-2],
            "width": sample.shape[-1],
            "n_timepoints": sample.shape[0] if sample.ndim == 3 else 1,
            "n_channels": len(channels),
            "n_zslices": 1,
            "n_multipoints": 1,
            "pixel_size_um": 1.0,
            "z_step_um": 1.0,
            "channel_names": list(channels.keys()),
            "channel_exposure_ms": [None] * len(channels),
            "channel_emission_nm": [None] * len(channels),
            "channel_excitation_nm": [None] * len(channels),
            "frame_timestamps_s": [],
            "stage_xy_um": [],
            "stage_z_um": [],
            "objective_name": "",
            "objective_magnification": None,
            "objective_na": None,
            "binning_x": None,
            "binning_y": None,
            "camera_name": "",
            "microscope_name": "",
            "loops": [],
        }
        return {
            "metadata": meta,
            "channels": channels,
            "channel_names": list(channels.keys()),
            "frame_timestamps_s": None,
            "volume": None,
            "source_type": "tiff",
        }
