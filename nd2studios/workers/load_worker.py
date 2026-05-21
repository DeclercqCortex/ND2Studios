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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.workers.base_worker import BaseWorker


class LoadWorker(BaseWorker):
    """Open a file (or set of files) and emit channels + rich metadata.

    Accepts either a single ``filepath`` or a list of ``filepaths``.
    When more than one path is supplied the worker takes the V1.28
    multi-file path: it builds a composite lazy volume that chains the
    files along ``chain_axis`` (one of ``T`` / ``M`` / ``Z`` / ``C``).
    The format is inferred from the file extensions — all files in a
    single import must share a format.
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        z_projection: str = "max",
        z_start: int = 0,
        z_end: Optional[int] = None,
        t_start: int = 0,
        t_end: Optional[int] = None,
        t_stride: int = 1,
        parent=None,
        filepaths: Optional[List[str]] = None,
        chain_axis: str = "Z",
        chain_mapping: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__(parent)
        if filepaths:
            self.filepaths: List[str] = list(filepaths)
            self.filepath: str = self.filepaths[0]
        elif filepath:
            self.filepaths = [filepath]
            self.filepath = filepath
        else:
            raise ValueError("LoadWorker needs filepath or filepaths")
        self.z_projection = z_projection
        self.z_start = z_start
        self.z_end = z_end
        self.t_start = t_start
        self.t_end = t_end
        self.t_stride = t_stride
        self.chain_axis = chain_axis
        self.chain_mapping = chain_mapping

    def run_task(self) -> Dict[str, Any]:
        if len(self.filepaths) > 1:
            exts = {os.path.splitext(p)[1].lower() for p in self.filepaths}
            if exts == {".nd2"}:
                return self._load_nd2_multi()
            if exts.issubset({".tif", ".tiff"}):
                return self._load_tiff_multi()
            raise ValueError(
                "Multi-file import requires all files to be the same "
                f"format (got extensions: {sorted(exts)})"
            )
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

    # ── ND2 multi-file path (V1.28 — chain along T/M/Z/C) ──
    def _load_nd2_multi(self) -> Dict[str, Any]:
        from nd2studios.backend.nd2_loader import (
            read_nd2_metadata_extended_multi,
        )
        from nd2studios.backend.nd2_volume import LazyMultiFileND2Volume

        n_files = len(self.filepaths)
        axis = self.chain_axis
        mapping = self.chain_mapping
        self.set_status(
            f"Combining metadata across {n_files} ND2 files (chain {axis})…"
        )
        ext_meta = read_nd2_metadata_extended_multi(
            self.filepaths, axis, chain_mapping=mapping,
        )
        self.set_progress(15)

        self.set_status(f"Opening composite volume (chain {axis})…")
        volume = LazyMultiFileND2Volume(
            self.filepaths, axis, chain_mapping=mapping,
        )
        self.set_progress(35)

        channel_names: List[str] = list(volume.channel_names)
        n_channels = volume.n_channels
        self.set_status(f"Building {n_channels} lazy channel proxies…")
        channels: Dict[str, Any] = {}
        for i, name in enumerate(channel_names):
            if self.cancelled:
                return {}
            channels[name] = volume.to_lazy_channel(
                c=i, m=0,
                z_mode=self.z_projection, z_index=0,
                z_start=self.z_start, z_end=self.z_end,
                t_start=self.t_start, t_end=self.t_end,
                t_stride=self.t_stride,
            )
            self.set_progress(35 + int(55 * (i + 1) / max(1, n_channels)))

        ts = ext_meta.get("frame_timestamps_s") or []
        ts_array = np.asarray(ts, dtype=np.float64) if ts else None

        self.set_progress(100)
        self.set_status(f"Done — {n_files} files chained on {axis}.")
        return {
            "metadata": ext_meta,
            "channels": channels,
            "channel_names": channel_names,
            "frame_timestamps_s": ts_array,
            "volume": volume,
            "source_type": "nd2_multi",
        }

    # ── TIFF multi-file path (V1.28) ──
    def _load_tiff_multi(self) -> Dict[str, Any]:
        from nd2studios.backend.tiff_loader import (
            LazyMultiFileTIFFVolume, read_tiff_meta_fast,
        )

        n_files = len(self.filepaths)
        axis = self.chain_axis
        self.set_status(
            f"Combining metadata across {n_files} TIFF files (chain {axis})…"
        )
        # File 0 metadata as a base for the import payload.
        sorted_paths = sorted(self.filepaths, key=lambda p: os.path.basename(p))
        f0 = read_tiff_meta_fast(sorted_paths[0])
        self.set_progress(15)

        self.set_status(f"Opening composite TIFF volume (chain {axis})…")
        volume = LazyMultiFileTIFFVolume(
            self.filepaths, axis, chain_mapping=self.chain_mapping,
        )
        self.set_progress(40)

        channel_names: List[str] = list(volume.channel_names)
        channels: Dict[str, Any] = {}
        for i, name in enumerate(channel_names):
            if self.cancelled:
                return {}
            channels[name] = volume.to_lazy_channel(
                c=i, m=0,
                z_mode=self.z_projection, z_index=0,
                t_start=self.t_start or 0, t_end=self.t_end,
                t_stride=self.t_stride,
            )
            self.set_progress(40 + int(50 * (i + 1) / max(1, len(channel_names))))

        meta = {
            "filepath": volume.filepath,
            "source_filepaths": list(volume.filepaths),
            "chain_axis": axis,
            "dtype": str(volume.dtype),
            "height": volume.height,
            "width": volume.width,
            "n_timepoints": volume.n_timepoints,
            "n_channels": volume.n_channels,
            "n_zslices": volume.n_zslices,
            "n_multipoints": volume.n_multipoints,
            "pixel_size_um": volume.pixel_size_um,
            "z_step_um": volume.z_step_um,
            "channel_names": channel_names,
            "channel_exposure_ms": [None] * len(channel_names),
            "channel_emission_nm": [None] * len(channel_names),
            "channel_excitation_nm": [None] * len(channel_names),
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

        self.set_progress(100)
        self.set_status(f"Done — {n_files} TIFFs chained on {axis}.")
        return {
            "metadata": meta,
            "channels": channels,
            "channel_names": channel_names,
            "frame_timestamps_s": None,
            "volume": volume,
            "source_type": "tiff_multi",
        }

    # ── TIFF path ──
    def _load_tiff(self) -> Dict[str, Any]:
        """Single-file TIFF load.

        Always builds a :class:`LazyMultiFileTIFFVolume` (with one
        member) so the multi-axis viewer can scroll Z exactly like it
        does for ND2 files. Z is preserved in the volume; the per-channel
        ``(T, H, W)`` proxies handed back honor the worker's
        ``z_projection`` setting so the recipe pipeline still sees a
        2-D timeseries.
        """
        from nd2studios.backend.tiff_loader import LazyMultiFileTIFFVolume

        self.set_status("Inspecting TIFF…")
        volume = LazyMultiFileTIFFVolume([self.filepath], chain_axis="Z")
        self.set_progress(30)

        channel_names: List[str] = list(volume.channel_names)
        n_channels = volume.n_channels
        self.set_status(f"Building {n_channels} lazy channel proxies…")
        channels: Dict[str, Any] = {}
        for i, name in enumerate(channel_names):
            if self.cancelled:
                return {}
            channels[name] = volume.to_lazy_channel(
                c=i, m=0,
                z_mode=self.z_projection, z_index=0,
                z_start=self.z_start or 0,
                z_end=self.z_end,
                t_start=self.t_start or 0,
                t_end=self.t_end,
                t_stride=self.t_stride,
            )
            self.set_progress(30 + int(60 * (i + 1) / max(1, n_channels)))

        meta = {
            "filepath": volume.filepath,
            "dtype": str(volume.dtype),
            "height": volume.height,
            "width": volume.width,
            "n_timepoints": volume.n_timepoints,
            "n_channels": volume.n_channels,
            "n_zslices": volume.n_zslices,
            "n_multipoints": volume.n_multipoints,
            "pixel_size_um": volume.pixel_size_um,
            "z_step_um": volume.z_step_um,
            "channel_names": channel_names,
            "channel_exposure_ms": [None] * len(channel_names),
            "channel_emission_nm": [None] * len(channel_names),
            "channel_excitation_nm": [None] * len(channel_names),
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
        self.set_progress(100)
        self.set_status("Done.")
        return {
            "metadata": meta,
            "channels": channels,
            "channel_names": channel_names,
            "frame_timestamps_s": None,
            "volume": volume,
            "source_type": "tiff",
        }
