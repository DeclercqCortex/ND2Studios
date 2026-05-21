"""
ND2 file loading with metadata extraction and Z-projection.

Originally adapted from CellTracker. ND2Studios extends it with
``read_nd2_metadata_extended()`` which captures exposure times, per-frame
timestamps, stage XY/Z positions, objective info, and binning — the
fields the McGhee Lab needs surfaced on the Import page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ND2Metadata:
    """Metadata extracted from an .nd2 file."""
    filepath: str = ""
    n_timepoints: int = 1
    n_zslices: int = 1
    n_channels: int = 1
    n_multipoints: int = 1
    height: int = 0
    width: int = 0
    dtype: np.dtype = np.dtype("uint16")
    pixel_size_um: float = 1.0
    z_step_um: float = 1.0
    channel_names: List[str] = field(default_factory=list)
    voxel_size_um: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def single_frame_bytes(self) -> int:
        return self.height * self.width * self.dtype.itemsize


def read_nd2_metadata(filepath: str) -> ND2Metadata:
    """Extract metadata from an ND2 file without loading pixel data."""
    import nd2

    meta = ND2Metadata(filepath=filepath)
    with nd2.ND2File(filepath) as f:
        meta.dtype = f.dtype
        sizes = f.sizes

        meta.n_timepoints = sizes.get("T", 1)
        meta.n_zslices = sizes.get("Z", 1)
        meta.n_channels = sizes.get("C", 1)
        meta.n_multipoints = sizes.get("P", sizes.get("M", 1))
        meta.height = sizes.get("Y", 0)
        meta.width = sizes.get("X", 0)

        try:
            vox = f.voxel_size()
            meta.pixel_size_um = vox.x if hasattr(vox, "x") else 1.0
            meta.z_step_um = vox.z if hasattr(vox, "z") else 1.0
            meta.voxel_size_um = (meta.z_step_um, meta.pixel_size_um, meta.pixel_size_um)
        except Exception:
            pass

        try:
            meta.channel_names = [ch.channel.name for ch in f.metadata.channels]
        except Exception:
            meta.channel_names = [f"Ch{i}" for i in range(meta.n_channels)]

    return meta


def load_nd2_timeseries(
    filepath: str,
    channel_index: int = 0,
    t_start: int = 0,
    t_end: Optional[int] = None,
    t_stride: int = 1,
    z_start: int = 0,
    z_end: Optional[int] = None,
    z_projection: str = "max",
    progress_cb=None,
) -> np.ndarray:
    """
    Load a single-channel 2D timeseries from an ND2 file.

    Reads T frames, applies Z-projection over [z_start, z_end) to produce
    a (T, H, W) array.

    Parameters
    ----------
    filepath : str
    channel_index : int, which channel to extract
    t_start, t_end : int, timepoint range (end exclusive)
    t_stride : int, keep every Nth frame (1 = every frame)
    z_start, z_end : int, Z-slice range for projection (end exclusive)
    z_projection : str, "max", "mean", "min", or "none" (single Z)
    progress_cb : callable(int), progress 0-100

    Returns
    -------
    np.ndarray (T, H, W)
    """
    import nd2

    with nd2.ND2File(filepath) as f:
        sizes = f.sizes
        dim_order = list(sizes.keys())

        T_total = sizes.get("T", 1)
        Z_total = sizes.get("Z", 1)
        H = sizes.get("Y", 0)
        W = sizes.get("X", 0)

        if t_end is None:
            t_end = T_total
        if z_end is None:
            z_end = Z_total

        t_stride = max(1, int(t_stride))
        t_indices = list(range(t_start, t_end, t_stride))
        n_frames = len(t_indices)
        result = np.zeros((n_frames, H, W), dtype=f.dtype)

        dask_arr = f.to_dask()  # shape matches dim_order, e.g. (T, Z, C, Y, X)

        def _build_index(t_val, z_val):
            """Build an indexing tuple for the dask array."""
            idx = []
            for d in dim_order:
                if d == "T":
                    idx.append(t_val)
                elif d == "C":
                    idx.append(channel_index)
                elif d == "Z":
                    idx.append(z_val)
                elif d in ("Y", "X"):
                    idx.append(slice(None))  # keep full spatial dims
                else:
                    idx.append(0)  # multipoint, large image, etc.
            return tuple(idx)

        for i, t in enumerate(t_indices):
            if Z_total > 1 and z_projection != "none":
                # Load Z range and project
                idx = _build_index(t, slice(z_start, z_end))
                z_stack = np.asarray(dask_arr[idx])  # shape: (Nz, H, W)

                if z_stack.ndim == 3:
                    if z_projection == "max":
                        result[i] = z_stack.max(axis=0)
                    elif z_projection == "mean":
                        result[i] = z_stack.mean(axis=0).astype(f.dtype)
                    elif z_projection == "min":
                        result[i] = z_stack.min(axis=0)
                    else:
                        result[i] = z_stack[0]
                elif z_stack.ndim == 2:
                    result[i] = z_stack
                else:
                    # Unexpected extra dims — squeeze them
                    squeezed = z_stack.squeeze()
                    if squeezed.ndim == 3:
                        result[i] = squeezed.max(axis=0) if z_projection == "max" else squeezed[0]
                    else:
                        result[i] = squeezed
            else:
                # Single Z slice
                z_val = z_start if "Z" in sizes else 0
                idx = _build_index(t, z_val)
                frame = np.asarray(dask_arr[idx])

                if frame.ndim == 2:
                    result[i] = frame
                else:
                    result[i] = frame.squeeze()

            if progress_cb and (i % 5 == 0 or i == n_frames - 1):
                progress_cb(int((i + 1) / n_frames * 100))

    return result


class LazyND2Channel:
    """
    Lazy (T, H, W) view of one channel of an ND2 file.

    Frames are read from disk on demand. Exposes enough of the numpy
    array protocol (shape, dtype, ndim, len, __getitem__) for downstream
    code that does per-frame indexing (`arr[t]`) or slicing
    (`arr[:, y0:y1, x0:x1]`) to work transparently.

    Calling np.asarray(...) materializes the whole stack — avoid in
    hot paths.
    """

    def __init__(self, filepath: str, channel_index: int,
                 t_start: int, t_end: int, z_start: int, z_end: int,
                 z_projection: str, height: int, width: int, dtype,
                 t_stride: int = 1, m_index: int = 0):
        self._filepath = filepath
        self._ch = channel_index
        self._m = int(m_index)
        self._t0 = t_start
        self._t1 = t_end
        self._t_stride = max(1, int(t_stride))
        self._z0 = z_start
        self._z1 = z_end
        self._zproj = z_projection
        # Number of frames after stride: ceil((t_end - t_start) / stride)
        n = max(0, (t_end - t_start + self._t_stride - 1) // self._t_stride)
        self.shape = (n, height, width)
        self.dtype = np.dtype(dtype)
        self.ndim = 3
        # Lazy file handle (opened on first read, kept open for reuse).
        self._file = None
        self._dask = None
        self._dim_order = None

    # ── file management ──
    def _ensure_open(self):
        if self._file is None:
            import nd2
            self._file = nd2.ND2File(self._filepath)
            self._dask = self._file.to_dask()
            self._dim_order = list(self._file.sizes.keys())
        return self._file

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._dask = None

    def __del__(self):
        self.close()

    # ── numpy-like API ──
    def __len__(self):
        return self.shape[0]

    def _read_frame(self, t_local: int) -> np.ndarray:
        """Read frame `t_local` (0-based, post-stride) and return (H, W)."""
        self._ensure_open()
        t_abs = self._t0 + t_local * self._t_stride
        sizes = self._file.sizes
        Z_total = sizes.get("Z", 1)

        def _idx(z_val):
            out = []
            for d in self._dim_order:
                if d == "T":          out.append(t_abs)
                elif d == "C":        out.append(self._ch)
                elif d == "Z":        out.append(z_val)
                elif d in ("P", "M"): out.append(self._m)
                elif d in ("Y", "X"): out.append(slice(None))
                else:                 out.append(0)
            return tuple(out)

        if Z_total > 1 and self._zproj != "none":
            z_stack = np.asarray(self._dask[_idx(slice(self._z0, self._z1))])
            if z_stack.ndim == 3:
                if self._zproj == "max":  return z_stack.max(axis=0)
                if self._zproj == "mean": return z_stack.mean(axis=0).astype(self.dtype)
                if self._zproj == "min":  return z_stack.min(axis=0)
                return z_stack[0]
            if z_stack.ndim == 2:
                return z_stack
            return z_stack.squeeze()
        else:
            z_val = self._z0 if "Z" in sizes else 0
            frame = np.asarray(self._dask[_idx(z_val)])
            return frame if frame.ndim == 2 else frame.squeeze()

    def __getitem__(self, key):
        """Support arr[t], arr[t, ...], arr[t0:t1], arr[:, y0:y1, x0:x1]."""
        # Normalize key to a tuple
        if not isinstance(key, tuple):
            key = (key,)

        # First axis = time
        t_key = key[0]
        spatial = key[1:] if len(key) > 1 else ()

        # Scalar time index → single (H, W) frame, then apply spatial slicing.
        if isinstance(t_key, (int, np.integer)):
            frame = self._read_frame(int(t_key))
            return frame[spatial] if spatial else frame

        # Slice or list/array of time indices → stack frames lazily.
        if isinstance(t_key, slice):
            indices = range(*t_key.indices(self.shape[0]))
        else:
            indices = list(t_key)

        frames = [self._read_frame(int(i)) for i in indices]
        if not frames:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)
        stacked = np.stack(frames, axis=0)
        return stacked[(slice(None),) + spatial] if spatial else stacked

    def __array__(self, dtype=None):
        """Materialize the entire stack — only for unavoidable cases."""
        full = np.stack([self._read_frame(t) for t in range(self.shape[0])], axis=0)
        return full.astype(dtype) if dtype is not None else full

    def copy(self) -> "LazyND2Channel":
        """Lazy proxies are read-only views — copy returns self.

        Existing code uses `v.copy()` to defend against in-place mutation
        of an ndarray. Since this proxy doesn't support mutation, returning
        self keeps the file handle shared instead of duplicating it.
        """
        return self

    def materialize(self) -> np.ndarray:
        """Read every frame and return a contiguous (T, H, W) ndarray.

        Safe to call from a worker thread even if the proxy has already
        been read from the main thread: this temporarily detaches the
        existing file handle, opens its own handle on the calling thread,
        reads, closes, then restores the original. The nd2 library's
        per-file state is not thread-safe, so sharing a handle across
        threads can segfault.
        """
        saved = (self._file, self._dask, self._dim_order)
        self._file = None
        self._dask = None
        self._dim_order = None
        try:
            result = np.stack(
                [self._read_frame(t) for t in range(self.shape[0])], axis=0,
            )
            # Close the handle this call opened — we don't want to leak it.
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
        finally:
            # Restore the original (likely main-thread) handle untouched.
            self._file, self._dask, self._dim_order = saved
        return result

    def crop(self, y0: int, y1: int, x0: int, x1: int) -> "LazyND2Channel":
        """Return a lazy view restricted to the given spatial bbox.
        Each frame read is sliced before being returned, so RAM use is
        proportional to the cropped frame size — not the original."""
        view = LazyND2Channel.__new__(LazyND2Channel)
        view._filepath = self._filepath
        view._ch = self._ch
        view._t0 = self._t0
        view._t1 = self._t1
        view._t_stride = self._t_stride
        view._z0 = self._z0
        view._z1 = self._z1
        view._zproj = self._zproj
        view.shape = (self.shape[0], y1 - y0, x1 - x0)
        view.dtype = self.dtype
        view.ndim = 3
        view._file = None
        view._dask = None
        view._dim_order = None
        # Wrap _read_frame to slice spatially.
        parent_read = self._read_frame
        view._read_frame = lambda t_local, _y0=y0, _y1=y1, _x0=x0, _x1=x1: \
            parent_read(t_local)[_y0:_y1, _x0:_x1]
        return view


def load_nd2_timeseries_lazy(
    filepath: str,
    channel_index: int = 0,
    t_start: int = 0,
    t_end: Optional[int] = None,
    t_stride: int = 1,
    z_start: int = 0,
    z_end: Optional[int] = None,
    z_projection: str = "max",
) -> LazyND2Channel:
    """Build a LazyND2Channel without reading any pixel data."""
    import nd2
    with nd2.ND2File(filepath) as f:
        sizes = f.sizes
        T_total = sizes.get("T", 1)
        Z_total = sizes.get("Z", 1)
        H = sizes.get("Y", 0)
        W = sizes.get("X", 0)
        dtype = f.dtype
    if t_end is None:
        t_end = T_total
    if z_end is None:
        z_end = Z_total
    return LazyND2Channel(
        filepath, channel_index, t_start, t_end, z_start, z_end,
        z_projection, H, W, dtype, t_stride=t_stride,
    )


def z_project(stack: np.ndarray, method: str = "max") -> np.ndarray:
    """
    Apply Z-projection to a stack with a Z dimension.

    Parameters
    ----------
    stack : np.ndarray with shape (..., Z, H, W)
    method : "max", "mean", "min"

    Returns
    -------
    np.ndarray with Z dimension collapsed
    """
    axis = -3  # Z is typically the third-to-last axis
    if method == "max":
        return


# ── Extended metadata ─────────────────────────────────────────────
# The `nd2` library exposes more than the bare ND2Metadata above:
# exposure times, per-frame timestamps, XY/Z stage positions, objective
# info, binning. The structure varies between SDK versions, so we read
# defensively and return whatever we can.

def _safe(get, default=None):
    """Run a getter that may explode on missing fields and return `default`."""
    try:
        return get()
    except Exception:
        return default


def read_nd2_metadata_extended(filepath):
    """Return a dict of all the metadata fields ND2Studios surfaces.

    Always-present keys (best-effort, may be empty):
        filepath, dim_order, sizes, dtype, height, width,
        n_timepoints, n_zslices, n_channels, n_multipoints,
        pixel_size_um, z_step_um, voxel_size_um,
        channel_names, channel_colors, channel_emission_nm,
        channel_excitation_nm, channel_exposure_ms,
        objective_name, objective_magnification, objective_na,
        objective_immersion, binning_x, binning_y,
        camera_name, microscope_name,
        acquisition_start, frame_timestamps_s,
        stage_xy_um, stage_z_um, loops
    """
    import nd2

    out = {"filepath": filepath}
    with nd2.ND2File(filepath) as f:
        sizes = dict(f.sizes)
        out["sizes"] = sizes
        out["dim_order"] = list(sizes.keys())
        out["dtype"] = str(f.dtype)
        out["height"] = sizes.get("Y", 0)
        out["width"] = sizes.get("X", 0)
        out["n_timepoints"] = sizes.get("T", 1)
        out["n_zslices"] = sizes.get("Z", 1)
        out["n_channels"] = sizes.get("C", 1)
        out["n_multipoints"] = sizes.get("P", sizes.get("M", 1))

        vox = _safe(f.voxel_size)
        if vox is not None:
            out["pixel_size_um"] = float(getattr(vox, "x", 1.0))
            out["z_step_um"] = float(getattr(vox, "z", 1.0))
            out["voxel_size_um"] = (
                float(getattr(vox, "z", 1.0)),
                float(getattr(vox, "y", out["pixel_size_um"])),
                float(getattr(vox, "x", out["pixel_size_um"])),
            )
        else:
            out["pixel_size_um"] = 1.0
            out["z_step_um"] = 1.0
            out["voxel_size_um"] = (1.0, 1.0, 1.0)

        meta = _safe(lambda: f.metadata)
        channels = _safe(lambda: list(meta.channels), default=[]) or []
        names = []
        colors = []
        emission = []
        excitation = []
        exposure_ms = []
        for ch in channels:
            ch_inner = getattr(ch, "channel", ch)
            names.append(str(getattr(ch_inner, "name", "Ch")))
            colors.append(_safe(lambda c=ch_inner: int(getattr(c, "colorRGB", None))))
            emission.append(_safe(lambda c=ch_inner: float(getattr(c, "emissionLambdaNm", None))))
            excitation.append(_safe(lambda c=ch_inner: float(getattr(c, "excitationLambdaNm", None))))
            exp = (
                _safe(lambda c=ch: float(getattr(c, "exposureTimeMs", None)))
                or _safe(lambda c=ch_inner: float(getattr(c, "exposureTimeMs", None)))
            )
            exposure_ms.append(exp)
        if not names:
            names = [f"Ch{i}" for i in range(out["n_channels"])]
        out["channel_names"] = names
        out["channel_colors"] = colors
        out["channel_emission_nm"] = emission
        out["channel_excitation_nm"] = excitation
        out["channel_exposure_ms"] = exposure_ms

        microscope = _safe(lambda: meta.channels[0].microscope) if channels else None
        if microscope is not None:
            out["objective_name"] = str(_safe(lambda: microscope.objectiveName) or "")
            out["objective_magnification"] = _safe(lambda: float(microscope.objectiveMagnification))
            out["objective_na"] = _safe(lambda: float(microscope.objectiveNumericalAperture))
            out["objective_immersion"] = str(_safe(lambda: microscope.immersionRefractiveIndex) or "")
        else:
            out["objective_name"] = ""
            out["objective_magnification"] = None
            out["objective_na"] = None
            out["objective_immersion"] = ""

        instrument = _safe(lambda: meta.channels[0].volume) if channels else None
        out["binning_x"] = _safe(lambda: int(instrument.cameraTransformationMatrix.binning))
        out["binning_y"] = out.get("binning_x")
        camera = _safe(lambda: meta.channels[0].instrument) if channels else None
        out["camera_name"] = str(_safe(lambda: camera.cameraName) or "")
        out["microscope_name"] = str(_safe(lambda: microscope.systemName) or "") if microscope else ""

        seq_dims = [d for d in out["dim_order"] if d not in ("Y", "X")]
        m_axis = "P" if "P" in seq_dims else ("M" if "M" in seq_dims else None)
        n_m = sizes.get(m_axis, 1) if m_axis else 1
        n_t = out["n_timepoints"]

        def _flat(coords: Dict[str, int]) -> int:
            idx = 0
            for d in seq_dims:
                idx = idx * sizes.get(d, 1) + int(coords.get(d, 0))
            return idx

        def _read_xy_from_experiment() -> List[Tuple[float, float]]:
            """Read planned stage XY positions from f.experiment (XYPosLoop)."""
            pts: List[Tuple[float, float]] = []
            for loop in (_safe(lambda: f.experiment, default=[]) or []):
                ltype = str(_safe(lambda lp=loop: lp.type) or "")
                if "XYPos" not in ltype:
                    continue
                points = (
                    _safe(lambda lp=loop: list(lp.parameters.points), default=[]) or []
                )
                for pt in points:
                    sx = _safe(lambda p=pt: float(p.stagePositionUm.x))
                    sy = _safe(lambda p=pt: float(p.stagePositionUm.y))
                    if sx is not None and sy is not None:
                        pts.append((sx, sy))
                if pts:
                    return pts
            return []

        # 1) Per-M stage positions — try f.experiment XYPosLoop first, then
        #    fall back to frame_metadata() per-M (which requires correct flat-
        #    index arithmetic). The experiment loop is the authoritative planned
        #    positions; frame_metadata is the actual per-frame readback.
        stage_xy: List[Tuple[float, float]] = []
        stage_z: List[float] = []

        stage_xy = _read_xy_from_experiment()
        xy_source_method = "experiment" if stage_xy else "frame_metadata"

        if not stage_xy and m_axis is not None:
            for m in range(n_m):
                coords = {m_axis: m, "T": 0, "Z": 0, "C": 0}
                seq_idx = _flat(coords)
                fm = _safe(lambda i=seq_idx: f.frame_metadata(i))
                if fm is None:
                    continue
                pos = (
                    _safe(lambda meta=fm: meta.channels[0].position) or
                    _safe(lambda meta=fm: meta.position)
                )
                if pos is None:
                    continue
                sx = _safe(lambda p=pos: float(p.stagePositionUm.x))
                sy = _safe(lambda p=pos: float(p.stagePositionUm.y))
                sz = _safe(lambda p=pos: float(p.stagePositionUm.z))
                if sx is not None and sy is not None:
                    stage_xy.append((sx, sy))
                if sz is not None:
                    stage_z.append(sz)
        elif not stage_xy:
            # Single-M file — record one nominal position from frame 0.
            fm0 = _safe(lambda: f.frame_metadata(0))
            if fm0 is not None:
                pos = (
                    _safe(lambda meta=fm0: meta.channels[0].position) or
                    _safe(lambda meta=fm0: meta.position)
                )
                if pos is not None:
                    sx = _safe(lambda p=pos: float(p.stagePositionUm.x))
                    sy = _safe(lambda p=pos: float(p.stagePositionUm.y))
                    if sx is not None and sy is not None:
                        stage_xy.append((sx, sy))

        n_xy_from_stage = len(stage_xy)

        # 2) Per-T frame timestamps.
        frame_ts: List[float] = []
        for t in range(n_t):
            coords = {"T": t, "Z": 0, "C": 0}
            if m_axis is not None:
                coords[m_axis] = 0
            seq_idx = _flat(coords)
            fm = _safe(lambda i=seq_idx: f.frame_metadata(i))
            if fm is None:
                continue
            ts = (
                _safe(lambda meta=fm: float(meta.channels[0].time.relativeTimeMs)) or
                _safe(lambda meta=fm: float(meta.relativeTimeMs))
            )
            if ts is not None:
                frame_ts.append(ts / 1000.0)

        if n_xy_from_stage == n_m and n_m > 0:
            stage_layout_source = f"stage_xy:{xy_source_method}"
        elif n_xy_from_stage > 0:
            stage_layout_source = f"partial:{n_xy_from_stage}/{n_m}:{xy_source_method}"
        else:
            stage_layout_source = "missing"

        out["frame_timestamps_s"] = frame_ts
        out["stage_xy_um"] = stage_xy
        out["stage_z_um"] = stage_z
        out["stage_layout_source"] = stage_layout_source
        out["acquisition_start"] = ""
        out["loops"] = [
            {
                "type": str(_safe(lambda lp=lp: lp.type) or ""),
                "count": int(_safe(lambda lp=lp: lp.count) or 0),
            }
            for lp in (_safe(lambda: f.experiment, default=[]) or [])
        ]

    return out


def read_nd2_metadata_extended_multi(
    filepaths: List[str],
    chain_axis: str = "Z",
    chain_mapping: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Extended metadata for a multi-file reconstruction (V1.28).

    Reads file 0's full extended metadata as the base, then overrides
    just the fields tied to ``chain_axis`` so the combined dataset
    reports the right total size and per-slice spacing/labels.

    ``chain_mapping`` (V1.31) is a list of ``(file_idx, local_idx)``
    pairs that defines the output order of slices on the chain axis.
    When ``None``, files concatenate in basename order. When provided,
    the per-axis overrides walk the mapping so timestamps / stage XY /
    channel names / stage Z reflect the user-chosen order.
    """
    if not filepaths:
        raise ValueError("read_nd2_metadata_extended_multi needs >=1 filepath")
    if chain_axis not in ("T", "M", "Z", "C"):
        raise ValueError(f"unknown chain_axis {chain_axis!r}")

    import os as _os
    paths = sorted(filepaths, key=lambda p: _os.path.basename(p))

    # Per-file probes — we read full extended metadata for every file
    # because the mapping may pull individual slices from any of them.
    per_file_meta: List[Dict[str, Any]] = []
    for p in paths:
        try:
            per_file_meta.append(read_nd2_metadata_extended(p))
        except Exception:
            per_file_meta.append({})

    # Default natural mapping per axis: file by file, in-file order.
    if chain_mapping is None:
        attr_map = {
            "T": "n_timepoints", "M": "n_multipoints",
            "Z": "n_zslices", "C": "n_channels",
        }
        attr = attr_map[chain_axis]
        chain_mapping = [
            (fi, li) for fi, fm in enumerate(per_file_meta)
            for li in range(int(fm.get(attr, 1)))
        ]

    base = dict(per_file_meta[0]) if per_file_meta[0] else read_nd2_metadata_extended(paths[0])
    base["source_filepaths"] = list(paths)
    base["chain_axis"] = chain_axis
    base["chain_mapping"] = list(chain_mapping)
    n_total = len(chain_mapping)

    if chain_axis == "Z":
        base["n_zslices"] = n_total
        per_file_z: List[Optional[float]] = []
        for p in paths:
            try:
                import nd2
                with nd2.ND2File(p) as f:
                    evts = list(f.events())
                if evts:
                    per_file_z.append(float(evts[0].get("Z Coord [µm]")) if evts[0].get("Z Coord [µm]") is not None else None)
                else:
                    per_file_z.append(None)
            except Exception:
                per_file_z.append(None)
        zs_ordered = [per_file_z[fi] for fi, _li in chain_mapping
                      if 0 <= fi < len(per_file_z) and per_file_z[fi] is not None]
        if len(zs_ordered) >= 2:
            diffs = np.diff(np.asarray(zs_ordered, dtype=np.float64))
            step = float(np.mean(np.abs(diffs)))
            if step > 0:
                base["z_step_um"] = step
                vox = base.get("voxel_size_um", (1.0, 1.0, 1.0))
                base["voxel_size_um"] = (step, vox[1], vox[2])
            base["stage_z_um"] = zs_ordered

    elif chain_axis == "T":
        base["n_timepoints"] = n_total
        ts_ordered: List[float] = []
        for fi, li in chain_mapping:
            if not (0 <= fi < len(per_file_meta)):
                ts_ordered.append(0.0)
                continue
            file_ts = per_file_meta[fi].get("frame_timestamps_s") or []
            if 0 <= li < len(file_ts):
                ts_ordered.append(float(file_ts[li]))
            elif file_ts:
                ts_ordered.append(float(file_ts[0]))
            else:
                ts_ordered.append(0.0)
        base["frame_timestamps_s"] = ts_ordered

    elif chain_axis == "M":
        base["n_multipoints"] = n_total
        xy_ordered: List[Tuple[float, float]] = []
        for fi, li in chain_mapping:
            if not (0 <= fi < len(per_file_meta)):
                continue
            file_xy = per_file_meta[fi].get("stage_xy_um") or []
            if 0 <= li < len(file_xy):
                xy_ordered.append(tuple(file_xy[li]))
        if xy_ordered:
            base["stage_xy_um"] = xy_ordered

    elif chain_axis == "C":
        base["n_channels"] = n_total
        names: List[str] = []
        exposures: List[Optional[float]] = []
        emissions: List[Optional[float]] = []
        excitations: List[Optional[float]] = []
        colors: List[Optional[int]] = []
        for fi, li in chain_mapping:
            if not (0 <= fi < len(per_file_meta)):
                continue
            fm = per_file_meta[fi]

            def _pick(lst, default=None):
                lst = lst or []
                return lst[li] if 0 <= li < len(lst) else default

            names.append(_pick(fm.get("channel_names"), f"C{li}") or f"C{li}")
            exposures.append(_pick(fm.get("channel_exposure_ms")))
            emissions.append(_pick(fm.get("channel_emission_nm")))
            excitations.append(_pick(fm.get("channel_excitation_nm")))
            colors.append(_pick(fm.get("channel_colors")))
        if names:
            base["channel_names"] = names
        if any(e is not None for e in exposures):
            base["channel_exposure_ms"] = exposures
        if any(e is not None for e in emissions):
            base["channel_emission_nm"] = emissions
        if any(e is not None for e in excitations):
            base["channel_excitation_nm"] = excitations
        if any(c is not None for c in colors):
            base["channel_colors"] = colors

    return base
