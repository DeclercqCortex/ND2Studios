"""
TIFF stack loading with dimension assignment.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tifffile


def get_tiff_info(filepath: str) -> dict:
    """Get shape and dtype of a TIFF stack without loading all data."""
    with tifffile.TiffFile(filepath) as tif:
        n_pages = len(tif.pages)
        page0 = tif.pages[0]
        return {
            "filepath": filepath,
            "n_pages": n_pages,
            "page_shape": page0.shape,
            "dtype": str(page0.dtype),
            "shape": (n_pages,) + page0.shape,
        }


def load_tiff_stack(filepath: str, key=None) -> np.ndarray:
    """Load a TIFF stack. Optionally select specific pages with key."""
    return tifffile.imread(filepath, key=key)


def load_tiff_frame(filepath: str, frame_idx: int) -> np.ndarray:
    """Load a single frame from a TIFF stack."""
    return tifffile.imread(filepath, key=frame_idx)


class LazyTIFFChannel:
    """
    Lazy (T, H, W) view of a TIFF stack. Frames are read on demand.

    Mirrors the LazyND2Channel API (shape/dtype/__getitem__/__len__).
    The TiffFile handle is opened on the first read and kept open for
    the lifetime of the object to avoid per-frame open/close overhead.
    """

    def __init__(self, filepath: str, t_start: int, t_end: int,
                 height: int, width: int, dtype, t_stride: int = 1,
                 n_pages_per_t: int = 1, page_within_t: int = 0,
                 n_z: int = 1, z_projection: str = "max",
                 z_stride: int = 1):
        self._filepath = filepath
        self._t0 = t_start
        self._t1 = t_end
        self._t_stride = max(1, int(t_stride))
        self._n_pages_per_t = max(1, int(n_pages_per_t))
        self._page_within_t = int(page_within_t)
        self._n_z = max(1, int(n_z))
        self._z_projection = z_projection
        # Page stride between consecutive Z planes for the same (T, C). 1
        # for single-channel multi-Z files; `n_c` for multi-channel ImageJ
        # TZCYX hyperstacks (page = t*n_z*n_c + z*n_c + c).
        self._z_stride = max(1, int(z_stride))
        n = max(0, (t_end - t_start + self._t_stride - 1) // self._t_stride)
        self.shape = (n, height, width)
        self.dtype = np.dtype(dtype)
        self.ndim = 3
        self._tiff: Optional[tifffile.TiffFile] = None

    def _ensure_open(self) -> None:
        if self._tiff is None:
            self._tiff = tifffile.TiffFile(self._filepath)

    def close(self) -> None:
        if self._tiff is not None:
            try:
                self._tiff.close()
            except Exception:
                pass
            self._tiff = None

    def __del__(self) -> None:
        self.close()

    def __len__(self):
        return self.shape[0]

    def _read_frame(self, t_local: int) -> np.ndarray:
        base_page = ((self._t0 + int(t_local) * self._t_stride)
                     * self._n_pages_per_t + self._page_within_t)
        self._ensure_open()
        if self._n_z <= 1:
            return self._tiff.pages[base_page].asarray()
        z_stack = np.stack(
            [self._tiff.pages[base_page + z * self._z_stride].asarray()
             for z in range(self._n_z)],
            axis=0,
        )
        if self._z_projection == "max":
            return z_stack.max(axis=0)
        if self._z_projection == "mean":
            return z_stack.mean(axis=0).astype(z_stack.dtype)
        if self._z_projection == "min":
            return z_stack.min(axis=0)
        return z_stack[self._n_z // 2]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        t_key = key[0]
        spatial = key[1:] if len(key) > 1 else ()

        if isinstance(t_key, (int, np.integer)):
            frame = self._read_frame(int(t_key))
            return frame[spatial] if spatial else frame

        if isinstance(t_key, slice):
            indices = range(*t_key.indices(self.shape[0]))
        else:
            indices = list(t_key)

        frames = [self._read_frame(i) for i in indices]
        if not frames:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)
        stacked = np.stack(frames, axis=0)
        return stacked[(slice(None),) + spatial] if spatial else stacked

    def __array__(self, dtype=None):
        full = np.stack([self._read_frame(t) for t in range(self.shape[0])], axis=0)
        return full.astype(dtype) if dtype is not None else full

    def copy(self) -> "LazyTIFFChannel":
        """Lazy proxies are read-only views — copy returns self."""
        return self

    def materialize(self) -> np.ndarray:
        """Read every frame and return a contiguous (T, H, W) ndarray."""
        return np.stack([self._read_frame(t) for t in range(self.shape[0])], axis=0)

    def crop(self, y0: int, y1: int, x0: int, x1: int) -> "LazyTIFFChannel":
        """Return a lazy view restricted to the given spatial bbox."""
        view = LazyTIFFChannel.__new__(LazyTIFFChannel)
        view._filepath = self._filepath
        view._t0 = self._t0
        view._t1 = self._t1
        view._t_stride = self._t_stride
        view._n_pages_per_t = self._n_pages_per_t
        view._page_within_t = self._page_within_t
        view.shape = (self.shape[0], y1 - y0, x1 - x0)
        view.dtype = self.dtype
        view.ndim = 3
        parent_read = self._read_frame
        view._read_frame = lambda t_local, _y0=y0, _y1=y1, _x0=x0, _x1=x1: \
            parent_read(t_local)[_y0:_y1, _x0:_x1]
        return view


def load_tiff_stack_lazy(filepath: str, t_start: int = 0,
                         t_end: Optional[int] = None,
                         t_stride: int = 1) -> LazyTIFFChannel:
    """Build a LazyTIFFChannel without reading frames."""
    info = get_tiff_info(filepath)
    n_pages = info["n_pages"]
    page_shape = info["page_shape"]
    if t_end is None or t_end > n_pages:
        t_end = n_pages
    h, w = page_shape[-2], page_shape[-1]
    return LazyTIFFChannel(filepath, t_start, t_end, h, w,
                           np.dtype(info["dtype"]), t_stride=t_stride)


def read_imagej_tiff_metadata(filepath: str) -> dict:
    """Read ImageJ metadata from a TIFF written with tifffile imagej=True.

    Returns a dict with keys: n_channels, n_timepoints, n_zslices,
    channel_names, pixel_size_um, n_pages_total, page_shape, dtype.
    Returns {} for non-ImageJ TIFFs so callers can fall back gracefully.
    """
    with tifffile.TiffFile(filepath) as tif:
        if not tif.is_imagej:
            return {}
        ij = tif.imagej_metadata or {}
        n_c = int(ij.get("channels", 1))
        n_t = int(ij.get("frames", 1))
        n_z = int(ij.get("slices", 1))
        labels = ij.get("Labels") or []
        if len(labels) < n_c:
            labels = list(labels) + [f"Ch{i}" for i in range(len(labels), n_c)]
        page0 = tif.pages[0]
        pixel_size_um: Optional[float] = None
        x_res = page0.tags.get("XResolution")
        if x_res is not None:
            val = x_res.value
            if isinstance(val, tuple) and len(val) == 2 and val[0]:
                pixel_size_um = float(val[1]) / float(val[0])
        return {
            "n_channels": n_c,
            "n_timepoints": n_t,
            "n_zslices": n_z,
            "channel_names": labels[:n_c],
            "pixel_size_um": pixel_size_um,
            "n_pages_total": len(tif.pages),
            "page_shape": page0.shape,
            "dtype": str(page0.dtype),
        }


def load_imagej_tiff_channels(
    filepath: str,
    ij_info: dict,
    t_start: int = 0,
    t_end: Optional[int] = None,
    t_stride: int = 1,
    z_projection: str = "max",
) -> Dict[str, "LazyTIFFChannel"]:
    """Build one LazyTIFFChannel per channel for an ImageJ multi-channel TIFF.

    Page layout written by tifffile with imagej=True and shape (T, Z, C, H, W):
        page index = t * n_z * n_c + z * n_c + c

    When Z > 1, each per-channel proxy projects across Z using
    ``z_projection`` (``"max"`` / ``"mean"`` / ``"min"`` / anything else
    → middle slice) so the returned (T, H, W) array is recipe-pipeline
    ready. Z stride between consecutive planes of the same (T, C) is
    ``n_c``.
    """
    n_c = ij_info["n_channels"]
    n_z = ij_info["n_zslices"]
    n_t = ij_info["n_timepoints"]
    page_shape = ij_info["page_shape"]
    h, w = page_shape[-2], page_shape[-1]
    dtype = np.dtype(ij_info["dtype"])
    names: list = ij_info["channel_names"]

    if t_end is None or t_end > n_t:
        t_end = n_t

    n_pages_per_t = n_z * n_c
    channels: Dict[str, LazyTIFFChannel] = {}
    for c_idx, name in enumerate(names[:n_c]):
        channels[name] = LazyTIFFChannel(
            filepath, t_start, t_end, h, w, dtype,
            t_stride=t_stride,
            n_pages_per_t=n_pages_per_t,
            page_within_t=c_idx,
            n_z=n_z,
            z_stride=n_c,
            z_projection=z_projection,
        )
    return channels


def assign_dimensions(
    stack: np.ndarray,
    dim_order: str = "TYX",
) -> dict:
    """
    Assign semantic meaning to stack dimensions.

    Parameters
    ----------
    stack : np.ndarray, loaded TIFF data
    dim_order : str, e.g. "TYX", "TZYX", "TCYX", "TZCYX"

    Returns
    -------
    dict with keys like 'T', 'Z', 'C', 'Y', 'X' mapping to sizes
    """
    if len(dim_order) != stack.ndim:
        raise ValueError(
            f"dim_order '{dim_order}' has {len(dim_order)} dims "
            f"but stack has {stack.ndim} dims (shape {stack.shape})"
        )
    return {dim: stack.shape[i] for i, dim in enumerate(dim_order)}


def extract_2d_timeseries(
    stack: np.ndarray,
    dim_order: str = "TYX",
    channel_index: int = 0,
    z_start: int = 0,
    z_end: Optional[int] = None,
    z_projection: str = "max",
) -> np.ndarray:
    """
    Extract a 2D timeseries (T, H, W) from a multi-dimensional TIFF stack.

    Parameters
    ----------
    stack : np.ndarray
    dim_order : str, dimension labels matching stack axes
    channel_index : int, which C index to extract
    z_start, z_end : int, Z range for projection
    z_projection : str, "max", "mean", "min"

    Returns
    -------
    np.ndarray (T, H, W)
    """
    dims = list(dim_order.upper())

    # Build a slice tuple
    slices = []
    z_axis = None
    for i, d in enumerate(dims):
        if d == "C":
            slices.append(channel_index)
        elif d == "Z":
            z_axis = i
            if z_end is None:
                slices.append(slice(z_start, None))
            else:
                slices.append(slice(z_start, z_end))
        else:
            slices.append(slice(None))

    sub = stack[tuple(slices)]

    # Apply Z projection if we still have a Z axis
    if z_axis is not None and sub.ndim > 3:
        # Find the Z axis position after slicing removed the C axis
        remaining_dims = [d for d in dims if d != "C"]
        z_pos = remaining_dims.index("Z")
        if z_projection == "max":
            sub = sub.max(axis=z_pos)
        elif z_projection == "mean":
            sub = sub.mean(axis=z_pos).astype(stack.dtype)
        elif z_projection == "min":
            sub = sub.min(axis=z_pos)

    # Result should be (T, H, W) — squeeze any remaining singleton dims
    while sub.ndim > 3:
        sub = sub.squeeze(axis=-3 if sub.shape[-3] == 1 else 0)

    return sub


# ── V1.28 multi-file TIFF reconstruction ─────────────────────────────

_TIFF_CHAIN_AXES = ("T", "M", "Z", "C")
_TIFF_CHAIN_ATTR = {
    "T": "n_timepoints",
    "M": "n_multipoints",
    "Z": "n_zslices",
    "C": "n_channels",
}


def read_tiff_meta_fast(filepath: str) -> Dict[str, Any]:
    """Cheap dim/dtype probe used by the Reconstruct dialog.

    Wraps :func:`read_imagej_tiff_metadata` and :func:`get_tiff_info`
    into a single dict with the same keys the ND2 path returns:
    ``{n_timepoints, n_zslices, n_channels, n_multipoints,
       height, width, dtype, channel_names, pixel_size_um}``.
    """
    ij = read_imagej_tiff_metadata(filepath)
    info = get_tiff_info(filepath)
    page_shape = info["page_shape"]
    h = int(page_shape[-2])
    w = int(page_shape[-1])
    if ij:
        return {
            "filepath": filepath,
            "n_timepoints": int(ij.get("n_timepoints", 1)),
            "n_zslices": int(ij.get("n_zslices", 1)),
            "n_channels": int(ij.get("n_channels", 1)),
            "n_multipoints": 1,
            "height": h,
            "width": w,
            "dtype": str(ij.get("dtype", info["dtype"])),
            "channel_names": list(ij.get("channel_names") or ["Ch0"]),
            "pixel_size_um": float(ij.get("pixel_size_um") or 1.0),
        }
    # Flat or non-ImageJ TIFF: treat as a (T, H, W) single-channel stack
    # where T == n_pages. Z and M are 1.
    return {
        "filepath": filepath,
        "n_timepoints": int(info["n_pages"]),
        "n_zslices": 1,
        "n_channels": 1,
        "n_multipoints": 1,
        "height": h,
        "width": w,
        "dtype": str(info["dtype"]),
        "channel_names": ["Ch0"],
        "pixel_size_um": 1.0,
    }


class _SingleFileTIFFView:
    """One member of :class:`LazyMultiFileTIFFVolume`.

    Holds the file's metadata, builds per-channel :class:`LazyTIFFChannel`
    proxies on demand, and exposes ``get_frame`` / ``to_lazy_channel`` /
    ``all_channels_as_lazy`` mirroring the ND2 volume surface. Single-
    file TIFF reads don't need an LRU since :class:`LazyTIFFChannel`
    already keeps its TiffFile handle open across reads.
    """

    def __init__(self, filepath: str):
        meta = read_tiff_meta_fast(filepath)
        self.filepath = filepath
        self.n_timepoints = int(meta["n_timepoints"])
        self.n_zslices = int(meta["n_zslices"])
        self.n_channels = int(meta["n_channels"])
        self.n_multipoints = 1  # TIFFs don't carry M.
        self.height = int(meta["height"])
        self.width = int(meta["width"])
        self.dtype = np.dtype(meta["dtype"])
        self.channel_names = list(meta["channel_names"])
        self.pixel_size_um = float(meta["pixel_size_um"])
        self.z_step_um = 1.0
        self._channels: Optional[Dict[str, LazyTIFFChannel]] = None
        # Direct page-reading handle for ImageJ multi-Z access — kept
        # open across get_frame calls; closed in close().
        self._tiff: Optional[tifffile.TiffFile] = None

    def _ensure_channels(self) -> Dict[str, LazyTIFFChannel]:
        if self._channels is not None:
            return self._channels
        ij = read_imagej_tiff_metadata(self.filepath)
        if ij and ij.get("n_channels", 1) > 1:
            # Multi-channel (Z >= 1) — load_imagej_tiff_channels now
            # honors Z by stacking + projecting across n_z planes with
            # stride = n_c.
            self._channels = load_imagej_tiff_channels(self.filepath, ij)
        elif ij and ij.get("n_zslices", 1) > 1:
            info = get_tiff_info(self.filepath)
            h, w = info["page_shape"][-2], info["page_shape"][-1]
            lazy = LazyTIFFChannel(
                self.filepath,
                t_start=0, t_end=int(ij["n_timepoints"]),
                height=h, width=w, dtype=np.dtype(info["dtype"]),
                n_pages_per_t=int(ij["n_zslices"]),
                page_within_t=0,
                n_z=int(ij["n_zslices"]),
                z_projection="max",
            )
            self._channels = {self.channel_names[0]: lazy}
        else:
            self._channels = {self.channel_names[0]: load_tiff_stack_lazy(self.filepath)}
        return self._channels

    def _ensure_tiff(self) -> tifffile.TiffFile:
        if self._tiff is None:
            self._tiff = tifffile.TiffFile(self.filepath)
        return self._tiff

    def close(self) -> None:
        if self._channels:
            for ch in self._channels.values():
                try:
                    ch.close()
                except Exception:
                    pass
        self._channels = None
        if self._tiff is not None:
            try:
                self._tiff.close()
            except Exception:
                pass
            self._tiff = None

    def get_frame(self, c: int = 0, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "none",
                  z_start: Optional[int] = None,
                  z_end: Optional[int] = None) -> np.ndarray:
        """Read one ``(H, W)`` frame from this TIFF.

        For multi-Z ImageJ hyperstacks (``n_zslices > 1``) the page is
        read directly using ``page = t*n_z*n_c + z*n_c + c`` so the
        request honors ``z`` (under ``z_mode='none'``) or projects
        across Z (``max``/``mean``/``min``). Flat TIFFs fall back to the
        cached :class:`LazyTIFFChannel`.
        """
        if self.n_zslices > 1:
            return self._read_imagej_frame(
                c=c, t=t, z=z, z_mode=z_mode,
                z_start=z_start, z_end=z_end,
            )

        channels = self._ensure_channels()
        name = self.channel_names[max(0, min(int(c), len(self.channel_names) - 1))]
        lazy = channels.get(name)
        if lazy is None:
            lazy = next(iter(channels.values()))
        t_idx = max(0, min(int(t), int(lazy.shape[0]) - 1)) if lazy.shape[0] else 0
        arr = np.asarray(lazy[t_idx])
        if arr.ndim != 2:
            arr = arr.squeeze()
        return arr

    def _read_imagej_frame(self, c: int, t: int, z: int, z_mode: str,
                           z_start: Optional[int],
                           z_end: Optional[int]) -> np.ndarray:
        """Read a (H, W) frame from an ImageJ TZCYX hyperstack by page index."""
        tif = self._ensure_tiff()
        n_c = max(1, int(self.n_channels))
        n_z = max(1, int(self.n_zslices))
        n_t = max(1, int(self.n_timepoints))
        c_i = max(0, min(int(c), n_c - 1))
        t_i = max(0, min(int(t), n_t - 1))

        def page(z_i: int) -> int:
            return t_i * n_z * n_c + z_i * n_c + c_i

        if z_mode == "none":
            z_i = max(0, min(int(z), n_z - 1))
            return tif.pages[page(z_i)].asarray()

        zs = 0 if z_start is None else max(0, int(z_start))
        ze = n_z if z_end is None else max(zs, min(int(z_end), n_z))
        planes = [tif.pages[page(zi)].asarray() for zi in range(zs, ze)]
        if not planes:
            return np.zeros((self.height, self.width), dtype=self.dtype)
        stack = np.stack(planes, axis=0)
        if z_mode == "max":
            return stack.max(axis=0)
        if z_mode == "mean":
            return stack.mean(axis=0).astype(stack.dtype)
        if z_mode == "min":
            return stack.min(axis=0)
        return stack[len(planes) // 2]

    def to_lazy_channel(self, c: int, m: int = 0,
                        z_mode: str = "max", z_index: int = 0,
                        z_start: int = 0, z_end: Optional[int] = None,
                        t_start: int = 0, t_end: Optional[int] = None,
                        t_stride: int = 1) -> LazyTIFFChannel:
        """Build a (T, H, W) lazy view for one channel honoring z_mode/z_index.

        ``z_mode='none'`` pins to ``z_index``; projection modes reduce
        across ``[z_start, z_end)`` (defaults to the full Z range).
        Single-Z files fall back to the cached per-channel proxy.
        """
        if self.n_zslices <= 1:
            channels = self._ensure_channels()
            names = list(channels.keys())
            ci = max(0, min(int(c), len(names) - 1))
            return channels[names[ci]]

        info = get_tiff_info(self.filepath)
        page_shape = info["page_shape"]
        h, w = page_shape[-2], page_shape[-1]
        dtype = np.dtype(info["dtype"])
        n_c = max(1, int(self.n_channels))
        n_z = max(1, int(self.n_zslices))
        ci = max(0, min(int(c), n_c - 1))
        if t_end is None:
            t_end = self.n_timepoints

        if z_mode == "none":
            z_local = max(0, min(int(z_index), n_z - 1))
            return LazyTIFFChannel(
                self.filepath,
                t_start=t_start, t_end=t_end,
                height=h, width=w, dtype=dtype,
                t_stride=t_stride,
                n_pages_per_t=n_z * n_c,
                page_within_t=z_local * n_c + ci,
                n_z=1,
            )

        zs = max(0, int(z_start))
        ze = n_z if z_end is None else max(zs, min(int(z_end), n_z))
        return LazyTIFFChannel(
            self.filepath,
            t_start=t_start, t_end=t_end,
            height=h, width=w, dtype=dtype,
            t_stride=t_stride,
            n_pages_per_t=n_z * n_c,
            page_within_t=zs * n_c + ci,
            n_z=max(1, ze - zs),
            z_stride=n_c,
            z_projection=z_mode,
        )

    def all_channels_as_lazy(self, m: int = 0,
                             z_mode: str = "max",
                             z_index: int = 0) -> Dict[str, LazyTIFFChannel]:
        return dict(self._ensure_channels())


class LazyMultiFileTIFFVolume:
    """Composite TIFF volume chained along ``chain_axis``.

    Parallel to :class:`nd2studios.backend.nd2_volume.LazyMultiFileND2Volume`,
    matched method-for-method so the rest of the app needs no per-format
    branching.
    """

    def __init__(self, filepaths: Iterable[str], chain_axis: str = "Z",
                 chain_mapping: Optional[List[Tuple[int, int]]] = None):
        if chain_axis not in _TIFF_CHAIN_AXES:
            raise ValueError(
                f"chain_axis must be one of {_TIFF_CHAIN_AXES}; got {chain_axis!r}"
            )

        paths = sorted(filepaths, key=lambda p: os.path.basename(p))
        if not paths:
            raise ValueError("LazyMultiFileTIFFVolume needs at least one file")

        self.chain_axis = chain_axis
        self.filepaths: List[str] = list(paths)
        self._files: List[_SingleFileTIFFView] = [
            _SingleFileTIFFView(p) for p in paths
        ]

        f0 = self._files[0]
        for f in self._files[1:]:
            mismatch = self._shape_mismatch(f0, f, chain_axis)
            if mismatch:
                raise ValueError(
                    "Multi-file TIFF import refused: shape mismatch between "
                    f"{os.path.basename(f0.filepath)} and "
                    f"{os.path.basename(f.filepath)} on non-chain "
                    f"axis(es) — {mismatch}"
                )

        attr = _TIFF_CHAIN_ATTR[chain_axis]
        self._chain_sizes: List[int] = [int(getattr(f, attr)) for f in self._files]

        # Same chain-mapping treatment as the ND2 composite: optional
        # user-provided per-slice ``(file_idx, local_idx)`` ordering,
        # defaulting to natural file-by-file concatenation.
        if chain_mapping is None:
            chain_mapping = [
                (fi, li) for fi, sz in enumerate(self._chain_sizes)
                for li in range(int(sz))
            ]
        self._chain_mapping: List[Tuple[int, int]] = [
            (int(fi), int(li)) for fi, li in chain_mapping
        ]
        self._validate_chain_mapping()

        self.filepath = f0.filepath
        self.dtype = f0.dtype
        self.height = f0.height
        self.width = f0.width
        self.pixel_size_um = float(f0.pixel_size_um)

        self.n_multipoints = f0.n_multipoints
        self.n_timepoints = f0.n_timepoints
        self.n_zslices = f0.n_zslices
        self.n_channels = f0.n_channels
        setattr(self, attr, len(self._chain_mapping))

        if chain_axis == "C":
            self.channel_names = [
                self._files[fi].channel_names[li]
                for fi, li in self._chain_mapping
            ]
        else:
            self.channel_names = list(f0.channel_names)

        self.z_step_um = float(f0.z_step_um)
        self.shape: Tuple[int, int, int, int, int] = (
            self.n_multipoints, self.n_timepoints, self.n_zslices,
            self.height, self.width,
        )

    def _validate_chain_mapping(self) -> None:
        for i, (fi, li) in enumerate(self._chain_mapping):
            if not (0 <= fi < len(self._files)):
                raise ValueError(
                    f"chain_mapping[{i}] = ({fi}, {li}) — "
                    f"file index out of range (n_files={len(self._files)})"
                )
            if not (0 <= li < self._chain_sizes[fi]):
                raise ValueError(
                    f"chain_mapping[{i}] = ({fi}, {li}) — local index "
                    f"out of range for file {fi} "
                    f"(size={self._chain_sizes[fi]})"
                )

    @staticmethod
    def _shape_mismatch(a: _SingleFileTIFFView, b: _SingleFileTIFFView,
                        chain_axis: str) -> str:
        all_keys = (
            ("n_multipoints", "M"), ("n_timepoints", "T"),
            ("n_zslices", "Z"), ("n_channels", "C"),
            ("height", "Y"), ("width", "X"),
        )
        skip = _TIFF_CHAIN_ATTR[chain_axis]
        diffs = [
            f"{label}={getattr(a, attr)} vs {getattr(b, attr)}"
            for attr, label in all_keys
            if attr != skip and getattr(a, attr) != getattr(b, attr)
        ]
        return ", ".join(diffs)

    def _lookup_chain(self, idx: int) -> Tuple[int, int]:
        if not self._chain_mapping:
            return 0, 0
        idx = int(idx)
        n = len(self._chain_mapping)
        idx = max(0, min(idx, n - 1))
        return self._chain_mapping[idx]

    def _route_coords(self, c: int, m: int, t: int, z: int
                      ) -> Tuple[int, int, int, int, int]:
        if self.chain_axis == "T":
            file_idx, t_local = self._lookup_chain(t)
            return file_idx, int(c), int(m), t_local, int(z)
        if self.chain_axis == "M":
            file_idx, m_local = self._lookup_chain(m)
            return file_idx, int(c), m_local, int(t), int(z)
        if self.chain_axis == "C":
            file_idx, c_local = self._lookup_chain(c)
            return file_idx, c_local, int(m), int(t), int(z)
        file_idx, z_local = self._lookup_chain(z)
        return file_idx, int(c), int(m), int(t), z_local

    def close(self) -> None:
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    def reopen(self) -> "LazyMultiFileTIFFVolume":
        return LazyMultiFileTIFFVolume(
            self.filepaths, self.chain_axis,
            chain_mapping=list(self._chain_mapping),
        )

    def get_frame(self, c: int = 0, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "none",
                  z_start: Optional[int] = None,
                  z_end: Optional[int] = None) -> np.ndarray:
        fi, cl, ml, tl, zl = self._route_coords(c, m, t, z)
        return self._files[fi].get_frame(
            c=cl, m=ml, t=tl, z=zl,
            z_mode=z_mode, z_start=z_start, z_end=z_end,
        )

    def to_lazy_channel(self, c: int, m: int = 0,
                        z_mode: str = "max", z_index: int = 0,
                        z_start: int = 0, z_end: Optional[int] = None,
                        t_start: int = 0, t_end: Optional[int] = None,
                        t_stride: int = 1):
        # For TIFFs only chain='T' or 'C' is common; M and Z are usually
        # single-valued. Always route via the per-axis lookup, but the
        # T-chain case demands a wrapper that walks files at read time.
        if self.chain_axis == "T":
            from nd2studios.backend.nd2_volume import MultiFileLazyChannel
            if t_end is None:
                t_end = self.n_timepoints
            return MultiFileLazyChannel(
                self, c=c, m=m, z_mode=z_mode, z_index=z_index,
                z_start=z_start, z_end=z_end if z_end is not None else self.n_zslices,
                t_start=t_start, t_end=t_end, t_stride=t_stride,
            )
        if self.chain_axis == "C":
            file_idx, c_local = self._lookup_chain(c)
            return self._files[file_idx].to_lazy_channel(
                c_local, m=m, z_mode=z_mode, z_index=z_index,
                z_start=z_start, z_end=z_end,
                t_start=t_start, t_end=t_end, t_stride=t_stride,
            )
        if self.chain_axis == "M":
            file_idx, _ = self._lookup_chain(m)
            return self._files[file_idx].to_lazy_channel(
                c, m=0, z_mode=z_mode, z_index=z_index,
                z_start=z_start, z_end=z_end,
                t_start=t_start, t_end=t_end, t_stride=t_stride,
            )
        # chain='Z' (and single-file fallthrough)
        file_idx, _ = self._lookup_chain(int(z_index))
        return self._files[file_idx].to_lazy_channel(
            c, m=m, z_mode=z_mode, z_index=z_index,
            z_start=z_start, z_end=z_end,
            t_start=t_start, t_end=t_end, t_stride=t_stride,
        )

    def all_channels_as_lazy(self, m: int = 0,
                             z_mode: str = "max",
                             z_index: int = 0):
        from collections import OrderedDict
        out = OrderedDict()
        for c, name in enumerate(self.channel_names):
            out[name] = self.to_lazy_channel(
                c, m=m, z_mode=z_mode, z_index=z_index,
            )
        return out
