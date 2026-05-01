"""
TIFF stack loading with dimension assignment.
"""
from __future__ import annotations

from typing import Optional, Tuple

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
    """

    def __init__(self, filepath: str, t_start: int, t_end: int,
                 height: int, width: int, dtype, t_stride: int = 1):
        self._filepath = filepath
        self._t0 = t_start
        self._t1 = t_end
        self._t_stride = max(1, int(t_stride))
        n = max(0, (t_end - t_start + self._t_stride - 1) // self._t_stride)
        self.shape = (n, height, width)
        self.dtype = np.dtype(dtype)
        self.ndim = 3

    def __len__(self):
        return self.shape[0]

    def _read_frame(self, t_local: int) -> np.ndarray:
        return tifffile.imread(
            self._filepath, key=self._t0 + int(t_local) * self._t_stride,
        )

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
