"""
``LazyND2Volume`` — axis-aware lazy reader exposing (M, T, Z, H, W) for
the multi-axis viewer.

Why this exists: ``LazyND2Channel`` (V1.0) collapses Z via projection
and pins M to position 0 at construction time. The V1.1 viewer needs to
let the user scroll M and Z, which means asking the file for an
arbitrary (m, t, z, c) frame on demand. This class wraps the same
``nd2.ND2File`` machinery and exposes one fast frame fetch.

Once the user has picked an M position and a Z mode (project / fixed
slice), :meth:`to_lazy_channel` produces a V1.0 :class:`LazyND2Channel`
so the recipe / export pipeline can keep operating on `(T, H, W)` —
plugins do not need to change.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from nd2studios.backend.nd2_loader import LazyND2Channel


Z_PROJECTION_MODES = ("none", "max", "mean", "min")


class LazyND2Volume:
    """Lazy view over an ND2 file with M/T/Z/C axes preserved.

    Attributes
    ----------
    filepath : str
    shape : (M, T, Z, H, W)
    n_channels : int
    channel_names : List[str]
    dtype : np.dtype
    pixel_size_um : float
    """

    def __init__(self, filepath: str):
        import nd2

        self.filepath = filepath
        self._file: Optional["nd2.ND2File"] = None
        self._dask = None
        self._dim_order: List[str] = []

        # Probe the file once to learn the shape, then close. The actual
        # working handle is opened lazily on the first frame fetch.
        with nd2.ND2File(filepath) as f:
            sizes = dict(f.sizes)
            self.dtype = np.dtype(f.dtype)
            self._sizes = sizes
            self._dim_order = list(sizes.keys())
            self.n_timepoints = sizes.get("T", 1)
            self.n_zslices = sizes.get("Z", 1)
            self.n_channels = sizes.get("C", 1)
            self.n_multipoints = sizes.get("P", sizes.get("M", 1))
            self.height = sizes.get("Y", 0)
            self.width = sizes.get("X", 0)
            try:
                vox = f.voxel_size()
                self.pixel_size_um = float(getattr(vox, "x", 1.0))
                self.z_step_um = float(getattr(vox, "z", 1.0))
            except Exception:
                self.pixel_size_um = 1.0
                self.z_step_um = 1.0
            try:
                self.channel_names = [
                    str(getattr(getattr(c, "channel", c), "name", f"Ch{i}"))
                    for i, c in enumerate(f.metadata.channels)
                ]
            except Exception:
                self.channel_names = [f"Ch{i}" for i in range(self.n_channels)]

        self.shape: Tuple[int, int, int, int, int] = (
            self.n_multipoints, self.n_timepoints, self.n_zslices,
            self.height, self.width,
        )

    # ── handle management ──
    def _ensure_open(self):
        if self._file is None:
            import nd2
            self._file = nd2.ND2File(self.filepath)
            self._dask = self._file.to_dask()
        return self._file

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._dask = None

    def __del__(self):
        self.close()

    # ── frame access ──
    def _index_for(self, c: int, m: int, t: int, z) -> Tuple:
        """Build a dask indexing tuple that selects (c, m, t, z) and
        leaves Y, X full-frame. ``z`` may be an int or a slice."""
        idx = []
        for d in self._dim_order:
            if d == "T":
                idx.append(int(t))
            elif d == "C":
                idx.append(int(c))
            elif d == "Z":
                idx.append(z if isinstance(z, slice) else int(z))
            elif d in ("P", "M"):
                idx.append(int(m))
            elif d in ("Y", "X"):
                idx.append(slice(None))
            else:
                idx.append(0)
        return tuple(idx)

    def get_frame(self, c: int, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "none",
                  z_start: Optional[int] = None,
                  z_end: Optional[int] = None) -> np.ndarray:
        """Return a single (H, W) frame.

        Parameters
        ----------
        c, m, t : int — channel / multipoint / timepoint indices.
        z : int — single Z slice (used when ``z_mode == 'none'``).
        z_mode : 'none' | 'max' | 'mean' | 'min' — Z projection mode.
        z_start, z_end : int — optional Z range for projection.

        For projection modes, ``z_start`` defaults to 0 and ``z_end``
        to ``self.n_zslices``.
        """
        self._ensure_open()
        if z_mode not in Z_PROJECTION_MODES:
            raise ValueError(f"unknown z_mode {z_mode!r}; "
                             f"expected one of {Z_PROJECTION_MODES}")

        if z_mode == "none" or self.n_zslices <= 1:
            arr = np.asarray(self._dask[self._index_for(c, m, t, z)])
            return arr if arr.ndim == 2 else arr.squeeze()

        if z_start is None:
            z_start = 0
        if z_end is None:
            z_end = self.n_zslices
        z_stack = np.asarray(self._dask[self._index_for(c, m, t, slice(z_start, z_end))])
        if z_stack.ndim == 2:
            return z_stack
        if z_stack.ndim != 3:
            z_stack = z_stack.squeeze()
        if z_mode == "max":
            return z_stack.max(axis=0)
        if z_mode == "min":
            return z_stack.min(axis=0)
        # mean
        return z_stack.mean(axis=0).astype(self.dtype)

    def to_lazy_channel(self, c: int, m: int = 0,
                        z_mode: str = "max",
                        z_index: int = 0,
                        z_start: int = 0,
                        z_end: Optional[int] = None,
                        t_start: int = 0,
                        t_end: Optional[int] = None,
                        t_stride: int = 1) -> LazyND2Channel:
        """Build a (T, H, W) lazy channel for the recipe/export pipeline.

        ``z_mode`` controls how Z is collapsed into the (T, H, W) shape:

        - ``'max' | 'mean' | 'min'`` — reduce over [z_start, z_end).
        - ``'none'`` — pin to a single Z slice (``z_index``).

        Returns a :class:`LazyND2Channel` whose API matches V1.0; the
        underlying loader uses the same nd2 file path and re-reads each
        frame (already efficient — we don't double-cache here).
        """
        if t_end is None:
            t_end = self.n_timepoints
        if z_end is None:
            z_end = self.n_zslices

        if z_mode == "none":
            # Pin to a single Z slice. We hand LazyND2Channel a degenerate
            # range (z_index, z_index+1) and tell it to "project" over
            # that single slice — projection over one slice is a no-op,
            # so we get exactly that slice.
            z_start_eff = int(z_index)
            z_end_eff = int(z_index) + 1
            zproj = "max"
        else:
            z_start_eff = z_start
            z_end_eff = z_end
            zproj = z_mode

        return LazyND2Channel(
            self.filepath, channel_index=c,
            t_start=t_start, t_end=t_end, t_stride=t_stride,
            z_start=z_start_eff, z_end=z_end_eff, z_projection=zproj,
            height=self.height, width=self.width, dtype=self.dtype,
        )

    def all_channels_as_lazy(self, m: int = 0,
                             z_mode: str = "max",
                             z_index: int = 0) -> "OrderedDict[str, LazyND2Channel]":
        """Convenience: return an OrderedDict of channel_name → lazy proxy.

        Used by ``LoadWorker`` to populate ``ND2StudiosRecord._raw_channels``
        once the user has picked M and Z mode.
        """
        from collections import OrderedDict
        out = OrderedDict()
        for c, name in enumerate(self.channel_names):
            out[name] = self.to_lazy_channel(c, m=m, z_mode=z_mode, z_index=z_index)
        return out
