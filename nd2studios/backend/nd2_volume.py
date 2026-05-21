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

import os
from typing import Iterable, List, Optional, Tuple

import numpy as np

from nd2studios.backend.nd2_loader import LazyND2Channel


Z_PROJECTION_MODES = ("none", "max", "mean", "min")

# Axes the V1.28 Reconstruct dialog can chain files along.
CHAIN_AXES = ("T", "M", "Z", "C")

_CHAIN_ATTR = {
    "T": "n_timepoints",
    "M": "n_multipoints",
    "Z": "n_zslices",
    "C": "n_channels",
}


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
            m_index=m,
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

    def reopen(self) -> "LazyND2Volume":
        """Return a fresh volume over the same file with its own handle.

        Used by the prefetch worker so background reads never share an
        ``nd2.ND2File`` handle with the main-thread reader (the nd2
        library's per-file state is not thread-safe).
        """
        return LazyND2Volume(self.filepath)


class MultiFileLazyChannel:
    """``(T, H, W)`` lazy proxy backed by a composite multi-file volume.

    Used whenever the per-T iteration has to route through the composite
    — i.e. when the volume's chain axis is ``T`` (each file owns a slab
    of timepoints) or when the chain axis is ``Z`` with a projection
    ``z_mode`` (each T frame is reduced across files). For
    chain-``M``/``C`` and for chain-``Z`` with ``z_mode='none'``,
    ``LazyMultiFileND2Volume.to_lazy_channel`` short-circuits to a
    plain :class:`LazyND2Channel` from the single owning file — those
    cases never construct this class.

    Mirrors the :class:`LazyND2Channel` numpy-protocol surface so the
    recipe and export pipelines stay format-agnostic.
    """

    def __init__(self, volume: "LazyMultiFileND2Volume", c: int, m: int,
                 z_mode: str, z_index: int,
                 z_start: int, z_end: int,
                 t_start: int, t_end: int, t_stride: int):
        self._volume = volume
        self._c = int(c)
        self._m = int(m)
        self._z_mode = z_mode
        self._z_index = int(z_index)
        self._z_start = int(z_start)
        self._z_end = int(z_end)
        self._t0 = int(t_start)
        self._t1 = int(t_end)
        self._t_stride = max(1, int(t_stride))
        n = max(0, (t_end - t_start + self._t_stride - 1) // self._t_stride)
        self.shape: Tuple[int, int, int] = (n, volume.height, volume.width)
        self.dtype = np.dtype(volume.dtype)
        self.ndim = 3

    def __len__(self) -> int:
        return self.shape[0]

    def _read_frame(self, t_local: int) -> np.ndarray:
        t_abs = self._t0 + int(t_local) * self._t_stride
        return self._volume.get_frame(
            c=self._c, m=self._m, t=t_abs, z=self._z_index,
            z_mode=self._z_mode,
            z_start=self._z_start, z_end=self._z_end,
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

        frames = [self._read_frame(int(i)) for i in indices]
        if not frames:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)
        stacked = np.stack(frames, axis=0)
        return stacked[(slice(None),) + spatial] if spatial else stacked

    def __array__(self, dtype=None) -> np.ndarray:
        full = np.stack(
            [self._read_frame(t) for t in range(self.shape[0])], axis=0,
        )
        return full.astype(dtype) if dtype is not None else full

    def copy(self) -> "MultiFileLazyChannel":
        return self

    def materialize(self) -> np.ndarray:
        return np.stack(
            [self._read_frame(t) for t in range(self.shape[0])], axis=0,
        )

    def crop(self, y0: int, y1: int, x0: int, x1: int) -> "MultiFileLazyChannel":
        view = MultiFileLazyChannel.__new__(MultiFileLazyChannel)
        view._volume = self._volume
        view._c = self._c
        view._m = self._m
        view._z_mode = self._z_mode
        view._z_index = self._z_index
        view._z_start = self._z_start
        view._z_end = self._z_end
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

    def close(self) -> None:
        # File handles live on the parent volume; nothing per-channel to free.
        pass


# V1.27 alias retained for any external consumer pinning the old name.
MultiFileLazyZChannel = MultiFileLazyChannel


class LazyMultiFileND2Volume:
    """Composite ND2 volume backed by several files chained along one axis.

    The user picks the *chain axis* (``T`` / ``M`` / ``Z`` / ``C``) in
    the Reconstruct dialog (V1.28). All axes *except* the chain axis
    must match across files; the chain axis is the sum of each file's
    native size on that axis.

    The public surface mirrors :class:`LazyND2Volume` so the viewer,
    prefetch worker, recipe pipeline, and exporters need no special-
    casing — only ``get_frame``, ``to_lazy_channel``, and
    ``all_channels_as_lazy`` are reached at runtime, plus a handful of
    scalar / list attributes (``shape``, ``n_*``, ``height``, ``width``,
    ``dtype``, ``pixel_size_um``, ``z_step_um``, ``channel_names``,
    ``filepath``, ``filepaths``, ``chain_axis``).
    """

    def __init__(self, filepaths: Iterable[str],
                 chain_axis: str = "Z",
                 chain_mapping: Optional[List[Tuple[int, int]]] = None):
        if chain_axis not in CHAIN_AXES:
            raise ValueError(
                f"chain_axis must be one of {CHAIN_AXES}; got {chain_axis!r}"
            )

        paths = sorted(filepaths, key=lambda p: os.path.basename(p))
        if not paths:
            raise ValueError("LazyMultiFileND2Volume needs at least one file")

        self.chain_axis = chain_axis
        self.filepaths: List[str] = list(paths)
        self._files: List[LazyND2Volume] = [LazyND2Volume(p) for p in paths]

        f0 = self._files[0]
        for f in self._files[1:]:
            mismatch = self._shape_mismatch(f0, f, chain_axis)
            if mismatch:
                raise ValueError(
                    "Multi-file import refused: shape mismatch between "
                    f"{os.path.basename(f0.filepath)} and "
                    f"{os.path.basename(f.filepath)} on non-chain "
                    f"axis(es) — {mismatch}"
                )

        # Per-file native sizes on the chain axis.
        attr = _CHAIN_ATTR[chain_axis]
        self._chain_sizes: List[int] = [int(getattr(f, attr)) for f in self._files]

        # Build the chain-axis mapping: a list whose length is the total
        # chain-axis size of the composite, where entry ``i`` is a
        # ``(file_idx, local_idx_in_file)`` pair.
        #
        # The default (natural) mapping concatenates each file's slices
        # in file order: ``[(0,0), (0,1), …, (0,n0-1), (1,0), …]``.
        # The Reconstruct dialog can pass a custom mapping (V1.31) to
        # interleave or reorder slices across files.
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

        # Default the non-chain axes to file 0's sizes; override the
        # chain axis with the mapping length.
        self.n_multipoints = f0.n_multipoints
        self.n_timepoints = f0.n_timepoints
        self.n_zslices = f0.n_zslices
        self.n_channels = f0.n_channels
        setattr(self, attr, len(self._chain_mapping))

        # Channel names: concatenate when chaining on C, respecting the
        # mapping; else inherit from file 0.
        if chain_axis == "C":
            self.channel_names = [
                self._files[fi].channel_names[li]
                for fi, li in self._chain_mapping
            ]
        else:
            self.channel_names = list(f0.channel_names)

        # Z step: only meaningful (across files) when chaining on Z.
        if chain_axis == "Z" and self.n_zslices > 1:
            self.z_step_um = self._infer_z_step(default=float(f0.z_step_um))
        else:
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
    def _shape_mismatch(a: LazyND2Volume, b: LazyND2Volume,
                        chain_axis: str) -> str:
        # Validate every axis EXCEPT the chain axis.
        all_keys = (
            ("n_multipoints", "M"), ("n_timepoints", "T"),
            ("n_zslices", "Z"), ("n_channels", "C"),
            ("height", "Y"), ("width", "X"),
        )
        skip = _CHAIN_ATTR[chain_axis]
        diffs = [
            f"{label}={getattr(a, attr)} vs {getattr(b, attr)}"
            for attr, label in all_keys
            if attr != skip and getattr(a, attr) != getattr(b, attr)
        ]
        return ", ".join(diffs)

    # ── chain-axis index routing ──
    def _lookup_chain(self, idx: int) -> Tuple[int, int]:
        """Return ``(file_idx, local_idx)`` for a chain-axis coordinate.

        Looks up the user-provided (or default natural) chain mapping.
        """
        if not self._chain_mapping:
            return 0, 0
        idx = int(idx)
        n = len(self._chain_mapping)
        idx = max(0, min(idx, n - 1))
        return self._chain_mapping[idx]

    def _route_coords(self, c: int, m: int, t: int, z: int
                      ) -> Tuple[int, int, int, int, int]:
        """Map (c, m, t, z) to (file_idx, c_local, m_local, t_local, z_local).

        Non-chain coordinates are passed through unchanged.
        """
        if self.chain_axis == "T":
            file_idx, t_local = self._lookup_chain(t)
            return file_idx, int(c), int(m), t_local, int(z)
        if self.chain_axis == "M":
            file_idx, m_local = self._lookup_chain(m)
            return file_idx, int(c), m_local, int(t), int(z)
        if self.chain_axis == "C":
            file_idx, c_local = self._lookup_chain(c)
            return file_idx, c_local, int(m), int(t), int(z)
        # chain_axis == 'Z'
        file_idx, z_local = self._lookup_chain(z)
        return file_idx, int(c), int(m), int(t), z_local

    def _infer_z_step(self, default: float) -> float:
        """Estimate Z step from per-file event Z coords (chain='Z' only).

        Uses the per-file Z coord (event 0) ordered by the chain mapping
        — i.e. by the user's chosen Z stack order, not raw file order.
        """
        per_file = self._raw_file_z_coords_um()
        if not per_file:
            return default
        zs = [per_file[fi] for fi, _li in self._chain_mapping
              if 0 <= fi < len(per_file)]
        if len(zs) < 2:
            return default
        diffs = np.diff(np.asarray(zs, dtype=np.float64))
        # Absolute value: stages may be acquired top-down or bottom-up.
        step = float(np.mean(np.abs(diffs)))
        return step if step > 0 else default

    def _raw_file_z_coords_um(self) -> List[float]:
        """Per-file event[0] Z coord in raw file order. Empty on failure."""
        out: List[float] = []
        for path in self.filepaths:
            try:
                import nd2
                with nd2.ND2File(path) as f:
                    evts = list(f.events())
                if not evts:
                    return []
                z = evts[0].get("Z Coord [µm]")
                if z is None:
                    return []
                out.append(float(z))
            except Exception:
                return []
        return out

    def file_z_coords_um(self) -> List[float]:
        """Per-slice Z stage coord in chain-axis (output) order.

        Empty list on read failure. When the chain axis is ``Z`` and
        each file contributes one slice, this is per-file Z; with a
        custom mapping the order follows the user's drag-reordering.
        """
        per_file = self._raw_file_z_coords_um()
        if not per_file:
            return []
        return [per_file[fi] for fi, _li in self._chain_mapping
                if 0 <= fi < len(per_file)]

    def close(self) -> None:
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    def reopen(self) -> "LazyMultiFileND2Volume":
        return LazyMultiFileND2Volume(
            self.filepaths, self.chain_axis,
            chain_mapping=list(self._chain_mapping),
        )

    # ── frame access ──
    def get_frame(self, c: int, m: int = 0, t: int = 0, z: int = 0,
                  z_mode: str = "none",
                  z_start: Optional[int] = None,
                  z_end: Optional[int] = None) -> np.ndarray:
        """Read one ``(H, W)`` frame.

        Routing rules per chain axis:
          - ``T``/``M``/``C`` — map the chain coord to its owning file
            and forward `(c_local, m_local, t_local, z, z_mode)`.
          - ``Z`` with ``z_mode='none'`` — pick the file owning ``z`` and
            read its local Z slice.
          - ``Z`` with projection ``z_mode`` — read each global Z plane
            in ``[z_start, z_end)`` through the per-file router and
            reduce across them.
        """
        if z_mode not in Z_PROJECTION_MODES:
            raise ValueError(f"unknown z_mode {z_mode!r}; "
                             f"expected one of {Z_PROJECTION_MODES}")

        if self.chain_axis == "Z" and z_mode != "none" and self.n_zslices > 1:
            if z_start is None:
                z_start = 0
            if z_end is None:
                z_end = self.n_zslices
            z_start = max(0, int(z_start))
            z_end = max(z_start, min(int(z_end), self.n_zslices))
            planes: List[np.ndarray] = []
            for zi in range(z_start, z_end):
                fi, zl = self._lookup_chain(zi)
                planes.append(self._files[fi].get_frame(
                    c=int(c), m=int(m), t=int(t), z=zl, z_mode="none",
                ))
            if not planes:
                return np.zeros((self.height, self.width), dtype=self.dtype)
            stack = np.stack(planes, axis=0)
            if z_mode == "max":
                return stack.max(axis=0)
            if z_mode == "min":
                return stack.min(axis=0)
            return stack.mean(axis=0).astype(self.dtype)

        fi, cl, ml, tl, zl = self._route_coords(c, m, t, z)
        return self._files[fi].get_frame(
            c=cl, m=ml, t=tl, z=zl,
            z_mode=z_mode, z_start=z_start, z_end=z_end,
        )

    # ── lazy (T, H, W) builders ──
    def to_lazy_channel(self, c: int, m: int = 0,
                        z_mode: str = "max",
                        z_index: int = 0,
                        z_start: int = 0,
                        z_end: Optional[int] = None,
                        t_start: int = 0,
                        t_end: Optional[int] = None,
                        t_stride: int = 1):
        """Return a ``(T, H, W)`` lazy proxy for the recipe pipeline.

        Whenever the T iteration can stay inside a single file, the
        underlying file's :class:`LazyND2Channel` is returned directly
        (free reuse of V1.0 code paths). Otherwise a
        :class:`MultiFileLazyChannel` walks the file boundary on each
        frame read.
        """
        if t_end is None:
            t_end = self.n_timepoints
        if z_end is None:
            z_end = self.n_zslices

        if self.chain_axis == "T":
            return MultiFileLazyChannel(
                self, c=c, m=m,
                z_mode=z_mode, z_index=z_index,
                z_start=z_start, z_end=z_end,
                t_start=t_start, t_end=t_end, t_stride=t_stride,
            )

        if self.chain_axis == "M":
            file_idx, m_local = self._lookup_chain(m)
            return self._files[file_idx].to_lazy_channel(
                c, m=m_local, z_mode=z_mode, z_index=z_index,
                z_start=z_start, z_end=z_end,
                t_start=t_start, t_end=t_end, t_stride=t_stride,
            )

        if self.chain_axis == "C":
            file_idx, c_local = self._lookup_chain(c)
            return self._files[file_idx].to_lazy_channel(
                c_local, m=m, z_mode=z_mode, z_index=z_index,
                z_start=z_start, z_end=z_end,
                t_start=t_start, t_end=t_end, t_stride=t_stride,
            )

        # chain_axis == 'Z'
        if z_mode == "none":
            file_idx, z_local = self._lookup_chain(int(z_index))
            return self._files[file_idx].to_lazy_channel(
                c, m=m, z_mode="none", z_index=z_local,
                t_start=t_start, t_end=t_end, t_stride=t_stride,
            )
        return MultiFileLazyChannel(
            self, c=c, m=m,
            z_mode=z_mode, z_index=z_index,
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
