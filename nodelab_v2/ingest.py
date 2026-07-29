"""ND2 → nodegraph v2 ingest (C4) — the app-layer, nd2-coupled reader.

``nodegraph`` stays nd2-free; this module is the seam that turns a real microscope
``.nd2`` into the two things the v2 engine consumes:

* a lazy voxel source — a :class:`~nodegraph.provider.B2ndProvider` (in-memory, or an
  on-disk planar-block ``.b2nd`` store written once and re-opened lazily), and
* a source :class:`~nodegraph.metadata.MetaEnvelope` (canonical axes + the calibration
  dict that drives every metadata-intelligent param).

The ND2 pixels are read via ``nd2.ND2File.to_dask()`` (the frame-wise ``read_frame``
path **segfaults** on the sample, per V2.01) and transposed into the canonical
``(M,T,Z,C,Y,X)`` order, inserting size-1 axes for absent dimensions. Calibration reuses
the proven ``read_nd2_metadata_extended`` (optics parsing is fiddly — don't re-derive it;
vendored from the retired v1 backend into :mod:`nodelab_v2.nd2_meta`), filtered to the v2
``CALIBRATION_KEYS`` vocabulary. Qt-free.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nodegraph.dataset import AxisSizes, CALIBRATION_KEYS
from nodegraph.metadata import MetaEnvelope
from nodegraph.provider import B2ndProvider

#: ND2 axis letters → the canonical nodegraph axes (``P`` = position/multipoint → M).
_ND_TO_CANON = {"P": "m", "T": "t", "Z": "z", "C": "c", "Y": "y", "X": "x"}
_CANON: Tuple[str, ...] = ("m", "t", "z", "c", "y", "x")


def _to_6d(arr: Any, dims: list) -> Any:
    """Reshape an array whose axes are the ND2 ``dims`` (a subset of P,T,Z,C,Y,X, in
    file order) into canonical ``(M,T,Z,C,Y,X)`` — size-1 axes for absent dimensions.
    Works for both numpy and dask (``np.transpose`` dispatches)."""
    present = [_ND_TO_CANON[d] for d in dims]
    missing = [ax for ax in _CANON if ax not in present]
    for _ in missing:
        arr = arr[..., None]                       # append the absent axes as size-1
    order = present + missing
    return np.transpose(arr, [order.index(ax) for ax in _CANON])


def read_calibration(path: str) -> Dict[str, Any]:
    """The v2 calibration dict for ``path`` (ND2 **or** TIFF) — filtered to
    :data:`~nodegraph.dataset.CALIBRATION_KEYS` (dropping absent / ``None`` scalars;
    ``channel_emission_nm`` stays a per-channel list even when some channels have no
    emission — e.g. a transmitted-light channel is ``None``). ND2 reuses the proven v1
    ``read_nd2_metadata_extended``; TIFF is best-effort (:func:`_tiff_calibration`)."""
    if _is_tiff(path):
        import tifffile
        with tifffile.TiffFile(path) as tf:
            return _tiff_calibration(tf)
    from nodelab_v2.nd2_meta import read_nd2_metadata_extended
    md = read_nd2_metadata_extended(path)
    calib = {k: md[k] for k in CALIBRATION_KEYS
             if md.get(k) is not None}
    # the significant sensor depth is not part of `read_nd2_metadata_extended`'s dict, but
    # it IS calibration now (nodes read it — a 12-bit ND2 must not be treated as 16-bit)
    bits = _nd2_bit_depth(path)
    if bits:
        calib["bit_depth"] = int(bits)
    return calib


#: extra per-channel display keys (names/optics/native color) the Viewer wants but that
#: are NOT part of the engine calibration schema (:data:`CALIBRATION_KEYS`).
CHANNEL_DISPLAY_KEYS = ("channel_names", "channel_emission_nm",
                        "channel_excitation_nm", "channel_colors")


def _nd2_bit_depth(path: str) -> Any:
    """The ND2's *significant* bit depth (e.g. 12) — the real sensor range, which the
    pixel values alone can't reveal (a dim 12-bit frame may max out below 1024). Read
    straight from ``nd2``'s attributes; ``None`` if unavailable."""
    try:
        import nd2
        with nd2.ND2File(path) as f:
            a = f.attributes
            for name in ("bitsPerComponentSignificant", "bitsPerComponentInMemory"):
                v = getattr(a, name, None)
                if v:
                    return int(v)
    except Exception:                        # noqa: BLE001 — bit depth is best-effort
        return None
    return None


def _tiff_bit_depth(path: str) -> Any:
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            bps = tf.pages[0].bitspersample
            return int(bps[0] if isinstance(bps, (tuple, list)) else bps)
    except Exception:                        # noqa: BLE001
        return None


def read_channel_display(path: str) -> Dict[str, Any]:
    """The per-channel *display* metadata for ``path`` — names, emission/excitation
    wavelengths, the native color, and the significant **bit depth** — used to label and
    tint the channel toggles and to size the LUT range (so 12-bit data windows to 4095,
    not just to its brightest pixel). A superset of the engine calibration; kept separate
    so the engine envelope stays the locked calibration schema. TIFF carries no optics, so
    it degrades to ``Ch0…`` names (emission absent → a neutral grey tint downstream)."""
    if _is_tiff(path):
        ax = _tiff_axes(path)
        out: Dict[str, Any] = {"channel_names": [f"Ch{i}" for i in range(ax.c)]}
        bits = _tiff_bit_depth(path)
        if bits:
            out["bit_depth"] = bits
        return out
    from nodelab_v2.nd2_meta import read_nd2_metadata_extended
    md = read_nd2_metadata_extended(path)
    out = {k: md[k] for k in CHANNEL_DISPLAY_KEYS if md.get(k) is not None}
    bits = _nd2_bit_depth(path)
    if bits:
        out["bit_depth"] = bits
    return out


def read_nd2(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read an ``.nd2`` into a canonical ``(M,T,Z,C,Y,X)`` numpy array + its calibration
    dict. Realizes the whole volume (ingest is a one-time cost); slice ``to_dask`` before
    calling if only a crop is needed."""
    import nd2
    calib = read_calibration(path)
    with nd2.ND2File(path) as f:
        vol = np.ascontiguousarray(np.asarray(_to_6d(f.to_dask(), list(f.sizes.keys()))))
    return vol, calib


#: file extensions this ingest layer reads (case-insensitive).
_ND2_EXT = (".nd2",)
_TIFF_EXT = (".tif", ".tiff")

#: tifffile axis letters → the ND2 letter space understood by :func:`_to_6d`
#: (``S`` sample-planes read as channels; ``I``/``Q`` sequence axes read as time).
_TIFF_TO_ND = {"T": "T", "Z": "Z", "C": "C", "Y": "Y", "X": "X",
               "S": "C", "I": "T", "Q": "T"}


def _tiff_dims(axes: str) -> list:
    """Translate a tifffile ``series.axes`` string to the ``_to_6d`` letter list,
    rejecting an axis we can't place (so a surprise layout fails loudly, not silently
    mis-shaped)."""
    dims = []
    for a in axes:
        nd = _TIFF_TO_ND.get(a.upper())
        if nd is None:
            raise ValueError(f"unsupported TIFF axis {a!r} in layout {axes!r}")
        if nd in dims:
            raise ValueError(f"TIFF layout {axes!r} maps two axes onto {nd!r}")
        dims.append(nd)
    if "Y" not in dims or "X" not in dims:
        raise ValueError(f"TIFF layout {axes!r} has no Y/X image plane")
    return dims


def _tiff_calibration(tf: Any) -> Dict[str, Any]:
    """Best-effort calibration for a TIFF: ImageJ ``spacing`` → z step, the XY
    resolution tag → pixel size (µm). Absent tags are simply omitted — a plain TIFF
    carries no optics, and the metadata-intelligent params fall back to their guards."""
    calib: Dict[str, Any] = {}
    ij = getattr(tf, "imagej_metadata", None) or {}
    unit = str(ij.get("unit", "")).lower()
    if ij.get("spacing") and unit in ("um", "micron", "microns", "µm", ""):
        try:
            calib["z_step_um"] = float(ij["spacing"])
        except (TypeError, ValueError):
            pass
    try:
        page = tf.pages[0]
        xres = page.tags.get("XResolution")
        if xres is not None and xres.value and xres.value[0]:
            num, den = xres.value
            per_unit = (num / den) if den else 0.0
            if per_unit:
                px = 1.0 / per_unit                 # distance per pixel, in the tag unit
                ru = page.tags.get("ResolutionUnit")
                if ru is not None and int(ru.value) == 3:   # 3 = centimeter
                    px *= 1.0e4                              # cm → µm
                elif ru is not None and int(ru.value) == 2:  # 2 = inch
                    px *= 25400.0                            # in → µm
                calib["pixel_size_um"] = float(px)
    except Exception:  # noqa: BLE001 — a missing/odd tag must never break ingest
        pass
    try:                                        # bits-per-sample = the CONTAINER depth
        bps = tf.pages[0].bitspersample
        calib["bit_depth"] = int(bps[0] if isinstance(bps, (tuple, list)) else bps)
    except Exception:  # noqa: BLE001
        pass
    return {k: v for k, v in calib.items() if k in CALIBRATION_KEYS}


def read_tiff(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a ``.tif``/``.tiff`` into a canonical ``(M,T,Z,C,Y,X)`` numpy array + a
    best-effort calibration dict (see :func:`_tiff_calibration`). Uses the first series."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        arr = series.asarray()
        vol = np.ascontiguousarray(np.asarray(_to_6d(arr, _tiff_dims(series.axes))))
        calib = _tiff_calibration(tf)
    return vol, calib


def _is_tiff(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _TIFF_EXT


def _tiff_axes(path: str) -> AxisSizes:
    """Canonical :class:`AxisSizes` for a TIFF from its series shape/axes — **no pixel
    read** (uses ``series.shape``, not ``asarray``)."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        dims = _tiff_dims(series.axes)
        size = {nd: n for nd, n in zip(dims, series.shape)}
    canon = {"m": size.get("P", 1), "t": size.get("T", 1), "z": size.get("Z", 1),
             "c": size.get("C", 1), "y": size.get("Y", 1), "x": size.get("X", 1)}
    return AxisSizes(**{k: int(v) for k, v in canon.items()})


def read_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read an ``.nd2`` **or** ``.tif``/``.tiff`` into a canonical ``(M,T,Z,C,Y,X)``
    numpy array + its calibration dict — the format-dispatching reader."""
    return read_tiff(path) if _is_tiff(path) else read_nd2(path)


def read_meta_only(path: str) -> Tuple[AxisSizes, Dict[str, Any], Dict[str, Any]]:
    """``(axes, calibration, channel_display)`` for ``path`` **without realizing pixels**
    — the cheap read the File-menu loader uses to seed a node's envelope + per-channel
    output sockets the instant a file is picked (the heavy ingest happens lazily on the
    first pull). Works for ND2 and TIFF."""
    calib = read_calibration(path)
    disp = read_channel_display(path)
    if _is_tiff(path):
        axes = _tiff_axes(path)
    else:
        import nd2
        with nd2.ND2File(path) as f:
            s = f.sizes
        axes = AxisSizes(m=int(s.get("P", 1)), t=int(s.get("T", 1)),
                         z=int(s.get("Z", 1)), c=int(s.get("C", 1)),
                         y=int(s.get("Y", 1)), x=int(s.get("X", 1)))
    return axes, calib, disp


def ingest_image(path: str, store_path: Optional[str] = None, *, levels: int = 1
                 ) -> Tuple[B2ndProvider, MetaEnvelope]:
    """Ingest an ``.nd2`` **or** ``.tif``/``.tiff`` → ``(provider, source_envelope)``
    ready to seed the engine.

    With ``store_path`` the volume is persisted to an on-disk planar-block ``.b2nd``
    store (a directory) and a lazy disk-backed provider is returned (re-open later with
    :meth:`B2ndProvider.open` — no re-ingest); without it, an in-memory provider. The
    returned :class:`MetaEnvelope` is the source seed for ``Engine(meta_seeds=...)`` and
    the edit-time ``propagate_meta`` pass.
    """
    vol, calib = read_image(path)
    m, t, z, c, y, x = vol.shape
    provider = (B2ndProvider.write(vol, store_path, levels=levels) if store_path
                else B2ndProvider.from_array(vol, levels=levels))
    envelope = MetaEnvelope(axes=AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x), metadata=calib)
    return provider, envelope


#: backwards-compatible alias (ND2-only callers) — now format-dispatching.
ingest_nd2 = ingest_image


def open_store(store_path: str) -> B2ndProvider:
    """Re-open a previously-written on-disk ``.b2nd`` store (no re-ingest)."""
    return B2ndProvider.open(store_path)


__all__ = ["read_calibration", "read_channel_display", "read_nd2", "read_tiff",
           "read_image", "read_meta_only", "ingest_image", "ingest_nd2",
           "open_store", "CHANNEL_DISPLAY_KEYS"]
