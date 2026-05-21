"""
TIFF stack exporter.

Two writers live here:

* :func:`export_tiff_stack` — single-channel multi-page TIFF
  (``(T, H, W)`` or ``(T, Z, H, W)``). Kept for callers that genuinely
  emit one stack at a time (e.g. label masks from the Analysis page).
* :func:`export_tiff_hyperstack` — multi-channel ImageJ TZCYX hyperstack
  with ``Labels=[<channel names>]``, matching the file construction the
  Stitch dialog writes via :func:`export_stitched_tiff`. This is the
  canonical writer for the Export page's TIFF tab.

Both write ``unit=um`` + ``spacing`` + resolution tags when a pixel size
is supplied, and auto-switch to BigTIFF above 3.9 GB.
"""
from __future__ import annotations

import contextlib
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import tifffile


@contextlib.contextmanager
def _silence_bigtiff_imagej_warning():
    """Suppress tifffile's 'nonconformant BigTIFF ImageJ' warning.

    Modern tifffile readers and recent Fiji versions handle BigTIFF
    files written with ``imagej=True`` correctly, and we rely on the
    ImageJ metadata (Labels, slices, channels, frames) for round-tripping
    stitched/hyperstack exports back into ND2Studios. The warning fires
    on every multi-GB write and is noise for our use case.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"<tifffile\.TiffWriter.*> writing nonconformant BigTIFF ImageJ",
        )
        yield


def _normalize_to_uint8(stack: np.ndarray, p_low: float = 0.5,
                       p_high: float = 99.5) -> np.ndarray:
    f = stack.astype(np.float32)
    lo = np.percentile(f, p_low)
    hi = np.percentile(f, p_high)
    f = np.clip((f - lo) / (hi - lo + 1e-10), 0, 1)
    return (f * 255).astype(np.uint8)


def _normalize_to_uint16(stack: np.ndarray, p_low: float = 0.5,
                         p_high: float = 99.5) -> np.ndarray:
    f = stack.astype(np.float32)
    lo = np.percentile(f, p_low)
    hi = np.percentile(f, p_high)
    f = np.clip((f - lo) / (hi - lo + 1e-10), 0, 1)
    return (f * 65535).astype(np.uint16)


def export_tiff_stack(
    stack: np.ndarray,
    filepath: str,
    bit_depth: str = "passthrough",
    pixel_size_um: Optional[float] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    """Write ``stack`` to a multi-page TIFF.

    Parameters
    ----------
    stack : (T, H, W) or (T, Z, H, W) array of any numeric dtype.
        - (T, H, W) — written as a flat T-frame stack (Z already projected).
        - (T, Z, H, W) — written as an ImageJ TZYX hyperstack so Fiji/napari
          can re-open it with all Z planes intact.
    filepath : output path. ``.tif`` is appended if missing.
    bit_depth : ``"passthrough"`` | ``"uint8"`` | ``"uint16"``.
        ``"passthrough"`` keeps the source dtype; the others rescale using
        the 0.5–99.5 percentile bounds of the whole stack.
    pixel_size_um : if given, written into the TIFF resolution / metadata
        tags so downstream tools (Fiji, napari) get scale right.
    progress_cb : 0–100 progress callback.
    """
    if stack.ndim not in (3, 4):
        raise ValueError(
            f"export_tiff_stack expects (T,H,W) or (T,Z,H,W) — got {stack.shape}"
        )
    if not filepath.lower().endswith((".tif", ".tiff")):
        filepath += ".tif"

    if bit_depth == "uint8":
        stack = _normalize_to_uint8(stack)
    elif bit_depth == "uint16":
        stack = _normalize_to_uint16(stack)
    elif bit_depth != "passthrough":
        raise ValueError(f"Unknown bit_depth: {bit_depth}")

    bigtiff = stack.nbytes > 3_900_000_000

    resolution: Optional[tuple] = None
    if pixel_size_um is not None and pixel_size_um > 0:
        resolution = (1.0 / pixel_size_um, 1.0 / pixel_size_um)

    if stack.ndim == 3:
        # (T, H, W) — flat T-frame stack, one page per timepoint.
        metadata = {}
        if pixel_size_um is not None and pixel_size_um > 0:
            metadata = {"unit": "um", "spacing": pixel_size_um}
        n = stack.shape[0]
        with tifffile.TiffWriter(filepath, bigtiff=bigtiff) as writer:
            for t in range(n):
                writer.write(
                    stack[t],
                    photometric="minisblack",
                    resolution=resolution,
                    resolutionunit="MICROMETER" if resolution else None,
                    metadata=metadata if t == 0 else None,
                )
                if progress_cb is not None and (t % 10 == 0 or t == n - 1):
                    progress_cb(int((t + 1) / n * 100))
    else:
        # (T, Z, H, W) — ImageJ TZYX hyperstack.
        ij_meta: dict = {"axes": "TZYX"}
        if pixel_size_um is not None and pixel_size_um > 0:
            ij_meta["unit"] = "um"
            ij_meta["spacing"] = float(pixel_size_um)
        with _silence_bigtiff_imagej_warning():
            tifffile.imwrite(
                filepath,
                stack,
                imagej=True,
                resolution=resolution,
                metadata=ij_meta,
                photometric="minisblack",
                bigtiff=bigtiff,
            )
        if progress_cb is not None:
            progress_cb(100)


def export_tiff_hyperstack(
    channels: Dict[str, np.ndarray],
    enabled: Dict[str, bool],
    filepath: str,
    bit_depth: str = "passthrough",
    pixel_size_um: Optional[float] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> str:
    """Write enabled channels as a single ImageJ TZCYX hyperstack.

    Parameters
    ----------
    channels : ``{channel_name: array}``. Each array must be ``(T, H, W)``
        or ``(T, Z, H, W)``. All enabled channels must share the same
        ``(T, Z, H, W)``.
    enabled : ``{channel_name: bool}``. Channels with ``False`` are
        skipped. Insertion order of ``channels`` defines the C order in
        the output file.
    filepath : output path. ``.tif`` is appended if missing.
    bit_depth : ``"passthrough"`` | ``"uint16"`` | ``"uint8"``. Per-channel
        0.5–99.5 percentile stretch is applied for the rescaling modes so
        each channel keeps its own contrast.
    pixel_size_um : if given, written as resolution + ``unit=um`` +
        ``spacing`` so downstream tools (Fiji, napari) read scale correctly.
    progress_cb : 0–100 progress callback.

    Returns
    -------
    The final filepath that was written (with ``.tif`` appended if needed).
    """
    enabled_names: List[str] = [n for n in channels if enabled.get(n, True)]
    if not enabled_names:
        raise ValueError("export_tiff_hyperstack: no enabled channels")

    if not filepath.lower().endswith((".tif", ".tiff")):
        filepath += ".tif"

    arrays: List[np.ndarray] = []
    ref_shape: Optional[tuple] = None
    for name in enabled_names:
        a = np.asarray(channels[name])
        if a.ndim == 3:
            a = a[:, np.newaxis, :, :]  # (T, 1, H, W)
        elif a.ndim == 4:
            pass  # (T, Z, H, W)
        else:
            raise ValueError(
                f"export_tiff_hyperstack: channel '{name}' has shape {a.shape}; "
                "expected (T, H, W) or (T, Z, H, W)"
            )
        if ref_shape is None:
            ref_shape = a.shape
        elif a.shape != ref_shape:
            raise ValueError(
                f"export_tiff_hyperstack: channel '{name}' shape {a.shape} "
                f"does not match {ref_shape}"
            )
        arrays.append(a)

    if bit_depth == "uint8":
        arrays = [_normalize_to_uint8(a) for a in arrays]
    elif bit_depth == "uint16":
        arrays = [_normalize_to_uint16(a) for a in arrays]
    elif bit_depth != "passthrough":
        raise ValueError(f"Unknown bit_depth: {bit_depth}")

    # Stack along a new C axis between Z and H so the final shape is
    # (T, Z, C, H, W) — the ImageJ TZCYX hyperstack layout.
    hyper = np.stack(arrays, axis=2)
    bigtiff = hyper.nbytes > 3_900_000_000

    ij_meta: Dict[str, object] = {"Labels": enabled_names}
    resolution: Optional[tuple] = None
    if pixel_size_um is not None and pixel_size_um > 0:
        resolution = (1.0 / pixel_size_um, 1.0 / pixel_size_um)
        ij_meta["unit"] = "um"
        ij_meta["spacing"] = float(pixel_size_um)

    with _silence_bigtiff_imagej_warning():
        tifffile.imwrite(
            filepath,
            hyper,
            imagej=True,
            photometric="minisblack",
            resolution=resolution,
            resolutionunit="MICROMETER" if resolution else None,
            metadata=ij_meta,
            bigtiff=bigtiff,
        )

    if progress_cb is not None:
        progress_cb(100)
    return filepath
