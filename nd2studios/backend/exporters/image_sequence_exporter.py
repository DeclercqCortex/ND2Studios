"""
Image-sequence exporter.

Writes one PNG per frame of a multi-axis dataset. The output filename is
built from a user-supplied basename plus per-axis index suffixes
(``_T01``, ``_M02``, ``_Z03``) that are only included when the
corresponding axis has more than one frame. Index widths track the digit
count of each axis's largest index (``T999`` keeps 3-digit padding;
``T9`` stays single-digit).

Two iteration paths are supported, mirroring the rest of the export
pipeline:

* **Current-channels** (default): iterates ``T`` over an in-memory
  ``Dict[channel_name, (T, H, W)]`` produced by the recipe — same data
  the movie and composite TIFF exporters consume. Only ``T`` varies;
  the filename uses only ``_T``.
* **Raw volume** (when supplied): iterates ``(M, T, Z)`` directly off a
  ``LazyND2Volume``. The recipe is *not* applied in this mode because
  the recipe is tuned for one specific (M, Z) view; iterating all
  positions would re-run the recipe per slice which is out of scope
  for V1.0 of this feature. Channel selection / colour / LUT / image
  adjustments still apply.

Per-frame compositing reuses :func:`_composite_frame` so the colour
pipeline (LUT → channel colour → image adjustments) matches the movie
exporter byte-for-byte.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.exporters.composite_exporter import (
    ImageAdjustments, _composite_frame,
)
from nd2studios.backend.exporters.movie_exporter import (
    MovieOptions, _draw_overlays,
)


def _width(n: int) -> int:
    """Digit count of the largest 1-based index for ``n`` frames."""
    return max(1, len(str(max(1, int(n)))))


def format_frame_name(
    basename: str,
    t: int,
    m: int,
    z: int,
    n_t: int,
    n_m: int,
    n_z: int,
    extension: str = ".png",
) -> str:
    """Build a per-frame filename including only axes with more than one frame.

    Indices are zero-based on input; the suffix is 1-based and padded to
    the digit width of the corresponding axis size.

    Examples (basename ``"sample"``, ``.png``):

    - ``n_t=10, n_m=1, n_z=1`` → ``sample_T01.png`` … ``sample_T10.png``
    - ``n_t=120, n_m=4, n_z=1`` → ``sample_T001_M1.png``
    - ``n_t=10, n_m=1, n_z=5`` → ``sample_T01_Z1.png``
    - ``n_t=1, n_m=1, n_z=1`` → ``sample.png`` (no suffix)
    """
    parts: List[str] = [basename]
    if n_t > 1:
        parts.append(f"T{int(t) + 1:0{_width(n_t)}d}")
    if n_m > 1:
        parts.append(f"M{int(m) + 1:0{_width(n_m)}d}")
    if n_z > 1:
        parts.append(f"Z{int(z) + 1:0{_width(n_z)}d}")
    if not extension.startswith("."):
        extension = "." + extension
    return "_".join(parts) + extension


@dataclass
class ImageSequenceRequest:
    """Description of one image-sequence export job."""

    output_dir: str
    basename: str

    # Source A: in-memory channels (T, H, W). Recipe-applied if the caller
    # passes the experiment's _processed_channels.
    channels: Dict[str, np.ndarray] = field(default_factory=dict)

    # Source B: lazy multi-axis volume for full (M, T, Z) iteration.
    # When non-None, ``channels`` is ignored — frames are pulled from
    # the volume on demand.
    raw_volume: Optional[Any] = None
    z_mode: str = "none"          # 'none' | 'max' | 'mean' | 'min'
    z_view_index: int = 0         # only used when z_mode != 'none' degenerates to 0
    crop_rect: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)

    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    enabled: Dict[str, bool] = field(default_factory=dict)
    lut_settings: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    image_adjustments: Optional[ImageAdjustments] = None

    # Overlays — same set as the movie exporter.
    movie_options: Optional[MovieOptions] = None
    pixel_size_um: float = 1.0
    frame_timestamps_s: Optional[np.ndarray] = None


def export_image_sequence(
    request: ImageSequenceRequest,
    progress_cb: Optional[Callable[[int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Write one PNG per (T, M, Z) frame and return the output directory.

    Caller decides which axes to iterate by what they put in ``request``:
    pass ``channels`` (T only) or ``raw_volume`` (M × T × Z). The
    function is a pure side-effect — every frame is independently
    composited and written; nothing is buffered between frames besides
    the active PIL handle.
    """
    from PIL import Image

    os.makedirs(request.output_dir, exist_ok=True)
    opts = request.movie_options or MovieOptions()
    adjustments = request.image_adjustments
    overlays_enabled = (
        opts.show_scale_bar or opts.show_timestamp or opts.show_channel_labels
    )

    if request.raw_volume is not None:
        return _export_from_volume(
            request, opts, adjustments, overlays_enabled,
            progress_cb=progress_cb, status_cb=status_cb,
        )

    if not request.channels:
        raise ValueError("export_image_sequence: no channels and no raw_volume")

    sample = next(iter(request.channels.values()))
    if sample.ndim != 3:
        raise ValueError(f"channels must be (T, H, W) — got {sample.shape}")
    n_t = sample.shape[0]
    channel_names = list(request.channels.keys())
    frame_ts = (
        np.asarray(request.frame_timestamps_s)
        if request.frame_timestamps_s is not None else None
    )

    for t in range(n_t):
        if status_cb is not None and (t == 0 or t % 8 == 0):
            status_cb(f"Writing frame {t + 1}/{n_t}…")
        frame_dict = {name: arr[t] for name, arr in request.channels.items()}
        rgb = _composite_frame(
            frame_dict, request.colors, request.enabled,
            request.lut_settings or None,
            image_adjustments=adjustments,
        )
        if overlays_enabled:
            rgb = _draw_overlays(
                rgb, t_index=t, opts=opts,
                pixel_size_um=request.pixel_size_um,
                channel_colors=request.colors,
                channel_enabled=request.enabled,
                channel_names=channel_names,
                frame_timestamps=frame_ts,
            )
        fname = format_frame_name(
            request.basename, t=t, m=0, z=0,
            n_t=n_t, n_m=1, n_z=1,
        )
        Image.fromarray(rgb).save(os.path.join(request.output_dir, fname))
        if progress_cb is not None and (t % 4 == 0 or t == n_t - 1):
            progress_cb(int((t + 1) / n_t * 100))
    return request.output_dir


def _export_from_volume(
    request: ImageSequenceRequest,
    opts: MovieOptions,
    adjustments: Optional[ImageAdjustments],
    overlays_enabled: bool,
    progress_cb: Optional[Callable[[int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Iterate (M, T, Z) off a ``LazyND2Volume`` and write one PNG per frame.

    The Z axis only iterates when ``z_mode == 'none'`` and the file has
    more than one Z slice; otherwise the projection collapses Z to a
    single image per (M, T).
    """
    from PIL import Image

    vol = request.raw_volume
    n_m = int(vol.n_multipoints)
    n_t = int(vol.n_timepoints)
    iterate_z = (request.z_mode == "none" and vol.n_zslices > 1)
    n_z_out = int(vol.n_zslices) if iterate_z else 1

    enabled_channels: List[Tuple[int, str]] = [
        (c, name) for c, name in enumerate(vol.channel_names)
        if request.enabled.get(name, True)
    ]
    if not enabled_channels:
        raise ValueError("image-sequence export: no enabled channels")

    cx, cy, cw, ch_px = (request.crop_rect if request.crop_rect
                         else (0, 0, 0, 0))
    total = n_m * n_t * n_z_out
    written = 0
    frame_ts = (
        np.asarray(request.frame_timestamps_s)
        if request.frame_timestamps_s is not None else None
    )

    for m in range(n_m):
        for t in range(n_t):
            for z_out in range(n_z_out):
                if status_cb is not None and (written == 0 or written % 8 == 0):
                    status_cb(
                        f"Writing M{m + 1}/{n_m} T{t + 1}/{n_t}"
                        + (f" Z{z_out + 1}/{n_z_out}" if iterate_z else "")
                        + "…"
                    )
                frame_dict: Dict[str, np.ndarray] = {}
                for c, name in enabled_channels:
                    frame = vol.get_frame(
                        c=c, m=m, t=t,
                        z=z_out if iterate_z else request.z_view_index,
                        z_mode=request.z_mode if not iterate_z else "none",
                    )
                    if request.crop_rect is not None and cw > 0 and ch_px > 0:
                        frame = frame[cy:cy + ch_px, cx:cx + cw]
                    frame_dict[name] = frame
                rgb = _composite_frame(
                    frame_dict, request.colors, request.enabled,
                    request.lut_settings or None,
                    image_adjustments=adjustments,
                )
                if overlays_enabled:
                    rgb = _draw_overlays(
                        rgb, t_index=t, opts=opts,
                        pixel_size_um=request.pixel_size_um,
                        channel_colors=request.colors,
                        channel_enabled=request.enabled,
                        channel_names=[name for _, name in enabled_channels],
                        frame_timestamps=frame_ts,
                    )
                fname = format_frame_name(
                    request.basename, t=t, m=m, z=z_out,
                    n_t=n_t, n_m=n_m, n_z=n_z_out,
                )
                Image.fromarray(rgb).save(
                    os.path.join(request.output_dir, fname)
                )
                written += 1
                if progress_cb is not None and (written % 4 == 0 or written == total):
                    progress_cb(int(written / total * 100))

    return request.output_dir
