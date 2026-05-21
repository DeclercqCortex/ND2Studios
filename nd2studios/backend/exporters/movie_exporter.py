"""
Time-lapse movie exporter (MP4 / GIF).

Renders an RGB composite movie from per-channel timeseries with optional
overlays:

- **Scale bar** — drawn at a configurable corner using `pixel_size_um`
  to set its physical length in micrometers.
- **Timestamp** — drawn in another corner, formatted as `mm:ss` or
  `hh:mm:ss`. Source can be ND2 acquisition timestamps (per frame) or a
  synthetic linear schedule from a user-supplied `dt_seconds`.
- **Channel labels** — small per-channel color swatches with names
  (top-left by default).

Rendering uses `imageio` with the `imageio-ffmpeg` plugin for MP4 and
the built-in plugin for GIF. Overlays are drawn with PIL so we don't
take a hard Qt dependency in the backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.exporters.composite_exporter import (
    CHANNEL_COLORS, ImageAdjustments, _composite_frame, _percentile_uint8,
)


@dataclass
class MovieOptions:
    """Render options for a movie export."""
    fps: float = 10.0
    # 'mp4' or 'gif'. Determined from filename if not set.
    codec: Optional[str] = None
    # Overlays
    show_scale_bar: bool = True
    scale_bar_um: float = 50.0          # physical length
    scale_bar_color: str = "white"
    scale_bar_position: str = "bottom-right"   # one of 4 corners
    scale_bar_thickness_px: int = 6
    show_timestamp: bool = True
    timestamp_position: str = "bottom-left"
    timestamp_color: str = "white"
    timestamp_dt_seconds: Optional[float] = None  # if None, use ND2 timestamps
    timestamp_font_size: int = 18
    show_channel_labels: bool = True
    channel_label_position: str = "top-left"
    channel_label_font_size: int = 14


def _draw_overlays(
    rgb: np.ndarray,
    t_index: int,
    opts: MovieOptions,
    pixel_size_um: float,
    channel_colors: Dict[str, Tuple[int, int, int]],
    channel_enabled: Dict[str, bool],
    channel_names: List[str],
    frame_timestamps: Optional[np.ndarray],
) -> np.ndarray:
    """Paint scale bar / timestamp / channel labels onto a copy of `rgb`."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    h, w = rgb.shape[:2]

    def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        # Try a few common system fonts; fall back to the default.
        for name in ("Helvetica.ttc", "DejaVuSans.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    # ── Scale bar ──
    if opts.show_scale_bar and pixel_size_um > 0:
        bar_px = int(round(opts.scale_bar_um / pixel_size_um))
        if bar_px > 0 and bar_px < w:
            margin = 24
            t = max(2, opts.scale_bar_thickness_px)
            if "right" in opts.scale_bar_position:
                x1 = w - margin
                x0 = x1 - bar_px
            else:
                x0 = margin
                x1 = x0 + bar_px
            if "bottom" in opts.scale_bar_position:
                y0 = h - margin - t
                y1 = h - margin
            else:
                y0 = margin
                y1 = margin + t
            draw.rectangle([x0, y0, x1, y1], fill=opts.scale_bar_color)
            label = f"{opts.scale_bar_um:g} µm"
            font = _font(opts.scale_bar_thickness_px * 2 + 4)
            tx, ty = ((x0 + x1) // 2, y0 - 2 * t - 6)
            tw = draw.textlength(label, font=font)
            draw.text((tx - tw / 2, ty), label, fill=opts.scale_bar_color, font=font)

    # ── Timestamp ──
    if opts.show_timestamp:
        secs: Optional[float] = None
        if (opts.timestamp_dt_seconds is None
                and frame_timestamps is not None
                and t_index < len(frame_timestamps)):
            secs = float(frame_timestamps[t_index])
        elif opts.timestamp_dt_seconds is not None:
            secs = float(opts.timestamp_dt_seconds) * t_index
        if secs is not None:
            text = _format_seconds(secs)
            font = _font(opts.timestamp_font_size)
            margin = 18
            tw = draw.textlength(text, font=font)
            th = opts.timestamp_font_size
            x = w - margin - tw if "right" in opts.timestamp_position else margin
            y = h - margin - th if "bottom" in opts.timestamp_position else margin
            # Light background pill for readability.
            pad = 4
            draw.rectangle(
                [x - pad, y - pad, x + tw + pad, y + th + pad],
                fill=(0, 0, 0, 160),
            )
            draw.text((x, y), text, fill=opts.timestamp_color, font=font)

    # ── Channel labels ──
    if opts.show_channel_labels and channel_names:
        font = _font(opts.channel_label_font_size)
        margin = 14
        line_h = opts.channel_label_font_size + 6
        x = margin if "left" in opts.channel_label_position else w - 200
        y = (margin if "top" in opts.channel_label_position
             else h - margin - line_h * len(channel_names))
        for name in channel_names:
            if not channel_enabled.get(name, True):
                continue
            color = channel_colors.get(name, (255, 255, 255))
            sw = opts.channel_label_font_size
            draw.rectangle([x, y, x + sw, y + sw], fill=color)
            draw.text((x + sw + 6, y - 1), name, fill=opts.timestamp_color, font=font)
            y += line_h

    return np.asarray(img)


# Largest edge libx264 can comfortably encode at Level 6.2 with the
# default 4-frame DPB and B-frames. Anything bigger gets a "frame MB
# size > level limit" error and produces an unreadable file. The
# stitched-panorama outputs from ND2Studios easily blow past this
# (e.g. 11264 × 6144), so we transparently scale down for MP4 export.
MAX_MP4_LONGEST_EDGE = 3840


def _downscale_for_mp4(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
    """Resize frames so the longest edge fits H.264 high-tier limits.

    Returns ``(frames_out, scale)`` where ``scale`` is the linear scale
    factor that was applied (1.0 if no resize was needed).  Dimensions
    are rounded to even numbers so libx264's yuv420p chroma subsampling
    is happy.
    """
    if not frames:
        return frames, 1.0
    h, w = frames[0].shape[:2]
    longest = max(h, w)
    if longest <= MAX_MP4_LONGEST_EDGE:
        # Still enforce even dimensions for yuv420p — pad by one black
        # row/col if needed rather than cropping.
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            out: List[np.ndarray] = []
            for f in frames:
                padded = np.zeros((h + pad_h, w + pad_w, f.shape[2]),
                                   dtype=f.dtype)
                padded[:h, :w] = f
                out.append(padded)
            return out, 1.0
        return frames, 1.0

    scale = MAX_MP4_LONGEST_EDGE / float(longest)
    new_w = max(2, int(round(w * scale)) // 2 * 2)
    new_h = max(2, int(round(h * scale)) // 2 * 2)
    from PIL import Image as _PILImage
    out: List[np.ndarray] = []
    for f in frames:
        img = _PILImage.fromarray(f)
        out.append(np.asarray(
            img.resize((new_w, new_h), _PILImage.LANCZOS)
        ))
    return out, scale


def _format_seconds(s: float) -> str:
    s = max(0.0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def export_movie(
    channels: Dict[str, np.ndarray],
    colors: Dict[str, Tuple[int, int, int]],
    enabled: Dict[str, bool],
    filepath: str,
    options: Optional[MovieOptions] = None,
    pixel_size_um: float = 1.0,
    frame_timestamps_s: Optional[np.ndarray] = None,
    lut_settings: Optional[Dict[str, Tuple[float, float, float]]] = None,
    image_adjustments: Optional[ImageAdjustments] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Write an MP4 / GIF time-lapse with optional overlays.

    Parameters
    ----------
    channels : {channel_name: (T, H, W) array}.
    colors : {channel_name: (R, G, B)}.
    enabled : {channel_name: bool}.
    filepath : output path. `.mp4` or `.gif` extension determines codec.
    options : MovieOptions (see dataclass for defaults).
    pixel_size_um : used to compute scale bar length.
    frame_timestamps_s : per-frame timestamps in seconds. If None and the
        options ask for a timestamp overlay, falls back to
        `options.timestamp_dt_seconds`.
    lut_settings : {channel_name: (lo, hi, gamma)} contrast bounds. When
        provided they override the default percentile auto-stretch so the
        movie matches what the user sees in the viewer.
    progress_cb : 0–100.
    """
    import imageio

    if not channels:
        raise ValueError("export_movie: no channels to export")
    opts = options or MovieOptions()
    sample = next(iter(channels.values()))
    if sample.ndim != 3:
        raise ValueError(f"channels must be (T,H,W) — got {sample.shape}")
    n = sample.shape[0]

    ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
    if not ext:
        filepath += ".mp4"
        ext = "mp4"
    codec = (opts.codec or ext).lower()

    channel_names = list(channels.keys())

    # Render all frames first so we can use format-specific writers that
    # don't suffer from imageio's plugin auto-routing (which can silently
    # pick tifffile and then fail on the 'fps' write kwarg).
    rendered: List[np.ndarray] = []
    for t in range(n):
        frame_dict = {name: arr[t] for name, arr in channels.items()}
        rgb = _composite_frame(frame_dict, colors, enabled, lut_settings,
                               image_adjustments=image_adjustments)
        rgb_with_overlays = _draw_overlays(
            rgb,
            t_index=t,
            opts=opts,
            pixel_size_um=pixel_size_um,
            channel_colors=colors,
            channel_enabled=enabled,
            channel_names=channel_names,
            frame_timestamps=(
                np.asarray(frame_timestamps_s)
                if frame_timestamps_s is not None else None
            ),
        )
        rendered.append(rgb_with_overlays)
        if progress_cb is not None and (t % 4 == 0 or t == n - 1):
            progress_cb(int((t + 1) / n * 90))

    if codec == "gif":
        from PIL import Image as _PILImage
        duration_ms = max(1, int(1000.0 / opts.fps))
        pil_frames = [_PILImage.fromarray(f) for f in rendered]
        pil_frames[0].save(
            filepath,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    elif codec in {"mp4", "mov", "m4v"}:
        try:
            import imageio_ffmpeg as _iffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "imageio-ffmpeg is required for MP4 export. "
                "Install it with: pip install imageio-ffmpeg"
            ) from exc
        # Cap the longest edge so the H.264 encoder doesn't bail with
        # "frame MB size > level limit" on stitched-panorama outputs.
        # Returns the input list unchanged when no resize is needed but
        # still pads odd dimensions to even for yuv420p.
        in_h, in_w = rendered[0].shape[:2]
        rendered, scale = _downscale_for_mp4(rendered)
        h, w = rendered[0].shape[:2]
        if (in_h, in_w) != (h, w):
            msg = (
                f"MP4: downscaled {in_w}×{in_h} → {w}×{h} "
                f"(scale {scale:.3f}) — H.264 Level 6.2 frame-MB cap"
            )
            if status_cb is not None:
                status_cb(msg)
            else:
                # No status callback wired: at least make it visible in
                # the console so the user sees the downscale happened.
                import sys as _sys
                print(f"[ND2Studios] {msg}", file=_sys.stderr)
        # Hard guard: if for any reason the downscale failed to bring
        # the longest edge under the H.264 frame-MB ceiling, fail loudly
        # here with an actionable error rather than handing libx264 an
        # input it'll reject (producing an unplayable MP4).
        max_mb_per_dim = MAX_MP4_LONGEST_EDGE // 16
        mb_w = (w + 15) // 16
        mb_h = (h + 15) // 16
        if mb_w > max_mb_per_dim or mb_h > max_mb_per_dim:
            raise RuntimeError(
                f"MP4 export: frame {w}×{h} ({mb_w}×{mb_h} macroblocks) "
                f"exceeds H.264 Level 6.2 limits even after downscale. "
                f"Use GIF export, the TIFF Z-stack exporter, or pre-crop "
                f"the canvas."
            )
        # Level 6.2 is libx264's highest supported H.264 level; combined
        # with refs=2 and no B-frames it covers ~3840 px on the longest
        # edge within the DPB macroblock budget.
        # Level 6.2 is libx264's highest supported H.264 level; combined
        # with refs=2 and no B-frames it covers ~3840 px on the longest
        # edge within the DPB macroblock budget.
        writer_gen = _iffmpeg.write_frames(
            filepath.replace("%", "%%"),
            size=(w, h),
            fps=opts.fps,
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            codec="libx264",
            macro_block_size=1,
            output_params=[
                "-crf", "18",
                "-preset", "medium",
                "-level", "6.2",
                "-refs", "2",
                "-bf", "0",
            ],
        )
        writer_gen.send(None)
        for frame in rendered:
            writer_gen.send(frame.astype(np.uint8).tobytes())
        writer_gen.close()
    elif codec in {"tif", "tiff"}:
        imageio.mimsave(filepath, rendered)
    else:
        imageio.mimsave(filepath, rendered, fps=opts.fps)

    if progress_cb is not None:
        progress_cb(100)
