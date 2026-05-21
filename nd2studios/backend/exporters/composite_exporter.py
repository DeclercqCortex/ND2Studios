"""
RGB composite TIFF exporter.

Takes a `Dict[channel_name, (T,H,W)]` plus per-channel colors/enable
flags, and writes a multi-page RGB TIFF (T, H, W, 3) uint8 with each
channel mapped to its assigned color and additively blended.

Also hosts :class:`ImageAdjustments` and :func:`apply_image_adjustments`,
the shared brightness / contrast / saturation / hue / fade pipeline used
by the movie exporter, image-sequence exporter, and the live preview
dialog. The adjustments operate on the final composited uint8 RGB frame
*before* overlays (scale bar, timestamp, channel labels) are drawn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import tifffile


@dataclass
class ImageAdjustments:
    """Colour adjustments shared by movie / image-sequence / preview.

    All values are user-facing slider scales. Identity = all zeros. The
    values map as:

    - ``brightness`` ∈ [-100, 100] — *multiplicative gain* applied to
      each channel's grayscale intensity before its colour LUT is
      mixed in. ``gain = 1 + brightness/100``, so +100 doubles signal
      and -100 collapses to black. Because it scales rather than
      offsets, background pixels stay near zero and only above-floor
      signal climbs visibly. Brightening a red channel therefore
      intensifies red without washing toward white; brightening a
      green channel intensifies green; etc.
    - ``contrast`` ∈ [-100, 100] — pivots around 128 on the final RGB.
      +100 doubles slope, -100 collapses to mid-grey.
    - ``saturation`` ∈ [-100, 100] — -100 desaturates to grey, +100
      doubles chroma.
    - ``hue`` ∈ [-180, 180] — rotates hue in HSV by the given degrees.
    - ``fade`` ∈ [0, 100] — % blend toward black (0 = unchanged).
    """
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    fade: float = 0.0

    def is_identity(self) -> bool:
        return (self.brightness == 0.0 and self.contrast == 0.0
                and self.saturation == 0.0 and self.hue == 0.0
                and self.fade == 0.0)

    def is_identity_post_composite(self) -> bool:
        """True when only brightness is non-zero (brightness is per-channel)."""
        return (self.contrast == 0.0 and self.saturation == 0.0
                and self.hue == 0.0 and self.fade == 0.0)


def _apply_brightness_to_gray(
    gray: np.ndarray, brightness: float,
) -> np.ndarray:
    """Scale a uint8 grayscale array by a multiplicative brightness gain.

    ``brightness`` is the user-facing slider value in [-100, 100]; the
    gain applied is ``1 + brightness/100``. So +100 doubles intensity
    (signal climbs, bright pixels clip at 255) and -100 collapses to
    black. The multiplicative semantics matter for fluorescence: a
    pixel sitting at the background floor (≈ 0) stays at the floor
    regardless of the brightness slider, while signal above background
    scales visibly. An *additive* brightness would lift the noise
    floor just as much as the peaks and wash the image out.

    This helper runs in channel-grayscale space — before the channel
    colour LUT is mixed in — so the gain intensifies the channel's
    assigned colour rather than shifting the final composite toward
    white.
    """
    if brightness == 0.0:
        return gray
    gain = 1.0 + (float(brightness) / 100.0)
    f = gray.astype(np.float32) * gain
    return np.clip(f, 0.0, 255.0).astype(np.uint8)


def apply_image_adjustments(
    rgb: np.ndarray,
    adj: Optional[ImageAdjustments],
) -> np.ndarray:
    """Apply brightness/contrast/saturation/hue/fade to a uint8 RGB image.

    Operates in-order: contrast → brightness → hue → saturation → fade.
    Returns a new uint8 array; the input is not modified.

    Brightness is implemented as multiplicative gain
    (``gain = 1 + brightness/100``), so background pixels at zero stay
    at zero and only signal above the noise floor scales visibly —
    matching :func:`_apply_brightness_to_gray` so the standalone helper
    and the composite path are consistent.

    .. note::
       When called from :func:`_composite_frame`, brightness is applied
       per-channel before this function runs and the post-composite
       call here skips the brightness step (the ``adj.brightness``
       field is set to 0 by the caller).
    """
    if adj is None or adj.is_identity():
        return rgb
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    f = rgb.astype(np.float32)

    if adj.contrast != 0.0:
        # Slope around 128. +100 → 2x; -100 → 0x.
        slope = 1.0 + (float(adj.contrast) / 100.0)
        f = (f - 128.0) * slope + 128.0

    if adj.brightness != 0.0:
        # Multiplicative gain — see _apply_brightness_to_gray docstring.
        gain = 1.0 + (float(adj.brightness) / 100.0)
        f = f * gain

    f = np.clip(f, 0.0, 255.0)

    if adj.hue != 0.0 or adj.saturation != 0.0:
        import colorsys
        # Vectorised RGB→HSV→RGB via numpy. Avoid colorsys per-pixel.
        rgb01 = f / 255.0
        r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        v = mx
        delta = mx - mn
        s = np.where(mx > 0, delta / np.where(mx > 0, mx, 1.0), 0.0)

        h = np.zeros_like(mx)
        nz = delta > 1e-8
        rc = np.where(nz, (mx - r) / np.where(nz, delta, 1.0), 0.0)
        gc = np.where(nz, (mx - g) / np.where(nz, delta, 1.0), 0.0)
        bc = np.where(nz, (mx - b) / np.where(nz, delta, 1.0), 0.0)
        h = np.where(r == mx, bc - gc, h)
        h = np.where(g == mx, 2.0 + rc - bc, h)
        h = np.where(b == mx, 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0
        h = np.where(nz, h, 0.0)

        if adj.hue != 0.0:
            h = (h + (float(adj.hue) / 360.0)) % 1.0
        if adj.saturation != 0.0:
            s = np.clip(s * (1.0 + float(adj.saturation) / 100.0), 0.0, 1.0)

        i = np.floor(h * 6.0).astype(np.int32)
        f_h = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f_h)
        t = v * (1.0 - s * (1.0 - f_h))
        i_mod = i % 6
        out = np.zeros_like(rgb01)
        for sector, channels in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                            (p, q, v), (t, p, v), (v, p, q)]):
            mask = i_mod == sector
            for ci, comp in enumerate(channels):
                out[..., ci] = np.where(mask, comp, out[..., ci])
        f = out * 255.0

    if adj.fade > 0.0:
        f = f * (1.0 - float(adj.fade) / 100.0)

    return np.clip(f, 0.0, 255.0).astype(np.uint8)


CHANNEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "gray": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (255, 0, 0),
    "blue": (0, 100, 255),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "white": (255, 255, 255),
}


def _percentile_uint8(frame: np.ndarray, p_low: float = 0.5,
                       p_high: float = 99.5) -> np.ndarray:
    f = frame.astype(np.float32)
    lo = np.percentile(f, p_low)
    hi = np.percentile(f, p_high)
    f = np.clip((f - lo) / (hi - lo + 1e-10), 0, 1)
    return (f * 255).astype(np.uint8)


def _apply_lut(frame: np.ndarray, lo: float, hi: float, gamma: float) -> np.ndarray:
    """Map frame to uint8 [0,255] using explicit contrast bounds + gamma."""
    f = frame.astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    f = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        f = f ** (1.0 / max(gamma, 0.05))
    return (f * 255).astype(np.uint8)


def _composite_frame(
    frames: Dict[str, np.ndarray],
    colors: Dict[str, Tuple[int, int, int]],
    enabled: Dict[str, bool],
    lut_settings: Optional[Dict[str, Tuple[float, float, float]]] = None,
    image_adjustments: Optional[ImageAdjustments] = None,
) -> np.ndarray:
    """Composite channels into a single RGB uint8 frame.

    When ``lut_settings`` provides ``(lo, hi, gamma)`` for a channel those
    values are used; otherwise a 0.5–99.5 percentile auto-stretch is applied.
    When ``image_adjustments`` is provided the brightness / contrast /
    saturation / hue / fade pipeline is applied to the final RGB before
    return.
    """
    sample = next(iter(frames.values()))
    h, w = sample.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    brightness = (float(image_adjustments.brightness)
                  if image_adjustments is not None else 0.0)
    for name, frame in frames.items():
        if not enabled.get(name, True):
            continue
        if lut_settings and name in lut_settings:
            lo, hi, gamma = lut_settings[name]
            gray = _apply_lut(frame, lo, hi, gamma)
        else:
            gray = _percentile_uint8(frame)
        if brightness != 0.0:
            gray = _apply_brightness_to_gray(gray, brightness)
        gray_f = gray.astype(np.float32)
        r, g, b = colors.get(name, (255, 255, 255))
        out[..., 0] += gray_f * (r / 255.0)
        out[..., 1] += gray_f * (g / 255.0)
        out[..., 2] += gray_f * (b / 255.0)
    rgb = np.clip(out, 0, 255).astype(np.uint8)

    if image_adjustments is not None and not image_adjustments.is_identity_post_composite():
        # Brightness was already baked into the per-channel grayscale above;
        # zero it out here so the post-composite pass only handles contrast,
        # saturation, hue, and fade.
        post = ImageAdjustments(
            brightness=0.0,
            contrast=image_adjustments.contrast,
            saturation=image_adjustments.saturation,
            hue=image_adjustments.hue,
            fade=image_adjustments.fade,
        )
        rgb = apply_image_adjustments(rgb, post)
    return rgb


def export_rgb_composite_tiff(
    channels: Dict[str, np.ndarray],
    colors: Dict[str, Tuple[int, int, int]],
    enabled: Dict[str, bool],
    filepath: str,
    pixel_size_um: Optional[float] = None,
    lut_settings: Optional[Dict[str, Tuple[float, float, float]]] = None,
    image_adjustments: Optional[ImageAdjustments] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    """Write an RGB composite TIFF stack.

    Parameters
    ----------
    channels : {channel_name: (T, H, W) array}
    colors : {channel_name: (R, G, B) uint8 tuple}
    enabled : {channel_name: bool}
    filepath : output path. `.tif` is appended if missing.
    pixel_size_um : optional scale tag.
    lut_settings : {channel_name: (lo, hi, gamma)} contrast bounds. When
        provided they override the default percentile auto-stretch so the
        export matches what the user sees in the viewer.
    progress_cb : 0–100.
    """
    if not channels:
        raise ValueError("export_rgb_composite_tiff: no channels to export")

    if not filepath.lower().endswith((".tif", ".tiff")):
        filepath += ".tif"

    sample = next(iter(channels.values()))
    if sample.ndim != 3:
        raise ValueError(f"channels must be (T,H,W) — got {sample.shape}")
    n = sample.shape[0]

    resolution = None
    metadata = {}
    if pixel_size_um is not None and pixel_size_um > 0:
        resolution = (1.0 / pixel_size_um, 1.0 / pixel_size_um)
        metadata["unit"] = "um"

    bigtiff = n * sample.shape[1] * sample.shape[2] * 3 > 3_900_000_000

    with tifffile.TiffWriter(filepath, bigtiff=bigtiff) as writer:
        for t in range(n):
            frame_dict = {name: arr[t] for name, arr in channels.items()}
            rgb = _composite_frame(frame_dict, colors, enabled, lut_settings,
                                   image_adjustments=image_adjustments)
            writer.write(
                rgb,
                photometric="rgb",
                resolution=resolution,
                resolutionunit="MICROMETER" if resolution else None,
                metadata=metadata if t == 0 else None,
            )
            if progress_cb is not None and (t % 5 == 0 or t == n - 1):
                progress_cb(int((t + 1) / n * 100))
