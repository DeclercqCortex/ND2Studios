"""
Built-in image enhancement plugins.
Wraps existing functions from timeseries_processor.py as EnhancementPlugins.
"""
from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from nd2studios.core.plugin_registry import EnhancementPlugin, ParamSpec, PluginBase


@PluginBase.register
class NormalizePlugin(EnhancementPlugin):
    name = "Normalize"
    description = "Percentile-based intensity normalization"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("p_low", "Low percentile", "float", 1.0, 0.0, 10.0, 0.5,
                      tooltip="Clip intensities below this percentile"),
            ParamSpec("p_high", "High percentile", "float", 99.8, 90.0, 100.0, 0.1,
                      tooltip="Clip intensities above this percentile"),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        T = volume.shape[0]
        result = np.zeros_like(volume)
        p_low = params.get("p_low", 1.0)
        p_high = params.get("p_high", 99.8)

        for t in range(T):
            frame = volume[t].astype(np.float32)
            lo = np.percentile(frame, p_low)
            hi = np.percentile(frame, p_high)
            frame = np.clip((frame - lo) / (hi - lo + 1e-10), 0, 1)
            result[t] = (frame * 65535).astype(volume.dtype) if volume.dtype == np.uint16 else (frame * 255).astype(volume.dtype)
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class CLAHEPlugin(EnhancementPlugin):
    name = "CLAHE"
    description = "Contrast Limited Adaptive Histogram Equalization"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("clip_limit", "Clip limit", "float", 2.0, 0.5, 20.0, 0.5,
                      tooltip="Contrast limit for CLAHE"),
            ParamSpec("tile_size", "Tile grid size", "int", 8, 2, 32, 1,
                      tooltip="Number of tiles per dimension"),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        clip = params.get("clip_limit", 2.0)
        tiles = params.get("tile_size", 8)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))

        T = volume.shape[0]
        result = np.zeros_like(volume)
        for t in range(T):
            frame = volume[t]
            if frame.dtype == np.uint16:
                result[t] = clahe.apply(frame)
            else:
                result[t] = clahe.apply(frame.astype(np.uint8))
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class GaussianBlurPlugin(EnhancementPlugin):
    name = "Gaussian Blur"
    description = "Gaussian smoothing filter"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("sigma", "Sigma (px)", "float", 1.0, 0.1, 20.0, 0.1),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        sigma = params.get("sigma", 1.0)
        T = volume.shape[0]
        result = np.zeros_like(volume)
        for t in range(T):
            result[t] = cv2.GaussianBlur(volume[t].astype(np.float32), (0, 0), sigma).astype(volume.dtype)
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class MedianFilterPlugin(EnhancementPlugin):
    name = "Median Filter"
    description = "Median noise reduction filter"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("kernel_size", "Kernel size", "int", 3, 3, 15, 2,
                      tooltip="Must be odd"),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        ksize = params.get("kernel_size", 3)
        if ksize % 2 == 0:
            ksize += 1
        T = volume.shape[0]
        result = np.zeros_like(volume)
        for t in range(T):
            result[t] = cv2.medianBlur(volume[t], ksize)
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class BackgroundSubtractPlugin(EnhancementPlugin):
    name = "Background Subtract"
    description = "Gaussian-blur background subtraction"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("sigma", "BG sigma (px)", "float", 100.0, 10.0, 500.0, 10.0,
                      tooltip="Gaussian sigma for background estimation"),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        sigma = params.get("sigma", 100.0)
        T = volume.shape[0]
        result = np.zeros_like(volume)
        for t in range(T):
            frame = volume[t].astype(np.float32)
            bg = cv2.GaussianBlur(frame, (0, 0), sigma)
            diff = np.clip(frame - bg, 0, None)
            result[t] = diff.astype(volume.dtype)
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class GammaCorrectionPlugin(EnhancementPlugin):
    name = "Gamma Correction"
    description = "Power-law intensity transform"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("gamma", "Gamma", "float", 1.0, 0.1, 5.0, 0.1,
                      tooltip="<1 brightens, >1 darkens"),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        gamma = params.get("gamma", 1.0)
        max_val = np.iinfo(volume.dtype).max if np.issubdtype(volume.dtype, np.integer) else 1.0
        T = volume.shape[0]
        result = np.zeros_like(volume)
        for t in range(T):
            frame = volume[t].astype(np.float64) / max_val
            frame = np.power(frame, gamma)
            result[t] = (frame * max_val).astype(volume.dtype)
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class BleachCorrectionPlugin(EnhancementPlugin):
    name = "Bleach Correction"
    description = "Frame-mean intensity normalization to correct photobleaching"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("method", "Method", "choice", "ratio",
                      choices=["ratio", "exponential"],
                      tooltip="ratio: simple mean normalization. exponential: fit decay curve"),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        from nd2studios.backend.normalization import normalize_timeseries
        return normalize_timeseries(volume, progress_cb=progress_cb)


# ═══════════════════════════════════════════════════════════════
#  Temporal intensity correction
# ═══════════════════════════════════════════════════════════════

@PluginBase.register
class TemporalFoldCorrectionPlugin(EnhancementPlugin):
    name = "Temporal Fold Correction"
    description = "Corrects aberrant frames by clamping each frame's mean to a rolling average"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("window", "Rolling window (frames)", "int", 11, 3, 51, 2,
                      tooltip="Size of the temporal rolling average window.\n"
                              "Each frame's mean is compared to the rolling\n"
                              "average of nearby frames. Odd numbers work best."),
            ParamSpec("threshold", "Correction threshold", "float", 0.1, 0.01, 1.0, 0.01,
                      tooltip="Maximum allowed fold deviation from the rolling mean.\n"
                              "0.1 = frames more than 10% off are corrected."),
            ParamSpec("mode", "Mode", "choice", "global",
                      choices=["global", "local-tile", "local-gaussian"],
                      tooltip="Global: correct using the whole-frame mean.\n"
                              "Local-tile: correct per rectangular tile.\n"
                              "Local-gaussian: smooth correction field with a\n"
                              "Gaussian — best for gradual spatial variation."),
            ParamSpec("tile_x", "Tile width (px)", "int", 128, 16, 1024, 16,
                      tooltip="Tile width for local-tile mode."),
            ParamSpec("tile_y", "Tile height (px)", "int", 128, 16, 1024, 16,
                      tooltip="Tile height for local-tile mode."),
            ParamSpec("local_sigma", "Local sigma (px)", "float", 100.0, 10.0, 500.0, 10.0,
                      tooltip="Gaussian sigma for local-gaussian mode.\n"
                              "Controls the spatial scale of correction.\n"
                              "Larger = smoother correction field."),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        window = params.get("window", 11)
        threshold = params.get("threshold", 0.1)
        mode = params.get("mode", "global")
        tile_x = params.get("tile_x", 128)
        tile_y = params.get("tile_y", 128)
        local_sigma = params.get("local_sigma", 100.0)

        if mode == "global":
            return self._correct_global(volume, window, threshold, progress_cb)
        elif mode == "local-gaussian":
            return self._correct_local_gaussian(volume, window, threshold, local_sigma, progress_cb)
        else:
            return self._correct_local(volume, window, threshold, tile_x, tile_y, progress_cb)

    def _correct_global(self, volume, window, threshold, progress_cb):
        T = volume.shape[0]
        means = np.array([float(volume[t].astype(np.float64).mean()) for t in range(T)])

        # Rolling average (padded at edges)
        kernel = np.ones(window) / window
        padded = np.pad(means, window // 2, mode="edge")
        rolling = np.convolve(padded, kernel, mode="same")[window // 2: window // 2 + T]

        # Correction ratios
        ratios = rolling / np.clip(means, 1e-10, None)

        result = np.zeros_like(volume)
        for t in range(T):
            deviation = abs(ratios[t] - 1.0)
            if deviation > threshold:
                # Correct this frame
                result[t] = np.clip(
                    volume[t].astype(np.float32) * ratios[t], 0, 65535
                ).astype(volume.dtype)
            else:
                result[t] = volume[t]
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result

    def _correct_local(self, volume, window, threshold, tile_x, tile_y, progress_cb):
        T, H, W = volume.shape
        result = volume.copy()

        ny = max(1, (H + tile_y - 1) // tile_y)
        nx = max(1, (W + tile_x - 1) // tile_x)

        for ty_i in range(ny):
            for tx_i in range(nx):
                y0 = ty_i * tile_y
                y1 = min(y0 + tile_y, H)
                x0 = tx_i * tile_x
                x1 = min(x0 + tile_x, W)

                tile = volume[:, y0:y1, x0:x1]
                means = np.array([float(tile[t].astype(np.float64).mean()) for t in range(T)])

                kernel = np.ones(window) / window
                padded = np.pad(means, window // 2, mode="edge")
                rolling = np.convolve(padded, kernel, mode="same")[window // 2: window // 2 + T]
                ratios = rolling / np.clip(means, 1e-10, None)

                for t in range(T):
                    if abs(ratios[t] - 1.0) > threshold:
                        result[t, y0:y1, x0:x1] = np.clip(
                            tile[t].astype(np.float32) * ratios[t], 0, 65535
                        ).astype(volume.dtype)

            if progress_cb:
                progress_cb(int((ty_i + 1) / ny * 100))

        return result

    def _correct_local_gaussian(self, volume, window, threshold, sigma, progress_cb):
        """Gaussian-based local correction: compute per-pixel rolling mean
        ratio using a Gaussian-blurred version of each frame, then correct
        pixels where the local fold deviation exceeds the threshold."""
        T, H, W = volume.shape
        result = volume.copy()

        # Compute temporal rolling average per pixel using blurred frames
        # This is memory intensive but produces smooth correction fields
        blurred = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            blurred[t] = cv2.GaussianBlur(volume[t].astype(np.float32), (0, 0), sigma)

        # For each pixel, compute rolling temporal mean of the blurred version
        kernel = np.ones(window) / window
        for y in range(H):
            for x in range(W):
                ts = blurred[:, y, x]
                padded = np.pad(ts, window // 2, mode="edge")
                rolling = np.convolve(padded, kernel, mode="same")[window // 2: window // 2 + T]
                ratios = rolling / np.clip(ts, 1e-10, None)
                for t in range(T):
                    if abs(ratios[t] - 1.0) > threshold:
                        result[t, y, x] = np.clip(
                            volume[t, y, x].astype(np.float32) * ratios[t], 0, 65535
                        ).astype(volume.dtype)

            if progress_cb and (y % 50 == 0):
                progress_cb(int((y + 1) / H * 100))

        return result


@PluginBase.register
class SpatialFlatnessPlugin(EnhancementPlugin):
    name = "Spatial Flatness"
    description = "Flattens spatial illumination variation across the field of view"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("method", "Method", "choice", "divide",
                      choices=["divide", "subtract"],
                      tooltip="Divide: output = image / background × mean.\n"
                              "  Preserves relative intensities.\n"
                              "Subtract: output = image - background + mean.\n"
                              "  Preserves absolute offsets."),
            ParamSpec("sigma", "Background sigma (px)", "float", 100.0, 10.0, 500.0, 10.0,
                      tooltip="Gaussian blur sigma for estimating the background.\n"
                              "Should be much larger than cell size.\n"
                              "Larger = more aggressive flattening."),
            ParamSpec("temporal", "Temporal averaging", "choice", "per-frame",
                      choices=["per-frame", "time-averaged"],
                      tooltip="Per-frame: estimate background each frame independently.\n"
                              "Time-averaged: average the background estimate\n"
                              "across all frames first, then apply to each frame.\n"
                              "Time-averaged is better for slow-varying illumination."),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        method = params.get("method", "divide")
        sigma = params.get("sigma", 100.0)
        temporal = params.get("temporal", "per-frame")
        T, H, W = volume.shape
        result = np.zeros_like(volume)

        if temporal == "time-averaged":
            # Compute average background across all frames
            bg_sum = np.zeros((H, W), dtype=np.float64)
            for t in range(T):
                bg_sum += cv2.GaussianBlur(volume[t].astype(np.float32), (0, 0), sigma)
            bg_avg = (bg_sum / T).astype(np.float32)
            bg_avg = np.clip(bg_avg, 1.0, None)
            global_mean = float(bg_avg.mean())

            for t in range(T):
                f = volume[t].astype(np.float32)
                if method == "divide":
                    corrected = (f / bg_avg) * global_mean
                else:
                    corrected = f - bg_avg + global_mean
                result[t] = np.clip(corrected, 0, 65535).astype(volume.dtype)
                if progress_cb and (t % 10 == 0):
                    progress_cb(int((t + 1) / T * 100))
        else:
            for t in range(T):
                f = volume[t].astype(np.float32)
                bg = cv2.GaussianBlur(f, (0, 0), sigma)
                bg = np.clip(bg, 1.0, None)
                frame_mean = float(bg.mean())
                if method == "divide":
                    corrected = (f / bg) * frame_mean
                else:
                    corrected = f - bg + frame_mean
                result[t] = np.clip(corrected, 0, 65535).astype(volume.dtype)
                if progress_cb and (t % 10 == 0):
                    progress_cb(int((t + 1) / T * 100))

        return result


# ═══════════════════════════════════════════════════════════════
#  Advanced: cell border and intensity enhancement
# ═══════════════════════════════════════════════════════════════

@PluginBase.register
class TopHatPlugin(EnhancementPlugin):
    name = "Top-Hat"
    description = "Morphological top-hat: extracts bright spots smaller than the kernel"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("mode", "Mode", "choice", "white",
                      choices=["white", "black"],
                      tooltip="white: bright on dark. black: dark on bright"),
            ParamSpec("disk_radius", "Disk radius (px)", "int", 15, 3, 100, 1,
                      tooltip="Size of structuring element. Should be larger than cells."),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        from skimage.morphology import disk, white_tophat, black_tophat
        radius = params.get("disk_radius", 15)
        mode = params.get("mode", "white")
        selem = disk(radius)
        T = volume.shape[0]
        result = np.zeros_like(volume)
        for t in range(T):
            if mode == "white":
                result[t] = white_tophat(volume[t], selem)
            else:
                result[t] = black_tophat(volume[t], selem)
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class DoGPlugin(EnhancementPlugin):
    name = "Difference of Gaussians"
    description = "Band-pass filter: enhances features of a specific size range"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("sigma_small", "Small sigma (px)", "float", 1.0, 0.5, 20.0, 0.5,
                      tooltip="Inner Gaussian — sets the smallest feature size"),
            ParamSpec("sigma_large", "Large sigma (px)", "float", 10.0, 2.0, 100.0, 1.0,
                      tooltip="Outer Gaussian — sets the largest feature size"),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        s1 = params.get("sigma_small", 1.0)
        s2 = params.get("sigma_large", 10.0)
        T = volume.shape[0]
        result = np.zeros_like(volume)
        for t in range(T):
            f = volume[t].astype(np.float32)
            g1 = cv2.GaussianBlur(f, (0, 0), s1)
            g2 = cv2.GaussianBlur(f, (0, 0), s2)
            diff = np.clip(g1 - g2, 0, None)
            # Rescale to fill dtype range
            if diff.max() > 0:
                diff = diff / diff.max() * np.iinfo(volume.dtype).max
            result[t] = diff.astype(volume.dtype)
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class UnsharpMaskPlugin(EnhancementPlugin):
    name = "Unsharp Mask"
    description = "Sharpens edges: output = original + amount * (original - blurred)"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("sigma", "Blur sigma (px)", "float", 3.0, 0.5, 20.0, 0.5),
            ParamSpec("amount", "Sharpening amount", "float", 1.5, 0.1, 10.0, 0.1,
                      tooltip="How much to amplify edges. 1.0 = subtle, 3.0 = aggressive"),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        sigma = params.get("sigma", 3.0)
        amount = params.get("amount", 1.5)
        T = volume.shape[0]
        result = np.zeros_like(volume)
        max_val = np.iinfo(volume.dtype).max if np.issubdtype(volume.dtype, np.integer) else 1.0
        for t in range(T):
            f = volume[t].astype(np.float32)
            blurred = cv2.GaussianBlur(f, (0, 0), sigma)
            sharpened = f + amount * (f - blurred)
            result[t] = np.clip(sharpened, 0, max_val).astype(volume.dtype)
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class BilateralDenoisePlugin(EnhancementPlugin):
    name = "Bilateral Denoise"
    description = "Edge-preserving noise reduction: smooths flat regions, keeps borders sharp"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("d", "Pixel neighborhood", "int", 9, 3, 25, 2,
                      tooltip="Diameter of pixel neighborhood"),
            ParamSpec("sigma_color", "Color sigma", "float", 75.0, 10.0, 200.0, 5.0,
                      tooltip="Larger = more colors mixed (stronger smoothing)"),
            ParamSpec("sigma_space", "Space sigma", "float", 75.0, 10.0, 200.0, 5.0,
                      tooltip="Larger = farther pixels influence each other"),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        d = params.get("d", 9)
        sc = params.get("sigma_color", 75.0)
        ss = params.get("sigma_space", 75.0)
        T = volume.shape[0]
        result = np.zeros_like(volume)
        for t in range(T):
            # bilateralFilter needs uint8 or float32
            if volume.dtype == np.uint16:
                # Scale to float32 for bilateral
                f = volume[t].astype(np.float32) / 65535.0
                filtered = cv2.bilateralFilter(f, d, sc / 65535.0, ss)
                result[t] = (filtered * 65535).astype(np.uint16)
            else:
                result[t] = cv2.bilateralFilter(volume[t].astype(np.float32), d, sc, ss).astype(volume.dtype)
            if progress_cb and (t % 5 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class MorphGradientPlugin(EnhancementPlugin):
    name = "Morphological Gradient"
    description = "Highlights cell borders: dilation minus erosion"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("kernel_size", "Kernel size (px)", "int", 3, 1, 15, 2),
            ParamSpec("blend", "Blend with original", "float", 0.5, 0.0, 1.0, 0.1,
                      tooltip="0 = borders only, 1 = original only, 0.5 = 50/50 mix"),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        ksize = params.get("kernel_size", 3)
        blend = params.get("blend", 0.5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        T = volume.shape[0]
        result = np.zeros_like(volume)
        max_val = np.iinfo(volume.dtype).max if np.issubdtype(volume.dtype, np.integer) else 1.0
        for t in range(T):
            gradient = cv2.morphologyEx(volume[t], cv2.MORPH_GRADIENT, kernel)
            # Rescale gradient to fill range
            g = gradient.astype(np.float32)
            if g.max() > 0:
                g = g / g.max() * max_val
            mixed = blend * volume[t].astype(np.float32) + (1 - blend) * g
            result[t] = np.clip(mixed, 0, max_val).astype(volume.dtype)
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class LocalContrastPlugin(EnhancementPlugin):
    name = "Local Contrast"
    description = "Enhances local contrast by dividing by a large-scale blurred version"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("sigma", "Background sigma (px)", "float", 50.0, 5.0, 200.0, 5.0,
                      tooltip="Gaussian sigma for estimating local background"),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        sigma = params.get("sigma", 50.0)
        T = volume.shape[0]
        result = np.zeros_like(volume)
        max_val = np.iinfo(volume.dtype).max if np.issubdtype(volume.dtype, np.integer) else 1.0
        for t in range(T):
            f = volume[t].astype(np.float32) + 1.0
            bg = cv2.GaussianBlur(f, (0, 0), sigma)
            bg = np.clip(bg, 1.0, None)
            enhanced = (f / bg)
            enhanced = np.clip((enhanced - 0.5) / 1.5, 0, 1) * max_val
            result[t] = enhanced.astype(volume.dtype)
            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


# ═══════════════════════════════════════════════════════════════
#  Blob detection & subtraction
# ═══════════════════════════════════════════════════════════════

@PluginBase.register
class BlobSubtractPlugin(EnhancementPlugin):
    name = "Blob Subtract"
    description = "Detect and subtract bright blobs (dust, debris, hot pixels)"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("method", "Detection", "choice", "LoG",
                      choices=["LoG", "DoG"],
                      tooltip="LoG: Laplacian of Gaussian (precise, slower).\n"
                              "DoG: Difference of Gaussians (faster)."),
            ParamSpec("min_sigma", "Min blob size", "float", 1.0, 0.5, 20.0, 0.5,
                      tooltip="Minimum blob radius in pixels."),
            ParamSpec("max_sigma", "Max blob size", "float", 10.0, 2.0, 50.0, 1.0,
                      tooltip="Maximum blob radius in pixels."),
            ParamSpec("threshold", "Threshold", "float", 0.05, 0.001, 1.0, 0.005,
                      tooltip="Detection threshold. Lower = more blobs detected.\n"
                              "Start at 0.05, decrease to catch fainter blobs."),
            ParamSpec("action", "Action", "choice", "zero",
                      choices=["zero", "interpolate", "median"],
                      tooltip="Zero: set blob pixels to 0.\n"
                              "Interpolate: fill with local mean.\n"
                              "Median: replace with local median."),
            ParamSpec("expand", "Expand radius", "float", 1.5, 1.0, 5.0, 0.5,
                      tooltip="Multiply detected blob radius by this factor\n"
                              "to ensure the full artifact is removed."),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        from skimage.feature import blob_log, blob_dog

        method = params.get("method", "LoG")
        min_s = params.get("min_sigma", 1.0)
        max_s = params.get("max_sigma", 10.0)
        thresh = params.get("threshold", 0.05)
        action = params.get("action", "zero")
        expand = params.get("expand", 1.5)
        T, H, W = volume.shape
        max_val = np.iinfo(volume.dtype).max if np.issubdtype(volume.dtype, np.integer) else 1.0
        result = volume.copy()

        for t in range(T):
            # Normalize to 0-1 for blob detection
            f = volume[t].astype(np.float64) / max_val

            if method == "LoG":
                blobs = blob_log(f, min_sigma=min_s, max_sigma=max_s,
                                 threshold=thresh, num_sigma=10)
            else:
                blobs = blob_dog(f, min_sigma=min_s, max_sigma=max_s,
                                 threshold=thresh)

            if len(blobs) == 0:
                if progress_cb and (t % 10 == 0):
                    progress_cb(int((t + 1) / T * 100))
                continue

            # Build mask of blob regions
            mask = np.zeros((H, W), dtype=bool)
            yy, xx = np.ogrid[:H, :W]
            for blob in blobs:
                cy, cx, r = blob[0], blob[1], blob[2] * expand
                dist2 = (yy - cy)**2 + (xx - cx)**2
                mask |= (dist2 <= r**2)

            if action == "zero":
                result[t][mask] = 0
            elif action == "interpolate":
                # Replace with local mean from a ring around each blob
                frame = result[t].astype(np.float32)
                dilated = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (15, 15)))
                ring = (dilated > 0) & ~mask
                if ring.any():
                    fill_val = frame[ring].mean()
                else:
                    fill_val = frame[~mask].mean() if (~mask).any() else 0
                result[t][mask] = int(np.clip(fill_val, 0, max_val))
            elif action == "median":
                frame = result[t].copy()
                med = cv2.medianBlur(frame, 15)
                result[t][mask] = med[mask]

            if progress_cb and (t % 10 == 0):
                progress_cb(int((t + 1) / T * 100))

        return result


# ═══════════════════════════════════════════════════════════════
#  Advanced denoising
# ═══════════════════════════════════════════════════════════════

@PluginBase.register
class NLMDenoisePlugin(EnhancementPlugin):
    name = "NLM Denoise"
    description = "Non-Local Means denoising — finds similar patches across the image"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("h", "Filter strength", "float", 10.0, 1.0, 100.0, 1.0,
                      tooltip="Denoising strength. Higher = more smoothing.\n"
                              "10 is good for moderate noise. 20+ for heavy noise.\n"
                              "Too high will blur real features."),
            ParamSpec("patch_size", "Patch size", "int", 7, 3, 15, 2,
                      tooltip="Size of patches used for comparison. Odd number.\n"
                              "Larger = slower but considers more context."),
            ParamSpec("search_size", "Search window", "int", 21, 7, 41, 2,
                      tooltip="Size of the search area for similar patches.\n"
                              "Larger = slower but finds better matches."),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        h = params.get("h", 10.0)
        patch = params.get("patch_size", 7)
        search = params.get("search_size", 21)
        T = volume.shape[0]
        result = np.zeros_like(volume)

        for t in range(T):
            frame = volume[t]
            if frame.dtype == np.uint16:
                # OpenCV NLM only works on uint8, so scale down and back
                f8 = (frame.astype(np.float32) / 65535 * 255).astype(np.uint8)
                denoised = cv2.fastNlMeansDenoising(f8, None, h=h,
                                                     templateWindowSize=patch,
                                                     searchWindowSize=search)
                result[t] = (denoised.astype(np.float32) / 255 * 65535).astype(np.uint16)
            else:
                result[t] = cv2.fastNlMeansDenoising(frame, None, h=h,
                                                      templateWindowSize=patch,
                                                      searchWindowSize=search)
            if progress_cb and (t % 5 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class WaveletDenoisePlugin(EnhancementPlugin):
    name = "Wavelet Denoise"
    description = "Wavelet-based denoising — best for preserving edges while removing noise"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("sigma", "Noise sigma", "float", 0.0, 0.0, 100.0, 1.0,
                      tooltip="Estimated noise standard deviation.\n"
                              "0 = auto-estimate from the image.\n"
                              "Higher = stronger denoising."),
            ParamSpec("wavelet", "Wavelet", "choice", "db1",
                      choices=["db1", "db2", "sym4", "coif1", "bior1.5"],
                      tooltip="Wavelet basis function.\n"
                              "db1 (Haar): fast, good for blocky features.\n"
                              "db2/sym4: smoother, good for cells."),
            ParamSpec("mode", "Mode", "choice", "soft",
                      choices=["soft", "hard"],
                      tooltip="Soft: attenuates noise coefficients (smoother).\n"
                              "Hard: zeroes small coefficients (sharper edges)."),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        from skimage.restoration import denoise_wavelet, estimate_sigma

        sigma = params.get("sigma", 0.0)
        wavelet = params.get("wavelet", "db1")
        mode = params.get("mode", "soft")
        T = volume.shape[0]
        max_val = np.iinfo(volume.dtype).max if np.issubdtype(volume.dtype, np.integer) else 1.0
        result = np.zeros_like(volume)

        for t in range(T):
            f = volume[t].astype(np.float64) / max_val

            if sigma == 0:
                sig = estimate_sigma(f)
            else:
                sig = sigma / max_val

            denoised = denoise_wavelet(f, sigma=sig, wavelet=wavelet,
                                        mode=mode, rescale_sigma=True)
            result[t] = np.clip(denoised * max_val, 0, max_val).astype(volume.dtype)

            if progress_cb and (t % 5 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result


@PluginBase.register
class TVDenoisePlugin(EnhancementPlugin):
    name = "TV Denoise"
    description = "Total Variation denoising — removes noise while preserving sharp edges"

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec("weight", "Denoising weight", "float", 0.1, 0.01, 1.0, 0.01,
                      tooltip="Regularization weight. Higher = more smoothing.\n"
                              "0.05-0.2 is typical for microscopy.\n"
                              "Too high will over-smooth."),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb=None) -> np.ndarray:
        from skimage.restoration import denoise_tv_chambolle

        weight = params.get("weight", 0.1)
        T = volume.shape[0]
        max_val = np.iinfo(volume.dtype).max if np.issubdtype(volume.dtype, np.integer) else 1.0
        result = np.zeros_like(volume)

        for t in range(T):
            f = volume[t].astype(np.float64) / max_val
            denoised = denoise_tv_chambolle(f, weight=weight)
            result[t] = np.clip(denoised * max_val, 0, max_val).astype(volume.dtype)
            if progress_cb and (t % 5 == 0):
                progress_cb(int((t + 1) / T * 100))
        return result
