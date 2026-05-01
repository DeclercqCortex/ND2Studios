"""
Frame-mean normalization for timeseries intensity correction.
Refactored from workflow.py.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import tifffile


def compute_frame_means(source: Union[str, np.ndarray]) -> np.ndarray:
    """
    Compute mean intensity per frame.

    Parameters
    ----------
    source : str (TIFF path) or np.ndarray (T, H, W)

    Returns
    -------
    np.ndarray of shape (T,) with per-frame means
    """
    if isinstance(source, str):
        means = []
        with tifffile.TiffFile(source) as tif:
            for page in tif.pages:
                frame = page.asarray()
                means.append(float(frame.mean()))
        return np.array(means)
    else:
        return np.array([float(source[t].mean()) for t in range(source.shape[0])])


def normalize_frame(
    frame: np.ndarray,
    frame_mean: float,
    grand_mean: float,
) -> np.ndarray:
    """
    Normalize a single frame by its mean, scaled to the grand mean.

    result = (frame / frame_mean) * grand_mean
    """
    if frame_mean <= 0:
        return frame
    normalized = (frame.astype(np.float32) / frame_mean) * grand_mean
    return np.clip(normalized, 0, 65535).astype(np.uint16)


def normalize_timeseries(
    timeseries: np.ndarray,
    frame_means: Optional[np.ndarray] = None,
    progress_cb=None,
) -> np.ndarray:
    """
    Normalize an entire timeseries (T, H, W) by per-frame means.

    Parameters
    ----------
    timeseries : np.ndarray (T, H, W)
    frame_means : np.ndarray (T,), precomputed. If None, computed from data.
    progress_cb : callable(int), progress 0-100

    Returns
    -------
    np.ndarray (T, H, W), same dtype as input
    """
    T = timeseries.shape[0]
    if frame_means is None:
        frame_means = compute_frame_means(timeseries)

    grand_mean = float(frame_means.mean())
    result = np.zeros_like(timeseries)

    for t in range(T):
        result[t] = normalize_frame(timeseries[t], frame_means[t], grand_mean)
        if progress_cb and (t % 10 == 0 or t == T - 1):
            progress_cb(int((t + 1) / T * 100))

    return result


def load_normalized_frame(
    tiff_path: str,
    frame_idx: int,
    frame_means: np.ndarray,
    grand_mean: float,
) -> np.ndarray:
    """Load one frame from disk and normalize it."""
    raw = tifffile.imread(tiff_path, key=frame_idx).astype(np.float32)
    fm = frame_means[frame_idx]
    if fm > 0:
        normalized = (raw / fm) * grand_mean
    else:
        normalized = raw
    return np.clip(normalized, 0, 65535).astype(np.uint16)
