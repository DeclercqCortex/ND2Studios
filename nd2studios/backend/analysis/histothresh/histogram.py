from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from skimage import filters

from .validation import BIT_DEPTH_MAX


@dataclass
class Histogram:
    """A full-LUT-range intensity histogram.

    Attributes
    ----------
    counts : np.ndarray
        Bin counts, length = bit_depth_max + 1 (one bin per integer value).
    bin_edges : np.ndarray
        Integer bin edges, length = bit_depth_max + 2.
    bit_depth : int
    total_pixels : int
    """

    counts: np.ndarray
    bin_edges: np.ndarray
    bit_depth: int
    total_pixels: int

    @property
    def values(self) -> np.ndarray:
        """Integer pixel values 0..bit_depth_max corresponding to each bin."""
        return np.arange(BIT_DEPTH_MAX[self.bit_depth] + 1)

    def cumulative(self) -> np.ndarray:
        """Cumulative distribution, length = bit_depth_max + 1, range [0, 1]."""
        return np.cumsum(self.counts) / self.total_pixels

    def percentile(self, p: float) -> int:
        """Return the integer pixel value at percentile `p` (0–100)."""
        if not 0 <= p <= 100:
            raise ValueError(f"Percentile must be in [0, 100], got {p}")
        cdf = self.cumulative()
        idx = int(np.searchsorted(cdf, p / 100.0))
        return int(min(idx, len(cdf) - 1))


def compute_histogram(
    image: np.ndarray,
    bit_depth: int = 12,
    mask: np.ndarray | None = None,
) -> Histogram:
    """Compute a histogram across the full LUT range [0, 2**bit_depth - 1].

    Bins every integer value as its own bin (no downsampling). For 12-bit
    this gives 4096 bins, which is fast and lossless.
    """
    max_val = BIT_DEPTH_MAX[bit_depth]
    flat = image[mask] if mask is not None else image.ravel()
    counts = np.bincount(flat.astype(np.int64), minlength=max_val + 1)
    counts = counts[: max_val + 1]
    edges = np.arange(max_val + 2)
    return Histogram(
        counts=counts,
        bin_edges=edges,
        bit_depth=bit_depth,
        total_pixels=int(flat.size),
    )


def suggest_threshold_otsu(hist: Histogram) -> int:
    """Otsu's method computed directly on histogram counts."""
    return int(filters.threshold_otsu(hist=(hist.counts, hist.values)))


def suggest_threshold_triangle(hist: Histogram) -> int:
    """Triangle method — preferred when the histogram is heavily skewed."""
    return int(filters.threshold_triangle(hist=(hist.counts, hist.values)))


def suggest_threshold_minimum(hist: Histogram) -> int:
    """Minimum method — find the valley between two histogram peaks."""
    return int(filters.threshold_minimum(hist=(hist.counts, hist.values)))


def suggest_thresholds_multi_otsu(hist: Histogram, classes: int = 3) -> list[int]:
    """Multi-Otsu — useful for finding both a dark and a saturated cutoff at once."""
    thresholds = filters.threshold_multiotsu(
        hist=(hist.counts, hist.values), classes=classes
    )
    return [int(t) for t in thresholds]
