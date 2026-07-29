"""histogram_threshold.py — vendored Histogram Threshold Segmenter kernel.

PURPOSE
    Single-frame, histogram-driven intensity-threshold segmentation of a 2-D
    image. Given an already-prepared 2-D integer frame, it thresholds it (one
    of 4 methods x 4 directions), cleans the binary mask with morphology,
    labels connected components, and measures per-region properties. Returns a
    SegmentationResult (mask + labels + regions + histogram + provenance).

WHERE THE REAL MATH LIVES
    In-repo. The entire algorithm is native to this repository — there is NO
    external analysis package behind it. It is built on top of stock
    scikit-image (skimage.filters / .morphology / .measure / .segmentation) and
    scipy.ndimage primitives; the orchestration, thresholding directions,
    hysteresis inversion trick, percentile->intensity mapping, and region
    measurement are all repo code.

PROVENANCE (branch: Version-1.45)
    Vendored verbatim (concatenated, byte-for-byte) from:
      nd2studios/backend/analysis/histothresh/config.py
      nd2studios/backend/analysis/histothresh/histogram.py
      nd2studios/backend/analysis/histothresh/thresholds.py
      nd2studios/backend/analysis/histothresh/morphology.py
      nd2studios/backend/analysis/histothresh/validation.py
      nd2studios/backend/analysis/histothresh/identifier.py
    The package __init__.py re-export shim was intentionally NOT vendored.
    NOT vendored (registry / prep / looping — caller's job):
      histogram_threshold_pipeline.py, plane_runner.py, source_utils.py.

VENDORED VERBATIM; IMPORTS NOTHING FROM nd2studios; CALLER OWNS ALL PREP.
    (no file I/O, no per-multipoint/per-timepoint looping, no
    crop/downsample/registration/exclusion — those are the caller's job.)

EDITS APPLIED (only what the vendoring rules permit)
    (a) Stripped all intra-package `from .validation/.histogram/.config/...`
        relative imports; every referenced symbol is now DEFINED in this file.
    (a) Collapsed the six per-file `from __future__ import annotations` lines
        into the single one below (a future-import must be the first statement
        of a module; duplicates deeper in the file are a SyntaxError).
    (b) BUG FIX 2026-07-28 in `threshold_hysteresis` (the ONE semantic deviation
        from the v1 source): its seeds are now offset by one count so the
        skimage `>` comparison implements the `>=`/`<=` contract its own
        docstring specifies. Without it, hysteresis was exclusive while every
        other method (single / percentile / relative, all of which route through
        `threshold_single`) was inclusive, so an object plateau sitting exactly
        at `strict` produced an EMPTY mask. v1 has the same defect upstream —
        see the note on the function.
    No private helpers were renamed (no name collisions across the 6 files).

DROPPED UI/REGISTRY-ONLY MEMBERS
    None. These six source files contain no get_params()/ParamSpec, no
    @Registry.register decorators, and no ParamEditor hooks — they were already
    the pure compute core. Everything here is on the compute path and is kept.
    (The registry/pipeline wrapper that DID carry those lived in the
    not-vendored histogram_threshold_pipeline.py.)

CONVENIENCE
    Construct a ThresholdConfig directly and pass it to
    HistogramThresholdSegmenter(config).run(image, ...). A headless helper,
    make_config(**kwargs), is provided at the bottom of this file; it forces
    bit_depth_strict=False by default (see the .md GOTCHAS — the dataclass
    default is strict=True, which raises BitDepthError on rescaled TIFFs).
"""
from __future__ import annotations

# ======================================================================
# vendored from: nd2studios/backend/analysis/histothresh/config.py
# ======================================================================

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ThresholdConfig:
    """Configuration for a single segmentation run.

    All intensity values are in raw integer counts in [0, 2**bit_depth - 1].
    """

    method: Literal["single", "hysteresis", "percentile", "relative"] = "hysteresis"
    direction: Literal["below", "above", "between", "outside"] = "below"

    # Single / hysteresis (raw integer counts)
    low: int | None = None
    high: int | None = None
    strict: int | None = None
    permissive: int | None = None

    # Percentile (0–100)
    percentile_low: float | None = None
    percentile_high: float | None = None
    sanity_floor: int | None = None
    sanity_ceiling: int | None = None

    # Relative (fraction of reference median)
    fraction_low: float | None = None
    fraction_high: float | None = None

    # Bit depth
    bit_depth: int = 12
    bit_depth_strict: bool = True

    # Spatial constraints
    min_area: int = 100
    max_area: int = 0
    opening_radius: int = 1
    closing_radius: int = 2
    min_hole_size: int = 50

    # Optional homogeneity gate
    homogeneity_gate: bool = False
    homogeneity_window: int = 7
    homogeneity_std_max: float = 20.0

    def __post_init__(self) -> None:
        if self.method == "single":
            if self.direction == "below" and self.low is None:
                raise ValueError("method=single, direction=below requires `low`")
            if self.direction == "above" and self.high is None:
                raise ValueError("method=single, direction=above requires `high`")
            if self.direction in ("between", "outside"):
                if self.low is None or self.high is None:
                    raise ValueError(
                        f"method=single, direction={self.direction} requires both `low` and `high`"
                    )
        elif self.method == "hysteresis":
            if self.direction not in ("below", "above"):
                raise ValueError(
                    f"hysteresis only supports below/above, got {self.direction}"
                )
            if self.strict is None or self.permissive is None:
                raise ValueError("method=hysteresis requires `strict` and `permissive`")
        elif self.method == "percentile":
            if self.direction == "below" and self.percentile_low is None:
                raise ValueError(
                    "method=percentile, direction=below requires `percentile_low`"
                )
            if self.direction == "above" and self.percentile_high is None:
                raise ValueError(
                    "method=percentile, direction=above requires `percentile_high`"
                )
            if self.direction in ("between", "outside"):
                if self.percentile_low is None or self.percentile_high is None:
                    raise ValueError(
                        f"method=percentile, direction={self.direction} requires both percentiles"
                    )
        elif self.method == "relative":
            if self.direction == "below" and self.fraction_low is None:
                raise ValueError(
                    "method=relative, direction=below requires `fraction_low`"
                )
            if self.direction == "above" and self.fraction_high is None:
                raise ValueError(
                    "method=relative, direction=above requires `fraction_high`"
                )


# ======================================================================
# vendored from: nd2studios/backend/analysis/histothresh/histogram.py
# ======================================================================

from dataclasses import dataclass
import numpy as np
from skimage import filters



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


# ======================================================================
# vendored from: nd2studios/backend/analysis/histothresh/thresholds.py
# ======================================================================

from typing import Literal
import numpy as np
from skimage.filters import apply_hysteresis_threshold


Direction = Literal["below", "above", "between", "outside"]


def threshold_single(
    image: np.ndarray,
    direction: Direction,
    *,
    low: int | None = None,
    high: int | None = None,
) -> np.ndarray:
    """Apply a single (non-hysteretic) threshold.

    Parameter requirements by direction:
      below   : low      → mask = image <= low
      above   : high     → mask = image >= high
      between : low,high → mask = low <= image <= high
      outside : low,high → mask = (image < low) | (image > high)
    """
    if direction == "below":
        if low is None:
            raise ValueError("`below` requires `low`")
        return image <= low
    if direction == "above":
        if high is None:
            raise ValueError("`above` requires `high`")
        return image >= high
    if direction == "between":
        if low is None or high is None:
            raise ValueError("`between` requires both `low` and `high`")
        if high < low:
            raise ValueError(f"`high` ({high}) must be >= `low` ({low})")
        return (image >= low) & (image <= high)
    if direction == "outside":
        if low is None or high is None:
            raise ValueError("`outside` requires both `low` and `high`")
        if high < low:
            raise ValueError(f"`high` ({high}) must be >= `low` ({low})")
        return (image < low) | (image > high)
    raise ValueError(f"Unknown direction: {direction!r}")


def threshold_hysteresis(
    image: np.ndarray,
    direction: Literal["below", "above"],
    *,
    strict: int,
    permissive: int,
) -> np.ndarray:
    """Hysteresis threshold for unidirectional cuts.

    For `below`: strict <= permissive. Pixels are included if they are
    <= permissive AND connected to a region of pixels <= strict.
    For `above`: strict >= permissive. Pixels are included if they are
    >= permissive AND connected to a region of pixels >= strict.

    scikit-image's apply_hysteresis_threshold uses (low, high) where
    low <= high and selects pixels above `low` connected to pixels above `high`.
    We adapt for both directions.

    **EDIT (2026-07-28, deviation from verbatim vendoring — bug fix).** skimage compares
    STRICTLY (``mask_low = image > low``; ``mask_high = image > high``), so passing the
    seeds through unchanged made this function EXCLUSIVE — contradicting the ``>=``/``<=``
    contract in the paragraph above and disagreeing with :func:`threshold_single` (and
    therefore with the percentile/relative methods, which resolve to an integer and
    dispatch to it). The visible symptom: on integer data whose objects sit exactly AT the
    core value — a dim 12-bit frame plateauing at 192, or a saturated one at 4095 —
    ``strict=192`` seeded NOTHING and hysteresis returned an empty mask where percentile
    at the same resolved threshold returned every object. Since this function's input is
    guaranteed integer (``validate_bit_depth`` rejects float dtype), offsetting each seed
    by one count makes ``>`` exactly ``>=`` and restores the documented semantics.
    """
    if direction == "below":
        if strict > permissive:
            raise ValueError(
                f"For `below`, strict ({strict}) must be <= permissive ({permissive})"
            )
        # int() before the subtraction: `image.max()` is an unsigned numpy scalar, and
        # M - permissive - 1 can go negative (permissive == M selects everything).
        m = int(image.max())
        inverted = m - image.astype(np.int64)
        return apply_hysteresis_threshold(
            inverted,
            low=m - int(permissive) - 1,
            high=m - int(strict) - 1,
        )
    if direction == "above":
        if strict < permissive:
            raise ValueError(
                f"For `above`, strict ({strict}) must be >= permissive ({permissive})"
            )
        return apply_hysteresis_threshold(
            image, low=int(permissive) - 1, high=int(strict) - 1)
    raise ValueError(f"Hysteresis only supports 'below' or 'above', got {direction!r}")


def threshold_percentile(
    image: np.ndarray,
    direction: Direction,
    *,
    bit_depth: int = 12,
    percentile_low: float | None = None,
    percentile_high: float | None = None,
    sanity_floor: int | None = None,
    sanity_ceiling: int | None = None,
    hist: "Histogram | None" = None,
) -> np.ndarray:
    """Threshold by percentile of the image's histogram.

    Maps percentiles to integer intensity values, then dispatches to
    `threshold_single`. `sanity_floor` clips the resolved `low` value to
    prevent calling bright pixels "dark" on images without genuine dark regions.
    `sanity_ceiling` is the symmetric guard for the bright end.

    Pass a pre-computed ``hist`` to avoid a second full-frame bincount when
    the caller has already built the histogram (e.g. :class:`HistogramThresholdSegmenter`).
    """
    if hist is None:
        hist = compute_histogram(image, bit_depth=bit_depth)
    low = hist.percentile(percentile_low) if percentile_low is not None else None
    high = hist.percentile(percentile_high) if percentile_high is not None else None

    if low is not None and sanity_floor is not None:
        low = min(low, sanity_floor)
    if high is not None and sanity_ceiling is not None:
        high = max(high, sanity_ceiling)

    return threshold_single(image, direction, low=low, high=high)


def threshold_relative(
    image: np.ndarray,
    direction: Direction,
    *,
    reference_mask: np.ndarray | None = None,
    fraction_low: float | None = None,
    fraction_high: float | None = None,
) -> np.ndarray:
    """Threshold relative to the median intensity inside `reference_mask`.

    Resolves `low = median * fraction_low` and/or `high = median * fraction_high`,
    then dispatches to `threshold_single`.
    """
    if reference_mask is None:
        reference = image
    else:
        reference = image[reference_mask]
    median = float(np.median(reference))
    low = int(round(median * fraction_low)) if fraction_low is not None else None
    high = int(round(median * fraction_high)) if fraction_high is not None else None
    return threshold_single(image, direction, low=low, high=high)


# ======================================================================
# vendored from: nd2studios/backend/analysis/histothresh/morphology.py
# ======================================================================

import numpy as np
from skimage.measure import label as _label_components
from skimage.morphology import (
    opening,
    closing,
    disk,
    ball,
    remove_small_objects,
    remove_small_holes,
)
from scipy.ndimage import generic_filter


def apply_spatial_constraints(
    mask: np.ndarray,
    *,
    min_area: int = 100,
    max_area: int = 0,
    opening_radius: int = 1,
    closing_radius: int = 2,
    min_hole_size: int = 50,
    is_3d: bool = False,
) -> np.ndarray:
    """Apply opening → closing → hole-fill → small-object removal in that order.

    Opening first kills isolated noise pixels before closing would bridge them.
    Closing bridges legitimate small gaps. Hole filling repairs interior pockets.
    Small-object removal applies the size threshold last, then large-object removal
    strips background blobs that exceed max_area (0 = disabled).
    """
    footprint_fn = ball if is_3d else disk
    out = mask.copy()
    if opening_radius > 0:
        out = opening(out, footprint=footprint_fn(opening_radius))
    if closing_radius > 0:
        out = closing(out, footprint=footprint_fn(closing_radius))
    if min_hole_size > 0:
        # max_size removes holes <= value; subtract 1 to match old "< value" semantics
        out = remove_small_holes(out, max_size=max(0, min_hole_size - 1))
    if min_area > 0:
        out = remove_small_objects(out, max_size=max(0, min_area - 1))
    if max_area > 0:
        labeled = _label_components(out, connectivity=2)
        sizes = np.bincount(labeled.ravel())
        too_large = sizes > max_area
        too_large[0] = False  # label 0 is background — never remove it
        out = out & ~too_large[labeled]
    return out


def homogeneity_gate(
    image: np.ndarray,
    *,
    window: int = 7,
    std_max: float = 20.0,
    is_3d: bool = False,
) -> np.ndarray:
    """Return a boolean mask of pixels whose local standard deviation is below `std_max`.

    Use to suppress noisy/textured regions that pass intensity thresholds on average
    but aren't truly homogeneous. O(N * window^d) — disabled by default.
    """
    size = (window,) * (3 if is_3d else 2)
    local_std = generic_filter(image.astype(np.float32), np.std, size=size)
    return local_std < std_max


# ======================================================================
# vendored from: nd2studios/backend/analysis/histothresh/validation.py
# ======================================================================

import warnings
import numpy as np

BIT_DEPTH_MAX: dict[int, int] = {8: 255, 10: 1023, 12: 4095, 14: 16383, 16: 65535}


class BitDepthError(ValueError):
    """Raised when image data does not match the expected bit depth."""


def validate_bit_depth(
    image: np.ndarray,
    bit_depth: int = 12,
    *,
    strict: bool = True,
) -> None:
    """Validate that `image` contains values consistent with `bit_depth`.

    Raises BitDepthError for non-integer dtype, unsupported bit depth, or
    (when strict=True) values exceeding the expected maximum.
    """
    if bit_depth not in BIT_DEPTH_MAX:
        raise BitDepthError(
            f"Unsupported bit depth {bit_depth}. Supported: {sorted(BIT_DEPTH_MAX)}"
        )
    if not np.issubdtype(image.dtype, np.integer):
        raise BitDepthError(
            f"Expected integer dtype, got {image.dtype}. "
            "Float inputs must be explicitly converted; this tool operates on raw counts."
        )

    expected_max = BIT_DEPTH_MAX[bit_depth]
    actual_max = int(image.max())

    if actual_max > expected_max:
        msg = (
            f"Image max value {actual_max} exceeds expected max {expected_max} "
            f"for {bit_depth}-bit data. Data may have been rescaled to fill its "
            f"container ({image.dtype}). Thresholds set in {bit_depth}-bit units "
            "will be incorrect on this input."
        )
        if strict:
            raise BitDepthError(msg)
        warnings.warn(msg, stacklevel=2)

    if actual_max < expected_max * 0.05:
        warnings.warn(
            f"Image max {actual_max} is <5% of {bit_depth}-bit range. "
            "Verify the bit depth setting; image may be lower-bit than declared.",
            stacklevel=2,
        )


def infer_bit_depth(image: np.ndarray) -> int:
    """Best-effort bit-depth inference from observed value range.

    Use only as a fallback; explicit configuration is always preferred.
    """
    actual_max = int(image.max())
    for bd in sorted(BIT_DEPTH_MAX):
        if actual_max <= BIT_DEPTH_MAX[bd]:
            return bd
    raise BitDepthError(f"Image max {actual_max} exceeds all supported bit depths.")


# ======================================================================
# vendored from: nd2studios/backend/analysis/histothresh/identifier.py
# ======================================================================

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
from skimage.measure import label, regionprops_table
from skimage.segmentation import clear_border



def _frame_hash(image: np.ndarray) -> str:
    """Cheap identity tag for provenance — samples 768 pixels instead of copying
    the whole frame (a 268 M-pixel uint16 frame would be a 536 MB tobytes() copy)."""
    flat = image.ravel()
    n = len(flat)
    probe = np.concatenate([flat[:256], flat[n // 2: n // 2 + 256], flat[-256:]])
    return hashlib.sha256(
        f"{image.shape}:{image.dtype}:".encode() + probe.tobytes()
    ).hexdigest()[:16]


@dataclass
class SegmentationResult:
    """Output from a single-frame HistogramThresholdSegmenter.run() call."""

    mask: np.ndarray
    labels: np.ndarray
    regions: list[dict[str, Any]]
    histogram: Histogram
    threshold_used: dict[str, Any]
    config: ThresholdConfig
    provenance: dict[str, Any] = field(default_factory=dict)


class HistogramThresholdSegmenter:
    """Orchestrates histogram-driven threshold segmentation on a single 2-D frame."""

    def __init__(self, config: ThresholdConfig) -> None:
        self.config = config

    def run(
        self,
        image: np.ndarray,
        *,
        reference_mask: np.ndarray | None = None,
        voxel_size: tuple[float, ...] | None = None,
    ) -> SegmentationResult:
        cfg = self.config
        validate_bit_depth(image, cfg.bit_depth, strict=cfg.bit_depth_strict)

        # Only run the full bincount for methods that resolve their threshold
        # from the histogram.  For hysteresis / single / relative the histogram
        # is not needed — skipping it avoids a 2 GB int64 intermediate on large
        # stitched frames (268 M px × 8 B = ~2 GB) that would otherwise block
        # the worker thread for seconds with no effect on the mask.
        needs_hist = (cfg.method == "percentile")
        hist = compute_histogram(image, bit_depth=cfg.bit_depth) if needs_hist else None

        mask = self._threshold_one(image, hist, reference_mask)

        if cfg.homogeneity_gate:
            gate = homogeneity_gate(
                image,
                window=cfg.homogeneity_window,
                std_max=cfg.homogeneity_std_max,
                is_3d=False,
            )
            mask = mask & gate

        mask = apply_spatial_constraints(
            mask,
            min_area=cfg.min_area,
            max_area=cfg.max_area,
            opening_radius=cfg.opening_radius,
            closing_radius=cfg.closing_radius,
            min_hole_size=cfg.min_hole_size,
            is_3d=False,
        )

        labels = label(mask, connectivity=2).astype(np.int32)
        regions = _measure_regions(labels, image, voxel_size)

        return SegmentationResult(
            mask=mask,
            labels=labels,
            regions=regions,
            histogram=hist,
            threshold_used=self._resolved_thresholds(image, hist, reference_mask),
            config=cfg,
            provenance={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_shape": image.shape,
                "input_dtype": str(image.dtype),
                "input_hash": _frame_hash(image),
            },
        )

    def _threshold_one(
        self,
        image: np.ndarray,
        hist: Histogram,
        reference_mask: np.ndarray | None,
    ) -> np.ndarray:
        cfg = self.config
        if cfg.method == "single":
            return threshold_single(image, cfg.direction, low=cfg.low, high=cfg.high)
        if cfg.method == "hysteresis":
            assert cfg.direction in ("below", "above")
            assert cfg.strict is not None and cfg.permissive is not None
            return threshold_hysteresis(
                image,
                cfg.direction,
                strict=cfg.strict,
                permissive=cfg.permissive,
            )
        if cfg.method == "percentile":
            return threshold_percentile(
                image,
                cfg.direction,
                bit_depth=cfg.bit_depth,
                percentile_low=cfg.percentile_low,
                percentile_high=cfg.percentile_high,
                sanity_floor=cfg.sanity_floor,
                sanity_ceiling=cfg.sanity_ceiling,
                hist=hist,
            )
        if cfg.method == "relative":
            return threshold_relative(
                image,
                cfg.direction,
                reference_mask=reference_mask,
                fraction_low=cfg.fraction_low,
                fraction_high=cfg.fraction_high,
            )
        raise ValueError(f"Unknown method {cfg.method}")

    def _resolved_thresholds(
        self,
        image: np.ndarray,
        hist: Histogram,
        reference_mask: np.ndarray | None,
    ) -> dict[str, Any]:
        cfg = self.config
        if cfg.method == "percentile":
            return {
                "method": "percentile",
                "low_resolved": hist.percentile(cfg.percentile_low) if cfg.percentile_low is not None else None,
                "high_resolved": hist.percentile(cfg.percentile_high) if cfg.percentile_high is not None else None,
                "percentile_low": cfg.percentile_low,
                "percentile_high": cfg.percentile_high,
            }
        if cfg.method == "relative":
            ref = image if reference_mask is None else image[reference_mask]
            median = float(np.median(ref))
            return {
                "method": "relative",
                "reference_median": median,
                "low_resolved": int(median * cfg.fraction_low) if cfg.fraction_low else None,
                "high_resolved": int(median * cfg.fraction_high) if cfg.fraction_high else None,
            }
        if cfg.method == "hysteresis":
            return {
                "method": "hysteresis",
                "strict": cfg.strict,
                "permissive": cfg.permissive,
            }
        return {"method": "single", "low": cfg.low, "high": cfg.high}


def _measure_regions(
    labels: np.ndarray,
    image: np.ndarray,
    voxel_size: tuple[float, ...] | None,
) -> list[dict[str, Any]]:
    if labels.max() == 0:
        return []

    properties = (
        "label",
        "area",
        "centroid",
        "mean_intensity",
        "min_intensity",
        "max_intensity",
    )
    table = regionprops_table(labels, intensity_image=image, properties=properties)

    n = len(table["label"])
    voxel_area = float(np.prod(voxel_size)) if voxel_size is not None else None

    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {
            "label_id": int(table["label"][i]),
            "area_px": int(table["area"][i]),
            "centroid_y": float(table["centroid-0"][i]),
            "centroid_x": float(table["centroid-1"][i]),
            "mean_intensity": float(table["mean_intensity"][i]),
            "min_intensity": float(table["min_intensity"][i]),
            "max_intensity": float(table["max_intensity"][i]),
        }
        if voxel_area is not None:
            row["area_um2"] = row["area_px"] * voxel_area
        rows.append(row)

    return rows


# ======================================================================
# convenience helper (NOT vendored — added for headless integration)
# ======================================================================
def make_config(*, bit_depth_strict: bool = False, **kwargs) -> "ThresholdConfig":
    """Construct a :class:`ThresholdConfig` for headless use.

    Thin wrapper over ``ThresholdConfig(**kwargs)`` that flips the
    ``bit_depth_strict`` default from True (the dataclass default) to False, so
    a rescaled/normalized TIFF whose max exceeds the declared bit-depth ceiling
    warns instead of raising :class:`BitDepthError`. Pass any ThresholdConfig
    field as a keyword (method, direction, low, high, strict, permissive,
    percentile_low, ..., min_area, max_area, opening_radius, closing_radius,
    min_hole_size, bit_depth, ...). Set ``bit_depth_strict=True`` to restore the
    strict behavior.
    """
    return ThresholdConfig(bit_depth_strict=bit_depth_strict, **kwargs)
