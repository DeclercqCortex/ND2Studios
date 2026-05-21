from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from skimage.draw import disk as draw_disk
from skimage.measure import label as sk_label, regionprops

from nd2studios.backend.analysis.histothresh.histogram import compute_histogram

from .config import SpotsConfig
from .detection import find_extrema
from .grow import grow_seeds_dilation, grow_seeds_watershed
from .scale_space import dog_response, log_response, resolve_sigma
from .symmetry import circularity, filter_by_symmetry
from .validation import validate_bit_depth, validate_diameter


@dataclass
class SpotsResult:
    """Output from a single-frame BrightDarkSpotsSegmenter.run() call."""

    mask: np.ndarray              # bool (H, W)
    labels: np.ndarray            # int32 (H, W), 0 = background
    centroids: np.ndarray         # (N, 2) float — (row, col) per surviving spot
    diameters_px: np.ndarray      # (N,) float — configured typical diameter
    contrast_scores: np.ndarray   # (N,) float — normalised DoG/LoG magnitude
    circularity_scores: np.ndarray  # (N,) float
    regions: list[dict[str, Any]]
    config: SpotsConfig
    provenance: dict[str, Any] = field(default_factory=dict)
    background_labels: np.ndarray | None = None   # int32 (H,W), 1=background, None when disabled


class BrightDarkSpotsSegmenter:
    """Single-2D-frame spot detector. The pipeline adapter loops over T."""

    def __init__(self, config: SpotsConfig) -> None:
        self.config = config

    def run(
        self,
        image: np.ndarray,
        *,
        pixel_size_um: float | None = None,
        boundary_mask: np.ndarray | None = None,
    ) -> SpotsResult:
        cfg = self.config

        # 1. Validate inputs
        validate_bit_depth(image, cfg.bit_depth, strict=cfg.bit_depth_strict)
        validate_diameter(cfg.typical_diameter_um)

        # 2. Convert diameter to pixels (fallback: 1 µm/px)
        px_um = pixel_size_um if (pixel_size_um and pixel_size_um > 0) else 1.0
        d_px = cfg.typical_diameter_um / px_um

        # 3. Resolve sigma
        sigma_in, sigma_out = resolve_sigma(d_px)

        # 4. Compute band-pass response
        if cfg.kernel == "dog":
            resp = dog_response(image, sigma_in, sigma_out)
        else:
            resp = log_response(image, sigma_in)

        # 5. Normalise response to [-1, 1]
        resp_max = float(np.abs(resp).max())
        resp = resp / max(resp_max, 1e-8)

        # 6. Build LUT-histogram intensity gate mask
        intensity_mask: np.ndarray | None = None
        if cfg.intensity_percentile is not None:
            hist = compute_histogram(image, bit_depth=cfg.bit_depth)
            gate_val = hist.percentile(cfg.intensity_percentile)
            if cfg.polarity == "bright":
                intensity_mask = image >= gate_val
            else:
                intensity_mask = image <= gate_val

        # 7. Localise extrema and record raw contrast scores
        min_dist = max(1, int(d_px / 2))
        centroids = find_extrema(
            resp,
            polarity=cfg.polarity,
            min_distance=min_dist,
            contrast=cfg.contrast,
            intensity_mask=intensity_mask,
        )

        if len(centroids) > 0:
            raw_contrast = np.abs(resp[centroids[:, 0], centroids[:, 1]])
        else:
            raw_contrast = np.zeros(0, dtype=np.float32)

        # 8. Build label map from centroids
        H, W = image.shape[:2]

        if len(centroids) == 0:
            label_map = np.zeros((H, W), dtype=np.int32)
        elif cfg.output_mode == "circular":
            label_map = _rasterise_disks(centroids, d_px / 2.0, H, W)
        else:
            # Region mode: point seeds → watershed expansion bounded by intensity gate
            seed_labels = np.zeros((H, W), dtype=np.int32)
            for i, (r, c) in enumerate(centroids):
                seed_labels[int(r), int(c)] = i + 1
            # Use intensity_mask as boundary so watershed doesn't fill the whole frame
            ws_boundary = boundary_mask if boundary_mask is not None else intensity_mask
            label_map = grow_seeds_watershed(
                image, seed_labels, polarity=cfg.polarity, boundary_mask=ws_boundary
            )

        # 9. Symmetry gate — operate on the mask of the label map
        mask = filter_by_symmetry(label_map > 0, setting=cfg.symmetry)
        label_map = label_map * mask

        # 10. Optional grow step (post-symmetry)
        if cfg.grow_method != "none" and cfg.grow_radius_um > 0:
            grow_px = max(1, round(cfg.grow_radius_um / px_um))
            if cfg.grow_method == "dilation":
                mask = grow_seeds_dilation(label_map > 0, grow_px)
                label_map = sk_label(mask, connectivity=2).astype(np.int32)
            else:
                seed_labels = label_map.copy()
                label_map = grow_seeds_watershed(
                    image, seed_labels, polarity=cfg.polarity,
                    boundary_mask=boundary_mask,
                )
            mask = label_map > 0

        # 11. Final clean label and regionprops
        labels = sk_label(label_map > 0, connectivity=2).astype(np.int32)
        voxel_area = px_um ** 2 if (pixel_size_um and pixel_size_um > 0) else None
        regions = _measure_regions(
            labels, image, voxel_area,
            centroids=centroids,
            raw_contrast=raw_contrast,
            d_px=d_px,
        )

        # Collect per-spot arrays (one entry per surviving label)
        if regions:
            out_centroids = np.array(
                [[r["centroid_y"], r["centroid_x"]] for r in regions], dtype=float
            )
            out_diameters = np.array([r["diameter_px"] for r in regions], dtype=float)
            out_contrast = np.array([r["contrast_score"] for r in regions], dtype=float)
            out_circ = np.array([r["circularity"] for r in regions], dtype=float)
        else:
            out_centroids = np.zeros((0, 2), dtype=float)
            out_diameters = np.zeros(0, dtype=float)
            out_contrast = np.zeros(0, dtype=float)
            out_circ = np.zeros(0, dtype=float)

        # 12. Background mask (inverse of detected spots)
        background_labels: np.ndarray | None = None
        if cfg.background_mask:
            background_labels = (~(labels > 0)).astype(np.int32)

        # 13. Provenance
        provenance: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_shape": image.shape,
            "input_dtype": str(image.dtype),
            "input_hash": hashlib.sha256(image.tobytes()).hexdigest()[:16],
        }

        return SpotsResult(
            mask=labels > 0,
            labels=labels,
            centroids=out_centroids,
            diameters_px=out_diameters,
            contrast_scores=out_contrast,
            circularity_scores=out_circ,
            regions=regions,
            config=cfg,
            provenance=provenance,
            background_labels=background_labels,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rasterise_disks(
    centroids: np.ndarray, radius_px: float, H: int, W: int
) -> np.ndarray:
    """Draw filled disks of given radius at each centroid; returns int32 label array."""
    out = np.zeros((H, W), dtype=np.int32)
    r_int = max(1, int(round(radius_px)))
    for i, (r, c) in enumerate(centroids):
        rr, cc = draw_disk((int(r), int(c)), r_int, shape=(H, W))
        out[rr, cc] = i + 1
    return out


def _measure_regions(
    labels: np.ndarray,
    image: np.ndarray,
    voxel_area: float | None,
    *,
    centroids: np.ndarray,
    raw_contrast: np.ndarray,
    d_px: float,
) -> list[dict[str, Any]]:
    """Measure each label region; attach contrast_score and circularity."""
    if labels.max() == 0:
        return []

    rows: list[dict[str, Any]] = []
    for prop in regionprops(labels, intensity_image=image.astype(np.float32)):
        cy, cx = prop.centroid

        # Map region centroid to nearest detection centroid to retrieve contrast score.
        if len(centroids) > 0:
            dists = np.hypot(centroids[:, 0] - cy, centroids[:, 1] - cx)
            cs = float(raw_contrast[int(np.argmin(dists))])
        else:
            cs = 0.0

        circ = circularity(float(prop.area), float(prop.perimeter))

        row: dict[str, Any] = {
            "label_id": int(prop.label),
            "area_px": int(prop.area),
            "area_um2": float(prop.area) * voxel_area if voxel_area is not None else 0.0,
            "centroid_y": float(cy),
            "centroid_x": float(cx),
            "mean_intensity": float(
                prop.intensity_mean if hasattr(prop, "intensity_mean") else prop.mean_intensity
            ),
            "diameter_px": float(d_px),
            "contrast_score": cs,
            "circularity": circ,
        }
        rows.append(row)

    return rows
