"""registration.py — self-contained image-registration (drift-correction) kernel.

PURPOSE
-------
Estimate per-frame spatial transforms that align each frame of a ``(T, H, W)``
time series onto a common anchor, then resample any channel through those
transforms ("register once on a reference channel, apply to all channels").
Supports pure sub-pixel translation (phase correlation), rigid/affine (ECC),
and large-motion feature alignment (ORB + RANSAC).

WHERE THE REAL MATH LIVES
-------------------------
IN-REPO. The algorithm is plain Python over numpy + established library
primitives — there is no external math package. It calls into:
  * scipy.ndimage       — sub-pixel resampling (``shift``, ``gaussian_filter``)
  * skimage.registration — ``phase_cross_correlation`` (sub-pixel translation)
  * skimage.feature/measure/transform — ORB, ``match_descriptors``, ``ransac``
  * cv2 (OpenCV)        — ``findTransformECC``, ``warpAffine``/``warpPerspective``
The windowing / high-pass / NCC scoring primitives (``_hann2d``, ``_highpass``,
``_ncc``) are the same ones the stitcher uses.

PROVENANCE (branch: Version-1.45)
---------------------------------
  * nd2studios/backend/registration/estimate.py   — the whole kernel body
  * nd2studios/backend/stitch/register.py          — helpers _highpass, _hann2d,
        _ncc (copied verbatim; the stitch module was NOT imported because its
        top level drags in StitchConfig / Dataset)
  * nd2studios/backend/analysis/manual_mask.py     — rasterize_shapes + its
        private helper _rasterize (copied verbatim; needed for the freeform
        "shapes" ROI branch of roi_to_mask)

Vendored verbatim; imports nothing from nd2studios; caller owns all prep
(no file I/O, no per-multipoint / per-timepoint looping, no crop / downsample /
registration-source selection / exclusion — the caller supplies a prepared
(T, H, W) single-channel array).

DROPPED UI / REGISTRY-ONLY MEMBERS
----------------------------------
  * From manual_mask.py: the ManualMaskPipeline class, its get_params()
    ParamSpec list, the @AnalysisPipeline.register decorator, and every helper
    that is NOT on the rasterize_shapes -> _rasterize compute path
    (generate_tiled_label_boxes, generate_tiled_mask, _normalise_frame_slot,
    _ordered_shapes, _voxel_counts_for_frame, shape_to_editable_polygon,
    expand_polygon_uniformly, _resample_closed_polygon). None of these are
    reachable from this kernel's entry points.
  * From register.py: everything except the three primitives above
    (_overlap_boxes, _pair_shift, _candidate_pairs, _solve_axis,
    refine_positions) — those are stitcher-only, off this kernel's path.
  * method.py / RegistrationResult were intentionally NOT vendored; callers use
    estimate_series / apply_series / apply_frame directly.

EDITS MADE (per the extraction rules; see registration.md "issues")
-------------------------------------------------------------------
  (a) Removed ``from nd2studios.backend.stitch.register import ...`` and
      ``from nd2studios.backend.analysis.manual_mask import rasterize_shapes``;
      both symbols are now defined locally in this file.
  No private-helper renames were needed (no name collisions between the three
  concatenated sources).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from skimage.draw import ellipse as sk_ellipse
from skimage.draw import polygon as sk_polygon


# ============================================================================
# Vendored verbatim from nd2studios/backend/stitch/register.py
# (helpers _highpass, _hann2d, _ncc — copied byte-for-byte)
# ============================================================================
def _highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    img = img.astype(np.float32)
    if sigma and sigma > 0:
        from scipy.ndimage import gaussian_filter
        img = img - gaussian_filter(img, sigma=sigma)
    return img


def _hann2d(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    return np.outer(wy, wx).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    da = float(np.sqrt((a * a).sum())); db = float(np.sqrt((b * b).sum()))
    if da < 1e-9 or db < 1e-9:
        return 0.0
    return float((a * b).sum() / (da * db))


# ============================================================================
# Vendored verbatim from nd2studios/backend/analysis/manual_mask.py
# (rasterize_shapes + its private helper _rasterize — copied byte-for-byte;
#  needed for the freeform "shapes" ROI branch of roi_to_mask)
# ============================================================================
def rasterize_shapes(
    frame: np.ndarray,
    shapes: List[Dict[str, Any]],
    H: int,
    W: int,
    label_offset: int = 0,
) -> None:
    """Paint each shape into ``frame`` with label_id (offset+1)..(offset+N) in place.

    ``label_offset`` lets callers reserve lower IDs for tiled masks so drawn
    shapes never collide with tiled-strip label IDs.
    """
    for rel_id, shape in enumerate(shapes, start=1):
        _rasterize(frame, shape, label_offset + rel_id, H, W)


def _rasterize(frame: np.ndarray, shape: Dict[str, Any],
               label_id: int, H: int, W: int) -> None:
    """Paint a single shape into a frame at the given label id (in place)."""
    s_type = shape.get("type")
    raw_verts = shape.get("vertices") or []
    if not raw_verts:
        return
    verts = np.asarray(raw_verts, dtype=float)

    if s_type == "rect" and verts.shape[0] == 2:
        y0, y1 = sorted([verts[0, 0], verts[1, 0]])
        x0, x1 = sorted([verts[0, 1], verts[1, 1]])
        iy0 = max(0, int(round(y0)))
        ix0 = max(0, int(round(x0)))
        iy1 = min(H, int(round(y1)) + 1)
        ix1 = min(W, int(round(x1)) + 1)
        if iy1 > iy0 and ix1 > ix0:
            frame[iy0:iy1, ix0:ix1] = label_id
        return

    if s_type == "ellipse" and verts.shape[0] == 2:
        cy = (verts[0, 0] + verts[1, 0]) / 2.0
        cx = (verts[0, 1] + verts[1, 1]) / 2.0
        ry = abs(verts[1, 0] - verts[0, 0]) / 2.0
        rx = abs(verts[1, 1] - verts[0, 1]) / 2.0
        if ry <= 0 or rx <= 0:
            return
        rr, cc = sk_ellipse(cy, cx, ry, rx, shape=(H, W))
        frame[rr, cc] = label_id
        return

    if s_type == "polygon" and verts.shape[0] >= 3:
        rr, cc = sk_polygon(verts[:, 0], verts[:, 1], shape=(H, W))
        frame[rr, cc] = label_id
        return


# ============================================================================
# Vendored verbatim from nd2studios/backend/registration/estimate.py
# (module body; the two nd2studios imports are removed — those symbols are
#  defined above in this file)
# ============================================================================
# Reference-frame selection modes. ``template`` is a robust two-pass anchor (see
# :func:`estimate_series`) recommended for long / photobleaching series.
REFERENCE_MODES: Tuple[str, ...] = ("first", "previous", "mean", "template")
# Transform models (increasing DOF). ``feature`` is ORB+RANSAC (large motion).
MODELS: Tuple[str, ...] = ("translation", "euclidean", "affine", "feature")


# ── dtype helpers ────────────────────────────────────────────────────────────

def _cast_like(arr_float: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast a float result back to ``dtype``, clipping integer ranges to avoid
    wrap-around from interpolation over/undershoot."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.round(arr_float), info.min, info.max).astype(dtype)
    return arr_float.astype(dtype, copy=False)


# ── translation (phase correlation) ──────────────────────────────────────────

def estimate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    upsample: int = 20,
    highpass_sigma: float = 2.0,
    window: bool = True,
    mask: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, float]:
    """Estimate the sub-pixel translation aligning ``moving`` onto ``reference``.

    Band-pass (high-pass) whitening + Hann windowing suppress DC and spectral
    leakage before phase correlation (``normalization="phase"`` for illumination
    robustness); ``disambiguate=True`` resolves the modulo-image wrap so large
    shifts are recovered without aliasing. Returns ``(shift (row, col), ncc)``
    where ``ncc`` is the normalized cross-correlation of the aligned overlap — a
    confidence in ``[-1, 1]`` for gating near-blank / failed frames.

    Region of interest (estimate on a sub-region, apply full-frame):
    - ``bbox=(y0, y1, x0, x1)`` crops both images to that box first — **subpixel**,
      used for a rectangular ROI; a pure shift is invariant to the crop origin.
    - ``mask`` (bool ``(H,W)``) restricts a freeform region via masked phase
      correlation (Padfield); note masked correlation is **integer-pixel** only.
    """
    from scipy.ndimage import shift as nd_shift
    from skimage.registration import phase_cross_correlation

    ref = np.asarray(reference)
    mov = np.asarray(moving)
    if bbox is not None:
        y0, y1, x0, x1 = (int(v) for v in bbox)
        ref = ref[y0:y1, x0:x1]
        mov = mov[y0:y1, x0:x1]
        mask = None  # region already isolated by the crop
    fa = _highpass(ref, highpass_sigma)
    fb = _highpass(mov, highpass_sigma)
    if window:
        win = _hann2d(fa.shape)
        fa_w, fb_w = fa * win, fb * win
    else:
        fa_w, fb_w = fa, fb
    if fa_w.std() < 1e-6 or fb_w.std() < 1e-6:
        # No texture to correlate — a peak would be spurious.
        return np.zeros(2, dtype=np.float64), 0.0
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        out = phase_cross_correlation(fa, fb, reference_mask=m, moving_mask=m)
        shift = np.asarray(out[0] if isinstance(out, tuple) else out,
                           dtype=np.float64)
    else:
        shift, _err, _phase = phase_cross_correlation(
            fa_w, fb_w, upsample_factor=max(1, int(upsample)),
            normalization="phase", disambiguate=True,
        )
        shift = np.asarray(shift, dtype=np.float64)
    b_aligned = nd_shift(fb, shift=shift, order=1, mode="constant", cval=0.0)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        return shift, (float(_ncc(fa[m], b_aligned[m])) if m.any() else 0.0)
    return shift, float(_ncc(fa, b_aligned))


def apply_shift(image: np.ndarray, shift: np.ndarray, order: int = 1) -> np.ndarray:
    """Resample ``image`` by ``shift`` ``(row, col)``, preserving dtype.

    ``order`` — 0 nearest, 1 bilinear (safe default for uint16), 3 cubic.
    """
    from scipy.ndimage import shift as nd_shift

    arr = np.asarray(image)
    moved = nd_shift(arr.astype(np.float32), shift=np.asarray(shift, dtype=np.float64),
                     order=int(order), mode="constant", cval=0.0)
    return _cast_like(moved, arr.dtype)


# ── rigid / affine (ECC) ──────────────────────────────────────────────────────

def _cv_motion(model: str):
    import cv2
    return {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
        "homography": cv2.MOTION_HOMOGRAPHY,
    }[model]


def ecc_align(
    reference: np.ndarray,
    moving: np.ndarray,
    model: str = "euclidean",
    init_shift: Optional[np.ndarray] = None,
    iters: int = 200,
    eps: float = 1e-6,
    gauss: int = 5,
    interp_order: int = 1,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Align ``moving`` onto ``reference`` with ECC (subpixel, brightness/contrast
    invariant). ``model`` ∈ {translation, euclidean, affine, homography}.

    Optionally seed the translation part from a phase-correlation ``init_shift``
    ``(row, col)`` for speed/robustness. ``mask`` (bool ``(H,W)``) restricts the
    estimate to a region via ECC's ``inputMask`` — kept in full-frame coordinates
    (not cropped) so the rotation center stays correct. Returns ``(warp_matrix,
    cc, aligned)``; on non-convergence returns the seed warp, ``cc=0.0`` and the
    *unaligned* ``moving`` (so callers can gate on ``cc`` and fall back to identity).
    """
    import cv2

    motion = _cv_motion(model)
    ref = np.asarray(reference).astype(np.float32)
    mov = np.asarray(moving).astype(np.float32)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)
    if init_shift is not None:
        # warp maps reference→moving coords; a +shift of moving means sampling the
        # moving image at reference_location + shift → translation (col, row).
        warp[0, 2] = float(init_shift[1])
        warp[1, 2] = float(init_shift[0])

    input_mask = None
    if mask is not None:
        input_mask = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))
    g = int(gauss) | 1  # gaussFiltSize must be odd
    try:
        cc, warp = cv2.findTransformECC(ref, mov, warp, motion, criteria, input_mask, g)
    except cv2.error:
        return warp, 0.0, np.asarray(moving)

    aligned = apply_warp(moving, warp, motion=motion, output_shape=ref.shape,
                         interp_order=interp_order)
    return warp, float(cc), aligned


def apply_warp(
    image: np.ndarray,
    warp_matrix: np.ndarray,
    motion: Optional[int] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    interp_order: int = 1,
) -> np.ndarray:
    """Resample ``image`` through an ECC ``warp_matrix`` (reference→moving map),
    using ``WARP_INVERSE_MAP``. Preserves dtype. ``motion`` selects affine vs
    perspective (inferred from the matrix shape if ``None``)."""
    import cv2

    arr = np.asarray(image)
    h, w = output_shape if output_shape is not None else arr.shape[:2]
    is_homography = (motion == cv2.MOTION_HOMOGRAPHY if motion is not None
                     else warp_matrix.shape[0] == 3)
    interp = cv2.INTER_LINEAR if int(interp_order) >= 1 else cv2.INTER_NEAREST
    flags = cv2.WARP_INVERSE_MAP | interp
    src = arr.astype(np.float32)
    if is_homography:
        out = cv2.warpPerspective(src, warp_matrix, (int(w), int(h)), flags=flags,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        out = cv2.warpAffine(src, warp_matrix, (int(w), int(h)), flags=flags,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _cast_like(out, arr.dtype)


# ── (T, H, W) series driver ───────────────────────────────────────────────────

def stabilize(
    volume: np.ndarray,
    model: str = "translation",
    reference: str = "previous",
    upsample: int = 20,
    highpass_sigma: float = 2.0,
    interp_order: int = 1,
    min_confidence: float = 0.0,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stabilize a ``(T, H, W)`` series onto a reference frame.

    Args:
        volume: ``(T, H, W)`` series (single channel).
        model: ``translation`` (phase correlation) | ``euclidean`` | ``affine``
            (ECC, seeded from a phase-correlation translation).
        reference: ``first`` (register every frame to frame 0), ``previous``
            (cumulative — each frame to the one before), or ``mean`` (to the
            series mean). Frame 0 is always the fixed anchor.
        upsample: phase-correlation sub-pixel factor (accuracy ≈ 1/upsample px).
        highpass_sigma: band-pass whitening sigma before correlation (0 = off).
        interp_order: resampling order (0 nearest, 1 bilinear, 3 cubic).
        min_confidence: frames whose registration confidence (NCC / ECC cc) is
            below this fall back to identity (guards spurious peaks on near-blank
            frames). 0 = never gate.

    Returns:
        ``(aligned (T,H,W) same dtype, shifts (T,2) row/col, confidence (T,))``.
        For non-translation models ``shifts`` holds the warp translation part.
    """
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"stabilize expects a (T,H,W) array, got shape {vol.shape}")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    if reference not in REFERENCE_MODES:
        raise ValueError(f"unknown reference {reference!r}; expected {REFERENCE_MODES}")

    T = int(vol.shape[0])
    out = np.empty_like(vol)
    shifts = np.zeros((T, 2), dtype=np.float64)
    confidence = np.ones(T, dtype=np.float64)
    cum_shift = np.zeros(2, dtype=np.float64)

    # `first` / `previous` anchor on frame 0 (left untouched). `mean` has no
    # natural anchor, so *every* frame — including frame 0 — is registered to the
    # series mean, otherwise frame 0 would sit at a constant offset from the rest.
    if reference == "mean":
        mean_ref = vol.mean(axis=0)
        start = 0
    else:
        mean_ref = None
        out[0] = vol[0]
        start = 1

    for t in range(start, T):
        if cancelled_cb is not None and cancelled_cb():
            # Leave the remaining (unprocessed) frames as-is.
            out[t:] = vol[t:]
            break

        if reference == "first":
            ref = vol[0]
        elif reference == "mean":
            ref = mean_ref
        else:  # previous
            ref = vol[t - 1]

        if model == "translation":
            sh, conf = estimate_translation(ref, vol[t], upsample=upsample,
                                            highpass_sigma=highpass_sigma)
            if conf < float(min_confidence):
                sh = np.zeros(2, dtype=np.float64)  # gate spurious peak
            if reference == "previous":
                cum_shift = cum_shift + sh
                eff = cum_shift.copy()
            else:
                eff = sh
            out[t] = apply_shift(vol[t], eff, order=interp_order)
            shifts[t] = eff
            confidence[t] = conf
        else:
            # ECC path. In `previous` mode reference the already-stabilized prior
            # output so any model accumulates without composing warp matrices.
            ecc_ref = out[t - 1] if reference == "previous" else ref
            seed, _ = estimate_translation(ecc_ref, vol[t], upsample=upsample,
                                           highpass_sigma=highpass_sigma)
            warp, cc, aligned = ecc_align(ecc_ref, vol[t], model=model,
                                          init_shift=seed, interp_order=interp_order)
            if cc < float(min_confidence):
                aligned = vol[t]  # gate: leave frame unaligned
                warp = np.eye(2, 3, dtype=np.float32)
            out[t] = aligned
            shifts[t] = (float(warp[1, 2]), float(warp[0, 2]))  # (row, col)
            confidence[t] = cc

        if progress_cb is not None:
            progress_cb(int(100 * t / max(T - 1, 1)))

    return out, shifts, confidence


# ── register once, apply to all (cross-channel) ───────────────────────────────

def _homog(mat: np.ndarray) -> np.ndarray:
    """Promote a 2x3 affine to a 3x3 homogeneous matrix."""
    h = np.eye(3, dtype=np.float64)
    h[:2, :] = np.asarray(mat, dtype=np.float64)
    return h


def common_translation_crop(
    shifts: np.ndarray, shape: Tuple[int, int], inset_edges: bool = True,
) -> Optional[Tuple[int, int, int, int]]:
    """Largest axis-aligned rectangle that is real (non-padded) data in **every**
    frame after :func:`apply_shift`, for the pure-translation model.

    ``apply_shift`` moves content by ``shift=(dy,dx)`` and zero-pads the vacated
    border, so frame ``t``'s valid region is ``[max(0,dy):H+min(0,dy),
    max(0,dx):W+min(0,dx)]``. The intersection over all frames (pass every series'
    shifts stacked as ``(N,2)`` row/col) is the common region — cropping to it makes
    all frames equal-size, recentred, and border-free.

    Returns ``(y0, y1, x0, x1)`` (row/col half-open), or ``None`` if the drift
    leaves no common overlap. ``inset_edges`` trims 1 px off each padded side to
    avoid bilinear edge fuzz. Translation only (rotation/affine footprints are not
    axis-aligned rectangles).
    """
    s = np.asarray(shifts, dtype=np.float64).reshape(-1, 2)
    H, W = int(shape[0]), int(shape[1])
    if s.size == 0:
        return (0, H, 0, W)
    dy, dx = s[:, 0], s[:, 1]
    y0 = int(np.ceil(max(0.0, float(dy.max()))))
    y1 = int(np.floor(min(float(H), float(H) + float(dy.min()))))
    x0 = int(np.ceil(max(0.0, float(dx.max()))))
    x1 = int(np.floor(min(float(W), float(W) + float(dx.min()))))
    if inset_edges:
        if float(dy.max()) > 0.0:
            y0 += 1
        if float(dy.min()) < 0.0:
            y1 -= 1
        if float(dx.max()) > 0.0:
            x0 += 1
        if float(dx.min()) < 0.0:
            x1 -= 1
    y0, y1 = max(0, min(y0, H)), max(0, min(y1, H))
    x0, x1 = max(0, min(x0, W)), max(0, min(x1, W))
    if y1 <= y0 or x1 <= x0:
        return None
    return (y0, y1, x0, x1)


def _normalize_series(vol: np.ndarray, normalize: str) -> np.ndarray:
    """Per-frame intensity normalization for *estimation only* (geometry is
    unaffected). ``"zscore"`` counters photobleaching / intensity decay so late,
    dim frames still correlate; ``"none"`` passes the series through."""
    if normalize == "zscore":
        out = np.empty(vol.shape, dtype=np.float32)
        for t in range(vol.shape[0]):
            f = vol[t].astype(np.float32)
            out[t] = (f - float(f.mean())) / (float(f.std()) + 1e-6)
        return out
    return vol


def roi_to_mask(
    roi: Optional[Dict[str, Any]], shape: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Turn a serializable ROI spec into ``(mask (H,W) bool | None, bbox
    (y0,y1,x0,x1) | None)``.

    - ``{"kind":"rect","x","y","w","h"}`` → a box mask **and** a bbox (the bbox
      drives the subpixel crop for translation; the mask drives ECC/feature).
    - ``{"kind":"shapes","shapes":[{"type","vertices":[[y,x]…]}…]}`` → a rasterized
      mask (no bbox; freeform uses the masked / ``inputMask`` paths).
    - falsy / unknown → ``(None, None)`` (whole frame).
    """
    if not roi:
        return None, None
    H, W = int(shape[0]), int(shape[1])
    kind = roi.get("kind")
    if kind == "rect":
        x, y = int(roi.get("x", 0)), int(roi.get("y", 0))
        w, h = int(roi.get("w", 0)), int(roi.get("h", 0))
        y0, y1 = max(0, y), min(H, y + h)
        x0, x1 = max(0, x), min(W, x + w)
        if y1 <= y0 or x1 <= x0:
            return None, None
        mask = np.zeros((H, W), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask, (y0, y1, x0, x1)
    if kind == "shapes":
        # rasterize_shapes is vendored above in this file (was a lazy
        # `from nd2studios.backend.analysis.manual_mask import rasterize_shapes`).
        frame = np.zeros((H, W), dtype=np.int32)
        rasterize_shapes(frame, list(roi.get("shapes", [])), H, W)
        mask = frame > 0
        return (mask, None) if mask.any() else (None, None)
    return None, None


def _build_template(
    est_vol: np.ndarray, upsample: int, highpass_sigma: float,
    mask: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Two-pass reference: roughly stabilize the series with a cheap
    translation-``previous`` pass, then average the stabilized frames into a
    single template. Registering every frame to this template (in
    :func:`estimate_series`) avoids both cumulative-error drift and appearance
    divergence from a single (possibly bleached / atypical) anchor frame."""
    T = int(est_vol.shape[0])
    if T == 0:
        return np.zeros(est_vol.shape[1:], dtype=np.float32)
    cum = np.zeros(2, dtype=np.float64)
    rough = np.zeros((T, 2), dtype=np.float64)
    for t in range(1, T):
        if cancelled_cb is not None and cancelled_cb():
            break
        sh, _ = estimate_translation(
            est_vol[t - 1], est_vol[t], upsample=upsample,
            highpass_sigma=highpass_sigma,
            mask=(mask if bbox is None else None), bbox=bbox)
        cum = cum + sh
        rough[t] = cum
    acc = np.zeros(est_vol.shape[1:], dtype=np.float64)
    for t in range(T):
        acc += apply_shift(est_vol[t].astype(np.float32), rough[t], order=1)
    return (acc / max(T, 1)).astype(np.float32)


def estimate_features(
    reference: np.ndarray,
    moving: np.ndarray,
    transform: str = "affine",
    mask: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    n_keypoints: int = 800,
    min_inliers: int = 8,
    residual_threshold: float = 2.0,
) -> Tuple[np.ndarray, float]:
    """Feature-based alignment: ORB keypoints + descriptor matching + RANSAC fit of
    a Euclidean / Similarity / Affine model. Handles large displacement / rotation
    / scale that phase correlation and ECC cannot.

    Returns ``(warp 2x3, confidence)`` where the warp maps **reference → moving**
    (the same convention :func:`apply_warp` expects under ``WARP_INVERSE_MAP``) and
    confidence is the RANSAC inlier fraction. On too few keypoints / matches /
    inliers returns ``(eye(2,3), 0.0)`` so the caller falls back to identity — the
    graceful failure mode on near-blank / sparse fields. ``bbox`` crops keypoint
    detection to a rectangular ROI; ``mask`` keeps only matches inside a freeform ROI.
    """
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import (
        AffineTransform, EuclideanTransform, SimilarityTransform,
    )

    tclass = {"euclidean": EuclideanTransform, "similarity": SimilarityTransform,
              "affine": AffineTransform}.get(transform, AffineTransform)
    ref = np.asarray(reference).astype(np.float32)
    mov = np.asarray(moving).astype(np.float32)
    identity = np.eye(2, 3, dtype=np.float32)

    if bbox is not None:
        y0, y1, x0, x1 = (int(v) for v in bbox)
        rr, mm = ref[y0:y1, x0:x1], mov[y0:y1, x0:x1]
        off = np.array([y0, x0], dtype=np.float64)
    else:
        rr, mm, off = ref, mov, np.zeros(2, dtype=np.float64)

    def _kp(img):
        orb = ORB(n_keypoints=int(n_keypoints), fast_threshold=0.05)
        try:
            orb.detect_and_extract(img)
        except Exception:  # noqa: BLE001 — blank / too-small patch
            return None, None
        return orb.keypoints, orb.descriptors

    kp_r, des_r = _kp(rr)
    kp_m, des_m = _kp(mm)
    if des_r is None or des_m is None or len(kp_r) < 3 or len(kp_m) < 3:
        return identity, 0.0
    matches = match_descriptors(des_r, des_m, cross_check=True)
    if len(matches) < max(3, int(min_inliers)):
        return identity, 0.0
    # keypoints are (row, col); to full-frame (x, y) = (col, row).
    src = (kp_r[matches[:, 0]] + off)[:, ::-1]
    dst = (kp_m[matches[:, 1]] + off)[:, ::-1]
    if mask is not None and bbox is None:
        m = np.asarray(mask, dtype=bool)
        keep = [i for i, (x, y) in enumerate(src)
                if 0 <= int(round(y)) < m.shape[0] and 0 <= int(round(x)) < m.shape[1]
                and m[int(round(y)), int(round(x))]]
        if len(keep) < max(3, int(min_inliers)):
            return identity, 0.0
        src, dst = src[keep], dst[keep]
    try:
        model, inliers = ransac(
            (src, dst), tclass, min_samples=3,
            residual_threshold=float(residual_threshold), max_trials=2000)
    except Exception:  # noqa: BLE001
        return identity, 0.0
    if model is None or inliers is None or int(np.sum(inliers)) < int(min_inliers):
        return identity, 0.0
    # model maps src(reference, x,y) → dst(moving, x,y): exactly reference→moving.
    warp = np.asarray(model.params, dtype=np.float32)[:2, :]
    return warp, float(np.sum(inliers)) / max(1, int(inliers.size))


def estimate_series(
    series: np.ndarray,
    model: str = "translation",
    reference: str = "previous",
    upsample: int = 20,
    highpass_sigma: float = 2.0,
    min_confidence: float = 0.0,
    normalize: str = "none",
    roi: Optional[Dict[str, Any]] = None,
    feature_transform: str = "affine",
    min_inliers: int = 8,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Estimate per-frame **effective** transforms aligning each frame of
    ``series`` ``(T, H, W)`` onto the anchor, *without applying them*.

    The returned transforms are **absolute** (each maps a frame directly onto the
    anchor), so they can be applied identically to any other channel via
    :func:`apply_series` — the "register once on a reference channel, apply to all
    channels" rule that preserves colocalization.

    Robustness (V1.60):
    - ``reference="template"`` — two-pass anchor (rough-stabilize → mean), robust to
      cumulative drift and to a bleached / atypical single anchor frame.
    - ``min_confidence`` gates a failed frame by **holding the last good transform**
      (absolute modes) or **skipping the increment** (``previous``) instead of
      snapping to identity — so one low-SNR frame never poisons the series.
    - ``normalize="zscore"`` per-frame normalization counters photobleaching.
    - ``roi`` (see :func:`roi_to_mask`) restricts the estimate to a region.
    - ``model="feature"`` uses ORB+RANSAC (``feature_transform`` / ``min_inliers``).

    Returns ``{"model","reference","shifts" (T,2) row/col,"warps" (T,2,3)|None,
    "confidence" (T,),"gated" (T,) bool}``. ``warps`` is ``None`` for pure
    translation (use ``shifts``); otherwise ``shifts[t]`` is the warp's translation
    part (for plotting).
    """
    vol = np.asarray(series)
    if vol.ndim != 3:
        raise ValueError(f"estimate_series expects (T,H,W), got {vol.shape}")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    if reference not in REFERENCE_MODES:
        raise ValueError(f"unknown reference {reference!r}; expected {REFERENCE_MODES}")

    T = int(vol.shape[0])
    shifts = np.zeros((T, 2), dtype=np.float64)
    confidence = np.ones(T, dtype=np.float64)
    gated = np.zeros(T, dtype=bool)
    use_warp = model != "translation"
    warps = (np.tile(np.eye(2, 3, dtype=np.float32), (T, 1, 1)) if use_warp else None)

    # ROI: rect → bbox (subpixel crop for translation) + box mask (ECC/feature);
    # freeform shapes → mask only. Estimate on the region, apply full-frame.
    mask, bbox = roi_to_mask(roi, vol.shape[1:])
    tmask = mask if bbox is None else None  # translation uses the crop for a rect

    # Normalization affects correlation only (not geometry); estimate on ``est``.
    est = _normalize_series(vol, normalize)

    # Anchor image for the absolute reference modes.
    if reference == "template":
        anchor = _build_template(est, upsample, highpass_sigma, mask, bbox, cancelled_cb)
    elif reference == "mean":
        anchor = est.mean(axis=0)
    elif reference == "first":
        anchor = est[0]
    else:
        anchor = None  # ``previous`` (cumulative)
    absolute = reference != "previous"

    cum_shift = np.zeros(2, dtype=np.float64)
    h_abs = np.eye(3, dtype=np.float64)          # composed absolute warp (cumulative)
    last_shift = np.zeros(2, dtype=np.float64)   # last good (absolute translation)
    last_warp = np.eye(2, 3, dtype=np.float32)   # last good (absolute warp)

    start = 0 if reference in ("mean", "template") else 1
    for t in range(start, T):
        if cancelled_cb is not None and cancelled_cb():
            break
        ref_img = anchor if absolute else est[t - 1]
        cur = est[t]

        if model == "translation":
            sh, conf = estimate_translation(
                ref_img, cur, upsample=upsample, highpass_sigma=highpass_sigma,
                mask=tmask, bbox=bbox)
            g = bool(conf < float(min_confidence))
            if absolute:
                eff = last_shift if g else sh
                shifts[t] = eff
                last_shift = eff
            else:                                # previous: skip a bad increment
                if not g:
                    cum_shift = cum_shift + sh
                shifts[t] = cum_shift.copy()
            confidence[t] = conf
            gated[t] = g
        else:
            if model == "feature":
                warp, conf = estimate_features(
                    ref_img, cur, transform=feature_transform, mask=mask,
                    bbox=bbox, min_inliers=min_inliers)
            else:
                seed, _ = estimate_translation(
                    ref_img, cur, upsample=upsample, highpass_sigma=highpass_sigma,
                    mask=tmask, bbox=bbox)
                warp, conf, _ = ecc_align(ref_img, cur, model=model,
                                          init_shift=seed, mask=mask)
            g = bool(conf < float(min_confidence))
            if absolute:
                eff = last_warp if g else np.asarray(warp, dtype=np.float32)
                warps[t] = eff
                last_warp = eff
            else:                                # previous: skip a bad increment
                if not g:
                    h_abs = _homog(warp) @ h_abs
                warps[t] = h_abs[:2, :].astype(np.float32)
            shifts[t] = (float(warps[t][1, 2]), float(warps[t][0, 2]))
            confidence[t] = conf
            gated[t] = g

        if progress_cb is not None:
            progress_cb(int(100 * (t - start + 1) / max(T - start, 1)))

    return {"model": model, "reference": reference, "shifts": shifts,
            "warps": warps, "confidence": confidence, "gated": gated}


def apply_series(
    series: np.ndarray,
    transforms: Dict[str, Any],
    interp_order: int = 1,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Apply per-frame effective ``transforms`` (from :func:`estimate_series`) to a
    ``(T, H, W)`` series — any channel — returning the aligned series (dtype
    preserved). This is the "apply to all channels" half of register-once."""
    vol = np.asarray(series)
    if vol.ndim != 3:
        raise ValueError(f"apply_series expects (T,H,W), got {vol.shape}")
    T = int(vol.shape[0])
    shifts = transforms.get("shifts")
    warps = transforms.get("warps")
    out = np.empty_like(vol)
    for t in range(T):
        if warps is None:
            sh = np.asarray(shifts[t], dtype=np.float64)
            out[t] = vol[t] if not np.any(sh) else apply_shift(vol[t], sh, order=interp_order)
        else:
            W = np.asarray(warps[t], dtype=np.float32)
            if np.allclose(W, np.eye(2, 3)):
                out[t] = vol[t]
            else:
                out[t] = apply_warp(vol[t], W, output_shape=vol.shape[1:],
                                    interp_order=interp_order)
        if progress_cb is not None:
            progress_cb(int(100 * (t + 1) / max(T, 1)))
    return out


def apply_frame(
    frame: np.ndarray,
    transforms: Dict[str, Any],
    t: int,
    interp_order: int = 1,
) -> np.ndarray:
    """Apply the ``t``-th per-frame transform from ``transforms`` to a single 2-D
    ``frame`` — the single-plane analogue of :func:`apply_series`, used by the
    Pipelines-page preview so a paused, cropped preview runs on the drift-corrected
    image. Returns the frame unchanged for an identity / absent transform or an
    out-of-range ``t`` (so it never breaks the read path)."""
    img = np.asarray(frame)
    shifts = transforms.get("shifts")
    warps = transforms.get("warps")
    ti = int(t)
    if warps is None:
        if shifts is None or ti < 0 or ti >= len(shifts):
            return img
        sh = np.asarray(shifts[ti], dtype=np.float64)
        return img if not np.any(sh) else apply_shift(img, sh, order=interp_order)
    if ti < 0 or ti >= len(warps):
        return img
    W = np.asarray(warps[ti], dtype=np.float32)
    if np.allclose(W, np.eye(2, 3)):
        return img
    return apply_warp(img, W, output_shape=img.shape[:2], interp_order=interp_order)
