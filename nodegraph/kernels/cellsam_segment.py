"""cellsam_segment — CellSAM foundation-model 2-D instance segmentation (thin glue).

PURPOSE
    Segment cells in a SINGLE already-prepared 2-D plane with CellSAM and return a
    contiguous ``int32`` label image of the same ``(H, W)`` shape. This is the
    2-D-per-plane math kernel ONLY: the caller owns all prep (file I/O, Z/channel
    selection, the per-``(m,t,z,c)`` loop, crop, enhancement) and every physical-unit
    decision (the µm² / µm³ size filters) — exactly the division of labour used by
    :mod:`nodegraph.kernels.stardist_segment`.

WHERE THE REAL MATH LIVES
    NOT here, and not in this repo. CellSAM is a SAM ViT-B backbone whose features feed
    (a) **CellFinder**, an Anchor-DETR set-prediction detector that emits one bounding
    box per cell, and (b) a fine-tuned SAM mask decoder that turns those boxes into
    instance masks — i.e. automatic prompt engineering for SAM (Marks, Israel et al.,
    "CellSAM: a foundation model for cell segmentation", *Nature Methods* 22:2585–2593,
    2025; https://github.com/vanvalenlab/cellSAM). All of it lives in the external
    ``cellSAM`` + ``segment_anything`` + ``torch`` packages plus the downloaded
    ``cellsam_general`` / ``cellsam_extra`` weights.

    The glue vendored here is thin and deliberately so:
      * a **process-singleton** model loader, so the weights are read once per process
        instead of once per plane,
      * the ``(H, W)`` → CellSAM channel-slot convention,
      * one ``segment_cellular_image`` call per plane, or ``segment_wsi`` when tiling a
        large FOV,
      * normalization of four upstream quirks (below),
      * a contiguous ``1..K`` relabel so the caller can offset ids globally across planes.

WHY NOT ``cellsam_pipeline``
    ``cellSAM.cellsam_pipeline`` is the documented one-shot entry point, but it calls
    ``get_model()`` **on every invocation**. Per-plane that re-reads the checkpoint for
    every frame of a time series, which dominates the run. It also min-max normalizes
    into ``img`` in place, and its ``low_contrast_enhancement=True`` branch is broken
    upstream (``cellSAM.utils.enhance_low_contrast`` assigns ``model.bbox_threshold``
    with no ``model`` in scope → ``NameError``). So this kernel drives the two lower-level
    entry points directly and keeps the model itself cached.

UPSTREAM QUIRKS THIS KERNEL NORMALIZES (verified against master, cellSAM 0.0.dev1)
    1. ``segment_cellular_image(img, model, ...)`` takes ``model`` as a REQUIRED
       positional argument, even though the project README shows
       ``segment_cellular_image(img, device='cuda')``. The README call raises
       ``TypeError``; we always pass a loaded model.
    2. **The no-cells path is broken upstream, in two layers.**
       (a) ``segment_cellular_image``'s guard is ``if preds is None``, but
       ``CellSAM.predict`` returns the 4-tuple ``(None, None, None, None)`` when no box
       survives — so the guard NEVER fires (upstream issue #98). Execution unpacks the
       tuple and calls ``fill_holes_and_remove_small_masks(None)``, raising
       ``AttributeError: 'NoneType' object has no attribute 'ndim'``. A blank plane, an
       empty FOV or the dark end slices of a z-stack would crash the whole pull, so
       :func:`segment_plane` absorbs exactly that AttributeError into an empty plane —
       repairing the guard, not inventing behaviour ("no cells" IS an empty segmentation,
       which is what upstream's own dead branch returns).
       (b) That dead branch is itself mis-shaped: it returns ``np.zeros(img.shape[1:])``
       where ``img`` has by then been replaced by the ``(1, 3, H, W)`` torch tensor, i.e.
       ``(3, H, W)`` rather than ``(H, W)``. :func:`segment_plane` also collapses any 3-D
       return to one plane, so an upstream fix for (a) lands safely.
       The tiled path is immune to both: ``cellSAM.wsi.segment_chunk`` wraps each block in
       ``try/except Exception`` and substitutes zeros.
    3. ``fill_holes_and_remove_small_masks`` (always applied inside
       ``segment_cellular_image``, ``min_size=25`` px) mutates its argument in place.
       Harmless here — the array is upstream-local — but it is why the returned ids are
       already re-numbered and why we never rely on the pre-filter numbering.
    4. ``postprocess=True`` runs ``cellSAM.model.postprocess_predictions``, which ends in
       ``np.max(new_masks, axis=0)`` over a list built from ``np.unique(...)[1:]``. If the
       decoder returned a non-empty prediction that nonetheless contains no non-zero
       label, that list is empty and numpy raises ``ValueError: zero-size array``. We do
       NOT swallow it: a crash from a documented upstream edge is more honest than a
       silently blank plane. Leave ``postprocess`` off unless the images are noisy.

    Not a quirk but worth recording: ``cellSAM/modelconfig.yaml`` declares
    ``device: cuda``, yet ``AnchorDETR.build_inference`` never reads ``args.device`` and
    ``CellSAM.predict`` takes its device from ``next(self.parameters()).device``. A
    CPU-only torch build therefore works — the model simply stays where it was loaded.

CHANNEL CONVENTION (why a single plane lands in the LAST slot)
    CellSAM was trained on 3-channel input ordered ``(blank, nuclear, whole-cell)``, and
    ``cellSAM.utils.format_image_shape`` right-aligns whatever it is given
    (``out[:, :, -C:] = img``). A 1-channel plane therefore occupies the whole-cell slot,
    which is exactly how the paper handles nuclear-only datasets ("We moved the green
    channel to blue for nuclear-only datasets … to keep the blue channel always
    occupied", Methods → Dataset construction). We pass each plane through as-is and let
    upstream place it, rather than re-inventing the padding.

MODEL WEIGHTS / LICENCE
    ``get_model()`` downloads to ``$HOME/.deepcell/models`` on first use and needs a
    DeepCell API token in ``DEEPCELL_ACCESS_TOKEN`` (https://users.deepcell.org). The
    weights are licensed for **non-commercial academic use**. ``model_path`` bypasses the
    download entirely and loads a local ``.pt`` via ``cellSAM.get_local_model``.

THIRD-PARTY IMPORTS
    ``cellSAM`` / ``torch`` are imported LAZILY inside the loader, so
    ``import nodegraph.kernels.cellsam_segment`` succeeds with neither installed — the
    whole node palette still registers, and only pulling the node raises the install
    hint. numpy is the sole top-level dependency, and availability is probed with
    ``importlib.util.find_spec`` (no import side effects) exactly as
    :mod:`nodegraph.kernels.dic_correlate` probes ``al_dic``.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Optional, Tuple

import numpy as np

#: CellSAM model singleton — (model, model_path, device) → loaded ``nn.Module``.
#: Loading reads a multi-hundred-MB checkpoint and builds a ViT-B, so it must happen
#: once per process, not once per plane (the same reason ``stardist_segment`` caches).
_cellsam_model = None
_cellsam_key: Optional[Tuple[str, str, str]] = None

#: Environment override for the inference device: ``auto`` (default) | ``cpu`` | ``cuda``.
#: This is an environment knob rather than a node socket for the same reason
#: ``NODELAB_STARDIST_CPU`` is: the model is a process singleton, so a per-graph device
#: control would silently stop taking effect after the first pull, and it would make the
#: memo non-deterministic (identical recipe hash, different device).
DEVICE_ENV = "NODELAB_CELLSAM_DEVICE"

_INSTALL_HINT = (
    "CellSAM is not installed. Install the package and its weights:\n"
    "    pip install git+https://github.com/vanvalenlab/cellSAM.git\n"
    "Model weights download to ~/.deepcell/models on first use and require a DeepCell "
    "API token (non-commercial academic licence):\n"
    "    set DEEPCELL_ACCESS_TOKEN to the token from https://users.deepcell.org\n"
    "Alternatively set the node's `model_path` to a local CellSAM .pt checkpoint, which "
    "skips the download (but still needs the cellSAM package)."
)

#: Raised when the package IS importable but its WEIGHTS could not be obtained — a
#: different problem with a different fix, so it must not print the install hint (which
#: sent one debugging session chasing a phantom missing package while the real cause was
#: TLS interception).
_WEIGHTS_HINT = (
    "The cellSAM package is installed, so this is a WEIGHTS problem, not an install one.\n"
    "FIX IT IN ONE COMMAND:  python scripts/setup_cellsam.py\n"
    "(it fetches + verifies the weights, walks you through the token, and handles the\n"
    "TLS-interception case). What can be wrong:\n"
    "  * no DEEPCELL_ACCESS_TOKEN, or an expired/mistyped one (paste it bare — no <>, no "
    "quotes) — create one at https://users.deepcell.org\n"
    "  * no network, or TLS interception (corporate proxy / antivirus HTTPS scanning). "
    "Python 3.13 verifies strictly, so an AV-generated CA can be rejected even when the "
    "browser and PowerShell accept it — download the archive with a client that uses the "
    "OS trust store and unpack it into ~/.deepcell/models, or `pip install truststore` "
    "and inject it.\n"
    "  * once ~/.deepcell/models/cellsam_v<ver>/<model>.pt exists, loading is OFFLINE and "
    "no token is consulted at all.\n"
    "  * or point the node's `model_path` at a local .pt to bypass all of the above."
)


def cellsam_available() -> bool:
    """True if the optional ``cellSAM`` package is importable — a side-effect-free probe
    (``find_spec``), so a GUI/selftest can ask without paying a multi-second torch import.
    Note this says nothing about the model WEIGHTS, which are a separate failure mode
    (:func:`get_cellsam_model` raises for those)."""
    return importlib.util.find_spec("cellSAM") is not None


def _require_cellsam():
    """Import and return ``(get_model, get_local_model, segment_cellular_image)``, or
    raise the friendly install hint."""
    if not cellsam_available():
        raise ImportError(_INSTALL_HINT)
    mod = importlib.import_module("cellSAM")
    return mod.get_model, mod.get_local_model, mod.segment_cellular_image


def resolve_device(requested: str = "") -> str:
    """The torch device string to run on: ``requested`` (or :data:`DEVICE_ENV`, or
    ``auto``) resolved against what torch can actually do.

    ``auto`` picks CUDA when it is available and CPU otherwise. An explicit ``cuda``
    that is not available is a hard error rather than a silent CPU fallback — a user who
    asked for the GPU wants to know the 10× slowdown happened. Raises ``ImportError``
    when torch is missing, which is the same failure the caller already handles.
    """
    want = (requested or os.environ.get(DEVICE_ENV, "") or "auto").strip().lower()
    if want not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto|cpu|cuda, got {want!r} "
                         f"(set {DEVICE_ENV} to override)")
    if want == "cpu":
        return "cpu"                # nothing to probe — do not pay the torch import
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - env dependent
        raise ImportError(_INSTALL_HINT) from exc
    available = bool(torch.cuda.is_available())
    if want == "auto":
        return "cuda" if available else "cpu"
    if want == "cuda" and not available:
        raise RuntimeError(
            "CellSAM was asked for device 'cuda' but torch reports no CUDA device "
            f"(this build is {torch.__version__}). Unset {DEVICE_ENV} to fall back to "
            "CPU automatically.")
    return want


def get_cellsam_model(model: str = "cellsam_general", *, model_path: str = "",
                      device: str = ""):
    """Get or load the CellSAM model (process singleton keyed by the load parameters).

    Parameters
    ----------
    model : str
        ``"cellsam_general"`` (the published generalist, for reproducing the paper) or
        ``"cellsam_extra"`` (extra training data, recommended for domains outside the
        paper). Ignored when *model_path* is given.
    model_path : str
        Path to a local ``.pt`` checkpoint. Skips the DeepCell download + API token.
    device : str
        ``auto`` | ``cpu`` | ``cuda`` — see :func:`resolve_device`.

    Raises
    ------
    ImportError
        With an actionable install hint when ``cellSAM`` (or torch) is unavailable, or
        when the weights cannot be fetched.
    """
    global _cellsam_model, _cellsam_key
    # probe the package BEFORE torch, so an absent cellSAM reports the install hint
    # rather than whatever torch happens to say.
    get_model, get_local_model, _ = _require_cellsam()
    dev = resolve_device(device)
    key = (str(model), str(model_path), dev)
    if _cellsam_model is not None and _cellsam_key == key:
        return _cellsam_model
    try:
        net = get_local_model(model_path) if model_path else get_model(model)
    except Exception as exc:                        # noqa: BLE001 - surface as one hint
        # The package imported (``_require_cellsam`` succeeded above), so this is about the
        # WEIGHTS: token, network, TLS, or a bad path. Do NOT print the install hint here.
        raise ImportError(f"CellSAM weights unavailable "
                          f"({'local ' + model_path if model_path else model}): "
                          f"{type(exc).__name__}: {exc}\n{_WEIGHTS_HINT}") from exc
    net = net.eval()
    if dev != "cpu":
        # Move once at load. ``segment_cellular_image`` also does ``model.to(device)``
        # per call, which is then a no-op instead of a per-plane host→device copy.
        net = net.to(dev)
    _cellsam_model, _cellsam_key = net, key
    return net


def relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    """Renumber an integer label image to a contiguous ``1..K`` (background stays 0).

    One ``O(pixels)`` bincount + LUT pass, the same shape as
    ``stardist_segment.filter_and_relabel`` minus the filtering (the caller applies the
    physical-unit size filter, which needs calibration this kernel does not have).
    Contiguity is what lets the caller offset ids per plane into globally unique ones.
    """
    lab = np.asarray(labels)
    if lab.size == 0 or int(lab.max()) == 0:
        return lab.astype(np.int32, copy=False)
    present = np.flatnonzero(np.bincount(lab.ravel()))
    present = present[present > 0]
    lut = np.zeros(int(lab.max()) + 1, dtype=np.int32)
    lut[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    return lut[lab]


def segment_plane(
    image: np.ndarray,
    model=None,
    *,
    bbox_threshold: float = 0.4,
    normalize: bool = True,
    postprocess: bool = False,
    remove_boundaries: bool = False,
    model_name: str = "cellsam_general",
    model_path: str = "",
    device: str = "",
    tile: bool = False,
    tile_size: int = 512,
    overlap: int = 56,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Segment cells in a single 2-D plane with CellSAM.

    Parameters
    ----------
    image : 2-D array (H, W), any dtype
        One prepared plane. Placed in CellSAM's whole-cell channel slot upstream (see
        the module docstring).
    model : nn.Module, optional
        A loaded CellSAM model; ``None`` uses/loads the singleton.
    bbox_threshold : float
        CellFinder box confidence cut — *the* precision/recall knob. Upstream default
        0.4; lower it for out-of-distribution images. (CellSAM then blends it with a
        per-image k-means split of the box confidences, ``0.66·T + 0.33·T_cluster``,
        which is the paper's dynamic ``T_box``.)
    normalize : bool
        Apply CellSAM's own preprocessing (99.9-percentile clip + per-channel rescale +
        CLAHE, kernel 128) — the paper's Methods pipeline. Leave on unless the caller
        has already matched it.
    postprocess : bool
        Upstream morphological cleanup, "recommended for noisy images". See quirk 4.
    remove_boundaries : bool
        Erode a one-pixel gap between touching cells.
    tile : bool
        Segment the plane in overlapping blocks and stitch by IoU
        (``cellSAM.wsi.segment_wsi``) instead of in one pass. Needed for large FOVs /
        very many cells (upstream suggests tiling above roughly 3000 cells per image).
    tile_size, overlap : int
        Block edge and overlap **in pixels** (the caller converts from µm). ``overlap``
        must be wide enough to contain a typical cell, and it doubles as upstream's
        ``iou_depth`` — which upstream requires to be ``<= overlap``, so passing the
        same value is both legal and maximal.
    iou_threshold : float
        IoU above which two blocks' labels are merged into one cell.

    Returns
    -------
    labels : 2-D ``int32`` array, contiguous ids ``1..K``, 0 = background.
    """
    plane = np.asarray(image)
    if plane.ndim != 2:
        raise ValueError(f"cellsam segment_plane expects a 2-D plane, got shape "
                         f"{plane.shape} — the caller owns the (m,t,z,c) loop")
    if model is None:
        model = get_cellsam_model(model_name, model_path=model_path, device=device)
    dev = resolve_device(device)
    kwargs = dict(normalize=bool(normalize), postprocess=bool(postprocess),
                  remove_boundaries=bool(remove_boundaries),
                  bbox_threshold=float(bbox_threshold), device=dev)

    if tile:
        try:
            from cellSAM.wsi import segment_wsi
        except ImportError as exc:                  # dask-image / sklearn ride along
            raise ImportError(f"CellSAM tiled inference needs cellSAM.wsi and its "
                              f"dask-image / scikit-learn dependencies: {exc}\n"
                              f"{_INSTALL_HINT}") from exc
        block = max(64, int(tile_size))
        lap = max(1, min(int(overlap), block - 1))
        out = segment_wsi(plane, block, lap, lap, float(iou_threshold),
                          model=model, **kwargs)
        lab = np.asarray(getattr(out, "compute", lambda: out)())
    else:
        segment_cellular_image = _require_cellsam()[2]
        try:
            lab = np.asarray(segment_cellular_image(plane, model, **kwargs)[0])
        except AttributeError as exc:
            # QUIRK 2b — the REAL no-cells path (upstream issue #98). ``CellSAM.predict``
            # returns the 4-tuple ``(None, None, None, None)`` when no box survives the
            # confidence/IoU filter, so ``segment_cellular_image``'s ``if preds is None``
            # guard never fires; it unpacks the tuple and calls
            # ``fill_holes_and_remove_small_masks(None)`` → ``AttributeError`` on
            # ``masks.ndim`` (or later on ``x.cpu()``). A blank plane, an empty FOV or the
            # dark end slices of a stack would otherwise CRASH the whole pull. "No cells"
            # is an empty segmentation, which is what upstream's own unreachable branch
            # returns, so this repairs a broken guard rather than inventing a behaviour —
            # and it stays narrow: only an AttributeError ON NoneState is absorbed.
            if "NoneType" not in str(exc):
                raise
            lab = np.zeros(plane.shape, dtype=np.int64)

    if lab.ndim == 3:
        # Quirk 2: the no-cells path returns the (C, H, W) tensor shape. Any genuine
        # per-channel stack would be identical across slots, so folding by max is safe
        # and keeps the all-zero case zero.
        lab = lab.max(axis=0) if lab.shape[0] <= 3 else lab.max(axis=-1)
    if lab.shape != plane.shape:
        raise ValueError(f"CellSAM returned a {lab.shape} mask for a {plane.shape} "
                         f"plane — refusing to guess the alignment")
    return relabel_contiguous(lab.astype(np.int64, copy=False))


def _reset_model_cache() -> None:
    """Drop the singleton — for tests that swap the backing module."""
    global _cellsam_model, _cellsam_key
    _cellsam_model, _cellsam_key = None, None
