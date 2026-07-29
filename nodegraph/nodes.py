"""The node port — spec + compute pairs the engine runs (nodegraph v2, Phase 3).

A *node* is a :class:`~nodegraph.registry.NodeSpec` (registered in ``NODES``) plus a
``compute(ctx)`` function (collected in :data:`COMPUTES`, keyed by ``op_key``). The
:class:`~nodegraph.engine.Engine` looks up the compute by ``op_key``; a compute reads
its upstream :class:`~nodegraph.dataset.Dataset` inputs, resolves its metadata-intelligent
params from the incoming envelope, honors the 2D/3D lever, and returns a new Dataset.

This module establishes the node-port **conventions** and ships the flagship vertical
slice — the directive A+B showcase (V2.03 / V2.00 §14.3):

* **Select Channel** (``channel.select``) — the deconvolve prerequisite (H12): a lazy
  channel-subset view + the ``channel_select`` metadata transform.
* **Deconvolve** (``enhance.deconvolve``) — the two-mode metadata-intelligent PSF node:
  its PSF is *derived from optics metadata* (NA, emission λ, pixel/z size) — a 2D lateral
  PSF in 2D mode, a 3D anisotropic PSF in 3D mode — and Richardson–Lucy runs on the
  already-installed scikit-image backend (a GPU backend like RedLionfish is a later swap).

Conventions: a compute returns a ``Dataset``; a realized image is wrapped in an
:class:`~nodegraph.provider.ArrayProvider`; calibration is read through ``ctx.calib``
(recorded → memo-fenced); physical params convert to pixels via :func:`to_pixels_v2`
(incl. the ``um_axial`` axial unit, V2.03 §2 A5). skimage is **lazily imported**.
Qt-free.
"""
from __future__ import annotations

import itertools
import warnings
from collections import defaultdict
from dataclasses import replace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import AXIS_ORDER, Domain, axes_of, is_lattice
from nodegraph.engine import Compute, EvalContext
from nodegraph.field import (
    Attr as FieldAttr, BinOp as FieldBinOp, Const as FieldConst,
    FieldCache, FieldContext, Input as FieldInput, UnaryOp as FieldUnaryOp,
    Where as FieldWhere,
)
from nodegraph.memo import digest as _digest
from nodegraph.provider import ArrayProvider, TileProvider
from nodegraph.registry import (
    DimMode, Granularity, InBool, InDataset, InFloat, InInt, InString, Mode, OutDataset,
    SocketSpec, define_node,
)
from nodegraph.streaming import (
    MapComputeProvider, PlaneRealizeProvider, TReduceProvider, VolumeComputeProvider,
    WindowView, ZReduceProvider, stream_fp,
)
from nodegraph.metadata import (
    bit_depth_after_sum as _bit_depth_after_sum,
    channel_select as _meta_channel_select,
    parse_channels,
    crop as _meta_crop,
    frame_slice as _meta_frame_slice,
    resample as _meta_resample,
    stack_time as _meta_stack_time,
    value_rescaled as _meta_value_rescaled,
    z_project as _meta_z_project,
)
from nodegraph.reducers import reduce as _reduce
from nodegraph.structure import (
    StructureTable, label_components, point_table, seeded_watershed,
)
from nodegraph.bridges import voxel_to_label


def _each_plane(ax: AxisSizes):
    """Iterate (m,t,z,c) over a dataset's acquisition axes."""
    return itertools.product(range(ax.m), range(ax.t), range(ax.z), range(ax.c))


def _each_plane_p(ctx: EvalContext, ax: AxisSizes, note: str = ""):
    """:func:`_each_plane` that reports **per-node progress** as it goes (``done`` counted
    on completion of each plane, so the bar reflects finished work, not started work).

    Use in an **eager** compute — one that realizes planes inside the call. A compute
    returning a lazy provider must not report a fraction it isn't paying: an unreported
    node is how the UI knows the cost is deferred to read time (see
    :data:`nodegraph.engine.Observer`)."""
    n = ax.m * ax.t * ax.z * ax.c
    ctx.progress(0, n, note)
    for i, unit in enumerate(_each_plane(ax)):
        yield unit
        ctx.progress(i + 1, n, note)


def _each_volume_p(ctx: EvalContext, ax: AxisSizes, note: str = ""):
    """As :func:`_each_plane_p`, but over ``(m, t, c)`` volumes — the unit of a
    WHOLE_VOLUME compute."""
    n = ax.m * ax.t * ax.c
    ctx.progress(0, n, note)
    for i, unit in enumerate(itertools.product(range(ax.m), range(ax.t), range(ax.c))):
        yield unit
        ctx.progress(i + 1, n, note)

#: op_key → compute(ctx) — the engine's compute lookup for the ported catalog.
COMPUTES: Dict[str, Compute] = {}


def register_node(compute: Compute, **spec_kwargs: Any):
    """Register a node: build+register its :class:`NodeSpec` and record its compute."""
    spec = define_node(**spec_kwargs)
    COMPUTES[spec.op_key] = compute
    return spec


# ── metadata-intelligent unit conversion (V2.03 §2 A5) ───────────────────────

def to_pixels_v2(value: float, unit: str, *, pixel_size_um: Optional[float] = None,
                 z_step_um: Optional[float] = None, dt_s: Optional[float] = None) -> float:
    """Convert a physical ``value`` to pixels/frames. Extends the V1.91 vocabulary with
    ``um_axial`` (÷ ``z_step_um``) for anisotropic 3D kernels (V2.03 §2 A5). A missing
    calibration degrades to a 1:1 factor (the caller decides whether that is acceptable)."""
    u = (unit or "").lower()
    if u in ("", "px"):
        return value
    if u == "um":
        return value / (pixel_size_um or 1.0)
    if u == "um_axial":
        return value / (z_step_um or 1.0)
    if u == "nm":
        return (value / 1000.0) / (pixel_size_um or 1.0)
    if u == "s":
        return value / (dt_s or 1.0)
    return value


# ── metadata-intelligent PSF (the directive A showcase) ───────────────────────

def diffraction_sigmas(emission_nm: Optional[float], na: Optional[float],
                       pixel_size_um: Optional[float], z_step_um: Optional[float],
                       is_3d: bool) -> Tuple[float, ...]:
    """Gaussian-approximation PSF sigmas **derived from optics metadata**: lateral
    ``σ_xy ≈ 0.21·λ/NA`` and axial ``σ_z ≈ 0.66·λ·n/NA²`` (n≈1.5 immersion), converted
    to pixels via ``pixel_size_um`` / ``z_step_um``. Returns ``(σ_y,σ_x)`` in 2D or
    ``(σ_z,σ_y,σ_x)`` in 3D. (A Gaussian PSF is the portable default; a Gibson–Lanni /
    measured PSF is a backend swap behind the same derived sampling.)"""
    lam_um = (emission_nm or 520.0) / 1000.0
    na = na or 1.4
    sxy_um = 0.21 * lam_um / na
    sxy = sxy_um / (pixel_size_um or 0.1)
    if not is_3d:
        return (sxy, sxy)
    sz_um = 0.66 * lam_um * 1.5 / (na * na)
    sz = sz_um / (z_step_um or 0.5)
    return (sz, sxy, sxy)


def gaussian_psf(sigmas: Sequence[float], *, radius_factor: float = 3.0) -> np.ndarray:
    """A normalized n-D Gaussian kernel with the given per-axis ``sigmas`` (pixels)."""
    sig = tuple(max(0.5, float(s)) for s in sigmas)
    radii = [max(1, int(round(radius_factor * s))) for s in sig]
    grids = np.meshgrid(*[np.arange(-r, r + 1) for r in radii], indexing="ij")
    g = np.ones_like(grids[0], dtype=float)
    for coord, s in zip(grids, sig):
        g = g * np.exp(-(coord.astype(float) ** 2) / (2.0 * s * s))
    total = g.sum()
    return g / total if total else g


# ── Select Channel (H12 — the deconvolve prerequisite) ────────────────────────

class _ChannelView(TileProvider):
    """A lazy channel-subset view over another provider (Select Channel output)."""

    def __init__(self, base: TileProvider, channels: Sequence[int]) -> None:
        self._base = base
        self._ch = [int(c) for c in channels]
        self.tile = base.tile
        self.levels = base.levels
        self.axes = replace(base.axes, c=len(self._ch))
        self.depth = getattr(base, "depth", 0) + 1
        self.cum_halo = getattr(base, "cum_halo", 0)
        # flat digest, computed once (a nested-tuple fp recurses in _canon and goes
        # quadratic/overflows on deep unrolled chains — C1 / V2.04 §6b)
        self._fp = _digest("channelview", base.fingerprint(), tuple(self._ch))

    def level_axes(self, level: int) -> AxisSizes:
        return replace(self._base.level_axes(level), c=len(self._ch))

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        return self._base.read_region(level, m, t, z, self._ch[c], y0, y1, x0, x1)

    def fingerprint(self) -> tuple:
        return ("channelview", self._fp)


class _FrameView(TileProvider):
    """A lazy single-timepoint (t==1) view of one frame of another provider — the
    per-frame-T slice (zone.frame). Reads always resolve to the fixed source frame."""

    def __init__(self, base: TileProvider, frame: int) -> None:
        self._base = base
        self._t = int(frame)
        self.tile = base.tile
        self.levels = base.levels
        self.axes = replace(base.axes, t=1)
        self.depth = getattr(base, "depth", 0) + 1
        self.cum_halo = getattr(base, "cum_halo", 0)
        self._fp = _digest("frameview", base.fingerprint(), self._t)   # flat (C1)

    def level_axes(self, level: int) -> AxisSizes:
        return replace(self._base.level_axes(level), t=1)

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        return self._base.read_region(level, m, self._t, z, c, y0, y1, x0, x1)

    def fingerprint(self) -> tuple:
        return ("frameview", self._fp)


def _compute_select_channel(ctx: EvalContext) -> Dataset:
    ds: Dataset = ctx.inputs[0]
    # parse_channels is SHARED with the `channel_select` meta_transform so the predicted
    # envelope and the produced payload cannot drift (build-node-v2 §2).
    channels = list(parse_channels(ctx.params.get("channels")) or range(ds.axes.c))
    channels = [c for c in channels if 0 <= c < ds.axes.c]
    new_axes = replace(ds.axes, c=len(channels))
    out = replace(ds, axes=new_axes)
    if ds.image is not None:
        out = out.with_image(_ChannelView(ds.image, channels))
    # calibration follows the selection in lockstep: the `channel_select` meta_transform
    # already subset the emission list in the envelope (single source of truth), so mirror
    # THAT onto the payload via ctx.calib (recorded → memo-fenced; strict-read safe) rather
    # than re-subsetting the raw input metadata.
    emis = ctx.calib("channel_emission_nm")
    if isinstance(emis, (list, tuple)):
        out = out.with_metadata(channel_emission_nm=list(emis))
    return out.reshaped_axes(new_axes)


register_node(
    _compute_select_channel,
    op_key="channel.select", label="Select Channel", category="channel",
    # The node is palette-visible (it is NOT in nodelab_v2.scene.HIDDEN_OP_PREFIXES), so
    # a user can drag it onto the canvas — but `channels` had no socket, leaving them a
    # card with no controls that silently passes every channel through. The GUI's
    # per-channel tap materializer writes a LIST here; a user types "0,2".
    inputs=[InDataset(),
            InString("channels", "Channels", field=False, default="")],
    outputs=[OutDataset()],
    granularity=Granularity.TILEABLE, meta_transform=_meta_channel_select,
    description="Subset/reorder the channel axis (metadata follows in lockstep).",
)


def _compute_split_channels(ctx: EvalContext) -> Dataset:
    """Split Channels — a domain-transparent pass-through of the full multi-channel
    bundle. Its single real ``out`` socket carries the input unchanged; the GUI adds
    one synthetic per-channel output socket (``ch0…chN-1``), and each wired per-channel
    tap is materialized into a ``channel.select`` at graph-build time (the engine is
    one-payload-per-node, so distinct per-channel payloads come from real select taps,
    not from this node). No calibration/axes change here (TILEABLE, no meta_transform)."""
    return ctx.inputs[0]


register_node(
    _compute_split_channels,
    op_key="channel.split", label="Split Channels", category="channel",
    inputs=[InDataset()], outputs=[OutDataset("out")],
    granularity=Granularity.TILEABLE,
    description="Fan a multi-channel Dataset out into per-channel outputs (each a "
                "single-channel Dataset); the full bundle also passes through 'out'.",
)


def _compute_reroute(ctx: EvalContext) -> Dataset:
    """Reroute — an identity pass-through of the Dataset on the wire (Blender's reroute
    node). Purely a wire-routing convenience for the GUI: it returns its input unchanged,
    changes no axes/calibration (TILEABLE, no meta_transform, identity envelope), so the
    engine and memo treat it as a transparent hop. Hidden from the palette (``rr.``
    prefix); created by double-clicking a wire in the canvas."""
    return ctx.inputs[0]


register_node(
    _compute_reroute,
    op_key="rr.reroute", label="Reroute", category="general",
    inputs=[InDataset()], outputs=[OutDataset()],
    granularity=Granularity.TILEABLE,
    description="Identity pass-through used to route wires cleanly on the canvas.",
)


# ── Deconvolve (the two-mode metadata-intelligent PSF flagship) ───────────────

def _rl(image: np.ndarray, psf: np.ndarray, iters: int) -> np.ndarray:
    from skimage.restoration import richardson_lucy
    mx = float(image.max()) or 1.0
    out = richardson_lucy(image / mx, psf, num_iter=iters, clip=False)
    return out * mx


def _compute_deconvolve(ctx: EvalContext) -> Dataset:
    ds: Dataset = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("deconvolve needs an image provider on its input Dataset")
    ax = prov.axes
    is_3d = ctx.granularity is Granularity.WHOLE_VOLUME
    iters = int(ctx.params.get("iterations", 10))
    px = ctx.calib("pixel_size_um") or 0.1
    # C8 taught this node to honour an `emission_nm`/`na` override but left `z_step_um`
    # behind: the 3D-only socket declares `derive="z_step_um"`, so an UNSET socket already
    # resolves to the calibration — but the compute read the calibration DIRECTLY, so a
    # SET socket was silently dropped (a live control the kernel ignores, which the node
    # charter forbids). The calib read stays unconditional so the memo fence on z_step_um
    # survives in the common unset case; an override then wins and re-keys through params.
    zs = ctx.calib("z_step_um") or 0.5
    _zs_override = ctx.params.get("z_step_um")
    if _zs_override not in (None, ""):
        zs = float(_zs_override)
    # Resolve EVERY per-channel PSF eagerly: a lazy closure must not call ctx at tile-pull
    # time (the ReadContext freezes when compute returns — C1 / V2.04 §6b), and the reads
    # recorded here fold into the streaming fingerprint below. C8/H12: emission λ and NA
    # resolve PER CHANNEL via ctx.channel — a user override of the socket wins (one value,
    # all channels), else each channel's own emission (derive "emission_nm or 520") and NA.
    psfs = {c: gaussian_psf(diffraction_sigmas(ctx.channel(c).param("emission_nm"),
                                               ctx.channel(c).param("na"),
                                               px, zs, is_3d))
            for c in range(ax.c)}
    cache = ctx.tiles
    unit_bytes = (ax.z if is_3d else 1) * ax.y * ax.x * 8
    if cache is None or unit_bytes > cache.budget // 2:   # pre-C1 eager fallback
        out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    if is_3d:
                        vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
                        out[m, t, :, c] = _rl(vol.astype(float), psfs[c], iters)
                    else:
                        for z in range(ax.z):
                            plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                            out[m, t, z, c] = _rl(plane.astype(float), psfs[c], iters)
        return ds.with_image(ArrayProvider(out))
    fp = stream_fp("map", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
    if is_3d:                                    # lazy per-(m,t,c) volume (RL is global)
        return ds.with_image(VolumeComputeProvider(
            prov, lambda v, m, t, c: _rl(v, psfs[c], iters), fp=fp, cache=cache))
    return ds.with_image(MapComputeProvider(     # lazy per-plane (iterative solver)
        prov, lambda a, m, t, z, c, gy0, gy1, gx0, gx1: _rl(a, psfs[c], iters),
        unit="plane", fp=fp, cache=cache))


register_node(
    _compute_deconvolve,
    op_key="enhance.deconvolve", label="Deconvolve", category="enhancement",
    inputs=[
        InDataset(),
        InFloat("na", "NA", unit="", field=True, derive="na or 1.4"),
        InFloat("emission_nm", "Emission λ", unit="nm", field=True,
                derive="emission_nm or 520"),
        # anisotropic axial sampling is 3D-only (paired-float pattern, V2.03 §3 aniso)
        InFloat("z_step_um", "Z step", unit="um_axial", field=True, derive="z_step_um",
                available_in={"dim": frozenset({"3D"})}),
        InInt("iterations", "Iterations", default=10, field=False),
    ],
    outputs=[OutDataset()],
    modes=[DimMode()],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})},
    supports_2d=True, supports_true_3d=True,
    description="Richardson–Lucy deconvolution with a PSF derived from optics metadata "
                "(NA, emission λ, pixel/z size); 2D lateral PSF vs 3D anisotropic PSF.",
)


# ── enhancement filters ───────────────────────────────────────────────────────

def _compute_gamma(ctx: EvalContext) -> Dataset:
    """γ correction, normalized per plane. The power law is pointwise but the
    ``plane.max()`` normalization is a **plane-global statistic** — declared
    ``WHOLE_PLANE`` so C1 streams it at the plane unit, never per tile (a tile would
    normalize by its local max — the misdeclared-TILEABLE trap, V2.04 §7)."""
    ds = ctx.inputs[0]
    g = float(ctx.params.get("gamma", 1.0))

    def plane(a: np.ndarray) -> np.ndarray:
        mx = float(a.max()) or 1.0
        return (a / mx) ** g * mx

    return _map_image(ctx, ds, plane_fn=plane)


register_node(
    _compute_gamma, op_key="enhance.gamma", label="Gamma", category="enhancement",
    inputs=[InDataset(), InFloat("gamma", "Gamma", default=1.0, field=True)],
    outputs=[OutDataset()], granularity=Granularity.WHOLE_PLANE,
    kernel_axes=frozenset(),
    description="Power-law γ, normalized per plane (plane-global max → plane unit).",
)


def _compute_gaussian(ctx: EvalContext) -> Dataset:
    """Gaussian blur; a metadata-intelligent σ (µm → px). The 2D/3D lever picks a
    per-plane 2D blur vs an anisotropic 3D blur (σ_z from the axial unit). Streams
    per tile in 2D with the probe-verified tight halo ``int(4σ+0.5)`` (scipy
    ``truncate=4.0``), per lazy volume in 3D (C1 / V2.04)."""
    from scipy.ndimage import gaussian_filter
    ds = ctx.inputs[0]
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    sigma_um = float(ctx.params.get("sigma", 0.5))
    sxy = to_pixels_v2(sigma_um, "um", pixel_size_um=px)
    sz = to_pixels_v2(float(ctx.params.get("sigma_z", sigma_um)),
                      "um_axial", z_step_um=zs)
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: gaussian_filter(a, sigma=sxy),
        volume_fn=lambda v: gaussian_filter(v, sigma=(sz, sxy, sxy)),
        halo=int(4.0 * sxy + 0.5))


register_node(
    _compute_gaussian, op_key="enhance.gaussian", label="Gaussian Blur",
    category="enhancement",
    inputs=[
        InDataset(),
        InFloat("sigma", "Sigma", unit="um", field=True, default=0.5, kernel_param=True),
        InFloat("sigma_z", "Sigma Z", unit="um_axial", field=True, default=0.5,
                available_in={"dim": frozenset({"3D"})}, kernel_param=True),
    ],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity={"2D": Granularity.TILEABLE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})},
    description="Gaussian blur with a metadata-intelligent σ (µm); 2D per-plane vs 3D.",
)


# ── analysis: threshold → label → measure ─────────────────────────────────────

#: histogram-based global threshold methods (skimage.filters.threshold_*).
_THRESHOLD_METHODS = ("fixed", "otsu", "li", "yen", "triangle", "mean")


#: the Field IR node classes (``field.Field`` is a typing Union — not isinstance-able).
_FIELD_TYPES = (FieldConst, FieldAttr, FieldInput, FieldBinOp, FieldUnaryOp, FieldWhere)


def _compute_threshold(ctx: EvalContext) -> Dataset:
    """Threshold the image to a binary **mask** (a Voxel-domain integer attribute).
    Pointwise apply is nD-agnostic (no lever). ``method`` = ``fixed`` (the ``threshold``
    param) or a histogram method (otsu/li/yen/triangle/mean) whose level is derived once
    from the whole image. A **Field** wired into the ``threshold`` socket thresholds
    per voxel: it is evaluated per plane through the engine's FieldCache with a
    **windowed** FieldContext (C1 / V2.04 §4 — same-domain Attr layers slice to the
    window; the field memo token is the full unit address)."""
    ds = ctx.inputs[0]
    prov = ds.image
    ax = prov.axes
    method = ctx.params.get("__modes__", {}).get("method", "fixed")
    # unset ⇒ the socket's derive: mid-range of the CURRENT declared bit depth, or 0.5
    # when there is none (post-Normalize [0,1] data). channel(0) because the depth is
    # channel-independent; a user-set value overrides, as always.
    thr = float(ctx.channel(0).param("threshold", 0.5))
    thr_field = ctx.input("threshold")
    use_field = isinstance(thr_field, _FIELD_TYPES) and method == "fixed"
    whole = None
    if method != "fixed":
        import skimage.filters as skf
        fn = {"otsu": skf.threshold_otsu, "li": skf.threshold_li,
              "yen": skf.threshold_yen, "triangle": skf.threshold_triangle,
              "mean": skf.threshold_mean}[method]
        whole = np.stack([prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                          for m, t, z, c in _each_plane(ax)]).astype(float)
        thr = float(fn(whole.ravel()))              # 1-D intensities (skimage RGB-shape guard)
    fc = ctx.fields if ctx.fields is not None else FieldCache()
    mask = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.int64)
    for i, (m, t, z, c) in enumerate(_each_plane_p(ctx, ax, "thresholding")):
        # the stat pass already realized every plane — don't pull the lazy chain twice
        plane = whole[i] if whole is not None \
            else prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
        if use_field:
            win = {"m": (m, m + 1), "t": (t, t + 1), "z": (z, z + 1), "c": (c, c + 1)}
            # token = the FULL unit address incl. the input provider's fp (V2.04 §6b):
            # the same field expression consumed by two threshold nodes over different
            # geometry must never share a cached materialization.
            val = fc.evaluate(
                thr_field,
                FieldContext(ds, Domain.VOXEL, ax, window=win),
                token=("thr", prov.fingerprint(), m, t, z, c))
            mask[m, t, z, c] = (plane > np.asarray(val).reshape(ax.y, ax.x)
                                ).astype(np.int64)
        else:
            mask[m, t, z, c] = (plane > thr).astype(np.int64)
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), mask)


register_node(
    _compute_threshold, op_key="analysis.threshold", label="Threshold",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            # `fixed`-only: every histogram method DERIVES the cut from the data and
            # ignores this socket (compute: `use_field`/`thr` are overwritten unless
            # method == "fixed"), so showing it under otsu/li/… is a lie.
            # a FIXED level is in the image's own units, so its default must follow the
            # declared intensity scale: mid-range of the current bit depth (2047.5 on
            # 12-bit, 32767.5 on 16-bit), falling back to 0.5 when there is no declared
            # integer scale — i.e. exactly after a Normalize dropped it (wire-node-v2 §7c).
            InFloat("threshold", "Threshold", default=0.5, field=True,
                    derive="((2**bit_depth - 1)/2) if bit_depth else 0.5",
                    available_in={"method": frozenset({"fixed"})}),
            InString("name", "Output layer", field=False, default="mask",
                     layer_out=(Domain.VOXEL,))],
    outputs=[OutDataset()], modes=[Mode("method", list(_THRESHOLD_METHODS))],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    description="Binarize to a Voxel mask — fixed, or a histogram method "
                "(otsu/li/yen/triangle/mean); pointwise, dimension-agnostic. The "
                "threshold socket shows only under `fixed` (the others self-derive).",
)


def _compute_multiotsu(ctx: EvalContext) -> Dataset:
    """Multi-level Otsu → a **class-index** Voxel raster (0..K-1) for multi-population
    segmentation (K = ``classes``). Thresholds are derived per (m,t,c) over that volume's
    histogram; a volume with too few distinct levels degrades to all-class-0 (no crash)."""
    from skimage.filters import threshold_multiotsu
    ds = ctx.inputs[0]
    prov = ds.image
    ax = prov.axes
    k = max(2, int(ctx.params.get("classes", 3)))
    out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.int64)
    for m, t, c in _each_volume_p(ctx, ax, "multi-otsu"):
        vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x).astype(float)
        try:
            th = threshold_multiotsu(vol.ravel(), classes=k)
        except ValueError:
            continue                               # too few distinct levels → all class 0
        out[m, t, :, c] = np.digitize(vol, th)
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), out)


register_node(
    _compute_multiotsu, op_key="analysis.multiotsu", label="Multi-Otsu",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(), InInt("classes", "Classes", default=3, field=False),
            InString("name", "Output layer", field=False, default="classes",
                     layer_out=(Domain.VOXEL,))],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset(),
    description="Multi-level Otsu → a class-index Voxel raster (0..K-1); needs the whole "
                "volume histogram.",
)


def _compute_threshold_local(ctx: EvalContext) -> Dataset:
    """Adaptive (local) threshold → a Voxel mask — robust to uneven illumination: each
    pixel is compared to a Gaussian-weighted local mean over a ``block_size`` (µm)
    neighbourhood minus ``offset``. Per-plane 2D (WHOLE_PLANE)."""
    from skimage.filters import threshold_local
    ds = ctx.inputs[0]
    prov = ds.image
    ax = prov.axes
    px = ctx.calib("pixel_size_um") or 0.1
    block = max(3, int(round(float(ctx.params.get("block_size", 1.5)) / px)) | 1)  # odd ≥3
    offset = float(ctx.params.get("offset", 0.0))
    mask = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.int64)
    for m, t, z, c in _each_plane_p(ctx, ax, "local threshold"):
        plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x).astype(float)
        loc = threshold_local(plane, block_size=min(block, _odd_leq(plane.shape)),
                              offset=offset)
        mask[m, t, z, c] = (plane > loc).astype(np.int64)
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), mask)


def _odd_leq(shape: Tuple[int, ...]) -> int:
    """Largest odd block edge that fits a plane (threshold_local needs block ≤ dims)."""
    m = max(3, min(shape))
    return m if m % 2 else m - 1


register_node(
    _compute_threshold_local, op_key="analysis.threshold_local", label="Local Threshold",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            InFloat("block_size", "Block size", unit="um", field=True, default=1.5),
            InFloat("offset", "Offset", unit="", field=True, default=0.0),
            InString("name", "Output layer", field=False, default="mask",
                     layer_out=(Domain.VOXEL,))],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_PLANE, kernel_axes=frozenset({"y", "x"}),
    description="Adaptive local threshold → a Voxel mask (uneven-illumination robust); "
                "per-plane 2D, block size in µm.")


def _compute_label(ctx: EvalContext) -> Dataset:
    """Connected-components label a mask into Label regions + a label raster. 2D
    labels each plane (4/8-conn); 3D labels the volume (6/26-conn). Region ids are
    global-unique; per-region attributes are stored as Label-domain layers."""
    ds = ctx.inputs[0]
    ax = ds.axes
    is_3d = ctx.granularity is Granularity.WHOLE_VOLUME
    conn = int(ctx.params.get("connectivity", 26 if is_3d else 8))
    mask_attr = ds.get(Domain.VOXEL, ctx.layer("mask"))
    if mask_attr is None:
        raise ValueError(f"no mask attribute {ctx.params.get('mask', 'mask')!r}")
    mask6 = mask_attr.values
    raster = np.zeros_like(mask6, dtype=np.int64)
    cols: Dict[str, list] = defaultdict(list)
    offset = 0

    def take(lab, tbl):
        nonlocal offset
        for k, v in tbl.columns.items():
            cols[k].extend(((v + offset) if k == "id" else v).tolist())
        return np.where(lab > 0, lab + offset, 0), offset + tbl.n

    for m in range(ax.m):
        for t in range(ax.t):
            for c in range(ax.c):
                if is_3d:
                    lab, tbl = label_components(mask6[m, t, :, c], conn, m=m, t=t, c=c)
                    raster[m, t, :, c], offset = take(lab, tbl)
                else:
                    for z in range(ax.z):
                        lab, tbl = label_components(mask6[m, t, z, c], conn,
                                                    m=m, t=t, c=c, z_index=z)
                        raster[m, t, z, c], offset = take(lab, tbl)
    layer = ctx.layer("name")
    out = ds.with_layer(Domain.VOXEL, layer, raster)
    if cols.get("id"):
        merged = StructureTable(
            Domain.LABEL, {k: np.array(v) for k, v in cols.items()},
            layer=layer, z_kind=("subpixel" if is_3d else "plane_index"))
        out = out.with_structure(merged)
    return out


register_node(
    _compute_label, op_key="analysis.label", label="Connected Components",
    reads_domains=frozenset({Domain.VOXEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),   # emits a label RASTER too
    category="analysis",
    inputs=[InDataset(),
            InString("mask", "Mask layer", field=False, default="mask",
                     layer_in=Domain.VOXEL),
            InInt("connectivity", "Connectivity", field=False),    # default per dim
            # one name, TWO domains: a Voxel raster AND the Label table
            InString("name", "Output layer", field=False, default="labels",
                     layer_out=(Domain.VOXEL, Domain.LABEL))],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})},
    description="Label a mask into connected regions (2D 4/8-conn vs 3D 6/18/26-conn); "
                "connectivity defaults to 8 (2D) / 26 (3D), overridable.",
)


#: measure stat → its Label column name (Voxel→Label bridge reducers).
_MEASURE_COLUMNS = {
    "mean": "mean_intensity", "max": "max_intensity", "min": "min_intensity",
    "sum": "total_intensity", "median": "median_intensity", "count": "area",
}


# ── edit-time layer catalog: the producers a `layer_out` socket cannot describe ──
#
# `SocketSpec.layer_out` covers any node that NAMES its output layer through a socket.
# These five do not: two write literal Frame layers with no socket at all, two derive the
# name from ANOTHER param, and one writes into the layer its READ socket names. Each is
# `NodeSpec.extra_layers` — `(params, modes) -> ((Domain, name), ...)`. They run inside
# `propagate_meta` on every keystroke, so they never raise: `.get` with a default, no
# indexing, no casts (`metadata._layer_names_out` also wraps them, belt and braces).

def _layers_drift(params, modes):
    """`align.drift` / `registration.stabilize` store the estimated per-frame shift as
    two LITERAL Frame layers — no socket names them, so nothing else can see them."""
    return ((Domain.FRAME, "drift_y"), (Domain.FRAME, "drift_x"))


def _layers_extract_boundary(params, modes):
    """Output name is DERIVED: empty `name` means `f"{labels}_boundary"`."""
    src = params.get("labels") or "labels"
    return ((Domain.POINT, params.get("name") or "%s_boundary" % src),)


def _layers_accumulate_field(params, modes):
    """Output name is DERIVED: empty `name` means `f"{source}_cumulative"`."""
    src = params.get("source") or "dvc"
    return ((Domain.POINT, params.get("name") or "%s_cumulative" % src),)


def _layers_measure(params, modes):
    """`analysis.measure` has no output-name socket: it attaches its statistic columns to
    the Label table named by its READ socket, creating that (LABEL, name) pair when the
    upstream produced only a Voxel raster."""
    return ((Domain.LABEL, params.get("labels") or "labels"),)


def _measure_stats(raw) -> list:
    """The ``stats`` selector → an ordered, de-duplicated list of reducer names.

    Accepts the socket's **comma-separated string** (``"mean,max,median"``) or an
    already-split sequence from a programmatic caller, because :class:`SocketType` has no
    LIST member. Unknown names are refused HERE, with the whole menu in the message,
    rather than deep inside ``bridges._group_reduce``; a blank selector falls back to the
    historical default set so an emptied text box is never a silent no-op."""
    names = ([s.strip() for s in raw.split(",")] if isinstance(raw, str)
             else [str(s).strip() for s in (raw or ())])
    names = [s for s in names if s] or ["mean", "max", "min", "count"]
    bad = [s for s in names if s not in _MEASURE_COLUMNS]
    if bad:
        raise ValueError(f"unknown measure stat(s) {bad} — choose from "
                         f"{list(_MEASURE_COLUMNS)} (comma-separated)")
    return list(dict.fromkeys(names))          # de-dup, preserve the user's order


#: The optional **raw-intensity** Dataset socket (2026-07-28). Declared AFTER ``data`` so
#: ``data`` stays the primary — ``graph.dataset_preds`` sorts by declared socket position,
#: so calibration/domain propagation keeps flowing from the main chain no matter which edge
#: the user wired first.
def _InRaw() -> SocketSpec:
    return InDataset("raw", label="Raw")


def _intensity_provider(ctx: EvalContext, ds: Dataset):
    """The pixel source for an intensity **measurement**: the optional ``raw`` Dataset
    input when one is wired, else the node's own image. Returns ``(provider, is_raw)``.

    This is the "segment on enhanced, measure on raw" seam. The scope is deliberately
    narrow — ``raw`` overrides **measurement** only, never segmentation: the mask, the
    label raster and every threshold still come from the main input, so the node's
    geometry and its calibration env stay one consistent chain and only the numbers being
    *reported* change. (A node that thresholds raw pixels doesn't need a lever — wire the
    raw Dataset into its main input.)

    Geometry must match **exactly**. A cropped/resampled/projected raw would be read
    voxel-for-voxel against the main raster and report neighbouring objects' intensities,
    which is silent corruption, so a mismatch is refused."""
    prov = ds.image
    raw = ctx.input("raw")
    if raw is None:
        return prov, False
    rprov = getattr(raw, "image", None)
    if rprov is None:
        raise ValueError(
            "the `raw` input carries no image provider — wire an image Dataset (the "
            "unenhanced source) into it, or leave it unwired to measure the main input.")
    if prov is not None and rprov.axes != prov.axes:
        shape = lambda a: (a.m, a.t, a.z, a.c, a.y, a.x)   # AxisSizes is not iterable
        raise ValueError(
            f"the `raw` input's geometry {shape(rprov.axes)} does not match the measured "
            f"input's {shape(prov.axes)} — they are read voxel-for-voxel, so a mismatch "
            "would report the wrong regions' intensities. Wire `raw` from BEFORE any "
            "crop / resample / z-project / channel tap, or apply the same one to both "
            "branches.")
    return rprov, True


def _compute_measure(ctx: EvalContext) -> Dataset:
    """Measure per-region statistics of the image over a label raster → Label-domain
    attributes (Voxel→Label bridge, V2.00 §6). Emits one column per requested stat
    (``mean_intensity``/``max_intensity``/``min_intensity``/``total_intensity``/
    ``area`` = voxel count); ``mean`` is always present so ``mean_intensity`` stays a
    stable downstream key. Every stat shares the sorted-id order of the same raster.

    The optional **``raw``** Dataset input redirects *which pixels are measured* while the
    label raster keeps coming from the main input — the "segment on enhanced, measure on
    raw" workflow (:func:`_intensity_provider`). Unwired, it measures its own image."""
    ds = ctx.inputs[0]
    prov, on_raw = _intensity_provider(ctx, ds)
    ax = ds.axes
    layer = ctx.layer("labels")
    raster_attr = ds.get(Domain.VOXEL, layer)
    if raster_attr is None:
        raise ValueError(f"no label raster {layer!r}")
    raster6 = raster_attr.values
    if prov is None:
        raise ValueError("measure needs an image provider (on its input, or via `raw`)")
    img = np.zeros_like(raster6, dtype=float)
    # this gather REALIZES the whole lazy chain plane by plane — for a deep enhancement
    # chain it is the run's real cost, so it is worth a determinate bar.
    for m, t, z, c in _each_plane_p(ctx, ax,
                                    "reading raw planes" if on_raw else "reading planes"):
        img[m, t, z, c] = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
    stats = _measure_stats(ctx.params.get("stats", ("mean", "max", "min", "count")))
    if "mean" not in stats:
        stats = ["mean", *stats]
    cols: Dict[str, np.ndarray] = {}
    for st in stats:
        ids, values = voxel_to_label(img, raster6, st)
        cols.setdefault("id", ids)                       # same sorted ids for every stat
        cols[_MEASURE_COLUMNS.get(st, st)] = values
    # Re-emits the SAME label layer with measurement columns — carry the label's z_kind
    # provenance forward (§7b) so it is not clobbered with the StructureTable default.
    zk = ds.structure_zkind(Domain.LABEL, layer) or "subpixel"
    return ds.with_structure(StructureTable(Domain.LABEL, cols, layer=layer, z_kind=zk))


register_node(
    _compute_measure, op_key="analysis.measure", label="Measure", category="analysis",
    extra_layers=_layers_measure,
    reads_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    adds_domains=frozenset({Domain.LABEL}),
    inputs=[InDataset(), _InRaw(),
            InString("labels", "Label layer", field=False, default="labels",
                     layer_in=Domain.VOXEL),
            # comma-separated: SocketType has no LIST member. `mean` is force-added by
            # the compute, so `mean_intensity` stays a stable downstream key.
            InString("stats", "Statistics", field=False,
                     default="mean,max,min,count")],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_VOLUME,
    description="Per-label statistics of the image (mean/max/min/area) via the "
                "Voxel→Label bridge. Optional `raw` input measures THOSE pixels instead "
                "(segment on enhanced, measure on raw); labels stay from the main input.",
)


# ── ported catalog (Phase 3) ───────────────────────────────────────────────────
#
# The batch below ports the enhancement / denoise / detection / axis-changing
# families from the backend menu (node_backend_reference.md), each declaring its
# per-dim data-access footprint (``granularity``/``kernel_axes``), its metadata-
# intelligent params (``unit``/``derive``), and — for the axis-changing ones — its
# ``meta_transform`` so the edit-time MetaEnvelope pass tracks the geometry change
# (V2.03 §2 A2 / §7 directive follow-up). Backends (scipy.ndimage / skimage) are
# **lazily imported** so the core + SyntheticProvider need neither.


def _win(radius_px: float) -> int:
    """A structuring-element / kernel edge length (odd, ≥1) for a pixel ``radius``."""
    return max(1, int(round(radius_px)) * 2 + 1)


def _radius_px(ctx: EvalContext, name: str, default_um: float) -> float:
    """A lateral radius param (µm) → pixels via ``pixel_size_um`` (recorded read)."""
    r_um = float(ctx.params.get(name, default_um))
    return to_pixels_v2(r_um, "um", pixel_size_um=ctx.calib("pixel_size_um") or 0.1)


def _radius_z_px(ctx: EvalContext, name: str, default_um: float) -> float:
    """The axial radius (``<name>_z``, µm_axial → px via ``z_step_um``); falls back to
    the lateral param when the 3D-only axial socket is unset (paired-float pattern)."""
    r_um = float(ctx.params.get(name + "_z", ctx.params.get(name, default_um)))
    return to_pixels_v2(r_um, "um_axial", z_step_um=ctx.calib("z_step_um") or 0.5)


def _kernel_field_varies(ctx: EvalContext) -> bool:
    """True iff a ``kernel_param`` socket is wired to a **non-Const** Field — a spatially
    varying kernel (the gate defers to `_FIELD_TYPES` defined below, so it is checked
    lazily). A varying kernel breaks a tile's translation-invariance and its halo sizing
    (the halo assumes one radius for the whole plane), so the consumer must stream at the
    WHOLE_PLANE unit, never tiled (V2.04 §6b Fork B — the kernel-param field gate). A
    Const field is spatially uniform, so it stays tileable."""
    spec = ctx.spec
    if spec is None:
        return False
    for s in getattr(spec, "inputs", ()):
        if not getattr(s, "kernel_param", False):
            continue
        payload = ctx.input(s.name)
        if isinstance(payload, _FIELD_TYPES) and not isinstance(payload, FieldConst):
            return True
    return False


def _map_image(ctx: EvalContext, ds: Dataset,
               plane_fn: Callable[[np.ndarray], np.ndarray],
               volume_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               halo: int = 0) -> Dataset:
    """Apply a spatial op across the image — **lazily** (C1 / V2.04): the returned
    Dataset carries a streaming compute provider instead of a realized array.

    In a 2D footprint ``plane_fn`` runs per ``(Y,X)`` unit — per canonical **tile**
    (window+``halo`` read from the base, overlap-recompute) when the resolved
    granularity is ``TILEABLE``, per whole **plane** when it is ``WHOLE_PLANE`` (an op
    with a plane-global statistic/solver must declare WHOLE_PLANE — a tile would see
    the wrong population). In a 3D (``WHOLE_VOLUME``) footprint ``volume_fn`` runs on
    the ``(Z,Y,X)`` volume per ``(m,t,c)``, computed lazily per touched volume. A node
    resolving to ``WHOLE_VOLUME`` MUST supply a ``volume_fn`` — a *stack-of-2D* node
    keeps its 3D granularity at ``WHOLE_PLANE`` so it stays on the plane path (H15).

    ``halo`` is the op's full **influence radius** in px (V2.04 §2: gaussian
    ``int(4σ+0.5)``, median ``w//2``, open/close/tophat ``2·(w//2)``, DoG from the
    larger σ). Falls back to the pre-C1 eager whole-realize when there is no engine
    cache (a bare ``EvalContext``), an unsupported granularity, or an oversize unit
    (> half the cache budget — V2.04 §6b pin/bypass policy).
    """
    prov = ds.image
    if prov is None:
        raise ValueError(f"{ctx.op_key} needs an image provider on its input Dataset")
    ax = prov.axes
    volumetric = ctx.is_volume
    if volumetric and volume_fn is None:
        raise ValueError(f"{ctx.op_key}: resolved to WHOLE_VOLUME but no volume op given")
    cache = ctx.tiles
    # kernel-param field gate (V2.04 §6b Fork B): a non-Const Field on a kernel_param
    # socket varies the kernel spatially → drop TILEABLE to the plane unit (a tile's halo
    # assumes one radius for the whole plane; a varying kernel breaks that + translation
    # invariance). Const/absent ⇒ genuinely tileable.
    tileable = (ctx.granularity is Granularity.TILEABLE
                and not _kernel_field_varies(ctx))
    # size the oversize-bypass check by the EFFECTIVE lazy unit: a true tile unit is
    # tiny regardless of plane size (review 2026-07-22 — sizing tileable ops by plane
    # bytes forced whole-series eager realization exactly when memory is scarce)
    if volumetric:
        unit_bytes = ax.z * ax.y * ax.x * 8
    elif tileable and 2 * (getattr(prov, "cum_halo", 0) + halo) < prov.tile:
        w = prov.tile + 2 * halo
        unit_bytes = w * w * 8
    else:                                        # plane unit (declared or fence-promoted)
        unit_bytes = ax.y * ax.x * 8
    lazy = (cache is not None
            and ctx.granularity in (Granularity.TILEABLE, Granularity.WHOLE_PLANE,
                                    Granularity.WHOLE_VOLUME)
            and unit_bytes <= cache.budget // 2)
    if not lazy:                                  # pre-C1 eager whole-realize path
        out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    if volumetric:
                        vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
                        out[m, t, :, c] = volume_fn(vol.astype(float))
                    else:
                        for z in range(ax.z):
                            plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                            out[m, t, z, c] = plane_fn(plane.astype(float))
        return ds.with_image(ArrayProvider(out))
    fp = stream_fp("map", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
    if volumetric:
        return ds.with_image(VolumeComputeProvider(
            prov, lambda v, m, t, c: volume_fn(v), fp=fp, cache=cache))
    unit = "tile" if tileable else "plane"
    return ds.with_image(MapComputeProvider(
        prov, lambda a, m, t, z, c, gy0, gy1, gx0, gx1: plane_fn(a),
        halo=halo, unit=unit, fp=fp, cache=cache))


#: the per-dim footprint every dim-lever spatial filter shares (2D per-plane, 3D volume).
_DIM_GRAN = {"2D": Granularity.TILEABLE, "3D": Granularity.WHOLE_VOLUME}
_DIM_KAX = {"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})}
#: per-dim footprint for ops with a plane/volume-GLOBAL statistic or solver (min/max
#: normalization, global σ estimate, iterative solver): they stream at the plane/
#: volume unit, never per tile — a tile would see the wrong statistical population
#: (the misdeclared-TILEABLE traps, C1 audit / V2.04 §7).
_DIM_GRAN_GLOBAL = {"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME}


def _InRadius(name: str = "radius", label: str = "Radius", default: float = 0.3):
    """A lateral radius socket + its 3D-only axial companion (``<name>_z``). Both are
    ``kernel_param`` — a non-Const Field wired here varies the kernel spatially, so the
    consumer must drop to the plane unit (the kernel-param field gate, V2.04 §6b)."""
    return [
        InFloat(name, label, unit="um", field=True, default=default, kernel_param=True),
        InFloat(f"{name}_z", f"{label} Z", unit="um_axial", field=True, default=default,
                available_in={"dim": frozenset({"3D"})}, kernel_param=True),
    ]


# ── Median ─────────────────────────────────────────────────────────────────────

def _compute_median(ctx: EvalContext) -> Dataset:
    from scipy.ndimage import median_filter
    ds = ctx.inputs[0]
    wy = _win(_radius_px(ctx, "radius", 0.3))
    wz = _win(_radius_z_px(ctx, "radius", 0.3)) if ctx.is_volume else wy
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: median_filter(a, size=(wy, wy)),
        volume_fn=lambda v: median_filter(v, size=(wz, wy, wy)),
        halo=wy // 2)


register_node(
    _compute_median, op_key="enhance.median", label="Median", category="enhancement",
    inputs=[InDataset(), *_InRadius()], outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN, kernel_axes=_DIM_KAX,
    description="Edge-preserving median filter; 2D per-plane vs anisotropic 3D window.")


# ── Morphology (erode / dilate / open / close) ──────────────────────────────────

def _compute_morphology(ctx: EvalContext) -> Dataset:
    from scipy import ndimage as ndi
    ds = ctx.inputs[0]
    op = ctx.params.get("__modes__", {}).get("op", "open")
    fn = {"erode": ndi.grey_erosion, "dilate": ndi.grey_dilation,
          "open": ndi.grey_opening, "close": ndi.grey_closing}[op]
    wy = _win(_radius_px(ctx, "radius", 0.3))
    wz = _win(_radius_z_px(ctx, "radius", 0.3)) if ctx.is_volume else wy
    # open/close = erosion∘dilation: TWO passes each reaching w//2 → influence 2·(w//2)
    # (halo=w//2 leaves real border errors — C1 audit / V2.04 §2)
    reach = (wy // 2) if op in ("erode", "dilate") else 2 * (wy // 2)
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: fn(a, size=(wy, wy)),
        volume_fn=lambda v: fn(v, size=(wz, wy, wy)),
        halo=reach)


register_node(
    _compute_morphology, op_key="enhance.morphology", label="Morphology",
    category="enhancement",
    inputs=[InDataset(), *_InRadius()], outputs=[OutDataset()],
    modes=[DimMode(), Mode("op", ["erode", "dilate", "open", "close"], default="open")],
    granularity=_DIM_GRAN, kernel_axes=_DIM_KAX,
    description="Grayscale morphology (erode/dilate/open/close); 2D vs anisotropic 3D.")


# ── Top-Hat (white / black — background subtraction / dark-spot pop) ────────────

def _compute_tophat(ctx: EvalContext) -> Dataset:
    from scipy import ndimage as ndi
    ds = ctx.inputs[0]
    variant = ctx.params.get("__modes__", {}).get("variant", "white")
    fn = ndi.white_tophat if variant == "white" else ndi.black_tophat
    wy = _win(_radius_px(ctx, "radius", 0.5))
    wz = _win(_radius_z_px(ctx, "radius", 0.5)) if ctx.is_volume else wy
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: fn(a, size=(wy, wy)),
        volume_fn=lambda v: fn(v, size=(wz, wy, wy)),
        halo=2 * (wy // 2))          # tophat wraps an opening/closing: 2-pass reach


register_node(
    _compute_tophat, op_key="enhance.tophat", label="Top-Hat", category="enhancement",
    inputs=[InDataset(), *_InRadius(default=0.5)], outputs=[OutDataset()],
    modes=[DimMode(), Mode("variant", ["white", "black"], default="white")],
    granularity=_DIM_GRAN, kernel_axes=_DIM_KAX,
    description="White/black top-hat (flatten background / pop dark features); "
                "2D vs anisotropic 3D.")


# ── Difference of Gaussians (band-pass feature enhancement) ─────────────────────

def _compute_dog(ctx: EvalContext) -> Dataset:
    from skimage.filters import difference_of_gaussians as dog
    ds = ctx.inputs[0]
    lo = _radius_px(ctx, "low_sigma", 0.15)
    px = ctx.calib("pixel_size_um") or 0.1
    hi_um = float(ctx.params.get("high_sigma", 0.0))
    hi = (hi_um / px) if hi_um > 0 else None
    # influence radius from the LARGER σ; high_sigma 0 ⇒ skimage's implicit 1.6·low
    # (a halo from low σ alone leaves border errors — C1 audit / V2.04 §2)
    hi_eff = hi if hi is not None else 1.6 * lo
    if hi is not None and hi < lo:
        # validate eagerly: under C1 the kernel runs at TILE-PULL time, so a backend
        # ValueError would otherwise be memoized into a poisoned lazy payload and
        # surface far from the misconfigured node (review 2026-07-22)
        raise ValueError(
            f"high_sigma ({hi_um} µm → {hi:.2f} px) must be ≥ low_sigma ({lo:.2f} px)")
    halo = int(4.0 * max(lo, hi_eff) + 0.5)
    if ctx.is_volume:
        zs = ctx.calib("z_step_um") or 0.5
        loz = _radius_z_px(ctx, "low_sigma", 0.15)
        hiz = (hi_um / zs) if hi_um > 0 else None
        low3 = (loz, lo, lo)
        high3 = (hiz, hi, hi) if hi is not None else None
        return _map_image(ctx, ds, plane_fn=lambda a: dog(a, lo, hi),
                          volume_fn=lambda v: dog(v, low3, high3), halo=halo)
    return _map_image(ctx, ds, plane_fn=lambda a: dog(a, lo, hi), halo=halo)


register_node(
    _compute_dog, op_key="enhance.dog", label="Difference of Gaussians",
    category="enhancement",
    inputs=[
        InDataset(),
        InFloat("low_sigma", "Low σ", unit="um", field=True, default=0.15,
                derive="0.5*0.61*(emission_nm or 520)/(na or 1.4)/1000", kernel_param=True),
        InFloat("low_sigma_z", "Low σ Z", unit="um_axial", field=True, default=0.15,
                available_in={"dim": frozenset({"3D"})}, kernel_param=True),
        InFloat("high_sigma", "High σ", unit="um", field=True, default=0.0,
                kernel_param=True),
    ],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN, kernel_axes=_DIM_KAX,
    description="Band-pass (blob) enhancement; low σ derives from the diffraction limit; "
                "high σ 0 ⇒ auto 1.6× low. 2D per-plane vs anisotropic 3D.")


# ── Unsharp Mask (edge sharpening) ──────────────────────────────────────────────

def _compute_unsharp(ctx: EvalContext) -> Dataset:
    from skimage.filters import unsharp_mask
    ds = ctx.inputs[0]
    amount = float(ctx.params.get("amount", 1.0))
    rxy = _radius_px(ctx, "radius", 0.3)
    halo = int(4.0 * rxy + 0.5)               # gaussian blur influence (truncate=4)
    if ctx.is_volume:
        rz = _radius_z_px(ctx, "radius", 0.3)
        return _map_image(
            ctx, ds,
            plane_fn=lambda a: unsharp_mask(a, radius=rxy, amount=amount, preserve_range=True),
            volume_fn=lambda v: unsharp_mask(v, radius=(rz, rxy, rxy), amount=amount,
                                             preserve_range=True),
            halo=halo)
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: unsharp_mask(a, radius=rxy, amount=amount, preserve_range=True),
        halo=halo)


register_node(
    _compute_unsharp, op_key="enhance.unsharp", label="Unsharp Mask",
    category="enhancement",
    inputs=[InDataset(), *_InRadius(),
            InFloat("amount", "Amount", unit="", field=True, default=1.0)],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN, kernel_axes=_DIM_KAX,
    description="Unsharp-mask sharpening (blur radius in µm); 2D vs anisotropic 3D.")


# ── TV denoise (Chambolle — genuinely n-D) ──────────────────────────────────────

def _compute_tv(ctx: EvalContext) -> Dataset:
    from skimage.restoration import denoise_tv_chambolle as tv
    ds = ctx.inputs[0]
    weight = float(ctx.params.get("weight", 0.1))
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: tv(a, weight=weight),
        volume_fn=lambda v: tv(v, weight=weight))


register_node(
    _compute_tv, op_key="enhance.tv_denoise", label="TV Denoise",
    category="enhancement",
    inputs=[InDataset(), InFloat("weight", "Weight", unit="", field=True, default=0.1)],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,     # global iterative solver
    supports_2d=True, supports_true_3d=True,
    description="Total-variation (Chambolle) edge-preserving denoise — genuinely 3D in "
                "3D mode (denoise_tv_chambolle is true n-D).")


# ── Wavelet denoise (the stack-of-2D trap — declared honestly, H15) ─────────────

def _compute_wavelet(ctx: EvalContext) -> Dataset:
    from skimage.restoration import denoise_wavelet as dw
    ds = ctx.inputs[0]

    def plane(a: np.ndarray) -> np.ndarray:
        # BayesShrink estimates the noise σ from the wavelet detail coefficients; on a
        # flat OR sparse plane that estimate degenerates to 0 and the shrink divides by
        # it, returning an all-NaN plane. A blank z-slice / empty channel / sparse mask
        # is a routine input, so: short-circuit the flat case, and if the backend still
        # returns non-finite (sparse), pass the plane through — there is nothing to
        # denoise when noise cannot be estimated.
        if float(a.max()) <= float(a.min()):
            return a.copy()
        out = dw(a, rescale_sigma=True)
        return out if np.all(np.isfinite(out)) else a.copy()

    # Always stack-of-2D: even in 3D mode the footprint stays WHOLE_PLANE, so
    # `ctx.is_volume` is False and this only ever takes the per-plane path.
    return _map_image(ctx, ds, plane_fn=plane)


register_node(
    _compute_wavelet, op_key="enhance.wavelet_denoise", label="Wavelet Denoise",
    category="enhancement",
    inputs=[InDataset()], outputs=[OutDataset()], modes=[DimMode()],
    granularity=Granularity.WHOLE_PLANE,      # full-plane DWT + global σ estimate (C1)
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"y", "x"})},
    supports_2d=True, supports_true_3d=False, three_d_fallback="stack_of_2d",
    description="Wavelet (BayesShrink) denoise — stack-of-2D even in 3D mode "
                "(skimage denoise_wavelet is 2D-only); declared as such (H15).")


# ── CLAHE (local contrast) ──────────────────────────────────────────────────────

def _compute_clahe(ctx: EvalContext) -> Dataset:
    from skimage.exposure import equalize_adapthist as clahe
    ds = ctx.inputs[0]
    clip = float(ctx.params.get("clip_limit", 0.01))

    def apply(a: np.ndarray) -> np.ndarray:
        mn, mx = float(a.min()), float(a.max())
        if mx <= mn:
            return a.copy()
        ks = tuple(max(2, s // 4) for s in a.shape)      # per-axis kernel, ≥2 (3D-safe)
        eq = clahe((a - mn) / (mx - mn), kernel_size=ks, clip_limit=clip)
        return eq * (mx - mn) + mn

    return _map_image(ctx, ds, plane_fn=apply, volume_fn=apply)


register_node(
    _compute_clahe, op_key="enhance.clahe", label="CLAHE", category="enhancement",
    inputs=[InDataset(), InFloat("clip_limit", "Clip limit", unit="", field=True,
                                 default=0.01)],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,   # shape-derived grid + min/max
    description="Contrast-limited adaptive histogram equalization; 2D per-plane vs 3D.")


# ── Normalize (percentile — scope Mode, NOT the dim lever, H24) ──────────────────

def _compute_normalize(ctx: EvalContext) -> Dataset:
    """Percentile-rescale to [0,1]. The statistics ``scope`` is a **Mode**
    (plane/volume/series), never the 2D/3D lever (H24): a lever picks the compute
    *footprint*, whereas normalization scope picks the *statistical population*."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("normalize needs an image provider on its input Dataset")
    ax = prov.axes
    scope = ctx.params.get("__modes__", {}).get("scope", "plane")
    lo_p = float(ctx.params.get("low_pct", 1.0))
    hi_p = float(ctx.params.get("high_pct", 99.0))

    def rescale(block: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(block, lo_p), np.percentile(block, hi_p)
        if hi <= lo:
            return np.zeros_like(block)
        return np.clip((block - lo) / (hi - lo), 0.0, 1.0)

    def apply_lohi(block: np.ndarray, lo: float, hi: float) -> np.ndarray:
        if hi <= lo:
            return np.zeros_like(block)
        return np.clip((block - lo) / (hi - lo), 0.0, 1.0)

    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        img = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m, t, z, c in _each_plane(ax):
            img[m, t, z, c] = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
        out = np.empty_like(img)
        if scope == "plane":
            for m, t, z, c in _each_plane(ax):
                out[m, t, z, c] = rescale(img[m, t, z, c])
        elif scope == "volume":
            for m in range(ax.m):
                for t in range(ax.t):
                    for c in range(ax.c):
                        out[m, t, :, c] = rescale(img[m, t, :, c])
        else:  # series — the whole T stack per (m,c)
            for m in range(ax.m):
                for c in range(ax.c):
                    out[m, :, :, c] = rescale(img[m, :, :, c])
        return ds.with_image(ArrayProvider(out)).with_metadata(bit_depth=None)
    # C1 (V2.04 §6b sliver): eager-stat / lazy-apply, unit = the statistics scope.
    # `plane` scope is SELF-CONTAINED per plane (percentiles of the plane it normalizes),
    # so it needs no pre-stat → a per-plane MapComputeProvider. `volume` scope per (m,t,c)
    # → VolumeComputeProvider. `series` scope's stat spans T, so the (lo,hi) per (m,c) is
    # computed EAGERLY once (a percentile needs the whole population regardless) and baked
    # into a lazy per-plane apply. Only touched units compute; the eager fallback above
    # uses the SAME rescale so the bytes are identical.
    fp = stream_fp("normalize", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
    if scope == "plane":
        return ds.with_image(MapComputeProvider(
            prov, lambda a, m, t, z, c, *_: rescale(a), unit="plane", fp=fp, cache=cache)
            ).with_metadata(bit_depth=None)
    if scope == "volume":
        return ds.with_image(VolumeComputeProvider(
            prov, lambda v, m, t, c: rescale(v), fp=fp, cache=cache)
            ).with_metadata(bit_depth=None)
    lohi: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for m in range(ax.m):
        for c in range(ax.c):
            block = np.stack([prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                              for t in range(ax.t) for z in range(ax.z)]).astype(float)
            lohi[(m, c)] = (float(np.percentile(block, lo_p)),
                            float(np.percentile(block, hi_p)))
    return ds.with_image(MapComputeProvider(
        prov, lambda a, m, t, z, c, *_: apply_lohi(a, *lohi[(m, c)]),
        unit="plane", fp=fp, cache=cache)).with_metadata(bit_depth=None)


register_node(
    _compute_normalize, op_key="enhance.normalize", label="Normalize",
    category="enhancement",
    inputs=[InDataset(),
            InFloat("low_pct", "Low %", unit="", field=True, default=1.0),
            InFloat("high_pct", "High %", unit="", field=True, default=99.0)],
    outputs=[OutDataset()],
    modes=[Mode("scope", ["plane", "volume", "series"], default="plane",
                label="Scope")],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    # axis-preserving, but it changes what the NUMBERS mean: [0,1] floats are not raw
    # counts, so `bit_depth` is dropped (edit-time + payload in lockstep) and every
    # downstream raw-count consumer sees "no declared integer scale" (wire-node-v2 §7c).
    meta_transform=_meta_value_rescaled,
    description="Percentile normalize to [0,1]; the statistics scope "
                "(plane/volume/series) is a Mode, not the 2D/3D lever (H24). Drops "
                "bit_depth — the output is no longer raw integer counts.")


# ── Spot detection (LoG → Points) ───────────────────────────────────────────────

def _compute_spots(ctx: EvalContext) -> Dataset:
    """Blob detection → a Point structure table. ``method`` picks LoG (``blob_log``) or
    DoG (``blob_dog``); ``polarity`` detects bright spots (default) or dark spots (the
    normalized image inverted). The metadata-intelligent radii derive from the
    diffraction limit; radius↔σ uses ``σ = r/√ndim`` (2D vs 3D, anisotropic axial σ from
    ``z_step_um``). 2D detects per plane (``z_kind="plane_index"``); 3D in the volume
    (subpixel z). Ids are global-unique across the whole detection."""
    from skimage.feature import blob_dog, blob_log
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("spot detection needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    blob_fn = blob_dog if modes.get("method") == "dog" else blob_log
    dark = modes.get("polarity") == "dark"
    px = ctx.calib("pixel_size_um") or 0.1
    thr = float(ctx.params.get("threshold", 0.1))
    layer = ctx.layer("name")
    is_3d = ctx.is_volume
    # Anisotropic 3D: the axial σ derives from z_step_um, the lateral from pixel_size_um
    # (the detector takes a per-axis σ sequence; a scalar σ would search Z at the lateral
    # scale). Read z_step_um ONLY in 3D so a 2D pull isn't memo-fenced on it (R1).
    zs = (ctx.calib("z_step_um") or 0.5) if is_3d else None

    # radius r ↔ Gaussian σ (the blob detector's characteristic scale): σ = r/√ndim.
    # Floor at 1.0 voxel: scipy's discretized LoG (gaussian_laplace, used by blob_log)
    # is ill-conditioned below ~1 voxel — a sub-voxel σ makes the second-derivative
    # kernel stop summing near zero and return a large spurious uniform response, which
    # blob_log then reports as thousands of false blobs (you cannot resolve a blob
    # smaller than a voxel anyway). Common anisotropic z-steps drive the axial σ
    # sub-voxel, so the floor is load-bearing, not cosmetic.
    def sig_xy(r_um: float, ndim: int) -> float:
        return max(1.0, (r_um / px) / np.sqrt(ndim))

    def sig_z(r_um: float) -> float:                        # anisotropic axial σ (3D only)
        return max(1.0, (r_um / zs) / np.sqrt(3))           # LoG floor (see sig_xy)

    # Per-channel radii (C8 / H12): min/max_radius DERIVE from THIS channel's emission λ
    # (0.61·λ/NA), so each channel gets its own diffraction-limited scale — unless the user
    # pinned a value (one widget ⇒ it applies to every channel). The 3D axial radius is a
    # user override else the per-channel lateral radius (paired-float pattern → tracks it).
    def channel_sigmas(c: int):
        ch = ctx.channel(c)
        rmin = float(ch.param("min_radius")); rmax = float(ch.param("max_radius"))
        if is_3d:
            rmin_z = float(ctx.params["min_radius_z"]) if "min_radius_z" in ctx.params else rmin
            rmax_z = float(ctx.params["max_radius_z"]) if "max_radius_z" in ctx.params else rmax
            return ((sig_z(rmin_z), sig_xy(rmin, 3), sig_xy(rmin, 3)),
                    (sig_z(rmax_z), sig_xy(rmax, 3), sig_xy(rmax, 3)))
        return (sig_xy(rmin, 2), sig_xy(rmax, 2))

    def prep(a: np.ndarray) -> np.ndarray:
        """Normalize to [0,1]; invert for dark-spot polarity so a dark blob reads as
        a bright peak the detector can find. A flat/blank image has no spots of EITHER
        polarity — guard it BEFORE inverting (else dark would turn all-zeros into a
        constant-1 field and flood the detector with false positives)."""
        mn, mx = float(a.min()), float(a.max())
        if mx <= mn:
            return np.zeros_like(a)
        v = (a - mn) / (mx - mn)
        return 1.0 - v if dark else v

    tables = []
    n_units = ax.m * ax.t * ax.c * (1 if is_3d else ax.z)      # eager blob detector
    done_units = 0
    ctx.progress(0, n_units, "detecting")
    for m in range(ax.m):
        for t in range(ax.t):
            for c in range(ax.c):
                min_sig, max_sig = channel_sigmas(c)         # per-channel derived scale
                if is_3d:
                    vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y,
                                                 0, ax.x).astype(float)
                    blobs = blob_fn(prep(vol), min_sigma=min_sig, max_sigma=max_sig,
                                    threshold=thr)
                    if len(blobs):
                        tables.append(point_table(blobs[:, :3], m=m, t=t, c=c,
                                                  z_kind="subpixel", layer=layer))
                    done_units += 1
                    ctx.progress(done_units, n_units, f"t={t} c={c}")
                else:
                    for z in range(ax.z):
                        plane = prov.get_region(0, m, t, z, c, 0, ax.y,
                                                0, ax.x).astype(float)
                        blobs = blob_fn(prep(plane), min_sigma=min_sig,
                                        max_sigma=max_sig, threshold=thr)
                        if len(blobs):
                            tables.append(point_table(blobs[:, :2], z=z, m=m, t=t, c=c,
                                                      z_kind="plane_index", layer=layer))
                        done_units += 1
                        ctx.progress(done_units, n_units, f"t={t} z={z}")
    zk = "subpixel" if is_3d else "plane_index"
    if not tables:
        merged = point_table(np.zeros((0, 3 if is_3d else 2)), z_kind=zk, layer=layer)
    else:
        cols = {k: np.concatenate([tb.columns[k] for tb in tables])
                for k in tables[0].columns}
        cols["id"] = np.arange(len(cols["id"]), dtype=np.int64)   # global-unique ids
        merged = StructureTable(Domain.POINT, cols, layer=layer, z_kind=zk)
    return ds.with_structure(merged)


register_node(
    _compute_spots, op_key="detect.spots", label="Spot Detection", category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InFloat("min_radius", "Min radius", unit="um", field=True, default=0.2,
                derive="0.5*0.61*(emission_nm or 520)/(na or 1.4)/1000"),
        InFloat("max_radius", "Max radius", unit="um", field=True, default=0.6,
                derive="1.5*0.61*(emission_nm or 520)/(na or 1.4)/1000"),
        # 3D-only axial radii (paired-float pattern) — default to the lateral radius
        InFloat("min_radius_z", "Min radius Z", unit="um_axial", field=True, default=0.2,
                available_in={"dim": frozenset({"3D"})}),
        InFloat("max_radius_z", "Max radius Z", unit="um_axial", field=True, default=0.6,
                available_in={"dim": frozenset({"3D"})}),
        InFloat("threshold", "Threshold", unit="", field=True, default=0.1),
        InString("name", "Output layer", field=False, default="spots",
                 layer_out=(Domain.POINT,)),
    ],
    outputs=[OutDataset()],
    modes=[DimMode(), Mode("method", ["log", "dog"], default="log"),
           Mode("polarity", ["bright", "dark"], default="bright")],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="LoG/DoG spot detection → Points (bright or dark polarity); radii "
                "derive from the diffraction limit (σ = r/√ndim); 2D per-plane vs 3D.")


# ── Z-Project (axis-changing: z→1 — the meta_transform showcase) ────────────────

def _compute_zproject(ctx: EvalContext) -> Dataset:
    """Collapse the Z axis with the chosen reducer → a z==1 Dataset, dropping
    ``z_step_um`` and stamping ``z_collapsed`` in lockstep with its ``z_project``
    meta_transform (V2.03 §2 A2). An axis-changing node MUST update calibration and
    axes together — :meth:`reshaped_axes` + :meth:`with_metadata`."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("z-project needs an image provider on its input Dataset")
    ax = prov.axes
    method = ctx.params.get("__modes__", {}).get("method", "max")
    new_axes = replace(ax, z=1)
    # a `sum` projection widens the intensity scale (n_z summed samples): the
    # meta_transform already computed the new depth, so SYNC the payload to the env value
    # instead of re-deriving it (§8 — re-deriving would widen twice on a re-pull).
    bd_out = ctx.calib("bit_depth")
    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        # SAME reducers as the lazy ZReduceProvider path (nodegraph.reducers — one
        # NaN policy for both paths; review 2026-07-22: the old plain-op lambdas
        # propagated NaN while the lazy path ignores it → path-dependent bytes)
        out = np.zeros((ax.m, ax.t, 1, ax.c, ax.y, ax.x), dtype=float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
                    out[m, t, 0, c] = _reduce(vol.astype(float), (0,), method)
        projected = ds.with_image(ArrayProvider(out)).reshaped_axes(new_axes)
        return projected.with_metadata(z_step_um=None, z_collapsed=True,
                                      bit_depth=bd_out)
    # C1: the engine-driven tree-reduce — output tile (iy,ix) folds the base's
    # z-planes of that window via the PartialReducer monoid (median stacks the
    # window's z-column); no whole plane is ever realized (V2.04 §1).
    fp = stream_fp("zreduce", ctx.op_key, ctx.params, ctx.reads.declared_reads(),
                   (), prov)
    projected = ds.with_image(
        ZReduceProvider(prov, method, fp=fp, cache=cache)).reshaped_axes(new_axes)
    return projected.with_metadata(z_step_um=None, z_collapsed=True,
                                   bit_depth=bd_out)


register_node(
    _compute_zproject, op_key="util.zproject", label="Z-Project", category="utility",
    inputs=[InDataset()], outputs=[OutDataset()],
    modes=[Mode("method", ["max", "mean", "sum", "min", "median"], default="max",
                label="Method")],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset({"z"}),
    meta_transform=_meta_z_project,
    description="Project the Z axis (max/mean/sum/min/median) → a 2D (z==1) Dataset; "
                "drops z_step_um and marks z_collapsed.")


# ── Crop (axis-changing: shrink Y,X and, in 3D, Z) ──────────────────────────────

def _compute_crop(ctx: EvalContext) -> Dataset:
    """Crop the spatial extent (and, in 3D mode, the Z range). Its ``crop``
    meta_transform tracks the new extent at edit time; pixel size is preserved
    (origin is deferred, V2.00 §16)."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("crop needs an image provider on its input Dataset")
    ax = prov.axes

    def bound(v, default, hi):
        return max(0, min(int(v) if v is not None else default, hi))

    y0, y1 = bound(ctx.params.get("y0"), 0, ax.y), bound(ctx.params.get("y1"), ax.y, ax.y)
    x0, x1 = bound(ctx.params.get("x0"), 0, ax.x), bound(ctx.params.get("x1"), ax.x, ax.x)
    if ctx.is_volume:
        z0 = bound(ctx.params.get("z0"), 0, ax.z)
        z1 = bound(ctx.params.get("z1"), ax.z, ax.z)
    else:
        z0, z1 = 0, ax.z
    if y1 <= y0 or x1 <= x0 or z1 <= z0:
        raise ValueError(f"crop produced an empty region "
                         f"(y[{y0}:{y1}] x[{x0}:{x1}] z[{z0}:{z1}])")
    ny, nx, nz = y1 - y0, x1 - x0, z1 - z0
    new_axes = replace(ax, z=nz, y=ny, x=nx)
    # C1: a pure lazy offset view (the _ChannelView pattern) — no pixels move, the
    # source dtype is preserved, and a kernel op downstream clips its halo at THIS
    # view's extents (= the eager reflect-at-crop-edge behavior, V2.04 §6b).
    view = WindowView(prov, z0=z0, y0=y0, x0=x0, axes=new_axes)
    return ds.with_image(view).reshaped_axes(new_axes)


register_node(
    _compute_crop, op_key="util.crop", label="Crop", category="utility",
    inputs=[
        InDataset(),
        InInt("y0", "Y start", unit="px", field=False),
        InInt("y1", "Y end", unit="px", field=False),
        InInt("x0", "X start", unit="px", field=False),
        InInt("x1", "X end", unit="px", field=False),
        InInt("z0", "Z start", unit="px", field=False,
              available_in={"dim": frozenset({"3D"})}),
        InInt("z1", "Z end", unit="px", field=False,
              available_in={"dim": frozenset({"3D"})}),
    ],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN, kernel_axes=frozenset(),
    meta_transform=_meta_crop,
    description="Crop Y,X (and Z in 3D mode); pixel size preserved (origin deferred).")


# ── Morphological Gradient (edge map) ───────────────────────────────────────────

def _compute_morph_gradient(ctx: EvalContext) -> Dataset:
    from scipy import ndimage as ndi
    ds = ctx.inputs[0]
    wy = _win(_radius_px(ctx, "radius", 0.3))
    wz = _win(_radius_z_px(ctx, "radius", 0.3)) if ctx.is_volume else wy
    return _map_image(
        ctx, ds,
        plane_fn=lambda a: ndi.morphological_gradient(a, size=(wy, wy)),
        volume_fn=lambda v: ndi.morphological_gradient(v, size=(wz, wy, wy)),
        halo=wy // 2)                # single dilate−erode pass: one-window reach


register_node(
    _compute_morph_gradient, op_key="enhance.morphological_gradient",
    label="Morphological Gradient", category="enhancement",
    inputs=[InDataset(), *_InRadius()], outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN, kernel_axes=_DIM_KAX,
    description="Dilation − erosion edge map; 2D per-plane vs anisotropic 3D.")


# ── Bilateral denoise (edge-aware; skimage is 2D → stack-of-2D, H15) ────────────

def _compute_bilateral(ctx: EvalContext) -> Dataset:
    from skimage.restoration import denoise_bilateral
    ds = ctx.inputs[0]
    sc = float(ctx.params.get("sigma_color", 0.1))
    ss = _radius_px(ctx, "sigma_spatial", 0.2)

    def plane(a: np.ndarray) -> np.ndarray:
        mn, mx = float(a.min()), float(a.max())
        if mx <= mn:
            return a.copy()
        n01 = (a - mn) / (mx - mn)                 # bilateral σ_color is range-relative
        return denoise_bilateral(n01, sigma_color=sc, sigma_spatial=ss) * (mx - mn) + mn

    return _map_image(ctx, ds, plane_fn=plane)


register_node(
    _compute_bilateral, op_key="enhance.bilateral", label="Bilateral Denoise",
    category="enhancement",
    inputs=[InDataset(),
            InFloat("sigma_spatial", "Sigma spatial", unit="um", field=True, default=0.2),
            InFloat("sigma_color", "Sigma color", unit="", field=True, default=0.1)],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=Granularity.WHOLE_PLANE,      # range-relative σ_color → plane min/max (C1)
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"y", "x"})},
    supports_2d=True, supports_true_3d=False, three_d_fallback="stack_of_2d",
    description="Edge-preserving bilateral denoise — stack-of-2D even in 3D mode "
                "(skimage denoise_bilateral is 2D-only, H15).")


# ── Non-local means denoise (patch-based; genuinely n-D) ────────────────────────

def _compute_nlm(ctx: EvalContext) -> Dataset:
    from skimage.restoration import denoise_nl_means
    ds = ctx.inputs[0]
    h = float(ctx.params.get("h", 0.1))
    ps = max(1, int(ctx.params.get("patch_size", 3)))
    pd = max(1, int(ctx.params.get("patch_distance", 3)))

    def apply(a: np.ndarray) -> np.ndarray:
        mn, mx = float(a.min()), float(a.max())
        if mx <= mn:
            return a.copy()
        n01 = (a - mn) / (mx - mn)
        out = denoise_nl_means(n01, patch_size=ps, patch_distance=pd, h=h, fast_mode=True)
        return out * (mx - mn) + mn

    return _map_image(ctx, ds, plane_fn=apply, volume_fn=apply)


register_node(
    _compute_nlm, op_key="enhance.nlm", label="Non-Local Means", category="enhancement",
    inputs=[InDataset(),
            InFloat("h", "Cut-off h", unit="", field=True, default=0.1),
            InInt("patch_size", "Patch size", unit="px", field=False, default=3),
            InInt("patch_distance", "Patch distance", unit="px", field=False, default=3)],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,   # plane/volume min/max normalize
    supports_2d=True, supports_true_3d=True,
    description="Non-local-means denoise — genuinely volumetric in 3D "
                "(denoise_nl_means is true n-D).")


# ── EDT (distance transform of a mask → a physical-µm Voxel field) ──────────────

def _compute_edt(ctx: EvalContext) -> Dataset:
    """Euclidean distance transform of a Voxel mask → a ``distance`` Voxel layer in
    **µm** (anisotropic sampling from pixel/z size). 2D distances each plane; 3D the
    whole volume."""
    from scipy import ndimage as ndi
    ds = ctx.inputs[0]
    ax = ds.axes
    mask_attr = ds.get(Domain.VOXEL, ctx.layer("mask"))
    if mask_attr is None:
        raise ValueError(f"EDT needs a Voxel mask {ctx.params.get('mask', 'mask')!r} "
                         f"(run Threshold first)")
    mask6 = mask_attr.values
    px = ctx.calib("pixel_size_um") or 0.1
    out = np.zeros_like(mask6, dtype=float)
    is_3d = ctx.is_volume
    for m in range(ax.m):
        for t in range(ax.t):
            for c in range(ax.c):
                if is_3d:
                    zs = ctx.calib("z_step_um") or 0.5
                    out[m, t, :, c] = ndi.distance_transform_edt(
                        mask6[m, t, :, c] != 0, sampling=(zs, px, px))
                else:
                    for z in range(ax.z):
                        out[m, t, z, c] = ndi.distance_transform_edt(
                            mask6[m, t, z, c] != 0, sampling=(px, px))
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), out)


register_node(
    _compute_edt, op_key="analysis.edt", label="Distance Transform",
    category="analysis",
    # same contract as analysis.threshold: consumes a Voxel mask, adds a Voxel layer
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            InString("mask", "Mask layer", field=False, default="mask",
                     layer_in=Domain.VOXEL),
            InString("name", "Output layer", field=False, default="distance",
                     layer_out=(Domain.VOXEL,))],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="Euclidean distance transform of a Voxel mask → a µm distance field "
                "(anisotropic in 3D); 2D per-plane vs 3D volumetric.")


# ── Watershed (split a mask by EDT-peak markers → Labels) ───────────────────────

def _labeled_table(raster: np.ndarray, *, m: int, t: int, c: int, layer: str,
                   is_3d: bool, z_index: int = 0) -> StructureTable:
    """A Label StructureTable (invariant ``id,m,t,c,area,z,y,x`` schema) from an
    ALREADY-labelled raster (ids are the label values). Region-props for nodes that
    produce labels by a means other than fresh CCL (e.g. watershed)."""
    from scipy import ndimage as ndi
    labs = np.unique(raster)
    labs = labs[labs != 0]
    k = len(labs)
    zk = "subpixel" if is_3d else "plane_index"
    if k == 0:
        cols = {n: np.zeros(0, dtype=(np.int64 if n in ("id", "m", "t", "c", "area")
                                      else float))
                for n in ("id", "m", "t", "c", "area", "z", "y", "x")}
        return StructureTable(Domain.LABEL, cols, layer=layer, z_kind=zk)
    ones = np.ones_like(raster, dtype=float)
    areas = np.asarray(ndi.sum_labels(ones, raster, labs))
    coms = np.asarray(ndi.center_of_mass(ones, raster, labs), dtype=float).reshape(k, -1)
    if is_3d:
        cz, cy, cx = coms[:, 0], coms[:, 1], coms[:, 2]
    else:
        cy, cx = coms[:, 0], coms[:, 1]
        cz = np.full(k, float(z_index))
    cols = {
        "id": labs.astype(np.int64), "m": np.full(k, m, np.int64),
        "t": np.full(k, t, np.int64), "c": np.full(k, c, np.int64),
        "area": areas.astype(np.int64), "z": cz, "y": cy, "x": cx,
    }
    return StructureTable(Domain.LABEL, cols, layer=layer, z_kind=zk)


# ── Segmentation — the ONE image→instance-labels node (`method` = the algorithm) ─
#
# V2.12. Every segmentation shares one data contract: an image in, a Voxel **label
# raster** + a per-object **Label table** out, with globally-unique ids. Only the
# algorithm that finds the objects differs. So the catalog carries ONE Segmentation node
# with a ``method`` Mode instead of one node per algorithm: ``analysis.watershed`` and
# ``detect.stardist_nuclei`` were folded in here and deleted, **CellSAM** is new, and
# every shared stage — the µm²/µm³ size filter, hole filling, the contiguous relabel, the
# global id offset, the Label table, the 2D/3D lever — is written exactly ONCE instead of
# once per method (which is how the old pair drifted: stardist filtered by area and
# watershed did not, stardist had no lever and watershed did).
#
# The methods are deliberately heterogeneous in what they need, and ``available_in`` keeps
# each method's controls to itself (wire-node-v2 §5b): two classical methods that cut a
# foreground and split it (``threshold``, ``watershed``) and two learned detectors that
# take an image and return instances directly (``stardist``, ``cellsam``).

_SEGMENT_METHODS = ("threshold", "watershed", "stardist", "cellsam")
#: Foreground level for the classical methods — the `analysis.threshold` menu, defaulting
#: to `otsu` because a segmentation node should work on an unseen image with no numbers.
_SEGMENT_LEVELS = ("otsu", "li", "yen", "triangle", "mean", "fixed")
#: Methods that CUT a foreground and therefore read the level controls.
_SEGMENT_CLASSICAL = frozenset({"threshold", "watershed"})
#: Methods whose backend is a 2-D-per-plane detector (StarDist2D; CellSAM's ViT + SAM
#: decoder). They cannot produce z-connected 3D instances, so 3D mode is REFUSED rather
#: than quietly returning a stack of per-plane labels while the lever claims 3D
#: (wire-node-v2 §5 — never silently loop 2D over Z while claiming 3D). 3D consensus
#: fusion of 2D slices is a real algorithm (u-Segment3D, CellSAM paper Fig. 3d), not
#: something to fake here.
_SEGMENT_2D_ONLY = frozenset({"stardist", "cellsam"})


def _segment_level(arr: np.ndarray, level: str, fixed: float) -> float:
    """The foreground cut for ONE segmentation unit (a plane in 2D, a volume in 3D).

    ``fixed`` is the user's level in the image's own intensity units; every other choice
    derives it from that unit's own histogram — **per unit**, which is what the node's
    declared footprint says it reads (WHOLE_PLANE in 2D / WHOLE_VOLUME in 3D). One level
    for the whole dataset is a different (also valid) recipe, and it is what
    ``analysis.threshold`` does: compose it with this node's ``watershed`` ``mask`` socket
    when you want a single global cut.

    A flat unit gets no histogram level: skimage's methods return the constant itself (or
    warn and divide by zero), and a cut AT the constant paints the WHOLE plane as one
    giant object. Returning just above the maximum yields an empty foreground, which is
    the honest answer for a blank plane.

    **Non-finite pixels are excluded from the histogram** (V2.12): a SINGLE NaN or inf —
    which a deconvolution, a normalize of a flat region or a resampled edge can produce —
    otherwise makes every skimage method raise ``autodetected range … is not finite``, and
    NaN also defeats the flat-unit guard above (``nan == nan`` is False). Cutting on the
    finite population is the useful answer; NaN voxels then fall out of the foreground on
    their own, since ``nan > level`` is False."""
    if level == "fixed":
        return float(fixed)
    import skimage.filters as skf
    fn = {"otsu": skf.threshold_otsu, "li": skf.threshold_li, "yen": skf.threshold_yen,
          "triangle": skf.threshold_triangle, "mean": skf.threshold_mean}[level]
    flat = np.asarray(arr, dtype=float).ravel()      # 1-D (skimage RGB-shape guard)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return float("inf")                          # nothing to cut ⇒ empty foreground
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        return hi + 1.0                              # blank unit ⇒ empty foreground
    return float(fn(finite))


def _segment_fill_holes(raster: np.ndarray) -> np.ndarray:
    """Fill each label region's interior holes — the "hole filling" half of the
    postprocess Cellpose and CellSAM both apply (CellSAM, *Nat. Methods* 22:2585, Methods
    → "CellSAM postprocessing"), here shared by every method.

    **Non-destructive**, unlike the upstream idiom. Cellpose's
    ``fill_holes_and_remove_small_masks`` writes ``masks[slc][filled] = id`` across the
    object's whole bounding box, which steals voxels that already belong to a NEIGHBOURING
    label whenever two objects share a box. Only voxels that are currently background are
    painted here, so filling can never move a boundary between two objects."""
    from scipy import ndimage as ndi
    out = np.asarray(raster)
    if out.size == 0 or int(out.max()) == 0:
        return out
    out = out.copy()
    for i, slc in enumerate(ndi.find_objects(out), start=1):
        if slc is None:                              # id absent (non-contiguous labels)
            continue
        sub = out[slc]                               # a VIEW — assignment writes through
        holes = ndi.binary_fill_holes(sub == i) & (sub == 0)
        if holes.any():
            sub[holes] = i
    return out


def _segment_size_filter(raster: np.ndarray, min_px: int, max_px: int) -> np.ndarray:
    """Drop objects outside ``[min_px, max_px]`` **voxels** (inclusive; ``0`` disables
    either bound) and relabel the survivors ``1..K`` — one ``O(voxels)`` bincount + LUT
    pass, the shape of ``stardist_segment.filter_and_relabel``.

    The relabel runs even with both bounds off, and that is load-bearing: the caller makes
    ids globally unique by adding a running offset per unit, which is only correct if each
    unit's own ids are contiguous from 1."""
    lab = np.asarray(raster).astype(np.int64, copy=False)
    top = int(lab.max()) if lab.size else 0
    if top == 0:
        return np.zeros(lab.shape, dtype=np.int64)
    counts = np.bincount(lab.ravel(), minlength=top + 1)
    keep = counts > 0
    keep[0] = False                                  # background is never an object
    if min_px > 0:
        keep &= counts >= min_px
    if max_px > 0:
        keep &= counts <= max_px
    lut = np.zeros(top + 1, dtype=np.int64)
    lut[np.flatnonzero(keep)] = np.arange(1, int(keep.sum()) + 1, dtype=np.int64)
    return lut[lab]


def _segment_watershed_split(fg: np.ndarray, sampling, footprint: np.ndarray) -> np.ndarray:
    """Split a foreground mask into touching objects by seeding a watershed at the peaks
    of the (anisotropic) distance transform — the classic "separate touching objects" step,
    carried over verbatim from the folded-in ``analysis.watershed``.

    Peak suppression is a **physical** radius expressed as a per-axis ``footprint``, not
    the isotropic index-unit ``min_distance``: on an anisotropic volume (z_step > pixel)
    an index-unit radius suppresses far too aggressively along Z and under-segments.

    **Adjacent peaks are ONE marker** (V2.12 fix). A distance transform is full of
    plateaus — a disk on an integer grid has several pixels at the same maximum, and an
    elongated object has a whole ridge of them — and ``peak_local_max`` returns *every*
    pixel of a plateau. The folded-in ``analysis.watershed`` numbered each returned pixel
    as its own marker, so one object was shattered into as many basins as its plateau had
    pixels (measured: four synthetic disks became 19 fragments per plane, the smallest of
    them 1 voxel). Connected-component labelling the peak mask first — the recipe from
    scikit-image's own watershed example — collapses each plateau to a single seed while
    leaving genuinely distinct maxima separate, which is the whole point of the node.

    That merge uses **full** connectivity (8 in 2D / 26 in 3D), not scipy's cross-shaped
    default: a plateau is frequently a diagonal ring — the equidistant crest inside any
    object with a hole in it — and a cross-connected label breaks such a ring into one
    marker per diagonal step (the same four disks then yield 11 basins instead of 4).

    **`exclude_border=False` is load-bearing** (V2.12 fix). ``peak_local_max`` defaults to
    excluding a border shell ``min_distance`` wide on EVERY axis — and ``min_distance`` is a
    parameter this call does not even use, since the physical suppression is the per-axis
    ``footprint``. Leaving the default on discards peaks near the frame edge, and in 3D on a
    volume with ``z <= 2`` there is **no interior z plane at all**, so it returns ZERO peaks:
    the fallback below then plants a single marker and the watershed collapses every object
    in the volume into one (measured on two disjoint disks over z=1 and z=2 — 1 object
    instead of 2, silently). Objects at the image border are real objects."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    if not fg.any():
        return np.zeros(fg.shape, dtype=np.int64)
    edt = ndi.distance_transform_edt(fg, sampling=sampling)
    peaks = peak_local_max(edt, footprint=footprint, labels=fg.astype(np.int64),
                           exclude_border=False)
    seeds = np.zeros(fg.shape, dtype=bool)
    if len(peaks):
        seeds[tuple(np.asarray(peaks).T)] = True
    full = ndi.generate_binary_structure(seeds.ndim, seeds.ndim)
    markers = np.asarray(ndi.label(seeds, structure=full)[0], dtype=np.int64)
    if markers.max() == 0:                           # no separable peak → one basin
        markers[np.unravel_index(int(np.argmax(edt)), edt.shape)] = 1
    return np.asarray(seeded_watershed(fg, markers, sampling=sampling), dtype=np.int64)


def _compute_segment(ctx: EvalContext) -> Dataset:
    """**Segmentation** — image → a Voxel **label raster** + a per-object **Label table**
    (ids globally unique across every unit); ``method`` picks the algorithm.

    THE segmentation node. Input and output domains are identical for every method, so the
    algorithm is a Mode rather than a separate node type, and the surrounding stages are
    shared:

    ``method``
        * **threshold** — cut a foreground at ``level`` (otsu/li/yen/triangle/mean, or a
          ``fixed`` level in the image's own units) and connected-component label it
          (``connectivity`` 4/8 in 2D, 6/18/26 in 3D; ``0`` = per-dim default).
        * **watershed** — the same foreground, split at the peaks of the anisotropic
          distance transform (``min_distance``, µm). An existing binary/label layer can be
          used as the foreground instead by naming it in ``mask`` — that is how the
          removed ``analysis.watershed`` behaved, and it is what to use when the mask comes
          from ``analysis.threshold`` / ``analysis.roi_mask`` /
          ``analysis.histogram_threshold``.
        * **stardist** — StarDist star-convex CNN (``prob_thresh``/``nms_thresh``/
          ``scale``/``model_name``), kernel :mod:`nodegraph.kernels.stardist_segment`.
        * **cellsam** — CellSAM: a SAM ViT-B whose mask decoder is prompted by CellFinder
          (Anchor-DETR) box detections, i.e. a *generalist* segmenter that needs no
          per-dataset tuning and no seeds (``bbox_threshold`` is the precision/recall
          knob; ``normalize``/``postprocess``/``remove_boundaries``/``tile``…), kernel
          :mod:`nodegraph.kernels.cellsam_segment`.

    Shared by all four: the output layer ``name`` (one name, two domains — the Voxel
    raster and the Label table), hole filling, the size filter in **µm² (2D) / µm³ (3D)**,
    the contiguous relabel, the global id offset, and the Label table's invariant
    ``id,m,t,c,area,z,y,x`` schema with ``area`` in voxels.

    **2D vs 3D is what the lever means here:** 2D segments each ``(m,t,z,c)`` plane
    independently and emits per-plane instances (``z_kind="plane_index"``); 3D segments
    each ``(m,t,c)`` volume and emits z-connected instances (``z_kind="subpixel"``). The
    two learned methods are 2-D-per-plane detectors and **refuse** 3D rather than
    pretending a stack of per-plane labels is a 3D segmentation.

    Reads ``pixel_size_um`` (and ``z_step_um`` in 3D) for the size filter and the seed
    radius; both are recorded, so the memo re-checks them.

    Per-channel: each channel is segmented independently, as every other structure
    producer in the catalog does. CellSAM's ``(blank, nuclear, whole-cell)`` multi-channel
    fusion is deliberately NOT exposed — a fused segmentation has no single ``c`` to file
    its Label rows under, and inventing one silently would corrupt every downstream
    per-channel join. Select the marker channel upstream (``channel.select``)."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("segmentation needs an image provider on its input Dataset")
    ax = prov.axes
    is_3d = ctx.is_volume
    modes = ctx.params.get("__modes__", {})
    method = str(modes.get("method") or "threshold")
    if method not in _SEGMENT_METHODS:
        raise ValueError(f"unknown segmentation method {method!r} — one of "
                         f"{list(_SEGMENT_METHODS)}")
    if is_3d and method in _SEGMENT_2D_ONLY:
        raise ValueError(
            f"segmentation: {method!r} is a 2-D-per-plane detector and cannot produce "
            "z-connected 3D instances. Set the 2D/3D lever to 2D to segment every plane "
            "independently (per-plane instances), or z-project first. (Fusing 2D slices "
            "into true 3D objects is its own algorithm — u-Segment3D in the CellSAM "
            "paper — not something this node fakes.) The `threshold` and `watershed` "
            "methods do run volumetrically in 3D.")
    layer = ctx.layer("name")

    # ── the shared size filter: µm² (2D) / µm³ (3D) → voxels ──────────────────
    px = ctx.calib("pixel_size_um") or 0.1
    # The four size params are read with LITERAL keys, per dim, deliberately: the socket
    # contract's index is an AST pass that only resolves string constants (and helper args),
    # so a `ctx.params.get(key_variable)` read would make all four sockets look DEAD and
    # fail the guard — the one place where saying it twice is required, not sloppy.
    if is_3d:
        zs = ctx.calib("z_step_um") or 0.5
        vox, unit, lo_key, hi_key = px * px * zs, "µm³", "min_volume", "max_volume"
        lo_um = float(ctx.params.get("min_volume", 0.0))
        hi_um = float(ctx.params.get("max_volume", 0.0))
    else:
        vox, unit, lo_key, hi_key = px * px, "µm²", "min_area", "max_area"
        lo_um = float(ctx.params.get("min_area", 0.0))
        hi_um = float(ctx.params.get("max_area", 0.0))
    min_px, max_px = int(round(lo_um / vox)), int(round(hi_um / vox))
    # 0 is the OFF sentinel for both bounds, which is safe for the lower one (every object
    # has at least one voxel, so a sub-voxel minimum filters nothing either way) but
    # INVERTS the upper one: a sub-voxel maximum would quantize to "no upper limit" and
    # keep everything instead of dropping all but sub-voxel specks. Refuse, exactly as
    # analysis.histogram_threshold does with the same arithmetic.
    if hi_um > 0.0 and max_px == 0:
        raise ValueError(
            f"segmentation: {hi_key}={hi_um:g} {unit} is under one voxel ({vox:g} {unit}) "
            f"— it quantizes to 0, which is this filter's OFF value, so it would keep "
            f"every object instead of dropping the large ones. Raise it, or set exactly 0 "
            f"to disable the upper bound deliberately.")
    if max_px > 0 and min_px > max_px:
        raise ValueError(
            f"segmentation: {lo_key} ({min_px} voxels) exceeds {hi_key} ({max_px} voxels) "
            f"— the size window is empty, so every object would be discarded. Widen it "
            f"({hi_key} 0 = no upper limit).")
    fill = bool(ctx.params.get("fill_holes", False))

    # ── per-method setup (everything read here, once, not per unit) ────────────
    level = str(modes.get("level") or "otsu")
    fixed = 0.5
    conn = 26 if is_3d else 8
    mask6 = None
    footprint = None
    sampling = (zs, px, px) if is_3d else (px, px)
    if method in _SEGMENT_CLASSICAL:
        if level not in _SEGMENT_LEVELS:
            raise ValueError(f"unknown threshold level {level!r} — one of "
                             f"{list(_SEGMENT_LEVELS)}")
        # unset ⇒ the socket's derive: mid-range of the CURRENT declared bit depth, or 0.5
        # when there is none (post-Normalize [0,1] data). channel(0) because the depth is
        # channel-independent (§7c).
        fixed = float(ctx.channel(0).param("threshold", 0.5))
    if method == "threshold":
        # A QSpinBox cannot express "unset", so it commits 0 the moment it is touched —
        # and 0 is not a legal connectivity. Treat it as "per-dim default" instead of
        # letting `_connectivity_rank` raise on a value the user never chose.
        conn = int(ctx.params.get("connectivity", 0) or 0) or (26 if is_3d else 8)
    if method == "watershed":
        src = ctx.layer("mask")
        if src:
            attr = ds.get(Domain.VOXEL, src)
            if attr is None:
                raise ValueError(
                    f"segmentation (watershed): no Voxel layer {src!r} to use as the "
                    "foreground. Pick an existing mask/label layer, or clear the `mask` "
                    "socket to cut the foreground from the image with the level controls.")
            mask6 = np.asarray(attr.values)
        min_um = float(ctx.params.get("min_distance", 0.3))
        rxy = max(1, int(round(to_pixels_v2(min_um, "um", pixel_size_um=px))))
        if is_3d:
            rz = max(1, int(round(to_pixels_v2(min_um, "um_axial", z_step_um=zs))))
            footprint = np.ones((2 * rz + 1, 2 * rxy + 1, 2 * rxy + 1), dtype=bool)
        else:
            footprint = np.ones((2 * rxy + 1, 2 * rxy + 1), dtype=bool)

    model = None
    model_id = ""
    if method == "stardist":
        from nodegraph.kernels.stardist_segment import get_stardist_model, segment_frame
        model_id = str(ctx.params.get("model_name") or "2D_versatile_fluo")
        prob = ctx.params.get("prob_thresh")
        prob = float(prob) if prob not in (None, "") else 0.5
        nms = float(ctx.params.get("nms_thresh", 0.3))
        scale = ctx.params.get("scale")
        scale = float(scale) if scale not in (None, "") else 0.0
        scale = scale if scale > 0 else None          # a 0 spin-box means "no rescale"
        # `disable_gpu` is an ENVIRONMENT knob, never a socket: the loader is a process
        # singleton that ignores the flag after the first call, and TF must see
        # CUDA_VISIBLE_DEVICES before its first import. As a param it would be a control
        # that silently stops working, and it would make the memo non-deterministic
        # (identical recipe hash, different device). Same argument as NODELAB_CELLSAM_DEVICE.
        import os as _os
        _cpu = _os.environ.get("NODELAB_STARDIST_CPU", "") in ("1", "true", "yes")
        try:
            model = get_stardist_model(model_id, disable_gpu=_cpu)
        except Exception as exc:                      # noqa: BLE001 — one clear message
            raise ImportError(f"StarDist model {model_id!r} unavailable "
                              f"(tensorflow/stardist/csbdeep + weights): {exc}") from exc
    if method == "cellsam":
        from nodegraph.kernels.cellsam_segment import get_cellsam_model, segment_plane
        model_id = str(ctx.params.get("cellsam_model") or "cellsam_general")
        weights = str(ctx.params.get("model_path") or "")
        cs = dict(bbox_threshold=float(ctx.params.get("bbox_threshold", 0.4)),
                  normalize=bool(ctx.params.get("normalize", True)),
                  postprocess=bool(ctx.params.get("postprocess", False)),
                  remove_boundaries=bool(ctx.params.get("remove_boundaries", False)),
                  tile=bool(ctx.params.get("tile", False)),
                  tile_size=int(ctx.params.get("tile_size", 512)),
                  overlap=int(ctx.params.get("tile_overlap", 56)))
        # Loaded ONCE for the whole pull (a checkpoint read + ViT build per plane would
        # dominate a time series); the kernel keys its singleton on (model, path, device).
        model = get_cellsam_model(model_id, model_path=weights)
        if weights:
            # `model_path` WINS inside the kernel (get_local_model), so the provenance must
            # name the checkpoint that actually ran, not the published-model socket the
            # loader ignored.
            model_id = weights

    # ── the unit loop: one segmentation per plane (2D) / per volume (3D) ───────
    def segment_unit(arr: np.ndarray, unit: tuple) -> np.ndarray:
        """One prepared unit → a label array of the same shape, ids contiguous from 1."""
        m, t, c = unit[0], unit[1], unit[-1]
        z = unit[2] if len(unit) == 4 else None
        if method == "stardist":
            return np.asarray(segment_frame(arr.astype(np.float32), model=model,
                                            prob_thresh=prob, nms_thresh=nms,
                                            scale=scale)[0], dtype=np.int64)
        if method == "cellsam":
            return np.asarray(segment_plane(arr, model=model, **cs), dtype=np.int64)
        if mask6 is not None:
            fg = (mask6[m, t, :, c] if z is None else mask6[m, t, z, c]) != 0
        else:
            fg = arr > _segment_level(arr, level, fixed)
            if arr.dtype.kind == "f":
                # A non-finite voxel is "no valid measurement here", so it is background —
                # never an object. NaN falls out on its own (`nan > level` is False) but
                # `inf > level` is True, and an inf from a deconvolution or a 0/0 would
                # otherwise appear as a phantom one-voxel cell in the table. Treating the
                # two the same is the only defensible reading; the guard is skipped on an
                # integer image, which cannot carry either.
                fg &= np.isfinite(arr)
        if method == "watershed":
            return _segment_watershed_split(fg, sampling, footprint)
        return np.asarray(label_components(fg, conn)[0], dtype=np.int64)

    raster = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.int64)
    cols: Dict[str, list] = defaultdict(list)
    offset = 0

    def take(lab: np.ndarray, m: int, t: int, c: int, z_index: int) -> np.ndarray:
        """Clean up one unit's labels, append its Label rows, and shift its ids into the
        globally-unique range."""
        nonlocal offset
        lab = _segment_size_filter(_segment_fill_holes(lab) if fill else lab,
                                   min_px, max_px)
        tbl = _labeled_table(lab, m=m, t=t, c=c, layer=layer, is_3d=is_3d,
                             z_index=z_index)
        for k, v in tbl.columns.items():
            cols[k].extend(((v + offset) if k == "id" else v).tolist())
        shifted = np.where(lab > 0, lab + offset, 0)
        offset += tbl.n
        return shifted

    note = "segmenting (%s)" % method
    if is_3d:
        for m, t, c in _each_volume_p(ctx, ax, note):
            vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
            raster[m, t, :, c] = take(segment_unit(vol, (m, t, c)), m, t, c, 0)
    else:
        for m, t, z, c in _each_plane_p(ctx, ax, note):
            plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
            raster[m, t, z, c] = take(segment_unit(plane, (m, t, z, c)), m, t, c, z)

    out = ds.with_layer(Domain.VOXEL, layer, raster)
    if cols.get("id"):
        out = out.with_structure(StructureTable(
            Domain.LABEL, {k: np.array(v) for k, v in cols.items()},
            layer=layer, z_kind=("subpixel" if is_3d else "plane_index")))
    # Provenance (§7b), the `track.objects` shape: namespaced non-calibration keys naming
    # HOW these labels were made, so a downstream node or a reader can tell a CNN
    # segmentation from a threshold without re-deriving it.
    prov_md = {"segment_method": method}
    if model_id:
        prov_md["segment_model"] = model_id
    return out.with_metadata(**prov_md)


register_node(
    _compute_segment, op_key="analysis.segment", label="Segmentation",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),   # a label RASTER and a table
    inputs=[
        InDataset(),
        # ── shared by every method ────────────────────────────────────────────
        InString("name", "Output layer", field=False, default="labels",
                 layer_out=(Domain.VOXEL, Domain.LABEL)),
        InBool("fill_holes", "Fill holes", field=False, default=False),
        # A 2D object has an area and a 3D object has a volume — different units, so
        # different sockets rather than one whose meaning silently changes with the lever.
        InFloat("min_area", "Min area", unit="um2", field=False, default=0.0,
                available_in={"dim": frozenset({"2D"})}),
        InFloat("max_area", "Max area", unit="um2", field=False, default=0.0,
                available_in={"dim": frozenset({"2D"})}),
        InFloat("min_volume", "Min volume", unit="um3", field=False, default=0.0,
                available_in={"dim": frozenset({"3D"})}),
        InFloat("max_volume", "Max volume", unit="um3", field=False, default=0.0,
                available_in={"dim": frozenset({"3D"})}),
        # ── threshold + watershed: the foreground cut ─────────────────────────
        # a FIXED level is in the image's own units, so its default follows the declared
        # intensity scale: mid-range of the current bit depth (2047.5 on 12-bit), falling
        # back to 0.5 when there is no declared integer scale — i.e. exactly after a
        # Normalize dropped it (wire-node-v2 §7c).
        InFloat("threshold", "Level", unit="", field=False, default=0.5,
                derive="((2**bit_depth - 1)/2) if bit_depth else 0.5",
                available_in={"method": frozenset({"threshold", "watershed"}),
                              "level": frozenset({"fixed"})}),
        # ── threshold only ───────────────────────────────────────────────────
        InInt("connectivity", "Connectivity", unit="", field=False, default=0,
              available_in={"method": frozenset({"threshold"})}),
        # ── watershed only ───────────────────────────────────────────────────
        # EMPTY by default: a Segmentation node segments the image. Naming a layer here
        # splits THAT foreground instead (the folded-in `analysis.watershed` behaviour).
        InString("mask", "Foreground layer", field=False, default="",
                 layer_in=Domain.VOXEL,
                 available_in={"method": frozenset({"watershed"})}),
        InFloat("min_distance", "Min seed distance", unit="um", field=False, default=0.3,
                available_in={"method": frozenset({"watershed"})}),
        # ── stardist only ────────────────────────────────────────────────────
        InFloat("prob_thresh", "Prob threshold", unit="", field=False, default=0.5,
                available_in={"method": frozenset({"stardist"})}),
        InFloat("nms_thresh", "NMS threshold", unit="", field=False, default=0.3,
                available_in={"method": frozenset({"stardist"})}),
        InFloat("scale", "Scale", unit="", field=False, default=0.0,
                available_in={"method": frozenset({"stardist"})}),
        InString("model_name", "StarDist model", field=False,
                 default="2D_versatile_fluo",
                 available_in={"method": frozenset({"stardist"})}),
        # ── cellsam only (socket names stay disjoint from stardist's: the node card
        #    relayouts on the active socket NAME list, so a same-named socket with a
        #    different default would not redraw when the method changes) ─────────
        InFloat("bbox_threshold", "Box threshold", unit="", field=False, default=0.4,
                available_in={"method": frozenset({"cellsam"})}),
        InString("cellsam_model", "CellSAM model", field=False,
                 default="cellsam_general",
                 available_in={"method": frozenset({"cellsam"})}),
        InString("model_path", "Local weights", field=False, default="",
                 available_in={"method": frozenset({"cellsam"})}),
        InBool("normalize", "Normalize", field=False, default=True,
               available_in={"method": frozenset({"cellsam"})}),
        InBool("postprocess", "Postprocess", field=False, default=False,
               available_in={"method": frozenset({"cellsam"})}),
        InBool("remove_boundaries", "Separate touching", field=False, default=False,
               available_in={"method": frozenset({"cellsam"})}),
        # Tiling is a memory/compute knob in the MODEL's own pixel space (CellSAM resizes
        # every tile to 1024²), not a physical extent — hence `px`, declared rather than
        # hidden as a bare constant. `tile_size`/`tile_overlap` stay visible while `tile`
        # is off because liveness that depends on another SOCKET's value cannot be
        # expressed in `available_in`, which sees mode state only (wire-node-v2 §5b).
        InBool("tile", "Tiled inference", field=False, default=False,
               available_in={"method": frozenset({"cellsam"})}),
        InInt("tile_size", "Tile size", unit="px", field=False, default=512,
              available_in={"method": frozenset({"cellsam"})}),
        InInt("tile_overlap", "Tile overlap", unit="px", field=False, default=56,
              available_in={"method": frozenset({"cellsam"})}),
    ],
    outputs=[OutDataset()],
    modes=[DimMode(),
           Mode("method", list(_SEGMENT_METHODS), default="threshold", label="Method"),
           # the foreground cut belongs to the classical methods only — a learned detector
           # never thresholds, so the dropdown is GATED AWAY rather than shown and ignored
           # (V2.12 `ModeSpec.available_in`, the Mode-level half of wire-node-v2 §5b).
           Mode("level", list(_SEGMENT_LEVELS), default="otsu", label="Level",
                available_in={"method": frozenset(_SEGMENT_CLASSICAL)})],
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,
    description="THE segmentation node: image → a Voxel label raster + a Label table, "
                "with the algorithm as a `method` Mode — threshold+CCL, "
                "distance-transform watershed, StarDist (CNN), or CellSAM (SAM + "
                "CellFinder foundation model). Shared across every method: hole filling, "
                "the µm²/µm³ size filter, globally-unique ids and the region table. 2D "
                "segments each plane independently, 3D each volume (the two learned "
                "methods are 2D-per-plane and refuse the 3D lever).")


# ── Resample (axis-changing: rescale Y,X and, in 3D, Z) ─────────────────────────

def _compute_resample(ctx: EvalContext) -> Dataset:
    """Rescale the spatial extent by ``scale_xy`` (and ``scale_z`` in 3D mode). The
    output sizes and the inverse pixel-size update mirror the ``resample``
    meta_transform EXACTLY (``round(size·scale)``) so header==payload; ``resize`` to the
    computed shape guarantees the match regardless of ``zoom`` rounding."""
    from skimage.transform import resize
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("resample needs an image provider on its input Dataset")
    ax = prov.axes
    is_3d = ctx.is_volume
    sxy = float(ctx.params.get("scale_xy", ctx.params.get("scale", 1.0)) or 1.0)
    sz = float(ctx.params.get("scale_z", 1.0) or 1.0)
    ny, nx = max(1, round(ax.y * sxy)), max(1, round(ax.x * sxy))
    nz = max(1, round(ax.z * sz)) if is_3d else ax.z
    new_axes = replace(ax, z=nz, y=ny, x=nx)

    def resize_plane(a: np.ndarray, m: int, t: int, z: int, c: int) -> np.ndarray:
        return resize(a, (ny, nx), order=1, preserve_range=True)

    def resize_volume(v: np.ndarray, m: int, t: int, c: int) -> np.ndarray:
        return resize(v, (nz, ny, nx), order=1, preserve_range=True)

    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        out = np.zeros((ax.m, ax.t, nz, ax.c, ny, nx), dtype=float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    if is_3d:
                        vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
                        out[m, t, :, c] = resize_volume(vol.astype(float), m, t, c)
                    else:
                        for z in range(ax.z):
                            plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                            out[m, t, z, c] = resize_plane(plane.astype(float), m, t, z, c)
        res = ds.with_image(ArrayProvider(out)).reshaped_axes(new_axes)
    else:
        # C1 (V2.04 §6b sliver): lazy per-UNIT realize. Resample is non-tileable — the
        # fractional output→input grid + auto anti-aliasing gathers the whole unit — so
        # each output plane (2D) / volume (3D) is resized once on first touch and cached.
        # Geometry changes, so a dedicated PlaneRealizeProvider (MapComputeProvider is 1:1).
        fp = stream_fp("resample", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
        res = ds.with_image(PlaneRealizeProvider(
            prov, new_axes, plane_fn=resize_plane, volume_fn=resize_volume,
            is_volume=is_3d, fp=fp, cache=cache)).reshaped_axes(new_axes)
    # The `resample` meta_transform already produced the post-scale pixel/z size in the
    # envelope (single source of truth); SYNC the payload to it rather than re-deriving
    # (ctx.calib reads the already-transformed value — re-dividing would double-count).
    changes: Dict[str, Any] = {}
    px_out = ctx.calib("pixel_size_um")
    if px_out is not None:
        changes["pixel_size_um"] = px_out
    if is_3d:
        zs_out = ctx.calib("z_step_um")
        if zs_out is not None:
            changes["z_step_um"] = zs_out
    return res.with_metadata(**changes) if changes else res


register_node(
    _compute_resample, op_key="util.resample", label="Resample", category="utility",
    inputs=[InDataset(),
            InFloat("scale_xy", "Scale XY", unit="", field=True, default=1.0),
            InFloat("scale_z", "Scale Z", unit="", field=True, default=1.0,
                    available_in={"dim": frozenset({"3D"})})],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX, meta_transform=_meta_resample,
    description="Rescale Y,X (and Z in 3D) by a factor; pixel size scales inversely "
                "(finer when upsampling).")


# ── Stack (axis-changing: T→1 SNR stacking with robust fusion) ──────────────────

def _compute_stack(ctx: EvalContext) -> Dataset:
    """Fuse the Timepoint axis into a single frame (T→1), boosting SNR. ``method`` picks
    the combiner (mean/median + the robust ``sigma_clip``/``trimmed_mean`` fusion
    reducers, or max/sum). Drops ``dt_s`` in lockstep with the ``stack_time``
    meta_transform."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("stack needs an image provider on its input Dataset")
    ax = prov.axes
    method = ctx.params.get("__modes__", {}).get("method", "mean")
    new_axes = replace(ax, t=1)
    bd_out = ctx.calib("bit_depth")        # widened by the transform when method == sum
    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        out = np.zeros((ax.m, 1, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m in range(ax.m):
            for z in range(ax.z):
                for c in range(ax.c):
                    series = np.stack(
                        [prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                         for t in range(ax.t)], axis=0).astype(float)
                    out[m, 0, z, c] = _reduce(series, (0,), method)  # reduce over T
        stacked = ds.with_image(ArrayProvider(out)).reshaped_axes(new_axes)
        return stacked.with_metadata(dt_s=None, bit_depth=bd_out)
    # C1 (V2.04 §6b sliver): engine-driven tree-reduce over T — the mirror of Z-Project's
    # ZReduceProvider. Monoid combiners (mean/sum/max/min) fold the T series incrementally
    # per tile (memory O(window)); median/sigma_clip/trimmed_mean stack the t-column. No
    # whole plane is ever realized — the eager fallback's SAME nodegraph.reducers keep the
    # bytes identical (one NaN policy across both paths).
    fp = stream_fp("treduce", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
    stacked = ds.with_image(
        TReduceProvider(prov, method, fp=fp, cache=cache)).reshaped_axes(new_axes)
    return stacked.with_metadata(dt_s=None, bit_depth=bd_out)


register_node(
    _compute_stack, op_key="util.stack", label="Stack (T→1)", category="utility",
    inputs=[InDataset()], outputs=[OutDataset()],
    modes=[Mode("method", ["mean", "median", "sigma_clip", "trimmed_mean", "max", "sum"],
                default="mean", label="Combine")],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"t"}),
    meta_transform=_meta_stack_time,
    description="Stack the Timepoint axis → one frame (SNR↑) with a robust combiner "
                "(mean/median/sigma-clip/trimmed); drops dt_s.")


# ── Drift correction (registration — axis-preserving, stores the shift) ─────────

def _compute_drift(ctx: EvalContext) -> Dataset:
    """Rigid drift correction across Timepoints via phase cross-correlation on a
    reference channel/plane; the estimated per-frame (Δy, Δx) shift is applied to every
    z and channel and also **stored as Frame-domain attributes** (``drift_y``/
    ``drift_x``) — provenance, per V2.03 §2 (registration stores its transform).
    Axis-preserving (geometry unchanged)."""
    from scipy.ndimage import shift as ndi_shift
    from skimage.registration import phase_cross_correlation
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("drift correction needs an image provider on its input Dataset")
    ax = prov.axes
    ref_z = ax.z // 2
    # Estimate the per-frame (Δy,Δx) shift EAGERLY — a global FFT per frame on the
    # reference channel/plane is inherently whole-plane (and cheap: one plane per t). The
    # APPLY (register-once/apply-all: the SAME shift on every c,z preserves colocalization)
    # is deferred to a lazy per-plane provider (C1 / V2.04 §6b sliver).
    shifts: Dict[Tuple[int, int], Tuple[float, float]] = {}
    dy = np.zeros((ax.m, ax.t), dtype=float)
    dx = np.zeros((ax.m, ax.t), dtype=float)
    for m in range(ax.m):
        ref = prov.get_region(0, m, 0, ref_z, 0, 0, ax.y, 0, ax.x).astype(float)
        for t in range(ax.t):
            mov = prov.get_region(0, m, t, ref_z, 0, 0, ax.y, 0, ax.x).astype(float)
            sh = phase_cross_correlation(ref, mov, upsample_factor=10)[0]
            shifts[(m, t)] = (float(sh[0]), float(sh[1]))
            dy[m, t], dx[m, t] = shifts[(m, t)]

    def apply_shift(a: np.ndarray, m: int, t: int) -> np.ndarray:
        return ndi_shift(a, shift=shifts[(m, t)], order=1, mode="constant")

    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    for z in range(ax.z):
                        plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x).astype(float)
                        out[m, t, z, c] = apply_shift(plane, m, t)
        res = ds.with_image(ArrayProvider(out))
    else:
        # WHOLE_PLANE unit: ndi_shift needs the whole plane (mode='constant' fills the
        # vacated edge); the shifts are baked from the eager estimate, and fold into the
        # provider fp via the base fingerprint (a base change re-estimates + re-keys).
        fp = stream_fp("drift", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
        res = ds.with_image(MapComputeProvider(
            prov, lambda a, m, t, z, c, *_: apply_shift(a, m, t),
            unit="plane", fp=fp, cache=cache))
    res = res.with_layer(Domain.FRAME, "drift_y", dy)
    return res.with_layer(Domain.FRAME, "drift_x", dx)


register_node(
    _compute_drift, op_key="align.drift", label="Drift Correction", category="registration",
    extra_layers=_layers_drift,
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.FRAME}),
    inputs=[InDataset()], outputs=[OutDataset()],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"t", "y", "x"}),
    description="Rigid drift correction across Timepoints (phase cross-correlation on a "
                "reference plane); stores the per-frame shift as Frame attributes.")



# ── Extract Boundary (Label raster → boundary Points) ───────────────────────────

def _compute_extract_boundary(ctx: EvalContext) -> Dataset:
    """Extract each Label region's outline as boundary Points (contours in 2D via
    ``find_contours``; surface vertices in 3D via ``marching_cubes``) — wraps
    :func:`nodegraph.boundary.extract_boundary` per (m,t,c[,z]) and threads the
    acquisition coordinates the compute-level helper leaves at 0. Ids are global-unique.

    Resolved spec (2026-07-28 socket/domain fix): the compute had always read a ``labels``
    (source Voxel layer) and an output-name param, but ``register_node`` declared **only**
    ``InDataset()`` — so neither had a socket and both were permanently pinned to their
    defaults: this node could not outline a Label raster that wasn't named ``labels``.
    Both are now real sockets. The output-name param is ``name`` (the catalog-wide
    convention — 16 other nodes) rather than the unreachable ``out`` it replaces, and
    follows the ``analysis.accumulate_field`` pattern where **empty means auto-derive**, so
    the default output layer stays ``f"{labels}_boundary"`` and existing graphs are
    unaffected. ``reads_domains``/``adds_domains`` were likewise empty despite the node
    hard-requiring a Voxel raster and adding POINT — so the GUI domain rail showed no chips
    and no red missing-domain validation here, unlike its ``detect.spots`` neighbour whose
    contract is identical.

    NOT changed: the 3D path's ``marching_cubes`` faces stay discarded. That is deliberate,
    not an oversight — this node's contract is a Point table, and it meshes the **union** of
    all non-zero labels as one surface (two touching labels give one shell), whereas
    ``analysis.tessellate``'s ``label_surface`` mode meshes **per region** and keeps the
    faces. label_surface is the MESH route; see :mod:`nodegraph.boundary`."""
    from dataclasses import replace as _dc_replace

    from nodegraph.boundary import extract_boundary
    ds = ctx.inputs[0]
    ax = ds.axes
    is_3d = ctx.is_volume
    layer = ctx.layer("labels")
    out_layer = ctx.layer("name") or f"{layer}_boundary"
    raster_attr = ds.get(Domain.VOXEL, layer)
    if raster_attr is None:
        raise ValueError(f"extract boundary needs a Label raster {layer!r}")
    raster6 = raster_attr.values

    def located(tbl, m, t, c, z=None):
        cols = dict(tbl.columns)
        cols["m"] = np.full(tbl.n, m, np.int64)
        cols["t"] = np.full(tbl.n, t, np.int64)
        cols["c"] = np.full(tbl.n, c, np.int64)
        if z is not None:
            cols["z"] = np.full(tbl.n, float(z))
        return _dc_replace(tbl, columns=cols)

    tables = []
    for m in range(ax.m):
        for t in range(ax.t):
            for c in range(ax.c):
                if is_3d:
                    tbl = extract_boundary(raster6[m, t, :, c].astype(float),
                                           dim="3D", layer=out_layer)
                    if tbl.n:
                        tables.append(located(tbl, m, t, c))
                else:
                    for z in range(ax.z):
                        tbl = extract_boundary(raster6[m, t, z, c].astype(float),
                                               dim="2D", layer=out_layer)
                        if tbl.n:
                            tables.append(located(tbl, m, t, c, z=z))
    zk = "subpixel" if is_3d else "plane_index"
    if not tables:
        merged = point_table(np.zeros((0, 3 if is_3d else 2)), z_kind=zk, layer=out_layer)
    else:
        keys = list(tables[0].columns)
        cols = {k: np.concatenate([tb.columns[k] for tb in tables]) for k in keys}
        cols["id"] = np.arange(len(cols["id"]), dtype=np.int64)   # global-unique ids
        merged = StructureTable(Domain.POINT, cols, layer=out_layer, z_kind=zk)
    return ds.with_structure(merged)


register_node(
    _compute_extract_boundary, op_key="analysis.extract_boundary",
    label="Extract Boundary", category="analysis",
    extra_layers=_layers_extract_boundary,
    # identical contract to detect.spots / detect.particles: a Voxel raster in, Points out
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[InDataset(),
            InString("labels", "Label layer", field=False, default="labels",
                     layer_in=Domain.VOXEL),
            # empty = auto-derive f"{labels}_boundary" (the accumulate_field convention),
            # so the default output layer name is unchanged
            InString("name", "Output layer", field=False, default="")],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="A label region's outline → boundary Points (2D contours / 3D surface "
                "vertices); wraps boundary.extract_boundary. For a MESH of a label raster "
                "use analysis.tessellate (label_surface), which meshes per region.")


# ── Track Linking (frame-to-frame tracking → a Track membership, C3) ──────────

def _compute_track_link(ctx: EvalContext) -> Dataset:
    """Frame-to-frame tracking → a Track membership attached to the Dataset (C3).

    The ``target`` Mode selects label-vs-point linking. Label mode links the per-t
    Label raster (a Domain.VOXEL layer whose ids are global-unique per (m,t,z,c), as
    ``analysis.label`` emits) by maximum IoU overlap; point mode links the Point
    structure by nearest neighbour within ``max_distance`` (µm → the point coordinate
    space). Tracking runs per (m, c) across the whole T axis; track ids from different
    (m,c) runs are offset so they stay unique. The membership is attached via
    ``ds.with_structure(membership.to_table(layer=...))``."""
    from nodegraph.tracking import link_labels, link_points
    from nodegraph.structure import TrackMembership

    ds: Dataset = ctx.inputs[0]
    ax = ds.axes
    modes = ctx.params.get("__modes__", {})
    target = modes.get("target", "label")
    out_layer = ctx.layer("name")
    parts = []                                        # (track_id, t, member_id) triples
    offset = 0

    def _accumulate(mem) -> None:
        nonlocal offset
        if mem.n:
            parts.append((mem.track_id + offset, mem.t, mem.member_id))
            offset += int(mem.track_id.max())         # keep (m,c) runs' track ids unique

    if target == "label":
        src = ctx.layer("labels")
        attr = ds.get(Domain.VOXEL, src)
        if attr is None:
            raise ValueError(f"track.link: no label raster {src!r} on the input Dataset")
        raster6 = attr.values                         # (m,t,z,c,y,x)
        iou = float(ctx.params.get("iou_threshold", 0.0))
        for m in range(ax.m):
            for c in range(ax.c):
                rasters_by_t = {t: raster6[m, t, :, c, :, :] for t in range(ax.t)}
                _accumulate(link_labels(rasters_by_t, iou_threshold=iou))
        member_domain = Domain.LABEL
    else:
        src = ctx.layer("points")
        cols = {k: ds.get(Domain.POINT, k, layer=src)
                for k in ("id", "m", "t", "c", "z", "y", "x")}
        if cols["id"] is None:
            raise ValueError(f"track.link: no Point structure {src!r} on the input Dataset")
        cid, cm, ct, cc, cz, cy, cx = (cols[k].values
                                       for k in ("id", "m", "t", "c", "z", "y", "x"))
        px = ctx.calib("pixel_size_um") or 1.0
        zs = ctx.calib("z_step_um") or 1.0            # µm-scaled coords → max_distance in µm
        coords = np.stack([cz * zs, cy * px, cx * px], axis=1)
        max_d = float(ctx.params.get("max_distance", float("inf")))
        for m in np.unique(cm).tolist():
            for c in np.unique(cc).tolist():
                grp = (cm == m) & (cc == c)
                if not np.any(grp):
                    continue
                positions_by_t = {}
                for t in np.unique(ct[grp]).tolist():
                    sel = grp & (ct == t)
                    positions_by_t[int(t)] = (cid[sel], coords[sel])
                _accumulate(link_points(positions_by_t, max_distance=max_d))
        member_domain = Domain.POINT

    if parts:
        track_id = np.concatenate([p[0] for p in parts])
        t_col = np.concatenate([p[1] for p in parts])
        member_id = np.concatenate([p[2] for p in parts])
    else:
        track_id = t_col = member_id = np.array([], dtype=np.int64)
    membership = TrackMembership(track_id=track_id, t=t_col, member_id=member_id,
                                 member_domain=member_domain)
    return ds.with_structure(membership.to_table(layer=out_layer))


register_node(
    _compute_track_link,
    op_key="track.link", label="Track Linking", category="analysis",
    adds_domains=frozenset({Domain.TRACK}),   # reads Label OR Point (per 'target' mode)
    inputs=[
        InDataset(),
        InFloat("max_distance", "Max distance", unit="um", field=True, default=2.0,
                available_in={"target": frozenset({"point"})}),
        InFloat("iou_threshold", "IoU threshold", unit="", field=True, default=0.0,
                available_in={"target": frozenset({"label"})}),
        # the member source layer, one per target mode (the compute picks by mode)
        InString("labels", "Label layer", field=False, default="labels",
                 layer_in=Domain.VOXEL,
                 available_in={"target": frozenset({"label"})}),
        InString("points", "Point layer", field=False, default="spots",
                 layer_in=Domain.POINT,
                 available_in={"target": frozenset({"point"})}),
        InString("name", "Output layer", field=False, default="tracks",
                 layer_out=(Domain.TRACK,)),
    ],
    outputs=[OutDataset()],
    modes=[Mode("target", ["label", "point"], default="label", label="Track")],
    granularity=Granularity.WHOLE_SERIES,
    kernel_axes=frozenset({"t", "z", "y", "x"}),
    description="Frame-to-frame tracking → a Track membership: label mode links by "
                "maximum IoU overlap, point mode by nearest neighbour within "
                "max_distance (µm). Runs per (m,c) across the whole T axis.",
)


# ── Transfer Domain (move a lattice attribute A→B: reduce / broadcast) ──────────
#
# The from/to dropdowns offer the LATTICE domains ONLY: `execute_transfer` raises
# NotImplementedError on a BridgeStep, so a Label/Point/Track/Mesh entry would be a
# control that always errors. Every lattice pair IS routable (transfer generates a
# reduce/broadcast for any pair), so the cross-product holds no dead choice.
_TRANSFER_DOMAINS: Tuple[str, ...] = tuple(d.value for d in Domain if is_lattice(d))
#: Mirrors nodegraph.reducers.REDUCERS. This node is WHOLE_VOLUME and reduces eagerly,
#: so the non-monoid entries (median / sigma_clip / trimmed_mean) are legal here.
_TRANSFER_REDUCERS: Tuple[str, ...] = (
    "mean", "sum", "max", "min", "median", "count", "first",
    "sigma_clip", "trimmed_mean")


def _compute_transfer_domain(ctx: EvalContext) -> Dataset:
    """Transfer a **lattice** attribute from one domain to another — coarsening reduces
    over the dropped axes, refining broadcasts (a generated :class:`TransferPlan`, wrapped
    as a node). Structure-domain (Label/Point/Track/Mesh) transfers need the geometry
    inputs the bridges carry (``execute_bridge_plan``, C2) and are refused here — which is
    why they are absent from the two dropdowns.

    Resolved spec (2026-07-28 socket fix): ALL FOUR functional params were previously
    unreachable — the node declared only ``InDataset()``, so from the GUI it could only
    ever move ``mask`` from voxel→frame with ``mean``, i.e. it was effectively unusable.
    ``from_domain``/``to_domain``/``reducer`` are **Modes** because each is a closed
    enumeration (the `util.stack`/`analysis.threshold` precedent for a fixed choice list);
    ``attr`` is a layer NAME, so it stays an ``InString`` like every other source-layer
    selector in the catalog."""
    from nodegraph.transfer import lattice_transfer
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {})

    # These three were PARAMS before they became Modes. The GUI could never have written
    # them (no sockets — the defect being fixed), but a headless or hand-authored graph
    # could, and nodegraph.selftest itself did. A silent fallback is impossible: the
    # engine passes the RESOLVED mode state (engine.py:404 `dict(state)`, defaults filled
    # in), so a param could only ever be consulted by second-guessing whether a mode value
    # was chosen or defaulted. Refusing is the honest option — a stale caller gets told
    # exactly what to change instead of silently running with the defaults, which is the
    # very failure this whole pass exists to remove.
    _legacy = [k for k in ("from_domain", "to_domain", "reducer") if k in ctx.params]
    if _legacy:
        raise ValueError(
            f"transfer_domain: {_legacy} are Modes now, not params — pass them as "
            f"modes={{{', '.join(f'{k!r}: ...' for k in _legacy)}}} on the NodeInstance. "
            "(They were params with no socket, so the GUI could never set them; only a "
            "headless caller reaches this.)")

    src = Domain(modes.get("from_domain", "voxel"))
    dst = Domain(modes.get("to_domain", "frame"))
    name = ctx.layer("attr")
    reducer = modes.get("reducer", "mean")
    if not (is_lattice(src) and is_lattice(dst)):
        raise ValueError(
            f"transfer_domain moves LATTICE attributes; {src.value}→{dst.value} needs a "
            f"structure bridge — use nodegraph.bridges / execute_bridge_plan (C2)")
    # wire-node-v2 §5b remedy (b): `reducer` only bites when the transfer COARSENS (there
    # are axes to collapse). A pure refinement/identity broadcasts the value untouched, so
    # a non-default reducer there would be a live control the kernel silently ignores —
    # refuse instead of lying. `available_in` can only gate a RECTANGLE of mode values and
    # "src drops an axis vs dst" is not one (it would have to admit frame→voxel to admit
    # voxel→frame), so the declarative remedy (a) is not expressible here.
    # Checked BEFORE the layer lookup: it is a pure configuration error, and the user
    # should see it whether or not the named attribute happens to exist.
    if not (axes_of(src) - axes_of(dst)) and reducer != "mean":
        raise ValueError(
            f"transfer_domain: {src.value}→{dst.value} drops no axis (a pure "
            f"broadcast/identity), so reducer={reducer!r} would be ignored — set "
            f"reducer back to 'mean'")
    layer = ds.get(src, name)
    if layer is None:
        raise ValueError(f"transfer_domain: no {src.value} attribute {name!r}")
    return ds.with_attribute(lattice_transfer(layer, dst, ds.axes, reducer))


register_node(
    _compute_transfer_domain, op_key="transform.transfer_domain",
    label="Transfer Domain", category="transform",
    # reads/adds stay EMPTY on purpose: both are per-INSTANCE here (whatever from_domain/
    # to_domain say) while NodeSpec's declarations are per-TYPE — the same reason
    # `track.link` leaves reads_domains empty for its Label-OR-Point target mode.
    inputs=[InDataset(),
            # the source DOMAIN is the `from_domain` lever, so the picker resolves
            # it from that mode rather than from a fixed domain
            InString("attr", "Attribute", field=False, default="mask",
                     layer_in_mode="from_domain")],
    outputs=[OutDataset()],
    modes=[Mode("from_domain", list(_TRANSFER_DOMAINS), default="voxel", label="From"),
           Mode("to_domain", list(_TRANSFER_DOMAINS), default="frame", label="To"),
           Mode("reducer", list(_TRANSFER_REDUCERS), default="mean", label="Reduce")],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset(),
    description="Move a lattice attribute between domains (reduce coarsens over dropped "
                "axes / broadcast refines); lattice only — structure bridges → use C2.")


# ── Frame slice (per-frame-T Simulation specialization) ────────────────────────

def _compute_zone_frame(ctx: EvalContext) -> Dataset:
    """Slice frame ``__frame__`` (a single timepoint) from the input — the per-frame-T
    slice a zone stamps per iteration (``nodegraph.zones.FRAME_OP``): inside a Simulation
    zone body, iteration *t* reads frame *t* of the (T-stacked) input while the Sim
    In/Out feedback carries state. Yields a t==1 Dataset — the image via a lazy
    :class:`_FrameView`; any t-bearing lattice attribute layer is sliced (keepdim) so it
    is not dropped; layers without a ``t`` axis (and structure layers) pass through."""
    ds: Dataset = ctx.inputs[0]
    t = int(ctx.params.get("__frame__", 0))
    ax = ds.axes
    if not (0 <= t < ax.t):
        raise ValueError(f"zone.frame: frame {t} out of range [0, {ax.t}) — set the "
                         f"zone's iterations to the input's T")
    out = Dataset(axes=replace(ax, t=1), metadata=dict(ds.metadata))
    if ds.image is not None:
        out = out.with_image(_FrameView(ds.image, t))
    for attr in ds.attributes.values():
        axset = axes_of(attr.domain) or frozenset()
        if is_lattice(attr.domain) and "t" in axset:
            order = [a for a in AXIS_ORDER if a in axset]
            sliced = np.take(attr.values, [t], axis=order.index("t"))   # keepdim → t==1
            out = out.with_layer(attr.domain, attr.name, sliced, attr.layer)
        else:
            out = out.with_layer(attr.domain, attr.name, attr.values, attr.layer)
    return out


register_node(
    _compute_zone_frame, op_key="zone.frame", label="Frame (per-t)", category="zone",
    inputs=[InDataset()], outputs=[OutDataset()],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    meta_transform=_meta_frame_slice,
    description="Per-frame-T slice: inside a zone body, iteration t yields frame t of "
                "its (T-stacked) input (t==1). Drives the Simulation per-frame pattern.")


# ══ Ported v1 analysis kernels (Phase 7 capability gaps, V2.05 §2) ══════════════
#
# Each wraps a vendored pure-compute kernel in `nodegraph.kernels.*` (byte-verbatim
# ND2Studios v1.45), lazily imported inside the compute so the engine core stays
# importable without the heavy per-kernel deps. The kernels act on ONE frame/volume;
# the node owns the m/t/c loop, derives voxel_size from calibration, and attaches the
# result (image / Voxel mask / Point / Label / Track). See each kernel's `.md`.


# ── Particle detection (LoG maxima or component centroids → Points) ────────────

def _compute_particles(ctx: EvalContext) -> Dataset:
    """Particle / small-object detection → a Point structure. A general blob/particle
    detector (beads, puncta, foci, …) ported from the v1 ``bead_detect`` kernel. ``mode``
    picks LoG local-maxima (sub-voxel parabola refinement) or connected-component centroids
    (radial-symmetry refinement in 3D). 2D detects per plane (``z_kind="plane_index"``), 3D
    in the volume (subpixel z). Ids are global-unique across the whole detection.

    Resolved spec: category analysis; op ``detect.particles``; **Point** output; DimMode
    lever (2D WHOLE_PLANE / 3D WHOLE_VOLUME, ``kernel_axes`` per dim). ``min_distance``
    µm→px via ``pixel_size_um`` (doubles as the LoG σ and the NMS radius, in voxels);
    ``voxel_size_um=(z_step_um, pixel_size_um, pixel_size_um)`` slowest-first. Kernel:
    :func:`nodegraph.kernels.bead_detect.detect_beads` (numpy/scipy/numba)."""
    from nodegraph.kernels.bead_detect import detect_beads
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("particle detection needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox = (zs, px, px)                                  # (dz, dy, dx) slowest-first
    min_dist_px = max(1.0, to_pixels_v2(float(ctx.params.get("min_distance", 0.5)),
                                        "um", pixel_size_um=px))
    kp = {
        "detect_mode": modes.get("mode", "log"),
        "min_distance_px": min_dist_px,
        "threshold": float(ctx.params.get("threshold", 0.0)),
        "min_intensity": float(ctx.params.get("min_intensity", 0.0)),
        "subpixel": bool(ctx.params.get("subpixel", True)),
        "min_size": max(1, int(ctx.params.get("min_size", 1))),
    }
    layer = ctx.layer("name")
    is_3d = ctx.is_volume
    tables = []
    for m in range(ax.m):
        for t in range(ax.t):
            for c in range(ax.c):
                kpp = {**kp, "m": m, "frame": t}
                if is_3d:
                    vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y,
                                                 0, ax.x).astype(float)
                    pts, _rows = detect_beads(vol, vox, kpp)
                    if len(pts):
                        tables.append(point_table(pts[:, :3], m=m, t=t, c=c,
                                                  z_kind="subpixel", layer=layer))
                else:
                    for z in range(ax.z):
                        plane = prov.get_region(0, m, t, z, c, 0, ax.y,
                                                0, ax.x).astype(float)
                        pts, _rows = detect_beads(plane, vox, kpp)
                        if len(pts):
                            # 2D fallback forces the z-column to 0 → record the true
                            # plane index (take the (y,x) columns, stamp z=z)
                            tables.append(point_table(pts[:, 1:3], z=z, m=m, t=t, c=c,
                                                      z_kind="plane_index", layer=layer))
    zk = "subpixel" if is_3d else "plane_index"
    if not tables:
        merged = point_table(np.zeros((0, 3 if is_3d else 2)), z_kind=zk, layer=layer)
    else:
        cols = {k: np.concatenate([tb.columns[k] for tb in tables])
                for k in tables[0].columns}
        cols["id"] = np.arange(len(cols["id"]), dtype=np.int64)   # global-unique ids
        merged = StructureTable(Domain.POINT, cols, layer=layer, z_kind=zk)
    return ds.with_structure(merged)


register_node(
    _compute_particles, op_key="detect.particles", label="Particle Detection",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InFloat("min_distance", "Min distance", unit="um", field=True, default=0.5,
                derive="0.61*(emission_nm or 520)/(na or 1.4)/1000"),
        InFloat("threshold", "Threshold", unit="", field=True, default=0.0),
        InFloat("min_intensity", "Min intensity", unit="", field=True, default=0.0),
        InInt("min_size", "Min size", unit="", field=False, default=1),
        InBool("subpixel", "Subpixel", field=False, default=True),
        InString("name", "Output layer", field=False, default="particles",
                 layer_out=(Domain.POINT,)),
    ],
    outputs=[OutDataset()],
    modes=[DimMode(), Mode("mode", ["log", "components"], default="log")],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="Particle/small-object detection → Points (LoG local-maxima or component "
                "centroids, sub-voxel refined; beads, puncta, foci); min distance = LoG σ "
                "+ NMS radius (µm→px); 2D per-plane vs 3D volume (ported v1 kernel).")


# ── Histogram Threshold Segmenter (2D → mask + labels + region table) ──────────

#: the bit depths the vendored kernel's LUT tables support (``BIT_DEPTH_MAX``).
_KERNEL_BIT_DEPTHS = (8, 10, 12, 14, 16)


def _snap_bit_depth(bits: Optional[float]) -> int:
    """The smallest kernel-supported depth that still holds ``bits`` (11 → 12, 12 → 12),
    or 16 when unknown. The kernel indexes ``BIT_DEPTH_MAX`` by exact key, so an odd
    sensor depth must round UP — never down, which would truncate the histogram LUT
    below the data and silently drop the bright tail out of every percentile."""
    if not bits:
        return 16
    b = int(bits)
    return next((d for d in _KERNEL_BIT_DEPTHS if d >= b), 16)


#: the (low-side, high-side) threshold param of each histogram-threshold method — what
#: ``direction`` selects, and what the inspector shows (mirrored by ``available_in``).
_THRESH_PARAMS = {"single": ("low", "high"),
                  "hysteresis": ("permissive", "strict"),
                  "percentile": ("percentile_low", "percentile_high"),
                  "relative": ("fraction_low", "fraction_high")}
#: image-relative defaults (percentile of the plane histogram / × its median) — **the v1
#: segmenter's own numbers** (`HistogramThresholdPipeline.get_params`: percentile 5/95,
#: fraction 0.30/1.50), not re-invented here. The absolute raw-count params
#: (low/high/strict/permissive) are deliberately ABSENT — v1 defaulted them to 30/80/50/500
#: for its 12-bit data, which is meaningless on another camera, so leaving one unset must
#: ask rather than guess. Keep in lockstep with the socket ``default=``s below (the engine
#: passes only stored params).
_THRESH_DEFAULTS = {"percentile_low": 5.0, "percentile_high": 95.0,
                    "fraction_low": 0.30, "fraction_high": 1.50}


def _compute_histogram_threshold(ctx: EvalContext) -> Dataset:
    """Histogram-driven threshold segmentation (ported v1 kernel) → a Voxel ``mask`` +
    a Label raster + a per-region **Label** table, all in one node (the v1 segmenter's
    bundle). **2D-only** — each ``(m,t,z,c)`` plane is segmented independently: one of
    4 methods (single / hysteresis / percentile / relative) × 4 directions, fixed-order
    morphology cleanup, 8-connected CCL, and region props.

    Thresholds (``low``/``high``/``strict``/``permissive``) are in the image's **raw
    integer counts** — the plane is cast to ``uint16`` (rint+clip to 0..65535) before
    the kernel, whose bit-depth check is set non-strict (``make_config``). The declared
    depth is **read from the file** (``bit_depth`` calibration ← ND2
    ``bitsPerComponentSignificant``, usually **12**), snapped up to a kernel-supported
    LUT (8/10/12/14/16) and overridable per node; only a silent metadata falls back to 16.
    That sizes the percentile LUT to the real sensor range and retires the bogus "max 192
    is <5% of 16-bit range" warning on dim 12-bit data (2026-07-28). Ids are
    global-unique across planes, and the two Voxel rasters are stored at the kernel's own
    dtypes — ``mask`` ``uint8`` (0/1) and the label raster ``int32`` — never upcast to
    int64, which cost 12 B/voxel of nothing (5 GiB per raster on a 16-position 2048²×10
    series). Morphology radii convert µm→px; area filters µm²→px².
    ``min_area``/``max_area`` bound the region size window (``max_area`` 0 = no upper
    limit; a sub-pixel value is refused, since rounding it to 0 px² would read as that
    off-sentinel and invert the request — restored socket 2026-07-28, the compute read
    the param all along but nothing could set it). Because that window is authored in
    **µm²**, the Label table reports **``area_um2``** beside the raw ``area`` (px²), so
    the domain data can be read in the same unit the filter is typed in; and a window
    that discards *every* region raises with the areas actually measured on one plane
    instead of returning a blank mask (2026-07-28).

    The optional **``raw``** Dataset input re-measures the region table's intensity columns
    on those pixels while the mask/labels stay on the main input's — segment on the chain
    you tuned, report intensities from the unenhanced source (:func:`_intensity_provider`).

    **Threshold params are method-specific** (``available_in`` gates each one to its
    method/direction, so the inspector shows only the 1–2 that the current mode pair
    actually uses) and **0 = unset** — the GUI's spin boxes cannot express ``None``, and
    a threshold of exactly 0 counts / the 0th percentile / 0× the median is meaningless,
    so zero reads as "not set". An unset param falls back to the image-relative default
    where one exists (``percentile``/``relative``); the absolute raw-count methods
    (``single``/``hysteresis``) have no image-independent default, so an unset one raises
    an actionable error naming the fields to fill in (fix 2026-07-28 — the vendored
    ``ThresholdConfig`` raised a bare "requires `strict` and `permissive`" traceback).

    Resolved spec: category analysis; op ``analysis.histogram_threshold``; no lever
    (inherently per-plane), ``WHOLE_PLANE`` ``{y,x}``. Kernel:
    :mod:`nodegraph.kernels.histogram_threshold` (numpy/scipy/skimage)."""
    from nodegraph.kernels.histogram_threshold import (
        HistogramThresholdSegmenter, _measure_regions, make_config,
    )
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("histogram threshold needs an image provider on its input Dataset")
    # optional `raw`: the mask/labels stay on the main input's pixels (thresholds are the
    # user's, on the chain they tuned), but the region table's intensity columns are
    # re-measured on the raw pixels — the same vendored `_measure_regions`, unquantized
    # (a measurement has no reason to round through uint16 the way a raw-count cut does).
    raw_prov, on_raw = _intensity_provider(ctx, ds)
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    # NO fabricated calibration. v1 passed ``voxel_size=None`` when pixel_size_um was
    # absent (areas then reported in px only); the old ``or 0.1`` here invented a scale,
    # which silently rescaled every µm/µm² filter by (real/0.1)² on an uncalibrated TIFF.
    px = ctx.calib("pixel_size_um")
    pxv = float(px) if px else 0.0
    # The SIGNIFICANT sensor depth, read from the file's metadata (most ND2s are 12-bit)
    # — not assumed to be 16. It sets the kernel's percentile LUT range and its bit-depth
    # validation, so declaring 16 on 12-bit data both mis-sized the LUT and produced the
    # bogus "max 192 is <5% of 16-bit range" warning. `bit_depth` is a v2 calibration key,
    # so this read is memo-fenced: re-ingesting the same file at another depth re-keys.
    bd_override = ctx.params.get("bit_depth")
    bd_from_meta = ctx.calib("bit_depth")
    bit_depth = (_snap_bit_depth(bd_override) if bd_override
                 else _snap_bit_depth(bd_from_meta))
    #: authoritative = the depth came from the file (or the user), so the kernel's
    #: "may be lower-bit than declared" heuristic has nothing left to warn about — a dim
    #: 12-bit frame legitimately maxes in the low hundreds. Kept when we defaulted to 16.
    bd_authoritative = bool(bd_override or bd_from_meta)

    def _uncalibrated(name: str, v: float, unit: str) -> str:
        return (f"histogram threshold: {name}={v:g} {unit} cannot be converted to pixels "
                "— this image carries no `pixel_size_um` calibration. Supply it upstream "
                "(an ND2 carries it; a bare TIFF usually does not), or set the param to 0 "
                f"to switch that filter off. (v1 authored these filters in PIXELS; v2 "
                f"authors them in {unit} so they follow the objective.)")

    def _cleanup_um(name: str) -> float:
        """A cleanup param in µm/µm², resolved override → ``derive`` → default. The
        derives encode **the v1 segmenter's own pixel defaults** (1 px opening, 2 px
        closing, 50 px² holes, 100 px² min area) expressed in µm, so the shipped
        behaviour matches v1 while still following the objective. Uncalibrated ⇒ the
        derives yield 0 ⇒ cleanup is simply off, v1's ``voxel_size=None`` path.
        (``channel(0)`` because none of these vary per channel — pixel size only.)"""
        return float(ctx.channel(0).param(name, 0.0) or 0.0)

    def _rad(name: str) -> int:                               # µm radius → px
        v = _cleanup_um(name)
        if v == 0.0:
            return 0
        if pxv <= 0.0:
            raise ValueError(_uncalibrated(name, v, "µm"))
        return int(round(to_pixels_v2(v, "um", pixel_size_um=pxv)))

    def _area(name: str) -> int:                              # µm² area → px²
        v = _cleanup_um(name)
        if v == 0.0:
            return 0
        if pxv <= 0.0:
            raise ValueError(_uncalibrated(name, v, "µm²"))
        return int(round(v / (pxv * pxv)))

    def _opt(name: str):
        """A method-specific threshold param, or ``None`` when unset. **0 = unset** (see
        the docstring); an unset param falls back to its image-relative default when the
        method has one — the engine does NOT inject socket defaults into ``ctx.params``,
        so `_THRESH_DEFAULTS` (mirrored by the socket ``default=``s) is the real source."""
        v = ctx.params.get(name)
        if v is None or v == "" or float(v) == 0.0:
            return _THRESH_DEFAULTS.get(name)
        return v

    # area filters (µm² → px²). ``min_area``/``min_hole_size`` quantizing to 0 px² is
    # harmless — "drop objects below half a pixel" IS a no-op. ``max_area`` is the
    # opposite: 0 is its off-sentinel, so a sub-pixel value would silently INVERT the
    # request (keep everything instead of dropping all but sub-pixel specks) → refuse.
    min_area_px, max_area_px = _area("min_area"), _area("max_area")
    max_area_um2 = _cleanup_um("max_area")
    if max_area_um2 > 0.0 and max_area_px == 0:
        raise ValueError(
            f"histogram threshold: max_area={max_area_um2:g} µm² is under one pixel "
            f"({pxv * pxv:g} µm²) — it quantizes to 0 px², which is this filter's "
            "OFF value, so it would keep every blob instead of dropping the large "
            "ones. Raise it, or set exactly 0 to disable the filter deliberately.")
    if max_area_px > 0 and min_area_px > max_area_px:
        raise ValueError(
            f"histogram threshold: min_area ({min_area_px} px²) exceeds max_area "
            f"({max_area_px} px²) — the size window is empty, so every region would be "
            "discarded. Widen the window (max_area 0 = no upper limit).")

    method = modes.get("method", "percentile")
    direction = modes.get("direction", "above")
    # Pre-flight the method/direction contract HERE, so a missing threshold reads as an
    # instruction instead of a dataclass traceback out of the vendored kernel.
    if method not in _THRESH_PARAMS:
        raise ValueError(f"histogram threshold: unknown method {method!r} — expected one "
                         f"of {', '.join(_THRESH_PARAMS)}.")
    if method == "hysteresis" and direction not in ("below", "above"):
        raise ValueError(
            "histogram threshold: method=hysteresis supports direction=below/above only "
            f"(got {direction!r}) — hysteresis grows a permissive region out of a strict "
            "seed, which has no two-sided form. Use method=single or percentile for "
            f"direction={direction}.")
    lo, hi = _THRESH_PARAMS[method]
    if method == "hysteresis":
        need = [lo, hi]                     # both seeds always, whichever direction
    elif direction == "below":
        need = [lo]
    elif direction == "above":
        need = [hi]
    else:
        need = [lo, hi]                     # between / outside are two-sided
    missing = [n for n in need if _opt(n) is None]
    if missing:
        raise ValueError(
            f"histogram threshold: method={method}, direction={direction} needs "
            + " and ".join(f"`{n}`" for n in missing)
            + f" — set {'it' if len(missing) == 1 else 'them'} in the node's Parameters "
            "(0 = unset). low/high/strict/permissive are absolute RAW INTEGER COUNTS, so "
            "they have no image-independent default; the percentile (0–100) and relative "
            "(× the plane median) methods self-scale and work out of the box.")
    if method == "hysteresis":
        # v1 (`histothresh.thresholds.threshold_hysteresis`) requires strict ≤ permissive
        # for `below` and strict ≥ permissive for `above` — the CORE cut is the more
        # extreme one in the direction being selected. It raises a bare comparison error;
        # say what the two seeds mean instead, because the ordering reads as inverted.
        s_v, p_v = float(_opt(hi)), float(_opt(lo))            # hi=strict, lo=permissive
        if (s_v > p_v) if direction == "below" else (s_v < p_v):
            raise ValueError(
                f"histogram threshold: hysteresis {direction} needs strict "
                f"{'≤' if direction == 'below' else '≥'} permissive, but strict={s_v:g} "
                f"and permissive={p_v:g}. `strict` is the CORE cut (pixels definitely "
                "inside the mask) and `permissive` the FRINGE, kept only where it touches "
                f"a core region — so selecting {direction} makes the core the "
                f"{'darker' if direction == 'below' else 'brighter'} of the two. Swap them.")

    cfg = make_config(
        method=method,
        direction=direction,
        bit_depth=bit_depth,
        low=_opt("low"), high=_opt("high"),
        strict=_opt("strict"), permissive=_opt("permissive"),
        percentile_low=_opt("percentile_low"),
        percentile_high=_opt("percentile_high"),
        fraction_low=_opt("fraction_low"), fraction_high=_opt("fraction_high"),
        min_area=min_area_px, max_area=max_area_px,
        opening_radius=_rad("opening_radius"),
        closing_radius=_rad("closing_radius"),
        min_hole_size=_area("min_hole_size"))
    seg = HistogramThresholdSegmenter(cfg)
    # v1 parity: no pixel size ⇒ no µm² measurement (the kernel then omits area_um2 and
    # the Label column is NaN) rather than a fabricated area.
    vox = (pxv, pxv) if pxv > 0.0 else None
    layer = ctx.layer("name")

    def _alloc(dtype, what: str) -> np.ndarray:
        """A full-series Voxel raster at the kernel's OWN dtype (uint8 mask / int32 CCL
        raster — what ``SegmentationResult`` already carries). Upcasting both to int64
        spent 12 B/voxel on zero information: a (16,1,10,1,2048,2048) series is 5.0 GiB
        **per raster**, so a big multi-position stack died in ``np.zeros`` before the
        first plane was read (fix 2026-07-28). Numpy's own MemoryError names the shape
        but not a way out, so re-raise with the levers."""
        try:
            return np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=dtype)
        except MemoryError as ex:                             # incl. _ArrayMemoryError
            gib = (ax.m * ax.t * ax.z * ax.c * ax.y * ax.x
                   * np.dtype(dtype).itemsize) / float(1 << 30)
            raise MemoryError(
                f"histogram threshold: the {what} raster needs {gib:.2f} GiB "
                f"({ax.m}×{ax.t}×{ax.z}×{ax.c}×{ax.y}×{ax.x} voxels of "
                f"{np.dtype(dtype).name}) and this machine cannot allocate it. This node "
                "segments the WHOLE series eagerly, so narrow the input first: util.crop "
                "(a region and/or a z range), channel.select or a chK tap (one channel), "
                "or a per-position graph. A previously viewed result is also still held "
                "in the cache — pulling a smaller graph releases it.") from ex

    mask6 = _alloc(np.uint8, "mask")                          # binary: 0/1
    labels6 = _alloc(np.int32, "label")                       # global-unique CCL ids
    tables = []
    offset = 0
    last: Any = None                                          # (coords, plane) for diagnosis
    for m, t, z, c in _each_plane_p(ctx, ax, "segmenting"):
        plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
        # thresholds are in RAW INTEGER counts. A raw-count image stored as float
        # (ND2/deconvolved) rounds losslessly; a NORMALIZED [0,1] float would quantize
        # to {0,1} and make raw-count thresholds meaningless — surface that loudly
        # instead of silently producing garbage (review 2026-07-23).
        if not np.array_equal(plane, np.rint(plane)) and float(np.max(plane)) <= 1.0:
            raise ValueError(
                "histogram threshold needs raw integer counts, but the input looks "
                f"normalized (fractional, max={float(np.max(plane)):.3g} ≤ 1.0). "
                "Run it before a normalize/deconvolve step, or on the raw image; its "
                "thresholds (low/high/strict/permissive) are in raw counts.")
        plane_int = np.rint(np.clip(plane, 0, 65535)).astype(np.uint16)
        last = ((m, t, z, c), plane_int)
        with warnings.catch_warnings():
            if bd_authoritative:
                # The kernel warns "max N is <5% of B-bit range … may be lower-bit than
                # declared" — a guess-checking heuristic. With the depth read from the
                # file that guess is settled, and a genuinely dim frame would fire it on
                # every plane. The OVER-range warning is NOT filtered: values above the
                # declared max mean the data really was rescaled, which does invalidate
                # raw-count thresholds. (v1 blanket-suppressed every warning here.)
                warnings.filterwarnings("ignore", message=r".*is <5% of.*",
                                        category=UserWarning)
            res = seg.run(plane_int, voxel_size=vox)
        regions = res.regions
        if on_raw:
            regions = _measure_regions(
                res.labels, raw_prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x), vox)
        mask6[m, t, z, c] = res.mask                          # bool → uint8 0/1
        lab = res.labels.astype(np.int32, copy=True)           # kernel already int32
        lab[lab > 0] += offset
        labels6[m, t, z, c] = lab
        k = int(res.labels.max())                             # labels are 1..k
        if regions:
            n = len(regions)
            tables.append(StructureTable(Domain.LABEL, {
                "id": np.array([r["label_id"] + offset for r in regions], dtype=np.int64),
                "m": np.full(n, m, dtype=np.int64),
                "t": np.full(n, t, dtype=np.int64),
                "c": np.full(n, c, dtype=np.int64),
                "z": np.full(n, z, dtype=float),
                "y": np.array([r["centroid_y"] for r in regions], dtype=float),
                "x": np.array([r["centroid_x"] for r in regions], dtype=float),
                "area": np.array([r["area_px"] for r in regions], dtype=np.int64),
                # the CALIBRATED area next to the raw pixel count: min_area/max_area are
                # authored in µm², so the domain data the user filters on must be readable
                # in µm² too (the kernel measures it; the node used to drop it). This is
                # what makes "pick a threshold from the spreadsheet" a real workflow.
                "area_um2": np.array([r.get("area_um2", float("nan"))
                                      for r in regions], dtype=float),
                "mean_intensity": np.array([r["mean_intensity"] for r in regions],
                                           dtype=float),
            }, layer=layer, z_kind="plane_index"))
        offset += k
    # An area window that discards EVERY region leaves a blank viewer with no hint of
    # which knob did it. Re-segment the last plane with the area filters off (one plane,
    # only on this dead-end path) and report the areas actually present, in the same µm²
    # the user typed — the threshold, not the area, is the culprit when that is empty too.
    if not tables and (min_area_px > 0 or max_area_px > 0) and last is not None:
        coords, plane_int = last
        raw = HistogramThresholdSegmenter(replace(cfg, min_area=0, max_area=0)).run(
            plane_int, voxel_size=vox)
        window = (f"min_area={_cleanup_um('min_area'):g} µm² ({min_area_px} px²)"
                  + (f", max_area={max_area_um2:g} µm² ({max_area_px} px²)"
                     if max_area_px > 0 else ""))
        if raw.regions:
            areas = sorted(float(r.get("area_um2", float("nan"))) for r in raw.regions)
            raise ValueError(
                f"histogram threshold: the area window ({window}) discarded every region "
                f"in the whole series. At (m,t,z,c)={coords} the threshold DOES find "
                f"{len(areas)} region(s), but they measure {areas[0]:.3g}–{areas[-1]:.3g} "
                f"µm² (median {float(np.median(areas)):.3g}) — all outside your window. "
                "min_area defaults to the v1 cleanup (100 px² worth of µm²), so pin it to "
                "0 in the inspector to see every region's `area_um2` on the Label domain "
                "in the Spreadsheet, then pick the cut from those numbers.")
        raise ValueError(
            f"histogram threshold: nothing was segmented anywhere, and the area window "
            f"({window}) is not the cause — with the area filters off, "
            f"(m,t,z,c)={coords} still yields no region, so the THRESHOLD is what "
            "rejected everything. Check method/direction and the threshold value against "
            "the image's raw counts (percentile self-scales; low/high/strict/permissive "
            "are absolute counts).")
    mask_layer = ctx.layer("mask_name")
    if mask_layer == layer:
        raise ValueError(
            f"histogram threshold: the mask and label layers are both named "
            f"{mask_layer!r} — the second write would overwrite the first. Give them "
            "different names (defaults: `mask` / `labels`).")
    out = (ds.with_layer(Domain.VOXEL, mask_layer, mask6)
             .with_layer(Domain.VOXEL, layer, labels6))
    if tables:
        cols = {kk: np.concatenate([tb.columns[kk] for tb in tables])
                for kk in tables[0].columns}
        out = out.with_structure(StructureTable(Domain.LABEL, cols, layer=layer,
                                                z_kind="plane_index"))
    return out


register_node(
    _compute_histogram_threshold, op_key="analysis.histogram_threshold",
    label="Histogram Threshold", category="analysis",
    reads_domains=frozenset({Domain.VOXEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    inputs=[
        InDataset(),
        # measurement-only pixel override: the mask/labels still come from `data`
        _InRaw(),
        # The significant sensor depth, read from the file (ND2 bitsPerComponentSignificant
        # → usually 12; TIFF bits-per-sample). 0 = use the metadata; a non-zero value
        # overrides it, for a file whose header lies or a rescaled TIFF. v1 exposed this as
        # a plain int param defaulting to 12 — v2 derives it instead, and only asks when
        # the metadata is silent (then 16).
        InInt("bit_depth", "Bit depth", unit="", field=False, default=0,
              derive="bit_depth or 16"),
        # Output layer names. Both Voxel rasters were hard-named by the compute with no
        # socket to set them, so a graph could not carry two thresholded results (they
        # collided on `mask`/`labels`). Defaults are exactly the old fallbacks.
        InString("mask_name", "Mask layer", field=False, default="mask",
                 layer_out=(Domain.VOXEL,)),
        InString("name", "Label layer", field=False, default="labels",
                 layer_out=(Domain.VOXEL, Domain.LABEL)),
        # Threshold params are method-specific: each is gated to the method that reads it
        # and to the directions that need that side, so the inspector shows the 1–2 live
        # fields instead of all 8 (0 = unset; defaults mirror `_THRESH_DEFAULTS`).
        InInt("low", "Low", unit="", field=False, default=0,
              available_in={"method": frozenset({"single"}),
                            "direction": frozenset({"below", "between", "outside"})}),
        InInt("high", "High", unit="", field=False, default=0,
              available_in={"method": frozenset({"single"}),
                            "direction": frozenset({"above", "between", "outside"})}),
        InInt("strict", "Strict", unit="", field=False, default=0,
              available_in={"method": frozenset({"hysteresis"})}),
        InInt("permissive", "Permissive", unit="", field=False, default=0,
              available_in={"method": frozenset({"hysteresis"})}),
        InFloat("percentile_low", "Percentile low", unit="", field=False, default=5.0,
                available_in={"method": frozenset({"percentile"}),
                              "direction": frozenset({"below", "between", "outside"})}),
        InFloat("percentile_high", "Percentile high", unit="", field=False, default=95.0,
                available_in={"method": frozenset({"percentile"}),
                              "direction": frozenset({"above", "between", "outside"})}),
        InFloat("fraction_low", "Fraction low", unit="", field=False, default=0.30,
                available_in={"method": frozenset({"relative"}),
                              "direction": frozenset({"below", "between", "outside"})}),
        InFloat("fraction_high", "Fraction high", unit="", field=False, default=1.50,
                available_in={"method": frozenset({"relative"}),
                              "direction": frozenset({"above", "between", "outside"})}),
        # Spatial cleanup. v1 shipped this ON (min_area 100 px², opening 1 px, closing
        # 2 px, min_hole 50 px²) and v2 shipped it all at 0 = off, which is why the same
        # image segmented to raw speckle here. The derives restore v1's numbers while
        # keeping them µm-authored, so they follow the objective instead of the sensor;
        # `or 0` means an UNCALIBRATED image resolves them to 0 = off (v1's None path).
        InFloat("min_area", "Min area", unit="um2", field=True, default=0.0,
                derive="100*(pixel_size_um or 0)**2"),        # v1: 100 px²
        # the v1 upper-area filter, previously read by the compute with no socket to set
        # it (so always off): drops blobs LARGER than this (0 = no upper limit, v1 default)
        InFloat("max_area", "Max area", unit="um2", field=True, default=0.0),
        InFloat("opening_radius", "Opening radius", unit="um", field=True, default=0.0,
                derive="1*(pixel_size_um or 0)"),             # v1: 1 px
        InFloat("closing_radius", "Closing radius", unit="um", field=True, default=0.0,
                derive="2*(pixel_size_um or 0)"),             # v1: 2 px
        InFloat("min_hole_size", "Min hole", unit="um2", field=True, default=0.0,
                derive="50*(pixel_size_um or 0)**2"),         # v1: 50 px²
    ],
    outputs=[OutDataset()],
    modes=[Mode("method", ["single", "hysteresis", "percentile", "relative"],
                default="percentile"),
           Mode("direction", ["below", "above", "between", "outside"], default="above")],
    granularity=Granularity.WHOLE_PLANE, kernel_axes=frozenset({"y", "x"}),
    description="Histogram threshold segmentation (2D per-plane) → Voxel mask + Label "
                "raster + region table; 4 methods × 4 directions, morphology cleanup "
                "(ported v1 kernel). low/high/strict/permissive are in raw integer "
                "counts (0 = unset, no default — the percentile/relative methods "
                "self-scale to the image); only the current method's params are shown. "
                "Optional `raw` input re-measures the region intensities on those pixels.")


# ── Registration / stabilization (register-once on a ref channel, apply-to-all) ─

def _compute_stabilize(ctx: EvalContext) -> Dataset:
    """Full registration / drift-stabilization (ported v1 ``registration`` kernel):
    estimate a per-frame transform bundle on a **reference channel** (mid-z of the T
    series) and apply the *same* bundle to every channel and z — the register-once/
    apply-to-all design that preserves colocalization. Axis-preserving; the per-frame
    ``(Δy, Δx)`` translation is stored as Frame attributes (``drift_y``/``drift_x``).

    Richer than :func:`_compute_drift` (``align.drift``): ``model`` ∈ translation
    (phase-corr) / euclidean·affine (ECC) / feature (ORB+RANSAC), and ``reference`` ∈
    first / previous (cumulative) / mean / template. WHOLE_SERIES; pixel-space (no
    calibration — pure geometric registration, like ``align.drift``). Kernel:
    :mod:`nodegraph.kernels.registration` (numpy/scipy/skimage/cv2)."""
    from nodegraph.kernels.registration import apply_series, estimate_series
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("stabilize needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    model = modes.get("model", "translation")
    reference = modes.get("reference", "previous")
    upsample = max(1, int(ctx.params.get("upsample", 20)))
    highpass = float(ctx.params.get("highpass_sigma", 2.0))
    min_conf = float(ctx.params.get("min_confidence", 0.0))
    ref_c = min(max(0, int(ctx.params.get("ref_channel", 0))), ax.c - 1)
    ref_z = ax.z // 2
    out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
    dy = np.zeros((ax.m, ax.t), dtype=float)
    dx = np.zeros((ax.m, ax.t), dtype=float)

    def series_of(m: int, z: int, c: int) -> np.ndarray:
        return np.stack([prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x).astype(float)
                         for t in range(ax.t)])

    for m in range(ax.m):
        tf = estimate_series(series_of(m, ref_z, ref_c), model=model,
                             reference=reference, upsample=upsample,
                             highpass_sigma=highpass, min_confidence=min_conf)
        shifts = np.asarray(tf["shifts"], dtype=float)
        dy[m, :], dx[m, :] = shifts[:, 0], shifts[:, 1]
        for c in range(ax.c):
            for z in range(ax.z):
                out[m, :, z, c] = apply_series(series_of(m, z, c), tf)
    res = ds.with_image(ArrayProvider(out)).with_layer(Domain.FRAME, "drift_y", dy)
    return res.with_layer(Domain.FRAME, "drift_x", dx)


register_node(
    _compute_stabilize, op_key="registration.stabilize", label="Registration",
    extra_layers=_layers_drift,
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.FRAME}),
    category="registration",
    inputs=[
        InDataset(),
        InInt("ref_channel", "Ref channel", unit="", field=False, default=0),
        InInt("upsample", "Upsample", unit="", field=False, default=20),
        InFloat("highpass_sigma", "Highpass σ", unit="px", field=True, default=2.0),
        InFloat("min_confidence", "Min confidence", unit="", field=True, default=0.0),
    ],
    outputs=[OutDataset()],
    modes=[Mode("model", ["translation", "euclidean", "affine", "feature"],
                default="translation"),
           Mode("reference", ["first", "previous", "mean", "template"],
                default="previous")],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"t", "y", "x"}),
    description="Register-once (on a reference channel) / apply-to-all-channels "
                "stabilization; translation/euclidean/affine/feature models, "
                "first/previous/mean/template anchor; stores per-frame shift (ported "
                "v1 kernel).")


# ── Boundary bands (Voxel label raster → outward band raster, 3D) ──────────────

def _compute_boundary_band(ctx: EvalContext) -> Dataset:
    """Outward boundary band per label region (general — any Voxel label raster: cells,
    nuclei, granules, …; ported v1 ``granule_boundary`` kernel): for each labelled region,
    the voxels within ``band_voxels`` (or ``band_um``) of its surface whose label is NOT
    that region. Reads a **Voxel label raster** (from CCL / watershed / a mask-volume node)
    and writes a Voxel **band-label** raster — each band voxel painted with its source
    region id. 3D per ``(m,t,c)`` volume; ``method`` = dilation (6-connectivity iterations)
    or EDT (µm threshold).

    voxel_size_um=(z_step_um, pixel_size_um, pixel_size_um) drives the EDT sampling.
    Kernel: :func:`nodegraph.kernels.granule_boundary.extract_boundary_bands`."""
    from nodegraph.kernels.granule_boundary import extract_boundary_bands
    ds = ctx.inputs[0]
    ax = ds.axes
    src = ctx.layer("labels")
    lab_attr = ds.get(Domain.VOXEL, src)
    if lab_attr is None:
        raise ValueError(f"boundary band needs a Voxel label layer {src!r} "
                         f"(run a label / watershed / mask-volume node first)")
    labels6 = np.asarray(lab_attr.values)
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox = (zs, px, px)
    kp = {
        "band_voxels": max(0, int(ctx.params.get("band_voxels", 1))),
        "band_method": ctx.params.get("__modes__", {}).get("method", "dilation"),
        "band_um": float(ctx.params.get("band_um", 0.0)),
        "include_neighbors": bool(ctx.params.get("include_neighbors", True)),
    }
    out = np.zeros_like(labels6)
    for m in range(ax.m):
        for t in range(ax.t):
            for c in range(ax.c):
                vol = labels6[m, t, :, c]                      # (Z, H, W)
                masks = {int(g): (vol == g) for g in np.unique(vol) if g != 0}
                if not masks:
                    continue
                _bands, combined = extract_boundary_bands(masks, vol, vox, kp)
                out[m, t, :, c] = combined
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), out)


register_node(
    _compute_boundary_band, op_key="analysis.boundary_band",
    label="Boundary Band", category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            InString("labels", "Label layer", field=False, default="labels",
                     layer_in=Domain.VOXEL),
            # band_voxels is live in BOTH methods — dilation iterations, and the `edt`
            # fallback threshold (N·finest-voxel) when band_um is 0. band_um is read
            # only by the EDT criterion (kernel: `edt <= edt_threshold`), so it stays
            # hidden under dilation, where it has no effect at all.
            InInt("band_voxels", "Band voxels", unit="", field=False, default=1),
            InFloat("band_um", "Band width", unit="um", field=True, default=0.0,
                    available_in={"method": frozenset({"edt"})}),
            InBool("include_neighbors", "Include neighbors", field=False, default=True),
            InString("name", "Output layer", field=False, default="bands",
                     layer_out=(Domain.VOXEL,))],
    outputs=[OutDataset()], modes=[Mode("method", ["dilation", "edt"], default="dilation")],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset({"z", "y", "x"}),
    description="Outward boundary band per label region (any Voxel label raster — cells, "
                "nuclei, granules; dilation or EDT, µm-aware) → band-label raster, 3D "
                "(ported v1 kernel).")


# ── Cluster a point cloud → a per-point cluster-id label column ────────────────
#
# Ports the vendored `granule_cluster` kernel (scikit-learn GaussianMixture / KMeans with a
# BIC model-order sweep). GENERAL: clusters ANY 3-D point cloud into groups (beads→granules
# is one use). Adds a per-point integer label column to the Point layer (per (m,t,c) group;
# ids 0..k-1 within a group — the natural key for the downstream `analysis.tessellate`, which
# also groups by (m,t,c)). The granule chain's front half: detect.particles → cluster_points
# → tessellate → rasterize_mesh. Requires scikit-learn (installed 2026-07-26).


def _compute_cluster_points(ctx: EvalContext) -> Dataset:
    """Cluster a Point cloud into groups → a per-point cluster-id label column (ported v1
    ``granule_cluster`` kernel). Fits a full-covariance GaussianMixture (``gmm``, keeps
    anisotropic clusters whole) or spherical ``kmeans``, selecting the model order ``k`` by a
    BIC sweep over ``n_clusters ± relax_pct%`` (both BICs directly comparable). Points are
    scaled to µm (``voxel_size_um``) before fitting so anisotropic Z doesn't bias the model.
    Clusters each ``(m,t,c)`` group independently (ids ``0..k-1`` per group).

    Resolved spec: category analysis; op ``analysis.cluster_points``; reads POINT, adds POINT
    (a label column on the ``source`` layer, default ``cluster``). ``voxel_size_um=(z_step_um,
    pixel_size_um, pixel_size_um)`` via ``ctx.calib`` (memo-fenced). Footprint WHOLE_VOLUME
    (clusters a whole 3-D cloud; no image access → ``kernel_axes`` empty). Kernel:
    :func:`nodegraph.kernels.granule_cluster.cluster_granules` (scikit-learn)."""
    from nodegraph.kernels.granule_cluster import cluster_granules
    ds = ctx.inputs[0]
    source = ctx.layer("source")
    name = ctx.params.get("name", "cluster")
    method = ctx.params.get("__modes__", {}).get("method", "gmm")
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox = (zs, px, px)
    cp = {"n_granules": max(1, int(ctx.params.get("n_clusters", 1))),
          "relax_pct": max(0.0, float(ctx.params.get("relax_pct", 0.0))),
          "method": method,
          "n_init": max(1, int(ctx.params.get("n_init", 1)))}
    pts = [a for a in ds.layers_on(Domain.POINT) if a.layer == source]
    if not pts:
        raise ValueError(f"cluster points needs a Point layer {source!r} "
                         "(run detect.particles first)")
    col = {a.name: np.asarray(a.values) for a in pts}
    for req in ("m", "t", "c", "z", "y", "x"):
        if req not in col:
            raise ValueError(f"Point layer {source!r} missing coordinate column {req!r}")
    m_all = col["m"].astype(int); t_all = col["t"].astype(int); c_all = col["c"].astype(int)
    z_all = col["z"].astype(float); y_all = col["y"].astype(float); x_all = col["x"].astype(float)
    out = np.zeros(len(z_all), dtype=np.int64)
    for m in sorted(set(m_all.tolist())):
        for t in sorted(set(t_all[m_all == m].tolist())):
            for c in sorted(set(c_all[(m_all == m) & (t_all == t)].tolist())):
                sel = (m_all == m) & (t_all == t) & (c_all == c)
                if not sel.any():
                    continue
                pzyx = np.column_stack([z_all[sel], y_all[sel], x_all[sel]])
                labels, _info = cluster_granules(pzyx, vox, cp)
                out[sel] = np.asarray(labels, dtype=np.int64)
    # Add the label column to the SAME Point layer (with_layer, not with_structure — leaves
    # the source's z_kind provenance untouched, no clobber).
    return ds.with_layer(Domain.POINT, name, out, layer=source)


register_node(
    _compute_cluster_points, op_key="analysis.cluster_points", label="Cluster Points",
    category="analysis",
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[InDataset(),
            InString("source", "Point layer", field=False, default="particles",
                     layer_in=Domain.POINT),
            InString("name", "Label column", field=False, default="cluster"),
            InInt("n_clusters", "Cluster count", unit="", field=True, default=1),
            InFloat("relax_pct", "Relax %", unit="", field=True, default=0.0),
            InInt("n_init", "Restarts", unit="", field=False, default=1)],
    outputs=[OutDataset()],
    modes=[Mode("method", ["gmm", "kmeans"], default="gmm", label="Model")],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset(),
    description="Cluster a Point cloud into groups (GaussianMixture / KMeans, BIC model-order "
                "sweep over n_clusters ± relax%) → a per-point cluster-id label column; µm-"
                "scaled so anisotropic Z doesn't bias the fit (ported v1 kernel; scikit-learn).")


# ── Tessellate points / labels → a MESH (3D) ───────────────────────────────────
#
# The tessellation stage on its own (V2.08; ports granule_tessellate). v1 always had this
# as its own node — `special:granule_tessellate` → `special:granule_mask` — and v2 fused
# the two because there was no domain able to carry a mesh across a wire. With
# ``Domain.MESH`` (nodegraph/mesh.py) the v1 split is restored: this node produces the
# surface, ``transform.rasterize_mesh`` consumes it.
#
# GENERAL node: it tessellates ANY labeled 3-D point cloud, or the surface of ANY Voxel
# label raster — the granule density-MERGE stays off (``merge_tol=0``) so input labels map
# one-to-one. Downstream: rasterize_mesh → boundary_band / measure.


def _tess_points(ctx: EvalContext, ds: Dataset, ax: AxisSizes, mode: str,
                 vox: Tuple[float, float, float]) -> list:
    """POINTS path — one mesh element per label of a Point layer's cluster-id column.

    The vendored kernel takes VOXEL ``(z,y,x)`` points and returns world ``(x,y,z)`` µm
    vertices, so the only conversion here is on the way out."""
    from nodegraph.kernels.granule_tessellate import tessellate_granules
    from nodegraph.kernels.mesh_raster import verts_um_to_zyx
    from nodegraph.mesh import MeshElement, surface_area_um2

    source = ctx.layer("points")
    labels_col = ctx.params.get("cluster", "cluster")
    if mode == "voronoi":
        tp: Dict[str, Any] = {"tess_mode": "voronoi"}
    elif mode == "alpha_shape":
        a_um = float(ctx.params.get("alpha_um", 0.0))
        tp = {"tess_mode": "alpha_shape", "alpha": (a_um if a_um > 0 else float("inf"))}
    else:                                                      # convex_hull (default)
        tp = {"tess_mode": "alpha_shape", "alpha": float("inf")}
    # MERGE OFF — general node: one element per input label, no density folding.
    tp["merge_tol"] = 0.0
    tp["min_granule_points"] = max(1, int(ctx.params.get("min_points", 4)))

    pts = [a for a in ds.layers_on(Domain.POINT) if a.layer == source]
    if not pts:
        raise ValueError(f"tessellate needs a Point layer {source!r} "
                         "(run detect.particles → cluster_points first)")
    col = {a.name: np.asarray(a.values) for a in pts}
    for req in ("m", "t", "c", "z", "y", "x"):
        if req not in col:
            raise ValueError(f"Point layer {source!r} missing coordinate column {req!r}")
    if labels_col not in col:
        raise ValueError(f"Point layer {source!r} missing label column {labels_col!r} "
                         "(run cluster_points, or provide a per-point label column)")
    m_all = col["m"].astype(int); t_all = col["t"].astype(int); c_all = col["c"].astype(int)
    z_all = col["z"].astype(float); y_all = col["y"].astype(float); x_all = col["x"].astype(float)
    lab_all = col[labels_col].astype(int)

    elements: list = []
    for m in sorted(set(m_all.tolist())):
        for t in sorted(set(t_all[m_all == m].tolist())):
            for c in sorted(set(c_all[(m_all == m) & (t_all == t)].tolist())):
                sel = (m_all == m) & (t_all == t) & (c_all == c)
                if not sel.any():
                    continue
                pzyx = np.column_stack([z_all[sel], y_all[sel], x_all[sel]])
                plab = lab_all[sel]
                tess = tessellate_granules(pzyx, plab, vox, tp)
                for oid, bnd in tess.boundaries.items():
                    verts = verts_um_to_zyx(bnd.vertices_um, vox)
                    mem = plab == oid
                    elements.append(MeshElement(
                        m=m, t=t, c=c, src_label=int(oid),
                        verts_zyx=verts, faces=bnd.faces,
                        centroid_zyx=(float(np.mean(z_all[sel][mem])),
                                      float(np.mean(y_all[sel][mem])),
                                      float(np.mean(x_all[sel][mem]))),
                        # the kernel's ANALYTIC volume (alpha-filtered where relevant) —
                        # no voxel-quantization loss, unlike a rasterized count
                        volume_um3=float(bnd.enclosed_volume_um3),
                        surface_area_um2=surface_area_um2(verts, bnd.faces, vox),
                        density=float(bnd.density), n_points=int(bnd.n_points)))
    return elements


def _tess_label_surface(ctx: EvalContext, ds: Dataset, ax: AxisSizes,
                        vox: Tuple[float, float, float]) -> list:
    """LABELS path — one mesh element per region of a Voxel label raster, via marching
    cubes. ``marching_cubes`` already returns voxel ``(z,y,x)`` vertices, so this path
    needs no coordinate conversion at all."""
    from skimage.measure import marching_cubes
    from nodegraph.mesh import MeshElement, enclosed_volume_um3, surface_area_um2

    src = ctx.layer("labels")
    level = float(ctx.params.get("iso_level", 0.5))
    step = max(1, int(ctx.params.get("decimate", 1)))
    lay = ds.get(Domain.VOXEL, src)
    if lay is None:
        raise ValueError(f"tessellate (label_surface) needs a Voxel layer {src!r} "
                         "(run analysis.label / analysis.threshold first)")
    raster = np.asarray(lay.values)
    elements: list = []
    for m in range(ax.m):
        for t in range(ax.t):
            for c in range(ax.c):
                vol = raster[m, t, :, c]
                ids = np.unique(vol)
                for g in ids[ids != 0].tolist():
                    binary = (vol == g)
                    if binary.sum() < 8:              # too thin for a closed surface
                        continue
                    try:
                        v, f, _n, _v = marching_cubes(
                            binary.astype(float), level=level, step_size=step)
                    except (RuntimeError, ValueError):
                        continue                      # degenerate region — skip, not fail
                    zc, yc, xc = (float(a.mean()) for a in np.nonzero(binary))
                    elements.append(MeshElement(
                        m=m, t=t, c=c, src_label=int(g), verts_zyx=v, faces=f,
                        centroid_zyx=(zc, yc, xc),
                        volume_um3=enclosed_volume_um3(v, f, vox),
                        surface_area_um2=surface_area_um2(v, f, vox),
                        density=0.0, n_points=int(binary.sum())))
    return elements


def _compute_tessellate(ctx: EvalContext) -> Dataset:
    """Tessellate a labeled Point cloud **or** a Voxel label raster into a **MESH** —
    one closed boundary surface per input label (ports v1 ``granule_tessellate``).

    Resolved spec (V2.08): category analysis; op ``analysis.tessellate``; adds MESH;
    **3D-only** (a surface is volumetric — ``WHOLE_VOLUME``, ``kernel_axes={z,y,x}``,
    ``z<2`` is a hard error). One four-choice ``boundary`` lever selects both the source
    kind and the algorithm: ``convex_hull`` (alpha=∞) / ``alpha_shape`` (finite
    ``alpha_um``, concave) / ``voronoi`` read the ``points`` layer's ``cluster`` column;
    ``label_surface`` marching-cubes the ``labels`` Voxel raster. It is ONE mode rather
    than a ``source`` × ``boundary`` pair because ``available_in`` gates sockets on modes
    but nothing gates a *mode* on another mode — a separate source lever would leave three
    selectable-but-meaningless combinations.

    ``reads_domains`` is empty on purpose (the ``track.objects`` / ``boundary_band``
    precedent): the required domain is POINT in three modes and VOXEL in the fourth, and
    ``reads_domains`` has no per-mode form, so declaring either would light a false red
    chip in the other. The compute raises with the exact layer to wire instead.

    Vertices are stored in **voxel ``(z,y,x)``** (the Mesh domain convention — see
    :mod:`nodegraph.mesh`); ``voxel_size_um=(z_step_um, pixel_size_um, pixel_size_um)`` is
    read via ``ctx.calib`` (memo-fenced) for the µm geometry columns. The boundary mode +
    source are **stamped as mesh provenance** so ``transform.rasterize_mesh`` derives its
    interior test instead of exposing a lever that could disagree with the data
    (`wire-node-v2` §7b)."""
    from nodegraph.mesh import build_mesh_tables, with_mesh

    ds = ctx.inputs[0]
    ax = ds.axes
    if ax.z < 2:
        raise ValueError("tessellate needs a 3D volume (z>1); a boundary surface is "
                         "volumetric (use a Z-stack)")
    mode = ctx.params.get("__modes__", {}).get("boundary", "convex_hull")
    name = ctx.layer("name")
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox = (zs, px, px)

    if mode == "label_surface":
        elements = _tess_label_surface(ctx, ds, ax, vox)
        src_kind, src_layer = "labels", ctx.layer("labels")
    else:
        elements = _tess_points(ctx, ds, ax, mode, vox)
        src_kind, src_layer = "points", ctx.layer("points")

    tables = build_mesh_tables(elements, layer=name, z_kind="subpixel")
    return with_mesh(ds, tables, provenance={
        "boundary": mode, "source": src_kind, "src_layer": str(src_layer),
        "voxel_size_um": [float(v) for v in vox]})


register_node(
    _compute_tessellate, op_key="analysis.tessellate", label="Tessellate",
    category="analysis",
    # deliberately empty — see the compute docstring (POINT in three modes, VOXEL in the
    # fourth, and reads_domains has no per-mode form)
    reads_domains=frozenset(),
    adds_domains=frozenset({Domain.MESH}),
    inputs=[InDataset(),
            InString("points", "Point layer", field=False, default="particles",
                     layer_in=Domain.POINT,
                     available_in={"boundary": frozenset(
                         {"convex_hull", "alpha_shape", "voronoi"})}),
            InString("cluster", "Label column", field=False, default="cluster",
                     available_in={"boundary": frozenset(
                         {"convex_hull", "alpha_shape", "voronoi"})}),
            # alpha is the ALPHA-SHAPE radius only: convex_hull pins alpha=inf, voronoi
            # takes no radius, label_surface does not use the kernel at all.
            InFloat("alpha_um", "Alpha (concave)", unit="um", field=True, default=0.0,
                    available_in={"boundary": frozenset({"alpha_shape"})}),
            InInt("min_points", "Min points/element", unit="", field=False, default=4,
                  available_in={"boundary": frozenset(
                      {"convex_hull", "alpha_shape", "voronoi"})}),
            InString("labels", "Voxel label layer", field=False, default="labels",
                     layer_in=Domain.VOXEL,
                     available_in={"boundary": frozenset({"label_surface"})}),
            InFloat("iso_level", "Iso level", unit="", field=True, default=0.5,
                    available_in={"boundary": frozenset({"label_surface"})}),
            InInt("decimate", "Vertex step", unit="", field=False, default=1,
                  available_in={"boundary": frozenset({"label_surface"})}),
            InString("name", "Output mesh", field=False, default="mesh",
                     layer_out=(Domain.MESH,))],
    outputs=[OutDataset()],
    modes=[Mode("boundary", ["convex_hull", "alpha_shape", "voronoi", "label_surface"],
                default="convex_hull", label="Boundary")],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset({"z", "y", "x"}),
    description="Tessellate a labeled Point cloud (convex/concave/Voronoi boundary) or a "
                "Voxel label raster (marching cubes) → a MESH: one closed surface per "
                "label, with analytic volume/area/density per element; 3D (ports v1 "
                "granule_tessellate). Feed transform.rasterize_mesh to get a Label volume.")


# ── Rasterize a MESH → a Voxel Label volume + geometry table (3D) ──────────────
#
# The downstream half of the v1 split (ports granule_volume_mask). Each mesh element is
# voxelized and the regions are painted into one combined Voxel **Label raster** (higher
# density wins on overlap), with a per-element **Label table** carrying the mesh's ANALYTIC
# geometry alongside the realized voxel_count.


def _compute_rasterize_mesh(ctx: EvalContext) -> Dataset:
    """Rasterize a **MESH** into a Voxel **Label** volume + a per-element geometry table
    (ports v1 ``granule_volume_mask``).

    Resolved spec (V2.08): category transform (beside ``transform.rasterize_field``); op
    ``transform.rasterize_mesh``; reads MESH + VOXEL (the image defines the target grid),
    adds VOXEL + LABEL — the same output contract the fused node had, so ``boundary_band``
    / ``measure`` chains are unchanged. ``WHOLE_VOLUME``, ``kernel_axes={z,y,x}``.

    **The interior test is INHERITED, not a lever** (`wire-node-v2` §7b, as
    ``rasterize_field`` inherits its dimensionality): ``convex_hull`` provenance keeps the
    cheap ``Delaunay.find_simplex`` path, while a possibly-concave mesh (``alpha_shape``
    with a finite alpha, ``label_surface``) uses the faces-based even-odd parity test — so
    the voxel mask finally agrees with the ``volume_um3`` / ``surface_area_um2`` the
    tessellation reports. A mesh with no provenance falls back to its own derived
    ``closed`` topology, so a hand-built mesh still resolves. This fixes a real latent
    bug: the alpha-shape producer stores the *unfiltered* Delaunay, whose simplices union
    to the convex hull, so every mode used to rasterize convex.

    ids are painted **verbatim** from the mesh element's ``id`` — no renumbering — so an
    element that voxelizes empty leaves a gap rather than shifting every id after it."""
    from nodegraph.kernels.mesh_raster import interior_test_for, voxelize_mesh
    from nodegraph.mesh import mesh_element, mesh_provenance, read_mesh

    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("rasterize mesh needs an image provider (defines the voxel grid)")
    ax = prov.axes
    if ax.z < 2:
        raise ValueError("rasterize mesh needs a 3D volume (z>1)")
    layer = ctx.layer("mesh")
    name = ctx.layer("name")
    smooth = max(0.0, float(ctx.params.get("smooth_um", 0.0)))
    fill = bool(ctx.params.get("fill_holes", False))
    min_vox = max(1, int(ctx.params.get("min_voxels", 1)))
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox = (zs, px, px)

    tables = read_mesh(ds, layer)
    prv = mesh_provenance(ds, layer)
    el = tables.element.columns
    e_id = np.asarray(el["id"], dtype=np.int64)
    e_m = np.asarray(el["m"], dtype=np.int64)
    e_t = np.asarray(el["t"], dtype=np.int64)
    e_c = np.asarray(el["c"], dtype=np.int64)
    e_dens = np.asarray(el["density"], dtype=float)
    e_closed = np.asarray(el["closed"], dtype=np.int64)

    raster = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.int64)
    rows: list = []
    for key in sorted({(int(a), int(b), int(d)) for a, b, d in zip(e_m, e_t, e_c)}):
        m, t, c = key
        sel = np.flatnonzero((e_m == m) & (e_t == t) & (e_c == c))
        painted: list = []                             # (density, -id, id, mask)
        for row in sel.tolist():
            verts, faces = mesh_element(tables, row)
            test = interior_test_for(prv, bool(e_closed[row]))
            mask = voxelize_mesh(verts, faces, (ax.z, ax.y, ax.x), vox, mode=test,
                                 density=float(e_dens[row]), smooth_um=smooth,
                                 fill_holes=fill, min_voxels=min_vox)
            n_vox = int(mask.sum())
            if n_vox == 0:
                continue                               # did not rasterize — id stays absent
            painted.append((float(e_dens[row]), -int(e_id[row]), int(e_id[row]), mask))
            rows.append({
                "id": int(e_id[row]), "m": m, "t": t, "c": c,
                "z": float(np.asarray(el["z"])[row]),
                "y": float(np.asarray(el["y"])[row]),
                "x": float(np.asarray(el["x"])[row]),
                "volume_um3": float(np.asarray(el["volume_um3"])[row]),
                "density": float(e_dens[row]),
                "n_points": int(np.asarray(el["n_points"])[row]),
                "surface_area_um2": float(np.asarray(el["surface_area_um2"])[row]),
                "voxel_count": n_vox,
            })
        combined = np.zeros((ax.z, ax.y, ax.x), dtype=np.int64)
        for _d, _negid, lid, mask in sorted(painted, key=lambda p: (p[0], p[1])):
            combined[mask] = lid                       # higher density wins (last write)
        raster[m, t, :, c] = combined

    res = ds.with_layer(Domain.VOXEL, name, raster)
    if not rows:
        return res
    merged = {k: np.array([r[k] for r in rows],
                          dtype=(np.int64 if k in ("id", "m", "t", "c", "n_points",
                                                   "voxel_count") else float))
              for k in rows[0]}
    return res.with_structure(StructureTable(Domain.LABEL, merged, layer=name,
                                             z_kind="subpixel"))


register_node(
    _compute_rasterize_mesh, op_key="transform.rasterize_mesh", label="Rasterize Mesh",
    category="transform",
    reads_domains=frozenset({Domain.MESH, Domain.VOXEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    inputs=[InDataset(),
            InString("mesh", "Mesh layer", field=False, default="mesh",
                     layer_in=Domain.MESH),
            InString("name", "Output layer", field=False, default="labels",
                     layer_out=(Domain.VOXEL, Domain.LABEL)),
            InFloat("smooth_um", "Smoothing", unit="um", field=True, default=0.0),
            InInt("min_voxels", "Min voxels/region", unit="", field=False, default=1),
            InBool("fill_holes", "Fill holes", field=False, default=False)],
    outputs=[OutDataset()],
    # No interior-test lever — it is derived from the mesh's stamped provenance (§7b).
    modes=[],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset({"z", "y", "x"}),
    description="Rasterize a MESH → one filled Voxel Label region per element (interior "
                "test inherited from the mesh provenance: convex hull, or faces-based "
                "parity so concavity survives) + a per-element Label table (centroid, "
                "analytic volume_um3/density/n_points/surface_area, realized voxel_count); "
                "3D (ports v1 granule_volume_mask).")


# ── ROI mask (serializable shape list → boolean Voxel mask, 2D) ────────────────

def _roi_shapes(raw):
    """The ``shapes`` param → a list of shape dicts, or ``None`` for a whole-frame ROI.

    Two producers, so two accepted forms: a **list** handed in programmatically (a
    headless caller or a future ROI-draw tool) and the socket's **JSON text**, since a
    shape list has no matching :class:`SocketType`. Malformed JSON raises with the parse
    error rather than silently degrading to a whole-frame ROI, which would look like the
    node ran fine and quietly analyse the entire image."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if not isinstance(raw, str):
        return raw
    import json as _json
    try:
        val = _json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            f"ROI mask: `shapes` is not valid JSON ({exc}). Expected a list of shape "
            'objects, e.g. [{"type": "rect", "op": "add", "vertices": [[5,5],[15,15]]}]'
        ) from exc
    if val in (None, [], {}):
        return None
    if not isinstance(val, list):
        raise ValueError(f"ROI mask: `shapes` must be a JSON list, got {type(val).__name__}")
    return val


def _compute_roi_mask(ctx: EvalContext) -> Dataset:
    """Rasterize an ordered list of vector shapes (rect/ellipse/circle/polygon/brush +
    invert/clear) into a boolean Voxel ROI mask — a general drawn-region mask (crop/
    exclude/analysis ROI; ported v1 ``dic_mesh_region`` kernel). Pure pixel geometry (no
    calibration): the ``shapes`` param carries the drawn ROI ([y,x] vertices, in-order
    stateful replay); an empty/absent list → a whole-frame ROI. Frame-independent (broadcast
    to all m/t/z/c). Kernel: :func:`nodegraph.kernels.dic_mesh_region.build_roi_mask`."""
    from nodegraph.kernels.dic_mesh_region import build_roi_mask, has_region
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("ROI mask needs an image provider on its input Dataset")
    ax = prov.axes
    shapes = _roi_shapes(ctx.params.get("shapes"))    # list of shape dicts, or None
    if has_region(shapes):
        m2d = np.asarray(build_roi_mask(shapes, ax.y, ax.x), dtype=np.int64)
    else:
        m2d = np.ones((ax.y, ax.x), dtype=np.int64)   # no shapes → whole-frame ROI
    mask6 = np.broadcast_to(m2d, (ax.m, ax.t, ax.z, ax.c, ax.y, ax.x)).copy()
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), mask6)


register_node(
    _compute_roi_mask, op_key="analysis.roi_mask", label="ROI Mask",
    category="analysis",
    # reads only the image EXTENT (pure pixel geometry), so nothing is required of the
    # input beyond a provider; it adds the ROI raster.
    adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            # The shape list is structured data with no matching SocketType, and v2 has
            # no ROI-drawing surface yet — but with NO socket at all the node was fixed
            # at a whole-frame ROI and therefore inert. A JSON text socket makes it
            # usable now and is what a future draw tool would write into.
            InString("shapes", "Shapes (JSON)", field=False, default=""),
            InString("name", "Output layer", field=False, default="roi_mask",
                     layer_out=(Domain.VOXEL,))],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_PLANE, kernel_axes=frozenset({"y", "x"}),
    description="Rasterize a serializable shape list ([y,x] verts, in-order add/cut/"
                "invert/clear) → a boolean Voxel ROI mask; empty → whole frame "
                "(ported v1 kernel).")


# ── DVC / ALDVC displacement + strain field (ref/def volume pair → Point field) ─
#
# Ports the vendored `aldvc_field` kernel (FranckLab ALDVC, in-repo numpy/scipy port).
# The kernel correlates ONE ref/def volume (or plane) pair → a dense displacement +
# strain field on a COARSE subset grid (spacing = `subset_spacing`), not the full voxel
# grid. Design (V2.06, user-locked 2026-07-24): the node owns the m/t loop and the
# ref/def pairing via a `reference_mode` lever {fixed_frame | previous_frame} + an
# OPTIONAL `reference` Dataset input — leave it unwired to self-reference the same
# series (fixed frame or previous frame per the lever), or wire a second loaded file to
# use THAT as the reference. Output is a faithful POINT structure: the subset centers
# (grid_coords z,y,x in voxels) carrying disp_z/disp_y/disp_x (µm), a disp magnitude,
# the strain-tensor components (dimensionless), and the per-subset qfactor. A separate
# `transform.rasterize_field` node interpolates the Point field up to Voxel layers.


def _dvc_rows(res, vox, *, m: int, t: int, c: int, z_plane: Optional[int]) -> Dict[str, np.ndarray]:
    """Flatten one :class:`DVCResult` (a coarse subset grid) into a column-dict of Point
    rows. ``vox`` = the ``voxel_size_um`` passed to the kernel (component order matches
    the displacement/grid axes, slowest-first) so displacement voxels → µm is a
    component-wise multiply. 2D stamps ``z = z_plane`` (plane index); 3D reads z from the
    grid. Strain ``[i,j] = ∂u_i/∂x_j`` becomes ``strain_<axis_i><axis_j>`` columns."""
    d = int(res.dim)
    grid = np.asarray(res.grid_coords, dtype=float).reshape(-1, d)          # voxel coords
    disp = np.asarray(res.displacement_field, dtype=float).reshape(-1, d)   # voxels
    disp_um = disp * np.asarray(vox, dtype=float)                            # → µm
    n = grid.shape[0]
    anames = ("z", "y", "x") if d == 3 else ("y", "x")
    cols: Dict[str, np.ndarray] = {}
    if d == 3:
        z, y, x = grid[:, 0], grid[:, 1], grid[:, 2]
        cols["disp_z"], cols["disp_y"], cols["disp_x"] = (disp_um[:, 0], disp_um[:, 1],
                                                          disp_um[:, 2])
    else:
        y, x = grid[:, 0], grid[:, 1]
        z = np.full(n, float(z_plane if z_plane is not None else 0))
        cols["disp_y"], cols["disp_x"] = disp_um[:, 0], disp_um[:, 1]
    cols["disp_mag_um"] = np.sqrt((disp_um ** 2).sum(axis=1))
    if res.strain_field is not None:
        s = np.asarray(res.strain_field, dtype=float).reshape(-1, d, d)
        for i in range(d):
            for j in range(d):
                cols[f"strain_{anames[i]}{anames[j]}"] = s[:, i, j]
    if res.qfactor is not None:
        cols["qfactor"] = np.asarray(res.qfactor, dtype=float).reshape(-1)
    row = {
        "m": np.full(n, m, dtype=np.int64), "t": np.full(n, t, dtype=np.int64),
        "c": np.full(n, c, dtype=np.int64),
        "z": z.astype(float), "y": y.astype(float), "x": x.astype(float),
    }
    row.update(cols)
    return row


def _compute_dvc_field(ctx: EvalContext) -> Dataset:
    """DVC / ALDVC displacement + strain field (ported v1 ``aldvc_field`` kernel) → a
    **Point** structure on the correlation subset grid. Loops m/t on a single reference
    channel; each ref/def pair (2D plane or 3D volume) runs the Augmented-Lagrangian
    IC-GN+ADMM solver. Displacements are stored in **µm** (voxels × ``voxel_size_um``);
    strain is dimensionless; ``qfactor`` is the per-subset ZNCC confidence.

    Resolved spec (V2.06): category analysis; op ``analysis.dvc_field``; **Point** output;
    DimMode lever (2D per-plane / 3D per-volume). ``reference_mode`` ∈ fixed_frame (ref =
    ``reference_frame`` of the same series) / previous_frame (ref = t−1, t=0 skipped). An
    optional ``reference`` Dataset input overrides self-reference — ref = the external
    dataset at ``reference_frame`` (clamped m/z/c), so a second file (a separate undeformed
    stack) becomes the reference. Footprint WHOLE_SERIES (crosses T), ``kernel_axes`` per
    dim (adds ``t``). ``voxel_size_um=(z_step_um, pixel_size_um, pixel_size_um)`` slowest-
    first (2-tuple in 2D); it drives the displacement→µm scale + anisotropic strain rescale.
    subset/spacing/search are voxel extents (unit ``px``, no optical derive — set by the
    speckle pattern). ``n_workers`` is forced to 1 (embedding-safe: no spawn/import hazard,
    kernel gotcha 9). Kernel: :func:`nodegraph.kernels.aldvc_field.run_aldvc`."""
    from nodegraph.kernels.aldvc_field import run_aldvc
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("DVC needs an image provider on its (deformed) input Dataset")
    ax = prov.axes
    ref_ds = ctx.input("reference")
    ref_prov = ref_ds.image if ref_ds is not None else None
    if ref_ds is not None and ref_prov is None:
        raise ValueError("the DVC 'reference' input Dataset has no image provider")
    modes = ctx.params.get("__modes__", {})
    is_3d = modes.get("dim") == "3D"
    if is_3d and ax.z < 2:
        raise ValueError("3D DVC needs z>1; use 2D mode for a single-plane series")
    ref_mode = modes.get("reference_mode", "fixed_frame")
    ref_frame = int(ctx.params.get("reference_frame", 0))
    c = min(max(0, int(ctx.params.get("channel", 0))), max(0, ax.c - 1))
    px = ctx.calib("pixel_size_um") or 0.1
    if is_3d:
        zs = ctx.calib("z_step_um") or 0.5
        vox: Tuple[float, ...] = (zs, px, px)
    else:
        vox = (px, px)
    dvc_params = {
        "subset_size": max(4, int(ctx.params.get("subset_size", 16))),
        "subset_spacing": max(1, int(ctx.params.get("subset_spacing", 10))),
        "search_radius": max(0, int(ctx.params.get("search_radius", 0))),
        "seed_levels": max(1, int(ctx.params.get("seed_levels", 3))),
        "correlation": modes.get("correlation", "zncc"),
        "mu": float(ctx.params.get("mu", 1e-3)),
        "admm_iterations": max(0, int(ctx.params.get("admm_iterations", 4))),
        "strain_type": modes.get("strain_type", "infinitesimal"),
        "strain_smooth": max(0.0, float(ctx.params.get("strain_smooth", 0.0))),
        "cc_thresh": float(ctx.params.get("cc_thresh", 0.5)),
        "n_workers": 1,          # embedding-safe (kernel gotcha 9: spawn/import hazard)
        "use_gpu": False,
    }
    layer = ctx.layer("name")
    # Cross-frame warm-start (ALDVC series workflow): seed each frame's IC-GN from the
    # previous frame's field (u0_seed, skipping the FFT search) — faithful to FranckLab's
    # series correlation + more robust for large motion. `newFFTSearch` forces a fresh
    # FFT seed every frame; the first computed frame of a series always FFT-seeds. The
    # warm-start applies in BOTH reference modes (v1 parity, pipelines_page._run_series_all_m).
    newfft = bool(ctx.params.get("newFFTSearch", False))

    def _resolve_ref(m: int, t: int):
        """(ref_provider, ref_m, ref_t) for this def frame, or None to skip it."""
        if ref_prov is not None:                              # external reference file
            rax = ref_prov.axes
            return (ref_prov, min(m, rax.m - 1),
                    min(max(0, ref_frame), rax.t - 1))
        if ref_mode == "previous_frame":
            return None if t == 0 else (prov, m, t - 1)        # no increment into t0
        return prov, m, min(max(0, ref_frame), ax.t - 1)       # fixed_frame

    def _match(ref_arr: np.ndarray, def_arr: np.ndarray) -> None:
        if ref_arr.shape != def_arr.shape:
            raise ValueError(
                f"DVC reference shape {ref_arr.shape} != deformed {def_arr.shape}; the "
                "reference Dataset must match the primary's Y/X (and Z in 3D)")

    def _seeded(ref_arr: np.ndarray, def_arr: np.ndarray, u0):
        """Run the kernel warm-started from ``u0`` (previous frame's ``(ndim,*grid)``
        field) unless ``u0`` is None or ``newFFTSearch`` — then FFT-seed. Returns
        ``(res, next_u0)`` where ``next_u0`` seeds the following frame."""
        warm = u0 is not None and not newfft
        res = run_aldvc(ref_arr, def_arr, voxel_size_um=vox, params=dvc_params,
                        u0_seed=(u0 if warm else None), use_fft_seed=not warm)
        return res, np.moveaxis(np.asarray(res.displacement_field), -1, 0)

    rows: list = []
    # per-node progress: correlation is the dominant cost in any graph that has it and it
    # is EAGER (one solver call per unit), so the unit count is a real denominator.
    n_units = ax.m * ax.t * (1 if is_3d else ax.z)
    done_units = 0
    ctx.progress(0, n_units, "correlating")
    for m in range(ax.m):
        prev_u: Dict[int, Any] = {}      # z-plane (or -1 for the whole 3D volume) → u0
        for t in range(ax.t):
            resolved = _resolve_ref(m, t)
            if resolved is None:
                done_units += 1 if is_3d else ax.z      # skipped t0 still advances the bar
                ctx.progress(done_units, n_units, "correlating")
                continue
            rprov, rm, rt = resolved
            rax = rprov.axes
            rc = min(c, max(0, rax.c - 1))
            if is_3d:
                dvol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y,
                                              0, ax.x).astype(float)
                rvol = rprov.get_region_volume(0, rm, rt, rc, 0, rax.z, 0, rax.y,
                                               0, rax.x).astype(float)
                _match(rvol, dvol)
                res, prev_u[-1] = _seeded(rvol, dvol, prev_u.get(-1))
                rows.append(_dvc_rows(res, vox, m=m, t=t, c=c, z_plane=None))
                done_units += 1
                ctx.progress(done_units, n_units, f"volume t={t}")
            else:
                for z in range(ax.z):
                    rz = min(z, max(0, rax.z - 1))
                    dpl = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x).astype(float)
                    rpl = rprov.get_region(0, rm, rt, rz, rc, 0, rax.y,
                                           0, rax.x).astype(float)
                    _match(rpl, dpl)
                    res, prev_u[z] = _seeded(rpl, dpl, prev_u.get(z))
                    rows.append(_dvc_rows(res, vox, m=m, t=t, c=c, z_plane=z))
                    done_units += 1
                    ctx.progress(done_units, n_units, f"t={t} z={z}")

    # Provenance stamp (metadata intelligence): record HOW this field was correlated so a
    # downstream `analysis.accumulate_field` inherits the reference config instead of the
    # user re-specifying it, and refuses an already-cumulative field. An external reference
    # is always a FIXED reference (the lever governs self-reference only, V2.06), so it
    # stamps `fixed_frame` — its output is not incremental and must not be accumulated.
    eff_mode = ref_mode if ref_prov is None else "fixed_frame"
    # DVC-specific provenance (§7b). Dimensionality is NOT stamped here — it flows generically
    # as the Point table's z_kind (preserved by `with_structure` → `__struct_zkind__`), which
    # `accumulate_field` / `rasterize_field` inherit via `ds.structure_zkind`.
    prov_md = {
        "dvc_reference_mode": eff_mode, "dvc_reference_frame": ref_frame,
        "dvc_strain_type": dvc_params["strain_type"],
        "dvc_strain_smooth": dvc_params["strain_smooth"],
    }
    zk = "subpixel" if is_3d else "plane_index"
    if not rows:
        empty = point_table(np.zeros((0, 3 if is_3d else 2)), z_kind=zk, layer=layer)
        return ds.with_structure(empty).with_metadata(**prov_md)
    merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    merged["id"] = np.arange(len(merged["m"]), dtype=np.int64)               # global ids
    return (ds.with_structure(StructureTable(Domain.POINT, merged, layer=layer, z_kind=zk))
            .with_metadata(**prov_md))


register_node(
    _compute_dvc_field, op_key="analysis.dvc_field", label="DVC (ALDVC)",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InDataset("reference", label="Reference"),
        InInt("channel", "Channel", unit="", field=False, default=0),
        InInt("reference_frame", "Reference frame", unit="", field=False, default=0),
        InInt("subset_size", "Subset size", unit="px", field=False, default=16),
        InInt("subset_spacing", "Subset spacing", unit="px", field=False, default=10),
        InInt("search_radius", "Search radius", unit="px", field=False, default=0),
        InInt("seed_levels", "Seed levels", unit="", field=False, default=3),
        InInt("admm_iterations", "ADMM iterations", unit="", field=False, default=4),
        InFloat("mu", "ADMM μ", unit="", field=True, default=1e-3),
        InFloat("cc_thresh", "Correlation threshold", unit="", field=True, default=0.5),
        InFloat("strain_smooth", "Strain smoothing", unit="", field=True, default=0.0),
        InBool("newFFTSearch", "Fresh FFT seed each frame", field=False, default=False),
        InString("name", "Output layer", field=False, default="dvc",
                 layer_out=(Domain.POINT,)),
    ],
    outputs=[OutDataset()],
    modes=[DimMode(),
           Mode("reference_mode", ["fixed_frame", "previous_frame"],
                default="fixed_frame", label="Reference"),
           Mode("correlation", ["zncc", "phase"], default="zncc", label="Correlation"),
           Mode("strain_type", ["infinitesimal", "green-lagrange", "almansi", "hencky"],
                default="infinitesimal", label="Strain")],
    granularity=Granularity.WHOLE_SERIES,
    kernel_axes={"2D": frozenset({"t", "y", "x"}), "3D": frozenset({"t", "z", "y", "x"})},
    description="DVC/ALDVC displacement + strain field on a ref/def volume pair → a Point "
                "field (subset centers with disp µm / strain / qfactor); fixed-frame or "
                "previous-frame self-reference, or an optional external reference Dataset; "
                "2D per-plane vs 3D volume (ported v1 kernel).")


# ── Rasterize a Point vector field → Voxel layers (interpolate the coarse grid) ─
#
# The DVC output (and any Point structure carrying scalar attribute columns) lives on a
# COARSE grid. This node interpolates every non-coordinate attribute column up to the
# full voxel grid → one Voxel layer per column (`<source>_<column>`), so an image-space
# heatmap / composite is available. Scattered-data interpolation (scipy `griddata`) with
# a nearest-fill for out-of-hull voxels (matches the kernel's own inpaint philosophy).
# The 2D/3D lever picks per-plane (y,x) vs volumetric (z,y,x) interpolation — z_kind is
# not preserved through `with_structure`, so the lever (not the table) drives routing.


def _compute_rasterize_field(ctx: EvalContext) -> Dataset:
    """Interpolate a Point structure's attribute columns onto the full voxel grid → Voxel
    layers (one per column, named ``<source>_<column>``). 2D interpolates each ``(m,t,z,c)``
    plane from the points on that integer z-plane; 3D interpolates the ``(z,y,x)`` scatter
    onto the whole volume per ``(m,t,c)``. Out-of-hull voxels get the nearest point value
    (dense/finite output). ``method`` ∈ linear / nearest.

    Resolved spec (V2.06 + §7b): category transform; op ``transform.rasterize_field``; reads
    POINT + VOXEL (the image defines the target grid), adds VOXEL. **Dimensionality is
    INHERITED from the source Point layer's ``z_kind`` provenance** (``subpixel`` → 3D
    volumetric interpolation; ``plane_index`` → 2D per-plane) — NOT an independent 2D/3D
    lever, which could disagree with how the field was produced and silently misread a 2D
    per-plane field's plane-index ``z`` as a 3D coordinate (metadata-intelligence directive,
    `wire-node-v2` §7b). No calibration (pure grid geometry). Backend:
    :func:`scipy.interpolate.griddata` (lazily imported)."""
    from scipy.interpolate import griddata
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("rasterize field needs an image provider to define the voxel grid")
    ax = prov.axes
    source = ctx.layer("source")
    prefix = ctx.layer("prefix") or source
    method = ctx.params.get("__modes__", {}).get("method", "linear")
    # Inherit the field's dimensionality from its stamped z_kind (§7b); fall back to the
    # image's own volume-ness only for a hand-built field with no structure provenance.
    zk = ds.structure_zkind(Domain.POINT, source)
    is_3d = (zk == "subpixel") if zk is not None else (ax.z > 1)
    pts = [a for a in ds.layers_on(Domain.POINT) if a.layer == source]
    if not pts:
        raise ValueError(f"rasterize field needs a Point layer {source!r} "
                         "(run a DVC / point-field node first)")
    col = {a.name: np.asarray(a.values) for a in pts}
    for req in ("m", "t", "c", "z", "y", "x"):
        if req not in col:
            raise ValueError(f"Point layer {source!r} missing coordinate column {req!r}")
    coord = {"id", "m", "t", "c", "z", "y", "x"}
    attr_names = [n for n in col if n not in coord]
    if not attr_names:
        raise ValueError(f"Point layer {source!r} has no attribute columns to rasterize")
    m_all = col["m"].astype(int); t_all = col["t"].astype(int); c_all = col["c"].astype(int)
    z_all, y_all, x_all = col["z"], col["y"], col["x"]
    out = {n: np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float) for n in attr_names}

    def _interp(points: np.ndarray, values: np.ndarray, grid_pts: np.ndarray):
        if len(points) == 0:
            return None
        try:
            if method == "nearest" or len(points) < points.shape[1] + 1:
                return griddata(points, values, grid_pts, method="nearest")
            g = griddata(points, values, grid_pts, method=method)
            nan = np.isnan(g)
            if nan.any():                                   # out-of-hull → nearest fill
                g[nan] = griddata(points, values, grid_pts[nan], method="nearest")
            return g
        except Exception:                                   # degenerate hull → nearest
            return griddata(points, values, grid_pts, method="nearest")

    if is_3d:
        gz, gy, gx = np.mgrid[0:ax.z, 0:ax.y, 0:ax.x]
        grid_pts = np.column_stack([gz.ravel(), gy.ravel(), gx.ravel()]).astype(float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    sel = (m_all == m) & (t_all == t) & (c_all == c)
                    if not sel.any():
                        continue
                    P = np.column_stack([z_all[sel], y_all[sel], x_all[sel]])
                    for n in attr_names:
                        g = _interp(P, col[n][sel], grid_pts)
                        if g is not None:
                            out[n][m, t, :, c] = g.reshape(ax.z, ax.y, ax.x)
    else:
        gy, gx = np.mgrid[0:ax.y, 0:ax.x]
        grid_pts = np.column_stack([gy.ravel(), gx.ravel()]).astype(float)
        zi_all = np.rint(z_all).astype(int)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    for z in range(ax.z):
                        sel = ((m_all == m) & (t_all == t) & (c_all == c) & (zi_all == z))
                        if not sel.any():
                            continue
                        P = np.column_stack([y_all[sel], x_all[sel]])
                        for n in attr_names:
                            g = _interp(P, col[n][sel], grid_pts)
                            if g is not None:
                                out[n][m, t, z, c] = g.reshape(ax.y, ax.x)
    res = ds
    for n in attr_names:
        res = res.with_layer(Domain.VOXEL, f"{prefix}_{n}", out[n])
    return res


register_node(
    _compute_rasterize_field, op_key="transform.rasterize_field", label="Rasterize Field",
    category="transform",
    reads_domains=frozenset({Domain.POINT, Domain.VOXEL}),
    adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            InString("source", "Point layer", field=False, default="dvc",
                     layer_in=Domain.POINT),
            # empty => the layer prefix follows `source` (it names a FAMILY of output
            # layers, one per field component, not a single layer)
            InString("prefix", "Output prefix", field=False, default="")],
    outputs=[OutDataset()],
    # No DimMode lever — dimensionality is INHERITED from the source field's z_kind (§7b),
    # so a fixed volumetric footprint (a superset of the per-plane case) is the honest
    # declaration; the compute loops (m,t,z,c) internally and writes the whole raster.
    modes=[Mode("method", ["linear", "nearest"], default="linear", label="Interpolation")],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset({"z", "y", "x"}),
    description="Interpolate a Point vector/scalar field (e.g. DVC) up to full-resolution "
                "Voxel layers (one per attribute column); 2D-per-plane vs volumetric is "
                "inherited from the field's z_kind (§7b); nearest-filled out-of-hull.")


# ── Accumulate incremental DVC increments → cumulative Lagrangian fields ───────
#
# Ports the vendored ALDVC accumulation (`build_accumulated_results` / `accumulate_
# incremental`). A `analysis.dvc_field` run in `previous_frame` mode emits per-step
# increment fields (frame t−1 → t) as a Point series. FranckLab's incremental workflow
# then COMPOSES them into cumulative displacement-from-reference by Lagrangian point-
# tracking (advect the reference grid points through the increments, interpolating each
# increment at the points' current drifted position) and RECOMPUTES strain from the
# cumulative displacement — NOT a naive per-grid-point sum, and NOT the same result as a
# direct fixed-frame correlation. This node is that post-process. Metadata intelligence:
# it inherits the reference config (mode / dim / strain measure) from the upstream DVC
# node's stamped provenance — no redundant reference-frame control — and refuses a field
# that is already cumulative (fixed_frame / external reference).


def _reconstruct_grid(coords: np.ndarray, ndim: int):
    """Rebuild the ALDVC :class:`Grid` from a Point layer's subset-center coords
    ``(N, ndim)`` (voxels, one timepoint). The DVC grid is regular, so the per-axis
    unique sorted coordinates recover ``axes``/``grid_shape``/``step``; ``coords`` is the
    ``ij`` meshgrid. Requires a full grid (``N == prod(grid_shape)``)."""
    from nodegraph.kernels.aldvc_field import Grid
    axes = [np.unique(coords[:, a]).astype(np.float64) for a in range(ndim)]
    grid_shape = tuple(int(len(a)) for a in axes)
    if int(np.prod(grid_shape)) != coords.shape[0]:
        raise ValueError(
            f"accumulate field: {coords.shape[0]} points do not form a full regular "
            f"{grid_shape} subset grid — is the source a DVC field?")
    mesh = np.meshgrid(*axes, indexing="ij")
    coords_arr = np.stack(mesh, axis=-1).astype(np.float64)          # (*grid_shape, ndim)
    step = np.asarray([float(np.median(np.diff(a))) if len(a) > 1 else 1.0
                       for a in axes], dtype=np.float64)
    return Grid(axes=axes, coords=coords_arr, grid_shape=grid_shape, step=step, ndim=ndim)


def _place_on_grid(coords: np.ndarray, values: np.ndarray, grid) -> np.ndarray:
    """Scatter per-point ``values`` (``(N,)`` or ``(N, k)``) onto the grid in the kernel's
    C-order, mapping each point to its cell via a per-axis ``searchsorted`` (robust to row
    ordering). Returns ``(*grid_shape[, k])``."""
    idx = tuple(np.searchsorted(grid.axes[a], coords[:, a]) for a in range(grid.ndim))
    if values.ndim == 1:
        out = np.zeros(grid.grid_shape, dtype=np.float64)
        out[idx] = values
        return out
    out = np.zeros((*grid.grid_shape, values.shape[1]), dtype=np.float64)
    for k in range(values.shape[1]):
        out[(*idx, k)] = values[:, k]
    return out


def _compute_accumulate_field(ctx: EvalContext) -> Dataset:
    """Compose a `previous_frame` DVC increment series into cumulative displacement +
    strain fields (Lagrangian point-tracking). Reads the increment Point layer, rebuilds
    the subset grid per correlation series (``(m,c)`` in 3D; ``(m,c,z-plane)`` in 2D),
    runs :func:`accumulate_incremental` over ascending ``t``, recomputes strain from the
    cumulative displacement with :func:`compute_strain`, and emits a cumulative Point
    layer with the same schema (disp µm / strain / qfactor).

    Resolved spec (V2.06 addendum): category analysis; op ``analysis.accumulate_field``;
    reads POINT, adds POINT; footprint WHOLE_SERIES (crosses T to compose). **Metadata
    intelligence** — dim / reference-mode / strain measure are inherited from the upstream
    DVC node's stamped provenance (``dvc_dim`` / ``dvc_reference_mode`` / ``dvc_strain_*``),
    so there is no redundant reference control and an already-cumulative field (fixed_frame
    or external reference) is refused. Displacement voxels↔µm via ``ctx.calib`` (the DVC
    output carries the same series calibration). Kernel:
    :func:`nodegraph.kernels.aldvc_field.accumulate_incremental` / ``compute_strain``."""
    from types import SimpleNamespace
    from nodegraph.kernels.aldvc_field import accumulate_incremental, compute_strain
    ds = ctx.inputs[0]
    source = ctx.layer("source")
    out_layer = ctx.layer("name") or f"{source}_cumulative"
    md = ds.metadata
    ref_mode = md.get("dvc_reference_mode")
    if ref_mode is None:
        raise ValueError(
            "accumulate field expects a DVC-produced Point field (missing dvc provenance); "
            "run analysis.dvc_field in previous_frame mode upstream")
    if ref_mode != "previous_frame":
        raise ValueError(
            f"accumulate field needs previous_frame increments, but the input DVC field was "
            f"correlated in {ref_mode!r} mode (already cumulative — nothing to accumulate)")
    # Dimensionality inherited from the field's generic z_kind provenance (§7b), not a lever.
    is_3d = ds.structure_zkind(Domain.POINT, source) == "subpixel"
    ndim = 3 if is_3d else 2
    strain_type = md.get("dvc_strain_type", "infinitesimal")
    strain_smooth = float(md.get("dvc_strain_smooth", 0.0))
    px = ctx.calib("pixel_size_um") or 0.1
    if is_3d:
        zs = ctx.calib("z_step_um") or 0.5
        vox = np.asarray([zs, px, px], dtype=np.float64)
    else:
        vox = np.asarray([px, px], dtype=np.float64)

    pts = [a for a in ds.layers_on(Domain.POINT) if a.layer == source]
    if not pts:
        raise ValueError(f"accumulate field needs a Point layer {source!r} "
                         "(run a previous_frame DVC node first)")
    col = {a.name: np.asarray(a.values) for a in pts}
    for req in ("m", "t", "c", "z", "y", "x"):
        if req not in col:
            raise ValueError(f"Point layer {source!r} missing coordinate column {req!r}")
    disp_cols = ("disp_z", "disp_y", "disp_x") if is_3d else ("disp_y", "disp_x")
    for dn in disp_cols:
        if dn not in col:
            raise ValueError(f"Point layer {source!r} missing displacement column {dn!r} "
                             f"(is it a {ndim}D DVC field?)")
    m_all = col["m"].astype(int); t_all = col["t"].astype(int); c_all = col["c"].astype(int)
    z_all = col["z"].astype(float); y_all = col["y"].astype(float); x_all = col["x"].astype(float)
    qf_all = col.get("qfactor")

    def _series(sel_series: np.ndarray, z_plane: Optional[int]) -> list:
        """Accumulate one correlation series (a boolean mask over all rows sharing the same
        (m,c[,z-plane])), emitting cumulative rows per timepoint. ``z_plane`` stamps the 2D
        plane index (None in 3D, where z is a grid coordinate)."""
        ts = sorted(set(t_all[sel_series].tolist()))
        if not ts:
            return []
        # coords of the (identical) subset grid — take the earliest timepoint present.
        first = sel_series & (t_all == ts[0])
        gc = (np.column_stack([z_all[first], y_all[first], x_all[first]]) if is_3d
              else np.column_stack([y_all[first], x_all[first]]))
        grid = _reconstruct_grid(gc, ndim)
        m_i = int(m_all[first][0]); c_i = int(c_all[first][0])
        # increments (voxels) per t, placed onto the grid in kernel C-order.
        incr = []
        qf_by_t: Dict[int, np.ndarray] = {}
        for t in ts:
            selt = sel_series & (t_all == t)
            ct = (np.column_stack([z_all[selt], y_all[selt], x_all[selt]]) if is_3d
                  else np.column_stack([y_all[selt], x_all[selt]]))
            disp_um = np.column_stack([col[dn][selt] for dn in disp_cols])   # (n, ndim)
            disp_vox = disp_um / vox                                          # µm → voxels
            u_grid = np.moveaxis(_place_on_grid(ct, disp_vox, grid), -1, 0)   # (ndim,*grid)
            incr.append((int(t), u_grid))
            if qf_all is not None:
                qf_by_t[int(t)] = _place_on_grid(ct, qf_all[selt].astype(float),
                                                 grid).reshape(-1)
        accum = accumulate_incremental(grid, incr)                            # cumulative
        gc_flat = grid.coords_flat()
        out_rows = []
        for t, u_acc in accum:                                               # (ndim,*grid)
            _F, strain = compute_strain(u_acc, grid.step, voxel_size=vox,
                                        strain_type=strain_type,
                                        smooth_sigma=strain_smooth)
            r = SimpleNamespace(
                dim=ndim, grid_coords=gc_flat,
                displacement_field=np.moveaxis(u_acc, 0, -1),                 # (*grid,ndim) vox
                strain_field=np.moveaxis(strain, (0, 1), (-2, -1)),          # (*grid,ndim,ndim)
                qfactor=qf_by_t.get(t))
            out_rows.append(_dvc_rows(r, vox, m=m_i, t=t, c=c_i, z_plane=z_plane))
        return out_rows

    rows: list = []
    for m in sorted(set(m_all.tolist())):
        for c in sorted(set(c_all[m_all == m].tolist())):
            base = (m_all == m) & (c_all == c)
            if is_3d:
                rows.extend(_series(base, None))
            else:
                for z in sorted(set(np.rint(z_all[base]).astype(int).tolist())):
                    rows.extend(_series(base & (np.rint(z_all).astype(int) == z), z))

    zk = "subpixel" if is_3d else "plane_index"        # → auto-stamped z_kind provenance
    prov_md = {"dvc_reference_mode": "cumulative",
               "dvc_strain_type": strain_type, "dvc_strain_smooth": strain_smooth}
    if not rows:
        empty = point_table(np.zeros((0, 3 if is_3d else 2)), z_kind=zk, layer=out_layer)
        return ds.with_structure(empty).with_metadata(**prov_md)
    merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    merged["id"] = np.arange(len(merged["m"]), dtype=np.int64)
    return (ds.with_structure(StructureTable(Domain.POINT, merged, layer=out_layer, z_kind=zk))
            .with_metadata(**prov_md))


register_node(
    _compute_accumulate_field, op_key="analysis.accumulate_field",
    label="Accumulate DVC Field", category="analysis",
    extra_layers=_layers_accumulate_field,
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[InDataset(),
            InString("source", "Increment layer", field=False, default="dvc",
                     layer_in=Domain.POINT),
            InString("name", "Output layer", field=False, default="")],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_SERIES,
    kernel_axes=frozenset({"t", "z", "y", "x"}),
    description="Compose a previous-frame DVC increment series into cumulative "
                "displacement + strain fields by Lagrangian point-tracking (ported ALDVC "
                "build_accumulated_results); inherits the reference config from the upstream "
                "DVC node; emits a cumulative Point layer with the same schema.")


# ── DIC (pyALDIC) — 2D digital image correlation, the image sibling of DVC ──────
#
# Wires the vendored `dic_correlate` kernel (nd2studios v1 → `nodegraph.kernels.dic_correlate`).
# The kernel is a thin adapter around the third-party **al-dic (pyALDIC)** solver, which it
# imports LAZILY (inside `run_pyaldic_pair`). So this node is FULLY wired — it owns the m/t/z
# loop, the reference pairing, unit handling, and the Point output — but its compute raises a
# friendly ImportError only when actually RUN, until `pip install al-dic` (upstream:
# https://github.com/zachtong/pyALDIC). This mirrors the DVC port (`analysis.dvc_field`): DIC
# is its 2D form (correlate a ref/def IMAGE pair → a displacement field on a coarse FE grid).


def _compute_dic_correlate(ctx: EvalContext) -> Dataset:
    """2D Digital Image Correlation (pyALDIC: IC-GN subset matching + ADMM over an adaptive
    quadtree FE mesh) → a **Point** displacement field on the correlation grid — the 2D image
    sibling of ``analysis.dvc_field``. Loops m/t/z on one reference channel; each ref/def
    PLANE pair is correlated by :func:`nodegraph.kernels.dic_correlate.run_pyaldic_pair`,
    which imports the external ``al-dic`` package **lazily** → the node is fully wired but
    raises a clear ImportError at run time until ``pip install al-dic`` (kernel ready).

    Reference pairing mirrors DVC: a ``reference_mode`` lever {fixed_frame | previous_frame}
    + ``reference_frame``, OR an optional external ``reference`` Dataset (a separate undeformed
    acquisition — always a FIXED reference). An optional ``roi`` Voxel mask layer (e.g. from
    ``analysis.roi_mask``) restricts correlation to a region. Displacements are stored in
    **µm** (grid px × ``pixel_size_um``, component order (dy, dx)); the FE grid is COARSER than
    the image (pitch = the snapped ``winstepsize``) — feed the Point field to
    ``transform.rasterize_field`` for a Voxel heatmap. 2D only (DIC has no volumetric form —
    use ``analysis.dvc_field`` for 3D). Footprint WHOLE_SERIES (crosses T); reuses the shared
    :func:`_dvc_rows` Point flattener (DIC's ``DVCResult`` has ``strain_field``/``qfactor`` =
    None, so those columns are simply absent)."""
    from nodegraph.kernels.dic_correlate import run_pyaldic_pair
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("DIC needs an image provider on its (deformed) input Dataset")
    ax = prov.axes
    ref_ds = ctx.input("reference")
    ref_prov = ref_ds.image if ref_ds is not None else None
    if ref_ds is not None and ref_prov is None:
        raise ValueError("the DIC 'reference' input Dataset has no image provider")
    modes = ctx.params.get("__modes__", {})
    ref_mode = modes.get("reference_mode", "fixed_frame")
    ref_frame = int(ctx.params.get("reference_frame", 0))
    c = min(max(0, int(ctx.params.get("channel", 0))), max(0, ax.c - 1))
    px = ctx.calib("pixel_size_um") or 0.1
    vox: Tuple[float, ...] = (px, px)                    # (y, x) µm/px — kernel §3
    params = {
        "winsize": max(2, int(ctx.params.get("winsize", 40))),
        "winstepsize": max(2, int(ctx.params.get("winstepsize", 16))),
        "winsize_min": max(2, int(ctx.params.get("winsize_min", 8))),
        "init_guess_mode": "auto",
        "mu": float(ctx.params.get("mu", 1e-3)),
        "tol": float(ctx.params.get("tol", 1e-2)),
        "admm_max_iter": max(1, int(ctx.params.get("admm_max_iter", 3))),
        "icgn_max_iter": max(1, int(ctx.params.get("icgn_max_iter", 100))),
        "disp_smoothness": max(0.0, float(ctx.params.get("disp_smoothness", 5e-4))),
        "strain_smoothness": max(0.0, float(ctx.params.get("strain_smoothness", 1e-5))),
        "compute_strain": False,     # adapter drops strain (strain_field=None); derive downstream
    }
    layer = ctx.layer("name")
    # optional ROI: a Voxel mask layer (e.g. analysis.roi_mask) limits correlation; absent → none
    roi_name = ctx.layer("roi")
    roi_attr = ds.get(Domain.VOXEL, roi_name) if roi_name else None
    roi6 = roi_attr.values if roi_attr is not None else None

    def _resolve_ref(m: int, t: int):
        """(ref_provider, ref_m, ref_t) for this def frame, or None to skip it."""
        if ref_prov is not None:                          # external reference file (fixed)
            rax = ref_prov.axes
            return ref_prov, min(m, rax.m - 1), min(max(0, ref_frame), rax.t - 1)
        if ref_mode == "previous_frame":
            return None if t == 0 else (prov, m, t - 1)    # no increment into t0
        return prov, m, min(max(0, ref_frame), ax.t - 1)   # fixed_frame

    def _roi_plane(m: int, t: int, z: int, cc: int):
        if roi6 is None:
            return None
        s = roi6.shape
        return roi6[min(m, s[0] - 1), min(t, s[1] - 1), min(z, s[2] - 1), min(cc, s[3] - 1)]

    rows: list = []
    n_units = ax.m * ax.t * ax.z          # eager: one IC-GN/ADMM solve per plane pair
    done_units = 0
    ctx.progress(0, n_units, "correlating")
    for m in range(ax.m):
        for t in range(ax.t):
            resolved = _resolve_ref(m, t)
            if resolved is None:
                done_units += ax.z                     # skipped t0 still advances the bar
                ctx.progress(done_units, n_units, "correlating")
                continue
            rprov, rm, rt = resolved
            rax = rprov.axes
            rc = min(c, max(0, rax.c - 1))
            for z in range(ax.z):
                rz = min(z, max(0, rax.z - 1))
                dpl = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x).astype(float)
                rpl = rprov.get_region(0, rm, rt, rz, rc, 0, rax.y, 0, rax.x).astype(float)
                if rpl.shape != dpl.shape:
                    raise ValueError(
                        f"DIC reference plane {rpl.shape} != deformed {dpl.shape}; the "
                        "reference Dataset must match the primary's Y/X")
                res = run_pyaldic_pair(rpl, dpl, vox, params,
                                       roi_mask=_roi_plane(m, t, z, c))
                rows.append(_dvc_rows(res, vox, m=m, t=t, c=c, z_plane=z))
                done_units += 1
                ctx.progress(done_units, n_units, f"t={t} z={z}")

    # Provenance (§7b): DIC-specific keys (NOT the dvc_* keys accumulate_field consumes — a
    # DIC 2D increment is not an ALDVC volume increment, so it must not be fed there).
    prov_md = {"dic_reference_mode": (ref_mode if ref_prov is None else "fixed_frame"),
               "dic_reference_frame": ref_frame}
    if not rows:
        empty = point_table(np.zeros((0, 2)), z_kind="plane_index", layer=layer)
        return ds.with_structure(empty).with_metadata(**prov_md)
    merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    merged["id"] = np.arange(len(merged["m"]), dtype=np.int64)               # global ids
    return (ds.with_structure(
        StructureTable(Domain.POINT, merged, layer=layer, z_kind="plane_index"))
        .with_metadata(**prov_md))


register_node(
    _compute_dic_correlate, op_key="analysis.dic_correlate", label="DIC (pyALDIC)",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InDataset("reference", label="Reference"),
        InInt("channel", "Channel", unit="", field=False, default=0),
        InInt("reference_frame", "Reference frame", unit="", field=False, default=0),
        InInt("winsize", "Subset size", unit="px", field=False, default=40),
        InInt("winstepsize", "Grid step", unit="px", field=False, default=16),
        InInt("winsize_min", "Min element", unit="px", field=False, default=8),
        InInt("icgn_max_iter", "IC-GN iterations", unit="", field=False, default=100),
        InInt("admm_max_iter", "ADMM iterations", unit="", field=False, default=3),
        InFloat("mu", "ADMM μ", unit="", field=True, default=1e-3),
        InFloat("tol", "Tolerance", unit="", field=True, default=1e-2),
        InFloat("disp_smoothness", "Displacement smoothing", unit="", field=True,
                default=5e-4),
        InFloat("strain_smoothness", "Strain smoothing", unit="", field=True, default=1e-5),
        InString("roi", "ROI mask layer", field=False, default="roi_mask",
                 layer_in=Domain.VOXEL),
        InString("name", "Output layer", field=False, default="dic",
                 layer_out=(Domain.POINT,)),
    ],
    outputs=[OutDataset()],
    modes=[Mode("reference_mode", ["fixed_frame", "previous_frame"],
                default="fixed_frame", label="Reference")],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"t", "y", "x"}),
    description="2D digital image correlation (pyALDIC: IC-GN + ADMM over an adaptive FE "
                "mesh) → a Point displacement field (grid centers, disp µm); fixed/previous "
                "self-reference or an external reference Dataset + optional ROI mask. Wired "
                "to the vendored kernel; needs al-dic (dep-gated: friendly ImportError at "
                "run time until installed).")


# ── Track Objects (the vendored v1 `track_objects` kernel — five linkers) ───────
#
# The richer sibling of `track.link`. Where `track.link` ships the two dep-free
# pure-numpy linkers in `nodegraph.tracking` (max-IoU overlap / nearest neighbour),
# this wraps the vendored v1 kernel and exposes its FIVE interchangeable linking
# methods. It is a SEPARATE node rather than a mode on `track.link` because the kernel
# imports numba and pandas at module scope, while `nodegraph.tracking` is deliberately
# dep-free — folding it in would make the built-in linker unimportable without numba.
#
# The kernel consumes measurement ROW-DICTS, which v2 Label/Point structure tables
# already carry (`id,m,t,c,z,y,x` + Label `area`, all in pixels — structure.py
# `_label_table`). Both trackers therefore read the same upstream layers and emit the
# same Track membership shape, so they are drop-in alternatives to each other.

#: v2 `method` mode value → the kernel's ``METHOD_*`` attribute name. The kernel's
#: dispatch ends in a bare ``else:`` that silently runs the centroid linker on an
#: unrecognized string, so the node maps + validates rather than passing one through.
_TRACK_OBJECT_METHODS = {
    "centroid": "METHOD_CENTROID",
    "serialtrack": "METHOD_SERIALTRACK",
    "topology": "METHOD_CT_TOPOLOGY",
    "fingerprint": "METHOD_CT_FINGERPRINT",
    "overlap": "METHOD_CT_OVERLAP",
}


def _compute_track_objects(ctx: EvalContext) -> Dataset:
    """Frame-to-frame object tracking via the vendored v1 ``track_objects`` kernel →
    a Track membership + a ``track_id`` column written back onto the member layer.

    **Resolved spec (§0 grill, 2026-07-27).**

    *Kind* analysis; reads a Label **or** Point structure layer (the ``target`` mode,
    mirroring ``track.link``) and adds the Track domain. *Members* keep their upstream
    layer; the tracker never re-derives geometry.

    *Rows.* Each member row becomes one kernel row-dict. v2 Label tables already carry
    every key the kernel needs — ``y``/``x`` → ``centroid_y_px``/``centroid_x_px`` and
    ``area`` → ``area_px``, all in pixels (``structure._label_table``). Point tables have
    **no** ``area`` column, so ``area_px`` is 0.0 there. For the *fingerprint* linker that
    is harmless (area is a weighted cost term, not a gate), but for the *centroid* linker
    ``max_size_diff_frac`` IS a gate and the kernel neutralises it on zero areas — so that
    socket is both hidden on Point members and refused below if explicitly set.

    *Grouping.* The kernel groups by ``(segmentation_channel, m_position)`` and runs each
    group independently off ONE shared track-id counter. This node keys those as
    ``(f"c{c}z{z}", m)`` — a compound channel key — so every ``(m, c, z)`` is tracked
    independently, which is what makes a 2D-only kernel correct on a z-stack: planes are
    never linked to each other. Do **not** pre-split; one call keeps ids unique.

    *2D only.* Every linker builds a two-column ``[y, x]`` array and ``track_overlap``
    hard-raises unless its masks are ``(T,H,W)``. A 3D (``z_kind="subpixel"``) structure
    is refused with a pointer to ``track.link`` point mode, which links in true µm 3D.
    Dimensionality is inherited from the members' ``z_kind`` (§7b) — never a DimMode
    lever, which could silently disagree with the data.

    *Determinism (memo invariant, §10).* Verified empirically: every linker is
    repeat-stable, and row order changes only the track-id NUMBERING, never the induced
    partition. Two guards make the node fully deterministic anyway — rows are emitted in
    a canonical ``lexsort`` order ``(m, c, z, t, id)``, and the result is renumbered
    through ``tracking.build_membership`` (contiguous ``1..K`` by first appearance, rows
    sorted ``(track_id, t, member_id)``) so ids match ``track.link``'s conventions
    exactly. ``st_use_prev_results`` is pinned ``False``: its POD-GPR warm start draws on
    numpy's global RNG with no seed, which would break memo determinism outright.

    *Hard refusals* where the kernel would otherwise degrade in silence: an unknown
    method (dispatch falls through to centroid), ``overlap`` without its label raster
    (falls back to the fingerprint linker with hardcoded weights), ``overlap`` on Point
    members, duplicate member ids (the Cell-Tracker bridge keys a plain dict on
    ``(frame, label_id)`` and both duplicates end up untracked), ragged columns, an
    out-of-range ``t``, an inert ``max_size_diff_frac`` on arealess members, and
    ``ct_max_gap=0`` under ``overlap`` (the kernel floors that path at 1). Sockets a
    method does not consume are hidden from it rather than accepted and discarded —
    notably ``max_distance``, which the ``overlap`` linker never receives.

    *Sockets* are all ``field=False``: this node has no lattice iteration and no
    ``FieldContext``, so a wired per-voxel Field would be silently discarded.
    ``min_circularity``/``max_eccentricity`` are deliberately **omitted** — nothing in
    the v2 catalog emits circularity or eccentricity, so those sockets could only ever
    error; the kernel defaults disable both filters.

    *Footprint* ``WHOLE_SERIES`` over ``{t,z,y,x}`` (linking is global in T, and the
    overlap linker reads the whole label raster).
    """
    from nodegraph.kernels import track_objects as _tk
    from nodegraph.tracking import build_membership

    ds: Dataset = ctx.inputs[0]
    ax = ds.axes
    modes = ctx.params.get("__modes__", {})
    target = modes.get("target", "label")
    method_key = modes.get("method", "centroid")
    if method_key not in _TRACK_OBJECT_METHODS:
        raise ValueError(
            f"track objects: unknown method {method_key!r} "
            f"(choose one of {sorted(_TRACK_OBJECT_METHODS)})")
    method = getattr(_tk, _TRACK_OBJECT_METHODS[method_key])

    member_domain = Domain.LABEL if target == "label" else Domain.POINT
    src = (ctx.layer("labels") if target == "label"
           else ctx.layer("points"))

    # ── pull + validate the member table ──────────────────────────────────────
    need = ("id", "m", "t", "c", "z", "y", "x")
    attrs = {k: ds.get(member_domain, k, layer=src) for k in need}
    if attrs["id"] is None:
        raise ValueError(
            f"track objects: no {member_domain.value} structure {src!r} on the input "
            f"Dataset (run analysis.label / detect.spots upstream, or point the layer "
            f"socket at the right layer)")
    missing = sorted(k for k, v in attrs.items() if v is None)
    if missing:
        raise ValueError(
            f"track objects: the {member_domain.value} layer {src!r} is missing the "
            f"invariant column(s) {missing} needed to build tracking rows")
    vals = {k: np.asarray(v.values) for k, v in attrs.items()}
    n = len(vals["id"])
    ragged = sorted(k for k, v in vals.items() if len(v) != n)
    if ragged:
        raise ValueError(
            f"track objects: column(s) {ragged} on layer {src!r} disagree in length with "
            f"'id' ({n}) — the rows would be built from misaligned data")

    zk = ds.structure_zkind(member_domain, src) or "plane_index"
    if zk == "subpixel":
        raise ValueError(
            "track objects is 2D-only: every linker in the vendored kernel correlates a "
            "two-column (y, x) centroid array and the overlap linker needs (T,H,W) masks. "
            f"The {member_domain.value} layer {src!r} is 3D (z_kind='subpixel'). Use "
            "track.link in point mode, which links in true 3D µm coordinates.")

    ids = vals["id"].astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise ValueError(
            f"track objects needs globally-unique member ids, but layer {src!r} repeats "
            f"some (n={ids.size}, unique={np.unique(ids).size}). The Cell-Tracker linkers "
            f"key a dict on (frame, label_id), so duplicates silently drop BOTH rows.")
    mm = vals["m"].astype(np.int64)
    cc = vals["c"].astype(np.int64)
    tt = np.rint(vals["t"]).astype(np.int64)
    zz = np.rint(vals["z"]).astype(np.int64)          # 2D ⇒ z is the plane index
    if n and (int(tt.min()) < 0 or int(tt.max()) >= ax.t):
        raise ValueError(
            f"track objects: layer {src!r} has t outside the Dataset's T axis "
            f"(t∈[{int(tt.min())},{int(tt.max())}], T={ax.t})")
    yy = vals["y"].astype(float)
    xx = vals["x"].astype(float)
    area_attr = ds.get(member_domain, "area", layer=src)
    area = (np.asarray(area_attr.values, dtype=float) if area_attr is not None
            and len(area_attr.values) == n else np.zeros(n, dtype=float))

    # Canonical row order: the kernel hands out track ids in first-appearance order, so a
    # stable input order is what makes the raw ids reproducible (we renumber below anyway).
    order = np.lexsort((ids, tt, zz, cc, mm))         # primary key last
    rows = [{"segmentation_channel": f"c{int(cc[i])}z{int(zz[i])}",
             "m_position": int(mm[i]),
             "frame": int(tt[i]),
             "label_id": int(ids[i]),
             "centroid_y_px": float(yy[i]),
             "centroid_x_px": float(xx[i]),
             "area_px": float(area[i])}
            for i in order.tolist()]

    # ── the overlap linker's (T,H,W) label masks, keyed like the row groups ───
    label_masks = None
    if method_key == "overlap":
        if member_domain is not Domain.LABEL:
            raise ValueError(
                "the 'overlap' method matches label masks by IoU and is unavailable for "
                "Point members (no raster to intersect) — pick centroid / topology / "
                "fingerprint / serialtrack, or track a Label layer.")
        raster_attr = ds.get(Domain.VOXEL, src)
        if raster_attr is None:
            raise ValueError(
                f"the 'overlap' method needs the Voxel label raster {src!r} that carries "
                f"the member ids (analysis.label emits the raster and the table under one "
                f"layer name). Without it the kernel silently falls back to the "
                f"fingerprint linker with hardcoded weights.")
        raster6 = raster_attr.values                  # (m,t,z,c,y,x); ids match the table
        label_masks = {}
        for m_i in np.unique(mm).tolist():
            for c_i in np.unique(cc).tolist():
                for z_i in np.unique(zz).tolist():
                    label_masks[(f"c{int(c_i)}z{int(z_i)}", int(m_i))] = \
                        raster6[int(m_i), :, int(z_i), int(c_i)]

    # `overlap` matches purely by mask IoU: the kernel's live path takes no distance
    # bound at all (`max_displacement_px` reaches only the fingerprint FALLBACK branch,
    # which the missing-raster refusal above makes unreachable). So the socket is hidden
    # for it — and calibration is not read either, since fencing the memo on a pixel size
    # that cannot change the result would invalidate cached tracks for nothing.
    if method_key == "overlap":
        max_disp_px = 100.0                   # the kernel default; never consulted
        # One socket serves both CT gap methods, but the kernel floors the overlap path at
        # `max(1, ct_max_gap)` while fingerprint honours 0 — so 0 would silently bridge a
        # one-frame hole here. Refuse rather than accept-and-rewrite (the vendored kernel
        # stays byte-verbatim; the deviation is documented in its .md as `>= 0`).
        if int(ctx.params.get("ct_max_gap", 3)) < 1:
            raise ValueError(
                "track objects: the 'overlap' linker floors its frame gap at 1, so "
                "ct_max_gap=0 would still bridge a one-frame hole instead of requiring "
                "consecutive detections. Use ct_max_gap>=1, or the 'fingerprint' method, "
                "which honours 0.")
    else:
        px = ctx.calib("pixel_size_um") or 0.1
        max_disp_px = max(1.0, to_pixels_v2(
            float(ctx.params.get("max_distance", 5.0)), "um", pixel_size_um=px))

    # The centroid linker's size gate divides by max(area); the kernel neutralises itself
    # on all-zero areas (`area_max[area_max <= 0] = 1.0` ⇒ size_diff ≡ 0), so on Point
    # members — which carry no `area` column, ever — an explicit gate would silently do
    # nothing and let the Hungarian solver swap identities it was set to keep apart.
    size_gate = float(ctx.params.get("max_size_diff_frac", 1.0))
    if method_key == "centroid" and size_gate < 1.0 and not area.any():
        raise ValueError(
            "track objects: 'Max size diff' gates the centroid linker on "
            f"|Δarea|/max(area), but layer {src!r} carries no 'area' column (Point tables "
            "never do), so the gate would be silently inert and identities could swap. "
            "Leave it at 1.0, or track a Label layer.")

    # ── link (the kernel mutates `rows` in place and returns the same list) ───
    kp = dict(
        max_displacement_px=max_disp_px,
        min_track_length=int(ctx.params.get("min_track_length", 2)),
        max_size_diff_frac=size_gate,
        max_frame_gap=int(ctx.params.get("max_frame_gap", 0)),
        method=method,
        st_n_neighbors=int(ctx.params.get("st_n_neighbors", 25)),
        st_smoothness=float(ctx.params.get("st_smoothness", 0.1)),
        st_use_prev_results=False,        # pinned — unseeded POD-GPR would break the memo
        ct_n_neighbors=int(ctx.params.get("ct_n_neighbors", 5)),
        ct_topo_weight=float(ctx.params.get("ct_topo_weight", 0.3)),
        ct_area_weight=float(ctx.params.get("ct_area_weight", 0.3)),
        ct_max_gap=int(ctx.params.get("ct_max_gap", 3)),
        ct_min_iou=float(ctx.params.get("ct_min_iou", 0.1)),
    )
    if label_masks is not None:
        kp["label_masks"] = label_masks
    _tk.link_objects(rows, **kp)

    # ── renumber through the shared membership builder (track.link conventions) ─
    # Kernel gotcha #8 ("two-frame minimum", track_objects.md §6): EVERY linker
    # early-returns on a (m,c,z) group spanning fewer than 2 distinct frames, leaving its
    # rows track_id=None BEFORE the min_track_length post-pass runs. At min_track_length
    # <= 1 the user has asked for 1-frame tracks, so seed those rows as singletons (what
    # track.link emits) — otherwise a lone detection's fate would depend on whether an
    # UNRELATED object in the same group happens to exist at some other frame. At the
    # default (>= 2) `short_groups` is empty, so this is a no-op on the normal path.
    short_groups: set = set()
    if int(ctx.params.get("min_track_length", 2)) <= 1:
        gframes: Dict[tuple, set] = {}
        for r in rows:
            gframes.setdefault((r["segmentation_channel"], r["m_position"]),
                               set()).add(int(r["frame"]))
        short_groups = {k for k, fs in gframes.items() if len(fs) < 2}
    t_of: Dict[int, int] = {}
    per_track: Dict[int, list] = {}
    for r in rows:
        tid = r.get("track_id")
        if tid is None:                   # excluded, or shorter than min_track_length
            if (r["segmentation_channel"], r["m_position"]) in short_groups:
                t_of[int(r["label_id"])] = int(r["frame"])      # singleton — no links
            continue
        mid = int(r["label_id"])
        t_of[mid] = int(r["frame"])
        per_track.setdefault(int(tid), []).append(mid)
    links: list = []
    for members in per_track.values():    # chain each kernel track into pairwise links
        chain = sorted(members, key=lambda mid2: (t_of[mid2], mid2))
        links.extend(zip(chain, chain[1:]))
    mem = build_membership(t_of, links, member_domain)

    # ── emit: the Track table (+ per-row extras) and the member write-back ────
    if mem.n:
        _uniq, inverse, counts = np.unique(mem.track_id, return_inverse=True,
                                           return_counts=True)
        length_col = counts[inverse].astype(np.int64)
        m_of = dict(zip(ids.tolist(), mm.tolist()))
        c_of = dict(zip(ids.tolist(), cc.tolist()))
        m_col = np.array([m_of[int(i)] for i in mem.member_id.tolist()], dtype=np.int64)
        c_col = np.array([c_of[int(i)] for i in mem.member_id.tolist()], dtype=np.int64)
    else:
        length_col = m_col = c_col = np.array([], dtype=np.int64)
    out_layer = ctx.layer("name")
    # Bypasses TrackMembership.to_table (a hard-coded 3-key literal) to carry the per-row
    # extras. The viewer's gate is a superset test, so they ride along for the spreadsheet
    # and CSV export; the Track bridges read only the three membership arrays.
    track_tbl = StructureTable(Domain.TRACK, {
        "track_id": mem.track_id, "t": mem.t, "member_id": mem.member_id,
        "track_length": length_col, "m": m_col, "c": c_col,
    }, layer=out_layer, z_kind="plane_index")

    # Write-back: a `track_id` column on the MEMBER layer, in that layer's original row
    # order (0 = untracked, matching the bridges' drop_nonpositive background rule). This
    # is what makes the result reachable to the rest of the catalog — nothing in v2
    # consumes Domain.TRACK, so without it the tracking would be viewer/export-only.
    # Only this one column is re-emitted, so the layer's other columns keep their order
    # (with_structure explodes a table into per-column layers — re-emitting `id` in a
    # different order would silently misalign every sibling column).
    tid_of = dict(zip(mem.member_id.tolist(), mem.track_id.tolist()))
    back = np.array([tid_of.get(int(i), 0) for i in ids.tolist()], dtype=np.int64)

    prov_md = {"track_method": method_key,
               "track_member_domain": member_domain.value,
               "track_member_layer": src}
    return (ds.with_structure(track_tbl)
              .with_structure(StructureTable(member_domain, {"track_id": back},
                                             layer=src, z_kind=zk))
              .with_metadata(**prov_md))


register_node(
    _compute_track_objects,
    op_key="track.objects", label="Track Objects", category="analysis",
    adds_domains=frozenset({Domain.TRACK}),   # reads Label OR Point (per 'target' mode)
    inputs=[
        InDataset(),
        InString("labels", "Label layer", field=False, default="labels",
                 layer_in=Domain.VOXEL,
                 available_in={"target": frozenset({"label"})}),
        InString("points", "Point layer", field=False, default="spots",
                 layer_in=Domain.POINT,
                 available_in={"target": frozenset({"point"})}),
        InString("name", "Output layer", field=False, default="tracks",
                 layer_out=(Domain.TRACK,)),
        InFloat("max_distance", "Max distance", unit="um", field=False, default=5.0,
                available_in={"method": frozenset({"centroid", "serialtrack",
                                                   "topology", "fingerprint"})}),
        InInt("min_track_length", "Min track length", unit="", field=False, default=2),
        InInt("max_frame_gap", "Max frame gap", unit="", field=False, default=0,
              available_in={"method": frozenset({"centroid"})}),
        InFloat("max_size_diff_frac", "Max size diff", unit="", field=False, default=1.0,
                available_in={"method": frozenset({"centroid"}),
                              "target": frozenset({"label"})}),
        InInt("ct_n_neighbors", "Neighbours", unit="", field=False, default=5,
              available_in={"method": frozenset({"topology"})}),
        InFloat("ct_topo_weight", "Topology weight", unit="", field=False, default=0.3,
                available_in={"method": frozenset({"topology"})}),
        InFloat("ct_area_weight", "Area weight", unit="", field=False, default=0.3,
                available_in={"method": frozenset({"fingerprint"})}),
        InInt("ct_max_gap", "Max gap", unit="", field=False, default=3,
              available_in={"method": frozenset({"fingerprint", "overlap"})}),
        InFloat("ct_min_iou", "Min IoU", unit="", field=False, default=0.1,
                available_in={"method": frozenset({"overlap"})}),
        InInt("st_n_neighbors", "ST neighbours", unit="", field=False, default=25,
              available_in={"method": frozenset({"serialtrack"})}),
        InFloat("st_smoothness", "ST smoothness", unit="", field=False, default=0.1,
                available_in={"method": frozenset({"serialtrack"})}),
    ],
    outputs=[OutDataset()],
    modes=[Mode("target", ["label", "point"], default="label", label="Members"),
           Mode("method", ["centroid", "serialtrack", "topology", "fingerprint",
                           "overlap"], default="centroid", label="Method")],
    granularity=Granularity.WHOLE_SERIES,
    kernel_axes=frozenset({"t", "z", "y", "x"}),
    description="Frame-to-frame object tracking with five interchangeable linkers "
                "(centroid Hungarian / SerialTrack topology PTV / Cell-Tracker topology, "
                "fingerprint, mask-overlap IoU) → a Track membership + a track_id column "
                "on the member layer. 2D per (m,c,z); Label or Point members. The richer "
                "alternative to track.link. NOTE: serialtrack is 1–2 orders of magnitude "
                "slower than the others (≈8-18 s on 1k–15k detections, plus a one-time "
                "numba JIT) — the cheap default is centroid.")


# ── zone boundary nodes (Repeat/Simulation In/Out — identity pass-through) ──────
#
# The paired boundary markers of a Repeat/Simulation zone (V2.00 §8; Phase 4a). They
# are pure pass-throughs: their compute returns the input Dataset unchanged. The zone's
# iteration semantics — the back-edge feedback + the unroll into a per-iteration chain —
# live in :mod:`nodegraph.zones`; these nodes only mark the boundary the unroll wires
# through (``In@0`` = external seed / re-init at t₀; ``In@i>0`` = the prior ``Out``).

def _compute_zone_passthrough(ctx: EvalContext) -> Dataset:
    return ctx.inputs[0]


for _op, _label in (("zone.repeat_in", "Repeat In"), ("zone.repeat_out", "Repeat Out"),
                    ("zone.sim_in", "Sim In"), ("zone.sim_out", "Sim Out")):
    register_node(
        _compute_zone_passthrough, op_key=_op, label=_label, category="zone",
        inputs=[InDataset()], outputs=[OutDataset()],
        granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
        description="Zone boundary marker (pass-through); paired In/Out bound the "
                    "iterated body — see nodegraph.zones.")


# ── group boundary nodes (Group Input / Output — identity pass-through) ─────────
#
# The paired interface markers of a node group (V2.00; Phase 4b). Pure pass-throughs:
# the compute returns the input Dataset unchanged. The group's inline-expand semantics
# (replace an instance node with a fresh unique-id copy of the body + stitch the single
# DATASET interface) live in :mod:`nodegraph.groups`; these nodes only mark the
# interface input/output that ``expand()`` wires through.

for _op, _label in (("group.input", "Group Input"), ("group.output", "Group Output")):
    register_node(
        _compute_zone_passthrough, op_key=_op, label=_label, category="group",
        inputs=[InDataset()], outputs=[OutDataset()],
        granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
        description="Group interface marker (pass-through); paired Input/Output bound a "
                    "reusable subgraph — see nodegraph.groups.")


__all__ = [
    "COMPUTES", "register_node", "to_pixels_v2", "diffraction_sigmas", "gaussian_psf",
]
