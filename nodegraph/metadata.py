"""Edit-time metadata propagation — the MetaEnvelope pass (nodegraph v2, V2.03 §2 A3).

Directive A makes a node's metadata intelligence come from **the data package on its
own input edge**, as transformed by upstream nodes — not from static file metadata.
But evaluation is lazy demand-driven pull (V2.00 §9): nothing computes until a Viewer
pulls, so at graph-edit time there is no materialized intermediate Dataset to read.

This module supplies the missing piece: a **cheap forward pass that propagates only
metadata** — ``AxisSizes`` + the calibration dict + the attribute-layer catalog —
through each node's declared :attr:`NodeSpec.meta_transform`, **touching no pixels**.
It runs on load and on every graph edit; its output feeds edit-time widget re-seeding
and the 2D/3D lever's default (``z>1 ⇒ 3D``).

It is **advisory** (V2.03 §1): the two-hash memo key stays driven by the *eval-time*
recording ``ReadContext`` (V2.02 §8), resolved upstream-first at pull time. Statically
unknowable sizes (e.g. a stitched Y,X extent) are marked UNKNOWN rather than guessed.

Qt-free; pure standard library + numpy-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence,
                    Tuple)

from nodegraph.dataset import AxisSizes, LayerKey
from nodegraph.domains import AXIS_ORDER, Domain, axes_of, is_lattice
from nodegraph.graph import Graph
from nodegraph.registry import layer_value


# ── the envelope ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetaEnvelope:
    """A node's resolved *metadata-only* output: axes, the calibration/metadata dict,
    the attribute-layer catalog, and the set of axes whose size is not statically
    knowable (e.g. a stitched extent — UNKNOWN, never a silent guess)."""

    axes: AxisSizes = field(default_factory=AxisSizes)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    layers: Tuple[LayerKey, ...] = ()
    unknown_axes: FrozenSet[str] = frozenset()
    #: The **accumulated domain-set** present on this node's Dataset output — the
    #: union of every upstream node's ``adds_domains`` (populated by
    #: :func:`propagate_meta`). Drives the GUI's socket domain-rail, the wire tint,
    #: and the domain-mismatch validation. Distinct from ``layers`` (the per-name
    #: attribute catalog, still a stub): this is the coarser domain granularity.
    domains: FrozenSet[Domain] = frozenset()
    #: The **layer catalog** (V2.11): the ``(domain, layer-name)`` pairs a Dataset on this
    #: edge is expected to carry, in first-appearance order. Drives the GUI's layer picker
    #: — a source-layer socket offers the layers actually present upstream instead of
    #: making the user retype a name.
    #:
    #: Distinct from ``layers`` above, which is a per-ATTRIBUTE ``LayerKey`` catalog and
    #: remains a stub. It could NOT be reused: ``LayerKey`` is ``(domain, layer, name)``,
    #: and the user-facing layer name sits in a *different* slot per domain family —
    #: ``with_layer`` leaves ``layer=None`` so a lattice layer is keyed
    #: ``(VOXEL, None, "mask")`` (name in slot 2), while ``with_structure`` stores each
    #: COLUMN separately as ``(POINT, "spots", "y")`` (layer in slot 1). Keying a picker on
    #: ``(domain, LayerKey[1])`` would collapse every Voxel layer in the graph to
    #: ``(VOXEL, None)`` and offer nothing. This field stores the user-facing name per
    #: domain directly, so the projection never arises.
    layer_names: Tuple[Tuple[Domain, str], ...] = ()

    @property
    def is_volumetric(self) -> bool:
        return self.axes.is_volumetric

    def with_domains(self, domains: FrozenSet[Domain]) -> "MetaEnvelope":
        return replace(self, domains=frozenset(domains))

    def with_layer_names(self, names: Sequence[Tuple[Domain, str]]) -> "MetaEnvelope":
        return replace(self, layer_names=tuple(names))

    def layers_in(self, domain: Domain) -> Tuple[str, ...]:
        """The layer names present on this edge for *domain*, in first-appearance
        order (the GUI picker's suggestion list)."""
        return tuple(n for d, n in self.layer_names if d is domain)

    def with_axes(self, axes: AxisSizes, *,
                  unknown: Optional[FrozenSet[str]] = None) -> "MetaEnvelope":
        return replace(self, axes=axes,
                       unknown_axes=self.unknown_axes if unknown is None else unknown)

    def with_metadata(self, **changes: Any) -> "MetaEnvelope":
        """Copy-on-write calibration edit; a ``None`` value removes the key
        (mirrors :meth:`nodegraph.dataset.Dataset.with_metadata`)."""
        new = {**self.metadata}
        for k, v in changes.items():
            if v is None:
                new.pop(k, None)
            else:
                new[k] = v
        return replace(self, metadata=new)


# ── named meta-transforms (V2.03 §2 A2) ───────────────────────────────────────
#
# Each maps an incoming envelope to the outgoing one given the node's params + its
# resolved mode state (incl. the 2D/3D ``dim`` lever). Axis-changing nodes MUST
# update calibration in lockstep (V2.03 §2 A2). ``params``/``modes`` are plain
# mappings; missing keys degrade to no-op (never a crash).

MetaTransform = Callable[["MetaEnvelope", Mapping[str, Any], Mapping[str, str]],
                         "MetaEnvelope"]


def identity(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    return env


# ── intensity provenance: calibration describes the CURRENT data, not the file ──
#
# The forward walk means every node reads its INPUT envelope, i.e. the most up-to-date
# metadata — so a node that changes what its numbers MEAN must restamp the affected key
# in lockstep, exactly like an axis-changing node restamps pixel size (V2.03 §2 A2).
# ``bit_depth`` is the intensity-domain instance of that rule: summing 16 12-bit frames
# yields 16-bit data, and normalizing to [0,1] yields data that is not integer counts at
# all. A downstream raw-count consumer that read the FILE's 12 bits in either case would
# be wrong. The two helpers below are the whole vocabulary; see `wire-node-v2` §7c.

def bit_depth_after_sum(env: MetaEnvelope, n: int) -> Dict[str, Any]:
    """``{"bit_depth": widened}`` for a reducer that SUMS ``n`` samples — ``b + ceil(log2
    n)`` bits, since n values of at most ``2**b - 1`` can total ``n*(2**b - 1)``. Empty
    when the incoming depth is unknown (nothing to widen) or ``n <= 1``. Mean/median/
    max/min/percentile reducers do NOT widen — they stay inside the input range."""
    b = env.metadata.get("bit_depth")
    if not b or n <= 1:
        return {}
    return {"bit_depth": int(b) + int(math.ceil(math.log2(int(n))))}


def value_rescaled(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """The meta_transform of a node whose output is no longer raw integer counts —
    percentile Normalize and CLAHE both return [0,1] floats. It DROPS ``bit_depth``:
    absent means "no declared integer scale", which is the honest signal for a
    raw-count consumer downstream (``analysis.histogram_threshold`` refuses such an
    input outright; ``analysis.threshold``'s fixed level falls back to its 0.5
    normalized-data default). Axis-preserving — only the intensity meaning changes."""
    return env.with_metadata(bit_depth=None)


def resample(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Rescale: new size = old·scale; pixel size scales inversely (finer when
    upsampling). ``z`` scales only in 3D mode (stack-of-2D leaves z untouched)."""
    sxy = float(params.get("scale_xy", params.get("scale", 1.0)) or 1.0)
    sz = float(params.get("scale_z", 1.0) or 1.0)
    ax = env.axes
    is_3d = modes.get("dim") == "3D"
    new_axes = replace(ax, y=max(1, round(ax.y * sxy)), x=max(1, round(ax.x * sxy)),
                       z=(max(1, round(ax.z * sz)) if is_3d else ax.z))
    changes: Dict[str, Any] = {}
    px = env.metadata.get("pixel_size_um")
    if px is not None and sxy:
        changes["pixel_size_um"] = px / sxy
    zs = env.metadata.get("z_step_um")
    if is_3d and zs is not None and sz:
        changes["z_step_um"] = zs / sz
    return env.with_axes(new_axes).with_metadata(**changes)


def z_project(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Collapse Z→1; drop ``z_step_um`` and mark ``z_collapsed`` provenance so a
    downstream lever defaults to 2D and no metric reads a meaningless z step. A ``sum``
    projection also widens ``bit_depth`` (n_z summed samples), per :func:`bit_depth_after_sum`."""
    widen = (bit_depth_after_sum(env, env.axes.z)
             if modes.get("method") == "sum" else {})
    return (env.with_axes(replace(env.axes, z=1))
               .with_metadata(z_step_um=None, z_collapsed=True, **widen))


def stack_time(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Temporal stack T→1; drop ``dt_s``. A ``sum`` combiner also widens ``bit_depth``
    (n_t summed samples) — the "12-bit in, 16-bit out" case."""
    widen = (bit_depth_after_sum(env, env.axes.t)
             if modes.get("method") == "sum" else {})
    return env.with_axes(replace(env.axes, t=1)).with_metadata(dt_s=None, **widen)


def frame_slice(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Per-frame-T slice (``zone.frame``): select one timepoint → T=1. The frame index
    picks *which* frame (a value, not geometry), so axes just collapse to t=1; ``dt_s``
    is kept (it still describes the source series' interval)."""
    return env.with_axes(replace(env.axes, t=1))


def parse_channels(raw) -> Optional[List[int]]:
    """The ``channels`` param → a list of channel indices, or ``None`` for "all".

    Shared by ``channel.select``'s compute and :func:`channel_select` below, which MUST
    agree: the meta_transform predicts the axes at edit time and the compute produces
    them at pull time, and a node whose payload disagrees with its envelope fails the
    build-node-v2 §2 gate. Accepts three forms because the param has three producers:
    a **list of ints** written by the GUI's per-channel tap materializer
    (``nodelab_v2/ops.py``), a **comma-separated string** typed into the socket by a user
    (``SocketType`` has no LIST member), and **empty/None** meaning every channel.
    Non-integer text is ignored rather than raising, so a half-typed "0," keeps the node
    previewing instead of erroring on each keystroke."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, str):
        out: List[int] = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except ValueError:
                continue
        return out or None
    items = list(raw)
    return items or None


def channel_select(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Subset/reindex channels; rewrite the per-channel emission list in lockstep.

    Out-of-range / negative indices are dropped FIRST, so the channel count and the
    emission list always agree (review #13: len(keep) could otherwise claim more
    channels than the source has)."""
    keep = parse_channels(params.get("channels"))
    if not keep:
        return env
    valid = [i for i in keep if 0 <= i < env.axes.c]     # lockstep count ↔ metadata
    new_axes = replace(env.axes, c=len(valid))
    changes: Dict[str, Any] = {}
    emis = env.metadata.get("channel_emission_nm")
    if isinstance(emis, (list, tuple)):
        changes["channel_emission_nm"] = [emis[i] for i in valid if i < len(emis)]
    return env.with_axes(new_axes).with_metadata(**changes)


def crop(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Change spatial extent; pixel size preserved (origin deferred, V2.00 §16).

    ``span`` mirrors the crop node's payload ``bound()`` EXACTLY (fill each missing
    endpoint independently — start→0, end→n — and clamp both into ``[0, n]``), so the
    predicted (header) extent equals the realized payload extent for one-sided and
    out-of-range crops alike (adversarial review 2026-07-21). A ``max(1, …)`` floor
    keeps the advisory transform crash-free where the payload would raise on an empty
    region."""
    ax = env.axes

    def span(a, b, n):
        lo = max(0, min(int(a), n)) if a is not None else 0
        hi = max(0, min(int(b), n)) if b is not None else n
        return max(1, hi - lo)

    new_axes = replace(
        ax, y=span(params.get("y0"), params.get("y1"), ax.y),
        x=span(params.get("x0"), params.get("x1"), ax.x),
        z=(span(params.get("z0"), params.get("z1"), ax.z)
           if modes.get("dim") == "3D" else ax.z))
    return env.with_axes(new_axes)


def stitch(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Tile stitch: M→1, Y/X grow. The output extent is UNKNOWN unless supplied
    (it depends on estimated registration) — never a silent guess (V2.03 §2 A3)."""
    ax = env.axes
    ny, nx = params.get("out_y"), params.get("out_x")
    unknown = set()
    if not ny:
        unknown.add("y")
    if not nx:
        unknown.add("x")
    new_axes = replace(ax, m=1, y=int(ny) if ny else ax.y, x=int(nx) if nx else ax.x)
    return env.with_axes(new_axes, unknown=frozenset(unknown))


META_TRANSFORMS: Dict[str, MetaTransform] = {
    "identity": identity, "resample": resample, "z_project": z_project,
    "stack_time": stack_time, "frame_slice": frame_slice,
    "channel_select": channel_select, "crop": crop, "stitch": stitch,
    "value_rescaled": value_rescaled,
}


def named_meta_transform(name: str) -> Optional[MetaTransform]:
    return META_TRANSFORMS.get(name)


# ── the forward pass ──────────────────────────────────────────────────────────

def propagate_meta(graph: Graph,
                   seeds: Optional[Mapping[str, MetaEnvelope]] = None
                   ) -> Dict[str, MetaEnvelope]:
    """Compute every node's output :class:`MetaEnvelope` by a forward topological
    walk (V2.03 §2 A3). A node's input envelope is its first DATASET-input
    predecessor's output; a root uses ``seeds[node_id]`` (a source declaring its
    file/provider metadata) or an empty envelope. Each node's declared
    ``meta_transform`` (identity by default) produces its output. Pixel-free."""
    seeds = seeds or {}
    out: Dict[str, MetaEnvelope] = {}
    for nid in graph.topo_order():
        node = graph.nodes[nid]
        spec = node.spec()
        dpreds = graph.dataset_preds(nid)
        env_in = out.get(dpreds[0].src, MetaEnvelope()) if dpreds \
            else seeds.get(nid, MetaEnvelope())
        transform = spec.meta_transform if spec is not None else None
        env_out = env_in if transform is None else transform(
            env_in, node.params, node.state(spec))
        # Domain accumulation: union EVERY Dataset predecessor's domain-set (a merge
        # node combines them), then add what this node produces. A root seeds from its
        # own envelope's domain-set. The meta_transform never touches domains, so this
        # is layered on afterward. (V2.06: the socket domain-rail + wire-tint source.)
        if dpreds:
            dom_in: FrozenSet[Domain] = frozenset().union(
                *(out.get(e.src, MetaEnvelope()).domains for e in dpreds))
        else:
            dom_in = env_in.domains
        adds = spec.adds_domains if spec is not None else frozenset()
        env_out = env_out.with_domains(dom_in | adds)
        out[nid] = env_out.with_layer_names(
            _layer_names_out(spec, node, env_in, env_out))
    return out


def _layer_names_out(spec, node, env_in: MetaEnvelope,
                     env_out: MetaEnvelope) -> Tuple[Tuple[Domain, str], ...]:
    """This node's outgoing layer catalog: the input's, minus what its axis change
    invalidates, plus what it writes (V2.11).

    **Total by contract** — this runs inside ``propagate_meta``, which the GUI calls on
    every keystroke, and whose caller catches only ``ValueError``: anything else escapes
    and takes the window down, while even a caught error blanks EVERY node's envelope
    (domain rails and derived spinboxes graph-wide). So every step is defensive, exactly
    like the meta_transforms above ("missing keys degrade to no-op, never a crash")."""
    names: List[Tuple[Domain, str]] = list(env_in.layer_names)

    # ── DROP: the catalog is NOT monotone ────────────────────────────────────────
    # `Dataset.reshaped_axes(drop_stale=True)` silently discards any LATTICE layer whose
    # array no longer matches the new axes, and five nodes rely on it (channel.select,
    # util.zproject, util.crop, util.resample, util.stack). Its rule is exactly "the
    # shape for this domain changed", and a domain's shape is built from `axes_of` — so
    # comparing the envelope's own before/after axes reproduces it for the whole catalog
    # with no per-node declaration. Structure domains are never reshaped, so they survive.
    try:
        changed = {ax for ax in AXIS_ORDER
                   if getattr(env_in.axes, ax, None) != getattr(env_out.axes, ax, None)}
    except Exception:                                    # pragma: no cover - defensive
        changed = set()
    if changed:
        names = [(d, n) for d, n in names
                 if not (is_lattice(d) and (axes_of(d) & changed))]

    # ── ADD: what this node creates ──────────────────────────────────────────────
    if spec is None:
        return tuple(dict.fromkeys(names))
    try:
        state = node.state(spec)
    except Exception:                                    # pragma: no cover - defensive
        state = {}
    params = getattr(node, "params", {}) or {}
    for sock in spec.inputs:
        if not sock.layer_out:
            continue
        try:
            if not sock.active_in(state):
                continue
            # ONE resolver, shared with `EvalContext.layer` — the compute and this
            # prediction must never disagree about which layer a socket denotes.
            value = layer_value(sock, params)
            if not value:
                continue
            for dom in sock.layer_out:
                names.append((dom, value))
        except Exception:                                # pragma: no cover - defensive
            continue
    extra = getattr(spec, "extra_layers", None)
    if extra is not None:
        try:
            for dom, nm in extra(params, state) or ():
                if isinstance(nm, str) and nm:
                    names.append((dom, nm))
        except Exception:                                # pragma: no cover - defensive
            pass
    return tuple(dict.fromkeys(names))                   # de-dup, keep first appearance


# ── derive-symbol source + the metadata-intelligent lever default ─────────────
#
# The SOURCE layer of directive A (V2.03 §2 A5): the symbols a ``derive`` may read,
# sourced from the resolved incoming envelope (per-edge) — NOT from static file
# metadata. Mirrors the V1.91 leaf contract without importing the nd2-coupled
# ``metadata_adapt`` (nodegraph stays nd2-free); the app/ctx layer feeds these
# symbols to ``metadata_adapt.adapt_defaults`` where the pipeline_kit path needs it.

_SAFE_FUNCS: Dict[str, Any] = {
    "min": min, "max": max, "abs": abs, "round": round,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "pi": math.pi,
}


def envelope_symbols(env: MetaEnvelope, channel_index: int = 0) -> Dict[str, Any]:
    """Derive symbols from an envelope (axes → counts; calibration → optics). Optics
    are for ``channel_index`` (the node's active channel — V2.03 §2 A6)."""
    md = env.metadata or {}
    emis = md.get("channel_emission_nm")
    emission = (emis[channel_index]
                if isinstance(emis, (list, tuple)) and 0 <= channel_index < len(emis)
                else (emis if not isinstance(emis, (list, tuple)) else None))
    return {
        "pixel_size_um": md.get("pixel_size_um"),
        "z_step_um": md.get("z_step_um"),
        "dt_s": md.get("dt_s"),
        "bit_depth": md.get("bit_depth"),      # significant sensor depth (12 on most ND2s)
        "emission_nm": emission,
        "na": md.get("objective_na"),
        "mag": md.get("objective_magnification"),
        "n_m": env.axes.m, "n_t": env.axes.t, "n_z": env.axes.z, "n_c": env.axes.c,
        "z_collapsed": bool(md.get("z_collapsed", False)),
        "is_3d": env.axes.is_volumetric,
    }


def eval_derive(expr: str, symbols: Mapping[str, Any]) -> Any:
    """Evaluate a trusted, pure-arithmetic ``derive`` expression (empty builtins).
    Mirrors the V1.91 ``metadata_adapt`` leaf contract (V2.03 §2 A5)."""
    ns = dict(_SAFE_FUNCS)
    ns.update(symbols)
    return eval(expr, {"__builtins__": {}}, ns)  # noqa: S307 — trusted, no builtins


def resolve_dim_default(spec: Any, env: MetaEnvelope) -> Optional[str]:
    """The metadata-intelligent 2D/3D lever default for ``spec`` given the incoming
    envelope (V2.03 §3 B4): evaluate the lever's ``derive`` (``z>1 ⇒ 3D``) against
    the envelope symbols; fall back to the static default on any failure. ``None``
    if the node bears no lever."""
    lever = spec.dim_lever() if hasattr(spec, "dim_lever") else None
    if lever is None:
        return None
    if not lever.derive:
        return lever.resolved_default()
    try:
        val = eval_derive(lever.derive, envelope_symbols(env))
    except Exception:  # noqa: BLE001 — a bad expression degrades to the static default
        return lever.resolved_default()
    return str(val) if val in ("2D", "3D") else lever.resolved_default()


__all__ = [
    "MetaEnvelope", "MetaTransform", "META_TRANSFORMS", "named_meta_transform",
    "identity", "resample", "z_project", "stack_time", "frame_slice",
    "channel_select", "crop", "stitch", "value_rescaled", "bit_depth_after_sum",
    "propagate_meta", "envelope_symbols",
    "eval_derive", "resolve_dim_default",
]
