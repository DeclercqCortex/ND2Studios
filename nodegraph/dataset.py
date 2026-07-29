"""The ``Dataset`` payload + ``AttributeLayer`` store (nodegraph v2).

A :class:`Dataset` is the single thing that flows on the main wire. It is a lazy,
immutable-by-convention bundle: acquisition axis sizes, the metadata dict (drives
metadata-intelligent params — reused from V1.91), an optional image provider, and
an **attribute store** keyed by ``(domain, layer, name)``. Nodes return a *new*
Dataset that shares the upstream store and adds/replaces layers (structural
sharing — cheap to derive).

An :class:`AttributeLayer` is one named array on one domain. For a **lattice**
domain its ``values`` array is shaped by that domain's axes in
:data:`~nodegraph.domains.AXIS_ORDER` (e.g. Frame → ``(M, T)``; Voxel →
``(M,T,Z,Y,X)``; Global → scalar). For a **structure** domain (Label/Point/Track)
``values`` is element-indexed (id → value); its exact layout is refined in a later
phase — this phase nails the lattice.

Qt-free; numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nodegraph.domains import AXIS_ORDER, Domain, axes_of, is_lattice
from nodegraph.revision import next_revision


def _zkind_key(domain: Domain, layer: Optional[str]) -> str:
    """Flat, deterministic key for the ``__struct_zkind__`` provenance map — a structure
    layer is identified by its domain + source layer name (``\\x00``-joined so it never
    collides with a real layer name)."""
    return f"{domain.value}\x00{layer or ''}"


@dataclass(frozen=True)
class AxisSizes:
    """Sizes of the acquisition axes for one dataset (order per ``AXIS_ORDER``:
    ``m,t,z,c,y,x`` — ``c`` is a first-class store/tile/memo axis, V2.01 §H)."""

    m: int = 1
    t: int = 1
    z: int = 1
    c: int = 1
    y: int = 1
    x: int = 1

    def size(self, axis: str) -> int:
        return int(getattr(self, axis))

    @property
    def is_volumetric(self) -> bool:
        """True if the data has real depth (``z > 1``) — the metadata-driven
        default for the 2D/3D lever (V2.03 §3 B4). Distinct from ``z_collapsed``
        provenance (a z-projected stack is z==1 but was volumetric)."""
        return self.z > 1

    def shape_for(self, domain: Domain) -> Tuple[int, ...]:
        """Canonical array shape for a **lattice** domain's attribute (the sizes
        of the axes it has, in :data:`AXIS_ORDER`). Global → ``()`` (scalar)."""
        ax = axes_of(domain)
        if ax is None:
            raise ValueError(f"{domain} is not a lattice domain")
        return tuple(self.size(a) for a in AXIS_ORDER if a in ax)

    def axis_list(self, domain: Domain) -> Tuple[str, ...]:
        """The lattice domain's axes present, in canonical order."""
        ax = axes_of(domain)
        if ax is None:
            raise ValueError(f"{domain} is not a lattice domain")
        return tuple(a for a in AXIS_ORDER if a in ax)


# store key: (domain, layer-or-None, name)
LayerKey = Tuple[Domain, Optional[str], str]

# The calibration subset of ``Dataset.metadata`` (V2.03 §2 A1). Reuses the V1.91
# vocabulary VERBATIM so ``metadata_adapt`` stays "reused as-is" (V2.00 §13); it is
# a validated key SCHEMA, not a competing typed model. Per-channel optics beyond
# ``channel_emission_nm`` live as Channel-domain AttributeLayers (V2.00 §3.2), not
# here. Per-axis (y,x) pixel size + ``origin_um`` are deferred (V2.00 §16).
CALIBRATION_KEYS: frozenset = frozenset({
    "pixel_size_um", "z_step_um", "dt_s",
    "channel_emission_nm", "objective_na", "objective_magnification",
    # ``bit_depth`` (2026-07-28) is the one addition to the V1.91 vocabulary: the
    # SIGNIFICANT sensor depth (ND2 ``bitsPerComponentSignificant`` — typically 12), which
    # the pixel values alone cannot reveal, since a dim 12-bit frame may max out in the
    # low hundreds. It belongs here rather than in the Viewer's display dict because it is
    # an acquisition property nodes must READ (a raw-count threshold, a percentile LUT, a
    # normalize-to-full-range) — and going through the envelope makes those reads
    # memo-fenced like any other calibration. Invariant under every meta_transform
    # (cropping or resampling does not change the sensor), so identity pass-through is
    # correct and no transform needs to touch it.
    "bit_depth",
})


@dataclass(frozen=True, eq=False)
class AttributeLayer:
    """One named attribute on one domain.

    ``eq=False`` (identity equality/hash): the ``values`` ndarray field makes an
    auto-generated ``__eq__``/``__hash__`` raise (ambiguous truth value / unhashable
    ndarray). Layers are stored as dict *values* keyed by :data:`LayerKey`, so
    identity semantics suffice; content identity is the monotonic ``revision``
    (review #14).

    The array is **immutable**: ``__post_init__`` freezes ``values`` (read-only)
    after owning its buffer, so a layer can be structural-shared across derived
    Datasets without a downstream node corrupting an upstream producer's array.
    ``revision`` is the layer's **content identity** for memoization — a fresh
    monotonic integer per constructed layer (never ``id()``, never a content
    hash; see :mod:`nodegraph.revision` and V2.02 §4). Replacing a layer mints a
    new revision, so any memo key embedding the old revision auto-invalidates.
    """

    domain: Domain
    name: str
    values: np.ndarray
    layer: Optional[str] = None   # source layer for multi-instance domains
    revision: int = field(default_factory=next_revision)

    def __post_init__(self) -> None:
        # Own the buffer before freezing it (the V2.02 §J aliasing footgun). Copy if
        # the array is a **view** (``base is not None`` — even a *read-only* view still
        # reflects writes through its base, review #5) OR if it is a writeable array we
        # were handed by reference (``a is self.values`` — freezing it would freeze the
        # caller's array). An owned, non-view array is safe to freeze in place.
        a = np.asarray(self.values)
        if a.base is not None or (a.flags.writeable and a is self.values):
            a = a.copy()
        a.flags.writeable = False
        object.__setattr__(self, "values", a)

    @property
    def key(self) -> LayerKey:
        return (self.domain, self.layer, self.name)

    def mutate(self, new_values: np.ndarray) -> "AttributeLayer":
        """A sibling layer with new values and a fresh revision (same key)."""
        return replace(self, values=new_values, revision=next_revision())


@dataclass(frozen=True)
class Dataset:
    """The main-wire payload — axes + metadata + an attribute store."""

    axes: AxisSizes = field(default_factory=AxisSizes)
    metadata: Dict[str, Any] = field(default_factory=dict)
    image: Optional[Any] = None                    # lazy voxel provider (later phase)
    attributes: Dict[LayerKey, AttributeLayer] = field(default_factory=dict)

    # ── read ────────────────────────────────────────────────────────────────
    def get(self, domain: Domain, name: str,
            layer: Optional[str] = None) -> Optional[AttributeLayer]:
        return self.attributes.get((domain, layer, name))

    def has(self, domain: Domain, name: str, layer: Optional[str] = None) -> bool:
        return (domain, layer, name) in self.attributes

    def layers_on(self, domain: Domain) -> Tuple[AttributeLayer, ...]:
        return tuple(a for a in self.attributes.values() if a.domain is domain)

    # ── derive (structural sharing) ───────────────────────────────────────────
    def with_attribute(self, attr: AttributeLayer) -> "Dataset":
        """A new Dataset with ``attr`` added/replaced (upstream store shared)."""
        if is_lattice(attr.domain):
            expected = self.axes.shape_for(attr.domain)
            if tuple(attr.values.shape) != expected:
                raise ValueError(
                    f"{attr.domain.value} attribute {attr.name!r} has shape "
                    f"{tuple(attr.values.shape)}, expected {expected}")
        new = dict(self.attributes)
        new[attr.key] = attr
        return replace(self, attributes=new)

    def with_layer(self, domain: Domain, name: str, values: np.ndarray,
                   layer: Optional[str] = None) -> "Dataset":
        return self.with_attribute(AttributeLayer(domain, name, np.asarray(values), layer))

    def without(self, domain: Domain, name: str,
                layer: Optional[str] = None) -> "Dataset":
        new = dict(self.attributes)
        new.pop((domain, layer, name), None)
        return replace(self, attributes=new)

    # ── metadata / calibration (copy-on-write) ────────────────────────────────
    def with_metadata(self, **changes: Any) -> "Dataset":
        """A new Dataset whose metadata dict has ``changes`` applied — the **only**
        sanctioned calibration WRITE path (V2.03 §2 A1). Copies before ``replace``,
        so the parent's and every sibling's metadata is never mutated. The READ path
        stays the eval-time ``ReadContext`` (V2.02 §8). A geometry-changing node must
        call this in lockstep with :meth:`reshaped_axes` (V2.03 §2 A2).

        A ``changes`` value of ``None`` **removes** that key (e.g. z-project drops
        ``z_step_um``); other keys are added/replaced.
        """
        new = {**self.metadata}
        for k, v in changes.items():
            if v is None:
                new.pop(k, None)
            else:
                new[k] = v
        return replace(self, metadata=new)

    def with_image(self, provider: Any) -> "Dataset":
        """A new Dataset whose lazy image ``provider`` (a ``TileProvider``) is set;
        the reserved image slot (V2.02 §3). Copy-on-write via ``replace``."""
        return replace(self, image=provider)

    def with_structure(self, table: Any) -> "Dataset":
        """Store a structure ``table``'s columns as per-element :class:`AttributeLayer`\\ s
        on its domain, keyed by its source ``layer`` (V2.00 §3.2: Label/Point/Track are
        named attribute layers over their domain, multi-instance by source layer).
        Duck-typed on ``.domain`` / ``.columns`` / ``.layer`` so ``dataset`` need not
        import :mod:`nodegraph.structure`.

        **Preserves the table's ``z_kind``** (2D ``plane_index`` vs 3D ``subpixel``) into a
        namespaced ``__struct_zkind__`` metadata map (keyed by ``domain``+``layer``), which
        would otherwise be lost when the table explodes into per-column attribute layers.
        This is the generic form of the metadata-intelligence provenance directive
        (`wire-node-v2` §7b): a structure-producing node self-describes its dimensionality,
        so a downstream node can **inherit** it (see :meth:`structure_zkind`) instead of
        guessing from an independent 2D/3D lever that could silently disagree with the data.
        """
        ds = self
        layer = getattr(table, "layer", None)
        for name, values in table.columns.items():
            ds = ds.with_attribute(
                AttributeLayer(table.domain, name, np.asarray(values), layer))
        z_kind = getattr(table, "z_kind", None)
        if z_kind is not None:
            zmap = dict(self.metadata.get("__struct_zkind__", {}))
            zmap[_zkind_key(table.domain, layer)] = z_kind
            ds = ds.with_metadata(__struct_zkind__=zmap)
        return ds

    def structure_zkind(self, domain: "Domain", layer: Optional[str] = None
                        ) -> Optional[str]:
        """The stored ``z_kind`` of a structure ``layer`` (``"plane_index"`` = 2D per-plane,
        ``"subpixel"`` = 3D), or ``None`` if this Dataset carries no such provenance —
        preserved by :meth:`with_structure`. A consumer reads this to inherit the
        structure's dimensionality (`wire-node-v2` §7b). Reading it off an input payload's
        metadata is memo-safe: the marker rides the payload, so an upstream change bumps the
        upstream revision, which already folds into the consumer's recipe hash; and it is a
        non-calibration key, so ``strict_reads`` (C7) passes it through."""
        return self.metadata.get("__struct_zkind__", {}).get(_zkind_key(domain, layer))

    def reshaped_axes(self, new_axes: AxisSizes, *,
                      drop_stale: bool = True) -> "Dataset":
        """A new Dataset with ``new_axes``, resolving lattice layers orphaned by the
        axis change (V2.03 §2 A2): a lattice layer whose shape no longer matches
        ``new_axes.shape_for(domain)`` is **dropped** (``drop_stale=True``) or raises
        (``drop_stale=False``). Structure-domain layers pass through untouched. Does
        not alter calibration — pair with :meth:`with_metadata`.
        """
        kept: Dict[LayerKey, AttributeLayer] = {}
        for key, attr in self.attributes.items():
            if is_lattice(attr.domain):
                expected = new_axes.shape_for(attr.domain)
                if tuple(attr.values.shape) != expected:
                    if drop_stale:
                        continue
                    raise ValueError(
                        f"lattice layer {attr.key} shape "
                        f"{tuple(attr.values.shape)} != {expected} under new axes")
            kept[key] = attr
        return replace(self, axes=new_axes, attributes=kept)


__all__ = ["AxisSizes", "AttributeLayer", "Dataset", "LayerKey", "CALIBRATION_KEYS"]
