"""The eleven attribute domains and the acquisition lattice (nodegraph v2).

Two families (see ``CodeLog/ClaudesPlan/V2.00_nodegraph_blender_revamp.md`` §3.2):

* **(a) Acquisition lattice** — coarsenings of the image hypercube over the axes
  ``{m,t,z,y,x}``. Each domain is the finer one with an axis-group aggregated
  away, forming a lattice closed under join (∪ of axes) and meet (∩ of axes)::

      Voxel (m,t,z,y,x) → Plane (m,t,z) → Frame (m,t) ─┬─ Multipoint (m) ─┐
                                                       └─ Timepoint (t) ──┴─ Global ()

  Because it is a lattice, the transfer between any two of these is **generated**
  (coarsen = reduce, refine = broadcast) rather than hand-written — see
  :mod:`nodegraph.transfer`. ``Channel (c)`` is a lattice axis too (V2.01 §H
  resolution: ``c`` is a first-class store/tile/memo axis), so ``Voxel`` is
  ``{m,t,z,c,y,x}`` and ``Channel = {c}`` is a generated lattice domain. Channel
  is **orthogonal** to the spatial/temporal chain, so ``meet`` with it is always
  a named domain but ``join`` across the ``c`` axis is not (e.g.
  ``join(Channel, Frame) = {m,t,c}`` is unnamed) — :func:`join` is therefore
  *partial*. Consequence: coarsening ``Voxel`` to a spatial/temporal domain
  reduces over ``c`` by default (channel-mean); for per-channel results transfer
  to ``Channel`` or select a channel at read time.

* **(b) Detected structures** — ``Label`` (a region of a label mask), ``Point``
  (sub-pixel detections), ``Track`` (a temporal identity running *over* the
  Timepoint domain), ``Mesh`` (a boundary surface: vertices + **faces**). These are
  defined by analysis, not by coarsening, so their transfers are explicit bridges.

  ``Mesh`` (V2.08) is the only domain that carries **topology**. Its irreducible
  content — which vertices form which face — is what no other domain encodes
  (vertices≈Point, the filled region≈Voxel Label, grouping≈Label-over-Points). It is
  stored as three flat, fixed-width **CSR strata** under one domain, addressed by the
  ``layer`` sub-key (``L`` / ``L/vert`` / ``L/face``); see :mod:`nodegraph.mesh`, which
  owns the only sanctioned read/write path.

Qt-free; pure standard library.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple

# Canonical ordering of the acquisition axes (V2.01 §H: ``c`` is a first-class
# store/tile/memo axis). A lattice domain's attribute array is shaped by the axes
# it *has*, in this order (see :mod:`nodegraph.dataset`).
AXIS_ORDER: Tuple[str, ...] = ("m", "t", "z", "c", "y", "x")


class Domain(Enum):
    """The eleven attribute domains. ``value`` is the stable serialization key."""

    # (a) acquisition lattice
    VOXEL = "voxel"
    PLANE = "plane"
    FRAME = "frame"
    TIMEPOINT = "timepoint"
    MULTIPOINT = "multipoint"
    GLOBAL = "global"
    # orthogonal axis-domain
    CHANNEL = "channel"
    # (b) detected structures
    LABEL = "label"
    POINT = "point"
    TRACK = "track"
    MESH = "mesh"


# ── presentation (theme-independent — the data model owns the domain identity) ──
# Colors + short chip labels for the domain **rail** the GUI draws on Dataset sockets
# and the domain-tint on wires. Mirrors ``sockets.SOCKET_COLOR``: a semantic property
# of the domain, not of any one theme (nodelab_v2.theme wraps these as QColors). The
# acquisition lattice runs green→blue (finer→coarser); the detected structures take
# the warm end (orange/red/magenta) so a structure wire is instantly distinct.
DOMAIN_COLOR: Dict[Domain, str] = {
    Domain.VOXEL: "#5fd06a",       # image / pixels (the source domain)
    Domain.PLANE: "#7bd3ad",       # per (m,t,z)
    Domain.FRAME: "#4bb8c0",       # per (m,t)
    Domain.TIMEPOINT: "#4a90d8",   # per t
    Domain.MULTIPOINT: "#6a7bd8",  # per m
    Domain.GLOBAL: "#8a93a1",      # one scalar
    Domain.CHANNEL: "#e8d44a",     # per c (orthogonal axis)
    Domain.LABEL: "#e08a3a",       # segmented regions
    Domain.POINT: "#e0555f",       # sub-pixel detections
    Domain.TRACK: "#c264a0",       # temporal identities
    # MESH closes the warm structure arc (orange → red → magenta → violet) and stays
    # clear of MULTIPOINT's #6a7bd8. This entry is MANDATORY in the same commit as the
    # enum member: ``nodelab_v2.theme`` builds its QColor map by iterating ``Domain``,
    # so a missing color used to KeyError at GUI import (now hardened there too).
    Domain.MESH: "#b06ad8",        # boundary surfaces (vertices + faces)
}

DOMAIN_ABBR: Dict[Domain, str] = {
    Domain.VOXEL: "VOX", Domain.PLANE: "PLN", Domain.FRAME: "FRM",
    Domain.TIMEPOINT: "TIM", Domain.MULTIPOINT: "MPT", Domain.GLOBAL: "GLB",
    Domain.CHANNEL: "CHN", Domain.LABEL: "LBL", Domain.POINT: "PT",
    Domain.TRACK: "TRK", Domain.MESH: "MSH",
}


def domain_color(domain: Domain) -> str:
    """The domain's rail/wire color (hex). Falls back to Global grey."""
    return DOMAIN_COLOR.get(domain, DOMAIN_COLOR[Domain.GLOBAL])


def domain_abbr(domain: Domain) -> str:
    """The domain's short chip label (≤3 chars)."""
    return DOMAIN_ABBR.get(domain, domain.value[:3].upper())


# Axis-set for each lattice domain (a subset of AXIS_ORDER). Domains absent from
# this map are non-lattice (the detected structures Label/Point/Track/Mesh) — the
# OMISSION is the declaration, which is why ``is_lattice``/``axes_of``/``_require_lattice``/
# ``shape_for``/``axis_list`` all give the right answer (or a clean raise) for free.
_LATTICE_AXES: Dict[Domain, FrozenSet[str]] = {
    Domain.VOXEL: frozenset({"m", "t", "z", "c", "y", "x"}),
    Domain.PLANE: frozenset({"m", "t", "z"}),
    Domain.FRAME: frozenset({"m", "t"}),
    Domain.TIMEPOINT: frozenset({"t"}),
    Domain.MULTIPOINT: frozenset({"m"}),
    Domain.CHANNEL: frozenset({"c"}),   # V2.01 §H: Channel is now a lattice axis-domain
    Domain.GLOBAL: frozenset(),
}

# Reverse lookup: axis-set → lattice domain. The named sets are closed under ∩
# (:func:`meet` is total) but NOT under ∪ once Channel (the orthogonal ``c`` axis)
# joins the lattice — e.g. ``{m,t} ∪ {c} = {m,t,c}`` is unnamed — so :func:`join`
# is partial (returns ``None`` for an unnamed union).
_AXES_TO_DOMAIN: Dict[FrozenSet[str], Domain] = {
    axes: dom for dom, axes in _LATTICE_AXES.items()
}

LATTICE_DOMAINS: FrozenSet[Domain] = frozenset(_LATTICE_AXES)
# Membership here is what unlocks the generic structure surfaces: ``is_structure``, the
# Spreadsheet panel's column gate, and CSV/Arrow export — all of which then need zero
# per-domain code for a new member.
STRUCTURE_DOMAINS: FrozenSet[Domain] = frozenset(
    {Domain.LABEL, Domain.POINT, Domain.TRACK, Domain.MESH}
)

# Domains that can carry more than one instance per dataset (keyed by a source
# layer: which mask / point set / tracking). The lattice domains are singletons.
MULTI_INSTANCE_DOMAINS: FrozenSet[Domain] = STRUCTURE_DOMAINS


# ── predicates ───────────────────────────────────────────────────────────────

def is_lattice(domain: Domain) -> bool:
    """True for the seven acquisition-lattice domains (transfers are generated)."""
    # NOTE: no MESH branch needed — it is absent from ``_LATTICE_AXES`` by design.
    return domain in _LATTICE_AXES


def is_structure(domain: Domain) -> bool:
    """True for a detected-structure domain (Label / Point / Track / Mesh)."""
    return domain in STRUCTURE_DOMAINS


def is_multi_instance(domain: Domain) -> bool:
    """True if a dataset may hold several instances of this domain (keyed by layer)."""
    return domain in MULTI_INSTANCE_DOMAINS


def axes_of(domain: Domain) -> Optional[FrozenSet[str]]:
    """The lattice axis-set for ``domain`` (``None`` for non-lattice domains)."""
    return _LATTICE_AXES.get(domain)


# ── lattice order ────────────────────────────────────────────────────────────

def _require_lattice(a: Domain, b: Domain) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    aa, ba = axes_of(a), axes_of(b)
    if aa is None or ba is None:
        raise ValueError(f"{a} / {b} are not both lattice domains")
    return aa, ba


def is_finer(a: Domain, b: Domain) -> bool:
    """True if lattice domain ``a`` is finer than or equal to ``b`` (has ⊇ axes)."""
    aa, ba = _require_lattice(a, b)
    return ba <= aa


def comparable(a: Domain, b: Domain) -> bool:
    """True if ``a`` and ``b`` are ordered (one is finer than the other)."""
    aa, ba = _require_lattice(a, b)
    return aa <= ba or ba <= aa


def join(a: Domain, b: Domain) -> Optional[Domain]:
    """Least common **refinement** — the lattice domain whose axes are ``a ∪ b``
    (e.g. ``join(Multipoint, Timepoint) == Frame``).

    **Partial:** returns ``None`` when the union is not a named domain, which
    happens for pairs straddling the orthogonal ``c`` axis (e.g.
    ``join(Channel, Frame)`` would be ``{m,t,c}``). The transfer generator does
    not need ``join`` (it reduces/​broadcasts axis-set differences directly), so
    this partiality is harmless there.
    """
    aa, ba = _require_lattice(a, b)
    return _AXES_TO_DOMAIN.get(frozenset(aa | ba))


def meet(a: Domain, b: Domain) -> Domain:
    """Greatest common **coarsening** — the lattice domain whose axes are
    ``a ∩ b`` (e.g. ``meet(Multipoint, Timepoint) == Global``)."""
    aa, ba = _require_lattice(a, b)
    return _AXES_TO_DOMAIN[frozenset(aa & ba)]


def dropped_axes(finer: Domain, coarser: Domain) -> FrozenSet[str]:
    """Axes present in ``finer`` but not ``coarser`` (the axes a reduce collapses
    / a broadcast expands). Both must be lattice domains."""
    fa, ca = _require_lattice(finer, coarser)
    return frozenset(fa - ca)


def domain_from_value(value: str) -> Domain:
    """Deserialize a domain from its ``.value`` key."""
    return Domain(value)


__all__ = [
    "AXIS_ORDER", "Domain",
    "DOMAIN_COLOR", "DOMAIN_ABBR", "domain_color", "domain_abbr",
    "LATTICE_DOMAINS", "STRUCTURE_DOMAINS", "MULTI_INSTANCE_DOMAINS",
    "is_lattice", "is_structure", "is_multi_instance", "axes_of",
    "is_finer", "comparable", "join", "meet", "dropped_axes", "domain_from_value",
]
