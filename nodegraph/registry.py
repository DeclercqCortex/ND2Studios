"""Node definition API + the unified registry (nodegraph v2).

A node **type** is a :class:`NodeSpec`: an ``op_key``, a label/category, its input
and output **socket specs**, its in-body **modes** (dropdowns that may reconfigure
sockets), and its data-access declarations. Value inputs carry the metadata-
intelligence fields (``unit``/``derive``) reused from V1.91.

**V2.03 directive additions** (per the 2026-07-21 metadata/2D-3D-toggle directive):

* **Variant sockets** — ``SocketSpec.available_in`` tags a socket as present only in
  certain mode states; ``NodeSpec.active_sockets(state)`` resolves the live socket
  set. This is the declarative backing that lets a mode/toggle reconfigure a node's
  sockets (V2.03 §3 B1), replacing the docstring-only promise.
* **The 2D/3D lever** — a distinguished in-body :class:`ModeSpec` with
  ``role="dim_lever"`` + ``presentation="header"`` and an optional metadata ``derive``
  default; it is a Mode in substance (not a socket, hashed as an in-body option), so
  memo/serialization are unchanged (V2.03 §3 B2).
* **Data-access footprint** — ``granularity`` and ``kernel_axes`` may be static or a
  ``{dim_value: value}`` map resolved per mode state (V2.03 §3 B3), plus the
  true-3D-vs-stack capability flags (V2.03 §3 B3 / H15).
* **Metadata propagation** — ``meta_transform`` declares a node's calibration/axes
  effect for the edit-time MetaEnvelope pass (V2.03 §2 A2; see :mod:`nodegraph.metadata`).

Qt-free; pure standard library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union,
)

from nodegraph.domains import Domain
from nodegraph.sockets import Direction, Socket, SocketType


# ── data-access footprint (V2.02 §7b / V2.03 §3 B3) ──────────────────────────

class Granularity(Enum):
    """Which axes a node must consume *whole* vs per-element — gates tiling, halo,
    and memo granularity in the scheduler (V2.02 §7b). For a 2D/3D-toggle node this
    is resolved per mode state (2D → TILEABLE/WHOLE_PLANE, 3D → WHOLE_VOLUME)."""

    TILEABLE = "tileable"          # pointwise / small-stencil 2D — honors the tile provider
    WHOLE_PLANE = "whole_plane"    # a full (Y,X) plane per (m,t,z,c)
    WHOLE_VOLUME = "whole_volume"  # a full (Z,Y,X) volume per (m,t,c)
    WHOLE_SERIES = "whole_series"  # the full T series per (m,c)
    MULTI_VIEW = "multi_view"      # several M (tile stitching / multi-view fusion)


#: The reserved mode name for the 2D/3D header lever (V2.03 §3 B2).
DIM_MODE = "dim"


# ── socket / mode specs ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class SocketSpec:
    """Blueprint for one socket on a node type.

    ``available_in`` (V2.03 §3 B1): ``{mode_name: {allowed values}}``. The socket is
    active only when the node's current state matches every entry; ``None`` = always
    active. e.g. ``{"dim": frozenset({"3D"})}`` marks a ``sigma_z`` socket 3D-only.
    """

    name: str
    type: SocketType
    direction: Direction
    label: str = ""
    is_field: bool = False
    multi: bool = False
    unit: str = ""
    derive: str = ""
    default: Any = None
    domain: Optional[Domain] = None   # for a field input: the domain it evaluates on
    dims: int = 3                     # VECTOR arity (2 or 3)
    available_in: Optional[Mapping[str, FrozenSet[str]]] = None
    #: ── layer-name sockets (V2.11) ────────────────────────────────────────────
    #: A STRING socket carrying an attribute-LAYER NAME rather than free text.
    #: ``layer_in`` = it SELECTS a layer that must already exist on the incoming
    #: Dataset, in this domain → the GUI offers the layers actually present.
    #: ``layer_in_mode`` names a Mode whose *value* is the domain instead (only
    #: ``transform.transfer_domain``, whose source domain is its ``from_domain`` lever).
    #: ``layer_out`` = it NAMES a layer this node CREATES, in each listed domain —
    #: several nodes write one name into two domains (``analysis.label`` emits both a
    #: Voxel raster and a Label table called ``labels``), which is why it is a tuple.
    #: These drive :func:`nodegraph.metadata.propagate_meta`'s edit-time layer catalog.
    layer_in: Optional[Domain] = None
    layer_in_mode: str = ""
    layer_out: Tuple[Domain, ...] = ()
    kernel_param: bool = False        # influences the spatial kernel (radius/σ): a
    #: NON-Const Field on this socket is spatially varying and breaks tile translation-
    #: invariance + halo sizing, so a consumer must stream at the plane unit, not tiled
    #: (V2.04 §6b Fork B — the kernel-param field gate).

    def instantiate(self) -> Socket:
        return Socket(
            name=self.name, type=self.type, direction=self.direction,
            is_field=self.is_field, multi=self.multi, unit=self.unit,
            derive=self.derive, default=self.default, dims=self.dims,
            domain=self.domain,
        )

    def active_in(self, state: Mapping[str, str]) -> bool:
        """True if this socket is present in mode ``state`` (V2.03 §3 B1)."""
        if not self.available_in:
            return True
        return all(state.get(m) in allowed for m, allowed in self.available_in.items())


@dataclass(frozen=True)
class ModeSpec:
    """An in-body enum dropdown (not a socket; may reconfigure sockets).

    ``presentation`` ("body" default | "header") controls rendering — the 2D/3D lever
    uses "header" (top-right). ``role`` ("dim_lever" for the toggle) lets the engine/
    GUI find it. ``derive`` is a metadata-intelligent default expression (e.g. the
    lever: ``"'3D' if n_z>1 else '2D'"``) — the toggle default is metadata-adaptive
    exactly like a value-socket default (V2.03 §1 / H9).

    ``available_in`` (V2.12) gates one Mode on ANOTHER Mode's value, exactly as
    :attr:`SocketSpec.available_in` gates a socket: ``{mode_name: {allowed values}}``,
    ``None`` = always shown. It exists because a node that unifies several algorithms
    behind one ``method`` Mode can have a *second* Mode only some methods read —
    ``analysis.segment``'s ``level`` (the foreground cut) means nothing to its learned
    detectors. Without gating that dropdown would sit there doing nothing, which is the
    same "live-looking control the selected kernel ignores" the node charter forbids for
    sockets. Gating is **edit-time only**: the resolved mode state still carries every
    mode (a hidden one keeps its value) and still folds into the recipe hash, so hiding a
    Mode never changes a memo key — the same rule as a hidden socket.
    """

    name: str
    choices: Sequence[str]
    default: str = ""
    label: str = ""
    presentation: str = "body"
    role: str = ""
    derive: str = ""
    available_in: Optional[Mapping[str, FrozenSet[str]]] = None

    def resolved_default(self) -> str:
        return self.default or (self.choices[0] if self.choices else "")

    @property
    def is_dim_lever(self) -> bool:
        return self.role == "dim_lever"

    def active_in(self, state: Mapping[str, str]) -> bool:
        """True if this Mode is shown in mode ``state`` (V2.12) — mirrors
        :meth:`SocketSpec.active_in`."""
        if not self.available_in:
            return True
        return all(state.get(m) in allowed for m, allowed in self.available_in.items())


@dataclass(frozen=True)
class NodeSpec:
    op_key: str
    label: str
    category: str = "general"
    inputs: Sequence[SocketSpec] = field(default_factory=tuple)
    outputs: Sequence[SocketSpec] = field(default_factory=tuple)
    modes: Sequence[ModeSpec] = field(default_factory=tuple)
    description: str = ""
    # V2.03 data-access + propagation declarations
    granularity: Union[Granularity, Mapping[str, Granularity], None] = None
    kernel_axes: Union[FrozenSet[str], Mapping[str, FrozenSet[str]], None] = None
    meta_transform: Optional[Callable[..., Any]] = None
    supports_2d: bool = True
    supports_true_3d: bool = True
    three_d_fallback: str = ""        # e.g. "stack_of_2d" when supports_true_3d is False
    # Domain interface (the GUI's socket domain-rail + wire tint; edit-time domain
    # propagation in metadata.propagate_meta). ``reads_domains`` = the domains this
    # node's compute REQUIRES present in the incoming Dataset (a missing one is a
    # validation error); ``adds_domains`` = the domains it PRODUCES/adds to the bundle
    # (unioned into the accumulated set that flows downstream). Both default empty —
    # an un-annotated node is domain-transparent (passes the upstream set through).
    reads_domains: FrozenSet[Domain] = frozenset()
    adds_domains: FrozenSet[Domain] = frozenset()
    #: Layers this node creates that no ``layer_out`` socket can describe (V2.11):
    #: ``(params, modes) -> ((Domain, name), ...)``. Needed by the handful of producers
    #: that name a layer with NO socket at all (``align.drift``/``registration.stabilize``
    #: write the literals ``drift_y``/``drift_x``), derive the name from ANOTHER param
    #: (``analysis.extract_boundary`` -> ``f"{labels}_boundary"``), or write into the layer
    #: their READ socket names (``analysis.measure`` adds Label columns to the raster it
    #: measures). MUST be total — see ``propagate_meta``, which runs on every keystroke.
    extra_layers: Optional[Callable[..., Any]] = None

    def input(self, name: str) -> Optional[SocketSpec]:
        return next((s for s in self.inputs if s.name == name), None)

    def output(self, name: str) -> Optional[SocketSpec]:
        return next((s for s in self.outputs if s.name == name), None)

    # ── mode state ────────────────────────────────────────────────────────────
    def default_state(self) -> Dict[str, str]:
        """The mode state with every mode at its resolved default."""
        return {m.name: m.resolved_default() for m in self.modes}

    def dim_lever(self) -> Optional[ModeSpec]:
        """The 2D/3D lever mode, if this node bears one (V2.03 §3 B2)."""
        return next((m for m in self.modes if m.is_dim_lever), None)

    def has_dim_lever(self) -> bool:
        return self.dim_lever() is not None

    # ── variant resolution (V2.03 §3 B1) ──────────────────────────────────────
    def active_inputs(self, state: Mapping[str, str]) -> tuple:
        return tuple(s for s in self.inputs if s.active_in(state))

    def active_outputs(self, state: Mapping[str, str]) -> tuple:
        return tuple(s for s in self.outputs if s.active_in(state))

    def active_sockets(self, state: Mapping[str, str]) -> tuple:
        return self.active_inputs(state) + self.active_outputs(state)

    def active_modes(self, state: Mapping[str, str]) -> tuple:
        """The Modes shown in ``state`` (V2.12). ``default_state`` deliberately still
        includes the hidden ones: a gated-away Mode keeps its value, so the compute's
        ``__modes__`` lookup and the memo key are unaffected by what the GUI draws."""
        return tuple(m for m in self.modes if m.active_in(state))

    # ── footprint resolution (V2.03 §3 B3) ────────────────────────────────────
    def resolve_granularity(self, state: Mapping[str, str]) -> Optional[Granularity]:
        g = self.granularity
        if isinstance(g, Mapping):
            return g.get(state.get(DIM_MODE, ""))
        return g

    def resolve_kernel_axes(self, state: Mapping[str, str]) -> Optional[FrozenSet[str]]:
        k = self.kernel_axes
        if isinstance(k, Mapping):
            return k.get(state.get(DIM_MODE, ""))
        return k

    # ── domain interface ──────────────────────────────────────────────────────
    def out_domains(self, incoming: FrozenSet[Domain]) -> FrozenSet[Domain]:
        """The accumulated domain-set on this node's Dataset output given the set
        ``incoming`` on its Dataset input(s): the upstream set unioned with what this
        node adds (domain-transparent by default)."""
        return incoming | self.adds_domains

    def missing_domains(self, incoming: FrozenSet[Domain]) -> FrozenSet[Domain]:
        """Required domains not present upstream — the GUI's red validation chips."""
        return self.reads_domains - incoming


# ── socket / mode factories (the §11 sketch) ─────────────────────────────────

def InDataset(name: str = "data", *, multi: bool = False, label: str = "",
              available_in: Optional[Mapping[str, FrozenSet[str]]] = None) -> SocketSpec:
    return SocketSpec(name, SocketType.DATASET, Direction.IN, label=label,
                      multi=multi, available_in=available_in)


def OutDataset(name: str = "out", *, label: str = "",
               available_in: Optional[Mapping[str, FrozenSet[str]]] = None) -> SocketSpec:
    return SocketSpec(name, SocketType.DATASET, Direction.OUT, label=label,
                      available_in=available_in)


def layer_value(sock: Optional[SocketSpec], params: Mapping[str, Any]) -> str:
    """Resolve a layer-name socket to the name it denotes: the user's override, else the
    socket's declared **default**. The empty string counts as unset.

    THE single resolution path, and the reason it lives here rather than in either caller:
    a layer name is resolved in two places that must never disagree — the compute at pull
    time (via ``EvalContext.layer``) and :func:`nodegraph.metadata.propagate_meta` at edit
    time, which predicts the layer catalog the GUI picker offers. They used to be
    independent, with each compute repeating its socket's default inline
    (``ctx.params.get("mask", "mask")``) because the engine passes params as raw
    OVERRIDES and never default-fills them. That made every default a *third* copy —
    socket, compute, envelope rule — with nothing keeping the three in step. Routing both
    through this function leaves exactly one copy: the ``SocketSpec`` default."""
    if sock is None:
        return ""
    value = params.get(sock.name)
    if value is None or value == "":
        value = sock.default
    return value if isinstance(value, str) else ""


def _in_value(t: SocketType):
    def make(name: str, label: str = "", *, default: Any = None, field: bool = True,
             unit: str = "", derive: str = "", domain: Optional[Domain] = None,
             multi: bool = False, dims: int = 3,
             available_in: Optional[Mapping[str, FrozenSet[str]]] = None,
             layer_in: Optional[Domain] = None, layer_in_mode: str = "",
             layer_out: Tuple[Domain, ...] = (),
             kernel_param: bool = False) -> SocketSpec:
        return SocketSpec(name, t, Direction.IN, label=label, is_field=field,
                          unit=unit, derive=derive, default=default, domain=domain,
                          multi=multi, dims=dims, available_in=available_in,
                          layer_in=layer_in, layer_in_mode=layer_in_mode,
                          layer_out=tuple(layer_out), kernel_param=kernel_param)
    return make


InFloat = _in_value(SocketType.FLOAT)
InInt = _in_value(SocketType.INT)
InBool = _in_value(SocketType.BOOL)
InVector = _in_value(SocketType.VECTOR)
InColor = _in_value(SocketType.COLOR)
InString = _in_value(SocketType.STRING)


def OutValue(name: str, t: SocketType, label: str = "", *, field: bool = True,
             dims: int = 3) -> SocketSpec:
    return SocketSpec(name, t, Direction.OUT, label=label, is_field=field, dims=dims)


def Mode(name: str, choices: Sequence[str], default: str = "", label: str = "",
         *, presentation: str = "body", role: str = "", derive: str = "",
         available_in: Optional[Mapping[str, FrozenSet[str]]] = None) -> ModeSpec:
    return ModeSpec(name, tuple(choices), default, label,
                    presentation=presentation, role=role, derive=derive,
                    available_in=available_in)


def DimMode(*, default: str = "2D",
            derive: str = "'3D' if (n_z or 1) > 1 else '2D'") -> ModeSpec:
    """The 2D/3D header lever (V2.03 §3 B2): an in-body Mode with header rendering,
    ``role="dim_lever"``, and a metadata-adaptive default (z>1 ⇒ 3D)."""
    return Mode(DIM_MODE, ["2D", "3D"], default=default, label="2D / 3D",
                presentation="header", role="dim_lever", derive=derive)


# ── the registry ─────────────────────────────────────────────────────────────

class NodeRegistry:
    """Insertion-ordered ``{op_key: NodeSpec}`` (palette order for un-sorted kinds)."""

    def __init__(self) -> None:
        self._by_key: Dict[str, NodeSpec] = {}

    def register(self, spec: NodeSpec) -> NodeSpec:
        self._by_key[spec.op_key] = spec
        return spec

    def get(self, op_key: str) -> Optional[NodeSpec]:
        return self._by_key.get(op_key)

    def all(self) -> List[NodeSpec]:
        return list(self._by_key.values())

    def __contains__(self, op_key: str) -> bool:
        return op_key in self._by_key


NODES = NodeRegistry()


def define_node(op_key: str, label: str, *, category: str = "general",
                inputs: Sequence[SocketSpec] = (), outputs: Sequence[SocketSpec] = (),
                modes: Sequence[ModeSpec] = (), description: str = "",
                granularity: Union[Granularity, Mapping[str, Granularity], None] = None,
                kernel_axes: Union[FrozenSet[str], Mapping[str, FrozenSet[str]], None] = None,
                meta_transform: Optional[Callable[..., Any]] = None,
                supports_2d: bool = True, supports_true_3d: bool = True,
                three_d_fallback: str = "",
                reads_domains: FrozenSet[Domain] = frozenset(),
                adds_domains: FrozenSet[Domain] = frozenset(),
                extra_layers: Optional[Callable[..., Any]] = None) -> NodeSpec:
    """Build and register a :class:`NodeSpec`."""
    return NODES.register(NodeSpec(
        op_key=op_key, label=label, category=category,
        inputs=tuple(inputs), outputs=tuple(outputs), modes=tuple(modes),
        description=description, granularity=granularity, kernel_axes=kernel_axes,
        meta_transform=meta_transform, supports_2d=supports_2d,
        supports_true_3d=supports_true_3d, three_d_fallback=three_d_fallback,
        reads_domains=frozenset(reads_domains), adds_domains=frozenset(adds_domains),
        extra_layers=extra_layers,
    ))


__all__ = [
    "SocketSpec", "ModeSpec", "NodeSpec", "NodeRegistry", "NODES",
    "Granularity", "DIM_MODE",
    "InDataset", "OutDataset", "OutValue", "Mode", "DimMode",
    "InFloat", "InInt", "InBool", "InVector", "InColor", "InString",
    "define_node", "layer_value",
]
