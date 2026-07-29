"""Socket types, implicit value conversion, and connection validity (nodegraph v2).

One **data** socket type (``DATASET`` — the thick main wire) plus the field-able
**value** types (``FLOAT/INT/BOOL/VECTOR/COLOR/STRING/MENU``). A value socket may
carry a single value *or* a field; a data socket carries a ``Dataset``.

Connection rule (:func:`can_connect`): an output feeds an input; a ``DATASET``
pairs only with ``DATASET``; two value sockets connect on an exact type match or an
available **implicit conversion** (numeric/vector widening — never a data-layer
transform, which stays an explicit node). Multi-input sockets accept many wires
(the scene enforces cardinality; the type rule is unchanged).

Qt-free; pure standard library.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, FrozenSet, Optional, Tuple

from nodegraph.domains import Domain


class SocketType(Enum):
    DATASET = "dataset"   # the main payload wire
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    VECTOR = "vector"
    COLOR = "color"
    STRING = "string"
    MENU = "menu"


VALUE_TYPES: FrozenSet[SocketType] = frozenset(
    {SocketType.FLOAT, SocketType.INT, SocketType.BOOL, SocketType.VECTOR,
     SocketType.COLOR, SocketType.STRING, SocketType.MENU}
)

# Blender-convention socket colors (used by the canvas later).
SOCKET_COLOR = {
    SocketType.DATASET: "#5fd06a",
    SocketType.FLOAT: "#a1a1a1",
    SocketType.INT: "#82c98a",
    SocketType.BOOL: "#e480c0",
    SocketType.VECTOR: "#6a7bd8",
    SocketType.COLOR: "#e8d44a",
    SocketType.STRING: "#9b6ad8",
    SocketType.MENU: "#4bb8c0",
}

# Directed implicit conversions (values only). Widen numeric, broadcast to
# vector, grayscale to color. Data-layer transforms are NEVER here.
_CONVERSIONS: FrozenSet[Tuple[SocketType, SocketType]] = frozenset({
    (SocketType.BOOL, SocketType.INT),
    (SocketType.BOOL, SocketType.FLOAT),
    (SocketType.INT, SocketType.FLOAT),
    (SocketType.INT, SocketType.BOOL),
    (SocketType.FLOAT, SocketType.INT),
    (SocketType.FLOAT, SocketType.VECTOR),
    (SocketType.INT, SocketType.VECTOR),
    (SocketType.FLOAT, SocketType.COLOR),
    (SocketType.VECTOR, SocketType.COLOR),
})


def can_convert(src: SocketType, dst: SocketType) -> bool:
    """True if an implicit conversion ``src → dst`` exists (identity included)."""
    return src is dst or (src, dst) in _CONVERSIONS


class Direction(Enum):
    IN = "in"
    OUT = "out"


@dataclass(frozen=True)
class Socket:
    """A concrete socket on a node instance."""

    name: str
    type: SocketType
    direction: Direction
    is_field: bool = False        # value sockets: may carry a field (diamond)
    multi: bool = False           # input accepts many wires (in order)
    unit: str = ""                # physical unit (metadata-intelligent params)
    derive: str = ""              # metadata-derived default expression
    default: Any = None
    dims: int = 3                 # VECTOR arity (2 or 3); ignored for other types
    domain: Optional[Domain] = None   # for a field input: the domain it evaluates on

    @property
    def color(self) -> str:
        return SOCKET_COLOR[self.type]


def can_connect(src: Socket, dst: Socket) -> bool:
    """True if a wire from output ``src`` to input ``dst`` is legal.

    Requires ``src`` an output and ``dst`` an input; ``DATASET`` pairs only with
    ``DATASET``; two value sockets connect on exact type or an implicit
    conversion. Field-ness never blocks a value connection (an input value socket
    accepts a value or a field).

    **Vector arity (V2.03 §4 C1):** a VECTOR→VECTOR wire connects on equal ``dims``
    or **widens** (``src.dims < dst.dims`` — the axial component is padded with 0, a
    value-level conversion); **narrowing** (``src.dims > dst.dims``) is rejected and
    needs an explicit drop/swizzle node. ``Float/Int→Vector`` broadcast fills the
    destination's ``dims`` (the destination arity is known at wire time).
    """
    if src.direction is not Direction.OUT or dst.direction is not Direction.IN:
        return False
    if SocketType.DATASET in (src.type, dst.type):
        return src.type is SocketType.DATASET and dst.type is SocketType.DATASET
    if src.type is SocketType.VECTOR and dst.type is SocketType.VECTOR:
        return src.dims <= dst.dims          # equal or widen; never narrow
    return can_convert(src.type, dst.type)


__all__ = [
    "SocketType", "VALUE_TYPES", "SOCKET_COLOR", "Direction", "Socket",
    "can_convert", "can_connect",
]
