"""Two-hash memo + revision fence (nodegraph v2, Phase 2a — V2.02 §6 / V2.03 H4).

The memo separates two hashes with different jobs (V2.02 §I.a, the restated invariant):

* **recipe_hash** — the *lookup* key, computed **before** compute from structure only
  (op + params + upstream recipe_hashes + upstream **revisions**). A cheap PROXY: it
  never reads pixels (you cannot hash an 85 MB plane just to look it up). Declared
  calibration reads (V2.02 §8) are stored on the entry and **re-validated on a hit**
  (Salsa-style verify), so a metadata change a node actually read invalidates it, while
  identity/propagation rides the monotonic **revision** (never ``id()``).
* **output_fingerprint** — a *content* hash computed **after** compute (per-tile for
  images, full for KB–MB structure tables), used for **cutoff** (a recompute that
  yields identical bytes doesn't dirty downstream) and **dedup** (identical outputs
  share one blob).

Hashing uses ``hashlib.blake2b`` (stdlib) rather than blake3 — the proxy hash is over
small metadata, so no third-party hash dependency is pulled into the core.

Qt-free; numpy + stdlib only.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import AxisSizes, Dataset, LayerKey
from nodegraph.revision import next_revision


# ── canonical hashing ─────────────────────────────────────────────────────────

def _canon(o: Any) -> str:
    """A deterministic, **injective** string encoding (order-stable for dicts/sets).

    Every token is self-delimiting — scalars end in ``;`` and strings/containers are
    **length/count-prefixed** — so a payload can never impersonate a structural
    delimiter. This closes the collision where a string value containing ``,``/``=``/
    ``{`` forged extra dict/list entries and produced a wrong memo hit (review #1)."""
    if o is None:
        return "N;"
    if isinstance(o, bool):
        return "b1;" if o else "b0;"
    if isinstance(o, Enum):
        s = f"{type(o).__name__}.{o.name}"
        return f"e{len(s)}:{s}"
    if isinstance(o, (int, np.integer)):
        s = str(int(o))
        return f"i{len(s)}:{s}"
    if isinstance(o, (float, np.floating)):
        s = repr(float(o))
        return f"f{len(s)}:{s}"
    if isinstance(o, str):
        return f"s{len(o)}:{o}"                 # length tells us where it ends
    if isinstance(o, bytes):
        h = o.hex()
        return f"y{len(h)}:{h}"
    if isinstance(o, np.ndarray):
        # An OBJECT-dtype array's ``tobytes()`` is its ``PyObject*`` POINTER bytes, not its
        # content, so hashing one is doubly wrong: identical content hashes differently (a
        # permanent memo miss — the node recomputes forever), and distinct content can
        # COLLIDE once CPython reuses a freed pointer slot (a wrong payload served on a
        # false hit). Refuse loudly instead. Callers with ragged data must flatten to a
        # fixed-width CSR layout (see :mod:`nodegraph.mesh` for the worked example).
        if o.dtype == object:
            raise TypeError(
                "cannot content-hash an object-dtype ndarray (tobytes() would hash "
                "pointers, not content) — flatten ragged data to fixed-width arrays "
                "plus offsets, as nodegraph.mesh does")
        d = (f"{o.dtype}|{o.shape}|"
             f"{hashlib.blake2b(np.ascontiguousarray(o).tobytes(), digest_size=8).hexdigest()}")
        return f"a{len(d)}:{d}"
    if isinstance(o, (tuple, list)):
        tag = "t" if isinstance(o, tuple) else "l"
        return f"{tag}{len(o)};" + "".join(_canon(x) for x in o)
    if isinstance(o, (set, frozenset)):
        parts = sorted(_canon(x) for x in o)
        return f"q{len(parts)};" + "".join(parts)
    if isinstance(o, dict):
        pairs = sorted((_canon(k), _canon(v)) for k, v in o.items())
        return f"d{len(pairs)};" + "".join(k + v for k, v in pairs)
    s = repr(o)
    return f"r{len(s)}:{s}"


def digest(*parts: Any) -> str:
    """blake2b hex digest over the canonical encoding of ``parts``."""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(_canon(p).encode("utf-8"))
        h.update(b"\x1e")                       # record separator
    return h.hexdigest()


def value_digest(v: Any) -> str:
    """Digest of a single (calibration) value — for ReadContext read-tracking."""
    return digest("v", v)


# ── recipe (lookup) hashes ─────────────────────────────────────────────────────

def leaf_recipe_hash(file_id: Any, byte_offset: int, mtime_ns: int, dtype: Any,
                     shape: Sequence[int], plane_index: Sequence[int],
                     reader_params: Any) -> str:
    """Proxy hash of a raw leaf read — structure only, no decode (V2.02 §6)."""
    return digest("leaf", file_id, byte_offset, mtime_ns, str(dtype),
                  tuple(shape), tuple(plane_index), reader_params)


def node_recipe_hash(op_key: str, params: Any,
                     upstream_recipe_hashes: Sequence[str],
                     upstream_revisions: Sequence[int]) -> str:
    """Structural lookup key: op + params (incl. the 2D/3D lever + modes, folded in
    by the engine) + upstream recipe hashes + upstream **revisions**. Declared reads
    are validated separately on a hit (V2.02 §8), not baked into the lookup key."""
    return digest("node", op_key, params,
                  tuple(upstream_recipe_hashes), tuple(upstream_revisions))


def full_recipe_hash(recipe_hash: str,
                     declared_reads: Sequence[Tuple[str, str]]) -> str:
    """The §6 identity incl. declared reads — recorded for reference/dedup; the
    engine keys the table by the structural ``recipe_hash`` and validates reads."""
    return digest("full", recipe_hash, tuple(sorted(declared_reads)))


# ── output fingerprint (content) ────────────────────────────────────────────────

def output_fingerprint(payload: Any) -> str:
    """Content hash for cutoff/dedup (per-tile for images, full for structure tables)."""
    if isinstance(payload, np.ndarray):
        h = hashlib.blake2b(digest_size=16)
        h.update(str(payload.dtype).encode())
        h.update(str(payload.shape).encode())
        h.update(np.ascontiguousarray(payload).tobytes())
        return h.hexdigest()
    if isinstance(payload, Dataset):
        # pair each key with ITS revision in one sorted pass (review #7 — the two
        # tuples were independently ordered, misaligning key↔revision). Sort by a
        # STRING key: LayerKey's first element is a Domain enum, which is unorderable
        # across different domains (a Dataset can hold Voxel + Label attributes).
        keyed = tuple((k, payload.attributes[k].revision)
                      for k in sorted(payload.attributes,
                                      key=lambda kk: (kk[0].value, kk[1] or "", kk[2])))
        # include the IMAGE provider's fingerprint — else two datasets that differ only
        # by their image collide and blob-dedup returns the wrong image (catalog bug).
        img = payload.image
        img_fp = (img.fingerprint() if img is not None and hasattr(img, "fingerprint")
                  else ("none" if img is None else type(img).__name__))
        return digest("ds", payload.axes, keyed, payload.metadata, img_fp)
    return digest("fp", payload)


# ── retained-byte estimate (Memo GC sizing) ───────────────────────────────────────

def payload_bytes(payload: Any) -> int:
    """Estimate the **realized, freeable** in-memory bytes a memo entry retains — the
    size the Memo GC (:meth:`Memo._evict_to_budget`) frees by dropping this payload.

    Only bytes an eviction actually reclaims are counted: a :class:`Dataset`'s in-memory
    image array (an ``ArrayProvider`` exposes ``nbytes``; a *lazy* compute / synthetic /
    disk provider holds no realized raster, so it counts as 0 — evicting it frees nothing)
    plus its attribute-layer arrays (the eager full-raster hazard — V2.04 §6b). Attribute
    layers are structurally shared across derived Datasets, so summing them per-entry can
    over-count a shared layer; that biases the GC toward evicting *slightly more* (always
    correctness-safe — an eviction only costs a recompute), never toward an OOM."""
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    if isinstance(payload, Dataset):
        total = 0
        img = payload.image
        if img is not None:
            total += int(getattr(img, "nbytes", 0) or 0)
        for a in payload.attributes.values():
            v = getattr(a, "values", None)
            if isinstance(v, np.ndarray):
                total += int(v.nbytes)
        return total
    return 0


# ── memo table ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OutputHeader:
    """The output's structural summary, cached so a memo-HIT chain can resolve
    downstream envelopes/digests without recompute (V2.03 H4 amendment 5)."""

    axes: Optional[AxisSizes] = None
    metadata_digest: str = ""
    layers: Tuple[LayerKey, ...] = ()
    layer_revisions: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Entry:
    recipe_hash: str                 # structural lookup key
    full_hash: str                   # recipe_hash + declared reads (reference)
    fingerprint: str                 # content output-fingerprint
    revision: int                    # monotonic identity (never id())
    payload: Any
    reads: Tuple[Tuple[str, str], ...] = ()   # (calibration key, value digest)
    header: Optional[OutputHeader] = None
    changed: bool = True             # fingerprint differs from this node's previous


class Memo:
    """recipe_hash → :class:`Entry`, with fingerprint-keyed blob dedup and per-node
    last-fingerprint tracking for the Salsa cutoff.

    **Memo GC (V2.04 §6b follow-up).** With ``budget_bytes`` set, the table is a
    byte-budget **LRU** over entries: when the retained realized bytes (see
    :func:`payload_bytes`) exceed the budget, least-recently-used entries are evicted
    until it fits. This bounds the memory hazard V2.04 flagged — an eager full-raster
    attribute node (threshold/label) inside a high-T unrolled zone otherwise pins one
    full raster per iteration in the persistent GUI memo forever.

    Eviction is **always correctness-safe**: dropping an entry only causes a later
    recompute — computes are deterministic (identical output fingerprint) and identity
    rides the monotonic ``revision``, never ``id()``. So no pinning is needed; under
    genuine memory pressure a recompute is the honest trade. ``budget_bytes=None``
    (the default) keeps the table **unbounded** — byte-identical to the pre-GC behavior.

    Byte accounting is keyed by the **unique blob**, not per entry: identical outputs
    from different recipe hashes share one deduped blob, so its bytes are counted once
    and freed only when the *last* referencing entry is evicted (an ``fp → refcount``
    keeps the dedup invariant the memo already promised). ``_last_fp`` (the tiny Salsa
    cutoff signal — one string per node) is **never** GC'd, so an evict-then-recompute
    that yields identical bytes still reports ``changed=False``."""

    def __init__(self, budget_bytes: Optional[int] = None) -> None:
        self.budget = None if budget_bytes is None else int(budget_bytes)
        self._entries: "OrderedDict[str, Entry]" = OrderedDict()  # LRU: front=oldest
        self._blobs: Dict[str, Any] = {}         # fingerprint -> payload (dedup)
        self._blob_bytes: Dict[str, int] = {}    # fingerprint -> retained bytes
        self._fp_refs: Dict[str, int] = {}       # fingerprint -> live entry count
        self._last_fp: Dict[str, str] = {}       # node_key -> last fingerprint
        self.nbytes = 0                          # sum of live blob bytes
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, recipe_hash: str) -> Optional[Entry]:
        e = self._entries.get(recipe_hash)
        if e is not None:
            if self.budget is not None:
                self._entries.move_to_end(recipe_hash)   # mark most-recently-used
            self.hits += 1
        else:
            self.misses += 1
        return e

    def put(self, recipe_hash: str, payload: Any, *, node_key: str,
            reads: Sequence[Tuple[str, str]] = (),
            header: Optional[OutputHeader] = None) -> Entry:
        fp = output_fingerprint(payload)
        changed = self._last_fp.get(node_key) != fp     # Salsa cutoff signal
        self._last_fp[node_key] = fp
        # Freeze a raw ndarray payload so an impure downstream compute cannot mutate a
        # cached (and possibly dedup-aliased) output in place (review #8). Datasets/
        # AttributeLayers are already immutable via copy-then-freeze.
        if isinstance(payload, np.ndarray) and payload.flags.writeable:
            try:
                payload.flags.writeable = False
            except ValueError:                           # non-owning view — own then freeze
                payload = np.array(payload)
                payload.flags.writeable = False
        # Release a prior entry at this recipe_hash BEFORE rebinding, so a re-put (same
        # rh, changed content) decrements the old blob's refcount/bytes exactly once.
        old = self._entries.pop(recipe_hash, None)
        if old is not None:
            self._release_fp(old.fingerprint)
        blob = self._blobs.get(fp)                        # dedup identical outputs
        if blob is None:
            blob = payload
            self._blobs[fp] = blob
            self._blob_bytes[fp] = payload_bytes(blob)
            self._fp_refs[fp] = 0
            self.nbytes += self._blob_bytes[fp]
        self._fp_refs[fp] += 1
        entry = Entry(
            recipe_hash=recipe_hash,
            full_hash=full_recipe_hash(recipe_hash, tuple(reads)),
            fingerprint=fp, revision=next_revision(), payload=blob,
            reads=tuple(reads), header=header, changed=changed,
        )
        self._entries[recipe_hash] = entry               # appended → most-recently-used
        self._evict_to_budget()
        return entry

    # ── byte-budget LRU (Memo GC) ─────────────────────────────────────────────
    def _release_fp(self, fp: str) -> None:
        """Drop one reference to blob ``fp``; free the blob + its bytes at the last ref."""
        r = self._fp_refs.get(fp, 0) - 1
        if r <= 0:
            self._blobs.pop(fp, None)
            self._fp_refs.pop(fp, None)
            self.nbytes -= self._blob_bytes.pop(fp, 0)
        else:
            self._fp_refs[fp] = r

    def _evict_to_budget(self) -> None:
        """Evict least-recently-used entries until the retained bytes fit the budget.
        The entry just put is at the MRU end, so ``popitem(last=False)`` (oldest) never
        touches it. Terminates: each iteration removes one entry (``len`` strictly
        decreases), and it stops at the last entry — an oversize sole payload is served
        but not shrunk below one entry (mirrors :class:`~nodegraph.streaming.TileCache`)."""
        if self.budget is None:
            return
        while self.nbytes > self.budget and len(self._entries) > 1:
            _, ent = self._entries.popitem(last=False)
            self._release_fp(ent.fingerprint)
            self.evictions += 1

    def invalidate(self, recipe_hash: str) -> None:
        ent = self._entries.pop(recipe_hash, None)
        if ent is not None:
            self._release_fp(ent.fingerprint)

    def clear(self) -> None:
        self._entries.clear()
        self._blobs.clear()
        self._blob_bytes.clear()
        self._fp_refs.clear()
        self._last_fp.clear()
        self.nbytes = 0
        self.hits = self.misses = self.evictions = 0


__all__ = [
    "digest", "value_digest", "leaf_recipe_hash", "node_recipe_hash",
    "full_recipe_hash", "output_fingerprint", "payload_bytes",
    "OutputHeader", "Entry", "Memo",
]
