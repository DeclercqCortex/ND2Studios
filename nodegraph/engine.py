"""Lazy pull scheduler + ReadContext + Granularity routing (nodegraph v2, Phase 2a).

The engine realizes V2.00 §9 "pull, don't push": a consumer :meth:`Engine.pull`\\ s a
node; the engine walks **backward**, computing only what that request needs, and
memoizes every node output through the two-hash :mod:`nodegraph.memo`.

Three mechanisms wire together here (V2.02 §6/§8 + V2.03):

* **Two-hash memo** — the lookup key is the structural ``recipe_hash`` (op + params +
  upstream recipe_hashes + upstream **revisions**); a hit is **re-validated** against
  the node's declared calibration reads (below), and the content ``output_fingerprint``
  drives cutoff/dedup.
* **ReadContext** (V2.02 §8) — nodes read calibration only through a recording context;
  each ``(key, value_digest)`` is stored on the memo entry and re-checked on a hit, so a
  metadata change a node **actually read** invalidates exactly that node — and a change
  it did **not** read does not.
* **Granularity routing** (V2.03 §3 B3 / H8) — the node's resolved ``Granularity`` (a
  function of its 2D/3D lever + modes) selects the provider read path: TILEABLE/
  WHOLE_PLANE → single-z tile/region; WHOLE_VOLUME → ``get_subvolume`` (a z-range brick).
  The lever value folds into ``recipe_hash`` via params, so 2D and 3D memoize distinctly.

Compute is supplied per ``op_key`` (the node port is a later phase); the engine is the
harness those node compute functions run inside. Qt-free; numpy + stdlib.
"""
from __future__ import annotations

import time
from collections.abc import Mapping as _ABCMapping
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple

from nodegraph.dataset import CALIBRATION_KEYS, Dataset
from nodegraph.field import FieldCache
from nodegraph.graph import Graph
from nodegraph.memo import (
    Entry, Memo, OutputHeader, digest, node_recipe_hash, value_digest,
)
from nodegraph.metadata import MetaEnvelope, envelope_symbols, eval_derive, propagate_meta
from nodegraph.registry import Granularity
from nodegraph.streaming import TileCache, recursion_headroom


# ── ReadContext (V2.02 §8) ────────────────────────────────────────────────────

class ReadContext:
    """Records every metadata read as ``(key, value_digest)`` so the set folds into
    the memo entry and is re-validated on a hit. ``calib`` validates the key against
    :data:`~nodegraph.dataset.CALIBRATION_KEYS` (a typo is a hard error, not a silent
    ``None`` — V2.03 §2 A1); ``meta`` reads any (un-modeled) key."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self._md = metadata
        self.reads: Dict[str, str] = {}
        self._frozen = False

    def record(self, key: str, value: Any) -> None:
        """Record one read. Raises once frozen — a ``ctx.calib``/``ctx.meta`` call from
        inside a lazy streaming closure (at tile-pull time) would silently escape the
        memo fence (the entry's reads snapshot is already frozen), so it is a hard
        error: resolve every calibration value BEFORE building the provider (C1 /
        V2.04 §6b)."""
        if self._frozen:
            raise RuntimeError(
                f"late metadata read {key!r} after compute returned — a lazy closure "
                f"must not call ctx.calib/ctx.meta at tile-pull time; resolve values "
                f"before constructing the streaming provider (V2.04 §6b)")
        self.reads[key] = value_digest(value)

    def freeze(self) -> None:
        self._frozen = True

    def meta(self, key: str, default: Any = None) -> Any:
        v = self._md.get(key, default)
        self.record(key, v)
        return v

    def calib(self, key: str, default: Any = None) -> Any:
        if key not in CALIBRATION_KEYS:
            raise KeyError(f"{key!r} is not a calibration key {sorted(CALIBRATION_KEYS)}")
        return self.meta(key, default)

    # convenience typed accessors (V2.03 §2 A1)
    def pixel_size_um(self, default: Any = None) -> Any:
        return self.calib("pixel_size_um", default)

    def z_step_um(self, default: Any = None) -> Any:
        return self.calib("z_step_um", default)

    def dt_s(self, default: Any = None) -> Any:
        return self.calib("dt_s", default)

    def declared_reads(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted(self.reads.items()))


class _StrictCalibMetadata(dict):
    """Debug guard (V2.02 §8 / C7): a ``Dataset.metadata`` dict that makes an
    un-contexted **calibration** read a hard error. A node must read calibration
    through ``ctx.calib`` (recorded → memo-fenced); reaching into an input
    ``Dataset.metadata`` for a calibration key bypasses the fence and risks a silent
    stale hit, so in strict mode it raises. Non-calibration keys pass through.

    It subclasses ``dict``, so ``{**d}`` / ``dict(d)`` / iteration / ``.items()`` use
    the C fast path and do NOT trip — ``with_metadata`` (``{**self.metadata}``),
    ``output_fingerprint`` (``_canon`` over ``.items()``) and ``dataclasses.replace``
    are unaffected; only an explicit ``md[key]`` / ``md.get(key)`` for a calibration
    key trips."""

    def __getitem__(self, key: str) -> Any:
        if key in CALIBRATION_KEYS:
            raise RuntimeError(
                f"un-contexted calibration read {key!r} on Dataset.metadata — read it "
                f"via ctx.calib({key!r}) so the memo can fence it (V2.02 §8 / C7)")
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in CALIBRATION_KEYS:
            raise RuntimeError(
                f"un-contexted calibration read {key!r} on Dataset.metadata — use "
                f"ctx.calib({key!r}) (V2.02 §8 / C7)")
        return super().get(key, default)


class _RecordingMetadata(_ABCMapping):
    """A metadata view that records every read into a :class:`ReadContext`, so a
    compute that reaches into ``ctx.env.metadata`` directly (instead of ``ctx.calib``/
    ``ctx.meta``) is still fenced by the memo (review #2 / V2.02 §8 debug proxy)."""

    def __init__(self, base: Mapping[str, Any], rc: ReadContext) -> None:
        self._base = base
        self._rc = rc

    def __getitem__(self, key: str) -> Any:
        v = self._base[key]
        self._rc.record(key, v)
        return v

    def get(self, key: str, default: Any = None) -> Any:
        v = self._base.get(key, default)
        self._rc.record(key, v)
        return v

    def __iter__(self):
        return iter(self._base)

    def __len__(self) -> int:
        return len(self._base)


# ── run observation (per-node progress) ───────────────────────────────────────

#: the events an :class:`Engine` observer receives, as ``(event, node_id, info)``:
#:
#: * ``"start"``  — the node missed the memo and its compute is about to run;
#:   ``info = {"op_key": …, "depth": …}``.
#: * ``"cached"`` — a validated memo hit; nothing was computed (``info = {"op_key": …}``).
#: * ``"progress"`` — a *fractional* update from inside a compute, emitted by
#:   :meth:`EvalContext.progress`; ``info = {"done": i, "total": n, "fraction": f,
#:   "note": str}``. Only eager, per-unit computes report this — a node that returns a
#:   **lazy provider** finishes in microseconds and does its real work later, when a
#:   consumer reads planes (so its bar is indeterminate by nature, not by omission).
#: * ``"done"``   — the compute returned; ``info = {"op_key": …, "seconds": …}``.
#: * ``"error"``  — the compute raised; ``info = {"op_key": …, "seconds": …,
#:   "error": repr}``. The exception still propagates.
#:
#: The observer is called on whatever thread drives the pull (a GUI runner marshals it
#: onto the UI thread) and must never raise — the engine swallows observer exceptions
#: so a broken progress sink can't fail a run.
Observer = Callable[[str, str, Dict[str, Any]], None]


# ── the compute context handed to a node's compute fn ─────────────────────────

@dataclass
class EvalContext:
    node_id: str
    op_key: str
    params: Mapping[str, Any]
    env: MetaEnvelope                       # resolved incoming metadata (advisory)
    granularity: Optional[Granularity]      # resolved from the node's mode state
    kernel_axes: Optional[frozenset]
    inputs: Tuple[Any, ...]                 # upstream payloads (already pulled)
    reads: ReadContext
    provider: Any = None                    # a TileProvider, for source nodes
    by_name: Optional[Mapping[str, Any]] = None   # dst_socket → wired payload (C1)
    tiles: Any = None                       # the engine's shared TileCache (C1)
    fields: Any = None                      # the engine's FieldCache (C1)
    spec: Any = None                        # the node's NodeSpec (C8 per-channel derive)
    observer: Optional[Observer] = None     # run-progress sink (per-node progress)

    def progress(self, done: int, total: int, note: str = "") -> None:
        """Report *fractional* progress out of an eager per-unit compute (frame/plane/
        volume loops): ``ctx.progress(i + 1, n)``. A no-op when nothing is observing, so
        it is safe to call from any compute at any rate — the sink (not the compute)
        decides how often to repaint.

        Only meaningful for computes that do their work **inside** the call. A compute
        returning a lazy provider should not fake a fraction: leaving it silent is what
        tells the UI the node's cost is deferred to read time."""
        if self.observer is None:
            return
        total = int(total)
        done = max(0, min(int(done), total))
        frac = (done / total) if total > 0 else 0.0
        try:
            self.observer("progress", self.node_id,
                          {"done": done, "total": total, "fraction": frac,
                           "note": str(note), "op_key": self.op_key})
        except Exception:  # noqa: BLE001 — a progress sink must never break a run
            pass

    def input(self, name: str, default: Any = None) -> Any:
        """The payload wired into socket ``name`` (or ``default`` if unwired) — the
        positional ``inputs`` tuple has gaps for unwired sockets, so a value/field
        payload on a named socket is only reachable by name (C1 / V2.04 §6b)."""
        if self.by_name is None:
            return default
        return self.by_name.get(name, default)

    # read passthroughs (all recorded)
    def calib(self, key: str, default: Any = None) -> Any:
        return self.reads.calib(key, default)

    def meta(self, key: str, default: Any = None) -> Any:
        return self.reads.meta(key, default)

    @property
    def is_volume(self) -> bool:
        """True when the resolved footprint is volumetric (3D-mode WHOLE_VOLUME)."""
        return self.granularity is Granularity.WHOLE_VOLUME

    def socket(self, name: str) -> Any:
        """The input :class:`SocketSpec` named ``name`` (or ``None``) — from the node's
        spec; the source of a value socket's ``derive``/``default`` (C8)."""
        if self.spec is None:
            return None
        for s in getattr(self.spec, "inputs", ()):
            if getattr(s, "name", None) == name:
                return s
        return None

    def layer(self, name: str) -> str:
        """The attribute-layer name a ``layer_in``/``layer_out`` socket denotes: the user's
        override, else the socket's declared default (V2.11).

        **Use this instead of ``ctx.params.get("mask", "mask")``.** The inline form
        repeats the socket's default inside the compute, which makes it a second copy that
        can silently drift from the declaration — and a third, since
        :func:`nodegraph.metadata.propagate_meta` must resolve the same name at edit time
        to predict the layer catalog the GUI picker offers. Both routes call
        :func:`nodegraph.registry.layer_value`, so the default exists in exactly one place:
        the ``SocketSpec``. A missing socket resolves to ``""`` rather than raising, so a
        hand-built graph degrades the way the old ``.get`` did."""
        from nodegraph.registry import layer_value
        return layer_value(self.socket(name), self.params)

    def channel(self, c: int) -> "ChannelContext":
        """A per-channel derive resolver (C8 / H12) for a **c-iterating** node — one whose
        params derive from ``emission_nm`` (per-channel emission λ), e.g. Deconvolve's PSF
        or Spot Detection's radii. ``ctx.channel(c).param(name)`` returns ``name`` resolved
        *for channel c*: a user override if the param was set (a single value applies to all
        channels — there is one widget), else the socket's ``derive`` re-evaluated with
        channel ``c``'s optics, else the socket default. See :class:`ChannelContext`."""
        return ChannelContext(self, int(c))


class ChannelContext:
    """Per-channel view over an :class:`EvalContext` that resolves metadata-intelligent
    params for one channel *at evaluate time* (C8 / H12). Built by :meth:`EvalContext.channel`.

    The v2 convention (`wire-node-v2` §7): a param is **auto/derived** iff it is *absent*
    from ``ctx.params`` (the GUI stores a value only when the user edits/pins it). So:

    - param **set** by the user → that single value (a user override; one widget ⇒ it
      applies to every channel).
    - param **unset** with a ``derive`` → the ``derive`` re-evaluated against
      ``envelope_symbols(ctx.env, channel_index=c)`` — i.e. **channel c's** ``emission_nm``.
      Reads are memo-fenced automatically: ``ctx.env.metadata`` is the engine's recording
      wrapper, so the optics symbols (a small, safe superset of what the expression uses)
      are recorded and the node re-validates on them (C7-safe; never serves a stale hit).
    - param **unset** without a ``derive`` → the socket default.

    Resolve values **before** returning a lazy provider — like every other ``ctx`` read,
    a call after the compute returns trips the frozen-ReadContext guard (V2.04 §6b)."""

    __slots__ = ("_ctx", "_c")

    def __init__(self, ctx: EvalContext, channel: int) -> None:
        self._ctx = ctx
        self._c = channel

    @property
    def index(self) -> int:
        return self._c

    def param(self, name: str, default: Any = None) -> Any:
        """``name`` resolved for this channel (override → derive → socket default →
        ``default``)."""
        ctx = self._ctx
        if name in ctx.params:                       # user override — one value, all channels
            return ctx.params[name]
        spec = ctx.socket(name)
        if spec is not None and getattr(spec, "derive", ""):
            try:                                     # channel-c symbols; reads auto-fenced
                return eval_derive(spec.derive, envelope_symbols(ctx.env, self._c))
            except Exception:                        # noqa: BLE001 — bad expr → static default
                pass
        if default is not None:
            return default
        return getattr(spec, "default", None) if spec is not None else None

    def emission_nm(self, default: Any = None) -> Any:
        """This channel's emission λ (nm) from ``channel_emission_nm`` (recorded →
        memo-fenced); a scalar/absent value applies to all channels."""
        em = self._ctx.calib("channel_emission_nm")
        if isinstance(em, (list, tuple)):
            return em[self._c] if 0 <= self._c < len(em) else default
        return em if em is not None else default


Compute = Callable[[EvalContext], Any]


# ── the engine ────────────────────────────────────────────────────────────────

class Engine:
    """Lazy pull scheduler over a :class:`~nodegraph.graph.Graph`.

    ``computes`` maps ``op_key`` → a ``compute(ctx)`` function; a node without a
    compute falls back to ``seeds[node_id]`` (a pure source payload). ``providers``
    attaches a ``TileProvider`` to a source node (available as ``ctx.provider``).
    ``meta_seeds`` are the source :class:`MetaEnvelope`\\ s for the metadata pass.
    """

    def __init__(self, graph: Graph, *, computes: Mapping[str, Compute],
                 memo: Optional[Memo] = None,
                 seeds: Optional[Mapping[str, Any]] = None,
                 providers: Optional[Mapping[str, Any]] = None,
                 meta_seeds: Optional[Mapping[str, MetaEnvelope]] = None,
                 strict_reads: bool = False,
                 cache_bytes: int = 1 << 30,
                 memo_bytes: Optional[int] = None,
                 observer: Optional[Observer] = None) -> None:
        self.graph = graph
        self.computes = dict(computes)
        # A supplied Memo keeps its own budget; a fresh one honors `memo_bytes`
        # (None = unbounded, the headless default — see Memo GC). The persistent GUI
        # memo is the flagged high-T hazard, so the GUI runner opts into a budget.
        self.memo = memo or Memo(budget_bytes=memo_bytes)
        self.seeds = dict(seeds or {})
        self.providers = dict(providers or {})
        self._meta_seeds = dict(meta_seeds or {})
        self._envs = propagate_meta(graph, self._meta_seeds)
        self.compute_count = 0
        # C7 debug guard: when True, a node's input Dataset payloads get a
        # `_StrictCalibMetadata` view so an un-contexted calibration read hard-errors.
        self.strict_reads = strict_reads
        # C1: the shared byte-budget tile/unit cache streaming providers fill, and
        # the field materialization cache backed by the same budget (V2.04 §3/§4).
        self.tiles = TileCache(cache_bytes)
        self.fields = FieldCache(store=self.tiles)
        # per-node run progress (see :data:`Observer`). Never affects results, and its
        # exceptions are swallowed — an engine run must not depend on who is watching.
        self.observer = observer

    # ── observation ───────────────────────────────────────────────────────────
    def _emit(self, event: str, node_id: str, **info: Any) -> None:
        if self.observer is None:
            return
        try:
            self.observer(event, node_id, info)
        except Exception:  # noqa: BLE001 — a progress sink must never break a run
            pass

    # ── metadata (edit-time) ──────────────────────────────────────────────────
    def reseed_meta(self, meta_seeds: Mapping[str, MetaEnvelope]) -> None:
        """Recompute the metadata envelopes (e.g. after a calibration/graph edit),
        keeping the memo — so a subsequent pull re-validates declared reads."""
        self._meta_seeds = dict(meta_seeds)
        self._envs = propagate_meta(self.graph, self._meta_seeds)

    def env(self, node_id: str) -> MetaEnvelope:
        return self._envs.get(node_id, MetaEnvelope())

    # ── pull ──────────────────────────────────────────────────────────────────
    def pull(self, node_id: str) -> Any:
        """Compute (or fetch from the memo) ``node_id``'s output payload."""
        return self.entry(node_id).payload

    def entry(self, node_id: str) -> Entry:
        """The memo :class:`Entry` for ``node_id`` (computing if needed). The
        recursion headroom is the C1 stopgap for deep unrolled-zone chains (the
        recursive ``_entry`` nests a few frames per node; iterative rewrite is a
        follow-up, V2.04 §6b)."""
        with recursion_headroom(8 * len(self.graph.nodes)):
            return self._entry(node_id, set())

    def _entry(self, node_id: str, stack: Set[str]) -> Entry:
        if node_id in stack:
            raise ValueError(f"cycle through {node_id!r} (rejected outside zones)")
        node = self.graph.nodes[node_id]
        spec = node.spec()
        state = node.state(spec)

        # upstream first (lazy, recursive) — in CANONICAL socket order so both the
        # inputs tuple and the memo key are independent of edge-insertion order
        # (review #4: a multi-input node otherwise gets args in the wrong slots).
        preds = self.graph.preds(node_id)
        if spec is not None:
            pos = {s.name: i for i, s in enumerate(spec.inputs)}
            preds = [e for _, e in sorted(
                enumerate(preds),
                key=lambda ie: (pos.get(ie[1].dst_socket, len(pos)), ie[0]))]
        up = [self._entry(e.src, stack | {node_id}) for e in preds]
        up_hashes = tuple(e.recipe_hash for e in up)
        up_revs = tuple(e.revision for e in up)

        # fold the mode/toggle state into params so the lever re-keys the memo
        params = {**node.params, "__modes__": dict(state)}
        # A source node's payload comes from OUTSIDE the recipe (a seed or provider), so
        # its identity must enter the key or two distinct sources with the same op+params
        # collide and share one cached payload (review #3). A provider's data-version
        # (file mtime/id) folds in once providers expose `.version` — until then the
        # node-id distinguishes distinct sources; a same-provider file change on disk is
        # still a deferred hole (needs leaf_recipe_hash provider identity).
        if not preds:
            prov = self.providers.get(node_id)
            if prov is not None:
                params["__source__"] = node_id
                params["__provider_version__"] = getattr(prov, "version", None)
            elif node_id in self.seeds:
                params["__source__"] = node_id
                # C1 review 2026-07-22: a seed's DATA identity must enter the key —
                # swapping engine.seeds[nid] (or a disk store changing under a seed
                # Dataset) must not serve the stale memoized chain. A Dataset seed's
                # image provider carries `version` (content/mtime); other seeds stay
                # node-id-keyed (frozen for the engine's lifetime by convention).
                seed_img = getattr(self.seeds[node_id], "image", None)
                ver = getattr(seed_img, "version", None)
                if ver is not None:
                    params["__seed_version__"] = ver
        rh = node_recipe_hash(node.op_key, params, up_hashes, up_revs)
        env = self.env(node_id)

        cached = self.memo.get(rh)
        if cached is not None and self._reads_valid(cached, env):
            self._emit("cached", node_id, op_key=node.op_key)
            return cached

        # compute
        gran = spec.resolve_granularity(state) if spec is not None else None
        kax = spec.resolve_kernel_axes(state) if spec is not None else None
        rc = ReadContext(env.metadata)
        env_view = replace(env, metadata=_RecordingMetadata(env.metadata, rc))
        inputs = tuple(self._guard_reads(e.payload) for e in up)
        # C1: socket-name → payload (the positional tuple has gaps for unwired
        # sockets, so a value/field payload is only reachable by name). A `multi`
        # socket collects a tuple in canonical order; a second edge into a NON-multi
        # socket is a hard error, not a silent last-wins overwrite (review 2026-07-22).
        multi_names = ({s.name for s in spec.inputs if s.multi}
                       if spec is not None else set())
        by_name: Dict[str, Any] = {}
        for e, p in zip(preds, inputs):
            nm = e.dst_socket
            if nm in multi_names:
                by_name.setdefault(nm, [])
                by_name[nm].append(p)
            elif nm in by_name:
                raise ValueError(
                    f"two edges wired into non-multi socket {nm!r} of {node_id!r}")
            else:
                by_name[nm] = p
        for nm in multi_names & set(by_name):
            by_name[nm] = tuple(by_name[nm])
        ctx = EvalContext(
            node_id=node_id, op_key=node.op_key, params=params, env=env_view,
            granularity=gran, kernel_axes=kax,
            inputs=inputs, reads=rc,
            provider=self.providers.get(node_id),
            by_name=by_name, tiles=self.tiles, fields=self.fields, spec=spec,
            observer=self.observer,
        )
        fn = self.computes.get(node.op_key)
        # per-node progress: the window between "start" and "done" is exactly the time
        # this node's compute owns. A node returning a lazy provider closes it almost
        # immediately — its cost lands later, at read time, and is honestly NOT reported
        # here (the sink shows such a node as instant, not as a filled bar).
        self._emit("start", node_id, op_key=node.op_key, depth=len(stack))
        t0 = time.perf_counter()
        try:
            if fn is not None:
                payload = fn(ctx)
            elif node_id in self.seeds:
                payload = self.seeds[node_id]
            else:
                raise KeyError(
                    f"no compute for op {node.op_key!r} and no seed for node {node_id!r}")
        except BaseException as exc:      # noqa: BLE001 — observe, then re-raise verbatim
            self._emit("error", node_id, op_key=node.op_key,
                       seconds=time.perf_counter() - t0, error=repr(exc))
            raise
        self._emit("done", node_id, op_key=node.op_key,
                   seconds=time.perf_counter() - t0)
        rc.freeze()          # a late ctx.calib from a lazy closure is a hard error (C1)
        self.compute_count += 1
        # C7: a compute that returns `ctx.inputs[0].with_image(...)` carries the input's
        # strict metadata wrapper into the OUTPUT via dataclasses.replace. Launder it so
        # the debug guard never enters the memo/returned payload (else a non-strict
        # consumer sharing the memo could trip on it).
        if (self.strict_reads and isinstance(payload, Dataset)
                and isinstance(payload.metadata, _StrictCalibMetadata)):
            payload = replace(payload, metadata=dict(payload.metadata))

        header = OutputHeader(
            axes=env.axes, metadata_digest=digest("md", dict(env.metadata)),
            layers=tuple(env.layers))
        return self.memo.put(rh, payload, node_key=node_id,
                             reads=rc.declared_reads(), header=header)

    def _guard_reads(self, payload: Any) -> Any:
        """In ``strict_reads`` mode, present an input ``Dataset``'s metadata as a
        :class:`_StrictCalibMetadata` so a compute that reaches past ``ctx.calib`` into
        the payload's calibration hard-errors (C7). A no-op otherwise, and idempotent."""
        if (self.strict_reads and isinstance(payload, Dataset)
                and not isinstance(payload.metadata, _StrictCalibMetadata)):
            return replace(payload, metadata=_StrictCalibMetadata(payload.metadata))
        return payload

    def _reads_valid(self, entry: Entry, env: MetaEnvelope) -> bool:
        """A hit is valid only if every calibration value the node read still has the
        same digest under the current envelope (V2.02 §8 verify step)."""
        md = env.metadata
        return all(value_digest(md.get(k)) == d for k, d in entry.reads)


__all__ = ["ReadContext", "EvalContext", "Compute", "Engine", "Observer"]
