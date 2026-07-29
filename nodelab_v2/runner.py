"""EngineRunner — canvas → Engine off the UI thread (G7, LOCKED 2026-07-22: QThreadPool
worker + **epoch registry**, no qasync — the engine is synchronous CPU work, so a Qt
worker thread + queued-signal delivery is the whole bridge; stale results are dropped
by epoch on arrival. One pull runs at a time (latest-wins queueing).

The runner owns the run-side model glue:

* **Snapshot at submit** — the headless :class:`Graph` is built on the GUI thread from
  the :class:`~nodelab_v2.document.GraphDocument` (``to_graph(for_run=True)``: strips
  the ``__locked__`` UI annotation, bypasses muted nodes), so the worker never touches
  live GUI state.
* **A persistent Memo across runs** — engines are rebuilt when the document revision
  changes, the two-hash memo carries over, so an unrelated edit recomputes only the
  invalidated chain (the C1 promise, now user-visible).
* **Source resolution** — an ``io.load`` root resolves its ``path`` param through
  :mod:`nodelab_v2.ingest` (ingested once to an on-disk b2nd store next to the file,
  re-opened lazily after); an empty path falls back to a deterministic
  :class:`~nodegraph.provider.SyntheticProvider` demo source with real calibration so
  the whole GUI runs out of the box. The resolved source envelope is delivered back to
  the document (``set_meta_seed``) — the G8 live widget re-seed.
* **Plane rendering** — a job may ask for a display plane at ``(m, t, z, c)``; it is
  read in the worker (through the C1 tile cache) and decimated to ``max_dim`` for the
  Viewer, so the GUI thread never blocks on a lazy-chain compute.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.engine import Engine
from nodegraph.graph import Graph
from nodegraph.memo import Memo
from nodegraph.metadata import MetaEnvelope
from nodegraph.provider import SyntheticProvider

#: byte-budget LRU cap for the persistent Memo (Memo GC). The persistent memo is the
#: V2.04-flagged hazard: an eager full-raster node in a high-T zone would otherwise
#: retain one raster per iteration forever. 1 GiB matches the engine's default tile
#: cache; eviction only costs a recompute (correctness-safe). Tune per available RAM.
MEMO_BUDGET_BYTES = 1 << 30

#: demo calibration for the synthetic fallback source (drives the ƒmd derive pills)
_SYNTH_META = {
    "pixel_size_um": 0.1, "z_step_um": 0.3, "objective_na": 1.4,
    "objective_magnification": 60.0, "channel_emission_nm": [520.0, 640.0],
}
_SYNTH_AXES = AxisSizes(m=1, t=1, z=5, c=2, y=512, x=512)


def ensure_gui_ops() -> None:
    """Register the GUI-facing ops (``io.load`` source + ``view.viewer`` pass-through).
    Delegates to the **Qt-free** :mod:`nodelab_v2.ops` so the same registration (incl.
    ``view.viewer``'s compute in ``COMPUTES``) is available to a headless consumer of a
    saved graph — the GUI is not the only path that must run one."""
    from nodelab_v2.ops import ensure_ops
    ensure_ops()


def render_plane(provider: Any, m: int, t: int, z: int, c: int,
                 *, max_dim: int = 2048) -> np.ndarray:
    """A display plane at ``(m,t,z,c)``: picks the coarsest pyramid level that still
    exceeds ``max_dim`` quality-wise, else reads level 0 and stride-decimates (a
    streaming provider has no pyramid — V2.04 §5-1). Returns float64 (Y', X')."""
    level = 0
    ax = provider.level_axes(0)
    for lv in range(getattr(provider, "levels", 1)):
        lax = provider.level_axes(lv)
        level, ax = lv, lax
        if max(lax.y, lax.x) <= max_dim:
            break
    plane = np.asarray(provider.get_region(level, m, t, z, c, 0, ax.y, 0, ax.x),
                       dtype=float)
    stride = max(1, int(np.ceil(max(plane.shape) / max_dim)))
    return plane[::stride, ::stride]


def render_plane_native(provider: Any, m: int, t: int, z: int, c: int,
                        *, max_dim: int = 2048) -> Tuple[np.ndarray, int]:
    """A display plane at ``(m,t,z,c)`` in its **native dtype** (uint8/uint16/float) —
    the GPU uploader picks the texture internal format from the dtype, and the CPU
    fallback casts to float. Same level-selection + stride-decimation as
    :func:`render_plane`, but without the ``dtype=float`` cast (which is the expensive
    per-frame work we push onto the GPU / do once). Returns ``(plane2d, level)``."""
    level = 0
    ax = provider.level_axes(0)
    for lv in range(getattr(provider, "levels", 1)):
        lax = provider.level_axes(lv)
        level, ax = lv, lax
        if max(lax.y, lax.x) <= max_dim:
            break
    plane = np.asarray(provider.get_region(level, m, t, z, c, 0, ax.y, 0, ax.x))
    stride = max(1, int(np.ceil(max(plane.shape) / max_dim)))
    if stride > 1:
        plane = plane[::stride, ::stride]
    return np.ascontiguousarray(plane), level


class PlaneCache:
    """A byte-budgeted LRU of decoded, display-decimated planes (native dtype), keyed by
    ``(node_id, revision, m, t, z, c)``. Separate from the engine ``Memo``/``TileCache``:
    it holds *ready-to-upload* planes so scrubbing/playback and the background prefetcher
    share one warm store. Thread-safe (the prefetch pool + the GUI thread both touch it).
    Decimated planes are small (2048² uint16 ≈ 8 MB), so hundreds of T fit in the budget."""

    def __init__(self, budget_bytes: int = 512 << 20) -> None:
        self._budget = int(budget_bytes)
        self._d: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[np.ndarray]:
        with self._lock:
            arr = self._d.get(key)
            if arr is not None:
                self._d.move_to_end(key)
            return arr

    def put(self, key: tuple, arr: np.ndarray) -> None:
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= int(getattr(old, "nbytes", 0))
            self._d[key] = arr
            self._bytes += int(getattr(arr, "nbytes", 0))
            while self._bytes > self._budget and len(self._d) > 1:
                _k, v = self._d.popitem(last=False)
                self._bytes -= int(getattr(v, "nbytes", 0))

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0


#: minimum gap between two *fractional* progress deliveries for one node (seconds). A
#: per-plane compute can call ``ctx.progress`` thousands of times a second; the card only
#: needs enough to look alive. Start/finish transitions are never throttled — and neither
#: is the final ``done == total`` update, so a bar always lands full.
PROGRESS_MIN_INTERVAL_S = 0.05


class _Job:
    __slots__ = ("epoch", "graph", "revision", "node_id", "coords", "channels",
                 "sources")

    def __init__(self, epoch: int, graph: Graph, revision: int, node_id: str,
                 coords: Optional[Tuple[int, int, int, int]],
                 channels: Optional[Tuple[int, ...]],
                 sources: Dict[str, Dict[str, Any]]) -> None:
        self.epoch = epoch
        self.graph = graph
        self.revision = revision
        self.node_id = node_id
        self.coords = coords
        self.channels = channels      # channels to render into a colour composite
        self.sources = sources        # node_id -> {"path": str} (io.load roots)


class _Worker(QRunnable):
    def __init__(self, runner: "EngineRunner", job: _Job) -> None:
        super().__init__()
        self._r = runner
        self._job = job

    def run(self) -> None:  # worker thread
        r, job = self._r, self._job
        t0 = time.perf_counter()
        try:
            engine = r._ensure_engine(job)
            engine.observer = r._make_observer(job.epoch)
            payload = engine.pull(job.node_id)
            plane = None                    # dict {channel_index: 2-D native plane}
            axes = None
            if isinstance(payload, Dataset) and payload.image is not None:
                axes = payload.axes
                if job.coords is not None:
                    # a lazy chain does its real work HERE, under the viewed node's name —
                    # report it as that node's state so the card isn't idle while the
                    # provider reads tiles (the honest counterpart to ctx.progress).
                    r._progress.emit(("decode", job.node_id,
                                      {"epoch": job.epoch, "op_key": ""}))
                    plane = r._decode_planes(payload.image, job.node_id,
                                             job.coords, job.channels, axes)
            dt = time.perf_counter() - t0
            r._done.emit((job.epoch, job.node_id, payload, plane, axes, dt, None,
                          job.revision, job.coords, job.channels))
        except Exception:  # noqa: BLE001 — full trace to the GUI, never a dead thread
            r._done.emit((job.epoch, job.node_id, None, None, None,
                          time.perf_counter() - t0, traceback.format_exc(),
                          job.revision, job.coords, job.channels))


class _PrefetchJob(QRunnable):
    """Decodes a list of adjacent-frame planes off the held viewer provider into the
    shared :class:`PlaneCache`, on a pool thread. A stale generation (the cursor moved
    on) short-circuits the remaining reads, so a fast scrub never backs up the pool."""

    def __init__(self, runner: "EngineRunner", gen: int,
                 jobs: List[Tuple[tuple, int, int, int, int]]) -> None:
        super().__init__()
        self._r = runner
        self._gen = gen
        self._jobs = jobs

    def run(self) -> None:  # worker thread
        r = self._r
        prov = r._viewer_provider
        if prov is None:
            return
        for (key, m, t, z, ch) in self._jobs:
            if self._gen != r._prefetch_gen:
                return                       # superseded by a newer cursor position
            if r._planes.get(key) is not None:
                continue
            try:
                with r._decode_lock:
                    if self._gen != r._prefetch_gen:
                        return
                    arr, _lv = render_plane_native(prov, m, t, z, ch)
                r._planes.put(key, arr)
            except Exception:                # noqa: BLE001 — prefetch is best-effort
                return


class EngineRunner(QObject):
    """Submit pulls; receive results on the GUI thread; drop stale epochs."""

    started = Signal(str)                        # node_id
    finished = Signal(str, object, object, object, float)   # id, payload, plane, axes, s
    plane_ready = Signal(str, object, object, float)   # id, planes, axes, s (fast path)
    failed = Signal(str, str)                    # node_id, traceback
    source_resolved = Signal(str, object)        # node_id, MetaEnvelope (G8 re-seed)
    #: the node set this pull may touch (the pulled node's ancestor closure), emitted on
    #: the GUI thread at submit so every participating card can show "queued" before any
    #: work starts: ``(target_node_id, [node_id, …])``.
    plan = Signal(str, object)
    #: per-node run progress, forwarded from the engine observer onto the GUI thread:
    #: ``(event, node_id, info)`` — see :data:`nodegraph.engine.Observer`, plus the
    #: runner-level ``"decode"`` event (the viewed node's planes are being read).
    node_progress = Signal(str, str, object)

    _done = Signal(object)                       # internal cross-thread delivery
    _progress = Signal(object)                   # internal cross-thread progress delivery

    def __init__(self, document) -> None:
        super().__init__()
        self.document = document
        self._pool = QThreadPool.globalInstance()
        # Persists across engine rebuilds — so it's the memory hazard V2.04 flagged: an
        # eager full-raster node in a high-T zone would otherwise pin one raster per
        # iteration forever. Cap it with a byte-budget LRU (Memo GC); eviction only costs
        # a recompute (correctness-safe). Budget matches the engine's default tile cache.
        self._memo = Memo(budget_bytes=MEMO_BUDGET_BYTES)
        self._engine: Optional[Engine] = None
        self._engine_rev = -1
        self._providers: Dict[Any, Tuple[Any, MetaEnvelope]] = {}
        self._channel_display: Dict[Any, Dict[str, Any]] = {}   # source key → Viewer meta
        self._epoch = 0
        self._busy = False
        self._pending: Optional[Tuple[str, Any, Any]] = None
        self._announced: Dict[str, Any] = {}     # node_id → last announced source key
        self._node_source_key: Dict[str, Any] = {}
        # ── decoded-plane fast path (scrub/play without a graph re-pull) ──────────
        self._planes = PlaneCache()              # (node,rev,m,t,z,c) → native plane
        self._decode_lock = threading.Lock()     # serialize provider reads (shared TileCache)
        self._viewer_provider: Any = None        # held image provider of the viewed node
        self._viewer_node: Optional[str] = None
        self._viewer_axes: Any = None
        self._viewer_rev: int = -1               # document.revision the provider belongs to
        self._prefetch_gen = 0
        # ── per-node progress bookkeeping ─────────────────────────────────────────
        self._last_progress: Dict[str, float] = {}   # node_id → last delivery (perf time)
        self._done.connect(self._deliver)
        self._progress.connect(self._deliver_progress)
        document.on_change(self._prune)

    # ── public API (GUI thread) ────────────────────────────────────────────────
    def pull(self, node_id: str,
             coords: Optional[Tuple[int, int, int, int]] = None,
             channels: Optional[Tuple[int, ...]] = None) -> None:
        """Request a full engine pull (+ optional display planes for ``channels``).
        Establishes/refreshes the held viewer provider. Latest-wins while busy."""
        if node_id not in self.document.nodes:
            return
        if self._busy:
            self._pending = (node_id, coords, channels)
            return
        self._submit(node_id, coords, channels)

    def request_plane(self, node_id: str,
                      coords: Optional[Tuple[int, int, int, int]] = None,
                      channels: Optional[Tuple[int, ...]] = None) -> None:
        """Coords-only request. When the viewed node + document revision are unchanged
        (only the M/T/Z cursor or the active-channel set moved), bypass the graph
        snapshot + ``engine.pull`` entirely and serve the plane straight from the
        :class:`PlaneCache` (decoding a miss synchronously — a single decimated read),
        then warm adjacent frames. Otherwise fall back to a full :meth:`pull`, which
        re-establishes the provider handle for this (node, revision)."""
        if node_id not in self.document.nodes:
            return
        if (coords is not None and node_id == self._viewer_node
                and self._viewer_provider is not None
                and self.document.revision == self._viewer_rev):
            self._serve_from_cache(node_id, coords, channels)
            return
        self.pull(node_id, coords, channels)

    def invalidate(self) -> None:
        """Drop any in-flight result (edits during a run) and drop the held provider —
        a graph edit may change pixels, so the next request must re-pull through the
        engine and the decoded-plane cache is no longer valid."""
        self._epoch += 1
        self._viewer_rev = -1
        self._prefetch_gen += 1
        self._planes.clear()

    # ── decoded-plane fast path ────────────────────────────────────────────────
    def _clamp_coords(self, coords, axes):
        return tuple(min(max(0, int(v)), s - 1) for v, s in
                     zip(coords, (axes.m, axes.t, axes.z, axes.c)))

    def _decode_planes(self, provider, node_id, coords, channels, axes
                       ) -> Dict[int, np.ndarray]:
        """Native per-channel planes at ``coords``, cache-first (used by the worker and
        the fast path). Misses decode under the shared decode lock and are cached. The
        cache key omits the revision — :meth:`invalidate` clears the whole cache on any
        edit, so a stale plane can never be served across a graph change."""
        m, t, z, c = self._clamp_coords(coords, axes)
        chans = channels if channels else (c,)
        out: Dict[int, np.ndarray] = {}
        for ch in chans:
            ch = min(max(0, int(ch)), axes.c - 1)
            if ch in out:
                continue
            key = (node_id, m, t, z, ch)
            arr = self._planes.get(key)
            if arr is None:
                with self._decode_lock:
                    arr, _lv = render_plane_native(provider, m, t, z, ch)
                self._planes.put(key, arr)
            out[ch] = arr
        return out

    def _serve_from_cache(self, node_id, coords, channels) -> None:  # GUI thread
        t0 = time.perf_counter()
        axes = self._viewer_axes
        planes = self._decode_planes(self._viewer_provider, node_id,
                                     coords, channels, axes)
        self.plane_ready.emit(node_id, planes, axes, time.perf_counter() - t0)
        self.prefetch(node_id, self._clamp_coords(coords, axes),
                      tuple(channels) if channels else None)

    def prefetch(self, node_id, center, channels, *, span: int = 8) -> None:
        """Warm the plane cache for frames around ``center`` on background pool threads —
        bidirectional in T (covers scrubbing), nearest-first (covers forward play). A new
        call supersedes older prefetch jobs via ``_prefetch_gen``."""
        prov, axes = self._viewer_provider, self._viewer_axes
        if prov is None or axes is None or getattr(axes, "t", 1) <= 1:
            return
        m, t, z, _c = center
        nt = axes.t
        chans = channels if channels else (min(max(0, center[3]), axes.c - 1),)
        self._prefetch_gen += 1
        gen = self._prefetch_gen
        jobs: List[Tuple[tuple, int, int, int, int]] = []
        for d in range(1, span + 1):
            for tt in ((t + d) % nt, (t - d) % nt):
                for ch in chans:
                    ch = min(max(0, int(ch)), axes.c - 1)
                    key = (node_id, m, tt, z, ch)
                    if self._planes.get(key) is None:
                        jobs.append((key, m, tt, z, ch))
        if jobs:
            self._pool.start(_PrefetchJob(self, gen, jobs))

    # ── per-node progress (engine observer → GUI thread) ──────────────────────
    def _make_observer(self, epoch: int):
        """An :data:`nodegraph.engine.Observer` that forwards node events to the GUI
        thread through ``_progress`` (a queued signal — the observer is called on the
        worker thread). Fractional ``progress`` events are rate-limited per node; every
        other event, and the final ``done == total``, always gets through."""
        def observe(event: str, node_id: str, info: Dict[str, Any]) -> None:
            if epoch != self._epoch:
                return                        # superseded pull — stop reporting for it
            if event == "progress":
                now = time.perf_counter()
                last = self._last_progress.get(node_id, 0.0)
                final = info.get("done") == info.get("total")
                if not final and (now - last) < PROGRESS_MIN_INTERVAL_S:
                    return
                self._last_progress[node_id] = now
            else:
                self._last_progress.pop(node_id, None)
            self._progress.emit((event, node_id, {**info, "epoch": epoch}))
        return observe

    def _deliver_progress(self, packet) -> None:     # GUI thread (queued)
        event, node_id, info = packet
        if info.get("epoch") != self._epoch:
            return                            # a stale pull's tail — the cards moved on
        self.node_progress.emit(event, node_id, info)

    def planned_nodes(self, node_id: str, graph: Optional[Graph] = None) -> List[str]:
        """Every node a pull of ``node_id`` may evaluate: itself plus its transitive
        upstream (a memo hit still *participates*, and reports itself ``cached``). Read
        off the run graph so it matches what the engine will actually walk — muted nodes
        are bypassed there, and group bodies are expanded."""
        try:
            graph = graph if graph is not None else self.document.to_graph(
                for_run=True, materialize=True)
        except Exception:  # noqa: BLE001 — an unbuildable graph plans as just the target
            return [node_id]
        if node_id not in graph.nodes:
            return [node_id]
        seen, stack = set(), [node_id]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(e.src for e in graph.preds(nid) if e.src not in seen)
        return sorted(seen)

    # ── internals ─────────────────────────────────────────────────────────────
    def _submit(self, node_id: str, coords, channels=None) -> None:
        self._epoch += 1
        self._last_progress.clear()
        sources = {
            rec.id: {"path": str(rec.params.get("path", "") or "")}
            for rec in self.document.nodes.values() if rec.op_key == "io.load"
        }
        graph = self.document.to_graph(for_run=True, materialize=True)
        job = _Job(self._epoch, graph,
                   self.document.revision, node_id, coords, channels, sources)
        self._busy = True
        self.plan.emit(node_id, self.planned_nodes(node_id, graph))
        self.started.emit(node_id)
        self._pool.start(_Worker(self, job))

    def _deliver(self, packet) -> None:          # GUI thread (queued)
        epoch, node_id, payload, plane, axes, dt, err, revision, coords, channels = packet
        self._busy = False
        # staleness is judged BEFORE the re-seed side effect below: delivering a
        # resolved source envelope notifies the document → the window calls
        # invalidate() → the epoch bumps — and would drop the very result being
        # delivered. The envelope seeding is display-only (the ENGINE resolved its
        # own meta seeds at run time), so it never stales this result.
        stale = epoch != self._epoch
        # G8 live re-seed: hand newly resolved source envelopes to the document
        for nid, env in self._fresh_envs():
            self.document.set_meta_seed(nid, env)
        pending, self._pending = self._pending, None
        if pending is not None:
            self._submit(*pending)               # latest-wins supersedes this result
            stale = True
        if stale:
            return
        if err is not None:
            self.failed.emit(node_id, err)
            return
        # Hold the image provider so subsequent coords-only requests skip the engine
        # (the fast path). Tie it to the CURRENT document revision — not the job's: the
        # G8 source re-seed just above (``set_meta_seed``) can bump the revision (display
        # metadata only, the graph/pixels are unchanged), and a genuine edit later runs
        # invalidate() → ``_viewer_rev = -1`` anyway, so the next request re-pulls.
        if isinstance(payload, Dataset) and payload.image is not None:
            self._viewer_provider = payload.image
            self._viewer_node = node_id
            self._viewer_axes = payload.axes
            self._viewer_rev = self.document.revision
        self.finished.emit(node_id, payload, plane, axes, dt)
        if (coords is not None and self._viewer_provider is not None
                and self._viewer_axes is not None):
            self.prefetch(node_id, self._clamp_coords(coords, self._viewer_axes),
                          tuple(channels) if channels else None)

    def _fresh_envs(self):
        # re-announce whenever a node's RESOLVED source key changed (a path edit →
        # a new provider/envelope), not only on first resolution — else the G8 pill
        # re-seed would freeze on the synthetic fallback forever (review 2026-07-22).
        out = []
        for nid, key in list(self._node_source_key.items()):
            if self._announced.get(nid) == key:
                continue
            entry = self._providers.get(key)
            if entry is not None:
                self._announced[nid] = key
                out.append((nid, entry[1]))
        return out

    def _prune(self) -> None:
        """Drop bookkeeping for deleted io.load nodes so it doesn't grow unbounded
        (the resolved-provider cache is keyed by source path, shared across nodes, so
        it is left intact — reopening the same file is a hit)."""
        live = set(self.document.nodes)
        for nid in list(self._node_source_key):
            if nid not in live:
                self._node_source_key.pop(nid, None)
                self._announced.pop(nid, None)
        if self._viewer_node is not None and self._viewer_node not in live:
            self._viewer_node = None
            self._viewer_provider = None
            self._viewer_rev = -1

    def _ensure_engine(self, job: _Job) -> Engine:   # worker thread
        seeds: Dict[str, Any] = {}
        meta_seeds: Dict[str, MetaEnvelope] = {}
        for nid, cfg in job.sources.items():
            prov, env = self._resolve_source(nid, cfg)
            # Display metadata (channel names/emission/colors) rides on the seed
            # Dataset only — NOT the engine meta-seed (kept to the calibration schema).
            disp = self._channel_display.get(self._node_source_key.get(nid), {})
            md = dict(env.metadata); md.update(disp)
            seeds[nid] = Dataset(axes=env.axes, metadata=md).with_image(prov)
            meta_seeds[nid] = env
        if self._engine is None or self._engine_rev != job.revision:
            from nodegraph.nodes import COMPUTES
            self._engine = Engine(job.graph, computes=COMPUTES, memo=self._memo,
                                  seeds=seeds, meta_seeds=meta_seeds)
            self._engine_rev = job.revision
        else:
            self._engine.seeds.update(seeds)
        return self._engine

    def _resolve_source(self, node_id: str, cfg: Dict[str, Any]
                        ) -> Tuple[Any, MetaEnvelope]:   # worker thread
        path = cfg.get("path", "")
        # Normalize the hand-entered path: strip whitespace and surrounding quotes
        # (Windows "Copy as path" wraps the path in double quotes — passing those to
        # the reader yields a cryptic OSError [Errno 22]).
        if isinstance(path, str):
            path = path.strip().strip('"').strip("'").strip()
        if path and not os.path.isfile(path):
            raise FileNotFoundError(
                f"No such ND2/TIFF file:\n  {path!r}\n"
                f"Reload it via File → Load ND2/TIFF file… (or fix the node's 'path' "
                f"field). Leave it empty for the synthetic demo source.")
        key = ("image", os.path.abspath(path)) if path else ("synthetic",)
        self._node_source_key[node_id] = key
        hit = self._providers.get(key)
        if hit is not None:
            return hit
        if not path:
            prov = SyntheticProvider(_SYNTH_AXES, tile=128)
            env = MetaEnvelope(axes=_SYNTH_AXES, metadata=dict(_SYNTH_META))
            disp = {"channel_names": [f"Ch{i}" for i in range(_SYNTH_AXES.c)],
                    "channel_emission_nm": list(_SYNTH_META["channel_emission_nm"])}
        else:
            from nodelab_v2.ingest import (
                ingest_image, open_store, read_calibration, read_channel_display)
            store = os.path.splitext(path)[0] + ".b2nd_store"
            if os.path.isdir(store):
                prov = open_store(store)
                axes = prov.axes
                env = MetaEnvelope(axes=axes, metadata=read_calibration(path))
            else:
                prov, env = ingest_image(path, store_path=store, levels=3)
            disp = read_channel_display(path)
        self._providers[key] = (prov, env)
        self._channel_display[key] = disp
        return prov, env


__all__ = ["EngineRunner", "ensure_gui_ops", "render_plane", "render_plane_native",
           "PlaneCache"]
