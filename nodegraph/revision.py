"""Process-monotonic revision counter — the memo *identity* for nodegraph v2.

Every derived value that participates in memoization — an :class:`~nodegraph.
dataset.AttributeLayer`, a memo entry, a computed tile — is stamped with a
``revision``: a strictly increasing integer minted by :func:`next_revision`.

Identity is this cheap monotonic counter, deliberately **not**:

* Python ``id()`` — recycled by the allocator, so a freed-then-reused object
  would collide with a stale memo key (V2.02 §6 cross-cutting invariant).
* a content hash of the payload — an 85 MB plane must never be read just to be
  *identified*; content hashing is reserved for the separate output-fingerprint
  (cutoff/dedup), not for identity (V2.02 §I.a / §6).

Because a recomputed value gets a *fresh* revision, any downstream memo key that
embeds an input's revision auto-invalidates the moment that input is replaced
(V2.02 §4). ``Dataset.with_attribute`` structural-shares via ``replace``, so a
new layer carries a new revision and stale keys fall out naturally.

Thread-safe: the Phase-2 pull scheduler may mint revisions from worker threads.
Qt-free; pure standard library.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_current = 0


def next_revision() -> int:
    """Return the next strictly-increasing revision (thread-safe, never 0)."""
    global _current
    with _lock:
        _current += 1
        return _current


def peek_revision() -> int:
    """The most-recently-issued revision without consuming one (0 before any)."""
    with _lock:
        return _current


__all__ = ["next_revision", "peek_revision"]
