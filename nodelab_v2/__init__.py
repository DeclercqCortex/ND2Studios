"""NodeLab v2 — the PySide6 GUI for the ``nodegraph`` v2 model.

The node editor that renders the ``nodegraph`` registry: sockets-on-card, the 2D/3D
header switch with live socket relayout, field-able value sockets, and a click-to-edit
properties inspector. Qt lives only here; the ``nodegraph`` core stays Qt-free.

(The ``_v2`` suffix is historical: this replaced a first-generation ``nodelab`` package on
the ``pipeline_kit`` backend, removed 2026-07-29.)

Run:  ``python -m nodelab_v2``  (or ``from nodelab_v2.app import run; run()``).
"""
from __future__ import annotations

__all__ = ["run"]


def run() -> int:
    from nodelab_v2.app import run as _run
    return _run()
