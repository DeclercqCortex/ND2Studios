"""Portable analysis math-kernels: vendored v1 code plus thin third-party adapters.

Pure numpy/scipy/… compute — imports **nothing** from ``nd2studios`` or ``nodegraph``.
Each ``<module>.py`` is **lazily imported inside** the relevant ``nodegraph.nodes``
compute, so the engine core stays importable without the heavy per-kernel deps
(numba / pandas / tensorflow / stardist / cellSAM / torch / cv2 / cupy). The
``<module>.md`` beside each
kernel is the authoritative integration contract (entry point, array shapes, units,
gotchas). This ``__init__`` intentionally imports none of them — each is loaded on the
compute path that needs it.

Most were vendored **byte-verbatim** from ND2Studios v1.45 and ported into the v2 catalog
under Phase 7 (V2.05 §2 capability gaps). Two are NOT vendored, and say so in their own
docstring: ``mesh_raster`` (new in-repo code, V2.08) and ``cellsam_segment`` (a new adapter
around the third-party ``cellSAM`` package, V2.12) — the latter in the same "external
wrapper" class as ``stardist_segment`` and ``dic_correlate``, which hold no algorithm
either.

Optional deps in this env: ``scikit-learn``, ``al-dic``, ``tensorflow``/``stardist``,
``torch`` AND ``cellSAM`` are all present, with the CellSAM weights cached under
``~/.deepcell/models`` — so every method of ``analysis.segment`` runs here. NOTE for anyone
debugging a model download on this machine: HTTPS is intercepted by antivirus, which Python
3.13 rejects outright; see ``cellsam_segment.md`` §8.
"""
