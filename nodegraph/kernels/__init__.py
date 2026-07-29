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

Optional deps, each gated at its own call site so the palette registers without any of them:
``scikit-learn`` (point clustering), ``al-dic`` (DIC), ``tensorflow``/``stardist`` and
``cellSAM``/``torch`` (the two learned Segmentation methods), ``numba``/``pandas``
(tracking), ``cupy`` (a GPU FFT seed). The learned methods additionally need model WEIGHTS,
which are a separate failure mode from a missing package: run
``python scripts/setup_cellsam.py`` once per machine for CellSAM. If a weights download
fails with a certificate error while your browser and ``pip`` work, your network intercepts
HTTPS — see ``cellsam_segment.md`` §8, which that script also handles.
"""
