"""Vendored pure-analysis math-kernels (byte-verbatim from ND2Studios v1.45).

Pure numpy/scipy/… compute — imports **nothing** from ``nd2studios`` or ``nodegraph``.
Each ``<module>.py`` is **lazily imported inside** the relevant ``nodegraph.nodes``
compute, so the engine core stays importable without the heavy per-kernel deps
(numba / pandas / tensorflow / stardist / cv2 / cupy). The ``<module>.md`` beside each
kernel is the authoritative integration contract (entry point, array shapes, units,
gotchas). This ``__init__`` intentionally imports none of them — each is loaded on the
compute path that needs it.

Ported into the v2 node catalog under Phase 7 (V2.05 §2 capability gaps). Deps ABSENT
in this env: ``scikit-learn`` (gates granule clustering's GMM/HDBSCAN fits) and
``al-dic`` (gates DIC IC-GN correlation + mesh-refinement policy).
"""
