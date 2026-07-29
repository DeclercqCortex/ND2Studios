"""Numba prototype + benchmark — greedy NMS (`bead_detect._suppress_close`).

WHY THIS EXISTS
    `_suppress_close` (nodegraph/kernels/bead_detect.py) is the one bead-detection
    hot loop still written in *pure Python*: a greedy, data-dependent O(n²)
    non-max suppression (keep the brightest point in each `min_distance`
    neighbourhood). Its numba siblings in the same file (`_radial_symmetry_3d`,
    the sub-pixel kernels) are already `@nb.njit`. This script (a) proves a numba
    port is bit-for-bit equivalent to the numpy version, and (b) measures the
    speed-up across realistic bead counts — the evidence behind "port this one".

    The greedy accept/reject is sequential (a later point's fate depends on the
    running kept-set), so it cannot vectorize cleanly in numpy — exactly the
    regime where numba wins. The O(n²) inner distance loop is the whole cost.

DESIGN NOTE (why `order` is precomputed outside the kernel)
    The intensity sort (`np.argsort`) is already fast C and its tie-breaking
    fixes which of two equal-intensity points survives. We compute `order` in
    numpy and pass it in, so (1) the njit kernel is *only* the O(n²) hot loop and
    (2) equivalence is exact regardless of numba's sort tie-breaking.

Manual (NOT in nodegraph.selftest — it is a timing tool):
    PYTHONUTF8=1 python scripts/_bench_nms_numba.py
    PYTHONUTF8=1 python scripts/_bench_nms_numba.py --full   # adds 20k/40k points
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numba as nb


# ── the current production implementation (verbatim behaviour) ────────────────
def _suppress_close_numpy(coords_zyx: np.ndarray, intensities: np.ndarray,
                          min_distance: float) -> np.ndarray:
    """Pure-Python greedy NMS — a byte-faithful copy of the shipping
    `bead_detect._suppress_close` (kept here so the bench is self-contained and
    can't drift if the source signature changes)."""
    n = coords_zyx.shape[0]
    if min_distance <= 0.0 or n <= 1:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(intensities)[::-1]
    md2 = float(min_distance) ** 2
    kept = []
    kept_pts = []
    for idx in order:
        p = coords_zyx[idx]
        ok = True
        for kp in kept_pts:
            d = p - kp
            if float(d @ d) < md2:
                ok = False
                break
        if ok:
            kept.append(int(idx))
            kept_pts.append(p)
    return np.array(sorted(kept), dtype=np.int64)


# ── the proposed numba port ───────────────────────────────────────────────────
@nb.njit(cache=True)
def _nms_kernel(coords: np.ndarray, order: np.ndarray, md2: float) -> np.ndarray:
    """Greedy NMS hot loop. `order` = point indices brightest-first; returns a
    boolean keep-mask indexed by original point index.

    ndim-agnostic (2 or 3): the distance is an explicit component loop so numba
    emits a tight scalar kernel with no temporary arrays."""
    n = coords.shape[0]
    ndim = coords.shape[1]
    keep = np.zeros(n, dtype=np.bool_)
    kept_idx = np.empty(n, dtype=np.int64)
    n_kept = 0
    for oi in range(order.shape[0]):
        idx = order[oi]
        ok = True
        for kk in range(n_kept):
            kp = kept_idx[kk]
            d2 = 0.0
            for c in range(ndim):
                diff = coords[idx, c] - coords[kp, c]
                d2 += diff * diff
            if d2 < md2:
                ok = False
                break
        if ok:
            keep[idx] = True
            kept_idx[n_kept] = idx
            n_kept += 1
    return keep


def _suppress_close_numba(coords_zyx: np.ndarray, intensities: np.ndarray,
                          min_distance: float) -> np.ndarray:
    """Drop-in replacement wrapper: same signature and return contract as
    `_suppress_close` (sorted int64 array of kept original indices)."""
    n = coords_zyx.shape[0]
    if min_distance <= 0.0 or n <= 1:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(intensities)[::-1].astype(np.int64)
    coords = np.ascontiguousarray(coords_zyx, dtype=np.float64)
    keep = _nms_kernel(coords, order, float(min_distance) ** 2)
    return np.nonzero(keep)[0].astype(np.int64)   # np.nonzero is already ascending


# ── data + timing ─────────────────────────────────────────────────────────────
def _make_field(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`n` random 3-D points in a 512³ box with random intensities — a dense
    bead field where NMS actually rejects a lot (worst case for the O(n²) loop)."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 512.0, size=(n, 3)).astype(np.float64)
    intens = rng.random(n).astype(np.float64)
    return coords, intens


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="add 20k/40k points")
    counts = [200, 1000, 5000, 10000] + ([20000, 40000] if ap.parse_args().full else [])
    min_distance = 5.0

    # --- equivalence: numpy vs numba must return identical kept indices ---------
    print("equivalence check (numpy vs numba):")
    for n in (50, 500, 3000):
        coords, intens = _make_field(n, seed=n)
        a = _suppress_close_numpy(coords, intens, min_distance)
        b = _suppress_close_numba(coords, intens, min_distance)
        same = np.array_equal(a, b)
        print(f"  n={n:>5}  kept={a.size:>5}  identical={same}")
        assert same, f"MISMATCH at n={n}: numba port is not equivalent"
    print("  -> equivalent on all sizes\n")

    # --- warm up the JIT once (compile cost is one-time, cached to disk) --------
    _c, _i = _make_field(16, seed=1)
    t_compile, _ = _timed(lambda: _suppress_close_numba(_c, _i, min_distance))
    print(f"first-call JIT (compile+run, one-time, cached): {t_compile*1e3:.0f} ms\n")

    # --- timing table -----------------------------------------------------------
    print(f"{'points':>8} {'kept':>7} {'numpy(ms)':>11} {'numba(ms)':>11} {'speedup':>9}")
    for n in counts:
        coords, intens = _make_field(n, seed=n)
        # best-of-3 to damp scheduler noise
        t_np = min(_timed(lambda: _suppress_close_numpy(coords, intens, min_distance))[0]
                   for _ in range(3))
        t_nb = min(_timed(lambda: _suppress_close_numba(coords, intens, min_distance))[0]
                   for _ in range(3))
        kept = _suppress_close_numba(coords, intens, min_distance).size
        speed = (t_np / t_nb) if t_nb > 0 else float("inf")
        print(f"{n:>8} {kept:>7} {t_np*1e3:>11.2f} {t_nb*1e3:>11.2f} {speed:>8.1f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
