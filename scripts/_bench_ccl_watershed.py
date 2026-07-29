"""C6 benchmark — whole-frame CCL / seeded-watershed cost.

Tests the V2.02 §7 "compute-once is affordable" assumption *before* trusting it at real
6554² sizes: it times the pure-numpy flood-fill CCL (`structure.label_components`, the
correctness baseline) against `scipy.ndimage.label` (C fast path) + `seeded_watershed`,
and extrapolates the flood-fill to 6554². The finding drives whether a scipy/cc3d fast
path is needed for whole-frame labelling.

Manual (NOT in `nodegraph.selftest` — it is a timing tool):
    PYTHONUTF8=1 python scripts/_bench_ccl_watershed.py            # quick (≤512²)
    PYTHONUTF8=1 python scripts/_bench_ccl_watershed.py --full     # adds 1024²/2048²/6554²
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: cap the O(fg-pixels) pure-python flood-fill above this edge (it does not scale).
_FLOODFILL_CAP = 512


def make_mask(n: int) -> np.ndarray:
    """A deterministic n×n mask: a regular grid of 2×2 foreground blobs on a 3-px pitch
    (≈ (n/3)² connected components) — a stress case for connected-components labelling."""
    m = np.zeros((n, n), dtype=np.int64)
    m[::3, ::3] = 1
    m[1::3, ::3] = 1
    m[::3, 1::3] = 1
    m[1::3, 1::3] = 1
    return m


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="add 1024²/2048²/6554²")
    sizes = [128, 256, 512] + ([1024, 2048, 6554] if ap.parse_args().full else [])

    from scipy import ndimage as ndi

    from nodegraph.structure import label_components, _label_components_flood, seeded_watershed

    # `floodfill` = the pure-numpy reference (`_label_components_flood`); `fast` = the SHIPPED
    # `label_components` (now scipy CCL + raster-canonical relabel + centroids); `scipy.label`
    # = raw `ndi.label` (no relabel/table), to show the relabel/table overhead on top.
    print(f"{'size':>6} {'fg':>10} {'floodfill(s)':>14} {'fast(s)':>10} "
          f"{'scipy.label(s)':>15} {'speedup':>9} {'watershed(s)':>13}")
    ff_per_fg = None
    for n in sizes:
        mask = make_mask(n)
        fg = int(mask.sum())

        ff = "  (skipped)"
        speedup = float("nan")
        if n <= _FLOODFILL_CAP:
            t_ff, _ = _timed(lambda: _label_components_flood(
                mask != 0, 8, m=0, t=0, c=0, z_index=0, layer=None))
            ff = f"{t_ff:14.4f}"
            ff_per_fg = t_ff / max(1, fg)                    # for the 6554² extrapolation

        t_fast, _ = _timed(lambda: label_components(mask, 8))  # the shipped fast path
        t_sc, _ = _timed(lambda: ndi.label(mask))             # raw scipy (no table)
        if n <= _FLOODFILL_CAP:
            speedup = float(ff.strip()) / t_fast

        # seeded watershed on the mask (EDT + marker-carrying split) at this size
        markers, _ = ndi.label(mask)                          # one marker per blob
        t_ws, _ = _timed(lambda: seeded_watershed(mask != 0, markers))

        print(f"{n:>6} {fg:>10} {ff:>14} {t_fast:10.4f} {t_sc:15.4f} "
              f"{speedup:9.1f} {t_ws:13.4f}")

    if ff_per_fg is not None:
        fg_6554 = int(make_mask(6554).sum())
        est = ff_per_fg * fg_6554
        print(f"\nExtrapolated pure-python flood-fill at 6554² (~{fg_6554} fg px): "
              f"~{est:.1f} s (why the fast path is needed). "
              f"`label_components` now ships the scipy fast path (C6 resolved).")
    print("\n(_label_components_flood stays the correctness baseline; the selftest asserts "
          "the scipy fast path is byte-identical to it.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
