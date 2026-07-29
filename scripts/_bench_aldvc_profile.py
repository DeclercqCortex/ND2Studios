"""Profiler — where does ALDVC (`aldvc_field.run_aldvc`) actually spend time?

WHY THIS EXISTS
    The IC-GN subset loop (`_icgn_subset`, called per-subset × up to icgn_max_iter
    iterations) is the theorised numba candidate: thousands of tiny numpy ops
    inside a Python loop. But each iteration also calls scipy
    `map_coordinates(order=3)` to resample the deformed volume — which numba
    CANNOT replace. This script measures the split so the port decision is made
    on evidence, not a hunch:

      * If `map_coordinates` dominates `_icgn_subset`'s self-time, a numba port of
        the surrounding vector math buys little — attack the resample instead.
      * If the surrounding assembly (pos/residual/solve) dominates, the staged
        numba port (keep map_coordinates in scipy, njit the rest) is worth it.

    Runs the SERIAL path (n_workers=1) on purpose: cProfile can't see into
    ProcessPool workers, so the parallel path would hide the very functions we
    want to weigh.

Manual (NOT in nodegraph.selftest — it is a profiling tool):
    PYTHONUTF8=1 python scripts/_bench_aldvc_profile.py            # 2-D DIC (fast)
    PYTHONUTF8=1 python scripts/_bench_aldvc_profile.py --3d       # 3-D DVC (the real cost case)
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _speckle(shape, sigma: float, seed: int) -> np.ndarray:
    """A smoothed-noise speckle pattern — the texture DVC needs to correlate."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    a = gaussian_filter(rng.random(shape).astype(np.float64), sigma=sigma)
    a -= a.min()
    return (a / (a.max() or 1.0)).astype(np.float32)


def _warp(ref: np.ndarray, disp: float) -> np.ndarray:
    """Warp `ref` by a smooth sub-voxel displacement (a gentle sinusoidal shear)
    via cubic sampling, so the solver has a real, spatially-varying field to find."""
    from scipy.ndimage import map_coordinates
    grids = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in ref.shape],
                        indexing="ij")
    # a smooth field: shift along the last axis modulated by the slowest axis.
    slow = grids[0] / max(1, ref.shape[0] - 1)
    coords = [g.copy() for g in grids]
    coords[-1] = coords[-1] + disp * np.sin(2 * np.pi * slow)
    warped = map_coordinates(ref.astype(np.float64), np.array(coords),
                             order=3, mode="nearest")
    return warped.astype(np.float32)


def _tot(stats: pstats.Stats, needle: str) -> tuple[float, float, int]:
    """Sum (tottime, cumtime, ncalls) over profiled functions whose (file,func)
    label contains `needle`."""
    tot = cum = 0.0
    ncalls = 0
    for (fn, _ln, name), (cc, nc, tt, ct, _cs) in stats.stats.items():  # type: ignore[attr-defined]
        label = f"{fn}:{name}"
        if needle in label:
            tot += tt
            cum += ct
            ncalls += int(cc)
    return tot, cum, ncalls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--3d", dest="d3", action="store_true",
                    help="profile a small 3-D volume (the dominant-cost case)")
    args = ap.parse_args()

    from nodegraph.kernels.aldvc_field import run_aldvc

    if args.d3:
        shape = (20, 96, 96)          # small but genuinely volumetric
        voxel = (2.0, 0.2, 0.2)
        params = dict(subset_size=16, subset_spacing=10, seed_levels=2,
                      admm_iterations=2, icgn_max_iter=60, n_workers=1)
    else:
        shape = (256, 256)
        voxel = (0.2, 0.2)
        params = dict(subset_size=24, subset_spacing=12, seed_levels=2,
                      admm_iterations=2, icgn_max_iter=80, n_workers=1)

    ref = _speckle(shape, sigma=1.2, seed=7)
    dfm = _warp(ref, disp=1.5)
    print(f"volume shape={shape}  ndim={len(shape)}  params={params}")

    # Warm-up on a tiny volume OUTSIDE the profiler: forces the one-time scipy.signal
    # / scipy.stats imports (~0.8 s cold) so they don't swamp the measured compute.
    print("warming imports (tiny throwaway run)...")
    _w = _speckle((8,) * len(shape), sigma=1.0, seed=1)
    run_aldvc(_w, _w.copy(), voxel, dict(subset_size=6, subset_spacing=4,
                                         seed_levels=1, admm_iterations=1,
                                         icgn_max_iter=5, n_workers=1))
    print("running run_aldvc under cProfile (serial path)...\n")

    prof = cProfile.Profile()
    prof.enable()
    result = run_aldvc(ref, dfm, voxel, params)
    prof.disable()

    stats = pstats.Stats(prof)
    grid = result.displacement_field.shape[:-1]
    print(f"done. grid={grid}  max|disp|={np.nanmax(np.abs(result.displacement_field)):.3f} vox\n")

    total = stats.total_tt  # type: ignore[attr-defined]

    print("── time by stage (tottime = self time, excludes callees) ──")
    print(f"{'component':<34} {'tottime(s)':>11} {'cumtime(s)':>11} {'calls':>9} {'%tot':>6}")
    probes = [
        ("local_icgn (whole IC-GN sweep)", ":local_icgn"),
        ("_icgn_subset (assembly self)", ":_icgn_subset"),
        ("map_coordinates (scipy resample)", "map_coordinates"),
        ("_prepare_reference (SD/Hessian)", "_prepare_reference"),
        ("np.linalg.solve/inv (small)", "linalg"),
        ("integer_search (FFT seed)", ":integer_search"),
        ("fftconvolve (seed NCC)", "fftconvolve"),
        ("global solve (factorized/splu)", "factoriz"),
        ("gradient/median/inpaint (numpy/scipy)", "outliers"),
    ]
    for label, needle in probes:
        tt, ct, nc = _tot(stats, needle)
        pct = (100.0 * tt / total) if total else 0.0
        print(f"{label:<34} {tt:>11.3f} {ct:>11.3f} {nc:>9} {pct:>5.1f}%")
    print(f"{'TOTAL (wall, profiled)':<34} {total:>11.3f}\n")

    # The load-bearing ratio for the port decision. Compare the WHOLE IC-GN sweep
    # (local_icgn cumtime = L) against the scipy resample inside it (map_coordinates
    # cumtime = M). Everything in L except M is Python glue + small numpy ops that a
    # staged numba port could take over; M is the piece numba cannot touch.
    _, L, _ = _tot(stats, ":local_icgn")
    _, M, _ = _tot(stats, "map_coordinates")
    addressable = max(0.0, L - M)
    print("── port-decision ratio (IC-GN sweep) ──")
    print(f"  local_icgn total (the whole sweep):              {L:.3f}s")
    print(f"  map_coordinates within it (numba CANNOT touch):  {M:.3f}s")
    print(f"  numba-addressable assembly (sweep - resample):   {addressable:.3f}s")
    if L > 0:
        share = 100.0 * addressable / L
        print(f"  => ~{share:.0f}% of IC-GN is numba-addressable vector assembly; "
              f"~{100-share:.0f}% is the scipy resample.")
        print("     (high % => stage-port worth it; low % => attack the resample/GPU)")

    print("\n── top 15 by cumulative time ──")
    stats.sort_stats("cumulative").print_stats(15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
