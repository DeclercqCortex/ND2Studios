"""Keystone benchmark — provider atomic read granularity (V2.01 §H.1).

Answers the load-bearing question before writing the Phase-2a `provider.py`:
**is a sub-frame ROI read meaningfully cheaper than a whole-plane read?** If yes,
sub-frame tiling (b2nd blocks) earns its complexity; if not, the frame is atomic
and "selective full-res ROI" collapses to trivial (decode plane, slice in RAM).

It measures, on a real 6554×6554 uint16 plane from the sample ND2 (falling back to
a synthetic plane if the file / nd2 is absent):

  * **nd2 whole-plane read** — the frame-atomic baseline (reopen each iter = cold-ish).
  * **numpy memmap** — whole-plane page-in vs a 512² sub-region.
  * **tiled BigTIFF** (tifffile) — whole-plane vs a single-tile region via the zarr
    store (a real sub-frame-tiled path available today).
  * **Blosc2 b2nd** — double-partitioned block ROI vs whole-array, at block sizes
    256/512/1024 — **only if `blosc2` is installed** (else skipped with a hint).

Run:  python scripts/_bench_provider_granularity.py [path-to.nd2]

Caveats printed inline: on Windows the OS page cache warms after the first touch,
so absolute cold-read numbers are approximate; the **ratio** ROI/whole is the
decision signal and is robust to warming. This is a scaffold — the b2nd rows
populate once `pip install blosc2` is available.
"""
from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from typing import Callable, Optional, Tuple

import numpy as np

TILE = 512                         # ROI / block size under test
DEFAULT_ND2 = os.path.join("sample_data", "7.10.26_NileBlue_52-75.nd2")
_TMP = tempfile.gettempdir()


# ── timing ───────────────────────────────────────────────────────────────────

def timeit(fn: Callable[[], object], n: int = 5) -> Tuple[float, float]:
    """Return (min, median) seconds over ``n`` runs (min ≈ best-case, warm)."""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def _fmt(sec: float) -> str:
    return f"{sec * 1e3:8.2f} ms"


# ── load a real (or synthetic) 6554² uint16 plane ──────────────────────────────

def _read_nd2_plane(nd2_path: str) -> np.ndarray:
    """Materialize channel-0's 2D uint16 plane via the dask interface (``read_frame``
    segfaults on some files; ``to_dask`` is the safe path). Computed while the file
    is open (nd2 dask arrays reference the open handle)."""
    import nd2
    with nd2.ND2File(nd2_path) as f:
        sub = f.to_dask()
        while sub.ndim > 2:
            sub = sub[0]                            # first channel / plane
        return np.ascontiguousarray(np.asarray(sub)).astype(np.uint16, copy=False)


def load_plane(nd2_path: str) -> Tuple[np.ndarray, str]:
    if os.path.exists(nd2_path):
        try:
            plane = _read_nd2_plane(nd2_path)
            return plane, f"nd2 {os.path.basename(nd2_path)} {plane.shape}"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not read ND2 ({exc}); using synthetic plane")
    rng = np.random.default_rng(0)
    plane = rng.integers(0, 4096, size=(6554, 6554), dtype=np.uint16)
    return plane, f"synthetic {plane.shape}"


def _center_roi(shape: Tuple[int, int]) -> Tuple[int, int]:
    h, w = shape
    return (h // 2 - TILE // 2, w // 2 - TILE // 2)


# ── measurements ───────────────────────────────────────────────────────────────

def bench_nd2_frame_atomic(nd2_path: str) -> Optional[Tuple[float, float]]:
    """Cold-ish whole-plane read: reopen the file each iteration."""
    if not os.path.exists(nd2_path):
        return None
    try:
        import nd2  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    return timeit(lambda: _read_nd2_plane(nd2_path), n=3)


def bench_memmap(plane: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    path = os.path.join(_TMP, "_bench_plane.raw")
    plane.tofile(path)
    h, w = plane.shape
    y0, x0 = _center_roi(plane.shape)

    def whole():
        mm = np.memmap(path, dtype=np.uint16, mode="r", shape=(h, w))
        return np.array(mm)                          # force full page-in

    def roi():
        mm = np.memmap(path, dtype=np.uint16, mode="r", shape=(h, w))
        return np.array(mm[y0:y0 + TILE, x0:x0 + TILE])

    return timeit(whole), timeit(roi)


def bench_tiled_tiff(plane: np.ndarray
                     ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], int]]:
    try:
        import tifffile
    except Exception:  # noqa: BLE001
        return None
    path = os.path.join(_TMP, "_bench_plane_tiled.tif")
    tifffile.imwrite(path, plane, tile=(TILE, TILE), photometric="minisblack")
    size = os.path.getsize(path)
    h, w = plane.shape
    y0, x0 = _center_roi(plane.shape)

    def whole():
        return tifffile.imread(path)

    def roi():
        store = tifffile.imread(path, aszarr=True)
        try:
            import zarr
            z = zarr.open(store, mode="r")
            return np.array(z[y0:y0 + TILE, x0:x0 + TILE])
        finally:
            store.close()

    try:
        return timeit(whole), timeit(roi), size
    except Exception as exc:  # noqa: BLE001 — zarr may be absent
        print(f"[warn] tiled-TIFF ROI read skipped ({exc})")
        return timeit(whole), None, size  # type: ignore[return-value]


def bench_b2nd(plane: np.ndarray):
    try:
        import blosc2
    except Exception:  # noqa: BLE001
        return None
    h, w = plane.shape
    y0, x0 = _center_roi(plane.shape)
    rows = []
    for block in (256, 512, 1024):
        # chunk = 2× block per axis (a small multiple, per §H.2)
        arr = blosc2.asarray(
            plane, chunks=(min(2 * block, h), min(2 * block, w)),
            blocks=(block, block),
            cparams={"codec": blosc2.Codec.ZSTD, "filters": [blosc2.Filter.BITSHUFFLE]},
        )

        def whole(a=arr):
            return a[:]

        def roi(a=arr):
            return a[y0:y0 + TILE, x0:x0 + TILE]

        rows.append((block, timeit(whole), timeit(roi), arr.schunk.cratio))
    return rows


# ── 3D: z-range subvolume vs whole-volume vs single plane (V2.03 §5 D2 / H26) ──
#
# The 2D rows above decide TILED for the (y,x) plane. Directive B makes WHOLE_VOLUME
# (a full (Z,Y,X) per (m,t,c)) a first-class pull, so the store's **z-block shape**
# (bz) must be chosen on real signal, not inferred from 2D. This measures how a
# z-ranged subvolume brick read scales with bz — bz=1 (plane-partitioned) forces one
# decompress per z in the range; a z-spanning block collapses it to fewer.

Z_DEPTH = 32                        # a realistic confocal stack depth
SUBVOL_Z = 4                        # z-range of the brick under test
SUBVOL_XY = 256                     # (y,x) window of the brick


def bench_b2nd_3d(z_depth: int = Z_DEPTH, xy: int = 512):
    try:
        import blosc2
    except Exception:  # noqa: BLE001
        return None
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 4096, size=(z_depth, xy, xy), dtype=np.uint16)
    zc = z_depth // 2
    z0, z1 = zc - SUBVOL_Z // 2, zc + SUBVOL_Z // 2
    y0 = x0 = xy // 2 - SUBVOL_XY // 2
    rows = []
    for bz in sorted({1, 4, z_depth}):
        chunk_z = min(z_depth, max(bz, 4) * (1 if bz >= z_depth else 4))
        arr = blosc2.asarray(
            vol, chunks=(chunk_z, min(512, xy), min(512, xy)),
            blocks=(bz, 256, 256),
            cparams={"codec": blosc2.Codec.ZSTD, "filters": [blosc2.Filter.BITSHUFFLE]},
        )

        def whole(a=arr):
            return a[:]

        def subvol(a=arr):
            return a[z0:z1, y0:y0 + SUBVOL_XY, x0:x0 + SUBVOL_XY]

        def plane(a=arr):
            return a[zc, :, :]

        rows.append((bz, timeit(whole), timeit(subvol), timeit(plane),
                     arr.schunk.cratio))
    return vol.nbytes, rows


# ── report ─────────────────────────────────────────────────────────────────────

def main(argv) -> int:
    nd2_path = argv[1] if len(argv) > 1 else DEFAULT_ND2
    plane, label = load_plane(nd2_path)
    nbytes = plane.nbytes
    print(f"\nplane: {label}  dtype={plane.dtype}  {nbytes/1e6:.1f} MB  "
          f"(ROI = {TILE}×{TILE})\n")
    print(f"{'path':30} {'whole (min)':>12} {'ROI (min)':>12} {'ROI/whole':>10}  extra")
    print("-" * 84)

    def row(name, whole, roi, extra=""):
        if whole is None:
            print(f"{name:30} {'—':>12} {'—':>12} {'—':>10}  {extra}")
            return
        wmin = whole[0]
        if roi is None:
            print(f"{name:30} {_fmt(wmin):>12} {'—':>12} {'—':>10}  {extra}")
            return
        rmin = roi[0]
        ratio = rmin / wmin if wmin else float("nan")
        print(f"{name:30} {_fmt(wmin):>12} {_fmt(rmin):>12} {ratio:>9.1%}  {extra}")

    nd2w = bench_nd2_frame_atomic(nd2_path)
    row("nd2 whole-plane (atomic)", nd2w, None, "frame-atomic baseline")

    mw, mr = bench_memmap(plane)
    row("numpy memmap", mw, mr, "warm (page cache)")

    tiff = bench_tiled_tiff(plane)
    if tiff is not None:
        tw, tr, tsz = tiff
        row("tiled BigTIFF (tifffile)", tw, tr, f"on-disk {tsz/1e6:.1f} MB")
    else:
        row("tiled BigTIFF (tifffile)", None, None, "tifffile absent")

    b = bench_b2nd(plane)
    if b is None:
        print(f"{'blosc2 b2nd (block ROI)':30} {'—':>12} {'—':>12} {'—':>10}"
              "  NOT INSTALLED → pip install blosc2")
    else:
        for block, whole, roi, cratio in b:
            row(f"b2nd block={block}", whole, roi, f"cratio {cratio:.1f}×")

    # ── 3D z-range subvolume section (V2.03 §5 D2 / H26) ──────────────────────
    b3 = bench_b2nd_3d()
    if b3 is not None:
        vbytes, rows3 = b3
        print(f"\n3D volume: ({Z_DEPTH},512,512) uint16  {vbytes/1e6:.1f} MB  "
              f"(subvol = {SUBVOL_Z}z×{SUBVOL_XY}²,  plane = 1z×512²)")
        print(f"{'b2nd z-block (bz)':30} {'whole':>12} {'subvol':>12} "
              f"{'sv/whole':>9} {'plane':>10} {'pl/whole':>9}  extra")
        print("-" * 96)
        for bz, whole, sub, pl, cratio in rows3:
            wmin, smin, pmin = whole[0], sub[0], pl[0]
            sv = smin / wmin if wmin else float("nan")
            pr = pmin / wmin if wmin else float("nan")
            print(f"{'bz=' + str(bz):30} {_fmt(wmin):>12} {_fmt(smin):>12} "
                  f"{sv:>8.1%} {_fmt(pmin):>10} {pr:>8.1%}  cratio {cratio:.1f}×")

    print("\nVERDICT CRITERION (V2.01 §H.1):")
    print("  If a sub-frame ROI read is a SMALL FRACTION of the whole-plane read")
    print("  (ratio ≪ 1, e.g. <25%), sub-frame tiling (b2nd blocks) is justified →")
    print("  decision = TILED at the storage/memo layer. If the ROI read is ~as")
    print("  costly as the whole plane (ratio ≈ 1), the frame is effectively ATOMIC")
    print("  → skip sub-frame streaming for 2D; tiling is a pure 5D/memo concern.")
    print("  NOTE: install blosc2 to populate the decisive b2nd rows; memmap/TIFF")
    print("  numbers here are warm-cache and indicative only.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
