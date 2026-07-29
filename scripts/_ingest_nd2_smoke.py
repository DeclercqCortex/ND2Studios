"""Smoke test for the C4 ND2 ingest (nodelab_v2.ingest) against the real sample.

Heavy (realizes the full volume + writes an on-disk b2nd store) and file-dependent, so
it is NOT part of `python -m nodegraph.selftest` (which stays fast + file-free). Run
manually:  PYTHONUTF8=1 python scripts/_ingest_nd2_smoke.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE = os.path.join("sample_data", "7.10.26_NileBlue_52-75.nd2")


def main() -> int:
    if not os.path.exists(SAMPLE):
        print(f"SKIP: sample not found at {SAMPLE}")
        return 0
    import nd2

    from nodegraph.dataset import Dataset
    from nodegraph.engine import Engine
    from nodegraph.graph import Graph, NodeInstance
    from nodegraph.nodes import COMPUTES
    from nodegraph.registry import OutDataset, define_node
    from nodelab_v2.ingest import ingest_nd2, open_store, read_calibration

    # raw reference (channel 0, an 8×8 corner) straight from nd2
    with nd2.ND2File(SAMPLE) as f:
        raw = np.asarray(f.to_dask()[0, 0:8, 0:8])          # dims C,Y,X → channel 0

    d = tempfile.mkdtemp(prefix="nd2_ingest_")
    try:
        store = os.path.join(d, "nileblue")
        prov, env = ingest_nd2(SAMPLE, store_path=store, levels=2)

        # geometry + calibration
        assert env.axes.c == 2 and env.axes.y == 6554 and env.axes.x == 6554, env.axes
        assert env.axes.m == 1 and env.axes.t == 1 and env.axes.z == 1, env.axes
        cal = env.metadata
        print("calibration:", {k: cal.get(k) for k in
                                ("pixel_size_um", "z_step_um", "channel_emission_nm",
                                 "objective_na", "objective_magnification")})
        assert abs(cal["pixel_size_um"] - 1.7182777601481225) < 1e-6
        assert cal["objective_na"] == 0.45 and cal["objective_magnification"] == 10.0
        assert cal["channel_emission_nm"][0] == 649.0

        # pixels match the raw nd2 (disk provider, lazy read)
        got = prov.get_region(0, 0, 0, 0, 0, 0, 8, 0, 8)     # (m,t,z,c=0), y[0:8], x[0:8]
        assert np.array_equal(got, raw), "disk provider pixels != raw nd2"
        assert prov.levels == 2 and prov.fingerprint()[0] == "b2nd-disk"

        # re-open the store without re-ingesting → same pixels + mtime-based identity
        re = open_store(store)
        assert np.array_equal(re.get_region(0, 0, 0, 0, 0, 0, 8, 0, 8), raw)
        assert re.version[0] == "b2nd-disk" and re.version[2] == prov._mtime_ns

        # end-to-end through the Engine: Select Channel (→ ch 1) then Gamma, on real data
        define_node("io.nd2", "Load ND2", outputs=[OutDataset()])
        ds = Dataset(axes=env.axes, metadata=cal).with_image(prov)
        g = Graph()
        g.add(NodeInstance("S", "io.nd2"))
        g.add(NodeInstance("C", "channel.select", params={"channels": [1]}))
        g.add(NodeInstance("G", "enhance.gamma", params={"gamma": 2.0}))
        g.connect("S", "C"); g.connect("C", "G")
        eng = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env})
        out = eng.pull("G")
        assert out.axes.c == 1                               # channel selected
        plane = out.image.get_region(0, 0, 0, 0, 0, 0, 16, 0, 16)
        assert plane.shape == (16, 16) and np.all(np.isfinite(plane))
        print(f"end-to-end pull OK: selected 1 channel, gamma plane max={plane.max():.1f}")
    finally:
        import gc
        gc.collect()
        shutil.rmtree(d, ignore_errors=True)

    print("\nND2 INGEST SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
