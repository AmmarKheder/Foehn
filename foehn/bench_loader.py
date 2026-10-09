#!/usr/bin/env python3
"""Checks of the block re-layout before relaunching the training:
1. exactness: random windows read from ghap1km_tiles.npy / static1km_tiles.npy equal the original arrays bit for bit;
2. speed: seconds per complete training sample (GlobalDataFP), one reader and 8 concurrent readers (one node)."""
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fullflow_global import FH, FW, G, GH, GW, Npy, header, tile_window  # noqa: E402
from fullflow_global_fp import GlobalDataFP  # noqa: E402

rng = np.random.default_rng(0)
h0 = header(G / "ghap1km_global.npy"); FT, ST = Npy(G / "ghap1km_tiles.npy"), Npy(G / "static1km_tiles.npy")
elev, land = np.load(G / "elev1km_global.npy", mmap_mode="r"), np.load(G / "land1km_global.npy", mmap_mode="r")
bad = 0
for _ in range(3):
    d = int(rng.integers(0, 2191)); r0 = 25 * int(rng.integers(0, (GH - FH) // 25)); c0 = 25 * int(rng.integers(0, (GW - FW) // 25))
    old = np.asarray(np.memmap(G / "ghap1km_global.npy", dtype=np.float16, mode="r", offset=h0 + d * GH * GW * 2, shape=(GH, GW))[r0:r0 + FH, c0:c0 + FW])
    new = tile_window(FT, d, r0, c0)
    e = tile_window(ST, 0, r0, c0); l = tile_window(ST, 1, r0, c0)
    ok = np.array_equal(old, new) and np.array_equal(np.asarray(elev[r0:r0 + FH, c0:c0 + FW]), e) and \
        np.array_equal(np.asarray(land[r0:r0 + FH, c0:c0 + FW]).astype(np.float16), l)
    bad += not ok
    print(f"day {d} window ({r0}, {c0}): identical = {ok}", flush=True)
print("EXACTNESS", "OK" if bad == 0 else f"FAILED ({bad})", flush=True)

ds = GlobalDataFP([2017, 2018, 2019, 2020], train=True, max_lead=7)


def timed(n):
    np.random.seed(n)
    t = time.time()
    for k in range(4):
        ds[int(np.random.randint(len(ds)))]
    return (time.time() - t) / 4


print(f"one reader: {timed(1):.2f} s per sample", flush=True)
with mp.Pool(8) as p:
    t = p.map(timed, range(8))
print(f"8 concurrent readers: {np.mean(t):.2f} s per sample (max {np.max(t):.2f}) -> one node feeds "
      f"{8 * 6 / np.mean(t):.1f} samples/s with 6 workers per GPU (need ~2.6 samples/s at 3.1 s/step)", flush=True)
