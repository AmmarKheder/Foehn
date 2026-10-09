#!/usr/bin/env python3
"""Storage re-layout of the global 1 km arrays for fast window reads (the model is unchanged: this is only how the
bytes are ordered on disk). A (14400, 36000) day map is padded to (14848, 36352) and stored as 29 x 71 contiguous blocks
of 512 x 512 px, so that a 4192 x 6992 window is read as ~10 contiguous runs of blocks instead of 4192 strided rows.
  ghap1km_tiles.npy  (D, 29, 71, 512, 512) float16    (0 = sea / no retrieval, as ghap1km_global.npy)
  static1km_tiles.npy (2, 29, 71, 512, 512) float16   (elevation (z - 300) / 500, land mask)
    python build_fine_tiles.py init | static | <task> <ntasks>"""
import sys
import time
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
H, W, T = 14400, 36000, 512
NR, NC = -(-H // T), -(-W // T)                 # 29, 71
SRC, DST = G / "ghap1km_global.npy", G / "ghap1km_tiles.npy"


def header(p):
    with open(p, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); return f.tell()


def to_tiles(a):
    p = np.zeros((NR * T, NC * T), np.float16)
    p[:a.shape[0], :a.shape[1]] = a
    return np.ascontiguousarray(p.reshape(NR, T, NC, T).transpose(0, 2, 1, 3))


def create(path, shape):
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": "<f2", "fortran_order": False, "shape": shape})
        hdr = f.tell(); f.truncate(hdr + int(np.prod(shape)) * 2)


cmd = sys.argv[1]
if cmd == "init":
    with open(SRC, "rb") as f:
        np.lib.format.read_magic(f); D = np.lib.format.read_array_header_1_0(f)[0][0]
    create(DST, (D, NR, NC, T, T)); print("created", DST, D)
elif cmd == "static":
    elev = np.load(G / "elev1km_global.npy", mmap_mode="r"); land = np.load(G / "land1km_global.npy", mmap_mode="r")
    np.save(G / "static1km_tiles.npy", np.stack([to_tiles(np.asarray(elev)), to_tiles(np.asarray(land).astype(np.float16))]))
    print("static tiles written")
else:
    task, n = int(cmd), int(sys.argv[2])
    hs, hd = header(SRC), header(DST)
    with open(SRC, "rb") as f:
        np.lib.format.read_magic(f); D = np.lib.format.read_array_header_1_0(f)[0][0]
    t0 = time.time()
    for d in range(task, D, n):
        a = np.fromfile(SRC, dtype=np.float16, count=H * W, offset=hs + d * H * W * 2).reshape(H, W)   # one contiguous read
        out = np.memmap(DST, dtype=np.float16, mode="r+", offset=hd + d * NR * NC * T * T * 2, shape=(NR, NC, T, T))
        out[:] = to_tiles(a); out.flush(); del out
        if (d // n) % 10 == 0:
            print(f"task {task}: day {d} {time.time() - t0:.0f}s", flush=True)
    print("done", task)
