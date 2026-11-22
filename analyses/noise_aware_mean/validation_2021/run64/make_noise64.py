#!/usr/bin/env python3
"""A14 run R2: G/noise_tiles64.npy, one global N(0, 1) field per member for 64 members, by the recipe of
experiments/global1km/prep_test.py ('noise' branch, read only): member m = numpy default_rng(1000 + m), 29 rows of float32
standard normals of shape (71, 512, 512), stored as float16, in a .npy written like prep_test.create.
  * members 0-15: copied byte for byte from noise_tiles16.npy, then compared with it byte for byte;
  * members 0 and 15: generated again with the recipe and asserted byte-identical to noise_tiles16.npy (the recipe is the one
    of the stored fields);
  * members 16-63: generated with the recipe;
  * checks: every member's row 0 differs from the previous member's, mean and sd of row 0 of every member.
No forecast, observation or station is read. CPU only.
    python make_noise64.py [processes] -> G/noise_tiles64.npy, a14/r2/results/noise64.json"""
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
OUT, SRC = G / "noise_tiles64.npy", G / "noise_tiles16.npy"
SHAPE, DESCR = (64, 29, 71, 512, 512), "<f2"
HERE = Path(__file__).resolve().parent


def recipe(m):
    rng = np.random.default_rng(1000 + m)
    for r in range(29):
        yield r, rng.standard_normal((71, 512, 512), dtype=np.float32).astype(np.float16)


def work(m):
    out = np.load(OUT, mmap_mode="r+")
    src = np.load(SRC, mmap_mode="r")
    res = dict(member=m)
    if m < 16:
        for r in range(29):
            out[m, r] = src[m, r]
        out.flush()
        res["copy_equal"] = all(np.array_equal(np.asarray(out[m, r]).view(np.uint16), np.asarray(src[m, r]).view(np.uint16))
                                for r in range(29))
        if m in (0, 15):
            res["recipe_equal_stored"] = all(np.array_equal(x.view(np.uint16), np.asarray(src[m, r]).view(np.uint16))
                                             for r, x in recipe(m))
    else:
        for r, x in recipe(m):
            out[m, r] = x
        out.flush()
    r0 = np.asarray(out[m, 0], np.float32)
    res["row0_mean"], res["row0_sd"] = float(r0.mean()), float(r0.std())
    if m > 0:
        res["row0_differs_from_previous"] = not np.array_equal(np.asarray(out[m, 0]).view(np.uint16),
                                                               np.asarray(out[m - 1, 0]).view(np.uint16))
    del out, src
    print("member", m, time.strftime("%H:%M:%S"), flush=True)
    return res


def main():
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t0 = time.time()
    assert np.load(SRC, mmap_mode="r").shape == (16,) + SHAPE[1:]
    if not OUT.exists():
        with open(OUT, "wb") as f:
            np.lib.format.write_array_header_1_0(f, {"descr": DESCR, "fortran_order": False, "shape": SHAPE})
            hdr = f.tell(); f.truncate(hdr + int(np.prod(SHAPE)) * np.dtype(DESCR).itemsize)
    assert np.load(OUT, mmap_mode="r").shape == SHAPE
    with get_context("fork").Pool(procs) as pool:
        rows = pool.map(work, range(64), chunksize=1)
    R = dict(what=__doc__, file=str(OUT), shape=SHAPE, seconds=round(time.time() - t0),
             members0_15_copy_equal=all(r["copy_equal"] for r in rows if r["member"] < 16),
             recipe_equal_stored_members_0_15=all(r["recipe_equal_stored"] for r in rows if r["member"] in (0, 15)),
             all_rows0_distinct=all(r["row0_differs_from_previous"] for r in rows if r["member"] > 0),
             row0_mean_range=[min(r["row0_mean"] for r in rows), max(r["row0_mean"] for r in rows)],
             row0_sd_range=[min(r["row0_sd"] for r in rows), max(r["row0_sd"] for r in rows)], rows=rows)
    R["passed"] = bool(R["members0_15_copy_equal"] and R["recipe_equal_stored_members_0_15"] and R["all_rows0_distinct"]
                       and abs(R["row0_mean_range"][0]) < 1e-2 and abs(R["row0_mean_range"][1]) < 1e-2
                       and 0.99 < R["row0_sd_range"][0] and R["row0_sd_range"][1] < 1.01)
    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / "noise64.json"; tmp = out.with_suffix(f".tmp{os.getpid()}"); txt = json.dumps(R, indent=1)
    with open(tmp, "w") as f_:
        f_.write(txt); f_.flush(); os.fsync(f_.fileno())
    assert tmp.read_text() == txt; os.replace(tmp, out)
    print(json.dumps({k: v for k, v in R.items() if k not in ("what", "rows")}, indent=1), flush=True)
    sys.exit(0 if R["passed"] else 1)


if __name__ == "__main__":
    main()
