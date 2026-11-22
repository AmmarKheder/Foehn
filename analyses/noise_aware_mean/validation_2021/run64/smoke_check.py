#!/usr/bin/env python3
"""A14 run R2, smoke test (section 5 of protocol_amendments/A14_infinite_ensemble.md), CPU only, no station or monitor file.
  (a) eval_global_green.py (unchanged), --year 2021 --pop, four members, start day 9, lead 1, checkpoint ckpts/epoch=01.ckpt
      (the one of eval2021_sel_crps_e01, info field): map25, map25_m0, map25_sd and every column of the region sums against
      eval2021_sel_crps_e01 for that forecast;
  (b) sample_m.py --year 2021 --noise noise_tiles64.npy, four members, final.ckpt, the same forecast: its members at the
      populated pixels against those of (a) (pop_0.npy). This also tests that final.ckpt samples as epoch=01.ckpt, and that
      fields 0-3 of noise_tiles64.npy are those of noise_tiles.npy;
  truth: ghap1km_tiles.npy at the populated pixels for the target day of (a) against (a)'s pop_ghap (the truth R2 uses;
         reported, does not gate the run: the scorer can take the truth as eval_global_green does).
Tolerances (fixed with this file, before the smoke ran): the reproduction level that A13 accepted for the same sampler on
another node (a13/warn16/results/check_repro_run16.json: smallest bit-identical share 0.286, largest |difference| 2.25 ug m-3)
plus mean |difference| <= 0.05 ug m-3; region sums sse, crps and var within 0.5 %. A wrong year, noise, checkpoint, step
count or particle draw moves the members by their spread (several ug m-3 on average) and fails these.
    python smoke_check.py -> a14/r2/results/smoke_check.json; exit 1 if a test fails"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
           "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
A_DIR = HERE / "smoke_run" / RUN.name / "smoke2021"
BASE = 365 + 365 + 365 + 366


def find(parts, key):
    for p in parts:
        z = np.load(p)
        for n, k in enumerate(z["keys"]):
            if (int(k[0]), int(k[1])) == key:
                return z, n, int(k[2])
    raise KeyError(key)


def cmp(a, b):
    a, b = np.asarray(a), np.asarray(b)
    fa, fb = np.isfinite(a.astype(np.float32)), np.isfinite(b.astype(np.float32))
    both = fa & fb
    d = np.abs(a.astype(np.float32)[both] - b.astype(np.float32)[both])
    same = (a.view(np.uint16) == b.view(np.uint16)) if a.dtype == np.float16 and b.dtype == np.float16 else (a == b)
    return dict(finite_pattern_equal=bool(np.array_equal(fa, fb)), bit_identical_share=float(same.mean()),
                max_abs=float(d.max()) if d.size else 0.0, mean_abs=float(d.mean()) if d.size else 0.0,
                share_above_0p5=float((d > 0.5).mean()) if d.size else 0.0)


def ok_members(c):
    return c["finite_pattern_equal"] and c["bit_identical_share"] >= 0.25 and c["max_abs"] <= 2.5 and c["mean_abs"] <= 0.05


def main():
    R = {"what": __doc__}
    key = (9, 1)
    za, na, ta = find(sorted(A_DIR.glob("part_*.npz")), key)
    zs, ns, ts = find(sorted(glob.glob(str(RUN / "eval2021_sel_crps_e01" / "part_*.npz"))), key)
    assert ta == ts == 10
    R["a_info"], R["stored_info"] = [str(x) for x in za["info"]], [str(x) for x in zs["info"]]
    R["a_vs_stored_maps"] = {f: cmp(za[f][na], zs[f][ns]) for f in ("map25", "map25_m0", "map25_sd", "map25_ghap")}
    cols = [str(c) for c in zs["rec_cols"]]
    assert cols == [str(c) for c in za["rec_cols"]]
    ra = {int(r[2]): r for r in za["recs"] if (int(r[0]), int(r[1])) == key}
    rs = {int(r[2]): r for r in zs["recs"] if (int(r[0]), int(r[1])) == key}
    assert set(ra) == set(rs) and ra
    rel = {}
    for c in cols[3:]:
        j = cols.index(c)
        rel[c] = max(abs(ra[g][j] - rs[g][j]) / max(abs(rs[g][j]), 1e-9) for g in rs)
    R["a_vs_stored_region_sums_max_rel"] = rel
    # (b) against (a) at the populated pixels
    pa = [p for p in sorted(A_DIR.glob("pop_*.npy")) if p.stem.split("_")[1].isdigit()]      # not pop_idx, pop_ghap_*
    assert len(pa) == 1, pa
    A = np.load(pa[0], mmap_mode="r")
    assert A.shape[0] == 1 and A.shape[2] == 4
    kb = np.load(HERE / "smoke" / "keys4_b9.npy")
    assert [tuple(int(v) for v in k) for k in kb] == [(9, 1, 10)]
    B = np.load(HERE / "smoke" / "pop4_b9.npy", mmap_mode="r")
    R["b_vs_a_members"] = {f"m{m}": cmp(np.asarray(B[0, :, m]), np.asarray(A[0, :, m])) for m in range(4)}
    # truth used by R2: ghap1km_tiles at the populated pixels, against (a)'s pop_ghap
    pg = sorted(A_DIR.glob("pop_ghap_*.npy"))
    R["truth"] = None
    if pg:
        idx = np.load(G / "pop_pixels.npz")["idx"]
        r, c = idx // 36000, idx % 36000
        T = np.load(G / "ghap1km_tiles.npy", mmap_mode="r")
        day = T[BASE + ta]
        t = np.asarray(day[r // 512, c // 512, r % 512, c % 512], np.float16)
        t = np.where(t.astype(np.float32) > 0, t, np.float16(np.nan))
        g = np.asarray(np.load(pg[0], mmap_mode="r")[0])
        R["truth"] = cmp(t.astype(np.float16), g.astype(np.float16))
    R["passed"] = bool(
        all(ok_members(v) for k, v in R["a_vs_stored_maps"].items() if k != "map25_ghap")
        and R["a_vs_stored_maps"]["map25_ghap"]["max_abs"] == 0.0
        and all(rel[c] <= 0.005 for c in ("sse", "crps", "var"))
        and all(ok_members(v) for v in R["b_vs_a_members"].values()))
    R["truth_ok"] = bool(R["truth"] is not None and R["truth"]["finite_pattern_equal"] and R["truth"]["max_abs"] == 0.0)   # reported, not gating
    out = HERE / "results" / "smoke_check.json"; out.parent.mkdir(exist_ok=True)
    txt = json.dumps(R, indent=1); tmp = out.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w") as f_:
        f_.write(txt); f_.flush(); os.fsync(f_.fileno())
    assert tmp.read_text() == txt; os.replace(tmp, out)
    print(json.dumps({k: v for k, v in R.items() if k != "what"}, indent=1), flush=True)
    sys.exit(0 if R["passed"] else 1)


if __name__ == "__main__":
    main()
