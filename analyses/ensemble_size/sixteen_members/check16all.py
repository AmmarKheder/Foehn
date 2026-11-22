#!/usr/bin/env python3
"""A13 warn16: checks of the 16-member run on all 179 start days (data/run16all), CPU only, no station or monitor file.
  1. completeness: every one of the 656 forecasts of jobs/run16all_pairs.txt is done once, with its target day;
  2. members 0-3 against the stored 4 members of eval_pop_crps_e01, for every lead-1 and lead-3 forecast (the stored run has
     no other lead): share of bit-identical values, max / 99.99th percentile / mean of |difference|, finite pattern, and the
     residents whose 'at least 2 of 4' warning differs (15, 37.5, 75 ug m-3), as check_repro.py;
  3. the forecasts sampled again in a fresh process (data/run16all_check, smoke task) against data/run16, all 16 members;
  4. per lead (1, 3, 5, 7): non-finite values, mean of the population-weighted ensemble mean and of the ensemble standard deviation
     (sanity of the new leads 5 and 7).
    python check16all.py [workers] [run dir, default data/run16all] [--test: print only, write nothing]
      -> results/check16all.json, results/check16all.txt"""
import glob
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVP = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
           "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/eval_pop_crps_e01"))
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
THR = (15.0, 37.5, 75.0)
S = {}


def index(D, M=16):
    idx = {}
    for fk in sorted(Path(D).glob(f"keys{M}_*.npy")):
        tag = fk.stem.split("_", 1)[1]
        done = set(json.loads((Path(D) / f"done{M}_{tag}.json").read_text()))
        for n, (d, h, t) in enumerate(np.load(fk)):
            if n in done:
                key = (int(d), int(h))
                assert key not in idx, f"forecast {key} twice"
                idx[key] = (str(Path(D) / f"pop{M}_{tag}.npy"), n, int(t))
    return idx


def stored_index():
    K = {}
    for p in glob.glob(str(EVP / "part_*.npz")):
        k = int(p.split("_")[-1].split(".")[0])
        for n, (d, h, t) in enumerate(np.load(p)["keys"]):
            K[(int(d), int(h))] = (k, n, int(t))
    return K


def cmp(a, b):
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    dd = np.abs(a[both].astype(np.float32) - b[both].astype(np.float32))
    return dict(bit_identical_share=float((a.view(np.uint16) == b.view(np.uint16)).mean()), finite_pattern_equal=bool(np.array_equal(fa, fb)),
                max_abs=float(dd.max()), p9999_abs=float(np.percentile(dd, 99.99)), mean_abs=float(dd.mean()))


def one(key):
    d, h = key
    p, n, t = S["ALL"][key]
    new = np.asarray(np.load(p, mmap_mode="r")[n], np.float16)
    pop = S["pop"]
    x = new.astype(np.float32)
    fin = np.isfinite(x).all(1)
    w = np.where(fin, pop, 0.0)
    mean, sd = x.mean(1), x.std(1)
    r = dict(start=d, lead=h, target=t, nonfinite_cells=int((~fin).sum()),
             pop_mean_ensemble_mean=float((w * np.where(fin, mean, 0)).sum() / w.sum()),
             pop_mean_ensemble_sd=float((w * np.where(fin, sd, 0)).sum() / w.sum()))
    if key in S["K"]:
        k, nn, t0 = S["K"][key]; assert t0 == t
        old = np.asarray(np.load(EVP / f"pop_{k}.npy", mmap_mode="r")[nn], np.float16)
        r["vs_stored"] = {f"m{m}": cmp(new[:, m], old[:, m]) for m in range(4)}
        wd = {}
        for thr in THR:
            wn = (new[:, :4].astype(np.float32) > thr).sum(1) >= 2
            wo = (old.astype(np.float32) > thr).sum(1) >= 2
            wd[f"{thr:g}"] = dict(residents_warned_old_M=float(pop[wo].sum() / 1e6), residents_with_a_different_warning_M=float(pop[wn != wo].sum() / 1e6))
        r["warning_2of4"] = wd
    return r


def main():
    TEST = "--test" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--test"]
    workers = int(argv[0]) if argv else 8
    RUN = HERE / (argv[1] if len(argv) > 1 else "data/run16all")
    t0 = time.time()
    pairs = [tuple(int(v) for v in it.split(":")) for line in (HERE / "jobs" / "run16all_pairs.txt").read_text().split("\n") if line
             for it in line.split()]
    assert len(pairs) == 656 and len(set(pairs)) == 656
    S["ALL"] = index(RUN)
    S["K"] = stored_index()
    S["pop"] = np.load(G / "pop_pixels.npz")["pop"].astype(np.float64)
    assert np.array_equal(np.load(G / "pop_pixels.npz")["idx"], np.load(EVP / "pop_idx.npy"))
    missing = [p for p in pairs if p not in S["ALL"]]
    extra = [p for p in S["ALL"] if p not in set(pairs)]
    wrong_target = [p for p in S["ALL"] if S["ALL"][p][2] != p[0] + p[1]]
    keys = sorted(S["ALL"])
    with get_context("fork").Pool(workers) as pool:
        rows = pool.map(one, keys, chunksize=1)
    R = dict(what=__doc__, pairs=len(pairs), done=len(S["ALL"]), missing=missing, extra=extra, wrong_target=wrong_target)
    vs = [r for r in rows if "vs_stored" in r]
    ident = [r for r in vs if all(r["vs_stored"][f"m{m}"]["bit_identical_share"] == 1.0 for m in range(4))]
    R["vs_stored"] = dict(
        forecasts=len(vs), bit_identical_forecasts=len(ident),
        not_identical=[(r["start"], r["lead"]) for r in vs if r not in ident],
        min_bit_identical_share=min((r["vs_stored"][f"m{m}"]["bit_identical_share"] for r in vs for m in range(4)), default=None),
        worst_max_abs=max((r["vs_stored"][f"m{m}"]["max_abs"] for r in vs for m in range(4)), default=None),
        worst_p9999_abs=max((r["vs_stored"][f"m{m}"]["p9999_abs"] for r in vs for m in range(4)), default=None),
        worst_mean_abs=max((r["vs_stored"][f"m{m}"]["mean_abs"] for r in vs for m in range(4)), default=None),
        finite_pattern_equal_everywhere=all(r["vs_stored"][f"m{m}"]["finite_pattern_equal"] for r in vs for m in range(4)),
        warning_change_pct_of_warned={f"{thr:g}": 100 * sum(r["warning_2of4"][f"{thr:g}"]["residents_with_a_different_warning_M"] for r in vs)
                                      / max(sum(r["warning_2of4"][f"{thr:g}"]["residents_warned_old_M"] for r in vs), 1e-9) for thr in THR},
        worst_warning_change_M={f"{thr:g}": max((r["warning_2of4"][f"{thr:g}"]["residents_with_a_different_warning_M"] for r in vs), default=None) for thr in THR})
    # 3. fresh process against data/run16 (all 16 members)
    chk = index(HERE / "data" / "run16all_check") if (HERE / "data" / "run16all_check").exists() else {}
    r16 = index(HERE / "data" / "run16")
    R["fresh_process_vs_run16"] = {}
    for key, (p, n, t) in sorted(chk.items()):
        a = np.asarray(np.load(p, mmap_mode="r")[n], np.float16)
        p2, n2, t2 = r16[key]; assert t2 == t
        b = np.asarray(np.load(p2, mmap_mode="r")[n2], np.float16)
        c = {f"m{m}": cmp(a[:, m], b[:, m]) for m in range(16)}
        k, nn, _ = S["K"][key]
        old = np.asarray(np.load(EVP / f"pop_{k}.npy", mmap_mode="r")[nn], np.float16)
        R["fresh_process_vs_run16"][f"{key[0]}:{key[1]}"] = dict(
            all_16_bit_identical=all(v["bit_identical_share"] == 1.0 for v in c.values()),
            min_share=min(v["bit_identical_share"] for v in c.values()), max_abs=max(v["max_abs"] for v in c.values()),
            members0_3_vs_stored_min_share=min(cmp(a[:, m], old[:, m])["bit_identical_share"] for m in range(4)),
            run16_members0_3_vs_stored_min_share=min(cmp(b[:, m], old[:, m])["bit_identical_share"] for m in range(4)))
    # 4. per lead
    R["by_lead"] = {str(h): dict(forecasts=sum(1 for r in rows if r["lead"] == h),
                                 nonfinite_cells_max=max((r["nonfinite_cells"] for r in rows if r["lead"] == h), default=None),
                                 pop_mean_ensemble_mean=float(np.mean([r["pop_mean_ensemble_mean"] for r in rows if r["lead"] == h] or [np.nan])),
                                 pop_mean_ensemble_sd=float(np.mean([r["pop_mean_ensemble_sd"] for r in rows if r["lead"] == h] or [np.nan])))
                    for h in (1, 3, 5, 7)}
    R["rows"] = rows
    if TEST:
        print(json.dumps({k: v for k, v in R.items() if k not in ("what", "rows", "missing")}, indent=1), f"\nmissing {len(missing)}", flush=True)
        return
    txt = json.dumps(R, indent=1)
    out = HERE / "results" / "check16all.json"; tmp = out.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w") as f_:
        f_.write(txt); f_.flush(); os.fsync(f_.fileno())
    assert tmp.read_text() == txt; os.replace(tmp, out)
    summ = {k: v for k, v in R.items() if k not in ("what", "rows")}
    t_ = json.dumps(summ, indent=1)
    out = HERE / "results" / "check16all.txt"; tmp = out.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w") as f_:
        f_.write(t_ + "\n"); f_.flush(); os.fsync(f_.fileno())
    os.replace(tmp, out)
    print(t_, flush=True)
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
