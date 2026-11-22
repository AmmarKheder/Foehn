#!/usr/bin/env python3
"""A14 K4 (protocol_amendments/A14_infinite_ensemble.md, section 6): checks of run R1, CPU only, no station or monitor file.
check16all.py of a13/warn16 restricted to the plan of R1 (its functions imported unchanged):
  1. completeness: the 298 forecasts (leads 1 and 3 on the 149 start days of 2022 not in a13/warn16/data/run16) are done
     once, each with its target day: 294 in a14/data/run16_13 (jobs/pairs13.txt) and 2:1 2:3 4:1 4:3 in
     a13/warn16/data/run16all (task 0 of job 23898998);
  2. members 0-3 against the stored 4 members of eval_pop_crps_e01 for every one of them (bit identity, max / 99.99th
     percentile / mean |difference|, finite pattern, residents whose 'at least 2 of 4' warning differs);
  3. the forecasts sampled again in a fresh process (a13/warn16/data/run16all_check) against a13/warn16/data/run16;
  4. per lead: non-finite cells, population-weighted ensemble mean and standard deviation.
    python check13.py [workers] [--test: print only] -> results/check13.json, results/check13.txt"""
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE.parent / "a13" / "warn16"
sys.path.insert(0, str(W))
import check16all as c16  # noqa: E402  (unchanged)

TASK0 = [(2, 1), (2, 3), (4, 1), (4, 3)]
# noise design (a14/DESIGN_SWITCH.md): unset = as stored; 'check'/'check0' = the stored runs, output in results_<D>; 'ar' = A16
# run G11 (a16/data/run16, the 358 forecasts of jobs/pairs_g11.txt, the 30 seen start days among them) against the members of G2
# (eval_pop_crps_e01_ar): completeness and members 0-3; there are no fresh-process forecasts of the 'ar' design (part 3 empty)
DESIGN = os.environ.get("FOEHN_DESIGN", "")
assert DESIGN in ("", "ar", "check", "check0"), f"FOEHN_DESIGN={DESIGN!r}"
AR, OUTD = DESIGN == "ar", HERE / (f"results_{DESIGN}" if DESIGN else "results")
A16 = HERE.parent / "a16"
if AR:
    c16.EVP = c16.EVP.parent / (c16.EVP.name + "_ar")               # read by c16.stored_index and c16.one at call time


def main():
    test = "--test" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--test"]
    workers = int(argv[0]) if argv else 8
    t0 = time.time()
    plan = json.loads((W / "results" / "plan16all.json").read_text())
    if AR:                                                             # A16 G11: all 179 start days x leads 1 and 3
        want = [(d, h) for d in plan["start_days"] for h in (1, 3)]
        pg = [tuple(int(v) for v in it.split(":")) for line in (A16 / "jobs" / "pairs_g11.txt").read_text().split("\n") if line
              for it in line.split()]
        assert len(want) == 358 and len(pg) == 358 and set(pg) == set(want)
        c16.S["ALL"] = c16.index(A16 / "data" / "run16")
    else:
        unseen = [d for d in plan["start_days"] if d not in set(plan["done_in_run16"])]
        want = [(d, h) for d in unseen for h in (1, 3)]
        assert len(want) == 298
        pairs13 = [tuple(int(v) for v in it.split(":")) for line in (HERE / "jobs" / "pairs13.txt").read_text().split("\n") if line
                   for it in line.split()]
        assert len(pairs13) == 294 and set(pairs13) | set(TASK0) == set(want) and not set(pairs13) & set(TASK0)
        a = c16.index(HERE / "data" / "run16_13")
        b = {k: v for k, v in c16.index(W / "data" / "run16all").items() if k in TASK0}
        assert not set(a) & set(b)
        c16.S["ALL"] = {**a, **b}
    c16.S["K"] = c16.stored_index()
    c16.S["pop"] = np.load(c16.G / "pop_pixels.npz")["pop"].astype(np.float64)
    assert np.array_equal(np.load(c16.G / "pop_pixels.npz")["idx"], np.load(c16.EVP / "pop_idx.npy"))
    have = c16.S["ALL"]
    R = dict(what=__doc__, forecasts_planned=len(want), done=sum(1 for k in want if k in have),
             missing=[k for k in want if k not in have], extra=[k for k in have if k not in set(want)],
             wrong_target=[k for k in have if have[k][2] != k[0] + k[1]])
    keys = sorted(k for k in have if k in set(want))
    with get_context("fork").Pool(workers) as pool:
        rows = pool.map(c16.one, keys, chunksize=1)
    vs = [r for r in rows if "vs_stored" in r]
    ident = [r for r in vs if all(r["vs_stored"][f"m{m}"]["bit_identical_share"] == 1.0 for m in range(4))]
    R["vs_stored"] = dict(
        forecasts=len(vs), bit_identical_forecasts=len(ident),
        not_identical=[(r["start"], r["lead"]) for r in vs if r not in ident],
        min_bit_identical_share=min((r["vs_stored"][f"m{m}"]["bit_identical_share"] for r in vs for m in range(4)), default=None),
        worst_max_abs=max((r["vs_stored"][f"m{m}"]["max_abs"] for r in vs for m in range(4)), default=None),
        worst_p9999_abs=max((r["vs_stored"][f"m{m}"]["p9999_abs"] for r in vs for m in range(4)), default=None),
        finite_pattern_equal_everywhere=all(r["vs_stored"][f"m{m}"]["finite_pattern_equal"] for r in vs for m in range(4)),
        warning_change_pct_of_warned={f"{t:g}": 100 * sum(r["warning_2of4"][f"{t:g}"]["residents_with_a_different_warning_M"] for r in vs)
                                      / max(sum(r["warning_2of4"][f"{t:g}"]["residents_warned_old_M"] for r in vs), 1e-9) for t in c16.THR})
    chk = c16.index(W / "data" / "run16all_check") if (W / "data" / "run16all_check").exists() and not AR else {}
    r16 = c16.index(W / "data" / "run16")
    R["fresh_process_vs_run16"] = {}
    for key, (p, n, t) in sorted(chk.items()):
        x = np.asarray(np.load(p, mmap_mode="r")[n], np.float16)
        p2, n2, t2 = r16[key]; assert t2 == t
        y = np.asarray(np.load(p2, mmap_mode="r")[n2], np.float16)
        c = {f"m{m}": c16.cmp(x[:, m], y[:, m]) for m in range(16)}
        R["fresh_process_vs_run16"][f"{key[0]}:{key[1]}"] = dict(
            all_16_bit_identical=all(v["bit_identical_share"] == 1.0 for v in c.values()),
            min_share=min(v["bit_identical_share"] for v in c.values()), max_abs=max(v["max_abs"] for v in c.values()))
    if AR:
        R["fresh_process_note"] = ("design 'ar': no fresh-process forecasts exist (A16 checks C2(b) and C3 compare members 0-3 of G11 "
                                   "with G2); part 3 is empty and does not enter 'passed'")
    R["by_lead"] = {str(h): dict(forecasts=sum(1 for r in rows if r["lead"] == h),
                                 nonfinite_cells_max=max((r["nonfinite_cells"] for r in rows if r["lead"] == h), default=None),
                                 pop_mean_ensemble_mean=float(np.mean([r["pop_mean_ensemble_mean"] for r in rows if r["lead"] == h] or [np.nan])),
                                 pop_mean_ensemble_sd=float(np.mean([r["pop_mean_ensemble_sd"] for r in rows if r["lead"] == h] or [np.nan])))
                    for h in (1, 3)}
    R["passed"] = bool(not R["missing"] and not R["extra"] and not R["wrong_target"]
                       and R["vs_stored"]["forecasts"] == len(want) and R["vs_stored"]["finite_pattern_equal_everywhere"]
                       and all(v["all_16_bit_identical"] for v in R["fresh_process_vs_run16"].values()))
    R["rows"] = rows
    summ = {k: v for k, v in R.items() if k not in ("what", "rows")}
    if test:
        print(json.dumps(summ, indent=1), flush=True)
        return
    OUTD.mkdir(exist_ok=True)
    for name, txt in (("check13.json", json.dumps(R, indent=1)), ("check13.txt", json.dumps(summ, indent=1))):
        out = OUTD / name; tmp = out.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "w") as f_:
            f_.write(txt); f_.flush(); os.fsync(f_.fileno())
        assert tmp.read_text() == txt; os.replace(tmp, out)
    print(json.dumps(summ, indent=1), f"\ndone {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
