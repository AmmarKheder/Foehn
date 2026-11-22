#!/usr/bin/env python3
"""A14a (protocol_amendments/A14a_check13_rounding.md, written 2026-11-21 before check13.py ran): the gate of run R1 in the
words of A14 section 5, from the output of check13.py (unchanged, job 23900798). CPU only, no station or monitor file.
check13.py sets 'passed' only if the fresh-process forecasts of a13/warn16/data/run16all_check are bit-identical to
a13/warn16/data/run16 in all 16 members. A13 had shown that this cannot hold (per-process GPU numerics,
a13/warn16/README.md), and check13.py does not judge members 0-3 against the stored members at a rounding level.
This file reads results/check13.json, compares the fresh-process forecasts again with check16all.cmp (imported unchanged),
and sets 'passed' by:
  1. completeness, as check13.py: the 298 forecasts done once with their target day, none missing or extra;
  2. members 0-3 against the stored members of eval_pop_crps_e01, every forecast and member: the same finite pattern,
     mean |difference| <= 0.05 ug m-3 and 99.99th percentile of |difference| <= 2.5 ug m-3 (A13 worst cases 0.034 and 1.0;
     a wrong noise, checkpoint, step count or particle draw moves the members by their spread, several ug m-3 on average);
  3. the fresh-process forecasts against run16, every one of the 16 members: the same rule.
Bit-identical shares and maxima are reported, not gated.
    python check13b.py          -> results/check13b.json, results/check13b.txt
    python check13b.py --test   -> part 3 only, printed; check13.json not read, nothing written"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE.parent / "a13" / "warn16"
sys.path.insert(0, str(W))
import check16all as c16  # noqa: E402  (unchanged)

MEAN_MAX, P9999_MAX = 0.05, 2.5
# noise design (a14/DESIGN_SWITCH.md): unset = as stored; 'check'/'check0' = the stored runs, results_<D>/; 'ar' = A16 run G11
# from results_ar/check13.json: rules 1-2 on the 358 forecasts (same tolerances); rule 3 has no forecasts under 'ar' (none run)
DESIGN = os.environ.get("FOEHN_DESIGN", "")
assert DESIGN in ("", "ar", "check", "check0"), f"FOEHN_DESIGN={DESIGN!r}"
AR, OUTD = DESIGN == "ar", HERE / (f"results_{DESIGN}" if DESIGN else "results")
NPLAN = 358 if AR else 298


def ok(c):
    return bool(c["finite_pattern_equal"] and c["mean_abs"] <= MEAN_MAX and c["p9999_abs"] <= P9999_MAX)


def summary(cs):
    cs = list(cs)
    return dict(n=len(cs), all_ok=all(ok(c) for c in cs),
                min_bit_identical_share=min(c["bit_identical_share"] for c in cs),
                worst_mean_abs=max(c["mean_abs"] for c in cs), worst_p9999_abs=max(c["p9999_abs"] for c in cs),
                worst_max_abs=max(c["max_abs"] for c in cs),
                finite_pattern_equal_everywhere=all(c["finite_pattern_equal"] for c in cs))


def main():
    test = "--test" in sys.argv
    t0 = time.time()
    R = dict(what=__doc__, rule=dict(mean_abs_max=MEAN_MAX, p9999_abs_max=P9999_MAX))
    # 3. fresh-process forecasts against run16 (A13 data, no R1 forecast); none under 'ar'
    chk = c16.index(W / "data" / "run16all_check") if not AR else {}
    r16 = c16.index(W / "data" / "run16")
    assert AR or sorted(chk) == [(0, 1), (12, 1)], sorted(chk)
    F = {}
    for key, (p, n, t) in sorted(chk.items()):
        x = np.asarray(np.load(p, mmap_mode="r")[n], np.float16)
        p2, n2, t2 = r16[key]
        assert t2 == t
        y = np.asarray(np.load(p2, mmap_mode="r")[n2], np.float16)
        c = {f"m{m}": c16.cmp(x[:, m], y[:, m]) for m in range(16)}
        F[f"{key[0]}:{key[1]}"] = dict(summary(c.values()), members=c)
    R["fresh_process_vs_run16"] = F
    R["fresh_process_ok"] = all(v["all_ok"] for v in F.values()) if not AR else None
    if AR:
        R["fresh_process_note"] = "design 'ar': no fresh-process forecasts (A16 C2(b)/C3 cover members 0-3 of G11 against G2); rule 3 not applicable"
    if test:
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "members"} for k, v in F.items()}, indent=1),
              "\nfresh_process_ok", R["fresh_process_ok"], f"\n{time.time() - t0:.0f} s", flush=True)
        return
    # 1-2. from the output of check13.py (unchanged)
    src = OUTD / "check13.json"
    C = json.loads(src.read_text())
    R["check13_json_mtime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(src.stat().st_mtime))
    R["check13_passed_as_coded"] = C["passed"]
    R["completeness"] = {k: C[k] for k in ("forecasts_planned", "done", "missing", "extra", "wrong_target")}
    rows = [r for r in C["rows"] if "vs_stored" in r]
    keys = [(r["start"], r["lead"]) for r in rows]
    assert len(keys) == len(set(keys))
    bad = [(r["start"], r["lead"]) for r in rows if not all(ok(r["vs_stored"][f"m{m}"]) for m in range(4))]
    R["members0_3_vs_stored"] = dict(summary(r["vs_stored"][f"m{m}"] for r in rows for m in range(4)),
                                     forecasts=len(rows), forecasts_failing=bad,
                                     bit_identical_forecasts=C["vs_stored"]["bit_identical_forecasts"],
                                     warning_change_pct_of_warned=C["vs_stored"]["warning_change_pct_of_warned"])
    R["by_lead"] = C["by_lead"]
    R["passed"] = bool(C["forecasts_planned"] == NPLAN and C["done"] == NPLAN and not C["missing"] and not C["extra"]
                       and not C["wrong_target"] and len(rows) == NPLAN and not bad and (AR or R["fresh_process_ok"]))
    summ = {k: v for k, v in R.items() if k != "what"}
    summ["fresh_process_vs_run16"] = {k: {kk: vv for kk, vv in v.items() if kk != "members"} for k, v in F.items()}
    for name, txt in (("check13b.json", json.dumps(R, indent=1)), ("check13b.txt", json.dumps(summ, indent=1))):
        out = OUTD / name
        tmp = out.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "w") as f_:
            f_.write(txt); f_.flush(); os.fsync(f_.fileno())
        assert tmp.read_text() == txt
        os.replace(tmp, out)
    print(json.dumps(summ, indent=1), f"\ndone {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
