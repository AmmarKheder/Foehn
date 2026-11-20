#!/usr/bin/env python3
"""A8(e)-1: the 200-step seed-0 smoke of the A8 launcher against the first lines of the frozen training log.
Pre-specified criteria (protocol_amendments/A8_seed_replicates.md (e)-1): days drawn by each rank at steps 0, 1, 2
identical; model line identical; window-mean loss within 0.005 at step 10 and within 0.02 at steps 100 and 200.
stdlib only. Writes --out (json) and prints the table.

  python compare_smoke.py --smoke LOG --frozen runs_ghap/logs/train_ghap_train_23246364.out --out results/smoke_seed0.json"""
import argparse
import json
import re

DAY = re.compile(r"^\[rank (\d+)\] step (\d+) days \[(\d+), (\d+)\]")
STEP = re.compile(r"^\[step (\d+)\] loss ([0-9.]+) \| (\d+) ms/step over (\d+) steps.*grad norm mean ([0-9.e+-]+) max ([0-9.e+-]+).*max mem ([0-9.]+) GiB")
MODEL = re.compile(r"^\[model\] (.*)$")
TOL = {10: 0.005, 100: 0.02, 200: 0.02}


def parse(path):
    days, steps, model = {}, {}, None
    for line in open(path, errors="replace"):
        m = DAY.match(line)
        if m:
            days[(int(m.group(1)), int(m.group(2)))] = (int(m.group(3)), int(m.group(4)))
            continue
        m = STEP.match(line)
        if m:
            steps[int(m.group(1))] = dict(loss=float(m.group(2)), ms_per_step=int(m.group(3)), n=int(m.group(4)),
                                          grad_norm_mean=float(m.group(5)), grad_norm_max=float(m.group(6)),
                                          max_mem_gib=float(m.group(7)))
            continue
        m = MODEL.match(line)
        if m and model is None:
            model = m.group(1).strip()
    return days, steps, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d1, s1, m1 = parse(a.smoke)
    d0, s0, m0 = parse(a.frozen)
    keys = sorted(k for k in d0 if k[1] <= 2)
    days_equal = len(keys) == 24 and all(d1.get(k) == d0[k] for k in keys)
    # added 2026-11-19 after the first smoke: today's train_aq.py prints one more token in the model line
    # ("stations=off", A3 edit of 2026-11-02); the literal comparison is kept, the token-free one is reported next to it
    m1n = None if m1 is None else m1.replace(" stations=off", "")
    out = {"smoke_log": a.smoke, "frozen_log": a.frozen, "model_line_equal": m0 == m1,
           "model_line_equal_without_stations_token": m0 == m1n, "model_line_frozen": m0,
           "model_line_smoke": m1, "n_rank_step_day_draws_compared": len(keys), "days_equal": days_equal,
           "days_mismatch": [{"rank": k[0], "step": k[1], "frozen": d0[k], "smoke": d1.get(k)} for k in keys if d1.get(k) != d0[k]],
           "steps": {}}
    ok = days_equal and m0 == m1
    print(f"model line identical: {m0 == m1} (without the 'stations=off' token of today's print: {m0 == m1n})\ndays drawn at steps 0-2 by the 8 ranks identical: {days_equal} ({len(keys)} draws)")
    print("step | loss frozen | loss smoke | diff | tol | grad-norm mean frozen / smoke | ms/step frozen / smoke")
    for st in sorted(TOL):
        if st not in s0 or st not in s1:
            out["steps"][str(st)] = {"missing": True}
            ok = False
            print(f"{st:4d} | missing in {'frozen' if st not in s0 else 'smoke'} log")
            continue
        diff = s1[st]["loss"] - s0[st]["loss"]
        passed = abs(diff) <= TOL[st] + 5e-5          # the logs print 4 decimals
        ok &= passed
        out["steps"][str(st)] = {"frozen": s0[st], "smoke": s1[st], "loss_diff": diff, "tolerance": TOL[st], "pass": passed}
        print(f"{st:4d} | {s0[st]['loss']:.4f} | {s1[st]['loss']:.4f} | {diff:+.4f} | {TOL[st]} | {s0[st]['grad_norm_mean']:.3g} / "
              f"{s1[st]['grad_norm_mean']:.3g} | {s0[st]['ms_per_step']} / {s1[st]['ms_per_step']}")
    out["pass_A8e1"] = bool(ok)                    # literal criterion of the protocol note
    out["pass_A8e1_without_stations_token"] = bool(days_equal and m0 == m1n and all(
        v.get("pass", False) for v in out["steps"].values()))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"A8(e)-1 pass (literal): {ok}; with the print-format token removed: {out['pass_A8e1_without_stations_token']} -> {a.out}")


if __name__ == "__main__":
    main()
