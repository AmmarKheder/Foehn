#!/usr/bin/env python3
"""Checks of the smoke job of A11 experiment M (jobs/smoke_job.sh); CPU, reads only files of a11/stageM/runs.

  smoke RUN_DIR STEP VAL_LAG_JSON   test (iii): the smoke run trained to STEP with one checkpoint and one in-training lag
        validation, everything finite; val_lag.py on that checkpoint (two days of 2021, a few thousand pixels, eight
        members, lags 1 and 5) is finite and its M numbers equal the in-training validation's (two calls of the same
        function on the same pixels and noise); prints the time per step and the cost of the full training.
  a3 VAL_LAG_JSON   the frozen arms on the full SEL set: val_lag.py must reproduce the SEL numbers that A3 stored for
        the same checkpoints (runs_ghap/pmS_final/README.md, computed by runs_ghap/val_st.py with train_aq.validate):
        pmG_base step 22,000: CRPS 3.349, beyond 10 km 3.343, within 10 km 5.113; pmS_base step 18,000 with the monitors
        of the day (lag 0): 3.267, 3.267, 3.217. The numbers are typed here from that README (3 decimals).
  -> tests/out/check_<what>_<tag>.json, exit 0 = pass
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A3_README = {"pmG_final": {"all": 3.349, "far": 3.343, "near": 5.113},          # runs_ghap/pmS_final/README.md
             "pmS_final": {"all": 3.267, "far": 3.267, "near": 3.217}}
FAR = ("10-50km", "50-150km", ">150km")


def write(name, rec):
    out = HERE / "out" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    print(f"[check_smoke] {'PASS' if rec['passed'] else 'FAIL'} -> {out}", flush=True)
    sys.exit(0 if rec["passed"] else 1)


def smoke(run_dir, step, vl_json):
    run_dir, step = Path(run_dir), int(step)
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl") if l.strip()]
    r = [x for x in rows if x["step"] == step][-1]
    tw = r["train_window"]
    ck = run_dir / "ckpt" / f"step{step:06d}.ckpt"
    args = json.loads((run_dir / "args.json").read_text())
    v = json.loads(Path(vl_json).read_text())
    z = np.load(str(Path(vl_json).with_suffix("")) + "_members.npz")
    arm = f"M@{step}"
    mem = {k: dict(shape=list(z[k].shape), finite=bool(np.isfinite(z[k]).all()), min=float(z[k].min()), median=float(np.median(z[k])),
                   max=float(z[k].max())) for k in z.files}
    m_keys = [k for k in z.files if k.startswith(arm)]
    diffs = {}
    for lg in r["lag"]["lags"]:
        for lb_t, lb_v in (("model", arm), ("model|nostations", arm + "|nostations")):
            for c, a in r["lag"]["scores"][lb_t][str(lg)].items():
                b = v["scores"][lb_v][str(lg)][c]
                if a is None or b is None:
                    assert a is None and b is None, (lg, lb_t, c)
                    continue
                for key in ("crps_fair", "rmse_ensmean", "spread_skill_fortin", "n"):
                    diffs[f"{lb_t}/lag{lg}/{c}/{key}"] = abs(a[key] - b[key])
    table = {lb: {lg: {c: (None if s is None else {k: s.get(k) for k in ("n", "crps_fair", "rmse_ensmean", "spread_skill_fortin")})
                       for c, s in by.items()} for lg, by in v["scores"][lb].items()} for lb in v["scores"]}   # 1-member arms: no spread
    ms = tw.get("ms_per_step")
    checks = {
        "trained_to_step": r["step"] == step and math.isfinite(tw.get("loss", float("nan"))),
        "one_checkpoint": ck.exists(),
        "lag_run": args.get("stn_lag") == "1,2,3,4,5" and args.get("stn_dim") == 63,
        "in_training_validation_finite": bool(r["lag"]["info"]["members_finite"] and r["lag"]["info"]["scores_finite"]
                                              and math.isfinite(r["model"]["ug"]["crps_fair"])),
        "no_heldout_block_in_metrics": "stations" not in r,
        "val_lag_all_finite": bool(v["all_finite"]),
        "eight_members_two_days_lags_1_5": v["info"]["members"] == 8 and len(v["info"]["days"]) == 2 and v["lags"] == [1, 5]
                                           and all(d.startswith("2021-") for d in v["info"]["days"]),
        "members_finite_and_positive": all(m["finite"] and m["min"] >= 0 for m in mem.values()) and len(m_keys) == 2 * 2 * 2,
        "val_lag_equals_in_training_validation": max(diffs.values()) <= 1e-3,
        "arms_present": {arm, arm + "|nostations", "pmS_final", "pmS_final|nostations", "pmG_final"} <= set(v["scores"]),
    }
    est = None
    if ms:                                                   # time per step of the last window of the smoke (steps 81-100)
        wait = tw.get("data_wait_ms") or 0.0
        est = dict(ms_per_step_smoke_1gcd=ms, data_wait_ms=wait, compute_ms_per_step=ms - wait, max_mem_gib=tw.get("max_mem_gib"),
                   loader_mb_s=tw.get("fill_mb_s"), in_training_validation_seconds_smoke=r["val_seconds"],
                   train_hours_24000_steps_at_compute_time=24000 * (ms - wait) / 3.6e6,
                   train_hours_24000_steps_at_smoke_time=24000 * ms / 3.6e6,
                   note="the first hundreds of steps wait for the loader (cold page cache: pmS_base logged 539-588 ms/step "
                        "at steps 10-100, then 355-361 ms/step with no data wait from step 300 to 24,000 on 8 GCDs); the "
                        "compute time per step, without the data wait, is the number to compare with pmS_base")
    rec = dict(passed=all(checks.values()), checks=checks, run=str(run_dir), step=step, loss=tw.get("loss"), cost=est,
               windows=[dict(step=x["train_window"].get("step"), ms_per_step=x["train_window"].get("ms_per_step")) for x in rows],
               max_abs_diff_vs_in_training=max(diffs.values()), in_training=dict(crps=r["model"]["ug"]["crps_fair"],
               rmse_ensmean=r["model"]["ug"]["rmse_ensmean"], ssF=r["model"]["ug"].get("spread_skill_fortin")),
               val_lag_scores=table, members=mem, class_weight=v["info"]["class_weight"], n_assim_valid=v["info"]["n_assim_valid"])
    write(f"check_smoke_{run_dir.name}.json", rec)


def a3(vl_json):
    v = json.loads(Path(vl_json).read_text())
    got, diffs = {}, {}
    for arm, want in A3_README.items():
        lg = "0"                                             # the monitors of the day: A3's own validation
        by = v["scores"][arm][lg]
        far_n = sum(by[c]["n"] for c in FAR if by[c] is not None)
        far = sum(by[c]["n"] * by[c]["crps_fair"] for c in FAR if by[c] is not None) / far_n
        got[arm] = {"all": by["all"]["crps_fair"], "far": far, "near": by["0-10km"]["crps_fair"], "n_near": by["0-10km"]["n"],
                    "n_far": far_n}
        for k in ("all", "far", "near"):
            diffs[f"{arm}/{k}"] = got[arm][k] - want[k]
    rec = dict(passed=max(abs(x) for x in diffs.values()) <= 1.5e-3 and bool(v["all_finite"]), a3_readme=A3_README, val_lag=got,
               difference=diffs, tolerance="1.5e-3 (the README gives 3 decimals)", a3_weights="near10km 5809, far10km 1737399",
               file=str(vl_json), seconds=v["seconds"])
    write("check_a3_consistency.json", rec)


if __name__ == "__main__":
    {"smoke": smoke, "a3": a3}[sys.argv[1]](*sys.argv[2:])
