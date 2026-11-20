#!/usr/bin/env python3
"""A7(g) check 3 as amended on 2026-11-19 (protocol note, "Deviations"): the reference of the same-day members is
  (i)  the frozen test's stored members (test2022_ghap/preds_st/cdffm) on the 348 days computed by its 31 sound processes,
  (ii) a second, independent computation (preds_recheck/, job 23865411, another node) on the 17 days computed by rank 30
       of the frozen test's job 23246562, whose stored members carry sporadic errors (results/diag_check3_*.json).
Thresholds unchanged from code/check_preds.py: share of bit-identical float16 values >= 0.999 and largest relative
difference <= 0.002, in each part. The comparison with the frozen test on the 17 days is reported, not a criterion.
Ring means against test2022_ghap/preds_ring{1,3,10}/cdffm are reported per part (never a criterion: those were
averaged from float16 members). Predictions only: no observation, no GHAP value.
    python -I check3_split.py -> results/check3_split.json (exit code 1 if a part fails)"""
import datetime as dt
import glob
import json
from pathlib import Path
import numpy as np
ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
F = ROOT / "test2022_ghap"
Y0 = dt.date(2022, 1, 1)
rank_of = {}
for f in sorted(glob.glob(str(F / "logs/predict_r*.jsonl"))):
    for l in open(f):
        rank_of[json.loads(l)["t"]] = int(f[-8:-6])
assert len(rank_of) == 365
bad = sorted(t for t, r in rank_of.items() if r == 30)
meta = np.load(O / "pos/st_meta_official.npz")
fidx = meta["frozen_idx"]
sh = fidx >= 0
fz = np.load(F / "st_ring_2022.npz")
fpix = np.searchsorted(fz["mon_idx"], fidx[sh])
assert np.array_equal(fz["mon_idx"][fpix], fidx[sh])
A = np.load(O / "preds/sameday_ds.npy", mmap_mode="r")
B = np.load(O / "preds_recheck/sameday_ds.npy", mmap_mode="r")
doneB = np.load(O / "preds_recheck/done_sameday.npy").astype(bool)
assert np.load(O / "preds/done_sameday.npy").all() and all(doneB[t] for t in bad)


def new():
    return {"days": 0, "values": 0, "identical": 0, "max_abs_diff_ug": 0.0, "max_rel_diff": 0.0, "n_rel_gt_2steps": 0,
            "min_member_day_frac_identical": 1.0}


def add(acc, x, y):
    idm = x.view(np.uint16) == y.view(np.uint16)
    df = np.abs(x.astype(np.float32) - y.astype(np.float32))
    rel = df / np.maximum(y.astype(np.float32), 1.0)
    acc["days"] += 1
    acc["values"] += int(x.size)
    acc["identical"] += int(idm.sum())
    acc["max_abs_diff_ug"] = max(acc["max_abs_diff_ug"], float(df.max()))
    acc["max_rel_diff"] = max(acc["max_rel_diff"], float(rel.max()))
    acc["n_rel_gt_2steps"] += int((rel > 0.0022).sum())
    acc["min_member_day_frac_identical"] = min(acc["min_member_day_frac_identical"], float(idm.mean(1).min()))


P = {"sound_days_vs_frozen_test": new(), "rank30_days_vs_recheck": new(), "rank30_days_vs_frozen_test_REPORTED_ONLY": new(),
     "recheck_all_station_pixels_and_rings_vs_full_run": new()}
ring = {k: {f"ring{dd}": [0.0, 0.0] for dd in (1, 3, 10)} for k in ("sound_days", "rank30_days")}
for t in range(365):
    d = Y0 + dt.timedelta(t)
    x = np.asarray(A[t])
    y = np.load(F / "preds_st" / "cdffm" / f"{d}.npz")["members"][:, fidx[sh]]
    if t in bad:
        add(P["rank30_days_vs_recheck"], x[:, sh, 0], np.asarray(B[t][:, sh, 0]))
        add(P["rank30_days_vs_frozen_test_REPORTED_ONLY"], x[:, sh, 0], y)
        xa, ya = x.reshape(8, -1), np.asarray(B[t]).reshape(8, -1)
        fin = np.isfinite(xa[0])
        assert np.array_equal(np.isfinite(xa), np.isfinite(ya))
        add(P["recheck_all_station_pixels_and_rings_vs_full_run"], xa[:, fin], ya[:, fin])
    else:
        add(P["sound_days_vs_frozen_test"], x[:, sh, 0], y)
    for j, dd in enumerate((1, 3, 10)):
        r = np.load(F / f"preds_ring{dd}" / "cdffm" / f"{d}.npz")["members"].astype(np.float32)[:, fpix]
        m = x[:, sh, j + 1].astype(np.float32)
        both = np.isfinite(r) & np.isfinite(m)
        e = np.abs(r - m)[both]
        k = ring["rank30_days" if t in bad else "sound_days"][f"ring{dd}"]
        k[0] = max(k[0], float(e.max()))
        k[1] = max(k[1], float((e / np.maximum(r[both], 1.0)).max()))
for v in P.values():
    v["frac_bit_identical"] = v.pop("identical") / max(v["values"], 1)
fails = []
for k in ("sound_days_vs_frozen_test", "rank30_days_vs_recheck"):
    if P[k]["frac_bit_identical"] < 0.999 or P[k]["max_rel_diff"] > 0.002:
        fails.append(k)
res = {"thresholds": {"frac_bit_identical_min": 0.999, "max_rel_diff_max": 0.002, "as": "code/check_preds.py (unchanged)"},
       "frozen_test_job": 23246562, "frozen_test_rank_with_errors": 30, "days_of_that_rank": bad, "recheck_job": 23865411,
       "parts": P, "ring_means_vs_frozen_test_max_abs_ug_and_rel_REPORTED_ONLY": ring,
       "other_checks_of_check_preds": "5a, 5b, finiteness and equality with preds_smoke passed in results/check_preds_preds.json; "
                                      "B1 (cams0) and B2 (bcsd) are 100 % bit-identical to the frozen test's on all 365 days",
       "failed_parts": fails, "pass": not fails}
(O / "results/check3_split.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
if fails:
    raise SystemExit("FAILED: " + ", ".join(fails))
