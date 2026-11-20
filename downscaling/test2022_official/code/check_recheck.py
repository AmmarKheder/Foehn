#!/usr/bin/env python3
"""A7(g) check 3, confirmation: preds/ (full run, nid005282) against preds_recheck/ (second computation of 24 days on
another node, job 23865411) and both against the frozen test's stored members, per day. Predictions only.
    python -I check_recheck.py -> results/check_recheck.json"""
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
    for it, l in enumerate(open(f)):
        rank_of[json.loads(l)["t"]] = int(f[-8:-6])
meta = np.load(O / "pos/st_meta_official.npz")
fidx = meta["frozen_idx"]
sh = fidx >= 0
A = np.load(O / "preds/sameday_ds.npy", mmap_mode="r")
B = np.load(O / "preds_recheck/sameday_ds.npy", mmap_mode="r")
done = np.where(np.load(O / "preds_recheck/done_sameday.npy"))[0]


def cmp(x, y):
    fin = np.isfinite(x) & np.isfinite(y)
    assert np.array_equal(np.isfinite(x), np.isfinite(y))
    xf, yf = x.astype(np.float32)[fin], y.astype(np.float32)[fin]
    rel = np.abs(xf - yf) / np.maximum(yf, 1.0)
    idm = (x.view(np.uint16) == y.view(np.uint16)) | ~fin
    return {"frac_bit_identical": float(idm.mean()), "min_member_frac_identical": float(idm.reshape(idm.shape[0], -1).mean(1).min()),
            "max_abs_diff_ug": float(np.abs(xf - yf).max()), "max_rel_diff": float(rel.max()), "n_rel_gt_2steps": int((rel > 0.0022).sum())}


rows = []
for t in done:
    d = Y0 + dt.timedelta(int(t))
    a, b = np.asarray(A[t]), np.asarray(B[t])                       # (8, n_pix, 4)
    y = np.load(F / "preds_st" / "cdffm" / f"{d}.npz")["members"][:, fidx[sh]]
    rows.append({"t": int(t), "date": str(d), "frozen_test_rank": rank_of[int(t)],
                 "full_vs_recheck_all_pixels_and_rings": cmp(a, b),
                 "full_vs_recheck_shared_pixels": cmp(a[:, sh, 0], b[:, sh, 0]),
                 "full_vs_frozen_shared_pixels": cmp(a[:, sh, 0], y), "recheck_vs_frozen_shared_pixels": cmp(b[:, sh, 0], y)})
r30 = [r for r in rows if r["frozen_test_rank"] == 30]
oth = [r for r in rows if r["frozen_test_rank"] != 30]
summ = lambda rr, k: {"days": len(rr), "min_frac_bit_identical": min(r[k]["frac_bit_identical"] for r in rr),            # noqa: E731
                      "min_member_frac_identical": min(r[k]["min_member_frac_identical"] for r in rr),
                      "max_abs_diff_ug": max(r[k]["max_abs_diff_ug"] for r in rr), "max_rel_diff": max(r[k]["max_rel_diff"] for r in rr),
                      "n_rel_gt_2steps": sum(r[k]["n_rel_gt_2steps"] for r in rr)}
res = {"recheck_job": 23865411, "days": [int(t) for t in done],
       "days_of_frozen_rank_30": {k: summ(r30, k) for k in ("full_vs_recheck_all_pixels_and_rings", "full_vs_recheck_shared_pixels",
                                                           "full_vs_frozen_shared_pixels", "recheck_vs_frozen_shared_pixels")},
       "control_days": {k: summ(oth, k) for k in ("full_vs_recheck_all_pixels_and_rings", "full_vs_recheck_shared_pixels",
                                                 "full_vs_frozen_shared_pixels", "recheck_vs_frozen_shared_pixels")},
       "rows": rows}
for k in ("sameday_cams0", "sameday_bcsd", "sameday_climens", "sameday_climmed"):
    x, y = np.asarray(np.load(O / f"preds/{k}.npy", mmap_mode="r")[done]), np.asarray(np.load(O / f"preds_recheck/{k}.npy", mmap_mode="r")[done])
    res[f"{k}_full_vs_recheck"] = cmp(x.reshape(-1, *x.shape[2:]), y.reshape(-1, *y.shape[2:]))
(O / "results/check_recheck.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
for r in rows:
    print(r["t"], r["date"], "frozen rank", r["frozen_test_rank"], "| full vs recheck:", r["full_vs_recheck_shared_pixels"]["min_member_frac_identical"], r["full_vs_recheck_shared_pixels"]["max_rel_diff"],
          "| full vs frozen:", round(r["full_vs_frozen_shared_pixels"]["min_member_frac_identical"], 4), round(r["full_vs_frozen_shared_pixels"]["max_rel_diff"], 4),
          "| recheck vs frozen:", round(r["recheck_vs_frozen_shared_pixels"]["min_member_frac_identical"], 4), round(r["recheck_vs_frozen_shared_pixels"]["max_rel_diff"], 4))
