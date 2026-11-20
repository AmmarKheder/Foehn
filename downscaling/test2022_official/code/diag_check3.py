#!/usr/bin/env python3
"""A7(g) check 3, diagnosis of the failure on the full run (no observation, no GHAP value: predictions only).
Where do sameday_ds (this amendment) and test2022_ghap/preds_st/cdffm (frozen test) differ: which days, members,
pixels, by how much, and at which values.
    python -I diag_check3.py [--preds preds] -> results/diag_check3_<preds>.json"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
F = ROOT / "test2022_ghap"
Y0 = dt.date(2022, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds")
    a = ap.parse_args()
    P = O / a.preds
    meta = np.load(O / "pos/st_meta_official.npz")
    fidx = meta["frozen_idx"]
    sh = fidx >= 0
    done = np.load(P / "done_sameday.npy")
    S = np.load(P / "sameday_ds.npy", mmap_mode="r")
    days = np.where(done)[0]
    per_day = []
    big = []                                   # (day, member, shared-pixel index, mine, theirs)
    hist_edges = np.array([0, 1e-9, 0.0011, 0.0022, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1e9])
    hist = np.zeros(hist_edges.size - 1, np.int64)
    per_mem = np.zeros((8, 2), np.int64)       # values differing by more than 2 float16 steps (rel > 0.0022), all
    per_pix = np.zeros(int(sh.sum()), np.int64)
    ens_mean_diff = []
    for t in days:
        d = Y0 + dt.timedelta(int(t))
        z = np.load(F / "preds_st" / "cdffm" / f"{d}.npz")
        y = z["members"][:, fidx[sh]]
        x = np.asarray(S[t][:, sh, 0])
        xf, yf = x.astype(np.float32), y.astype(np.float32)
        df = np.abs(xf - yf)
        rel = df / np.maximum(yf, 1.0)
        hist += np.histogram(rel, hist_edges)[0]
        nb = rel > 0.0022
        per_mem[:, 0] += nb.sum(1)
        per_mem[:, 1] += nb.shape[1]
        per_pix += nb.sum(0)
        emd = np.abs(xf.mean(0) - yf.mean(0))
        ens_mean_diff.append([float(emd.max()), float(emd.mean())])
        per_day.append({"t": int(t), "date": str(d), "frac_bit_identical": float((x.view(np.uint16) == y.view(np.uint16)).mean()),
                        "n_rel_gt_2steps": int(nb.sum()), "max_abs": float(df.max()), "max_rel": float(rel.max()),
                        "n_pix_any_big": int(nb.any(0).sum()), "members_with_big": [int(v) for v in np.where(nb.any(1))[0]],
                        "frozen_keys": sorted(z.files)})
        for m, k in zip(*np.where(rel > 0.01)):
            big.append([int(t), int(m), int(k), float(xf[m, k]), float(yf[m, k])])
    nbig = np.array([r["n_rel_gt_2steps"] for r in per_day])
    worst = sorted(per_day, key=lambda r: -r["n_rel_gt_2steps"])[:25]
    emd = np.array(ens_mean_diff)
    res = {"preds": str(P), "days": int(days.size), "n_shared_pixels": int(sh.sum()),
           "rel_diff_histogram": {f"({hist_edges[i]:g}, {hist_edges[i + 1]:g}]": int(hist[i]) for i in range(hist.size)},
           "values_total": int(hist.sum()),
           "days_with_any_rel_gt_2steps": int((nbig > 0).sum()),
           "n_rel_gt_2steps_total": int(nbig.sum()),
           "per_member_n_rel_gt_2steps": [int(v) for v in per_mem[:, 0]],
           "pixels_with_any_rel_gt_2steps": int((per_pix > 0).sum()),
           "pixels_top": [[int(k), int(per_pix[k])] for k in np.argsort(-per_pix)[:20]],
           "ens_mean_abs_diff_ug": {"max_over_days_of_max": float(emd[:, 0].max()), "mean_over_days_of_mean": float(emd[:, 1].mean())},
           "worst_days": worst,
           "n_rel_gt_0.01": len(big), "big_examples_day_member_pix_mine_theirs": big[:400]}
    (O / "results" / f"diag_check3_{a.preds}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("big_examples_day_member_pix_mine_theirs", "worst_days")}, indent=1))
    print("worst days:")
    for r in worst:
        print(r)
    print("first big examples (day, member, pix, mine, theirs):")
    for b in big[:60]:
        print(b)
    # cumulative count by day of the year, to see clustering
    print("n_rel_gt_2steps by day (t: n):", {int(r["t"]): r["n_rel_gt_2steps"] for r in per_day if r["n_rel_gt_2steps"] > 0})


if __name__ == "__main__":
    main()
