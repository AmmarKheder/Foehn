#!/usr/bin/env python3
"""A7(g) checks 3 and 5 on a prediction directory (preds_smoke/ or preds/). No observation and no GHAP value is read:
only predictions (this amendment's and the frozen test's) and the rank logs.

  3  same-day members at the station pixels shared with the frozen test equal test2022_ghap/preds_st/cdffm/<date>.npz
     (and B1 cams_bilinear, B2 bcsd); ring means against test2022_ghap/preds_ring{1,3,10}/cdffm (those were averaged
     from float16 members, so small differences are expected there).
  5a camsfc_ds at lead day 1 equals sameday_ds of the same day (same channel 0, same seeds).
  5b gather() on the store's first-day CAMS field of v - 1 reproduces the stored transport channel 41 (rank logs).
  + timings from the rank logs, and (--against DIR) equality of the done units of two prediction directories.
    python -I check_preds.py <preds dir> [--tag smoke] [--against <other preds dir>]"""
import argparse
import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
F = ROOT / "test2022_ghap"
Y0, I0 = dt.date(2022, 1, 1), 120
ARMS_S = ("sameday_ds", "sameday_cams0", "sameday_bcsd", "sameday_climens", "sameday_climmed")
ARMS_C = ("chain", "chain_t6fc", "chain_rtseed", "camsfc_ds", "bcsd_fc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--against", default=None)
    ap.add_argument("--max-days", type=int, default=365)
    a = ap.parse_args()
    P = Path(a.preds)
    meta = np.load(O / "pos/st_meta_official.npz")
    fidx = meta["frozen_idx"]                                   # my station pixel -> index in the frozen 'pos' (or -1)
    sh = fidx >= 0
    fz = np.load(F / "st_ring_2022.npz")
    # frozen pixel-station index of each shared pixel (frozen mon_idx is sorted like its pixel list)
    fpix = np.searchsorted(fz["mon_idx"], fidx[sh])
    assert np.array_equal(fz["mon_idx"][fpix], fidx[sh])
    done_s, done_c = np.load(P / "done_sameday.npy"), np.load(P / "done_chain.npy")
    res = {"preds": str(P), "n_shared_station_pixels": int(sh.sum()), "sameday_days_done": int(done_s.sum()),
           "issue_days_done": int(done_c.sum())}
    S = {k: np.load(P / f"{k}.npy", mmap_mode="r") for k in ARMS_S}
    C = {k: np.load(P / f"{k}.npy", mmap_mode="r") for k in ARMS_C}
    days = np.where(done_s)[0][:a.max_days]
    chk3 = {"days": int(days.size)}
    for mine, theirs in (("sameday_ds", "cdffm"), ("sameday_cams0", "cams_bilinear"), ("sameday_bcsd", "bcsd")):
        mx, nid, ntot, mxrel = 0.0, 0, 0, 0.0
        ring = {dd: [0.0, 0.0] for dd in (1, 3, 10)}
        for t in days:
            d = Y0 + dt.timedelta(int(t))
            x = np.asarray(S[mine][t]).astype(np.float32)                     # (M, n_pix, 4)
            y = np.load(F / "preds_st" / theirs / f"{d}.npz")["members"]      # (M, n_frozen_pos) float16
            ys = y[:, fidx[sh]]
            xs = np.asarray(S[mine][t][:, sh, 0])
            ntot += xs.size
            nid += int((xs.view(np.uint16) == ys.view(np.uint16)).sum())
            df = np.abs(xs.astype(np.float32) - ys.astype(np.float32))
            mx = max(mx, float(df.max()))
            mxrel = max(mxrel, float((df / np.maximum(ys.astype(np.float32), 1.0)).max()))
            for j, dd in enumerate((1, 3, 10)):
                r = np.load(F / f"preds_ring{dd}" / theirs / f"{d}.npz")["members"].astype(np.float32)[:, fpix]
                m = x[:, sh, j + 1]
                both = np.isfinite(r) & np.isfinite(m)
                assert np.array_equal(np.isfinite(r), np.isfinite(m)), (mine, str(d), dd)
                e = np.abs(r - m)[both]
                ring[dd][0] = max(ring[dd][0], float(e.max()))
                ring[dd][1] = max(ring[dd][1], float((e / np.maximum(r[both], 1.0)).max()))
        chk3[mine] = {"vs": f"test2022_ghap/preds_st/{theirs}", "values": ntot, "frac_bit_identical": nid / max(ntot, 1),
                      "max_abs_diff_ug": mx, "max_rel_diff": mxrel,
                      "ring_means_max_abs_diff_ug_and_rel": {f"ring{dd}": v for dd, v in ring.items()}}
    res["check3_sameday_vs_frozen_test"] = chk3
    iss = np.where(done_c)[0]
    c5 = {"issue_days_with_sameday_day_done": 0, "max_abs_diff_ug": 0.0, "frac_bit_identical": None}
    nid = ntot = 0
    for i in iss:
        v = I0 + int(i)
        if not done_s[v]:
            continue
        x, y = np.asarray(C["camsfc_ds"][i, 0]), np.asarray(S["sameday_ds"][v])
        c5["issue_days_with_sameday_day_done"] += 1
        fin = np.isfinite(y)
        assert np.array_equal(fin, np.isfinite(x))
        nid += int((x.view(np.uint16) == y.view(np.uint16))[fin].sum())
        ntot += int(fin.sum())
        c5["max_abs_diff_ug"] = max(c5["max_abs_diff_ug"], float(np.abs(x.astype(np.float32) - y.astype(np.float32))[fin].max()))
    c5["frac_bit_identical"] = nid / max(ntot, 1)
    res["check5a_camsfc_ds_lead1_vs_sameday"] = c5
    rows = [json.loads(l) for f in sorted(glob.glob(str(O / "logs" / f"predict_{a.tag}_r*.jsonl"))) for l in open(f)]
    ch = [r for r in rows if r["mode"] == "chain"]
    sd = [r for r in rows if r["mode"] == "sameday"]
    t6 = [r["leads"]["1"]["t6_check"] for r in ch]
    res["check5b_gather_reproduces_stored_ch41"] = {"issue_days": len(t6), "max_abs_diff": max([x["max_abs_diff"] for x in t6], default=None),
                                                   "min_frac_bit_identical": min([x["frac_bit_identical"] for x in t6], default=None)}
    if ch:
        res["timing_chain_issue_day_s"] = {"n": len(ch), "median_total": float(np.median([r["total_s"] for r in ch])),
                                           "max_total": float(np.max([r["total_s"] for r in ch])),
                                           "median_parts": {k: float(np.median([r["times"][k] for r in ch])) for k in ch[0]["times"]},
                                           "median_prep": float(np.median([r["prep_s"] for r in ch]))}
        san = {}
        for arm in ("chain", "chain_t6fc", "chain_rtseed", "camsfc_ds"):
            v = [r["leads"][L][arm] for r in ch for L in r["leads"]]
            san[arm] = {"max_ug": max(x["max_ug"] for x in v), "n_at_pmax": sum(x["n_at_pmax"] for x in v),
                        "n_at_bound": sum(x["n_at_bound"] for x in v), "n_z0_trunc": sum(x["n_z0_trunc"] for x in v),
                        "all_finite": all(x["finite"] for x in v), "units": len(v)}
        res["sanity_forecast_arms"] = san
        res["t6fc_ch41_mean_minus_frozen_by_lead"] = {L: float(np.mean([r["leads"][L]["t6fc_ch41_mean_minus_frozen"] for r in ch]))
                                                      for L in ("1", "2", "3", "4", "5")}
    if sd:
        res["timing_sameday_day_s"] = {"n": len(sd), "median_total": float(np.median([r["total_s"] for r in sd])),
                                       "max_total": float(np.max([r["total_s"] for r in sd]))}
        res["sanity_sameday"] = {"max_ug": max(r["sanity"]["max_ug"] for r in sd), "n_at_pmax": sum(r["sanity"]["n_at_pmax"] for r in sd),
                                 "n_at_bound": sum(r["sanity"]["n_at_bound"] for r in sd), "n_z0_trunc": sum(r["sanity"]["n_z0_trunc"] for r in sd)}
    # descriptive (no truth): ensemble means and spreads of the arms at the station pixel
    desc = {}
    for k in ARMS_S:
        x = np.asarray(S[k][days][..., 0]).astype(np.float32)
        desc[k] = {"mean_ug": float(x.mean()), "member_sd_ug": float(x.std(1).mean()) if x.shape[1] > 1 else None}
    for k in ARMS_C:
        x = np.asarray(C[k][iss][..., 0]).astype(np.float32)                 # (n_issue, 5, M, n_pix)
        desc[k] = {"mean_ug_by_lead": [float(x[:, L].mean()) for L in range(5)],
                   "member_sd_ug_by_lead": [float(x[:, L].std(1).mean()) for L in range(5)]}
    res["descriptive_no_truth"] = desc
    if a.against:
        Q = Path(a.against)
        ds2, dc2 = np.load(Q / "done_sameday.npy"), np.load(Q / "done_chain.npy")
        eq = {}
        for k in ARMS_S:
            t = np.where(done_s & ds2)[0]
            x, y = np.asarray(S[k][t]), np.asarray(np.load(Q / f"{k}.npy", mmap_mode="r")[t])
            fin = np.isfinite(x)
            eq[k] = {"units": int(t.size), "frac_bit_identical": float((x.view(np.uint16) == y.view(np.uint16))[fin].mean()),
                     "max_abs_diff_ug": float(np.abs(x.astype(np.float32) - y.astype(np.float32))[fin].max())}
        for k in ARMS_C:
            t = np.where(done_c & dc2)[0]
            x, y = np.asarray(C[k][t]), np.asarray(np.load(Q / f"{k}.npy", mmap_mode="r")[t])
            fin = np.isfinite(x)
            eq[k] = {"units": int(t.size), "frac_bit_identical": float((x.view(np.uint16) == y.view(np.uint16))[fin].mean()),
                     "max_abs_diff_ug": float(np.abs(x.astype(np.float32) - y.astype(np.float32))[fin].max())}
        res[f"equal_to_{Q.name}"] = eq
    # A7(g): a failed check stops the chain (exit code 1) until it is understood; the file is written either way
    fails = []
    c3 = res["check3_sameday_vs_frozen_test"]
    if c3["days"] > 0:
        if c3["sameday_ds"]["frac_bit_identical"] < 0.999 or c3["sameday_ds"]["max_rel_diff"] > 0.002:
            fails.append("check 3: same-day members differ from the frozen test's")
        for k in ("sameday_cams0", "sameday_bcsd"):
            if c3[k]["frac_bit_identical"] < 0.9999:
                fails.append(f"check 3: {k} differs from the frozen test's")
    c5 = res["check5a_camsfc_ds_lead1_vs_sameday"]
    if c5["issue_days_with_sameday_day_done"] > 0 and (c5["frac_bit_identical"] < 0.999 or c5["max_abs_diff_ug"] > 0.5):
        fails.append("check 5a: camsfc_ds at lead day 1 differs from the same-day arm")
    c5b = res["check5b_gather_reproduces_stored_ch41"]
    if c5b["issue_days"] > 0 and c5b["min_frac_bit_identical"] < 0.99:
        fails.append("check 5b: gather() does not reproduce the stored transport channel 41")
    for arm, v in res.get("sanity_forecast_arms", {}).items():
        if not v["all_finite"]:
            fails.append(f"non-finite values in {arm}")
    for k, v in res.get(f"equal_to_{Path(a.against).name}" if a.against else "-", {}).items():
        if v["units"] > 0 and k != "sameday_climens" and (v["frac_bit_identical"] < 0.999 or v["max_abs_diff_ug"] > 1.0):
            fails.append(f"{k}: the two prediction directories differ")
    res["failed_checks"] = fails
    res["pass"] = not fails
    out = O / "results" / f"check_preds_{P.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    if fails:
        raise SystemExit("FAILED CHECKS: " + "; ".join(fails))


if __name__ == "__main__":
    main()
