#!/usr/bin/env python3
"""A7, description of the observations and of the target at the official stations (scoring side; no model score is
changed, nothing is selected). Added 2026-11-19 after the tables were read (protocol note, "Deviations"): (1) lagged
agreement of GHAP 2022 at the station pixel and of the CAMS first-day field with the station values (is the day of the
daily means the same?), (2) how much of the pooled squared error sits in a few station-days.
    python -I diag_obs_target.py -> results/diag_obs_target.json"""
import json
from pathlib import Path
import numpy as np
ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
NETS = {0: "Europe (EEA)", 1: "China (CNEMC)", 3: "USA (EPA)", 4: "Korea (AirKorea)", 5: "Japan (NIES)", 6: "India (CPCB)"}
st = np.load(G / "stations_2022.npz")
meta = np.load(O / "pos/st_meta_official.npz")
obs = st["daily"].astype(np.float32)[meta["id"]].T                 # (365, S)
src, pix = meta["src"], meta["pix"]
gh = np.load(O / "truth/ghap_official_2022.npy")[:, pix, 0]        # (365, S) GHAP at the station pixel
c0 = np.asarray(np.load(O / "preds/sameday_cams0.npy", mmap_mode="r")[:, 0, :, 0]).astype(np.float32)[:, pix]
ds = np.asarray(np.load(O / "preds/sameday_ds.npy", mmap_mode="r")[..., 0]).astype(np.float32).mean(1)[:, pix]


def lagstats(x, y, lag):
    """x (365, S) field, y (365, S) obs; compare x[t + lag] with y[t]"""
    if lag > 0:
        a, b = x[lag:], y[:-lag]
    elif lag < 0:
        a, b = x[:lag], y[-lag:]
    else:
        a, b = x, y
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(0)
    good = n >= 30
    with np.errstate(all="ignore"):
        A, B = np.where(ok, a, 0.0).astype(np.float64), np.where(ok, b, 0.0).astype(np.float64)
        r = ((A * B).sum(0) / n - A.sum(0) * B.sum(0) / n ** 2) / np.sqrt(((A * A).sum(0) / n - (A.sum(0) / n) ** 2) * ((B * B).sum(0) / n - (B.sum(0) / n) ** 2))
    d = (a - b)[ok]
    return {"median_station_r": float(np.nanmedian(r[good])), "rmse": float(np.sqrt((d ** 2).mean())),
            "share_within_0p5": float((np.abs(d) <= 0.5).mean()), "median_abs_err": float(np.median(np.abs(d)))}


res = {"lagged_agreement": {}, "observations": {}, "squared_error_concentration": {}}
for k, nm in NETS.items():
    m = src == k
    res["lagged_agreement"][nm] = {name: {str(lag): lagstats(x[:, m], obs[:, m], lag) for lag in (-2, -1, 0, 1, 2)}
                                   for name, x in (("ghap_pixel", gh), ("cams_first_day_bilinear", c0), ("downscaler_sameday", ds))}
    o = obs[:, m]
    f = np.isfinite(o)
    v = o[f]
    res["observations"][nm] = {"station_days": int(f.sum()), "mean": float(v.mean()), "median": float(np.median(v)),
                               "p99": float(np.percentile(v, 99)), "p99.9": float(np.percentile(v, 99.9)), "max": float(v.max()),
                               "n_gt_500": int((v > 500).sum()), "n_gt_1000": int((v > 1000).sum()), "n_le_0": int((v <= 0).sum())}
    conc = {}
    for name, x in (("ghap_pixel", gh), ("cams_first_day_bilinear", c0), ("downscaler_sameday", ds)):
        ok = f & np.isfinite(x[:, m])
        e2 = np.sort(((x[:, m] - o)[ok].astype(np.float64)) ** 2)[::-1]
        tot = e2.sum()
        ntop = max(1, int(round(0.001 * e2.size)))
        # per station: share of the network's squared error
        se_st = np.where(ok, (x[:, m] - o).astype(np.float64) ** 2, 0.0).sum(0)
        top_st = np.sort(se_st)[::-1]
        conc[name] = {"rmse": float(np.sqrt(tot / e2.size)), "share_of_squared_error_in_top_0p1pct_station_days": float(e2[:ntop].sum() / tot),
                      "rmse_without_top_0p1pct_station_days": float(np.sqrt(e2[ntop:].sum() / (e2.size - ntop))),
                      "share_of_squared_error_in_top_1pct_stations": float(top_st[:max(1, int(round(0.01 * top_st.size)))].sum() / tot),
                      "share_of_squared_error_in_top_10_stations": float(top_st[:10].sum() / tot),
                      "median_abs_error": float(np.median(np.abs((x[:, m] - o)[ok])))}
    res["squared_error_concentration"][nm] = conc
(O / "results/diag_obs_target.json").write_text(json.dumps(res, indent=1))
for nm in NETS.values():
    print("==", nm, res["observations"][nm])
    for name in ("ghap_pixel", "cams_first_day_bilinear", "downscaler_sameday"):
        L = res["lagged_agreement"][nm][name]
        print("   ", name, "| r by lag -2..2:", [round(L[str(l)]["median_station_r"], 3) for l in (-2, -1, 0, 1, 2)],
              "| within 0.5 by lag:", [round(L[str(l)]["share_within_0p5"], 3) for l in (-2, -1, 0, 1, 2)],
              "| rmse by lag:", [round(L[str(l)]["rmse"], 2) for l in (-2, -1, 0, 1, 2)])
        c = res["squared_error_concentration"][nm][name]
        print("        ", {k: round(v, 3) for k, v in c.items()})
