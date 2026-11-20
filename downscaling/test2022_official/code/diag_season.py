#!/usr/bin/env python3
"""A7, descriptive addition of 2026-11-19 (protocol note, "Deviations"; made after the tables were read): the same-day
table S split by season and by month, to see WHEN the downscaler loses to its coarse input in Korea and Japan. Same
common mask and same 7-day moving-block bootstrap as table S (score_official.py); the row "all 365 days" must reproduce
table S. Station pixel only. Nothing is selected or changed on it.
    python -I diag_season.py -> results/diag_season.json"""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
sys.path.insert(0, str(O / "code"))
from score_official import NETS, GROUPS, G, block_idx, BLOCK  # noqa: E402
st = np.load(G / "stations_2022.npz")
meta = np.load(O / "pos/st_meta_official.npz")
obs = st["daily"].astype(np.float32)[meta["id"]].T
src, pix = meta["src"], meta["pix"]
S = obs.shape[1]
ld = lambda k: np.asarray(np.load(O / f"preds/{k}.npy", mmap_mode="r")[..., 0]).astype(np.float32)[:, :, pix]      # noqa: E731
X = {"ds_sameday": ld("sameday_ds").mean(1), "cams0_bil": ld("sameday_cams0")[:, 0], "bcsd": ld("sameday_bcsd")[:, 0],
     "clim_med": ld("sameday_climmed")[:, 0], "clim_ens": ld("sameday_climens").mean(1),
     "ghap": np.load(O / "truth/ghap_official_2022.npy")[:, pix, 0],
     "clim_paper": np.load(G / "clim" / "st_clim_2022.npy").astype(np.float32)[:, meta["id"]]}
ok = np.isfinite(obs)
for v in X.values():
    ok &= np.isfinite(v)
gmask = np.stack([src == k for k in NETS] + [np.ones(S, bool)])
days = np.arange(365)
month = np.array([(np.datetime64("2022-01-01") + int(t)).astype("datetime64[M]").astype(int) % 12 + 1 for t in days])
periods = {"all 365 days": np.ones(365, bool), "May-Nov (days 120-333)": (days >= 120) & (days <= 333),
           "Jan-Apr + Dec": (days < 120) | (days > 333), "Dec-Feb": np.isin(month, (12, 1, 2)), "Mar-Apr": np.isin(month, (3, 4))}
periods.update({f"month {m:02d}": month == m for m in range(1, 13)})
se = {n: np.stack([np.where(ok, (x - obs) ** 2, 0.0).astype(np.float64)[:, m].sum(1) for m in gmask]) for n, x in X.items()}   # (G, 365)
e = {n: np.stack([np.where(ok, x - obs, 0.0).astype(np.float64)[:, m].sum(1) for m in gmask]) for n, x in X.items()}
nn = np.stack([ok[:, m].sum(1) for m in gmask]).astype(np.float64)
res = {"note": "same-day mode, station pixel, table S mask; relative RMSE change A / B - 1 with the 7-day moving-block bootstrap "
               "over the days of the period (10,000 draws, seed 0); the cold-season periods join non-adjacent days", "periods": {}}
for pn, pm in periods.items():
    t = np.where(pm)[0]
    idx = block_idx(len(t)) if len(t) >= BLOCK else None
    out = {}
    for g, gn in enumerate(GROUPS):
        n = nn[g, t].sum()
        r = {"n": int(n), "rmse": {a: float(np.sqrt(se[a][g, t].sum() / n)) for a in X}, "bias": {a: float(e[a][g, t].sum() / n) for a in X}}
        for A, B in (("ds_sameday", "cams0_bil"), ("ds_sameday", "bcsd"), ("ds_sameday", "clim_paper"), ("ds_sameday", "clim_med"), ("bcsd", "cams0_bil")):
            a, b = se[A][g, t], se[B][g, t]
            c = {"rel_rmse": float(np.sqrt(a.sum() / b.sum()) - 1)}
            if idx is not None and not pn.startswith("month"):
                with np.errstate(all="ignore"):
                    bs = np.sqrt(a[idx].sum(1) / b[idx].sum(1)) - 1
                c["ci"] = [float(v) for v in np.nanpercentile(bs, [2.5, 97.5])]
            r[f"{A} - {B}"] = c
        out[gn] = r
    res["periods"][pn] = out
(O / "results/diag_season.json").write_text(json.dumps(res, indent=1))
f = lambda c: f"{100 * c['rel_rmse']:+.1f} %" + (f" [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}]" if "ci" in c else "")   # noqa: E731
for pn in list(periods)[:5]:
    print("==", pn)
    for gn in GROUPS:
        r = res["periods"][pn][gn]
        print(f"   {gn:20s} n {r['n']:8d} | rmse ds {r['rmse']['ds_sameday']:.2f} cams0 {r['rmse']['cams0_bil']:.2f} bcsd {r['rmse']['bcsd']:.2f} clim_paper {r['rmse']['clim_paper']:.2f} "
              f"| bias ds {r['bias']['ds_sameday']:+.2f} cams0 {r['bias']['cams0_bil']:+.2f} | ds-cams0 {f(r['ds_sameday - cams0_bil'])} | ds-bcsd {f(r['ds_sameday - bcsd'])} | ds-clim_paper {f(r['ds_sameday - clim_paper'])}")
print("== monthly ds_sameday - cams0_bil (relative RMSE change), then bias of ds_sameday")
for gn in GROUPS:
    print(f"   {gn:20s}", " ".join(f"{100 * res['periods'][f'month {m:02d}'][gn]['ds_sameday - cams0_bil']['rel_rmse']:+5.0f}" for m in range(1, 13)),
          "| bias", " ".join(f"{res['periods'][f'month {m:02d}'][gn]['bias']['ds_sameday']:+5.1f}" for m in range(1, 13)))
