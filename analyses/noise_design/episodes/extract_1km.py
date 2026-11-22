#!/usr/bin/env python3
"""episodes_diag: extraction of S1 (2021, validation) or S2 (2022, test) of Foehn 1 km crps_e01 on the populated 0.25 deg
cells -> data/ext_<year>.npz and results/checks_<year>.json (definitions: ../README.md).
Reads only map25, map25_m0, map25_ghap, keys and info of the part files (never st_ens / st_ghap), pm.npy, pop025.npy,
the HRES precipitation channel of the forecast inputs and the HRES 3-hourly 10 m winds.
    python extract_1km.py 2021 [--runs N --stride S --out DIR]   (N, S: smoke test on a subsample)"""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import G, L_HIST, ROOT, RUN, Npy, region_masks, save_json, save_npz, season_of_month  # noqa: E402

H = 7


def main(a):
    t0 = time.time()
    year = a.year
    base = {2021: 1461, 2022: 1826}[year]
    ev = RUN / {2021: "eval2021_sel_crps_e01", 2022: "eval_final_crps_e01"}[year]
    out = Path(a.out) if a.out else ROOT
    chk = {"sample": f"S{1 if year == 2021 else 2}", "year": year, "eval_folder": str(ev)}

    # ---------------- keys: (start day, lead, target day) of every stored forecast
    parts = sorted(ev.glob("part_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    loc, info = {}, None
    for p in parts:
        with np.load(p) as z:
            k = z["keys"]
            if info is None:
                info = z["info"].tolist()
        for row, (d, h, tday) in enumerate(k.tolist()):
            assert (d, h) not in loc, ("duplicate forecast", d, h)
            assert tday == d + h, ("target day", d, h, tday)
            loc[(d, h)] = (p, row)
    starts = sorted({d for d, _ in loc})
    complete = [d for d in starts if all((d, h) in loc for h in range(1, H + 1))]
    missing = [(d, h) for d in starts for h in range(1, H + 1) if (d, h) not in loc]
    runs = complete[:a.runs] if a.runs else complete
    chk.update(n_parts=len(parts), n_forecasts=len(loc), info=info, start_days=starts, missing=missing,
               complete_runs=complete, runs_used=runs, target_day_is_start_plus_lead=True)
    print(f"{ev.name}: {len(parts)} parts, {len(loc)} forecasts, {len(starts)} start days, missing {missing}, "
          f"{len(complete)} complete runs, using {len(runs)}", flush=True)

    # ---------------- cells
    pop2 = np.load(G / "pop025.npy").astype(np.float64)
    cells = np.flatnonzero(pop2.ravel() > 0)[::a.stride]
    kr, kc = cells // 1440, cells % 1440
    lat, lon = 84.0 - 0.25 * kr, -180.0 + 0.25 * kc
    pop = pop2.ravel()[cells]
    C, N = len(cells), len(runs)
    ri = {d: i for i, d in enumerate(runs)}
    chk.update(n_cells=C, residents=float(pop.sum()), residents_all_cells=float(pop2[pop2 > 0].sum()))

    # ---------------- forecasts and the truth of the target days (float16 as stored)
    X = np.full((N, 1, H, C), np.nan, np.float16)
    Mn = np.full((N, H, C), np.nan, np.float16)
    Y = np.full((N, H, C), np.nan, np.float16)
    for p in parts:
        rows = [(row, d, h) for (d, h), (pp, row) in loc.items() if pp == p and d in ri]
        if not rows:
            continue
        with np.load(p) as z:
            m25, m0, g25 = z["map25"], z["map25_m0"], z["map25_ghap"]
        for row, d, h in rows:
            X[ri[d], 0, h - 1] = m0[row].reshape(-1)[cells]
            Mn[ri[d], h - 1] = m25[row].reshape(-1)[cells]
            Y[ri[d], h - 1] = g25[row].reshape(-1)[cells]
    print(f"forecasts read {time.time() - t0:.0f}s", flush=True)

    # ---------------- truth series from pm.npy: history (60 days) .. d + 8
    pm = Npy(G / "pm.npy")
    gi0 = base + min(runs) - L_HIST - 1
    gi1 = min(base + max(runs) + H + 1, pm.shape[0] - 1)
    truth = np.stack([pm.plane((gi,)).reshape(-1)[cells] for gi in range(gi0, gi1 + 1)]).astype(np.float32)
    run_tidx = np.array([base + d - gi0 for d in runs])
    valid = np.load(G / "valid.npy")
    chk["truth_series"] = {"global_index_first": int(gi0), "global_index_last": int(gi1),
                           "invalid_days_inside": [int(g) for g in range(gi0, gi1 + 1) if not valid[g]]}

    # ---------------- P95 of the daily truth 2017-2020 (pm.npy indices 0..1460), cells with >= 1000 finite days
    clim = np.empty((1461, C), np.float32)
    for gi in range(1461):
        clim[gi] = pm.plane((gi,)).reshape(-1)[cells]
    nfin = np.isfinite(clim).sum(0)
    s = np.sort(clim, axis=0)                                  # NaN last
    pos = 0.95 * (np.maximum(nfin, 1) - 1)
    lo = np.floor(pos).astype(int); fr = pos - lo
    hi = np.minimum(lo + 1, np.maximum(nfin - 1, 0))
    p95 = (np.take_along_axis(s, lo[None], 0)[0] * (1 - fr) + np.take_along_axis(s, hi[None], 0)[0] * fr).astype(np.float32)
    p95[nfin < 1000] = np.nan
    del clim, s
    chk["p95"] = {"cells_defined": int(np.isfinite(p95).sum()), "quantiles_5_50_95": np.nanpercentile(p95, [5, 50, 95]).tolist()}
    print(f"truth + P95 {time.time() - t0:.0f}s", flush=True)

    # ---------------- HRES inputs of the run of day d: precipitation of each target day (mm), daily mean 10 m wind speed
    if year == 2021:
        hb, hb_alt, hw = Npy(G / "hres_block_train.npy"), Npy(G / "hres_block_2021.npy"), Npy(G / "hres_wind_2021.npy")
        bi = lambda d: base + d                                 # the store read by the 2021 evaluation (global index)
    else:
        hb, hb_alt, hw = Npy(G / "hres_block.npy"), None, Npy(G / "hres_wind.npy")
        bi = lambda d: d
    P = np.empty((N, H, C), np.float16)
    W = np.empty((N, H + 1, C), np.float16)
    r05, c05 = kr // 2, kc // 2
    for i, d in enumerate(runs):
        for e in range(1, H + 1):
            P[i, e - 1] = (hb.plane((bi(d), e - 1, 30)).reshape(-1)[cells].astype(np.float32) * 1000.0).astype(np.float16)
        for t in range(H + 1):
            sp = np.zeros((288, 720), np.float32)
            for st in range(8 * t, 8 * t + 8):
                uv = hw.block((d, st, 0), 2).astype(np.float32)   # u10, v10 (m/s)
                sp += np.sqrt(uv[0] ** 2 + uv[1] ** 2)
            W[i, t] = (sp / 8.0)[r05, c05].astype(np.float16)
    hres = {"precip_mm_mean": float(np.nanmean(P.astype(np.float32))), "precip_mm_p99": float(np.nanpercentile(P.astype(np.float32), 99)),
            "wind_ms_mean": float(np.nanmean(W.astype(np.float32))), "wind_ms_p99": float(np.nanpercentile(W.astype(np.float32), 99)),
            "precip_nonfinite_share": float((~np.isfinite(P)).mean()), "wind_nonfinite_share": float((~np.isfinite(W)).mean())}
    if hb_alt is not None:                                       # the 2021 block of build_hres_year.py must be the same runs
        dif = []
        for d in runs[:3]:
            for e in (1, 7):
                a1 = hb.plane((bi(d), e - 1, 30)).astype(np.float32); a2 = hb_alt.plane((d, e - 1, 30)).astype(np.float32)
                b1 = hb.plane((bi(d), e - 1, 0)).astype(np.float32); b2 = hb_alt.plane((d, e - 1, 0)).astype(np.float32)
                dif.append(float(np.nanmax(np.abs(a1 - a2)) + np.nanmax(np.abs(b1 - b2))))
        hres["max_abs_diff_block_train_vs_block_2021"] = max(dif)
    chk["hres"] = hres
    print(f"HRES {time.time() - t0:.0f}s", flush=True)

    # ---------------- pairing checks
    Yf, Mf, Xf = Y.astype(np.float32), Mn.astype(np.float32), X[:, 0].astype(np.float32)
    pmT = np.stack([truth[run_tidx + h] for h in range(1, H + 1)], 1)    # (N, H, C) pm.npy on the target days
    y0 = truth[run_tidx]
    both = np.isfinite(pmT) & np.isfinite(Yf)
    rel = np.abs(pmT - Yf)[both] / np.maximum(Yf[both], 1.0)
    rng = np.random.default_rng(0)
    sub = rel[rng.integers(0, rel.size, min(rel.size, 5_000_000))] if rel.size else rel
    tabs = {}
    for name, T in (("37.5", 37.5), ("75", 75.0), ("P95", None)):
        Tc = (p95[None, None, :] * np.ones_like(Yf)) if T is None else T
        okT = both & (np.isfinite(Tc) if T is None else True)
        a_, b_ = (pmT > Tc) & okT, (Yf > Tc) & okT
        n11, n10, n01 = int((a_ & b_).sum()), int((a_ & ~b_).sum()), int((~a_ & b_).sum())
        tabs[name] = {"both_above": n11, "pm_only": n10, "ghap25_only": n01,
                      "disagree_share_of_above": (n10 + n01) / max(n11 + n10 + n01, 1)}
    complete_cr = np.isfinite(y0) & np.isfinite(Yf).all(1) & np.isfinite(Mf).all(1) & np.isfinite(Xf).all(1)
    o = (4 * Mf - Xf) / 3
    chk["pairing"] = {
        "pm_vs_map25_ghap_rel_diff_quantiles_50_90_99_999": np.percentile(sub, [50, 90, 99, 99.9]).tolist() if sub.size else None,
        "pm_vs_map25_ghap_n": int(both.sum()),
        "finite_in_one_only": {"pm_only": int((np.isfinite(pmT) & ~np.isfinite(Yf)).sum()), "ghap25_only": int((~np.isfinite(pmT) & np.isfinite(Yf)).sum())},
        "exceedance_tables": tabs,
        "nan_pattern_mismatch_member0_vs_ghap25": int((np.isfinite(Xf) != np.isfinite(Yf)).sum()),
        "nan_pattern_mismatch_mean_vs_ghap25": int((np.isfinite(Mf) != np.isfinite(Yf)).sum()),
        "complete_cell_runs": int(complete_cr.sum()), "complete_share": float(complete_cr.mean()),
        "complete_share_per_run_min_max": [float(complete_cr.mean(1).min()), float(complete_cr.mean(1).max())],
        "others_mean_negative_share": float((o[np.isfinite(o)] < 0).mean()),
        "others_mean_min": float(np.nanmin(o)),
        "member0_over_mean_ratio_quantiles_1_50_99": np.nanpercentile((Xf / np.maximum(Mf, 0.1))[np.isfinite(Xf) & np.isfinite(Mf)][::97], [1, 50, 99]).tolist(),
    }
    print("pairing", chk["pairing"], flush=True)

    season = np.array([season_of_month((dt.date(year, 1, 1) + dt.timedelta(days=int(d))).month) for d in runs])
    save_npz(out / "data" / f"ext_{year}.npz", sample=np.array(chk["sample"]), H=np.array(H), M=np.array(4), Mx=np.array(1),
             run_label=np.array(runs), run_season=season, cells=cells, lat=lat, lon=lon, pop=pop, p95=p95, p95_n=nfin,
             truth=truth, truth_gi0=np.array(gi0), run_tidx=run_tidx, Y=Y, X=X, Mn=Mn, P=P, W=W,
             regions=region_masks(lat, lon), info=np.array(info))
    chk["seconds"] = time.time() - t0
    save_json(out / "results" / f"checks_{year}.json", chk)
    print(f"saved ext_{year}.npz and checks_{year}.json, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int, choices=[2021, 2022])
    ap.add_argument("--runs", type=int, default=0, help="smoke test: first N complete runs")
    ap.add_argument("--stride", type=int, default=1, help="smoke test: every n-th populated cell")
    ap.add_argument("--out", default=None, help="smoke test: output root (data/, results/ inside; $EPI_SCRATCH)")
    main(ap.parse_args())
