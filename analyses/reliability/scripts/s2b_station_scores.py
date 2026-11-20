#!/usr/bin/env python3
"""Step 2b: every station score with its intervals, from data/sums_<tag>.npz (s2_station.py).
Intervals (95 %): 'day' = moving-block bootstrap over the start days (blocks of 7 consecutive start days, 10,000
resamples: gc_style.block_bootstrap, the convention of the paper); 'cell' = bootstrap over 2-degree cells of stations
(2,000 resamples). 'jk' = jackknife standard error over the four members (the four fixed noise fields).
    python s2b_station_scores.py [final|before]   -> results/station_scores_<tag>.json, results/station_scores_<tag>.txt"""
import json
import sys

import numpy as np

from rel_common import DATA, HERE, block_weights

TAG = sys.argv[1] if len(sys.argv) > 1 else "final"
RES = HERE / "results"; RES.mkdir(exist_ok=True)
Z = np.load(DATA / f"sums_{TAG}.npz")
K = Z["keys"]; GROUPS = [str(g) for g in Z["groups"]]; THR = Z["thr"]; M = int(Z["members"]); SD_EDGES = Z["sd_edges"]
LAT, LON, SRC = Z["lat"], Z["lon"], Z["src"]
NAMES = [k[5:] for k in Z.files if k.startswith("day__")]
DAY = {k: Z["day__" + k] for k in NAMES}                         # (forecasts, 7 groups, ...)
STN = {k: Z["st__" + k].astype(np.float64) for k in NAMES}       # (7 leads, stations, ...)
SHAPE = {k: DAY[k].shape[2:] for k in NAMES}
F = (M + 1) / M
NETCODES = [0, 1, 3, 4, 5, 6]


def pack(get):
    """concatenate the quantities along one last axis; get(name) -> (rows, ...)"""
    cols, sl, c = [], {}, 0
    for k in NAMES:
        a = get(k); a = a.reshape(a.shape[0], -1)
        cols.append(a); sl[k] = slice(c, c + a.shape[1]); c += a.shape[1]
    return np.concatenate(cols, 1), sl


def unpack(X, sl):
    return {k: X[..., sl[k]].reshape(X.shape[:-1] + SHAPE[k]) for k in NAMES}


def div(a, b):
    with np.errstate(all="ignore"):
        return np.where(b != 0, a / np.where(b != 0, b, 1), np.nan)


def derive(S):
    """scores from summed terms (any leading batch dimensions)"""
    o = {}
    n = S["n"]; n1 = n[..., None]
    o["n"] = n
    o["spread"] = np.sqrt(F * div(S["var"], n))
    o["ghap_rmse"] = np.sqrt(div(S["ghap_se"], n)); o["ghap_bias"] = div(S["ghap_err"], n); o["ghap_mae"] = div(S["ghap_ae"], n)
    o["rel_n"] = S["relc"]
    o["ho_ref"] = div(S["hox"], S["hoc"])
    pk = np.arange(M + 1) / M
    for v in (0, 1):
        s = f"_{v}"
        o["rmse" + s] = np.sqrt(div(S["sse" + s], n)); o["bias" + s] = div(S["err" + s], n)
        o["mae" + s] = div(S["sae" + s], n); o["medae" + s] = div(S["medae" + s], n); o["crps" + s] = div(S["crps" + s], n)
        o["ss" + s] = np.sqrt(F * div(S["var"], S["sse" + s]))
        o["ss_unb" + s] = np.sqrt(F * div(S["var"], S["sse" + s] - div(S["err" + s] ** 2, n)))
        o["rank" + s] = div(S["rank" + s], S["rank" + s].sum(-1, keepdims=True))
        o["cover" + s] = div(S["cover" + s], n); o["below" + s] = div(S["below" + s], n); o["above" + s] = div(S["above" + s], n)
        o["c_crps" + s] = div(S["c_crps" + s], n); o["c_mae" + s] = div(S["c_sae" + s], n)
        o["c_rmse" + s] = np.sqrt(div(S["c_sse" + s], n))
        o["crps_vs_clim" + s] = 100 * (div(S["crps" + s], S["c_crps" + s]) - 1)
        o["rmse_vs_clim" + s] = 100 * (np.sqrt(div(S["sse" + s], S["c_sse" + s])) - 1)
        base = div(S["base" + s], n1); o["base" + s] = base
        fr = div(S["relo" + s], S["relc"]); o["rel_freq" + s] = fr
        bs = div(S["bs" + s], n1); cbs = div(S["c_bs" + s], n1); unc = base * (1 - base)
        o["bs" + s] = bs
        o["bss_base" + s] = 1 - div(bs, unc)
        o["bss_clim" + s] = 1 - div(bs, cbs)
        o["bss_clim_fair" + s] = 1 - div(S["bsf" + s], S["c_bsf" + s])
        o["bss_base_fair" + s] = 1 - div(div(S["bsf" + s], n1), unc)
        o["clim_bss_base" + s] = 1 - div(cbs, unc)
        o["rel_term" + s] = div((S["relc"] * np.nan_to_num(pk - fr) ** 2).sum(-1), n1)
        o["res_term" + s] = div((S["relc"] * np.nan_to_num(fr - base[..., None]) ** 2).sum(-1), n1)
        o["unc_term" + s] = unc
        o["ho_freq" + s] = div(S["hoo" + s], S["hoc"])
        o["sd_rmse" + s] = np.sqrt(div(S["sd_se" + s], S["sd_n"]))
        o["mem_rmse" + s] = np.sqrt(div(S["mem_sse" + s], n1))
        o["loo_crps" + s] = div(S["loo_crps" + s], n1); o["loo_rmse" + s] = np.sqrt(div(S["loo_sse" + s], n1))
        o["loo_ss" + s] = np.sqrt(M / (M - 1) * div(S["loo_var"], S["loo_sse" + s]))
    o["sd_spread"] = np.sqrt(F * div(S["sd_var"], S["sd_n"])); o["sd_n"] = S["sd_n"]
    # the station-days of the manuscript (figdata_station_crps.py)
    pn = S["p_n"]; pn1 = pn[..., None]
    o["p_n"] = pn
    for a, b in (("p_crps", "p_crps"), ("p_mae", "p_sae"), ("p_medae", "p_medae"), ("p_bias", "p_err"), ("p_clim_mae", "p_clim_ae"),
                 ("p_clim_crps", "p_clim_crps"), ("p_clim_medae", "p_clim_medae"), ("p_cams_mae", "p_cams_ae"),
                 ("p_aur_mae", "p_aur_ae"), ("p_cams_bias", "p_cams_err"), ("p_aur_bias", "p_aur_err")):
        o[a] = div(S[b], pn)
    for a, b in (("p_rmse", "p_sse"), ("p_clim_rmse", "p_clim_se"), ("p_cams_rmse", "p_cams_se"), ("p_aur_rmse", "p_aur_se")):
        o[a] = np.sqrt(div(S[b], pn))
    o["p_ss"] = np.sqrt(F * div(S["p_var"], S["p_sse"]))
    pct = lambda a, b: 100 * (div(S[a], S[b]) - 1)
    o["p_crps_vs_climens"] = pct("p_crps", "p_clim_crps"); o["p_crps_vs_climmae"] = pct("p_crps", "p_clim_ae")
    o["p_mae_vs_cams"] = pct("p_sae", "p_cams_ae"); o["p_mae_vs_aur"] = pct("p_sae", "p_aur_ae")
    o["p_medae_vs_cams"] = pct("p_medae", "p_cams_ae"); o["p_medae_vs_aur"] = pct("p_medae", "p_aur_ae")
    o["p_crps_vs_camsmae"] = pct("p_crps", "p_cams_ae"); o["p_crps_vs_aurmae"] = pct("p_crps", "p_aur_ae")
    o["p_mae_vs_climmae"] = pct("p_sae", "p_clim_ae")
    o["p_climens_vs_camsmae"] = pct("p_clim_crps", "p_cams_ae"); o["p_climens_vs_aurmae"] = pct("p_clim_crps", "p_aur_ae")
    o["p_rmse_vs_cams"] = 100 * (np.sqrt(div(S["p_sse"], S["p_cams_se"])) - 1)
    o["p_rmse_vs_aur"] = 100 * (np.sqrt(div(S["p_sse"], S["p_aur_se"])) - 1)
    o["p_loo_crps"] = div(S["p_loo_crps"], pn1); o["p_loo_mae"] = div(S["p_loo_sae"], pn1)
    o["p_loo_rmse"] = np.sqrt(div(S["p_loo_sse"], pn1)); o["p_loo_ss"] = np.sqrt(M / (M - 1) * div(S["p_loo_var"], S["p_loo_sse"]))
    return o


def jk(o, key):
    """jackknife standard error over the members from the leave-one-out values (last axis)"""
    x = o[key]
    return np.sqrt((M - 1) / M * ((x - x.mean(-1, keepdims=True)) ** 2).sum(-1))


def tolist(a):
    a = np.asarray(a, float)
    return None if a.ndim == 0 and not np.isfinite(a) else (float(a) if a.ndim == 0 else [None if not np.isfinite(x) else float(x) for x in a.ravel()] if a.ndim == 1 else [tolist(x) for x in a])


cell_of = np.floor((LAT + 90) / 2).astype(int) * 1000 + np.floor((LON + 180) / 2).astype(int)
OUT = {"tag": TAG, "members": M, "groups": GROUPS, "thresholds": [float(t) for t in THR], "sd_edges": [float(x) for x in SD_EDGES],
       "leads": {}}
rng = np.random.default_rng(0)
for h in range(1, 8):
    rows = np.where(K[:, 1] == h)[0]
    rows = rows[np.argsort(K[rows, 0])]
    OUT["leads"][h] = {}
    for g, gname in enumerate(GROUPS):
        X, sl = pack(lambda k: DAY[k][rows, g])
        pt = derive(unpack(X.sum(0), sl))
        # ---- start-day blocks; days without any station-day of the sample are dropped first (as gc_style.block_bootstrap)
        boot = {}
        for sample, cnt in (("cal", "n"), ("paper", "p_n")):
            ok = DAY[cnt][rows, g] > 0
            if ok.sum() < 14:
                continue
            W = block_weights(int(ok.sum()), B=7, n=10000, seed=0)
            bo = derive(unpack(W @ X[ok], sl))
            for k_, v_ in bo.items():
                if k_.startswith("p_") == (sample == "paper"):
                    boot[k_] = v_
        # ---- 2-degree cells of stations
        sel = np.where(np.isin(SRC, NETCODES) if g == 6 else SRC == NETCODES[g])[0]
        Xs, sl2 = pack(lambda k: STN[k][h - 1][sel])
        cu, ci_ = np.unique(cell_of[sel], return_inverse=True)
        C = np.zeros((len(cu), Xs.shape[1])); np.add.at(C, ci_, Xs)
        Wc = rng.multinomial(len(cu), np.full(len(cu), 1 / len(cu)), size=2000).astype(np.float64)
        bc = derive(unpack(Wc @ C, sl2))
        ptc = derive(unpack(C.sum(0), sl2))
        rec = {"n_cells": int(len(cu)), "n_stations": int((STN["n"][h - 1][sel] > 0).sum())}
        for k_, v_ in pt.items():
            e = {"v": tolist(v_)}
            if k_ in boot:
                with np.errstate(all="ignore"):
                    e["ci_day"] = [tolist(np.nanpercentile(boot[k_], 2.5, 0)), tolist(np.nanpercentile(boot[k_], 97.5, 0))]
            with np.errstate(all="ignore"):
                e["ci_cell"] = [tolist(np.nanpercentile(bc[k_], 2.5, 0)), tolist(np.nanpercentile(bc[k_], 97.5, 0))]
            rec[k_] = e
        # sums over stations and over forecasts must agree
        assert abs(float(ptc["n"]) - float(pt["n"])) < 0.5, (h, g, float(ptc["n"]), float(pt["n"]))
        for key, src in (("crps_0", "loo_crps_0"), ("crps_1", "loo_crps_1"), ("rmse_0", "loo_rmse_0"), ("rmse_1", "loo_rmse_1"),
                         ("ss_0", "loo_ss_0"), ("ss_1", "loo_ss_1"), ("p_crps", "p_loo_crps"), ("p_mae", "p_loo_mae"),
                         ("p_rmse", "p_loo_rmse"), ("p_ss", "p_loo_ss")):
            rec[key]["jk_se"] = tolist(jk(pt, src))
        OUT["leads"][h][gname] = rec
    print("lead", h, "done", flush=True)

with open(RES / f"station_scores_{TAG}.json", "w") as f:
    json.dump(OUT, f)

# ---------------------------------------------------------------- readable tables
L = OUT["leads"]; A = "All official networks"
lines = []
P = lines.append


def q(x, d=3, sign=False):
    return "n/a" if x is None else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def fmt(e, d=2, ci="ci_day", sign=False):
    if e["v"] is None:
        return "n/a"
    s = q(e["v"], d, sign)
    if ci in e and e[ci][0] is not None:
        s += f" [{q(e[ci][0], d, sign)}, {q(e[ci][1], d, sign)}]"
    return s


def el(e, j, d=3, ci="ci_day", sign=True):
    """element j of a vector score with its interval"""
    s = q(e["v"][j], d, sign)
    if ci in e:
        s += f" [{q(e[ci][0][j], d, sign)}, {q(e[ci][1][j], d, sign)}]"
    return s


def vec(v, d=3):
    return " / ".join(q(x, d) for x in v)


P(f"STATION SCORES, model '{TAG}', {M} members, 2022. Source: data/sums_{TAG}.npz (scripts/s2_station.py), scored by "
  f"scripts/s2b_station_scores.py. Intervals: 95 %, block bootstrap over start days (7-start-day blocks, 10,000) unless 'cell' "
  f"(2-degree cells of stations, 2,000). v=0: against the measurements; v=1: against the 1 km estimate at the same pixel, same station-days.")
P("")
P("== A. Spread-to-error ratio sqrt((M+1)/M mean variance) / RMSE of the ensemble mean, 'cal' sample (identical station-days for both references)")
for gname in GROUPS[::-1]:
    P(f"-- {gname}")
    for h in range(1, 8):
        r = L[h][gname]
        P(f"  lead {h}: n {int(r['n']['v']):,} ({r['n_stations']} stations, {r['n_cells']} cells) | vs measurements {fmt(r['ss_0'])} cell {fmt(r['ss_0'], ci='ci_cell')} jk-se {q(r['ss_0']['jk_se'])}"
          f" | vs estimates {fmt(r['ss_1'])} cell {fmt(r['ss_1'], ci='ci_cell')} jk-se {q(r['ss_1']['jk_se'])} | bias removed {q(r['ss_unb_0']['v'], 2)} / {q(r['ss_unb_1']['v'], 2)}"
          f" | spread {q(r['spread']['v'], 2)} RMSE {q(r['rmse_0']['v'], 2)} / {q(r['rmse_1']['v'], 2)} bias {q(r['bias_0']['v'], 2, True)} / {q(r['bias_1']['v'], 2, True)}"
          f" | estimates vs measurements RMSE {q(r['ghap_rmse']['v'], 2)} bias {q(r['ghap_bias']['v'], 2, True)}")
P("")
P("== B. Rank histogram of the verification among the four members (five bins, 0.20 each if exchangeable) and member range (0.60 expected)")
for gname in GROUPS[::-1]:
    P(f"-- {gname}")
    for h in (1, 3, 7):
        r = L[h][gname]
        for v, nm in ((0, "measurements"), (1, "estimates   ")):
            rk = r[f"rank_{v}"]
            P(f"  lead {h} vs {nm}: " + vec(rk["v"]) + " | day CI low " + vec(rk["ci_day"][0]) + " high " + vec(rk["ci_day"][1])
              + " | cell CI low " + vec(rk["ci_cell"][0]) + " high " + vec(rk["ci_cell"][1])
              + f" | inside {fmt(r[f'cover_{v}'], 3)} cell {fmt(r[f'cover_{v}'], 3, 'ci_cell')} below {q(r[f'below_{v}']['v'])} above {q(r[f'above_{v}']['v'])}")
P("")
P("== C. Reliability of the exceedance probabilities (members above the threshold: 0..4 of 4), 'cal' sample")
for gname in (A,) + tuple(GROUPS[:-1]):
    P(f"-- {gname}")
    for j, T in enumerate(THR):
        for h in (1, 3, 7):
            r = L[h][gname]
            nk = r["rel_n"]["v"][j]
            for v, nm in ((0, "measurements"), (1, "estimates   ")):
                fr = r[f"rel_freq_{v}"]
                P(f"  > {T:g} lead {h} vs {nm}: base {q(r[f'base_{v}']['v'][j])} | n per bin " + " / ".join(f"{int(x):,}" for x in nk)
                  + " | observed " + vec(fr["v"][j]) + " | CI low " + vec(fr["ci_day"][0][j]) + " high " + vec(fr["ci_day"][1][j])
                  + " | cell CI low " + vec(fr["ci_cell"][0][j]) + " high " + vec(fr["ci_cell"][1][j])
                  + f" | Brier {q(r[f'bs_{v}']['v'][j], 4)} | BSS vs pooled frequency {el(r[f'bss_base_{v}'], j)}"
                  + f" | BSS vs local climatological frequency {el(r[f'bss_clim_{v}'], j)} cell {el(r[f'bss_clim_{v}'], j, ci='ci_cell')} fair {q(r[f'bss_clim_fair_{v}']['v'][j], 3, True)}"
                  + f" | local climatological frequency vs pooled frequency {q(r[f'clim_bss_base_{v}']['v'][j], 3, True)}"
                  + f" | reliability / resolution / uncertainty terms {q(r[f'rel_term_{v}']['v'][j], 4)} / {q(r[f'res_term_{v}']['v'][j], 4)} / {q(r[f'unc_term_{v}']['v'][j], 4)}")
            P(f"  > {T:g} lead {h} three members -> fourth member (a perfectly reliable ensemble): " + vec(r["ho_ref"]["v"][j])
              + " | -> measurement " + vec(r["ho_freq_0"]["v"][j]) + " | -> estimate " + vec(r["ho_freq_1"]["v"][j]))
P("")
P("== D. CRPS (fair, four members) against the climatological ensemble of the 1 km estimates at the station pixel (2017-2020, +-7 calendar days; fair CRPS), 'cal' sample")
for gname in GROUPS[::-1]:
    P(f"-- {gname}")
    for h in range(1, 8):
        r = L[h][gname]
        P(f"  lead {h}: vs measurements CRPS {fmt(r['crps_0'])} climatological ensemble {fmt(r['c_crps_0'])} change {fmt(r['crps_vs_clim_0'], 1, sign=True)} % cell {fmt(r['crps_vs_clim_0'], 1, 'ci_cell', True)}"
          f" | vs estimates CRPS {fmt(r['crps_1'])} climatological ensemble {fmt(r['c_crps_1'])} change {fmt(r['crps_vs_clim_1'], 1, sign=True)} %"
          f" | MAE of the mean {q(r['mae_0']['v'], 2)} / {q(r['mae_1']['v'], 2)}, of the climatological mean {q(r['c_mae_0']['v'], 2)} / {q(r['c_mae_1']['v'], 2)}")
P("")
P("== E. The station-days of the manuscript (figdata_station_crps.py): CRPS of the four members, absolute error of single forecasts, climatological ensemble")
for gname in GROUPS[::-1]:
    P(f"-- {gname}")
    for h in range(1, 8):
        r = L[h][gname]
        if r["p_n"]["v"] in (None, 0.0):
            continue
        s = (f"  lead {h}: n {int(r['p_n']['v']):,} | Foehn CRPS {fmt(r['p_crps'])} jk-se {q(r['p_crps']['jk_se'])} | MAE mean {fmt(r['p_mae'])} median {q(r['p_medae']['v'], 2)} RMSE {fmt(r['p_rmse'])}"
             f" | climatological ensemble CRPS {fmt(r['p_clim_crps'])} | climatological mean MAE {fmt(r['p_clim_mae'])} RMSE {q(r['p_clim_rmse']['v'], 2)}"
             f" | CRPS vs climatological ensemble {fmt(r['p_crps_vs_climens'], 1, sign=True)} % cell {fmt(r['p_crps_vs_climens'], 1, 'ci_cell', True)} | spread/error {fmt(r['p_ss'])}")
        if h <= 4:
            s += (f" | CAMS MAE {fmt(r['p_cams_mae'])} RMSE {q(r['p_cams_rmse']['v'], 2)} Aurora MAE {fmt(r['p_aur_mae'])} RMSE {q(r['p_aur_rmse']['v'], 2)}"
                  f" | MAE of the mean vs CAMS {fmt(r['p_mae_vs_cams'], 1, sign=True)} % cell {fmt(r['p_mae_vs_cams'], 1, 'ci_cell', True)} vs Aurora {fmt(r['p_mae_vs_aur'], 1, sign=True)} % cell {fmt(r['p_mae_vs_aur'], 1, 'ci_cell', True)}"
                  f" | CRPS vs CAMS MAE {fmt(r['p_crps_vs_camsmae'], 1, sign=True)} % vs Aurora MAE {fmt(r['p_crps_vs_aurmae'], 1, sign=True)} %"
                  f" | climatological-ensemble CRPS vs CAMS MAE {fmt(r['p_climens_vs_camsmae'], 1, sign=True)} % vs Aurora MAE {fmt(r['p_climens_vs_aurmae'], 1, sign=True)} %"
                  f" | RMSE vs CAMS {fmt(r['p_rmse_vs_cams'], 1, sign=True)} % vs Aurora {fmt(r['p_rmse_vs_aur'], 1, sign=True)} %")
        P(s)
P("")
P("== F. Spread against error in bins of ensemble spread (all official networks, lead 1 / 3 / 7): bin (ug m-3), share of station-days, spread, RMSE vs measurements, RMSE vs estimates")
for h in (1, 3, 7):
    r = L[h][A]; tot = sum(r["sd_n"]["v"])
    P(f"-- lead {h}")
    for b in range(len(SD_EDGES) - 1):
        if r["sd_n"]["v"][b] > 0:
            P(f"  {SD_EDGES[b]:g}-{SD_EDGES[b + 1]:g}: {100 * r['sd_n']['v'][b] / tot:.1f} % | spread {q(r['sd_spread']['v'][b], 2)} | RMSE {q(r['sd_rmse_0']['v'][b], 2)} / {q(r['sd_rmse_1']['v'][b], 2)}")
P("")
P("== G. The four members one by one and the jackknife over them (all official networks): RMSE of each member vs measurements; leave-one-out CRPS; jackknife se")
for h in (1, 3, 7):
    r = L[h][A]
    P(f"  lead {h}: member RMSE " + vec(r["mem_rmse_0"]["v"]) + " | leave-one-out CRPS " + vec(r["loo_crps_0"]["v"], 4)
      + f" | CRPS {q(r['crps_0']['v'], 4)} jk-se {q(r['crps_0']['jk_se'], 4)} day CI [{q(r['crps_0']['ci_day'][0])}, {q(r['crps_0']['ci_day'][1])}] | RMSE {q(r['rmse_0']['v'])} jk-se {q(r['rmse_0']['jk_se'], 4)}"
      + f" | spread/error {q(r['ss_0']['v'])} jk-se {q(r['ss_0']['jk_se'], 4)}")
(RES / f"station_scores_{TAG}.txt").write_text("\n".join(lines) + "\n")
print(f"wrote results/station_scores_{TAG}.json and .txt ({len(lines)} lines)")
