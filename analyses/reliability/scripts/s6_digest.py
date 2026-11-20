#!/usr/bin/env python3
"""Step 6: the numbers of the replacement text in one place, the tables for the Supplementary Information, and one
sensitivity: how much of the squared error at the stations sits in a few measurements (values above 500 ug m-3).
Reads results/*.json (s2b, s34, s5) and data/st_final.npz; writes results/digest.txt, results/digest.json,
results/supp_table_reliability.csv, results/supp_table_spread_error.csv, results/supp_table_crps.csv.
    python s6_digest.py"""
import csv
import json

import numpy as np

from rel_common import DATA, HERE, NETS, stations

RES = HERE / "results"
S = json.load(open(RES / "station_scores_final.json")); SB = json.load(open(RES / "station_scores_before.json"))
EST = json.load(open(RES / "estimates_scores.json")); M16 = json.load(open(RES / "members16_noise.json"))
A, PP, NINE = "All official networks", "All populated pixels", "Nine regions pooled"
THR = S["thresholds"]
D = {}
out = []
P = out.append
st = lambda h, tag=S: tag["leads"][str(h)][A]
pop = lambda h: EST["pop"]["leads"][str(h)][PP]["pixels"]
popw = lambda h: EST["pop"]["leads"][str(h)][PP]["residents"]


def f(e, d=2, ci="ci_day", j=None, sign=False):
    v = e["v"] if j is None else e["v"][j]
    lo, hi = (e[ci][0], e[ci][1]) if j is None else (e[ci][0][j], e[ci][1][j])
    s = "+" if sign else ""
    return f"{v:{s}.{d}f} [{lo:{s}.{d}f}, {hi:{s}.{d}f}]"


P("DIGEST of the calibration analysis (every number with its interval; sources: results/station_scores_final.json = SF, "
  "station_scores_before.json = SB, estimates_scores.json = ES, members16_noise.json = MN; intervals 95 %, block bootstrap over start days)")
P("")
P("1. Against the 1 km estimates")
for h in (1, 3):
    r = pop(h); w = popw(h)
    P(f"  populated pixels, lead {h} [ES pop.leads.{h}.'{PP}'.pixels]: {int(EST['pop']['n_pixels'][0]):,} pixels, {r['n']['v']:.4g} pixel-days | spread/error {f(r['ss'], 2)} | rank "
      + " / ".join(f"{x:.3f}" for x in r["rank"]["v"]) + f" | inside the member range {f(r['cover'], 3)} | weighted by residents: spread/error {f(w['ss'], 2)}, inside {f(w['cover'], 3)}, rank "
      + " / ".join(f"{x:.3f}" for x in w["rank"]["v"]))
    for j, T in enumerate(THR):
        P(f"      > {T:g}: observed at forecast 0 / .25 / .5 / .75 / 1: " + " / ".join(f"{x:.3f}" for x in r["rel_freq"]["v"][j]) + f" | BSS vs pooled frequency {f(r['bss_base'], 2, j=j, sign=True)}"
          f" | vs the 2022 frequency of each pixel {r['bss_pixel_frequency']['v'][j]:+.2f} | three members -> fourth " + " / ".join(f"{x:.3f}" for x in r["ho_ref"]["v"][j])
          + " ; -> estimate " + " / ".join(f"{x:.3f}" for x in r["ho_freq"]["v"][j]))
L22, L21, L22b = (EST["land"][k]["leads"] for k in ("2022 final", "2021 final (validation year)", "2022 before CRPS training"))
P("  land, nine regions pooled [ES land.'2022 final']: " + "  ".join(f"{L22[str(h)][NINE]['ss']:.2f} [{L22[str(h)][NINE]['ss_ci'][0]:.2f}, {L22[str(h)][NINE]['ss_ci'][1]:.2f}]" for h in range(1, 8)))
regs = [r for r in EST["land"]["2022 final"]["regions"] if r != NINE]
allv = [(L22[str(h)][r]["ss"], r, h) for h in range(1, 8) for r in regs]
P(f"      by region and lead: min {min(allv)[0]:.2f} ({min(allv)[1]}, lead {min(allv)[2]}), max {max(allv)[0]:.2f} ({max(allv)[1]}, lead {max(allv)[2]}); "
  + "; ".join(f"{r} {min(L22[str(h)][r]['ss'] for h in range(1, 8)):.2f}-{max(L22[str(h)][r]['ss'] for h in range(1, 8)):.2f}" for r in regs))
P("      2021 (validation year): " + "  ".join(f"{L21[str(h)][NINE]['ss']:.2f}" for h in range(1, 8)) + " | 2022 before the CRPS training: " + "  ".join(f"{L22b[str(h)][NINE]['ss']:.2f}" for h in range(1, 8)))
P("")
P("2. At the 6,324 official stations ('cal' sample: measurement, estimate and climatological ensemble all available)")
for h in range(1, 8):
    r = st(h); b = st(h, SB)
    P(f"  lead {h}: n {int(r['n']['v']):,} ({r['n_stations']} stations) | spread/error vs measurements {f(r['ss_0'])} (cells {r['ss_0']['ci_cell'][0]:.2f}-{r['ss_0']['ci_cell'][1]:.2f}), vs estimates at the same pixel {f(r['ss_1'])}"
      f" | before the CRPS training {b['ss_0']['v']:.2f} / {b['ss_1']['v']:.2f} | inside the member range: measurements {f(r['cover_0'], 3)}, estimates {f(r['cover_1'], 3)} (before: {b['cover_0']['v']:.3f} / {b['cover_1']['v']:.3f})"
      f" | below all members {r['below_0']['v']:.3f}, above {r['above_0']['v']:.3f} | RMSE vs measurements {r['rmse_0']['v']:.2f}, vs estimates {r['rmse_1']['v']:.2f}, estimates vs measurements {r['ghap_rmse']['v']:.2f}, spread {r['spread']['v']:.2f}")
for h in (1, 3, 7):
    r = st(h)
    P(f"  lead {h} rank of the measurement: " + " / ".join(f"{x:.3f}" for x in r["rank_0"]["v"]) + " | of the estimate: " + " / ".join(f"{x:.3f}" for x in r["rank_1"]["v"]))
    for j, T in enumerate(THR):
        P(f"      > {T:g}: n per class " + " / ".join(f"{int(x):,}" for x in r["rel_n"]["v"][j]) + " | observed (measurements) " + " / ".join(f"{x:.3f}" for x in r["rel_freq_0"]["v"][j])
          + " | (estimates) " + " / ".join(f"{x:.3f}" for x in r["rel_freq_1"]["v"][j]) + f" | BSS vs pooled frequency {f(r['bss_base_0'], 2, j=j, sign=True)} | vs local climatological frequency {f(r['bss_clim_0'], 2, j=j, sign=True)}"
          f" (cells {r['bss_clim_0']['ci_cell'][0][j]:+.2f}, {r['bss_clim_0']['ci_cell'][1][j]:+.2f}); fair {f(r['bss_clim_fair_0'], 2, j=j, sign=True)} | reliability term / uncertainty {r['rel_term_0']['v'][j] / r['unc_term_0']['v'][j]:.3f}, resolution / uncertainty {r['res_term_0']['v'][j] / r['unc_term_0']['v'][j]:.3f}"
          f" | three members -> fourth " + " / ".join(f"{x:.3f}" for x in r["ho_ref"]["v"][j]) + " ; -> measurement " + " / ".join(f"{x:.3f}" for x in r["ho_freq_0"]["v"][j]) + " ; -> estimate " + " / ".join(f"{x:.3f}" for x in r["ho_freq_1"]["v"][j]))
P("  by network, lead 1 / 7: " + "; ".join(f"{g}: vs measurements {S['leads']['1'][g]['ss_0']['v']:.2f} [{S['leads']['1'][g]['ss_0']['ci_day'][0]:.2f}, {S['leads']['1'][g]['ss_0']['ci_day'][1]:.2f}] / {S['leads']['7'][g]['ss_0']['v']:.2f}, "
                                          f"vs estimates {S['leads']['1'][g]['ss_1']['v']:.2f} / {S['leads']['7'][g]['ss_1']['v']:.2f}, inside {S['leads']['1'][g]['cover_0']['v']:.3f} / {S['leads']['7'][g]['cover_0']['v']:.3f}, "
                                          f"estimates vs measurements RMSE {S['leads']['1'][g]['ghap_rmse']['v']:.1f}" for g in S["groups"] if g != A))
P("")
P("3. CRPS on the station-days of the manuscript (figdata_station_crps.py)")
for h in range(1, 8):
    r = st(h); b = st(h, SB)
    s = (f"  lead {h}: n {int(r['p_n']['v']):,} | Foehn CRPS {f(r['p_crps'])} | climatological ensemble CRPS {f(r['p_clim_crps'])} | change {f(r['p_crps_vs_climens'], 1, sign=True)} % (cells {r['p_crps_vs_climens']['ci_cell'][0]:+.1f}, {r['p_crps_vs_climens']['ci_cell'][1]:+.1f})"
         f" | MAE of the mean {f(r['p_mae'])}, of the climatological mean {f(r['p_clim_mae'])} | before the CRPS training: CRPS {b['p_crps']['v']:.2f} ({100 * (r['p_crps']['v'] / b['p_crps']['v'] - 1):+.1f} % with it), vs climatological ensemble {f(b['p_crps_vs_climens'], 1, sign=True)} %")
    if h <= 4:
        s += (f" | CAMS MAE {f(r['p_cams_mae'])}, Aurora MAE {f(r['p_aur_mae'])} | MAE of the mean vs CAMS {f(r['p_mae_vs_cams'], 1, sign=True)} %, vs Aurora {f(r['p_mae_vs_aur'], 1, sign=True)} %"
              f" | CRPS vs CAMS MAE {f(r['p_crps_vs_camsmae'], 1, sign=True)} %, vs Aurora MAE {f(r['p_crps_vs_aurmae'], 1, sign=True)} %")
    P(s)
P("")
P("4. Sixteen members and the noise fields [MN]")
for tag in ("ea16", "gir16"):
    a = M16["A"][tag]["leads"]["all"]
    c = lambda e, q: f"{e[q]:.3f}" + (f" [{e[q + '_ci'][0]:.3f}, {e[q + '_ci'][1]:.3f}]" if q + "_ci" in e else "")
    P(f"  {tag} ({M16['A'][tag]['what']}; {M16['A'][tag]['stations']} stations, {int(a['M16_v0']['n']):,} station-days): inside the range, 16 members: measurements {c(a['M16_v0'], 'cover')}, estimates {c(a['M16_v1'], 'cover')} (0.882 expected)"
      f" | first four: {c(a['M4_v0'], 'cover')}, {c(a['M4_v1'], 'cover')} (0.600 expected) | spread/error 16: {a['M16_v0']['ss']:.2f} / {a['M16_v1']['ss']:.2f}; first four {a['M4_v0']['ss']:.2f} / {a['M4_v1']['ss']:.2f}")
    b = M16["B"][tag]
    P(f"      quartets of the 16 noise fields (1,820): " + "; ".join(f"{q}: vs measurements mean {b['mean'][0][i]:.3f} sd {b['sd'][0][i]:.3f} (independent draws {b['sd_independent'][0][i]:.3f} = {100 * b['sd_independent'][0][i] / b['mean'][0][i]:.1f} %), members 0-3 {b['members_0_3'][0][i]:.3f}"
                                                                  for i, q in enumerate(b["stat"])))
y = M16["B"]["year"]
P(f"  4-member evaluation of the year: same member highest at a station in {100 * y['1']['same_top_member_share_median']:.0f} % of the forecasts (median station), {100 * y['1']['p10']:.0f} % at the 10th percentile of stations; "
  f"{100 * y['1']['fixed_share_of_variance']:.0f} % of the variance of the standardised departures constant in time; pattern correlation between start days {y['1']['pattern_correlation_between_start_days'][0]:.2f}, "
  f"between leads 1 and 7 {y['lead1_vs_lead7_same_start_day'][0]:.2f}")
r = st(1)
P(f"  jackknife over the four noise fields (lead 1, all stations) [SF]: CRPS {r['crps_0']['v']:.3f} se {r['crps_0']['jk_se']:.3f} ({100 * r['crps_0']['jk_se'] / r['crps_0']['v']:.1f} %); RMSE {r['rmse_0']['v']:.2f} se {r['rmse_0']['jk_se']:.3f} "
  f"({100 * r['rmse_0']['jk_se'] / r['rmse_0']['v']:.1f} %); spread/error {r['ss_0']['v']:.3f} se {r['ss_0']['jk_se']:.3f}; RMSE of the four members one by one " + " / ".join(f"{x:.2f}" for x in r["mem_rmse_0"]["v"]))

# ---- sensitivity: measurements above 500 ug m-3
P("")
P("5. Weight of a few measurements in the squared error at the stations (data/st_final.npz; measurement above 500 ug m-3 = 50 times the European mean)")
ST = stations(); OBS, KEEP, SRC = ST["obs"], ST["keep"], ST["src"]
z = np.load(DATA / "st_final.npz"); K = z["keys"]
D["outliers"] = {}
for h in range(1, 8):
    rows = np.where(K[:, 1] == h)[0]
    E = z["ens"][rows][:, KEEP].astype(np.float32); G_ = z["ghap"][rows][:, KEEP].astype(np.float32); Y = OBS[KEEP][:, K[rows, 2]].T
    w = np.isfinite(Y) & np.isfinite(G_) & np.isfinite(E).all(-1)
    mean = E.mean(-1); var = E.var(-1, ddof=1)
    rec = {}
    for nm, col in [("All official networks", np.ones(KEEP.sum(), bool))] + [(NETS[s], SRC[KEEP] == s) for s in NETS]:
        ww = w & col[None]
        y_, m_, v_ = Y[ww], mean[ww], var[ww]
        se = (m_ - y_) ** 2; big = y_ > 500
        o = np.sort(se)[::-1]; n = len(se)
        rec[nm] = dict(n=int(n), above_500=int(big.sum()), equal_999=int((y_ == 999).sum()), share_se_above_500=float(se[big].sum() / se.sum()), share_se_top_0p1pct=float(o[: max(1, n // 1000)].sum() / se.sum()),
                       ss_all=float(np.sqrt(1.25 * v_.mean() / se.mean())), ss_without_above_500=float(np.sqrt(1.25 * v_[~big].mean() / se[~big].mean())),
                       rmse_all=float(np.sqrt(se.mean())), rmse_without_above_500=float(np.sqrt(se[~big].mean())))
    D["outliers"][h] = rec
    a, e = rec["All official networks"], rec["Europe (EEA)"]
    P(f"  lead {h}: all networks: {a['above_500']} of {a['n']:,} measurements above 500 ({a['equal_999']} equal to 999) hold {100 * a['share_se_above_500']:.0f} % of the squared error; spread/error {a['ss_all']:.2f} -> {a['ss_without_above_500']:.2f} without them; "
      f"RMSE {a['rmse_all']:.2f} -> {a['rmse_without_above_500']:.2f} | Europe: {e['above_500']} of {e['n']:,} hold {100 * e['share_se_above_500']:.0f} %; spread/error {e['ss_all']:.2f} -> {e['ss_without_above_500']:.2f}; RMSE {e['rmse_all']:.2f} -> {e['rmse_without_above_500']:.2f}; "
      f"top 0.1 % of station-days hold {100 * e['share_se_top_0p1pct']:.0f} % (Europe), {100 * a['share_se_top_0p1pct']:.0f} % (all)")
(RES / "digest.txt").write_text("\n".join(out) + "\n")
json.dump(D, open(RES / "digest.json", "w"), indent=1)

# ---- tables for the Supplementary Information
with open(RES / "supp_table_reliability.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["verification", "threshold_ug_m3", "lead_days", "base_rate", "members_above", "forecast_probability", "n_forecasts", "observed_frequency", "ci_low", "ci_high",
                "brier_skill_vs_pooled_frequency", "bss_ci_low", "bss_ci_high", "brier_skill_vs_local_climatological_frequency", "bssc_ci_low", "bssc_ci_high",
                "three_members_to_fourth_member", "three_members_to_verification"])
    for j, T in enumerate(THR):
        for nm, hs, get in (("1 km estimates, populated pixels", (1, 3), lambda h: (pop(h), "rel_freq", "base", "bss_base", None, "ho_freq")),
                            ("1 km estimates, station pixels", (1, 3, 7), lambda h: (st(h), "rel_freq_1", "base_1", "bss_base_1", "bss_clim_1", "ho_freq_1")),
                            ("measurements, stations", (1, 3, 7), lambda h: (st(h), "rel_freq_0", "base_0", "bss_base_0", "bss_clim_0", "ho_freq_0"))):
            for h in hs:
                r, kf, kb, ks, kc, kh = get(h)
                for k in range(5):
                    w.writerow([nm, T, h, f"{r[kb]['v'][j]:.4f}", k, k / 4, f"{r['rel_n']['v'][j][k]:.0f}", f"{r[kf]['v'][j][k]:.4f}", f"{r[kf]['ci_day'][0][j][k]:.4f}", f"{r[kf]['ci_day'][1][j][k]:.4f}",
                                f"{r[ks]['v'][j]:.3f}", f"{r[ks]['ci_day'][0][j]:.3f}", f"{r[ks]['ci_day'][1][j]:.3f}"]
                               + ([f"{r[kc]['v'][j]:.3f}", f"{r[kc]['ci_day'][0][j]:.3f}", f"{r[kc]['ci_day'][1][j]:.3f}"] if kc else ["", "", ""])
                               + ([f"{r['ho_ref']['v'][j][k]:.4f}", f"{r[kh]['v'][j][k]:.4f}"] if k < 4 else ["", ""]))
with open(RES / "supp_table_spread_error.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["sample", "verification", "lead_days", "n", "spread_to_error", "ci_low", "ci_high", "inside_member_range", "rank_1", "rank_2", "rank_3", "rank_4", "rank_5", "rmse", "bias"])
    for h in range(1, 8):
        w.writerow(["land pixels of the nine regions, 2022", "1 km estimates", h, f"{L22[str(h)][NINE]['n']:.0f}", f"{L22[str(h)][NINE]['ss']:.3f}", f"{L22[str(h)][NINE]['ss_ci'][0]:.3f}", f"{L22[str(h)][NINE]['ss_ci'][1]:.3f}",
                    "", "", "", "", "", "", f"{L22[str(h)][NINE]['rmse']:.2f}", f"{L22[str(h)][NINE]['bias']:.2f}"])
    for r_ in regs:
        for h in range(1, 8):
            e = L22[str(h)][r_]
            w.writerow([f"land pixels, {r_}, 2022", "1 km estimates", h, f"{e['n']:.0f}", f"{e['ss']:.3f}", f"{e['ss_ci'][0]:.3f}", f"{e['ss_ci'][1]:.3f}", "", "", "", "", "", "", f"{e['rmse']:.2f}", f"{e['bias']:.2f}"])
    for h in (1, 3):
        for r_ in EST["pop"]["regions"][::-1]:
            e = EST["pop"]["leads"][str(h)][r_]["pixels"]
            w.writerow([f"populated pixels, {r_}", "1 km estimates", h, f"{e['n']['v']:.0f}", f"{e['ss']['v']:.3f}", f"{e['ss']['ci_day'][0]:.3f}", f"{e['ss']['ci_day'][1]:.3f}", f"{e['cover']['v']:.3f}"]
                       + [f"{x:.3f}" for x in e["rank"]["v"]] + [f"{e['rmse']['v']:.2f}", f"{e['bias']['v']:.2f}"])
    for g in [A] + [g for g in S["groups"] if g != A]:
        for h in range(1, 8):
            e = S["leads"][str(h)][g]
            for v, nm in ((1, "1 km estimates at the station pixel"), (0, "measurements")):
                w.writerow([f"stations, {g}", nm, h, f"{e['n']['v']:.0f}", f"{e[f'ss_{v}']['v']:.3f}", f"{e[f'ss_{v}']['ci_day'][0]:.3f}", f"{e[f'ss_{v}']['ci_day'][1]:.3f}", f"{e[f'cover_{v}']['v']:.3f}"]
                           + [f"{x:.3f}" for x in e[f"rank_{v}"]["v"]] + [f"{e[f'rmse_{v}']['v']:.2f}", f"{e[f'bias_{v}']['v']:.2f}"])
with open(RES / "supp_table_crps.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["lead_days", "station_days", "foehn_crps", "ci_low", "ci_high", "foehn_mae_of_mean", "climatological_ensemble_crps", "climatological_mean_mae", "crps_change_vs_climatological_ensemble_pct", "ci_low", "ci_high",
                "cams_mae", "aurora_mae", "mae_change_vs_cams_pct", "ci_low", "ci_high", "mae_change_vs_aurora_pct", "ci_low", "ci_high", "foehn_crps_before_crps_training"])
    for h in range(1, 8):
        r = st(h); b = st(h, SB)
        row = [h, f"{r['p_n']['v']:.0f}", f"{r['p_crps']['v']:.3f}", f"{r['p_crps']['ci_day'][0]:.3f}", f"{r['p_crps']['ci_day'][1]:.3f}", f"{r['p_mae']['v']:.3f}", f"{r['p_clim_crps']['v']:.3f}", f"{r['p_clim_mae']['v']:.3f}",
               f"{r['p_crps_vs_climens']['v']:.1f}", f"{r['p_crps_vs_climens']['ci_day'][0]:.1f}", f"{r['p_crps_vs_climens']['ci_day'][1]:.1f}"]
        row += ([f"{r['p_cams_mae']['v']:.3f}", f"{r['p_aur_mae']['v']:.3f}", f"{r['p_mae_vs_cams']['v']:.1f}", f"{r['p_mae_vs_cams']['ci_day'][0]:.1f}", f"{r['p_mae_vs_cams']['ci_day'][1]:.1f}",
                 f"{r['p_mae_vs_aur']['v']:.1f}", f"{r['p_mae_vs_aur']['ci_day'][0]:.1f}", f"{r['p_mae_vs_aur']['ci_day'][1]:.1f}"] if h <= 4 else [""] * 8) + [f"{b['p_crps']['v']:.3f}"]
        w.writerow(row)
print("\n".join(out))
