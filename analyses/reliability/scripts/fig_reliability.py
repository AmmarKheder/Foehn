#!/usr/bin/env python3
"""Figure (Supplementary Information or main text): reliability of Foehn's ensemble, 2022, against the 1 km estimates and
against the measurements of the 6,324 official stations (panel style of experiments/figures/figED_fair.py).
a-c, reliability diagrams of the probability of exceeding 15, 37.5 and 75 ug m-3 (WHO 24-h guideline level, interim
     targets 3 and 1): observed frequency against the forecast probability (0 to 4 of the four members above the
     threshold), against the estimates at the 8.05 million populated 1 km pixels (1 and 3 days ahead) and against the
     measurements at the stations (1, 3 and 7 days ahead); whiskers, 95 % intervals (block bootstrap over start days);
     inset, share of the forecasts in each probability class, 1 day ahead (logarithmic scale).
d,   rank histogram of the verification among the four members, 1 day ahead (ticks: 7 days ahead at the stations, 3 days
     at the populated pixels).
e,   spread-to-error ratio by lead time: land pixels of the nine regions and populated pixels against the estimates,
     station pixels against the estimates and against the measurements, and the model before the CRPS training.
f,   spread-to-error ratio at the stations of each network, 1 day ahead (ticks: 7 days ahead), against the measurements
     and against the estimates at the same pixel.
g,   share of the verifications inside the range of the members against its expected value (M - 1) / (M + 1), for the
     four members of the evaluation of the year and for the 16-member evaluations (East Asia, 28 start days of January;
     Europe, 12 start days of July, 1 to 3 days ahead) with their first four members.
Reads results/{station_scores_final,station_scores_before,estimates_scores,members16_noise}.json (this directory);
writes figs/fig_reliability.{pdf,png} and figs/fig_reliability_data.json. Imports the house style read-only.
    python -B fig_reliability.py"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
FIG = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))
sys.path.insert(0, str(FIG))
from nature_style import DOUBLE, MM, plt  # noqa: E402  (rcParams of the paper; nothing is saved through nature_style)
import gc_style as GC  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
RES = HERE / f"results{SFX}"
S = json.load(open(RES / "station_scores_final.json")); SB = json.load(open(HERE / "results" / "station_scores_before.json"))   # 'before' (e02): stored
EST = json.load(open(RES / "estimates_scores.json")); M16 = json.load(open(RES / "members16_noise.json"))
A, PP, NINE = "All official networks", "All populated pixels", "Nine regions pooled"
BLUE, LBLUE, INK, GREY = GC.FOEHN, "#8FB2DD", "#111111", "#999999"
LS = {1: "-", 3: (0, (3, 1.6)), 7: (0, (1, 1.2))}
THR = S["thresholds"]
PK = np.arange(5) / 4
PROV = {}
LETTERS = os.environ.get("LETTERS", "fair")     # "fair": bold letter in the title as figED_fair.py; "before": 8 pt letter, gc_style.letter_before
TITLES = []


def title(ax, letter, text):
    if LETTERS == "fair":
        ax.set_title(f"$\\bf{{{letter}}}$  {text}", fontsize=6.5, loc="left", pad=3)
    else:
        TITLES.append((ax.set_title(text, fontsize=6.5, loc="left", pad=3), letter))


def clean(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=5.5, length=2)


def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def main():
    W, H = DOUBLE, 121 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    st = lambda h: S["leads"][str(h)][A]
    pop = lambda h: EST["pop"]["leads"][str(h)][PP]["pixels"]
    # ------------------------------------------------------------- a-c reliability diagrams
    for j, T in enumerate(THR):
        ax = fig.add_axes([fx(12 + 62 * j), fy(63), fx(44), fy(44)])
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
        PROV[f"reliability_{T:g}"] = {}
        for h in (1, 3):
            r = pop(h); f = arr(r["rel_freq"]["v"][j]); lo, hi = arr(r["rel_freq"]["ci_day"][0][j]), arr(r["rel_freq"]["ci_day"][1][j])
            ax.errorbar(PK, f, yerr=[f - lo, hi - f], color=BLUE, ls=LS[h], marker="s", ms=2.5, lw=1.3, elinewidth=0.5, capsize=0, zorder=4)
            PROV[f"reliability_{T:g}"][f"estimates_populated_pixels_lead{h}"] = dict(observed=f.tolist(), low=lo.tolist(), high=hi.tolist(), n=r["rel_n"]["v"][j],
                                                                                    bss_pooled_frequency=r["bss_base"]["v"][j], bss_pixel_frequency_2022=r["bss_pixel_frequency"]["v"][j])
        for h in (1, 3, 7):
            r = st(h); f = arr(r["rel_freq_0"]["v"][j]); lo, hi = arr(r["rel_freq_0"]["ci_day"][0][j]), arr(r["rel_freq_0"]["ci_day"][1][j])
            ax.errorbar(PK, f, yerr=[f - lo, hi - f], color=INK, ls=LS[h], marker="o", ms=2.9, mfc="white", mew=0.7, lw=0.8, elinewidth=0.5, capsize=0, zorder=5)
            PROV[f"reliability_{T:g}"][f"measurements_stations_lead{h}"] = dict(observed=f.tolist(), low=lo.tolist(), high=hi.tolist(), n=r["rel_n"]["v"][j],
                                                                               bss_pooled_frequency=r["bss_base_0"]["v"][j], bss_local_climatological_frequency=r["bss_clim_0"]["v"][j],
                                                                               bss_local_climatological_frequency_ci=[r["bss_clim_0"]["ci_day"][0][j], r["bss_clim_0"]["ci_day"][1][j]],
                                                                               three_members_to_fourth=r["ho_ref"]["v"][j], three_members_to_measurement=r["ho_freq_0"]["v"][j])
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
        ax.set_xticks(PK); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        clean(ax)
        ax.set_xlabel("Forecast probability (share of the four members)", fontsize=6, labelpad=1.5)
        if j == 0:
            ax.set_ylabel("Observed frequency", fontsize=6)
        title(ax, "abc"[j], f"Exceedance of {T:g} µg m$^{{-3}}$")
        # inset: share of the forecasts in each probability class, 1 day ahead
        ia = ax.inset_axes([0.17, 0.67, 0.34, 0.26])
        npx = arr(pop(1)["rel_n"]["v"][j]); nst = arr(st(1)["rel_n"]["v"][j])
        ia.bar(np.arange(5) - 0.2, npx / npx.sum(), width=0.38, color=BLUE, lw=0)
        ia.bar(np.arange(5) + 0.2, nst / nst.sum(), width=0.38, color="white", edgecolor=INK, lw=0.5)
        ia.set_yscale("log"); ia.set_ylim(1e-3, 1.5); ia.set_yticks([1e-3, 1e-2, 1e-1, 1]); ia.set_yticklabels(["0.1%", "1%", "10%", "100%"])
        ia.set_xticks([0, 2, 4]); ia.set_xticklabels(["0", "0.5", "1"])
        ia.tick_params(labelsize=4.6, length=1.5, width=0.4, pad=1); ia.minorticks_off()
        for s_ in ("top", "right"):
            ia.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ia.spines[s_].set_linewidth(0.4)
        ia.patch.set_alpha(0.0)
        ia.set_title("Share of forecasts", fontsize=4.8, pad=1.5, loc="left")
    # ------------------------------------------------------------- d rank histogram
    ad = fig.add_axes([fx(12), fy(9.5), fx(34), fy(37)])
    x = np.arange(5)
    rp1, rp3 = arr(pop(1)["rank"]["v"]), arr(pop(3)["rank"]["v"])
    re1, re7 = arr(st(1)["rank_1"]["v"]), arr(st(7)["rank_1"]["v"])
    rm1, rm7 = arr(st(1)["rank_0"]["v"]), arr(st(7)["rank_0"]["v"])
    ad.bar(x - 0.27, rp1, width=0.25, color=BLUE, lw=0, zorder=3)
    ad.bar(x, re1, width=0.25, color=LBLUE, lw=0, zorder=3)
    ad.bar(x + 0.27, rm1, width=0.25, color="white", edgecolor=INK, lw=0.6, zorder=3)
    for dx, v in ((-0.27, rp3), (0.0, re7), (0.27, rm7)):
        ad.plot(x + dx, v, ls="none", marker="D", ms=1.9, mfc="white", mec="#333333", mew=0.5, zorder=5)
    ad.axhline(0.2, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ad.set_xticks(x); ad.set_xticklabels(["1", "2", "3", "4", "5"]); ad.set_ylim(0, 0.45); ad.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ad.set_xlabel("Rank of the verification among the members", fontsize=6, labelpad=1.5); ad.set_ylabel("Frequency", fontsize=6); clean(ad)
    title(ad, "d", "Rank histogram")
    PROV["rank_histogram"] = dict(estimates_populated_pixels_lead1=rp1.tolist(), estimates_populated_pixels_lead3=rp3.tolist(), estimates_station_pixels_lead1=re1.tolist(),
                                  estimates_station_pixels_lead7=re7.tolist(), measurements_stations_lead1=rm1.tolist(), measurements_stations_lead7=rm7.tolist())
    # ------------------------------------------------------------- e spread-to-error by lead
    ae = fig.add_axes([fx(60), fy(9.5), fx(38), fy(37)])
    leads = np.arange(1, 8)
    land = EST["land"]["2022 final"]["leads"]; land_b = EST["land"]["2022 before CRPS training"]["leads"]
    v = np.array([land[str(h)][NINE]["ss"] for h in leads]); lo = np.array([land[str(h)][NINE]["ss_ci"][0] for h in leads]); hi = np.array([land[str(h)][NINE]["ss_ci"][1] for h in leads])
    ae.fill_between(leads, lo, hi, color=BLUE, alpha=0.15, lw=0, zorder=2)
    ae.plot(leads, v, color=BLUE, marker="o", ms=2.6, lw=1.3, zorder=5)
    vb = np.array([land_b[str(h)][NINE]["ss"] for h in leads])
    ae.plot(leads, vb, color=BLUE, ls=(0, (3, 2)), marker="o", ms=2.0, lw=0.9, mfc="white", zorder=4)
    pv = [(h, pop(h)["ss"]["v"], pop(h)["ss"]["ci_day"]) for h in (1, 3)]
    for h, val, c in pv:
        ae.errorbar([h + 0.16], [val], yerr=[[val - c[0]], [c[1] - val]], color=BLUE, marker="s", ms=2.6, mfc=BLUE, lw=0, elinewidth=0.5, capsize=0, zorder=6)
    se = np.array([st(h)["ss_1"]["v"] for h in leads]); sm = np.array([st(h)["ss_0"]["v"] for h in leads])
    sml = np.array([st(h)["ss_0"]["ci_day"][0] for h in leads]); smh = np.array([st(h)["ss_0"]["ci_day"][1] for h in leads])
    smb = np.array([SB["leads"][str(h)][A]["ss_0"]["v"] for h in leads])
    ae.plot(leads, se, color=LBLUE, marker="o", ms=2.6, lw=1.1, zorder=4)
    ae.fill_between(leads, sml, smh, color=INK, alpha=0.08, lw=0, zorder=2)
    ae.plot(leads, sm, color=INK, marker="o", ms=2.9, mfc="white", mew=0.7, lw=0.8, zorder=5)
    ae.plot(leads, smb, color=INK, ls=(0, (3, 2)), marker="o", ms=2.0, mfc="white", mew=0.5, lw=0.6, zorder=4)
    ae.axhline(1, color=GREY, lw=0.6, ls=(0, (3, 2)))
    ae.set_ylim(0, 1.2); ae.set_xticks(leads); ae.set_xlabel("Lead time (days)", fontsize=6, labelpad=1.5); ae.set_ylabel("Spread / RMSE", fontsize=6); clean(ae)
    title(ae, "e", "Spread and error")
    PROV["spread_to_error"] = dict(leads=leads.tolist(), estimates_land_nine_regions=v.tolist(), estimates_land_ci=[lo.tolist(), hi.tolist()], estimates_land_before_crps_training=vb.tolist(),
                                   estimates_populated_pixels={str(h): [val, c] for h, val, c in pv}, estimates_station_pixels=se.tolist(), measurements_stations=sm.tolist(),
                                   measurements_stations_ci=[sml.tolist(), smh.tolist()], measurements_stations_before_crps_training=smb.tolist())
    # ------------------------------------------------------------- f by network
    af = fig.add_axes([fx(118), fy(9.5), fx(26), fy(37)])
    nets = [g for g in S["groups"] if g != A]
    nets.sort(key=lambda g: S["leads"]["1"][g]["ss_0"]["v"])
    PROV["spread_to_error_by_network"] = {}
    for k, g in enumerate(nets):
        r1, r7 = S["leads"]["1"][g], S["leads"]["7"][g]
        for key, col, mfc, dy in (("ss_1", BLUE, BLUE, 0.14), ("ss_0", INK, "white", -0.14)):
            val, c = r1[key]["v"], r1[key]["ci_day"]
            af.errorbar([val], [k + dy], xerr=[[val - c[0]], [c[1] - val]], color=col, marker="o", ms=2.8, mfc=mfc, mew=0.7, lw=0, elinewidth=0.6, capsize=0, zorder=5)
            af.plot([r7[key]["v"]], [k + dy], ls="none", marker="D", ms=1.9, mfc="white", mec="#333333", mew=0.5, zorder=6)
        PROV["spread_to_error_by_network"][g] = dict(measurements_lead1=[r1["ss_0"]["v"], r1["ss_0"]["ci_day"]], estimates_lead1=[r1["ss_1"]["v"], r1["ss_1"]["ci_day"]],
                                                     measurements_lead7=r7["ss_0"]["v"], estimates_lead7=r7["ss_1"]["v"], stations=r1["n_stations"])
    af.axvline(1, color=GREY, lw=0.6, ls=(0, (3, 2)))
    af.set_yticks(range(len(nets))); af.set_yticklabels([g.replace(" (", "\n(") for g in nets], fontsize=5.2, linespacing=0.95)
    af.set_xlim(0, 1.6); af.set_xticks([0, 0.5, 1, 1.5]); af.set_ylim(-0.6, len(nets) - 0.4)
    af.set_xlabel("Spread / RMSE", fontsize=6, labelpad=1.5); clean(af); af.tick_params(axis="y", length=0)
    title(af, "f", "By network")
    # ------------------------------------------------------------- g member range
    ag = fig.add_axes([fx(155), fy(9.5), fx(26), fy(37)])
    ea, gi = M16["A"]["ea16"]["leads"]["all"], M16["A"]["gir16"]["leads"]["all"]
    cats = [("4", "2022", 0.6, st(1)["cover_1"]["v"], st(1)["cover_0"]["v"], pop(1)["cover"]["v"]),
            ("4", "Jan.", 0.6, ea["M4_v1"]["cover"], ea["M4_v0"]["cover"], None), ("16", "Jan.", 15 / 17, ea["M16_v1"]["cover"], ea["M16_v0"]["cover"], None),
            ("4", "Jul.", 0.6, gi["M4_v1"]["cover"], gi["M4_v0"]["cover"], None), ("16", "Jul.", 15 / 17, gi["M16_v1"]["cover"], gi["M16_v0"]["cover"], None)]
    PROV["member_range"] = []
    for k, (m_, per, exp, ce, cm, cp) in enumerate(cats):
        ag.plot([k - 0.36, k + 0.36], [exp, exp], color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=2)
        if cp is not None:
            ag.plot([k - 0.2], [cp], ls="none", marker="s", ms=2.6, color=BLUE, zorder=5)
        ag.plot([k], [ce], ls="none", marker="o", ms=2.8, color=LBLUE, zorder=5)
        ag.plot([k + 0.2 if cp is not None else k], [cm], ls="none", marker="o", ms=2.9, mfc="white", mew=0.7, color=INK, zorder=6)
        PROV["member_range"].append(dict(members=m_, period=per, expected=exp, estimates_station_pixels=ce, measurements_stations=cm, estimates_populated_pixels=cp))
    for xsep in (0.5, 2.5):
        ag.axvline(xsep, color="#DDDDDD", lw=0.5, zorder=1)
    ag.set_xticks(range(5)); ag.set_xticklabels([m_ for m_, *_ in cats])
    ag.set_xlim(-0.6, 4.6); ag.set_ylim(0.3, 1.1); ag.set_yticks([0.4, 0.6, 0.8, 1.0]); ag.set_yticklabels(["40%", "60%", "80%", "100%"])
    for xc, txt in ((0.0, "2022"), (1.5, "East Asia,\nJanuary"), (3.5, "Europe,\nJuly")):
        ag.text(xc, 1.085, txt, fontsize=4.9, color="#444444", ha="center", va="top", linespacing=0.95)
    ag.set_xlabel("Members of the ensemble", fontsize=6, labelpad=1.5); ag.set_ylabel("Inside the member range", fontsize=6, labelpad=1.5); clean(ag)
    title(ag, "g", "Range of the members")
    # ------------------------------------------------------------- legend above the panels
    hd = [Line2D([], [], color=BLUE, marker="s", ms=2.5, lw=1.3, label="1 km estimates, populated pixels"),
          Line2D([], [], color=BLUE, marker="o", ms=2.6, lw=1.3, label="1 km estimates, land (e)"),
          Line2D([], [], color=LBLUE, marker="o", ms=2.6, lw=1.1, label="1 km estimates, station pixels"),
          Line2D([], [], color=INK, marker="o", ms=2.9, mfc="white", mew=0.7, lw=0.8, label="Measurements, stations"),
          Line2D([], [], color=GREY, lw=0.6, ls=(0, (3, 2)), label="Reliable ensemble")]
    fig.legend(handles=hd, fontsize=5.6, loc="lower center", bbox_to_anchor=(0.5, fy(117.2)), ncol=5, frameon=False, handlelength=2.4, columnspacing=1.4)
    h2 = [Line2D([], [], color="#444444", ls=LS[1], lw=1.0, label="1 day ahead"), Line2D([], [], color="#444444", ls=LS[3], lw=1.0, label="3 days ahead"),
          Line2D([], [], color="#444444", ls=LS[7], lw=1.0, label="7 days ahead (a–c)"),
          Line2D([], [], ls="none", marker="D", ms=1.9, mfc="white", mec="#333333", mew=0.5, label="7 days ahead; populated pixels, 3 days (d, f)"),
          Line2D([], [], color="#666666", ls=(0, (3, 2)), marker="o", ms=2.0, mfc="white", mew=0.5, lw=0.7, label="Before the CRPS training (e)")]
    fig.legend(handles=h2, fontsize=5.6, loc="lower center", bbox_to_anchor=(0.5, fy(113.4)), ncol=5, frameon=False, handlelength=2.4, columnspacing=1.4)
    if TITLES:
        fig.canvas.draw()
        for t, letter in TITLES:
            GC.letter_before(fig, t, letter)
    out = HERE.parents[1] / "rebuild" / f"figs{SFX}" if SFX else HERE / "figs"; out.mkdir(exist_ok=True)
    name = "fig_reliability" if LETTERS == "fair" else "fig_reliability_letters8pt"
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300)
    json.dump(PROV, open(out / "fig_reliability_data.json", "w"), indent=1)
    print("saved", out / f"{name}.pdf")


if __name__ == "__main__":
    main()
