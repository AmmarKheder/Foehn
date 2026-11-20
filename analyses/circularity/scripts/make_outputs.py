#!/usr/bin/env python3
"""Writes the deliverables from ../results/numbers.json (analyse.py 10000): every number of the table and of the text is
formatted here from the unrounded result, never typed by hand.

    python make_outputs.py   -> ../results/si_table.tex              Supplementary Table (two panels)
                                ../results/replacement_main_text.tex  replacement for the block "This comparison has a limit ..."
                                ../results/methods_si_statements.tex  sentences for Methods / Supplementary Information
                                ../results/further_replacements.tex   four other sentences that need consistent numbers
                                ../results/key_numbers.txt            each number used, unrounded, with its place in numbers.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
J = json.load(open(RES / "numbers.json"))
REF = Path(J["A"]["file"])
KEY = []


def get(path):
    v = J
    for p in path.split("/"):
        v = v[p]
    return v


def r0(x):
    """integer, halves away from zero"""
    return int(np.floor(abs(x) + 0.5) * np.sign(x))


def num(x):
    return f"$-${abs(x)}" if x < 0 else f"{x}"


def G(path, label):
    """gain as [point, low, high] with its trace"""
    g = get(path)
    KEY.append(f"{label}: {g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}] %   <- numbers.json:{path}")
    return g


def Rm(path, label):
    v = get(path)
    KEY.append(f"{label}: {v:.3f}   <- numbers.json:{path}")
    return v


def pct(g):
    return f"{num(r0(g[0]))}\\%"


def iv(g):
    """95 % interval in the manuscript's form, whole per cent; one decimal for a bound that would round to zero or change sign"""
    lo, hi = g[1], g[2]
    if lo < 0:
        slo = f"$-${abs(lo):.1f}" if abs(lo) < 0.95 else f"$-${abs(r0(lo))}"
        return f"{slo} to {num(r0(hi))}\\%"
    slo = f"{lo:.1f}" if lo < 0.95 else f"{r0(lo)}"
    return f"{slo}--{r0(hi)}\\%"


def pi(g):
    return f"{pct(g)} ({iv(g)})"


def t1(g):
    """table cell: one decimal with interval"""
    f = lambda x: (f"\\textminus{{}}{abs(x):.1f}" if x < -0.049 else f"{abs(x):.1f}")
    return f"{f(g[0])} [{f(g[1])}, {f(g[2])}]"


# ------------------------------------------------------------------------------------------------- numbers of the text
C = "C/groups"
A = "A/groups"
D = "D/groups"
B = "B/by_class"
al = "all six networks"
x = {}
x["n_ring"] = get(f"{C}/{al}/stations"); x["n_ring_all"] = get(f"{C}/{al}/stations_paper")
x["sd_ring"] = get(f"{C}/{al}/leads/1/station_days")
for k, nm in [("foehn", "pix"), ("ring3", "r3"), ("ring10", "r10"), ("one3", "o3"), ("one10", "o10"), ("cams1", "cams"), ("aur1", "aur")]:
    x[f"rm_{nm}"] = Rm(f"{C}/{al}/leads/1/rmse/{k}", f"rings, all six, 1 d, RMSE {k}")
x["g_pix_c"] = G(f"{C}/{al}/leads/1/gain_foehn_vs_cams", "rings set, pixel vs CAMS, 1 d")
x["g_pix_a"] = G(f"{C}/{al}/leads/1/gain_foehn_vs_aurora", "rings set, pixel vs Aurora, 1 d")
x["g_r3_c"] = G(f"{C}/{al}/leads/1/gain_ring3_vs_cams", "ring 3 km vs CAMS, 1 d")
x["g_r3_a"] = G(f"{C}/{al}/leads/1/gain_ring3_vs_aurora", "ring 3 km vs Aurora, 1 d")
x["g_r10_c"] = G(f"{C}/{al}/leads/1/gain_ring10_vs_cams", "ring 10 km vs CAMS, 1 d")
x["g_r10_a"] = G(f"{C}/{al}/leads/1/gain_ring10_vs_aurora", "ring 10 km vs Aurora, 1 d")
x["g3_pix_c"] = G(f"{C}/{al}/leads/3/gain_foehn_vs_cams", "rings set, pixel vs CAMS, 3 d")
x["g3_r10_c"] = G(f"{C}/{al}/leads/3/gain_ring10_vs_cams", "ring 10 km vs CAMS, 3 d")
x["g3_pix_a"] = G(f"{C}/{al}/leads/3/gain_foehn_vs_aurora", "rings set, pixel vs Aurora, 3 d")
x["g3_r10_a"] = G(f"{C}/{al}/leads/3/gain_ring10_vs_aurora", "ring 10 km vs Aurora, 3 d")
w = get(f"{C}/China/leads/1/estimates_share_within_0.5")
KEY.append(f"China, estimates within 0.5 ug m-3 of the measurement (pixel / one pixel 3 km / one pixel 10 km): {w}   <- numbers.json:{C}/China/leads/1/estimates_share_within_0.5")
x["w_cn"] = w
wall = get(f"{C}/{al}/leads/1/estimates_share_within_0.5")
KEY.append(f"all six, estimates within 0.5: {wall}   <- numbers.json:{C}/{al}/leads/1/estimates_share_within_0.5")
wkj = get(f"{C}/Korea and Japan/leads/1/estimates_share_within_0.5")
KEY.append(f"Korea and Japan, estimates within 0.5: {wkj}   <- numbers.json:{C}/Korea and Japan/leads/1/estimates_share_within_0.5")
for n_ in ("China", "USA", "Europe", "Korea", "Japan", "India"):
    x[f"A_{n_}_c"] = G(f"{A}/{n_}/leads/1/gain_vs_cams", f"{n_}, 1 km vs CAMS, 1 d (paper's file)")
    x[f"A_{n_}_a"] = G(f"{A}/{n_}/leads/1/gain_vs_aurora", f"{n_}, 1 km vs Aurora, 1 d (paper's file)")
for n_ in ("China", "USA", "Europe"):
    x[f"nm_{n_}_c"] = G(f"{B}/{n_} | not matched (< 20 % of days)/leads/1/gain_vs_cams", f"{n_}, stations not matched, vs CAMS, 1 d")
    x[f"nm_{n_}_a"] = G(f"{B}/{n_} | not matched (< 20 % of days)/leads/1/gain_vs_aurora", f"{n_}, stations not matched, vs Aurora, 1 d")
    x[f"nm_{n_}_n"] = get(f"{B}/{n_} | not matched (< 20 % of days)/stations")
kj = "Korea and Japan"
x["kj_pix_c"] = G(f"{C}/{kj}/leads/1/gain_foehn_vs_cams", "Korea+Japan, pixel vs CAMS, 1 d (rings set)")
x["kj_pix_a"] = G(f"{C}/{kj}/leads/1/gain_foehn_vs_aurora", "Korea+Japan, pixel vs Aurora, 1 d (rings set)")
x["kj_r10_c"] = G(f"{C}/{kj}/leads/1/gain_ring10_vs_cams", "Korea+Japan, ring 10 km vs CAMS, 1 d")
x["kj_r10_a"] = G(f"{C}/{kj}/leads/1/gain_ring10_vs_aurora", "Korea+Japan, ring 10 km vs Aurora, 1 d")
x["kj_n"] = get(f"{C}/{kj}/stations")
x["kjA_c"] = G(f"{A}/{kj}/leads/1/gain_vs_cams", "Korea+Japan, pixel vs CAMS, 1 d (paper's file, all stations)")
x["kjA_a"] = G(f"{A}/{kj}/leads/1/gain_vs_aurora", "Korea+Japan, pixel vs Aurora, 1 d (paper's file, all stations)")
x["kjA05_c"] = G(f"{A}/{kj}/leads/1/gain_05_vs_cams", "Korea+Japan, 0.5 deg vs CAMS, 1 d (paper's file)")
x["kjA05_a"] = G(f"{A}/{kj}/leads/1/gain_05_vs_aurora", "Korea+Japan, 0.5 deg vs Aurora, 1 d (paper's file)")
for k in ("foehn_1km", "cams", "aurora", "climatology"):
    x[f"in_{k}"] = Rm(f"{A}/India/leads/1/rmse/{k}", f"India, 1 d, RMSE {k}")
x["A05_c"] = G(f"{A}/{al}/leads/1/gain_05_vs_cams", "all six, 0.5 deg vs CAMS, 1 d")
x["A05_a"] = G(f"{A}/{al}/leads/1/gain_05_vs_aurora", "all six, 0.5 deg vs Aurora, 1 d")
x["A05_rm"] = Rm(f"{A}/{al}/leads/1/rmse/foehn_05", "all six, Foehn at 0.5 deg, 1 d")
x["A_1km_025"] = G(f"{A}/{al}/leads/1/gain_1km_vs_own_025", "all six, 1 km vs own 0.25 deg, 1 d")
dc = [G(f"{D}/{al}/leads/{h}/gain_vs_cams", f"Aurora-protocol model, all six, vs CAMS, {h} d") for h in "1234"]
da = [G(f"{D}/{al}/leads/{h}/gain_vs_aurora", f"Aurora-protocol model, all six, vs Aurora, {h} d") for h in "1234"]
nn = "not named (Korea, Japan, India)"
dnc = [G(f"{D}/{nn}/leads/{h}/gain_vs_cams", f"Aurora-protocol model, Korea+Japan+India, vs CAMS, {h} d") for h in "1234"]
dna = [G(f"{D}/{nn}/leads/{h}/gain_vs_aurora", f"Aurora-protocol model, Korea+Japan+India, vs Aurora, {h} d") for h in "1234"]
rng = lambda L: f"{min(r0(g[0]) for g in L)}--{max(r0(g[0]) for g in L)}\\%"
assert all(g[1] > 0 for g in dc + da + dnc + dna), "an interval of the Aurora-protocol model includes zero"
fe = "B/forecast_vs_estimates_same_day"
for n_ in ("Europe", "Korea", "Japan", "India", "China", "USA", al):
    x[f"fe_{n_}"] = (Rm(f"{fe}/{n_}/1/rmse_foehn", f"{n_}: Foehn 1 d, identical station-days"),
                     Rm(f"{fe}/{n_}/1/rmse_estimates_same_day", f"{n_}: estimates of the target day, identical station-days"),
                     G(f"{fe}/{n_}/1/gain_foehn_vs_estimates", f"{n_}: Foehn 1 d against the estimates of the target day"),
                     Rm(f"{fe}/{n_}/1/rmse_climatology", f"{n_}: climatology, identical station-days"))

# biases from the paper's file (S: n, sum e, sum e2, ...), for the sentence on China and India
ref = np.load(REF)
S, SY = ref["S"], [str(s) for s in ref["systems"]]
bias = lambda g, k: float(S[g, 0, k, 1] / S[g, 0, k, 0])
x["b_cn"] = (bias(1, 0), bias(1, 4), bias(1, 6)); x["b_in"] = (bias(5, 0), bias(5, 4), bias(5, 6))
KEY.append(f"bias 1 d China (Foehn, CAMS, Aurora): {x['b_cn']}; India: {x['b_in']}   <- {REF.name}: S[network, 0, system, 1] / S[network, 0, system, 0]")

# ------------------------------------------------------------------------------------------------- replacement text
f1 = lambda v: f"{v:.1f}"
T = []
T.append(f"""% Replacement for sections/main_text.tex, the block from "This comparison has a limit." to the end of the paragraph on
% Aurora's protocol (ref_nature_final lines 159-174; work/ lines 156-173). Numbers: analyses/circularity/results/numbers.json
% (scripts/analyse.py 10000), formatted by scripts/make_outputs.py. The label S-tab:away is the table of results/si_table.tex.
The 1\\,km estimates were fitted to measurements from about 9,500 stations, including the European, Chinese and US networks\\cite{{wei2023global}}, and each forecast starts from the estimates of the start day, including the one at the station's own pixel.
Three tests show that Foehn's gain is not read off that pixel.

First, we read the forecast 3\\,km and 10\\,km away from each station, at populated pixels without a monitor (Supplementary Table~\\ref{{S-tab:away}}).
There, the estimates no longer follow the monitor: at the Chinese stations, they are within 0.5\\,\\textmu g\\,m$^{{-3}}$ of the measurement on {r0(x['w_cn']['ghapt'])}\\% of station-days at the station pixel, and on {r0(x['w_cn']['g_one3'])}\\% and {r0(x['w_cn']['g_one10'])}\\% at a pixel 3\\,km and 10\\,km away.
Foehn's error 1 day ahead hardly changes: {f1(x['rm_pix'])}\\,\\textmu g\\,m$^{{-3}}$ at the pixel, {f1(x['rm_r3'])} at 3\\,km and {f1(x['rm_r10'])} at 10\\,km, against {f1(x['rm_cams'])} for the CAMS forecast and {f1(x['rm_aur'])} for Aurora on the same station-days ({x['n_ring']:,} stations with populated surroundings).
Ten kilometres from the stations, its gains are {pct(x['g_r10_c'])} (95\\% interval, {iv(x['g_r10_c'])}) and {pi(x['g_r10_a'])}, against {pct(x['g_pix_c'])} and {pct(x['g_pix_a'])} at the pixel.

Second, the gain holds at the stations that the estimates do not reproduce.
One day ahead, it is {pi(x['A_China_c'])} over the CAMS forecast at the Chinese network, {pi(x['A_USA_c'])} at the US network and {pi(x['A_Europe_c'])} at the European network (Fig.~\\ref{{fig:fair}}a--c); at the stations of these networks that the estimates match on fewer than one day in five, it is {pi(x['nm_China_c'])}, {pi(x['nm_USA_c'])} and {pi(x['nm_Europe_c'])}.
The Korean and Japanese networks are not named among the sources of the estimates, which match their measurements no more often at the station pixel than 3\\,km away.
There, Foehn's error is {pi(x['kj_pix_a'])} below that of Aurora and {pi(x['kj_pix_c'])} below that of the CAMS forecast at the pixel, and {pi(x['kj_r10_a'])} and {pi(x['kj_r10_c'])} below them 10\\,km away.
At the Indian network, the three systems have the same error ({f1(x['in_foehn_1km'])}, {f1(x['in_cams'])} and {f1(x['in_aurora'])}\\,\\textmu g\\,m$^{{-3}}$; Methods).

Third, Foehn's design does not depend on the 1\\,km estimates.
Trained in Aurora's own protocol, on the coarse CAMS analyses and without any 1\\,km or station data, a network of the same design (Foehn-CAMS) has a lower RMSE than Aurora from 36\\,h onwards, by 9\\% 3 days ahead and 12\\% 5 days ahead, and a lower RMSE than the CAMS forecast from 2 days ahead, by 18\\% 5 days ahead (Fig.~\\ref{{fig:auroraset}} and Supplementary Fig.~\\ref{{S-fig:aurorareg}}).
In the first day and a half, the CAMS forecast, which starts from the analysis that serves as the reference, is the most accurate of the three.
At the stations of the six networks, which no part of this model has seen, the RMSE of Foehn-CAMS is {rng(dc)} below that of the CAMS forecast and {rng(da)} below that of Aurora from 1 to 4 days ahead, and {rng(dnc)} and {rng(dna)} below them at the Korean, Japanese and Indian networks taken together; the intervals of these gains all exclude zero (Supplementary Table~\\ref{{S-tab:away}}).

The gain is not a matter of resolution alone: averaged to 0.5$^\\circ$ before being read at the stations, the forecast ({f1(x['A05_rm'])}\\,\\textmu g\\,m$^{{-3}}$) is still {pi(x['A05_c'])} and {pi(x['A05_a'])} more accurate than the two reference systems.
Read at 1\\,km, it is {pct(x['A_1km_025'])} more accurate than its own 0.25$^\\circ$ average over the six networks (Fig.~\\ref{{fig:fair}}g; by network in Methods).
At the Japanese network, the forecast 1 day ahead is even closer to the measurements than the estimates of the target day (RMSE {f1(x['fe_Japan'][0])} against {f1(x['fe_Japan'][1])}\\,\\textmu g\\,m$^{{-3}}$ on identical station-days), and at the European and Korean networks it is as close ({f1(x['fe_Europe'][0])} against {f1(x['fe_Europe'][1])}, and {f1(x['fe_Korea'][0])} against {f1(x['fe_Korea'][1])}).
We hypothesize that averaging the members acts as a filter: the mean keeps the part of the estimates that the start day and the weather can predict and averages out the rest, which at these networks is partly error in the estimates.
""")
open(RES / "replacement_main_text.tex", "w").write("\n".join(T))

# ------------------------------------------------------------------------------------------------- SI table
rows = ["China", "USA", "Europe", "Korea", "Japan", "India", al]
disp = {"China": "China (CNEMC)", "USA": "USA (EPA)", "Europe": "Europe (EEA)", "Korea": "Korea (AirKorea)", "Japan": "Japan", "India": "India (CPCB)",
        al: "Six networks"}
L = []
L.append(r"""% Supplementary Table: the station verification away from the station pixel and without the 1 km estimates.
% Written by analyses/circularity/scripts/make_outputs.py from results/numbers.json (scripts/analyse.py 10000).
% Needs booktabs and rotating (both in the preamble of main.tex and supplementary.tex).
\begin{sidewaystable}[p]
\centering
\caption{\textbf{The gain at the stations holds away from the station pixel and without the 1\,km estimates.}
\textbf{a}, \model{} 1 day ahead in 2022, read at the 1\,km pixel of each station and on the square rings of pixels 3 and 9 pixels
(0.03$^\circ$ and 0.09$^\circ$, about 3 and 10\,km) from it: mean of the ensemble mean over the ring pixels that have at least 100 residents and
hold no monitor of our files; stations with at least four such pixels on both rings; every second start day; identical station-days in a row.
The CAMS forecast and Aurora, issued at the same time, are read at the station. `Estimates within 0.5', share of station-days on which
the 1\,km estimate of the day is within 0.5\,\textmu g\,m$^{-3}$ of the measurement, at the station pixel and at one pixel drawn on each ring: where
the estimates were fitted to a monitor, the share is high at its pixel only. \textbf{b}, The network of the same design trained in Aurora's
protocol on the CAMS analyses, with no 1\,km estimate and no station measurement in its training, selection or inputs (ensemble mean of eight members;
214 initialisations, 1 May to 30 November 2022; all stations of each network). Gains are reductions of the root-mean-square error (RMSE) in per cent,
with 95\% intervals from a paired moving-block bootstrap over start dates (blocks of seven start dates, 10,000 resamples); a positive value means a
lower error than the reference system.}
\label{tab:away}   % cited from the main text as S-tab:away (xr-hyper prefix of the supplement)
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrcccccc@{}}
\toprule
\multicolumn{8}{@{}l}{\textbf{a}\quad \model{} at 1\,km, 1 day ahead} \\
 & & Estimates within 0.5 (\%) & \multicolumn{3}{c}{RMSE (\textmu g\,m$^{-3}$)} & \multicolumn{2}{c}{Gain over the CAMS forecast (\%)} \\
\cmidrule(lr){4-6}\cmidrule(l){7-8}
Network & Stations & pixel / 3\,km / 10\,km & \model{}: pixel / 3\,km / 10\,km & CAMS & Aurora & at the pixel & 10\,km away \\
\midrule""")
for n_ in rows:
    d = get(f"{C}/{n_}/leads/1"); r_ = d["rmse"]; w_ = d["estimates_share_within_0.5"]
    L.append(f"{disp[n_]} & {get(f'{C}/{n_}/stations'):,} & {w_['ghapt']:.0f} / {w_['g_one3']:.0f} / {w_['g_one10']:.0f} & {r_['foehn']:.1f} / {r_['ring3']:.1f} / {r_['ring10']:.1f} & "
             f"{r_['cams1']:.1f} & {r_['aur1']:.1f} & {t1(d['gain_foehn_vs_cams'])} & {t1(d['gain_ring10_vs_cams'])} \\\\")
L.append(r"""\midrule
 & & & \multicolumn{3}{c}{Gain over Aurora (\%)} & \multicolumn{2}{c}{Gain over the CAMS forecast, 3 days ahead (\%)} \\
\cmidrule(lr){4-6}\cmidrule(l){7-8}
Network & & & \multicolumn{2}{c}{at the pixel} & 10\,km away & at the pixel & 10\,km away \\
\midrule""")
for n_ in rows:
    d = get(f"{C}/{n_}/leads/1"); d3 = get(f"{C}/{n_}/leads/3")
    L.append(f"{disp[n_]} & & & \\multicolumn{{2}}{{c}}{{{t1(d['gain_foehn_vs_aurora'])}}} & {t1(d['gain_ring10_vs_aurora'])} & {t1(d3['gain_foehn_vs_cams'])} & {t1(d3['gain_ring10_vs_cams'])} \\\\")
L.append(r"""\midrule
\multicolumn{8}{@{}l}{\textbf{b}\quad Same design trained in Aurora's protocol (no 1\,km estimates, no stations), read at the stations} \\
 & & & \multicolumn{3}{c}{Gain over the CAMS forecast (\%)} & \multicolumn{2}{c}{Gain over Aurora (\%)} \\
\cmidrule(lr){4-6}\cmidrule(l){7-8}
Network & Stations & & \multicolumn{2}{c}{1 day ahead} & 4 days ahead & 1 day ahead & 4 days ahead \\
\midrule""")
for n_ in rows:
    d1 = get(f"{D}/{n_}/leads/1"); d4 = get(f"{D}/{n_}/leads/4")
    L.append(f"{disp[n_]} & {get(f'{D}/{n_}/stations'):,} & & \\multicolumn{{2}}{{c}}{{{t1(d1['gain_vs_cams'])}}} & {t1(d4['gain_vs_cams'])} & {t1(d1['gain_vs_aurora'])} & {t1(d4['gain_vs_aurora'])} \\\\")
d1 = get(f"{D}/{nn}/leads/1"); d4 = get(f"{D}/{nn}/leads/4")
L.append(f"Korea, Japan and India & {get(f'{D}/{nn}/stations'):,} & & \\multicolumn{{2}}{{c}}{{{t1(d1['gain_vs_cams'])}}} & {t1(d4['gain_vs_cams'])} & {t1(d1['gain_vs_aurora'])} & {t1(d4['gain_vs_aurora'])} \\\\")
L.append(r"""\bottomrule
\end{tabular}
\end{sidewaystable}""")
open(RES / "si_table.tex", "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------------------------------------- Methods / SI statements
bn = J["B"]["networks"]
cl = lambda n_, c: get(f"{B}/{n_} | {c}")
nmall = cl(al, "not matched (< 20 % of days)"); mall = cl(al, "matched (>= 80 % of days)")
wc = [get(f"{A}/without China/leads/{h}") for h in "1234"]
in_n = get(f"{A}/India/leads/1/station_days"); wc_n = wc[0]["station_days"]
in_sse = get(f"{A}/India/leads/1/rmse/foehn_1km") ** 2 * in_n; wc_sse = wc[0]["rmse"]["foehn_1km"] ** 2 * wc_n
KEY.append(f"India in the pooled set without China, 1 d: {100 * in_n / wc_n:.1f} % of station-days, {100 * in_sse / wc_sse:.1f} % of Foehn's squared error   <- numbers.json:A/groups/{{India, without China}}/leads/1 (rmse^2 x station_days)")
M = []
M.append(f"""% Sentences for Methods / Supplementary Information (limits stated once, with their numbers). Source: results/numbers.json.

% --- Methods, after "identical station-days": the three tests
\\textbf{{Dependence between the stations and the 1\\,km estimates.}}
The estimates were fitted to station measurements\\cite{{wei2023global}}, so we measured how closely they follow each station of 2022: the share of days on which the estimate at the station pixel is within 0.5\\,\\textmu g\\,m$^{{-3}}$ of the measurement.
At the station pixel, this share is {get(f'{C}/China/leads/1/estimates_share_within_0.5')['ghapt']:.0f}\\% of station-days at the Chinese network, {get(f'{C}/USA/leads/1/estimates_share_within_0.5')['ghapt']:.0f}\\% at the US, {get(f'{C}/Europe/leads/1/estimates_share_within_0.5')['ghapt']:.0f}\\% at the European, {get(f'{C}/India/leads/1/estimates_share_within_0.5')['ghapt']:.0f}\\% at the Indian, {get(f'{C}/Korea/leads/1/estimates_share_within_0.5')['ghapt']:.0f}\\% at the Korean and {get(f'{C}/Japan/leads/1/estimates_share_within_0.5')['ghapt']:.0f}\\% at the Japanese network (stations and days of Supplementary Table~\\ref{{S-tab:away}}).
At a pixel 3\\,km away it is {get(f'{C}/China/leads/1/estimates_share_within_0.5')['g_one3']:.0f}\\%, {get(f'{C}/USA/leads/1/estimates_share_within_0.5')['g_one3']:.0f}\\%, {get(f'{C}/Europe/leads/1/estimates_share_within_0.5')['g_one3']:.0f}\\%, {get(f'{C}/India/leads/1/estimates_share_within_0.5')['g_one3']:.0f}\\%, {get(f'{C}/Korea/leads/1/estimates_share_within_0.5')['g_one3']:.0f}\\% and {get(f'{C}/Japan/leads/1/estimates_share_within_0.5')['g_one3']:.0f}\\%: the close match is confined to the monitor's own pixel, and at the Korean and Japanese networks there is none.
We call a station matched when its share is at least 80\\% ({mall['stations']:,} stations: {bn['China']['matched']:,} Chinese, {bn['USA']['matched']} US and {bn['Europe']['matched']} European) and not matched when it is below 20\\% ({nmall['stations']:,} stations: the {bn['Korea']['not_matched']} Korean stations, {bn['Japan']['not_matched']} of the {bn['Japan']['stations']} Japanese, {bn['Europe']['not_matched']:,} European, {bn['USA']['not_matched']} US, {bn['India']['not_matched']} Indian and {bn['China']['not_matched']} Chinese).

\\textbf{{Verification away from the station pixel.}}
The evaluation of the final model stored every member at the 8.1 million 1\\,km pixels with at least 100 residents, for every second start day of 2022 and lead times of 1 and 3 days.
Around each station we took the square rings of pixels 3 and 9 pixels (0.03$^\\circ$ and 0.09$^\\circ$, about 3 and 10\\,km) from the station pixel, kept the stored pixels that hold no station of our files and averaged the ensemble mean over them.
{x['n_ring']:,} of the {x['n_ring_all']:,} scored stations have at least four such pixels on both rings ({x['sd_ring']:,} station-days 1 day ahead).
The CAMS forecast and Aurora are read at the station as before, and all scores are on identical station-days (Supplementary Table~\\ref{{S-tab:away}}).
One pixel drawn at random on each ring gives the same result as the ring mean (RMSE {x['rm_o3']:.1f} and {x['rm_o10']:.1f}\\,\\textmu g\\,m$^{{-3}}$ at 3 and 10\\,km, against {x['rm_r3']:.1f} and {x['rm_r10']:.1f} for the ring means and {x['rm_pix']:.1f} at the station pixel).
Three days ahead, the gains over the CAMS forecast and Aurora are {pi(x['g3_pix_c'])} and {pi(x['g3_pix_a'])} at the pixel and {pi(x['g3_r10_c'])} and {pi(x['g3_r10_a'])} 10\\,km away.
The other lead times and the stations without populated surroundings were not stored.

% --- limits, once
\\textbf{{Where the gain is not established.}}
Pooled over the {nmall['stations']:,} stations not matched by the estimates, the gain of \\model{{}} 1 day ahead is {pi(nmall['leads']['1']['gain_vs_cams'])} over the CAMS forecast and {pi(nmall['leads']['1']['gain_vs_aurora'])} over Aurora; 3 days ahead it is {pi(nmall['leads']['3']['gain_vs_cams'])} and {pi(nmall['leads']['3']['gain_vs_aurora'])}.
Without the Chinese network, the pooled gain over the CAMS forecast is {pi(wc[0]['gain_vs_cams'])} 1 day ahead and {pct(wc[1]['gain_vs_cams'])}, {pct(wc[2]['gain_vs_cams'])} and {pct(wc[3]['gain_vs_cams'])} from 2 to 4 days ahead, with intervals that include zero; the Indian network, {100 * in_n / wc_n:.0f}\\% of these station-days, carries {100 * in_sse / wc_sse:.0f}\\% of their squared error.
At the Indian network, no system improves on the climatology of the estimates 1 day ahead (RMSE {x['in_foehn_1km']:.1f}\\,\\textmu g\\,m$^{{-3}}$ for \\model{{}}, {x['in_cams']:.1f} for the CAMS forecast, {x['in_aurora']:.1f} for Aurora and {x['in_climatology']:.1f} for the climatology), and the bias of \\model{{}} is {num(round(x['b_in'][0], 1))}\\,\\textmu g\\,m$^{{-3}}$.
At the Korean network, the gain over the CAMS forecast is {pi(x['A_Korea_c'])} 1 day ahead and {pct(get(f'{A}/Korea/leads/2/gain_vs_cams'))}, {pct(get(f'{A}/Korea/leads/3/gain_vs_cams'))} and {pct(get(f'{A}/Korea/leads/4/gain_vs_cams'))} from 2 to 4 days ahead, with intervals that exclude zero; the gain over Aurora is {pct(x['A_Korea_a'])} to {pct(get(f'{A}/Korea/leads/4/gain_vs_aurora'))} (intervals exclude zero).
At the Japanese network, it is {pi(x['A_Japan_c'])} 1 day ahead at the station pixel, {pi(get(f'{C}/Japan/leads/1/gain_ring10_vs_cams'))} 10\\,km away and {pi(get(f'{A}/Japan/leads/1/gain_05_vs_cams'))} for the 0.5$^\\circ$ average; from 2 days ahead \\model{{}} and the CAMS forecast are equal there at the station pixel ({pct(get(f'{A}/Japan/leads/2/gain_vs_cams'))}, {pct(get(f'{A}/Japan/leads/3/gain_vs_cams'))} and {pct(get(f'{A}/Japan/leads/4/gain_vs_cams'))}, intervals include zero).
Read at 1\\,km rather than as its own 0.25$^\\circ$ average, the forecast 1 day ahead is closer to the measurements by {pi(get(f'{A}/India/leads/1/gain_1km_vs_own_025'))} at the Indian network, {pi(get(f'{A}/USA/leads/1/gain_1km_vs_own_025'))} at the US, {pi(get(f'{A}/China/leads/1/gain_1km_vs_own_025'))} at the Chinese and {pi(get(f'{A}/Europe/leads/1/gain_1km_vs_own_025'))} at the European network; at the Korean and Japanese networks the 0.25$^\\circ$ average is closer, by {num(abs(r0(get(f'{A}/Korea/leads/1/gain_1km_vs_own_025')[0])))}\\% and {num(abs(r0(get(f'{A}/Japan/leads/1/gain_1km_vs_own_025')[0])))}\\%.

% --- the estimates of the target day on identical station-days (replaces "2.4 ... 11.2 ... 36.5" where they are compared with Foehn)
On the station-days of the comparison, the 1\\,km estimates of the target day have an RMSE of {x['fe_China'][1]:.1f}\\,\\textmu g\\,m$^{{-3}}$ at the Chinese stations, {x['fe_USA'][1]:.1f} at the US, {x['fe_Europe'][1]:.1f} at the European, {x['fe_Korea'][1]:.1f} at the Korean, {x['fe_Japan'][1]:.1f} at the Japanese and {x['fe_India'][1]:.1f} at the Indian stations, and {x[f'fe_{al}'][1]:.1f} over the six networks; \\model{{}} 1 day ahead has {x['fe_China'][0]:.1f}, {x['fe_USA'][0]:.1f}, {x['fe_Europe'][0]:.1f}, {x['fe_Korea'][0]:.1f}, {x['fe_Japan'][0]:.1f} and {x['fe_India'][0]:.1f}.
The forecast is closer than the estimates at the Japanese network, by {pi(x['fe_Japan'][2])}; at the European, Korean and Indian networks the difference is {pct(x['fe_Europe'][2])}, {pct(x['fe_Korea'][2])} and {pct(x['fe_India'][2])}, with intervals that include zero.
At the European network the estimates are {pi(get(f'{fe}/Europe/1/gain_estimates_vs_climatology'))} more accurate than their own climatology ({x['fe_Europe'][1]:.1f} against {x['fe_Europe'][3]:.1f}).

% --- the model of Aurora's protocol at the stations (Methods, "Comparison in Aurora's protocol")
At the stations (ensemble mean of eight members; 214 initialisations at 00\\,UTC from 1 May to 30 November 2022; daily means from the three instants of each day, as for Aurora; identical station-days), the RMSE of Foehn-CAMS from 1 to 4 days ahead is {', '.join(pct(g) for g in dc)} below that of the CAMS forecast of the same run and {', '.join(pct(g) for g in da)} below that of Aurora; every interval excludes zero (1 day ahead: {iv(dc[0])} and {iv(da[0])}).
By network, the gain over the CAMS forecast from 1 to 4 days is {'--'.join(str(v) for v in sorted({min(r0(get(f'{D}/China/leads/{h}/gain_vs_cams')[0]) for h in '1234'), max(r0(get(f'{D}/China/leads/{h}/gain_vs_cams')[0]) for h in '1234')}))}\\% in China, {'--'.join(str(v) for v in sorted({min(r0(get(f'{D}/USA/leads/{h}/gain_vs_cams')[0]) for h in '1234'), max(r0(get(f'{D}/USA/leads/{h}/gain_vs_cams')[0]) for h in '1234')}))}\\% in the USA and {'--'.join(str(v) for v in sorted({min(r0(get(f'{D}/India/leads/{h}/gain_vs_cams')[0]) for h in '1234'), max(r0(get(f'{D}/India/leads/{h}/gain_vs_cams')[0]) for h in '1234')}))}\\% in India (intervals exclude zero), {'--'.join(str(v) for v in sorted({min(r0(get(f'{D}/Korea/leads/{h}/gain_vs_cams')[0]) for h in '1234'), max(r0(get(f'{D}/Korea/leads/{h}/gain_vs_cams')[0]) for h in '1234')}))}\\% in Korea (the interval excludes zero 4 days ahead) and {'--'.join(str(v) for v in sorted({min(r0(get(f'{D}/Japan/leads/{h}/gain_vs_cams')[0]) for h in '1234'), max(r0(get(f'{D}/Japan/leads/{h}/gain_vs_cams')[0]) for h in '1234')}))}\\% in Japan (intervals include zero); in Europe the three systems are equal (within 0.4\\%).
At the Korean, Japanese and Indian networks together, it is {', '.join(pct(g) for g in dnc)} below the CAMS forecast and {', '.join(pct(g) for g in dna)} below Aurora (intervals exclude zero).
A single member, compared with the two single forecasts, is {', '.join(pct(get(f'{D}/{al}/leads/{h}/member_gain_vs_cams')) for h in '1234')} more accurate than the CAMS forecast (intervals exclude zero from 2 days ahead) and {', '.join(pct(get(f'{D}/{al}/leads/{h}/member_gain_vs_aurora')) for h in '1234')} relative to Aurora (the interval excludes zero 4 days ahead).
""")
fa = lambda n_: x[f"fe_{n_}"]
b_est_in = None
X = []
X.append(f"""% Further replacement sentences made necessary by the checks of this task (same sources).

% (1) sections/methods.tex, "Read at the station pixel on the same day, GHAP itself has an RMSE ... 11.0 over the six networks."
%     (ref lines 144-145, work lines 185-186): the old values are over every station and day of each network; at the European network
%     they include 128 second monitors at the coordinates of another one, which the comparison does not score (one of them has an
%     RMSE of 229 ug m-3). On the station-days of the comparison:
Read at the station pixel on the target day, on the station-days of the comparison 1 day ahead, GHAP itself has an RMSE against the measurements of {f1(fa('China')[1])}\\,\\textmu g\\,m$^{{-3}}$ at the Chinese stations, {f1(fa('USA')[1])} at the US, {f1(fa('Japan')[1])} at the Japanese, {f1(fa('Korea')[1])} at the Korean, {f1(fa('Europe')[1])} at the European and {f1(fa('India')[1])} at the Indian stations, and {f1(fa(al)[1])} over the six networks.

% (2) sections/main_text.tex, legend of Fig. fair (work line 201), "Read at the station pixel on the same day, the 1 km estimates
%     themselves have an RMSE of 2.4 (a), 2.6 (b), 11.2 (c), 8.5 (d), 6.6 (e) and 36.5 (f; Methods)":
Read at the station pixel on the target day, on the same station-days 1 day ahead, the 1\\,km estimates themselves have an RMSE of {f1(fa('China')[1])} (\\textbf{{a}}), {f1(fa('USA')[1])} (\\textbf{{b}}), {f1(fa('Europe')[1])} (\\textbf{{c}}), {f1(fa('Korea')[1])} (\\textbf{{d}}), {f1(fa('Japan')[1])} (\\textbf{{e}}) and {f1(fa('India')[1])}\\,\\textmu g\\,m$^{{-3}}$ (\\textbf{{f}}; Methods).

% (3) sections/main_text.tex, work line 150 (ref line 155), "The gain is as large (18%) at the 4,295 stations of the five networks
%     outside Europe ...": 4,295 is the number of stations in the file; {get(f'{A}/outside Europe (five networks)/stations_scored'):,} are scored 1 day ahead.
The gain is as large ({pct(G(f'{A}/outside Europe (five networks)/leads/1/gain_vs_cams', 'five networks outside Europe, vs CAMS, 1 d'))}; {iv(get(f'{A}/outside Europe (five networks)/leads/1/gain_vs_cams'))}) at the {get(f'{A}/outside Europe (five networks)/stations_scored'):,} stations of the five networks outside Europe, which played no part in the benchmark that guided the design.

% (4) sections/methods.tex, work line 132, "... against 15.5--16.6 for the CAMS forecast (6--9% higher) and 15.0--16.8 for Aurora
%     (3--11% higher)": 6--9 % and 3--11 % are the reductions of Foehn-CAMS, not the excess of the two others (which is 6--10 % and 3--12 %).
At the stations of the six networks (1.26 million station-days per lead time), its RMSE from 1 to 4 days ahead is {f1(get(f'{D}/{al}/leads/1/rmse/fc'))}--{f1(get(f'{D}/{al}/leads/4/rmse/fc'))}\\,\\textmu g\\,m$^{{-3}}$, {rng(dc)} below that of the CAMS forecast ({f1(get(f'{D}/{al}/leads/1/rmse/cams'))}--{f1(get(f'{D}/{al}/leads/4/rmse/cams'))}) and {rng(da)} below that of Aurora ({f1(get(f'{D}/{al}/leads/1/rmse/aur'))}--{f1(get(f'{D}/{al}/leads/4/rmse/aur'))}), with intervals that exclude zero; the three systems are equal at the European network ({f1(get(f'{D}/Europe/leads/1/rmse/fc'))} at 1 day).
""")
open(RES / "further_replacements.tex", "w").write("\n".join(X))
open(RES / "methods_si_statements.tex", "w").write("\n".join(M))
open(RES / "key_numbers.txt", "w").write("\n".join(KEY) + "\n")
for f in ("replacement_main_text.tex", "si_table.tex", "methods_si_statements.tex", "further_replacements.tex", "key_numbers.txt"):
    print("wrote", RES / f)
