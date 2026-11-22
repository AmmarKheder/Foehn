#!/usr/bin/env python3
"""LaTeX tables for the Supplementary Information, written from results/numbers.json (tables.py); no number is typed by hand.
  results/si_table_sensitivity.tex   the count of hidden people: denominator, reverse count, daily range, block size, origin
                                     of the blocks, margin around the threshold, monitor pixels
  results/si_table_warned.tex        who is warned: Foehn, Foehn on 40 km blocks, the CAMS forecast, Aurora, a perfect 40 km
                                     forecast; thresholds 15, 37.5 and 75 ug m-3; 1 and 3 days ahead
  results/si_table_regions.tex       hidden people and warnings by region (country groups, a partition of the world)
    python make_si.py"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
RES = HERE / f"results{SFX}"
N = json.loads((RES / "numbers.json").read_text())
MU = r"\,\textmu g\,m$^{-3}$"


def ci(v, d=0, unit=""):
    """value with its 95 % interval: 186 [171, 200]"""
    return f"{num(v[0], d)}{unit} [{num(v[1], d)}, {num(v[2], d)}]"


def num(v, d=0):
    return f"{v:,.{d}f}".replace(",", "{,}")


def sensitivity():
    X, S = N["exposure_all_pixels"], N["sensitivity"]
    thr = ("15", "25", "37.5", "50", "75")
    rows = []
    add = lambda name, fn: rows.append(name + " & " + " & ".join(fn(t) for t in thr) + r" \\")
    rows.append(r"\multicolumn{6}{l}{\textit{The count (40\,km blocks, the definition of the paper)}} \\")
    add("People above the threshold at 1\\,km", lambda t: ci(X[f"thr{t}"]["World"]["exposed_M"]))
    add("Hidden: 1\\,km above, 40\\,km mean not above", lambda t: ci(X[f"thr{t}"]["World"]["hidden_M"]))
    add("\\quad share of the people above (\\%)", lambda t: ci(X[f"thr{t}"]["World"]["hidden_pct"], 1))
    add("\\quad daily minimum--maximum", lambda t: f"{X[f'thr{t}']['World']['daily_hidden_M'][0]:.0f}--{X[f'thr{t}']['World']['daily_hidden_M'][-1]:.0f}")
    add("Reverse: 40\\,km mean above, 1\\,km not above", lambda t: ci(X[f"thr{t}"]["World"]["reverse_M"]))
    add("Hidden minus reverse", lambda t: ci(X[f"thr{t}"]["World"]["net_M"]))
    rows.append(r"\multicolumn{6}{l}{\textit{Size of the blocks}} \\")
    add("Hidden by 25\\,km blocks (0.25$^\\circ$)", lambda t: ci(S[f"thr{t}"]["blocks_25px"]["hidden_M"]))
    add("Hidden by 10\\,km blocks (0.1$^\\circ$)", lambda t: ci(S[f"thr{t}"]["blocks_10px"]["hidden_M"]))
    rows.append(r"\multicolumn{6}{l}{\textit{Origin of the 40\,km blocks}} \\")
    add("Shifted by 10, 20 and 30\\,km in both directions", lambda t: " / ".join(num(S[f"thr{t}"][f"blocks_40px_shift{s}"]["hidden_M"][0]) for s in (10, 20, 30)))
    rows.append(r"\multicolumn{6}{l}{\textit{Margin $\delta$: 1\,km above the threshold plus $\delta$, 40\,km mean not above the threshold minus $\delta$}} \\")
    for m, lab in (("rel:0.05", "5\\% of the threshold"), ("rel:0.1", "10\\% of the threshold"), ("rel:0.2", "20\\% of the threshold"),
                   ("abs:2", "2" + MU), ("abs:5", "5" + MU), ("abs:10", "10" + MU)):
        add(f"$\\delta$ = {lab}", lambda t, m=m: "--" if S[f"thr{t}"][f"margin_{m}"]["two_sided_M"][0] < 0.05 else
            ci(S[f"thr{t}"][f"margin_{m}"]["two_sided_M"], 0 if S[f"thr{t}"][f"margin_{m}"]["two_sided_M"][0] >= 10 else 1))
    rows.append(r"\multicolumn{6}{l}{\textit{Monitoring stations (8,110 monitors of 2022)}} \\")
    add("Hidden people in the 1\\,km pixel of a monitor", lambda t: num(S[f"thr{t}"]["monitors_0"]["hidden_M"], 2) + f" ({S[f'thr{t}']['monitors_0']['share_of_hidden_pct']:.1f}\\%)")
    add("Hidden people within 2\\,km of a monitor", lambda t: num(S[f"thr{t}"]["monitors_1"]["hidden_M"], 1) + f" ({S[f'thr{t}']['monitors_1']['share_of_hidden_pct']:.1f}\\%)")
    P = N["systems"]
    rows.append(r"\multicolumn{6}{l}{\textit{Pixels with at least 100 residents (94\% of residents; the pixels at which the forecast is tested)}} \\")
    add("People above the threshold at 1\\,km", lambda t: ci(P[f"thr{t}"]["lead1"]["World"]["foehn_p50"]["exposed_M"]))
    add("Hidden", lambda t: ci(P[f"thr{t}"]["lead1"]["World"]["foehn_p50"]["hidden_M"]))
    add("Hidden with $\\delta$ = 10\\% of the threshold", lambda t: ci(P[f"thr{t}"]["lead1"]["World"]["foehn_p50"]["hidden_margin_M"]))
    head = ("\\begin{tabular}{lrrrrr}\n\\hline\nThreshold (" + MU.replace("\\,", "", 1) + ") & 15 & 25 & 37.5 & 50 & 75 \\\\\n\\hline\n")
    (RES / "si_table_sensitivity.tex").write_text(
        "% Supplementary Table: sensitivity of the count of people hidden by a 40 km grid (make_si.py from results/numbers.json)\n"
        "% People a day, millions; mean over the 365 days of 2022 with 95 % intervals (moving-block bootstrap over days, blocks of 7 days);\n"
        "% last block: mean over 179 target days (every second day), blocks of 4 forecasts.\n" + head + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


SYS = (("foehn_p50", "\\model{}, at least 2 of 4 members"), ("foehn_mean", "\\model{}, ensemble mean"), ("foehn_40km", "\\model{}, mean over 40\\,km blocks"),
       ("cams", "CAMS forecast"), ("aurora", "Aurora"), ("perfect_40km", "Perfect 40\\,km forecast"))


def warned():
    P = N["systems"]; rows = []
    for t in ("15", "37.5", "75"):
        for h in (1, 3):
            W = P[f"thr{t}"][f"lead{h}"]["World"]; b = W["foehn_p50"]
            rows.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{Above {t}{MU}, {h} day{'s' if h > 1 else ''} ahead: {num(b['exposed_M'][0])} million people exposed a day, "
                        f"{num(b['hidden_M'][0], 1)} million hidden, {num(b['hidden_margin_M'][0], 1)} million hidden with a 10\\% margin}}}} \\\\")
            for s, name in SYS:
                d = W[s]
                dash = s == "perfect_40km"
                rows.append(" & ".join([name, num(d["warned_M"][0]), ci(d["right_M"]), ci(d["false_M"]), ci(d["reach_pct"], 1), ci(d["correct_pct"], 1),
                                        "0" if dash else ci(d["hidden_warned_M"], 1), "0" if dash else ci(d["hidden_warned_pct"], 1),
                                        "0" if dash else ci(d["hidden_margin_warned_pct"], 1), "--" if dash else ci(d["false_pct_in_silent40"], 1)]) + r" \\")
    head = ("\\begin{tabular}{lrrrrrrrrr}\n\\hline\n & Warned & Rightly warned & Wrongly warned & Reach (\\%) & Correct (\\%) & Hidden warned & "
            "Hidden warned (\\%) & With 10\\% margin (\\%) & Wrong where the 40\\,km mean is below (\\%) \\\\\n\\hline\n")
    (RES / "si_table_warned.tex").write_text(
        "% Supplementary Table: who is warned (make_si.py from results/numbers.json). People a day in millions; 95 % intervals over target days.\n"
        + head + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def regions():
    P, S = N["systems"]["thr37.5"], N["sensitivity"]["thr37.5"]["groups"]
    order = sorted((g for g in S if S[g]["hidden_M"][0] >= 0.05), key=lambda g: -S[g]["hidden_M"][0])
    rows = []
    for g in order:
        a = S[g]; r1, r3 = P["lead1"][f"G:{g}"], P["lead3"][f"G:{g}"]
        rows.append(" & ".join([g, ci(a["hidden_M"], 1), f"{a['share_of_world_hidden_pct']:.1f}", ci(a["hidden_pct_of_exposed"], 1) if a["hidden_pct_of_exposed"] else "--",
                                num(a["hidden_10km_M"], 1), num(a["hidden_margin10pct_M"], 1),
                                ci(r1["foehn_p50"]["hidden_warned_pct"], 1), ci(r3["foehn_p50"]["hidden_warned_pct"], 1), ci(r1["cams"]["hidden_warned_pct"], 1),
                                f"{r1['foehn_p50']['reach_pct'][0]:.1f} / {r1['cams']['reach_pct'][0]:.1f}",
                                f"{r1['foehn_p50']['correct_pct'][0]:.1f} / {r1['cams']['correct_pct'][0]:.1f}"]) + r" \\")
    head = ("\\begin{tabular}{lrrrrrrrrrr}\n\\hline\nRegion & Hidden (millions a day) & Share of the world's hidden (\\%) & Share of the region's exposed (\\%) & "
            "10\\,km blocks & 10\\% margin & Warned by \\model{}, 1 day (\\%) & 3 days (\\%) & By the CAMS forecast, 1 day (\\%) & Reach, \\model{} / CAMS (\\%) & "
            "Correct, \\model{} / CAMS (\\%) \\\\\n\\hline\n")
    (RES / "si_table_regions.tex").write_text(
        "% Supplementary Table: hidden people by region, above 37.5 ug m-3 (make_si.py from results/numbers.json). Regions: countries grouped as in\n"
        "% regions.py (a partition of the world). Columns 2-6: all pixels with residents, 365 days; columns 7-11: pixels with at least 100 residents, 179 target days.\n"
        + head + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


if __name__ == "__main__":
    warned(); print("si_table_warned.tex")
    if "sensitivity" in N:
        sensitivity(); print("si_table_sensitivity.tex")
        regions(); print("si_table_regions.tex")
    else:
        print("no sensitivity results yet")
