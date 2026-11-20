#!/usr/bin/env python3
"""LaTeX tables of version 2 for the Supplementary Information, written from results/numbers_v2.json (tables_v2.py); no
number is typed by hand. From a copy of ../make_si.py (function warned(); that script is not modified).
  results/si_table_warned_v2.tex            who is warned: the rows of version 1 and the two bias-corrected CAMS references;
                                            the last column of version 1 (share of wrong warnings where the 40 km mean is
                                            below, %) is replaced by the number of people wrongly warned there and by that
                                            number per hidden person warned
  results/si_table_warned_v2_new_rows.tex   the rows of the two new systems only (to paste into the table of version 1; they
                                            carry the columns of version 1, the share of wrong warnings included)
  results/si_table_warned_v2_with_climatology.tex   the same table with the row of the 1 km climatology alone (diagnostic)
  results/si_table_matched_v2.tex           the references with their decision threshold moved until they warn as many people
                                            as Foehn: reach, correct warnings, hidden people warned, and Foehn minus them
    python make_si_v2.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "results" / "numbers_v2.json").read_text())
MU = r"\,\textmu g\,m$^{-3}$"


def num(v, d=0):
    return f"{v:,.{d}f}".replace(",", "{,}")


def ci(v, d=0):
    """value with its 95 % interval: 186 [171, 200]"""
    return f"{num(v[0], d)} [{num(v[1], d)}, {num(v[2], d)}]"


SYS = (("foehn_p50", "\\model{}, at least 2 of 4 members"), ("foehn_mean", "\\model{}, ensemble mean"), ("foehn_40km", "\\model{}, mean over 40\\,km cells"),
       ("cams", "CAMS forecast"), ("aurora", "Aurora"), ("cams_qm", "CAMS forecast, bias-corrected"),
       ("cams_bcsd", "CAMS forecast, bias-corrected, with the climatological 1\\,km pattern"), ("clim_1km", "Climatology of the 1\\,km estimates alone"),
       ("perfect_40km", "Perfect 40\\,km forecast"))


def head_line(t, h, b, ncol):
    return (f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{Above {t}{MU}, {h} day{'s' if h > 1 else ''} ahead: {num(b['exposed_M'][0])} million people exposed a day, "
            f"{num(b['hidden_M'][0], 1)} million hidden, {num(b['hidden_margin_M'][0], 1)} million hidden with a 10\\% margin}}}} \\\\")


def warned(systems, name, note=""):
    P = N["systems"]; rows = []
    for t in ("15", "37.5", "75"):
        for h in (1, 3):
            W = P[f"thr{t}"][f"lead{h}"]["World"]
            rows.append(head_line(t, h, W["foehn_p50"], 11))
            for s, label in SYS:
                if s not in systems:
                    continue
                d = W[s]; dash = s == "perfect_40km"
                rows.append(" & ".join([label, num(d["warned_M"][0]), ci(d["right_M"]), ci(d["false_M"]), ci(d["reach_pct"], 1), ci(d["correct_pct"], 1),
                                        "0" if dash else ci(d["hidden_warned_M"], 1), "0" if dash else ci(d["hidden_warned_pct"], 1),
                                        "0" if dash else ci(d["hidden_margin_warned_pct"], 1), "0" if dash else ci(d["false_in_silent40_M"]),
                                        "--" if dash else ci(d["false_per_hidden_warned"], 1)]) + r" \\")
    head = ("\\begin{tabular}{lrrrrrrrrrr}\n\\hline\n & Warned & Rightly warned & Wrongly warned & Reach (\\%) & Correct (\\%) & Hidden warned & "
            "Hidden warned (\\%) & With 10\\% margin (\\%) & Wrongly warned where the 40\\,km mean is below & per hidden person warned \\\\\n\\hline\n")
    (HERE / "results" / name).write_text(
        "% Supplementary Table: who is warned, version 2 (v2/make_si_v2.py from v2/results/numbers_v2.json). People a day in millions; 95 % intervals over\n"
        "% target days (moving-block bootstrap, blocks of 7 days, 10,000 resamples). Bias-corrected: quantile mapping of the CAMS forecast towards the 1 km\n"
        "% estimates per 0.25 degree cell and calendar month, fitted on 2017-2020; every resident of a cell receives the cell value, or the cell value\n"
        "% distributed by the climatological 1 km pattern of the estimates. Last two columns: people wrongly warned where the mean of the 40 km cell is\n"
        "% below the threshold, and that number divided by the hidden people warned.\n" + note + head + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def new_rows():
    """the two new systems in the columns of version 1 (../results/si_table_warned.tex)"""
    P = N["systems"]; rows = []
    for t in ("15", "37.5", "75"):
        for h in (1, 3):
            W = P[f"thr{t}"][f"lead{h}"]["World"]
            rows.append(f"% above {t} ug m-3, {h} day{'s' if h > 1 else ''} ahead (after the row of Aurora)")
            for s, label in SYS:
                if s not in ("cams_qm", "cams_bcsd"):
                    continue
                d = W[s]
                rows.append(" & ".join([label, num(d["warned_M"][0]), ci(d["right_M"]), ci(d["false_M"]), ci(d["reach_pct"], 1), ci(d["correct_pct"], 1),
                                        ci(d["hidden_warned_M"], 1), ci(d["hidden_warned_pct"], 1), ci(d["hidden_margin_warned_pct"], 1),
                                        ci(d["false_pct_in_silent40"], 1)]) + r" \\")
    (HERE / "results" / "si_table_warned_v2_new_rows.tex").write_text(
        "% Rows of the two new reference systems in the ten columns of ../results/si_table_warned.tex (v2/make_si_v2.py from v2/results/numbers_v2.json):\n"
        "% Warned & Rightly warned & Wrongly warned & Reach (%) & Correct (%) & Hidden warned & Hidden warned (%) & With 10% margin (%) &\n"
        "% Wrong where the 40 km mean is below (%). People a day in millions; 95 % intervals over target days.\n" + "\n".join(rows) + "\n")


def matched():
    P = N["systems"]; rows = []
    names = (("cams", "CAMS forecast"), ("aurora", "Aurora"), ("cams_qm", "CAMS forecast, bias-corrected"),
             ("cams_bcsd", "CAMS forecast, bias-corrected, with the climatological 1\\,km pattern"))
    for t in ("15", "37.5", "75"):
        for h in (1, 3):
            W = P[f"thr{t}"][f"lead{h}"]["World"]; b = W["foehn_p50"]
            rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{Above {t}{MU}, {h} day{'s' if h > 1 else ''} ahead: \\model{{}} warns {num(b['warned_M'][0])} million people a day, "
                        f"reach {b['reach_pct'][0]:.1f}\\%, correct {b['correct_pct'][0]:.1f}\\%, {num(b['hidden_warned_M'][0], 1)} million hidden people warned}}}} \\\\")
            for s, label in names:
                q = W["equal_number_warned"].get(s)
                if q is None:
                    rows.append(label + " & \\multicolumn{7}{l}{outside the range of decision thresholds examined (0.5 to 2 times the threshold)} \\\\"); continue
                rows.append(" & ".join([label, f"{q['decision_threshold_factor']:.2f}", ci(q["reach_pct"], 1), ci(q["correct_pct"], 1), ci(q["hidden_warned_M"], 1),
                                        ci(q["foehn_p50_minus_reach_pct"], 1), ci(q["foehn_p50_minus_correct_pct"], 1), ci(q["foehn_p50_minus_hidden_warned_M"], 1)]) + r" \\")
    head = ("\\begin{tabular}{lrrrrrrr}\n\\hline\n & Factor on the threshold & Reach (\\%) & Correct (\\%) & Hidden warned & \\model{} minus reference: reach (points) & "
            "correct (points) & hidden warned \\\\\n\\hline\n")
    (HERE / "results" / "si_table_matched_v2.tex").write_text(
        "% Supplementary Table: the reference forecasts at the number of people warned by Foehn (v2/make_si_v2.py from v2/results/numbers_v2.json,\n"
        "% keys systems/thr*/lead*/World/equal_number_warned). The reference warns a resident when its value is above the factor times the threshold;\n"
        "% the factor is the one for which it warns as many people a day as Foehn (linear interpolation between factors 0.5, 0.6, ..., 2).\n"
        "% Hidden people warned in millions a day; 95 % intervals over target days, paired for the differences.\n" + head + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


if __name__ == "__main__":
    base = ("foehn_p50", "foehn_mean", "foehn_40km", "cams", "aurora", "cams_qm", "cams_bcsd", "perfect_40km")
    warned(base, "si_table_warned_v2.tex")
    warned(base + ("clim_1km",), "si_table_warned_v2_with_climatology.tex",
           "% Climatology alone: a resident is warned when the harmonic climatology of the 1 km estimates (2017-2020) in their grid cell is above the threshold.\n")
    new_rows(); matched()
    print("si_table_warned_v2.tex, si_table_warned_v2_with_climatology.tex, si_table_warned_v2_new_rows.tex, si_table_matched_v2.tex")
