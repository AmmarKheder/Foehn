#!/usr/bin/env python3
"""Numbers of the legends of fig_rt_skill against numbers.json, in both directions (reads only; writes nothing).

(1) forward, as legend_count.py: every entry of numbers.json["legend_text_values"] is printed in the legend in its printed
    form (the two entries of panel f are asked only of a legend that has a panel f);
(2) reverse: once these printed values and the fixed descriptors listed in FIXED are taken out, no digit is left in the
    legend, so a stale number cannot stay unnoticed. The fixed descriptors that numbers.json can confirm (issue days, lead
    days, rings) are confirmed against it.
Legends checked: legend.tex (six panels, a-f) and legend_fig_rt_skill_main.tex (five panels, a-e: the Supplementary figure
with \\label{fig:rtskill}); with --supp, also the caption that carries \\label{fig:rtskill} in that supplementary.tex.

    python -B legend_check.py [--numbers numbers.json] [--supp /path/to/work/supplementary.tex]
Exit status 1 when a legend of the folder fails (the caption of --supp is reported, not enforced)."""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "build_test"))
import wordcount as W  # noqa: E402  (the paper's counting tool, imported read-only)

PANEL_F = ("outside_the_members", "outside_expected")
MONTH = {9: "September", 10: "October", 11: "November"}


def runs(days):
    """'6--20 September and 10 October--11 November 2026' from the list of issue days"""
    d = [dt.date.fromisoformat(s) for s in days]
    out, a, p = [], d[0], d[0]
    for x in d[1:]:
        if (x - p).days != 1:
            out.append((a, p))
            a = x
        p = x
    out.append((a, p))
    txt = [f"{a.day}--{b.day} {MONTH[a.month]}" if a.month == b.month else f"{a.day} {MONTH[a.month]}--{b.day} {MONTH[b.month]}" for a, b in out]
    return " and ".join(txt) + f" {d[-1].year}"


def check(name, caption, num, enforce=True):
    txt = " ".join(W.detex(caption).split())
    has_f = bool(re.search(r"(^|\s)f, ", txt))
    vals = {k: v["printed"] for k, v in num["legend_text_values"].items() if has_f or k not in PANEL_F}
    missing = [(k, p) for k, p in vals.items() if p not in txt]
    # ---- fixed descriptors (what they are, and the value of numbers.json that confirms them when there is one)
    s = num["sample"]
    period = runs(s["issue_day_list"])
    leads = f"lead days {s['lead_days'][0]}--{s['lead_days'][-1]}"
    rings = [w for w in num["d_rings"]["where"] if w.endswith(" km")]                    # ['1 km', '3 km', '10 km']
    ring_txt = "rings " + ", ".join(r.split()[0] for r in rings[:-1]) + f" and {rings[-1]}"
    cell = [w for w in num["d_rings"]["where"] if "cell" in w][0].split()[0]             # '0.25°'
    FIXED = [(period, "issue days of the sample (sample.issue_day_list)"), (leads, "lead days (sample.lead_days)"),
             (ring_txt, "rings of panel d (d_rings.where)"), (f"{rings[-1]} away", "the ring of the number quoted in d"),
             (f"{cell} cell", "cell mean of panel d (d_rings.where)"), ("0.4° input (40 km)", "grid of the coarse forecast"),
             ("40 km forecasts", "the two coarse forecasts"), ("2° cells", "space blocks of the bootstrap (sc_lib)"),
             ("2° cell", "cells of panel a (sc_lib)"), ("Intervals, 95%", "2.5th-97.5th percentiles (sc_lib.pct)"),
             ("1 km", "grid of the forecast and of the climatology")]
    rest = txt
    for p in sorted(vals.values(), key=len, reverse=True):
        rest = rest.replace(p, " ")
    absent = []
    for p, _ in FIXED:
        if p not in rest and p not in ("2° cells", "2° cell"):
            absent.append(p)
        rest = rest.replace(p, " ")
    left = re.findall(r"\S*\d\S*", rest)
    cap_title = caption[caption.index("\\textbf{") + 8:W.braces(caption, caption.index("\\textbf{") + 7) - 1]
    ok = not missing and not left
    print(f"== {name}: panels a-{'f' if has_f else 'e'}, {W.nwords(caption)} words (title {W.nwords(cap_title)})")
    print("   (1) values of numbers.json printed in the legend:", f"all {len(vals)} found" if not missing else f"NOT FOUND: {missing}")
    print("   (2) numbers of the legend that are neither such a value nor a fixed descriptor:", "none" if not left else left)
    if absent:
        print("   note: fixed descriptors not present in this legend:", absent)
    if not ok and not enforce:
        print("   (reported only: this caption is outside the figure folder)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", default=str(HERE / "numbers.json"))
    ap.add_argument("--supp", default="")
    a = ap.parse_args()
    num = json.loads(Path(a.numbers).read_text())
    print(f"numbers: {a.numbers} (made {num['made']}, scorer directory {num['scorer_directory']})")
    ok = True
    for f in ("legend.tex", "legend_fig_rt_skill_main.tex"):
        if (HERE / f).exists():
            ok &= check(f, W.captions(W.strip_comments((HERE / f).read_text()))[0], num)
    if a.supp:
        s = W.strip_comments(Path(a.supp).read_text())
        blocks = [b for b in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", s, flags=re.S) if "\\label{fig:rtskill}" in b]
        assert len(blocks) == 1, f"{len(blocks)} figure blocks with \\label{{fig:rtskill}} in {a.supp}"
        same = (HERE / "legend_fig_rt_skill_main.tex").exists() and \
            W.captions(blocks[0])[0] == W.captions(W.strip_comments((HERE / "legend_fig_rt_skill_main.tex").read_text()))[0]
        check(f"{a.supp} (fig:rtskill)", W.captions(blocks[0])[0], num, enforce=False)
        print("   caption identical to legend_fig_rt_skill_main.tex:", same)
    print("ALL OK" if ok else "A LEGEND OF THE FOLDER FAILS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
