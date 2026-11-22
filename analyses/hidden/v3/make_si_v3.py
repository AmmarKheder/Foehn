#!/usr/bin/env python3
"""Supplementary Tables 'warned' and 'matched' of version 3, written from results/numbers_v3.json (tables_v3.py); no number
is typed by hand.
  results/si_table_warned_v3.tex     complete table (1 and 3 days), the rows of the frozen table (version 2 with the
                                     climatology) and the start-day estimate kept unchanged (persistence)
  results/si_table_matched_v3.tex    complete table: every reference at the number of people warned by Foehn, at the exact factor
  results/warned_v3.json, results/matched_v3.json   the same in the format of source_data/supp_tables_full/{warned,matched}.json
                                     (converted with the functions of tools/si_tables_portrait.py: plain, val3, num); every row
                                     whose numbers did not change is checked to equal the frozen file
  results/si_tables_portrait_v3.tex  the shortened portrait tables (1 day ahead) as typeset by tools/si_tables_portrait.py
    python make_si_v3.py"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms"))
sys.path.insert(0, str(W / "tools"))
from si_tables_portrait import LAB, num as pnum, plain, v, val3  # noqa: E402  (read-only import; nothing is written by it)

SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
RES = HERE / f"results{SFX}"
N = json.loads((RES / "numbers_v3.json").read_text())
FROZEN = W / "source_data" / "supp_tables_full"
MU = r"\,\textmu g\,m$^{-3}$"
PERSIST = "Start-day 1\\,km estimate kept unchanged (persistence)"


def safe_write(path, text):
    """temp file + fsync + read-back + rename (a plain write was once dropped by the file system)"""
    import os
    path = str(path); data = text.encode("utf-8"); tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data); fh.flush(); os.fsync(fh.fileno())
    assert open(tmp, "rb").read() == data, path
    os.replace(tmp, path)
    assert open(path, "rb").read() == data, path


def num(x, d=0):                                     # as make_si_v2.num
    return f"{x:,.{d}f}".replace(",", "{,}")


def ci(x, d=0):
    return f"{num(x[0], d)} [{num(x[1], d)}, {num(x[2], d)}]"


SYS = (("foehn_p50", "\\model{}, at least 2 of 4 members"), ("foehn_mean", "\\model{}, ensemble mean"), ("foehn_40km", "\\model{}, mean over 40\\,km cells"),
       ("cams", "CAMS forecast"), ("aurora", "Aurora"), ("cams_qm", "CAMS forecast, bias-corrected"),
       ("cams_bcsd", "CAMS forecast, bias-corrected, with the climatological 1\\,km pattern"), ("clim_1km", "Climatology of the 1\\,km estimates alone"),
       ("persist", PERSIST), ("perfect_40km", "Perfect 40\\,km forecast"))
REFS = (("cams", "CAMS forecast"), ("aurora", "Aurora"), ("cams_qm", "CAMS forecast, bias-corrected"),
        ("cams_bcsd", "CAMS forecast, bias-corrected, with the climatological 1\\,km pattern"), ("persist", PERSIST))


def warned_rows():
    P = N["systems"]; blocks = []
    for t in ("15", "37.5", "75"):
        for h in (1, 3):
            Wd = P[f"thr{t}"][f"lead{h}"]["World"]; b = Wd["foehn_p50"]
            head = (f"Above {t}{MU}, {h} day{'s' if h > 1 else ''} ahead: {num(b['exposed_M'][0])} million people exposed a day, "
                    f"{num(b['hidden_M'][0], 1)} million hidden, {num(b['hidden_margin_M'][0], 1)} million hidden with a 10\\% margin")
            rows = []
            for s, label in SYS:
                d = Wd[s]; dash = s == "perfect_40km"
                rows.append([label, num(d["warned_M"][0]), ci(d["right_M"]), ci(d["false_M"]), ci(d["reach_pct"], 1), ci(d["correct_pct"], 1),
                             "0" if dash else ci(d["hidden_warned_M"], 1), "0" if dash else ci(d["hidden_warned_pct"], 1),
                             "0" if dash else ci(d["hidden_margin_warned_pct"], 1), "0" if dash else ci(d["false_in_silent40_M"]),
                             "--" if dash else ci(d["false_per_hidden_warned"], 1)])
            blocks.append((head, rows))
    return blocks


def matched_rows():
    P = N["systems"]; blocks = []
    for t in ("15", "37.5", "75"):
        for h in (1, 3):
            Wd = P[f"thr{t}"][f"lead{h}"]["World"]; b = Wd["foehn_p50"]
            head = (f"Above {t}{MU}, {h} day{'s' if h > 1 else ''} ahead: \\model{{}} warns {num(b['warned_M'][0])} million people a day, "
                    f"reach {b['reach_pct'][0]:.1f}\\%, correct {b['correct_pct'][0]:.1f}\\%, {num(b['hidden_warned_M'][0], 1)} million hidden people warned")
            rows = []
            for s, label in REFS:
                q = Wd["equal_number_warned"][s]
                rows.append([label, f"{q['decision_threshold_factor']:.3f}", ci(q["reach_pct"], 1), ci(q["correct_pct"], 1), ci(q["hidden_warned_M"], 1),
                             ci(q["foehn_p50_minus_reach_pct"], 1), ci(q["foehn_p50_minus_correct_pct"], 1), ci(q["foehn_p50_minus_hidden_warned_M"], 1)])
            blocks.append((head, rows))
    return blocks


def tex(blocks, ncol, head, comment, name):
    L = [comment, head]
    for h_, rows in blocks:
        L.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{h_}}}}} \\\\")
        L += [" & ".join(r) + r" \\" for r in rows]
    safe_write(RES / name, "\n".join(L) + "\n\\hline\n\\end{tabular}\n")


def to_json(blocks, frozen_name, out_name, desc_add):
    fz = json.loads((FROZEN / f"{frozen_name}.json").read_text())
    rows = [[plain(h_), plain(r[0]), pnum(r[1])] + sum((val3(c) for c in r[2:]), []) for h_, rs in blocks for r in rs]
    assert all(len(r) == len(fz["cols"]) for r in rows)
    old = {(r[0], r[1]): r for r in fz["rows"]}
    same, changed, new = 0, [], []
    for r in rows:
        o = old.get((r[0], r[1]))
        if o is None:
            new.append(r[1]); continue
        if o == r:
            same += 1
        else:
            changed.append((r[0][:40], r[1], [(fz["cols"][i], o[i], r[i]) for i in range(2, len(r)) if o[i] != r[i]]))
    # 2026-11-21 (fix, backup make_si_v3.py.bak_20261121_fix): under a design (FOEHN_DESIGN set) the 'Version 3' sentence is
    # added only if the frozen desc does not already hold it (it does since the freeze of 2026-11-20); unset = unchanged.
    desc = fz["desc"] if (SFX and desc_add.strip() in fz["desc"]) else fz["desc"] + desc_add
    d = dict(label=fz["label"], part=fz["part"], what=fz["what"], desc=desc, cols=fz["cols"], rows=rows)
    safe_write(RES / f"{out_name}.json", json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    return same, changed, new


def portrait(wb, mb):
    """the shortened 1-day tables, with the cells chosen as in tools/si_tables_portrait.py (do_warned, do_matched)"""
    L = ["% Supplementary Table 'warned', shortened portrait version (1 day ahead), version 3 (rows as tools/si_tables_portrait.py do_warned)",
         "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
         " & Warned & Reach & Correct & \\multicolumn{2}{c}{Hidden people warned} & Wrongly warned, \\\\",
         " & (millions) & (\\%) & (\\%) & (millions) & (\\%) & 40\\,km mean below \\\\", "\\midrule"]
    k = 0
    for h_, rs in wb:
        if ", 1 day ahead" not in h_:
            continue
        m = re.fullmatch(r"(Above .*?), 1 day ahead: (.*? million hidden), [\d\.]+ million hidden with a 10\\% margin", h_)
        if k:
            L.append("\\addlinespace")
        k += 1
        L.append("\\multicolumn{7}{@{}l}{\\textit{%s: %s}} \\\\" % (m.group(1), m.group(2)))
        for r in rs:
            L.append(" & ".join([LAB.get(r[0], r[0]), r[1], v(r[4]), v(r[5]), v(r[6]), v(r[7]), v(r[9])]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "",
          "% Supplementary Table 'matched', shortened portrait version (1 day ahead), version 3 (rows as tools/si_tables_portrait.py do_matched)",
          "\\begin{tabular}{@{}lrrrrcc@{}}", "\\toprule",
          " & Factor on & Reach & Correct & Hidden warned & \\multicolumn{2}{c}{\\model{} minus reference (points)} \\\\",
          "Reference & the threshold & (\\%) & (\\%) & (millions) & reach & correct \\\\", "\\midrule"]
    k = 0
    for h_, rs in mb:
        if ", 1 day ahead" not in h_:
            continue
        m = re.fullmatch(r"(Above .*?), 1 day ahead: \\model\{\} warns (.*?) million people a day, reach (.*?)\\%, correct (.*?)\\%, (.*?) million hidden people warned", h_)
        if k:
            L.append("\\addlinespace")
        k += 1
        L.append("\\multicolumn{7}{@{}l}{\\textit{%s: \\model{} warns %s million people a day (reach %s\\%%, correct %s\\%%, %s million hidden)}} \\\\" % m.groups())
        for r in rs:
            L.append(" & ".join([LAB.get(r[0], r[0]), r[1], v(r[2]), v(r[3]), v(r[4]), r[5], r[6]]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    safe_write(RES / "si_tables_portrait_v3.tex", "\n".join(L) + "\n")


if __name__ == "__main__":
    wb, mb = warned_rows(), matched_rows()
    tex(wb, 11, "\\begin{tabular}{lrrrrrrrrrr}\n\\hline\n & Warned & Rightly warned & Wrongly warned & Reach (\\%) & Correct (\\%) & Hidden warned & "
        "Hidden warned (\\%) & With 10\\% margin (\\%) & Wrongly warned where the 40\\,km mean is below & per hidden person warned \\\\\n\\hline",
        "% Supplementary Table 'warned', version 3 (v3/make_si_v3.py from v3/results/numbers_v3.json): version 2 with the start-day 1 km estimate kept\n"
        "% unchanged (persistence). People a day in millions; 95 % intervals over target days (moving-block bootstrap, blocks of 7 days, 10,000 resamples).",
        "si_table_warned_v3.tex")
    tex(mb, 8, "\\begin{tabular}{lrrrrrrr}\n\\hline\n & Factor on the threshold & Reach (\\%) & Correct (\\%) & Hidden warned & \\model{} minus reference: reach (points) & "
        "correct (points) & hidden warned \\\\\n\\hline",
        "% Supplementary Table 'matched', version 3 (v3/make_si_v3.py from v3/results/numbers_v3.json, keys systems/thr*/lead*/World/equal_number_warned):\n"
        "% each reference at the exact factor for which it warns as many people a day as Foehn (re-evaluated from the fields at that factor; the factor\n"
        "% found again in each bootstrap resample). Hidden people warned in millions a day; 95 % intervals over target days, paired for the differences.",
        "si_table_matched_v3.tex")
    a = to_json(wb, "warned", "warned_v3", " Version 3: with the start-day 1 km estimate kept unchanged (persistence), a resident being warned when "
                "the estimate of the start day in their grid cell exceeds the threshold.")
    b = to_json(mb, "matched", "matched_v3", " Version 3: the factor is the exact one (each reference re-evaluated from the fields at that factor; "
                "found again in each bootstrap resample), and the start-day 1 km estimate kept unchanged (persistence) is added.")
    portrait(wb, mb)
    for name, (same, changed, new) in (("warned_v3.json", a), ("matched_v3.json", b)):
        print(f"{name}: {same} rows identical to the frozen file, {len(changed)} changed, {len(new)} new ({sorted(set(new))})")
        for c in changed:
            print("   changed:", c)
