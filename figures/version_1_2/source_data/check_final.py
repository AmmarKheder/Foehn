#!/usr/bin/env python3
"""Checks of source_data/Source_Data_final.xlsx (through its CSV copy csv_final.zip, same values) against the CURRENT
manuscript (work/*.tex) and the printed figure labels (text of work/figs/<stem>.pdf), 2026-11-22. Reads only; writes
check_final.json here (temporary file, fsync, rename).
  1. coverage: panel letters of every legend against the panels of MANIFEST_final.md;
  2. per sheet, up to ten values (spread over the rows), each looked for at its printed precision in the figure's own labels,
     its legend, or the table block (found = confirmed; not found = not printed, NOT a mismatch);
  3. targeted comparisons with a reference value (legend or label numbers, the figure scripts' number files): every
     difference is a mismatch."""
import csv
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
N = HERE.parents[3]
W = N / "work"
OUT = {"coverage": {}, "per_sheet": {}, "targeted": [], "mismatches": []}


def norm(t):
    t = t.replace("{,}", "").replace("\\textminus{}", "-").replace("$-$", "-").replace("−", "-").replace("–", " ").replace("\u2212", "-")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    return t


NUM = re.compile(r"-?\d+(?:\.\d+)?")


def tokens(t):
    return set(NUM.findall(norm(t)))


# ---------------------------------------------------------------- sheets
z = zipfile.ZipFile(HERE / "csv_final.zip")
SH = {}
for nm in z.namelist():
    rows = list(csv.reader(io.StringIO(z.read(nm).decode("utf-8"))))
    SH[Path(nm).stem] = [[""]] + rows                     # as in the workbook: row 0 description, row 1 header, then data
readme = SH.pop("README")
META = {r[0]: dict(figure=r[1], panel=r[2], stem=r[3][:-4]) for r in readme[2:]}

# ---------------------------------------------------------------- manuscript: captions by figure, table blocks
main = (W / "sections" / "main_text.tex").read_text(encoding="utf-8")
meth = (W / "sections" / "methods.tex").read_text(encoding="utf-8")
supp = (W / "supplementary.tex").read_text(encoding="utf-8")


def floats(src, env):
    out = []
    for m in re.finditer(r"\\begin\{" + env + r"\}(.*?)\\end\{" + env + r"\}", src, re.S):
        blk = m.group(1)
        g = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{figs/([^}]+?)(?:\.pdf)?\}", blk)
        lab = re.search(r"\\label\{([^}]+)\}", blk)
        out.append(dict(stem=g.group(1) if g else None, label=lab.group(1) if lab else None, text=blk))
    return out


FIGS = floats(main + meth, "figure") + floats(supp, "figure")
CAP = {f["stem"]: f["text"] for f in FIGS if f["stem"]}
TABS = {}
for t in floats(supp, "table"):
    if t["label"]:
        TABS[t["label"].split(":")[1]] = t["text"]
    elif TABS:                                            # a continued float: appended to the previous table
        k = list(TABS)[-1]; TABS[k] += t["text"]
TABLAB = [m.group(1) for m in re.finditer(r"\\label\{tab:([A-Za-z0-9_]+)\}", supp)]
PDFTXT = {}


def pdftext(stem):
    if stem not in PDFTXT:
        p = W / "figs" / f"{stem}.pdf"
        PDFTXT[stem] = "".join(pg.get_text() for pg in fitz.open(p)) if p.exists() else ""
    return PDFTXT[stem]


# ---------------------------------------------------------------- 1. coverage
man = (N / "source_data" / "MANIFEST_final.md").read_text(encoding="utf-8")
panels = {}
for line in man.split("## Panel by panel")[1].split("## Sheets")[0].splitlines():
    if line.startswith("| ") and not line.startswith("| Figure") and not line.startswith("|---"):
        c = [x.strip() for x in line.strip("|").split("|")]
        panels.setdefault(c[0], []).append(c[1:])
order = re.findall(r"\| (Fig\. \d+|Supplementary Fig\. \d+) \| `([^`]+)\.pdf` \|", man)
for fg, stem in order:
    cap = CAP.get(stem, "")
    let = sorted(set(re.findall(r"\\textbf\{([a-z])\}", cap.split("}", 1)[1] if "}" in cap else cap)))
    have = panels.get(fg, [])
    OUT["coverage"][fg] = dict(stem=stem, legend_letters="".join(let), manifest=[" / ".join(h[:1] + h[2:3]) for h in have])

# ---------------------------------------------------------------- 2. per sheet, up to ten values looked for
def fmt_variants(v):
    out = set()
    for d in (0, 1, 2, 3):
        s = f"{v:.{d}f}"
        if s.lstrip("-") in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0.0", "0.00", "0.000", "1.0", "1.00"):
            continue
        out.add(s); out.add(s.lstrip("-"))
    return out


for name, rows in SH.items():
    meta = META[name]; stem = meta["stem"]
    if stem == "supplementary":
        lab = TABLAB[int(re.match(r"STable(\d+)", name).group(1)) - 1] if name.startswith("STable") else None
        scope = {"table block": tokens(TABS.get(lab, "")) if lab else set()}
    else:
        scope = {"figure labels": tokens(pdftext(stem)), "legend": tokens(CAP.get(stem, ""))}
    body = rows[2:] if len(rows) > 2 else []
    if name.startswith("SNote"):                          # the Note on opposite pairs and the noise-aware mean: the two Outcomes
        P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution/protocol_amendments"))
        scope = {"Outcomes A19/A20": tokens((P / "A19_opposite_pairs_test_year.md").read_text().split("## Outcome")[-1]
                                            + (P / "A20_noise_aware_mean_test_year.md").read_text().split("## Outcome")[-1])}
    cand = []
    for r in body:
        for c in r:
            try:
                v = float(c)
            except ValueError:
                continue
            if v != int(v) or abs(v) >= 10:
                cand.append(v)
    picked = []
    if cand:
        step = max(1, len(cand) // 10)
        picked = cand[::step][:10]
    res = []
    for v in picked:
        hit = [k for k, tok in scope.items() if fmt_variants(v) & tok]
        res.append([v, hit[0] if hit else None])
    OUT["per_sheet"][name] = dict(rows=len(body), checked=len(res), found=sum(1 for _, h in res if h), values=res)


# ---------------------------------------------------------------- 3. targeted comparisons
def col(name, c):
    rows = SH[name]; i = rows[1].index(c); return [r[i] for r in rows[2:]]


def where(name, **kw):
    rows = SH[name]; hd = rows[1]
    return [dict(zip(hd, r)) for r in rows[2:] if all(dict(zip(hd, r))[k] == v for k, v in kw.items())]


def T(what, got, ref, ok=None, src=""):
    ok = (got == ref) if ok is None else ok
    OUT["targeted"].append(dict(what=what, sheet_value=got, reference=ref, ok=bool(ok), source=src))
    if not ok:
        OUT["mismatches"].append(dict(what=what, sheet_value=got, reference=ref, source=src))


def r1(x):
    return f"{float(x):.1f}"


j3 = json.loads((N / "rebuild" / "figs_show" / "final3" / "fig3_numbers.json").read_text())
B3 = {"cams": "CAMS forecast (40 km)", "ours": "Foehn (1 km)", "ghap": "GHAP (1 km estimates, not a forecast)"}
for nm, d in j3["boxes_day"].items():
    if not isinstance(d, dict) or "cams" not in d:
        continue
    for k, lab in B3.items():
        r = where("Fig3b", basin=nm, system=lab)[0]
        T(f"Fig3b {nm} {k} RMSE on 9 Feb (printed)", r["as printed"], d[k]["printed"], src="final3/fig3_numbers.json boxes_day")
    T(f"Fig3b {nm} stations scored", where("Fig3b", basin=nm, system=B3['cams'])[0]["stations scored"], str(d["stations_scored"]), src="fig3_numbers.json")
    T(f"Fig3b_stations {nm} circles", str(sum(1 for r in SH["Fig3b_stations"][2:] if r[1] == nm)), str(d["dots_drawn"]), src="fig3_numbers.json")
cap3 = norm(CAP["fig_overview"])
for nm, d in j3["boxes_yearly_for_legend"].items():
    if not isinstance(d, dict):
        continue
    for k, lab in B3.items():
        v = [r for r in where("Fig3b_legend", basin=nm) if r["quantity"].endswith(f"{lab} (µg m-3)")][0]["value"]
        T(f"Fig3b_legend {nm} yearly {k}", r1(v), f"{d[k]:.1f}", src="fig3_numbers.json boxes_yearly_for_legend")
for b, (lo, hi) in zip(("Delhi", "Po", "LA", "Seoul"), ((115, 63), (72, 106), (159, 19), (81, 97))):
    nm = [k for k in j3["boxes_day"] if k.startswith({"Delhi": "Delhi", "Po": "Po", "LA": "Los", "Seoul": "Seoul"}[b])][0]
    rr = where("Fig3b_legend", basin=nm)
    T(f"Fig3b_legend {b} days below CAMS (legend)", [r["value"] for r in rr if "below" in r["quantity"]][0], str(lo), src="Fig. 3 legend")
    T(f"Fig3b_legend {b} days above CAMS (legend)", [r["value"] for r in rr if "above" in r["quantity"]][0], str(hi), src="Fig. 3 legend")
for pn, sh in (("Stations of the six networks", "Fig3c"), ("China", "Fig3d")):
    for nm, d in j3["band"][pn].items():
        if not isinstance(d, dict) or "rmse" not in d:
            continue
        got = [r1(r["RMSE (µg m-3)"]) for r in where(sh, system=nm)]
        T(f"{sh} {nm} RMSE leads 1-4", got, [f"{v:.1f}" for v in d["rmse"]], src="fig3_numbers.json band")
for net, d in j3["band_e"]["values"].items():
    r = [x for x in SH["Fig3e"][2:] if x[0] == d["network"]][0]
    T(f"Fig3e {net} printed bar value", r[5].replace("-", "−"), d["printed"], src="fig3_numbers.json band_e")
    T(f"Fig3e {net} interval", [f"{float(r[2]):.0f}", f"{float(r[3]):.0f}"], d["printed_interval"], src="fig3_numbers.json band_e")
e_leg = {(r[0], r[1]): r[2:] for r in SH["Fig3e_legend"][2:]}
sh_ = [float(e_leg[("share of China (CNEMC) in the station-days (%)", str(h))][0]) for h in range(1, 5)]
T("Fig3e_legend China share of station-days, range (legend 22.5-22.6)", [f"{min(sh_):.1f}", f"{max(sh_):.1f}"], ["22.5", "22.6"], src="Fig. 3 legend")
sr = [float(e_leg[("share of China (CNEMC) in the reduction of the summed squared error against the CAMS forecast (%)", str(h))][0]) for h in range(1, 5)]
T("Fig3e_legend China share of the reduction, range (legend 85-89)", [f"{min(sr):.0f}", f"{max(sr):.0f}"], ["85", "89"], src="Fig. 3 legend")
oc = e_leg[("RMSE reduction (%) against the CAMS forecast, the five networks pooled outside China", "1")] if False else \
    e_leg[("RMSE reduction (%) against the CAMS forecast, the five networks outside China, pooled", "1")]
T("Fig3e_legend outside China 1 day (legend 4.2 (0.4-9.7))", [r1(x) for x in oc], ["4.2", "0.4", "9.7"], src="Fig. 3 legend")
oc2 = [e_leg[("RMSE reduction (%) against the CAMS forecast, the five networks outside China, pooled", str(h))] for h in (2, 3, 4)]
T("Fig3e_legend outside China, interval includes zero from 2 days (legend)", [float(x[1]) <= 0 <= float(x[2]) for x in oc2], [True, True, True], src="Fig. 3 legend")

j5 = json.loads((N / "rebuild" / "figs_show" / "final2" / "fig5_numbers.json").read_text())
for h, L in ((1, "h"), (3, "p")):
    ref = j5["rows"][f"lead{h}"]; lam = ref["wavelength_band_near_10km_km"]
    def at(field):
        rr = [r for r in where(f"Fig5{L}", field=field) if abs(float(r["wavelength (km)"]) - lam) < 0.05]
        return f"{float(rr[0]['variance per decade ((µg m-3)^2)']):.3f}" if rr else None
    for fld, k in (("GHAP (1 km estimates)", "ghap"), ("Foehn, median of the 16 members", "members_median"), ("Foehn, lowest of the 16 members (band)", "members_min"),
                   ("Foehn, highest of the 16 members (band)", "members_max"), ("Foehn, ensemble mean", "ensemble_mean")):
        g = at(fld); rv = f"{ref['variance_per_decade_at_that_band'][k]:.3f}"
        T(f"Fig5{L} {fld} at {lam} km", g, rv, ok=g is not None and abs(float(g) - float(rv)) <= 0.0015 * max(1, float(rv)), src="final2/fig5_numbers.json rows")
bx = {(r[0], r[2]): r[3] for r in SH["Fig5_box"][2:]}
for h, key in ((1, "1_day"), (3, "3_days"), (5, "5_days"), (7, "7_days")):
    ref = j5["box_numbers"]["leads"][key]
    T(f"Fig5_box lead {h} ranks of members 1-3", [bx[(str(h), f"rank of member {k} among the 16 by that mean (1 = highest)")] for k in (1, 2, 3)],
      [str(x) for x in ref["members_1_2_3"]["rank_of_16_1=highest"]], src="fig5_numbers.json box_numbers")
    T(f"Fig5_box lead {h} RMSE of the ensemble mean vs GHAP", f"{float(bx[(str(h), 'RMSE against GHAP over the land of the panel, Foehn' + chr(39) + 's ensemble mean')]):.2f}",
      f"{ref['rmse_vs_ghap']['ensemble_mean']:.2f}", src="fig5_numbers.json box_numbers")
    T(f"Fig5_box lead {h} P(>75) where GHAP > 75", f"{float(bx[(str(h), 'mean probability of exceeding 75 µg m-3 (share of the 16 members) where GHAP exceeds 75')]):.3f}",
      f"{ref['prob_gt75_mean']['where_ghap_gt75']:.3f}", src="fig5_numbers.json box_numbers")
T("Fig5_box ranks 1 day (legend 'ranks 2, 1 and 7')", [bx[("1", f"rank of member {k} among the 16 by that mean (1 = highest)")] for k in (1, 2, 3)], ["2", "1", "7"], src="Fig. 5 legend")
T("Fig5_box ranks 3 days (legend 'rank 6, 16 and 1')", [bx[("3", f"rank of member {k} among the 16 by that mean (1 = highest)")] for k in (1, 2, 3)], ["6", "16", "1"], src="Fig. 5 legend")
lg = {(r[0], r[1]): r[2] for r in SH["Fig5_legend"][2:]}
T("Fig5_legend days below both, 1/3/5 days (legend 14, 12, 10)", [lg[(f"lead {h} days: days of the 22 on which Foehn's RMSE is below both references", str(h))] for h in (1, 3, 5)], ["14", "12", "10"], src="Fig. 5 legend")
T("Fig5_legend days highest of three, 1/3/5 days (legend 7, 4, 6)", [lg[(f"lead {h} days: days of the 22 on which Foehn's RMSE is the highest of the three", str(h))] for h in (1, 3, 5)], ["7", "4", "6"], src="Fig. 5 legend")
st5 = [r1(lg[(f"RMSE at the 137 official stations of the plain on 24 January 2022 (µg m-3), {nm}", str(h))]) for h in (1, 3, 5) for nm in ("Foehn's ensemble mean", "CAMS forecast", "Aurora")]
T("Fig5_legend RMSE at the 137 stations (legend 24.8, 35.7, 26.4 / 28.5, 33.2, 30.1 / 32.7, -, 29.7)", [st5[i] for i in (0, 1, 2, 3, 4, 5, 6, 8)],
  ["24.8", "35.7", "26.4", "28.5", "33.2", "30.1", "32.7", "29.7"], src="Fig. 5 legend")
d5 = [r for r in SH["Fig5_days"][2:] if r[0] == "2022-01-24"]
six = 0
for dd in sorted({r[0] for r in SH["Fig5_days"][2:]}):
    rr = {r[1]: r for r in SH["Fig5_days"][2:] if r[0] == dd}
    if all(float(rr[h][3]) < min(float(rr[h][4]), float(rr[h][5])) for h in ("1", "3", "5")):
        six += 1
T("Fig5_days days below both references at 1, 3 and 5 days (legend 'the 6')", str(six), "6", src="Fig. 5 legend")
pol = max(((r[0], float(r[2])) for r in SH["Fig5_days"][2:]), key=lambda x: x[1])
T("Fig5_days most polluted day of the 22 (Methods: 24 January, 89.6)", [pol[0], f"{pol[1]:.1f}"], ["2022-01-24", "89.6"], src="Methods")
u = SH["Fig5u"][2:]
mx = max(u, key=lambda r: float(r[6]))
T("Fig5u top of the band (fig5_numbers: 202.2 on 23 January)", [mx[1], r1(mx[6])], ["23", "202.2"], src="fig5_numbers.json series_u")
pk = max(u, key=lambda r: float(r[2]))
T("Fig5u day of the highest measured value (legend: 24 January)", pk[1], "24", src="Fig. 5 legend")
# Supplementary Fig. 8 (21 January): yearly numbers printed in the maps = its legend
for nm, f_, c_, g_ in (("Delhi and the Ganges plain", "39.7", "42.1", "40.0"), ("Po Valley and the Alps", "9.5", "9.0", "12.6"),
                       ("Los Angeles basin", "4.7", "10.4", "2.8"), ("Seoul and the Han basin", "10.7", "10.0", "9.7")):
    got = [r1(where("SFig8b", basin=nm, system=s_)[0]["RMSE (µg m-3)"]) for s_ in ("Foehn (1 km)", "CAMS forecast (40 km)", "GHAP (1 km estimates, not a forecast)")]
    T(f"SFig8b {nm} (legend Foehn/CAMS/GHAP)", got, [f_, c_, g_], src="Suppl. Fig. 8 legend")
# Supplementary Fig. 11 (25 January): the numbers of its legend
l11 = {(r[0], r[1]): r[2] for r in SH["SFig11_legend"][2:]}
for lab, ref in (("RMSE against GHAP over the land of the panel, Foehn's ensemble mean", ["32.2", "36.0", "51.5"]),
                 ("RMSE against GHAP over the land of the panel, CAMS forecast issued at the same time", ["29.4", "24.2", "32.8"]),
                 ("RMSE against GHAP over the land of the panel, Aurora issued at the same time", ["25.1", "26.2", "31.1"])):
    T(f"SFig11_legend {lab.split(', ')[-1]} at 1/3/5 days (legend)", [r1(l11[(str(h), lab)]) for h in (1, 3, 5)], ref, src="Suppl. Fig. 11 legend")
pw = [float(l11[(str(h), "mean probability of exceeding 75 µg m-3 (share of the 16 members) where GHAP exceeds 75")]) for h in range(1, 8)]
pe = [float(l11[(str(h), "mean probability of exceeding 75 µg m-3 elsewhere on land")]) for h in range(1, 8)]
em = [float(l11[(str(h), "mean of Foehn's ensemble mean where GHAP exceeds 75")]) for h in range(1, 8)]
T("SFig11_legend P(>75) where GHAP > 75, 1-3 days (legend 0.5-0.6)", [f"{min(pw[:3]):.1f}", f"{max(pw[:3]):.1f}"], ["0.5", "0.6"], src="Suppl. Fig. 11 legend")
T("SFig11_legend P(>75) elsewhere, 1-3 days (legend below 0.1)", max(pe[:3]) < 0.1, True, src="Suppl. Fig. 11 legend")
T("SFig11_legend P(>75) where GHAP > 75, 4-7 days (legend 0.2-0.4)", [f"{min(pw[3:]):.1f}", f"{max(pw[3:]):.1f}"], ["0.2", "0.4"], src="Suppl. Fig. 11 legend")
T("SFig11_legend forecast where GHAP > 75 (legend 62-83 against 114)", [f"{min(em):.0f}", f"{max(em):.0f}", f"{float(l11[('0', 'mean of GHAP where it exceeds 75')]):.0f}"],
  ["62", "83", "114"], src="Suppl. Fig. 11 legend")
T("SFig11_legend member 1 highest at 1 day and lowest at 5 days (legend)",
  [l11[(str(h), "ranks of members 1, 2 and 3 among the 16 by their mean over the plain (1 = highest)")].split(",")[0] for h in (1, 5)], ["1", "16"], src="Suppl. Fig. 11 legend")
# Supplementary Table 'away': the printed cells
NA = f"STable{TABLAB.index('away') + 1}"
S7 = {r[0]: r for r in SH[NA][2:] if r[1] == "1"}
hd = SH[NA][1]
blk = norm(TABS["away"])
for line in blk.split("\\midrule")[1].split("\\bottomrule")[0].strip().split("\\\\"):
    c = [x.strip() for x in line.split("&")]
    if len(c) < 10:
        continue
    r = S7.get(c[0])
    if r is None:
        continue
    i = hd.index
    got = [str(int(r[i("stations with station-days at this lead in the ring sample")])), f"{float(r[i('estimate within 0.5 µg m-3 of the measurement, at the station pixel (% of station-days)')]):.0f}",
           f"{float(r[i('the same, one pixel 10 km away (%)')]):.0f}", r1(r[i("RMSE of Foehn at the station pixel (µg m-3)")]), r1(r[i("RMSE of Foehn on the ring 10 km away (µg m-3)")]),
           r1(r[i("RMSE of the CAMS forecast (µg m-3)")]), r1(r[i("RMSE of Aurora (µg m-3)")]),
           f"{r1(r[i('gain over the CAMS forecast at the pixel (%)')])} [{r1(r[i('gain over the CAMS forecast at the pixel (%), 95% interval low')])}, {r1(r[i('gain over the CAMS forecast at the pixel (%), 95% interval high')])}]",
           f"{r1(r[i('gain over the CAMS forecast 10 km away (%)')])} [{r1(r[i('gain over the CAMS forecast 10 km away (%), 95% interval low')])}, {r1(r[i('gain over the CAMS forecast 10 km away (%), 95% interval high')])}]"]
    ref = [c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]]
    for k, (a, b) in enumerate(zip(got, ref)):
        T(f"{NA} {c[0]} column {k + 2}", a, b, src=f"Suppl. Table {NA[6:]} as printed")
# Methods: spread/skill on 2021 after the CRPS fine-tuning, 0.90 at 1 day and 0.79 after a week (Supplementary Fig. 6h)
h6 = {(r[0], r[1]): r[2] for r in SH["SFig6h"][2:]}
after = [k for k in h6 if k[0].endswith("after the CRPS training")]
T("SFig6h spread/skill 2021 after the CRPS training, 1 and 7 days (Methods 0.90, 0.79)",
  [f"{float(h6[(after[0][0], '1')]):.2f}", f"{float(h6[(after[0][0], '7')]):.2f}"], ["0.90", "0.79"], src="Methods, Calibration fine-tuning")
before = [k for k in h6 if k[0].endswith("before the CRPS training")]
T("SFig6h spread/skill 2021 before the CRPS training, 1 and 7 days (Methods 0.51, 0.43)",
  [f"{float(h6[(before[0][0], '1')]):.2f}", f"{float(h6[(before[0][0], '7')]):.2f}"], ["0.51", "0.43"], src="Methods, Calibration fine-tuning")
# the Note sheet against the two Outcomes
sn = {(r[0], r[1], r[2]): r[3:6] for r in SH[next(s for s in SH if s.startswith("SNote"))][2:]}
for (q, e, h), ref in [(("RMSE of the ensemble mean", "four independent members, as run (the paper's evaluation)", "1"), "13.03"),
                       (("RMSE of the ensemble mean", "four independent members, as run (the paper's evaluation)", "3"), "14.24"),
                       (("RMSE of the ensemble mean", "two opposite pairs (four members), as run", "1"), "12.67"),
                       (("RMSE of the ensemble mean", "four opposite pairs (eight members), as run", "3"), "13.71"),
                       (("RMSE of the ensemble mean", "eight independent members, the paper's form", "1"), "12.81"),
                       (("RMSE of the ensemble mean", "infinitely many independent members, the paper's form", "3"), "13.63"),
                       (("RMSE of the ensemble mean", "noise-aware mean of the four members as run", "1"), "12.74"),
                       (("RMSE of the ensemble mean", "noise-aware mean of the four members as run", "3"), "13.80")]:
    T(f"SNote {e} lead {h}", f"{float(sn[(q, e, h)][0]):.2f}", ref, src="A19/A20 Outcome")
for (q, e, h), ref in [(("MSE(eight independent members, the paper's form) - MSE(four opposite pairs)", "D (pairs)", "1"), ["4.59", "3.24", "5.82"]),
                       (("MSE(plain mean of four) - MSE(noise-aware mean of four)", "D (noise-aware mean)", "3"), ["12.42", "8.11", "15.24"]),
                       (("MSE(eight independent members, the paper's form) - MSE(noise-aware mean of four)", "noise-aware mean", "1"), ["1.87", "1.12", "3.09"]),
                       (("equivalent number of plain members (truth-free)", "noise-aware mean of four", "1"), ["15.2", "13.8", "16.4"]),
                       (("RMSE reduction against the CAMS forecast", "four opposite pairs", "1"), ["18.1", "12.9", "25.2"]),
                       (("RMSE reduction against the CAMS forecast", "noise-aware mean of the four members as run", "3"), ["14.1", "9.3", "21.5"])]:
    nd = len(ref[0].split(".")[1])
    T(f"SNote {q} [{e}] lead {h}", [f"{float(x):.{nd}f}" for x in sn[(q, e, h)]], ref, src="A19/A20 Outcome")
for th, refv in (("37.5", ["0.0478", "0.0487", "0.0525"]), ("75", ["0.0167", "0.0169", "0.0181"])):
    q = f"Brier score above {th} µg m-3, residents of the populated 1 km grid cells"
    got = [f"{float(sn[(q, e, '1')][0]):.4f}" for e in ("closed form from the four members (the law)", "counted on sixteen members", "the same form without the law (control)")]
    T(f"SNote Brier above {th}, 1 day: closed form / sixteen / control", got, refv, src="A20 Outcome")

# ---------------------------------------------------------------- 4. tables, the other way round: every number printed in a
# tabular of a Supplementary Table that has sheets, looked for among the values of its sheets at the printed precision
OUT["tables_printed_in_sheets"] = {}
for k, lab in enumerate(TABLAB, 1):
    names = [s for s in SH if re.fullmatch(rf"STable{k}(?:[a-z]|_four_members)?", s)]
    if not names:
        continue
    vals = []
    for s in names:
        for r in SH[s][2:]:
            for c in r:
                for x in re.findall(r"-?\d+(?:\.\d+)?", c.replace(",", "")) if not re.fullmatch(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", c) else [c]:
                    try:
                        vals.append(float(x))
                    except ValueError:
                        pass
    blk = norm(TABS[lab])
    tab = " ".join(m.group(1) for m in re.finditer(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", blk, re.S))
    tab = re.sub(r"\\(cmidrule|multicolumn|setlength|arraybackslash|centering|p)\{[^}]*\}(\{[^}]*\})?", " ", tab)
    tab = re.sub(r"\{@\{\}[^}]*\}|p\{[\d.]+cm\}|\\textbf\{[a-z]\}|\\,|\\%|[0-9.]+pt|\\fitwidth", " ", tab)
    toks = NUM.findall(tab)
    miss = []
    for t_ in toks:
        d = len(t_.split(".")[1]) if "." in t_ else 0
        tv = float(t_)
        if not any(abs(round(v, d) - tv) < 1e-9 or abs(round(abs(v), d) - abs(tv)) < 1e-9 for v in vals):
            miss.append(t_)
    OUT["tables_printed_in_sheets"][f"Supplementary Table {k} ({lab})"] = dict(sheets=names, printed_numbers=len(toks), not_found=miss)
    print(f"Table {k} {lab}: {len(toks)} printed numbers, {len(miss)} not in its sheets: {miss[:40]}")

txt = json.dumps(OUT, indent=1, ensure_ascii=False) + "\n"
tmp = HERE / ".check_final.json.tmp"
with open(tmp, "w") as f:
    f.write(txt); f.flush(); os.fsync(f.fileno())
os.replace(tmp, HERE / "check_final.json")
assert (HERE / "check_final.json").read_text() == txt
print("targeted:", len(OUT["targeted"]), "ok:", sum(t["ok"] for t in OUT["targeted"]), "mismatches:", len(OUT["mismatches"]))
for m in OUT["mismatches"]:
    print("MISMATCH", m)
print("sheets:", len(OUT["per_sheet"]), "values checked:", sum(v["checked"] for v in OUT["per_sheet"].values()),
      "found:", sum(v["found"] for v in OUT["per_sheet"].values()))
for fg, c in OUT["coverage"].items():
    print(fg, c["stem"], "legend:", c["legend_letters"], "| manifest:", "; ".join(c["manifest"])[:300])
