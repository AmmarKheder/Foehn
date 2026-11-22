#!/usr/bin/env python3
"""Old (stored noise fields, the current manuscript) against new (design 'ar', A16: noise drawn anew at every start day):
the headline quantities of the paper, read from the output files of the two designs, never typed.

    $FOEHN_ENVS/world/bin/python -B old_vs_new.py     -> old_vs_new.json, old_vs_new.md (this folder)

Read only. Every value carries its file and key. Relative change = (new - old) / |old| x 100. Lists:
  (i)   reference columns (CAMS, Aurora, persistence, climatology, bias-corrected CAMS, 1 km estimates, other models):
        stored against 'ar' in the outputs of both designs (must be identical);
  (ii)  A16 Addendum 3 point 2(e): under 'ar', a conclusion whose sign of the point value, or whose exclusion of zero by
        its interval, differs between two of the forms 4 (as run), 8, infinitely many;
  (iii) the same rule between the stored design and 'ar', form by form."""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

HERE = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/print"))
N = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms"))
P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
FIG = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))
FILES = {}


def sha(p):
    p = Path(p)
    if str(p) not in FILES:
        FILES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return FILES[str(p)]


_J = {}


def J(p):
    p = Path(p)
    if str(p) not in _J:
        sha(p)
        _J[str(p)] = json.loads(p.read_text())
    return _J[str(p)]


def dig(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def tri(x):
    """(point, low, high) of the usual record forms; None if absent"""
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return (float(x), None, None)
    if isinstance(x, (list, tuple)) and len(x) == 3 and all(isinstance(v, (int, float)) or v is None for v in x):
        return tuple(None if v is None else float(v) for v in x)
    if isinstance(x, dict):
        for pk, ik in (("v", "ci"), ("v", "ci_day"), ("v", "ci_boot"), ("value", "interval95"), ("point", "interval95"),
                       ("est", "two_way"), ("diff", None)):
            if pk in x and isinstance(x[pk], (int, float)):
                iv = x.get(ik) if ik else None
                if ik is None and "ci_low" in x:
                    iv = [x["ci_low"], x["ci_high"]]
                if isinstance(iv, (list, tuple)) and len(iv) == 2 and all(isinstance(v, (int, float)) for v in iv):
                    return (float(x[pk]), float(iv[0]), float(iv[1]))
                return (float(x[pk]), None, None)
    return None


ROWS = []        # one entry per (category, quantity, form)


def add(cat, quantity, form, fo, fn, key, old, new, note=""):
    o, n = tri(old), tri(new)
    rel = None
    if o and n and o[0] not in (None, 0.0) and n[0] is not None:
        rel = (n[0] - o[0]) / abs(o[0]) * 100
    ROWS.append(dict(cat=cat, quantity=quantity, form=form, old=o, new=n, rel_change_pct=rel,
                     src_old=str(fo), src_new=str(fn), key=key, note=note))


def pair(cat, quantity, form, fo, fn, path, note="", conv=None):
    try:
        a, b = dig(J(fo), path), dig(J(fn), path)
    except (KeyError, IndexError, TypeError) as e:
        add(cat, quantity, form, fo, fn, "/".join(map(str, path)), None, None, f"key missing: {e!r}")
        return
    if conv:
        a, b = conv(a), conv(b)
    add(cat, quantity, form, fo, fn, "/".join(map(str, path)), a, b, note)


# ------------------------------------------------------------------------------------------------------ (A) stations
D5O, D5N = HERE / "d5" / "out_stored" / "d5.json", HERE / "d5" / "out_ar" / "d5.json"
FORMS5 = [("4 (as run)", "4 members (as run)"), ("8", "8 members"), ("16", "16 members"), ("inf", "infinitely many members")]
d5o, d5n = J(D5O), J(D5N)
for ro, rn in zip(d5o["rows"], d5n["rows"]):
    assert ro[:2] == rn[:2], (ro[:2], rn[:2])
    cat = "A" if ro[0].startswith("Stations") else "C"
    for f, col in FORMS5:
        i = d5o["cols"].index(col)
        add(cat, ro[1] + (" [D5 table]" if cat == "C" else ""), f, D5O, D5N, f"rows['{ro[1]}']/cols['{col}' +low/high]",
            ro[i:i + 3], rn[i:i + 3])

RCO, RCN = P / "a14/classU/results/rel_classU_179.json", P / "a14/classU/results_ar/rel_classU_179.json"
AO = "All official networks"
for L in ("1", "3"):
    for q, lab in (("ss", "spread/skill against the measurements at the stations"),
                   ("crps_vs_clim", "CRPS change against the climatological ensemble at the stations (%, negative = better; forms as the CRPS row)"),
                   ("crps", "CRPS at the stations (ug m-3; 4: fair CRPS of the four as run; 8: expected CRPS of eight members; inf: fair CRPS of the sixteen)"), ("rmse", "RMSE of the mean, reliability station-days (ug m-3)")):
        pair("A", f"{lab}, {L} day(s)", "4 (as run)", RCO, RCN, ["stations_reliability", "by_lead", L, AO, f"{q}_0", "as_run"])
        for f in ("8", "inf"):
            pair("A", f"{lab}, {L} day(s)", f, RCO, RCN, ["stations_reliability_16", "by_lead", L, AO, f"{q}|{f}|0"])

# ------------------------------------------------------------------------------------------- (B) the 1 km estimates
LO, LN = P / "a14/classU/results/land25_classU.json", P / "a14/classU/results_ar/land25_classU.json"
G = "Global (84N-60S)"
for L in ("1", "4", "7"):
    for f, k in (("4 (as run)", "as_run"), ("8", "M8"), ("inf", "inf")):
        pair("B", f"correlation with the estimates on 0.25 deg land cells, {L} day(s)", f, LO, LN, ["leads", G, L, "r", k])
for L in ("1", "7"):
    for f, k in (("4 (as run)", "as_run"), ("8", "M8"), ("inf", "inf")):
        pair("B", f"RMSE change against the climatology on 0.25 deg cells, {L} day(s) (%, negative = better)", f, LO, LN,
             ["leads", G, L, "references", "climatology", "gain_pct", k])
ESO, ESN = N / "analyses/reliability/results/estimates_scores.json", N / "analyses/reliability/results_ar/estimates_scores.json"
for L in map(str, range(1, 8)):
    for q, lab in (("crps", "fair CRPS against the estimates, land, nine regions pooled (ug m-3)"),
                   ("rmse", "RMSE of the mean against the estimates, land, nine regions pooled (ug m-3)")):
        pair("B", f"{lab}, {L} day(s)", "4 (as run)", ESO, ESN, ["land", "2022 final", "leads", L, "Nine regions pooled", q])

# --------------------------------------------------------------------------------------------------- (C) warnings
CEO, CEN = P / "a14/classE/results/classE_all2022.json", P / "a14/classE/results_ar/classE_all2022.json"
FORMSE = [("8", "M8_liberal"), ("16", "M16_liberal_measured"), ("4", "M4_stored"), ("inf", "infinite")]
QE = [("issued:warned_M", "people warned (millions a day)"), ("issued:reach_pct", "reach (%)"),
      ("issued:correct_pct", "share of correct warnings (%)"), ("issued:hidden_warned_M", "hidden people warned (millions a day)"),
      ("issued:hidden_warned_pct", "hidden people warned (%)")]
for r in ("cams", "cams_qm", "cams_bcsd", "aurora", "persist"):
    QE.append((f"eqn:{r}:reach_pct", f"reach minus that of {r} at Foehn's number warned (points)"))
for q in ("hidden_warned_pct", "hidden_margin_warned_pct", "hidden_warned_M"):
    QE.append((f"diff:cams:{q}", f"{q} minus that of the CAMS forecast, as issued"))
for t in ("37.5", "75"):
    for L in ("1", "3"):
        for qk, lab in QE:
            for f, mk in FORMSE:
                pair("C", f"> {t}, {L} d: {lab}", f, CEO, CEN, ["results", "world", "conf149", f"thr{t}", f"lead{L}", "quantities", qk, mk])

# --------------------------------------------------------------------------------------------- (D) reliability
SSO, SSN = N / "analyses/reliability/results/station_scores_final.json", N / "analyses/reliability/results_ar/station_scores_final.json"
for L in ("1", "3"):
    pair("D", f"stations: share inside the member range (4 members, expected 0.6), {L} d", "4 (as run)", SSO, SSN, ["leads", L, AO, "cover_0"])
    for j, t in enumerate(("15", "37.5", "75")):
        pair("D", f"stations: Brier skill against the local climatology > {t}, {L} d", "4 (as run)", SSO, SSN, ["leads", L, AO, "bss_clim_0"],
             conv=lambda x, j=j: (x["v"][j], x["ci_day"][0][j], x["ci_day"][1][j]))
        for f in ("8", "inf"):
            pair("D", f"stations: Brier skill against the local climatology > {t}, {L} d", f, RCO, RCN,
                 ["stations_reliability_16", "by_lead", L, AO, f"bss_clim|{f}|0"],
                 conv=lambda x, j=j: (x["v"][j], x["ci"][0][j], x["ci"][1][j]))
    PP = ["pop", "leads", L, "All populated pixels", "pixels"]
    pair("D", f"populated pixels: spread/skill against the estimates, {L} d", "4 (as run)", ESO, ESN, PP + ["ss"])
    pair("D", f"populated pixels: share inside the member range (expected 0.6), {L} d", "4 (as run)", ESO, ESN, PP + ["cover"])
    for f, k in (("8", "ss_8"), ("inf", "ss_inf")):
        pair("D", f"populated pixels: spread/skill against the estimates, {L} d", f, RCO, RCN, ["populated_pixels", "by_lead", L, "pixels", k])
for L in map(str, range(1, 8)):
    pair("D", f"land, nine regions pooled: spread/skill against the estimates, {L} d", "4 (as run)", ESO, ESN,
         ["land", "2022 final", "leads", L, "Nine regions pooled"], conv=lambda x: (x["ss"], x["ss_ci"][0], x["ss_ci"][1]))

# ------------------------------------------------------------------------------------------- (E) extremes, exposure
DVO, DVN = N / "analyses/issuance/data/derived.json", N / "analyses/issuance/data_ar/derived.json"
for dom in ("land", "stations"):
    for thr in ("P99", "P95"):
        for kind in ("Foehn", "Foehn minus CAMS, equal lead", "Foehn minus Aurora, equal lead", "Foehn at lead h minus CAMS at 1 day"):
            for L in (1, 3, 5, 7):
                def cv(x, L=L, kind=kind):
                    v = x["sedi"] if kind == "Foehn" else x["diff"]
                    if L > len(v):
                        return None
                    return (v[L - 1], x["ci_low"][L - 1], x["ci_high"][L - 1])
                RG = "Global" if dom == "land" else "all"
                try:
                    dig(J(DVO), [dom, "ghap", thr, RG, kind])
                except KeyError:
                    continue
                pair("E", f"SEDI ({dom}, threshold {thr} of the estimates, {RG}): {kind}, {L} d", "4 (as run)", DVO, DVN,
                     [dom, "ghap", thr, RG, kind], conv=cv)
        try:
            pair("E", f"SEDI ({dom}, {thr}): lead at which Foehn matches the CAMS forecast at 1 day (crossing)", "4 (as run)", DVO, DVN,
                 [dom, "ghap", thr, "Global" if dom == "land" else "all", "match with CAMS at 1 day"],
                 conv=lambda x: (x["crossing_lead"], *(x["crossing_ci"] if isinstance(x.get("crossing_ci"), list) else (None, None))))
        except Exception:
            pass
HXO, HXN = N / "analyses/hidden/results/numbers.json", N / "analyses/hidden/results_ar/numbers.json"
for t in ("37.5", "75"):
    for q in ("hidden_M", "hidden_pct", "exposed_M"):
        pair("E", f"exposure (truth only), world, > {t}: {q}", "-", HXO, HXN, ["exposure_all_pixels", f"thr{t}", "World", q])
    for L in ("lead1", "lead3", "lead5", "lead7"):
        for q in ("reach_pct", "correct_pct", "hidden_warned_pct"):
            pair("E", f"lead ladder (four members, 25 km blocks), > {t}, {L}: {q}", "4 (as run)", HXO, HXN,
                 ["lead_ladder", f"thr{t}", "foehn_25km", L, q])

# ------------------------------------------------------------------------------------------------ (F) circularity
CIO, CIN = N / "analyses/circularity/results/numbers.json", N / "analyses/circularity/results_ar/numbers.json"
for g in ("all six networks", "China", "USA", "Europe", "Korea", "Japan", "India"):
    for L in ("1", "3"):
        for k in ("foehn", "ring3", "ring10"):
            for ref in ("cams", "aurora"):
                pair("F", f"rings, {g}: gain of Foehn read at the {'pixel' if k == 'foehn' else k} over {ref}, {L} d (%)", "4 (as run)",
                     CIO, CIN, ["C", "groups", g, "leads", L, f"gain_{k}_vs_{ref}"])
    pair("F", f"rings, {g}: stations with station-days 1 day ahead (option --stations-by-lead; absent in the stored file)", "-",
         CIO, CIN, ["C", "groups", g, "leads", "1", "station_days"], note="station-days, 1 day")

# ------------------------------------------------------------------------------- (G) case studies, official stations
OFO, OFN = P / "test2022_official/results/scores_official_2022.json", P / "test2022_official/results_ar/scores_official_2022.json"
for L in ("1", "2", "3", "4", "5"):
    for q in ("rmse", "crps", "spread_skill", "coverage_member_range"):
        pair("G", f"official stations, table F, Foehn (4 members), six networks: {q}, lead day {L}", "4 (as run)", OFO, OFN,
             ["tables", "pixel", "F", "arms", "foehn", "All six networks", L, q])
CASE_O = {"transport": FIG / "logs/render_fig_transport.log", "smoke": FIG / "logs/render_fig_smoke.log"}
CASE_N = N / "rebuild/s1_chain/logs/s1_step9_23910629.out"


def case_lines(p, pat):
    out = {}
    for line in Path(p).read_text(errors="replace").splitlines():
        m = re.search(pat, line)
        if m:
            out[m.group(1)] = float(m.group(2))
    sha(p)
    return out


ko = case_lines(CASE_O["transport"], r"issued (\d) d before: members at Kano .*P>100 ([\d.]+)")
kn = case_lines(CASE_N, r"issued (\d) d before: members at Kano .*P>100 ([\d.]+)")
for d in sorted(ko):
    add("G", f"Kano 16 May 2022: probability of > 100 ug m-3, issued {d} d before (16 members)", "16", CASE_O["transport"], CASE_N,
        f"log line 'issued {d} d before: members at Kano ... P>100'", ko.get(d), kn.get(d))
go = case_lines(CASE_O["smoke"], r"issued (\d) d before .*reserve P>25 ([\d.]+)")
gn = case_lines(CASE_N, r"issued (\d) d before .*reserve P>25 ([\d.]+)")
for d in sorted(go):
    add("G", f"Gironde 18 July 2022: reserve probability of > 25 ug m-3, issued {d} d before (16 members)", "16", CASE_O["smoke"], CASE_N,
        f"log line 'issued {d} d before ... reserve P>25'", go.get(d), gn.get(d))

# ------------------------------------------------------------------------------------------------------- (H) 2021
for L in ("1", "7"):
    pair("H", f"2021 (validation): spread/skill against the estimates on land, nine regions pooled, {L} d", "4 (as run)", ESO, ESN,
         ["land", "2021 final (validation year)", "leads", L, "Nine regions pooled"], conv=lambda x: (x["ss"], *(x.get("ss_ci") or (None, None))))
    pair("H", f"2021 (validation): RMSE of the mean against the estimates on land, {L} d", "4 (as run)", ESO, ESN,
         ["land", "2021 final (validation year)", "leads", L, "Nine regions pooled", "rmse"])
    pair("H", f"2021 (validation): fair CRPS against the estimates on land, {L} d", "4 (as run)", ESO, ESN,
         ["land", "2021 final (validation year)", "leads", L, "Nine regions pooled", "crps"])

# ---------------------------------------------------------------------------------- (I) episode statistics, A16 Q1-Q6
S22 = P / "a16/results/score22.json"
s22 = J(S22)
for t in ("37.5", "P95", "75"):
    for q, v in s22["Q"][t].items():
        if not isinstance(v, dict) or "stored" not in v:
            continue
        ROWS.append(dict(cat="I", quantity=f"{q} at {t}", form="member 0", old=tri(v.get("stored")), new=tri(v.get("arm")),
                         truth=tri(v.get("truth")), D=tri(v.get("D")), rel_change_pct=None, src_old=str(S22), src_new=str(S22),
                         key=f"Q/{t}/{q}/{{stored,arm,truth,D}} (est, two_way)", note=""))
q6 = s22["Q6"]
for part, lab in (("a_member0", "member 0"), ("b_mean", "mean of the four")):
    x = q6[part]
    for i in range(7):
        ROWS.append(dict(cat="I", quantity=f"Q6 RMSE of the time mean over the 179 starts, lead {i + 1} ({lab}; ug m-3)", form=lab,
                         old=(x["rmse_st"][i], None, None), new=(x["rmse_arm"][i], None, None), truth=None,
                         D=(x["ratio_m1"][i] * 100, x["ratio_m1_runs_ci"][0][i] * 100, x["ratio_m1_runs_ci"][1][i] * 100),
                         rel_change_pct=x["ratio_m1"][i] * 100, src_old=str(S22), src_new=str(S22), key=f"Q6/{part}/rmse_st|rmse_arm|ratio_m1[{i}]", note=""))


# ============================================================================================================= lists
def sig(t):
    """(sign of the point, interval excludes zero) ; None parts when unknown"""
    if t is None or t[0] is None:
        return None
    s = int(np.sign(round(t[0], 12)))
    ex = None if t[1] is None or t[2] is None else bool(t[1] > 0 or t[2] < 0)
    return s, ex


def differs(a, b):
    sa, sb = sig(a), sig(b)
    if sa is None or sb is None:
        return None
    why = []
    if sa[0] != sb[0]:
        why.append("sign")
    if sa[1] is not None and sb[1] is not None and sa[1] != sb[1]:
        why.append("exclusion of zero")
    return why


NUM = r"[-+]?\d+(?:\.\d+)?"
TRI = re.compile(rf"({NUM})\s*\(({NUM})\s*to\s*({NUM})\)")


def parse_cell(s, as_run=None):
    if s is None:
        return []
    s = str(s)
    if s.strip().startswith("= as run") and as_run is not None:
        s = as_run
    if re.search(r"not (computable|stored|computed|recomputed)", s):
        return []
    s = re.sub(r"\[[^\]]*\]", "", s)
    t = TRI.findall(s)
    if t:
        return [tuple(map(float, x)) for x in t]
    s = re.sub(r"\([^)]*[A-Za-z][^)]*\)", "", s)
    return [(float(x), None, None) for x in re.findall(rf"(?<![\w.]){NUM}(?![\w.])", s)]


CUO, CUN = P / "a14/classU/results/classU_table_179.json", P / "a14/classU/results_ar/classU_table_179.json"
cuo, cun = J(CUO)["rows"], J(CUN)["rows"]
assert [r["quantity"] for r in cuo] == [r["quantity"] for r in cun]
LIST2, LIST3, UNALIGNED = [], [], []
FU = (("4 (as run)", "as_run"), ("8", "M8"), ("inf", "inf"))
for i, (ro, rn) in enumerate(zip(cuo, cun), 1):
    cells_n = {f: parse_cell(rn.get(k), rn.get("as_run")) for f, k in FU}
    cells_o = {f: parse_cell(ro.get(k), ro.get("as_run")) for f, k in FU}
    # (ii) between forms, design 'ar': only rows whose three cells hold the same number of values (aligned); the others
    # (a cell 'RMSE = as run' beside 'CRPS ...', different sub-quantities per form) are listed as not compared
    # as-run cells that print a second set 'with Foehn's draw' (stored design) beside the as-run set: the first half is compared
    for cells, row in ((cells_n, rn), (cells_o, ro)):
        a_, m_, i_ = (len(cells[f]) for f, _ in FU)
        if a_ and m_ and a_ == 2 * m_ == 2 * i_ and "with Foehn's draw" in str(row.get("as_run")):
            cells["4 (as run)"] = cells["4 (as run)"][:m_]
    lens = {len(cells_n[f]) for f, _ in FU if cells_n[f]}
    if len(lens) > 1:
        UNALIGNED.append(dict(source=str(CUN), row=i, quantity=rn["quantity"], counts={f: len(cells_n[f]) for f, _ in FU}))
    for fa, fb in (("4 (as run)", "8"), ("4 (as run)", "inf"), ("8", "inf")):
        A_, B_ = cells_n[fa], cells_n[fb]
        if len(lens) == 1 and A_ and B_ and len(A_) == len(B_):
            for j, (a, b) in enumerate(zip(A_, B_)):
                w = differs(a, b)
                if w:
                    LIST2.append(dict(source=str(CUN), row=i, quantity=rn["quantity"], position=j + 1, forms=f"{fa} vs {fb}", why=w,
                                      values={fa: a, fb: b}, all_forms={f: (cells_n[f][j] if len(cells_n[f]) > j else None) for f, _ in FU}))
    # (iii) stored against 'ar', same form
    for f, _ in FU:
        A_, B_ = cells_o[f], cells_n[f]
        if A_ and B_ and len(A_) == len(B_):
            for j, (a, b) in enumerate(zip(A_, B_)):
                w = differs(a, b)
                if w:
                    LIST3.append(dict(source=f"{CUO} | {CUN}", row=i, quantity=ro["quantity"], position=j + 1, form=f, why=w, old=a, new=b))

# class E contrasts: every diff:/eqn: quantity of reach, correct, hidden warned, on conf149 (all forms) and all179 (four members)
ceo, cen = J(CEO)["results"]["world"], J(CEN)["results"]["world"]
CQ = re.compile(r"^(diff|eqn):[^:]+:(reach_pct|correct_pct|hidden_warned_pct|hidden_margin_warned_pct)$")
for smp in ("conf149", "all179"):
    for t in ("thr15", "thr37.5", "thr75"):
        for L in ("lead1", "lead3"):
            qo, qn = ceo[smp][t][L]["quantities"], cen[smp][t][L]["quantities"]
            for qk in sorted(qn):
                if not CQ.match(qk):
                    continue
                vals_n = {f: tri(qn[qk].get(mk)) for f, mk in FORMSE}
                vals_o = {f: tri(qo.get(qk, {}).get(mk)) for f, mk in FORMSE}
                if smp == "conf149":
                    for fa, fb in (("4", "8"), ("4", "inf"), ("8", "inf")):
                        w = differs(vals_n[fa], vals_n[fb])
                        if w:
                            LIST2.append(dict(source=str(CEN), row=f"{smp}/{t}/{L}", quantity=qk, position=1, forms=f"{fa} vs {fb}", why=w,
                                              values={fa: vals_n[fa], fb: vals_n[fb]}, all_forms=vals_n))
                for f, _ in FORMSE:
                    if smp == "all179" and f != "4":
                        continue
                    w = differs(vals_o[f], vals_n[f])
                    if w:
                        LIST3.append(dict(source=f"{CEO} | {CEN}", row=f"{smp}/{t}/{L}", quantity=qk, position=1, form=f, why=w,
                                          old=vals_o[f], new=vals_n[f]))
# every row of (A)-(H) with an interval, stored against 'ar'
for r in ROWS:
    if r["cat"] in "ABDEFGH" and r["form"] != "-":          # C: covered by the class E walk above
        w = differs(r["old"], r["new"])
        if w and not any(x.get("key") == r["key"] and x.get("form") == r["form"] for x in LIST3):
            LIST3.append(dict(source=f"{r['src_old']} | {r['src_new']}", row=r["cat"], quantity=r["quantity"], position=1,
                              form=r["form"], why=w, old=r["old"], new=r["new"], key=r["key"]))
# official stations: every contrast of tables F and C, all six networks and by network
for tb in ("F", "C"):
    co, cn_ = J(OFO)["tables"]["pixel"][tb]["contrasts"], J(OFN)["tables"]["pixel"][tb]["contrasts"]

    def walkc(a, b, p):
        if isinstance(b, dict):
            t_ = tri(b)
            if t_ is not None and isinstance(a, dict):
                w = differs(tri(a), t_)
                if w:
                    LIST3.append(dict(source=f"{OFO} | {OFN}", row=f"tables/pixel/{tb}/contrasts", quantity="/".join(p), position=1,
                                      form="4 (as run)", why=w, old=tri(a), new=t_))
                return
            for k in b:
                if isinstance(a, dict) and k in a:
                    walkc(a[k], b[k], p + [k])
    walkc(co, cn_, [])
# derived.json: every 'diff' with its interval (SEDI contrasts), land and stations, every region and threshold
def walkd(a, b, p):
    if isinstance(b, dict):
        if "diff" in b and "ci_low" in b and isinstance(b["diff"], list):
            for i_, (x, y) in enumerate(zip(a["diff"], b["diff"])):
                if x is None or y is None:
                    continue
                o = (x, a["ci_low"][i_], a["ci_high"][i_]); n_ = (y, b["ci_low"][i_], b["ci_high"][i_])
                w = differs(o, n_)
                if w:
                    LIST3.append(dict(source=f"{DVO} | {DVN}", row="derived", quantity="/".join(p) + f" [lead {i_ + 1}]", position=1,
                                      form="4 (as run)", why=w, old=o, new=n_))
            return
        for k in b:
            if isinstance(a, dict) and k in a:
                walkd(a[k], b[k], p + [k])
walkd(J(DVO), J(DVN), [])
# circularity: every gain of the ring and pixel analyses (groups A and C, leads)
def walkg(a, b, p):
    if isinstance(b, dict):
        for k in b:
            if isinstance(a, dict) and k in a:
                walkg(a[k], b[k], p + [k])
    elif isinstance(b, list) and len(b) == 3 and p and p[-1].startswith("gain") and isinstance(a, list):
        w = differs(tri(a), tri(b))
        if w:
            LIST3.append(dict(source=f"{CIO} | {CIN}", row="circularity", quantity="/".join(p), position=1, form="4 (as run)",
                              why=w, old=tri(a), new=tri(b)))
walkg(J(CIO), J(CIN), [])

# ------------------------------------------------------------------------------------- (i) reference columns
LIST1 = []
REFSYS = re.compile(r"(cams|aurora|persist|clim|bcsd|qm|ghap|estimate|truth|coarse|ds_|chain|bil|perfect)", re.I)
# leaves that name a reference but are Foehn's own quantities (contrasts, Foehn against a reference, a reference moved to
# Foehn's number warned, Foehn read on rings, Foehn's skill against the climatology)
EXCL = (r"(foehn|crps_e01|issued|diff:|eqn:|eqc:|gain|minus|vs_|ratio|factor|identity|seconds|built|written|file|sha|better|"
        r"ring|band|one3|one10|bss_clim|bss_base|_vs_clim|equal_number|match with|crossing|share_stations|(^|/)p_|\|inf|\|8)")
FOEHN = re.compile(r"(foehn|crps_e01|1 km\)$)", re.I)
for name, ax_cands in (("compare_fair", None), ("compare_ghap_global", None), ("extremes_stations", None), ("tail_events", None),
                       ("extremes_stations_mean", None), ("extremes_stations_own", None)):
    fo, fn = FIG / f"data/{name}_crps_e01.npz", FIG / f"data/{name}_crps_e01_ar.npz"
    if not (fo.exists() and fn.exists()):
        LIST1.append(dict(what=f"{name}: file absent", identical=None)); continue
    zo, zn = np.load(fo, allow_pickle=True), np.load(fn, allow_pickle=True)
    sha(fo); sha(fn)
    syst = [str(s) for s in zn["systems"]] if "systems" in zn.files else None
    for k in zn.files:
        if zn[k].dtype.kind not in "fiu" or syst is None:
            continue
        a, b = zo[k], zn[k]
        axes = [i for i, n in enumerate(b.shape) if n == len(syst)]
        res = []
        for ax in axes:
            refs = [i for i, s in enumerate(syst) if not FOEHN.search(s)]
            ok = all(np.array_equal(np.take(a, i, axis=ax), np.take(b, i, axis=ax), equal_nan=True) for i in refs)
            foe = [i for i in range(len(syst)) if i not in refs]
            fchg = any(not np.array_equal(np.take(a, i, axis=ax), np.take(b, i, axis=ax), equal_nan=True) for i in foe)
            res.append(dict(axis=ax, references_identical=ok, foehn_changed=fchg))
        if res:
            good = [r_ for r_ in res if r_["references_identical"]]
            LIST1.append(dict(what=f"{fn.name}:{k} (systems {syst})", candidates=res,
                              identical=bool(good), file_old=str(fo), file_new=str(fn)))
# JSON records: every leaf of the reference systems
def walkr(a, b, p, src, out):
    if isinstance(b, dict):
        for k in b:
            if isinstance(a, dict) and k in a:
                walkr(a[k], b[k], p + [str(k)], src, out)
    elif isinstance(b, list):
        if isinstance(a, list) and len(a) == len(b):
            for i_, (x, y) in enumerate(zip(a, b)):
                walkr(x, y, p + [f"[{i_}]"], src, out)
    else:
        path = "/".join(p)
        if (isinstance(b, (int, float)) and not isinstance(b, bool)) and REFSYS.search(path) and not re.search(EXCL, path, re.I):
            out[0] += 1
            if isinstance(a, (int, float)) and a != b and not (a != a and b != b):
                out[1].append((path, a, b))
                if abs(a - b) > 1e-12 * max(abs(a), abs(b), 1e-300):
                    out[2].append((path, a, b))


for fo, fn, lab, sub in ((LO, LN, "land25_classU references", ["leads"]), (OFO, OFN, "official stations: every arm but Foehn", ["tables"]),
                         (DVO, DVN, "issuance derived: CAMS and Aurora", ["land"]), (DVO, DVN, "issuance derived: CAMS and Aurora", ["stations"]),
                         (CIO, CIN, "circularity: CAMS, Aurora, climatology, estimates", ["C"]), (CIO, CIN, "circularity", ["A"]),
                         (CIO, CIN, "circularity, Foehn-CAMS (class d)", ["D"]), (SSO, SSN, "station scores: estimates", ["leads"]),
                         (HXO, HXN, "hidden v1: references and exposure", [])):
    out = [0, [], []]
    a, b = dig(J(fo), sub), dig(J(fn), sub)
    if lab.startswith("official"):
        for tb in a:
            for t2 in a[tb]:
                for arm in a[tb][t2].get("arms", {}):
                    if arm != "foehn":
                        walkr(a[tb][t2]["arms"][arm], b[tb][t2]["arms"][arm], [tb, t2, "arms", arm, "ref"], fo, out)
    else:
        walkr(a, b, sub, fo, out)
    LIST1.append(dict(what=f"{lab} ({fn.name}, under {'/'.join(sub) or 'root'})", leaves_compared=out[0], differing=len(out[1]),
                      differing_beyond_1e12_rel=len(out[2]), first_differences=(out[2] or out[1])[:8], identical=(len(out[1]) == 0),
                      identical_to_1e12=(len(out[2]) == 0), file_old=str(fo), file_new=str(fn)))
# class E: the references' own values (issued minus diff) as issued, every sample, threshold, lead, M8 (any M gives the same)
out = [0, []]
for smp in ceo:
    for t in ceo[smp]:
        for L in ceo[smp][t]:
            qo, qn = ceo[smp][t][L]["quantities"], cen[smp][t][L]["quantities"]
            for qk in qn:
                m = re.match(r"^diff:([^:]+):(.+)$", qk)
                if not m or f"issued:{m.group(2)}" not in qn or m.group(1) == "foehn_40km":   # foehn_40km: Foehn's own 40 km mean
                    continue
                for mk in ("M8_liberal", "M4_stored"):
                    try:
                        ro_ = qo[f"issued:{m.group(2)}"][mk]["value"] - qo[qk][mk]["value"]
                        rn_ = qn[f"issued:{m.group(2)}"][mk]["value"] - qn[qk][mk]["value"]
                    except (KeyError, TypeError):
                        continue
                    out[0] += 1
                    if abs(ro_ - rn_) > 1e-9 * max(1.0, abs(ro_)):
                        out[1].append((f"{smp}/{t}/{L}/{m.group(1)}:{m.group(2)} ({mk})", ro_, rn_))
LIST1.append(dict(what="class E: each reference's own value (issued minus diff; persist, cams, aurora, cams_qm, cams_bcsd, clim_1km; foehn_40km is Foehn's own 40 km mean and is left out), every sample, threshold, lead, M8 and M4 (tolerance 1e-9 rel.)",
                  leaves_compared=out[0], differing=len(out[1]), first_differences=out[1][:8], identical=(len(out[1]) == 0),
                  file_old=str(CEO), file_new=str(CEN)))
LDF = HERE / "audit/leafdiff_ar.json"
if LDF.exists():
    ld = json.loads(LDF.read_text())          # 1.1 GB: read once, not hashed
    fl = [x for x in ld if x.get("ref_flag")]
    nch = len(ld); del ld
    cls = {"Foehn-dependent by construction (path holds a contrast, Foehn's skill, or a reference moved to Foehn's number warned)": [],
           "absent under 'ar' (the paper-identity checks of score_official are not run under a design: --no-paper-check)": [],
           "float rounding (relative difference <= 1e-12)": [], "UNEXPLAINED": []}
    for x in fl:
        o_, n_ = x["old"], x["new"]
        if re.search(EXCL, x["path"], re.I) or re.search(r"(curve|fits|intercept|K1|equal_number|ratio:|_cells|crossing|/C/groups/.*/leads/|land_sedi_region_lead)", x["path"]):
            cls[list(cls)[0]].append(x)
        elif n_ == "<absent>" and "/checks/" in x["path"]:
            cls[list(cls)[1]].append(x)
        elif isinstance(o_, (int, float)) and isinstance(n_, (int, float)) and abs(o_ - n_) <= 1e-12 * max(abs(o_), abs(n_), 1e-300):
            cls[list(cls)[2]].append(x)
        else:
            cls["UNEXPLAINED"].append(x)
    LIST1.append(dict(what="leafdiff.py ar: changed leaves whose path names a reference and no Foehn term of leafdiff's own list, classified",
                      leaves_compared=f"{nch} changed leaves in all", differing=len(fl),
                      classes={k: len(v) for k, v in cls.items()},
                      first_differences=[(x["file"].replace(str(N), "N").replace(str(P), "P"), x["path"], x["old"], x["new"]) for x in cls["UNEXPLAINED"][:12]],
                      identical=(len(cls["UNEXPLAINED"]) == 0), file_old="(leafdiff)", file_new=str(LDF)))

# ================================================================================================================ write
def fmt(t, d=3):
    if t is None or t[0] is None:
        return "n/a"
    s = f"{t[0]:.{d}f}"
    if t[1] is not None:
        s += f" [{t[1]:.{d}f}, {t[2]:.{d}f}]"
    return s


def short(p):
    return str(p).replace(str(N), "N").replace(str(P), "P").replace(str(FIG), "FIG")


res = dict(what=__doc__, files={short(k): v for k, v in sorted(FILES.items())}, rows=ROWS, list_i_references=LIST1,
           list_ii_forms_under_ar=LIST2, list_ii_class_u_rows_not_compared=UNALIGNED, list_iii_stored_vs_ar=LIST3)


def safe_write(path, text):
    data = text.encode("utf-8"); tmp = str(path) + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data); fh.flush(); os.fsync(fh.fileno())
    assert open(tmp, "rb").read() == data
    os.replace(tmp, path)
    assert open(path, "rb").read() == data


def dflt(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


safe_write(HERE / "old_vs_new.json", json.dumps(res, indent=1, default=dflt, ensure_ascii=False) + "\n")

CATS = {"A": "Stations of 2022 (RMSE of the mean, reductions against CAMS and Aurora, CRPS, spread/skill)",
        "B": "Against the 1 km estimates", "C": "Warnings on the 149 confirmatory start days (class E record; D5 rows)",
        "D": "Reliability and coverage", "E": "Extremes (SEDI)", "F": "Circularity (rings at 3 and 10 km)",
        "G": "Case studies and the official stations (table F)", "H": "2021 (validation year) of the final model",
        "I": "Episode statistics and time means of 2022 (A16 Q1-Q6, a16/results/score22.json)"}
L_ = ["# Old (stored noise fields) against new (design 'ar', A16)", "",
      "Written by `old_vs_new.py` (this folder). Every value is read from the file and key given; old = the stored design (the "
      "current manuscript), new = 'ar'. Intervals: 95 %, as in each record (stored: widened for the draw where A14 4c applies; "
      "'ar': V_draw = 0, plain bootstrap). Relative change = (new - old)/|old| x 100. Files and sha256 (first 16 hex): see the end.", ""]
for c, title in CATS.items():
    rs = [r for r in ROWS if r["cat"] == c]
    if not rs:
        continue
    L_ += [f"## ({c}) {title}", ""]
    if c == "I":
        L_ += ["| statistic | stored | new | truth | D (or new/stored - 1, %) | key |", "|---|---|---|---|---|---|"]
        for r in rs:
            L_.append(f"| {r['quantity']} | {fmt(r['old'])} | {fmt(r['new'])} | {fmt(r.get('truth'))} | {fmt(r.get('D'))} | `{r['key']}` |")
    else:
        L_ += ["| quantity | form | old | new | rel. change (%) | file old / new : key |", "|---|---|---|---|---|---|"]
        for r in rs:
            rc = "n/a" if r["rel_change_pct"] is None else f"{r['rel_change_pct']:+.2f}"
            src = short(r["src_old"]) if r["src_old"] == r["src_new"] else f"{short(r['src_old'])} / {short(r['src_new'])}"
            L_.append(f"| {r['quantity']} | {r['form']} | {fmt(r['old'])} | {fmt(r['new'])} | {rc} | {src} : `{r['key']}`{(' ' + r['note']) if r['note'] else ''} |")
    L_.append("")
L_ += ["## List (i): reference columns, stored against 'ar'", "",
       "| check | leaves / arrays compared | differing | identical | first differences |", "|---|---|---|---|---|"]
for x in LIST1:
    extra = f" (beyond 1e-12 relative: {x['differing_beyond_1e12_rel']})" if "differing_beyond_1e12_rel" in x else ""
    extra += f" {x['classes']}" if "classes" in x else ""
    L_.append(f"| {x['what']} | {x.get('leaves_compared', x.get('candidates', ''))} | {x.get('differing', '')}{extra} | "
              f"{x.get('identical')} | {str(x.get('first_differences', ''))[:300]} |")
L_ += ["", "## List (ii): A16 Addendum 3, 2(e): under 'ar', sign of the point or exclusion of zero differs between two forms (4, 8, infinite)", "",
       "| source | row | quantity | position | forms | why | 4 / 8 / infinite |", "|---|---|---|---|---|---|---|"]
for x in LIST2:
    af = x["all_forms"]
    L_.append(f"| {short(x['source'])} | {x['row']} | {x['quantity'][:110]} | {x['position']} | {x['forms']} | {', '.join(x['why'])} | "
              f"{fmt(af.get('4 (as run)', af.get('4')), 2)} / {fmt(af.get('8'), 2)} / {fmt(af.get('inf'), 2)} |")
L_ += ["", "Class-U rows not compared by the rule (the three cells hold different numbers of values; to be read by hand):", ""]
L_ += [f"- row {x['row']}: {x['quantity'][:140]} (values per form {x['counts']})" for x in UNALIGNED]
L_ += ["", "## List (iii): sign of the point or exclusion of zero differs between the stored design and 'ar' (same form)", "",
       "| source | row | quantity | position | form | why | old | new |", "|---|---|---|---|---|---|---|---|"]
for x in LIST3:
    L_.append(f"| {short(x['source'])} | {x['row']} | {str(x['quantity'])[:120]} | {x['position']} | {x['form']} | {', '.join(x['why'])} | {fmt(x['old'], 2)} | {fmt(x['new'], 2)} |")
L_ += ["", "## Files (sha256, first 16 hex)", ""] + [f"- {short(k)}: {v[:16]}" for k, v in sorted(FILES.items())]
safe_write(HERE / "old_vs_new.md", "\n".join(L_) + "\n")
print(f"rows {len(ROWS)}; list (i) {len(LIST1)} checks, not identical: {sum(1 for x in LIST1 if x.get('identical') is False)}; "
      f"list (ii) {len(LIST2)}; list (iii) {len(LIST3)}")
