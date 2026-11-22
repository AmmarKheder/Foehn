#!/usr/bin/env python3
"""A14 class U: the ONE table (results/classU_table.md + .json) of every class-U quantity the paper reports, with its place in
the paper (file:line), the as-run value (identity with the paper's printed value), the eight-member and infinite values and
their intervals (A14 4c), from the result files of this folder. Sixteen-member values: the 179-day files (*_179.json) when
R1 has passed check13, else the 30-day files (labelled provisional). Gains are written as reductions in % (positive = Foehn
better), like the paper.
    python make_table.py   -> results/classU_table.md, results/classU_table.json"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from u_common import AR, DESIGN_NOTE, P, RES, RES0, SECT, SUPP, safe_json, safe_text

OTHER = ("cams12h_classU.json", "cams12h_classU_cams12h_lag1b.json", "ds2022_classU.json", "rt_classU.json")   # other systems: stored records under every design

TEX = {"main": SECT / "main_text.tex", "methods": SECT / "methods.tex", "SI": SUPP}
LINES = {k: p.read_text().split("\n") for k, p in TEX.items()}


def loc(*specs):
    """specs: (file key, regex) -> 'file:line' for each match (first match per spec)"""
    out = []
    for fk, pat in specs:
        rx = re.compile(pat)
        hit = [i + 1 for i, l in enumerate(LINES[fk]) if rx.search(l)]
        out.append(f"{TEX[fk].name}:{hit[0]}" if hit else f"{TEX[fk].name}:?({pat})")
    return "; ".join(out)


def J(name):
    R_ = RES0 if name in OTHER else RES
    p179 = R_ / name.replace(".json", os.environ.get("A14U_PREFER", "_179") + ".json")
    p = p179 if p179.exists() else R_ / name
    return (json.loads(p.read_text()), p.name) if p.exists() else (None, None)


def f_(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def vci(v, ci=None, d=1, neg=False):
    s = -1 if neg else 1
    if v is None:
        return "n/a"
    if ci is None or ci[0] is None:
        return f_(s * v, d)
    lo, hi = sorted([s * ci[0], s * ci[1]])
    return f"{f_(s * v, d)} ({f_(lo, d)} to {f_(hi, d)})"


ROWS = []
CONC = []            # conclusions that differ between forms (A14 section 3), as-run value first
CHK = []             # checked, no change of conclusion


def row(section, quantity, where, printed, as_run, m8, inf, notes="", identity=None):
    ROWS.append(dict(section=section, quantity=quantity, paper=where, printed=printed, as_run=as_run, M8=m8, inf=inf,
                     identity=identity, notes=notes))


# ============================================================================ 1. stations, compare_fair.py
st, stf = J("st_cf_classU.json")
if st:
    G = st["groups"]
    rep = st["as_run_lines_reproduced"]
    idok = all(rep.values())

    def g(gn, h, sys_="Foehn 1 km"):
        return G[gn][str(h)][sys_]

    def e16(gn, h):
        """the sixteen-member values of a group and lead when every start day has them (A14 section 4), else {}"""
        x = G[gn][str(h)]["Foehn 1 km"].get("sixteen_members_provisional", {}) if h in (1, 3) else {}
        return x if "rmse16_inf" in x else {}
    ALL16 = bool(e16("all", 1))

    def rm(gn, h, form, sys_="Foehn 1 km", src="auto"):
        """RMSE in a form; at leads 1 and 3 the M=8 and infinite values come from the sixteen members when all start days have
        them (marked *); as run = the four stored members (the paper's), interval widened for their draw (fix 9)"""
        x = e16(gn, h)
        if src == "auto" and sys_ == "Foehn 1 km" and form != "as_run" and f"rmse16_{form}" in x:
            return vci(x[f"rmse16_{form}"]["v"], x[f"rmse16_{form}"]["ci"], 2) + "*"
        e = g(gn, h, sys_)["rmse"][form]
        return vci(e["v"], e.get("ci", e["ci_boot"]), 2)

    def gain(gn, h, ref, form, sys_="Foehn 1 km", d=1, src="auto"):
        x = e16(gn, h)
        if src == "auto" and sys_ == "Foehn 1 km" and form != "as_run" and f"gain16_{form}_{ref}" in x:
            return vci(x[f"gain16_{form}_{ref}"]["v"], x[f"gain16_{form}_{ref}"]["ci"], d, neg=True) + "*"
        e = g(gn, h, sys_)["gain_pct"][ref][form]
        return vci(e["v"], e.get("ci", e["ci_boot"]), d, neg=True)

    def beside(items):
        """the four-member M=8 / inf values beside the sixteen-member ones: items = list of (label, fn(form, src))"""
        if not ALL16:
            return ""
        return "four members beside: " + "; ".join(f"{lab} M=8 {fn('M8', '4')}, inf {fn('inf', '4')}" for lab, fn in items)

    def s16(gn, h, ref=None):
        e = g(gn, h).get("sixteen_members_provisional")
        if not e:
            return ""
        lab = f"16 members, {e['days']} days" + ("" if e["days"] >= 170 else " (provisional)")
        if "rmse16_inf" in e:                                   # all start days: paper bootstrap + 16-field jackknife
            if ref is None:
                return (f"{lab}: RMSE inf {vci(e['rmse16_inf']['v'], e['rmse16_inf']['ci'], 2)}, M=8 {vci(e['rmse16_M8']['v'], e['rmse16_M8']['ci'], 2)}; "
                        f"stored 4 on the same days: as run {e['stored4_same_days']['as_run']:.2f}, inf {e['stored4_same_days']['inf']:.2f}")
            gi, g8 = e[f"gain16_inf_{ref}"], e[f"gain16_M8_{ref}"]
            return f"{lab}: gain inf {vci(gi['v'], gi['ci'], 1, True)}, M=8 {vci(g8['v'], g8['ci'], 1, True)}"
        if ref is None:
            return f"{lab}: RMSE inf {e['sixteen_inf']:.2f} (jackknife sd {e['jk16_sd_inf']:.2f}), M=8 {e['sixteen_M8']:.2f}; stored 4 on the same days: as run {e['stored4_same_days']['as_run']:.2f}, inf {e['stored4_same_days']['inf']:.2f}"
        if f"gain_pct_{ref}" not in e:
            return ""
        q = e[f"gain_pct_{ref}"]
        return (f"{lab}: gain inf {-q['inf_16']:.1f} (jackknife sd {q['jk16_sd_inf']:.1f}), M=8 {-q['M8_16']:.1f}; stored 4 on the same days: "
                f"as run {-q['as_run_stored4']:.1f}, inf {-q['inf_stored4']:.1f}")
    A, CA, AU = "all", "CAMS d+1 (equal issuance)", "Aurora 00 UTC d+1 (equal issuance)"
    def row16(sec_, q_, where, printed, gn, h, ref, idok_, extra=""):
        """a row at lead h whose M=8 / inf columns are sixteen-member values when available (four members beside)"""
        f4 = (lambda f, src="auto": rm(gn, h, f, src=src)) if ref is None else (lambda f, src="auto": gain(gn, h, ref, f, src=src))
        note = beside([("", f4)]) if (ALL16 and h in (1, 3)) else (s16(gn, h, ref) if h in (1, 3) else "")
        row(sec_, q_, where, printed, f4("as_run"), f4("M8"), f4("inf"), "; ".join(x for x in (note, extra) if x), idok_)

    sec = ("1 Stations, retrospective Foehn (compare_fair.py). As run = the four stored members, interval widened for their draw (the bootstrap-only interval is the identity check). "
           + (f"M=8 / inf at leads 1 and 3: sixteen members on all {e16('all', 1).get('days', '?')} start days (marked *), four members beside; other leads: four members"
              if ALL16 else "M=8 / inf: four stored members; at leads 1 and 3 they are provisional (the sixteen-member values follow from r1_pipeline_b)"))
    row16(sec, "RMSE of the ensemble mean 1 day ahead (ug m-3)", loc(("methods", r"the RMSE of \\model\{\} is 12\.93"), ("main", r"1 day ahead is 12\.9")),
          "12.93 (12.9)", A, 1, None, idok)
    row16(sec, "RMSE reduction against the CAMS forecast, 1 day ahead (%)",
          loc(("main", r"16\\% below that of the operational"), ("main", r"These are gains of 16"), ("methods", r"reductions of 16\.0"), ("main", r"its error is 16\\% below")),
          "16.0 (11-23)", A, 1, CA, idok)
    row16(sec, "RMSE reduction against Aurora, 1 day ahead (%)", loc(("main", r"These are gains of 16"), ("methods", r"reductions of 16\.0\\% and 18\.5")),
          "18.5 (15-24)", A, 1, AU, idok)
    row16(sec, "RMSE 3 days ahead (ug m-3)", loc(("SI", r"against 13\.9, 14\.2 and 14\.6")), "14.2", A, 3, None, idok)
    row16(sec, "RMSE reduction against the CAMS forecast / Aurora, 3 days ahead (%): CAMS", loc(("SI", r"Three days ahead, the gains over the CAMS forecast and Aurora are 12")), "12 (7-21) (ring table)", A, 3, CA, None,
          "the SI's 12 / 17 % are on the ring-analysis stations (not rebuilt); as run here on compare_fair's station set")
    row16(sec, "RMSE reduction against Aurora, 3 days ahead (%)", loc(("SI", r"Three days ahead, the gains over the CAMS forecast and Aurora are 12")), "17 (13-23) (ring table)", A, 3, AU, None)
    row16(sec, "RMSE reduction against the climatology, 1 day ahead (%) [lead-1 primary]", loc(("main", r"20\\% below that of the climatology")), "20", A, 1, "climatology", idok)
    row16(sec, "RMSE reduction against the start-day estimate, 1 day ahead (%) [lead-1 primary]", loc(("main", r"12\\% \(9--15\\%\) and 21")), "12 (9-15)", A, 1, "GHAP persistence (start-day estimate)", idok)
    row(sec, "RMSE 4 days ahead (ug m-3)", loc(("SI", r"RMSE four days ahead is 14\.6")), "14.6", rm(A, 4, "as_run"), rm(A, 4, "M8"), rm(A, 4, "inf"), "", idok)
    row(sec, "RMSE reduction against the CAMS forecast, 4 days ahead (%)", loc(("main", r"4 days ahead, the gains are 11"), ("SI", r"11\\% \(95\\% interval 6--19")),
        "11 (6-19)", gain(A, 4, CA, "as_run"), gain(A, 4, CA, "M8"), gain(A, 4, CA, "inf"), "", idok)
    row(sec, "RMSE reduction against Aurora, 4 days ahead (%)", loc(("main", r"4 days ahead, the gains are 11")), "16 (13-21)",
        gain(A, 4, AU, "as_run"), gain(A, 4, AU, "M8"), gain(A, 4, AU, "inf"), "", idok)
    for h in (2,):
        row(sec, f"RMSE {h} days ahead and reduction against the CAMS forecast (%)", loc(("SI", r"against 13\.9, 14\.2 and 14\.6")),
            {2: "13.9 (RMSE)", 3: "14.2 (RMSE)"}[h], f"{rm(A, h, 'as_run')}; {gain(A, h, CA, 'as_run')}", f"{rm(A, h, 'M8')}; {gain(A, h, CA, 'M8')}",
            f"{rm(A, h, 'inf')}; {gain(A, h, CA, 'inf')}", s16(A, 3, CA) if h == 3 else "", idok)
    for h, pr in ((4, "10"),):
        row(sec, f"RMSE reduction against the climatology of the estimates, {h} day(s) ahead (%)", loc(("main", r"20\\% below that of the climatology")),
            pr, gain(A, h, "climatology", "as_run"), gain(A, h, "climatology", "M8"), gain(A, h, "climatology", "inf"), "", idok)
    for h, pr in ((4, "21 (18-24)"),):
        row(sec, f"RMSE reduction against the start-day estimate kept unchanged, {h} day(s) ahead (%)", loc(("main", r"12\\% \(9--15\\%\) and 21")),
            pr, gain(A, h, "GHAP persistence (start-day estimate)", "as_run"), gain(A, h, "GHAP persistence (start-day estimate)", "M8"),
            gain(A, h, "GHAP persistence (start-day estimate)", "inf"), "", idok)
    S5 = "Foehn 0.5 deg average"
    row(sec, "Averaged to 0.5 deg: RMSE 1 day ahead (ug m-3)", loc(("SI", r"its error one day ahead is 13\.8")), "13.8",
        rm(A, 1, "as_run", S5), rm(A, 1, "M8", S5), rm(A, 1, "inf", S5), "four members; M=8 and inf: member 0 only (s2 of the averaged field = (4/3)(m0 - mean)^2, A14 4a); V_draw: CV of the 1 km station RMSE carried + the single-member draw of the correction (fix 11)", idok)
    row(sec, "Averaged to 0.5 deg: RMSE reduction against the CAMS forecast / Aurora, 1 day ahead (%)", loc(("main", r"still 11\\% \(5--19\\%\) and 13")),
        "11 (5-19) / 13 (9-20)", gain(A, 1, CA, "as_run", S5) + " / " + gain(A, 1, AU, "as_run", S5), gain(A, 1, CA, "M8", S5) + " / " + gain(A, 1, AU, "M8", S5),
        gain(A, 1, CA, "inf", S5) + " / " + gain(A, 1, AU, "inf", S5), "four members; M=8 and inf: member 0 only (fix 11)", idok)
    row(sec, "1 km against its own 0.25 deg average, 1 day ahead (RMSE reduction, %)", loc(("main", r"5\\% more accurate than its own 0\.25")),
        "5", gain(A, 1, "Foehn 0.25 deg average", "as_run"), gain(A, 1, "Foehn 0.25 deg average", "M8"), gain(A, 1, "Foehn 0.25 deg average", "inf"),
        "both sides in the same form (1 km: four members; 0.25 deg: member 0 only); intervals: 1 km draw only (the average treated as fixed; the draw of the contrast is not estimable from stored members) (fix 16)", idok)
    nets = {"Europe (EEA)": "Europe", "China (CNEMC)": "China", "USA (EPA)": "USA", "Korea (AirKorea)": "Korea", "Japan (NIES)": "Japan", "India (CPCB)": "India"}
    row(sec, "1 km against the 0.25 deg average by network, 1 day ahead (%): " + ", ".join(nets.values()), loc(("SI", r"closer to the measurements by 9\\%")),
        "1 / 5 / 8 / -1 / -10 / 9", " / ".join(gain(n, 1, "Foehn 0.25 deg average", "as_run", d=0) for n in nets),
        " / ".join(gain(n, 1, "Foehn 0.25 deg average", "M8", d=0) for n in nets), " / ".join(gain(n, 1, "Foehn 0.25 deg average", "inf", d=0) for n in nets), "order: " + ", ".join(nets.values()) + "; intervals: 1 km draw only (the average treated as fixed; the draw of the contrast is not estimable from stored members) (fix 16)", idok)
    row(sec, "Five networks outside Europe: RMSE reduction against the CAMS forecast, 1 day ahead (%)", loc(("main", r"The gain is as large \(18"), ("SI", r"outside Europe, the gain of \\model\{\} over the CAMS forecast 1 day ahead is 18")),
        "18 (12-27)", gain("five networks outside Europe", 1, CA, "as_run"), gain("five networks outside Europe", 1, CA, "M8"), gain("five networks outside Europe", 1, CA, "inf"), "; ".join(x for x in ("compare_fair bootstrap on PDN sums; printed value reproduced to its rounding",
        beside([("", lambda f, src: gain("five networks outside Europe", 1, CA, f, src=src))])) if x), True)
    row(sec, "Korea and Japan: RMSE reduction against the CAMS forecast / Aurora, 1 day ahead (%)", loc(("main", r"10\\% \(\$-\$0\.1 to 21")),
        "10 (-0.1 to 21) / 13 (5-22)", gain("Korea and Japan", 1, CA, "as_run") + " / " + gain("Korea and Japan", 1, AU, "as_run"),
        gain("Korea and Japan", 1, CA, "M8") + " / " + gain("Korea and Japan", 1, AU, "M8"), gain("Korea and Japan", 1, CA, "inf") + " / " + gain("Korea and Japan", 1, AU, "inf"),
        "; ".join(x for x in ("NOT the paper's sample: the printed 10 / 13 % come from the 10 km-ring analysis (stations with populated rings, Supplementary Table away, analyses/circularity), not rebuilt here; values on compare_fair's station set",
        beside([("CAMS", lambda f, src: gain("Korea and Japan", 1, CA, f, src=src)), ("Aurora", lambda f, src: gain("Korea and Japan", 1, AU, f, src=src))])) if x), False)
    row(sec, "Outside the Chinese network: RMSE reduction against the CAMS forecast, 1 / 2 / 3 / 4 days ahead (%)", loc(("SI", r"outside the Chinese network, the pooled gain")),
        "5 (0.6-10) / 3 / 2 / 2", " / ".join(gain("outside China", h, CA, "as_run") for h in range(1, 5)), " / ".join(gain("outside China", h, CA, "M8") for h in range(1, 5)),
        " / ".join(gain("outside China", h, CA, "inf") for h in range(1, 5)), beside([(f"lead {h}", (lambda hh: (lambda f, src: gain("outside China", hh, CA, f, src=src)))(h)) for h in (1, 3)]), True)
    row(sec, "Korea: RMSE reduction against the CAMS forecast 1-4 days (%) and against Aurora 1-4 days", loc(("SI", r"At the Korean network, the gain over the CAMS forecast is 7")),
        "7 (-5 to 21) / 16 / 14 / 19; Aurora 18-31", " / ".join(gain("Korea (AirKorea)", h, CA, "as_run", d=0) for h in range(1, 5)) + "; " + " / ".join(gain("Korea (AirKorea)", h, AU, "as_run", d=0) for h in range(1, 5)),
        " / ".join(gain("Korea (AirKorea)", h, CA, "M8", d=0) for h in range(1, 5)) + "; " + " / ".join(gain("Korea (AirKorea)", h, AU, "M8", d=0) for h in range(1, 5)),
        " / ".join(gain("Korea (AirKorea)", h, CA, "inf", d=0) for h in range(1, 5)) + "; " + " / ".join(gain("Korea (AirKorea)", h, AU, "inf", d=0) for h in range(1, 5)),
        beside([(f"CAMS lead {h}", (lambda hh: (lambda f, src: gain("Korea (AirKorea)", hh, CA, f, d=0, src=src)))(h)) for h in (1, 3)]), True)
    row(sec, "Japan: RMSE reduction against the CAMS forecast 1 day ahead at the pixel and for the 0.5 deg average; 2-4 days (%)", loc(("SI", r"At the Japanese network, it is 11")),
        "11 (-0.4 to 25); 0.5 deg 23 (14-34); 5 / -2 / 1", gain("Japan (NIES)", 1, CA, "as_run") + "; 0.5 deg " + gain("Japan (NIES)", 1, CA, "as_run", S5) + "; " + " / ".join(gain("Japan (NIES)", h, CA, "as_run", d=0) for h in (2, 3, 4)),
        gain("Japan (NIES)", 1, CA, "M8") + "; 0.5 deg " + gain("Japan (NIES)", 1, CA, "M8", S5) + "; " + " / ".join(gain("Japan (NIES)", h, CA, "M8", d=0) for h in (2, 3, 4)),
        gain("Japan (NIES)", 1, CA, "inf") + "; 0.5 deg " + gain("Japan (NIES)", 1, CA, "inf", S5) + "; " + " / ".join(gain("Japan (NIES)", h, CA, "inf", d=0) for h in (2, 3, 4)),
        "; ".join(x for x in ("0.5 deg: four members, M=8 / inf member 0 only", beside([(f"pixel lead {h}", (lambda hh: (lambda f, src: gain("Japan (NIES)", hh, CA, f, src=src)))(h)) for h in (1, 3)])) if x), True)
    row(sec, "Europe 1 day ahead: RMSE (ug m-3) and reduction against the CAMS forecast / Aurora (%)", loc(("SI", r"the RMSE of \\model\{\} is 11\.0"), ("methods", r"against 11\.0 for \\model\{\} 1 day ahead")),
        "11.0; 6 / 7", rm("Europe (EEA)", 1, "as_run") + "; " + gain("Europe (EEA)", 1, CA, "as_run") + " / " + gain("Europe (EEA)", 1, AU, "as_run"),
        rm("Europe (EEA)", 1, "M8") + "; " + gain("Europe (EEA)", 1, CA, "M8") + " / " + gain("Europe (EEA)", 1, AU, "M8"),
        rm("Europe (EEA)", 1, "inf") + "; " + gain("Europe (EEA)", 1, CA, "inf") + " / " + gain("Europe (EEA)", 1, AU, "inf"),
        beside([("RMSE", lambda f, src: rm("Europe (EEA)", 1, f, src=src)), ("CAMS", lambda f, src: gain("Europe (EEA)", 1, CA, f, src=src)), ("Aurora", lambda f, src: gain("Europe (EEA)", 1, AU, f, src=src))]), True)
    row(sec, "RMSE 1 day ahead by network (ug m-3): " + ", ".join(nets.values()), loc(("SI", r"1 day ahead has 15\.3, 4\.7, 11\.0")),
        "11.0 / 15.3 / 4.7 / 8.4 / 5.9 / 35.4", " / ".join(f_(g(n, 1)["rmse"]["as_run"]["v"], 1) for n in nets),
        " / ".join((f_(e16(n, 1)["rmse16_M8"]["v"], 1) + "*") if e16(n, 1) else f_(g(n, 1)["rmse"]["M8"]["v"], 1) for n in nets),
        " / ".join((f_(e16(n, 1)["rmse16_inf"]["v"], 1) + "*") if e16(n, 1) else f_(g(n, 1)["rmse"]["inf"]["v"], 1) for n in nets),
        "order: " + ", ".join(nets.values()) + ("; four members beside: M=8 " + " / ".join(f_(g(n, 1)["rmse"]["M8"]["v"], 1) for n in nets) + ", inf " + " / ".join(f_(g(n, 1)["rmse"]["inf"]["v"], 1) for n in nets) if ALL16 else ""), True)
    row(sec, "India 1 day ahead: RMSE (ug m-3), CAMS / Aurora / climatology 35.6 / 35.6 / 35.3", loc(("SI", r"RMSE 35\.4\\,\\textmu g")), "35.4",
        rm("India (CPCB)", 1, "as_run"), rm("India (CPCB)", 1, "M8"), rm("India (CPCB)", 1, "inf"), beside([("", lambda f, src: rm("India (CPCB)", 1, f, src=src))]), True)
    dr1 = st.get("dropped_16_station_days")
    if dr1:
        row(sec, "Sixteen-member station-days of compare_fair's sample, leads 1 and 3: fallback and dropped (fix 17)", "-", "-",
            f"{dr1['fallback4_total']} station-days kept with their four stored members (stations without a populated pixel), {dr1['unexpected_total']} unexpected, over {dr1['forecasts']} forecasts",
            "", "", dr1.get("note", ""), None)

# ============================================================================ 2. stations, analyses/reliability
rl, rlf = J("rel_classU.json")
if rl:
    S = rl["stations_reliability"]["by_lead"]; A = "All official networks"
    idok = not rl["stations_reliability"]["identity_failures"]

    S16 = rl.get("stations_reliability_16", {}).get("by_lead", {})

    def r16(h, key, gn):
        x = S16.get(str(h), {}).get(gn, {}).get(key)
        return x if (x is not None and "ci" in x) else None
    ALL16R = bool(r16(1, "ss|inf|0", A))

    def RV(h, k, form=None, j=None, gn=A, v=0, src="auto"):
        """numeric value: as run (four members) or the M=8 / inf form; sixteen members at leads 1 and 3 when all start days have them"""
        if form is None:
            e = S[str(h)][gn][k]["as_run"]
        else:
            key = f"{k}|{form}" + ("" if k.startswith("p_") else f"|{v}")
            e = (r16(h, key, gn) if (src == "auto" and h in (1, 3)) else None) or S[str(h)][gn][key]
        return e["v"] if j is None else e["v"][j]

    def R(h, k, form=None, j=None, gn=A, d=2, neg=False, v=0, src="auto"):
        if form is not None and src == "auto" and h in (1, 3):
            key = f"{k}|{form}" + ("" if k.startswith("p_") else f"|{v}")
            x = r16(h, key, gn)
            if x is not None:
                val = x["v"] if j is None else x["v"][j]
                ci = x["ci"] if j is None else [x["ci"][0][j], x["ci"][1][j]]
                return vci(val, ci, d, neg) + "*"
        e = S[str(h)][gn][k] if form is None else S[str(h)][gn][f"{k}|{form}" + ("" if k.startswith("p_") else f"|{v}")]
        e = e["as_run"] if form is None else e
        val = e["v"] if j is None else e["v"][j]
        ci = e.get("ci", e.get("ci_boot"))
        ci = None if ci is None else ([ci[0], ci[1]] if j is None else [ci[0][j], ci[1][j]])
        return vci(val, ci, d, neg)

    def besideR(items):
        if not ALL16R:
            return ""
        return "four members beside: " + "; ".join(f"{lab} M=8 {fn('8')}, inf {fn('inf')}" for lab, fn in items)

    def prov16(h, key, j=None, d=3, v=None, neg=False):
        """the provisional sixteen-member value of the 30 seen days (point and jackknife sd), when all days are not yet in"""
        if ALL16R:
            return ""
        x = S16.get(str(h), {}).get(A, {}).get(key)
        if x is None:
            return ""
        val = x["v"] if j is None else x["v"][j]; sd = x["jk16_sd"] if j is None else x["jk16_sd"][j]
        val = -val if neg else val
        return f"16 members ({x['days']} forecasts at this lead, provisional): {val:.{d}f} (jackknife sd {sd:.{d}f})"
    sec = ("2 Stations, reliability analysis (analyses/reliability). As run = the four stored members, interval widened for their draw (fix 9). "
           + ("M=8 / inf at leads 1 and 3: sixteen members on all start days (marked *; the 57 stations without them keep their four), four members beside; other leads: four members"
              if ALL16R else "M=8 / inf: four stored members; at leads 1 and 3 they are provisional"))
    row(sec, "RMSE on the station-days of the probabilistic scores, 1 day ahead (ug m-3); CAMS 15.42, Aurora 15.90", loc(("methods", r"RMSE 12\.96, 15\.42 and 15\.90")),
        "12.96", R(1, "p_rmse"), R(1, "p_rmse", "8"), R(1, "p_rmse", "inf"), "; ".join(x for x in ("the same station-days give gains against CAMS of " +
        f"{-RV(1, 'p_rmse_vs_cams'):.1f} / {-RV(1, 'p_rmse_vs_cams', '8'):.1f} / {-RV(1, 'p_rmse_vs_cams', 'inf'):.1f} % (as run / 8 / inf)",
        besideR([("", lambda f: R(1, "p_rmse", f, src="4"))]), prov16(1, "p_rmse|inf", d=2)) if x), idok)
    for h, pr in ((1, "32 (30-34)"), (7, "17 (14-20)")):
        row(sec, f"CRPS reduction against the climatological ensemble, {h} day(s) ahead (%)", loc(("main", r"its CRPS is 32\\% \(30--34")),
            pr, R(h, "p_crps_vs_climens", d=1, neg=True), R(h, "p_crps_vs_climens", "8", d=1, neg=True), R(h, "p_crps_vs_climens", "inf", d=1, neg=True),
            "; ".join(x for x in ("fair CRPS = infinite value (the as-run fair CRPS is the four-member estimate of it); M=8 = expected CRPS of eight members (pair term x 7/8) against the fair climatological ensemble",
                                  besideR([("", lambda f, hh=h: R(hh, "p_crps_vs_climens", f, d=1, neg=True, src="4"))]) if h == 1 else "", prov16(h, "p_crps_vs_climens|inf", d=1, neg=True) if h == 1 else "") if x), idok)
    row(sec, "Spread/skill against the measurements, 1 day ahead", loc(("main", r"ratio 0\.64--0\.71"), ("SI", r"the ratio is 0\.64 \(0\.58--0\.71\)")),
        "0.64 (0.58-0.71)", R(1, "ss_0"), R(1, "ss", "8"), R(1, "ss", "inf"),
        "; ".join(x for x in ("as run and M=8: Fortin form sqrt((M+1)/M); inf: sqrt(mean s2)/RMSE_inf", besideR([("", lambda f: R(1, "ss", f, src="4"))]), prov16(1, "ss|inf|0")) if x), idok)
    rngR = lambda k, form, hs, v: f"{min(RV(h, k, form, v=v) for h in hs):.2f}-{max(RV(h, k, form, v=v) for h in hs):.2f}"
    row(sec, "Spread/skill against the measurements, 2-7 days ahead (range)", loc(("SI", r"0\.68--0\.71 from 2 to 7 days")), "0.68-0.71",
        f"{min(RV(h, 'ss_0') for h in range(2, 8)):.2f}-{max(RV(h, 'ss_0') for h in range(2, 8)):.2f}", rngR("ss", "8", range(2, 8), 0), rngR("ss", "inf", range(2, 8), 0),
        "; ".join(x for x in ("lead 3 sixteen members, the others four" if ALL16R else "", "lead 3: " + R(3, "ss", "inf") + " (inf)",
                              besideR([("lead 3", lambda f: R(3, "ss", f, src="4"))])) if x), idok)
    row(sec, "Spread/skill against the estimates at the station pixels, 1-7 days (range)", loc(("SI", r"Against GHAP at the same pixels and days the ratio is 0\.82--0\.85")), "0.82-0.85",
        f"{min(RV(h, 'ss_1') for h in range(1, 8)):.2f}-{max(RV(h, 'ss_1') for h in range(1, 8)):.2f}", rngR("ss", "8", range(1, 8), 1), rngR("ss", "inf", range(1, 8), 1),
        "; ".join(x for x in ("leads 1 and 3 sixteen members, the others four" if ALL16R else "", besideR([(f"lead {h}", (lambda hh: (lambda f: R(hh, "ss", f, v=1, src="4")))(h)) for h in (1, 3)])) if x), idok)
    NN = ["China (CNEMC)", "Korea (AirKorea)", "Japan (NIES)", "USA (EPA)", "India (CPCB)", "Europe (EEA)"]
    row(sec, "Spread/skill against the measurements by network, 1 day ahead: China, Korea, Japan, USA, India, Europe", loc(("SI", r"1 day ahead the ratio against the measurements is 0\.82")),
        "0.82 / 1.11 / 0.98 / 0.61 / 0.49 / 0.37", " / ".join(f"{RV(1, 'ss_0', gn=n):.2f}" for n in NN), " / ".join(f"{RV(1, 'ss', '8', gn=n):.2f}" for n in NN),
        " / ".join(f"{RV(1, 'ss', 'inf', gn=n):.2f}" for n in NN),
        ("sixteen members; four members beside: inf " + " / ".join(f"{RV(1, 'ss', 'inf', gn=n, src='4'):.2f}" for n in NN)) if ALL16R else "", idok)
    for j, T, pr in ((1, "37.5", "0.27 (0.24-0.30) / 0.15 (0.12-0.19) / 0.07 (0.03-0.11)"), (0, "15", "0.26 / 0.17 / 0.08"), (2, "75", "0.15 (0.06-0.19) / 0.02 / -0.09; fair 0.26 / 0.17 / 0.07")):
        row(sec, f"Brier skill score against the local climatological frequency, > {T} ug m-3, 1 / 3 / 7 days", loc(("main", r"the Brier skill score against the local climatological frequency is 0\.27"), ("SI", r"Brier skill scores at the stations against the local climatological frequency")),
            pr, " / ".join(R(h, "bss_clim_0", j=j) for h in (1, 3, 7)), " / ".join(R(h, "bss_clim", "8", j=j) for h in (1, 3, 7)), " / ".join(R(h, "bss_clim", "inf", j=j) for h in (1, 3, 7)),
            "; ".join(x for x in ("as run: raw Brier of the four members against the raw climatological frequency; inf: fair Brier of both (= the 'fair' BSS of the SI for 75); M=8: Ferro et al. 2008 against the fair climatological ensemble",
                                  besideR([(f"lead {h}", (lambda hh: (lambda f: R(hh, "bss_clim", f, j=j, src="4")))(h)) for h in (1, 3)]), prov16(1, "bss_clim|inf|0", j=j)) if x), idok)
    row(sec, "RMSE of the ensemble mean on the station-days of the reliability note, 1 / 3 days (ug m-3)", loc(("SI", r"\(ensemble mean, 13\.1\)")), "13.1 / 14.3",
        R(1, "rmse_0") + " / " + R(3, "rmse_0"), R(1, "rmse", "8") + " / " + R(3, "rmse", "8"), R(1, "rmse", "inf") + " / " + R(3, "rmse", "inf"),
        besideR([(f"lead {h}", (lambda hh: (lambda f: R(hh, "rmse", f, src="4")))(h)) for h in (1, 3)]), idok)
    dr = rl.get("stations_reliability_16", {}).get("dropped", {})
    if dr:
        row(sec, "Sixteen-member station sums: fallback and dropped station-days (fix 17)", "-", "-",
            "; ".join(f"lead {h}: {v['forecasts']} forecasts, {v['fallback_station_days']} station-days with the four stored members (the 57 stations), {v['unexpected']} unexpected" for h, v in dr.items()), "", "", "", None)
    P_ = rl["populated_pixels"]["by_lead"]
    sec = "3 Against the 1 km estimates (four stored members unless stated)"
    for h, pr in ((1, "1.05 (0.99-1.09)"), (3, "1.07 (1.00-1.12)")):
        e = P_[str(h)]["pixels"]
        k2 = rl["populated_pixels"]["k2_populated_pixels"].get(str(h), {}).get("pixels")
        n16 = (f"16 members ({k2['days']} days{'' if k2['days'] >= 170 else ', provisional'}): inf " +
               (vci(k2['boot16_ssr_inf']['v'], k2['boot16_ssr_inf']['ci'], 3) if 'boot16_ssr_inf' in k2 else f"{k2['ssr_inf']['all16']:.3f}") +
               ", Fortin-8 " + (vci(k2['boot16_ssr_fortin_8']['v'], k2['boot16_ssr_fortin_8']['ci'], 3) if 'boot16_ssr_fortin_8' in k2 else f"{k2['ssr_fortin_8']['all16']:.3f}") +
               f"; stored 4 on the same pixel-days: inf {k2['stored4_same_pixels']['ssr_inf']:.3f}") if k2 else ""
        if k2 and "boot16_ssr_inf" in k2:
            row(sec, f"Spread/skill at the 8.05 M populated pixels, {h} day(s) ahead [M=8 / inf: sixteen members, 179 start days]", loc(("SI", r"it is 1\.05 \(95\\% interval, 0\.99--1\.09\)")), pr,
                vci(e["ss_as_run"]["v"], e["ss_as_run"]["ci_boot"], 2), vci(k2["boot16_ssr_fortin_8"]["v"], k2["boot16_ssr_fortin_8"]["ci"], 2), vci(k2["boot16_ssr_inf"]["v"], k2["boot16_ssr_inf"]["ci"], 2),
                f"four members beside: M=8 {vci(e['ss_8']['v'], e['ss_8']['ci'], 2)}, inf {vci(e['ss_inf']['v'], e['ss_inf']['ci'], 2)}", e["ss_as_run"]["identical_to_paper_json"])
        else:
            row(sec, f"Spread/skill at the 8.05 M populated pixels, {h} day(s) ahead", loc(("SI", r"it is 1\.05 \(95\\% interval, 0\.99--1\.09\)")), pr,
                vci(e["ss_as_run"]["v"], e["ss_as_run"]["ci_boot"], 2), vci(e["ss_8"]["v"], e["ss_8"]["ci"], 2), vci(e["ss_inf"]["v"], e["ss_inf"]["ci"], 2), n16, e["ss_as_run"]["identical_to_paper_json"])
    dp = rl["populated_pixels"].get("dropped_16")
    if dp:
        row(sec, "Sixteen-member populated pixels: pixel-days with the four stored members finite but not the sixteen (fix 17)", "-", "-",
            f"{dp['pixel_days_4_not_16']:.0f} pixel-days, in {dp['forecasts_with_drops']} of {dp['forecasts']} forecasts", "", "", dp.get("note", ""), None)
    L = rl["land_recs"]["2022 final"]["leads"]
    rng_ = lambda form: f"{min(L[str(h)]['Nine regions pooled'][form]['v'] for h in range(1, 8)):.2f}-{max(L[str(h)]['Nine regions pooled'][form]['v'] for h in range(1, 8)):.2f}"
    row(sec, "Spread/skill over the land of the nine regions pooled, 1-7 days (range)", loc(("main", r"ratio 0\.96--0\.99 over the land"), ("methods", r"on 2022 the ratio was 0\.96--0\.99"), ("SI", r"spread--skill ratio of the four members in 2022 is 0\.96--0\.99")),
        "0.96-0.99", rng_("ss_as_run"), rng_("ss_8"), rng_("ss_inf"), "lead 1: " + vci(L["1"]["Nine regions pooled"]["ss_inf"]["v"], L["1"]["Nine regions pooled"]["ss_inf"].get("ci"), 3) + " (inf; V_draw: CV of the populated pixels carried)",
        all(L[str(h)]["Nine regions pooled"]["ss_as_run"].get("identical_to_paper_json") for h in range(1, 8)))
    regs = [r for r in L["1"] if r != "Nine regions pooled"]
    mn = {f: min((L[str(h)][r][f]["v"], r, h) for h in range(1, 8) for r in regs) for f in ("ss_as_run", "ss_8", "ss_inf")}
    mx = {f: max((L[str(h)][r][f]["v"], r, h) for h in range(1, 8) for r in regs) for f in ("ss_as_run", "ss_8", "ss_inf")}
    row(sec, "Spread/skill by region over land, extremes (region, lead)", loc(("SI", r"by region it ranges from 0\.70 \(Oceania")), "0.70 (Oceania, 1 day) .. 1.14",
        f"{mn['ss_as_run'][0]:.2f} ({mn['ss_as_run'][1]}, {mn['ss_as_run'][2]}) .. {mx['ss_as_run'][0]:.2f} ({mx['ss_as_run'][1]}, {mx['ss_as_run'][2]})",
        f"{mn['ss_8'][0]:.2f} .. {mx['ss_8'][0]:.2f}", f"{mn['ss_inf'][0]:.2f} ({mn['ss_inf'][1]}, {mn['ss_inf'][2]}) .. {mx['ss_inf'][0]:.2f} ({mx['ss_inf'][1]}, {mx['ss_inf'][2]})", "", None)
    L21 = rl["land_recs"].get("2021 final (validation year)", {}).get("leads", {}); B21 = rl["land_recs"].get("2021 before CRPS training", {}).get("leads", {})
    if L21:
        row(sec, "Spread/skill on 2021 (validation) after the calibration, 1 / 7 days; before it 1 / 7 days", loc(("methods", r"0\.92 one day ahead and 0\.82 after a week"), ("methods", r"ratio of four members was 0\.51")),
            "0.92 / 0.82; before 0.51 / 0.43", " / ".join(f"{L21[str(h)]['Nine regions pooled']['ss_as_run']['v']:.2f}" for h in (1, 7)) + "; " + " / ".join(f"{B21[str(h)]['Nine regions pooled']['ss_as_run']['v']:.2f}" for h in (1, 7)),
            " / ".join(f"{L21[str(h)]['Nine regions pooled']['ss_8']['v']:.2f}" for h in (1, 7)) + "; " + " / ".join(f"{B21[str(h)]['Nine regions pooled']['ss_8']['v']:.2f}" for h in (1, 7)),
            " / ".join(f"{L21[str(h)]['Nine regions pooled']['ss_inf']['v']:.2f}" for h in (1, 7)) + "; " + " / ".join(f"{B21[str(h)]['Nine regions pooled']['ss_inf']['v']:.2f}" for h in (1, 7)),
            "selection-related (the calibration epoch was chosen on the 2021 CRPS, not on this ratio); 2021 is not a test score", None)
        r1a, r1b = L21["1"]["Nine regions pooled"], B21["1"]["Nine regions pooled"]
        K21 = rl["land_recs"].get("2021 before CRPS training, 4 Euler steps", {}).get("leads", {})
        r1c = K21["1"]["Nine regions pooled"] if K21 else None
        chg = lambda a, b: " / ".join(f"{100 * (a[k] / b[k] - 1):+.1f}" for k in ("rmse_as_run", "rmse_8", "rmse_inf"))
        w43 = loc(("methods", r"for a 2\\% increase in the error of the ensemble mean"))
        row(sec, "2021, 1 day ahead: change of the RMSE of the ensemble mean with the calibration (%)", w43, "+2",
            f"{100 * (r1a['rmse_as_run'] / r1b['rmse_as_run'] - 1):+.1f}", f"{100 * (r1a['rmse_8'] / r1b['rmse_8'] - 1):+.1f}", f"{100 * (r1a['rmse_inf'] / r1b['rmse_inf'] - 1):+.1f}",
            "recs of eval2021_sel_crps_e01 vs eval2021_sel_e02 (both four members); the sign changes between forms (A14 section 3: stated with 4, 8 and infinite values); "
            "the 12 % CRPS change is fair (= infinite); identity None: the change itself has no stored reference (fix 14)" +
            (f"; equal-step sensitivity (the model before calibration also with 4 Euler steps, eval2021_sel_e02_k4): {chg(r1a, r1c)} (as run / M=8 / inf)" if r1c else ""), None)
        CONC.append(f"{w43}: 2021 calibration, RMSE of the ensemble mean 1 day ahead (four members): as run {chg(r1a, r1b).split(' / ')[0]} %, M = 8 {chg(r1a, r1b).split(' / ')[1]} %, "
                    f"infinite {chg(r1a, r1b).split(' / ')[2]} %" + (f" (with equal Euler steps {chg(r1a, r1c)} %)" if r1c else "") + "; 'a 2 % increase in the error of the ensemble mean' holds as run only")
    # emissions and transport (fix 4: the paired four-field draw in all three forms; fix 15: the 0.25 deg cell means)
    EM = rl["emissions_transport_km1"]; EPD = rl.get("emissions_pair_draw", {})
    sec = ("4 Emissions and transport (recs of the four models; four members each; reductions in %). Intervals: the paper's block bootstrap, widened in all three forms "
           "by the PAIRED four-field jackknife of the same gain at the 8,110 station pixels against GHAP (field i removed from both models, which share their noise fields), "
           "carried to land as V_draw/V_boot (fix 4; an approximation: the exact land jackknife needs per-member land sums); the bootstrap-only intervals (identity) are in rel_classU.json")
    FORMS3 = ("as_run", "M8", "inf")

    def ege(comp, reg, h, form):
        return EM[comp][reg][str(h)][f"rmse_{form}"]

    def eci(comp, reg, h, form):
        e = ege(comp, reg, h, form)
        return e.get("ci", e["ci_boot"])

    def eg(comp, reg, h, form):
        return vci(ege(comp, reg, h, form)["v"], eci(comp, reg, h, form), 1)

    def zero_in(comp, reg, hs, form):
        return [h for h in hs if eci(comp, reg, h, form)[0] <= 0 <= eci(comp, reg, h, form)[1]]

    def shared(comp):
        e = EPD.get(comp)
        if not e:
            return "field sharing not checked"
        c = e["departure_corr_same_m"]
        return f"the two models share their four fields (correlation of the departures of the same member {min(c):.3f}-{max(c):.3f})"

    def ratios(comp, form):
        v = [ege(comp, "All regions", h, form).get("v_draw_ratio") for h in range(1, 8)]
        return "n/a" if None in v else f"{min(v):.2f}-{max(v):.2f}"
    z57 = {f: zero_in("emissions", "All regions", (5, 6, 7), f) for f in FORMS3}
    st57 = ("from 5 days ahead every interval includes zero in all three forms (as the paper states as run)" if all(len(v) == 3 for v in z57.values())
            else "5-7 days, leads whose interval includes zero: " + "; ".join(f"{f} {v}" for f, v in z57.items()))
    row(sec, "Emission gain in the RMSE, all regions, 1 / 4 / 5 / 6 / 7 days", loc(("SI", r"the RMSE of the ensemble mean falls by 2\.3\\%"), ("main", r"the RMSE by about 2\\% in the first 4 days only")),
        "2.3 (1.4-3.0) / 1.7 (0.2-3.0) / 0.1 / 0.9 / -0.6", " / ".join(eg("emissions", "All regions", h, "as_run") for h in (1, 4, 5, 6, 7)),
        " / ".join(eg("emissions", "All regions", h, "M8") for h in (1, 4, 5, 6, 7)), " / ".join(eg("emissions", "All regions", h, "inf") for h in (1, 4, 5, 6, 7)),
        f"{shared('emissions')}; V_jack/V_boot of the same gain at the stations {ratios('emissions', 'inf')} (inf), {ratios('emissions', 'as_run')} (as run); {st57}",
        all(EM["emissions"]["All regions"][str(h)]["rmse_as_run"]["identical_to_paper_csv"] for h in range(1, 8)))
    b6 = ege("emissions", "All regions", 6, "inf")["ci_boot"]
    CHK.append(f"emission gain over all regions ({loc(('SI', r'its interval includes zero from 5 days ahead'))}): {st57}; 6 days as run {eg('emissions', 'All regions', 6, 'as_run')}, "
               f"M = 8 {eg('emissions', 'All regions', 6, 'M8')}, infinite {eg('emissions', 'All regions', 6, 'inf')} (the infinite interval of the bootstrap alone, "
               f"{b6[0]:.1f} to {b6[1]:.1f}, was the first version's 'conclusion that changes'; with the paired draw of fix 4 it includes zero)")
    for reg, pr, wh in (("East Asia", "4.1-5.5", loc(("main", r"In East Asia, they lower the RMSE"))), ("South Asia", "2.4-2.9", loc(("main", r"in South Asia by 2\.4--2\.9"))),
                        ("Europe", "about 3", loc(("main", r"in Europe by approximately 3"))), ("North America", "2-4 at 1 day, < 2 from 5", loc(("main", r"In North America, South America and Southeast Asia"))),
                        ("South America", "2-4 at 1 day, < 2 from 5", loc(("main", r"In North America, South America and Southeast Asia"))),
                        ("Southeast Asia", "3.7 at 1 day, <= 1.6 from 5", loc(("SI", r"it is 3\.7\\% 1 day ahead in Southeast Asia"))), ("Africa", "2 at 1 day", loc(("main", r"there, the gain is 2\\% at 1 day"))),
                        ("Middle East", "1-2 at every lead", loc(("SI", r"1--2\\% at every lead time in the Middle East"))), ("Oceania", "1.9 at 1 day, -2 from 5", loc(("SI", r"1\.9\\% 1 day ahead in Oceania")))):
        row(sec, f"Emission gain in the RMSE, {reg}, leads 1-7", wh, pr, " / ".join(eg("emissions", reg, h, "as_run") for h in range(1, 8)),
            " / ".join(eg("emissions", reg, h, "M8") for h in range(1, 8)), " / ".join(eg("emissions", reg, h, "inf") for h in range(1, 8)),
            "draw: the all-station paired ratio of the same lead carried to the region; leads whose interval includes zero: " +
            "; ".join(f"{f} {zero_in('emissions', reg, range(1, 8), f) or 'none'}" for f in FORMS3),
            all(EM["emissions"][reg][str(h)]["rmse_as_run"]["identical_to_paper_csv"] for h in range(1, 8)))
    ss = lambda reg, nm, f: EM["emissions"][reg]["7"][f"ss_{nm}"][f]
    row(sec, "Spread/skill after a week, model with emissions / control (all regions; East Asia; Africa)", loc(("SI", r"the spread--skill ratio is 0\.52 for the model with emissions")),
        "0.52 / 0.40; 0.50 / 0.42; 0.52 / 0.38", "; ".join(f"{ss(r, 'model', 'as_run'):.2f} / {ss(r, 'reference', 'as_run'):.2f}" for r in ("All regions", "East Asia", "Africa")),
        "; ".join(f"{ss(r, 'model', 'M8'):.2f} / {ss(r, 'reference', 'M8'):.2f}" for r in ("All regions", "East Asia", "Africa")),
        "; ".join(f"{ss(r, 'model', 'inf'):.2f} / {ss(r, 'reference', 'inf'):.2f}" for r in ("All regions", "East Asia", "Africa")), "", None)
    row(sec, "Transport gain in the RMSE (base model against the network without transport), all regions, 1 / 3 / 7 days", loc(("SI", r"Gain in the accuracy of the ensemble mean")),
        "Supplementary Fig. emislead a-f", " / ".join(eg("transport", "All regions", h, "as_run") for h in (1, 3, 7)), " / ".join(eg("transport", "All regions", h, "M8") for h in (1, 3, 7)),
        " / ".join(eg("transport", "All regions", h, "inf") for h in (1, 3, 7)),
        f"{shared('transport')}; paired V_jack/V_boot at the stations {ratios('transport', 'inf')} (inf); leads 1-7 whose interval includes zero: " +
        "; ".join(f"{f} {zero_in('transport', 'All regions', range(1, 8), f) or 'none'}" for f in FORMS3),
        all(EM["transport"]["All regions"][str(h)]["rmse_as_run"]["identical_to_paper_csv"] for h in range(1, 8)))
    row(sec, "CRPS gains: emissions 4.7 % (1 day), 2.0 % (7 days); transport about 6 % (1 day)", loc(("main", r"the emissions lower the CRPS by 4\.7"), ("main", r"it lowers the CRPS by approximately 6")),
        "4.7 / 2.0; 6", f"{EM['emissions']['All regions']['1']['crps_as_run']['v']:.1f} / {EM['emissions']['All regions']['7']['crps_as_run']['v']:.1f}; {EM['transport']['All regions']['1']['crps_as_run']['v']:.1f}",
        "not computable (the recs store the fair CRPS but not its pair term)", "= as run (fair CRPS)", "", None)
    C25 = rl.get("emissions_c25", {}).get("by_box", {}).get("emissions", {})
    if C25:
        def c25v(nm, f, lk="1-7"):
            e = C25[nm][lk][f]
            return vci(e["v"], e.get("ci", e["ci_boot"]), 1)
        for nm, pr, wh in (("North China Plain", "5.6 (1.8-8.8)", loc(("main", r"the gain over the week is 5\.6\\%"), ("SI", r"the gain over the seven lead times is 5\.6"))),
                           ("Indo-Gangetic Plain", "3.0 (0.4-5.5)", loc(("main", r"the gain over the week is 5\.6\\%"), ("SI", r"the gain over the seven lead times is 5\.6"))),
                           ("East Asia", "2.6 (significant)", loc(("SI", r"also significant in East Asia \(2\.6"))),
                           ("South Asia", "2.5 (significant)", loc(("SI", r"also significant in East Asia \(2\.6")))):
            if nm not in C25:
                continue
            row(sec, f"Emission gain on 0.25 deg cell means over the seven lead times, {nm} (%)", wh, pr, c25v(nm, "as_run"),
                c25v(nm, "M8") + " [member 0 only]", c25v(nm, "inf") + " [member 0 only]",
                "recompute.py c25 replayed (masks, boxes, pooling of leads 1-7, block bootstrap); M=8 / inf: each model's own member-0 term (4/3)(m0 - xbar)^2 of its 0.25 deg means "
                "(member 0 only); V_draw: the paired station ratio of the pooled leads carried (fix 4 caveat) + the single-member draw of the member-0 term (fix 11; both models scaled "
                "alike, member 0 being the same field in both)", C25[nm]["1-7"]["as_run"].get("identical_to_paper_csv"))
        oth = [n for n in ("Europe", "North America", "South America", "Africa", "Middle East", "Southeast Asia", "Oceania") if n in C25]
        row(sec, "Emission gain on 0.25 deg cell means over the seven lead times, other regions: " + ", ".join(oth) + " (%)", loc(("SI", r"It is 0\.6\\% or less in the Middle East")),
            "<= 0.6 (Middle East, Africa, Americas); Europe -2.1, Southeast Asia -5.2, Oceania -2.6",
            " / ".join(f_(C25[n]["1-7"]["as_run"]["v"], 1) for n in oth), " / ".join(f_(C25[n]["1-7"]["M8"]["v"], 1) for n in oth),
            " / ".join(f_(C25[n]["1-7"]["inf"]["v"], 1) for n in oth), "order as in the quantity; member 0 only for M=8 / inf; intervals in rel_classU.json (emissions_c25)",
            all(C25[n]["1-7"]["as_run"].get("identical_to_paper_csv") for n in oth))

# ============================================================================ 3b. land 0.25 deg
ld, _ = J("land25_classU.json")
if ld:
    Lg = ld["leads"]["Global (84N-60S)"]
    sec = "3 Against the 1 km estimates (four stored members unless stated)"
    J_M0 = json.loads(((RES / "j_m0.json") if AR else (P / "a14" / "classU_review" / "judge" / "j_m0.json")).read_text())["ratios"]   # the reference j_m0.py; under 'ar' j_m0_design.py on the new run (A16 Addendum 2 point 2)
    LD = lambda h: "lead1" if h == 1 else "lead3"                                                        # noqa: E731
    RDRAW = {h: 1 / np.array(J_M0[f"{LD(h)}_all"]["ratio_by_member"]) for h in range(1, 8)}               # the four members' ratios: single-member draw
    RPROX = {h: 1 / np.array([J_M0[f"{LD(h)}_{c}"]["ratio_by_member"][0] for c in ("all", "ge100", "ge400")]) for h in range(1, 8)}   # member 0, three cell sets

    def m0_corr(h, c, S):
        """pooled correlation with the member-0 term scaled by s: r_as / sqrt(1 - c s u/var), u/var from the stored as-run and inf values"""
        ra, ri = Lg[str(h)]["r"]["as_run"], Lg[str(h)]["r"]["inf"]
        return ra / np.sqrt(1 - c * S[h] * (1 - (ra / ri) ** 2))

    def m0_gain(h, c, S, ref="climatology"):
        e = Lg[str(h)]; ma = e["rmse"]["as_run"] ** 2; qn = ma - e["rmse"]["inf"] ** 2
        return 100 * (1 - np.sqrt(ma - c * S[h] * qn) / e["references"][ref]["rmse"])

    def m0_lab(v, fn, h, c, d):
        px, dr = fn(h, c, RPROX), fn(h, c, RDRAW)
        if AR:                                                             # A16 4.3: the noise is renewed at every start, V_draw = 0
            return f"{f_(v, d)} [member 0 only; proxy {f_(min(px.min(), v), d)} to {f_(max(px.max(), v), d)} (ratios of the new run); draw sd 0 by design]"
        return f"{f_(v, d)} [member 0 only; proxy {f_(min(px.min(), v), d)} to {f_(max(px.max(), v), d)}; draw sd {f_(dr.std(), d + 1)}]"
    row(sec, "Correlation with the estimates on 0.25 deg cell means, 1 / 4 / 7 days (pooled)", loc(("main", r"correlation with the 1\\,km estimates is 0\.89"), ("main", r"correlation 0\.76 against 0\.75")),
        "0.89 / 0.80 / 0.76", " / ".join(f"{Lg[str(h)]['r']['as_run']:.3f}" for h in (1, 4, 7)), " / ".join(m0_lab(Lg[str(h)]["r"]["M8"], m0_corr, h, 0.5, 3) for h in (1, 4, 7)),
        " / ".join(m0_lab(Lg[str(h)]["r"]["inf"], m0_corr, h, 1.0, 3) for h in (1, 4, 7)),
        "pooled correlation, one centering; var_inf(xbar) = var(xbar) - var(m0 - xbar)/3 (member 0 only, fix 11); proxy = the range of the value with the member-0 term divided by "
        "member 0's ratio (4/3)u0^2/s2 over all populated cells, cells with >= 100 and with >= 400 populated pixels (reference j_m0.json; lead 1 for 1 day, lead 3 carried to 4 and 7 days); "
        "draw sd = sd (ddof 0) of the value over the four members' ratios (all cells), the single-member draw (V_draw); no interval in the paper; no one-sided correction applied; "
        "climatology 0.749-0.750", ld["identity_vs_paper_npz"]["S"] and ld["replay_identity"]["S"])
    gl = lambda h, f: -Lg[str(h)]["references"]["climatology"]["gain_pct"][f]
    row(sec, "RMSE of the 0.25 deg cell means below the climatology, 1 / 7 days (%)", loc(("main", r"a third below that of each place's climatology")),
        "a third / 3", f"{gl(1, 'as_run'):.1f} / {gl(7, 'as_run'):.1f}", " / ".join(m0_lab(gl(h, "M8"), m0_gain, h, 0.5, 1) for h in (1, 7)),
        " / ".join(m0_lab(gl(h, "inf"), m0_gain, h, 1.0, 1) for h in (1, 7)),
        "no interval in the paper; member 0 only (proxy range and single-member draw as in the row above, fix 11); main_text:138 'after a week, it is 3 % below it' is the four-member value",
        ld["identity_vs_paper_npz"]["S"])
    CONC.append(f"{loc(('main', r'a third below that of each place.s climatology'))}: RMSE of the 0.25 deg cell means against each place's climatology: one day ahead "
                f"{gl(1, 'as_run'):.1f} % below as run ('a third'), {m0_lab(gl(1, 'inf'), m0_gain, 1, 1.0, 1)} infinite; after a week {gl(7, 'as_run'):.1f} % as run ('3 % below it'), "
                f"{m0_lab(gl(7, 'M8'), m0_gain, 7, 0.5, 1)} at M = 8, {m0_lab(gl(7, 'inf'), m0_gain, 7, 1.0, 1)} infinite; correlation after a week "
                f"{Lg['7']['r']['as_run']:.3f} / {Lg['7']['r']['M8']:.3f} / {Lg['7']['r']['inf']:.3f} (as run / M = 8 / infinite, member 0 only) against 0.75 for the climatology")

# ============================================================================ 5. Foehn-CAMS, Aurora protocol
c12, _ = J("cams12h_classU.json")


def below_from(g_, L_, key):
    """first lead from which every later interval lies below zero (Foehn-CAMS better)"""
    e = g_[key]
    for k in range(len(L_)):
        if all(e["ci"][1][j] < 0 for j in range(k, len(L_))):
            return f"{L_[k]} h"
    return "never"


def at_h(g_, L_, key, h, d=1):
    e = g_[key]; k = L_.index(h)
    return vci(e["v"][k], [e["ci"][0][k], e["ci"][1][k]], d)


if c12:
    gg = c12["by_region"]["Global"]; LH = c12["leads_h"]
    sec = "5 Foehn-CAMS in Aurora's protocol (blind; eight members as run, fresh noise each initialisation: V_draw = 0)"
    hrs = [12, 24, 36, 72, 120]
    ix = [LH.index(h) for h in hrs]
    for nm, pr, wh in (("Aurora", "+3 / 0 / -5 .. -12; 3 days -9, 5 days -12", loc(("main", r"by 9\\% at 3 days and 12\\% at 5 days"), ("methods", r"its RMSE is 3\\% above that of Aurora at 12"))),
                       ("CAMS forecast", "+46 / +11 / ...; -18 at 5 days", loc(("main", r"by 18\\% at 5 days"), ("methods", r"it is 46\\% above at 12"), ("main", r"In the first day and a half, the CAMS forecast")))):
        a_ = gg[f"change_pct_vs_{nm}_as_run"]; i_ = gg[f"change_pct_vs_{nm}_inf"]
        ka, ki = f"change_pct_vs_{nm}_as_run", f"change_pct_vs_{nm}_inf"
        side = lambda key, h: ("the reference better" if gg[key]["ci"][0][LH.index(h)] > 0 else ("Foehn-CAMS better" if gg[key]["ci"][1][LH.index(h)] < 0 else "equal, interval includes zero"))
        if nm == "Aurora":
            nt = (f"below Aurora from {below_from(gg, LH, ka)} as run and from {below_from(gg, LH, ki)} in the infinite form (12 h inf {at_h(gg, LH, ki, 12)}: {side(ki, 12)}); "
                  "methods:132 'equal to it at 24 h and 5-12 % below it from 36 h' differs between forms (A14 section 3)")
            wAur = loc(("methods", r"its RMSE is 3\\% above that of Aurora at 12"))
            CONC.append(f"{wAur}: Foehn-CAMS against Aurora (eight members, fresh noise): as run {at_h(gg, LH, ka, 12)} at 12 h "
                        f"and {at_h(gg, LH, ka, 24)} at 24 h ('3 % above at 12 h, equal at 24 h'), below Aurora from {below_from(gg, LH, ka)}; infinite {at_h(gg, LH, ki, 12)} "
                        f"({side(ki, 12)}) and {at_h(gg, LH, ki, 24)} ({side(ki, 24)}): below Aurora from {below_from(gg, LH, ki)}")
        else:
            nt = (f"below the CAMS forecast from {below_from(gg, LH, ka)} as run and from {below_from(gg, LH, ki)} inf; 36 h: as run {at_h(gg, LH, ka, 36)} ({side(ka, 36)}), "
                  f"inf {at_h(gg, LH, ki, 36)} ({side(ki, 36)}): main_text:155 'in the first day and a half, the CAMS forecast is the most accurate of the three'")
            CONC.append(f"{loc(('main', r'In the first day and a half, the CAMS forecast'))}: 'in the first day and a half, the CAMS forecast is the most accurate of the three': at 36 h, "
                        f"Foehn-CAMS against the CAMS forecast as run {at_h(gg, LH, ka, 36)} ({side(ka, 36)}), infinite {at_h(gg, LH, ki, 36)} ({side(ki, 36)}); at 24 h infinite "
                        f"{at_h(gg, LH, ki, 24)} ({side(ki, 24)})")
        row(sec, f"Change of the latitude-weighted RMSE against {nm} at 12 / 24 / 36 / 72 / 120 h (%; negative = Foehn-CAMS better)", wh, pr,
            " / ".join(vci(a_["v"][k], [a_["ci"][0][k], a_["ci"][1][k]], 1) for k in ix), "= as run (eight members)",
            " / ".join(vci(i_["v"][k], [i_["ci"][0][k], i_["ci"][1][k]], 1) for k in ix), nt, c12["identity"]["rmse_per_init"])
    for lagf in ("cams12h_classU_cams12h_lag1b.json",):          # lag1 is a failed run (every per-initialisation RMSE inf); lag1b is the paper's
        lg, _ = J(lagf)
        if not lg:
            continue
        gl_ = lg["by_region"]["Global"]; LL = lg["leads_h"]
        if not np.isfinite(gl_["change_pct_vs_Aurora_as_run"]["v"][LL.index(120)]):
            row(sec, f"Second score, HRES run of the previous day ({lg['run']})", loc(("methods", r"scored once more with the HRES run of the previous day")),
                "Aurora -3 / -10; CAMS -3 / -15", "not reproduced", "", "",
                f"the stored per-initialisation RMSEs of {lg['reference_file']} are not finite for some initialisations (identity of the replay holds, inf included): the printed numbers came from another reduction; not used", False)
            continue
        HS = (24, 36, 48, 60, 120)
        lv = lambda ref, form: " / ".join(at_h(gl_, LL, f"change_pct_vs_{ref}_{form}", h) for h in HS)
        row(sec, f"Second score, HRES run of the previous day ({lg['run']}): change against Aurora; against the CAMS forecast, at 24 / 36 / 48 / 60 / 120 h (%)",
            loc(("methods", r"scored once more with the HRES run of the previous day")), "Aurora -3 (36 h) / -10 (120 h); CAMS -3 (60 h) / -15 (120 h)",
            "Aurora " + lv("Aurora", "as_run") + "; CAMS " + lv("CAMS forecast", "as_run"), "= as run (eight members)",
            "Aurora " + lv("Aurora", "inf") + "; CAMS " + lv("CAMS forecast", "inf"),
            f"reference file {lg['reference_file']}; V_draw = 0 (fresh noise); below Aurora from {below_from(gl_, LL, 'change_pct_vs_Aurora_as_run')} as run and from "
            f"{below_from(gl_, LL, 'change_pct_vs_Aurora_inf')} inf (24 h inf {at_h(gl_, LL, 'change_pct_vs_Aurora_inf', 24)}); below the CAMS forecast from "
            f"{below_from(gl_, LL, 'change_pct_vs_CAMS forecast_as_run')} as run and from {below_from(gl_, LL, 'change_pct_vs_CAMS forecast_inf')} inf (fix 6)",
            lg["identity"]["rmse_per_init"])
        CONC.append(f"{loc(('methods', r'scored once more with the HRES run of the previous day'))}: second score with the previous day's HRES run: below Aurora from "
                    f"{below_from(gl_, LL, 'change_pct_vs_Aurora_as_run')} as run and from {below_from(gl_, LL, 'change_pct_vs_Aurora_inf')} infinite (24 h infinite "
                    f"{at_h(gl_, LL, 'change_pct_vs_Aurora_inf', 24)}); below the CAMS forecast from {below_from(gl_, LL, 'change_pct_vs_CAMS forecast_as_run')} as run "
                    f"(48 h {at_h(gl_, LL, 'change_pct_vs_CAMS forecast_as_run', 48)}) and from {below_from(gl_, LL, 'change_pct_vs_CAMS forecast_inf')} infinite "
                    f"(48 h {at_h(gl_, LL, 'change_pct_vs_CAMS forecast_inf', 48)})")
    row(sec, "Spread/skill ratio, 12-120 h (range)", loc(("methods", r"Its spread--skill ratio is 0\.72--0\.81")), "0.72-0.81",
        f"{min(gg['spread_skill_as_run']):.2f}-{max(gg['spread_skill_as_run']):.2f}", "= as run", f"{min(gg['spread_skill_inf']):.2f}-{max(gg['spread_skill_inf']):.2f}", "", c12["identity"]["spread_skill_maxabs"] < 1e-9)

# ============================================================================ 6/7. chain 2022 and Foehn-CAMS at the stations
ch, _ = J("chain_classU.json")
if ch:
    C = ch["C"]; F = ch["F"]; Sx = ch["S"]; ALL = "All six networks"

    def cc(T, A_, B_, ln, form, key="rel_rmse"):
        e = T["contrasts"][f"{A_} - {B_}"]
        r = e["as_run"][ALL][ln] if form == "as_run" else e[form][ALL][ln]
        if r is None:
            return "n/a"
        ci = r.get("rel_rmse_ci_inflated") if key == "rel_rmse" and "rel_rmse_ci_inflated" in r else r.get(key + "_ci")
        return vci(100 * r[key], None if ci is None else [100 * ci[0], 100 * ci[1]], 1, neg=True)
    sec = "6 Foehn-CAMS read at the official stations (score_official.py 'coarse'; eight members, fresh noise: V_draw = 0)"
    row(sec, "RMSE reduction against the CAMS forecast (3-hourly mean), lead days 2-5 (%)", loc(("methods", r"2--6\\% below that of the CAMS forecast averaged"), ("main", r"2--6\\% below that of the CAMS forecast")),
        "2-6", " / ".join(cc(C, "coarse", "camsfc", str(L), "as_run") for L in range(2, 6)), "= as run", " / ".join(cc(C, "coarse", "camsfc", str(L), "inf") for L in range(2, 6)), "", all(C["identity"].values()))
    row(sec, "RMSE reduction against Aurora, lead days 2-5 (%)", loc(("methods", r"3--11\\% below that of Aurora")), "3-11",
        " / ".join(cc(C, "coarse", "aurora", str(L), "as_run") for L in range(2, 6)), "= as run", " / ".join(cc(C, "coarse", "aurora", str(L), "inf") for L in range(2, 6)), "", all(C["identity"].values()))
    ar = C["arms"]["coarse"]
    row(sec, "RMSE of Foehn-CAMS at the stations, lead days 2-5 (ug m-3)", loc(("methods", r"its RMSE from the second to the fifth day of the forecast is 14\.7--15\.1")), "14.7-15.1",
        " / ".join(f"{ar[f'{ALL}|{L}']['rmse_as_run']:.2f}" for L in range(2, 6)), "= as run", " / ".join(f"{ar[f'{ALL}|{L}']['rmse_inf']:.2f}" for L in range(2, 6)), "", all(C["identity"].values()))
    NW = ["Europe (EEA)", "China (CNEMC)", "USA (EPA)", "Korea (AirKorea)", "Japan (NIES)", "India (CPCB)", ALL]
    def ccg(T, A_, B_, ln, form, gn):
        e = T["contrasts"][f"{A_} - {B_}"]
        r = e["as_run"][gn][ln] if form == "as_run" else e[form][gn][ln]
        return "n/a" if r is None else f"{-100 * r['rel_rmse']:.1f}"
    for B_, nm in (("camsfc", "CAMS forecast"), ("aurora", "Aurora")):
        row(sec, f"By network (EEA, CNEMC, EPA, AirKorea, Japan, CPCB, six networks): RMSE reduction against {nm}, second / fifth day (%)", loc(("SI", r"Same design trained in Aurora's protocol")),
            "Supplementary Table away b", "; ".join(f"{ccg(C, 'coarse', B_, '2', 'as_run', g)} / {ccg(C, 'coarse', B_, '5', 'as_run', g)}" for g in NW), "= as run",
            "; ".join(f"{ccg(C, 'coarse', B_, '2', 'inf', g)} / {ccg(C, 'coarse', B_, '5', 'inf', g)}" for g in NW),
            "NOT the paper's station set: Supplementary Table away b comes from experiments/figures/eval_cams12h_stations.py (6,296 stations); these are score_official's 6,227 stations (values within 0.1-1.6 points of the printed ones); intervals in chain_classU.json", False)
    sec = "7 Two-stage system and downscaler at the official stations of 2022 (score_official.py; blind; eight members, fresh noise: V_draw = 0)"
    row(sec, "Two-stage RMSE reduction against the CAMS forecast, lead day 1 / all five (%)", loc(("main", r"two-stage system is 17\\% \(12--24")), "17 (12-24) / 21",
        cc(C, "chain", "camsfc", "1", "as_run") + " / " + cc(C, "chain", "camsfc", "all", "as_run"), "= as run", cc(C, "chain", "camsfc", "1", "inf") + " / " + cc(C, "chain", "camsfc", "all", "inf"), "", all(C["identity"].values()))
    for B_, pr, wh in (("coarse", "18.7 (14.7-24.4)", "against its 0.4 deg input"), ("camsfc", "21.2 (16.1-28.2)", "against the CAMS forecast"), ("aurora", "23.5", "against Aurora"),
                       ("clim_paper", "13.0 (10.1-15.5)", "against the climatology"), ("bcsd_fc", "4.6 (1.7-7.2)", "against the quantile mapping of the same members")):
        row(sec, f"Two-stage RMSE over the five lead days, reduction {wh} (%)", loc(("SI", r"the RMSE of the two-stage system over the five lead days is 12\.0")), pr,
            cc(C, "chain", B_, "all", "as_run"), "= as run", cc(C, "chain", B_, "all", "inf"), "", all(C["identity"].values()))
    row(sec, "With the CAMS forecast as coarse input: RMSE change against the two-stage system (%)", loc(("SI", r"the downscaled RMSE is 0\.9\\%")), "0.9 (0.4-1.4) lower",
        (lambda r: vci(100 * r["rel_rmse"], [100 * r["rel_rmse_ci"][0], 100 * r["rel_rmse_ci"][1]], 1))(C["contrasts"]["chain - camsfc_ds"]["as_run"][ALL]["all"]), "= as run",
        (lambda r: vci(100 * r["rel_rmse"], [100 * r["rel_rmse_ci"][0], 100 * r["rel_rmse_ci"][1]], 1))(C["contrasts"]["chain - camsfc_ds"]["inf"][ALL]["all"]),
        "lower RMSE of the CAMS-driven downscaling than the two-stage system, in %", all(C["identity"].values()))
    ca = C["arms"]["chain"][f"{ALL}|all"]
    row(sec, "Two-stage spread/skill at the stations", loc(("SI", r"its spread--skill ratio is 0\.56")), "0.56", f"{ca['ssr_fortin_as_run']:.3f}", "= as run", f"{ca['ssr_inf']:.3f}", "", all(C["identity"].values()))
    fa = F["arms"]["foehn"]
    row(sec, "Against the retrospective Foehn (table F): RMSE of the two-stage system / Foehn, lead day 1 and five lead days (ug m-3)", loc(("SI", r"the RMSE of the two-stage system is 11\.8"), ("main", r"12\.5 against 12\.0")),
        "11.8 / 11.5; 12.0 / 12.5", f"{F['arms']['chain'][f'{ALL}|1']['rmse_as_run']:.2f} / {fa[f'{ALL}|1']['rmse_as_run']:.2f}; {F['arms']['chain'][f'{ALL}|all']['rmse_as_run']:.2f} / {fa[f'{ALL}|all']['rmse_as_run']:.2f}",
        f"= as run / {fa[f'{ALL}|1']['rmse_8']:.2f}; = as run / {fa[f'{ALL}|all']['rmse_8']:.2f}", f"{F['arms']['chain'][f'{ALL}|1']['rmse_inf']:.2f} / {fa[f'{ALL}|1']['rmse_inf']:.2f}; {F['arms']['chain'][f'{ALL}|all']['rmse_inf']:.2f} / {fa[f'{ALL}|all']['rmse_inf']:.2f}",
        "Foehn: four stored members (M=8 from them); the two-stage system: eight (as run = M=8)", all(F["identity"].values()))
    cf_ = F["contrasts"]["chain - foehn"]

    def pct_ci(r, key="rel_rmse_ci_inflated"):
        if r is None:
            return "n/a"
        ci = r.get(key) or r.get("rel_rmse_ci")
        return vci(100 * r["rel_rmse"], None if ci is None else [100 * ci[0], 100 * ci[1]], 1)
    row(sec, "Two-stage against Foehn, lead day 1: RMSE change (%)", loc(("SI", r"against 11\.5 for \\model\{\} \(2\.4\\% higher")), "+2.4 (-0.6 to 7.9)",
        pct_ci(cf_["as_run"][ALL]["1"], "rel_rmse_ci") + "; with Foehn's draw " + (pct_ci(cf_["as_run_widened"][ALL]["1"]) if "as_run_widened" in cf_ else "n/a"),
        pct_ci(cf_["M8"][ALL]["1"]), pct_ci(cf_["inf"][ALL]["1"]),
        "positive = Foehn better; first as-run interval = the paper's bootstrap (identity); Foehn's draw (jackknife over its 4 fields, max with the carried quartet CV) in the other intervals (fix 9)",
        all(F["identity"].values()))
    if "foehn - bcsd_fc" in F["contrasts"]:
        rb = lambda form, L: cc(F, "foehn", "bcsd_fc", str(L), form)
        row(sec, "Foehn against the quantile mapping of the 0.4 deg forecast (table F): RMSE reduction, lead days 1 / 3 / 5 (%)", loc(("SI", r"the RMSE of \\model\{\} is 6\\% lower on lead day 1"), ("main", r"Foehn keeps an advantage of 6\\%")),
            "6 / within 2", " / ".join(rb("as_run", L) for L in (1, 3, 5)) + "; with Foehn's draw " + " / ".join(rb("as_run_widened", L) for L in (1, 3, 5)),
            " / ".join(rb("M8", L) for L in (1, 3, 5)), " / ".join(rb("inf", L) for L in (1, 3, 5)),
            "score_official block bootstrap of the contrast Foehn - bcsd_fc (an added pair of table F: no stored reference, identity None), widened with Foehn's draw (four-field jackknife, "
            "max with the carried quartet CV; the mapped members: eight, fresh noise); main_text:159 'matches Foehn's RMSE from 3 days ahead' is the four-member value (fix 12)", None)

        w159 = loc(("main", r"matches Foehn's RMSE from 3 days ahead"))
        CONC.append(f"{w159} and SI: Foehn against the quantile mapping of the same 0.4 deg forecast (table F), RMSE reduction at lead days 1 / 3 / 5: as run (four members) "
                    f"{' / '.join(rb('as_run_widened', L) for L in (1, 3, 5))} (Foehn's draw in the interval; 'matches Foehn's RMSE from 3 days ahead'); M = 8 "
                    f"{' / '.join(rb('M8', L) for L in (1, 3, 5))}; infinite {' / '.join(rb('inf', L) for L in (1, 3, 5))}")

    ARMS_T = [("chain", "two-stage"), ("camsfc_ds", "CAMS forecast, downscaled"), ("bcsd_fc", "quantile mapping"), ("coarse", "0.4 deg forecast"), ("persist_ds", "persisted downscaled field"), ("ds_sameday", "downscaler, inputs of the valid day")]
    row(sec, "Supplementary Table chain2022a: RMSE by lead day 1-5 / all of the ensemble arms (ug m-3): " + ", ".join(n for _, n in ARMS_T), loc(("SI", r"Two-stage system \(0\.4")),
        "11.89 .. 12.04; 11.76 ..; 12.45 ..; 14.53 ..; 12.57 ..; 11.76 ..", "; ".join("/".join(f"{C['arms'][a][f'{ALL}|{L}']['rmse_as_run']:.2f}" for L in ["1", "2", "3", "4", "5", "all"]) for a, _ in ARMS_T if a in C["arms"]),
        "; ".join("/".join(f"{C['arms'][a][f'{ALL}|{L}']['rmse_8']:.2f}" for L in ["1", "2", "3", "4", "5", "all"]) for a, _ in ARMS_T if a in C["arms"]),
        "; ".join("/".join(f"{C['arms'][a][f'{ALL}|{L}']['rmse_inf']:.2f}" for L in ["1", "2", "3", "4", "5", "all"]) for a, _ in ARMS_T if a in C["arms"]),
        "members: " + ", ".join(f"{n} {C['arms'][a][f'{ALL}|all']['M']}" for a, n in ARMS_T if a in C["arms"]) + "; CAMS, Aurora, climatology and the estimates are deterministic", all(C["identity"].values()))
    row(sec, "Supplementary Table chain2022b: same-day downscaler against the coarse field by network (EEA, CNEMC, EPA, AirKorea, Japan, CPCB, six networks) (%)", loc(("SI", r"Europe \(EEA\) & \$-\$4\.8")),
        "4.8 / 18.1 / 7.2 / -22.1 / -14.6 / 4.9 / 9.1", " / ".join(ccg(Sx, "ds_sameday", "cams0_bil", "sameday", "as_run", g) for g in NW), "= as run",
        " / ".join(ccg(Sx, "ds_sameday", "cams0_bil", "sameday", "inf", g) for g in NW), "reductions (negative = less accurate than the coarse field)", all(Sx["identity"].values()))
    for B_, pr, wh in (("cams0_bil", "9.1 (6.2-13.2)", "against the coarse field"), ("bcsd", "4.1 (1.3-6.5)", "against its quantile mapping"), ("clim_paper", "15.4", "against the climatology")):
        row(sec, f"Same-day downscaler at the stations, RMSE reduction {wh} (%)", loc(("SI", r"the RMSE of the downscaled field is 14\.1")), pr,
            cc(Sx, "ds_sameday", B_, "sameday", "as_run"), "= as run", cc(Sx, "ds_sameday", B_, "sameday", "inf"), "", all(Sx["identity"].values()))

# ============================================================================ 8. downscaler test 2022 against the estimates
ds, _ = J("ds2022_classU.json")
if ds:
    g_ = ds["regions"]["global"]
    sec = "8 Downscaler test of 2022 against the 1 km estimates (eight members, seeds keyed by the day: V_draw = 0)"
    row(sec, "RMSE of the ensemble mean (ug m-3); coarse field 13.50, quantile mapping 12.37", loc(("methods", r"The RMSE of the ensemble mean is 9\.98")), "9.98",
        f"{g_['cdffm']['as_run']:.2f}", "= as run", f"{g_['cdffm']['inf']:.2f}", "", ds["identity_all"])
    row(sec, "RMSE reduction against the coarse field / the quantile mapping (%)", loc(("main", r"RMSE 26\\% and 19\\% lower")), "26 / 19",
        vci(g_["rmse_change_pct_vs_cams_bilinear_as_run"]["v"], g_["rmse_change_pct_vs_cams_bilinear_as_run"]["ci"], 1, True) + " / " + vci(g_["rmse_change_pct_vs_bcsd_as_run"]["v"], g_["rmse_change_pct_vs_bcsd_as_run"]["ci"], 1, True),
        "= as run", vci(g_["rmse_change_pct_vs_cams_bilinear_inf"]["v"], g_["rmse_change_pct_vs_cams_bilinear_inf"]["ci"], 1, True) + " / " + vci(g_["rmse_change_pct_vs_bcsd_inf"]["v"], g_["rmse_change_pct_vs_bcsd_inf"]["ci"], 1, True), "", ds["identity_all"])
    row(sec, "Fair CRPS 3.13 against 3.85 for the climatological ensemble (19 % below)", loc(("methods", r"the fair CRPS is 3\.13"), ("main", r"their fair CRPS is 19\\% below")), "3.13 / 3.85; 19",
        f"{g_['cdffm']['crps_fair']:.3f} / {g_['clim']['crps_fair']:.3f}; {-g_['crps_change_pct_vs_clim_fair_equals_inf']:.1f}", "not stored (pair term)", "= as run (fair)", "", ds["identity_all"])

# ============================================================================ 8b. selection of the widened prior on 2021 (methods.tex:180)
PMG = P / "runs_ghap" / "pmG_cal" / "results"
if (PMG / "score_TT_k1.0_tau1.60.json").exists():
    jl = lambda n: json.loads((PMG / n).read_text())["results"]            # read only
    TTj, D0j, FFj = jl("score_TT_k1.0_tau1.60.json"), jl("d0_frozen.json"), jl("score_frozen_fit.json")

    def pm(u):
        mi = u["D_predictive_mean_sq_error"]
        assert abs(u["rmse_ensmean"] ** 2 - u["member_variance"] / 8 - mi) < 1e-3 * mi, "D_predictive_mean_sq_error is not RMSE_8^2 - s2/8"
        return dict(r8=u["rmse_ensmean"], ri=float(np.sqrt(mi)), f8=u["spread_skill_fortin"], si=float(np.sqrt(u["member_variance"] / mi)), crps=u["crps_fair"])
    s0, s1, t0_, t1_ = pm(D0j["sel"]["ug"]), pm(TTj["sel"]["ug"]), pm(FFj["fit"]["ug"]), pm(TTj["fit"]["ug"])
    chp = lambda a, b, k, d=2: f"{100 * (b[k] / a[k] - 1):+.{d}f}"
    sec = "8b Selection of the widened prior on 2021 (downscaler against the 1 km estimates; eight members, fresh seeds by day: V_draw = 0; selection-related, not a test score)"
    w180 = loc(("methods", r"their spread--skill ratio against the 1\\,km estimates is 0\.67"))
    row(sec, "Widened prior on 2021: 24 selection days spread/skill before -> after, CRPS change, RMSE change of the ensemble mean (%); 48 other days the same",
        w180, "0.67 -> 0.90; -2.0; +0.9. Other days 0.62 -> 0.83; -1.9; -",
        f"{s0['f8']:.3f} -> {s1['f8']:.3f}; {chp(s0, s1, 'crps')}; {chp(s0, s1, 'r8')}. Other days {t0_['f8']:.3f} -> {t1_['f8']:.3f}; {chp(t0_, t1_, 'crps')}; {chp(t0_, t1_, 'r8')}",
        "= as run (eight members)",
        f"{s0['si']:.3f} -> {s1['si']:.3f}; {chp(s0, s1, 'crps')} (fair = inf); {chp(s0, s1, 'ri')}. Other days {t0_['si']:.3f} -> {t1_['si']:.3f}; {chp(t0_, t1_, 'crps')}; {chp(t0_, t1_, 'ri')}",
        "runs_ghap/pmG_cal/results d0_frozen.json (sel), score_frozen_fit.json (fit), score_TT_k1.0_tau1.60.json (sel, fit), 'ug' blocks, read only; as run: Fortin-8 ratio and "
        "RMSE of the eight-member mean; inf: SSR_inf = sqrt(s2)/RMSE_inf, RMSE_inf = sqrt(D_predictive_mean_sq_error) (= RMSE_8^2 - s2/8, checked); CRPS_8 not stored (no pair "
        "term); no interval (the paper prints none); the selection is not re-applied (fix 5)", None)
    CONC.append(f"{w180}: selection of the widened prior on 2021, RMSE of the ensemble mean: as run (eight members) {chp(s0, s1, 'r8')} % on the 24 selection days and "
                f"{chp(t0_, t1_, 'r8')} % on the 48 other days ('a 0.9 % increase'); infinite {chp(s0, s1, 'ri')} % and {chp(t0_, t1_, 'ri')} % (a decrease); spread/skill "
                f"{s0['f8']:.3f} -> {s1['f8']:.3f} (Fortin, eight members) and {s0['si']:.3f} -> {s1['si']:.3f} (infinite); CRPS {chp(s0, s1, 'crps')} % in both (fair)")

# ============================================================================ 9. real time
rt, _ = J("rt_classU.json")
if rt:
    T3 = rt["T03_contrasts_clean_period"]["contrasts"]; idr = rt["identity_all"]

    def tc(b, k, form, d=1):
        e = T3[f"km_vs_{b}"][k]
        x = e["as_run"] if form == "as_run" else e["inf"]
        return vci(x["est"], x["ci"], d, neg=True)
    w458 = loc(("SI", r"the reported intervals do not include the draw of these fields"))
    sec = ("9 Real-time record of 2026 (eight members as issued; the same eight noise fields serve every issue day; V_draw: paired delete-one-member jackknife over the 8 members). "
           "RMSE rows: as run = the eight-member value, its interval widened with the member jackknife of the eight-member statistic itself (fix 8; bootstrap only = identity, in rt_classU.json). "
           "Fair-CRPS rows: as run = infinite (the fair CRPS); M=8 = fair + pair/8. For the paper-edit step: " + w458 + " says the reported intervals do not include the draw of these fields")
    row(sec, "RMSE reduction of the 1 km forecast against the CAMS forecast of the same cycle, all monitors (%)",
        loc(("main", r"error is 14\\% below that of the operational forecast"), ("main", r"14\\% below that of the CAMS forecast of the same cycle \(95"), ("SI", r"14\\% lower; 95\\% interval, 7--21"), ("SI", r"All monitors \(RMSE 16\.48")),
        "14 (7-21); 14.4 (7.2-20.7)", tc("cams", "all|all", "as_run"), "= as run", tc("cams", "all|all", "inf"),
        f"A14 section 8 sentence: X = {100 * (1 - rt['T05_ensemble']['arms']['km']['all|all']['rmse_inf'] / rt['T05_ensemble']['arms']['km']['all|all']['rmse_as_run']):.1f} % lower RMSE with infinitely many members", idr)
    row(sec, "Mean over the eight regions (%)", loc(("main", r"12\\% \(5--17\\%\) as the unweighted mean"), ("SI", r"Mean over the eight regions")), "12 (5-17)", tc("cams", "all|regmean", "as_run"), "= as run", tc("cams", "all|regmean", "inf"), "", idr)
    row(sec, "Against the 0.4 deg forecast from which it is drawn (%)", loc(("main", r"10\\% \(5--15\\%\) below that of the 0\.4"), ("SI", r"Relative to the 0\.4\$\^\\circ\$ forecast")), "10 (5-15)",
        tc("rt04", "all|all", "as_run"), "= as run", tc("rt04", "all|all", "inf"), "both ensembles in the infinite form (eight members each)", idr)
    row(sec, "By lead day 1-5 against the CAMS forecast (%)", loc(("main", r"12\\% on the day of issue"), ("SI", r"Lead day 1 \(day of issue\)")), "11.9 / 13.7 / 13.6 / 17.0 / 15.1",
        " / ".join(tc("cams", f"{L}|all", "as_run") for L in range(1, 6)), "= as run", " / ".join(tc("cams", f"{L}|all", "inf") for L in range(1, 6)), "", idr)
    pr_ = rt["T11_periods"]
    row(sec, "By period: September reforecasts / October-November reforecasts / issued 3-11 November (%)", loc(("main", r"in each of the three periods of the record"), ("SI", r"Reforecasts of 6--20 September")),
        "14.1 / 12.2 / 16.6", " / ".join(vci(pr_[p]["km_vs_cams"]["all|all"]["as_run"]["est"], pr_[p]["km_vs_cams"]["all|all"]["as_run"]["ci"], 1, True) for p in ("sep", "oct", "live")), "= as run",
        " / ".join(vci(pr_[p]["km_vs_cams"]["all|all"]["inf"]["est"], pr_[p]["km_vs_cams"]["all|all"]["inf"]["ci"], 1, True) for p in ("sep", "oct", "live")), "", idr)
    regs = ["East Asia", "Africa", "South Asia", "North America", "Europe"]
    row(sec, "By region: East Asia (Korea, western Japan, Taiwan) / Africa / South Asia / North America / Europe (%)", loc(("main", r"Its interval excludes zero in Korea"), ("SI", r"East Asia & \$-\$32\.5")),
        "32.5 / 29.0 / 15.0 / 11.9 / -2.3", " / ".join(tc("cams", f"all|{r}", "as_run") for r in regs), "= as run", " / ".join(tc("cams", f"all|{r}", "inf") for r in regs), "", idr)
    row(sec, "Against the climatology of the 1 km estimates (%)", loc(("main", r"5\\% \(3--8\\%\) more accurate than the climatology"), ("SI", r"Relative to the climatology of the 1")), "5 (3-8)",
        tc("clim", "all|all", "as_run"), "= as run", tc("clim", "all|all", "inf"), "", idr)
    r10 = rt["T08_rings"]["contrasts"]["km_ring10_vs_cams"]["all|all"]
    row(sec, "Read 10 km away from the monitor, against the CAMS forecast (%)", loc(("main", r"Read 10\\,km away from the monitors, it is still 12"), ("SI", r"Forecast read 10\\,km away")), "12 (5-18)",
        vci(r10["as_run"]["est"], r10["as_run"]["ci"], 1, True), "= as run", vci(r10["inf"]["est"], r10["inf"]["ci"], 1, True), "ring members (eight) in the infinite form", idr)
    row(sec, "Against the BCSD baseline: all monitors / mean over regions (%)", loc(("main", r"9\.8\\% \(4\.2--13\.6\\%\) below that of this baseline"), ("methods", r"9\.8\\% \(4\.2--13\.6\\%\) below")),
        "9.8 (4.2-13.6) / 1.4 (-1.0 to 4.2)", tc("bcsd", "all|all", "as_run") + " / " + tc("bcsd", "all|regmean", "as_run"), "= as run", tc("bcsd", "all|all", "inf") + " / " + tc("bcsd", "all|regmean", "inf"), "", idr)
    cr = rt["T05_ensemble"]["crps_contrasts"]
    row(sec, "Fair CRPS against the 0.4 deg members / the quantile-mapped members (%)", loc(("main", r"its fair CRPS is 15\\% \(9--20\\%\) below")), "15 (9-20) / 13 (8-18)",
        vci(cr["km_vs_rt04"]["fair_as_run_equals_inf"]["est"], cr["km_vs_rt04"]["fair_as_run_equals_inf"]["ci"], 1, True) + " / " + vci(cr["km_vs_bcsd_m"]["fair_as_run_equals_inf"]["est"], cr["km_vs_bcsd_m"]["fair_as_run_equals_inf"]["ci"], 1, True),
        vci(cr["km_vs_rt04"]["eight_member"]["est"], cr["km_vs_rt04"]["eight_member"]["ci"], 1, True) + " / " + vci(cr["km_vs_bcsd_m"]["eight_member"]["est"], cr["km_vs_bcsd_m"]["eight_member"]["ci"], 1, True),
        "= as run (fair CRPS is already infinite)", "M=8 column: the eight-member CRPS (fair + pair/8); intervals widened with the paired delete-one-member jackknife (fix 7)", idr)
    am = rt["T04_amplitude"]
    row(sec, "Day-to-day amplitude at a monitor (median over monitor x lead-day series), 1 km / 0.4 deg", loc(("main", r"0\.47 of the amplitude of the measured ones")), "0.47 / 0.70",
        vci(am['km']['as_run']['est'], am['km']['as_run'].get('ci'), 3) + " / " + vci(am['rt04']['as_run']['est'], am['rt04']['as_run'].get('ci'), 3), "= as run", vci(am["km"]["inf_pair_form_median"]["est"], am["km"]["inf_pair_form_median"]["ci"], 3) + " / " + vci(am["rt04"]["inf_pair_form_median"]["est"], am["rt04"]["inf_pair_form_median"]["ci"], 3),
        "pair form per series (var_t(xbar) - mean_i var_t(x_i - xbar)/7), then the median (a median is class E, A14 4b)", idr)
    ens = rt["T05_ensemble"]["arms"]["km"]["all|all"]; cal = rt.get("calibrated_kmcal", {})
    row(sec, "Spread/skill at the monitors: as issued / with the widened prior", loc(("methods", r"the spread--skill ratio from 0\.40 to 0\.54")), "0.40 / 0.54",
        vci(ens['spread_skill_as_run']['est'], ens['spread_skill_as_run'].get('ci'), 3) + " / " + (vci(cal['kmcal']['spread_skill_as_run']['est'], cal['kmcal']['spread_skill_as_run'].get('ci'), 3) if 'kmcal' in cal else 'n/a'), "= as run",
        vci(ens['spread_skill_inf']['est'], ens['spread_skill_inf'].get('ci'), 3) + " / " + (vci(cal['kmcal']['spread_skill_inf'], cal['kmcal']['spread_skill_inf_ci']['ci'], 3) if 'kmcal' in cal else 'n/a'), "intervals widened with the member jackknife (fix 8)", idr)
    if cal:
        cf8 = cal["crps_change_kmcal_vs_km_fair_equals_inf"]; c8_ = cal["crps_change_kmcal_vs_km_eight_member"]; rmc = cal["rmse_change_kmcal_vs_km"]
        row(sec, "Widened prior: CRPS change (%) and RMSE change (%)", loc(("methods", r"the CRPS falls by 4\.7\\% \(95\\% interval, 4\.2--5\.3")), "-4.7 (-5.3 to -4.2); RMSE unchanged",
            f"CRPS {vci(cf8['est'], cf8['ci'], 1)}; RMSE {vci(rmc['as_run']['est'], rmc['as_run']['ci'], 2)}", f"CRPS {vci(c8_['est'], c8_['ci'], 1)}; RMSE = as run",
            f"CRPS = as run (fair); RMSE {vci(rmc['inf']['est'], rmc['inf']['ci'], 2)}",
            f"changes, negative = the widened prior better; intervals widened with the paired delete-one-member jackknife (fixes 7-8); bootstrap only (the paper's): CRPS {vci(cf8['est'], cf8['ci_boot'], 1)}", idr)
    km = rt["T05_ensemble"]["arms"]
    row(sec, "RMSE over the five lead days (ug m-3): 1 km / 0.4 deg (CAMS 19.24, climatology 17.4: deterministic)", loc(("SI", r"the RMSE of the 1\\,km forecast is 16\.5")), "16.48 / 18.4",
        f"{km['km']['all|all']['rmse_as_run']:.2f} / {km['rt04']['all|all']['rmse_as_run']:.2f}", "= as run", f"{km['km']['all|all']['rmse_inf']:.2f} / {km['rt04']['all|all']['rmse_inf']:.2f}", "", idr)
    row(sec, "Not computed: forecast vs analysis inputs (0.2 %, Methods), reference-site rows (Supplementary Table rt)", loc(("methods", r"the RMSE is 0\.2\\% higher with forecast inputs"), ("SI", r"Reference sites, United States")),
        "0.2 (-2.0 to 1.5); 29.5 / 6.1", "not recomputed", "", "", "the samples of score.py tables T09 / T13 were not rebuilt here", None)

# ============================================================================ write
hdr = "| # | section | quantity | paper (file:line) | printed | as run (95 %) | M = 8 (95 %) | M = inf (95 %) | identity | notes |"
md = [f"# A14 class U: every class-U quantity of the paper at M as run, M = 8 and M = infinity\n",
      f"Built {time.strftime('%Y-%m-%d %H:%M:%S')} by a14/classU/scripts/make_table.py from results/*.json. Rulebook: protocol_amendments/A14_infinite_ensemble.md; "
      "interpretation fixed before any score in a14/classU/NOTES.md. Gains are reductions in % (positive = Foehn better) unless a row says 'change'. "
      "Intervals: the paper's bootstrap of each record; the M = 8 and infinite half-widths, and the as-run half-widths of records whose members are fixed fields (the four stored members, the real-time eight; fixes 8-9), are multiplied by sqrt(1 + V_draw/V_boot) (A14 4c); the bootstrap-only as-run intervals are the identity check (in the result JSONs). "
      "'identity' = the as-run value and interval reproduce the paper's stored result with its own code (True), None = no stored per-row reference.\n",
      hdr, "|" + "---|" * 10]
if DESIGN_NOTE:                                                      # A16 4.3: stated above the table; the notes of the rows describe the fixed design
    md.insert(2, f"**Noise design FOEHN_DESIGN={DESIGN_NOTE['FOEHN_DESIGN']}.** {DESIGN_NOTE['rule']}. Where a row's notes describe a V_draw "
                 "of the final model, that V_draw enters no interval here (every final-model interval is the bootstrap interval, factor 1).\n")
sec0 = None
for i, r in enumerate(ROWS, 1):
    md.append(f"| {i} | {r['section'].split(' ')[0]} | {r['quantity']} | {r['paper']} | {r['printed']} | {r['as_run']} | {r['M8']} | {r['inf']} | {r['identity']} | {r['notes']} |")
md.append("\nSections: " + "; ".join(sorted({r['section'] for r in ROWS})))
# ---------------------------------------------------------------- conclusions that differ between forms (A14 section 3)
md.append("\n## Conclusions that differ between forms (A14 section 3; as-run value first, D4)\n")
md += [f"{i}. {c}" for i, c in enumerate(CONC, 1)]
if CHK:
    md.append("\nChecked, no change of conclusion: " + "; ".join(CHK) + ".")
# systematic scan of the cells above: intervals that include zero in one form and not in another (candidates for A14 section 3)
RXI = re.compile(r"(-?\d+(?:\.\d+)?) \((-?\d+(?:\.\d+)?) to (-?\d+(?:\.\d+)?)\)")


def _items(s):
    return [(float(a), float(b), float(c)) for a, b, c in RXI.findall(str(s))]


SCAN = []
for i, r in enumerate(ROWS, 1):
    A_ = _items(str(r["as_run"]).split("with Foehn's draw")[-1])
    for nm, X in (("M = 8", _items(r["M8"])), ("infinite", _items(r["inf"]))):
        if A_ and len(X) == len(A_):
            for k, (a, x) in enumerate(zip(A_, X), 1):
                if (a[1] <= 0 <= a[2]) != (x[1] <= 0 <= x[2]):
                    SCAN.append(f"row {i} ({r['paper'].split(';')[0]}), item {k}: as run {a[0]:g} ({a[1]:g} to {a[2]:g}), {nm} {x[0]:g} ({x[1]:g} to {x[2]:g}); {r['quantity'][:90]}")
md.append("\n### Systematic scan: intervals that include zero in one form and not in another\n")
md.append("Candidates for A14 section 3, read from the cells above (as run = the widened four-member or eight-member interval; rows marked 'NOT the paper's sample' "
          "or '1 km draw only' keep their caveats; a candidate is a changed conclusion only where the paper draws one from that interval).\n")
md += [f"- {s}" for s in SCAN] if SCAN else ["- none"]
# ---------------------------------------------------------------- K2 (A14 section 6): the stored quartet against the sixteen members
K2ROWS = []
K2Q = ("q0123", "all16", "quartet_mean", "quartet_sd", "z", "rank", "perm_p_two_sided", "flag_2p5", "z_vs_quartet_mean", "draw_sd", "z_drawscale", "flag_drawscale_2p5")
FINITE_M = ("rmse_m", "ssr_fortin_m")


def k2rows(setname, h, rep):
    for k in ("rmse_inf", "rmse_8", "spread", "ssr_inf", "ssr_fortin_8", "crps_fair", "crps_8") + FINITE_M:
        if k in rep and isinstance(rep[k], dict):
            K2ROWS.append(dict(set=setname, lead=h, stat=k, classU=k not in FINITE_M, **{q: rep[k].get(q) for q in K2Q}))
    for k in ("bs_fair", "bs_8"):
        if k in rep:
            for j, T in enumerate((15, 37.5, 75)):
                K2ROWS.append(dict(set=setname, lead=h, stat=f"{k} > {T:g}", classU=True, **{q: rep[k][j].get(q) for q in K2Q}))


fz = lambda x, fmt: "n/a" if x is None else format(x, fmt)                # noqa: E731
ndays = None
if rl:
    for h, d_ in rl["populated_pixels"]["k2_populated_pixels"].items():
        for wn, rep in d_.items():
            k2rows(f"populated pixels ({wn}) vs estimates", h, rep); ndays = rep.get("days")
    for h, d_ in rl["stations_reliability"]["k2_cal_station_pixels"].items():
        for v, gg in d_.items():
            k2rows(f"station pixels, reliability sample, vs {v}", h, gg["All official networks"])
if st:
    for h, gg in st["k2_station_pixels"].items():
        k2rows("station pixels, compare_fair station-days, vs measurements", h, gg["all"])
md.append(f"\n## K2 (A14 section 6): members 0-3 of the sixteen-member run against all sixteen ({ndays} start days{'' if (ndays or 0) >= 170 else ', provisional until R1'})\n")
md.append("z = (quartet 0-3 - sixteen) / sd, sd = the PLAIN sd (ddof 0) of the same statistic over the 1,820 quartets of the 16 fields; flag = |z| > 2.5; rank = mid-rank of the "
          "quartet 0-3 among the 1,820 quartets (0 = lowest, 1 = highest); perm p = two-sided permutation p, 2 min(P(t_Q >= t_0123), P(t_Q <= t_0123)). Corrected "
          "(NOTES.md item 5, fix 1): the first version divided by the draw-scale sd (sd x sqrt(16/12)), kept beside as 'z (draw scale)'; the 16/12 factor belongs to V_draw only. "
          "Finite-M rows (rmse_m, ssr_fortin_m: four-member values whose sixteen-member counterpart is another estimand): z = (quartet 0-3 - quartet mean) / sd, not counted (fix 10).\n")
md.append("| set | lead | statistic | quartet 0-3 | 16 members | quartet mean | quartet sd | z | rank | perm p | flag | z (draw scale) |")
md.append("|" + "---|" * 12)
for r in K2ROWS:
    zz = r["z"] if r["classU"] else r["z_vs_quartet_mean"]
    zd = r["z_drawscale"] if r["classU"] else (None if zz is None else zz / np.sqrt(16 / 12))
    md.append(f"| {r['set']} | {r['lead']} | {r['stat']}{'' if r['classU'] else ' (finite M: z vs the quartet mean, not counted)'} | {fz(r['q0123'], '.4g')} | {fz(r['all16'], '.4g')} | "
              f"{fz(r['quartet_mean'], '.4g')} | {fz(r['quartet_sd'], '.3g')} | {fz(zz, '+.2f')} | {fz(r['rank'], '.3f')} | {fz(r['perm_p_two_sided'], '.3f')} | "
              f"{r['flag_2p5'] if r['classU'] else '-'} | {fz(zd, '+.2f')} |")
cu = [r for r in K2ROWS if r["classU"]]
nflag = sum(bool(r["flag_2p5"]) for r in cu); nflag_d = sum(bool(r["flag_drawscale_2p5"]) for r in cu)
md.append(f"\nClass-U statistics flagged (|z| > 2.5, plain sd): {nflag} of {len(cu)}" +
          (": " + "; ".join(f"{r['set']} lead {r['lead']} {r['stat']} (z {r['z']:+.2f}, rank {r['rank']:.3f}, p {r['perm_p_two_sided']:.3f})" for r in cu if r["flag_2p5"]) if nflag else "") +
          f". On the draw scale of the first version: {nflag_d} of {len(cu)}. Finite-M rows: " +
          "; ".join(f"{r['set']} lead {r['lead']} {r['stat']} z {fz(r['z_vs_quartet_mean'], '+.2f')}" for r in K2ROWS if not r["classU"] and r["stat"] == "ssr_fortin_m") + ".")
SUFX = os.environ.get("A14U_OUT", "")
safe_text(RES / f"classU_table{SUFX}.md", "\n".join(md) + "\n")
safe_json(RES / f"classU_table{SUFX}.json", {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": ROWS, "k2": K2ROWS,
                                              **({"design": DESIGN_NOTE} if DESIGN_NOTE else {})})
print(f"{len(ROWS)} rows -> results/classU_table{SUFX}.md, .json", flush=True)
