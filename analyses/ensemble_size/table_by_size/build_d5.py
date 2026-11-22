#!/usr/bin/env python3
"""A14 decision D5 (A16 Addendum 3, point 2(c)): the Supplementary Table 'Scores by ensemble size' (label tab:si_members).

    python -B pass16.py                      # (pass16.py writes out_<label>/t16.npz, then runs this script)
    python -B build_d5.py                    # FOEHN_DESIGN unset: the stored records (results/, data/), writes out_stored/
    FOEHN_DESIGN=ar python -B build_d5.py    # the records of the noise drawn anew at every start day (results_ar/, data_ar/), out_ar/

Writes only out_<label>/d5.json (the format of source_data/supp_tables_full/*.json), out_<label>/d5.tex (one short upright
table with its caption) and out_<label>/identity.json. Every write: temporary file, flush, fsync, read back, os.replace.
Reads only (nothing under P/a14 or analyses/ is changed; no station, monitor or OpenAQ file is opened by this script):
  P/a14/classU/results<SFX>/{classU_table_179,st_cf_classU_179,rel_classU_179}.json   the class-U record and its numbers
  P/a14/classU/data<SFX>/{st_cf_terms,rel_extra}.npz        per-start-day sums of the four members (pass outputs)
  analyses/reliability/data<SFX>/sums_final.npz             per-start-day sums of the reliability samples (four members)
  analyses/reliability/scripts/rel_common.py                block_weights only (taken from the unchanged source by ast)
  P/a14/classE/results<SFX>/classE_all2022.{json,md}        the class-E record
  out_<label>/t16.npz                                        pass16.py: the M = 16 values and intervals and their proof

Columns: 4 members (as run), 8, 16, infinitely many.
- Class U, stations, 4 members: recomputed here from the per-start-day sums of the pass outputs with the record's estimators
  and bootstraps (compare_fair: default_rng(0), blocks of 7 start days, 10,000 resamples; reliability: block_weights(B=7,
  n=10000, seed=0)), intervals widened with the record's V_draw (stored design) or not (V_draw = 0, design 'ar'). The
  four-member eight-member and infinite forms are recomputed the same way as a check of the estimator code.
- Class U, 8 and infinitely many members: the sixteen-member values of the record (st_cf_score.py rmse16_* / gain16_*,
  rel_score.py stations_reliability_16), unrounded.
- Class U, 16 members: the same estimator at M = 16 (MSE_16 = MSE_inf + mean(s2)/16; CRPS_16 = fair CRPS + pair/16;
  spread/skill sqrt(17/16) Fortin form) with the record's bootstrap, from t16.npz (pass16.py: the sixteen-member per-day
  sums rebuilt by the scorers' own statements, whose M = 8 and infinite values and intervals reproduce the record); its
  point is checked here against the exact algebra on the record values ((MSE_inf + MSE_8)/2 etc.).
- CRPS, 4 members: the expected CRPS of four members, fair CRPS + pair/4 of the four members of the evaluation (the code path
  is checked at pair/8 against the record's four-member eight-member value); the record's as-run value (fair CRPS of the four,
  a second estimate of the infinite value) is reproduced in the identity and given in the description of d5.json.
- Class E: copied from the record (M4_stored, M8_liberal, M16_liberal_measured, infinite point / range / interval).
Identity (identity.json): the 4 / 8 / infinite strings of every printed class-U row equal those of classU_table_179.json
(make_table.vci formatting); the class-E cells equal the JSON and the markdown of the record. A cell that differs is not
printed."""
import ast
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/print/d5"))
NAT = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms"))
P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
DESIGN = os.environ.get("FOEHN_DESIGN", "")
if DESIGN not in ("", "ar"):
    sys.exit(f"FOEHN_DESIGN={DESIGN!r}: build_d5.py knows only unset (stored) and 'ar'")
LABEL, SFX, VDRAW0 = ("ar", "_ar", True) if DESIGN == "ar" else ("stored", "", False)
OUT = HERE / f"out_{LABEL}"
U, E, REL = P / "a14" / "classU", P / "a14" / "classE", NAT / "analyses" / "reliability"
IN = {"classU_table": U / f"results{SFX}" / "classU_table_179.json", "st_cf": U / f"results{SFX}" / "st_cf_classU_179.json",
      "rel": U / f"results{SFX}" / "rel_classU_179.json", "st_cf_terms": U / f"data{SFX}" / "st_cf_terms.npz",
      "rel_extra": U / f"data{SFX}" / "rel_extra.npz", "sums_final": REL / f"data{SFX}" / "sums_final.npz",
      "rel_common": REL / "scripts" / "rel_common.py", "classE": E / f"results{SFX}" / "classE_all2022.json",
      "classE_md": E / f"results{SFX}" / "classE_all2022.md", "t16": OUT / "t16.npz"}
MISSING = [str(p) for p in IN.values() if not p.exists()]
if MISSING:
    sys.exit("build_d5.py (FOEHN_DESIGN=%r): inputs missing, nothing written (t16.npz is written by pass16.py):\n  %s" % (DESIGN, "\n  ".join(MISSING)))
TOL = 1e-9                                   # relative tolerance of the recomputations (float sums in another order)
A_ = "All official networks"
CA, AU = "CAMS d+1 (equal issuance)", "Aurora 00 UTC d+1 (equal issuance)"


# ------------------------------------------------------------------------------------------------ writing, formatting
def safe_write(path, text):
    path = Path(path)
    assert str(path).startswith(str(HERE) + "/"), path
    data = text.encode("utf-8"); tmp = str(path) + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data); fh.flush(); os.fsync(fh.fileno())
    with open(tmp, "rb") as fh:
        assert fh.read() == data, tmp
    os.replace(tmp, path)
    with open(path, "rb") as fh:
        assert fh.read() == data, path


def jdump(obj):
    def d(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(obj, indent=1, ensure_ascii=False, default=d, allow_nan=False) + "\n"


def f_(x, d):                                   # make_table.f_
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def vci(v, ci, d, neg=False):                   # make_table.vci: the string of the record
    s = -1 if neg else 1
    if v is None:
        return "n/a"
    if ci is None or ci[0] is None:
        return f_(s * v, d)
    lo, hi = sorted([s * ci[0], s * ci[1]])
    return f"{f_(s * v, d)} ({f_(lo, d)} to {f_(hi, d)})"


def rnd(x, d):
    return None if x is None else float(f"{x:.{d}f}")


def tnum(x, d):                                 # LaTeX number of the manuscript: 1{,}497.1, $-$4.8
    s = f"{x:,.{d}f}".replace(",", "{,}")
    return "$-$" + s[1:] if s.startswith("-") else s


def rel_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fn_from(src, name):
    """one function of an unchanged source file, defined alone (nothing else of the file runs)"""
    node = next(n for n in ast.parse(Path(src).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"np": np}
    exec(compile(ast.Module([node], type_ignores=[]), str(src), "exec"), ns)
    return ns[name]


def pctl(bs):
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return float(lo), float(hi)


def widen(pt, lo, hi, vb, vd):
    """A14 4c (u_common.inflate / rel_score.form): half-widths x sqrt(1 + V_draw/V_boot); design 'ar': V_draw = 0 (A16 4.3)"""
    if VDRAW0:
        return [lo, hi]
    f = float(np.sqrt(1.0 + vd / vb)) if vb > 0 and np.isfinite(vd) else float("nan")
    return [pt - f * (pt - lo), pt + f * (hi - pt)]


CHECKS = []                                     # recomputation checks: (name, max relative difference, passed)


def check(name, mine, rec, tol=TOL):
    r = rel_diff(mine, rec); CHECKS.append({"check": name, "max_rel_diff": r, "passed": bool(r <= tol)}); return r <= tol


# ------------------------------------------------------------------------------------------------ records
TAB = json.loads(IN["classU_table"].read_text())
ST = json.loads(IN["st_cf"].read_text())
RL = json.loads(IN["rel"].read_text())
CE = json.loads(IN["classE"].read_text())
MD = IN["classE_md"].read_text()
for name, rec in (("st_cf", ST), ("rel", RL)):        # the record must be of the requested design
    rd = (rec.get("design") or {}).get("FOEHN_DESIGN", "")
    if rd != DESIGN:
        sys.exit(f"{IN[name]} is of design {rd!r}, not {DESIGN!r}: nothing written")
if VDRAW0 and "V_draw = 0" not in json.dumps(CE.get("design", "")) + MD[:3000]:
    sys.exit(f"{IN['classE']}: no 'V_draw = 0' design note: nothing written")
if not VDRAW0 and ("design" in CE or "FOEHN_DESIGN=" in MD[:3000]):
    sys.exit(f"{IN['classE']} carries a design note: not the stored record, nothing written")
with np.load(IN["t16"]) as _z:                          # pass16.py: sixteen-member sums, their proof, the M = 16 intervals
    T16M = json.loads(str(_z["meta"]))
if T16M["design"] != LABEL or not T16M["all_passed"]:
    sys.exit(f"{IN['t16']}: design {T16M['design']!r}, proof passed {T16M['all_passed']}: nothing written")
for _k, _n in (("st_cf_record", "st_cf"), ("rel_record", "rel")):
    if T16M["inputs"][_k]["sha256"] != sha(IN[_n]):
        sys.exit(f"{IN['t16']} was built from another {IN[_n].name}: rerun pass16.py; nothing written")
R16M = T16M["results16"]

# ------------------------------------------------------------------------------------------------ 1. compare_fair station-days
Z = np.load(IN["st_cf_terms"])
PD, S2 = Z["PD"], Z["s2"]
SYSN = [str(s) for s in Z["systems"]]                   # st_cf_score.REFS: index 4 = CA, 6 = AU, 0 = Foehn 1 km
assert [str(s) for s in Z["nets"]][6] == "all" and SYSN[0].endswith("(1 km)"), (SYSN, list(Z["nets"]))
assert SYSN[4] == "CAMS (run d+1, equal issuance)" and SYSN[6] == "Aurora (00 UTC d+1, equal issuance)", SYSN
OK = np.isfinite(PD[0, :, 0, 0]); ND = int(OK.sum())
_rng = np.random.default_rng(0)                         # compare_fair's bootstrap, verbatim (st_cf_score.py)
IDX = (_rng.integers(0, ND - 7 + 1, size=(10000, -(-ND // 7)))[:, :, None] + np.arange(7)).reshape(10000, -1)[:, :ND]


def cf_sys(k, h):
    a = PD[k, OK, h - 1]; return a[:, 0].astype(np.float64), a[:, 1].astype(np.float64)


def cf_rmse(num, n):
    return float(np.sqrt(np.nansum(num) / np.nansum(n))), np.sqrt(np.nansum(num[IDX], 1) / np.nansum(n[IDX], 1))


def cf_ratio(na, ca, nb, cb):
    pt = np.sqrt(np.nansum(na) / np.nansum(ca)) / np.sqrt(np.nansum(nb) / np.nansum(cb)) - 1
    bs = np.sqrt(np.nansum(na[IDX], 1) / np.nansum(ca[IDX], 1)) / np.sqrt(np.nansum(nb[IDX], 1) / np.nansum(cb[IDX], 1)) - 1
    return float(pt), bs


CU = {}                                                  # (quantity key, lead) -> {form: (v, [lo, hi] or None)}
for h in (1, 3):
    G = ST["groups"]["all"][str(h)]["Foehn 1 km"]
    sse, n = cf_sys(0, h); s2 = S2[6][OK, h - 1]
    NUM4 = {"as_run": sse, "M8": sse - s2 / 4 + s2 / 8, "inf": sse - s2 / 4}      # st_cf_score.forms_foehn, four members
    rm4 = {}
    for f, num in NUM4.items():
        pt, bs = cf_rmse(num, n); lo, hi = pctl(bs); r = G["rmse"][f]
        ci = widen(pt, lo, hi, float(np.nanvar(bs, ddof=1)), r["v_draw"])
        check(f"compare_fair RMSE, four members, {f}, lead {h}: point, bootstrap interval, interval", [pt, lo, hi] + ci, [r["v"]] + r["ci_boot"] + r["ci"])
        rm4[f] = (pt, ci)
    CU[("rmse", h)] = {"as_run": rm4["as_run"]}
    sub = G["sixteen_members_provisional"]
    assert "rmse16_inf" in sub and sub["days"] == ND, "the record has no sixteen-member value on every start day"
    i16, e16 = sub["rmse16_inf"]["v"], sub["rmse16_M8"]["v"]
    check(f"compare_fair RMSE, sixteen members, lead {h}: rmse16_inf = sixteen_inf, rmse16_M8 = sixteen_M8", [i16, e16], [sub["sixteen_inf"], sub["sixteen_M8"]])
    r16 = float(np.sqrt((i16 ** 2 + e16 ** 2) / 2))                                  # MSE_16 = (MSE_inf + MSE_8)/2
    x16 = R16M[f"rmse|{h}"]                                                           # pass16.py, the record's bootstrap at M = 16
    check(f"compare_fair RMSE, sixteen members, lead {h}: M = 16 point of t16 = (MSE_inf + MSE_8)/2 of the record", x16["v"], r16)
    CU[("rmse", h)].update(M8=(e16, sub["rmse16_M8"]["ci"]), M16=(x16["v"], x16["ci"]), inf=(i16, sub["rmse16_inf"]["ci"]))
    for ref, j in ((CA, 4), (AU, 6)):
        sb, cb = cf_sys(j, h); rref = float(np.sqrt(np.nansum(sb) / np.nansum(cb)))
        g = G["gain_pct"][ref]
        pt, bs = cf_ratio(NUM4["as_run"], n, sb, cb)
        lo, hi = pctl(100 * bs)
        ci = widen(100 * pt, lo, hi, float(np.nanvar(100 * bs, ddof=1)), G["rmse"]["as_run"]["v_draw"] / rref ** 2 * 1e4)
        check(f"compare_fair gain over {ref}, four members, as run, lead {h}", [100 * pt, lo, hi] + ci, [g["as_run"]["v"]] + g["as_run"]["ci_boot"] + g["as_run"]["ci"])
        for f in ("M8", "inf"):
            p_, b_ = cf_ratio(NUM4[f], n, sb, cb); l_, h_ = pctl(100 * b_)
            c_ = widen(100 * p_, l_, h_, float(np.nanvar(100 * b_, ddof=1)), G["rmse"][f]["v_draw"] / rref ** 2 * 1e4)
            check(f"compare_fair gain over {ref}, four members, {f}, lead {h}", [100 * p_, l_, h_] + c_, [g[f]["v"]] + g[f]["ci_boot"] + g[f]["ci"])
        gi, g8 = sub[f"gain16_inf_{ref}"], sub[f"gain16_M8_{ref}"]
        check(f"gain over {ref}, sixteen members, lead {h}: 100 (RMSE16/RMSE_ref - 1) with RMSE_ref from PD", [100 * (i16 / rref - 1), 100 * (e16 / rref - 1)], [gi["v"], g8["v"]])
        key = "cams" if j == 4 else "aurora"
        y16 = R16M[f"gain|{h}|{ref}"]
        check(f"gain over {ref}, sixteen members, lead {h}: M = 16 point of t16 = 100 (RMSE_16/RMSE_ref - 1)", y16["v"], 100 * (r16 / rref - 1))
        CU[(key, h)] = {"as_run": (100 * pt, ci), "M8": (g8["v"], g8["ci"]), "M16": (y16["v"], y16["ci"]), "inf": (gi["v"], gi["ci"])}

# ------------------------------------------------------------------------------------------------ 2. reliability samples (lead 1)
block_weights = fn_from(IN["rel_common"], "block_weights")
ZS, X2 = np.load(IN["sums_final"]), np.load(IN["rel_extra"])
assert np.array_equal(ZS["keys"], X2["keys"]) and int(ZS["members"]) == 4
GI = [str(x) for x in ZS["groups"]].index(A_)
KK = ZS["keys"]; ROWS = np.where(KK[:, 1] == 1)[0]; ROWS = ROWS[np.argsort(KK[ROWS, 0])]
D = {k: ZS["day__" + k][ROWS, GI].astype(np.float64) for k in ("n", "var", "sse_0", "p_n", "p_crps", "p_clim_crps")}
D["p_pair"] = X2["day__p_pair"][ROWS, GI].astype(np.float64)


def div(a, b):                                          # s2b_station_scores.div
    with np.errstate(all="ignore"):
        return np.where(b != 0, a / np.where(b != 0, b, 1), np.nan)


OKC, OKP = D["n"] > 0, D["p_n"] > 0
WC, WP = block_weights(int(OKC.sum()), B=7, n=10000, seed=0), block_weights(int(OKP.sum()), B=7, n=10000, seed=0)
SC = {k: (D[k].sum(), WC @ D[k][OKC]) for k in ("n", "var", "sse_0")}
SP = {k: (D[k].sum(), WP @ D[k][OKP]) for k in ("p_crps", "p_clim_crps", "p_pair")}
S1 = RL["stations_reliability"]["by_lead"]["1"][A_]
R16 = RL["stations_reliability_16"]["by_lead"]["1"][A_]


def rel_forms(t):                                       # t = 0 point, 1 resamples
    n, var, sse = SC["n"][t], SC["var"][t], SC["sse_0"][t]
    pc, pcl, pp = SP["p_crps"][t], SP["p_clim_crps"][t], SP["p_pair"][t]
    return {"ss_0": np.sqrt(5 / 4 * div(var, sse)),                                           # derive (F = (M+1)/M, M = 4)
            "ss|8|0": np.sqrt(9 / 8 * div(var, n)) / np.sqrt(div(sse - var / 4 + var / 8, n)),  # derive_u, four members
            "ss|inf|0": np.sqrt(div(var, n)) / np.sqrt(div(sse - var / 4, n)),
            "p_crps_vs_climens": 100 * (div(pc, pcl) - 1),
            "p_crps_vs_climens|8": 100 * (div(pc + pp / 8, pcl) - 1), "p_crps_vs_climens|inf": 100 * (div(pc, pcl) - 1)}


RP, RB = rel_forms(0), rel_forms(1)
REL4 = {}
for k in RP:
    pt = float(RP[k]); lo, hi = pctl(RB[k]); vb = float(np.nanvar(RB[k], ddof=1))
    if k in ("ss_0", "p_crps_vs_climens"):                                                   # as run (fix 9)
        r = S1[k]["as_run"]; vd = max(np.nan_to_num(r["v_quart"]), np.nan_to_num(r["v_jack4"]))
    else:
        r = S1[k]; vd = r["v_draw"]
    ci = widen(pt, lo, hi, vb, vd)
    check(f"reliability sample, four members, {k}, lead 1: point, bootstrap interval, interval", [pt, lo, hi] + ci, [r["v"]] + r["ci_boot"] + r["ci"])
    REL4[k] = (pt, ci)
for k in ("ss|8|0", "ss|inf|0", "rmse|8|0", "rmse|inf|0", "p_crps_vs_climens|8", "p_crps_vs_climens|inf"):
    assert R16[k]["days"] == len(ROWS) and "ci" in R16[k], (k, R16[k].get("days"))
rm8, rmi = R16["rmse|8|0"]["v"], R16["rmse|inf|0"]["v"]
s2n = 8 * (rm8 ** 2 - rmi ** 2)                                                                 # mean member variance
check("spread/skill, sixteen members: sqrt(s2)/RMSE_inf and sqrt(9/8 s2)/RMSE_8 from the record's RMSE_inf, RMSE_8",
      [np.sqrt(s2n) / rmi, np.sqrt(9 / 8 * s2n) / rm8], [R16["ss|inf|0"]["v"], R16["ss|8|0"]["v"]])
x16 = R16M["ss|0|1"]
check("spread/skill, sixteen members: M = 16 point of t16 = sqrt(17/16 s2)/RMSE_16 from the record", x16["v"], np.sqrt(17 / 16 * s2n) / np.sqrt(rmi ** 2 + s2n / 16))
CU[("ssr", 1)] = {"as_run": REL4["ss_0"], "M8": (R16["ss|8|0"]["v"], R16["ss|8|0"]["ci"]),
                  "M16": (x16["v"], x16["ci"]), "inf": (R16["ss|inf|0"]["v"], R16["ss|inf|0"]["ci"])}
c8, ci_ = R16["p_crps_vs_climens|8"]["v"], R16["p_crps_vs_climens|inf"]["v"]
y16 = R16M["p_crps_vs_climens|1"]
check("CRPS, sixteen members: M = 16 point of t16 = inf + (M8 - inf)/2 of the record (pair term x 1/16)", y16["v"], ci_ + (c8 - ci_) / 2)

# the four members of the evaluation: expected CRPS of four members = fair CRPS + pair/4 (rel_score.derive_u with c8 -> 1/4 on the
# 'paper' sample; jackknife over the four fields with 3-member replicates; quartet CV of the same statistic: crps_8 from the
# record's K2 for the check at 1/8, crps_4 from pass16.py for 1/4); interval widened as the record's as-run values (fix 9)
LOO = {"crps": ZS["day__p_loo_crps"][ROWS, GI].astype(np.float64).sum(0), "pair": X2["day__p_loo_pair"][ROWS, GI].astype(np.float64).sum(0)}


def crps4(c, cv):
    pc, pcl, pp = SP["p_crps"], SP["p_clim_crps"], SP["p_pair"]
    pt = float(100 * (div(pc[0] + c * pp[0], pcl[0]) - 1)); bs = 100 * (div(pc[1] + c * pp[1], pcl[1]) - 1)
    lo, hi = pctl(bs)
    jk = 100 * (div(LOO["crps"] + c * LOO["pair"], pcl[0]) - 1)
    vj = float(3 / 4 * ((jk - jk.mean()) ** 2).sum()); vq = float((100 * cv * (1 + pt / 100)) ** 2)
    return pt, [lo, hi], widen(pt, lo, hi, float(np.nanvar(bs, ddof=1)), max(np.nan_to_num(vq), np.nan_to_num(vj))), vj, vq


K2R = RL["stations_reliability"]["k2_cal_station_pixels"]["1"]["measurements"][A_]
cv8 = float(np.sqrt(K2R["crps_8"]["v_quart"]) / abs(K2R["crps_8"]["quartet_mean"]))
p8 = crps4(1 / 8, cv8); r8 = S1["p_crps_vs_climens|8"]
check("four members, CRPS with pair x 1/8 by this code (expected-CRPS code path): point, bootstrap interval, interval, V_jack4, V_quart",
      [p8[0]] + p8[1] + p8[2] + [p8[3], p8[4]], [r8["v"]] + r8["ci_boot"] + r8["ci"] + [r8["v_jack4"], r8["v_quart"]])
K2C4 = T16M["k2_crps4_cal_lead1_measurements_all"]
P4 = crps4(1 / 4, float(np.sqrt(K2C4["v_quart"]) / abs(K2C4["quartet_mean"])))
CU[("crps", 1)] = {"as_run": REL4["p_crps_vs_climens"], "as_run_printed": (P4[0], P4[2]), "M8": (c8, R16["p_crps_vs_climens|8"]["ci"]),
                   "M16": (y16["v"], y16["ci"]), "inf": (ci_, R16["p_crps_vs_climens|inf"]["ci"])}

# ------------------------------------------------------------------------------------------------ 3. class-U rows, identity
UROWS = [  # key, lead, record quantity, decimals, neg (reduction), label in the table
    ("rmse", 1, "RMSE of the ensemble mean 1 day ahead (ug m-3)", 2, False, "RMSE of the ensemble mean (\\textmu g\\,m$^{-3}$)"),
    ("rmse", 3, "RMSE 3 days ahead (ug m-3)", 2, False, "\\quad 3 days ahead"),
    ("cams", 1, "RMSE reduction against the CAMS forecast, 1 day ahead (%)", 1, True, "RMSE reduction against the CAMS forecast (\\%)"),
    ("cams", 3, "RMSE reduction against the CAMS forecast / Aurora, 3 days ahead (%): CAMS", 1, True, "\\quad 3 days ahead"),
    ("aurora", 1, "RMSE reduction against Aurora, 1 day ahead (%)", 1, True, "RMSE reduction against Aurora (\\%)"),
    ("aurora", 3, "RMSE reduction against Aurora, 3 days ahead (%)", 1, True, "\\quad 3 days ahead"),
    ("ssr", 1, "Spread/skill against the measurements, 1 day ahead", 2, False, "Spread/skill against the measurements"),
    ("crps", 1, "CRPS reduction against the climatological ensemble, 1 day(s) ahead (%)", 1, True,
     "CRPS reduction against the climatological ensemble (\\%)"),
]
PLAIN_U = {"rmse": "RMSE of the ensemble mean at the stations (ug m-3)", "cams": "RMSE reduction against the CAMS forecast at the stations (%)",
           "aurora": "RMSE reduction against Aurora at the stations (%)", "ssr": "spread/skill against the measurements at the stations",
           "crps": "CRPS reduction against the climatological ensemble at the stations (%)"}
FORMS = ("as_run", "M8", "M16", "inf")
IDU, CELLS_U = [], []
for key, h, qrec, d, neg, lab in UROWS:
    hit = [r for r in TAB["rows"] if r["quantity"] == qrec]
    assert len(hit) == 1, (qrec, len(hit))
    rec = hit[0]; cell = {}; eq = {}
    for f in FORMS:
        v, ci = CU[(key, h)][f]
        if neg:
            v = -v; ci = None if ci is None else sorted([-ci[0], -ci[1]])
        cell[f] = (v, ci)
        if f != "M16":
            mine = vci(v, ci, d)
            want = rec[f].rstrip("*")
            eq[f] = {"record": rec[f], "mine": mine, "equal": mine == want,
                     "sixteen_members_marker": rec[f].endswith("*") if f != "as_run" else None}
            if mine != want:                    # not reproduced: not printed
                cell[f] = (None, None)
    printed = None
    if "as_run_printed" in CU[(key, h)]:        # CRPS: the record's as-run cell (fair CRPS of the four) is reproduced, not printed
        v, ci = CU[(key, h)]["as_run_printed"]
        if neg:
            v = -v; ci = sorted([-ci[0], -ci[1]])
        cell["as_run"] = (v, ci) if eq["as_run"]["equal"] else (None, None)
        printed = {"value": v, "interval": ci, "string": vci(v, ci, d),
                   "note": "printed under '4 (as run)': the expected CRPS reduction of four members (fair CRPS + pair/4 of the four members of "
                           "the evaluation), not in the record; its code path at pair x 1/8 reproduces the record's four-member eight-member value"}
    IDU.append({"quantity": qrec, "lead_days": h, "record_section": rec["section"][:60], "cells": eq, "as_run_printed_instead": printed,
                "M16": {"value": cell["M16"][0], "interval": cell["M16"][1], "string": vci(cell["M16"][0], cell["M16"][1], d),
                        "note": "t16.npz (pass16.py): the record's bootstrap and jackknife at M = 16 on the sixteen-member per-day sums, whose "
                                "M = 8 and infinite values and intervals reproduce the record (proof in identity.json, 'sixteen_member_sums')"}})
    CELLS_U.append((key, h, d, lab, cell))

# ------------------------------------------------------------------------------------------------ 4. class E (conf149, > 37.5, lead 1)
BE = CE["results"]["world"]["conf149"]["thr37.5"]["lead1"]
NSTART = int(BE["info"]["n_start_days"])
EQS = [("warned_M", "People warned (millions a day)", "people warned (millions a day)"), ("reach_pct", "Reach (\\%)", "reach (%)"),
       ("correct_pct", "Correct warnings (\\%)", "share of correct warnings (%)"),
       ("hidden_warned_M", "Hidden people warned (millions a day)", "hidden people warned (millions a day)")]
FMT = {"warned_M": 1, "reach_pct": 2, "correct_pct": 2, "hidden_warned_M": 2}                   # classE.py FMT


def md_table(md):
    """rows of the table '> 37.5 ug m-3, lead 1 d' of the confirmatory section of the class-E markdown"""
    a = md.index("\n## confirmatory"); b = md.find("\n## ", a + 5); sec = md[a:b if b > 0 else None]
    c = sec.index("\n### > 37.5 ug m-3, lead 1 d"); d = sec.find("\n### ", c + 5); blk = sec[c:d if d > 0 else None]
    return {r[0]: r[1:] for r in ([x.strip() for x in line.strip().strip("|").split("|")] for line in blk.split("\n") if line.startswith("| ") and " as issued " in line)}


def ecell(e, d):                                        # classE._cell
    lo, hi = e["interval95"]; st = e.get("interval_status") or ""
    mark = ("" if st == "inflated" else "‡" if st.startswith("NOT inflated") else "") + (f"†{e['resamples_nan']}" if "truncated" in st else "")
    return f"{f_(e['value'], d)} [{f_(lo, d)}, {f_(hi, d)}]{mark}"


MDT = md_table(MD)
IDE, CELLS_E = [], []
for m, lab, plain in EQS:
    e = BE["quantities"][f"issued:{m}"]; d = FMT[m]; inf = e["infinite"]
    mine = [ecell(e["M8_liberal"], d), ecell(e["M16_liberal_measured"], d), ecell(e["M4_stored"], d), f_(inf["point"], d),
            f"[{f_(inf['range'][0], d)}, {f_(inf['range'][1], d)}]" + ("" if not inf["undefined_fits"] else f" ({6 - len(inf['undefined_fits'])} of 6)"),
            f"[{f_(inf['interval95'][0], d)}, {f_(inf['interval95'][1], d)}]" + ("†" if inf["interval_parts_truncated"] else "")]
    want = MDT.get(f"{m} as issued")
    ok = want is not None and want == mine
    cell = {"as_run": (e["M4_stored"]["value"], e["M4_stored"]["interval95"]), "M8": (e["M8_liberal"]["value"], e["M8_liberal"]["interval95"]),
            "M16": (e["M16_liberal_measured"]["value"], e["M16_liberal_measured"]["interval95"]),
            "inf": (inf["point"], inf["interval95"]), "range": inf["range"]}
    status = {f: e[k].get("interval_status") for f, k in (("as_run", "M4_stored"), ("M8", "M8_liberal"), ("M16", "M16_liberal_measured"))}
    IDE.append({"quantity": f"issued:{m}", "record_markdown": want, "mine": mine, "equal": ok, "interval_status": status,
                "infinite_point_label": inf["point_label"], "range_over": inf["range_over"], "undefined_fits": inf["undefined_fits"]})
    if not ok:
        cell = {k: (None, None) for k in ("as_run", "M8", "M16", "inf")}; cell["range"] = None
    CELLS_E.append((m, d, lab, plain, cell, status))

ALL_EQUAL = all(c["equal"] for r in IDU for c in r["cells"].values()) and all(r["equal"] for r in IDE)
ALL_CHECKS = all(c["passed"] for c in CHECKS)

# ------------------------------------------------------------------------------------------------ 5. K2 sentence (stored quartet), from the record
K2FLAG = [x for x in TAB.get("k2", []) if x.get("classU") and x.get("flag_2p5") and str(x.get("set", "")).startswith("station pixels")
          and "vs measurements" in str(x.get("set", "")) and x.get("stat") in ("spread", "ssr_inf")]

# ------------------------------------------------------------------------------------------------ 6. d5.json
def tri(name):
    return [name, name + ", 95% interval low", name + ", 95% interval high"]


COLS = (["block of rows", "quantity"] + tri("4 members (as run)") + tri("8 members") + tri("16 members") + tri("infinitely many members")
        + ["infinitely many members, range of the six extrapolations, low", "infinitely many members, range of the six extrapolations, high"])
ROWS_J = []
BU = "Stations of 2022, 1 day ahead unless stated"
for key, h, d, lab, cell in CELLS_U:
    r = [BU, PLAIN_U[key] + (f", {h} days ahead" if h != 1 else ", 1 day ahead")]
    for f in FORMS:
        v, ci = cell[f]
        r += [rnd(v, d), rnd(ci[0], d) if ci else None, rnd(ci[1], d) if ci else None]
    ROWS_J.append(r + [None, None])
BE_LAB = f"Warnings above 37.5 µg m-3, 1 day ahead, {NSTART} start days (at least half of the members above the threshold)"
for m, d, lab, plain, cell, status in CELLS_E:
    r = [BE_LAB, plain]
    for f in FORMS:
        v, ci = cell[f]
        r += [rnd(v, d), rnd(ci[0], d) if ci else None, rnd(ci[1], d) if ci else None]
    rg = cell["range"]
    ROWS_J.append(r + [rnd(rg[0], d) if rg else None, rnd(rg[1], d) if rg else None])
assert all(len(r) == len(COLS) for r in ROWS_J)
FAIR4 = REL4["p_crps_vs_climens"]                       # fair CRPS of the four (the record's as-run value), for the description
NOISE_J = ("the noise of each member is drawn anew at every start day; intervals: the bootstrap percentile intervals" if VDRAW0 else
           "the noise field of each member is the same for every start day and lead time; intervals widened for the draw of the fields, "
           "half-widths times sqrt(1 + V_draw/V_boot)")
DESC = ("Complete Supplementary Table: scores of Foehn in 2022 for 4, 8 and 16 members and for infinitely many members. "
        "4 members: the evaluation as run. 8, 16 and infinitely many members: from the sixteen members run for every start day at lead times of "
        "1 and 3 days. Stations: RMSE of the ensemble mean and its reductions on the station-days of the comparison with the CAMS forecast and "
        f"Aurora issued at the same time ({ND} start days); spread/skill and CRPS on the station-days of the probabilistic scores ({len(ROWS)} start days). "
        "MSE of the mean of M members = MSE of infinitely many + mean member variance / M; spread/skill with the factor sqrt((M+1)/M) for M "
        "members, sqrt(mean member variance)/RMSE for infinitely many; CRPS: the 4-, 8- and 16-member values are the expected CRPS of an "
        "ensemble of that size (fair CRPS + pair term / M; four members: those of the evaluation; 8 and 16: from the sixteen), the infinite value "
        "the fair CRPS of the sixteen; the fair CRPS of the four members of the evaluation, a second estimate of the infinite value, gives a "
        f"reduction of {vci(-FAIR4[0], sorted([-FAIR4[1][0], -FAIR4[1][1]]), 1).replace(' to ', ', ').replace(' (', ' [').replace(')', ']')} %. "
        "Warnings: a resident of a 1 km grid cell with at least "
        "100 residents is warned when at least half of the members exceed 37.5 µg m-3; hidden: the 1 km value exceeds the threshold and the mean "
        f"of the 40 km cell does not; {NSTART} start days (the other {len(ROWS) - NSTART} were examined in a first sixteen-member run); 8 "
        "members: expected values of eight members drawn from the sixteen; 16 members: the sixteen; 4 members: at least 2 of the 4 members of "
        "the evaluation; infinitely many members: point = intercept of the symmetric rule (weight 1/2 at exactly M/2) fitted in 1/M on M = 4..16, "
        "range = the six extrapolations (three rules with the same limit, fits in 1/M and 1/(M+1)), 95% interval = union of their intervals. "
        "95% intervals: moving-block bootstrap over start days (blocks of 7 start days for the station scores, of 4 for the "
        f"warnings; 10,000 resamples); {NOISE_J}.")
D5J = {"label": "si_members", "part": "",
       "what": ("complete table: every value with its 95% interval, and for the warnings with infinitely many members the range of the six "
                "extrapolations; " + ("noise drawn anew at every start day" if VDRAW0 else "noise fields fixed for every start day")),
       "desc": DESC, "cols": COLS, "rows": ROWS_J}

# ------------------------------------------------------------------------------------------------ 7. d5.tex
def tcell(v, ci, d):
    if v is None:
        return "n.r."
    return tnum(v, d) + ("" if ci is None else f" [{tnum(ci[0], d)}, {tnum(ci[1], d)}]")


L = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.4cm}cccc@{}}", "\\toprule",
     " & \\multicolumn{4}{c}{Members} \\\\", "\\cmidrule(l){2-5}", " & 4 (as run) & 8 & 16 & Infinitely many \\\\", "\\midrule",
     "\\multicolumn{5}{@{}l}{\\textit{At the stations, 1 day ahead unless stated}} \\\\"]
for key, h, d, lab, cell in CELLS_U:
    L.append(lab + " & " + " & ".join(tcell(*cell[f], d) for f in FORMS) + " \\\\")
L += ["\\addlinespace", "\\multicolumn{5}{@{}l}{\\textit{Warnings above 37.5\\,\\textmu g\\,m$^{-3}$, 1 day ahead, %d start days}} \\\\" % NSTART]
DAG = False
for m, d, lab, plain, cell, status in CELLS_E:
    cs = []
    for f in ("as_run", "M8", "M16"):
        c = tcell(*cell[f], d)
        if status[f] and "truncated" in status[f]:
            c += "$^\\dagger$"; DAG = True
        cs.append(c)
    v, rg = cell["inf"][0], cell["range"]
    cs.append("n.r." if v is None else f"{tnum(v, d)} ({tnum(rg[0], d)}--{tnum(rg[1], d)})")
    L.append(lab + " & " + " & ".join(cs) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
NOISE_T = ("The noise of each member is drawn anew at every start day; the intervals are those of the bootstrap."
           if VDRAW0 else
           "The noise field of each member is the same for every start day and lead time, so the four and the sixteen fields are each one "
           "draw; the intervals are widened for this draw.")
K2_T = (" At the stations the four members of the evaluation are atypically wide compared with the other sets of four of the sixteen "
        "members, which raises their spread and spread/skill." if (K2FLAG and not VDRAW0) else "")      # stored design only
CAP = ("\\textbf{Scores by ensemble size.} Scores of \\model{} in 2022 for ensembles of 4, 8 and 16 members and for infinitely many members. "
       "Four members: the evaluation as run. Eight, sixteen and infinitely many members: from sixteen members run for every start day 1 and 3 days "
       "ahead. The mean-squared error of the mean of $M$ members is that of infinitely many members plus the mean variance of the members "
       "divided by $M$; the spread/skill ratio of $M$ members is the ensemble spread times $\\sqrt{(M+1)/M}$ over the RMSE of their mean. "
       "CRPS: for 4, 8 and 16 members, the expected CRPS of an ensemble of that size, the fair CRPS plus the mean absolute difference between "
       "two members divided by $2M$ (for four members, that of the four members of the evaluation); for infinitely many members, the fair CRPS "
       "of the sixteen. RMSE and its reductions: station-days of the "
       "comparison with the CAMS forecast and Aurora issued at the same time; spread/skill and CRPS: station-days of the probabilistic scores. "
       "Warnings: a resident of a 1\\,km grid cell with at least 100 residents "
       "is warned when at least half of the members exceed the threshold; hidden people, residents whose 1\\,km \\pmtf{} exceeds the threshold "
       f"while the mean of their 40\\,km cell does not; {NSTART} of the {len(ROWS)} start days (the other {len(ROWS) - NSTART} were examined in a "
       "first sixteen-member run). For eight members, the expected values of eight members drawn from the sixteen; for infinitely many "
       "members, the limit extrapolated in $1/M$ from the values for 4 to 16 members, with in parentheses the range of six extrapolations "
       "(three warning rules with the same limit, fits in $1/M$ and $1/(M+1)$); its 95\\% interval is given in the Source Data file. 95\\% intervals in brackets: moving-block "
       "bootstrap over start days (blocks of seven start days for the station scores, four for the warnings; 10,000 resamples). " + NOISE_T + K2_T
       + (" $^\\dagger$Interval truncated (resamples without a matching factor dropped)." if DAG else "")
       + (" n.r., not reproduced from the record." if not ALL_EQUAL else ""))
TEX = ("\\begin{table}[p]\n\\centering\n\\caption{" + CAP + "}\n\\label{tab:si_members}\n\\fitwidth{%\n\\footnotesize%\n"
       "\\setlength{\\tabcolsep}{4.5pt}%\n" + "\n".join(L) + "}\n\\end{table}\n")
FORBIDDEN = ("/scr" + "atch/", "/p" + "fs/", "/proj" + "appl/")    # source_data/build_source_data.py: never in a delivered file
for what, text in (("d5.json", jdump(D5J)), ("d5.tex", TEX)):
    bad = [w for w in FORBIDDEN if w in text.lower()]
    if bad:
        sys.exit(f"forbidden string {bad} in {what}: nothing written")

# ------------------------------------------------------------------------------------------------ 8. identity.json and writes
IDJ = {"design": LABEL, "FOEHN_DESIGN": DESIGN, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
       "inputs": {k: {"path": str(p), "sha256": sha(p)} for k, p in IN.items()},
       "classU": {"rows": IDU, "rule": "4 / 8 / infinite cells formatted with make_table.vci must equal the strings of classU_table_179.json "
                                       "(the trailing '*' of the record marks sixteen-member values)"},
       "classE": {"sample": "conf149", "threshold": 37.5, "lead_days": 1, "n_start_days": NSTART, "bootstrap": BE["info"]["bootstrap"], "rows": IDE,
                  "rule": "values and intervals copied from classE_all2022.json; formatted with classE._cell they must equal the markdown table of the record"},
       "recomputations": CHECKS, "all_cells_equal": ALL_EQUAL, "all_recomputations_passed": ALL_CHECKS,
       "k2_station_flags_used_in_caption": [{k: x.get(k) for k in ("set", "lead", "stat", "z")} for x in K2FLAG],
       "sixteen_member_sums": {"file": str(IN["t16"]), "built_by": "pass16.py", "built": T16M["built"], "tags": T16M["tags"],
                               "scorer_statements_executed": T16M["statements_executed"], "proof_all_passed": T16M["all_passed"],
                               "proof": T16M["proof"], "results16": R16M},
       "crps_four_members": {"printed": "expected CRPS reduction of four members: fair CRPS + pair/4 of the four members of the evaluation",
                             "value_interval": [-P4[0], sorted([-P4[2][0], -P4[2][1]])], "bootstrap_interval": sorted([-P4[1][0], -P4[1][1]]),
                             "V_jack4": P4[3], "V_quart": P4[4], "V_draw_used": (0.0 if VDRAW0 else max(P4[3], P4[4])),
                             "quartet_statistic": K2C4, "fair_crps_of_the_four_as_run": [-FAIR4[0], sorted([-FAIR4[1][0], -FAIR4[1][1]])]},
       "design_note": NOISE_J}
if not ALL_CHECKS:
    for c in CHECKS:
        if not c["passed"]:
            print("FAILED:", c)
    sys.exit("a recomputation check failed: nothing written")
OUT.mkdir(parents=False, exist_ok=True)
safe_write(OUT / "d5.json", jdump(D5J))
safe_write(OUT / "d5.tex", TEX)
safe_write(OUT / "identity.json", jdump(IDJ))
print(f"design {LABEL}: cells equal to the record {ALL_EQUAL}; recomputations passed {ALL_CHECKS} ({sum(c['passed'] for c in CHECKS)} of {len(CHECKS)})")
for r in ROWS_J:
    print(" | ".join("" if x is None else str(x) for x in r[1:]))
print("wrote", OUT / "d5.json", OUT / "d5.tex", OUT / "identity.json")
