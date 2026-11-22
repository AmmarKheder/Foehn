#!/usr/bin/env python3
"""D5: the per-start-day sixteen-member sums at the stations, their proof against the A14 record, the 16-member intervals.

    python -B pass16.py                    # FOEHN_DESIGN unset: stored records -> out_stored/t16.npz, then build_d5.py
    FOEHN_DESIGN=ar python -B pass16.py    # 'ar' records -> out_ar/t16.npz, then build_d5.py

The sixteen-member sums against the measurements are rebuilt by the scorers' own code, read-only:
  - P/a14/classU/scripts/st_cf_score.py: its top-level statements from 'Z = np.load(DATA / "st_cf_terms.npz")' to
    'D16 = sorted(have16)' are executed unchanged (taken from the source by ast) in a namespace of this script: the T16 sums
    on compare_fair's station-days, and the scorer's own sys_day, rmse_boot, boot_ratio, ci and bootstrap index;
  - P/a14/classU/scripts/rel_score.py: its statements from 'st = np.load(G / "stations_2022.npz")' to the end of the loop
    'for tag in TAGS:' (ens_terms, mix_sums, the ACC16 sums of the reliability samples and the K2 sums SUF), unchanged;
  - u_common.py (jk_var, inflate), rel_score.py (cil, form), u_quart.py (stats, k2), rel_common.py (block_weights): single
    functions taken from the unchanged sources by ast (u_common itself is never imported: it would create folders).
The station measurements are read only by those statements of the scorers. Nothing under P/a14 or analyses/ is written.
Proof (written into t16.npz, key 'meta'): the totals equal the record (st_cf_classU_179.json, rel_classU_179.json), and the
record's bootstrap and jackknife applied to the saved per-day sums at M = 8 and M = infinity reproduce its values and
intervals. Then the same expressions at M = 16 (MSE_16 = MSE_inf + s2/16: num = sse - s2M + s2raw/16; CRPS + pair/16;
spread/skill with sqrt(17/16)), with the record's bootstrap, the 16-field jackknife and, under the stored design, its
widening. Also the quartet distribution of the expected four-member CRPS (fair + pair/4) for the stored-design widening of
the four-member CRPS cell (u_quart.stats, k2; checked on crps_fair and crps_8 against the record's K2)."""
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

T0 = time.time()
HERE = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/print/d5"))
NAT = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms"))
P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
DESIGN = os.environ.get("FOEHN_DESIGN", "")
if DESIGN not in ("", "ar"):
    sys.exit(f"FOEHN_DESIGN={DESIGN!r}: pass16.py knows only unset (stored) and 'ar'")
LABEL, SFX, VDRAW0 = ("ar", "_ar", True) if DESIGN == "ar" else ("stored", "", False)
OUT = HERE / f"out_{LABEL}"
U = P / "a14" / "classU"; SCR = U / "scripts"; DATA = U / f"data{SFX}"; RES = U / f"results{SFX}"
REL = NAT / "analyses" / "reliability"; RELD = REL / f"data{SFX}"
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
TAGS = ["run16"] if VDRAW0 else ["run16", "r1", "t0"]          # r1_pipeline_b.py: the tags of st_cf_score / rel_score
IN = {"st_cf_score.py": SCR / "st_cf_score.py", "rel_score.py": SCR / "rel_score.py", "u_quart.py": SCR / "u_quart.py",
      "u_common.py": SCR / "u_common.py", "rel_common.py": REL / "scripts" / "rel_common.py",
      "st_cf_terms": DATA / "st_cf_terms.npz", "rel_extra": DATA / "rel_extra.npz", **{f"st16_{t}": DATA / f"st16_{t}.npz" for t in TAGS},
      "st_final": RELD / "st_final.npz", "sums_final": RELD / "sums_final.npz", "st_cf_identity": RES / "st_cf_identity.json",
      "st_cf_record": RES / "st_cf_classU_179.json", "rel_record": RES / "rel_classU_179.json",
      "stations (read by the scorers' statements)": G / "stations_2022.npz", "build_d5.py": HERE / "build_d5.py"}
MISSING = [str(p) for p in IN.values() if not p.exists()]
if MISSING:
    sys.exit("pass16.py (FOEHN_DESIGN=%r): inputs missing, nothing written:\n  %s" % (DESIGN, "\n  ".join(MISSING)))
ST = json.loads(IN["st_cf_record"].read_text()); RL = json.loads(IN["rel_record"].read_text())
for name, rec in (("st_cf_record", ST), ("rel_record", RL)):
    if (rec.get("design") or {}).get("FOEHN_DESIGN", "") != DESIGN:
        sys.exit(f"{IN[name]} is not of design {DESIGN!r}: nothing written")
TOL = 1e-9
A_ = "All official networks"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fns(src, names, ns):
    """single functions of an unchanged source, defined alone in ns (nothing else of the file runs)"""
    found = []
    for node in ast.parse(Path(src).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], type_ignores=[]), str(src), "exec"), ns); found.append(node.name)
    assert sorted(found) == sorted(names), (src, found, names)
    return ns


def statements(src, first, last):
    """the unchanged top-level statements of src from the one whose source starts with `first` to the first later one whose
    source starts with `last`; refused if they could write"""
    text = Path(src).read_text(); body = ast.parse(text).body
    seg = lambda n: ast.get_source_segment(text, n)                                     # noqa: E731
    i0 = next(i for i, n in enumerate(body) if seg(n).startswith(first))
    i1 = next(i for i in range(i0, len(body)) if seg(body[i]).startswith(last))
    code = "\n".join(seg(n) for n in body[i0:i1 + 1])
    for bad in ("safe_json", "safe_npz", "safe_text", "np.save", "open(", "mkdir", "unlink", "replace(", "rename", "write"):
        assert bad not in code, (str(src), bad)
    return compile(ast.Module(body[i0:i1 + 1], type_ignores=[]), str(src), "exec"), [body[i0].lineno, body[i1].end_lineno]


CHECKS = []


def check(name, mine, rec, tol=TOL):
    a, b = np.asarray(mine, float), np.asarray(rec, float)
    r = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) if a.shape == b.shape else float("inf")
    CHECKS.append({"check": name, "max_rel_diff": r, "passed": bool(r <= tol)})
    return r <= tol


# ------------------------------------------------------------------------------------------------ helpers of the record
H = fns(IN["u_common.py"], ["jk_var", "inflate"], {"np": np, "VDRAW0": VDRAW0})
fns(IN["rel_score.py"], ["cil", "form"], H)
fns(IN["u_quart.py"], ["stats", "k2"], H); H["F_FPC"] = 16.0 / 12.0
fns(IN["rel_common.py"], ["block_weights"], H)
jk_var, inflate, form, stats, k2, block_weights = (H[k] for k in ("jk_var", "inflate", "form", "stats", "k2", "block_weights"))
QUARTETS = np.array(list(combinations(range(16), 4)), np.int64)

# ------------------------------------------------------------------------------------------------ 1. compare_fair station-days
code, LINES_CF = statements(IN["st_cf_score.py"], 'Z = np.load(DATA / "st_cf_terms.npz")', "D16 = sorted(have16)")
NC = {"np": np, "json": json, "os": os, "sys": sys, "DATA": DATA, "RES": RES, "G": G, "RELD": RELD, "TAGS": TAGS, "__name__": "st_cf_score_part"}
exec(code, NC)
print(f"st_cf_score.py lines {LINES_CF} executed: {len(NC['D16'])} sixteen-member forecasts, {time.time() - T0:.0f} s", flush=True)
ok_d, T16, DAYS_OK = NC["ok_d"], NC["T16"], NC["DAYS_OK"]
sys_day, rmse_boot, boot_ratio, ci = NC["sys_day"], NC["rmse_boot"], NC["boot_ratio"], NC["ci"]
REFS = {"CAMS d+1 (equal issuance)": 4, "Aurora 00 UTC d+1 (equal issuance)": 6, "climatology": 9, "GHAP persistence (start-day estimate)": 8}
SAVE = {"cf_start_days": np.array(DAYS_OK, np.int64)}
RES16 = {}
dr = ST["dropped_16_station_days"]; DR = NC["DROP"]
check("compare_fair: station-days kept with four members, unexpected, forecasts (totals)",
      [sum(v["fallback4"] for v in DR.values()), sum(v["unexpected"] for v in DR.values()), len(DR)], [dr["fallback4_total"], dr["unexpected_total"], dr["forecasts"]], 0)
for h in (1, 3):
    li = (1, 3).index(h)
    sub = ST["groups"]["all"][str(h)]["Foehn 1 km"]["sixteen_members_provisional"]
    dmask = np.array([(d, h) in NC["have16"] for d in DAYS_OK]); assert dmask.all(), "a start day without sixteen members"
    q = {kk: NC["gsum"](T16[kk], [6])[ok_d, li] for kk in T16}
    for kk in ("n", "sse", "s2M", "s2raw", "n16", "jk_num", "jk_num8"):
        SAVE[f"cf_{h}_{kk}"] = q[kk]
    N_ = q["n"].sum()
    check(f"compare_fair lead {h}: station-days, station-days with 16 (totals)", [N_, q["n16"].sum()], [sub["station_days"], sub["station_days_with_16"]], 0)
    jki = np.sqrt(q["jk_num"].sum(0) / N_); jk8 = np.sqrt(q["jk_num8"].sum(0) / N_)
    jk16 = np.sqrt((q["jk_num"] + (q["jk_num8"] - q["jk_num"]) / 2).sum(0) / N_)   # replicate of the M = 16 form: s2/16 = (s2/8)/2
    check(f"compare_fair lead {h}: sixteen_inf, sixteen_M8, jk16 sd (totals)",
          [np.sqrt((q["sse"].sum() - q["s2M"].sum()) / N_), np.sqrt((q["sse"].sum() - q["s2M"].sum() + q["s2raw"].sum() / 8) / N_),
           np.sqrt(jk_var(jki)), np.sqrt(jk_var(jk8))], [sub["sixteen_inf"], sub["sixteen_M8"], sub["jk16_sd_inf"], sub["jk16_sd_M8"]])
    n_ = q["n"]
    for form_, num, reps in (("inf", q["sse"] - q["s2M"], jki), ("M8", q["sse"] - q["s2M"] + q["s2raw"] / 8, jk8),
                             ("M16", q["sse"] - q["s2M"] + q["s2raw"] / 16, jk16)):
        pt, bs = rmse_boot(num, n_); lo, hi = ci(bs); vj = float(jk_var(reps))
        cc_, fac = inflate(pt, lo, hi, float(np.nanvar(bs, ddof=1)), vj)
        e = {"v": pt, "ci_boot": [lo, hi], "ci": cc_, "factor": fac, "v_jack16": vj}
        if form_ != "M16":
            r = sub[f"rmse16_{form_}"]
            check(f"compare_fair lead {h}: rmse16_{form_} point, bootstrap interval, interval, factor, V_jack16",
                  [e["v"]] + e["ci_boot"] + e["ci"] + [e["factor"], e["v_jack16"]], [r["v"]] + r["ci_boot"] + r["ci"] + [r["factor"], r["v_jack16"]])
        else:
            RES16[f"rmse|{h}"] = e
        for rname, j in REFS.items():
            sb, cb = sys_day(j, [6], h)
            if j in (4, 6):
                SAVE[f"cf_{h}_ref{j}_sse"], SAVE[f"cf_{h}_ref{j}_n"] = sb, cb
            pt, bs = boot_ratio(num, n_, sb, cb); lo, hi = ci(bs)
            rr = np.sqrt(np.nansum(sb) / np.nansum(cb))
            cc_, fac = inflate(100 * pt, 100 * lo, 100 * hi, float(np.nanvar(100 * bs, ddof=1)), vj / rr ** 2 * 1e4)
            e = {"v": 100 * pt, "ci_boot": [100 * lo, 100 * hi], "ci": cc_, "factor": fac}
            if form_ != "M16":
                r = sub[f"gain16_{form_}_{rname}"]
                check(f"compare_fair lead {h}: gain16_{form_} over {rname}", [e["v"]] + e["ci_boot"] + e["ci"] + [e["factor"]],
                      [r["v"]] + r["ci_boot"] + r["ci"] + [r["factor"]])
            elif j in (4, 6):
                RES16[f"gain|{h}|{rname}"] = e
print(f"compare_fair part done, {time.time() - T0:.0f} s", flush=True)

# ------------------------------------------------------------------------------------------------ 2. reliability samples
X2 = np.load(IN["rel_extra"])
code, LINES_REL = statements(IN["rel_score.py"], 'st = np.load(G / "stations_2022.npz")', "for tag in TAGS:")
NR = {"np": np, "G": G, "RELD": RELD, "DATA": DATA, "TAGS": TAGS, "X2": X2, "__name__": "rel_score_part"}
exec(code, NR)
print(f"rel_score.py lines {LINES_REL} executed: {len(NR['have16'])} sixteen-member forecasts, {time.time() - T0:.0f} s", flush=True)
ZS = np.load(IN["sums_final"]); GROUPS = [str(x) for x in ZS["groups"]]; g = GROUPS.index(A_)
assert g == 6 and np.array_equal(ZS["keys"], X2["keys"])
IQ = {name: j for j, name in enumerate(NR["Q16C"] + NR["Q16P"])}
SAVE["rel_quantities"] = np.array(NR["Q16C"] + NR["Q16P"])
R16 = RL["stations_reliability_16"]["by_lead"]


def forms(S, Rs, c):
    """rel_score.py forms16 (nested in its ST16 loop) for the keys of D5, at c = 1/M' (0: infinitely many members);
    the record's expressions with c8 -> c and 9/8 -> 1 + c"""
    q = lambda name: S[..., IQ[name], :]                                                  # noqa: E731
    n, s2, s2M, pair = q("n"), q("s2"), q("s2M"), q("pair")
    mse = (q("sse_0") - s2M + c * s2) / n
    o = {"rmse|0": np.sqrt(mse), "ss|0": np.sqrt((1 + c) * s2 / n) / np.sqrt(mse),
         "crps_vs_clim|0": 100 * ((q("crps_0") + c * pair) / Rs["c_crps_0"][..., None] - 1)}
    num = q("p_sse") - q("p_s2M") + c * q("p_s2")
    o["p_rmse"] = np.sqrt(num / q("p_n"))
    o["p_crps_vs_climens"] = 100 * ((q("p_crps") + c * q("p_pair")) / Rs["p_clim_crps"][..., None] - 1)
    return o


def rkey(k, f):                                                                            # record key of a form
    return f"{k.split('|')[0]}|{f}" + ("|0" if k.endswith("|0") else "")


dr16 = RL["stations_reliability_16"]["dropped"]
for h in (1, 3):
    lst = sorted(NR["ACC16"][h], key=lambda x: x[0])
    check(f"reliability lead {h}: forecasts, fallback station-days, unexpected (totals)",
          [len(lst), sum(x[3] for x in lst), sum(x[4] for x in lst)], [dr16[str(h)]["forecasts"], dr16[str(h)]["fallback_station_days"], dr16[str(h)]["unexpected"]], 0)
    rows16 = np.array([x[1] for x in lst]); Ag = np.stack([x[2] for x in lst])[:, g]          # (forecasts, quantities, 17)
    Rf = {k: ZS["day__" + k][rows16, g] for k in ("c_crps_0", "p_clim_crps")}
    okc, okp = ZS["day__n"][rows16, g] > 0, ZS["day__p_n"][rows16, g] > 0
    SAVE[f"rel_{h}_start_days"] = np.array([x[0] for x in lst], np.int64); SAVE[f"rel_{h}_rows"] = rows16; SAVE[f"rel_{h}_A"] = Ag
    for k, v in Rf.items():
        SAVE[f"rel_{h}_{k}"] = v
    SAVE[f"rel_{h}_okc"], SAVE[f"rel_{h}_okp"] = okc, okp
    Wc = block_weights(int(okc.sum()), B=7, n=10000, seed=0); Wp = block_weights(int(okp.sum()), B=7, n=10000, seed=0)
    rec = R16[str(h)][A_]
    with np.errstate(all="ignore"):
        for f, c in (("inf", 0.0), ("8", 1 / 8), ("16", 1 / 16)):
            pt = forms(Ag.sum(0), {k: v.sum(0) for k, v in Rf.items()}, c)
            bc = forms(np.tensordot(Wc, Ag[okc], axes=(1, 0)), {k: np.tensordot(Wc, v[okc], axes=(1, 0)) for k, v in Rf.items()}, c)
            bp = forms(np.tensordot(Wp, Ag[okp], axes=(1, 0)), {k: np.tensordot(Wp, v[okp], axes=(1, 0)) for k, v in Rf.items()}, c)
            for k, val in pt.items():
                full = float(val[0]); reps = np.asarray(val[1:], float)
                vj = float((15 / 16) * ((reps - reps.mean()) ** 2).sum())
                e = form(full, (bp if k.startswith("p_") else bc)[k][..., 0], vj); e["v"] = full; e["v_jack16"] = vj
                if f != "16":
                    r = rec[rkey(k, f)]
                    check(f"reliability lead {h}: {rkey(k, f)} point, V_jack16, bootstrap interval, interval",
                          [e["v"], e["v_jack16"]] + e["ci_boot"] + e["ci"], [r["v"], r["v_jack16"]] + r["ci_boot"] + r["ci"])
                else:
                    RES16[f"{k}|{h}"] = e
print(f"reliability part done, {time.time() - T0:.0f} s", flush=True)

# ------------------------------------------------------------------------------------------------ 3. quartets: expected CRPS of four members
S_ = NR["SUF"][(1, 0, 6)]                                    # lead 1, measurements, all official networks (cal sample)
Pm = S_["P"] + S_["P"].T
sq = stats(S_["D"], S_["n"], QUARTETS, S_["A"], Pm); sa = stats(S_["D"], S_["n"], np.arange(16)[None], S_["A"], Pm)
k2r = RL["stations_reliability"]["k2_cal_station_pixels"]["1"]["measurements"][A_]
for key in ("crps_fair", "crps_8"):
    mine = k2(sq[key], sa[key][0], sq[key][0])
    check(f"K2 lead 1 measurements {key}: q0123, all16, quartet mean, v_quart", [mine[x] for x in ("q0123", "all16", "quartet_mean", "v_quart")],
          [k2r[key][x] for x in ("q0123", "all16", "quartet_mean", "v_quart")])
c4q = sq["crps_fair"] + 2 * (sq["crps_8"] - sq["crps_fair"]); c4a = sa["crps_fair"] + 2 * (sa["crps_8"] - sa["crps_fair"])   # + pair/4
K2C4 = k2(c4q, c4a[0], c4q[0])

# ------------------------------------------------------------------------------------------------ 4. t16.npz, then the table
ALL = all(c["passed"] for c in CHECKS)
META = {"design": LABEL, "FOEHN_DESIGN": DESIGN, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - T0, 1),
        "tags": TAGS, "statements_executed": {"st_cf_score.py": LINES_CF, "rel_score.py": LINES_REL},
        "inputs": {k: ({"path": str(p), "sha256": sha(p)} if not (k.startswith("stations") or k == "st_final") else   # station-pixel
                       {"path": str(p), "bytes": p.stat().st_size, "mtime": p.stat().st_mtime})       # files: opened by the scorers' statements only
                   for k, p in IN.items()},
        "proof": CHECKS, "all_passed": ALL, "results16": RES16, "k2_crps4_cal_lead1_measurements_all": K2C4,
        "note": "cf_<lead>_*: compare_fair's station-days, group 'all', per ok start day (cf_start_days); rel_<lead>_A: per forecast "
                "(rel_<lead>_start_days, rows rel_<lead>_rows of sums_final), group 'All official networks', quantities rel_quantities, "
                "column 0 the sixteen members (four stored for the stations without them), columns 1-16 the delete-one-field replicates"}
SAVE["meta"] = np.array(json.dumps(META, default=float))
OUT.mkdir(exist_ok=True)
dst = OUT / "t16.npz"; tmp = OUT / f"t16.npz.tmp{os.getpid()}.npz"
np.savez(tmp, **SAVE)
fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
with np.load(tmp) as z:
    assert sorted(z.files) == sorted(SAVE) and all(np.array_equal(z[k], SAVE[k]) for k in SAVE), "t16 read-back differs"
os.replace(tmp, dst)
with np.load(dst) as z:
    assert all(np.array_equal(z[k], SAVE[k]) for k in SAVE), "t16 read-back differs"
print(f"proof: {sum(c['passed'] for c in CHECKS)} of {len(CHECKS)} checks passed; wrote {dst} ({time.time() - T0:.0f} s)", flush=True)
for c in CHECKS:
    if not c["passed"]:
        print("FAILED:", c, flush=True)
if not ALL:
    sys.exit("proof failed: the table is not rebuilt")
subprocess.run([sys.executable, "-B", str(HERE / "build_d5.py")], check=True, env=dict(os.environ))
