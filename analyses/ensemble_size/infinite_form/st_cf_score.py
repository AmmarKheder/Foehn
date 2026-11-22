#!/usr/bin/env python3
"""A14 class U at the stations for the quantities of experiments/figures/compare_fair.py (RMSE of the ensemble mean at the
stations 1-4 days ahead; gains over the CAMS forecast and Aurora issued at the same time, the climatology, the persistences;
the 0.25 / 0.5 deg averages), from data/st_cf_terms.npz (st_cf_pass.py) and, at leads 1 and 3, the sixteen members at the
station pixels (data/st16_<tag>.npz, pop_pass.py run16).
As run: compare_fair's bootstrap verbatim (default_rng(0), blocks of 7 start days, 10,000 resamples) on its PD/PDN; the
printed lines of compare_fair must be reproduced. M = 8 and infinity: A14 4a with the four stored members (all 178 days,
'four members') and, where the sixteen-member run has the forecast, the sixteen members (57 stations without them keep their
four stored members); intervals A14 4c (NOTES.md items 3-4).
    python st_cf_score.py [st16 tag ...]   -> results/st_cf_classU.json"""
import json
import os
import sys

import numpy as np

from u_common import DATA, DESIGN_NOTE, F_FPC, G, P, QUARTETS, REL, RELD, RES, inflate, jk_var, safe_json
from u_quart import full_report, stats

DRY179 = os.environ.get("A14U_DRY179") == "1"     # dry run of r1_pipeline_b.py only: the 179-day branch forced on the days present (run16 standing in for R1)

TAGS = sys.argv[1:] or ["run16"]
Z = np.load(DATA / "st_cf_terms.npz")
days = [int(x) for x in Z["days"]]; PD, PDN = Z["PD"], Z["PDN"]
SYSN = [str(s) for s in Z["systems"]]; NETN = [str(s) for s in Z["nets"]]
T = {k: Z[k] for k in ("n", "sse", "s2", "q25", "q50", "sse_jk", "s2_jk", "bad25", "bad50")}
NS = int(Z["NS"])
MK = {(int(d), int(h)): i for i, (d, h) in enumerate(Z["keys"])}
MASKS = Z["masks"]
# ---- compare_fair's bootstrap, verbatim
rng = np.random.default_rng(0)
ok_d = np.isfinite(PD[0, :, 0, 0])
nd, B = int(ok_d.sum()), 7
idx = (rng.integers(0, nd - B + 1, size=(10000, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(10000, -1)[:, :nd]
DAYS_OK = [d for d, o in zip(days, ok_d) if o]

GROUPS = {name: [g] for g, name in enumerate(NETN)}                     # 6 networks + 'all'
GROUPS["five networks outside Europe"] = [1, 2, 3, 4, 5]
GROUPS["outside China"] = [0, 2, 3, 4, 5]
GROUPS["Korea and Japan"] = [3, 4]
REFS = {"CAMS d+1 (equal issuance)": 4, "Aurora 00 UTC d+1 (equal issuance)": 6, "climatology": 9, "GHAP persistence (start-day estimate)": 8,
        "persistence (station measurement of day d)": 7, "CAMS run d": 3, "Aurora analyses d": 5}


def gsum(arr, gl):
    """per-day array of a group list: arr indexed [g, day, lead, ...]"""
    return arr[gl].sum(0)


def sys_day(k, gl, h):
    """(sse, n) per ok day for system k, group list, lead h (as compare_fair: PD for 'all', PDN by network)"""
    if gl == [6]:
        a = PD[k, ok_d, h - 1]
    else:
        a = PDN[k][gl].sum(0)[ok_d, h - 1]
    return a[:, 0].astype(np.float64), a[:, 1].astype(np.float64)


def boot_ratio(na, ca, nb, cb):
    """compare_fair's statistic: sqrt(sum a / count) / sqrt(sum b / count) - 1, point and the 10,000 resamples"""
    pt = np.sqrt(np.nansum(na) / np.nansum(ca)) / np.sqrt(np.nansum(nb) / np.nansum(cb)) - 1
    bs = np.sqrt(np.nansum(na[idx], 1) / np.nansum(ca[idx], 1)) / np.sqrt(np.nansum(nb[idx], 1) / np.nansum(cb[idx], 1)) - 1
    return float(pt), bs


def rmse_boot(na, ca):
    pt = np.sqrt(np.nansum(na) / np.nansum(ca))
    return float(pt), np.sqrt(np.nansum(na[idx], 1) / np.nansum(ca[idx], 1))


def ci(bs):
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return float(lo), float(hi)


# ---- 1. as run: the printed lines of compare_fair
ident = json.loads((RES / "st_cf_identity.json").read_text())
printed = ident["printed_tail"]
mine = []
for j in range(1, len(SYSN)):
    o = []
    for h in range(1, 5):
        sa, ca = sys_day(0, [6], h); sb, cb = sys_day(j, [6], h)
        pt, bs = boot_ratio(sa, ca, sb, cb); lo, hi = ci(bs)
        o.append(f"J+{h} {100 * pt:+.0f}% [{100 * lo:+.0f},{100 * hi:+.0f}]")
    mine.append(f"{SYSN[0]} vs {SYSN[j]}: " + " ".join(o))
REPRO = {line: (line in printed) for line in mine}
print("as-run lines of compare_fair reproduced:", sum(REPRO.values()), "of", len(REPRO), flush=True)

# ---- 2. sixteen-member terms at the stations (leads 1 and 3), on compare_fair's station-days
st = np.load(G / "stations_2022.npz"); OBS = st["daily"].astype(np.float32); SRC = st["source"]
NETS = {0: "Europe (EEA)", 1: "China (CNEMC)", 3: "USA (EPA)", 4: "Korea (AirKorea)", 5: "Japan (NIES)", 6: "India (CPCB)"}
gid = np.array([list(NETS).index(s) if s in NETS else -1 for s in SRC])
GMASK = [gid == g for g in range(6)] + [np.ones(NS, bool)]
s4 = np.load(RELD / "st_final.npz")                                   # design: reliability/data_ar (S1 step 5)
K4 = {(int(d), int(h)): i for i, (d, h, t) in enumerate(s4["keys"])}
E4ALL = s4["ens"]
ND = len(days)
T16 = {k: np.zeros((7, ND, 2)) for k in ("n", "sse", "s2M", "s2raw", "n16", "sse4", "s2_4", "sse4_16only", "s2_4_16only")}
T16["jk_num"] = np.zeros((7, ND, 2, 16)); T16["jk_num8"] = np.zeros((7, ND, 2, 16))
SUF = {h: {g: dict(D=np.zeros((16, 16)), A=np.zeros(16), P=np.zeros((16, 16)), n=0.0) for g in range(7)} for h in (1, 3)}
have16 = set()
DROP = {}                                    # fix 17: station-days of compare_fair's sample without 16 members at a station that has a populated pixel
for tag in TAGS:
    z = np.load(DATA / f"st16_{tag}.npz")
    STP = z["stpos"]
    for r, (d, h, t) in enumerate(z["keys"]):
        d, h = int(d), int(h)
        if (d, h) not in MK or h not in (1, 3):
            continue
        assert (d, h) not in have16, (d, h)
        have16.add((d, h))
        ok = np.unpackbits(MASKS[MK[(d, h)]])[:NS].astype(bool)
        E16 = z["ens16"][r].astype(np.float32)
        E4 = E4ALL[K4[(d, h)]].astype(np.float32)
        y = OBS[:, d + h].astype(np.float64)
        f16 = ok & np.isfinite(E16).all(1)
        f4 = ok & ~f16
        DROP[(d, h)] = dict(fallback4=int(f4.sum()), unexpected=int((f4 & (STP >= 0)).sum()))
        di = days.index(d); li = (1, 3).index(h)
        x16 = E16.mean(1).astype(np.float64); v16 = E16.var(1, ddof=1).astype(np.float64)
        x4 = E4.mean(1).astype(np.float64); v4 = E4.var(1, ddof=1).astype(np.float64)
        jk16_x = np.stack([np.delete(E16, i, 1).mean(1) for i in range(16)], 1).astype(np.float64)
        jk16_v = np.stack([np.delete(E16, i, 1).var(1, ddof=1) for i in range(16)], 1).astype(np.float64)
        jk4_x = np.stack([np.delete(E4, i, 1).mean(1) for i in range(4)], 1).astype(np.float64)
        jk4_v = np.stack([np.delete(E4, i, 1).var(1, ddof=1) for i in range(4)], 1).astype(np.float64)
        dd = E16.astype(np.float64) - y[:, None]
        for g in range(7):
            m16, m4 = f16 & GMASK[g], f4 & GMASK[g]
            mo = ok & GMASK[g]
            T16["n"][g, di, li] = mo.sum(); T16["n16"][g, di, li] = m16.sum()
            T16["sse"][g, di, li] = ((x16[m16] - y[m16]) ** 2).sum() + ((x4[m4] - y[m4]) ** 2).sum()
            T16["s2M"][g, di, li] = v16[m16].sum() / 16 + v4[m4].sum() / 4
            T16["s2raw"][g, di, li] = v16[m16].sum() + v4[m4].sum()
            T16["sse4"][g, di, li] = ((x4[mo] - y[mo]) ** 2).sum(); T16["s2_4"][g, di, li] = v4[mo].sum()
            T16["sse4_16only"][g, di, li] = ((x4[m16] - y[m16]) ** 2).sum(); T16["s2_4_16only"][g, di, li] = v4[m16].sum()
            for i in range(16):
                a16 = ((jk16_x[m16, i] - y[m16]) ** 2).sum() - jk16_v[m16, i].sum() / 15
                if i < 4:
                    a4 = ((jk4_x[m4, i] - y[m4]) ** 2).sum() - jk4_v[m4, i].sum() / 3; r4 = jk4_v[m4, i].sum()
                else:
                    a4 = ((x4[m4] - y[m4]) ** 2).sum() - v4[m4].sum() / 4; r4 = v4[m4].sum()
                T16["jk_num"][g, di, li, i] = a16 + a4
                T16["jk_num8"][g, di, li, i] = a16 + a4 + (jk16_v[m16, i].sum() + r4) / 8
            Dg = dd[m16]
            S_ = SUF[h][g]; S_["D"] += Dg.T @ Dg; S_["A"] += np.abs(Dg).sum(0); S_["n"] += float(m16.sum())
            Xg = E16[m16].astype(np.float64)
            for i in range(15):
                S_["P"][i, i + 1:] += np.abs(Xg[:, i:i + 1] - Xg[:, i + 1:]).sum(0)
for h in (1, 3):
    for g in range(7):
        P_ = SUF[h][g]["P"]; SUF[h][g]["P"] = P_ + P_.T
D16 = sorted(have16)
print(f"sixteen-member forecasts at the stations: {len(D16)} ({sorted({d for d, _ in D16})[:5]}...)", flush=True)

# ---- 3. K2 at the station pixels (members 0-3 of the 16-member run against the 16) and the quartet CVs
K2 = {}
CV = {}
for h in (1, 3):
    K2[h] = {}
    for g in range(7):
        S_ = SUF[h][g]
        if S_["n"] < 1000:
            continue
        K2[h][NETN[g]] = full_report(S_["D"], S_["n"], S_["A"], S_["P"])
        K2[h][NETN[g]]["n_station_days"] = S_["n"]
        CV[(h, g)] = {k: np.sqrt(K2[h][NETN[g]][k]["v_quart"]) / K2[h][NETN[g]][k]["quartet_mean"] for k in ("rmse_inf", "rmse_8", "rmse_m")}


J_M0 = json.loads((P / "a14" / "classU_review" / "judge" / "j_m0.json").read_text())["ratios"]   # read only (reference j_m0.py)
CVM0 = {L: float(np.std(J_M0[f"lead{L}_all"]["ratio_by_member"]) / np.mean(J_M0[f"lead{L}_all"]["ratio_by_member"])) for L in (1, 3)}


def cv_m0(h):
    """relative sd over the four members of the member-to-ensemble variance ratio (4/3)u_m^2/s2 of cell means (fix 11)"""
    return CVM0[h] if h in (1, 3) else (0.5 * (CVM0[1] + CVM0[3]) if h == 2 else CVM0[3])


def cv_of(h, gl, key):
    """carried coefficient of variation of the quartet draw (NOTES item 4): leads 1/3 their own, lead 2 the mean, 4 that of 3"""
    g = gl[0] if len(gl) == 1 else 6
    if (1, g) not in CV:
        g = 6
    if h in (1, 3):
        return CV[(h, g)][key]
    if h == 2:
        return 0.5 * (CV[(1, g)][key] + CV[(3, g)][key])
    return CV[(3, g)][key]


# ---- 4. forms and intervals
def forms_foehn(k, gl, h):
    """per ok day: numerators of the three forms (as run, M = 8, infinity) for Foehn 1 km (k 0) or its averages (k 1, 2)"""
    sse, n = sys_day(k, gl, h)
    if k == 0:
        s2 = gsum(T["s2"], gl)[ok_d, h - 1]
        return {"as_run": sse, "M8": sse - s2 / 4 + s2 / 8, "inf": sse - s2 / 4}, n
    q = gsum(T["q25" if k == 1 else "q50"], gl)[ok_d, h - 1]
    return {"as_run": sse, "M8": sse - q / 2, "inf": sse - q}, n


def jk_rmse(gl, h):
    """jackknife over the four stored fields of RMSE_inf and RMSE_8 (pooled over the days)"""
    n = gsum(T["n"], gl)[ok_d, h - 1].sum()
    sse = gsum(T["sse_jk"], gl)[ok_d, h - 1].sum(0); s2 = gsum(T["s2_jk"], gl)[ok_d, h - 1].sum(0)
    inf = np.sqrt((sse - s2 / 3) / n); m8 = np.sqrt((sse - s2 / 3 + s2 / 8) / n)
    return {"inf": inf, "M8": m8, "as_run": np.sqrt(sse / n)}                 # as run: RMSE of the 3-member mean (the as-run form)


OUT = {"as_run_lines_reproduced": REPRO, "bootstrap": "compare_fair.py verbatim: default_rng(0), blocks of 7 start days, 10,000 resamples, "
       f"{nd} start days", "groups": {}, "k2_station_pixels": K2, "sixteen_member_forecasts": [list(x) for x in D16]}
for gname, gl in GROUPS.items():
    G_ = OUT["groups"][gname] = {}
    for h in range(1, 5):
        L_ = G_[h] = {}
        for k, kname in ((0, "Foehn 1 km"), (1, "Foehn 0.25 deg average"), (2, "Foehn 0.5 deg average")):
            F, n = forms_foehn(k, gl, h)
            rm = {f: rmse_boot(F[f], n) for f in F}
            jk = jk_rmse(gl, h) if k == 0 else None
            vd = {}
            jk1 = jk_rmse(gl, h)
            for f in ("as_run", "M8", "inf"):                    # fix 9: the as-run (four-member) value is widened too
                key = {"as_run": "rmse_m", "M8": "rmse_8", "inf": "rmse_inf"}[f]
                cvq = cv_of(h, gl, key)
                if k == 0:
                    v_q = (cvq * rm[f][0]) ** 2; v_j = float(jk_var(jk1[f]))
                    vd[f] = dict(v_quart=float(v_q), v_jack4=v_j, v_draw=float(max(v_q, v_j)), cv_quart=float(cvq))
                else:                                                # member 0 only: the CV of the 1 km statistic is carried
                    F0, n0 = forms_foehn(0, gl, h); r1 = np.sqrt(np.nansum(F0[f]) / np.nansum(n0))
                    cvj = float(np.sqrt(jk_var(jk1[f])) / r1)
                    cvc = float(max(cvq, cvj)); v_c = float((cvc * rm[f][0]) ** 2)
                    v_m0 = 0.0
                    if f != "as_run":                                # fix 11: the single-member draw of the member-0 correction
                        qq = gsum(T["q25" if k == 1 else "q50"], gl)[ok_d, h - 1]
                        corr = float(np.nansum(qq) / np.nansum(n)) * (1.0 if f == "inf" else 0.5)
                        v_m0 = float((cv_m0(h) * corr / (2 * rm[f][0])) ** 2)
                    vd[f] = dict(cv_carried=cvc, v_m0=v_m0, v_draw=v_c + v_m0,
                                 note="member 0 only: CV of the 1 km station RMSE carried (max of quartet and four-field jackknife) + single-member draw of the correction (j_m0 ratios)")
            E_ = L_[kname] = {"n": float(np.nansum(n)), "rmse": {}}
            for f in ("as_run", "M8", "inf"):
                pt, bs = rm[f]; lo, hi = ci(bs)
                e = {"v": pt, "ci_boot": [lo, hi]}
                vb = float(np.nanvar(bs, ddof=1)); e["ci"], e["factor"] = inflate(pt, lo, hi, vb, vd[f]["v_draw"]); e.update(vd[f])
                E_["rmse"][f] = e
            E_["gain_pct"] = {}
            refs = dict(REFS)
            if k == 0:
                refs.update({"Foehn 0.25 deg average": 1, "Foehn 0.5 deg average": 2})
            for rname, j in refs.items():
                sb, cb = sys_day(j, gl, h)
                if not np.isfinite(sb).any() or np.nansum(cb) == 0:
                    continue
                gr = {}
                for f in ("as_run", "M8", "inf"):
                    if j in (1, 2):                                        # 1 km against its own average: same form on both
                        Fb, _ = forms_foehn(j, gl, h); nb_ = Fb[f]
                    else:
                        nb_ = sb
                    pt, bs = boot_ratio(F[f], n, nb_, cb); lo, hi = ci(bs)
                    e = {"v": 100 * pt, "ci_boot": [100 * lo, 100 * hi]}
                    rref = np.sqrt(np.nansum(nb_) / np.nansum(cb))
                    vdraw = vd[f]["v_draw"] / rref ** 2 * 1e4
                    vb = float(np.nanvar(100 * bs, ddof=1))
                    e["ci"], e["factor"] = inflate(100 * pt, 100 * lo, 100 * hi, vb, vdraw)
                    if j in (1, 2):                                # fix 16
                        e["draw_note"] = "1 km draw only (the average treated as fixed; the draw of the contrast is not estimable from stored members)"
                    gr[f] = e
                E_["gain_pct"][rname] = gr
            # sixteen members at the station pixels (leads 1 and 3, the days of the sixteen-member run)
            if k == 0 and h in (1, 3) and D16:
                li = (1, 3).index(h)
                dmask = np.array([(d, h) in have16 for d in DAYS_OK])
                q = {kk: gsum(T16[kk], gl)[ok_d, li][dmask] for kk in T16}
                N_ = q["n"].sum()
                inf16 = np.sqrt((q["sse"].sum() - q["s2M"].sum()) / N_)
                m8_16 = np.sqrt((q["sse"].sum() - q["s2M"].sum() + q["s2raw"].sum() / 8) / N_)
                jki = np.sqrt(q["jk_num"].sum(0) / N_); jk8 = np.sqrt(q["jk_num8"].sum(0) / N_)
                sub = {"days": int(dmask.sum()), "station_days": float(N_), "station_days_with_16": float(q["n16"].sum()),
                       "stored4_same_days": {"as_run": float(np.sqrt(q["sse4"].sum() / N_)),
                                             "inf": float(np.sqrt((q["sse4"].sum() - q["s2_4"].sum() / 4) / N_)),
                                             "M8": float(np.sqrt((q["sse4"].sum() - q["s2_4"].sum() / 8) / N_))},
                       "sixteen_inf": float(inf16), "sixteen_M8": float(m8_16),
                       "jk16_sd_inf": float(np.sqrt(jk_var(jki))), "jk16_sd_M8": float(np.sqrt(jk_var(jk8)))}
                for rname, j in (("CAMS d+1 (equal issuance)", 4), ("Aurora 00 UTC d+1 (equal issuance)", 6)):
                    sb, cb = sys_day(j, gl, h)
                    rr = np.sqrt(np.nansum(sb[dmask]) / np.nansum(cb[dmask]))
                    assert np.nansum(cb[dmask]) == N_
                    sub["gain_pct_" + rname] = {"as_run_stored4": 100 * (sub["stored4_same_days"]["as_run"] / rr - 1),
                                                "inf_stored4": 100 * (sub["stored4_same_days"]["inf"] / rr - 1),
                                                "inf_16": 100 * (inf16 / rr - 1), "M8_16": 100 * (m8_16 / rr - 1),
                                                "jk16_sd_inf": 100 * sub["jk16_sd_inf"] / rr}
                if dmask.all() or (DRY179 and dmask.any()):       # all start days: the paper's bootstrap and the 16-field jackknife
                    if not dmask.all():                           # dry run only: full-length arrays, zeros on the days without 16 members
                        q = {kk: gsum(T16[kk], gl)[ok_d, li] for kk in T16}
                    n_ = q["n"]
                    for form, num, reps in (("inf", q["sse"] - q["s2M"], jki), ("M8", q["sse"] - q["s2M"] + q["s2raw"] / 8, jk8)):
                        pt, bs = rmse_boot(num, n_); lo, hi = ci(bs)
                        vj = float(jk_var(reps))
                        cc_, fac = inflate(pt, lo, hi, float(np.nanvar(bs, ddof=1)), vj)
                        sub[f"rmse16_{form}"] = {"v": pt, "ci_boot": [lo, hi], "ci": cc_, "factor": fac, "v_jack16": vj}
                        for rname, j in (("CAMS d+1 (equal issuance)", 4), ("Aurora 00 UTC d+1 (equal issuance)", 6), ("climatology", 9), ("GHAP persistence (start-day estimate)", 8)):
                            sb, cb = sys_day(j, gl, h)
                            pt, bs = boot_ratio(num, n_, sb, cb); lo, hi = ci(bs)
                            rr = np.sqrt(np.nansum(sb) / np.nansum(cb))
                            cc_, fac = inflate(100 * pt, 100 * lo, 100 * hi, float(np.nanvar(100 * bs, ddof=1)), vj / rr ** 2 * 1e4)
                            sub[f"gain16_{form}_{rname}"] = {"v": 100 * pt, "ci_boot": [100 * lo, 100 * hi], "ci": cc_, "factor": fac}
                E_["sixteen_members_provisional"] = sub
    print("group", gname, "done", flush=True)

for h in (1, 3):
    for gname in list(K2[h]):
        r = K2[h][gname]
        OUT.setdefault("k2_summary", {}).setdefault(str(h), {})[gname] = {k: {q: r[k][q] for q in ("q0123", "all16", "quartet_sd", "z", "rank", "perm_p_two_sided", "z_vs_quartet_mean", "flag_2p5", "z_drawscale")} for k in r if isinstance(r[k], dict)}
OUT["dropped_16_station_days"] = dict(fallback4_total=int(sum(v["fallback4"] for v in DROP.values())),
                                      unexpected_total=int(sum(v["unexpected"] for v in DROP.values())), forecasts=len(DROP),
                                      note="fallback4 = station-days of the 57 stations without a populated pixel (their four stored members kept); unexpected = station-days at a populated-pixel station whose 16 members are not finite (fix 17)")
if DESIGN_NOTE:
    OUT["design"] = DESIGN_NOTE
OUTN = f"st_cf_classU{os.environ.get('A14U_OUT', '')}.json"
safe_json(RES / OUTN, OUT)
print("wrote results/" + OUTN, flush=True)
