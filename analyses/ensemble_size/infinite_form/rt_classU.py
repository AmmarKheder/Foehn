#!/usr/bin/env python3
"""A14 class U for the real-time record of 2026 (eight members as issued; D4: as-run first, infinite beside).
The paper's numbers come from realtime_1km/replay/score_cams3h (the unchanged scorer score/score.py rerun with the CAMS arm = the
mean of its 3-hourly steps); its station-day table score_cams3h/work/pairs.npz and score_cal/work/pairs.npz (calibrated arm
kmcal) are read only. score.py and sc_lib.py are imported unchanged (no bytecode); their samples, bootstrap (two-way blocks,
2,000 resamples, seed 20261119; 400 for the temporal statistics) and contrasts give the as-run values, checked against
score_cams3h/numbers.json and score_cal/numbers.json. Infinite forms (A14 4a): per station-day (xbar - y)^2 - s2/8 for every
eight-member arm (the 1 km forecast, the 0.4 deg input, the rings); spread/skill sqrt(mean s2)/RMSE_inf; the day-to-day
amplitude at a monitor with the pair form (var_t(xbar) - mean_i var_t(x_i - xbar)/7 per series); the fair CRPS is already
infinite (its eight-member value, fair + pair/8, beside). M = 8 is the issued size: the as-run value is the eight-member
value. V_draw: jackknife over the eight members (the downscaler's noise is fixed per member and lead day).
Monitors are evaluation data; nothing is fitted or selected.
    python rt_classU.py   -> results/rt_classU.json"""
import json
import sys
import time

import numpy as np

from u_common import RES, P, inflate, jk_var, safe_json

R = P / "realtime_1km" / "replay"
sys.path.insert(0, str(R / "score"))
import sc_lib as L  # noqa: E402  (unchanged)
import score as SC  # noqa: E402  (unchanged; main() is not run)

t0 = time.time()
OUT = {}


def num_index(path):
    return {e["name"]: e for e in json.loads(Path_(path).read_text())}


def Path_(p):
    from pathlib import Path
    return Path(p)


def point_views(smp, q):
    """point sums of a per-row quantity by (lead key, group key) as sc_lib.views"""
    S = smp.sums(q)
    return L.views(np.einsum("luw,ug->lg", S, smp.Ind))


def contrast_keys(res, gnames):
    out = {}
    for Lk, o in res.items():
        for gk, e in o.items():
            out[(Lk, gk)] = e
    return out


def jk_contrast(smp, D, a, ref, members_a, members_ref=None, form="inf"):
    """paired jackknife over the eight members (the same member removed from both ensemble arms) of the contrast a vs ref,
    every view: returns {(lead, group): var}. form: 'inf' (RMSE_inf), 'as_run' (RMSE of the 7-member mean, the as-run form),
    'crps' (fair CRPS) or 'crps8' (eight-member CRPS = fair + pair/8)"""
    rows = smp.rows
    y = D.obs[rows]
    Ea = D.e[a][rows].astype(np.float64)
    Er = D.e[ref][rows].astype(np.float64) if members_ref else None
    cnt = point_views(smp, np.ones(smp.n))

    def stat(E):
        M = E.shape[1]
        if form == "inf":
            return (E.mean(1) - y) ** 2 - E.var(1, ddof=1) / M
        if form == "as_run":
            return (E.mean(1) - y) ** 2
        a_ = np.abs(E - y[:, None]).mean(1)
        Es = np.sort(E, 1); w = (2 * np.arange(M) - M + 1).astype(np.float64)
        pair = (Es * w).sum(1) / (M * (M - 1))
        return a_ - pair if form == "crps" else a_ - pair + pair / 8
    if form in ("inf", "as_run"):
        det = (D.m[ref][rows] - y) ** 2
        fc = lambda A, B: 100 * (np.sqrt(A / B) - 1)          # noqa: E731
    else:
        det = np.abs(D.m[ref][rows] - y)
        fc = lambda A, B: 100 * (A / B - 1)                    # noqa: E731
    reps = []
    for i in range(Ea.shape[1]):
        qa = stat(np.delete(Ea, i, 1))
        qb = stat(np.delete(Er, i, 1)) if Er is not None else det
        va, vb = point_views(smp, qa), point_views(smp, qb)
        reps.append({k: fc(va[k], vb[k]) for k in va})
    keys = reps[0].keys()
    out = {}
    for k in keys:
        out[k] = float(jk_var(np.array([r[k] for r in reps])))
    # mean over regions: replicate of the mean of the regional contrasts with >= 100 station-days
    for Lk in ["all", "1", "2", "3", "4", "5"]:
        lk = Lk if Lk == "all" else int(Lk)
        gs = [g for g in range(smp.G) if cnt[(lk, g)] >= 100]
        if gs:
            out[(lk, "regmean")] = float(jk_var(np.array([np.mean([r[(lk, g)] for g in gs]) for r in reps])))
    return out


def q_inf(D, smp, a):
    rows = smp.rows
    st = D.ens_stats(a)
    return (D.m[a][rows] - D.obs[rows]) ** 2 - st["var"][rows] / st["M"]


def widen_views(smp, res, qa, qb, cnt, jk, f):
    """res = ratio_contrast output; widen every (lead, group) interval with its jackknife variance (A14 4c, section 8)"""
    (pa, Ra), (pb, Rb) = qa, qb
    wa, wb = L.views(Ra), L.views(Rb)
    out = {}
    for Lk, o in res.items():
        lk = Lk if Lk == "all" else int(Lk)
        for gk, e in o.items():
            if gk == "regmean":
                cnt0 = L.views(cnt[0])
                gs = [g for g in range(smp.G) if cnt0[(lk, g)] >= 100]
                bs = np.nanmean(np.stack([f(wa[(lk, g)], wb[(lk, g)]) for g in gs]), axis=0)
                key = (lk, "regmean")
            else:
                g = gk if gk == "all" else smp.gnames.index(gk)
                bs = f(wa[(lk, g)], wb[(lk, g)]); key = (lk, g)
            vb = float(np.nanvar(bs, ddof=1)); vd = jk.get(key, np.nan)
            ci, fac = inflate(e["est"], e["ci"][0], e["ci"][1], vb, vd)
            out[f"{Lk}|{gk}"] = dict(est=e["est"], ci_boot=e["ci"], ci=ci, factor=fac, v_jack8=vd, n=e["n"])
    return out


def contrast_forms(smp, D, a, b, kinds):
    """RMSE contrast a vs b: as run (score.contrast; eight members = the eight-member value) and infinite form, both widened with
    the paired member jackknife of their own statistic (fix 8)"""
    rows = smp.rows
    as_run = SC.contrast(smp, a, b, "rmse")
    cnt = smp.q("n", lambda: np.ones(smp.n))
    qa_as = smp.q(f"rmse_{a}", lambda: (D.m[a][rows] - D.obs[rows]) ** 2)
    qb_as = smp.q(f"rmse_{b}", lambda: (D.m[b][rows] - D.obs[rows]) ** 2)
    qa = smp.q(f"rmseinf_{a}", lambda: q_inf(D, smp, a)) if kinds[a] == "ens" else qa_as
    qb = smp.q(f"rmseinf_{b}", lambda: q_inf(D, smp, b)) if kinds[b] == "ens" else qb_as
    inf = L.ratio_contrast(smp, qa, qb, cnt, L.rel_rmse)
    jk_i = jk_contrast(smp, D, a, b, True, kinds[b] == "ens", "inf")
    jk_a = jk_contrast(smp, D, a, b, True, kinds[b] == "ens", "as_run")
    wi = widen_views(smp, inf, qa, qb, cnt, jk_i, L.rel_rmse)
    wa = widen_views(smp, as_run, qa_as, qb_as, cnt, jk_a, L.rel_rmse)
    return {k: {"as_run": wa[k], "inf": wi[k]} for k in wi}


def crps_forms(smp, D, a, b, kinds):
    """CRPS contrast a vs b: fair (= the infinite value, as run) and eight-member (fair + pair/8), both widened with the paired
    delete-one-member jackknife (fix 7)"""
    rows = smp.rows
    cnt = smp.q("n", lambda: np.ones(smp.n))

    def c8(x):
        if kinds[x] != "ens":
            return np.abs(D.m[x][rows] - D.obs[rows])
        st = D.ens_stats(x); ae = np.abs(D.e[x][rows].astype(np.float64) - D.obs[rows][:, None]).mean(1)
        return st["crps"][rows] + (ae - st["crps"][rows]) / st["M"]
    qa, qb = smp.q(f"crps_{a}", lambda: D.crps(a)[rows]), smp.q(f"crps_{b}", lambda: D.crps(b)[rows])
    qa8, qb8 = smp.q(f"crps8_{a}", lambda: c8(a)), smp.q(f"crps8_{b}", lambda: c8(b))
    fair = L.ratio_contrast(smp, qa, qb, cnt, L.rel_mean); eight = L.ratio_contrast(smp, qa8, qb8, cnt, L.rel_mean)
    jf = jk_contrast(smp, D, a, b, True, kinds[b] == "ens", "crps"); j8 = jk_contrast(smp, D, a, b, True, kinds[b] == "ens", "crps8")
    wf = widen_views(smp, fair, qa, qb, cnt, jf, L.rel_mean); w8 = widen_views(smp, eight, qa8, qb8, cnt, j8, L.rel_mean)
    return {k: {"fair_as_run_equals_inf": wf[k], "eight_member": w8[k]} for k in wf}


# =========================================================================== the paper's run (score_cams3h)
D = L.Data(str(R / "score_cams3h" / "work" / "pairs.npz"))
NUM = num_index(R / "score_cams3h" / "numbers.json")
print(f"loaded {D.N:,} station-days, arms {D.arms} ({time.time() - t0:.0f} s)", flush=True)
focus = ["km"]; base = ["rt04", "cams", "clim", "bcsd"]; B5 = focus + base
kinds = D.kinds


def sample(arms, period, extra=None):
    m = D.finite(arms) & D.issues(*SC.PERIODS[period][:2])
    if extra is not None:
        m &= extra
    return L.Sample(D, m, None, None, B=2000)


sB = sample(B5, "clean")
ident = {}
T03 = {}
for b in base:
    r = contrast_forms(sB, D, "km", b, kinds)
    T03[f"km_vs_{b}"] = r
    for k, v in r.items():
        Lk, g = k.split("|")
        nm = f"rmse_change_pct.km_vs_{b}.withCAMS.lead_{Lk}.{g}"
        if nm in NUM:
            ident[nm] = bool(abs(NUM[nm]["value"] - round(v["as_run"]["est"], 4)) < 1e-3 and
                             all(abs(x - round(y, 4)) < 1e-3 for x, y in zip(NUM[nm]["interval"], v["as_run"]["ci_boot"])))
OUT["T03_contrasts_clean_period"] = {"sample": sB.describe(), "contrasts": T03}
print("T03 done", time.time() - t0, flush=True)

# periods (T11): issue-day periods of the record
PER = {}
for p in ("sep", "oct", "live"):
    s = sample(B5, p)
    PER[p] = {"sample": s.describe(), "km_vs_cams": contrast_forms(s, D, "km", "cams", kinds)}
    for k, v in PER[p]["km_vs_cams"].items():
        Lk, g = k.split("|")
        if Lk == "all" and g in ("all", "regmean"):
            nm = f"period_{p}.with_CAMS.rmse_change_pct.km_vs_cams.{g}"
            if nm in NUM:
                ident[nm] = bool(abs(NUM[nm]["value"] - round(v["as_run"]["est"], 4)) < 1e-3)
OUT["T11_periods"] = PER

# rings (T08): the forecast read 10 km away
rings = [x for x in ["km_ring1", "km_ring3", "km_ring10", "km_cell_lin"] if x in D.arms]
sR = sample(["km"] + rings + base, "clean")
RG = {}
for r_ in ("km_ring10", "km_ring3"):
    RG[f"{r_}_vs_cams"] = contrast_forms(sR, D, r_, "cams", kinds)
    v = RG[f"{r_}_vs_cams"]["all|all"]
    nm = f"rings.rmse_change_pct.{r_}_vs_cams.withCAMS.all"
    if nm in NUM:
        ident[nm] = bool(abs(NUM[nm]["value"] - round(v["as_run"]["est"], 4)) < 1e-3)
OUT["T08_rings"] = {"sample": sR.describe(), "contrasts": RG}
print("T08/T11 done", time.time() - t0, flush=True)

# ensemble (T05): spread/skill, CRPS contrasts
sE = sample(B5 + ["bcsd_m"], "clean")
ens_arms = ["km", "rt04", "bcsd_m"]
E5 = SC.ensemble(sE, ens_arms, ["cams", "clim", "bcsd"])
ENS = {}
cnt = sE.q("n", lambda: np.ones(sE.n)); vn, wn = L.views(cnt[0]), L.views(cnt[1])
for a in ens_arms:
    st = D.ens_stats(a); M = st["M"]; rows = sE.rows
    qv = sE.q(f"var_{a}", lambda: st["var"][rows]); qs = sE.q(f"rmse_{a}", lambda: (D.m[a][rows] - D.obs[rows]) ** 2)
    V0, V1, S0, S1 = L.views(qv[0]), L.views(qv[1]), L.views(qs[0]), L.views(qs[1])
    # mean |x_i - y| for the eight-member CRPS (fair + pair/8)
    ae = np.abs(D.e[a][rows].astype(np.float64) - D.obs[rows][:, None]).mean(1)
    qa = sE.q(f"ae_{a}", lambda: ae); qc = sE.q(f"crps_{a}", lambda: st["crps"][rows])
    A0, A1, C0, C1 = L.views(qa[0]), L.views(qa[1]), L.views(qc[0]), L.views(qc[1])
    # jackknife over the members of the infinite spread/skill (all|all)
    Ea = D.e[a][rows].astype(np.float64); y = D.obs[rows]
    reps = []; reps_as = []
    for i in range(M):
        X_ = np.delete(Ea, i, 1); v_ = X_.var(1, ddof=1)
        reps.append(np.sqrt(v_.sum() / len(y)) / np.sqrt((((X_.mean(1) - y) ** 2).sum() - v_.sum() / (M - 1)) / len(y)))
        reps_as.append(np.sqrt(M / (M - 1) * v_.sum() / len(y)) / np.sqrt(((X_.mean(1) - y) ** 2).sum() / len(y)))   # Fortin with 7 members
    o = {}
    for k in [("all", "all")] + [(Ld, "all") for Ld in range(1, 6)]:
        if vn[k] < 100:
            continue
        f = (M + 1) / M
        ss_inf = np.sqrt(V0[k] / vn[k]) / np.sqrt((S0[k] - V0[k] / M) / vn[k])
        bs = np.sqrt(V1[k] / wn[k]) / np.sqrt((S1[k] - V1[k] / M) / wn[k])
        lo, hi = L.pct(bs)
        e = {"spread_skill_as_run": E5[a][f"{k[0]}|all"]["spread_skill"],
             "spread_skill_inf": {"est": float(ss_inf), "ci_boot": [lo, hi]},
             "rmse_as_run": float(np.sqrt(S0[k] / vn[k])), "rmse_inf": float(np.sqrt((S0[k] - V0[k] / M) / vn[k])),
             "crps_fair_as_run_equals_inf": float(C0[k] / vn[k]), "crps_8": float((C0[k] + (A0[k] - C0[k]) / M) / vn[k])}
        if k == ("all", "all"):
            vd = float(jk_var(np.array(reps)))
            e["spread_skill_inf"]["ci"], e["spread_skill_inf"]["factor"] = inflate(float(ss_inf), lo, hi, float(np.nanvar(bs, ddof=1)), vd)
            e["spread_skill_inf"]["v_jack8"] = vd
            bs_as = np.sqrt(f * V1[k] / S1[k]); va_ = float(jk_var(np.array(reps_as)))         # fix 8: the as-run ratio widened too
            ar_ = e["spread_skill_as_run"]
            ci_as, fac_as = inflate(ar_["est"], ar_["ci"][0], ar_["ci"][1], float(np.nanvar(bs_as, ddof=1)), va_)
            e["spread_skill_as_run"] = dict(ar_, ci_boot=ar_["ci"], ci=ci_as, factor=fac_as, v_jack8=va_)
            nm = f"spread_skill.{a}.withCAMS.lead_all.all"
            if nm in NUM:
                ident[nm] = bool(abs(NUM[nm]["value"] - round(E5[a]["all|all"]["spread_skill"]["est"], 4)) < 1e-3)
        o[f"{k[0]}|all"] = e
    ENS[a] = o
# CRPS contrasts of the 1 km members against the 0.4 deg members and the BCSD members: fair (= infinite) and eight-member,
# widened with the paired delete-one-member jackknife (fix 7)
CR = {}
for b in ("rt04", "bcsd_m"):
    res = SC.contrast(sE, "km", b, "crps")
    cf = crps_forms(sE, D, "km", b, kinds)
    CR[f"km_vs_{b}"] = {"fair_as_run_equals_inf": cf["all|all"]["fair_as_run_equals_inf"], "eight_member": cf["all|all"]["eight_member"],
                        "scorer_fair": res["all"]["all"]}
    nm = f"crps_change_pct.km_vs_{b}.withCAMS.lead_all.all"
    if nm in NUM:
        ident[nm] = bool(abs(NUM[nm]["value"] - round(res["all"]["all"]["est"], 4)) < 1e-3)
OUT["T05_ensemble"] = {"sample": sE.describe(), "arms": ENS, "crps_contrasts": CR}
print("T05 done", time.time() - t0, flush=True)

# temporal (T04): amplitude ratio at a monitor, pair form
TT = SC.temporal(sB, B5, focus, Bt=400)
amp = {}
rows = sB.rows
us, s_ = np.unique(D.mon[rows] * 5 + (D.lead[rows] - 1), return_inverse=True)
ns_ = us.size; Wn = sB.W
key = s_ * Wn + sB.w
sw = lambda q: np.bincount(key, weights=q, minlength=ns_ * Wn).reshape(ns_, Wn)
y = D.obs[rows]
n_sw, y_sw, yy_sw = sw(np.ones(y.size)), sw(y), sw(y * y)
sc_ = np.zeros(ns_, np.int64); sc_[s_] = sB.ci
sg = np.zeros(ns_, np.int64); sg[s_] = sB.g


def series_stats(n, sy, syy, X, nmin):
    """X = list of (sx, sxx, sxy) of each arm; returns r, amp (as temporal)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        my = sy / n; vy = syy / n - my * my
        out = []
        for sx, sxx, sxy in X:
            mx = sx / n; vx = sxx / n - mx * mx; cov = sxy / n - mx * my
            ok = (n >= nmin) & (vx > 1e-9) & (vy > 1e-9)
            out.append((np.where(ok, cov / np.sqrt(vx * vy), np.nan), np.where(ok, np.sqrt(vx / vy), np.nan)))
    return out


XS = {}
for a in B5:
    x = D.m[a][rows]; XS[a] = (sw(x), sw(x * x), sw(x * y))
n = n_sw.sum(1)
base_stats = series_stats(n, y_sw.sum(1), yy_sw.sum(1), [(XS[a][0].sum(1), XS[a][1].sum(1), XS[a][2].sum(1)) for a in B5], SC.MIN_DAYS)
ok_series = np.all([np.isfinite(r) for r, _ in base_stats], axis=0)
for a in ("km", "rt04"):
    E_ = D.e[a][rows].astype(np.float64); xb = D.m[a][rows]
    U = E_ - xb[:, None]
    us1 = np.stack([sw(U[:, i]) for i in range(U.shape[1])]); us2 = np.stack([sw(U[:, i] ** 2) for i in range(U.shape[1])])
    M = U.shape[1]

    def amp_pair(nw, sx, sxx, sy, syy, u1, u2, nmin):
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = sx / nw; vx = sxx / nw - mx * mx; my = sy / nw; vy = syy / nw - my * my
            vu = np.mean([u2[i] / nw - (u1[i] / nw) ** 2 for i in range(M)], axis=0)
            vinf = vx - vu / (M - 1)
            ok = (nw >= nmin) & (vx > 1e-9) & (vy > 1e-9)
            return np.where(ok, np.sqrt(np.maximum(vinf, 0) / vy), np.nan), np.where(ok, vinf / vy, np.nan)
    A_inf, ratio2 = amp_pair(n, XS[a][0].sum(1), XS[a][1].sum(1), y_sw.sum(1), yy_sw.sum(1), us1.sum(2), us2.sum(2), SC.MIN_DAYS)
    m_all = ok_series
    est = float(np.nanmedian(A_inf[m_all])); n_neg = int((ratio2[m_all] <= 0).sum())
    # jackknife over members: the series mean is that of all eight; dropping member i changes xbar: recompute from members
    reps = []
    for i in range(M):
        Xd = np.delete(E_, i, 1); xd = Xd.mean(1); Ud = Xd - xd[:, None]
        sx_, sxx_ = sw(xd).sum(1), sw(xd * xd).sum(1)
        u1 = [sw(Ud[:, j]).sum(1) for j in range(M - 1)]; u2 = [sw(Ud[:, j] ** 2).sum(1) for j in range(M - 1)]
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = sx_ / n; vx = sxx_ / n - mx * mx; my = y_sw.sum(1) / n; vy = yy_sw.sum(1) / n - my * my
            vu = np.mean([u2[j] / n - (u1[j] / n) ** 2 for j in range(M - 1)], axis=0)
            vinf = vx - vu / (M - 2)
            Ai = np.where((n >= SC.MIN_DAYS) & (vx > 1e-9) & (vy > 1e-9), np.sqrt(np.maximum(vinf, 0) / vy), np.nan)
        reps.append(float(np.nanmedian(Ai[m_all])))
    # bootstrap (as temporal: weeks reweighted in the series, cells as weights of the median), 400 resamples
    Bt = 400; bs = np.full(Bt, np.nan)
    for b0 in range(0, Bt, 50):
        b1 = min(b0 + 50, Bt)
        mw, mc = sB.Mw[b0:b1], sB.Mc[b0:b1][:, sc_]
        nb = mw @ n_sw.T
        Ab, _ = amp_pair(nb, mw @ XS[a][0].T, mw @ XS[a][1].T, mw @ y_sw.T, mw @ yy_sw.T,
                         np.stack([mw @ us1[i].T for i in range(M)]), np.stack([mw @ us2[i].T for i in range(M)]), 5)
        ii = np.flatnonzero(m_all)
        bs[b0:b1] = L.wmedian(Ab[:, ii], mc[:, ii])
    lo, hi = L.pct(bs)
    vd = float(jk_var(np.array(reps)))
    ci, f = inflate(est, lo, hi, float(np.nanvar(bs, ddof=1)), vd)
    # fix 8: the as-run amplitude (8-member mean) widened with the jackknife of the 7-member mean
    reps_as = []
    for i in range(M):
        xd = np.delete(E_, i, 1).mean(1)
        sx_, sxx_ = sw(xd).sum(1), sw(xd * xd).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = sx_ / n; vx = sxx_ / n - mx * mx; my = y_sw.sum(1) / n; vy = yy_sw.sum(1) / n - my * my
            Ai = np.where((n >= SC.MIN_DAYS) & (vx > 1e-9) & (vy > 1e-9), np.sqrt(vx / vy), np.nan)
        reps_as.append(float(np.nanmedian(Ai[m_all])))
    bs_as = np.full(Bt, np.nan)
    for b0 in range(0, Bt, 50):
        b1 = min(b0 + 50, Bt)
        mw, mc = sB.Mw[b0:b1], sB.Mc[b0:b1][:, sc_]
        nb = mw @ n_sw.T
        with np.errstate(divide="ignore", invalid="ignore"):
            mxb = (mw @ XS[a][0].T) / nb; vxb = (mw @ XS[a][1].T) / nb - mxb * mxb; myb = (mw @ y_sw.T) / nb; vyb = (mw @ yy_sw.T) / nb - myb * myb
            Ab_as = np.where((nb >= 5) & (vxb > 1e-9) & (vyb > 1e-9), np.sqrt(vxb / vyb), np.nan)
        ii = np.flatnonzero(m_all)
        bs_as[b0:b1] = L.wmedian(Ab_as[:, ii], mc[:, ii])
    arA = TT["groups"]["all"][a]["amp"]
    va_ = float(jk_var(np.array(reps_as)))
    ci_as, f_as = inflate(arA["est"], arA["ci"][0], arA["ci"][1], float(np.nanvar(bs_as, ddof=1)), va_)
    as_run_w = dict(arA, ci_boot=arA["ci"], ci=ci_as, factor=f_as, v_jack8=va_)
    amp[a] = {"as_run": as_run_w, "inf_pair_form_median": {"est": est, "ci_boot": [lo, hi], "ci": ci, "factor": f, "v_jack8": vd,
              "series_with_nonpositive_var_inf_set_to_0": n_neg, "n_series": int(m_all.sum())},
              "note": "median over (monitor, lead day) series of sqrt(var_inf,t(xbar)/var_t(y)), var_inf by the pair form; a median is class E (A14 4b)"}
    nm = f"amplitude_ratio_median.{a}.withCAMS.all"
    if nm in NUM:
        ident[nm] = bool(abs(NUM[nm]["value"] - round(TT["groups"]["all"][a]["amp"]["est"], 4)) < 1e-3)
OUT["T04_amplitude"] = amp
print("T04 done", time.time() - t0, flush=True)

# =========================================================================== calibrated arm (score_cal)
DC = L.Data(str(R / "score_cal" / "work" / "pairs.npz"))
NUMC = num_index(R / "score_cal" / "numbers.json")
if "kmcal" in DC.arms:
    armsC = ["km", "kmcal", "rt04", "cams", "clim", "bcsd"]
    sC3 = L.Sample(DC, DC.finite(armsC) & DC.issues(*SC.PERIODS["clean"][:2]), None, None, B=2000)
    sC5 = L.Sample(DC, DC.finite(armsC + ["bcsd_m"]) & DC.issues(*SC.PERIODS["clean"][:2]), None, None, B=2000)
    EC = SC.ensemble(sC5, ["km", "kmcal"], ["cams", "clim", "bcsd"])
    cal = {"sample_T05": sC5.describe(), "sample_T03": sC3.describe()}
    rows = sC5.rows
    cn5 = sC5.q("n", lambda: np.ones(sC5.n)); wn5 = L.views(cn5[1])
    for a in ("km", "kmcal"):
        st = DC.ens_stats(a); M = st["M"]
        v = st["var"][rows]; se = (DC.m[a][rows] - DC.obs[rows]) ** 2
        Vv = L.views(sC5.q(f"var_{a}", lambda: st["var"][rows])[1]); Ss = L.views(sC5.q(f"rmse_{a}", lambda: (DC.m[a][rows] - DC.obs[rows]) ** 2)[1])
        k0 = ("all", "all")
        bs_as = np.sqrt((M + 1) / M * Vv[k0] / Ss[k0]); bs_in = np.sqrt(Vv[k0] / wn5[k0]) / np.sqrt((Ss[k0] - Vv[k0] / M) / wn5[k0])
        Ea = DC.e[a][rows].astype(np.float64); yo = DC.obs[rows]
        ra, ri = [], []
        for i in range(M):                                       # fix 8: member jackknife of both forms
            X_ = np.delete(Ea, i, 1); v_ = X_.var(1, ddof=1); se_ = ((X_.mean(1) - yo) ** 2).sum()
            ra.append(np.sqrt(M / (M - 1) * v_.sum() / se_)); ri.append(np.sqrt(v_.sum() / len(yo)) / np.sqrt((se_ - v_.sum() / (M - 1)) / len(yo)))
        ar_ = EC[a]["all|all"]["spread_skill"]; si = float(np.sqrt(v.mean()) / np.sqrt((se.sum() - v.sum() / M) / len(se)))
        ci_a, f_a = inflate(ar_["est"], ar_["ci"][0], ar_["ci"][1], float(np.nanvar(bs_as, ddof=1)), float(jk_var(np.array(ra))))
        lo_i, hi_i = L.pct(bs_in); ci_i, f_i = inflate(si, lo_i, hi_i, float(np.nanvar(bs_in, ddof=1)), float(jk_var(np.array(ri))))
        cal[a] = {"spread_skill_as_run": dict(ar_, ci_boot=ar_["ci"], ci=ci_a, factor=f_a),
                  "spread_skill_inf": si, "spread_skill_inf_ci": dict(ci_boot=[lo_i, hi_i], ci=ci_i, factor=f_i),
                  "rmse_as_run": float(np.sqrt(se.mean())), "rmse_inf": float(np.sqrt((se.sum() - v.sum() / M) / len(se)))}
        nm = f"spread_skill.{a}.withCAMS.lead_all.all"
        if nm in NUMC:
            ident["cal:" + nm] = bool(abs(NUMC[nm]["value"] - round(EC[a]["all|all"]["spread_skill"]["est"], 4)) < 1e-3)
    rc = SC.contrast(sC3, "kmcal", "km", "crps")
    nm = "crps_change_pct.kmcal_vs_km.withCAMS.lead_all.all"
    if nm in NUMC:
        ident["cal:" + nm] = bool(abs(NUMC[nm]["value"] - round(rc["all"]["all"]["est"], 4)) < 1e-3)
    cfc = crps_forms(sC3, DC, "kmcal", "km", DC.kinds)["all|all"]                 # fix 7
    cal["crps_change_kmcal_vs_km_fair_equals_inf"] = cfc["fair_as_run_equals_inf"]
    cal["crps_change_kmcal_vs_km_eight_member"] = cfc["eight_member"]
    cal["rmse_change_kmcal_vs_km"] = contrast_forms(sC3, DC, "kmcal", "km", DC.kinds)["all|all"]
    OUT["calibrated_kmcal"] = cal
OUT["identity_with_paper_numbers_json"] = ident
OUT["identity_all"] = bool(all(ident.values()))
safe_json(RES / "rt_classU.json", OUT)
print("identity:", sum(ident.values()), "of", len(ident), "| done", time.time() - t0, flush=True)
