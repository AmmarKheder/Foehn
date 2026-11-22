#!/usr/bin/env python3
"""A20 scorer and decision (a20/A20_draft.md sections 7-9), CPU only, in the job container (numpy 1.26.3).
Frozen: its sha256 is written into every output (field score20_sha256) and, by the main session at the filing, into
a20/FILED; the decision refuses to run unless a20/FILED exists and holds this file's sha256. The code and data this
scorer depends on are frozen through it: DEP holds the sha256 of common20.py, pix20.npz, meta20.npz and law20.npz (checked
at start); meta20.npz holds the sha256 of the four member / feature arrays (recomputed by guard G_data); score19.py is
imported unchanged (its sha256 is checked) for the paper's station scorer.

Ensembles. Plain mean of four = the mean of G2's members 0-3 (as run). Noise-aware mean of four (A18 4.2, primary): with
g_k = F_k . gamma_b (F_k the 12 features of member k's own noise field at the pixel, gamma_b the law of the lead fitted on
the start-day blocks other than b, fit20.py), m1 = xbar (1 - gbar), cbar = xbar - m1 gbar. Variants (reported): no
iteration xbar - xbar gbar, fixed point xbar / max(1 + gbar, 0.5), the A18 first-design law (c).
STATIONS (the decision): the station-days of the paper's station score (compare_fair.py executed unchanged through
score19.station_prep, its identities asserted), leads 1 and 3, on the sample S' of A19 (the 8,053 stations with a
populated pixel; members read at the station rows of a20/data, bit copies of G2's and G11's populated-pixel rows).
  D_h = MSE(plain mean of four) - MSE(noise-aware mean of four), lead h, with compare_fair's paired moving-block bootstrap
  (default_rng(0), blocks of 7 start days, 10,000 resamples, its 178 start days), leads pooled by start day as well.
POPULATED PIXELS (reported; Q2 class): the 1,000,000 sampled pixels (pix20), every member of G2 and G11 finite, against the
1 km estimates of the target day (G2's pop_ghap), weights: pixels and residents; the start-day bootstrap of 179 days.
  Q2 (A18 4.3 / 8.2, thresholds 37.5 and 75 ug m-3): closed form from the four members 'an4' (log-normal, mean cbar,
  relative variance sigmaL2 + kappa2(cbar) (1 + 1/4)), 'ga4' (Gaussian, same moments), the no-law control 'nl4'
  (log-normal, mean xbar, tau2(xbar) (1 + 1/4)), 'tk4' (mean cbar, tau2(cbar) + kappa2(cbar) / 4); counted: G2's four
  ('fr4') and G11's sixteen ('fr16'); Brier scores and tie-split warnings; parameters from fit20 (other blocks).
Reported beside (never gating): RMSE, reductions against the CAMS forecast and Aurora (issued at the same time) of the
plain, noise-aware and paper's forms (eight members MSE_inf + V/8 and infinitely many, from G11's sixteen), the truth-free
equivalent number of plain members of the noise-aware mean (A18 4.2, reference G11's members 4-15), the bias, the
members-only R^2 of the law at the scored elements, the spread of the corrected members, the same at the populated pixels.
Guards (INVALID): G_id identities (compare_fair executed = stored, replay, the paper's full-sample station values and forms
= a14 st_cf_classU_179.json, G11's station members = a14 st16_run16, the plain four on S' = A19's tests19.json values);
G_data (hashes); G_fin (no station-day of S' dropped for a non-finite member); G_bias (|sum(cbar - xbar)| / sum(xbar) at
S' <= TOL_BIAS at each lead).
Decision Q1 (stations S', ug m-3): INVALID (a guard) / SUCCESS (D_1 > 0, D_3 > 0, lower end of the 95 % interval of D_1
> 0) / FAIL. Q2 (populated pixels, residents, lead 1; A18 8.2, never changing Q1): INVALID / SUCCESS / ORDER /
DRESSING / FAIL.
TESTS (--tests; no corrected quantity is ever compared with a 2022 truth): the identities on the real data (plain mean and
forms only); synthetic worlds on the real S' station-days, the real station and null-row features and, as level, the mean
of G11's members 4-15 (independent of G2's noise) (members AND truth synthetic; laws fitted by common20 on the synthetic members at the 100,000 null pixel rows);
and nulls on the REAL members with a member-drawn truth (G11's member 15; forms from G11 0-14, reference G11 4-14).
    python -B -u score20.py --tests    -> code/out/tests20.json
    python -B -u score20.py            -> code/out/score20.json (after the filing only)"""
import sys

sys.dont_write_bytecode = True
import common20 as C  # noqa: E402,I001  (first: hooks)

MODE = "tests" if "--tests" in sys.argv[1:] else "decision"
C.install(MODE)
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402

SELF_SHA256 = C.sha_file(__file__)
SCORE19_SHA256 = "a83a558347677752404be487eb158a0a7efba07a4934d209dc7eb3e6ca43c570"     # A19_draft.md 6.4 and 7 (after its
# review; against the first freeze 830291d3... only A19's own guard G changed, none of the functions imported here)
DEP = {"common20.py": "a2cdd59ddde1c3ccf9d15cfb2176dbbc473e50149dad5695c2a4871c68fa3918",
       "pix20.npz": "8ea36b03528c65fb5b0a2acf4b30c455387530371f07f4ec119f21bd87c1ff57",
       "meta20.npz": "64124e26a57faf112c18419d8c55dc4bc27a9471cd3241a2f9d2165f9fba3826",
       "law20.npz": "d8aa5a2ace719c203a4eafbcba7c5594f7f5f6b173f40fbfdeda37922240cc4e"}
TOL_BIAS = 0.01      # draft 6.3: members-only |bias| <= 0.12 % (95 % ends within 0.31 %) at the station rows and pixels
                     # (diag20); a 1 % bias adds about 0.055 (ug m-3)^2 to the MSE at the stations
A19_SPRIME = {  # the plain four, the forms and the counts on S' (a19/code/out/tests19.json, score19 a83a5583..., case N0)
    "1": dict(rmse_4=13.033072332886759, rmse_form8=12.813988272496516, rmse_inf=12.587639305038355, n=1027324),
    "3": dict(rmse_4=14.244487743870446, rmse_form8=13.939028370258558, rmse_inf=13.631973267757617, n=1022208)}
THR = (37.5, 75.0)
QM = ("an4", "ga4", "nl4", "tk4", "fr4", "fr16")
S = {}


def deps_ok():
    got = {"common20.py": C.sha_file(C.CODE / "common20.py"), "pix20.npz": C.sha_file(C.DATA / "pix20.npz"),
           "meta20.npz": C.sha_file(C.DATA / "meta20.npz"), "law20.npz": C.sha_file(C.DATA / "law20.npz")}
    return got, all(got[k] == DEP[k] for k in DEP)


# ======================================================================== per-element columns
def q1_columns(X, gd, YF, YR, y, refs):
    """X (n, 4) members, gd {'corr': g (n, 4)[, 'A18': g]}, YF (n, nF) members of the paper's forms, YR (n, nR) reference
    members of the truth-free estimator (independent of X), y (n,) the truth of this case, refs {name: (n,)}"""
    g = gd["corr"]
    cr = C.correct(X, g)
    xb = cr["plain"]
    e = lambda f: (f - y) ** 2                                         # noqa: E731
    nR = YR.shape[1]
    xr, s2r = YR.mean(1), YR.var(1, ddof=1)
    D = X - xb[:, None]
    pred = xb[:, None] * (g - g.mean(1, keepdims=True))
    cm = X - cr["m1"][:, None] * g
    col = dict(one=np.ones(len(y)), y=y, sse_p=e(xb), sse_c=e(cr["corr"]), sse_n=e(cr["noiter"]), sse_f=e(cr["fixpt"]),
               sse_F=e(YF.mean(1)), s2F=YF.var(1, ddof=1), V=s2r, tfp=(xb - xr) ** 2 - s2r / nR,
               tfc=(cr["corr"] - xr) ** 2 - s2r / nR, bias=cr["corr"] - xb, xbar=xb, sD=(D * D).sum(1),
               sse_law=((D - pred) ** 2).sum(1), s2_4=X.var(1, ddof=1), s2_c=cm.var(1, ddof=1))
    if "A18" in gd:
        ca = C.correct(X, gd["A18"])
        col.update(sse_A=e(ca["corr"]), biasA=ca["corr"] - xb)
    for k, v in refs.items():
        col["sse_" + k] = e(v)
    return col, cr


def q2_columns(X, cr, YF, y, q2):
    """Brier and warning columns of the closed forms and the counted members (A18 4.3); q2 = (sL2, kappa2, tau2)"""
    sL2, k2, t2 = q2
    xb, cb = cr["plain"], cr["corr"]
    fM = 1.0 + 1.0 / 4.0
    vL = sL2 + C.lookup(k2, cb) * fM
    vN = C.lookup(t2, xb) * fM
    vT = C.lookup(t2, cb) + C.lookup(k2, cb) / 4.0
    col = {}
    for T in THR:
        o = (y > T).astype(np.float64)
        col[f"o_{T}"] = o
        P = {"an4": C.p_logn(cb, vL, T), "ga4": C.p_gauss(cb, vL, T), "nl4": C.p_logn(xb, vN, T), "tk4": C.p_logn(cb, vT, T),
             "fr4": (X > T).mean(1), "fr16": (YF > T).mean(1)}
        for m, p in P.items():
            wn = C.warn(p)
            col[f"bs_{m}_{T}"] = (p - o) ** 2
            col[f"w_{m}_{T}"] = wn
            col[f"h_{m}_{T}"] = wn * o
    return col


def sums_of(col, w=None):
    return {k: float(np.sum(v)) if w is None else float(w @ v) for k, v in col.items()}


# ======================================================================== statistics of summed columns
def derive(t, nF):
    """t {col: (1 + 10,000,)} point and resamples -> statistics"""
    with np.errstate(all="ignore"):
        n = t["one"]
        m = lambda k: t[k] / n                                          # noqa: E731
        o = {"n": n, "mean_y": m("y")}
        for k in ("p", "c", "n", "f", "A"):
            if f"sse_{k}" in t:
                o[f"mse_{k}"] = m(f"sse_{k}")
                o[f"rmse_{k}"] = np.sqrt(o[f"mse_{k}"])
        VF = m("s2F")
        inf = m("sse_F") - VF / nF
        o.update(V_forms=VF, mse_inf=inf, mse_form8=inf + VF / 8, rmse_inf=np.sqrt(inf), rmse_form8=np.sqrt(inf + VF / 8))
        V = m("V")
        o["V_member"] = V
        o["D"] = o["mse_p"] - o["mse_c"]
        o["D_rel_V"] = o["D"] / V
        o["D_noiter"] = o["mse_p"] - o["mse_n"]
        o["D_fixpt"] = o["mse_p"] - o["mse_f"]
        if "mse_A" in o:
            o["D_A18law"] = o["mse_p"] - o["mse_A"]
            o["bias_rel_A18law"] = t["biasA"] / t["xbar"]
        o["d_form8_minus_corr"] = o["mse_form8"] - o["mse_c"]
        o["inv_meq_tf_corr"] = m("tfc") / V
        o["inv_meq_tf_plain"] = m("tfp") / V
        o["inv_meq_corr_vs_forms"] = (o["mse_c"] - inf) / VF
        o["inv_meq_plain_vs_forms"] = (o["mse_p"] - inf) / VF
        o["bias_rel"] = t["bias"] / t["xbar"]
        o["R2_law"] = 1 - t["sse_law"] / t["sD"]
        o["spread_corr_over_plain"] = t["s2_c"] / t["s2_4"]
        for r in ("cams", "aur", "g1"):
            if f"sse_{r}" in t:
                mr = m(f"sse_{r}")
                o[f"rmse_{r}"] = np.sqrt(mr)
                if r == "g1":
                    continue
                sysm = {"plain4": o["mse_p"], "corr4": o["mse_c"], "form8": o["mse_form8"], "inf": inf}
                if "mse_A" in o:
                    sysm["A18law4"] = o["mse_A"]
                for s_, v in sysm.items():
                    o[f"red_{s_}_vs_{r}"] = 100 * (1 - np.sqrt(v / mr))
        for T in THR:
            if f"o_{T}" not in t:
                continue
            o[f"event_share_{T}"] = m(f"o_{T}")
            for q in QM:
                o[f"bs_{q}_{T}"] = m(f"bs_{q}_{T}")
                o[f"warned_{q}_{T}"] = m(f"w_{q}_{T}")
                o[f"correct_{q}_{T}"] = t[f"h_{q}_{T}"] / t[f"w_{q}_{T}"]
                o[f"reach_{q}_{T}"] = t[f"h_{q}_{T}"] / t[f"o_{T}"]
            for ref in ("nl4", "fr16", "fr4", "tk4", "ga4"):
                o[f"d_{ref}_an4_{T}"] = o[f"bs_{ref}_{T}"] - o[f"bs_an4_{T}"]
            o[f"d_fr16_nl4_{T}"] = o[f"bs_fr16_{T}"] - o[f"bs_nl4_{T}"]
    return o


def summarize(o):
    out = {k: C.ci(v) for k, v in o.items()}
    for k in [k for k in out if k.startswith("inv_meq")]:
        out["meq" + k[7:]] = C.inv_ci(out[k])
    return out


def boot_stats(Sd, W, nF):
    """Sd {col: (days, 2 leads)} -> {L: summary} for L = 1, 3 and 13 (pooled by start day)"""
    res = {}
    for L in ("1", "3", "13"):
        t = {}
        for k, a in Sd.items():
            v = a[:, 0] if L == "1" else (a[:, 1] if L == "3" else a[:, 0] + a[:, 1])
            t[k] = np.concatenate([[v.sum()], W @ v])
        res[L] = summarize(derive(t, nF))
    return res


# ======================================================================== decisions (fixed in advance, draft section 8)
def decide_q1(st, invalid):
    est = lambda L, k: float(st[L][k]["est"])                            # noqa: E731
    lo = lambda L, k: float(st[L][k]["ci"][0])                           # noqa: E731
    cond = {"D_1_above_0": est("1", "D") > 0, "D_3_above_0": est("3", "D") > 0, "D_1_interval_above_0": lo("1", "D") > 0}
    rule = "SUCCESS" if all(cond.values()) else "FAIL"
    return dict(outcome="INVALID" if invalid else rule, rule=rule, invalid_reasons=list(invalid), conditions=cond)


def decide_q2(px, invalid):
    """A18 8.2 on the populated pixels, residents, lead 1"""
    est = lambda k: float(px["1"][k]["est"])                             # noqa: E731
    lo = lambda k: float(px["1"][k]["ci"][0])                            # noqa: E731
    attr = "LAW" if lo("d_nl4_an4_37.5") > 0 else "NOT-LAW"
    beats16 = est("d_fr16_an4_37.5") > 0 and est("d_fr16_an4_75.0") > 0 and lo("d_fr16_an4_37.5") > 0
    beats4 = lo("d_fr4_an4_37.5") > 0
    if beats16 and attr == "LAW":
        rule = "SUCCESS"
    elif beats4 and attr == "LAW":
        rule = "ORDER"
    elif beats16 or beats4:
        rule = "DRESSING"
    else:
        rule = "FAIL"
    return dict(outcome="INVALID" if invalid else rule, rule=rule, attribution=attr, beats16=beats16, beats4=beats4,
                invalid_reasons=list(invalid))


def bias_guard(st):
    return {f"bias_L{L}_within_tol": abs(float(st[L]["bias_rel"]["est"])) <= TOL_BIAS for L in ("1", "3")}


# ======================================================================== data
def load_common():
    pos = C.positions()
    S["pos"] = pos
    S["NP"], S["NST"] = int(pos["NP"]), int(pos["NST"])
    S["stcol"] = pos["stcol"]
    z = np.load(C.DATA / "law20.npz")
    S["law"] = {k: z[k] for k in z.files}
    meta = np.load(C.DATA / "meta20.npz")
    S["hashes"] = json.loads(str(meta["sha256"]))
    S["K2"] = {(int(d), int(h)): (int(r), int(n), int(t)) for d, h, r, n, t in meta["keys_g2"]}
    S["blocks"] = S["law"]["blocks"]
    FG, FN = (np.load(C.DATA / f"{n}.npy", mmap_mode="r") for n in ("feat_g2", "feat_null"))
    M2, M11 = (np.load(C.DATA / f"{n}.npy", mmap_mode="r") for n in ("mem_g2", "mem_g11"))
    NP = S["NP"]
    nnp = int((pos["nullrows"] < NP).sum())
    S["nnp"] = nnp
    S["FST"] = np.stack([np.asarray(FG[C.LEADS3.index(h), :, :, NP:]) for h in C.LEADS])     # (2, 179, 4, NST, 12)
    S["FNST"] = np.asarray(FN[:, :, :, nnp:])                                                 # (2, 179, 4, NST, 12)
    S["MST2"], S["MST11"] = np.asarray(M2[:, :, NP:]), np.asarray(M11[:, :, NP:])
    assert S["FNST"].shape[3] == S["NST"]


def law_q2(li, b):
    L = S["law"]
    return float(L["sL2"][li, b]), L["kappa2"][li, b], L["tau2"][li, b]


def data_guard():
    with ThreadPoolExecutor(4) as ex:
        got = dict(zip(S["hashes"], ex.map(lambda nm: C.sha_file(C.DATA / f"{nm}.npy"), S["hashes"])))
    return got, all(got[k] == S["hashes"][k] for k in got)


# ======================================================================== stations
def station_block(prep):
    """per forecast at leads 1 and 3: the S' rows, and the paper's full-sample sums for the identity (score19 arithmetic)"""
    import score19 as s19
    days = prep["days"]
    NDs = len(days)
    Fd = {k: np.zeros((NDs, 2)) for k in s19.FORM_COLS}
    Z16 = np.load(C.ROOT / "a14" / "classU" / "data_ar" / "st16_run16.npz")
    K16 = {(int(d), int(h)): i for i, (d, h, t) in enumerate(Z16["keys"])}
    E16a = Z16["ens16"]
    stpos_equal = bool(np.array_equal(Z16["stpos"], S["pos"]["stpos"]))
    stcol = S["stcol"]
    has_all = stcol >= 0
    st16_ok, recs = True, {}
    nst = {"S_prime_station_days": [0, 0], "full_station_days": [0, 0], "dropped_nonfinite_at_populated_stations": [0, 0]}
    for key in sorted(prep["rec"], key=lambda k: (k[1], k[0])):
        d, h = key
        r = prep["rec"][key]
        li, di, j = C.LEADS.index(h), days.index(d), C.DAYS.index(d)
        g16all = np.full((len(stcol), 16), np.nan, np.float16)
        g16all[has_all] = S["MST11"][j, li][stcol[has_all]]
        st16_ok &= bool(np.array_equal(g16all.view(np.uint16), E16a[K16[key]].view(np.uint16)))
        s = r["s"]
        y64 = r["y"].astype(np.float64)
        E16 = g16all[s].astype(np.float32); E4 = r["E4"]                 # ---- score19.station_block, the identity sums
        f16 = np.isfinite(E16).all(1); f4 = ~f16
        x16 = E16.mean(1).astype(np.float64); v16 = E16.var(1, ddof=1).astype(np.float64)
        x4 = E4.mean(1).astype(np.float64); v4 = E4.var(1, ddof=1).astype(np.float64)
        Fd["one"][di, li] = len(s)
        Fd["sseF"][di, li] = ((x16[f16] - y64[f16]) ** 2).sum() + ((x4[f4] - y64[f4]) ** 2).sum()
        Fd["s2MF"][di, li] = v16[f16].sum() / 16 + v4[f4].sum() / 4
        Fd["s2rawF"][di, li] = v16[f16].sum() + v4[f4].sum()
        Fd["sse_g1"][di, li] = ((r["g1"].astype(np.float64) - y64) ** 2).sum()
        Fd["s2_g1"][di, li] = v4.sum()
        nst["full_station_days"][li] += len(s)
        col = stcol[s]
        has = col >= 0
        X4 = S["MST2"][j, li][col[has]].astype(np.float64)
        Y16 = S["MST11"][j, li][col[has]].astype(np.float64)
        fin = np.isfinite(X4).all(1) & np.isfinite(Y16).all(1)
        nst["dropped_nonfinite_at_populated_stations"][li] += int((~fin).sum())
        nst["S_prime_station_days"][li] += int(fin.sum())
        sel = lambda a: a[has][fin]                                      # noqa: E731
        recs[key] = dict(j=j, li=li, di=di, b=int(S["blocks"][j]), sc=col[has][fin], X4=X4[fin], Y16=Y16[fin],
                         y=sel(y64), cams=sel(r["cams"].astype(np.float64)), aur=sel(r["aur"].astype(np.float64)),
                         g1=sel(r["g1"].astype(np.float64)))
    ok_d = np.isfinite(prep["PD"][0, :, 0, 0])
    return recs, Fd, ok_d, nst, dict(stpos_equal_a14_st16=stpos_equal, g11_station_members_equal_a14_st16=bool(st16_ok))


def st_feats(arr, li, j, sc):
    return np.asarray(arr[li, j][:, sc], np.float64).transpose(1, 0, 2)          # (n, 4, 12)


# ======================================================================== the cases of the tests
GU = None
SW = dict(s=0.25, k=0.25 / np.sqrt(3.0))                             # linear worlds: R^2 = s^2 / (s^2 + k^2) = 0.75
SL = dict(s=0.35, k=0.10)                                            # log world: A18's law_log
WORLDS = ("W_law", "W_none", "W_wrong", "W_bias", "W_log")
REALS = ("R_null_member", "R_null_day", "R_mismatch", "R_primary")
CASE_ID = {c: i for i, c in enumerate(WORLDS + REALS)}
EXPECT = {   # D_rel_V = D / V (per-member variance). Targets 'MC': the expectation of the world itself, by a Monte Carlo with
    # i.i.d. Gaussian noise and the attenuated slope (mc_target; no data). Development run (job 23920273, kept as
    # out/tests20.dev.json): the first-order targets R^2/4 and -3 R^2/4 omitted the second-order terms, and R_mismatch was
    # expected FAIL although its bias is sigmaL2 / 4 (2.3); both corrected here, before the recorded run.
    "W_law": dict(desc="linear law of known R^2 = 0.75 (first order D / V = R^2 / 4)", Q1="SUCCESS", D_rel_V="MC",
                  meq_tf_corr=(12.0, 22.0)),
    "W_none": dict(desc="no law (members driven by the noise of members 4-7): the fitted law is about 0", Q1="FAIL",
                   D_rel_V=0.0, D_interval_holds_0=True, Q2_attribution="NOT-LAW"),
    "W_wrong": dict(desc="law of the wrong sign (members 1 - s u, law fitted on W_law; first order D / V = -3 R^2 / 4): its "
                         "bias +s^2 / 2 trips the bias guard", Q1_rule="FAIL", Q1="INVALID", D_rel_V="MC", bias_guard_fails=True),
    "W_bias": dict(desc="the right law with an offset 0.05 in g: lower variance, mean biased by about -4.6 %",
                   Q1="INVALID", Q1_rule="SUCCESS", D_rel_V="MC", bias_guard_fails=True),
    "W_log": dict(desc="log-form law (A18 law_log, s 0.35, kappa 0.10)", Q1="SUCCESS", Q2="SUCCESS", Q2_attribution="LAW"),
    "R_null_member": dict(desc="real members; law fitted and applied with the features of members 4-7 (independent noise)",
                          Q1="FAIL", D_interval_holds_0=True),
    "R_null_day": dict(desc="real members; law fitted and applied with the features of another start day", Q1="FAIL",
                       D_interval_holds_0=True),
    "R_mismatch": dict(desc="real members; the primary law applied to the features of members 4-7: the variance rises and "
                            "the mean is biased by sigmaL2 / 4 (1.25 % / 2.0 %), so the bias guard fires", Q1="INVALID",
                       Q1_rule="FAIL", D_1_below_0=True, bias_guard_fails=True),
    "R_primary": dict(desc="real members, the primary law, member-drawn truth (G11 member 15): a members-only prediction "
                           "of D (reported, no expectation)"),
}


def mc_target(case, n=4_000_000):
    """the expected D / V of a linear synthetic world: i.i.d. N(0, 1) u and e, the least-squares slope attenuated by the
    level xbar, beta = s / (1 + (s^2 + k^2) / 4) (fit20 --tests), truth one more independent draw (no data)"""
    s, k = SW["s"], SW["k"]
    beta = s / (1 + (s * s + k * k) / 4)
    sg, off = (-1.0, 0.0) if case == "W_wrong" else ((1.0, 0.05) if case == "W_bias" else (1.0, 0.0))
    rng = np.random.default_rng([20261121, 20, 99, CASE_ID[case]])
    u, e = rng.standard_normal((n, 4)), rng.standard_normal((n, 4))
    x = 1 + sg * s * u + k * e
    cr = C.correct(x, beta * u + off)
    y = 1 + s * rng.standard_normal(n) + k * rng.standard_normal(n)
    return float((((cr["plain"] - y) ** 2).mean() - ((cr["corr"] - y) ** 2).mean()) / (s * s + k * k))


def synth(case, mu, F_own, F_null, rng):
    """synthetic members (n, 4), truth (n,) and sixteen independent members (n, 16) around the level mu"""
    n = len(mu)
    if case == "W_log":
        s, k = SL["s"], SL["k"]
        f = lambda u, e: mu[:, None] * np.exp(s * u - s * s / 2 + k * e - k * k / 2) if u.ndim == 2 else \
            mu * np.exp(s * u - s * s / 2 + k * e - k * k / 2)               # noqa: E731
    else:
        s, k = SW["s"], SW["k"]
        sg = -1.0 if case == "W_wrong" else 1.0
        f = lambda u, e: mu[:, None] * (1 + sg * s * u + k * e) if u.ndim == 2 else mu * (1 + s * u + k * e)   # noqa: E731
    u = (F_null if case == "W_none" else F_own) @ GU
    X = f(u, rng.standard_normal((n, 4)))
    y = f(rng.standard_normal(n), rng.standard_normal(n))
    Y16 = np.stack([f(rng.standard_normal(n), rng.standard_normal(n)) for _ in range(16)], 1)
    return X, y, Y16


def fit_world(case):
    """the law of a synthetic world, fitted by common20 on its members at the null pixel rows (cross-fitted blocks)"""
    St = [C.Stats(C.NF, 1, True) for _ in range(2)]
    for li in range(2):
        for j in range(C.ND):
            X, F, FNl, mu = px_rows(li, j)
            Xs, _, _ = synth(case, mu, F, FNl, np.random.default_rng([20261121, 20, CASE_ID[case], li, j, 1]))
            ok = S["px_ok"][j, li]
            St[li].add(int(S["blocks"][j]), 0, Xs[ok], F[ok], np.ones(int(ok.sum())))
    gam = np.stack([C.cross_fit(St[li], 0)[0] for li in range(2)])
    q2 = [C.q2_params(St[li], gam[li], 0) for li in range(2)]
    return dict(gam=gam, off=0.0, sL2=np.stack([q["sL2"] for q in q2]), kappa2=np.stack([q["kappa2"] for q in q2]),
                tau2=np.stack([q["tau2"] for q in q2]), R2=[C.r2(St[li], gam[li], 0) for li in range(2)])


def px_rows(li, j):
    """real G2 members, their features, the features of members 4-7, and the synthetic level mu = the mean of G11's
    members 4-15 (independent of G2's noise: the plain mean of G2's four carries their own noise law)"""
    X = S["MP2"][j, li].astype(np.float64)
    F = S["FP"][li, j].astype(np.float64).transpose(1, 0, 2)
    FNl = S["FPN"][li, j].astype(np.float64).transpose(1, 0, 2)
    mu = np.maximum(np.nan_to_num(S["MP11"][j, li][:, 4:].astype(np.float64)).mean(1), 0.5)
    return X, F, FNl, mu


def case_forecast(job):
    """station and pixel day sums of one test case for one forecast (li, j)"""
    case, li, j = job
    law = S["laws"][case]
    b = int(S["blocks"][j])
    gl = lambda F: F @ law["gam"][li, b] + law["off"]                    # noqa: E731
    out = {"st": None, "px": None}
    key = (C.DAYS[j], C.LEADS[li])
    rec = S["recs"].get(key)
    for where in ("st", "px"):
        if where == "st":
            if rec is None:
                continue
            sc = rec["sc"]
            F, FNl = st_feats(S["FST"], li, j, sc), st_feats(S["FNST"], li, j, sc)
            X, Y16 = rec["X4"], rec["Y16"]
            mu = np.maximum(Y16[:, 4:].mean(1), 0.5)                    # level independent of G2's noise (G11 4-15)
            ok = np.ones(len(X), bool)
            w = None
        else:
            X, F, FNl, mu = px_rows(li, j)
            Y16 = S["MP11"][j, li].astype(np.float64)
            ok = S["px_ok"][j, li]
            X, F, FNl, mu, Y16 = X[ok], F[ok], FNl[ok], mu[ok], Y16[ok]
            w = S["pxpop"][ok]
        if case in WORLDS:
            rng = np.random.default_rng([20261121, 20, CASE_ID[case], li, j, 0 if where == "st" else 1])
            Xc, y, YF = synth(case, mu, F, FNl, rng)
            # at the pixels these are the members the world's law was fitted on: block b is scored with the law of the others
            col, cr = q1_columns(Xc, {"corr": gl(F)}, YF, YF[:, 4:], y, {})
            if where == "px":
                col.update(q2_columns(Xc, cr, YF, y, (float(law["sL2"][li, b]), law["kappa2"][li, b], law["tau2"][li, b])))
        else:
            y, YF, YR = Y16[:, 15], Y16[:, :15], Y16[:, 4:15]
            if case == "R_null_member":
                Fa = FNl
            elif case == "R_null_day":
                jn = (j + 1) % C.ND
                if where == "st":
                    Fa = st_feats(S["FST"], li, jn, sc)
                else:
                    Fa = S["FP"][li, jn].astype(np.float64).transpose(1, 0, 2)[ok]
            elif case == "R_mismatch":
                Fa = FNl
            else:
                Fa = F
            col, cr = q1_columns(X, {"corr": gl(Fa)}, YF, YR, y, {})
        out[where] = (sums_of(col), sums_of(col, w)) if where == "px" else sums_of(col)
    return case, li, j, out


def identity_forecast(key):
    """the plain four (no correction: g = 0) and the forms against the real station truth, for the identity only"""
    rec = S["recs"][key]
    X, Y16 = rec["X4"], rec["Y16"]
    col, _ = q1_columns(X, {"corr": np.zeros_like(X)}, Y16, Y16[:, 4:], rec["y"],
                        {"cams": rec["cams"], "aur": rec["aur"], "g1": rec["g1"]})
    keep = ("one", "y", "sse_p", "sse_F", "s2F", "V", "tfp", "xbar", "sse_cams", "sse_aur", "sse_g1", "s2_4")
    return key, {k: v for k, v in sums_of(col).items() if k in keep}


def identity_block(prep, recs, Fd, ok_d, st_ident):
    import score19 as s19
    sid, sid_ok = s19.station_identity(prep, Fd, ok_d)
    days = prep["days"]
    Sd = {}
    for key in recs:
        _, cs = identity_forecast(key)
        for k, v in cs.items():
            Sd.setdefault(k, np.zeros((len(days), 2)))[recs[key]["di"], recs[key]["li"]] = v
    W = C.boot_counts(int(ok_d.sum()))
    t = {}
    res = {}
    for L in ("1", "3"):
        li = int(L == "3")
        v = {k: a[ok_d, li] for k, a in Sd.items()}
        n = v["one"].sum()
        e = lambda k: np.concatenate([[v[k].sum()], W @ v[k]])           # noqa: E731
        N = e("one")
        VF = e("s2F") / N
        inf = e("sse_F") / N - VF / 16
        t[L] = dict(rmse_4=C.ci(np.sqrt(e("sse_p") / N)), rmse_form8=C.ci(np.sqrt(inf + VF / 8)), rmse_inf=C.ci(np.sqrt(inf)),
                    n=float(n), red_four_vs_cams=C.ci(100 * (1 - np.sqrt(e("sse_p") / e("sse_cams")))),
                    red_four_vs_aur=C.ci(100 * (1 - np.sqrt(e("sse_p") / e("sse_aur")))))
        rel = lambda a, b: abs(a - b) / abs(b)                           # noqa: E731
        res[L] = {"rmse_4": rel(t[L]["rmse_4"]["est"], A19_SPRIME[L]["rmse_4"]) <= 1e-9,
                  "rmse_form8": rel(t[L]["rmse_form8"]["est"], A19_SPRIME[L]["rmse_form8"]) <= 1e-9,
                  "rmse_inf": rel(t[L]["rmse_inf"]["est"], A19_SPRIME[L]["rmse_inf"]) <= 1e-9,
                  "n": int(n) == A19_SPRIME[L]["n"]}
    ident = dict(compare_fair=prep["ident"], compare_fair_sha256=prep["cf_sha"], stations_full_sample=sid,
                 stations_full_sample_passed=bool(sid_ok), station_members=st_ident, S_prime_plain_and_forms=t,
                 S_prime_equal_A19=res)
    ok = (all(v is True or (isinstance(v, int) and not isinstance(v, bool)) for v in prep["ident"].values()) and bool(sid_ok)
          and all(st_ident.values()) and all(all(v.values()) for v in res.values()))
    return ident, bool(ok)


def near(v, t, tol):
    lo, hi = v["ci"]
    return dict(est=v["est"], target=t, ci=[lo, hi], ok=bool(abs(v["est"] - t) <= max(1.5 * (hi - lo) / 2, tol)))


def run_tests():
    global GU
    t0 = time.time()
    got, dep_ok = deps_ok()
    C.log("deps", json.dumps(got), "equal to DEP:", dep_ok)
    load_common()
    import score19 as s19                                               # noqa: F401  (hash checked in main)
    prep = s19.station_prep()
    recs, Fd, ok_d, nst, st_ident = station_block(prep)
    S["recs"] = recs
    ident, id_ok = identity_block(prep, recs, Fd, ok_d, st_ident)
    C.log("identity", id_ok, json.dumps(C.jsonable(ident["S_prime_equal_A19"])))
    hashes, data_ok = data_guard()
    # ---- the null pixel rows (100,000 pixels), preloaded
    pos = S["pos"]
    NR = pos["nullrows"][:S["nnp"]]
    FG, FN = (np.load(C.DATA / f"{n}.npy", mmap_mode="r") for n in ("feat_g2", "feat_null"))
    M2, M11 = (np.load(C.DATA / f"{n}.npy", mmap_mode="r") for n in ("mem_g2", "mem_g11"))

    def ld(li_j):
        li, j = li_j
        return li, j, np.asarray(FG[C.LEADS3.index(C.LEADS[li]), j][:, NR]), np.asarray(FN[li, j][:, :S["nnp"]])
    S["FP"] = np.zeros((2, C.ND, 4, len(NR), C.NF), np.float16)
    S["FPN"] = np.zeros((2, C.ND, 4, len(NR), C.NF), np.float16)
    with ThreadPoolExecutor(8) as ex:
        for li, j, a, b in ex.map(ld, [(li, j) for li in range(2) for j in range(C.ND)]):
            S["FP"][li, j], S["FPN"][li, j] = a, b
    S["MP2"], S["MP11"] = np.asarray(M2[:, :, NR]), np.asarray(M11[:, :, NR])
    S["px_ok"] = np.isfinite(S["MP2"]).all(3) & np.isfinite(S["MP11"]).all(3)                   # (179, 2, n)
    S["pxpop"] = pos["pop"][NR].astype(np.float64)
    C.log(f"null pixel rows preloaded ({len(NR)}), {time.time() - t0:.0f} s")
    GU = np.load(C.DATA / "law20.npz")["gamma_primary_full"][0]
    GU = GU / np.sqrt(GU @ C.CFEAT @ GU)                             # the real law's profile, unit variance under white noise
    L = S["law"]
    laws = {}
    with get_context("fork").Pool(2) as pool:
        for case, lw in zip(("W_law", "W_none", "W_log"), pool.map(fit_world, ("W_law", "W_none", "W_log"))):
            laws[case] = lw
    laws["W_wrong"] = dict(laws["W_law"])
    laws["W_bias"] = dict(laws["W_law"], off=0.05)
    laws["R_null_member"] = dict(gam=np.stack([L[f"gamma_null_m_lead{h}"] for h in C.LEADS]), off=0.0)
    laws["R_null_day"] = dict(gam=np.stack([L[f"gamma_null_d_lead{h}"] for h in C.LEADS]), off=0.0)
    laws["R_mismatch"] = dict(gam=L["gamma_primary"], off=0.0)
    laws["R_primary"] = dict(gam=L["gamma_primary"], off=0.0)
    S["laws"] = laws
    C.log("laws of the test cases", json.dumps({c: v.get("R2") for c, v in laws.items()}), f"{time.time() - t0:.0f} s")
    jobs = [(c, li, j) for c in WORLDS + REALS for li in range(2) for j in range(C.ND)]
    days = prep["days"]
    Sst = {c: {} for c in WORLDS + REALS}
    Spx = {c: ({}, {}) for c in WORLDS + REALS}
    with get_context("fork").Pool(16) as pool:
        for n, (case, li, j, out) in enumerate(pool.imap_unordered(case_forecast, jobs, chunksize=4)):
            if out["st"] is not None:
                di = days.index(C.DAYS[j])
                for k, v in out["st"].items():
                    Sst[case].setdefault(k, np.zeros((len(days), 2)))[di, li] = v
            for wi in range(2):
                for k, v in out["px"][wi].items():
                    Spx[case][wi].setdefault(k, np.zeros((C.ND, 2)))[j, li] = v
            if n % 500 == 0:
                C.log(f"cases {n + 1}/{len(jobs)} {time.time() - t0:.0f} s")
    Ws, Wp = C.boot_counts(int(ok_d.sum())), C.boot_counts(C.ND)
    cases = {}
    base_invalid = ([] if id_ok else ["identity checks"]) + ([] if data_ok else ["data hashes"]) + \
        ([] if sum(nst["dropped_nonfinite_at_populated_stations"]) == 0 else ["station-days dropped"]) + \
        ([] if dep_ok else ["DEP hashes"])
    for c in WORLDS + REALS:
        nF = 16 if c in WORLDS else 15
        st = boot_stats({k: v[ok_d] for k, v in Sst[c].items()}, Ws, nF)
        px = {wn: boot_stats(Spx[c][wi], Wp, nF) for wi, wn in enumerate(("pixels", "residents"))}
        bg = bias_guard(st)
        inv = base_invalid + [k for k, v in bg.items() if not v]
        q1 = decide_q1(st, inv)
        q2 = decide_q2(px["residents"], inv) if c in WORLDS else None
        E = EXPECT[c]
        chk = {}
        if "Q1" in E:
            chk["Q1"] = q1["outcome"] == E["Q1"]
        if "Q1_rule" in E:
            chk["Q1_rule"] = q1["rule"] == E["Q1_rule"]
        if "Q1_in" in E:
            chk["Q1_in"] = q1["outcome"] in E["Q1_in"]
        if "D_rel_V" in E:
            tgt = mc_target(c) if E["D_rel_V"] == "MC" else E["D_rel_V"]
            for L_ in ("1", "3"):
                chk[f"D_rel_V_L{L_}"] = near(st[L_]["D_rel_V"], tgt, 0.02)
        if "meq_tf_corr" in E:
            chk["meq_tf_corr_L1_in_range"] = E["meq_tf_corr"][0] <= st["1"]["meq_tf_corr"]["est"] <= E["meq_tf_corr"][1]
        if E.get("D_interval_holds_0"):
            for L_ in ("1", "3"):
                lo, hi = st[L_]["D"]["ci"]
                chk[f"D_L{L_}_interval_holds_0"] = dict(ci=[lo, hi], ok=bool(lo <= 0 <= hi))
        if E.get("D_1_below_0"):
            chk["D_1_below_0"] = st["1"]["D"]["est"] < 0
        if E.get("bias_guard_fails"):
            chk["bias_guard_fails"] = not all(bg.values())
        if "Q2" in E:
            chk["Q2"] = q2["outcome"] == E["Q2"]
        if "Q2_attribution" in E:
            chk["Q2_attribution"] = q2["attribution"] == E["Q2_attribution"]
        chk["G3_plain_truth_free_identity"] = near(st["13"]["inv_meq_tf_plain"], 0.25, 0.01)
        passed = all(v if isinstance(v, bool) else v["ok"] for v in chk.values())
        cases[c] = dict(expect=E, checks=chk, passed=bool(passed), Q1=q1, Q2=q2, bias_guard=bg,
                        stations=headline(st), pixels={wn: headline(v) for wn, v in px.items()},
                        law_R2_fit=laws[c].get("R2"))
        C.log(f"case {c}: {'PASS' if passed else 'FAIL'} Q1 {q1['outcome']} ({q1['rule']}) Q2 {q2 and q2['outcome']} "
              f"D_1 {st['1']['D']} D_rel_V_1 {st['1']['D_rel_V']['est']:.4f} bias_1 {st['1']['bias_rel']['est']:.5f}")
    # ---- dry run of the decision path with a member-drawn truth (the 1 km estimates are not opened), and its refusal
    dst, dpx, dnv = decision_scores(recs, days, ok_d, pseudo=True)
    dbg = bias_guard(dst)
    dinv = base_invalid + [k for k, v in dbg.items() if not v]
    dry = dict(note="the decision's own code with G11's member 15 as the truth (members only); not a score of A20",
               Q1=decide_q1(dst, dinv), Q2=decide_q2(dpx["residents"], dinv), bias_guard=dbg, pixel_forecasts_valid=dnv,
               stations=headline(dst), pixels={wn: headline(v) for wn, v in dpx.items()})
    try:
        run_decision()
        dry["decision_refused_without_FILED"] = False
    except AssertionError as e:
        dry["decision_refused_without_FILED"] = "FILED" in str(e)
    C.log("dry decision", json.dumps(C.jsonable({k: dry[k] for k in ("Q1", "Q2", "decision_refused_without_FILED")}))[:600])
    R = {"what": __doc__, "score20_sha256": SELF_SHA256, "mode": "tests (no corrected quantity against a 2022 truth)",
         "dry_decision_member_truth": dry,
         "deps": got, "deps_equal_DEP": dep_ok, "TOL_BIAS": TOL_BIAS, "identity": ident, "identity_passed": id_ok,
         "data_hashes": hashes, "data_passed": data_ok, "station_sample": nst, "cases": cases,
         "worlds": dict(linear=SW, log=SL, profile_GU=GU)}
    R["all_passed"] = bool(id_ok and data_ok and dep_ok and all(v["passed"] for v in cases.values())
                           and dry["decision_refused_without_FILED"] is True)
    R["seconds"] = round(time.time() - t0)
    C.save_json(C.OUT / "tests20.json", R)
    C.log("ALL PASSED" if R["all_passed"] else "SOME TESTS FAILED", f"score20 sha256 {SELF_SHA256}", f"{R['seconds']} s")
    return 0 if R["all_passed"] else 1


HEAD = ("n", "mean_y", "rmse_p", "rmse_c", "rmse_n", "rmse_f", "rmse_A", "rmse_form8", "rmse_inf", "D", "D_rel_V", "D_noiter",
        "D_fixpt", "D_A18law", "d_form8_minus_corr", "meq_tf_corr", "meq_tf_plain", "meq_corr_vs_forms", "meq_plain_vs_forms",
        "bias_rel", "bias_rel_A18law", "R2_law", "spread_corr_over_plain", "V_member", "red_plain4_vs_cams",
        "red_corr4_vs_cams", "red_form8_vs_cams", "red_inf_vs_cams", "red_A18law4_vs_cams", "red_plain4_vs_aur",
        "red_corr4_vs_aur", "red_form8_vs_aur", "red_inf_vs_aur", "red_A18law4_vs_aur")
HEADQ2 = tuple(f"{k}_{T}" for T in THR for k in ("event_share", "bs_an4", "bs_ga4", "bs_nl4", "bs_tk4", "bs_fr4", "bs_fr16",
                                                 "d_nl4_an4", "d_fr16_an4", "d_fr4_an4", "d_tk4_an4", "d_ga4_an4", "d_fr16_nl4",
                                                 "warned_an4", "correct_an4", "reach_an4", "warned_fr16", "correct_fr16",
                                                 "reach_fr16", "warned_fr4", "correct_fr4", "reach_fr4"))


def headline(st):
    return {L: {k: st[L][k] for k in HEAD + HEADQ2 if k in st[L]} for L in st}


# ======================================================================== the decision (after the filing)
def pixel_forecast(job):
    """the real case at the 1,000,000 sampled pixels for one forecast: Q1 and Q2 columns, both weights. pseudo: the truth is
    G11's member 15 (forms from G11 0-14, reference 4-14) and the 1 km estimates are not opened (the dry run of the tests)"""
    li, j, pseudo = job
    h, d = C.LEADS[li], C.DAYS[j]
    b = int(S["blocks"][j])
    NP = S["NP"]
    X = np.asarray(np.load(C.DATA / "mem_g2.npy", mmap_mode="r")[j, li, :NP], np.float64)
    Y16 = np.asarray(np.load(C.DATA / "mem_g11.npy", mmap_mode="r")[j, li, :NP], np.float64)
    if pseudo:
        y, YF, YR = Y16[:, 15].copy(), Y16[:, :15], Y16[:, 4:15]
    else:
        r2, n2, _ = S["K2"][(d, h)]
        y = np.asarray(np.load(C.G2D / f"pop_ghap_{r2}.npy", mmap_mode="r")[n2], np.float64)[S["pos"]["pixpos"]]
        YF, YR = Y16, Y16[:, 4:]
    F = np.asarray(np.load(C.DATA / "feat_g2.npy", mmap_mode="r")[C.LEADS3.index(h), j, :, :NP], np.float64).transpose(1, 0, 2)
    ok = np.isfinite(y) & np.isfinite(X).all(1) & np.isfinite(Y16).all(1)
    X, YF, YR, F, y = X[ok], YF[ok], YR[ok], F[ok], y[ok]
    L = S["law"]
    gd = {"corr": F @ L["gamma_primary"][li, b], "A18": F @ L["gamma_A18"][li]}
    col, cr = q1_columns(X, gd, YF, YR, y, {})
    col.update(q2_columns(X, cr, YF, y, law_q2(li, b)))
    w = S["pos"]["pop"][:NP][ok].astype(np.float64)
    return li, j, (sums_of(col), sums_of(col, w)), int(ok.sum())


def decision_scores(recs, days, ok_d, pseudo):
    """the decision's station and pixel statistics (pseudo: member-drawn truth, for the dry run of the tests)"""
    L = S["law"]
    Sd = {}
    for key, rec in recs.items():
        li, j, b, sc = rec["li"], rec["j"], rec["b"], rec["sc"]
        F = st_feats(S["FST"], li, j, sc)
        gd = {"corr": F @ L["gamma_primary"][li, b], "A18": F @ L["gamma_A18"][li]}
        Y16 = rec["Y16"]
        if pseudo:
            y, YF, YR, refs = Y16[:, 15], Y16[:, :15], Y16[:, 4:15], {}
        else:
            y, YF, YR, refs = rec["y"], Y16, Y16[:, 4:], {"cams": rec["cams"], "aur": rec["aur"]}
        col, _ = q1_columns(rec["X4"], gd, YF, YR, y, refs)
        for k, v in sums_of(col).items():
            Sd.setdefault(k, np.zeros((len(days), 2)))[rec["di"], li] = v
    nF = 15 if pseudo else 16
    st = boot_stats({k: v[ok_d] for k, v in Sd.items()}, C.boot_counts(int(ok_d.sum())), nF)
    Spx = ({}, {})
    nvalid = 0
    with get_context("fork").Pool(16) as pool:
        for li, j, sw, nv in pool.imap_unordered(pixel_forecast, [(li, j, pseudo) for li in range(2) for j in range(C.ND)],
                                                 chunksize=1):
            nvalid += nv
            for wi in range(2):
                for k, v in sw[wi].items():
                    Spx[wi].setdefault(k, np.zeros((C.ND, 2)))[j, li] = v
    Wp = C.boot_counts(C.ND)
    px = {wn: boot_stats(Spx[wi], Wp, nF) for wi, wn in enumerate(("pixels", "residents"))}
    return st, px, nvalid


def run_decision():
    t0 = time.time()
    filed = C.A20 / "FILED"
    assert filed.exists() and SELF_SHA256 in filed.read_text(), "A20: the decision runs only after the filing (a20/FILED)"
    got, dep_ok = deps_ok()
    assert TOL_BIAS is not None, "TOL_BIAS not set"
    load_common()
    import score19 as s19
    prep = s19.station_prep()
    recs, Fd, ok_d, nst, st_ident = station_block(prep)
    S["recs"] = recs
    ident, id_ok = identity_block(prep, recs, Fd, ok_d, st_ident)
    hashes, data_ok = data_guard()
    st, px, nvalid = decision_scores(recs, prep["days"], ok_d, pseudo=False)
    C.log(f"stations and pixels scored {time.time() - t0:.0f} s")
    bg = bias_guard(st)
    inv = ([] if id_ok else ["identity checks"]) + ([] if data_ok else ["data hashes"]) + ([] if dep_ok else ["DEP hashes"]) + \
        ([] if sum(nst["dropped_nonfinite_at_populated_stations"]) == 0 else ["station-days dropped"]) + \
        [k for k, v in bg.items() if not v]
    q1 = decide_q1(st, inv)
    q2 = decide_q2(px["residents"], inv)
    q1_px = {wn: decide_q1(v, inv) for wn, v in px.items()}
    R = {"what": __doc__, "score20_sha256": SELF_SHA256, "mode": "decision", "Q1": q1, "Q2": q2,
         "Q1_rule_at_populated_pixels_reported": q1_px, "guards": dict(bias=bg, identity=id_ok, data=data_ok, deps=dep_ok,
                                                                      station_days_dropped=nst["dropped_nonfinite_at_populated_stations"]),
         "TOL_BIAS": TOL_BIAS, "headline_stations": headline(st), "headline_pixels": {wn: headline(v) for wn, v in px.items()},
         "stations": st, "pixels": px, "station_sample": nst, "pixel_forecasts_valid": nvalid, "identity": ident,
         "data_hashes": hashes, "deps": got, "seconds": round(time.time() - t0)}
    C.save_json(C.OUT / "score20.json", R)
    C.log(f"OUTCOME Q1 {q1['outcome']} Q2 {q2['outcome']} | score20 sha256 {SELF_SHA256}")
    return 0


def main():
    assert np.__version__ == "1.26.3", f"A20 scores in the job container (numpy 1.26.3), not {np.__version__}"
    assert C.sha_file(C.A19C / "score19.py") == SCORE19_SHA256, "score19.py is not the frozen A19 scorer"
    sys.path.insert(0, str(C.A19C))
    C.log("score20 sha256", SELF_SHA256, "mode", MODE)
    return run_tests() if MODE == "tests" else run_decision()


if __name__ == "__main__":
    sys.exit(main())
