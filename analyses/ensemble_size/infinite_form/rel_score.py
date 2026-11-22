#!/usr/bin/env python3
"""A14 class U for the scores of analyses/reliability (stations: spread/skill, Brier skill, CRPS against the climatological
ensemble, RMSE on the station-days of the probabilistic scores; populated pixels; land of the nine regions) and of
analyses/emissions (gains of the emissions and of the transport, recs of the four models).
As run: the paper's functions taken unchanged from s2b_station_scores.py (pack, unpack, div, derive, jk) and s34_estimates.py
(pop_scores, div) with the paper's bootstrap (rel_common.block_weights: blocks of 7 start days, 10,000 resamples, seed 0) and
emissions/recompute.py (block_counts, rmse, crps, boot, gain); identity against results/station_scores_final.json,
results/estimates_scores.json and emissions/out/gains.csv. Then the M = 8 and infinite forms (NOTES.md item 2) on the same
resamples and their draw variance (item 4).
    python rel_score.py [16-member tags ...]   -> results/rel_classU.json"""
import csv
import glob
import json
import os
import sys

import numpy as np

from u_common import (AR, DATA, DATA0, DESIGN_NOTE, EVF, F_FPC, G, ISFX, REL, RELD, RELR, RES, RES0, RUN, VDRAW0, functions_from,
                      inflate, jk_var, safe_json)
from u_quart import full_report

sys.path.insert(0, str(REL / "scripts"))
from rel_common import block_weights  # noqa: E402  (unchanged)
MIN16 = 20 if os.environ.get("A14U_DRY179") == "1" else 170   # the 179-day branches; 20 only in the dry run of r1_pipeline_b.py (run16 standing in for R1)

TAGS = sys.argv[1:] or ["run16"]
OUT = {}


def cil(b):
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)
    return lo, hi


def close(a, b, tol=5e-7):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return bool(a.shape == b.shape and np.allclose(a, b, rtol=tol, atol=tol, equal_nan=True))


def form(pt, bs, v_draw=None, fixed=False):
    """fixed=True: a stored model with fixed fields (the models before the CRPS training, the emission control, the transport
    ablation; A16 4.4): its V_draw is kept under every design. Under VDRAW0 (A16 4.3) the final model's factor is 1."""
    lo, hi = cil(bs)
    e = {"v": np.asarray(pt).tolist(), "ci_boot": [np.asarray(lo).tolist(), np.asarray(hi).tolist()]}
    if v_draw is not None:
        vb = np.nanvar(bs, axis=0, ddof=1)
        f = np.sqrt(1 + np.asarray(v_draw) / vb)
        if VDRAW0 and not fixed:
            e["factor"] = np.ones_like(np.asarray(f, float)).tolist(); e["v_draw"] = np.asarray(v_draw).tolist()
            e["ci"] = [np.asarray(lo).tolist(), np.asarray(hi).tolist()]
            return e
        e["factor"] = f.tolist(); e["v_draw"] = np.asarray(v_draw).tolist()
        e["ci"] = [(pt - f * (pt - lo)).tolist(), (pt + f * (hi - pt)).tolist()]
    return e


# =========================================================================== (A) stations, analyses/reliability
S2B = REL / "scripts" / "s2b_station_scores.py"
Z = np.load(RELD / "sums_final.npz"); X2 = np.load(DATA / "rel_extra.npz")             # design: reliability/data_ar (S1 step 5)
assert np.array_equal(Z["keys"], X2["keys"])
K = Z["keys"]; GROUPS = [str(g) for g in Z["groups"]]; M = int(Z["members"])
NAMES = [k[5:] for k in Z.files if k.startswith("day__")]
DAY = {k: Z["day__" + k] for k in NAMES}
SHAPE = {k: DAY[k].shape[2:] for k in NAMES}
ns = dict(np=np, M=M, F=(M + 1) / M, NAMES=NAMES, SHAPE=SHAPE)
functions_from(S2B, ["pack", "unpack", "div", "derive", "jk"], ns)
EXTRA = {k[5:]: X2[k] for k in X2.files if k.startswith("day__") and k[5:] not in ("n", "var", "p_n")}
REF = json.loads((RELR / "station_scores_final.json").read_text())
div = ns["div"]


META = {}                                                          # key -> (cv key of Foehn's own score, kind, verification)


def derive_u(S, E, jk=False):
    """M = 8 and infinite forms from the sums S (paper) and E (rel_extra); jk=True: from the delete-one-member sums
    (3 members, last axis = the deleted member). Keys '<score>|<form>|<v>'."""
    o = {}
    x = (lambda a: a[..., None]) if jk else (lambda a: a)
    m = 3 if jk else M
    n = x(S["n"]); var = S["loo_var"] if jk else S["var"]
    for v in (0, 1):
        s = f"_{v}"
        sse = S["loo_sse" + s] if jk else S["sse" + s]
        crps = S["loo_crps" + s] if jk else S["crps" + s]
        pair = E["loo_pair"] if jk else E["pair"]
        for f, c8 in (("inf", 0.0), ("8", 1 / 8)):
            mse = div(sse - var / m + c8 * var, n)
            o[f"rmse|{f}|{v}"] = np.sqrt(mse); META[f"rmse|{f}|{v}"] = ("rmse_" + f, "plain", v)
            o[f"ss|{f}|{v}"] = np.sqrt((9 / 8 if f == "8" else 1.0) * div(var, n)) / np.sqrt(mse)
            META[f"ss|{f}|{v}"] = ("ssr_inf" if f == "inf" else "ssr_fortin_8", "plain", v)
            num = crps + c8 * pair
            o[f"crps|{f}|{v}"] = div(num, n); META[f"crps|{f}|{v}"] = ("crps_fair" if f == "inf" else "crps_8", "plain", v)
            o[f"crps_vs_clim|{f}|{v}"] = 100 * (div(num, x(S["c_crps" + s])) - 1)
            META[f"crps_vs_clim|{f}|{v}"] = ("crps_fair" if f == "inf" else "crps_8", "ratio100", v)
            bf = E["loo_bsf" + s] if jk else S["bsf" + s]
            br = E["loo_bs" + s] if jk else S["bs" + s]
            bs_f = bf if f == "inf" else bf + m * (br - bf) / 8
            o[f"bss_clim|{f}|{v}"] = 1 - div(bs_f, x(S["c_bsf" + s])); META[f"bss_clim|{f}|{v}"] = ("bs_fair" if f == "inf" else "bs_8", "skill", v)
    pn = x(S["p_n"]); pv = S["p_loo_var"] if jk else S["p_var"]; ps = S["p_loo_sse"] if jk else S["p_sse"]
    pc = S["p_loo_crps"] if jk else S["p_crps"]; pp_ = E["p_loo_pair"] if jk else E["p_pair"]
    for f, c8 in (("inf", 0.0), ("8", 1 / 8)):
        num = ps - pv / m + c8 * pv
        o[f"p_rmse|{f}"] = np.sqrt(div(num, pn)); META[f"p_rmse|{f}"] = ("rmse_" + f, "plain", 0)
        o[f"p_ss|{f}"] = np.sqrt((9 / 8 if f == "8" else 1.0) * div(pv, pn)) / o[f"p_rmse|{f}"]
        META[f"p_ss|{f}"] = ("ssr_inf" if f == "inf" else "ssr_fortin_8", "plain", 0)
        for ref in ("cams", "aur"):
            o[f"p_rmse_vs_{ref}|{f}"] = 100 * (np.sqrt(div(num, x(S[f"p_{ref}_se"]))) - 1); META[f"p_rmse_vs_{ref}|{f}"] = ("rmse_" + f, "ratio100", 0)
        o[f"p_crps_vs_climens|{f}"] = 100 * (div(pc + c8 * pp_, x(S["p_clim_crps"])) - 1)
        META[f"p_crps_vs_climens|{f}"] = ("crps_fair" if f == "inf" else "crps_8", "ratio100", 0)
        o[f"p_crps|{f}"] = div(pc + c8 * pp_, pn); META[f"p_crps|{f}"] = ("crps_fair" if f == "inf" else "crps_8", "plain", 0)
    return o


def v_quart_of(key, value, cv):
    """carried quartet draw variance of a reported quantity from the CV of Foehn's own score (NOTES item 4)"""
    kind = META[key][1]
    value = np.asarray(value, float); cv = np.asarray(cv, float)
    if kind == "plain":
        return (cv * np.abs(value)) ** 2
    if kind == "ratio100":                                           # 100 (score_F / score_ref - 1)
        return (100 * cv * (1 + value / 100)) ** 2
    return (cv * (1 - value)) ** 2                                   # skill 1 - BS_F / BS_ref


# ---- sixteen members at the station pixels on the reliability samples (cal): K2 and quartet CVs, leads 1 and 3
st = np.load(G / "stations_2022.npz"); OBS = st["daily"].astype(np.float32); SRC = st["source"]
NETC = [0, 1, 3, 4, 5, 6]
GM = [SRC == s for s in NETC] + [np.isin(SRC, NETC)]
s4 = np.load(RELD / "st_final.npz")
K4 = {(int(d), int(h)): i for i, (d, h, t) in enumerate(s4["keys"])}
THR = (15.0, 37.5, 75.0)
SUF = {(h, v, g): dict(D=np.zeros((16, 16)), A=np.zeros(16), P=np.zeros((16, 16)), E=np.zeros((3, 16, 16)), F=np.zeros((3, 16)),
                       O=np.zeros(3), n=0.0) for h in (1, 3) for v in (0, 1) for g in range(7)}
have16 = set()
PERF = {}                                                          # (h, v) -> list of (start day, D, n) for group 'all'
Q16C = ["n", "s2", "s2M", "sse_0", "sse_1", "crps_0", "crps_1", "pair"] + [f"bsf_{v}_{j}" for v in (0, 1) for j in range(3)] + [f"kmk_{j}" for j in range(3)]
Q16P = ["p_n", "p_s2", "p_s2M", "p_sse", "p_crps", "p_pair"]
ACC16 = {}                                                         # h -> list of (start day, row of K, sums (7 groups, cal+paper quantities, 17))


def ens_terms(X, ys):
    """members X (n, M) float64 and truths ys (list of (n,)): per-station full values and delete-one-member replicates (n, M)"""
    n, M = X.shape
    s1 = X.sum(1); s2_ = (X * X).sum(1)
    xb = s1 / M; var = (s2_ - M * xb * xb) / (M - 1)
    xbr = (s1[:, None] - X) / (M - 1); varr = (s2_[:, None] - X * X - (M - 1) * xbr * xbr) / (M - 2)
    Dm = np.abs(X[:, :, None] - X[:, None, :]); Pt = Dm.sum((1, 2)) / 2; Si = Dm.sum(2)
    pair = Pt / (M * (M - 1)); pairr = (Pt[:, None] - Si) / ((M - 1) * (M - 2))
    out = dict(s2=(var, varr), s2M=(var / M, varr / (M - 1)), pair=(pair, pairr))
    for v, y in enumerate(ys):
        A_ = np.abs(X - y[:, None])
        out[f"sse_{v}"] = ((xb - y) ** 2, (xbr - y[:, None]) ** 2)
        out[f"crps_{v}"] = (A_.mean(1) - pair, (A_.sum(1)[:, None] - A_) / (M - 1) - pairr)
        for j, T_ in enumerate(THR):
            e = (X > T_).astype(np.float64); k = e.sum(1); o = (y > T_).astype(np.float64)
            kr = k[:, None] - e; Mr = M - 1
            out[f"bsf_{v}_{j}"] = ((k / M - o) ** 2 - k * (M - k) / (M * M * (M - 1)), (kr / Mr - o[:, None]) ** 2 - kr * (Mr - kr) / (Mr * Mr * (Mr - 1)))
            if v == 0:
                out[f"kmk_{j}"] = (k * (M - k) / (M * (M - 1)), kr * (Mr - kr) / (Mr * (Mr - 1)))
    return out


def mix_sums(E16, E4, f16, fb, ys, masks):
    """group sums (7, quantities, 17) with sixteen members where finite (f16) and the four stored members elsewhere (fb), the
    sixteen-field jackknife in columns 1-16 (fallback stations: 3-member replicate for i < 4, their full value for i >= 4)"""
    NSx = len(f16)
    t16 = ens_terms(E16[f16].astype(np.float64), [y[f16] for y in ys]) if f16.any() else None
    t4 = ens_terms(E4[fb].astype(np.float64), [y[fb] for y in ys]) if fb.any() else None
    keys = list((t16 or t4).keys())
    res = {}
    for key in keys:
        full = np.zeros(NSx); rep = np.zeros((NSx, 16))
        if t16 is not None:
            full[f16] = t16[key][0]; rep[f16] = t16[key][1]
        if t4 is not None:
            full[fb] = t4[key][0]; rep[fb, :4] = t4[key][1]; rep[fb, 4:] = t4[key][0][:, None]
        res[key] = (full, rep)
    out = {}
    for mname, mk in masks.items():
        mkf = mk.astype(np.float64)
        G_ = GM * mkf[None]                                          # (7, NS)
        cols = {"n": np.repeat((G_.sum(1))[:, None], 17, 1)}
        for key, (full, rep) in res.items():
            fz = np.nan_to_num(full); rz = np.nan_to_num(rep)
            cols[key] = np.concatenate([(G_ @ fz)[:, None], G_ @ rz], 1)
        out[mname] = cols
    return out


for tag in TAGS:
    p16 = DATA / f"st16_{tag}.npz"
    if not p16.exists():
        continue
    z = np.load(p16)
    for r, (d, h, t) in enumerate(z["keys"]):
        d, h, t = int(d), int(h), int(t)
        if h not in (1, 3):
            continue
        have16.add((d, h))
        i4 = K4[(d, h)]
        cal = np.unpackbits(X2["cal_masks"][i4])[:len(SRC)].astype(bool)
        E16 = z["ens16"][r].astype(np.float32)
        f16 = cal & np.isfinite(E16).all(1)
        y = OBS[:, t]; gh = s4["ghap"][i4].astype(np.float32)
        pp = np.unpackbits(X2["p_masks"][i4])[:len(SRC)].astype(bool)
        fin16 = np.isfinite(E16).all(1)
        E4s = s4["ens"][i4].astype(np.float32)
        ysc = [np.where(cal, y, 0.0).astype(np.float64), np.where(cal, gh, 0.0).astype(np.float64)]
        mc = mix_sums(E16, E4s, cal & fin16, cal & ~fin16, ysc, {"cal": cal})["cal"]
        ysp = [np.where(pp, y, 0.0).astype(np.float64)]
        mp = mix_sums(E16, E4s, pp & fin16, pp & ~fin16, ysp, {"pp": pp})["pp"]
        arr = np.stack([mc[q] for q in Q16C] + [mp[{"p_n": "n", "p_s2": "s2", "p_s2M": "s2M", "p_sse": "sse_0", "p_crps": "crps_0", "p_pair": "pair"}[q]] for q in Q16P], 1)
        ACC16.setdefault(h, []).append((d, i4, arr, int((cal & ~fin16).sum()), int((cal & ~fin16 & (z["stpos"] >= 0)).sum())))
        for v, tv in enumerate((y, gh)):
            for g in range(7):
                m = f16 & GM[g]
                Xg = E16[m].astype(np.float64); yy = tv[m].astype(np.float64); dd = Xg - yy[:, None]
                S_ = SUF[(h, v, g)]
                S_["D"] += dd.T @ dd; S_["A"] += np.abs(dd).sum(0); S_["n"] += float(m.sum())
                if g == 6:
                    PERF.setdefault((h, v), []).append((d, dd.T @ dd, float(m.sum())))
                for i in range(15):
                    S_["P"][i, i + 1:] += np.abs(Xg[:, i:i + 1] - Xg[:, i + 1:]).sum(0)
                for j, T_ in enumerate(THR):
                    e = (Xg > T_).astype(np.float64); o = yy > T_
                    S_["E"][j] += e.T @ e; S_["F"][j] += e[o].sum(0); S_["O"][j] += o.sum()
K2S = {}; CVS = {}

for (h, v, g), S_ in SUF.items():
    if S_["n"] < 1000:
        continue
    Pm = S_["P"] + S_["P"].T
    rep = full_report(S_["D"], S_["n"], S_["A"], Pm, S_["E"], S_["F"], S_["O"])
    rep["n_station_days"] = S_["n"]
    K2S.setdefault(str(h), {}).setdefault(["measurements", "estimates"][v], {})[GROUPS[g]] = rep
    CVS[(h, v, g)] = {k: (np.sqrt(rep[k]["v_quart"]) / abs(rep[k]["quartet_mean"]) if isinstance(rep[k], dict)
                          else [np.sqrt(x["v_quart"]) / abs(x["quartet_mean"]) if x["quartet_mean"] else np.nan for x in rep[k]])
                      for k in ("rmse_inf", "rmse_8", "ssr_inf", "ssr_fortin_8", "crps_fair", "crps_8", "bs_fair", "bs_8", "rmse_m", "ssr_fortin_m", "bs_raw")}
OUT["stations_16_days"] = sorted([list(x) for x in have16])
for (h, v), lst in PERF.items():
    if len(lst) < MIN16:
        continue
    lst = sorted(lst, key=lambda x: x[0])
    Dd = np.stack([x[1] for x in lst]); nn = np.array([x[2] for x in lst])
    Ds = Dd.sum((1, 2)); Dg = np.einsum("fii->f", Dd); sse_d = Ds / 256; s2_d = (Dg - Ds / 16) / 15
    Wb = block_weights(len(nn), B=7, n=10000, seed=0)
    bn, bse, bs2 = Wb @ nn, Wb @ sse_d, Wb @ s2_d
    rep = K2S[str(h)][["measurements", "estimates"][v]]["All official networks"]
    for kname, bsv in (("ssr_inf", np.sqrt(bs2 / bn) / np.sqrt((bse - bs2 / 16) / bn)), ("ssr_fortin_8", np.sqrt(9 / 8 * bs2 / bn) / np.sqrt((bse - bs2 / 16 + bs2 / 8) / bn)),
                       ("rmse_inf", np.sqrt((bse - bs2 / 16) / bn))):
        lo, hi = cil(bsv); ptv = rep[kname]["all16"]
        cc_, fac = inflate(float(ptv), float(lo), float(hi), float(np.nanvar(bsv, ddof=1)), rep[kname]["jk16_var"])
        rep[f"boot16_{kname}"] = {"v": float(ptv), "ci_boot": [float(lo), float(hi)], "ci": cc_, "factor": fac,
                                  "note": "station-days of the reliability sample with sixteen members (the 57 stations without them left out)"}


# ---- sixteen-member values on the reliability samples (cal and paper), leads 1 and 3 (fixes 2, 3)
ST16 = {}
DROP16 = {}
for h, lst in ACC16.items():
    lst = sorted(lst, key=lambda x: x[0])
    rows16 = np.array([x[1] for x in lst]); A16 = np.stack([x[2] for x in lst])          # (forecasts, 7, quantities, 17)
    DROP16[str(h)] = dict(forecasts=len(lst), fallback_station_days=int(sum(x[3] for x in lst)), unexpected=int(sum(x[4] for x in lst)))
    iq = {q: j for j, q in enumerate(Q16C + Q16P)}
    ST16[str(h)] = {}
    for g, gname in enumerate(GROUPS):
        Ag = A16[:, g]                                                # (forecasts, quantities, 17)
        Rf = {k: DAY[k][rows16, g] for k in ("c_crps_0", "c_crps_1", "c_bsf_0", "c_bsf_1", "p_clim_crps", "p_cams_se", "p_aur_se")}

        def forms16(S, Rs):
            """S (..., quantities, 17), Rs reference sums (...[, 3]): dict key -> (..., 17) values (column 0 full, 1-16 replicates)"""
            q = lambda name: S[..., iq[name], :]
            o = {}
            n = q("n"); s2 = q("s2"); s2M = q("s2M"); pair = q("pair")
            for v in (0, 1):
                sse = q(f"sse_{v}"); crps = q(f"crps_{v}")
                for f, c8 in (("inf", 0.0), ("8", 1 / 8)):
                    mse = (sse - s2M + c8 * s2) / n
                    o[f"rmse|{f}|{v}"] = np.sqrt(mse)
                    o[f"ss|{f}|{v}"] = np.sqrt((9 / 8 if f == "8" else 1.0) * s2 / n) / np.sqrt(mse)
                    o[f"crps|{f}|{v}"] = (crps + c8 * pair) / n
                    o[f"crps_vs_clim|{f}|{v}"] = 100 * ((crps + c8 * pair) / Rs[f"c_crps_{v}"][..., None] - 1)
                    bsf = np.stack([q(f"bsf_{v}_{j}") + c8 * q(f"kmk_{j}") for j in range(3)], -2)     # (..., 3, 17)
                    o[f"bss_clim|{f}|{v}"] = 1 - bsf / Rs[f"c_bsf_{v}"][..., :, None]
            pn = q("p_n"); ps2 = q("p_s2"); ps2M = q("p_s2M"); psse = q("p_sse"); pcr = q("p_crps"); ppair = q("p_pair")
            for f, c8 in (("inf", 0.0), ("8", 1 / 8)):
                num = psse - ps2M + c8 * ps2
                o[f"p_rmse|{f}"] = np.sqrt(num / pn)
                o[f"p_ss|{f}"] = np.sqrt((9 / 8 if f == "8" else 1.0) * ps2 / pn) / np.sqrt(num / pn)
                o[f"p_rmse_vs_cams|{f}"] = 100 * (np.sqrt(num / Rs["p_cams_se"][..., None]) - 1)
                o[f"p_rmse_vs_aur|{f}"] = 100 * (np.sqrt(num / Rs["p_aur_se"][..., None]) - 1)
                o[f"p_crps_vs_climens|{f}"] = 100 * ((pcr + c8 * ppair) / Rs["p_clim_crps"][..., None] - 1)
            return o
        with np.errstate(all="ignore"):
            pt = forms16(Ag.sum(0), {k: v.sum(0) for k, v in Rf.items()})
            okc, okp = DAY["n"][rows16, g] > 0, DAY["p_n"][rows16, g] > 0
            boot = None
            if len(lst) >= MIN16 and okc.sum() >= 14:
                Wc = block_weights(int(okc.sum()), B=7, n=10000, seed=0); Wp = block_weights(int(okp.sum()), B=7, n=10000, seed=0)
                bc = forms16(np.tensordot(Wc, Ag[okc], axes=(1, 0)), {k: np.tensordot(Wc, v[okc], axes=(1, 0)) for k, v in Rf.items()})
                bp = forms16(np.tensordot(Wp, Ag[okp], axes=(1, 0)), {k: np.tensordot(Wp, v[okp], axes=(1, 0)) for k, v in Rf.items()})
                boot = {k: (bp[k] if k.startswith("p_") else bc[k])[..., 0] for k in pt}
        rec = {}
        for k_, val in pt.items():
            full = np.asarray(val[..., 0], float); reps = np.asarray(val[..., 1:], float)
            vj = (15 / 16) * ((reps - reps.mean(-1, keepdims=True)) ** 2).sum(-1)
            e = {"v": full.tolist(), "v_jack16": vj.tolist(), "jk16_sd": np.sqrt(vj).tolist(), "days": len(lst)}
            if boot is not None:
                e.update(form(full, boot[k_], vj))
                e["v"] = full.tolist()
            rec[k_] = e
        ST16[str(h)][gname] = rec
OUT["stations_reliability_16"] = {"by_lead": ST16, "dropped": DROP16,
                                  "note": "sixteen members where finite, the four stored members of the 57 stations without a populated pixel (A14 section 3); "
                                          "cal and paper samples as the as-run values; reference sums of the climatological ensemble and of CAMS/Aurora on the same station-days; "
                                          "V_draw = jackknife over the 16 fields (fallback: 3-member replicate for i < 4, fixed for i >= 4); bootstrap only when all start days are present"}


def cv_station(h, v, g, key):
    g = g if (1, v, g) in CVS else 6
    if (1, v, g) not in CVS:
        return np.nan
    c = lambda L: np.asarray(CVS[(L, v, g)][key], float)
    return c(h) if h in (1, 3) else (0.5 * (c(1) + c(3)) if h == 2 else c(3))


AR_META = {"ss_0": ("ssr_fortin_m", "plain", 0), "ss_1": ("ssr_fortin_m", "plain", 1), "rmse_0": ("rmse_m", "plain", 0), "rmse_1": ("rmse_m", "plain", 1),
           "crps_0": ("crps_fair", "plain", 0), "crps_1": ("crps_fair", "plain", 1), "crps_vs_clim_0": ("crps_fair", "ratio100", 0),
           "crps_vs_clim_1": ("crps_fair", "ratio100", 1), "bss_clim_0": ("bs_raw", "skill", 0), "bss_clim_1": ("bs_raw", "skill", 1),
           "bss_clim_fair_0": ("bs_fair", "skill", 0), "bss_clim_fair_1": ("bs_fair", "skill", 1), "p_rmse": ("rmse_m", "plain", 0),
           "p_rmse_vs_cams": ("rmse_m", "ratio100", 0), "p_rmse_vs_aur": ("rmse_m", "ratio100", 0), "p_ss": ("ssr_fortin_m", "plain", 0),
           "p_crps": ("crps_fair", "plain", 0), "p_crps_vs_climens": ("crps_fair", "ratio100", 0)}


def asrun_reps(S, E):
    """delete-one-member replicates (3 members) of the as-run (four-member) statistics, last axis = the deleted member"""
    o = {}
    n = S["n"][..., None]; lv = S["loo_var"]
    with np.errstate(all="ignore"):
        for v in (0, 1):
            s = f"_{v}"
            o["ss" + s] = np.sqrt(4 / 3 * lv / S["loo_sse" + s]); o["rmse" + s] = np.sqrt(S["loo_sse" + s] / n)
            o["crps" + s] = S["loo_crps" + s] / n; o["crps_vs_clim" + s] = 100 * (S["loo_crps" + s] / S["c_crps" + s][..., None] - 1)
            o["bss_clim" + s] = 1 - E["loo_bs" + s] / S["c_bs" + s][..., None]
            o["bss_clim_fair" + s] = 1 - E["loo_bsf" + s] / S["c_bsf" + s][..., None]
        pn = S["p_n"][..., None]
        o["p_rmse"] = np.sqrt(S["p_loo_sse"] / pn); o["p_ss"] = np.sqrt(4 / 3 * S["p_loo_var"] / S["p_loo_sse"])
        o["p_rmse_vs_cams"] = 100 * (np.sqrt(S["p_loo_sse"] / S["p_cams_se"][..., None]) - 1)
        o["p_rmse_vs_aur"] = 100 * (np.sqrt(S["p_loo_sse"] / S["p_aur_se"][..., None]) - 1)
        o["p_crps"] = S["p_loo_crps"] / pn; o["p_crps_vs_climens"] = 100 * (S["p_loo_crps"] / S["p_clim_crps"][..., None] - 1)
    return o


# ---- scores by lead and group
ST = {}
ident_fail = []
for h in range(1, 8):
    rows = np.where(K[:, 1] == h)[0]; rows = rows[np.argsort(K[rows, 0])]
    ST[h] = {}
    for g, gname in enumerate(GROUPS):
        X, sl = ns["pack"](lambda k: DAY[k][rows, g])
        pt = ns["derive"](ns["unpack"](X.sum(0), sl))
        Sg = {k: DAY[k][rows, g] for k in NAMES}; Eg = {k: EXTRA[k][rows, g] for k in EXTRA}
        Spt = {k: v.sum(0) for k, v in Sg.items()}; Ept = {k: v.sum(0) for k, v in Eg.items()}
        u_pt = derive_u(Spt, Ept); u_jk = derive_u(Spt, Ept, jk=True)
        rec = {}
        for sample, cnt in (("cal", "n"), ("paper", "p_n")):
            ok = DAY[cnt][rows, g] > 0
            if ok.sum() < 14:
                continue
            W = block_weights(int(ok.sum()), B=7, n=10000, seed=0)
            bo = ns["derive"](ns["unpack"](W @ X[ok], sl))
            ub = derive_u({k: np.tensordot(W, v[ok], axes=(1, 0)) for k, v in Sg.items()},
                          {k: np.tensordot(W, v[ok], axes=(1, 0)) for k, v in Eg.items()})
            keys_as = (["ss_0", "ss_1", "rmse_0", "rmse_1", "crps_0", "crps_1", "crps_vs_clim_0", "crps_vs_clim_1", "bss_clim_0",
                        "bss_clim_1", "bss_clim_fair_0", "bss_clim_fair_1", "spread"] if sample == "cal" else
                       ["p_rmse", "p_cams_rmse", "p_aur_rmse", "p_rmse_vs_cams", "p_rmse_vs_aur", "p_ss", "p_crps", "p_crps_vs_climens",
                        "p_mae", "p_cams_mae", "p_aur_mae", "p_clim_crps"])
            AR = asrun_reps(Spt, Ept)
            for k_ in keys_as:
                lo, hi = cil(bo[k_])
                r = REF["leads"][str(h)][gname][k_]
                same = close(pt[k_], r["v"]) and ("ci_day" not in r or (close(lo, r["ci_day"][0]) and close(hi, r["ci_day"][1])))
                if not same:
                    ident_fail.append((h, gname, k_))
                rec[k_] = {"as_run": {"v": np.asarray(pt[k_]).tolist(), "ci_boot": [np.asarray(lo).tolist(), np.asarray(hi).tolist()],
                                      "identical_to_paper_json": same}}
                if k_ in AR:                                         # fix 9: the draw of the four stored fields in the as-run interval
                    cvk, kind, v = AR_META[k_]
                    vj = jk_var(AR[k_]); cv = cv_station(h, v, g, cvk)
                    val = np.asarray(pt[k_], float)
                    vq = (np.asarray(cv) * np.abs(val)) ** 2 if kind == "plain" else ((100 * np.asarray(cv) * (1 + val / 100)) ** 2 if kind == "ratio100" else (np.asarray(cv) * (1 - val)) ** 2)
                    vd = np.fmax(np.nan_to_num(np.asarray(vq, float)), np.nan_to_num(np.asarray(vj, float)))
                    fe = form(val, bo[k_], vd)
                    rec[k_]["as_run"].update(ci=fe.get("ci"), factor=fe.get("factor"), v_jack4=np.asarray(vj).tolist(), v_quart=np.asarray(vq).tolist())
            for k_ in u_pt:
                if k_.startswith("p_") != (sample == "paper"):
                    continue
                cvk, kind, v = META[k_]
                jkv = jk_var(u_jk[k_])
                cv = cv_station(h, v, g, cvk)
                vq = v_quart_of(k_, u_pt[k_], cv)
                vdraw = np.fmax(np.nan_to_num(np.asarray(vq, float), nan=0.0), np.nan_to_num(np.asarray(jkv, float), nan=0.0))
                e = form(np.asarray(u_pt[k_], float), ub[k_], vdraw)
                e.update(v_jack4=np.asarray(jkv).tolist(), v_quart=np.asarray(vq).tolist(), cv_quart=np.asarray(cv).tolist())
                rec[k_] = e
        ST[h][gname] = rec
OUT["stations_reliability"] = {"by_lead": ST, "identity_failures": ident_fail, "k2_cal_station_pixels": K2S,
                               "note": "cal sample: SSR, BSS, CRPS vs climatological ensemble (v 0 measurements, 1 estimates); paper sample: RMSE 12.96/15.42/15.90 and CRPS -32 %"}
print("stations (reliability): identity failures", len(ident_fail), ident_fail[:10], flush=True)

# =========================================================================== (B) populated pixels
S34 = REL / "scripts" / "s34_estimates.py"
ZP = np.load(RELD / "sums_pop.npz")
nsp = dict(np=np, M=4, F=5 / 4, pk=np.arange(5) / 4)
functions_from(S34, ["div", "pop_scores"], nsp)
REFE = json.loads((RELR / "estimates_scores.json").read_text())
KP = ZP["keys"]; REGP = [str(r) for r in ZP["regions"]]
NAMESP = [k for k in ZP.files if k not in ("keys", "regions", "thr", "sd_edges", "leads", "pixref", "pixref_loo", "n_pixels", "residents")
          and not k.startswith("sd_")]
P4 = np.load(DATA / "pop4_terms.npz") if (DATA / "pop4_terms.npz").exists() else None
POP = {}
K2P = {}
ZZ = [np.load(DATA / f"pop16_{tag}.npz") for tag in TAGS if (DATA / f"pop16_{tag}.npz").exists()]
if ZZ:                                                             # the sixteen-member forecasts of every tag together
    z = {k: np.concatenate([zz[k] for zz in ZZ]) for k in ("keys", "n", "n4", "D", "A", "P", "E", "F", "O", "sse4", "var4")}
    kk_ = [tuple(x) for x in z["keys"].tolist()]
    assert len(set(kk_)) == len(kk_), "a forecast appears in two sixteen-member files"
    if True:
        for li, h in enumerate((1, 3)):
            sel = z["keys"][:, 1] == h
            for wi, wn in enumerate(("pixels", "residents")):
                D = z["D"][sel, wi].sum(0); n = z["n"][sel, wi].sum(); A = z["A"][sel, wi].sum(0); Pm = z["P"][sel, wi].sum(0)
                E = z["E"][sel, wi].sum(0); Fo = z["F"][sel, wi].sum(0); O = z["O"][sel, wi].sum(0)
                rep = full_report(D, n, A, Pm, E, Fo, O)
                rep["n"] = float(n); rep["days"] = int(sel.sum())
                if int(sel.sum()) >= MIN16:                       # all start days: bootstrap over start days (paper blocks)
                    kk = z["keys"][sel]; o_ = np.argsort(kk[:, 0])
                    Dd = z["D"][sel, wi][o_]; nn = z["n"][sel, wi][o_]
                    Ds = Dd.sum((1, 2)); Dg = np.einsum("fii->f", Dd)
                    sse_d = Ds / 256; s2_d = (Dg - Ds / 16) / 15
                    Wb = block_weights(len(nn), B=7, n=10000, seed=0)
                    bn, bse, bs2 = Wb @ nn, Wb @ sse_d, Wb @ s2_d
                    for kname, ptv, bsv, jkk in (("ssr_inf", rep["ssr_inf"]["all16"], np.sqrt(bs2 / bn) / np.sqrt((bse - bs2 / 16) / bn), "ssr_inf"),
                                                 ("ssr_fortin_8", rep["ssr_fortin_8"]["all16"], np.sqrt(9 / 8 * bs2 / bn) / np.sqrt((bse - bs2 / 16 + bs2 / 8) / bn), "ssr_fortin_8"),
                                                 ("rmse_inf", rep["rmse_inf"]["all16"], np.sqrt((bse - bs2 / 16) / bn), "rmse_inf")):
                        lo, hi = cil(bsv); vj = rep[jkk]["jk16_var"]
                        cc_, fac = inflate(float(ptv), float(lo), float(hi), float(np.nanvar(bsv, ddof=1)), vj)
                        rep[f"boot16_{kname}"] = {"v": float(ptv), "ci_boot": [float(lo), float(hi)], "ci": cc_, "factor": fac}
                rep["stored4_same_pixels"] = {"ss_fortin4": float(np.sqrt(1.25 * z["var4"][sel, wi].sum() / z["sse4"][sel, wi].sum())),
                                              "rmse_inf": float(np.sqrt((z["sse4"][sel, wi].sum() - z["var4"][sel, wi].sum() / 4) / n)),
                                              "ssr_inf": float(np.sqrt(z["var4"][sel, wi].sum() / n) / np.sqrt((z["sse4"][sel, wi].sum() - z["var4"][sel, wi].sum() / 4) / n))}
                K2P.setdefault(str(h), {})[wn] = rep
for li, h in enumerate((1, 3)):
    rows = np.where(KP[:, 1] == h)[0]; rows = rows[np.argsort(KP[rows, 0])]
    W = block_weights(len(rows), B=7, n=10000, seed=0)
    POP[h] = {}
    r = -1                                                             # 'All populated pixels'
    S = {k: ZP[k][rows, r] for k in NAMESP}
    for pre, wn in (("", "pixels"), ("w", "residents")):
        pt = nsp["pop_scores"]({k: v.sum(0) for k, v in S.items()}, pre)
        bo = nsp["pop_scores"]({k: np.tensordot(W, v, axes=(1, 0)) for k, v in S.items()}, pre)
        refv = REFE["pop"]["leads"][str(h)]["All populated pixels"][wn]["ss"]
        lo, hi = cil(bo["ss"])
        same = close(pt["ss"], refv["v"]) and close(lo, refv["ci_day"][0]) and close(hi, refv["ci_day"][1])
        n = S[pre + "n"]; var = S[pre + "var"]; sse = S[pre + "sse"]
        Wn, Wv, Ws = W @ n, W @ var, W @ sse
        inf_pt = np.sqrt(var.sum() / n.sum()) / np.sqrt((sse.sum() - var.sum() / 4) / n.sum())
        inf_bs = np.sqrt(Wv / Wn) / np.sqrt((Ws - Wv / 4) / Wn)
        m8_pt = np.sqrt(9 / 8 * var.sum() / n.sum()) / np.sqrt((sse.sum() - var.sum() / 8) / n.sum())
        m8_bs = np.sqrt(9 / 8 * Wv / Wn) / np.sqrt((Ws - Wv / 8) / Wn)
        rinf_pt = np.sqrt((sse.sum() - var.sum() / 4) / n.sum()); rinf_bs = np.sqrt((Ws - Wv / 4) / Wn)
        r8_pt = np.sqrt((sse.sum() - var.sum() / 8) / n.sum()); r8_bs = np.sqrt((Ws - Wv / 8) / Wn)
        vj = {}
        if P4 is not None:
            pk = [tuple(x) for x in P4["keys"].tolist()]; pos = [pk.index(tuple(x)) for x in KP[rows].tolist()]
            ln, ls, lv = P4[pre + "n"][pos].sum(), P4[pre + "loo_sse"][pos].sum(0), P4[pre + "loo_var"][pos].sum(0)
            jr = np.sqrt((ls - lv / 3) / ln); jr8 = np.sqrt((ls - lv / 3 + lv / 8) / ln)
            vj = {"ss_inf": jk_var(np.sqrt(lv / ln) / jr), "ss_8": jk_var(np.sqrt(9 / 8 * lv / ln) / jr8), "rmse_inf": jk_var(jr), "rmse_8": jk_var(jr8)}
        cvq = {}
        if str(h) in K2P:
            rep = K2P[str(h)][wn]
            cvq = {"ss_inf": np.sqrt(rep["ssr_inf"]["v_quart"]) / rep["ssr_inf"]["quartet_mean"],
                   "ss_8": np.sqrt(rep["ssr_fortin_8"]["v_quart"]) / rep["ssr_fortin_8"]["quartet_mean"],
                   "rmse_inf": np.sqrt(rep["rmse_inf"]["v_quart"]) / rep["rmse_inf"]["quartet_mean"],
                   "rmse_8": np.sqrt(rep["rmse_8"]["v_quart"]) / rep["rmse_8"]["quartet_mean"]}
        rec = {"ss_as_run": {"v": float(pt["ss"]), "ci_boot": [float(lo), float(hi)], "identical_to_paper_json": same}}
        for k_, (p_, b_) in {"ss_inf": (inf_pt, inf_bs), "ss_8": (m8_pt, m8_bs), "rmse_inf": (rinf_pt, rinf_bs), "rmse_8": (r8_pt, r8_bs)}.items():
            vq = (cvq.get(k_, np.nan) * p_) ** 2
            vd = max(np.nan_to_num(vq), np.nan_to_num(vj.get(k_, np.nan)))
            e = form(p_, b_, vd); e.update(v_jack4=float(vj.get(k_, np.nan)), v_quart=float(vq), cv_quart=float(cvq.get(k_, np.nan)))
            rec[k_] = e
        rec["rmse_as_run"] = {"v": float(np.sqrt(sse.sum() / n.sum()))}
        POP[h][wn] = rec
OUT["populated_pixels"] = {"by_lead": POP, "k2_populated_pixels": K2P}

# carried CV for land: the populated pixels against the estimates (pixels, unweighted)
# 'ar': the stored models before the CRPS training keep the draw of the stored fields (A16 4.4): their carried CV is that of
# the stored record (results/rel_classU_179.json), the CV of the new runs is not a fixed-field draw
K2P_FIXED = json.loads((RES0 / "rel_classU_179.json").read_text())["populated_pixels"]["k2_populated_pixels"] if AR else None


def cv_land(h, key, k2p=None):
    k2p = K2P if k2p is None else k2p
    if not k2p:
        return np.nan
    c = lambda L: np.sqrt(k2p[str(L)]["pixels"][key]["v_quart"]) / k2p[str(L)]["pixels"][key]["quartet_mean"]
    return c(h) if h in (1, 3) else (0.5 * (c(1) + c(3)) if h == 2 else c(3))


# =========================================================================== (C) land, recs of the evaluations (s34 section 4)
LAND = {}
BEFORE = RUN.parent / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12"
for tag, ev in (("2022 final", EVF), ("2021 final (validation year)", RUN / f"eval2021_sel_crps_e01{ISFX}"),
                ("2022 before CRPS training", BEFORE / "eval_final_e02"), ("2021 before CRPS training", BEFORE / "eval2021_sel_e02"),
                ("2021 before CRPS training, 4 Euler steps", BEFORE / "eval2021_sel_e02_k4")):
    parts = sorted(glob.glob(str(ev / "part_*.npz")))
    if not parts:
        continue
    fixed = ev.parent == BEFORE                                        # a stored fixed-field model (A16 4.4)
    k2f = K2P_FIXED if fixed else None
    z0 = np.load(parts[0]); C = [str(c) for c in z0["rec_cols"]]; Mx = int(z0["info"][5]); regs = [str(r) for r in z0["regions"]]
    R = np.concatenate([np.load(p)["recs"] for p in parts])
    c = lambda q: C.index(q)
    fac = (Mx + 1) / Mx
    days = np.unique(R[:, c("start_doy")]).astype(int)
    LAND[tag] = {"members": Mx, "start_days": int(len(days)), "leads": {}}
    for h in range(1, 8):
        rh = R[R[:, c("lead")] == h]
        T = np.zeros((len(days), len(regs) + 1, 5))
        di = np.searchsorted(days, rh[:, c("start_doy")].astype(int)); ri = rh[:, c("region")].astype(int)
        vals = np.stack([rh[:, c(q)] for q in ("n", "sse", "var", "crps", "sum_err")], 1)
        np.add.at(T, (di, ri), vals); T[:, -1] = T[:, :-1].sum(1)
        ok = T[:, -1, 0] > 0
        W = block_weights(int(ok.sum()), B=7, n=10000, seed=0) if ok.sum() >= 14 else None
        rec = {}
        for r, rname in enumerate(regs + ["Nine regions pooled"]):
            s = T[ok, r].sum(0)
            refe = REFE["land"].get(tag, {}).get("leads", {}).get(str(h), {}).get(rname)
            e = {"n": float(s[0])}
            ss = float(np.sqrt(fac * s[2] / s[1]))
            e["ss_as_run"] = {"v": ss}
            if W is not None:
                b = W @ T[ok, r]
                lo, hi = cil(np.sqrt(fac * b[:, 2] / b[:, 1]))
                e["ss_as_run"]["ci_boot"] = [float(lo), float(hi)]
                e["ss_as_run"]["identical_to_paper_json"] = bool(refe is not None and close(ss, refe["ss"]) and close([lo, hi], refe["ss_ci"]))
                pi = np.sqrt(s[2] / s[0]) / np.sqrt((s[1] - s[2] / Mx) / s[0]); bi = np.sqrt(b[:, 2] / b[:, 0]) / np.sqrt((b[:, 1] - b[:, 2] / Mx) / b[:, 0])
                p8 = np.sqrt(9 / 8 * s[2] / s[0]) / np.sqrt((s[1] - s[2] / Mx + s[2] / 8) / s[0])
                b8 = np.sqrt(9 / 8 * b[:, 2] / b[:, 0]) / np.sqrt((b[:, 1] - b[:, 2] / Mx + b[:, 2] / 8) / b[:, 0])
                cvi, cv8 = cv_land(h, "ssr_inf", k2f), cv_land(h, "ssr_fortin_8", k2f)
                e["ss_inf"] = form(pi, bi, (cvi * pi) ** 2 if np.isfinite(cvi) else None, fixed=fixed)
                e["ss_8"] = form(p8, b8, (cv8 * p8) ** 2 if np.isfinite(cv8) else None, fixed=fixed)
                e["rmse_as_run"] = float(np.sqrt(s[1] / s[0])); e["rmse_inf"] = float(np.sqrt((s[1] - s[2] / Mx) / s[0]))
                e["rmse_8"] = float(np.sqrt((s[1] - s[2] / Mx + s[2] / 8) / s[0]))
            rec[rname] = e
        LAND[tag]["leads"][h] = rec
OUT["land_recs"] = LAND

# =========================================================================== (D) emissions and transport (recs of the four models)
E_DIR = REL.parent / "emissions"
zk = np.load(E_DIR / "out" / "sums_km1.npz")
Tk = zk["T"]; names = [str(x) for x in zk["names"]]; models = [str(x) for x in zk["models"]]; stats_ = [str(x) for x in zk["stats"]]
nse = dict(np=np, NB=10000, BLOCK=7, SEED=0)                       # the module constants of recompute.py (its defaults)
functions_from(E_DIR / "recompute.py", ["block_index", "block_counts", "rmse", "crps", "boot", "gain"], nse)
nse["SCORE"] = {"rmse": nse["rmse"], "crps": nse["crps"]}
gains_ref = list(csv.DictReader(open(E_DIR / "out" / "gains.csv")))
gref = {(r["statistic"], r["region"], r["lead_days"], r["comparison"], r["metric"]): r for r in gains_ref}
nd = Tk.shape[3]
Wk = nse["block_counts"](nd)
EM = {}
iC, iA, iE, iR = (models.index(x) for x in ("C", "A", "e02", "ctrl"))
EP = json.loads((RES0 / "emis_pair.json").read_text())["comparisons"] if (RES0 / "emis_pair.json").exists() else None   # other systems: stored record


def pair_ratio(comp, lk, form):
    """V_jack/V_boot of the paired four-field jackknife of the same gain at the station pixels (emis_pair.py), carried to land"""
    if EP is None:
        return None
    return EP[comp]["by_lead"][str(lk)]["as_run" if form == "as_run" else form]["ratio"]
assert stats_[:4] == ["n", "sse", "crps", "var"], stats_
for comp, ref_m, mod_m in (("emissions", iR, iE), ("transport", iC, iA)):
    for r, rname in enumerate(names):
        for h in range(1, 8):
            Rr, Aa = Tk[ref_m, r, h - 1], Tk[mod_m, r, h - 1]
            e = {}
            for metric in ("rmse", "crps"):
                pt, lo, hi, bs = nse["gain"](Rr, Aa, metric, Wk)
                rr = gref.get(("km1", rname, str(h), comp, metric))
                e[metric + "_as_run"] = {"v": pt, "ci_boot": [lo, hi],
                                         "identical_to_paper_csv": bool(rr is not None and abs(pt - float(rr["gain_pct"])) < 6e-4 and abs(lo - float(rr["ci95_lo"])) < 6e-4 and abs(hi - float(rr["ci95_hi"])) < 6e-4)}
                if metric == "rmse":                                   # fix 4: the paired draw in the as-run interval too
                    rat = pair_ratio(comp, h, "as_run")
                    if rat is not None:
                        fe = form(pt, bs, rat * float(np.nanvar(bs, ddof=1)), fixed=True)
                        e["rmse_as_run"].update(ci=fe["ci"], factor=fe["factor"], v_draw_ratio=rat)
            for f, c8 in (("inf", 0.0), ("M8", 1 / 8)):
                def r_(X, W=None):
                    S = X if W is None else W @ X
                    S = S if W is not None else S.sum(0)
                    return np.sqrt((S[..., 1] - S[..., 3] / 4 + c8 * S[..., 3]) / S[..., 0])
                pt = 100 * (1 - r_(Aa) / r_(Rr)); bs = 100 * (1 - r_(Aa, Wk) / r_(Rr, Wk))
                rat = pair_ratio(comp, h, f)
                e[f"rmse_{f}"] = form(pt, bs, rat * float(np.nanvar(bs, ddof=1)) if rat is not None else None, fixed=True)
                e[f"rmse_{f}"]["v_draw_note"] = ("paired four-field jackknife (field i removed from both models) of the same gain at the 8,110 station pixels "
                                                 "against GHAP, carried to land as V_draw/V_boot (approximation: the exact land jackknife needs per-member land sums)")
                e[f"rmse_{f}"]["v_draw_ratio"] = rat
            # CRPS: fair = infinite; eight members not available from recs (the pair term is not stored)
            e["crps_inf"] = "= as run (fair CRPS)"
            fac = 5 / 4
            for nm, mi in (("model", mod_m), ("reference", ref_m)):
                s = Tk[mi, r, h - 1].sum(0)
                e[f"ss_{nm}"] = {"as_run": float(np.sqrt(fac * s[3] / s[1])), "inf": float(np.sqrt(s[3] / s[0]) / np.sqrt((s[1] - s[3] / 4) / s[0])),
                                 "M8": float(np.sqrt(9 / 8 * s[3] / s[0]) / np.sqrt((s[1] - s[3] / 4 + s[3] / 8) / s[0]))}
            EM.setdefault(comp, {}).setdefault(rname, {})[h] = e
OUT["emissions_transport_km1"] = EM
if EP is not None:
    OUT["emissions_pair_draw"] = {c: {k: v for k, v in EP[c].items() if k != "by_lead"} for c in EP}
C25F = DATA0 / "c25_terms.npz"                                         # other systems (c25_pass.py): stored record
if C25F.exists():
    zc = np.load(C25F); Tc = zc["T"]; cnames = [str(x) for x in zc["names"]]; cmodels = [str(x) for x in zc["models"]]
    ndc = Tc.shape[3]; Wc25 = nse["block_counts"](ndc)
    POOLS = {"1-7": range(7), "1-3": range(0, 3), "5-7": range(4, 7)}
    JM0 = json.loads((DATA.parent.parent / "classU_review" / "judge" / "j_m0.json").read_text())["ratios"]          # read only (reference j_m0.py)
    SDRAW = {L: 1 / np.array(JM0[f"lead{L}_all"]["ratio_by_member"]) for L in (1, 3)}                            # the four members' ratios
    C25 = {}
    for comp, (rf, md) in (("emissions", ("ctrl", "e02")), ("transport", ("C", "A"))):
        ir, im = cmodels.index(rf), cmodels.index(md)
        for b, nm in enumerate(cnames):
            for lk in [str(h) for h in range(1, 8)] + list(POOLS):
                hs = [int(lk) - 1] if lk.isdigit() else list(POOLS[lk])
                Xr, Xm = Tc[ir, b, hs].sum(0), Tc[im, b, hs].sum(0)          # (days, 3): n, sse, q
                if Xr[:, 0].sum() == 0:
                    continue
                e = {}
                for f, cq in (("as_run", 0.0), ("M8", 0.5), ("inf", 1.0)):
                    def rr_(X, W=None):
                        S = X.sum(0) if W is None else W @ X
                        return np.sqrt((S[..., 1] - cq * S[..., 2]) / S[..., 0])
                    pt = float(100 * (1 - rr_(Xm) / rr_(Xr))); bs = 100 * (1 - rr_(Xm, Wc25) / rr_(Xr, Wc25))
                    rat = pair_ratio(comp, lk, f)
                    v_m0 = 0.0                                       # fix 11: the single-member draw of the member-0 term, both models scaled alike
                    if cq > 0:
                        Sm, Sr = Xm.sum(0), Xr.sum(0)
                        gs = [100 * (1 - np.sqrt((Sm[1] - cq * s * Sm[2]) / Sm[0]) / np.sqrt((Sr[1] - cq * s * Sr[2]) / Sr[0])) for s in SDRAW[1 if lk == "1" else 3]]
                        v_m0 = float(np.var(gs))
                    fe = form(pt, bs, rat * float(np.nanvar(bs, ddof=1)) + v_m0 if rat is not None else None, fixed=True)
                    fe["v_draw_ratio"] = rat; fe["v_m0"] = v_m0
                    if f == "as_run":
                        g_ = gref.get(("c25", nm, lk, comp, "rmse"))
                        fe["identical_to_paper_csv"] = bool(g_ is not None and abs(pt - float(g_["gain_pct"])) < 6e-4 and abs(fe["ci_boot"][0] - float(g_["ci95_lo"])) < 6e-4
                                                            and abs(fe["ci_boot"][1] - float(g_["ci95_hi"])) < 6e-4)
                    e[f] = fe
                C25.setdefault(comp, {}).setdefault(nm, {})[lk] = e
    OUT["emissions_c25"] = {"by_box": C25, "note": "0.25 deg cell means (recompute.py c25), member 0 only for the M=8 and infinite forms; "
                                                   "V_draw: the paired station ratio of the same lead or pool carried (fix 4 caveat) + v_m0, the variance of the gain over "
                                                   "the four members' ratios of j_m0.json (lead 1 for lead 1, lead 3 for the other leads and the pools; fix 11)"}
if ZZ:
    OUT["populated_pixels"]["dropped_16"] = {"pixel_days_4_not_16": float((z["n4"][:, 0] - z["n"][:, 0]).sum()),
                                             "forecasts_with_drops": int(((z["n4"][:, 0] - z["n"][:, 0]) > 0).sum()), "forecasts": int(len(z["n"])),
                                             "note": "pixels with the four stored members finite but not the sixteen (fix 17)"}
if DESIGN_NOTE:
    OUT["design"] = DESIGN_NOTE
OUTN = f"rel_classU{os.environ.get('A14U_OUT', '')}.json"
safe_json(RES / OUTN, OUT)
print("wrote results/" + OUTN, flush=True)
