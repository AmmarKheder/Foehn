#!/usr/bin/env python3
"""A20 step 2 (CPU only; members and regenerated noise features only: no truth, no station file, no score).
Reads a20/data (collect20.py) and the A18 law (a18/code/out/score18.json 'law2022', members only); writes
a20/data/law20.npz (the laws) and a20/code/out/diag20.json (the members-only diagnostics of A20_draft.md section 4).
Laws, one per lead (1, 3), the level law of A18 4.1 (D_k = xbar (F_k - Fbar) . gamma, relative ridge 1e-3), fitted on the
populated-pixel sample (rows 0..NP-1, each pixel-forecast weight 1) of G2's four members:
  primary (b)   cross-fitted over NBLK = 6 contiguous blocks of the 179 start days: the forecasts of block b are corrected
                with gamma_b fitted on the five other blocks (out of sample in start days); no G11 member enters the fit
  (c)           the 2022 first-design law of A18 (score18.json law2022, 16 fixed-field members), transferred unchanged
                (sensitivity only: fitted on the same 179 start days, so not out of sample in start days)
  ext / ctrl    lead 3 with the fields of leads 1 and 2 of the same member added (36 features); lead 1 with the fields of
                leads 2 and 3 added (a negative control: the lead-1 forecast cannot depend on later noise)
  nulls         the same fit with the features of another member (members 4-7 of the same keys, independent noise) or of
                another start day (day index j + 1, cyclic), on the null rows (every 10th pixel + the station rows)
Q2 parameters per lead and block (A18 4.1, from the other blocks): sigmaL2, kappa2(level bin), tau2(level bin).
Diagnostics (members only, every forecast): out-of-sample R^2 at the pixel sample (pixel and resident weights), at the
station rows (each station weight 1) and by level regime; the predicted equivalent number 4 / (1 - R^2); the truth-free
equivalent number of plain members of the noise-aware mean of four (A18 4.2, with G11's members 4-15, whose noise and
particles are independent of G2's, as the reference: mean[(cbar - xr)^2 - s2r / 12] / mean[s2r]); the relative bias
sum(cbar - xbar) / sum(xbar) (the evidence for the guard tolerance); intervals: compare_fair's moving-block bootstrap
over the 179 start days (blocks of 7, 10,000 resamples, default_rng(0)).
    python -B -u fit20.py [--workers 12]
    python -B -u fit20.py --tests       synthetic members on the real features (null rows): recovery of a known law"""
import common20 as C  # noqa: I001  (first: hooks)

C.install("members")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402

S = {}
SUBP = ("pix", "pixres", "st")                  # primary stats subsets
SUB2 = ("pix", "st")                            # extended / control
SUBN = ("pnull", "st")                          # null rows
PROF = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.15, 0.35, 0.7, 1.0, 0.6, 0.25, 0.05])   # score18 synthetic profile


def setup():
    pos = C.positions()
    S.update(NP=int(pos["NP"]), NALL=len(pos["rows"]), pop=pos["pop"].astype(np.float64), nullrows=pos["nullrows"])
    S["isst"] = np.arange(S["NALL"]) >= S["NP"]
    nr = S["nullrows"]
    S["n_isst"] = nr >= S["NP"]


def arr(name):
    return np.load(C.DATA / f"{name}.npy", mmap_mode="r")


def feats(FG, h, j):
    """(NALL, 4, 12) float64 features of G2's members for lead h, start-day index j"""
    return np.asarray(FG[C.LEADS3.index(h), j], np.float64).transpose(1, 0, 2)


def members(j, li):
    X = np.asarray(arr("mem_g2")[j, li], np.float64)
    D = X - X.mean(1, keepdims=True)
    valid = np.isfinite(X).all(1) & (np.abs(np.nan_to_num(D)).max(1) > 0)
    return X, valid


def pass1(job):
    li, j = job
    h, b = C.LEADS[li], C.block_of(j)
    FG = arr("feat_g2")
    X, valid = members(j, li)
    F = feats(FG, h, j)
    pix, st = valid & ~S["isst"], valid & S["isst"]
    one = np.ones(S["NALL"])
    out = {}
    P = C.Stats(C.NF, 3, binned=True)
    P.add(b, 0, X[pix], F[pix], one[pix])
    P.add(b, 1, X[pix], F[pix], S["pop"][pix])
    P.add(b, 2, X[st], F[st], one[st])
    out.update(P.arrays("p_"))
    other = [hh for hh in C.LEADS3 if hh != h]
    Fe = np.concatenate([F] + [feats(FG, hh, j) for hh in other], axis=2)
    E = C.Stats(3 * C.NF, 2, binned=False)
    E.add(b, 0, X[pix], Fe[pix], one[pix])
    E.add(b, 1, X[st], Fe[st], one[st])
    out.update(E.arrays("e_"))
    del Fe
    nr = S["nullrows"]
    Xn, vn = X[nr], valid[nr]
    pn, sn = vn & ~S["n_isst"], vn & S["n_isst"]
    on = np.ones(len(nr))
    for pre, Fx in (("q_", F[nr]), ("d_", feats(FG, h, (j + 1) % C.ND)[nr]),
                    ("m_", np.asarray(arr("feat_null")[li, j], np.float64).transpose(1, 0, 2))):
        Q = C.Stats(C.NF, 2, binned=False)
        Q.add(b, 0, Xn[pn], Fx[pn], on[pn])
        Q.add(b, 1, Xn[sn], Fx[sn], on[sn])
        out.update(Q.arrays(pre))
    return li, j, out


def pass2(job):
    """per-forecast sums of the members-only diagnostics under the fitted laws"""
    li, j = job
    h, b = C.LEADS[li], C.block_of(j)
    X, valid = members(j, li)
    Y = np.asarray(arr("mem_g11")[j, li], np.float64)
    valid &= np.isfinite(Y).all(1)
    F = feats(arr("feat_g2"), h, j)
    X, Y, F = np.nan_to_num(X), np.nan_to_num(Y), np.nan_to_num(F)
    g = F @ S["gam"][li, b]
    gC = F @ S["gamC"][li]
    cr, cC = C.correct(X, g), C.correct(X, gC)
    ref = Y[:, 4:]
    xr, s2r = ref.mean(1), ref.var(1, ddof=1)
    xb = cr["plain"]
    D = X - xb[:, None]
    pred = xb[:, None] * (g - g.mean(1, keepdims=True))
    cm = X - cr["m1"][:, None] * g                                   # the corrected members c_k (estimators of the mean)
    tf = lambda e: (e - xr) ** 2 - s2r / 12.0                       # noqa: E731
    cols = dict(one=np.ones(len(xb)), V12=s2r, tfp=tf(xb), tfc=tf(cr["corr"]), tfn=tf(cr["noiter"]), tff=tf(cr["fixpt"]),
                tfC=tf(cC["corr"]), bias=cr["corr"] - xb, biasC=cC["corr"] - xb, xbar=xb, sD=(D * D).sum(1),
                sse_law=((D - pred) ** 2).sum(1), s2_4=X.var(1, ddof=1), s2_corr=cm.var(1, ddof=1))
    sums = {}
    for sub, sel, w in (("st", valid & S["isst"], None), ("pix", valid & ~S["isst"], None), ("pixres", valid & ~S["isst"], S["pop"])):
        sums[sub] = {k: float(v[sel].sum() if w is None else w[sel] @ v[sel]) for k, v in cols.items()}
    return li, j, sums


def stats_of(G, NB):
    """G {col: (179 days,)} per-day sums of one lead -> point and resamples of every diagnostic"""
    W = NB
    t = {k: np.concatenate([[v.sum()], W @ v]) for k, v in G.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        o = {"pixel_forecasts": C.ci(t["one"]), "V_member": C.ci(t["V12"] / t["one"]),
             "R2_law_check": C.ci(1 - t["sse_law"] / t["sD"]),
             "bias_rel_corr": C.ci(t["bias"] / t["xbar"]), "bias_rel_A18law": C.ci(t["biasC"] / t["xbar"]),
             "spread_corr_over_plain": C.ci(t["s2_corr"] / t["s2_4"])}
        for nm, k in (("plain", "tfp"), ("corr", "tfc"), ("noiter", "tfn"), ("fixpt", "tff"), ("A18law", "tfC")):
            inv = C.ci(t[k] / t["V12"])
            o[f"inv_meq_tf_{nm}"] = inv
            o[f"meq_tf_{nm}"] = C.inv_ci(inv)
    return o


def run_fit(workers):
    t0 = time.time()
    jobs = [(li, j) for li in range(2) for j in range(C.ND)]
    P = [C.Stats(C.NF, 3, True) for _ in range(2)]
    E = [C.Stats(3 * C.NF, 2, False) for _ in range(2)]
    Q = {p: [C.Stats(C.NF, 2, False) for _ in range(2)] for p in ("q_", "d_", "m_")}
    for x in P + E + Q["q_"] + Q["d_"] + Q["m_"]:
        x.M = 4
    with get_context("fork").Pool(workers) as pool:
        for n, (li, j, out) in enumerate(pool.imap_unordered(pass1, jobs, chunksize=1)):
            for obj, pre in [(P[li], "p_"), (E[li], "e_")] + [(Q[p][li], p) for p in Q]:
                o = C.Stats.from_arrays(out, pre, 4)
                obj.merge(o)
            if n % 60 == 0:
                C.log(f"pass 1: {n + 1}/{len(jobs)} {time.time() - t0:.0f} s")
    C.log(f"pass 1 done {time.time() - t0:.0f} s")
    R = {"what": __doc__, "laws": {}, "R2_out_of_sample": {}, "prediction": {}}
    L = {}
    gam, gfull, gamC = np.zeros((2, C.NBLK, C.NF)), np.zeros((2, C.NF)), np.zeros((2, C.NF))
    r18 = json.loads((C.ROOT / "a18" / "code" / "out" / "score18.json").read_text())["law2022"]
    for li in range(2):
        gamC[li] = np.asarray(r18[li]["gamma"], np.float64)
        gam[li], gfull[li] = C.cross_fit(P[li], 0)
    q2 = [C.q2_params(P[li], gam[li], 0) for li in range(2)]
    regimes = {"lt15": list(range(0, 5)), "15to75": list(range(5, 9)), "gt75": list(range(9, C.NBIN))}
    for li, h in enumerate(C.LEADS):
        lk = f"lead{h}"
        rr = {}
        for si, sub in enumerate(SUBP):
            rr[sub] = dict(primary=C.r2(P[li], gam[li], si), primary_full_fit_in_sample=C.r2(P[li], np.tile(gfull[li], (C.NBLK, 1)), si),
                           A18law=C.r2(P[li], np.tile(gamC[li], (C.NBLK, 1)), si),
                           by_regime={k: C.r2(P[li], gam[li], si, bins=v) for k, v in regimes.items()},
                           pixel_forecasts=float(P[li].n[:, si].sum()))
        ge, _ = C.cross_fit(E[li], 0)
        ext = {sub: dict(R2_36=C.r2(E[li], ge, si), R2_12_same_rows=rr["pix" if sub == "pix" else "st"]["primary"])
               for si, sub in enumerate(SUB2)}
        for sub in ext:
            ext[sub]["increment"] = ext[sub]["R2_36"] - ext[sub]["R2_12_same_rows"]
        L[f"gamma_ext_{lk}"] = ge
        nulls = {}
        for p, nm in (("q_", "own_features_null_rows"), ("d_", "features_of_another_start_day"), ("m_", "features_of_another_member")):
            gq, gqf = C.cross_fit(Q[p][li], 0)
            nulls[nm] = {sub: C.r2(Q[p][li], gq, si) for si, sub in enumerate(SUBN)}
            nulls[nm]["gamma_full_norm"] = float(np.sqrt(gqf @ C.CFEAT @ gqf))
            L[f"gamma_null_{p[0]}_{lk}"] = gq
        R["R2_out_of_sample"][lk] = dict(rr, earlier_or_later_leads=ext, nulls=nulls)
        R["prediction"][lk] = {sub: dict(R2=rr[sub]["primary"], M_eq_pred_4=4.0 / (1.0 - rr[sub]["primary"]),
                                         M_eq_pred_1=1.0 / (1.0 - rr[sub]["primary"])) for sub in SUBP}
        R["laws"][lk] = dict(gamma_by_block=gam[li], gamma_full=gfull[li], gamma_A18=gamC[li],
                             corr_full_vs_A18=float(np.corrcoef(gfull[li], gamC[li])[0, 1]),
                             corr_blocks_min=float(min(np.corrcoef(gam[li][b], gfull[li])[0, 1] for b in range(C.NBLK))),
                             sigmaL2_by_block=q2[li]["sL2"], sigmaL2_analytic_by_block=q2[li]["sL2_analytic"],
                             kappa2_by_block=q2[li]["kappa2"], tau2_by_block=q2[li]["tau2"], n_bin=q2[li]["n_bin"][0],
                             level_edges=C.LEV)
    # ---- pass 2: per-pixel diagnostics under the laws
    S["gam"], S["gamC"] = gam, gamC
    days = {sub: {} for sub in SUBP}
    with get_context("fork").Pool(workers) as pool:
        for n, (li, j, sums) in enumerate(pool.imap_unordered(pass2, jobs, chunksize=1)):
            for sub, d in sums.items():
                for k, v in d.items():
                    days[sub].setdefault(k, np.zeros((2, C.ND)))[li, j] = v
            if n % 60 == 0:
                C.log(f"pass 2: {n + 1}/{len(jobs)} {time.time() - t0:.0f} s")
    NB = C.boot_counts(C.ND)
    R["members_only"] = {f"lead{h}": {sub: stats_of({k: v[li] for k, v in days[sub].items()}, NB) for sub in SUBP}
                         for li, h in enumerate(C.LEADS)}
    for li, h in enumerate(C.LEADS):
        for sub in ("pix", "st"):
            a = R["members_only"][f"lead{h}"][sub]["R2_law_check"]["est"]
            R["members_only"][f"lead{h}"][sub]["R2_check_equals_pass1"] = abs(a - R["R2_out_of_sample"][f"lead{h}"][sub]["primary"]) < 1e-6
    R["primary_choice"] = dict(
        chosen="(b) cross-fitted over 6 blocks of 2022 start days, G2's four members, pixel sample",
        a_2021="not available at 1 km: RUN/eval2021_sel_crps_e01_ar holds the members on the 25 km grid only (map25_mk), "
               "a16/data/cal2021 likewise (plus 5 km maps of the mean); no 1 km member of 2021 under 'ar' is stored",
        c_A18=f"not out of sample in start days (fitted on the same 179 start days of 2022, first design); out-of-sample R2 here "
              f"{[round(R['R2_out_of_sample'][f'lead{h}']['pix']['A18law'], 4) for h in C.LEADS]} against "
              f"{[round(R['R2_out_of_sample'][f'lead{h}']['pix']['primary'], 4) for h in C.LEADS]} for (b) (pixels, leads 1, 3)")
    L.update(gamma_primary=gam, gamma_primary_full=gfull, gamma_A18=gamC, sL2=np.stack([q["sL2"] for q in q2]),
             sL2_analytic=np.stack([q["sL2_analytic"] for q in q2]), kappa2=np.stack([q["kappa2"] for q in q2]),
             tau2=np.stack([q["tau2"] for q in q2]), kappa2_raw=np.stack([q["kappa2_raw"] for q in q2]),
             tau2_raw=np.stack([q["tau2_raw"] for q in q2]), n_bin=np.stack([q["n_bin"] for q in q2]), LEV=C.LEV,
             blocks=np.array([C.block_of(j) for j in range(C.ND)], np.int64), days=np.array(C.DAYS, np.int64))
    for li in range(2):
        L.update({k + f"_{li}": v for k, v in P[li].arrays("P_").items()})
    L["meta20_sha256"] = np.array(C.sha_file(C.DATA / "meta20.npz"))
    L["code_sha256"] = np.array(json.dumps({n: C.sha_file(C.CODE / n) for n in ("common20.py", "fit20.py")}))
    C.save_npz(C.DATA / "law20.npz", **L)
    R["law20_sha256"] = C.sha_file(C.DATA / "law20.npz")
    R["meta20_sha256"] = str(L["meta20_sha256"])
    R["code_sha256"] = json.loads(str(L["code_sha256"]))
    R["seconds"] = round(time.time() - t0)
    C.save_json(C.OUT / "diag20.json", R)
    C.log("diag20 written", json.dumps(C.jsonable(R["prediction"])), f"{R['seconds']} s")


# ======================================================================== tests: recovery of a known law (synthetic members)
def run_tests():
    t0 = time.time()
    nr = S["nullrows"]
    gu = PROF / np.sqrt(PROF @ C.CFEAT @ PROF)
    s, k = 0.25, 0.25 / np.sqrt(3.0)                                 # R^2 = s^2 / (s^2 + k^2) = 0.75
    worlds = {"law": "lin", "none": "lin", "log": "log"}
    out = {"what": "fit20 --tests: synthetic members x_k = mu (1 + s u_k + kappa e_k) (or log form) on the REAL features of "
                   "G2's members at the null rows, mu = the mean of G11's members 4-15 (members only, independent of G2's noise); "
                   "'none': u_k from the "
                   "features of members 4-7 (independent noise)", "s": s, "kappa": k, "worlds": {}}
    FG, FN = arr("feat_g2"), arr("feat_null")
    St = {wn: [C.Stats(C.NF, 1, True) for _ in range(2)] for wn in worlds}
    for li, h in enumerate(C.LEADS):
        for j in range(C.ND):
            X, valid = members(j, li)
            X, valid = X[nr], valid[nr] & ~S["n_isst"]
            F = feats(FG, h, j)[nr]
            FH = np.asarray(FN[li, j], np.float64).transpose(1, 0, 2)
            # the level: the mean of G11's members 4-15, independent of G2's noise (the plain mean of G2's four carries
            # their own noise law, x_bar = mu (1 + Lbar), and would correlate the level with u)
            mu = np.maximum(np.nan_to_num(np.asarray(arr("mem_g11")[j, li][nr, 4:], np.float64)).mean(1), 0.5)
            e = np.random.default_rng([20261121, 20, li, j]).standard_normal((len(nr), 4))
            for wn, form in worlds.items():
                u = (FH if wn == "none" else F) @ gu
                Xs = mu[:, None] * (np.exp(s * u - s * s / 2 + k * e - k * k / 2) if form == "log" else 1 + s * u + k * e)
                St[wn][li].add(C.block_of(j), 0, Xs[valid], F[valid], np.ones(int(valid.sum())))
        C.log(f"fit20 tests: lead {h} accumulated {time.time() - t0:.0f} s")
    for wn in worlds:
        res = {}
        for li, h in enumerate(C.LEADS):
            g, gf = C.cross_fit(St[wn][li], 0)
            q = C.q2_params(St[wn][li], g, 0)
            ok_bins = q["n_bin"][0] >= C.MINBIN
            res[f"lead{h}"] = dict(R2_out_of_sample=C.r2(St[wn][li], g, 0), corr_gamma_true=[float(np.corrcoef(g[b], gu)[0, 1]) for b in range(C.NBLK)],
                                   gamma_scale=float(gf @ gu / (gu @ gu)), sigmaL2=float(q["sL2"].mean()),
                                   kappa2_median_ratio=float(np.median(q["kappa2"][:, ok_bins] / (k * k))),
                                   tau2_median_ratio=float(np.median(q["tau2"][:, ok_bins] / (s * s + k * k))))
        if wn == "law":
            # the law's level is the four-member mean xbar = mu (1 + c), Var c = (s^2 + k^2) / 4: least squares attenuates the
            # slope to beta = s / (1 + Var c) and R^2 to 1 - ((s - beta)^2 + beta^2 Var c + k^2) / (s^2 + k^2) (analytic)
            vc = (s * s + k * k) / 4
            beta = s / (1 + vc)
            r2t = 1 - ((s - beta) ** 2 + beta * beta * vc + k * k) / (s * s + k * k)
            out["targets_law"] = dict(beta=beta, R2=r2t, sigmaL2=beta * beta)
            ok = all(abs(res[L]["R2_out_of_sample"] - r2t) < 0.01 and min(res[L]["corr_gamma_true"]) > 0.99
                     and abs(res[L]["gamma_scale"] / beta - 1) < 0.02 and abs(res[L]["sigmaL2"] / beta ** 2 - 1) < 0.05
                     and 0.95 <= res[L]["kappa2_median_ratio"] <= 1.20 and abs(res[L]["tau2_median_ratio"] - 1) < 0.1 for L in res)
        elif wn == "none":
            ok = all(abs(res[L]["R2_out_of_sample"]) < 0.01 for L in res)
        else:
            ok = all(res[L]["R2_out_of_sample"] > 0.6 and min(res[L]["corr_gamma_true"]) > 0.98 for L in res)
        out["worlds"][wn] = dict(result=res, passed=bool(ok))
        C.log(f"fit20 test {wn}: {'PASS' if ok else 'FAIL'} {json.dumps(C.jsonable(res))[:400]}")
    out["all_passed"] = all(v["passed"] for v in out["worlds"].values())
    out["seconds"] = round(time.time() - t0)
    out["code_sha256"] = {n: C.sha_file(C.CODE / n) for n in ("common20.py", "fit20.py")}
    C.save_json(C.OUT / "fit20_tests.json", out)
    return 0 if out["all_passed"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tests", action="store_true")
    a = ap.parse_args()
    assert np.__version__ == "1.26.3", f"A20 runs in the job container (numpy 1.26.3), not {np.__version__}"
    setup()
    if a.tests:
        return run_tests()
    run_fit(a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
