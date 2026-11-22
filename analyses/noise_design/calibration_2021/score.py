#!/usr/bin/env python3
"""A15 scoring and decision (a15/A15_draft.md sections 6.3 and 7), CPU, after run M. 2021 only (audit hook against 2022
files), no station or monitor file. Definitions and code: episodes_diag/code/revise.py, imported read only.
Comparison: the arm (data/main, rho*) against the stored run eval2021_sel_crps_e01 on the 30 Conf starts, leads 1-7, the
cells complete in every stored 2021 run (prep.py), like for like on member 0. ONE revise.Agg holds both designs (groups
'st|<thr>', 'arm|<thr>', 'st|E5', 'arm|E5'), so the paired two-way bootstrap (runs x 10-degree tiles, 2000 draws, seed
20261121) shares its draws; D = |stored - target| - |arm - target| > 0 means the arm is closer.
  stored series: y = Y, x = member 0, m = mean (other slots = member 0, unused);
  arm series:    y = Y, x = arm member 0, m = arm mean (map25), xs / xo / xe1 = arm members 1 / 2 / 3, xer = xe0 = member 0;
  E5 of each design is centered with its own pass-1 maps over the 30 runs.
G (code guard): fair CRPS and RMSE of the mean per lead from the region sums, and the mean seam per lead.
Q1 haz_within_x, Q2 drop_within_x, Q3 P1_err_x and P1_2day_err_x, Q4 E2_late_ratio_x (target 1) and P7b_x (target 0),
Q5 the arm's centered E5/log/cen/k1/d_rho, Q6 RMSE (residents) of the 30-start time means per lead (member 0, mean).
Outcome by the rule of section 7.3: INVALID / SUCCESS / FAIL / UNDECIDED.
    python score.py -> results/score.json"""
import arnoise as ar  # noqa: I001  (first)
import json
import time

import numpy as np

ar.deny_2022()
NBOOT, SEED, M, H = 2000, 20261121, 4, 7
EPS = 0.1
TRUTH_ACF = {2: 0.36, 3: 0.26, 4: 0.21, 5: 0.17, 6: 0.13}


def load_arm(cells):
    parts = [p for p in sorted((ar.A15 / "data" / "main").glob("part_*.npz")) if ".tmp" not in p.name]
    arm, recs, seams, rhos, cols = {}, [], [], set(), None
    for p in parts:
        with np.load(p) as z:
            assert str(z["base"]) == "philox", p
            c = [str(x) for x in z["rec_cols"]]
            assert cols is None or c == cols
            cols = c
            keys, kr = z["keys"].tolist(), z["keys_rho"].tolist()
            mk, m25, g25 = z["map25_mk"], z["map25"], z["map25_ghap"]
            for j, ((d, h, dt), r) in enumerate(zip(keys, kr)):
                assert (d, h) not in arm and dt == d + h, (d, h)
                rhos.add(float(r))
                arm[(d, h)] = (mk[j].reshape(M, -1)[:, cells], m25[j].reshape(-1)[cells], g25[j].reshape(-1)[cells])
            recs.append(z["recs"])
            seams.append(z["seams"])
    return arm, np.concatenate(recs), np.concatenate(seams), rhos, cols, [p.name for p in parts]


def load_stored_recs():
    recs, seams, cols = [], [], None
    for p in sorted(ar.EV.glob("part_*.npz")):
        with np.load(p) as z:                                            # keys, recs and seams only
            c = [str(x) for x in z["rec_cols"]]
            assert cols is None or c == cols
            cols = c
            recs.append(z["recs"])
            seams.append(z["seams"])
    return np.concatenate(recs), np.concatenate(seams), cols


def guard(ra, sa, rs, ss, cols):
    """per lead: sum crps / sum n, sqrt(sum sse / sum n) over the region rows of the Conf starts; mean seam"""
    j = {c: cols.index(c) for c in ("start_doy", "lead", "region", "n", "sse", "crps")}
    conf = np.array(ar.CONF)

    def sel(r):
        return r[np.isin(r[:, j["start_doy"]].astype(int), conf)]
    ra, rs = sel(ra), sel(rs)
    ka = sorted(map(tuple, ra[:, :3].astype(int).tolist()))
    ks = sorted(map(tuple, rs[:, :3].astype(int).tolist()))
    assert ka == ks and len(set(ka)) == len(ka), "region rows of arm and stored differ"
    sa, ss = sa[np.isin(sa[:, 0].astype(int), conf)], ss[np.isin(ss[:, 0].astype(int), conf)]
    out = {"per_lead": {}, "rows": len(ka)}
    crps, rmse, seam = {"arm": [], "st": []}, {"arm": [], "st": []}, {"arm": [], "st": []}
    for h in range(1, H + 1):
        for nm, r, s in (("arm", ra, sa), ("st", rs, ss)):
            q = r[r[:, j["lead"]].astype(int) == h]
            n = q[:, j["n"]].sum()
            crps[nm].append(q[:, j["crps"]].sum() / n)
            rmse[nm].append(np.sqrt(q[:, j["sse"]].sum() / n))
            sv = s[s[:, 1].astype(int) == h][:, 2]
            assert len(sv) == 30, (nm, h, len(sv))
            seam[nm].append(float(np.mean(sv)))
    ok = True
    for h in range(H):
        rc, rr = crps["arm"][h] / crps["st"][h] - 1, rmse["arm"][h] / rmse["st"][h] - 1
        sok = seam["arm"][h] <= seam["st"][h] + 0.02
        good = abs(rc) <= 0.025 and abs(rr) <= 0.025 and sok
        ok &= good
        out["per_lead"][h + 1] = {"crps_arm": crps["arm"][h], "crps_st": crps["st"][h], "crps_ratio_m1": rc,
                                  "rmse_arm": rmse["arm"][h], "rmse_st": rmse["st"][h], "rmse_ratio_m1": rr,
                                  "seam_arm": seam["arm"][h], "seam_st": seam["st"][h], "pass": good}
    mc = np.mean(crps["arm"]) / np.mean(crps["st"]) - 1
    mr = np.mean(rmse["arm"]) / np.mean(rmse["st"]) - 1
    out["mean_over_leads"] = {"crps_ratio_m1": mc, "rmse_ratio_m1": mr}
    out["pass"] = bool(ok and abs(mc) <= 0.015 and abs(mr) <= 0.015)
    return out


def q6(Fst, Farm, Y, w, Kr, chunk=50):
    """RMSE (residents) per lead of the 30-start mean of F against the mean of Y; arm/stored - 1, run bootstrap"""
    N = Y.shape[0]
    W = w / w.sum()

    def tmean(K, F):                                                    # K (b, N) counts -> (b, H, C) time means
        return np.einsum("bn,nhc->bhc", K, F) / K.sum(1)[:, None, None]

    def rmse(K, F, mY):
        return np.sqrt(((tmean(K, F) - mY) ** 2 * W).sum(-1))
    one = np.ones((1, N))
    mY1 = tmean(one, Y)
    pst, parm = rmse(one, Fst, mY1)[0], rmse(one, Farm, mY1)[0]
    bs, ba = [], []
    for b in range(0, Kr.shape[0], chunk):
        mY = tmean(Kr[b:b + chunk], Y)
        bs.append(rmse(Kr[b:b + chunk], Fst, mY))
        ba.append(rmse(Kr[b:b + chunk], Farm, mY))
    bs, ba = np.concatenate(bs), np.concatenate(ba)
    rat = ba / bs - 1
    ratm = ba.mean(1) / bs.mean(1) - 1
    return {"rmse_st": pst, "rmse_arm": parm, "ratio_m1": parm / pst - 1, "ratio_m1_runs_ci": np.percentile(rat, [2.5, 97.5], 0),
            "mean_over_leads_ratio_m1": parm.mean() / pst.mean() - 1,
            "mean_over_leads_ratio_m1_runs_ci": np.percentile(ratm, [2.5, 97.5])}


def lag_pairs(Xs, Mn, Y, w, members=None):
    """per lead pair (h, h+1): centered lag-1 correlation of A (member vs others; pooled over member choices when
    `members` (N, 4, H, C) is given) and the de-diluted one of the truth's deviation B (member-0 o)"""
    def dev(X, o):
        return np.log(np.maximum(X, EPS)) - np.log(np.maximum(o, EPS))
    o = (M * Mn - Xs) / (M - 1)
    A, Bv = dev(Xs, o), dev(Y, o)
    A, Bv = A - A.mean(0), Bv - Bv.mean(0)
    S = lambda a, b: float((a * b * w).sum())                          # noqa: E731
    out = {}
    for h in range(H - 1):
        ca, va1, va2 = S(A[:, h], A[:, h + 1]), S(A[:, h], A[:, h]), S(A[:, h + 1], A[:, h + 1])
        cb, vb1, vb2 = S(Bv[:, h], Bv[:, h + 1]), S(Bv[:, h], Bv[:, h]), S(Bv[:, h + 1], Bv[:, h + 1])
        out[f"{h + 1}-{h + 2}"] = {"rho_A_m0": ca / np.sqrt(va1 * va2),
                                   "rho_y_dediluted": (cb - ca / M) / np.sqrt((vb1 - va1 / M) * (vb2 - va2 / M))}
    if members is not None:
        Ssum = members.sum(1)
        Am = np.stack([dev(members[:, m], (Ssum - members[:, m]) / (M - 1)) for m in range(M)], 1)   # (N, 4, H, C)
        Am = Am - Am.mean(0)
        for h in range(H - 1):
            c = float((Am[:, :, h] * Am[:, :, h + 1] * w).sum())
            v1, v2 = float((Am[:, :, h] ** 2 * w).sum()), float((Am[:, :, h + 1] ** 2 * w).sum())
            out[f"{h + 1}-{h + 2}"]["rho_A_4choices"] = c / np.sqrt(v1 * v2)
    return out


def custom(Dall, revise):
    o = {}
    for pfx in ("st", "arm"):
        Dp = {k[len(pfx) + 1:]: v for k, v in Dall.items() if k.startswith(pfx + "|")}
        for tk in revise.THR:
            for k, v in revise.stats_thr(Dp, tk).items():
                o[f"{pfx}/{tk}/{k}"] = v
        for k, v in revise.stats_e5(Dp).items():
            o[f"{pfx}/{k}"] = v
    for tk in revise.THR:
        g = lambda p, k: o[f"{p}/{tk}/{k}"]                              # noqa: E731
        o[f"D/{tk}/Q1_haz_within"] = np.abs(g("st", "haz_within_x") - g("st", "haz_within_y")) - \
            np.abs(g("arm", "haz_within_x") - g("st", "haz_within_y"))
        o[f"D/{tk}/Q2_drop_within"] = np.abs(g("st", "drop_within_x") - g("st", "drop_within_y")) - \
            np.abs(g("arm", "drop_within_x") - g("st", "drop_within_y"))
        for k in ("P1_err_x", "P1_2day_err_x", "P7b_x"):
            o[f"D/{tk}/Q3Q4_{k}"] = np.abs(g("st", k)) - np.abs(g("arm", k))
        o[f"D/{tk}/Q4_E2_late_ratio_x"] = np.abs(g("st", "E2_late_ratio_x") - 1) - np.abs(g("arm", "E2_late_ratio_x") - 1)
        for k in ("haz_within", "drop_within", "P1_err", "E2_late_ratio", "P7b"):    # arm: mean over its four members
            o[f"arm4/{tk}/{k}"] = np.mean([g("arm", f"{k}_{s}") for s in ("x", "xs", "xo", "xe1")], 0)
    return o


def main():
    t0 = time.time()
    R = {"what": __doc__}
    rj = json.loads((ar.A15 / "results" / "rho.json").read_text())
    sm = json.loads((ar.A15 / "results" / "smoke_check.json").read_text())
    prep = json.loads((ar.A15 / "results" / "prep.json").read_text())
    R["cal"] = {"tau_Cal": prep["tau_Cal"], "tau_Cal_E5": prep["tau_Cal_E5"], "f_hat": rj["f_hat"], "f_iso": rj["f_iso"],
                "rho_star": rj["rho_star"], "clipped": rj["clipped"], "f_hat_per_member_choice": rj["f_hat_per_member_choice"]}
    if rj["clipped"]:
        R.update(outcome="UNDECIDED", reasons=["rho* clipped: A15 stops before run M (draft 4.5)"])
        ar.save_json(ar.A15 / "results" / "score.json", R)
        return
    D, revise = ar.episodes_2021()
    assert ar.cells_hash(D["cells"]) == prep["cells_sha256"]
    lab = [int(v) for v in D["labels"]]
    Ds = ar.subset_runs(D, [lab.index(d) for d in ar.CONF])
    cells, w = Ds["cells"], Ds["pop"].astype(np.float64)
    C, N = len(cells), len(ar.CONF)
    arm, ra, sa, rhos, cols, parts = load_arm(cells)
    want = {(d, h) for d in ar.CONF for h in range(1, H + 1)}
    assert set(arm) == want and len(arm) == 210, sorted(set(arm) ^ want)[:10]
    assert rhos == {float(rj["rho_star"])}, (rhos, rj["rho_star"])
    Xa = np.empty((N, M, H, C), np.float32)
    Ma = np.empty((N, H, C), np.float32)
    bad = 0
    for r, d in enumerate(ar.CONF):
        for h in range(1, H + 1):
            mk, m25, g25 = arm[(d, h)]
            Xa[r, :, h - 1] = mk.astype(np.float32)
            Ma[r, h - 1] = m25.astype(np.float32)
            bad += not np.array_equal(g25.view(np.uint16), Ds["Y"][r, h - 1].view(np.uint16))
    assert bad == 0, f"arm map25_ghap differs from ext_2021 Y in {bad} forecasts"
    assert np.isfinite(Xa).all() and np.isfinite(Ma).all()
    del arm
    rs_, ss_, cols_s = load_stored_recs()
    assert cols == cols_s
    R["G"] = guard(ra, sa, rs_, ss_, cols)
    print(f"loaded {time.time() - t0:.0f}s: {C} cells, rho* {rj['rho_star']}, G pass {R['G']['pass']}", flush=True)

    # ---------------- pass 1 of each design (centering maps over the 30 runs) and the paired aggregation
    Da = dict(Ds)
    Da["X"], Da["Mn"] = Xa[:, 0].copy(), Ma.copy()                    # float16 values of the part files, as float32
    P1 = {}
    for nm, DD in (("st", Ds), ("arm", Da)):
        p = revise.pass1(DD)
        P1[nm] = revise.p1_summary(p, DD)
    maps = {nm: {"cenA_log": P1[nm][1]["A_log"], "cenB_log": P1[nm][1]["B_log"], "cenA_lin": P1[nm][1]["A_lin"],
                 "cenB_lin": P1[nm][1]["B_lin"]} for nm in P1}
    R["pass1"] = {nm: {k: P1[nm][0][k] for k in P1[nm][0] if k.startswith(("pass1_", "static_share"))} for nm in P1}
    tid = Ds["tid"]
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    agg = revise.Agg(N, starts, w)
    nt = Ds["truth"].shape[0]
    gi0 = int(D["truth_gi0"])
    for r in range(N):
        t = int(Ds["tidx"][r])
        assert 0 <= gi0 + t - 1461 <= 364 and gi0 + t - 1461 == ar.CONF[r], (r, t)
        y0 = Ds["truth"][t].astype(np.float32)
        Y = Ds["Y"][r].astype(np.float32)
        Xs, Mns = Ds["X"][r].astype(np.float32), Ds["Mn"][r].astype(np.float32)
        P, W = Ds["P"][r].astype(np.float32), Ds["W"][r].astype(np.float32)
        after = []
        for k in (8, 9):
            if t + k < nt:
                assert gi0 + t + k - 1461 <= 364
                after.append(Ds["truth"][t + k])
            else:
                after.append(np.full(C, np.nan, np.float32))
        Yext = np.vstack([Y, np.stack(after).astype(np.float32)])
        A = [Xa[r, m] for m in range(M)]
        S = {"st": {"y": Y, "x": Xs, "m": Mns, "xs": Xs, "xo": Xs, "xe1": Xs, "xer": Xs, "xe0": Xs},
             "arm": {"y": Y, "x": A[0], "m": Ma[r], "xs": A[1], "xo": A[2], "xe1": A[3], "xer": A[0], "xe0": A[0]}}
        for nm in S:
            assert tuple(S[nm]) == revise.SER
        for tk, T in revise.THR.items():
            Tc = Ds["p95"].astype(np.float32) if T is None else np.full(C, T, np.float32)
            f = np.isfinite(Tc)
            Tc = np.where(f, Tc, np.float32(np.inf))
            for nm in S:
                agg.add(r, f"{nm}|{tk}", {f"{nm}|{tk}/{k}": v for k, v in revise.thr_cols(y0, S[nm], P, W, Yext, Tc, f).items()})
        agg.add(r, "st|E5", {f"st|{k}": v for k, v in revise.e5_cols(Xs, Y, Mns, maps["st"], C).items()})
        agg.add(r, "arm|E5", {f"arm|{k}": v for k, v in revise.e5_cols(A[0], Y, Ma[r], maps["arm"], C).items()})
        if r % 10 == 0:
            print(f"run {r + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    res, info = revise.summarise(agg, N, 1, NBOOT, SEED, fn=lambda Dd: custom(Dd, revise))
    R["bootstrap"] = {"draws": NBOOT, "seed": SEED, "run_block": 1, "N": N, **info}
    print(f"summarised {time.time() - t0:.0f}s {info}", flush=True)
    g = lambda k: res[k]                                                 # noqa: E731

    # ---------------- primary statistics
    Q = {}
    for tk in ("37.5", "P95", "75"):
        Q[tk] = {"Q1_haz_within": {"D": g(f"D/{tk}/Q1_haz_within"), "stored": g(f"st/{tk}/haz_within_x"),
                                   "arm": g(f"arm/{tk}/haz_within_x"), "truth": g(f"st/{tk}/haz_within_y")},
                 "Q2_drop_within": {"D": g(f"D/{tk}/Q2_drop_within"), "stored": g(f"st/{tk}/drop_within_x"),
                                    "arm": g(f"arm/{tk}/drop_within_x"), "truth": g(f"st/{tk}/drop_within_y")},
                 "Q3_P1_err": {"D": g(f"D/{tk}/Q3Q4_P1_err_x"), "stored": g(f"st/{tk}/P1_err_x"), "arm": g(f"arm/{tk}/P1_err_x")},
                 "Q3_P1_2day_err": {"D": g(f"D/{tk}/Q3Q4_P1_2day_err_x"), "stored": g(f"st/{tk}/P1_2day_err_x"),
                                    "arm": g(f"arm/{tk}/P1_2day_err_x")},
                 "Q4_E2_late_ratio": {"D": g(f"D/{tk}/Q4_E2_late_ratio_x"), "stored": g(f"st/{tk}/E2_late_ratio_x"),
                                      "arm": g(f"arm/{tk}/E2_late_ratio_x")},
                 "Q4_P7b": {"D": g(f"D/{tk}/Q3Q4_P7b_x"), "stored": g(f"st/{tk}/P7b_x"), "arm": g(f"arm/{tk}/P7b_x")}}
    R["Q"] = Q
    R["Q5"] = {"d_rho_arm_cen": g("arm/E5/log/cen/k1/d_rho"), "rho_A_arm_cen": g("arm/E5/log/cen/k1/rho_A"),
               "rho_B_arm_cen": g("arm/E5/log/cen/k1/rho_B"), "rho_y_dediluted_arm_cen": g("arm/E5/log/cen/k1/rho_y_dediluted"),
               "rho_y_dediluted_stored_cen": g("st/E5/log/cen/k1/rho_y_dediluted"),
               "stored_member_uncentered_d_rho_P6": g("st/E5/log/raw/k1/d_rho")}
    Kr = revise.boot_counts(N, 1, NBOOT, SEED)
    Ys = Ds["Y"].astype(np.float32)
    R["Q6"] = {"a_member0": q6(Ds["X"].astype(np.float32), Xa[:, 0], Ys, w, Kr),
               "b_mean": q6(Ds["Mn"].astype(np.float32), Ma, Ys, w, Kr)}
    print(f"Q6 {time.time() - t0:.0f}s", flush=True)

    # ---------------- reported (draft 6.3), never gating
    rep = {}
    rep["per_pair_lag1"] = {"arm": lag_pairs(Xa[:, 0], Ma, Ys, w, members=np.ascontiguousarray(Xa)),
                            "stored": lag_pairs(Ds["X"].astype(np.float32), Ds["Mn"].astype(np.float32), Ys, w)}
    rep["acf_lags_2_6"] = {k: {"arm_rho_A_cen": g(f"arm/E5/log/cen/k{k}/rho_A")["est"],
                               "truth_dediluted_cen_arm_o": g(f"arm/E5/log/cen/k{k}/rho_y_dediluted")["est"],
                               "truth_dediluted_cen_stored_o": g(f"st/E5/log/cen/k{k}/rho_y_dediluted")["est"],
                               "truth_45_runs_reference": TRUTH_ACF[k]} for k in range(2, 7)}
    rep["spread_over_error_by_lead"] = {"arm_cen": g("arm/E5/log/cen/spread_over_error_by_lead"),
                                        "stored_raw": g("st/E5/log/raw/spread_over_error_by_lead"),
                                        "level_ratio_A2_over_B2_arm_cen": g("arm/E5/log/cen/level_ratio_A2_over_B2"),
                                        "level_ratio_A2_over_B2_stored_raw": g("st/E5/log/raw/level_ratio_A2_over_B2")}
    rep["hazard_by_weather_class"] = {f"{nm}/{tk}/{k}": g(f"{nm}/{tk}/{k}") for nm in ("st", "arm") for tk in ("37.5", "P95")
                                      for k in ("hazc_first_x", "hazc_within_x", "hazc_first_y", "hazc_within_y",
                                                "hazc_first_x_over_y", "hazc_within_x_over_y")}
    rep["arm_four_member_means"] = {k: v for k, v in res.items() if k.startswith("arm4/")}
    emu = json.loads((ar.EPI / "results" / "revision_S1.json").read_text())["stats"]
    rep["emulation_45_runs_revision_S1"] = {k: emu.get(f"37.5/{k}") for k in
                                            ("haz_within_xer", "haz_within_x", "haz_within_y", "drop_within_xer", "drop_within_x",
                                             "drop_within_y", "P1_err_xer", "P1_err_x", "E2_late_ratio_xer", "E2_late_ratio_x",
                                             "P7b_xer", "P7b_x")}
    R["reported"] = rep

    # ---------------- decision (draft 7.3)
    q = Q["37.5"]
    pt = lambda k: float(q[k]["D"]["est"])                               # noqa: E731
    lo = lambda k: float(q[k]["D"]["two_way"][0])                         # noqa: E731
    hi = lambda k: float(q[k]["D"]["two_way"][1])                         # noqa: E731
    q5 = float(R["Q5"]["d_rho_arm_cen"]["est"])
    q6b = R["Q6"]["b_mean"]
    reasons, invalid = [], []
    if not sm.get("all_passed"):
        invalid.append("smoke gate failed")
    if not R["G"]["pass"]:
        invalid.append("code guard G failed")
    cond = {"G": R["G"]["pass"], "Q1_ci_above_0": lo("Q1_haz_within") > 0, "Q2_ci_above_0": lo("Q2_drop_within") > 0,
            "Q3_first_day_point": pt("Q3_P1_err") > 0, "Q3_2day_point": pt("Q3_P1_2day_err") > 0,
            "Q4_late_ratio_point": pt("Q4_E2_late_ratio") > 0, "Q4_P7b_point": pt("Q4_P7b") > 0, "Q5_abs_le_0.10": abs(q5) <= 0.10,
            "Q6b_every_lead_le_1pct": bool(np.all(np.asarray(q6b["ratio_m1"]) <= 0.01)),
            "P95_Q1_point": float(Q["P95"]["Q1_haz_within"]["D"]["est"]) > 0,
            "P95_Q2_point": float(Q["P95"]["Q2_drop_within"]["D"]["est"]) > 0}
    fail = {"Q1_ci_below_0": hi("Q1_haz_within") < 0, "Q2_ci_below_0": hi("Q2_drop_within") < 0,
            "Q6b_mean_over_leads_gt_1pct": float(q6b["mean_over_leads_ratio_m1"]) > 0.01}
    if invalid:
        outcome, reasons = "INVALID", invalid
    elif all(cond.values()):
        outcome, reasons = "SUCCESS", ["every SUCCESS condition of 7.3 holds"]
    elif any(fail.values()):
        outcome, reasons = "FAIL", [k for k, v in fail.items() if v]
    else:
        outcome, reasons = "UNDECIDED", [k for k, v in cond.items() if not v]
    R.update(outcome=outcome, reasons=reasons, success_conditions=cond, fail_conditions=fail,
             n_cells=C, residents=float(w.sum()), parts=parts, rho_star=rj["rho_star"], seconds=round(time.time() - t0, 1))
    R["all_stats"] = res
    ar.save_json(ar.A15 / "results" / "score.json", R)
    print(f"OUTCOME {outcome}: {reasons} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
