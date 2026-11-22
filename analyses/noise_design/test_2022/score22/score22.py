#!/usr/bin/env python3
"""A16 section 4 point 5 and Addendum 1 point 1 (protocol_amendments/A16_test_year_evolving_noise.md): the replication of
A15 on a year, CPU. Q1-Q6 of A15 section 7.2 and the reported items of A15 section 6.3 for one pair of run folders, arm
(2022: G1 = RUN/eval_final_crps_e01_ar) against stored (2022: F1 = RUN/eval_final_crps_e01), with the code of A15's scorer:
a15/score.py (custom, q6, lag_pairs) and episodes_diag/code/revise.py, imported read only (prep22.setup). Inputs, cells
and pairing asserts: prep22.build. Like for like on member 0 (thresholds 37.5 ug m-3, P95, 75 ug m-3), the mean of the
four arm members beside (map25_mk); D = |stored - target| - |arm - target| > 0 means the arm is closer. ONE revise.Agg
holds both designs, so the paired two-way bootstrap (runs x 10-degree tiles, 2000 draws, seed 20261121) shares its draws.
Changed from a15/score.py (A16 Addendum 1 point 1, nothing else):
  year, arm folder and stored folder are arguments; the starts are those of the arm (2021: the 30 Conf starts; 2022: the
  179 starts); the cells are those of prep22 (complete in every forecast of both runs);
  the runs of the bootstrap are drawn in moving blocks of --block consecutive starts (revise.summarise and
  revise.boot_counts with block = 4 for 2022; block = 1 reproduces A15), the tile draw is unchanged;
  Q6 is the RMSE (residents) of the time mean over all scored starts, per lead, for member 0 and for the mean.
Dropped or not applicable (listed in the output under 'not_applicable'): the calibration block of A15 (tau_Cal, f_hat,
f_iso, rho*: 2021 Cal starts), the smoke gate of A15 6.2, the pass flags of the guard G (on 2022 the guard is check C4 of
A16, judged by a16/check_runs.py; its values are reported here), the outcome classes of A15 7.3 (success / fail
conditions, reasons; A16 4.5: reported, never gating), and, for a year other than 2021, the 2021 reference values of the
truth's ACF (45 runs) and the 45-run 2021 emulation values of revision_S1.json; for the null test (arm = stored), the
four-member items (the stored run has no map25_mk).
Optional: --compare <a15 score.json> (regression: max |difference| over the numeric leaves of Q, Q5, Q6 and the others).
No station or monitor file (audit hook of a15/arnoise.py). Reports, decides nothing.
    python score22.py --year 2022 --arm <G1> --stored <F1> --block 4 --prep ../results/score22_prep.json \
                      --out ../results/score22.json --md ../results/score22.md"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep22  # noqa: E402

H, M = prep22.H, prep22.M
NBOOT, SEED = 2000, 20261121
TK = ("37.5", "P95", "75")


def guard(ra, sa, rs, ss, cols, starts):
    """a15/score.guard on the given starts, without the pass flags (A16: check C4 of a16/check_runs.py): per lead
    sum crps / sum n, sqrt(sum sse / sum n) over the region rows, and the mean seam"""
    j = {c: cols.index(c) for c in ("start_doy", "lead", "region", "n", "sse", "crps")}
    st = np.array(starts)

    def sel(r):
        return r[np.isin(r[:, j["start_doy"]].astype(int), st)]
    ra, rs = sel(ra), sel(rs)
    ka = sorted(map(tuple, ra[:, :3].astype(int).tolist()))
    ks = sorted(map(tuple, rs[:, :3].astype(int).tolist()))
    assert ka == ks and len(set(ka)) == len(ka), "region rows of arm and stored differ"
    sa, ss = sa[np.isin(sa[:, 0].astype(int), st)], ss[np.isin(ss[:, 0].astype(int), st)]
    out = {"per_lead": {}, "rows": len(ka)}
    crps, rmse, seam = {"arm": [], "st": []}, {"arm": [], "st": []}, {"arm": [], "st": []}
    for h in range(1, H + 1):
        for nm, r, s in (("arm", ra, sa), ("st", rs, ss)):
            q = r[r[:, j["lead"]].astype(int) == h]
            n = q[:, j["n"]].sum()
            crps[nm].append(q[:, j["crps"]].sum() / n)
            rmse[nm].append(np.sqrt(q[:, j["sse"]].sum() / n))
            sv = s[s[:, 1].astype(int) == h][:, 2]
            assert len(sv) == len(starts), (nm, h, len(sv))
            seam[nm].append(float(np.mean(sv)))
    for h in range(H):
        out["per_lead"][h + 1] = {"crps_arm": crps["arm"][h], "crps_st": crps["st"][h],
                                  "crps_ratio_m1": crps["arm"][h] / crps["st"][h] - 1,
                                  "rmse_arm": rmse["arm"][h], "rmse_st": rmse["st"][h],
                                  "rmse_ratio_m1": rmse["arm"][h] / rmse["st"][h] - 1,
                                  "seam_arm": seam["arm"][h], "seam_st": seam["st"][h]}
    out["mean_over_leads"] = {"crps_ratio_m1": np.mean(crps["arm"]) / np.mean(crps["st"]) - 1,
                              "rmse_ratio_m1": np.mean(rmse["arm"]) / np.mean(rmse["st"]) - 1}
    out["note"] = "values only; on 2022 the guard is check C4 of A16, judged by a16/check_runs.py"
    return out


# ======================================================================== comparisons (regression T1, null test T2)
def leaves(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{pre}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{pre}[{i}]")
    else:
        yield pre, x


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare(mine, ref):
    """max |mine - ref| over the numeric leaves present in both; leaves present in one only; kind or value mismatches
    of non-numeric leaves (None = non-finite in both files)"""
    A, B = dict(leaves(mine)), dict(leaves(ref))
    mx, at, n, other = 0.0, None, 0, []
    for k, v in A.items():
        if k not in B:
            continue
        u = B[k]
        if num(v) and num(u):
            n += 1
            d = abs(float(v) - float(u))
            if at is None or d > mx:
                mx, at = d, k
        elif v != u:
            other.append(k)
    only_m, only_r = [k for k in A if k not in B], [k for k in B if k not in A]
    return {"max_abs_diff": mx, "at": at, "n_numeric_compared": n, "n_other_mismatch": len(other), "other_first": other[:5],
            "n_only_in_mine": len(only_m), "only_in_mine_first": only_m[:5],
            "n_only_in_reference": len(only_r), "only_in_reference_first": only_r[:5]}


def maxabs(x, pick=lambda k: True):
    vals = [abs(float(v)) for k, v in leaves(x) if num(v) and pick(k)]
    return (max(vals) if vals else None), len(vals)


def null_check(Rj):
    """F1 against itself: every D, every guard ratio, every Q6 ratio must be 0; every arm statistic equal to the stored"""
    out = {}
    out["max_abs_D_Q1_Q4"], out["n_D"] = maxabs({tk: {k: v["D"] for k, v in Rj["Q"][tk].items()} for tk in TK})
    out["max_abs_guard_ratio"], out["n_guard"] = maxabs(Rj["G"], lambda k: "ratio_m1" in k)
    out["max_abs_Q6_ratio"], out["n_Q6"] = maxabs(Rj["Q6"], lambda k: "ratio_m1" in k)
    st = {k[len("st/"):]: v for k, v in Rj["all_stats"].items() if k.startswith("st/")}
    arm = {k[len("arm/"):]: v for k, v in Rj["all_stats"].items() if k.startswith("arm/")}
    assert set(st) == set(arm)
    c = compare(arm, st)
    out["arm_vs_stored_all_stats"] = {"max_abs_diff": c["max_abs_diff"], "n_numeric_compared": c["n_numeric_compared"],
                                      "n_other_mismatch": c["n_other_mismatch"]}
    out["all_zero"] = bool(out["max_abs_D_Q1_Q4"] == 0 and out["max_abs_guard_ratio"] == 0 and out["max_abs_Q6_ratio"] == 0
                           and c["max_abs_diff"] == 0 and c["n_other_mismatch"] == 0)
    return out


# ======================================================================== Markdown table
def ci(d, sign=False, nd=3):
    if d is None or d.get("est") is None:
        return "-"
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    lo, hi = d.get("two_way", [None, None])
    s = f.format(d["est"])
    return s + (f" [{f.format(lo)}, {f.format(hi)}]" if lo is not None and hi is not None else "")


def markdown(Rj, a):
    inp = Rj["inputs"]
    L = [f"# A16: A15 statistics on {a.year}, arm against stored ({inp['n_runs']} starts x leads 1-7)", "",
         f"arm `{inp['arm']}`; stored `{inp['stored']}`; {inp['n_cells']} cells (complete in every forecast of both runs), "
         f"resident weights. Member 0 against member 0; the 4-member column is the mean of the statistic over the arm's four "
         f"members. Intervals: paired two-way bootstrap, 95 %, {NBOOT} draws, seed {SEED}, runs in moving blocks of "
         f"{a.block} consecutive starts x 10-degree tiles. D = |stored - truth| - |arm - truth| (Q3, Q4: distance to the target, "
         f"0 or 1 for the late-onset ratio); D > 0: the arm is closer. Reported; decides nothing.", "",
         "| statistic | threshold | truth | stored | arm | arm, 4-member mean | D |", "|---|---|---|---|---|---|---|"]
    rows = (("Q1_haz_within", "Q1 end hazard", "haz_within"), ("Q2_drop_within", "Q2 one-day drops > 40 %", "drop_within"),
            ("Q3_P1_err", "Q3 end-day error, first-day rule (days)", "P1_err"),
            ("Q3_P1_2day_err", "Q3 end-day error, 2-day rule (days)", None),
            ("Q4_E2_late_ratio", "Q4 late onsets / truth (target 1)", "E2_late_ratio"), ("Q4_P7b", "Q4 P7b (target 0)", "P7b"))
    four = Rj["reported"]["arm_four_member_means"]
    for tk in TK:
        tl = tk if tk == "P95" else f"{tk} ug m-3"
        for key, label, k4 in rows:
            q = Rj["Q"][tk][key]
            m4 = four.get(f"arm4/{tk}/{k4}") if (k4 and isinstance(four, dict)) else None
            sg = key.startswith(("Q3", "Q4_P7b"))
            L.append(f"| {label} | {tl} | {ci(q.get('truth'))} | {ci(q['stored'], sg)} | {ci(q['arm'], sg)} | "
                     f"{ci(m4, sg)} | {ci(q['D'], True)} |")
    q5 = Rj["Q5"]
    L += ["", "| Q5 (log, k = 1) | value |", "|---|---|",
          f"| arm, centered d_rho = rho_A - rho_B | {ci(q5['d_rho_arm_cen'], True)} |",
          f"| arm rho_A / rho_B, centered | {ci(q5['rho_A_arm_cen'])} / {ci(q5['rho_B_arm_cen'])} |",
          f"| truth, de-diluted, centered (arm o / stored o) | {ci(q5['rho_y_dediluted_arm_cen'])} / "
          f"{ci(q5['rho_y_dediluted_stored_cen'])} |",
          f"| stored member, uncentered d_rho (P6) | {ci(q5['stored_member_uncentered_d_rho_P6'], True)} |", "",
          f"Q6: RMSE (residents, ug m-3) of the time mean over the {inp['n_runs']} starts against the truth's time mean; "
          f"interval of arm/stored - 1 over runs (blocks of {a.block}).", "",
          "| lead | (a) member 0, stored | arm | arm/stored - 1 | (b) mean, stored | arm | arm/stored - 1 |",
          "|---|---|---|---|---|---|---|"]
    qa, qb = Rj["Q6"]["a_member0"], Rj["Q6"]["b_mean"]

    def pr(q, h):
        lo, hi = q["ratio_m1_runs_ci"][0][h], q["ratio_m1_runs_ci"][1][h]
        return f"{100 * q['ratio_m1'][h]:+.1f} % [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    for h in range(H):
        L.append(f"| {h + 1} | {qa['rmse_st'][h]:.2f} | {qa['rmse_arm'][h]:.2f} | {pr(qa, h)} | {qb['rmse_st'][h]:.2f} | "
                 f"{qb['rmse_arm'][h]:.2f} | {pr(qb, h)} |")
    ma, mb = qa["mean_over_leads_ratio_m1_runs_ci"], qb["mean_over_leads_ratio_m1_runs_ci"]
    L.append(f"| mean over leads | | | {100 * qa['mean_over_leads_ratio_m1']:+.1f} % [{100 * ma[0]:+.1f}, {100 * ma[1]:+.1f}] | | | "
             f"{100 * qb['mean_over_leads_ratio_m1']:+.1f} % [{100 * mb[0]:+.1f}, {100 * mb[1]:+.1f}] |")
    g = Rj["G"]
    L += ["", "Guard values (check C4 of A16 is judged by a16/check_runs.py, not here): fair CRPS arm/stored - 1 by lead "
          + ", ".join(f"{100 * g['per_lead'][str(h)]['crps_ratio_m1']:+.2f} %" for h in range(1, H + 1))
          + "; RMSE of the mean " + ", ".join(f"{100 * g['per_lead'][str(h)]['rmse_ratio_m1']:+.2f} %" for h in range(1, H + 1))
          + f"; mean over leads {100 * g['mean_over_leads']['crps_ratio_m1']:+.2f} % / {100 * g['mean_over_leads']['rmse_ratio_m1']:+.2f} %."]
    if "null_check" in Rj:
        L += ["", f"Null test (arm = stored): {json.dumps(Rj['null_check'])}"]
    return "\n".join(L) + "\n"


# ======================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True, choices=sorted(prep22.BASE))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--stored", required=True)
    ap.add_argument("--block", type=int, required=True, help="consecutive starts per bootstrap block (2022: 4; 2021: 1)")
    ap.add_argument("--prep", required=True, help="the JSON of prep22.py for the same year and folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--compare", default=None, help="regression: a15/results/score.json")
    a = ap.parse_args()
    for p in (a.out, a.md):
        if p:
            prep22.allowed(p)
    t0 = time.time()
    ar, a15s, revise = prep22.setup(a.year)
    pj = json.loads(Path(a.prep).read_text())
    I = prep22.build(a.year, a.arm, a.stored)
    for k in ("year", "arm", "stored", "starts", "n_cells", "cells_sha256", "null_test"):
        assert pj[k] == I["info"][k], (k, pj[k], I["info"][k])
    Ds, Xa, Ma, starts, null = I["Ds"], I["Xa"], I["Ma"], I["starts"], I["null"]
    cells, w = Ds["cells"], Ds["pop"].astype(np.float64)
    C, N = len(cells), len(starts)
    base = prep22.BASE[a.year]
    R = {"what": __doc__, "inputs": I["info"]}
    R["G"] = guard(I["recs_arm"], I["seams_arm"], I["recs_st"], I["seams_st"], I["cols"], starts)
    print(f"loaded {time.time() - t0:.0f}s: {C} cells, {N} runs, null test {null}", flush=True)

    # ---------------- pass 1 of each design (centering maps over the scored runs) and the paired aggregation
    Da = dict(Ds)
    Da["X"], Da["Mn"] = Xa[:, 0].copy(), Ma.copy()
    P1 = {}
    for nm, DD in (("st", Ds), ("arm", Da)):
        p = revise.pass1(DD)
        P1[nm] = revise.p1_summary(p, DD)
    maps = {nm: {"cenA_log": P1[nm][1]["A_log"], "cenB_log": P1[nm][1]["B_log"], "cenA_lin": P1[nm][1]["A_lin"],
                 "cenB_lin": P1[nm][1]["B_lin"]} for nm in P1}
    R["pass1"] = {nm: {k: P1[nm][0][k] for k in P1[nm][0] if k.startswith(("pass1_", "static_share"))} for nm in P1}
    del Da, P1
    tid = Ds["tid"]
    tstarts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    agg = revise.Agg(N, tstarts, w)
    nt = Ds["truth"].shape[0]
    gi0 = int(Ds["truth_gi0"])
    for r in range(N):
        t = int(Ds["tidx"][r])
        assert 0 <= gi0 + t - base <= 364 and gi0 + t - base == starts[r], (r, t)
        y0 = Ds["truth"][t].astype(np.float32)
        Y = Ds["Y"][r].astype(np.float32)
        Xs, Mns = Ds["X"][r].astype(np.float32), Ds["Mn"][r].astype(np.float32)
        P, W = Ds["P"][r].astype(np.float32), Ds["W"][r].astype(np.float32)
        after = []
        for k in (8, 9):
            if t + k < nt:
                assert gi0 + t + k - base <= 364
                after.append(Ds["truth"][t + k])
            else:
                after.append(np.full(C, np.nan, np.float32))
        Yext = np.vstack([Y, np.stack(after).astype(np.float32)])
        if null:
            A0 = Xa[r, 0]
            S_arm = {"y": Y, "x": A0, "m": Ma[r], "xs": A0, "xo": A0, "xe1": A0, "xer": A0, "xe0": A0}
        else:
            A = [Xa[r, m] for m in range(M)]
            S_arm = {"y": Y, "x": A[0], "m": Ma[r], "xs": A[1], "xo": A[2], "xe1": A[3], "xer": A[0], "xe0": A[0]}
        S = {"st": {"y": Y, "x": Xs, "m": Mns, "xs": Xs, "xo": Xs, "xe1": Xs, "xer": Xs, "xe0": Xs}, "arm": S_arm}
        for nm in S:
            assert tuple(S[nm]) == revise.SER
        for tk, T in revise.THR.items():
            Tc = Ds["p95"].astype(np.float32) if T is None else np.full(C, T, np.float32)
            f = np.isfinite(Tc)
            Tc = np.where(f, Tc, np.float32(np.inf))
            for nm in S:
                agg.add(r, f"{nm}|{tk}", {f"{nm}|{tk}/{k}": v for k, v in revise.thr_cols(y0, S[nm], P, W, Yext, Tc, f).items()})
        agg.add(r, "st|E5", {f"st|{k}": v for k, v in revise.e5_cols(Xs, Y, Mns, maps["st"], C).items()})
        agg.add(r, "arm|E5", {f"arm|{k}": v for k, v in revise.e5_cols(Xa[r, 0], Y, Ma[r], maps["arm"], C).items()})
        if r % 10 == 0:
            print(f"run {r + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    res, info = revise.summarise(agg, N, a.block, NBOOT, SEED, fn=lambda Dd: a15s.custom(Dd, revise))
    del agg
    R["bootstrap"] = {"draws": NBOOT, "seed": SEED, "run_block": a.block, "N": N, "tile_deg": 10, **info}
    print(f"summarised {time.time() - t0:.0f}s {info}", flush=True)
    g = lambda k: res[k]                                                 # noqa: E731

    # ---------------- primary statistics (A15 7.2)
    Q = {}
    for tk in TK:
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
    Kr = revise.boot_counts(N, a.block, NBOOT, SEED)
    Ys = Ds["Y"].astype(np.float32)
    R["Q6"] = {"a_member0": a15s.q6(Ds["X"].astype(np.float32), Xa[:, 0], Ys, w, Kr),
               "b_mean": a15s.q6(Ds["Mn"].astype(np.float32), Ma, Ys, w, Kr)}
    print(f"Q6 {time.time() - t0:.0f}s", flush=True)

    # ---------------- reported (A15 6.3), never gating
    na = {"cal": "tau_Cal, f_hat, f_iso, rho* (A15 4.5): calibration on the 2021 Cal starts; rho* = 0.59 is an input of the "
                 "arm, recorded under inputs.arm_design.keys_rho",
          "smoke_gate": "A15 6.2 (a15/results/smoke_check.json); A16's smoke is check C2 (a16/smoke_check.py), not read here",
          "G_pass_flags": "A15 7.1 pass flags; on 2022 the guard is check C4 of A16, judged by a16/check_runs.py; values reported",
          "outcome": "A15 7.3 outcome classes, success_conditions, fail_conditions, reasons: A16 4.5 'reported and never gating'",
          "cal_values_6.3": "A15 6.3 'the Cal values': no Cal / Conf split outside A15's 2021 design"}
    rep = {}
    rep["per_pair_lag1"] = {"arm": a15s.lag_pairs(Xa[:, 0], Ma, Ys, w, members=None if null else np.ascontiguousarray(Xa)),
                            "stored": a15s.lag_pairs(Ds["X"].astype(np.float32), Ds["Mn"].astype(np.float32), Ys, w)}
    rep["acf_lags_2_6"] = {k: {"arm_rho_A_cen": g(f"arm/E5/log/cen/k{k}/rho_A")["est"],
                               "truth_dediluted_cen_arm_o": g(f"arm/E5/log/cen/k{k}/rho_y_dediluted")["est"],
                               "truth_dediluted_cen_stored_o": g(f"st/E5/log/cen/k{k}/rho_y_dediluted")["est"]}
                           for k in range(2, 7)}
    if a.year == 2021:
        for k in range(2, 7):
            rep["acf_lags_2_6"][k]["truth_45_runs_reference"] = a15s.TRUTH_ACF[k]
    else:
        na["truth_45_runs_reference"] = "the truth's ACF on the 45 runs of 2021 (A15 2.1); the truth's ACF on this year is reported"
    rep["spread_over_error_by_lead"] = {"arm_cen": g("arm/E5/log/cen/spread_over_error_by_lead"),
                                        "stored_raw": g("st/E5/log/raw/spread_over_error_by_lead"),
                                        "level_ratio_A2_over_B2_arm_cen": g("arm/E5/log/cen/level_ratio_A2_over_B2"),
                                        "level_ratio_A2_over_B2_stored_raw": g("st/E5/log/raw/level_ratio_A2_over_B2")}
    rep["hazard_by_weather_class"] = {f"{nm}/{tk}/{k}": g(f"{nm}/{tk}/{k}") for nm in ("st", "arm") for tk in ("37.5", "P95")
                                      for k in ("hazc_first_x", "hazc_within_x", "hazc_first_y", "hazc_within_y",
                                                "hazc_first_x_over_y", "hazc_within_x_over_y")}
    if null:
        rep["arm_four_member_means"] = "not applicable: the arm is the stored run (no map25_mk)"
        na["arm_four_member_means"] = rep["arm_four_member_means"]
    else:
        rep["arm_four_member_means"] = {k: v for k, v in res.items() if k.startswith("arm4/")}
    if a.year == 2021:
        emu = json.loads((prep22.ROOT / "episodes_diag" / "results" / "revision_S1.json").read_text())["stats"]
        rep["emulation_45_runs_revision_S1"] = {k: emu.get(f"37.5/{k}") for k in
                                                ("haz_within_xer", "haz_within_x", "haz_within_y", "drop_within_xer",
                                                 "drop_within_x", "drop_within_y", "P1_err_xer", "P1_err_x", "E2_late_ratio_xer",
                                                 "E2_late_ratio_x", "P7b_xer", "P7b_x")}
    else:
        na["emulation_45_runs_revision_S1"] = "the statistical emulation on the 45 runs of 2021 (revision_S1.json); not a quantity of this year"
    R["reported"] = rep
    R["not_applicable"] = na
    R.update(n_cells=C, residents=float(w.sum()), parts=I["info"]["parts"]["arm"], parts_stored=I["info"]["parts"]["stored"],
             rho_arm=I["info"]["arm_design"]["keys_rho"])
    R["all_stats"] = res
    Rj = json.loads(json.dumps(ar.jsonable(R)))
    if null:
        Rj["null_check"] = null_check(Rj)
        print(f"null check {Rj['null_check']}", flush=True)
    if a.compare:
        ref = json.loads(Path(a.compare).read_text())
        Rj["regression_vs_a15"] = {"reference": a.compare,
                                   "Q1_Q6": compare({k: Rj[k] for k in ("Q", "Q5", "Q6")}, {k: ref[k] for k in ("Q", "Q5", "Q6")}),
                                   **{k: compare(Rj[k], ref[k]) for k in ("G", "pass1", "bootstrap", "reported", "all_stats")},
                                   "n_cells_equal": Rj["n_cells"] == ref["n_cells"], "residents_diff": Rj["residents"] - ref["residents"]}
        print(f"regression vs {a.compare}: Q1-Q6 {Rj['regression_vs_a15']['Q1_Q6']}", flush=True)
    Rj["seconds"] = round(time.time() - t0, 1)
    Rj["max_rss_gb"] = round(prep22.max_rss_gb(), 2)
    prep22.write_text(a.out, json.dumps(Rj, indent=1) + "\n")
    if a.md:
        prep22.write_text(a.md, markdown(Rj, a))
    print(f"score22 {a.year} done {Rj['seconds']:.0f}s, max RSS {Rj['max_rss_gb']:.1f} GB -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
