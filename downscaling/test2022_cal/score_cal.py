#!/usr/bin/env python3
"""Scoring of the 2022 pass of the CALIBRATED downscaler (runs_ghap/pmG_cal/final; PROTOCOL.md A6 + draft amendment
protocol_amendments/A9_calibrated_2022.md) with the scorers of the frozen system's 2022 test, UNMODIFIED:
test2022_ghap/score_2022.py (GHAP subsample: fair CRPS, RMSE and bias of the ensemble mean, 1 deg block RMSE, Fortin
spread/skill, cos-lat weights, 53 week blocks, 5,000 draws) and test2022_ghap/score_st_2022.py (monitors, monitor pixel and
1/3/10 km rings). This wrapper only points them at test2022_cal/ (module attributes; no line of either script is changed) and
adds two tables that need both passes. World venv, numpy only; no model code.

  score    scores_ghap_2022.{json,md} + accum_2022.npz from test2022_cal/preds against the GHAP 2022 truth extracted for the
           frozen system's test (data/processed/ghap/sealed/ghap_sub_2022.npy: read, not re-extracted; logged in
           logs/UNSEALING.md). Refuses to run twice.
  split    preds_st -> preds_st_mon/<arm>, preds_ring{1,3,10}/<arm> (+ the reference arm ghap), as A2(e)
  stscore  --where mon|ring1|ring3|ring10: score_st_2022.main with the --preds / --out and the environment of
           test2022_ghap/score.sh -> results/st_monitor/scores_st_2022.{json,md}, results/st_ring<d>km/...
  stpair   extra: calibrated - frozen at the monitor pixel, paired on identical pixel-days (score_st_2022.contrast with the
           pair (cdffm, cdffm_frozen)) -> results/st_monitor_pair/scores_st_2022_cal_vs_frozen.{json,md}
  compare  extra: calibrated - frozen against GHAP 2022 from the two accum_2022.npz (paired week-block bootstrap of the
           downscaler's CRPS, RMSE, bias, block RMSE, spread/skill by region; the baseline arms must be the same in both
           passes) + the monitor rows of both passes -> results/compare_cal_vs_frozen_2022.{json,md}
Rehearsal (selection-year data only): --mech DIR --system frozen|cal scores DIR/<system> against DIR/truth.npy (2021 days
under the scorers' day names) and the pseudo-observations of DIR (mech_setup.py); the unsealing log is DIR/UNSEALING_test.md.
  python -I test2022_cal/score_cal.py score|split|stscore|stpair|compare [--procs 32] [--where mon] [--mech DIR --system cal]"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
OC = ROOT / "test2022_cal"
OF = ROOT / "test2022_ghap"
sys.path.insert(0, str(OF))
import score_2022 as M  # noqa: E402  (the frozen system's scorer; its command line block does not run on import)

S = M.S
WHERE = {"mon": ("preds_st_mon", "st_monitor", "monitor pixel", ""),
         **{f"ring{d}": (f"preds_ring{d}", f"st_ring{d}km",
                         f"ring of {d} km around the monitor pixel (Chebyshev distance {d}, monitor pixel excluded, A2(e))", f"_ring{d}km")
            for d in (1, 3, 10)}}
LABEL = "GHAP system pmG_cal = pmG_final with the calibrated sampler, PROTOCOL.md A6 / A9"


def setup(a):
    """point the unmodified scorers at the calibrated pass (or at a rehearsal directory); returns (O, frozen dir, mech)"""
    mech = Path(a.mech) if a.mech else None          # not resolved: the scorers print paths relative to ROOT (/scratch/...)
    assert mech is None or ROOT in mech.parents, f"--mech must be given as a path under {ROOT}"
    if mech is None:
        O, F, name = OC, OF, "test2022_cal"
        assert a.system in (None, "cal"), "the real pass scores the calibrated system only"
    else:
        assert a.system in ("frozen", "cal"), "--mech needs --system frozen|cal"
        O, F, name = mech / a.system, mech / "frozen", f"{mech.name}/{a.system}"
        K = len(json.loads((mech / "days.json").read_text())["days_2021"])
        M.TRUTH, M.TRUTH_ST, M.ND = mech / "truth.npy", mech / "truth_st.npy", K
        S.TRUTH, S.ND, S.UNSEAL = M.TRUTH, K, mech / "UNSEALING_test.md"
    M.O = O
    S.OUT, S.RES = O, O / "results"
    S.log_unseal = lambda text: M._log0(text.replace("final2024/preds", f"[ghap-cal] {name}/preds")
                                        .replace("final2024/results/scores_lghap_2024.{json,md}", f"[ghap-cal] {name}/results (renamed scores_ghap_2022)")
                                        .replace("[ghap] scores written: test2022_ghap/results", f"[ghap-cal] scores written: {name}/results"))
    return O, F, mech


def cmd_score(a):
    O, F, mech = setup(a)
    res = O / "results"
    assert mech is not None or not (res / "scores_ghap_2022.json").exists(), f"{res}/scores_ghap_2022.json exists: the pass is scored once"
    assert M.TRUTH.exists(), f"{M.TRUTH} missing"
    sysj = json.loads((O / "SYSTEM.json").read_text())
    if mech is None:
        assert (O / "GO_2022").exists(), "no GO_2022: the lead starts the pass"
        S.log_unseal(f"[ghap-cal] READING the GHAP 2022 truth {M.TRUTH.relative_to(ROOT)} (extracted on 2026-10-28 for the frozen system's "
                     f"A2(d) pass; not re-extracted) for the SINGLE 2022 pass of the calibrated system {sysj['system']} ({sysj['system_dir']}; "
                     f"PROTOCOL.md A6, amendment A9); predictions test2022_cal/preds written before this reading under the GhapStore firewall.")
    M.cmd_score(a)
    for ext in ("json", "md"):
        p = res / f"scores_ghap_2022.{ext}"
        p.write_text(p.read_text().replace("(GHAP system pmG_final)", f"({LABEL})" if a.system != "frozen" else "(GHAP system pmG_final, rehearsal)")
                     .replace("PROTOCOL.md A2(d) single 2022 pass", "protocol of PROTOCOL.md A2(d), single 2022 pass of the calibrated system (A9)"))


def cmd_split(a):
    O, F, mech = setup(a)
    assert (O / "st_ring_2022.npz").exists(), f"{O}/st_ring_2022.npz missing (link to the positions file the predictions used)"
    if mech is None and M.TRUTH_ST.exists():
        S.log_unseal(f"[ghap-cal] reading {M.TRUTH_ST.relative_to(ROOT)} (GHAP 2022 at the monitor and ring pixels, extracted for the frozen "
                     f"system's pass) to write the reference arm 'ghap' next to the calibrated system's station predictions (A2(e), A9).")
    M.cmd_split(a)


def station_scorer(mech):
    import score_st_2022 as T  # noqa: E402  (the frozen system's station scorer)
    if mech is not None:                     # rehearsal: pseudo-observations, a dozen days
        T.C = mech
        T.CFG[2022].update(min_days_lead=5, min_days_all=5)
    return T


def cmd_stscore(a):
    O, F, mech = setup(a)
    sub, out, where, tag = WHERE[a.where]
    T = station_scorer(mech)
    os.environ["GHAP_ST_WHERE"] = where
    os.environ["GHAP_ST_TAG"] = tag
    if mech is None:
        S.log_unseal(f"[ghap-cal] station score 2022 of the calibrated system ({a.where}): {T.C.relative_to(ROOT)}/st_truth_2022.npz read by "
                     f"test2022_ghap/score_st_2022.py on test2022_cal/{sub} (evaluation only; A2(d)-2, A2(e), A9).")
    sys.argv = ["score_st_2022.py", "2022", "--preds", str(O / sub), "--out", str(O / "results" / out)]
    T.main()
    p = O / "results" / out / f"scores_st_2022{tag}.md"
    p.write_text(p.read_text().replace("(GHAP system pmG_final)", f"({LABEL}; run through test2022_cal/score_cal.py)" if a.system != "frozen"
                                       else "(GHAP system pmG_final, rehearsal)"))


def cmd_stpair(a):
    O, F, mech = setup(a)
    T = station_scorer(mech)
    pd_ = O / "pair_st_mon"
    pd_.mkdir(parents=True, exist_ok=True)
    for name, src in (("cdffm", O / "preds_st_mon" / "cdffm"), ("cdffm_frozen", F / "preds_st_mon" / "cdffm")):
        assert src.is_dir(), src
        ln = pd_ / name
        if ln.is_symlink():
            ln.unlink()
        os.symlink(src, ln)
    pair = [("cdffm", "cdffm_frozen")]
    T.CFG[2022].update(pairs=pair, key_pairs=pair, fig_pairs=pair)
    T.LABELS.update(cdffm="calibrated sampler (pmG_cal)", cdffm_frozen="frozen system (pmG_final)")
    T.figure = lambda *x, **k: None
    os.environ["GHAP_ST_WHERE"] = "monitor pixel; calibrated - frozen, paired on identical pixel-days"
    os.environ["GHAP_ST_TAG"] = "_cal_vs_frozen"
    sys.argv = ["score_st_2022.py", "2022", "--preds", str(pd_), "--out", str(O / "results" / "st_monitor_pair")]
    T.main()


def cmd_compare(a):
    O, F, mech = setup(a)
    zc, zf = np.load(O / "results/accum_2022.npz"), np.load(F / "results/accum_2022.npz")
    Dc, Df = zc["D"], zf["D"]
    arms, regions = [str(x) for x in zc["arms"]], [str(x) for x in zc["regions"]]
    assert Dc.shape == Df.shape and arms == [str(x) for x in zf["arms"]] and regions == [str(x) for x in zf["regions"]], "the two passes differ in layout"
    Mc = json.loads((O / "results/scores_ghap_2022.json").read_text())["members"]["cdffm"]
    Mf = json.loads((F / "results/scores_ghap_2022.json").read_text())["members"]["cdffm"]
    ci = arms.index("cdffm")
    nd = Dc.shape[0]
    # same truth, same pixels, same weights: the weight sums of the downscaler's arm are those of the frozen pass
    same_w = float(np.abs(Dc[:, ci, :, 0] - Df[:, ci, :, 0]).max())
    base = {arm: float(np.abs(Dc[:, ai] - Df[:, ai]).max() / max(float(np.abs(Df[:, ai]).max()), 1e-30)) for ai, arm in enumerate(arms) if arm != "cdffm"}
    wk = {}
    for t in range(nd):
        wk.setdefault(t // 7, []).append(t)
    B = S.boot(np.stack([Dc[:, ci], Df[:, ci]], 1), [np.array(v) for v in wk.values()])
    pc, pf = S.metrics(Dc[:, ci].sum(0), Mc), S.metrics(Df[:, ci].sum(0), Mf)
    bc, bf = S.metrics(B[:, 0], Mc), S.metrics(B[:, 1], Mf)
    res = {"calibrated": str(O), "frozen": str(F), "n_days": nd, "n_week_blocks": len(wk), "bootstrap": "paired, week blocks (t // 7), 5,000 draws, seed 0, percentile 95 %",
           "max_abs_diff_weight_sums": same_w, "baseline_arms_max_rel_diff_of_daily_sums": base, "members": {"calibrated": Mc, "frozen": Mf}, "regions": {}}
    # The 2022 scorer (runs_global/score_2024.py, X.var(0)) takes the member variance with divisor M; the selection evaluation
    # (cdffm_aq.data_aq.Scores, X.var(0, unbiased=True)) takes the unbiased one, divisor M - 1, as Fortin et al. (2014) do.
    # The two spread/skill ratios differ by sqrt(M / (M - 1)) = 1.069 for M = 8; both are written, the scorer's unchanged.
    for m_, mx in ((pc, Mc), (pf, Mf), (bc, Mc), (bf, Mf)):
        m_["ssF_u"] = m_["ssF"] * np.sqrt(mx / (mx - 1.0))
    res["spread_skill_definitions"] = {"ssF": "sqrt((M + 1) / M x member variance with divisor M) / RMSE: the number the 2022 scorer writes (scores_ghap_2022.md)",
                                       "ssF_u": "sqrt((M + 1) / M x unbiased member variance, divisor M - 1) / RMSE: Fortin et al. (2014), the definition of the "
                                                "selection evaluation (eval_x) in which the band 0.9-1.1 of A6(g) was stated; = ssF x sqrt(M / (M - 1))"}
    names = {"crps": "fair CRPS", "rmse": "RMSE of the ensemble mean", "block_rmse": "1° block RMSE", "bias": "bias",
             "ssF": "spread/skill as the 2022 scorer writes it (member variance with divisor M)",
             "ssF_u": "spread/skill with the unbiased member variance (divisor M - 1; Fortin et al. 2014; the selection evaluation's definition)"}
    L = ["# Calibrated (pmG_cal) - frozen (pmG_final) downscaler against GHAP V1: paired differences", "",
         f"Calibrated pass: {O}; frozen pass: {F}. Same subsample (every 8th mask pixel), same truth, same weights (largest difference of a "
         f"daily weight sum {same_w:.3g}); {nd} days, {len(wk)} week blocks; {res['bootstrap']}. Baseline arms recomputed in the calibrated pass: "
         "largest relative difference of a daily sum to the frozen pass " + ", ".join(f"{k} {v:.2g}" for k, v in base.items()) + ".", ""]
    for k, nm in names.items():
        L += [f"## {nm}", "", "| region | frozen | calibrated | calibrated - frozen [95 %] | relative [95 %] |", "|---|---|---|---|---|"]
        for ri, r in enumerate(regions):
            d = bc[k][:, ri] - bf[k][:, ri]
            rel = bc[k][:, ri] / bf[k][:, ri] - 1.0
            row = {"frozen": float(pf[k][ri]), "calibrated": float(pc[k][ri]), "diff": float(pc[k][ri] - pf[k][ri]),
                   "diff_ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                   "ci95_frozen": [float(np.nanpercentile(bf[k][:, ri], 2.5)), float(np.nanpercentile(bf[k][:, ri], 97.5))],
                   "ci95_calibrated": [float(np.nanpercentile(bc[k][:, ri], 2.5)), float(np.nanpercentile(bc[k][:, ri], 97.5))]}
            if k != "bias":
                row["rel"] = float(pc[k][ri] / pf[k][ri] - 1.0)
                row["rel_ci95"] = [float(np.nanpercentile(rel, 2.5)), float(np.nanpercentile(rel, 97.5))]
            res["regions"].setdefault(r, {})[k] = row
            relt = "" if k == "bias" else f"{100 * row['rel']:+.2f} % [{100 * row['rel_ci95'][0]:+.2f}, {100 * row['rel_ci95'][1]:+.2f}]"
            L.append(f"| {r} | {row['frozen']:.4f} | {row['calibrated']:.4f} | {row['diff']:+.4f} [{row['diff_ci95'][0]:+.4f}, {row['diff_ci95'][1]:+.4f}] | {relt} |")
        L.append("")
    st = {}
    keys = ("n_pixel_days", "rmse", "rmse_ci", "crps", "crps_ci", "bias", "r", "median_station_r")
    for key, (sub, out, where, tag) in WHERE.items():
        fc, ff = O / "results" / out / f"scores_st_2022{tag}.json", F / "results" / out / f"scores_st_2022{tag}.json"
        if fc.exists() and ff.exists():
            rc_, rf_ = (json.loads(f.read_text())["arms"]["cdffm"]["leads"]["all"].get("All/Global") for f in (fc, ff))
            if rc_ and rf_:
                st[key] = {"calibrated": {k: rc_.get(k) for k in keys}, "frozen": {k: rf_.get(k) for k in keys}}
    if st:
        res["monitors_all_global"] = st
        L += ["## At the monitors (All networks, Global; each pass's own table, intervals over calendar weeks)", "",
              "| where | pixel-days | RMSE frozen | RMSE calibrated | CRPS frozen | CRPS calibrated | bias frozen / calibrated | R frozen / calibrated |", "|---|---|---|---|---|---|---|---|"]
        f2 = lambda r, k: f"{r[k]:.3f} [{r[k + '_ci'][0]:.3f}, {r[k + '_ci'][1]:.3f}]"
        for key, v in st.items():
            c_, f_ = v["calibrated"], v["frozen"]
            L.append(f"| {key} | {c_['n_pixel_days']} | {f2(f_, 'rmse')} | {f2(c_, 'rmse')} | {f2(f_, 'crps')} | {f2(c_, 'crps')} | {f_['bias']:+.3f} / {c_['bias']:+.3f} | {f_['r']:.3f} / {c_['r']:.3f} |")
        L.append("")
    pj = O / "results/st_monitor_pair/scores_st_2022_cal_vs_frozen.json"
    if pj.exists():
        c = next((c for c in json.loads(pj.read_text())["contrasts"] if c["A"] == "cdffm" and c["B"] == "cdffm_frozen" and c["lead"] == "all"), None)
        if c:
            res["monitors_paired"] = c["groups"]
            L += ["## At the monitor pixel, paired (calibrated - frozen, identical pixel-days; All networks by region)", "",
                  "| group | pixel-days | RMSE frozen | RMSE calibrated | dRMSE % [95 %] | dCRPS [95 %] | dCRPS % [95 %] |", "|---|---|---|---|---|---|---|"]
            for g, r in c["groups"].items():
                if r and g.startswith("All/"):
                    L.append(f"| {g} | {r['n_pixel_days']} | {r['rmse_B']:.3f} | {r['rmse_A']:.3f} | {100 * r['rel_rmse']:+.2f} [{100 * r['rel_rmse_ci'][0]:+.2f}, {100 * r['rel_rmse_ci'][1]:+.2f}] | "
                             f"{r['d_crps']:+.3f} [{r['d_crps_ci'][0]:+.3f}, {r['d_crps_ci'][1]:+.3f}] | {100 * r['rel_crps']:+.2f} [{100 * r['rel_crps_ci'][0]:+.2f}, {100 * r['rel_crps_ci'][1]:+.2f}] |")
            L.append("")
    (O / "results/compare_cal_vs_frozen_2022.json").write_text(json.dumps(res, indent=1))
    (O / "results/compare_cal_vs_frozen_2022.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "split", "stscore", "stpair", "compare"])
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--where", default="mon", choices=list(WHERE))
    ap.add_argument("--mech", default=None, help="rehearsal directory (selection-year data, pseudo-monitors)")
    ap.add_argument("--system", default=None, choices=[None, "frozen", "cal"])
    a = ap.parse_args()
    {"score": cmd_score, "split": cmd_split, "stscore": cmd_stscore, "stpair": cmd_stpair, "compare": cmd_compare}[a.cmd](a)
