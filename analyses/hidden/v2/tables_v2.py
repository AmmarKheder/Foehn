#!/usr/bin/env python3
"""Numbers of version 2 of 'who is warned'  ->  results/numbers_v2.json, results/tables_v2.txt

The machinery of ../tables.py is imported unchanged (parts, metrics, Tab, mbb, tri, f): the same pooled sums and the same
moving-block bootstrap over target days (blocks of 4 consecutive forecasts = 7 days, 10,000 resamples, seed 0, 95 %). The
systems of version 1 therefore get exactly the values and intervals of ../results/numbers.json (asserted below), and the
differences between old and new systems are paired (the same resampled days).

Inputs (nothing is recomputed from the fields here)
  ../data/hidden_systems_pop.npz   job 23864988, ../figdata_hidden_systems.py: foehn_p50 (at least 2 of 4 members), foehn_mean,
                                   foehn_40km, cams, aurora, perfect_40km, decision-threshold sweeps of foehn_mean / cams / aurora
  data/hidden_bc_pop.npz           figdata_hidden_bc.py: cams_qm (CAMS forecast, bias-corrected: quantile map of 2017-2020 per
                                   0.25 deg cell and month, cell value), cams_bcsd (the same with the climatological 1 km
                                   pattern), clim_1km (the 1 km climatology alone), sweeps of cams_qm / cams_bcsd, error sums
Added to version 1: the two bias-corrected references in every table; Foehn issued 3 days ahead against the references
issued 1 day ahead on the common target days (paired); people wrongly warned where the 40 km mean is below the threshold as
a number and per hidden person warned; the comparisons at an equal share of correct warnings and at an equal number of
people warned for the new references.
    python tables_v2.py"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tables import DIFFM, Tab, f, mbb, tri  # noqa: E402  (../tables.py, unchanged)

SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
RES = HERE / f"results{SFX}"
OLDF, NEWF = HERE.parent / f"data{SFX}" / "hidden_systems_pop.npz", HERE / f"data{SFX}" / "hidden_bc_pop.npz"
SRC = {"old": "../data/hidden_systems_pop.npz (job 23864988, ../figdata_hidden_systems.py)",
       "new": "data/hidden_bc_pop.npz (v2/figdata_hidden_bc.py)"}
NAMES = {"foehn_p50": "Foehn, 2 of 4 members", "foehn_mean": "Foehn, ensemble mean", "foehn_40km": "Foehn mean on 40 km cells",
         "cams": "CAMS forecast", "aurora": "Aurora", "cams_qm": "CAMS forecast, bias-corrected",
         "cams_bcsd": "CAMS forecast, bias-corrected + 1 km pattern", "clim_1km": "1 km climatology alone",
         "perfect_40km": "perfect 40 km forecast"}
BASE = list(NAMES)
REFS = ("cams", "aurora", "cams_qm", "cams_bcsd", "foehn_40km", "clim_1km")
PAIRS = [("foehn_p50", r) for r in REFS] + [("foehn_mean", "foehn_40km"), ("foehn_mean", "cams_bcsd"), ("foehn_mean", "cams_qm"),
                                              ("cams_qm", "cams"), ("cams_bcsd", "cams"), ("cams_bcsd", "cams_qm"), ("foehn_40km", "cams_bcsd")]
DIFF2 = tuple(DIFFM) + ("warned_in_silent40_M", "false_per_hidden_warned")
RATIO = ("right_M", "false_M", "hidden_warned_M", "warned_M", "false_in_silent40_M", "hidden_margin_warned_M")
CROSS = ("reach_pct", "correct_pct", "right_M", "false_M", "warned_M", "hidden_warned_M", "hidden_warned_pct", "hidden_margin_warned_pct", "csi")
LINES = []


def out(s=""):
    print(s); LINES.append(s)


def main():
    zo, zn = np.load(OLDF), np.load(NEWF)
    K = zo["keys"]
    assert np.array_equal(K, zn["keys"]), "the two files do not hold the same forecasts in the same order"
    reg, thr = [str(r) for r in zo["regions"]], [float(t) for t in zo["thr"]]
    assert reg[0] == "World"
    assert reg == [str(r) for r in zn["regions"]] and thr == [float(t) for t in zn["thr"]] and float(zo["margin"]) == float(zn["margin"])
    so, sn = list(zo["sys1"]), list(zn["sys1"])
    To, Tn = zo["T1"], zn["T1"]
    mult = [float(m) for m in zo["mult"]]; assert mult == [float(m) for m in zn["mult"]]
    mus = sorted(mult + [1.0])
    new_sys = {s for s in sn if s.split("@")[0] in ("cams_qm", "cams_bcsd", "clim_1km")}
    # identity of the systems computed twice (same pixels, days, valid residents)
    ident = {s: bool(np.array_equal(Tn[:, :, :, sn.index(s)], To[:, :, :, so.index(s)])) for s in ("foehn_p50", "foehn_mean", "foehn_40km", "cams")}
    assert all(ident.values()), ident
    LEADS = sorted(int(h) for h in set(K[:, 1]))
    SEL = {h: np.flatnonzero(K[:, 1] == h) for h in LEADS}
    for h in LEADS:
        assert (np.diff(K[SEL[h], 0]) == 2).all()

    def arr(s):
        return (Tn[:, :, :, sn.index(s)], "new") if s in new_sys else (To[:, :, :, so.index(s)], "old")

    cache = {}

    def T(h, t, r, s):
        key = (h, t, r, s)
        if key not in cache:
            a, _ = arr(s)
            x = Tab(a[SEL[h], reg.index(r), thr.index(t)], mbb(len(SEL[h]), 4))
            if r != "World":
                return x
            cache[key] = x
        return cache[key]

    def swept(h, t, s):
        return [T(h, t, "World", s if mu == 1.0 else f"{s}@{mu:g}") for mu in mus]

    N = {"what": "version 2 of the numbers of 'who is warned': the systems of version 1 and two bias-corrected CAMS references",
         "population": "residents of the 8.1 M 1 km grid cells with at least 100 residents (7,388 M, 94 % of the world's residents); "
                       "358 retrospective forecasts of 2022 (179 per lead time, start days every second day)",
         "interval": "[value, lower, upper]: 95 % interval of a moving-block bootstrap over target days, blocks of 4 forecasts (7 days), "
                     "10,000 resamples, seed 0 (../tables.py: mbb, Tab), paired for differences",
         "sources": {s: (SRC["new"] if s in new_sys else SRC["old"]) + f", system '{s}'" for s in BASE},
         "definitions": {"warned": "the forecast read at the resident's 1 km grid cell is above the threshold (Foehn: at least 2 of its 4 members)",
                         "exposed": "the 1 km estimate of the target day is above the threshold",
                         "hidden": "exposed while the mean of the estimates over the 40 km cell is not above the threshold",
                         "hidden with a margin": "grid cell more than 10 % above the threshold and 40 km mean more than 10 % below it",
                         "reach_pct": "exposed and warned / exposed", "correct_pct": "exposed and warned / warned",
                         "warned_in_silent40_M": "people warned where the 40 km mean is not above the threshold",
                         "false_in_silent40_M": "of whom not exposed (wrongly warned)",
                         "false_per_hidden_warned": "false_in_silent40 / hidden people warned"},
         "identity_of_recomputed_systems": ident}
    S = N["systems"] = {}
    old = json.loads((HERE.parent / f"results{SFX}" / "numbers.json").read_text())["systems"]
    worst = 0.0
    for t in (37.5, 15.0, 75.0, 25.0, 50.0):
        for h in LEADS:
            b = T(h, t, "World", "foehn_p50")
            out(f"\n=== World, > {t:g} ug m-3, {h} day{'s' if h > 1 else ''} ahead, {b.n} target days. People a day (millions): exposed "
                f"{f(b.ci('exposed_M'))}, hidden {f(b.ci('hidden_M'))}, hidden with a 10 % margin {f(b.ci('hidden_margin_M'))}")
            out(f"  {'system':44s} {'warned':>7s} {'rightly M':>20s} {'wrongly M':>18s} {'reach %':>18s} {'correct %':>18s} {'hidden warned M':>18s} "
                f"{'of hidden %':>18s} {'margin-hidden %':>18s} | 40 km mean below: {'warned M':>8s} {'wrongly M':>18s} {'per hidden warned':>18s}")
            W = S.setdefault(f"thr{t:g}", {}).setdefault(f"lead{h}", {}).setdefault("World", {})
            for s in BASE:
                x = T(h, t, "World", s); a = x.all(); a["source"] = N["sources"][s]; W[s] = a
                out(f"  {NAMES[s]:44s} {a['warned_M'][0]:7.0f} {f(a['right_M'], 0):>20s} {f(a['false_M'], 0):>18s} {f(a['reach_pct']):>18s} {f(a['correct_pct']):>18s} "
                    f"{f(a['hidden_warned_M']):>18s} {f(a['hidden_warned_pct']):>18s} {f(a['hidden_margin_warned_pct']):>18s} | {'':17s} "
                    f"{a['warned_in_silent40_M'][0]:8.0f} {f(a['false_in_silent40_M'], 0):>18s} {f(a['false_per_hidden_warned'], 2):>18s}")
                if s in old[f"thr{t:g}"][f"lead{h}"]["World"]:                # version 1 values reproduced
                    o = old[f"thr{t:g}"][f"lead{h}"]["World"][s]
                    worst = max([worst] + [abs(a[k][j] - o[k][j]) for k in o for j in range(3) if o[k][j] is not None and np.isfinite(o[k][j])])
            out("  paired differences (first minus second):")
            D = W["differences"] = {}
            for a_, b_ in PAIRS:
                xa, xb = T(h, t, "World", a_), T(h, t, "World", b_)
                d = D[f"{a_}-{b_}"] = xa.diffs(xb, DIFF2)
                D[f"{a_}/{b_}"] = xa.ratios(xb, RATIO)
                out(f"    {a_:10s} - {b_:10s}: rightly {f(d['right_M'])} M | wrongly {f(d['false_M'])} M | reach {f(d['reach_pct'])} pts | correct "
                    f"{f(d['correct_pct'])} pts | hidden {f(d['hidden_warned_M'])} M, {f(d['hidden_warned_pct'])} pts | margin-hidden "
                    f"{f(d['hidden_margin_warned_pct'])} pts | wrongly where the 40 km mean is below {f(d['false_in_silent40_M'])} M | csi {f(d['csi'], 3)}")
            # ---- the references with their decision threshold moved (as ../tables.py, 'equal_correct_share' and 'equal_number_warned')
            ref = T(h, t, "World", "foehn_p50"); refbs = ref.bs

            def interp(target, w, v):                                            # linear interpolation, w increasing along axis 0
                if w.ndim == 1:
                    return np.interp(target, w, v)
                i = np.clip((w < target[None]).sum(0), 1, len(w) - 1); j = np.arange(w.shape[1])
                fr = (target - w[i - 1, j]) / (w[i, j] - w[i - 1, j])
                return v[i - 1, j] + fr * (v[i, j] - v[i - 1, j])
            QC = W["equal_correct_share"] = {}
            for moved, fixed in (("cams_qm", "foehn_p50"), ("cams_bcsd", "foehn_p50"), ("cams", "foehn_p50"), ("foehn_mean", "cams_qm"),
                                 ("foehn_mean", "cams_bcsd"), ("foehn_mean", "cams"), ("foehn_mean", "clim_1km")):
                tabs = swept(h, t, moved); fx_ = T(h, t, "World", fixed); fbs = fx_.bs; tbs = [x.bs for x in tabs]
                Cp = np.array([x.pt["correct_pct"] for x in tabs])
                if not (Cp.min() <= fx_.pt["correct_pct"] <= Cp.max()) or (np.diff(Cp) <= 0).any():
                    out(f"  {moved} at the share of correct warnings of {fixed}: outside the swept range ({Cp.min():.1f}-{Cp.max():.1f} % against {fx_.pt['correct_pct']:.1f} %)")
                    continue
                res = {"correct_pct": fx_.ci("correct_pct"), "decision_threshold_factor": float(np.interp(fx_.pt["correct_pct"], Cp, np.array(mus)))}
                for m in ("warned_M", "right_M", "false_M", "reach_pct", "hidden_warned_M", "hidden_warned_pct", "hidden_margin_warned_pct"):
                    pt = interp(fx_.pt["correct_pct"], Cp, np.array([x.pt[m] for x in tabs]))
                    bs = interp(fbs["correct_pct"], np.stack([x["correct_pct"] for x in tbs]), np.stack([x[m] for x in tbs]))
                    res[m] = tri(pt, bs); res[f"{fixed}_{m}"] = fx_.ci(m); res[f"moved_minus_fixed_{m}"] = tri(pt - fx_.pt[m], bs - fbs[m])
                    if m in ("right_M", "hidden_warned_M"):
                        res[f"moved_over_fixed_{m}"] = tri(pt / fx_.pt[m], bs / fbs[m])
                QC[f"{moved}_at_correct_share_of_{fixed}"] = res
                out(f"  {moved} at the share of correct warnings of {fixed} ({fx_.pt['correct_pct']:.1f} %; threshold x {res['decision_threshold_factor']:.2f}): "
                    f"warned {f(res['warned_M'], 0)} M, reach {f(res['reach_pct'])} % (fixed {fx_.pt['reach_pct']:.1f}), hidden {f(res['hidden_warned_M'])} M = "
                    f"{f(res['hidden_warned_pct'])} % (fixed {fx_.pt['hidden_warned_M']:.1f} M, {fx_.pt['hidden_warned_pct']:.1f} %); moved minus fixed: reach "
                    f"{f(res['moved_minus_fixed_reach_pct'])} pts, hidden {f(res['moved_minus_fixed_hidden_warned_M'])} M; ratio hidden {f(res['moved_over_fixed_hidden_warned_M'], 2)}")
            QN = W["equal_number_warned"] = {}
            for sw in ("cams_qm", "cams_bcsd", "cams", "aurora"):
                tabs = swept(h, t, sw)
                Wp = np.array([x.pt["warned_M"] for x in tabs])[::-1]             # increasing
                if not (Wp.min() <= ref.pt["warned_M"] <= Wp.max()):
                    out(f"  {sw} at Foehn's number of people warned: outside the swept range"); continue
                tbs = [x.bs for x in tabs][::-1]; res = {}
                for m in ("right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M", "hidden_warned_pct", "hidden_margin_warned_pct"):
                    pt = interp(ref.pt["warned_M"], Wp, np.array([x.pt[m] for x in tabs])[::-1])
                    bs = interp(refbs["warned_M"], np.stack([x["warned_M"] for x in tbs]), np.stack([x[m] for x in tbs]))
                    res[m] = tri(pt, bs); res[f"foehn_p50_minus_{m}"] = tri(ref.pt[m] - pt, refbs[m] - bs)
                res["decision_threshold_factor"] = float(np.interp(ref.pt["warned_M"], Wp, np.array(mus)[::-1]))
                QN[sw] = res
                out(f"  {sw} at Foehn's number of people warned ({ref.pt['warned_M']:.0f} M; its threshold x {res['decision_threshold_factor']:.2f}): reach "
                    f"{f(res['reach_pct'])} %, correct {f(res['correct_pct'])} %, hidden {f(res['hidden_warned_M'])} M ({f(res['hidden_warned_pct'])} %) | Foehn minus it: "
                    f"rightly {f(res['foehn_p50_minus_right_M'])} M, reach {f(res['foehn_p50_minus_reach_pct'])} pts, correct {f(res['foehn_p50_minus_correct_pct'])} pts, "
                    f"hidden {f(res['foehn_p50_minus_hidden_warned_M'])} M")
            # Foehn's ensemble mean with its decision threshold moved until it warns as many people as a reference that warns more or fewer
            QG = W["foehn_mean_at_the_number_warned_by"] = {}
            tabs = swept(h, t, "foehn_mean"); Wp = np.array([x.pt["warned_M"] for x in tabs])[::-1]; tbs = [x.bs for x in tabs][::-1]
            for fixed in ("clim_1km", "cams_bcsd", "cams_qm", "cams"):
                fx_ = T(h, t, "World", fixed); fbs = fx_.bs
                if not (Wp.min() <= fx_.pt["warned_M"] <= Wp.max()):
                    out(f"  foehn_mean at the number of people warned by {fixed}: outside the swept range"); continue
                res = {"warned_M": fx_.ci("warned_M"), "decision_threshold_factor": float(np.interp(fx_.pt["warned_M"], Wp, np.array(mus)[::-1]))}
                for m in ("right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M", "hidden_warned_pct", "hidden_margin_warned_pct"):
                    pt = interp(fx_.pt["warned_M"], Wp, np.array([x.pt[m] for x in tabs])[::-1])
                    bs = interp(fbs["warned_M"], np.stack([x["warned_M"] for x in tbs]), np.stack([x[m] for x in tbs]))
                    res[m] = tri(pt, bs); res[f"{fixed}_{m}"] = fx_.ci(m); res[f"foehn_mean_minus_{fixed}_{m}"] = tri(pt - fx_.pt[m], bs - fbs[m])
                QG[fixed] = res
                out(f"  foehn_mean at the number of people warned by {fixed} ({fx_.pt['warned_M']:.0f} M; threshold x {res['decision_threshold_factor']:.2f}): reach "
                    f"{f(res['reach_pct'])} % (ref {fx_.pt['reach_pct']:.1f}), correct {f(res['correct_pct'])} % (ref {fx_.pt['correct_pct']:.1f}), hidden "
                    f"{f(res['hidden_warned_M'])} M (ref {fx_.pt['hidden_warned_M']:.1f}) | Foehn mean minus it: reach {f(res[f'foehn_mean_minus_{fixed}_reach_pct'])} pts, "
                    f"correct {f(res[f'foehn_mean_minus_{fixed}_correct_pct'])} pts, hidden {f(res[f'foehn_mean_minus_{fixed}_hidden_warned_M'])} M")
            W["decision_threshold_sweep"] = {sw: {f"{mu:g}": {m: x.pt[m] for m in ("warned_M", "right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M",
                                                                                 "hidden_warned_pct", "false_in_silent40_M")}
                                                  for mu, x in zip(mus, swept(h, t, sw))} for sw in ("foehn_mean", "cams", "aurora", "cams_qm", "cams_bcsd")}
    N["check_version1_reproduced"] = {"max_abs_deviation_from_../results/numbers.json": worst}
    assert worst < 1e-9, worst
    out(f"\n(version 1 reproduced: the values and intervals of the old systems equal ../results/numbers.json, worst deviation {worst:.1e})")
    # ---------------------------------------------------------------- Foehn issued 3 days ahead against the references issued 1 day ahead
    X = N["foehn_3_days_against_references_1_day"] = {
        "note": "common target days of the forecasts issued 3 days and 1 day ahead (paired bootstrap over these days, blocks of 4)"}
    t1, t3 = K[SEL[1], 2], K[SEL[3], 2]
    common = np.intersect1d(t1, t3); assert (np.diff(common) == 2).all()
    i1, i3 = SEL[1][np.searchsorted(t1, common)], SEL[3][np.searchsorted(t3, common)]
    assert np.array_equal(K[i1, 2], common) and np.array_equal(K[i3, 2], common)
    cnt = mbb(len(common), 4); X["n_common_target_days"] = int(len(common))
    for t in (15.0, 37.5, 75.0):
        k = thr.index(t)
        out(f"\n=== Foehn (2 of 4 members) issued 3 days ahead against the references issued 1 day ahead, World, > {t:g}, {len(common)} common target days")
        fo = Tab(arr("foehn_p50")[0][i3, 0, k], cnt)
        X[f"thr{t:g}"] = {"foehn_p50_lead3": fo.all()}
        for r_ in ("cams", "aurora", "cams_qm", "cams_bcsd", "foehn_40km"):
            rf = Tab(arr(r_)[0][i1, 0, k], cnt)
            d = fo.diffs(rf, CROSS)
            X[f"thr{t:g}"][f"{r_}_lead1"] = rf.all(); X[f"thr{t:g}"][f"foehn_p50_lead3-{r_}_lead1"] = d
            out(f"    Foehn 3 d: reach {fo.pt['reach_pct']:.1f} %, correct {fo.pt['correct_pct']:.1f} % | {r_:10s} 1 d: reach {rf.pt['reach_pct']:.1f} %, correct "
                f"{rf.pt['correct_pct']:.1f} % | difference: reach {f(d['reach_pct'])} pts, correct {f(d['correct_pct'])} pts, rightly {f(d['right_M'], 0)} M, wrongly "
                f"{f(d['false_M'], 0)} M, hidden {f(d['hidden_warned_M'])} M ({f(d['hidden_warned_pct'])} pts), csi {f(d['csi'], 3)}")
    # ---------------------------------------------------------------- regions (country groups, a partition of the world)
    R = N["regions"] = {"note": "country groups of ../regions.py (a partition of the world)"}
    for t in (37.5, 75.0, 15.0):
        for h in LEADS:
            out(f"\n=== Country groups, > {t:g}, {h} day{'s' if h > 1 else ''} ahead: reach % / correct % / hidden warned % for Foehn | CAMS forecast | bias-corrected | "
                f"bias-corrected + pattern; Foehn minus bias-corrected + pattern: rightly M, wrongly M, hidden M")
            for r in reg:
                if not r.startswith("G:"):
                    continue
                x = {s: T(h, t, r, s) for s in ("foehn_p50", "foehn_40km", "cams", "aurora", "cams_qm", "cams_bcsd")}
                if x["foehn_p50"].pt["exposed_M"] < 0.05:
                    continue
                e = R.setdefault(f"thr{t:g}", {}).setdefault(f"lead{h}", {})[r[2:]] = {s: v.all() for s, v in x.items()}
                d = e["foehn_p50-cams_bcsd"] = x["foehn_p50"].diffs(x["cams_bcsd"], DIFF2)
                e["foehn_p50-cams_qm"] = x["foehn_p50"].diffs(x["cams_qm"], DIFF2)
                out(f"  {r[2:]:32s} exposed {x['foehn_p50'].pt['exposed_M']:7.1f} hidden {x['foehn_p50'].pt['hidden_M']:6.1f} | "
                    + " | ".join(f"{x[s].pt['reach_pct']:5.1f} / {x[s].pt['correct_pct']:5.1f} / {x[s].pt['hidden_warned_pct']:5.1f}" for s in ("foehn_p50", "cams", "cams_qm", "cams_bcsd"))
                    + f" | {f(d['right_M'])}, {f(d['false_M'])}, {f(d['hidden_warned_M'])}")
    # ---------------------------------------------------------------- checks of the bias-corrected references
    C = N["checks_of_the_bias_corrected_references"] = {
        "map": "GH/ghap_clim/bcsd_G.npz (runs_ghap/fit_bcsd_ghap.py): per 0.25 deg cell and calendar month, 21 quantiles of log1p(CAMS day-0 daily mean, "
               "operational 00 UTC run) and of the cell mean of log1p(max(round(GHAP), 1)); " + "; ".join(str(v) for v in zn["fit"]),
        "coverage": {str(v).split("=")[0]: float(str(v).split("=")[1]) for v in zn["coverage"]},
        "max_deviation_of_the_vectorised_map_from_fit_bcsd_ghap_qmap": float(zn["diag"][:, 5].max())}
    dg = zn["diag"]
    out("\n=== Checks of the bias-corrected references")
    out(f"  map: {C['map']}")
    out("  coverage: " + ", ".join(f"{k_} {v:,.1f}" for k_, v in C["coverage"].items()))
    for h in LEADS:
        d = dg[SEL[h]]
        C[f"lead{h}"] = {"valid_residents_M": float(d[:, 0].mean() / 1e6), "without_cell_map_M": float(d[:, 1].mean() / 1e6),
                         "outside_mask_no_pattern_M": float(d[:, 2].mean() / 1e6), "cell_value_below_fitted_range_M": float(d[:, 3].mean() / 1e6),
                         "cell_value_above_fitted_range_M": float(d[:, 4].mean() / 1e6),
                         "cell_value_below_fitted_range_pct": float(100 * d[:, 3].sum() / d[:, 0].sum()),
                         "cell_value_above_fitted_range_pct": float(100 * d[:, 4].sum() / d[:, 0].sum()), "errors_population_weighted": {}}
        out(f"  lead {h}: valid residents {C[f'lead{h}']['valid_residents_M']:.1f} M; without a cell map {C[f'lead{h}']['without_cell_map_M']:.3f} M; outside the mask "
            f"{C[f'lead{h}']['outside_mask_no_pattern_M']:.3f} M; forecast value below / above the fitted range {C[f'lead{h}']['cell_value_below_fitted_range_pct']:.2f} / "
            f"{C[f'lead{h}']['cell_value_above_fitted_range_pct']:.2f} % of resident-days")
        cnt_h = mbb(len(SEL[h]), 4)
        for j, s in enumerate(zn["diag2_systems"]):
            q = zn["diag2"][SEL[h], j]; qs = q.sum(0); bq = cnt_h @ q
            e = C[f"lead{h}"]["errors_population_weighted"][str(s)] = {
                "mean_forecast": tri(qs[1] / qs[0], bq[:, 1] / bq[:, 0]), "mean_estimate": tri(qs[2] / qs[0], bq[:, 2] / bq[:, 0]),
                "bias": tri((qs[1] - qs[2]) / qs[0], (bq[:, 1] - bq[:, 2]) / bq[:, 0]), "rmse": tri(np.sqrt(qs[3] / qs[0]), np.sqrt(bq[:, 3] / bq[:, 0])),
                "mae": tri(qs[4] / qs[0], bq[:, 4] / bq[:, 0])}
            out(f"    {str(s):10s} population-weighted mean {e['mean_forecast'][0]:6.2f} against {e['mean_estimate'][0]:6.2f} ug m-3: bias {f(e['bias'], 2)}, RMSE {f(e['rmse'], 2)}, "
                f"MAE {f(e['mae'], 2)}")
    RES.mkdir(exist_ok=True)

    def clean(v):                                                             # non-finite values (0 / 0 of the perfect 40 km forecast) -> null
        if isinstance(v, dict):
            return {k_: clean(x) for k_, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return None if isinstance(v, float) and not np.isfinite(v) else v
    (RES / "numbers_v2.json").write_text(json.dumps(clean(N), indent=1, allow_nan=False))
    (RES / "tables_v2.txt").write_text("\n".join(LINES) + "\n")
    print("\nsaved", RES / "numbers_v2.json")


if __name__ == "__main__":
    main()
