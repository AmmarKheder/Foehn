#!/usr/bin/env python3
"""Numbers of version 3 of 'who is warned'  ->  results/numbers_v3.json, results/tables_v3.txt

Version 2 (tables_v2.py of ../v2/, unchanged) with the three corrections of the code audit
(H-1, M-1, L-6; see figdata_hidden_v3.py):
  H-1  a new reference 'persist': the 1 km estimate of the start day kept unchanged, in every table, as issued and at the
       number of people warned by Foehn (and at Foehn's share of correct warnings), with the same bootstrap;
  M-1  every comparison at an equal number of people warned or at an equal share of correct warnings is computed at the
       exact factor: the point value from the fields at that factor (data/hidden_v3_exact.npz), the 95 % interval with the
       factor found again in every bootstrap resample (as version 2 did), on the curves of data/hidden_v3_pop.npz (factor
       grid 0.400, 0.401, ..., 2.000, linear inside one step of 0.001) instead of the factors 0.5, 0.6, ..., 2;
  L-6  the lead ladder (Foehn's ensemble mean on 0.25 deg blocks, 1, 3, 5 and 7 days) with the block of the resident's own
       cell; persistence added to it (at 1 km, and averaged over the same blocks as Foehn).
The machinery of ../tables.py is imported unchanged (parts, metrics, Tab, mbb, tri, f): pooled sums, moving-block bootstrap
over target days (blocks of 4 forecasts = 7 days), 10,000 resamples, seed 0, paired differences. The systems of version 2
keep exactly the values of v2/results/numbers_v2.json where nothing changed (asserted).
    python tables_v3.py"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tables import DIFFM, Tab, f, mbb, metrics, parts, tri  # noqa: E402  (../tables.py, unchanged)

TEST = "--test" in sys.argv                                                    # python tables_v3.py --test: on the files of the smoke test
SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
RES = HERE / f"results{SFX}" / ("test" if TEST else "")
OLDF, NEWF = HERE.parent / f"data{SFX}" / "hidden_systems_pop.npz", (HERE.parent / "v2" / f"data{SFX}" if SFX else HERE / "data") / "hidden_bc_pop.npz"
V3F, EXF = (HERE / f"data{SFX}" / f"hidden_v3_pop{'_test' if TEST else ''}.npz", HERE / f"data{SFX}" / f"hidden_v3_exact{'_test' if TEST else ''}.npz")
SRC = {"old": "../data/hidden_systems_pop.npz (job 23864988, ../figdata_hidden_systems.py)",
       "new": "data/hidden_bc_pop.npz (copy of v2/data/hidden_bc_pop.npz, figdata_hidden_bc.py)",
       "v3": "data/hidden_v3_pop.npz (v3/figdata_hidden_v3.py)"}
NAMES = {"foehn_p50": "Foehn, 2 of 4 members", "foehn_mean": "Foehn, ensemble mean", "foehn_40km": "Foehn mean on 40 km cells",
         "cams": "CAMS forecast", "aurora": "Aurora", "cams_qm": "CAMS forecast, bias-corrected",
         "cams_bcsd": "CAMS forecast, bias-corrected + 1 km pattern", "clim_1km": "1 km climatology alone",
         "persist": "start-day 1 km estimate kept unchanged", "perfect_40km": "perfect 40 km forecast"}
BASE = list(NAMES)
V3SYS = ("persist",)
REFS = ("cams", "aurora", "cams_qm", "cams_bcsd", "foehn_40km", "clim_1km", "persist")
PAIRS = [("foehn_p50", r) for r in REFS] + [("foehn_mean", "foehn_40km"), ("foehn_mean", "cams_bcsd"), ("foehn_mean", "cams_qm"),
                                              ("cams_qm", "cams"), ("cams_bcsd", "cams"), ("cams_bcsd", "cams_qm"), ("foehn_40km", "cams_bcsd"),
                                              ("foehn_mean", "persist"), ("persist", "cams_bcsd")]
DIFF2 = tuple(DIFFM) + ("warned_in_silent40_M", "false_per_hidden_warned")
RATIO = ("right_M", "false_M", "hidden_warned_M", "warned_M", "false_in_silent40_M", "hidden_margin_warned_M")
CROSS = ("reach_pct", "correct_pct", "right_M", "false_M", "warned_M", "hidden_warned_M", "hidden_warned_pct", "hidden_margin_warned_pct", "csi")
MQ = ("warned_M", "right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M", "hidden_warned_pct", "hidden_margin_warned_pct",
      "false_in_silent40_M", "csi")
LINES = []


def safe_write(path, text):
    """temp file + fsync + read-back + rename (a plain write was once dropped by the file system)"""
    import os
    path = str(path); data = text.encode("utf-8"); tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data); fh.flush(); os.fsync(fh.fileno())
    assert open(tmp, "rb").read() == data, path
    os.replace(tmp, path)
    assert open(path, "rb").read() == data, path


def out(s=""):
    print(s); LINES.append(s)


def main():
    zo, zn, z3, zx = np.load(OLDF), np.load(NEWF), np.load(V3F), np.load(EXF)
    if TEST:                                                                   # smoke test: the forecasts of the test files only
        zo, zn, z3, zx = ({k_: z[k_] for k_ in z.files} for z in (zo, zn, z3, zx))
        K3 = z3["keys"]; rows = [int(np.flatnonzero((zo["keys"][:, 0] == d) & (zo["keys"][:, 1] == h))[0]) for d, h, _ in K3]
        for z in (zo, zn):
            z["keys"], z["T1"] = z["keys"][rows], z["T1"][rows]
            if "diag" in z:
                z["diag"] = z["diag"][rows]
        r2 = [int(np.flatnonzero(zo["targets2"] == t_)[0]) for t_ in z3["targets2"]]
        zo["targets2"], zo["T2"] = zo["targets2"][r2], zo["T2"][r2]
    K = zo["keys"]
    assert np.array_equal(K, zn["keys"]) and np.array_equal(K, z3["keys"]) and np.array_equal(K, zx["keys"]), "forecasts differ"
    reg, thr = [str(r) for r in zo["regions"]], [float(t) for t in zo["thr"]]
    assert reg == [str(r) for r in z3["regions"]] and thr == [float(t) for t in z3["thr"]] and float(z3["margin"]) == float(zo["margin"])
    so, sn, s3 = list(zo["sys1"]), list(zn["sys1"]), list(z3["sys1"])
    To, Tn, T3 = zo["T1"], zn["T1"], z3["T1"]
    mult = [float(m) for m in zo["mult"]]
    mus = sorted(mult + [1.0])
    new_sys = {s for s in sn if s.split("@")[0] in ("cams_qm", "cams_bcsd", "clim_1km")}
    # ---- identity of everything computed twice
    ident = {s: bool(np.array_equal(T3[:, :, :, s3.index(s)], To[:, :, :, so.index(s)])) for s in ("foehn_p50", "foehn_mean", "foehn_40km", "cams", "aurora")}
    ident.update({s: bool(np.array_equal(T3[:, :, :, s3.index(s)], Tn[:, :, :, sn.index(s)])) for s in ("cams_qm", "cams_bcsd")})
    ident["foehn_25km read as in version 2"] = bool(np.array_equal(T3[:, :, :, s3.index("foehn_25km_oldindex")], To[:, :, :, so.index("foehn_25km")]))
    o2, n2 = list(zo["sys2"]), list(z3["sys2"])
    assert np.array_equal(zo["targets2"], z3["targets2"]) and [int(v) for v in zo["leads2"]] == [int(v) for v in z3["leads2"]]
    for a_, b_ in (("foehn_25km_oldindex", "foehn_25km"), ("cams", "cams"), ("aurora", "aurora")):
        x, y = z3["T2"][:, :, :, :, n2.index(a_)], zo["T2"][:, :, :, :, o2.index(b_)]
        ident[f"ladder {a_}"] = bool(np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)]))
    assert all(ident.values()), ident
    LEADS = sorted(int(h) for h in set(K[:, 1]))
    SEL = {h: np.flatnonzero(K[:, 1] == h) for h in LEADS}
    for h in LEADS:
        assert TEST or (np.diff(K[SEL[h], 0]) == 2).all()
    H, hsys, MUS = z3["H"], [str(s) for s in z3["hsys"]], z3["mus"]
    MU0, DMU = float(MUS[0]), float(np.round(MUS[1] - MUS[0], 6))
    SPECS = json.loads(str(zx["specs"]))
    # ---- the curves at the factors of version 2 equal its sweeps (the grid comparison is the same)
    worst_curve = 0.0
    for sname, src, lst in (("cams_qm", Tn, sn), ("cams_bcsd", Tn, sn), ("cams", To, so), ("aurora", To, so), ("foehn_mean", To, so)):
        for mu in mult:
            j = int(np.flatnonzero(MUS == mu)[0]); q = hsys.index(sname)
            cw = np.cumsum(H[:, :, q, :, ::-1].astype(np.float64), axis=-1)[..., ::-1][..., j + 1]       # (forecasts, thr, 8)
            ref = src[:, 0, :, lst.index(f"{sname}@{mu:g}")][..., 8:]
            worst_curve = max(worst_curve, float((np.abs(cw - ref).sum(-1) / np.maximum(ref.sum(-1), 1.0)).max()))
    assert worst_curve < 1e-5, worst_curve

    def arr(s):
        if s in V3SYS:
            return T3[:, :, :, s3.index(s)], "v3"
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

    # ---- a moved system at its exact factor: point from the fields (stage 3), interval with the factor found again per resample
    def at_factor(spec, h, t, fixed_tab):
        sel = SEL[h]; n = len(sel); cnt = mbb(n, 4)
        k, q = thr.index(t), hsys.index(spec["moved"])
        x = Tab(zx[f"spec{spec['index']}"][:, 0], cnt)                                  # exact, from the fields
        Hd = H[sel, k, q].astype(np.float64)                                            # (days, 8, NB + 2)
        Cd = np.cumsum(Hd[:, :, ::-1], axis=2)[:, :, ::-1][:, :, 1:]                    # (days, 8, NB + 1): warned at each factor
        jm = int(round((spec["mu"] - MU0) / DMU)); lo, hi = max(0, jm - 250), min(Cd.shape[2], jm + 251)
        Cw = Cd[:, :, lo:hi]; nw = hi - lo
        bW = (cnt @ Cw.reshape(n, -1)).reshape(-1, 8, nw)                              # (B, 8, window)
        a = fixed_tab.a; bN = cnt @ (a[:, :8] + a[:, 8:])                                # residents by truth code (independent of the system)
        if spec["kind"] == "equal_correct_share":
            fb = fixed_tab.bsum.astype(np.float64); tgt = parts(fb)["tp"] / parts(fb)["warn"]
            yc = bW[:, 1::2].sum(1) / np.maximum(bW.sum(1), 1e-9); s_ = yc - tgt[:, None]
        else:
            tgt = parts(fixed_tab.bsum.astype(np.float64))["warn"]; s_ = tgt[:, None] - bW.sum(1)
        cross = (s_[:, :-1] < 0) & (s_[:, 1:] >= 0)
        okb = cross.any(1); j0 = np.argmax(cross, axis=1); rr = np.arange(len(j0))
        fr = -s_[rr, j0] / (s_[rr, j0 + 1] - s_[rr, j0])
        Wb = bW[rr, :, j0] + fr[:, None] * (bW[rr, :, j0 + 1] - bW[rr, :, j0])          # (B, 8)
        A16 = np.concatenate([bN - Wb, Wb], axis=1); A16[~okb] = np.nan
        bs = metrics(parts(A16), n)
        # the same interpolation on the pooled curve (check of the point value from the fields)
        Pw = Cw.sum(0); sp = (Pw[1::2].sum(0) / np.maximum(Pw.sum(0), 1e-9) - spec["target"]) if spec["kind"] == "equal_correct_share" else (spec["target"] - Pw.sum(0))
        j = int(np.flatnonzero((sp[:-1] < 0) & (sp[1:] >= 0))[0]); fr0 = -sp[j] / (sp[j + 1] - sp[j])
        P = Pw[:, j] + fr0 * (Pw[:, j + 1] - Pw[:, j])
        N0 = a.sum(0)[:8] + a.sum(0)[8:]
        hp = metrics(parts(np.concatenate([N0 - P, P])), n)
        if spec["moved"] == "persist":
            # the start-day estimate takes discrete values (half precision): many residents share each value, so that no factor
            # gives exactly the target (one value can hold 0.1-0.4 % of the people warned). Its point is therefore taken between
            # the two neighbouring factors of the grid, both evaluated exactly from the fields (curves of stage 1), linearly in the
            # target: a share of the residents at the tied value is warned. The value at the root factor itself is kept as 'at_root'.
            class P_:
                pass
            y = P_(); y.pt = hp; y.at_root = x.pt
            return y, bs, int((~okb).sum()), x.pt
        return x, bs, int((~okb).sum()), hp

    N = {"what": "version 3 of the numbers of 'who is warned': version 2 with the start-day estimate kept unchanged as a reference (H-1), every "
                 "comparison at an equal number of people warned or an equal share of correct warnings at the exact factor (M-1), and the lead "
                 "ladder read from the 0.25 deg block of the resident's own cell (L-6)",
         "population": "residents of the 8.1 M 1 km grid cells with at least 100 residents (7,388 M, 94 % of the world's residents); "
                       "358 retrospective forecasts of 2022 (179 per lead time, start days every second day)",
         "interval": "[value, lower, upper]: 95 % interval of a moving-block bootstrap over target days, blocks of 4 forecasts (7 days), "
                     "10,000 resamples, seed 0 (../tables.py: mbb, Tab), paired for differences",
         "exact_factor": "systems moved to a target (Foehn's number of people warned, or a share of correct warnings): the factor is found on "
                         "the pooled curve of the system on the grid 0.400, 0.401, ..., 2.000 (linear inside one step) and the system is "
                         "re-evaluated from the fields at that factor (point values); in each bootstrap resample the factor is found again "
                         "on the resampled curve (intervals)",
         "sources": {s: SRC["v3" if s in V3SYS else "new" if s in new_sys else "old"] + f", system '{s}'" for s in BASE},
         "definitions": {"warned": "the forecast read at the resident's 1 km grid cell is above the threshold (Foehn: at least 2 of its 4 members)",
                         "persist": "warned when the 1 km estimate of the start day of Foehn's forecast in the resident's grid cell is above the "
                                    "threshold (ghap1km_global.npy; no estimate: not warned)",
                         "exposed": "the 1 km estimate of the target day is above the threshold",
                         "hidden": "exposed while the mean of the estimates over the 40 km cell is not above the threshold",
                         "hidden with a margin": "grid cell more than 10 % above the threshold and 40 km mean more than 10 % below it",
                         "reach_pct": "exposed and warned / exposed", "correct_pct": "exposed and warned / warned",
                         "warned_in_silent40_M": "people warned where the 40 km mean is not above the threshold",
                         "false_in_silent40_M": "of whom not exposed (wrongly warned)",
                         "false_per_hidden_warned": "false_in_silent40 / hidden people warned"},
         "identity_of_recomputed_systems": ident,
         "check_curves_equal_version2_sweeps_max_rel_dev": worst_curve,
         "checks_stage0": json.loads(str(z3["checks0"]))}
    S = N["systems"] = {}
    old = json.loads((HERE.parent / "v2" / f"results{SFX}" / "numbers_v2.json").read_text())["systems"]
    worst = 0.0
    for t in (37.5, 15.0, 75.0, 25.0, 50.0):
        for h in LEADS:
            b = T(h, t, "World", "foehn_p50")
            out(f"\n=== World, > {t:g} ug m-3, {h} day{'s' if h > 1 else ''} ahead, {b.n} target days. People a day (millions): exposed "
                f"{f(b.ci('exposed_M'))}, hidden {f(b.ci('hidden_M'))}, hidden with a 10 % margin {f(b.ci('hidden_margin_M'))}")
            out(f"  {'system':44s} {'warned':>7s} {'rightly M':>20s} {'wrongly M':>18s} {'reach %':>18s} {'correct %':>18s} {'hidden warned M':>18s} "
                f"{'of hidden %':>18s} {'margin-hidden %':>18s} | 40 km mean below: {'warned M':>8s} {'wrongly M':>18s} {'per hidden warned':>18s} | csi")
            W = S.setdefault(f"thr{t:g}", {}).setdefault(f"lead{h}", {}).setdefault("World", {})
            for s in BASE:
                x = T(h, t, "World", s); a = x.all(); a["source"] = N["sources"][s]; W[s] = a
                out(f"  {NAMES[s]:44s} {a['warned_M'][0]:7.0f} {f(a['right_M'], 0):>20s} {f(a['false_M'], 0):>18s} {f(a['reach_pct']):>18s} {f(a['correct_pct']):>18s} "
                    f"{f(a['hidden_warned_M']):>18s} {f(a['hidden_warned_pct']):>18s} {f(a['hidden_margin_warned_pct']):>18s} | {'':17s} "
                    f"{a['warned_in_silent40_M'][0]:8.0f} {f(a['false_in_silent40_M'], 0):>18s} {f(a['false_per_hidden_warned'], 2):>18s} | {f(a['csi'], 3)}")
                if s in old[f"thr{t:g}"][f"lead{h}"]["World"]:                # version 2 values reproduced
                    o = old[f"thr{t:g}"][f"lead{h}"]["World"][s]
                    worst = max([worst] + [abs(a[k][j] - o[k][j]) for k in o if k != "source" for j in range(3) if o[k][j] is not None and np.isfinite(o[k][j])])
            out("  paired differences (first minus second):")
            D = W["differences"] = {}
            for a_, b_ in PAIRS:
                xa, xb = T(h, t, "World", a_), T(h, t, "World", b_)
                d = D[f"{a_}-{b_}"] = xa.diffs(xb, DIFF2)
                D[f"{a_}/{b_}"] = xa.ratios(xb, RATIO)
                out(f"    {a_:10s} - {b_:10s}: rightly {f(d['right_M'])} M | wrongly {f(d['false_M'])} M | reach {f(d['reach_pct'])} pts | correct "
                    f"{f(d['correct_pct'])} pts | hidden {f(d['hidden_warned_M'])} M, {f(d['hidden_warned_pct'])} pts | margin-hidden "
                    f"{f(d['hidden_margin_warned_pct'])} pts | wrongly where the 40 km mean is below {f(d['false_in_silent40_M'])} M | csi {f(d['csi'], 3)}")
            ref = T(h, t, "World", "foehn_p50"); refbs = ref.bs
            sp_here = [s_ for s_ in SPECS if s_["lead"] == h and s_["thr"] == t]
            # ---- equal share of correct warnings (exact factor)
            QC = W["equal_correct_share"] = {}
            for s_ in [s_ for s_ in sp_here if s_["kind"] == "equal_correct_share"]:
                moved, fixed = s_["moved"], s_["fixed"]
                if s_["mu"] is None:
                    out(f"  {moved} at the share of correct warnings of {fixed}: outside the factors 0.4-2"); continue
                fx_ = T(h, t, "World", fixed); fbs = fx_.bs
                x, bs, nfail, hp = at_factor(s_, h, t, fx_)
                res = {"correct_pct": fx_.ci("correct_pct"), "decision_threshold_factor": s_["mu"], "crossings_on_the_grid": s_["crossings"],
                       "resamples_without_a_crossing": nfail, "moved_correct_pct_at_the_factor": tri(x.pt["correct_pct"], bs["correct_pct"])}
                for m in ("warned_M", "right_M", "false_M", "reach_pct", "hidden_warned_M", "hidden_warned_pct", "hidden_margin_warned_pct", "csi"):
                    res[m] = tri(x.pt[m], bs[m]); res[f"{fixed}_{m}"] = fx_.ci(m); res[f"moved_minus_fixed_{m}"] = tri(x.pt[m] - fx_.pt[m], bs[m] - fbs[m])
                    if m in ("right_M", "hidden_warned_M"):
                        res[f"moved_over_fixed_{m}"] = tri(x.pt[m] / fx_.pt[m], bs[m] / fbs[m])
                res["check_point_from_the_curve" if s_["moved"] != "persist" else "check_point_at_the_root_factor_from_the_fields"] = {m: hp[m] for m in ("warned_M", "reach_pct", "correct_pct", "hidden_warned_M")}
                QC[f"{moved}_at_correct_share_of_{fixed}"] = res
                out(f"  {moved} at the share of correct warnings of {fixed} ({fx_.pt['correct_pct']:.1f} %; threshold x {s_['mu']:.3f}; its share {x.pt['correct_pct']:.2f} %): "
                    f"warned {f(res['warned_M'], 0)} M, reach {f(res['reach_pct'])} % (fixed {fx_.pt['reach_pct']:.1f}), hidden {f(res['hidden_warned_M'])} M = "
                    f"{f(res['hidden_warned_pct'])} % (fixed {fx_.pt['hidden_warned_M']:.1f} M, {fx_.pt['hidden_warned_pct']:.1f} %); moved minus fixed: reach "
                    f"{f(res['moved_minus_fixed_reach_pct'])} pts, hidden {f(res['moved_minus_fixed_hidden_warned_M'])} M; ratio hidden "
                    f"{f(res['moved_over_fixed_hidden_warned_M'], 2)}; resamples without a crossing {nfail}")
            # ---- equal number of people warned (exact factor)
            QN = W["equal_number_warned"] = {}
            for s_ in [s_ for s_ in sp_here if s_["kind"] == "equal_number"]:
                sw = s_["moved"]
                if s_["mu"] is None:
                    out(f"  {sw} at Foehn's number of people warned: outside the factors 0.4-2"); continue
                x, bs, nfail, hp = at_factor(s_, h, t, ref)
                res = {"decision_threshold_factor": s_["mu"], "crossings_on_the_grid": s_["crossings"], "resamples_without_a_crossing": nfail,
                       "warned_M_at_the_factor": tri(x.pt["warned_M"], bs["warned_M"]),
                       "point_from": "the fields at the factor" if sw != "persist" else "linear between the two neighbouring factors of the grid of 0.001 (exact field evaluations), tied values", "foehn_p50_warned_M": ref.ci("warned_M")}
                for m in ("right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M", "hidden_warned_pct", "hidden_margin_warned_pct", "csi"):
                    res[m] = tri(x.pt[m], bs[m]); res[f"foehn_p50_minus_{m}"] = tri(ref.pt[m] - x.pt[m], refbs[m] - bs[m])
                res["check_point_from_the_curve" if s_["moved"] != "persist" else "check_point_at_the_root_factor_from_the_fields"] = {m: hp[m] for m in ("warned_M", "reach_pct", "correct_pct", "hidden_warned_M")}
                QN[sw] = res
                out(f"  {sw} at Foehn's number of people warned ({ref.pt['warned_M']:.1f} M; its threshold x {s_['mu']:.3f}; it warns {x.pt['warned_M']:.2f} M): reach "
                    f"{f(res['reach_pct'])} %, correct {f(res['correct_pct'])} %, hidden {f(res['hidden_warned_M'])} M ({f(res['hidden_warned_pct'])} %) | Foehn minus it: "
                    f"rightly {f(res['foehn_p50_minus_right_M'])} M, reach {f(res['foehn_p50_minus_reach_pct'])} pts, correct {f(res['foehn_p50_minus_correct_pct'])} pts, "
                    f"hidden {f(res['foehn_p50_minus_hidden_warned_M'])} M, csi {f(res['foehn_p50_minus_csi'], 3)} | other point: warned {hp['warned_M']:.2f}, reach {hp['reach_pct']:.3f}, "
                    f"correct {hp['correct_pct']:.3f} (reported {x.pt['reach_pct']:.3f}, {x.pt['correct_pct']:.3f}); resamples without a crossing {nfail}")
            # ---- Foehn's ensemble mean at the number of people warned by a reference (exact factor)
            QG = W["foehn_mean_at_the_number_warned_by"] = {}
            for s_ in [s_ for s_ in sp_here if s_["kind"] == "foehn_mean_at_number_warned_by"]:
                fixed = s_["fixed"]
                if s_["mu"] is None:
                    out(f"  foehn_mean at the number of people warned by {fixed}: outside the factors 0.4-2"); continue
                fx_ = T(h, t, "World", fixed); fbs = fx_.bs
                x, bs, nfail, hp = at_factor(s_, h, t, fx_)
                res = {"warned_M": fx_.ci("warned_M"), "decision_threshold_factor": s_["mu"], "crossings_on_the_grid": s_["crossings"],
                       "resamples_without_a_crossing": nfail}
                for m in ("right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M", "hidden_warned_pct", "hidden_margin_warned_pct", "csi"):
                    res[m] = tri(x.pt[m], bs[m]); res[f"{fixed}_{m}"] = fx_.ci(m); res[f"foehn_mean_minus_{fixed}_{m}"] = tri(x.pt[m] - fx_.pt[m], bs[m] - fbs[m])
                QG[fixed] = res
                out(f"  foehn_mean at the number of people warned by {fixed} ({fx_.pt['warned_M']:.0f} M; threshold x {s_['mu']:.3f}): reach "
                    f"{f(res['reach_pct'])} % (ref {fx_.pt['reach_pct']:.1f}), correct {f(res['correct_pct'])} % (ref {fx_.pt['correct_pct']:.1f}), hidden "
                    f"{f(res['hidden_warned_M'])} M (ref {fx_.pt['hidden_warned_M']:.1f}) | Foehn mean minus it: reach {f(res[f'foehn_mean_minus_{fixed}_reach_pct'])} pts, "
                    f"correct {f(res[f'foehn_mean_minus_{fixed}_correct_pct'])} pts, hidden {f(res[f'foehn_mean_minus_{fixed}_hidden_warned_M'])} M")
            W["decision_threshold_sweep"] = {sw: {f"{mu:g}": {m: x.pt[m] for m in ("warned_M", "right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M",
                                                                                 "hidden_warned_pct", "false_in_silent40_M")}
                                                  for mu, x in zip(mus, swept(h, t, sw))} for sw in ("foehn_mean", "cams", "aurora", "cams_qm", "cams_bcsd")}
            # the persistence curve on the grid of version 2 (from the stored curves; for the figure)
            q = hsys.index("persist"); Cd = np.cumsum(H[SEL[h], thr.index(t), q, :, ::-1].astype(np.float64), axis=-1)[..., ::-1]
            a0 = T(h, t, "World", "persist").a; N0 = a0[:, :8] + a0[:, 8:]
            W["decision_threshold_sweep"]["persist"] = {}
            for mu in mus:
                j = int(np.flatnonzero(MUS == mu)[0]); Wc = Cd[:, :, j + 1]
                W["decision_threshold_sweep"]["persist"][f"{mu:g}"] = {m: v for m, v in metrics(parts(np.concatenate([N0 - Wc, Wc], 1).sum(0)), len(SEL[h])).items()
                                                                       if m in ("warned_M", "right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M",
                                                                                "hidden_warned_pct", "false_in_silent40_M")}
    N["check_version2_reproduced"] = {"max_abs_deviation_from_v2/results/numbers_v2.json (systems as issued)": worst}
    assert TEST or worst < 1e-9, worst
    out(f"\n(version 2 reproduced for the systems as issued: worst deviation {worst:.1e})")
    # ---------------------------------------------------------------- Foehn issued 3 days ahead against the references issued 1 day ahead (as version 2)
    X = N["foehn_3_days_against_references_1_day"] = {
        "note": "common target days of the forecasts issued 3 days and 1 day ahead (paired bootstrap over these days, blocks of 4)"}
    t1, t3 = K[SEL[1], 2], K[SEL[3], 2]
    common = np.intersect1d(t1, t3); assert TEST or (np.diff(common) == 2).all()
    i1, i3 = SEL[1][np.searchsorted(t1, common)], SEL[3][np.searchsorted(t3, common)]
    cnt = mbb(len(common), 4) if len(common) else None; X["n_common_target_days"] = int(len(common))
    for t in ((15.0, 37.5, 75.0) if len(common) else ()):
        k = thr.index(t)
        out(f"\n=== Foehn (2 of 4 members) issued 3 days ahead against the references issued 1 day ahead, World, > {t:g}, {len(common)} common target days")
        fo = Tab(arr("foehn_p50")[0][i3, 0, k], cnt)
        X[f"thr{t:g}"] = {"foehn_p50_lead3": fo.all()}
        for r_ in ("cams", "aurora", "cams_qm", "cams_bcsd", "foehn_40km"):
            rf = Tab(arr(r_)[0][i1, 0, k], cnt)
            d = fo.diffs(rf, CROSS)
            X[f"thr{t:g}"][f"{r_}_lead1"] = rf.all(); X[f"thr{t:g}"][f"foehn_p50_lead3-{r_}_lead1"] = d
            out(f"    Foehn 3 d: reach {fo.pt['reach_pct']:.1f} %, correct {fo.pt['correct_pct']:.1f} % | {r_:10s} 1 d: reach {rf.pt['reach_pct']:.1f} %, correct "
                f"{rf.pt['correct_pct']:.1f} % | difference: reach {f(d['reach_pct'])} pts, correct {f(d['correct_pct'])} pts")
    # ---------------------------------------------------------------- regions (country groups), with persistence
    R = N["regions"] = {"note": "country groups of ../regions.py (a partition of the world)"}
    for t in (37.5, 75.0, 15.0):
        for h in LEADS:
            out(f"\n=== Country groups, > {t:g}, {h} day{'s' if h > 1 else ''} ahead: reach % / correct % / hidden warned % for Foehn | CAMS | bias-corrected + pattern | persistence")
            for r in reg:
                if not r.startswith("G:"):
                    continue
                x = {s: T(h, t, r, s) for s in ("foehn_p50", "foehn_40km", "cams", "aurora", "cams_qm", "cams_bcsd", "persist")}
                if x["foehn_p50"].pt["exposed_M"] < 0.05:
                    continue
                e = R.setdefault(f"thr{t:g}", {}).setdefault(f"lead{h}", {})[r[2:]] = {s: v.all() for s, v in x.items()}
                e["foehn_p50-cams_bcsd"] = x["foehn_p50"].diffs(x["cams_bcsd"], DIFF2)
                e["foehn_p50-cams_qm"] = x["foehn_p50"].diffs(x["cams_qm"], DIFF2)
                e["foehn_p50-persist"] = x["foehn_p50"].diffs(x["persist"], DIFF2)
                out(f"  {r[2:]:32s} exposed {x['foehn_p50'].pt['exposed_M']:7.1f} | "
                    + " | ".join(f"{x[s].pt['reach_pct']:5.1f} / {x[s].pt['correct_pct']:5.1f} / {x[s].pt['hidden_warned_pct']:5.1f}" for s in ("foehn_p50", "cams", "cams_bcsd", "persist")))
    # ---------------------------------------------------------------- lead ladder (L-6: the block of the resident's own cell; persistence added)
    T2, sys2, leads2 = z3["T2"], [str(s) for s in z3["sys2"]], [int(v) for v in z3["leads2"]]
    okd = np.isfinite(T2[:, 0, 0, :, sys2.index("foehn_25km"), 0]).all(1)
    cnt2 = mbb(int(okd.sum()), min(4, int(okd.sum())))
    L = N["lead_ladder"] = {"n_target_days": int(okd.sum()), "note": "target days of the 1-day forecasts for which the forecasts started 1, 3, 5 and 7 days "
                            "before exist; foehn_25km = Foehn's ensemble mean averaged over the centred 0.25 deg block of the resident's own cell (version 2 read "
                            "the block centred on the north-west corner point: foehn_25km_oldindex); persist = start-day 1 km estimate kept unchanged; "
                            "persist_25km = the same averaged over the block of foehn_25km"}
    for t in (37.5, 15.0, 75.0):
        k = thr.index(t)
        out(f"\n=== Lead ladder, World, > {t:g} ug m-3, {int(okd.sum())} common target days: reach % / correct % / hidden warned %")
        Xl = {}
        for s in sys2:
            row = []
            for li, ld in enumerate(leads2):
                a = T2[okd, 0, k, li, sys2.index(s)]
                if not np.isfinite(a).all():
                    continue
                x = Xl[(s, ld)] = Tab(a, cnt2)
                L.setdefault(f"thr{t:g}", {}).setdefault(s, {})[f"lead{ld}"] = x.all()
                row.append(f"{ld} d: {f(x.ci('reach_pct'))} / {f(x.ci('correct_pct'))} / {f(x.ci('hidden_warned_pct'))}")
            out(f"  {s:20s} " + " | ".join(row))
        D = L[f"thr{t:g}"]["differences"] = {}
        for ref_ in ("cams", "aurora"):
            for ld, lr in ((1, 1), (3, 3), (5, 5), (3, 1), (5, 1), (7, 1)):
                for fs in ("foehn_25km", "foehn_25km_oldindex"):
                    d = D[f"{fs}_lead{ld}-{ref_}_lead{lr}"] = Xl[(fs, ld)].diffs(Xl[(ref_, lr)], DIFFM)
                    out(f"    {fs} {ld} d - {ref_} {lr} d: reach {f(d['reach_pct'])} pts | correct {f(d['correct_pct'])} pts | csi {f(d['csi'], 3)}")
        for ld in leads2:
            for ref_ in ("persist", "persist_25km"):
                d = D[f"foehn_25km_lead{ld}-{ref_}_lead{ld}"] = Xl[("foehn_25km", ld)].diffs(Xl[(ref_, ld)], DIFFM)
                out(f"    foehn_25km {ld} d - {ref_} {ld} d: reach {f(d['reach_pct'])} pts | correct {f(d['correct_pct'])} pts | hidden {f(d['hidden_warned_pct'])} pts")
    # ---------------------------------------------------------------- diagnostics of the new reference
    dg = z3["diag"]
    N["persistence_coverage"] = {f"lead{h}": {"valid_residents_M": float(dg[SEL[h], 0].mean() / 1e6),
                                              "valid_residents_without_a_start_day_estimate_M": float(dg[SEL[h], 1].mean() / 1e6),
                                              "valid_residents_without_a_0.25deg_block_M": float(dg[SEL[h], 2].mean() / 1e6)} for h in LEADS}
    N["exact_factor_checks"] = [{k_: s_[k_] for k_ in ("kind", "lead", "thr", "moved", "fixed", "mu", "crossings", "target", "exact_warned", "exact_right") if k_ in s_}
                                for s_ in SPECS]
    out("\n=== persistence coverage: " + json.dumps(N["persistence_coverage"]))
    rel = [abs(s_["exact_warned"] - s_["target"]) / s_["target"] for s_ in SPECS if s_["kind"] != "equal_correct_share" and s_.get("mu") is not None
           and s_["moved"] != "persist"]
    relp = [abs(s_["exact_warned"] - s_["target"]) / s_["target"] for s_ in SPECS if s_["kind"] != "equal_correct_share" and s_.get("mu") is not None
            and s_["moved"] == "persist"]
    out(f"=== exact factors: worst relative gap between the number warned at the factor and the target {max(rel):.2e} (continuous systems); "
        f"{max(relp):.2e} for the start-day estimate at its root factor (tied values; its point is taken between the two neighbouring grid factors)")
    N["exact_factor_worst_relative_gap_in_number_warned"] = {"continuous systems": max(rel), "persist at the root factor (not used as its point)": max(relp)}
    RES.mkdir(exist_ok=True, parents=True)

    def clean(v):
        if isinstance(v, dict):
            return {k_: clean(x) for k_, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        return None if isinstance(v, float) and not np.isfinite(v) else v
    safe_write(RES / "numbers_v3.json", json.dumps(clean(N), indent=1, allow_nan=False))
    safe_write(RES / "tables_v3.txt", "\n".join(LINES) + "\n")
    print("\nsaved", RES / "numbers_v3.json")


if __name__ == "__main__":
    main()
