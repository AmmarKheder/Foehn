#!/usr/bin/env python3
"""Numbers of the hidden-population analysis, with intervals over days  ->  results/numbers.json, results/tables.txt

Inputs (nothing is recomputed from the fields here)
  data/hidden_systems_pop.npz       figdata_hidden_systems.py: residents by (pixel above, 40 km cell above, hidden with a
                                    margin, warned) per forecast, region, threshold and system; the lead ladder
  FIG/data/exposure_2022.npz        figures/figdata_exposure.py: all pixels, 365 days of 2022, the boxes of the paper
  data/exposure_sens_2022.npz       figdata_exposure_sens.py: blocks of 10/25/40 km, shifted origins, margins, monitor pixels,
                                    country groups (read when the job has finished)
  FIG/data/exposure_income_2022.npz figures/figdata_exposure_income.py: World Bank income groups, quintiles within countries

Definitions (populated pixels: the 8.1 M 1 km pixels with at least 100 residents, 94 % of the world's residents)
  exposed   the 1 km estimate of the target day at the resident's pixel is above the threshold
  hidden    exposed, and the mean of the estimates over the 40 km cell is not above the threshold
  warned    the forecast read at the resident's pixel is above the threshold (Foehn: at least 2 of its 4 members)
  reach     exposed and warned / exposed            (hit rate)
  correct   exposed and warned / warned             (1 - false-alarm ratio)
Intervals: moving-block bootstrap over the target days, blocks of 4 consecutive forecasts (start days every second day: 7
days, the block length of the paper), 10,000 resamples, 95 %; differences between systems are paired (the same resampled
days). Sums are pooled over days before any ratio is taken.
    python tables.py"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIG = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures")
RES = HERE / "results"
B = 10000
c16 = np.arange(16)
E, C, HM, WN = (c16 & 1) > 0, (c16 & 2) > 0, (c16 & 4) > 0, (c16 & 8) > 0
H = E & ~C
LINES = []


def out(s=""):
    print(s); LINES.append(s)


_CNT = {}


def mbb(n, L, seed=0):
    """count matrix (B, n) of a moving-block bootstrap (blocks of L consecutive items, truncated to n; as compare_fair.py)"""
    if (n, L, seed) in _CNT:
        return _CNT[(n, L, seed)]
    rng = np.random.default_rng(seed)
    st = rng.integers(0, n - L + 1, size=(B, -(-n // L)))
    idx = (st[:, :, None] + np.arange(L)).reshape(B, -1)[:, :n]
    cnt = np.zeros((B, n))
    np.add.at(cnt, (np.arange(B)[:, None], idx), 1.0)
    _CNT[(n, L, seed)] = cnt
    return cnt


def parts(a):
    """a (..., 16 codes) -> sums of residents"""
    return dict(valid=a.sum(-1), exp=a[..., E].sum(-1), hid=a[..., H].sum(-1), hidm=a[..., HM].sum(-1), rev=a[..., ~E & C].sum(-1),
                warn=a[..., WN].sum(-1), tp=a[..., WN & E].sum(-1), fa=a[..., WN & ~E].sum(-1), hw=a[..., WN & H].sum(-1),
                hmw=a[..., WN & HM].sum(-1), wc=a[..., WN & ~C].sum(-1), wcfa=a[..., WN & ~C & ~E].sum(-1),
                seenw=a[..., WN & E & C].sum(-1), revw=a[..., WN & ~E & C].sum(-1))


def metrics(p, n):
    M = 1e6 * n
    with np.errstate(all="ignore"):
        return dict(
            valid_M=p["valid"] / M, exposed_M=p["exp"] / M, hidden_M=p["hid"] / M, hidden_margin_M=p["hidm"] / M, reverse_M=p["rev"] / M,
            hidden_pct_of_exposed=100 * p["hid"] / p["exp"],
            warned_M=p["warn"] / M, right_M=p["tp"] / M, false_M=p["fa"] / M,
            reach_pct=100 * p["tp"] / p["exp"], correct_pct=100 * p["tp"] / p["warn"], false_pct=100 * p["fa"] / p["warn"],
            hidden_warned_M=p["hw"] / M, hidden_warned_pct=100 * p["hw"] / p["hid"],
            hidden_margin_warned_M=p["hmw"] / M, hidden_margin_warned_pct=100 * p["hmw"] / p["hidm"],
            seen_warned_pct=100 * p["seenw"] / (p["exp"] - p["hid"]),
            warned_in_silent40_M=p["wc"] / M, false_in_silent40_M=p["wcfa"] / M, false_pct_in_silent40=100 * p["wcfa"] / p["wc"],
            false_per_hidden_warned=p["wcfa"] / p["hw"],
            csi=p["tp"] / (p["exp"] + p["fa"]), warned_over_exposed=p["warn"] / p["exp"])


def tri(pt, bs):
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return [float(pt), float(lo), float(hi)]


class Tab:
    """one code table (days, 16) with its bootstrap sums"""

    def __init__(self, a, cnt):
        self.n = len(a); self.a = a; self.pt = metrics(parts(a.sum(0)), self.n); self.bsum = (cnt @ a).astype(np.float32)

    @property
    def bs(self):
        return metrics(parts(self.bsum.astype(np.float64)), self.n)

    @property
    def daily(self):
        return metrics(parts(self.a), 1)

    def ci(self, k):
        return tri(self.pt[k], self.bs[k])

    def all(self):
        bs = self.bs
        return {k: tri(self.pt[k], bs[k]) for k in self.pt}

    def diff(self, other, k):
        return tri(self.pt[k] - other.pt[k], self.bs[k] - other.bs[k])

    def diffs(self, other, ks):
        a, b = self.bs, other.bs
        return {k: tri(self.pt[k] - other.pt[k], a[k] - b[k]) for k in ks}

    def ratios(self, other, ks):
        a, b = self.bs, other.bs
        return {k: tri(self.pt[k] / other.pt[k], a[k] / b[k]) for k in ks}

    def rng(self, k):
        d = self.daily[k]
        return [float(np.nanmin(d)), float(np.nanpercentile(d, 5)), float(np.nanmedian(d)), float(np.nanpercentile(d, 95)), float(np.nanmax(d))]


def f(v, d=1, pct=False):
    u = " %" if pct else ""
    return f"{v[0]:.{d}f}{u} [{v[1]:.{d}f}, {v[2]:.{d}f}]"


NAMES = {"foehn_p50": "Foehn, 2 of 4 members", "foehn_mean": "Foehn, ensemble mean", "foehn_p25": "Foehn, 1 of 4 members",
         "foehn_p75": "Foehn, 3 of 4 members", "foehn_p100": "Foehn, 4 of 4 members", "foehn_40km": "Foehn mean on 40 km blocks",
         "foehn_25km": "Foehn mean on 0.25 deg blocks", "cams": "CAMS forecast (bilinear)", "cams_cell": "CAMS forecast (grid point)",
         "aurora": "Aurora (bilinear)", "aurora_cell": "Aurora (grid point)", "perfect_40km": "perfect 40 km forecast"}
BASE = list(NAMES)
KEYM = ("warned_M", "right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M", "hidden_warned_pct",
        "hidden_margin_warned_pct", "false_pct_in_silent40")
PAIRS = (("foehn_p50", "cams"), ("foehn_p50", "aurora"), ("foehn_p50", "foehn_40km"), ("foehn_mean", "foehn_40km"),
         ("foehn_mean", "cams"), ("foehn_mean", "aurora"), ("foehn_p50", "foehn_mean"), ("foehn_p50", "cams_cell"), ("foehn_40km", "cams"),
         ("foehn_p50", "perfect_40km"), ("foehn_p50", "foehn_25km"))
DIFFM = ("warned_M", "right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M", "hidden_warned_pct",
         "hidden_margin_warned_M", "hidden_margin_warned_pct", "false_in_silent40_M", "false_pct_in_silent40", "csi")


def systems(N):
    z = np.load(HERE / "data" / os.environ.get("HID_FILE", "hidden_systems_pop.npz"))   # HID_FILE=hidden_systems_pop_leads.npz: more leads
    LEADS = sorted(int(h) for h in set(z["keys"][:, 1]))
    K, T1, sys1, thr, reg = z["keys"], z["T1"], list(z["sys1"]), [float(t) for t in z["thr"]], [str(r) for r in z["regions"]]
    mult = [float(m) for m in z["mult"]]
    S = N["systems"] = {"population": "residents of the 8.1 M 1 km pixels with at least 100 residents (7,388 M, 94 % of the world's residents)",
                        "interval": f"moving-block bootstrap over target days, blocks of 4 forecasts (7 days), {B} resamples, 95 %"}
    SEL = {}
    for h in LEADS:
        sel = np.flatnonzero(K[:, 1] == h); n = len(sel)
        assert (np.diff(K[sel, 0]) == 2).all(), "start days are not every second day"
        SEL[h] = sel
        S[f"lead{h}"] = {"n_target_days": n, "first_target": int(K[sel, 2].min()), "last_target": int(K[sel, 2].max())}

    class Lazy(dict):
        def __missing__(self, key):
            h, t, r, s = key
            x = Tab(T1[SEL[h], reg.index(r), thr.index(t), sys1.index(s)], mbb(len(SEL[h]), 4))
            if r == "World":
                self[key] = x
            return x
    T = Lazy()
    # ---------------------------------------------------------------- world, every system
    for t in (37.5, 15.0, 75.0, 25.0, 50.0):
        for h in LEADS:
            b = T[(h, t, "World", "foehn_p50")]
            out(f"\n=== World, > {t:g} ug m-3, {h} day{'s' if h > 1 else ''} ahead, {b.n} target days. People a day (millions): exposed "
                f"{f(b.ci('exposed_M'))}, hidden {f(b.ci('hidden_M'))} ({f(b.ci('hidden_pct_of_exposed'), 1, True)} of the exposed), "
                f"hidden with a 10 % margin {f(b.ci('hidden_margin_M'))}, reverse (cell above, pixel not) {f(b.ci('reverse_M'))}; "
                f"valid residents {b.pt['valid_M']:.0f}")
            out(f"  daily hidden (min, p5, median, p95, max): " + ", ".join(f"{v:.1f}" for v in b.rng("hidden_M")))
            out(f"  {'system':30s} {'warned M':>22s} {'rightly M':>22s} {'falsely M':>20s} {'reach %':>18s} {'correct %':>18s} "
                f"{'hidden warned M':>18s} {'of hidden %':>18s} {'margin-hidden %':>18s} {'false % silent40':>18s}")
            for s in BASE:
                x = T[(h, t, "World", s)]
                out(f"  {NAMES[s]:30s} " + " ".join(f"{f(x.ci(m)):>{w}s}" for m, w in zip(KEYM, (22, 22, 20, 18, 18, 18, 18, 18, 18))))
                S.setdefault(f"thr{t:g}", {}).setdefault(f"lead{h}", {}).setdefault("World", {})[s] = x.all()
            for s in ("foehn_p50_not_cams", "cams_not_foehn_p50", "foehn_p50_and_cams"):
                x = T[(h, t, "World", s)]
                out(f"  joint: {s:23s} warned {f(x.ci('warned_M'))} M, rightly {f(x.ci('right_M'))} M, falsely {f(x.ci('false_M'))} M, "
                    f"hidden {f(x.ci('hidden_warned_M'))} M, margin-hidden {f(x.ci('hidden_margin_warned_M'))} M")
                S[f"thr{t:g}"][f"lead{h}"]["World"][s] = x.all()
            out("  paired differences (first minus second):")
            D = S[f"thr{t:g}"][f"lead{h}"]["World"]["differences"] = {}
            for a_, b_ in PAIRS:
                xa, xb = T[(h, t, "World", a_)], T[(h, t, "World", b_)]
                D[f"{a_}-{b_}"] = xa.diffs(xb, DIFFM)
                D[f"{a_}/{b_}"] = xa.ratios(xb, ("right_M", "false_M", "hidden_warned_M", "warned_M"))
                d = D[f"{a_}-{b_}"]
                out(f"    {a_:11s} - {b_:12s}: warned {f(d['warned_M'])} M | rightly {f(d['right_M'])} M | falsely {f(d['false_M'])} M | "
                    f"reach {f(d['reach_pct'])} pts | correct {f(d['correct_pct'])} pts | hidden {f(d['hidden_warned_M'])} M, "
                    f"{f(d['hidden_warned_pct'])} pts | margin-hidden {f(d['hidden_margin_warned_pct'])} pts | false in silent cells "
                    f"{f(d['false_in_silent40_M'])} M")
            # the reference forecasts at the number of people that Foehn warns (their decision threshold moved)
            ref = T[(h, t, "World", "foehn_p50")]
            Q = S[f"thr{t:g}"][f"lead{h}"]["World"]["equal_number_warned"] = {}
            for sw in ("cams", "aurora", "foehn_mean"):
                mus = sorted(mult + [1.0])
                tabs = [T[(h, t, "World", sw if mu == 1.0 else f"{sw}@{mu:g}")] for mu in mus]
                Wp = np.array([x.pt["warned_M"] for x in tabs])                       # decreasing with mu
                if not (Wp.min() <= ref.pt["warned_M"] <= Wp.max()):
                    continue

                def at(target, vals_w, vals):                 # linear interpolation in the number warned (decreasing with mu)
                    w, v = vals_w[::-1], vals[::-1]               # increasing
                    if w.ndim == 1:
                        return np.interp(target, w, v)
                    i = np.clip((w < target[None]).sum(0), 1, len(w) - 1); j = np.arange(w.shape[1])
                    fr = (target - w[i - 1, j]) / (w[i, j] - w[i - 1, j])
                    return v[i - 1, j] + fr * (v[i, j] - v[i - 1, j])
                res = {}
                refbs = ref.bs; tbs = [x.bs for x in tabs]
                for m in ("right_M", "false_M", "hidden_warned_M", "hidden_margin_warned_M"):
                    pt = at(ref.pt["warned_M"], Wp, np.array([x.pt[m] for x in tabs]))
                    bs = at(refbs["warned_M"], np.stack([x["warned_M"] for x in tbs]), np.stack([x[m] for x in tbs]))
                    res[m] = tri(pt, bs); res[f"foehn_p50_minus_{m}"] = tri(ref.pt[m] - pt, refbs[m] - bs)
                res["decision_threshold_factor"] = float(np.interp(ref.pt["warned_M"], Wp[::-1], np.array(mus)[::-1]))
                Q[sw] = res
                out(f"  {sw} at Foehn's number of people warned ({ref.pt['warned_M']:.0f} M; its threshold x {res['decision_threshold_factor']:.2f}): "
                    f"rightly {f(res['right_M'])} M (Foehn {f(res['foehn_p50_minus_right_M'])} M more), hidden {f(res['hidden_warned_M'])} M "
                    f"(Foehn {f(res['foehn_p50_minus_hidden_warned_M'])} M more), margin-hidden {f(res['hidden_margin_warned_M'])} M "
                    f"(Foehn {f(res['foehn_p50_minus_hidden_margin_warned_M'])} M more)")
            # the reference forecasts at Foehn's share of correct warnings (their decision threshold raised until their
            # warnings are as often correct as Foehn's), and Foehn's ensemble mean at theirs (its threshold lowered)
            QC = S[f"thr{t:g}"][f"lead{h}"]["World"]["equal_correct_share"] = {}

            def at_inc(target, w, v):                         # linear interpolation, w increasing with mu
                if w.ndim == 1:
                    return np.interp(target, w, v)
                i = np.clip((w < target[None]).sum(0), 1, len(w) - 1); j = np.arange(w.shape[1])
                fr = (target - w[i - 1, j]) / (w[i, j] - w[i - 1, j])
                return v[i - 1, j] + fr * (v[i, j] - v[i - 1, j])
            mus = sorted(mult + [1.0]); refbs = ref.bs
            for moved, fixed in (("cams", "foehn_p50"), ("aurora", "foehn_p50"), ("foehn_mean", "cams"), ("foehn_mean", "aurora")):
                tabs = [T[(h, t, "World", moved if mu == 1.0 else f"{moved}@{mu:g}")] for mu in mus]
                fx_ = T[(h, t, "World", fixed)]; fbs = fx_.bs; tbs = [x.bs for x in tabs]
                Cp = np.array([x.pt["correct_pct"] for x in tabs])
                if not (Cp.min() <= fx_.pt["correct_pct"] <= Cp.max()) or (np.diff(Cp) <= 0).any():
                    continue
                res = {"correct_pct": fx_.ci("correct_pct"), "decision_threshold_factor": float(np.interp(fx_.pt["correct_pct"], Cp, np.array(mus)))}
                for m in ("warned_M", "right_M", "false_M", "reach_pct", "hidden_warned_M", "hidden_warned_pct", "hidden_margin_warned_pct"):
                    pt = at_inc(fx_.pt["correct_pct"], Cp, np.array([x.pt[m] for x in tabs]))
                    bs = at_inc(fbs["correct_pct"], np.stack([x["correct_pct"] for x in tbs]), np.stack([x[m] for x in tbs]))
                    res[m] = tri(pt, bs); res[f"{fixed}_{m}"] = fx_.ci(m); res[f"moved_minus_fixed_{m}"] = tri(pt - fx_.pt[m], bs - fbs[m])
                    if m in ("right_M", "hidden_warned_M"):
                        res[f"moved_over_fixed_{m}"] = tri(pt / fx_.pt[m], bs / fbs[m])
                QC[f"{moved}_at_correct_share_of_{fixed}"] = res
                out(f"  {moved} at the share of correct warnings of {fixed} ({fx_.pt['correct_pct']:.1f} %; threshold x {res['decision_threshold_factor']:.2f}): "
                    f"warned {f(res['warned_M'])} M, reach {f(res['reach_pct'])} % (fixed {fx_.pt['reach_pct']:.1f}), hidden {f(res['hidden_warned_M'])} M = "
                    f"{f(res['hidden_warned_pct'])} % (fixed {fx_.pt['hidden_warned_M']:.1f} M, {fx_.pt['hidden_warned_pct']:.1f} %); moved minus fixed: "
                    f"rightly {f(res['moved_minus_fixed_right_M'])} M, hidden {f(res['moved_minus_fixed_hidden_warned_M'])} M; ratio hidden {f(res['moved_over_fixed_hidden_warned_M'], 2)}")
            SW = S[f"thr{t:g}"][f"lead{h}"]["World"]["decision_threshold_sweep"] = {}
            for sw in ("foehn_mean", "cams", "aurora"):
                SW[sw] = {f"{mu:g}": {m: T[(h, t, "World", sw if mu == 1.0 else f"{sw}@{mu:g}")].pt[m]
                                      for m in ("warned_M", "right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M", "hidden_warned_pct")}
                          for mu in sorted(mult + [1.0])}
    # ---------------------------------------------------------------- regions
    for t in (37.5, 15.0, 75.0):
        for h in LEADS:
            out(f"\n=== Regions, > {t:g} ug m-3, {h} day{'s' if h > 1 else ''} ahead (populated pixels; G: = country groups, a partition)")
            out(f"  {'region':36s} {'exposed M':>9s} {'hidden M':>20s} {'hid/exp %':>9s} | hidden warned: {'Foehn %':>18s} {'Foehn M':>7s} {'CAMS %':>18s} "
                f"{'Aurora %':>8s} {'F40km %':>8s} | reach: {'Foehn':>6s} {'CAMS':>6s} {'Aurora':>6s} | correct: {'Foehn':>6s} {'CAMS':>6s} {'Aurora':>6s} | "
                f"{'Foehn - CAMS rightly M':>24s}")
            for r in reg:
                x = {s: T[(h, t, r, s)] for s in ("foehn_p50", "foehn_mean", "foehn_40km", "cams", "aurora", "perfect_40km")}
                S[f"thr{t:g}"][f"lead{h}"][r] = {s: v.all() for s, v in x.items()} if r != "World" else S[f"thr{t:g}"][f"lead{h}"][r]
                if r != "World":
                    S[f"thr{t:g}"][f"lead{h}"][r]["differences"] = {
                        "foehn_p50-cams": x["foehn_p50"].diffs(x["cams"], DIFFM),
                        "foehn_p50-aurora": x["foehn_p50"].diffs(x["aurora"], DIFFM)}
                a = x["foehn_p50"]
                if a.pt["exposed_M"] < 0.05:
                    continue
                out(f"  {r:36s} {a.pt['exposed_M']:9.1f} {f(a.ci('hidden_M')):>20s} {a.pt['hidden_pct_of_exposed']:9.1f} | "
                    f"{'':15s}{f(a.ci('hidden_warned_pct')):>18s} {a.pt['hidden_warned_M']:7.1f} {f(x['cams'].ci('hidden_warned_pct')):>18s} "
                    f"{x['aurora'].pt['hidden_warned_pct']:8.1f} {x['foehn_40km'].pt['hidden_warned_pct']:8.1f} | {'':6s} {a.pt['reach_pct']:6.1f} "
                    f"{x['cams'].pt['reach_pct']:6.1f} {x['aurora'].pt['reach_pct']:6.1f} | {'':8s} {a.pt['correct_pct']:6.1f} {x['cams'].pt['correct_pct']:6.1f} "
                    f"{x['aurora'].pt['correct_pct']:6.1f} | {f(a.diff(x['cams'], 'right_M')):>24s}")
    # ---------------------------------------------------------------- reference forecasts with a regional decision threshold
    # chosen in hindsight: in each country group (a partition of the world) the factor mu (the forecast warns above mu x
    # threshold; mu = 0.5 ... 2) that maximises the critical success index of that forecast in that region over the same
    # days. An upper bound of what a regional recalibration of the decision threshold can give (Foehn is left untouched).
    gi = [g for g, r in enumerate(reg) if r.startswith("G:")]
    mus = sorted(mult + [1.0])
    O = N["regional_decision_threshold"] = {"note": "factor chosen in hindsight per country group to maximise the critical success index; "
                                                    "Foehn (2 of 4 members) is not tuned"}
    out("\n=== Reference forecasts with the decision threshold chosen in hindsight in each of the country groups (best CSI), World")
    for t in (37.5, 15.0, 75.0):
        k = thr.index(t)
        for h in LEADS:
            sel = SEL[h]; cnt = mbb(len(sel), 4); ref = T[(h, t, "World", "foehn_p50")]
            for sw in ("cams", "aurora"):
                A = np.zeros((len(sel), 16)); chosen = {}
                for g in gi:
                    best = None
                    for mu in mus:
                        a = T1[sel, g, k, sys1.index(sw if mu == 1.0 else f"{sw}@{mu:g}")]
                        p = parts(a.sum(0)); csi = p["tp"] / max(p["exp"] + p["fa"], 1e-9)
                        if best is None or csi > best[0]:
                            best = (csi, mu, a)
                    A += best[2]; chosen[reg[g][2:]] = best[1]
                x = Tab(A, cnt); d = ref.diffs(x, DIFFM)
                O.setdefault(f"thr{t:g}", {}).setdefault(f"lead{h}", {})[sw] = {"metrics": x.all(), "factor_by_group": chosen, "foehn_p50_minus": d}
                out(f"  > {t:g}, {h} d, {sw:6s}: warned {x.pt['warned_M']:.0f} M, rightly {f(x.ci('right_M'))} M, falsely {f(x.ci('false_M'))} M, reach {f(x.ci('reach_pct'))} %, "
                    f"correct {f(x.ci('correct_pct'))} %, hidden {f(x.ci('hidden_warned_M'))} M ({f(x.ci('hidden_warned_pct'))} %), csi {f(x.ci('csi'), 3)} | Foehn minus it: "
                    f"rightly {f(d['right_M'])} M, falsely {f(d['false_M'])} M, reach {f(d['reach_pct'])} pts, correct {f(d['correct_pct'])} pts, hidden {f(d['hidden_warned_M'])} M, "
                    f"csi {f(d['csi'], 3)} | factors " + ", ".join(f"{n_[:12]} {v:g}" for n_, v in chosen.items()))
    # ---------------------------------------------------------------- lead ladder (0.25 deg block means of Foehn; CAMS; Aurora)
    T2, sys2, leads2 = z["T2"], list(z["sys2"]), [int(v) for v in z["leads2"]]
    okd = np.isfinite(T2[:, 0, 0, :, 0, 0]).all(1)                                    # targets with the four leads of Foehn
    cnt2 = mbb(int(okd.sum()), 4)
    L = N["lead_ladder"] = {"n_target_days": int(okd.sum()), "note": "target days of the 1-day forecasts for which the forecasts started "
                            "1, 3, 5 and 7 days before exist; Foehn = ensemble mean averaged over 0.25 deg blocks (the 1 km members at "
                            "the residents' pixels are stored for leads 1 and 3 only)"}
    for t in (37.5, 15.0, 75.0):
        k = thr.index(t)
        out(f"\n=== Lead ladder, World, > {t:g} ug m-3, {int(okd.sum())} common target days: reach % / correct % / hidden warned % (M a day)")
        X = {}
        for s in sys2:
            row = []
            for li, ld in enumerate(leads2):
                a = T2[okd, 0, k, li, sys2.index(s)]
                if not np.isfinite(a).all():
                    continue
                x = X[(s, ld)] = Tab(a, cnt2)
                L.setdefault(f"thr{t:g}", {}).setdefault(s, {})[f"lead{ld}"] = x.all()
                row.append(f"{ld} d: {f(x.ci('reach_pct'))} / {f(x.ci('correct_pct'))} / {f(x.ci('hidden_warned_pct'))} ({x.pt['hidden_warned_M']:.1f} M; "
                           f"rightly {x.pt['right_M']:.0f} M, falsely {x.pt['false_M']:.0f} M)")
            out(f"  {s:12s} " + " | ".join(row))
        # Foehn (0.25 deg block means) at each lead against the reference forecasts at the same lead and at one day (paired)
        D = L[f"thr{t:g}"]["differences"] = {}
        for ref in ("cams", "aurora"):
            for ld, lr in ((1, 1), (3, 3), (5, 5), (3, 1), (5, 1), (7, 1)):
                d = D[f"foehn_25km_lead{ld}-{ref}_lead{lr}"] = X[("foehn_25km", ld)].diffs(X[(ref, lr)], DIFFM)
                out(f"    Foehn 0.25 deg, {ld} d - {ref}, {lr} d: rightly {f(d['right_M'])} M | falsely {f(d['false_M'])} M | reach {f(d['reach_pct'])} pts | "
                    f"correct {f(d['correct_pct'])} pts | csi {f(d['csi'], 3)}")
    return T


def boot_days(v, L=7, seed=1):
    """mean over the 365 days of 2022 of a daily series (or of a ratio of two), with a moving-block interval (blocks of 7 days)"""
    cnt = mbb(len(v[0]) if isinstance(v, tuple) else len(v), L, seed)
    if isinstance(v, tuple):
        return tri(v[0].sum() / v[1].sum(), (cnt @ v[0]) / (cnt @ v[1]))
    return tri(v.mean(), cnt @ v / len(v))


def exposure(N):
    """the count of hidden people itself (all pixels with residents, 365 days): boxes of the paper, country groups, sensitivity"""
    e = np.load(FIG / "data" / "exposure_2022.npz"); res, R, thr = e["res"], [str(r) for r in e["regions"]], [float(t) for t in e["thr"]]
    X = N["exposure_all_pixels"] = {"source": "figures/data/exposure_2022.npz (figdata_exposure.py), 365 days, every pixel with residents",
                                    "interval": "moving-block bootstrap over the days of 2022 (blocks of 7 days), 95 %"}
    out("\n=== The count itself, all pixels with residents, 365 days of 2022 (exposure_2022.npz): people a day (millions)")
    out(f"  {'threshold':>9s} {'exposed at 1 km':>24s} {'above at 40 km':>14s} {'hidden':>22s} {'% of exposed':>19s} {'reverse':>20s} {'net':>7s} "
        f"{'daily hidden: min / p5 / median / p95 / max':>44s}")
    for k, t in enumerate(thr):
        a1, a4, hid, rev = (res[:, 0, j, k] for j in range(4))
        d = dict(exposed_M=boot_days(a1 / 1e6), above40_M=boot_days(a4 / 1e6), hidden_M=boot_days(hid / 1e6), reverse_M=boot_days(rev / 1e6),
                 hidden_pct=[100 * v for v in boot_days((hid, a1))], net_M=boot_days((hid - rev) / 1e6),
                 daily_hidden_M=[float(v) for v in (hid.min() / 1e6, np.percentile(hid, 5) / 1e6, np.median(hid) / 1e6, np.percentile(hid, 95) / 1e6, hid.max() / 1e6)],
                 day_min=int(hid.argmin()), day_max=int(hid.argmax()))
        X[f"thr{t:g}"] = {"World": d}
        out(f"  {t:9g} {f(d['exposed_M']):>24s} {d['above40_M'][0]:14.1f} {f(d['hidden_M']):>22s} {f(d['hidden_pct'], 1, True):>19s} {f(d['reverse_M']):>20s} "
            f"{d['net_M'][0]:7.1f} " + " / ".join(f"{v:.0f}" for v in d["daily_hidden_M"]))
    for t in (37.5, 15.0, 75.0):
        k = thr.index(t)
        out(f"\n  boxes of the paper, > {t:g}: hidden M a day, share of the world's hidden, share of the box's exposed people who are hidden")
        tot = res[:, 0, 2, k].sum()
        for g, r in enumerate(R):
            hid, a1 = res[:, g, 2, k], res[:, g, 0, k]
            d = dict(hidden_M=boot_days(hid / 1e6), share_of_world_hidden_pct=100 * hid.sum() / tot, exposed_M=float(a1.mean() / 1e6),
                     hidden_pct_of_exposed=[100 * v for v in boot_days((hid, a1))] if a1.sum() > 0 else None, reverse_M=float(res[:, g, 3, k].mean() / 1e6))
            X[f"thr{t:g}"].setdefault("boxes", {})[r] = d
            out(f"    {r:16s} {f(d['hidden_M']):>22s}  {d['share_of_world_hidden_pct']:5.1f} %  exposed {d['exposed_M']:7.1f} M, hidden "
                + (f"{f(d['hidden_pct_of_exposed'], 1, True)}" if d["hidden_pct_of_exposed"] else "-"))
        out(f"    the eight boxes sum to {100 * res[:, 1:, 2, k].sum() / tot:.1f} % of the world count (Africa and Middle East overlap)")
    p = HERE / "data" / "exposure_sens_2022.npz"
    if not p.exists():
        out("\n(exposure_sens_2022.npz is not there yet: the sensitivity table and the country groups are not written)")
        return
    z = np.load(p); tab, marg, stn = z["tab"], z["marg"], z["stn"]
    REG, var, margins = [str(r) for r in z["regions"]], [tuple(int(x) for x in v) for v in z["variants"]], [str(m) for m in z["margins"]]
    Sx = N["sensitivity"] = {"source": "data/exposure_sens_2022.npz (figdata_exposure_sens.py), 365 days, every pixel with residents",
                             "check": f"worst relative deviation from exposure_2022.npz {float(z['worst']):.1e}",
                             "residents_outside_shifted_blocks_M": {f"{F_}px_shift{s_}": float(z["lost"][:, v].mean() / 1e6) for v, (F_, s_) in enumerate(var)}}
    out(f"\n=== Sensitivity of the count (exposure_sens_2022.npz; worst relative deviation from exposure_2022.npz {float(z['worst']):.1e})")
    for k, t in enumerate(thr):
        d = Sx[f"thr{t:g}"] = {}
        out(f"\n  > {t:g} ug m-3, world, people a day (millions)")
        for v, (F_, s_) in enumerate(var):
            a = tab[:, v, 0, k]                                                  # neither, hidden, reverse, both
            hid, rev, ex = a[:, 1], a[:, 2], a[:, 1] + a[:, 3]
            key = f"blocks_{F_}px" + (f"_shift{s_}" if s_ else "")
            d[key] = dict(hidden_M=boot_days(hid / 1e6), reverse_M=boot_days(rev / 1e6), exposed_M=float(ex.mean() / 1e6),
                          hidden_pct=[100 * x for x in boot_days((hid, ex))],
                          daily_hidden_M=[float(x) for x in (hid.min() / 1e6, np.percentile(hid, 5) / 1e6, np.median(hid) / 1e6, np.percentile(hid, 95) / 1e6, hid.max() / 1e6)])
            out(f"    blocks of {F_:2d} pixels, origin shifted by {s_:2d}: hidden {f(d[key]['hidden_M']):>22s} ({f(d[key]['hidden_pct'], 1, True)} of "
                f"{d[key]['exposed_M']:.0f} M exposed), reverse {f(d[key]['reverse_M'])}, daily range {d[key]['daily_hidden_M'][0]:.0f}-{d[key]['daily_hidden_M'][-1]:.0f}")
        for j, m in enumerate(margins):
            kind, val = m.split(":")
            lab = f"{100 * float(val):g} % of the threshold" if kind == "rel" else f"{float(val):g} ug m-3"
            d[f"margin_{m}"] = dict(two_sided_M=boot_days(marg[:, 0, k, j, 0] / 1e6), pixel_only_M=boot_days(marg[:, 0, k, j, 1] / 1e6))
            out(f"    margin {lab:24s}: pixel above thr + d and cell not above thr - d {f(d[f'margin_{m}']['two_sided_M']):>22s}; "
                f"pixel above thr + d, cell not above thr {f(d[f'margin_{m}']['pixel_only_M'])}")
        base = tab[:, 0, 0, k]
        for q, lab in enumerate(("in the 1 km pixel of a monitor", "within 2 pixels of a monitor")):
            ab, hd = stn[:, 0, k, q, 0], stn[:, 0, k, q, 1]
            d[f"monitors_{q}"] = dict(label=lab, exposed_M=float(ab.mean() / 1e6), hidden_M=float(hd.mean() / 1e6),
                                      share_of_hidden_pct=float(100 * hd.sum() / base[:, 1].sum()),
                                      share_of_exposed_pct=float(100 * ab.sum() / (base[:, 1] + base[:, 3]).sum()),
                                      hidden_without_M=float((base[:, 1] - hd).mean() / 1e6))
            out(f"    residents {lab}: {d[f'monitors_{q}']['hidden_M']:.2f} M of the hidden ({d[f'monitors_{q}']['share_of_hidden_pct']:.2f} %), "
                f"{d[f'monitors_{q}']['share_of_exposed_pct']:.2f} % of the exposed; count without them {d[f'monitors_{q}']['hidden_without_M']:.1f} M")
    for t in (37.5, 15.0, 75.0):
        k = thr.index(t)
        out(f"\n  country groups (a partition of the world), > {t:g}: hidden M a day, share of the world's hidden, hidden share of the group's exposed, "
            f"with 10 km blocks M, with a two-sided 10 % margin M")
        tot = tab[:, 0, 0, k, 1].sum(); G = Sx[f"thr{t:g}"]["groups"] = {}
        for g, r in enumerate(REG):
            if not r.startswith("G:"):
                continue
            hid, ex = tab[:, 0, g, k, 1], tab[:, 0, g, k, 1] + tab[:, 0, g, k, 3]
            G[r[2:]] = dict(hidden_M=boot_days(hid / 1e6), share_of_world_hidden_pct=float(100 * hid.sum() / tot), exposed_M=float(ex.mean() / 1e6),
                            hidden_pct_of_exposed=[100 * x for x in boot_days((hid, ex))] if ex.sum() > 0 else None,
                            reverse_M=float(tab[:, 0, g, k, 2].mean() / 1e6), hidden_10km_M=float(tab[:, 2, g, k, 1].mean() / 1e6),
                            hidden_25km_M=float(tab[:, 1, g, k, 1].mean() / 1e6), hidden_margin10pct_M=float(marg[:, g, k, 1, 0].mean() / 1e6))
            x = G[r[2:]]
            out(f"    {r[2:]:34s} {f(x['hidden_M']):>22s} {x['share_of_world_hidden_pct']:5.1f} %  "
                + (f"{f(x['hidden_pct_of_exposed'], 1, True):>22s}" if x["hidden_pct_of_exposed"] else f"{'-':>22s}")
                + f"  exposed {x['exposed_M']:7.1f}  10 km {x['hidden_10km_M']:6.1f}  margin {x['hidden_margin10pct_M']:6.1f}")
        af = [G[n]["hidden_M"][0] for n in ("North Africa", "Sub-Saharan Africa")]
        out(f"    Africa as a continent (North + Sub-Saharan): {sum(af):.1f} M a day, {G['North Africa']['share_of_world_hidden_pct'] + G['Sub-Saharan Africa']['share_of_world_hidden_pct']:.1f} %")


def income(N):
    z = np.load(FIG / "data" / "exposure_income_2022.npz"); T = [float(t) for t in z["thr"]]; nd = int(z["ndays"])
    split = lambda a, k: (a[..., k + 1:, :].sum((-1, -2)), a[..., k + 1:, :k + 1].sum((-1, -2)))
    I = N["income"] = {"source": "figures/data/exposure_income_2022.npz (figdata_exposure_income.py): World Bank groups FY2024, GDP per capita "
                                 "PPP 2022 of Kummu et al. 2025 at the second administrative level, GHS-POP 2020"}
    out("\n=== Income (exposure_income_2022.npz)")
    for t in (37.5, 75.0, 15.0):
        k = T.index(t); e, h = split(z["wb"], k); pop = z["pop_wb"]; tot = h.sum()
        d = I[f"thr{t:g}"] = {"world_hidden_M": float(tot / nd / 1e6), "groups": {}}
        out(f"  > {t:g}: hidden {tot / nd / 1e6:.1f} M a day")
        for c, nm in enumerate(list(z["groups"]) + ["not classified"]):
            d["groups"][str(nm)] = dict(residents_bn=float(pop[c] / 1e9), hidden_days_per_resident=float(h[c] / pop[c]), exposed_days_per_resident=float(e[c] / pop[c]),
                                        hidden_M=float(h[c] / nd / 1e6), share_of_world_hidden_pct=float(100 * h[c] / tot), hidden_pct_of_exposed=float(100 * h[c] / max(e[c], 1)))
            x = d["groups"][str(nm)]
            out(f"    {str(nm):15s} residents {x['residents_bn']:.2f} bn | days above {x['exposed_days_per_resident']:6.1f} | days hidden {x['hidden_days_per_resident']:5.2f} | "
                f"hidden {x['hidden_M']:6.1f} M a day = {x['share_of_world_hidden_pct']:4.1f} % of the world's | {x['hidden_pct_of_exposed']:4.1f} % of the group's exposure")
        e, h = split(z["cq"], k); pop = z["pop_cq"]; nc = len(pop) // 6
        e, h, pop = e.reshape(nc, 6), h.reshape(nc, 6), pop.reshape(nc, 6)
        # the recipe of figures/fig_income.py (panel c): countries of more than a million residents whose quintiles each hold
        # 10-30 % of their residents; the same country mix in every quintile, so differences are within countries
        share = pop[:, :5] / np.maximum(pop[:, :5].sum(1, keepdims=True), 1)
        good = (share.min(1) > 0.10) & (share.max(1) < 0.30) & (pop[:, :5].sum(1) > 1e6); good[0] = False
        d["within_country_quintiles_hidden_days"] = [float(v) for v in h[good, :5].sum(0) / pop[good, :5].sum(0)]
        d["within_country_quintiles_exposed_days"] = [float(v) for v in e[good, :5].sum(0) / pop[good, :5].sum(0)]
        rate = h[good, :5] / np.maximum(pop[good, :5], 1); okc = rate.sum(1) > 0
        d["countries"] = int(good.sum()); d["countries_with_hidden"] = int(okc.sum())
        d["countries_richest_fifth_more_hidden"] = int((rate[okc, 4] > rate[okc, 0]).sum())
        d["share_of_residents_in_these_countries_pct"] = float(100 * pop[good, :5].sum() / pop[1:, :5].sum())
        out(f"    within countries ({d['countries']} countries, {d['share_of_residents_in_these_countries_pct']:.0f} % of residents), hidden days per resident "
            f"by quintile of GDP per capita (poorest first): " + ", ".join(f"{v:.2f}" for v in d["within_country_quintiles_hidden_days"])
            + f" (days above: " + ", ".join(f"{v:.1f}" for v in d["within_country_quintiles_exposed_days"])
            + f"); richest fifth more often hidden than the poorest in {d['countries_richest_fifth_more_hidden']} of {d['countries_with_hidden']} countries")


def main():
    RES.mkdir(exist_ok=True)
    N = {}
    only = sys.argv[1:] or ["systems", "exposure", "income"]
    old = RES / "numbers.json"
    if old.exists() and len(only) < 3:
        N = json.loads(old.read_text())
    for name in only:
        {"systems": systems, "exposure": exposure, "income": income}[name](N)
        (RES / f"tables_{name}.txt").write_text("\n".join(LINES) + "\n"); LINES.clear()
    old.write_text(json.dumps(N, indent=1))
    print("\nsaved", old)


if __name__ == "__main__":
    main()
