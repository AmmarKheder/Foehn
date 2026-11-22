#!/usr/bin/env python3
"""A14 class U, fix 4: the draw of a two-model gain (emissions: model with emissions e02 against its control ctrl;
transport: base model A against the network without transport C). Both models of a pair run on the same four noise fields
(checked here: correlation of the member departures for the same m). Paired delete-one-field jackknife (field i removed from
both models, 3-member estimators) of the RMSE gain in the as-run, eight-member and infinite forms, at the 8,110 station pixels
against GHAP (st_ens / st_ghap of the stored evaluations), per lead and for the lead pools of analyses/emissions/recompute.py,
against the moving-block bootstrap of the same gain (blocks of 7 start days, 10,000, seed 0). The ratio V_jack/V_boot is
carried to the land gains (rel_score.py) and to the 0.25 deg gains (c25): an approximation, the exact land jackknife needs
per-member land sums. Adapted from the reference a14/classU_review/judge/j_emis.py. No station is read.
    python emis_pair.py   -> results/emis_pair.json"""
import os  # FOEHN_WORK, FOEHN_ENVS: roots of the working trees and of the environments
import glob
import time

import numpy as np

from u_common import RES, safe_json

RUNS = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres")
EV = {"A": "_ghres/eval_final", "C": "_notransport/eval_final", "e02": "_ghres_ftemis12/eval_final_e02", "ctrl": "_ghres_ftemis12ctrl0/eval_final_e02"}
COMP = {"emissions": ("ctrl", "e02"), "transport": ("C", "A")}
POOLS = {"1-7": range(7), "1-3": range(0, 3), "5-7": range(4, 7)}
t0 = time.time()


def load(ev):
    d = {}
    for p in sorted(glob.glob(f"{RUNS}{ev}/part_*.npz")):
        z = np.load(p)
        for r, (s, h, t) in enumerate(z["keys"]):
            d[(int(s), int(h))] = (z["st_ens"][r].astype(np.float64), z["st_ghap"][r].astype(np.float64))
    return d


OUT = {"what": __doc__, "comparisons": {}}
for comp, (ref, mod) in COMP.items():
    Rm, Am = load(EV[ref]), load(EV[mod])
    keys = sorted(set(Rm) & set(Am))
    days = sorted({s for s, _ in keys}); di = {s: i for i, s in enumerate(days)}
    T = np.zeros((len(days), 7, 2, 5, 3))                  # day, lead, model (0 ref, 1 model), replicate (0 full, 1-4 field i-1 out), (n, sse, s2)
    cs = np.zeros((4, 4)); ca = np.zeros(4); cc = np.zeros(4)
    for (s, h) in keys:
        Er, g = Rm[(s, h)]; Ea, g2 = Am[(s, h)]
        ok = np.isfinite(g) & (g > 0) & np.isfinite(Er).all(1) & np.isfinite(Ea).all(1)
        y = g[ok]
        for k, E in enumerate((Er[ok], Ea[ok])):
            for r in range(5):
                X = E if r == 0 else np.delete(E, r - 1, 1)
                T[di[s], h - 1, k, r] += (ok.sum(), ((X.mean(1) - y) ** 2).sum(), X.var(1, ddof=1).sum())
        if s % 14 == 0:
            dr = Er[ok] - Er[ok].mean(1, keepdims=True); da = Ea[ok] - Ea[ok].mean(1, keepdims=True)
            cs += dr.T @ da; ca += (dr ** 2).sum(0); cc += (da ** 2).sum(0)
    corr = cs / np.sqrt(np.outer(ca, cc))
    nd = len(days); B = 7
    rng = np.random.default_rng(0)
    idx = (rng.integers(0, nd - B + 1, size=(10000, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(10000, -1)[:, :nd]

    def gain(S, form, M):
        def rm(k):
            n, sse, s2 = S[..., k, 0], S[..., k, 1], S[..., k, 2]
            if form == "as_run":
                return np.sqrt(sse / n)
            return np.sqrt((sse - s2 / M + (s2 / 8 if form == "M8" else 0.0)) / n)
        return 100 * (1 - rm(1) / rm(0))
    res = {"departure_corr_same_m": np.diag(corr).tolist(), "departure_corr_diff_m_mean": float(corr[~np.eye(4, dtype=bool)].mean()),
           "fields_shared": bool(np.diag(corr).min() > 0.9), "forecasts": len(keys), "start_days": nd, "by_lead": {}}
    for lk in [str(h + 1) for h in range(7)] + list(POOLS):
        hs = [int(lk) - 1] if lk.isdigit() else list(POOLS[lk])
        Th = T[:, hs].sum(1)                                 # (days, 2, 5, 3)
        o = {}
        for form in ("as_run", "M8", "inf"):
            full = float(gain(Th[:, :, 0].sum(0), form, 4))
            reps = np.array([gain(Th[:, :, r].sum(0), form, 3) for r in range(1, 5)])
            vj = float(3 / 4 * ((reps - reps.mean()) ** 2).sum())
            bs = gain(Th[:, :, 0][idx].sum(1), form, 4)
            vb = float(np.var(bs, ddof=1))
            o[form] = dict(gain=full, jk_reps=reps.tolist(), v_jack4=vj, v_boot=vb, ratio=vj / vb,
                           ci_boot=[float(x) for x in np.percentile(bs, [2.5, 97.5])])
        res["by_lead"][lk] = o
    OUT["comparisons"][comp] = res
    print(comp, "shared fields:", res["fields_shared"], np.round(np.diag(corr), 4), "| ratio by lead (inf):",
          {k: round(v["inf"]["ratio"], 3) for k, v in res["by_lead"].items()}, f"{time.time() - t0:.0f} s", flush=True)
safe_json(RES / "emis_pair.json", OUT)
print("done", flush=True)
