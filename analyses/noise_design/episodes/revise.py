#!/usr/bin/env python3
"""episodes_diag REVISION after two internal checks of 2026-11-21 (statistics: review_statistics/, data:
review_data/). README.md stays as pre-registered and is never edited; every change of estimator, definition or reading
is listed in results_revision.md ('Deviations'). Own implementation of every statistic below (from common.py only the
atomic JSON save and the bootstrap counts). Inputs: data/ext_2021.npz (S1, PRIMARY) and data/ext_2022.npz (S2, report
only; the extracts equal the raw part files, review_data/recompute_S1.json); s3check reads the stored Foehn-CAMS members
(val_e35/pred.npy) and the CAMS analyses. No station or monitor value anywhere.
  E5  estimator that stays unbiased when member m reuses ONE noise field for every run and lead (eval_global_green.py,
      tile_window(noise, m, ...)): UNCENTERED (globally centered) lag-k correlations of A = ln x - ln o and B = ln y - ln o
      (o = mean of the other 3 members); static (across-run) share of their second moments; the static map of member 0
      between runs, leads and years; de-diluted persistence of the truth's deviation from the infinite-ensemble mean
      (uncentered: with the static bias of the model; centered: its run-varying part = what a renewed noise must
      reproduce). The centered A-vs-B comparison of the README is kept for reference only (biased under fixed noise).
  SYN synthetic member xs = m exp(static map of ln x - ln m) (own year), xo = the same with the OTHER year's map (out of
      sample); emulated renewed members (statistical emulation, no model run) xe = m exp(sig z - sig^2/2), z AR(1) over
      leads drawn afresh for every run and cell, rho = 1 (xe1), rho = the truth's de-diluted run-varying lag-1
      correlation measured in pass 1 (xer), rho = 0 (xe0); sig^2 = 3/4 of the across-run mean of A^2 per cell and lead.
  E6  replaced by the end hazard conditioned on the HRES weather class of the NEXT day (inputs only, no selection on the
      outcome), first step and within forecast; the forward - reverse class test (0 under exchangeability of truth and
      member) and the symmetrised H0 reference of the end error conditional on the truth having ended.
  DIP end rules tolerant to one-day dips, for truth and forecasts alike: '2day' (ends on the first day e with z_e <= T
      and z_e+1 <= T; at the horizon z_7 <= T suffices) and 'hyst' (continues while z > 0.8 T); depth of the true end
      day and return of the truth above T on the 1 or 2 days after it.
  F1  first-step defects: onsets at lead 1 vs later leads; level / time-structure split of the survival excess.
  H5  equal horizon with S3 (5 pseudo-days): P1, P4, P5, P7a, P7b restricted to 5 days.
  CI  95 % percentile intervals: bootstrap over runs (S1 runs, S2 moving blocks of 4 runs; as README) and two-way runs x
      10-degree tiles (cells sorted by tile; the tile draw is shared by all runs of a draw).
    python revise.py test | S12 | s3check   [--stride k --nboot B --out DIR]"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EPS, ROOT, boot_counts, save_json  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
H, M = 7, 4
THR = {"37.5": 37.5, "75": 75.0, "P95": None}
LN06, LN14 = float(np.log(0.6)), float(np.log(1.4))
CLASSES = ("wet", "wind_rise", "light_rain", "quiet")
SER = ("y", "x", "m", "xs", "xo", "xe1", "xer", "xe0")
SER_C = ("y", "x", "m", "xer")                      # series of the weather-class hazard
NHIST = 20                                          # histogram of z_e / T on [0, 1], bins of 0.05
YEAR = {"S1": 2021, "S2": 2022}
BLOCK = {"S1": 1, "S2": 4}
SEED = 20261122


# ======================================================================== run-length rules and classes
def lead_run(ex):
    """ex (H, n) bool -> number of leading True (0..H)"""
    return np.cumprod(ex, axis=0, dtype=np.int8).sum(0, dtype=np.int64)


def run_len_2day(ex):
    """remaining duration when a single day <= T followed by a day > T does not end the episode: the end is the first
    day e with z_e <= T and z_e+1 <= T (e < H) or z_H <= T (e = H, the day after is unknown); R = e - 1, H if no end"""
    below = ~ex
    end = np.zeros_like(ex)
    end[:-1] = below[:-1] & below[1:]
    end[-1] = below[-1]
    return np.where(end.any(0), end.argmax(0), ex.shape[0]).astype(np.int64)


def first_true(ex):
    """1-based index of the first True along axis 0, n + 1 if none"""
    return np.where(ex.any(0), ex.argmax(0) + 1, ex.shape[0] + 1).astype(np.int64)


def wclass(Pmm, dW):
    """0 wet (P >= 5 mm), 1 wind rise (P < 5, dW >= 1.5 m/s), 2 light rain (1 <= P < 5, dW < 1.5), 3 quiet (P < 1,
    dW < 1.5): the classes of README E6, from the HRES inputs of the run (P of the day, dW = W(day) - W(day before))"""
    c = np.full(np.shape(Pmm), 3, np.int8)
    c[(Pmm >= 1) & (Pmm < 5)] = 2
    c[(Pmm < 5) & (dW >= 1.5)] = 1
    c[Pmm >= 5] = 0
    return c


# ======================================================================== aggregation
class Agg:
    """per-run sums of named per-cell columns: resident weights per (run, 10-deg tile), cell weights per run"""

    def __init__(self, N, starts, w):
        self.N, self.starts, self.w, self.g = N, starts, w, {}

    def add(self, r, group, cols):
        names = list(cols)
        Q = np.empty((len(self.w), len(names)))
        for j, v in enumerate(cols.values()):
            Q[:, j] = v
        if group not in self.g:
            self.g[group] = (names, np.zeros((self.N, len(self.starts), len(names))), np.zeros((self.N, len(names))))
        gn, T, Cc = self.g[group]
        assert gn == names, group
        T[r] = np.add.reduceat(Q * self.w[:, None], self.starts, axis=0)
        Cc[r] = Q.sum(0)

    def stack(self):
        names = [n for g in self.g.values() for n in g[0]]
        T = np.concatenate([g[1] for g in self.g.values()], -1)
        Cc = np.concatenate([g[2] for g in self.g.values()], -1)
        return names, T, Cc


# ======================================================================== per-run columns
def e5_cols(X, Y, Mn, maps, C):
    o = (M * Mn - X) / (M - 1)
    lo = np.log(np.maximum(o, EPS))
    AB = {"log": (np.log(np.maximum(X, EPS)) - lo, np.log(np.maximum(Y, EPS)) - lo), "lin": (X - o, Y - o)}
    c = {}
    for sp, (A, B) in AB.items():
        for mode in ("raw", "cen"):
            for nm, Z in (("A", A), ("B", B)):
                Z = Z.astype(np.float64) - (maps[f"cen{nm}_{sp}"] if mode == "cen" else 0.0)
                for k in (range(1, H) if sp == "log" else (1,)):
                    z1, z2 = Z[:H - k], Z[k:]
                    p = f"E5/{sp}/{mode}/{nm}/k{k}/"
                    c[p + "w"] = np.full(C, float(H - k))
                    c[p + "s1"], c[p + "s2"] = z1.sum(0), z2.sum(0)
                    c[p + "s11"], c[p + "s22"], c[p + "s12"] = (z1 * z1).sum(0), (z2 * z2).sum(0), (z1 * z2).sum(0)
                for h in range(H):
                    c[f"E5/{sp}/{mode}/{nm}/sq{h + 1}"] = Z[h] ** 2
    return c


def emulated(lm, sig, rho_e, rng, C):
    """renewed members (statistical emulation): m exp(sig z - sig^2/2), z AR(1) over leads, fresh per run and cell;
    common innovations for the three rho values (paired comparison)"""
    e0 = rng.standard_normal((H, C)).astype(np.float32)
    out = {}
    for name, rho in (("xe1", 1.0), ("xer", float(rho_e)), ("xe0", 0.0)):
        z = np.empty((H, C), np.float32)
        z[0] = e0[0]
        for h in range(1, H):
            z[h] = rho * z[h - 1] + np.float32(np.sqrt(max(1.0 - rho * rho, 0.0))) * e0[h]
        out[name] = np.exp(lm + sig * z - 0.5 * sig * sig)
    return out


def thr_cols(y0, S, P, W, Yext, Tc, f):
    """all threshold columns of one run (cells with f False contribute nothing)"""
    C = len(y0)
    ar = np.arange(C)
    Y = S["y"]
    ex = {s: S[s] > Tc for s in S}
    R = {s: lead_run(ex[s]) for s in S}
    c = {}
    # ---------------- E1 ongoing episodes
    u = f & (y0 > Tc)
    uf = u.astype(np.float64)
    c["E1/n"] = uf
    for s in S:
        if s != "y":
            c[f"E1/err_{s}"] = uf * (R[s] - R["y"])
    c["E1/err_pers"] = uf * (H - R["y"])
    for s in S:
        c[f"E1/S3_{s}"] = uf * (R[s] >= 3)
        c[f"E1/S7_{s}"] = uf * (R[s] >= H)
    for s in ("y", "x", "m"):
        for h in range(H):
            c[f"E1/marg{h + 1}_{s}"] = uf * ex[s][h]
    jidx = R["y"] * (H + 1) + R["x"]
    for j in range((H + 1) ** 2):
        c[f"E1/J{j}"] = uf * (jidx == j)
    for s in ("x", "m"):
        c[f"E1/errH5_{s}"] = uf * (np.minimum(R[s], 5) - np.minimum(R["y"], 5))
    c["E1/errH5_pers"] = uf * (5 - np.minimum(R["y"], 5))
    R2 = {s: run_len_2day(ex[s]) for s in ("y", "x", "m")}
    Rh = {s: lead_run(S[s] > 0.8 * Tc) for s in ("y", "x", "m")}
    for s in ("x", "m"):
        c[f"E1/err2_{s}"] = uf * (R2[s] - R2["y"])
        c[f"E1/errh_{s}"] = uf * (Rh[s] - Rh["y"])
    c["E1/err2_pers"] = uf * (H - R2["y"])
    c["E1/errh_pers"] = uf * (H - Rh["y"])
    for s in ("y", "x"):
        c[f"E1/S7_2_{s}"] = uf * (R2[s] >= H)

    # ---------------- E6 forward (class on the true end day) / reverse (class on the member end day)
    def end_class(Rv):
        e = np.minimum(Rv, H - 1)                    # end lead R + 1: P index R, W indices R + 1 and R
        return wclass(P[e, ar], W[e + 1, ar] - W[e, ar])

    errx = (R["x"] - R["y"]).astype(np.float64)
    errm = (R["m"] - R["y"]).astype(np.float64)
    cy, cx = end_class(R["y"]), end_class(R["x"])
    endy, endx = u & (R["y"] <= H - 1), u & (R["x"] <= H - 1)
    ye = Y[np.minimum(R["y"], H - 1), ar]           # truth on the true end day
    ratio = np.where(np.isfinite(Tc), ye / Tc, 0.0)
    nxt1 = Yext[np.minimum(R["y"] + 1, H + 1), ar]   # the day after the end day (d+8, d+9 from pm.npy)
    nxt2 = Yext[np.minimum(R["y"] + 2, H + 1), ar]
    ret1 = nxt1 > Tc
    ret2 = ret1 | (nxt2 > Tc)
    deep = ratio < 0.6
    hb = np.clip((ratio * NHIST).astype(np.int64), 0, NHIST - 1)
    for k, cn in enumerate(CLASSES):
        iy = (endy & (cy == k)).astype(np.float64)
        c[f"E6f/n_{cn}"] = iy
        c[f"E6f/errx_{cn}"], c[f"E6f/errm_{cn}"], c[f"E6f/errp_{cn}"] = iy * errx, iy * errm, iy * (H - R["y"])
        c[f"E6f/ge08_{cn}"], c[f"E6f/ret1_{cn}"], c[f"E6f/ret2_{cn}"] = iy * (ratio >= 0.8), iy * ret1, iy * ret2
        c[f"E6f/ratio_{cn}"], c[f"E6f/deep_{cn}"] = iy * ratio, iy * deep
        c[f"E6f/errx_deep_{cn}"], c[f"E6f/errx_ret2_{cn}"] = iy * deep * errx, iy * ret2 * errx
        for b in range(NHIST):
            c[f"E6f/hist{b}_{cn}"] = iy * (hb == b)
        ix = (endx & (cx == k)).astype(np.float64)
        c[f"E6r/n_{cn}"], c[f"E6r/err_{cn}"] = ix, ix * (R["y"] - R["x"])
    cy2, cx2 = end_class(R2["y"]), end_class(R2["x"])
    endy2, endx2 = u & (R2["y"] <= H - 1), u & (R2["x"] <= H - 1)
    for k, cn in enumerate(CLASSES):
        iy = (endy2 & (cy2 == k)).astype(np.float64)
        ix = (endx2 & (cx2 == k)).astype(np.float64)
        c[f"E6f2/n_{cn}"], c[f"E6f2/errx_{cn}"] = iy, iy * (R2["x"] - R2["y"])
        c[f"E6r2/n_{cn}"], c[f"E6r2/err_{cn}"] = ix, ix * (R2["y"] - R2["x"])

    # ---------------- E2 onsets
    v = f & (y0 <= Tc)
    vf = v.astype(np.float64)
    c["E2/n"] = vf
    O = {s: first_true(ex[s]) for s in S}
    for s in S:
        c[f"E2/lead1_{s}"] = vf * ex[s][0]
        c[f"E2/late_{s}"] = vf * (~ex[s][0] & ex[s][1:].any(0))
        c[f"E2/any_{s}"] = vf * (O[s] <= H)
        c[f"E2/any5_{s}"] = vf * (O[s] <= 5)
        if s != "y":
            hit = vf * ((O["y"] <= H) & (O[s] <= H))
            hit5 = vf * ((O["y"] <= 5) & (O[s] <= 5))
            c[f"E2/hit_{s}"], c[f"E2/tim_{s}"] = hit, hit * (O[s] - O["y"])
            c[f"E2/hit5_{s}"], c[f"E2/tim5_{s}"] = hit5, hit5 * (O[s] - O["y"])

    # ---------------- E3 / E4 transitions (first step: day d truth -> lead 1; within: lead t -> t + 1)
    cls_t = [wclass(P[t], W[t + 1] - W[t]) for t in range(H)]
    for s, Z in S.items():
        Zf = np.vstack([y0[None], Z])
        L = np.log(np.maximum(Zf, EPS))
        acc = {}
        for t in range(H):
            rk = (f & (Zf[t] > Tc)).astype(np.float64)
            dl = (L[t + 1] - L[t]).astype(np.float64)
            vals = {"risk": rk, "end": rk * (Zf[t + 1] <= Tc), "drop": rk * (dl < LN06), "rise": rk * (dl > LN14),
                    "d1": rk * dl, "d2": rk * dl ** 2, "d3": rk * dl ** 3}
            st = "first" if t == 0 else "within"
            for kk, val in vals.items():
                acc[f"E3/{st}_{kk}_{s}"] = acc.get(f"E3/{st}_{kk}_{s}", 0.0) + val
            if 1 <= t <= 4:
                for kk in ("risk", "end", "drop"):
                    acc[f"E3/w5_{kk}_{s}"] = acc.get(f"E3/w5_{kk}_{s}", 0.0) + vals[kk]
            if s in SER_C:
                for k, cn in enumerate(CLASSES):
                    ic = cls_t[t] == k
                    for kk in ("risk", "end"):
                        key = f"E3c/{st}_{kk}_{cn}_{s}"
                        acc[key] = acc.get(key, 0.0) + vals[kk] * ic
        c.update(acc)
    return c


# ======================================================================== statistics (functions of summed columns)
def corr_cov(q, D):
    w = D[q + "w"]
    m1, m2 = D[q + "s1"] / w, D[q + "s2"] / w
    cov = D[q + "s12"] / w - m1 * m2
    v1, v2 = D[q + "s11"] / w - m1 ** 2, D[q + "s22"] / w - m2 ** 2
    return cov / np.sqrt(v1 * v2), cov, v1, v2


def stats_e5(D):
    o = {}
    for sp in ("log", "lin"):
        for mode in ("raw", "cen"):
            for k in (range(1, H) if sp == "log" else (1,)):
                ra = corr_cov(f"E5/{sp}/{mode}/A/k{k}/", D)
                rb = corr_cov(f"E5/{sp}/{mode}/B/k{k}/", D)
                pp = f"E5/{sp}/{mode}/k{k}/"
                o[pp + "rho_A"], o[pp + "rho_B"], o[pp + "d_rho"] = ra[0], rb[0], ra[0] - rb[0]
                cy, v1y, v2y = rb[1] - ra[1] / M, rb[2] - ra[2] / M, rb[3] - ra[3] / M
                o[pp + "rho_y_dediluted"] = cy / np.sqrt(v1y * v2y)
                if k == 1:
                    o[pp + "var_ratio_member_over_truth_dediluted"] = (ra[2] * (M - 1) / M) / v1y
            qa = np.stack([D[f"E5/{sp}/{mode}/A/sq{h}"] for h in range(1, H + 1)], -1)
            qb = np.stack([D[f"E5/{sp}/{mode}/B/sq{h}"] for h in range(1, H + 1)], -1)
            o[f"E5/{sp}/{mode}/spread_over_error_by_lead"] = (qa * (M - 1) / M) / (qb - qa / M)
            sa, sb = qa.sum(-1), qb.sum(-1)
            o[f"E5/{sp}/{mode}/spread_over_error_all"] = (sa * (M - 1) / M) / (sb - sa / M)
            o[f"E5/{sp}/{mode}/level_ratio_A2_over_B2"] = sa / sb
    return o


def stats_thr(D, tk):
    G = lambda k: D[f"{tk}/{k}"]
    o = {}
    n = G("E1/n")
    # ---------------- E1
    for s in SER[1:]:
        o[f"P1_err_{s}"] = G(f"E1/err_{s}") / n
    o["P1_err_pers"] = G("E1/err_pers") / n
    for s in ("x", "m", "pers"):
        o[f"P1H5_err_{s}"] = G(f"E1/errH5_{s}") / n
        o[f"P1_2day_err_{s}"] = G(f"E1/err2_{s}") / n
        o[f"P1_hyst_err_{s}"] = G(f"E1/errh_{s}") / n
    for s in SER:
        o[f"S3_{s}"], o[f"S7_{s}"] = G(f"E1/S3_{s}") / n, G(f"E1/S7_{s}") / n
    for s in SER[1:]:
        o[f"P2_S3_{s}_minus_y"] = o[f"S3_{s}"] - o["S3_y"]
        o[f"S7_{s}_minus_y"] = o[f"S7_{s}"] - o["S7_y"]
    for s in ("y", "x", "m"):
        o[f"marg_{s}"] = np.stack([G(f"E1/marg{h}_{s}") for h in range(1, H + 1)], -1) / n[..., None]
    o["marg7_x_minus_y"] = o["marg_x"][..., -1] - o["marg_y"][..., -1]
    o["S7_x_minus_y_structure_part"] = o["S7_x_minus_y"] - o["marg7_x_minus_y"]
    o["S7_2day_x_minus_y"] = (G("E1/S7_2_x") - G("E1/S7_2_y")) / n
    J = np.stack([G(f"E1/J{j}") for j in range((H + 1) ** 2)], -1).reshape(n.shape + (H + 1, H + 1)) / n[..., None, None]
    o["share_77"], o["P_Ry_eq_7"] = J[..., H, H], J[..., H, :].sum(-1)
    Rv = np.arange(H + 1)
    dif = (Rv[None, :] - Rv[:, None]).astype(np.float64)              # member R - truth R
    endm = (Rv <= H - 1)[:, None] * np.ones((1, H + 1))
    Js = 0.5 * (J + np.swapaxes(J, -1, -2))
    o["E6cond_err_observed"] = (J * dif * endm).sum((-1, -2)) / (J * endm).sum((-1, -2))
    o["E6cond_err_H0_symmetrised"] = (Js * dif * endm).sum((-1, -2)) / (Js * endm).sum((-1, -2))
    o["E6cond_err_excess_over_H0"] = o["E6cond_err_observed"] - o["E6cond_err_H0_symmetrised"]
    o["mean_R_y"] = (J * Rv[:, None]).sum((-1, -2))
    # ---------------- E6 forward / reverse by class, dips
    nf = np.stack([G(f"E6f/n_{c}") for c in CLASSES], -1)
    nr = np.stack([G(f"E6r/n_{c}") for c in CLASSES], -1)
    sf = lambda k: np.stack([G(f"E6f/{k}_{c}") for c in CLASSES], -1)
    o["E6_share_of_true_ends"] = nf / nf.sum(-1, keepdims=True)
    o["E6_fwd_err_x"], o["E6_fwd_err_m"], o["E6_fwd_err_pers"] = sf("errx") / nf, sf("errm") / nf, sf("errp") / nf
    o["E6_rev_err_x"] = np.stack([G(f"E6r/err_{c}") for c in CLASSES], -1) / nr
    o["E6_fwd_minus_rev_x"] = o["E6_fwd_err_x"] - o["E6_rev_err_x"]
    o["E6_rev_share_of_member_ends"] = nr / nr.sum(-1, keepdims=True)
    o["dip_ge08_share"], o["dip_ret1_share"], o["dip_ret2_share"] = sf("ge08") / nf, sf("ret1") / nf, sf("ret2") / nf
    o["dip_ratio_mean"], o["dip_deep_share"] = sf("ratio") / nf, sf("deep") / nf
    o["dip_errx_shallow"] = (sf("errx") - sf("errx_deep")) / (nf - sf("deep"))
    o["dip_errx_deep"] = sf("errx_deep") / sf("deep")
    o["dip_errx_returning"] = sf("errx_ret2") / sf("ret2")
    o["dip_errx_not_returning"] = (sf("errx") - sf("errx_ret2")) / (nf - sf("ret2"))
    hist = np.stack([np.stack([G(f"E6f/hist{b}_{c}") for b in range(NHIST)], -1) for c in CLASSES], -2)  # (..., 4, NHIST)
    o["dip_ratio_hist"] = hist / hist.sum(-1, keepdims=True)
    tot = nf.sum(-1)
    o["dip_ret2_share_all"] = sf("ret2").sum(-1) / tot
    o["dip_ret1_share_all"] = sf("ret1").sum(-1) / tot
    o["dip_errx_returning_all"] = sf("errx_ret2").sum(-1) / sf("ret2").sum(-1)
    o["dip_errx_not_returning_all"] = (sf("errx").sum(-1) - sf("errx_ret2").sum(-1)) / (tot - sf("ret2").sum(-1))
    nf2 = np.stack([G(f"E6f2/n_{c}") for c in CLASSES], -1)
    nr2 = np.stack([G(f"E6r2/n_{c}") for c in CLASSES], -1)
    o["E6_2day_share_of_true_ends"] = nf2 / nf2.sum(-1, keepdims=True)
    o["E6_2day_fwd_minus_rev_x"] = (np.stack([G(f"E6f2/errx_{c}") for c in CLASSES], -1) / nf2
                                    - np.stack([G(f"E6r2/err_{c}") for c in CLASSES], -1) / nr2)
    # ---------------- E2
    n2 = G("E2/n")
    for s in SER:
        o[f"E2_P_lead1_{s}"], o[f"E2_P_late_{s}"] = G(f"E2/lead1_{s}") / n2, G(f"E2/late_{s}") / n2
        o[f"E2_P_any_{s}"] = G(f"E2/any_{s}") / n2
    for s in SER[1:]:
        o[f"E2_lead1_ratio_{s}"] = G(f"E2/lead1_{s}") / G("E2/lead1_y")
        o[f"E2_late_ratio_{s}"] = G(f"E2/late_{s}") / G("E2/late_y")
        o[f"P7b_{s}"] = G(f"E2/any_{s}") / G("E2/any_y") - 1
        o[f"P7a_{s}"] = G(f"E2/tim_{s}") / G(f"E2/hit_{s}")
        o[f"P7bH5_{s}"] = G(f"E2/any5_{s}") / G("E2/any5_y") - 1
        o[f"P7aH5_{s}"] = G(f"E2/tim5_{s}") / G(f"E2/hit5_{s}")
    # ---------------- E3 / E4
    for s in SER:
        for st in ("first", "within"):
            rk = G(f"E3/{st}_risk_{s}")
            o[f"haz_{st}_{s}"] = G(f"E3/{st}_end_{s}") / rk
            o[f"drop_{st}_{s}"], o[f"rise_{st}_{s}"] = G(f"E3/{st}_drop_{s}") / rk, G(f"E3/{st}_rise_{s}") / rk
            mu, e2, e3 = G(f"E3/{st}_d1_{s}") / rk, G(f"E3/{st}_d2_{s}") / rk, G(f"E3/{st}_d3_{s}") / rk
            var = e2 - mu ** 2
            o[f"sd_{st}_{s}"] = np.sqrt(var)
            o[f"skew_{st}_{s}"] = (e3 - 3 * mu * e2 + 2 * mu ** 3) / var ** 1.5
        rk5 = G(f"E3/w5_risk_{s}")
        o[f"haz_w5_{s}"], o[f"drop_w5_{s}"] = G(f"E3/w5_end_{s}") / rk5, G(f"E3/w5_drop_{s}") / rk5
    for s in SER[1:]:
        for k in ("haz_first", "haz_within", "drop_first", "drop_within", "rise_first", "rise_within", "sd_first",
                  "sd_within", "skew_first", "skew_within", "haz_w5", "drop_w5"):
            o[f"{k}_{s}_minus_y"] = o[f"{k}_{s}"] - o[f"{k}_y"]
    # ---------------- end hazard by the weather class of the next day (inputs only)
    for st in ("first", "within"):
        lam = {s: np.stack([G(f"E3c/{st}_end_{c}_{s}") / G(f"E3c/{st}_risk_{c}_{s}") for c in CLASSES], -1) for s in SER_C}
        for s in SER_C:
            o[f"hazc_{st}_{s}"] = lam[s]
            if s != "y":
                o[f"hazc_{st}_{s}_minus_y"], o[f"hazc_{st}_{s}_over_y"] = lam[s] - lam["y"], lam[s] / lam["y"]
        rky = np.stack([G(f"E3c/{st}_risk_{c}_y") for c in CLASSES], -1)
        o[f"hazc_{st}_risk_share_y"] = rky / rky.sum(-1, keepdims=True)
    return o


def stats_all(D):
    o = {f"{tk}/{k}": v for tk in THR for k, v in stats_thr(D, tk).items()}
    o.update(stats_e5(D))
    return o


# ======================================================================== bootstrap
def summarise(agg, N, block, nboot, seed, fn=stats_all, chunk=40):
    names, T, Cc = agg.stack()
    NT, K = T.shape[1], T.shape[2]
    mk = lambda arr: {n: arr[..., j] for j, n in enumerate(names)}
    est, est_cell = fn(mk(T.sum((0, 1)))), fn(mk(Cc.sum(0)))
    Kr = boot_counts(N, block, nboot, seed)
    rng = np.random.default_rng(seed + 7)
    Kt = np.stack([np.bincount(rng.integers(0, NT, NT), minlength=NT) for _ in range(nboot)]).astype(np.float64)
    runs = fn(mk(Kr @ T.sum(1)))
    Tf = T.reshape(N, NT * K)
    two = np.concatenate([np.einsum("btk,bt->bk", (Kr[b:b + chunk] @ Tf).reshape(-1, NT, K), Kt[b:b + chunk])
                          for b in range(0, nboot, chunk)])
    two = fn(mk(two))
    out = {}
    for k, v in est.items():
        out[k] = {"est": v, "est_cell": est_cell[k]}
        for nm, bs in (("runs", runs[k]), ("two_way", two[k])):
            b = np.asarray(bs)
            out[k][nm] = np.nanpercentile(b, [2.5, 97.5], axis=0) if np.isfinite(b).any() else [np.nan, np.nan]
    return out, {"columns": K, "tiles": NT}


# ======================================================================== data
def load(sample, stride):
    z = np.load(ROOT / "data" / f"ext_{YEAR[sample]}.npz")
    sl = slice(None, None, stride)
    assert int(z["H"]) == H and int(z["M"]) == M and int(z["Mx"]) == 1
    D = {"truth": z["truth"][:, sl], "tidx": z["run_tidx"], "Y": z["Y"][:, :, sl], "X": z["X"][:, 0, :, sl],
         "Mn": z["Mn"][:, :, sl], "P": z["P"][:, :, sl], "W": z["W"][:, :, sl], "pop": z["pop"][sl].astype(np.float64),
         "p95": z["p95"][sl], "lat": z["lat"][sl], "lon": z["lon"][sl], "labels": z["run_label"], "cells": z["cells"][sl]}
    y0 = D["truth"][D["tidx"]]
    D["ok"] = np.isfinite(y0) & np.isfinite(D["Y"]).all(1) & np.isfinite(D["X"]).all(1) & np.isfinite(D["Mn"]).all(1)
    return D


def restrict(D, keep, order):
    for k in ("truth", "Y", "X", "Mn", "P", "W"):
        D[k] = np.ascontiguousarray(D[k][..., keep][..., order])
    for k in ("pop", "p95", "lat", "lon", "cells"):
        D[k] = D[k][keep][order]


def wcorr(a, b, w):
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    return float((w * (a - ma) * (b - mb)).sum() / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum()))


def pass1(D):
    """across-run sums per (lead, cell) of A, B (log, lin) and ln x - ln m; lag-1 products; lead-1 maps; signs"""
    N, C = len(D["tidx"]), len(D["pop"])
    ks = ("A_log", "B_log", "A_lin", "B_lin", "Am")
    s1 = {k: np.zeros((H, C)) for k in ks}
    s2 = {k: np.zeros((H, C)) for k in ks}
    lag = {k: np.zeros((H - 1, C)) for k in ks[:4]}
    maps1 = {k: np.empty((N, C), np.float32) for k in ("A_log", "B_log")}
    pos = np.zeros((H, C))
    for r in range(N):
        X, Y, Mn = (D[k][r].astype(np.float32) for k in ("X", "Y", "Mn"))
        o = (M * Mn - X) / (M - 1)
        lo = np.log(np.maximum(o, EPS))
        lx = np.log(np.maximum(X, EPS))
        V = {"A_log": lx - lo, "B_log": np.log(np.maximum(Y, EPS)) - lo, "A_lin": X - o, "B_lin": Y - o,
             "Am": lx - np.log(np.maximum(Mn, EPS))}
        for k, v in V.items():
            v = v.astype(np.float64)
            s1[k] += v
            s2[k] += v * v
            if k in lag:
                lag[k] += v[:-1] * v[1:]
        for k in maps1:
            maps1[k][r] = V[k][0]
        pos += V["A_log"] > 0
    return {"N": N, "s1": s1, "s2": s2, "lag": lag, "maps1": maps1, "pos": pos}


def p1_summary(p, D):
    """point estimates of pass 1: static shares, de-diluted lag-1 persistence (raw, centered), run-pair map correlations"""
    N, w = p["N"], D["pop"]
    out, cen = {}, {}
    for k in p["s1"]:
        cen[k] = p["s1"][k] / N
    for wk, ww in (("res", w), ("cell", np.ones_like(w))):
        for k in ("A_log", "B_log", "A_lin", "B_lin"):
            tot = (p["s2"][k] * ww).sum()
            stat = (N * cen[k] ** 2 * ww).sum()
            out[f"static_share/{k}/{wk}"] = stat / tot
            out[f"second_moment_total/{k}/{wk}"] = tot / (N * H * ww.sum())
            out[f"second_moment_static/{k}/{wk}"] = stat / (N * H * ww.sum())
            out[f"second_moment_varying/{k}/{wk}"] = (tot - stat) / (N * H * ww.sum())

    def lag1(k, mode):
        """pooled (Sw, S1, S2, S11, S22, S12) of the lag-1 pairs, resident weights"""
        a = cen[k]
        if mode == "raw":
            s1v, s2v, s11, s22, s12 = p["s1"][k][:-1], p["s1"][k][1:], p["s2"][k][:-1], p["s2"][k][1:], p["lag"][k]
        else:
            s1v, s2v = np.zeros_like(a[:-1]), np.zeros_like(a[1:])
            s11, s22 = p["s2"][k][:-1] - N * a[:-1] ** 2, p["s2"][k][1:] - N * a[1:] ** 2
            s12 = p["lag"][k] - N * a[:-1] * a[1:]
        sw = N * (H - 1) * w.sum()
        f = lambda q: (q * w).sum() / sw
        m1, m2 = f(s1v), f(s2v)
        return f(s12) - m1 * m2, f(s11) - m1 ** 2, f(s22) - m2 ** 2
    for sp in ("log", "lin"):
        for mode in ("raw", "cen"):
            ca, va1, va2 = lag1(f"A_{sp}", mode)
            cb, vb1, vb2 = lag1(f"B_{sp}", mode)
            out[f"pass1_rho_A/{sp}/{mode}"] = ca / np.sqrt(va1 * va2)
            out[f"pass1_rho_B/{sp}/{mode}"] = cb / np.sqrt(vb1 * vb2)
            out[f"pass1_rho_y_dediluted/{sp}/{mode}"] = (cb - ca / M) / np.sqrt((vb1 - va1 / M) * (vb2 - va2 / M))
    # run-pair correlations of the lead-1 maps of A and B (resident weights) by separation of the start days
    lab = np.asarray(D["labels"])
    sep = np.abs(lab[:, None] - lab[None, :])
    for k in ("A_log", "B_log"):
        Z = p["maps1"][k].astype(np.float64)
        wn = w / w.sum()
        Z = Z - (Z * wn).sum(1, keepdims=True)
        Z = Z / np.sqrt((Z * Z * wn).sum(1, keepdims=True))
        R = (Z * wn) @ Z.T
        d = {}
        for s in sorted(set(sep[np.triu_indices(len(lab), 1)].tolist())):
            vals = R[(sep == s) & np.triu(np.ones_like(R, bool), 1)]
            d[int(s)] = [float(np.median(vals)), float(vals.min()), float(vals.max()), int(len(vals))]
        out[f"run_pair_corr_lead1/{k}"] = d
    maj = np.maximum(p["pos"], N - p["pos"]) / N
    out["same_side_share_residents_lead1_ge_8of9"] = float((w * (maj[0] >= 40 / 45 - 1e-9)).sum() / w.sum())
    out["same_side_share_residents_all_leads_ge_8of9"] = float((w[None] * (maj >= 40 / 45 - 1e-9)).sum() / (H * w.sum()))
    out["same_side_mean_majority_fraction_residents"] = float((w[None] * maj).sum() / (H * w.sum()))
    return out, cen


# ======================================================================== main sample loop
def run_sample(name, D, cen, other_map, rho_e, a, extra):
    N, C = len(D["tidx"]), len(D["pop"])
    tid = D["tid"]
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    agg = Agg(N, starts, D["pop"])
    maps = {"cenA_log": cen["A_log"], "cenB_log": cen["B_log"], "cenA_lin": cen["A_lin"], "cenB_lin": cen["B_lin"]}
    sig = np.sqrt(0.75 * D["s2A_log"] / N).astype(np.float32)
    own = cen["Am"].astype(np.float32)
    oth = other_map.astype(np.float32)
    nt = D["truth"].shape[0]
    t0 = time.time()
    for r in range(N):
        t = int(D["tidx"][r])
        y0 = D["truth"][t].astype(np.float32)
        Y, X, Mn = (D[k][r].astype(np.float32) for k in ("Y", "X", "Mn"))
        P, W = D["P"][r].astype(np.float32), D["W"][r].astype(np.float32)
        after = np.stack([D["truth"][t + k] if t + k < nt else np.full(C, np.nan, np.float32) for k in (8, 9)])
        lm = np.log(np.maximum(Mn, EPS))
        S = {"y": Y, "x": X, "m": Mn, "xs": np.exp(lm + own), "xo": np.exp(lm + oth)}
        S.update(emulated(lm, sig, rho_e, np.random.default_rng(SEED + 1000 * YEAR[name] + r), C))
        assert tuple(S) == SER
        Yext = np.vstack([Y, after])
        for tk, T in THR.items():
            Tc = D["p95"].astype(np.float32) if T is None else np.full(C, T, np.float32)
            f = np.isfinite(Tc)
            Tc = np.where(f, Tc, np.float32(np.inf))
            agg.add(r, tk, {f"{tk}/{k}": v for k, v in thr_cols(y0, S, P, W, Yext, Tc, f).items()})
        agg.add(r, "E5", e5_cols(X, Y, Mn, maps, C))
        if r % 10 == 0:
            print(f"{name} run {r + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    res, info = summarise(agg, N, BLOCK[name], a.nboot, SEED + YEAR[name])
    out = {"sample": name, "year": YEAR[name], "n_runs": N, "n_cells": C, "residents": float(D["pop"].sum()),
           "bootstrap": {"draws": a.nboot, "run_block": BLOCK[name], "tile_deg": 10, **info},
           "rho_emulation": float(rho_e), "stats": res, **extra}
    print(f"{name} summarised {time.time() - t0:.0f}s ({info})", flush=True)
    return out


def main_s12(a):
    t0 = time.time()
    out_root = Path(a.out) if a.out else ROOT
    D = {s: load(s, a.stride) for s in ("S1", "S2")}
    same = {s: bool((D[s]["ok"] == D[s]["ok"][0]).all()) for s in D}
    keep = D["S1"]["ok"].all(0) & D["S2"]["ok"].all(0)
    chk = {"same_complete_pattern_every_run": same, "kept_cells": int(keep.sum()),
           "complete_cells_S1": int(D["S1"]["ok"].all(0).sum()), "complete_cells_S2": int(D["S2"]["ok"].all(0).sum()),
           "cells_equal": bool(np.array_equal(D["S1"]["cells"], D["S2"]["cells"]))}
    assert chk["cells_equal"] and all(same.values()), chk
    lat, lon = D["S1"]["lat"][keep], D["S1"]["lon"][keep]
    t = ((84.0 - lat) // 10).astype(int) * 36 + ((lon + 180.0) // 10).astype(int)
    _, tid = np.unique(t, return_inverse=True)
    order = np.argsort(tid, kind="stable")
    for s in D:
        restrict(D[s], keep, order)
        D[s]["tid"] = tid[order]
    print(f"loaded {time.time() - t0:.0f}s {chk}", flush=True)
    P1, CEN, SUM = {}, {}, {}
    for s in D:
        P1[s] = pass1(D[s])
        SUM[s], CEN[s] = p1_summary(P1[s], D[s])
        D[s]["s2A_log"] = P1[s]["s2"]["A_log"]
        print(f"{s} pass 1 {time.time() - t0:.0f}s: static share A_log {SUM[s]['static_share/A_log/res']:.3f}, "
              f"rho_y de-diluted log raw {SUM[s]['pass1_rho_y_dediluted/log/raw']:.3f} cen "
              f"{SUM[s]['pass1_rho_y_dediluted/log/cen']:.3f}", flush=True)
    w = D["S1"]["pop"]
    years = {}
    for k in ("A_log", "B_log", "Am"):
        years[f"static_map_corr_2021_vs_2022_by_lead/{k}"] = [wcorr(CEN["S1"][k][h], CEN["S2"][k][h], w) for h in range(H)]
        years[f"static_map_corr_lead1_vs_lead7/{k}"] = [wcorr(CEN[s][k][0], CEN[s][k][H - 1], w) for s in ("S1", "S2")]
        years[f"static_map_rms/{k}"] = [float(np.sqrt((w[None] * CEN[s][k] ** 2).sum() / (H * w.sum()))) for s in ("S1", "S2")]
    for k in ("A_log", "B_log"):
        del P1["S1"]["maps1"][k], P1["S2"]["maps1"][k]
    print("years", {k: np.round(v, 4).tolist() for k, v in years.items()}, flush=True)
    tests = synthetic_tests()
    try:
        s3 = s3check(a)
    except Exception as e:                                                # the S3 design check must not stop S1 / S2
        s3 = {"error": repr(e)}
    print("s3check", s3, flush=True)
    for s, other in (("S1", "S2"), ("S2", "S1")):
        rho_e = SUM[s]["pass1_rho_y_dediluted/log/cen"]
        extra = {"checks": chk, "pass1": SUM[s], "years": years, "synthetic_tests": tests if s == "S1" else "see S1",
                 "S3_design_check": s3 if s == "S1" else "see S1"}
        res = run_sample(s, D[s], CEN[s], CEN[other]["Am"], rho_e, a, extra)
        res["seconds"] = time.time() - t0
        save_json(out_root / "results" / f"revision_{s}.json", res)
        print(f"saved revision_{s}.json {time.time() - t0:.0f}s", flush=True)


# ======================================================================== synthetic tests
def synthetic_tests(seed=5):
    """(a) run-length rules on hand-made sequences; (b)-(d) E5 estimators on simulated 4-member ensembles in log space:
    (b) H0 with FRESH noise every run, (c) H0 with each member's noise FIXED across runs (94 % static share; the truth,
    a new draw every run, is exchangeable with the members inside each run), (d) frozen members (as c) against a truth
    whose deviation renews with rho 0.6 (no static part)"""
    res, ok = {}, True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        res[name] = {"pass": bool(cond), "info": info}
        print(("PASS " if cond else "FAIL ") + name + " " + info, flush=True)
    seqs = {"dip at day 2": ([1, 0, 1, 1, 1, 1, 1], 7, 1), "two days below at 2-3": ([1, 0, 0, 1, 1, 1, 1], 1, 1),
            "end at the horizon": ([1, 1, 1, 1, 1, 1, 0], 6, 6), "dip at day 1": ([0, 1, 1, 1, 1, 1, 1], 7, 0),
            "all below": ([0, 0, 0, 0, 0, 0, 0], 0, 0), "dip at day 6": ([1, 1, 1, 1, 1, 0, 1], 7, 5),
            "all above": ([1, 1, 1, 1, 1, 1, 1], 7, 7)}
    for k, (sq, r2, r1) in seqs.items():
        ex = np.array(sq, bool)[:, None]
        check(f"run_len_2day {k}", int(run_len_2day(ex)[0]) == r2 and int(lead_run(ex)[0]) == r1,
              f"{int(run_len_2day(ex)[0])} / {int(lead_run(ex)[0])}")
    check("wclass", wclass(np.array([6, 2, 2, 0.5, 0.5, 4.9]), np.array([0, 0, 2, 0, 2, 1.49])).tolist() == [0, 2, 1, 3, 1, 2])
    rng = np.random.default_rng(seed)
    Cn, N = 3000, 45

    def ar1(rho, shape):
        z = np.empty(shape)
        z[..., 0, :] = rng.standard_normal(shape[:-2] + shape[-1:])
        for h in range(1, shape[-2]):
            z[..., h, :] = rho * z[..., h - 1, :] + np.sqrt(1 - rho ** 2) * rng.standard_normal(shape[:-2] + shape[-1:])
        return z

    def simulate(static_share, rho_s, rho_v, truth_mode):
        sig = 0.33
        s_fix = ar1(rho_s, (M, H, Cn))                                     # member noise imprint, fixed across runs
        runs = []
        for r in range(N):
            mu = np.log(40.0) + 0.7 * rng.standard_normal((1, Cn)) + np.cumsum(0.12 * rng.standard_normal((H, Cn)), 0)
            v = ar1(rho_v, (M, H, Cn))
            eps = np.sqrt(static_share) * s_fix + np.sqrt(1 - static_share) * v
            if truth_mode == "exchangeable":                               # a fresh member-like draw every run
                eta = np.sqrt(static_share) * ar1(rho_s, (1, H, Cn))[0] + np.sqrt(1 - static_share) * ar1(rho_v, (1, H, Cn))[0]
            else:                                                          # renewing truth, rho 0.6, no static part
                eta = ar1(0.6, (1, H, Cn))[0]
            x = np.exp(mu[None] + sig * eps)
            runs.append((x[0], np.exp(mu + sig * eta), x.mean(0)))
        # centering maps over the runs (as pass 1), then E5 columns summed over runs and cells
        acc = {}
        lgA = np.mean([np.log(np.maximum(X, EPS)) - np.log(np.maximum((M * Mn - X) / (M - 1), EPS)) for X, Y, Mn in runs], 0)
        lgB = np.mean([np.log(np.maximum(Y, EPS)) - np.log(np.maximum((M * Mn - X) / (M - 1), EPS)) for X, Y, Mn in runs], 0)
        liA = np.mean([X - (M * Mn - X) / (M - 1) for X, Y, Mn in runs], 0)
        liB = np.mean([Y - (M * Mn - X) / (M - 1) for X, Y, Mn in runs], 0)
        maps = {"cenA_log": lgA, "cenB_log": lgB, "cenA_lin": liA, "cenB_lin": liB}
        for X, Y, Mn in runs:
            for k, v in e5_cols(X, Y, Mn, maps, Cn).items():
                acc[k] = acc.get(k, 0.0) + v.sum()
        return stats_e5({k: np.asarray(v) for k, v in acc.items()})
    g = lambda o, k: float(np.asarray(o[k]))
    o = simulate(0.0, 0.99, 0.6, "exchangeable")
    check("E5 (b) H0 fresh noise: raw d_rho ~ 0", abs(g(o, "E5/log/raw/k1/d_rho")) < 0.02, f"{g(o, 'E5/log/raw/k1/d_rho'):+.4f}")
    check("E5 (b) H0 fresh noise: centered d_rho ~ 0", abs(g(o, "E5/log/cen/k1/d_rho")) < 0.03, f"{g(o, 'E5/log/cen/k1/d_rho'):+.4f}")
    check("E5 (b) H0 fresh noise: raw level ratio ~ 1", abs(g(o, "E5/log/raw/level_ratio_A2_over_B2") - 1) < 0.05,
          f"{g(o, 'E5/log/raw/level_ratio_A2_over_B2'):.3f}")
    o = simulate(0.94, 0.99, 0.6, "exchangeable")
    check("E5 (c) H0 fixed noise: raw d_rho ~ 0 (unbiased)", abs(g(o, "E5/log/raw/k1/d_rho")) < 0.02,
          f"{g(o, 'E5/log/raw/k1/d_rho'):+.4f}")
    check("E5 (c) H0 fixed noise: raw level ratio ~ 1", abs(g(o, "E5/log/raw/level_ratio_A2_over_B2") - 1) < 0.06,
          f"{g(o, 'E5/log/raw/level_ratio_A2_over_B2'):.3f}")
    check("E5 (c) H0 fixed noise: CENTERED level ratio biased (<< 1)", g(o, "E5/log/cen/level_ratio_A2_over_B2") < 0.2,
          f"{g(o, 'E5/log/cen/level_ratio_A2_over_B2'):.3f}")
    check("E5 (c) H0 fixed noise: CENTERED d_rho biased (< -0.1)", g(o, "E5/log/cen/k1/d_rho") < -0.1,
          f"{g(o, 'E5/log/cen/k1/d_rho'):+.4f}")
    res["(c) values"] = {k: g(o, k) for k in ("E5/log/raw/k1/d_rho", "E5/log/cen/k1/d_rho", "E5/log/raw/level_ratio_A2_over_B2",
                                              "E5/log/cen/level_ratio_A2_over_B2", "E5/log/raw/k1/rho_A", "E5/log/raw/k1/rho_B")}
    o = simulate(0.94, 0.99, 0.6, "renewing")
    check("E5 (d) frozen members vs renewing truth: raw d_rho > 0.2", g(o, "E5/log/raw/k1/d_rho") > 0.2,
          f"{g(o, 'E5/log/raw/k1/d_rho'):+.3f}")
    check("E5 (d) raw de-diluted rho_y ~ 0.6", abs(g(o, "E5/log/raw/k1/rho_y_dediluted") - 0.6) < 0.05,
          f"{g(o, 'E5/log/raw/k1/rho_y_dediluted'):.3f}")
    check("E5 (d) centered de-diluted rho_y ~ 0.6", abs(g(o, "E5/log/cen/k1/rho_y_dediluted") - 0.6) < 0.05,
          f"{g(o, 'E5/log/cen/k1/rho_y_dediluted'):.3f}")
    res["all_pass"] = ok
    print("ALL PASS" if ok else "SOME TESTS FAILED", flush=True)
    return res


# ======================================================================== S3 design check (numbers of the second check, other days)
def s3check(a):
    from common import CAMS_PRED, G, Npy
    z = np.load(ROOT / "data" / "ext_cams.npz")
    cells, popc = z["cells"], z["pop"].astype(np.float64)
    pred = Npy(CAMS_PRED / "pred.npy")
    an = Npy(G / "cams_series.npy")
    days = np.array(np.load(CAMS_PRED / "days.npy"), "datetime64[D]")
    lg = lambda v: np.log(np.maximum(v, EPS))
    out = {"rows": [], "member_dev_corr_12h_24h": [], "member_dev_corr_24h_36h": [], "truth_dev_corr_12h_24h": [],
           "member_dev_var_ratio_pseudoday_over_halfday": [], "truth_dev_var_ratio_pseudoday_over_halfday": []}
    for row in (10, 30, 50, 70, 90, 110):
        blk = np.stack([np.stack([pred.plane((row, m, k)).reshape(-1)[cells].astype(np.float32) for k in range(3)])
                        for m in range(8)])                                   # (8, 3, C): leads 12, 24, 36 h
        dev = lg(blk) - lg(blk.mean(0, keepdims=True))
        w = np.broadcast_to(popc, dev[:, 0].shape).ravel()
        t0 = 2 * int((days[row] - np.datetime64("2017-10-01")).astype(int))
        ana = [an.plane((t0 + 1 + k, 1)).reshape(-1)[cells].astype(np.float32) for k in range(2)]
        ty = [lg(ana[k]) - lg(blk[:, k].mean(0)) for k in range(2)]
        ty = [q - (popc * q).sum() / popc.sum() for q in ty]
        out["rows"].append(row)
        out["member_dev_corr_12h_24h"].append(wcorr(dev[:, 0].ravel(), dev[:, 1].ravel(), w))
        out["member_dev_corr_24h_36h"].append(wcorr(dev[:, 1].ravel(), dev[:, 2].ravel(), w))
        out["truth_dev_corr_12h_24h"].append(wcorr(ty[0], ty[1], popc))
        pdm = lg(0.5 * (blk[:, 0] + blk[:, 1])) - lg(0.5 * (blk[:, 0] + blk[:, 1]).mean(0, keepdims=True))
        ww = np.broadcast_to(popc, pdm.shape)
        out["member_dev_var_ratio_pseudoday_over_halfday"].append(
            float((ww * pdm ** 2).sum() / (0.5 * (ww * (dev[:, 0] ** 2 + dev[:, 1] ** 2)).sum())))
        tpd = lg(0.5 * (ana[0] + ana[1])) - lg(0.5 * (blk[:, 0] + blk[:, 1]).mean(0))
        tpd = tpd - (popc * tpd).sum() / popc.sum()
        out["truth_dev_var_ratio_pseudoday_over_halfday"].append(
            float((popc * tpd ** 2).sum() / (0.5 * (popc * (ty[0] ** 2 + ty[1] ** 2)).sum())))
        print(row, {k: (round(v[-1], 4) if isinstance(v[-1], float) else v[-1]) for k, v in out.items()}, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test", "S12", "s3check"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="smoke test: output root with results/ inside ($EPI_SCRATCH)")
    a = ap.parse_args()
    if a.cmd == "test":
        r = synthetic_tests()
        sys.exit(0 if r["all_pass"] else 1)
    elif a.cmd == "s3check":
        o = s3check(a)
        print(json.dumps(o), flush=True)
    else:
        main_s12(a)
