#!/usr/bin/env python3
"""episodes_diag: E1-E6 of README.md on one extracted sample -> results/results_<sample>.json
Every statistic is a ratio of per-run weighted sums; the bootstrap redraws runs (S1 plain, S2 blocks of 4, S3 blocks of 3)
and recomputes every ratio from the redrawn sums (paired differences).
    python analyze.py S1|S2|S3 [--ext FILE --out DIR --nboot B]"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EPS, L_HIST, NBOOT, REGIONS, ROOT, SEASONS, age_back, boot_counts, first_true, lead_run,  # noqa: E402
                    save_json)

warnings.filterwarnings("ignore", category=RuntimeWarning)
EXT = {"S1": "ext_2021.npz", "S2": "ext_2022.npz", "S3": "ext_cams.npz"}
BLOCK = {"S1": 1, "S2": 4, "S3": 3}
THR = {"37.5": 37.5, "75": 75.0, "P95": None}
WTS = ("cell", "res")
AGE1 = [(1, 1), (2, 2), (3, 4), (5, 7), (8, 10 ** 6)]                 # E1 strata of the age on day d
AGE1_NAMES = ["1", "2", "3-4", "5-7", ">=8", "all"]
AGE3 = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 7), (8, 14), (15, 10 ** 6)]  # E3 classes
AGE3_NAMES = ["1", "2", "3", "4", "5-7", "8-14", ">=15", "all"]
LN06, LN14 = float(np.log(0.6)), float(np.log(1.4))
NB, HLO, HHI = 800, -4.0, 4.0                                          # E4 histogram: bins of 0.01
E5_STRATA = ["all", "y0>37.5", "y0>75", "y0>P95"]
E5_SPACES = ["logc", "logr", "linc"]                                    # log centered / log raw / linear centered
E6_CLASSES = ["wet", "wind_rise", "light_rain", "quiet", "not_ended"]
SERIES = ["truth", "member", "mean"]


class Acc:
    """per-run sums: key -> array (N, ...)"""

    def __init__(self, n):
        self.n, self.d = n, {}

    def add(self, key, r, val):
        val = np.asarray(val, np.float64)
        if key not in self.d:
            self.d[key] = np.zeros((self.n,) + val.shape, np.float64)
        self.d[key][r] += val


def onehot(v, n):
    """v (..., k) ints 1..n -> (..., k, n) float"""
    return (v[..., None] == np.arange(1, n + 1)).astype(np.float64)


def class_of(a, cens, bounds, cens_below):
    """age class index (or -1: censored age below cens_below, or no episode)"""
    c = np.full(a.shape, -1, np.int64)
    for i, (lo, hi) in enumerate(bounds):
        c[(a >= lo) & (a <= hi)] = i
    c[cens & (a < cens_below)] = -1
    return c


def episode_sums(acc, r, tk, y0, hist, Y, X, Mn, Tc, W, reg, H, Mx, PW):
    """E1-E4 (+E6) sums of one run for one threshold; arrays restricted to complete cell-runs with a finite threshold"""
    above0 = y0 > Tc
    exY, exX, exM = Y > Tc, X > Tc, Mn > Tc                            # (H, n), (Mx, H, n), (H, n)
    Ry, Rx, Rm = lead_run(exY), lead_run(np.swapaxes(exX, 0, 1)), lead_run(exM)   # (n,), (Mx, n), (n,)
    nback, cens = age_back(hist, Tc)
    A = 1 + nback
    ks = np.arange(1, H + 1)
    # ---------------- E1: ongoing episodes
    u = np.flatnonzero(above0)
    ry, rx, rm = Ry[u].astype(np.float64), Rx[:, u].astype(np.float64), Rm[u].astype(np.float64)
    Q = np.column_stack([np.ones(len(u)), ry, rx.mean(0), rm,
                         np.abs(rx - ry).mean(0), np.abs(rm - ry), H - ry,
                         (rx == ry).mean(0), rm == ry, ry == H,
                         ry[:, None] >= ks, (rx[:, :, None] >= ks).mean(0), rm[:, None] >= ks])
    s1 = class_of(A[u], cens[u], AGE1, 8)
    S = np.column_stack([(s1 == i) for i in range(len(AGE1))] + [np.ones(len(u), bool)]).astype(np.float64)
    OHy, OHx, OHm = onehot(Ry[u] + 1, H + 1), onehot(Rx[:, u] + 1, H + 1).mean(0), onehot(Rm[u] + 1, H + 1)
    Qr = np.column_stack([np.ones(len(u)), rx.mean(0) - ry, rm - ry, ry >= 3, (rx >= 3).mean(0), rm >= 3])
    for wk, w in W.items():
        wu = w[u][:, None]
        acc.add(f"E1/{tk}/{wk}/q", r, S.T @ (wu * Q))
        acc.add(f"E1/{tk}/{wk}/conf", r, np.stack([OHy.T @ (wu * OHx), OHy.T @ (wu * OHm)]))
        acc.add(f"E1/{tk}/{wk}/marg", r, np.stack([(exY[:, u] * wu.T).sum(1), (exX[:, :, u].mean(0) * wu.T).sum(1),
                                                    (exM[:, u] * wu.T).sum(1)]))
        acc.add(f"E1/{tk}/{wk}/reg", r, reg[:, u].astype(np.float64) @ (wu * Qr))
    # ---------------- E2: onsets
    u2 = np.flatnonzero(~above0)
    Oy, Ox, Om = first_true(exY[:, u2]), first_true(np.swapaxes(exX[:, :, u2], 0, 1)), first_true(exM[:, u2])
    Py, Px, Pm = onehot(Oy, H + 1), onehot(Ox, H + 1).mean(0), onehot(Om, H + 1)
    for wk, w in W.items():
        wu = w[u2][:, None]
        acc.add(f"E2/{tk}/{wk}/cnt", r, wu.sum())
        acc.add(f"E2/{tk}/{wk}/dist", r, np.stack([(wu * Py).sum(0), (wu * Px).sum(0), (wu * Pm).sum(0)]))
        acc.add(f"E2/{tk}/{wk}/joint", r, np.stack([Py.T @ (wu * Px), Py.T @ (wu * Pm)]))
        acc.add(f"E2/{tk}/{wk}/marg", r, np.stack([(exY[:, u2] * wu.T).sum(1), (exX[:, :, u2].mean(0) * wu.T).sum(1),
                                                    (exM[:, u2] * wu.T).sum(1)]))
    # ---------------- E3 hazard and E4 day-to-day changes on the same at-risk transitions
    traj = {"truth": [Y], "member": [X[j] for j in range(Mx)], "mean": [Mn]}
    for si, sname in enumerate(SERIES):
        risk = {wk: np.zeros((2, 8)) for wk in W}; ended = {wk: np.zeros((2, 8)) for wk in W}
        mom = {wk: np.zeros((2, 6)) for wk in W}; hist_ = {wk: np.zeros((2, NB)) for wk in W}
        for F in traj[sname]:
            Z = np.vstack([y0[None], F])                                  # (H + 1, n): day d (shared truth), then leads
            a = np.where(above0, A, 0); c = cens & above0
            for t in range(H):
                rk = Z[t] > Tc
                en = Z[t + 1] <= Tc
                cl = class_of(a, c, AGE3, 15)
                st = 0 if t == 0 else 1
                dl = np.log(np.maximum(Z[t + 1], EPS)) - np.log(np.maximum(Z[t], EPS))
                b = np.clip(((dl - HLO) / (HHI - HLO) * NB).astype(np.int64), 0, NB - 1)
                for wk, w in W.items():
                    wr = np.where(rk, w, 0.0)
                    risk[wk][st] += np.bincount(np.where(cl >= 0, cl, 7), weights=wr, minlength=8)[:8] * np.r_[np.ones(7), 0]
                    risk[wk][st, 7] += wr.sum()
                    we = wr * en
                    ended[wk][st] += np.bincount(np.where(cl >= 0, cl, 7), weights=we, minlength=8)[:8] * np.r_[np.ones(7), 0]
                    ended[wk][st, 7] += we.sum()
                    mom[wk][st] += [wr.sum(), (wr * dl).sum(), (wr * dl ** 2).sum(), (wr * dl ** 3).sum(),
                                    wr[dl < LN06].sum(), wr[dl > LN14].sum()]
                    hist_[wk][st] += np.bincount(b, weights=wr, minlength=NB)
                up = Z[t + 1] > Tc
                a = np.where(up, a + 1, 0); c = c & up
        k = len(traj[sname])
        for wk in W:
            acc.add(f"E3/{tk}/{wk}/risk/{sname}", r, risk[wk] / k)
            acc.add(f"E3/{tk}/{wk}/ended/{sname}", r, ended[wk] / k)
            acc.add(f"E4/{tk}/{wk}/mom/{sname}", r, mom[wk] / k)
            acc.add(f"E4/{tk}/{wk}/hist/{sname}", r, hist_[wk] / k)
    # ---------------- E6: weather of the run of day d on the true end day
    if PW is not None:
        Pr, Wr = PW
        sel = ry <= H - 1
        ue = u[sel]
        e = (ry[sel] + 1).astype(np.int64)                                # end lead 1..H
        Pe = Pr[e - 1, ue]; dW = Wr[e, ue] - Wr[e - 1, ue]
        cls = np.full(len(ue), -1)
        cls[Pe >= 5] = 0
        cls[(Pe < 5) & (dW >= 1.5)] = 1
        cls[(Pe >= 1) & (Pe < 5) & (dW < 1.5)] = 2
        cls[(Pe < 1) & (dW < 1.5)] = 3
        ex, em, ep = rx - ry, rm - ry, H - ry                              # (Mx, nu), (nu,), (nu,)
        Q6 = np.column_stack([np.ones(len(u)), ex.mean(0), em, ep, (ex > 0).mean(0), (ex == 0).mean(0), (ex < 0).mean(0),
                              em > 0, em == 0, em < 0, np.zeros(len(u)), np.zeros(len(u))])
        Q6[sel, 10], Q6[sel, 11] = Pe, dW
        allc = np.full(len(u), -1); allc[sel] = cls; allc[ry == H] = 4
        O6 = (allc[:, None] == np.arange(5)).astype(np.float64)
        for wk, w in W.items():
            acc.add(f"E6/{tk}/{wk}/q", r, O6.T @ (w[u][:, None] * Q6))
            acc.add(f"E6/{tk}/{wk}/unclassified", r, w[ue][cls < 0].sum())


def e5_arrays(X, Mn, Y, M):
    O = (M * Mn[None] - X) / (M - 1)
    lo = np.log(np.maximum(O, EPS))
    return {"A_log": np.log(np.maximum(X, EPS)) - lo, "B_log": np.log(np.maximum(Y, EPS))[None] - lo,
            "A_lin": X - O, "B_lin": Y[None] - O}


def e5_sums(acc, r, arr, cen, v, strata, W, H, Mx):
    """pair sums (count, S1, S2, S11, S22, S12) of (Z_h, Z_h+k), pooled over h and the member choices, per stratum and weight"""
    sp = {"logc": ("A_log", "B_log", True), "logr": ("A_log", "B_log", False), "linc": ("A_lin", "B_lin", True)}
    for spn, (ka, kb, centered) in sp.items():
        per = np.zeros((2, H - 1, 6, len(v)))                          # per-unit sums
        for zi, key in enumerate((ka, kb)):
            Z = (arr[key] - cen[key][:, :, v]) if centered else arr[key]  # (Mx, H, n)
            for k in range(1, H):
                z1, z2 = Z[:, :H - k], Z[:, k:]
                per[zi, k - 1, 0] = Mx * (H - k)
                per[zi, k - 1, 1], per[zi, k - 1, 2] = z1.sum((0, 1)), z2.sum((0, 1))
                per[zi, k - 1, 3], per[zi, k - 1, 4] = (z1 * z1).sum((0, 1)), (z2 * z2).sum((0, 1))
                per[zi, k - 1, 5] = (z1 * z2).sum((0, 1))
        for stn, smask in strata.items():
            for wk, w in W.items():
                acc.add(f"E5/{wk}/{stn}/{spn}", r, (per @ np.where(smask, w, 0.0)) / Mx)


# ======================================================================== finalisation (leading axis = draws)
def fin_e1(S, H):
    q = S["q"]                                                         # (b, 6, 10 + 3H)
    cnt = q[..., 0]
    with np.errstate(all="ignore"):
        R = q[..., 1:4] / cnt[..., None]
        out = {"n": cnt, "mean_R": R,
               "err_member": (q[..., 2] - q[..., 1]) / cnt, "err_mean": (q[..., 3] - q[..., 1]) / cnt,
               "err_persistence": H - q[..., 1] / cnt,
               "mae": q[..., 4:7] / cnt[..., None], "exact": q[..., 7:10] / cnt[..., None]}
        sv = q[..., 10:10 + 3 * H].reshape(q.shape[:-1] + (3, H)) / cnt[..., None, None]
        out["surv"] = sv
        out["surv_member_minus_truth"] = sv[..., 1, :] - sv[..., 0, :]
        out["surv_mean_minus_truth"] = sv[..., 2, :] - sv[..., 0, :]
        c5 = cnt[..., 3] + cnt[..., 4]
        ex5 = (q[..., 3, 2] - q[..., 3, 1] + q[..., 4, 2] - q[..., 4, 1]) / c5
        em5 = (q[..., 3, 3] - q[..., 3, 1] + q[..., 4, 3] - q[..., 4, 1]) / c5
        out["err_member_age>=5"], out["err_mean_age>=5"] = ex5, em5
        out["P3_age_effect_member"] = ex5 - out["err_member"][..., 1]
        out["age_effect_mean"] = em5 - out["err_mean"][..., 1]
        out["age_effect_truth_meanR"] = (q[..., 3, 1] + q[..., 4, 1]) / c5 - R[..., 1, 0]
        out["age_effect_member_meanR"] = (q[..., 3, 2] + q[..., 4, 2]) / c5 - R[..., 1, 1]
        call = cnt[..., 5]
        if "conf" in S:
            out["conf"] = S["conf"] / call[..., None, None, None]
            out["marg"] = S["marg"] / call[..., None, None]
            g = S["reg"]                                                   # (b, 10, 6)
            out["reg_n"] = g[..., 0]
            out["reg_err_member"], out["reg_err_mean"] = g[..., 1] / g[..., 0], g[..., 2] / g[..., 0]
            out["reg_S3_member_minus_truth"] = (g[..., 4] - g[..., 3]) / g[..., 0]
            out["reg_S3_mean_minus_truth"] = (g[..., 5] - g[..., 3]) / g[..., 0]
    return out


def fin_e2(S, H):
    cnt, dist, J = S["cnt"], S["dist"], S["joint"]                     # (b,), (b, 3, H+1), (b, 2, H+1, H+1)
    out = {"n": cnt}
    with np.errstate(all="ignore"):
        out["onset_dist"] = dist / cnt[..., None, None]
        row = J.sum(-1)                                                # (b, 2, H+1)
        hitm = J[..., :H].sum(-1)
        out["hit_rate_by_true_lead"] = hitm[..., :H] / row[..., :H]
        lead = np.arange(H + 1)
        near = (np.abs(lead[:, None] - lead[None, :]) <= 1) & (lead[None, :] < H)
        out["timely_hit_by_true_lead"] = (J * near).sum(-1)[..., :H] / row[..., :H]
        dlt = (lead[None, :] - lead[:, None]).astype(np.float64)        # forecast onset - true onset
        Jh = J[..., :H, :H]
        out["timing_err_by_true_lead"] = (Jh * dlt[:H, :H]).sum(-1) / Jh.sum(-1)
        out["timing_err_hits"] = (Jh * dlt[:H, :H]).sum((-1, -2)) / Jh.sum((-1, -2))
        out["false_onset_rate"] = J[..., H, :H].sum(-1) / row[..., H]
        out["freq_bias"] = J[..., :, :H].sum((-1, -2)) / J[..., :H, :].sum((-1, -2))
        out["hit_rate_all"] = Jh.sum((-1, -2)) / J[..., :H, :].sum((-1, -2))
        out["marg"] = S["marg"] / cnt[..., None, None]
    return out


def fin_e3(S):
    risk = np.stack([S[f"risk/{s}"] for s in SERIES], -2)              # (b, 2, 3, 8)
    ended = np.stack([S[f"ended/{s}"] for s in SERIES], -2)
    with np.errstate(all="ignore"):
        lam = ended / risk
    return {"at_risk": risk, "hazard": lam, "member_minus_truth": lam[..., 1, :] - lam[..., 0, :],
            "mean_minus_truth": lam[..., 2, :] - lam[..., 0, :]}


def quantiles_from_hist(h, qs):
    cum = np.cumsum(h, -1); tot = cum[..., -1:]
    bw = (HHI - HLO) / NB
    out = []
    for q in qs:
        tgt = q * tot
        idx = np.argmax(cum >= tgt, -1)
        prev = np.take_along_axis(np.concatenate([np.zeros(cum.shape[:-1] + (1,)), cum], -1), idx[..., None], -1)[..., 0]
        hb = np.take_along_axis(h, idx[..., None], -1)[..., 0]
        with np.errstate(all="ignore"):
            out.append(HLO + bw * (idx + np.clip((tgt[..., 0] - prev) / hb, 0, 1)))
    return np.stack(out, -1)


def fin_e4(S):
    m = np.stack([S[f"mom/{s}"] for s in SERIES], -2)                  # (b, 2, 3, 6)
    h = np.stack([S[f"hist/{s}"] for s in SERIES], -2)                 # (b, 2, 3, NB)
    with np.errstate(all="ignore"):
        n = m[..., 0]; mu = m[..., 1] / n; e2 = m[..., 2] / n; e3 = m[..., 3] / n
        var = e2 - mu ** 2
        skew = (e3 - 3 * mu * e2 + 2 * mu ** 3) / var ** 1.5
        drop, rise = m[..., 4] / n, m[..., 5] / n
    out = {"n": n, "mean": mu, "sd": np.sqrt(var), "skew": skew, "big_drop_share": drop, "big_rise_share": rise,
           "quantiles_5_25_50_75_95": quantiles_from_hist(h, [0.05, 0.25, 0.5, 0.75, 0.95])}
    for k in ("big_drop_share", "big_rise_share", "skew", "sd"):
        out[f"{k}_member_minus_truth"] = out[k][..., 1] - out[k][..., 0]
        out[f"{k}_mean_minus_truth"] = out[k][..., 2] - out[k][..., 0]
    return out


def corr_from(s):
    """s (..., 6) = Sw, S1, S2, S11, S22, S12 -> corr, cov, var1, var2"""
    w = s[..., 0]
    with np.errstate(all="ignore"):
        m1, m2 = s[..., 1] / w, s[..., 2] / w
        cov = s[..., 5] / w - m1 * m2
        v1, v2 = s[..., 3] / w - m1 ** 2, s[..., 4] / w - m2 ** 2
        return cov / np.sqrt(v1 * v2), cov, v1, v2


def fin_e5(S, M):
    out = {}
    for spn in E5_SPACES:
        s = S[spn]                                                     # (b, 2, H-1, 6)
        rho, cov, v1, v2 = corr_from(s)
        o = {"rho_A": rho[..., 0, :], "rho_B": rho[..., 1, :], "d_rho": rho[..., 0, :] - rho[..., 1, :],
             "n_pairs_weight": s[..., 0, :, 0]}
        with np.errstate(all="ignore"):
            if spn == "logr":
                g = (s[..., 3] + s[..., 4] - 2 * s[..., 5])
                o["variogram_ratio_A_over_B"] = g[..., 0, :] / g[..., 1, :]
            if spn == "logc":
                lv = (s[..., 3] + s[..., 4])
                o["level_ratio_A2_over_B2"] = lv[..., 0, :] / lv[..., 1, :]
            if spn == "linc":
                cy = cov[..., 1, :] - cov[..., 0, :] / M
                v1y, v2y = v1[..., 1, :] - v1[..., 0, :] / M, v2[..., 1, :] - v2[..., 0, :] / M
                o["rho_x_dediluted"] = rho[..., 0, :]
                o["rho_y_dediluted"] = cy / np.sqrt(v1y * v2y)
                o["var_ratio_member_over_truth_dediluted"] = (v1[..., 0, :] * (M - 1) / M) / v1y
        out.update({f"{spn}/{k}": v for k, v in o.items()})
    return out


def fin_e6(S):
    q = S["q"]                                                         # (b, 5, 12)
    n = q[..., 0]
    ended_tot = n[..., :4].sum(-1) + S["unclassified"]
    with np.errstate(all="ignore"):
        return {"n": n, "share_of_ended": n / ended_tot[..., None], "err_member": q[..., 1] / n, "err_mean": q[..., 2] / n,
                "err_persistence": q[..., 3] / n, "member_too_long": q[..., 4] / n, "member_exact": q[..., 5] / n,
                "member_too_short": q[..., 6] / n, "mean_too_long": q[..., 7] / n, "mean_exact": q[..., 8] / n,
                "mean_too_short": q[..., 9] / n, "precip_mm_mean": q[..., 10] / n, "dW_mean": q[..., 11] / n}


def summarise(acc, runs, block, H, M, nboot, keys_prefix=None):
    """point estimates and percentile intervals of every finalised statistic over the given runs"""
    K = boot_counts(len(runs), block, nboot)
    one = np.ones((1, len(runs)))
    sel = {k: v[runs] for k, v in acc.d.items() if keys_prefix is None or k.startswith(keys_prefix)}

    def agg(Kw):
        return {k: np.tensordot(Kw, v, axes=(1, 0)) for k, v in sel.items()}

    res = {}
    for which, Kw in (("est", one), ("boot", K)):
        A = agg(Kw)
        R = {}
        groups = {}
        for k, v in A.items():
            parts = k.split("/")
            if parts[0] in ("E1", "E2", "E6"):
                gk, leaf = "/".join(parts[:3]), parts[3]
            elif parts[0] in ("E3", "E4"):
                gk, leaf = "/".join(parts[:3]), "/".join(parts[3:])
            else:                                                       # E5/w/stratum/space
                gk, leaf = "/".join(parts[:3]), parts[3]
            groups.setdefault(gk, {})[leaf] = v
        for gk, S in groups.items():
            e = gk.split("/")[0]
            R[gk] = {"E1": lambda S: fin_e1(S, H), "E2": lambda S: fin_e2(S, H), "E3": fin_e3, "E4": fin_e4,
                     "E5": lambda S: fin_e5(S, M), "E6": fin_e6}[e](S)
        res[which] = R
    out = {}
    for gk, st in res["est"].items():
        out[gk] = {}
        for name, val in st.items():
            b = res["boot"][gk][name]
            with np.errstate(all="ignore"):
                lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0) if np.isfinite(b).any() else (b[0] * np.nan, b[0] * np.nan)
            out[gk][name] = {"est": val[0], "lo": lo, "hi": hi}
    return out


def main(a):
    t0 = time.time()
    sample = a.sample
    z = np.load(a.ext or (ROOT / "data" / EXT[sample]))
    D = {k: z[k] for k in z.files}
    H, M, Mx = int(D["H"]), int(D["M"]), int(D["Mx"])
    N = len(D["run_label"])
    truth, tidx, pop, p95 = D["truth"], D["run_tidx"], D["pop"].astype(np.float64), D["p95"]
    regions = np.vstack([D["regions"], np.ones((1, len(pop)), bool)])
    has_pw = "P" in D
    print(f"{sample}: {N} runs, {len(pop)} cells, H {H}, M {M}, Mx {Mx}, E6 {'yes' if has_pw else 'no'}", flush=True)

    def run_arrays(r):
        t = tidx[r]
        y0 = truth[t]
        hist = truth[t - np.arange(1, L_HIST + 1)]
        Y, X, Mn = D["Y"][r].astype(np.float32), D["X"][r].astype(np.float32), D["Mn"][r].astype(np.float32)
        ok = np.isfinite(y0) & np.isfinite(Y).all(0) & np.isfinite(X).all((0, 1)) & np.isfinite(Mn).all(0)
        v = np.flatnonzero(ok)
        return v, y0[v], hist[:, v], Y[:, v], X[:, :, v], Mn[:, v]

    # ---------------- pass 1: E5 centering means per (member choice, lead, cell) over the runs
    cen = {k: np.zeros((Mx, H, len(pop)), np.float64) for k in ("A_log", "B_log", "A_lin", "B_lin")}
    cnt = np.zeros(len(pop))
    n_complete = np.zeros(N, np.int64)
    for r in range(N):
        v, y0, hist, Y, X, Mn = run_arrays(r)
        n_complete[r] = len(v)
        arr = e5_arrays(X, Mn, Y, M)
        for k in cen:
            cen[k][:, :, v] += arr[k]
        cnt[v] += 1
    for k in cen:
        cen[k] /= np.maximum(cnt, 1)[None, None]
    print(f"pass 1 {time.time() - t0:.0f}s; complete cell-runs {int(n_complete.sum())}", flush=True)

    # ---------------- pass 2: per-run sums
    acc = Acc(N)
    for r in range(N):
        v, y0, hist, Y, X, Mn = run_arrays(r)
        W = {"cell": np.ones(len(v)), "res": pop[v]}
        reg = regions[:, v]
        PW = (D["P"][r][:, v].astype(np.float32), D["W"][r][:, v].astype(np.float32)) if has_pw else None
        for tk, T in THR.items():
            Tc = p95[v] if T is None else np.full(len(v), T, np.float32)
            f = np.isfinite(Tc)
            fi = np.flatnonzero(f)
            sub = lambda q, ax=-1: np.take(q, fi, axis=ax)
            episode_sums(acc, r, tk, y0[fi], hist[:, fi], Y[:, fi], X[:, :, fi], Mn[:, fi], Tc[fi],
                         {k: w[fi] for k, w in W.items()}, reg[:, fi], H, Mx,
                         None if PW is None else (sub(PW[0]), sub(PW[1])))
        arr = e5_arrays(X, Mn, Y, M)
        strata = {"all": np.ones(len(v), bool), "y0>37.5": y0 > 37.5, "y0>75": y0 > 75.0,
                  "y0>P95": np.isfinite(p95[v]) & (y0 > np.nan_to_num(p95[v], nan=np.inf))}
        e5_sums(acc, r, arr, cen, v, strata, W, H, Mx)
        if r % 10 == 0:
            print(f"run {r + 1}/{N} {time.time() - t0:.0f}s", flush=True)

    # ---------------- summaries
    allruns = np.arange(N)
    res = {"sample": sample, "n_runs": N, "n_cells": len(pop), "H": H, "M": M, "member_trajectories": Mx,
           "run_labels": D["run_label"], "complete_cell_runs": int(n_complete.sum()),
           "bootstrap": {"draws": a.nboot, "block_runs": BLOCK[sample]}, "regions": list(REGIONS) + ["world"],
           "age1": AGE1_NAMES, "age3": AGE3_NAMES, "e6_classes": E6_CLASSES, "series": SERIES,
           "stats": summarise(acc, allruns, BLOCK[sample], H, M, a.nboot)}
    print(f"summaries {time.time() - t0:.0f}s", flush=True)
    seas = {}
    for si, sn in enumerate(SEASONS):
        rs = np.flatnonzero(D["run_season"] == si)
        if len(rs) >= 4:
            seas[sn] = {"n_runs": len(rs), "stats": summarise(acc, rs, BLOCK[sample], H, M, a.nboot, keys_prefix="E1/")}
    res["seasons"] = seas
    # ---------------- pre-registered contrasts
    st = res["stats"]
    con = {}
    for tk in THR:
        for wk in WTS:
            e1, e2 = st[f"E1/{tk}/{wk}"], st[f"E2/{tk}/{wk}"]
            e3, e4 = st[f"E3/{tk}/{wk}"], st[f"E4/{tk}/{wk}"]
            pick = lambda d, idx: {k: np.asarray(d[k])[idx] for k in ("est", "lo", "hi")}
            con[f"{tk}/{wk}"] = {
                "P1_err_member": pick(e1["err_member"], 5),
                "P2_S3_member_minus_truth": pick(e1["surv_member_minus_truth"], (5, 2)),
                "P3_age_effect_member": e1["P3_age_effect_member"],
                "P4_within_hazard_member_minus_truth": pick(e3["member_minus_truth"], (1, 7)),
                "P5_within_big_drop_member_minus_truth": pick(e4["big_drop_share_member_minus_truth"], 1),
                "P7a_onset_timing_err_member": pick(e2["timing_err_hits"], 0),
                "P7b_onset_freq_bias_member_minus_1": {k: np.asarray(e2["freq_bias"][k])[0] - 1 for k in ("est", "lo", "hi")},
            }
    for wk in WTS:
        con[f"P6/{wk}"] = {k: np.asarray(st[f"E5/{wk}/all"]["logc/d_rho"][k])[0] for k in ("est", "lo", "hi")}
    res["contrasts"] = con
    res["seconds"] = time.time() - t0
    out = Path(a.out) if a.out else ROOT
    save_json(out / "results" / f"results_{sample}.json", res)
    print(f"saved results_{sample}.json {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sample", choices=["S1", "S2", "S3"])
    ap.add_argument("--ext", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nboot", type=int, default=NBOOT)
    main(ap.parse_args())
