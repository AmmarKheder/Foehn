"""A14 class U from sufficient statistics of a set of members (any subset Q of the 16 fields).
Inputs summed over cases (weights included): D_ij = sum d_i d_j (d = x - y), A_i = sum |d_i|, P_ij = sum |x_i - x_j|, n;
for thresholds E_ij = sum e_i e_j, F_i = sum e_i o, O = sum o. Every value below is the A14 4a estimator of the subset."""
import numpy as np

from u_common import F_FPC, QUARTETS


def subsets_jk(m=16):
    return np.array([[j for j in range(m) if j != i] for i in range(m)], np.int64)


def stats(D, n, Q, A=None, P=None):
    """class-U scores of each subset (rows of Q) of the members"""
    Q = np.atleast_2d(Q); m = Q.shape[1]
    Dsum = D[Q[:, :, None], Q[:, None, :]].sum((1, 2)); Ddiag = D[Q, Q].sum(1)
    sse = Dsum / m ** 2                                    # sum (xbar - y)^2
    s2 = (Ddiag - Dsum / m) / (m - 1)                      # sum of the unbiased member variance
    o = {}
    mse_m = sse / n; mse_inf = (sse - s2 / m) / n; mse_8 = mse_inf + s2 / (8 * n)
    o["rmse_m"] = np.sqrt(mse_m); o["rmse_inf"] = np.sqrt(mse_inf); o["rmse_8"] = np.sqrt(mse_8)
    o["spread"] = np.sqrt(s2 / n)
    o["ssr_inf"] = np.sqrt(s2 / n) / np.sqrt(mse_inf)
    o["ssr_fortin_m"] = np.sqrt((m + 1) / m * s2 / n) / np.sqrt(mse_m)
    o["ssr_fortin_8"] = np.sqrt(9 / 8 * s2 / n) / np.sqrt(mse_8)
    if A is not None and P is not None:
        absum = A[Q].sum(1) / m
        pair = P[Q[:, :, None], Q[:, None, :]].sum((1, 2)) / 2 / (m * (m - 1))
        o["crps_fair"] = (absum - pair) / n
        o["crps_8"] = (absum - pair + pair / 8) / n
    return o


def brier(E, F, O, Q, n):
    """fair, raw (M = |Q|) and eight-member Brier scores (means over cases) of each subset, per threshold (E: (T, 16, 16))"""
    Q = np.atleast_2d(Q); m = Q.shape[1]
    out = {"bs_fair": [], "bs_raw": [], "bs_8": []}
    for t in range(E.shape[0]):
        Es = E[t][Q[:, :, None], Q[:, None, :]].sum((1, 2)); Ed = E[t][Q, Q].sum(1); Fs = F[t][Q].sum(1)
        fair = (Es - Ed) / (m * (m - 1)) - 2 * Fs / m + O[t]
        raw = Es / m ** 2 - 2 * Fs / m + O[t]
        kmk = m * Ed - Es
        out["bs_fair"].append(fair / n); out["bs_raw"].append(raw / n); out["bs_8"].append((fair + kmk / (8 * m * (m - 1))) / n)
    return {k: np.stack(v, -1) for k, v in out.items()}                # (subsets, thresholds)


def k2(values_q, value_16, value_0123):
    """K2 of A14 section 6 for one statistic (corrected, 2026-11-21): the null distribution of
    t_0123 - t_16 is that of t_Q - t_16 over the 1,820 quartets, whose sd is the PLAIN quartet sd (ddof 0); z and the 2.5 flag
    use it, with the exact rank and the two-sided permutation p of t_0123 among the quartets. The 16/12 factor stays in V_draw
    only (v_quart). z_vs_quartet_mean = (t_0123 - mean_Q)/sd is the form for finite-M statistics, whose 16-member value is
    another estimand."""
    v = np.asarray(values_q, np.float64)
    sd = float(v.std()); sdd = sd * np.sqrt(F_FPC)
    t0 = float(value_0123); diff = float(t0 - value_16)
    rank = float((v < t0).mean() + 0.5 * (v == t0).mean())
    p2 = float(min(1.0, 2 * min((v >= t0).mean(), (v <= t0).mean())))
    return dict(q0123=t0, all16=float(value_16), quartet_mean=float(v.mean()), quartet_sd=sd,
                z=diff / sd if sd > 0 else None, flag_2p5=bool(sd > 0 and abs(diff) > 2.5 * sd), rank=rank, perm_p_two_sided=p2,
                z_vs_quartet_mean=float((t0 - v.mean()) / sd) if sd > 0 else None,
                draw_sd=sdd, z_drawscale=diff / sdd if sdd > 0 else None, flag_drawscale_2p5=bool(sdd > 0 and abs(diff) > 2.5 * sdd),
                quartet_min=float(v.min()), quartet_max=float(v.max()), v_quart=float(v.var() * F_FPC))


def full_report(D, n, A=None, P=None, E=None, F=None, O=None):
    """values of the 16 members, of the stored quartet (fields 0-3), the 1,820 quartets and the 16 delete-one subsets"""
    all16 = np.arange(16)[None]
    res = {"all16": stats(D, n, all16, A, P), "q": stats(D, n, QUARTETS, A, P), "jk": stats(D, n, subsets_jk(), A, P)}
    if E is not None:
        for key, Q in (("all16", all16), ("q", QUARTETS), ("jk", subsets_jk())):
            res[key].update(brier(E, F, O, Q, n))
    out = {}
    for k in res["all16"]:
        a16 = np.asarray(res["all16"][k]); q = np.asarray(res["q"][k]); jk = np.asarray(res["jk"][k])
        if a16.ndim == 1 and a16.shape[0] == 1:
            out[k] = dict(k2(q, a16[0], q[0]), jk16_var=float((15 / 16) * ((jk - jk.mean()) ** 2).sum()))
        else:                                                    # per threshold
            out[k] = [dict(k2(q[:, t], a16[0, t], q[0, t]), jk16_var=float((15 / 16) * ((jk[:, t] - jk[:, t].mean()) ** 2).sum()))
                      for t in range(a16.shape[1])]
    return out
