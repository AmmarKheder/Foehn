#!/usr/bin/env python3
"""A17 diag, step 2 (CPU only; reads only work/ of this folder, written by collect.py). Is a member's departure from the
ensemble mean an ordered, nearly LINEAR (hence odd) function of its own initial noise field?

Spaces. The network's own space is n = (x - 15) / 20 with x in ug m-3 (dejavu.norm, MU = 15, SD = 20; the sampler output
is mapped back by x = n * 20 + 15, no clipping). The transform is affine, so a departure in the network's space is the
departure in ug m-3 divided by 20: every share, correlation, R2, skewness and kurtosis below is the same number in both
spaces (checked numerically: 'space_check'); magnitudes are given in both units. 'rel' quantities use D / max(mean, 1)
and 'log' quantities use log x (where all members > 0): these are NOT the network's space, given beside because the
departure scales with the level.

Notation: x_m(p, t) member m (16, FIXED design: the same noise field m at every start day and lead), pixel p (300,000
random populated pixels), forecast t (179 start days of 2022 per lead); mean_t(p) = ensemble mean; D = x_m - mean;
S = time mean of D (static map); regimes of the ensemble mean: < 15, 15-75, > 75 ug m-3 (per forecast, or per pixel
for time means). Pooled sums are pixel-weighted (each sampled pixel counts once) unless 'pop' (residents) is said.
    python analyse.py -> results.json (and a printed summary)"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = Path(os.environ.get("A17_WORK", str(HERE / "work")))          # override only for the synthetic self-test
OUTF = Path(os.environ.get("A17_OUT", str(HERE / "results.json")))
REG = (15.0, 75.0)
RNAME = ("lt15", "15to75", "gt75")
MBIN = np.geomspace(1.0, 512.0, 19)
LIN = list(range(12))                     # z, 2x2 block, boxes 3 .. 1025 (odd, linear)
LNAME = ["z1", "blk2", "w3", "w5", "w9", "w17", "w33", "w65", "w129", "w257", "w513", "w1025"]
CUBW, EVW = [0, 4, 7, 9], [0, 4, 7, 9]      # columns of LIN for the Hermite cubic (odd) and square (even) terms
NB = 10                                    # bins of |u| per side (oddness)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)


def feature_sets(F):
    """F: (N, M, 17) raw unit-variance features -> dict of centred (across members) design arrays"""
    lin = F[..., LIN]
    cub = (lin[..., CUBW] ** 3 - 3 * lin[..., CUBW]) / np.sqrt(6.0)          # He3: odd, orthogonal to the linear term
    even = np.concatenate([(lin[..., EVW] ** 2 - 1) / np.sqrt(2.0), F[..., 12:17]], -1)   # He2 + local energies: even
    c = lambda a: (a - a.mean(1, keepdims=True)).astype(np.float64)
    L, C, E = c(lin), c(cub), c(even)
    return {"lin": L, "lin+cub": np.concatenate([L, C], -1), "even": E, "lin+even": np.concatenate([L, E], -1),
            "lin+cub+even": np.concatenate([L, C, E], -1)}


def accumulate(fc, li, doy, npar, extra16=False):
    """one pass over the forecasts of lead index li -> sums by (parity of t, regime of the ensemble mean)"""
    _, T, N, M = fc.shape
    z = lambda *s: np.zeros(s, np.float64)
    A = dict(S1=z(npar, 3, N, M), S2=z(npar, 3, N, M), SM=z(npar, 3, N, M), cnt=z(npar, 3, N), Smb=z(npar, 3, N),
             Smb2=z(npar, 3, N), S1H=z(2, N, M), nH=z(2), S1rel=z(N, M), S2rel=z(N, M),
             g3=z(3), g4=z(3), gn=z(3), l3=z(3), l4=z(3), ln=z(3), ssq=z(len(MBIN) + 1), sn=z(len(MBIN) + 1),
             bad=np.zeros(N, bool), npos=0, nall=0, T=T, M=M, xmin=np.inf, chk=[])
    if extra16:
        A["S1_16"] = z(N, 16)
    for t in range(T):
        x = np.asarray(fc[li, t], np.float32)
        A["bad"] |= ~np.isfinite(x).all(1)
        x = np.nan_to_num(x)
        mb = x.mean(1, dtype=np.float64)
        D = (x - mb[:, None]).astype(np.float64)
        A["bad"] |= (np.abs(D).max(1) == 0)
        A["xmin"] = min(A["xmin"], float(x.min()))
        reg = np.digitize(mb, REG)
        par = t % npar
        for r in range(3):
            w = (reg == r).astype(np.float64)
            Dw = D * w[:, None]
            A["S1"][par, r] += Dw; A["S2"][par, r] += Dw * D; A["SM"][par, r] += Dw * mb[:, None]
            A["cnt"][par, r] += w; A["Smb"][par, r] += w * mb; A["Smb2"][par, r] += w * mb * mb
        h = int(doy[t] >= 181)
        A["S1H"][h] += D; A["nH"][h] += 1
        Dr = D / np.maximum(mb, 1.0)[:, None]
        A["S1rel"] += Dr; A["S2rel"] += Dr * Dr
        sd = np.sqrt((D * D).mean(1))
        ok = sd > 0
        zz = D[ok] / sd[ok, None]
        m3, m4 = (zz ** 3).mean(1), (zz ** 4).mean(1)
        for r in range(3):
            k = reg[ok] == r
            A["g3"][r] += m3[k].sum(); A["g4"][r] += m4[k].sum(); A["gn"][r] += k.sum()
        pos = (x > 0).all(1)
        A["npos"] += int(pos.sum()); A["nall"] += len(pos)
        lx = np.log(x[pos].astype(np.float64)); Dl = lx - lx.mean(1, keepdims=True)
        sl = np.sqrt((Dl * Dl).mean(1)); okl = sl > 0
        zl = Dl[okl] / sl[okl, None]; rl = reg[pos][okl]
        l3, l4 = (zl ** 3).mean(1), (zl ** 4).mean(1)
        for r in range(3):
            k = rl == r
            A["l3"][r] += l3[k].sum(); A["l4"][r] += l4[k].sum(); A["ln"][r] += k.sum()
        b = np.digitize(mb, MBIN)
        A["ssq"] += np.bincount(b, weights=(D * D).mean(1), minlength=len(MBIN) + 1)
        A["sn"] += np.bincount(b, minlength=len(MBIN) + 1)
        if extra16:
            x16 = x[:, :16].astype(np.float64)
            A["S1_16"] += x16 - x16.mean(1, keepdims=True)
        if t == 0:                                            # the network's space gives the same dimensionless numbers
            n = (x.astype(np.float64) - 15.0) / 20.0
            Dn = n - n.mean(1, keepdims=True)
            A["chk"] = [float(np.abs(Dn * 20.0 - D).max()), float(np.corrcoef(Dn[:, 0], D[:, 0])[0, 1])]
    return A


def tot(A, k, regs=(0, 1, 2), pars=None):
    a = A[k]
    pars = range(a.shape[0]) if pars is None else pars
    return sum(a[p_, r] for p_ in pars for r in regs)


def static_share(S1, S2, T, mask):
    """pooled ANOVA: static variance corrected for the estimation noise of the time mean"""
    S = S1[mask] / T
    VR = (S2[mask] - T * S * S).sum() / (S.size * (T - 1))
    VSh = (S * S).mean()
    VS = VSh - VR / T
    return dict(static_share=VS / (VS + VR), V_static=VS, V_timevarying=VR, V_static_hat=VSh, ceiling=VS / VSh,
                rms_D_ug=float(np.sqrt(VS + VR)), rms_D_net=float(np.sqrt(VS + VR)) / 20.0)


def cv_time(A, mask):
    """static maps fitted on even forecasts, scored on odd ones (and back): additive S, multiplicative K*mean, affine a+b*mean"""
    out = {}
    for name in ("add", "mul", "aff"):
        sse = ss = 0.0
        for fp, tp in ((0, 1), (1, 0)):
            g = lambda k, p_: tot(A, k, pars=[p_])
            n_f, mb_f, mb2_f = g("cnt", fp)[mask], g("Smb", fp)[mask], g("Smb2", fp)[mask]
            S1f, SMf = g("S1", fp)[mask], g("SM", fp)[mask]
            n_t, mb_t, mb2_t = g("cnt", tp)[mask], g("Smb", tp)[mask], g("Smb2", tp)[mask]
            S1t, SMt, S2t = g("S1", tp)[mask], g("SM", tp)[mask], g("S2", tp)[mask]
            if name == "add":
                a = S1f / n_f[:, None]; b = 0.0
            elif name == "mul":
                a = 0.0; b = SMf / mb2_f[:, None]
            else:
                det = n_f * mb2_f - mb_f ** 2
                a = (mb2_f[:, None] * S1f - mb_f[:, None] * SMf) / det[:, None]
                b = (n_f[:, None] * SMf - mb_f[:, None] * S1f) / det[:, None]
            sse += (S2t - 2 * a * S1t - 2 * b * SMt + a * a * n_t[:, None] + 2 * a * b * mb_t[:, None]
                    + b * b * mb2_t[:, None]).sum()
            ss += S2t.sum()
        out[name] = 1.0 - sse / ss
    return out


def ridge_fit(X, y, w=None, lam=1e-3):
    K = X.shape[-1]
    Xf, yf = X.reshape(-1, K), y.reshape(-1)
    Xw = Xf if w is None else Xf * w.reshape(-1)[:, None]
    XtX = Xw.T @ Xf
    return np.linalg.solve(XtX + lam * np.trace(XtX) / K * np.eye(K), Xw.T @ yf)


def folds(M, k=4):
    return [np.arange(f * M // k, (f + 1) * M // k) for f in range(k)]


def ridge_cv(X, y, pmask, M, wpix=None):
    """member folds (4 x 4 of 16): fit on 12 members, predict the other 4. Returns out-of-fold prediction, R2, betas"""
    pred = np.zeros_like(y); betas = []
    for te in folds(M):
        tr = np.setdiff1d(np.arange(M), te)
        w = None if wpix is None else np.broadcast_to(wpix[pmask][:, None], (pmask.sum(), len(tr)))
        b = ridge_fit(X[pmask][:, tr], y[pmask][:, tr], w)
        pred[:, te] = X[:, te] @ b
        betas.append(b)
    return pred, np.array(betas)


def r2(y, p, mask, w=None):
    e, s = (y[mask] - p[mask]) ** 2, y[mask] ** 2
    if w is not None:
        e, s = e * w[mask][:, None], s * w[mask][:, None]
    return float(1.0 - e.sum() / s.sum())


def corr0(a, b):
    a, b = a.ravel(), b.ravel()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def oddness(y, u, nb=NB):
    """E[y | u] on bins symmetric in u (edges: quantiles of |u|): odd and even parts, their rms ratio"""
    au = np.abs(u)
    edges = np.quantile(au, np.linspace(0, 1, nb + 1)); edges[-1] = np.inf
    k = np.clip(np.searchsorted(edges, au, side="right") - 1, 0, nb - 1)
    pos = u > 0
    rows = []
    for j in range(nb):
        ip, im = (k == j) & pos, (k == j) & ~pos
        mp, mm = y[ip].mean(), y[im].mean()
        rows.append(dict(abs_u_lo=float(edges[j]), n_pos=int(ip.sum()), n_neg=int(im.sum()), mean_pos=float(mp),
                         mean_neg=float(mm), odd=float((mp - mm) / 2), even=float((mp + mm) / 2)))
    n = np.array([r["n_pos"] + r["n_neg"] for r in rows], np.float64)
    odd, ev = np.array([r["odd"] for r in rows]), np.array([r["even"] for r in rows])
    return dict(even_over_odd=float(np.sqrt((n * ev ** 2).sum() / (n * odd ** 2).sum())), bins=rows)


def m3_coef(F, A, pmask, tr, use):
    """least squares (tiny ridge) of the per-forecast D on [F, mean * F] over members tr, from the sums of accumulate"""
    K = F.shape[-1]
    cnt, Smb, Smb2 = tot(A, "cnt"), tot(A, "Smb"), tot(A, "Smb2")
    Fp = F[pmask][:, tr]
    Xf = Fp.reshape(-1, K)
    blk = lambda wp: (Xf * np.repeat(wp[pmask], len(tr))[:, None]).T @ Xf
    y1 = Xf.T @ tot(A, "S1")[pmask][:, tr].reshape(-1)
    y2 = Xf.T @ tot(A, "SM")[pmask][:, tr].reshape(-1)
    if use == ("a",):
        XtX, Xty = blk(cnt), y1
    elif use == ("g",):
        XtX, Xty = blk(Smb2), y2
    else:
        B12 = blk(Smb)
        XtX, Xty = np.block([[blk(cnt), B12], [B12, blk(Smb2)]]), np.concatenate([y1, y2])
    KK = XtX.shape[0]
    sol = np.linalg.solve(XtX + 1e-3 * np.trace(XtX) / KK * np.eye(KK), Xty)
    if use == ("a",):
        return sol, np.zeros(K)
    if use == ("g",):
        return np.zeros(K), sol
    return sol[:K], sol[K:]


def m3_eval(A, pmask, a, g, members):
    """R2 of the per-forecast D by regime of the ensemble mean, for predictions a + mean * g ((N, M) arrays)"""
    out = {}
    for r in (0, 1, 2, "all"):
        rr = (0, 1, 2) if r == "all" else (r,)
        c, mb, mb2 = tot(A, "cnt", rr)[pmask], tot(A, "Smb", rr)[pmask], tot(A, "Smb2", rr)[pmask]
        s1, sm, s2 = (tot(A, k, rr)[pmask][:, members] for k in ("S1", "SM", "S2"))
        aa, gg = a[pmask][:, members], g[pmask][:, members]
        sse = (s2 - 2 * (aa * s1 + gg * sm) + aa * aa * c[:, None] + 2 * aa * gg * mb[:, None] + gg * gg * mb2[:, None]).sum()
        out[RNAME[r] if r != "all" else "all"] = dict(R2=float(1 - sse / s2.sum()), n_pixel_forecasts=float(c.sum()))
    return out


def m3_fit(F, A, pmask, M, use=("a", "g")):
    """per-forecast model D = F.beta + mean * F.gamma ('a' static, 'g' level-scaled), member folds: out-of-fold a, g and
    the R2 by regime of the per-forecast ensemble mean"""
    S1 = tot(A, "S1")
    a_oof, g_oof = np.zeros(S1.shape), np.zeros(S1.shape)
    for te in folds(M):
        b, g = m3_coef(F, A, pmask, np.setdiff1d(np.arange(M), te), use)
        a_oof[:, te] = F[:, te] @ b; g_oof[:, te] = F[:, te] @ g
    return a_oof, g_oof, m3_eval(A, pmask, a_oof, g_oof, np.arange(M))


def tv_level_fit(F, A, pmask, M):
    """time-varying part R = D - S regressed on (mean - time mean) * F (noise x level anomaly), member folds"""
    K = F.shape[-1]
    T = tot(A, "cnt")
    S1, SM, S2 = tot(A, "S1"), tot(A, "SM"), tot(A, "S2")
    mbar = tot(A, "Smb") / T
    V = tot(A, "Smb2") - tot(A, "Smb") ** 2 / T
    C = SM - mbar[:, None] * S1
    SSR = S2 - S1 ** 2 / T[:, None]
    sse = 0.0; g_oof = np.zeros(S1.shape)
    for te in folds(M):
        tr = np.setdiff1d(np.arange(M), te)
        Xf = F[pmask][:, tr].reshape(-1, K)
        XtX = (Xf * np.repeat(V[pmask], len(tr))[:, None]).T @ Xf
        g = np.linalg.solve(XtX + 1e-3 * np.trace(XtX) / K * np.eye(K), Xf.T @ C[pmask][:, tr].reshape(-1))
        g_oof[:, te] = F[:, te] @ g
    gg = g_oof[pmask]
    sse = (SSR[pmask] - 2 * gg * C[pmask] + gg * gg * V[pmask][:, None]).sum()
    return float(1 - sse / SSR[pmask].sum())


def odd_pass(fc, li, a_oof, g_oof, pmask, sub_t=None):
    """E[D | u] in regimes of the per-forecast ensemble mean, u = out-of-fold a + mean * g (2 passes: edges, then sums)"""
    T = fc.shape[1]
    tt = range(T) if sub_t is None else sub_t
    samp = {r: [] for r in range(3)}
    for t in list(tt)[::6]:
        x = np.nan_to_num(np.asarray(fc[li, t], np.float32))[pmask].astype(np.float64)
        mb = x.mean(1); reg = np.digitize(mb, REG)
        u = a_oof[pmask] + mb[:, None] * g_oof[pmask]
        for r in range(3):
            samp[r].append(np.abs(u[reg == r]).ravel()[::7])
    edges = {}
    for r in range(3):
        s = np.concatenate(samp[r])
        e = np.quantile(s, np.linspace(0, 1, NB + 1)) if s.size > 100 else np.linspace(0, 1, NB + 1)
        e[0], e[-1] = 0.0, np.inf
        edges[r] = e
    sums = np.zeros((3, NB, 2)); cnts = np.zeros((3, NB, 2))
    for t in tt:
        x = np.nan_to_num(np.asarray(fc[li, t], np.float32))[pmask].astype(np.float64)
        mb = x.mean(1); reg = np.digitize(mb, REG)
        D = x - mb[:, None]
        u = a_oof[pmask] + mb[:, None] * g_oof[pmask]
        for r in range(3):
            k = reg == r
            uu, dd = u[k].ravel(), D[k].ravel()
            b = np.clip(np.searchsorted(edges[r], np.abs(uu), side="right") - 1, 0, NB - 1)
            j = b * 2 + (uu > 0).astype(np.int64)
            sums[r] += np.bincount(j, weights=dd, minlength=2 * NB).reshape(NB, 2)
            cnts[r] += np.bincount(j, minlength=2 * NB).reshape(NB, 2)
    out = {}
    for r in range(3):
        n = cnts[r].sum(1)
        if n.sum() == 0:
            continue
        mp, mm = sums[r][:, 1] / np.maximum(cnts[r][:, 1], 1), sums[r][:, 0] / np.maximum(cnts[r][:, 0], 1)
        odd, ev = (mp - mm) / 2, (mp + mm) / 2
        out[RNAME[r]] = dict(even_over_odd=float(np.sqrt((n * ev ** 2).sum() / (n * odd ** 2).sum())),
                             n_values=float(n.sum()), odd_by_bin=odd.round(3).tolist(), even_by_bin=ev.round(3).tolist(),
                             abs_u_edges=[float(v) for v in edges[r][:-1]])
    return out


def gauss_ref(M):
    return 3.0 * (M - 1) / (M + 1) - 3.0       # E of the sample excess kurtosis (ddof 0 standardisation) for Gaussian data


def main():
    meta = np.load(W / "meta.npz")
    fc16 = np.load(W / "fc16.npy", mmap_mode="r")
    fc64 = np.load(W / "fc64.npy", mmap_mode="r")
    feat = np.load(W / "feat.npy")
    N = fc16.shape[2]
    pop = meta["pop"].astype(np.float64)
    R = dict(what=__doc__, n_pixels_sampled=int(N), seed=int(meta["seed"]), checks_collect=json.loads(str(meta["checks"])))
    lat, lon = 84.0 - 0.01 * (meta["r"] + 0.5), -180.0 + 0.01 * (meta["c"] + 0.5)
    REGIONS = {"Europe": (72, 35, -25, 45), "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30),
               "Africa": (37, -35, -20, 52), "Middle East": (42, 12, 35, 63), "South Asia": (37, 5, 60, 97),
               "East Asia": (54, 18, 97, 146), "Southeast Asia": (18, -11, 92, 142), "Oceania": (-10, -48, 110, 180)}
    R["pixels_by_region"] = {k: int(((lat <= n) & (lat > s) & (lon >= w) & (lon < e)).sum()) for k, (n, s, w, e) in REGIONS.items()}
    F16 = feature_sets(feat[:16].transpose(1, 0, 2))
    log("features ready")
    res22, keep = {}, {}
    for li, h in enumerate((1, 3)):
        doy = meta["keys16"][li, :, 0]
        A = accumulate(fc16, li, doy, 2)
        T = A["T"]
        pm = ~A["bad"]
        log(f"lead {h}: pass done, valid pixels {pm.sum()} of {N}, T {T}")
        S1, S2 = tot(A, "S1"), tot(A, "S2")
        S = S1 / T
        mlev = tot(A, "Smb") / T                                 # time-mean ensemble mean at the pixel
        preg = np.digitize(mlev, REG)
        out = dict(T=T, valid_pixels=int(pm.sum()), min_member_value=A["xmin"], space_check=A["chk"],
                   pixels_by_level=[int(((preg == r) & pm).sum()) for r in range(3)])
        out["static"] = {"all": static_share(S1, S2, T, pm)}
        for r in range(3):
            out["static"]["pixlev_" + RNAME[r]] = static_share(S1, S2, T, pm & (preg == r))
        Srel = A["S1rel"] / T
        out["static"]["rel"] = static_share(A["S1rel"], A["S2rel"], T, pm)
        out["cv_time_explained"] = cv_time(A, pm)
        out["cv_time_explained_by_regime"] = {}
        for r in range(3):                                       # global even/odd maps, scored within regime r (odd half)
            sse = {k: 0.0 for k in ("add", "mul")}; ss = 0.0
            for fp, tp in ((0, 1), (1, 0)):
                Sf = tot(A, "S1", pars=[fp])[pm] / tot(A, "cnt", pars=[fp])[pm][:, None]
                Kf = tot(A, "SM", pars=[fp])[pm] / tot(A, "Smb2", pars=[fp])[pm][:, None]
                s1, s2, sm = (A[k][tp, r][pm] for k in ("S1", "S2", "SM"))
                c, mb2 = A["cnt"][tp, r][pm], A["Smb2"][tp, r][pm]
                sse["add"] += (s2 - 2 * Sf * s1 + Sf * Sf * c[:, None]).sum()
                sse["mul"] += (s2 - 2 * Kf * sm + Kf * Kf * mb2[:, None]).sum()
                ss += s2.sum()
            out["cv_time_explained_by_regime"][RNAME[r]] = {k: 1 - v / ss for k, v in sse.items()}
        # stability of the static map: two halves of the year, and (below) lead 1 vs lead 3
        SH = A["S1H"] / A["nH"][:, None, None]
        out["static_halves_corr"] = corr0(SH[0][pm], SH[1][pm])
        out["static_halves_n"] = A["nH"].tolist()
        # dependence of the departure size on the level (per forecast): rms D in bins of the ensemble mean
        ok = A["sn"] > 1000
        cen = np.r_[np.nan, np.sqrt(MBIN[:-1] * MBIN[1:]), np.nan]
        rms = np.sqrt(A["ssq"] / np.maximum(A["sn"], 1))
        k = ok & np.isfinite(cen) & (cen >= 2)
        out["rmsD_by_level"] = dict(level_ug=cen[k].round(2).tolist(), rms_ug=rms[k].round(3).tolist(),
                                    n=A["sn"][k].tolist(), loglog_slope=float(np.polyfit(np.log(cen[k]), np.log(rms[k]), 1,
                                                                                        w=np.sqrt(A["sn"][k]))[0]))
        # Gaussian shape across the 16 members
        out["shape"] = {RNAME[r]: dict(skew=A["g3"][r] / A["gn"][r], excess_kurt=A["g4"][r] / A["gn"][r] - 3,
                                       log_skew=A["l3"][r] / max(A["ln"][r], 1), log_excess_kurt=A["l4"][r] / max(A["ln"][r], 1) - 3,
                                       n=A["gn"][r]) for r in range(3)}
        out["shape"]["gaussian_ref_excess_kurt"] = gauss_ref(16)
        out["shape"]["share_all_members_positive"] = A["npos"] / A["nall"]
        g3, g4, gn = A["g3"].sum(), A["g4"].sum(), A["gn"].sum()
        out["shape"]["all"] = dict(skew=g3 / gn, excess_kurt=g4 / gn - 3, log_skew=A["l3"].sum() / A["ln"].sum(),
                                   log_excess_kurt=A["l4"].sum() / A["ln"].sum() - 3)
        # 2. dependence on the noise: static map
        Lc = F16["lin"]
        out["corr_static_vs_scale"] = {LNAME[j]: corr0(S[pm], Lc[pm][..., j]) for j in range(12)}
        jb = max(range(12), key=lambda j: abs(out["corr_static_vs_scale"][LNAME[j]]))
        out["best_single_scale"] = dict(feature=LNAME[jb], corr=out["corr_static_vs_scale"][LNAME[jb]],
                                        R2_if_alone=out["corr_static_vs_scale"][LNAME[jb]] ** 2)
        out["corr_static_rel_vs_scale"] = {LNAME[j]: corr0(Srel[pm], Lc[pm][..., j]) for j in range(12)}
        out["ridge_static"] = {}
        preds = {}
        for nm, X in F16.items():
            p_, b_ = ridge_cv(X, S, pm, 16)
            preds[nm] = p_
            out["ridge_static"][nm] = dict(R2_oos=r2(S, p_, pm), R2_oos_over_ceiling=r2(S, p_, pm) / out["static"]["all"]["ceiling"])
            if nm == "lin":
                out["ridge_static"][nm]["beta_ug_per_sd_fold_mean"] = dict(zip(LNAME, b_.mean(0).round(3).tolist()))
                out["ridge_static"][nm]["beta_fold_spread"] = float(np.abs(b_ - b_.mean(0)).max())
                out["ridge_static"][nm]["R2_oos_pop_weighted"] = r2(S, p_, pm, pop)
                for r in range(3):
                    mk = pm & (preg == r)
                    out["ridge_static"][nm][f"R2_oos_pixlev_{RNAME[r]}_globalfit"] = r2(S, p_, mk)
                    p2, _ = ridge_cv(X, S, mk, 16)
                    out["ridge_static"][nm][f"R2_oos_pixlev_{RNAME[r]}_refit"] = r2(S, p2, mk)
                    out["ridge_static"][nm][f"ceiling_pixlev_{RNAME[r]}"] = out["static"]["pixlev_" + RNAME[r]]["ceiling"]
        p_rel, b_rel = ridge_cv(Lc, Srel, pm, 16)
        out["ridge_static_rel_lin"] = dict(R2_oos=r2(Srel, p_rel, pm), ceiling=out["static"]["rel"]["ceiling"])
        for r in range(3):
            out["ridge_static_rel_lin"][f"R2_oos_pixlev_{RNAME[r]}"] = r2(Srel, p_rel, pm & (preg == r))
        # 3. oddness along the fitted linear predictor (static), overall and by pixel level
        u = preds["lin"]
        out["odd_static"] = {"all": oddness(S[pm].ravel(), u[pm].ravel())}
        for r in range(3):
            mk = pm & (preg == r)
            out["odd_static"]["pixlev_" + RNAME[r]] = {k_: v for k_, v in oddness(S[mk].ravel(), u[mk].ravel()).items()
                                                       if k_ == "even_over_odd"}
        # 4. per forecast: static linear (a), level-scaled linear (g), both; time-varying part vs noise x level anomaly
        out["per_forecast"] = {}
        for nm, use in (("static_lin", ("a",)), ("level_lin", ("g",)), ("static+level_lin", ("a", "g"))):
            a_o, g_o, rr = m3_fit(Lc, A, pm, 16, use=use)
            out["per_forecast"][nm] = rr
            if use == ("a", "g"):
                keep[h] = (a_o, g_o)
        for nm in ("even", "lin+cub+even"):
            _, _, rr = m3_fit(F16[nm], A, pm, 16, use=("a", "g"))
            out["per_forecast"][nm + "_static+level"] = rr
        _, _, rr = m3_fit(F16["even"], A, pm, 16, use=("a",))
        out["per_forecast"]["even_static"] = rr
        out["tv_explained_by_noise_x_level_anomaly"] = tv_level_fit(Lc, A, pm, 16)
        out["tv_share_of_D"] = 1 - out["static"]["all"]["static_share"]
        out["odd_per_forecast"] = odd_pass(fc16, li, keep[h][0], keep[h][1], pm)
        res22[f"lead{h}"] = out
        keep[f"S{h}"], keep[f"pm{h}"], keep[f"Srel{h}"] = S, pm, Srel
        keep[f"rel{h}"] = out["static"]["all"]["ceiling"]
        keep[f"beta{h}"] = ridge_fit(Lc[pm], S[pm])
        keep[f"betarel{h}"] = ridge_fit(Lc[pm], Srel[pm])
        keep[f"m3coef{h}"] = m3_coef(Lc, A, pm, np.arange(16), ("a", "g"))
        keep[f"m1coef{h}"] = m3_coef(Lc, A, pm, np.arange(16), ("a",))[0]
        log(f"lead {h}: done", json.dumps({"static": out["static"]["all"]["static_share"], "lin": out["ridge_static"]["lin"]["R2_oos"]}))
        del A
    pmb = keep["pm1"] & keep["pm3"]
    c13 = corr0(keep["S1"][pmb], keep["S3"][pmb])
    res22["static_corr_lead1_lead3"] = dict(raw=c13, noise_corrected=c13 / np.sqrt(keep["rel1"] * keep["rel3"]))
    # 6. prediction (not a result): antithetic pair correlation from the measured shares
    pred = {}
    for h in (1, 3):
        o = res22[f"lead{h}"]["per_forecast"]
        pred[f"lead{h}"] = {}
        for rk in list(RNAME) + ["all"]:
            lin = o["static+level_lin"][rk]["R2"]
            ev = max(o["even_static+level"][rk]["R2"], 0.0)
            rho = -lin
            rho2 = -lin + ev
            pred[f"lead{h}"][rk] = dict(linear_share=lin, even_share=ev, rho_pred=rho, n_equiv_random=2 / (1 + rho),
                                        rho_pred_with_even=rho2, n_equiv_with_even=2 / (1 + rho2))
    R["prediction_antithetic"] = pred
    R["y2022"] = res22
    log("2022 done")
    # 7. cross-check: 64 fields on 15 start days x 2 leads of 2021 (fields 16-63 never used in a 2022 fit)
    F64 = feature_sets(feat.transpose(1, 0, 2))["lin"]
    res21 = {}
    for li, h in enumerate((1, 3)):
        doy = meta["keys64"][li, :, 0]
        A = accumulate(fc64, li, doy, 1, extra16=True)
        T = A["T"]; pm = ~A["bad"]
        S1, S2 = tot(A, "S1"), tot(A, "S2")
        S = S1 / T
        st = static_share(S1, S2, T, pm)
        mlev = tot(A, "Smb") / T
        preg = np.digitize(mlev, REG)
        o = dict(T=T, valid_pixels=int(pm.sum()), static=st, pixels_by_level=[int(((preg == r) & pm).sum()) for r in range(3)])
        b22 = keep[f"beta{h}"]
        p22 = F64 @ b22
        new = np.arange(16, 64)
        o["law2022_on_fields16_63"] = dict(R2=r2(S[:, new], p22[:, new], pm), ceiling=st["ceiling"])
        o["law2022_on_fields16_63"]["R2_over_ceiling"] = o["law2022_on_fields16_63"]["R2"] / st["ceiling"]
        o["law2022_on_fields16_63"]["corr"] = corr0(S[pm][:, new], p22[pm][:, new])
        o["law2022_on_fields16_63"]["amplitude_ratio_fit"] = float((S[pm][:, new] * p22[pm][:, new]).sum()
                                                                   / (p22[pm][:, new] ** 2).sum())
        for r in range(3):
            mk = pm & (preg == r)
            o["law2022_on_fields16_63"][f"R2_pixlev_{RNAME[r]}"] = r2(S[:, new], p22[:, new], mk) if mk.sum() else None
        Srel = A["S1rel"] / T
        st_rel = static_share(A["S1rel"], A["S2rel"], T, pm)
        prel = F64 @ keep[f"betarel{h}"]
        o["rel_law2022_on_fields16_63"] = dict(R2=r2(Srel[:, new], prel[:, new], pm), ceiling=st_rel["ceiling"],
                                               amplitude_ratio_fit=float((Srel[pm][:, new] * prel[pm][:, new]).sum()
                                                                         / (prel[pm][:, new] ** 2).sum()))
        p21, b21 = ridge_cv(F64, S, pm, 64)
        o["law2021_own_folds"] = dict(R2_oos=r2(S, p21, pm), beta_ug_per_sd=dict(zip(LNAME, b21.mean(0).round(3).tolist())),
                                      beta2022_ug_per_sd=dict(zip(LNAME, b22.round(3).tolist())),
                                      corr_beta_2021_2022=float(np.corrcoef(b21.mean(0), b22)[0, 1]))
        # per-forecast transfer of the 2022 static+level law (fitted on all 16 fields of 2022) to the unseen fields
        b_, g_ = keep[f"m3coef{h}"]
        o["law2022_per_forecast_static+level_fields16_63"] = m3_eval(A, pm, F64 @ b_, F64 @ g_, new)
        o["law2022_per_forecast_static_only_fields16_63"] = m3_eval(A, pm, F64 @ keep[f"m1coef{h}"], np.zeros_like(S), new)
        S16 = A["S1_16"] / T
        S22 = keep[f"S{h}"]
        mk = pm & keep[f"pm{h}"]
        o["static_map_fields0_15_corr_2021_vs_2022"] = corr0(S16[mk], S22[mk])
        o["shape"] = dict(excess_kurt=A["g4"].sum() / A["gn"].sum() - 3, skew=A["g3"].sum() / A["gn"].sum(),
                          gaussian_ref_excess_kurt=gauss_ref(64))
        res21[f"lead{h}"] = o
        log(f"2021 lead {h} done", json.dumps(o["law2022_on_fields16_63"]))
        del A
    R["y2021_run64"] = res21
    R["seconds"] = round(time.time() - T0)
    out = OUTF
    txt = json.dumps(R, indent=1, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else float(v))
    tmp = out.parent / f"{out.name}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(txt); f.flush(); os.fsync(f.fileno())
    assert tmp.read_text() == txt
    os.replace(tmp, out)
    assert out.read_text() == txt
    log("written", out)


if __name__ == "__main__":
    main()
