#!/usr/bin/env python3
"""A18 scorer (frozen: its sha256 is printed in a18/A18_draft.md section 7 and written by this file into every output as
'score18_sha256'). CPU only, stored data only. Two questions, every score on 2021 (15 strong-episode start days of the
64-field run a14/r2/data/run64, leads 1 and 3, truth = 1 km GHAP estimates); the 16 FIXED-design members of 2022 serve
ONLY to fit the noise law (no 2022 truth exists in work/, none is read).

Law (fit_law, the per-forecast level law of a17/diag/analyse.py m3_fit with use ('g',)): for a member k with noise field
z_k and 12 linear features F_k(p) of that field at pixel p (collect18.py), D_k = x_k - xbar = xbar * (F_k - Fbar) . gamma,
least squares pooled over pixels, members and forecasts of 2022 (relative ridge 1e-3, as a17/diag), one gamma per lead.
Applied to a new field: g_k = L(z_k) = F_k . gamma (uncentred: E[F] = 0 under white noise, so E[g] = 0).
Also fitted (reported only): the static + level law D = (F - Fbar) . beta + xbar (F - Fbar) . gamma_s (a17/diag headline).
Residual model (2022 members only): with e_k = D_k - xbar (F_k - Fbar) . gamma, kappa2(bin) = sum s2(e) / sum xbar^2 and
tau2(bin) = sum s2(D) / sum xbar^2 (s2 over the 16 members, ddof 1) in bins of xbar (edges LEV); sigmaL2 = mean over the
16 fields and the pixels of g^2 (the analytic gamma' C gamma is reported beside).

Q1 (noise-aware mean). For a set S of M of the 48 unseen fields (16-63): xbar = mean x_k, gbar = mean g_k,
  m1 = xbar - xbar * gbar   (the mean of the members corrected with m_hat = xbar: iterated once),
  c_k = x_k - m1 * g_k,  cbar = xbar - m1 * gbar   (the noise-aware mean; = xbar (1 - gbar + gbar^2)).
Variants reported: no iteration (m_hat = xbar), exact fixed point xbar / max(1 + gbar, 0.5), the static + level law,
the law refitted on 2021 fields 0-15 (members only; sensitivity). Sets: M = 1: the 48 fields; M = 2, 4, 8, 16: NPERM = 4
fixed random permutations of the 48 fields (default_rng(PSEED)), each cut into 48 / M disjoint sets.
1/M law from all 48: MSE_inf = mean[(xbar48 - y)^2 - s2_48 / 48], V = mean s2_48 (ddof 1); MSE_M = MSE_inf + V / M.
inv_meq = (MSE - MSE_inf) / V, M_eq = 1 / inv_meq (interval inverted from that of inv_meq).
Truth-free M_eq (review C1; members only, the filed prediction of M_eq): for a set S of M fields and the mean xr and
variance s2r (ddof 1) of the 48 - M OTHER fields of the same forecast, Var(est_S - mu) = mean[(est_S - xr)^2 - s2r / (48 - M)],
inv_tf = that / V, M_eq_tf = 1 / inv_tf, for est = cbar ('corr') and xbar ('plain', whose expectation is exactly 1 / M).
Q2 (analytic exceedance probability, thresholds 37.5 and 75 ug m-3). From M = 2, 4, 8 members: log-normal with mean
cbar and relative variance v = sigmaL2 + kappa2(cbar) (1 + 1/M): s2 = ln(1 + v), P(x > T) = Phi((ln cbar - s2/2 - ln T) / s)
(0 where cbar <= 0). Reported beside: the Gaussian form (cbar, sd = cbar sqrt v), the no-law control (plain xbar,
v = tau2(xbar) (1 + 1/M): same form, no noise law: separates the law from parametric smoothing), the 2021-refit law,
and (review C6) the tau-kappa form 'tk': log-normal, mean cbar, v = tau2(cbar) + kappa2(cbar) / M (limit tau2, the
model's own spread). Smoothing part (review C3): BS(16 counted) - BS(no-law control from 4).
Counted frequency: share of the M members above T (M = 2, 4, 8, 16 from the permutations, 48). Brier score, its
reliability / resolution / uncertainty (10 probability bins, sets of the first permutation), warnings with the tie-split
rule (review C2): weight 1 when p > 1/2, 1/2 when p = 1/2 exactly (exactly half of the members), 0 otherwise.
Domain (review C5, reported only): Kish effective numbers of days and tiles (weights: residents; resident-weighted squared
error of the plain mean of 8), and the leave-one-day-out sign of D1_1 = MSE(8 plain) - MSE(4 corrected) at lead 1.
Weights: residents of the pixel (decision) and pixel count (reported). Parts: all; regime of the mean of 2021 fields 0-15
(< 15, 15-75, > 75 ug m-3; fields not in the test sample); coast (land share in 51 x 51 px < 0.9) and inland (>= 0.999).
Intervals: two-way bootstrap, 15 start days x 10-degree tiles (both leads resampled together), NBOOT = 2000, seed BSEED,
95 % percentile; every statistic a ratio of sums recomputed in each draw. Decisions: decide() (section 6 of A18): guards
G1-G3 (else INVALID); Q1 SUCCESS / ORDER / FAIL; Q2 SUCCESS / ORDER only with the attribution LAW (the closed form beats
the no-law control), DRESSING when the closed form beats counted members without it, else FAIL.
    python -B -u score18.py --tests   synthetic worlds only (work/ is refused) -> out/tests18.json
    python -B -u score18.py --plumb   real members only, truth refused (fit, transfer, feature checks) -> out/plumb18.json
    python -B -u score18.py           the A18 score (needs collect18.py --part all) -> out/score18.json"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
SELF_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
A18_TAG = "/project_462001804/ammar/global_model_forecasting_air_pollution/a18/"
MODE = {"truth": False, "work": True}


def _audit(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = os.fsdecode(args[0])
    pl = p.lower()
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    wr = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or \
         (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND)))
    if not wr and (pl.endswith((".py", ".pyc", ".so")) or "/site-packages/" in pl):      # code, not data
        return
    if "station" in pl or "openaq" in pl or "monitor" in pl or pl.rstrip("/").endswith("_ar") or "_ar/" in pl \
            or "/a16/data" in pl or "/a16/results" in pl or "ghap" in pl:
        raise PermissionError(f"A18: refused path {p}")
    if ("/a18/work" in pl and not MODE["work"]) or ("truth" in os.path.basename(pl) and not MODE["truth"]):
        raise PermissionError(f"A18: refused in this mode: {p}")
    if wr:
        rp = os.path.realpath(p)
        if not (A18_TAG in rp or A18_TAG in os.path.abspath(p) or rp.startswith("/tmp/") or rp.startswith("/dev/")):
            raise PermissionError(f"A18: write outside a18 refused: {p}")


sys.addaudithook(_audit)
WORK = HERE.parent / "work"
OUT = HERE / "out"
REG = (15.0, 75.0)
THR = (37.5, 75.0)
MS = (1, 2, 4, 8, 16)
MQ2 = (2, 4, 8)
MFR = (2, 4, 8, 16)
UNSEEN = np.arange(16, 64)
REGF = np.arange(16)
NPERM, PSEED = 4, 20261122
NBOOT, BSEED = 2000, 20261121
LAM = 1e-3
LEV = np.array([0.0, 2.0, 5.0, 10.0, 15.0, 25.0, 37.5, 50.0, 75.0, 100.0, 150.0, 250.0])
MINBIN = 1000
COAST_LT, INLAND_GE = 0.9, 0.999
VARIANTS = ("plain", "corr", "noiter", "fixpt", "statlev", "sens")
TFV = ("plain", "corr")                      # truth-free M_eq (review C1)
QN = ("an", "ga", "nl", "sz", "tk")          # closed forms; 'tk' added by review C6
PARTS = ("all", "lt15", "15to75", "gt75", "coast", "inland")
WN = ("res", "pix")
SIZES = np.array([1, 4, 9, 25, 81, 289, 1089, 4225, 16641, 66049, 263169, 1050625], np.float64)
CFEAT = np.sqrt(np.minimum.outer(SIZES, SIZES) / np.maximum.outer(SIZES, SIZES))   # nested boxes, white noise
T0 = time.time()
CTX = {}


def log(*a):
    print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)


def write_json(path, R):
    txt = json.dumps(R, indent=1, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else float(v))
    tmp = path.parent / f"{path.name}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(txt); f.flush(); os.fsync(f.fileno())
    assert tmp.read_text() == txt
    os.replace(tmp, path)
    assert path.read_text() == txt


# ============================================================================ the law (members and noise only)
def fill_bins(v, n):
    v = v.copy()
    ok = n >= MINBIN
    assert ok.any(), "no level bin has enough pixel-forecasts"
    good = np.flatnonzero(ok)
    for b in np.flatnonzero(~ok):
        v[b] = v[good[np.argmin(np.abs(good - b))]]
    return v


def fit_law(fc, cols, F):
    """fc (2, T, N, Mtot) members, cols the M member columns whose fields have features F (M, N, 12)"""
    cols = np.asarray(cols)
    M, N, K = F.shape
    assert M == len(cols)
    Fu = np.ascontiguousarray(np.asarray(F, np.float64).transpose(1, 0, 2))
    Fc = Fu - Fu.mean(1, keepdims=True)
    out = []
    for li in range(2):
        T = fc.shape[1]
        cnt, Smb, Smb2 = np.zeros(N), np.zeros(N), np.zeros(N)
        S1, SM, S2 = np.zeros((N, M)), np.zeros((N, M)), np.zeros((N, M))
        bad = np.zeros(N, bool)
        for t in range(T):
            x = np.asarray(fc[li, t], np.float64)[:, cols]
            bad |= ~np.isfinite(x).all(1)
            x = np.nan_to_num(x)
            mb = x.mean(1)
            D = x - mb[:, None]
            bad |= np.abs(D).max(1) == 0
            cnt += 1; Smb += mb; Smb2 += mb * mb; S1 += D; SM += D * mb[:, None]; S2 += D * D
        pm = ~bad
        Fp = Fc[pm].reshape(-1, K)
        blk = lambda wp: (Fp * np.repeat(wp[pm], M)[:, None]).T @ Fp
        XtX = blk(Smb2)
        gam = np.linalg.solve(XtX + LAM * np.trace(XtX) / K * np.eye(K), Fp.T @ SM[pm].reshape(-1))
        B12 = blk(Smb)
        XX = np.block([[blk(cnt), B12], [B12, XtX]])
        sol = np.linalg.solve(XX + LAM * np.trace(XX) / (2 * K) * np.eye(2 * K),
                              np.concatenate([Fp.T @ S1[pm].reshape(-1), Fp.T @ SM[pm].reshape(-1)]))
        beta_s, gam_s = sol[:K], sol[K:]
        g = Fc[pm] @ gam
        R2 = 1.0 - (S2[pm] - 2 * g * SM[pm] + g * g * Smb2[pm][:, None]).sum() / S2[pm].sum()
        a_, g_ = Fc[pm] @ beta_s, Fc[pm] @ gam_s
        R2s = 1.0 - (S2[pm] - 2 * (a_ * S1[pm] + g_ * SM[pm]) + a_ * a_ * cnt[pm][:, None] + 2 * a_ * g_ * Smb[pm][:, None]
                     + g_ * g_ * Smb2[pm][:, None]).sum() / S2[pm].sum()
        del g, a_, g_, S1, SM, S2
        gc = Fc @ gam
        nb = len(LEV) + 1
        se, st, sm, sn = np.zeros(nb), np.zeros(nb), np.zeros(nb), np.zeros(nb)
        for t in range(T):
            x = np.nan_to_num(np.asarray(fc[li, t], np.float64)[:, cols])[pm]
            mb = x.mean(1)
            D = x - mb[:, None]
            e = D - mb[:, None] * gc[pm]
            b = np.digitize(mb, LEV)
            se += np.bincount(b, (e * e).sum(1) / (M - 1), nb); st += np.bincount(b, (D * D).sum(1) / (M - 1), nb)
            sm += np.bincount(b, mb * mb, nb); sn += np.bincount(b, minlength=nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            k2, t2 = se / sm, st / sm
        out.append(dict(gamma=gam, beta_s=beta_s, gamma_s=gam_s, R2_level=float(R2), R2_static_level=float(R2s),
                        sigmaL2=float(((Fu[pm] @ gam) ** 2).mean()), sigmaL2_analytic=float(gam @ CFEAT @ gam),
                        kappa2=fill_bins(k2, sn), tau2=fill_bins(t2, sn), kappa2_raw=k2, tau2_raw=t2, n_bin=sn,
                        level_edges=LEV, valid_pixels=int(pm.sum()), N=int(N), T=int(T), members=int(M)))
        log(f"law lead index {li}: R2 level {R2:.4f}, static+level {R2s:.4f}, sigmaL2 {out[-1]['sigmaL2']:.4f} "
            f"(analytic {out[-1]['sigmaL2_analytic']:.4f}), valid {pm.sum()}")
    return out


# ============================================================================ per-forecast pixel statistics
def sets_by_m():
    rng = np.random.default_rng(PSEED)
    perms = [rng.permutation(48) for _ in range(NPERM)]
    S = {1: [(np.array([k]), True) for k in range(48)]}
    for M in MS[1:]:
        S[M] = [(perm[j * M:(j + 1) * M], n == 0) for n, perm in enumerate(perms) for j in range(48 // M)]
    return S, perms


def channel_names():
    ch = ["one", "e48", "inf48", "s2_48"]
    ch += [f"q1_{v}_{M}" for v in VARIANTS for M in MS]
    ch += [f"tf_{v}_{M}" for v in TFV for M in MS]
    ch += [f"sp_{v}_{M}" for v in ("plain", "corr") for M in (4, 8, 16)]
    for T in THR:
        ch.append(f"o_{T}")
        meth = [f"{nm}{M}" for nm in QN for M in MQ2] + [f"fr{M}" for M in MFR + (48,)]
        for m in meth:
            ch += [f"bs_{m}_{T}", f"w_{m}_{T}", f"h_{m}_{T}"]
    return ch


DEC = [f"{nm}{M}" for nm in ("an", "ga", "nl", "tk") for M in MQ2] + [f"fr{M}" for M in MFR + (48,)]


def warn(p):
    """tie-split warning (review C2): 1 if p > 1/2, 1/2 if p == 1/2 (exactly half of the members), 0 otherwise"""
    return (p > 0.5).astype(np.float64) + 0.5 * (p == 0.5)


def lookup(tab, m):
    return tab[np.digitize(m, LEV)]


def p_logn(m, v, T):
    s2 = np.log1p(v)
    s = np.sqrt(s2)
    return np.where(m > 0, ndtr((np.log(np.maximum(m, 1e-12)) - 0.5 * s2 - np.log(T)) / s), 0.0)


def p_gauss(m, v, T):
    return np.where(m > 0, ndtr((m - T) / (np.sqrt(v) * np.maximum(m, 1e-12))), 0.0)


def score_forecast(job):
    li, t = job
    C = CTX
    x = np.asarray(C["fc64"][li, t], np.float64)
    y = np.asarray(C["truth"][li, t], np.float64)
    fin = np.isfinite(x).all(1)
    valid = fin & np.isfinite(y) & (y > 0)
    x = np.where(fin[:, None], x, 0.0)
    y = np.where(valid, y, 0.0)
    reg = np.digitize(x[:, REGF].mean(1), REG)
    XU = np.ascontiguousarray(x[:, UNSEEN])
    del x
    L = C["laws"][li]
    ch = {k: 0.0 for k in C["channels"]}
    wv = (C["pop"] * valid, valid.astype(np.float64))
    dec = {(m, T, w): np.zeros((5, 10)) for m in DEC for T in THR for w in WN}
    obs = {T: (y > T).astype(np.float64) for T in THR}

    def add_dec(m, T, p, o):
        b = np.minimum((p * 10).astype(np.int64), 9)
        for wi, w in enumerate(WN):
            ww = wv[wi]
            for q, v in enumerate((ww, ww * p, ww * o, ww * p * p, ww * p * o)):
                dec[(m, T, w)][q] += np.bincount(b, v, 10)

    def add_prob(m, p, ns, do_dec):
        for T in THR:
            pp = p[T]
            o = obs[T]
            ch[f"bs_{m}_{T}"] = ch[f"bs_{m}_{T}"] + (pp - o) ** 2 / ns
            wn = warn(pp)
            ch[f"w_{m}_{T}"] = ch[f"w_{m}_{T}"] + wn / ns
            ch[f"h_{m}_{T}"] = ch[f"h_{m}_{T}"] + wn * o / ns
            if do_dec and m in DEC:
                add_dec(m, T, pp, o)

    xb48 = XU.mean(1)
    s2_48 = XU.var(1, ddof=1)
    sum48 = XU.sum(1)
    sq48 = (XU * XU).sum(1)
    ch["one"] = np.ones_like(y)
    ch["e48"] = (xb48 - y) ** 2
    ch["inf48"] = ch["e48"] - s2_48 / 48.0
    ch["s2_48"] = s2_48
    for T in THR:
        ch[f"o_{T}"] = obs[T]
    add_prob("fr48", {T: (XU > T).mean(1) for T in THR}, 1, True)
    gl_all = L["L"]
    D = XU - xb48[:, None]
    pr = xb48[:, None] * (gl_all - gl_all.mean(1, keepdims=True))
    g2 = (float(((D - pr) ** 2)[fin].sum()), float((D * D)[fin].sum()))
    del D, pr
    for M in MS:
        sets = C["sets"][M]
        ns = len(sets)
        for S, first in sets:
            xs = XU[:, S]
            xb = xs.mean(1)
            gl = gl_all[:, S]
            gb = gl.mean(1)
            m1 = xb - xb * gb
            cb = xb - m1 * gb
            ab = L["A"][:, S].mean(1)
            gbs = L["GS"][:, S].mean(1)
            m1s = xb - ab - xb * gbs
            gbz = L["Z"][:, S].mean(1)
            m1z = xb - xb * gbz
            cz = xb - m1z * gbz
            est = dict(plain=xb, corr=cb, noiter=m1, fixpt=xb / np.maximum(1.0 + gb, 0.5), statlev=xb - ab - m1s * gbs,
                       sens=cz)
            for v, e in est.items():
                ch[f"q1_{v}_{M}"] = ch[f"q1_{v}_{M}"] + (e - y) ** 2 / ns
            nr = 48 - M                                   # truth-free: the 48 - M other fields of this forecast
            xr = (sum48 - xs.sum(1)) / nr
            s2r = (sq48 - (xs * xs).sum(1) - nr * xr * xr) / (nr - 1)
            for v in TFV:
                ch[f"tf_{v}_{M}"] = ch[f"tf_{v}_{M}"] + ((est[v] - xr) ** 2 - s2r / nr) / ns
            if M in (4, 8, 16):
                ch[f"sp_plain_{M}"] = ch[f"sp_plain_{M}"] + xs.var(1, ddof=1) / ns
                ch[f"sp_corr_{M}"] = ch[f"sp_corr_{M}"] + (xs - m1[:, None] * gl).var(1, ddof=1) / ns
            if M in MQ2:
                fM = 1.0 + 1.0 / M
                vL = L["sL2"] + lookup(L["kappa2"], cb) * fM
                vN = lookup(L["tau2"], xb) * fM
                vZ = L["sL2_Z"] + lookup(L["kappa2_Z"], cz) * fM
                add_prob(f"an{M}", {T: p_logn(cb, vL, T) for T in THR}, ns, first)
                add_prob(f"ga{M}", {T: p_gauss(cb, vL, T) for T in THR}, ns, first)
                add_prob(f"nl{M}", {T: p_logn(xb, vN, T) for T in THR}, ns, first)
                add_prob(f"sz{M}", {T: p_logn(cz, vZ, T) for T in THR}, ns, first)
                vT = lookup(L["tau2"], cb) + lookup(L["kappa2"], cb) / M
                add_prob(f"tk{M}", {T: p_logn(cb, vT, T) for T in THR}, ns, first)
            if M in MFR:
                add_prob(f"fr{M}", {T: (xs > T).mean(1) for T in THR}, ns, first)
    NT = C["NT"]
    tid = C["tid"]
    ireg = reg * NT + tid
    icc = C["cc"] * NT + tid
    blocks = np.zeros((2, len(C["channels"]), 5, NT))
    for j, k in enumerate(C["channels"]):
        v = ch[k]
        assert isinstance(v, np.ndarray) and v.shape == y.shape, k
        for wi in range(2):
            q = wv[wi] * v
            blocks[wi, j, :3] = np.bincount(ireg, q, 3 * NT).reshape(3, NT)
            blocks[wi, j, 3:] = np.bincount(icc, q, 3 * NT).reshape(3, NT)[:2]
    return li, t, blocks, dec, g2, (int(valid.sum()), int(fin.sum()), int(len(y)))


# ============================================================================ bootstrap, statistics, decision
def boot_counts(n, B, seed):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.float64)


def six(a):
    """(..., 5 parts) -> (..., 6): all = sum of the three regimes"""
    return np.concatenate([a[..., :3].sum(-1, keepdims=True), a], -1)


def derive(Sd):
    """Sd: channel -> array (..., 2 leads, 2 weights, 6 parts) of sums; returns statistic -> array of the same shape"""
    with np.errstate(divide="ignore", invalid="ignore"):
        W = Sd["one"]
        q = lambda k: Sd[k] / W
        st = {"mse_inf": q("inf48"), "V": q("s2_48"), "mse_48": q("e48")}
        for M in (8, 16, 32, 64):
            st[f"law_{M}"] = st["mse_inf"] + st["V"] / M
        for v in VARIANTS:
            for M in MS:
                st[f"mse_{v}_{M}"] = q(f"q1_{v}_{M}")
                st[f"inv_{v}_{M}"] = (st[f"mse_{v}_{M}"] - st["mse_inf"]) / st["V"]
        for v in TFV:
            for M in MS:
                st[f"inv_tf_{v}_{M}"] = q(f"tf_{v}_{M}") / st["V"]
        st["d_p8_c4"] = st["mse_plain_8"] - st["mse_corr_4"]
        st["d_p4_c4"] = st["mse_plain_4"] - st["mse_corr_4"]
        st["d_p16_c4"] = st["mse_plain_16"] - st["mse_corr_4"]
        st["d_law8_c4"] = st["law_8"] - st["mse_corr_4"]
        for v in ("plain", "corr"):
            for M in (4, 8, 16):
                st[f"sperr_{v}_{M}"] = np.sqrt((M + 1) / M * q(f"sp_{v}_{M}")) / np.sqrt(st[f"mse_{v}_{M}"])
        for T in THR:
            meth = [f"{nm}{M}" for nm in QN for M in MQ2] + [f"fr{M}" for M in MFR + (48,)]
            for m in meth:
                st[f"bs_{m}_{T}"] = q(f"bs_{m}_{T}")
                st[f"warned_{m}_{T}"] = q(f"w_{m}_{T}")
                st[f"correct_{m}_{T}"] = Sd[f"h_{m}_{T}"] / Sd[f"w_{m}_{T}"]
                st[f"reach_{m}_{T}"] = Sd[f"h_{m}_{T}"] / Sd[f"o_{T}"]
            st[f"event_share_{T}"] = q(f"o_{T}")
            for ref in ("fr2", "fr4", "fr8", "fr16", "fr48", "nl4", "ga4", "an2", "an8", "tk4"):
                st[f"d_{ref}_an4_{T}"] = st[f"bs_{ref}_{T}"] - st[f"bs_an4_{T}"]
            st[f"d_fr16_nl4_{T}"] = st[f"bs_fr16_{T}"] - st[f"bs_nl4_{T}"]       # smoothing part (review C3)
            for ref in ("fr4", "fr16"):
                st[f"dcorrect_an4_{ref}_{T}"] = st[f"correct_an4_{T}"] - st[f"correct_{ref}_{T}"]
                st[f"dreach_an4_{ref}_{T}"] = st[f"reach_an4_{T}"] - st[f"reach_{ref}_{T}"]
    return st


def summarise(B, channels, NT):
    """B (2 leads, 2 w, nch, 15 days, 5 parts, NT) -> point and bootstrap statistics"""
    nl, nw, nch, ND, npart, _ = B.shape
    X = B.transpose(0, 1, 2, 4, 3, 5).reshape(-1, ND, NT)
    point = six(X.sum((1, 2)).reshape(nl, nw, nch, npart))
    Kd, Kt = boot_counts(ND, NBOOT, BSEED), boot_counts(NT, NBOOT, BSEED + 7)
    bt = np.empty((NBOOT, X.shape[0]))
    for c0 in range(0, X.shape[0], 256):
        tmp = X[c0:c0 + 256] @ Kt.T
        bt[:, c0:c0 + 256] = np.einsum("xdb,bd->bx", tmp, Kd)
    boot = six(bt.reshape(NBOOT, nl, nw, nch, npart))
    mk = lambda A: {k: A[..., j, :] for j, k in enumerate(channels)}
    est, bst = derive(mk(point)), derive(mk(boot))
    pooled_names = ("d_p4_c4", "d_p8_c4")
    R, raw = {}, {}
    warnings.simplefilter("ignore", RuntimeWarning)          # empty parts (e.g. > 75 on coasts) give NaN, by design
    for k, e in est.items():
        b = bst[k]
        lo, hi = np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)
        raw[k] = (e, lo, hi)
        R[k] = {f"lead{h}": {w: {p: [float(e[li, wi, pi]), float(lo[li, wi, pi]), float(hi[li, wi, pi])]
                                 for pi, p in enumerate(PARTS)} for wi, w in enumerate(WN)} for li, h in enumerate((1, 3))}
    pooled = {}
    for k in pooled_names:
        e, b = est[k].sum(0), bst[k].sum(1)
        lo, hi = np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)
        pooled[k] = {w: {p: [float(e[wi, pi]), float(lo[wi, pi]), float(hi[wi, pi])] for pi, p in enumerate(PARTS)}
                     for wi, w in enumerate(WN)}
    meq = {}
    names = [f"{v}_{M}" for v in VARIANTS for M in MS] + [f"tf_{v}_{M}" for v in TFV for M in MS]
    for nm in names:
        e, lo, hi = raw[f"inv_{nm}"]
        with np.errstate(divide="ignore", invalid="ignore"):
            me = np.where(e > 0, 1.0 / e, np.inf)
            mlo = np.where(hi > 0, 1.0 / hi, np.inf)
            mhi = np.where(lo > 0, 1.0 / lo, np.inf)
        meq[nm] = {f"lead{h}": {w: {p: [float(me[li, wi, pi]), float(mlo[li, wi, pi]), float(mhi[li, wi, pi])]
                                    for pi, p in enumerate(PARTS)} for wi, w in enumerate(WN)}
                   for li, h in enumerate((1, 3))}
    return R, pooled, meq


def domain(B, channels):
    """review C5, reported only: Kish effective numbers of days and tiles (residents, part all) and the leave-one-day-out
    D1_1 = MSE(8 plain) - MSE(4 corrected) at lead 1"""
    j = {k: i for i, k in enumerate(channels)}
    A = B[:, 0, :, :, :3, :].sum(3)                     # (2 leads, channels, days, tiles): residents, all
    kish = lambda w: float(w.sum() ** 2 / (w * w).sum())
    out = {}
    for li, h in enumerate((1, 3)):
        out[f"lead{h}"] = {nm: dict(days=kish(A[li, j[c]].sum(1)), tiles=kish(A[li, j[c]].sum(0)),
                                    n_days=int(A.shape[2]), n_tiles=int(A.shape[3]))
                           for nm, c in (("weight_residents", "one"), ("weight_sq_error_plain8", "q1_plain_8"))}
    a8, c4, w = (A[0, j[k]].sum(1) for k in ("q1_plain_8", "q1_corr_4", "one"))
    loo = [float((a8.sum() - a8[d] - c4.sum() + c4[d]) / (w.sum() - w[d])) for d in range(len(w))]
    out["loo_D1_lead1"] = dict(values_in_day_order=loo, positive=int(sum(v > 0 for v in loo)), days=len(loo))
    return out


def decomposition(dec):
    out = {}
    for (m, T, w), a in dec.items():
        Wb, Sp, So, Sp2, Spo = a
        W = Wb.sum()
        ob = So.sum() / W
        with np.errstate(divide="ignore", invalid="ignore"):
            pb, obb = Sp / Wb, So / Wb
            rel = np.nansum(Wb * (pb - obb) ** 2) / W
            res = np.nansum(Wb * (obb - ob) ** 2) / W
        bs = (Sp2.sum() - 2 * Spo.sum() + So.sum()) / W
        unc = ob * (1 - ob)
        out[f"{m}|{T}|{w}"] = dict(BS=float(bs), REL=float(rel), RES=float(res), UNC=float(unc),
                                   within_bin=float(bs - (rel - res + unc)), weight_by_bin=(Wb / W).round(6).tolist())
    return out


def decide(R, guards):
    s = lambda k, h: R["stats"][k][f"lead{h}"]["res"]["all"]
    pool = R["pooled"]["d_p4_c4"]["res"]["all"]
    if s("d_p8_c4", 1)[0] > 0 and s("d_p8_c4", 3)[0] > 0 and s("d_p8_c4", 1)[1] > 0:
        q1 = "SUCCESS"
    elif pool[1] > 0:
        q1 = "ORDER"
    else:
        q1 = "FAIL"
    attr = "LAW" if s("d_nl4_an4_37.5", 1)[1] > 0 else "NOT-LAW"
    beats16 = s("d_fr16_an4_37.5", 1)[0] > 0 and s("d_fr16_an4_75.0", 1)[0] > 0 and s("d_fr16_an4_37.5", 1)[1] > 0
    beats4 = s("d_fr4_an4_37.5", 1)[1] > 0
    if beats16 and attr == "LAW":
        q2 = "SUCCESS"
    elif beats4 and attr == "LAW":
        q2 = "ORDER"
    elif beats16 or beats4:
        q2 = "DRESSING"          # the closed form beats counted members, but not through the noise law
    else:
        q2 = "FAIL"
    ok = bool(all(guards.values()))
    return dict(guards_passed=ok, Q1=q1 if ok else "INVALID", Q2=q2 if ok else "INVALID", Q1_rule=q1, Q2_rule=q2,
                Q2_attribution=attr)


def laws_at(F64, law, lawz):
    FU = np.ascontiguousarray(np.asarray(F64[UNSEEN], np.float64).transpose(1, 0, 2))
    out = []
    for li in range(2):
        a, z = law[li], lawz[li]
        out.append(dict(L=FU @ a["gamma"], A=FU @ a["beta_s"], GS=FU @ a["gamma_s"], Z=FU @ z["gamma"],
                        sL2=a["sigmaL2"], kappa2=a["kappa2"], tau2=a["tau2"], sL2_Z=z["sigmaL2"], kappa2_Z=z["kappa2"]))
    return out


def score(fc64, truth, F64, pop, land, r, c, law, lawz, workers):
    N = len(pop)
    lat, lon = 84.0 - 0.01 * (r + 0.5), -180.0 + 0.01 * (c + 0.5)
    t10 = ((84.0 - lat) // 10).astype(np.int64) * 36 + ((lon + 180.0) // 10).astype(np.int64)
    tiles, tid = np.unique(t10, return_inverse=True)
    NT = len(tiles)
    cc = np.where(land < COAST_LT, 0, np.where(land >= INLAND_GE, 1, 2)).astype(np.int64)
    sets, perms = sets_by_m()
    channels = channel_names()
    CTX.clear()
    CTX.update(fc64=fc64, truth=truth, pop=np.asarray(pop, np.float64), tid=tid, NT=NT, cc=cc, sets=sets,
               channels=channels, laws=laws_at(F64, law, lawz))
    ND = fc64.shape[1]
    B = np.zeros((2, 2, len(channels), ND, 5, NT))
    dec_all, g2s, cnts = {}, np.zeros((2, 2)), np.zeros((2, 3))
    jobs = [(li, t) for li in range(2) for t in range(ND)]
    ctx = get_context("fork")
    with ctx.Pool(workers) if workers > 1 else _Serial() as pool:
        for li, t, blocks, dec, g2, cn in pool.imap_unordered(score_forecast, jobs, chunksize=1):
            B[li, :, :, t] = blocks
            for k, v in dec.items():
                kk = (li,) + k
                dec_all[kk] = dec_all.get(kk, 0) + v
            g2s[li] += g2
            cnts[li] += cn
            log(f"forecast lead index {li} day {t} done")
    stats, pooled, meq = summarise(B, channels, NT)
    R = dict(stats=stats, pooled=pooled, meq=meq, domain=domain(B, channels))
    R["decomposition"] = {f"lead{h}": decomposition({k[1:]: v for k, v in dec_all.items() if k[0] == li})
                          for li, h in enumerate((1, 3))}
    tr = [float(1 - g2s[li, 0] / g2s[li, 1]) for li in range(2)]
    vshare = [float(cnts[li, 0] / cnts[li, 2]) for li in range(2)]
    fshare = [float(cnts[li, 1] / cnts[li, 2]) for li in range(2)]
    s = lambda k, li: stats[k][f"lead{(1, 3)[li]}"]["res"]["all"][0]
    guards = {
        "G1_truth_valid_share_ge_0.5": all(v >= 0.5 for v in vshare),
        "G1_members_finite_share_ge_0.99": all(v >= 0.99 for v in fshare),
        "G2_law2022_R2_ge_0.2": all(law[li]["R2_level"] >= 0.2 for li in range(2)),
        "G2_transfer2021_R2_ge_0.2": all(v >= 0.2 for v in tr),
        "G3_identity_M1": all(abs(s("inv_plain_1", li) - 1.0) <= 1e-6 for li in range(2)),
        "G3_identity_M2": all(abs(s("inv_plain_2", li) - 0.5) <= 0.04 for li in range(2)),
        "G3_identity_M4": all(abs(s("inv_plain_4", li) - 0.25) <= 0.02 for li in range(2)),
    }
    R["checks"] = dict(transfer_R2_2021_unseen=tr, truth_valid_share=vshare, members_finite_share=fshare, tiles=int(NT),
                       pixels=int(N), start_days=int(ND), permutations=[p.tolist() for p in perms],
                       sets_per_M={str(M): len(v) for M, v in sets.items()})
    R["guards"] = guards
    R["decision"] = decide(R, guards)
    return R


class _Serial:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, f, jobs, chunksize=1):
        return map(f, jobs)


def law_summary(law):
    keep = ("gamma", "beta_s", "gamma_s", "R2_level", "R2_static_level", "sigmaL2", "sigmaL2_analytic", "kappa2",
            "tau2", "kappa2_raw", "tau2_raw", "n_bin", "level_edges", "valid_pixels", "N", "T", "members")
    return [{k: v[k] for k in keep} for v in law]


def headline(R):
    """compact view of the decision numbers (residents, all), both leads"""
    H = {}
    for h in (1, 3):
        g = lambda k: R["stats"][k][f"lead{h}"]["res"]["all"]
        H[f"lead{h}"] = dict(
            mse_plain={M: g(f"mse_plain_{M}") for M in MS}, mse_corr={M: g(f"mse_corr_{M}") for M in MS},
            law={M: g(f"law_{M}") for M in (8, 16, 32, 64)}, mse_inf=g("mse_inf"), mse_48=g("mse_48"), V=g("V"),
            meq_corr={M: R["meq"][f"corr_{M}"][f"lead{h}"]["res"]["all"] for M in MS},
            meq_corr_by_part={p: R["meq"]["corr_4"][f"lead{h}"]["res"][p] for p in PARTS},
            prediction_meq_truth_free_corr={M: R["meq"][f"tf_corr_{M}"][f"lead{h}"]["res"]["all"] for M in MS},
            meq_truth_free_plain={M: R["meq"][f"tf_plain_{M}"][f"lead{h}"]["res"]["all"] for M in MS},
            d_p8_c4=g("d_p8_c4"), d_p4_c4=g("d_p4_c4"),
            bs={f"{m}_{T}": g(f"bs_{m}_{T}") for T in THR for m in ("fr4", "fr16", "fr48", "an4", "nl4", "tk4")},
            d_fr16_an4={T: g(f"d_fr16_an4_{T}") for T in THR}, d_fr4_an4={T: g(f"d_fr4_an4_{T}") for T in THR},
            d_nl4_an4={T: g(f"d_nl4_an4_{T}") for T in THR}, d_fr16_nl4={T: g(f"d_fr16_nl4_{T}") for T in THR},
            d_tk4_an4={T: g(f"d_tk4_an4_{T}") for T in THR},
            warnings_tie_split={T: {m: dict(warned=g(f"warned_{m}_{T}"), correct=g(f"correct_{m}_{T}"),
                                            reach=g(f"reach_{m}_{T}")) for m in ("fr4", "fr16", "fr48", "an4", "nl4")}
                                for T in THR},
            d_warn_an4_minus={T: {r: dict(correct=g(f"dcorrect_an4_{r}_{T}"), reach=g(f"dreach_an4_{r}_{T}"))
                                  for r in ("fr4", "fr16")} for T in THR},
            mse_net_space_plain4=g("mse_plain_4")[0] / 400.0, mse_net_space_corr4=g("mse_corr_4")[0] / 400.0)
    H["pooled_d_p4_c4"] = R["pooled"]["d_p4_c4"]["res"]["all"]
    H["domain"] = R["domain"]
    H["decision"] = R["decision"]
    return H


# ============================================================================ synthetic worlds (test mode)
PROF = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.15, 0.35, 0.7, 1.0, 0.6, 0.25, 0.05])
WORLDS = {  # form, s (law amplitude), kappa (residual), source of the members' noise in 2022 / 2021 (F stored, H hidden)
    "law_log": ("log", 0.35, 0.10, "F", "F"),
    "law_lin": ("lin", 0.35, 0.00, "F", "F"),
    "null": ("log", 0.35, 0.10, "H", "H"),
    "partial": ("log", 0.20, 0.30, "F", "F"),
    "mismatch": ("log", 0.35, 0.10, "F", "H"),
}
EXPECT = {  # Q1 and guards fixed with the code; the Q2 entries of null and partial after one development run (N = 6000)
    # review: tf_meq4_in_truth_interval (perfect-model truth: the truth-free M_eq(4) lies in the truth-based interval)
    "law_log": dict(Q1="SUCCESS", Q2="SUCCESS", Q2_attribution="LAW", guards_passed=True, meq4_lead1_min=16.0,
                    tf_meq4_in_truth_interval=True),
    "law_lin": dict(Q1="SUCCESS", guards_passed=True, meq4_lead1_min=50.0),
    "null": dict(Q1_rule="FAIL", Q2_rule="DRESSING", guards_passed=False, Q2_attribution="NOT-LAW", inv4_abs_dev_max=0.03),
    "partial": dict(Q1="ORDER", Q2="ORDER", Q2_attribution="LAW", guards_passed=True, meq4_lead1_range=(4.0, 8.0)),
    "mismatch": dict(Q1_rule="FAIL", guards_passed=False, Q2_attribution="NOT-LAW", meq4_lead1_max=4.0),
}


def world(kind, seed, N=24000, T22=40):
    form, s, kap, src22, src21 = WORLDS[kind]
    rng = np.random.default_rng(seed)
    r = rng.integers(2400, 12000, N)
    c = rng.integers(0, 36000, N)
    pop = np.exp(rng.normal(np.log(1500.0), 1.3, N)).astype(np.float32)
    land = rng.choice(np.array([0.5, 0.95, 1.0], np.float32), N, p=[0.2, 0.1, 0.7])
    Lc = np.linalg.cholesky(CFEAT)
    F = (rng.standard_normal((64, N, 12)) @ Lc.T).astype(np.float32)
    H = rng.standard_normal((64, N, 12)) @ Lc.T
    gu = PROF / np.sqrt(PROF @ CFEAT @ PROF)
    src = {"F": F.astype(np.float64), "H": H}

    def law(mu, u, eps):
        if form == "log":
            return mu * np.exp(s * u - s * s / 2 + kap * eps - kap * kap / 2)
        return mu * (1.0 + s * u + kap * eps)

    a_p = rng.normal(np.log(20.0), 0.8, N)
    U22 = (src[src22][:16] @ gu).T
    U21 = (src[src21] @ gu).T
    fc16 = np.empty((2, T22, N, 16), np.float16)
    fc64 = np.empty((2, 15, N, 64), np.float16)
    truth = np.empty((2, 15, N), np.float32)
    for li in range(2):
        mu = np.exp(a_p[None] + rng.normal(0.0, 0.5, (T22, N)))
        for t in range(T22):
            fc16[li, t] = law(mu[t][:, None], U22, rng.standard_normal((N, 16)))
        mu = np.exp(a_p[None] + rng.normal(0.0, 0.5, (15, N)))
        for t in range(15):
            fc64[li, t] = law(mu[t][:, None], U21, rng.standard_normal((N, 64)))
            y = law(mu[t], rng.standard_normal(N), rng.standard_normal(N))
            y[rng.random(N) < 0.02] = np.nan
            truth[li, t] = y
    return dict(fc16=fc16, fc64=fc64, truth=truth, F=F, pop=pop, land=land, r=r, c=c)


def check_expect(kind, R):
    E, d = EXPECT[kind], R["decision"]
    got = {}
    for k in ("Q1", "Q2", "Q1_rule", "Q2_rule", "Q2_attribution", "guards_passed"):
        if k in E:
            got[k] = d[k] == E[k]
    m4 = R["meq"]["corr_4"]["lead1"]["res"]["all"][0]
    if "meq4_lead1_min" in E:
        got["meq4_lead1_min"] = m4 > E["meq4_lead1_min"]
    if "meq4_lead1_max" in E:
        got["meq4_lead1_max"] = m4 < E["meq4_lead1_max"]
    if "meq4_lead1_range" in E:
        got["meq4_lead1_range"] = E["meq4_lead1_range"][0] < m4 < E["meq4_lead1_range"][1]
    if "inv4_abs_dev_max" in E:
        got["inv4_abs_dev_max"] = all(abs(R["stats"]["inv_corr_4"][f"lead{h}"]["res"]["all"][0] - 0.25)
                                      <= E["inv4_abs_dev_max"] for h in (1, 3))
    if E.get("tf_meq4_in_truth_interval"):
        lo_hi = R["meq"]["corr_4"]["lead1"]["res"]["all"]
        got["tf_meq4_in_truth_interval"] = lo_hi[1] <= R["meq"]["tf_corr_4"]["lead1"]["res"]["all"][0] <= lo_hi[2]
    got["G3_identities"] = all(v for k, v in R["guards"].items() if k.startswith("G3"))
    tf = lambda M, h: R["stats"][f"inv_tf_plain_{M}"][f"lead{h}"]["res"]["all"][0]
    got["tf_plain_identity"] = all(abs(tf(1, h) - 1.0) <= 0.04 and abs(tf(4, h) - 0.25) <= 0.02 for h in (1, 3))
    return got


def run_tests(workers):
    MODE["work"] = False
    MODE["truth"] = False
    out = {"what": __doc__, "score18_sha256": SELF_SHA256, "expectations": EXPECT, "worlds": WORLDS, "results": {}}
    tie = warn(np.array([0.25, 0.5, 0.75, 0.5 + 1e-12]))
    out["warn_tie_split_check"] = bool(np.array_equal(tie, [0.0, 0.5, 1.0, 1.0]))
    allok = out["warn_tie_split_check"]
    for n, kind in enumerate(WORLDS):
        W = world(kind, 777 + n)
        law = fit_law(W["fc16"], np.arange(16), W["F"][:16])
        lawz = fit_law(W["fc64"], np.arange(16), W["F"][:16])
        R = score(W["fc64"], W["truth"], W["F"], W["pop"], W["land"], W["r"], W["c"], law, lawz, workers)
        ok = check_expect(kind, R)
        allok &= all(ok.values())
        out["results"][kind] = dict(passed=ok, decision=R["decision"], guards=R["guards"], checks={
            k: v for k, v in R["checks"].items() if k != "permutations"}, law=law_summary(law), headline=headline(R))
        log(f"world {kind}: {R['decision']} meq4 lead1 {R['meq']['corr_4']['lead1']['res']['all']} checks {ok}")
    out["all_passed"] = bool(allok)
    out["seconds"] = round(time.time() - T0)
    write_json(OUT / "tests18.json", out)
    log("tests written, all passed:", allok)
    return allok


def load_work(truth):
    meta = np.load(WORK / "meta.npz")
    fc16 = np.load(WORK / "fc16.npy", mmap_mode="r")
    fc64 = np.load(WORK / "fc64.npy", mmap_mode="r")
    F = np.load(WORK / "feat.npy")
    tr = np.load(WORK / "truth.npy", mmap_mode="r") if truth else None
    assert fc16.shape[2] == fc64.shape[2] == F.shape[1] == len(meta["pop"]), "work arrays of different pixel samples"
    return meta, fc16, fc64, F, tr


def run_plumb():
    MODE["truth"] = False
    meta, fc16, fc64, F, _ = load_work(False)
    law = fit_law(fc16, np.arange(16), F[:16])
    lawz = fit_law(fc64, np.arange(16), F[:16])
    FU = np.asarray(F[UNSEEN], np.float64).transpose(1, 0, 2)
    tr = []
    for li in range(2):
        g = FU @ law[li]["gamma"]
        sse = ss = 0.0
        for t in range(fc64.shape[1]):
            x = np.asarray(fc64[li, t], np.float64)[:, UNSEEN]
            ok = np.isfinite(x).all(1)
            xb = x.mean(1, keepdims=True)
            D = x - xb
            pr = xb * (g - g.mean(1, keepdims=True))
            sse += ((D - pr) ** 2)[ok].sum(); ss += (D * D)[ok].sum()
        tr.append(float(1 - sse / ss))
    Ff = np.asarray(F, np.float64).reshape(64 * F.shape[1], 12)
    Cemp = Ff.T @ Ff / len(Ff)
    out = {"what": "A18 plumbing check on the real stored MEMBERS (no truth): " + __doc__.split("\n")[0],
           "score18_sha256": SELF_SHA256, "pixels": int(F.shape[1]), "law2022": law_summary(law),
           "law2021_fields0_15": law_summary(lawz), "transfer_R2_2021_unseen_level_law": tr,
           "feature_cov_max_abs_dev_from_white_noise": float(np.abs(Cemp - CFEAT).max()),
           "feature_sd_range": [float(np.sqrt(np.diag(Cemp)).min()), float(np.sqrt(np.diag(Cemp)).max())],
           "corr_gamma_2022_vs_2021_fields0_15": [float(np.corrcoef(law[li]["gamma"], lawz[li]["gamma"])[0, 1])
                                                  for li in range(2)],
           "collect_checks": json.loads(str(meta["checks"])), "seconds": round(time.time() - T0)}
    write_json(OUT / "plumb18.json", out)
    log("plumb written", json.dumps({k: out[k] for k in ("transfer_R2_2021_unseen_level_law",
                                                          "feature_cov_max_abs_dev_from_white_noise")}))


def run_score(workers):
    MODE["truth"] = True
    meta, fc16, fc64, F, truth = load_work(True)
    law = fit_law(fc16, np.arange(16), F[:16])
    lawz = fit_law(fc64, np.arange(16), F[:16])
    R = score(fc64, truth, F, meta["pop"], meta["land51"], meta["r"], meta["c"], law, lawz, workers)
    R.update(what=__doc__, score18_sha256=SELF_SHA256, law2022=law_summary(law), law2021_fields0_15=law_summary(lawz),
             collect_checks=json.loads(str(meta["checks"])), keys64=meta["keys64"], seconds=round(time.time() - T0))
    R["headline"] = headline(R)
    write_json(OUT / "score18.json", R)
    log("score written", json.dumps(R["decision"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--plumb", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    log("score18 sha256", SELF_SHA256)
    if a.tests:
        sys.exit(0 if run_tests(a.workers) else 1)
    if a.plumb:
        run_plumb()
        return
    run_score(a.workers)


if __name__ == "__main__":
    main()
