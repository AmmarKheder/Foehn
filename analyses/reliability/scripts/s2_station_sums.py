#!/usr/bin/env python3
"""Step 2: per-forecast sums at the station pixels, from which every station score and its bootstrap interval follow
(s2b_station_scores.py). One row per forecast (start day, lead) and group (six networks + all).

Two samples, both on identical station-days for everything that is compared inside them:
  cal    official station with >= 30 valid days (6,324), measurement of the target day, the four members, the 1 km
         estimate (GHAP) of the target day at the station pixel and the climatological ensemble all available.
         Used for reliability, rank histograms, spread-to-error, coverage, Brier scores and the CRPS against the
         climatological ensemble, against the measurement (v = 0) and against the estimate at the same pixel (v = 1).
  paper  exactly the station-days of experiments/figures/figdata_station_crps.py (the 4.3 / 7.9 / 8.3 / 9.7 of the
         manuscript): official network, measurement of the target day and of the start day, climatological mean, and,
         up to 4 days, the CAMS forecast and Aurora at equal issuance. The climatological ensemble is added on them.

Climatological ensemble of a station and target day: GHAP at the station pixel on the days of 2017-2020 within +-7
calendar days of the target day (clim/st_ghap_2017_2020.npy; 60 or 61 days; values <= 0 are missing), i.e. exactly the
days whose mean is the paper's climatology (clim/st_clim_2022.npy; checked below).
    python s2_station_sums.py [final|before]      -> data/sums_<tag>.npz"""
import sys

import numpy as np

from rel_common import DATA, G, NETS, THR, fair_crps, fair_crps_nan, stations

TAG = sys.argv[1] if len(sys.argv) > 1 else "final"
ST = stations(); OBS, KEEP, SRC = ST["obs"], ST["keep"], ST["src"]
OFF_PAPER = SRC != 2                                            # the mask of figdata_station_crps.py
NS = len(SRC)
GM = np.stack([SRC == s for s in NETS] + [np.isin(SRC, list(NETS))]).astype(np.float64)   # (7 groups, stations)
GROUPS = list(NETS.values()) + ["All official networks"]
z = np.load(DATA / "st_final.npz")
K = z["keys"]; CAMS, AUR = z["cams"], z["aur"]
E_ALL = np.load(DATA / f"st_{TAG}.npz")["ens"] if TAG != "final" else z["ens"]
assert np.array_equal(np.load(DATA / f"st_{TAG}.npz")["keys"], K)
GH22 = z["ghap"]
M = E_ALL.shape[-1]
H = np.load(G / "clim" / "st_ghap_2017_2020.npy").astype(np.float32); H[~(H > 0)] = np.nan
CLIM = np.load(G / "clim" / "st_clim_2022.npy").astype(np.float32)
DOY = np.load(G / "meta.npz")["doy"][:1461]
SD_EDGES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 1e9])     # bins of ensemble spread (ug m-3)
NB = len(SD_EDGES) - 1
N = len(K)
R = {k: np.zeros((N, 7) + s) for k, s in dict(
    n=(), var=(), sd_n=(NB,), sd_var=(NB,), relc=(3, M + 1), hoc=(3, M), hox=(3, M),          # independent of the verification
    sse=(2,), sae=(2,), err=(2,), crps=(2,), rank=(2, M + 1), cover=(2,), below=(2,), above=(2,), sd_se=(2, NB),
    relo=(2, 3, M + 1), hoo=(2, 3, M), base=(2, 3),
    c_crps=(2,), c_sae=(2,), c_sse=(2,), c_err=(2,), c_bs=(2, 3), c_bsf=(2, 3), c_m=(), c_medae=(2,),
    medae=(2,), ghap_se=(), ghap_err=(), sse_single=(2,),
    # paper sample (against the measurement only)
    p_n=(), p_crps=(), p_sae=(), p_sse=(), p_cams_ae=(), p_cams_se=(), p_aur_ae=(), p_aur_se=(), p_clim_ae=(),
    p_clim_se=(), p_clim_crps=(), p_medae=(), p_clim_medae=(), p_var=(), p_err=(), p_cams_err=(), p_aur_err=()).items()}
rng = np.random.default_rng(20261119)
clim_check = 0.0
for n, (d, h, t) in enumerate(K):
    d, h, t = int(d), int(h), int(t)
    E = E_ALL[n].astype(np.float32)                              # (stations, M)
    y = OBS[:, t]; g = GH22[n].astype(np.float32)
    dd = np.abs(DOY - t); sel = np.minimum(dd, 365 - dd) <= 7
    Hc = H[sel]                                                  # (60 or 61, stations), NaN = missing
    with np.errstate(all="ignore"):
        cm = np.nanmean(Hc, 0); cmed = np.nanmedian(Hc, 0)
    both = np.isfinite(cm) & np.isfinite(CLIM[t])
    clim_check = max(clim_check, float(np.max(np.abs(cm[both] - CLIM[t][both]))) if both.any() else 0.0)
    fin = np.isfinite(E).all(1)
    mean = E.mean(1); var = E.var(1, ddof=1); med = np.median(E, 1)
    sd_bin = np.clip(np.digitize(np.sqrt(var), SD_EDGES) - 1, 0, NB - 1)
    cal = KEEP & np.isfinite(y) & fin & np.isfinite(g)
    m_c = np.isfinite(Hc).sum(0)
    cal &= m_c >= 2
    w = cal.astype(np.float64)
    R["n"][n] = GM @ w; R["var"][n] = GM @ np.where(cal, var, 0); R["c_m"][n] = GM @ np.where(cal, m_c, 0)
    oh = np.eye(NB)[sd_bin] * w[:, None]
    R["sd_n"][n] = GM @ oh; R["sd_var"][n] = GM @ (oh * np.nan_to_num(var)[:, None])
    kT = np.stack([(E > T).sum(1) for T in THR])                 # (3, stations): members above the threshold
    for j in range(3):
        R["relc"][n][:, j] = GM @ (np.eye(M + 1)[kT[j]] * w[:, None])
    # held-out member: bins by the M - 1 other members, the probability of each bin measured on the member left out
    ex = np.stack([(E > T) for T in THR]).astype(np.int8)        # (3, stations, M)
    for j in range(3):
        for mo in range(M):
            k3 = kT[j] - ex[j][:, mo]
            ohk = np.eye(M)[k3] * w[:, None]
            R["hoc"][n][:, j] += GM @ ohk; R["hox"][n][:, j] += GM @ (ohk * ex[j][:, mo][:, None])
    R["ghap_se"][n] = GM @ np.where(cal, (g - y) ** 2, 0); R["ghap_err"][n] = GM @ np.where(cal, g - y, 0)
    for v, tv in enumerate((y, g)):
        e = np.where(cal, mean - tv, 0.0)
        R["sse"][n][:, v] = GM @ e ** 2; R["sae"][n][:, v] = GM @ np.abs(e); R["err"][n][:, v] = GM @ e
        R["medae"][n][:, v] = GM @ np.where(cal, np.abs(med - tv), 0)
        R["sse_single"][n][:, v] = GM @ np.where(cal, ((E - tv[:, None]) ** 2).mean(1), 0)
        R["crps"][n][:, v] = GM @ np.where(cal, fair_crps(E, tv), 0)
        lt = (E < tv[:, None]).sum(1); eq = (E == tv[:, None]).sum(1)
        rk = lt + (rng.integers(0, 1 << 30, NS) % (eq + 1))       # ties broken at random
        R["rank"][n][:, v] = GM @ (np.eye(M + 1)[np.clip(rk, 0, M)] * w[:, None])
        inside = (tv >= E.min(1)) & (tv <= E.max(1))
        R["cover"][n][:, v] = GM @ (inside & cal); R["below"][n][:, v] = GM @ ((tv < E.min(1)) & cal)
        R["above"][n][:, v] = GM @ ((tv > E.max(1)) & cal)
        R["sd_se"][n][:, v] = GM @ (oh * (e ** 2)[:, None])
        cc, _ = fair_crps_nan(Hc, tv)
        R["c_crps"][n][:, v] = GM @ np.where(cal, cc, 0)
        ce = np.where(cal, cm - tv, 0.0)
        R["c_sae"][n][:, v] = GM @ np.abs(ce); R["c_sse"][n][:, v] = GM @ ce ** 2; R["c_err"][n][:, v] = GM @ ce
        R["c_medae"][n][:, v] = GM @ np.where(cal, np.abs(cmed - tv), 0)
        for j, T in enumerate(THR):
            o = (tv > T) & cal
            R["base"][n][:, v, j] = GM @ o
            R["relo"][n][:, v, j] = GM @ (np.eye(M + 1)[kT[j]] * o[:, None])
            for mo in range(M):
                k3 = kT[j] - ex[j][:, mo]
                R["hoo"][n][:, v, j] += GM @ (np.eye(M)[k3] * o[:, None])
            with np.errstate(all="ignore"):
                pc = np.nansum(Hc > T, 0) / np.maximum(m_c, 1)
            bs = (pc - o) ** 2
            R["c_bs"][n][:, v, j] = GM @ np.where(cal, bs, 0)
            R["c_bsf"][n][:, v, j] = GM @ np.where(cal, bs - pc * (1 - pc) / np.maximum(m_c - 1, 1), 0)
    # ---- the station-days of figdata_station_crps.py
    if t < 365 and d + 1 <= 364:
        pp = OFF_PAPER & np.isfinite(y) & fin & np.isfinite(OBS[:, d]) & np.isfinite(CLIM[t])
        if h <= 4:
            pp &= np.isfinite(CAMS[n]) & np.isfinite(AUR[n])
            if not np.isfinite(AUR[n]).any():
                pp[:] = False
        wp = pp.astype(np.float64)
        cc, _ = fair_crps_nan(Hc, y)
        R["p_n"][n] = GM @ wp
        for key, val in (("p_crps", fair_crps(E, y)), ("p_sae", np.abs(mean - y)), ("p_sse", (mean - y) ** 2),
                         ("p_err", mean - y), ("p_medae", np.abs(med - y)), ("p_var", var),
                         ("p_clim_ae", np.abs(CLIM[t] - y)), ("p_clim_se", (CLIM[t] - y) ** 2), ("p_clim_crps", cc),
                         ("p_clim_medae", np.abs(cmed - y))):
            R[key][n] = GM @ np.where(pp, val, 0)
        if h <= 4:
            for key, val in (("p_cams_ae", np.abs(CAMS[n] - y)), ("p_cams_se", (CAMS[n] - y) ** 2), ("p_cams_err", CAMS[n] - y),
                             ("p_aur_ae", np.abs(AUR[n] - y)), ("p_aur_se", (AUR[n] - y) ** 2), ("p_aur_err", AUR[n] - y)):
                R[key][n] = GM @ np.where(pp, val, 0)
    if n % 200 == 0:
        print(n, "/", N, flush=True)
np.savez(DATA / f"sums_{TAG}.npz", keys=K, groups=np.array(GROUPS), thr=np.array(THR), sd_edges=SD_EDGES, members=M, **R)
print(f"{TAG}: {N} forecasts; largest difference between the mean of the climatological ensemble and st_clim_2022.npy: "
      f"{clim_check:.2e} ug m-3")
for h in range(1, 8):
    k = K[:, 1] == h
    print(f"lead {h}: cal station-days {int(R['n'][k, 6].sum()):,} | paper station-days {int(R['p_n'][k, 6].sum()):,} | "
          f"paper CRPS {R['p_crps'][k, 6].sum() / R['p_n'][k, 6].sum():.3f} MAE mean {R['p_sae'][k, 6].sum() / R['p_n'][k, 6].sum():.3f} "
          f"RMSE {np.sqrt(R['p_sse'][k, 6].sum() / R['p_n'][k, 6].sum()):.3f}")
