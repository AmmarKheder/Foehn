#!/usr/bin/env python3
"""Step 2 (supersedes s2_station_sums.py, which was never run): sums at the station pixels from which every station score
and its interval follow (s2b_station_scores.py). Two aggregations of the same per-(forecast, station) terms:
  day  (forecast, group)        -> moving-block bootstrap over the start days (the convention of the paper)
  st   (lead, station)          -> bootstrap over 2-degree cells of stations (the noise fields are fixed in time, so the
                                   sampling of the noise is a sampling over places, not over days)
Groups: the six official networks + all of them.

Two samples, both on identical station-days for everything that is compared inside them:
  cal    official station with >= 30 valid days (6,324), measurement of the target day, the four members, the 1 km
         estimate (GHAP) of the target day at the station pixel and the climatological ensemble all available.
         Reliability, rank histograms, spread-to-error, coverage, Brier scores and the CRPS against the climatological
         ensemble, against the measurement (v = 0) and against the estimate at the same pixel (v = 1).
  paper  exactly the station-days of experiments/figures/figdata_station_crps.py (the 4.3 / 7.9 / 8.3 / 9.7 of the
         manuscript): official network, measurement of the target day and of the start day, climatological mean, and,
         up to 4 days, the CAMS forecast and Aurora at equal issuance. The climatological ensemble is added on them.

Climatological ensemble of a station and target day: GHAP at the station pixel on the days of 2017-2020 within +-7
calendar days of the target day (clim/st_ghap_2017_2020.npy; 60 or 61 days; values <= 0 are missing), i.e. exactly the
days whose mean is the paper's climatology (clim/st_clim_2022.npy; the largest difference is printed).
    python s2_station.py [final|before]      -> data/sums_<tag>.npz"""
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
zz = np.load(DATA / f"st_{TAG}.npz")
assert np.array_equal(zz["keys"], K)
E_ALL = zz["ens"]; GH22 = z["ghap"]
M = E_ALL.shape[-1]
H = np.load(G / "clim" / "st_ghap_2017_2020.npy").astype(np.float32); H[~(H > 0)] = np.nan
CLIM = np.load(G / "clim" / "st_clim_2022.npy").astype(np.float32)
DOY = np.load(G / "meta.npz")["doy"][:1461]
SD_EDGES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 1e9])     # bins of ensemble spread (ug m-3)
NB = len(SD_EDGES) - 1
N = len(K)
Q = {}                                                          # name -> [day (N, 7, ...), st (7 leads, NS, ...)]


def put(name, n, h, vec):
    """vec (NS, ...), already zero outside the sample"""
    vec = np.asarray(vec, np.float64)
    if name not in Q:
        Q[name] = [np.zeros((N, 7) + vec.shape[1:]), np.zeros((7, NS) + vec.shape[1:])]
    Q[name][0][n] = np.tensordot(GM, vec, axes=(1, 0))
    Q[name][1][h - 1] += vec


def mk(x, m):
    """x where the mask m (NS,) holds, 0 elsewhere (NaN-safe)"""
    x = np.asarray(x, np.float64)
    return np.where(m.reshape((NS,) + (1,) * (x.ndim - 1)), np.nan_to_num(x), 0.0)


rng = np.random.default_rng(20261119)
clim_cache = {}
clim_check = 0.0


def clim_of(t, g):
    """everything of the climatological ensemble that depends on the target day only"""
    if t not in clim_cache:
        if len(clim_cache) > 12:
            clim_cache.pop(next(iter(clim_cache)))
        dd = np.abs(DOY - t); sel = np.minimum(dd, 365 - dd) <= 7
        Hc = H[sel]                                             # (60 or 61, stations), NaN = missing
        m_c = np.isfinite(Hc).sum(0)
        with np.errstate(all="ignore"):
            cm = np.nanmean(Hc, 0); cmed = np.nanmedian(Hc, 0)
            pc = np.stack([np.nansum(Hc > T, 0) / np.maximum(m_c, 1) for T in THR])       # (3, NS)
        y = OBS[:, t]
        clim_cache[t] = dict(m=m_c, mean=cm, med=cmed, pc=pc, n_sel=int(sel.sum()),
                             crps=(fair_crps_nan(Hc, y)[0], fair_crps_nan(Hc, g)[0]))
    return clim_cache[t]


for n, (d, h, t) in enumerate(K):
    d, h, t = int(d), int(h), int(t)
    E = E_ALL[n].astype(np.float32)                              # (stations, M)
    y = OBS[:, t]; g = GH22[n].astype(np.float32)
    C = clim_of(t, g)
    m_c, cm, cmed, pc = C["m"], C["mean"], C["med"], C["pc"]
    both = np.isfinite(cm) & np.isfinite(CLIM[t])
    if both.any():
        clim_check = max(clim_check, float(np.max(np.abs(cm[both] - CLIM[t][both]))))
    fin = np.isfinite(E).all(1)
    Ez = np.where(fin[:, None], E, 0.0)
    mean = Ez.mean(1); var = Ez.var(1, ddof=1); med = np.median(Ez, 1)
    sd_bin = np.clip(np.digitize(np.sqrt(var), SD_EDGES) - 1, 0, NB - 1)
    cal = KEEP & np.isfinite(y) & fin & np.isfinite(g) & (m_c >= 2)
    w = cal.astype(np.float64)
    put("n", n, h, w); put("var", n, h, mk(var, cal)); put("c_m", n, h, mk(m_c, cal))
    oh = np.eye(NB)[sd_bin] * w[:, None]
    put("sd_n", n, h, oh); put("sd_var", n, h, oh * var[:, None])
    ex = np.stack([(Ez > T) for T in THR], 1).astype(np.int8)    # (stations, 3, M)
    kT = ex.sum(2)                                               # (stations, 3): members above the threshold
    relc = np.eye(M + 1)[kT] * w[:, None, None]                  # (stations, 3, M + 1)
    put("relc", n, h, relc)
    # held-out member: bins by the M - 1 other members; the frequency of each bin measured on the member left out is
    # what an ensemble whose verification is one more member would give
    hoc = np.zeros((NS, 3, M)); hox = np.zeros((NS, 3, M)); k3 = []
    for mo in range(M):
        k3.append(np.eye(M)[kT - ex[:, :, mo]])                  # (stations, 3, M)
        hoc += k3[-1]; hox += k3[-1] * ex[:, :, mo][:, :, None]
    put("hoc", n, h, hoc * w[:, None, None]); put("hox", n, h, hox * w[:, None, None])
    put("ghap_se", n, h, mk((g - y) ** 2, cal)); put("ghap_err", n, h, mk(g - y, cal)); put("ghap_ae", n, h, mk(np.abs(g - y), cal))
    # leave-one-member-out ensembles (jackknife over the fixed noise fields)
    loo = [np.delete(Ez, mo, 1) for mo in range(M)]
    put("loo_var", n, h, mk(np.stack([L.var(1, ddof=1) for L in loo], 1), cal))
    for v, tv in enumerate((y, g)):
        s = f"_{v}"
        tvz = np.where(cal, tv, 0.0).astype(np.float32)
        e = np.where(cal, mean - tvz, 0.0)
        put("sse" + s, n, h, e ** 2); put("sae" + s, n, h, np.abs(e)); put("err" + s, n, h, e)
        put("medae" + s, n, h, mk(np.abs(med - tvz), cal))
        put("mem_sse" + s, n, h, mk((Ez - tvz[:, None]) ** 2, cal)); put("mem_err" + s, n, h, mk(Ez - tvz[:, None], cal))
        put("crps" + s, n, h, mk(fair_crps(Ez, tvz), cal))
        put("loo_crps" + s, n, h, mk(np.stack([fair_crps(L, tvz) for L in loo], 1), cal))
        put("loo_sse" + s, n, h, mk(np.stack([(L.mean(1) - tvz) ** 2 for L in loo], 1), cal))
        lt = (Ez < tvz[:, None]).sum(1); eq = (Ez == tvz[:, None]).sum(1)
        rk = lt + (rng.integers(0, 1 << 30, NS) % (eq + 1))       # ties broken at random
        put("rank" + s, n, h, np.eye(M + 1)[np.clip(rk, 0, M)] * w[:, None])
        lo_, hi_ = Ez.min(1), Ez.max(1)
        put("cover" + s, n, h, ((tvz >= lo_) & (tvz <= hi_) & cal)); put("below" + s, n, h, ((tvz < lo_) & cal))
        put("above" + s, n, h, ((tvz > hi_) & cal))
        put("sd_se" + s, n, h, oh * (e ** 2)[:, None])
        put("c_crps" + s, n, h, mk(C["crps"][v], cal))
        ce = np.where(cal, np.nan_to_num(cm) - tvz, 0.0)
        put("c_sae" + s, n, h, np.abs(ce)); put("c_sse" + s, n, h, ce ** 2); put("c_err" + s, n, h, ce)
        put("c_medae" + s, n, h, mk(np.abs(cmed - tvz), cal))
        o = np.stack([(tvz > T) & cal for T in THR], 1).astype(np.float64)        # (stations, 3)
        put("base" + s, n, h, o)
        put("relo" + s, n, h, relc * o[:, :, None])
        hoo = np.zeros((NS, 3, M))
        for mo in range(M):
            hoo += k3[mo] * o[:, :, None]
        put("hoo" + s, n, h, hoo * w[:, None, None])
        p = kT / M
        put("bs" + s, n, h, mk((p - o) ** 2, cal)); put("bsf" + s, n, h, mk((p - o) ** 2 - p * (1 - p) / (M - 1), cal))
        pcz = np.nan_to_num(pc.T)                                # (stations, 3)
        put("c_bs" + s, n, h, mk((pcz - o) ** 2, cal))
        put("c_bsf" + s, n, h, mk((pcz - o) ** 2 - pcz * (1 - pcz) / np.maximum(m_c - 1, 1)[:, None], cal))
    # ---- the station-days of figdata_station_crps.py
    pp = np.zeros(NS, bool)
    if t < 365 and d + 1 <= 364:
        pp = OFF_PAPER & np.isfinite(y) & fin & np.isfinite(OBS[:, d]) & np.isfinite(CLIM[t])
        if h <= 4:
            pp &= np.isfinite(CAMS[n]) & np.isfinite(AUR[n])
    yz = np.where(pp, y, 0.0).astype(np.float32)
    put("p_n", n, h, pp.astype(np.float64))
    put("p_n_k", n, h, (pp & KEEP).astype(np.float64))           # of which stations with >= 30 valid days
    put("p_n_c2", n, h, (pp & (m_c >= 2)).astype(np.float64))    # of which a climatological ensemble of >= 2 days exists
    for key, val in (("p_crps", fair_crps(Ez, yz)), ("p_sae", np.abs(mean - yz)), ("p_sse", (mean - yz) ** 2),
                     ("p_err", mean - yz), ("p_medae", np.abs(med - yz)), ("p_var", var),
                     ("p_clim_ae", np.abs(CLIM[t] - yz)), ("p_clim_se", (CLIM[t] - yz) ** 2),
                     ("p_clim_crps", C["crps"][0]), ("p_clim_medae", np.abs(cmed - yz)),
                     ("p_loo_crps", np.stack([fair_crps(L, yz) for L in loo], 1)),
                     ("p_loo_sae", np.stack([np.abs(L.mean(1) - yz) for L in loo], 1)),
                     ("p_loo_sse", np.stack([(L.mean(1) - yz) ** 2 for L in loo], 1)),
                     ("p_loo_var", np.stack([L.var(1, ddof=1) for L in loo], 1))):
        put(key, n, h, mk(val, pp))
    cz, az = np.nan_to_num(CAMS[n]), np.nan_to_num(AUR[n])
    p4 = pp & (h <= 4)
    for key, val in (("p_cams_ae", np.abs(cz - yz)), ("p_cams_se", (cz - yz) ** 2), ("p_cams_err", cz - yz),
                     ("p_aur_ae", np.abs(az - yz)), ("p_aur_se", (az - yz) ** 2), ("p_aur_err", az - yz)):
        put(key, n, h, mk(val, p4))
    if n % 200 == 0:
        print(n, "/", N, flush=True)

np.savez(DATA / f"sums_{TAG}.npz", keys=K, groups=np.array(GROUPS), thr=np.array(THR), sd_edges=SD_EDGES, members=M,
         lat=ST["lat"], lon=ST["lon"], src=SRC, keep=KEEP,
         **{f"day__{k}": v[0] for k, v in Q.items()}, **{f"st__{k}": v[1].astype(np.float32) for k, v in Q.items()})
print(f"{TAG}: {N} forecasts; largest difference between the mean of the climatological ensemble and st_clim_2022.npy: "
      f"{clim_check:.2e} ug m-3")
D = {k: v[0] for k, v in Q.items()}
for h in range(1, 8):
    k = K[:, 1] == h
    s = lambda q: D[q][k, 6].sum()
    print(f"lead {h}: cal station-days {int(s('n')):,} | paper station-days {int(s('p_n')):,} (stations >= 30 d: {int(s('p_n_k')):,}; "
          f"clim ensemble >= 2: {int(s('p_n_c2')):,}) | paper CRPS {s('p_crps') / s('p_n'):.3f} MAE mean {s('p_sae') / s('p_n'):.3f} "
          f"RMSE {np.sqrt(s('p_sse') / s('p_n')):.3f} | clim MAE {s('p_clim_ae') / s('p_n'):.3f} clim-ensemble CRPS {s('p_clim_crps') / s('p_n'):.3f}"
          + (f" | CAMS MAE {s('p_cams_ae') / s('p_n'):.3f} Aurora MAE {s('p_aur_ae') / s('p_n'):.3f}" if h <= 4 else ""))
