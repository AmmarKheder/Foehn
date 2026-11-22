#!/usr/bin/env python3
"""A14 class U, stations, scorer analyses/reliability (s2_station.py -> sums_final.npz -> s2b_station_scores.py): the extra
per-forecast sums that the eight-member forms and the jackknife over the four fields need and sums_final.npz lacks.
The samples of s2_station.py are rebuilt with its expressions ('cal': official station with >= 30 valid days, measurement,
four members, estimate and climatological ensemble of >= 2 days; 'paper': the station-days of figdata_station_crps.py) and
checked against sums_final.npz (counts equal, variance sums equal to rounding). Added, by forecast and group:
  pair (sum_{i<j}|x_i - x_j| / (M (M-1))) on cal and paper, its delete-one-member values (3 members);
  for v = measurement / estimate and the three thresholds: the Brier score of the 4 members, raw and fair, and their
  delete-one-member values (raw and fair, 3 members).
The cal / paper masks of every forecast are stored (sixteen-member terms, st16_score.py).
    python rel_pass.py   -> data/rel_extra.npz"""
import sys
import time

import numpy as np

from u_common import DATA, G, REL, RELD, safe_npz

sys.path.insert(0, str(REL / "scripts"))
from rel_common import NETS, THR, stations  # noqa: E402  (unchanged, no bytecode)

t0 = time.time()
ST = stations(); OBS, KEEP, SRC = ST["obs"], ST["keep"], ST["src"]
OFF_PAPER = SRC != 2
NS = len(SRC)
GM = np.stack([SRC == s for s in NETS] + [np.isin(SRC, list(NETS))]).astype(np.float64)
z = np.load(RELD / "st_final.npz")                                    # design: reliability/data_ar (S1 step 5)
K = z["keys"]; CAMS, AUR = z["cams"], z["aur"]; E_ALL = z["ens"]; GH22 = z["ghap"]
M = E_ALL.shape[-1]
H = np.load(G / "clim" / "st_ghap_2017_2020.npy").astype(np.float32); H[~(H > 0)] = np.nan
CLIM = np.load(G / "clim" / "st_clim_2022.npy").astype(np.float32)
DOY = np.load(G / "meta.npz")["doy"][:1461]
ref = np.load(RELD / "sums_final.npz")
assert np.array_equal(ref["keys"], K)
N = len(K)
Q = {}
mc_cache = {}


def put(name, n, vec):
    vec = np.asarray(vec, np.float64)
    if name not in Q:
        Q[name] = np.zeros((N, GM.shape[0]) + vec.shape[1:])
    Q[name][n] = np.tensordot(GM, vec, axes=(1, 0))


def mk(x, m):
    x = np.asarray(x, np.float64)
    return np.where(m.reshape((NS,) + (1,) * (x.ndim - 1)), np.nan_to_num(x), 0.0)


def m_c_of(t):
    if t not in mc_cache:
        dd = np.abs(DOY - t); sel = np.minimum(dd, 365 - dd) <= 7
        mc_cache[t] = np.isfinite(H[sel]).sum(0)
    return mc_cache[t]


def pair_of(Ez):
    m = Ez.shape[-1]
    Es = np.sort(Ez, -1); w = (2 * np.arange(m) - m + 1).astype(Ez.dtype)
    return (Es * w).sum(-1) / (m * (m - 1))


cal_masks, p_masks = [], []
for n, (d, h, t) in enumerate(K):
    d, h, t = int(d), int(h), int(t)
    E = E_ALL[n].astype(np.float32)
    y = OBS[:, t]; g = GH22[n].astype(np.float32)
    m_c = m_c_of(t)
    fin = np.isfinite(E).all(1)
    Ez = np.where(fin[:, None], E, 0.0)
    var = Ez.var(1, ddof=1)
    cal = KEEP & np.isfinite(y) & fin & np.isfinite(g) & (m_c >= 2)
    pp = np.zeros(NS, bool)
    if t < 365 and d + 1 <= 364:
        pp = OFF_PAPER & np.isfinite(y) & fin & np.isfinite(OBS[:, d]) & np.isfinite(CLIM[t])
        if h <= 4:
            pp &= np.isfinite(CAMS[n]) & np.isfinite(AUR[n])
    put("n", n, cal.astype(np.float64)); put("var", n, mk(var, cal)); put("p_n", n, pp.astype(np.float64))
    pair = pair_of(Ez)
    loo = [np.delete(Ez, mo, 1) for mo in range(M)]
    lpair = np.stack([pair_of(L) for L in loo], 1)
    put("pair", n, mk(pair, cal)); put("loo_pair", n, mk(lpair, cal))
    put("p_pair", n, mk(pair, pp)); put("p_loo_pair", n, mk(lpair, pp))
    ex = np.stack([(Ez > T) for T in THR], 1).astype(np.int8)          # (stations, 3, M)
    kT = ex.sum(2)
    for v, tv in enumerate((y, g)):
        s = f"_{v}"
        tvz = np.where(cal, tv, 0.0).astype(np.float32)
        o = np.stack([(tvz > T) & cal for T in THR], 1).astype(np.float64)
        p = kT / M
        put("bs" + s, n, mk((p - o) ** 2, cal)); put("bsf" + s, n, mk((p - o) ** 2 - p * (1 - p) / (M - 1), cal))
        k3 = kT[:, :, None] - ex                                        # (stations, 3, M): members above among the other three
        p3 = k3 / (M - 1); o3 = o[:, :, None]
        put("loo_bs" + s, n, mk((p3 - o3) ** 2, cal))
        put("loo_bsf" + s, n, mk((p3 - o3) ** 2 - p3 * (1 - p3) / (M - 2), cal))
    cal_masks.append(np.packbits(cal)); p_masks.append(np.packbits(pp))
    if n % 200 == 0:
        print(n, "/", N, f"{time.time() - t0:.0f} s", flush=True)

chk = {}
for a, b in (("n", "day__n"), ("p_n", "day__p_n")):
    chk[a] = bool(np.array_equal(Q[a], ref[b]))
for a in ("var", "bs_0", "bsf_0", "bs_1", "bsf_1"):
    r = ref["day__" + a]
    chk[a] = float(np.max(np.abs(Q[a] - r) / np.maximum(np.abs(r), 1e-9)))
print("identity with sums_final.npz (counts equal; largest relative difference of the sums):", chk, flush=True)
assert chk["n"] and chk["p_n"] and all(chk[a] < 1e-9 for a in ("var", "bs_0", "bsf_0", "bs_1", "bsf_1")), chk
safe_npz(DATA / "rel_extra.npz", keys=K, cal_masks=np.array(cal_masks), p_masks=np.array(p_masks), members=M,
         identity=np.array(str(chk)), **{f"day__{k}": v for k, v in Q.items()})
print(f"done {N} forecasts, {time.time() - t0:.0f} s", flush=True)
