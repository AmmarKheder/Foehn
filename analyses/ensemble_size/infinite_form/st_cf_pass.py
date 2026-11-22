#!/usr/bin/env python3
"""A14 class U, stations, scorer experiments/figures/compare_fair.py (the paper's RMSE at the stations 1-4 days ahead and the
gains over the CAMS forecast, Aurora, the climatology, the persistences, and of the 0.25 / 0.5 deg averages).

1. compare_fair.py is executed from its unchanged source with the arguments of the paper's run
   (crps_e01=<final run>:eval_final_crps_e01); its np.savez goes to data/exec_compare_fair_crps_e01.npz. Identity: S, PD,
   PDN, SST must equal the paper's experiments/figures/data/compare_fair_crps_e01.npz.
2. Its loop is replayed with the same expressions (the functions and dictionaries of the executed namespace): every
   system's per-day sum of squared errors and count must equal PD (and PDN by network), bit for bit. On the same
   station-days the replay adds, per network and start day:
     n, sse (float64), sum s2 of the four stored members (ddof 1), the delete-one-member sums (3 members), and for the
     0.25 / 0.5 deg averages sum (m0 - xbar)^2 / 3 (= s2/M from member 0, A14 4a) with m0 = the same reading of map25_m0.
   The station-day masks (leads 1-4) are stored for the sixteen-member terms (st_cf_score.py).
Stations are evaluation data: they enter only these sums.
    python st_cf_pass.py   -> data/st_cf_terms.npz, data/exec_compare_fair_crps_e01.npz, results/st_cf_identity.json"""
import sys
import time

import numpy as np

from u_common import DATA, FIGS, ISFX, RES, exec_script, safe_json, safe_npz

t0 = time.time()
SRC = FIGS / "compare_fair.py"
ARGV = [f"crps_e01{ISFX}=green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4:eval_final_crps_e01{ISFX}"]   # design: label and folder _ar
ns, out, sha, written = exec_script(SRC, ARGV, DATA)
print(f"\n[a14] compare_fair.py executed unchanged (sha256 {sha}) in {time.time() - t0:.0f} s; wrote {written}", flush=True)
ref = np.load(FIGS / "data" / f"compare_fair_crps_e01{ISFX}.npz")             # design: the S1 step-2 output of the same design
ident = {}
for k in ("S", "PD", "PDN", "SST"):
    a, b = np.asarray(ns[k]), ref[k]
    ident[k] = dict(equal=bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True)),
                    max_abs_diff=float(np.nanmax(np.abs(a - b))) if a.shape == b.shape else None)
print("[a14] identity with the paper's compare_fair_crps_e01.npz:", ident, flush=True)

runs, common, days = ns["runs"], ns["common"], ns["days"]
obs, official, GH, CLIM, gid = ns["obs"], ns["official"], ns["GH"], ns["CLIM"], ns["gid"]
cams, aur_d, aur_e, analysis00, coarse, bil = ns["cams"], ns["aur_d"], ns["aur_e"], ns["analysis00"], ns["coarse"], ns["bil"]
sets, NS, NG, SYS, NETS, PD, PDN = ns["sets"], ns["NS"], ns["NG"], ns["SYS"], ns["NETS"], ns["PD"], ns["PDN"]
index = runs[0][1]; ensmean = runs[0][2]
ND = len(days)
T = {k: np.zeros((NG, ND, 4)) for k in ("n", "sse", "s2", "q25", "q50", "sse25", "sse50", "bad25", "bad50")}
T["sse_jk"] = np.zeros((NG, ND, 4, 4)); T["s2_jk"] = np.zeros((NG, ND, 4, 4))
masks, mkeys = [], []
pc = {}


def part(p):
    if p not in pc:
        if len(pc) > 4:
            pc.pop(next(iter(pc)))
        z = np.load(p)
        pc[p] = (z["st_ens"], z["map25_m0"])
    return pc[p]


def coarse_m0(d, h, res):
    """compare_fair.coarse with map25_m0 in place of map25 (same arithmetic)"""
    p, n = index[(d, h)]
    m = part(p)[1][n].astype(np.float32)
    if res == 0.5:
        with np.errstate(all="ignore"):
            m = np.nanmean(m.reshape(288, 2, 720, 2), (1, 3))
    lat0, dl = (84.0 - 0.125, 0.25) if res == 0.25 else (84.0 - 0.25, 0.5)
    ok = np.isfinite(m).astype(np.float32)
    with np.errstate(all="ignore"):
        return bil(np.nan_to_num(m) * ok, lat0, dl, -180.0 + dl / 2, dl) / bil(ok, lat0, dl, -180.0 + dl / 2, dl), m


nchk = 0
for c, (d, h) in enumerate(common):
    if h > 4 or d + h >= 365 or d + 1 > 364 or d not in aur_d or (d + 1) not in aur_e or d not in GH:
        continue
    t = d + h
    y = obs[:, t]
    X = [r[2][(d, h)] for r in runs]
    c25, m25 = coarse(0, d, h, 0.25)
    c50, m50 = coarse(0, d, h, 0.5)
    X += [c25, c50]
    X.append(bil(np.asarray(cams[d, 0, h], np.float32), 90.0, 0.4, 0.0, 0.4))
    X.append(bil(np.asarray(cams[d + 1, 0, h - 1], np.float32), 90.0, 0.4, 0.0, 0.4))
    X.append(aur_d[d][h - 1])
    A = aur_e[d + 1]
    k0 = 2 * (h - 1)
    first = analysis00(d + 1) if h == 1 else A[k0 - 1]
    X.append(0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1])
    X.append(obs[:, d])
    X.append(GH[d])
    X.append(CLIM[t])
    X = np.stack(X)
    ok = official & np.isfinite(y) & np.all(np.isfinite(X), 0)
    di = days.index(d)
    for k in range(len(SYS)):                                        # identity of the replay with the executed loop
        e = X[k, ok] - y[ok]
        assert PD[k, di, h - 1, 0] == (e * e).sum() and PD[k, di, h - 1, 1] == ok.sum(), (d, h, k)
        for g in range(NG):
            mg = ok & ((gid == g) if g < len(NETS) else True)
            eg = X[k, mg] - y[mg]
            assert PDN[k, g, di, h - 1, 0] == (eg * eg).sum() and PDN[k, g, di, h - 1, 1] == mg.sum(), (d, h, k, g)
    # the four stored members at the stations, mapped as compare_fair maps their mean
    p, n = index[(d, h)]
    e4 = part(p)[0]; idx = sets[e4.shape[1]]
    E = np.full((NS, e4.shape[2]), np.nan, np.float32); E[idx[idx >= 0]] = e4[n, idx >= 0].astype(np.float32)
    assert np.array_equal(E.mean(-1)[ok], X[0][ok]) and np.array_equal(ensmean[(d, h)][ok], X[0][ok]), (d, h)
    s2 = E.var(-1, ddof=1)
    jk = [np.delete(E, i, -1) for i in range(E.shape[1])]
    jx = np.stack([J.mean(-1) for J in jk], -1); js2 = np.stack([J.var(-1, ddof=1) for J in jk], -1)
    g25, _ = coarse_m0(d, h, 0.25); g50, _ = coarse_m0(d, h, 0.5)
    b25, b50 = ~np.isfinite(g25), ~np.isfinite(g50)                  # member 0 missing where the mean is finite (counted)
    g25, g50 = np.where(b25, X[1], g25), np.where(b50, X[2], g50)
    y64 = y.astype(np.float64)
    for g in range(NG):
        mg = ok & ((gid == g) if g < len(NETS) else True)
        T["n"][g, di, h - 1] = mg.sum()
        T["sse"][g, di, h - 1] = ((X[0][mg].astype(np.float64) - y64[mg]) ** 2).sum()
        T["s2"][g, di, h - 1] = s2[mg].astype(np.float64).sum()
        T["sse_jk"][g, di, h - 1] = ((jx[mg].astype(np.float64) - y64[mg, None]) ** 2).sum(0)
        T["s2_jk"][g, di, h - 1] = js2[mg].astype(np.float64).sum(0)
        T["sse25"][g, di, h - 1] = ((X[1][mg].astype(np.float64) - y64[mg]) ** 2).sum()
        T["sse50"][g, di, h - 1] = ((X[2][mg].astype(np.float64) - y64[mg]) ** 2).sum()
        T["q25"][g, di, h - 1] = (((g25[mg] - X[1][mg]).astype(np.float64)) ** 2).sum() / 3.0
        T["q50"][g, di, h - 1] = (((g50[mg] - X[2][mg]).astype(np.float64)) ** 2).sum() / 3.0
        T["bad25"][g, di, h - 1] = (mg & b25).sum(); T["bad50"][g, di, h - 1] = (mg & b50).sum()
    masks.append(np.packbits(ok)); mkeys.append((d, h))
    nchk += 1
    if nchk % 100 == 0:
        print(f"[a14] replay {nchk} forecasts, {time.time() - t0:.0f} s", flush=True)

safe_npz(DATA / "st_cf_terms.npz", days=np.array(days), keys=np.array(mkeys), masks=np.array(masks), NS=NS,
         systems=np.array(SYS), nets=np.array(list(NETS.values()) + ["all"]), PD=PD, PDN=PDN, **T)
safe_json(RES / "st_cf_identity.json", dict(
    script=str(SRC), sha256=sha, argv=ARGV, executed_unchanged=True, written_redirected=written,
    identity_vs_paper_npz=ident, replay_forecasts=nchk,
    replay_identity="every system's per-day sum of squared errors and count, all networks, equal to the executed PD/PDN (asserted)",
    printed_tail=out[-6000:]))
print(f"[a14] done: {nchk} forecasts replayed, {time.time() - t0:.0f} s", flush=True)
