#!/usr/bin/env python3
"""Integrity check of the final Figs. 3 and 5 (2026-11-21). Read only, independent code.
Recomputes from the inputs that FIG/compare_fair.py reads (eval_final_crps_e01_ar parts st_ens/st_ghap/map25, stations_2022,
CAMS daily zarr, Aurora out_global + out_12h_all + raw analyses, station climatology):
  1. RMSE by lead (1-4) of Foehn (four-member mean, 1 km pixel), CAMS run 00 UTC d+1 and Aurora 00 UTC d+1 (12 h model),
     on the paper mask (official network, measurement finite, the eleven systems of compare_fair finite), per network,
     six networks pooled, China, outside China; 7-day block bootstrap (10,000) of the RMSE ratio per group and lead;
  2. the boxes of Fig. 3b on 9 February (start 38, lead 1), figure pairing and paper mask, plus Aurora;
  3. the tally of the choice of day (Foehn below CAMS in all four basins, ranks) under both pairings;
  4. the yearly lead-1 values of the four basins (figure pairing, all lead-1 keys), for the legend.
    python -I -B recompute_fig3_fig5.py OUT.json [--smoke]"""
import glob
import json
import os
import sys
import time

import numpy as np
import zarr

OUT = sys.argv[1]
SMOKE = "--smoke" in sys.argv
RUN = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4")
EVD = RUN + "/eval_final_crps_e01_ar"
GD = (os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km")
AU = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/benchmark/aurora")
NETNAME = {0: "Europe (EEA)", 1: "China (CNEMC)", 3: "USA (EPA)", 4: "Korea (AirKorea)", 5: "Japan (NIES)", 6: "India (CPCB)"}
T0 = time.time()

S = np.load(GD + "/stations_2022.npz")
LA = np.asarray(S["lat"], np.float64); LO = np.asarray(S["lon"], np.float64); SRC = np.asarray(S["source"]); OBS = np.asarray(S["daily"], np.float32)
NST = LA.size
OFFICIAL = np.isin(SRC, list(NETNAME))
CLIM = np.load(GD + "/clim/st_clim_2022.npy").astype(np.float32)


class Grid:
    """bilinear reader at the stations for a regular grid: first row at lat0, step dlat southward; first column at lon0, step
    dlon eastward, periodic in longitude; rows clamped to the grid (as the evaluation does)"""

    def __init__(self, nlat, nlon, lat0, dlat, lon0, dlon):
        r = (lat0 - LA) / dlat
        c = np.mod(LO - lon0, 360.0) / dlon
        self.r0 = np.clip(np.floor(r).astype(np.int64), 0, nlat - 2)
        self.c0 = np.floor(c).astype(np.int64) % nlon
        self.c1 = (self.c0 + 1) % nlon
        self.wr = np.clip(r - self.r0, 0.0, 1.0)
        self.wc = c - np.floor(c)

    def __call__(self, f):
        f = np.asarray(f, np.float32)
        top = f[self.r0, self.c0] * (1 - self.wc) + f[self.r0, self.c1] * self.wc
        bot = f[self.r0 + 1, self.c0] * (1 - self.wc) + f[self.r0 + 1, self.c1] * self.wc
        return (top * (1 - self.wr) + bot * self.wr).astype(np.float32)


G_CAMS = Grid(451, 900, 90.0, 0.4, 0.0, 0.4)
G25 = Grid(576, 1440, 84.0 - 0.125, 0.25, -180.0 + 0.125, 0.25)
G50 = Grid(288, 720, 84.0 - 0.25, 0.5, -180.0 + 0.25, 0.5)


def valid_mean_at(g, m):
    """bilinear over the finite neighbours only (weights renormalised); NaN where none is finite"""
    ok = np.isfinite(m).astype(np.float32)
    with np.errstate(all="ignore"):
        return g(np.where(ok > 0, m, 0.0)) / g(ok)


# ---- run parts: index of every (start, lead), GHAP at the target day from the first part that has it
PARTS = sorted(p for p in glob.glob(EVD + "/part_*.npz") if ".tmp" not in p)
IDX, GHT = {}, {}
for p in PARTS:
    z = np.load(p)
    assert z["st_ens"].shape[1] == NST, p
    for n, (d, h, t) in enumerate(z["keys"].tolist()):
        assert (d, h) not in IDX
        IDX[(d, h)] = (p, n)
        if t not in GHT:
            GHT[t] = np.asarray(z["st_ghap"][n], np.float32)
print(len(IDX), "keys in", len(PARTS), "parts;", round(time.time() - T0), "s", flush=True)

# ---- Aurora: day -> (file, row), rows read on demand (one cache per product)
AIDX = {"global": {}, "12h": {}}
for tag, sub in (("global", "out_global"), ("12h", "out_12h_all")):
    for p in sorted(glob.glob(f"{AU}/{sub}/part_*.npz")):
        z = np.load(p)
        for r, d in enumerate(z["days"].tolist()):
            AIDX[tag][int(d)] = (p, r)
_APRED, _AGRID = {}, {}


def aur(tag, d):
    p, r = AIDX[tag][d]
    if p not in _APRED:
        z = np.load(p)
        _APRED[p] = z["pred"]
        la_, lo_ = np.asarray(z["lat"], np.float64), np.asarray(z["lon"], np.float64)
        _AGRID[p] = Grid(la_.size, lo_.size, float(la_[0]), float(la_[0] - la_[1]), float(lo_[0]), float(lo_[1] - lo_[0]))
    return _APRED[p][r], _AGRID[p]


def an00(d):
    z = np.load(f"{AU}/raw/{d:03d}.npz")
    la_, lo_ = np.asarray(z["lat"], np.float64), np.asarray(z["lon"], np.float64)
    g = Grid(la_.size, lo_.size, float(la_[0]), float(la_[0] - la_[1]), float(lo_[0]), float(lo_[1] - lo_[0]))
    return g(np.asarray(z["surf"][0, 5], np.float32) * 1e9)


CZ = zarr.open((os.environ["FOEHN_WORK"] + "/eccv/data/world/cams_forecast_daily/2022.zarr"), mode="r")
_CC = {}


def cams_run(d):
    if d not in _CC:
        if len(_CC) > 3:
            _CC.pop(min(_CC))
        _CC[d] = np.asarray(CZ[d, 0], np.float32)          # (5 lead days, 451, 900); lead 0 = the day of the run
    return _CC[d]


# ---- the keys scored by compare_fair
KEYS = [k for k in IDX if k[1] <= 4 and k[0] + k[1] < 365 and k[0] + 1 <= 364 and k[0] in AIDX["global"]
        and (k[0] + 1) in AIDX["12h"] and k[0] in GHT]
if SMOKE:
    KEYS = [k for k in KEYS if k[0] == 38]
STARTS = sorted({d for d, h in KEYS})
SI = {d: i for i, d in enumerate(STARTS)}
NSD = len(STARTS)
GROUPS = {"six networks": OFFICIAL, "China": SRC == 1, "outside China": OFFICIAL & (SRC != 1)}
GROUPS.update({NETNAME[k]: SRC == k for k in NETNAME})
SYS = ("Foehn", "CAMS", "Aurora")
PERDAY = {g: np.zeros((3, NSD, 4, 2)) for g in GROUPS}            # system, start, lead, (SSE, n)
L1 = {k: np.full((NSD, NST), np.nan, np.float32) for k in ("F", "C", "A", "Y", "GT")}
M1 = {"paper": np.zeros((NSD, NST), bool), "figure": np.zeros((NSD, NST), bool)}

# order of work: by part, so that each part's map25 is read once
by_part = {}
for k in KEYS:
    by_part.setdefault(IDX[k][0], []).append(k)
done = 0
for p, ks in sorted(by_part.items()):
    z = np.load(p)
    ens = z["st_ens"]; gh = z["st_ghap"]; m25 = z["map25"]
    for (d, h) in sorted(ks):
        n = IDX[(d, h)][1]; t = d + h
        y = OBS[:, t]
        fo = np.asarray(ens[n], np.float32).mean(-1)
        m = np.asarray(m25[n], np.float32)
        f25 = valid_mean_at(G25, m)
        with np.errstate(all="ignore"):
            f50 = valid_mean_at(G50, np.nanmean(m.reshape(288, 2, 720, 2), axis=(1, 3)))
        c_d = G_CAMS(cams_run(d)[h]); c_d1 = G_CAMS(cams_run(d + 1)[h - 1])
        ag, gg = aur("global", d); a_d = gg(np.asarray(ag[h - 1], np.float32))
        ae, ge = aur("12h", d + 1)
        j = 2 * (h - 1)
        first = an00(d + 1) if h == 1 else ge(np.asarray(ae[j - 1], np.float32))
        a_e = 0.25 * first + 0.5 * ge(np.asarray(ae[j], np.float32)) + 0.25 * ge(np.asarray(ae[j + 1], np.float32))
        allsys = np.vstack([fo, f25, f50, c_d, c_d1, a_d, a_e, OBS[:, d], GHT[d], CLIM[t]])
        mask = OFFICIAL & np.isfinite(y) & np.isfinite(allsys).all(0)
        i = SI[d]
        for g, sel in GROUPS.items():
            mm = mask & sel
            for s, x in enumerate((fo, c_d1, a_e)):
                e = x[mm] - y[mm]
                PERDAY[g][s, i, h - 1] = [float(np.dot(e.astype(np.float64), e.astype(np.float64))), int(mm.sum())]
        if h == 1:
            gt = np.asarray(gh[n], np.float32)
            L1["F"][i], L1["C"][i], L1["A"][i], L1["Y"][i], L1["GT"][i] = fo, c_d1, a_e, y, gt
            M1["paper"][i] = mask
            M1["figure"][i] = OFFICIAL & np.isfinite(y) & np.isfinite(fo) & np.isfinite(c_d1) & np.isfinite(gt)
        done += 1
    print(done, "/", len(KEYS), round(time.time() - T0), "s", flush=True)
    del z, ens, gh, m25

R = {"inputs": {"eval": EVD, "parts": len(PARTS), "keys_scored": len(KEYS), "start_days": NSD}}
# ---- 1. RMSE by lead, bootstrap of the ratio Foehn / reference
rng = np.random.default_rng(12345)
B, NB = 7, 10000
if not SMOKE:
    starts_b = rng.integers(0, NSD - B + 1, size=(NB, -(-NSD // B)))
    RS = (starts_b[:, :, None] + np.arange(B)).reshape(NB, -1)[:, :NSD]
R["rmse"] = {}
for g, A in PERDAY.items():
    tot = A.sum(1)                                                    # system, lead, (SSE, n)
    assert np.all(tot[0, :, 1] == tot[1, :, 1]) and np.all(tot[0, :, 1] == tot[2, :, 1])
    rec = {"station_days": [int(v) for v in tot[0, :, 1]]}
    for s, nm in enumerate(SYS):
        rec[nm] = [round(float(np.sqrt(tot[s, h, 0] / tot[s, h, 1])), 4) for h in range(4)]
    for s, nm in ((1, "CAMS"), (2, "Aurora")):
        gains = []
        for h in range(4):
            if SMOKE:
                gains.append(None); continue
            f = A[0, :, h]; r = A[s, :, h]
            bs = np.sqrt(f[RS, 0].sum(1) / f[RS, 1].sum(1)) / np.sqrt(r[RS, 0].sum(1) / r[RS, 1].sum(1))
            pt = np.sqrt(f[:, 0].sum() / f[:, 1].sum()) / np.sqrt(r[:, 0].sum() / r[:, 1].sum())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            gains.append([round(100 * (1 - pt), 1), round(100 * (1 - hi), 1), round(100 * (1 - lo), 1)])
        rec[f"gain_vs_{nm}_pct_[point,lo,hi]"] = gains
    R["rmse"][g] = rec

# ---- 2-4. the four basins of Fig. 3b
CENTRES = {"Delhi": (28.7, 77.6, 700), "Po": (45.35, 10.0, 600), "Los Angeles": (34.05, -117.9, 300), "Seoul": (37.56, 127.0, 300)}
BOX = {}
for nm, (la, lo, wkm) in CENTRES.items():
    dla = wkm / 1.5 / 111.2 / 2; dlo = wkm / (111.2 * np.cos(np.deg2rad(la))) / 2
    BOX[nm] = (LA <= la + dla) & (LA >= la - dla) & (LO >= lo - dlo) & (LO <= lo + dlo)
date = lambda d: str(np.datetime64("2022-01-01") + np.timedelta64(int(d), "D"))


def rm(x, y, m):
    return float(np.sqrt(np.mean((x[m] - y[m]) ** 2))) if m.any() else float("nan")


if 38 in SI:
    i = SI[38]
    R["day_9feb"] = {}
    for mk in ("figure", "paper"):
        R["day_9feb"][mk] = {nm: {"Foehn": round(rm(L1["F"][i], L1["Y"][i], M1[mk][i] & b), 2),
                                  "CAMS": round(rm(L1["C"][i], L1["Y"][i], M1[mk][i] & b), 2),
                                  "Aurora": round(rm(L1["A"][i], L1["Y"][i], M1[mk][i] & b & np.isfinite(L1["A"][i])), 2),
                                  "GHAP": round(rm(L1["GT"][i], L1["Y"][i], M1[mk][i] & b), 2),
                                  "stations": int((M1[mk][i] & b).sum())} for nm, b in BOX.items()}
    # stations in the paper mask's complement on that day (why the two pairings differ)
    diff = (M1["figure"][i] != M1["paper"][i])
    R["day_9feb"]["stations_figure_not_paper"] = {nm: int((diff & b).sum()) for nm, b in BOX.items()}
    why = {}
    for nm, b in BOX.items():
        k = np.flatnonzero(diff & b)
        why[nm] = {"obs_previous_day_missing": int((~np.isfinite(OBS[k, 38])).sum()), "n": int(k.size)}
    R["day_9feb"]["why"] = why

if not SMOKE:
    R["tally"] = {}
    for mk in ("paper", "figure"):
        for minst in (1, 3):
            F_ = np.full((4, NSD), np.nan); C_ = F_.copy(); A_ = F_.copy()
            for bi, (nm, b) in enumerate(BOX.items()):
                for i in range(NSD):
                    m = M1[mk][i] & b
                    if m.sum() >= minst:
                        F_[bi, i] = rm(L1["F"][i], L1["Y"][i], m); C_[bi, i] = rm(L1["C"][i], L1["Y"][i], m)
                        A_[bi, i] = rm(L1["A"][i], L1["Y"][i], m) if mk == "paper" else np.nan
            valid = np.isfinite(F_).all(0)
            below = (F_ < C_) & np.isfinite(F_)
            all4 = below.all(0) & valid
            ratio = np.where(valid, np.min(C_ / F_, 0), np.nan); marg = np.where(valid, np.min(C_ - F_, 0), np.nan)
            o1 = np.argsort(-np.nan_to_num(ratio, nan=-1e9))[:3]; o2 = np.argsort(-np.nan_to_num(marg, nan=-1e9))[:3]
            rec = {"days_all_four_valid": int(valid.sum()), "below_CAMS_all_four": int(all4.sum()),
                   "per_basin_below_CAMS": {nm: int(below[bi].sum()) for bi, nm in enumerate(BOX)},
                   "per_basin_above_CAMS": {nm: int(((F_[bi] > C_[bi]) & np.isfinite(F_[bi])).sum()) for bi, nm in enumerate(BOX)},
                   "per_basin_days_valid": {nm: int(np.isfinite(F_[bi]).sum()) for bi, nm in enumerate(BOX)},
                   "top_min_ratio": [(date(STARTS[k] + 1), round(float(ratio[k]), 3)) for k in o1],
                   "top_min_margin": [(date(STARTS[k] + 1), round(float(marg[k]), 2)) for k in o2]}
            if mk == "paper":
                rec["below_CAMS_and_Aurora_all_four"] = int((((F_ < C_) & (F_ < A_)).all(0) & valid).sum())
            R["tally"][f"{mk}_min{minst}stations"] = rec
    # yearly lead-1 values (figure pairing over every lead-1 key of the run, as the overview cache)
    ks1 = sorted(k for k in IDX if k[1] == 1 and k[0] + 1 < 365)
    SSE = np.zeros((3, 4)); NN = np.zeros(4); NTOT = 0
    by_p = {}
    for k in ks1:
        by_p.setdefault(IDX[k][0], []).append(k)
    for p, ks in sorted(by_p.items()):
        z = np.load(p); ens = z["st_ens"]; gh = z["st_ghap"]
        for (d, h) in ks:
            n = IDX[(d, h)][1]; t = d + 1
            fo = np.asarray(ens[n], np.float32).mean(-1); gt = np.asarray(gh[n], np.float32); c1 = G_CAMS(cams_run(d + 1)[0]); y = OBS[:, t]
            m = OFFICIAL & np.isfinite(y) & np.isfinite(fo) & np.isfinite(c1) & np.isfinite(gt)
            NTOT += int(m.sum())
            for bi, b in enumerate(BOX.values()):
                mb = m & b
                for s, x in enumerate((fo, c1, gt)):
                    SSE[s, bi] += float(((x[mb] - y[mb]).astype(np.float64) ** 2).sum())
                NN[bi] += mb.sum()
    R["yearly_lead1_figure_pairing"] = {"lead1_keys": len(ks1), "station_days_all_stations": NTOT}
    for bi, nm in enumerate(BOX):
        R["yearly_lead1_figure_pairing"][nm] = {s: round(float(np.sqrt(SSE[k, bi] / NN[bi])), 2) for k, s in enumerate(("Foehn", "CAMS", "GHAP"))}
        R["yearly_lead1_figure_pairing"][nm]["station_days"] = int(NN[bi])
R["seconds"] = round(time.time() - T0)
txt = json.dumps(R, indent=1) + "\n"
tmp = OUT + ".tmp"
with open(tmp, "w") as f:
    f.write(txt); f.flush(); os.fsync(f.fileno())
assert open(tmp).read() == txt
os.replace(tmp, OUT)
print("saved", OUT, R["seconds"], "s", flush=True)
