#!/usr/bin/env python3
"""Who is warned, real forecast against real forecast (2022). Extends figures/figdata_hidden_warned.py (not modified) to the
systems a comparison needs, on the same pixels, days and definitions:

  truth     the 1 km estimates (GHAP) of the target day at the 8.1 M populated 1 km pixels (>= 100 residents, 94 % of the
            world's residents; eval_pop_crps_e01/pop_ghap_*.npy) and their mean over 40 x 40-pixel cells (cell40_*.npy[:, 1]).
            exposed: pixel above the threshold. hidden: exposed while the 40 km cell mean is not above. hidden with a margin:
            pixel more than 10 % above the threshold and cell mean more than 10 % below it.
  warnings  a resident is warned when the forecast read at their pixel is above the threshold:
            foehn_p50     at least 2 of Foehn's 4 members (ensemble probability >= 0.5; the definition of the project)
            foehn_p25 / p75 / p100   at least 1 / 3 / 4 of the 4 members
            foehn_mean    Foehn's ensemble mean at the 1 km pixel
            foehn_40km    Foehn's ensemble mean averaged over the 40 km cell (cell40_*.npy[:, 0]: the same blocks as the truth)
            foehn_25km    Foehn's ensemble mean averaged over the 0.25 deg cell (map25 of the same forecast)
            cams          the CAMS forecast issued at the same time (run of 00 UTC of day d+1, lead day h-1; equal issuance as in
                          figures/compare_fair.py), interpolated bilinearly to the pixel as at the stations
            cams_cell     the same, value of the nearest 0.4 deg grid point (no interpolation)
            aurora        Aurora initialised at 00 UTC of day d+1 (12-hour model, benchmark/aurora/out_12h_all; daily mean
                          0.25 x 00 UTC + 0.5 x 12 UTC + 0.25 x 00 UTC as in compare_fair.py), bilinear
            aurora_cell   the same, nearest grid point
            perfect_40km  the 40 km cell mean of the estimates themselves (what a perfect 40 km forecast gives)
  table 1   per forecast of the --pop evaluation (start day d every second day, lead h = 1 and 3, target d+h), region,
            threshold and system: residents by (pixel above, cell above, hidden with margin, warned), a 16-entry table
            (code = above + 2 * cell above + 4 * hidden with margin + 8 * warned). Pixel-days are valid when the estimate,
            its 40 km mean and the four members are finite (the mask of figdata_hidden_warned.py).
  table 2   lead ladder on the target days of the 1-day forecasts: leads 1, 3, 5 and 7 for foehn_25km (the 0.25 deg block
            means of the SAME forecasts, stored for every lead in eval_final_crps_e01) and leads 1, 3, 5 for cams, cams_cell,
            aurora, aurora_cell (the stores end at lead day 4 / 120 h). The members at the residents' pixels exist only for
            leads 1 and 3, so the 1 km forecast cannot be scored at 5 days without a new GPU run.
  regions   the nine boxes of the paper (pixel centre, as figdata_hidden_warned.py) and the country groups of regions.py.

One process, forked workers (the Aurora fields are shared):
    python figdata_hidden_systems.py [workers]      -> data/hidden_systems_pop.npz
Added 2026-11-19 after the run of job 23864988 (the script as run is kept in figdata_hidden_systems.py.as_run_23864988; without
EVP_EXTRA nothing changes): EVP_EXTRA=<dir>[:<dir>] adds further --pop evaluations of the same model on the same pixels
(for example eval_pop5_crps_e01 with lead 5, which does not exist yet) -> data/hidden_systems_pop_leads.npz. The CAMS
forecast and Aurora exist to lead day 5; beyond, their warnings are empty."""
import glob
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from regions import BOXES, GROUP_NAMES, pixel_groups  # noqa: E402

FIGS = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
           "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
EVP, EVF = RUN / f"eval_pop_crps_e01{SFX}", RUN / f"eval_final_crps_e01{SFX}"
EXTRA = [RUN / q if "/" not in q else Path(q) for q in os.environ.get("EVP_EXTRA", "").split(":") if q]
AURD = FIGS.parent / "benchmark" / "aurora"
CAMS = (os.environ["FOEHN_WORK"] + "/eccv/data/world/cams_forecast_daily/2022.zarr")
GW = 36000
THR = (15.0, 25.0, 37.5, 50.0, 75.0)
MARGIN = 0.10
BASE1 = ("foehn_p50", "foehn_mean", "foehn_p25", "foehn_p75", "foehn_p100", "foehn_40km", "foehn_25km", "cams", "cams_cell",
         "aurora", "aurora_cell", "perfect_40km")
# decision threshold moved for the single-valued forecasts: "<system>@<mu>" warns when the forecast is above mu x threshold
# (the exposure threshold itself is not moved): the curve reach / correct warnings of each system, for the comparison at an
# equal number of people warned. Joint tables of Foehn (2 of 4 members) and the CAMS forecast (bilinear): warned by one only, by both.
MULT = (0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0)
SWEEP = ("foehn_mean", "cams", "aurora")
SYS1 = BASE1 + tuple(f"{s}@{mu:g}" for s in SWEEP for mu in MULT) + ("foehn_p50_not_cams", "cams_not_foehn_p50", "foehn_p50_and_cams")
SYS2 = ("foehn_25km", "cams", "cams_cell", "aurora", "aurora_cell")
LEADS2 = (1, 3, 5, 7)
REG = list(BOXES) + [f"G:{n}" for n in GROUP_NAMES]                          # boxes, then the country groups

S = {}                                                                         # shared state (built before the fork)


def part_no(p):
    return int(p.split("_")[-1].split(".")[0])


def setup():
    idx = np.load(EVP / "pop_idx.npy")
    pz = np.load(G / "pop_pixels.npz"); assert np.array_equal(pz["idx"], idx), "pixel sets differ"
    pop = pz["pop"].astype(np.float64)
    r, c = idx // GW, idx % GW
    lat, lon = 84.0 - 0.01 * (r + 0.5), -180.0 + 0.01 * (c + 0.5)
    S.update(pop=pop, cell=(r // 40) * (GW // 40) + c // 40, r25=(r // 25).astype(np.int32), c25=(c // 25).astype(np.int32))
    # bilinear and nearest reading of a 0.4 deg field (rows from 90N, columns from 0E), as compare_fair.bil
    fi = (90.0 - lat) / 0.4; fj = (lon % 360.0) / 0.4
    i0 = np.floor(fi).astype(np.int32); j0 = np.floor(fj).astype(np.int32) % 900
    S.update(I0=i0, J0=j0, J1=(j0 + 1) % 900, WI=(fi - i0).astype(np.float32), WJ=(fj - np.floor(fj)).astype(np.float32),
             IN=np.rint(fi).astype(np.int32), JN=np.rint(fj).astype(np.int32) % 900)
    # regions: one label per combination of (boxes, country group), and the matrix combination -> region
    bits = np.zeros(len(idx), np.int64)
    for j, b in enumerate(list(BOXES.values())[1:]):
        bits |= ((lat <= b[0]) & (lat > b[1]) & (lon >= b[2]) & (lon < b[3])).astype(np.int64) << j
    grp = pixel_groups(r, c)
    combo, lab = np.unique(bits + 256 * grp, return_inverse=True)
    M = np.zeros((len(REG), len(combo)))
    M[0] = 1.0
    for j in range(len(BOXES) - 1):
        M[1 + j] = (combo % 256 >> j) & 1
    for g in range(len(GROUP_NAMES)):
        M[len(BOXES) + g] = combo // 256 == g
    S.update(LAB16=(lab * 16).astype(np.int64), NC=len(combo), M=M)
    # index of the forecasts
    S["pop_index"], S["fin_index"] = {}, {}
    for ev in [EVP] + EXTRA:
        assert np.array_equal(np.load(ev / "pop_idx.npy"), idx), f"the pixel set of {ev} differs"
        for p in sorted(glob.glob(str(ev / "part_*.npz")), key=part_no):
            for n, (d, h, t) in enumerate(np.load(p)["keys"]):
                S["pop_index"][(int(d), int(h))] = (ev, part_no(p), n, int(t))
    for p in sorted(glob.glob(str(EVF / "part_*.npz")), key=part_no):
        for n, (d, h, t) in enumerate(np.load(p)["keys"]):
            S["fin_index"][(int(d), int(h))] = (p, n)
    # Aurora, 12-hour model initialised at 00 UTC of each day: (10 leads, 450, 900) per initialisation day
    S["aur"] = {}
    for p in sorted(glob.glob(str(AURD / "out_12h_all" / "part_*.npz"))):
        z = np.load(p); pred = z["pred"]
        assert float(z["lat"][0]) == 90.0 and abs(float(z["lat"][0] - z["lat"][1]) - 0.4) < 1e-4 and float(z["lon"][0]) == 0.0
        for k, d in enumerate(z["days"]):
            S["aur"][int(d)] = pred[k]
    print(f"setup: {len(idx)} pixels, {pop.sum() / 1e6:.1f} M residents, {len(combo)} region combinations, "
          f"{len(S['pop_index'])} forecasts at the pixels, {len(S['fin_index'])} in the full evaluation, "
          f"{len(S['aur'])} Aurora initialisations", flush=True)


def bil(f):
    I0, J0, J1, WI, WJ = S["I0"], S["J0"], S["J1"], S["WI"], S["WJ"]
    return ((1 - WI) * ((1 - WJ) * f[I0, J0] + WJ * f[I0, J1]) + WI * ((1 - WJ) * f[I0 + 1, J0] + WJ * f[I0 + 1, J1])).astype(np.float32)


def near(f):
    return f[S["IN"], S["JN"]].astype(np.float32)


def fields04(d, h):
    """daily means on the 0.4 deg grid of the CAMS forecast and of Aurora issued at 00 UTC of day d+1 for the day d+h"""
    if h > 5:                                                                 # no reference forecast beyond lead day 5
        nan = np.full((451, 900), np.nan, np.float32)
        return nan, nan[:450]
    cams = np.asarray(zarr.open(CAMS, mode="r")[d + 1, 0, h - 1], np.float32)
    A = S["aur"][d + 1]; k0 = 2 * (h - 1)                                    # leads 12, 24, ... h after 00 UTC of d+1
    if h == 1:                                                                # the first 00 UTC is the analysis
        z = np.load(AURD / "raw" / f"{d + 1:03d}.npz")
        first = (z["surf"][0, 5] * 1e9).astype(np.float32)[:450]
    else:
        first = A[k0 - 1].astype(np.float32)
    aur = 0.25 * first + 0.5 * A[k0].astype(np.float32) + 0.25 * A[k0 + 1].astype(np.float32)
    return cams, aur


def tables(w, g1, g40, warn_of):
    """warn_of: list of functions thr -> bool array (warned). -> (regions, thresholds, systems, 16)"""
    out = np.zeros((len(REG), len(THR), len(warn_of), 16))
    for k, thr in enumerate(THR):
        E, C = g1 > thr, g40 > thr
        Hm = (g1 > thr * (1 + MARGIN)) & (g40 <= thr * (1 - MARGIN))
        base = S["LAB16"] + E + 2 * C + 4 * Hm
        for s, fn in enumerate(warn_of):
            t = np.bincount(base + 8 * fn(thr, C), weights=w, minlength=S["NC"] * 16).reshape(S["NC"], 16)
            out[:, k, s] = S["M"] @ t
    return out


def one(key):
    d, h = key
    ev, k, n, t = S["pop_index"][key]
    t0 = time.time()
    m = np.asarray(np.load(ev / f"pop_{k}.npy", mmap_mode="r")[n], np.float32)           # (pixels, 4 members)
    g1 = np.asarray(np.load(ev / f"pop_ghap_{k}.npy", mmap_mode="r")[n], np.float32)
    c40 = np.asarray(np.load(ev / f"cell40_{k}.npy", mmap_mode="r")[n], np.float32).reshape(2, -1)
    f40, g40 = c40[0][S["cell"]], c40[1][S["cell"]]
    ok = np.isfinite(g1) & np.isfinite(g40) & np.isfinite(m).all(1)                     # as figdata_hidden_warned.py
    w = np.where(ok, S["pop"], 0.0)
    f25 = np.load(ev / f"part_{k}.npz")["map25"][n].astype(np.float32)[S["r25"], S["c25"]]
    cams, aur = fields04(d, h)
    cb, cn, ab, an = bil(cams), near(cams), bil(aur), near(aur)
    mm = m.mean(1)
    cnt = {}

    def nab(thr):
        if thr not in cnt:
            cnt.clear(); cnt[thr] = (m > thr).sum(1)
        return cnt[thr]
    fns = [lambda thr, C: nab(thr) >= 2, lambda thr, C: mm > thr, lambda thr, C: nab(thr) >= 1, lambda thr, C: nab(thr) >= 3,
           lambda thr, C: nab(thr) >= 4, lambda thr, C: f40 > thr, lambda thr, C: f25 > thr, lambda thr, C: cb > thr,
           lambda thr, C: cn > thr, lambda thr, C: ab > thr, lambda thr, C: an > thr, lambda thr, C: C]
    src = {"foehn_mean": mm, "cams": cb, "aurora": ab}
    for sname in SWEEP:
        for mu in MULT:
            fns.append(lambda thr, C, x=src[sname], mu=mu: x > np.float32(mu * thr))
    fns += [lambda thr, C: (nab(thr) >= 2) & ~(cb > thr), lambda thr, C: ~(nab(thr) >= 2) & (cb > thr),
            lambda thr, C: (nab(thr) >= 2) & (cb > thr)]
    assert len(fns) == len(SYS1)
    T1 = tables(w, g1, g40, fns)
    diag = [w.sum(), w[~np.isfinite(f40)].sum(), w[~np.isfinite(f25)].sum(), w[~np.isfinite(cb)].sum(), w[~np.isfinite(ab)].sum()]
    T2 = None
    if h == 1:                                                                   # lead ladder on this target day
        T2 = np.full((len(REG), len(THR), len(LEADS2), len(SYS2), 16), np.nan)
        for li, L in enumerate(LEADS2):
            s0 = t - L
            if (s0, L) not in S["fin_index"]:
                continue
            p, nn = S["fin_index"][(s0, L)]
            x25 = np.load(p)["map25"][nn].astype(np.float32)[S["r25"], S["c25"]]
            fn2 = [lambda thr, C, x=x25: x > thr]
            if L <= 5:
                ca, au = fields04(s0, L)
                fn2 += [lambda thr, C, x=bil(ca): x > thr, lambda thr, C, x=near(ca): x > thr,
                        lambda thr, C, x=bil(au): x > thr, lambda thr, C, x=near(au): x > thr]
            T2[:, :, li, :len(fn2)] = tables(w, g1, g40, fn2)
    print(f"start {d} lead {h} target {t}: {w.sum() / 1e6:.0f} M valid residents, {time.time() - t0:.0f} s", flush=True)
    return key, t, T1, T2, diag


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    only = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else None   # smoke test: start days
    setup()
    keys = sorted(S["pop_index"], key=lambda k: (k[1], k[0]))
    if only:
        keys = [k for k in keys if k[0] in only]
    with get_context("fork").Pool(workers) as pool:
        res = pool.map(one, keys, chunksize=1)
    res.sort(key=lambda r: (r[0][1], r[0][0]))
    K = np.array([(k[0], k[1], t) for k, t, *_ in res])
    T1 = np.stack([r[2] for r in res])
    t2 = [(r[1], r[3]) for r in res if r[3] is not None]
    out = HERE / f"data{SFX}" / ("hidden_systems_pop_test2.npz" if only else "hidden_systems_pop_leads.npz" if EXTRA else "hidden_systems_pop.npz")
    if out.name == "hidden_systems_pop.npz" and out.exists():
        out = out.with_name("hidden_systems_pop_rerun.npz")                    # the file of job 23864988 is never overwritten
    np.savez(out, keys=K, T1=T1, targets2=np.array([t for t, _ in t2]), T2=np.stack([x for _, x in t2]),
             diag=np.array([r[4] for r in res]), regions=np.array(REG), thr=np.array(THR), sys1=np.array(SYS1),
             sys2=np.array(SYS2), leads2=np.array(LEADS2), margin=MARGIN, mult=np.array(MULT), sweep=np.array(SWEEP),
             code=np.array("above + 2 * cell above + 4 * hidden with margin + 8 * warned"))
    print("saved", out, T1.shape)


if __name__ == "__main__":
    main()
