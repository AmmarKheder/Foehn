#!/usr/bin/env python3
"""Who is warned, version 2: two more reference forecasts that see the 1 km estimates as Foehn does (2022).
Written from a copy of ../figdata_hidden_systems.py (not modified): the same pixels, forecasts, valid residents, regions
and 16-entry code table, so that the tables of this file can be put beside ../data/hidden_systems_pop.npz system by system.

New systems (no monitoring station enters anywhere; nothing of 2022 is fitted):
  cams_qm     the CAMS forecast issued at the same time (the same field as 'cams' in ../figdata_hidden_systems.py: run of
              00 UTC of day d+1, lead day h-1), bias-corrected towards the 1 km estimates by the quantile map of the
              downscaling project (GH/ghap_clim/bcsd_G.npz, runs_ghap/fit_bcsd_ghap.py): per 0.25 deg cell (257,281 cells,
              nearest 0.25 deg point of each 1 km pixel) and calendar month, 21 quantiles (levels 0, 0.05, ..., 1) of
                x = log1p of the day-0 daily mean PM2.5 of the CAMS operational 00 UTC run (x 8/7 before 2017-09-27),
                    regridded bilinearly in log1p from 0.4 to 0.25 deg (scripts/build_global_coarse.py), at the cell's point
                y = the mean over the cell's 1 km pixels of log1p(max(round(GHAP), 1)) of the same day
              on the 1,412 training days of 2017-01-02 .. 2020-12-31 (78 to 124 days per calendar month); linear between the
              quantiles, slope-1 offsets beyond the end quantiles. Applied here exactly as fitted: the forecast field is
              regridded with the same code (float16 as in the store), mapped with the month of the target day, and every
              resident of the cell receives expm1(QM_c,m(x_c)).
  cams_bcsd   the same value distributed inside the cell by the climatological 1 km pattern of the estimates (bias
              correction and spatial disaggregation, 'bcsd' of realtime_1km/rc/coupling_live.py):
                expm1(mu_p(t) - mubar_c(t) + QM_c,m(x_c)),
              mu_p(t) = harmonic climatology (annual and semi-annual) of log1p(max(round(GHAP), 1)) at the pixel, fitted on
              2017-2020 (GH/ghap_clim/clim_mean_G.npy), mubar_c(t) = its mean over the cell (bcsd_G.npz clim_cell_coef).
  clim_1km    expm1(mu_p(t)): the 1 km climatology alone (diagnostic: what the pattern gives without any forecast).
  cams_qm@mu, cams_bcsd@mu   decision threshold moved (warns above mu x threshold), as the sweep of the original script.
Recomputed for the identity check against ../data/hidden_systems_pop.npz (must be equal): foehn_p50, foehn_mean,
foehn_40km, cams.
Pixels outside the mask of the downscaling project (no climatology there) take the cell value in cams_bcsd; pixels whose
cell has no map take no value and are never warned by the new systems (counted in diag).

    python figdata_hidden_bc.py [workers] [start days of a test]   -> data/hidden_bc_pop.npz (or data/hidden_bc_pop_test.npz)"""
import datetime as dt
import glob
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from regions import BOXES, GROUP_NAMES, pixel_groups  # noqa: E402  (../regions.py, unchanged)

G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
RUN = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/runs/"
           "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4")
EVP = RUN / "eval_pop_crps_e01"
CAMS = "/scratch/project_462001804/ammar/eccv/data/world/cams_forecast_daily/2022.zarr"
GH = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/data/processed/ghap")
GW = 36000
THR = (15.0, 25.0, 37.5, 50.0, 75.0)
MARGIN = 0.10
MULT = (0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0)
NEW = ("cams_qm", "cams_bcsd", "clim_1km")
CHECK = ("foehn_p50", "foehn_mean", "foehn_40km", "cams")
SWEEP = ("cams_qm", "cams_bcsd")
SYS1 = NEW + CHECK + tuple(f"{s}@{mu:g}" for s in SWEEP for mu in MULT)
DIAG2 = ("foehn_mean", "cams", "cams_qm", "cams_bcsd", "clim_1km")          # population-weighted error sums against the estimates
REG = list(BOXES) + [f"G:{n}" for n in GROUP_NAMES]
D2022 = dt.date(2022, 1, 1)

S = {}


def part_no(p):
    return int(p.split("_")[-1].split(".")[0])


# ------------------------------------------------------------------ verbatim from scripts/build_global_coarse.py (the fit's input)
H25, W25 = 576, 1440


def regrid_weights():
    lat = 84.0 - 0.25 * np.arange(H25)
    lon = -180.0 + 0.25 * np.arange(W25)
    fi = (90.0 - lat) / 0.4
    i0 = np.floor(fi).astype(np.int64)
    wi = fi - i0
    fj = (lon % 360.0) / 0.4
    jf = np.floor(fj)
    j0 = jf.astype(np.int64) % 900
    return i0, wi, j0, (j0 + 1) % 900, fj - jf


def regrid(src, wts):
    i0, wi, j0, j1, wj = wts
    L = np.log1p(np.maximum(src, 0.0).astype(np.float64))
    top = L[i0][:, j0] * (1 - wj) + L[i0][:, j1] * wj
    bot = L[i0 + 1][:, j0] * (1 - wj) + L[i0 + 1][:, j1] * wj
    return np.expm1(top * (1 - wi)[:, None] + bot * wi[:, None])


# ------------------------------------------------------------------ verbatim from runs_ghap/fit_bcsd_ghap.py (reference of the check)
def qmap_ref(x, qx, qy):
    """quantile map of one cell-month: linear inside [qx0, qx20], slope-1 offsets outside"""
    y = np.interp(x, qx, qy)
    y = np.where(x < qx[0], qy[0] + (x - qx[0]), y)
    return np.where(x > qx[-1], qy[-1] + (x - qx[-1]), y)


def harmonic_basis(doy):
    ang = 2 * np.pi * float(doy) / 365.0
    return np.array([1.0, np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)])


def qmap_vec(x, qx, qy):
    """the same map for all cells at once: x (n,), qx, qy (n, 21)"""
    j = np.clip((x[:, None] >= qx).sum(1) - 1, 0, qx.shape[1] - 2)
    a = np.arange(x.size)
    x0, x1, y0, y1 = qx[a, j], qx[a, j + 1], qy[a, j], qy[a, j + 1]
    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    y = np.where(x < qx[:, 0], qy[:, 0] + (x - qx[:, 0]), y)
    return np.where(x > qx[:, -1], qy[:, -1] + (x - qx[:, -1]), y)


def setup():
    t0 = time.time()
    idx = np.load(EVP / "pop_idx.npy")
    pz = np.load(G / "pop_pixels.npz"); assert np.array_equal(pz["idx"], idx), "pixel sets differ"
    pop = pz["pop"].astype(np.float64)
    r, c = idx // GW, idx % GW
    lat, lon = 84.0 - 0.01 * (r + 0.5), -180.0 + 0.01 * (c + 0.5)
    S.update(pop=pop, cell=(r // 40) * (GW // 40) + c // 40)
    fi = (90.0 - lat) / 0.4; fj = (lon % 360.0) / 0.4
    i0 = np.floor(fi).astype(np.int32); j0 = np.floor(fj).astype(np.int32) % 900
    S.update(I0=i0, J0=j0, J1=(j0 + 1) % 900, WI=(fi - i0).astype(np.float32), WJ=(fj - np.floor(fj)).astype(np.float32))
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
    S["pop_index"] = {}
    for p in sorted(glob.glob(str(EVP / "part_*.npz")), key=part_no):
        for n, (d, h, t) in enumerate(np.load(p)["keys"]):
            S["pop_index"][(int(d), int(h))] = (part_no(p), n, int(t))
    # ---------------------------------------------------------------- the map of the downscaling project at the residents' pixels
    mask = np.load(GH / "grid/ghap_mask.npy")                                  # flat ids r * 36000 + c of the lattice of the estimates
    assert mask.ndim == 1 and (np.diff(mask.astype(np.int64)) > 0).all(), "the mask is not sorted"
    k = np.minimum(np.searchsorted(mask, idx), mask.size - 1)
    inmask = mask[k].astype(np.int64) == idx
    cid = ((2 * r + 26) // 50) * W25 + ((2 * c + 26) // 50) % W25              # nearest 0.25 deg point (build_ghap.py R_MAX rule)
    in_grid = (2 * r + 26) // 50 < H25
    cell_id = np.load(GH / "grid/cell_id.npy", mmap_mode="r")
    cid_file = np.asarray(cell_id[k[inmask]]).astype(np.int64)
    assert np.array_equal(cid_file, cid[inmask]), "cell ids of the downscaling project differ from the nearest 0.25 deg point"
    b = np.load(GH / "ghap_clim/bcsd_G.npz")
    cells = b["cells"].astype(np.int64)
    assert [str(v) for v in b["train_first_last"]] == ["2017-01-01", "2020-12-31"] and b["levels"].size == 21
    cm = np.load(GH / "ghap_clim/cellmean_train_G.npz")
    dates = cm["dates"].astype("datetime64[D]")
    assert dates.min() >= np.datetime64("2017-01-01") and dates.max() <= np.datetime64("2020-12-31"), "fit days outside 2017-2020"
    assert np.array_equal(cm["cells"].astype(np.int64), cells)
    cp = np.minimum(np.searchsorted(cells, cid), cells.size - 1)
    incell = (cells[cp] == cid) & in_grid
    ucell, uinv = np.unique(cp[incell], return_inverse=True)                   # positions in 'cells' of the cells with residents
    U = np.full(len(idx), -1, np.int64); U[incell] = uinv
    qx = b["qx"][ucell].astype(np.float64); qy = b["qy"][ucell].astype(np.float64)   # (cells with residents, 12, 21)
    assert (np.diff(qx, axis=2) > 0).all()
    coef = np.zeros((len(idx), 5), np.float32)                                 # pixel climatology (zero where the pixel is outside the mask)
    clim = np.load(GH / "ghap_clim/clim_mean_G.npy", mmap_mode="r")
    assert clim.shape == (mask.size, 5)
    kk, where = k[inmask], np.flatnonzero(inmask)
    assert (np.diff(kk) > 0).all()
    blk = 400_000
    for a0 in range(0, kk.size, blk):                                         # sorted positions: read block by block
        s_ = kk[a0:a0 + blk]
        seg = np.asarray(clim[s_[0]:s_[-1] + 1])
        coef[where[a0:a0 + blk]] = seg[s_ - s_[0]]
    S.update(U=U, INMASK=inmask, QX=qx, QY=qy, CC=b["clim_cell_coef"][ucell].astype(np.float64), CELL_FLAT=cells[ucell], COEF=coef,
             WTS=regrid_weights(), NDATES=int(dates.size), FIT_FIRST=str(dates.min()), FIT_LAST=str(dates.max()),
             NDAYS_MONTH=b["ndays_per_month"].astype(int))
    cov = dict(residents_M=pop.sum() / 1e6, in_mask_M=pop[inmask].sum() / 1e6, with_map_M=pop[incell].sum() / 1e6,
               with_map_and_pattern_M=pop[incell & inmask].sum() / 1e6, pixels=len(idx), pixels_in_mask=int(inmask.sum()),
               pixels_with_map=int(incell.sum()), cells_with_residents=int(ucell.size))
    S["COV"] = cov
    print(f"setup: {len(idx)} pixels, {pop.sum() / 1e6:.1f} M residents, {len(combo)} region combinations, {len(S['pop_index'])} forecasts; "
          f"map: {cov['pixels_with_map']} pixels with a cell map ({cov['with_map_M']:.1f} M residents), {cov['pixels_in_mask']} in the mask "
          f"({cov['in_mask_M']:.1f} M), {ucell.size} cells with residents; fit days {dates.size} from {dates.min()} to {dates.max()}; "
          f"cell ids identical to the nearest 0.25 deg point on all {int(inmask.sum())} mask pixels ({time.time() - t0:.0f} s)", flush=True)


def bil(f):
    I0, J0, J1, WI, WJ = S["I0"], S["J0"], S["J1"], S["WI"], S["WJ"]
    return ((1 - WI) * ((1 - WJ) * f[I0, J0] + WJ * f[I0, J1]) + WI * ((1 - WJ) * f[I0 + 1, J0] + WJ * f[I0 + 1, J1])).astype(np.float32)


def tables(w, g1, g40, warn_of):
    """warn_of: list of functions thr -> bool array (warned). -> (regions, thresholds, systems, 16)   [as the original]"""
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
    k, n, t = S["pop_index"][key]
    t0 = time.time()
    m = np.asarray(np.load(EVP / f"pop_{k}.npy", mmap_mode="r")[n], np.float32)           # (pixels, 4 members)
    g1 = np.asarray(np.load(EVP / f"pop_ghap_{k}.npy", mmap_mode="r")[n], np.float32)
    c40 = np.asarray(np.load(EVP / f"cell40_{k}.npy", mmap_mode="r")[n], np.float32).reshape(2, -1)
    f40, g40 = c40[0][S["cell"]], c40[1][S["cell"]]
    ok = np.isfinite(g1) & np.isfinite(g40) & np.isfinite(m).all(1)                     # as figdata_hidden_warned.py
    w = np.where(ok, S["pop"], 0.0)
    cams = np.asarray(zarr.open(CAMS, mode="r")[d + 1, 0, h - 1], np.float32)            # equal issuance, as the original
    cb = bil(cams)
    mm = m.mean(1)
    # ---------------------------------------------------------------- the bias-corrected CAMS forecast
    day = D2022 + dt.timedelta(t); assert day.year == 2022
    mon = day.month
    x025 = regrid(cams, S["WTS"]).astype(np.float16).astype(np.float64)                  # as the store that the map was fitted on
    xc = np.log1p(np.maximum(x025.reshape(-1)[S["CELL_FLAT"]], 0.0))
    qx, qy = S["QX"][:, mon - 1], S["QY"][:, mon - 1]
    qm = qmap_vec(xc, qx, qy)                                                             # log1p, per cell
    rng = np.random.default_rng(1000 * d + h); pick = rng.choice(xc.size, 2000, replace=False)
    dev = max(abs(float(qmap_ref(xc[j], qx[j], qy[j])) - qm[j]) for j in pick)           # against the function of the fit
    B = harmonic_basis(t)                                                                 # t = day of year, 0-based
    U = S["U"]; has = U >= 0
    qm_p = np.full(U.size, np.nan); qm_p[has] = qm[U[has]]
    mubar = S["CC"] @ B
    ano = np.zeros(U.size)                                                                # mu_p - mubar_c (zero outside the mask: cell value)
    both = has & S["INMASK"]
    mu_p = (S["COEF"] @ B.astype(np.float32)).astype(np.float64)
    ano[both] = mu_p[both] - mubar[U[both]]
    v_qm = np.expm1(qm_p).astype(np.float32)
    v_bc = np.expm1(qm_p + ano).astype(np.float32)
    v_cl = np.where(S["INMASK"], np.expm1(mu_p), np.nan).astype(np.float32)
    cnt = {}

    def nab(thr):
        if thr not in cnt:
            cnt.clear(); cnt[thr] = (m > thr).sum(1)
        return cnt[thr]
    fns = [lambda thr, C: v_qm > thr, lambda thr, C: v_bc > thr, lambda thr, C: v_cl > thr,
           lambda thr, C: nab(thr) >= 2, lambda thr, C: mm > thr, lambda thr, C: f40 > thr, lambda thr, C: cb > thr]
    src = {"cams_qm": v_qm, "cams_bcsd": v_bc}
    for sname in SWEEP:
        for mu in MULT:
            fns.append(lambda thr, C, x=src[sname], mu=mu: x > np.float32(mu * thr))
    assert len(fns) == len(SYS1)
    with np.errstate(invalid="ignore"):
        T1 = tables(w, g1, g40, fns)
    # diagnostics: valid residents; residents without a value; residents whose cell value was extrapolated beyond the fitted range
    lo = np.zeros(U.size, bool); hi = np.zeros(U.size, bool)
    lo[has] = (xc < qx[:, 0])[U[has]]; hi[has] = (xc > qx[:, -1])[U[has]]
    diag = [w.sum(), w[~np.isfinite(v_qm)].sum(), w[~S["INMASK"]].sum(), w[lo].sum(), w[hi].sum(), dev]
    g = g1.astype(np.float64)
    diag2 = []
    for x in (mm, cb, v_qm, v_bc, v_cl):                                                  # the order of DIAG2
        x = x.astype(np.float64); q = np.isfinite(x) & ok; ww = S["pop"][q]; e = x[q] - g[q]
        diag2.append([ww.sum(), (ww * x[q]).sum(), (ww * g[q]).sum(), (ww * e * e).sum(), (ww * np.abs(e)).sum()])
    print(f"start {d} lead {h} target {t} ({day}, month {mon}): {w.sum() / 1e6:.0f} M valid residents, qmap deviation {dev:.1e}, "
          f"extrapolated low/high {w[lo].sum() / 1e6:.0f}/{w[hi].sum() / 1e6:.0f} M, {time.time() - t0:.0f} s", flush=True)
    return key, t, T1, diag, diag2


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
    out = HERE / "data" / ("hidden_bc_pop_test.npz" if only else "hidden_bc_pop.npz")
    np.savez(out, keys=K, T1=T1, diag=np.array([r[3] for r in res]), diag2=np.array([r[4] for r in res]), regions=np.array(REG),
             thr=np.array(THR), sys1=np.array(SYS1), margin=MARGIN, mult=np.array(MULT), sweep=np.array(SWEEP), diag2_systems=np.array(DIAG2),
             diag_cols=np.array(["valid residents", "valid residents without a cell map", "valid residents outside the mask (no pattern)",
                                 "valid residents with x below the fitted range", "valid residents with x above the fitted range",
                                 "max deviation of the vectorised map from fit_bcsd_ghap.qmap on 2000 cells"]),
             diag2_cols=np.array(["residents", "sum pop x forecast", "sum pop x estimate", "sum pop x error^2", "sum pop x |error|"]),
             coverage=np.array([f"{k_}={v}" for k_, v in S["COV"].items()]),
             fit=np.array([f"days={S['NDATES']}", f"first={S['FIT_FIRST']}", f"last={S['FIT_LAST']}",
                           "days_per_month=" + ",".join(str(v) for v in S["NDAYS_MONTH"])]),
             code=np.array("above + 2 * cell above + 4 * hidden with margin + 8 * warned"))
    print("saved", out, T1.shape)


if __name__ == "__main__":
    main()
