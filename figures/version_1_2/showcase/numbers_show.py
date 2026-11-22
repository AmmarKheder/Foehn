#!/usr/bin/env python3
"""Numbers of the Fig. 5 candidate (2026-11-21, N/rebuild/figs_show): read-only, JSON on stdout.
    python -I -B numbers_show.py <target doy> [<target doy> ...]      (23 = 24 January 2022, 24 = 25 January 2022)
Part 'box' (North China Plain box of Fig. 5, N 41.5 S 35.5 W 113.5 E 121.5; land = GHAP > 0 as in the figure): for leads
1, 3, 5, 7, box-mean (land) of GHAP, of Foehn's 16-member mean, lowest and highest member, CAMS and Aurora issued at the same
time (fields of fig_gc2_realism_show.py: cams[d+1, 0, h-1] and aur_field), the 0.4 deg fields read at each 1 km pixel
(nearest grid point = the cell drawn in the figure; bilinear given as a check); RMSE against GHAP over that land; share of
the land above 75 in GHAP and Foehn's mean P(> 75) there (16 members); rank of member 1 (index 0) among the 16 for its
variance at 10 km (the figure's spectrum function) 1 and 5 days ahead; ranks of members 1-3 by box mean.
Part 'r3' (main-text paragraph values): the procedure of an independent recomputation, r3_prob.py (whole crop 42-33 N,
112-131 E; land = static land mask and GHAP > 0; y = GHAP > 75), for leads 1-7, GHAP and land read from the tile stores
(r3_prob reads them from FIG/data/case_eastasia_<RUN>.npz, which holds only the peak day)."""
import glob
import json
import os
import sys

import numpy as np
import zarr

E = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments")
D = E + "/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/eval_case_eastasia_crps_e01_ar"
G = (os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km")
AURD = E + "/benchmark/aurora"
SUB = (41.5, 35.5, 113.5, 121.5)
TH = 75.0
TILE, NTR, NTC = 512, 29, 71
cams = zarr.open((os.environ["FOEHN_WORK"] + "/eccv/data/world/cams_forecast_daily/2022.zarr"), mode="r")


def tiles_box(path, k, r0, r1, c0, c1):                       # as figdata_case.py
    with open(path, "rb") as fh:
        np.lib.format.read_magic(fh); sh, _, dt = np.lib.format.read_array_header_1_0(fh); hdr = fh.tell()
    tr0, tr1, tc0, tc1 = r0 // TILE, (r1 - 1) // TILE, c0 // TILE, (c1 - 1) // TILE
    nb, B = tc1 - tc0 + 1, TILE * TILE
    blk = np.stack([np.fromfile(path, dtype=dt, count=nb * B, offset=hdr + (((k * NTR + tr) * NTC + tc0) * B) * 2).reshape(nb, TILE, TILE)
                    for tr in range(tr0, tr1 + 1)])
    blk = blk.transpose(0, 2, 1, 3).reshape((tr1 - tr0 + 1) * TILE, nb * TILE)
    return blk[r0 - tr0 * TILE:r1 - tr0 * TILE, c0 - tc0 * TILE:c1 - tc0 * TILE].astype(np.float32)


aur = {}
for p in sorted(glob.glob(AURD + "/out_12h_all/part_*.npz")):
    z = np.load(p)
    for k, d in enumerate(z["days"]):
        aur[int(d)] = (p, k)


def aur_field(d, h):                                           # as fig_gc2_realism (Fig. 5)
    if d + 1 not in aur or h > 5:
        return None
    p, k = aur[d + 1]; A = np.load(p)["pred"][k].astype(np.float32); k0 = 2 * (h - 1)
    first = (np.load(f"{AURD}/raw/{d + 1:03d}.npz")["surf"][0, 5, :A.shape[1]].astype(np.float32) * 1e9) if h == 1 else A[k0 - 1]
    return 0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1]


def cams_field(d, h):
    return np.asarray(cams[d + 1, 0, h - 1], np.float32) if h <= 5 else None


def at_pixels(f, la, lo, bilinear):
    if f is None:
        return None
    if not bilinear:
        i = np.rint((90.0 - la) / 0.4).astype(int); j = np.rint((lo % 360.0) / 0.4).astype(int) % 900
        return f[i, j]
    fi = (90.0 - la) / 0.4; fj = (lo % 360.0) / 0.4; i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int) % 900
    wi = fi - i0; wj = fj - np.floor(fj); j1 = (j0 + 1) % 900
    return (1 - wi) * ((1 - wj) * f[i0, j0] + wj * f[i0, j1]) + wi * ((1 - wj) * f[i0 + 1, j0] + wj * f[i0 + 1, j1])


def spectrum(x, dx_km):                                        # as fig_gc2_realism (Fig. 5)
    x = np.where(np.isfinite(x), x, np.nanmean(x)); x = x - x.mean()
    ny, nx = x.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    P = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2 / (nx * ny) ** 2 / np.mean(win ** 2)
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dx_km)); kx = np.fft.fftshift(np.fft.fftfreq(nx, dx_km))
    K = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
    nb = 24 if dx_km < 5 else 7
    bins = np.logspace(np.log10(1.0 / (min(nx, ny) * dx_km)), np.log10(0.5 / dx_km), nb)
    idx = np.digitize(K.ravel(), bins)
    band = np.array([P.ravel()[idx == i].sum() if np.any(idx == i) else np.nan for i in range(1, len(bins))])
    kc = np.sqrt(bins[1:] * bins[:-1])
    return 1.0 / kc, band / np.diff(np.log10(bins))


r2 = lambda x: None if x is None else round(float(x), 2)
r3 = lambda x: None if x is None else round(float(x), 3)


def one(peak):
    c0 = np.load(f"{D}/crop0_d{peak - 1}_h1.npz"); box = c0["box"]; (R0, R1), (C0, C1) = c0["rows"], c0["cols"]
    n, s, w, e = SUB
    sr0, sr1 = int(round((box[0] - n) * 100)), int(round((box[0] - s) * 100)); sc0, sc1 = int(round((w - box[2]) * 100)), int(round((e - box[2]) * 100))
    ghc = tiles_box(f"{G}/ghap1km_tiles.npy", 1826 + peak, int(R0), int(R1), int(C0), int(C1))   # whole crop
    gh = ghc[sr0:sr1, sc0:sc1]; land = gh > 0
    la = (box[0] - (np.arange(sr0, sr1) + 0.5) * 0.01)[:, None] * np.ones((1, sc1 - sc0))
    lo = (box[2] + (np.arange(sc0, sc1) + 0.5) * 0.01)[None, :] * np.ones((sr1 - sr0, 1))
    la, lo = la[land], lo[land]; g = gh[land]; y = g > TH
    out = {"target_doy": peak, "date": str(np.datetime64("2022-01-01") + np.timedelta64(peak, "D")), "box": {}, "r3": {}}
    out["box"]["n_land_px"] = int(land.sum()); out["box"]["ghap_mean"] = r2(g.mean()); out["box"]["share_gt75"] = r3(y.mean())
    for h in (1, 3, 5, 7):
        m = np.load(f"{D}/crop0_d{peak - h}_h{h}.npz")["members"].astype(np.float32)[:, sr0:sr1, sc0:sc1]
        ml = m[:, land]; mu = ml.mean(0); bm = ml.mean(1)
        row = {"ghap": r2(g.mean()), "foehn_mean": r2(mu.mean()), "member_min": r2(bm.min()), "member_max": r2(bm.max()),
               "member1": r2(bm[0]), "rank_by_boxmean_members123_1=highest": [int((bm > bm[k]).sum()) + 1 for k in range(3)],
               "rmse_foehn_mean": r2(np.sqrt(np.mean((mu - g) ** 2))),
               "foehn_P75_where_ghap_gt75": r3((ml[:, y] > TH).mean()), "foehn_P75_elsewhere": r3((ml[:, ~y] > TH).mean())}
        for nm, f in (("cams", cams_field(peak - h, h)), ("aurora", aur_field(peak - h, h))):
            for bl, tag in ((False, ""), (True, "_bilinear")):
                v = at_pixels(f, la, lo, bl)
                row[nm + tag] = None if v is None else r2(v.mean())
                row["rmse_" + nm + tag] = None if v is None else r2(np.sqrt(np.mean((v - g) ** 2)))
        if h in (1, 5):                                        # variance at 10 km of each member, rank of members 1-3
            mm = m.copy(); mm[:, ~land] = np.nan
            import warnings; warnings.filterwarnings('ignore')
            lam = None; v10 = []
            for k in range(16):
                lam, P = spectrum(mm[k], 1.0)
                v10.append(np.exp(np.interp(np.log(10.0), np.log(lam[::-1]), np.log(P[::-1]))))
            ghl, ghP = spectrum(np.where(land, gh, np.nan), 1.0)
            v10 = np.array(v10); order = np.argsort(-v10)
            row["var10km_member1"] = r2(v10[0]); row["var10km_rank_member1_of16_1=highest"] = int(np.where(order == 0)[0][0]) + 1
            row["var10km_rank_members123"] = [int(np.where(order == k)[0][0]) + 1 for k in range(3)]
            row["var10km_members_min_median_max"] = [r2(v10.min()), r2(np.median(v10)), r2(v10.max())]
            row["var10km_ghap"] = r2(np.exp(np.interp(np.log(10.0), np.log(ghl[::-1]), np.log(ghP[::-1]))))
            row["var10km_ensmean"] = r2(np.exp(np.interp(np.log(10.0), np.log(lam[::-1]), np.log(spectrum(np.nanmean(mm, 0), 1.0)[1][::-1]))))
        out["box"][f"lead{h}"] = row
    # r3_prob procedure on the whole crop
    landc = tiles_box(f"{G}/static1km_tiles.npy", 1, int(R0), int(R1), int(C0), int(C1)) > 0
    ok = landc & (ghc > 0); yy = ok & (ghc > TH); nn = ok & (ghc <= TH)
    out["r3"]["share_gt75_of_land"] = r3(yy.sum() / ok.sum()); out["r3"]["ghap_mean_where_gt75"] = r2(ghc[yy].mean())
    for h in range(1, 8):
        m = np.load(f"{D}/crop0_d{peak - h}_h{h}.npz")["members"].astype(np.float32)
        p = (m > TH).mean(0); mu = m.mean(0)
        out["r3"][f"lead{h}"] = {"P_where_gt75": r3(p[yy].mean()), "share_P_ge_0.5": r3((p[yy] >= 0.5).mean()),
                                 "P_elsewhere": r3(p[nn].mean()), "ensmean_where_gt75": r2(mu[yy].mean()),
                                 "max_ensmean": r2(np.nanmax(np.where(ok, mu, np.nan)))}
    return out


res = [one(int(a)) for a in sys.argv[1:]]
print(json.dumps({"script": os.path.abspath(__file__), "eval": D, "results": res}, indent=1))
