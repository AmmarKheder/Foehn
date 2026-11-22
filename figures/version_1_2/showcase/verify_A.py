#!/usr/bin/env python3
"""Independent check of item A (North China Plain, target days 8-29 January 2022). Read only.
Inputs: the run's crop files (eval_case_eastasia_crps_e01_ar, 16 members), the GHAP 1 km tile store, the CAMS daily
forecast zarr (run of 00 UTC d+1, lead h-1 = equal issuance), Aurora out_12h_all + raw analyses (as compare_fair.py).
Two scorings: (1) at 1 km over the box land (GHAP > 0), the 0.4 deg fields read at each pixel (nearest cell and bilinear);
(2) every field on the 0.4 deg cells: GHAP and Foehn's mean averaged over the land pixels of the box in each cell,
cells with >= 800 of their 1600 pixels being land inside the box."""
import glob
import json
import os
import sys

import numpy as np
import zarr

RUN = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4")
D = f"{RUN}/eval_case_eastasia_crps_e01_ar"
G = (os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km")
AURD = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/benchmark/aurora")
N_, S_, W_, E_ = 41.5, 35.5, 113.5, 121.5
R0, C0 = int(round((84 - N_) * 100)), int(round((W_ + 180) * 100))         # global 1 km row/col of the box corner
NR, NC = int(round((N_ - S_) * 100)), int(round((E_ - W_) * 100))
LEADS = (1, 2, 3, 4, 5, 6, 7)
TARGETS = list(range(7, 29))                                               # 0-based doy: 8..29 January

TP = f"{G}/ghap1km_tiles.npy"                                               # (days, 29, 71, 512, 512); mmap refused on the login node
with open(TP, "rb") as _f:
    np.lib.format.read_magic(_f); TSH, _, TDT = np.lib.format.read_array_header_1_0(_f); TH0 = _f.tell()
assert TSH[1:] == (29, 71, 512, 512), TSH


def ghap(doy):
    out = np.empty((NR, NC), np.float32)
    tr0, tr1 = R0 // 512, (R0 + NR - 1) // 512; tc0, tc1 = C0 // 512, (C0 + NC - 1) // 512
    fd = os.open(TP, os.O_RDONLY)
    blk = np.empty((tr1 - tr0 + 1, tc1 - tc0 + 1, 512, 512), np.float32)
    for a, tr in enumerate(range(tr0, tr1 + 1)):
        for b, tc in enumerate(range(tc0, tc1 + 1)):
            off = TH0 + (((1826 + doy) * 29 + tr) * 71 + tc) * 512 * 512 * TDT.itemsize
            blk[a, b] = np.frombuffer(os.pread(fd, 512 * 512 * TDT.itemsize, off), TDT).reshape(512, 512)
    os.close(fd)
    full = blk.transpose(0, 2, 1, 3).reshape((tr1 - tr0 + 1) * 512, (tc1 - tc0 + 1) * 512)
    r, c = R0 - tr0 * 512, C0 - tc0 * 512
    out[:] = full[r:r + NR, c:c + NC]
    return out


lat_px = N_ - (np.arange(NR) + 0.5) * 0.01
lon_px = W_ + (np.arange(NC) + 0.5) * 0.01
ki = np.rint((90.0 - lat_px) / 0.4).astype(int); kj = np.rint((lon_px % 360) / 0.4).astype(int) % 900
fi = (90.0 - lat_px) / 0.4; fj = (lon_px % 360) / 0.4
i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int); wi = (fi - i0)[:, None]; wj = (fj - j0)[None, :]
cell_id = (ki[:, None] * 900 + kj[None, :])


def nearest(f):
    return f[ki[:, None], kj[None, :]]


def bilin(f):
    return ((1 - wi) * ((1 - wj) * f[i0[:, None], j0[None, :]] + wj * f[i0[:, None], j0[None, :] + 1])
            + wi * ((1 - wj) * f[i0[:, None] + 1, j0[None, :]] + wj * f[i0[:, None] + 1, j0[None, :] + 1]))


cams = zarr.open((os.environ["FOEHN_WORK"] + "/eccv/data/world/cams_forecast_daily/2022.zarr"), mode="r")
aur = {}
for p in sorted(glob.glob(f"{AURD}/out_12h_all/part_*.npz")):
    z = np.load(p)
    for k, d in enumerate(z["days"]):
        aur[int(d)] = (p, k)
lat_a = np.load(sorted(glob.glob(f"{AURD}/out_12h_all/part_*.npz"))[0])["lat"]
assert abs(lat_a[0] - 90) < 1e-4 and abs(lat_a[1] - 89.6) < 1e-4
_ac = {}


def aurora(d, h):
    """Aurora initialised 00 UTC d+1; daily mean of day d+h = 0.25 x first + 0.5 x 12 UTC + 0.25 x next 00 UTC"""
    p, k = aur[d + 1]
    if p not in _ac:
        _ac.clear(); _ac[p] = np.load(p)["pred"]
    A = _ac[p][k].astype(np.float32); k0 = 2 * (h - 1)
    if h == 1:
        first = np.load(f"{AURD}/raw/{d + 1:03d}.npz")["surf"][0, 5, :A.shape[1]].astype(np.float32) * 1e9
    else:
        first = A[k0 - 1]
    f = 0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1]
    return np.vstack([f, f[-1:]])                                            # pad one row so the 0.4 index stays valid


rows = []
for t in TARGETS:
    gh = ghap(t); land = gh > 0
    cid = cell_id[land]; ucell, inv, cnt = np.unique(cid, return_inverse=True, return_counts=True); keep = cnt >= 800
    for h in LEADS:
        d = t - h
        m = np.load(f"{D}/crop0_d{d}_h{h}.npz")
        assert np.allclose(m["box"], [42, 33, 112, 131])
        r0 = int(round((42 - N_) * 100)); c0 = int(round((W_ - 112) * 100))
        mem = m["members"][:, r0:r0 + NR, c0:c0 + NC].astype(np.float32)
        mean = mem.mean(0)
        g = gh[land]; F = mean[land]
        rec = dict(t=t, date=f"2022-01-{t + 1:02d}", h=h, n_land=int(land.sum()), ghap_mean=float(g.mean()),
                   frac75=float((g > 75).mean()), foehn_mean=float(F.mean()), foehn_rmse=float(np.sqrt(np.mean((F - g) ** 2))),
                   foehn_finite=bool(np.isfinite(F).all()))
        # 0.4 deg cells
        gc = np.bincount(inv, g) / cnt; Fc = np.bincount(inv, F) / cnt
        rec["foehn_rmse_cell"] = float(np.sqrt(np.mean((Fc - gc)[keep] ** 2)))
        if h <= 5:
            cf = np.asarray(cams[d + 1, 0, h - 1], np.float32)
            af = aurora(d, h)
            for nm, f in (("cams", cf), ("aurora", af)):
                vn = nearest(f)[land]; vb = bilin(f)[land]
                rec[f"{nm}_mean"] = float(vn.mean())
                rec[f"{nm}_rmse"] = float(np.sqrt(np.mean((vn - g) ** 2)))
                rec[f"{nm}_rmse_bil"] = float(np.sqrt(np.mean((vb - g) ** 2)))
                vc = np.bincount(inv, vn) / cnt                               # constant within a cell: the cell value
                rec[f"{nm}_rmse_cell"] = float(np.sqrt(np.mean((vc - gc)[keep] ** 2)))
        rec["n_cells"] = int(keep.sum())
        rows.append(rec)
    print(t, flush=True)

out = {"rows": rows}
summ = {}
for h in (1, 2, 3, 4, 5):
    R = [r for r in rows if r["h"] == h]
    for sfx in ("", "_bil", "_cell"):
        fk = "foehn_rmse" + ("_cell" if sfx == "_cell" else "")
        f = np.array([r[fk] for r in R]); c = np.array([r["cams_rmse" + sfx] for r in R]); a = np.array([r["aurora_rmse" + sfx] for r in R])
        summ[f"h{h}{sfx}"] = dict(n=len(R), below_cams=int((f < c).sum()), below_aurora=int((f < a).sum()), below_both=int(((f < c) & (f < a)).sum()),
                                  mean_rmse=[round(float(f.mean()), 2), round(float(c.mean()), 2), round(float(a.mean()), 2)],
                                  worst_of_three=int(((f > c) & (f > a)).sum()))
out["summary"] = summ
txt = json.dumps(out, indent=1)
path = sys.argv[1]
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    fh.write(txt); fh.flush(); os.fsync(fh.fileno())
os.replace(tmp, path)
print(json.dumps(summ, indent=1))
