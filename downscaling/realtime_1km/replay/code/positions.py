#!/usr/bin/env python3
"""WP1a: the mask pixels at which the downscaler is evaluated for the replay (positions only: no observation is read).

From realtime_1km/verif/catalogue.npz (12,136 OpenAQ monitors, 'pos' = index of the monitor's pixel in the GHAP mask order):
  mon      (n,)        index into `pos` of each monitor's own pixel
  ring<d>  (n, 8 d)    d = 1, 3, 10: the square ring at Chebyshev distance d pixels around the monitor pixel (monitor pixel
                       excluded), exactly as test2022_ghap/stations_2022.py (offsets (u, v), max(|u|, |v|) = d, u outer
                       loop, v inner loop; rows 0..14399, columns periodic); index into `pos`, -1 outside the mask
  cell     (n,)        index into `ucell` of the monitor's 0.25 degree cell (data/processed/ghap/grid/cell_id.npy, flat
                       r * 1440 + c of the 576 x 1440 encoder grid)
  cell_idx (C, 25)     a 5 x 5 sample of the mask pixels of each such cell: the 25 x 25 pixel block of cell (cr, cc) is
                       rows 25 cr - 13 .. 25 cr + 11 and columns 25 cc - 13 .. 25 cc + 11 of the GHAP lattice (row 0 =
                       83.995 N, column 0 = 179.995 W); the sample is the centre pixel of each of its 5 x 5 sub-blocks,
                       rows 25 cr + a, columns 25 cc + b, a, b in (-11, -6, -1, 4, 9); index into `pos`, -1 when the pixel
                       is not in the mask (or not attributed to that cell by cell_id.npy)
  pos      (U,)        sorted unique mask positions of all the above (the pixels the model process evaluates)
-> replay/positions.npz (+ ring<d>_n, cell_n: number of valid pixels per monitor; id, lat, lon, region copied) and
replay/positions_store.npz (the rows of the GHAP store files at `pos`: flat lattice id, clim_mean, clim_var, static).
Check: the rings equal those of hindcast/2026/st_ring_2026.npz for every catalogue monitor.
    python -I positions.py"""
import json
import time
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
R = P / "realtime_1km/replay"
GH = P / "data/processed/ghap"
NR, NC = 14400, 36000
RINGS = (1, 3, 10)
OFF5 = np.array([-11, -6, -1, 4, 9], np.int64)


def lookup(mask, g):
    """flat lattice ids g (any shape, -1 = outside the lattice) -> position in the mask order, -1 if not in the mask"""
    q = np.searchsorted(mask, np.maximum(g, 0))
    ok = (g >= 0) & (q < mask.size) & (mask[np.minimum(q, mask.size - 1)] == g)
    return np.where(ok, q, -1)


def main():
    t0 = time.time()
    cat = np.load(P / "realtime_1km/verif/catalogue.npz")
    cpos = cat["pos"].astype(np.int64)
    n = cpos.size
    mask = np.load(GH / "grid/ghap_mask.npy").astype(np.int64)
    assert (np.diff(mask) > 0).all()
    cid = np.load(GH / "grid/cell_id.npy", mmap_mode="r")
    gid = mask[cpos]
    r0, c0 = gid // NC, gid % NC
    ring_pos = {}
    for dd in RINGS:
        off = np.array([(u, v) for u in range(-dd, dd + 1) for v in range(-dd, dd + 1) if max(abs(u), abs(v)) == dd], np.int64)
        rr = r0[:, None] + off[None, :, 0]
        cc = (c0[:, None] + off[None, :, 1]) % NC
        ring_pos[dd] = lookup(mask, np.where((rr >= 0) & (rr < NR), rr * NC + cc, -1))
    mcell = np.asarray(cid[np.sort(cpos)]).astype(np.int64)[np.argsort(np.argsort(cpos))]
    ucell, cinv = np.unique(mcell, return_inverse=True)
    cr, cc_ = ucell // 1440, ucell % 1440
    rr = (25 * cr)[:, None, None] + OFF5[None, :, None]
    cc = ((25 * cc_)[:, None, None] + OFF5[None, None, :]) % NC
    g = np.where((rr >= 0) & (rr < NR), rr * NC + cc, -1).reshape(ucell.size, 25)
    cp = lookup(mask, g)
    okc = cp >= 0
    same = np.zeros_like(okc)
    same[okc] = np.asarray(cid[np.sort(cp[okc])]).astype(np.int64)[np.argsort(np.argsort(cp[okc]))] == np.broadcast_to(ucell[:, None], cp.shape)[okc]
    n_other = int((okc & ~same).sum())
    cp = np.where(same, cp, -1)
    pos = np.unique(np.concatenate([cpos] + [v[v >= 0] for v in ring_pos.values()] + [cp[cp >= 0]]))
    idx = lambda v: np.where(v >= 0, np.searchsorted(pos, np.maximum(v, 0)), -1).astype(np.int32)
    out = dict(pos=pos, mon=idx(cpos), cell=cinv.astype(np.int32), ucell=ucell, cell_idx=idx(cp),
               cell_n=(cp >= 0).sum(1)[cinv].astype(np.int16), rings=np.array(RINGS),
               id=cat["id"], lat=cat["lat"], lon=cat["lon"], region=cat["region"], catalogue_pos=cpos)
    for dd in RINGS:
        out[f"ring{dd}"] = idx(ring_pos[dd])
        out[f"ring{dd}_n"] = (ring_pos[dd] >= 0).sum(1).astype(np.int16)
    # check against the ring file of the 2026 station set (built with the stations_2022.py definition)
    ref = np.load(P / "hindcast/2026/st_ring_2026.npz")
    rpix = np.load(P / "hindcast/2026/st_pix_2026.npz")["pos"].astype(np.int64)
    j = np.searchsorted(rpix, cpos)
    assert (rpix[j] == cpos).all()
    chk = {}
    for dd in RINGS:
        a = ref[f"ring{dd}"][j]
        refpos = np.where(a >= 0, ref["pos"][np.maximum(a, 0)], -1)
        chk[f"ring{dd}_equal_to_st_ring_2026"] = bool(np.array_equal(refpos, ring_pos[dd]))
    assert all(chk.values()), chk
    assert np.array_equal(ref["pos"][ref["mon_idx"][j]], cpos)
    np.savez(R / "positions.npz", **out)
    # the rows of the GHAP store at these positions (what GhapStore would give infer.py), so that the model processes do
    # not page through the 9 GB files: flat lattice id, climatology coefficients, variance coefficients, statics
    t1 = time.time()
    take = lambda f: np.ascontiguousarray(np.load(GH / f, mmap_mode="r")[pos])
    np.savez(R / "positions_store.npz", pos=pos, flat=mask[pos], clim_mean=take("ghap_clim/clim_mean_G.npy"),
             clim_var=take("ghap_clim/clim_var_G.npy"), static=take("ghap_clim/static_G.npy"))
    print(f"positions_store.npz written ({time.time() - t1:.0f} s)", flush=True)
    info = dict(n_monitors=int(n), n_monitor_pixels=int(np.unique(cpos).size), n_cells=int(ucell.size), n_positions=int(pos.size),
                ring_valid_fraction={f"ring{dd}": float((ring_pos[dd] >= 0).mean()) for dd in RINGS},
                ring_half_valid_monitors={f"ring{dd}": int(((ring_pos[dd] >= 0).sum(1) >= 4 * dd).sum()) for dd in RINGS},
                ring_no_pixel_monitors={f"ring{dd}": int(((ring_pos[dd] >= 0).sum(1) == 0).sum()) for dd in RINGS},
                cell_sample_pixels=dict(mean=float((cp >= 0).sum(1).mean()), min=int((cp >= 0).sum(1).min()),
                                        cells_with_25=int(((cp >= 0).sum(1) == 25).sum()), cells_with_0=int(((cp >= 0).sum(1) == 0).sum()),
                                        monitors_with_0=int((out["cell_n"] == 0).sum()), dropped_other_cell=n_other),
                checks=chk, seconds=round(time.time() - t0, 1),
                sources=dict(catalogue=str(P / "realtime_1km/verif/catalogue.npz"), mask=str(GH / "grid/ghap_mask.npy"),
                             cell_id=str(GH / "grid/cell_id.npy")))
    (R / "positions.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
