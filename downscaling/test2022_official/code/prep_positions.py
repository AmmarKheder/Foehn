#!/usr/bin/env python3
"""A7(b): the official stations of the paper on the downscaler's lattice (positions only for the model process).

Station set: eccv/data/world/global1km/stations_2022.npz, source codes 0 (EEA), 1 (CNEMC), 3 (EPA AQS), 4 (AirKorea),
5 (Japan NIES), 6 (India CPCB) with >= 30 valid days in 2022 (the 6,324 stations of the paper). Pixel and rings exactly
as test2022_ghap/stations_2022.py (r = floor((84 - lat) / 0.01 + 1e-9), c = floor((lon + 180) / 0.01 + 1e-9) % 36000;
square rings at Chebyshev distance 1, 3, 10, station pixel excluded, ring pixels outside the GHAP mask = -1). The only
use of the observations here is the count of valid days per station (the 30-day rule); no value is written.
Outputs (test2022_official/pos/):
  st_ring_official.npz   pos (sorted unique mask positions: station pixels + ring pixels), mon_idx (n_pix,) into pos,
                         ring1 / ring3 / ring10 (n_pix, 8 d) into pos (-1 outside the mask), rings; positions only
  st_meta_official.npz   per kept station: id (row of stations_2022.npz), src (network code), lat, lon, pix (index of its
                         pixel among the n_pix unique station pixels); per pixel: pix_lat, pix_lon (pixel centres), pix_pos
                         (mask position); and for the reproduction check: frozen_idx (n_pix,) index of the same pixel in
                         test2022_ghap/st_ring_2022.npz 'pos' (-1 if absent)
  st_qc_official.json    counts
    python -I prep_positions.py"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
SRC = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km/stations_2022.npz")
MASK = ROOT / "data/processed/ghap/grid/ghap_mask.npy"
FROZEN_RING = ROOT / "test2022_ghap/st_ring_2022.npz"
NAMES = {0: "EEA", 1: "CNEMC", 3: "EPA_AQS", 4: "AirKorea", 5: "Japan_NIES", 6: "India_CPCB"}
RINGS = (1, 3, 10)


def main():
    assert (ROOT / "runs_ghap/pmG_final/VALIDATION_GATES.json").exists(), "system not frozen"
    (O / "pos").mkdir(parents=True, exist_ok=True)
    with open(O / "UNSEALING_A7.md", "a") as f:
        f.write(f"- {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Helsinki): [A7] prep_positions.py reads {SRC} "
                f"(coordinates, source codes, and the number of valid days per station for the 30-day rule; no value is "
                f"kept) to place the 6 official networks on the downscaler's lattice. Evaluation only.\n")
    z = np.load(SRC)
    lat, lon, src = z["lat"].astype(np.float64), z["lon"].astype(np.float64), z["source"].astype(np.int64)
    nvalid = np.isfinite(z["daily"]).sum(1)
    assert z["daily"].shape[1] == 365
    qc = {"source_file": str(SRC), "n_in_file": int(lat.size)}
    off = np.isin(src, list(NAMES)) & (nvalid >= 30)
    qc["official_ge30_days"] = {NAMES[k]: int((off & (src == k)).sum()) for k in NAMES}
    qc["official_ge30_days_total"] = int(off.sum())
    qc["official_in_file_lt30_days"] = {NAMES[k]: int(((src == k) & (nvalid < 30)).sum()) for k in NAMES}
    r = np.floor((84.0 - lat) / 0.01 + 1e-9).astype(np.int64)
    c = np.floor((lon + 180.0) / 0.01 + 1e-9).astype(np.int64) % 36000
    # the paper's evaluation pixel (eval_global_green.py l.336-337)
    r2 = np.clip(np.floor((84.0 - lat) / 0.01).astype(np.int64), 0, 14399)
    c2 = np.clip(np.floor((lon + 180.0) / 0.01).astype(np.int64), 0, 35999)
    qc["pixel_differs_from_paper_rule"] = int(((r != r2) | (c != c2))[off].sum())
    ok = (r >= 0) & (r < 14400)
    gid = np.where(ok, r * 36000 + c, -1)
    mask = np.load(MASK).astype(np.int64)
    p = np.searchsorted(mask, gid)
    inm = ok & (p < mask.size) & (mask[np.minimum(p, mask.size - 1)] == gid)
    qc["dropped_not_in_mask"] = {NAMES[k]: int((off & ~inm & (src == k)).sum()) for k in NAMES}
    qc["dropped_not_in_mask_total"] = int((off & ~inm).sum())
    keep = off & inm
    idx = np.where(keep)[0]
    pos = np.unique(p[idx])                                   # mask positions of the unique station pixels
    pix = np.searchsorted(pos, p[idx])
    mgid = mask[pos]
    r0, c0 = mgid // 36000, mgid % 36000
    ring_g = {}
    for dd in RINGS:
        offs = np.array([(u, v) for u in range(-dd, dd + 1) for v in range(-dd, dd + 1) if max(abs(u), abs(v)) == dd], np.int64)
        rr = r0[:, None] + offs[None, :, 0]
        cc = (c0[:, None] + offs[None, :, 1]) % 36000
        g = np.where((rr >= 0) & (rr < 14400), rr * 36000 + cc, -1)
        q = np.searchsorted(mask, g)
        okr = (g >= 0) & (q < mask.size) & (mask[np.minimum(q, mask.size - 1)] == g)
        ring_g[dd] = np.where(okr, q, -1)
    allpos = np.unique(np.concatenate([pos] + [v[v >= 0] for v in ring_g.values()]))
    ring_idx = {f"ring{dd}": np.where(v >= 0, np.searchsorted(allpos, np.maximum(v, 0)), -1) for dd, v in ring_g.items()}
    np.savez(O / "pos/st_ring_official.npz", pos=allpos.astype(np.int64), mon_idx=np.searchsorted(allpos, pos),
             rings=np.array(RINGS), **ring_idx)
    fz = np.load(FROZEN_RING)["pos"].astype(np.int64)
    j = np.searchsorted(fz, pos)
    hit = (j < fz.size) & (fz[np.minimum(j, fz.size - 1)] == pos)
    frozen_idx = np.where(hit, j, -1)
    np.savez(O / "pos/st_meta_official.npz", id=idx.astype(np.int64), src=src[idx], lat=lat[idx].astype(np.float32),
             lon=lon[idx].astype(np.float32), pix=pix.astype(np.int64), pix_pos=pos.astype(np.int64),
             pix_lat=(83.995 - 0.01 * r0).astype(np.float64), pix_lon=(-179.995 + 0.01 * c0).astype(np.float64),
             frozen_idx=frozen_idx.astype(np.int64), names=np.array([f"{k}:{v}" for k, v in NAMES.items()]))
    qc["rings"] = {f"ring{dd}": {"pixels_per_ring": 8 * dd, "frac_ring_pixels_in_mask": float((v >= 0).mean()),
                                 "station_pixels_with_half_ring_in_mask": int(((v >= 0).sum(1) >= 4 * dd).sum())}
                   for dd, v in ring_g.items()}
    qc.update(n_stations=int(idx.size), n_pixels=int(pos.size), n_positions=int(allpos.size),
              by_network={NAMES[k]: int((src[idx] == k).sum()) for k in NAMES},
              pixels_by_network={NAMES[k]: int(np.unique(pix[src[idx] == k]).size) for k in NAMES},
              station_pixels_also_in_frozen_test=int(hit.sum()),
              stations_sharing_a_pixel=int(idx.size - pos.size),
              rules="six official networks, >= 30 valid days in 2022, pixel in the GHAP mask; rings as A2(e)")
    (O / "pos/st_qc_official.json").write_text(json.dumps(qc, indent=1))
    print(json.dumps(qc, indent=1), flush=True)


if __name__ == "__main__":
    main()
