#!/usr/bin/env python3
"""Station truth 2022 for the GHAP system (PROTOCOL.md A2(d); evaluation only; opened after the freeze, logged).

Source: the Foehn evaluation file eccv/data/world/global1km/stations_2022_utc.npz (build_stations_utc.py: UTC-day means,
>= 12 hourly values, 0-1000 ug m-3, stations with >= 30 valid days on the local day; OpenAQ inside the EEA box and within
2 km of an official station dropped). Networks kept: CNEMC (1), OpenAQ (2), EPA AQS (3), AirKorea (4),
Japan NIES (5), India CAAQMS (6); EEA (0) is not in the list and is left out (so Europe has no OpenAQ either).
Rules here: >= 30 valid UTC days in 2022; pixel = the GHAP 1 km lattice pixel containing the monitor
(r = floor((84 - lat) / 0.01), c = floor((lon + 180) / 0.01)); monitors whose pixel is not in the GHAP mask (sea,
coast, permanent ice, < 95 % valid days) are dropped and counted.
Outputs (test2022_ghap/): st_pix_2022.npz (pos: sorted unique monitor-pixel positions in the GHAP mask order),
st_ring_2022.npz (A2(e): pos = monitor pixels + ring pixels at Chebyshev distance 1, 3, 10 inside the mask, the
positions the model process writes; mon_idx; ring<d> (n_monitor_pixels, 8 d) indices into pos, -1 outside the mask), st_truth_2022.npz (id, network, lat, lon, pix, daily (N, 365) float32, daily12 = daily), st_qc_2022.json
  python -I test2022_ghap/stations_2022.py"""
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_ghap"
SRC = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km/stations_2022_utc.npz")
MASK = ROOT / "data/processed/ghap/grid/ghap_mask.npy"
UNSEAL = ROOT / "logs/UNSEALING.md"
NAMES = {0: "EEA", 1: "CNEMC", 2: "OpenAQ", 3: "EPA_AQS", 4: "AirKorea", 5: "Japan_NIES", 6: "India_CAAQMS"}
KEEP = (1, 2, 3, 4, 5, 6)
RINGS = (1, 3, 10)                                   # A2(e): Chebyshev distance in 1 km pixels


def main():
    assert (ROOT / "runs_ghap/pmG_final/VALIDATION_GATES.json").exists(), "system not frozen: stations stay closed"
    with open(UNSEAL, "a") as f:
        f.write(f"- {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Helsinki), job {os.environ.get('SLURM_JOB_ID', '-')}: "
                f"[ghap] OPENING the 2022 station file {SRC} (UTC days; networks CNEMC, OpenAQ, EPA AQS, AirKorea, Japan NIES, "
                f"India CAAQMS) for the A2(d) station score of the frozen runs_ghap/pmG_final (evaluation only).\n")
    z = np.load(SRC)
    lat, lon, src, daily = z["lat"].astype(np.float64), z["lon"].astype(np.float64), z["source"], z["daily"].astype(np.float32)
    assert daily.shape[1] == 365
    qc = {"source_file": str(SRC), "n_in_file": int(lat.size), "by_source_in_file": {NAMES[k]: int((src == k).sum()) for k in NAMES}}
    keep = np.isin(src, KEEP)
    nvalid = np.isfinite(daily).sum(1)
    qc["dropped_eea"] = int((src == 0).sum())
    qc["dropped_lt30_days"] = int((keep & (nvalid < 30)).sum())
    keep &= nvalid >= 30
    r = np.floor((84.0 - lat) / 0.01 + 1e-9).astype(np.int64)
    c = np.floor((lon + 180.0) / 0.01 + 1e-9).astype(np.int64) % 36000
    ok = (r >= 0) & (r < 14400)
    gid = np.where(ok, r * 36000 + c, -1)
    mask = np.load(MASK).astype(np.int64)
    p = np.searchsorted(mask, gid)
    inm = ok & (p < mask.size) & (mask[np.minimum(p, mask.size - 1)] == gid)
    qc["dropped_not_in_ghap_mask"] = {NAMES[k]: int((keep & ~inm & (src == k)).sum()) for k in KEEP}
    keep &= inm
    idx = np.where(keep)[0]
    pos = np.unique(p[idx])
    pix = np.searchsorted(pos, p[idx])
    net = np.array([NAMES[int(k)] for k in src[idx]])
    np.savez(O / "st_pix_2022.npz", pos=pos.astype(np.int64))
    # A2(e): square rings at Chebyshev distance d = 1, 3, 10 pixels around each monitor pixel (monitor pixel excluded,
    # product_audit/audit_2021.py ring_offsets); ring pixels outside the GHAP mask get index -1
    mgid = mask[pos]
    r0, c0 = mgid // 36000, mgid % 36000
    ring_g = {}
    for dd in RINGS:
        off = np.array([(u, v) for u in range(-dd, dd + 1) for v in range(-dd, dd + 1) if max(abs(u), abs(v)) == dd], np.int64)
        rr = r0[:, None] + off[None, :, 0]
        cc = (c0[:, None] + off[None, :, 1]) % 36000
        g = np.where((rr >= 0) & (rr < 14400), rr * 36000 + cc, -1)
        q = np.searchsorted(mask, g)
        ok = (g >= 0) & (q < mask.size) & (mask[np.minimum(q, mask.size - 1)] == g)
        ring_g[dd] = np.where(ok, q, -1)
    allpos = np.unique(np.concatenate([pos] + [v[v >= 0] for v in ring_g.values()]))
    ring_idx = {f"ring{dd}": np.where(v >= 0, np.searchsorted(allpos, np.maximum(v, 0)), -1) for dd, v in ring_g.items()}
    np.savez(O / "st_ring_2022.npz", pos=allpos.astype(np.int64), mon_idx=np.searchsorted(allpos, pos),
             rings=np.array(RINGS), **ring_idx)
    qc["rings"] = {f"ring{dd}": {"pixels_per_ring": 8 * dd, "frac_ring_pixels_in_mask": float((v >= 0).mean()),
                                 "monitor_pixels_with_half_ring_in_mask": int(((v >= 0).sum(1) >= 4 * dd).sum())}
                   for dd, v in ring_g.items()}
    qc["n_positions_written_by_the_model"] = int(allpos.size)
    np.savez(O / "st_truth_2022.npz", id=idx.astype(np.int64), network=net, lat=lat[idx].astype(np.float32),
             lon=lon[idx].astype(np.float32), pix=pix.astype(np.int64), daily=daily[idx], daily12=daily[idx])
    qc.update(n_monitors=int(idx.size), n_pixels=int(pos.size), by_network={n: int((net == n).sum()) for n in sorted(set(net))},
              station_days=int(np.isfinite(daily[idx]).sum()), rules="UTC days, >= 12 h (Foehn), >= 30 valid days, GHAP mask pixel")
    (O / "st_qc_2022.json").write_text(json.dumps(qc, indent=1))
    print(json.dumps(qc, indent=1), flush=True)


if __name__ == "__main__":
    main()
