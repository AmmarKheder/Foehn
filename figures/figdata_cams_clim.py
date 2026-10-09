#!/usr/bin/env python3
"""Climatology baseline in Aurora's setup: for each valid time of the test forecasts (initialisations 00 UTC 1 May - 30 Nov 2022,
leads 12-120 h), the mean CAMS-analysis PM2.5 of the training years 2018-2021 at the same hour (00 or 12 UTC) within +-15 days
of the calendar day (cams_series.npy: the analyses Foehn-CAMS is trained on; the test period never enters). Scored like
eval_cams12h.py: latitude-weighted RMSE against the CAMS analysis at the valid time, per initialisation, then averaged; per
region; on the initialisations of the scores of the other systems.
    python figdata_cams_clim.py   -> data/cams12h_clim.npz (rmse (leads, regions), rmse_per_init (inits, leads, regions))"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "global1km"))
from eval_cams12h import LAT, NX, NY, REGIONS, truth_days  # noqa: E402

HERE = Path(__file__).resolve().parent
SERIES = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km/cams_series.npy")
FIRST = np.datetime64("2017-10-01")
YEARS = (2018, 2019, 2020, 2021)
WIN = 15


def main():
    s = np.load(SERIES, mmap_mode="r")
    cz = np.load(HERE / "data" / "cams12h_scores_foehn_emis.npz")
    doys = [int(d) for d in cz["doys"]]; NL = len(cz["leads_h"])
    need = sorted({(2 * d + k) // 2 for d in doys for k in range(1, NL + 1)})
    lo, hi = min(need) - WIN, max(need) + WIN
    # per calendar day and hour: mean over the four training years
    daily = {}
    for doy in range(lo, hi + 1):
        acc = np.zeros((2, NY, NX), np.float64)
        for y in YEARS:
            t = 2 * int((np.datetime64(f"{y}-01-01") + np.timedelta64(doy, "D") - FIRST).astype(int))
            acc += np.asarray(s[t:t + 2, 1], np.float32)
        daily[doy] = (acc / len(YEARS)).astype(np.float32)
        if doy % 30 == 0:
            print("day", doy, flush=True)
    clim = {doy: np.mean([daily[x] for x in range(doy - WIN, doy + WIN + 1)], 0) for doy in need}
    w = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, NX))
    LA2, LO2 = np.meshgrid(LAT, -180.0 + 0.4 * np.arange(NX), indexing="ij")
    RM = [np.ones((NY, NX), bool) if b is None else (LA2 <= b[0]) & (LA2 > b[1]) & (LO2 >= b[2]) & (LO2 < b[3]) for b in REGIONS.values()]
    T = truth_days(range(min(doys), max(need) + 1))
    per = np.full((len(doys), NL, len(RM)), np.nan, np.float32)
    for c, d in enumerate(doys):
        for k in range(1, NL + 1):
            vt = 2 * d + k
            e2 = w * (clim[vt // 2][vt % 2] - T[vt // 2][vt % 2]) ** 2
            for r, m in enumerate(RM):
                per[c, k - 1, r] = np.sqrt((e2 * m).sum() / (w * m).sum())
    rm = per.mean(0)
    np.savez(HERE / "data" / "cams12h_clim.npz", rmse=rm, rmse_per_init=per, doys=np.array(doys), regions=np.array(list(REGIONS)),
             leads_h=cz["leads_h"])
    print("climatology RMSE, global, by lead:", np.round(rm[:, 0], 2))


if __name__ == "__main__":
    main()
