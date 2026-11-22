#!/usr/bin/env python3
"""Station scores of the North China Plain case, target days 8-29 January 2022 (22 days), leads 1, 3 and 5 days.

Recomputes, from the stored forecasts, the station numbers of three places of the manuscript:
  Methods, "Diagnostics and case studies": the number of the 22 target days on which the RMSE of Foehn's ensemble mean
      (16 members) at the official stations between 36-41 N and 114-119.5 E is below that of the CAMS forecast AND below
      that of Aurora, 1, 3 and 5 days ahead (13, 11 and 9), and the number of stations scored (137 of the 139 of the box);
  legend of Supplementary Fig. 11, 25 January 2022: at the 137 official stations of the plain, 49.6, 54.2 and 71.6 for
      Foehn, against 34.6-43.8 for the other two;
  legend of Fig. 5, 24 January 2022: Foehn 24.8 and 28.5, CAMS 35.7 and 33.2, Aurora 26.4 and 30.1 (1 and 3 days), 5 days
      ahead Aurora 29.7 against Foehn 32.7.

Definitions (those of recheck_final2.py, section E, which this script does not import):
  stations   stations_2022.npz, source == 1 (CNEMC), 36 <= lat <= 41, 114 <= lon <= 119.5, finite daily mean on the
             target day (column = 0-based day of year, 7..28 = 8..29 January);
  Foehn      run eval_case_eastasia_crps_e01_ar, key (start d, lead h, target t) with d = t - h, st_ens (station pixels of
             the evaluation, 16 members, float16) cast to float32, plain mean over the 16 members;
  CAMS       cams_forecast_daily/2022.zarr [d + 1, 0 (pm2p5), h - 1]: run of 00 UTC on day d + 1, lead day h - 1;
  Aurora     initialised 00 UTC on day d + 1 (out_12h_all, 12-hourly steps 12..120 h, float16): daily mean of day d + h =
             0.25 x first + 0.5 x step(2h - 2) + 0.25 x step(2h - 1), first = the analysis raw/{d+1}.npz surf[0, 5] x 1e9
             for h = 1 and step(2h - 3) otherwise;
  reads      both 0.4 deg references bilinearly at the station coordinates (row (90 - lat) / 0.4, column (lon mod 360) / 0.4);
  RMSE       over the stations where Foehn, CAMS and Aurora are all finite; tallies compare the three RMSEs of a day.
Gate: the 24 January values must equal those stored in recheck_final2.json (E_fig5 / stations_plain_24jan) to 2 decimals;
otherwise the script stops with status 2 after writing its JSON.
Read only on every input. Deterministic. Output: stations_plain_22days.json (temporary file, fsync, read-back, rename).
    unset PYTHONPATH; export PYTHONNOUSERSITE=1; <env>/bin/python -I -B stations_plain_22days.py"""
import json
import os
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import zarr

# ---------------------------------------------------------------- constants (the only absolute paths)
STATIONS = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km/stations_2022.npz"))
CASE = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
            "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/eval_case_eastasia_crps_e01_ar"))
CAMSZ = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/cams_forecast_daily/2022.zarr"))
AURD = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/benchmark/aurora"))
RECHECK = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/print/show/recheck_final2.json"))
OUT = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/print/show/stations_plain_22days.json"))

BOX = dict(lat=(36.0, 41.0), lon=(114.0, 119.5))       # Methods box, bounds included
SOURCE = 1                                              # CNEMC
TARGETS = list(range(7, 29))                            # 0-based day of year: 8..29 January 2022
LEADS = (1, 3, 5)
MEMBERS = 16
DAY_24, DAY_25 = 23, 24
PRINTED = {
    "below_both_methods": {"1": 13, "3": 11, "5": 9},
    "stations_scored": 137,
    "24jan_fig5_legend": {"foehn": {"1": 24.8, "3": 28.5, "5": 32.7}, "cams": {"1": 35.7, "3": 33.2},
                             "aurora": {"1": 26.4, "3": 30.1, "5": 29.7}},
    "25jan_supp_fig11_legend": {"foehn": {"1": 49.6, "3": 54.2, "5": 71.6}, "others_range": [34.6, 43.8]},
}


def npz_memmap(path, name):
    """Read-only memory map of an uncompressed (stored) array member of an .npz archive."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name + ".npy")
        assert info.compress_type == zipfile.ZIP_STORED, (path, name, info.compress_type)
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        head = f.read(30)
        assert head[:4] == b"PK\x03\x04", path
        nlen, xlen = struct.unpack("<HH", head[26:30])
        f.seek(info.header_offset + 30 + nlen + xlen)
        major, _ = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(f)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


def atomic_json(path, obj):
    data = (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode()
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    assert tmp.read_bytes() == data, tmp
    os.replace(tmp, path)
    assert path.read_bytes() == data, path


t0 = time.time()
# ---------------------------------------------------------------- stations
st = np.load(STATIONS)
lat, lon, src, daily = st["lat"], st["lon"], st["source"], st["daily"]
inbox = (src == SOURCE) & (lat >= BOX["lat"][0]) & (lat <= BOX["lat"][1]) & (lon >= BOX["lon"][0]) & (lon <= BOX["lon"][1])

# ---------------------------------------------------------------- Foehn: station values of the evaluation, by key
KEY = {}
for p in sorted(CASE.glob("part_*.npz")):
    if ".tmp" in p.name:
        continue
    keys = np.load(p)["keys"]
    for r, k in enumerate(keys.tolist()):
        assert tuple(k) not in KEY, ("duplicate key", k)
        KEY[tuple(int(v) for v in k)] = (p, r)
_st_mm = {}


def foehn_stations(d, h, t):
    p, r = KEY[(d, h, t)]
    if p not in _st_mm:
        _st_mm[p] = npz_memmap(p, "st_ens")
    row = np.asarray(_st_mm[p][r]).astype(np.float32)                    # (stations, members)
    assert row.shape == (len(lat), MEMBERS), row.shape
    return row


# ---------------------------------------------------------------- references
cams = zarr.open(str(CAMSZ), mode="r")
assert cams.shape == (365, 2, 5, 451, 900), cams.shape
AUR = {}
aur_parts = sorted((AURD / "out_12h_all").glob("part_*.npz"))
for p in aur_parts:
    for k, d in enumerate(np.load(p)["days"]):
        AUR[int(d)] = (p, k)
assert np.array_equal(np.asarray(npz_memmap(aur_parts[0], "lat")), np.load(aur_parts[0])["lat"])   # offset logic check
_pred_mm = {}


def aurora(d, h):
    p, k = AUR[d + 1]
    if p not in _pred_mm:
        _pred_mm[p] = npz_memmap(p, "pred")
    A = np.asarray(_pred_mm[p][k]).astype(np.float32)                    # steps 12, 24, ..., 120 h after 00 UTC d + 1
    if h == 1:
        first = np.asarray(npz_memmap(AURD / "raw" / f"{d + 1:03d}.npz", "surf")[0, 5, :450]).astype(np.float32) * 1e9
    else:
        first = A[2 * h - 3]
    return 0.25 * first + 0.5 * A[2 * h - 2] + 0.25 * A[2 * h - 1]


# ---------------------------------------------------------------- scores per target day and lead
rows, scored = [], {}
for t in TARGETS:
    obs = daily[:, t]
    sel = inbox & np.isfinite(obs)
    idx = np.flatnonzero(sel)
    fi = (90 - lat[sel]) / 0.4; fj = (lon[sel] % 360) / 0.4
    i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int); wi, wj = fi - i0, fj - j0
    bil = lambda f: (1 - wi) * ((1 - wj) * f[i0, j0] + wj * f[i0, j0 + 1]) + wi * ((1 - wj) * f[i0 + 1, j0] + wj * f[i0 + 1, j0 + 1])
    o = obs[sel]
    for h in LEADS:
        d = t - h
        F = foehn_stations(d, h, t)[sel].mean(1)
        C = bil(np.asarray(cams[d + 1, 0, h - 1], np.float64))
        A = bil(aurora(d, h).astype(np.float64))
        ok = np.isfinite(F) & np.isfinite(C) & np.isfinite(A)
        rf = float(np.sqrt(np.mean((F - o)[ok] ** 2)))
        rc = float(np.sqrt(np.mean((C - o)[ok] ** 2)))
        ra = float(np.sqrt(np.mean((A - o)[ok] ** 2)))
        scored[(t, h)] = set(idx[ok].tolist())
        rows.append(dict(target_doy0=t, date=f"2022-01-{t + 1:02d}", lead=h, start_doy0=d, key=[d, h, t],
                         stations_with_measurement=int(sel.sum()), n=int(ok.sum()),
                         rmse=dict(foehn=rf, cams=rc, aurora=ra),
                         rmse_3dec=dict(foehn=round(rf, 3), cams=round(rc, 3), aurora=round(ra, 3)),
                         below_cams=rf < rc, below_aurora=rf < ra, below_both=(rf < rc) and (rf < ra),
                         highest_of_three=(rf > rc) and (rf > ra)))
    print(f"target {t + 1:2d} January done ({time.time() - t0:.0f} s)", flush=True)

R = {"definition": __doc__.split("Definitions")[1].split("Gate:")[0].strip(), "printed": PRINTED, "rows": rows}

# ---------------------------------------------------------------- tallies over the 22 days
tal = {}
for h in LEADS:
    rr = [r for r in rows if r["lead"] == h]
    assert len(rr) == len(TARGETS)
    tal[str(h)] = dict(days=len(rr), below_cams=sum(r["below_cams"] for r in rr), below_aurora=sum(r["below_aurora"] for r in rr),
                       below_both=sum(r["below_both"] for r in rr), highest_of_three=sum(r["highest_of_three"] for r in rr),
                       days_below_both=[r["date"] for r in rr if r["below_both"]],
                       closest_margin_to_the_best_reference=sorted(
                           [(r["date"], round(r["rmse"]["foehn"] - min(r["rmse"]["cams"], r["rmse"]["aurora"]), 3)) for r in rr],
                           key=lambda x: abs(x[1]))[:3],
                       mean_rmse_3dec=[round(float(np.mean([r["rmse"][s] for r in rr])), 3) for s in ("foehn", "cams", "aurora")])
R["tallies"] = tal
R["tallies_match_printed"] = {
    "below_both": {h: [tal[h]["below_both"], PRINTED["below_both_methods"][h], tal[h]["below_both"] == PRINTED["below_both_methods"][h]]
                   for h in tal},
    "below_cams_aurora_both": {h: [tal[h]["below_cams"], tal[h]["below_aurora"], tal[h]["below_both"]] for h in tal}}

# ---------------------------------------------------------------- stations scored
alls = set().union(*scored.values())
per = [len(v) for v in scored.values()]
R["stations"] = dict(cnemc_in_box=int(inbox.sum()), distinct_scored_over_22_days_and_3_leads=len(alls),
                     scored_on_every_day_and_lead=len(set.intersection(*scored.values())),
                     per_day_and_lead_min_max=[min(per), max(per)],
                     per_day_at_lead_1={f"2022-01-{t + 1:02d}": len(scored[(t, 1)]) for t in TARGETS},
                     same_stations_at_the_three_leads_every_day=all(scored[(t, 1)] == scored[(t, 3)] == scored[(t, 5)] for t in TARGETS),
                     scored_24jan=len(scored[(DAY_24, 1)]), scored_25jan=len(scored[(DAY_25, 1)]))


# ---------------------------------------------------------------- 24 and 25 January
def day(t):
    return {str(r["lead"]): dict(n=r["n"], **r["rmse_3dec"]) for r in rows if r["target_doy0"] == t}


R["24jan"], R["25jan"] = day(DAY_24), day(DAY_25)
six = [R["25jan"][h][s] for h in ("1", "3", "5") for s in ("cams", "aurora")]
R["25jan_others_range_3dec"] = [min(six), max(six)]
rp = lambda x: float(f"{x:.1f}")
R["printed_vs_recomputed"] = {
    "24jan": {f"{s} {h} d": [PRINTED["24jan_fig5_legend"][s][h], R["24jan"][h][s], rp(R["24jan"][h][s]) == PRINTED["24jan_fig5_legend"][s][h]]
              for s in ("foehn", "cams", "aurora") for h in PRINTED["24jan_fig5_legend"][s]},
    "25jan": {**{f"foehn {h} d": [PRINTED["25jan_supp_fig11_legend"]["foehn"][h], R["25jan"][h]["foehn"],
                                  rp(R["25jan"][h]["foehn"]) == PRINTED["25jan_supp_fig11_legend"]["foehn"][h]] for h in ("1", "3", "5")},
              "others range": [PRINTED["25jan_supp_fig11_legend"]["others_range"], R["25jan_others_range_3dec"],
                               [rp(v) for v in R["25jan_others_range_3dec"]] == PRINTED["25jan_supp_fig11_legend"]["others_range"]]}}

# ---------------------------------------------------------------- gate: the 24 January values of recheck_final2.json
ref = json.loads(RECHECK.read_text())["E_fig5"]["stations_plain_24jan"]
g = {"stored_stations": ref["stations"], "recomputed_stations": int((inbox & np.isfinite(daily[:, DAY_24])).sum()), "leads": {}}
ok_all = g["stored_stations"] == g["recomputed_stations"]
for h in ("1", "3", "5"):
    mine = R["24jan"][h]
    g["leads"][h] = {s: [ref[h][s], round(mine[s], 2)] for s in ("foehn", "cams", "aurora")}
    g["leads"][h]["n"] = [ref[h]["n"], mine["n"]]
    ok_all &= all(ref[h][s] == round(mine[s], 2) for s in ("foehn", "cams", "aurora")) and ref[h]["n"] == mine["n"]
g["passed"] = bool(ok_all)
R["gate_24jan_recheck_final2"] = g

atomic_json(OUT, R)                                     # no clock value in the JSON: identical bytes on every run
print(json.dumps({k: R[k] for k in ("gate_24jan_recheck_final2", "tallies", "tallies_match_printed", "stations", "24jan", "25jan",
                                    "25jan_others_range_3dec", "printed_vs_recomputed")}, indent=1))
print(f"run time {time.time() - t0:.1f} s")
if not g["passed"]:
    sys.exit(2)
