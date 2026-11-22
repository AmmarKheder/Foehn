#!/usr/bin/env python3
"""Agreement of the re-run 1 km forecast for 9 February 2022 with the evaluated forecast (run G1).

Recomputes the two numbers of the sentence of the Methods ("Diagnostics and case studies"):
  "This map was computed again with the same model, noise fields and seeds, and its ensemble mean differs from that of the
   evaluated forecast by at most 0.0625 ug m-3 on the 0.25 deg cells and 0.25 ug m-3 at the station pixels."
Forecast: start day index 38 (0-based day of year, 8 February 2022), lead 1 day, target 39 (9 February), 4 members.
  re-run     figs_show/run/eval_show_d38_crps_e01_ar: map1km_d38_h1.npy (the 1 km ensemble mean, float16, NaN outside the
             windows) and part_0.npz (st_ens, map25 of the same forecast, written by the same code);
  evaluated  eval_final_crps_e01_ar, the part file whose keys hold (38, 1, 39).
Comparisons:
  station pixels   the 1 km map read at the station pixels with the pixel rule of the evaluation (eval_show.py:
                   row floor((84 - lat) / 0.01), column floor((lon + 180) / 0.01), float32 coordinates, clipped,
                   stations with -60 < lat < 84 inside the covered area), against the mean of the 4 members that the
                   evaluation stored at the same pixels (st_ens, float16 members, mean in float32);
                   also: the re-run's own stored members against the evaluated ones (same rule, both stored), the map
                   against the re-run's own stored members (storage only), and the map read with a float64 x 100 pixel
                   rule (to show that the indexing rule matters);
  0.25 deg cells   the stored cell means of the ensemble mean (map25, float16, cells of land with an estimate) of the two
                   runs, cell by cell (the definition of show_numbers.json, identity / map25_mean).
Read only on every input (memory maps). Deterministic. Output: rerun_9feb_station_mean.json (temporary file, fsync,
read-back, rename).
    unset PYTHONPATH; export PYTHONNOUSERSITE=1; <env>/bin/python -I -B rerun_9feb_station_mean.py"""
import json
import os
import struct
import time
import zipfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants (the only absolute paths)
STATIONS = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km/stations_2022.npz"))
RERUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/figs_show/run/eval_show_d38_crps_e01_ar"))
G1 = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
          "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/eval_final_crps_e01_ar"))
SHOW_NUMBERS = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/print/show/show_numbers.json"))
OUT = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/print/show/rerun_9feb_station_mean.json"))

KEY = (38, 1, 39)
MAP = RERUN / f"map1km_d{KEY[0]}_h{KEY[1]}.npy"
PRINTED = {"cells_max_methods": 0.0625, "station_pixels_max_methods": 0.25}


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


def row_of(folder, key):
    hits = []
    for p in sorted(folder.glob("part_*.npz")):
        if ".tmp" in p.name:
            continue
        k = np.load(p)["keys"]
        hits += [(p, int(r)) for r in np.flatnonzero((k == np.array(key)).all(1))]
    assert len(hits) == 1, (folder, key, hits)
    return hits[0]


def compare(a, b, lat=None, lon=None):
    """a, b float32 vectors or maps: finite pattern, |a - b| statistics where both are finite (float64 sums)."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    d = np.abs(a[both].astype(np.float64) - b[both].astype(np.float64))
    out = dict(finite_pattern_equal=bool(np.array_equal(fa, fb)), n=int(both.sum()), mean_abs=float(d.mean()), max_abs=float(d.max()),
               p9999_abs=float(np.percentile(d, 99.99)), share_identical=float(np.mean(d == 0)),
               mean_abs_4dec=float(f"{d.mean():.4f}"), max_abs_4dec=float(f"{d.max():.4f}"))
    if lat is not None:
        j = np.flatnonzero(both)[int(np.argmax(d))]
        out["at_max"] = dict(station=int(j), lat=float(lat[j]), lon=float(lon[j]), a=float(a[j]), b=float(b[j]))
    return out


t0 = time.time()
R = {"key": list(KEY), "printed": PRINTED}
# ---------------------------------------------------------------- the two forecasts
pn, rn = row_of(RERUN, KEY)
po, ro = row_of(G1, KEY)
zn, zo = np.load(pn), np.load(po)
R["files"] = {"rerun_part": f"{pn.name} row {rn}", "evaluated_part": f"{po.name} row {ro}", "rerun_map": MAP.name}
R["same_run_settings"] = dict(info_equal=[str(x) for x in zn["info"]] == [str(x) for x in zo["info"]],
                              info=[str(x) for x in zo["info"]], rho=[float(zn["keys_rho"][rn]), float(zo["keys_rho"][ro])],
                              base=[str(zn["base"]), str(zo["base"])])
members = int(str(zo["info"][5]))
ens_new = np.asarray(npz_memmap(pn, "st_ens")[rn]).astype(np.float32)       # (stations, members), float16 stored
ens_g1 = np.asarray(npz_memmap(po, "st_ens")[ro]).astype(np.float32)
assert ens_new.shape == ens_g1.shape and ens_g1.shape[1] == members == 4, (ens_new.shape, ens_g1.shape, members)
mean_new_st, mean_g1_st = ens_new.mean(1), ens_g1.mean(1)

# ---------------------------------------------------------------- station pixels, rule of the evaluation
st = np.load(STATIONS)
lat, lon = st["lat"], st["lon"]
assert lat.dtype == np.float32 and lon.dtype == np.float32 and len(lat) == ens_g1.shape[0]
m1 = np.load(MAP, mmap_mode="r")
GH, GW = m1.shape
assert (GH, GW) == (14400, 36000) and m1.dtype == np.float16, (m1.shape, m1.dtype)
inside = (lat < 84.0) & (lat > -60.0)
sr = np.clip(np.floor((84.0 - lat) / 0.01).astype(int), 0, GH - 1)
sc = np.clip(np.floor((lon + 180.0) / 0.01).astype(int), 0, GW - 1)
flat = sr.astype(np.int64) * GW + sc
order = np.argsort(flat, kind="stable")
m_flat = m1.reshape(-1)
at = np.empty(len(flat), np.float32)
at[order] = np.asarray(m_flat[flat[order]]).astype(np.float32)               # the re-run 1 km ensemble mean at the pixels
at = np.where(inside, at, np.nan).astype(np.float32)                         # the evaluation stores NaN outside -60..84
R["station_pixels"] = dict(stations=int(len(lat)), inside=int(inside.sum()),
                           distinct_pixels=int(len(np.unique(flat[inside & np.isfinite(at)]))),
                           finite_in_map=int(np.isfinite(at).sum()), finite_in_evaluated=int(np.isfinite(mean_g1_st).sum()))
R["map_vs_evaluated_mean"] = compare(at, mean_g1_st, lat, lon)                # the number of the sentence
R["rerun_members_vs_evaluated_members_mean"] = compare(mean_new_st, mean_g1_st, lat, lon)
R["map_vs_rerun_members_mean"] = compare(at, mean_new_st, lat, lon)
R["members_single"] = {f"m{m}": {k: v for k, v in compare(ens_new[:, m], ens_g1[:, m]).items() if k in ("n", "mean_abs", "max_abs")}
                       for m in range(members)}

# ---------------------------------------------------------------- another pixel rule (float64, x 100)
sr64 = np.clip(np.floor((84.0 - lat.astype(np.float64)) * 100).astype(int), 0, GH - 1)
sc64 = np.clip(np.floor((lon.astype(np.float64) + 180.0) * 100).astype(int), 0, GW - 1)
moved = (sr64 != sr) | (sc64 != sc)
f64 = sr64.astype(np.int64) * GW + sc64
o64 = np.argsort(f64, kind="stable")
at64 = np.empty(len(f64), np.float32)
at64[o64] = np.asarray(m_flat[f64[o64]]).astype(np.float32)
at64 = np.where(inside, at64, np.nan).astype(np.float32)
R["float64_x100_rule"] = dict(stations_on_another_pixel=int((moved & inside).sum()), **compare(at64, mean_g1_st, lat, lon))

# ---------------------------------------------------------------- 0.25 deg cells (stored cell means of the ensemble mean)
c_new = np.asarray(npz_memmap(pn, "map25")[rn]).astype(np.float32)
c_g1 = np.asarray(npz_memmap(po, "map25")[ro]).astype(np.float32)
R["cells_025"] = dict(shape=list(c_new.shape), **compare(c_new, c_g1))
sn = json.loads(SHOW_NUMBERS.read_text())["identity"]["map25_mean"]
R["cells_025"]["show_numbers_json_max_abs"] = sn["max_abs"]
R["cells_025"]["reproduces_show_numbers"] = bool(R["cells_025"]["max_abs"] == sn["max_abs"]
                                                 and abs(R["cells_025"]["mean_abs"] - sn["mean_abs"]) < 1e-9)

# ---------------------------------------------------------------- against the printed values
R["printed_vs_recomputed"] = {
    "cells max": [PRINTED["cells_max_methods"], R["cells_025"]["max_abs"], R["cells_025"]["max_abs"] <= PRINTED["cells_max_methods"]],
    "station pixels max": [PRINTED["station_pixels_max_methods"], R["map_vs_evaluated_mean"]["max_abs"],
                           R["map_vs_evaluated_mean"]["max_abs"] <= PRINTED["station_pixels_max_methods"]],
    "station pixels n": R["map_vs_evaluated_mean"]["n"]}
atomic_json(OUT, R)                                     # no clock value in the JSON: identical bytes on every run
print(json.dumps(R, indent=1))
print(f"run time {time.time() - t0:.1f} s")
