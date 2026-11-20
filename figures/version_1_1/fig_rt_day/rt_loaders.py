"""Read-only loaders of the real-time forecasts for the figure 'a real-time forecast at 1 km' (after Appendix A of
facts/figures.md; the arithmetic of the monitor comparison is that of the live verification script live_baselines.py).

Systems (lead day L of the 00 UTC cycle of day d = UTC day d + L - 1):
  km     real-time 1 km forecast: statistics of the 8 downscaled members per pixel (mean, p10, p90), 10 x 10 degree tiles
  rt04   real-time 0.4 degree forecast (8 members, instants every 12 h); the input of the downscaling
  cams   CAMS operational forecast of the same cycle (0.4 degree)
Daily means of the two 0.4 degree systems (since 2026-11-19 they differ):
  cams   cams_daily_3h: the mean of the EIGHT 3-hourly steps 00, 03, ..., 21 UTC of the valid day (forecast hours
         24(L-1) + 0, 3, ..., 21; the step of 24 UTC is not in the day), floored at 0: the rule of the paper's Methods ("the
         mean of its 3-hourly steps over the target day") and of the real-time scores (replay/score_cams3h, arm cams3h).
         Steps from replay/cams3h/raw/*.nc through replay/cams3h/code/c3h_lib.py (loaded read-only, no bytecode).
  rt04   daily04, UNCHANGED: 0.25 x instant 24(L-1) h + 0.5 x instant 24L-12 h + 0.25 x instant 24L h, clipped at 0, the
         0 h instant being the CAMS analysis of 00 UTC (what the downscaling receives). Its members exist every 12 h
         only, so its daily mean stays this three-instant mean.
  cams_daily (the same three instants applied to the CAMS forecast) was the reading of the figure until 2026-11-19; it is
  kept as the reference of the former reading (replay/score_cams3h/lahore.py calls it) and no longer feeds the figure.
Nothing here writes anything."""
import ast
import datetime as dt
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

DS = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
KM_ROOT = DS / "realtime_1km"            # real-time 1 km chain (folder name internal: paths only, never a label or a text)
KM, VERIF = KM_ROOT / "out", KM_ROOT / "verif"
C3H = KM_ROOT / "replay" / "cams3h"  # 3-hourly steps of the CAMS forecast (raw/*.nc) and their library (code/c3h_lib.py): read-only
TAG, TILE, RES, LON0 = "pmG_final_P3T3", 1000, 0.01, -179.995
ISSUES = [dt.date(2026, 11, 3) + dt.timedelta(k) for k in range(9)]            # 1 km tiles of the GHAP-trained system
THR = 37.5                                                                     # WHO 24-h interim target 3


def cities():
    """the 60 cities of the public list of the real-time chain (name, lat, lon), parsed without importing the module"""
    src = (RT / "products.py").read_text()
    return ast.literal_eval(re.search(r"^CITIES = (\[.*?\])\n", src, re.S | re.M).group(1))


def city_box(la, lo):
    """+-0.5 degree of latitude and the same distance in longitude: (south, north, west, east)"""
    h = 0.5 / math.cos(math.radians(la))
    return la - 0.5, la + 0.5, lo - h, lo + h


def tile_index(day, tag=TAG):
    idx, lat0 = {}, None
    for p in sorted((KM / str(day) / tag).glob("part*.json")):
        m = json.loads(p.read_text()); lat0 = float(m["lattice"]["lat_row0"])
        idx.update({t: int(m["rank"]) for t in m["tiles"]})
    return idx, lat0


def window(day, L, product, la0, la1, lo0, lo1, tag=TAG):
    """north-up float32 window of one product of lead day L (NaN outside the mask) and its extent (W, E, S, N)"""
    idx, lat0 = tile_index(day, tag)
    s0, w0 = lat0 - RES / 2, LON0 - RES / 2
    i0, i1 = int(math.floor((la0 - s0) / RES + 1e-6)), int(math.ceil((la1 - s0) / RES - 1e-6))
    j0, j1 = int(math.floor((lo0 - w0) / RES + 1e-6)), int(math.ceil((lo1 - w0) / RES - 1e-6))
    out = np.full((i1 - i0, j1 - j0), np.nan, np.float32)
    files = {}
    for r in range(i0 // TILE, (i1 - 1) // TILE + 1):
        for c in range(j0 // TILE, (j1 - 1) // TILE + 1):
            t = f"r{r:02d}c{c:02d}"
            if t not in idx:
                continue
            if idx[t] not in files:
                files[idx[t]] = np.load(KM / str(day) / tag / f"lead{L}_part{idx[t]}.npz")
            tile = files[idx[t]][f"{product}/{t}"][::-1].astype(np.float32)
            a0, a1, b0, b1 = max(i0, r * TILE), min(i1, (r + 1) * TILE), max(j0, c * TILE), min(j1, (c + 1) * TILE)
            out[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = tile[a0 - r * TILE:a1 - r * TILE, b0 - c * TILE:b1 - c * TILE]
    return out[::-1], (w0 + j0 * RES, w0 + j1 * RES, s0 + i0 * RES, s0 + i1 * RES)


def world(day, L, product, k=5, tag=TAG):
    """world mosaic from every k-th 1 km pixel: (15000 / k, 36000 / k), row 0 at 90 N; rows [600 / k:] = 84 N .. 60 S"""
    n = TILE // k
    img = np.full((15 * n, 36 * n), np.nan, np.float32)
    for f in sorted((KM / str(day) / tag).glob(f"lead{L}_part*.npz")):
        z = np.load(f)
        for key in z.files:
            if key.startswith(product + "/"):
                r, c = int(key[-5:-3]), int(key[-2:])
                img[(14 - r) * n:(15 - r) * n, c * n:(c + 1) * n] = z[key][k // 2::k, k // 2::k]
    return img


def grid_window(ext):
    """rows and columns of the 1 km grids of the paper (row 0 = 83.995 N, column 0 = 179.995 W) for an extent (W, E, S, N)"""
    w, e, s, n = ext
    return int(round((84 - n) * 100)), int(round((84 - s) * 100)), int(round((w + 180) * 100)), int(round((e + 180) * 100))


def _daily(inst, L):
    return np.maximum(0.25 * inst(24 * (L - 1)) + 0.5 * inst(24 * L - 12) + 0.25 * inst(24 * L), 0.0)


def daily04(day, L, members=False):
    """lead day L of the real-time 0.4 degree forecast: (361, 900), cell centres 84 - 0.4 i, -180 + 0.4 j"""
    pred = np.load(RT / "out" / str(day) / "pred.npy").astype(np.float32)
    an = np.load(RT / "data" / str(day) / "cams.npy")[1, 1].astype(np.float32)
    dm = _daily(lambda h: an[None] if h == 0 else pred[:, h // 12 - 1], L)
    return dm if members else dm.mean(0)


def cams_daily(day, L):
    """FORMER reading (three instants 00, 12, 24 UTC; not used by the figure since 2026-11-19, see cams_daily_3h): lead day L
    of the CAMS operational forecast of the same cycle, on the grid of daily04 (361, 900)"""
    c = np.load(RT / "data" / str(day) / "cams_fc.npz")
    cl, co, pm = c["lat"], c["lon"], c["pm25"].astype(np.float32)
    assert abs(cl[0] - 90.0) < 1e-6 and abs((cl[0] - cl[1]) - 0.4) < 1e-6 and list(c["leads"]) == list(range(12, 121, 12))
    if co[0] >= 0:
        pm = np.roll(pm, -int(np.argmin(np.abs(co - 180.0))), axis=-1)
    an = np.load(RT / "data" / str(day) / "cams.npy")[1, 1].astype(np.float32)
    pm = pm[:, 15:15 + 361]                                                    # 84 N .. 60 S
    return _daily(lambda h: an if h == 0 else pm[h // 12 - 1], L)


def _c3h_lib():
    """replay/cams3h/code/c3h_lib.py loaded from its path: nothing is written beside it (no bytecode), sys.path is untouched"""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("c3h_lib", C3H / "code" / "c3h_lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert Path(mod.RAW) == C3H / "raw", mod.RAW
    return mod


def cams_daily_3h(days, check_three=True):
    """the daily means of the CAMS operational forecast by the rule of the paper, for the 00 UTC cycles `days`:
    ({(issue day, L): field (361, 900) float64 on the grid of daily04, L = 1..5}, a dict that describes what was read).
    Lead day L = the mean of the EIGHT 3-hourly steps 00, 03, ..., 21 UTC of the UTC day `issue day + L - 1` (forecast hours
    24(L-1) + 0, 3, ..., 21, equal weights; the step of 24 UTC is not in the day), floored at 0. The steps are the float16
    values that the real-time chain stores (c3h_lib.to_f16), as in the arm cams3h of the real-time scorer, so that the
    figure and the scores read the same numbers; the fields are then read bilinearly (bilinear) like the 0.4 degree forecast.
    check_three: where the chain's own ten-lead file is still on disk, the three-instant mean recomputed from the same raw
    files must reproduce cams_daily (the former reading) to the last bit: the raw files hold the forecast of that cycle."""
    C = _c3h_lib()
    idx = C.index_raw()
    by_file = {}
    for d in days:
        assert d in idx, f"no 3-hourly CAMS forecast for the 00 UTC cycle of {d} in {C.RAW}"
        by_file.setdefault(idx[d][0], []).append(d)
    out, worst, nchk = {}, 0.0, 0
    for f, ds in sorted(by_file.items()):
        dates, x32, _, _ = C.read_file(f)                                      # checked there: grid, lead hours, valid times, finite, >= 0
        for d in ds:
            i = idx[d][1]; assert dates[i] == d, (f.name, d, dates[i])
            r16 = C.roll180(C.to_f16(x32[i]))                                  # (41 steps, 451, 900): 90 N .. 90 S, 180 W .. 179.6 E
            for L in range(1, 6):
                assert sorted(C.weights("paper", L)) == [24 * (L - 1) + 3 * k for k in range(8)] and set(C.weights("paper", L).values()) == {0.125}
                out[d, L] = C.daily_rule(r16, L, "paper")[15:15 + 361]           # 84 N .. 60 S, the grid of daily04
                if check_three and (RT / "data" / str(d) / "cams_fc.npz").exists():
                    old = cams_daily(d, L)
                    worst = max(worst, float(np.abs(C.daily_three(r16, L)[15:15 + 361].astype(np.float64) - old.astype(np.float64)).max())); nchk += 1
        del x32
    assert worst == 0.0, f"the three-instant mean from the raw files differs from cams_daily by up to {worst} ug m-3"
    info = dict(rule="mean of the eight 3-hourly steps 00, 03, ..., 21 UTC of the valid day (forecast hours 24(L-1) + 0, 3, ..., 21 of the 00 UTC cycle), "
                     "float16 steps as the chain stores them, floor at 0; read bilinearly on the 0.4 degree grid",
                raw_files=sorted(str(f) for f in by_file), cycles=[str(d) for d in days], fields=len(out),
                three_instants_from_the_same_raw_files_vs_former_loader=dict(fields_compared=nchk, max_abs_diff=worst) if check_three else None)
    return out, info


def bilinear(F, lat, lon, lat0=84.0, d=0.4, lon0=-180.0):
    H, W = F.shape[-2:]
    r = np.clip((lat0 - lat) / d, 0.0, H - 1.0); c = np.remainder((lon - lon0) / d, W)
    r0 = np.minimum(np.floor(r), H - 2).astype(np.int64); c0 = np.floor(c).astype(np.int64) % W; c1 = (c0 + 1) % W
    fr, fc = r - r0, c - np.floor(c)
    return ((1 - fr) * (1 - fc) * F[..., r0, c0] + (1 - fr) * fc * F[..., r0, c1] + fr * (1 - fc) * F[..., r0 + 1, c0]
            + fr * fc * F[..., r0 + 1, c1])


def cells04(ext):
    """the 0.4 degree cells that cover an extent (W, E, S, N): row slice, column slice and the extent of those cells"""
    w, e, s, n = ext
    i0, i1 = int(math.floor((84.2 - n) / 0.4 + 1e-6)), int(math.ceil((84.2 - s) / 0.4 - 1e-6))
    j0, j1 = int(math.floor((w + 180.2) / 0.4 + 1e-6)), int(math.ceil((e + 180.2) / 0.4 - 1e-6))
    return slice(i0, i1), slice(j0, j1), (-180.2 + 0.4 * j0, -180.2 + 0.4 * j1, 84.2 - 0.4 * i1, 84.2 - 0.4 * i0)


def nearest04(F, ext, shape):
    """a 0.4 degree field on the 1 km pixels of a window (value of the cell whose centre is nearest, no interpolation)"""
    w, e, s, n = ext
    la = n - RES / 2 - RES * np.arange(shape[0]); lo = w + RES / 2 + RES * np.arange(shape[1])
    i = np.clip(np.round((84.0 - la) / 0.4).astype(int), 0, 360); j = np.round((lo + 180.0) / 0.4).astype(int) % 900
    return F[np.ix_(i, j)]


def inside(ext, shape, box):
    """the 1 km pixels of a window whose centres lie inside a box (south, north, west, east)"""
    w, e, s, n = ext
    la = n - RES / 2 - RES * np.arange(shape[0]); lo = w + RES / 2 + RES * np.arange(shape[1])
    return ((la >= box[0]) & (la <= box[1]))[:, None] & ((lo >= box[2]) & (lo <= box[3]))[None, :]


def catalogue():
    c = np.load(VERIF / "catalogue.npz")
    return c["id"], c["lat"].astype(np.float64), c["lon"].astype(np.float64)


def observations(day, ids):
    """UTC daily means of the monitors on the catalogue order (NaN where there is none) and the file's 'final' flag"""
    z = np.load(VERIF / "obs" / f"{day}.npz")
    pos = {k: i for i, k in enumerate(ids.tolist())}
    v = np.full(ids.size, np.nan)
    k = np.array([pos.get(i, -1) for i in z["id"].tolist()], np.int64)
    v[k[k >= 0]] = z["value"][k >= 0]
    return v, bool(z["final"])


def snapshot(day):
    """1 km forecast at the monitor pixels: mean, p10, p90, each (5 lead days, monitors); None if not the GHAP system"""
    f = VERIF / "fc" / f"{day}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return (z["mean"], z["p10"], z["p90"]) if str(z["tag"]) == TAG else None
