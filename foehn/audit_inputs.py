#!/usr/bin/env python3
"""Audit of every input of the global 1 km model (written after the 180 deg emission shift of 2026-09-30):
geographic alignment (known places), temporal alignment (the same day in two independent sources agrees better than
neighbouring days), time-of-day and sign conventions, and stability across years. Prints PASS / FAIL per check.
    source world/env.sh; python audit_inputs.py"""
import os
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
T0 = np.datetime64("2017-01-01")
day = lambda s: int((np.datetime64(s) - T0).astype(int))
NFAIL = 0


def check(name, ok, detail=""):
    global NFAIL
    NFAIL += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


class R:
    """direct reads of a .npy file"""
    def __init__(self, name):
        self.p = str(G / name)
        with open(self.p, "rb") as f:
            np.lib.format.read_magic(f); self.shape, _, self.dt = np.lib.format.read_array_header_1_0(f); self.hdr = f.tell()
        self.fd = os.open(self.p, os.O_RDONLY)

    def plane(self, *lead):
        H, W = self.shape[-2:]; k = int(np.ravel_multi_index(lead, self.shape[:-2])); n = H * W
        b = os.pread(self.fd, n * self.dt.itemsize, self.hdr + k * n * self.dt.itemsize)
        return np.frombuffer(b, self.dt).reshape(H, W).astype(np.float32)


class Tiles(R):
    """pixel reads of a (K, 29, 71, 512, 512) block file (global 1 km grid)"""
    def pixels(self, k, rows, cols):
        rows, cols = np.asarray(rows), np.asarray(cols)
        off = (((k * 29 + rows // 512) * 71 + cols // 512) * 512 + rows % 512) * 512 + cols % 512
        return np.array([np.frombuffer(os.pread(self.fd, 2, self.hdr + 2 * int(o)), np.float16)[0] for o in off], np.float32)


c25 = lambda la, lo: (int((84 - la) / 0.25), int((lo + 180) / 0.25) % 1440)
c05 = lambda la, lo: (int((84 - la) / 0.5), int((lo + 180) / 0.5) % 720)
p1 = lambda la, lo: (int((84 - la) / 0.01), int((lo + 180) / 0.01) % 36000)
corr = lambda a, b: float(np.corrcoef(np.ravel(a)[np.isfinite(np.ravel(a)) & np.isfinite(np.ravel(b))],
                                      np.ravel(b)[np.isfinite(np.ravel(a)) & np.isfinite(np.ravel(b))])[0, 1])

print("== 1. static 1 km (elevation (z - 300) / 500, land)")
ST = Tiles("static1km_tiles.npy")
places = {"Everest": (27.99, 86.93), "Dead Sea": (31.50, 35.50), "Paris": (48.86, 2.35), "Denver": (39.74, -104.99),
          "La Paz": (-16.50, -68.15), "mid-Atlantic": (30.0, -40.0)}
z = {k: 500 * ST.pixels(0, [p1(*v)[0]], [p1(*v)[1]])[0] + 300 for k, v in places.items()}
land = {k: ST.pixels(1, [p1(*v)[0]], [p1(*v)[1]])[0] for k, v in places.items()}
check("elevation at known places", z["Everest"] > 5500 and z["Dead Sea"] < 0 and 0 < z["Paris"] < 200 and 1300 < z["Denver"] < 2000
      and z["La Paz"] > 3000, ", ".join(f"{k} {v:.0f} m" for k, v in z.items()))
check("land mask", land["Paris"] == 1 and land["mid-Atlantic"] == 0, ", ".join(f"{k} {v:.0f}" for k, v in land.items()))

print("== 2. GHAP 1 km vs stations (pixel and day alignment)")
FT = Tiles("ghap1km_tiles.npy")
st = np.load(G / "stations_2022.npz")
sel = np.where(st["source"] == 1)[0][::4]                                       # CNEMC (in GHAP training: sharp test)
sr = np.floor((84 - st["lat"][sel]) / 0.01).astype(int); sc = np.floor((st["lon"][sel] + 180) / 0.01).astype(int)
res = {-1: [], 0: [], 1: []}; shifted = {}
for t in range(10, 360, 25):
    obs = st["daily"][sel, t]
    for lag in (-1, 0, 1):
        g = FT.pixels(1826 + t + lag, sr, sc); g[g <= 0] = np.nan
        res[lag].append(corr(g, obs))
r = {k: np.mean(v) for k, v in res.items()}
check("GHAP day alignment (CNEMC, 15 days)", r[0] > r[-1] and r[0] > r[1], f"corr same day {r[0]:.3f}, day-1 {r[-1]:.3f}, day+1 {r[1]:.3f}")
t = 20; obs = st["daily"][sel, t]
for dr, dc in ((0, 0), (0, 30), (0, -30), (30, 0), (-30, 0)):
    g = FT.pixels(1826 + t, sr + dr, sc + dc); g[g <= 0] = np.nan; shifted[(dr, dc)] = corr(g, obs)
check("GHAP pixel alignment (30 km shifts lower the station correlation)", all(shifted[(0, 0)] >= v for v in shifted.values()),
      ", ".join(f"{k}: {v:.3f}" for k, v in shifted.items()))

print("== 3. registration 0.25 deg <-> 1 km as the network samples it (FullNet grid16: coarse point k at fine edge 25 k)")
PM = R("pm.npy")
def bilinear(c, y, x):
    y0 = np.clip(np.floor(y).astype(int), 0, c.shape[0] - 2); x0 = np.clip(np.floor(x).astype(int), 0, c.shape[1] - 2)
    fy, fx = (y - y0)[:, None], (x - x0)[None]
    return (1 - fy) * ((1 - fx) * c[np.ix_(y0, x0)] + fx * c[np.ix_(y0, x0 + 1)]) + fy * ((1 - fx) * c[np.ix_(y0 + 1, x0)] + fx * c[np.ix_(y0 + 1, x0 + 1)])
d, k0, j0 = day("2022-01-21"), 120, 1100
na, nb = 4192 // 16, 6992 // 16
cs = bilinear(PM.plane(d)[k0:k0 + 168, j0:j0 + 280], (np.arange(na) * 16 + 8) / 25.0, (np.arange(nb) * 16 + 8) / 25.0)
err = {}
for sh in (-12, 0, 12):
    rr, cc = np.arange(25 * k0 + sh, 25 * k0 + sh + na * 16), np.arange(25 * j0 + sh, 25 * j0 + sh + nb * 16)
    w = np.stack([FT.pixels(d, np.full(len(cc), r), cc) for r in rr[::16]])            # one fine row per 16 km cell
    w[w <= 0] = np.nan
    with np.errstate(all="ignore"):
        f16 = np.nanmean(w.reshape(na, nb, 16), 2)
    ok = np.isfinite(f16) & np.isfinite(cs); err[sh] = float(np.sqrt(np.mean((f16[ok] - cs[ok]) ** 2)))
check("fine window of coarse points k0.. starts at 25 k0 (loader origin)", err[0] < min(err[-12], err[12]),
      ", ".join(f"shift {k:+d} px: RMSE {v:.2f}" for k, v in err.items()))

print("== 4. ERA5 weather (met.npy): places, day alignment with HRES, stability across years")
MET = R("met.npy"); S = np.load(G / "stats.npz"); em, es = S["era5_mean"], S["era5_std"]
d = day("2022-07-10")
sp = MET.plane(d, 4) * es[4] + em[4]; t2 = MET.plane(d, 2) * es[2] + em[2]
check("surface pressure: Tibet << ocean", sp[c25(32, 88)] < sp[c25(25, -92)] - 3, f"Tibet {sp[c25(32, 88)]:+.2f}, Gulf of Mexico {sp[c25(25, -92)]:+.2f} (store units)")
check("July t2m: Greenland << Sahara", t2[c25(72, -40)] < t2[c25(23, 10)] - 1, f"Greenland {t2[c25(72, -40)]:+.2f}, Sahara {t2[c25(23, 10)]:+.2f}")
HB = R("hres_block.npy")
for c, nm in ((0, "u10"), (2, "t2m"), (11, "u925"), (29, "z500")):
    h = HB.plane(day("2022-03-01") - 1826, 0, c)                          # run of 1 Mar, lead 1 = 2 Mar
    cs = {lag: corr(h, MET.plane(day("2022-03-02") + lag, c) * es[c] + em[c]) for lag in (-1, 0, 1)}
    check(f"HRES lead 1 vs ERA5 day alignment ({nm})", cs[0] > cs[-1] and cs[0] > cs[1], ", ".join(f"lag {k}: {v:.3f}" for k, v in cs.items()))
hr = HB.plane(day("2022-03-01") - 1826, 0, 30); er = MET.plane(day("2022-03-02"), 33) * S["extra_std"][3] + S["extra_mean"][3]
check("precipitation: HRES tp (m) vs ERA5 log1p(mm)", corr(np.log1p(np.maximum(hr, 0) * 1000), er) > 0.7,
      f"corr {corr(np.log1p(np.maximum(hr, 0) * 1000), er):.3f}")
lm = np.load(G / "meta.npz")["land"] > 0.5
yr = {y: np.array([np.nanmean(MET.plane(day(f"{y}-{m}-15"), c)[lm]) for c in range(37) for m in ("01", "07")]).reshape(37, 2).mean(1)
      for y in range(2017, 2023)}
jump = np.max(np.abs(np.array([yr[y] for y in range(2017, 2023)]) - yr[2017]), 0)
check("met channels stable across years (|mean change| < 0.5 std)", jump.max() < 0.5, f"largest change {jump.max():.2f} (channel {int(np.argmax(jump))})")

print("== 5. 3-hourly winds (wind05.npy): variables, signs, day and time-of-day alignment")
WD = R("wind05.npy"); d = day("2021-10-05")
for ch, mc, nm in ((0, 0, "u10"), (1, 1, "v10"), (2, 11, "u925"), (3, 16, "v925")):
    wm = np.mean([WD.plane(d, s, ch) for s in range(8)], 0)
    cs = {lag: corr(wm, (MET.plane(d + lag, mc) * es[mc] + em[mc]).reshape(288, 2, 720, 2).mean((1, 3))) for lag in (-1, 0, 1)}
    check(f"wind05 {nm} = ERA5 daily {nm}, same day", cs[0] > 0.95 and cs[0] > max(cs[-1], cs[1]), ", ".join(f"lag {k}: {v:.3f}" for k, v in cs.items()))
HW = R("hres_wind.npy"); d0 = day("2022-05-10")
for s in (0, 4):
    a = HW.plane(d0 - 1826, s, 0)
    cs = {k: corr(a, WD.plane(d0, min(max(s + k, 0), 7), 0)) for k in (-1, 0, 1)}
    check(f"HRES 3-hourly step {s} ({3 * s:02d} UTC) vs ERA5 wind05 steps", cs[0] >= max(cs[-1], cs[1]), ", ".join(f"step {s + k}: {v:.3f}" for k, v in cs.items()))

print("== 6. anthropogenic emissions (emis_ant_fixed.npy): hotspots, empty places, seasons")
EA = R("emis_ant_fixed.npy")
hot = {"Beijing": (39.9, 116.4), "Delhi": (28.6, 77.2), "New York": (40.7, -74.0), "Paris": (48.9, 2.35), "Lagos": (6.5, 3.4),
       "Sao Paulo": (-23.5, -46.6), "Highveld": (-26.0, 29.2)}
empty = {"Sahara": (23.0, 5.0), "Amazon": (-5.0, -62.0), "Tibet": (33.0, 86.0), "Pacific": (30.0, -150.0), "Greenland": (72.0, -40.0)}
nox = EA.plane(60, 3)
check("NOx: every hotspot > every empty place", min(nox[c25(*v)] for v in hot.values()) > max(nox[c25(*v)] for v in empty.values()),
      "hot " + ", ".join(f"{k} {nox[c25(*v)]:+.1f}" for k, v in hot.items()) + " | empty " + ", ".join(f"{k} {nox[c25(*v)]:+.1f}" for k, v in empty.items()))
for sp_, nm in enumerate(["BC", "OC", "SO2", "NOx", "NH3", "CO"]):
    a = EA.plane(60, sp_)
    check(f"{nm}: Beijing > Pacific and Sahara", a[c25(39.9, 116.4)] > max(a[c25(30, -150)], a[c25(23, 5)]),
          f"Beijing {a[c25(39.9, 116.4)]:+.2f}, Pacific {a[c25(30, -150)]:+.2f}, Sahara {a[c25(23, 5)]:+.2f}")
bcj, bcjul = EA.plane(60, 0)[c25(39.9, 116.4)], EA.plane(66, 0)[c25(39.9, 116.4)]
check("BC Beijing: January (heating) >= July", bcj >= bcjul, f"Jan {bcj:+.2f}, Jul {bcjul:+.2f}")

print("== 7. fires (fires.npy): known events at the right place and date")
FI = R("fires.npy")
events = {"Black Summer, SE Australia": ((-36.5, 149.5), "2020-01-01", "2019-06-01"),
          "August Complex, California": ((39.8, -122.9), "2020-09-15", "2020-04-15"),
          "Yakutia, Siberia": ((63.0, 128.0), "2021-08-05", "2021-02-20"),
          "Landiras, Gironde": ((44.5, -0.9), "2022-07-14", "2022-04-14")}
for nm, ((la, lo), on, off) in events.items():
    win = lambda dd, la_, lo_: float(np.max(FI.plane(day(dd), 0)[c25(la_ + 3, 0)[0]:c25(la_ - 3, 0)[0]][:, np.arange(int((lo_ - 3 + 180) / 0.25), int((lo_ + 3 + 180) / 0.25)) % 1440]))
    a, b, mirror = win(on, la, lo), win(off, la, lo), win(on, la, lo + 180 if lo < 0 else lo - 180)
    check(f"fire {nm}", a > b + 1 and a > mirror + 1, f"on {on}: {a:+.2f}, {off}: {b:+.2f}, mirror (lon+180): {mirror:+.2f}")

print("== 8. population (pop1km_global.npy)")
pop = np.load(G / "pop1km_global.npy", mmap_mode="r")
tot = float(np.sum(np.asarray(pop, np.float64)))
dens = {k: float(np.asarray(pop[p1(*v)[0] - 3:p1(*v)[0] + 4, p1(*v)[1] - 3:p1(*v)[1] + 4]).mean()) for k, v in
        {"Dhaka": (23.8, 90.4), "Paris": (48.86, 2.35), "Cairo": (30.05, 31.25), "Sahara": (23.0, 5.0), "Siberia": (65.0, 110.0)}.items()}
check("population: total and densities", 7.5e9 < tot < 8.2e9 and min(dens["Dhaka"], dens["Paris"], dens["Cairo"]) > 5000 and max(dens["Sahara"], dens["Siberia"]) < 5,
      f"total {tot / 1e9:.2f} bn; residents per km2: " + ", ".join(f"{k} {v:.0f}" for k, v in dens.items()))

print(f"\n{NFAIL} FAIL")
