#!/usr/bin/env python3
"""Fields of the architecture overview (fig_gc1_schematic.py), at display resolution, in data/arch_fields.npz.
Start day 20 January 2022 (0-based day of year 19): GHAP PM2.5 at 1 km and at 0.25 deg, GFAS fires and the CAMS-GLOB-ANT black
carbon of January 2022 (0.25 deg), the HRES run of that day (850 hPa wind and 2 m temperature at 00 UTC, 1 deg), the transport
maps of lead 5 computed from the stored FLEXGREEN origins (advected PM2.5 = mean start-day PM2.5 at the origins of the air in
the mixed layer, weighted by survival; share of the surviving air that was above the mixed layer; survival), and Foehn's
1 km forecasts from that start day (member 0, leads 1-7, eval_jan21_leads_crps_e01) averaged to 0.1 deg for the discs.
    python figdata_arch_fields.py [static|leads|all]      (world venv, compute node: reads 1 km global tiles)"""
import os
import sys
from pathlib import Path

import numpy as np

from fig1_teaser_v6 import tiles
import final_model as FM

HERE = Path(__file__).resolve().parent
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
D0, LEAD_T = 19, 5
OUT = HERE / "data" / f"arch_fields{FM.SFX}.npz"                     # A16 rebuild: arch_fields_<D>.npz under FOEHN_DESIGN


def block_mean(a, k):
    ny, nx = a.shape[0] // k, a.shape[1] // k
    b = a[:ny * k, :nx * k].reshape(ny, k, nx, k)
    with np.errstate(all="ignore"):
        return np.nanmean(b, axis=(1, 3))


def origins_block(day, lead):
    """(288, 720, P, 4) float32 origins (lon, lat, z / mixed height, survival) of the 0.5 deg receptors, one start day and lead"""
    f = G / "green" / "hres" / "origins_2022.npy"
    with open(f, "rb") as fh:
        ver = np.lib.format.read_magic(fh)
        shape, _, dtype = (np.lib.format.read_array_header_1_0(fh) if ver == (1, 0) else np.lib.format.read_array_header_2_0(fh))
        hdr = fh.tell()
    per = int(np.prod(shape[2:])); off = hdr + ((day * shape[1] + lead) * per) * 2
    fd = os.open(f, os.O_RDONLY)
    a = np.frombuffer(os.pread(fd, per * 2, off), np.float16).reshape(shape[2:]).astype(np.float32)
    os.close(fd)
    return a


def static():
    land1 = tiles("static1km_tiles.npy", 1) > 0
    g = tiles("ghap1km_tiles.npy", 1826 + D0).astype(np.float32); g[(g <= 0) | ~land1] = np.nan
    pm025 = block_mean(g, 25)                                            # (576, 1440), 84 N .. 60 S
    pm005 = g[2::5, 2::5].astype(np.float16)                              # (2880, 7200) 1 km pixels every 5 km (texture kept)
    r0, c0 = int((84 - 30.6) * 100), int((76.0 + 180) * 100)             # 1 km lens: Delhi and the upper Ganges plain
    lens = g[r0:r0 + 400, c0:c0 + 400].astype(np.float16)
    land025 = np.load(G / "meta.npz")["land"].astype(bool)
    fires = np.asarray(np.load(G / "fires.npy", mmap_mode="r")[1826 + D0, 0], np.float32)
    emis = np.asarray(np.load(G / "emis_ant_fixed.npy", mmap_mode="r")[5 * 12 + 0, 0], np.float32)   # BC, January 2022
    hb = np.load(G / "wb2" / "hres_B_2022_01.npy", mmap_mode="r")
    u850, v850 = (np.asarray(hb[D0, 0, k], np.float32) for k in (2, 10))
    t2 = np.asarray(hb[D0, 0, 50], np.float32)
    out = dict(pm025=pm025.astype(np.float32), pm005=pm005, lens=lens, lens_box=np.array([30.6, 26.6, 76.0, 80.0]),
               land025=land025, fires=fires, emis=emis, u850=u850, v850=v850, t2=t2)
    out.update(tmaps(pm025))
    return out


def tmaps(pm025):
    land025 = np.load(G / "meta.npz")["land"].astype(bool)
    o = origins_block(D0, LEAD_T - 1)                                    # (288, 720, 32, 4)
    olon, olat, oz, osv = o[..., 0], o[..., 1], o[..., 2], o[..., 3]
    ii = np.clip(((84.0 - olat) / 0.25).astype(int), 0, 575); jj = ((olon + 180.0) / 0.25).astype(int) % 1440
    pm_at = pm025[ii, jj]
    w = osv * (oz <= 1.0) * np.isfinite(pm_at)
    with np.errstate(all="ignore"):
        adv = np.nansum(w * np.nan_to_num(pm_at), -1) / np.sum(w, -1)
        aloft = np.sum(osv * (oz > 1.0), -1) / np.sum(osv, -1)
        surv = np.mean(osv, -1)
    land05 = land025.reshape(288, 2, 720, 2).any((1, 3))
    tmaps = np.stack([np.where(land05, adv, np.nan), np.where(land05, aloft, np.nan), np.where(land05, surv, np.nan)]).astype(np.float32)
    return dict(tmaps=tmaps)


def leads():
    d = FM.RUN_DIR / FM.ev("eval_jan21_leads_crps_e01")
    land1 = tiles("static1km_tiles.npy", 1) > 0                           # the forecasts are shown over land, as in every figure
    m = []
    for h in range(1, 8):
        a = np.load(d / f"map1km_d{D0}_h{h}_member0.npy").astype(np.float32)
        a[~land1] = np.nan
        m.append(block_mean(a, 10).astype(np.float16))                    # 0.1 deg means of the 1 km member over land
    return dict(members=np.stack(m))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    old = dict(np.load(OUT)) if OUT.exists() else dict(np.load(HERE / "data" / "arch_fields.npz")) if FM.SFX else {}   # design: static fields of the stored file
    if what in ("static", "all"):
        old.update(static())
    if what == "tmaps":
        old.update(tmaps(old["pm025"]))
    if what in ("leads", "all"):
        old.update(leads())
    np.savez(OUT, **old)
    print("saved", OUT, {k: v.shape for k, v in old.items()})
