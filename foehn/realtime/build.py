#!/usr/bin/env python3
"""RT step 2: the inputs of start day d in the exact format of training, data/<yyyy-mm-dd>/:
  cams.npy    (2, 13, 361, 900) float16  the CAMS analysis at t0 - 12 h (index 0) and t0 = 00 UTC of d (index 1), as two
              rows of cams_series.npy: build_cams_series.NAMES, rows 84 N .. 60 S, longitudes -180 .. 179.6, columns in
              mg m-2, clipped to [0, 65000]
  hres_B.npy  (33, 53, 181, 360) float16  the IFS 00 UTC run of d at steps 0..192 h, as one day of wb2/hres_B_<y>_<m>.npy:
              dl_wb2.CHANNELS, units of dl_wb2.V3/V2 (z / 9806.65 -> km, q x 1000 -> g/kg, t - 273.15, sp / 100, the 6-h
              precipitation ending at the step from the run's accumulation, NaN at step 0), 1 deg box means (dl_wb2.to_1deg)
  fires.npy   (3, 576, 1440) float16  GFAS of day d - 1 as one row of fires.npy (build_transport.to025, then its log
              transform and standardisation)
With --from-archive the three files are copied from the training stores (cams_series.npy, hres_B, fires.npy): the
non-regression test of the chain on a day of 2022.
    python build.py 2026-10-10 [--from-archive]"""
import argparse
import datetime as dt
import importlib.util
import sys
import tempfile
import zipfile

import numpy as np

import common as C


def build_transport_module():
    """build_transport.py runs its command at import: with an unknown command it only defines its functions"""
    argv = sys.argv
    sys.argv = ["build_transport.py", "--import-only"]
    try:
        spec = importlib.util.spec_from_file_location("build_transport", C.GLOBAL1KM / "build_transport.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


def npy_row(path, i):
    """block i of a .npy store (one row of the training series)"""
    return np.array(np.load(path, mmap_mode="r")[i])


# ------------------------------------------------------------------------------------------------------- CAMS
def cams_from_zip(p):
    import xarray as xr
    from build_cams_series import COLS, NAMES, crop_roll
    from dl_cams_analysis import PL, SL
    scale = {short: sc for _, (short, sc) in {**SL, **PL}.items()}
    arrs, lat, lon = {}, None, None
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(p) as z:
        z.extractall(tmp)
        for f in sorted(C.Path(tmp).glob("*.nc")):
            with xr.open_dataset(f) as ds:
                for v in ds.data_vars:
                    if v in scale:
                        a = ds[v].squeeze(drop=True)
                        arrs[v] = a.transpose("latitude", "longitude").values.astype(np.float32) * scale[v]
                        lat, lon = ds["latitude"].values, ds["longitude"].values
    missing = [n for n in NAMES if n not in arrs]
    assert not missing, f"{p.name}: missing {missing}"
    block = np.stack([arrs[n] for n in NAMES]) * COLS                                  # (13, 451, 900)
    return crop_roll(block, lat, lon)


def build_cams(d, D, archive):
    if archive:
        t0 = 2 * (d - C.CAMS0).days
        x = np.stack([npy_row(C.G / "cams_series.npy", t0 - 1), npy_row(C.G / "cams_series.npy", t0)])
    else:
        x = np.stack([cams_from_zip(D / "cams_12.zip"), cams_from_zip(D / "cams_00.zip")])
        x = np.clip(x, 0, 65000)
    x = x.astype(np.float16)
    np.save(D / "cams.npy", x)
    C.log(d, f"cams.npy {x.shape}: PM2.5 mean at t0 {float(x[1, 1].astype(np.float32).mean()):.2f} ug m-3, t0-12h "
             f"{float(x[0, 1].astype(np.float32).mean()):.2f}")


# ------------------------------------------------------------------------------------------------------- IFS
def read_grib_step(p):
    """{(param, level): (721, 1440) float32} on lat 90..-90, lon -180..179.75"""
    import eccodes
    out = {}
    with open(p, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                name = eccodes.codes_get(h, "shortName")
                lev = int(eccodes.codes_get(h, "level")) if eccodes.codes_get(h, "typeOfLevel") == "isobaricInhPa" else 0
                ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
                la0, lo0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
                jpos = eccodes.codes_get(h, "jScansPositively")
                v = eccodes.codes_get_values(h).reshape(nj, ni).astype(np.float32)
            finally:
                eccodes.codes_release(h)
            if jpos:
                v = v[::-1]; la0 = -la0
            assert (nj, ni) == (721, 1440) and abs(la0 - 90) < 1e-6, (name, nj, ni, la0)
            shift = int(round(((lo0 + 180.0) % 360.0) / 0.25))                       # columns to roll so that -180 is first
            out[(name, lev)] = np.roll(v, shift, axis=1) if shift else v
    return out


def build_hres(d, D, archive):
    from dl_wb2 import BSTEPS, CHANNELS, L8, V2, V3, to_1deg
    if archive:
        WB = C.G / "wb2"
        doys = list(np.load(WB / f"hres_B_{d.year}_{d.month:02d}_doy.npy"))
        x = npy_row(WB / f"hres_B_{d.year}_{d.month:02d}.npy", doys.index(C.doy_of(d)))
        np.save(D / "hres_B.npy", x)
        C.log(d, f"hres_B.npy {x.shape} copied from the archive")
        return
    gname = {"u_component_of_wind": "u", "v_component_of_wind": "v", "vertical_velocity": "w", "temperature": "t",
             "specific_humidity": "q", "geopotential": "z"}
    out = np.zeros((len(BSTEPS), len(CHANNELS), 181, 360), np.float16)
    tp_prev = None
    for si, s in enumerate(BSTEPS):
        f = read_grib_step(D / "ifs" / f"{s:03d}.grib2")
        for p_ in L8:                                                                 # geopotential height (gpm) -> geopotential, exact
            if ("z", p_) not in f and ("gh", p_) in f:
                f[("z", p_)] = f[("gh", p_)] * 9.80665
                if si == 0 and p_ == L8[0]:
                    C.log(d, "IFS open data of this date holds gh, not z: z = 9.80665 gh (exact conversion of the same field)")
        ch = [f[(gname[v], p)] * sc + off for v, sc, off in V3 for p in L8]
        tp = f[("tp", 0)]                                                             # m, accumulated since the run start
        tp6 = np.zeros_like(tp) if tp_prev is None else np.maximum(tp - tp_prev, 0)
        tp_prev = tp
        sfc = [f[("10u", 0)], f[("10v", 0)], f[("2t", 0)], f[("sp", 0)], tp6]
        ch += [a * sc + off for a, (_, sc, off) in zip(sfc, V2)]
        arr = to_1deg(np.stack(ch), True)
        if s == 0:
            arr[-1] = np.nan                                                          # no accumulation at step 0
        out[si] = arr.astype(np.float16)
    np.save(D / "hres_B.npy", out)
    C.log(d, f"hres_B.npy {out.shape}: u850 mean {float(out[0, 2].astype(np.float32).mean()):.2f} m/s, t2m mean "
             f"{float(out[0, 50].astype(np.float32).mean()):.2f} C, z500 mean {float(out[0, 45].astype(np.float32).mean()):.3f} km, "
             f"precipitation 0-6 h mean {float(np.nanmean(out[1, 52].astype(np.float32))):.3f} mm")


# ------------------------------------------------------------------------------------------------------- GFAS
def build_fires(d, D, archive):
    if archive:
        x = npy_row(C.G / "fires.npy", C.gi_of(d) - 1)
        np.save(D / "fires.npy", x)
        C.log(d, f"fires.npy {x.shape} copied from the archive (GFAS of {d - dt.timedelta(days=1)})")
        return
    import xarray as xr
    bt = build_transport_module()
    with xr.open_dataset(D / "gfas.nc") as ds:
        lat, lon = ds["latitude"].values, ds["longitude"].values
        assert abs(lat[0] - 89.95) < 1e-3 and abs(lat[-1] + 89.95) < 1e-3 and len(lon) == 3600, (lat[:2], lon[:2])
        inj = "mami" if "mami" in ds.data_vars else "injh"                  # v1.2: mami (training); v1.4.2: injh
        g = np.stack([ds[v].squeeze(drop=True).transpose("latitude", "longitude").values.astype(np.float32)
                      for v in ("tpmfire", "frpfire", inj)])
    if inj != "mami":
        C.log(d, "fires: injection height from GFAS injh (IS4FIRES); the training used mami (plume-rise model), which GFAS v1.4.2 "
                 "does not provide (decision 2026-10-10: injh, validated by the 2026 hindcast)")
    C.meta_update(d, fire_injection_variable=inj)
    g = bt.to025(g, float(lon[0]) - 0.05)                                             # tpm, frp, injection (as fires.npy)
    x = np.stack([np.log10(np.maximum(g[0], 0) + 1e-14), np.log10(np.maximum(g[1], 0) + 1e-6), np.log10(np.maximum(g[2], 0) + 1.0)])
    x = (x - np.array([-13.5, -5.5, 0.1])[:, None, None]) / np.array([1.0, 1.0, 0.6])[:, None, None]
    x = x.astype(np.float16)
    np.save(D / "fires.npy", x)
    C.log(d, f"fires.npy {x.shape}: {int((x[1] > -0.5).sum())} cells of 0.25 deg with FRP > 0.3 W m-2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--from-archive", action="store_true", help="copy the inputs from the training stores (test)")
    ap.add_argument("--only", nargs="*", default=["cams", "hres", "fires"])
    a = ap.parse_args()
    d = C.parse_date(a.date)
    D = C.day_dir(d)
    C.meta_update(d, date=str(d), gi=C.gi_of(d), year=d.year, doy=C.doy_of(d), archive=bool(a.from_archive))
    C.log(d, f"build {d}" + (" from the training archive" if a.from_archive else ""))
    if "cams" in a.only:
        build_cams(d, D, a.from_archive)
    if "hres" in a.only:
        build_hres(d, D, a.from_archive)
    if "fires" in a.only:
        build_fires(d, D, a.from_archive)
    C.log(d, "build done")


if __name__ == "__main__":
    main()
