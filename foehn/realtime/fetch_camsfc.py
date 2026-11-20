"""The CAMS operational forecast of PM2.5 (00 UTC run of day d, the ten leads 12 .. 120 h; `--two` for 24 and 120 h only)
for the checks and the daily verification. Output: data/<date>/cams_fc.npz (pm25 (n, 451, 900) ug m-3, lat, lon, leads)."""
import sys, tempfile, zipfile
import numpy as np
import xarray as xr
import common as C
d = C.parse_date(sys.argv[1]); D = C.day_dir(d)
LEADS = [24, 120] if "--two" in sys.argv else [12 * k for k in range(1, 11)]
if (D / "cams_fc.npz").exists() and len(np.load(D / "cams_fc.npz")["leads"]) >= len(LEADS):
    sys.exit(0)
from fetch import ads, nc_files
tmp = D / "cams_fc.zip"
ads().retrieve("cams-global-atmospheric-composition-forecasts",
               {"type": ["forecast"], "leadtime_hour": [str(L) for L in LEADS], "date": [f"{d}/{d}"], "time": ["00:00"],
                "variable": ["particulate_matter_2.5um"], "data_format": "netcdf", "download_format": "zip"}, str(tmp))
if True:
    f = nc_files(tmp)[0]
    with xr.open_dataset(f) as ds:
        v = ds["pm2p5"].squeeze(drop=True)
        tdim = [x for x in v.dims if x not in ("latitude", "longitude")][0]
        a = v.transpose(tdim, "latitude", "longitude").values.astype(np.float32) * 1e9
        np.savez(D / "cams_fc.npz", pm25=a.astype(np.float16), lat=ds["latitude"].values, lon=ds["longitude"].values, leads=np.array(LEADS))
C.log(d, f"CAMS forecast 00 UTC leads {LEADS[0]}..{LEADS[-1]} h ({len(LEADS)}): cams_fc.npz, mean {a[0].mean():.2f} / {a[-1].mean():.2f} ug m-3")
