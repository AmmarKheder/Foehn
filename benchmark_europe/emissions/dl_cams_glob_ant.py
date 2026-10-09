#!/usr/bin/env python3
"""CAMS-GLOB-ANT v6.2 anthropogenic emissions (ADS), monthly, 0.1 deg, all sectors, for the species that form or trace
PM2.5: black and organic carbon (primary), SO2, NOx, NH3 (secondary-aerosol precursors), CO (combustion tracer).
Output: data/world/emissions/CAMS-GLOB-ANT_v6.2_<species>_<year>.nc (one file per species-year).
    python dl_cams_glob_ant.py [years...]"""
import os
import sys
import zipfile
from pathlib import Path

import cdsapi

OUT = Path("/scratch/project_462001140/ammar/eccv/data/world/emissions")
SPECIES = ["black_carbon", "organic_carbon", "sulphur_dioxide", "nitrogen_oxides", "ammonia", "carbon_monoxide"]
YEARS = sys.argv[1:] or [str(y) for y in range(2017, 2023)]
cfg = {}
for line in open(os.path.expanduser("~/.adsapirc")):
    k, _, v = line.partition(":"); cfg[k.strip()] = v.strip()
c = cdsapi.Client(url=cfg["url"], key=cfg["key"], quiet=True, progress=False)
for y in YEARS:
    for sp in SPECIES:
        if list(OUT.glob(f"CAMS-GLOB-ANT_v6.2_{sp.replace('_', '-')}_{y}.nc")):
            print("exists", sp, y, flush=True); continue
        z = OUT / f"tmp_{sp}_{y}.zip"
        c.retrieve("cams-global-emission-inventories",
                   {"variable": [sp], "source": ["anthropogenic"], "version": ["v6.2"], "year": [y]}, str(z))
        with zipfile.ZipFile(z) as f:
            f.extractall(OUT); names = f.namelist()
        z.unlink()
        print("ok", sp, y, names, flush=True)
