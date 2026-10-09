#!/usr/bin/env python3
"""South Korea, 2022: AirKorea final confirmed hourly data (최종확정 측정자료, airkorea.or.kr, file 2022.zip, 12 monthly
xlsx: region, network, station code, name, KST time YYYYMMDDHH with HH = 01..24 (hour ending), PM25 ug/m3, -999 =
missing, address). Station coordinates: geocoded road-name addresses (OpenStreetMap Nominatim, 1 request / s, cached),
checked against the AirKorea stations of OpenAQ 2024-2025 (official coordinates, same station names).
  airkorea/korea_2022_hourly.npz   codes, names, networks, addresses, time (UTC s), pm25 (stations x hours)
  airkorea/geocode_cache.json
    source world/env.sh; PYTHONPATH=/projappl/project_462001140/envs/pylib_xlsx python build_korea.py"""
import glob
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import openpyxl

D = Path("/scratch/project_462001140/ammar/eccv/data/world/stations/airkorea")
T0 = np.datetime64("2022-01-01T00:00:00")

codes, info, recs = {}, {}, []
for f in sorted(glob.glob(str(D / "y2022/2022/*.xlsx"))):
    t = time.time()
    ws = openpyxl.load_workbook(f, read_only=True).worksheets[0]
    n = 0
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            assert row[2] == "측정소코드" and row[4] == "측정일시" and row[10] == "PM25", row
            continue
        code = str(row[2]); info.setdefault(code, (row[0], row[1], row[3], row[11]))
        k = codes.setdefault(code, len(codes))
        ts = str(row[4])                                   # YYYYMMDDHH, HH 01..24 = hour ending (KST)
        end = np.datetime64(f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T00:00") + np.timedelta64(int(ts[8:10]), "h")
        v = row[10]
        recs.append((k, int((end - np.timedelta64(30, "m") - np.timedelta64(9, "h") - T0) / np.timedelta64(1, "s")),
                     float(v) if v is not None and v != "" and float(v) >= 0 else np.nan))   # mid-hour, UTC
        n += 1
    print(f"{Path(f).name}: {n} rows, {time.time() - t:.0f}s", flush=True)
r = np.array(recs, np.float64)
sid = list(codes)
np.savez(D / "korea_2022_hourly.npz", codes=np.array(sid), names=np.array([info[c][2] for c in sid]),
         regions=np.array([info[c][0] for c in sid]), networks=np.array([info[c][1] for c in sid]),
         addresses=np.array([info[c][3] or "" for c in sid]), station=r[:, 0].astype(np.int32), time=r[:, 1].astype(np.int64),
         pm25=r[:, 2].astype(np.float32))
print(len(sid), "stations,", np.isfinite(r[:, 2]).sum(), "valid hourly PM2.5", flush=True)

cache_f = D / "geocode_cache.json"
cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
def geocode(q):
    if q in cache:
        return cache[q]
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"q": q, "format": "json", "limit": 1, "countrycodes": "kr"})
    req = urllib.request.Request(url, headers={"User-Agent": "pm25-station-geocoding-research/1.0"})
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=30)); break
        except Exception:
            time.sleep(5); r = []
    time.sleep(1.1)
    cache[q] = [float(r[0]["lat"]), float(r[0]["lon"]), r[0].get("type", ""), r[0].get("display_name", "")] if r else None
    cache_f.write_text(json.dumps(cache, ensure_ascii=False))
    return cache[q]
ok = 0
for c in sid:
    region, net, name, addr = info[c]
    addr = (addr or "").strip()
    g = geocode(addr) if addr else None
    if g is None and addr:                                  # drop building details after the street number
        parts = addr.split(); g = geocode(" ".join(parts[:4])) if len(parts) > 4 else None
    if g is None:
        g = geocode(f"{region} {name}")
    ok += g is not None
print(f"geocoded {ok} / {len(sid)}", flush=True)
