#!/usr/bin/env python3
"""Official coordinates of the AirKorea stations from OpenAQ (archive 2024, location name = AirKorea station name +
'-' + id): each AirKorea station of 2022 is matched by name to the OpenAQ location nearest to its geocoded address;
matches within 5 km replace the geocoded position. Prints the geocoding error distribution.
  airkorea/korea_coords.npz   codes, lat, lon, how (1 = OpenAQ official, 2 = geocoded), dist_km (geocoded vs official)"""
import gzip
import json
from pathlib import Path

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config

S = Path("/scratch/project_462001140/ammar/eccv/data/world/stations")
s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
names_f = S / "airkorea/openaq_korea_names.json"
if names_f.exists():
    oq = json.loads(names_f.read_text())
else:
    oq = {}
    for y in (2024, 2025):
        z = np.load(S / f"openaq/openaq_{y}_pm25.npz", allow_pickle=True)
        ids, first = np.unique(z["location_ids"], return_index=True)
        la, lo = z["lats"][first], z["lons"][first]
        for i, a, b in zip(ids, la, lo):
            if not (33 < a < 39 and 124 < b < 131.5) or str(i) in oq:
                continue
            r = s3.list_objects_v2(Bucket="openaq-data-archive", Prefix=f"records/csv.gz/locationid={int(i)}/year={y}/", MaxKeys=1)
            if not r.get("Contents"):
                continue
            line = gzip.decompress(s3.get_object(Bucket="openaq-data-archive", Key=r["Contents"][0]["Key"])["Body"].read()).decode().splitlines()[1]
            name = line.split('"')[1] if '"' in line else line.split(",")[2]
            oq[str(i)] = [name.rsplit("-", 1)[0], float(a), float(b)]
        print(y, len(oq), "Korean OpenAQ locations named", flush=True)
    names_f.write_text(json.dumps(oq, ensure_ascii=False))
k = np.load(S / "airkorea/korea_2022_hourly.npz")
cache = json.loads((S / "airkorea/geocode_cache.json").read_text())
def geo(i):
    addr, region, name = str(k["addresses"][i]).strip(), str(k["regions"][i]), str(k["names"][i])
    p = addr.split()
    for q in (addr, " ".join(p[:4]) if len(p) > 4 else None, f"{region} {name}"):
        if q and cache.get(q):
            return cache[q][0], cache[q][1]
    return np.nan, np.nan
by_name = {}
for i, (nm, a, b) in oq.items():
    by_name.setdefault(nm, []).append((a, b))
lat, lon, how, dist = [], [], [], []
for i in range(len(k["codes"])):
    ga, gb = geo(i)
    cand = by_name.get(str(k["names"][i]), [])
    best = None
    if cand and np.isfinite(ga):
        d = [111.2 * np.hypot(a - ga, (b - gb) * np.cos(np.deg2rad(ga))) for a, b in cand]
        j = int(np.argmin(d)); best = (cand[j], d[j])
    if best and best[1] < 5.0:
        lat.append(best[0][0]); lon.append(best[0][1]); how.append(1); dist.append(best[1])
    else:
        lat.append(ga); lon.append(gb); how.append(2 if np.isfinite(ga) else 0); dist.append(np.nan)
lat, lon, how, dist = map(np.array, (lat, lon, how, dist))
np.savez(S / "airkorea/korea_coords.npz", codes=k["codes"], lat=lat, lon=lon, how=how, dist_km=dist)
d = dist[how == 1]
print(f"{(how == 1).sum()} stations with official OpenAQ coordinates, {(how == 2).sum()} geocoded only, {(how == 0).sum()} without; "
      f"geocoding error vs official: median {np.median(d):.2f} km, 90% {np.quantile(d, .9):.2f} km, max {d.max():.2f} km")
