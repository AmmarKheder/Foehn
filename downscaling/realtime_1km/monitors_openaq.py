#!/usr/bin/env python3
"""RT 1 km monitors of the verification (evaluation only: the monitors are never an input of RT and are
never used to train, choose or adjust it). Only RT's own errors are computed.

  catalogue  the OpenAQ monitors of the 2026 station set inside the GHAP mask (hindcast/2026/st_truth_2026.npz, QC of
             PROTOCOL.md A3(b)) minus data/processed/assim/qc_flipped_coords_v1.json, with their 1 km pixel and tile
             -> verif/catalogue.npz
  snapshot D the PUBLISHED 1 km forecast of issue day D (tag of web/runs/tiles_D.json): ensemble mean, p10, p90 of the
             leads 1..5 at the catalogue pixels -> verif/fc/D.npz, kept for good (the 1 km fields are deleted after 7 days)
  obs        OpenAQ public archive (s3://openaq-data-archive, no key; local-day files, about 4 days behind): the files of the
             catalogue locations for the open local days -> verif/raw/D.npz; UTC daily means with the QC of the station
             sets (coupling/build_st.py constants): 0 < PM <= 1000, location-hour means, runs >= 24 h of identical values
             removed (inside the 3-day window), day kept with >= 18 valid hours, location dropped if its recorded position
             moves (SD > 0.01 deg) or lies > 0.02 deg from its catalogue position -> verif/obs/D.npz
    python monitors_openaq.py catalogue | snapshot 2026-11-05 [--tag T] | obs [--today 2026-11-06]"""
import argparse
import datetime as dt
import gzip
import io
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

W = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = W / "realtime_1km"
V = O / "verif"
MASK = W / "data/processed/ghap/grid/ghap_mask.npy"
TRUTH = W / "hindcast/2026/st_truth_2026.npz"
PIX = W / "hindcast/2026/st_pix_2026.npz"
FLIPPED = W / "data/processed/assim/qc_flipped_coords_v1.json"
STATUS = W / "logs/RT1KM_STATUS.md"
sys.path.insert(0, str(W / "coupling"))
from build_st import MIN_HOURS, MOVE_SD, STUCK  # noqa: E402  (18 h, 0.01 deg, 24 h: the station-set QC)

BUCKET, PREFIX = "openaq-data-archive", "records/csv.gz/"
OPEN_DAYS = 7             # a local-day file is re-fetched until it is this many days old (late uploads to the archive)
FETCH_BACK = 10           # oldest local day fetched by a run (today - 10): verified days today-9 .. today-2
POS_TOL = 0.02            # deg: a day whose recorded position is farther from the catalogue position is dropped
MIN_N = 10                # rows need at least this many monitors
LEADS = 5
LAT_ROW0, LON_COL0, RES, TILE = -59.995, -179.995, 0.01, 1000      # GHAP-store lattice of infer.py (part<r>.json)
REGIONS = [("Europe", 35.0, 60.0, -10.0, 35.0), ("East Asia", 20.0, 50.0, 100.0, 135.0), ("South Asia", 5.0, 35.0, 60.0, 95.0),
           ("North America", 15.0, 50.0, -130.0, -60.0), ("Africa", -35.0, 35.0, -18.0, 52.0),
           ("South America", -40.0, 12.0, -82.0, -34.0), ("North", 60.0, 90.0, -180.0, 180.0)]   # = test2022_pmS/a5_common.py


def note(msg):
    with open(STATUS, "a") as f:
        f.write(f"- {dt.datetime.now():%Y-%m-%d %H:%M} [rt1km-verif] {msg}\n")


def region_of(lat, lon):
    r = np.full(np.shape(lat), "Other", dtype=object)
    for name, la0, la1, lo0, lo1 in REGIONS[::-1]:
        r[(lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)] = name
    return r


# --------------------------------------------------------------------------------------------------------------------
def catalogue():
    t = np.load(TRUTH)
    pos = np.load(PIX)["pos"].astype(np.int64)[t["pix"].astype(np.int64)]
    ids, net = t["id"].astype(str), t["network"].astype(str)
    lat, lon = t["lat"].astype(np.float64), t["lon"].astype(np.float64)
    flipped = set(json.loads(FLIPPED.read_text())["ids"]) if FLIPPED.exists() else set()
    keep = (net == "OpenAQ") & (pos >= 0) & ~np.isin(ids, sorted(flipped))
    ids, lat, lon, pos = ids[keep], lat[keep], lon[keep], pos[keep]
    flat = np.load(MASK, mmap_mode="r")[pos].astype(np.int64)
    i = 14399 - flat // 36000                                   # GHAP-store row from 59.995 S (flat rows start at 83.995 N)
    j = flat % 36000
    rr, cc = i // TILE, j // TILE
    row = TILE - 1 - (i - rr * TILE)                            # tiles are north-up
    col = j - cc * TILE
    loc = np.array([int(s.split(":")[1]) for s in ids], np.int64)
    np.savez(V / "catalogue.npz", id=ids, loc=loc, lat=lat.astype(np.float32), lon=lon.astype(np.float32), pos=pos,
             tile=np.array([f"r{a:02d}c{b:02d}" for a, b in zip(rr, cc)]), row=row.astype(np.int16), col=col.astype(np.int16),
             region=region_of(lat, lon).astype(str), source=str(TRUTH), flipped_removed=len(flipped & set(t["id"].astype(str))))
    print(f"[catalogue] {keep.sum():,} OpenAQ monitors inside the mask ({len(flipped)} flipped ids excluded) -> {V / 'catalogue.npz'}")


def load_cat():
    z = np.load(V / "catalogue.npz")
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------------------------------------------------
def published_tag(day):
    f = O / "web" / "runs" / f"tiles_{day}.json"
    if f.exists():
        return json.loads(f.read_text())["tag"]
    env = {}
    for line in (O / "web" / "system.env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1); env[k.strip()] = v.strip()
    return env.get("TAG", "pmA_final_K0T3")


def snapshot(day, tag=None):
    tag = tag or published_tag(day)
    cat = load_cat()
    src = O / "out" / day / tag
    n = cat["id"].size
    M = np.full((3, LEADS, n), np.nan, np.float32)
    # the tile lattice of THIS run (part0.json): GHAP store rows from 59.995 S, LGHAP (pmA, before 2026-11-03) from 57.995 S
    lat_row0 = float(json.loads((src / "part0.json").read_text())["lattice"]["lat_row0"])
    flat = np.load(MASK, mmap_mode="r")[cat["pos"]].astype(np.int64)
    i = np.rint((83.995 - RES * (flat // 36000) - lat_row0) / RES).astype(np.int64)
    j = flat % 36000
    rr, cc = i // TILE, j // TILE
    rows_, cols_ = TILE - 1 - (i - rr * TILE), j - cc * TILE
    tiles = np.array([f"r{a:02d}c{b:02d}" for a, b in zip(rr, cc)])
    by_tile = {}
    for k, t in enumerate(tiles.tolist()):
        if i[k] >= 0:
            by_tile.setdefault(t, []).append(k)
    by_tile = {t: np.array(v) for t, v in by_tile.items()}
    for L in range(1, LEADS + 1):
        files = sorted(src.glob(f"lead{L}_part*.npz"))
        if not files:
            raise FileNotFoundError(f"{src}/lead{L}_part*.npz")
        for f in files:
            z = np.load(f)
            for t in {k[5:] for k in z.files if k.startswith("mean/")} & set(by_tile):
                idx = by_tile[t]
                r, c = rows_[idx], cols_[idx]
                for p, prod in enumerate(("mean", "p10", "p90")):
                    M[p, L - 1, idx] = z[f"{prod}/{t}"][r, c].astype(np.float32)
    nan = int(np.isnan(M[0, 0]).sum())
    (V / "fc").mkdir(parents=True, exist_ok=True)
    np.savez(V / "fc" / f"{day}.npz", id=cat["id"], mean=M[0], p10=M[1], p90=M[2], tag=tag, day=day, lat_row0=lat_row0,
             created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    print(f"[snapshot] {day} ({tag}): {n:,} monitors x {LEADS} leads, {nan} without a value at lead 1 "
          f"(lead-1 mean over monitors {np.nanmean(M[0, 0]):.2f} ug m-3) -> verif/fc/{day}.npz")
    return nan, n


# --------------------------------------------------------------------------------------------------------------------
def s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", region_name="us-east-1",
                        config=Config(signature_version=UNSIGNED, max_pool_connections=64,
                                      retries={"max_attempts": 6, "mode": "adaptive"}))


def fetch_one(client, loc, day):
    """records (t UTC epoch s, value, lat, lon) of PM2.5 of one location's local-day file; None on a transient error"""
    from botocore.exceptions import ClientError
    key = f"{PREFIX}locationid={loc}/year={day.year}/month={day.month:02d}/location-{loc}-{day:%Y%m%d}.csv.gz"
    try:
        body = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return []
        return None
    except Exception:  # noqa: BLE001
        return None
    out = []
    try:
        with gzip.open(io.BytesIO(body), "rt", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("parameter", "").lower() not in ("pm25", "pm2.5"):
                    continue
                try:
                    ts = dt.datetime.fromisoformat(row["datetime"].replace("Z", "+00:00")).timestamp()
                    out.append((ts, float(row["value"]), float(row["lat"]), float(row["lon"])))
                except (ValueError, KeyError, TypeError):
                    continue
    except Exception:  # noqa: BLE001  (a corrupt file counts as transient)
        return None
    return out


def fetch_day(client, cat, day, workers=48):
    locs = cat["loc"].tolist()
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda l: fetch_one(client, l, day), locs))
    err = sum(r is None for r in res)
    L, T, X, LA, LO = [], [], [], [], []
    for loc, r in zip(locs, res):
        for ts, v, la, lo in (r or []):
            L.append(loc); T.append(ts); X.append(v); LA.append(la); LO.append(lo)
    np.savez(V / "raw" / f"{day}.npz", loc=np.array(L, np.int64), t=np.array(T, np.int64), v=np.array(X, np.float32),
             lat=np.array(LA, np.float32), lon=np.array(LO, np.float32), errors=err,
             fetched=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    n_loc = len({l for l, r in zip(locs, res) if r})
    print(f"[obs] local day {day}: {n_loc:,} locations with PM2.5, {len(X):,} records, {err} transient errors "
          f"({time.time() - t0:.0f} s)", flush=True)
    return err


def daily_means(cat, v):
    """UTC daily means of day v from the local-day files v-1, v, v+1 (QC above) -> (index into catalogue, value, hours)"""
    parts = [np.load(V / "raw" / f"{v + dt.timedelta(k)}.npz") for k in (-1, 0, 1)]
    loc = np.concatenate([p["loc"] for p in parts]); t = np.concatenate([p["t"] for p in parts])
    x = np.concatenate([p["v"] for p in parts]).astype(np.float64)
    la = np.concatenate([p["lat"] for p in parts]).astype(np.float64); lo = np.concatenate([p["lon"] for p in parts]).astype(np.float64)
    ci = {l: k for k, l in enumerate(cat["loc"].tolist())}
    if loc.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.uint8)
    t0 = int(dt.datetime(v.year, v.month, v.day, tzinfo=dt.timezone.utc).timestamp()) - 86400
    h = np.floor((t - t0) / 3600.0).astype(np.int64)                 # hour index in the 72-h window (as build_st.hourly)
    ok = np.isfinite(x) & (x > 0) & (x <= 1000) & (h >= 0) & (h < 72)
    loc, h, x, la, lo = loc[ok], h[ok], x[ok], la[ok], lo[ok]
    ids, inv = np.unique(loc, return_inverse=True)
    nl = ids.size
    cnt = np.bincount(inv, minlength=nl).astype(np.float64)
    mla, mlo = np.bincount(inv, la, nl) / cnt, np.bincount(inv, lo, nl) / cnt
    sla = np.sqrt(np.maximum(np.bincount(inv, la * la, nl) / cnt - mla ** 2, 0))
    slo = np.sqrt(np.maximum(np.bincount(inv, lo * lo, nl) / cnt - mlo ** 2, 0))
    key = inv * 72 + h
    uk, kinv = np.unique(key, return_inverse=True)
    hv = np.bincount(kinv, x) / np.bincount(kinv)                     # location-hour means
    li, hh = uk // 72, uk % 72
    same = np.zeros(uk.size, bool)
    same[1:] = (li[1:] == li[:-1]) & (hh[1:] == hh[:-1] + 1) & (hv[1:] == hv[:-1])
    run = np.cumsum(~same) - 1
    keep = np.bincount(run)[run] < STUCK
    li, hh, hv = li[keep], hh[keep], hv[keep]
    inday = (hh >= 24) & (hh < 48)
    n = np.bincount(li[inday], minlength=nl)
    s = np.bincount(li[inday], hv[inday], minlength=nl)
    k_cat = np.array([ci.get(int(l), -1) for l in ids])
    good = (n >= MIN_HOURS) & (sla <= MOVE_SD) & (slo <= MOVE_SD) & (k_cat >= 0)
    good &= np.abs(mla - cat["lat"][np.maximum(k_cat, 0)]) <= POS_TOL
    good &= np.abs(((mlo - cat["lon"][np.maximum(k_cat, 0)] + 180) % 360) - 180) <= POS_TOL
    return k_cat[good], (s[good] / n[good]).astype(np.float32), np.minimum(n[good], 24).astype(np.uint8)


def obs(today):
    cat = load_cat()
    (V / "raw").mkdir(parents=True, exist_ok=True); (V / "obs").mkdir(parents=True, exist_ok=True)
    client = s3()
    days = [today - dt.timedelta(k) for k in range(FETCH_BACK, 0, -1)]
    errs = {}
    for d in days:
        f = V / "raw" / f"{d}.npz"
        if f.exists():
            z = np.load(f)
            fetched = dt.datetime.fromisoformat(str(z["fetched"])).date()
            if (fetched - d).days >= OPEN_DAYS and int(z["errors"]) == 0:
                continue                                              # closed: fetched when already OPEN_DAYS old, no error
        errs[str(d)] = fetch_day(client, cat, d)
    made = []
    for v in days[1:-1]:
        if not all((V / "raw" / f"{v + dt.timedelta(k)}.npz").exists() for k in (-1, 0, 1)):
            continue
        k, x, nh = daily_means(cat, v)
        final = all((dt.datetime.fromisoformat(str(np.load(V / "raw" / f"{v + dt.timedelta(j)}.npz")["fetched"])).date()
                     - (v + dt.timedelta(j))).days >= OPEN_DAYS for j in (-1, 0, 1))
        np.savez(V / "obs" / f"{v}.npz", id=cat["id"][k], value=x, nhours=nh, final=final)
        made.append(f"{v}:{x.size}")
    print(f"[obs] UTC daily means (day:monitors): {' '.join(made)}", flush=True)
    return errs, made


# --------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["catalogue", "snapshot", "obs"])
    ap.add_argument("day", nargs="?")
    ap.add_argument("--tag")
    ap.add_argument("--today")
    a = ap.parse_args()
    V.mkdir(parents=True, exist_ok=True)
    if a.cmd == "catalogue":
        catalogue()
    elif a.cmd == "snapshot":
        snapshot(a.day, a.tag)
    else:
        obs(dt.date.fromisoformat(a.today) if a.today else dt.datetime.now(dt.timezone.utc).date())


if __name__ == "__main__":
    main()
