#!/usr/bin/env python3
"""Monitor inputs of the final test (draft Addendum 8 of A11), CPU, numpy only, no network.

For every issue day d of the record (ft_lib.issue_days(), 48 days) and two arms:
  arm 1 (primary)      the UTC daily means of day d - 1 (the day before the issue): lead day L reads them at lag L;
  arm 2 (sensitivity)  those of day d - 2 (surely complete at the issue): lead day L reads them at lag L + 1 (L <= 4;
                       lead day 5 would need lag 6, outside M's training: the station-free member is used there).
Monitors: the ASSIMILATED monitors by the place rule of Addendum 3 = the listed OpenAQ monitors of the verification
catalogue (a11/stageM/feed/list, the population of the feed's replay day files) that lie OUTSIDE every held-out 2-degree
cell of split_v1 (and whose id is not a held-out id: asserted, none). Values: the feed's replay day files
a11/stageM/feed/replay/stations_<day>.npz (archive assim_2026 to 2026-10-21, verif/obs after, QC of A3(b), built by
feed/build_replay.py); the one source day without such a file (2026-10-08, arm 2 of issue 2026-10-10) is read from
data/processed/assim/assim_2026.npz with the rule of build_replay.Archive2026.day (finite and > 0).
No held-out value is read: the 'value' member of a day file (and the 'daily' member of assim_2026.npz) is memory-mapped
and only the rows of assimilated-by-place monitors are gathered (ft_lib.gather_rows); the other members read are ids,
positions, dates. Every observation file used as a truth is behind the audit-hook firewall (ft_lib.TRUTH_MARKERS).
split_v1.json (data/processed/stations, firewalled afterwards) is read once at the start, key heldout_blocks only, and
checked against the SHA-256 published in split_v1_public.json and against the independent reconstruction of the cells.

Also written (positions only): for each catalogue monitor its place-rule flag, its id-rule flag, the distance from its
1 km pixel to the nearest assimilated monitor (any day: 'dstatic') and, per source day, to the nearest assimilated
monitor WITH a valid value that day ('dday': the quantity that sets the blend and the distance classes).

    python -I -B build_inputs.py [--feed-dir D] [--assim F] [--days d1,d2] [--out data/inputs.npz] [--trace out/x.json]
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_lib as F  # noqa: E402

ISSUE_HOUR_UTC = 11.0      # the daily chain starts at 11:00 UTC (daily.sh, --begin 11:00 UTC; CAMS 00 UTC step 0 ~10:00 UTC)


def place_rule():
    """the 188 held-out 2-degree cells of split_v1 (read before the firewall), with the checks"""
    pub = json.loads(F.SPLIT_PUBLIC.read_text())
    sha = F.sha256_file(F.SPLIT_FULL)
    assert sha == pub["split_v1_sha256"], "split_v1.json differs from the published SHA-256"
    cells = np.array(sorted(int(c) for c in json.loads(F.SPLIT_FULL.read_text())["heldout_blocks"]), np.int64)
    rev = np.array(sorted(json.loads(F.REVIEW_CELLS.read_text())["heldout_2deg_cells"]), np.int64)
    assert cells.size == 188 and np.array_equal(cells, rev), "held-out cells differ from the independent reconstruction"
    return cells, sorted(pub["heldout_ids"]), sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed-dir", default=str(F.FEED_REPLAY))
    ap.add_argument("--assim", default=str(F.ASSIM_2026))
    ap.add_argument("--days", default=None, help="issue days (default: the 48 of the record)")
    ap.add_argument("--out", default=str(F.INPUTS))
    ap.add_argument("--trace", default=None, help="write the list of opened files to this JSON")
    a = ap.parse_args()
    t0 = time.time()
    cells, held_ids, split_sha = place_rule()
    F.install_truth_firewall(trace=True)
    feed, assim = Path(a.feed_dir), Path(a.assim)
    days = F.issue_days() if a.days is None else sorted(a.days.split(","))
    assert set(days) <= set(F.issue_days()), "issue days outside the record"

    # ---- catalogue (positions only) ------------------------------------------------------------------------------
    cat = np.load(F.CATALOGUE)
    cid = cat["id"].astype(str)
    clat, clon, cpos = cat["lat"].astype(np.float64), cat["lon"].astype(np.float64), cat["pos"].astype(np.int64)
    n = cid.size
    ccell = F.cell2(clat, clon)
    held_place = np.isin(ccell, cells)
    held_id = np.isin(cid, held_ids)
    assert not (held_id & ~held_place).any(), "an id-held-out monitor outside the held-out cells"
    cplat, cplon = F.pixel_latlon(cpos)                      # query points of the inference: the monitors' 1 km pixels

    # ---- the assimilated monitors by place (positions only) -------------------------------------------------------
    L = np.load(F.FEED_LIST)
    lid, lat, lon, lpos, incat = (L["id"].astype(str), L["lat"].astype(np.float32), L["lon"].astype(np.float32),
                                  L["pos"].astype(np.int64), L["in_catalogue"].astype(bool))
    assert not np.isin(lid, held_ids).any(), "the feed list holds a held-out id"
    lcell = F.cell2(lat, lon)
    keep = incat & ~np.isin(lcell, cells)
    uid, ulat, ulon, upos = lid[keep], lat[keep], lon[keep], lpos[keep]
    A = uid.size
    cix = {s: i for i, s in enumerate(cid.tolist())}
    ucat = np.array([cix[s] for s in uid.tolist()], np.int64)
    assert np.array_equal(cpos[ucat], upos), "list pos differs from the catalogue pos"
    assert not held_place[ucat].any()
    plat, plon = F.pixel_latlon(upos)
    dpix = max(float(np.abs(plat - ulat).max()), float(np.abs(plon - ulon).max()))
    assert dpix <= 0.5 * F.GHAP_RES + 1e-4, f"pos is not the monitor's own pixel ({dpix})"
    order = np.argsort(upos, kind="stable")
    cm = np.load(F.GHAP_ROOT / "ghap_clim/clim_mean_G.npy", mmap_mode="r")
    uclim = np.zeros((A, 5), np.float32)
    uclim[order] = np.asarray(cm[upos[order]], np.float32)
    assert np.isfinite(uclim).all()
    uix = {s: i for i, s in enumerate(uid.tolist())}
    n_list_in_held = int((incat & np.isin(lcell, cells)).sum())

    # ---- source days ----------------------------------------------------------------------------------------------
    need = {}
    for d in days:
        dd = dt.date.fromisoformat(d)
        for k in (1, 2):
            need.setdefault(str(dd - dt.timedelta(days=k)), []).append((d, k))
    src_days = sorted(need)
    rows_all, vals_all, offs, kinds, files, rep = [], [], [0], [], [], {}
    for s in src_days:
        sd = dt.date.fromisoformat(s)
        f = feed / f"stations_{s}.npz"
        if f.exists():
            z = np.load(f)
            assert str(z["date"]) == s, (f, str(z["date"]))
            fid = z["id"].astype(str)
            r = np.array([k for k, x in enumerate(fid.tolist()) if x in uix], np.int64)          # sorted file rows
            assert not np.isin(fid, held_ids).any(), f"{f}: held-out id"
            v = F.gather_rows(f, "value", r)
            ui = np.array([uix[x] for x in fid[r].tolist()], np.int64)
            dll = max(float(np.abs(z["lat"][r] - ulat[ui]).max(initial=0)), float(np.abs(z["lon"][r] - ulon[ui]).max(initial=0)))
            assert dll < 1e-4, (f, dll)
            kind = "feed replay day file"
            n_rows, n_twin = int(fid.size), int(np.isin(F.cell2(z["lat"].astype(np.float64), z["lon"].astype(np.float64)), cells).sum())
        else:
            za = np.load(assim)
            aid = za["id"].astype(str)
            assert int(za["year"]) == sd.year if "year" in za.files else True
            aix = {x: k for k, x in enumerate(aid.tolist())}
            pairs = sorted((aix[x], i) for i, x in enumerate(uid.tolist()) if x in aix)
            r = np.array([p for p, _ in pairs], np.int64)
            ui = np.array([i for _, i in pairs], np.int64)
            col = (sd - dt.date(sd.year, 1, 1)).days
            v = F.gather_rows(assim, "daily", r, col=col)
            kind = f"assim_2026.npz column {col} (rule of feed/build_replay.Archive2026.day)"
            n_rows, n_twin = None, None
        ok = np.isfinite(v) & (v > 0)
        ui, v = ui[ok], v[ok]
        o = np.argsort(ui)
        rows_all.append(ui[o].astype(np.int32))
        vals_all.append(v[o].astype(np.float32))
        offs.append(offs[-1] + int(ok.sum()))
        kinds.append(kind)
        files.append(str(f if f.exists() else assim))
        rep[s] = dict(source=kind, file=files[-1], monitors=int(ok.sum()), file_rows=n_rows,
                      file_rows_inside_heldout_cells_dropped_unread=n_twin, used_by=need[s])
        assert sd < min(dt.date.fromisoformat(x) for x, _ in need[s]), "a source day not before its issue day"
    src_row, src_val = np.concatenate(rows_all), np.concatenate(vals_all)
    sidx = {s: i for i, s in enumerate(src_days)}
    issue_src = np.array([[sidx[str(dt.date.fromisoformat(d) - dt.timedelta(days=k))] for k in (1, 2)] for d in days], np.int64)

    # ---- distances (positions only) -------------------------------------------------------------------------------
    dstatic = F.nearest_km(cplat, cplon, ulat, ulon).astype(np.float32)
    dday = np.zeros((len(src_days), n), np.float32)
    for i in range(len(src_days)):
        rr = src_row[offs[i]:offs[i + 1]]
        dday[i] = F.nearest_km(cplat, cplon, ulat[rr], ulon[rr])
    av = json.loads(F.FEED_AVAIL.read_text())["days"][0]
    curve = av["thresholds"]["all records of the day in (last stamp >= next day 00:00)"]["hours_after_midnight"]
    hh = np.array([float(k) for k in curve]), np.array([float(v) for v in curve.values()])
    share = float(np.interp(ISSUE_HOUR_UTC, *hh))
    held_rows = np.flatnonzero(held_place)
    cls = {c[0]: [int((F.dist_class(dday[issue_src[i, 0], held_rows]) == j).sum()) for i in range(len(days))]
           for j, c in enumerate(F.DIST_CLASSES)}
    prov = dict(written_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), code_sha256=F.sha256_file(__file__),
                lib_sha256=F.sha256_file(F.__file__), split_v1_sha256=split_sha, feed_dir=str(feed), assim=str(assim),
                n_catalogue=n, n_heldout_place=int(held_place.sum()), n_heldout_id=int(held_id.sum()),
                n_list_in_catalogue=int(incat.sum()), n_list_in_catalogue_inside_heldout_cells=n_list_in_held,
                n_assimilated_by_place=A, max_pos_offset_deg=dpix,
                availability=dict(issue_hour_utc=ISSUE_HOUR_UTC, share_of_monitors_with_the_whole_previous_day_in=share,
                                  curve=curve, measured_day=av["day"], n_candidates=av["n_candidates"],
                                  note="feed/availability.py, one UTC day (2026-11-20) of the live OpenAQ API; the replay uses "
                                       "the archived daily means (an upper bound of what the live feed delivers at 11 UTC)"),
                heldout_place_by_distance_class_arm1=cls, source_days=rep, seconds=round(time.time() - t0, 1))
    F.save_npz(a.out, issue=np.array(days), issue_src=issue_src, src_day=np.array(src_days), src_off=np.array(offs, np.int64),
               src_row=src_row, src_val=src_val, src_kind=np.array(kinds), src_file=np.array(files),
               u_id=uid, u_lat=ulat, u_lon=ulon, u_pos=upos, u_clim=uclim, u_cat=ucat,
               cat_id=cid, cat_heldout_place=held_place, cat_heldout_id=held_id, cat_cell=ccell,
               cat_pixlat=cplat, cat_pixlon=cplon, dstatic=dstatic, dday=dday, heldout_cells=cells,
               prov=np.array(json.dumps(prov, default=F._jdef)))
    F.save_json(Path(a.out).with_suffix(".json"), prov)
    if a.trace:
        F.save_json(a.trace, dict(opened=F.opened_paths()))
    print(f"[build_inputs] {len(days)} issue days, {len(src_days)} source days {src_days[0]}..{src_days[-1]}; "
          f"{A:,} assimilated monitors by place ({n_list_in_held:,} listed catalogue monitors inside held-out cells dropped); "
          f"held-out by place {int(held_place.sum()):,} (by id {int(held_id.sum())}); monitors per source day "
          f"{min(np.diff(offs)):,}..{max(np.diff(offs)):,}; available at {ISSUE_HOUR_UTC:g} UTC: {share:.3f}; "
          f"{time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
