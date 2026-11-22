#!/usr/bin/env python3
"""Replay day files of experiment M (Amendment A11): for every issue day d of the 2026 record
(realtime_1km/replay/days_all.txt), the assimilated monitors of day d - 1 as a day file (dayfile.py), built from the
observations already stored for the verification of the forecasts. No network.

Sources (the rule of the replay scorer, realtime_1km/replay/score/methods.md, for the same monitors):
  monitor day <= 2026-10-21 : the 2026 station archive. Its assimilated part is data/processed/assim/assim_2026.npz
                              (the model-readable file: held-out monitors are absent by construction), so the
                              file of all monitors (hindcast/2026/st_truth_2026.npz) is NOT opened here.
  monitor day >= 2026-10-22 : realtime_1km/verif/obs/<day>.npz (the archive re-read with the same rules). That file holds
                              every catalogue monitor, held-out ones included: their rows are dropped BY IDENTIFIER
                              before any value is used, and the result is asserted free of held-out identifiers.
Monitor set of every file: the OpenAQ monitors of the frozen list that are in the verification catalogue (11,671 at
most): the same population on every day of the record and the one the live feed can deliver (the Chinese network and
the monitors outside the 1 km mask, present in assim_2026.npz, are not in the stored observations of the later days).
On the days that both sources hold (2026-10-20 .. 10-29) they are compared at the assimilated monitors (--check).

Also reported: for the held-out monitors of the catalogue (positions only, no value), how many have an assimilated
monitor WITH A VALUE THAT DAY within 10 km and within 50 km (great-circle), per file, and against the whole list.
    python build_replay.py [--check]      -> replay/stations_<d-1>.npz, replay/INDEX.json, reports/replay_day_files.md
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dayfile  # noqa: E402
import feed_openaq as fo  # noqa: E402

PROJECT = fo.PROJECT
DAYS_ALL = PROJECT / "realtime_1km/replay/days_all.txt"
VERIF_OBS = PROJECT / "realtime_1km/verif/obs"
CATALOGUE = PROJECT / "realtime_1km/verif/catalogue.npz"
ASSIM_2026 = PROJECT / "data/processed/assim/assim_2026.npz"
SWITCH = dt.date(2026, 10, 22)            # first monitor day read from verif/obs (the scorer's rule)
OUT = HERE / "replay"
R_KM = 6371.0
RULE = ("A11-M replay: UTC daily mean of the assimilated OpenAQ monitors of the verification catalogue on the day "
        "before the issue, from the stored verification observations (QC of A3(b): 0 < PM <= 1000, location-hour means, "
        "stuck runs >= 24 h removed, >= 18 valid hours); held-out ids of split_v1 removed (asserted)")


def unit(lat, lon):
    la, lo = np.deg2rad(np.asarray(lat, np.float64)), np.deg2rad(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def nearest_km(qlat, qlon, mlat, mlon):
    """great-circle distance (km) from each query point to its nearest monitor (inf when there is no monitor)"""
    if len(mlat) == 0:
        return np.full(len(qlat), np.inf)
    cosang = np.clip(unit(qlat, qlon) @ unit(mlat, mlon).T, -1.0, 1.0)
    return R_KM * np.arccos(cosang.max(1))


class Archive2026:
    """assimilated OpenAQ monitors of the 2026 station archive (assim_2026.npz), restricted to the list"""

    def __init__(self, lst):
        z = np.load(ASSIM_2026, allow_pickle=False)
        ids = z["id"].astype(str)
        dayfile.assert_no_heldout(ids, what=str(ASSIM_2026))
        ix = {s: k for k, s in enumerate(ids.tolist())}
        self.rows = np.array([ix[s] for s in lst.id.tolist()], np.int64)       # every listed monitor is in the archive
        self.daily = z["daily"]
        self.last_day = dt.date(2026, 1, 1) + dt.timedelta(days=int(np.nonzero(np.isfinite(self.daily[self.rows]).any(0))[0].max()))

    def day(self, lst, d, sel):
        v = self.daily[self.rows, (d - dt.date(2026, 1, 1)).days]
        ok = sel & np.isfinite(v) & (v > 0)
        return np.nonzero(ok)[0], v[ok].astype(np.float32), np.zeros(int(ok.sum()), np.uint8)


def verif_day(lst, d, sel):
    """assimilated monitors of verif/obs/<d>.npz -> (list index, value, valid hours, held-out rows dropped, final flag)"""
    held = dayfile.heldout_ids()
    with np.load(VERIF_OBS / f"{d}.npz", allow_pickle=False) as z:
        ids = z["id"].astype(str)
        keep = np.array([i not in held for i in ids.tolist()], bool)          # identifiers only
        rows = np.nonzero(keep)[0]
        value, nh = z["value"][rows], z["nhours"][rows]                       # held-out rows never leave the archive array
        final = bool(z["final"])
    ids = ids[rows]
    dayfile.assert_no_heldout(ids, what=f"verif/obs/{d}.npz after the removal")
    k = np.array([lst.index.get(int(s.split(":")[1]), -1) for s in ids.tolist()], np.int64)
    assert (k >= 0).all(), f"verif/obs/{d}.npz: {int((k < 0).sum())} monitors neither held out nor in the list"
    ok = sel[k] & np.isfinite(value) & (value > 0)
    return k[ok], value[ok].astype(np.float32), nh[ok].astype(np.uint8), int((~keep).sum()), final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare the two sources on the days both hold (assimilated monitors)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--api-snapshot", default=None, help="a work/<day>/locations.json of feed_openaq.py: counts, per file, the monitors "
                    "that the OpenAQ API does not serve (the live feed cannot deliver them); the files are not changed")
    a = ap.parse_args()
    out = Path(a.out)
    lst = fo.MonitorList()
    sel = lst.in_catalogue
    arch = Archive2026(lst)
    cat = np.load(CATALOGUE, allow_pickle=False)
    cid = cat["id"].astype(str)
    held = dayfile.heldout_ids()
    hm = np.array([i in held for i in cid.tolist()], bool)                    # held-out monitors of the catalogue: positions only
    hlat, hlon = cat["lat"][hm], cat["lon"][hm]
    assert not set(cid[hm].tolist()) & set(lst.id.tolist())
    static = {}
    cell = lambda la, lo: np.floor((np.asarray(la, np.float64) + 90.0) / 2.0).astype(np.int64) * 180 + np.floor((np.asarray(lo, np.float64) + 180.0) / 2.0).astype(np.int64)   # the 2-degree cells of split_v1
    for name, m in (("list_in_catalogue", sel), ("list_all", np.ones(sel.size, bool))):
        cosang = np.clip(unit(hlat, hlon) @ unit(lst.lat[m], lst.lon[m]).T, -1.0, 1.0)
        j = cosang.argmax(1)
        dkm = R_KM * np.arccos(cosang[np.arange(hlat.size), j])
        same = cell(hlat, hlon) == cell(lst.lat[m][j], lst.lon[m][j])     # the nearest assimilated monitor lies in the SAME cell:
        static[name] = dict(n_assimilated=int(m.sum()), heldout_within_10km=int((dkm <= 10).sum()), heldout_within_50km=int((dkm <= 50).sum()),
                            heldout_beyond_300km=int((dkm > 300).sum()), median_km=float(np.median(dkm)),
                            within_10km_nearest_in_same_2deg_cell=int(((dkm <= 10) & same).sum()),     # a monitor newer than the split, assimilated (A3(e))
                            within_10km_nearest_in_another_cell=int(((dkm <= 10) & ~same).sum()),      # across the border of the held-out cell
                            within_50km_nearest_in_same_2deg_cell=int(((dkm <= 50) & same).sum()),
                            assimilated_monitors_inside_heldout_cells=int(np.isin(cell(lst.lat[m], lst.lon[m]), np.unique(cell(hlat, hlon))).sum()))
    api_absent = None
    if a.api_snapshot:
        snap = json.loads(Path(a.api_snapshot).read_text())
        served = {int(k_) for k_ in snap["loc"]}
        api_absent = np.array([int(l) not in served for l in lst.loc.tolist()], bool)
    days = [dt.date.fromisoformat(x) for x in DAYS_ALL.read_text().split()]
    index = dict(written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), rule=RULE, days_file=str(DAYS_ALL),
                 n_issue_days=len(days), n_heldout_catalogue=int(hm.sum()), n_list=int(lst.id.size), n_list_in_catalogue=int(sel.sum()),
                 archive_last_day=str(arch.last_day), switch_to_verif_obs=str(SWITCH), static_neighbours=static, files=[],
                 **({} if api_absent is None else {"api_snapshot": dict(file=str(a.api_snapshot), taken=fo.iso(snap["t"]), list_monitors_not_served=int(api_absent.sum()),
                                                                     of_which_in_catalogue=int((api_absent & sel).sum()))}))
    for d in days:
        md = dayfile.monitor_day(d)
        if md < SWITCH:
            assert md <= arch.last_day
            k, v, nh = arch.day(lst, md, sel)
            src, dropped, final = "data/processed/assim/assim_2026.npz", 0, True
        else:
            k, v, nh, dropped, final = verif_day(lst, md, sel)
            src = f"realtime_1km/verif/obs/{md}.npz"
        meta = dict(issue_day=str(d), monitor_day=str(md), source=src, heldout_rows_dropped_from_source=dropped, source_final=final,
                    nhours_known=bool(md >= SWITCH), built=index["written"])
        p = dayfile.write_day_file(out, md, lst.id[k], lst.lat[k], lst.lon[k], v, nhours=nh, rule=RULE, meta=meta)
        f = dayfile.load_for_issue(out, d, valid_day=d, lag=1)                # loader + firewall + lag rule on the written file
        dkm = nearest_km(hlat, hlon, f["lat"], f["lon"])
        index["files"].append(dict(issue_day=str(d), monitor_day=str(md), file=p.name, n=int(f["id"].size), source=src,
                                   source_final=final, heldout_rows_dropped_from_source=dropped,
                                   heldout_within_10km=int((dkm <= 10).sum()), heldout_within_50km=int((dkm <= 50).sum()),
                                   heldout_beyond_300km=int((dkm > 300).sum()), value_median=float(np.median(f["value"])),
                                   **({} if api_absent is None else {"n_not_served_by_the_api": int(api_absent[k].sum())}),
                                   sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if a.check:
        chk = []
        d = SWITCH - dt.timedelta(days=5)
        while d <= arch.last_day:
            if (VERIF_OBS / f"{d}.npz").exists():
                ka, va, _ = arch.day(lst, d, sel)
                kv, vv, _, _, _ = verif_day(lst, d, sel)
                common, ia, iv = np.intersect1d(ka, kv, return_indices=True)
                diff = np.abs(va[ia] - vv[iv])
                chk.append(dict(day=str(d), archive=int(ka.size), verif_obs=int(kv.size), common=int(common.size),
                                identical=int((diff == 0).sum()), max_abs_diff=float(diff.max()) if diff.size else None))
            d += dt.timedelta(days=1)
        index["source_check"] = chk
    fo.atomic_write_text(out / "INDEX.json", json.dumps(index, indent=1))
    n = np.array([x["n"] for x in index["files"]])
    w10 = np.array([x["heldout_within_10km"] for x in index["files"]])
    w50 = np.array([x["heldout_within_50km"] for x in index["files"]])
    lines = ["# Replay day files of experiment M", "",
             f"{len(days)} issue days ({days[0]} .. {days[-1]}), one file `replay/stations_<issue day - 1>.npz` each; "
             f"monitors per file: min {n.min()}, median {int(np.median(n))}, max {n.max()} (of {int(sel.sum())} assimilated catalogue monitors).",
             f"Held-out monitors of the catalogue: {int(hm.sum())}. With an assimilated monitor that has a value that day within 10 km: "
             f"min {w10.min()}, median {int(np.median(w10))}, max {w10.max()}; within 50 km: min {w50.min()}, median {int(np.median(w50))}, max {w50.max()}.",
             f"Against every monitor of the list (whether or not it reports that day): within 10 km {static['list_in_catalogue']['heldout_within_10km']}, "
             f"within 50 km {static['list_in_catalogue']['heldout_within_50km']} (catalogue monitors of the list); "
             f"{static['list_all']['heldout_within_10km']} and {static['list_all']['heldout_within_50km']} (whole list, {static['list_all']['n_assimilated']} monitors).",
             f"Of the {static['list_in_catalogue']['heldout_within_10km']} held-out monitors with a listed monitor within 10 km, the nearest one lies in the same 2-degree cell for "
             f"{static['list_in_catalogue']['within_10km_nearest_in_same_2deg_cell']} (a monitor that did not exist in the 2017-2022 union of the split and is assimilated by the rule of A3(e)) "
             f"and in another cell for {static['list_in_catalogue']['within_10km_nearest_in_another_cell']} (across the border of the held-out cell); "
             f"{static['list_in_catalogue']['assimilated_monitors_inside_heldout_cells']} listed catalogue monitors lie inside the 2-degree cells of the held-out catalogue monitors.",
             *([] if api_absent is None else [f"OpenAQ API ({index['api_snapshot']['taken']}): {index['api_snapshot']['list_monitors_not_served']} monitors of the list are not served by "
                                              f"/v3/locations ({index['api_snapshot']['of_which_in_catalogue']} of them in the catalogue): the live feed cannot deliver them; per file "
                                              f"they are {min(x['n_not_served_by_the_api'] for x in index['files'])} to {max(x['n_not_served_by_the_api'] for x in index['files'])} of the monitors "
                                              f"(median {int(np.median([x['n_not_served_by_the_api'] for x in index['files']]))})."]), "",
             "| issue day | monitor day | monitors | held-out with a neighbour <= 10 km | <= 50 km | none within 300 km | source |", "|---|---|---|---|---|---|---|"]
    for x in index["files"]:
        lines.append(f"| {x['issue_day']} | {x['monitor_day']} | {x['n']} | {x['heldout_within_10km']} | {x['heldout_within_50km']} | "
                     f"{x['heldout_beyond_300km']} | {'archive (assim_2026.npz)' if 'assim_2026' in x['source'] else 'verif/obs' + ('' if x['source_final'] else ' (not final)')} |")
    if a.check:
        lines += ["", "Check of the two sources on the days both hold (assimilated catalogue monitors):", "",
                  "| day | archive | verif/obs | common | identical values | max abs difference |", "|---|---|---|---|---|---|"]
        lines += [f"| {c['day']} | {c['archive']} | {c['verif_obs']} | {c['common']} | {c['identical']} | {c['max_abs_diff']} |" for c in index["source_check"]]
    (HERE / "reports").mkdir(exist_ok=True)
    fo.atomic_write_text(HERE / "reports" / "replay_day_files.md", "\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    print(f"-> {out} ({len(index['files'])} files), {out / 'INDEX.json'}, reports/replay_day_files.md")


if __name__ == "__main__":
    main()
