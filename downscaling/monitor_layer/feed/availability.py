#!/usr/bin/env python3
"""At which hour after midnight UTC is the previous day's daily mean available? Direct measurement (experiment M feed).

  poll    every --every minutes for --hours hours: one pass of /v3/locations?parameters_id=2 (21 requests, paged by id)
          -> for every monitor of the frozen assimilated list, the time of its last measurement (no value is read:
          this endpoint carries none). All polls in ONE file availability/polls.npz (rewritten atomically).
  report  for every UTC day D whose end lies inside the polled period: the first poll at which a monitor's last
          measurement had reached the end of D (stamp >= D+1 00:00: every record of D is in; and stamp >= D 23:00, the
          last hourly record of the project's day) -> share of the monitors available by each hour after midnight.
          The times are those of the polls: an arrival is known to the polling interval.
The lag of a monitor at a poll (poll time - last stamp) is also summarised: it is the inference from timestamps
that can be made from a single pass.
    python availability.py poll --every 30 --hours 26 ; python availability.py report
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dayfile  # noqa: E402
import feed_openaq as fo  # noqa: E402

OUT = HERE / "availability"
UTC = dt.timezone.utc


def load_polls(path):
    if not Path(path).exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def cmd_poll(a):
    OUT.mkdir(parents=True, exist_ok=True)
    lst = fo.MonitorList()
    api = fo.OpenAQ(a.key_file, fo.RateLimiter(per_min=a.per_min, per_hour=a.per_hour, spacing=60.0 / a.per_min,
                                              state_file=OUT / "requests_last_hour.json"))
    path = OUT / "polls.npz"
    old = load_polls(path)
    T = [] if old is None else old["t_poll"].tolist()
    L = [] if old is None else [r for r in old["last"]]
    if old is not None:
        assert np.array_equal(old["loc"], lst.loc), "polls.npz was made with another list"
    t_end = time.time() + a.hours * 3600
    k = 0
    while True:
        t0 = time.time()
        try:
            snap = fo.snapshot_locations(api, lst)
        except fo.RateLimited as e:
            fo.log(f"[poll] STOPPED: {e}")
            return 3
        except fo.ApiError as e:
            fo.log(f"[poll] pass failed ({e}); next pass as scheduled")
            snap = None
        if snap is not None:
            last = np.full(lst.loc.size, np.nan)
            for lid, v in snap["loc"].items():
                last[lst.index[lid]] = v["last"]
            T.append(0.5 * (snap["t"] + snap["t_end"]))
            L.append(last)
            dayfile._atomic_savez(path, loc=lst.loc, t_poll=np.array(T, np.float64), last=np.array(L, np.float64))
            api.lim.save()
            lag = (T[-1] - last) / 3600.0
            live = lag <= 48
            fo.log(f"[poll {len(T)}] {snap['requests']} requests in {snap['t_end'] - snap['t']:.0f} s; {int(live.sum())} listed monitors "
                   f"reported within 48 h; lag of those: median {np.nanmedian(lag[live]):.2f} h, 90 % {np.nanpercentile(lag[live], 90):.2f} h, "
                   f"95 % {np.nanpercentile(lag[live], 95):.2f} h")
        k += 1
        nxt = t0 + a.every * 60.0
        if nxt >= t_end:
            break
        time.sleep(max(1.0, nxt - time.time()))
    return 0


def availability(t_poll, last, day, live_since_h=17):
    """-> dict for UTC day `day`: per threshold, hours after midnight (end of the day) at which each monitor was first
    seen complete; monitors = those that reported at least to `day` 17:00 at some poll (they can give >= 18 hours)"""
    end = dt.datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() + 86400
    seen = np.nanmax(last, axis=0)
    cand = seen >= end - (24 - live_since_h) * 3600
    out = {"day": str(day), "n_candidates": int(cand.sum()), "polls_after_midnight": int((t_poll >= end).sum()),
           "first_poll_after_midnight_h": None, "last_poll_h": float((t_poll.max() - end) / 3600), "thresholds": {}}
    after = t_poll >= end - 2 * 3600
    if after.any():
        out["first_poll_after_midnight_h"] = float((t_poll[t_poll >= end].min() - end) / 3600) if (t_poll >= end).any() else None
    for name, thr in (("all records of the day in (last stamp >= next day 00:00)", end),
                      ("last hourly record of the project's day in (last stamp >= 23:00)", end - 3600)):
        ok = last >= thr                                                     # (polls, monitors)
        first = np.where(ok.any(0), t_poll[np.argmax(ok, axis=0)], np.inf)   # first poll that saw it
        h = (first[cand] - end) / 3600.0
        out["thresholds"][name] = {"hours_after_midnight": {str(x): float(np.mean(h <= x)) for x in (0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15)},
                                   "never_in_the_polled_period": float(np.mean(~np.isfinite(h))),
                                   "percentile_h": {str(q): (float(np.percentile(h, q)) if np.isfinite(np.percentile(h, q)) else None) for q in (50, 75, 90, 95)}}
    return out


def cmd_report(a):
    z = load_polls(OUT / "polls.npz")
    if z is None:
        raise SystemExit("no polls yet")
    t_poll, last = z["t_poll"], z["last"]
    lines = [f"polls: {t_poll.size}, {dt.datetime.fromtimestamp(t_poll.min(), UTC):%Y-%m-%d %H:%M} .. "
             f"{dt.datetime.fromtimestamp(t_poll.max(), UTC):%Y-%m-%d %H:%M} UTC; listed monitors {last.shape[1]}"]
    res = {"polls": int(t_poll.size), "t_first": float(t_poll.min()), "t_last": float(t_poll.max()), "lag": [], "days": []}
    for k in range(t_poll.size):
        lag = (t_poll[k] - last[k]) / 3600.0
        live = lag <= 48
        row = dict(t=dt.datetime.fromtimestamp(t_poll[k], UTC).strftime("%Y-%m-%d %H:%M"), live=int(live.sum()),
                   **{f"p{q}": float(np.nanpercentile(lag[live], q)) for q in (50, 75, 90, 95, 99)},
                   **{f"within_{x}h": float(np.mean(lag[live] <= x)) for x in (1, 2, 3, 4, 6, 10)})
        res["lag"].append(row)
    lines.append("lag at each poll (monitors that reported within 48 h): time, n, median / 75 / 90 / 95 % lag (h), share within 2 / 4 / 6 / 10 h")
    for r in res["lag"][:: max(1, len(res["lag"]) // 30)]:
        lines.append(f"  {r['t']}  n {r['live']:5d}  {r['p50']:.2f} / {r['p75']:.2f} / {r['p90']:.2f} / {r['p95']:.2f}   "
                     f"{r['within_2h']:.3f} / {r['within_4h']:.3f} / {r['within_6h']:.3f} / {r['within_10h']:.3f}")
    d0 = dt.datetime.fromtimestamp(t_poll.min(), UTC).date()
    d1 = dt.datetime.fromtimestamp(t_poll.max(), UTC).date()
    day = d0
    while day < d1:
        end = dt.datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() + 86400
        if t_poll.min() <= end <= t_poll.max():
            r = availability(t_poll, last, day)
            res["days"].append(r)
            lines.append(f"UTC day {day}: {r['n_candidates']} monitors reported to 17:00 or later; polls reach {r['last_poll_h']:.1f} h after midnight")
            for name, v in r["thresholds"].items():
                lines.append(f"  {name}:")
                lines.append("    share available by hour after midnight: " + ", ".join(f"{h} h {s:.3f}" for h, s in v["hours_after_midnight"].items()))
                lines.append(f"    never in the polled period {v['never_in_the_polled_period']:.3f}; percentiles (h) {v['percentile_h']}")
        day += dt.timedelta(days=1)
    (HERE / "reports").mkdir(exist_ok=True)
    fo.atomic_write_text(HERE / "reports" / "availability.json", json.dumps(res, indent=1))
    fo.atomic_write_text(HERE / "reports" / "availability.txt", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poll")
    p.add_argument("--every", type=float, default=30.0, help="minutes between passes")
    p.add_argument("--hours", type=float, default=26.0)
    p.add_argument("--per-min", type=int, default=20)
    p.add_argument("--per-hour", type=int, default=120)
    p.add_argument("--key-file", default=None)
    sub.add_parser("report")
    a = ap.parse_args()
    return {"poll": cmd_poll, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
