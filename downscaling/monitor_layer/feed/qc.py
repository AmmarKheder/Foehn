"""Quality rules of the verification observations (realtime_1km/verif/verif.py daily_means, constants of
coupling/build_st.py) applied to HOURLY rows of the OpenAQ v3 API instead of the archive's raw records.

The archive stores one record per measurement, stamped at the END of its averaging period, and the station sets of
the project (training, assimilation, verification) put a record into the UTC hour floor(stamp / 3600). The "UTC day"
of an hourly monitor is therefore its 24 hourly means stamped 00:00 .. 23:00, i.e. the period 23:00 of the day before
to 23:00 (a fact of every station file of the project, kept here so that the feed means what the training inputs mean).

An hourly row of the API (/v3/sensors/{id}/hours) is the mean of the records stamped in (datetimeFrom, datetimeTo]:
  - a row made of ONE record: its hour is floor(stamp of that record), the stamp being coverage.datetimeTo
    (= period.datetimeTo for an hourly monitor): identical to the archive rule;
  - a row made of several records (sub-hourly sensors): the hour that contains the middle of the period, i.e. the hour
    of datetimeFrom for whole-hour time zones; the record stamped exactly on the closing hour falls, in the archive
    rule, into the next hour: one record of the hour differs (measured in the test, see README);
  - rows of the same location and hour (two sensors, or a boundary record) are averaged with the number of records as
    weight, as the archive rule pools the records of a location-hour.
Value rule on the hourly mean: 0 < PM <= 1000 (the archive rule applies it record by record: identical for hourly
monitors). Stuck runs (>= 24 consecutive hours with identical values) are removed inside the 72-hour window day-1 ..
day+1 (as verif.py); a UTC day needs >= 18 valid hours.
"""
import datetime as dt

import numpy as np

MIN_HOURS = 18            # coupling/build_st.py MIN_HOURS (verif.py imports it)
STUCK = 24                # coupling/build_st.py STUCK
VMAX = 1000.0             # 0 < PM <= 1000 ug m-3
POS_TOL = 0.02            # verif.py POS_TOL (deg): recorded position vs list position
WINDOW_H = 72             # day - 1, day, day + 1


def day_epoch(day):
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp())


def hour_index(t_from, t_to, cov_to, count):
    """epoch hour of each row under the archive rule hour = floor(record stamp / 3600)"""
    t_from, t_to = np.asarray(t_from, np.float64), np.asarray(t_to, np.float64)
    cov_to, count = np.asarray(cov_to, np.float64), np.asarray(count, np.float64)
    single = ~(count > 1)
    stamp_single = np.where(np.isfinite(cov_to) & (cov_to > t_from) & (cov_to <= t_to), cov_to, t_to)
    stamp = np.where(single, stamp_single, 0.5 * (t_from + t_to))
    return np.floor(stamp / 3600.0).astype(np.int64)


def daily_means(loc, hour, value, weight, day):
    """rows (location id, epoch hour, hourly mean, number of records) -> (location ids, UTC daily mean float32,
    valid hours uint8) of `day` for the locations with >= MIN_HOURS valid hours; rules of verif.daily_means"""
    loc = np.asarray(loc, np.int64)
    x = np.asarray(value, np.float64)
    w = np.asarray(weight, np.float64)
    w = np.where(np.isfinite(w) & (w >= 1), w, 1.0)
    h = np.asarray(hour, np.int64) - (day_epoch(day) // 3600 - 24)          # hour index in the 72-h window
    ok = np.isfinite(x) & (x > 0) & (x <= VMAX) & (h >= 0) & (h < WINDOW_H)
    loc, h, x, w = loc[ok], h[ok], x[ok], w[ok]
    if loc.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.uint8)
    ids, inv = np.unique(loc, return_inverse=True)
    nl = ids.size
    key = inv * WINDOW_H + h
    uk, kinv = np.unique(key, return_inverse=True)
    hv = np.bincount(kinv, x * w) / np.bincount(kinv, w)                    # location-hour means
    li, hh = uk // WINDOW_H, uk % WINDOW_H
    same = np.zeros(uk.size, bool)
    same[1:] = (li[1:] == li[:-1]) & (hh[1:] == hh[:-1] + 1) & (hv[1:] == hv[:-1])
    run = np.cumsum(~same) - 1
    keep = np.bincount(run)[run] < STUCK
    li, hh, hv = li[keep], hh[keep], hv[keep]
    inday = (hh >= 24) & (hh < 48)
    n = np.bincount(li[inday], minlength=nl)
    s = np.bincount(li[inday], hv[inday], minlength=nl)
    good = n >= MIN_HOURS
    return ids[good], (s[good] / n[good]).astype(np.float32), np.minimum(n[good], 24).astype(np.uint8)


def position_ok(lat, lon, ref_lat, ref_lon, tol=POS_TOL):
    """recorded position within tol degrees of the list position (verif.py rule); NaN recorded position -> False"""
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    dla = np.abs(lat - np.asarray(ref_lat, np.float64))
    dlo = np.abs(((lon - np.asarray(ref_lon, np.float64) + 180.0) % 360.0) - 180.0)
    return np.isfinite(dla) & np.isfinite(dlo) & (dla <= tol) & (dlo <= tol)
