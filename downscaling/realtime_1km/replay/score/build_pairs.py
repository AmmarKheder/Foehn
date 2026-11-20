#!/usr/bin/env python3
"""WP1b step 1: the table of station-days (evaluation only; nothing is fitted, selected or corrected here).

One row per (issue day d, lead day L, monitor k) that has an observation on the valid UTC day v = d + L - 1.
Forecast values are read from the points files of the reforecast helper (replay/README.md); no forecast is recomputed.

Arms. An arm is given as NAME (array NAME of the main points directory) or NAME=DIR:KEY (array KEY of another points
directory in the same format, e.g. kmcal=/.../replay/points_cal:km), so that a second 1 km ensemble is scored on the same
station-days without a code change. Arrays (5, M, n) are ensembles (members kept), arrays (5, n) are deterministic.
  rings: km_ring1 / km_ring3 / km_ring10 are kept only where ring<d>_n >= 4 d (rule of the 2022 station test);
  km_cell_lin (mean of the 1 km members over a 5 x 5 sample of the monitor's 0.25 degree cell) only where cell_n >= 13.
Every arm is floored at 0 (as rc/coupling_live.py).

Observations (UTC daily means, station-set quality control: >= 18 valid hours, stuck and moving monitors removed):
  valid day <= 2026-10-21: hindcast/2026/st_truth_2026.npz (ids = catalogue ids);
  valid day >= 2026-10-22: realtime_1km/verif/obs/<day>.npz (the same rules applied to the archive by verif.py obs).
  The two sources are compared on every day that both hold (written to pairs_meta.json, key obs_overlap).

  python build_pairs.py --points R/points --days R/days_all.txt --out S/work [--arms ...]"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

W = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
V = W / "realtime_1km/verif"
TRUTH = W / "hindcast/2026/st_truth_2026.npz"
TRUTH_LAST = dt.date(2026, 10, 21)
D0 = dt.date(2026, 1, 1)
RULE = {"km_ring1": ("ring1_n", 4), "km_ring3": ("ring3_n", 12), "km_ring10": ("ring10_n", 40), "km_cell_lin": ("cell_n", 13)}
DEFAULT_ARMS = "km,km_ring1,km_ring3,km_ring10,km_cell_lin,rt04,cams,clim,bcsd,bcsd_m,an"


def parse_arms(spec, main):
    out = []
    for s in spec.split(","):
        s = s.strip()
        if not s:
            continue
        if "=" in s:
            name, rest = s.split("=", 1)
            d, key = rest.rsplit(":", 1) if ":" in rest else (rest, "km")
            out.append((name, Path(d), key))
        else:
            out.append((s, Path(main), s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--days", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = parse_arms(a.arms, a.points)
    days = [dt.date.fromisoformat(x) for x in Path(a.days).read_text().split()]
    cat = np.load(V / "catalogue.npz")
    cid = cat["id"].astype(str)
    n = cid.size
    cix = {s: i for i, s in enumerate(cid.tolist())}
    lat, lon = cat["lat"].astype(np.float64), cat["lon"].astype(np.float64)
    cell = (np.floor(lat / 2).astype(np.int64) + 90) * 1000 + (np.floor(lon / 2).astype(np.int64) + 180)   # 2-degree cell, as rc/*.py

    # ---- observations -------------------------------------------------------------------------------------------
    tr = np.load(TRUTH)
    tix = {s: i for i, s in enumerate(tr["id"].astype(str).tolist())}
    trow = np.array([tix[s] for s in cid.tolist()], np.int64)
    tdaily = tr["daily"][trow]                                       # (n, 365)
    valid_days = sorted({d + dt.timedelta(L) for d in days for L in range(5)})

    def obs_truth(v):
        return tdaily[:, (v - D0).days].astype(np.float32)

    def obs_verif(v):
        f = V / "obs" / f"{v}.npz"
        if not f.exists():
            return None, None
        z = np.load(f)
        y = np.full(n, np.nan, np.float32)
        k = np.array([cix.get(s, -1) for s in z["id"].astype(str).tolist()], np.int64)
        y[k[k >= 0]] = z["value"][k >= 0]
        return y, bool(z["final"])

    obs, src, overlap, per_day = {}, {}, {}, {}
    for v in valid_days:
        yt = obs_truth(v) if (v - D0).days < tdaily.shape[1] else np.full(n, np.nan, np.float32)
        yv, fin = obs_verif(v)
        if yv is not None and np.isfinite(yt).any():
            both = np.isfinite(yt) & np.isfinite(yv)
            overlap[str(v)] = dict(truth=int(np.isfinite(yt).sum()), archive=int(np.isfinite(yv).sum()), both=int(both.sum()),
                                   identical=int((yt[both] == yv[both]).sum()),
                                   max_abs_diff=float(np.abs(yt[both] - yv[both]).max()) if both.any() else None)
        if v <= TRUTH_LAST:
            obs[v], src[v], fin = yt, 0, True
        elif yv is not None:
            obs[v], src[v] = yv, 1
        else:
            obs[v], src[v], fin = np.full(n, np.nan, np.float32), -1, None
        per_day[str(v)] = dict(n=int(np.isfinite(obs[v]).sum()), source=["truth file", "archive (verif.py obs)", "none"][src[v]], final=fin)

    # ---- forecasts ----------------------------------------------------------------------------------------------
    cols = {"issue": [], "lead": [], "mon": [], "vday": [], "obs": [], "src": []}
    vals = {name: [] for name, _, _ in arms}
    kinds, present = {}, {}
    for i, d in enumerate(days):
        zs = {}
        for name, pdir, key in arms:
            f = pdir / f"{d}.npz"
            if pdir not in zs:
                zs[pdir] = np.load(f) if f.exists() else None
        zmain = zs[Path(a.points)]
        assert zmain is not None and (zmain["id"].astype(str) == cid).all(), f"{d}: monitor order differs from the catalogue"
        arr = {}
        for name, pdir, key in arms:
            z = zs[pdir]
            if z is None or key not in z.files:
                arr[name] = None
                continue
            assert (z["id"].astype(str) == cid).all()
            x = z[key].astype(np.float32)
            if key in RULE:
                cnt, mn = RULE[key]
                x = np.where((z[cnt] >= mn)[None, None, :], x, np.nan)
            arr[name] = np.maximum(x, 0.0)                              # NaN stays NaN
            kinds[name] = "ens" if x.ndim == 3 else "det"
        present[str(d)] = {}
        for L in range(1, 6):
            v = d + dt.timedelta(L - 1)
            y = obs[v]
            k = np.flatnonzero(np.isfinite(y))
            if k.size == 0:
                continue
            cols["issue"].append(np.full(k.size, i, np.int16)); cols["lead"].append(np.full(k.size, L, np.int8))
            cols["mon"].append(k.astype(np.int32)); cols["vday"].append(np.full(k.size, (v - D0).days, np.int16))
            cols["obs"].append(y[k]); cols["src"].append(np.full(k.size, src[v], np.int8))
            pres = []
            for name, _, _ in arms:
                x = arr[name]
                if x is None:
                    vals[name].append(None)
                    continue
                xl = x[L - 1][..., k]                                   # (M, k) or (k,)
                vals[name].append(xl.T.copy() if xl.ndim == 2 else xl.copy())
                if np.isfinite(xl).any():
                    pres.append(name)
            present[str(d)][str(L)] = pres
        print(f"{d}: rows so far {sum(c.size for c in cols['obs']):,} ({time.time() - t0:.0f} s)", flush=True)
    save = {k: np.concatenate(v) for k, v in cols.items()}
    N = save["obs"].size
    sizes = [c.size for c in cols["obs"]]
    M = 8
    for name, _, _ in arms:
        if name not in kinds:
            print(f"arm {name}: no file holds it, skipped")
            continue
        parts = []
        for x, s in zip(vals[name], sizes):
            if x is None:
                x = np.full((s, M) if kinds[name] == "ens" else s, np.nan, np.float32)
            parts.append(x)
        X = np.concatenate(parts)
        if kinds[name] == "ens":
            save[f"e_{name}"] = X
            save[f"m_{name}"] = X.mean(1)
        else:
            save[f"m_{name}"] = X
    np.savez(out / "pairs.npz", issue_days=np.array([str(d) for d in days]), arms=np.array([nm for nm, _, _ in arms if nm in kinds]),
             kinds=np.array([kinds[nm] for nm, _, _ in arms if nm in kinds]), id=cid, lat=cat["lat"], lon=cat["lon"], region=cat["region"],
             cell=cell, **save)
    meta = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), n_station_days=int(N), n_issue_days=len(days),
                arms={nm: dict(dir=str(p), key=k, kind=kinds.get(nm)) for nm, p, k in arms},
                obs_rule=f"valid day <= {TRUTH_LAST}: {TRUTH}; later: {V}/obs/<day>.npz", obs_per_valid_day=per_day, obs_overlap=overlap,
                arms_present=present)
    (out / "pairs_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{N:,} station-days, {np.unique(save['mon']).size:,} monitors, {len(days)} issue days -> {out / 'pairs.npz'} ({time.time() - t0:.0f} s)")
    for nm in save:
        if nm.startswith("m_"):
            print(f"  {nm[2:]:12s} finite on {int(np.isfinite(save[nm]).sum()):,} station-days")


if __name__ == "__main__":
    main()
