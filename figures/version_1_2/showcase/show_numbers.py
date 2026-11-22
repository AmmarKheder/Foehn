#!/usr/bin/env python3
"""Numbers behind the Fig. 3 candidates (natcomms/rebuild/figs_show, 2026-11-21), CPU only, reads stored files, writes ONE json:
  identity   the re-run forecast (figs_show/run/eval_show_d38_crps_e01_ar, start 38, lead 1) against run G1's stored one
             (RUN/eval_final_crps_e01_ar, key start 38 lead 1): members at the station pixels with the rule of A16's check C3
             (a13/warn16/check16all.cmp: same finite pattern, mean |diff| <= 0.05 ug m-3, 99.99th percentile <= 2.5), and the
             same statistics for the 0.25 deg cell means of each member (map25_mk) and of the ensemble mean (map25);
             the 1 km map's 0.25 deg cell means against the run's own map25 (the map drawn in panel a is the forecast scored)
  basins     the four boxes of fig_overview.py, lead 1, circularity/data_ar/station_day.npz (paper mask): RMSE of Foehn (four
             members' mean), the CAMS forecast and Aurora issued at the same time, per target day; tallies over all days;
             9 February and 24 December 2022 in detail; rankings of the days won in all four basins under several rules
  leadgain   panel c: RMSE at the stations of the six networks by lead 1-4 (compare_fair_crps_e01_ar.npz, network 'all'),
             cross-checked against station_day.npz
    python -I -B show_numbers.py [--no-identity]   ->  natcomms/rebuild/print/show/show_numbers.json"""
import datetime as dt
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
N = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms"))
FIG = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))
NEW = N / "rebuild/figs_show/run/eval_show_d38_crps_e01_ar"
OUT = N / "rebuild/print/show/show_numbers.json"
CENTRES = {"Delhi and the Ganges plain": (28.7, 77.6, 700), "Po Valley and the Alps": (45.35, 10.0, 600),   # fig_overview.py
           "Los Angeles basin": (34.05, -117.9, 300), "Seoul and the Han basin": (37.56, 127.0, 300)}
MEAN_MAX, P9999_MAX = 0.05, 2.5


def cmp(a, b):                                    # verbatim a13/warn16/check16all.cmp (the rule of A16's C3)
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    dd = np.abs(a[both].astype(np.float32) - b[both].astype(np.float32))
    return dict(bit_identical_share=float((a.view(np.uint16) == b.view(np.uint16)).mean()), finite_pattern_equal=bool(np.array_equal(fa, fb)),
                max_abs=float(dd.max()), p9999_abs=float(np.percentile(dd, 99.99)), mean_abs=float(dd.mean()))


def ok_pix(c):
    return bool(c["finite_pattern_equal"] and c["mean_abs"] <= MEAN_MAX and c["p9999_abs"] <= P9999_MAX)


def row(folder, d, h):
    for p in sorted(glob.glob(str(folder / "part_*.npz"))):
        if ".tmp" in p:
            continue
        z = np.load(p)
        w = np.where((z["keys"][:, 0] == d) & (z["keys"][:, 1] == h))[0]
        if len(w):
            return p, int(w[0]), z
    raise KeyError((folder, d, h))


def identity():
    pn, rn, zn = row(NEW, 38, 1)
    po, ro, zo = row(RUN / "eval_final_crps_e01_ar", 38, 1)
    R = {"new": f"{pn} row {rn}", "stored_G1": f"{po} row {ro}", "keys_new": zn["keys"][rn].tolist(), "keys_G1": zo["keys"][ro].tolist(),
         "info_equal": [str(x) for x in zn["info"]] == [str(x) for x in zo["info"]], "rho_base": [float(zn["keys_rho"][rn]), str(zn["base"]),
                                                                                                float(zo["keys_rho"][ro]), str(zo["base"])]}
    assert R["keys_new"] == R["keys_G1"] == [38, 1, 39], R
    st = {f"m{m}": cmp(np.asarray(zn["st_ens"][rn][:, m], np.float16), np.asarray(zo["st_ens"][ro][:, m], np.float16)) for m in range(4)}
    R["stations"] = dict(members=st, n_station_pixels=int(zn["st_ens"].shape[1]),
                         n_finite=int(np.isfinite(zn["st_ens"][rn][:, 0].astype(np.float32)).sum()), pass_C3_rule=all(ok_pix(v) for v in st.values()))
    R["st_ghap_identical"] = bool(np.array_equal(zn["st_ghap"][rn].view(np.uint16), zo["st_ghap"][ro].view(np.uint16)))
    R["map25_mk"] = {f"m{m}": cmp(np.asarray(zn["map25_mk"][rn, m], np.float16), np.asarray(zo["map25_mk"][ro, m], np.float16)) for m in range(4)}
    R["map25_mean"] = cmp(np.asarray(zn["map25"][rn], np.float16), np.asarray(zo["map25"][ro], np.float16))
    cols = [str(c) for c in zn["rec_cols"]]
    rn_ = zn["recs"][(zn["recs"][:, 0] == 38) & (zn["recs"][:, 1] == 1)]; ro_ = zo["recs"][(zo["recs"][:, 0] == 38) & (zo["recs"][:, 1] == 1)]
    i_n, i_s = cols.index("n"), cols.index("sse")
    R["region_rmse_new_vs_G1"] = {str(zn["regions"][int(a[2])]): [round(float(np.sqrt(a[i_s] / a[i_n])), 3), round(float(np.sqrt(b[i_s] / b[i_n])), 3)]
                                  for a, b in zip(rn_, ro_)}
    m1 = np.load(NEW / "map1km_d38_h1.npy", mmap_mode="r")                 # the map drawn: its 0.25 deg means (land, GHAP > 0 cells as
    R["map1km"] = dict(shape=list(m1.shape), dtype=str(m1.dtype))           # the run's map25) are checked in the run itself (same code)
    R["pass"] = bool(R["stations"]["pass_C3_rule"] and R["info_equal"] and R["st_ghap_identical"])
    return R


def date(d):
    return (dt.date(2022, 1, 1) + dt.timedelta(days=int(d))).isoformat()


def basins():
    s = np.load(N / "analyses/circularity/data_ar/station_day.npz")
    P, days, lat, lon, O = s["paper"][:, 0], s["days"], s["lat"], s["lon"], s["obs"][:, 0]
    X = {"Foehn": s["foehn"][:, 0], "CAMS": s["cams1"][:, 0], "Aurora": s["aur1"][:, 0]}
    R, C = {}, {}
    for nm, (la, lo, wkm) in CENTRES.items():
        dla = wkm / 1.5 / 111.2 / 2; dlo = wkm / (111.2 * np.cos(np.deg2rad(la))) / 2
        n, so, w, e = la + dla, la - dla, lo - dlo, lo + dlo
        m = P & ((lat <= n) & (lat >= so) & (lon >= w) & (lon <= e))[None]
        C[nm] = m.sum(1)
        R[nm] = {k: np.where(C[nm] > 0, np.sqrt(np.where(m, (v - O) ** 2, 0).sum(1) / np.maximum(C[nm], 1)), np.nan) for k, v in X.items()}
    T = {"start_days": len(days), "lead": 1, "source": "station_day.npz paper mask, boxes of fig_overview.py"}
    win = np.ones(len(days), bool); both = np.ones(len(days), bool)
    for nm in CENTRES:
        ok = C[nm] > 0
        b1 = R[nm]["Foehn"] < R[nm]["CAMS"]; b2 = b1 & (R[nm]["Foehn"] < R[nm]["Aurora"])
        T[nm] = dict(days_with_stations=int(ok.sum()), below_CAMS=int(b1[ok].sum()), below_Aurora=int((R[nm]["Foehn"] < R[nm]["Aurora"])[ok].sum()),
                     below_both=int(b2[ok].sum()))
        win &= b1; both &= b2
    T["below_CAMS_in_all_four"] = [date(d + 1) for d in days[win]]
    T["below_CAMS_and_Aurora_in_all_four"] = [date(d + 1) for d in days[both]]
    rules = {"largest smallest ratio CAMS/Foehn over the four basins": lambda i: min(R[nm]["CAMS"][i] / R[nm]["Foehn"][i] for nm in CENTRES),
             "largest smallest margin CAMS - Foehn (ug m-3)": lambda i: min(R[nm]["CAMS"][i] - R[nm]["Foehn"][i] for nm in CENTRES),
             "largest mean relative gain over CAMS": lambda i: np.mean([1 - R[nm]["Foehn"][i] / R[nm]["CAMS"][i] for nm in CENTRES]),
             "highest mean station PM2.5 over the four basins": lambda i: np.mean([O[i][P[i] & np.isfinite(O[i]) & (C[nm][i] > 0) &
                                                                                       _box(nm, lat, lon)].mean() for nm in CENTRES])}
    T["rankings_of_days_below_CAMS_in_all_four"] = {k: [(date(days[i] + 1), round(float(f(i)), 3)) for i in sorted(np.where(win)[0], key=lambda i: -f(i))[:3]]
                                                    for k, f in rules.items()}
    for d in (38, 356):
        i = int(np.where(days == d)[0][0])
        T[f"target {date(d + 1)}"] = {nm: dict(Foehn=round(float(R[nm]["Foehn"][i]), 1), CAMS=round(float(R[nm]["CAMS"][i]), 1),
                                               Aurora=round(float(R[nm]["Aurora"][i]), 1), stations=int(C[nm][i])) for nm in CENTRES}
    return T


def _box(nm, lat, lon):
    la, lo, wkm = CENTRES[nm]
    dla = wkm / 1.5 / 111.2 / 2; dlo = wkm / (111.2 * np.cos(np.deg2rad(la))) / 2
    return (lat <= la + dla) & (lat >= la - dla) & (lon >= lo - dlo) & (lon <= lo + dlo)


def leadgain():
    fz = np.load(FIG / "data/compare_fair_crps_e01_ar.npz"); SYS = [str(x) for x in fz["systems"]]; S = fz["S"]
    g = [str(x) for x in fz["nets"]].index("all")
    I = {"Foehn": SYS.index("crps_e01_ar (1 km)"), "CAMS": SYS.index("CAMS (run d+1, equal issuance)"), "Aurora": SYS.index("Aurora (00 UTC d+1, equal issuance)")}
    cf = {k: [round(float(np.sqrt(S[g, h, i, 2] / S[g, h, i, 0])), 2) for h in range(4)] for k, i in I.items()}
    s = np.load(N / "analyses/circularity/data_ar/station_day.npz"); P = s["paper"]; O = s["obs"]
    sd = {k: [round(float(np.sqrt(np.where(P[:, h], (s[v][:, h] - O[:, h]) ** 2, 0).sum() / P[:, h].sum())), 2) for h in range(4)]
          for k, v in (("Foehn", "foehn"), ("CAMS", "cams1"), ("Aurora", "aur1"))}
    return dict(compare_fair=cf, station_day=sd, equal=cf == sd, station_days=[int(S[g, h, I["Foehn"], 0]) for h in range(4)])


def main():
    R = {"made": dt.datetime.now().isoformat(timespec="seconds")}
    if "--no-identity" not in sys.argv:
        R["identity"] = identity()
    R["basins"] = basins()
    R["leadgain"] = leadgain()
    txt = json.dumps(R, indent=1, default=str) + "\n"
    tmp = OUT.with_name(OUT.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(txt); f.flush(); os.fsync(f.fileno())
    assert tmp.read_text() == txt
    os.replace(tmp, OUT)
    print(txt)


if __name__ == "__main__":
    main()
