#!/usr/bin/env python3
"""Source Data of the CURRENT manuscript (last pass, 2026-11-22), noise design 'ar'.

Imports source_data/build_source_data.py (never edited, never run as main here) with FOEHN_DESIGN=ar, so that every
export function reads the final model's '_ar' folders exactly as the builder does, then:
  * maps the figure files of today's manuscript to export functions:
      fig_overview.pdf        (Fig. 3, 9 February 2022)  -> fig3_feb   (lines of rebuild/figs_show/final3/fig_overview_final3.py)
      fig_gc2_realism.pdf     (Fig. 5, 24 January 2022)  -> fig5_jan24 (lines of rebuild/figs_show/final2/fig_gc2_realism_final2.py)
      figS_overview_21jan.pdf (Suppl. Fig. 8)            -> the builder's fig3 (the former Fig. 3, 21 January 2022)
      figS_realism_25jan.pdf  (Suppl. Fig. 11)           -> the builder's fig6 (the former Fig. 5, 25 January 2022)
  * Supplementary Tables: the builder's supp_tables logic, plus panel a of the table 'away' from
    analyses/circularity/results_ar/numbers.json (the stored away_a.json was hand-edited and is not used), the
    four-member tables named <sheet>_four_members (not 'b': the printed tables have no part b), and one provisional sheet
    for the Supplementary Note on opposite pairs and the noise-aware mean (a19/a20 score files) while its table is not in
    supplementary.tex;
  * writes source_data/Source_Data_final.xlsx, MANIFEST_final.md, MANIFEST_internal_final.md and, here, csv_final.zip
    (the CSV copies in one file: inode quota). Nothing else is written; Source_Data_ar.xlsx and csv_ar/ are not touched.
    FOEHN_DESIGN=ar OMP_NUM_THREADS=4 python -B build_final.py
"""
import datetime as dt
import glob
import importlib.util
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

assert os.environ.get("FOEHN_DESIGN") == "ar", "run with FOEHN_DESIGN=ar"
sys.dont_write_bytecode = True
HERE_SD = Path(__file__).resolve().parent
N = HERE_SD.parents[3]                                   # paper/natcomms (sd -> lastpass -> print -> rebuild -> natcomms)
assert (N / "work" / "supplementary.tex").exists(), N
_OUT = ("Source_Data_final.xlsx", "MANIFEST_final.md", "MANIFEST_internal_final.md")
_WB = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


def _ok(p):
    """writes allowed: this folder, /tmp, /dev, and the three outputs (and their temporary files) in source_data/"""
    a = os.path.realpath(os.fsdecode(p))
    if a == "/tmp" or a.startswith((str(HERE_SD) + "/", "/tmp/", "/dev/")):   # '/tmp' itself: openpyxl's O_TMPFILE
        return True
    d, f = os.path.split(a)
    return d == os.path.realpath(N / "source_data") and any(f == o or f.startswith(o + ".tmp") for o in _OUT)


def _hook(ev, args):                                     # write guard: any other write raises, nothing is written
    if ev == "open":
        p, mode, flags = args
        if p is None or isinstance(p, int):
            return
        wr = bool(flags & _WB) if isinstance(flags, int) else bool(mode and any(c in mode for c in "wax+"))
        if wr and not _ok(p):
            raise PermissionError(f"build_final write guard: write {p} refused")
    elif ev in ("os.rename", "os.replace") and not _ok(args[1]):
        raise PermissionError(f"build_final write guard: {ev} to {args[1]} refused")
    elif ev in ("os.remove", "os.rmdir", "os.mkdir", "shutil.rmtree") and isinstance(args[0], (str, bytes, os.PathLike)) and not _ok(args[0]):
        raise PermissionError(f"build_final write guard: {ev} {args[0]} refused")


sys.addaudithook(_hook)
spec = importlib.util.spec_from_file_location("bsd", N / "source_data" / "build_source_data.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
assert B.SFX == "_ar" and B.LABEL == "crps_e01_ar", (B.SFX, B.LABEL)
P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))   # A19/A20 scores (internal manifest only)
npz, jload, used, sheet, panel, sig, dec = B.npz, B.jload, B.used, B.sheet, B.panel, B.sig, B.dec
CENTRES = {"Delhi and the Ganges plain": (28.7, 77.6, 700), "Po Valley and the Alps": (45.35, 10.0, 600),
           "Los Angeles basin": (34.05, -117.9, 300), "Seoul and the Han basin": (37.56, 127.0, 300)}
SHORT = {"Delhi and the Ganges plain": "Delhi", "Po Valley and the Alps": "Po", "Los Angeles basin": "LA", "Seoul and the Han basin": "Seoul"}


# ================================================================================================ Fig. 3 (9 February 2022)
def fig3_feb(fg, sh):
    """rebuild/figs_show/final3/fig_overview_final3.py: b, RMSE on the day shown (day_pairing, L96-116 and L171-210) and the
    station circles (L124-126, L196-197); legend of b, yearly values (L199-202) and the tallies of print/show/verify_BC.json
    (L260-269); c-d, band_values (L290-315); e, bars_values (L318-378)."""
    scr = N / "rebuild" / "figs_show" / "final3" / "fig_overview_final3.py"
    D0, H = 38, 1; DT = D0 + H; ASPECT = 1.5
    fs, fc = B.G1 / "stations_2022.npz", B.DATA / f"overview_cache_{B.LABEL}.npz"
    st = npz(fs); off = st["source"] != 2
    obs = st["daily"][:, DT]; okst = np.isfinite(obs) & off
    slat, slon, sobs = st["lat"][okst], st["lon"][okst], obs[okst]
    la_all, lo_all = st["lat"], st["lon"]
    zc = npz(fc); SSE, NSD = zc["SSE"], zc["NSD"]
    # day_pairing (L96-116): Foehn = mean of the four members as run, CAMS = 00 UTC run of the target day read bilinearly,
    # GHAP at the station pixel; official networks where the measurement and the three fields are finite
    found = []
    for p in sorted(glob.glob(str(B.EVAL_DIR / "part_*.npz"))):
        if ".tmp" in p:
            continue
        z = np.load(p); w = np.where((z["keys"][:, 0] == D0) & (z["keys"][:, 1] == H))[0]
        found += [(p, int(k), z) for k in w]
    assert len(found) == 1, [r[:2] for r in found]
    p, n_, z = found[0]; used(p)
    t = int(z["keys"][n_, 2]); assert t == DT, (t, DT)
    cs = B.bilinear04(np.asarray(B.cams_daily()[t, 0, 0], np.float32), la_all, lo_all)
    XD = np.stack([z["st_ens"][n_].astype(np.float32).mean(-1), cs, z["st_ghap"][n_].astype(np.float32)]); OD = st["daily"][:, t]
    MD = np.isfinite(OD) & np.all(np.isfinite(XD), 0) & off
    vb = jload(N / "rebuild" / "print" / "show" / "verify_BC.json")["B_paper_mask"]
    names = {"cams": "CAMS forecast (40 km)", "ours": "Foehn (1 km)", "ghap": "GHAP (1 km estimates, not a forecast)"}
    rows, pts, leg = [], [], []
    for j, (nm, (la, lo, wkm)) in enumerate(CENTRES.items()):
        dla = wkm / ASPECT / 111.2 / 2; dlo = wkm / (111.2 * np.cos(np.deg2rad(la))) / 2
        n, s, w, e = la + dla, la - dla, lo - dlo, lo + dlo
        sel = (slat <= n) & (slat >= s) & (slon >= w) & (slon <= e)
        inbox = (la_all <= n) & (la_all >= s) & (lo_all >= w) & (lo_all <= e) & (NSD > 0) & off
        inday = (la_all <= n) & (la_all >= s) & (lo_all >= w) & (lo_all <= e) & MD
        for key in ("cams", "ours", "ghap"):
            k3 = {"ours": 0, "cams": 1, "ghap": 2}[key]
            rm = np.sqrt(((XD[k3, inday] - OD[inday]) ** 2).sum() / inday.sum())
            rows.append([j + 1, nm, names[key], sig(rm), f"{rm:.1f}", int(inday.sum()), int(sel.sum())])
            ry = np.sqrt(SSE[inbox, k3].sum() / NSD[inbox].sum())
            leg.append([j + 1, nm, f"RMSE over 2022 at the stations of the basin, 1 day ahead, {names[key]} (µg m-3)", sig(ry)])
        leg.append([j + 1, nm, "stations of the basin with station-days in 2022 (official networks)", int(inbox.sum())])
        b_ = vb[SHORT[nm]]
        leg.append([j + 1, nm, "start days of 2022 on which Foehn's RMSE at the basin's stations was below that of the CAMS forecast", int(b_["below_cams"])])
        leg.append([j + 1, nm, "start days of 2022 on which Foehn's RMSE at the basin's stations was above that of the CAMS forecast", int(b_["above_cams"])])
        for a, b, c in zip(slat[sel], slon[sel], sobs[sel]):
            pts.append([j + 1, nm, dec(a, 4), dec(b, 4), sig(c)])
    leg += [[0, "all four basins", "start days of 2022 scored in all four basins", int(vb["days_all_four_valid"])],
            [0, "all four basins", "start days on which Foehn's RMSE was below that of the CAMS forecast in all four basins", int(vb["below_cams_all_four"])],
            [0, "all four basins", "start days on which Foehn's RMSE was below those of the CAMS forecast and Aurora in all four basins", int(vb["below_both_all_four"])]]
    sheet(f"{sh}b", fg, "b",
          "Root-mean-square error (µg m-3) at the stations of each basin on 9 February 2022 (forecasts issued 1 day ahead: Foehn's mean of "
          "four members issued at the end of 8 February, the CAMS forecast of 00 UTC on 9 February read bilinearly, the 1 km estimates at "
          "the station pixel), on the stations where the measurement and the three fields exist: the numbers written in the twelve maps.",
          ["basin number", "basin", "system", "RMSE on 9 February 2022 (µg m-3)", "as printed", "stations scored", "stations drawn as circles"],
          rows, scr, [p, fs, B.WORLD / "cams_forecast_daily" / "2022.zarr"])
    sheet(f"{sh}b_stations", fg, "b",
          "Daily mean PM2.5 (µg m-3) measured on 9 February 2022 at the stations of the official networks drawn as circles in the maps "
          "of each basin (latitude and longitude in degrees).",
          ["basin number", "basin", "latitude (°N)", "longitude (°E)", "measured PM2.5 (µg m-3)"], pts, scr, [fs])
    sheet(f"{sh}b_legend", fg, "b",
          "Numbers of the legend of b: RMSE over 2022 at the stations of each basin, 1 day ahead, on identical station-days (µg m-3), and "
          "the number of start days of 2022 on which Foehn's error at the basin's stations was below or above that of the CAMS forecast "
          "(station-days of the paper's evaluation: official networks, the ten systems finite).",
          ["basin number", "basin", "quantity", "value"], leg, scr, [fc, N / "rebuild" / "print" / "show" / "verify_BC.json"])
    # c-d: band_values (L290-315), identical station-days of the three systems
    fz = npz(B.DATA / f"compare_fair_{B.LABEL}.npz"); SYS = [str(s) for s in fz["systems"]]; S = fz["S"]; nets = [str(s) for s in fz["nets"]]
    SYSN = {"Foehn (1 km)": f"{B.LABEL} (1 km)", "CAMS forecast (40 km)": "CAMS (run d+1, equal issuance)",
            "Aurora (40 km)": "Aurora (00 UTC d+1, equal issuance)"}
    G_ = {"c": ("Stations of the six networks", [nets.index("all")]), "d": ("China (CNEMC)", [nets.index("China (CNEMC)")])}
    for L, (pn, gg) in G_.items():
        rws = []
        for nm, sn in SYSN.items():
            for h in range(4):
                cnt = int(S[gg, h, SYS.index(sn), 0].sum())
                rws.append([pn, nm, h + 1, sig(np.sqrt(S[gg, h, SYS.index(sn), 2].sum() / cnt)), cnt, "yes" if h in (0, 3) else "no"])
        sheet(f"{sh}{L}", fg, L,
              f"Root-mean-square error (µg m-3) of the daily PM2.5 forecasts of 2022 by lead time (days) at the stations of "
              f"{'the six official networks' if L == 'c' else 'China (CNEMC)'}, on station-days shared by the three systems (Foehn's mean of "
              "four members; the CAMS forecast and Aurora issued at the same time); the dotted line is the CAMS forecast 1 day ahead.",
              ["stations", "system", "lead time (days)", "RMSE (µg m-3)", "station-days", "value written in the panel"], rws, scr,
              [B.DATA / f"compare_fair_{B.LABEL}.npz"])
    # e: bars_values (L318-378), compare_fair.py's bootstrap verbatim
    PD, PDN = fz["PD"], fz["PDN"]
    kF, kC, kA = (SYS.index(SYSN[k]) for k in ("Foehn (1 km)", "CAMS forecast (40 km)", "Aurora (40 km)"))
    assert kF == 0 and np.allclose(PDN[:, :6].sum(1), PDN[:, 6])
    rng = np.random.default_rng(0)
    ok_d = np.isfinite(PD[0, :, 0, 0]); nd, BL = int(ok_d.sum()), 7
    idx = (rng.integers(0, nd - BL + 1, size=(10000, -(-nd // BL)))[:, :, None] + np.arange(BL)).reshape(10000, -1)[:, :nd]

    def red(gl, h, k):
        a, b = PDN[kF][gl].sum(0)[ok_d, h - 1], PDN[k][gl].sum(0)[ok_d, h - 1]
        r = lambda ix: np.sqrt(np.nansum(a[ix, 0], 1) / np.nansum(a[ix, 1], 1)) / np.sqrt(np.nansum(b[ix, 0], 1) / np.nansum(b[ix, 1], 1)) - 1
        bs = r(idx); pt = np.sqrt(np.nansum(a[:, 0]) / np.nansum(a[:, 1])) / np.sqrt(np.nansum(b[:, 0]) / np.nansum(b[:, 1])) - 1
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        return dict(v=float(-100 * pt), lo=float(-100 * hi), hi=float(-100 * lo), rf=float(np.sqrt(np.nansum(a[:, 0]) / np.nansum(a[:, 1]))),
                    rr=float(np.sqrt(np.nansum(b[:, 0]) / np.nansum(b[:, 1]))), n=int(np.nansum(a[:, 1])))
    E = {net: red([g], 1, kC) for g, net in enumerate(nets[:6])}
    order = sorted(E, key=lambda k: -E[k]["v"])
    rws = [[net, sig(E[net]["v"]), sig(E[net]["lo"]), sig(E[net]["hi"]), E[net]["lo"] <= 0 <= E[net]["hi"], f"{E[net]['v']:.0f}",
            sig(E[net]["rf"]), sig(E[net]["rr"]), E[net]["n"]] for net in order]
    sheet(f"{sh}e", fg, "e",
          "Reduction of the root-mean-square error (%) of Foehn's mean of four members against the CAMS forecast issued at the same time, "
          "1 day ahead in 2022, by network, with the 95% interval (moving-block bootstrap over start days, blocks of 7, 10,000 resamples); "
          "bars in decreasing order; hatched where the interval includes zero.",
          ["network", "RMSE reduction (%)", "95% interval, low", "95% interval, high", "interval includes zero (hatched)", "value written above the bar",
           "RMSE of Foehn (µg m-3)", "RMSE of the CAMS forecast (µg m-3)", "station-days"], rws, scr, [B.DATA / f"compare_fair_{B.LABEL}.npz"])
    oc = [g for g, s_ in enumerate(nets[:6]) if s_ != "China (CNEMC)"]
    gC, gA = nets.index("China (CNEMC)"), nets.index("all")
    rws = []
    for h in range(1, 5):
        rws.append(["share of China (CNEMC) in the station-days (%)", h, sig(100 * S[gC, h - 1, kF, 0] / S[gA, h - 1, kF, 0]), None, None])
        rws.append(["share of China (CNEMC) in the reduction of the summed squared error against the CAMS forecast (%)", h,
                    sig(100 * (S[gC, h - 1, kC, 2] - S[gC, h - 1, kF, 2]) / (S[gA, h - 1, kC, 2] - S[gA, h - 1, kF, 2])), None, None])
        for gl, lab in ((oc, "the five networks outside China, pooled"), ([gC], "China (CNEMC)"), ([6], "the six networks")):
            for k, ref in ((kC, "the CAMS forecast"), (kA, "Aurora")):
                r_ = red(gl, h, k)
                rws.append([f"RMSE reduction (%) against {ref}, {lab}", h, sig(r_["v"]), sig(r_["lo"]), sig(r_["hi"])])
    sheet(f"{sh}e_legend", fg, "e",
          "Numbers of the legend of e by lead time (days): China's share of the station-days and of the reduction of the summed squared "
          "error against the CAMS forecast (%), and the RMSE reductions (%) of Foehn's mean of four members, pooled, with 95% intervals "
          "(the bootstrap of e).", ["quantity", "lead time (days)", "value", "95% interval, low", "95% interval, high"], rws, scr,
          [B.DATA / f"compare_fair_{B.LABEL}.npz"])
    panel(fg, "a", "world map of Foehn's 1 km forecast for 9 February 2022 (ensemble mean of four members), with the four basins of b", "map, not exported")
    panel(fg, "b", "four basins x (CAMS forecast, Foehn, 1 km estimates): maps with that day's station RMSE written in each and the stations as circles",
          f"maps not exported; printed numbers in {sh}b; station circles in {sh}b_stations; legend numbers in {sh}b_legend")
    panel(fg, "c", "RMSE by lead time at the stations of the six networks, three systems (lines, logarithmic axis)", f"{sh}c")
    panel(fg, "d", "RMSE by lead time at the stations of China (CNEMC), three systems (lines, logarithmic axis)", f"{sh}d")
    panel(fg, "e", "RMSE reduction against the CAMS forecast 1 day ahead by network (bars, 95% intervals)", f"{sh}e; legend numbers in {sh}e_legend")


# ================================================================================================ Fig. 5 (24 January 2022)
def fig5_jan24(fg, sh):
    """rebuild/figs_show/final2/fig_gc2_realism_final2.py (SHOW_DAY = 23): spectra h, p (L214-275: GHAP, median and range of the
    16 members, ensemble mean, CAMS, Aurora); u (L345-375); the box numbers of the legend (box_numbers, L152-191); the tallies of
    the legend (print/show/verify_A.json) and the RMSEs at the plain's stations (print/show/recheck_final2.json)."""
    scr = N / "rebuild" / "figs_show" / "final2" / "fig_gc2_realism_final2.py"
    AURD = B.EXP / "benchmark" / "aurora"
    SUB = (41.5, 35.5, 113.5, 121.5); TH = 75.0; LAT0 = 38.5
    peak = 23
    D = B.CASE_DIR
    c0 = npz(D / f"crop0_d{peak - 1}_h1.npz"); box = c0["box"]; (r0, r1), (cc0, cc1) = c0["rows"], c0["cols"]
    n, s, w, e = SUB
    sr0, sr1 = int(round((box[0] - n) * 100)), int(round((box[0] - s) * 100)); sc0, sc1 = int(round((w - box[2]) * 100)), int(round((e - box[2]) * 100))
    gh = B.ghap_box(1826 + peak, int(r0), int(r1), int(cc0), int(cc1))[sr0:sr1, sc0:sc1]; land = gh > 0; gh = np.where(land, gh, np.nan)
    cams = B.cams_daily()
    aur = {}
    for q in sorted((AURD / "out_12h_all").glob("part_*.npz")):
        zq = np.load(q)
        for k, d in enumerate(zq["days"]):
            aur[int(d)] = (q, k)

    def aur_field(d, h):                                  # L94-98
        q, k = aur[d + 1]; used(q); A = np.load(q)["pred"][k].astype(np.float32); k0 = 2 * (h - 1)
        first = (npz(AURD / "raw" / f"{d + 1:03d}.npz")["surf"][0, 5, :A.shape[1]].astype(np.float32) * 1e9) if h == 1 else A[k0 - 1]
        return 0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1]

    def sub40(field, lat0=90.0, dlat=0.4, lon0=0.0, dlon=0.4):       # L101-106 (the field only)
        lats = lat0 - dlat * np.arange(field.shape[0]); lons = (lon0 + dlon * np.arange(field.shape[1]) + 180) % 360 - 180
        ri = np.where((lats <= n + 0.2) & (lats >= s - 0.2))[0]; ci = np.where((lons >= w - 0.2) & (lons <= e + 0.2))[0]
        return field[np.ix_(ri, ci)]

    dx40 = 0.4 * 111 * np.cos(np.deg2rad(LAT0))
    files = [D / "crop0_d<start>_h<lead>.npz", B.G1 / "ghap1km_tiles.npy", B.WORLD / "cams_forecast_daily" / "2022.zarr",
             AURD / "out_12h_all" / "part_*.npz", AURD / "raw" / "<day of year>.npz"]
    for row, h in enumerate((1, 3)):
        m = npz(D / f"crop0_d{peak - h}_h{h}.npz")["members"].astype(np.float32)[:, sr0:sr1, sc0:sc1]
        m[:, ~land] = np.nan
        cf = sub40(np.asarray(cams[peak - h + 1, 0, h - 1], np.float32))
        af = sub40(aur_field(peak - h, h))
        Pm = []
        for k in range(m.shape[0]):
            lam1, P_ = B.spectrum(m[k], 1.0); Pm.append(P_)
        Pm = np.array(Pm)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med, lo, hi = np.nanmedian(Pm, 0), np.nanmin(Pm, 0), np.nanmax(Pm, 0)
        curves = [("GHAP (1 km estimates)",) + B.spectrum(gh, 1.0), ("Foehn, median of the 16 members", lam1, med),
                  ("Foehn, lowest of the 16 members (band)", lam1, lo), ("Foehn, highest of the 16 members (band)", lam1, hi),
                  ("Foehn, ensemble mean",) + B.spectrum(np.nanmean(m, 0), 1.0),
                  ("CAMS forecast (0.4°)",) + B.spectrum(cf, dx40), ("Aurora (0.4°)",) + B.spectrum(af, dx40)]
        rws = []
        for lab, lam, Pw in curves:
            for a, b in zip(lam, Pw):
                rws.append([lab, h, sig(a), sig(b), "yes" if (np.isfinite(b) and 2 <= a <= 600) else "no"])
        L = "hp"[row]
        sheet(f"{sh}{L}", fg, L,
              f"Variance per decade of wavelength ((µg m-3)^2) of each field of daily PM2.5 over the North China Plain panel "
              f"(35.5-41.5°N, 113.5-121.5°E) on 24 January 2022, forecasts issued {h} day{'s' if h > 1 else ''} ahead: radially averaged power "
              "spectrum by wavelength (km); for Foehn, the median and the lowest and highest of the spectra of its 16 members (line and band) "
              "and the spectrum of the ensemble mean; the 0.4° forecasts on their own grid.",
              ["field", "issued ahead (days)", "wavelength (km)", "variance per decade ((µg m-3)^2)", "inside the axis range 600-2 km"], rws, scr, files)
    # box numbers of the legend (box_numbers, L152-191)
    la = (n - (np.arange(gh.shape[0]) + 0.5) * 0.01)[:, None] * np.ones((1, gh.shape[1]))
    lo_ = (w + (np.arange(gh.shape[1]) + 0.5) * 0.01)[None, :] * np.ones((gh.shape[0], 1))
    la, lo_, g = la[land], lo_[land], gh[land].astype(np.float64); y = g > TH
    near = lambda f: np.asarray(f, np.float64)[np.rint((90.0 - la) / 0.4).astype(int), np.rint((lo_ % 360.0) / 0.4).astype(int) % 900]
    rmse = lambda v: float(np.sqrt(np.mean((v - g) ** 2)))
    rws = []
    for h in (1, 3, 5, 7):
        pn = {1: "a-g, h, t", 3: "i-o, p, s", 5: "r", 7: "q"}[h]
        ml = npz(D / f"crop0_d{peak - h}_h{h}.npz")["members"].astype(np.float32)[:, sr0:sr1, sc0:sc1][:, land].astype(np.float64)
        mu = ml.mean(0); bm = ml.mean(1); rk = [int(np.sum(bm > bm[k])) + 1 for k in range(ml.shape[0])]
        pr = (ml > TH).mean(0)
        cf = near(cams[peak - h + 1, 0, h - 1]) if h <= 5 else None
        af = near(aur_field(peak - h, h)) if h <= 5 else None
        Q = [("mean over the land of the panel, the 1 km estimates (GHAP)", g.mean(), "µg m-3"),
             ("mean over the land of the panel, Foehn's ensemble mean", mu.mean(), "µg m-3"),
             ("mean over the land of the panel, lowest of the 16 members", bm.min(), "µg m-3"),
             ("mean over the land of the panel, highest of the 16 members", bm.max(), "µg m-3")]
        Q += [(f"mean over the land of the panel, member {k + 1}", bm[k], "µg m-3") for k in range(3)]
        Q += [(f"rank of member {k + 1} among the 16 by that mean (1 = highest)", rk[k], "rank") for k in range(3)]
        Q += [("RMSE against GHAP over the land of the panel, Foehn's ensemble mean", rmse(mu), "µg m-3")]
        if cf is not None:
            Q += [("RMSE against GHAP over the land of the panel, CAMS forecast issued at the same time", rmse(cf), "µg m-3"),
                  ("RMSE against GHAP over the land of the panel, Aurora issued at the same time", rmse(af), "µg m-3")]
        Q += [("mean probability of exceeding 75 µg m-3 (share of the 16 members) where GHAP exceeds 75", pr[y].mean(), "probability"),
              ("mean probability of exceeding 75 µg m-3 (share of the 16 members) elsewhere on land", pr[~y].mean(), "probability")]
        for q, v, u in Q:
            rws.append([h, pn, q, v if u == "rank" else sig(v), u])
    rws.append([0, "contour of q-t", "share of the land of the panel where GHAP exceeds 75 µg m-3", sig(y.mean()), "fraction"])
    sheet(f"{sh}_box", fg, "a-g, i-o, q-t",
          "Summary numbers of the maps over the land of the North China Plain panel (35.5-41.5°N, 113.5-121.5°E; 1 km pixels where GHAP > 0) "
          "on 24 January 2022, by the lead time at which the forecasts were issued: means of the fields, ranks of members 1-3 (fixed "
          "indices, not selected), RMSE against the 1 km estimates (0.4° forecasts read in the cell drawn at each pixel) and mean "
          "probabilities of exceeding 75 µg m-3; the maps themselves are not exported.",
          ["issued ahead (days)", "panels", "quantity", "value", "unit"], rws, scr, files)
    # tallies of the legend over the 22 days and the RMSE at the plain's 137 official stations
    va = jload(N / "rebuild" / "print" / "show" / "verify_A.json"); rc = jload(N / "rebuild" / "print" / "show" / "recheck_final2.json")
    rws = []
    for r in va["rows"]:
        if r["h"] <= 5:
            rws.append([r["date"], int(r["h"]), sig(r["ghap_mean"]), sig(r["foehn_rmse"]), sig(r["cams_rmse"]), sig(r["aurora_rmse"])])
    sm = va["summary"]
    tal = [[f"lead {h} days: days of the 22 on which Foehn's RMSE is below both references", h, int(sm[f"h{h}"]["below_both"])] for h in range(1, 6)]
    tal += [[f"lead {h} days: days of the 22 on which Foehn's RMSE is the highest of the three", h, int(sm[f"h{h}"]["worst_of_three"])] for h in range(1, 6)]
    # the plain's official stations (box 36-41 N, 114-119.5 E), 24 January: lines of print/show/recheck_final2.py L454-465, here at
    # full precision (the check file stores two decimals); cross-checked against it
    sp = rc["E_fig5"]["stations_plain_24jan"]
    stp = npz(B.G1 / "stations_2022.npz"); ob = stp["daily"][:, peak]
    sl = (stp["source"] == 1) & (stp["lat"] >= 36) & (stp["lat"] <= 41) & (stp["lon"] >= 114) & (stp["lon"] <= 119.5) & np.isfinite(ob)
    pq = [np.load(q) for q in sorted(glob.glob(str(B.CASE_DIR / "part_*.npz")))]
    kq = np.concatenate([q["keys"] for q in pq]); eq = np.concatenate([q["st_ens"] for q in pq])
    kpos = {tuple(int(x) for x in k): i for i, k in enumerate(kq)}
    fi = (90 - stp["lat"][sl]) / 0.4; fj = (stp["lon"][sl] % 360) / 0.4; i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int); wi, wj = fi - i0, fj - j0
    bil = lambda f: (1 - wi) * ((1 - wj) * f[i0, j0] + wj * f[i0, j0 + 1]) + wi * ((1 - wj) * f[i0 + 1, j0] + wj * f[i0 + 1, j0 + 1])
    for h in (1, 3, 5):
        d_ = peak - h; F = eq[kpos[(d_, h, peak)]][sl].astype(np.float32).mean(1); o = ob[sl]
        C = bil(np.asarray(cams[d_ + 1, 0, h - 1], np.float64)); A = bil(aur_field(d_, h).astype(np.float64))
        ok = np.isfinite(F) & np.isfinite(C) & np.isfinite(A)
        for X, k, nm in ((F, "foehn", "Foehn's ensemble mean"), (C, "cams", "CAMS forecast"), (A, "aurora", "Aurora")):
            v = float(np.sqrt(np.mean((X - o)[ok] ** 2)))
            assert abs(v - sp[str(h)][k]) <= 0.0051 and int(ok.sum()) == sp[str(h)]["n"], (h, k, v, sp[str(h)])
            tal.append([f"RMSE at the {int(ok.sum())} official stations of the plain on 24 January 2022 (µg m-3), {nm}", h, sig(v, 5)])
    sheet(f"{sh}_days", fg, "legend",
          "Root-mean-square error (µg m-3) against the 1 km estimates over the land of the North China Plain panel, for each of the 22 days "
          "from 8 to 29 January 2022 and each lead time (days) at which the forecasts were issued: Foehn's ensemble mean of 16 members, the "
          "CAMS forecast and Aurora issued at the same time (0.4° fields read at the 1 km pixels); with the mean of the estimates over the "
          "plain (the basis of the choice of the day, Methods).",
          ["date", "issued ahead (days)", "mean of GHAP over the plain (µg m-3)", "RMSE Foehn (µg m-3)", "RMSE CAMS forecast (µg m-3)",
           "RMSE Aurora (µg m-3)"], rws, scr, [N / "rebuild" / "print" / "show" / "verify_A.json"])
    sheet(f"{sh}_legend", fg, "legend",
          "Numbers of the legend: tallies over the 22 days of Fig5_days, and the RMSE at the official stations of the plain on 24 January 2022 "
          "by the lead time at which the forecasts were issued (days).", ["quantity", "issued ahead (days)", "value"], tal,
          N / "rebuild" / "print" / "show" / "recheck_final2.py",
          [N / "rebuild" / "print" / "show" / "verify_A.json", N / "rebuild" / "print" / "show" / "recheck_final2.json", B.G1 / "stations_2022.npz",
           B.CASE_DIR / "part_*.npz", B.WORLD / "cams_forecast_daily" / "2022.zarr", AURD / "out_12h_all" / "part_*.npz"])
    # u: the monitors of central Beijing (L345-375; same code as the builder's fig6)
    fs = B.G1 / "stations_2022.npz"; st = npz(fs)
    sel = (st["source"] == 1) & (np.abs(st["lat"] - 39.9) < 0.3) & (np.abs(st["lon"] - 116.4) < 0.4)
    obs = np.nanmean(st["daily"][sel].astype(np.float32), 0)
    parts = [np.load(q) for q in sorted(glob.glob(str(B.CASE_DIR / "part_*.npz")))]
    kk = np.concatenate([q["keys"] for q in parts]); ee = np.concatenate([q["st_ens"][:, sel] for q in parts]); gg = np.concatenate([q["st_ghap"][:, sel] for q in parts])
    Em, GH = {}, {}
    for n_, (d_, h_, t_) in enumerate(kk):
        Em[(int(d_), int(h_))] = np.nanmean(ee[n_].astype(np.float32), 0)
        GH[int(t_)] = float(np.nanmean(gg[n_].astype(np.float32)))
    la_, lo2 = st["lat"][sel], st["lon"][sel]
    days = np.arange(1, 29); cal = days + 1
    m1 = np.stack([Em[(t - 1, 1)] for t in days]); m3 = {t: Em[(t - 3, 3)] for t in days if t >= 3}
    rws = []
    for i, t in enumerate(days):
        cs = B.bilinear04(np.asarray(cams[int(t), 0, 0], np.float32), la_, lo2)
        rws.append([f"{B.day2022(t):%Y-%m-%d}", int(cal[i]), sig(obs[t]), sig(GH[int(t)]), sig(m1[i].mean()), sig(m1[i].min()), sig(m1[i].max()),
                    sig(m3[t].mean()) if t in m3 else None, sig(float(np.nanmean(cs))), "yes" if int(cal[i]) == 24 else "no"])
    sheet(f"{sh}u", fg, "u",
          "Daily mean PM2.5 (µg m-3, UTC day) in central Beijing in January 2022: mean of the measurements at the 12 official monitors, the "
          "1 km estimates (GHAP) at their grid cells, and the forecasts read at the monitors, averaged over them: Foehn issued 1 day earlier "
          "(ensemble mean, lowest and highest of the 16 members), Foehn's ensemble mean issued 3 days earlier and the CAMS forecast issued "
          "at the same time as the 1-day forecast; the vertical line marks 24 January.",
          ["date", "day of January 2022", "measured, mean of the monitors", "GHAP at the monitor grid cells", "Foehn 1 day ahead, ensemble mean",
           "Foehn 1 day ahead, lowest member", "Foehn 1 day ahead, highest member", "Foehn 3 days ahead, ensemble mean", "CAMS forecast",
           "vertical line"], rws, scr, [fs, B.CASE_DIR / "part_*.npz", B.WORLD / "cams_forecast_daily" / "2022.zarr"],
          note=f"{int(sel.sum())} monitors selected")
    panel(fg, "a-g", "North China Plain, 24 January 2022: the 1 km estimates and six forecast fields issued 1 day ahead",
          f"maps not exported; their means, members' ranks and RMSEs in {sh}_box")
    panel(fg, "h", "power spectra of the fields of a-g (lines and band, logarithmic axes)", f"{sh}h")
    panel(fg, "i-o", "the same fields issued 3 days ahead", f"maps not exported; summary numbers in {sh}_box")
    panel(fg, "p", "power spectra of the fields of i-o (lines and band)", f"{sh}p")
    panel(fg, "q-t", "probability of exceeding 75 µg m-3 from 16 members, issued 7, 5, 3 and 1 days ahead, with the 75 µg m-3 contour of GHAP",
          f"maps not exported; mean probabilities in {sh}_box")
    panel(fg, "u", "daily series at the 12 monitors of central Beijing, January 2022 (points, lines, band)", f"{sh}u")
    panel(fg, "legend", "choice of the day and scores over the 22 days of January; RMSE at the plain's stations", f"{sh}_days, {sh}_legend")


# ================================================================================================ Suppl. Fig. 11 (25 January 2022)
def sfig_realism_25jan(fg, sh):
    """the former Fig. 5 (the builder's fig6: h, p, u), plus the numbers of its legend: the box values of 25 January from
    print/show/fig5_24jan_numbers.json (results for target day 24), as rebuild/print/integr/si.md (SI-F3) cites them"""
    B.fig6(fg, sh)
    fn = N / "rebuild" / "print" / "show" / "fig5_24jan_numbers.json"
    r = [x for x in jload(fn)["results"] if x["date"] == "2022-01-25"][0]
    rws = []
    for h in (1, 3, 5, 7):
        bx = r["box"][f"lead{h}"]
        for k, lab in (("rmse_foehn_mean", "RMSE against GHAP over the land of the panel, Foehn's ensemble mean"),
                       ("rmse_cams", "RMSE against GHAP over the land of the panel, CAMS forecast issued at the same time"),
                       ("rmse_aurora", "RMSE against GHAP over the land of the panel, Aurora issued at the same time"),
                       ("foehn_mean", "mean over the land of the panel, Foehn's ensemble mean"), ("ghap", "mean over the land of the panel, GHAP")):
            if bx.get(k) is not None:
                rws.append([h, lab, sig(bx[k]), "µg m-3"])
        rws.append([h, "ranks of members 1, 2 and 3 among the 16 by their mean over the plain (1 = highest)",
                    ", ".join(str(x) for x in bx["rank_by_boxmean_members123_1=highest"]), "rank"])
    for h in range(1, 8):
        q = r["r3"][f"lead{h}"]
        rws += [[h, "mean probability of exceeding 75 µg m-3 (share of the 16 members) where GHAP exceeds 75", sig(q["P_where_gt75"]), "probability"],
                [h, "mean probability of exceeding 75 µg m-3 elsewhere on land", sig(q["P_elsewhere"]), "probability"],
                [h, "mean of Foehn's ensemble mean where GHAP exceeds 75", sig(q["ensmean_where_gt75"]), "µg m-3"]]
    rws.append([0, "mean of GHAP where it exceeds 75", sig(r["r3"]["ghap_mean_where_gt75"]), "µg m-3"])
    sheet(f"{sh}_legend", fg, "legend, q-t",
          "Numbers of the legend for 25 January 2022 over the land of the North China Plain panel, by the lead time (days) at which the "
          "forecasts were issued: RMSE against the 1 km estimates (GHAP; 0.4° forecasts read at the 1 km pixels), means over the plain, "
          "ranks of members 1-3, and the mean probability of exceeding 75 µg m-3 from the 16 members where GHAP exceeds 75 and elsewhere.",
          ["issued ahead (days)", "quantity", "value", "unit"], rws, N / "rebuild" / "figs_show" / "numbers_show.py", [fn])
    panel(fg, "legend", "box values of 25 January (RMSE against the estimates, members' ranks, mean probabilities)", f"{sh}_legend")
    B.NOTES.append(f"{fg}: the RMSEs at the 137 official stations of the plain on 25 January given in its legend have no sheet; they are "
                   "in the file score_tables/case_studies_stations/stations_plain_22days.json that accompanies the Source Data.")


# ================================================================================================ Supplementary Tables
def away_a(t_label_n):
    """panel a of the table 'away' (and the values the legend sends to the Source Data: the ring at 3 km, the gains over Aurora,
    3 days ahead) from analyses/circularity/results_ar/numbers.json C/groups (the dict that make_outputs.py prints, L218-226),
    every value at the precision stored there; RMSE with four significant digits"""
    fn = B.ANA / "circularity" / "results_ar" / "numbers.json"; J = jload(fn)["C"]["groups"]
    disp = {"China": "China (CNEMC)", "USA": "USA (EPA)", "Europe": "Europe (EEA)", "Korea": "Korea (AirKorea)", "Japan": "Japan",
            "India": "India (CPCB)", "all six networks": "Six networks"}
    cols = ["network", "lead time (days)", "stations with station-days at this lead in the ring sample", "station-days",
            "estimate within 0.5 µg m-3 of the measurement, at the station pixel (% of station-days)", "the same, one pixel 3 km away (%)",
            "the same, one pixel 10 km away (%)", "RMSE of Foehn at the station pixel (µg m-3)", "RMSE of Foehn on the ring 3 km away (µg m-3)",
            "RMSE of Foehn on the ring 10 km away (µg m-3)", "RMSE of the CAMS forecast (µg m-3)", "RMSE of Aurora (µg m-3)"]
    keys = [("foehn", "at the pixel"), ("ring3", "3 km away"), ("ring10", "10 km away")]
    for ref in ("cams", "aurora"):
        for k, lab in keys:
            nm = "the CAMS forecast" if ref == "cams" else "Aurora"
            cols += [f"gain over {nm} {lab} (%)", f"gain over {nm} {lab} (%), 95% interval low", f"gain over {nm} {lab} (%), 95% interval high"]
    rows = []
    for g in disp:
        for h in ("1", "3"):
            d = J[g]["leads"][h]; r_ = d["rmse"]; w_ = d["estimates_share_within_0.5"]
            row = [disp[g], int(h), int(d["stations_at_lead"]), int(d["station_days"]), w_["ghapt"], w_["g_one3"], w_["g_one10"],
                   sig(r_["foehn"], 5), sig(r_["ring3"], 5), sig(r_["ring10"], 5), sig(r_["cams1"], 5), sig(r_["aur1"], 5)]
            for ref in ("cams", "aurora"):
                for k, _ in keys:
                    v, lo, hi = d[f"gain_{k}_vs_{ref}"]           # stored with two decimals; the point value again from the RMSEs
                    pt = 100 * (1 - r_[k] / r_["cams1" if ref == "cams" else "aur1"])
                    assert abs(pt - v) <= 0.0051, (g, h, k, ref, pt, v)
                    row += [sig(pt, 5), lo, hi]
            rows.append(row)
    return dict(label="away", part="", file=fn, script=B.ANA / "circularity" / "scripts" / "make_outputs.py", cols=cols, rows=rows,
                what="complete table: the rings at 3 and 10 km, the gains over the CAMS forecast and Aurora, 1 and 3 days ahead",
                desc="Complete Supplementary Table: Foehn (ensemble mean of four members) in 2022 read at the 1 km pixel of each "
                     "station and on the square rings of pixels 3 and 9 pixels (about 3 and 10 km) from it (ring pixels with at least 100 "
                     "residents and no monitor; stations with at least four such pixels on each ring; every second start day; identical "
                     "station-days in a row), against the CAMS forecast and Aurora issued at the same time and read at the station, 1 and "
                     "3 days ahead. Gains are reductions of the RMSE in per cent with 95% intervals (paired moving-block bootstrap over "
                     "start dates, blocks of 7, 10,000 resamples).")


def pairs_note():
    """the Supplementary Note on opposite pairs (A19) and the noise-aware mean (A20): values of the decision files score19.json and
    score20.json (stations of 2022, sample S'; populated pixels for the probabilities), estimates and 95% intervals as stored"""
    f19, f20 = P / "a19" / "code" / "out" / "score19.json", P / "a20" / "code" / "out" / "score20.json"
    a, b = jload(f19)["headline_stations"], jload(f20)
    bs, bp = b["headline_stations"], b["headline_pixels"]["residents"]
    rows = []

    def add(q, ens, h, x, unit, src):
        if unit == "station-days":                        # a count: exact, no interval
            rows.append([q, ens, int(h), int(x["est"]), None, None, unit, src]); return
        rows.append([q, ens, int(h), sig(x["est"], 5), sig(x["ci"][0], 5), sig(x["ci"][1], 5), unit, src])
    U, U2 = "µg m-3", "(µg m-3)^2"
    for h in ("1", "3"):
        add("station-days of the sample S'", "all", h, a[h]["n"], "station-days", "pairs")
        for k, ens in (("rmse_4", "four independent members, as run (the paper's evaluation)"), ("rmse_2p", "two opposite pairs (four members), as run"),
                       ("rmse_8p", "four opposite pairs (eight members), as run"), ("rmse_i8", "eight independent members, as run"),
                       ("rmse_form8", "eight independent members, the paper's form"), ("rmse_inf", "infinitely many independent members, the paper's form")):
            add("RMSE of the ensemble mean", ens, h, a[h][k], U, "pairs")
        add("RMSE of the ensemble mean", "noise-aware mean of the four members as run", h, bs[h]["rmse_c"], U, "noise-aware mean")
        for ref, rk in (("the CAMS forecast", "cams"), ("Aurora", "aur")):
            for k, ens in ((f"red_four_vs_{rk}", "four independent members, as run"), (f"red_two_pairs_vs_{rk}", "two opposite pairs"),
                           (f"red_four_pairs_vs_{rk}", "four opposite pairs"), (f"red_form8_vs_{rk}", "eight independent members, the paper's form"),
                           (f"red_inf_vs_{rk}", "infinitely many independent members, the paper's form")):
                if k in a[h]:
                    add(f"RMSE reduction against {ref}", ens, h, a[h][k], "%", "pairs")
            add(f"RMSE reduction against {ref}", "noise-aware mean of the four members as run", h, bs[h][f"red_corr4_vs_{rk}"], "%", "noise-aware mean")
        add("MSE(eight independent members, the paper's form) - MSE(four opposite pairs)", "D (pairs)", h, a[h]["D"], U2, "pairs")
        add("MSE(four independent members as run) - MSE(two opposite pairs)", "O (pairs)", h, a[h]["O"], U2, "pairs")
        add("MSE(eight independent members as run) - MSE(four opposite pairs)", "D as run (pairs)", h, a[h]["D_asrun"], U2, "pairs")
        add("within-pair correlation of the departures (concentration; reference = mean of the other pairs, A17 4.4)", "opposite pairs", h, a[h]["rho_anti"], "correlation", "pairs")
        add("within-pair correlation of the departures (raw estimator, concentration)", "opposite pairs", h, a[h]["rho_anti_raw"], "correlation", "pairs")
        add("correlation for pairs of independent members (null)", "independent members", h, a[h]["rho_null"], "correlation", "pairs")
        add("equivalent number of independent members (against the paper's forms; truth-based)", "four opposite pairs", h, a[h]["meq_8p"], "members", "pairs")
        add("equivalent number of independent members (against the paper's forms; truth-based)", "two opposite pairs", h, a[h]["meq_2p"], "members", "pairs")
        add("MSE(plain mean of four) - MSE(noise-aware mean of four)", "D (noise-aware mean)", h, bs[h]["D"], U2, "noise-aware mean")
        add("MSE(eight independent members, the paper's form) - MSE(noise-aware mean of four)", "noise-aware mean", h, bs[h]["d_form8_minus_corr"], U2, "noise-aware mean")
        add("equivalent number of plain members (truth-free)", "noise-aware mean of four", h, bs[h]["meq_tf_corr"], "members", "noise-aware mean")
        add("equivalent number of plain members (truth-free)", "plain mean of four", h, bs[h]["meq_tf_plain"], "members", "noise-aware mean")
        add("equivalent number of members against the paper's forms (truth-based)", "noise-aware mean of four", h, bs[h]["meq_corr_vs_forms"], "members", "noise-aware mean")
        add("share of a member's departure explained by the law, out of sample", "noise-aware mean", h, bs[h]["R2_law"], "fraction", "noise-aware mean")
        add("D without the iteration", "noise-aware mean (variant)", h, bs[h]["D_noiter"], U2, "noise-aware mean")
        add("D with the law of the first design", "noise-aware mean (variant)", h, bs[h]["D_A18law"], U2, "noise-aware mean")
        add("bias of the noise-aware mean against the plain mean, relative to the mean measurement", "noise-aware mean", h, bs[h]["bias_rel"], "fraction", "noise-aware mean")
        for th in ("37.5", "75.0"):
            for k, ens in ((f"bs_an4_{th}", "closed form from the four members (the law)"), (f"bs_nl4_{th}", "the same form without the law (control)"),
                           (f"bs_fr4_{th}", "counted on the four members"), (f"bs_fr16_{th}", "counted on sixteen members")):
                add(f"Brier score above {float(th):g} µg m-3, residents of the populated 1 km grid cells", ens, h, bp[h][k], "Brier score", "noise-aware mean")
            add(f"Brier score above {float(th):g} µg m-3: counted on sixteen minus closed form from four", "difference", h, bp[h][f"d_fr16_an4_{th}"], "Brier score", "noise-aware mean")
            add(f"Brier score above {float(th):g} µg m-3: control without the law minus closed form", "difference", h, bp[h][f"d_nl4_an4_{th}"], "Brier score", "noise-aware mean")
    return rows, [f19, f20]


def supp_tables_final():
    """the builder's supp_tables (L1862-1891) with: the four-member tables named _four_members (no part b in the print),
    panel a of 'away' from results_ar, and the provisional sheet of the Note on opposite pairs and the noise-aware mean"""
    import re
    W = B.W
    sup = W / "work" / "supplementary.tex"; used(sup)
    labels = []
    for m in re.finditer(r"\\label\{tab:([A-Za-z0-9_]+)\}", sup.read_text(encoding="utf-8")):
        if m.group(1) not in labels:
            labels.append(m.group(1))
    B.CURRENT["stem"] = "supplementary"
    tabs = [jload(fn) | {"file": fn} for fn in sorted((B.HERE / f"supp_tables_full{B.SFX}").glob("*.json"))]
    for t in tabs:
        if t["label"] == "away" and t["part"] == "b":     # the former part b is its own table since the split (label awaycams)
            t["label"], t["part"] = "awaycams", ""
            t["what"] = "the table as printed, with numeric columns"
            t["desc"] = t["desc"].replace("Supplementary Table, panel b: ", "Supplementary Table: ")
        if t["label"] in ("warned", "matched", "regions") and t["part"] == "":
            t["part"] = "_four_members"
            t["what"] = "the same table with the four members of the evaluation (at least 2 of 4), all 179 start days of 2022"
            t["desc"] += " Foehn: at least 2 of its 4 members (the evaluation as run), all 179 start days of 2022."
    w8, d5 = W / "rebuild" / "print" / "warn8" / f"out{B.SFX}", W / "rebuild" / "print" / "d5" / f"out{B.SFX}" / "d5.json"
    for fn, sc in [(w8 / f"{k}.json", W / "rebuild" / "print" / "warn8" / "build_warn8.py") for k in ("warned", "matched", "regions")] + [
            (d5, W / "rebuild" / "print" / "d5" / "build_d5.py")]:
        t = jload(fn) | {"file": fn, "script": sc}
        t["desc"] = t["desc"].replace(" (A16 section 4, point 3)", "")
        tabs.append(t)
    tabs.append(away_a(None))
    for t in sorted(tabs, key=lambda t: (labels.index(t["label"]), t["part"])):
        n = labels.index(t["label"]) + 1
        fg, name = f"Supplementary Table {n}", f"STable{n}{t['part']}"
        sheet(name, fg, t["part"].strip("_").replace("_", " "), t["desc"], t["cols"], t["rows"], t.get("script", W / "tools" / "si_tables_portrait.py"), [t["file"]])
        panel(fg, t["part"].strip("_").replace("_", " "), t["what"], name)
    for k, lab in enumerate(labels, 1):
        if not any(s["figure"] == f"Supplementary Table {k}" for s in B.SHEETS):
            panel(f"Supplementary Table {k}", "all", f"table '{lab}' of the Supplementary Information",
                  "no sheet: the table prints its values and sends none to the Source Data")
    rows, files = pairs_note()
    notes = re.findall(r"\\snote\{note:([A-Za-z0-9_]+)\}", sup.read_text(encoding="utf-8"))
    nn, nt = notes.index("order") + 1, labels.index("order") + 1
    sheet(f"SNote{nn}", f"Supplementary Note {nn}", "",
          "Ensemble drawn as opposite pairs and noise-aware mean of the four members (same model, weights and sampler), stations of the six "
          "official networks in 2022 with a populated pixel (sample S'), 1 and 3 days ahead; Brier scores at the residents of the 1 km grid "
          "cells with at least 100 residents. Estimates with 95% intervals (moving-block bootstrap over start days). "
          f"Supplementary Table {nt} prints its own values, all of which are in this sheet.",
          ["quantity", "ensemble or form", "lead time (days)", "estimate", "95% interval, low", "95% interval, high", "unit", "test"], rows,
          P / "a20" / "code" / "score20.py", files)
    panel(f"Supplementary Note {nn}", "", f"values of the Note on opposite pairs and the noise-aware mean (Supplementary Table {nt} among them)", f"SNote{nn}")


# ================================================================================================ main
def main():
    t0 = time.time()
    B.BUILDERS.update({"fig_overview": fig3_feb, "fig_gc2_realism": fig5_jan24, "figS_overview_21jan": B.fig3, "figS_realism_25jan": sfig_realism_25jan})
    figs = B.manuscript_figures()
    for fg, sh, stem, remark in figs:
        B.CURRENT["stem"] = stem
        fn = B.fig1 if stem.startswith(B.SCHEMATIC) else B.BUILDERS.get(stem)
        print(f"{fg} = {stem} ...", flush=True)
        if fn is None:
            panel(fg, "all", "figure file unknown to build_source_data.py", "NOT EXPORTED: no export function for this figure file")
            B.NOTES.append(f"`{stem}.pdf` ({fg}) has no export function: NOTHING is exported for it.")
            continue
        fn(fg, sh)
    supp_tables_final()
    B.FIGLIST[:] = figs
    # wording only (no value changes): the paper names this reference 'start-day estimate' and never 'persistence'
    for s in B.SHEETS:
        s["desc"] = s["desc"].replace(" (persistence)", "")
        s["rows"] = [[v.replace(" (persistence)", "") if isinstance(v, str) else v for v in r] for r in s["rows"]]
    for p_ in B.PANELS:
        p_["what"] = p_["what"].replace(" (persistence)", "")
    left = [s["name"] for s in B.SHEETS if "persisten" in (s["desc"] + " ".join(str(v) for r in s["rows"] for v in r)).lower()]
    assert not left, left
    # outputs: the builder's writers with the output suffix '_final' (every input was read above with '_ar')
    B.SFX = "_final"
    # providers and terms of the measured values that the workbook reproduces: README sheet (row 3) and remarks of the manifest
    ind = [s["name"] for s in B.SHEETS if s["name"].endswith(("_stations", "_monitors"))]
    assert len(ind) == 5, ind
    measured = ("Measured values. The sheets " + ", ".join(ind) + " give daily means at individual stations or monitors, reproduced "
                "with attribution to their providers and under their terms: European Environment Agency, Air Quality e-Reporting "
                "(CC BY 4.0); US Environmental Protection Agency, Air Quality System (public domain); AirKorea, final confirmed data "
                "of the Korea Environment Corporation and the National Institute of Environmental Research (open government data of "
                "Korea); Central Pollution Control Board of India and the state boards (open government data of India); and, through "
                "OpenAQ, monitors of AirGradient and Hawanama (CC BY 4.0), AirNow (US public domain), Clarity (CC0 1.0), the Central "
                "Pollution Control Board and other government monitors of India. The measurements of the China National Environmental "
                "Monitoring Centre are given only as means over monitors. All other sheets hold forecasts, estimates or scores.")
    B.NOTES.append(measured)
    B.write_xlsx()
    import openpyxl
    xp = B.HERE / "Source_Data_final.xlsx"; raw = xp.read_bytes()
    wb = openpyxl.load_workbook(io.BytesIO(raw)); ws = wb["README"]
    assert ws["A3"].value is None and str(ws["A4"].value) == B.README_COLS[0], (ws["A3"].value, ws["A4"].value)
    ws["A3"] = measured
    buf = io.BytesIO(); wb.save(buf)
    a, b = openpyxl.load_workbook(io.BytesIO(raw)), openpyxl.load_workbook(io.BytesIO(buf.getvalue()))
    assert a.sheetnames == b.sheetnames
    for nm in a.sheetnames:                                # every value of every sheet unchanged by the second save, README row 3 aside
        va = [[c.value for c in r] for r in a[nm].iter_rows()]; vb = [[c.value for c in r] for r in b[nm].iter_rows()]
        if nm == "README":
            va[2] = vb[2] = None
        assert va == vb, nm
    B.safe_write(xp, buf.getvalue())
    B.write_manifests()
    # the opening sentence of the builder names its own outputs (Source_Data.xlsx, csv/): name the files written here
    mp = B.HERE / "MANIFEST_final.md"; mt = mp.read_text(encoding="utf-8")
    o1 = ("`Source_Data.xlsx` holds one sheet per figure panel with graph data (lines, bars, points, histograms, forest plots, violins) "
          "and `csv/<sheet>.csv` the same tables as plain text (header row, then the values; `csv/README.csv` lists the sheets). ")
    n1 = ("`Source_Data_final.xlsx` holds one sheet per figure panel with graph data (lines, bars, points, histograms, forest plots, violins) "
          "and `csv_final.zip` the same tables as plain text, one file `csv_final/<sheet>.csv` per sheet (header row, then the values; "
          "`csv_final/README.csv` lists the sheets). ")
    assert mt.count(o1) == 1, "opening sentence of the manifest not found"
    mt = mt.replace(o1, n1); B.check_clean(mt, "MANIFEST_final.md"); B.safe_write(mp, mt)
    with zipfile.ZipFile(B.HERE / "Source_Data_final.xlsx") as z:
        for nm in z.namelist():
            B.check_clean(z.read(nm).decode("utf-8", "replace"), f"Source_Data_final.xlsx:{nm}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for s in B.SHEETS:
            t = B.csv_text(s["cols"], s["rows"]); B.check_clean(t, s["name"]); z.writestr(f"csv_final/{s['name']}.csv", t)
        t = B.csv_text(B.README_COLS, B.readme_rows()); B.check_clean(t, "README.csv"); z.writestr("csv_final/README.csv", t)
    B.safe_write(HERE_SD / "csv_final.zip", buf.getvalue())
    print(f"done in {time.time() - t0:.0f} s: {len(B.SHEETS)} sheets, {len(B.PANELS)} panel rows", flush=True)
    for nt in B.NOTES:
        print("NOTE:", nt)


if __name__ == "__main__":
    main()
