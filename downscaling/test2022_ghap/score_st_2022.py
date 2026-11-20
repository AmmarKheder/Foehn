#!/usr/bin/env python3
"""GHAP 2022 copy (PROTOCOL.md A2(d): truth test2022_ghap/st_truth_2022.npz, GHAP mask, region North > 60 N added, wording: GHAP is fitted to ~9,500 stations 2017-2022, monitors NOT independent) of: Station scores (PROTOCOL A1(h), section 6; evaluation only): is the 1 km field better than the 40 km field at the
monitors?

Truth: st_truth_<year>.npz (build_st.py): UTC-day means with >= 18 valid hours. Monitors sharing a 1 km pixel are
averaged per pixel-day (within a network for the network rows; across networks for the 'All' rows), so every score is
over pixel-stations.
Predictions:
  2023  coupling/preds_st/<arm>/<init>_v<valid>.npz      members (M, len(st_pix_2023.pos)) float16 (common.save_pred_st)
  2024  final2024/preds_st/<arm>/<YYYY-MM-DD>.npz          members (M, len(st_pix_2024.pos)) (or <init>_v<valid>.npz)
Metrics per arm x lead x group (network x region; regions = coupling/score.py REGIONS, by pixel centre), pooled over
pixel-days unless stated:
  rmse / bias of the ensemble mean, fair CRPS (Ferro 2014; M = 1: absolute error), Pearson R (pooled), median
  per-pixel-station temporal R (pixels with >= min_days pairs), rmse_cell (0.25 deg cells weighted equally,
  secondary), n pixels and pixel-days.
Contrasts (paired on common (init, valid) / days and common valid pixel-days): 1 km - 40 km, and every arm - climatology:
  delta rmse, crps, R pooled (and relative change); for lead 'all' also delta median per-station R and the share of
  pixel-stations where the first arm's RMSE is lower (pixels with >= min_days pairs in both).
95 % intervals: percentile bootstrap, B = 2000, resampling the inits (2023) or calendar weeks (2024, days 7k..7k+6 of
the year) with replacement, the same draw for both arms of a contrast. Per-station quantities (share, median per-station
R): bootstrap over 2-degree clusters of pixel-stations (B = 1000), each station keeping all its days.
    cdffm_run python score_st.py 2023   |   cdffm_run python score_st.py 2024
"""
import argparse
import os
import datetime as dt
import json
import re
import time
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
C = P / "test2022_ghap"
MASK = P / "data/processed/ghap/grid/ghap_mask.npy"
REGIONS = [("Europe", 35.0, 60.0, -10.0, 35.0), ("East Asia", 20.0, 50.0, 100.0, 135.0), ("South Asia", 5.0, 35.0, 60.0, 95.0),
           ("North America", 15.0, 50.0, -130.0, -60.0), ("Africa", -35.0, 35.0, -18.0, 52.0),
           ("South America", -40.0, 12.0, -82.0, -34.0), ("North", 60.0, 90.0, -180.0, 180.0)]   # + North (> 60 N, A2(d))
RNAMES = ["Global"] + [r[0] for r in REGIONS]
CFG = {
    2023: dict(preds=C / "preds_st", extra=[C / "st_ref", C / "st_derived"], out=C / "results", unit="init", min_days_lead=12, min_days_all=40,
               pairs=[("rt_cdffm", "rt_bilinear"), ("camsfc_cdffm", "camsfc_bilinear"), ("foehn_cdffm", "rt_bilinear"),
                      ("foehn_cdffm_camsN", "rt_bilinear"), ("rt_cdffm_t6fc", "rt_bilinear"),
                      ("camsfc_cdffm_t6fc", "camsfc_bilinear"), ("foehn_cdffm", "cdffm_persist"),
                      ("foehn_cdffm_camsN", "cdffm_persist"), ("foehn_cdffm_camsN", "foehn_cdffm")],
               key_pairs=[("rt_cdffm", "rt_bilinear"), ("camsfc_cdffm", "camsfc_bilinear"),
                          ("foehn_cdffm", "rt_bilinear"), ("foehn_cdffm_camsN", "rt_bilinear"),
                          ("foehn_cdffm", "cdffm_persist"), ("foehn_cdffm_camsN", "cdffm_persist")],
               fig_pairs=[("rt_cdffm", "rt_bilinear"), ("camsfc_cdffm", "camsfc_bilinear"),
                          ("foehn_cdffm", "rt_bilinear"), ("foehn_cdffm_camsN", "rt_bilinear")]),
    2022: dict(preds=C / "preds_st", extra=[], out=C / "results", unit="week", min_days_lead=120, min_days_all=120,
               pairs=[("cdffm", "cams_bilinear"), ("cdffm", "bcsd"), ("bcsd", "cams_bilinear")], key_pairs=[("cdffm", "cams_bilinear")],
               fig_pairs=[("cdffm", "cams_bilinear")]),
    2024: dict(preds=P / "final2024/preds_st", extra=[P / "final2024/st_derived"], out=P / "final2024/results", unit="week", min_days_lead=120, min_days_all=120,
               pairs=[("cdffm", "cams_bilinear"), ("cdffm", "bcsd"), ("bcsd", "cams_bilinear")], key_pairs=[("cdffm", "cams_bilinear")],
               fig_pairs=[("cdffm", "cams_bilinear")]),
}
CLIM_NAMES = ("clim", "b0", "B0", "climatology")
LABELS = {"foehn_cdffm_camsN": "sensitivity arm, not in DESIGN.md (north-of-60N hole filled with CAMS)",
          "foehn_cdffm": "T2: Foehn 1 km from the CD-FFM initial state (DESIGN.md)",
          "rt_cdffm": "T1: RT 40 km forecast downscaled by CD-FFM", "rt_bilinear": "RT 40 km, bilinear",
          "camsfc_cdffm": "CAMS 00 UTC forecast downscaled by CD-FFM", "camsfc_bilinear": "CAMS 00 UTC forecast, bilinear",
          "cdffm_persist": "CD-FFM of the init day, persisted (1 km)", "clim": "LGHAP climatology ensemble (B0)",
          "camsfc_cdffm_t6fc": "diagnostic (T6 from the arm's own forecast)", "rt_cdffm_t6fc": "diagnostic (T6 from the arm's own forecast)",
          "lghap": "LGHAP v2 itself (target product, station-fitted): reference only, not a model",
          "cdffm": "CD-FFM 1 km (final system)", "cams_bilinear": "B1: CAMS 40 km, bilinear", "bcsd": "B2: BCSD 1 km"}


def label(n):
    if n.endswith("__cell25"):
        return f"q4 diagnostic: {n[:-8]} averaged over its 0.25 deg cell"
    return LABELS.get(n, "")
REF_NAMES = ("lghap", "lghap_target", "ghap")                                    # reference product, not a model: no contrasts
B, B_ST = 2000, 1000
RNG_SEED = 20261028


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fair_crps(X, y):
    M = X.shape[0]
    e = np.abs(X - y[None]).mean(0)
    if M == 1:
        return e
    xs = np.sort(X, axis=0)
    i = np.arange(M, dtype=np.float32)[:, None]
    return e - 2.0 * ((2.0 * i - M + 1.0) * xs).sum(0) / (2.0 * M * (M - 1))


# ------------------------------------------------------------------------------------------------ truth and groups
def load_truth(year, field="daily", local=False):
    pos = np.load(C / f"st_pix_{year}.npz")["pos"]
    t = np.load(C / (f"st_truth_{year}_local.npz" if local else f"st_truth_{year}.npz"))
    flat = np.load(MASK, mmap_mode="r")[pos].astype(np.int64)
    plat, plon = 83.995 - 0.01 * (flat // 36000), -179.995 + 0.01 * (flat % 36000)        # GHAP lattice
    nets = ["All"] + sorted(set(t["network"].tolist()))
    npix, nd = len(pos), t[field].shape[1]
    T = np.full((len(nets), npix, nd), np.nan, np.float32)
    for k, n in enumerate(nets):
        m = np.ones(len(t["id"]), bool) if n == "All" else (t["network"] == n)
        d = t[field][m].astype(np.float64)
        ok = np.isfinite(d)
        s = np.zeros((npix, nd)); c = np.zeros((npix, nd))
        np.add.at(s, t["pix"][m], np.where(ok, d, 0.0)); np.add.at(c, t["pix"][m], ok)
        T[k] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    groups = []
    for k, n in enumerate(nets):
        has = np.isfinite(T[k]).any(1)
        for r in RNAMES:
            if r == "Global":
                inr = np.ones(npix, bool)
            else:
                _, la0, la1, lo0, lo1 = next(x for x in REGIONS if x[0] == r)
                inr = (plat >= la0) & (plat <= la1) & (plon >= lo0) & (plon <= lo1)
            g = np.where(has & inr)[0]
            if len(g) >= 5:
                groups.append((f"{n}/{r}", k, g))
    cell = (np.floor((plat + 90) / 0.25) * 1440 + np.floor((plon + 180) / 0.25)).astype(np.int64)
    global CL2
    CL2 = (np.floor((plat + 90) / 2.0) * 180 + np.floor((plon + 180) / 2.0)).astype(np.int64)
    info = dict(n_monitors={n: int((t["network"] == n).sum()) for n in nets[1:]}, n_pixels=int(npix),
                n_pixels_net={n: int(np.isfinite(T[k]).any(1).sum()) for k, n in enumerate(nets)},
                pixel_station_days={n: int(np.isfinite(T[k]).sum()) for k, n in enumerate(nets)})
    return pos, T, nets, groups, cell, info, dt.date(year, 1, 1)


# ------------------------------------------------------------------------------------------------ predictions
PAT2 = re.compile(r"^(\d{4}-\d{2}-\d{2})_v(\d{4}-\d{2}-\d{2})\.npz$")
PAT1 = re.compile(r"^(\d{4}-\d{2}-\d{2})\.npz$")


def load_arm(d, npix, T, day0, unit_kind):
    keys, leads, days, units, xm, cr, Ms = [], [], [], [], [], [], set()
    inits = []
    for f in sorted(d.glob("*.npz")):
        if ".tmp" in f.name:
            continue
        m2, m1 = PAT2.match(f.name), PAT1.match(f.name)
        if not (m2 or m1):
            continue
        try:
            z = np.load(f)
            X = np.maximum(z["members"].astype(np.float32), 0.0)                # floored at 0 as coupling/score.py (v2)
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        except Exception as e:                                           # file being written
            log(f"  skip {f}: {e}")
            continue
        if X.ndim != 2 or X.shape[1] != npix:
            log(f"  skip {f}: shape {X.shape} != (M, {npix})")
            continue
        if m2:
            init, valid = dt.date.fromisoformat(m2.group(1)), dt.date.fromisoformat(m2.group(2))
            lead = (valid - init).days
        else:
            valid = dt.date.fromisoformat(m1.group(1)); init = None
            lead = int(meta.get("lead", 0))
        di = (valid - day0).days
        if not 0 <= di < T.shape[2]:
            continue
        Ms.add(X.shape[0])
        keys.append((str(init), str(valid))); leads.append(lead); days.append(di); inits.append(init)
        units.append(di // 7 if unit_kind == "week" else None)
        xm.append(X.mean(0))
        cr.append(np.stack([fair_crps(X, T[k][:, di]) for k in range(T.shape[0])]))
    if not keys:
        return None
    if unit_kind == "init":
        ui = {s: k for k, s in enumerate(sorted(set(inits)))}
        units = [ui[s] for s in inits]
    return dict(keys=keys, kidx={k: i for i, k in enumerate(keys)}, lead=np.array(leads), day=np.array(days),
                unit=np.array(units, np.int64), xm=np.stack(xm), crps=np.stack(cr), M=sorted(Ms))


# ------------------------------------------------------------------------------------------------ metrics
CL2 = None                                                                   # 2-degree cluster id per pixel (load_truth)


def wmedian(v, W):
    """v (n,), W (B, n) non-negative weights -> (B,) weighted medians"""
    o = np.argsort(v)
    cw = np.cumsum(W[:, o], 1)
    idx = (cw < 0.5 * cw[:, -1:]).sum(1)
    return v[o][np.minimum(idx, len(v) - 1)]


def boot_counts(U, nb, rng):
    return np.stack([np.bincount(rng.integers(0, U, U), minlength=U) for _ in range(nb)]).astype(np.float64)


def pooled(S):
    """S (..., 9): n, se, e, crps, x, y, xx, yy, xy"""
    n = S[..., 0]
    with np.errstate(all="ignore"):
        cov = S[..., 8] / n - S[..., 4] * S[..., 5] / n ** 2
        vx = S[..., 6] / n - (S[..., 4] / n) ** 2
        vy = S[..., 7] / n - (S[..., 5] / n) ** 2
        return dict(rmse=np.sqrt(S[..., 1] / n), bias=S[..., 2] / n, crps=S[..., 3] / n, r=cov / np.sqrt(vx * vy))


def per_pix_r(S):
    """S (..., 6) per pixel: n, x, y, xx, yy, xy -> r"""
    n = S[..., 0]
    with np.errstate(all="ignore"):
        cov = S[..., 5] / n - S[..., 1] * S[..., 2] / n ** 2
        vx = S[..., 3] / n - (S[..., 1] / n) ** 2
        vy = S[..., 4] / n - (S[..., 2] / n) ** 2
        return cov / np.sqrt(vx * vy)


def gather(arm, sel, k, g, T):
    """files sel of arm, group pixels g, network k -> X, Y, CR (F, ng) with NaN where invalid"""
    X = arm["xm"][sel][:, g]
    Y = T[k][g][:, arm["day"][sel]].T
    CR = arm["crps"][sel, k][:, g]
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(CR)
    return np.where(ok, X, np.nan), np.where(ok, Y, np.nan), np.where(ok, CR, np.nan), ok


def unit_sums(vals, ok, units, U):
    """vals list of (F, ng) arrays (NaN-free where ok) -> (U, len(vals)) sums over files of each unit and pixels"""
    out = np.zeros((U, len(vals)))
    for j, v in enumerate(vals):
        out[:, j] = np.bincount(units, np.where(ok, v, 0.0).sum(1), minlength=U)
    return out


def arm_metrics(arm, sel, k, g, T, cell, min_days, rng):
    X, Y, CR, ok = gather(arm, sel, k, g, T)
    n = int(ok.sum())
    if n == 0:
        return None
    E = X - Y
    uniq, uu = np.unique(arm["unit"][sel], return_inverse=True)
    U = len(uniq)
    S = unit_sums([ok.astype(float), E * E, E, CR, X, Y, X * X, Y * Y, X * Y], ok, uu, U)
    pt = {kk: float(v) for kk, v in pooled(S.sum(0)).items()}
    bs = pooled(boot_counts(U, B, rng) @ S)
    res = dict(n_pixel_days=n, n_pixels=int(ok.any(0).sum()), n_units=U, obs_mean=float(np.nanmean(Y)))
    for kk in ("rmse", "bias", "crps", "r"):
        res[kk] = pt[kk]
        res[kk + "_ci"] = [float(np.nanpercentile(bs[kk], 2.5)), float(np.nanpercentile(bs[kk], 97.5))]
    npx = ok.sum(0)
    rp = per_pix_r(np.stack([npx, np.nansum(X, 0), np.nansum(Y, 0), np.nansum(X * X, 0), np.nansum(Y * Y, 0),
                             np.nansum(X * Y, 0)], -1).astype(np.float64))
    good = npx >= min_days
    res["median_station_r"] = float(np.nanmedian(rp[good])) if good.any() else None
    res["n_pixels_station_stats"] = int(good.sum())
    cg = cell[g]
    cu, ci = np.unique(cg, return_inverse=True)
    cse = np.bincount(ci, np.nansum(E * E, 0), len(cu)); cn = np.bincount(ci, npx, len(cu))
    res["rmse_cell"] = float(np.sqrt(np.mean(cse[cn > 0] / cn[cn > 0])))
    res["n_cells"] = int((cn > 0).sum())
    return res


def contrast(A, Bm, lead, k, g, T, cell, min_days, rng, station_stats):
    """paired A - B on common files and common valid pixel-days"""
    common = sorted(set(A["keys"]) & set(Bm["keys"]))
    if lead != "all":
        common = [c for c in common if A["lead"][A["kidx"][c]] == lead]
    if not common:
        return None
    ia = np.array([A["kidx"][c] for c in common]); ib = np.array([Bm["kidx"][c] for c in common])
    XA, Y, CA, okA = gather(A, ia, k, g, T)
    XB, _, CB, okB = gather(Bm, ib, k, g, T)
    ok = okA & okB
    n = int(ok.sum())
    if n == 0:
        return None
    EA, EB = XA - Y, XB - Y
    uniq, uu = np.unique(A["unit"][ia], return_inverse=True)
    U = len(uniq)
    S = unit_sums([ok.astype(float), EA * EA, EA, CA, XA, Y, XA * XA, Y * Y, XA * Y,
                   EB * EB, EB, CB, XB, XB * XB, XB * Y], ok, uu, U)
    ia_ = [0, 1, 2, 3, 4, 5, 6, 7, 8]; ib_ = [0, 9, 10, 11, 12, 5, 13, 7, 14]

    def both(Sx):
        return pooled(Sx[..., ia_]), pooled(Sx[..., ib_])
    pa, pb = both(S.sum(0))
    ba, bb = both(boot_counts(U, B, rng) @ S)
    res = dict(n_pixel_days=n, n_pixels=int(ok.any(0).sum()), n_units=U, n_files=len(common))
    for kk in ("rmse", "crps", "r", "bias"):
        res[kk + "_A"], res[kk + "_B"] = float(pa[kk]), float(pb[kk])
        d = ba[kk] - bb[kk]
        res["d_" + kk] = float(pa[kk] - pb[kk])
        res["d_" + kk + "_ci"] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        res[kk + "_A_ci"] = [float(np.nanpercentile(ba[kk], 2.5)), float(np.nanpercentile(ba[kk], 97.5))]
        res[kk + "_B_ci"] = [float(np.nanpercentile(bb[kk], 2.5)), float(np.nanpercentile(bb[kk], 97.5))]
        if kk in ("rmse", "crps"):
            rel = ba[kk] / bb[kk] - 1.0
            res["rel_" + kk] = float(pa[kk] / pb[kk] - 1.0)
            res["rel_" + kk + "_ci"] = [float(np.nanpercentile(rel, 2.5)), float(np.nanpercentile(rel, 97.5))]
    if station_stats:
        # per pixel-station over the common days; interval: bootstrap over 2-degree clusters of pixel-stations (resampling
        # the inits would shrink each station's sample and bias the share towards 0.5)
        Z = lambda v: np.where(ok, v, 0.0).sum(0)
        PT = np.stack([ok.sum(0).astype(float), Z(EA * EA), Z(EB * EB), Z(XA), Z(XB), Z(Y), Z(XA * XA), Z(XB * XB),
                       Z(Y * Y), Z(XA * Y), Z(XB * Y)], -1)
        good = PT[:, 0] >= min_days
        res["n_pixels_station_stats"] = int(good.sum())
        if good.sum() >= 5:
            with np.errstate(all="ignore"):
                win = (PT[good, 1] < PT[good, 2]).astype(float)
                ra = per_pix_r(PT[good][:, [0, 3, 5, 6, 8, 9]]); rb = per_pix_r(PT[good][:, [0, 4, 5, 7, 8, 10]])
            fin = np.isfinite(ra) & np.isfinite(rb)
            res.update(share_A_beats_B=float(win.mean()), median_r_A=float(np.median(ra[fin])),
                       median_r_B=float(np.median(rb[fin])))
            res["d_median_r"] = res["median_r_A"] - res["median_r_B"]
            _, cl = np.unique(CL2[g][good], return_inverse=True)
            cnt = boot_counts(cl.max() + 1, B_ST, rng)[:, cl]                    # (B_ST, n good pixels) weights
            sh = (cnt * win).sum(1) / cnt.sum(1)
            dm = wmedian(ra[fin], cnt[:, fin]) - wmedian(rb[fin], cnt[:, fin])
            res["share_A_beats_B_ci"] = [float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))]
            res["d_median_r_ci"] = [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))]
    return res


# ------------------------------------------------------------------------------------------------ report
def fci(v, ci, nd=2, pct=False):
    if v is None:
        return "-"
    f = (lambda x: f"{100 * x:+.{nd}f} %") if pct else (lambda x: f"{x:+.{nd}f}")
    return f"{f(v)} [{f(ci[0])}, {f(ci[1])}]"


def markdown(year, R):
    L = [f"# Station scores {year}: 1 km vs 40 km at the monitors ({R.get('day', 'UTC')} station days)", "",
         f"Written {time.strftime('%Y-%m-%d %H:%M')} by test2022_ghap/score_st_2022.py (GHAP system pmG_final). Evaluation "
         "only: the monitors were never used to train, select, calibrate or post-process the model. GHAP is fitted to about "
         "9,500 stations 2017-2022 (extra-trees), so these monitors are NOT independent of the target: the 1 km field carries "
         "a station imprint through GHAP itself (statics, climatology, training targets), and in 2022 GHAP was fitted to "
         "these very monitors. Every 1 km gain is therefore read against both the 40 km field and the climatology. "
         "Pinning (2021 product audit, product_audit/results/audit_2021.md): GHAP V1 is within 0.5 ug m-3 of the monitor's "
         "UTC daily mean on 81-84 % of monitor-days at the monitor's own pixel and has R > 0.999 for 74-77 % of monitors, "
         "but only 0-17 % one pixel away; a GHAP-trained model inherits station information at monitor pixels (per-pixel "
         "climatology, statics, training targets). The same scores on 1, 3 and 10 km rings around the monitors (monitor "
         "pixel excluded; A2(e)) are in results/st_ring*; GHAP 2022's own pinning is in results/pinning_2022.md. "
         f"Predictions scored here: {os.environ.get('GHAP_ST_WHERE', 'monitor pixel')}.", "",
         f"Truth: UTC-day means (GHAP's day convention, Wei et al. 2023; Foehn file stations_2022_utc.npz), >= 12 valid hours (Foehn rule; the coupling rule was 18), monitors in one 1 km pixel averaged per pixel-day. Monitors: "
         f"{R['info']['n_monitors']}; pixel-stations: {R['info']['n_pixels_net']}. 95 % intervals: paired bootstrap over "
         f"{'the inits' if year == 2023 else 'calendar weeks'}. Pooled over pixel-days unless stated. "
         "Delta = first arm - second arm (negative RMSE/CRPS delta = first arm better; 'share' = fraction of pixel-stations "
         "where the first arm's RMSE is lower).", ""]
    if year == 2023:
        L += ["Weather of every forecast arm: ERA5 (perfect prognosis). 2023 = development year: the coupling arms were "
              "frozen before any station number existed (PROTOCOL A1(h)).", ""]
    L += ["## Key contrasts (A - B), all leads pooled", "",
          "1 km vs 40 km pairs, plus (2023) the Foehn arms vs CD-FFM persistence (1 km vs 1 km). Members of every arm are "
          "floored at 0 ug m-3 before scoring (Foehn has rare negative members).", ""]
    for c in [c for c in R["contrasts"] if c["key"] and c["lead"] == "all"]:
        la = label(c["A"])
        L += [f"### {c['A']} - {c['B']}" + (f" ({c['A']}: {la})" if "sensitivity" in la else ""), "",
              "| group | px | RMSE B | RMSE A | dRMSE | dRMSE % | dCRPS | R B / A | dR | median-station dR | share A better |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for gname, r in c["groups"].items():
            if r is None:
                continue
            L.append(f"| {gname} | {r['n_pixels']} | {r['rmse_B']:.2f} | {r['rmse_A']:.2f} | {fci(r['d_rmse'], r['d_rmse_ci'])} | "
                     f"{fci(r['rel_rmse'], r['rel_rmse_ci'], 1, True)} | {fci(r['d_crps'], r['d_crps_ci'])} | "
                     f"{r['r_B']:.3f} / {r['r_A']:.3f} | {fci(r['d_r'], r['d_r_ci'], 3)} | "
                     f"{fci(r.get('d_median_r'), r.get('d_median_r_ci', [0, 0]), 3)} | "
                     + (f"{r['share_A_beats_B']:.2f} [{r['share_A_beats_B_ci'][0]:.2f}, {r['share_A_beats_B_ci'][1]:.2f}]"
                        if r.get("share_A_beats_B") is not None else "-") + " |")
        L.append("")
    leads = sorted({c["lead"] for c in R["contrasts"] if c["lead"] != "all"})
    if leads:
        L += ["## Key contrasts by lead (All/Global): dRMSE [95 % CI] (dCRPS)", "",
              "| contrast | " + " | ".join(f"lead {x}" for x in leads) + " |", "|---|" + "---|" * len(leads)]
        for A, Bn in R["key_pairs"]:
            row = []
            for x in leads:
                c = next((c for c in R["contrasts"] if c["A"] == A and c["B"] == Bn and c["lead"] == x), None)
                r = c and c["groups"].get("All/Global")
                row.append("-" if not r else f"{fci(r['d_rmse'], r['d_rmse_ci'])} ({r['d_crps']:+.2f})")
            L.append(f"| {A} - {Bn} | " + " | ".join(row) + " |")
        L.append("")
    det = [c for c in R["contrasts"] if c.get("detail") and c["lead"] == "all"]
    if det:
        L += ["## 1 km detail: the same 1 km arm at the station pixel (q2) - its 0.25 deg cell mean (q4), all leads pooled", "",
              "q4 = each member averaged over the subsample pixels (every 8th land pixel) of the 0.25 deg cell holding the "
              "station (cellmean_st.py). Negative dRMSE = the sub-0.25 deg detail helps (PROTOCOL section 6 rule).", "",
              "| arm | group | px | RMSE q4 | RMSE q2 | dRMSE | dRMSE % | dCRPS | dR | share q2 better |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in det:
            for gname, r in c["groups"].items():
                if r is None or not (gname.startswith("All/") or gname.endswith("/Global")):
                    continue
                L.append(f"| {c['A']} | {gname} | {r['n_pixels']} | {r['rmse_B']:.2f} | {r['rmse_A']:.2f} | "
                         f"{fci(r['d_rmse'], r['d_rmse_ci'])} | {fci(r['rel_rmse'], r['rel_rmse_ci'], 1, True)} | "
                         f"{fci(r['d_crps'], r['d_crps_ci'])} | {fci(r['d_r'], r['d_r_ci'], 3)} | "
                         + (f"{r['share_A_beats_B']:.2f} [{r['share_A_beats_B_ci'][0]:.2f}, {r['share_A_beats_B_ci'][1]:.2f}]"
                            if r.get("share_A_beats_B") is not None else "-") + " |")
        L.append("")
    L += ["## Every arm vs the climatology (All/Global, all leads pooled)", "",
          "| arm - clim | dRMSE | dRMSE % | dCRPS | dCRPS % | dR | share arm better |", "|---|---|---|---|---|---|---|"]
    for c in [c for c in R["contrasts"] if c["clim"] and c["lead"] == "all"]:
        r = c["groups"].get("All/Global")
        if r:
            L.append(f"| {c['A']} - {c['B']} | {fci(r['d_rmse'], r['d_rmse_ci'])} | {fci(r['rel_rmse'], r['rel_rmse_ci'], 1, True)} | "
                     f"{fci(r['d_crps'], r['d_crps_ci'])} | {fci(r['rel_crps'], r['rel_crps_ci'], 1, True)} | "
                     f"{fci(r['d_r'], r['d_r_ci'], 3)} | "
                     + (f"{r['share_A_beats_B']:.2f} [{r['share_A_beats_B_ci'][0]:.2f}, {r['share_A_beats_B_ci'][1]:.2f}]"
                        if r.get("share_A_beats_B") is not None else "-") + " |")
    L += ["", "## Arms", "", "| arm | what | M | files |", "|---|---|---|---|"]
    for a_, per in R["arms"].items():
        L.append(f"| {a_} | {label(a_) or '-'} | {per['M']} | {per['n_files']} |")
    L += ["", "## Every arm, All/Global (point values; RMSE/CRPS/bias in ug m-3)", "",
          "| arm | M | lead | px-days | RMSE | RMSE cell-eq. | CRPS | bias | R | median-station R |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a, per in R["arms"].items():
        for lead, gr in per["leads"].items():
            r = gr.get("All/Global")
            if r:
                L.append(f"| {a} | {per['M']} | {lead} | {r['n_pixel_days']} | {r['rmse']:.2f} | {r['rmse_cell']:.2f} | "
                         f"{r['crps']:.2f} | {r['bias']:+.2f} | {r['r']:.3f} | "
                         + (f"{r['median_station_r']:.3f}" if r['median_station_r'] is not None else "-") + " |")
    L += ["", "## Every arm by group, all leads pooled: RMSE / CRPS / R", ""]
    gnames = sorted({g for per in R["arms"].values() for g in per["leads"].get("all", {})},
                    key=lambda s: (s.split("/")[0] != "All", s))
    L += ["| arm | " + " | ".join(gnames) + " |", "|---|" + "---|" * len(gnames)]
    for a, per in R["arms"].items():
        gr = per["leads"].get("all", {})
        L.append(f"| {a} | " + " | ".join(f"{gr[g]['rmse']:.1f} / {gr[g]['crps']:.1f} / {gr[g]['r']:.2f}" if gr.get(g) else "-"
                                          for g in gnames) + " |")
    L += ["", "Notes: " + "; ".join(R["notes"]), ""]
    return "\n".join(L)


def figure(year, R, out, suffix=""):
    """rows = key pairs, columns = RMSE / Pearson R by region (All networks); 40 km vs 1 km bars with 95 % bootstrap
    intervals, climatology as a dashed tick; white background, no panel titles, bold panel letters"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    pairs = [tuple(p) for p in R["fig_pairs"]
             if any(c["A"] == p[0] and c["B"] == p[1] and c["lead"] == "all" for c in R["contrasts"])]
    if not pairs:
        return None
    regs = ["Global"] + [r[0] for r in REGIONS]
    short = {"North": "North >60N", "Global": "Global", "Europe": "Europe", "East Asia": "E Asia", "South Asia": "S Asia",
             "North America": "N America", "Africa": "Africa", "South America": "S America"}
    c40, c1, cc = "#86b6ef", "#1c5cab", "#52514e"
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
    nr = len(pairs)
    fig, axes = plt.subplots(nr, 2, figsize=(7.2, 2.25 * nr + 0.55), squeeze=False, facecolor="white")
    for i, (A, Bn) in enumerate(pairs):
        c = next(c for c in R["contrasts"] if c["A"] == A and c["B"] == Bn and c["lead"] == "all")
        cl = next((x for x in R["contrasts"] if x["clim"] and x["A"] == A and x["lead"] == "all"), None)
        for j, (m, lab) in enumerate((("rmse", "RMSE (\u00b5g m$^{-3}$)"), ("r", "Pearson R"))):
            ax = axes[i, j]
            ax.set_facecolor("white")
            used = []
            for q, rg in enumerate(regs):
                r = c["groups"].get(f"All/{rg}")
                if not r:
                    continue
                used.append(q)
                for off, side, col in ((-0.2, "B", c40), (0.2, "A", c1)):
                    v, ci = r[f"{m}_{side}"], r[f"{m}_{side}_ci"]
                    ax.bar(q + off, v, 0.38, color=col, edgecolor="white", linewidth=0.8)
                    ax.errorbar(q + off, v, yerr=[[max(v - ci[0], 0)], [max(ci[1] - v, 0)]], color="#0b0b0b", lw=0.8, capsize=1.8)
                rc = cl and cl["groups"].get(f"All/{rg}")
                if rc:
                    ax.plot([q - 0.42, q + 0.42], [rc[f"{m}_B"]] * 2, color=cc, lw=1.2, ls=(0, (2, 1.5)))
            ax.set_xticks(used)
            ax.set_xticklabels([short[regs[q]] for q in used], rotation=35, ha="right") if i == nr - 1 else ax.set_xticklabels([])
            ax.set_ylabel(lab)
            if m == "r":
                ax.set_ylim(0, 1)
            ax.grid(axis="y", color="#e6e6e3", lw=0.6)
            ax.set_axisbelow(True)
            ax.text(-0.16, 1.02, "abcdefghij"[2 * i + j], transform=ax.transAxes, fontweight="bold", fontsize=10,
                    va="bottom", ha="left")
        axes[i, 1].text(1.02, 0.5, f"1 km: {A}\n40 km: {Bn}", transform=axes[i, 1].transAxes, rotation=90,
                        va="center", ha="left", fontsize=6.5, color="#52514e")
    fig.legend([Patch(color=c40), Patch(color=c1), Line2D([0], [0], color=cc, lw=1.2, ls=(0, (2, 1.5)))],
               ["40 km (bilinear)", "1 km", "climatology"], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.045 if nr > 1 else 0.09, 0.97, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_st_{year}{suffix}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    return str(out / f"fig_st_{year}{suffix}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int, choices=(2022,))
    ap.add_argument("--preds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default=None, help="comma list (default: every arm directory)")
    ap.add_argument("--truth", default="daily", choices=("daily", "daily12"), help="daily12: >= 12 valid hours (sensitivity)")
    ap.add_argument("--local", action="store_true", help="station days = local solar day (st_truth_<year>_local.npz)")
    a = ap.parse_args()
    cfg = CFG[a.year]
    preds = Path(a.preds) if a.preds else cfg["preds"]
    out = Path(a.out) if a.out else cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    pos, T, nets, groups, cell, info, day0 = load_truth(a.year, a.truth, a.local)
    log(f"{a.year}: {len(pos)} pixels, networks {nets}, {len(groups)} groups")
    dirs = {}
    for root in [preds] + ([] if a.preds else cfg["extra"]):              # extra = reference products (st_ref/)
        if root.exists():
            for p in sorted(root.iterdir()):
                if p.is_dir() and p.name not in dirs:
                    dirs[p.name] = p
    names = a.arms.split(",") if a.arms else list(dirs)
    arms = {}
    for n in names:
        if n.startswith("TEST") or n not in dirs:
            continue
        arm = load_arm(dirs[n], len(pos), T, day0, cfg["unit"])
        if arm is not None:
            arms[n] = arm
            log(f"  arm {n}: {len(arm['keys'])} files, M {arm['M']}, leads {sorted(set(arm['lead'].tolist()))}")
    if not arms:
        log("no prediction files: nothing to score"); return
    R = dict(year=a.year, info=info, key_pairs=cfg["key_pairs"], fig_pairs=cfg["fig_pairs"], labels={}, arms={}, contrasts=[], notes=[
        "members of every arm floored at 0 ug m-3 before scoring (Foehn has rare negative members; as coupling/score.py v2); pixel-days pooled; rmse_cell = 0.25 deg cells weighted equally (secondary)",
        f"per-station statistics on pixels with >= {cfg['min_days_lead']} (per lead) / {cfg['min_days_all']} (all leads) pairs",
        "OpenAQ = all archive locations (isMonitor not applied, 2022-rule fallback)",
        "intervals: pooled metrics by bootstrap over inits (2023) / weeks (2024); per-station share and median R by "
        "bootstrap over 2-degree clusters of pixel-stations",
        "regions by pixel centre (coupling/score.py boxes); 'All' = networks pooled per pixel-day"])
    leadsets = {n: ["all"] + sorted(set(x["lead"].tolist())) for n, x in arms.items()}
    for n, arm in arms.items():
        R["arms"][n] = dict(M=arm["M"], n_files=len(arm["keys"]), label=label(n), leads={})
        for lead in leadsets[n]:
            sel = np.arange(len(arm["keys"])) if lead == "all" else np.where(arm["lead"] == lead)[0]
            md = cfg["min_days_all"] if lead == "all" else cfg["min_days_lead"]
            R["arms"][n]["leads"][str(lead)] = {gn: arm_metrics(arm, sel, k, g, T, cell, md, rng) for gn, k, g in groups}
        log(f"  metrics {n} done")
    clim = next((c for c in CLIM_NAMES if c in arms), None)
    todo = [(A, Bn, (A, Bn) in [tuple(p) for p in cfg["key_pairs"]], False) for A, Bn in cfg["pairs"] if A in arms and Bn in arms]
    if clim:
        todo += [(n, clim, False, True) for n in arms if n != clim and n not in REF_NAMES and not n.endswith("__cell25")]
    todo += [(n[:-8], n, False, "detail") for n in arms if n.endswith("__cell25") and n[:-8] in arms]
    for A, Bn, key, isclim in todo:
        lds = ["all"] + sorted(set(arms[A]["lead"].tolist()) & set(arms[Bn]["lead"].tolist()))
        for lead in lds:
            md = cfg["min_days_all"] if lead == "all" else cfg["min_days_lead"]
            gr = {gn: contrast(arms[A], arms[Bn], lead, k, g, T, cell, md, rng, station_stats=(lead == "all"))
                  for gn, k, g in groups}
            R["contrasts"].append(dict(A=A, B=Bn, key=key, clim=isclim is True, detail=isclim == "detail",
                                       lead=lead if lead == "all" else int(lead), groups=gr))
        log(f"  contrast {A} - {Bn} done")
    for c in R["contrasts"]:
        c["lead"] = c["lead"] if c["lead"] == "all" else int(c["lead"])
    R["truth"] = a.truth
    R["day"] = "local" if a.local else "UTC"
    tag = f"scores_st_{a.year}" + os.environ.get("GHAP_ST_TAG", "") + ("_local" if a.local else "") + ("_h12" if a.truth == "daily12" else "")
    (out / f"{tag}.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (out / f"{tag}.md").write_text(markdown(a.year, R))
    fig = figure(a.year, R, out, "_local" if a.local else "") if a.truth == "daily" else None
    log(f"wrote {out / tag}.json/.md, figure {fig}")


if __name__ == "__main__":
    main()
