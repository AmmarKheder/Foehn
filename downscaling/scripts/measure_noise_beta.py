#!/usr/bin/env python3
"""Fit of the power-law noise prior on LGHAP z-fields of TRAINING days (PLAN.md 4.2 "Noise-prior fitting"; smoke).

  * days: 5 per month of the smoke training window (days 3, 9, 15, 21, 27 of 2020-10 .. 2021-02 = 25 days), read from
    the LGHAP zips in memory (build_smoke_data.read_box, never extracted to disk);
  * z = (log1p(max(LGHAP, 1)) - mu_p) / sigma_p with the smoke climatology (training days only), on the mask;
  * up to 10 non-overlapping ~512 x 512 km boxes with >= 99.5 % valid pixels (the rest mean-filled), box mean removed,
    Hann window, radially averaged per-mode PSD (cdffm_aq.spectra), averaged over boxes and days;
  * beta = -slope of log PSD vs log k over 5-100 km wavelengths (the band of CD-FFM's fit, test_noise_kernel.py);
  * lmin (rule fixed before looking at the data): the shortest wavelength in [4, 8] km down to which the PSD, minus
    the white floor of the integer quantisation (mean of (1 / (1 + v))^2 / 12 / sigma^2), stays within a factor 2 of
    the fitted law; if the law already fails above 8 km, the longest failing wavelength in (8, 50] km. 4 km = two
    1 km pixels per half wave (no aliasing of the noise on the 1 km grid);
  * lmax = 500 km (CD-FFM): not measurable with 512 km boxes;
  * n_features = 1024 (CD-FFM); test_s0.py's stripe test may raise it to 4096 and records it here.
Output: data/processed/smoke/noise_fit_smoke.json (key "chosen" is what train_aq.py reads).

  python -I scripts/measure_noise_beta.py [--procs 8]          (world venv: netCDF4; partition small, S0 job)
"""
import argparse
import datetime as dt
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import build_smoke_data as bsd  # noqa: E402  (read_box, box constants)
from cdffm_aq.spectra import box_psd, band_slopes, default_kbins, find_boxes, fit_slope  # noqa: E402

STORE = ROOT / "data/processed/smoke"
SIGMA_FLOOR = 0.1
FIT_BAND_KM = (5.0, 100.0)
DAYS_OF_MONTH = (3, 9, 15, 21, 27)


def fit_days():
    first, n = bsd.TRAIN
    last = first + dt.timedelta(n - 1)
    out, d = [], first
    while d <= last:
        if d.day in DAYS_OF_MONTH:
            out.append(d)
        d += dt.timedelta(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(STORE))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--max-boxes", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    store = Path(a.store)
    out_path = Path(a.out) if a.out else store / "noise_fit_smoke.json"
    mask = np.load(store / "grid/lghap_mask.npy").astype(np.int64)
    i, j = mask // 36000, mask % 36000
    r, c = i - bsd.I0, j - bsd.J0
    TH, TW = bsd.TH, bsd.TW
    valid = np.zeros((TH, TW), bool)
    valid[r, c] = True
    row_lat = -57.995 + 0.01 * (bsd.I0 + np.arange(TH))
    cm = np.load(store / "lghap_clim/clim_mean_smoke.npy")
    cv = np.load(store / "lghap_clim/clim_var_smoke.npy")
    boxes = find_boxes(valid, row_lat, size_km=512.0, res_deg=0.01, min_frac=0.995, stride=50, max_boxes=a.max_boxes)
    print(f"{len(boxes)} boxes: " + "; ".join(f"rows {b[0]}+{b[2]} cols {b[1]}+{b[3]} lat {b[5]:.1f} "
                                             f"lon {-9.995 + 0.01 * (b[1] + b[3] / 2):.1f} valid {b[4]:.4f}"
                                             for b in boxes), flush=True)
    assert boxes, "no 512 km box inside the mask"
    kbins = default_kbins(512.0, 2.0, 40)
    days = fit_days()
    psd = np.zeros((len(boxes), len(kbins) - 1))
    npsd = np.zeros((len(boxes), len(kbins) - 1))          # contributions per bin (empty log bins at low k -> NaN)
    cnt = np.zeros(len(boxes))
    qfloor = np.zeros(len(boxes))
    zstats = []
    ctx = mp.get_context("fork")
    with ctx.Pool(a.procs) as pool:
        for d, tile, err in pool.imap_unordered(bsd.read_box, days):
            assert err < 1e-4
            v = tile[r, c].astype(np.float64)
            assert (v != bsd.FILL).all()
            ang = 2 * np.pi * ((d - dt.date(d.year, 1, 1)).days) / 365.0
            b = np.array([1, np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)])
            mu = cm.astype(np.float64) @ b
            sig = np.sqrt(np.maximum(cv.astype(np.float64) @ b, SIGMA_FLOOR ** 2))
            vv = np.maximum(v, 1.0)
            z = (np.log1p(vv) - mu) / sig
            zt = np.full((TH, TW), np.nan)
            zt[r, c] = z
            qt = np.full((TH, TW), np.nan)
            qt[r, c] = (1.0 / (1.0 + vv)) ** 2 / 12.0 / sig ** 2
            zstats.append((str(d), float(z.mean()), float(z.std())))
            for bi, bx in enumerate(boxes):
                _, p, _ = box_psd(zt, valid, bx, kbins)
                psd[bi] += np.nan_to_num(p)
                npsd[bi] += np.isfinite(p)
                r0, c0, nr, nc = bx[:4]
                qfloor[bi] += np.nanmean(qt[r0:r0 + nr, c0:c0 + nc])
                cnt[bi] += 1
            print(f"  {d}: z mean {z.mean():+.3f} std {z.std():.3f} ({time.time() - t0:.0f} s)", flush=True)
    psd = np.where(npsd > 0, psd / np.maximum(npsd, 1), np.nan)
    qfloor /= cnt
    k = np.sqrt(kbins[:-1] * kbins[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN (empty) bins stay NaN
        mean_psd = np.nanmean(psd, 0)
    q = float(qfloor.mean())
    fit = fit_slope(k, mean_psd, *FIT_BAND_KM)
    beta = -fit["slope"]
    per_box = [dict(lat=float(bx[5]), lon=float(-9.995 + 0.01 * (bx[1] + bx[3] / 2)), valid=float(bx[4]),
                    beta_5_100=-fit_slope(k, psd[bi], *FIT_BAND_KM)["slope"], quant_floor=float(qfloor[bi]))
               for bi, bx in enumerate(boxes)]
    law = np.exp(fit["intercept"]) * k ** fit["slope"]
    ratio = (mean_psd - q) / law
    lam = 1.0 / k
    ok = (ratio >= 0.5) & (ratio <= 2.0)
    # lmin rule (module docstring)
    lmin = 8.0
    small = np.where((lam <= 8.0) & (lam >= 4.0))[0]
    small = small[np.argsort(-lam[small])]            # from 8 km down to 4 km
    if all(ok[np.where((lam > 8.0) & (lam <= 50.0))[0]]):
        for idx in small:
            if not ok[idx]:
                break
            lmin = float(lam[idx])
        lmin = max(4.0, round(lmin, 1))
    else:
        bad = np.where((lam > 8.0) & (lam <= 50.0) & ~ok)[0]
        lmin = round(float(lam[bad].max()), 1)
    res = {
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "days": [str(d) for d in days],
        "boxes": per_box,
        "fit_band_km": FIT_BAND_KM,
        "fit": fit,
        "beta": beta,
        "beta_other_bands": {f"{lo:g}-{hi:g}km": -fit_slope(k, mean_psd, lo, hi)["slope"]
                             for lo, hi in ((8, 100), (10, 100), (5, 50), (16, 128))},
        "beta_per_box_mean_std": [float(np.mean([b["beta_5_100"] for b in per_box])),
                                  float(np.std([b["beta_5_100"] for b in per_box]))],
        "local_slopes": band_slopes(k, mean_psd, (2.3, 4, 8, 16, 32, 64, 128, 256)),
        "quantisation_floor_psd": q,
        "psd_over_law_minus_floor": {f"{l_:.1f}km": float(rt) for l_, rt in zip(lam, ratio) if np.isfinite(rt)},
        "lmin_rule": "shortest wavelength in [4, 8] km down to which (PSD - quantisation floor) / fitted law stays in "
                     "[0.5, 2] (longer than 8 km if the law fails in (8, 50] km)",
        "k_cycles_per_km": k.tolist(), "psd_mean": [None if not np.isfinite(x) else float(x) for x in mean_psd],
        "psd_per_box": [[None if not np.isfinite(x) else float(x) for x in row] for row in psd],
        "z_mean_std_per_day": zstats,
        "chosen": {"beta": round(beta, 3), "lmin_km": lmin, "lmax_km": 500.0, "n_features": 1024,
                   "source": "measure_noise_beta.py on smoke training days (LGHAP z-fields)"},
        "seconds": time.time() - t0,
    }
    if out_path.exists():                       # keep a stripe-test verdict recorded by test_s0.py, if any
        old = json.loads(out_path.read_text())
        if "stripe_test" in old:
            res["stripe_test"] = old["stripe_test"]
            res["chosen"]["n_features"] = old["chosen"].get("n_features", 1024)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(out_path)
    print(json.dumps({k_: res[k_] for k_ in ("fit", "beta", "beta_other_bands", "beta_per_box_mean_std",
                                              "local_slopes", "quantisation_floor_psd", "chosen")}, indent=1))
    print(f"-> {out_path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
