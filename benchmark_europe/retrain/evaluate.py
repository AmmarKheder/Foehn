#!/usr/bin/env python3
"""Full-domain evaluation of a retrained CRAN-PM variant with the paper protocol (Sec. 6): the whole
4192x6992 Europe grid is tiled with 512x512 local patches and stitched, then scored on every valid
GHAP land pixel and at the EEA stations, test year 2022, lead times T+1..T+4, next to persistence.

--baseline      scores a retrained 0.25 deg baseline (train_baselines.py), bilinearly upsampled to 1 km;
--cams          scores the CAMS EAC4 PM2.5 analysis of the target day (the paper's CAMS reference).
--wind shift  keeps every input true but drives the wind prior (token order / upwind bias) with the
             wind of a day 15 days away: no change in skill means the model does not use its wind prior.
--met shift  feeds the model the ERA5 + CAMS of a day 15 days away (same season, wrong weather):
             a model that uses the meteorology must get worse, one that copies today's GHAP will not.

One rank per GCD (srun), days split across ranks, each rank writes part_<rank>.npz; then
    python evaluate.py --merge <outdir>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import VARIANTS, RetrainModule  # noqa: E402
from train_baselines import CH, CW, BaselineModule, CoarseDatasetV2  # noqa: E402

from cranpm.data.europe_dataset_v2 import (  # noqa: E402
    ELEV_MEAN, ELEV_STD, ERA5_H, ERA5_RES, ERA5_W, GHAP_C0, GHAP_H, GHAP_MEAN, GHAP_R0, GHAP_RES, GHAP_STD,
    GHAP_W, LAT_NORTH, LON_WEST, EuropeDatasetV2, _bilinear, wind10)
from cranpm.inference.tiling import iter_tiles, stitch_tiles  # noqa: E402

PS = 512


def ghap_day(ds, y, d):
    return np.asarray(ds.ghap[y][d, GHAP_R0:GHAP_R0 + GHAP_H, GHAP_C0:GHAP_C0 + GHAP_W], np.float32)


def global_input(ds, y, d, met_day):
    return ds.global_stack(y, met_day)


def upsample(coarse):
    """(168, 280) field on ERA5 points -> (4192, 6992) GHAP pixel centres, bilinear, edges clamped."""
    def axis(n, m):
        x = np.clip((np.arange(n) + 0.5) / 25.0, 0, m - 1)
        i0 = np.minimum(np.floor(x).astype(int), m - 2)
        return i0, (x - i0).astype(np.float32)
    r0, fr = axis(GHAP_H, coarse.shape[0])
    c0, fc = axis(GHAP_W, coarse.shape[1])
    rows = (1 - fr)[:, None] * coarse[r0] + fr[:, None] * coarse[r0 + 1]
    return (1 - fc)[None] * rows[:, c0] + fc[None] * rows[:, c0 + 1]


HRES_DIR = Path(__file__).resolve().parents[1] / "dejavu" / "data" / "hres"


def load_hres(ds):
    """(d, h) -> (31, 169, 281) HRES forecast block in the dataset's standardisation (30 ERA5 channels + rain)."""
    table = {}
    for f in sorted(HRES_DIR.glob("*.npz")):
        with np.load(f) as m:
            blk, days = m["block"], m["day_of_year"]
        table.update({int(dd): blk[k] for k, dd in enumerate(days)})
    em, es = ds.era5_mean[:, None, None], ds.era5_std[:, None, None]
    xm, xs = ds.extra_mean[3], ds.extra_std[3]

    def get(d, h):
        raw = table[int(d)][h - 1].astype(np.float32)
        tp = (np.log1p(np.maximum(raw[30], 0) * 1000.0) - xm) / xs
        return np.concatenate([(raw[:30] - em) / es, tp[None]]).astype(np.float32)
    return get


def coarsen(a):
    """1 km (4192 x 6992) -> 0.25 deg (168 x 280): mean of the 25 x 25 block of pixels centred on each ERA5 point,
    NaN pixels ignored (same blocks as ghap_025deg: 1 km rows 25j-12 .. 25j+12, cols 25c-12 .. 25c+12)."""
    p = np.full((168 * 25, 280 * 25), np.nan, np.float32)
    p[12:, 12:] = a[:168 * 25 - 12, :280 * 25 - 12]
    with np.errstate(all="ignore"):
        return np.nanmean(p.reshape(168, 25, 280, 25), axis=(1, 3))


def cranpm_predictor(args, dev):
    mode = VARIANTS[args.variant][0]
    module = RetrainModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    model = module.model.to(dev).eval()
    takes_wind = module.config["model"]["arch"] in ("order", "upwind")
    # wind prior: the one the model was trained with, or (--wind shift) that of a day 15 days away
    fcst = module.config["data"].get("fcst", False)
    ds = EuropeDatasetV2([args.year], mode=mode, horizons=args.horizons, train=False,
                         wind_shift=15 if args.wind == "shift" else module.config["data"].get("wind_shift", 0),
                         extra=module.config["data"].get("extra", False), fcst=fcst)
    y = args.year
    hres = load_hres(ds) if (fcst and args.weather == "hres") else None

    def forecast_block(d, met_day, h):
        if hres is not None:  # the real IFS HRES forecast issued at 00 UTC of day d, for day d + h
            return hres(d, h)
        return ds.fcst_block(y, met_day + h)  # perfect forecast (ERA5)
    tiles = list(iter_tiles(stride=args.stride))
    static = []
    for t in tiles:
        rows = LAT_NORTH - GHAP_RES * np.arange(t.row, t.row + PS, dtype=np.float32)
        cols = LON_WEST + GHAP_RES * np.arange(t.col, t.col + PS, dtype=np.float32)
        clat, clon = LAT_NORTH - (t.row + PS / 2) * GHAP_RES, LON_WEST + (t.col + PS / 2) * GHAP_RES
        static.append(dict(
            elev=ds._elev_hires(t.row, t.col),
            lat=np.repeat(((rows - 30.0) / 42.0)[:, None], PS, 1), lon=np.repeat(((cols + 25.0) / 70.0)[None, :], PS, 0),
            center=(clat, clon),
            er=int(np.clip((LAT_NORTH - clat) / ERA5_RES, 0, ERA5_H - 1)),
            ec=int(np.clip((clon - LON_WEST) / ERA5_RES, 0, ERA5_W - 1))))
    elev_coarse = torch.from_numpy((ds.elev_coarse - ELEV_MEAN) / ELEV_STD).to(dev)

    def predict(d, met_day, hs, g_t, g_p):
        era5, store_t = global_input(ds, y, d, met_day)
        wind_b = torch.from_numpy(wind10(ds._era5_store(y, ds.wind_day(y, met_day)))).to(dev)
        out = {}
        for h in hs:
            era5_b = torch.from_numpy(np.concatenate([era5, forecast_block(d, met_day, h)]) if fcst else era5).to(dev)
            preds = []
            for i in range(0, len(tiles), args.batch):
                chunk = list(range(i, min(i + args.batch, len(tiles))))
                local, elev_h, center, wap = [], [], [], []
                for k in chunk:
                    t, s = tiles[k], static[k]
                    sl = np.s_[t.row:t.row + PS, t.col:t.col + PS]
                    local.append(np.stack([(g_t[sl] - GHAP_MEAN) / GHAP_STD, s["elev"], s["lat"], s["lon"],
                                           (g_p[sl] - GHAP_MEAN) / GHAP_STD]))
                    elev_h.append(s["elev"])
                    center.append(s["center"])
                    wap.append((store_t[0, s["er"], s["ec"]], store_t[1, s["er"], s["ec"]]))
                n = len(chunk)
                kw = dict(era5=era5_b.expand(n, -1, -1, -1), elevation_coarse=elev_coarse.expand(n, -1, -1),
                          ghap_patch=torch.from_numpy(np.stack(local).astype(np.float32)).to(dev),
                          elevation_hires=torch.from_numpy(np.stack(elev_h)).to(dev),
                          lead_time=torch.full((n,), float(h), device=dev),
                          patch_center=torch.tensor(center, dtype=torch.float32, device=dev),
                          wind_at_patch=torch.tensor(np.array(wap), dtype=torch.float32, device=dev))
                if takes_wind:
                    kw["wind"] = wind_b.expand(n, -1, -1, -1)
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    p = model(**kw)
                p = p.float().cpu().numpy()[:, 0] * GHAP_STD + GHAP_MEAN
                preds += [(tiles[k], p[j]) for j, k in enumerate(chunk)]
            out[h] = stitch_tiles(preds)
        return out

    return ds, predict


REGION = (2250, 2500, 768, 1536)   # regional study window (rows, cols from 72 N / 25 W; size), ../regional/regional.py
FSS_SCALES = (1, 5, 11, 25)        # km (odd windows)


def boxsum(a, k):
    """Sum of a over the k x k window centred on each pixel (zero outside), same shape as a."""
    p = k // 2
    c = np.pad(a.astype(np.float64), ((p + 1, p), (p + 1, p))).cumsum(0).cumsum(1)
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def scale_aware(pred, tgt, m, crop, acc, h):
    """Metrics that do not penalise a correctly forecast detail twice: Fractions Skill Score of exceedance of 25 ug/m3
    (Roberts & Lean 2008) at several window sizes, the error of the sub-0.25-deg detail (anomaly from the 25 x 25 block
    mean) against the 'no detail' reference, and the pixel CSI at 1 km."""
    ap = pred - upsample(np.nan_to_num(coarsen(np.where(m, pred, np.nan))))
    at = tgt - upsample(np.nan_to_num(coarsen(np.where(m, tgt, np.nan))))
    P, T, M, AP, AT = (x[crop] for x in (pred, tgt, m, ap, at))
    acc["fine_se"][h] += float(((AP - AT)[M].astype(np.float64) ** 2).sum()); acc["fine_ref"][h] += float((AT[M].astype(np.float64) ** 2).sum())
    ef, eo = (P > 25) & M, (T > 25) & M
    acc["loc_hit"][h] += float((ef & eo).sum()); acc["loc_uni"][h] += float((ef | eo).sum())
    nv = M.astype(np.float32)
    for k in FSS_SCALES:
        n = np.maximum(boxsum(nv, k), 1)
        ff, fo = boxsum(ef.astype(np.float32), k) / n, boxsum(eo.astype(np.float32), k) / n
        acc[f"fss_num_{k}"][h] += float(((ff - fo)[M] ** 2).sum()); acc[f"fss_den_{k}"][h] += float((ff[M] ** 2 + fo[M] ** 2).sum())


def fine_anomaly(g_t):
    """Today's sub-grid pattern at 1 km: GHAP today minus its 0.25 deg block means interpolated back to 1 km (0 at sea).
    Adding it to an interpolated 0.25 deg forecast = persistence of the 1 km detail (cities, valleys)."""
    land = g_t > 0
    c = np.nan_to_num(coarsen(np.where(land, g_t, np.nan)))
    return np.where(land, g_t - upsample(c), 0.0).astype(np.float32)


def to_1km(p, g_t, pattern):
    """(168, 280) forecast in ug/m3 -> 1 km: bilinear, optionally + today's sub-grid pattern, clipped at 0."""
    out = upsample(p)
    if pattern == "add":
        out = np.maximum(out + fine_anomaly(g_t), 0.0)
    return out.astype(np.float32)


def crps_members(E, T):
    """Fair-free ensemble CRPS summed over pixels: mean|E - T| - 0.5 mean|Ei - Ej|, pairwise term by sorting. E (M, N)."""
    M = E.shape[0]
    Es = np.sort(E, axis=0)
    k = np.arange(M, dtype=np.float32)[:, None]
    pair = (2 * (2 * k - M + 1) * Es).sum(0) / max(M * (M - 1), 1)   # = mean over i != j of |Ei - Ej|
    return float((np.abs(E - T).mean(0) - 0.5 * pair).sum())


def baseline_predictor(args, dev):
    """A 0.25 deg baseline (train_baselines.py) brought to 1 km by bilinear interpolation (+ today's 1 km pattern with
    --fine-pattern add). met_day target31 models get the forecast-day block; --weather hres swaps in the real HRES."""
    ck = Path(args.ckpt)
    if ck.is_dir():  # a run's ckpt directory -> its best-validation checkpoint
        ck = sorted(ck.glob("best-*.ckpt"))[0]
    model = BaselineModule.load_from_checkpoint(ck, map_location="cpu").to(dev).eval()
    met_day = model.hparams.get("met_day", "none")
    ds = CoarseDatasetV2([args.year], mode="fixed", horizons=args.horizons, train=False, extra=met_day != "none")
    ds.met_day = met_day
    if met_day == "target31" and args.weather == "hres":
        from eval_coarse import swap_forecast_block
        swap_forecast_block(ds, "hres", args.year)
    index = {(d, h): n for n, (_, d, h) in enumerate(ds.samples)}

    def predict(d, met_day_, hs, g_t, g_p):
        hs = [h for h in hs if (d, h) in index]
        if not hs:
            return {}
        items = [ds[index[(d, h)]] for h in hs]
        batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in items[0]}
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            p = model(batch).float().cpu().numpy()[:, 0] * GHAP_STD + GHAP_MEAN
        return {h: to_1km(p[i], g_t, args.fine_pattern) for i, h in enumerate(hs)}

    return ds, predict


def grid_predictor(args, dev):
    """A precomputed 0.25 deg forecast (365, 4, 168, 280) in ug/m3 indexed by start day of year and lead (the CAMS
    operational forecast, Aurora), brought to 1 km like the baselines."""
    arr = np.load(args.grid_forecast, mmap_mode="r")
    ds = EuropeDatasetV2([args.year], mode="fixed", horizons=args.horizons, train=False)

    def predict(d, met_day, hs, g_t, g_p):
        out = {}
        for h in hs:
            if d < arr.shape[0] and np.isfinite(arr[d, h - 1]).all():
                out[h] = to_1km(np.asarray(arr[d, h - 1], np.float32), g_t, args.fine_pattern)
        return out

    return ds, predict


def dejavu_predictor(args, dev):
    """The 0.25 deg flow-matching model (../dejavu, e.g. runs/noise_t31_fpstab): M members, ensemble mean brought to
    1 km like the baselines (bilinear, optionally + today's 1 km pattern); members kept for the 1 km CRPS."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dejavu"))
    from dejavu import DVData, DVModule, attach_hres, MU, SD
    run = Path(args.dejavu)
    model = DVModule.load_from_checkpoint(run / "final.ckpt", map_location="cpu").to(dev).eval()
    hp = model.hparams
    t31, fpr = hp.get("t31", False), hp.get("footprint", False)
    dv = DVData([args.year], "noise", train=False, t31=t31, footprint=fpr, fp_met=hp.get("fp_met", "blh"),
                blh_persist=hp.get("blh_persist", False))
    if args.weather == "hres":
        attach_hres(dv, t31, fpr)
    index = {(i, h): n for n, (i, h) in enumerate(dv.samples)}
    off = int(np.where(dv.year == args.year)[0][0])
    ds = EuropeDatasetV2([args.year], mode="fixed", horizons=args.horizons, train=False)
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    M, S = args.members, args.steps1
    extra = {k: np.zeros(max(args.horizons) + 1) for k in ("crps_n", "crps", "spread2")}

    def predict(d, met_day, hs, g_t, g_p):
        out = {}
        tgt_day = lambda h: ghap_day(ds, args.year, d + h)
        for h in hs:
            if (off + d, h) not in index:
                continue
            it = dv[index[(off + d, h)]]
            cond = it["cond"][None].to(dev).expand(M, -1, -1, -1)
            lead = torch.full((M,), float(h), device=dev)
            fp = tuple(it[k][None].to(dev).expand(M, *([-1] * it[k].dim())) for k in ("wind", "rain", "stab") if k in it) if fpr else None
            x0 = torch.randn(M, 1, *it["x1"].shape[-2:], device=dev, generator=gen)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                E = model.sample(x0, cond, lead, S, fp)[:, 0].float().cpu().numpy() * SD + MU     # (M, 168, 280)
            members = np.stack([to_1km(e, g_t, args.fine_pattern) for e in E])
            tgt = tgt_day(h); m = np.isfinite(tgt) & (tgt > 0)
            Em, T = members[:, m], tgt[m]
            extra["crps_n"][h] += m.sum(); extra["crps"][h] += crps_members(Em, T); extra["spread2"][h] += Em.var(0).sum()
            out[h] = members.mean(0)
        return out

    predict.extra = extra
    return ds, predict


def cams_predictor(args, dev):
    ds = EuropeDatasetV2([args.year], mode="fixed", horizons=args.horizons, train=False)
    y = args.year
    lat = LAT_NORTH - GHAP_RES * (np.arange(GHAP_H) + 0.5)
    lon = LON_WEST + GHAP_RES * (np.arange(GHAP_W) + 0.5)

    def predict(d, met_day, hs, g_t, g_p):  # analysis of the target day, not a forecast
        return {h: _bilinear(np.asarray(ds.eac4[y]["pm25"][d + h], np.float32) * 1e9, (90 - lat) / 0.75,
                             (lon % 360) / 0.75).astype(np.float32) for h in hs}

    return ds, predict


def run(args):
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    dev = torch.device(f"cuda:{int(os.environ.get('SLURM_LOCALID', 0))}" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cascade:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascade"))
        from cascade import cascade_predictor
        args.ckpt = args.cascade
    if args.full:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fullimage"))
        from fullflow import full_predictor
        args.ckpt = args.full
    if args.regional:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regional"))
        from regional import regional_predictor
        args.ckpt = args.regional
    make = (regional_predictor if args.regional else full_predictor if args.full else cascade_predictor if args.cascade
            else cams_predictor if args.cams else dejavu_predictor if args.dejavu else grid_predictor if args.grid_forecast
            else baseline_predictor if args.baseline
            else cranpm_predictor)
    ds, predict = make(args, dev)
    y = args.year
    by_day = {}
    for _, d, h in ds.samples:
        by_day.setdefault(d, []).append(h)
    days = sorted(by_day)[::args.day_stride]
    if args.doy_ranges:  # e.g. "72-80,190-205": transport episodes
        rng_ = [tuple(map(int, r.split("-"))) for r in args.doy_ranges.split(",")]
        days = [d for d in days if any(lo <= d <= hi for lo, hi in rng_)]
    days = days[rank::world]

    st_r, st_c = np.floor(ds.st_row).astype(int), np.floor(ds.st_col).astype(int)
    st_in = np.where((st_r >= 0) & (st_r < GHAP_H) & (st_c >= 0) & (st_c < GHAP_W))[0]
    H = max(args.horizons) + 1
    acc = {k: np.zeros(H) for k in ("n", "sse", "sae", "se", "sse_p", "sae_p", "se_p", "st", "st2",
                                     "nc", "ssec", "ssepc", "hit", "uni", "hitp", "unip",
                                     "fine_se", "fine_ref", "loc_hit", "loc_uni")}
    acc.update({f"fss_{w}_{k}": np.zeros(H) for w in ("num", "den") for k in FSS_SCALES})
    region = np.zeros((GHAP_H, GHAP_W), bool)
    r0, c0, nr, nc = REGION
    region[r0:r0 + nr, c0:c0 + nc] = True
    crop = (slice(r0, r0 + nr), slice(c0, c0 + nc))   # scale-aware metrics always on the regional window (cost)
    jr, ic = np.arange(168) * 25, np.arange(280) * 25    # 0.25 deg cells whose 25 x 25 block lies inside the window
    cregion = ((jr - 12 >= r0) & (jr + 12 < r0 + nr))[:, None] & ((ic - 12 >= c0) & (ic + 12 < c0 + nc))[None]
    if args.region:
        st_in = st_in[region[st_r[st_in], st_c[st_in]]]
    stations = []  # (station, day, h, pred, persistence, ghap_target, obs)
    daily = []  # (day, h, pixels, sse, sse_persistence, sum_err): per-day scores for paired block bootstrap / DM tests
    saved = {}

    for d in days:
        met_day = d
        if args.met == "shift":
            met_day = d + 15 if d + 15 < ds.ndays[y] else d - 15
        g_t, g_p = np.nan_to_num(ghap_day(ds, y, d)), np.nan_to_num(ghap_day(ds, y, d - 1))
        for h, pred in predict(d, met_day, by_day[d], g_t, g_p).items():
            tgt = ghap_day(ds, y, d + h)
            m = np.isfinite(tgt) & (tgt > 0)
            if args.region:
                m &= region
            e, ep, tv = (pred - tgt)[m], (g_t - tgt)[m], tgt[m]
            for k, v in (("n", m.sum()), ("sse", (e ** 2).sum()), ("sae", np.abs(e).sum()), ("se", e.sum()),
                         ("sse_p", (ep ** 2).sum()), ("sae_p", np.abs(ep).sum()), ("se_p", ep.sum()),
                         ("st", tv.sum()), ("st2", (tv.astype(np.float64) ** 2).sum())):
                acc[k][h] += float(v)
            # the same forecast at 0.25 deg (25 x 25 block means on land), comparable with the coarse models
            tc = coarsen(np.where(m, tgt, np.nan)); pc = coarsen(np.where(m, pred, np.nan)); gc = coarsen(np.where(m, g_t, np.nan))
            mc = np.isfinite(tc) & np.isfinite(pc) & (cregion if args.region else True)
            scale_aware(np.nan_to_num(pred), np.nan_to_num(tgt), m, crop, acc, h)
            for k, v in (("nc", mc.sum()), ("ssec", ((pc - tc)[mc] ** 2).sum()), ("ssepc", ((gc - tc)[mc] ** 2).sum()),
                         ("hit", ((tc > 25) & (pc > 25))[mc].sum()), ("uni", ((tc > 25) | (pc > 25))[mc].sum()),
                         ("hitp", ((tc > 25) & (gc > 25))[mc].sum()), ("unip", ((tc > 25) | (gc > 25))[mc].sum())):
                acc[k][h] += float(v)
            daily.append((d, h, float(m.sum()), float((e.astype(np.float64) ** 2).sum()),
                          float((ep.astype(np.float64) ** 2).sum()), float(e.sum())))

            if y in ds.st_daily and d + h < ds.st_daily[y].shape[0]:
                obs = ds.st_daily[y][d + h, st_in]
                ok = np.isfinite(obs) & (obs >= 0)
                idx, r, c = st_in[ok], st_r[st_in[ok]], st_c[st_in[ok]]
                stations.append(np.stack([idx, np.full(len(idx), d), np.full(len(idx), h),
                                          pred[r, c], g_t[r, c], tgt[r, c], obs[ok]], 1))
            if d in args.save_days:
                saved[f"pred_d{d}_h{h}"] = pred.astype(np.float16)
        print(f"rank {rank}: day {d} done ({len(by_day[d])} horizons)", flush=True)

    np.savez(out / f"part_{rank}.npz", stations=np.concatenate(stations) if stations else np.zeros((0, 7)),
             daily=np.array(daily, np.float64).reshape(-1, 6),
             **{f"acc_{k}": v for k, v in acc.items()}, **{f"acc_{k}": v for k, v in getattr(predict, "extra", {}).items()}, **saved)


def merge(outdir):
    out = Path(outdir)
    parts = [np.load(p) for p in sorted(out.glob("part_*.npz"))]
    acc = {k[4:]: sum(p[k] for p in parts) for k in parts[0].files if k.startswith("acc_")}
    st = np.concatenate([p["stations"] for p in parts])
    daily = np.concatenate([p["daily"] for p in parts])
    np.savez(out / "records.npz", daily=daily[np.lexsort((daily[:, 1], daily[:, 0]))], stations=st.astype(np.float32))
    res = {}
    for h in range(1, len(acc["n"])):
        n = acc["n"][h]
        if n == 0:
            continue
        var = acc["st2"][h] / n - (acc["st"][h] / n) ** 2
        s = st[st[:, 2] == h]
        ghap_ok = np.isfinite(s[:, 5])
        res[f"T+{h}"] = {
            "grid": {"pixels": int(n),
                     "model": {"rmse": np.sqrt(acc["sse"][h] / n), "mae": acc["sae"][h] / n, "bias": acc["se"][h] / n,
                               "r2": 1 - acc["sse"][h] / n / var},
                     "persistence": {"rmse": np.sqrt(acc["sse_p"][h] / n), "mae": acc["sae_p"][h] / n,
                                     "bias": acc["se_p"][h] / n, "r2": 1 - acc["sse_p"][h] / n / var}},
            "stations": {"obs": int(len(s)),
                         "model_rmse": float(np.sqrt(np.mean((s[:, 3] - s[:, 6]) ** 2))),
                         "persistence_rmse": float(np.sqrt(np.mean((s[:, 4] - s[:, 6]) ** 2))),
                         "ghap_target_rmse": float(np.sqrt(np.mean((s[ghap_ok, 5] - s[ghap_ok, 6]) ** 2))),
                         "model_bias": float(np.mean(s[:, 3] - s[:, 6]))},
            **({} if "crps_n" not in acc or acc["crps_n"][h] == 0 else {"ensemble": {
                "crps": float(acc["crps"][h] / acc["crps_n"][h]), "spread": float(np.sqrt(acc["spread2"][h] / acc["crps_n"][h]))}}),
            **({} if "fine_ref" not in acc or acc["fine_ref"][h] == 0 else {"scale_aware_region": {
                "fine_detail_skill": float(1 - acc["fine_se"][h] / acc["fine_ref"][h]),
                "csi25_1km": float(acc["loc_hit"][h] / max(acc["loc_uni"][h], 1)),
                **{f"fss25_{k}km": float(1 - acc[f"fss_num_{k}"][h] / max(acc[f"fss_den_{k}"][h], 1e-9)) for k in FSS_SCALES}}}),
            **({} if "nc" not in acc else {"coarse_025deg": {"rmse": float(np.sqrt(acc["ssec"][h] / max(acc["nc"][h], 1))),
                              "persistence_rmse": float(np.sqrt(acc["ssepc"][h] / max(acc["nc"][h], 1))),
                              "csi25": float(acc["hit"][h] / max(acc["uni"][h], 1)),
                              "csi25_persistence": float(acc["hitp"][h] / max(acc["unip"][h], 1))}})}
    res = json.loads(json.dumps(res, default=float))
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(f"{'lead':6s}{'grid RMSE':>11s}{'persist':>9s}{'R2':>7s}{'bias':>7s} | {'EEA RMSE':>9s}{'persist':>9s}{'GHAP':>7s}")
    for k, v in res.items():
        g, s = v["grid"], v["stations"]
        print(f"{k:6s}{g['model']['rmse']:11.2f}{g['persistence']['rmse']:9.2f}{g['model']['r2']:7.3f}"
              f"{g['model']['bias']:7.2f} | {s['model_rmse']:9.2f}{s['persistence_rmse']:9.2f}{s['ghap_target_rmse']:7.2f}")
        if "scale_aware_region" in v:
            a = v["scale_aware_region"]
            print(f"       detail skill {a['fine_detail_skill']:+.3f} | CSI25 1km {a['csi25_1km']:.3f} | FSS25 "
                  + " ".join(f"{k}km {a[f'fss25_{k}km']:.3f}" for k in FSS_SCALES))
        if "coarse_025deg" in v or "ensemble" in v:
            c, e = v.get("coarse_025deg", {}), v.get("ensemble", {})
            print(f"       0.25deg RMSE {c.get('rmse', float('nan')):.2f} (persist {c.get('persistence_rmse', float('nan')):.2f}) "
                  f"CSI25 {c.get('csi25', float('nan')):.3f} | CRPS 1km {e.get('crps', float('nan')):.2f} spread {e.get('spread', float('nan')):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", metavar="OUTDIR")
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--baseline", action="store_true", help="--ckpt is a train_baselines.py checkpoint")
    ap.add_argument("--cams", action="store_true", help="score the CAMS EAC4 analysis (no checkpoint)")
    ap.add_argument("--cascade", metavar="RUN", help="score the 1 km cascade (cascade/cascade.py run dir or final.ckpt)")
    ap.add_argument("--dejavu", metavar="RUN", help="score a 0.25 deg flow run of ../dejavu (e.g. ../dejavu/runs/noise_t31_fpstab)")
    ap.add_argument("--fine-pattern", choices=["none", "add"], default="none",
                    help="0.25 deg models: add today's 1 km sub-grid pattern to the interpolated forecast")
    ap.add_argument("--grid-forecast", metavar="NPY", help="precomputed 0.25 deg forecast (365, 4, 168, 280), ug/m3 (CAMS, Aurora)")
    ap.add_argument("--regional", metavar="RUN", help="score a ../regional/regional.py model (use with --region)")
    ap.add_argument("--region", action="store_true", help="score only the regional window (pixels, stations, 0.25 deg cells)")
    ap.add_argument("--full", metavar="RUN", help="score the whole-image 1 km model (fullimage/fullflow.py run dir or final.ckpt)")
    ap.add_argument("--coarse-ckpt", default=str(Path(__file__).resolve().parents[1] / "dejavu/runs/noise_t31_fpstab/final.ckpt"))
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--member-batch", type=int, default=1, help="maximum whole-image ensemble members per GPU batch")
    ap.add_argument("--steps1", type=int, default=8)
    ap.add_argument("--steps2", type=int, default=8)
    ap.add_argument("--cond-noise", type=float, default=0.0)
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--day-stride", type=int, default=1)
    ap.add_argument("--doy-ranges", default="", help="restrict start days to day-of-year ranges, e.g. 72-80,190-205")
    ap.add_argument("--stride", type=int, default=512, help="tile stride (512 = no overlap, last tile anchored)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--met", choices=["real", "shift"], default="real")
    ap.add_argument("--weather", choices=["era5", "hres"], default="era5", help="forecast-day block source (fcst models)")
    ap.add_argument("--wind", choices=["trained", "shift"], default="trained",
                    help="shift: at inference, the wind prior (order / upwind bias) of a day 15 days away")
    ap.add_argument("--save-days", type=int, nargs="*", default=[316])  # 2022-11-13, Po Valley figure
    a = ap.parse_args()
    merge(a.merge) if a.merge else run(a)
