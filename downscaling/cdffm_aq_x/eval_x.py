"""cdffm_aq_x.eval_x -- cdffm_aq.eval_aq for every variant, plus block-mean (joint-structure) scores.

Same protocol as eval_aq.main (hold-out days, all mask pixels, M members x --ode-steps Euler, fair CRPS / RMSE /
spread-skill / thresholds in ug m-3 and log1p, climatology and CAMS baselines, spectra, contact sheet), with
  * the members drawn through model_x.Sampler (for a base/estab run the generator use is identical to eval_aq, so
    the baseline's eval_aq numbers are reproduced; test_x checks it on the toy run);
  * BlockScores at 1, 2 and 4 encoder cells (0.25, 0.5, 1 deg) for the model, the iid climatology ensemble, the
    climatology median and CAMS bilinear: fair CRPS, RMSE and spread-skill of block means (data_x);
  * --prior-file: a fitted conditional copula prior (copula_x) replacing the stationary fine noise at inference;
  * --pixel-stride S > 1: every S-th mask pixel only (cost probes and cheap checks on the global store; spectra, maps
    and contact sheets need full fields and are skipped). S = 1 (default) is the unchanged full evaluation.
  * --psd-boxes-only: only the pixels of the spectra boxes (find_boxes), spectra on every day (stop-rule probe of the
    global run: member/LGHAP power ratio at 10-140 km); scores then refer to those pixels only.
  * --z-bound B / --pmax-ug P (PROTOCOL.md A1(e)): flow state kept in [-B, B] (model_aq.integrate) and members capped at
    P ug m-3 (data_aq.z_to_ug); the sanity block of metrics.json counts the pixels with a member above 440 ug m-3 (the
    LGHAP maximum of the training days) and above 1,000, the member points at the bound / cap, the largest member value,
    and extremes.npz maps the count of pixels with a member > 440 per encoder cell (the A1(e) gates G0a-G0e).
Runs of train_aq (runs/) and train_x (runs/x/) are both accepted (--runs-root).

  srun -n 8 ... python -u -m cdffm_aq_x.eval_x --run NAME --runs-root ... --ckpt last --members 8 --ode-steps 10
"""
import argparse
import dataclasses
import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from cdffm_aq.data_aq import (PixelArrays, Scores, assemble_inputs, bilinear_log1p_at, doy_of, open_store,
                              raw_to_device, store_kwargs, z_to_ug)
from cdffm_aq.eval_aq import dist_setup, save_contact_sheet, save_maps, tile_of
from cdffm_aq.spectra import box_psd, default_kbins, find_boxes
from cdffm_aq.train_aq import noise_spec, model_grid
from .train_x import coarse_spec

from . import RUNS_X
from .data_x import BlockScores, CellGeometry, CellMeans
from .model_x import Sampler, load_run_x

LGHAP_TRAIN_MAX = 440.0          # largest LGHAP value of the training-day point files (2019-07-11 .. 2022-12-31)
ABSURD_UG = 1000.0               # physical cap of the members in A1(e) (P_max)


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-root", default=str(RUNS_X))
    ap.add_argument("--ckpt", default="last", help="last | best (lowest validation CRPS) | STEP")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--days", default=None, help="comma-separated hold-out day indices (default: all)")
    ap.add_argument("--map-dates", default="", help="comma-separated dates for map PNGs ('' = none)")
    ap.add_argument("--zoom", default="45.4,9.5,300,420", help="lat,lon,rows,cols of the 1 km map window (Po Valley)")
    ap.add_argument("--psd-every", type=int, default=3)
    ap.add_argument("--contact-dates", default="", help="comma-separated dates for ONE contact-sheet PNG ('' = none)")
    ap.add_argument("--contact-stride", type=int, default=3)
    ap.add_argument("--prior-file", default=None, help="copula_x fitted prior (.pt) replacing the stationary noise")
    ap.add_argument("--pixel-stride", type=int, default=1, help="> 1: every S-th mask pixel only (no spectra/maps)")
    ap.add_argument("--psd-boxes-only", action="store_true", help="only the pixels of the spectra boxes, PSD every day")
    ap.add_argument("--psd-centres", default=None, help="lat:lon,lat:lon,... centres of ~512 km spectra boxes replacing "
                    "find_boxes (e.g. the 6 regime boxes of PROTOCOL.md Amendment A1(c))")
    ap.add_argument("--z-bound", type=float, default=None, help="flow state kept in [-B, B] (PROTOCOL.md A1(e))")
    ap.add_argument("--pmax-ug", type=float, default=None, help="physical cap of the members in ug m-3 (A1(e))")
    ap.add_argument("--z0-trunc", type=float, default=None, help="base noise clamped to [-T, T] before the flow (A1(g))")
    ap.add_argument("--stations", default="auto", help="A3(c): auto = the run's assimilation root (args.stations) with the "
                    "day's monitors; none = every monitor hidden (pmS_nostations counterfactual); DIR = another root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    t0 = time.time()
    rank, world, dev = dist_setup(a.device)
    rargs, model, step = load_run_x(a.run, dev, a.ckpt, a.runs_root)
    variant = getattr(rargs, "variant", "base")
    spec = noise_spec(rargs)
    store = open_store(rargs.store, **store_kwargs(rargs))
    pix = PixelArrays(store, dev)
    TMAX = float(getattr(store, "train_max_ug", LGHAP_TRAIN_MAX))   # G0a threshold: 440 (LGHAP) or T_G (GHAP, A2(c))
    holes = bool(getattr(store, "has_holes", False))               # GHAP: hold-out -1 = hole, zero weight (A2(a))
    grid = model_grid(rargs.store, rargs.grid)
    geom = CellGeometry(store, grid)
    cm = CellMeans(geom, dev) if variant == "hcf" else None
    flowmap = rargs.objective == "flowmap"
    if rargs.objective == "regression":
        raise SystemExit("eval_x scores generative runs only (use eval_aq for the regression twin)")
    M = a.members
    prior = None
    if a.prior_file:
        from .copula_x import load_prior
        prior = load_prior(a.prior_file, model, spec, M, dev)
    days = list(range(len(store.hold_dates))) if a.days is None else [int(x) for x in a.days.split(",")]
    mine = days[rank::world]
    map_dates = {dt.date.fromisoformat(s) for s in a.map_dates.split(",") if s}
    contact_dates = {dt.date.fromisoformat(s) for s in a.contact_dates.split(",") if s}
    contact = []
    zoom = tuple(float(x) if i < 2 else int(x) for i, x in enumerate(a.zoom.split(","))) if a.zoom else None
    tag = a.tag or (f"x_{a.ckpt}_M{M}_s{a.ode_steps}" + (f"_{Path(a.prior_file).stem}" if a.prior_file else ""))
    out_dir = Path(a.runs_root) / a.run / f"eval_{tag}"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval_x] run={a.run} variant={variant} step={step} M={M} ode_steps={a.ode_steps} days={len(days)} "
              f"world={world} noise={spec} prior={a.prior_file} z_bound={a.z_bound} pmax_ug={a.pmax_ug} "
              f"z0_trunc={a.z0_trunc} -> {out_dir}",
              flush=True)
    names = ["model", "climatology_median", "climatology_ens", "cams_bilinear"]
    msize = {"model": M, "climatology_median": 1, "climatology_ens": max(M, 2), "cams_bilinear": 1}
    tot = {n: Scores(msize[n], dev) for n in names}
    blocks = {n: BlockScores(msize[n], geom, dev) for n in names}
    daily = []
    valid_tile = ~np.isnan(tile_of(store, np.zeros(store.n_pix, np.float32)))
    boxes = find_boxes(valid_tile, store.tile_row_lat, res_deg=store.res_deg, max_boxes=6)
    if a.psd_centres:                           # fixed boxes centred on given points (box_shape of their latitude)
        from cdffm_aq.spectra import box_shape
        boxes = []
        for pair in a.psd_centres.split(","):
            la, lo = (float(v) for v in pair.split(":"))
            nr, nc, _, _ = box_shape(la, 512.0, store.res_deg)
            r = int(round((la - store.tile_row_lat[0]) / store.res_deg))
            c = int(round((lo - store.tile_col_lon0) / store.res_deg))
            r0 = int(np.clip(r - nr // 2, 0, store.tile_shape[0] - nr))
            c0 = int(np.clip(c - nc // 2, 0, store.tile_shape[1] - nc))
            boxes.append((r0, c0, nr, nc, float(valid_tile[r0:r0 + nr, c0:c0 + nc].mean()), la))
    kbins = default_kbins(512.0, 2.0, 40)
    psd_fields = ["truth", "member0", "ensmean", "cams_bilinear"]
    psd_sum = np.zeros((len(psd_fields), len(kbins) - 1))
    psd_cnt = np.zeros((len(psd_fields), len(kbins) - 1))
    psd_n = np.zeros(1)
    N = store.n_pix
    sel = None
    if a.pixel_stride > 1:                      # cost probe / cheap check: a regular subset of the mask pixels
        sel = torch.arange(0, store.n_pix, a.pixel_stride, device=dev)
        N = sel.numel()
        boxes, map_dates, contact_dates = [], set(), set()
    elif a.psd_boxes_only:                      # stop-rule probe: the spectra boxes only, PSD on every day
        inb = np.zeros(store.n_pix, bool)
        for (r0, c0, nr, nc, _, _) in boxes:
            inb |= ((store.tile_row >= r0) & (store.tile_row < r0 + nr) & (store.tile_col >= c0)
                    & (store.tile_col < c0 + nc))
        sel = torch.from_numpy(np.nonzero(inb)[0]).to(dev)
        N = sel.numel()
        map_dates, contact_dates = set(), set()
        a.psd_every = 1
    sel_np = None if sel is None else sel.cpu().numpy()

    def expand(v):
        """(N,) values of the evaluated pixels -> (n_pix,) with NaN elsewhere (spectra of a pixel subset)"""
        if sel_np is None:
            return v
        out = np.full(store.n_pix, np.nan, np.float32)
        out[sel_np] = v
        return out
    smp = Sampler(model, variant, spec, M, a.ode_steps, dev, cm=cm, flowmap=flowmap, prior=prior,
                  coarse_spec=coarse_spec(rargs, spec), z_bound=a.z_bound, z0_trunc=a.z0_trunc)
    stations = None
    if getattr(model, "stations", False) and a.stations != "none":
        from cdffm_aq.stations import StationSet
        root = getattr(rargs, "stations", None) if a.stations == "auto" else a.stations
        stations = StationSet(root, years=sorted({store.hold_dates[t].year for t in days}), device=dev)
        if rank == 0:
            print(f"[eval_x] stations: {stations.n_total:,} assimilation monitors from {root}", flush=True)
    elif getattr(model, "stations", False) and rank == 0:
        print("[eval_x] stations model evaluated with EVERY monitor hidden (counterfactual)", flush=True)
    # A1(e) sanity counters (sums over days and ranks) and the per-cell map of pixels with a member > LGHAP_TRAIN_MAX
    SAN = ("n_pix", "n_pix_gt_440", "n_pix_gt_1000", "n_member_gt_1000", "n_member_at_bound", "n_member_at_pmax", "n_member_z0_trunc")
    san_tot = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in SAN}
    san_max = torch.zeros((), dtype=torch.float64, device=dev)
    ext_cells = torch.zeros(geom.H * geom.W, dtype=torch.float64, device=dev)
    for it, t in enumerate(mine):
        d = store.hold_dates[t]
        doy = torch.tensor([doy_of(d)], device=dev)
        raw = raw_to_device(store.raw_inputs(d), dev)
        feat = model.encode(assemble_inputs(raw, pix, rargs.grid, rargs.transport))
        hv = store.holdout(t)
        truth = torch.from_numpy(hv.astype(np.float32)).to(dev).clamp(min=1.0)
        okw = torch.from_numpy(hv >= 0).to(dev).float() if holes else None
        if sel is not None:
            truth = truth[sel]
            okw = None if okw is None else okw[sel]
        g = torch.Generator(device=dev)
        g.manual_seed(a.seed * 1_000_003 + t * 1009 + 23)
        gc = torch.Generator(device=dev)
        gc.manual_seed(a.seed * 1_000_003 + t * 1009 + 29)
        gco = torch.Generator(device=dev)
        gco.manual_seed(a.seed * 1_000_003 + t * 1009 + 37)
        gp = torch.Generator(device=dev)
        gp.manual_seed(a.seed * 1_000_003 + t * 1009 + 41)
        smp.start_day(feat, g, gen_coarse=gco, gen_prior=gp)
        smp.stn_fn = (lambda la, lo, d_=d: stations.features(d_, la, lo)[0]) if stations is not None else None
        want_psd = bool(boxes) and (days.index(t) % a.psd_every == 0)
        want_map = d in map_dates
        want_contact = d in contact_dates
        full = {k: np.empty(N, np.float32) for k in ("member0", "member1", "ensmean", "ensstd", "cams")} \
            if (want_psd or want_map or want_contact) else None
        day_sc = {n: Scores(msize[n], dev) for n in names}
        san = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in SAN}
        day_max = torch.zeros((), dtype=torch.float64, device=dev)
        smp.n_bound = smp.n_trunc = 0
        for s0 in range(0, N, a.chunk):
            e = min(N, s0 + a.chunk)
            kk = torch.arange(s0, e, device=dev) if sel is None else sel[s0:e]
            lat, lon, w = pix.lat[kk], pix.lon[kk], pix.w_area[kk]
            if okw is not None:
                w = w * okw[s0:e]
            mu, sig = pix.mu_sigma(kk[None], doy)
            z = smp.members(lat, lon, pix.static[kk], doy)                       # (M, n)
            ens = z_to_ug(z, mu, sig, pmax=a.pmax_ug)
            emax = ens.max(0).values
            san["n_pix"] += e - s0
            san["n_pix_gt_440"] += (emax > TMAX).sum()
            san["n_pix_gt_1000"] += (emax > ABSURD_UG).sum()
            san["n_member_gt_1000"] += (ens > ABSURD_UG).sum()
            if a.pmax_ug is not None:
                san["n_member_at_pmax"] += (ens >= a.pmax_ug).sum()
            day_max = torch.maximum(day_max, emax.max().double())
            ext_cells.index_add_(0, blocks["model"].bidx[1][kk], (emax > TMAX).double())
            y = truth[s0:e]
            clim_med = z_to_ug(torch.zeros_like(mu), mu, sig)
            zc = torch.randn(msize["climatology_ens"], e - s0, device=dev, generator=gc)
            clim_ens = z_to_ug(zc, mu, sig)
            cams = bilinear_log1p_at(raw["cams"][0], lat, lon, store.grid_native)[None]
            for n, X in (("model", ens), ("climatology_median", clim_med), ("climatology_ens", clim_ens),
                         ("cams_bilinear", cams)):
                day_sc[n].add(X, y, w)
                blocks[n].add(kk, X, y, w)
            if full is not None:
                full["member0"][s0:e] = ens[0].cpu().numpy()
                full["member1"][s0:e] = ens[min(1, M - 1)].cpu().numpy()
                full["ensmean"][s0:e] = ens.mean(0).cpu().numpy()
                full["ensstd"][s0:e] = (ens.std(0, unbiased=False) if M > 1 else torch.zeros_like(ens[0])).cpu().numpy()
                full["cams"][s0:e] = cams[0].cpu().numpy()
        san["n_member_at_bound"] += smp.n_bound
        san["n_member_z0_trunc"] += smp.n_trunc
        row = {"date": str(d), "sanity": {k: float(v) for k, v in san.items()}}
        row["sanity"]["max_member_ug"] = float(day_max)
        for k in SAN:
            san_tot[k] += san[k]
        san_max = torch.maximum(san_max, day_max)
        for n in names:
            sm = day_sc[n].summary()
            row[n] = {"crps_fair": sm["ug"]["crps_fair"], "rmse_ensmean": sm["ug"]["rmse_ensmean"],
                      "bias": sm["ug"]["bias"], "crps_fair_log": sm["log"]["crps_fair"],
                      "spread_skill_fortin": sm["ug"].get("spread_skill_fortin")}
            for k, v in day_sc[n].s.items():
                tot[n].s[k] += v
            br = blocks[n].end_day(d)
            row[n]["blocks"] = {str(s): br[s] for s in blocks[n].scales}
        daily.append(row)
        if want_psd:
            tr = np.log1p(store.holdout(t).astype(np.float32).clip(min=1))
            if holes:                                   # GHAP holes: the ensemble mean stands in (spectra only)
                tr = np.where(hv < 0, np.log1p(full["ensmean"]), tr)
            flds = {"truth": tr, "member0": expand(np.log1p(full["member0"])),
                    "ensmean": expand(np.log1p(full["ensmean"])), "cams_bilinear": expand(np.log1p(full["cams"]))}
            for fi, fname in enumerate(psd_fields):
                tile = tile_of(store, flds[fname])
                for bx in boxes:
                    _, p, _ = box_psd(tile, valid_tile, bx, kbins, store.res_deg)
                    psd_sum[fi] += np.nan_to_num(p)
                    psd_cnt[fi] += np.isfinite(p)
            psd_n += len(boxes)
        if want_map:
            save_maps(out_dir / f"maps_{d}.png", f"{a.run} step {step} | {d}", store,
                      {"LGHAP truth": store.holdout(t).astype(np.float32).clip(min=1), "CAMS bilinear": full["cams"],
                       "ensemble mean": full["ensmean"], "member 0": full["member0"], "member 1": full["member1"],
                       "ensemble std": full["ensstd"]}, zoom=zoom)
        if want_contact:
            cs = a.contact_stride
            sub = lambda v: tile_of(store, v)[::-cs, ::cs].copy()
            contact.append((str(d), {"LGHAP": sub(store.holdout(t).astype(np.float32).clip(min=1)),
                                     "CAMS bilinear": sub(full["cams"]), "member 0": sub(full["member0"]),
                                     "ensemble mean": sub(full["ensmean"]), "ensemble std": sub(full["ensstd"])}))
        r_m = row["model"]
        b4 = row["model"]["blocks"]["4"]["ug"]
        sn = row["sanity"]
        print(f"  [rank {rank}] {d}: model crps {r_m['crps_fair']:.3f} rmseM {r_m['rmse_ensmean']:.3f} ssF "
              f"{r_m['spread_skill_fortin']:.2f} | 1deg-block crps {b4['crps_fair']:.3f} ssF "
              f"{b4.get('spread_skill_fortin', float('nan')):.2f} | clim-ens crps {row['climatology_ens']['crps_fair']:.3f}"
              f" | cams rmse {row['cams_bilinear']['rmse_ensmean']:.3f} | pix>440 {sn['n_pix_gt_440']:.0f} >1000 "
              f"{sn['n_pix_gt_1000']:.0f} max {sn['max_member_ug']:.0f} bound {sn['n_member_at_bound']:.0f} cap "
              f"{sn['n_member_at_pmax']:.0f} trunc {sn['n_member_z0_trunc']:.0f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if world > 1:
        import torch.distributed as dist
        for n in names:
            tot[n].all_reduce()
            blocks[n].all_reduce()
        for v in san_tot.values():
            dist.all_reduce(v)
        dist.all_reduce(san_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(ext_cells)
        ps = torch.from_numpy(np.concatenate([psd_sum.ravel(), psd_cnt.ravel(), psd_n])).to(dev)
        dist.all_reduce(ps)
        ps = ps.cpu().numpy()
        m_ = psd_sum.size
        psd_sum, psd_cnt, psd_n = ps[:m_].reshape(psd_sum.shape), ps[m_:2 * m_].reshape(psd_cnt.shape), ps[-1:]
        gathered = [None] * world
        dist.all_gather_object(gathered, daily)
        if contact_dates:
            gc_ = [None] * world
            dist.all_gather_object(gc_, contact)
            contact = [r for part in gc_ for r in part]
        daily = sorted([r for part in gathered for r in part], key=lambda r: r["date"])
    if rank == 0:
        res = {n: tot[n].summary() for n in names}
        blk = {n: blocks[n].summary() for n in names}
        m, ce, cm_, cb = (res[n]["ug"] for n in names)
        st = {k: float(v) for k, v in san_tot.items()}
        npd = max(st["n_pix"], 1.0)
        sanity = dict(st, max_member_ug=float(san_max), lghap_train_max_ug=TMAX, absurd_ug=ABSURD_UG,
                      gt_440_means_gt_ug=TMAX,
                      frac_pix_days_any_member_gt_440=st["n_pix_gt_440"] / npd,
                      frac_pix_days_any_member_gt_1000=st["n_pix_gt_1000"] / npd,
                      frac_member_pixel_days_gt_1000=st["n_member_gt_1000"] / (npd * M),
                      frac_member_pixel_days_guard_bound=(st["n_member_at_bound"] + st["n_member_at_pmax"]) / (npd * M),
                      n_cells_with_pix_gt_440=int((ext_cells > 0).sum()))
        gates = {
            "crps_model_lt_climatology_ens": m["crps_fair"] < ce["crps_fair"],
            "rmse_ensmean_model_lt_climatology_median": m["rmse_ensmean"] < cm_["rmse_ensmean"],
            "frac_pix_days_any_member_gt_440_le_1e-6": sanity["frac_pix_days_any_member_gt_440"] <= 1e-6,   # > TMAX
            "crps_model_lt_climatology_median_mae": m["crps_fair"] < cm_["crps_fair"],
            "rmse_ensmean_model_lt_cams_bilinear": m["rmse_ensmean"] < cb["rmse_ensmean"],
            "spread_skill_in_0.5_1.5": (0.5 <= m.get("spread_skill", float("nan")) <= 1.5),
            "spread_skill_fortin_in_0.5_1.5": (0.5 <= m.get("spread_skill_fortin", float("nan")) <= 1.5),
        }
        skill = {"crps_skill_vs_climatology_ens": 1 - m["crps_fair"] / ce["crps_fair"],
                 "crps_skill_vs_cams_bilinear": 1 - m["crps_fair"] / cb["crps_fair"],
                 "rmse_skill_vs_cams_bilinear": 1 - m["rmse_ensmean"] / cb["rmse_ensmean"]}
        out = {"run": a.run, "variant": variant, "step": step, "ckpt": a.ckpt, "objective": rargs.objective,
               "members": M, "ode_steps": a.ode_steps, "n_days": len(days), "pixel_stride": a.pixel_stride,
               "psd_boxes_only": bool(a.psd_boxes_only),
               "dates": [str(store.hold_dates[t]) for t in days], "n_pixels_per_day": N, "weights": "cos(lat) area",
               "truth": "max(LGHAP, 1) ug m-3", "noise": dataclasses.asdict(spec), "prior_file": a.prior_file,
               "guard": {"z_bound": a.z_bound, "pmax_ug": a.pmax_ug, "z0_trunc": a.z0_trunc}, "sanity": sanity,
               "stations": None if not getattr(model, "stations", False) else ("none (all monitors hidden)" if stations is None else str(stations.root)),
               "scores": res, "blocks": blk, "skill": skill, "gates": gates, "daily": daily,
               "minutes": (time.time() - t0) / 60, "world": world, "psd_boxes": [list(map(float, b)) for b in boxes]}
        (out_dir / "metrics.json").write_text(json.dumps(out, indent=1, default=str))
        np.savez_compressed(out_dir / "extremes.npz", cells_pix_gt_440=ext_cells.cpu().numpy().reshape(geom.H, geom.W),
                            dates=np.array([r["date"] for r in daily]),
                            daily_pix_gt_440=np.array([r["sanity"]["n_pix_gt_440"] for r in daily]),
                            daily_pix_gt_1000=np.array([r["sanity"]["n_pix_gt_1000"] for r in daily]),
                            daily_max_member_ug=np.array([r["sanity"]["max_member_ug"] for r in daily]),
                            daily_rmse_ensmean=np.array([r["model"]["rmse_ensmean"] for r in daily]))
        if contact:
            contact.sort(key=lambda r: r[0])
            lat0, lat1 = store.tile_row_lat[0], store.tile_row_lat[-1]
            lon0 = store.tile_col_lon0
            extent = [lon0, lon0 + store.res_deg * (store.tile_shape[1] - 1), lat0, lat1]
            save_contact_sheet(out_dir / "contact_sheet.png", f"{a.run} ({variant}) step {step} | M={M}, "
                               f"{a.ode_steps} Euler steps | hold-out days", contact, extent)
        if psd_n[0] > 0:
            k = np.sqrt(kbins[:-1] * kbins[1:])
            np.savez(out_dir / "psd.npz", k_cycles_per_km=k, fields=np.array(psd_fields),
                     psd=np.where(psd_cnt > 0, psd_sum / np.maximum(psd_cnt, 1), np.nan), n_box_days=psd_n[0])
        print(f"\n=== {a.run} ({variant}) | step {step} | {len(days)} hold-out days x {N:,} pixels | M={M} ===", flush=True)
        for n in names:
            u, lg = res[n]["ug"], res[n]["log"]
            ss = f" ss={u['spread_skill']:.2f} ssF={u['spread_skill_fortin']:.2f}" if "spread_skill" in u else ""
            print(f"{n:20s} crps={u['crps_fair']:.3f} rmseM={u['rmse_ensmean']:.3f} rmse1={u['rmse_1member']:.3f} "
                  f"mae={u['mae_ensmean']:.3f} bias={u['bias']:+.3f}{ss} | log1p crps={lg['crps_fair']:.4f}", flush=True)
            bs = " | ".join(f"{s}c: crps={blk[n][s]['ug']['crps_fair']:.3f} rmse={blk[n][s]['ug']['rmse_ensmean']:.3f}"
                            + (f" ssF={blk[n][s]['ug']['spread_skill_fortin']:.2f}" if "spread_skill_fortin" in blk[n][s]["ug"] else "")
                            for s in ("1", "2", "4"))
            print(f"{'':20s} blocks {bs}", flush=True)
        print("sanity: " + json.dumps({k: (round(v, 10) if isinstance(v, float) else v) for k, v in sanity.items()}), flush=True)
        print("gates: " + json.dumps(gates), flush=True)
        print(f"[done] {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
