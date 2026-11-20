#!/usr/bin/env python3
"""RT 1 km, step 3 (container, one process per GCD: srun -n 8): CD-FFM downscaling of each lead day of RT day d.

Model: any checkpoint of cdffm_aq/train_aq or cdffm_aq_x/train_x (args.json two directories up from the .ckpt, EMA
weights), sampled through cdffm_aq_x.model_x.Sampler (the evaluation sampler of both packages: base, estab, hcf).
Ensemble: member m of the 1 km ensemble is conditioned on RT member m (its own PM2.5 channel and T6), so the 1 km
spread carries RT's forecast spread plus the downscaling spread; --per-member k draws k CD-FFM members per RT
member (M = 8 k). Noise parameters are seeded by (seed, lead day, member) only, so a member is one coherent field
across tiles and ranks.
Prior: --prior-file auto (default) uses the fitted copula prior named in <run dir>/PRIOR.json when it exists (the
fallback final system base + P1r, PROTOCOL A1(a), A1(c''')); FILE.pt:NAME forces one; none disables it.
Guard (PROTOCOL A1(e)): --z-bound / --pmax-ug auto (default) read <run dir>/SAMPLER.json {"z_bound": B, "pmax_ug": P}
when it exists: flow state kept in [-B, B], members capped at P ug m-3; a number forces one, none disables it. The
member points at the bound / cap are counted in part<rank>.json ("guard_hits"). The run's --sigma-k (args.json) selects
the target sigma of the store (seasonal or annual), as in training.
GHAP system (PROTOCOL.md A2, runs_ghap/pmG_final, store 'ghap'): tiles on the GHAP lattice, row 0 of tile row r00 at
-59.995 (15 tile rows to 84 N; 'lattice' block of part<rank>.json); --out-root for tests outside realtime_1km/out.
Stations (PROTOCOL.md A3(e), prepared, not run): --stations FILE = the day's assimilation set written by stations_rt.py
(id, network, lat, lon, value, date); a stations model (runs_ghap/pmS_final) then conditions the lead whose valid day
equals the file's date (the analysis) on those monitors; every other lead and a missing file = every monitor hidden
(the pmG-like counterfactual, stated in part<rank>.json).
Domain: every pixel of the run's store (the global LGHAP mask, or the smoke box for runs/s2_europe), optionally cut to
--bbox lat0,lat1,lon0,lon1. Work unit = a 10 x 10 deg tile of the 0.01 deg LGHAP lattice (1000 x 1000 pixels, row r
= floor(i / 1000), column c = floor(j / 1000) of LGHAP row i / column j); tiles are dealt to the ranks by pixel count.
Products per lead day, per tile, float16, NORTH-UP (row 0 = northern edge), NaN outside the mask:
  mean, p10, p90 (member quantiles, linear interpolation), prob15, prob37_5, prob75 (member fraction above 15 / 37.5
  / 75 ug m-3, the WHO 24-h guideline and interim targets 2 and 1)
written as out/<d>/<tag>/lead<L>_part<rank>.npz (np.savez_compressed, keys '<product>/r<RR>c<CC>') + part<rank>.json.
    srun -n 8 python -u infer.py 2026-10-26 --ckpt runs/.../ckpt/last.ckpt [--bbox 35,60,-10,35] [--tag smoke_eu]"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))
from cdffm_aq.data_aq import (LGHAP_LAT0, LGHAP_LON0, LGHAP_RES, PixelArrays, assemble_inputs, in_channels,  # noqa: E402
                              raw_to_device, z_to_ug)
from cdffm_aq.train_aq import model_grid, noise_spec  # noqa: E402
from cdffm_aq_x.model_x import Sampler, build_model_x  # noqa: E402
from cdffm_aq_x.train_x import coarse_spec  # noqa: E402
from prep import NLEAD, WORK  # noqa: E402

OUT = HERE / "out"
PRODUCTS = ("mean", "p10", "p90", "prob15", "prob37_5", "prob75")
THR = (15.0, 37.5, 75.0)
TILE = 1000


def open_store_any(rargs):
    name = rargs.store
    if name == "smoke":
        from cdffm_aq.data_aq import SmokeStore
        return SmokeStore()
    if name.startswith("global"):
        from cdffm_aq.store_global import GlobalStore
        return GlobalStore(sigma_k=getattr(rargs, "sigma_k", 2))
    if name == "ghap":                     # PROTOCOL.md A2: GHAP system (runs_ghap/pmG_final), GHAP lattice to 84 N
        from cdffm_aq.store_ghap import GhapStore
        return GhapStore(sigma_k=getattr(rargs, "sigma_k", 0))
    raise SystemExit(f"store {name!r}: unknown to realtime_1km (add it to open_store_any)")


def guard_value(opt, key, run_dir):
    """--z-bound / --pmax-ug: auto -> <run dir>/SAMPLER.json[key] (None if absent), none -> None, else float"""
    if opt == "auto":
        sj = run_dir / "SAMPLER.json"
        v = json.loads(sj.read_text()).get(key) if sj.exists() else None
        return None if v is None else float(v)
    return None if opt == "none" else float(opt)


def load_ckpt(path, device):
    """(args, EMA model, step) from an explicit checkpoint path; run dir = path.parent.parent (args.json there)"""
    path = Path(path).resolve()
    run_dir = path.parent.parent
    args = SimpleNamespace(**json.loads((run_dir / "args.json").read_text()))
    if not hasattr(args, "variant"):
        args.variant = "base"
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = {k[len("ema."):]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
    model = build_model_x(args, in_channels(args.transport), args.static_dim, model_grid(args.store, args.grid))
    model.load_state_dict(sd)
    return args, model.to(device).eval(), int(ck.get("global_step", -1)), run_dir


def assign_tiles(tile_of_pix, world):
    tiles, counts = np.unique(tile_of_pix, return_counts=True)
    order = np.argsort(-counts)
    load = np.zeros(world, np.int64)
    owner = {}
    for k in order:
        r = int(np.argmin(load))
        owner[int(tiles[k])] = r
        load[r] += counts[k]
    return owner, load


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bbox", default=None, help="lat0,lat1,lon0,lon1 (degrees) to restrict the domain")
    ap.add_argument("--tag", default=None, help="output sub-directory (default: run name)")
    ap.add_argument("--per-member", type=int, default=1)
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--leads", default="1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior-file", default="auto", help="auto | none | priors.pt:NAME (copula_x fitted prior)")
    ap.add_argument("--z-bound", default="auto", help="auto (SAMPLER.json) | none | B: flow state in [-B, B] (A1(e))")
    ap.add_argument("--pmax-ug", default="auto", help="auto (SAMPLER.json) | none | P: member cap in ug m-3 (A1(e))")
    ap.add_argument("--z0-trunc", default="auto", help="auto (SAMPLER.json) | none | T: base noise clamped to [-T, T] (A1(g))")
    ap.add_argument("--out-root", default=str(OUT), help="output root (default realtime_1km/out; tests elsewhere)")
    ap.add_argument("--stations", default=None, help="A3(e): day file of stations_rt.py (assimilation monitors of the analysis day)")
    a = ap.parse_args()
    t_start = time.time()
    e = os.environ
    rank, world = int(e.get("SLURM_PROCID", 0)), int(e.get("SLURM_NTASKS", 1))
    local = int(e.get("SLURM_LOCALID", 0))
    dev = torch.device(f"cuda:{local}") if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    d = dt.date.fromisoformat(a.date)
    work = WORK / str(d)
    rargs, model, step, run_dir = load_ckpt(a.ckpt, dev)
    if rargs.objective == "regression":
        raise SystemExit("regression runs are deterministic: realtime_1km needs a generative (fm / flowmap) run")
    variant = getattr(rargs, "variant", "base")
    spec = noise_spec(rargs)
    store = open_store_any(rargs)
    sigma_k = getattr(store, "sigma_k", 2)              # kept after `del store` below (meta)
    grid = model_grid(rargs.store, rargs.grid)
    z_bound, pmax_ug = guard_value(a.z_bound, "z_bound", run_dir), guard_value(a.pmax_ug, "pmax_ug", run_dir)
    z0_trunc = guard_value(a.z0_trunc, "z0_trunc", run_dir)
    cm = None
    if variant == "hcf":
        from cdffm_aq_x.data_x import CellGeometry, CellMeans
        cm = CellMeans(CellGeometry(store, grid), dev)
    prior_spec = None
    if a.prior_file == "auto":
        pj = run_dir / "PRIOR.json"
        if pj.exists():
            pr = json.loads(pj.read_text())
            prior_spec = f"{run_dir / pr['file']}:{pr['name']}"
    elif a.prior_file != "none":
        prior_spec = a.prior_file
    prior_net = None
    if prior_spec:
        from cdffm_aq_x.copula_x import SampledPrior, load_prior
        prior_net = load_prior(prior_spec, model, spec, a.per_member, dev).prior
    tag = a.tag or run_dir.name
    stn_day, stn_set, stn_date = None, None, None
    if getattr(model, "stations", False):
        if a.stations:
            from cdffm_aq.stations import unit_xyz
            zs = np.load(a.stations)
            vv = np.asarray(zs["value"], np.float32)
            okv = np.isfinite(vv) & (vv > 0)
            stn_date = dt.date.fromisoformat(str(zs["date"]))
            la_ = torch.from_numpy(np.asarray(zs["lat"], np.float32)[okv]).to(dev)
            lo_ = torch.from_numpy(np.asarray(zs["lon"], np.float32)[okv]).to(dev)
            stn_set = (la_, lo_, torch.from_numpy(vv[okv]).to(dev), unit_xyz(la_, lo_))
            stn_day = f"{a.stations}: {int(okv.sum())} monitors for {stn_date}"
        else:
            stn_day = "stations model without --stations: every monitor hidden on every lead (counterfactual)"
    out = Path(a.out_root) / str(d) / tag
    out.mkdir(parents=True, exist_ok=True)
    # pixels and tiles
    lat_all, lon_all = np.asarray(store.lat), np.asarray(store.lon)
    sel = np.ones(lat_all.size, bool)
    if a.bbox:
        la0, la1, lo0, lo1 = map(float, a.bbox.split(","))
        sel = (lat_all >= la0) & (lat_all <= la1) & (lon_all >= lo0) & (lon_all <= lo1)
    pid = np.nonzero(sel)[0]
    # tile lattice: LGHAP rows from -57.995 (12 tile rows); GHAP store (A2): rows from -59.995 (15 tile rows, to 84 N)
    lat0 = -59.995 if rargs.store == "ghap" else LGHAP_LAT0
    tmax = float(getattr(store, "train_max_ug", 440.0))        # G0a threshold: 440 (LGHAP) or T_G (GHAP)
    ii = np.rint((lat_all[pid] - lat0) / LGHAP_RES).astype(np.int64)
    jj = np.rint((lon_all[pid] - LGHAP_LON0) / LGHAP_RES).astype(np.int64)
    tid = (ii // TILE) * 36 + jj // TILE
    owner, load = assign_tiles(tid, world)
    mine_t = sorted(t for t, r in owner.items() if r == rank)
    keep = np.isin(tid, mine_t)
    order = np.lexsort((jj[keep], ii[keep], tid[keep]))
    pid, ii, jj, tid = pid[keep][order], ii[keep][order], jj[keep][order], tid[keep][order]
    sub = SimpleNamespace(lat=lat_all[pid], lon=lon_all[pid], clim_mean=store.clim_mean[pid], clim_var=store.clim_var[pid],
                          static=store.static[pid], sigma_floor=store.sigma_floor, norm=store.norm,
                          sigma_k=getattr(store, "sigma_k", 2))
    pix = PixelArrays(sub, dev)
    n_pix = pid.size
    if rank == 0:
        print(f"[rt1km] {d} ckpt={a.ckpt} (run {run_dir.name}, store {rargs.store}, variant {variant}, step {step}, "
              f"prior {prior_spec}, sigma_k {sigma_k}, z_bound {z_bound}, pmax_ug {pmax_ug}, "
              f"z0_trunc {z0_trunc}); "
              f"{int(sel.sum()):,} pixels in {len(owner)} tiles over {world} ranks (load {load.min():,}..{load.max():,}) "
              f"-> {out}", flush=True)
    del store
    leads = [int(x) for x in a.leads.split(",")]
    k = a.per_member
    timing = {}
    bounds = np.searchsorted(tid, mine_t + [10 ** 9])
    for L in leads:
        chan = np.load(work / f"chan_L{L}.npz")
        v = dt.date.fromisoformat(str(chan["valid"]))
        doy = torch.tensor([int(chan["doy"])], device=dev)
        Mo = chan["cams"].shape[0]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        samplers = []
        for m in range(Mo):
            raw = raw_to_device({"cams": chan["cams"][m], "met": chan["met"], "fires": chan["fires"], "t6": chan["t6"][m]}, dev)
            feat = model.encode(assemble_inputs(raw, pix, rargs.grid, rargs.transport))
            smp = Sampler(model, variant, spec, k, a.ode_steps, dev, cm=cm, flowmap=rargs.objective == "flowmap",
                          coarse_spec=coarse_spec(rargs, spec),
                          prior=None if prior_net is None else SampledPrior(prior_net, k, dev), z_bound=z_bound,
                          z0_trunc=z0_trunc)
            base = a.seed * 1_000_003 + L * 1009 + m * 7919
            g = torch.Generator(device=dev).manual_seed(base + 23)
            gco = torch.Generator(device=dev).manual_seed(base + 37)
            gp = torch.Generator(device=dev).manual_seed(base + 41)
            smp.start_day(feat, g, gen_coarse=gco, gen_prior=gp)
            if stn_set is not None and v == stn_date:
                from cdffm_aq.stations import station_features
                smp.stn_fn = lambda la, lo, s_=stn_set: station_features(la, lo, s_[0], s_[1], s_[2], mxyz=s_[3])[0]
            samplers.append(smp)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_enc = time.time() - t0
        tiles = {}
        hits = {"at_bound": 0, "at_pmax": 0, "gt_440": 0, "max_ug": 0.0}
        for ti, t in enumerate(mine_t):
            p0, p1 = bounds[ti], bounds[ti + 1]
            prod = np.full((len(PRODUCTS), TILE, TILE), np.nan, np.float16)
            r_loc = TILE - 1 - (ii[p0:p1] - (t // 36) * TILE)                 # north-up
            c_loc = jj[p0:p1] - (t % 36) * TILE
            for s0 in range(p0, p1, a.chunk):
                s1 = min(p1, s0 + a.chunk)
                kk = torch.arange(s0, s1, device=dev)
                lat, lon = pix.lat[kk], pix.lon[kk]
                mu, sig = pix.mu_sigma(kk[None], doy)
                z = torch.cat([smp.members(lat, lon, pix.static[kk], doy) for smp in samplers], 0)   # (Mo k, n)
                ens = z_to_ug(z, mu, sig, pmax=pmax_ug)
                hits["gt_440"] += int((ens > tmax).any(0).sum())
                hits["max_ug"] = max(hits["max_ug"], float(ens.max()))
                if pmax_ug is not None:
                    hits["at_pmax"] += int((ens >= pmax_ug).sum())
                q = torch.quantile(ens, torch.tensor([0.1, 0.9], device=dev), dim=0)
                res = [ens.mean(0), q[0], q[1]] + [(ens > th).float().mean(0) for th in THR]
                rr, cc = r_loc[s0 - p0:s1 - p0], c_loc[s0 - p0:s1 - p0]
                for pi, x in enumerate(res):
                    prod[pi, rr, cc] = x.clamp(max=65000).cpu().numpy().astype(np.float16)
            for pi, name in enumerate(PRODUCTS):
                tiles[f"{name}/r{t // 36:02d}c{t % 36:02d}"] = prod[pi]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_smp = time.time() - t0 - t_enc
        tw = time.time()
        np.savez_compressed(out / f"lead{L}_part{rank}.npz", **tiles)
        hits["at_bound"] = sum(smp.n_bound for smp in samplers)
        hits["z0_trunc"] = sum(smp.n_trunc for smp in samplers)
        timing[L] = {"valid": str(v), "encode_s": round(t_enc, 2), "sample_s": round(t_smp, 2),
                     "write_s": round(time.time() - tw, 2), "pixels": int(n_pix), "members": Mo * k, "guard_hits": hits}
        print(f"  [rank {rank}] lead day {L} ({v}): {n_pix:,} px, {Mo} encodes {t_enc:.1f}s, sampling {t_smp:.1f}s "
              f"({t_smp / max(n_pix, 1) * 1e6 / (Mo * k):.3f} s per M px per member), write {time.time() - tw:.1f}s | "
              f"pix>440 {hits['gt_440']} max {hits['max_ug']:.0f} ug, at bound {hits['at_bound']}, at cap {hits['at_pmax']}", flush=True)
    meta = {"rank": rank, "world": world, "date": str(d), "ckpt": str(Path(a.ckpt).resolve()), "run": run_dir.name,
            "store": rargs.store, "variant": variant, "objective": rargs.objective, "step": step,
            "ode_steps": a.ode_steps, "members": samplers and len(samplers) * k, "per_member": k, "bbox": a.bbox,
            "prior": prior_spec, "guard": {"z_bound": z_bound, "pmax_ug": pmax_ug, "z0_trunc": z0_trunc},
            "sigma_k": sigma_k, "stations": stn_day,
            "lattice": {"lat_row0": lat0, "lon_col0": LGHAP_LON0, "res": LGHAP_RES, "tile": TILE,
                        "tile_r": "floor(i / 1000), i = (lat - lat_row0) / res (south -> north)", "gt_440_means_gt_ug": tmax},
            "tiles": [f"r{t // 36:02d}c{t % 36:02d}" for t in mine_t], "timing": timing,
            "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "total_s": round(time.time() - t_start, 1)}
    (out / f"part{rank}.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
