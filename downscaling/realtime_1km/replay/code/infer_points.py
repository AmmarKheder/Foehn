#!/usr/bin/env python3
"""WP1a step C: realtime_1km/infer.py evaluated at a list of mask pixels (one process per GCD).

Same system as the production inference: the checkpoint given by --ckpt (runs_ghap/pmG_final/ckpt/best.ckpt), loaded by
the production infer.load_ckpt (EMA weights, args.json of the run), the fitted prior of <run>/PRIOR.json, the guard of
<run>/SAMPLER.json (state bound, member cap, truncated base noise), 10 Euler steps, cdffm_aq_x.model_x.Sampler, one
downscaled member per coarse member (per_member = 1), the 47 channels of work/<d>/chan_L<L>.npz.
Noise: as in infer.py the parameters of member m at lead day L are drawn once per (seed, L, m),
    base = seed * 1000003 + L * 1009 + m * 7919; generators base + 23 (training noise, consumed, unused with a prior),
    base + 37 (coarse flow, unused for this variant) and base + 41 (the five band fields of the prior),
and the fields are then EVALUATED at (lat, lon). The LIIF condition, the prior weights, the 10 Euler steps and the
back-transform are pointwise, so a pixel has the same member values whichever other pixels are evaluated with it:
this script evaluates the positions of replay/positions.npz only (monitor pixels, rings, cell samples) instead of the
147,289,344 mask pixels. Per-pixel arrays (climatology, statics, lat/lon) are the rows of the GHAP store files read
through a memory map, with the formulas of cdffm_aq.store_ghap.GhapStore (not the 9 GB full read).
Firewall: the audit hook of the model processes is installed (no raw GHAP, sealed or station file can be opened);
positions.npz holds positions only.
Output replay/km/<d>.npz (float32, ug m-3 unless stated; lead axis = lead days 1..5, NaN for a lead day not made):
  km (5, M, n) members at the monitor pixel; km_ring1 / km_ring3 / km_ring10 (5, M, n) member means over the valid ring
  pixels (NaN where none); km_cell (5, M, n) member mean of log1p(member) over the 5 x 5 cell sample (log1p units);
  km_cell_lin (5, M, n) the same mean in ug m-3; valid (5,) dates; doy; guard hits per lead day; meta (json)
and replay/raw/<d>.npy (5, M, U) float32: the members at every evaluated position (order of positions.npz 'pos').
    srun -n 8 python -u infer_points.py --ckpt <best.ckpt> 2026-09-01 2026-09-02 ...   (rank r takes days r, r + 8, ...)"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
R = P / "realtime_1km/replay"
sys.path.insert(0, str(P))
sys.path.insert(0, str(P / "realtime_1km"))
from cdffm_aq.data_aq import Norm, PixelArrays, assemble_inputs, raw_to_device, z_to_ug  # noqa: E402
from cdffm_aq.store_ghap import GHAP_LAT0, GHAP_LON0, GHAP_MARKERS, GHAP_NC, GHAP_RES, GHAP_ROOT  # noqa: E402
from cdffm_aq.store_global import install_firewall  # noqa: E402
from cdffm_aq.train_aq import model_grid, noise_spec  # noqa: E402
from cdffm_aq_x.model_x import Sampler  # noqa: E402
from cdffm_aq_x.train_x import coarse_spec  # noqa: E402
import infer as PROD  # noqa: E402  (the production module: load_ckpt, guard_value; main() guarded)

NLEAD = 5


def sub_store(pos, rargs, cache):
    """the GhapStore attributes infer.py uses, for the mask positions `pos` (sorted) only: rows cached by positions.py in
    positions_store.npz (spot-checked here against the memory-mapped store files), formulas of GhapStore"""
    assert rargs.store == "ghap", rargs.store
    man = json.loads((GHAP_ROOT / "manifest.json").read_text())
    C = np.load(cache)
    assert np.array_equal(C["pos"], pos), "positions_store.npz does not match positions.npz"
    flat = C["flat"].astype(np.int64)
    k = np.sort(np.random.default_rng(0).choice(pos.size, 512, replace=False))
    assert np.array_equal(np.asarray(np.load(GHAP_ROOT / "grid/ghap_mask.npy", mmap_mode="r")[pos[k]]).astype(np.int64), flat[k])
    for name, f in (("clim_mean", "ghap_clim/clim_mean_G.npy"), ("clim_var", "ghap_clim/clim_var_G.npy"), ("static", "ghap_clim/static_G.npy")):
        ref = np.load(GHAP_ROOT / f, mmap_mode="r")
        assert C[name].dtype == ref.dtype and np.array_equal(np.asarray(ref[pos[k]]), C[name][k]), f"{name}: cache differs from {f}"
    r, c = flat // GHAP_NC, flat % GHAP_NC
    nc = man["normalisation"]
    return SimpleNamespace(lat=GHAP_LAT0 - GHAP_RES * r, lon=GHAP_LON0 + GHAP_RES * c,
                           clim_mean=np.ascontiguousarray(C["clim_mean"]), clim_var=np.ascontiguousarray(C["clim_var"]),
                           static=np.ascontiguousarray(C["static"]), sigma_floor=float(man["static"]["sigma_floor"]),
                           norm=Norm(nc["cams_log1p_mean_std"][0], nc["cams_log1p_mean_std"][1], nc["t6_channel_mean"], nc["t6_channel_std"]),
                           sigma_k=int(getattr(rargs, "sigma_k", 0)), train_max_ug=float(man["T_G_train_max_points"]))


def masked_mean(x, idx):
    """x (M, U), idx (n, k) long with -1 = invalid -> (M, n) mean over the valid entries, NaN where none"""
    ok = idx >= 0
    v = x[:, idx.clamp(min=0)] * ok[None]
    n = ok.sum(1)
    return torch.where(n[None] > 0, v.sum(2) / n.clamp(min=1)[None], torch.full_like(v[:, :, 0], float("nan")))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="+")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work-root", default=str(R / "work"), help="directory holding <d>/chan_L<L>.npz")
    ap.add_argument("--out", default=str(R / "km"))
    ap.add_argument("--raw", default=str(R / "raw"), help="directory of the raw member files ('none' to skip)")
    ap.add_argument("--positions", default=str(R / "positions.npz"))
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior-file", default="auto")
    ap.add_argument("--z-bound", default="auto")
    ap.add_argument("--pmax-ug", default="auto")
    ap.add_argument("--z0-trunc", default="auto")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    e = os.environ
    rank, world = int(e.get("SLURM_PROCID", 0)), int(e.get("SLURM_NTASKS", 1))
    if torch.cuda.is_available():
        dev = torch.device("cuda", int(e.get("SLURM_LOCALID", 0)) if torch.cuda.device_count() > 1 else 0)
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")
    install_firewall(GHAP_MARKERS)
    days = sorted(a.days)[rank::world]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rawdir = None if a.raw == "none" else Path(a.raw)
    if rawdir is not None:
        rawdir.mkdir(parents=True, exist_ok=True)
    days = [d for d in days if a.overwrite or not (out / f"{d}.npz").exists()]
    print(f"[points r{rank}/{world}] {len(days)} days: {' '.join(days)}", flush=True)
    if not days:
        return
    rargs, model, step, run_dir = PROD.load_ckpt(a.ckpt, dev)
    assert rargs.objective != "regression"
    variant = getattr(rargs, "variant", "base")
    assert variant in ("base", "estab"), variant               # pointwise variants only (hcf needs the cell-mean flow)
    assert not getattr(model, "stations", False), "stations model: not the real-time system"
    spec = noise_spec(rargs)
    z_bound, pmax_ug = PROD.guard_value(a.z_bound, "z_bound", run_dir), PROD.guard_value(a.pmax_ug, "pmax_ug", run_dir)
    z0_trunc = PROD.guard_value(a.z0_trunc, "z0_trunc", run_dir)
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
        prior_net = load_prior(prior_spec, model, spec, 1, dev).prior
    Z = np.load(a.positions)
    pos = Z["pos"].astype(np.int64)
    assert (np.diff(pos) > 0).all()
    tl = lambda x: torch.from_numpy(np.ascontiguousarray(x.astype(np.int64))).to(dev)
    mon, rings = tl(Z["mon"]), {int(dd): tl(Z[f"ring{int(dd)}"]) for dd in Z["rings"]}
    cell_idx = tl(Z["cell_idx"])[tl(Z["cell"])]                 # (n, 25) per monitor
    n, U = mon.numel(), pos.size
    t0 = time.time()
    store = sub_store(pos, rargs, Path(a.positions).with_name(Path(a.positions).stem + "_store.npz"))
    pix = PixelArrays(store, dev)
    tmax = store.train_max_ug
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ck_sha = sha(Path(a.ckpt).resolve())
    pr_sha = sha(prior_spec.split(":")[0]) if prior_spec else None
    print(f"[points r{rank}] ckpt={a.ckpt} (run {run_dir.name}, step {step}, variant {variant}, prior {prior_spec}, z_bound {z_bound}, "
          f"pmax_ug {pmax_ug}, z0_trunc {z0_trunc}); {U:,} positions for {n:,} monitors; store rows {time.time() - t0:.0f}s, "
          f"device {torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}", flush=True)
    for d in days:
        td = time.time()
        work = Path(a.work_root) / d
        km = {k: np.full((NLEAD, 0, n), np.nan, np.float32) for k in ("km", "km_ring1", "km_ring3", "km_ring10", "km_cell", "km_cell_lin")}
        raw, valid, doys, timing, Mo = None, [""] * NLEAD, [-1] * NLEAD, {}, None
        for L in range(1, NLEAD + 1):
            f = work / f"chan_L{L}.npz"
            if not f.exists():
                print(f"[points r{rank}] {d} lead day {L}: {f.name} missing, left NaN", flush=True)
                continue
            chan = np.load(f)
            v = dt.date.fromisoformat(str(chan["valid"]))
            assert v == dt.date.fromisoformat(d) + dt.timedelta(L - 1), (d, L, v)
            doy = torch.tensor([int(chan["doy"])], device=dev)
            if Mo is None:
                Mo = chan["cams"].shape[0]
                km = {k: np.full((NLEAD, Mo, n), np.nan, np.float32) for k in km}
                raw = np.full((NLEAD, Mo, U), np.nan, np.float32)
            assert chan["cams"].shape[0] == Mo
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            samplers = []
            for m in range(Mo):
                rw = raw_to_device({"cams": chan["cams"][m], "met": chan["met"], "fires": chan["fires"], "t6": chan["t6"][m]}, dev)
                feat = model.encode(assemble_inputs(rw, pix, rargs.grid, rargs.transport))
                smp = Sampler(model, variant, spec, 1, a.ode_steps, dev, cm=None, flowmap=rargs.objective == "flowmap",
                              coarse_spec=coarse_spec(rargs, spec),
                              prior=None if prior_net is None else SampledPrior(prior_net, 1, dev), z_bound=z_bound,
                              z0_trunc=z0_trunc)
                base = a.seed * 1_000_003 + L * 1009 + m * 7919
                g = torch.Generator(device=dev).manual_seed(base + 23)
                gco = torch.Generator(device=dev).manual_seed(base + 37)
                gp = torch.Generator(device=dev).manual_seed(base + 41)
                smp.start_day(feat, g, gen_coarse=gco, gen_prior=gp)
                samplers.append(smp)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_enc = time.time() - t0
            ens = torch.empty(Mo, U, device=dev)
            hits = {"at_pmax": 0, "gt_train_max": 0, "max_ug": 0.0}
            for s0 in range(0, U, a.chunk):
                kk = torch.arange(s0, min(U, s0 + a.chunk), device=dev)
                lat, lon = pix.lat[kk], pix.lon[kk]
                mu, sig = pix.mu_sigma(kk[None], doy)
                z = torch.cat([smp.members(lat, lon, pix.static[kk], doy) for smp in samplers], 0)
                x = z_to_ug(z, mu, sig, pmax=pmax_ug)
                ens[:, s0:s0 + a.chunk] = x
                hits["gt_train_max"] += int((x > tmax).any(0).sum())
                hits["max_ug"] = max(hits["max_ug"], float(x.max()))
                if pmax_ug is not None:
                    hits["at_pmax"] += int((x >= pmax_ug).sum())
            hits["at_bound"] = sum(smp.n_bound for smp in samplers)
            hits["z0_trunc"] = sum(smp.n_trunc for smp in samplers)
            assert torch.isfinite(ens).all()
            km["km"][L - 1] = ens[:, mon].cpu().numpy()
            for dd, idx in rings.items():
                km[f"km_ring{dd}"][L - 1] = masked_mean(ens, idx).cpu().numpy()
            km["km_cell"][L - 1] = masked_mean(torch.log1p(ens), cell_idx).cpu().numpy()
            km["km_cell_lin"][L - 1] = masked_mean(ens, cell_idx).cpu().numpy()
            raw[L - 1] = ens.cpu().numpy()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            valid[L - 1], doys[L - 1] = str(v), int(chan["doy"])
            timing[L] = {"valid": str(v), "encode_s": round(t_enc, 2), "sample_s": round(time.time() - t0 - t_enc, 2), "guard_hits": hits}
            del samplers
        if Mo is None:
            print(f"[points r{rank}] {d}: no channel file at all, skipped", flush=True)
            continue
        meta = dict(date=d, ckpt=str(Path(a.ckpt).resolve()), ckpt_sha256=ck_sha, run=run_dir.name, step=step, variant=variant,
                    objective=rargs.objective, store=rargs.store, ode_steps=a.ode_steps, members=Mo, per_member=1, seed=a.seed,
                    prior=prior_spec, prior_sha256=pr_sha, guard={"z_bound": z_bound, "pmax_ug": pmax_ug, "z0_trunc": z0_trunc},
                    sigma_k=store.sigma_k, positions=str(a.positions), n_positions=int(U), n_monitors=int(n),
                    work=str(work), leads_made=[L for L in range(1, NLEAD + 1) if valid[L - 1]], timing=timing,
                    device=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu", torch=torch.__version__,
                    seconds=round(time.time() - td, 1), job=e.get("SLURM_JOB_ID"), rank=rank,
                    written_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    noise="parameters seeded by (seed, lead day, member) as realtime_1km/infer.py, fields evaluated at the positions")
        tmp = out / f"{d}.tmp.npz"
        np.savez(tmp, valid=np.array(valid), doy=np.array(doys), meta=np.array(json.dumps(meta)), **km)
        os.replace(tmp, out / f"{d}.npz")
        if rawdir is not None:
            tmp = rawdir / f"{d}.tmp.npy"
            np.save(tmp, raw)
            os.replace(tmp, rawdir / f"{d}.npy")
        mx = max(t["guard_hits"]["max_ug"] for t in timing.values())
        print(f"[points r{rank}] {d}: leads {meta['leads_made']}, {Mo} members, {time.time() - td:.0f}s "
              f"(encode {sum(t['encode_s'] for t in timing.values()):.0f}s, sample {sum(t['sample_s'] for t in timing.values()):.0f}s), "
              f"monitor-pixel mean L1 {np.nanmean(km['km'][0]):.2f} ug m-3, max member {mx:.0f}", flush=True)
    print(f"[points r{rank}] done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
