"""cdffm_aq.eval_aq -- streaming evaluation of a train_aq run on the hold-out days (PLAN.md 6; smoke S2 gates).

For every hold-out day the encoder runs once; all mask pixels are then processed in chunks: condition, M members
(sphere-noise parameters drawn once per member and day, so a member is one field over the whole domain) integrated
with --ode-steps Euler steps (flow-map jumps for a flowmap run), PM2.5 = expm1(mu + sigma z). Truth = max(LGHAP, 1).
Scores (cos(lat) area-weighted sums, data_aq.Scores) in ug m-3 and in log1p: fair CRPS (Ferro 2014), RMSE of one
member and of the ensemble mean, MAE, bias, spread, spread/skill (mean std / RMSE and the Fortin et al. 2014
sqrt((M+1)/M var) form), rank histogram, WHO 24-h thresholds (POD, FAR, CSI, SEDI of member 0 and of the ensemble
mean, Brier of the member fraction). Baselines on the same days, pixels and truth:
  climatology_median  B0: expm1(mu_p(doy)) (deterministic: CRPS = MAE)
  climatology_ens     B0 as a distribution: M members expm1(mu + sigma z), z ~ N(0, 1) iid
  cams_bilinear       B1: the CAMS day-0 field bilinear (in log1p, periodic) at every pixel (CRPS = MAE)
Also: radially averaged spectra of log1p truth / member 0 / ensemble mean / CAMS on ~512 km boxes inside the mask
(every --psd-every-th day) and map PNGs (whole box + a 1 km zoom) for --map-dates.
Several processes (srun / torchrun) split the days and add their sums (all_reduce); rank 0 writes
<run>/eval_<tag>/metrics.json, psd.npz and the maps.

  python -m cdffm_aq.eval_aq --run RUN [--ckpt best|last|STEP] [--members 8] [--ode-steps 10]
"""
import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import RUNS_ROOT
from .data_aq import (PixelArrays, Scores, assemble_inputs, bilinear_log1p_at, doy_of, in_channels, open_store,
                      raw_to_device, store_kwargs, z_to_ug)
from .model_aq import build_model, draw_sphere_noise, eval_sphere_noise, integrate, noise_chunk
from .spectra import box_psd, default_kbins, find_boxes


def saved_steps(ckdir):
    out = {}
    for f in sorted(os.listdir(ckdir)) if os.path.isdir(ckdir) else []:
        m = re.match(r"step(\d+)(-v\d+)?\.ckpt$", f)
        if m and int(m.group(1)) not in out:
            out[int(m.group(1))] = os.path.join(ckdir, f)
    return out


def best_val_step(run_dir):
    """step of the saved checkpoint with the lowest validation fair CRPS (ug m-3); the test days never enter"""
    rows = [json.loads(line) for line in open(Path(run_dir) / "metrics.jsonl") if line.strip()]
    saved = saved_steps(Path(run_dir) / "ckpt")
    rows = [r for r in rows if r["step"] in saved]
    if not rows:
        raise SystemExit(f"{run_dir}: no validated checkpoint")
    return min(rows, key=lambda r: r["model"]["ug"]["crps_fair"])["step"]


def ckpt_path(run_dir, which):
    run_dir = Path(run_dir)
    if which == "last":
        return run_dir / "ckpt" / "last.ckpt"
    step = best_val_step(run_dir) if which == "best" else int(which)
    return Path(saved_steps(run_dir / "ckpt")[step])


def load_run(name, device, which="best", runs_root=RUNS_ROOT):
    """(args, EMA model in eval mode on device, global step) of a train_aq run"""
    from .train_aq import model_grid
    run_dir = Path(runs_root) / name
    args = SimpleNamespace(**json.loads((run_dir / "args.json").read_text()))
    ck = torch.load(ckpt_path(run_dir, which), map_location=device, weights_only=False)
    sd = {k[len("ema."):]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
    model = build_model(args, in_channels(args.transport), args.static_dim, model_grid(args.store, args.grid))
    model.load_state_dict(sd)
    return args, model.to(device).eval(), int(ck.get("global_step", -1))


def dist_setup(device_pref):
    """(rank, world, device); initialises torch.distributed when launched with several processes"""
    e = os.environ
    rank = int(e.get("RANK", e.get("SLURM_PROCID", "0")))
    world = int(e.get("WORLD_SIZE", e.get("SLURM_NTASKS", "1")))
    local = int(e.get("LOCAL_RANK", e.get("SLURM_LOCALID", "0")))
    gpu = device_pref != "cpu" and torch.cuda.is_available()
    dev = torch.device(f"cuda:{local}") if gpu else torch.device("cpu")
    if gpu:
        torch.cuda.set_device(dev)
    if world > 1:
        import torch.distributed as dist
        e.setdefault("MASTER_ADDR", "localhost")
        e.setdefault("MASTER_PORT", "29533")
        dist.init_process_group("nccl" if gpu else "gloo", rank=rank, world_size=world)
    return rank, world, dev


def tile_of(store, values):
    """(n_pix,) values -> (rows, cols) tile with NaN outside the mask"""
    g = np.full(store.tile_shape, np.nan, np.float32)
    g[store.tile_row, store.tile_col] = values
    return g


def save_maps(path, title, store, fields, zoom=None):
    """fields: {name: (n_pix,) ug m-3}; whole box (every 4th pixel) and, if zoom = (lat, lon, rows, cols), a 1 km
    window; north up; log colour scale"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    names = list(fields)
    tiles = {n: tile_of(store, v) for n, v in fields.items()}
    vals = np.concatenate([v for n, v in fields.items() if n != "ensemble std"])
    lo, hi = max(np.nanpercentile(vals, 0.5), 0.5), np.nanpercentile(vals, 99.8)
    views = [("box", lambda g: g[::-4, ::4])]                       # tile rows run south -> north: flip
    if zoom is not None:
        la, lo_, nr, nc = zoom
        r = int(np.argmin(np.abs(store.tile_row_lat - la)))
        c = int(round((lo_ - store.tile_col_lon0) / store.res_deg))
        r0 = int(np.clip(r - nr // 2, 0, max(store.tile_shape[0] - nr, 0)))
        c0 = int(np.clip(c - nc // 2, 0, max(store.tile_shape[1] - nc, 0)))
        views.append(("zoom", lambda g: g[r0:r0 + nr, c0:c0 + nc][::-1]))
    for vname, view in views:
        ncol = 3
        nrow = math.ceil(len(names) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow if vname == "box" else 4.6 * nrow),
                                 squeeze=False)
        for ax, n in zip(axes.ravel(), names):
            g = view(tiles[n])
            if n == "ensemble std":
                im = ax.imshow(g, cmap="viridis", interpolation="nearest")
            else:
                im = ax.imshow(g, cmap="RdYlBu_r", norm=LogNorm(vmin=lo, vmax=hi), interpolation="nearest")
            ax.set_title(n, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.035)
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        fig.suptitle(f"{title} ({vname}, ug m-3)")
        fig.tight_layout()
        fig.savefig(str(path).replace(".png", f"_{vname}.png"), dpi=110)
        plt.close(fig)


def save_contact_sheet(path, title, rows, extent):
    """ONE PNG, one row per date: LGHAP / CAMS bilinear / member 0 / ensemble mean (shared log scale per row) and the
    ensemble std; rows = [(date, {name: 2-D tile, north up})]"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    cols = ["LGHAP", "CAMS bilinear", "member 0", "ensemble mean", "ensemble std"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.6 * len(cols), 2.3 * len(rows) + 0.6), squeeze=False)
    for i, (d, f) in enumerate(rows):
        vals = np.concatenate([f[c][np.isfinite(f[c])] for c in cols[:4]])
        lo, hi = max(np.percentile(vals, 0.5), 0.5), np.percentile(vals, 99.8)
        for j, c in enumerate(cols):
            ax = axes[i, j]
            if c == "ensemble std":
                im = ax.imshow(f[c], cmap="viridis", interpolation="nearest", extent=extent,
                               vmin=0, vmax=np.nanpercentile(f[c], 99.5))
            else:
                im = ax.imshow(f[c], cmap="RdYlBu_r", norm=LogNorm(vmin=lo, vmax=hi), interpolation="nearest",
                               extent=extent)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(c, fontsize=10)
            if j == 0:
                ax.set_ylabel(d, fontsize=10)
            if j >= 3:
                fig.colorbar(im, ax=ax, fraction=0.035)
    fig.suptitle(f"{title} (ug m-3)", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(path), dpi=100)
    plt.close(fig)


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-root", default=str(RUNS_ROOT))
    ap.add_argument("--ckpt", default="best", help="best (lowest validation CRPS) | last | STEP")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--days", default=None, help="comma-separated hold-out day indices (default: all)")
    ap.add_argument("--map-dates", default="2021-03-15", help="comma-separated dates for map PNGs ('' = none)")
    ap.add_argument("--zoom", default="45.4,9.5,300,420", help="lat,lon,rows,cols of the 1 km map window (Po Valley)")
    ap.add_argument("--psd-every", type=int, default=3)
    ap.add_argument("--contact-dates", default="", help="comma-separated dates for ONE contact-sheet PNG ('' = none)")
    ap.add_argument("--contact-stride", type=int, default=3, help="pixel stride of the contact-sheet tiles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    t0 = time.time()
    rank, world, dev = dist_setup(a.device)
    rargs, model, step = load_run(a.run, dev, a.ckpt, a.runs_root)
    from .train_aq import noise_spec
    spec = noise_spec(rargs)
    store = open_store(rargs.store, **store_kwargs(rargs))
    pix = PixelArrays(store, dev)
    regression = rargs.objective == "regression"
    flowmap = rargs.objective == "flowmap"
    M = 1 if regression else a.members
    days = list(range(len(store.hold_dates))) if a.days is None else [int(x) for x in a.days.split(",")]
    mine = days[rank::world]
    map_dates = {dt.date.fromisoformat(s) for s in a.map_dates.split(",") if s}
    contact_dates = {dt.date.fromisoformat(s) for s in a.contact_dates.split(",") if s}
    contact = []
    zoom = tuple(float(x) if i < 2 else int(x) for i, x in enumerate(a.zoom.split(","))) if a.zoom else None
    tag = a.tag or f"{a.ckpt}_M{M}_s{a.ode_steps}"
    out_dir = Path(a.runs_root) / a.run / f"eval_{tag}"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval] run={a.run} step={step} objective={rargs.objective} M={M} ode_steps={a.ode_steps} "
              f"days={len(days)} world={world} noise={spec} -> {out_dir}", flush=True)
    names = ["model", "climatology_median", "climatology_ens", "cams_bilinear"]
    msize = {"model": M, "climatology_median": 1, "climatology_ens": max(M, 2), "cams_bilinear": 1}
    tot = {n: Scores(msize[n], dev) for n in names}
    daily = []
    valid_tile = ~np.isnan(tile_of(store, np.zeros(store.n_pix, np.float32)))
    boxes = find_boxes(valid_tile, store.tile_row_lat, res_deg=store.res_deg, max_boxes=6)
    kbins = default_kbins(512.0, 2.0, 40)
    psd_fields = ["truth", "member0", "ensmean", "cams_bilinear"]
    psd_sum = np.zeros((len(psd_fields), len(kbins) - 1))
    psd_cnt = np.zeros((len(psd_fields), len(kbins) - 1))      # empty log-k bins (low k) end up NaN
    psd_n = np.zeros(1)
    N = store.n_pix
    for it, t in enumerate(mine):
        d = store.hold_dates[t]
        doy = torch.tensor([doy_of(d)], device=dev)
        raw = raw_to_device(store.raw_inputs(d), dev)
        feat = model.encode(assemble_inputs(raw, pix, rargs.grid, rargs.transport))
        truth = torch.from_numpy(store.holdout(t).astype(np.float32)).to(dev).clamp(min=1.0)
        g = torch.Generator(device=dev)
        g.manual_seed(a.seed * 1_000_003 + t * 1009 + 23)
        params = None if regression else draw_sphere_noise(M, 1, spec, dev, g)
        gc = torch.Generator(device=dev)
        gc.manual_seed(a.seed * 1_000_003 + t * 1009 + 29)
        want_psd = bool(boxes) and (days.index(t) % a.psd_every == 0)
        want_map = d in map_dates
        want_contact = d in contact_dates
        full = {k: np.empty(N, np.float32) for k in ("member0", "member1", "ensmean", "ensstd", "cams")} \
            if (want_psd or want_map or want_contact) else None
        day_sc = {n: Scores(msize[n], dev) for n in names}
        for s0 in range(0, N, a.chunk):
            e = min(N, s0 + a.chunk)
            kk = torch.arange(s0, e, device=dev)
            lat, lon, w = pix.lat[kk], pix.lon[kk], pix.w_area[kk]
            mu, sig = pix.mu_sigma(kk[None], doy)
            cond = model.condition(feat, lat[None], lon[None], pix.static[kk][None], doy)
            if regression:
                z = model.velocity(torch.zeros(1, e - s0, 1, device=dev), cond, torch.zeros(1, e - s0, device=dev))
            else:
                z0 = eval_sphere_noise(params, lat[None].expand(M, -1), lon[None].expand(M, -1),
                                       chunk=noise_chunk(M, spec))
                z = integrate(model, z0, cond.expand(M, -1, -1), a.ode_steps, flowmap=flowmap)
            ens = z_to_ug(z[..., 0], mu, sig)
            y = truth[s0:e]
            day_sc["model"].add(ens, y, w)
            day_sc["climatology_median"].add(z_to_ug(torch.zeros_like(mu), mu, sig), y, w)
            zc = torch.randn(msize["climatology_ens"], e - s0, device=dev, generator=gc)
            day_sc["climatology_ens"].add(z_to_ug(zc, mu, sig), y, w)
            cams = bilinear_log1p_at(raw["cams"][0], lat, lon, store.grid_native)
            day_sc["cams_bilinear"].add(cams[None], y, w)
            if full is not None:
                full["member0"][s0:e] = ens[0].cpu().numpy()
                full["member1"][s0:e] = ens[min(1, M - 1)].cpu().numpy()
                full["ensmean"][s0:e] = ens.mean(0).cpu().numpy()
                full["ensstd"][s0:e] = (ens.std(0, unbiased=False) if M > 1 else torch.zeros_like(ens[0])).cpu().numpy()
                full["cams"][s0:e] = cams.cpu().numpy()
        row = {"date": str(d)}
        for n in names:
            sm = day_sc[n].summary()
            row[n] = {"crps_fair": sm["ug"]["crps_fair"], "rmse_ensmean": sm["ug"]["rmse_ensmean"],
                      "bias": sm["ug"]["bias"], "crps_fair_log": sm["log"]["crps_fair"]}
            for k, v in day_sc[n].s.items():
                tot[n].s[k] += v
        daily.append(row)
        if want_psd:
            tr = np.log1p(store.holdout(t).astype(np.float32).clip(min=1))
            flds = {"truth": tr, "member0": np.log1p(full["member0"]), "ensmean": np.log1p(full["ensmean"]),
                    "cams_bilinear": np.log1p(full["cams"])}
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
            sub = lambda v: tile_of(store, v)[::-cs, ::cs].copy()          # tile rows run south -> north: flip
            contact.append((str(d), {"LGHAP": sub(store.holdout(t).astype(np.float32).clip(min=1)),
                                     "CAMS bilinear": sub(full["cams"]), "member 0": sub(full["member0"]),
                                     "ensemble mean": sub(full["ensmean"]), "ensemble std": sub(full["ensstd"])}))
        r_m = row["model"]
        print(f"  [rank {rank}] {d}: model crps {r_m['crps_fair']:.3f} rmseM {r_m['rmse_ensmean']:.3f} | clim-ens crps "
              f"{row['climatology_ens']['crps_fair']:.3f} | cams rmse {row['cams_bilinear']['rmse_ensmean']:.3f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    if world > 1:
        import torch.distributed as dist
        for n in names:
            tot[n].all_reduce()
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
        m, ce, cm, cb = (res[n]["ug"] for n in names)
        gates = {
            "crps_model_lt_climatology_ens": m["crps_fair"] < ce["crps_fair"],
            "crps_model_lt_climatology_median_mae": m["crps_fair"] < cm["crps_fair"],
            "rmse_ensmean_model_lt_cams_bilinear": m["rmse_ensmean"] < cb["rmse_ensmean"],
            "spread_skill_in_0.5_1.5": (0.5 <= m.get("spread_skill", float("nan")) <= 1.5),
            "spread_skill_fortin_in_0.5_1.5": (0.5 <= m.get("spread_skill_fortin", float("nan")) <= 1.5),
        }
        skill = {"crps_skill_vs_climatology_ens": 1 - m["crps_fair"] / ce["crps_fair"],
                 "crps_skill_vs_cams_bilinear": 1 - m["crps_fair"] / cb["crps_fair"],
                 "rmse_skill_vs_cams_bilinear": 1 - m["rmse_ensmean"] / cb["rmse_ensmean"]}
        out = {"run": a.run, "step": step, "ckpt": a.ckpt, "objective": rargs.objective, "members": M,
               "ode_steps": a.ode_steps, "n_days": len(days), "dates": [str(store.hold_dates[t]) for t in days],
               "n_pixels_per_day": N, "weights": "cos(lat) area", "truth": "max(LGHAP, 1) ug m-3",
               "noise": dataclasses.asdict(spec), "scores": res, "skill": skill,
               "gates": gates, "daily": daily, "minutes": (time.time() - t0) / 60, "world": world,
               "psd_boxes": [list(map(float, b)) for b in boxes]}
        (out_dir / "metrics.json").write_text(json.dumps(out, indent=1, default=str))
        if contact:
            contact.sort(key=lambda r: r[0])
            lat0, lat1 = store.tile_row_lat[0], store.tile_row_lat[-1]
            lon0 = store.tile_col_lon0
            extent = [lon0, lon0 + store.res_deg * (store.tile_shape[1] - 1), lat0, lat1]
            save_contact_sheet(out_dir / "contact_sheet.png", f"{a.run} step {step} | M={M}, {a.ode_steps} Euler "
                               f"steps | hold-out days", contact, extent)
        if psd_n[0] > 0:
            k = np.sqrt(kbins[:-1] * kbins[1:])
            np.savez(out_dir / "psd.npz", k_cycles_per_km=k, fields=np.array(psd_fields),
                     psd=np.where(psd_cnt > 0, psd_sum / np.maximum(psd_cnt, 1), np.nan), n_box_days=psd_n[0])
        print(f"\n=== {a.run} | step {step} | {len(days)} hold-out days x {N:,} pixels | M={M} ===", flush=True)
        for n in names:
            u, lg = res[n]["ug"], res[n]["log"]
            ss = f" ss={u['spread_skill']:.2f} ssF={u['spread_skill_fortin']:.2f}" if "spread_skill" in u else ""
            print(f"{n:20s} crps={u['crps_fair']:.3f} rmseM={u['rmse_ensmean']:.3f} rmse1={u['rmse_1member']:.3f} "
                  f"mae={u['mae_ensmean']:.3f} bias={u['bias']:+.3f}{ss} | log1p crps={lg['crps_fair']:.4f}", flush=True)
        print("gates: " + json.dumps(gates), flush=True)
        print(f"[done] {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
