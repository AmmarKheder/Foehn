#!/usr/bin/env python3
"""A8(e)-2(ii): spectra of the FROZEN system recomputed with today's code, on the box pixels only.

eval_x (full evaluation of 2026-10-28) stored psd.npz = radially averaged power of log1p fields (GHAP truth, member 0,
ensemble mean, CAMS bilinear) in six ~512 km boxes on every third selection day (8 days), averaged over box-days.
The model is pointwise: a member's value at a pixel does not depend on which other pixels are evaluated (noise =
random-Fourier-feature fields drawn once per day from generators seeded by the day; model_x.Sampler). So the same
members can be recomputed on the pixels of the boxes alone, at about 1 % of the cost of the full evaluation, and the
spectra must come out the same if today's code path (checkpoint loading, encoder, LIIF condition, fitted prior P3,
truncation, guarded Euler integration, cap) is what it was on 2026-10-28. (eval_x --psd-boxes-only would do this but
fails on the GHAP store: shape mismatch in the hole handling, eval_x.py l.255-256.)

Every model call below is the one eval_x makes, with the same generator seeds (eval_x.py l.184-192, 203-211). The box
geometry and the spectra are eval_x's (l.117-128, 253-265) applied to the box sub-tile instead of the whole tile; the
truth and CAMS spectra do not involve the model and check this driver itself.
Read-only on the run; one day per srun task; writes <out>/psdrepro_r<rank>.npz.

  srun -n 8 ... python psdrepro.py --out .../a8/results/psdrepro        (defaults = the frozen system)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

W = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(W))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cdffm_aq.data_aq import (PixelArrays, assemble_inputs, bilinear_log1p_at, doy_of, open_store, raw_to_device,  # noqa: E402
                              store_kwargs, z_to_ug)
from cdffm_aq.spectra import box_psd, box_shape, default_kbins  # noqa: E402
from cdffm_aq.train_aq import noise_spec  # noqa: E402
from cdffm_aq_x.copula_x import load_prior  # noqa: E402
from cdffm_aq_x.model_x import Sampler, load_run_x  # noqa: E402
from cdffm_aq_x.train_x import coarse_spec  # noqa: E402

RG = W / "runs_ghap"
FIELDS = ["truth", "member0", "ensmean", "cams_bilinear"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pmG_base")
    ap.add_argument("--runs-root", default=str(RG))
    ap.add_argument("--ckpt", default="22000")
    ap.add_argument("--prior-file", default=str(RG / "pmG_base/copula_s22000/priors.pt") + ":P3")
    ap.add_argument("--days", default="0,45,90,135,180,225,270,315", help="hold-out indices = every 3rd SEL day")
    ap.add_argument("--psd-centres", default="45.0:10.0,36.5:116.0,26.5:81.0,13.0:2.0,-11.0:-62.0,39.5:-85.0")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--z-bound", type=float, default=10.0)
    ap.add_argument("--pmax-ug", type=float, default=1000.0)
    ap.add_argument("--z0-trunc", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    dev = torch.device(f"cuda:{int(os.environ.get('SLURM_LOCALID', 0))}")
    torch.cuda.set_device(dev)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = [int(x) for x in a.days.split(",")]
    mine = days[rank::world]
    rargs, model, step = load_run_x(a.run, dev, a.ckpt, a.runs_root)
    variant = getattr(rargs, "variant", "base")
    assert variant == "base" and not getattr(model, "stations", False)
    spec = noise_spec(rargs)
    store = open_store(rargs.store, **store_kwargs(rargs))
    pix = PixelArrays(store, dev)
    holes = bool(getattr(store, "has_holes", False))
    M = a.members
    prior = load_prior(a.prior_file, model, spec, M, dev)
    # boxes: eval_x l.117-127 (box_shape of the centre's latitude, centred, clipped to the tile)
    boxes, per_box = [], []
    for pair in a.psd_centres.split(","):
        la, lo = (float(v) for v in pair.split(":"))
        nr, nc, _, _ = box_shape(la, 512.0, store.res_deg)
        r = int(round((la - store.tile_row_lat[0]) / store.res_deg))
        c = int(round((lo - store.tile_col_lon0) / store.res_deg))
        r0 = int(np.clip(r - nr // 2, 0, store.tile_shape[0] - nr))
        c0 = int(np.clip(c - nc // 2, 0, store.tile_shape[1] - nc))
        inb = ((store.tile_row >= r0) & (store.tile_row < r0 + nr) & (store.tile_col >= c0) & (store.tile_col < c0 + nc))
        ids = np.nonzero(inb)[0]
        valid = np.zeros((nr, nc), bool)
        valid[store.tile_row[ids] - r0, store.tile_col[ids] - c0] = True
        boxes.append((r0, c0, nr, nc, float(valid.mean()), la))
        per_box.append((ids, valid))
    sel_np = np.unique(np.concatenate([ids for ids, _ in per_box]))
    assert sel_np.size == sum(ids.size for ids, _ in per_box), "the boxes overlap"
    sel = torch.from_numpy(sel_np).to(dev)
    N = sel.numel()
    kbins = default_kbins(512.0, 2.0, 40)
    psd_sum = np.zeros((len(FIELDS), len(kbins) - 1))
    psd_cnt = np.zeros((len(FIELDS), len(kbins) - 1))
    smp = Sampler(model, variant, spec, M, a.ode_steps, dev, cm=None, flowmap=False, prior=prior,
                  coarse_spec=coarse_spec(rargs, spec), z_bound=a.z_bound, z0_trunc=a.z0_trunc)
    info = []
    print(f"[psdrepro r{rank}] run={a.run} step={step} prior={a.prior_file} days={mine} boxes={len(boxes)} "
          f"pixels={N:,} ({time.time() - t0:.0f} s)", flush=True)
    for t in mine:
        d = store.hold_dates[t]
        doy = torch.tensor([doy_of(d)], device=dev)
        raw = raw_to_device(store.raw_inputs(d), dev)
        feat = model.encode(assemble_inputs(raw, pix, rargs.grid, rargs.transport))
        hv = store.holdout(t)
        g = torch.Generator(device=dev)
        g.manual_seed(a.seed * 1_000_003 + t * 1009 + 23)
        gco = torch.Generator(device=dev)
        gco.manual_seed(a.seed * 1_000_003 + t * 1009 + 37)
        gp = torch.Generator(device=dev)
        gp.manual_seed(a.seed * 1_000_003 + t * 1009 + 41)
        smp.start_day(feat, g, gen_coarse=gco, gen_prior=gp)
        smp.stn_fn = None
        smp.n_bound = smp.n_trunc = 0
        m0 = np.empty(N, np.float32)
        em = np.empty(N, np.float32)
        cams = np.empty(N, np.float32)
        n_cap = 0
        emax = 0.0
        for s0 in range(0, N, a.chunk):
            e = min(N, s0 + a.chunk)
            kk = sel[s0:e]
            lat, lon = pix.lat[kk], pix.lon[kk]
            mu, sig = pix.mu_sigma(kk[None], doy)
            z = smp.members(lat, lon, pix.static[kk], doy)                       # (M, n)
            ens = z_to_ug(z, mu, sig, pmax=a.pmax_ug)
            n_cap += int((ens >= a.pmax_ug).sum())
            emax = max(emax, float(ens.max()))
            m0[s0:e] = ens[0].cpu().numpy()
            em[s0:e] = ens.mean(0).cpu().numpy()
            cams[s0:e] = bilinear_log1p_at(raw["cams"][0], lat, lon, store.grid_native).cpu().numpy()
        hs = hv[sel_np]
        tr = np.log1p(hs.astype(np.float32).clip(min=1))
        if holes:                                   # GHAP holes: the ensemble mean stands in (eval_x l.255-256)
            tr = np.where(hs < 0, np.log1p(em), tr)
        flds = {"truth": tr, "member0": np.log1p(m0), "ensmean": np.log1p(em), "cams_bilinear": np.log1p(cams)}
        for fi, fname in enumerate(FIELDS):
            v = flds[fname]
            for (ids, valid), bx in zip(per_box, boxes):
                r0, c0, nr, nc, frac, la = bx
                sub = np.full((nr, nc), np.nan, np.float32)
                sub[store.tile_row[ids] - r0, store.tile_col[ids] - c0] = v[np.searchsorted(sel_np, ids)]
                _, p, _ = box_psd(sub, valid, (0, 0, nr, nc, frac, la), kbins, store.res_deg)
                psd_sum[fi] += np.nan_to_num(p)
                psd_cnt[fi] += np.isfinite(p)
        info.append({"date": str(d), "t": int(t), "n_member_at_bound": int(smp.n_bound), "n_member_z0_trunc": int(smp.n_trunc),
                     "n_member_at_pmax": n_cap, "max_member_ug": emax, "n_holes_in_boxes": int((hs < 0).sum())})
        print(f"[psdrepro r{rank}] {d}: box pixels {N:,} bound {smp.n_bound} cap {n_cap} trunc {smp.n_trunc} max {emax:.0f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    np.savez(out_dir / f"psdrepro_r{rank}.npz", psd_sum=psd_sum, psd_cnt=psd_cnt, n_box_days=len(boxes) * len(mine),
             k_cycles_per_km=np.sqrt(kbins[:-1] * kbins[1:]), fields=np.array(FIELDS), days=np.array(mine),
             boxes=np.array(boxes, np.float64), info=np.array(json.dumps(info)),
             meta=np.array(json.dumps({"run": a.run, "step": step, "prior_file": a.prior_file, "members": M,
                                       "ode_steps": a.ode_steps, "z_bound": a.z_bound, "pmax_ug": a.pmax_ug,
                                       "z0_trunc": a.z0_trunc, "seed": a.seed, "n_box_pixels": int(N)})))
    print(f"[psdrepro r{rank}] done in {time.time() - t0:.0f} s -> {out_dir}/psdrepro_r{rank}.npz", flush=True)


if __name__ == "__main__":
    main()
