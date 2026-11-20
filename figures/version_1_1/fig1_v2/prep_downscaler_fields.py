#!/usr/bin/env python3
"""Fields of panel e of the new Fig. 1 (how one coarse member becomes a 1 km field), for the forecast issued on
11 November 2026, lead day 1, member 1 (index 0), window 29.4-33.4 N, 72.2-76.2 E (Lahore).

The eight 1 km members of the real-time forecast are not stored (only their mean, two quantiles and three
probabilities). What panel e needs is member-level: the noise a member starts from, the state of the flow at each of the
ten sampling steps, and the member itself. This script therefore runs the frozen downscaler again, on CPU, for the
pixels of the window (and, with --globe, for every 5th pixel of the globe), from the archived inputs of that issue day:

  model      runs_ghap/pmG_final/ckpt/best.ckpt (EMA weights, step 22,000), prior runs_ghap/pmG_final/prior.pt:P3, guard
             of SAMPLER.json (state bound 10, noise truncated at 3, cap 1,000 ug m-3), 10 Euler steps: the settings
             written in realtime_1km/out/2026-11-11/pmG_final_P3T3/part0.json;
  inputs     realtime_1km/work/2026-11-11/chan_L1.npz, the 47 encoder channels of lead day 1 exactly as the operational
             run read them (PM2.5 of each 0.4 degree member, IFS weather, fires, one-day transport of each member);
  code       the project's own modules, imported, not copied (cdffm_aq, cdffm_aq_x); the sampling loop below repeats
             model_aq.integrate() step by step to keep the intermediate states, and is asserted equal to
             Sampler.members() (the operational code path) on every chunk;
  seeds      those of realtime_1km/infer.py: seed 0, base = 1009 L + 7919 m, generators base + 23 / + 41.

What is NOT identical to the operational run: the random numbers. The operational generators lived on the GPU, and a
CPU generator gives another stream for the same seed. The members made here are therefore other draws of the same
conditional law (same network, same inputs, same prior), not the eight operational members. The check at the end
compares the mean of the eight members made here with the stored operational mean over the window.

Nothing is written outside this directory (python -B: no byte code beside the project's modules); no GPU, no job.

    source /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/env.sh
    OMP_NUM_THREADS=12 cdffm_run python -B -u prep_downscaler_fields.py            # window: about 3 min
    OMP_NUM_THREADS=12 cdffm_run python -B -u prep_downscaler_fields.py --globe    # + member 1 at every 5th pixel
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DS = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
sys.path.insert(0, str(DS))
from cdffm_aq.data_aq import Norm, PixelArrays, assemble_inputs, in_channels, raw_to_device, z_to_ug   # noqa: E402
from cdffm_aq.model_aq import GRIDS, NoiseSpec, eval_sphere_noise, integrate, noise_chunk                # noqa: E402
from cdffm_aq_x.copula_x import BANDS_KM, SampledPrior, load_prior                                         # noqa: E402
from cdffm_aq_x.model_x import Sampler, build_model_x                                                      # noqa: E402

DAY, LEAD, SEED = "2026-11-11", 1, 0
RUN = DS / "runs_ghap" / "pmG_final"
GH = DS / "data" / "processed" / "ghap"
WORK = DS / "realtime_1km" / "work" / DAY
LENS = (33.4, 29.4, 72.2, 76.2)                                    # N, S, W, E: 400 x 400 pixels, 10 x 10 cells of 0.4 deg
NR, NC, LAT0, LON0, RES = 14400, 36000, 83.995, -179.995, 0.01     # the GHAP lattice (cdffm_aq/store_ghap.py)
OUT = HERE / "data" / f"ds_fields_{DAY}.npz"
OUT_GLOBE = HERE / "data" / f"ds_globe_{DAY}.npz"


def pixels(rows, cols, mask):
    """positions in the mask (-1 outside it) of the lattice pixels rows x cols, row-major"""
    ids = (rows[:, None].astype(np.int64) * NC + cols[None, :]).ravel()
    pos = np.searchsorted(mask, ids)
    pos[pos >= mask.size] = mask.size - 1
    return np.where(mask[pos] == ids, pos, -1)


def pixel_arrays(pos, dev, man):
    """PixelArrays of the chosen mask pixels (as realtime_1km/infer.py builds `sub` from the store, read here through memory maps)"""
    mask = np.load(GH / "grid/ghap_mask.npy", mmap_mode="r")
    ids = np.asarray(mask[pos]).astype(np.int64)
    r, c = ids // NC, ids % NC
    nc = man["normalisation"]
    sub = SimpleNamespace(lat=LAT0 - RES * r, lon=LON0 + RES * c,
                          clim_mean=np.asarray(np.load(GH / "ghap_clim/clim_mean_G.npy", mmap_mode="r")[pos]),
                          clim_var=np.asarray(np.load(GH / "ghap_clim/clim_var_G.npy", mmap_mode="r")[pos]),
                          static=np.asarray(np.load(GH / "ghap_clim/static_G.npy", mmap_mode="r")[pos]),
                          sigma_floor=float(man["static"]["sigma_floor"]),
                          norm=Norm(nc["cams_log1p_mean_std"][0], nc["cams_log1p_mean_std"][1], nc["t6_channel_mean"], nc["t6_channel_std"]),
                          sigma_k=0)
    return PixelArrays(sub, dev)


@torch.no_grad()
def member(smp, model, pix, doy, n_steps, z_bound, z0_trunc, pmax, chunk=65536, keep=False):
    """one member at every pixel of pix: PM2.5 (ug m-3); with keep, also the noise (before and after truncation), the
    state after each step, the band fields of the prior and their weights. The loop is model_aq.integrate() written
    out; its result is asserted equal to Sampler.members(), the operational code path."""
    n = pix.lat.numel()
    out = dict(pm=np.empty(n, np.float32))
    if keep:
        out.update(z0_raw=np.empty(n, np.float32), states=np.empty((n_steps + 1, n), np.float32),
                   xi=np.empty((len(BANDS_KM), n), np.float32), w2=np.empty((len(BANDS_KM), n), np.float32))
    dt_ = 1.0 / n_steps
    for s0 in range(0, n, chunk):
        kk = torch.arange(s0, min(n, s0 + chunk))
        lat, lon, st = pix.lat[kk], pix.lon[kk], pix.static[kk]
        mu, sig = pix.mu_sigma(kk[None], doy)
        cond = smp.condition(lat, lon, st, doy)                                    # (1, n, 64 + 2)
        z0_raw = smp.prior.field(lat, lon, cond, st)                               # (1, n, 1): the fitted prior
        x = z0_raw.clamp(-z0_trunc, z0_trunc).clamp(-z_bound, z_bound)
        states = [x[0, :, 0].numpy().copy()]
        for s in range(n_steps):
            t = torch.full(x.shape[:2], s * dt_)
            x = (x + model.velocity(x, cond, t) * dt_).clamp(-z_bound, z_bound)
            states.append(x[0, :, 0].numpy().copy())
        ref = smp.members(lat, lon, st, doy)                                       # the operational code path
        assert torch.equal(ref[0], x[0, :, 0]), "the written-out loop differs from Sampler.members()"
        out["pm"][s0:s0 + kk.numel()] = z_to_ug(x[..., 0], mu, sig, pmax=pmax)[0].numpy()
        if keep:
            out["z0_raw"][s0:s0 + kk.numel()] = z0_raw[0, :, 0].numpy()
            out["states"][:, s0:s0 + kk.numel()] = np.stack(states)
            pr = smp.prior
            la, lo = lat[None], lon[None]
            xi = torch.stack([eval_sphere_noise(p, la, lo, chunk=noise_chunk(1, s_))[..., 0] for p, s_ in zip(pr.params, pr.prior.specs)], -1)
            w = pr.prior.weights(cond, st[None])
            assert torch.allclose((w * xi).sum(-1, keepdim=True), z0_raw, atol=1e-5)
            out["xi"][:, s0:s0 + kk.numel()] = xi[0].T.numpy()
            out["w2"][:, s0:s0 + kk.numel()] = (w[0] ** 2).T.numpy()
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--globe", action="store_true", help="also member 1 at every 5th pixel of the globe (ds_globe_<day>.npz)")
    ap.add_argument("--members", type=int, default=8)
    a = ap.parse_args()
    t_start = time.time()
    dev = torch.device("cpu")
    man = json.loads((GH / "manifest.json").read_text())
    rargs = SimpleNamespace(**json.loads((RUN / "args.json").read_text()))
    rargs.variant = getattr(rargs, "variant", "base")
    smj, prj = json.loads((RUN / "SAMPLER.json").read_text()), json.loads((RUN / "PRIOR.json").read_text())
    z_bound, pmax, z0_trunc, n_steps = float(smj["z_bound"]), float(smj["pmax_ug"]), float(smj["z0_trunc"]), 10
    ck = torch.load(RUN / "ckpt" / "best.ckpt", map_location=dev, weights_only=False)
    step = int(ck.get("global_step", -1))
    model = build_model_x(rargs, in_channels(rargs.transport), rargs.static_dim, GRIDS[rargs.grid])
    model.load_state_dict({k[len("ema."):]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")})
    model.eval()
    spec = NoiseSpec(int(rargs.noise_features), float(rargs.noise_beta), float(rargs.noise_lmin_km), float(rargs.noise_lmax_km))
    prior_net = load_prior(f"{RUN / prj['file']}:{prj['name']}", model, spec, 1, dev).prior
    print(f"model step {step}, prior {prj['name']} ({prior_net.kind}), bands {prior_net.bands}, {time.time() - t_start:.0f} s", flush=True)

    chan = np.load(WORK / f"chan_L{LEAD}.npz")
    assert str(chan["valid"]) == DAY and chan["cams"].shape[0] == 8
    doy = torch.tensor([int(chan["doy"])])
    mask = np.load(GH / "grid/ghap_mask.npy", mmap_mode="r")

    def sampler(m):
        """Sampler of member m, as realtime_1km/infer.py lines 222-233; returns it with the encoder features"""
        raw = raw_to_device({"cams": chan["cams"][m], "met": chan["met"], "fires": chan["fires"], "t6": chan["t6"][m]}, dev)
        feat = model.encode(assemble_inputs(raw, PIX, rargs.grid, rargs.transport))
        smp = Sampler(model, rargs.variant, spec, 1, n_steps, dev, cm=None, flowmap=False, coarse_spec=spec,
                      prior=SampledPrior(prior_net, 1, dev), z_bound=z_bound, z0_trunc=z0_trunc)
        base = SEED * 1_000_003 + LEAD * 1009 + m * 7919
        g, gco, gp = (torch.Generator(device=dev).manual_seed(base + k) for k in (23, 37, 41))
        smp.start_day(feat, g, gen_coarse=gco, gen_prior=gp)
        return smp, feat

    # ---- the window
    r0, c0 = int(round((LAT0 - (LENS[0] - RES / 2)) / RES)), int(round((LENS[2] + RES / 2 - LON0) / RES))
    rows, cols = np.arange(r0, r0 + 400), np.arange(c0, c0 + 400)
    assert abs(LAT0 - RES * rows[-1] - (LENS[1] + RES / 2)) < 1e-6 and abs(LON0 + RES * cols[-1] - (LENS[3] - RES / 2)) < 1e-6
    pos = pixels(rows, cols, mask)
    ok = pos >= 0
    print(f"window: rows {rows[0]}..{rows[-1]}, cols {cols[0]}..{cols[-1]}, {int(ok.sum())} of {ok.size} pixels in the mask", flush=True)
    PIX = pixel_arrays(pos[ok], dev, man)

    def grid(v):
        o = np.full(ok.size, np.nan, np.float32); o[ok] = v
        return o.reshape(400, 400)

    res, t_m = {}, []
    members = np.empty((a.members, 400, 400), np.float32)
    for m in range(a.members):
        t0 = time.time()
        smp, feat = sampler(m)
        t1 = time.time()
        o = member(smp, model, PIX, doy, n_steps, z_bound, z0_trunc, pmax, keep=m == 0)
        members[m] = grid(o["pm"])
        t_m.append((round(t1 - t0, 1), round(time.time() - t1, 1)))
        print(f"  member {m + 1}: encoder {t_m[-1][0]} s, window {t_m[-1][1]} s, PM2.5 mean {np.nanmean(members[m]):.1f} max {np.nanmax(members[m]):.1f}", flush=True)
        if m == 0:
            res.update(z0_raw=grid(o["z0_raw"]), states=np.stack([grid(s) for s in o["states"]]),
                       xi=np.stack([grid(s) for s in o["xi"]]), w2=np.stack([grid(s) for s in o["w2"]]))
            # the encoding of member 1 around the window (128 features on the 0.25 degree grid: point r at 84 - 0.25 r, c at -180 + 0.25 c)
            fr0, fr1, fc0, fc1 = 190, 232, 996, 1040
            res.update(feat=feat[0, :, fr0:fr1, fc0:fc1].numpy().astype(np.float16), feat_rc0=np.array([fr0, fc0]))
            mu, sig = PIX.mu_sigma(torch.arange(PIX.lat.numel())[None], doy)
            st = PIX.static.float().numpy() * np.asarray(man["static"]["raw_std"], np.float32) + np.asarray(man["static"]["raw_mean"], np.float32)
            res.update(mu_doy=grid(mu[0].numpy()), sigma=grid(sig[0].numpy()), static=np.stack([grid(st[:, k]) for k in range(st.shape[1])]))
            if a.globe:
                t0 = time.time()
                m_all = np.asarray(mask)
                selg = np.nonzero(((m_all // NC) % 5 == 2) & ((m_all % NC) % 5 == 2))[0]          # the pixels world() of the figure script reads
                idg = m_all[selg].astype(np.int64)
                del m_all
                PG = pixel_arrays(selg, dev, man)
                og = member(smp, model, PG, doy, n_steps, z_bound, z0_trunc, pmax)
                w005 = np.full((3000, 7200), np.nan, np.float32)                                    # row 0 at 90 N, as world(): GHAP row r -> (r + 600) // 5
                w005[(idg // NC + 600) // 5, (idg % NC) // 5] = og["pm"]
                np.savez_compressed(OUT_GLOBE, w005_member=w005.astype(np.float16),
                                    meta=json.dumps(dict(day=DAY, lead_day=LEAD, member=1, pixels=int(selg.size), seconds=round(time.time() - t0),
                                                         note="member 1 made again on CPU (see prep_downscaler_fields.py), every 5th pixel as w005 of rt_fields")))
                print(f"  globe: {selg.size:,} pixels, {time.time() - t0:.0f} s -> {OUT_GLOBE}", flush=True)
                del PG, og
        del smp, feat

    # ---- what is drawn of the inputs (member 1): one-day transport, forecast weather
    t6 = chan["t6"][0].astype(np.float32)                                                           # (6, 288, 720), 0.5 degree, centres 83.875 N, 179.875 W
    hb = np.load(RT / "data" / DAY / "hres_B.npy", mmap_mode="r")                                    # IFS 00 UTC run, 1 degree, steps every 6 h
    res.update(t6_pm=np.expm1(t6[0]), t6_aloft=t6[1], t6_surv=t6[2],
               t2m=np.asarray(hb[0:4, 50], np.float32).mean(0),                                      # 2 m temperature, mean of 00, 06, 12, 18 UTC
               cams025=chan["cams"][0, 190:232, 996:1040].astype(np.float32))                        # member 1 as the encoder reads it (0.25 degree)

    # ---- check against the stored operational mean of the window
    rt = np.load(HERE / "data" / f"rt_fields_{DAY}.npz")
    op = rt["lens1km"].astype(np.float32)
    mine = members.mean(0)
    v = np.isfinite(op) & np.isfinite(mine)
    sd = members.std(0, ddof=1)
    chk = dict(pixels=int(v.sum()), operational_mean=float(op[v].mean()), mean_made_here=float(mine[v].mean()),
               correlation=float(np.corrcoef(op[v], mine[v])[0, 1]), correlation_log=float(np.corrcoef(np.log1p(op[v]), np.log1p(mine[v]))[0, 1]),
               mean_abs_difference=float(np.abs(op[v] - mine[v]).mean()), rms_difference=float(np.sqrt(((op[v] - mine[v]) ** 2).mean())),
               mean_member_sd=float(sd[v].mean()), expected_rms_difference_of_two_8_member_means=float(np.sqrt((2 * sd[v] ** 2 / 8).mean())),
               member1_mean=float(np.nanmean(members[0])), member1_max=float(np.nanmax(members[0])))
    z0, zr = res["states"][0], res["z0_raw"]
    meta = dict(day=DAY, lead_day=LEAD, member_drawn=1, lens_NSWE=LENS, device="cpu", torch=torch.__version__, threads=torch.get_num_threads(),
                ckpt=str(RUN / "ckpt" / "best.ckpt"), step=step, prior=f"{RUN / prj['file']}:{prj['name']}", bands_km=[list(b) for b in BANDS_KM],
                noise_beta=float(rargs.noise_beta), ode_steps=n_steps, guard=dict(z_bound=z_bound, pmax_ug=pmax, z0_trunc=z0_trunc), seed=SEED,
                inputs=str(WORK / f"chan_L{LEAD}.npz"), doy=int(chan["doy"]), seconds_per_member_encoder_window=t_m,
                noise=dict(mean=float(np.nanmean(z0)), sd=float(np.nanstd(z0)), share_truncated_at_3=float((np.abs(zr[np.isfinite(zr)]) > z0_trunc).mean()),
                           mean_band_variance_in_window=[float(np.nanmean(x)) for x in res["w2"]]),
                check_against_operational_mean=chk,
                static_columns=man["static"]["columns"],
                note="members made again on CPU from the archived inputs: same network, prior, sampler and seeds as the operational run, "
                     "another random stream (CPU generator), so other draws of the same conditional law")
    np.savez_compressed(OUT, members=members.astype(np.float16), states=res["states"].astype(np.float16), z0_raw=res["z0_raw"].astype(np.float16),
                        xi=res["xi"].astype(np.float16), w2=res["w2"].astype(np.float16), feat=res["feat"], feat_rc0=res["feat_rc0"],
                        mu_doy=res["mu_doy"], sigma=res["sigma"], static=res["static"].astype(np.float32),
                        t6_pm=res["t6_pm"].astype(np.float16), t6_aloft=res["t6_aloft"].astype(np.float16), t6_surv=res["t6_surv"].astype(np.float16),
                        t2m=res["t2m"], cams025=res["cams025"], meta=json.dumps(meta))
    print(json.dumps(meta, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
