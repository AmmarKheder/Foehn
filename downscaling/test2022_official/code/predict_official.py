#!/usr/bin/env python3
"""Amendment A7 (protocol_amendments/A7_official_stations_2022.md), steps (c) and (d): the MODEL process.

Frozen system runs_ghap/pmG_final (loaded with test2022_ghap/predict_2022.load_system), under the GhapStore firewall: no
raw GHAP file, no sealed file and no station file can be opened here. It reads positions only
(test2022_official/pos/st_ring_official.npz: station pixels + ring pixels), the frozen input stores, the 0.4 deg
forecasts of the paper (global1km/runs/cams04_flow_shift2.6_emis/cams12h/pred.npy) and the two input files written by
prep_inputs.py. One process per GCD (srun -n 8), work dealt round-robin.

Same-day mode (A7(c)), every day v of 2022, exactly as predict_2022.py (same inputs, sampler, prior, guard, seeds
seed * 1000003 + (1000 + day index) * 1009 + {23, 29, 37, 41}):
  sameday_ds       8 members            sameday_cams0    B1, CAMS first-day mean, bilinear in log1p
  sameday_bcsd     B2 (bcsd_G.npz)      sameday_climens  B0, 8 draws expm1(mu_p + sigma_p z)
  sameday_climmed  expm1(mu_p(doy)), the z = 0 state
Forecast mode (A7(d)), issue day I = 120..333 (2022-05-01 .. 11-30), lead day L = 1..5, valid day v = I + L - 1:
  channel 0 = member m's daily mean (trapezoid of the instants 24(L-1), 24L-12, 24L h; 0 h = the CAMS analysis of 00 UTC
  of I), regridded with realtime_1km/prep.py regrid (imported); channels 1-46 = the frozen stores of v (ERA5, GFAS, T6).
  chain         one member per coarse member; seeds keyed by ((1000 + v) * 8 + L) * 64 + m
  chain_rtseed  same features; the real-time adapter's seeds (realtime_1km/infer.py: L * 1009 + m * 7919, no date)
  chain_t6fc    channel 41 rebuilt from the forecast (channels.py t6_members: gather() with the ERA5-driven origins of
                start day v - 1 on the PM2.5 of v - 1 = CAMS-analysis trapezoid of I - 1 for L = 1, member m's own lead
                day L - 1 otherwise); seeds of chain
  camsfc_ds     channel 0 = CAMS forecast of the 00 UTC run of I at lead day L - 1 (inputs/camsfc025.npy); 8 members,
                same-day seeds of v
  bcsd_fc       B2 applied to each coarse member's channel-0 field (8 members)
Output (few files, inode quota): <out>/<arm>.npy, float16 ug m-3, last two axes (n_pix, 4) = station pixel, ring 1,
ring 3, ring 10 km (ring mean over the in-mask ring pixels, member by member; NaN if less than half of the ring is in
the mask); same-day arms (365, M, n_pix, 4), forecast arms (214, 5, M, n_pix, 4). Rows are written with pwrite; a unit
is flagged in done_sameday.npy / done_chain.npy after all its rows are on disk (a resubmission continues).
  python init_preds.py <out>          (once, before the srun)
  srun -n 8 python -u predict_official.py --out <out> [--sameday all|none|list] [--issues all|none|list]"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
for p_ in (ROOT / "realtime_1km", ROOT / "scripts", ROOT / "test2022_ghap", ROOT / "runs_global", ROOT):
    if str(p_) not in sys.path:
        sys.path.insert(0, str(p_))
from cdffm_aq.data_aq import (MET_FIRST, PixelArrays, assemble_inputs, bilinear_log1p_at, doy_of, open_store,  # noqa: E402
                              raw_to_device, store_kwargs, z_to_ug)
from cdffm_aq.train_aq import model_grid, noise_spec  # noqa: E402
from cdffm_aq_x.copula_x import SampledPrior, load_prior  # noqa: E402
from cdffm_aq_x.data_x import CellGeometry  # noqa: E402
from cdffm_aq_x.model_x import Sampler  # noqa: E402
from cdffm_aq_x.train_x import coarse_spec  # noqa: E402
from diag_extremes import bcsd_anomaly_grid, bilinear_at  # noqa: E402
from predict_2022 import load_system  # noqa: E402  (the frozen test's loader)
from build_global_t6 import gather  # noqa: E402  (the training transport operator)
import prep as rt_prep  # noqa: E402  (real-time adapter: regrid_weights, regrid; it puts an eccodes folder on sys.path)
sys.path[:] = [p for p in sys.path if not p.endswith("/realtime/pylib")]

SYSD = ROOT / "runs_ghap/pmG_final"
BCSD = ROOT / "data/processed/ghap/ghap_clim/bcsd_G.npz"
PRED = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/runs/cams04_flow_shift2.6_emis/cams12h")
ORG = ROOT / "data/processed/global/flexgreen/era5/origins_2022.npy"
Y0, ND = dt.date(2022, 1, 1), 365
I0, NI, NL, AN0 = 120, 214, 5, 119
RINGS = (1, 3, 10)
ARMS_S = {"sameday_ds": 8, "sameday_cams0": 1, "sameday_bcsd": 1, "sameday_climens": 8, "sameday_climmed": 1}
ARMS_C = {"chain": 8, "chain_t6fc": 8, "chain_rtseed": 8, "camsfc_ds": 8, "bcsd_fc": 8}
NP_ORG = 288 * 720 * 32 * 4


def npy_info(path):
    with open(path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        shape, fo, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        return tuple(shape), np.dtype(dtype), f.tell()


def npy_write(path, row, arr, row_shape):
    """arr (row_shape) -> row `row` of the C-order .npy (first axes flattened into rows of row_shape)"""
    shape, dtype, hdr = npy_info(path)
    assert tuple(arr.shape) == tuple(row_shape) and shape[len(shape) - len(row_shape):] == tuple(row_shape), (path, arr.shape, shape)
    assert 0 <= row < int(np.prod(shape[:len(shape) - len(row_shape)])), (path, row, shape)
    b = np.ascontiguousarray(arr, dtype=dtype)
    fd = os.open(path, os.O_WRONLY)
    try:
        n = os.pwrite(fd, b.tobytes(), hdr + row * b.nbytes)
        assert n == b.nbytes, (path, n, b.nbytes)
    finally:
        os.close(fd)


def parse_list(s, full):
    if s == "all":
        return list(full)
    if s == "none":
        return []
    return [int(x) for x in s.split(",")]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sameday", default="all", help="all | none | comma list of day indices of 2022")
    ap.add_argument("--issues", default="all", help="all | none | comma list of issue day indices (120..333)")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    e = os.environ
    rank, world = int(e.get("SLURM_PROCID", 0)), int(e.get("SLURM_NTASKS", 1))
    dev = torch.device(f"cuda:{int(e.get('SLURM_LOCALID', 0))}") if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    t0 = time.time()
    out = Path(a.out)
    M = a.members
    assert M == 8, "the 0.4 deg forecasts have 8 members; the frozen test uses 8 members"
    rargs, model, step, prior_spec, sj = load_system(SYSD, dev)
    assert prior_spec is not None and rargs.store == "ghap"
    spec = noise_spec(rargs)
    store = open_store(rargs.store, **store_kwargs(rargs))            # firewall on: sealed / station / raw GHAP unreadable
    grid = model_grid(rargs.store, rargs.grid)
    geom = CellGeometry(store, grid)
    variant = getattr(rargs, "variant", "base")
    assert variant == "base"
    bcsd = dict(np.load(BCSD))
    assert np.array_equal(bcsd["cells"].astype(np.int64), geom.cells), "bcsd_G.npz cells differ from CellGeometry"
    zr = np.load(O / "pos/st_ring_official.npz")
    pos = np.asarray(zr["pos"], np.int64)
    assert pos.ndim == 1 and pos.min() >= 0 and pos.max() < store.n_pix and np.all(np.diff(pos) > 0)
    n_pos, mon_idx = pos.size, np.asarray(zr["mon_idx"], np.int64)
    n_pix = mon_idx.size
    from types import SimpleNamespace
    sub = SimpleNamespace(lat=store.lat[pos], lon=store.lon[pos], clim_mean=store.clim_mean[pos], clim_var=store.clim_var[pos],
                          static=store.static[pos], sigma_floor=store.sigma_floor, norm=store.norm,
                          sigma_k=getattr(store, "sigma_k", 2))
    pix = PixelArrays(sub, dev)
    store.clim_mean = store.clim_var = store.static = store.lat = store.lon = None        # free the 147 M-pixel arrays
    geom.cell_id = None
    prior8 = load_prior(prior_spec, model, spec, M, dev)
    prior_net = prior8.prior
    zb, pmax, z0t = (sj.get(k) for k in ("z_bound", "pmax_ug", "z0_trunc"))
    skw = dict(cm=None, flowmap=rargs.objective == "flowmap", coarse_spec=coarse_spec(rargs, spec), z_bound=zb, z0_trunc=z0t)
    smp8 = Sampler(model, variant, spec, M, a.ode_steps, dev, prior=prior8, **skw)
    mon_t = torch.from_numpy(mon_idx).to(dev)
    ring_t = []
    for dd in RINGS:
        idx = np.asarray(zr[f"ring{dd}"], np.int64)
        okm = torch.from_numpy((idx >= 0).astype(np.float32)).to(dev)
        ring_t.append((torch.from_numpy(np.maximum(idx, 0)).to(dev), okm, okm.sum(1) >= 4 * dd))
    wts = rt_prep.regrid_weights(84.0)
    pred = np.load(PRED / "pred.npy", mmap_mode="r")
    doys = [int(x) for x in np.load(PRED / "doys.npy")]
    assert pred.shape == (NI, M, 10, 361, 900) and doys == list(range(I0, I0 + NI)), (pred.shape, doys[:3])
    an = np.load(O / "inputs/cams_an_2022.npy", mmap_mode="r")
    camsfc = np.load(O / "inputs/camsfc025.npy", mmap_mode="r")
    assert an.shape == (216, 2, 361, 900) and camsfc.shape == (NI, NL, 576, 1440)
    org_shape, org_dtype, org_hdr = npy_info(ORG)
    assert org_shape == (365, 1, 288, 720, 32, 4) and org_dtype == np.float16
    system = {"dir": str(SYSD), "args_name": rargs.name, "step": step, "prior": prior_spec,
              "sampler": {"z_bound": zb, "pmax_ug": pmax, "z0_trunc": z0t}, "members": M, "ode_steps": a.ode_steps}
    if rank == 0:
        print(f"[a7] system {system} | world {world} | positions {n_pos:,} ({n_pix:,} station pixels) | "
              f"start-up {time.time() - t0:.0f} s", flush=True)

    def gen(s):
        g = torch.Generator(device=dev)
        g.manual_seed(int(s))
        return g

    def compact(X):
        """X (m, n_pos) float32 on the device -> (m, n_pix, 4) float16 numpy: station pixel, ring means 1 / 3 / 10 km"""
        res = torch.empty(X.shape[0], n_pix, 4, device=dev)
        res[..., 0] = X[:, mon_t]
        for j, (idx, okm, valid) in enumerate(ring_t):
            mean = (X[:, idx] * okm).sum(-1) / okm.sum(-1).clamp(min=1.0)
            res[..., j + 1] = torch.where(valid[None], mean, torch.full_like(mean, float("nan")))
        return res.half().cpu().numpy()

    def chunks():
        for s0 in range(0, n_pos, a.chunk):
            yield torch.arange(s0, min(n_pos, s0 + a.chunk), device=dev)

    def members_all(samplers, doy):
        res = torch.empty(sum(s.M for s in samplers), n_pos, device=dev)
        for kk in chunks():
            mu, sig = pix.mu_sigma(kk[None], doy)
            z = torch.cat([s.members(pix.lat[kk], pix.lon[kk], pix.static[kk], doy) for s in samplers], 0)
            res[:, kk] = z_to_ug(z, mu, sig, pmax=pmax)
        return res

    def encode(raw):
        return model.encode(assemble_inputs(raw, pix, rargs.grid, rargs.transport))

    def start8(feat, key):
        gs = [gen(a.seed * 1_000_003 + key * 1009 + off) for off in (23, 29, 37, 41)]
        smp8.start_day(feat, gs[0], gen_coarse=gs[2], gen_prior=gs[3])
        smp8.n_bound = smp8.n_trunc = 0
        return gs[1]

    def member_samplers(feats, base_of):
        """one single-member Sampler per coarse member, as realtime_1km/infer.py l.221-237"""
        res = []
        for m, feat in enumerate(feats):
            smp = Sampler(model, variant, spec, 1, a.ode_steps, dev, prior=SampledPrior(prior_net, 1, dev), **skw)
            base = base_of(m)
            smp.start_day(feat, gen(base + 23), gen_coarse=gen(base + 37), gen_prior=gen(base + 41))
            res.append(smp)
        return res

    def sanity(ens, samplers):
        return {"max_ug": float(ens.max()), "min_ug": float(ens.min()), "n_at_pmax": int((ens >= pmax * 0.999).sum()),
                "n_at_bound": int(sum(s.n_bound for s in samplers)), "n_z0_trunc": int(sum(s.n_trunc for s in samplers)),
                "finite": bool(torch.isfinite(ens).all())}

    def bcsd_field(cams_np, d, doy):
        """B2 at all positions from a (576, 1440) coarse field in ug m-3 (predict_2022.py l.282-293)"""
        anom = torch.from_numpy(bcsd_anomaly_grid(bcsd, geom, cams_np, doy_of(d), d.month)).to(dev)
        res = torch.empty(n_pos, device=dev)
        for kk in chunks():
            mu, _ = pix.mu_sigma(kk[None], doy)
            res[kk] = torch.expm1(mu[0] + bilinear_at(anom, pix.lat[kk], pix.lon[kk], grid)).clamp(min=0.0)
        return res

    log_path = O / "logs" / f"predict_{a.tag}_r{rank:02d}.jsonl"

    def log(row):
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------------------------------------ same-day mode
    done_s = np.load(out / "done_sameday.npy")
    todo = [t for t in parse_list(a.sameday, range(ND)) if not done_s[t]]
    mine = todo[rank::world]
    print(f"[a7 r{rank}] same-day: {len(mine)} of {len(todo)} pending days", flush=True)
    for it, t in enumerate(mine):
        d = Y0 + dt.timedelta(t)
        t1 = time.time()
        doy = torch.tensor([doy_of(d)], device=dev)
        raw_np = store.raw_inputs(d)
        raw = raw_to_device(raw_np, dev)
        gc = start8(encode(raw), 1000 + t)
        ens = members_all([smp8], doy)
        tm = time.time() - t1
        san = sanity(ens, [smp8])
        cams0 = torch.empty(1, n_pos, device=dev)
        ce = torch.empty(M, n_pos, device=dev)
        cm_ = torch.empty(1, n_pos, device=dev)
        for kk in chunks():
            mu, sig = pix.mu_sigma(kk[None], doy)
            cams0[0, kk] = bilinear_log1p_at(raw["cams"][0], pix.lat[kk], pix.lon[kk], store.grid_native)
            ce[:, kk] = z_to_ug(torch.randn(M, kk.numel(), device=dev, generator=gc), mu, sig)
            cm_[0, kk] = z_to_ug(torch.zeros_like(mu), mu, sig)[0]
        bc = bcsd_field(raw["cams"][0].float().cpu().numpy(), d, doy)[None]
        for name, x in (("sameday_ds", ens), ("sameday_cams0", cams0), ("sameday_bcsd", bc), ("sameday_climens", ce),
                        ("sameday_climmed", cm_)):
            assert torch.isfinite(x).all(), (name, str(d))
            npy_write(out / f"{name}.npy", t, compact(x), (ARMS_S[name], n_pix, 4))
        npy_write(out / "done_sameday.npy", t, np.ones((), np.uint8), ())
        log(dict(mode="sameday", date=str(d), t=t, model_s=tm, total_s=time.time() - t1, sanity=san))
        if it % 10 == 0 or it == len(mine) - 1:
            print(f"[a7 r{rank}] same-day {d}: model {tm:.1f} s, total {time.time() - t1:.1f} s | max {san['max_ug']:.0f} "
                  f"bound {san['n_at_bound']} trunc {san['n_z0_trunc']} ({it + 1}/{len(mine)}, {(time.time() - t0) / 60:.1f} min)", flush=True)

    # ------------------------------------------------------------------------------------------------ forecast mode
    done_c = np.load(out / "done_chain.npy")
    todo = [I for I in parse_list(a.issues, range(I0, I0 + NI)) if not done_c[I - I0]]
    mine = todo[rank::world]
    print(f"[a7 r{rank}] forecast mode: {len(mine)} of {len(todo)} pending issue days", flush=True)
    for it, I in enumerate(mine):
        i = I - I0
        t1 = time.time()
        P = np.asarray(pred[i], np.float32)                                   # (M, 10, 361, 900) ug m-3, 12..120 h
        an0 = np.asarray(an[I - AN0, 0], np.float32)                          # CAMS analysis, 00 UTC of I
        inst = lambda h: np.repeat(an0[None], M, 0) if h == 0 else P[:, h // 12 - 1]   # noqa: E731  (prep.build_pm)
        pm025 = np.zeros((M, NL, 576, 1440), np.float16)
        for L in range(1, NL + 1):
            a_, b_, c_ = 24 * (L - 1), 24 * L - 12, 24 * L
            day = np.maximum(0.25 * inst(a_) + 0.5 * inst(b_) + 0.25 * inst(c_), 0.0)
            for m in range(M):
                pm025[m, L - 1] = rt_prep.regrid(day[m], wts).astype(np.float16)
        prev = np.maximum(0.25 * np.asarray(an[I - 1 - AN0, 0], np.float32) + 0.5 * np.asarray(an[I - 1 - AN0, 1], np.float32)
                          + 0.25 * an0, 0.0)                                  # PM2.5 of day I - 1 (prep.build_pm)
        prev025 = rt_prep.regrid(prev, wts).astype(np.float16)
        t_prep = time.time() - t1
        rows = {k: [] for k in ARMS_C}
        info = {"leads": {}}
        tt = {"encode": 0.0, "sample": 0.0, "bcsd": 0.0, "gather": 0.0}
        for L in range(1, NL + 1):
            v = I + L - 1
            d = Y0 + dt.timedelta(v)
            doy = torch.tensor([doy_of(d)], device=dev)
            raw_np = store.raw_inputs(d)
            raw = raw_to_device(raw_np, dev)
            cams_t = lambda x: torch.from_numpy(np.ascontiguousarray(x))[None].to(dev)   # noqa: E731
            # chain and chain_rtseed: frozen transport, forecast composition in channel 0
            ts = time.time()
            feats = [encode(dict(raw, cams=cams_t(pm025[m, L - 1]))) for m in range(M)]
            tt["encode"] += time.time() - ts
            ts = time.time()
            s1 = member_samplers(feats, lambda m: a.seed * 1_000_003 + (((1000 + v) * 8 + L) * 64 + m) * 1009)
            e1 = members_all(s1, doy)
            s2 = member_samplers(feats, lambda m: a.seed * 1_000_003 + L * 1009 + m * 7919)
            e2 = members_all(s2, doy)
            tt["sample"] += time.time() - ts
            li = {"valid": str(d), "chain": sanity(e1, s1), "chain_rtseed": sanity(e2, s2)}
            del feats, s1, s2
            # chain_t6fc: channel 41 from the forecast (channels.py t6_members)
            ts = time.time()
            s_ = v - 1                                                        # transport start day (doy index of 2022)
            org = np.fromfile(ORG, dtype=np.float16, count=NP_ORG, offset=org_hdr + s_ * NP_ORG * 2).reshape(288, 720, 32, 4)
            o = torch.from_numpy(org.astype(np.float32))[None].to(dev)

            def ch41(src025):
                c25 = torch.from_numpy(src025.astype(np.float32))[None, None].to(dev)
                g = gather(F.avg_pool2d(c25, 2), o)[0].cpu().numpy()
                return np.log1p(np.maximum(g[0], 0)).astype(np.float16)

            if L == 1:                                                        # check A7(g)-5: the stored channel 41
                ref = ch41(store.cams.read((d - dt.timedelta(1) - MET_FIRST).days))
                st0 = raw_np["t6"][0]
                diff = np.abs(ref.astype(np.float32) - st0.astype(np.float32))
                li["t6_check"] = {"max_abs_diff": float(diff.max()), "frac_bit_identical": float((ref.view(np.uint16) == st0.view(np.uint16)).mean())}
            t6s = []
            for m in range(M):
                if L == 1 and m > 0:
                    t6s.append(t6s[0])
                    continue
                t6m = raw_np["t6"].copy()
                t6m[0] = ch41(prev025 if L == 1 else pm025[m, L - 2])
                assert np.isfinite(t6m.astype(np.float32)).all()
                t6s.append(t6m)
            li["t6fc_ch41_mean_minus_frozen"] = float(np.mean([x[0].astype(np.float32).mean() for x in t6s]) - raw_np["t6"][0].astype(np.float32).mean())
            tt["gather"] += time.time() - ts
            ts = time.time()
            feats = [encode(dict(raw, cams=cams_t(pm025[m, L - 1]), t6=cams_t(t6s[m]))) for m in range(M)]
            tt["encode"] += time.time() - ts
            ts = time.time()
            s3 = member_samplers(feats, lambda m: a.seed * 1_000_003 + (((1000 + v) * 8 + L) * 64 + m) * 1009)
            e3 = members_all(s3, doy)
            tt["sample"] += time.time() - ts
            li["chain_t6fc"] = sanity(e3, s3)
            del feats, s3, o
            # camsfc_ds: the CAMS forecast of run I at lead day L - 1 in channel 0, same-day seeds of v
            ts = time.time()
            feat = encode(dict(raw, cams=cams_t(np.array(camsfc[i, L - 1]))))
            tt["encode"] += time.time() - ts
            ts = time.time()
            start8(feat, 1000 + v)
            e4 = members_all([smp8], doy)
            tt["sample"] += time.time() - ts
            li["camsfc_ds"] = sanity(e4, [smp8])
            # bcsd_fc: B2 on each coarse member
            ts = time.time()
            e5 = torch.stack([bcsd_field(pm025[m, L - 1].astype(np.float32), d, doy) for m in range(M)])
            tt["bcsd"] += time.time() - ts
            for name, x in (("chain", e1), ("chain_t6fc", e3), ("chain_rtseed", e2), ("camsfc_ds", e4), ("bcsd_fc", e5)):
                assert torch.isfinite(x).all(), (name, I, L)
                rows[name].append(compact(x))
            li["pm_member_mean_ug"] = float(pm025[:, L - 1].astype(np.float32).mean())
            info["leads"][L] = li
        for name in ARMS_C:
            for L in range(1, NL + 1):
                npy_write(out / f"{name}.npy", i * NL + L - 1, rows[name][L - 1], (ARMS_C[name], n_pix, 4))
        npy_write(out / "done_chain.npy", i, np.ones((), np.uint8), ())
        tot = time.time() - t1
        log(dict(mode="chain", issue=str(Y0 + dt.timedelta(I)), I=I, prep_s=t_prep, times=tt, total_s=tot, **info))
        print(f"[a7 r{rank}] issue {Y0 + dt.timedelta(I)}: prep {t_prep:.0f} s, encode {tt['encode']:.0f} s, sample {tt['sample']:.0f} s, "
              f"gather {tt['gather']:.0f} s, bcsd {tt['bcsd']:.0f} s, total {tot:.0f} s | t6 check {info['leads'][1]['t6_check']} "
              f"({it + 1}/{len(mine)}, {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[a7 r{rank}] done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
