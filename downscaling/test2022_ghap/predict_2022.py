#!/usr/bin/env python3
"""PROTOCOL.md A2(d), GHAP 2022 single pass, step (a): the 2022 predictions of the frozen GHAP system (runs_ghap/pmG_final),
written by a MODEL process under the GhapStore firewall (no raw GHAP file, no data/processed/ghap/sealed file and no
station file can be opened here). Copy of runs_global/predict_2024.py with the GHAP paths (BCSD bcsd_G.npz, station
positions test2022_ghap/st_pix_2022.npz, G0a threshold T_G); the LGHAP text below is the original's.
Original docstring: the 2024 predictions, written by a MODEL process under the
GlobalStore firewall (no LGHAP 2024 file, no sealed file and no station file can be opened here; the inputs of a 2024 day
are CAMS cams_pm25_025.npy, met_2023_2024.npy, fires_injh_2023_2024.npy and transport/t6_2024.npy through
GlobalStore.raw_inputs). Scoring is a separate process (score_2024.py) that opens the truth.

System (--system DIR: runs_global/pmA_final23 or runs_global/pmA_final): DIR/ckpt/best.ckpt (EMA weights; args.json in
DIR), the fitted copula prior of DIR/PRIOR.json (DIR/prior.pt), the sampler of DIR/SAMPLER.json (flow state |z| <= z_bound,
members capped at pmax_ug, base noise truncated at |z0| <= z0_trunc): exactly the eval_x sampler of the gates.
Arms, per day, at the fixed scoring subsample (every 8th LGHAP mask pixel = coupling/common.py sub_index) and at the 2024
station pixels (coupling/st_pix_2024.npz 'pos', positions only):
  cdffm          M members x --ode-steps Euler (Sampler, prior, guard, truncation)
  cams_bilinear  B1: CAMS day-0 on the 0.25 deg grid, bilinear in log1p to the pixel (as eval_x), M = 1
  bcsd           B2: lghap_clim/bcsd_A.npz as fitted (training days of split A), Wood-2004 form of A1(d) R1-F3:
                 expm1(mu_p + bilinear(QM_c,m(x_c) - mubar_c)) (runs_global/diag_extremes.bcsd_anomaly_grid), M = 1
  clim           B0: climatology ensemble as eval_x, expm1(mu_p + sigma_p z), z iid N(0, 1), M members
  -> <out>/preds/<arm>/<date>.npz and <out>/preds_st/<arm>/<date>.npz: members (M, n) float16 ug m-3 + meta (json).
Full fields (days with day index % --full-every == 0): cdffm over all 114,305,974 mask pixels ->
  <out>/fields/cdffm/<date>.npz: ensmean float16 (ug m-3), n_gt_37p5 and n_gt_75 uint8 (number of members above
  37.5 / 75 ug m-3; probability = n / M), mask order; the subsample and station members of that day are taken from the
  same pass (the noise is a function of position, so they equal the subsample-only computation up to rounding).
Seeds: as eval_x with t = 1000 + day index of the year (eval_x uses the hold-out index of 2023): fine noise
seed * 1000003 + t * 1009 + 23, climatology + 29, coarse + 37, prior + 41. Days are dealt to the ranks by a fixed
cost-balanced rule; a day whose files all exist is skipped (resubmission continues).
  srun -n 8N python -u runs_global/predict_2024.py --system runs_global/pmA_final23 --full-every 3
  srun -n 8 python -u runs_global/predict_2024.py --system runs_global/pmA_final --check 2023-01-08 --out <scratch>
    (--check: reproduction of eval_x on one 2023 hold-out day, full field split over the ranks, scored against
     full_2023 (not sealed); plus the subsample path compared with the full-field pass at the subsample)"""
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

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
for p_ in (ROOT, ROOT / "runs_global"):
    if str(p_) not in sys.path:
        sys.path.insert(0, str(p_))
from cdffm_aq.data_aq import (PixelArrays, assemble_inputs, bilinear_log1p_at, doy_of, fair_crps, in_channels,  # noqa: E402
                              open_store, raw_to_device, store_kwargs, z_to_ug)
from cdffm_aq.train_aq import model_grid, noise_spec  # noqa: E402
from cdffm_aq_x.data_x import CellGeometry, CellMeans  # noqa: E402
from cdffm_aq_x.model_x import Sampler, build_model_x  # noqa: E402
from cdffm_aq_x.train_x import coarse_spec  # noqa: E402
from diag_extremes import bcsd_anomaly_grid, bilinear_at  # noqa: E402

OUT = ROOT / "test2022_ghap"
ST_POS = OUT / "st_ring_2022.npz"          # A2(e): monitor pixels + 1/3/10 km ring pixels
BCSD = ROOT / "data/processed/ghap/ghap_clim/bcsd_G.npz"
SUB_STRIDE = 8
ARMS = ("cdffm", "cams_bilinear", "bcsd", "clim")
COST_FULL, COST_SUB = 8.0, 1.0


def load_system(sdir, device):
    """(args, EMA model, step, prior spec or None, sampler dict) of a final-system folder"""
    sdir = Path(sdir)
    args = SimpleNamespace(**json.loads((sdir / "args.json").read_text()))
    if not hasattr(args, "variant"):
        args.variant = "base"
    ck = torch.load(sdir / "ckpt" / "best.ckpt", map_location=device, weights_only=False)
    sd = {k[len("ema."):]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
    model = build_model_x(args, in_channels(args.transport), args.static_dim, model_grid(args.store, args.grid))
    model.load_state_dict(sd)
    pj = sdir / "PRIOR.json"
    prior = f"{sdir / json.loads(pj.read_text())['file']}:{json.loads(pj.read_text())['name']}" if pj.exists() else None
    smp = json.loads((sdir / "SAMPLER.json").read_text())
    return args, model.to(device).eval(), int(ck.get("global_step", -1)), prior, smp


def assign(days, full, world):
    """deterministic cost-balanced deal of the days to the ranks (full-field days first)"""
    order = sorted(days, key=lambda t: (-(COST_FULL if t in full else COST_SUB), t))
    load, mine = np.zeros(world), [[] for _ in range(world)]
    for t in order:
        r = int(np.argmin(load))
        mine[r].append(t)
        load[r] += COST_FULL if t in full else COST_SUB
    return mine


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", required=True, help="final-system folder (runs_global/pmA_final23 or pmA_final)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--days", default="all", help="all | comma-separated day indices of the year")
    ap.add_argument("--full-every", type=int, default=3, help="full-field products every K-th day (0 = none)")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--st-pos", default=str(ST_POS))
    ap.add_argument("--check", default=None, help="2023 hold-out date: eval_x reproduction check only (no 2024)")
    a = ap.parse_args()
    assert a.chunk % SUB_STRIDE == 0
    e = os.environ
    rank, world = int(e.get("SLURM_PROCID", 0)), int(e.get("SLURM_NTASKS", 1))
    dev = torch.device(f"cuda:{int(e.get('SLURM_LOCALID', 0))}") if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    t0 = time.time()
    rargs, model, step, prior_spec, sj = load_system(a.system, dev)
    spec = noise_spec(rargs)
    store = open_store(rargs.store, **store_kwargs(rargs))          # firewall on: sealed / station files unreadable
    pix = PixelArrays(store, dev)
    grid = model_grid(rargs.store, rargs.grid)
    geom = CellGeometry(store, grid)
    variant = getattr(rargs, "variant", "base")
    cm = CellMeans(geom, dev) if variant == "hcf" else None
    M = a.members
    prior = None
    if prior_spec:
        from cdffm_aq_x.copula_x import load_prior
        prior = load_prior(prior_spec, model, spec, M, dev)
    zb, pmax, z0t = (sj.get(k) for k in ("z_bound", "pmax_ug", "z0_trunc"))
    smp = Sampler(model, variant, spec, M, a.ode_steps, dev, cm=cm, flowmap=rargs.objective == "flowmap", prior=prior,
                  coarse_spec=coarse_spec(rargs, spec), z_bound=zb, z0_trunc=z0t)
    bcsd = dict(np.load(BCSD)) if BCSD.exists() else None     # A2(d): B2 only if the GHAP BCSD was fitted in time
    if bcsd is not None:
        assert np.array_equal(bcsd["cells"].astype(np.int64), geom.cells), "bcsd_G.npz cells differ from CellGeometry"
    arms_ = ARMS if bcsd is not None else tuple(x for x in ARMS if x != "bcsd")
    TMAX = float(getattr(store, "train_max_ug", 440.0))
    N = store.n_pix
    sub = np.arange(0, N, SUB_STRIDE)
    out = Path(a.out)
    system = {"dir": str(a.system), "run": Path(a.system).name, "args_name": rargs.name, "step": step, "split": rargs.split,
              "sigma_k": rargs.sigma_k, "prior": prior_spec, "sampler": {"z_bound": zb, "pmax_ug": pmax, "z0_trunc": z0t},
              "members": M, "ode_steps": a.ode_steps}
    if rank == 0:
        print(f"[predict] system {system} | world {world} | store split {store.split} n_pix {N:,} sub {sub.size:,} "
              f"({time.time() - t0:.0f} s)", flush=True)

    def start(d, t):
        doy = torch.tensor([doy_of(d)], device=dev)
        raw = raw_to_device(store.raw_inputs(d), dev)
        feat = model.encode(assemble_inputs(raw, pix, rargs.grid, rargs.transport))
        gens = []
        for off in (23, 29, 37, 41):
            g = torch.Generator(device=dev)
            g.manual_seed(a.seed * 1_000_003 + t * 1009 + off)
            gens.append(g)
        smp.start_day(feat, gens[0], gen_coarse=gens[2], gen_prior=gens[3])
        smp.n_bound = smp.n_trunc = 0
        return doy, raw, gens[1]

    def members_at(kk, doy):
        mu, sig = pix.mu_sigma(kk[None], doy)
        z = smp.members(pix.lat[kk], pix.lon[kk], pix.static[kk], doy)
        return z_to_ug(z, mu, sig, pmax=pmax)

    def model_pass(doy, pos):
        """cdffm members (M, len(pos)) float16 on the device at sorted positions pos"""
        pt = torch.from_numpy(pos).to(dev)
        res = torch.empty(M, pos.size, dtype=torch.float16, device=dev)
        for s0 in range(0, pos.size, a.chunk):
            res[:, s0:s0 + a.chunk] = members_at(pt[s0:s0 + a.chunk], doy).half()
        return res

    def full_pass(doy, st_sorted, truth=None):
        """all mask pixels: ensmean, counts > 37.5 / 75, members at the subsample and at st_sorted; truth (check mode):
        only this rank's chunks (chunk index % world == rank), with the eval_x day scores accumulated"""
        mean = torch.empty(N, dtype=torch.float16, device=dev)
        n1 = torch.empty(N, dtype=torch.uint8, device=dev)
        n2 = torch.empty(N, dtype=torch.uint8, device=dev)
        msub = torch.empty(M, sub.size, dtype=torch.float16, device=dev)
        mst = torch.empty(M, st_sorted.size, dtype=torch.float16, device=dev)
        st_t = torch.from_numpy(st_sorted).to(dev)
        acc = {"crps": 0.0, "se": 0.0, "w": 0.0}
        for ci, s0 in enumerate(range(0, N, a.chunk)):
            if truth is not None and ci % world != rank:
                continue
            e_ = min(N, s0 + a.chunk)
            kk = torch.arange(s0, e_, device=dev)
            ens = members_at(kk, doy)
            mean[s0:e_] = ens.mean(0).half()
            n1[s0:e_] = (ens > 37.5).sum(0).to(torch.uint8)
            n2[s0:e_] = (ens > 75.0).sum(0).to(torch.uint8)
            msub[:, s0 // SUB_STRIDE:(e_ + SUB_STRIDE - 1) // SUB_STRIDE] = ens[:, ::SUB_STRIDE].half()
            i0, i1 = np.searchsorted(st_sorted, [s0, e_])
            if i1 > i0:
                mst[:, i0:i1] = ens[:, st_t[i0:i1] - s0].half()
            if truth is not None:
                y = truth[kk]
                w = pix.w_area[kk].double()
                acc["crps"] += float((fair_crps(ens, y).double() * w).sum())
                acc["se"] += float((((ens.mean(0) - y).double() ** 2) * w).sum())
                acc["w"] += float(w.sum())
        return mean, n1, n2, msub, mst, acc

    if a.check:                                   # --------------------------- 2023 reproduction check (no 2024)
        d = dt.date.fromisoformat(a.check)
        t = store.hold_dates.index(d)
        doy, raw, _ = start(d, t)
        truth = torch.from_numpy(store.holdout(t).astype(np.float32)).to(dev).clamp(min=1.0)
        stp = np.sort(np.random.default_rng(0).choice(N, 20000, replace=False))
        t1 = time.time()
        _, _, _, msub, mst, acc = full_pass(doy, stp, truth=truth)
        tf = time.time() - t1
        sums = torch.tensor([acc["crps"], acc["se"], acc["w"], smp.n_bound, smp.n_trunc], dtype=torch.float64, device=dev)
        # subsample-only path on the same day (fresh day start: same seeds), compared at this rank's chunks
        doy, raw, _ = start(d, t)
        t1 = time.time()
        ms2 = model_pass(doy, sub)
        ts = time.time() - t1
        mine = np.zeros(sub.size, bool)
        for ci, s0 in enumerate(range(0, N, a.chunk)):
            if ci % world == rank:
                mine[s0 // SUB_STRIDE:(min(N, s0 + a.chunk) + SUB_STRIDE - 1) // SUB_STRIDE] = True
        mt = torch.from_numpy(mine).to(dev)
        diff = float((msub[:, mt].float() - ms2[:, mt].float()).abs().max())
        rel = float(((msub[:, mt].float() - ms2[:, mt].float()).abs() / ms2[:, mt].float().clamp(min=1)).max())
        row = dict(rank=rank, date=str(d), full_pass_s_for_1_over_world=tf, sub_pass_s=ts, max_abs_diff_sub_vs_full=diff,
                   max_rel_diff=rel, sums=sums.tolist())
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f"check_{d}_r{rank}.json").write_text(json.dumps(row))
        print(f"[check r{rank}] {row}", flush=True)
        return

    # ------------------------------------------------------------------------------------------- 2024 predictions
    st = np.asarray(np.load(a.st_pos)["pos"], np.int64)
    assert st.ndim == 1 and st.size > 0 and st.min() >= 0 and st.max() < N, (st.shape, st.min(), st.max())
    st_sorted = np.unique(st)
    st_back = np.searchsorted(st_sorted, st)                      # st = st_sorted[st_back]
    U = np.union1d(sub, st_sorted)
    tt = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.int64)).to(dev)
    i_sub, i_st, st_back_t = tt(np.searchsorted(U, sub)), tt(np.searchsorted(U, st)), tt(st_back)
    y0 = dt.date(a.year, 1, 1)
    ndays = (dt.date(a.year + 1, 1, 1) - y0).days
    days = list(range(ndays)) if a.days == "all" else [int(x) for x in a.days.split(",")]
    full = {t for t in days if a.full_every > 0 and t % a.full_every == 0}
    mine = assign(days, full, world)[rank]
    log = out / "logs" / f"predict_r{rank:02d}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[predict r{rank}] {len(mine)} days ({sum(t in full for t in mine)} full) | st {st.size} positions "
          f"({st_sorted.size} distinct) | U {U.size:,}", flush=True)
    for it, t in enumerate(mine):
        d = y0 + dt.timedelta(t)
        files = [out / s / arm / f"{d}.npz" for s in ("preds", "preds_st") for arm in arms_]
        if t in full:
            files.append(out / "fields" / "cdffm" / f"{d}.npz")
        if all(f.exists() for f in files):
            continue
        t1 = time.time()
        doy, raw, gc = start(d, 1000 + t)
        meta = dict(date=str(d), year_day=t, seed_key=1000 + t, system=system, subsample=f"every {SUB_STRIDE}th mask pixel",
                    st_pos=str(a.st_pos), units="ug m-3", truth_not_read=True)
        san = {}
        if t in full:
            mean, n1, n2, msub, mst, _ = full_pass(doy, st_sorted)
            ens_sub, ens_st = msub, mst[:, st_back_t]
            save_npz(out / "fields" / "cdffm" / f"{d}.npz", ensmean=mean.cpu().numpy(), n_gt_37p5=n1.cpu().numpy(),
                     n_gt_75=n2.cpu().numpy(), meta=np.array(json.dumps(dict(meta, arm="cdffm", members=M,
                     order="GHAP mask order (data/processed/ghap/grid/ghap_mask.npy)", prob="n / members"))))
            san["full_max_ensmean"] = float(mean.float().max())
        else:
            mu_ = model_pass(doy, U)
            ens_sub, ens_st = mu_[:, i_sub], mu_[:, i_st]
        tm = time.time() - t1
        san.update(n_member_at_bound=int(smp.n_bound), n_member_z0_trunc=int(smp.n_trunc),
                   max_member_sub=float(ens_sub.float().max()), n_pix_sub_gt_440=int((ens_sub.float().max(0).values > TMAX).sum()), gt_440_means_gt_ug=TMAX,
                   n_member_sub_at_pmax=int((ens_sub.float() >= (pmax or np.inf) * 0.999).sum()))
        # baselines at U
        anom = torch.from_numpy(bcsd_anomaly_grid(bcsd, geom, raw["cams"][0].float().cpu().numpy(), doy_of(d), d.month)).to(dev) \
            if bcsd is not None else None
        pt = torch.from_numpy(U).to(dev)
        base = {k: torch.empty(n_, U.size, dtype=torch.float16, device=dev)
                for k, n_ in (("cams_bilinear", 1), ("bcsd", 1), ("clim", M)) if k in arms_}
        for s0 in range(0, U.size, a.chunk):
            kk = pt[s0:s0 + a.chunk]
            lat, lon = pix.lat[kk], pix.lon[kk]
            mu, sig = pix.mu_sigma(kk[None], doy)
            base["cams_bilinear"][0, s0:s0 + a.chunk] = bilinear_log1p_at(raw["cams"][0], lat, lon, store.grid_native).half()
            if anom is not None:
                base["bcsd"][0, s0:s0 + a.chunk] = torch.expm1(mu[0] + bilinear_at(anom, lat, lon, grid)).clamp(min=0.0).half()
            zc = torch.randn(M, kk.numel(), device=dev, generator=gc)
            base["clim"][:, s0:s0 + a.chunk] = z_to_ug(zc, mu, sig).half()
        arms = {"cdffm": (ens_sub, ens_st)}
        arms.update({k: (v[:, i_sub], v[:, i_st]) for k, v in base.items()})
        for arm, (xs, xt) in arms.items():
            m_ = dict(meta, arm=arm, members=int(xs.shape[0]))
            if arm != "cdffm":
                m_["system"] = {"baseline": arm, "file": str(BCSD) if arm == "bcsd" else None}
            save_npz(out / "preds" / arm / f"{d}.npz", members=xs.cpu().numpy(), meta=np.array(json.dumps(m_)))
            save_npz(out / "preds_st" / arm / f"{d}.npz", members=xt.cpu().numpy(), meta=np.array(json.dumps(m_)))
        row = dict(date=str(d), t=t, full=t in full, model_s=tm, total_s=time.time() - t1, sanity=san)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[predict r{rank}] {d} {'FULL' if t in full else 'sub '} model {tm:.0f} s total {time.time() - t1:.0f} s | "
              f"max member {san['max_member_sub']:.0f} >440 {san['n_pix_sub_gt_440']} bound {san['n_member_at_bound']} "
              f"({it + 1}/{len(mine)}, {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[predict r{rank}] done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
