#!/usr/bin/env python3
"""Chain-mode inference of the monitor-fed layer (experiment M) at the 12,136 catalogue monitor pixels, for issue days of
the autumn 2026 record (draft Addendum 8 of A11, ADDENDUM_draft.md). One GCD. Nothing here reads an observation used
as a truth (audit-hook firewall, ft_lib.TRUTH_MARKERS) and no held-out value (the monitor inputs are data/inputs.npz,
assimilated monitors by place only).

Code path = realtime_1km/replay/code/infer_points.py (the record's pointwise inference), lines copied and marked "[copy
infer_points]", evaluated at the monitor pixels only (the 11,278 distinct pixels of positions.npz 'mon' instead of the
875,425 positions; the computation is pointwise, so a pixel's members do not depend on the other pixels). Per issue day d
and lead day L (valid day d + L - 1), for coarse member m = 0..7 (member m of the record's coarse forecast, through the
record's encoder channels work/<d>/chan_L<L>.npz), noise parameters drawn once per (seed, L, m):
  base = seed * 1000003 + L * 1009 + m * 7919; generators base + 23 (sphere noise), + 37 (coarse flow, unused), + 41 (prior).
Arms:
  free  the station-free chain recomputed (pmG_final best.ckpt, fitted prior P3, guard of SAMPLER.json, 10 Euler steps):
        must equal the record's km (identity check, tolerance of the replay's own check);
  free0 DIAGNOSTIC (check 4, not in the decision): pmG_final WITHOUT the fitted prior, i.e. the stationary sphere
        noise of generator base + 23 -- the very base noise of m1 and m2 -- with the same guard and steps; m1 vs free0
        isolates the monitors (and M's network) at matched noise and no prior, free0 vs free the prior;
  m1    M (data/selection.json: pmM_base, RULE S) with the assimilated monitors of day d - 1 at lag L (the primary arm);
  m2    M with those of day d - 2 at lag L + 1, lead days 1-4 (lead day 5 would need lag 6: NaN here, the station-free
        member in the scorer) -- the declared sensitivity arm.
Where M differs from the station-free chain: (1) its own network (EMA weights of the selected step, loaded with the
copy cdffm_aq_m as a11/stageM/val_lag.py.load_ema); (2) NO fitted prior: the copula prior P3 of pmG_final was fitted on
pmG_base's 66-dimensional condition and cannot take M's 98-dimensional one, and M's 2021 validation used no prior; the
base noise is then the stationary sphere noise of generator base + 23 (Sampler with prior=None); (3) the station
features (cdffm_aq_m.stations.lag_features, 63 floats) of the monitors of the input day. Same encoder channels, same
guard (z_bound 10, member cap 1000 ug m-3, base noise truncated at |z0| = 3), same 10 Euler steps, same seeds, same
production Sampler (cdffm_aq_x.model_x.Sampler).

    python -u infer_m.py --days 2026-10-20 [--arms free,free0,m1,m2] [--inputs data/inputs.npz] [--out out/forecasts.npz]
                         [--keep-free] [--trace out/trace.json] [--save-every 6]
    --days all is THE FULL RUN: it refuses any configuration but the frozen one (FULL_RUN below; check 2).
Output (one .npz, resumable, written through a temporary file + fsync + rename): days (D,), done (D,), free0, m1, m2 [, free]
(D, 5, 8, 12136) float32 ug m-3 at the catalogue monitors (catalogue order; NaN where not made), dmin1, dmin2 (D, 12136)
float32 = distance (km) from the monitor's pixel to the nearest assimilated monitor with a value on the input day (as
the network sees it), nmon (D, 2), prov (D,) JSON per day (identity, guard hits, timings, SHA-256 of the code and inputs).
"""
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

FT = Path(__file__).resolve().parent
sys.path.insert(0, str(FT))
import ft_lib as F  # noqa: E402

P = F.P
for p_ in (str(F.SM), str(P / "realtime_1km"), str(P)):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from cdffm_aq.data_aq import Norm, PixelArrays, assemble_inputs, in_channels, raw_to_device, z_to_ug  # noqa: E402
from cdffm_aq.store_ghap import GHAP_LAT0, GHAP_LON0, GHAP_MARKERS, GHAP_NC, GHAP_RES, GHAP_ROOT  # noqa: E402
from cdffm_aq.store_global import install_firewall  # noqa: E402
from cdffm_aq.train_aq import model_grid, noise_spec  # noqa: E402
from cdffm_aq_x.copula_x import SampledPrior, load_prior  # noqa: E402
from cdffm_aq_x.model_x import Sampler  # noqa: E402
from cdffm_aq_x.train_x import coarse_spec  # noqa: E402
import infer as PROD  # noqa: E402  (production module: load_ckpt, guard_value; main() guarded)
from cdffm_aq_m.model_aq import build_model as build_model_m, eval_sphere_noise, integrate  # noqa: E402
from cdffm_aq_m.stations import LAG_MAX, STN_DIM_LAG, harmonic_basis, lag_features, unit_xyz  # noqa: E402

NLEAD, NMEM = 5, 8
ARMS = ("free", "free0", "m1", "m2")
FULL_RUN = dict(arms="free,free0,m1,m2", seed=0, ode_steps=10, members=NMEM, leads="1,2,3,4,5")   # the only --days all configuration


def load_m(ckpt, device):
    """[copy a11/stageM/val_lag.py load_ema] (args, EMA model in eval mode, step) of a pmM_base checkpoint, built with the copy"""
    run_dir = Path(ckpt).resolve().parent.parent
    args = SimpleNamespace(**json.loads((run_dir / "args.json").read_text()))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k[len("ema."):]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
    model = build_model_m(args, in_channels(args.transport), args.static_dim, model_grid(args.store, args.grid))
    model.load_state_dict(sd)
    return args, model.to(device).eval(), int(ck.get("global_step", -1))


def sub_store_rows(q, rargs):
    """[copy infer_points.sub_store] the GhapStore attributes for the rows q of positions.npz (the monitor pixels),
    from positions_store.npz, spot-checked against the memory-mapped store files"""
    assert rargs.store == "ghap", rargs.store
    man = json.loads((GHAP_ROOT / "manifest.json").read_text())
    Z = np.load(F.POSITIONS)
    C = np.load(F.POSITIONS.with_name("positions_store.npz"))
    assert np.array_equal(C["pos"], Z["pos"]), "positions_store.npz does not match positions.npz"
    pos = Z["pos"].astype(np.int64)[q]
    flat = C["flat"].astype(np.int64)[q]
    k = np.sort(np.random.default_rng(0).choice(q.size, 512, replace=False))
    assert np.array_equal(np.asarray(np.load(GHAP_ROOT / "grid/ghap_mask.npy", mmap_mode="r")[pos[k]]).astype(np.int64), flat[k])
    cols = {}
    for name, f in (("clim_mean", "ghap_clim/clim_mean_G.npy"), ("clim_var", "ghap_clim/clim_var_G.npy"), ("static", "ghap_clim/static_G.npy")):
        cols[name] = np.ascontiguousarray(C[name][q])
        ref = np.load(GHAP_ROOT / f, mmap_mode="r")
        assert cols[name].dtype == ref.dtype and np.array_equal(np.asarray(ref[pos[k]]), cols[name][k]), f"{name}: cache differs from {f}"
    r, c = flat // GHAP_NC, flat % GHAP_NC
    nc = man["normalisation"]
    return SimpleNamespace(lat=GHAP_LAT0 - GHAP_RES * r, lon=GHAP_LON0 + GHAP_RES * c, clim_mean=cols["clim_mean"],
                           clim_var=cols["clim_var"], static=cols["static"], sigma_floor=float(man["static"]["sigma_floor"]),
                           norm=Norm(nc["cams_log1p_mean_std"][0], nc["cams_log1p_mean_std"][1], nc["t6_channel_mean"], nc["t6_channel_std"]),
                           sigma_k=int(getattr(rargs, "sigma_k", 0)), train_max_ug=float(man["T_G_train_max_points"])), pos


def mon_from_arrays(lat, lon, val, clim, pos_ok, src, lag, device):
    """the dict of cdffm_aq_m.stations.LagStationSet.lagged(d, lag) built from arrays of the monitors of the input day
    `src` (lat, lon float32 degrees; val ug m-3; clim (m, 5) harmonic coefficients of the 1 km climatology at the
    monitor's own pixel; pos_ok: pixel inside the mask). Same arithmetic as lagged(): anomaly = log1p(max(val, 1)) - mu"""
    lag = int(lag)
    assert 1 <= lag <= LAG_MAX, lag
    t = lambda x, dtype: torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype)
    la, lo, v = t(lat.astype(np.float32), torch.float32), t(lon.astype(np.float32), torch.float32), t(val.astype(np.float32), torch.float32)
    col = (src - dt.date(src.year, 1, 1)).days
    mu = (t(clim.astype(np.float32), torch.float32) * harmonic_basis(col).to(device)).sum(-1)
    anom = torch.log1p(v.clamp(min=1.0)) - mu
    ok = t(pos_ok.astype(bool), torch.bool)
    return dict(lat=la, lon=lo, val=v, xyz=unit_xyz(la, lo), anom=anom, anom_ok=ok, idx=torch.arange(v.numel(), device=device),
                src=src, year=src.year, lag=lag)


@torch.no_grad()
def members_from_z0(model, feat, lat, lon, static, doy, z0, stn, ode_steps, z_bound=None, z0_trunc=None):
    """Sampler.members (cdffm_aq_x.model_x) for a pointwise model WITHOUT prior, from a given base noise z0 (M, n, 1);
    stn (n, 63) or None (every monitor hidden). Used by the tests to tie this path to cdffm_aq_m.lagval.validate_lag"""
    M = z0.shape[0]
    cond = model.condition(feat, lat[None], lon[None], static[None], doy, stn=None if stn is None else stn[None])
    cond = cond.expand(M, -1, -1)
    if z0_trunc is not None:
        z0 = z0.clamp(-float(z0_trunc), float(z0_trunc))
    return integrate(model, z0, cond, ode_steps, flowmap=False, bound=None if z_bound is None else float(z_bound))[..., 0]


def identity(mine, rec):
    """station-free arm recomputed vs the record's km at the monitors (one lead day, (8, n) each)"""
    fa, fb = np.isfinite(mine), np.isfinite(rec)
    both = fa & fb
    if not both.any():
        return dict(n=0, same_nan=bool(np.array_equal(fa, fb)))
    d = np.abs(mine[both].astype(np.float64) - rec[both])
    rel = d / np.maximum(np.abs(rec[both].astype(np.float64)), 1.0)
    return dict(n=int(both.sum()), same_nan=bool(np.array_equal(fa, fb)), max_abs=float(d.max()), mean_abs=float(d.mean()),
                max_rel=float(rel.max()), r=float(np.corrcoef(mine[both], rec[both])[0, 1]), identical=float((d == 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True, help="comma-separated issue days, or 'all' (the 48 of the record)")
    ap.add_argument("--arms", default="free,m1,m2")
    ap.add_argument("--inputs", default=str(F.INPUTS))
    ap.add_argument("--out", default=str(F.FORECASTS))
    ap.add_argument("--keep-free", action="store_true", help="also store the recomputed station-free members (smoke)")
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=6)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--members", type=int, default=NMEM, help="DRY RUNS ONLY: members 0..K-1 (the others stay NaN)")
    ap.add_argument("--leads", default="1,2,3,4,5", help="DRY RUNS ONLY: lead days to make")
    a = ap.parse_args()
    if a.days == "all":                                     # THE FULL RUN: the frozen configuration only (check 2)
        cfg = dict(arms=a.arms, seed=a.seed, ode_steps=a.ode_steps, members=a.members, leads=a.leads)
        if cfg != FULL_RUN or Path(a.inputs).resolve() != F.INPUTS.resolve() or Path(a.out).resolve() != F.FORECASTS.resolve():
            sys.exit(f"[infer_m] REFUSED: --days all is the full run; only {FULL_RUN} with --inputs {F.INPUTS} --out {F.FORECASTS} "
                     f"is allowed (got {cfg}, --inputs {a.inputs}, --out {a.out})")
    leads = [int(x) for x in a.leads.split(",")]
    t_start = time.time()
    arms = [x for x in a.arms.split(",") if x]
    assert set(arms) <= set(ARMS) and arms, arms
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    install_firewall(GHAP_MARKERS)                          # [copy infer_points] no raw GHAP, sealed or station file
    F.install_truth_firewall(trace=a.trace is not None)     # no observation file of any kind
    record_days = F.issue_days()
    days = record_days if a.days == "all" else sorted(a.days.split(","))
    assert set(days) <= set(record_days), "issue days outside the record"
    sel = F.load_selection()
    ck_m = Path(sel["checkpoint"]["path"])
    sha_m = F.sha256_file(ck_m)
    assert sha_m == sel["checkpoint"]["sha256"], "the checkpoint of M differs from data/selection.json"
    t0 = time.time()

    # ---- the station-free system of the record [copy infer_points] -------------------------------------------------
    rargs, modelG, stepG, run_dir = PROD.load_ckpt(str(F.CKPT_FREE), dev)
    variant = getattr(rargs, "variant", "base")
    assert variant in ("base", "estab") and rargs.objective != "regression"
    assert not getattr(modelG, "stations", False)
    specG = noise_spec(rargs)
    z_bound, pmax_ug = PROD.guard_value("auto", "z_bound", run_dir), PROD.guard_value("auto", "pmax_ug", run_dir)
    z0_trunc = PROD.guard_value("auto", "z0_trunc", run_dir)
    pr = json.loads((run_dir / "PRIOR.json").read_text())
    prior_spec = f"{run_dir / pr['file']}:{pr['name']}"
    prior_net = load_prior(prior_spec, modelG, specG, 1, dev).prior
    # ---- M ---------------------------------------------------------------------------------------------------------
    margs, modelM, stepM = load_m(ck_m, dev)
    assert stepM == sel["checkpoint"]["step"] and getattr(modelM, "stations", False) and int(modelM.stn_dim) == STN_DIM_LAG
    specM = noise_spec(margs)
    assert specM == specG and margs.grid == rargs.grid and margs.transport == rargs.transport and margs.store == rargs.store
    assert margs.split == "G" and not margs.tune_stations and float(margs.stn_lag0_prob) == 0.0
    t_load = time.time() - t0

    # ---- pixels: the catalogue monitors' 1 km pixels ---------------------------------------------------------------
    Z = np.load(F.POSITIONS)
    mon = Z["mon"].astype(np.int64)
    q = np.unique(mon)
    mon_q = np.searchsorted(q, mon)
    assert np.array_equal(q[mon_q], mon)
    store, qpos = sub_store_rows(q, rargs)
    pix = PixelArrays(store, dev)
    U, n = q.size, mon.size
    lat, lon, static = pix.lat, pix.lon, pix.static
    kk = torch.arange(U, device=dev)
    with np.load(a.inputs) as z_:
        I = {k_: z_[k_] for k_ in z_.files}
    assert np.array_equal(I["cat_id"], Z["id"]), "catalogue order differs between the inputs and positions.npz"
    dlat = float(np.abs(I["cat_pixlat"] - store.lat[mon_q]).max())
    assert dlat < 1e-6, f"query pixels differ from the inputs' pixels ({dlat})"
    issue_list = [str(x) for x in I["issue"]]
    u = dict(lat=I["u_lat"], lon=I["u_lon"], clim=I["u_clim"], pos_ok=I["u_pos"] >= 0)
    sha = dict(infer_m=F.sha256_file(__file__), ft_lib=F.sha256_file(F.__file__), inputs=F.sha256_file(a.inputs),
               ckpt_m=sha_m, ckpt_free=F.sha256_file(F.CKPT_FREE), prior=F.sha256_file(prior_spec.split(":")[0]),
               selection=F.sha256_file(F.SELECTION))

    def mon_of(i_issue, k, lag):
        si = int(I["issue_src"][i_issue, k - 1])
        o0, o1 = int(I["src_off"][si]), int(I["src_off"][si + 1])
        rr = I["src_row"][o0:o1].astype(np.int64)
        src = dt.date.fromisoformat(str(I["src_day"][si]))
        return mon_from_arrays(u["lat"][rr], u["lon"][rr], I["src_val"][o0:o1], u["clim"][rr], u["pos_ok"][rr], src, lag, dev), src, rr.size

    # ---- output (resumable) ----------------------------------------------------------------------------------------
    out = Path(a.out)
    keys = arms if a.keep_free else [x for x in arms if x != "free"]
    if out.exists():
        z = np.load(out)
        assert [str(x) for x in z["days"]] == days, f"{out} holds other days: remove it or give the same --days"
        res = {k: z[k].copy() for k in z.files if k not in ("days", "prov")}
        prov = [str(x) for x in z["prov"]]
        assert set(keys) <= set(res), (keys, list(res))
    else:
        res = {k: np.full((len(days), NLEAD, NMEM, n), np.nan, np.float32) for k in keys}
        res.update(done=np.zeros(len(days), bool), dmin1=np.full((len(days), n), np.nan, np.float32),
                   dmin2=np.full((len(days), n), np.nan, np.float32), nmon=np.zeros((len(days), 2), np.int64))
        prov = [""] * len(days)

    def save():
        F.save_npz(out, days=np.array(days), prov=np.array(prov), **res)

    print(f"[infer_m] {len(days)} issue days, arms {arms}; M = {ck_m.name} (step {stepM}); free = pmG_final step {stepG}, prior "
          f"{pr['name']} (cond_dim {prior_net.cond_dim if hasattr(prior_net, 'cond_dim') else '?'}); guard z_bound {z_bound} pmax {pmax_ug} "
          f"z0_trunc {z0_trunc}; {U:,} pixels for {n:,} monitors; load {t_load:.0f} s; device "
          f"{torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}", flush=True)
    n_since = 0
    for i, d in enumerate(days):
        if res["done"][i]:
            continue
        td = time.time()
        dd = dt.date.fromisoformat(d)
        ii = issue_list.index(d)
        tim = dict(features=0.0, chan=0.0, encode_free=0.0, encode_m=0.0, sample_free=0.0, sample_free0=0.0, sample_m=0.0)
        # station features of the two arms (lag-dependent one-hot; the neighbours do not depend on the lag)
        t0 = time.time()
        feats, src_of = {1: {}, 2: {}}, {}
        for k in (1, 2):
            if f"m{k}" not in arms:
                continue
            for L in range(1, NLEAD + 1):
                lag = L + k - 1
                if lag > LAG_MAX:
                    continue
                mo, src, nm = mon_of(ii, k, lag)
                assert src == dd - dt.timedelta(days=k) and src < dd and (dd + dt.timedelta(days=L - 1) - src).days == lag
                fe, dmin = lag_features(lat, lon, mo)
                feats[k][L] = fe
                src_of[k] = str(src)
                res["nmon"][i, k - 1] = nm
                res[f"dmin{k}"][i] = dmin.cpu().numpy()[mon_q]
        tim["features"] = time.time() - t0
        rec = np.load(F.POINTS / f"{d}.npz")["km"]                    # the record's station-free members (no observation)
        leads_made, idn, hits = [], {}, {}
        for L in leads:
            f = F.WORK / d / f"chan_L{L}.npz"
            if not f.exists():
                continue
            t0 = time.time()
            chan = np.load(f)
            v = dt.date.fromisoformat(str(chan["valid"]))
            assert v == dd + dt.timedelta(days=L - 1), (d, L, v)
            doy = torch.tensor([int(chan["doy"])], device=dev)
            Mo = chan["cams"].shape[0]
            assert Mo == NMEM
            cams, met, fires, t6 = chan["cams"], chan["met"], chan["fires"], chan["t6"]
            tim["chan"] += time.time() - t0
            mu, sig = pix.mu_sigma(kk[None], doy)
            X = {k_: torch.empty(NMEM, U, device=dev) for k_ in arms}
            hl = {k_: dict(at_bound=0, z0_trunc=0, at_pmax=0) for k_ in arms}
            for m in range(a.members):
                rw = raw_to_device({"cams": cams[m], "met": met, "fires": fires, "t6": t6[m]}, dev)
                xin = assemble_inputs(rw, pix, rargs.grid, rargs.transport)
                base = a.seed * 1_000_003 + L * 1009 + m * 7919

                def gens():
                    g = torch.Generator(device=dev).manual_seed(base + 23)
                    gco = torch.Generator(device=dev).manual_seed(base + 37)
                    gp = torch.Generator(device=dev).manual_seed(base + 41)
                    return g, gco, gp
                if "free" in arms or "free0" in arms:
                    t0 = time.time()
                    featG = modelG.encode(xin)
                    tim["encode_free"] += time.time() - t0
                if "free" in arms:
                    t0 = time.time()
                    smp = Sampler(modelG, variant, specG, 1, a.ode_steps, dev, cm=None, flowmap=rargs.objective == "flowmap",
                                  coarse_spec=coarse_spec(rargs, specG), prior=SampledPrior(prior_net, 1, dev),
                                  z_bound=z_bound, z0_trunc=z0_trunc)
                    g, gco, gp = gens()
                    smp.start_day(featG, g, gen_coarse=gco, gen_prior=gp)
                    X["free"][m] = z_to_ug(smp.members(lat, lon, static, doy), mu, sig, pmax=pmax_ug)[0]
                    hl["free"]["at_bound"] += smp.n_bound
                    hl["free"]["z0_trunc"] += smp.n_trunc
                    del smp
                    tim["sample_free"] += time.time() - t0
                if "free0" in arms:                          # DIAGNOSTIC: pmG_final, no fitted prior, the base noise of m1/m2
                    t0 = time.time()
                    smp = Sampler(modelG, variant, specG, 1, a.ode_steps, dev, cm=None, flowmap=rargs.objective == "flowmap",
                                  coarse_spec=coarse_spec(rargs, specG), prior=None, z_bound=z_bound, z0_trunc=z0_trunc)
                    g, gco, gp = gens()
                    smp.start_day(featG, g, gen_coarse=gco, gen_prior=gp)
                    X["free0"][m] = z_to_ug(smp.members(lat, lon, static, doy), mu, sig, pmax=pmax_ug)[0]
                    hl["free0"]["at_bound"] += smp.n_bound
                    hl["free0"]["z0_trunc"] += smp.n_trunc
                    del smp
                    tim["sample_free0"] += time.time() - t0
                if "free" in arms or "free0" in arms:
                    del featG
                mk = [k_ for k_ in (1, 2) if f"m{k_}" in arms and L in feats[k_]]
                if mk:
                    t0 = time.time()
                    featM = modelM.encode(xin)
                    tim["encode_m"] += time.time() - t0
                    t0 = time.time()
                    for k_ in mk:
                        fe = feats[k_][L]
                        smp = Sampler(modelM, "base", specM, 1, a.ode_steps, dev, cm=None, flowmap=False,
                                      coarse_spec=coarse_spec(margs, specM), prior=None, z_bound=z_bound, z0_trunc=z0_trunc,
                                      stn_fn=lambda la, lo, fe=fe: fe)
                        g, gco, gp = gens()
                        smp.start_day(featM, g, gen_coarse=gco, gen_prior=gp)
                        X[f"m{k_}"][m] = z_to_ug(smp.members(lat, lon, static, doy), mu, sig, pmax=pmax_ug)[0]
                        hl[f"m{k_}"]["at_bound"] += smp.n_bound
                        hl[f"m{k_}"]["z0_trunc"] += smp.n_trunc
                        del smp
                    del featM
                    tim["sample_m"] += time.time() - t0
                del rw, xin
            for k_ in arms:
                if k_ in ("m1", "m2") and L not in feats[int(k_[1])]:
                    continue
                x = X[k_][:a.members]
                assert bool(torch.isfinite(x).all()), (d, L, k_, "non-finite member")
                hl[k_]["at_pmax"] = int((x >= pmax_ug).sum()) if pmax_ug is not None else 0
                hl[k_]["max_ug"] = float(x.max())
                xm = x.cpu().numpy()[:, mon_q]
                if k_ in res:
                    res[k_][i, L - 1, :a.members] = xm
                if k_ == "free":
                    idn[L] = identity(xm, rec[L - 1][:a.members])
            hits[L] = hl
            leads_made.append(L)
            del X
        if dev.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - td
        pv = dict(issue=d, leads_made=leads_made, src_days=src_of, n_monitors=res["nmon"][i].tolist(), identity=idn,
                  guard_hits=hits, timing_s={k_: round(v_, 2) for k_, v_ in tim.items()}, seconds=round(sec, 1),
                  load_seconds=round(t_load, 1), process_start_to_day_end_s=round(time.time() - t_start, 1),
                  pixels=int(U), points_per_member=int(U), members=a.members, ode_steps=a.ode_steps, seed=a.seed,
                  guard=dict(z_bound=z_bound, pmax_ug=pmax_ug, z0_trunc=z0_trunc), prior_free=prior_spec, prior_m=None, prior_free0=None,
                  ckpt_m=str(ck_m), step_m=stepM, ckpt_free=str(F.CKPT_FREE), step_free=stepG, sha256=sha, arms=arms,
                  inputs=str(a.inputs), job=os.environ.get("SLURM_JOB_ID"), torch=torch.__version__,
                  device=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                  written_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        prov[i] = json.dumps(pv, default=F._jdef)
        res["done"][i] = True
        n_since += 1
        worst = max((v_["max_rel"] for v_ in idn.values() if v_.get("n")), default=float("nan"))
        print(f"[infer_m] {d}: leads {leads_made}, {sec:.0f} s ({', '.join(f'{k_} {v_:.0f}' for k_, v_ in tim.items())}); "
              f"monitors d-1 {res['nmon'][i, 0]}, d-2 {res['nmon'][i, 1]}; identity free vs record max rel {worst:.2e}", flush=True)
        if n_since >= a.save_every:
            save()
            n_since = 0
    save()
    if a.trace:
        F.save_json(a.trace, dict(opened=F.opened_paths()))
    print(f"[infer_m] done {int(res['done'].sum())}/{len(days)} days in {(time.time() - t_start) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
