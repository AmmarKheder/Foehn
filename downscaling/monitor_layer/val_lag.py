#!/usr/bin/env python3
"""A11 experiment M (Amendment A11, Addendum 2 point 1): validation of the monitor-fed layer for FORECASTS against the
1 km estimates of 2021 ONLY, at the pixels of the selection set of Amendments A2/A3 (the 24 SEL days of 2021 x the
100,000 mask pixels of train_aq.ValData, seed 0). No monitor value is read as a truth; the held-out monitors are not
read at all (LagStationSet: assimilation files only, held-out ids asserted, held-out value files firewalled).

Arms, on the same pixels with the same noise (cdffm_aq_m/lagval.py; 8 members x 10 Euler steps, the sampler of the
in-training validation, for every arm: no prior, no guard, so that a new run and the frozen systems are compared alike):
  M@<step>            a --stn-lag run (kind "lag"): the 63 features of the assimilation monitors of day t - lag
  M@<step>|nostations the same network with every monitor hidden (must stay close to the station-free model)
  pmS_final           the same-day model of A3 fed with the monitors of day t - lag: the NAIVE baseline (out of its design
                      for lag >= 1; lag 0 is its design, the analysis of the day)
  pmS_final|nostations
  pmG_final           the station-free model (does not depend on the lag)
  climatology_median, cams_bilinear   B0 and B1 of A2 (one "member": fair CRPS = MAE)
Scores: fair CRPS, RMSE of the ensemble mean, spread/skill (mean std / RMSE, and Fortin's form), bias, in ug m-3, by lag
and by distance class of the pixel to its nearest assimilation monitor with a valid value on day t - lag: all, 0-10,
10-50, 50-150 km, beyond 150 km.

  python -u val_lag.py --run pmM_base --ckpt all                 every saved, validated step of the run + the frozen arms
  python -u val_lag.py --run pmM_smoke --ckpt 100 --days 0,180 --pixels 4000 --lags 1,5 --save-members     (smoke)
  python -u val_lag.py --lags 0,1,2,3,4,5                         the frozen arms only (no --run)
-> <runs-root>/<run>/val_lag/<tag>.json (+ .md table; + _members.npz with --save-members); exit 1 if anything is not finite.
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

HERE = Path(__file__).resolve().parent
ROOT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from cdffm_aq_m.data_aq import PixelArrays, in_channels, open_store, store_kwargs  # noqa: E402
from cdffm_aq_m.eval_aq import saved_steps  # noqa: E402
from cdffm_aq_m.lagval import CLASS_NAMES, NOSTN, arm_kind, validate_lag  # noqa: E402
from cdffm_aq_m.model_aq import build_model  # noqa: E402
from cdffm_aq_m.stations import ASSIM_ROOT, LagStationSet  # noqa: E402
from cdffm_aq_m.store_global import SEL_HOLD_IDX  # noqa: E402
from cdffm_aq_m.train_aq import ValData, model_grid, noise_spec  # noqa: E402

SAME = ("store", "fires", "split", "sigma_k", "grid", "transport", "objective", "doy_cond", "noise_features", "noise_beta",
        "noise_lmin_km", "noise_lmax_km", "static_dim")


def load_ema(args_json, ckpt, device):
    """(args, EMA model in eval mode on the device, step) from a run's args.json and one checkpoint, built with the COPY"""
    args = SimpleNamespace(**json.loads(Path(args_json).read_text()))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k[len("ema."):]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
    model = build_model(args, in_channels(args.transport), args.static_dim, model_grid(args.store, args.grid))
    model.load_state_dict(sd)
    return args, model.to(device).eval(), int(ck.get("global_step", -1))


def run_steps(run_dir, which):
    """the checkpoints of a run to validate: all = saved steps that the training validated; last; or a list of steps"""
    saved = saved_steps(run_dir / "ckpt")
    if which == "all":
        mf = run_dir / "metrics.jsonl"
        done = {json.loads(line)["step"] for line in open(mf) if line.strip()} if mf.exists() else set(saved)
        steps = sorted(set(saved) & done)
    elif which == "last":
        steps = [max(saved)]
    else:
        steps = [int(x) for x in which.split(",")]
    missing = [s for s in steps if s not in saved]
    if missing or not steps:
        raise SystemExit(f"{run_dir}: no checkpoint for steps {missing or which} (saved: {sorted(saved)})")
    return [(s, Path(saved[s])) for s in steps]


def table(res, key, fmt="{:.3f}"):
    """markdown: one block per lag, rows = arms, columns = distance classes"""
    out = []
    for lg in res["lags"]:
        out.append(f"\n**{key}, lag {lg}** (monitors of day t - {lg})\n")
        out.append("| arm | " + " | ".join(CLASS_NAMES) + " |")
        out.append("|---|" + "---|" * len(CLASS_NAMES))
        for lb, by in res["scores"].items():
            cells = [(fmt.format(by[str(lg)][c][key]) if by[str(lg)][c] is not None and key in by[str(lg)][c] else "-")
                     for c in CLASS_NAMES]
            out.append(f"| {lb} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=None, help="the --stn-lag run under --runs-root (omit: frozen arms only)")
    ap.add_argument("--runs-root", default=str(HERE / "runs"))
    ap.add_argument("--ckpt", default="all", help="all (saved and validated steps) | last | STEP[,STEP...]")
    ap.add_argument("--free", default=str(ROOT / "runs_ghap/pmG_final"), help="station-free frozen system ('' = none)")
    ap.add_argument("--naive", default=str(ROOT / "runs_ghap/pmS_final"), help="same-day A3 system fed with lagged monitors ('' = none)")
    ap.add_argument("--lags", default="1,2,3,4,5", help="lags to score; 0 = the same-day analysis (reference, not a forecast)")
    ap.add_argument("--days", default=None, help="hold-out day indices of 2021 (default: the 24 SEL days 0,15,..,345)")
    ap.add_argument("--pixels", type=int, default=100000, help="validation pixels (ValData, seed 0)")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0, help="noise seed of train_aq.validate (the runs' --seed 0)")
    ap.add_argument("--assim-root", default=str(ASSIM_ROOT))
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--save-members", action="store_true", help="write the members (small runs only)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto", help="auto (GPU if any) | cpu")
    a = ap.parse_args()
    t0 = time.time()
    if a.device != "cpu" and torch.cuda.is_available():
        dev = torch.device(f"cuda:{int(os.environ.get('SLURM_LOCALID', 0))}")
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")
    lags = [int(x) for x in a.lags.split(",")]
    arms, meta, ref = [], {}, None

    def add_arm(name, args_json, ckpt, want):
        nonlocal ref
        rargs, model, step = load_ema(args_json, ckpt, dev)
        kind = arm_kind(model)
        if kind != want:
            raise SystemExit(f"{name}: a '{kind}' model where a '{want}' model is expected ({args_json})")
        if ref is None:
            ref = rargs
        diff = {k: (getattr(rargs, k, None), getattr(ref, k, None)) for k in SAME if getattr(rargs, k, None) != getattr(ref, k, None)}
        if diff:
            raise SystemExit(f"{name}: settings differ from the first arm: {diff}")
        arms.append(dict(name=name, model=model, kind=kind, grid=rargs.grid, transport=rargs.transport,
                         flowmap=rargs.objective == "flowmap"))
        meta[name] = dict(kind=kind, args=str(args_json), ckpt=str(ckpt), step=step, run=getattr(rargs, "name", None),
                          stn_lag=getattr(rargs, "stn_lag", None), stn_lag0_prob=getattr(rargs, "stn_lag0_prob", None))

    run_dir = None
    if a.run:
        run_dir = Path(a.runs_root) / a.run
        for step, path in run_steps(run_dir, a.ckpt):
            add_arm(f"M@{step}", run_dir / "args.json", path, "lag")
    if a.naive:
        add_arm(Path(a.naive).name, Path(a.naive) / "args.json", Path(a.naive) / "ckpt" / "best.ckpt", "a3")
    if a.free:
        add_arm(Path(a.free).name, Path(a.free) / "args.json", Path(a.free) / "ckpt" / "best.ckpt", "free")
    if not arms:
        raise SystemExit("nothing to validate")
    if ref.store not in ("ghap", "toy") or ref.objective != "fm":      # toy: the CPU test of tests/test_m.py only
        raise SystemExit("val_lag.py: flow-matching runs on the GHAP store only")
    spec = noise_spec(ref)
    store = open_store(ref.store, **store_kwargs(ref))                     # GhapStore: firewall on
    days = list(SEL_HOLD_IDX) if a.days is None else [int(x) for x in a.days.split(",")]
    dates = [store.hold_dates[t] for t in days]
    assert all(d.year == 2021 for d in dates), "validation against the 1 km estimates of 2021 only"
    years = sorted({(d - dt.timedelta(days=lg)).year for d in dates for lg in lags})
    stations = LagStationSet(a.assim_root, years=years, device=dev, clim_mean=store.clim_mean, pix_lat=store.lat,
                             pix_lon=store.lon)                            # + firewall on the held-out value files
    pix = PixelArrays(store, dev)
    val = ValData(store, a.pixels, days, heldout_file=None)                # never the held-out monitor file
    print(f"[val_lag] arms {[(x['name'], x['kind']) for x in arms]} | lags {lags} | {len(days)} days of 2021 x "
          f"{len(val.pix):,} pixels | M={a.members} x {a.ode_steps} Euler | {stations.n_total:,} assimilation monitors "
          f"over {years}, held-out ids refused: {len(stations.heldout_ids):,} | device {dev} ({time.time() - t0:.0f} s)", flush=True)
    members = {} if a.save_members else None
    res = validate_lag(arms, store, pix, val, stations, lags, a.members, a.ode_steps, spec, a.seed, dev, chunk=a.chunk,
                       baselines=not a.no_baselines, members_out=members, log=lambda s: print(s, flush=True))
    free_name = Path(a.free).name if a.free else None
    vs_free = {}
    if free_name:                                                          # CRPS ratio to the station-free system
        for lb, by in res["scores"].items():
            vs_free[lb] = {lg: {c: (by[lg][c]["crps_fair"] / res["scores"][free_name][lg][c]["crps_fair"]
                                    if by[lg][c] is not None and res["scores"][free_name][lg][c] is not None else None)
                                for c in CLASS_NAMES} for lg in by}
    finite = res["info"]["members_finite"] and res["info"]["scores_finite"]
    tag = a.tag or f"lags{''.join(str(x) for x in lags)}_d{len(days)}_p{len(val.pix)}_M{a.members}s{a.ode_steps}"
    out = Path(a.out) if a.out else ((run_dir if run_dir else Path(a.runs_root) / "_frozen_arms") / "val_lag" / f"{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(what="A11 experiment M: lag validation against the 1 km estimates of 2021 (val_lag.py)", arms=meta,
               lags=res["lags"], classes=res["classes"], scores=res["scores"], crps_ratio_to_free=vs_free, pooled=res["pooled"],
               baselines=res["baselines"], info=res["info"], all_finite=bool(finite), seconds=time.time() - t0,
               assim_root=str(a.assim_root), split_v1_sha256=stations.split_sha256,
               rule="no monitor value is a truth; assimilation monitors of day t - lag are inputs; held-out monitors never read")
    out.write_text(json.dumps(rec, indent=1))
    md = [f"# val_lag {tag}", "", f"{len(days)} days of 2021 x {len(val.pix):,} pixels, {a.members} members x {a.ode_steps} Euler; "
          f"truth = 1 km estimates; arms: " + ", ".join(f"{k} ({v['kind']}, step {v['step']})" for k, v in meta.items()), "",
          "Weight of each distance class (cos(lat)-weighted pixel-days): " + "; ".join(
              f"lag {lg}: " + ", ".join(f"{c} {w:.0f}" for c, w in cw.items()) for lg, cw in res["info"]["class_weight"].items()),
          table(res, "crps_fair"), table(res, "rmse_ensmean"), table(res, "spread_skill_fortin", "{:.2f}"),
          table(res, "spread_skill", "{:.2f}"), table(res, "bias", "{:+.3f}")]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    if members is not None:
        np.savez(str(out.with_suffix("")) + "_members.npz", **{f"{lb}|lag{lg}|{d}": x for (lb, lg, d), x in members.items()})
    print("\n".join(md), flush=True)
    print(f"[val_lag] members finite {res['info']['members_finite']} scores finite {res['info']['scores_finite']} | "
          f"{time.time() - t0:.0f} s -> {out}", flush=True)
    sys.exit(0 if finite else 1)


if __name__ == "__main__":
    main()
