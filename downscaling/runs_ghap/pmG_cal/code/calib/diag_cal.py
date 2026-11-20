"""calib.diag_cal -- scoring of ONE fixed model (the frozen system or a fine-tuned checkpoint) with the inference sampler
on the A6 validation sets (calib.core.validate_set), without any training. Several processes (srun -n 8) share the
days and add their sums; every process computes the same pooled numbers after the reduction, so the temperature search
runs identically on all of them. Tasks:
  cache   build the validation sets (one process): V100k 'sel', hot-spot 'probe', 'fit' (48 V3-odd days), in-sample
          'ins2020' / 'ins2018'  -> cache/valset_<name>.npz
  d0      A6(c) reference + A6(d) diagnostics of a model: sel, probe, ins2020, ins2018 with 10 Euler steps; sel with 20
          and 40 Euler steps
  score   the sets of --sets with --ode-steps and --tau
  tfit    A6(e) T1: tau with pooled pixel spread/skill (ug, Fortin) = 1 on 'fit' (bisection in [1, 3], tolerance 0.005)
  tgrid   A6(e) T*: the grid tau = 1.05 .. 2.00 on 'sel' and 'probe' (stage-1 inputs); with --tau-list, those values
Models: --run NAME --ckpt STEP under --runs-root (default pmG_cal/runs; 'pmG_ref' = the frozen system: args.json copy and
a link to pmG_final/ckpt/best.ckpt as step022000.ckpt). Output: results/<tag>.json (+ <tag>_rows.npz, per-day sums).

  srun -n 8 ... python -u -m calib.diag_cal d0 --run pmG_ref --ckpt 22000 --tag d0_frozen
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cdffm_aq.data_aq import PixelArrays, open_store, store_kwargs
from cdffm_aq.eval_aq import dist_setup
from cdffm_aq_x.model_x import load_run_x

from .core import (CAL, RESULTS, ValSet, boot_pooled, build_valsets, daily_table, guard, load_prior_module, reduce_rows,
                   summarise, validate_set)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=["cache", "d0", "score", "tfit", "tgrid", "tapergrid"])
    ap.add_argument("--run", default="pmG_ref")
    ap.add_argument("--runs-root", default=str(CAL / "runs"))
    ap.add_argument("--ckpt", default="22000")
    ap.add_argument("--sets", default="sel,probe")
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tau-list", default=None)
    ap.add_argument("--knee", type=float, default=None, help="tapered temperature (post hoc addendum): the knee a")
    ap.add_argument("--knees", default="1.0,1.5,2.0", help="tapergrid: the knees")
    ap.add_argument("--max-days", type=int, default=None, help="timing tests: only the first days of every set")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.task == "cache":
        from types import SimpleNamespace
        rargs = SimpleNamespace(**json.loads((Path(a.runs_root) / a.run / "args.json").read_text()))
        store = open_store(rargs.store, **store_kwargs(rargs))
        build_valsets(store)
        print(f"[cache] done in {(time.time() - t0) / 60:.1f} min", flush=True)
        return
    rank, world, dev = dist_setup(a.device)
    rargs, model, step = load_run_x(a.run, dev, a.ckpt, a.runs_root)
    store = open_store(rargs.store, **store_kwargs(rargs))           # GhapStore, firewall on
    pix = PixelArrays(store, dev)
    prior = load_prior_module(dev)
    g = guard()
    M = a.members
    cache = {}
    tag = a.tag or f"{a.task}_{a.run}_{a.ckpt}"
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)
    log(f"[diag] task {a.task} run {a.run} step {step} world {world} M {M} guard {g} -> results/{tag}.json "
        f"(model + store + prior ready after {time.time() - t0:.0f} s)")

    def run_set(name, ode_steps=a.ode_steps, tau=a.tau, knee=a.knee):
        vs = ValSet.load(name)
        if a.max_days:
            vs.days, vs.dates, vs.truth = vs.days[:a.max_days], vs.dates[:a.max_days], vs.truth[:a.max_days]
            if vs.kind == "train":
                vs.pix = vs.pix[:a.max_days]
        t1 = time.time()
        rows = validate_set(model, store, pix, vs, prior, dev, rank=rank, world=world, M=M, ode_steps=ode_steps, tau=tau,
                            seed=0, cache=cache, cond_frozen=True, grid=rargs.grid, transport=rargs.transport, g=g, knee=knee)
        rows = reduce_rows(rows, world).cpu().numpy()
        return vs, rows, time.time() - t1

    out = {"task": a.task, "run": a.run, "step": step, "members": M, "guard": g, "world": world, "knee": a.knee, "results": {}}
    arrs = {}

    def record(key, vs, rows, secs, **extra):
        s = summarise(rows, M)
        s["daily"] = daily_table(rows, M, vs)
        s["seconds"] = secs
        s.update(extra)
        if vs.name in ("sel", "fit", "ins2020", "ins2018") and rows.shape[0] >= 4:
            s["boot95"] = boot_pooled(rows, M)
        out["results"][key] = s
        arrs[key] = rows
        u = s["ug"]
        log(f"[{key}] {s['n_days']} days ({secs:.0f} s): crps {u['crps_fair']:.4f} rmseM {u['rmse_ensmean']:.4f} bias "
            f"{u['bias']:+.3f} ssF {u['spread_skill_fortin']:.4f} D {u['D_predictive_mean_sq_error']:.2f} | log ssF "
            f"{s['log']['spread_skill_fortin']:.4f} crps {s['log']['crps_fair']:.4f} | z crps {s['z']['crps_fair']:.4f} "
            f"ssF {s['z']['spread_skill_fortin']:.4f} bias {s['z']['bias']:+.3f} | pix>TG "
            f"{s['sanity']['n_pix_gt_TG']:.0f} cap {s['sanity']['n_member_at_pmax']:.0f} bound "
            f"{s['sanity']['n_member_at_bound']:.0f} trunc {s['sanity']['n_member_z0_trunc']:.0f} max "
            f"{s['sanity']['max_member_ug']:.0f}")

    if a.task == "d0":
        for name in ("sel", "probe", "ins2020", "ins2018"):
            record(name, *run_set(name, 10, 1.0), ode_steps=10, tau=1.0)
        for k in (20, 40):
            record(f"sel_euler{k}", *run_set("sel", k, 1.0), ode_steps=k, tau=1.0)
    elif a.task == "score":
        for name in a.sets.split(","):
            record(name, *run_set(name), ode_steps=a.ode_steps, tau=a.tau)
    elif a.task == "tfit":
        def f(tau):
            vs, rows, secs = run_set("fit", a.ode_steps, tau)
            s = summarise(rows, M)["ug"]["spread_skill_fortin"]
            log(f"[tfit] tau {tau:.4f}: fit-days pixel ssF {s:.4f} ({secs:.0f} s)")
            return s, vs, rows, secs
        trace = []
        lo, hi = 1.0, (3.0 if a.knee is None else 3.0 / a.knee)
        slo, vs, rows, secs = f(lo)
        trace.append([lo, slo])
        record("fit_tau1", vs, rows, secs, tau=1.0)
        shi = f(hi)[0]
        trace.append([hi, shi])
        tau1, reached = None, shi >= 1.0
        if reached:
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                sm, vs, rows, secs = f(mid)
                trace.append([mid, sm])
                if abs(sm - 1.0) <= 0.005:
                    tau1 = mid
                    break
                if sm < 1.0:
                    lo = mid
                else:
                    hi = mid
            if tau1 is None:
                tau1 = mid
        else:                                                    # 1 not reached: the tau of the maximum on a grid
            grid = [t for t in (1.0 + 0.1 * i for i in range(21)) if a.knee is None or t * a.knee <= 3.0]
            vals = [f(t)[0] for t in grid]
            trace += [[t, v] for t, v in zip(grid, vals)]
            tau1 = grid[int(np.argmax(vals))]
            sm, vs, rows, secs = f(tau1)
        tau1 = round(float(tau1), 4)
        _, vs, rows, secs = f(tau1)
        record("fit_tau1_fitted", vs, rows, secs, tau=tau1)
        out["tau1"], out["reached_1"], out["trace"] = tau1, bool(reached), trace
        for name in ("sel", "probe"):
            record(f"{name}_tau1", *run_set(name, a.ode_steps, tau1), tau=tau1)
        log(f"[tfit] tau1 = {tau1} (spread/skill 1 reached: {reached})")
    elif a.task == "tgrid":
        taus = [float(x) for x in a.tau_list.split(",")] if a.tau_list else [round(1.05 + 0.05 * i, 2) for i in range(20)]
        out["taus"] = taus
        for tau in taus:
            for name in ("sel", "probe"):
                record(f"{name}_tau{tau:.2f}", *run_set(name, a.ode_steps, tau), tau=tau)
    elif a.task == "tapergrid":
        out["configs"] = []
        for knee in [float(x) for x in a.knees.split(",")]:
            tau = 1.10
            while tau <= min(2.50, 3.0 / knee) + 1e-9:
                for name in ("sel", "probe"):
                    record(f"{name}_k{knee:.2f}_tau{tau:.2f}", *run_set(name, a.ode_steps, tau, knee), tau=tau, knee=knee)
                out["configs"].append([knee, round(tau, 2)])
                tau = round(tau + 0.05, 2)
    if rank == 0:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out["minutes"] = (time.time() - t0) / 60
        (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=1))
        np.savez(RESULTS / f"{tag}_rows.npz", **arrs)
        log(f"[done] {out['minutes']:.1f} min -> {RESULTS / (tag + '.json')}")
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
