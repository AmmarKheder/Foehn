#!/usr/bin/env python3
"""A16 run G11 (and smoke step b16): a13/warn16/sample16.py, the 16-member sampler of the frozen final 1 km model at the
8.1 M populated pixels, with ONLY the initial noise replaced by the evolving noise of A15 (arnoise22.py). Same output
format as a14/data/run16_13: per array task pop16_<tag>.npy (forecasts, pixels, 16) float16, keys16_<tag>.npy and
done16_<tag>.json, all tasks in ONE folder (a16/data/run16), plus the folder's A16 manifest a16_run.json.

The audit hook of sample16.py is kept verbatim (no station, monitor or OpenAQ file can be opened by this process).
run() is a verbatim copy of sample16.run; every changed or added line ends with '# A16-S<n>' and every original line that
does not survive verbatim is listed in REMOVED_LINES (check_cpu.py checks both):
  S1  start16(): the forecast list must be the plan's (G11: line k+1 of jobs/pairs_g11.txt for tag k<kk>), the folder is
      claimed (a16_run.json: plan, arguments, rho, base, year, code hashes; foreign files or another manifest refused)
  S2  WindowsAR22 replaces eval_global_green.Windows: x0[m] = tile_window(ARNoise(2022, ds.doy[i], h, rho, base, 16), m, ...)
  S3  the keys and done files are written atomically (fsync, read-back, rename)
  S4  the progress line also gives the GPU seconds of the first window and of the median window (warm cost, smoke)
gate(): rho* with base philox; check_cpu.json and (except in the smoke) smoke_check.json all_passed.
    srun ... python sample16_ar.py --plan G11 --ckpt <run>/final.ckpt --members 16 --pairs 0:1 0:3 ... --out a16/data/run16
             --tag k00 --rho 0.59 --base philox"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _audit(event, args):                       # stations and monitors are evaluation data only: never opened here
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]).lower()
        if p.endswith((".py", ".pyc", ".so")) or "/site-packages/" in p:     # code, not data
            return
        if "station" in p or "openaq" in p or "monitor" in p:
            raise PermissionError(f"A13 warn16: station/monitor file refused: {p}")


sys.addaudithook(_audit)

GLOBAL1KM = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km"))
sys.path.insert(0, str(GLOBAL1KM))
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import eval_global_green as egg  # noqa: E402  (the paper's evaluation, imported unchanged)
import arnoise22 as an  # noqa: E402  (A16: numpy 1.26.3, the evolving noise, plans and gates)

G, FH, FW, GH, GW = egg.G, egg.FH, egg.FW, egg.GH, egg.GW


def pop_windows(pop_idx, win):
    """for every window: positions (in pop_idx) of the populated pixels inside it, and their flat offset in the window"""
    pr, pc = (pop_idx // GW).astype(np.int64), (pop_idx % GW).astype(np.int64)
    out = []
    for k0, j0 in win:
        r0, c0 = 25 * k0, 25 * j0
        sel = np.flatnonzero((pr >= r0) & (pr < r0 + FH) & (pc >= c0) & (pc < c0 + FW))
        out.append((sel, (pr[sel] - r0) * FW + (pc[sel] - c0)))
    return out


def plan_match16(a):
    """the arguments must be those of the plan; G11: the pairs of line k+1 of jobs/pairs_g11.txt for tag k<kk>"""
    plan = an.SAMPLE_PLANS[a.plan]
    got = dict(ckpt=an.resolved(a.ckpt), members=a.members, steps=a.steps, psub=a.psub, max_lead=a.max_lead, doys=a.doys,
               out=an.resolved(a.out))
    want = {k: (an.resolved(plan[k]) if k in ("ckpt", "out") else plan[k]) for k in got}
    assert got == want, f"A16 plan {a.plan}: arguments differ from the plan: {got} != {want}"
    if a.plan == "G11":
        lines = an.pairs_g11()
        allp = [p for ln in lines for p in ln]
        expect = [f"{d}:{h}" for d in range(0, 357, 2) for h in (1, 3)]
        assert allp == expect and len(lines) == 45 and all(len(ln) == 8 for ln in lines[:-1]), "jobs/pairs_g11.txt"
        assert a.tag is not None and len(a.tag) == 3 and a.tag[0] == "k" and a.tag[1:].isdigit(), a.tag
        assert a.pairs == lines[int(a.tag[1:])], f"A16 G11: task {a.tag} must sample line {int(a.tag[1:]) + 1} of pairs_g11.txt"
    else:
        assert a.pairs == plan["pairs"] and a.tag == plan["tag"], (a.pairs, a.tag)
    return plan


def start16(a, todo_all, world):
    """A16 gates of a task: its forecasts, the claim of the (shared) output folder"""
    plan = an.SAMPLE_PLANS[a.plan]
    assert world == 1 and len(todo_all) == len(a.pairs), (world, len(todo_all))
    args = dict(ckpt=an.resolved(a.ckpt), members=a.members, steps=a.steps, psub=a.psub, max_lead=a.max_lead,
                out=an.resolved(a.out), forecasts=plan["n"],
                pairs_sha256=an.sha_file(plan["pairs_file"]) if "pairs_file" in plan else plan["pairs"])
    meta = an.manifest(a.plan, args, a.rho, a.base, 2022, "one GCD per task", [Path(__file__).resolve()])
    return an.claim(a.out, meta, fresh=a.plan in an.SMOKE)


def run(a):
    rank, world, local = (int(os.environ.get(k, 0)) for k in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"))
    world = max(world, 1)
    tag = str(rank) if a.tag is None else a.tag      # --tag: one array task per start day, files pop<M>_<tag>.npy in one folder
    if a.tag is not None:
        assert world == 1, "--tag is for one-task jobs (array tasks)"
    ck = Path(a.ckpt)
    state = torch.load(ck, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    epoch = state.get("epoch")
    assert any(k.startswith("net.enc.") for k in state["state_dict"]) and not any(k.startswith("net.fp.") for k in state["state_dict"])
    assert hp["max_lead"] == a.max_lead, hp
    store, emis = hp.get("store", "green"), bool(hp.get("emis", False))
    P = int(egg.Npy(G / store / "hres" / "origins_2022.npy").shape[-2])
    ds = egg.TestGreen(a.max_lead, P, store, emis, 2022, hp.get("emis_store"))
    win = egg.windows(np.load(G / "meta.npz")["land"])
    have = set(ds.samples)
    if a.pairs:                                      # --pairs d:h ...: exactly these (start day of year, lead), in this order
        idx_of = {int(ds.doy[i]): i for i in sorted({i for i, h in ds.samples})}
        assert len(idx_of) == len({i for i, h in ds.samples}), "two start indices share a day of year"
        want = [tuple(int(v) for v in p_.split(":")) for p_ in a.pairs]
        assert len(set(want)) == len(want), "a pair is repeated"
        assert all(d in idx_of for d, h in want), "a start day is not in the test set"
        todo_all = [(idx_of[d], h) for d, h in want]
        assert all(x in have for x in todo_all), "a (start day, lead) is not in the test set"
    else:
        starts = sorted({i for i, h in ds.samples})
        starts = [i for i in starts if int(ds.doy[i]) in set(a.doys)]
        assert len(starts) == len(set(a.doys)), "a start day is not in the test set"
        todo_all = [(i, h) for i in starts for h in a.leads if (i, h) in have]
        assert len(todo_all) == len(starts) * len(a.leads)
    todo = todo_all[rank::world]
    out = start16(a, todo_all, world)  # A16-S1
    pop_idx = np.load(G / "pop_pixels.npz")["idx"]
    keys = np.array([(int(ds.doy[i]), h, int(ds.doy[i + h])) for i, h in todo], np.int32).reshape(-1, 3)
    fn, fk, fd = out / f"pop{a.members}_{tag}.npy", out / f"keys{a.members}_{tag}.npy", out / f"done{a.members}_{tag}.json"
    if fn.exists() and fk.exists() and np.array_equal(np.load(fk), keys):
        arr = np.lib.format.open_memmap(fn, mode="r+")
        done = set(json.loads(fd.read_text())) if fd.exists() else set()
    else:
        arr = np.lib.format.open_memmap(fn, mode="w+", dtype=np.float16, shape=(len(todo), len(pop_idx), a.members))
        an.asave(fk, keys); done = set()  # A16-S3
        an.write_bytes(fd, json.dumps([]).encode())  # A16-S3
    if rank == 0 and a.tag is None:
        np.save(out / "pop_idx.npy", pop_idx)
    jobs = [(i, h, w) for n, (i, h) in enumerate(todo) if n not in done for w in range(len(win))]
    loader = DataLoader(an.WindowsAR22(ds, jobs, win, a.members, [], 2022, a.rho, a.base), batch_size=None, num_workers=a.workers,  # A16-S2
                        prefetch_factor=2 if a.workers else None)
    batches = iter(loader)
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    model = egg.FullModuleGreen.load_from_checkpoint(ck, map_location="cpu").to(dev).eval()
    del state
    if rank == 0:
        print(f"{ck} (epoch {epoch}) | store {store}: {P} particles, {a.psub} per member | {len(win)} windows, {a.members} members, "
              f"{a.steps} steps | {len(todo_all)} forecasts on {world} GCDs", flush=True)
    # blend weights at the populated pixels (as eval_global_green.run: sum over ALL windows, inverse where covered)
    wgt = np.outer(egg.ramp(FH, egg.RAMP), egg.ramp(FW, egg.RAMP))
    wgt_d = torch.from_numpy(wgt).to(dev)
    wsum = np.zeros((GH, GW), np.float32)
    for k0, j0 in win:
        wsum[25 * k0:25 * k0 + FH, 25 * j0:25 * j0 + FW] += wgt
    covered = wsum > 0
    inv_w = np.where(covered, 1.0 / np.maximum(wsum, 1e-12), 0.0).astype(np.float32).ravel()[pop_idx]
    del wsum, covered
    pw = pop_windows(pop_idx, win)
    pw_d = [(sel, torch.from_numpy(loc).to(dev)) for sel, loc in pw]
    acc = np.zeros((a.members, len(pop_idx)), np.float32)
    t_start = time.time()
    for n, (i, h) in enumerate(todo):
        if n in done:
            continue
        acc.fill(0.0)
        t_io = t_gpu = 0.0
        t_w = []  # A16-S4
        for w, (k0, j0) in enumerate(win):
            t = time.time(); b = next(batches); t_io += time.time() - t
            t = time.time()
            bt = {k: (v[None] if v.dim() > 0 else v.reshape(1)).to(dev, non_blocking=True) for k, v in b.items() if k != "x0"}
            sel, loc = pw_d[w]
            for m in range(a.members):
                bm = dict(bt)
                org_m = bm["org"]
                bm["org"] = org_m[..., torch.from_numpy(egg.member_particles(i, h, m, org_m.shape[-2], a.psub)).to(dev), :]
                x0 = b["x0"][m].to(dev).float()[None, None]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = egg.sample(model, bm, x0, a.steps, 1, "euler")
                v = (wgt_d * (x[0, 0, :FH, :FW].float() * egg.SD + egg.MU)).reshape(-1)[loc]      # the same float32 product, gathered
                acc[m, sel] += v.cpu().numpy()
            dt_ = time.time() - t; t_gpu += dt_; t_w.append(dt_)  # A16-S4
        acc *= inv_w[None]
        arr[n] = acc.T.astype(np.float16)
        arr.flush()
        done.add(n)
        an.write_bytes(fd, json.dumps(sorted(done)).encode())  # A16-S3
        print(f"rank {tag}: start {int(ds.doy[i])} lead {h}: io wait {t_io:.0f}s gpu {t_gpu:.0f}s | {len(done)}/{len(todo)} "
              f"{time.time() - t_start:.0f}s | gpu first window {t_w[0]:.1f}s median window {float(np.median(t_w)):.1f}s", flush=True)  # A16-S4
    print("rank", rank, "done", len(done), "forecasts", flush=True)


REMOVED_LINES = [
    'out = Path(a.out); out.mkdir(parents=True, exist_ok=True)',
    'np.save(fk, keys); done = set()',
    'fd.write_text(json.dumps([]))',
    'loader = DataLoader(egg.Windows(ds, jobs, win, a.members), batch_size=None, num_workers=a.workers,',
    't_gpu += time.time() - t',
    'tmp = fd.with_suffix(f".tmp{os.getpid()}"); tmp.write_text(json.dumps(sorted(done))); os.replace(tmp, fd)',
    'f"{time.time() - t_start:.0f}s", flush=True)',
]
HUNKS = (1, 2, 3, 4)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--psub", type=int, default=16)
    ap.add_argument("--doys", type=int, nargs="+", default=None)
    ap.add_argument("--pairs", nargs="+", default=None, help="start day of year:lead pairs (instead of --doys x --leads)")
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--max-lead", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None, help="file suffix instead of the rank (one-task array jobs writing into one folder)")
    ap.add_argument("--plan", required=True, choices=sorted(an.SAMPLE_PLANS))
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--base", required=True, choices=["philox", "stored"])
    args = ap.parse_args(argv)
    assert (args.doys is None) != (args.pairs is None), "give --doys (with --leads) or --pairs"
    plan_match16(args)
    return args


if __name__ == "__main__":
    a_ = parse_args()
    an.gate(a_.plan, a_.rho, a_.base)
    run(a_)
