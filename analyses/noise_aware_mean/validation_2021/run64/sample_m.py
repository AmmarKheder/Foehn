#!/usr/bin/env python3
"""A14 run R2: copy of a13/warn16/sample16.py; the ONLY changes are --year (default 2022) and --noise (a noise file of G
with at least --members global fields, default: the file eval_global_green.Windows chooses). Original docstring:
A13 section 4 (warnings with 16 members, no training): the frozen final 1 km model sampled with M members at the 8.1 M
grid cells with at least 100 residents (pop_pixels.npz), for chosen start days and leads.

The sampling is the code of the paper's evaluation at the population pixels, imported UNCHANGED from
eccv/cran-pm/experiments/global1km/eval_global_green.py (TestGreen, Windows, windows, ramp, sample, member_particles,
full_map; read only): the same dataset indices, windows and cosine-ramp blend, the same global noise fields cut into every
window (noise_tiles16.npy for more than four members; its members 0-3 must equal noise_tiles.npy, checked by
check_inputs.py), the same particles of each member (member_particles(i, h, m)), the same sampler (Euler on the training
tau quantile grid, --steps 4, bf16 autocast, one member per forward pass, batch 1), the same float32 accumulation. Hence
members 0-3 of a 16-member run are, operation by operation, the members of the stored 4-member run (eval_pop_crps_e01).

What differs from eval_global_green.run (nothing that touches the numbers):
  * only the populated pixels are accumulated (float32, the same additions in the same window order, then the same
    multiplication by the inverse blend weight), instead of the full 14,400 x 36,000 map: bit-identical values there;
  * nothing else is computed or written (no regions, maps, seams, 40 km means; and NO station file is opened: an audit
    hook refuses any path containing 'station');
  * output: per rank one preallocated .npy (forecasts of the rank, pixels, members) float16 written row by row, a keys
    file and a 'done' file (restartable: rows already done are skipped).

    srun ... python sample16.py --ckpt C --members 16 --doys 0 12 ... --leads 1 3 --out DIR
    srun ... python sample16.py --ckpt C --members 16 --pairs 2:1 2:3 2:5 ... --out DIR --tag k00     (array task)
"""
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


class WindowsN(egg.Windows):
    """eval_global_green.Windows reading the global noise fields from the file `noise` of G (same cutting, same members)"""

    def __init__(self, ds, jobs, win, members, noise, lags=()):
        self.ds, self.jobs, self.win, self.members, self.lags = ds, jobs, win, members, sorted({l for l in lags if l > 0})
        self.noise = egg.Npy(G / noise)
        assert members <= self.noise.shape[0], f"{noise} has {self.noise.shape[0]} members"


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
    P = int(egg.Npy(G / store / "hres" / f"origins_{a.year}.npy").shape[-2])
    ds = egg.TestGreen(a.max_lead, P, store, emis, a.year, hp.get("emis_store"))
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
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pop_idx = np.load(G / "pop_pixels.npz")["idx"]
    keys = np.array([(int(ds.doy[i]), h, int(ds.doy[i + h])) for i, h in todo], np.int32).reshape(-1, 3)
    fn, fk, fd = out / f"pop{a.members}_{tag}.npy", out / f"keys{a.members}_{tag}.npy", out / f"done{a.members}_{tag}.json"
    if fn.exists() and fk.exists() and np.array_equal(np.load(fk), keys):
        arr = np.lib.format.open_memmap(fn, mode="r+")
        done = set(json.loads(fd.read_text())) if fd.exists() else set()
    else:
        arr = np.lib.format.open_memmap(fn, mode="w+", dtype=np.float16, shape=(len(todo), len(pop_idx), a.members))
        np.save(fk, keys); done = set()
        fd.write_text(json.dumps([]))
    if rank == 0 and a.tag is None:
        np.save(out / "pop_idx.npy", pop_idx)
    jobs = [(i, h, w) for n, (i, h) in enumerate(todo) if n not in done for w in range(len(win))]
    loader = DataLoader(egg.Windows(ds, jobs, win, a.members) if a.noise is None else WindowsN(ds, jobs, win, a.members, a.noise), batch_size=None, num_workers=a.workers,
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
            t_gpu += time.time() - t
        acc *= inv_w[None]
        arr[n] = acc.T.astype(np.float16)
        arr.flush()
        done.add(n)
        tmp = fd.with_suffix(f".tmp{os.getpid()}"); tmp.write_text(json.dumps(sorted(done))); os.replace(tmp, fd)
        print(f"rank {tag}: start {int(ds.doy[i])} lead {h}: io wait {t_io:.0f}s gpu {t_gpu:.0f}s | {len(done)}/{len(todo)} "
              f"{time.time() - t_start:.0f}s", flush=True)
    print("rank", rank, "done", len(done), "forecasts", flush=True)


if __name__ == "__main__":
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
    ap.add_argument("--year", type=int, default=2022, help="2021: the validation year (eval_global_green --year)")
    ap.add_argument("--noise", default=None, help="noise file of G, e.g. noise_tiles64.npy (members 0-15 = noise_tiles16.npy)")
    ap.add_argument("--tag", default=None, help="file suffix instead of the rank (one-task array jobs writing into one folder)")
    args = ap.parse_args()
    assert (args.doys is None) != (args.pairs is None), "give --doys (with --leads) or --pairs"
    run(args)
