#!/usr/bin/env python3
"""WP1a step A: the 0.4 degree real-time forecast of past issue days, recomputed with the production chain.

For every day given (yyyy-mm-dd ...) a directory <root>/<d>/ is made with symlinks to the production inputs
RT/data/<d>/{cams,hres_B,fires}.npy and a copy of meta.json (the production directories are only read). Then, per day:
  A1  FLEXGREEN: the ORIGINAL global1km/flexgreen.py as a subprocess with the options of realtime/flexgreen_batch.sh and
      rt_job.sh (--src hres --rt-dir <root>/<d> --particles 32 --emis --emis-fallback --instant --all-cells); it writes
      <root>/<d>/green/ (skipped when green/done_<year>/<doy> exists, or with --link-green: green -> the archived store
      RT/data/<d>/green, for the check of the sampler alone);
  A2  the forecast: the loop of the ORIGINAL realtime/predict.py main() (same RtCamsData class, imported; same checkpoint
      realtime/model.json -> runs/rt12_a/final.ckpt; 8 members; Euler on flow_grid(model, 8, 3) = 12 steps; bf16
      autocast; noise seed gi * 1000 + k; particle subsets seeded by (gi, k, member)) -> <root>/<d>/pred.npy
      (8, 10, 361, 900) float16 ug m-3 and pred_meta.json. Nothing is written into realtime/data or realtime/out
      (the original main() calls C.day_dir / C.out_dir / C.log, which write there: they are not used here).
One process per GCD (srun -n 8): rank r takes the days r, r + 8, ...
    srun -n 8 python -u coarse_replay.py --root <replay>/coarse 2026-09-01 2026-09-02 ...
    python -u coarse_replay.py --root <replay>/check/coarse_archgreen --link-green 2026-11-05"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
G1 = RT.parent
INPUTS = ("cams.npy", "hres_B.npy", "fires.npy")


def make_dir(root, d, link_green=False):
    src, dst = RT / "data" / d, Path(root) / d
    for f in INPUTS + ("meta.json",):
        assert (src / f).exists(), f"{src / f} missing"
    dst.mkdir(parents=True, exist_ok=True)
    for f in INPUTS:
        if not (dst / f).exists():
            (dst / f).symlink_to(src / f)
    if not (dst / "meta.json").exists():
        shutil.copyfile(src / "meta.json", dst / "meta.json")
    if link_green and not (dst / "green").exists():
        assert (src / "green").is_dir(), f"{src / 'green'}: no archived FLEXGREEN store"
        (dst / "green").symlink_to(src / "green")
    return dst


def flexgreen(dst, log):
    m = json.loads((dst / "meta.json").read_text())
    done = dst / "green" / f"done_{m['year']}" / f"{int(m['doy']):03d}"
    if done.exists():
        return 0.0
    t = time.time()
    env = dict(os.environ, SLURM_PROCID="0", SLURM_NTASKS="1")          # a one-rank run (flexgreen_batch.sh); SLURM_LOCALID keeps the GPU
    cmd = [sys.executable, "-u", "flexgreen.py", "run", "--src", "hres", "--rt-dir", str(dst), "--particles", "32",
           "--emis", "--emis-fallback", "--instant", "--all-cells"]
    r = subprocess.run(cmd, cwd=str(G1), env=env, capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(f"== FLEXGREEN {dst.name} rc {r.returncode} in {time.time() - t:.0f}s\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}\n")
    if r.returncode != 0 or not done.exists():
        raise RuntimeError(f"FLEXGREEN {dst.name} failed (rc {r.returncode}): {r.stderr[-400:]}")
    return time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="+")
    ap.add_argument("--root", required=True)
    ap.add_argument("--link-green", action="store_true", help="use the archived RT/data/<d>/green (sampler check)")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--refine-tail", type=int, default=3)
    ap.add_argument("--no-flexgreen", action="store_true", help="prepare the directories only")
    a = ap.parse_args()
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    days = sorted(a.days)[rank::world]
    log = root.parent / "logs" / f"coarse_{os.environ.get('SLURM_JOB_ID', 'local')}_r{rank}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[coarse r{rank}/{world}] {len(days)} days: {' '.join(days)}", flush=True)
    dirs = {d: make_dir(root, d, a.link_green) for d in days}
    if a.no_flexgreen:
        return
    # A1 first for all the rank's days (subprocesses), then the model is loaded once for A2
    tg = {}
    for d in days:
        if (dirs[d] / "pred.npy").exists():
            continue
        tg[d] = flexgreen(dirs[d], log)
        print(f"[coarse r{rank}] {d}: FLEXGREEN {tg[d]:.0f}s", flush=True)
    todo = [d for d in days if not (dirs[d] / "pred.npy").exists()]
    if not todo:
        return
    import torch
    sys.path.insert(0, str(RT))
    import common as C                                       # noqa: E402  the chain's sys.path, model.json -> C.CKPT, C.VERSION
    import predict as P                                      # noqa: E402  the ORIGINAL module (main() guarded): RtCamsData
    from foehn_cams import NL, NX, NY, PX, PY, CamsModule, collate, flow_grid, rewrap  # noqa: E402
    from dejavu import MU, SD                                # noqa: E402
    dev = torch.device("cuda", int(os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(dev)
    ckpt = Path(C.CKPT)
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    model = CamsModule.load_from_checkpoint(str(ckpt), map_location=dev).eval()
    grid = flow_grid(model, a.steps, a.refine_tail)
    for d in todo:
        D = dirs[d]
        ds = P.RtCamsData(D, psub=model.hparams.psub, emis=model.hparams.emis, no_fires=False)
        gi = ds.gi
        t0 = time.time()
        pred = np.zeros((a.members, NL, NY, NX), np.float16)
        with torch.no_grad():
            for k in range(1, NL + 1):
                b = collate([ds.item(gi, k, member=m) for m in range(a.members)])
                b = {kk: v.to(dev, non_blocking=True) for kk, v in b.items()}
                gen = torch.Generator(device=dev).manual_seed(gi * 1000 + k)                 # the seeds of foehn_cams.predict
                x = rewrap(torch.randn(a.members, 1, PY, PX, device=dev, generator=gen))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for s in range(len(grid) - 1):
                        tau = torch.full((a.members,), float(grid[s]), device=dev)
                        x = rewrap(x + model.velocity(x, b, tau).float() * float(grid[s + 1] - grid[s]))
                pred[:, k - 1] = (x[:, 0, :NY, :NX] * SD + MU).cpu().numpy().astype(np.float16)
        tmp = D / "pred.tmp.npy"
        np.save(tmp, pred)
        os.replace(tmp, D / "pred.npy")
        meta = dict(date=d, gi=gi, members=a.members, steps=a.steps, refine_tail=a.refine_tail, euler_steps=len(grid) - 1,
                    ckpt=str(ckpt), ckpt_sha256=sha, model_version=C.VERSION, emis=bool(model.hparams.emis),
                    psub=int(model.hparams.psub), leads_h=[12 * k for k in range(1, NL + 1)],
                    inputs=json.loads((D / "meta.json").read_text()), green="archived (symlink)" if a.link_green else "recomputed",
                    flexgreen_s=round(tg.get(d, 0.0)), predict_s=round(time.time() - t0),
                    lead_means_ug=[round(float(pred[:, k].astype(np.float32).mean()), 3) for k in range(NL)],
                    device=torch.cuda.get_device_name(dev), torch=torch.__version__,
                    written_utc=dt.datetime.utcnow().isoformat(timespec="seconds"), job=os.environ.get("SLURM_JOB_ID"))
        (D / "pred_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[coarse r{rank}] {d}: pred.npy written (gi {gi}, predict {time.time() - t0:.0f}s, mean 24 h "
              f"{meta['lead_means_ug'][1]:.2f}, 120 h {meta['lead_means_ug'][9]:.2f} ug m-3)", flush=True)


if __name__ == "__main__":
    main()
