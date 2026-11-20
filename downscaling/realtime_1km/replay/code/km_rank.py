#!/usr/bin/env python3
"""WP1a steps B1(pm) + B2 + C for the days of one rank (one GCD): for each issue day of the list taken by this rank
(days[rank::world]) run, as separate processes and only when their outputs are missing,
  prep_replay.py <d> --only pm   (member daily means at 0.25 deg from the coarse forecast; numpy only)
  channels_replay.py <d>         (mixing height, FLEXGREEN of the 5 start days on this GCD, chan_L<L>.npz)
  infer_points.py <d>            (the downscaled members at the positions) -> km/<d>.npz, raw/<d>.npy
A failed day is reported and the next one is tried; the job can be resubmitted (finished days are skipped).
The rank's GCD is selected by HIP_VISIBLE_DEVICES in the job script, so every process here sees one device (index 0).
    srun -n 8 python -u km_rank.py --ckpt <best.ckpt> --list days_all.txt"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

R = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")
HERE = Path(__file__).resolve().parent


def run(cmd, env, tag):
    t = time.time()
    r = subprocess.run(cmd, env=env, cwd=str(HERE), capture_output=True, text=True)
    tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-(4 if r.returncode == 0 else 25):])
    print(f"--- {tag}: rc {r.returncode} in {time.time() - t:.0f}s\n{tail}", flush=True)
    return r.returncode, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--keep-fg", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    days = sorted(x.strip() for x in Path(a.list).read_text().split() if x.strip())[rank::world]
    env = dict(os.environ, SLURM_PROCID="0", SLURM_NTASKS="1", SLURM_LOCALID="0")      # children: one rank, device 0 of the visible set
    print(f"[km r{rank}/{world}] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} days: {' '.join(days)}", flush=True)
    t0 = time.time()
    summary = {}
    for d in days:
        work = R / "work" / d
        if (R / "km" / f"{d}.npz").exists() and not a.overwrite:
            summary[d] = "done before"
            continue
        if not (work / "met_ifs.npy").exists():
            summary[d] = "FAILED: met_ifs.npy missing (prep_replay.py --only met)"
            print(f"[km r{rank}] {d}: {summary[d]}", flush=True)
            continue
        tm = {}
        if not (work / "pm_rt025.npy").exists() or a.overwrite:
            rc, tm["pm_s"] = run([sys.executable, "-u", "prep_replay.py", d, "--only", "pm"], env, f"{d} prep pm")
            if rc:
                summary[d] = "FAILED: prep pm (no coarse forecast?)"
                continue
        noprev = bool(json.loads((work / "prep.json").read_text()).get("no_prev"))
        leads = [2, 3, 4, 5] if noprev else [1, 2, 3, 4, 5]
        if a.overwrite or not all((work / f"chan_L{L}.npz").exists() for L in leads):
            rc, tm["channels_s"] = run([sys.executable, "-u", "channels_replay.py", d] + (["--keep-fg"] if a.keep_fg else []), env, f"{d} channels")
            if rc:
                summary[d] = "FAILED: channels"
                continue
        rc, tm["points_s"] = run([sys.executable, "-u", "infer_points.py", "--ckpt", a.ckpt, d] + (["--overwrite"] if a.overwrite else []), env, f"{d} points")
        summary[d] = "FAILED: points" if rc or not (R / "km" / f"{d}.npz").exists() else "ok " + json.dumps({k: round(v) for k, v in tm.items()})
        print(f"[km r{rank}] {d}: {summary[d]} (elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[km r{rank}] SUMMARY {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
