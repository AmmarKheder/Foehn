#!/usr/bin/env python3
"""A8(e)-2(i): the in-training validation of a FROZEN checkpoint of pmG_base recomputed with today's code.

train_aq._run_val calls validate(EMA model, store, pix, ValData(store, val_pixels, val_days), args, spec, device) and
writes the result to metrics.jsonl. This script makes the same call for one or more saved steps of a run (read-only:
args.json, ckpt/stepNNNNNN.ckpt, metrics.jsonl) and compares with the stored record. One GPU, about 20 s per step
after the store is open. Nothing is written in the run directory; the result goes to --out.

  python valrepro.py --run pmG_base --runs-root .../runs_ghap --steps 22000 --out .../a8/results/valrepro_seed0.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

W = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(W))

import torch  # noqa: E402

from cdffm_aq.data_aq import PixelArrays, open_store, store_kwargs  # noqa: E402
from cdffm_aq.eval_aq import load_run  # noqa: E402
from cdffm_aq.train_aq import ValData, noise_spec, validate  # noqa: E402

KEYS = ("crps_fair", "rmse_ensmean", "rmse_1member", "mae_ensmean", "bias", "spread_skill_fortin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pmG_base")
    ap.add_argument("--runs-root", default=str(W / "runs_ghap"))
    ap.add_argument("--steps", default="22000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-seeds", default=None, help="comma-separated: ALSO recompute the validation with these noise "
                    "seeds (train_aq.validate seeds its noise with args.seed; the checkpoint is unchanged): sampling "
                    "noise of the selection metric")
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device(f"cuda:{int(os.environ.get('SLURM_LOCALID', 0))}")
    torch.cuda.set_device(dev)
    run_dir = Path(a.runs_root) / a.run
    stored = {}
    for line in open(run_dir / "metrics.jsonl"):
        if line.strip():
            r = json.loads(line)
            stored[int(r["step"])] = r
    out = {"run": a.run, "run_dir": str(run_dir), "steps": {}}
    store = pix = val = None
    for s in [int(x) for x in a.steps.split(",")]:
        args, model, step = load_run(a.run, dev, str(s), a.runs_root)
        assert step == s, (step, s)
        if store is None:
            store = open_store(args.store, **store_kwargs(args))
            pix = PixelArrays(store, dev)
            val = ValData(store, args.val_pixels, [int(x) for x in args.val_days.split(",")])
            print(f"[valrepro] store open, {len(val.days)} days x {len(val.pix)} pixels ({time.time() - t0:.0f} s)", flush=True)
        t1 = time.time()
        res, base, _ = validate(model, store, pix, val, args, noise_spec(args), dev, baselines=True)
        row = {"seconds": time.time() - t1, "recomputed": res, "stored": stored[s]["model"], "baselines_recomputed": base,
               "diff": {sp: {k: res[sp][k] - stored[s]["model"][sp][k] for k in KEYS} for sp in ("ug", "log")}}
        row["max_abs_rank_hist_diff"] = max(abs(x - y) for x, y in zip(res["ug"]["rank_hist"], stored[s]["model"]["ug"]["rank_hist"]))
        if a.val_seeds:
            seed0 = args.seed
            row["val_noise_seeds"] = {}
            for vs in [int(x) for x in a.val_seeds.split(",")]:
                args.seed = vs
                r2, _, _ = validate(model, store, pix, val, args, noise_spec(args), dev, baselines=False)
                row["val_noise_seeds"][str(vs)] = {k: r2["ug"][k] for k in KEYS}
                print(f"[valrepro] step {s}, validation noise seed {vs}: crps {r2['ug']['crps_fair']:.6f} rmseM "
                      f"{r2['ug']['rmse_ensmean']:.6f} bias {r2['ug']['bias']:+.6f} ssF {r2['ug']['spread_skill_fortin']:.6f}", flush=True)
            args.seed = seed0
        out["steps"][str(s)] = row
        u, v = res["ug"], stored[s]["model"]["ug"]
        print(f"[valrepro] step {s}: recomputed crps {u['crps_fair']:.6f} rmseM {u['rmse_ensmean']:.6f} bias {u['bias']:+.6f} "
              f"ssF {u['spread_skill_fortin']:.6f} | stored crps {v['crps_fair']:.6f} rmseM {v['rmse_ensmean']:.6f} bias "
              f"{v['bias']:+.6f} ssF {v['spread_skill_fortin']:.6f} | diff crps {u['crps_fair'] - v['crps_fair']:+.2e} "
              f"rmseM {u['rmse_ensmean'] - v['rmse_ensmean']:+.2e} ({row['seconds']:.0f} s)", flush=True)
    b0 = next((r["baselines"] for r in stored.values() if "baselines" in r), None)
    if b0 is not None:
        out["baselines_stored"] = b0
        out["baselines_diff_crps"] = {n: base[n]["ug"]["crps_fair"] - b0[n]["ug"]["crps_fair"] for n in base}
        print(f"[valrepro] baselines: diff of fair CRPS {out['baselines_diff_crps']}", flush=True)
    out["total_seconds"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[valrepro] done in {out['total_seconds']:.0f} s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
