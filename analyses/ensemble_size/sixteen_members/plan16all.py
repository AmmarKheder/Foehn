#!/usr/bin/env python3
"""A13 section 4, follow-up of 2026-11-20 after the rule was met on the 30 start days: the 16-member
run on ALL 179 start days of analyses/hidden/v3. Builds the list of forecasts and checks that every one is a test sample.
  * leads 1 and 3 on the 149 start days not done in data/run16 (pre-registered: A13 section 4 'then run on all 179 start days');
  * leads 5 and 7 on all 179 start days (ADDITION for the consistency of the paper's lead ladder, forecasts started 1, 3, 5 and
    7 days before a target day, so that it uses one warning rule; not part of the rule of A13 section 4).
656 forecasts, sorted by (start day, lead), cut into tasks of 8 -> jobs/run16all_pairs.txt (line k+1 = task k, 'd:h' items)
and results/plan16all.json. CPU only; no station file (the audit hook of sample16.py is installed by the import).
    python plan16all.py"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sample16 as s16  # noqa: E402  (audit hook; eval_global_green imported unchanged)
import torch  # noqa: E402

egg, G = s16.egg, s16.G
RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
           "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
EVP = RUN / "eval_pop_crps_e01"
DONE30 = list(range(0, 349, 12))
PER_TASK = 8


def main():
    K = {}
    for p in glob.glob(str(EVP / "part_*.npz")):
        for d, h, t in np.load(p)["keys"]:
            K[(int(d), int(h))] = int(t)
    S179 = sorted(d for d, h in K if h == 1)
    assert S179 == sorted(d for d, h in K if h == 3) and len(S179) == 179 and S179 == list(range(0, 357, 2)), S179[:5]
    assert S179[::6] == DONE30
    # the forecasts of data/run16 (30 start days, leads 1 and 3)
    run16 = set()
    for fk in sorted((HERE / "data" / "run16").glob("keys16_*.npy")):
        done = set(json.loads((fk.parent / f"done16_{fk.stem.split('_')[1]}.json").read_text()))
        run16 |= {(int(d), int(h)) for n, (d, h, t) in enumerate(np.load(fk)) if n in done}
    assert run16 == {(d, h) for d in DONE30 for h in (1, 3)}, sorted(run16)[:4]
    pairs = sorted([(d, h) for d in S179 if d not in DONE30 for h in (1, 3)] + [(d, h) for d in S179 for h in (5, 7)])
    assert len(pairs) == 656 and len(set(pairs)) == 656
    # every pair is a test sample of the frozen model's dataset (as sample16.py builds it)
    state = torch.load(RUN / "final.ckpt", map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]; del state
    store, emis = hp.get("store", "green"), bool(hp.get("emis", False))
    P = int(egg.Npy(G / store / "hres" / "origins_2022.npy").shape[-2])
    ds = egg.TestGreen(hp["max_lead"], P, store, emis, 2022, hp.get("emis_store"))
    have = {(int(ds.doy[i]), int(h)) for i, h in ds.samples}
    missing = [p for p in pairs if p not in have]
    assert not missing, f"not test samples: {missing}"
    idx_of = {int(ds.doy[i]): i for i, h in ds.samples}
    targets = {f"{d}:{h}": int(ds.doy[idx_of[d] + h]) for d, h in pairs}
    assert all(targets[f"{d}:{h}"] == d + h for d, h in pairs)
    # targets of leads 1 and 3 equal the stored keys
    assert all(K[(d, h)] == d + h for d, h in pairs if h in (1, 3))
    tasks = [pairs[i:i + PER_TASK] for i in range(0, len(pairs), PER_TASK)]
    lines = "\n".join(" ".join(f"{d}:{h}" for d, h in t) for t in tasks) + "\n"
    out = HERE / "jobs" / "run16all_pairs.txt"; tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f_:
        f_.write(lines); f_.flush(); os.fsync(f_.fileno())
    assert tmp.read_text() == lines; os.replace(tmp, out)
    R = dict(what=__doc__, start_days=S179, n_start_days=len(S179), done_in_run16=DONE30,
             n_pairs=len(pairs), by_lead={str(h): sum(1 for p in pairs if p[1] == h) for h in (1, 3, 5, 7)},
             n_tasks=len(tasks), per_task=PER_TASK, last_task_size=len(tasks[-1]), max_lead=hp["max_lead"], particles=P,
             all_pairs_are_test_samples=True, targets_equal_start_plus_lead=True)
    txt = json.dumps(R, indent=1); out = HERE / "results" / "plan16all.json"; tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f_:
        f_.write(txt); f_.flush(); os.fsync(f_.fileno())
    assert tmp.read_text() == txt; os.replace(tmp, out)
    print(json.dumps({k: v for k, v in R.items() if k not in ("what", "start_days")}), flush=True)
    print("PLAN OK", flush=True)


if __name__ == "__main__":
    main()
