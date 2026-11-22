#!/usr/bin/env python3
"""A14 run R2: the 15 start days of 2021, by the rule fixed in protocol_amendments/A14_infinite_ensemble.md (section 5)
BEFORE this script was run. Only the 1 km estimates of 2021 (0.25 degree block means, pm.npy) and the residents (pop025.npy)
are read; no forecast, no station, no monitor.
  candidates: start days d of 2021 that the test dataset of eval_global_green (year 2021, the final model's store) has with
              leads 1 and 3 (targets d + 1 and d + 3);
  score S(d): residents in 0.25 degree cells where the block mean of the 1 km estimates of day d + 1 exceeds 75 ug m-3;
  choice:     the 15 largest S(d), taken greedily in that order, with at least 7 days between any two chosen start days.
    python select2021.py -> a14/r2/results/days2021.json (chosen days, their S, the 'start day:lead' pairs, 2 per array task)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

GLOBAL1KM = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km"))
sys.path.insert(0, str(GLOBAL1KM))
import torch  # noqa: E402

import eval_global_green as egg  # noqa: E402  (unchanged)

CKPT = GLOBAL1KM / "runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/final.ckpt"
HERE = Path(__file__).resolve().parent
BASE = 365 + 365 + 365 + 366          # index of 2021-01-01 in pm.npy (days of 2017-2022, build_coarse.DAYS)


def main():
    hp = torch.load(CKPT, map_location="cpu", weights_only=False)["hyper_parameters"]
    store, emis = hp.get("store", "green"), bool(hp.get("emis", False))
    P = int(egg.Npy(egg.G / store / "hres" / "origins_2021.npy").shape[-2])
    ds = egg.TestGreen(hp["max_lead"], P, store, emis, 2021, hp.get("emis_store"))
    have = set(ds.samples)
    starts = sorted({i for i, h in ds.samples})
    cand = []
    for i in starts:
        if (i, 1) in have and (i, 3) in have:
            d = int(ds.doy[i])
            assert int(ds.doy[i + 1]) == d + 1 and int(ds.doy[i + 3]) == d + 3, "targets are not d + 1 and d + 3"
            cand.append(d)
    assert len(cand) == len(set(cand))
    pm = np.load(egg.G / "pm.npy", mmap_mode="r")
    pop = np.nan_to_num(np.load(egg.G / "pop025.npy").astype(np.float64))
    assert pm.shape[1:] == pop.shape
    S = {}
    for d in cand:
        x = np.asarray(pm[BASE + d + 1], np.float64)
        S[d] = float(pop[np.isfinite(x) & (x > 75.0)].sum())
    order = sorted(cand, key=lambda d: (-S[d], d))
    chosen = []
    for d in order:
        if all(abs(d - c) >= 7 for c in chosen):
            chosen.append(d)
        if len(chosen) == 15:
            break
    assert len(chosen) == 15
    chosen_sorted = sorted(chosen)
    lines = [f"{d}:1 {d}:3" for d in chosen_sorted]
    R = dict(what=__doc__, n_candidates=len(cand), candidates_first_last=[min(cand), max(cand)],
             chosen_in_order_of_choice=chosen, chosen=chosen_sorted,
             residents_above_75_on_day_d_plus_1_M={str(d): round(S[d] / 1e6, 1) for d in chosen_sorted},
             median_over_candidates_M=round(float(np.median(list(S.values()))) / 1e6, 1),
             pairs_per_task=lines)
    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / "days2021.json"; txt = json.dumps(R, indent=1); tmp = out.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w") as f_:
        f_.write(txt); f_.flush(); os.fsync(f_.fileno())
    assert tmp.read_text() == txt; os.replace(tmp, out)
    (HERE / "jobs" / "pairs2021.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in R.items() if k != "what"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
