#!/usr/bin/env python3
"""Seed variability (multi-seed): station CRPS and RMSE per lead of the three trainings (seeds 42, 43, 44) of one
configuration, on identical station-days (the start days evaluated for every seed).
    python figdata_seeds.py <run seed 42> -> data/seeds_<run>.npz"""
import glob, sys
from pathlib import Path
import numpy as np
RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
R = Path(__file__).resolve().parents[1] / "global1km" / "runs"
runs = [RUN, RUN.replace("_L7", "_L7_s43") if "_s4" not in RUN else RUN, RUN.replace("_L7", "_L7_s44")]
runs = [RUN] + [r for r in (RUN.replace("global_L7", "global_L7_s43"), RUN.replace("global_L7", "global_L7_s44"))]
st = np.load("/scratch/project_462001140/ammar/eccv/data/world/global1km/stations_2022.npz"); obs = st["daily"]
data = {}
for r in runs:
    P = [np.load(p) for p in sorted(glob.glob(str(R / r / "eval_global" / "part_*.npz")))]
    k = np.concatenate([p["keys"] for p in P]); e = np.concatenate([p["st_ens"] for p in P])
    data[r] = {(int(d), int(h)): n for n, (d, h, t) in enumerate(k)}, e, k
common = set.intersection(*[set(v[0]) for v in data.values()])
print({r: len(v[0]) for r, v in data.items()}, "common forecasts", len(common))
out = {"runs": np.array(runs)}
for h in range(1, 8):
    keys = sorted(x for x in common if x[1] == h)
    S = np.zeros((len(runs), 3))                                   # n, sum crps, sum se
    for (d, hh) in keys:
        t = d + h
        y = obs[:, t]
        Es = [data[r][1][data[r][0][(d, hh)]].astype(np.float32) for r in runs]
        ok = np.isfinite(y) & np.all([np.isfinite(E).all(1) for E in Es], 0)
        for i, E in enumerate(Es):
            Eo, yo = E[ok], y[ok]; M = Eo.shape[1]
            Es_ = np.sort(Eo, 1); pair = (Es_ * (2 * np.arange(1, M + 1) - M - 1)).sum(1) / (M * (M - 1))
            S[i] += [ok.sum(), (np.abs(Eo - yo[:, None]).mean(1) - pair).sum(), ((Eo.mean(1) - yo) ** 2).sum()]
    out[f"crps_{h}"] = S[:, 1] / S[:, 0]; out[f"rmse_{h}"] = np.sqrt(S[:, 2] / S[:, 0])
    print(f"lead {h}: CRPS " + " ".join(f"{v:.3f}" for v in out[f"crps_{h}"]) + " | RMSE " + " ".join(f"{v:.2f}" for v in out[f"rmse_{h}"])
          + f" | CRPS spread across seeds {100 * np.ptp(out[f'crps_{h}']) / np.mean(out[f'crps_{h}']):.1f} %")
np.savez(Path(__file__).resolve().parent / "data" / f"seeds_{RUN}.npz", **out)
# multi-seed ensemble: the members of the three seeds pooled (3 x 4 = 12 members), same station-days
print("pooled multi-seed ensemble (fair CRPS, spread/skill with (M+1)/M):")
for h in range(1, 8):
    keys = sorted(x for x in common if x[1] == h)
    S = np.zeros(4); SS = np.zeros(3)
    for (d, hh) in keys:
        y = obs[:, d + h]
        Es = [data[r][1][data[r][0][(d, hh)]].astype(np.float32) for r in runs]
        ok = np.isfinite(y) & np.all([np.isfinite(E).all(1) for E in Es], 0)
        E = np.concatenate([E_[ok] for E_ in Es], 1); yo = y[ok]; M = E.shape[1]
        Es_ = np.sort(E, 1); pair = (Es_ * (2 * np.arange(1, M + 1) - M - 1)).sum(1) / (M * (M - 1))
        S += [ok.sum(), (np.abs(E - yo[:, None]).mean(1) - pair).sum(), ((E.mean(1) - yo) ** 2).sum(), E.var(1, ddof=1).sum()]
        E4 = Es[0][ok]; SS += [E4.var(1, ddof=1).sum(), ((E4.mean(1) - yo) ** 2).sum(), ok.sum()]
    ss12 = np.sqrt(13 / 12 * S[3] / S[0]) / np.sqrt(S[2] / S[0]); ss4 = np.sqrt(5 / 4 * SS[0] / SS[2]) / np.sqrt(SS[1] / SS[2])
    out[f"crps_pool_{h}"] = S[1] / S[0]; out[f"rmse_pool_{h}"] = np.sqrt(S[2] / S[0])
    print(f"lead {h}: CRPS 12 members {S[1] / S[0]:.3f} (seed 42 alone {out[f'crps_{h}'][0]:.3f}) | RMSE {np.sqrt(S[2] / S[0]):.2f} | spread/skill 12 members {ss12:.2f} vs seed 42 {ss4:.2f}")
np.savez(Path(__file__).resolve().parent / "data" / f"seeds_{RUN}.npz", **out)
