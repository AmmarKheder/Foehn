# Reforecasts in real-time mode at the monitors (WP1a): format

One file per issue day d (00 UTC cycle): `points/<d>.npz`. Lead day L = 1..5 covers forecast hours 24(L-1)..24L; its
valid UTC day is d + L - 1 (lead day 1 is the issue day itself). Axis order: (lead day, member, monitor). Monitors are in
the order of `realtime_1km/verif/catalogue.npz` (n = 12,136 OpenAQ monitors; `id` is repeated in every file). Units ug m-3
unless stated. M = 8 members; downscaled member m is conditioned on coarse member m. No observation is in these files.

| key | shape | content |
|---|---|---|
| `id`, `lat`, `lon` | (n,) | monitor identifier and catalogue position |
| `issue`, `valid` | (), (5,) | issue day; valid day of each lead day (ISO strings) |
| `km` | (5, 8, n) float32 | the 8 downscaled members at the monitor's 1 km pixel, sampled as the production inference (same checkpoint, fitted prior, guard, 10 Euler steps, same noise seeds) |
| `km_ring1`, `km_ring3`, `km_ring10` | (5, 8, n) | per member, mean over the square ring at Chebyshev distance 1, 3, 10 pixels around the monitor pixel (8, 24, 80 pixels; monitor pixel excluded; pixels outside the 1 km mask ignored; NaN when the ring has no mask pixel) |
| `ring1_n`, `ring3_n`, `ring10_n` | (n,) int16 | number of mask pixels in each ring. The 2022 station test kept a ring only with at least 4 d of its 8 d pixels (`ring<d>_n >= 4 d`): apply that rule when scoring |
| `km_cell` | (5, 8, n) | per member, mean of **log1p(member)** over a regular 5 x 5 sample of the monitor's 0.25 degree cell (**log1p units**) |
| `km_cell_lin` | (5, 8, n) | the same sample mean in ug m-3 (extra) |
| `cell_n` | (n,) int16 | number of sample pixels inside the mask (25 for most cells; 0 for 14 monitors: NaN there) |
| `rt04` | (5, 8, n) | each 0.4 degree member, daily mean of the lead day, bilinear at the monitor |
| `cams` | (5, n) | CAMS operational forecast of the same 00 UTC cycle, same daily mean, bilinear (NaN when the ten leads are not on disk) |
| `clim` | (5, n) | expm1 of the harmonic climatology of log1p of the 1 km target (2017-2020) at the pixel, for the valid day |
| `bcsd` | (5, n) | quantile map + climatological pattern: expm1(mu_p - mubar_c + log1p(mean over members of expm1(QM_c,month(log1p x_c,m)))) |
| `an` | (5, 8, n) | analysis-mode downscaled members valid on the same day at the monitor pixel (first-day mean of the CAMS run and reanalysis weather as inputs), NaN when not on disk (valid days after 2026-10-21) |
| `bcsd_m`, `qm_cell` | (5, 8, n) | extras: the BCSD arm per member (no member mean), and QM_c,month(log1p x_c,m) in log1p units (cell level of a coupling) |
| `mu_cell` | (5, n) | extra: mubar_c, the cell climatology of log1p of the target, for the valid day |
| `provenance` | () | JSON: checkpoints and their SHA-256, sampler, seeds, input sources, which coarse forecast (archived or recomputed), timings, code hashes |

Daily mean of a coarse field (`rt04`, `cams`, input of `bcsd`): max(1/4 x(24(L-1)) + 1/2 x(24L-12) + 1/4 x(24L), 0); the
0 h instant is the CAMS analysis of the issue time. A lead day that could not be made is NaN on its whole row
(`provenance["leads_made"]`).

`points/INDEX.json` lists, per issue day, the lead days present for `km`, `cams`, `an` and the other arrays, the source of
the coarse forecast (archived or replayed) and the overlap with the fine-tuning targets (`code/make_index.py`; it is a
snapshot: rerun that script after `cams` has been filled for the September days).

Reading example:

```python
import json, numpy as np
z = np.load("points/2026-10-20.npz")
km = z["km"]                       # (5, 8, 12136)
prov = json.loads(str(z["provenance"]))
ring3 = np.where(z["ring3_n"] >= 12, z["km_ring3"], np.nan)     # rule of the 2022 station test
```

Other products: `km/<d>.npz` (the downscaler part alone), `raw/<d>.npy` (5, 8, 875425) float32 = the members at every
evaluated pixel, in the order of `positions.npz['pos']` (indices `mon`, `ring1`, `ring3`, `ring10`, `cell`, `cell_idx`
there); `coarse/<d>/pred.npy` (8, 10, 361, 900) float16 = the 0.4 degree members at +12..+120 h (a symlink to the
archived forecast for 2026-11-01..11-11, recomputed for the other days; `coarse/<d>/green/` = its transport stores);
`work/<d>/chan_L<L>.npz` = the 47 encoder channels of each lead day. Log, checks and numbers: `STATUS.md`, `check/`.

## What a file does and does not contain

- Forecast mode means: coarse forecast of the real-time 0.4 degree model from the inputs of the day (CAMS analyses of
  00 UTC and 12 h earlier, IFS open data of the 00 UTC run, fires of the previous day), then the downscaler on IFS
  forecast weather, persisted fires and forecast transport, exactly as the daily chain. Nothing uses a monitor.
- For 2026-11-01..11-11 the coarse members are the archived production files; for the other days they were recomputed
  (transport + sampler) and reproduce the archived evidence (see `STATUS.md`, checks).
- Issue days 2026-09-01..09-04: some predicted instants were targets of the fine-tuning of the coarse model (analyses up
  to 2026-09-05 00 UTC); the count per lead day is in `provenance["coarse"]["predicted_instants_that_were_training_targets"]`.
- 2026-10-10: lead day 1 of the downscaled arrays (`km*`) is NaN (no inputs of 2026-10-09 on disk for its transport
  channel); the baselines of that lead day are present.
- `rt04`: each member's daily mean is floored at 0 before the bilinear reading (as the downscaler input); the member mean
  therefore differs by up to about 0.05 ug m-3 at a few monitors from a mean taken before the floor.
- `km_cell` is in log1p units; every other forecast array is in ug m-3. Members are not sorted and member m of `km`,
  `rt04`, `bcsd_m`, `qm_cell` belongs to the same coarse member m; `an` members are independent draws.

## Re-running (all code in `code/`; nothing is written outside this directory)

```bash
R=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay; cd $R/code
export PYTHONDONTWRITEBYTECODE=1                              # the production modules are imported: no .pyc next to them
python -I positions.py                                        # once: the pixels to evaluate (CPU)
sbatch prep_job.sh met $R/days_all.txt                        # CPU, internet: weather channels (about 5 min per day, 8 at a time)
sbatch coarse_job.sh coarse 2026-09-01 2026-09-02 ...         # GPU node: transport (14 min per day per GCD) + coarse sampler (1.5 min)
sbatch km_job.sh $R/days_all.txt                              # GPU node: channels + pointwise downscaling, 3.5 min per day per GCD
python -I assemble.py --list $R/days_all.txt                  # CPU: baselines + provenance -> points/<d>.npz (a few seconds per day)
python -I check_all.py                                        # CPU: summary table and comparison with the published snapshots
```

Checks: `check_coarse.py` (replayed coarse forecast against an archived one), `check_coarse_archive.py` (against the
published NetCDF files of October and the archived September scores), `check_points.py <day> --tiles` (pointwise members
against the published snapshot and tiles, and the 1 km inputs against the production ones when they are still on disk).
