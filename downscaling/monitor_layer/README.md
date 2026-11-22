# A layer fed by the monitors (reforecast)

Results, *Kilometre-scale forecasts in real time* (the layer fed by the monitors), the Methods paragraph *Monitor-fed layer*
and Supplementary Note *The downscaler, the two-stage system at the official stations of 2022 and in real time, and a
layer fed by the monitors*. A second downscaling network of the same design as `downscaling/cdffm_aq/` receives, as an input, the daily
means of the monitors on the day before the issue (lags 1 to 5 for lead days 1 to 5); it is trained on 2017-2020 with the
1 km estimates as its only target and selected on 2021 against them, and it replaces the station-free member within 150 km
of a monitor that reported. No monitor is a target or selects the layer. The 48 issue days of the real-time record were
reforecast with the archived monitor values and scored at the OpenAQ monitors that lie inside the 188 held-out cells.

| Folder | Working folder (under `$FOEHN_WORK`) | Content |
|---|---|---|
| `cdffm_aq_m/` | `global_model_forecasting_air_pollution/a11/stageM/cdffm_aq_m` | copy of `downscaling/cdffm_aq/` with the lag option (model, training, lag validation) and the station module |
| `feed/` | `global_model_forecasting_air_pollution/a11/stageM/feed` | the day files of archived monitor values that `final_test/build_inputs.py` reads (`build_replay.py`), their format and firewall (`dayfile.py`), quality rules (`qc.py`), the OpenAQ reader (`feed_openaq.py`; its key is read at run time from a file named by `--key-file` or `$OPENAQ_API_KEY_FILE`) and the measurement of the hour at which the daily means of the day before are available (`availability.py`) |
| `./ (val_lag.py, jobs/, tests/)` | `global_model_forecasting_air_pollution/a11/stageM` | training (`jobs/train_M.sh`), validation on 2021 (`val_lag.py`), tests and smoke |
| `final_test/` | `global_model_forecasting_air_pollution/a11/stageM/final_test` | selection on 2021, inputs of the 48 issue days, inference at the monitor pixels, and the scorer `score_final.py` |

`cdffm_aq_m/stations.py` holds the station features, the lag option and the firewall on held-out monitor files. The module
of another experiment, station fine-tuning (`tune_stations.py`), is not part of this release, as in version 1.1.0:
`cdffm_aq_m/train_aq.py` imports it in a `try`, and the layer (`--stn-lag`, without `--tune-stations`) uses none of its
code. `eval_aq.py`, `spectra.py`, `store_ghap.py`, `store_global.py` are identical to the
modules of `downscaling/cdffm_aq/`; `__init__.py`, `data_aq.py`, `model_aq.py`, `train_aq.py` differ from them (path lines, the lag option, the
docstring of the package); `lagval.py`, `stations.py` have no counterpart there. The jobs source
`global_model_forecasting_air_pollution/env.sh`. `final_test/score_final.py` reuses `downscaling/realtime_1km/replay/score/`
and checks the SHA-256 of the files it depends on (`DEPS`): those values are the files as they ran, so the check fails
outside the working tree; its decision and `final_test/jobs/full_job.sh` refuse to run without the filing record.

**Scorers fixed before the runs.** Each was frozen before the run it scored; its SHA-256 is written into its outputs.

| File | SHA-256 of the file that ran | Identical in this release | Difference |
|---|---|---|---|
| `score_final.py` | `79ce9249c0bff086e9fac7d5b67543dbce8eeab595d1c46e0419628dacadb183` | no | the internal name of the real-time folder replaced by `realtime_1km` in four lines (two docstring lines and the two keys of `DEPS` that name the replay scorer), as in version 1.1.0 |

The comments of these files are left as they were when the files were fixed: a 'review' there is an internal check of the code before the run, and a 'session' the working session that supervised the runs.

The scripts name the experiments by working codes (A11 for the layer fed by the monitors, A13 and A14 for the
ensemble-size estimators, A15 and A16 for the noise design on 2021 and 2022, A17 and A19 for the opposite pairs on 2021 and
2022, A18 and A20 for the noise-aware mean on 2021 and 2022); the folders of this release carry descriptive names. The
protocols the scripts refer to (sections, checks, decisions) are not part of the repository.

**Paths and environment.** The scripts are the working files of the study as they ran on the LUMI supercomputer,
with their absolute paths replaced by two environment variables (except the scorers fixed before their runs, released as they
ran): `FOEHN_WORK`, the folder that holds the working trees, and `FOEHN_ENVS`, the folder of the Python environments (`world`, a numpy/xarray virtual environment, and `torchenv`, used inside
the PyTorch container of LUMI). Both must be set: a Python script stops with `KeyError` and a shell script with a message when
one is missing. The scripts expect the layout of the working project under `$FOEHN_WORK`, not that of this repository:
`eccv/cran-pm/experiments/global1km` (here `foehn/`), `eccv/cran-pm/experiments/figures` (`figures/`), the manuscript
workspace `eccv/cran-pm/paper/natcomms` (its `analyses/` is `analyses/` here, its `rebuild/` folders are mapped in the tables
of the READMEs) and the downscaling project `global_model_forecasting_air_pollution` (its downscaling folders are
`downscaling/` here; its experiment folders are mapped in the tables). The folder of the real-time 1 km forecast is named
`realtime_1km`, as in version 1.1.0, and must be reachable under that name. Slurm scripts carry the account placeholder
`<project>` (submit with `sbatch --account=...`) and the GPU jobs run in the PyTorch container image of LUMI
(`/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif`). Data, trained weights, logs
and outputs are not in the repository (see *Data availability* and *Code availability* in the paper).

**Jobs.** Each Slurm script is submitted from the working folder below (under `$FOEHN_WORK`): the folder that holds
its `jobs/` folder, or the script's own folder, as its usage lines show. Its `#SBATCH --output` path is relative to
that folder, and scripts that read their own log back find it only when submitted from there.

| Submitted from (under `$FOEHN_WORK`) | Scripts |
|---|---|
| `global_model_forecasting_air_pollution/a11/stageM/final_test` | `final_test/jobs/cpu_job.sh`, `final_test/jobs/dry_cpu_job.sh`, `final_test/jobs/full_job.sh`, `final_test/jobs/smoke_job.sh` |
| `global_model_forecasting_air_pollution/a11/stageM` | `jobs/smoke_job.sh`, `jobs/train_M.sh`, `jobs/val_lag_job.sh` |

**Not in this release.**

- `downscaling/cdffm_aq/stations.py`, imported by `tests/test_m.py`.
- The CD-FFM modules `flower_layers`, `model` and `fast_train`, imported from the CD-FFM code (see *Code availability* in the paper).
- `global_model_forecasting_air_pollution/env.sh`, the filing records of the protocol, the monitor archives and the trained weights.

## Files

### `cdffm_aq_m/`
| File | What it does |
|---|---|
| `__init__.py` | cdffm_aq_m: the A11 experiment M copy (2026-11-20) of the package cdffm_aq, with ONE option added, off by default: the lag of the station conditioning (--stn-lag; stations.py, lagval.py, train_aq.py, model_aq.py; a11/stageM/README … |
| `data_aq.py` | cdffm_aq.data_aq -- data side of the CD-FFM adaptation (PLAN.md 2.1, 2.5, 2.6, 4.2-1, 5 "Data path") |
| `eval_aq.py` | cdffm_aq.eval_aq -- streaming evaluation of a train_aq run on the hold-out days (PLAN.md 6; smoke S2 gates) |
| `lagval.py` | cdffm_aq_m.lagval -- A11 experiment M: validation of the lagged-monitor conditioning against the 1 km estimates |
| `model_aq.py` | cdffm_aq_m.model_aq -- the network of the minimal CD-FFM adaptation (PLAN.md 2.5, 4.1, 4.2) |
| `spectra.py` | cdffm_aq.spectra -- numpy-only spectral tools (no torch, so the world venv can import it too) |
| `stations.py` | cdffm_aq_m.stations -- station conditioning of the A3 system (PROTOCOL.md Amendment A3(b), A3(c), A3(d')) and, at the end of the file, its LAG option |
| `store_ghap.py` | cdffm_aq.store_ghap -- GhapStore: the GHAP V1 store of the CAMS -> GHAP 1 km downscaler (PROTOCOL.md Amendment A2; data/processed/ghap, built by scripts/build_ghap.py, see its manifest.json), with the SmokeStore / GlobalStore inte … |
| `store_global.py` | cdffm_aq.store_global -- GlobalStore: the global 2017-2024 store (data/processed/global, see its manifest.json) with the SmokeStore interface of data_aq.py, so BatchStream, PixelArrays, assemble_inputs, train_aq and eval_aq run on … |
| `train_aq.py` | cdffm_aq_m.train_aq -- training of the CD-FFM adaptation (PLAN.md 5): CD-FFM fast_train.py + DDP on one LUMI node |

### `feed/`
| File | What it does |
|---|---|
| `availability.py` | At which hour after midnight UTC is the previous day's daily mean available? Direct measurement (experiment M feed) |
| `build_replay.py` | Replay day files of experiment M (Amendment A11): for every issue day d of the 2026 record (realtime_1km/replay/days_all.txt), the assimilated monitors of day d - 1 as a day file (dayfile.py), built from the observations already s … |
| `dayfile.py` | Day files of the monitor-fed layer: writer, loader, firewall |
| `feed_openaq.py` | Live feed of the monitor-fed layer (experiment M of Amendment A11): for a UTC day, the daily mean PM2.5 of every ASSIMILATED OpenAQ monitor, from the OpenAQ v3 API, as a day file of realtime_1km/stations_rt.py (dayfile.py) |
| `qc.py` | Quality rules of the verification observations (realtime_1km/verif/verif.py daily_means, constants of coupling/build_st.py) applied to HOURLY rows of the OpenAQ v3 API instead of the archive's raw records |

### `./`, `jobs/`, `tests/`
| File | What it does |
|---|---|
| `jobs/m_lib.sh` | m_lib.sh -- sourced by the job scripts of A11 experiment M (host side of a Slurm job on LUMI-G) |
| `jobs/smoke_job.sh` | the tests and the smoke test of this step on ONE GCD (limit 50 min = 0.83 GCD-hour) |
| `jobs/train_M.sh` | the pmS_base recipe (runs_ghap/train_S.sh, PROTOCOL.md A3(c): the pmG recipe + --stations data/processed/assim, dropout 0-100 %) with ONE addition, the lag option: for a training day t the station features are those of the assimil … |
| `jobs/val_lag_job.sh` | val_lag.py on ONE GCD (1 GCD-hour requested): every saved and validated checkpoint of the run, the naive baseline (runs_ghap/pmS_final fed with lagged monitors) and the station-free runs_ghap/pmG_final, on the 24 SEL days of 2021  … |
| `tests/check_smoke.py` | Checks of the smoke job of A11 experiment M (jobs/smoke_job.sh); CPU, reads only files of a11/stageM/runs |
| `tests/test_m.py` | Tests of A11 experiment M (lagged monitors as an input; a11/stageM/README.md) |
| `val_lag.py` | validation of the monitor-fed layer for FORECASTS against the 1 km estimates of 2021 ONLY, at the pixels of the selection set of Amendments A2/A3 (the 24 SEL days of 2021 x the 100,000 mask pixels of train_aq.ValData, seed 0) |

### `final_test/`
| File | What it does |
|---|---|
| `build_inputs.py` | Monitor inputs of the final test, CPU, numpy only, no network |
| `ft_lib.py` | a11/stageM/final_test/ft_lib.py -- shared definitions of the FINAL TEST of the monitor-fed layer (experiment M) on the autumn 2026 real-time record |
| `infer_m.py` | Chain-mode inference of the monitor-fed layer (experiment M) at the 12,136 catalogue monitor pixels, for issue days of the autumn 2026 record |
| `jobs/cpu_job.sh` | CPU steps of the final test, 0 GCD-hours, the record scorer's Python (world env, numpy only): sbatch jobs/cpu_job.sh 'score_final.py --tests' known-answer tests of the frozen scorer (no observation file) sbatch jobs/cpu_job.sh 'te … |
| `jobs/dry_cpu_job.sh` | CPU DRY RUN of infer_m.py, 0 GCD-hours, the job container on a CPU node: R the full-run guard: 'infer_m.py --days all' with a non-frozen configuration must be refused before anything loads; D one issue day (2026-10-20), 2 members, … |
| `jobs/full_job.sh` | THE FULL RUN of the final test, submitted separately once FILED exists: the 48 issue days, arms free (identity with the record), free0 (diagnostic of check 4: pmG_final without the fitted prior, the base noise of m1/m2), m1 (prima … |
| `jobs/smoke_job.sh` | THE ONE GPU smoke of the final test of the monitor-fed layer: 1 GCD, limit 40 min (<= 0.67 GCD-h) |
| `score_final.py` | FROZEN scorer of the final test of the monitor-fed layer on the autumn 2026 record |
| `select_2021.py` | Checkpoint of M and blending radius R for the final test, from the 2021 validation ONLY (val_lag tables of pmM_base: runs/pmM_base/val_lag/sel2021_all.json; truth = the 1 km estimates of 2021; no monitor value of any year is read  … |
| `test_final.py` | Tests of the final test of the monitor-fed layer |
