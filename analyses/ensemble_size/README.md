# Scores for eight and for infinitely many members

Supplementary Table *Scores by ensemble size*, the values "with eight members" and "for infinitely many members" given
beside the four members as run throughout the paper, Fig. 2f, Supplementary Fig. 4 (*People reached, correct warnings and
hidden people warned, for Foehn and five references*) and Supplementary Tables *Who is warned, 1 day ahead*, *The reference
forecasts at the number of people warned by Foehn, 1 day ahead* and *Hidden people by region, above 37.5 ug m-3*, with eight
members. Sixteen members were run for every start day of 2022, 1 and 3 days ahead, at the populated grid cells.
Mean-squared errors, spread/skill and expected CRPS for M members follow from per-forecast sufficient statistics (the
mean-squared error of the mean of M members is that of infinitely many members plus the mean variance of the members
divided by M); warnings for eight members are the expected values of eight members drawn from the sixteen without
replacement, and the infinite limit is extrapolated in 1/M from 4 to 16 members.

| Folder | Working folder (under `$FOEHN_WORK`) | Content |
|---|---|---|
| `./ (check13*.py, run_design.sh, jobs/)` | `global_model_forecasting_air_pollution/a14` | gate of the sixteen-member run and the driver of the whole pass under a design (`FOEHN_DESIGN=ar`) |
| `infinite_form/` | `global_model_forecasting_air_pollution/a14/classU/scripts` | every score in its infinite form, its eight-member form and their intervals (stations, populated pixels, land, real time, downscaler, Foehn-CAMS); `j_m0.py`, the independent check of the member-0 estimator, comes from `a14/classU_review/judge`, a folder name the scripts keep |
| `counted_members/` | `global_model_forecasting_air_pollution/a14/classE` | warnings and other counted scores at 4, 8, 16 and infinitely many members |
| `sixteen_members/` | `global_model_forecasting_air_pollution/a13/warn16` | the sixteen-member sampler of the first design and the definitions and checks of the sixteen-member warnings, imported unchanged; the forecast lists and jobs of the sixteen-member runs of the first design (`check_inputs.py`, `plan16all.py`, `jobs/`; the run on the 149 other start days of 2022 is `../jobs/array16_13.sh` with `../jobs/pairs13.txt`) |
| `table_by_size/` | `eccv/cran-pm/paper/natcomms/rebuild/print/d5` | Supplementary Table *Scores by ensemble size* |
| `warnings_eight/` | `eccv/cran-pm/paper/natcomms/rebuild/print/warn8` | the three warning tables with eight members, the bar heights of Fig. 2f and Supplementary Fig. 4 (`fig_hidden_warned_m8.py`) |

**Noise design switch.** `FOEHN_DESIGN=ar` selects the design of the paper (the noise field of each member drawn anew
at every start day and evolving between lead days as a first-order autoregressive process, rho = 0.59); every run folder
and derived file of that design carries the suffix `_ar`. With `FOEHN_DESIGN` unset, the scripts read and write the files of
the first design (fields drawn once), as in version 1.1.0.

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
| `global_model_forecasting_air_pollution/a14/classE` | `counted_members/jobs/cpu_job.sh`, `counted_members/jobs/run2022.sh` |
| `global_model_forecasting_air_pollution/a14/classU` | `infinite_form/cpu_job.sh` |
| `global_model_forecasting_air_pollution/a14` | `jobs/array16_13.sh`, `jobs/cpu_job.sh`, `run_design.sh` |
| `global_model_forecasting_air_pollution/a13/warn16` | `sixteen_members/jobs/array16.sh`, `sixteen_members/jobs/array16all.sh` |

**Not in this release.**

- `downscaling/runs_global/score_2024.py` (working folder `global_model_forecasting_air_pollution/runs_global`), imported by `infinite_form/ds2022_classU.py`.

## Files

### `./`
| File | What it does |
|---|---|
| `check13.py` | checks of run R1, CPU only, no station or monitor file |
| `check13b.py` | the gate of run R1 in the words of A14 section 5, from the output of check13.py (unchanged, job 23900798) |
| `jobs/array16_13.sh` | usage (from a14): sbatch --array=0-36 --time=03:15:00 jobs/array16_13.sh A14 run R1: copy of a13/warn16/jobs/array16all.sh whose only changes are the job name, the log folder, the forecast list (a14/jobs/pairs13.txt: leads 1 and 3 … |
| `jobs/cpu_job.sh` | usage: sbatch jobs/cpu_job.sh "<python command run in a14>" (CPU only, the training container's env) |
| `run_design.sh` | REBUILD_PLAN 4c step 8 (A14) under a noise design, in order, CPU only (a14/DESIGN_SWITCH.md): sbatch --export=ALL,FOEHN_DESIGN=ar run_design.sh the runs of A16 (noise drawn anew at every start day) sbatch --export=ALL,FOEHN_DESIGN … |

### `infinite_form/`
| File | What it does |
|---|---|
| `c25_pass.py` | the emission gains on 0.25 deg cell means (main_text.tex:275, supplementary.tex:510: North China Plain 5.6, Indo-Gangetic Plain 3.0 over the week; East Asia 2.6, South Asia 2.5) |
| `cams12h_classU.py` | A14 class U for Foehn-CAMS in Aurora's protocol (blind record, D4: its first score, eight members, as the Methods state) |
| `chain_classU.py` | A14 class U for the downscaler and the two-stage system at the official stations of 2022 (Supplementary Note chain2022, Supplementary Table chain2022; D4: blind record, as-run first) and for Foehn-CAMS read at the stations (its 'c … |
| `cpu_job.sh` | A14 class U, CPU only (partition small, no GPU): sbatch [--job-name N --time T --mem M] jobs/cpu_job.sh <script.py> [args] world venv (numpy, zarr), as analyses/reliability/scripts/run_small.sh; no bytecode written anywhere (-B). |
| `ds2022_classU.py` | A14 class U for the test of the real-time downscaler on 2022 against the 1 km estimates (Methods and Supplementary Methods 'Tests of the downscaler': RMSE of the ensemble mean 9.98 against 13.50 for the coarse field and 12.37 for  … |
| `emis_pair.py` | the draw of a two-model gain (emissions: model with emissions e02 against its control ctrl; transport: base model A against the network without transport C) |
| `j_m0.py` | Independent check of the member-0 estimator of A14 4a on 0.25 deg cells: at the populated pixels, where all four stored members exist (RUN/eval_pop_crps_e01), the cell mean of each member over the populated pixels of each 0.25 deg … |
| `j_m0_design.py` | the reference member-0 ratio (a14/classU_review/judge/j_m0.py) recomputed on the populated-pixel run of a noise design, RUN/eval_pop_crps_e01<ISFX> -> results_<D>/j_m0.json |
| `land25_pass.py` | A14 class U over land on 0.25 deg cell means (experiments/figures/compare_ghap_global.py: the paper's correlation with the 1 km estimates 0.89 / 0.80 / 0.76 at 1 / 4 / 7 days, RMSE a third below the climatology at 1 day and 3 % at … |
| `make_table.py` | the ONE table (results/classU_table.md + .json) of every class-U quantity the paper reports, with its place in the paper (file:line), the as-run value (identity with the paper's printed value), the eight-member and infinite values … |
| `pop_pass.py` | A14 class U at the 8.1 M populated pixels (against the 1 km estimates; no station or monitor is read here) |
| `r1_pipeline_b.py` | A14 class U on all 179 start days, after R1 (a14/data/run16_13) has passed check13b (a14/results/check13b.txt 'passed'; amendment A14a, the rounding tolerance of check 13) |
| `rel_pass.py` | A14 class U, stations, scorer analyses/reliability (s2_station.py -> sums_final.npz -> s2b_station_scores.py): the extra per-forecast sums that the eight-member forms and the jackknife over the four fields need and sums_final.npz  … |
| `rel_score.py` | A14 class U for the scores of analyses/reliability (stations: spread/skill, Brier skill, CRPS against the climatological ensemble, RMSE on the station-days of the probabilistic scores; populated pixels; land of the nine regions) a … |
| `rt_classU.py` | A14 class U for the real-time record of 2026 (eight members as issued; D4: as-run first, infinite beside) |
| `st_cf_pass.py` | A14 class U, stations, scorer experiments/figures/compare_fair.py (the paper's RMSE at the stations 1-4 days ahead and the gains over the CAMS forecast, Aurora, the climatology, the persistences, and of the 0.25 / 0.5 deg averages … |
| `st_cf_score.py` | A14 class U at the stations for the quantities of experiments/figures/compare_fair.py (RMSE of the ensemble mean at the stations 1-4 days ahead; gains over the CAMS forecast and Aurora issued at the same time, the climatology, the … |
| `u_common.py` | common paths and helpers (P/a14/classU) |
| `u_quart.py` | A14 class U from sufficient statistics of a set of members (any subset Q of the 16 fields) |

### `counted_members/`
| File | What it does |
|---|---|
| `classE.py` | A14 class E, count rules |
| `jobs/cpu_job.sh` | copy of a14/jobs/cpu_job.sh for A14 class E: usage (from a14/classE): sbatch [--job-name N --time T] jobs/cpu_job.sh "<command>" CPU only (partition small, no GPU) |
| `jobs/run2022.sh` | A14 class E of 2022 on all 179 start days (classE.py run2022, version 2): samples conf149 (confirmatory), seen30, seen30_paperblocks, all179; World, Foehn 3 days against the references 1 day, regions at 4/8/16 members |

### `sixteen_members/`
| File | What it does |
|---|---|
| `check16all.py` | checks of the 16-member run on all 179 start days (data/run16all), CPU only, no station or monitor file |
| `check_inputs.py` | CPU checks before the first sixteen-member run (30 start days): identity of the checkpoints, noise members 0-3, accumulation at the populated pixels, start days and keys |
| `jobs/array16.sh` | usage: sbatch --array=0-14 --time=... jobs/array16.sh Array task k (0..14) samples, with 16 members, the start days 24k and 24k + 12 (two of the 30 start days 0, 12, ..., 348 = every sixth start day of analyses/hidden/v3), leads 1 … |
| `jobs/array16all.sh` | usage: sbatch --array=0-81 --time=03:30:00 jobs/array16all.sh The 16-member run on all 179 start days of analyses/hidden/v3 (plan16all.py): array task k samples the 8 forecasts of line k+1 of jobs/run16all_pairs.txt ('start day:le … |
| `plan16all.py` | A13 section 4, follow-up of 2026-11-20 after the rule was met on the 30 start days: the 16-member run on ALL 179 start days of analyses/hidden/v3 |
| `sample16.py` | A13 section 4 (warnings with 16 members, no training): the frozen final 1 km model sampled with M members at the 8.1 M grid cells with at least 100 residents (pop_pixels.npz), for chosen start days and leads |
| `score16.py` | score the warnings of Foehn with 4 and with 16 members against the start-day 1 km estimate kept unchanged, on every sixth start day of analyses/hidden/v3 (30 start days, leads 1 and 3) |

### `table_by_size/`
| File | What it does |
|---|---|
| `build_d5.py` | A14 decision D5 (A16 Addendum 3, point 2(c)): the Supplementary Table 'Scores by ensemble size' (label tab:si_members) |
| `pass16.py` | D5: the per-start-day sixteen-member sums at the stations, their proof against the A14 record, the 16-member intervals |

### `warnings_eight/`
| File | What it does |
|---|---|
| `build_warn8.py` | Warnings of Foehn with EIGHT members: the complete Supplementary Tables 'warned', 'matched' and 'regions', their short printed versions (LaTeX) and the bar heights of Fig. 2f on the 149 confirmatory start days of 2022 (sample conf … |
| `complete_warn8.py` | What the class-E record does not hold, computed on the record's OWN resamples with the paper's own code imported read-only (nothing of it is modified): a14/classE/classE.py sample_rows (conf149 / all179: tables.mbb(179, 4), seed 0 … |
| `fig_hidden_warned_m8.py` | Supplementary Fig. 4 as submitted: who is warned, Foehn with eight members |
| `test_identity.py` | The proof of the key mapping of build_warn8.py |
