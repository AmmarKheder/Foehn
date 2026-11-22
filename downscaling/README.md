# Generative downscaling to 1 km and the real-time 1 km forecast

The second stage of the real-time system of the paper: a generative model that draws a 1 km member of daily PM2.5 over all
land from each member of a coarse forecast (Methods and Supplementary Methods, *Real-time forecasting*: *Downscaling to 1 km*, *From the coarse ensemble
to 1 km*, *Calibration of the real-time ensemble*, *Tests of the downscaler*, *Verification at monitors*).

**Dependency.** The encoder, the query and the pointwise flow are those of the functional flow matching code of Liu, Toropainen & Boy (CD-FFM; https://github.com/Holmes-Alan/CD-FFM, archived at https://doi.org/10.5281/zenodo.23499076). That code is imported, not copied:
clone it into `CD-FFM/` beside this folder (`cdffm_aq/__init__.py` holds the path).

**Layout** (the layout of our working project; the folder of the 1 km forecast is named `realtime_1km` here).

- `cdffm_aq/`, `cdffm_aq_x/`: the model for PM2.5 on the whole globe (periodic encoder with 47 input fields, query at each
  1 km pixel, noise on the sphere, stores of the training data, training, evaluation; the prior refitted after training and
  the sampler are in `cdffm_aq_x/`).
- `scripts/`, `jobs/`: construction of the training stores (1 km estimates, coarse fields, weather, fires, one-day transport
  from `transport/flexgreen_pm.py`) and their Slurm jobs.
- `runs_ghap/`: the downscaler of the paper. `train_G.sh` trains it, `p1r_G.sh` refits its noise prior, `fit_bcsd_ghap.py`
  fits the statistical baseline (quantile mapping per 0.25° cell and month with the climatological 1 km pattern),
  `pmG_cal/` holds the calibration of the ensemble (tapered widening of the noise prior, chosen on 2021 against the 1 km
  estimates) and `pmG_seed1/a8/` the two trainings with other seeds and their comparison.
- `test2022_ghap/`, `test2022_official/`, `test2022_cal/`: the tests on 2022 (against the 1 km estimates, at the official
  stations of the paper with the two-stage system, and the prepared pass of the calibrated sampler).
- `realtime_1km/`: `prep.py` and `channels.py` build the inputs of a cycle, `infer.py` draws the eight 1 km members for five
  lead days, `monitors_openaq.py` builds the daily means of the OpenAQ monitors used for verification only;
  `replay/code/` recomputes past cycles (reforecasts) at the monitors from the data of their day, `replay/score/` is the
  scorer of the paper (identical station-days, two-way block bootstrap), and `replay/cams3h/` with `replay/score_cams3h/`
  builds the daily mean of the CAMS forecast from its eight 3-hourly steps and holds the checks of that reading.
- `monitor_layer/` (version 1.2.0): a second downscaling network that receives the monitors of the day before as an input,
  and the reforecast of the 48 issue days that tests it at held-out monitors (its own README).

**Stations and monitors are evaluation data only** in every result of the paper, with one stated exception: the layer fed by
the monitors (`monitor_layer/`, version 1.2.0) receives the measurements of the day before as an input; no monitor is a
target or selects it, and it is scored at monitors held out by place. `cdffm_aq/train_aq.py` and `realtime_1km/infer.py`
also carry options for experiments made after the system of the paper was frozen (28 October 2026); the options are off by
default and their modules are not part of `cdffm_aq/` or `realtime_1km/` (the copy of the package used by the layer,
`monitor_layer/cdffm_aq_m/`, holds `stations.py`, which the layer needs).

The scripts are the working files of the study, as they ran on the LUMI supercomputer: they carry the paths of our project space and the Slurm settings of that machine, like the rest of this repository. Data, trained weights, logs and outputs are not in the repository (see *Data availability* and *Code availability* in the paper).

## Files

### `cdffm_aq/`, `cdffm_aq_x/`, `transport/`
| File | What it does (first line of its own header) |
|---|---|
| `__init__.py` | cdffm_aq: the minimal CD-FFM adaptation for CAMS PM2.5 (0.4 deg) -> LGHAP v2 (1 km) downscaling (PLAN.md 4.2) |
| `bench_s1.py` | cdffm_aq.bench_s1 -- S1 encoder benchmark of PLAN.md section 7 (dev-g, 1 GCD, <= 30 min; jobs/bench_encoder.sh) |
| `data_aq.py` | cdffm_aq.data_aq -- data side of the CD-FFM adaptation (PLAN.md 2.1, 2.5, 2.6, 4.2-1, 5 "Data path") |
| `eval_aq.py` | cdffm_aq.eval_aq -- streaming evaluation of a train_aq run on the hold-out days (PLAN.md 6; smoke S2 gates) |
| `model_aq.py` | cdffm_aq.model_aq -- the network of the minimal CD-FFM adaptation (PLAN.md 2.5, 4.1, 4.2) |
| `spectra.py` | cdffm_aq.spectra -- numpy-only spectral tools (no torch, so the world venv can import it too) |
| `store_ghap.py` | cdffm_aq.store_ghap -- GhapStore: the GHAP V1 store of the CAMS -> GHAP 1 km downscaler (PROTOCOL.md Amendment A2; data/processed/ghap, built by scripts/build_ghap.py, see its manifest.json), with the SmokeStore / GlobalStore inte … |
| `store_global.py` | cdffm_aq.store_global -- GlobalStore: the global 2017-2024 store (data/processed/global, see its manifest.json) with the SmokeStore interface of data_aq.py, so BatchStream, PixelArrays, assemble_inputs, train_aq and eval_aq run on … |
| `test_s0.py` | cdffm_aq.test_s0 -- S0 CPU checks of PLAN.md section 7 for the model code |
| `train_aq.py` | cdffm_aq.train_aq -- training of the CD-FFM adaptation (PLAN.md 5): CD-FFM fast_train.py + DDP on one LUMI node |

| File | What it does (first line of its own header) |
|---|---|
| `__init__.py` | cdffm_aq_x: mechanism variants of the CD-FFM adaptation (IDEAS_REPORT.md), built ON TOP of cdffm_aq |
| `copula_x.py` | cdffm_aq_x.copula_x -- the copula study (PROTOCOL.md 0.3 and 4.4, C0/C1) on the FROZEN baseline, smoke setup |
| `data_x.py` | cdffm_aq_x.data_x -- data-side additions for the variants |
| `eval_x.py` | cdffm_aq_x.eval_x -- cdffm_aq.eval_aq for every variant, plus block-mean (joint-structure) scores |
| `fig_x.py` | cdffm_aq_x.fig_x -- one diagnostic PNG: (a) member / LGHAP and ensemble-mean / LGHAP power ratios against wavelength for several runs; (b) spread/skill (Fortin) of block means at 1 km (pixels), 0.25, 0.5 and 1 deg |
| `model_x.py` | cdffm_aq_x.model_x -- the variant mechanisms (see the package docstring) and one member sampler for all of them |
| `report_x.py` | cdffm_aq_x.report_x -- results table of the smoke comparison (pointwise, block and spectral scores) with paired day-block bootstrap intervals against a reference run |
| `test_x.py` | cdffm_aq_x.test_x -- CPU checks of the variants (toy grid 72 x 144, toy store), in the spirit of cdffm_aq.test_s0 |
| `train_x.py` | cdffm_aq_x.train_x -- training of the variants: cdffm_aq.train_aq with --variant {base, estab, hcf}. Everything else is the baseline's (losses, optimiser, schedule, EMA, DDP, validation protocol, checkpoints): AQModuleX subclasses … |

| File | What it does (first line of its own header) |
|---|---|
| `flexgreen_pm.py` | flexgreen_pm.py -- copy of Foehn's cran-pm/experiments/global1km/flexgreen.py (2026-10-20) for the CAMS -> LGHAP project (PLAN.md 2.7 item 5), with behaviour-preserving options only (the defaults reproduce flexgreen.py exactly): - … |

### `scripts/` and `jobs/`
| File | What it does (first line of its own header) |
|---|---|
| `backfill_2025_2026.py` | backfill_2025_2026.py -- the inputs of the 1 km analysis from 2025-01-01 to the last complete day, with the SAME code paths as the 2017-2024 stores (functions imported from bu … |
| `build_ghap.py` | GHAP V1 store of the CAMS -> GHAP 1 km downscaler (PROTOCOL.md Amendment A2) in data/processed/ghap/. Source (read only): eccv/data/world/global1km/ghap1km_global.npy (2,191 days 2017-01-01 .. 2022-12-31, 14,400 x 36,000 float16;  … |
| `build_global_cellmeans.py` | z cell-mean table of the global store for the hcf variant (cdffm_aq_x), = data_x.build_cellmeans on the global store, parallel over days (fork pool; the store and its per-pixel arrays are shared copy-on-write) |
| `build_global_coarse.py` | Global coarse inputs of the CAMS -> LGHAP v2 downscaling (PLAN.md 2.1, 2.6, 2.7 items 1-3) into data/processed/global/coarse/: met_2023_2024.npy (731, 37, 576, 1440) float16 ERA5 37 channels of 2023-01-01 .. 2024-12-31, row = (dat … |
| `build_global_manifest.py` | data/processed/global/manifest.json: merges manifest_parts/*.json, adds the training-day normalisation numbers (CAMS log1p mean / std, T6 channel mean / std: split-A training days only, global, unweighted, as the smoke store), eve … |
| `build_global_t6.py` | T6 FLEXGREEN transport channels of every target day D = 2017-01-02 .. 2024-12-31 (PLAN.md 2.1 ch 41-46, 2.7 item 6), same code as build_smoke_t6.py (gather() verbatim from fullflow_global_green.py), into data/processed/global/tran … |
| `check_flexgreen_pm.py` | Non-regression of transport/flexgreen_pm.py (--max-lead 1, --out-root, --fires-new) against the Foehn ERA5 store: lead index 0 of the origins and fire terms of the given start days (year, doys) must be bit-identical |
| `check_injh_store.py` | GlobalStore(fires=injh) check (global-prep): channels 0-1 identical to mami, channel 2 = injh, real batch finite. |
| `dl_era5_B_arco.py` | ERA5 3-D FLEXGREEN driving fields (PLAN.md 2.7 item 4) from ARCO-ERA5 (the WeatherBench2 ERA5 archive stops on 2023-01-10), in the exact format of Foehn's dl_wb2.py era5(): wb2/era5_B_<year>_<month>.npy (days, 4, 53, 181, 360) flo … |
| `dl_era5_daily_arco.py` | ERA5 daily stores of 2026 (and any later year) from ARCO-ERA5, in the format of the archives the met builder reads (build_global_coarse.met_day): backfill of the 1 km analysis inputs |
| `dl_gfas_injh.py` | GFAS injection height 'injh' (IS4FIRES) 2017-2024 and the injh version of the global fires store (the fires input of the 1 km model used in real time) |
| `env_check.py` | env_check.py -- checks of the cdffm environment (inside the SIF, after `source env.sh`; jobs/env_check_job.sh) |
| `era5_2026_repair.py` | Repair of the ERA5 2026 stores after the parallel array 23422009 (2026-11-03): 10 tasks opened the same yearly zarr at once -> 8 died on the create race (mode 'w' on an existing path), and the survivors overwrote attrs done_days/d … |
| `measure_noise_beta.py` | Fit of the power-law noise prior on LGHAP z-fields of TRAINING days (PLAN.md 4.2 "Noise-prior fitting"; smoke) |
| `north_fill.py` | North of the LGHAP domain (60 N): there is no 1 km truth, so no CD-FFM. Fill = CAMS (or RT, its emulator) corrected by a boreal quantile mapping (BQM): per calendar month, the quantiles of log1p CAMS day-0 at 0.25 deg land cells a … |
| `test_global_hcf.py` | CPU checks of the HCF variant ON THE GLOBAL STORE (2026-10-26; PROTOCOL.md Amendment A1) |
| `test_global_loader.py` | CPU dataloader test of the global store (cdffm_aq.store_global.GlobalStore + data_aq.BatchStream, unchanged): 1 |
| `test_global_wiring.py` | CPU checks of the --store global wiring (global-prep, 2026-10-26) |
| `test_store_backfill.py` | Unit test of the 2025-2026 backfill in GlobalStore / GhapStore (backfill 2026-11-02): for the reference days of backfill/refs.npz (3 per year, rows recomputed from the sources by scripts/backfill_2025_2026.py refs) raw_inputs(d) m … |

| File | What it does (first line of its own header) |
|---|---|
| `bf_build.sh` | Backfill: coarse/{met,fires_injh,cams_pm25_025}_2025_2026.npy from the archives (rows available, NaN beyond) + coverage. |
| `bf_dl_cams.sh` | Backfill: CAMS 00 UTC forecasts 2026 (pm2p5, pm10, 5 lead days) month by month from the ADS into eccv/data/world/cams_forecast_daily/2026.zarr (partial current month); resumable (done_days attr). |
| `bf_dl_gfas.sh` | Backfill: GFAS monthly files 2025-01 .. this month (tpmfire, frpfire, injh) from the ADS into data/raw/gfas_injh/ (<y>-<m>.nc, partial month <y>-<m>_to<dd>.nc); resumable. |
| `bf_era5.sh` | Backfill ERA5 from ARCO, one month per task: 0-10 daily 30-ch 2026-01..11 (era5_global_daily/2026.zarr), 11-21 extra 7-ch 2026-01..11 (era5_extra_daily/2026.zarr), 22-43 era5_B 3-D FLEXGREEN fields 2025-02 .. 2026-11 (data/process … |
| `bf_era5_fix.sh` | Backfill: repair of the ERA5 2026 stores after the create race of array 23422009 (task 0 = daily 30-ch, 1 = extra 7-ch): rebuild the attrs from the chunks on disk, then fill the missing months SEQUENTIALLY (one process per store). |
| `bf_era5_last.sh` | Backfill: last finite ERA5 day in ARCO -> backfill/arco_last_day.txt; ARCO daily builder validated on 3 days of 2025. |
| `bf_fg_20241231.sh` | Backfill: the FLEXGREEN start day 2024-12-31 (doy index 365 of 2024, lead 1), absent from the 2024 store because its ERA5 days 2025-01-01.. did not exist when 2024 was built |
| `bf_flexgreen.sh` | Backfill: FLEXGREEN ERA5 lead 1 for the start days 2025-01-01 .. (last era5_B day - 8, and <= last fires day), as jobs/global_flexgreen.sh (8 ranks share the days); fires of 2025+ from the injh store (--fires-2025) |
| `bf_flexgreen_tail.sh` | Backfill: the last start days (era5_B day - 8 < s <= era5_B day - 2) with --era5-days 3 (lead 1 needs days d .. d+2 only) |
| `bf_t6.sh` | Backfill: T6 transport channels 2025, 2026 (NaN after the last FLEXGREEN/CAMS day) + coverage json. |
| `bf_test.sh` | Backfill: reference rows from the sources (world venv) -> GlobalStore/GhapStore unit test (container) -> coverage json. |
| `env_check_job.sh` | Checks the cdffm environment on 2 GCDs of a dev-g node (<= 0.33 GPU-h; random data, writes nothing but this log): 1 |
| `flexgreen_check.sh` | flexgreen_pm.py lead 1 on two 2021 start days (doy 100, 364) vs the Foehn green/era5 store (bit equality), timing. |
| `gfas_injh.sh` | The fires input of the 1 km model used in real time: GFAS injh 2017-2024 (ADS, monthly) -> coarse/fires_injh_{2017_2022,2023_2024}.npy -> checks. |
| `global_bcsd.sh` | B2 BCSD baseline (PROTOCOL.md 3), fitted on the 1,270 split-A training days only -> lghap_clim/bcsd_A.npz, + check. |
| `global_cellmeans.sh` | hcf z cell-mean table of the global store (training + validation days only), then the direct-recompute check. |
| `global_coarse.sh` | Global coarse inputs (PLAN.md 2.7 items 1-3): met/fires 2023-2024, CAMS 0.25 deg 2017-2024, non-regression, days.npz. |
| `global_era5B.sh` | ERA5 3-D FLEXGREEN fields from ARCO (PLAN.md 2.7 item 4): task 0 = full-month validation of 2022-12 against WB2, tasks 1..25 = months 2023-01 .. 2025-01 into data/processed/global/wb2/. |
| `global_flexgreen.sh` | FLEXGREEN ERA5 lead 1, start days 2022-01-01 .. 2024-12-30 (PLAN.md 2.7 item 5) -> data/processed/global/flexgreen/era5/ 8 ranks share the days of each year (flexgreen.py convention); done markers are folded into done_<y>.txt at t … |
| `global_injh_check.sh` | GlobalStore(fires='injh') opens, channels 0-1 bit-identical to mami, channel 2 = injh, a real training batch is finite. |
| `global_manifest.sh` | manifest (no hash) -> CPU loader test (GlobalStore + unchanged data_aq.BatchStream) -> manifest with SHA-256. |
| `global_prep_tests.sh` | Non-regression after the --store global wiring (CPU, 0 GPU-h): test_x all; test_s0 shapes liif dateline train ddp store (full); test_s0 noise --quick (the full noise check rewrites data/processed/smoke/noise_fit_smoke.json, which  … |
| `global_t6.sh` | T6 transport channels per target year (PLAN.md 2.7 item 6) |
| `global_train_tests.sh` | CPU checks of HCF on the global store (global-train; 0 GPU-h): scripts/test_global_hcf.py [checks...] sbatch jobs/global_train_tests.sh [shapes resfree dateline injh] |
| `o1k_calib_blh.sh` | RT1KM: calibration of the BLH substitute (encoder channels 31-33) on ERA5 training-period days -> realtime_1km/blh_calib.json |

### `runs_ghap/`
| File | What it does (first line of its own header) |
|---|---|
| `bcsd.sh` | PROTOCOL.md A2(d): B2 BCSD fitted on the GHAP training days 2017-2020 (runs_ghap/fit_bcsd_ghap.py fit + check) |
| `build_finish.sh` | PROTOCOL.md A2: GHAP store, step 'finish' of scripts/build_ghap.py |
| `build_ice.sh` | PROTOCOL.md A2: GHAP store, step 'ice' of scripts/build_ghap.py |
| `build_merge.sh` | PROTOCOL.md A2: GHAP store, step 'merge' of scripts/build_ghap.py |
| `build_points.sh` | PROTOCOL.md A2, pass 2 of the GHAP store: training points 2017-2020, 2021 hold-out full fields, cell means; one band per task |
| `build_scan.sh` | PROTOCOL.md A2, pass 1 of the GHAP store: validity scan 2017-2021 + training-day harmonic sums, one latitude band per task |
| `fit_bcsd_ghap.py` | GHAP version (PROTOCOL.md A2(d); paths and split G = 2017-2020; NaN cell means = cells without a valid pixel that day, ignored) of the B2 BCSD baseline of PROTOCOL.md section 3, fitted on the 1,270 split-A training days ONLY (2019 … |
| `p1r_G.sh` | PROTOCOL.md A2(c): selection, prior refit, SEL evaluation, gates and freeze of the GHAP system (as runs_global/p1r_K0.sh + p1r_final23.sh, A1(c'''), A1(e), A1(g)): 1 |
| `plumb_test.sh` | Code test (no truth read, no 2022 truth, no station file): the 2022 prediction script and the RT 1 km GHAP switch with the smoke checkpoint of pmG_base (step 2,000) -> runs_ghap/plumb/ (deleted after the check) |
| `pmG_cal/code/calib/__init__.py` | calib -- PROTOCOL.md Amendment A6: CRPS calibration of the GHAP downscaler ensemble (pmG_final), station-free |
| `pmG_cal/code/calib/compare_cal.py` | calib.compare_cal -- before/after tables of PROTOCOL.md A6(g)-(i) from eval_x outputs (CPU, numpy only) |
| `pmG_cal/code/calib/core.py` | calib.core -- shared pieces of PROTOCOL.md A6 (see the package docstring) |
| `pmG_cal/code/calib/diag_cal.py` | calib.diag_cal -- scoring of ONE fixed model (the frozen system or a fine-tuned checkpoint) with the inference sampler on the A6 validation sets (calib.core.validate_set), without any training |
| `pmG_cal/code/calib/eval_cal.py` | calib.eval_cal -- the existing selection evaluation, cdffm_aq_x.eval_x, run UNMODIFIED on an A6 system |
| `pmG_cal/code/calib/gates_cal.py` | calib.gates_cal -- reads the VALIDATION_GATES.json that runs_global/validation_gates.py wrote for an A6 evaluation and applies the pre-stated rules of PROTOCOL.md A6 (CPU, no torch): screen E1 E2 stage 2 (A6(f)): E1 = eval_x at ev … |
| `pmG_cal/code/calib/select_cal.py` | calib.select_cal -- stage 1 of PROTOCOL.md A6(f), CPU only (no torch): reads the V100k validation of every saved checkpoint of every arm (runs/<arm>/metrics.jsonl), the frozen system's V100k reference (results/d0_frozen.json) and, … |
| `pmG_cal/code/calib/test_cal.py` | calib.test_cal -- CPU plumbing checks of the A6 code on the synthetic ToyStore (no project data, no score): 1 |
| `pmG_cal/code/calib/train_cal.py` | calib.train_cal -- PROTOCOL.md A6(b): CRPS fine-tuning of the GHAP downscaler, as the calibration fine-tuning of the 1 km forecast model of the paper (global1km/fullflow_global_green.py --crps-steps / --crps-fm) |
| `pmG_cal/code/job_cache.sh` | PROTOCOL.md A6(c): the validation sets (truth values of 2021 hold-out days and of training-day points; no model). |
| `pmG_cal/code/job_d0.sh` | PROTOCOL.md A6(c) reference, A6(d) diagnostics and A6(e) temperature fit/grid, all on the FROZEN system (no training). |
| `pmG_cal/code/job_diag.sh` | one or several calib.diag_cal tasks on one node; ARGS = ';'-separated argument lists, e.g |
| `pmG_cal/code/job_full.sh` | PROTOCOL.md A6(f) stage 3: the full selection evaluation of A2(c) (eval_x, all mask pixels x 24 SEL days, 8 members x 10 Euler, P3, guard T3, regime-box spectra) + validation_gates.py |
| `pmG_cal/code/job_head.sh` | sourced by the A6 GPU jobs: paths, environment, run helpers (one srun task per GCD, as runs_ghap/train_G.sh / p1r_G.sh) |
| `pmG_cal/code/job_screen.sh` | PROTOCOL.md A6(f) stage 2: eval_x at every 8th mask pixel (24 SEL days) + the regime-box spectra, then the gates |
| `pmG_cal/code/job_smoke.sh` | PROTOCOL.md A6(j) smoke test: time per step and memory of a velocity-net arm and of the all-weights arm (30 steps each, weights discarded, nothing scored), the validation on the frozen weights (8 SEL days), the eval_x wrapper on 8 … |
| `pmG_cal/code/job_train.sh` | PROTOCOL.md A6(b): one fine-tuning arm |
| `pmG_cal/replay_code/assemble_cal.py` | A6(b), CPU: realtime_1km/replay/points_cal/<d>.npz = the downscaled members of the CALIBRATED sampler at the monitors, in the format of replay/points/<d>.npz (replay/README.md) for the arrays of the downscaler: id, lat, l … |
| `pmG_cal/replay_code/compare_points.py` | A6(b), proof: the copy infer_points_cal.py run with the frozen settings (tau 1) against the existing replay files of the frozen system: replay/km/<d>.npz (downscaler part), replay/raw/<d>.npy (members at every evaluated p … |
| `pmG_cal/replay_code/infer_points_cal.py` | COPY of realtime_1km/replay/code/infer_points.py for PROTOCOL.md A6 (runs_ghap/pmG_cal/replay_code/infer_points_cal.py; the original is not edited) |
| `pmG_cal/replay_code/points_cal_job.sh` | PROTOCOL.md A6(b): the reforecasts in real-time mode at the monitor pixels with the calibrated sampler |
| `pmG_cal/replay_code/points_rank.sh` | One srun task per GCD, INSIDE the container (started by points_cal_job.sh): the issue days of this rank, one process per day as realtime_1km/replay/code/km_rank.py does (children see one rank and device 0 of the visible set) |
| `pmG_seed1/a8/a8_lib.sh` | a8_lib.sh -- sourced by the A8 job scripts (host side of a Slurm job on ONE LUMI-G node, 8 GCDs) |
| `pmG_seed1/a8/compare_seeds.py` | A8(f)-(g): seeds 0 (frozen system, files read as they are), 1 and 2 side by side on the 24 selection days of 2021 |
| `pmG_seed1/a8/compare_smoke.py` | A8(e)-1: the 200-step seed-0 smoke of the A8 launcher against the first lines of the frozen training log |
| `pmG_seed1/a8/post_job.sh` | protocol_amendments/A8_seed_replicates.md (c): selection, prior refit, SEL evaluation and gates of the seed replicates: the five steps of runs_ghap/p1r_G.sh for each run of $RUNS, on ONE node, interleaved so that the GCDs are not  … |
| `pmG_seed1/a8/psdrepro.py` | A8(e)-2(ii): spectra of the FROZEN system recomputed with today's code, on the box pixels only |
| `pmG_seed1/a8/psdrepro_compare.py` | A8(e)-2(ii): merge the per-rank files of psdrepro.py and compare with the stored spectra of the frozen evaluation (pmG_base/eval_sel2021_M8_s10_P3_guard_T3/psd.npz) |
| `pmG_seed1/a8/rank_latents.sh` | A8: one srun task (inside the container) = the latents of this rank's days, exactly the command of runs_ghap/p1r_G.sh step 2 (same day list per rank, same seeds, same --n-per-radius), for one split or for both splits at the same t … |
| `pmG_seed1/a8/rank_ref.sh` | A8(e)-2: one srun task (inside the container): this rank's spectra day of the FROZEN system (psdrepro.py), and on rank 0 also the in-training validation of the frozen step 22,000 (valrepro.py) |
| `pmG_seed1/a8/refday_compare.py` | A8 reference check (added to (e)-2, see STATUS.md): full selection days of the FROZEN system re-evaluated today with the unmodified evaluation script (cdffm_aq_x.eval_x, same options, one day per process, all 147,289,344 mask pixe … |
| `pmG_seed1/a8/refday_eval.py` | A8 reference check: one or more full selection days of the FROZEN system re-evaluated with the UNMODIFIED evaluation script (cdffm_aq_x.eval_x.main, same options as the frozen evaluation), plus the location of the member points th … |
| `pmG_seed1/a8/refday_hits_analysis.py` | A8: where the member points at the state bound are, for a reference day of the FROZEN system (a8/results/refday_hits_day<t>.npz written by refday_eval.py: latitude, longitude, member, starting noise z0, final z) |
| `pmG_seed1/a8/smoke_job.sh` | protocol_amendments/A8_seed_replicates.md (e): the checks before the full runs, one node, about 12 min |
| `pmG_seed1/a8/train_job.sh` | protocol_amendments/A8_seed_replicates.md (c): the training of the frozen system (runs_ghap/train_G.sh, pmG_base) with another seed and nothing else changed, in the frozen run's two segments (0 -> 2,000 as job 23246364, then --res … |
| `pmG_seed1/a8/valrepro.py` | A8(e)-2(i): the in-training validation of a FROZEN checkpoint of pmG_base recomputed with today's code |
| `test_store.sh` | PROTOCOL.md A2: unit test of GhapStore + LGHAP store non-regression (runs_ghap/test_store_ghap.py) |
| `test_store_ghap.py` | Unit test of cdffm_aq.store_ghap.GhapStore (PROTOCOL.md A2) and non-regression of the LGHAP store paths |
| `train_G.sh` | PROTOCOL.md A2(c): pmG_base = the frozen K0 recipe (pmA_baseK0 flags, --sigma-k 0) on the GHAP store (--store ghap, split G: train 2017-2020, SEL = 24 days of 2021) |

### Tests on 2022
| File | What it does (first line of its own header) |
|---|---|
| `fig_day_ghap.py` | One day of the GHAP 2022 pass (style of final2024/fig_day.py): world maps (Robinson) of (A) monitors, (B) GHAP V1, (C) CAMS 0.4 deg day 0, (D) CD-FFM 1 km ensemble mean (GHAP system; outside its mask = permanent ice: CAMS, lighter … |
| `fig_world_ghap.py` | World maps (Robinson) of the GHAP 2022 single pass (PROTOCOL.md A2(d)), in the style of final2024/fig_world_results.py (PM2.5 classes, white ocean, land without data #F0F0F0, bold letters, RMSE / R / bias boxes, no titles) |
| `predict_2022.py` | PROTOCOL.md A2(d), GHAP 2022 single pass, step (a): the 2022 predictions of the frozen GHAP system (runs_ghap/pmG_final), written by a MODEL process under the GhapStore firewall (no raw GHAP file, no data/processed/ghap/sealed fil … |
| `predict_2022.sh` | PROTOCOL.md A2(d) step (a): test2022_ghap/predict_2022.py with the frozen runs_ghap/pmG_final (model process under the GhapStore firewall; no truth is read) |
| `score.sh` | PROTOCOL.md A2(d) step (b): GHAP 2022 truth (logged opening), GHAP-subsample scores, station scores, figures |
| `score_2022.py` | PROTOCOL.md A2(d), GHAP 2022 single pass, step (b): the SCORING process (world venv, numpy only; no model code, no GhapStore) |
| `score_st_2022.py` | GHAP 2022 copy (PROTOCOL.md A2(d): truth test2022_ghap/st_truth_2022.npz, GHAP mask, region North > 60 N added, wording: GHAP is fitted to ~9,500 stations 2017-2022, monitors NOT independent) of: Station scores (PROTOCOL A1(h), se … |
| `stations.sh` | PROTOCOL.md A2(d), after the freeze: SYSTEM.json of the 2022 pass + 2022 station positions/truth (logged opening) |
| `stations_2022.py` | Station truth 2022 for the GHAP system (PROTOCOL.md A2(d); evaluation only; opened after the freeze, logged) |

| File | What it does (first line of its own header) |
|---|---|
| `code/check3_split.py` | A7(g) check 3 as amended on 2026-11-19 (protocol note, "Deviations"): the reference of the same-day members is (i) the frozen test's stored members (test2022_ghap/preds_st/cdffm) on the 348 days computed by its 31 sound processes, … |
| `code/check_preds.py` | A7(g) checks 3 and 5 on a prediction directory (preds_smoke/ or preds/) |
| `code/check_recheck.py` | A7(g) check 3, confirmation: preds/ (full run, nid005282) against preds_recheck/ (second computation of 24 days on another node, job 23865411) and both against the frozen test's stored members, per day |
| `code/diag_check3.py` | A7(g) check 3, diagnosis of the failure on the full run (no observation, no GHAP value: predictions only) |
| `code/diag_check3b.py` | A7(g) check 3, diagnosis (2): per-day share of bit-identical same-day values against the frozen test, per member, and the same against preds_smoke where available |
| `code/diag_check3c.py` | A7(g) check 3, diagnosis (3): spatial structure of the difference to the frozen test on the anomalous member-days |
| `code/diag_obs_target.py` | A7, description of the observations and of the target at the official stations (scoring side; no model score is changed, nothing is selected) |
| `code/diag_season.py` | A7, descriptive addition of 2026-11-19 (protocol note, "Deviations"; made after the tables were read): the same-day table S split by season and by month, to see WHEN the downscaler loses to its coarse input in Korea and Japan |
| `code/init_preds.py` | A7: create the prediction arrays of predict_official.py (numpy only; run once before the srun; existing files with the right shape are kept, so a resubmission continues) |
| `code/make_report.py` | A7: REPORT.md from the result files (no data read other than the json files of results/ and screened/results/) |
| `code/predict.sh` | Amendment A7, model process (frozen runs_ghap/pmG_final under the GhapStore firewall; no truth is read) |
| `code/predict_official.py` | Amendment A7 (protocol_amendments/A7_official_stations_2022.md), steps (c) and (d): the MODEL process |
| `code/predict_recheck.sh` | Amendment A7, diagnosis of the failed check 3 (2026-11-19): the same-day members of the 17 days that ONE process of the frozen test's job 23246562 computed (its rank 30) are recomputed a second time, on another node, with the unch … |
| `code/prep.sh` | A7 step 1 (CPU, world venv): station positions and rings; CAMS analyses and CAMS forecast fields on the model grids. |
| `code/prep_inputs.py` | A7(d): coarse inputs of the forecast-mode arms that are not in the frozen stores, extracted once on CPU (world venv) so that the model process opens only .npy files (no observation, no GHAP) |
| `code/prep_positions.py` | A7(b): the official stations of the paper on the downscaler's lattice (positions only for the model process) |
| `code/report_tables.py` | A7: markdown tables from results/scores_official_2022.json (written by score_official.py) -> results/tables_official_2022.md |
| `code/score.sh` | Amendment A7, scoring side (CPU, world venv; no model code) |
| `code/score_official.py` | Amendment A7 (protocol_amendments/A7_official_stations_2022.md), steps (e)-(g): the SCORING process (world venv, numpy + zarr; no model code) |
| `code/score_screened.py` | A7 sensitivity (v) (protocol note, "Deviations", 2026-11-19 17:40): the scoring of score_official.py with the implausible EEA values removed (EEA station-day set to missing when its value is <= 0 or > 300 ug m-3) |
| `code/score_screened.sh` | Amendment A7, sensitivity (v): the scoring with the implausible EEA values removed (code/score_screened.py). |
| `code/truth_official.py` | A7(e), scoring side: GHAP V1 2022 at the station pixels and on the rings of the official stations (reference product, not a model) |

| File | What it does (first line of its own header) |
|---|---|
| `mech_setup.py` | Rehearsal of the calibrated 2022 pass on SELECTION-YEAR data only (2021; no value of 2022, no station file) |
| `predict_cal.py` | 2022 protocol of the frozen downscaler (PROTOCOL.md A2(d)) with the CALIBRATED sampler of runs_ghap/pmG_cal/final (PROTOCOL.md A6 + draft amendment protocol_amendments/A9_calibrated_2022.md) |
| `predict_cal.sh` | The 2022 protocol of the frozen system (test2022_ghap/predict_2022.sh, PROTOCOL.md A2(d) step (a)) with the calibrated sampler of runs_ghap/pmG_cal/final: test2022_cal/predict_cal.py = test2022_ghap/predict_2022.py unmodified + th … |
| `score_cal.py` | Scoring of the 2022 pass of the CALIBRATED downscaler (runs_ghap/pmG_cal/final; PROTOCOL.md A6 + draft amendment protocol_amendments/A9_calibrated_2022.md) with the scorers of the frozen system's 2022 test, UNMODIFIED: test2022_gh … |
| `score_cal.sh` | Scoring of the calibrated system's 2022 pass with the frozen system's scorers (test2022_ghap/score.sh: steps score, split, stscore; "truth" and "pinning" are not repeated: the extracted GHAP 2022 truth and the pinning do not depen … |
| `submit_2022_cal.sh` | THE SINGLE 2022 PASS OF THE CALIBRATED DOWNSCALER (runs_ghap/pmG_cal/final) |

### `realtime_1km/`
| File | What it does (first line of its own header) |
|---|---|
| `blh.py` | realtime_1km.blh -- boundary-layer-height substitute (encoder channels 31-33) from 1 deg profiles (see calib_blh.py) |
| `calib_blh.py` | BLH substitute for the real-time encoder channels 31-33 (log ERA5 boundary-layer height: daily mean, max, min) |
| `channels.py` | RT 1 km, step 2 (container, GPUs): complete the 47 encoder channels of each lead day of RT day d |
| `infer.py` | RT 1 km, step 3 (container, one process per GCD: srun -n 8): CD-FFM downscaling of each lead day of RT day d |
| `infer_only_job.sh` | infer.py + quicklook.py only, on an already prepared day (realtime_1km/work/<day>/ with chan_L1..5.npz): for several test checkpoints of the same day in parallel without re-running prep.py (whose IFS fetch dir is shared and remove … |
| `monitors_openaq.py` | RT 1 km monitors of the verification (evaluation only: the monitors are never an input of RT and are never used to train, choose or adjust it) |
| `prep.py` | RT 1 km, step 1 (CPU, world venv, internet): the real-time conditioning inputs of one RT forecast day d that do not need torch, for the 5 lead days L = 1..5 (valid day d + L - 1, forecast hours 24(L-1) .. 24L after 00 UTC of d) |
| `replay/cams3h/code/build_arms.py` | cams3h step 2: the CAMS forecast at the catalogue monitors with the daily mean of the paper, and the checks |
| `replay/cams3h/code/c3h_lib.py` | cams3h: shared definitions |
| `replay/cams3h/code/fetch3h.py` | cams3h step 1: the CAMS operational forecast of surface PM2.5 at EVERY 3-hourly step, 0 .. 120 h, for the 53 cycles of 00 UTC that the real-time scorer uses (issue days 2026-09-01..09-20 and 2026-10-10..11-11) |
| `replay/cams3h/code/fetch3h_job.sh` | cams3h step 1 on a CPU node: the 3-hourly CAMS forecast steps of the 53 cycles (see fetch3h.py) |
| `replay/cams3h/code/safe_io.py` | Safe writes for the cams3h work (the file system dropped a write on 2026-11-19): every file is written to a temporary name in the same directory, flushed, fsynced, read back and compared, then renamed with os.replace |
| `replay/code/assemble.py` | WP1a step D (CPU): the per-day deliverable replay/points/<d>.npz = the downscaled members of replay/km/<d>.npz plus the station-free baselines at the monitors, with the definitions of realtime_1km/rc/coupling_live.py (its function … |
| `replay/code/cams_points.py` | WP1a: the baseline 'cams' at the catalogue monitors, exactly as realtime_1km/rc/coupling_live.py (and live_baselines.py) define it: for issue day d and lead day L = 1..5 (valid day d + L - 1), the daily mean of the CAM … |
| `replay/code/camsfc2_job.sh` | WP1a, second version: ONE request for all the cycles that lack the ten leads of the CAMS forecast (camsfc_fetch2.py; a request already queued, running or done is re-attached, never sent twice) |
| `replay/code/camsfc_fetch.py` | WP1a (2026-11-19): the ten leads 12, 24, ..., 120 h of the CAMS operational PM2.5 forecast (00 UTC cycle) for the issue days whose realtime/data/<d>/cams_fc.npz holds only the leads 24 and 120 h (202 … |
| `replay/code/camsfc_fetch2.py` | WP1a (2026-11-19), second version of the fetch (camsfc_fetch.py asked one cycle per request): the ten leads 12, 24, ..., 120 h of the CAMS operational PM2.5 forecast (00 UTC cycle) for the issue days … |
| `replay/code/camsfc_job.sh` | WP1a: the ten leads of the CAMS operational forecast for the cycles that only have 24 h and 120 h, into replay/camsfc/<d>/ (one request after the other; the chain's own request and client) |
| `replay/code/camsfill_job.sh` | WP1a, after the CAMS forecast fetch (rp_camsfc): 'cams' at the monitors for every issue day that has the ten leads (camsfc/points/<d>.npz), then the safety net: fill 'cams' in the FINISHED points/<d>.npz files where a  … |
| `replay/code/channels_replay.py` | WP1a step B2: realtime_1km/channels.py for a past issue day, with its work directory redirected to replay/work |
| `replay/code/check_all.py` | WP1a final checks over all delivered days (CPU; no observation is read) -> replay/check/summary.json and a table |
| `replay/code/check_coarse.py` | WP1a check of step A: a replayed coarse forecast against the archived production forecast of the same day |
| `replay/code/check_coarse_archive.py` | WP1a check of step A on the days whose production forecast file was deleted (CPU; no monitor observation is read) |
| `replay/code/check_coarse_job.sh` | WP1a check of step A on one archived day: task 0 = the sampler alone on the ARCHIVED FLEXGREEN store (symlink), task 1 = FLEXGREEN recomputed + sampler |
| `replay/code/check_points.py` | WP1a check of step C: the pointwise members against the PUBLISHED full-field products of the same issue day |
| `replay/code/coarse_job.sh` | WP1a step A (replay of the 0.4 degree real-time forecast): FLEXGREEN + forecast, one GCD per day stream |
| `replay/code/coarse_replay.py` | WP1a step A: the 0.4 degree real-time forecast of past issue days, recomputed with the production chain |
| `replay/code/infer_points.py` | WP1a step C: realtime_1km/infer.py evaluated at a list of mask pixels (one process per GCD) |
| `replay/code/km_job.sh` | WP1a steps B1(pm) + B2 + C: one day stream per GCD (km_rank.py) |
| `replay/code/km_rank.py` | WP1a steps B1(pm) + B2 + C for the days of one rank (one GCD): for each issue day of the list taken by this rank (days[rank::world]) run, as separate processes and only when their outputs are missing, prep_replay.py <d> --only pm  … |
| `replay/code/make_index.py` | WP1a: replay/points/INDEX.json, one row per issue day (read from the files themselves): lead days made, which baselines are present per lead day, the source of the coarse forecast, the overlap with the fine-tuning targets of the c … |
| `replay/code/positions.py` | WP1a: the mask pixels at which the downscaler is evaluated for the replay (positions only: no observation is read) |
| `replay/code/prep_job.sh` | WP1a step B1 (CPU, internet): prep_replay.py for a list of issue days, 8 at a time |
| `replay/code/prep_replay.py` | WP1a step B1: realtime_1km/prep.py for a past issue day, with its paths redirected to the replay tree |
| `replay/score/build_pairs.py` | WP1b step 1: the table of station-days (evaluation only; nothing is fitted, selected or corrected here) |
| `replay/score/make_report.py` | WP1b step 3: REPORT.md = reading.md (written by hand after the scores) + methods.md + every table of tables/. python make_report.py --out S |
| `replay/score/obs_gap.sh` | Evaluation only (WP1b): fetch the OpenAQ archive days of the gap 2026-10-22..10-27 for the scoring of the reforecasts |
| `replay/score/refgrade.py` | WP1b, question 5: which catalogue monitors sit at a site of an official reference network? No file on disk carries the sensor type of the OpenAQ locations (coupling/build_st.py l.18) |
| `replay/score/run_all.sh` | WP1b: the whole scoring in ONE re-runnable command (evaluation only; nothing is fitted; nothing is published) |
| `replay/score/run_all_job.sh` | WP1b scoring on a CPU node: sbatch run_all_job.sh [POINTS_DIR] [OUT_DIR] [ARMS] [FOCUS] (see run_all.sh) |
| `replay/score/sc_lib.py` | WP1b scoring library: station-day table, block bootstrap, scores |
| `replay/score/score.py` | WP1b step 2: every score and table of the real-time-mode reforecasts at the monitors |
| `replay/score_cams3h/check_figure_script.py` | cams3h: does this directory work as a scorer directory for the figure of the real-time skill? The data script of the figure (paper/natcomms/figures_new/fig_rt_skill/rt_skill_data.py) writes beside itself, so it is NOT run in place … |
| `replay/score_cams3h/check_manual.py` | cams3h, self-check before reporting (evaluation only) |
| `replay/score_cams3h/lahore.py` | cams3h: the numbers of the figure 'a real-time forecast at 1 km' that involve the CAMS forecast, with its two daily means |
| `replay/score_cams3h/make_readme.py` | cams3h: README.md and reading.md of this directory, with every number taken from the JSON files written by the checks (side_by_side.json, lahore.json, check_manual.json, ../cams3h/build_report.json, ../cams3h/raw/*.json) |
| `replay/score_cams3h/side_by_side.py` | cams3h: the real-time scores against the CAMS forecast with its two daily means, side by side (evaluation only) |
