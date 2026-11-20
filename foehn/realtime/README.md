# Real-time forecast at 0.4°

The first stage of the real-time system of the paper (*Kilometre-scale forecasts in real time*; Methods, *Real-time
forecasting*, paragraph *Coarse forecast*): the network of the comparison in Aurora's protocol (`../foehn_cams.py`), fine-tuned
on the inputs available every morning in 2026 and run each day from the CAMS analysis of 00 UTC, the open-data weather
forecast of ECMWF and the fire emissions of the previous day, with eight members and five lead days.

Order of a daily run: `fetch.py` (inputs of the cycle) → `build.py` (fields on the model grid) → `flexgreen_batch.sh`
(computed transport) → `predict.py` (eight members, 12-hourly, 120 h). `finetune.py` is the fine-tuning on January–August 2026,
`hindcast.py` and `bulk.py` rebuild past cycles from the data of their day, `fetch_camsfc.py` fetches the CAMS forecast used as
reference, and `verify.py`, `compare.py` and `compare_maps.py` score the forecasts against the analyses.
The wrappers that ran the chain every day and drew its maps for a web page are not part of the release.

The scripts are the working files of the study, as they ran on the LUMI supercomputer: they carry the paths of our project space and the Slurm settings of that machine, like the rest of this repository. Data, trained weights, logs and outputs are not in the repository (see *Data availability* and *Code availability* in the paper).

| File | What it does (first line of its own header) |
|---|---|
| `build.py` | RT step 2: the inputs of start day d in the exact format of training, data/<yyyy-mm-dd>/: cams.npy (2, 13, 361, 900) float16 the CAMS analysis at t0 - 12 h (index 0) and t0 = 00 UTC of d (index 1), as two rows of cams_series.npy:  … |
| `bulk.py` | RT 1.2 data pipeline: the live inputs for a range of past days, built in the day directories (data/<date>/) that fetch.py/build.py/flexgreen use, with the CAMS analyses and the GFAS fires requested month by month (one ADS request  … |
| `common.py` | RT, the real-time chain of Foehn-CAMS (foehn_cams.py, the model of the Aurora protocol, 0.4 deg): shared paths, dates, journal and retries |
| `compare.py` | RT step 4a, the non-regression test: the forecast of a 2022 day by the real-time chain (inputs copied from the training archive by build.py --from-archive, Green's functions recomputed by flexgreen.py --rt-dir, forecast by predict … |
| `compare_maps.py` | RT step 4b: the ensemble mean of a real day at +24 h and +120 h next to the CAMS operational forecast of the same run (data/<date>/cams_fc.npz, fetch_camsfc.py), to spot a gross bias of the real-time inputs |
| `fetch.py` | RT step 1: download the inputs of start day d (forecast issued from the 00 UTC run of d) into data/<yyyy-mm-dd>/. cams_00.zip CAMS analysis 00 UTC of d } dataset cams-global-atmospheric-composition-forecasts, type forecast, cams_1 … |
| `fetch_camsfc.py` | The CAMS operational forecast of PM2.5 (00 UTC run of day d, the ten leads 12 .. 120 h; `--two` for 24 and 120 h only) for the checks and the daily verification |
| `finetune.py` | RT 1.2: fine-tuning of Foehn-CAMS (foehn_cams.py: same network, flow-matching loss, shifted schedule, optimiser) on day directories built from the live inputs (realtime/data/<date>/: cams.npy, hres_B.npy, fires.npy, green/), the w … |
| `finetune_job.sh` | RT 1.2 fine-tuning on one node (8 GCDs, DDP), the environment of cams_job.sh |
| `flexgreen_batch.sh` | FLEXGREEN for a range of built day directories, 8 days at a time on one node (one GCD per day, ~14 min each): sbatch flexgreen_batch.sh 2026-06-01 2026-07-05 Days already holding green/origins_<year>.npy are skipped; each rank tak … |
| `hindcast.py` | RT step 4c, the hindcast that validates the real-time inputs of 2026 (GFAS v1.4.2 with injh, ECMWF open data, CAMS of 2026) before any publication: forecasts for a window of past start days, with and without fires, scored against  … |
| `pipeline.sh` | RT 1.2 data pipeline for a date range: waits for the downloads (bulk.py cams/gfas/ifs) of every day of the range, builds the day directories (bulk.py build), runs FLEXGREEN in batches of one node (flexgreen_batch.sh), and reports |
| `pipeline2.sh` | RT 1.2 data pipeline for a date range: waits for the downloads (bulk.py cams/gfas/ifs) of every day of the range, builds the day directories (bulk.py build), runs FLEXGREEN in batches of one node (flexgreen_batch.sh), and reports |
| `predict.py` | RT step 3b: the forecast of start day d from the inputs of data/<yyyy-mm-dd>/ (build.py) and its Green's functions (flexgreen.py --rt-dir), with the model, checkpoint, sampler and seeds of the paper's foehn_cams.py predict() |
| `validate_job.sh` | Validation of a fine-tuned checkpoint on the September 2026 day directories: predict.py on every day (8 days at a time, one GCD each) -> out/<d>/pred_<tag>.npy, then hindcast.py score with the tag |
| `verify.py` | RT daily verification (public): the forecasts issued in the last five days are scored against the CAMS analyses that have arrived since, taken from the day directories (data/<d>/cams.npy holds the analyses of 00 UTC of d and 12 UT … |
