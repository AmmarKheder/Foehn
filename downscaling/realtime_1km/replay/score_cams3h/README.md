# Real-time scores against the CAMS forecast with the daily mean of the paper's rule (2026-11-19)

One job: the arm `cams` of the real-time scorer read the CAMS forecast at three instants of each day; the paper's rule is the
mean of its 3-hourly steps. The steps were fetched for the 53 cycles, the arm was rebuilt at the monitors, and the UNMODIFIED
scorer (`../score/run_all.sh`, same station-days, same two-way block bootstrap, same seed) was rerun into this directory.
**In this directory the arm `cams` IS the 3-hourly daily mean**, so the tables, `numbers.json` and `REPORT.md` have the layout
of `../score/` and can replace it for any script that takes a scorer directory. Monitors are evaluation only: nothing was
fitted, selected or tuned. Nothing outside `../cams3h/` and this directory was written; no existing file was changed.

## 1. The two definitions (written before anything was changed)

Day: lead day L = 1..5 of issue day d (00 UTC cycle) is the UTC day d + L - 1, forecast hours 24(L-1) .. 24L.

**Existing arm `cams`** (`../code/assemble.py` l.86-146 and `../code/cams_points.py`, lines of `realtime_1km/rc/coupling_live.py`):
files `realtime/data/<d>/cams_fc.npz` (cycles from 2026-10-11) or `../camsfc/<d>/cams_fc.npz` (1-20 September, 10 October):
the ten leads 12, 24, ..., 120 h, float16, 451 x 900 at 0.4 degree (90 N .. 90 S, 0 .. 359.6 E, rolled to 180 W); the 0 h
instant is the analysis `realtime/data/<d>/cams.npy[1, 1]` (84 N .. 60 S). Daily mean of lead day L =
max(1/4 x(24(L-1)) + 1/2 x(24L-12) + 1/4 x(24L), 0) in float32: **three instants, 00, 12 and 24 UTC**. Reading:
bilinear in linear space on the 0.4 degree grid, `coupling_live.bilinear(F, 90, 0.4, -180, 0.4, lat, lon)`.

**The paper's rule for 2022** (Methods: "The daily mean of the CAMS forecast is the mean of its 3-hourly steps over the target
day"). Code: `eccv/world/data_pipeline/dl_cams_forecast.py`, `to_daily_leads`: steps 0, 3, ..., 117 h of the 00 UTC run, values
float32 x 1e9, lead day = mean of the steps with 24 L <= hour < 24 (L + 1): **the eight steps 00, 03, ..., 21 UTC of the day,
equal weights; the step of 24 UTC is not in the day** (neither 03 .. 24 nor 00 .. 24 with half weights). Stored in
`eccv/data/world/cams_forecast_daily/2022.zarr` and read by `experiments/figures/compare_fair.py` (l.71, 131-132:
`cams[d + 1, 0, h - 1]`, bilinear on the grid 90 N, 0 E, 0.4 degree, linear space).

**New arms** (`../cams3h/points/<d>.npz`, (5 lead days, 12,136 monitors), catalogue order; code `../cams3h/code/`):
- `cams3h` = the eight steps 24(L-1) + 0, 3, ..., 21 h, equal weights; steps in the float16 values that the chain stores,
  floor at 0, the SAME bilinear function as the existing arm. Only the time sampling differs from `cams`.
- `cams3i` = the three instants recomputed from the new files with the lines of the existing arm.
- extras: `cams3h_f32` (float32 steps), `cams3h_trap` (nine steps 00 .. 24 UTC, half weights on the ends), `cams3h_0324`
  (eight steps 03 .. 24 UTC).
The 0.4 degree input of the 1 km forecast (`rt04`) keeps its three instants: its members exist every 12 h only.

## 2. Data: the download succeeded for all 53 cycles

Dataset `cams-global-atmospheric-composition-forecasts` of the Atmosphere Data Store, type forecast, 00 UTC, variable
`particulate_matter_2.5um`, lead hours 0, 3, ..., 120 (the chain's request, `realtime/fetch_camsfc.py`, with 41 lead hours and
date ranges), through the chain's own API client (`realtime/fetch.py`, `ads()`: the only code that reads the credentials;
nothing here prints, copies or logs them). Five requests (job 23870417, sent 20:36-20:37 UTC), each answered within 2 to
3.5 minutes; files in `../cams3h/raw/` (2.16 GB, SHA-256 in the `.json` beside each, re-read and identical):

| file | cycles | fields | MB | request finished (UTC) |
|---|---|---|---|---|
| 2026-09a.nc | 2026-09-01/2026-09-10 | 410 | 406.7 | 2026-11-19 20:38:55 |
| 2026-09b.nc | 2026-09-11/2026-09-20 | 410 | 403.8 | 2026-11-19 20:38:55 |
| 2026-10a.nc | 2026-10-10/2026-10-20 | 451 | 441.0 | 2026-11-19 20:39:49 |
| 2026-10b.nc | 2026-10-21/2026-10-31 | 451 | 447.0 | 2026-11-19 20:38:55 |
| 2026-11.nc | 2026-11-01/2026-11-11 | 451 | 461.5 | 2026-11-19 20:40:34 |

The first job was cancelled after 3 minutes because of an error of mine in `fetch3h.py` (it expected a zip; the service
answers a one-variable request with a bare netCDF file); job 23870445 re-attached to the five requests already sent and
downloaded them. No request was sent twice and none is left in the queue.

Already on disk and used only as an independent check: `cams_forecast_daily/2026.zarr`, daily means written by the paper's own
builder from its own download, cycles to 2026-11-03 (45 of the 53; it holds no steps, and not the cycles of 4-11 November).

## 3. Result in one screen

Primary set unchanged: **1,819,647 station-days**, 48 issue days (6-20 September, 10 October - 11 November 2026), lead days 1-5,
11,121 monitors, 846 2-degree cells, 8 week blocks; station-days lost: 0 (the new arm is finite wherever the old one is; all 53
issue days: 2,042,972). RMSE of the 1 km forecast 16.48 ug m-3 in both columns.

| quantity (RMSE change in %, 95 % interval, unless stated) | old: three instants (`score/`) | new: 3-hourly mean (this directory) |
|---|---|---|
| RMSE of the CAMS forecast, pooled (ug m-3) | 19.85 [13.20, 27.67] | 19.24 [12.86, 26.94] |
| three instants against 3-hourly mean, RMSE of CAMS |  | ratio 1.0314: +3.1 [+1.9, +4.6] % pooled, +2.9 [+0.5, +5.1] % mean over regions |
| **1 km vs CAMS, all monitors pooled** | **-17.0 [-23.1, -10.0]** | **-14.4 [-20.7, -7.2]** |
| **1 km vs CAMS, mean over the 8 regions** | **-14.3 [-19.0, -5.8]** | **-12.4 [-17.0, -4.7]** |
| lead day 1, pooled | -14.5 [-19.8, -7.9] | -11.9 [-17.4, -5.3] |
| lead day 2, pooled | -16.4 [-24.3, -7.8] | -13.7 [-22.1, -5.0] |
| lead day 3, pooled | -16.1 [-22.5, -8.7] | -13.6 [-20.1, -6.0] |
| lead day 4, pooled | -19.4 [-25.7, -12.5] | -17.0 [-23.5, -9.6] |
| lead day 5, pooled | -17.9 [-23.8, -10.8] | -15.1 [-21.2, -7.9] |
| Europe | +0.6 [-5.5, +6.3] | +2.3 [-4.1, +8.3] |
| East Asia (Korea, Japan west of 135 E, Taiwan) | -41.8 [-52.9, -22.3] | -32.5 [-45.3, -16.6] |
| South Asia | -17.3 [-25.1, -7.0] | -15.0 [-23.1, -4.1] |
| North America | -12.9 [-24.3, -5.1] | -11.9 [-22.8, -4.6] |
| Africa | -27.5 [-49.7, -11.1] | -29.0 [-52.1, -10.6] |
| South America | -9.5 [-20.9, +8.1] | -9.4 [-21.5, +8.0] |
| North of 60 N | +12.4 [-10.3, +56.4] | +9.5 [-12.4, +50.0] |
| Other | -18.1 [-34.0, +3.8] | -13.1 [-24.8, +7.0] |
| issue days 6-20 September | -15.7 [-24.0, -6.7] | -14.1 [-21.2, -4.7] |
| issue days 10 October - 2 November | -15.2 [-21.3, -9.0] | -12.2 [-18.5, -5.8] |
| issue days 3-11 November (live) | -19.1 [-27.1, -8.9] | -16.6 [-25.2, -6.0] |
| 53 issue days (1-5 September included) | -17.0 [-22.8, -10.1] | -14.4 [-20.5, -7.5] |
| without the valid days whose observations are not final | -16.5 [-22.4, -9.5] | -14.1 [-20.2, -7.1] |
| read 10 km from the monitor (rings sample, 1,658,975 station-days) | -14.6 [-20.2, -8.1] | -11.9 [-17.8, -5.2] |
| same sample, at the monitor's pixel | -17.3 [-23.1, -10.5] | -14.7 [-21.0, -7.7] |
| United States, monitors at official sites (proxy) | -32.8 [-44.8, -20.4] | -29.5 [-43.2, -17.5] |
| Europe, monitors at official sites (proxy) | -6.4 [-14.6, +2.2] | -6.1 [-14.2, +2.6] |
| issue days with the lower pooled RMSE | 48 of 48 (range -21.8 .. -9.2, median -14.8) | 48 of 48 (range -19.5 .. -6.9, median -12.2) |
| issue days whose own interval (2-degree cells) excludes zero | 48 of 48 | 46 of 48 |
| 2-degree cells with the lower RMSE (of 768 with at least 100 station-days), % | 66.8 [60.6, 69.5] (513 cells) | 64.3 [59.2, 67.7] (494 cells) |
| monitors with the lower RMSE (of 10,738 with at least 30 station-days), % | 67.8 [60.5, 71.2] | 66.2 [59.7, 69.8] |
| 0.4 degree input vs CAMS, pooled | -7.3 [-10.4, -3.5] | -4.4 [-7.7, -0.3] |
| 0.4 degree input vs CAMS, mean over regions | -6.1 [-10.1, -2.6] | -3.5 [-8.3, +0.8] |
| fair CRPS of the 1 km members (6.15) vs absolute error of CAMS (9.32 / 8.93), pooled | -34.0 [-38.6, -29.2] | -31.2 [-35.5, -26.6] |
| the same, mean over regions | -26.8 [-31.3, -20.0] | -25.3 [-29.4, -18.8] |
| per-monitor temporal correlation of CAMS (median over series) | 0.60 [0.55, 0.64] | 0.62 [0.57, 0.67] |
| 1 km minus CAMS, temporal correlation (median of paired differences) | -0.027 [-0.059, +0.027] | -0.041 [-0.078, +0.012] |
| amplitude ratio of CAMS, sd(forecast) / sd(observation) | 1.09 [0.94, 1.25] | 1.10 [0.97, 1.25] |
| SEDI of CAMS, above 37.5 ug m-3 (1 km: 0.79) | 0.69 [0.55, 0.79]; 1 km minus CAMS +0.10 [+0.02, +0.20] | 0.72 [0.60, 0.81]; 1 km minus CAMS +0.07 [-0.01, +0.15] |
| SEDI of CAMS, above 75 ug m-3 (1 km: 0.76) | 0.66 [0.44, 0.78]; 1 km minus CAMS +0.09 [+0.01, +0.17] | 0.69 [0.48, 0.80]; 1 km minus CAMS +0.06 [-0.02, +0.13] |
| SEDI of CAMS, monitor's own 95th percentile, same threshold (1 km: 0.23) | 0.40 [0.30, 0.54]; 1 km minus CAMS -0.17 [-0.31, -0.07] | 0.42 [0.32, 0.55]; 1 km minus CAMS -0.19 [-0.32, -0.09] |
| SEDI of CAMS, own 95th percentile of each series (1 km: 0.44) | 0.50 [0.44, 0.54]; 1 km minus CAMS -0.06 [-0.12, -0.00] | 0.51 [0.45, 0.57]; 1 km minus CAMS -0.07 [-0.14, -0.01] |
| SEDI of CAMS, monitor's own 99th percentile, same threshold (1 km: 0.21) | 0.40 [0.29, 0.55]; 1 km minus CAMS -0.19 [-0.33, -0.08] | 0.42 [0.31, 0.56]; 1 km minus CAMS -0.21 [-0.35, -0.10] |
| SEDI of CAMS, own 99th percentile of each series (1 km: 0.37) | 0.43 [0.36, 0.47]; 1 km minus CAMS -0.06 [-0.11, -0.01] | 0.45 [0.38, 0.51]; 1 km minus CAMS -0.08 [-0.14, -0.02] |
| Lahore, 11 November, CAMS at the 55 city monitors (measured 265.5, 1 km 202.4) | 142.4 | 149.1 |
| the 221 monitors of the plain, CAMS RMSE / bias / r between monitors | 91.0 / -54.0 / 0.49 | 88.4 / -51.1 / 0.52 |
| the plain, 1 km vs CAMS, RMSE change (monitors resampled; 0.4 degree cells resampled) | -18.5 [-24.3, -12.9]; -18.5 [-31.8, +1.0] | -16.1 [-22.0, -10.4]; -16.1 [-29.2, +2.6] |
| the plain, city-minus-outside contrast of CAMS (measured 80.7, 1 km 75.5) | 44.7 | 45.5 |
| Lahore / Delhi, mean of CAMS over the lead-day-1 days of 3-11 November | 99.5 / 115.2 | 107.0 / 120.0 |
| Karachi, 10 November, CAMS at the 20 monitors (measured 38.7, 1 km 40.4) | 14.9 | 18.4 |
| Karachi, largest CAMS cell over the city box; residents above 37.5 at 1 km in CAMS cells at or below it (million) | 16.2; 14.417 | 21.0; 14.417 |

Regions with a lower RMSE than CAMS: 6 of 8 (old), 6 of 8 (new); interval entirely below zero in 4 and 4
(the same regions: East Asia, South Asia, North America, Africa). Title rule (T12): unchanged (not met in the strict
reading, met in the lenient one). Full tables: `side_by_side.md` (both readings), `tables/` and `REPORT.md` (new
reading, every table of the scorer), `lahore.md`.

**What the change is.** Read at three instants, the CAMS forecast had an RMSE 3.1 % [1.9, 4.6] higher than with its eight steps
(the review notes give about 4 % at the 2022 stations). The handicap is largest in East Asia (+16.0 [+2.7, +26.2] %) and "Other"
(+6.2 [+2.0, +14.9] %); it is +1.7 [-0.2, +3.0] % in Europe, +2.8 [+1.7, +3.9] % in South Asia, +1.2 [-0.9, +3.4] % in North America, and there is none in
Africa (-2.2 [-7.7, +1.3] %), South America (+0.1 [-1.8, +2.4] %) and north of 60 N (-2.5 [-8.8, +3.2] %). By lead day it is 3.0 to 3.4 %.
The headline goes from 17.0 % to **14.4 % [7.2, 20.7]** pooled and from 14.3 % to **12.4 % [4.7, 17.0]** as the mean over
regions. The pooled interval excludes zero at 5 of 5 lead days and in 3 of the 3 periods; the interval of the mean over regions at
5 of 5 lead days (by period it includes zero for: issue days 6-20 September, as with three instants). The pooled RMSE of the day is
lower on 48 of 48 issue days (46 with an interval of the day that excludes zero).

**What it does not change.** Every number that does not involve the CAMS forecast is identical to `../score/` to the last
digit (1,368 of the 1,649 entries of `numbers.json`; the 281 that changed all carry `cams` in their name; entries with a
different number of station-days: 0).

**Statements of `../score/reading.md` that need the new value** (old -> new):
- Europe against CAMS: +0.6 [-5.5, +6.3] -> +2.3 [-4.1, +8.3]; episode of 6-9 November +12.3 [+7.4, +15.5] -> +15.9 [+11.7, +18.8]
  (cells-only intervals); the other live days -8.6 [-12.4, -4.6] -> -5.9 [-9.7, -1.9]; by period -4.1 / -2.2 / +4.7 ->
  -4.9 / -0.9 / +8.1 % (September, October, live).
- The 0.4 degree input against CAMS: -7.3 [-10.4, -3.5] -> -4.4 [-7.7, -0.3] pooled; as the mean over regions the interval now includes
  zero (-3.5 [-8.3, +0.8]); lead day 1 -4.3 [-7.2, -1.6] -> -1.5 [-4.1, +1.5].
- SEDI above 37.5 and 75 ug m-3: CAMS 0.69 -> 0.72 and 0.66 -> 0.69; the difference 1 km minus CAMS no longer
  excludes zero (+0.07 [-0.01, +0.15] and +0.06 [-0.02, +0.13]).
- Temporal correlation: CAMS 0.60 -> 0.62 (1 km: 0.56); inside October-November the paired difference 1 km minus CAMS is
  -0.065 [-0.091, -0.012] -> -0.081 [-0.107, -0.026].
- Days with the analysis-mode arm (27 issue days): 1 km against CAMS -13.7 -> -11.6 [-18.7, -5.4].
- Live period, East Asia: -35.2 [-49.1, -2.2] -> -20.0 [-32.2, +8.0].

**Sensitivity to the rule** (same station-days; `side_by_side.md`): 1 km against CAMS -14.4 [-20.7, -7.2] with the paper's eight
steps 00 .. 21 UTC, -14.5 [-20.9, -7.3] with the trapezoid 00 .. 24 UTC, -14.7 [-21.2, -7.6] with the steps 03 .. 24 UTC; RMSE of CAMS
19.24, 19.27, 19.33: the paper's rule gives CAMS its lowest RMSE of the three. Float32 instead of float16 steps:
RMSE 19.243 against 19.243, same contrast.

## 4. Checks

1. **The new files are the product of the existing arm**: for 53 of 53 cycles the leads 12 .. 120 h of the download are
   bit-identical to the ten-lead file that the existing arm read, and for 53 of 53 the step 0 h equals the analysis used as
   its 0 h instant (`../cams3h/build_report.json`).
2. **`cams3i` reproduces the existing arm `cams`**: largest difference 0 ug m-3, 100 % of the values identical, same finite
   monitors, on every one of the 53 cycles (2,050,705 rows of the station-day table: largest difference 0).
3. **The rule is the paper's**: for the 45 cycles that the paper's own builder holds (to 3 November), its daily means and
   mine agree to 0.0003 ug m-3 on the grid (relative 3e-07) and to 3e-05 at the monitors (float32 steps);
   the float16 storage of the steps moves a monitor value by at most 0.036 ug m-3.
4. **By hand from the raw files** (`check_manual.md`): issue days 2026-09-12 (lead days 2, 5), 2026-10-24 (1, 4) and 2026-11-08
   (3, 5), ten monitors drawn once: scalar arithmetic on the four grid nodes, each step interpolated then averaged. Largest
   difference with the stored `cams3h` 3e-06, with `cams3i` 3e-06, with the existing arm `cams` 3e-06 ug m-3.
   Every row of `work/pairs.npz` equals the points files (2,050,705 of 2,050,705 for each arm).
5. **Pooled RMSE without scorer code** (plain numpy on `work/pairs.npz`): 1,819,647 station-days; 1 km 16.4784, CAMS three
   instants 19.8485, CAMS 3-hourly 19.2434, ratio 1.0314; changes -16.979 % and -14.368 %: the values of
   the scorer's tables T02 / T03 of `../score/` and of this directory.
6. **Same station-days, same draws**: the two station-day tables have the same rows and observations; the contrasts of the
   1 km forecast against its input, the climatology and BCSD are identical in the two runs; `side_by_side.py` recomputes both
   headline contrasts on one sample and finds the values of both runs (14 checks, all identical, among them the shares of
   cells, monitors and issue days of `fig_rt_skill/numbers.json` for the old reading).
7. **Lahore and Karachi** (`lahore.md`): the old reading reproduces 59 of 59 numbers of `fig_rt_day/numbers.json`.
8. Expected and not an error: table T14 of this directory compares the arm `cams` with the first live comparison, which used
   three instants, so its `cams` row differs; the agreement of the old arm is in `../score/tables/T14_check_live.md`.
9. **This directory works as a scorer directory for the figure of the real-time skill** (`figure_script_check.json`): a copy of
   `fig_rt_skill/rt_skill_data.py`, run in a temporary directory with `--score` set to this directory (nothing written under
   the paper folder), passes its 8 own checks against these tables (all identical) and gives the shares of cells, monitors and
   issue days of the table above (identical: True). On the rings sample it gives the 0.4 degree input against CAMS at
   -4.3 [-7.6, +0.4] % (the figure's status file quotes -7.2 [-10.1, -2.6] with three instants).

Limits, unchanged: eight week blocks, so the intervals are indicative; South Asia holds 9 % of the station-days and most of
the pooled squared error.

## 5. Files

`../cams3h/` (new): `raw/2026-09a.nc`, `2026-09b.nc`, `2026-10a.nc`, `2026-10b.nc`, `2026-11.nc` (+ one `.json` each,
`requests.json`); `points/<d>.npz` (53 files: `cams3h`, `cams3i`, `cams3h_f32`, `cams3h_trap`, `cams3h_0324`, `id`, `issue`,
`valid`); `build_report.json`; `code/` (`fetch3h.py`, `fetch3h_job.sh`, `c3h_lib.py`, `build_arms.py`, `safe_io.py`); `logs/`.

This directory (new): `README.md`; `work/pairs.npz` (+ `pairs_meta.json`, `refgrade.*`): arms of the scorer with
`cams` = `cams3h`, plus `cams_old` (the existing arm), `cams3i`, `cams3h`, `cams3h_f32`, `cams3h_trap`, `cams3h_0324`;
`tables/T01..T14.{md,json}`, `numbers.json`, `REPORT.md`, `methods.md`, `reading.md` (scorer output and its two text parts);
`side_by_side.{py,json,md}`; `check_manual.{py,json,md}`; `lahore.{py,json,md}`; `check_figure_script.py`,
`figure_script_check.json`; `make_readme.py`; `logs/` (job 23870539 = the scorer; `arms.txt` = its ARMS argument).

## 6. Re-run

```bash
R=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay; PY="/projappl/project_462001804/envs/world/bin/python -I -B"
sbatch $R/cams3h/code/fetch3h_job.sh --deadline <UTC time>      # only if raw/ is empty; re-attaches to its own requests
$PY $R/cams3h/code/build_arms.py                               # raw -> points + build_report.json (2 min, login node)
sbatch --output=$R/score_cams3h/logs/score_%j.out $R/score/run_all_job.sh $R/points $R/score_cams3h "$(cat $R/score_cams3h/logs/arms.txt)" km
$PY $R/score_cams3h/check_manual.py; $PY $R/score_cams3h/side_by_side.py; $PY $R/score_cams3h/lahore.py
$PY $R/score_cams3h/check_figure_script.py; $PY $R/score_cams3h/make_readme.py
$PY $R/score/make_report.py --out $R/score_cams3h              # REPORT.md again, with reading.md of this directory
```
