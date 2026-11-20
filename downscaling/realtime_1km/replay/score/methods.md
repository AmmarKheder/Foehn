# How the scores are computed

**Forecasts.** One file per issue day d (00 UTC cycle) from the reforecast step (`../README.md`): the eight 1 km members at each
monitor's pixel (`km`), on the rings 1, 3 and 10 pixels around it and as the mean of its 0.25-degree cell; the eight members of the
0.4-degree forecast that the 1 km members downscale, read bilinearly (`rt04`, "0.4 deg input"); the CAMS operational forecast of the
same 00 UTC cycle, read bilinearly (`cams`); the harmonic climatology of the 1 km training target, 2017-2020, at the pixel (`clim`);
quantile mapping plus climatological pattern applied to the same 0.4-degree members (`bcsd`, and `bcsd_m` member by member); the
same downscaler run with analysis-type inputs for the valid day (`an`). Lead day L = 1..5 is the UTC day d + L - 1 (lead day 1 is
the issue day itself). The 0.4-degree arms are daily means built from the instants 24(L-1), 24L-12 and 24L hours (weights 1/4, 1/2,
1/4). Ensemble mean = mean of the eight members. Nothing is recomputed, fitted, selected or corrected here.

**Observations.** UTC daily means of the OpenAQ monitors of the 2026 station set (12,136 monitors inside the 1 km domain; at least
18 valid hours; stuck and moving monitors removed; `realtime_1km/verif/verif.py`, `coupling/build_st.py`). Valid days up to 2026-10-21:
`hindcast/2026/st_truth_2026.npz`; from 2026-10-22: the archive re-read with the same rules (`verif/obs/<day>.npz`). The two
sources are identical on the ten days that both hold (T01). The catalogue mixes reference instruments and low-cost sensors (T13
separates them by a proxy), and it does not contain the Chinese official network. Regions are the latitude-longitude boxes of the
catalogue (`verif.py` l.54-56), and their names are wider than their content: "East Asia" (20-50 N, 100-135 E; 1,121 monitors) is
Korea (577), Japan west of 135 E (371), Taiwan (90), Hong Kong and northern Indochina, with fewer than ten monitors in mainland
China; "South Asia" (1,007) is India for 852 monitors; "Africa" (474) includes about 60 monitors of the Levant and the Arabian
Peninsula; "North" = north of 60 N (82); "Other" (2,127) = Japan east of 135 E (791, Tokyo included), South-East Asia (about 580,
mostly Thailand), Central Asia (about 220), Australia and New Zealand (about 190), Canada north of 50 N (120) and scattered sites.

**Identical station-days.** A table lists its arms; a station-day (issue day, lead day, monitor) enters the table only if the
observation and every one of these arms are finite. So the arms of a table are always compared on the same station-days, and two
tables with different arms can have different samples (each table prints its issue days, station-days, monitors, cells and weeks).
The CAMS forecast is on disk for the issue days listed in T01; tables "with the CAMS forecast" use those days, tables "without"
use every issue day. Primary period: issue days from 2026-09-06 (the coarse model was fine-tuned on analyses up to 2026-09-05
00 UTC; issue days 1-5 September are scored only in T11, as a sensitivity).

**Scores.** RMSE, MAE, bias (forecast minus observation) and Pearson r of the ensemble mean over station-days. Relative change =
100 x (RMSE_a / RMSE_b - 1), negative when a is better; the same with mean CRPS. "Pooled" = all station-days of the group in one
RMSE. "Mean over regions" = unweighted mean of the eight regional relative changes; it is given next to every pooled value because
South Asia (9 % of the station-days) carries most of the pooled squared error in the haze season. Fair CRPS of an M-member ensemble:
mean_i |x_i - y| - sum_{i<j} |x_i - x_j| / (M (M - 1)); for a deterministic arm the absolute error. Spread / skill = sqrt((M + 1)/M x
mean unbiased member variance) / RMSE of the ensemble mean. Coverage = share of observations inside the 10th-90th percentile range
of the eight members, with the linear quantiles of the published products (a reliable 8-member ensemble gives 64 % by simulation
with normal or log-normal values, 62 % with uniform ones, not 80 %); rank histogram = rank of the observation among the eight
members (a reliable ensemble leaves 22 % of the observations outside the ensemble). No measurement or representativeness error of
the monitors is added to the members, so a perfect forecast of the 1 km field would still look too narrow by an unknown amount.
Probability of exceedance = share of members above the threshold; Brier score = mean (p - o)^2; for a deterministic arm p is 0 or 1.
Per-monitor temporal correlation and amplitude ratio sd(forecast) / sd(observation): on each (monitor, lead day) series with at
least 15 days, then the median over series; spatial correlation: correlation across monitors (at least 15 station-days) of the
mean forecast and the mean observation. SEDI = (ln F - ln H - ln(1-F) + ln(1-H)) / (ln F + ln H + ln(1-F) + ln(1-H)) with H the hit
rate and F the false-alarm rate of the ensemble mean.

**Intervals: two-way block bootstrap, exactly.** Space blocks are the 2-degree cells, cell = (floor(lat/2) + 90) x 1000 +
floor(lon/2) + 180. Time blocks are the ISO calendar weeks (Monday-Sunday) of the VALID day; a week with fewer than four valid
days in the sample is merged with the neighbouring week of the same run of consecutive weeks. One resample draws C cells with
replacement among the C cells of the sample (multiplicities m_c) and, independently, W weeks with replacement among its W weeks
(multiplicities m_w); each station-day of cell c and week w then counts m_c x m_w times. The statistic is recomputed from the
weighted sums (sums of squared errors, of CRPS, of the four counts of a contingency table, ...). The same 2,000 draws serve every
arm and every group of a table, so contrasts are paired; the mean over regions is recomputed inside each resample; the interval is
the 2.5th-97.5th percentile. For the per-monitor temporal statistics the weeks are reweighted inside every series through its
sufficient statistics and the cells enter as weights of the median over series (400 resamples). The record has eight week blocks
with the September days and five without: with so few time blocks the intervals are indicative, and they are likely too narrow
rather than too wide. Tables restricted to two or three weeks also give the cells-only interval (the convention of the first live
comparison, `rc/live_baselines.py`), which ignores the dependence in time.
