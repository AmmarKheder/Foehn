# Analyses added in version 1.1.0

The analyses made for the revised version of the paper, one folder each.

| Folder | In the paper |
|---|---|
| `issuance/` | every comparison of the global model with the CAMS forecast and Aurora at equal issuance time, over land and at the stations; detection of local extremes (section *Extreme pollution episodes*, Supplementary Note *Detection of local extremes*) |
| `hidden/` | the people hidden by a 40 km grid and the warnings (section *Hidden exposure and public health*, Fig. 2f, Supplementary Note *Hidden exposure and warnings*); `hidden/v2/` adds the two references that see the 1 km estimates (bias-corrected CAMS forecast, with and without the climatological 1 km pattern) |
| `reliability/` | reliability of the ensembles against the 1 km estimates and at the stations (Supplementary Note *Reliability of the ensembles*) |
| `circularity/` | the three tests of the dependence between the stations and the 1 km estimates: forecast read away from the station pixel, stations matched or not by the estimates, model trained without 1 km data (section *Kilometre-scale forecasts*, Supplementary Note *The stations and the 1 km estimates*) |
| `emissions/` | effect of the emission inputs against a control that received the same additional training (section *Transport and emissions*, Supplementary Note *Transport and emissions*) |
| `si_models/` | checks behind the Supplementary Table of the models and the Supplementary Note on the European benchmark |

The scripts are the working files of the study, as they ran on the LUMI supercomputer: they carry the paths of our project space and the Slurm settings of that machine, like the rest of this repository. Data, trained weights, logs and outputs are not in the repository (see *Data availability* and *Code availability* in the paper).

## Files
| File | What it does (first line of its own header) |
|---|---|
| `circularity/scripts/analyse.py` | The defence against circularity, in numbers (test year 2022, final model crps_e01) |
| `circularity/scripts/extract_foehn_cams.py` | The model of the Aurora protocol (Foehn-CAMS: trained on the CAMS analyses at 0.4 deg; no 1 km estimate and no station anywhere in its training, selection or inputs) at the official stations: station-day values |
| `circularity/scripts/extract_job.sh` | CPU only: reads stored forecasts (no network is run) |
| `circularity/scripts/extract_rings.py` | Foehn read AWAY from the station pixel (test year 2022, final model crps_e01; the stations only score) |
| `circularity/scripts/extract_station_day.py` | Station-day values behind the station comparison of the paper (test year 2022, final model crps_e01; the stations only score, nothing here is used to train, select or correct a model) |
| `circularity/scripts/make_outputs.py` | Writes the deliverables from ../results/numbers.json (analyse.py 10000): every number of the table and of the text is formatted here from the unrounded result, never typed by hand |
| `emissions/check_ckpt.py` | What differs between the control checkpoint and the checkpoint with emissions (read only, CPU) |
| `emissions/recompute.py` | Emission and transport gains at equal training, recomputed from the 2022 evaluation files (read only) |
| `emissions/recompute_job.sh` | gains of the emissions and of the transport at equal training, from the 2022 evaluation files (read only): sbatch recompute_job.sh |
| `hidden/check_independent.py` | Independent check of data/hidden_systems_pop.npz for one forecast (start day 64, lead 3): the people warned by the CAMS forecast and by Aurora are recomputed here with the interpolation function of figures/compare_fair.py (copied  … |
| `hidden/fig_hidden_warned.py` | Who is warned, real forecast against real forecast (2022) |
| `hidden/figdata_exposure_sens.py` | Sensitivity of the count of people hidden by a coarse grid (data only, no forecast): a modified COPY of figures/figdata_exposure.py (the original is not touched; its result data/exposure_2022.npz is reproduced as a check) |
| `hidden/figdata_hidden_systems.py` | Who is warned, real forecast against real forecast (2022) |
| `hidden/job_sens.sh` | sensitivity of the count of hidden people (figdata_exposure_sens.py): 16 ranks over the days of 2022, then merge |
| `hidden/job_sys.sh` | people exposed / hidden / warned at the populated 1 km pixels, 2022: Foehn, Foehn at 40 km, CAMS forecast, Aurora (figdata_hidden_systems.py, 32 forked workers) -> data/hidden_systems_pop.npz |
| `hidden/job_test.sh` | smoke tests of the two data scripts (nothing kept except data/pop_pos.npz and data/hidden_systems_pop_test.npz) |
| `hidden/job_test2.sh` | check that figdata_hidden_systems.py with the EVP_EXTRA option (unset) still gives the file of job 23864988 on 4 forecasts |
| `hidden/lead5_PREPARED_not_submitted.sh` | PREPARED, NOT SUBMITTED (a GPU job is not in the mandate of analyses/hidden; the lead decides) |
| `hidden/main_numbers.py` | The numbers that the text carries, one per line, each with its 95 % interval and its key in results/numbers.json (written by tables.py; nothing is computed here) |
| `hidden/make_si.py` | LaTeX tables for the Supplementary Information, written from results/numbers.json (tables.py); no number is typed by hand |
| `hidden/regions.py` | Regions of the hidden-exposure analyses (written 2026-11-19 for the Nature Communications version) |
| `hidden/tables.py` | Numbers of the hidden-population analysis, with intervals over days -> results/numbers.json, results/tables.txt Inputs (nothing is recomputed from the fields here) data/hidden_systems_pop.npz figdata_hidden_systems.py: residents b … |
| `hidden/v2/check_identity.py` | The systems recomputed by figdata_hidden_bc.py (foehn_p50, foehn_mean, foehn_40km, cams) must equal those of ../data/hidden_systems_pop.npz (job 23864988) on every forecast, region, threshold and code: the proof that the new syste … |
| `hidden/v2/check_text_numbers_v2.py` | Every number of the version 2 texts against results/numbers_v2.json |
| `hidden/v2/fig_hidden_warned_v2.py` | Who is warned, version 2 (after the author's feedback) |
| `hidden/v2/figdata_hidden_bc.py` | Who is warned, version 2: two more reference forecasts that see the 1 km estimates as Foehn does (2022) |
| `hidden/v2/job_bc.sh` | people warned by the bias-corrected CAMS forecast (quantile map 2017-2020 of the downscaling project; cell value and climatological 1 km pattern) at the populated 1 km pixels, 358 forecasts of 2022 (v2/figdata_hidden_bc.py, 32 for … |
| `hidden/v2/job_bc_test.sh` | smoke test of v2/figdata_hidden_bc.py on 4 forecasts (start days 0 and 64, leads 1 and 3) + identity check of the recomputed systems against ../data/hidden_systems_pop.npz (job 23864988) |
| `hidden/v2/main_numbers_v2.py` | The numbers that the version 2 texts carry (legend, main-text paragraph, Methods, Supplementary note), one per line, each with its 95 % interval and its key in results/numbers_v2.json (written by tables_v2.py; nothing is computed  … |
| `hidden/v2/make_si_v2.py` | LaTeX tables of version 2 for the Supplementary Information, written from results/numbers_v2.json (tables_v2.py); no number is typed by hand |
| `hidden/v2/tables_v2.py` | Numbers of version 2 of 'who is warned' -> results/numbers_v2.json, results/tables_v2.txt The machinery of ../tables.py is imported unchanged (parts, metrics, Tab, mbb, tri, f): the same pooled sums and the same moving-block boots … |
| `issuance/scripts/common.py` | Shared helpers of the issuance analysis: SEDI and its companions from contingency counts, and the moving-block bootstrap over start days of the paper (compare_fair.py L166-169: blocks of seven consecutive start days, 10,000 resamp … |
| `issuance/scripts/compare_ghap_global_eqiss.py` | COPY of experiments/figures/compare_ghap_global.py, with the issuance of the baselines as a switch (see tail_events_eqiss.py) |
| `issuance/scripts/derive.py` | Under which definition does "Foehn h days ahead detects local extremes as well as the CAMS forecast 1 day ahead" hold? One table for every definition of the threshold, over land (truth: the 1 km estimates, 0.25 deg block means) an … |
| `issuance/scripts/diff_fig_exposure.py` | Proof that the re-rendered Extended Data Fig |
| `issuance/scripts/extremes_panels_eqiss.py` | COPY of experiments/figures/extremes_panels.py (the four panels g-j of Extended Data Fig |
| `issuance/scripts/extremes_stations_matched.py` | COPY of extremes_stations_own_eqiss.py (itself a copy of experiments/figures/extremes_stations_own.py) with ONE more kind of threshold, 'matched': the events of kind 'ghap' (measurement above the local percentile of GHAP 2017-2020 … |
| `issuance/scripts/extremes_stations_own_eqiss.py` | COPY of experiments/figures/extremes_stations_own.py, extended; the definitions are unchanged |
| `issuance/scripts/fig_exposure_eqiss.py` | COPY of experiments/figures/fig_exposure.py (Extended Data Fig |
| `issuance/scripts/fig_sedi_definitions.py` | NEW figure (Supplementary): detection of local extremes under every definition of the threshold, at equal issuance |
| `issuance/scripts/job.sh` | usage: sbatch --job-name=<name> [--export=ALL,ISSUANCE=d1,DUMP=1] job.sh <script.py> [args] |
| `issuance/scripts/land_freqmatch.py` | Detection of local extremes over land against the 1 km estimates (GHAP 0.25 deg block means, 2022) with thresholds matched in frequency, at equal issuance |
| `issuance/scripts/make_table.py` | Supplementary table from ../data/derived.json: SEDI by lead with 95 % intervals for every definition, tier and system, and the lead up to which Foehn's ensemble mean matches the CAMS forecast and Aurora of 1 day |
| `issuance/scripts/summarise.py` | Tables of the issuance analysis, from the files of ../data (nothing else is read except the submitted result files of experiments/figures/data, for the reproduction checks) |
| `issuance/scripts/tail_events_eqiss.py` | COPY of experiments/figures/tail_events.py (its 'evaluate' part), with the issuance of the baselines as a switch |
| `reliability/scripts/fig_reliability.py` | Figure (Supplementary Information or main text): reliability of Foehn's ensemble, 2022, against the 1 km estimates and against the measurements of the 6,324 official stations (panel style of experiments/figures/figED_fair.py) |
| `reliability/scripts/rel_common.py` | Common paths and helpers of the calibration analysis (helper "reliability", 2026-11-19) |
| `reliability/scripts/run_small.sh` | CPU job of the calibration analysis (world venv, no GPU): sbatch --job-name=<name> run_small.sh <script.py> [args] |
| `reliability/scripts/s1_assemble.py` | Step 1: gather, at the station pixels, every stored ensemble member of the 2022 evaluation and the references |
| `reliability/scripts/s2_station.py` | Step 2 (supersedes s2_station_sums.py, which was never run): sums at the station pixels from which every station score and its interval follow (s2b_station_scores.py) |
| `reliability/scripts/s2_station_sums.py` | Step 2: per-forecast sums at the station pixels, from which every station score and its bootstrap interval follow (s2b_station_scores.py) |
| `reliability/scripts/s2b_station_scores.py` | Step 2b: every station score with its intervals, from data/sums_<tag>.npz (s2_station.py) |
| `reliability/scripts/s34_estimates.py` | Steps 3b and 4: calibration against the 1 km estimates |
| `reliability/scripts/s3_pop.py` | Step 3: calibration against the 1 km estimates where people live |
| `reliability/scripts/s5_members16_noise.py` | Step 5: (A) the 16-member evaluations at the stations (case-study runs of the final model: East Asia, 28 start days of January 2022, leads 1-7; Europe window of the Gironde case, 12 start days of July 2022 at leads 1-3 and 15 at l … |
| `reliability/scripts/s6_digest.py` | Step 6: the numbers of the replacement text in one place, the tables for the Supplementary Information, and one sensitivity: how much of the squared error at the stations sits in a few measurements (values above 500 ug m-3) |
| `si_models/scripts/count_words.py` | Word counts of the deliverables with the project's counter (build_test/wordcount.py): the note (heading and \todo excluded) and every new legend of legends_si.tex |
| `si_models/scripts/scorecard_cells.py` | Recompute the 324 cells of the European scorecard (Supplementary Fig |
| `si_models/scripts/seam_check.py` | Recompute the seam ratios of Supplementary Fig |
