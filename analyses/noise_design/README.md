# Noise design of the ensemble

Supplementary Note *Noise design of the ensemble* and Supplementary Table *Main scores under the two noise designs*; every
score of the 1 km model's forecasts of 2022 in the paper is computed under this design. The noise field of each member is
drawn anew at every start day and evolves from one lead day to the next as a first-order autoregressive process,
z_h = rho z_(h-1) + sqrt(1 - rho^2) eps_h, so that each lead day keeps the white-noise law of the first design. rho = 0.59
was chosen on 15 start days of 2021 and the design was confirmed on 30 other start days of 2021. Neither the weights of the
model nor its sampler change: the noise fields that `foehn/eval_global_green.py` reads from `noise_tiles.npy` are replaced by
Gaussian tiles generated on the fly by a counter-based generator (Philox) keyed by year, start day, member, lead day and tile
(`test_2022/arnoise22.py`).

| Folder | Working folder (under `$FOEHN_WORK`) | Content |
|---|---|---|
| `calibration_2021/` | `global_model_forecasting_air_pollution/a15` | choice of rho and confirmation on 2021 |
| `test_2022/` | `global_model_forecasting_air_pollution/a16` | the GPU runs of 2022 under the design (evaluation, populated pixels, case studies, sixteen members) and the completion of the 2021 evaluation (`jobs/g9.sh`, `assemble2021.py`), their checks, the 2022 scores and the check of the member law |
| `episodes/` | `global_model_forecasting_air_pollution/episodes_diag/code` | how often a member ends an episode or drops by more than 40 % (first design against the 1 km estimates), imported by the two folders above |
| `rebuild_chain/` | `eccv/cran-pm/paper/natcomms/rebuild/s1_chain` | the CPU chain that recomputes, under the design, every analysis of the 1 km model's forecasts of 2022 (steps 2-7, 9, 10 and 12; step 8 is `analyses/ensemble_size/run_design.sh`); `figfix_render.py` comes from `eccv/cran-pm/paper/natcomms/review/figfix/render.py` |

Order: in `calibration_2021/`, `check_cpu.py` and `prep.py` (the target, 0.57; through `jobs/cpu.sh`), `jobs/smoke.sh`,
`jobs/calib.sh` then `calib.py` (rho = 0.59), `jobs/main.sh` (confirmation run) and `score.py`; then `test_2022/check_cpu.py`,
`jobs/smoke.sh`, `jobs/g1.sh` ... `g11.sh`, `assemble2021.py`, `check_runs.py`; then `rebuild_chain/step2.sh` ... `step12.sh`
(each with `FOEHN_DESIGN=ar`; `run.py` refuses any write to a path without the `_ar` label). `rebuild_chain/compare.py` with
`check_links.sh` validates the chain on the stored runs (label `check`). The scripts of `analyses/`, `figures/` and
`downscaling/test2022_official/code/` updated in this version are the ones the chain runs. Foehn-CAMS, the two-stage and
real-time systems and the monitor-fed layer do not depend on the design and are read as stored. Some figure and
table numbers in the comments of `rebuild_chain/` (`step5.sh`, `step6.sh`, `step7.sh`, `step9.sh`) are those of an earlier draft;
the tables of this README name the items as submitted.

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
| `global_model_forecasting_air_pollution/a15` | `calibration_2021/jobs/calib.sh`, `calibration_2021/jobs/cpu.sh`, `calibration_2021/jobs/main.sh`, `calibration_2021/jobs/smoke.sh` |
| `eccv/cran-pm/paper/natcomms/rebuild/s1_chain` | `rebuild_chain/step12.sh`, `rebuild_chain/step2.sh`, `rebuild_chain/step3.sh`, `rebuild_chain/step4.sh`, `rebuild_chain/step5.sh`, `rebuild_chain/step6.sh`, `rebuild_chain/step7.sh`, `rebuild_chain/step9.sh` |
| `global_model_forecasting_air_pollution/a16` | `test_2022/jobs/cpu.sh`, `test_2022/jobs/g1.sh`, `test_2022/jobs/g11.sh`, `test_2022/jobs/g2.sh`, `test_2022/jobs/g3.sh`, `test_2022/jobs/g4.sh`, `test_2022/jobs/g5.sh`, `test_2022/jobs/g6.sh`, `test_2022/jobs/g7.sh`, `test_2022/jobs/g8.sh`, `test_2022/jobs/g9.sh`, `test_2022/jobs/smoke.sh` |
| `global_model_forecasting_air_pollution/a16/lawcheck` | `test_2022/lawcheck/lawjob.sh` |
| `global_model_forecasting_air_pollution/a16/score22` | `test_2022/score22/cpu22.sh` |

**Not in this release.**

- The build of the manuscript (step 11 of the chain).

## Files

### `calibration_2021/`
| File | What it does |
|---|---|
| `arnoise.py` | an EVOLVING noise for the frozen Foehn 1 km model (no retraining, no weight change) |
| `calib.py` | A15 rho* from GPU run C, CPU, model output only (no truth enters f-hat): f-hat(rho) = lag-1 (leads 1-2) correlation of A_h = ln max(x_m, 0.1) - ln max(o_m, 0.1) (member m against the mean of the other three, 0.25 deg cell means ma … |
| `check_cpu.py` | A15 CPU checks before any GPU run, in the job container (numpy 1.26.3), CPU only: I1 identity: ARNoise(base 'stored', rho 1) through the UNCHANGED tile_window is uint16-equal to noise_tiles.npy, all 21 windows x 4 members x 7 lead … |
| `eval_ar.py` | eval_global_green.run with the evolving noise of arnoise.py |
| `jobs/calib.sh` | A15 run C (calibration, 60 forecasts: Cal lead 1 at rho 0.70 + Cal lead 2 at rho 0.50, 0.70, 0.90) -> data/calib/part_0..7.npz (a15/A15_draft.md section 5; eval_ar.py --plan calib, 8 ranks, items sorted by (lead, start, rho), todo … |
| `jobs/cpu.sh` | usage: sbatch [--time T] jobs/cpu.sh "<python command run in a15>" (A15, CPU only, the job container's env) sbatch jobs/cpu.sh 'python -u check_cpu.py && python -u prep.py' (limit 1:00) sbatch --time=00:30:00 jobs/cpu.sh 'python - … |
| `jobs/main.sh` | A15 run M (main, 210 forecasts: Conf leads 1-7 at rho* of results/rho.json; resumable) -> data/main/part_0..7.npz (a15/A15_draft.md section 5; eval_ar.py --plan main, 8 ranks, items sorted by (lead, start, rho), todo[rank::8]) |
| `jobs/smoke.sh` | A15 run S, one GCD, checkpoint ckpts/epoch=01.ckpt (that of eval2021_sel_crps_e01): (a) eval_ar.py --plan smoke_stored: base stored, rho 1 (= the stored design), start day 9, leads 1 and 2 -> data/smoke/part_stored.npz (b) eval_ar … |
| `prep.py` | A15 target of the calibration, CPU, before any GPU run, from stored 2021 data only: tau_Cal = the truth's de-diluted lag-1 correlation of its run-varying deviation (log, centered per (cell, lead) over the 15 Cal runs, pooled over  … |
| `score.py` | A15 scoring and decision, CPU, after run M. 2021 only (audit hook against 2022 files), no station or monitor file |
| `smoke_check.py` | A15 smoke gate, run on the GPU node at the end of jobs/smoke.sh, no station file: (a) data/smoke/part_stored.npz (base stored, rho 1 = the stored design) against RUN/eval2021_sel_crps_e01 for (9, 1) and (9, 2), at A14's smoke tole … |

### `test_2022/`
| File | What it does |
|---|---|
| `arnoise22.py` | the evolving noise of A15 for any year, the fixed plans of the A16 runs, their gates, output-folder claims and atomic writes |
| `assemble2021.py` | the 2021 evaluation of the final 1 km model under the evolving noise (rho*), assembled on CPU into ONE folder RUN/eval2021_sel_crps_e01_ar in the format of the stored RUN/eval2021_sel_crps_e01 (F9), from three sets of forecasts th … |
| `check_cpu.py` | A16 check C1, CPU only, in the job container (numpy 1.26.3): GOLD15 the generator of arnoise22.py gives the golden hashes of a15/golden.json (2021 keys), here and in two processes A15_BYTES z tiles and a window cut for 2021 keys ( … |
| `check_runs.py` | A16 checks C3 and C4, CPU only, after the runs |
| `eval_ar22.py` | eval_global_green.run with ONLY the initial noise replaced by the evolving noise of A15 (arnoise22.py) |
| `jobs/cpu.sh` | usage (from a16): sbatch [--time T] jobs/cpu.sh "<python command run in a16>" (A16, CPU only, the job container: numpy 1.26.3) sbatch jobs/cpu.sh 'python -u check_cpu.py' (check C1, before the smoke) sbatch --time=01:30:00 jobs/cp … |
| `jobs/g1.sh` | replaces F1 of natcomms/rebuild/REBUILD_PLAN.md section 1 (2022, 179 start days x leads 1-7, full map, 4 members) |
| `jobs/g11.sh` | replaces F11 of natcomms/rebuild/REBUILD_PLAN.md (16 members at the 8.1 M populated pixels, 2022, the 179 start days x leads 1 and 3 = 358 forecasts) as ONE array of 45 one-GCD tasks: task k samples the 8 forecasts of line k+1 of  … |
| `jobs/g2.sh` | replaces F2 of natcomms/rebuild/REBUILD_PLAN.md section 1 (2022, 179 start days x leads 1 and 3, populated pixels, 4 members) |
| `jobs/g3.sh` | replaces F3 of natcomms/rebuild/REBUILD_PLAN.md section 1 (20 January 2022, lead 1, 1 km maps, 4 members) |
| `jobs/g4.sh` | replaces F4 of natcomms/rebuild/REBUILD_PLAN.md section 1 (20 January 2022, leads 1-7, 1 km maps, 4 members) |
| `jobs/g5.sh` | replaces F5 of natcomms/rebuild/REBUILD_PLAN.md section 1 (East Asian haze, start days 0-27 x leads 1-7, 16 members, crop-only) |
| `jobs/g6.sh` | replaces F6 of natcomms/rebuild/REBUILD_PLAN.md section 1 (Kano dust, start days 132-134 x leads 1-3, 16 members, crop-only) |
| `jobs/g7.sh` | replaces F7 of natcomms/rebuild/REBUILD_PLAN.md section 1 (Gironde smoke, start days 192-203 x leads 1-3, 16 members, crop-only) |
| `jobs/g8.sh` | replaces F8 of natcomms/rebuild/REBUILD_PLAN.md section 1 (Gironde smoke, start days 189-203 x leads 4-7, 16 members, crop-only) |
| `jobs/g9.sh` | completes F9 of natcomms/rebuild/REBUILD_PLAN.md section 1 (2021, the 15 Cal start days x leads 2-7, 4 members; format of a15/data/main) |
| `jobs/smoke.sh` | A16 check C2, one GCD, about 1.0 GCD-hour; the runs G1-G9 and G11 refuse to start unless results/smoke_check.json says all_passed |
| `lawcheck/lawcheck.py` | A16 law check (CPU only, reads only): is the 'ar' design (fresh Philox noise per start day, AR(1) across leads) the same per-forecast law as the stored fixed-field design, with the stored results being the luck of one draw of fixe … |
| `lawcheck/lawfinal.py` | A16 law check, final step (imported by lawcheck.py final): reads pop_stats.npz, ncp_stats.npz, part_noise.json and the part files of F1 (eval_final_crps_e01) and G1 (eval_final_crps_e01_ar); writes lawcheck.json |
| `lawcheck/lawjob.sh` | A16 law check, CPU only, reads only (writes under a16/lawcheck) |
| `lawcheck/lawregion.py` | A16 law check, regional add-on (CPU, reads only): for small boxes (Gironde, Korea, NCP) the member's box-level departure is driven by the mean of its noise over the box |
| `lawcheck/v_law.py` | A16 law check, independent re-check (CPU only, reads only) |
| `sample16_ar.py` | A16 run G11 (and smoke step b16): a13/warn16/sample16.py, the 16-member sampler of the frozen final 1 km model at the 8.1 M populated pixels, with ONLY the initial noise replaced by the evolving noise of A15 (arnoise22.py) |
| `score22/cpu22.sh` | usage (from a16/score22): sbatch [--time T] [--output F] cpu22.sh "<python command run in a16/score22>" CPU only, the job container of a15/jobs/cpu.sh (lumi-pytorch-rocm-6.1.3, torchenv, numpy 1.26.3), 0 GCD-hours. |
| `score22/prep22.py` | the inputs of the replication of A15's statistics for one year and one pair of run folders (arm, stored), CPU. It generalises the input part of A15's scorer (a15/prep.py: cells and their hash; a15/score.py: load_arm, load_stored_r … |
| `score22/score22.py` | the replication of A15 on a year, CPU. Q1-Q6 of A15 section 7.2 and the reported items of A15 section 6.3 for one pair of run folders, arm (2022: G1 = RUN/eval_final_crps_e01_ar) against stored (2022: F1 = RUN/eval_final_crps_e01) … |
| `smoke_check.py` | A16 check C2, run on the GPU node at the end of jobs/smoke.sh; the runs G1-G9 and G11 refuse to start unless results/smoke_check.json says all_passed |

### `episodes/`
| File | What it does |
|---|---|
| `analyze.py` | episodes_diag: E1-E6 of README.md on one extracted sample -> results/results_<sample>.json Every statistic is a ratio of per-run weighted sums; the bootstrap redraws runs (S1 plain, S2 blocks of 4, S3 blocks of 3) and recomputes e … |
| `common.py` | episodes_diag: shared helpers (readers by os.pread, atomic saves, run-length functions, bootstrap counts, regions) |
| `extract_1km.py` | episodes_diag: extraction of S1 (2021, validation) or S2 (2022, test) of Foehn 1 km crps_e01 on the populated 0.25 deg cells -> data/ext_<year>.npz and results/checks_<year>.json (definitions: ../README.md) |
| `revise.py` | episodes_diag REVISION after two internal checks of 2026-11-21 (statistics: review_statistics/, data: review_data/) |

### `rebuild_chain/`
| File | What it does |
|---|---|
| `check_links.sh` | Validation helper of the 'check' label: RUN/<stored eval>_check -> <stored eval> (relative symlinks), so that the chain run with FOEHN_DESIGN=check reads the STORED runs through exactly the code path of a new design (folder + _<D> … |
| `common.sh` | Sourced by every driver of the S1 CPU chain (README.md) |
| `compare.py` | Validation of the S1 chain on the stored runs (label 'check'), CPU, read-only on the paper's files: python -I -B compare.py [--report] (after the drivers ran with FOEHN_DESIGN=check FOEHN_AUDIT=logs/audit_step<k>_check.tsv) 1 |
| `figfix_render.py` | Render a figure script of experiments/figures into review/figfix/out instead of its default folder (paper/nature/figs). |
| `refreeze.py` | Step 10 of REBUILD_PLAN 4c for FOEHN_DESIGN=<D>: the complete Supplementary Tables as JSON (the sheets STable<n> of source_data/build_source_data.py) in source_data/supp_tables_full_<D>/. Run through run.py (write guard) |
| `run.py` | Run one script of the S1 CPU chain under a noise design: FOEHN_DESIGN=<D> python -I -B run.py <script.py> [args ...] * as review/keyfix/scripts/run.py: the script's folder goes first on sys.path and becomes the working directory,  … |
| `step10.sh` | step 10: the full versions of the Supplementary Tables made in steps 4 and 6, frozen with the label of the design (`refreeze.py`) |
| `step12.sh` | step 12: the Source Data under the design |
| `step2.sh` | step 2: post-processing of the evaluation runs of the final model |
| `step3.sh` | step 3: issuance analyses (land, tail events, station extremes) |
| `step4.sh` | step 4: circularity rings and the tables of the analysis of circularity |
| `step5.sh` | step 5: reliability (Supplementary Fig. 10) |
| `step6.sh` | step 6: warnings (Fig. 2f, Supplementary Fig. 4, Supplementary Tables 'Who is warned, 1 day ahead', 'The reference forecasts at the number of people warned by Foehn, 1 day ahead', 'Hidden people by region') |
| `step7.sh` | step 7: official stations (Supplementary Table 'The downscaler and the two-stage system at the official stations of 2022, by lead day') |
| `step9.sh` | step 9: the figure data and renders not made in steps 3, 5 and 6, among them Supplementary Fig. 8 (`figures/fig_overview.py`) and Supplementary Fig. 9 (`figures/figED_regions.py`) |
