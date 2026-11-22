# Opposite pairs

Supplementary Note *Order in the noise: opposite pairs and the noise-aware mean*, Supplementary Table *Opposite pairs and
the noise-aware mean at the stations of 2022* and the Methods paragraph *Opposite pairs and the noise-aware mean*. A
member's departure from the ensemble mean is close to a linear function of its own noise field, so a member started from
the negated noise has the opposite departure. The pairs were tested on 30 start days of 2021 against the 1 km estimates,
then on 2022 at the stations under rules fixed before the run: four members run from the negated noise of the four
evaluated members, with the same model, weights and sampler.

| Folder | Working folder (under `$FOEHN_WORK`) | Content |
|---|---|---|
| `validation_2021/` | `global_model_forecasting_air_pollution/a17/code` | one opposite pair against four independent members on 30 start days of 2021 |
| `law_diagnostic/` | `global_model_forecasting_air_pollution/a17/diag` | the member law: departure against the member's own noise, smoothed over about 130-160 km |
| `test_2022/` | `global_model_forecasting_air_pollution/a19/code` | the test of 2022 at the stations |

`validation_2021/` and `test_2022/` each run CPU checks (`check*.py` through `cpu.sh`), the GPU smoke (`smoke.sh`), the main
run (`main.sh`) and then the scorer (`score*.py` through `cpu.sh`); `law_diagnostic/` runs `collect.py`, `analyse.py` and
`extra.py` through `job.sh` on stored forecasts.

**Scorers fixed before the runs.** Each was frozen before the run it scored; its SHA-256 is written into its outputs.

| File | SHA-256 of the file that ran | Identical in this release | Difference |
|---|---|---|---|
| `score17.py` | `6c7469b1e04385b520dac31ddc16030efb400b60124ab1cb1773cb73ea6b416e` | yes |  |
| `score19.py` | `a83a558347677752404be487eb158a0a7efba07a4934d209dc7eb3e6ca43c570` | yes |  |

The comments of these files are left as they were when the files were fixed: a 'review' there is an internal check of the code before the run.

`score19.py` names its two working roots literally; `check19.py` compares its write guard with that of `noise19.py`, which
names them through `FOEHN_WORK`.

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
| `global_model_forecasting_air_pollution/a19/code` | `test_2022/cpu.sh`, `test_2022/main.sh`, `test_2022/smoke.sh` |
| `global_model_forecasting_air_pollution/a17/code` | `validation_2021/cpu.sh`, `validation_2021/main.sh`, `validation_2021/smoke.sh` |

## Files

### `validation_2021/`
| File | What it does |
|---|---|
| `check17.py` | A17 CPU checks before any GPU run, in the job container (numpy 1.26.3), CPU only: DIFF every line of eval17.run17 that differs from a15/eval_ar.run_ar ends with '# A17-H<n>' (H1-H6) and the lines removed are exactly eval17.REMOVED … |
| `cpu.sh` | A17 CPU jobs, CPU only, in the job container (numpy 1.26.3); 0 GCD-hours |
| `eval17.py` | the A15 evaluator a15/eval_ar.run_ar with the opposite-noise members |
| `main.sh` | the 4 negated members (ids 4..7) of the 30 Conf starts x leads 1 and 3 of 2021 (60 forecasts) at rho* -> data/main/part_0..7.npz (format of a15/data/main + member_ids) |
| `noise17.py` | opposite-noise pairs on top of the evolving noise of A15/A16 (2021, validation year only) |
| `score17.py` | A17 scorer and decision, CPU only |
| `smoke.sh` | A17 smoke gate, one GCD, about 0.6 GCD-hour; main.sh refuses to start unless data/smoke_check.json says all_passed with the same code |
| `smoke17.py` | A17 smoke gate, run on the GPU node at the end of code/smoke.sh; main.sh and eval17.py --plan main refuse to start unless data/smoke_check.json says all_passed with the same code |

### `law_diagnostic/`
| File | What it does |
|---|---|
| `analyse.py` | A17 diag, step 2 (CPU only; reads only work/ of this folder, written by collect.py) |
| `collect.py` | A17 diag, step 1 (CPU only, stored forecasts only, no model run) |
| `extra.py` | A17 diag, step 3 (CPU only): what the isotropic linear law of analyse.py leaves out |
| `job.sh` | usage (from anywhere): sbatch --output=<log> a17/diag/job.sh "<python command run in a17/diag>" (CPU only; no GPU) The container of P/a15/jobs/cpu.sh (numpy 1.26.3) |

### `test_2022/`
| File | What it does |
|---|---|
| `check19.py` | A19 CPU checks before any GPU run, in the job container (numpy 1.26.3), CPU only: DIFF every line of eval19.run19 that differs from a16/eval_ar22.run_ar22 ends with '# A19-H<n>' (H1-H4) and the lines removed are exactly eval19.REM … |
| `cpu.sh` | A19 CPU jobs, CPU only, in the job container of a16/jobs/cpu.sh (numpy 1.26.3); 0 GCD-hours |
| `eval19.py` | A16's populated-pixel evaluation of 2022 (run G2: eval_ar22.run_ar22, --pop, 179 start days x leads 1 and 3, 64 GCDs) with the member ids of the plan, ids 4..7 being the exact negatives of the noise of members 0..3 (A17's construc … |
| `main.sh` | the 4 negated members (ids 4..7) of A16's G2 forecasts (2022, 179 start days x leads 1 and 3 = 358, populated pixels) -> data/pop_neg (format of RUN/eval_pop_crps_e01_ar, part_<r> = the forecasts of G2's part_<r>) |
| `noise19.py` | eight members as four opposite pairs on the test year 2022, populated pixels, leads 1 and 3 |
| `score19.py` | A19 scorer and decision, CPU only, in the job container (numpy 1.26.3) |
| `smoke.sh` | A19 smoke gate, one GCD, about 0.7 GCD-hour (limit 1:00 = at most 1.0); main.sh refuses to start unless data/smoke_check.json says all_passed with the same code |
| `smoke19.py` | A19 smoke gate, run on the GPU node at the end of code/smoke.sh; main.sh and eval19.py --plan main refuse to start unless data/smoke_check.json says all_passed with the same code |
