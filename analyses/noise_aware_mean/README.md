# Noise-aware mean

Supplementary Note *Order in the noise: opposite pairs and the noise-aware mean* and Supplementary Table *Opposite pairs and
the noise-aware mean at the stations of 2022*. Applied after the forecast, the member law is a control variate: each member
as run minus the departure predicted from its own noise field, with the law fitted on other start days from the members
alone (no measurement). Validated on 15 strong-episode start days of 2021, then tested on 2022 at the stations under rules
fixed before the run.

| Folder | Working folder (under `$FOEHN_WORK`) | Content |
|---|---|---|
| `validation_2021/` | `global_model_forecasting_air_pollution/a18/code` | four corrected members against eight plain members on 15 strong-episode start days of 2021, from the 64-member run of 2021; the law fitted under the first design on the 2022 sixteen-member runs, read by the test as a sensitivity |
| `validation_2021/run64/` | `global_model_forecasting_air_pollution/a14/r2` | the 64-member run of 2021 that the validation reads: the choice of the 15 start days (`select2021.py`), the 64 noise fields (`make_noise64.py`), the sampler (`sample_m.py`), its checks and jobs |
| `test_2022/` | `global_model_forecasting_air_pollution/a20/code` | the test of 2022 at the stations and the closed-form exceedance probability |

Order in `test_2022/`: `stpos20.py` (pixel sample), `collect20.py` (noise features and members), `fit20.py` (the law, from
the members only), `check20.py`, then `score20.py`, all through `cpu.sh`. The decision of `score20.py` refuses to run unless
the filing record of the protocol holds its SHA-256.

**Scorers fixed before the runs.** Each was frozen before the run it scored; its SHA-256 is written into its outputs.

| File | SHA-256 of the file that ran | Identical in this release | Difference |
|---|---|---|---|
| `score18.py` | `73ea36a35d3337df5d3d88058f6e4449afcd2ba3a872cb6c1bd6681de161e019` | yes |  |
| `score20.py` | `8060bdbf973755057ab5288eff72b89f8323d33a8569959130970453d557c438` | yes |  |

The comments of these files are left as they were when the files were fixed: a 'review' there is an internal check of the code before the run, and a 'session' the working session that supervised the runs.

`score18.py` keeps the write-guard tag of the working tree; `collect18.py` builds its tag from `FOEHN_WORK`.

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
| `global_model_forecasting_air_pollution/a20/code` | `test_2022/cpu.sh` |
| `global_model_forecasting_air_pollution/a18/code` | `validation_2021/cpu.sh` |
| `global_model_forecasting_air_pollution/a14/r2` | `validation_2021/run64/jobs/array64.sh`, `validation_2021/run64/jobs/cpu_job.sh`, `validation_2021/run64/jobs/smoke_job.sh` |

**Not in this release.**

- The filing records of the protocol.

## Files

### `validation_2021/`
| File | What it does |
|---|---|
| `collect18.py` | A18 step 1 (CPU only, stored data only, no model run) |
| `cpu.sh` | A18, CPU only (no GPU), the job container and numpy (1.26.3) of a15/jobs/cpu.sh |
| `run64/check_c.py` | A14 run R2, test (c) of section 5: members 0-3 of the first 64-member forecast (data/run64, task k00, row 0 = start day 1, lead 1 of 2021) against the four-member run of sample_m.py on the same forecast in the smoke job (smoke/pop … |
| `run64/jobs/array64.sh` | usage (from a14/r2): sbatch --array=0-14 --time=03:15:00 --dependency=afterok:<smoke> jobs/array64.sh A14 run R2: array task k samples the 2 forecasts of line k+1 of jobs/pairs2021.txt (start day d of 2021, leads 1 and 3; days fix … |
| `run64/jobs/cpu_job.sh` | usage: sbatch jobs/cpu_job.sh "<python command run in a14/r2>" (A14 run R2, CPU only, the training container's env) |
| `run64/jobs/smoke_job.sh` | the smoke job of the 64-member run of 2021 (one GCD), which gates `array64.sh` |
| `run64/make_noise64.py` | G/noise_tiles64.npy, one global N(0, 1) field per member for 64 members, by the recipe of experiments/global1km/prep_test.py ('noise' branch, read only): member m = numpy default_rng(1000 + m), 29 rows of float32 standard normals  … |
| `run64/sample_m.py` | copy of a13/warn16/sample16.py; the ONLY changes are --year (default 2022) and --noise (a noise file of G with at least --members global fields, default: the file eval_global_green.Windows chooses) |
| `run64/select2021.py` | the 15 start days of 2021, by the rule fixed in protocol_amendments/A14_infinite_ensemble.md (section 5) BEFORE this script was run |
| `run64/smoke_check.py` | A14 run R2, smoke test, CPU only, no station or monitor file |
| `score18.py` | A18 scorer |

### `test_2022/`
| File | What it does |
|---|---|
| `check20.py` | A20 CPU checks before the collection, in the job container (numpy 1.26.3), CPU only: SOURCE common20.innovation and common20.coeffs have the same source text as a16/arnoise22.py (the generator of G2 / G11) NOISE common20.z_tiles ( … |
| `collect20.py` | A20 step 1 (CPU only, members and regenerated noise only; no truth, no station file): the noise features of every member, start day and lead of G2, and the members of G2 and G11, at the rows of a20/data/pix20.npz (stpos20.py) |
| `common20.py` | A20 common code |
| `cpu.sh` | A20 CPU jobs, CPU only, in the job container of a19/code/cpu.sh (numpy 1.26.3); 0 GCD-hours |
| `fit20.py` | A20 step 2 (CPU only; members and regenerated noise features only: no truth, no station file, no score) |
| `score20.py` | A20 scorer and decision, CPU only, in the job container (numpy 1.26.3) |
| `stpos20.py` | A20 step 0 (CPU, no member, no truth): the pixel sample of A20 -> a20/data/pix20.npz |
