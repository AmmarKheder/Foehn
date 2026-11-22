# Warnings of six systems, version 3

Supplementary Note *Hidden exposure and warnings*: version 2 of `../v2/` (same grid cells, forecasts, residents and
regions) with three corrections: a further reference, the 1 km estimate of the start day kept unchanged ('persist'), exact
warning factors and centred 0.25 degree blocks (see the header of `tables_v3.py`). In the manuscript as submitted,
Supplementary Fig. 4 (*People reached, correct warnings and hidden people warned, for Foehn and five references*) and the
Supplementary Tables *Who is warned, 1 day ahead*, *The reference forecasts at the number of people warned by Foehn, 1 day
ahead* and *Hidden people by region, above 37.5 ug m-3* show Foehn with eight members and are built from these data by
`analyses/ensemble_size/warnings_eight/`; this folder gives the values of the four members of the evaluation (on all 179
start days in the Source Data). The scripts follow the noise design switch (`FOEHN_DESIGN=ar`: data_ar/, results_ar/).

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
| `eccv/cran-pm/paper/natcomms/analyses/hidden/v3` | `job_v3.sh` |

## Files
| File | What it does |
|---|---|
| `fig_hidden_warned_v3.py` | the who-is-warned figure with the four members of the evaluation (version 3 data) |
| `figdata_hidden_bc.py` | two reference forecasts that see the 1 km estimates as Foehn does (copy of `../v2/figdata_hidden_bc.py`, imported by `figdata_hidden_v3.py`) |
| `figdata_hidden_v3.py` | Who is warned, version 3 (2026-11-20): three corrections of 'who is warned' (H-1, M-1, L-6 of the code audit) |
| `job_v3.sh` | who is warned, version 3 (persistence, exact factors, centred 0.25 deg blocks), 358 forecasts -> v3/data/*.npz |
| `make_si_v3.py` | Supplementary Tables 'warned' and 'matched' of version 3, written from results/numbers_v3.json (tables_v3.py); no number is typed by hand |
| `tables_v3.py` | Numbers of version 3 of 'who is warned' -> results/numbers_v3.json, results/tables_v3.txt Version 2 (tables_v2.py of ../v2/, unchanged) with the three corrections of the code audit (H-1, M-1, L-6; see figdata_hidden_v3.py): H-1 a  … |
