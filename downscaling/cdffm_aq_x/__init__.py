"""cdffm_aq_x: mechanism variants of the CD-FFM adaptation (IDEAS_REPORT.md), built ON TOP of cdffm_aq.

The baseline package cdffm_aq is imported and subclassed, never modified. Variants (--variant of train_x):
  base   the baseline itself (used for the second seed of the baseline; byte-identical code path)
  estab  stabilised encoder: SelfWarp displacements in cell units + zero-initialised flow heads (identity warp at
         init); same function class as the baseline (a linear reparametrisation), different optimiser geometry
  hcf    hierarchical coarse flow: a convolutional flow on the 0.25 deg grid generates the day's cell-mean field of z
         jointly (day-specific, spatially coherent coarse uncertainty); the pointwise flow is conditioned on it
         through the LIIF (two extra feature channels). Training: teacher forcing with the true cell means.
  (ccp)  conditional copula prior: no retraining; fitted post hoc on the frozen baseline (copula_x.py).
Modules: data_x (cell means, block-mean scores), model_x (variants, samplers), train_x (training, validation),
eval_x (evaluation = eval_aq + block scores, same code path for every variant), copula_x (C0/C1 study), test_x.
Runs go to runs/x/<name>, jobs and logs to logs/x/.
"""
from pathlib import Path

from cdffm_aq import PROJECT, CDFFM_SCRIPTS, SMOKE_ROOT, RUNS_ROOT, NOISE_FIT_SMOKE  # noqa: F401

LOGS_X = PROJECT / "logs" / "x"
RUNS_X = PROJECT / "runs" / "x"
CELLMEAN_SMOKE = LOGS_X / "cellmean_smoke.npz"
# global store: z cell means of the 1,270 training days (stored points) and the 358 validation days (full fields),
# built by scripts/build_global_cellmeans.py (CPU job, never inside a GPU job)
CELLMEAN_GLOBAL = PROJECT / "data" / "processed" / "global" / "lghap_clim" / "cellmean_z_A.npz"
