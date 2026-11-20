"""cdffm_aq: the minimal CD-FFM adaptation for CAMS PM2.5 (0.4 deg) -> LGHAP v2 (1 km) downscaling (PLAN.md 4.2).

Modules (few files on purpose, inode quota):
  spectra.py   numpy only: radially averaged PSD on lat-lon boxes, slope fits (noise-prior fit, evaluation spectra)
  model_aq.py  periodic Flower encoder (47 channels, whole 0.25 deg globe), periodic LIIF query, sphere noise prior,
               velocity nets imported from CD-FFM/scripts/model.py
  data_aq.py   stores (smoke store; toy store for CPU tests), per-rank prefetch stream, input assembly, target
               transform, metric accumulators
  train_aq.py  training (flow matching / regression / flow-map distillation as CD-FFM fast_train.py) with DDP
  eval_aq.py   streaming evaluation: fair CRPS, ensemble-mean RMSE/bias, spread/skill vs LGHAP, against CAMS
               bilinear and the climatology
  test_s0.py   S0 CPU checks of PLAN.md section 7 (dateline, sphere noise, toy-grid shape test, DDP, real store)
CD-FFM itself stays untouched; its modules are imported from CD-FFM/scripts (model_aq puts that folder on sys.path).
"""
from pathlib import Path

PROJECT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
CDFFM_SCRIPTS = PROJECT / "CD-FFM" / "scripts"
SMOKE_ROOT = PROJECT / "data" / "processed" / "smoke"
RUNS_ROOT = PROJECT / "runs"
NOISE_FIT_SMOKE = SMOKE_ROOT / "noise_fit_smoke.json"
