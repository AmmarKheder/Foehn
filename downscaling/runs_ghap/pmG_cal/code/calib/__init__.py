"""calib -- PROTOCOL.md Amendment A6: CRPS calibration of the GHAP downscaler ensemble (pmG_final), station-free.

Built ON TOP of cdffm_aq / cdffm_aq_x, which are imported and never modified. Everything is written under
runs_ghap/pmG_cal/. Modules:
  core.py         paths, the frozen prior P3 and guard, the differentiable inference sampler (10 Euler steps with the
                  clamps of model_aq.integrate), validation sets (V100k, hot-spot probe, V3-odd fit days, in-sample days)
                  and the validation with the inference sampler (per-day sums, pooled summaries, day bootstrap)
  train_cal.py    the fine-tuning: fair CRPS of 2 members drawn as inference draws them + lambda x flow matching
  diag_cal.py     frozen-system diagnostics (A6(d)), temperature fit and grid (A6(e)), checkpoint scoring
  eval_cal.py     cdffm_aq_x.eval_x unmodified, with an optional temperature on the prior field (A6(e))
  select_cal.py   stage-1 rule of A6(f) (CPU)
  compare_cal.py  before/after tables with intervals over the selection days (CPU)
Run with the working directory = the project root (cdffm_aq importable) and PYTHONPATH = runs_ghap/pmG_cal/code.
"""
