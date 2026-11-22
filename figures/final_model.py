"""The final Foehn model of the paper and its evaluation directories (one place to switch every figure).
label: name used in the data files of the analysis scripts (compare_fair_<label>.npz, extremes_stations_<label>.npz,
tail_events_<label>.npz); run: run directory in global1km/runs; eval: 2022 test (every second start day, leads 1-7, 4 members);
jan21: 1 km map of 21 January 2022; case: January 2022 East Asian haze (16 members); sel21: 2021 selection evaluation.
FOEHN_DESIGN=<D> (2026-11-21, A16 rebuild, natcomms/rebuild/s1_chain): the final model's folders get the suffix _<D>
(eval_final_crps_e01_<D>, ...) and LABEL becomes crps_e01_<D>, so every derived file carries the design; unset = unchanged."""
import os
import re
from pathlib import Path

RUNS = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs"))
MODELS = {
    "e02": dict(run="green_flow_shift25_global_L7_noant_whres_ghres_ftemis12", eval="eval_final_e02", jan21="eval_jan21",
                case="eval_case_eastasia", sel21="eval2021_sel_e02", nfe=12),
    "crps_e00": dict(run="green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4", eval="eval_final_crps_e00",
                     jan21="eval_jan21_crps_e00", case="eval_case_eastasia_crps_e00", sel21="eval2021_sel_crps_e00", nfe=4),
    "crps_e01": dict(run="green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4", eval="eval_final_crps_e01",
                     jan21="eval_jan21_crps_e01", case="eval_case_eastasia_crps_e01", sel21="eval2021_sel_crps_e01", nfe=4),
}
LABEL = "crps_e01"                                 # switched to the CRPS fine-tuned checkpoint once selected on 2021
DESIGN = os.environ.get("FOEHN_DESIGN", "")          # noise design of the final model's runs ('' = the stored runs)
assert re.fullmatch(r"[a-z0-9]*", DESIGN), DESIGN
SFX = f"_{DESIGN}" if DESIGN else ""
if DESIGN:
    MODELS[LABEL + SFX] = {k: (v + SFX if k in ("eval", "jan21", "case", "sel21") else v) for k, v in MODELS[LABEL].items()}
    LABEL += SFX


def ev(name):
    """an evaluation folder of the final model under the design (e.g. eval_case_kano_crps_e01 -> eval_case_kano_crps_e01_<D>)"""
    return name + SFX


M = MODELS[LABEL]
RUN_DIR = RUNS / M["run"]
EVAL_DIR, JAN21_DIR, CASE_DIR, SEL21_DIR = (RUN_DIR / M[k] for k in ("eval", "jan21", "case", "sel21"))
