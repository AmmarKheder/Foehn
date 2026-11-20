#!/usr/bin/env python3
"""The numbers that the version 2 texts carry (legend, main-text paragraph, Methods, Supplementary note), one per line, each
with its 95 % interval and its key in results/numbers_v2.json (written by tables_v2.py; nothing is computed here).
    python main_numbers_v2.py  ->  results/main_numbers_v2.txt"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "results" / "numbers_v2.json").read_text())
L = []
NAMES = {"foehn_p50": "Foehn (2 of 4 members)", "foehn_mean": "Foehn (ensemble mean)", "foehn_40km": "Foehn on 40 km cells", "cams": "CAMS forecast",
         "aurora": "Aurora", "cams_qm": "CAMS forecast, bias-corrected", "cams_bcsd": "CAMS forecast, bias-corrected + 1 km pattern",
         "clim_1km": "1 km climatology alone"}


def get(path):
    v = N
    for k in path.split("/"):
        v = v[k]
    return v


def line(label, path, d=1, unit=""):
    v = get(path)
    if isinstance(v, list) and len(v) == 3:
        s = "n/a" if v[0] is None else f"{v[0]:.{d}f}{unit} [{v[1]:.{d}f}, {v[2]:.{d}f}]"
    else:
        s = f"{v:.{d}f}{unit}" if isinstance(v, float) else str(v)
    L.append(f"{label}: {s}   <- {path}")


def main():
    L.append("Sources of the systems (numbers_v2.json/sources):")
    for s, src in N["sources"].items():
        L.append(f"  {s}: {src}")
    for t in ("37.5", "75", "15"):
        for h in (1, 3):
            b = f"systems/thr{t}/lead{h}/World"
            L.append(f"\n--- above {t} ug m-3, {h} day(s) ahead, people a day in millions (populated grid cells, 179 forecasts)")
            line("exposed", f"{b}/foehn_p50/exposed_M"); line("hidden", f"{b}/foehn_p50/hidden_M"); line("hidden with a 10 % margin", f"{b}/foehn_p50/hidden_margin_M")
            for s, nm in NAMES.items():
                for m, lab, u, d in (("reach_pct", "share of the exposed reached", " %", 1), ("correct_pct", "share of warnings correct", " %", 1),
                                     ("warned_M", "people warned", " M", 0), ("right_M", "rightly warned", " M", 0), ("false_M", "wrongly warned", " M", 0),
                                     ("hidden_warned_M", "hidden people warned", " M", 1), ("hidden_warned_pct", "share of the hidden warned", " %", 1),
                                     ("hidden_margin_warned_pct", "share of the margin-hidden warned", " %", 1),
                                     ("warned_in_silent40_M", "warned where the 40 km mean is below", " M", 0),
                                     ("false_in_silent40_M", "wrongly warned where the 40 km mean is below", " M", 0),
                                     ("false_per_hidden_warned", "wrongly warned there per hidden person warned", "", 2), ("csi", "critical success index", "", 3)):
                    line(f"{nm}, {lab}", f"{b}/{s}/{m}", d, u)
            for pair in ("foehn_p50-cams", "foehn_p50-aurora", "foehn_p50-foehn_40km", "foehn_p50-cams_qm", "foehn_p50-cams_bcsd", "foehn_p50-clim_1km",
                         "cams_bcsd-cams_qm", "cams_bcsd-cams", "cams_qm-cams"):
                for m, lab, u, d in (("right_M", "rightly warned", " M", 0), ("false_M", "wrongly warned", " M", 0), ("reach_pct", "reach", " pts", 1),
                                     ("correct_pct", "correct", " pts", 1), ("hidden_warned_M", "hidden warned", " M", 1), ("hidden_warned_pct", "hidden warned", " pts", 1),
                                     ("hidden_margin_warned_pct", "margin-hidden", " pts", 1), ("false_in_silent40_M", "wrongly warned where the 40 km mean is below", " M", 0),
                                     ("csi", "critical success index", "", 3)):
                    line(f"difference {pair}, {lab}", f"{b}/differences/{pair}/{m}", d, u)
            for sw in ("cams_qm", "cams_bcsd", "cams", "aurora"):
                q = f"{b}/equal_number_warned/{sw}"
                try:
                    get(q)
                except KeyError:
                    L.append(f"{NAMES[sw]} at Foehn's number of people warned: outside the swept range"); continue
                line(f"{NAMES[sw]} at Foehn's number of people warned: factor on its threshold", f"{q}/decision_threshold_factor", 2)
                for m, lab, u in (("reach_pct", "reach", " %"), ("correct_pct", "correct", " %"), ("hidden_warned_M", "hidden warned", " M"), ("hidden_warned_pct", "hidden warned", " %")):
                    line(f"    its {lab}", f"{q}/{m}", 1, u)
                for m, lab, u in (("reach_pct", "reach", " pts"), ("correct_pct", "correct", " pts"), ("right_M", "rightly warned", " M"), ("hidden_warned_M", "hidden warned", " M")):
                    line(f"    Foehn minus it, {lab}", f"{q}/foehn_p50_minus_{m}", 1, u)
            for k in ("cams_qm_at_correct_share_of_foehn_p50", "cams_bcsd_at_correct_share_of_foehn_p50", "cams_at_correct_share_of_foehn_p50",
                      "foehn_mean_at_correct_share_of_cams_qm", "foehn_mean_at_correct_share_of_cams_bcsd", "foehn_mean_at_correct_share_of_cams",
                      "foehn_mean_at_correct_share_of_clim_1km"):
                q = f"{b}/equal_correct_share/{k}"
                try:
                    get(q)
                except KeyError:
                    L.append(f"{k}: outside the swept range"); continue
                line(f"{k}: factor on the threshold", f"{q}/decision_threshold_factor", 2); line("    reach", f"{q}/reach_pct", 1, " %")
                line("    reach, moved minus fixed", f"{q}/moved_minus_fixed_reach_pct", 1, " pts"); line("    hidden warned", f"{q}/hidden_warned_M", 1, " M")
                line("    hidden warned, moved minus fixed", f"{q}/moved_minus_fixed_hidden_warned_M", 1, " M")
                line("    hidden warned, ratio moved / fixed", f"{q}/moved_over_fixed_hidden_warned_M", 2)
            for fixed in ("clim_1km", "cams_bcsd", "cams_qm"):
                q = f"{b}/foehn_mean_at_the_number_warned_by/{fixed}"
                try:
                    get(q)
                except KeyError:
                    continue
                for m, lab, u in (("reach_pct", "reach", " pts"), ("correct_pct", "correct", " pts"), ("hidden_warned_M", "hidden warned", " M")):
                    line(f"Foehn's ensemble mean at the number of people warned by {NAMES[fixed]}, minus it: {lab}", f"{q}/foehn_mean_minus_{fixed}_{m}", 1, u)
    X = "foehn_3_days_against_references_1_day"
    L.append(f"\n--- Foehn issued 3 days ahead against the references issued 1 day ahead ({get(X + '/n_common_target_days')} common target days, paired)")
    for t in ("15", "37.5", "75"):
        line(f"> {t}: Foehn 3 d, reach", f"{X}/thr{t}/foehn_p50_lead3/reach_pct", 1, " %"); line(f"> {t}: Foehn 3 d, correct", f"{X}/thr{t}/foehn_p50_lead3/correct_pct", 1, " %")
        for r in ("cams", "aurora", "cams_qm", "cams_bcsd", "foehn_40km"):
            line(f"> {t}: {NAMES[r]} 1 d, reach", f"{X}/thr{t}/{r}_lead1/reach_pct", 1, " %"); line(f"> {t}: {NAMES[r]} 1 d, correct", f"{X}/thr{t}/{r}_lead1/correct_pct", 1, " %")
            for m, lab, u, d in (("reach_pct", "reach", " pts", 1), ("correct_pct", "correct", " pts", 1), ("right_M", "rightly warned", " M", 0), ("false_M", "wrongly warned", " M", 0),
                                 ("csi", "critical success index", "", 3)):
                line(f"> {t}: Foehn 3 d minus {NAMES[r]} 1 d, {lab}", f"{X}/thr{t}/foehn_p50_lead3-{r}_lead1/{m}", d, u)
    C = "checks_of_the_bias_corrected_references"
    L.append("\n--- checks of the bias-corrected references")
    L.append(f"map: {get(C + '/map')}   <- {C}/map")
    for k, v in get(C + "/coverage").items():
        L.append(f"coverage, {k}: {v:,.1f}   <- {C}/coverage/{k}")
    line("largest deviation of the vectorised map from fit_bcsd_ghap.qmap (log1p units)", f"{C}/max_deviation_of_the_vectorised_map_from_fit_bcsd_ghap_qmap", 16)
    L.append(f"identity of the recomputed systems with ../data/hidden_systems_pop.npz: {get('identity_of_recomputed_systems')}   <- identity_of_recomputed_systems")
    L.append(f"version 1 reproduced, largest deviation: {get('check_version1_reproduced')}   <- check_version1_reproduced")
    for h in (1, 3):
        for k in ("valid_residents_M", "without_cell_map_M", "outside_mask_no_pattern_M", "cell_value_below_fitted_range_pct", "cell_value_above_fitted_range_pct"):
            line(f"lead {h}: {k}", f"{C}/lead{h}/{k}", 3)
        for s in ("foehn_mean", "cams", "cams_qm", "cams_bcsd", "clim_1km"):
            for m in ("mean_forecast", "mean_estimate", "bias", "rmse", "mae"):
                line(f"lead {h}: {NAMES[s]}, population-weighted {m} (ug m-3)", f"{C}/lead{h}/errors_population_weighted/{s}/{m}", 2)
    L.append("\n--- country groups, above 37.5, 1 day ahead: Foehn against the bias-corrected CAMS forecast with the 1 km pattern")
    for g, e in get("regions/thr37.5/lead1").items():
        b = f"regions/thr37.5/lead1/{g}"
        for s in ("foehn_p50", "cams", "cams_bcsd"):
            line(f"{g}: {NAMES[s]}, reach", f"{b}/{s}/reach_pct", 1, " %"); line(f"{g}: {NAMES[s]}, correct", f"{b}/{s}/correct_pct", 1, " %")
            line(f"{g}: {NAMES[s]}, hidden warned", f"{b}/{s}/hidden_warned_pct", 1, " %")
        line(f"{g}: Foehn minus bias-corrected + pattern, rightly warned", f"{b}/foehn_p50-cams_bcsd/right_M", 1, " M")
        line(f"{g}: Foehn minus bias-corrected + pattern, critical success index", f"{b}/foehn_p50-cams_bcsd/csi", 3)
    (HERE / "results" / "main_numbers_v2.txt").write_text("\n".join(L) + "\n")
    print(len(L), "lines ->", HERE / "results" / "main_numbers_v2.txt")


if __name__ == "__main__":
    main()
