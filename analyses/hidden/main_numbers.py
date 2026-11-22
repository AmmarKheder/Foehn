#!/usr/bin/env python3
"""The numbers that the text carries, one per line, each with its 95 % interval and its key in results/numbers.json
(written by tables.py; nothing is computed here).   python main_numbers.py  ->  results/main_numbers.txt"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
RES = HERE / f"results{SFX}"
N = json.loads((RES / "numbers.json").read_text())
L = []


def get(path):
    v = N
    for k in path.split("/"):
        v = v[k]
    return v


def line(label, path, d=1, unit=""):
    v = get(path)
    s = f"{v[0]:.{d}f}{unit} [{v[1]:.{d}f}, {v[2]:.{d}f}]" if isinstance(v, list) and len(v) == 3 else (f"{v:.{d}f}{unit}" if isinstance(v, float) else str(v))
    L.append(f"{label}: {s}   <- {path}")


def systems():
    names = {"foehn_p50": "Foehn (2 of 4 members)", "foehn_mean": "Foehn (ensemble mean)", "foehn_40km": "Foehn on 40 km blocks", "cams": "CAMS forecast",
             "aurora": "Aurora", "perfect_40km": "perfect 40 km forecast"}
    for t in ("37.5", "15", "75"):
        for h in (1, 3):
            b = f"systems/thr{t}/lead{h}/World"
            L.append(f"\n--- above {t} ug m-3, {h} day(s) ahead, people a day in millions (populated pixels, 179 forecasts)")
            line("exposed", f"{b}/foehn_p50/exposed_M"); line("hidden", f"{b}/foehn_p50/hidden_M"); line("hidden with a 10 % margin", f"{b}/foehn_p50/hidden_margin_M")
            for s, nm in names.items():
                for m, lab, u in (("hidden_warned_M", "hidden people warned", " M"), ("hidden_warned_pct", "share of the hidden warned", " %"),
                                  ("reach_pct", "share of the exposed reached", " %"), ("correct_pct", "share of warnings correct", " %"),
                                  ("warned_M", "people warned", " M"), ("right_M", "rightly warned", " M"), ("false_M", "wrongly warned", " M"),
                                  ("hidden_margin_warned_pct", "share of the margin-hidden warned", " %")):
                    line(f"{nm}, {lab}", f"{b}/{s}/{m}", 1, u)
            for pair in ("foehn_p50-cams", "foehn_p50-aurora", "foehn_p50-foehn_40km", "foehn_40km-cams"):
                for m, lab, u in (("right_M", "rightly warned", " M"), ("false_M", "wrongly warned", " M"), ("hidden_warned_M", "hidden warned", " M"),
                                  ("reach_pct", "reach", " pts"), ("correct_pct", "correct", " pts"), ("hidden_margin_warned_pct", "margin-hidden", " pts")):
                    line(f"difference {pair}, {lab}", f"{b}/differences/{pair}/{m}", 1, u)
            for k, lab in (("foehn_mean_at_correct_share_of_cams", "Foehn mean at the CAMS forecast's share of correct warnings"),
                           ("cams_at_correct_share_of_foehn_p50", "CAMS forecast at Foehn's share of correct warnings")):
                q = f"{b}/equal_correct_share/{k}"
                try:
                    get(q)
                except KeyError:
                    continue
                line(f"{lab}: factor on the threshold", f"{q}/decision_threshold_factor", 2)
                line(f"{lab}: reach", f"{q}/reach_pct", 1, " %"); line(f"{lab}: hidden warned", f"{q}/hidden_warned_M", 1, " M")
                line(f"{lab}: hidden warned, ratio to the other system", f"{q}/moved_over_fixed_hidden_warned_M", 2)
            q = f"regional_decision_threshold/thr{t}/lead{h}/cams"
            line("CAMS forecast with regional thresholds chosen in hindsight: hidden warned", f"{q}/metrics/hidden_warned_M", 1, " M")
            line("    its wrongly warned", f"{q}/metrics/false_M", 1, " M"); line("    its critical success index", f"{q}/metrics/csi", 3)
            line("    Foehn minus it, critical success index", f"{q}/foehn_p50_minus/csi", 3)
            line("Foehn, critical success index", f"{b}/foehn_p50/csi", 3)
            line("Foehn, warned where the 40 km mean is below", f"{b}/foehn_p50/warned_in_silent40_M", 1, " M")
            line("Foehn, wrongly warned there", f"{b}/foehn_p50/false_in_silent40_M", 1, " M"); line("Foehn, share wrong there", f"{b}/foehn_p50/false_pct_in_silent40", 1, " %")
            line("CAMS forecast, warned where the 40 km mean is below", f"{b}/cams/warned_in_silent40_M", 1, " M")
            line("CAMS forecast, wrongly warned there", f"{b}/cams/false_in_silent40_M", 1, " M"); line("CAMS forecast, share wrong there", f"{b}/cams/false_pct_in_silent40", 1, " %")
    L.append(f"\n--- lead ladder above 37.5 (Foehn's ensemble mean on 0.25 deg blocks; {N['lead_ladder']['n_target_days']} common target days)")
    for s, leads in (("foehn_25km", (1, 3, 5, 7)), ("cams", (1, 3, 5)), ("aurora", (1, 3, 5))):
        for ld in leads:
            for m in ("reach_pct", "correct_pct", "hidden_warned_pct"):
                line(f"{s}, {ld} d, {m}", f"lead_ladder/thr37.5/{s}/lead{ld}/{m}", 1, " %")
    for k in ("foehn_25km_lead5-cams_lead1", "foehn_25km_lead7-cams_lead1", "foehn_25km_lead5-cams_lead5"):
        for m in ("reach_pct", "correct_pct", "right_M", "false_M"):
            line(f"difference {k}, {m}", f"lead_ladder/thr37.5/differences/{k}/{m}", 1)
    L.append("\n--- regions (country groups), above 37.5, 1 day ahead")
    for g in ("South Asia", "East Asia", "Southeast Asia", "Central Asia", "Middle East", "North Africa", "Sub-Saharan Africa", "Europe", "North America",
              "Latin America and the Caribbean"):
        b = f"systems/thr37.5/lead1/G:{g}"
        line(f"{g}: hidden (populated pixels)", f"{b}/foehn_p50/hidden_M", 1, " M"); line(f"{g}: Foehn warns of the hidden", f"{b}/foehn_p50/hidden_warned_pct", 1, " %")
        line(f"{g}: CAMS forecast warns of the hidden", f"{b}/cams/hidden_warned_pct", 1, " %"); line(f"{g}: Aurora warns of the hidden", f"{b}/aurora/hidden_warned_pct", 1, " %")
        line(f"{g}: reach Foehn", f"{b}/foehn_p50/reach_pct", 1, " %"); line(f"{g}: reach CAMS forecast", f"{b}/cams/reach_pct", 1, " %")
        line(f"{g}: correct Foehn", f"{b}/foehn_p50/correct_pct", 1, " %"); line(f"{g}: correct CAMS forecast", f"{b}/cams/correct_pct", 1, " %")
        line(f"{g}: Foehn minus CAMS forecast, rightly warned", f"{b}/differences/foehn_p50-cams/right_M", 1, " M")


def exposure():
    L.append("\n--- the count itself (all pixels with residents, 365 days), people a day in millions")
    for t in ("15", "25", "37.5", "50", "75"):
        b = f"exposure_all_pixels/thr{t}/World"
        line(f"> {t}: exposed", f"{b}/exposed_M"); line(f"> {t}: hidden", f"{b}/hidden_M"); line(f"> {t}: hidden, share of the exposed", f"{b}/hidden_pct", 1, " %")
        line(f"> {t}: reverse", f"{b}/reverse_M"); line(f"> {t}: hidden minus reverse", f"{b}/net_M")
        v = get(f"{b}/daily_hidden_M"); L.append(f"> {t}: daily hidden, min / p5 / median / p95 / max: " + " / ".join(f"{x:.1f}" for x in v) + f"   <- {b}/daily_hidden_M")
    for r in ("South Asia", "East Asia", "Africa", "Middle East", "Southeast Asia", "Europe", "North America", "South America"):
        b = f"exposure_all_pixels/thr37.5/boxes/{r}"
        line(f"box {r}: hidden", f"{b}/hidden_M"); line(f"box {r}: share of the world's hidden", f"{b}/share_of_world_hidden_pct", 1, " %")
        line(f"box {r}: hidden share of the exposed", f"{b}/hidden_pct_of_exposed", 1, " %")
    if "sensitivity" not in N:
        return
    L.append("\n--- sensitivity (data/exposure_sens_2022.npz), people a day in millions")
    for t in ("15", "37.5", "75"):
        b = f"sensitivity/thr{t}"
        for k in ("blocks_40px", "blocks_25px", "blocks_10px", "blocks_40px_shift10", "blocks_40px_shift20", "blocks_40px_shift30"):
            line(f"> {t}: hidden, {k}", f"{b}/{k}/hidden_M"); line(f"> {t}: reverse, {k}", f"{b}/{k}/reverse_M")
        for m in ("rel:0.05", "rel:0.1", "rel:0.2", "abs:1", "abs:2", "abs:5", "abs:10"):
            line(f"> {t}: hidden with a two-sided margin {m}", f"{b}/margin_{m}/two_sided_M"); line(f"> {t}: pixel-only margin {m}", f"{b}/margin_{m}/pixel_only_M")
        for q in ("monitors_0", "monitors_1"):
            line(f"> {t}: hidden {get(f'{b}/{q}/label')}", f"{b}/{q}/hidden_M", 2, " M"); line("    share of the hidden", f"{b}/{q}/share_of_hidden_pct", 2, " %")
    L.append("\n--- country groups (all pixels, 365 days), above 37.5")
    for g, d in N["sensitivity"]["thr37.5"]["groups"].items():
        b = f"sensitivity/thr37.5/groups/{g}"
        line(f"{g}: hidden", f"{b}/hidden_M", 1, " M"); line(f"{g}: share of the world's hidden", f"{b}/share_of_world_hidden_pct", 1, " %")
        if d["hidden_pct_of_exposed"]:
            line(f"{g}: hidden share of the exposed", f"{b}/hidden_pct_of_exposed", 1, " %")
        line(f"{g}: hidden with 10 km blocks", f"{b}/hidden_10km_M", 1, " M"); line(f"{g}: hidden with a 10 % margin", f"{b}/hidden_margin10pct_M", 1, " M")


def income():
    L.append("\n--- income (figures/data/exposure_income_2022.npz), above 37.5")
    for g in ("L", "LM", "UM", "H"):
        b = f"income/thr37.5/groups/{g}"
        line(f"{g}: hidden days per resident and year", f"{b}/hidden_days_per_resident", 2); line(f"{g}: hidden", f"{b}/hidden_M", 1, " M")
        line(f"{g}: share of the world's hidden", f"{b}/share_of_world_hidden_pct", 1, " %"); line(f"{g}: hidden share of the group's exposure", f"{b}/hidden_pct_of_exposed", 1, " %")
    b = "income/thr37.5"
    L.append("within countries, hidden days by quintile (poorest first): " + ", ".join(f"{v:.2f}" for v in get(f"{b}/within_country_quintiles_hidden_days"))
             + f"; richest fifth above the poorest in {get(f'{b}/countries_richest_fifth_more_hidden')} of {get(f'{b}/countries_with_hidden')} countries "
             + f"({get(f'{b}/share_of_residents_in_these_countries_pct'):.0f} % of residents)   <- {b}/within_country_*")
    line("above 75: LM share of the world's hidden", "income/thr75/groups/LM/share_of_world_hidden_pct", 1, " %"); line("above 75: LM hidden", "income/thr75/groups/LM/hidden_M", 1, " M")


if __name__ == "__main__":
    systems(); exposure(); income()
    (RES / "main_numbers.txt").write_text("\n".join(L) + "\n")
    print(len(L), "lines ->", RES / "main_numbers.txt")
