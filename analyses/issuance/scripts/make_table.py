#!/usr/bin/env python3
"""Supplementary table from ../data/derived.json: SEDI by lead with 95 % intervals for every definition, tier and system,
and the lead up to which Foehn's ensemble mean matches the CAMS forecast and Aurora of 1 day.
    python make_table.py   -> ../data/table_sedi_definitions.tsv"""
import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"
R = json.loads((D / "derived.json").read_text())
ROWS = [("land", "Global", "1 km estimates, all land", [("ghap", "local climate 2017-2020, calendar day", "same threshold"),
                                                          ("matched", "local climate 2017-2020, calendar day", "matched frequency"),
                                                          ("truth2022", "year 2022", "same threshold"), ("own", "year 2022", "matched frequency")]),
        ("stations", "all", "measurements, six official networks", [("ghap", "local climate 2017-2020, calendar day", "same threshold"),
                                                                     ("matched", "local climate 2017-2020, calendar day", "matched frequency"),
                                                                     ("station", "year 2022", "same threshold"), ("own", "year 2022", "matched frequency")])]
f = lambda v: "" if v is None else f"{v:.3f}"
out = ["\t".join(["truth", "event", "threshold", "tier", "system"] + [f"SEDI_{h}d [95% CI]" for h in range(1, 8)]
                 + ["Foehn_minus_system_equal_lead_1..5 [95% CI]", "Foehn_matches_system_of_1_day_up_to (point / interval, days)", "crossing_lead [95% CI]"])]
for dom, reg, truth, kinds in ROWS:
    for kind, ev, th in kinds:
        for tier in ("P99", "P95", "P90"):
            d = R[dom][kind][tier][reg]
            for s in ("Foehn", "CAMS", "Aurora"):
                x = d[s]; n = len(x["sedi"])
                cells = [f"{f(x['sedi'][i])} [{f(x['ci_low'][i])}, {f(x['ci_high'][i])}]" if i < n and x["sedi"][i] is not None else "" for i in range(7)]
                extra = ["", "", ""]
                if s != "Foehn":
                    e = d[f"Foehn minus {s}, equal lead"]; m = d[f"match with {s} at 1 day"]
                    extra = ["; ".join(f"{a:+.3f} [{b:+.3f}, {c:+.3f}]" for a, b, c in zip(e["diff"], e["ci_low"], e["ci_high"])),
                             f"{m['match_point_lead']} / {m['match_interval_lead']}", f"{m['crossing_lead']:.2f} [{m['crossing_ci'][0]:.2f}, {m['crossing_ci'][1]:.2f}]"]
                out.append("\t".join([truth, ev, th, tier, {"Foehn": "Foehn (ensemble mean)", "CAMS": "CAMS forecast", "Aurora": "Aurora"}[s]] + cells + extra))
(D / "table_sedi_definitions.tsv").write_text("\n".join(out) + "\n")
print("saved", D / "table_sedi_definitions.tsv", len(out) - 1, "rows")
