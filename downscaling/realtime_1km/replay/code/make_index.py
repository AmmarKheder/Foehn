#!/usr/bin/env python3
"""WP1a: replay/points/INDEX.json, one row per issue day (read from the files themselves): lead days made, which baselines
are present per lead day, the source of the coarse forecast, the overlap with the fine-tuning targets of the coarse model.
    python -I make_index.py"""
import json
from pathlib import Path

import numpy as np

R = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")


def main():
    rows = {}
    for f in sorted((R / "points").glob("20*.npz")):
        z = np.load(f)
        pr = json.loads(str(z["provenance"]))
        has = lambda k: [int(L + 1) for L in range(5) if np.isfinite(z[k][L]).any()]
        rows[f.stem] = dict(valid=[str(v) for v in z["valid"]], km_leads=has("km"), cams_leads=has("cams"), an_leads=has("an"),
                            rt04_leads=has("rt04"), bcsd_leads=has("bcsd"), clim_leads=has("clim"),
                            coarse=pr["coarse"]["source"].split()[0],
                            predicted_instants_that_were_training_targets=list(pr["coarse"]["predicted_instants_that_were_training_targets"].values()),
                            cams_file=pr["baselines"].get("cams_forecast"))
    out = dict(n_days=len(rows), n_monitors=int(z["id"].size), members=int(z["km"].shape[1]),
               note="lead day L = 1..5, valid day = issue + L - 1; *_leads = lead days with values; km_leads also holds for km_ring*, km_cell*",
               days=rows)
    (R / "points" / "INDEX.json").write_text(json.dumps(out, indent=1))
    print(f"{len(rows)} days; km complete {sum(len(r['km_leads']) == 5 for r in rows.values())}, cams complete {sum(len(r['cams_leads']) == 5 for r in rows.values())}, "
          f"an complete {sum(len(r['an_leads']) == 5 for r in rows.values())}, an partial {sum(0 < len(r['an_leads']) < 5 for r in rows.values())}")


if __name__ == "__main__":
    main()
