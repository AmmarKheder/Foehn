#!/usr/bin/env python3
"""A7 sensitivity (v) (protocol note, "Deviations", 2026-11-19 17:40): the scoring of score_official.py with the
implausible EEA values removed (EEA station-day set to missing when its value is <= 0 or > 300 ug m-3). Nothing else
changes: this file imports score_official, replaces the observations after they are loaded and sends the output to
test2022_official/screened/results/ (the folder screened/ holds links to pos/, truth/, inputs/, preds/).
    python -I score_screened.py score     -> screened/results/scores_official_2022.json
    python -I score_screened.py tables    -> screened/results/tables_official_2022.md"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SCR = HERE.parent / "screened"
LO, HI = 0.0, 300.0


def score():
    import score_official as so
    so.O = SCR
    orig = so.Data.__init__

    def patched(self, a):
        orig(self, a)
        bad = (self.SRC == 0)[:, None] & np.isfinite(self.OBS) & ((self.OBS <= LO) | (self.OBS > HI))
        self.OBS = np.where(bad, np.nan, self.OBS).astype(np.float32)
        self.obs = self.OBS[self.id].T.copy()
        info = {"rule": f"EEA station-day missing when value <= {LO} or > {HI} ug m-3", "eea_station_days_removed_all_8110_rows": int(bad.sum()),
                "eea_station_days_removed_scored_stations": int(bad[self.id].sum()),
                "eea_station_days_before_scored_stations": int((np.isfinite(self.OBS[self.id]) & (self.SRC[self.id] == 0)[:, None]).sum() + bad[self.id].sum())}
        import json
        (SCR / "results").mkdir(exist_ok=True)
        (SCR / "results/screen.json").write_text(json.dumps(info, indent=1))
        so.log(f"SCREENED observations: {info}")

    so.Data.__init__ = patched
    sys.argv = [sys.argv[0], "--no-paper-check"]
    so.main()


def tables():
    import report_tables as rt
    rt.O = SCR
    sys.argv = [sys.argv[0]]
    rt.main()


if __name__ == "__main__":
    {"score": score, "tables": tables}[sys.argv[1]]()
