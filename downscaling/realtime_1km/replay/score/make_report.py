#!/usr/bin/env python3
"""WP1b step 3: REPORT.md = reading.md (written by hand after the scores) + methods.md + every table of tables/.
  python make_report.py --out S"""
import argparse
import datetime as dt
import json
from pathlib import Path

ORDER = ["T01_samples", "T02_scores", "T03_contrasts", "T04_temporal", "T05_ensemble", "T06_reliability", "T07_sedi", "T08_rings",
         "T09_analysis_mode", "T10_europe", "T11_periods", "T12_title_rule", "T13_official_sites", "T14_check_live"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.out)
    here = Path(__file__).resolve().parent
    parts = ["# Real-time-mode reforecasts at the monitors: scores (WP1b)\n",
             f"Assembled {dt.datetime.now(dt.timezone.utc).isoformat(timespec='minutes')} by `make_report.py` from `tables/*.md` "
             f"(numbers: `tables/*.json`, flat list: `numbers.json`). Re-run everything: `bash run_all.sh` (or `sbatch run_all_job.sh`).\n"]
    for name, title in (("reading.md", None), ("methods.md", None)):
        f = S / name if (S / name).exists() else here / name
        if f.exists():
            parts.append(f.read_text())
        else:
            parts.append(f"*({name} not written yet)*\n")
    parts.append("# Tables\n")
    for t in ORDER:
        f = S / "tables" / f"{t}.md"
        if f.exists():
            parts.append(f.read_text())
        else:
            parts.append(f"## {t}\n\n*(not computed)*\n")
    (S / "REPORT.md").write_text("\n".join(parts))
    n = len(json.loads((S / "numbers.json").read_text())) if (S / "numbers.json").exists() else 0
    print(f"REPORT.md written ({sum(len(p) for p in parts):,} characters); numbers.json holds {n} entries")


if __name__ == "__main__":
    main()
