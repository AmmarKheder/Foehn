#!/usr/bin/env python3
"""Collect runs/*/<eval dir>/metrics.json into one markdown table (RMSE vs GHAP at 1 km, then EEA stations).
    python summarize.py [--eval eval_test2022] [--shift eval_test2022_shift]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "runs"
ORDER = ["V0", "V1", "V2", "B_topoflow", "B_climax", "B_earthformer", "B_simvp", "B_convlstm", "cams"]


def load(run, sub):
    p = ROOT / run / sub / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval_test2022")
    ap.add_argument("--shift", default="eval_test2022_shift")
    a = ap.parse_args()
    runs = [r for r in ORDER if load(r, a.eval)]
    if not runs:
        print("no metrics yet")
        return
    leads = list(load(runs[0], a.eval))
    print(f"## RMSE vs GHAP, 1 km land pixels ({a.eval})\n")
    print("| model | " + " | ".join(leads) + " |\n|---|" + "---|" * len(leads))
    pers = load(runs[0], a.eval)
    print("| persistence | " + " | ".join(f"{pers[k]['grid']['persistence']['rmse']:.2f}" for k in leads) + " |")
    for r in runs:
        m = load(r, a.eval)
        print(f"| {r} | " + " | ".join(f"{m[k]['grid']['model']['rmse']:.2f}" for k in leads) + " |")

    print("\n## RMSE at EEA stations\n")
    print("| model | " + " | ".join(leads) + " |\n|---|" + "---|" * len(leads))
    print("| persistence | " + " | ".join(f"{pers[k]['stations']['persistence_rmse']:.2f}" for k in leads) + " |")
    print("| GHAP of the target day | " + " | ".join(f"{pers[k]['stations']['ghap_target_rmse']:.2f}" for k in leads) + " |")
    for r in runs:
        m = load(r, a.eval)
        print(f"| {r} | " + " | ".join(f"{m[k]['stations']['model_rmse']:.2f}" for k in leads) + " |")

    shifted = [r for r in runs if load(r, a.shift)]
    if shifted:
        print("\n## Meteorology reliance: RMSE vs GHAP with the weather of a day 15 days away (increase = uses weather)\n")
        print("| model | " + " | ".join(leads) + " |\n|---|" + "---|" * len(leads))
        for r in shifted:
            m, s = load(r, a.eval), load(r, a.shift)
            print(f"| {r} | " + " | ".join(
                f"{s[k]['grid']['model']['rmse']:.2f} (+{s[k]['grid']['model']['rmse'] - m[k]['grid']['model']['rmse']:.2f})"
                for k in leads) + " |")


if __name__ == "__main__":
    main()
