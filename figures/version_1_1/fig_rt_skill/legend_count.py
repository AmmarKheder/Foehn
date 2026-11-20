#!/usr/bin/env python3
"""Word count of legend.tex with the rules of the paper's counting tool (build_test/wordcount.py, imported read-only).
    python -B legend_count.py"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "build_test"))
import wordcount as W  # noqa: E402

cap = W.captions(W.strip_comments((HERE / "legend.tex").read_text()))[0]
title = cap[cap.index("\\textbf{") + 8:W.braces(cap, cap.index("\\textbf{") + 7) - 1]
print(f"legend: {W.nwords(cap)} words (limit 350), of which title {W.nwords(title)}")
print(" ".join(W.detex(cap).split()))

# every number printed in the legend against numbers.json (key legend_text_values, written by rt_skill_data.py)
import json  # noqa: E402

txt = " ".join(W.detex(cap).split())
bad = [(k, v["printed"]) for k, v in json.loads((HERE / "numbers.json").read_text())["legend_text_values"].items() if v["printed"] not in txt]
print("numbers of the legend against numbers.json:", "all found" if not bad else f"NOT FOUND: {bad}")
