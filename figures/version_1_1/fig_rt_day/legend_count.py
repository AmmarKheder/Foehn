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
