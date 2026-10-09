#!/usr/bin/env python3
"""Seam score of saved 1 km maps (evaluate.py --save-days): mean absolute jump across the tile boundaries of a
512-stride tiling (columns / rows 512, 1024, ...) divided by the mean absolute jump between any two neighbouring
pixels. 1.0 = the boundaries are invisible; CRAN-PM's tiles show up as a ratio well above 1.
    python seam_metric.py <eval dir> [<eval dir> ...]"""
import sys
from pathlib import Path

import numpy as np

for out in sys.argv[1:]:
    keys = {}
    for p in sorted(Path(out).glob("part_*.npz")):
        with np.load(p) as z:
            for k in z.files:
                if k.startswith("pred_"):
                    keys[k] = z[k].astype(np.float32)
    if not keys:
        print(out, "no saved maps"); continue
    rows, cols = [], []
    for k, a in sorted(keys.items()):
        a = np.where(a > 0, a, np.nan)
        dc, dr = np.abs(np.diff(a, axis=1)), np.abs(np.diff(a, axis=0))
        bc = np.arange(512, a.shape[1] - 1, 512) - 1   # diff index c-1 = jump between columns c-1 and c
        br = np.arange(512, a.shape[0] - 1, 512) - 1
        rc = np.nanmean(dc[:, bc]) / np.nanmean(dc); rr = np.nanmean(dr[br]) / np.nanmean(dr)
        rows.append(rr); cols.append(rc)
        print(f"{Path(out).name:30s} {k:14s} seam ratio columns {rc:.3f} rows {rr:.3f}")
    print(f"{Path(out).name:30s} MEAN           seam ratio columns {np.mean(cols):.3f} rows {np.mean(rows):.3f}")
