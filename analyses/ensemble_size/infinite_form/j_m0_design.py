#!/usr/bin/env python3
"""A16 Addendum 2 point 2: the reference member-0 ratio (a14/classU_review/judge/j_m0.py) recomputed on the populated-pixel run of
a noise design, RUN/eval_pop_crps_e01<ISFX> -> results_<D>/j_m0.json. The computation between the two marker lines is the
reference, verbatim (j_m0_design.py --self-test compares the two sources). Under 'ar' make_table.py reads this file instead of
the reference record, so that no fixed-design value enters a table of the new design. Refused without FOEHN_DESIGN: the reference
record is never rebuilt. Under 'check' / 'check0' (the stored run) the ratios must equal the reference (asserted)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

from u_common import DESIGN, EVP, P, RES, safe_json

if not DESIGN:
    sys.exit("j_m0_design.py runs only under FOEHN_DESIGN (the reference record j_m0.json is read as stored otherwise)")
REF_DIR = P / "a14" / "classU_review" / "judge"
if "--self-test" in sys.argv:
    src, me = (REF_DIR / "j_m0.py").read_text(), Path(__file__).read_text()
    a, b = src.index('idx = np.load(EVP / "pop_idx.npy")'), src.index("print(json.dumps(res, indent=1))")
    m0, m1 = me.index("# ---- reference computation, verbatim: begin\n") + len("# ---- reference computation, verbatim: begin\n"), me.rindex("# ---- reference computation, verbatim: end")
    ok = src[a:b] == me[m0:m1]
    print("self-test: computation identical to the reference source:", ok)
    sys.exit(0 if ok else 1)
# ---- reference computation, verbatim: begin
idx = np.load(EVP / "pop_idx.npy").astype(np.int64)
row, col = idx // 36000, idx % 36000
cell = (row // 25) * 1440 + (col // 25)
uc, ci = np.unique(cell, return_inverse=True)
npix = np.bincount(ci).astype(np.float64)
big = npix >= 100
big4 = npix >= 400
fc = []
for p in sorted(glob.glob(str(EVP / "part_*.npz")), key=lambda s: int(s.split("_")[-1].split(".")[0])):
    k = int(p.split("_")[-1].split(".")[0])
    for n, (d, h, t) in enumerate(np.load(p)["keys"]):
        if int(h) in (1, 3):
            fc.append((int(d), int(h), k, n))
fc = sorted(fc)
acc = {(h, s): dict(u2=np.zeros(4), s2=0.0, ncd=0.0) for h in (1, 3) for s in ("all", "ge100", "ge400")}
used = {1: 0, 3: 0}
for j, (d, h, k, n) in enumerate(fc):
    X = np.asarray(np.load(EVP / f"pop_{k}.npy", mmap_mode="r")[n], np.float32)
    fin = np.isfinite(X).all(1)
    w = fin.astype(np.float64)
    cnt = np.bincount(ci, weights=w, minlength=len(uc))
    cm = np.stack([np.bincount(ci, weights=np.where(fin, X[:, m], 0.0), minlength=len(uc)) for m in range(4)], 1)
    okc = cnt > 0
    cm = cm[okc] / cnt[okc, None]
    xb = cm.mean(1, keepdims=True); u = cm - xb
    s2 = (u ** 2).sum(1) / 3
    for s, sel in (("all", np.ones(okc.sum(), bool)), ("ge100", big[okc]), ("ge400", big4[okc])):
        a = acc[(h, s)]
        a["u2"] += (4 / 3 * u[sel] ** 2).sum(0); a["s2"] += s2[sel].sum(); a["ncd"] += sel.sum()
    used[h] += 1
res = {"forecasts_used": used, "ratios": {}}
for (h, s), a in acc.items():
    res["ratios"][f"lead{h}_{s}"] = dict(ratio_by_member=(a["u2"] / a["s2"]).tolist(), cell_days=a["ncd"])
# ---- reference computation, verbatim: end
res["source"] = str(EVP)
res["design"] = DESIGN
if DESIGN in ("check", "check0"):
    ref = json.loads((REF_DIR / "j_m0.json").read_text())
    assert res["forecasts_used"] == {int(k): v for k, v in ref["forecasts_used"].items()} or \
        {str(k): v for k, v in res["forecasts_used"].items()} == ref["forecasts_used"], (res["forecasts_used"], ref["forecasts_used"])
    worst = max(abs(x - y) for k in ref["ratios"] for x, y in zip(res["ratios"][k]["ratio_by_member"], ref["ratios"][k]["ratio_by_member"]))
    print("largest |ratio - reference ratio|:", worst)
    assert worst <= 1e-12, worst
print(json.dumps({k: v["ratio_by_member"] for k, v in res["ratios"].items()}))
safe_json(RES / "j_m0.json", res)
