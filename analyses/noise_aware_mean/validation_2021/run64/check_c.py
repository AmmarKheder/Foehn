#!/usr/bin/env python3
"""A14 run R2, test (c) of section 5: members 0-3 of the first 64-member forecast (data/run64, task k00, row 0 = start day 1,
lead 1 of 2021) against the four-member run of sample_m.py on the same forecast in the smoke job (smoke/pop4_b2.npy, tag b2),
at the populated pixels. Section 5 asks for bit identity. It is reported as written, beside the rounding-level rule of A14a
(same finite pattern, mean |difference| <= 0.05 and 99.99th percentile <= 2.5 ug m-3), because the two runs ran in different
GPU processes (A13: the sampler reproduces at the rounding level across processes). R2 decides nothing; this test neither.
No truth, station or monitor is read. CPU only.
    python check_c.py -> results/check_c.json"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def cmp(a, b):
    fa, fb = np.isfinite(a.astype(np.float32)), np.isfinite(b.astype(np.float32))
    both = fa & fb
    d = np.abs(a.astype(np.float32)[both] - b.astype(np.float32)[both])
    return dict(finite_pattern_equal=bool(np.array_equal(fa, fb)),
                bit_identical_share=float((a.view(np.uint16) == b.view(np.uint16)).mean()),
                max_abs=float(d.max()) if d.size else 0.0, p9999_abs=float(np.percentile(d, 99.99)) if d.size else 0.0,
                mean_abs=float(d.mean()) if d.size else 0.0)


def main():
    k64 = np.load(HERE / "data" / "run64" / "keys64_k00.npy")
    done = set(json.loads((HERE / "data" / "run64" / "done64_k00.json").read_text()))
    kb = np.load(HERE / "smoke" / "keys4_b2.npy")
    assert 0 in done and tuple(int(v) for v in k64[0]) == tuple(int(v) for v in kb[0]) == (1, 1, 2), (k64, kb)
    A = np.load(HERE / "data" / "run64" / "pop64_k00.npy", mmap_mode="r")
    B = np.load(HERE / "smoke" / "pop4_b2.npy", mmap_mode="r")
    assert A.shape[1] == B.shape[1] and A.shape[2] == 64 and B.shape[2] == 4
    R = dict(what=__doc__, forecast="start day 1, lead 1, target day 2 (2021)")
    R["members"] = {f"m{m}": cmp(np.asarray(A[0, :, m]), np.asarray(B[0, :, m])) for m in range(4)}
    R["bit_identical_as_section_5"] = all(v["bit_identical_share"] == 1.0 and v["finite_pattern_equal"] for v in R["members"].values())
    R["rounding_level_A14a"] = all(v["finite_pattern_equal"] and v["mean_abs"] <= 0.05 and v["p9999_abs"] <= 2.5
                                   for v in R["members"].values())
    out = HERE / "results" / "check_c.json"
    txt = json.dumps(R, indent=1)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w") as f_:
        f_.write(txt); f_.flush(); os.fsync(f_.fileno())
    assert tmp.read_text() == txt
    os.replace(tmp, out)
    print(json.dumps({k: v for k, v in R.items() if k != "what"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
