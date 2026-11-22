#!/usr/bin/env python3
"""Leaf-by-leaf comparison of the stored outputs of the paper's analysis chain with the outputs of a design (default 'ar').
Read-only. For every pair (stored file, labelled file) of the folders below: JSON leaves and npz arrays are compared; the
changed leaves are written to leafdiff_<D>.json (path, old, new) and a summary is printed. A leaf whose path names a
reference system (cams, aurora, clim, persist, bcsd, qm, ghap, truth, exposed, hidden counts of the truth) is flagged if it
changed: the references are deterministic and must not move with the noise design.
    python -B leafdiff.py [design]"""
import json, os, re, sys
from pathlib import Path
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "ar"
N = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms"))
P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
PAIRS = [  # (stored folder, labelled folder)
    (N / "analyses/circularity/results", N / f"analyses/circularity/results_{D}"),
    (N / "analyses/reliability/results", N / f"analyses/reliability/results_{D}"),
    (N / "analyses/hidden/results", N / f"analyses/hidden/results_{D}"),
    (N / "analyses/hidden/v2/results", N / f"analyses/hidden/v2/results_{D}"),
    (N / "analyses/hidden/v3/results", N / f"analyses/hidden/v3/results_{D}"),
    (N / "analyses/issuance/data", N / f"analyses/issuance/data_{D}"),
    (P / "test2022_official/results", P / f"test2022_official/results_{D}"),
    (N / "source_data/supp_tables_full", N / f"source_data/supp_tables_full_{D}"),
    (P / "a14/classU/results", P / f"a14/classU/results_{D}"),
    (P / "a14/classE/results", P / f"a14/classE/results_{D}"),
]
REF = re.compile(r"(cams|aurora|clim|persist|bcsd|qm|ghap|truth|exposed|perfect)", re.I)
FOE = re.compile(r"(foehn|crps_e01|ensemble|member|m4|m8|m16|inf|issued|diff|eqn|eqc|spread|ssr|crps|rank|brier|cover)", re.I)


def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, x


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    if isinstance(a, str) and isinstance(b, str):
        return a.replace(f"_{D}", "") == b.replace(f"_{D}", "")
    return a == b


out, summ = [], []
for A, B in PAIRS:
    if not B.exists():
        summ.append((str(B), "ABSENT", 0, 0, 0)); continue
    for fb in sorted(B.iterdir()):
        fa = A / fb.name.replace(f"_{D}", "")
        if not fa.exists():
            summ.append((str(fb), "no stored counterpart", 0, 0, 0)); continue
        if fb.suffix == ".json":
            try:
                la, lb = dict(leaves(json.loads(fa.read_text()))), dict(leaves(json.loads(fb.read_text())))
            except Exception as e:
                summ.append((str(fb), f"unreadable {e}", 0, 0, 0)); continue
            ch = [(k, la.get(k, "<absent>"), lb.get(k, "<absent>")) for k in sorted(set(la) | set(lb)) if not same(la.get(k, "<absent>"), lb.get(k, "<absent>"))]
            flag = [c for c in ch if REF.search(c[0]) and not FOE.search(c[0])]
            summ.append((str(fb), "json", len(lb), len(ch), len(flag)))
            out += [{"file": str(fb), "path": k, "old": a, "new": b, "ref_flag": bool(REF.search(k) and not FOE.search(k))} for k, a, b in ch]
        elif fb.suffix == ".npz":
            try:
                za, zb = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
            except Exception as e:
                summ.append((str(fb), f"unreadable {e}", 0, 0, 0)); continue
            ch = [k for k in zb.files if k not in za.files or za[k].shape != zb[k].shape or not (np.array_equal(za[k], zb[k], equal_nan=True) if za[k].dtype.kind in "fiub" else list(map(str, za[k].ravel())) == [str(v).replace(f"_{D}", "") for v in zb[k].ravel()])]
            summ.append((str(fb), "npz", len(zb.files), len(ch), 0))
            out += [{"file": str(fb), "path": k, "old": "<array>", "new": "<array>", "ref_flag": False} for k in ch]
        elif fb.suffix in (".txt", ".md", ".tex", ".tsv", ".csv"):
            ta, tb = fa.read_text(errors="replace").splitlines(), fb.read_text(errors="replace").replace(f"_{D}", "").splitlines()
            ch = sum(1 for x, y in zip(ta, tb) if x != y) + abs(len(ta) - len(tb))
            summ.append((str(fb), "text", len(tb), ch, 0))
tmp = Path(__file__).with_name(f"leafdiff_{D}.json.tmp")
with open(tmp, "w") as fh:
    json.dump(out, fh, indent=0, default=str); fh.flush(); os.fsync(fh.fileno())
os.replace(tmp, Path(__file__).with_name(f"leafdiff_{D}.json"))
for f, kind, n, ch, fl in summ:
    print(f"{kind:8s} leaves/lines {n:6d} changed {ch:6d} ref-flag {fl:4d}  {f.replace(str(N), 'N').replace(str(P), 'P')}")
print("changed leaves:", len(out), "| flagged reference leaves:", sum(1 for o in out if o["ref_flag"]))
