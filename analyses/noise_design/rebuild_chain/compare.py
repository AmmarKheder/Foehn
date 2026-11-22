#!/usr/bin/env python3
"""Validation of the S1 chain on the stored runs (label 'check'), CPU, read-only on the paper's files:
  python -I -B compare.py [--report]      (after the drivers ran with FOEHN_DESIGN=check FOEHN_AUDIT=logs/audit_step<k>_check.tsv)
1. every file written by the 'check' run (W lines of the audits) is compared with the stored file the paper uses: the same
   path without the '_check' tokens, or, for a figure render, the render of the same script that the paper's figure comes
   from (FIGMAP). npz: array equality key by key (NaN = NaN; text arrays after removing the label); json: equality of the
   decoded objects after removing the label; text: equality after removing the label; png: pixel equality; pdf: bytes
   without the /CreationDate entry.
2. read audit: a read of an unlabelled file that the chain itself writes under the label (a missed redirect), or of a
   final-model evaluation folder without '_check', is listed.
--report writes check_report.json next to this script (else prints only)."""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_W = os.environ["FOEHN_WORK"].rstrip("/"); _TOP = "/" + _W.strip("/").split("/")[0] + "/"   # top folder of FOEHN_WORK
_LUS = os.path.realpath(_W)[:-len(_W)] if os.path.realpath(_W).endswith(_W) else ""   # prefix of its resolved spelling
canon = lambda p: str(p).replace(_LUS + _TOP, _TOP)   # noqa: E731  (one spelling of the project paths)
S1 = Path(canon(Path(__file__).resolve().parent))
N = S1.parents[1]
ROOT = N.parents[1]
D = "check"
TOK = re.compile(r"_check(?=$|[._/\"' ,)\]}:])")
_T2 = re.compile(r"(data|results|figs|csv|supp_tables_full|Source_Data|MANIFEST|MANIFEST_internal|crps4|UNSEALING_A7|arch_fields)_check(?=$|[./])")
strip = lambda s: _T2.sub(r"\1", s.replace("crps_e01_check", "crps_e01"))   # noqa: E731  (the label only, not words such as pixel_check)
FIGS_D = N / "rebuild" / f"figs_{D}"
NAT = ROOT / "paper" / "nature" / "figs"
ISSF, HIDF, RELF = N / "analyses/issuance/figs", N / "analyses/hidden/v3/figs", N / "analyses/reliability/figs"
# the render of each script that the manuscript's figure comes from (keyfix renders: the paper's SFigs. 4, 6, 10)
FIGMAP = {"fig_overview": NAT, "fig_gc2_realism": NAT, "fig_smoke": NAT, "figED_regions": NAT, "figED_fair": (NAT, "_keyfix"),
          "fig_transport": N / "review/figfix/out", "fig_gc1_schematic_rt2": N / "figures_new/fig1_v2",
          "fig_hidden_v4": N / "figures_new/fig2_v3", "fig_hidden_warned_v3": HIDF, "fig_reliability": RELF,
          "fig_reliability_data": RELF, "fig_sedi_definitions_P99": (ISSF, "_keyfix"), "fig_sedi_definitions_P95": (ISSF, "_keyfix"),
          "fig_sedi_definitions_P90": (ISSF, "_keyfix"), "fig_exposure_eqiss": (ISSF, "_keyfix"), "extremes_row_eqiss": (ISSF, "_keyfix")}


def norm(x):
    if isinstance(x, str):
        return strip(x)
    if isinstance(x, list):
        return [norm(v) for v in x]
    if isinstance(x, dict):
        return {strip(k): norm(v) for k, v in x.items()}
    return x


def jdiff(a, b, path="", out=None):
    """paths where two decoded json objects differ (numbers: largest absolute difference)"""
    out = out if out is not None else {"paths": [], "maxabs": 0.0, "maxrel": 0.0, "other": 0}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out["paths"].append(f"{path}/{k} (only in {'stored' if k in b else 'check'})")
            else:
                jdiff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            jdiff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (a == b or (a != a and b != b)):
            out["paths"].append(path); out["maxabs"] = max(out["maxabs"], abs(a - b)); out["maxrel"] = max(out["maxrel"], abs(a - b) / max(abs(a), abs(b), 1e-300))
    elif a != b:
        out["paths"].append(path); out["other"] += 1
    return out


def cmp_npz(a, b):
    za, zb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    ka, kb = set(za.files), set(zb.files)
    if ka != kb:
        return "differs", f"keys only in check {sorted(ka - kb)}, only in stored {sorted(kb - ka)}"
    bad, worst = [], 0.0
    for k in sorted(ka):
        x, y = za[k], zb[k]
        if x.dtype.kind in "US" or y.dtype.kind in "US":
            if x.shape != y.shape or [strip(str(v)) for v in x.ravel()] != [str(v) for v in y.ravel()]:
                bad.append(k)
        elif x.shape != y.shape or x.dtype != y.dtype:
            bad.append(f"{k} shape/dtype {x.shape}{x.dtype} vs {y.shape}{y.dtype}")
        elif not np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"):
            X, Y = x.astype(np.float64), y.astype(np.float64)
            d = float(np.nanmax(np.abs(X - Y))); r = float(np.nanmax(np.abs(X - Y) / np.maximum(np.maximum(np.abs(X), np.abs(Y)), 1e-300)))
            nanmis = bool((np.isnan(X) != np.isnan(Y)).any())
            worst = max(worst, r if not nanmis else 1.0); bad.append(f"{k} (max |diff| {d:.3g}, relative {r:.2g}{', NaN pattern differs' if nanmis else ''})")
    st = "identical" if not bad else "equal within tolerance" if worst <= 1e-12 and all("shape" not in b for b in bad) else "differs"
    return st, ("all arrays equal" if not bad else "; ".join(bad[:8]) + (f" (+{len(bad) - 8})" if len(bad) > 8 else ""))


def cmp_json(a, b, skip=()):
    x, y = norm(json.loads(Path(a).read_text())), json.loads(Path(b).read_text())
    for k in skip:
        x.pop(k, None); y.pop(k, None)
    if "checks" in x and "checks" in y:                              # step 7 runs with --no-paper-check
        for k in ("paper_compare_fair", "paper_cams12h_stations"):
            y["checks"].pop(k, None)
    if x == y:
        return "identical", "decoded objects equal" + (f" (without {', '.join(skip)})" if skip else "")
    d = jdiff(x, y)
    st = "equal within tolerance" if d["maxrel"] <= 1e-12 and d["other"] == 0 and all("only in" not in p for p in d["paths"]) else "differs"
    return st, f"{len(d['paths'])} values differ (max |diff| {d['maxabs']:.3g}, max relative {d['maxrel']:.2g}): {d['paths'][:4]}"


def cmp_text(a, b, drop=None):
    x, y = strip(Path(a).read_text(errors="replace")).splitlines(), Path(b).read_text(errors="replace").splitlines()
    if drop:
        x, y = [l for l in x if not re.search(drop, l)], [l for l in y if not re.search(drop, l)]
    if x == y:
        return "identical", "equal after removing the label" + (f" (lines matching {drop!r} ignored)" if drop else "")
    import difflib
    dl = [l for l in difflib.unified_diff(y, x, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    only_s, only_c = [l[1:] for l in dl if l[0] == "-"], [l[1:] for l in dl if l[0] == "+"]
    return "differs", (f"{len(only_s)} lines only in stored, {len(only_c)} only in check (of {len(y)}); first stored {only_s[:2]} "
                       f"first check {only_c[:2]}")


def cmp_png(a, b):
    from PIL import Image
    x, y = np.asarray(Image.open(a)), np.asarray(Image.open(b))
    if x.shape != y.shape:
        return "differs", f"size {x.shape} vs {y.shape}"
    n = int((x != y).any(-1).sum()) if x.ndim == 3 else int((x != y).sum())
    return ("identical" if n == 0 else "differs"), f"{n} of {x.shape[0] * x.shape[1]} pixels differ"


def cmp_pdf(a, b):
    rx = lambda p: re.sub(rb"/CreationDate \(D:[^)]*\)", b"", Path(p).read_bytes())   # noqa: E731
    return ("identical", "bytes equal without /CreationDate") if rx(a) == rx(b) else ("differs", "bytes differ (rendering)")


def stored_of(p):
    p = Path(canon(p))
    if FIGS_D in p.parents:
        stem, ext = p.stem, p.suffix
        m = FIGMAP.get(stem)
        if m is None:
            return None
        d, tag = m if isinstance(m, tuple) else (m, "")
        return d / f"{stem}{tag}{ext}"
    return Path(strip(str(p)))


def compare(a, b):
    ext = Path(a).suffix
    if ext == ".npz":
        return cmp_npz(a, b)
    if ext == ".npy":
        x, y = np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")
        if x.shape != y.shape or x.dtype != y.dtype:
            return "differs", f"{x.shape}{x.dtype} vs {y.shape}{y.dtype}"
        eq = all(np.array_equal(x[i], y[i], equal_nan=True) for i in range(x.shape[0])) if x.ndim else np.array_equal(x, y)
        return ("identical", "array equal") if eq else ("differs", "arrays differ")
    if ext == ".json":
        skip = ("written",) if Path(a).name == "scores_official_2022.json" else ()
        return cmp_json(a, b, skip)
    if ext == ".png":
        return cmp_png(a, b)
    if ext == ".pdf":
        return cmp_pdf(a, b)
    if ext in (".md",):
        return cmp_text(a, b, drop=r"Built |\| `?/|modif|Modified| s\.$")
    return cmp_text(a, b)


REASONS = {   # why a 'check' output is not identical to the stored one (established by hand on 2026-11-21)
    "case_eastasia_": "stored file of 2026-10-08 predates the rebuild of stations_2022.npz (2026-10-11 20:10): only the station "
                      "observations and the Beijing/Seoul station-mean series differ; the forecast columns at the peak are equal",
    "paper_numbers_check.txt": "one line: 'BTH obs' 106.5 (current station file) against 108.5 (stored case file of 2026-10-08)",
    "warned": "desc only: make_si_v3.py appends its 'Version 3' sentence to the desc of supp_tables_full/warned.json, which since "
              "2026-11-20 17:20 already holds it (copy of its own output); a re-run repeats the sentence. Rows identical",
    "matched": "desc only (same cause as warned)",
    "README.csv": "STable3/4 descriptions (the repeated make_si_v3 sentence) and the STable6a row (away_a.json is not re-frozen)",
    "MANIFEST_check.md": "the STable6a lines (away_a.json not re-frozen)",
    "Source_Data_check.xlsx": "zip container (timestamps) and the missing sheet STable6a; every other sheet equals its CSV",
    "tables_official_2022.md": "header line (date, source path) and the 10 lines of the two A7(g) paper checks skipped by --no-paper-check",
    "fig_gc2_realism": "gc_style.py changed 2026-10-14 22:42, after the stored render (22:39): panel letters (REBUILD_PLAN 5c); "
                       "panel u also uses the case file rebuilt with the current station file",
    "fig_smoke": "gc_style.py changed 2026-10-14 22:42, after the stored render (22:40): panel letters (REBUILD_PLAN 5c)",
    "fig_transport": "rendering only, 2,419 px (0.02 %) in the bottom row (panels h-j, no final-model data); the same on a login "
                     "and a compute node; cause not established"}


def main():
    rep = {"label": D, "steps": {}, "audit": {}}
    labelled_written = set()
    for aud in sorted(glob.glob(str(S1 / "logs" / "audit_step*_check.tsv"))):
        step = re.search(r"audit_(step\d+)_check", aud).group(1)
        W, R = set(), set()
        for line in open(aud):
            m, p = line.rstrip("\n").split("\t", 1)
            (W if m == "W" else R).add(canon(p))
        labelled_written |= W
        rows = []
        W = {re.sub(r"\.tmp\d+$", "", w) for w in W}                   # safe_write: temp file renamed onto the output
        for w in sorted(W):
            if w.endswith("/UNSEALING_A7_check.md") or not os.path.exists(w):
                continue
            s = stored_of(w)
            if s is None or not s.exists() or str(s) == canon(w):
                rows.append({"check": w, "stored": str(s) if s else None, "status": "no stored counterpart"}); continue
            try:
                st, det = compare(w, s)
            except Exception as e:                                   # noqa: BLE001
                st, det = "error", repr(e)
            why = next((r for k, r in REASONS.items() if k in Path(w).name), None) if st == "differs" else None
            rows.append({"check": w, "stored": str(s), "status": st, "detail": det} | ({"reason": why} if why else {}))
        rep["steps"][step] = rows
        rep["audit"][step] = {"reads": len(R), "writes": len(W), "R": sorted(R)}
    # read audit: missed redirects
    stored_of_written = {strip(w) for w in labelled_written}
    evals = re.compile(r"/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/(eval\w*crps_e01)/")
    for step, a in rep["audit"].items():
        miss = sorted(r for r in a["R"] if r in stored_of_written)
        ev = sorted({m.group(1) for r in a["R"] for m in [evals.search(r)] if m})
        a["missed_redirect_reads"] = miss
        a["stored_eval_folders_read_without_label"] = ev
        a["unlabelled_derived_reads"] = sorted({r for r in a["R"] if re.search(r"/(experiments/figures/data|analyses/[a-z]+(/v[23])?/(data|results)|source_data/supp_tables_full)/", r)
                                                and not TOK.search(r) and "/global1km/runs/" not in r})
        del a["R"]
    # paper_numbers.py prints only: its stdout of the keyfix run (review/keyfix/logs/keyfix_23892389.out) against step 3's
    def pn(path):
        L = open(path, errors="replace").read().splitlines()
        i = next(k for k, l in enumerate(L) if l.startswith("=====") and "paper_numbers.py" in l and "rc=" not in l)
        j = next(k for k in range(i + 1, len(L)) if L[k].startswith("===== rc="))
        return [strip(l) for l in L[i + 1:j] if "findfont" not in l]
    pc, ps = S1 / "logs" / "paper_numbers_check.txt", N / "review/keyfix/logs/keyfix_23892389.out"
    if pc.exists():
        a, b = pn(pc), pn(ps)
        rep["steps"].setdefault("step3", []).append({"reason": REASONS["paper_numbers_check.txt"] if a != b else None,
                                                     "check": str(pc), "stored": str(ps) + " (paper_numbers.py section)",
                                                     "status": "identical" if a == b else "differs",
                                                     "detail": f"{len(a)} printed lines" + ("" if a == b else f"; differing: {[x for x in a if x not in b][:3]}")})
    rep["timings"] = {}
    for lg in sorted(glob.glob(str(S1 / "logs" / "s1_step*_*.out"))):
        st, job = re.search(r"s1_(step\d+)_(\d+)\.out", lg).groups()
        rep["timings"][st] = {"job": job, "steps": [l.strip()[6:] for l in open(lg, errors="replace") if l.startswith("===== rc=")]}
    if "--report" in sys.argv:
        (S1 / "check_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    for step, rows in rep["steps"].items():
        print(f"== {step}")
        for r in rows:
            print(f"  {r['status']:22s} {Path(r['check']).name:48s} {r.get('detail', '')[:150]}")
        a = rep["audit"][step]
        print(f"  audit: {a['reads']} reads, {a['writes']} writes; missed redirects {a['missed_redirect_reads']}; "
              f"unlabelled eval folders {a['stored_eval_folders_read_without_label']}")
        print("  unlabelled derived reads (references expected):", [Path(x).name for x in a["unlabelled_derived_reads"]])


if __name__ == "__main__":
    main()
