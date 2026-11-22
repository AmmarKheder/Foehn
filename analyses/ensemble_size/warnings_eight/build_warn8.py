#!/usr/bin/env python3
"""Warnings of Foehn with EIGHT members (A14 decision D2, unchanged by A16 Addendum 3, point 2d): the complete
Supplementary Tables 'warned', 'matched' and 'regions', their short printed versions (LaTeX) and the bar heights of Fig. 2f on
the 149 confirmatory start days of 2022 (sample conf149), every reference on the same days; sixteen and four members beside.

Sources (nothing retyped):
  - the class-E record of A14 (a14/classE/results[_ar]/classE_all2022.json, classE_regions_all2022.json): Foehn's values and
    the paired differences with their interval of record ('interval95'); a reference's own value is the record's Foehn value
    minus the record's paired difference (system minus reference);
  - out_<label>/complements.json (complete_warn8.py, the record's own resamples and the paper's own functions): the rows the
    record lacks (perfect 40 km forecast; ensemble mean of the four members) and the references' own intervals. Every
    complement that the record also determines is checked against it (points, factors, paired differences and their
    bootstrap intervals) before anything is written;
  - source_data/supp_tables_full/regions.json (stored, frozen): the truth-only columns 2-10 of the regions table (all 1 km
    cells with residents, 365 days of 2022; no forecast, so independent of the sample and of the noise design), copied.

    python -B build_warn8.py                    # stored design  -> out_stored/
    FOEHN_DESIGN=ar python -B build_warn8.py    # evolving noise -> out_ar/   (needs a14/classE/results_ar and out_ar/complements.json)

Writes ONLY into out_<label>/: warned.json, matched.json, regions.json (format and columns of source_data/supp_tables_full),
tab_warned.tex, tab_matched.tex, tab_regions.tex (the printed tables of work/supplementary.tex, regenerated from these JSON
files), fig2f.json (Fig. 2f, the values of the Supplementary warnings figure in its layout, the numbers its legend and the
main text quote) and provenance.json. test_identity.py runs the same functions with M = 4 on all 179 start days."""
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
N = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms"))
DESIGN = os.environ.get("FOEHN_DESIGN", "")
DESIGNS = {"": ("stored", P / "a14" / "classE" / "results"), "ar": ("ar", P / "a14" / "classE" / "results_ar")}
if DESIGN not in DESIGNS:
    raise SystemExit(f"FOEHN_DESIGN={DESIGN!r}: build_warn8 knows only unset (stored) and 'ar'")
LABEL, RECDIR = DESIGNS[DESIGN]
OUT = HERE / f"out_{LABEL}"
FROZEN = N / "source_data" / "supp_tables_full"          # column names, and the truth-only columns of 'regions' (stored, frozen)

MU = " µg m-3"
THRS, LEADS = ("15", "37.5", "75"), (1, 3)
M8, M16, M4 = "M8_liberal", "M16_liberal_measured", "M4_stored"
MEMBERS = {M8: 8, M16: 16, M4: 4}
RULE = {M8: "at least 4 of 8 members", M16: "at least 8 of 16 members", M4: "at least 2 of 4 members"}
FOEHN = {M: f"Foehn, {RULE[M]}" for M in RULE}
PRINT_SYSTEMS = (M8, M16, M4)                             # headline first, then the two given beside (A14 D2)
COMP_ONLY = ("foehn_mean", "perfect_40km")                # rows the record lacks: point and interval from complements.json
REFS_W = (("foehn_mean", "Foehn, ensemble mean of the four members"),
          ("foehn_40km", "Foehn, mean of the four members over 40 km cells"), ("cams", "CAMS forecast"), ("aurora", "Aurora"),
          ("cams_qm", "CAMS forecast, bias-corrected"),
          ("cams_bcsd", "CAMS forecast, bias-corrected, with the climatological 1 km pattern"),
          ("clim_1km", "Climatology of the 1 km estimates alone"),
          ("persist", "Start-day 1 km estimate kept unchanged (persistence)"), ("perfect_40km", "Perfect 40 km forecast"))
REFS_M = ("cams", "aurora", "cams_qm", "cams_bcsd", "persist")
REFLAB = dict(REFS_W)
FROZEN_LABEL = {**REFLAB, "foehn_mean": "Foehn, ensemble mean", "foehn_40km": "Foehn, mean over 40 km cells", M4: FOEHN[M4]}
TEXLAB = {M8: "\\model{}, at least 4 of 8 members", M16: "\\model{}, at least 8 of 16 members", M4: "\\model{}, at least 2 of 4 members",
          "foehn_mean": "\\model{}, ensemble mean of the four members", "foehn_40km": "\\model{}, mean of the four members over 40\\,km cells",
          "cams": "CAMS forecast", "aurora": "Aurora", "cams_qm": "CAMS forecast, bias-corrected",
          "cams_bcsd": "\\quad with the climatological 1\\,km pattern", "clim_1km": "Climatology of the 1\\,km estimates alone",
          "persist": "Start-day 1\\,km estimate kept unchanged", "perfect_40km": "Perfect 40\\,km forecast"}
REGIONS = ("South Asia", "East Asia", "Sub-Saharan Africa", "Middle East", "North Africa", "Southeast Asia", "Europe",
           "Latin America and the Caribbean", "Central Asia", "North America", "Unassigned")      # rows of the frozen table
DASH = ("hidden_warned_M", "hidden_warned_pct", "hidden_margin_warned_pct", "false_in_silent40_M")   # perfect 40 km: '0' (make_si_v3)


def tri(name):
    return [name, name + ", 95% interval low", name + ", 95% interval high"]


# columns: (metric of the record, decimals as typeset, with interval)
WMET = (("warned_M", 0, False), ("right_M", 0, True), ("false_M", 0, True), ("reach_pct", 1, True), ("correct_pct", 1, True),
        ("hidden_warned_M", 1, True), ("hidden_warned_pct", 1, True), ("hidden_margin_warned_pct", 1, True),
        ("false_in_silent40_M", 0, True), ("false_per_hidden_warned", 1, True))
WNAMES = ("rightly warned (millions a day)", "wrongly warned (millions a day)", "reach (%)", "correct (%)",
          "hidden people warned (millions a day)", "hidden people warned (%)", "hidden people warned, with a 10% margin (%)",
          "wrongly warned where the mean of the 40 km cell is below the threshold (millions a day)", "the same, per hidden person warned")
WCOLS = ["threshold, lead time and people exposed", "system", "warned (millions a day)"] + sum((tri(n) for n in WNAMES), [])
MMET = ("reach_pct", "correct_pct", "hidden_warned_M")
MNAMES = ("reach (%)", "correct (%)", "hidden people warned (millions a day)")
MCOLS = (["threshold, lead time and Foehn", "reference", "factor on the threshold"] + sum((tri(n) for n in MNAMES), [])
         + sum((tri("Foehn minus reference: " + n.replace("(%)", "(points)")) for n in MNAMES), []))
RCOLS = (["region"] + tri("hidden people (millions a day)") + ["share of the world's hidden people (%)"]
         + tri("share of the region's exposed people (%)") + ["hidden by 10 km blocks (millions a day)", "hidden with a 10% margin (millions a day)"]
         + tri("hidden people warned by Foehn, 1 day ahead (%)") + tri("hidden people warned by Foehn, 3 days ahead (%)")
         + tri("hidden people warned by the CAMS forecast, 1 day ahead (%)")
         + ["reach of Foehn (%)", "reach of the CAMS forecast (%)", "correct warnings of Foehn (%)", "correct warnings of the CAMS forecast (%)"])
RCOLS_TRUTH = tuple(RCOLS[1:10])                           # truth only, all 1 km cells, 365 days: copied from the frozen table
RCOLS_OMITTED = RCOLS_TRUTH                                # (name kept for test_identity.py)
TOL = 1e-9                                                 # relative tolerance of every check complements <-> record


# ------------------------------------------------------------------------------------------------ writing and numbers
def safe_write(path, data):
    """temp file + flush + fsync + read back and compare + os.replace (+ read back after the rename)"""
    path = Path(path)
    data = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data); fh.flush(); os.fsync(fh.fileno())
    if open(tmp, "rb").read() != data:
        raise OSError(f"read-back of {tmp} differs")
    os.replace(tmp, path)
    if open(path, "rb").read() != data:
        raise OSError(f"read-back of {path} differs after the rename")


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False) + "\n"


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def pnum(x, d):
    """the value as typeset with d decimals, read back as the frozen JSON files do (si_tables_portrait.num)"""
    if not fin(x):
        return None
    s = f"{x:,.{d}f}".replace(",", "")
    return int(s) if re.fullmatch(r"-?\d+", s) else float(s)


def plural(h):
    return f"{h} day{'s' if h > 1 else ''} ahead"


# ------------------------------------------------------------------------------------------------ the record and its complements
class Record:
    """the two class-E files of one design and complements.json; every access is logged and every complement that the
    record also determines is checked against it"""

    def __init__(self, ikey="interval95", complements=True):
        self.paths = [RECDIR / "classE_all2022.json", RECDIR / "classE_regions_all2022.json"]
        for p in self.paths:
            if not p.exists():
                raise SystemExit(f"record not found: {p} (design {LABEL!r}); nothing written")
        self.sha256, docs = [], []
        for p in self.paths:
            b = p.read_bytes(); self.sha256.append(hashlib.sha256(b).hexdigest()); docs.append(json.loads(b)); del b
        self.W, self.G = docs
        self.cpath = OUT / "complements.json"
        self.C = None
        if complements:
            if not self.cpath.exists():
                raise SystemExit(f"{self.cpath} not found: run complete_warn8.py first (design {LABEL!r}); nothing written")
            b = self.cpath.read_bytes(); self.csha = hashlib.sha256(b).hexdigest(); self.C = json.loads(b)
            if self.C.get("design") != LABEL:
                raise SystemExit(f"{self.cpath} is of design {self.C.get('design')!r}, not {LABEL!r}")
        self.ikey = ikey
        self.absent, self.status, self.flagged = [], Counter(), []
        self.maxdev = Counter()                            # name -> largest deviation of a consistency check
        self.frozen_regions = {r[0]: r for r in json.loads((FROZEN / "regions.json").read_text(encoding="utf-8"))["rows"]}

    def world(self, sample, t, h):
        return self.W["results"]["world"][sample][f"thr{t}"][f"lead{h}"]

    def region(self, sample, t, h, reg):
        return self.G["results"][sample][f"thr{t}"][f"lead{h}"]["regions"][reg]

    def comp(self, sample, t, h):
        return self.C["world"][sample][f"thr{t}"][f"lead{h}"]

    def n_days(self, sample):
        return int(self.world(sample, THRS[0], LEADS[0])["info"]["n_start_days"])

    def design_note(self):
        return self.W.get("design")

    def dev(self, name, a, b, rel=False):
        if fin(a) and fin(b):
            d = abs(a - b) / max(abs(b), 1.0) if rel else abs(a - b)
            self.maxdev[name] = max(self.maxdev[name], d)
        elif (a is None) != (b is None):
            self.maxdev[name] = float("inf")

    def entry(self, Q, q, M, where):
        e = Q.get(q, {}).get(M) if isinstance(Q.get(q), dict) else None
        if e is None:
            self.absent.append(f"{where}/quantities/{q}/{M}")
        return e

    def value(self, Q, q, M, where):
        e = self.entry(Q, q, M, where)
        return e.get("value") if e is not None and fin(e.get("value")) else None

    def interval(self, e, where, q, M):
        st = e.get("interval_status", "")
        self.status[st.split(";")[0]] += 1
        if "truncated" in st or "NOT inflated" in st or "no interval" in st:
            self.flagged.append(dict(cell=f"{where}/quantities/{q}/{M}", status=st, resamples_nan=e.get("resamples_nan")))
        lo, hi = (e.get(self.ikey) or [None, None])
        return (lo if fin(lo) else None), (hi if fin(hi) else None)

    def system(self, Q, m, M, where):
        """Foehn with M members: value and interval of record of issued:<m>"""
        e = self.entry(Q, f"issued:{m}", M, where)
        if e is None or not fin(e.get("value")):
            return None, None, None
        return (e["value"],) + self.interval(e, where, f"issued:{m}", M)

    def compare(self, Q, kind, r, m, M, where):
        """a paired comparison of record (kind diff or eqn): value and interval of record"""
        q = f"{kind}:{r}:{m}"
        e = self.entry(Q, q, M, where)
        if e is None or not fin(e.get("value")):
            return None, None, None
        return (e["value"],) + self.interval(e, where, q, M)

    def reference(self, Q, kind, r, m, M, where):
        """the reference's own value: issued:<m> minus <kind>:<r>:<m> of the same system M"""
        a, b = self.value(Q, f"issued:{m}", M, where), self.value(Q, f"{kind}:{r}:{m}", M, where)
        if a is None or b is None:
            return None
        if kind == "diff":                                 # the reference as issued does not depend on M: check it
            for M2 in MEMBERS:
                a2, b2 = Q.get(f"issued:{m}", {}).get(M2, {}).get("value"), Q.get(f"{kind}:{r}:{m}", {}).get(M2, {}).get("value")
                if fin(a2) and fin(b2):
                    self.dev("reference as issued, the same from every M (max abs)", a2 - b2, a - b)
        return a - b

    def factor(self, sample, t, h, M, r):
        f = self.world(sample, t, h)["info"].get(M, {}).get(f"eqn:{r}", {})
        return f.get("factor") if fin(f.get("factor")) else None

    # ---- the references with their own intervals (record point, complements interval; checked)
    def ref_issued(self, sample, t, h, r, m, M):
        """(value, low, high) of a reference as issued: the record's point (issued - diff) and the interval of complements.json;
        the rows the record lacks (COMP_ONLY) entirely from complements.json"""
        where = f"world/{sample}/thr{t}/lead{h}"
        c = self.comp(sample, t, h)["issued"][r][m]
        if r in COMP_ONLY:
            return c[0], c[1], c[2]
        v = self.reference(self.world(sample, t, h)["quantities"], "diff", r, m, M, where)
        self.dev("complements: reference as issued = record (issued - diff) (max rel)", c[0], v, rel=True)
        return v, c[1], c[2]

    def ref_moved(self, sample, t, h, r, m, M):
        """(value, low, high) of a reference at Foehn's number warned: the record's point (issued - eqn), complements' interval"""
        where = f"world/{sample}/thr{t}/lead{h}"
        Q = self.world(sample, t, h)["quantities"]
        mv = self.comp(sample, t, h)["moved"][M][r]
        v = self.reference(Q, "eqn", r, m, M, where)
        if isinstance(mv, str):
            return v, None, None
        self.dev("complements: moved reference point = record (issued - eqn) (max rel)", mv["own"][m][0], v, rel=True)
        self.dev("complements: factor = record (max abs)", mv["factor"], self.factor(sample, t, h, M, r))
        if m in mv["difference_as_in_the_record"]:
            e = Q.get(f"eqn:{r}:{m}", {}).get(M, {})
            d = mv["difference_as_in_the_record"][m]
            self.dev("complements: paired difference = record value (max rel)", d[0], e.get("value"), rel=True)
            for i in (0, 1):
                self.dev("complements: paired difference interval = record boot95 (max rel)", d[1 + i], (e.get("boot95") or [None, None])[i], rel=True)
        return v, mv["own"][m][1], mv["own"][m][2]

    def check_foehn(self, sample, t, h):
        """Foehn's points of complements.json (its tables rebuilt from the pass) = the record's"""
        Q = self.world(sample, t, h)["quantities"]
        for M, pts in self.comp(sample, t, h)["foehn"].items():
            for m, v in pts.items():
                if f"issued:{m}" in Q:
                    self.dev("complements: Foehn's points = record (max rel)", v, self.value(Q, f"issued:{m}", M, ""), rel=True)

    def check_days(self):
        for sample in self.C["world"]:
            for h in LEADS:
                d = self.C["start_days"].get(f"{sample}/lead{h}")
                r = self.world(sample, THRS[0], h)["info"].get("start_days")
                if r is not None and d != r:
                    raise SystemExit(f"complements.json: start days of {sample} lead {h} differ from the record")

    def assert_checks(self):
        bad = {k: v for k, v in self.maxdev.items() if k.startswith("complements") and not (v <= TOL)}
        if bad:
            raise SystemExit(f"complements.json disagrees with the record: {bad}; nothing written")

    def counts(self, sample, t, h, M):
        """people exposed (regional record, World), hidden and hidden with the 10 % margin (from issued shares of M)"""
        where = f"world/{sample}/thr{t}/lead{h}"
        Q = self.world(sample, t, h)["quantities"]
        ex = self.region(sample, t, h, "World").get("exposed_M")

        def ratio(num, pct, M_):
            a, b = self.value(Q, f"issued:{num}", M_, where), self.value(Q, f"issued:{pct}", M_, where)
            return 100.0 * a / b if a is not None and b else None
        hid, mar = ratio("hidden_warned_M", "hidden_warned_pct", M), ratio("hidden_margin_warned_M", "hidden_margin_warned_pct", M)
        for M2 in MEMBERS:
            self.dev("hidden people from every M (max rel)", ratio("hidden_warned_M", "hidden_warned_pct", M2), hid, rel=True)
            self.dev("hidden with the margin from every M (max rel)", ratio("hidden_margin_warned_M", "hidden_margin_warned_pct", M2), mar, rel=True)
            self.dev("exposed = right/reach of every M vs regional World exposed_M (max rel)", ratio("right_M", "reach_pct", M2), ex, rel=True)
        if self.C is not None:
            c = self.comp(sample, t, h)["issued"]["perfect_40km"]
            self.dev("complements: exposed (perfect 40 km table) = record (max rel)", c["exposed_M"][0], ex, rel=True)
            self.dev("complements: hidden (perfect 40 km table) = record (max rel)", c["hidden_M"][0], hid, rel=True)
        return (ex if fin(ex) else None), hid, mar


# ------------------------------------------------------------------------------------------------ the tables
def warned_rows(rec, sample, systems):
    """-> [((t, h), key, row)]; systems = M keys, the first is the one the references are read from"""
    out = []
    for t in THRS:
        for h in LEADS:
            where = f"world/{sample}/thr{t}/lead{h}"
            Q = rec.world(sample, t, h)["quantities"]
            rec.check_foehn(sample, t, h)
            ex, hid, mar = rec.counts(sample, t, h, systems[0])
            head = (f"Above {t}{MU}, {plural(h)}: {ex:,.0f} million people exposed a day, {hid:,.1f} million hidden, "
                    f"{mar:,.1f} million hidden with a 10% margin")
            for M in systems:
                row = [head, FOEHN[M]]
                for m, d, wci in WMET:
                    v, lo, hi = rec.system(Q, m, M, where)
                    row += [pnum(v, d)] + ([pnum(lo, d), pnum(hi, d)] if wci else [])
                out.append(((t, h), M, row))
            for r, lab in REFS_W:
                row = [head, lab]
                for m, d, wci in WMET:
                    v, lo, hi = rec.ref_issued(sample, t, h, r, m, systems[0])
                    if r == "perfect_40km" and m in DASH:      # a perfect 40 km forecast warns no hidden person (make_si_v3: '0')
                        if v != 0:
                            raise SystemExit(f"perfect 40 km forecast: {m} = {v}, not 0")
                        row += [0, None, None]; continue
                    if r == "perfect_40km" and m == "false_per_hidden_warned":
                        row += [None, None, None]; continue
                    row += [pnum(v, d)] + ([pnum(lo, d), pnum(hi, d)] if wci else [])
                out.append(((t, h), r, row))
    return out


def matched_rows(rec, sample, systems, named=True):
    """-> [((t, h), M, r, row)]: every reference at the number of people warned by Foehn with M members"""
    out = []
    for t in THRS:
        for h in LEADS:
            where = f"world/{sample}/thr{t}/lead{h}"
            Q = rec.world(sample, t, h)["quantities"]
            for M in systems:
                w, re_, co, hw = (rec.value(Q, f"issued:{m}", M, where) for m in ("warned_M", "reach_pct", "correct_pct", "hidden_warned_M"))
                who = f"Foehn, {RULE[M]}," if named else "Foehn"
                head = (f"Above {t}{MU}, {plural(h)}: {who} warns {w:,.0f} million people a day, reach {re_:.1f}%, correct {co:.1f}%, "
                        f"{hw:,.1f} million hidden people warned")
                for r in REFS_M:
                    fac = rec.factor(sample, t, h, M, r)
                    row = [head, REFLAB[r], pnum(fac, 3)]
                    for m in MMET:
                        v, lo, hi = rec.ref_moved(sample, t, h, r, m, M) if fac is not None else (None, None, None)
                        row += [pnum(v, 1), pnum(lo, 1), pnum(hi, 1)]
                    for m in MMET:
                        v, lo, hi = rec.compare(Q, "eqn", r, m, M, where) if fac is not None else (None, None, None)
                        row += [pnum(v, 1), pnum(lo, 1), pnum(hi, 1)]
                    out.append(((t, h), M, r, row))
    return out


def regions_rows(rec, sample, M):
    out = []
    for g in REGIONS:
        where = f"regions/{sample}/thr37.5/G:{g}"
        Q1 = rec.region(sample, "37.5", 1, f"G:{g}")["quantities"]
        Q3 = rec.region(sample, "37.5", 3, f"G:{g}")["quantities"]
        fz = rec.frozen_regions[g]                         # truth only (365 days, all cells): copied
        row = [g] + list(fz[1:10])
        for Q, h in ((Q1, 1), (Q3, 3)):
            v, lo, hi = rec.system(Q, "hidden_warned_pct", M, where + f"/lead{h}")
            row += [pnum(v, 1), pnum(lo, 1), pnum(hi, 1)]
        v = rec.reference(Q1, "diff", "cams", "hidden_warned_pct", M, where + "/lead1")
        c = rec.C["regions"][sample]["lead1"][f"G:{g}"]["cams"]
        rec.dev("complements: regional CAMS forecast = record (issued - diff) (max rel)", c["hidden_warned_pct"][0], v, rel=True)
        row += [pnum(v, 1), pnum(c["hidden_warned_pct"][1], 1), pnum(c["hidden_warned_pct"][2], 1)]
        for m in ("reach_pct", "correct_pct"):
            row += [pnum(rec.value(Q1, f"issued:{m}", M, where + "/lead1"), 1), pnum(rec.reference(Q1, "diff", "cams", m, M, where + "/lead1"), 1)]
        out.append((g, row))
    return out


# ------------------------------------------------------------------------------------------------ the figures
def fkey(M):
    return f"foehn_m{MEMBERS[M]}"


def fig2f(rec, sample, M):
    """Fig. 2f: share of the exposed people warned (reach_pct) and share of correct warnings (correct_pct), World, 1 day
    ahead, above 37.5 and 75, Foehn with M members, the CAMS forecast and Aurora; layout systems/thr<t>/lead1/World/<system>/<metric>
    = [value, low, high] as the figure reads numbers_v2.json"""
    systems, bars = {}, []
    for t in ("37.5", "75"):
        where = f"world/{sample}/thr{t}/lead1"
        Q = rec.world(sample, t, 1)["quantities"]
        S = systems.setdefault(f"thr{t}", {}).setdefault("lead1", {}).setdefault("World", {})
        for m in ("reach_pct", "correct_pct"):
            v, lo, hi = rec.system(Q, m, M, where)
            S.setdefault(fkey(M), {})[m] = [v, lo, hi]
            for r in ("cams", "aurora"):
                S.setdefault(r, {})[m] = list(rec.ref_issued(sample, t, 1, r, m, M))
    for t in ("37.5", "75"):
        for m in ("reach_pct", "correct_pct"):
            for s in (fkey(M), "cams", "aurora"):
                v = systems[f"thr{t}"]["lead1"]["World"][s][m]
                base = f"results/world/{sample}/thr{t}/lead1/quantities"
                keys = ([f"{base}/issued:{m}/{M}/value", f"{base}/issued:{m}/{M}/{rec.ikey}"] if s == fkey(M) else
                        [f"{base}/issued:{m}/{M}/value", f"minus {base}/diff:{s}:{m}/{M}/value",
                         f"interval: complements.json world/{sample}/thr{t}/lead1/issued/{s}/{m}"])
                bars.append(dict(threshold=t, metric=m, system=s, value=v[0], interval95=None if v[1] is None else v[1:],
                                 printed_on_the_bar=None if v[0] is None else f"{v[0]:.0f}", record_keys=keys))
    return systems, bars


SUPP_METS = ("reach_pct", "correct_pct", "hidden_warned_pct", "hidden_margin_warned_pct", "hidden_warned_M", "false_in_silent40_M",
             "warned_in_silent40_M", "false_per_hidden_warned")


def supp_fig_values(rec, sample, Ms):
    """the values analyses/hidden/v3/fig_hidden_warned_v3.py plots, in the layout of numbers_v3.json (systems/thr<t>/lead<h>/World/...)"""
    S = {}
    for t in THRS:
        for h in LEADS:
            where = f"world/{sample}/thr{t}/lead{h}"
            Q = rec.world(sample, t, h)["quantities"]
            D = S.setdefault(f"thr{t}", {}).setdefault(f"lead{h}", {}).setdefault("World", {})
            for M in Ms:
                D[fkey(M)] = {m: list(rec.system(Q, m, M, where)) for m in SUPP_METS}
            for r, _ in REFS_W:
                D[r] = {m: list(rec.ref_issued(sample, t, h, r, m, Ms[0])) for m in SUPP_METS}
            if h == 1:                                     # the tails of panel a, at Foehn's eight-member number warned
                D["equal_number_warned"] = {
                    r: dict(decision_threshold_factor=rec.factor(sample, t, h, Ms[0], r),
                            **{m: list(rec.ref_moved(sample, t, h, r, m, Ms[0])) for m in MMET}) for r in REFS_M}
                D["decision_threshold_sweep"] = rec.comp(sample, t, h)["sweep"]
    return S


def legend_numbers(supp, bars, n_days):
    """every number quoted in the legend of the Supplementary warnings figure (supplementary.tex, fig:warned) and in Fig. 2f's
    text (main_text.tex line 103 and the caption of Fig. 2), with its key in fig2f.json and its new value"""
    def g(t, h, s, m):
        return supp[f"thr{t}"][f"lead{h}"]["World"][s][m][0]
    rows = []

    def add(where, what, t, h, s, m, d):
        v = g(t, h, s, m)
        rows.append(dict(where=where, quantity=what, key=f"supp_fig_warned/systems/thr{t}/lead{h}/World/{s}/{m}/0", value=v,
                         as_quoted=None if v is None else f"{v:.{d}f}"))
    for s, n in (("foehn_m8", "Foehn"), ("cams", "CAMS forecast"), ("aurora", "Aurora")):
        for m in ("reach_pct", "correct_pct"):
            add("fig:warned a, sentence 'One day ahead above 75'", f"{n}, {m}, > 75, 1 day", "75", 1, s, m, 1)
    for t in ("37.5", "75"):
        for m in ("reach_pct", "correct_pct"):
            add("fig:warned a, sentence 'Issued 3 days ahead'", f"Foehn, {m}, > {t}, 3 days", t, 3, "foehn_m8", m, 1)
    for s, n in (("cams", "CAMS forecast"), ("aurora", "Aurora")):
        for m in ("reach_pct", "correct_pct"):
            add("fig:warned a, sentence 'Issued 3 days ahead'", f"{n}, {m}, > 37.5, 1 day", "37.5", 1, s, m, 1)
    for s, n in (("foehn_m8", "Foehn at 1 km"), ("foehn_40km", "mean of the four members over 40 km cells")):
        for m in ("hidden_warned_pct", "hidden_margin_warned_pct"):
            add("fig:warned b", f"{n}, {m}, > 37.5, 1 day", "37.5", 1, s, m, 1)
    for s, n in (("cams", "CAMS forecast"), ("aurora", "Aurora"), ("foehn_m8", "Foehn")):
        for m in ("hidden_warned_M", "false_in_silent40_M"):
            add("fig:warned c, sentence 'A coarse grid can warn'", f"{n}, {m}, > 37.5, 1 day", "37.5", 1, s, m, 0)
    claims = {}
    for t in ("37.5", "75"):
        f3 = (g(t, 3, "foehn_m8", "reach_pct"), g(t, 3, "foehn_m8", "correct_pct"))
        claims[f"Foehn 3 days ahead lies above and to the right of the CAMS forecast and Aurora 1 day ahead, > {t}"] = all(
            f3[0] > g(t, 1, s, "reach_pct") and f3[1] > g(t, 1, s, "correct_pct") for s in ("cams", "aurora"))
    s75 = {s: (g("75", 1, s, "reach_pct"), g("75", 1, s, "correct_pct")) for s in ("foehn_m8", "cams", "aurora")}
    claims["> 75, 1 day: Foehn reaches more and is more often correct than both references"] = all(
        s75["foehn_m8"][i] > s75[s][i] for s in ("cams", "aurora") for i in (0, 1))
    b37 = {s: g("37.5", 1, s, "hidden_warned_pct") for s in ("foehn_m8", "foehn_40km", "cams", "aurora")}
    claims["> 37.5, 1 day: Foehn at 1 km warns a larger share of the hidden people than its four-member mean on 40 km cells"] = b37["foehn_m8"] > b37["foehn_40km"]
    claims["> 37.5, 1 day: Foehn warns fewer people wrongly where the 40 km mean is below than the CAMS forecast and Aurora"] = all(
        g("37.5", 1, "foehn_m8", "false_in_silent40_M") < g("37.5", 1, s, "false_in_silent40_M") for s in ("cams", "aurora"))
    claims["> 37.5, 1 day: Foehn warns more hidden people (millions) than the CAMS forecast and Aurora"] = all(
        g("37.5", 1, "foehn_m8", "hidden_warned_M") > g("37.5", 1, s, "hidden_warned_M") for s in ("cams", "aurora"))
    fig2 = [dict(where="main_text.tex line 103 and Fig. 2 caption (panel f)", quantity=f"{b['system']}, {b['metric']}, > {b['threshold']}, 1 day",
                 key=f"systems/thr{b['threshold']}/lead1/World/{b['system']}/{b['metric']}/0", value=b["value"], as_quoted=b["printed_on_the_bar"])
            for b in bars]
    return dict(supplementary_figure=rows, claims_of_the_legend=claims, fig2f_and_main_text=fig2,
                rule="Foehn warns when at least 4 of its 8 members exceed the threshold (legend: 'at least two of its four members')",
                n_start_days=n_days)


# ------------------------------------------------------------------------------------------------ text
def interval_text(rec, sample):
    info = rec.world(sample, THRS[0], LEADS[0])["info"]
    boot = ("paired moving-block bootstrap over the start days (blocks of 4 consecutive forecasts, the seen start days removed "
            "from every resample)")
    if rec.design_note():
        return (f"95% intervals: {boot}, not widened: under this noise design the noise is drawn anew at every start day, so its "
                f"draw is part of the day-to-day sampling (A16 section 4, point 3).")
    if "30 seen start days removed" not in info.get("bootstrap", "") and sample == "conf149":
        raise SystemExit("unexpected bootstrap description in the record: " + info.get("bootstrap", ""))
    return (f"95% intervals: {boot}, with the half-widths of the values of Foehn with 8, 16 and 4 members widened by "
            f"sqrt(1 + V_draw/V_boot) for the single draw of the members' noise fields, which does not average out over days "
            f"(A14 section 4c); the references and the two rows built on the mean of the four members carry the plain bootstrap "
            f"interval of the same resamples (the draw of the four fields is not in their tables).")


def sample_text(rec, sample):
    n, n_all = rec.n_days(sample), rec.n_days("all179")
    return (f"the {n} start days of 2022 not used when the sixteen-member rule was designed (the other {n_all - n} of the {n_all} "
            f"start days were), every system on the same days")


def members_text():
    return ("Foehn warns a resident when at least 4 of its 8 members exceed the threshold in the resident's grid cell (eight members: "
            "the exact expected value of eight members drawn without replacement from the sixteen members of the run); beside it, on "
            "the same days, sixteen members (at least 8 of 16) and the four members of the evaluation run (at least 2 of 4).")


def tables(rec, sample):
    st, mt, it = sample_text(rec, sample), members_text(), interval_text(rec, sample)
    w = warned_rows(rec, sample, PRINT_SYSTEMS)
    m = matched_rows(rec, sample, PRINT_SYSTEMS)
    g = regions_rows(rec, sample, M8)
    ntr = sum(1 for f in {x["cell"]: x for x in rec.flagged}.values() if "truncated" in f["status"] and "/eqn:" in f["cell"])
    warned = dict(label="warned", part="", what="complete table: 1 and 3 days ahead, Foehn with eight members (sixteen and four beside), "
                  "every count and its 95% interval",
                  desc=("Complete Supplementary Table: people warned a day by each system at the 1 km grid cells with at least 100 residents, "
                        f"for three thresholds (15, 37.5 and 75 µg m-3) and two lead times (1 and 3 days), over {st}. {mt} The ensemble mean "
                        "and the mean over 40 km cells are those of the four members of the evaluation run. The start-day 1 km estimate kept "
                        "unchanged (persistence) warns a resident when the estimate of the start day in their grid cell exceeds the threshold; "
                        "the perfect 40 km forecast warns where the mean of the 1 km estimates over the 40 km cell exceeds it (it warns no "
                        "hidden person). Reach, share of the exposed people who are warned; correct, share of the warned people who are "
                        f"exposed. {it}"),
                  cols=list(WCOLS), rows=[r for _, _, r in w])
    matched = dict(label="matched", part="", what="complete table: 1 and 3 days ahead, the references at the number of people warned by "
                   "Foehn with eight members (sixteen and four beside), every 95% interval",
                   desc=("Complete Supplementary Table: the reference forecasts made to warn as many people a day as Foehn (each warns a "
                         f"resident when its value exceeds the factor times the threshold), over {st}, for three thresholds "
                         "(15, 37.5 and 75 µg m-3) and two lead times. One block of rows for Foehn with eight members (at least 4 of 8), "
                         "then the same with sixteen (at least 8 of 16) and four members (at least 2 of 4), each with its own factor. The "
                         "factor is found on the reference's pooled curve of people warned against the factor (steps of 0.001, linear "
                         "inside one step) and found again in each bootstrap resample within 0.25 of it; the reference's values are read "
                         f"on that curve. {it.replace(' and the two rows built on the mean of the four members', '').replace('the references carry', 'the references (their own values at the factor) carry').replace('of the values of Foehn with 8, 16 and 4 members', 'of the differences')}"
                         + (f" The intervals of {ntr} differences are conditional on a crossing within 0.25 of the point factor (the "
                            "resamples without one are dropped, as in the record)." if ntr else "")),
                   cols=list(MCOLS), rows=[r for *_, r in m])
    regions = dict(label="regions", part="", what="complete table: blocks of 10 km, margin, 3 days ahead, Foehn with eight members, every "
                   "95% interval",
                   desc=("Complete Supplementary Table: hidden people by region above 37.5 µg m-3 (regions are groups of countries that "
                         "partition the world). Counts and shares of hidden people: all 1 km grid cells with residents, 365 days of 2022 "
                         "(no forecast). Warnings: grid cells with at least 100 residents, over "
                         f"{st}; Foehn warns when at least 4 of its 8 members exceed the threshold. {it.replace(' and the two rows built on the mean of the four members', '')}"),
                   cols=list(RCOLS), rows=[r for _, r in g])
    return warned, matched, regions


# ------------------------------------------------------------------------------------------------ LaTeX (printed tables)
def tnum(x, d):
    if x is None:
        return "--"
    neg = str(x).startswith("-")
    s = f"{abs(x):,d}" if isinstance(x, int) else f"{abs(x):,.{d}f}"
    return ("\\textminus{}" if neg else "") + s.replace(",", "{,}")


def tex_text(s):
    s = re.sub(r"(\d),(\d)", r"\1{,}\2", s)
    return s.replace(" µg m-3", "\\,\\textmu g\\,m$^{-3}$").replace("%", "\\%").replace("Foehn", "\\model{}")


def tex_table(caption, label, tabular):
    """tools/si_tables_portrait.py: table() and fit(), the typesetting of the printed tables"""
    return ("\\begin{table}[p]\n\\centering\n\\caption{" + caption + "}\n\\label{tab:" + label + "}\n"
            + "\\fitwidth{%\n\\footnotesize%\n\\setlength{\\tabcolsep}{4.5pt}%\n" + tabular + "}\n\\end{table}\n")


def tex_all(W, Mt, Rg, prov):
    col = {name: i for i, name in enumerate(W["cols"])}
    n = prov["n_start_days"]["conf149"]; n_all = prov["n_start_days"]["all179"]
    why = (f"the {n} start days of 2022 not used when the sixteen-member rule was designed (the other {n_all - n} were used then), "
           f"every system on the same days")
    stored = prov["design_note_of_the_record"] is None
    keys = {lab: k for k, lab in [(M, FOEHN[M]) for M in PRINT_SYSTEMS] + [(r, l) for r, l in REFS_W]}
    L = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
         " & Warned & Reach & Correct & \\multicolumn{2}{c}{Hidden people warned} & Wrongly warned, \\\\",
         " & (millions) & (\\%) & (\\%) & (millions) & (\\%) & 40\\,km mean below \\\\", "\\midrule"]
    k = 0
    for r in W["rows"]:
        if ", 1 day ahead" not in r[0]:
            continue
        if r[1] == FOEHN[M8]:
            m = re.fullmatch(r"(Above .*?), 1 day ahead: (.*? million hidden), [\d\.,]+ million hidden with a 10% margin", r[0])
            if k:
                L.append("\\addlinespace")
            k += 1
            L.append("\\multicolumn{7}{@{}l}{\\textit{%s: %s}} \\\\" % (tex_text(m.group(1)), tex_text(m.group(2))))
        L.append(" & ".join([TEXLAB[keys[r[1]]], tnum(r[col["warned (millions a day)"]], 0), tnum(r[col["reach (%)"]], 1),
                             tnum(r[col["correct (%)"]], 1), tnum(r[col["hidden people warned (millions a day)"]], 1),
                             tnum(r[col["hidden people warned (%)"]], 1), tnum(r[col[WNAMES[7]]], 0)]) + " \\\\")
    assert k == 3
    L += ["\\bottomrule", "\\end{tabular}"]
    cap_w = ("\\textbf{Who is warned, 1 day ahead.} People a day (millions) warned by each system at the 8.1 million 1\\,km grid cells "
             f"with at least 100 residents, for three thresholds; means over {why}. \\model{{}} warns a resident when at least 4 of its 8 "
             "members exceed the threshold in their grid cell (the expected value of eight members drawn from the sixteen of the run); "
             "the same with sixteen members (at least 8 of 16) and with the four members of the evaluation (at least 2 of 4) is given "
             "below it, and the ensemble mean and the mean over 40\\,km cells are those of the four members. Reach, share of the exposed "
             "people who are warned; Correct, share of the warned people who are exposed. Bias-corrected: the CAMS forecast mapped "
             "towards the 1\\,km estimates in each 0.25$^\\circ$ cell and calendar month (fitted on 2017--2020), every resident receiving "
             "the cell value or this value distributed by the climatological 1\\,km pattern of the estimates. Climatology alone: a "
             "resident is warned when the climatology of the 1\\,km estimates (2017--2020) in their grid cell exceeds the threshold. "
             "Start-day estimate: a resident is warned when the 1\\,km estimate of the start day in their grid cell exceeds the "
             "threshold. Last column: people wrongly warned (millions) where the mean of the 40\\,km cell is below the threshold. The "
             "95\\% intervals, the values 3 days ahead and the other counts are given in the Source Data file.")
    tw = tex_table(cap_w, "warned", "\n".join(L))
    mc = {name: i for i, name in enumerate(Mt["cols"])}
    L = ["\\begin{tabular}{@{}lrrrrcc@{}}", "\\toprule",
         " & Factor on & Reach & Correct & Hidden warned & \\multicolumn{2}{c}{\\model{} minus reference (points)} \\\\",
         "Reference & the threshold & (\\%) & (\\%) & (millions) & reach & correct \\\\", "\\midrule"]
    k, head = 0, None
    rkey = {lab: r for r, lab in REFS_W}
    for r in Mt["rows"]:
        m = re.fullmatch(r"(Above .*?), 1 day ahead: Foehn, at least 4 of 8 members, warns (.*?) million people a day, reach (.*?)%, "
                         r"correct (.*?)%, (.*?) million hidden people warned", r[0])
        if m is None:
            continue
        if r[0] != head:
            head = r[0]
            if k:
                L.append("\\addlinespace")
            k += 1
            L.append("\\multicolumn{7}{@{}l}{\\textit{%s: \\model{}, at least 4 of 8 members, warns %s million people a day (reach %s\\%%, "
                     "correct %s\\%%, %s million hidden)}} \\\\" % tuple(tex_text(x) for x in m.groups()))
        iv = lambda name: "%s [%s, %s]" % tuple(tnum(r[mc[c]], 1) for c in tri(name))   # noqa: E731
        L.append(" & ".join([TEXLAB[rkey[r[1]]], tnum(r[mc["factor on the threshold"]], 3), tnum(r[mc["reach (%)"]], 1),
                             tnum(r[mc["correct (%)"]], 1), tnum(r[mc["hidden people warned (millions a day)"]], 1),
                             iv("Foehn minus reference: reach (points)"), iv("Foehn minus reference: correct (points)")]) + " \\\\")
    assert k == 3
    L += ["\\bottomrule", "\\end{tabular}"]
    cap_m = ("\\textbf{The reference forecasts at the number of people warned by \\model{}, 1 day ahead.} Each reference warns a resident "
             "when its value exceeds the factor times the threshold; the factor is the one for which it warns as many people a day as "
             f"\\model{{}} with eight members (at least 4 of 8) over {why}; it is found to 0.001 on the reference's curve of the people it "
             "warns against the factor (linear between two neighbouring factors), on which the reference is read (a choice made in "
             "hindsight, which favours the reference). Start-day estimate: the 1\\,km estimate of the start day kept unchanged. Hidden "
             "people warned in millions a day; differences with 95\\% intervals over start days (paired"
             + ("; widened for the single draw of the members' noise fields, which does not average out over days" if stored else "")
             + "). The same comparisons with sixteen and four members, the values 3 days ahead and the other intervals are given in "
             "the Source Data file.")
    tm = tex_table(cap_m, "matched", "\n".join(L))
    rc = {name: i for i, name in enumerate(Rg["cols"])}
    L = ["\\begin{tabular}{@{}lcrrrrcc@{}}", "\\toprule",
         " & Hidden & Share of & Share of & \\multicolumn{2}{c}{Hidden warned} & Reach & Correct \\\\",
         " & (millions & the world's & the region's & \\multicolumn{2}{c}{1 day ahead (\\%)} & (\\%) & (\\%) \\\\",
         "Region & a day) & hidden (\\%) & exposed (\\%) & \\model{} & CAMS & \\model{} / CAMS & \\model{} / CAMS \\\\", "\\midrule"]
    for r in Rg["rows"]:
        hp = "%s [%s, %s]" % tuple(tnum(r[rc[c]], 1) for c in tri("hidden people (millions a day)"))
        L.append(" & ".join([r[0], hp, tnum(r[rc["share of the world's hidden people (%)"]], 1),
                             tnum(r[rc["share of the region's exposed people (%)"]], 1),
                             tnum(r[rc["hidden people warned by Foehn, 1 day ahead (%)"]], 1),
                             tnum(r[rc["hidden people warned by the CAMS forecast, 1 day ahead (%)"]], 1),
                             tnum(r[rc["reach of Foehn (%)"]], 1) + " / " + tnum(r[rc["reach of the CAMS forecast (%)"]], 1),
                             tnum(r[rc["correct warnings of Foehn (%)"]], 1) + " / " + tnum(r[rc["correct warnings of the CAMS forecast (%)"]], 1)])
                 + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    cap_r = ("\\textbf{Hidden people by region, above 37.5\\,\\textmu g\\,m$^{-3}$.} Regions are groups of countries that partition the "
             "world. Columns 2--4: all 1\\,km grid cells with residents, 365 days of 2022 (95\\% interval in brackets). Columns 5--8: "
             f"grid cells with at least 100 residents, 1 day ahead, {why}; \\model{{}} warns when at least 4 of its 8 members exceed the "
             "threshold; CAMS, the CAMS forecast. The counts for blocks of 10\\,km and with a margin, the values 3 days ahead and the "
             "other intervals are given in the Source Data file.")
    tr = tex_table(cap_r, "regions", "\n".join(L))
    return tw, tm, tr


def column_keys():
    """the source of every column (text for provenance.json and NOTES.md)"""
    wk = {"threshold, lead time and people exposed": "exposed: regions/<s>/thr<t>/lead<h>/regions/World/exposed_M; hidden: "
          "issued:hidden_warned_M / issued:hidden_warned_pct x 100 of M8; with a 10% margin: issued:hidden_margin_warned_M / "
          "issued:hidden_margin_warned_pct x 100 (world/<s>/thr<t>/lead<h>/quantities)", "system": "label"}
    for (m, d, wci), name in zip(WMET, ["warned (millions a day)"] + list(WNAMES)):
        wk[name] = (f"Foehn: issued:{m}/<M>/value; references: issued:{m}/M8_liberal/value - diff:<ref>:{m}/M8_liberal/value; "
                    f"foehn_mean, perfect_40km: complements world/<s>/thr<t>/lead<h>/issued/<row>/{m}/0 ({d} decimals)")
        if wci:
            wk[name + ", 95% interval low"] = wk[name + ", 95% interval high"] = (
                f"Foehn: issued:{m}/<M>/interval95; every other row: complements world/<s>/thr<t>/lead<h>/issued/<row>/{m}/1,2")
    mk = {"threshold, lead time and Foehn": "issued:warned_M, reach_pct, correct_pct, hidden_warned_M of Foehn with <M>",
          "reference": "label", "factor on the threshold": "info/<M>/eqn:<ref>/factor (3 decimals)"}
    for m, n in zip(MMET, MNAMES):
        mk[n] = f"issued:{m}/<M>/value - eqn:<ref>:{m}/<M>/value (1 decimal)"
        mk[n + ", 95% interval low"] = mk[n + ", 95% interval high"] = f"complements world/<s>/thr<t>/lead<h>/moved/<M>/<ref>/own/{m}/1,2"
        dn = "Foehn minus reference: " + n.replace("(%)", "(points)")
        mk[dn] = f"eqn:<ref>:{m}/<M>/value"
        mk[dn + ", 95% interval low"] = mk[dn + ", 95% interval high"] = f"eqn:<ref>:{m}/<M>/interval95"
    rk = {"region": "label (region G:<name> of the regional record)"}
    for c in RCOLS_TRUTH:
        rk[c] = "copied from source_data/supp_tables_full/regions.json (truth only: all 1 km cells, 365 days, no forecast)"
    for h in (1, 3):
        n = f"hidden people warned by Foehn, {plural(h)} (%)"
        rk[n] = f"regions/<s>/thr37.5/lead{h}/regions/G:<region>/quantities/issued:hidden_warned_pct/M8_liberal/value"
        rk[n + ", 95% interval low"] = rk[n + ", 95% interval high"] = rk[n].replace("/value", "/interval95")
    n = "hidden people warned by the CAMS forecast, 1 day ahead (%)"
    rk[n] = "lead1 issued:hidden_warned_pct/M8_liberal/value - diff:cams:hidden_warned_pct/M8_liberal/value"
    rk[n + ", 95% interval low"] = rk[n + ", 95% interval high"] = "complements regions/<s>/lead1/G:<region>/cams/hidden_warned_pct/1,2"
    for m, a, b in (("reach_pct", "reach of Foehn (%)", "reach of the CAMS forecast (%)"),
                    ("correct_pct", "correct warnings of Foehn (%)", "correct warnings of the CAMS forecast (%)")):
        rk[a] = f"lead1 issued:{m}/M8_liberal/value"
        rk[b] = f"lead1 issued:{m}/M8_liberal/value - diff:cams:{m}/M8_liberal/value"
    return dict(warned=wk, matched=mk, regions=rk)


def check_columns():
    for name, cols in (("warned", WCOLS), ("matched", MCOLS), ("regions", RCOLS)):
        fz = json.loads((FROZEN / f"{name}.json").read_text(encoding="utf-8"))["cols"]
        if fz != list(cols):
            raise SystemExit(f"columns of {name} differ from the frozen file")


def main():
    check_columns()
    rec = Record("interval95")
    rec.check_days()
    warned, matched, regions = tables(rec, "conf149")
    systems, bars = fig2f(rec, "conf149", M8)
    beside = {M: fig2f(rec, "conf149", M)[1] for M in (M16, M4)}
    sf = supp_fig_values(rec, "conf149", PRINT_SYSTEMS)
    rec.assert_checks()
    nulls = Counter()
    for name, tab in (("warned", warned), ("matched", matched), ("regions", regions)):
        for r in tab["rows"]:
            for c, v in zip(tab["cols"], r):
                if v is None:
                    nulls[f"{name}: {c}"] += 1
    fig = dict(
        what=("Fig. 2f (figures_new/fig2_v3/fig_hidden_v4.py, panel f) on the confirmatory sample: share of the people exposed who are "
              "warned (reach_pct) and share of the warnings that are correct (correct_pct), World, 1 day ahead, above 37.5 and 75 "
              "µg m-3, Foehn with eight members (at least 4 of 8), the CAMS forecast and Aurora on the same days. 'systems' has the "
              "layout the figure reads; 'bars' lists the twelve heights with their keys."),
        design=LABEL, record=str(rec.paths[0]), record_sha256=rec.sha256[0], complements_sha256=rec.csha, sample="conf149",
        sample_description=rec.W["samples"]["conf149"], n_start_days=rec.n_days("conf149"), foehn=FOEHN[M8], interval_key=rec.ikey,
        systems=systems, bars=bars,
        beside={FOEHN[M]: [b for b in beside[M] if b["system"].startswith("foehn")] for M in beside},
        supp_fig_warned=dict(
            what=("the values that analyses/hidden/v3/fig_hidden_warned_v3.py plots, on the same sample, in the layout of numbers_v3.json "
                  "(systems/thr<t>/lead<h>/World/<system>/<metric> = [value, low, high]; equal_number_warned and decision_threshold_sweep "
                  "at lead 1): Foehn with 8 (foehn_m8, headline), 16 and 4 members; the references and the rows of the four members' mean "
                  "with the intervals of complements.json; the references at Foehn's eight-member number of people warned (factor, points "
                  "on the curve, intervals) and the sweep of cams_qm and cams_bcsd (v3's tables at the factors of version 2)"),
            systems=sf),
        legend=legend_numbers(sf, bars, rec.n_days("conf149")))
    prov = dict(design=LABEL, design_note_of_the_record=rec.design_note(),
                records=[dict(path=str(p), sha256=s) for p, s in zip(rec.paths, rec.sha256)],
                complements=dict(path=str(rec.cpath), sha256=rec.csha, inputs=rec.C["inputs"], code=rec.C["code"], checks=rec.C["checks"]),
                frozen_regions=dict(path=str(FROZEN / "regions.json"), sha256=hashlib.sha256((FROZEN / "regions.json").read_bytes()).hexdigest(),
                                    columns_copied=list(RCOLS_TRUTH)),
                interval_key=rec.ikey, samples=rec.W["samples"], n_start_days={s: rec.n_days(s) for s in ("conf149", "all179")},
                bootstrap_of_the_record=rec.world("conf149", THRS[0], LEADS[0])["info"]["bootstrap"],
                columns=column_keys(), empty_cells=dict(nulls), keys_absent_in_the_record=sorted(set(rec.absent)),
                interval_statuses_of_the_cells_used=dict(rec.status),
                intervals_flagged=list({(f["cell"], f["status"]): f for f in rec.flagged}.values()),
                consistency_checks=dict(rec.maxdev), tolerance_of_the_complement_checks=TOL,
                record_not_delivered=dict(k3=rec.W.get("k3"), relative_economic_value=rec.W.get("relative_economic_value")))
    OUT.mkdir(exist_ok=True)
    for name, obj in (("warned.json", warned), ("matched.json", matched), ("regions.json", regions), ("fig2f.json", fig),
                      ("provenance.json", prov)):
        safe_write(OUT / name, dumps(obj))
        print("wrote", OUT / name)
    J = {t: json.loads((OUT / f"{t}.json").read_text(encoding="utf-8")) for t in ("warned", "matched", "regions")}   # tex from the files
    for name, text in zip(("tab_warned.tex", "tab_matched.tex", "tab_regions.tex"), tex_all(J["warned"], J["matched"], J["regions"], prov)):
        safe_write(OUT / name, text)
        print("wrote", OUT / name)
    print("checks:", json.dumps(dict(rec.maxdev), indent=1))
    print("absent keys:", len(set(rec.absent)), "| flagged intervals:", len(prov["intervals_flagged"]), "| empty cells:", dict(nulls))
    for b in bars:
        print(f"fig2f > {b['threshold']} {b['metric']:<12} {b['system']:<9} {b['value']:.2f} (bar {b['printed_on_the_bar']})")


if __name__ == "__main__":
    main()
# end of build_warn8.py
