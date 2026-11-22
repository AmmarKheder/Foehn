#!/usr/bin/env python3
"""The proof of the key mapping of build_warn8.py. The SAME functions (warned_rows, matched_rows, regions_rows, fig2f,
supp_fig_values), run with M = 4 (record key M4_stored, 'at least 2 of 4') on ALL 179 start days (sample all179; the
complements of complements.json on the same sample), must reproduce cell by cell, at the printed precision, the frozen
four-member tables (stored design: source_data/supp_tables_full/*.json; FOEHN_DESIGN=ar: source_data/supp_tables_full_ar/*.json,
skipped if that folder does not exist), the bars of the current Fig. 2f (analyses/hidden/v2/results[_ar]/numbers_v2.json) and
the values of the current Supplementary warnings figure (analyses/hidden/v3/results[_ar]/numbers_v3.json).

Two runs: 'interval95' is exactly the code path of the printed tables (the record's interval of record for Foehn and the
paired differences); 'boot95' is the same with the record's plain bootstrap percentile interval. Every cell is reported as
equal / different (both values) / not in the record / relabelled (a row label changed on purpose) / copied (the truth-only
columns of the regions table, copied from the frozen file: not a test). Every difference is investigated:
  - interval cells that differ only under 'interval95': the record's interval of record is checked to be the bootstrap
    interval widened by its own factor sqrt(1 + V_draw/V_boot) (A14 4c), and the same cell under 'boot95' to equal the frozen one;
  - points of the 'matched' table: the record's reference point is checked against v3's own curve point
    (numbers_v3.json check_point_from_the_curve) and the frozen value against v3's field point (classE README R-3);
  - anything else is reported as 'cause NOT established'.
    python -B test_identity.py            (FOEHN_DESIGN=ar for the other design) -> out_<label>/identity.json"""
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_warn8 as B  # noqa: E402

SFX = "" if B.LABEL == "stored" else f"_{B.LABEL}"
FROZEN = B.N / "source_data" / f"supp_tables_full{SFX}"
V2 = B.N / "analyses" / "hidden" / "v2" / f"results{SFX}" / "numbers_v2.json"
V3 = B.N / "analyses" / "hidden" / "v3" / f"results{SFX}" / "numbers_v3.json"
SAMPLE, M = "all179", B.M4
FKEY = {B.M4: "foehn_p50"}                                           # Foehn's key in numbers_v2/v3 (at least 2 of 4)


def eq(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if a is None or b is None:
        return a is None and b is None
    return a == b


def close(a, b, tol=1e-9):
    return B.fin(a) and B.fin(b) and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


class Cells:
    def __init__(self):
        self.cells, self.n = [], Counter()

    def add(self, table, block, row, col, status, rec=None, frozen=None, why=None):
        c = dict(table=table, block=block, row=row, column=col, status=status, record=rec, frozen=frozen)
        if why:
            c["reason"] = why
        self.cells.append(c); self.n[(table, status)] += 1
        return c


def put(C, table, block, row, col, v, f, why_none):
    if v is None and f is not None:
        return C.add(table, block, row, col, "not in the record", None, f, why_none)
    return C.add(table, block, row, col, "equal" if eq(v, f) else "different", v, f)


# ------------------------------------------------------------------------------------------------ the record entry of a cell
def entry_of(rec, table, key, col):
    """the record entry an interval cell of ours comes from, for the A14 4c check (None for other cells)"""
    base = col.replace(", 95% interval low", "").replace(", 95% interval high", "")
    if table == "warned":
        (t, h), syskey = key
        if syskey != M:
            return None
        m = dict(zip(B.WNAMES, [x[0] for x in B.WMET[1:]])).get(base)
        return rec.world(SAMPLE, t, h)["quantities"][f"issued:{m}"][M] if m else None
    if table == "matched":
        (t, h), r = key
        for m, n in zip(B.MMET, B.MNAMES):
            if base == "Foehn minus reference: " + n.replace("(%)", "(points)"):
                return rec.world(SAMPLE, t, h)["quantities"][f"eqn:{r}:{m}"][M]
        return None
    if table == "regions":
        g = key
        for h in (1, 3):
            if base == f"hidden people warned by Foehn, {B.plural(h)} (%)":
                return rec.region(SAMPLE, "37.5", h, f"G:{g}")["quantities"]["issued:hidden_warned_pct"][M]
    return None


def widened(e):
    """the interval of record of entry e is its bootstrap interval widened by its factor sqrt(1 + V_draw/V_boot)"""
    v, (blo, bhi), (lo, hi), f = e["value"], e["boot95"], e["interval95"], e.get("inflation")
    ok_f = B.fin(f) and B.fin(e.get("V_draw")) and B.fin(e.get("V_boot")) and close(f, math.sqrt(1 + e["V_draw"] / e["V_boot"]))
    return bool(ok_f and close(lo, v - (v - blo) * f) and close(hi, v + (bhi - v) * f)), f


# ------------------------------------------------------------------------------------------------ the three tables
def compare_tables(rec, frozen, C):
    """-> {(table, block, row, col): (our value, key for the investigation)}"""
    ours = {}
    fw = frozen["warned"]
    O = {(r[0].split(":")[0], B.FROZEN_LABEL[k]): (th, k, r) for th, k, r in B.warned_rows(rec, SAMPLE, [M])}
    for fr in fw["rows"]:
        blk, lab = fr[0].split(":")[0], fr[1]
        th_k = O.get((blk, lab))
        for i, col in enumerate(fw["cols"]):
            if th_k is None:
                C.add("warned", blk, lab, col, "not in the record", None, fr[i], "row absent"); continue
            (th, k, r) = th_k
            if i == 1:
                if r[1] == fr[1]:
                    C.add("warned", blk, lab, col, "equal", r[1], fr[1])
                else:
                    C.add("warned", blk, lab, col, "relabelled", r[1], fr[1], "label changed on purpose (truthful under the "
                          "eight-member headline)")
                continue
            put(C, "warned", blk, lab, col, r[i], fr[i], "empty in our row")
            ours[("warned", blk, lab, col)] = (r[i], (th, k))
    fm = frozen["matched"]
    O = {(r[0].split(":")[0], r[1]): (th, rr, r) for th, _, rr, r in B.matched_rows(rec, SAMPLE, [M], named=False)}
    for fr in fm["rows"]:
        blk, lab = fr[0].split(":")[0], fr[1]
        th, rr, r = O[(blk, lab)]
        for i, col in enumerate(fm["cols"]):
            put(C, "matched", blk, lab, col, r[i], fr[i], "empty in our row")
            ours[("matched", blk, lab, col)] = (r[i], (th, rr))
    fg = frozen["regions"]
    O = dict(B.regions_rows(rec, SAMPLE, M))
    for fr in fg["rows"]:
        g = fr[0]
        r = O.get(g)
        for i, col in enumerate(fg["cols"]):
            if r is None:
                C.add("regions", "37.5", g, col, "not in the record", None, fr[i], "region absent"); continue
            v = r[B.RCOLS.index(col)]
            if col in B.RCOLS_TRUTH:
                C.add("regions", "37.5", g, col, "copied", v, fr[i], "truth only (all 1 km cells, 365 days, no forecast): copied from "
                      "source_data/supp_tables_full/regions.json"); continue
            put(C, "regions", "37.5", g, col, v, fr[i], "empty in our row")
            ours[("regions", "37.5", g, col)] = (v, g)
    return ours


def investigate(rec, runs, v3):
    """the cause of every difference"""
    out = []
    for run, (C, ours) in runs.items():
        other = runs["boot95" if run == "interval95" else "interval95"][0]
        oc = {(c["table"], c["block"], c["row"], c["column"]): c for c in other.cells}
        for c in C.cells:
            if c["status"] != "different":
                continue
            k = (c["table"], c["block"], c["row"], c["column"])
            d = dict(run=run, **{x: c[x] for x in ("table", "block", "row", "column", "record", "frozen")})
            key = ours.get(k, (None, None))[1]
            e = entry_of(rec, c["table"], key, c["column"]) if key is not None else None
            if "95% interval" in c["column"] and e is not None:
                ok, f = widened(e)
                bo = oc[k]
                if run == "interval95" and ok and bo["status"] == "equal":
                    d.update(cause="A14 section 4c: the record's interval of record (interval95) is its bootstrap interval with the "
                                   "half-widths widened by sqrt(1 + V_draw/V_boot); the frozen table prints the plain bootstrap interval",
                             evidence=dict(inflation=f, V_draw=e["V_draw"], V_boot=e["V_boot"], boot95=e["boot95"], interval95=e["interval95"],
                                           identity_checked="interval95 = value -/+ (value - boot95) x inflation (1e-9)",
                                           same_cell_with_boot95=bo["record"]))
                    out.append(d); continue
            if c["table"] == "matched" and v3 is not None and "95% interval" not in c["column"]:
                (t, h), r = key
                E = v3["systems"][f"thr{t}"][f"lead{h}"]["World"]["equal_number_warned"][r]
                cp = E.get("check_point_from_the_curve", {})
                Q = rec.world(SAMPLE, t, h)["quantities"]
                for m, n in zip(B.MMET, B.MNAMES):
                    dn = "Foehn minus reference: " + n.replace("(%)", "(points)")
                    if c["column"] in (n, dn):
                        ref_rec = Q[f"issued:{m}"][M]["value"] - Q[f"eqn:{r}:{m}"][M]["value"]
                        field = E[m][0]
                        if c["column"] == n:
                            exp_frozen, exp_rec = B.pnum(field, 1), B.pnum(ref_rec, 1)
                        else:
                            exp_frozen, exp_rec = B.pnum(E[f"{FKEY[M]}_minus_{m}"][0], 1), B.pnum(Q[f"eqn:{r}:{m}"][M]["value"], 1)
                        if close(ref_rec, cp.get(m), 1e-9) and exp_frozen == c["frozen"] and exp_rec == c["record"]:
                            d.update(cause="the frozen table prints v3's field point (the reference re-evaluated from the fields at the "
                                           "exact factor); the record reads the reference on its factor curve (classE README R-3), which "
                                           "equals v3's own check point from the curve; the two round differently at the printed precision",
                                     evidence=dict(record_reference_point=ref_rec, v3_check_point_from_the_curve=cp.get(m),
                                                   v3_field_point=field, factor_record=rec.factor(SAMPLE, t, h, M, r),
                                                   factor_v3=E.get("decision_threshold_factor")))
                        break
                if "cause" in d:
                    out.append(d); continue
            d["cause"] = "cause NOT established"
            out.append(d)
    return out


# ------------------------------------------------------------------------------------------------ the figures
def fig2f_identity(rec):
    if not V2.exists():
        return dict(skipped=f"{V2} does not exist")
    S2 = json.loads(V2.read_text(encoding="utf-8"))["systems"]
    _, bars = B.fig2f(rec, SAMPLE, M)
    _, new = B.fig2f(rec, "conf149", B.M8)
    rows, n = [], Counter()
    for b, nb in zip(bars, new):
        s2 = FKEY[M] if b["system"].startswith("foehn") else b["system"]
        cur = S2[f"thr{b['threshold']}"]["lead1"]["World"][s2][b["metric"]][0]
        st = ("equal" if f"{b['value']:.0f}" == f"{cur:.0f}" and f"{b['value']:.1f}" == f"{cur:.1f}" else "different")
        n[st] += 1
        rows.append(dict(threshold=b["threshold"], metric=b["metric"], system=s2, status=st, record_M4_all179=b["value"],
                         current_figure_numbers_v2=cur, abs_difference=abs(b["value"] - cur),
                         current_figure_bar=f"{cur:.0f}", new_conf149_M8=nb["value"], new_bar=nb["printed_on_the_bar"]))
    return dict(what=f"the twelve bars of Fig. 2f: the code path of fig2f.json with M = 4 on all 179 start days against {V2} "
                     "(the file the current figure reads), compared at the precision printed on the bars (0 decimals) and in its "
                     "log (1 decimal); 'new_*' are the bars of out_<label>/fig2f.json (eight members, conf149)",
                summary=dict(n), bars=rows)


def supp_identity(rec, v3):
    """every value the Supplementary warnings figure reads: supp_fig_values with M = 4 on all179 against numbers_v3.json,
    at 1 decimal (the precision of the figure's log and legend) and at full precision"""
    if v3 is None:
        return dict(skipped=f"{V3} does not exist")
    rec.ikey = "boot95"                                                  # the figure's whiskers are plain bootstrap intervals in v3
    ours = B.supp_fig_values(rec, SAMPLE, [M])
    rec.ikey = "interval95"
    S3 = v3["systems"]
    rows, n, worst = [], Counter(), Counter()
    names = {B.fkey(M): FKEY[M]}

    def cmp(what, a, b, kind):
        st = "equal" if (a is None and b is None) or (B.fin(a) and B.fin(b) and f"{a:.1f}" == f"{b:.1f}") else "different"
        n[(kind, st)] += 1
        if B.fin(a) and B.fin(b):
            worst[kind] = max(worst[kind], abs(a - b))
        if st == "different":
            rows.append(dict(what=what, ours=a, numbers_v3=b, kind=kind))
    for t in B.THRS:
        for h in B.LEADS:
            D, E = ours[f"thr{t}"][f"lead{h}"]["World"], S3[f"thr{t}"][f"lead{h}"]["World"]
            for s in (B.fkey(M), "foehn_40km", "cams", "aurora", "cams_qm", "cams_bcsd", "clim_1km"):
                for m in ("reach_pct", "correct_pct", "hidden_warned_M", "false_in_silent40_M", "warned_in_silent40_M", "false_per_hidden_warned"):
                    cmp(f"{t}/{h}/{s}/{m}", D[s][m][0], E[names.get(s, s)][m][0], "points")
                if t == "37.5" and s in (B.fkey(M), "foehn_40km"):
                    for m in ("hidden_warned_pct", "hidden_margin_warned_pct"):
                        for i in range(3):
                            cmp(f"{t}/{h}/{s}/{m}/{i}", D[s][m][i], E[names.get(s, s)][m][i], "panel b bars and whiskers")
            if h == 1:
                for s in ("cams_qm", "cams_bcsd"):
                    q, e = D["equal_number_warned"][s], E["equal_number_warned"][s]
                    cmp(f"{t}/{s}/factor", q["decision_threshold_factor"], e["decision_threshold_factor"], "tails: factor")
                    for m in ("reach_pct", "correct_pct"):
                        cmp(f"{t}/{s}/eqn/{m}", q[m][0], e[m][0], "tails: end point (curve point against v3's field point)")
                    for mu, x in E["decision_threshold_sweep"][s].items():
                        for m in ("reach_pct", "correct_pct"):
                            cmp(f"{t}/{s}/sweep {mu}/{m}", D["decision_threshold_sweep"][s].get(mu, {}).get(m), x[m], "tails: sweep")
    return dict(what=f"supp_fig_values with M = 4 on all 179 start days against {V3} (the file fig_hidden_warned_v3.py reads); "
                     "intervals: the record's bootstrap (boot95), as v3 prints plain bootstrap intervals",
                summary={f"{k} {s}": v for (k, s), v in n.items()}, largest_abs_difference=dict(worst), different=rows)


def main():
    rec = B.Record("interval95")
    rec.check_days()
    res = dict(what="identity of the key mapping of build_warn8.py (module docstring)", design=B.LABEL, sample=SAMPLE, members=M,
               records=[dict(path=str(p), sha256=s) for p, s in zip(rec.paths, rec.sha256)], complements_sha256=rec.csha,
               frozen_dir=str(FROZEN))
    v3 = json.loads(V3.read_text(encoding="utf-8")) if V3.exists() else None
    if not FROZEN.exists():
        res["tables"] = f"SKIPPED: {FROZEN} does not exist"
        print(res["tables"])
    else:
        frozen = {t: json.loads((FROZEN / f"{t}.json").read_text(encoding="utf-8")) for t in ("warned", "matched", "regions")}
        res["frozen_sha256"] = {t: hashlib.sha256((FROZEN / f"{t}.json").read_bytes()).hexdigest() for t in frozen}
        runs = {}
        for ikey in ("interval95", "boot95"):
            rec.ikey = ikey
            C = Cells()
            runs[ikey] = (C, compare_tables(rec, frozen, C))
        rec.ikey = "interval95"
        diffs = investigate(rec, runs, v3)
        res["runs"] = {}
        for ikey, (C, _) in runs.items():
            summ = {}
            for t in ("warned", "matched", "regions"):
                e, d, a, r, cp = (C.n[(t, s)] for s in ("equal", "different", "not in the record", "relabelled", "copied"))
                summ[t] = dict(compared=e + d, equal=e, different=d, not_in_the_record=a, relabelled=r, copied=cp)
            summ["all"] = {k: sum(summ[t][k] for t in ("warned", "matched", "regions")) for k in summ["warned"]}
            res["runs"][ikey] = dict(what=("exactly the code path of the printed tables (interval of record)" if ikey == "interval95"
                                           else "the same with the record's bootstrap percentile interval in place of the interval of record"),
                                     summary=summ, cells=C.cells)
        res["differences"] = diffs
        res["differences_by_cause"] = dict(Counter(f"{d['run']} {d['table']}: {d['cause'][:60]}" for d in diffs))
        res["unexplained_differences"] = sum(d["cause"] == "cause NOT established" for d in diffs)
        for ikey in runs:
            print(ikey, json.dumps(res["runs"][ikey]["summary"]["all"]))
        print("differences by cause:", json.dumps(res["differences_by_cause"], indent=1))
        for d in diffs:
            if d["run"] == "boot95" or d["cause"] == "cause NOT established":
                print("  ", d["run"], d["table"], d["block"], "|", d["row"], "|", d["column"], "| record", d["record"], "frozen", d["frozen"],
                      "|", d["cause"][:80])
    res["fig2f"] = fig2f_identity(rec)
    print("fig2f:", res["fig2f"].get("summary", res["fig2f"]))
    res["supp_fig"] = supp_identity(rec, v3)
    print("supplementary figure:", json.dumps(res["supp_fig"].get("summary", res["supp_fig"]), indent=1),
          json.dumps(res["supp_fig"].get("largest_abs_difference", {}), indent=1))
    for d in res["supp_fig"].get("different", []):
        print("   supp:", d)
    B.OUT.mkdir(exist_ok=True)
    B.safe_write(B.OUT / "identity.json", B.dumps(res))
    print("wrote", B.OUT / "identity.json")


if __name__ == "__main__":
    main()
# end of test_identity.py
