#!/usr/bin/env python3
"""Step 10 of REBUILD_PLAN 4c for FOEHN_DESIGN=<D>: the complete Supplementary Tables as JSON (the sheets STable<n> of
source_data/build_source_data.py) in source_data/supp_tables_full_<D>/. Run through run.py (write guard).
  warned.json, matched.json  byte copies of hidden/v3/results_<D>/{warned,matched}_v3.json (make_si_v3.py writes them in the
                             format of the frozen files; the stored supp_tables_full/{warned,matched}.json are byte copies of
                             the stored v3 files)
  regions.json, sens.json    tools/si_tables_portrait.py do_regions / do_sens (imported, unchanged) on the frozen typeset tables
                             of 2026-11-20 in which the tabular of that table is replaced by hidden/results_<D>/si_table_
                             {regions,sensitivity}.tex; on the stored runs this reproduces the stored JSON byte for byte
  away_b.json                byte copy of the stored file (Foehn-CAMS at the stations, class d: A16 does not change it)
  away_a.json                NOT written: the stored file was edited by hand after the freeze (a column title; ring-sample
                             station counts 1600 / 5399 where circularity/results/si_table.tex gives 1613 / 5412); the values of
                             the design are in circularity/results_<D>/si_table.tex
The manuscript is neither read for writing nor changed: si_tables_portrait.main() is never called."""
import os
import sys

D = os.environ["FOEHN_DESIGN"]
N = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms")
sys.path.insert(0, N + "/tools")
import si_tables_portrait as T  # noqa: E402

FULL0, OUT = N + "/source_data/supp_tables_full", N + f"/source_data/supp_tables_full_{D}"
os.makedirs(OUT, exist_ok=True)
T.FULL = OUT                                          # do_* write their JSON into T.FULL


def copy(src, dst):
    T.safe_write(dst, open(src, encoding="utf-8").read())
    print("copied", src, "->", dst)


s = open(FULL0 + "/tables_as_typeset_20261120.tex", encoding="utf-8").read()


def sub(s, label, gen):
    """the frozen block of tab:<label> with its tabular replaced by the tabular of the generated table <gen>"""
    blk, g = T.block(s, label), open(gen, encoding="utf-8").read()
    a, b = blk.index("\\begin{tabular}"), blk.index("\\end{tabular}")
    ga, gb = g.index("\\begin{tabular}"), g.index("\\end{tabular}")
    assert s.count(blk) == 1
    return s.replace(blk, blk[:a] + g[ga:gb] + blk[b:])


s = sub(s, "regions", N + f"/analyses/hidden/results_{D}/si_table_regions.tex")
s = sub(s, "sens", N + f"/analyses/hidden/results_{D}/si_table_sensitivity.tex")
T.do_regions(s)
T.do_sens(s)
copy(N + f"/analyses/hidden/v3/results_{D}/warned_v3.json", OUT + "/warned.json")
copy(N + f"/analyses/hidden/v3/results_{D}/matched_v3.json", OUT + "/matched.json")
copy(FULL0 + "/away_b.json", OUT + "/away_b.json")
print("not written: away_a.json (hand-edited after the freeze; see the header of this script)")
