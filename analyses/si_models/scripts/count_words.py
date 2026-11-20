#!/usr/bin/env python3
"""Word counts of the deliverables with the project's counter (build_test/wordcount.py): the note (heading and \\todo
excluded) and every new legend of legends_si.tex.    python3 count_words.py"""
import re, sys
from pathlib import Path
sys.path.insert(0, "/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/build_test")
import wordcount as wc
H = Path(__file__).resolve().parents[1]
s = wc.strip_comments((H / "note_european_benchmark.tex").read_text())
s = re.sub(r"\\snote\{[^}]*\}\{[^}]*\}", "", s)
s2 = wc.unwrap(s, ["todo"], "")
print("note, body without heading and \\todo:", wc.nwords(s2), "words | with the \\todo text:", wc.nwords(wc.unwrap(s, ["todo"])))
f = H / "legends_si.tex"
if f.exists():
    t = wc.strip_comments(f.read_text())
    for m in re.finditer(r"\\caption\s*\{", t):
        c = t[m.end():wc.braces(t, m.end() - 1) - 1]
        lab = re.search(r"\\label\{([^}]*)\}", t[m.end():m.end() + len(c) + 200])
        print("legend", lab.group(1) if lab else "?", ":", wc.nwords(c), "words")
