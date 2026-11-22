"""Render a figure script of experiments/figures into review/figfix/out instead of its default folder (paper/nature/figs)."""
import os, runpy, sys
from pathlib import Path
FIG = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures")
sys.path.insert(0, FIG); os.chdir(FIG)
import fig_style, nature_style
out = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/review/figfix/out"))
if os.environ.get("FOEHN_DESIGN"):   # A16 rebuild: renders of a noise design go to natcomms/rebuild/figs_<D>
    out = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild")) / f"figs_{os.environ['FOEHN_DESIGN']}"
fig_style.OUT = out; nature_style.OUT = out
script = sys.argv[1]; sys.argv = [script] + sys.argv[2:]
runpy.run_path(os.path.join(FIG, script), run_name="__main__")
