#!/usr/bin/env python3
"""Render Fig. 7 (fig_smoke) for the polish pass, as N/rebuild/print/figs_letters/render_asold.py renders it, but into
N/rebuild/figs_show/polish/fig7 (or a folder under /tmp for the reproduction check), PDF only, never overwriting a file.

    cd N/rebuild/figs_show/polish/fig7
    FOEHN_DESIGN=ar MPLCONFIGDIR=<writable> OMP_NUM_THREADS=4 python -I -B render_polish.py fig_smoke_polish.py --suffix _polish
    FOEHN_DESIGN=ar ... python -I -B render_polish.py fig_smoke_polish.py --suffix _repro --out /tmp/...      (reproduction)

* experiments/figures goes first on sys.path and becomes the working directory (as render_asold.py);
* FOEHN_DESIGN=ar is required: final_model reads the '_ar' runs (the manuscript's figure is the 'ar' render);
* write guard: any write, rename, removal or new folder outside the output folder, /tmp, /dev and MPLCONFIGDIR is refused;
* the PDF is written to a hidden temporary file, fsync'ed, then renamed."""
import argparse
import os
import runpy
import sys
from pathlib import Path

FIG = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures")
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("script"); ap.add_argument("--suffix", required=True); ap.add_argument("--out")
a = ap.parse_args()
assert a.suffix.startswith("_") and len(a.suffix) > 1
assert os.environ.get("FOEHN_DESIGN") == "ar", "FOEHN_DESIGN=ar is required"
script = Path(a.script).resolve()
out = Path(a.out).resolve() if a.out else HERE
assert out == HERE or out.as_posix().startswith("/tmp/"), out
assert out.is_dir(), out
ALLOW = tuple(p.rstrip("/") + "/" for p in (str(out), "/tmp", "/dev", os.environ.get("MPLCONFIGDIR", "/tmp")))
WBITS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


def ok(p):
    return (os.path.abspath(os.fsdecode(p)) + "/").startswith(ALLOW)


def hook(ev, args):
    if ev == "open":
        p, mode, flags = args
        if p is None or isinstance(p, int):
            return
        w = bool(flags & WBITS) if isinstance(flags, int) else bool(mode and any(c in mode for c in "wax+"))
        if w and not ok(p):
            raise PermissionError(f"render_polish write guard: write {p} refused")
    elif ev in ("os.rename", "shutil.copyfile"):
        if not ok(args[1]):
            raise PermissionError(f"render_polish write guard: {ev} to {args[1]} refused")
    elif ev in ("os.remove", "os.rmdir", "os.truncate", "shutil.rmtree", "os.chmod", "os.utime"):
        if isinstance(args[0], (str, bytes, os.PathLike)) and not ok(args[0]):
            raise PermissionError(f"render_polish write guard: {ev} {args[0]} refused")
    elif ev == "os.mkdir":
        if not ok(args[0]) and not os.path.exists(args[0]):
            raise PermissionError(f"render_polish write guard: mkdir {args[0]} refused")


sys.addaudithook(hook)
sys.path.insert(0, FIG); os.chdir(FIG)
import nature_style  # noqa: E402


def save(fig, name):
    p = out / f"{name}{a.suffix}.pdf"
    if p.exists():
        raise FileExistsError(f"refuse to overwrite {p}")
    tmp = out / f".{name}{a.suffix}.pdf.tmp"
    fig.savefig(tmp, format="pdf")
    fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
    os.rename(tmp, p)
    print("saved", p)
    if os.environ.get("POLISH_PICKLE"):                       # layout check on the login node without re-reading the data
        import pickle
        from matplotlib.ticker import FixedFormatter, FuncFormatter
        for ax in fig.axes:                                    # geo_ticks' lambda formatters cannot be pickled: freeze their labels
            for axis in (ax.xaxis, ax.yaxis):
                if isinstance(axis.get_major_formatter(), FuncFormatter):
                    axis.set_major_formatter(FixedFormatter([t.get_text() for t in axis.get_ticklabels()]))
        with open(out / f".{name}{a.suffix}.pkl", "wb") as f:
            pickle.dump(fig, f)


nature_style.OUT = out
nature_style.save = save
print(f"render_polish: {script.name} design={os.environ['FOEHN_DESIGN']} out={out}")
sys.argv = [str(script)]
runpy.run_path(str(script), run_name="__main__")
