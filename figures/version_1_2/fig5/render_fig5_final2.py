#!/usr/bin/env python3
# COPY made 2026-11-21 of N/rebuild/figs_show/final/render_fig5_final.py, placed in N/rebuild/figs_show/final2: its output
# folder is its own folder (HERE), so the code is unchanged; only the names in the texts change (FIX:).
# ---- header of the copied script (kept) ----
# COPY made 2026-11-21 of N/rebuild/figs_show/polish/fig7/render_polish.py (itself modelled on figs_letters/render_asold.py).
# CHANGES marked FINAL:
#   FINAL-1  output folder = N/rebuild/figs_show/final (or a folder under /tmp for trial renders); --suffix optional;
#   FINAL-2  after the PDF, a 300 dpi PNG rasterised from that PDF by PyMuPDF (temporary file, fsync, rename);
#   FINAL-3  no pickle option.
"""Render Fig. 5 (fig_gc2_realism_final2.py) into N/rebuild/figs_show/final2,   (FIX:) as render_asold.py renders the manuscript's figures,
PDF + 300 dpi PNG, never overwriting a file.

    cd N/rebuild/figs_show/final2
    FOEHN_DESIGN=ar MPLCONFIGDIR=<writable> OMP_NUM_THREADS=4 python -I -B render_fig5_final2.py fig_gc2_realism_final2.py
    ... --out /tmp/<folder>        (trial render)

* experiments/figures goes first on sys.path and becomes the working directory (as render_asold.py);
* FOEHN_DESIGN=ar is required: final_model reads the '_ar' runs (the manuscript's figure is the 'ar' render);
* write guard: any write, rename, removal or new folder outside the output folder, /tmp, /dev and MPLCONFIGDIR is refused;
* each file is written to a hidden temporary file, fsync'ed, then renamed."""
import argparse
import os
import runpy
import sys
from pathlib import Path

FIG = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures")
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("script"); ap.add_argument("--suffix", default=""); ap.add_argument("--out")   # FINAL-1
a = ap.parse_args()
assert a.suffix == "" or (a.suffix.startswith("_") and len(a.suffix) > 1)
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
            raise PermissionError(f"render_fig5_final write guard: write {p} refused")
    elif ev in ("os.rename", "shutil.copyfile"):
        if not ok(args[1]):
            raise PermissionError(f"render_fig5_final write guard: {ev} to {args[1]} refused")
    elif ev in ("os.remove", "os.rmdir", "os.truncate", "shutil.rmtree", "os.chmod", "os.utime"):
        if isinstance(args[0], (str, bytes, os.PathLike)) and not ok(args[0]):
            raise PermissionError(f"render_fig5_final write guard: {ev} {args[0]} refused")
    elif ev == "os.mkdir":
        if not ok(args[0]) and not os.path.exists(args[0]):
            raise PermissionError(f"render_fig5_final write guard: mkdir {args[0]} refused")


sys.addaudithook(hook)
sys.path.insert(0, FIG); os.chdir(FIG)
import nature_style  # noqa: E402


def _fsync_rename(tmp, p):
    fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
    os.rename(tmp, p)


def save(fig, name):
    p = out / f"{name}{a.suffix}.pdf"
    q = out / f"{name}{a.suffix}.png"
    for x in (p, q):
        if x.exists():
            raise FileExistsError(f"refuse to overwrite {x}")
    tmp = out / f".{name}{a.suffix}.pdf.tmp"
    fig.savefig(tmp, format="pdf")
    _fsync_rename(tmp, p)
    print("saved", p, p.stat().st_size)
    import fitz                                                                  # FINAL-2: PNG at 300 dpi from the PDF itself
    doc = fitz.open(p); pg = doc[0]
    print(f"page {pg.rect.width / 72 * 25.4:.1f} x {pg.rect.height / 72 * 25.4:.1f} mm")
    tq = out / f".{name}{a.suffix}.png.tmp"
    pg.get_pixmap(dpi=300, alpha=False).save(str(tq), output="png")
    _fsync_rename(tq, q)
    print("saved", q, q.stat().st_size)


nature_style.OUT = out
nature_style.save = save
print(f"render_fig5_final2: {script.name} design={os.environ['FOEHN_DESIGN']} out={out}")
sys.argv = [str(script)]
runpy.run_path(str(script), run_name="__main__")
