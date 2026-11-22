#!/usr/bin/env python3
"""POLISH render wrapper of Fig. 6 (execution pass, 2026-11-21). Renders a figure script exactly as
N/rebuild/print/figs_letters/render_asold.py rendered the manuscript's Fig. 6 (experiments/figures first on sys.path and as the
working directory; --fig-style imports fig_style before nature_style; NATURE unset), but writes ONLY the PDF, into --out, as
<--name>.pdf, through a temporary file + fsync + rename, and never overwrites a file.

    FOEHN_DESIGN=ar MPLCONFIGDIR=<dir> python -I -B render_polish.py <script.py> --out <dir> --name <stem> --fig-style

Write guard: any write, rename, removal or new folder outside --out, /tmp, /dev and MPLCONFIGDIR is refused."""
import argparse
import os
import runpy
import sys
from pathlib import Path

FIG = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures")
ap = argparse.ArgumentParser()
ap.add_argument("script"); ap.add_argument("--out", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--fig-style", action="store_true")
a = ap.parse_args()
script = Path(a.script).resolve()
out = Path(a.out).resolve()
assert out.is_dir(), out
assert os.environ.get("NATURE", "") == "", "NATURE must be unset (as the manuscript's render)"
final = out / f"{a.name}.pdf"
if final.exists():
    raise FileExistsError(f"refuse to overwrite {final}")
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
    elif ev in ("os.rename", "os.replace", "shutil.copyfile"):
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
mods = []
if a.fig_style:
    import fig_style  # noqa: E402  (rcParams, as render_asold.py / review/figfix/render.py)
    mods.append(fig_style)
import nature_style  # noqa: E402
mods.append(nature_style)


def save(fig, name):
    tmp = out / f".{a.name}.tmp.pdf"
    fig.savefig(tmp)
    fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
    if final.exists():
        os.remove(tmp); raise FileExistsError(f"refuse to overwrite {final}")
    os.replace(tmp, final)
    print("saved", final, "(script name:", name + ")")


for m in mods:
    m.OUT = out
    m.save = save
print(f"render_polish: {script.name} design={os.environ.get('FOEHN_DESIGN', '') or 'stored'} fig_style={a.fig_style} out={out}")
sys.argv = [str(script)]
runpy.run_path(str(script), run_name="__main__")
