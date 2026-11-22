#!/usr/bin/env python3
"""Render a figure script of experiments/figures as the manuscript's version was rendered, under a noise design or not,
WITHOUT overwriting any stored figure (A16 figure pass, 2026-11-21; N/rebuild/print/figures_ar.md).

    FOEHN_DESIGN=ar python -I -B ../../s1_chain/run.py render_asold.py <script.py> --suffix _letters [--fig-style]
    (no design, validation only)  python -I -B render_asold.py <script.py> --suffix _letters --out <dir under /tmp>

* experiments/figures goes first on sys.path and becomes the working directory (as review/figfix/render.py does);
* --fig-style imports fig_style before nature_style, as review/figfix/render.py does (Fig. 6 of the manuscript was rendered
  that way on 2026-11-20 with NATURE unset: its stray top/right ticks of h and j come from fig_style's rcParams);
* output folder: N/rebuild/figs_<D> under FOEHN_DESIGN=<D>, else --out (must lie under /tmp); every name gets --suffix and
  an existing file is never overwritten;
* write guard: any write, rename, removal or new folder outside the output folder, /tmp, /dev and MPLCONFIGDIR is refused."""
import argparse
import os
import runpy
import sys
from pathlib import Path

FIG = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures")
N = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms"))
ap = argparse.ArgumentParser()
ap.add_argument("script"); ap.add_argument("--suffix", required=True); ap.add_argument("--fig-style", action="store_true")
ap.add_argument("--out")
a = ap.parse_args()
assert a.suffix.startswith("_") and len(a.suffix) > 1
script = Path(a.script).resolve()
D = os.environ.get("FOEHN_DESIGN", "")
if D:
    out = N / "rebuild" / f"figs_{D}"
    assert a.out is None, "--out is only for the stored design"
else:
    assert a.out and Path(a.out).resolve().as_posix().startswith("/tmp/"), "without a design, --out must lie under /tmp"
    out = Path(a.out).resolve()
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
            raise PermissionError(f"render_asold write guard: write {p} refused")
    elif ev in ("os.rename", "shutil.copyfile"):
        if not ok(args[1]):
            raise PermissionError(f"render_asold write guard: {ev} to {args[1]} refused")
    elif ev in ("os.remove", "os.rmdir", "os.truncate", "shutil.rmtree", "os.chmod", "os.utime"):
        if isinstance(args[0], (str, bytes, os.PathLike)) and not ok(args[0]):
            raise PermissionError(f"render_asold write guard: {ev} {args[0]} refused")
    elif ev == "os.mkdir":
        if not ok(args[0]) and not os.path.exists(args[0]):
            raise PermissionError(f"render_asold write guard: mkdir {args[0]} refused")


sys.addaudithook(hook)
sys.path.insert(0, FIG); os.chdir(FIG)
mods = []
if a.fig_style:
    import fig_style  # noqa: E402  (rcParams, as review/figfix/render.py)
    mods.append(fig_style)
import nature_style  # noqa: E402
mods.append(nature_style)


def make_save(mod):
    def save(fig, name):
        for ext in ("pdf", "png"):
            p = out / f"{name}{a.suffix}.{ext}"
            if p.exists():
                raise FileExistsError(f"refuse to overwrite {p}")
            fig.savefig(p)
        print("saved", out / f"{name}{a.suffix}")
    return save


for m in mods:
    m.OUT = out
    m.save = make_save(m)
print(f"render_asold: {script.name} design={D or 'stored'} fig_style={a.fig_style} NATURE={os.environ.get('NATURE', '')!r} out={out}")
sys.argv = [str(script)]
runpy.run_path(str(script), run_name="__main__")
