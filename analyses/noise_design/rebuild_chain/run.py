#!/usr/bin/env python3
"""Run one script of the S1 CPU chain under a noise design:   FOEHN_DESIGN=<D> python -I -B run.py <script.py> [args ...]

* as review/keyfix/scripts/run.py: the script's folder goes first on sys.path and becomes the working directory, and the
  script runs as __main__ (python -I drops the script folder, which the scripts need for final_model, gc_style, common ...);
* write guard: before any file is opened for writing, renamed, removed, or a new directory is made, the path must carry the
  design label (a path component containing '_<D>' followed by '.', '_' or its end: compare_fair_crps_e01_ar.npz,
  data_ar/, figs_ar/, ...) or lie under /tmp, $TMPDIR or the cache folder of the environment; otherwise PermissionError is
  raised and nothing is written. No file of the paper as built can therefore be overwritten under a design;
* FOEHN_AUDIT=<file>: every path opened (r/w) is appended to <file> ('R' or 'W', path) for the read audit of 'check'."""
import os
import re
import runpy
import sys
from pathlib import Path

D = os.environ.get("FOEHN_DESIGN", "")
if not re.fullmatch(r"[a-z0-9]+", D):
    sys.exit("run.py: FOEHN_DESIGN=<label> (lower-case letters/digits, e.g. ar) is required")
TOK = re.compile(rf"_{re.escape(D)}(?=$|[._])")
ALLOW = tuple(p.rstrip("/") + "/" for p in ("/tmp", "/dev", os.environ.get("TMPDIR", "/tmp"),
                                             (os.environ["FOEHN_ENVS"] + "/.cache"), os.environ.get("MPLCONFIGDIR", "/tmp")))
WBITS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
AUD = open(os.environ["FOEHN_AUDIT"], "a", buffering=1) if os.environ.get("FOEHN_AUDIT") else None
AUD_ROOTS = tuple("/" + q.strip("/").split("/")[0] + "/" for q in (os.environ["FOEHN_WORK"], os.path.realpath(os.environ["FOEHN_WORK"]))) \
    if AUD is not None else ()                                  # top folders of FOEHN_WORK and of its resolved path


def ok(path):
    p = os.path.abspath(os.fsdecode(path))
    return (p + "/").startswith(ALLOW) or any(TOK.search(c) for c in p.split("/"))   # (p + "/"): the folder itself (O_TMPFILE)


def deny(what, path):
    raise PermissionError(f"run.py write guard (FOEHN_DESIGN={D}): {what} {path} refused: the path carries no '_{D}' label")


def hook(ev, args):
    if ev == "open":
        path, mode, flags = args
        if path is None or isinstance(path, int):
            return
        w = bool(flags & WBITS) if isinstance(flags, int) else bool(mode and any(c in mode for c in "wax+"))
        if AUD is not None and os.path.abspath(os.fsdecode(path)).startswith(AUD_ROOTS):   # project files only
            AUD.write(f"{'W' if w else 'R'}\t{os.path.abspath(os.fsdecode(path))}\n")
        if w and not ok(path):
            deny("write", path)
    elif ev == "os.rename":
        if AUD is not None and isinstance(args[1], (str, bytes, os.PathLike)):
            AUD.write(f"W\t{os.path.abspath(os.fsdecode(args[1]))}\n")
        if not ok(args[1]):
            deny("rename to", args[1])
    elif ev in ("os.remove", "os.rmdir", "os.truncate", "shutil.rmtree", "os.symlink", "os.link", "os.chmod", "os.utime"):
        p = args[1] if ev in ("os.symlink", "os.link") else args[0]
        if isinstance(p, (str, bytes, os.PathLike)) and not ok(p):
            deny(ev, p)
    elif ev == "os.mkdir":
        if not ok(args[0]) and not os.path.exists(args[0]):
            deny("mkdir", args[0])
    elif ev == "shutil.copyfile":
        if not ok(args[1]):
            deny("copy to", args[1])


script = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script.parent))
os.chdir(script.parent)
sys.argv = [str(script)] + sys.argv[2:]
sys.addaudithook(hook)
runpy.run_path(str(script), run_name="__main__")
