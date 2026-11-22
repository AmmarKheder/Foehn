"""A14 class U: common paths and helpers (P/a14/classU). Writes only under P/a14/classU, atomically."""
import ast
import contextlib
import hashlib
import io
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_REAL_SAVEZ, _REAL_SAVE = np.savez, np.save
P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
U = P / "a14" / "classU"
# ---- noise design (P/a14/DESIGN_SWITCH.md; A16 section 4 point 3). FOEHN_DESIGN unset: the stored runs, every path as before.
#   ar      the final model's runs with the noise drawn anew at every start day (RUN/<eval>_ar, P/a16/data/run16, the S1 outputs
#           labelled _ar); V_draw = 0 and factor 1 in every interval of the final model (VDRAW0)
#   check   the stored runs, outputs labelled _check (proof that the switch changes nothing)
#   check0  'check' with V_draw = 0 and factor 1 (proof that only the intervals of the final model change)
DESIGN = os.environ.get("FOEHN_DESIGN", "")
assert DESIGN in ("", "ar", "check", "check0"), f"FOEHN_DESIGN={DESIGN!r}: A14 knows only '', 'ar', 'check', 'check0'"
SFX = f"_{DESIGN}" if DESIGN else ""                  # outputs: data_<D>/, results_<D>/
ISFX = "_ar" if DESIGN == "ar" else ""                # inputs: the final model's folders and the S1 outputs of the design
AR, VDRAW0 = DESIGN == "ar", DESIGN in ("ar", "check0")
DESIGN_SCRIPTS = ("st_cf_pass.py", "rel_pass.py", "pop_pass.py", "land25_pass.py", "st_cf_score.py", "rel_score.py",
                  "chain_classU.py", "make_table.py", "r1_pipeline_b.py", "j_m0_design.py")
if DESIGN and Path(sys.argv[0]).name not in DESIGN_SCRIPTS:
    sys.exit(f"FOEHN_DESIGN={DESIGN}: {Path(sys.argv[0]).name} has no design switch (it reads other systems or is not part of "
             "REBUILD_PLAN 4c step 8); run it without FOEHN_DESIGN")
DESIGN_NOTE = (dict(FOEHN_DESIGN=DESIGN, rule="A16 section 4 point 3: V_draw = 0 and factor 1 in every interval of the final model "
                    "(the jackknife over fields and the variance over quartets enter no interval; the fields v_draw, v_quart, v_jack*, "
                    "cv_* are the fixed-design diagnostics, kept for information); estimators of A14 4(a) and 4(b) unchanged; other "
                    "systems (stored fixed-field models) keep their intervals")
               if VDRAW0 else None)
DATA, RES, LOGS = U / f"data{SFX}", Path(os.environ.get("A14U_RES", str(U / f"results{SFX}"))), U / "logs"
DATA0, RES0 = (U / "data", U / "results") if DESIGN else (DATA, RES)   # other systems' records: read as stored under a design
if DESIGN:
    DATA.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
CRAN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm"))
RUN = CRAN / "experiments" / "global1km" / "runs" / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"
EVF, EVP = RUN / f"eval_final_crps_e01{ISFX}", RUN / f"eval_pop_crps_e01{ISFX}"
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
FIGS = CRAN / "experiments" / "figures"
REL = CRAN / "paper" / "natcomms" / "analyses" / "reliability"
RELD, RELR = REL / f"data{ISFX}", REL / f"results{ISFX}"   # reliability sums and as-run records of the design (S1 step 5)
RUN16_AR = P / "a16" / "data" / "run16"               # 'ar': the 358 sixteen-member forecasts in one folder (A16 run G11)
SECT = CRAN / "paper" / "natcomms" / "work" / "sections"
SUPP = CRAN / "paper" / "natcomms" / "work" / "supplementary.tex"
W16 = P / "a13" / "warn16"
RUN16_DIRS = {"run16": W16 / "data" / "run16", "run16_13": P / "a14" / "data" / "run16_13", "run16all": W16 / "data" / "run16all"}
TASK0 = [(2, 1), (2, 3), (4, 1), (4, 3)]                 # forecasts of R1 written by task 0 of job 23898998 (run16all)
QUARTETS = np.array(list(combinations(range(16), 4)), np.int64)   # 1,820
assert len(QUARTETS) == 1820
F_FPC = 16.0 / 12.0                                       # drawing 4 of 16 without replacement -> independent draws


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _commit(tmp, path):
    fd = os.open(tmp, os.O_RDONLY)
    os.fsync(fd); os.close(fd)
    os.replace(tmp, path)


def safe_text(path, text, replace=True):
    path = Path(path); assert str(path).startswith(str(U)), path
    if path.exists() and not replace:
        raise FileExistsError(path)
    data = text.encode("utf-8"); tmp = str(path) + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data); fh.flush(); os.fsync(fh.fileno())
    assert open(tmp, "rb").read() == data, path
    os.replace(tmp, path)
    assert open(path, "rb").read() == data, path


def safe_json(path, obj):
    safe_text(path, json.dumps(obj, indent=1, default=_jdefault, allow_nan=True) + "\n")


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def safe_npz(path, **arrays):
    path = Path(path); assert str(path).startswith(str(U)), path
    tmp = str(path) + f".tmp{os.getpid()}.npz"
    _REAL_SAVEZ(tmp, **arrays)
    _commit(tmp, path)
    z = np.load(path)                                     # read-back
    for k, v in arrays.items():
        a = z[k]
        assert a.shape == np.asarray(v).shape, (path, k)
        if a.dtype.kind in "fc":
            assert np.array_equal(a, np.asarray(v), equal_nan=True), (path, k)
        else:
            assert np.array_equal(a, np.asarray(v)), (path, k)


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.s = streams

    def write(self, x):
        for s in self.s:
            s.write(x)
        return len(x)

    def flush(self):
        for s in self.s:
            s.flush()


def exec_script(src, argv, savez_dir, prefix="exec_", env=None):
    """execute a paper script from its UNCHANGED source text; np.savez / np.save are redirected into savez_dir (nothing is
    written beside the script). Returns (namespace, captured stdout, sha256 of the source, files written)."""
    src = Path(src); code = src.read_text()
    ns = {"__name__": "__main__", "__file__": str(src)}
    written = {}
    real_savez, real_save = _REAL_SAVEZ, _REAL_SAVE

    def savez_r(file, *a, **kw):
        out = Path(savez_dir) / (prefix + Path(str(file)).name)
        written[str(file)] = str(out)
        safe_npz(out, **kw) if not a else real_savez(out, *a, **kw)

    def save_r(file, arr, *a, **kw):
        out = Path(savez_dir) / (prefix + Path(str(file)).name)
        written[str(file)] = str(out)
        real_save(out, arr, *a, **kw)
    old_argv, old_env = sys.argv, dict(os.environ)
    sys.argv = [str(src)] + list(argv)
    if env:
        os.environ.update(env)
    buf = io.StringIO()
    np.savez, np.save = savez_r, save_r
    try:
        with contextlib.redirect_stdout(Tee(buf, sys.stdout)):
            exec(compile(code, str(src), "exec"), ns)
    finally:
        np.savez, np.save = real_savez, real_save
        sys.argv = old_argv
        os.environ.clear(); os.environ.update(old_env)
    return ns, buf.getvalue(), hashlib.sha256(code.encode()).hexdigest(), written


def functions_from(src, names, ns):
    """define the functions `names` of the unchanged source file src in the namespace ns (ast; nothing else runs)"""
    tree = ast.parse(Path(src).read_text())
    found = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], type_ignores=[]), str(src), "exec"), ns)
            found.append(node.name)
    assert sorted(found) == sorted(names), (found, names)
    return ns


def station_pixels(lat, lon):
    """1 km pixel of each station, as eval_global_green.py (l. 336-338): row * 36000 + column"""
    GH, GW = 14400, 36000
    sr = np.clip(np.floor((84.0 - lat) / 0.01).astype(int), 0, GH - 1)
    sc = np.clip(np.floor((lon + 180.0) / 0.01).astype(int), 0, GW - 1)
    return sr.astype(np.int64) * GW + sc


def stations_in_pop():
    """position of each of the 8,110 stations in the populated-pixel list (pop_pixels.npz idx), -1 if absent"""
    st = np.load(G / "stations_2022.npz")
    idx = np.load(G / "pop_pixels.npz")["idx"]
    pix = station_pixels(st["lat"], st["lon"])
    pos = np.clip(np.searchsorted(idx, pix), 0, len(idx) - 1)
    return np.where(idx[pos] == pix, pos, -1)


def index16(dirs):
    """(start day, lead) -> (file, row, target) for every forecast done in the given 16-member directories"""
    out = {}
    for D in dirs:
        D = Path(D)
        for fk in sorted(D.glob("keys16_*.npy")):
            rank = fk.stem.split("_")[1]
            dj = D / f"done16_{rank}.json"
            if not dj.exists():
                continue
            done = set(json.loads(dj.read_text()))
            for n, (d, h, t) in enumerate(np.load(fk)):
                if n in done:
                    key = (int(d), int(h))
                    assert key not in out, key
                    out[key] = (str(D / f"pop16_{rank}.npy"), n, int(t))
    return out


def jk_var(reps):
    """delete-one jackknife variance from the replicates (last axis)"""
    reps = np.asarray(reps, np.float64)
    m = reps.shape[-1]
    return (m - 1) / m * ((reps - reps.mean(-1, keepdims=True)) ** 2).sum(-1)


def inflate(pt, lo, hi, v_boot, v_draw):
    """A14 4c: half-widths multiplied by sqrt(1 + V_draw / V_boot); under VDRAW0 (A16 4.3) V_draw = 0: factor 1, the bootstrap
    interval (every caller of this function is a statistic of the final model)"""
    if VDRAW0:
        return [lo, hi], 1.0
    f = float(np.sqrt(1.0 + v_draw / v_boot)) if v_boot > 0 and np.isfinite(v_draw) else float("nan")
    return [pt - f * (pt - lo), pt + f * (hi - pt)], f


@contextlib.contextmanager
def upstream_env():
    """import the paper's modules that carry their own FOEHN_DESIGN switch (evaluated at import): under 'check'/'check0' they
    must read the stored files, so FOEHN_DESIGN is hidden from them while they are imported; under 'ar' they see it"""
    held = os.environ.pop("FOEHN_DESIGN", None) if DESIGN in ("check", "check0") else None
    try:
        yield
    finally:
        if held is not None:
            os.environ["FOEHN_DESIGN"] = held
