#!/usr/bin/env python3
"""A17 (a17/A17_draft.md): opposite-noise pairs on top of the evolving noise of A15/A16 (2021, validation year only).
Imported FIRST by every A17 script (numpy 1.26.3, the job container; the Philox normals differ between numpy versions).

Noise. Member m = 0..3 is the A15/A16 member m, bit for bit: z(m, d, h) of arnoise22.ARNoise(2021, d, h, rho*, 'philox')
(A16's generator, imported unchanged; keys [1521, 2021, d, m, k, tile row, tile column]). Member m = 4..7 is its exact
negative: z(m, d, h) = -z(m - 4, d, h), the float16 sign bit flipped. Because z_h = sum_k a_hk eps_k is a signed sum
and IEEE rounding is symmetric, this is also the field built from the negated innovations -eps_k (same keys) at every
lead: the AR(1) structure between leads is kept and every member is still white N(0, 1) at every (d, h)
(check17.py NEG_*, LAW). ARNoise17 is a duck-typed Npy, so the UNCHANGED tile_window / full_map cut it exactly as
noise_tiles.npy. WindowsAR17 = arnoise22.WindowsAR22 whose x0 stacks the members of a list of member ids (lines that
differ end with '# A17-W1'); the particles of member id m are member_particles(i, h, m) as in an 8-member run.

Rules. An audit hook refuses any open of a station / monitor / OpenAQ file (verbatim copy of a15/arnoise._audit) and a
second one refuses the folders of the pre-registered evaluation that runs meanwhile (any path component ending in
'_ar', a16/data, a16/results). Atomic writes (temporary file, fsync, read-back, rename) only under a17/."""
import sys

sys.dont_write_bytecode = True                                  # no __pycache__ in a15/, a16/ or a17/ (file quota)
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402


def _audit(event, args):                       # stations and monitors are evaluation data only: never opened here
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]).lower()
        if p.endswith((".py", ".pyc", ".so")) or "/site-packages/" in p:     # code, not data
            return
        if "station" in p or "openaq" in p or "monitor" in p:
            raise PermissionError(f"A13 warn16: station/monitor file refused: {p}")


sys.addaudithook(_audit)
_FORBID = re.compile(r"(/a16/(data|results)(/|$))|(_ar(/|$))")


def _audit17(event, args):                     # the folders of the pre-registered evaluation running meanwhile
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if p.endswith((".py", ".pyc", ".so")) or "/site-packages/" in p:
            return
        if _FORBID.search(p):
            raise PermissionError(f"A17: path of the running pre-registered evaluation refused: {p}")


sys.addaudithook(_audit17)

ROOT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution")).resolve()
A17 = ROOT / "a17"
CODE, OUT, DATA = A17 / "code", A17 / "code" / "out", A17 / "data"
A15_DIR, A16_DIR = ROOT / "a15", ROOT / "a16"
sys.path.insert(0, str(A16_DIR))
import arnoise22 as an  # noqa: E402  (A16's generator, imported unchanged; numpy 1.26.3 asserted there)
import numpy as np  # noqa: E402
import torch  # noqa: E402

egg = an.egg
NTR, NTC, TILE, B = an.NTR, an.NTC, an.TILE, an.B
YEAR, NPAIR, MEMBERS = 2021, 4, 8
RHO_A17 = 0.59                                                  # rho* of a15/results/rho.json (A15 4.5, A16 section 1)
CONF, CAL = list(an.CONF), list(an.CAL)
LEADS = (1, 3)
PLANS = {   # (start day of year, lead) and member ids of every A17 GPU run (A17_draft.md section 5)
    "smoke_alone": dict(items=[(9, 2), (17, 2)], mids=(4, 5, 6, 7)),
    "smoke_eight": dict(items=[(9, 2)], mids=tuple(range(MEMBERS))),
    "main": dict(items=[(d, h) for d in CONF for h in LEADS], mids=(4, 5, 6, 7)),
}
SMOKE = ("smoke_alone", "smoke_eight")
assert len(PLANS["main"]["items"]) == 60 and not set(PLANS["smoke_alone"]["items"]) & set(PLANS["main"]["items"])
CHECK_JSON, SMOKE_JSON, GPU_LOG = OUT / "check17.json", DATA / "smoke_check.json", DATA / "a17_gpu.out"
RUN_CODE = ("code/noise17.py", "code/eval17.py", "code/smoke17.py", "code/smoke.sh", "code/main.sh")
STORED_CODE = ("eval_global_green", "fullflow_global", "fullflow_global_fp", "fullflow_global_green", "fullflow", "dejavu")


# ======================================================================== the noise
class ARNoise17(an.ARNoise):
    """duck-typed Npy of the 8 members of one (start day, lead): shape (8, 29, 71, 512, 512), '<f2'; members 4..7 are the
    exact negatives of members 0..3"""

    def __init__(self, doy, lead, rho):
        super().__init__(YEAR, doy, lead, rho, "philox", MEMBERS)

    def tile(self, m, tr, tc):
        assert 0 <= m < MEMBERS, m
        if m >= NPAIR:
            return np.negative(super().tile(m - NPAIR, tr, tc))        # float16 negation: the sign bit, exactly
        return super().tile(m, tr, tc)


class WindowsAR17(egg.Windows):
    """eval_global_green.Windows with the A17 noise: jobs [(start index, lead, window)], member ids `mids`, one rho"""

    def __init__(self, ds, jobs, win, mids, rho):
        assert all(0 <= m < MEMBERS for m in mids) and len(set(mids)) == len(mids), mids
        self.ds, self.jobs, self.win, self.members, self.lags = ds, jobs, win, len(mids), []     # no lagged members
        self.mids, self.rho = tuple(int(m) for m in mids), float(rho)

    def __getitem__(self, n):
        i, h, w = self.jobs[n]
        k0, j0 = (int(v) for v in self.win[w])
        b = self.ds.item(i, h, k0, j0)
        del b["x1"], b["mask"]
        assert int(self.ds.year[i]) == YEAR, (int(self.ds.year[i]), YEAR)  # A17-W1
        nz = ARNoise17(int(self.ds.doy[i]), h, self.rho)  # A17-W1
        b["x0"] = torch.from_numpy(np.stack([egg.tile_window(nz, m, 25 * k0, 25 * j0, egg.PH, egg.PW) for m in self.mids]))  # A17-W1
        for l in self.lags:                                              # lagged weather runs (members with --lags l)
            for k, v in self.ds.lagged(i, h, k0, j0, l).items():
                b[f"lag{l}_{k}"] = v
        return b


REMOVED_LINES_WINDOWS = [   # the lines of arnoise22.WindowsAR22.__getitem__ that do not survive (check17.py DIFF)
    'assert int(self.ds.year[i]) == self.year, (int(self.ds.year[i]), self.year)  # A16-W1',
    'nz = ARNoise(self.year, int(self.ds.doy[i]), h, self.rho, self.base, self.members)  # A16-W1',
    'b["x0"] = torch.from_numpy(np.stack([egg.tile_window(nz, m, 25 * k0, 25 * j0, egg.PH, egg.PW) for m in range(self.members)]))  # A16-W1',
]


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


GOLDEN17_ITEMS = ((9, 1, 4, 14, 35), (9, 3, 5, 0, 0), (353, 3, 7, 28, 70), (105, 1, 6, 7, 30), (17, 2, 4, 20, 50))


def golden17():
    """golden SHA-256 of A17 tiles (2021 keys, rho*): negated members (and their partners), plus A16's 2021 hashes"""
    out = {"numpy": np.__version__, "a16_golden_2021": an.golden_hashes()}
    for d, h, m, tr, tc in GOLDEN17_ITEMS:
        nz = ARNoise17(d, h, RHO_A17)
        out[f"z_{d}_lead{h}_m{m}_tile{tr}_{tc}/f16"] = sha(nz.tile(m, tr, tc))
        out[f"z_{d}_lead{h}_m{m - NPAIR}_tile{tr}_{tc}/f16"] = sha(nz.tile(m - NPAIR, tr, tc))
    return out


# ======================================================================== gates
def rho_star():
    rj = json.loads((A15_DIR / "results" / "rho.json").read_text())
    assert rj.get("clipped") is False and rj.get("all_passed") is True, "a15/results/rho.json: clipped or not passed"
    r = float(rj["rho_star"])
    assert r == RHO_A17, f"rho* of a15/results/rho.json is {r}, A17 uses {RHO_A17}"
    return r


def run_code_sha():
    """sha256 of the A17 run code and job files and of the stored modules they import (recorded by check17 and smoke17)"""
    d = {n: sha_file(A17 / n) for n in RUN_CODE}
    d["a16/arnoise22.py"] = sha_file(an.__file__)
    d.update({m: sha_file(sys.modules[m].__file__) for m in STORED_CODE})
    return d


def gate_state(check, smoke, code_now, plan):
    """pure gate rule (tested in check17 GATES): smoke runs need check17 passed with the same code; main also needs the
    smoke passed with the same code. Returns (ok, reason)."""
    if not (check and check.get("all_passed") is True):
        return False, "code/out/check17.json missing or not all_passed"
    if check.get("code_sha256") != code_now:
        return False, "code or job files differ from those checked by check17.py"
    if plan in SMOKE:
        return True, "ok"
    assert plan == "main", plan
    if not (smoke and smoke.get("all_passed") is True):
        return False, "data/smoke_check.json missing or not all_passed"
    if smoke.get("code_sha256") != code_now:
        return False, "code or job files differ from those of the smoke"
    return True, "ok"


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def gate(plan):
    ok, why = gate_state(read_json(CHECK_JSON), read_json(SMOKE_JSON), run_code_sha(), plan)
    assert ok, f"A17 {plan}: {why}"
    return rho_star()


# ======================================================================== atomic writes, JSON
def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inside_a17(path):
    p = Path(path).resolve()
    assert str(p).startswith(str(A17) + os.sep), f"{p} is outside {A17}"
    assert not str(p).startswith(str(A17 / "diag")), f"{p}: a17/diag belongs to another step"
    return p


def write_bytes(path, data):
    p = inside_a17(path)
    tmp = p.with_name(p.name + f".tmp_a17_{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    with open(tmp, "rb") as f:
        assert f.read() == data, tmp
    os.replace(tmp, p)
    fsync_dir(p.parent)
    with open(p, "rb") as f:
        assert f.read() == data, p


def npz_readback(path, names):
    """CRC read-back of a written npz (every member present and intact)"""
    with zipfile.ZipFile(path) as zf:
        assert sorted(zf.namelist()) == sorted(k + ".npy" for k in names), path
        assert zf.testzip() is None, f"A17: CRC read-back of {path} failed"


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(path, obj):
    write_bytes(path, (json.dumps(jsonable(obj), indent=1) + "\n").encode())
