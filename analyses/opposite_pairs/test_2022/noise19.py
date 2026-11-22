#!/usr/bin/env python3
"""A19 (a19/A19_draft.md): eight members as four opposite pairs on the test year 2022, populated pixels, leads 1 and 3.
Imported FIRST by every A19 run-side script (eval19, check19, smoke19); numpy 1.26.3 only (the job container).

Noise. Member m = 0..3 is the A16 member m of 2022, bit for bit: z(m, d, h) of arnoise22.ARNoise(2022, d, h, rho*, 'philox')
(A16's generator, imported unchanged; keys [1521, 2022, d, m, k, tile row, tile column]; the noise of run G2). Member
m = 4..7 is its exact negative, z(m, d, h) = -z(m - 4, d, h) (float16 sign bit), the construction of a17/code/noise17.py
with the year 2022. ARNoise19 is a duck-typed Npy, so the UNCHANGED tile_window / full_map cut it exactly as noise_tiles.npy.
WindowsAR19 = arnoise22.WindowsAR22 whose x0 stacks the members of a list of member ids (lines that differ end with
'# A19-W1'); the particles of member id m are member_particles(i, h, m), as in an eight-member run.

Rules.
  _audit    the audit hook of the populated-pixel samplers (a16/sample16_ar.py, verbatim): no station, monitor or OpenAQ
            file can be opened by a process that imports this module
  _audit19  no write (open for writing, rename, remove, mkdir, link, ...) into the folders A19 must not modify: a14, a15,
            a16, a17, protocol_amendments, eccv/cran-pm/experiments (runs, figures), eccv/cran-pm/paper/natcomms, eccv/data
  writes    atomic (temporary file, fsync, read-back, rename), only under a19/
  plans     PLANS fixes every argument of every A19 GPU run (main = A16's G2 with member ids 4..7 and a new folder); gate()
            refuses another rho or base, a missing or failed check19, changed code, and for main a missing or failed smoke
            or a smoke whose projection breaks the limit or the cap; claim() refuses a folder holding a file A19 does not
            write, files without the A19 manifest, or the manifest of another run or code version"""
import sys

sys.dont_write_bytecode = True                                  # no __pycache__ (file quota)
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

# ---- A19 write guard (begin; verbatim in score19.py, check19 DIFF)
_BASES = (os.environ["FOEHN_WORK"], os.path.realpath(os.environ["FOEHN_WORK"]))
_KEEP = ("global_model_forecasting_air_pollution/a14", "global_model_forecasting_air_pollution/a15",
         "global_model_forecasting_air_pollution/a16", "global_model_forecasting_air_pollution/a17",
         "global_model_forecasting_air_pollution/protocol_amendments", "eccv/cran-pm/experiments",
         "eccv/cran-pm/paper/natcomms", "eccv/data")
PROTECTED = tuple(f"{b}/{k}" for b in _BASES for k in _KEEP)
_WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_WEVENTS = {"os.rename": (0, 1), "os.remove": (0,), "os.rmdir": (0,), "os.mkdir": (0,), "os.link": (1,), "os.symlink": (1,),
            "os.truncate": (0,), "os.chmod": (0,), "os.chown": (0,), "os.utime": (0,), "shutil.copyfile": (1,),
            "shutil.copytree": (1,), "shutil.move": (0, 1), "shutil.rmtree": (0,)}


def protected(path):
    """True if `path` lies in a folder that A19 must not modify (both spellings of the scratch mount)"""
    p = os.fsdecode(path)
    if not p.startswith("/"):
        try:
            p = os.path.join(os.getcwd(), p)
        except OSError:
            return False
    p = os.path.normpath(p)
    return any(p == q or p.startswith(q + "/") for q in PROTECTED)


def _audit19(event, args):                     # A19 writes only under a19/: never into a folder it must not modify
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and bool(flags & _WFLAGS))
        if w and protected(args[0]):
            raise PermissionError(f"A19: write refused in a protected folder: {os.fsdecode(args[0])}")
    elif event in _WEVENTS:
        for j in _WEVENTS[event]:
            a = args[j] if len(args) > j else None
            if isinstance(a, (str, bytes, os.PathLike)) and protected(a):
                raise PermissionError(f"A19: {event} refused in a protected folder: {os.fsdecode(a)}")


sys.addaudithook(_audit19)
# ---- A19 write guard (end)

ROOT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution")).resolve()
A19 = ROOT / "a19"
CODE, OUT, DATA = A19 / "code", A19 / "code" / "out", A19 / "data"
A15_DIR, A16_DIR, A17_DIR = ROOT / "a15", ROOT / "a16", ROOT / "a17"
sys.path.insert(0, str(A16_DIR))
import arnoise22 as an  # noqa: E402  (A16's generator, plans helpers and eval_global_green, imported unchanged; numpy 1.26.3)
import numpy as np  # noqa: E402
import torch  # noqa: E402

egg = an.egg
NTR, NTC, TILE, B = an.NTR, an.NTC, an.TILE, an.B
YEAR, NPAIR, MEMBERS = 2022, 4, 8
RHO_A19 = 0.59                                                  # rho* of a15/results/rho.json (A15 4.5, A16 section 1)
RUN = an.RUN
CKPT = an.CKPT
G2_DIR = RUN / "eval_pop_crps_e01_ar"                           # A16 run G2: members 0-3 at the populated pixels (read only)
LEADS = (1, 3)
_A = dict(run=str(RUN), ckpt=str(CKPT), steps=4, refine_tail=1, psub=16, every=1, doys=None, leads=[1, 3], save_1km=[],
          crop=None, crop_only=False, crop_days=[], pop=True, year=YEAR)


def _plan(**kw):
    d = dict(_A)
    d.update(kw)
    return d


PLANS = {   # every argument of every A19 GPU run; main = A16 G2 (an.EVAL_PLANS['G2']) with member ids 4..7 and a new folder
    "main": _plan(out=str(DATA / "pop_neg"), every=2, members=4, mids=(4, 5, 6, 7), n=358, world=64),
    "smoke_alone": _plan(out=str(DATA / "smoke_alone"), doys=[2], leads=[1, 3], members=4, mids=(4, 5, 6, 7), n=2, world=1),
    "smoke_eight": _plan(out=str(DATA / "smoke_eight"), doys=[2], leads=[3], members=8, mids=tuple(range(MEMBERS)), n=1, world=1),
}
SMOKE = ("smoke_alone", "smoke_eight")
PLAN_KEYS = tuple(an.EVAL_KEYS)                                 # the arguments compared with the command line
CHECK_JSON, SMOKE_JSON, GPU_LOG = OUT / "check19.json", DATA / "smoke_check.json", DATA / "a19_gpu.out"
LIMIT_MAIN, LIMIT_SMOKE, CAP, MAIN_GCDS = "00:55:00", "01:00:00", 60.0, 64   # main.sh / smoke.sh --time; A19 cap (GCD-hours)
RUN_CODE = ("code/noise19.py", "code/eval19.py", "code/smoke19.py", "code/smoke.sh", "code/main.sh")
REF_CODE = ("a16/arnoise22.py", "a16/eval_ar22.py", "a16/sample16_ar.py")
STORED_CODE = ("eval_global_green", "fullflow_global", "fullflow_global_fp", "fullflow_global_green", "fullflow", "dejavu")


# ======================================================================== the noise
class ARNoise19(an.ARNoise):
    """duck-typed Npy of the 8 members of one (2022 start day, lead): shape (8, 29, 71, 512, 512), '<f2'; members 4..7 are the
    exact negatives of members 0..3"""

    def __init__(self, doy, lead, rho):
        super().__init__(YEAR, doy, lead, rho, "philox", MEMBERS)

    def tile(self, m, tr, tc):
        assert 0 <= m < MEMBERS, m
        if m >= NPAIR:
            return np.negative(super().tile(m - NPAIR, tr, tc))        # float16 negation: the sign bit, exactly
        return super().tile(m, tr, tc)


class WindowsAR19(egg.Windows):
    """eval_global_green.Windows with the A19 noise: jobs [(start index, lead, window)], member ids `mids`, one rho"""

    def __init__(self, ds, jobs, win, mids, rho):
        assert all(0 <= m < MEMBERS for m in mids) and len(set(mids)) == len(mids), mids
        self.ds, self.jobs, self.win, self.members, self.lags = ds, jobs, win, len(mids), []     # no lagged members
        self.mids, self.rho = tuple(int(m) for m in mids), float(rho)

    def __getitem__(self, n):
        i, h, w = self.jobs[n]
        k0, j0 = (int(v) for v in self.win[w])
        b = self.ds.item(i, h, k0, j0)
        del b["x1"], b["mask"]
        assert int(self.ds.year[i]) == YEAR, (int(self.ds.year[i]), YEAR)  # A19-W1
        nz = ARNoise19(int(self.ds.doy[i]), h, self.rho)  # A19-W1
        b["x0"] = torch.from_numpy(np.stack([egg.tile_window(nz, m, 25 * k0, 25 * j0, egg.PH, egg.PW) for m in self.mids]))  # A19-W1
        for l in self.lags:                                              # lagged weather runs (members with --lags l)
            for k, v in self.ds.lagged(i, h, k0, j0, l).items():
                b[f"lag{l}_{k}"] = v
        return b


REMOVED_LINES_WINDOWS = [   # the lines of arnoise22.WindowsAR22.__getitem__ that do not survive (check19.py DIFF)
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


GOLDEN19_ITEMS = ((2, 3, 4, 14, 35), (2, 1, 5, 0, 0), (356, 3, 7, 28, 70), (190, 1, 6, 7, 30), (64, 3, 4, 20, 50))


def golden19():
    """golden SHA-256 of A19 tiles (2022 keys, rho*): negated members and their partners, with A16's 2021 and 2022 hashes"""
    out = {"numpy": np.__version__, "a16_golden_2021": an.golden_hashes(), "a16_golden22": an.golden22_hashes()}
    for d, h, m, tr, tc in GOLDEN19_ITEMS:
        z = ARNoise19(d, h, RHO_A19)
        out[f"z_{d}_lead{h}_m{m}_tile{tr}_{tc}/f16"] = sha(z.tile(m, tr, tc))
        out[f"z_{d}_lead{h}_m{m - NPAIR}_tile{tr}_{tc}/f16"] = sha(z.tile(m - NPAIR, tr, tc))
    return out


# ======================================================================== plans and gates
def resolved(p):
    p = Path(p)
    return str(p.parent.resolve() / p.name)


def args_of(a):
    """the canonical arguments of an eval19.py run (plan check and manifest): A16's eval_args and the member ids"""
    d = an.eval_args(a)
    d["mids"] = [int(m) for m in a.mids]
    return d


def plan_match(plan, got):
    """the arguments given must be exactly those of the plan (two sources: the job script and this table)"""
    want = {k: PLANS[plan][k] for k in PLAN_KEYS}
    want["run"], want["ckpt"], want["out"] = resolved(want["run"]), resolved(want["ckpt"]), resolved(want["out"])
    want["mids"] = [int(m) for m in PLANS[plan]["mids"]]
    bad = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
    assert not bad and set(got) == set(want), f"A19 plan {plan}: arguments differ from the plan (given, plan): {bad}"
    return want


def rho_star():
    rj = json.loads((A15_DIR / "results" / "rho.json").read_text())
    assert rj.get("clipped") is False and rj.get("all_passed") is True, "a15/results/rho.json: clipped or not passed"
    r = float(rj["rho_star"])
    assert r == RHO_A19, f"rho* of a15/results/rho.json is {r}, A19 uses {RHO_A19}"
    return r


def run_code_sha():
    """sha256 of the A19 run code and job files, of A16's reference code and of the stored modules (check19, smoke19, gate)"""
    d = {n: sha_file(A19 / n) for n in RUN_CODE}
    d.update({n: sha_file(ROOT / n) for n in REF_CODE})
    d.update({m: sha_file(sys.modules[m].__file__) for m in STORED_CODE})
    return d


def gate_state(check, smoke, code_now, plan, rho, base):
    """pure gate rule (tested in check19 GATES). Returns (ok, reason)."""
    if base != "philox" or float(rho) != RHO_A19:
        return False, f"A19 runs base philox and rho {RHO_A19} only (got {base}, {rho})"
    if not (check and check.get("all_passed") is True):
        return False, "code/out/check19.json missing or not all_passed"
    if check.get("code_sha256") != code_now:
        return False, "code or job files differ from those checked by check19.py"
    if plan in SMOKE:
        return True, "ok"
    if plan != "main":
        return False, f"unknown plan {plan}"
    if not (smoke and smoke.get("all_passed") is True):
        return False, "data/smoke_check.json missing or not all_passed"
    if smoke.get("code_sha256") != code_now:
        return False, "code or job files differ from those of the smoke"
    cost = (smoke.get("tests") or {}).get("cost") or {}
    if not ((cost.get("main") or {}).get("within_limit") is True and (cost.get("main") or {}).get("within_cap") is True):
        return False, "the smoke's projection of the main run exceeds its limit or the cap"
    return True, "ok"


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def gate(plan, rho, base):
    ok, why = gate_state(read_json(CHECK_JSON), read_json(SMOKE_JSON), run_code_sha(), plan, rho, base)
    assert ok, f"A19 {plan}: {why}"
    assert float(rho) == rho_star(), "rho differs from a15/results/rho.json"


def manifest(plan, args, rho, base, year, world):
    return {"amendment": "A19 (a19/A19_draft.md)", "plan": plan, "args": args, "member_ids": [int(m) for m in PLANS[plan]["mids"]],
            "rho": float(rho), "base": base, "year": int(year), "world": world,
            "noise": "z(m, d, h) of arnoise22.ARNoise, keys [1521, year, d, m, k, tr, tc], for m = 0..3; z(4 + k) = -z(k)",
            "rho_json": str(A15_DIR / "results" / "rho.json"), "rho_json_sha256": sha_file(A15_DIR / "results" / "rho.json"),
            "numpy": np.__version__, "code_sha256": run_code_sha()}


# ======================================================================== output folders and atomic writes
NAME_OK = re.compile(r"(a19_run\.json|part_\d+\.npz|pop_\d+\.npy|pop_ghap_\d+\.npy|cell40_\d+\.npy|pop_idx\.npy)(\.tmp_a19_\d+)?")
CLAIMED = set()


def claim_state(names, manifest_text, want_text, fresh):
    """pure folder rule (tested in check19 GATES): names in the folder, the text of its manifest (None if absent). (ok, why)"""
    bad = [n for n in names if not NAME_OK.fullmatch(n)]
    if bad:
        return False, f"holds files that A19 does not write: {bad[:10]}"
    if fresh and names:
        return False, f"a smoke folder must start empty: {names[:10]}"
    real = [n for n in names if ".tmp" not in n]
    if real and "a19_run.json" not in real:
        return False, f"holds files without an A19 manifest: {real[:10]}"
    if manifest_text is not None and manifest_text != want_text:
        try:
            g, w = json.loads(manifest_text), json.loads(want_text)
            diff = sorted(k for k in w if g.get(k) != w[k])
        except Exception:                                           # noqa: BLE001
            diff = ["unreadable"]
        return False, f"manifest of another run or code version: {diff}"
    return True, "ok"


def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def claim(out, meta, fresh):
    """use `out` (a folder of a19/data) only if it is new or holds nothing but files of this same run (same manifest)"""
    out = Path(resolved(out))
    data = Path(resolved(DATA))
    assert out.parent == data, f"A19: {out} is not a folder of {data}"
    out.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(jsonable(meta), sort_keys=True, indent=1) + "\n"
    mf = out / "a19_run.json"
    names = sorted(os.listdir(out))
    ok, why = claim_state(names, mf.read_text() if mf.exists() else None, txt, fresh)
    assert ok, f"A19: {out}: {why}"
    if not mf.exists():
        tmp = out / f"a19_run.json.tmp_a19_{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(txt); f.flush(); os.fsync(f.fileno())
        assert tmp.read_text() == txt, tmp
        try:
            os.link(tmp, mf)                                      # atomic create-if-absent (concurrent ranks)
        except FileExistsError:
            pass
        os.unlink(tmp)
        fsync_dir(out)
    assert mf.read_text() == txt, f"A19: {mf} belongs to another run or code version"
    CLAIMED.add(out)
    return out


def _target(path):
    p = Path(resolved(path))
    assert p.parent in CLAIMED or str(p).startswith(str(A19.resolve()) + os.sep), f"A19: {p} is outside the claimed folders and a19/"
    assert not protected(p), p
    return p


def asave(path, arr):
    """np.save, atomically (temporary file, fsync, full read-back, rename, fsync of the folder)"""
    p = _target(path)
    arr = np.asarray(arr)
    tmp = p.with_name(f"{p.name}.tmp_a19_{os.getpid()}")
    with open(tmp, "wb") as f:
        np.save(f, arr); f.flush(); os.fsync(f.fileno()); size = f.tell()
    assert os.path.getsize(tmp) == size, tmp
    back = np.load(tmp, mmap_mode="r" if arr.size else None)
    ok = back.shape == arr.shape and back.dtype == arr.dtype and np.array_equal(
        np.ascontiguousarray(back).reshape(-1).view(np.uint8), np.ascontiguousarray(arr).reshape(-1).view(np.uint8))
    del back
    assert ok, f"A19: read-back of {tmp} differs"
    os.replace(tmp, p)
    fsync_dir(p.parent)


def asavez(path, **kw):
    """np.savez, atomically; every member read back through its CRC (zipfile.testzip)"""
    p = _target(path)
    tmp = p.with_name(f"{p.name}.tmp_a19_{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez(f, **kw); f.flush(); os.fsync(f.fileno()); size = f.tell()
    assert os.path.getsize(tmp) == size, tmp
    with zipfile.ZipFile(tmp) as zf:
        assert sorted(zf.namelist()) == sorted(k + ".npy" for k in kw), tmp
        assert zf.testzip() is None, f"A19: CRC read-back of {tmp} failed"
    os.replace(tmp, p)
    fsync_dir(p.parent)


def write_bytes(path, data):
    p = _target(path)
    tmp = p.with_name(f"{p.name}.tmp_a19_{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    with open(tmp, "rb") as f:
        assert f.read() == data, tmp
    os.replace(tmp, p)
    fsync_dir(p.parent)
    with open(p, "rb") as f:
        assert f.read() == data, p


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
