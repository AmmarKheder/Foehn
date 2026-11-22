#!/usr/bin/env python3
"""A16 (protocol_amendments/A16_test_year_evolving_noise.md): the evolving noise of A15 for any year, the fixed plans of
the A16 runs, their gates, output-folder claims and atomic writes. Imported first by every A16 script (numpy 1.26.3 only).

Noise. A verified copy of a15/arnoise.py, which is NOT imported: importing it installs a permanent audit hook that
refuses station files (the stored evaluation opens stations_2022.npz for pixel coordinates in G1-G9) and fixes YEAR 2021.
  innovation(), coeffs()  verbatim copies (check_cpu.py: same source text as a15/arnoise.py, same bytes as a15/golden.json
                          and as A15's own module run in a subprocess);
  ARNoise                 A15's class; every line that differs ends with '# A16-N1':
     N1  'members' is explicit and picks the stored file exactly as eval_global_green.Windows does (noise_tiles.npy for at
         most 4 members, noise_tiles16.npy for 5-16), so that base 'stored', rho 1 is the stored design for 4 AND 16
         members; the philox bytes do not depend on it.
  eps_k[tr, tc] = Generator(Philox(SeedSequence([1521, year, d, m, k, tr, tc]))).standard_normal((512, 512), float32),
  d = ds.doy[i] (0-based day of year of the start), k = 1..lead, any member m >= 0;
  z(m, d, 1) = eps_1, z(m, d, h) = rho^(h-1) eps_1 + sqrt(1 - rho^2) sum_{k=2..h} rho^(h-k) eps_k (A15 section 4).
  Members 0-3 of a 16-member run are, by the keys, the members of the 4-member run of the same (d, h).
WindowsAR22 = eval_global_green.Windows whose x0[m] = tile_window(ARNoise(year, ds.doy[i], h, rho, base, members), m, ...)
(lines that differ from Windows.__getitem__ end with '# A16-W1'). No lagged members (no stored command uses --lags).

Runs. EVAL_PLANS (eval_ar22.py) and SAMPLE_PLANS (sample16_ar.py) fix every argument of every A16 run (the stored commands
F1-F9 and F11 of natcomms/rebuild/REBUILD_PLAN.md section 1, new output folders); gate() refuses a rho other than rho* of
a15/results/rho.json (0.59, A16 section 1; rho 1 with base 'stored' only in smoke step (a)) and any non-smoke run unless
results/check_cpu.json and results/smoke_check.json say all_passed; claim() refuses an output folder holding a file that
A16 does not write, or files without the A16 manifest a16_run.json, or a manifest of another run or code version."""
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

NUMPY_REQUIRED = "1.26.3"
assert np.__version__ == NUMPY_REQUIRED, f"A16 runs only in the job container (numpy {NUMPY_REQUIRED}), not {np.__version__}"

A16 = Path(__file__).resolve().parent
ROOT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution")).resolve()
A15_DIR = ROOT / "a15"
GLOBAL1KM = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km")).resolve()
RUN = GLOBAL1KM / "runs" / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"
CKPT = RUN / "ckpts" / "epoch=01.ckpt"
FINAL = RUN / "final.ckpt"
RHO_JSON = A15_DIR / "results" / "rho.json"
RES = A16 / "results"
RHO_A16 = 0.59                                                  # A16 section 1: rho = 0.59 read from a15/results/rho.json
sys.path.insert(0, str(GLOBAL1KM))
import torch  # noqa: E402

import eval_global_green as egg  # noqa: E402  (imported unchanged, read only)
from fullflow_global import NTC, NTR, TILE  # noqa: E402

ROOT_WORD = 1521
B = TILE * TILE
H7 = 7
CAL = [1 + 24 * q for q in range(15)]                          # A15 section 4.4
CONF = [1 + 8 * j for j in range(45) if j % 3 != 0]
assert TILE == 512 and (NTR, NTC) == (29, 71) and len(CAL) == 15 and len(CONF) == 30 and not set(CAL) & set(CONF)


# ======================================================================== the noise (verified copy of a15/arnoise.py)
def innovation(year, doy, m, k, tr, tc):
    """eps_k of member m, start day doy, storage tile (tr, tc): a fresh Philox stream per 7-word key (fixed length)"""
    key = (ROOT_WORD, year, doy, m, k, tr, tc)
    for v in key:
        assert isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and 0 <= int(v) < 2 ** 32, key
    ss = np.random.SeedSequence([int(v) for v in key])
    return np.random.Generator(np.random.Philox(ss)).standard_normal((TILE, TILE), dtype=np.float32)


def coeffs(h, rho):
    """a_h1 .. a_hh of z_h = sum_k a_hk eps_k: float64, returned as float32"""
    h, rho = int(h), float(rho)
    assert h >= 1 and 0.0 <= rho <= 1.0, (h, rho)
    s = np.sqrt(1.0 - rho * rho)
    return np.array([rho ** (h - 1)] + [s * rho ** (h - k) for k in range(2, h + 1)], np.float64).astype(np.float32)


class ARNoise:
    """duck-typed Npy of z(m, d, h) for one (year, start day, lead, rho): shape (members, 29, 71, 512, 512), '<f2'"""

    def __init__(self, year, doy, lead, rho, base, members):  # A16-N1
        assert base in ("philox", "stored"), base
        self.year, self.doy, self.lead, self.rho, self.base = int(year), int(doy), int(lead), float(rho), base
        assert 1 <= self.lead <= H7 and 0.0 <= self.rho <= 1.0, (lead, rho)
        self.identity = base == "stored" and self.rho == 1.0
        self.c = coeffs(self.lead, self.rho)
        assert int(members) >= 1 and (base == "philox" or int(members) <= 16), (base, members)  # A16-N1
        self.file = "noise_tiles.npy" if int(members) <= 4 else "noise_tiles16.npy"  # A16-N1 (as eval_global_green.Windows)
        self.shape = (int(members), NTR, NTC, TILE, TILE)  # A16-N1
        self.dtype = np.dtype("<f2")
        self._npy = None

    @property
    def npy(self):                             # noise_tiles.npy, opened lazily (egg.Npy, os.pread)
        if self._npy is None:
            self._npy = egg.Npy(egg.G / self.file)  # A16-N1
            assert self._npy.dtype == self.dtype and tuple(self._npy.shape[1:]) == (NTR, NTC, TILE, TILE), self._npy.shape
        return self._npy

    def _stored(self, m, tr, tc):
        assert 0 <= m < self.npy.shape[0], m
        return self.npy._read(((m * NTR + tr) * NTC + tc) * B, B).reshape(TILE, TILE)

    def _arith(self, m, tr, tc):
        acc = np.zeros((TILE, TILE), np.float32)
        for k in range(1, self.lead + 1):
            ck = self.c[k - 1]
            if ck == 0:
                continue
            e = self._stored(m, tr, tc).astype(np.float32) if (k == 1 and self.base == "stored") else \
                innovation(self.year, self.doy, m, k, tr, tc)
            acc += ck * e
        return acc.astype(np.float16)

    def tile(self, m, tr, tc):
        return self._stored(m, tr, tc) if self.identity else self._arith(m, tr, tc)

    def _read(self, offset, count):
        """count values from flat offset (whole tiles only; a read may cross tile rows and members)"""
        assert offset % B == 0 and count % B == 0 and count > 0, (offset, count)
        t0, n = offset // B, count // B
        out = np.empty(count, np.float16)
        for j in range(n):
            m, r = divmod(t0 + j, NTR * NTC)
            tr, tc = divmod(r, NTC)
            assert m < self.shape[0], (m, self.shape)
            out[j * B:(j + 1) * B] = self.tile(m, tr, tc).reshape(-1)
        return out

    def __getstate__(self):                    # DataLoader workers reopen their own descriptor
        d = dict(self.__dict__)
        d["_npy"] = None
        return d


class WindowsAR22(egg.Windows):
    """eval_global_green.Windows with the evolving noise: jobs [(start index, lead, window)]; one year, rho, base per run"""

    def __init__(self, ds, jobs, win, members, lags, year, rho, base):
        assert base in ("philox", "stored") and (base == "philox" or members <= 16), (base, members)
        self.ds, self.jobs, self.win, self.members, self.lags = ds, jobs, win, members, sorted({l for l in lags if l > 0})
        assert not self.lags, "A16: no lagged members"
        self.year, self.rho, self.base = int(year), float(rho), base

    def __getitem__(self, n):
        i, h, w = self.jobs[n]
        k0, j0 = (int(v) for v in self.win[w])
        b = self.ds.item(i, h, k0, j0)
        del b["x1"], b["mask"]
        assert int(self.ds.year[i]) == self.year, (int(self.ds.year[i]), self.year)  # A16-W1
        nz = ARNoise(self.year, int(self.ds.doy[i]), h, self.rho, self.base, self.members)  # A16-W1
        b["x0"] = torch.from_numpy(np.stack([egg.tile_window(nz, m, 25 * k0, 25 * j0, egg.PH, egg.PW) for m in range(self.members)]))  # A16-W1
        for l in self.lags:                                              # lagged weather runs (members with --lags l)
            for k, v in self.ds.lagged(i, h, k0, j0, l).items():
                b[f"lag{l}_{k}"] = v
        return b


REMOVED_LINES_ARNOISE = [   # the lines of a15/arnoise.py ARNoise that do not survive (check_cpu.py DIFF)
    'def __init__(self, year, doy, lead, rho, base, members=2 ** 20):',
    'self.shape = (int(members) if base == "philox" else min(int(members), 4), NTR, NTC, TILE, TILE)',
    'self._npy = egg.Npy(egg.G / "noise_tiles.npy")',
]
REMOVED_LINES_WINDOWS = [   # the line of eval_global_green.Windows.__getitem__ that does not survive
    'b["x0"] = torch.from_numpy(np.stack([tile_window(self.noise, m, 25 * k0, 25 * j0, PH, PW) for m in range(self.members)]))',
]


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


GOLDEN_KEYS = ((2021, 1, 0, 1, 0, 0), (2021, 353, 3, 7, 28, 70), (2021, 9, 1, 2, 14, 35), (2021, 177, 63, 4, 7, 60))  # A15's
GOLDEN22_KEYS = ((2022, 0, 0, 1, 0, 0), (2022, 356, 3, 7, 28, 70), (2022, 2, 15, 3, 14, 35), (2022, 19, 1, 2, 7, 60),
                 (2022, 189, 12, 5, 10, 40))


def golden_hashes():
    """the hashes of a15/golden.json, recomputed with this copy (2021 keys)"""
    out = {"numpy": np.__version__}
    for key in GOLDEN_KEYS:
        e = innovation(*key)
        nm = "eps_" + "_".join(str(v) for v in key)
        out[nm + "/f32"], out[nm + "/f16"] = sha(e), sha(e.astype(np.float16))
    out["z_2021_9_lead7_rho0.7_philox_m2_tile14_35/f16"] = sha(ARNoise(2021, 9, 7, 0.7, "philox", 4).tile(2, 14, 35))
    return out


def golden22_hashes():
    """golden22.json: 2022 keys (innovations, and z at rho 0.59 for members 15 and 0)"""
    out = {"numpy": np.__version__}
    for key in GOLDEN22_KEYS:
        e = innovation(*key)
        nm = "eps_" + "_".join(str(v) for v in key)
        out[nm + "/f32"], out[nm + "/f16"] = sha(e), sha(e.astype(np.float16))
    out["z_2022_2_lead3_rho0.59_philox_m15_tile14_35/f16"] = sha(ARNoise(2022, 2, 3, 0.59, "philox", 16).tile(15, 14, 35))
    out["z_2022_0_lead7_rho0.59_philox_m0_tile7_30/f16"] = sha(ARNoise(2022, 0, 7, 0.59, "philox", 4).tile(0, 7, 30))
    return out


# ======================================================================== the fixed plans of the A16 runs
_EA, _KANO, _GIR, _GIRL = list(range(28)), [132, 133, 134], list(range(192, 204)), list(range(189, 204))
_E = dict(every=1, doys=None, leads=None, save_1km=[], crop=None, crop_only=False, crop_days=[], pop=False, year=2022,
          members=4, run=str(RUN), ckpt=str(CKPT), steps=4, refine_tail=1, psub=16)


def _p(**kw):
    d = dict(_E)
    d.update(kw)
    return d


EVAL_PLANS = {   # out, then the arguments of the stored command (REBUILD_PLAN section 1), forecasts n, GCDs (world)
    "G1": _p(out=str(RUN / "eval_final_crps_e01_ar"), every=2, n=1253, world=64, stored="eval_final_crps_e01"),
    "G2": _p(out=str(RUN / "eval_pop_crps_e01_ar"), every=2, leads=[1, 3], pop=True, n=358, world=64, stored="eval_pop_crps_e01"),
    "G3": _p(out=str(RUN / "eval_jan21_crps_e01_ar"), doys=[19], leads=[1], save_1km=[19], n=1, world=8, stored="eval_jan21_crps_e01"),
    "G4": _p(out=str(RUN / "eval_jan21_leads_crps_e01_ar"), doys=[19], leads=list(range(1, 8)), save_1km=[19], n=7, world=8,
             stored="eval_jan21_leads_crps_e01"),
    "G5": _p(out=str(RUN / "eval_case_eastasia_crps_e01_ar"), members=16, crop=[[42.0, 33.0, 112.0, 131.0]], crop_only=True,
             crop_days=_EA, doys=_EA, n=196, world=32, stored="eval_case_eastasia_crps_e01"),
    "G6": _p(out=str(RUN / "eval_case_kano_crps_e01_ar"), members=16, crop=[[28.0, 6.0, 0.0, 26.0]], crop_only=True,
             crop_days=_KANO, doys=_KANO, leads=[1, 2, 3], n=9, world=8, stored="eval_case_kano_crps_e01"),
    "G7": _p(out=str(RUN / "eval_case_gironde_crps_e01_ar"), members=16, crop=[[47.0, 43.0, -3.0, 1.0]], crop_only=True,
             crop_days=_GIR, doys=_GIR, leads=[1, 2, 3], n=36, world=8, stored="eval_case_gironde_crps_e01"),
    "G8": _p(out=str(RUN / "eval_case_gironde_long_crps_e01_ar"), members=16, crop=[[47.0, 43.0, -3.0, 1.0]], crop_only=True,
             crop_days=_GIRL, doys=_GIRL, leads=[4, 5, 6, 7], n=60, world=8, stored="eval_case_gironde_long_crps_e01"),
    "G9": _p(out=str(A16 / "data" / "cal2021"), year=2021, doys=list(CAL), leads=[2, 3, 4, 5, 6, 7], n=90, world=8,
             stored="eval2021_sel_crps_e01"),
    "S_a": _p(out=str(A16 / "data" / "smoke" / "a"), doys=[0], leads=[1, 2], n=2, world=1, stored="eval_final_crps_e01"),
    "S_b4": _p(out=str(A16 / "data" / "smoke" / "b4"), pop=True, doys=[2], leads=[3], n=1, world=1, stored=None),
    "S_c": _p(out=str(A16 / "data" / "smoke" / "c"), year=2021, doys=[9], leads=[2], n=1, world=1, stored=None),
}
EVAL_KEYS = sorted(_E) + ["out"]
SAMPLE_PLANS = {  # the 16-member sampler at the populated pixels (a13/warn16/sample16.py; the stored command uses final.ckpt)
    "G11": dict(out=str(A16 / "data" / "run16"), ckpt=str(FINAL), members=16, steps=4, psub=16, max_lead=7, doys=None,
                pairs_file=str(A16 / "jobs" / "pairs_g11.txt"), n=358),
    "S_b16": dict(out=str(A16 / "data" / "smoke" / "b16"), ckpt=str(FINAL), members=16, steps=4, psub=16, max_lead=7, doys=None,
                  pairs=["2:3"], tag="s", n=1),
}
SMOKE = {"S_a", "S_b4", "S_c", "S_b16"}


def pairs_g11():
    """the 45 lines of jobs/pairs_g11.txt: 179 start days (0, 2, ..., 356) x leads 1 and 3, 8 forecasts per line"""
    lines = [ln.split() for ln in Path(SAMPLE_PLANS["G11"]["pairs_file"]).read_text().split("\n") if ln.strip()]
    return lines


def resolved(p):
    p = Path(p)
    return str(p.parent.resolve() / p.name)


def eval_args(a):
    """the canonical arguments of an eval_ar22.py run (for the plan check and the manifest)"""
    return dict(run=resolved(a.run), ckpt=resolved(a.ckpt) if a.ckpt else None, out=resolved(Path(a.run) / a.out),
                members=a.members, steps=a.steps,
                refine_tail=a.refine_tail, psub=a.psub, every=a.every, doys=a.doys, leads=a.leads, save_1km=list(a.save_1km),
                crop=[list(c) for c in a.crop] if a.crop else None, crop_only=bool(a.crop_only), crop_days=list(a.crop_days),
                pop=bool(a.pop), year=a.year)


def plan_match(plan, got):
    """the arguments given must be exactly those of the plan (two independent sources: the job script and this table)"""
    want = {k: v for k, v in EVAL_PLANS[plan].items() if k in EVAL_KEYS}
    want["run"], want["ckpt"], want["out"] = resolved(want["run"]), resolved(want["ckpt"]), resolved(want["out"])
    bad = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
    assert not bad and set(got) == set(want), f"A16 plan {plan}: arguments differ from the plan (given, plan): {bad}"
    return want


# ======================================================================== gates
def passed(path):
    p = Path(path)
    return p.exists() and json.loads(p.read_text()).get("all_passed") is True


def rho_star():
    rj = json.loads(RHO_JSON.read_text())
    assert rj.get("clipped") is False and rj.get("all_passed") is True, f"{RHO_JSON}: clipped or not passed"
    r = float(rj["rho_star"])
    assert r == RHO_A16, f"rho* of {RHO_JSON} is {r}, A16 section 1 says {RHO_A16}"
    return r


RUN_CODE = ("arnoise22.py", "eval_ar22.py", "sample16_ar.py", "jobs/g1.sh", "jobs/g2.sh", "jobs/g3.sh", "jobs/g4.sh", "jobs/g5.sh",
            "jobs/g6.sh", "jobs/g7.sh", "jobs/g8.sh", "jobs/g9.sh", "jobs/g11.sh", "jobs/smoke.sh", "jobs/pairs_g11.txt")
STORED_CODE = ("eval_global_green", "fullflow_global", "fullflow_global_fp", "fullflow_global_green", "fullflow", "dejavu")


def run_code_sha():
    """sha256 of the code and job files of the runs (recorded by check_cpu.py and smoke_check.py; gate() compares)"""
    d = {n: sha_file(A16 / n) for n in RUN_CODE}
    d.update({m: sha_file(sys.modules[m].__file__) for m in STORED_CODE})
    return d


def same_code(path):
    p = Path(path)
    return p.exists() and json.loads(p.read_text()).get("code_sha256") == run_code_sha()


def gate(plan, rho, base):
    """refuse anything but the A16 design: rho* (philox) for every run, rho 1 (stored) for smoke step (a) only; the code
    must be the code that check_cpu.py (and, for the runs, smoke_check.py) checked"""
    assert passed(RES / "check_cpu.json"), "A16: results/check_cpu.json missing or not all_passed"
    assert same_code(RES / "check_cpu.json"), "A16: the code or job files differ from those checked by check_cpu.py"
    if plan == "S_a":
        assert base == "stored" and float(rho) == 1.0, f"A16 smoke (a) is base stored, rho 1 (got {base}, {rho})"
        return
    assert base == "philox" and float(rho) == rho_star(), f"A16 runs use base philox and rho* = {rho_star()} (got {base}, {rho})"
    if plan not in SMOKE:
        assert passed(RES / "smoke_check.json"), "A16: results/smoke_check.json missing or not all_passed"
        assert same_code(RES / "smoke_check.json"), "A16: the code or job files differ from those of the smoke"


def code_sha(extra=()):
    """sha256 of the code that runs: A16 files, eval_global_green and the modules it imports"""
    files = [A16 / "arnoise22.py"] + [Path(e) for e in extra]
    for mod in ("eval_global_green", "fullflow_global", "fullflow_global_fp", "fullflow_global_green", "fullflow", "dejavu"):
        files.append(Path(sys.modules[mod].__file__))
    return {str(Path(f).resolve()): sha_file(f) for f in files}


def manifest(plan, args, rho, base, year, world, extra_code=()):
    return {"amendment": "A16 (protocol_amendments/A16_test_year_evolving_noise.md)", "plan": plan, "args": args,
            "rho": float(rho), "base": base, "year": int(year), "world": world, "noise_key": "[1521, year, d, m, k, tr, tc]",
            "rho_json": str(RHO_JSON), "rho_json_sha256": sha_file(RHO_JSON), "numpy": np.__version__,
            "code_sha256": code_sha(extra_code)}


# ======================================================================== output folders and atomic writes
NAME_OK = re.compile(r"(a16_run\.json|part_\d+\.npz|pop_\d+\.npy|pop_ghap_\d+\.npy|cell40_\d+\.npy|pop_idx\.npy"
                     r"|map5km_d\d+_h\d+\.npy|map1km_d\d+_h\d+(_member0)?\.npy|crop\d+_d\d+_h\d+\.npz"
                     r"|pop16_[a-z0-9]+\.npy|keys16_[a-z0-9]+\.npy|done16_[a-z0-9]+\.json|done16_[a-z0-9]+\.tmp\d+)(\.tmp_a16_\d+)?")
CLAIMED = set()


def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def claim(out, meta, fresh):
    """use `out` only if it is new, or holds nothing but files of this same run (same manifest); fresh: must be empty"""
    out = Path(resolved(out))
    out.mkdir(parents=str(out).startswith(str(A16) + os.sep), exist_ok=True)
    names = sorted(os.listdir(out))
    bad = [n for n in names if not NAME_OK.fullmatch(n)]
    assert not bad, f"A16: {out} holds files that A16 does not write: {bad[:10]}"
    real = [n for n in names if ".tmp" not in n]
    assert not (fresh and names), f"A16: the smoke folder {out} must be empty: {names[:10]}"
    assert not real or "a16_run.json" in real, f"A16: {out} holds files without an A16 manifest: {real[:10]}"
    txt = json.dumps(jsonable(meta), sort_keys=True, indent=1) + "\n"
    mf = out / "a16_run.json"
    if not mf.exists():
        tmp = out / f"a16_run.json.tmp_a16_{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(txt); f.flush(); os.fsync(f.fileno())
        assert tmp.read_text() == txt, tmp
        try:
            os.link(tmp, mf)                                      # atomic create-if-absent (concurrent ranks / array tasks)
        except FileExistsError:
            pass
        os.unlink(tmp)
        fsync_dir(out)
    got = mf.read_text()
    if got != txt:
        g, w = json.loads(got), json.loads(txt)
        raise AssertionError(f"A16: {mf} belongs to another run or code version: {sorted(k for k in w if g.get(k) != w[k])}")
    CLAIMED.add(out)
    return out


def _target(path):
    p = Path(resolved(path))
    assert p.parent in CLAIMED or str(p).startswith(str(A16) + os.sep), f"A16: {p} is outside the claimed folders and a16/"
    return p


def asave(path, arr):
    """np.save, atomically (temporary file, fsync, full read-back, rename, fsync of the folder)"""
    p = _target(path)
    arr = np.asarray(arr)
    tmp = p.with_name(f"{p.name}.tmp_a16_{os.getpid()}")
    with open(tmp, "wb") as f:
        np.save(f, arr); f.flush(); os.fsync(f.fileno()); size = f.tell()
    assert os.path.getsize(tmp) == size, tmp
    back = np.load(tmp, mmap_mode="r" if arr.size else None)
    ok = back.shape == arr.shape and back.dtype == arr.dtype and np.array_equal(
        np.ascontiguousarray(back).reshape(-1).view(np.uint8), np.ascontiguousarray(arr).reshape(-1).view(np.uint8))
    del back
    assert ok, f"A16: read-back of {tmp} differs"
    os.replace(tmp, p)
    fsync_dir(p.parent)


def asavez(path, **kw):
    """np.savez, atomically; every member read back through its CRC (zipfile.testzip)"""
    p = _target(path)
    tmp = p.with_name(f"{p.name}.tmp_a16_{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez(f, **kw); f.flush(); os.fsync(f.fileno()); size = f.tell()
    assert os.path.getsize(tmp) == size, tmp
    with zipfile.ZipFile(tmp) as zf:
        assert sorted(zf.namelist()) == sorted(k + ".npy" for k in kw), tmp
        assert zf.testzip() is None, f"A16: CRC read-back of {tmp} failed"
    os.replace(tmp, p)
    fsync_dir(p.parent)


def write_bytes(path, data):
    p = _target(path)
    tmp = p.with_name(f"{p.name}.tmp_a16_{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    with open(tmp, "rb") as f:
        assert f.read() == data, tmp
    os.replace(tmp, p)
    fsync_dir(p.parent)


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
