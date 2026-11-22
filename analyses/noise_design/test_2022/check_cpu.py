#!/usr/bin/env python3
"""A16 check C1 (protocol_amendments/A16_test_year_evolving_noise.md section 3), CPU only, in the job container (numpy 1.26.3):
  GOLD15     the generator of arnoise22.py gives the golden hashes of a15/golden.json (2021 keys), here and in two processes
  A15_BYTES  z tiles and a window cut for 2021 keys (rho 0.59 and 0.7 philox, stored with rho 1 and 0.59) are byte-equal to
             those of A15's own module a15/arnoise.py, run in a subprocess (its audit hook stays in that subprocess)
  STORED4    base stored, rho 1, 4 members: the whole of noise_tiles.npy read through ARNoise, and every window cut of the
             21 windows x 4 members x 7 leads through the unchanged tile_window, are uint16-equal to noise_tiles.npy
  STORED16   the same for 16 members and noise_tiles16.npy (whole file; window cuts at leads 1 and 7, leads 2-6 on 3 windows)
  ARITH      the arithmetic path with coefficients (1, 0, ..., 0) returns the stored bytes (4 and 16 members)
  WINDOWS    x0 of WindowsAR22(stored, rho 1) equals x0 of eval_global_green.Windows on the real 2022 dataset (4 members:
             12 random items, 16 members: 5), all other inputs equal; with philox it is the ARNoise cut of year 2022
  KEYS22     2022 keys differ from 2021 keys (innovation bytes; |correlation| < 0.01), also for z at rho 0.59
  LAW22      A15 I6 on 2022 starts 0 and 2 (rho 0.5, 0.59, 0.9; members 0, 1, 15; tiles (7, 30), (20, 50); leads 1-7):
             mean 0 +- 0.01, variance 1 +- 0.01, lag-k correlation rho^k +- 0.01, |correlation| < 0.01 between members,
             starts, neighbouring pixels and members at any pair of leads; rho 0: |lag correlation| < 0.01
  LEAD1      z(m, d, 1) is the same for rho 0.5, 0.59 and 0.9 and equals eps_1 (2022)
  SEAMS      full_map(ARNoise(2022, 2, 3, 0.59, philox, 16), member 15) equals each of its 21 window cuts
  NESTED     members 0-3 of a 16-member ARNoise are byte-equal to the 4-member ARNoise (philox; stored: noise_tiles16 = noise_tiles)
  GOLD22     golden22.json hashes (2022 keys) identical here and in two processes (and equal to golden22.json once written)
  LOADER     identical windows through a DataLoader with 0 and 4 workers (fork and spawn) and after a pickle round trip
  DIFF       every line of eval_ar22.run_ar22 that differs from eval_global_green.run ends with '# A16-H<n>' (n listed),
             the lines removed are exactly eval_ar22.REMOVED_LINES; the same for sample16_ar.run against a13 sample16.run
             (S<n>), ARNoise against a15/arnoise.py (N1), WindowsAR22.__getitem__ against Windows.__getitem__ (W1);
             innovation, coeffs (a15) and _audit, pop_windows (sample16) are verbatim
  PLANS      every plan's forecast count from the real datasets (TestGreen 2022 and 2021, the stored forecast list) equals the
             plan; G1's (start, lead) = the keys of eval_final_crps_e01, G2's and G11's = the keys of eval_pop_crps_e01;
             the arguments in every job script parse and equal their plan (sample16_ar in a subprocess: its audit hook)
  GATES      gate() refuses another rho, base or a missing smoke; plan_match refuses changed arguments; claim() refuses a
             foreign file, files without manifest, another manifest, a non-empty smoke folder; atomic writes read back
  cost       CPU seconds to build one window of noise (worker cost), reported
  code_sha256 of the run code and job files (arnoise22.RUN_CODE and the stored modules): gate() refuses other code
    python check_cpu.py -> results/check_cpu.json (exit 1 if a check fails); golden22.json written once, by a fully passing run"""
import arnoise22 as an  # noqa: I001  (first)
import ast
import collections
import difflib
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import eval_ar22

egg = an.egg
PH, PW, GH, GW = egg.PH, egg.PW, egg.GH, egg.GW
RESF = an.RES / "check_cpu.json"
GOLD15 = an.A15_DIR / "golden.json"
GOLD22 = an.A16 / "golden22.json"
S16_SRC = an.ROOT / "a13" / "warn16" / "sample16.py"
S16_NEW = an.A16 / "sample16_ar.py"
R = {"what": __doc__, "numpy": np.__version__, "torch": torch.__version__, "python": sys.version.split()[0], "tests": {}}
WIN = egg.windows(np.load(egg.G / "meta.npz")["land"])
assert len(WIN) == 21, len(WIN)
DS = {}


def check(name, cond, **info):
    R["tests"][name] = {"pass": bool(cond), **info}
    print(("PASS " if cond else "FAIL ") + name + " " + json.dumps(an.jsonable(info))[:700], flush=True)


def guarded(name, fn):
    t = time.time()
    try:
        fn()
    except Exception as e:                                              # one failing check must not hide the others
        traceback.print_exc()
        check(name, False, error=repr(e))
    R["tests"].setdefault(name, {"pass": False, "error": "no result"})
    R["tests"][name]["seconds"] = round(time.time() - t, 1)


def u16(a):
    return np.ascontiguousarray(a).view(np.uint16)


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def dataset(year):
    if year not in DS:
        state = torch.load(an.CKPT, map_location="cpu", weights_only=False)
        hp = state["hyper_parameters"]
        del state
        store, emis = hp.get("store", "green"), bool(hp.get("emis", False))
        P = int(egg.Npy(egg.G / store / "hres" / f"origins_{year}.npy").shape[-2])
        DS[year] = (egg.TestGreen(7, P, store, emis, year, hp.get("emis_store")), store, P)
    return DS[year][0]


# ---------------------------------------------------------------- DIFF
def fsrc(path, name, cls=None):
    """source lines of a top-level function / class (or a method of cls) of a file, as inspect.getsource gives them"""
    txt = Path(path).read_text()
    body, lines = ast.parse(txt).body, txt.splitlines()
    if cls:
        body = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls][0].body
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            return lines[n.lineno - 1:n.end_lineno]
    raise KeyError(name)


def const(path, name):
    for n in ast.parse(Path(path).read_text()).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name:
            return ast.literal_eval(n.value)
    raise KeyError(name)


def diff(src, new, letter, declared, hunks_declared):
    mk = re.compile(r"  # A16-%s(\d+)\b" % letter)
    removed, unmarked, hunks = [], [], collections.Counter()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        removed += [l.strip() for l in src[i1:i2]]
        for l in new[j1:j2]:
            m = mk.search(l)
            if m:
                hunks[int(m.group(1))] += 1
            else:
                unmarked.append(l)
    marked_total = sum(1 for l in new if mk.search(l))
    same = collections.Counter(removed) == collections.Counter(declared)
    ok = not unmarked and same and marked_total == sum(hunks.values()) and set(hunks) == set(hunks_declared)
    return ok, dict(original_lines=len(src), new_lines=len(new), removed=len(removed), declared_removed=len(declared),
                    unmarked_changed_lines=unmarked, marked_lines_by_hunk={f"{letter}{k}": v for k, v in sorted(hunks.items())},
                    removed_equal_declared=same, removed_not_declared=[l for l in removed if l not in declared],
                    declared_not_removed=[l for l in declared if l not in removed])


def t_diff():
    E = Path(egg.__file__)
    ok_e, i_e = diff(fsrc(E, "run"), fsrc(eval_ar22.__file__, "run_ar22"), "H", eval_ar22.REMOVED_LINES, eval_ar22.HUNKS)
    ok_s, i_s = diff(fsrc(S16_SRC, "run"), fsrc(S16_NEW, "run"), "S", const(S16_NEW, "REMOVED_LINES"), const(S16_NEW, "HUNKS"))
    A15 = an.A15_DIR / "arnoise.py"
    A = Path(an.__file__)
    ok_n, i_n = diff(fsrc(A15, "ARNoise"), fsrc(A, "ARNoise"), "N", an.REMOVED_LINES_ARNOISE, (1,))
    ok_w, i_w = diff(fsrc(E, "__getitem__", "Windows"), fsrc(A, "__getitem__", "WindowsAR22"), "W", an.REMOVED_LINES_WINDOWS, (1,))
    verb = {"innovation_a15": fsrc(A15, "innovation") == fsrc(A, "innovation"), "coeffs_a15": fsrc(A15, "coeffs") == fsrc(A, "coeffs"),
            "_audit_sample16": fsrc(S16_SRC, "_audit") == fsrc(S16_NEW, "_audit"),
            "pop_windows_sample16": fsrc(S16_SRC, "pop_windows") == fsrc(S16_NEW, "pop_windows"),
            "hook_installed_first": "sys.addaudithook(_audit)" in S16_NEW.read_text().split("import torch")[0]}
    check("DIFF", ok_e and ok_s and ok_n and ok_w and all(verb.values()), files={"eval_global_green": str(E), "sample16": str(S16_SRC),
          "a15_arnoise": str(A15)}, eval_ar22=i_e, sample16_ar=i_s, ARNoise=i_n, WindowsAR22=i_w, verbatim=verb)


# ---------------------------------------------------------------- golden hashes, A15 bytes
def subrun(code, prefix, cwd):
    p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=str(cwd), timeout=1800)
    line = [l for l in p.stdout.splitlines() if l.startswith(prefix + " ")]
    return json.loads(line[-1][len(prefix) + 1:]) if p.returncode == 0 and line else {"error": p.stderr[-3000:]}


SUB = "import sys; sys.path.insert(0, %r); import arnoise22, json; print('GOLDEN ' + json.dumps(arnoise22.%s(), sort_keys=True), flush=True)"


def t_gold15():
    here = an.golden_hashes()
    subs = [subrun(SUB % (str(an.A16), "golden_hashes"), "GOLDEN", an.A16) for _ in range(2)]
    gold = json.loads(GOLD15.read_text())
    check("GOLD15", here == gold and all(s == here for s in subs), golden_json=str(GOLD15), equal_to_a15_golden=here == gold,
          subprocess_equal=[s == here for s in subs], mismatched=[k for k in gold if here.get(k) != gold[k]])


A15_ITEMS = [(9, 2, 0.59, "philox", 0, 14, 35), (9, 7, 0.59, "philox", 3, 0, 0), (353, 7, 0.59, "philox", 1, 28, 70),
             (105, 4, 0.59, "philox", 2, 7, 30), (1, 1, 0.59, "philox", 0, 3, 66), (1, 7, 0.7, "philox", 2, 20, 50),
             (9, 3, 1.0, "stored", 1, 14, 35), (17, 5, 0.59, "stored", 3, 7, 30)]
SUB15 = ("import sys, json; sys.path.insert(0, %r); import arnoise as ar; import numpy as np; items = %r; "
         "out = [ar.sha(ar.ARNoise(2021, d, h, r, b).tile(m, tr, tc)) for d, h, r, b, m, tr, tc in items]; "
         "W = ar.egg.windows(np.load(ar.egg.G / 'meta.npz')['land']); k0, j0 = W[10]; "
         "out.append(ar.sha(ar.egg.tile_window(ar.ARNoise(2021, 9, 2, 0.59, 'philox'), 1, 25 * k0, 25 * j0, ar.egg.PH, ar.egg.PW))); "
         "print('A15SHA ' + json.dumps(out), flush=True)")


def t_a15_bytes():
    mine = [an.sha(an.ARNoise(2021, d, h, r, b, 4).tile(m, tr, tc)) for d, h, r, b, m, tr, tc in A15_ITEMS]
    k0, j0 = WIN[10]
    mine.append(an.sha(egg.tile_window(an.ARNoise(2021, 9, 2, 0.59, "philox", 4), 1, 25 * k0, 25 * j0, PH, PW)))
    theirs = subrun(SUB15 % (str(an.A15_DIR), A15_ITEMS), "A15SHA", an.A15_DIR)
    check("A15_BYTES", mine == theirs, items=[list(x) + ["tile"] for x in A15_ITEMS] + [["window 10 cut", 9, 2, 0.59, "philox", 1]],
          equal=[a == b for a, b in zip(mine, theirs)] if isinstance(theirs, list) else theirs, first16=[m[:16] for m in mine])


# ---------------------------------------------------------------- identity with the stored fields
def whole_file(M, fname):
    npy = egg.Npy(egg.G / fname)
    assert npy.shape[0] == M, (fname, npy.shape)
    nz = an.ARNoise(2022, 0, 7, 1.0, "stored", M)
    assert nz.file == fname, (nz.file, fname)
    bad, n = 0, an.NTR * an.NTC * an.B
    for m in range(M):
        bad += not np.array_equal(u16(nz._read(m * n, n)), u16(npy._read(m * n, n)))
    return npy, bad


def t_stored4():
    npy, bad_file = whole_file(4, "noise_tiles.npy")
    bad = n = 0
    for w, (k0, j0) in enumerate(WIN):
        d = 2 * (w % 179)
        for m in range(4):
            ref = u16(egg.tile_window(npy, m, 25 * k0, 25 * j0, PH, PW))
            for h in range(1, 8):
                got = u16(egg.tile_window(an.ARNoise(2022, d, h, 1.0, "stored", 4), m, 25 * k0, 25 * j0, PH, PW))
                n += 1
                bad += not np.array_equal(ref, got)
    check("STORED4", bad_file == 0 and bad == 0 and n == 21 * 4 * 7, whole_file_members_mismatching=bad_file, window_cuts=n,
          window_mismatches=bad, file="noise_tiles.npy")


def t_stored16():
    npy, bad_file = whole_file(16, "noise_tiles16.npy")
    bad = n = 0
    for w, (k0, j0) in enumerate(WIN):
        for m in range(16):
            ref = u16(egg.tile_window(npy, m, 25 * k0, 25 * j0, PH, PW))
            for h in ((1, 7) + ((2, 3, 4, 5, 6) if w in (0, 10, 20) else ())):
                got = u16(egg.tile_window(an.ARNoise(2022, 100 + w, h, 1.0, "stored", 16), m, 25 * k0, 25 * j0, PH, PW))
                n += 1
                bad += not np.array_equal(ref, got)
    check("STORED16", bad_file == 0 and bad == 0 and n == 21 * 16 * 2 + 3 * 16 * 5, whole_file_members_mismatching=bad_file,
          window_cuts=n, window_mismatches=bad, file="noise_tiles16.npy")


TILES = ((0, 0), (7, 30), (14, 35), (20, 50), (28, 70), (3, 66))


def t_arith():
    coef_ok = all(np.array_equal(an.coeffs(h, 1.0), np.r_[1.0, np.zeros(h - 1)].astype(np.float32)) for h in range(1, 8))
    bad = n = 0
    for M, fname, ms in ((4, "noise_tiles.npy", range(4)), (16, "noise_tiles16.npy", (0, 5, 11, 15))):
        npy = egg.Npy(egg.G / fname)
        for h in range(1, 8):
            nz = an.ARNoise(2022, 4, h, 1.0, "stored", M)
            for m in ms:
                for tr, tc in TILES:
                    ref = npy._read(((m * an.NTR + tr) * an.NTC + tc) * an.B, an.B).reshape(an.TILE, an.TILE)
                    n += 1
                    bad += not np.array_equal(u16(nz._arith(m, tr, tc)), u16(ref))
    check("ARITH", coef_ok and bad == 0 and n == 7 * 8 * 6, compared_tiles=n, mismatches=bad, coefficients_exact=coef_ok)


def t_windows():
    ds = dataset(2022)
    samples = sorted(ds.samples)
    rng = np.random.default_rng(1622)
    out, ok = {}, True
    for M, nitem in ((4, 12), (16, 5)):
        pick = rng.choice(len(samples), nitem, replace=False)
        jobs = [(samples[j][0], samples[j][1], int(w)) for j, w in zip(pick, rng.integers(0, len(WIN), nitem))]
        A = egg.Windows(ds, jobs, WIN, M)
        Bw = an.WindowsAR22(ds, jobs, WIN, M, [0] * M, 2022, 1.0, "stored")
        Cw = an.WindowsAR22(ds, jobs, WIN, M, [0] * M, 2022, 0.59, "philox")
        bad_x0 = bad_other = bad_ph = 0
        keys_same = True
        for n in range(len(jobs)):
            a, b = A[n], Bw[n]
            c = Cw[n] if (M == 4 or n < 2) else b
            keys_same &= set(a) == set(b) == set(c)
            bad_x0 += not np.array_equal(u16(a["x0"].numpy()), u16(b["x0"].numpy()))
            bad_other += sum(not (a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].numpy().tobytes() == b[k].numpy().tobytes())
                             for k in a if k != "x0")
            if M == 4 or n < 2:                                         # 16 members: 2 items (about 50 s of noise each)
                i, h, w = jobs[n]
                k0, j0 = WIN[w]
                nz = an.ARNoise(2022, int(ds.doy[i]), h, 0.59, "philox", M)
                ref = np.stack([egg.tile_window(nz, m, 25 * k0, 25 * j0, PH, PW) for m in range(M)])
                bad_ph += not np.array_equal(u16(c["x0"].numpy()), u16(ref))
        good = bad_x0 == 0 and bad_other == 0 and bad_ph == 0 and keys_same
        ok &= good
        out[f"members_{M}"] = dict(items=len(jobs), x0_mismatches_stored=bad_x0, other_inputs_mismatching=bad_other,
                                   philox_x0_mismatches=bad_ph, same_keys=keys_same, jobs=[(int(ds.doy[i]), h, w) for i, h, w in jobs])
    check("WINDOWS", ok, store=DS[2022][1], particles=DS[2022][2], n_samples_2022=len(samples), **out)


# ---------------------------------------------------------------- the law and keys of 2022
def t_keys22():
    keys = [(d, m, k, tr, tc) for d, m, k, tr, tc in ((0, 0, 1, 0, 0), (2, 3, 2, 14, 35), (9, 15, 7, 28, 70), (19, 1, 1, 7, 30),
                                                        (189, 8, 4, 20, 50), (353, 2, 3, 3, 66))]
    same = maxc = 0.0
    for d, m, k, tr, tc in keys:
        a, b = an.innovation(2022, d, m, k, tr, tc), an.innovation(2021, d, m, k, tr, tc)
        same += np.array_equal(a.view(np.uint32), b.view(np.uint32))
        maxc = max(maxc, abs(corr(a.astype(np.float64), b.astype(np.float64))))
    zc = max(abs(corr(an.ARNoise(2022, d, 3, 0.59, "philox", 16).tile(m, tr, tc).astype(np.float64),
                      an.ARNoise(2021, d, 3, 0.59, "philox", 16).tile(m, tr, tc).astype(np.float64))) for d, m, _, tr, tc in keys)
    check("KEYS22", same == 0 and maxc < 0.01 and zc < 0.01, keys=len(keys), identical=int(same), max_abs_corr_innovations=maxc,
          max_abs_corr_z_rho059_lead3=zc)


def t_law22():
    TL = ((7, 30), (20, 50))

    def Z(rho, d, m, h):
        nz = an.ARNoise(2022, d, h, rho, "philox", 16)
        return np.stack([nz.tile(m, tr, tc).astype(np.float64) for tr, tc in TL])
    out, ok, MS, DS_ = {}, True, (0, 1, 15), (0, 2)
    for rho in (0.5, 0.59, 0.9):
        z = {(d, m, h): Z(rho, d, m, h) for d in DS_ for m in MS for h in range(1, 8)}
        mean_dev = max(abs(float(v.mean())) for v in z.values())
        var_dev = max(abs(float(v.var()) - 1.0) for v in z.values())
        lag = {k: max(abs(corr(z[(d, m, h)], z[(d, m, h + k)]) - rho ** k) for d in DS_ for m in MS for h in range(1, 8 - k))
               for k in range(1, 7)}
        lag_val = {k: float(np.mean([corr(z[(d, m, h)], z[(d, m, h + k)]) for d in DS_ for m in MS for h in range(1, 8 - k)]))
                   for k in range(1, 7)}
        mem = max(abs(corr(z[(d, a, h)], z[(d, b, h)])) for d in DS_ for h in range(1, 8) for a, b in ((0, 1), (0, 15), (1, 15)))
        sta = max(abs(corr(z[(0, m, h)], z[(2, m, h)])) for m in MS for h in range(1, 8))
        nbr = max(max(abs(corr(v[:, :, :-1], v[:, :, 1:])), abs(corr(v[:, :-1, :], v[:, 1:, :]))) for v in z.values())
        cross = max(abs(corr(z[(d, 0, h)], z[(d, 15, h2)])) for d in DS_ for h in range(1, 8) for h2 in range(1, 8))
        good = mean_dev <= 0.01 and var_dev <= 0.01 and max(lag.values()) <= 0.01 and mem < 0.01 and sta < 0.01 and nbr < 0.01 \
            and cross < 0.01
        ok &= good
        out[f"rho_{rho}"] = {"max_abs_mean": mean_dev, "max_abs_var_minus_1": var_dev, "lag_corr_mean": lag_val,
                             "lag_corr_max_abs_dev_from_rho_k": lag, "max_abs_corr_members": mem, "max_abs_corr_starts": sta,
                             "max_abs_corr_neighbour_pixels": nbr, "max_abs_corr_members_any_leads": cross, "pass": good}
    z0 = {h: Z(0.0, 2, 0, h) for h in range(1, 8)}
    lag0 = max(abs(corr(z0[h], z0[h2])) for h in range(1, 8) for h2 in range(h + 1, 8))
    out["rho_0"] = {"max_abs_lag_corr": lag0}
    ok &= lag0 < 0.01
    check("LAW22", ok, year=2022, starts=DS_, members=MS, tiles=TL, values_per_field=2 * an.B, **out)


def t_lead1():
    bad = n = 0
    for d in (0, 2, 356):
        for m in (0, 3, 15):
            for tr, tc in TILES[:3]:
                z = [u16(an.ARNoise(2022, d, 1, r, "philox", 16).tile(m, tr, tc)) for r in (0.5, 0.59, 0.9)]
                e = u16(an.innovation(2022, d, m, 1, tr, tc).astype(np.float16))
                n += 1
                bad += not (np.array_equal(z[0], z[1]) and np.array_equal(z[0], z[2]) and np.array_equal(z[0], e))
    check("LEAD1", bad == 0, tiles=n, mismatches=bad)


def t_seams():
    nz = an.ARNoise(2022, 2, 3, 0.59, "philox", 16)
    full = egg.full_map(nz, 15)
    bad, cov = 0, np.zeros((GH, GW), np.uint8)
    for k0, j0 in WIN:
        r0, c0 = 25 * k0, 25 * j0
        cut = egg.tile_window(nz, 15, r0, c0, PH, PW)
        r1, c1 = min(r0 + PH, GH), min(c0 + PW, GW)
        bad += not np.array_equal(u16(cut[:r1 - r0, :c1 - c0]), u16(full[r0:r1, c0:c1]))
        cov[r0:r1, c0:c1] += 1
    other = egg.tile_window(nz, 14, 25 * WIN[0][0], 25 * WIN[0][1], PH, PW)
    differs = not np.array_equal(u16(other[:100, :100]), u16(full[25 * WIN[0][0]:25 * WIN[0][0] + 100, 25 * WIN[0][1]:25 * WIN[0][1] + 100]))
    check("SEAMS", bad == 0 and differs, windows=len(WIN), mismatching_windows=bad, pixels_in_two_or_more_windows=int((cov >= 2).sum()),
          other_member_differs=differs)


def t_nested():
    bad = n = 0
    for d, h in ((2, 3), (0, 7), (19, 1)):
        a4, a16 = an.ARNoise(2022, d, h, 0.59, "philox", 4), an.ARNoise(2022, d, h, 0.59, "philox", 16)
        for m in range(4):
            for tr, tc in TILES[:4]:
                n += 1
                bad += not np.array_equal(u16(a4.tile(m, tr, tc)), u16(a16.tile(m, tr, tc)))
    k0, j0 = WIN[10]
    w4 = [u16(egg.tile_window(an.ARNoise(2022, 2, 3, 0.59, "philox", 4), m, 25 * k0, 25 * j0, PH, PW)) for m in range(4)]
    w16 = [u16(egg.tile_window(an.ARNoise(2022, 2, 3, 0.59, "philox", 16), m, 25 * k0, 25 * j0, PH, PW)) for m in range(4)]
    win_ok = all(np.array_equal(x, y) for x, y in zip(w4, w16))
    s4, s16 = an.ARNoise(2022, 2, 3, 1.0, "stored", 4), an.ARNoise(2022, 2, 3, 1.0, "stored", 16)
    st_ok = all(np.array_equal(u16(s4.tile(m, tr, tc)), u16(s16.tile(m, tr, tc))) for m in range(4) for tr, tc in TILES)
    check("NESTED", bad == 0 and win_ok and st_ok, philox_tiles=n, mismatches=bad, window_equal=win_ok, stored_members0_3_equal=st_ok)


def t_gold22():
    here = an.golden22_hashes()
    subs = [subrun(SUB % (str(an.A16), "golden22_hashes"), "GOLDEN", an.A16) for _ in range(2)]
    R["golden22_here"] = here
    ex = GOLD22.exists()
    gold = json.loads(GOLD22.read_text()) if ex else None
    check("GOLD22", all(s == here for s in subs) and here["numpy"] == an.NUMPY_REQUIRED and (gold == here if ex else True),
          subprocess_equal=[s == here for s in subs], golden22_json_exists=ex, golden22_json_equal=(gold == here) if ex else None,
          first16={k: v[:16] for k, v in here.items()})


class Cuts(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        nz, m, r0, c0 = self.items[n]
        return torch.from_numpy(np.ascontiguousarray(egg.tile_window(nz, m, r0, c0, PH, PW)))


def t_loader():
    items = []
    for (d, h, r, base, M, ms), w in zip(((2, 3, 0.59, "philox", 16, (0, 15)), (0, 7, 0.59, "philox", 4, (0, 3)),
                                          (19, 2, 1.0, "stored", 4, (0, 3)), (189, 4, 1.0, "stored", 16, (5, 15))), (0, 10, 20, 5)):
        nz = an.ARNoise(2022, d, h, r, base, M)
        if base == "stored":
            nz.tile(0, 0, 0)                                           # descriptor open before the workers start
        for m in ms:
            items.append((nz, m, 25 * WIN[w][0], 25 * WIN[w][1]))
    ds = Cuts(items)
    ref = [u16(x.numpy()) for x in DataLoader(ds, batch_size=None, num_workers=0)]
    fork = [u16(x.numpy()) for x in DataLoader(ds, batch_size=None, num_workers=4)]
    spawn = [u16(x.numpy()) for x in DataLoader(ds, batch_size=None, num_workers=4, multiprocessing_context="spawn")]
    pk = [pickle.loads(pickle.dumps(it[0])) for it in items]
    pk_ok = all(p._npy is None for p in pk) and all(
        np.array_equal(u16(egg.tile_window(p, it[1], it[2], it[3], PH, PW)), r) for p, it, r in zip(pk, items, ref))
    ok_f = len(fork) == len(ref) and all(np.array_equal(a, b) for a, b in zip(ref, fork))
    ok_s = len(spawn) == len(ref) and all(np.array_equal(a, b) for a, b in zip(ref, spawn))
    check("LOADER", ok_f and ok_s and pk_ok, windows=len(ref), workers0_vs_4_fork=ok_f, workers0_vs_4_spawn=ok_s, pickle_round_trip=pk_ok)


# ---------------------------------------------------------------- plans and job scripts
def n_eval(p):
    """the forecast list of eval_global_green.run for the arguments of plan p, from the real dataset"""
    ds = dataset(p["year"])
    starts = sorted({i for i, h in ds.samples})
    starts = [i for i in starts if int(ds.doy[i]) in set(p["doys"])] if p["doys"] else starts[::p["every"]]
    have, leads = set(ds.samples), p["leads"] or list(range(1, 8))
    todo = [(i, h) for i in starts for h in leads if (i, h) in have]
    return [(int(ds.doy[i]), h) for i, h in todo], (sorted(set(p["doys"])) == sorted({int(ds.doy[i]) for i in starts}) if p["doys"] else True)


def stored_keys(folder):
    K = []
    for f in sorted((an.RUN / folder).glob("part_*.npz")):
        with np.load(f) as z:
            K += [(int(d), int(h)) for d, h, _ in z["keys"].tolist()]
    return K


def job_args(path):
    """the ARGS* lines of a job script with its variables substituted (A16, RUN, CK, FINAL; RHO = 0.59; G11 for task k)"""
    sub = {"$A16": str(an.A16), "$RUN": str(an.RUN), "$CK": str(an.CKPT), "$FINAL": str(an.FINAL), "$RHO": "0.59"}
    out = {}
    for m in re.finditer(r'^(ARGS\w*)="(.*)"$', Path(path).read_text(), re.M):
        s = m.group(2)
        for k in sorted(sub, key=len, reverse=True):
            s = s.replace(k, sub[k])
        out[m.group(1)] = s
    return out


SUB16 = ("import sys, json; sys.path.insert(0, %r); import sample16_ar as s; res = []\n"
         "for argv in json.loads(%r):\n"
         "    a = s.parse_args(argv); res.append([a.plan, a.tag, len(a.pairs)])\n"
         "print('PARSED ' + json.dumps(res), flush=True)")


def t_plans():
    info, ok = {}, True
    for name, p in an.EVAL_PLANS.items():
        lst, doys_ok = n_eval(p)
        good = len(lst) == p["n"] and len(set(lst)) == len(lst) and doys_ok
        if name in ("G1", "G2"):
            good &= sorted(lst) == sorted(stored_keys(p["stored"]))
        ok &= good
        info[name] = dict(n=len(lst), plan_n=p["n"], doys_all_found=doys_ok, pass_=good)
    lines = an.pairs_g11()
    g11 = [tuple(int(v) for v in it.split(":")) for ln in lines for it in ln]
    f2 = stored_keys("eval_pop_crps_e01")
    g2, _ = n_eval(an.EVAL_PLANS["G2"])
    ok_g11 = len(g11) == 358 == len(set(g11)) and sorted(g11) == sorted(f2) == sorted(g2) and len(lines) == 45
    ok &= ok_g11
    info["G11"] = dict(n=len(g11), lines=len(lines), equals_F2_keys=sorted(g11) == sorted(f2), equals_G2=sorted(g11) == sorted(g2))
    # job scripts: eval plans parse (plan_match) in this process; sample16_ar in a subprocess (its audit hook)
    jobs, argv16 = {}, []
    for f in sorted((an.A16 / "jobs").glob("*.sh")):
        for var, s in job_args(f).items():
            if "--plan G11" in s:
                for k, ln in enumerate(lines):
                    argv16.append(s.replace("$PAIRS", " ".join(ln)).replace("k$KK", f"k{k:02d}").split())
                jobs[f"{f.name}:{var}"] = "sample16_ar (45 tasks)"
            elif "--plan S_b16" in s:
                argv16.append(s.split())
                jobs[f"{f.name}:{var}"] = "sample16_ar"
            else:
                a = eval_ar22.parse_args(s.split())
                jobs[f"{f.name}:{var}"] = a.plan
    parsed = subrun(SUB16 % (str(an.A16), json.dumps(argv16)), "PARSED", an.A16)
    ok16 = isinstance(parsed, list) and sorted(p[0] for p in parsed) == sorted(["S_b16"] + ["G11"] * 45)
    eval_plans_in_jobs = sorted(v for v in jobs.values() if not v.startswith("sample16"))
    ok_jobs = eval_plans_in_jobs == sorted(an.EVAL_PLANS) and ok16
    ok &= ok_jobs
    check("PLANS", ok, plans=info, job_scripts=jobs, sample16_parsed=len(parsed) if isinstance(parsed, list) else parsed,
          eval_plans_in_job_scripts=eval_plans_in_jobs, sample16_ok=ok16)


# ---------------------------------------------------------------- gates, claims, atomic writes
def refused(fn):
    try:
        fn()
    except (AssertionError, PermissionError, ValueError, SystemExit):
        return True
    return False


def t_gates():
    res = {}
    real, real_code = an.passed, an.same_code
    try:
        an.same_code = lambda p: True
        an.passed = lambda p: True
        res["G1_rho059_philox_accepted"] = not refused(lambda: an.gate("G1", 0.59, "philox"))
        res["G1_rho060_refused"] = refused(lambda: an.gate("G1", 0.60, "philox"))
        res["G1_stored_refused"] = refused(lambda: an.gate("G1", 0.59, "stored"))
        res["G1_rho1_stored_refused"] = refused(lambda: an.gate("G1", 1.0, "stored"))
        res["G11_rho07_refused"] = refused(lambda: an.gate("G11", 0.7, "philox"))
        res["S_a_rho1_stored_accepted"] = not refused(lambda: an.gate("S_a", 1.0, "stored"))
        res["S_a_philox_refused"] = refused(lambda: an.gate("S_a", 0.59, "philox"))
        an.passed = lambda p: Path(p).name != "smoke_check.json"
        res["G1_without_smoke_refused"] = refused(lambda: an.gate("G1", 0.59, "philox"))
        res["G11_without_smoke_refused"] = refused(lambda: an.gate("G11", 0.59, "philox"))
        res["S_c_without_smoke_accepted"] = not refused(lambda: an.gate("S_c", 0.59, "philox"))
        an.passed = lambda p: False
        res["S_a_without_check_cpu_refused"] = refused(lambda: an.gate("S_a", 1.0, "stored"))
        an.passed = lambda p: True
        an.same_code = lambda p: Path(p).name != "check_cpu.json"
        res["S_a_other_code_than_check_cpu_refused"] = refused(lambda: an.gate("S_a", 1.0, "stored"))
        an.same_code = lambda p: Path(p).name != "smoke_check.json"
        res["G1_other_code_than_smoke_refused"] = refused(lambda: an.gate("G1", 0.59, "philox"))
    finally:
        an.passed, an.same_code = real, real_code
    g1 = job_args(an.A16 / "jobs" / "g1.sh")["ARGS"].split()
    res["plan_G1_args_accepted"] = not refused(lambda: eval_ar22.parse_args(g1))
    res["plan_G1_every3_refused"] = refused(lambda: eval_ar22.parse_args([x if x != "2" else "3" for x in g1]))
    res["plan_G1_other_out_refused"] = refused(lambda: eval_ar22.parse_args([x if x != "eval_final_crps_e01_ar" else "eval_final_crps_e01" for x in g1]))
    res["plan_G1_steps8_refused"] = refused(lambda: eval_ar22.parse_args(g1[:g1.index("--steps") + 1] + ["8"] + g1[g1.index("--steps") + 2:]))
    with tempfile.TemporaryDirectory(prefix="a16chk_") as td:
        td = Path(td)
        meta = {"plan": "test", "x": 1}
        f = td / "f"
        res["claim_new_folder"] = not refused(lambda: an.claim(f, meta, fresh=False)) and (f / "a16_run.json").exists()
        res["claim_same_manifest_resume"] = not refused(lambda: an.claim(f, meta, fresh=False))
        res["claim_other_manifest_refused"] = refused(lambda: an.claim(f, {"plan": "test", "x": 2}, fresh=False))
        x = np.arange(12, dtype=np.float16).reshape(3, 4)
        an.asave(f / "map5km_d0_h1.npy", x)
        an.asavez(f / "part_0.npz", a=x, b=np.array("s"))
        with np.load(f / "part_0.npz") as z:
            res["atomic_write_read_back"] = np.array_equal(np.load(f / "map5km_d0_h1.npy"), x) and np.array_equal(z["a"], x) \
                and not [n for n in os.listdir(f) if ".tmp" in n]
        res["write_outside_claimed_refused"] = refused(lambda: an.asave(td / "other" / "x.npy", x))
        g = td / "g"
        g.mkdir()
        (g / "notes.txt").write_text("x")
        res["claim_foreign_file_refused"] = refused(lambda: an.claim(g, meta, fresh=False))
        h = td / "h"
        h.mkdir()
        (h / "part_0.npz").write_bytes(b"x")
        res["claim_files_without_manifest_refused"] = refused(lambda: an.claim(h, meta, fresh=False))
        res["claim_fresh_nonempty_refused"] = refused(lambda: an.claim(f, meta, fresh=True))
        an.CLAIMED.difference_update({Path(an.resolved(p)) for p in (f, g, h)})
    for k in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"):
        res[f"{k}_output_is_new_folder"] = Path(an.EVAL_PLANS[k]["out"]).name.endswith("_ar") and not Path(an.EVAL_PLANS[k]["out"]).exists()
    for k in ("G9", "S_a", "S_b4", "S_c"):
        res[f"{k}_output_absent"] = not Path(an.EVAL_PLANS[k]["out"]).exists()
    for k in ("G11", "S_b16"):
        res[f"{k}_output_absent"] = not Path(an.SAMPLE_PLANS[k]["out"]).exists()
    check("GATES", all(res.values()), **res)


def t_cost():
    k0, j0 = WIN[10]
    out = {}
    for M, h in ((4, 7), (16, 3), (16, 7)):
        t = time.time()
        x = [egg.tile_window(an.ARNoise(2022, 2, h, 0.59, "philox", M), m, 25 * k0, 25 * j0, PH, PW) for m in range(M)]
        out[f"window_lead{h}_{M}members_s"] = round(time.time() - t, 1)
        out[f"finite_{M}_{h}"] = bool(all(np.isfinite(v.astype(np.float32)).all() for v in x))
    t = time.time()
    an.innovation(2022, 1, 0, 1, 0, 0)
    out["one_innovation_tile_s"] = round(time.time() - t, 4)
    R["cost"] = out
    print("cost", out, flush=True)


def main():
    t0 = time.time()
    for name, fn in (("DIFF", t_diff), ("GOLD15", t_gold15), ("A15_BYTES", t_a15_bytes), ("ARITH", t_arith), ("LEAD1", t_lead1),
                     ("KEYS22", t_keys22), ("LAW22", t_law22), ("NESTED", t_nested), ("GOLD22", t_gold22), ("SEAMS", t_seams),
                     ("STORED4", t_stored4), ("STORED16", t_stored16), ("LOADER", t_loader), ("GATES", t_gates),
                     ("WINDOWS", t_windows), ("PLANS", t_plans)):
        guarded(name, fn)
    try:
        t_cost()
    except Exception as e:
        R["cost"] = {"error": repr(e)}
    others = all(v["pass"] for v in R["tests"].values())
    if others and not GOLD22.exists():                                  # written once, by the first fully passing run
        an.save_json(GOLD22, R["golden22_here"])
        R["golden22_json"] = "written by this run"
        check("GOLD22_json", json.loads(GOLD22.read_text()) == R["golden22_here"], written=True)
    R["all_passed"] = all(v["pass"] for v in R["tests"].values()) and GOLD22.exists()
    R["code_sha256"] = an.run_code_sha()                                # gate(): the runs must use exactly this code
    R["seconds"] = round(time.time() - t0, 1)
    R["job"] = os.environ.get("SLURM_JOB_ID")
    an.save_json(RESF, R)
    print("ALL PASSED" if R["all_passed"] else "SOME CHECKS FAILED", f"{R['seconds']:.0f}s", flush=True)
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
