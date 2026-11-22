#!/usr/bin/env python3
"""A15 CPU checks before any GPU run (a15/A15_draft.md section 6.1), in the job container (numpy 1.26.3), CPU only:
  I1  identity: ARNoise(base 'stored', rho 1) through the UNCHANGED tile_window is uint16-equal to noise_tiles.npy, all 21
      windows x 4 members x 7 leads;
  I2  x0 of WindowsAR(stored, rho 1) equals x0 of eval_global_green.Windows on 20 random (start, lead, window) of the real
      2021 dataset (TestGreen built as in eval_global_green.run from the checkpoint's hyper-parameters, seed 1521);
  I4  the arithmetic path with coefficients (1, 0, ..., 0) returns the stored bytes;
  I5  no seams: full_map(ARNoise(philox, rho 0.7, lead 3), member 1) equals each of its 21 tile_window cuts on their overlap;
  I6  law at rho 0.5 and 0.9 (tiles (7, 30) and (20, 50), leads 1-7, members 0 and 1, starts 9 and 17): mean within 0 +- 0.01,
      variance within 1 +- 0.01, lag-k correlation within rho^k +- 0.01 for every pair of leads, |correlation| < 0.01
      between members, between starts and between neighbouring pixels; rho 0: |lag correlation| < 0.01;
  I7  determinism: golden SHA-256 identical in this process and in two subprocesses, and equal to golden.json (written on
      the first run where every other check passed); identical windows through a DataLoader with 0 and 4 workers (fork and
      spawn) and after a pickle round trip;
  I8  z(m, d, 1) is uint16-equal for rho 0.50, 0.70 and 0.90 (one Cal lead-1 run serves every rho);
  DIFF every line of eval_ar.run_ar that differs from inspect.getsource(eval_global_green.run) ends with '# A15-H<1..8>' and
      the original lines that do not survive are exactly eval_ar.REMOVED_LINES.
    python check_cpu.py -> results/check_cpu.json (exit 1 if a check fails)"""
import arnoise as ar  # noqa: I001  (first)
import collections
import difflib
import inspect
import json
import pickle
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import eval_ar

egg = ar.egg
PH, PW, GH, GW = egg.PH, egg.PW, egg.GH, egg.GW
RES = ar.A15 / "results" / "check_cpu.json"
GOLD = ar.A15 / "golden.json"
R = {"what": __doc__, "numpy": np.__version__, "torch": torch.__version__, "python": sys.version.split()[0], "tests": {}}


def check(name, cond, **info):
    R["tests"][name] = {"pass": bool(cond), **info}
    print(("PASS " if cond else "FAIL ") + name + " " + json.dumps(ar.jsonable(info))[:600], flush=True)


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


WIN = egg.windows(np.load(egg.G / "meta.npz")["land"])
assert len(WIN) == 21, len(WIN)


# ---------------------------------------------------------------- DIFF
def t_diff():
    src = inspect.getsource(egg.run).splitlines()
    new = inspect.getsource(eval_ar.run_ar).splitlines()
    mk = re.compile(r"  # A15-H([1-8])$")
    removed, unmarked, hunks = [], [], collections.Counter()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        removed += [l.strip() for l in src[i1:i2]]
        for l in new[j1:j2]:
            m = mk.search(l.rstrip())
            if m:
                hunks[int(m.group(1))] += 1
            else:
                unmarked.append(l)
    marked_total = sum(1 for l in new if mk.search(l.rstrip()))
    same_removed = collections.Counter(removed) == collections.Counter(eval_ar.REMOVED_LINES)
    check("DIFF", not unmarked and same_removed and marked_total == sum(hunks.values()) and set(hunks) == set(range(1, 9)),
          original_lines=len(src), new_lines=len(new), removed=len(removed), declared_removed=len(eval_ar.REMOVED_LINES),
          unmarked_changed_lines=unmarked, marked_lines_by_hunk={f"H{k}": v for k, v in sorted(hunks.items())},
          removed_equal_declared=same_removed)


# ---------------------------------------------------------------- golden hashes (I7, part 1)
SUB = ("import sys; sys.path.insert(0, %r); import arnoise, json; "
       "print('GOLDEN ' + json.dumps(arnoise.golden_hashes(), sort_keys=True), flush=True)") % str(ar.A15)


def t_golden():
    here = ar.golden_hashes()
    subs = []
    for _ in range(2):
        p = subprocess.run([sys.executable, "-c", SUB], capture_output=True, text=True, cwd=str(ar.A15), timeout=900)
        line = [l for l in p.stdout.splitlines() if l.startswith("GOLDEN ")]
        subs.append(json.loads(line[-1][7:]) if p.returncode == 0 and line else {"error": p.stderr[-2000:]})
    same = all(s == here for s in subs)
    R["golden_here"] = here
    gold_exists = GOLD.exists()
    gold = json.loads(GOLD.read_text()) if gold_exists else None
    check("I7_golden_processes", same and here["numpy"] == ar.NUMPY_REQUIRED, numpy=here["numpy"],
          subprocess_equal=[s == here for s in subs], golden_json_exists=gold_exists,
          golden_json_equal=(gold == here) if gold_exists else None, first16={k: v[:16] for k, v in here.items()})
    if gold_exists:
        check("I7_golden_json", gold == here, mismatched=[k for k in here if gold.get(k) != here[k]])


# ---------------------------------------------------------------- I4: arithmetic path with (1, 0, ..., 0)
TILES = ((0, 0), (7, 30), (14, 35), (20, 50), (28, 70), (3, 66))


def t_i4():
    npy = egg.Npy(egg.G / "noise_tiles.npy")
    bad = n = 0
    coef_ok = all(np.array_equal(ar.coeffs(h, 1.0), np.r_[1.0, np.zeros(h - 1)].astype(np.float32)) for h in range(1, 8))
    for h in range(1, 8):
        nz = ar.ARNoise(2021, 9, h, 1.0, "stored")
        for m in range(4):
            for tr, tc in TILES:
                ref = npy._read(((m * ar.NTR + tr) * ar.NTC + tc) * ar.B, ar.B).reshape(ar.TILE, ar.TILE)
                n += 1
                bad += not np.array_equal(u16(nz._arith(m, tr, tc)), u16(ref))
    check("I4_arith_identity", coef_ok and bad == 0 and n == 168, compared_tiles=n, mismatches=bad, coefficients_exact=coef_ok)


# ---------------------------------------------------------------- I8: lead 1 independent of rho
def t_i8():
    bad = n = 0
    for d in (1, 25, 9):
        for m in range(4):
            for tr, tc in TILES[:3]:
                z = [u16(ar.ARNoise(2021, d, 1, r, "philox").tile(m, tr, tc)) for r in (0.50, 0.70, 0.90)]
                n += 1
                bad += not (np.array_equal(z[0], z[1]) and np.array_equal(z[0], z[2]))
    k0, j0 = WIN[10]
    w = [u16(egg.tile_window(ar.ARNoise(2021, 25, 1, r, "philox"), 0, 25 * k0, 25 * j0, PH, PW)) for r in (0.50, 0.70, 0.90)]
    win_ok = np.array_equal(w[0], w[1]) and np.array_equal(w[0], w[2])
    eq_eps = np.array_equal(u16(ar.ARNoise(2021, 25, 1, 0.5, "philox").tile(2, 7, 30)),
                            u16(ar.innovation(2021, 25, 2, 1, 7, 30).astype(np.float16)))
    check("I8_lead1_rho_free", bad == 0 and win_ok and eq_eps, tiles=n, mismatches=bad, window_equal=win_ok,
          lead1_equals_eps1=eq_eps)


# ---------------------------------------------------------------- I6: law of the generated fields
def t_i6():
    TL = ((7, 30), (20, 50))

    def Z(rho, d, m, h):
        nz = ar.ARNoise(2021, d, h, rho, "philox")
        return np.stack([nz.tile(m, tr, tc).astype(np.float64) for tr, tc in TL])       # (2, 512, 512)

    def corr(a, b):
        a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    out, ok = {}, True
    for rho in (0.5, 0.9):
        z = {(d, m, h): Z(rho, d, m, h) for d in (9, 17) for m in (0, 1) for h in range(1, 8)}
        mean_dev = max(abs(float(v.mean())) for v in z.values())
        var_dev = max(abs(float(v.var()) - 1.0) for v in z.values())
        lag = {k: max(abs(corr(z[(d, m, h)], z[(d, m, h + k)]) - rho ** k) for d in (9, 17) for m in (0, 1)
                      for h in range(1, 8 - k)) for k in range(1, 7)}
        lag_val = {k: float(np.mean([corr(z[(d, m, h)], z[(d, m, h + k)]) for d in (9, 17) for m in (0, 1)
                                     for h in range(1, 8 - k)])) for k in range(1, 7)}
        mem = max(abs(corr(z[(d, 0, h)], z[(d, 1, h)])) for d in (9, 17) for h in range(1, 8))
        sta = max(abs(corr(z[(9, m, h)], z[(17, m, h)])) for m in (0, 1) for h in range(1, 8))
        nbr = max(max(abs(corr(v[:, :, :-1], v[:, :, 1:])), abs(corr(v[:, :-1, :], v[:, 1:, :]))) for v in z.values())
        cross = max(abs(corr(z[(d, 0, h)], z[(d, 1, h2)])) for d in (9, 17) for h in range(1, 8) for h2 in range(1, 8))
        good = mean_dev <= 0.01 and var_dev <= 0.01 and max(lag.values()) <= 0.01 and mem < 0.01 and sta < 0.01 and nbr < 0.01 \
            and cross < 0.01
        ok &= good
        out[f"rho_{rho}"] = {"max_abs_mean": mean_dev, "max_abs_var_minus_1": var_dev, "lag_corr_mean": lag_val,
                             "lag_corr_max_abs_dev_from_rho_k": lag, "max_abs_corr_members": mem, "max_abs_corr_starts": sta,
                             "max_abs_corr_neighbour_pixels": nbr, "max_abs_corr_members_any_leads": cross, "pass": good,
                             "values_per_field": 2 * ar.B}
    z0 = {h: Z(0.0, 9, 0, h) for h in range(1, 8)}
    lag0 = max(abs(corr(z0[h], z0[h2])) for h in range(1, 8) for h2 in range(h + 1, 8))
    out["rho_0"] = {"max_abs_lag_corr": lag0, "max_abs_var_minus_1": max(abs(float(v.var()) - 1) for v in z0.values())}
    ok &= lag0 < 0.01
    check("I6_law", ok, **out)


# ---------------------------------------------------------------- I5: no seams (one global field per member, start, lead)
def t_i5():
    nz = ar.ARNoise(2021, 9, 3, 0.7, "philox")
    t = time.time()
    full = ar.egg.full_map(nz, 1)
    t_full = time.time() - t
    bad, cov = 0, np.zeros((GH, GW), np.uint8)
    for k0, j0 in WIN:
        r0, c0 = 25 * k0, 25 * j0
        cut = egg.tile_window(nz, 1, r0, c0, PH, PW)
        r1, c1 = min(r0 + PH, GH), min(c0 + PW, GW)
        bad += not np.array_equal(u16(cut[:r1 - r0, :c1 - c0]), u16(full[r0:r1, c0:c1]))
        cov[r0:r1, c0:c1] += 1
    other = egg.tile_window(ar.ARNoise(2021, 9, 3, 0.7, "philox"), 2, 25 * WIN[0][0], 25 * WIN[0][1], PH, PW)
    differs = not np.array_equal(u16(other[:100, :100]), u16(full[25 * WIN[0][0]:25 * WIN[0][0] + 100, 25 * WIN[0][1]:25 * WIN[0][1] + 100]))
    check("I5_no_seams", bad == 0 and differs, windows=len(WIN), mismatching_windows=bad,
          pixels_in_two_or_more_windows=int((cov >= 2).sum()), max_windows_per_pixel=int(cov.max()),
          full_map_seconds=round(t_full, 1), other_member_differs=differs)


# ---------------------------------------------------------------- I1: identity with noise_tiles.npy
def t_i1():
    npy = egg.Npy(egg.G / "noise_tiles.npy")
    bad = n = 0
    for w, (k0, j0) in enumerate(WIN):
        d = 1 + 8 * (w % 45)
        for m in range(4):
            ref = u16(egg.tile_window(npy, m, 25 * k0, 25 * j0, PH, PW))
            for h in range(1, 8):
                got = u16(egg.tile_window(ar.ARNoise(2021, d, h, 1.0, "stored"), m, 25 * k0, 25 * j0, PH, PW))
                n += 1
                bad += not np.array_equal(ref, got)
    check("I1_identity", bad == 0 and n == 21 * 4 * 7, windows=len(WIN), members=4, leads=7, compared=n, mismatches=bad)


# ---------------------------------------------------------------- I7 part 2: DataLoader workers, pickle
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
    for (d, h, r, base), w in zip(((9, 3, 0.7, "philox"), (17, 2, 0.5, "stored"), (353, 2, 0.9, "philox"),
                                   (1, 1, 1.0, "stored")), (0, 10, 20, 5)):
        nz = ar.ARNoise(2021, d, h, r, base)
        if base == "stored":
            nz.tile(0, 0, 0)                                           # descriptor open before the workers start
        for m in (0, 3):
            items.append((nz, m, 25 * WIN[w][0], 25 * WIN[w][1]))
    ds = Cuts(items)
    t = time.time()
    ref = [u16(x.numpy()) for x in DataLoader(ds, batch_size=None, num_workers=0)]
    t0 = time.time() - t
    fork = [u16(x.numpy()) for x in DataLoader(ds, batch_size=None, num_workers=4)]
    spawn = [u16(x.numpy()) for x in DataLoader(ds, batch_size=None, num_workers=4, multiprocessing_context="spawn")]
    pk = [pickle.loads(pickle.dumps(it[0])) for it in items]
    pk_ok = all(p._npy is None for p in pk) and all(
        np.array_equal(u16(egg.tile_window(p, it[1], it[2], it[3], PH, PW)), r) for p, it, r in zip(pk, items, ref))
    ok_f = all(np.array_equal(a, b) for a, b in zip(ref, fork)) and len(fork) == len(ref)
    ok_s = all(np.array_equal(a, b) for a, b in zip(ref, spawn)) and len(spawn) == len(ref)
    check("I7_dataloader", ok_f and ok_s and pk_ok, windows=len(ref), workers0_vs_4_fork=ok_f, workers0_vs_4_spawn=ok_s,
          pickle_round_trip=pk_ok, seconds_8_windows_0_workers=round(t0, 1))


def t_cost():
    """reported: CPU seconds of the noise of one window at lead 7 for 4 members (DataLoader worker cost)"""
    t = time.time()
    e = ar.innovation(2021, 1, 0, 1, 0, 0)
    t1 = time.time() - t
    k0, j0 = WIN[10]
    t = time.time()
    x = [egg.tile_window(ar.ARNoise(2021, 1, 7, 0.7, "philox"), m, 25 * k0, 25 * j0, PH, PW) for m in range(4)]
    t7 = time.time() - t
    R["cost"] = {"one_innovation_tile_s": round(t1, 4), "window_lead7_4members_s": round(t7, 1),
                 "window_shape": list(x[0].shape), "finite": bool(all(np.isfinite(v.astype(np.float32)).all() for v in x)),
                 "eps_dtype": str(e.dtype)}
    print("cost", R["cost"], flush=True)


# ---------------------------------------------------------------- I2: WindowsAR(stored, 1) against Windows, real data
def t_i2():
    ck = ar.RUN / "ckpts" / "epoch=01.ckpt"
    state = torch.load(ck, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    del state
    store, emis = hp.get("store", "green"), bool(hp.get("emis", False))
    P = int(egg.Npy(egg.G / store / "hres" / "origins_2021.npy").shape[-2])
    ds = egg.TestGreen(7, P, store, emis, 2021, hp.get("emis_store"))
    samples = sorted(ds.samples)
    rng = np.random.default_rng(1521)
    pick = rng.choice(len(samples), 20, replace=False)
    wins = rng.integers(0, len(WIN), 20)
    jobs = [(samples[j][0], samples[j][1], int(w)) for j, w in zip(pick, wins)]
    A = egg.Windows(ds, jobs, WIN, 4)
    Bw = ar.WindowsAR(ds, jobs, WIN, 4, [1.0] * len(jobs), "stored")
    bad_x0, bad_other, keys_same = 0, 0, True
    for n in range(len(jobs)):
        a, b = A[n], Bw[n]
        keys_same &= set(a) == set(b)
        bad_x0 += not np.array_equal(u16(a["x0"].numpy()), u16(b["x0"].numpy()))
        bad_other += sum(not (a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                              and a[k].numpy().tobytes() == b[k].numpy().tobytes()) for k in a if k != "x0")
    check("I2_windows", bad_x0 == 0 and keys_same, items=len(jobs), x0_mismatches=bad_x0, same_keys=keys_same,
          other_inputs_mismatching=bad_other, store=store, particles=P, n_samples_2021=len(samples),
          jobs=[(int(ds.doy[i]), h, w) for i, h, w in jobs])


def main():
    t0 = time.time()
    for name, fn in (("DIFF", t_diff), ("I7_golden_processes", t_golden), ("I4_arith_identity", t_i4),
                     ("I8_lead1_rho_free", t_i8), ("I6_law", t_i6), ("I5_no_seams", t_i5), ("I1_identity", t_i1),
                     ("I7_dataloader", t_loader), ("I2_windows", t_i2)):
        guarded(name, fn)
    try:
        t_cost()
    except Exception as e:
        R["cost"] = {"error": repr(e)}
    others = all(v["pass"] for v in R["tests"].values())
    if others and not GOLD.exists():                                    # written once, by the first fully passing run
        ar.save_json(GOLD, R["golden_here"])
        R["golden_json"] = "written by this run"
        check("I7_golden_json", json.loads(GOLD.read_text()) == R["golden_here"], written=True)
    R["all_passed"] = all(v["pass"] for v in R["tests"].values()) and GOLD.exists()
    R["seconds"] = round(time.time() - t0, 1)
    ar.save_json(RES, R)
    print("ALL PASSED" if R["all_passed"] else "SOME CHECKS FAILED", f"{R['seconds']:.0f}s", flush=True)
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
