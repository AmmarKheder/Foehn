#!/usr/bin/env python3
"""A19 CPU checks before any GPU run (a19/A19_draft.md section 6.1), in the job container (numpy 1.26.3), CPU only:
  DIFF        every line of eval19.run19 that differs from a16/eval_ar22.run_ar22 ends with '# A19-H<n>' (H1-H4) and the lines
              removed are exactly eval19.REMOVED_LINES; WindowsAR19.__getitem__ against arnoise22.WindowsAR22.__getitem__ (W1,
              noise19.REMOVED_LINES_WINDOWS); _audit verbatim from a16/sample16_ar.py (the populated-pixel sampler); both hooks
              installed before any import of arnoise22; eval19 imports noise19 first; the write guard and its helper are
              verbatim in score19.py; eval19.start against eval_ar22.start (reported)
  GOLD        A16's generator gives a15/golden.json (2021 keys) and a16/golden22.json (2022 keys); golden19() (negated members
              and their partners, 2022, rho*) identical in this process and in two fresh processes -> golden19 field
  A16_BYTES   members 0-3 of ARNoise19 (2022, rho*) are byte-equal to arnoise22.ARNoise(2022, ..., 4 members) (the noise of
              G2) and to ARNoise(..., 16 members) (G11), tiles and one window cut; the golden22 field of member 0 reproduced
  NEG_TILES   member 4+k is the exact negative of member k (uint16 sign bit; 144 tiles: starts 2, 190, 356 x leads 1, 2, 3, 7 x
              k = 0..3 x 3 tiles) and equals in value the field built from the negated innovations (signed zeros only differ)
  NEG_WINDOWS through the UNCHANGED tile_window: the 21 windows at (2, 3) and 3 windows at (356, 1); full_map(member 5) =
              sign flip of full_map(member 1), equal to each of its 21 window cuts
  LAW         negated fields of 2022 (members 4, 5 against 0, 1, starts 2 and 4, leads 1-7, two tiles): mean 0, variance 1,
              correlation -1 with the partner, lag correlations rho^j and -rho^|h - h'| with the partner, |correlation| < 0.01
              with every other member and start (tolerance 0.01)
  LOADER      the real 2022 dataset: x0 of WindowsAR19(ids 0..3) = x0 of arnoise22.WindowsAR22(2022, rho*, philox, 4 members)
              (the loader of G2); WindowsAR19(ids 4..7) = sign flip; ids 0..7 = both; all other inputs byte-equal; the same
              through a DataLoader with 0 and 4 workers (fork, spawn) and pickle
  PLANS       the forecast list of plan main, built by run19's own code on the 2022 dataset, is G2's: 358 (start day, lead,
              target) and the split todo[rank::64] gives part_<r> the keys of G2's part_<r> in the same order; the arguments of
              main equal G2's manifest (RUN/eval_pop_crps_e01_ar/a16_run.json) except the output folder and the member ids; the
              smoke forecasts are G2 forecasts; the job scripts call exactly the plans, parse and match, and write under a19
  GATES       noise19.gate_state refuses another rho or base, a missing / failed check19 or smoke, changed code, a smoke whose
              projection breaks the limit or the cap; claim_state refuses a foreign file, files without a manifest, another
              manifest, a non-empty smoke folder; the hooks refuse a station path and writes into protected folders, and let
              reads of protected folders and writes under a19 through (fake paths: nothing is opened or created)
  cost        CPU seconds of one window of 4 negated members at lead 3 (DataLoader worker cost), reported
    python check19.py -> code/out/check19.json (exit 1 if a check fails)"""
import os  # FOEHN_WORK, FOEHN_ENVS: roots of the working trees and of the environments
import noise19 as nz  # noqa: I001  (first)
import ast
import collections
import difflib
import json
import pickle
import re
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import eval19

an, egg = nz.an, nz.egg
PH, PW, GH, GW = egg.PH, egg.PW, egg.GH, egg.GW
R = {"what": __doc__, "numpy": np.__version__, "torch": torch.__version__, "python": sys.version.split()[0], "tests": {}}
WIN = egg.windows(np.load(egg.G / "meta.npz")["land"])
assert len(WIN) == 21, len(WIN)
RHO = nz.rho_star()
SIGN = np.uint16(0x8000)
YEAR = nz.YEAR


def check(name, cond, **info):
    R["tests"][name] = {"pass": bool(cond), **info}
    print(("PASS " if cond else "FAIL ") + name + " " + json.dumps(nz.jsonable(info))[:900], flush=True)


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


def flipped(neg, pos):
    return np.array_equal(u16(neg), u16(pos) ^ SIGN)


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# ---------------------------------------------------------------- DIFF
def fsrc(path, name, cls=None):
    txt = Path(path).read_text()
    body, lines = ast.parse(txt).body, txt.splitlines()
    if cls:
        body = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls][0].body
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            return lines[n.lineno - 1:n.end_lineno]
    raise KeyError(name)


def diff(src, new, letter, declared, hunks_declared):
    mk = re.compile(r"  # A19-%s(\d+)$" % letter)
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
    same = collections.Counter(removed) == collections.Counter(declared)
    ok = not unmarked and same and marked_total == sum(hunks.values()) and set(hunks) == set(hunks_declared)
    return ok, dict(original_lines=len(src), new_lines=len(new), removed=len(removed), declared_removed=len(declared),
                    unmarked_changed_lines=unmarked, marked_lines_by_hunk={f"{letter}{k}": v for k, v in sorted(hunks.items())},
                    removed_equal_declared=same, removed_not_declared=[l for l in removed if l not in declared],
                    declared_not_removed=[l for l in declared if l not in removed])


def guard_block(path):
    """the text between the write-guard markers (noise19.py, score19.py)"""
    t = Path(path).read_text()
    a, b = "# ---- A19 write guard (begin", "# ---- A19 write guard (end)"
    assert t.count(a) == 1 and t.count(b) == 1, path
    return t[t.index(a):t.index(b)]


def t_diff():
    E22, A16N, S16 = nz.A16_DIR / "eval_ar22.py", Path(an.__file__), nz.A16_DIR / "sample16_ar.py"
    SC = nz.CODE / "score19.py"
    W_ = os.environ["FOEHN_WORK"]                # score19.py names the two roots literally, noise19.py by FOEHN_WORK
    BASES_FORMS = ('_BASES = (os.environ["FOEHN_WORK"], os.path.realpath(os.environ["FOEHN_WORK"]))',
                   f'_BASES = ("{W_}", "{os.path.realpath(W_)}")')
    ok_e, i_e = diff(fsrc(E22, "run_ar22"), fsrc(eval19.__file__, "run19"), "H", eval19.REMOVED_LINES, eval19.HUNKS)
    ok_w, i_w = diff(fsrc(A16N, "__getitem__", "WindowsAR22"), fsrc(nz.__file__, "__getitem__", "WindowsAR19"), "W",
                     nz.REMOVED_LINES_WINDOWS, (1,))
    head = Path(nz.__file__).read_text().split("import arnoise22")[0]
    verb = {"_audit_sample16_ar": fsrc(S16, "_audit") == fsrc(nz.__file__, "_audit"),
            "hooks_before_imports": "sys.addaudithook(_audit)" in head and "sys.addaudithook(_audit19)" in head,
            "eval19_imports_noise19_first": Path(eval19.__file__).read_text().split('"""')[2].lstrip().startswith("import noise19"),
            "score19_write_guard_block_verbatim": guard_block(SC) == guard_block(nz.__file__).replace(*BASES_FORMS)}
    sd = list(difflib.unified_diff(fsrc(E22, "start"), fsrc(eval19.__file__, "start"), lineterm="", n=0))
    check("DIFF", ok_e and ok_w and all(verb.values()), files={"a16_eval_ar22": str(E22), "a16_arnoise22": str(A16N),
          "a16_sample16_ar": str(S16)}, run19=i_e, WindowsAR19=i_w, verbatim=verb, start_vs_eval_ar22_start_reported=sd)


# ---------------------------------------------------------------- GOLD
def subrun(code, prefix, cwd):
    p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=str(cwd), timeout=1800)
    line = [l for l in p.stdout.splitlines() if l.startswith(prefix + " ")]
    return json.loads(line[-1][len(prefix) + 1:]) if p.returncode == 0 and line else {"error": p.stderr[-3000:]}


SUB19 = ("import sys; sys.path.insert(0, %r); import noise19, json; "
         "print('G19 ' + json.dumps(noise19.golden19(), sort_keys=True), flush=True)") % str(nz.CODE)


def t_gold():
    g15 = an.golden_hashes() == json.loads((nz.A15_DIR / "golden.json").read_text())
    g22 = an.golden22_hashes() == json.loads((nz.A16_DIR / "golden22.json").read_text())
    here = nz.golden19()
    subs = [subrun(SUB19, "G19", nz.CODE) for _ in range(2)]
    R["golden19"] = here
    check("GOLD", g15 and g22 and all(s == here for s in subs) and here["numpy"] == "1.26.3", a15_golden_json=g15,
          a16_golden22_json=g22, golden19_subprocess_equal=[s == here for s in subs],
          first16={k: v[:16] for k, v in here.items() if isinstance(v, str)})


# ---------------------------------------------------------------- A16_BYTES
A16_ITEMS = [(0, 1, 0, 14, 35), (2, 3, 3, 0, 0), (356, 3, 1, 28, 70), (190, 1, 2, 7, 30), (64, 3, 0, 20, 50), (330, 1, 3, 3, 66)]


def t_a16_bytes():
    mine = [nz.sha(nz.ARNoise19(d, h, RHO).tile(m, tr, tc)) for d, h, m, tr, tc in A16_ITEMS]
    g2 = [nz.sha(an.ARNoise(YEAR, d, h, RHO, "philox", 4).tile(m, tr, tc)) for d, h, m, tr, tc in A16_ITEMS]
    g11 = [nz.sha(an.ARNoise(YEAR, d, h, RHO, "philox", 16).tile(m, tr, tc)) for d, h, m, tr, tc in A16_ITEMS]
    k0, j0 = WIN[10]
    mine.append(nz.sha(egg.tile_window(nz.ARNoise19(2, 3, RHO), 1, 25 * k0, 25 * j0, PH, PW)))
    g2.append(nz.sha(egg.tile_window(an.ARNoise(YEAR, 2, 3, RHO, "philox", 4), 1, 25 * k0, 25 * j0, PH, PW)))
    g11.append(nz.sha(egg.tile_window(an.ARNoise(YEAR, 2, 3, RHO, "philox", 16), 1, 25 * k0, 25 * j0, PH, PW)))
    gold = json.loads((nz.A16_DIR / "golden22.json").read_text())["z_2022_0_lead7_rho0.59_philox_m0_tile7_30/f16"]
    g0 = nz.sha(nz.ARNoise19(0, 7, RHO).tile(0, 7, 30)) == gold
    check("A16_BYTES", mine == g2 and mine == g11 and g0, items=[list(x) for x in A16_ITEMS] + [["window 10 cut", 2, 3, 1]],
          equal_G2_generator=[a == b for a, b in zip(mine, g2)], equal_G11_generator=[a == b for a, b in zip(mine, g11)],
          golden22_member0_reproduced=g0)


# ---------------------------------------------------------------- NEG_TILES
def neg_arith(d, h, k, tr, tc):
    """the field of member k built from the negated innovations -eps_j (same keys, same coefficients, same order)"""
    c = an.coeffs(h, RHO)
    acc = np.zeros((nz.TILE, nz.TILE), np.float32)
    for j in range(1, h + 1):
        if c[j - 1] == 0:
            continue
        acc += c[j - 1] * np.negative(an.innovation(YEAR, d, k, j, tr, tc))
    return acc.astype(np.float16)


def t_neg_tiles():
    n = bad_flip = bad_val = zero_sign = 0
    for d in (2, 190, 356):
        for h in (1, 2, 3, 7):
            z = nz.ARNoise19(d, h, RHO)
            for k in range(4):
                for tr, tc in ((7, 30), (14, 35), (28, 70)):
                    pos, neg = z.tile(k, tr, tc), z.tile(4 + k, tr, tc)
                    ar_ = neg_arith(d, h, k, tr, tc)
                    n += 1
                    bad_flip += not flipped(neg, pos)
                    bad_val += not np.array_equal(neg.astype(np.float32), ar_.astype(np.float32))
                    zero_sign += int((u16(neg) != u16(ar_)).sum())
    check("NEG_TILES", n == 144 and bad_flip == 0 and bad_val == 0, tiles=n, sign_flip_mismatches=bad_flip,
          value_mismatches_vs_negated_innovations=bad_val, bit_differences_on_signed_zeros=zero_sign)


# ---------------------------------------------------------------- NEG_WINDOWS
def t_neg_windows():
    bad, n = 0, 0
    z = nz.ARNoise19(2, 3, RHO)
    cuts5 = {}
    for w, (k0, j0) in enumerate(WIN):
        for k in range(4):
            pos = egg.tile_window(z, k, 25 * k0, 25 * j0, PH, PW)
            neg = egg.tile_window(z, 4 + k, 25 * k0, 25 * j0, PH, PW)
            n += 1
            bad += not flipped(neg, pos)
            if k == 1:
                cuts5[w] = neg
    z1 = nz.ARNoise19(356, 1, RHO)
    for w in (0, 10, 20):
        k0, j0 = WIN[w]
        for k in range(4):
            n += 1
            bad += not flipped(egg.tile_window(z1, 4 + k, 25 * k0, 25 * j0, PH, PW), egg.tile_window(z1, k, 25 * k0, 25 * j0, PH, PW))
    t = time.time()
    f5, f1 = egg.full_map(z, 5), egg.full_map(z, 1)
    full_ok = flipped(f5, f1)
    del f1
    seam_bad = 0
    for w, (k0, j0) in enumerate(WIN):
        r0, c0 = 25 * k0, 25 * j0
        r1, c1 = min(r0 + PH, GH), min(c0 + PW, GW)
        seam_bad += not np.array_equal(u16(cuts5[w][:r1 - r0, :c1 - c0]), u16(f5[r0:r1, c0:c1]))
    check("NEG_WINDOWS", bad == 0 and n == 21 * 4 + 12 and full_ok and seam_bad == 0, window_cuts=n, mismatches=bad,
          full_map_member5_is_flip_of_member1=full_ok, member5_window_cuts_vs_full_map_mismatches=seam_bad,
          full_map_seconds=round(time.time() - t, 1))


# ---------------------------------------------------------------- LAW
def t_law():
    TL = ((7, 30), (20, 50))

    def Z(d, m, h):
        z = nz.ARNoise19(d, h, RHO)
        return np.stack([z.tile(m, tr, tc).astype(np.float64) for tr, tc in TL])
    z = {(d, m, h): Z(d, m, h) for d in (2, 4) for m in (0, 1, 4, 5) for h in range(1, 8)}
    neg = [(d, m, h) for d in (2, 4) for m in (4, 5) for h in range(1, 8)]
    mean_dev = max(abs(float(z[k].mean())) for k in neg)
    var_dev = max(abs(float(z[k].var()) - 1.0) for k in neg)
    partner = min(-corr(z[(d, m, h)], z[(d, m - 4, h)]) for d, m, h in neg)
    lag = max(abs(corr(z[(d, m, h)], z[(d, m, h + j)]) - RHO ** j) for d in (2, 4) for m in (4, 5) for h in range(1, 8)
              for j in range(1, 8 - h))
    cross_partner = max(abs(corr(z[(d, m, h)], z[(d, m - 4, h2)]) + RHO ** abs(h - h2)) for d in (2, 4) for m in (4, 5)
                        for h in range(1, 8) for h2 in range(1, 8))
    others = max(abs(corr(z[(d, m, h)], z[(d, o, h2)])) for d in (2, 4) for m in (4, 5) for o in (0, 1, 4, 5)
                 if o not in (m, m - 4) for h in range(1, 8) for h2 in range(1, 8))
    starts = max(abs(corr(z[(2, m, h)], z[(4, o, h2)])) for m in (4, 5) for o in (0, 1, 4, 5) for h in range(1, 8)
                 for h2 in range(1, 8))
    ok = mean_dev <= 0.01 and var_dev <= 0.01 and partner >= 1 - 1e-6 and lag <= 0.01 and cross_partner <= 0.01 \
        and others < 0.01 and starts < 0.01
    check("LAW", ok, rho=RHO, max_abs_mean=mean_dev, max_abs_var_minus_1=var_dev, min_minus_corr_with_partner=partner,
          max_abs_lag_corr_dev_from_rho_j=lag, max_abs_dev_partner_other_lead_from_minus_rho_dh=cross_partner,
          max_abs_corr_other_members_any_leads=others, max_abs_corr_other_start=starts, values_per_field=2 * nz.B)


# ---------------------------------------------------------------- LOADER
DS = {}


def dataset():
    if "ds" not in DS:
        state = torch.load(nz.CKPT, map_location="cpu", weights_only=False)
        hp = state["hyper_parameters"]
        del state
        store, emis = hp.get("store", "green"), bool(hp.get("emis", False))
        P = int(egg.Npy(egg.G / store / "hres" / f"origins_{YEAR}.npy").shape[-2])
        DS["ds"] = egg.TestGreen(7, P, store, emis, YEAR, hp.get("emis_store"))
        DS["info"] = dict(store=store, particles=P)
    return DS["ds"]


class Cuts(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        z, m, r0, c0 = self.items[n]
        return torch.from_numpy(np.ascontiguousarray(egg.tile_window(z, m, r0, c0, PH, PW)))


def main_todo(ds):
    """the forecast list of plan main, built with run19's own lines (starts[::every], leads, (i, h) in ds.samples)"""
    starts = sorted({i for i, h in ds.samples})
    starts = starts[::nz.PLANS["main"]["every"]]
    have, leads = set(ds.samples), nz.PLANS["main"]["leads"]
    return [(i, h) for i in starts for h in leads if (i, h) in have]


def t_loader():
    ds = dataset()
    todo = main_todo(ds)
    rng = np.random.default_rng(1521)
    pick = [todo[j] for j in rng.choice(len(todo), 6, replace=False)]
    jobs = [(i, h, int(w)) for (i, h), w in zip(pick, rng.integers(0, len(WIN), 6))]
    A = an.WindowsAR22(ds, jobs, WIN, 4, [], YEAR, RHO, "philox")
    P0 = nz.WindowsAR19(ds, jobs, WIN, (0, 1, 2, 3), RHO)
    N4 = nz.WindowsAR19(ds, jobs, WIN, (4, 5, 6, 7), RHO)
    E8 = nz.WindowsAR19(ds, jobs[:2], WIN, tuple(range(8)), RHO)
    bad_a16 = bad_neg = bad_other = 0
    keys_same = True
    for n in range(len(jobs)):
        a, p, q = A[n], P0[n], N4[n]
        keys_same &= set(a) == set(p) == set(q)
        bad_a16 += not np.array_equal(u16(a["x0"].numpy()), u16(p["x0"].numpy()))
        bad_neg += not flipped(q["x0"].numpy(), p["x0"].numpy())
        bad_other += sum(not (a[k].dtype == v[k].dtype and a[k].shape == v[k].shape and a[k].numpy().tobytes() == v[k].numpy().tobytes())
                         for v in (p, q) for k in a if k != "x0")
    bad8 = 0
    for n in range(2):
        e = E8[n]["x0"].numpy()
        bad8 += not (np.array_equal(u16(e[:4]), u16(P0[n]["x0"].numpy())) and np.array_equal(u16(e[4:]), u16(N4[n]["x0"].numpy())))
    items = []
    for (d, h), w in (((2, 3), 0), ((356, 1), 10), ((64, 3), 20)):
        z = nz.ARNoise19(d, h, RHO)
        for m in (0, 5):
            items.append((z, m, 25 * WIN[w][0], 25 * WIN[w][1]))
    cd = Cuts(items)
    ref = [u16(x.numpy()) for x in DataLoader(cd, batch_size=None, num_workers=0)]
    fork = [u16(x.numpy()) for x in DataLoader(cd, batch_size=None, num_workers=4)]
    spawn = [u16(x.numpy()) for x in DataLoader(cd, batch_size=None, num_workers=4, multiprocessing_context="spawn")]
    pk = [pickle.loads(pickle.dumps(it[0])) for it in items]
    pk_ok = all(np.array_equal(u16(egg.tile_window(p, it[1], it[2], it[3], PH, PW)), r) for p, it, r in zip(pk, items, ref))
    ok_f = len(fork) == len(ref) and all(np.array_equal(a, b) for a, b in zip(ref, fork))
    ok_s = len(spawn) == len(ref) and all(np.array_equal(a, b) for a, b in zip(ref, spawn))
    check("LOADER", bad_a16 == 0 and bad_neg == 0 and bad_other == 0 and keys_same and bad8 == 0 and ok_f and ok_s and pk_ok,
          items=[(int(ds.doy[i]), h, w) for i, h, w in jobs], x0_ids0_3_vs_WindowsAR22_G2_mismatches=bad_a16,
          x0_ids4_7_not_sign_flip=bad_neg, other_inputs_mismatching=bad_other, same_keys=keys_same, eight_member_mismatches=bad8,
          loader_workers0_vs_4_fork=ok_f, loader_workers0_vs_4_spawn=ok_s, pickle_round_trip=pk_ok, **DS["info"])


# ---------------------------------------------------------------- PLANS
def g2_keys():
    """{rank: [(start day, lead, target), ...]} of G2's part files, and G2's manifest"""
    out = {}
    for p in sorted(nz.G2_DIR.glob("part_*.npz")):
        assert ".tmp" not in p.name, p
        with np.load(p) as z:
            out[int(p.stem.split("_")[1])] = [tuple(int(v) for v in k) for k in z["keys"].tolist()]
    return out, json.loads((nz.G2_DIR / "a16_run.json").read_text())


def job_args(sh, var):
    """the eval19.py arguments of a job script (variable `var`), with $RUN, $CK, $RHO, $D substituted"""
    t = (nz.CODE / sh).read_text()
    env = {"RUN": str(nz.RUN), "CK": str(nz.CKPT), "RHO": "0.59", "D": str(nz.DATA)}
    m = re.search(r'^%s="([^"]*)"' % var, t, re.M)
    assert m, (sh, var)
    s = m.group(1)
    for k, v in env.items():
        s = s.replace("$" + k, v)
    assert "$" not in s, s
    return shlex.split(s)


def t_plans():
    ds = dataset()
    todo = main_todo(ds)
    keys = [(int(ds.doy[i]), h, int(ds.doy[i + h])) for i, h in todo]
    g2, man = g2_keys()
    split = {r: keys[r::nz.PLANS["main"]["world"]] for r in range(nz.PLANS["main"]["world"])}
    same_split = sorted(g2) == list(range(64)) and all(split[r] == g2[r] for r in range(64))
    all358 = len(keys) == 358 == len(set(keys)) and sorted(keys) == sorted(k for v in g2.values() for k in v)
    sm = {"smoke_alone": [(2, 1, 3), (2, 3, 5)], "smoke_eight": [(2, 3, 5)]}
    smoke_in_g2 = all(k in set(keys) for v in sm.values() for k in v)
    a_main = vars(eval19.parse_args(job_args("main.sh", "ARGS")))
    a_al = vars(eval19.parse_args(job_args("smoke.sh", "ARGS_A")))
    a_ei = vars(eval19.parse_args(job_args("smoke.sh", "ARGS_E")))
    mine = nz.args_of(eval19.parse_args(job_args("main.sh", "ARGS")))
    g2a = dict(man["args"])
    diff_g2 = {k: (mine.get(k), g2a.get(k)) for k in set(mine) | set(g2a) if mine.get(k) != g2a.get(k)}
    args_g2 = set(diff_g2) == {"out", "mids"} and man["plan"] == "G2" and float(man["rho"]) == RHO and man["base"] == "philox"
    plans_ok = (a_main["plan"] == "main" and a_al["plan"] == "smoke_alone" and a_ei["plan"] == "smoke_eight"
                and a_main["mids"] == (4, 5, 6, 7) == a_al["mids"] and a_ei["mids"] == tuple(range(8)))
    # the smoke lists, by run19's own lines (doys, leads)
    st = sorted({i for i, h in ds.samples}); have = set(ds.samples)
    sm_built = {}
    for pl in nz.SMOKE:
        p = nz.PLANS[pl]
        ss = [i for i in st if int(ds.doy[i]) in set(p["doys"])]
        sm_built[pl] = [(int(ds.doy[i]), h, int(ds.doy[i + h])) for i in ss for h in p["leads"] if (i, h) in have]
    smoke_ok = sm_built == sm and all(len(sm[pl]) == nz.PLANS[pl]["n"] for pl in nz.SMOKE)
    sh = {n: (nz.CODE / n).read_text() for n in ("smoke.sh", "main.sh", "cpu.sh")}
    calls = {n: re.findall(r"eval19\.py \$(\w+)", "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("#"))) for n, t in sh.items()}
    outs = {n: re.findall(r"#SBATCH --output=(\S+)", t) for n, t in sh.items()}
    jobs_ok = (calls["smoke.sh"] == ["ARGS_A", "ARGS_E"] and calls["main.sh"] == ["ARGS"] and calls["cpu.sh"] == []
               and all(str((nz.CODE / o).resolve()).startswith(str(nz.DATA.resolve()) + "/") for n in ("smoke.sh", "main.sh") for o in outs[n])
               and all(str((nz.CODE / o).resolve()).startswith(str(nz.OUT.resolve()) + "/") for o in outs["cpu.sh"])
               and all(re.search(r"^#SBATCH --account=\S+", t, re.M) for t in sh.values())
               and "--partition=small-g" in sh["smoke.sh"] and "--partition=standard-g" in sh["main.sh"] and "--partition=small\n" in sh["cpu.sh"]
               and f"--time={nz.LIMIT_MAIN}" in sh["main.sh"] and f"--time={nz.LIMIT_SMOKE}" in sh["smoke.sh"]
               and "--nodes=8" in sh["main.sh"] and "--ntasks-per-node=8" in sh["main.sh"] and "--gpus-per-node=8" in sh["main.sh"])
    g2sh = (nz.A16_DIR / "jobs" / "g2.sh").read_text()
    same_layout = all(s in g2sh and s in sh["main.sh"] for s in ("--nodes=8", "--gpus-per-node=8", "--ntasks-per-node=8",
                                                                 "--cpus-per-task=7", "--mem=480G", "--exclusive"))
    check("PLANS", same_split and all358 and smoke_in_g2 and args_g2 and plans_ok and smoke_ok and jobs_ok and same_layout,
          forecasts=len(keys), same_rank_split_as_G2=same_split, all_358_equal_G2=all358, smoke_items_are_G2_forecasts=smoke_in_g2,
          main_args_vs_G2_manifest_differ_only_in=sorted(diff_g2), args_equal_G2_except_out_and_ids=args_g2, plans_parse=plans_ok,
          smoke_lists=sm_built, smoke_ok=smoke_ok, job_script_calls=calls, job_outputs=outs, job_scripts_ok=jobs_ok,
          layout_of_g2_sh=same_layout)


# ---------------------------------------------------------------- GATES
def t_gates():
    code = nz.run_code_sha()
    other = dict(code, **{"code/eval19.py": "0" * 64})
    good = {"all_passed": True, "code_sha256": code}
    cost_ok = {"tests": {"cost": {"main": {"within_limit": True, "within_cap": True}}}}
    sm_ok = dict(all_passed=True, code_sha256=code, **cost_ok)
    R_ = 0.59
    cases = {
        "no_check": (None, None, "smoke_alone", R_, "philox", False),
        "check_failed": ({"all_passed": False, "code_sha256": code}, None, "smoke_alone", R_, "philox", False),
        "check_other_code": ({"all_passed": True, "code_sha256": other}, None, "smoke_eight", R_, "philox", False),
        "smoke_ok_after_check": (good, None, "smoke_alone", R_, "philox", True),
        "other_rho_refused": (good, sm_ok, "main", 0.5, "philox", False),
        "rho_1_stored_refused": (good, sm_ok, "smoke_alone", 1.0, "stored", False),
        "main_without_smoke": (good, None, "main", R_, "philox", False),
        "main_smoke_failed": (good, dict(sm_ok, all_passed=False), "main", R_, "philox", False),
        "main_smoke_other_code": (good, dict(sm_ok, code_sha256=other), "main", R_, "philox", False),
        "main_smoke_over_cap": (good, dict(sm_ok, tests={"cost": {"main": {"within_limit": True, "within_cap": False}}}), "main", R_, "philox", False),
        "main_smoke_over_limit": (good, dict(sm_ok, tests={"cost": {"main": {"within_limit": False, "within_cap": True}}}), "main", R_, "philox", False),
        "main_ok": (good, sm_ok, "main", R_, "philox", True)}
    res = {k: nz.gate_state(c, s, code, p, r, b)[0] == want for k, (c, s, p, r, b, want) in cases.items()}
    want_txt = json.dumps({"a": 1}, sort_keys=True, indent=1) + "\n"
    folder = {
        "foreign_file": (["part_0.npz", "a19_run.json", "notes.txt"], want_txt, False, False),
        "map5km_refused": (["a19_run.json", "map5km_d0_h1.npy"], want_txt, False, False),
        "files_without_manifest": (["part_0.npz", "pop_0.npy"], None, False, False),
        "other_manifest": (["a19_run.json", "part_0.npz"], json.dumps({"a": 2}, sort_keys=True, indent=1) + "\n", False, False),
        "a16_manifest_refused": (["a16_run.json", "part_0.npz"], None, False, False),
        "smoke_not_empty": (["a19_run.json"], want_txt, True, False),
        "fresh_smoke_ok": ([], None, True, True),
        "resume_same_manifest_ok": (["a19_run.json", "part_0.npz", "pop_0.npy", "pop_ghap_0.npy", "cell40_0.npy", "pop_idx.npy",
                                     "part_1.npz.tmp_a19_123"], want_txt, False, True),
        "concurrent_ranks_tmp_manifest_ok": (["a19_run.json.tmp_a19_77"], None, False, True)}
    for k, (names, mtxt, fresh, want) in folder.items():
        res["claim_" + k] = nz.claim_state(names, mtxt, want_txt, fresh)[0] == want

    def refused(path, mode="rb"):
        try:
            open(path, mode).close()
        except PermissionError:
            return True
        except OSError:
            return False
        return False
    base = os.environ["FOEHN_WORK"]
    res["hook_station"] = refused("/nonexistent_a19_test/stations_x.npz")
    res["hook_openaq"] = refused("/nonexistent_a19_test/OpenAQ/x.csv")
    res["hook_monitor"] = refused("/nonexistent_a19_test/monitors/x.csv")
    for nm, p in (("a16_data", "global_model_forecasting_air_pollution/a16/data/nonexistent_a19_test/x.npy"),
                  ("a17", "global_model_forecasting_air_pollution/a17/nonexistent_a19_test/x.npy"),
                  ("a14", "global_model_forecasting_air_pollution/a14/nonexistent_a19_test/x.json"),
                  ("a15", "global_model_forecasting_air_pollution/a15/nonexistent_a19_test/x.json"),
                  ("amendments", "global_model_forecasting_air_pollution/protocol_amendments/nonexistent_a19_test/x.md"),
                  ("run_folder_E", "eccv/cran-pm/experiments/global1km/runs/nonexistent_a19_test/x.npy"),
                  ("natcomms_N", "eccv/cran-pm/paper/natcomms/nonexistent_a19_test/x.tex")):
        res[f"hook_write_{nm}_refused"] = refused(f"{base}/{p}", "wb") and refused(os.path.realpath(base) + f"/{p}", "ab")
        res[f"hook_read_{nm}_allowed"] = not refused(f"{base}/{p}", "rb")
    try:
        import os as _os
        _os.rename("/nonexistent_a19_test_src", f"{base}/global_model_forecasting_air_pollution/a16/nonexistent_a19_test/y")
        res["hook_rename_into_a16_refused"] = False
    except PermissionError:
        res["hook_rename_into_a16_refused"] = True
    except OSError:
        res["hook_rename_into_a16_refused"] = False
    res["hook_write_a19_allowed"] = not refused(str(nz.A19 / "nonexistent_a19_test" / "x.npy"), "wb")
    check("GATES", all(res.values()), **res)


def t_cost():
    k0, j0 = WIN[10]
    t = time.time()
    x = [egg.tile_window(nz.ARNoise19(2, 3, RHO), m, 25 * k0, 25 * j0, PH, PW) for m in (4, 5, 6, 7)]
    R["cost"] = {"window_lead3_4_negated_members_s": round(time.time() - t, 1), "window_shape": list(x[0].shape),
                 "finite": bool(all(np.isfinite(v.astype(np.float32)).all() for v in x))}
    print("cost", R["cost"], flush=True)


def main():
    t0 = time.time()
    for name, fn in (("DIFF", t_diff), ("GOLD", t_gold), ("A16_BYTES", t_a16_bytes), ("NEG_TILES", t_neg_tiles),
                     ("LAW", t_law), ("GATES", t_gates), ("PLANS", t_plans), ("LOADER", t_loader), ("NEG_WINDOWS", t_neg_windows)):
        guarded(name, fn)
    try:
        t_cost()
    except Exception as e:
        R["cost"] = {"error": repr(e)}
    R["code_sha256"] = nz.run_code_sha()
    R["all_passed"] = all(v["pass"] for v in R["tests"].values()) and "golden19" in R
    R["seconds"] = round(time.time() - t0, 1)
    nz.save_json(nz.CHECK_JSON, R)
    print("ALL PASSED" if R["all_passed"] else "SOME CHECKS FAILED", f"{R['seconds']:.0f}s", flush=True)
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
