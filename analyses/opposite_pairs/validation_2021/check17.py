#!/usr/bin/env python3
"""A17 CPU checks before any GPU run (a17/A17_draft.md section 6.1), in the job container (numpy 1.26.3), CPU only:
  DIFF       every line of eval17.run17 that differs from a15/eval_ar.run_ar ends with '# A17-H<n>' (H1-H6) and the lines
             removed are exactly eval17.REMOVED_LINES; WindowsAR17.__getitem__ against arnoise22.WindowsAR22.__getitem__
             (W1, noise17.REMOVED_LINES_WINDOWS); _audit verbatim from a15/arnoise.py; hooks installed before any import
  GOLD       A16's generator gives a15/golden.json (2021 keys) and a16/golden22.json (2022 keys); golden17() (negated
             members and their partners at rho*) identical in this process and in two fresh processes -> golden17 field
  A15_BYTES  members 0-3 of ARNoise17 (rho*) are byte-equal to arnoise22.ARNoise (in process) and to A15's own module
             a15/arnoise.py (fresh process: its audit hook stays there), tiles and one window cut
  NEG_TILES  member 4+k is the exact negative of member k: uint16 equal to member k with the sign bit flipped (144 tiles:
             starts 9, 105, 353 x leads 1, 2, 3, 7 x k = 0..3 x 3 tiles), and equal in value to the field built from the
             negated innovations -eps_j with the same keys and coefficients (bit differences allowed only on signed zeros)
  NEG_WINDOWS through the UNCHANGED tile_window: all 21 windows at (9, 3) and 3 windows at (353, 1), members 4..7 = sign
             flip of 0..3; full_map(member 5, (9, 3)) = sign flip of full_map(member 1) and equals each of its 21 window cuts
  LAW        negated fields (rho*, members 4, 5 against 0, 1, starts 9 and 17, leads 1-7, tiles (7, 30) and (20, 50)): mean
             0 +- 0.01, variance 1 +- 0.01, correlation with the partner -1, lag-j correlation rho^j +- 0.01, correlation
             with the partner at another lead -rho^|h - h'| +- 0.01, |correlation| < 0.01 with every other member at any
             pair of leads and with the other start
  LOADER     the real 2021 dataset (TestGreen from the checkpoint's hyper-parameters): x0 of WindowsAR17(ids 0..3) = x0 of
             arnoise22.WindowsAR22(2021, rho*, philox, 4 members); WindowsAR17(ids 4..7) = sign flip of it; ids 0..7 = both;
             all other inputs byte-equal; identical through a DataLoader with 0 and 4 workers (fork, spawn) and pickle
  PLANS      main = the 30 Conf starts x leads 1 and 3 (60), all present in a15/data/main at rho* (keys only); the smoke
             items are Conf lead-2 forecasts of a15/data/main, outside main; every item is a 2021 sample; the job scripts
             call exactly these plans and write only under a17/data
  GATES      noise17.gate_state refuses a missing / failed check17 or smoke or changed code; the audit hooks refuse a
             station path and a path of the running evaluation (fake paths: nothing is opened)
  cost       CPU seconds of one window of 4 negated members at lead 3 (DataLoader worker cost), reported
    python check17.py -> code/out/check17.json (exit 1 if a check fails)"""
import noise17 as nz  # noqa: I001  (first)
import ast
import collections
import difflib
import json
import pickle
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import eval17

an, egg = nz.an, nz.egg
PH, PW, GH, GW = egg.PH, egg.PW, egg.GH, egg.GW
R = {"what": __doc__, "numpy": np.__version__, "torch": torch.__version__, "python": sys.version.split()[0], "tests": {}}
WIN = egg.windows(np.load(egg.G / "meta.npz")["land"])
assert len(WIN) == 21, len(WIN)
RHO = nz.rho_star()
SIGN = np.uint16(0x8000)


def check(name, cond, **info):
    R["tests"][name] = {"pass": bool(cond), **info}
    print(("PASS " if cond else "FAIL ") + name + " " + json.dumps(nz.jsonable(info))[:700], flush=True)


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
    mk = re.compile(r"  # A17-%s(\d+)$" % letter)
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


def t_diff():
    A15E, A15N, A16N = nz.A15_DIR / "eval_ar.py", nz.A15_DIR / "arnoise.py", Path(an.__file__)
    ok_e, i_e = diff(fsrc(A15E, "run_ar"), fsrc(eval17.__file__, "run17"), "H", eval17.REMOVED_LINES, eval17.HUNKS)
    ok_w, i_w = diff(fsrc(A16N, "__getitem__", "WindowsAR22"), fsrc(nz.__file__, "__getitem__", "WindowsAR17"), "W",
                     nz.REMOVED_LINES_WINDOWS, (1,))
    head = Path(nz.__file__).read_text().split("import arnoise22")[0]
    verb = {"_audit_a15": fsrc(A15N, "_audit") == fsrc(nz.__file__, "_audit"),
            "hooks_before_imports": "sys.addaudithook(_audit)" in head and "sys.addaudithook(_audit17)" in head,
            "eval17_imports_noise17_first": Path(eval17.__file__).read_text().split('"""')[2].lstrip().startswith("import noise17")}
    check("DIFF", ok_e and ok_w and all(verb.values()), files={"a15_eval_ar": str(A15E), "a16_arnoise22": str(A16N)},
          run17=i_e, WindowsAR17=i_w, verbatim=verb)


# ---------------------------------------------------------------- GOLD
def subrun(code, prefix, cwd):
    p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=str(cwd), timeout=1800)
    line = [l for l in p.stdout.splitlines() if l.startswith(prefix + " ")]
    return json.loads(line[-1][len(prefix) + 1:]) if p.returncode == 0 and line else {"error": p.stderr[-3000:]}


SUB17 = ("import sys; sys.path.insert(0, %r); import noise17, json; "
         "print('G17 ' + json.dumps(noise17.golden17(), sort_keys=True), flush=True)") % str(nz.CODE)


def t_gold():
    g15 = an.golden_hashes() == json.loads((nz.A15_DIR / "golden.json").read_text())
    g22 = an.golden22_hashes() == json.loads((nz.A16_DIR / "golden22.json").read_text())
    here = nz.golden17()
    subs = [subrun(SUB17, "G17", nz.CODE) for _ in range(2)]
    R["golden17"] = here
    check("GOLD", g15 and g22 and all(s == here for s in subs) and here["numpy"] == "1.26.3", a15_golden_json=g15,
          a16_golden22_json=g22, golden17_subprocess_equal=[s == here for s in subs],
          first16={k: v[:16] for k, v in here.items() if isinstance(v, str)})


# ---------------------------------------------------------------- A15_BYTES
A15_ITEMS = [(9, 1, 0, 14, 35), (9, 3, 3, 0, 0), (353, 3, 1, 28, 70), (105, 1, 2, 7, 30), (17, 3, 0, 20, 50), (329, 3, 3, 3, 66)]
SUB15 = ("import sys, json; sys.path.insert(0, %r); import arnoise as ar; import numpy as np; items = %r; "
         "out = [ar.sha(ar.ARNoise(2021, d, h, %r, 'philox').tile(m, tr, tc)) for d, h, m, tr, tc in items]; "
         "W = ar.egg.windows(np.load(ar.egg.G / 'meta.npz')['land']); k0, j0 = W[10]; "
         "out.append(ar.sha(ar.egg.tile_window(ar.ARNoise(2021, 9, 3, %r, 'philox'), 1, 25 * k0, 25 * j0, ar.egg.PH, ar.egg.PW))); "
         "print('A15SHA ' + json.dumps(out), flush=True)")


def t_a15_bytes():
    mine = [nz.sha(nz.ARNoise17(d, h, RHO).tile(m, tr, tc)) for d, h, m, tr, tc in A15_ITEMS]
    a16 = [nz.sha(an.ARNoise(2021, d, h, RHO, "philox", 4).tile(m, tr, tc)) for d, h, m, tr, tc in A15_ITEMS]
    k0, j0 = WIN[10]
    mine.append(nz.sha(egg.tile_window(nz.ARNoise17(9, 3, RHO), 1, 25 * k0, 25 * j0, PH, PW)))
    a16.append(nz.sha(egg.tile_window(an.ARNoise(2021, 9, 3, RHO, "philox", 4), 1, 25 * k0, 25 * j0, PH, PW)))
    theirs = subrun(SUB15 % (str(nz.A15_DIR), A15_ITEMS, RHO, RHO), "A15SHA", nz.A15_DIR)
    check("A15_BYTES", mine == a16 and mine == theirs, items=[list(x) for x in A15_ITEMS] + [["window 10 cut", 9, 3, 1]],
          equal_a16=[a == b for a, b in zip(mine, a16)],
          equal_a15=[a == b for a, b in zip(mine, theirs)] if isinstance(theirs, list) else theirs)


# ---------------------------------------------------------------- NEG_TILES
def neg_arith(d, h, k, tr, tc):
    """the field of member k built from the negated innovations -eps_j (same keys, same coefficients, same order)"""
    c = an.coeffs(h, RHO)
    acc = np.zeros((nz.TILE, nz.TILE), np.float32)
    for j in range(1, h + 1):
        if c[j - 1] == 0:
            continue
        acc += c[j - 1] * np.negative(an.innovation(2021, d, k, j, tr, tc))
    return acc.astype(np.float16)


def t_neg_tiles():
    n = bad_flip = bad_val = zero_sign = 0
    for d in (9, 105, 353):
        for h in (1, 2, 3, 7):
            z = nz.ARNoise17(d, h, RHO)
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
    z = nz.ARNoise17(9, 3, RHO)
    cuts5 = {}
    for w, (k0, j0) in enumerate(WIN):
        for k in range(4):
            pos = egg.tile_window(z, k, 25 * k0, 25 * j0, PH, PW)
            neg = egg.tile_window(z, 4 + k, 25 * k0, 25 * j0, PH, PW)
            n += 1
            bad += not flipped(neg, pos)
            if k == 1:
                cuts5[w] = neg
    z1 = nz.ARNoise17(353, 1, RHO)
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
        z = nz.ARNoise17(d, h, RHO)
        return np.stack([z.tile(m, tr, tc).astype(np.float64) for tr, tc in TL])
    z = {(d, m, h): Z(d, m, h) for d in (9, 17) for m in (0, 1, 4, 5) for h in range(1, 8)}
    neg = [(d, m, h) for d in (9, 17) for m in (4, 5) for h in range(1, 8)]
    mean_dev = max(abs(float(z[k].mean())) for k in neg)
    var_dev = max(abs(float(z[k].var()) - 1.0) for k in neg)
    partner = min(-corr(z[(d, m, h)], z[(d, m - 4, h)]) for d, m, h in neg)
    lag = max(abs(corr(z[(d, m, h)], z[(d, m, h + j)]) - RHO ** j) for d in (9, 17) for m in (4, 5) for h in range(1, 8)
              for j in range(1, 8 - h))
    cross_partner = max(abs(corr(z[(d, m, h)], z[(d, m - 4, h2)]) + RHO ** abs(h - h2)) for d in (9, 17) for m in (4, 5)
                        for h in range(1, 8) for h2 in range(1, 8))
    others = max(abs(corr(z[(d, m, h)], z[(d, o, h2)])) for d in (9, 17) for m in (4, 5) for o in (0, 1, 4, 5)
                 if o not in (m, m - 4) for h in range(1, 8) for h2 in range(1, 8))
    starts = max(abs(corr(z[(9, m, h)], z[(17, o, h2)])) for m in (4, 5) for o in (0, 1, 4, 5) for h in range(1, 8)
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
        state = torch.load(an.CKPT, map_location="cpu", weights_only=False)
        hp = state["hyper_parameters"]
        del state
        store, emis = hp.get("store", "green"), bool(hp.get("emis", False))
        P = int(egg.Npy(egg.G / store / "hres" / "origins_2021.npy").shape[-2])
        DS["ds"] = egg.TestGreen(7, P, store, emis, 2021, hp.get("emis_store"))
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


def t_loader():
    ds = dataset()
    idx_of = {int(ds.doy[i]): i for i in sorted({i for i, h in ds.samples})}
    rng = np.random.default_rng(1521)
    pick = [nz.PLANS["main"]["items"][j] for j in rng.choice(60, 6, replace=False)]
    jobs = [(idx_of[d], h, int(w)) for (d, h), w in zip(pick, rng.integers(0, len(WIN), 6))]
    A = an.WindowsAR22(ds, jobs, WIN, 4, [], 2021, RHO, "philox")
    P0 = nz.WindowsAR17(ds, jobs, WIN, (0, 1, 2, 3), RHO)
    N4 = nz.WindowsAR17(ds, jobs, WIN, (4, 5, 6, 7), RHO)
    E8 = nz.WindowsAR17(ds, jobs[:2], WIN, tuple(range(8)), RHO)
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
    parts = {f"{int(ds.doy[i])}_{h}": {"ids_0_3": [nz.egg.member_particles(i, h, m, DS["info"]["particles"], 16).tolist() for m in range(4)][:1],
                                       "ids_4_7": [nz.egg.member_particles(i, h, m, DS["info"]["particles"], 16).tolist() for m in range(4, 8)][:1]}
             for i, h, _ in jobs[:1]}
    items = []
    for (d, h), w in (((9, 3), 0), ((353, 1), 10), ((17, 3), 20)):
        z = nz.ARNoise17(d, h, RHO)
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
          items=[(int(ds.doy[i]), h, w) for i, h, w in jobs], x0_ids0_3_vs_WindowsAR22_mismatches=bad_a16,
          x0_ids4_7_not_sign_flip=bad_neg, other_inputs_mismatching=bad_other, same_keys=keys_same, eight_member_mismatches=bad8,
          loader_workers0_vs_4_fork=ok_f, loader_workers0_vs_4_spawn=ok_s, pickle_round_trip=pk_ok,
          particles_example_informative=parts, **DS["info"])


# ---------------------------------------------------------------- PLANS
def a15_main_keys():
    out = {}
    for p in sorted((nz.A15_DIR / "data" / "main").glob("part_*.npz")):
        if ".tmp" in p.name:
            continue
        with np.load(p) as z:
            assert str(z["base"]) == "philox"
            for (d, h, t), r in zip(z["keys"].tolist(), z["keys_rho"].tolist()):
                assert (d, h) not in out
                out[(d, h)] = (t, float(r))
    return out


def t_plans():
    keys = a15_main_keys()
    main, sm = nz.PLANS["main"]["items"], nz.PLANS["smoke_alone"]["items"] + nz.PLANS["smoke_eight"]["items"]
    in15 = all(it in keys and keys[it] == (it[0] + it[1], RHO) for it in main + sm)
    ds = dataset()
    idx_of = {int(ds.doy[i]): i for i in sorted({i for i, h in ds.samples})}
    have = set(ds.samples)
    in_ds = all(d in idx_of and (idx_of[d], h) in have for d, h in main + sm)
    conf = all(d in nz.CONF for d, _ in main + sm) and sorted(main) == sorted((d, h) for d in nz.CONF for h in (1, 3))
    smoke_outside = not (set(sm) & set(main)) and all(h == 2 for _, h in sm)
    mids = nz.PLANS["main"]["mids"] == (4, 5, 6, 7) == nz.PLANS["smoke_alone"]["mids"] and nz.PLANS["smoke_eight"]["mids"] == tuple(range(8))
    sh = {n: (nz.CODE / n).read_text() for n in ("smoke.sh", "main.sh", "cpu.sh")}
    calls = {n: re.findall(r"eval17\.py --plan (\w+)", "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("#")))
             for n, t in sh.items()}
    outs = {n: re.findall(r"#SBATCH --output=(\S+)", t) for n, t in sh.items()}
    jobs_ok = (calls["smoke.sh"] == ["smoke_alone", "smoke_eight"] and calls["main.sh"] == ["main"] and calls["cpu.sh"] == []
               and all(str((nz.CODE / o).resolve()).startswith(str(nz.DATA) + "/") for n in ("smoke.sh", "main.sh") for o in outs[n])
               and all(str((nz.CODE / o).resolve()).startswith(str(nz.OUT) + "/") for o in outs["cpu.sh"])
               and all(re.search(r"^#SBATCH --account=\S+", t, re.M) for t in sh.values())
               and "--partition=small-g" in sh["smoke.sh"] and "--partition=standard-g" in sh["main.sh"] and "--partition=small\n" in sh["cpu.sh"])
    args = {p: vars(eval17.parse_args(["--plan", p])) for p in nz.PLANS}
    args_ok = all(a["members"] == len(nz.PLANS[p]["mids"]) and a["year"] == 2021 and a["base"] == "philox" and a["steps"] == 4
                  and a["psub"] == 16 and a["ckpt"].endswith("ckpts/epoch=01.ckpt") for p, a in args.items())
    check("PLANS", in15 and in_ds and conf and smoke_outside and mids and jobs_ok and args_ok, main=len(main), smoke=sm,
          all_in_a15_main_at_rho=in15, all_2021_samples=in_ds, conf_only=conf, smoke_outside_main=smoke_outside, member_ids=mids,
          job_script_calls=calls, job_outputs=outs, job_scripts_ok=jobs_ok, args_ok=args_ok)


# ---------------------------------------------------------------- GATES
def t_gates():
    code = nz.run_code_sha()
    other = dict(code, **{"code/eval17.py": "0" * 64})
    good = {"all_passed": True, "code_sha256": code}
    cases = {
        "no_check": (None, None, "smoke_alone", False), "check_failed": ({"all_passed": False, "code_sha256": code}, None, "smoke_alone", False),
        "check_other_code": ({"all_passed": True, "code_sha256": other}, None, "smoke_eight", False),
        "smoke_ok_after_check": (good, None, "smoke_alone", True), "main_without_smoke": (good, None, "main", False),
        "main_smoke_failed": (good, {"all_passed": False, "code_sha256": code}, "main", False),
        "main_smoke_other_code": (good, {"all_passed": True, "code_sha256": other}, "main", False),
        "main_ok": (good, {"all_passed": True, "code_sha256": code}, "main", True)}
    res = {k: nz.gate_state(c, s, code, p)[0] == want for k, (c, s, p, want) in cases.items()}

    def refused(path):
        try:
            open(path, "rb").close()
        except PermissionError:
            return True
        except OSError:
            return False
        return False
    res["hook_station"] = refused("/nonexistent_a17_test/stations_x.npz")
    res["hook_openaq"] = refused("/nonexistent_a17_test/OpenAQ/x.csv")
    res["hook_running_eval_ar_folder"] = refused("/nonexistent_a17_test/eval_x_ar/part_0.npz")
    res["hook_a16_data"] = refused("/nonexistent_a17_test/a16/data/x.npz")
    res["hook_a16_results"] = refused("/nonexistent_a17_test/a16/results/x.json")
    res["hook_allows_a15_data"] = not refused("/nonexistent_a17_test/a15/data/main/part_0.npz")
    res["part_names"] = all(eval17.PART_NAME.fullmatch(n) for n in ("part_0.npz", "part_7.npz", "part_0.tmp.npz")) and not any(
        eval17.PART_NAME.fullmatch(n) for n in ("pop_0.npy", "part_x.npz", "a16_run.json"))
    check("GATES", all(res.values()), **res)


def t_cost():
    k0, j0 = WIN[10]
    t = time.time()
    x = [egg.tile_window(nz.ARNoise17(9, 3, RHO), m, 25 * k0, 25 * j0, PH, PW) for m in (4, 5, 6, 7)]
    R["cost"] = {"window_lead3_4_negated_members_s": round(time.time() - t, 1), "window_shape": list(x[0].shape),
                 "finite": bool(all(np.isfinite(v.astype(np.float32)).all() for v in x))}
    print("cost", R["cost"], flush=True)


def main():
    t0 = time.time()
    for name, fn in (("DIFF", t_diff), ("GOLD", t_gold), ("A15_BYTES", t_a15_bytes), ("NEG_TILES", t_neg_tiles),
                     ("LAW", t_law), ("GATES", t_gates), ("PLANS", t_plans), ("LOADER", t_loader), ("NEG_WINDOWS", t_neg_windows)):
        guarded(name, fn)
    try:
        t_cost()
    except Exception as e:
        R["cost"] = {"error": repr(e)}
    R["code_sha256"] = nz.run_code_sha()
    R["all_passed"] = all(v["pass"] for v in R["tests"].values()) and "golden17" in R
    R["seconds"] = round(time.time() - t0, 1)
    nz.save_json(nz.CHECK_JSON, R)
    print("ALL PASSED" if R["all_passed"] else "SOME CHECKS FAILED", f"{R['seconds']:.0f}s", flush=True)
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
