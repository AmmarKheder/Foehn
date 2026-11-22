#!/usr/bin/env python3
"""A17 scorer and decision (a17/A17_draft.md sections 4.4-4.6 and 7), CPU only. 2021 only (audit hook against 2022 files),
no station or monitor file. Cells, truth, weights and bootstrap are those of A15 (a15/prep.py, a15/score.py):
  cells    the populated 0.25 deg cells complete in every stored 2021 run (a15/arnoise.episodes_2021, sha256 of prep.json)
  truth    map25_ghap of the part files (asserted bit-identical between a15/data/main, a17/data/main and ext_2021 Y)
  weights  residents; cell-weight values are kept as est_cell
  CI       paired two-way bootstrap (30 runs x 10-degree tiles, revise.summarise), 2000 draws, seed 20261121, 95 %
Members (25 km cell means, ug m-3): ids 0..3 = A15's run M (a15/data/main, map25_mk), ids 4..7 = the A17 main run
(a17/data/main, the exact negatives of the noise of ids 0..3). Pairs (k, 4 + k); independent members 0..3.
Per lead (1, 3) and pooled (13), regimes by the STORED fixed-design 4-member mean of the same forecast (ext_2021 Mn: an
ensemble mean drawn with other noise but the particles member_particles(i, h, 0..3) of members 0..3, not those of 4..7;
it affects reported statistics only; all, < 15, 15-75, > 75 ug m-3):
  MSE of the ensemble mean: one member (0..3), one partner (4..7), two independent members (6 pairs of 0..3), one
  opposite pair (4), four independent (0..3), two opposite pairs (6 choices; (0, 4, 1, 5) is one), all members (8 = four
  pairs); 1/M law from 0..3: MSE_inf = mean[(xbar - y)^2 - s2/4], V = mean[s2], MSE_M = MSE_inf + V/M (M = 1, 2, 4, 8,
  16, inf); the same law from 4..7 alone (check); inv_meq = (MSE - MSE_inf)/V, equivalent independent members 1/inv_meq.
  rho_anti   correlation of the departures of the two members of a pair from the mean r of the OTHER pairs, corrected for
             the reference: with p, t the half sum and half difference of the pair and K other pairs,
             rho = (A - B)/(A + B), A = K/(K + 1) sum (p - r)^2 (unbiased for the variance of a pair mean), B = sum t^2;
             this is the pooled within-pair correlation c of the departures from the infinite mean (expected 0 for
             independent members, -1 for an exactly odd response); the uncorrected correlation is kept as rho_anti_raw.
  rho_null   the same with pseudo-pairs of independent members, one FIXED pairing per set of four independent members:
             (0, 1) | (2, 3) and, when there are four pairs, (4, 5) | (6, 7), each pair against the other (K = 1). Pooling
             the 3 pairings of the same four members is not used: it is identically 0 (sum of A = sum of B for any data).
             Both rho are also computed in log space.
  bias       mean(xbar_all - xbar_0..3) = (mean of the pair means - four-independent mean), ug m-3, relative to the truth,
             and in the network's space (x - 15)/20 (dejavu.MU, SD: the network's space is AFFINE, so every MSE is /400,
             every correlation, ratio and equivalent count identical; the network-space MSEs are computed and compared)
  probability  spread over error sqrt((M + 1)/M s2)/RMSE for 0..3, for all members, and the value the 1/M law expects for
             as many independent members; Brier of the exceedance frequency above 37.5 and 75 for all members against
             the 1/M-law expectation BS_inf + q(1 - q)/M (unbiased from 0..3)
Decision (section 7, applied mechanically on the regime 'all', in this order): INVALID (smoke or guard G), SUCCESS, ORDER,
FAIL. This scorer is frozen: its sha256 is written into its outputs and stated in A17_draft.md 6.4 and 7; any later change
needs an addendum.
    python score17.py            -> code/out/score17.json (needs a17/data/main and data/smoke_check.json)
    python score17.py --tests    -> code/out/null17.json  (null and known-answer tests on A15's 4 members, no A17 output)"""
import noise17 as nz  # noqa: I001  (first)
import argparse
import itertools
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(nz.A15_DIR))
import arnoise as ar  # noqa: E402  (A15: episodes_2021, cells_hash, deny_2022; installs its station hook too)

ar.deny_2022()
egg = nz.egg
SELF_SHA256 = nz.sha_file(__file__)                                       # the frozen scorer (A17_draft.md 6.4, 7)
NBOOT, SEED, EPS = 2000, 20261121, 0.1
MU, SD = float(egg.MU), float(egg.SD)
THRS = (37.5, 75.0)
REG = {"all": (-np.inf, np.inf), "lt15": (-np.inf, 15.0), "15_75": (15.0, 75.0), "gt75": (75.0, np.inf)}
LEADS = (1, 3)


# ======================================================================== data
def load_members(folder, want, cells, mids):
    out = {}
    for p in sorted(folder.glob("part_*.npz")):
        if ".tmp" in p.name:
            continue
        with np.load(p) as z:
            assert str(z["base"]) == "philox", p
            if mids is not None:
                assert [int(m) for m in z["member_ids"]] == list(mids), (p, z["member_ids"])
            keys, kr = z["keys"].tolist(), z["keys_rho"].tolist()
            sel = [j for j, (d, h, _) in enumerate(keys) if (d, h) in want]
            if not sel:
                continue
            mk, g25 = z["map25_mk"], z["map25_ghap"]
            for j in sel:
                d, h, dt = keys[j]
                assert (d, h) not in out and dt == d + h and float(kr[j]) == nz.RHO_A17, (p, keys[j], kr[j])
                out[(d, h)] = (mk[j].reshape(mk.shape[1], -1)[:, cells].astype(np.float32), g25[j].reshape(-1)[cells].copy())
    assert set(out) == set(want), (folder, sorted(set(want) - set(out))[:5])
    return out


def base_data():
    prep = json.loads((nz.A15_DIR / "results" / "prep.json").read_text())
    D, revise = ar.episodes_2021()
    assert ar.cells_hash(D["cells"]) == prep["cells_sha256"]
    lab = [int(v) for v in D["labels"]]
    Ds = ar.subset_runs(D, [lab.index(d) for d in nz.CONF])
    cells, w = Ds["cells"], Ds["pop"].astype(np.float64)
    tid = Ds["tid"]
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    Y16 = np.stack([Ds["Y"][:, h - 1] for h in LEADS], 1)                     # (N, 2, C) float16, the truth
    Mst = np.stack([Ds["Mn"][:, h - 1] for h in LEADS], 1).astype(np.float32)   # stored fixed-design mean: regimes
    want = {(d, h) for d in nz.CONF for h in LEADS}
    a15 = load_members(nz.A15_DIR / "data" / "main", want, cells, None)
    N, C = len(nz.CONF), len(cells)
    X = np.empty((N, 8, 2, C), np.float32)
    bad = 0
    for r, d in enumerate(nz.CONF):
        for li, h in enumerate(LEADS):
            mk, g25 = a15[(d, h)]
            assert mk.shape == (4, C), mk.shape
            X[r, :4, li] = mk
            bad += not np.array_equal(g25.view(np.uint16), Y16[r, li].view(np.uint16))
    assert bad == 0, f"a15 map25_ghap differs from ext_2021 Y in {bad} forecasts"
    assert np.isfinite(X[:, :4]).all() and np.isfinite(Mst).all()
    info = dict(n_cells=C, residents=float(w.sum()), cells_sha256=prep["cells_sha256"], tiles=int(len(starts)), runs=N)
    return X, Y16.astype(np.float32), Mst, w, starts, revise, cells, info


def add_a17(X, Y, cells):
    want = {(d, h) for d in nz.CONF for h in LEADS}
    a17 = load_members(nz.DATA / "main", want, cells, (4, 5, 6, 7))
    bad = 0
    for r, d in enumerate(nz.CONF):
        for li, h in enumerate(LEADS):
            mk, g25 = a17[(d, h)]
            X[r, 4:, li] = mk
            bad += not np.array_equal(g25.astype(np.float32), Y[r, li])
    assert bad == 0, f"a17 map25_ghap differs from the truth in {bad} forecasts"
    assert np.isfinite(X).all()


# ======================================================================== per-cell columns
def rho_cols(x, pairs, pre):
    """sums over pairs of A = K/(K+1)(p - r)^2, B = t^2 and of the raw departure products (reference = other pairs)"""
    P = len(pairs)
    K = P - 1
    a_, b_ = [a for a, _ in pairs], [b for _, b in pairs]
    pm, th = 0.5 * (x[a_] + x[b_]), 0.5 * (x[a_] - x[b_])
    r = (pm.sum(0)[None] - pm) / K
    da, db = x[a_] - r, x[b_] - r
    return {pre + "A": (K / (K + 1) * (pm - r) ** 2).sum(0), pre + "B": (th ** 2).sum(0), pre + "dd": (da * db).sum(0),
            pre + "da2": (da ** 2).sum(0), pre + "db2": (db ** 2).sum(0)}


def null_cols(x, sets, pre):
    """pseudo-pairs of independent members: for every set of four independent members (a, b, c, d), the fixed pairing
    (a, b) | (c, d), each pair against the other pair (K = 1). (Summing the 3 pairings of one set gives A = B exactly.)"""
    acc = {}
    for s in sets:
        assert len(s) == 4, s
        for n, v in rho_cols(x[list(s)], [(0, 1), (2, 3)], pre).items():   # rho_cols takes each pair against the other
            acc[n] = acc.get(n, 0.0) + v
    return acc


def mse_cols(x, y, pairs, ind, sfx=""):
    e = lambda f: (f - y) ** 2                                         # noqa: E731
    xi = x[ind]
    b_ = [b for _, b in pairs]
    pm = np.stack([0.5 * (x[a] + x[b]) for a, b in pairs])
    c = {"m1_ind": e(xi).mean(0), "m1_part": e(x[b_]).mean(0),
         "m2_ind": np.mean([e(0.5 * (xi[i] + xi[j])) for i, j in itertools.combinations(range(len(ind)), 2)], 0),
         "m_pair": e(pm).mean(0), "m4_ind": e(xi.mean(0)), "m_all": e(x.mean(0)),
         "m_2pairs": np.mean([e(0.5 * (pm[j] + pm[k])) for j, k in itertools.combinations(range(len(pairs)), 2)], 0),
         "m_2pairs_first": e(0.5 * (pm[0] + pm[1])), "s2_ind": xi.var(0, ddof=1), "bias": x.mean(0) - xi.mean(0)}
    if len(b_) >= 2:
        c["m4_part"], c["s2_part"] = e(x[b_].mean(0)), x[b_].var(0, ddof=1)
    return {k + sfx: v for k, v in c.items()}


def cell_cols(x, y, pairs, ind):
    """per-cell columns of one (run, lead): x (members, C) and y (C,) in ug m-3, float64"""
    c = {"one": np.ones_like(y), "y": y}
    c.update(mse_cols(x, y, pairs, ind))
    c["s2_all"] = x.var(0, ddof=1)
    sets = [ind] + ([[b for _, b in pairs]] if len(pairs) == 4 else [])   # 0..3 and, with four pairs, the partners 4..7
    c.update(rho_cols(x, pairs, "ra_"))
    c.update(null_cols(x, sets, "rn_"))
    lx = np.log(np.maximum(x, EPS))
    c.update(rho_cols(lx, pairs, "la_"))
    c.update(null_cols(lx, sets, "ln_"))
    xn, yn = (x - MU) / SD, (y - MU) / SD                              # the network's own space (affine)
    c.update({k: v for k, v in mse_cols(xn, yn, pairs, ind, "_n").items() if k[:-2] in
              ("m1_ind", "m2_ind", "m_pair", "m4_ind", "m_all", "m_2pairs", "s2_ind", "bias")})
    xi = x[ind]
    for T in THRS:
        o = (y > T).astype(np.float64)
        pa, pi = (x > T).mean(0), (xi > T).mean(0)
        c[f"bs_all_{T:g}"], c[f"bs_ind_{T:g}"], c[f"pq_ind_{T:g}"], c[f"o_{T:g}"] = (pa - o) ** 2, (pi - o) ** 2, pi * (1 - pi), o
    return c


def aggregate(X, Y, Mst, w, starts, revise, pairs, ind):
    N = X.shape[0]
    agg = revise.Agg(N, starts, w)
    for r in range(N):
        for li, h in enumerate(LEADS):
            cols = cell_cols(X[r, :, li].astype(np.float64), Y[r, li].astype(np.float64), pairs, ind)
            for g, (lo, hi) in REG.items():
                mk = np.ones(len(w)) if g == "all" else ((Mst[r, li] >= lo) & (Mst[r, li] <= hi) if lo == 15.0 else
                                                         (Mst[r, li] > lo) & (Mst[r, li] < hi)).astype(np.float64)
                agg.add(r, f"L{h}|{g}", {f"L{h}|{g}|{k}": v * mk for k, v in cols.items()})
    return agg


# ======================================================================== statistics of summed columns
def stats(S, M_ind, M_all, M_part):
    out = {}
    for g in REG:
        for L in ("1", "3", "13"):
            def get(k):
                return S[f"L1|{g}|{k}"] + S[f"L3|{g}|{k}"] if L == "13" else S[f"L{L}|{g}|{k}"]
            pre = f"{g}/L{L}/"
            n = get("one")
            mse = {k: get(k) / n for k in ("m1_ind", "m1_part", "m2_ind", "m_pair", "m4_ind", "m_2pairs", "m_2pairs_first", "m_all")}
            V = get("s2_ind") / n
            inf_ = mse["m4_ind"] - V / M_ind
            for k, v in mse.items():
                out[pre + "mse_" + k] = v
            out[pre + "V"], out[pre + "mse_law_inf"] = V, inf_
            for Mm in (1, 2, 4, 8, 16):
                out[pre + f"mse_law_{Mm}"] = inf_ + V / Mm
            if f"L1|{g}|m4_part" in S:
                Vp = get("s2_part") / n
                out[pre + "V_part"], out[pre + "mse_law_inf_part"] = Vp, get("m4_part") / n - Vp / M_part
            out[pre + "d_4ind_minus_pair"] = mse["m4_ind"] - mse["m_pair"]
            out[pre + "d_2ind_minus_pair"] = mse["m2_ind"] - mse["m_pair"]
            out[pre + "d_law_all_minus_all"] = inf_ + V / M_all - mse["m_all"]
            out[pre + "inv_meq_pair"] = (mse["m_pair"] - inf_) / V
            out[pre + "inv_meq_2pairs"] = (mse["m_2pairs"] - inf_) / V
            out[pre + "inv_meq_all"] = (mse["m_all"] - inf_) / V
            out[pre + "rho_from_mse_pair"] = 2 * out[pre + "inv_meq_pair"] - 1
            for nm, p_ in (("rho_anti", "ra_"), ("rho_null", "rn_"), ("rho_anti_log", "la_"), ("rho_null_log", "ln_")):
                A, Bv = get(p_ + "A"), get(p_ + "B")
                out[pre + nm] = (A - Bv) / (A + Bv)
                out[pre + nm + "_raw"] = get(p_ + "dd") / np.sqrt(get(p_ + "da2") * get(p_ + "db2"))
            out[pre + "mean_y"] = get("y") / n
            out[pre + "bias"], out[pre + "bias_rel"], out[pre + "bias_n"] = get("bias") / n, get("bias") / get("y"), get("bias_n") / n
            for k in ("m1_ind", "m2_ind", "m_pair", "m4_ind", "m_all", "m_2pairs"):
                out[pre + "mse_n_" + k] = get(k + "_n") / n
                out[pre + "net_over_ug_times_400_" + k] = get(k + "_n") * SD * SD / get(k)
            out[pre + "rmse_ratio_part_over_ind"] = np.sqrt(mse["m1_part"] / mse["m1_ind"])
            if f"L1|{g}|s2_part" in S:
                out[pre + "spread_ratio_part_over_ind"] = get("s2_part") / get("s2_ind")
            out[pre + "soe_ind"] = np.sqrt((M_ind + 1) / M_ind * V) / np.sqrt(mse["m4_ind"])
            out[pre + "soe_all"] = np.sqrt((M_all + 1) / M_all * get("s2_all") / n) / np.sqrt(mse["m_all"])
            out[pre + "soe_law_all"] = np.sqrt((M_all + 1) / M_all * V) / np.sqrt(inf_ + V / M_all)
            for T in THRS:
                t = f"{T:g}"
                bs_inf = (get(f"bs_ind_{t}") - get(f"pq_ind_{t}") / (M_ind - 1)) / n
                qq = get(f"pq_ind_{t}") * M_ind / (M_ind - 1) / n
                out[pre + f"brier_all_{t}"] = get(f"bs_all_{t}") / n
                out[pre + f"brier_ind_{t}"] = get(f"bs_ind_{t}") / n
                out[pre + f"brier_law_inf_{t}"] = bs_inf
                out[pre + f"brier_law_all_{t}"] = bs_inf + qq / M_all
                out[pre + f"d_brier_law_all_minus_all_{t}"] = bs_inf + qq / M_all - get(f"bs_all_{t}") / n
                out[pre + f"base_rate_{t}"] = get(f"o_{t}") / n
            out[pre + "residents_share"] = n / (S["L1|all|one"] + S["L3|all|one"]) * (2 if L != "13" else 1)
    return out


def meq(res, key):
    """equivalent number of independent members 1/inv_meq, with the interval inverted from that of inv_meq"""
    v = res[key]
    est = 1.0 / v["est"] if v["est"] > 0 else float("inf")
    lo, hi = (float(x) for x in v["two_way"])
    return {"est": est, "two_way": [1.0 / hi if hi > 0 else float("inf"), 1.0 / lo if lo > 0 else float("inf")]}


def score(X, Y, Mst, w, starts, revise, pairs, ind):
    t0 = time.time()
    agg = aggregate(X, Y, Mst, w, starts, revise, pairs, ind)
    M_all, M_part = X.shape[1], len(pairs)
    res, info = revise.summarise(agg, X.shape[0], 1, NBOOT, SEED, fn=lambda Dd: stats(Dd, len(ind), M_all, M_part))
    for g in REG:
        for L in ("1", "3", "13"):
            for k in ("pair", "2pairs", "all"):
                res[f"{g}/L{L}/meq_{k}"] = meq(res, f"{g}/L{L}/inv_meq_{k}")
    info["seconds"] = round(time.time() - t0, 1)
    return res, info


# ======================================================================== decision (section 7, fixed in advance)
def decide(res, smoke_ok):
    est = lambda k: float(res[k]["est"])                                  # noqa: E731
    lo = lambda k: float(res[k]["two_way"][0])                            # noqa: E731
    G = {}
    for L in (1, 3):   # each tolerance >= 4.8 null standard deviations (design review); INVALID: no retuned tolerance, ever
        G[f"rmse_ratio_partner_L{L}_within_5pct"] = abs(est(f"all/L{L}/rmse_ratio_part_over_ind") - 1) <= 0.05
        G[f"spread_ratio_partner_L{L}_within_15pct"] = abs(est(f"all/L{L}/spread_ratio_part_over_ind") - 1) <= 0.15
        G[f"bias_L{L}_within_2pct_of_truth"] = abs(est(f"all/L{L}/bias_rel")) <= 0.02
    cond = {"pair_below_4ind_L1": est("all/L1/d_4ind_minus_pair") > 0, "pair_below_4ind_L3": est("all/L3/d_4ind_minus_pair") > 0,
            "L1_two_way_interval_above_0": lo("all/L1/d_4ind_minus_pair") > 0}
    fail = {"pooled_two_way_interval_2ind_minus_pair_contains_0_or_below": lo("all/L13/d_2ind_minus_pair") <= 0}
    if not smoke_ok or not all(G.values()):
        outcome = "INVALID"
        reasons = (["smoke gate not passed"] if not smoke_ok else []) + [k for k, v in G.items() if not v]
    elif all(cond.values()):
        outcome, reasons = "SUCCESS", ["every SUCCESS condition of section 7 holds"]
    elif not any(fail.values()):
        outcome, reasons = "ORDER", (["pooled two-way interval of MSE(two independent) - MSE(pair) above 0; SUCCESS not met:"]
                                     + [k for k, v in cond.items() if not v])
    else:
        outcome, reasons = "FAIL", [k for k, v in fail.items() if v]
    return dict(outcome=outcome, reasons=reasons, guard=G, success_conditions=cond, fail_conditions=fail)


HEAD = ("mse_m1_ind", "mse_m2_ind", "mse_m_pair", "mse_m4_ind", "mse_m_2pairs", "mse_m_all", "mse_law_8", "mse_law_16", "mse_law_inf",
        "d_4ind_minus_pair", "d_2ind_minus_pair", "rho_anti", "rho_anti_raw", "rho_null", "rho_anti_log", "rho_null_log", "meq_pair",
        "meq_2pairs", "meq_all", "bias", "bias_rel", "soe_ind", "soe_all", "soe_law_all", "brier_all_37.5", "brier_law_all_37.5",
        "brier_all_75", "brier_law_all_75")


def headline(res):
    return {f"L{L}": {k: {"est": res[f"all/L{L}/{k}"]["est"], "two_way": res[f"all/L{L}/{k}"]["two_way"]} for k in HEAD}
            for L in ("1", "3", "13")}


# ======================================================================== tests (no A17 output read)
def near(v, t):
    """known-answer rule of the tests: |estimate - target| <= 1.5 x the half-width of the two-way 95 % interval (about 3
    standard errors; several targets are tested at once), and whether the interval itself holds the target"""
    lo, hi = (float(x) for x in v["two_way"])
    return dict(est=float(v["est"]), target=t, two_way=[lo, hi], interval_holds=bool(lo <= t <= hi),
                ok=bool(abs(float(v["est"]) - t) <= max(1.5 * (hi - lo) / 2, 1e-9)))


def tests(X, Y, Mst, w, starts, revise, info):
    out = {"base": info}
    N, _, _, C = X.shape
    x4 = np.ascontiguousarray(X[:, :4])
    mu = x4.mean(1)
    sig = np.sqrt(x4.var(1, ddof=1))
    rng = np.random.default_rng(SEED + 17)
    e1 = sig[:, None] * rng.standard_normal((N, 4, 2, C), dtype=np.float32)
    e2 = sig[:, None] * rng.standard_normal((N, 4, 2, C), dtype=np.float32)
    pairs8, ind = [(k, 4 + k) for k in range(4)], [0, 1, 2, 3]
    g = lambda res, k: res[f"all/L13/{k}"]                                 # noqa: E731
    cases = {}
    # N0: the real A15 members 0..3 (no A17 output), pseudo-pairs (0, 1), (2, 3) of independent members
    res, inf = score(x4, Y, Mst, w, starts, revise, [(0, 1), (2, 3)], ind)
    d = decide(res, True)
    chk = {"rho_anti": near(g(res, "rho_anti"), 0.0), "rho_null": near(g(res, "rho_null"), 0.0),
           "inv_meq_pair": near(g(res, "inv_meq_pair"), 0.5), "rho_anti_log": near(g(res, "rho_anti_log"), 0.0)}
    cases["N0_real_a15_pseudo_pairs"] = dict(
        expect="independent pseudo-pairs of the real members: rho_anti 0, rho_null 0 (also in log), inv_meq_pair 1/2 (one "
               "pseudo-pair = two independent members), class FAIL", checks=chk, outcome=d["outcome"],
        passed=all(v["ok"] for v in chk.values()) and d["outcome"] == "FAIL", headline=headline(res), bootstrap=inf)
    for nm, c, want in (("N1_synthetic_c0", 0.0, "FAIL"), ("N2_synthetic_c-0.8", -0.8, "SUCCESS"), ("N5_synthetic_c-0.3", -0.3, "ORDER")):
        Xs = np.empty((N, 8, 2, C), np.float32)
        Xs[:, :4] = mu[:, None] + e1
        Xs[:, 4:] = mu[:, None] + np.float32(c) * e1 + np.float32(np.sqrt(1 - c * c)) * e2
        res, inf = score(Xs, Y, Mst, w, starts, revise, pairs8, ind)
        d = decide(res, True)
        chk = {"rho_anti": near(g(res, "rho_anti"), c), "rho_null": near(g(res, "rho_null"), 0.0),
               "inv_meq_pair": near(g(res, "inv_meq_pair"), (1 + c) / 2), "inv_meq_2pairs": near(g(res, "inv_meq_2pairs"), (1 + c) / 4),
               "inv_meq_all": near(g(res, "inv_meq_all"), (1 + c) / 8), "bias_rel": near(g(res, "bias_rel"), 0.0)}
        cls = {} if want != "ORDER" else {
            "pooled_two_way_2ind_minus_pair_above_0": float(g(res, "d_2ind_minus_pair")["two_way"][0]) > 0,
            "pair_not_below_4ind_L1_and_L3": all(float(res[f"all/L{L}/d_4ind_minus_pair"]["est"]) <= 0 for L in (1, 3))}
        cases[nm] = dict(expect=f"Gaussian members around the real 4-member mean with the real spread, within-pair correlation {c}: "
                                f"rho_anti {c}, rho_null 0, inv_meq of one pair, two pairs, eight members {(1 + c) / 2}, {(1 + c) / 4}, "
                                f"{(1 + c) / 8}; class " + want + (" (interval of MSE(two independent) - MSE(pair) above 0, pair not "
                                                                   "better than four members)" if want == "ORDER" else ""),
                         checks=chk, class_checks=cls, outcome=d["outcome"], guard=d["guard"],
                         passed=all(v["ok"] for v in chk.values()) and all(cls.values()) and d["outcome"] == want,
                         headline=headline(res), bootstrap=inf)
        del Xs
    for nm in ("N3_reflection", "N4_copy"):
        Xs = np.empty((N, 8, 2, C), np.float32)
        Xs[:, :4] = x4
        Xs[:, 4:] = (2 * mu[:, None] - x4) if nm == "N3_reflection" else x4
        res, inf = score(Xs, Y, Mst, w, starts, revise, pairs8, ind)
        d = decide(res, True)
        e = lambda k: float(res[f"all/L13/{k}"]["est"])                    # noqa: E731
        if nm == "N3_reflection":
            ok = abs(e("rho_anti") + 1) < 1e-5 and abs(e("mse_m_pair") / e("mse_m4_ind") - 1) < 1e-5 and abs(e("inv_meq_pair") - 0.25) < 1e-4 \
                and d["outcome"] == "ORDER"
            expect = ("x(4+k) = 2 mean - x(k): rho_anti = -1, MSE(pair) = MSE(four independent), inv_meq_pair = 1/4 (exact up to float32), "
                      "class ORDER (D_h = 0: the pair mean is the 4-member mean, so one pair equals, not beats, four members)")
        else:
            ok = abs(e("rho_anti") - 1) < 1e-9 and abs(e("mse_m_pair") / e("mse_m1_ind") - 1) < 1e-9 and abs(e("inv_meq_pair") - 1) < 1e-6 \
                and d["outcome"] == "FAIL"
            expect = "x(4+k) = x(k): rho_anti = +1, MSE(pair) = MSE(one member), inv_meq_pair = 1, class FAIL (exact)"
        cases[nm] = dict(expect=expect, rho_anti_L13=e("rho_anti"), mse_pair_L13=e("mse_m_pair"), mse_4ind_L13=e("mse_m4_ind"),
                         mse_1_L13=e("mse_m1_ind"), inv_meq_pair_L13=e("inv_meq_pair"),
                         d_2ind_minus_pair_L13=res["all/L13/d_2ind_minus_pair"], d_4ind_minus_pair_L1=res["all/L1/d_4ind_minus_pair"],
                         d_4ind_minus_pair_L3=res["all/L3/d_4ind_minus_pair"], outcome=d["outcome"], passed=bool(ok),
                         bootstrap=inf)
        del Xs
    out["cases"] = cases
    out["all_passed"] = all(v["passed"] for v in cases.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    X, Y, Mst, w, starts, revise, cells, info = base_data()
    print(f"loaded {time.time() - t0:.0f}s: {info}", flush=True)
    if a.tests:
        R = {"what": __doc__, "score17_sha256": SELF_SHA256, **tests(X, Y, Mst, w, starts, revise, info),
             "seconds": round(time.time() - t0, 1)}
        nz.save_json(nz.OUT / "null17.json", R)
        print(json.dumps(nz.jsonable({k: (v["passed"], v["outcome"]) for k, v in R["cases"].items()})),
              "ALL PASSED" if R["all_passed"] else "SOME TESTS FAILED", f"{R['seconds']:.0f}s score17 sha256 {SELF_SHA256}", flush=True)
        return 0 if R["all_passed"] else 1
    sm = nz.read_json(nz.SMOKE_JSON) or {}
    add_a17(X, Y, cells)
    res, binfo = score(X, Y, Mst, w, starts, revise, [(k, 4 + k) for k in range(4)], [0, 1, 2, 3])
    d = decide(res, sm.get("all_passed") is True)
    R = {"what": __doc__, "score17_sha256": SELF_SHA256, **d, "headline": headline(res), "data": info,
         "bootstrap": {"draws": NBOOT, "seed": SEED, **binfo},
         "all_stats": res, "seconds": round(time.time() - t0, 1)}
    nz.save_json(nz.OUT / "score17.json", R)
    print(f"OUTCOME {d['outcome']}: {d['reasons']} | score17 sha256 {SELF_SHA256} | {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
