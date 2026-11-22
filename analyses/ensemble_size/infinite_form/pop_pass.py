#!/usr/bin/env python3
"""A14 class U at the 8.1 M populated pixels (against the 1 km estimates; no station or monitor is read here).

  python pop_pass.py stored [workers]
      the four stored members (RUN/eval_pop_crps_e01, 179 start days x leads 1 and 3): per forecast, unweighted and weighted
      by residents, the sums n, sse, var of analyses/reliability/scripts/s3_pop.py (pixels >= 100 residents, estimate and
      four members finite; identity asserted against its data/sums_pop.npz, 'All populated pixels'), the fair CRPS and its
      pair term, and the delete-one-member sums (3 members)                         -> data/pop4_terms.npz
  python pop_pass.py run16 TAG DIR [DIR ...] [--workers W]
      the sixteen-member forecasts of the given directories (a13/warn16/sample16.py output): on the pixels of s3_pop.py where
      the 16 members are also finite, the sufficient statistics of every class-U score of any subset of the 16 members
      (d_i = x_i - y): D_ij = sum w d_i d_j, A_i = sum w |d_i|, P_ij = sum w |x_i - x_j|, and for 15 / 37.5 / 75 ug m-3
      E_ij = sum w e_i e_j, F_i = sum w e_i o, O = sum w o (e_i = x_i > T, o = y > T); unweighted and weighted by residents;
      with the stored four members on the same pixels (n, sse, var). The members at the 8,053 station pixels are kept
      for the station scorers (stations are not read here)                        -> data/pop16_<TAG>.npz, data/st16_<TAG>.npz"""
import glob
import sys
import time
from multiprocessing import get_context

import numpy as np

from u_common import DATA, EVP, G, index16, safe_npz, stations_in_pop

THR = (15.0, 37.5, 75.0)
pp = np.load(G / "pop_pixels.npz")
POP = pp["pop"].astype(np.float64); MINPOP = float(pp["min_pop"])
assert np.array_equal(pp["idx"], np.load(EVP / "pop_idx.npy"))
POPOK = POP >= MINPOP
STPOS = stations_in_pop()
S = {}


def stored_index():
    out = {}
    for p in sorted(glob.glob(str(EVP / "part_*.npz"))):
        rank = int(p.split("_")[-1][:-4])
        keys = np.load(p)["keys"]
        for n, (d, h, t) in enumerate(keys):
            out[(int(d), int(h))] = (rank, n, int(t))
    return out


def pair_sorted(E):
    m = E.shape[1]
    Es = np.sort(E, 1); w = (2 * np.arange(m) - m + 1).astype(np.float32)
    return (Es * w).sum(1) / (m * (m - 1))


def one_stored(key):
    rank, n, t = S["K4"][key]
    E = np.asarray(np.load(EVP / f"pop_{rank}.npy", mmap_mode="r")[n], np.float32)
    g = np.asarray(np.load(EVP / f"pop_ghap_{rank}.npy", mmap_mode="r")[n], np.float32)
    ok = np.isfinite(g) & np.isfinite(E).all(1) & POPOK
    E = np.where(ok[:, None], E, 0.0); g = np.where(ok, g, 0.0)
    out = {}
    for nm, w in (("", ok.astype(np.float64)), ("w", POP * ok)):
        def s(x):
            return float(np.dot(w, np.asarray(x, np.float64)))
        mean = E.mean(1); var = E.var(1, ddof=1); err = mean - g
        pair = pair_sorted(E); crps = np.abs(E - g[:, None]).mean(1) - pair
        r = dict(n=float(w.sum()), sse=s(err * err), var=s(var), pair=s(pair), crps=s(crps))
        lsse, lvar, lpair = [], [], []
        for i in range(4):
            L = np.delete(E, i, 1)
            lsse.append(s((L.mean(1) - g) ** 2)); lvar.append(s(L.var(1, ddof=1))); lpair.append(s(pair_sorted(L)))
        r.update(loo_sse=lsse, loo_var=lvar, loo_pair=lpair)
        out[nm] = r
    return key, t, out


def one16(key):
    p16, n16, t16 = S["R16"][key]
    rank, n, t = S["K4"][key]
    assert t == t16, key
    X = np.asarray(np.load(p16, mmap_mode="r")[n16], np.float32)                 # (pixels, 16)
    E4 = np.asarray(np.load(EVP / f"pop_{rank}.npy", mmap_mode="r")[n], np.float32)
    g = np.asarray(np.load(EVP / f"pop_ghap_{rank}.npy", mmap_mode="r")[n], np.float32)
    st16 = X[np.clip(STPOS, 0, None)].astype(np.float16); st16[STPOS < 0] = np.nan
    ok4 = np.isfinite(g) & np.isfinite(E4).all(1) & POPOK
    ok = ok4 & np.isfinite(X).all(1)
    Xo, go, E4o = X[ok], g[ok], E4[ok]
    D = Xo.astype(np.float64) - go.astype(np.float64)[:, None]
    res = {}
    for nm, w in (("", np.ones(ok.sum())), ("w", POP[ok])):
        r = {"n": float(w.sum()), "n4": float((POP * ok4).sum() if nm == "w" else ok4.sum())}
        Dw = D * w[:, None]
        r["D"] = Dw.T @ D
        r["A"] = (np.abs(D) * w[:, None]).sum(0)
        P = np.zeros((16, 16))
        for i in range(15):
            a = np.abs(Xo[:, i:i + 1] - Xo[:, i + 1:]).astype(np.float64)
            P[i, i + 1:] = w @ a
        r["P"] = P + P.T
        Ee, Fo, Oo = [], [], []
        for T in THR:
            e = (Xo > T).astype(np.float32); o = (go > T)
            ew = e * w[:, None].astype(np.float32)
            Ee.append((ew.T @ e).astype(np.float64)); Fo.append(w[o] @ e[o].astype(np.float64)); Oo.append(float(w[o].sum()))
        r["E"], r["F"], r["O"] = np.stack(Ee), np.stack(Fo), np.array(Oo)
        m4 = E4o.mean(1); v4 = E4o.var(1, ddof=1)
        r["sse4"] = float(w @ ((m4 - go).astype(np.float64) ** 2)); r["var4"] = float(w @ v4.astype(np.float64))
        res[nm] = r
    return key, t, res, st16


def main():
    mode = sys.argv[1]
    t0 = time.time()
    S["K4"] = stored_index()
    if mode == "stored":
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        keys = sorted(S["K4"], key=lambda k: (k[1], k[0]))
        with get_context("fork").Pool(workers) as pool:
            rows = pool.map(one_stored, keys, chunksize=2)
        out = {"keys": np.array([(d, h, t) for (d, h), t, _ in rows], np.int32)}
        for nm in ("", "w"):
            for q in ("n", "sse", "var", "pair", "crps", "loo_sse", "loo_var", "loo_pair"):
                out[nm + q] = np.array([r[2][nm][q] for r in rows])
        ref = np.load(__import__("u_common").RELD / "sums_pop.npz")          # design: reliability/data_ar (S1 step 5)
        rk = [tuple(x) for x in ref["keys"].tolist()]
        pos = [rk.index(tuple(x)) for x in out["keys"].tolist()]
        chk = {}
        for nm in ("", "w"):
            for q in ("n", "var", "sse", "crps"):
                a, b = out[nm + q], ref[nm + q][pos, -1]
                chk[nm + q] = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-9)))
        print("identity with analyses/reliability/data/sums_pop.npz ('All populated pixels'), largest relative difference:", chk, flush=True)
        assert all(v < 1e-6 for v in chk.values()), chk
        out["identity"] = np.array(str(chk))
        safe_npz(DATA / "pop4_terms.npz", **out)
    else:
        tag = sys.argv[2]
        args = [a for a in sys.argv[3:]]
        workers = 8
        if "--workers" in args:
            i = args.index("--workers"); workers = int(args[i + 1]); args = args[:i] + args[i + 2:]
        only = None
        if "--only" in args:
            i = args.index("--only"); only = {tuple(int(v) for v in x.split(":")) for x in args[i + 1].split(",")}; args = args[:i] + args[i + 2:]
        R16 = index16(args)
        if only is not None:
            R16 = {k: v for k, v in R16.items() if k in only}
        S["R16"] = R16
        keys = sorted(R16, key=lambda k: (k[1], k[0]))
        print(f"{len(keys)} sixteen-member forecasts in {args}", flush=True)
        with get_context("fork").Pool(workers) as pool:
            rows = pool.map(one16, keys, chunksize=1)
        out = {"keys": np.array([(d, h, t) for (d, h), t, _, _ in rows], np.int32)}
        for q in ("n", "n4", "D", "A", "P", "E", "F", "O", "sse4", "var4"):
            out[q] = np.stack([np.stack([np.asarray(r[2][nm][q], np.float64) for nm in ("", "w")]) for r in rows])
        safe_npz(DATA / f"pop16_{tag}.npz", **out)
        safe_npz(DATA / f"st16_{tag}.npz", keys=out["keys"], ens16=np.stack([r[3] for r in rows]), stpos=STPOS)
    print(f"done {mode}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
