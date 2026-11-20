"""Shared helpers of the issuance analysis: SEDI and its companions from contingency counts, and the moving-block bootstrap
over start days of the paper (compare_fair.py L166-169: blocks of seven consecutive start days, 10,000 resamples, seed 0)."""
import numpy as np


def rates(c):
    """c (..., 4) = hits, misses, false alarms, correct negatives -> dict of SEDI, hit rate (POD), false-alarm ratio,
    frequency bias (warnings / events), false-alarm rate and base rate"""
    h, m, f, n = np.moveaxis(np.asarray(c, np.float64), -1, 0)
    with np.errstate(all="ignore"):
        pod = h / (h + m); pofd = f / (f + n)
        H = np.clip(pod, 1e-6, 1 - 1e-6); F = np.clip(pofd, 1e-6, 1 - 1e-6)
        sedi = (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)) / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))
        return dict(sedi=sedi, pod=pod, far=f / (h + f), fb=(h + f) / (h + m), pofd=pofd, base=(h + m) / (h + m + f + n),
                    n=h + m + f + n)


def sedi(c):
    return rates(c)["sedi"]


def block_index(nd, B=7, n=10000, seed=0):
    """(n, nd) indices of a moving-block bootstrap over nd consecutive start days, as compare_fair.py"""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, nd - B + 1, size=(n, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(n, -1)[:, :nd]


def boot_counts(cd, idx):
    """cd (days, ..., 4) counts per start day -> (n, ..., 4) counts of every resample (sum over the drawn days)"""
    out = np.zeros((idx.shape[0],) + cd.shape[1:])
    for j in range(idx.shape[1]):                                       # nd additions of (n, ...) arrays: light in memory
        out += cd[idx[:, j]]
    return out


def ci(x, q=(2.5, 97.5)):
    return np.nanpercentile(x, q, axis=0)
