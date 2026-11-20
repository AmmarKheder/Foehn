"""Common paths and helpers of the calibration analysis (helper "reliability", 2026-11-19). Read-only on the project;
every output goes under analyses/reliability/."""
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/eccv/cran-pm")
HERE = ROOT / "paper" / "natcomms" / "analyses" / "reliability"
DATA, FIGS = HERE / "data", HERE / "figs"
FIG = ROOT / "experiments" / "figures"                       # house style modules (imported read-only, python -B)
G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
RUNS = ROOT / "experiments" / "global1km" / "runs"
FINAL = RUNS / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"
BEFORE = RUNS / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12"      # the same model before the CRPS training
AURD = ROOT / "experiments" / "benchmark" / "aurora"
CAMS_ZARR = "/scratch/project_462001804/ammar/eccv/data/world/cams_forecast_daily/2022.zarr"

NETS = {0: "Europe (EEA)", 1: "China (CNEMC)", 3: "USA (EPA)", 4: "Korea (AirKorea)", 5: "Japan (NIES)", 6: "India (CPCB)"}
NET_SHORT = {0: "EEA", 1: "CNEMC", 3: "EPA", 4: "AirKorea", 5: "NIES", 6: "CPCB"}
THR = (15.0, 37.5, 75.0)                                      # WHO 24-h guideline level, interim targets 3 and 1
LEADS = (1, 2, 3, 4, 5, 6, 7)


def stations():
    """station file of the paper (UTC-day means) and the 6,324 official stations with at least 30 valid days"""
    st = np.load(G / "stations_2022.npz")
    src = st["source"]; obs = st["daily"].astype(np.float32)
    official = np.isin(src, list(NETS))
    keep = official & (np.isfinite(obs).sum(1) >= 30)
    return dict(lat=st["lat"], lon=st["lon"], src=src, obs=obs, official=official, keep=keep)


def fair_crps(E, y):
    """fair CRPS (Ferro 2014) of members E (..., M) against y (...): mean |x - y| - sum_{i<j} |xi - xj| / (M (M - 1))"""
    M = E.shape[-1]
    Es = np.sort(E, -1)
    w = (2 * np.arange(M) - M + 1).astype(E.dtype)             # sum_{i<j} (x(j) - x(i)) = sum_k (2k - M + 1) x(k)
    pair = (Es * w).sum(-1) / (M * (M - 1))
    return np.abs(E - y[..., None]).mean(-1) - pair


def fair_crps_nan(E, y):
    """fair CRPS of a ragged ensemble: E (m, N) with NaN for missing members, y (N,); returns (crps (N,), size (N,))"""
    Es = np.sort(E, 0)                                          # NaN last
    m = np.isfinite(Es).sum(0)
    k = np.arange(E.shape[0])[:, None]
    w = np.where(k < m[None], 2 * k - m[None] + 1, 0).astype(np.float64)
    pair = np.nansum(np.where(k < m[None], Es, 0.0) * w, 0) / np.maximum(m * (m - 1), 1)
    crps = np.nansum(np.abs(E - y[None]), 0) / np.maximum(m, 1) - pair
    return np.where(m >= 2, crps, np.nan), m


def block_weights(nd, B=7, n=10000, seed=0):
    """moving-block bootstrap over the nd start days in order (blocks of B consecutive start days, as
    gc_style.block_bootstrap): matrix (n, nd) of the number of times each start day is drawn"""
    rng = np.random.default_rng(seed)
    idx = (rng.integers(0, nd - B + 1, size=(n, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(n, -1)[:, :nd]
    W = np.zeros((n, nd), np.float64)
    np.add.at(W, (np.arange(n)[:, None], idx), 1.0)
    return W


def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float)
    return float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))
