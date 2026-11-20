"""cdffm_aq_x.copula_x -- the copula study (PROTOCOL.md 0.3 and 4.4, C0/C1) on the FROZEN baseline, smoke setup.

A pointwise functional flow is a Gaussian copula: x1(p) = Phi_c(p)(x0(p)) with Phi monotone, so the joint law of a
sample is the Gaussian copula of the prior x0 with the learned marginals. The flow-matching loss is pointwise, hence
blind to the prior's correlation: the prior can be replaced after training without changing any pointwise score.
  latents  normal scores z_hat = Phi^-1(x1) of the (dequantised) truth at points of patches (disks of 25, 100 and
           300 km, 256 points) on training days (stored points) and hold-out days (full fields): backward RK4 of the
           EMA velocity from t = 1 to 0, with the day's condition c(p) and statics stored next to them.
  fit      priors x0(p) = sum_m w_m(p) xi_m(p), sum_m w_m(p)^2 = 1 (unit marginals, pointwise scores unchanged),
           xi_m independent sphere power-law RFF fields in wavelength bands (1-4.3, 4.3-20, 20-100, 100-500,
           500-3000 km), fitted by the exact Gaussian-copula likelihood of the patches (Cholesky, 256 x 256):
             P0   iid (nugget only, reference)
             P1   the baseline prior itself (bands 4.3-500 km with the masses of the k^(1-beta) law, no fit)
             P1s  stationary: fitted fixed band weights
             P1r  static non-stationary: weights from the 6 pixel statics (MLP 6 -> 16 -> K)
             P3   conditional: weights from the network's own condition c(p) (LIIF 64 + doy 2; MLP 66 -> 32 -> K)
  test     Delta LS = mean copula log-likelihood per point on the hold-out patches, prior minus P1 (nats/point),
           with a day-block bootstrap interval; by radius. The marginal terms cancel exactly.
  (eval_x --prior-file priors.pt:NAME samples members with a fitted prior: block scores, spectra, maps.)

  python -m cdffm_aq_x.copula_x latents --run s2_europe --runs-root runs --out logs/x/copula
  python -m cdffm_aq_x.copula_x fit --dir logs/x/copula ; python -m cdffm_aq_x.copula_x test --dir logs/x/copula
"""
import argparse
import json
import os
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdffm_aq.data_aq import PixelArrays, assemble_inputs, doy_of, open_store, raw_to_device, store_kwargs, target_z
from cdffm_aq.model_aq import NoiseSpec, draw_sphere_noise, eval_sphere_noise, noise_chunk, sphere_xyz_km
from cdffm_aq.train_aq import noise_spec

from . import LOGS_X, RUNS_ROOT
from .model_x import load_run_x

BANDS_KM = ((1.0, 4.3), (4.3, 20.0), (20.0, 100.0), (100.0, 500.0), (500.0, 3000.0))
BAND_NAMES = ("1-4km", "4-20km", "20-100km", "100-500km", "500-3000km")
RADII_KM = (25.0, 100.0, 300.0)


# ---------------------------------------------------------------------------------------------------------------
# bands: specs, correlation functions, the baseline's band masses
# ---------------------------------------------------------------------------------------------------------------

def band_specs(beta, n_features=1024, bands=BANDS_KM):
    return [NoiseSpec(n_features, beta, lo, hi) for lo, hi in bands]


def band_corr_table(spec, d_max_km=7000.0, dd_km=0.1, nk=3000):
    """C(d) = E_k[sin(2 pi k d) / (2 pi k d)] for |k| ~ k^(1 - beta) on [1/lmax, 1/lmin] (3-D isotropic directions
    on the sphere's chord distance d); returns (d grid, C) as float64 tensors"""
    k0, k1 = 1.0 / spec.lmax_km, 1.0 / spec.lmin_km
    lk = np.linspace(np.log(k0), np.log(k1), nk)
    k = np.exp(lk)
    wk = k ** (2.0 - spec.beta)                     # density k^(1-beta) times dk = k dlogk
    wk /= wk.sum()
    d = np.arange(0.0, d_max_km + dd_km, dd_km)
    arg = 2.0 * np.pi * np.outer(d, k)
    s = np.where(arg > 1e-9, np.sin(arg) / np.where(arg > 1e-9, arg, 1.0), 1.0)
    return torch.from_numpy(d), torch.from_numpy(s @ wk)


def baseline_band_masses(spec, bands=BANDS_KM):
    """share of the radial law's mass of the baseline prior (spec) in each band -> (K,) sums to 1"""
    e = 2.0 - spec.beta
    k0, k1 = 1.0 / spec.lmax_km, 1.0 / spec.lmin_km
    tot = k1 ** e - k0 ** e
    out = []
    for lo, hi in bands:
        a, b = max(1.0 / hi, k0), min(1.0 / lo, k1)
        out.append((b ** e - a ** e) / tot if b > a else 0.0)          # e < 0: both signs negative
    return np.array(out)


class CopulaPrior(nn.Module):
    """x0 = sum_m w_m(p) xi_m(p); weights: 'iid' (band 0 only), 'fixed' (K logits), 'static' (MLP on the statics),
    'cond' (MLP on the condition); w = sqrt(softmax(logits)) so sum w^2 = 1"""

    def __init__(self, kind, beta, cond_dim=66, static_dim=6, init_masses=None, bands=BANDS_KM, n_features=1024):
        super().__init__()
        self.kind, self.beta, self.bands = kind, float(beta), tuple(bands)
        self.K = len(bands)
        self.specs = band_specs(beta, n_features, bands)
        init = np.log(np.maximum(init_masses, 1e-4)) if init_masses is not None else np.zeros(self.K)
        if kind == "iid":
            init = np.full(self.K, -20.0)
            init[0] = 0.0
        self.logits = nn.Parameter(torch.tensor(init, dtype=torch.float32), requires_grad=kind == "fixed")
        if kind == "static":
            self.net = nn.Sequential(nn.Linear(static_dim, 16), nn.GELU(), nn.Linear(16, self.K))
        elif kind == "cond":
            self.net = nn.Sequential(nn.Linear(cond_dim, 32), nn.GELU(), nn.Linear(32, self.K))
        else:
            self.net = None
        if self.net is not None:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        tabs = [band_corr_table(s) for s in self.specs]
        self.register_buffer("d_grid", tabs[0][0])
        self.register_buffer("corr", torch.stack([c for _, c in tabs]))        # (K, nd) float64
        self.dd = float(self.d_grid[1] - self.d_grid[0])

    def weights(self, cond=None, static=None):
        """(..., K) with sum of squares 1"""
        if self.kind == "static":
            logit = self.logits + self.net(static.float())
        elif self.kind == "cond":
            logit = self.logits + self.net(cond.float())
        else:
            ref = cond if cond is not None else static
            logit = self.logits.expand(*ref.shape[:-1], self.K)
        return torch.sqrt(F.softmax(logit, dim=-1).clamp(min=1e-12))

    def corr_of_distance(self, d_km):
        """(K, ...) correlation of each band at chord distances d_km (float64)"""
        x = (d_km / self.dd).clamp(0, self.d_grid.numel() - 1.001)
        i0 = x.floor().long()
        f = (x - i0.double())
        c = self.corr
        return c[:, i0] * (1 - f) + c[:, i0 + 1] * f

    def corr_matrix(self, lat, lon, w, iid_band0=True):
        """lat, lon (P, n) degrees, w (P, n, K) -> R (P, n, n) float64"""
        p = sphere_xyz_km(lat, lon)                                               # (P, n, 3)
        d = torch.cdist(p, p)
        C = self.corr_of_distance(d)                                              # (K, P, n, n)
        if iid_band0 and self.bands[0][0] <= 1.0:
            C = C.clone()
            eye = torch.eye(d.shape[-1], dtype=torch.bool, device=d.device)
            C[0] = torch.where(eye, torch.ones_like(C[0]), torch.zeros_like(C[0]))   # band 0 as a pure nugget
        wd = w.double()
        R = torch.einsum("pik,pjk,kpij->pij", wd, wd, C)
        return R

    def patch_loglik(self, zhat, lat, lon, cond=None, static=None, jitter=1e-4):
        """mean Gaussian copula log-density per point of each patch, (P,) float64"""
        w = self.weights(cond, static)
        R = self.corr_matrix(lat, lon, w)
        n = R.shape[-1]
        eye = torch.eye(n, dtype=R.dtype, device=R.device)
        L, info = torch.linalg.cholesky_ex(R + jitter * eye)
        bad = info > 0
        if bad.any():                                   # retry the failed patches with more jitter (rare; logged)
            for j in (1e-3, 1e-2):
                Lb, ib = torch.linalg.cholesky_ex(R[bad] + j * eye)
                L[bad] = Lb
                info[bad] = ib
                bad = info > 0
                if not bad.any():
                    break
            self.n_retry = getattr(self, "n_retry", 0) + 1
        z = zhat.double()[..., None]
        sol = torch.cholesky_solve(z, L)
        quad = (z * sol).sum((-2, -1))
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        return (-0.5 * quad - 0.5 * logdet - 0.5 * n * math.log(2 * math.pi)) / n


# ---------------------------------------------------------------------------------------------------------------
# latents
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def backward_rk4(model, x1, cond, n_steps=20):
    """Phi^-1: integrate dx/dt = v(x, c, t) from t = 1 to t = 0 (RK4), x1 (B, N, 1) -> x0"""
    B, N = x1.shape[:2]
    h = -1.0 / n_steps
    x = x1
    for s in range(n_steps):
        t = torch.full((B, N), 1.0 + s * h, device=x1.device)
        k1 = model.velocity(x, cond, t)
        k2 = model.velocity(x + 0.5 * h * k1, cond, t + 0.5 * h)
        k3 = model.velocity(x + 0.5 * h * k2, cond, t + 0.5 * h)
        k4 = model.velocity(x + h * k3, cond, t + h)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


class DayPoints:
    """points available on a day: stored training points (k, v) or all pixels (hold-out), sorted by tile row"""

    def __init__(self, store, k, v):
        self.k, self.v = k, v
        self.row, self.col = store.tile_row[k], store.tile_col[k]
        self.lat, self.lon = store.lat[k], store.lon[k]
        o = np.argsort(self.row, kind="stable")
        self.o = o
        self.row_sorted = self.row[o]

    def disk(self, i_centre, radius_km, rng, n_pts):
        """indices (into k) of n_pts points within radius_km of point i_centre, or None"""
        lat0, lon0 = self.lat[i_centre], self.lon[i_centre]
        pr = radius_km / 1.112
        r0, c0 = self.row[i_centre], self.col[i_centre]
        a = np.searchsorted(self.row_sorted, r0 - pr - 1)
        b = np.searchsorted(self.row_sorted, r0 + pr + 1, side="right")
        cand = self.o[a:b]
        # column pre-filter (integer compare) before the trigonometry: the same metric as below, so the candidate set
        # and its order are unchanged (the row band alone holds up to ~10 M pixels on the global mask: 100 s per day)
        pc = pr / max(np.cos(np.deg2rad(lat0)), 0.05) + 1.0
        cand = cand[np.abs(self.col[cand].astype(np.int64) - int(c0)) <= pc]
        dlat = np.deg2rad(self.lat[cand] - lat0)
        dlon = np.deg2rad(self.lon[cand] - lon0) * np.cos(np.deg2rad(lat0))
        d = 6371.0 * np.sqrt(dlat ** 2 + dlon ** 2)
        cand = cand[d <= radius_km]
        if cand.size < n_pts:
            return None
        return rng.choice(cand, n_pts, replace=False)


@torch.no_grad()
def make_latents(run, runs_root, ckpt, split, out_path, device, radii=RADII_KM, n_per_radius=16, n_pts=256,
                 rk4_steps=20, seed=0, max_days=None, day_idx=None):
    """split 'train' (stored training points) or any other name (hold-out full fields, e.g. 'hold', 'v3odd');
    day_idx: indices into the split's dates (default all; the global fallback prior uses 2023 V3 days, A1(c'''))."""
    rargs, model, step = load_run_x(run, device, ckpt, runs_root)
    spec = noise_spec(rargs)
    store = open_store(rargs.store, **store_kwargs(rargs))          # the run's fires / split / sigma_k
    pix = PixelArrays(store, device)
    dates = list(enumerate(store.train_dates if split == "train" else store.hold_dates))
    stations = None                                                  # A3(c): a stations run conditions on the day's monitors
    if getattr(rargs, "stations", None) and getattr(model, "stations", False):
        from cdffm_aq.stations import open_stations
        stations = open_stations(rargs, sorted({d.year for _, d in dates}), device)
    if day_idx is not None:
        dates = [dates[i] for i in day_idx]
    if max_days:
        dates = dates[:max_days]
    out = {k: [] for k in ("lat", "lon", "zhat", "x1", "cond", "static", "day", "radius", "k")}
    t0 = time.time()
    dp_hold = None
    for it, (t, d) in enumerate(dates):
        if split == "train":
            k, v = store.points(t)
            k = k.astype(np.int64)
            dp = DayPoints(store, k, v)
        else:
            v = store.holdout(t)
            k = np.arange(store.n_pix)
            if dp_hold is None:                                                   # same pixels every hold-out day
                dp_hold = DayPoints(store, k, v)
            dp = dp_hold
        doy = torch.tensor([doy_of(d)], device=device)
        raw = raw_to_device(store.raw_inputs(d), device)
        feat = model.encode(assemble_inputs(raw, pix, rargs.grid, rargs.transport))
        # patch centres seeded per day (not one stream over the days), so the latents of a day do not depend on the
        # days run before it: the day list can be spread over several processes (runs_global/p1r_K0.sh)
        rng = np.random.default_rng([seed, 0 if split == "train" else 1, int(t)])
        sel, rad = [], []
        for R in radii:
            got = 0
            tries = 0
            while got < n_per_radius and tries < 50 * n_per_radius:
                tries += 1
                ii = dp.disk(int(rng.integers(0, k.size)), R, rng, n_pts)
                if ii is not None:
                    sel.append(ii)
                    rad.append(R)
                    got += 1
        if not sel:
            continue
        ii = np.stack(sel)                                                        # (P, n) indices into k
        if getattr(store, "has_holes", False):        # GHAP (A2(a)): patches touching a hole (-1) are dropped
            ii = ii[(v[ii] >= 0).all(1)]
            rad = [r_ for r_, ok in zip(rad, (v[np.stack(sel)] >= 0).all(1)) if ok]
            if ii.shape[0] == 0:
                continue
        kk = torch.from_numpy(k[ii]).to(device)
        vv = torch.from_numpy(v[ii].astype(np.int16)).to(device)
        P, n = kk.shape
        lat, lon, st = pix.lat[kk], pix.lon[kk], pix.static[kk]
        mu, sig = pix.mu_sigma(kk, doy.expand(P))
        g = torch.Generator(device=device)
        g.manual_seed(seed * 1_000_003 + t * 1009 + (101 if split == "train" else 103))
        u = torch.rand(P, n, device=device, generator=g) - 0.5
        x1 = target_z(vv, mu, sig, u)[..., None]
        stn = None
        if stations is not None:
            stn = stations.features(d, lat.reshape(-1), lon.reshape(-1))[0][None]
        cond = model.condition(feat, lat.reshape(1, -1), lon.reshape(1, -1), st.reshape(1, -1, st.shape[-1]),
                               doy, stn=stn).reshape(P, n, -1)
        zhat = backward_rk4(model, x1, cond, rk4_steps)[..., 0]
        out["lat"].append(lat.cpu().numpy())
        out["lon"].append(lon.cpu().numpy())
        out["zhat"].append(zhat.cpu().numpy().astype(np.float32))
        out["x1"].append(x1[..., 0].cpu().numpy().astype(np.float32))
        out["cond"].append(cond.cpu().numpy().astype(np.float16))
        out["static"].append(st.cpu().numpy().astype(np.float16))
        out["k"].append(k[ii].astype(np.int64))
        out["day"].append(np.full(P, t, np.int32))
        out["radius"].append(np.array(rad, np.float32))
        if it % 20 == 0 or it == len(dates) - 1:
            z = zhat
            print(f"  [{split}] day {it}/{len(dates)} {d}: {P} patches; zhat mean {float(z.mean()):+.3f} std "
                  f"{float(z.std()):.3f} min {float(z.min()):+.2f} max {float(z.max()):+.2f} ({time.time() - t0:.0f} s)",
                  flush=True)
    arr = {k: np.concatenate(v) for k, v in out.items()}
    arr["meta"] = np.array(json.dumps({"run": run, "step": step, "split": split, "radii": list(radii), "n_pts": n_pts,
                                       "rk4_steps": rk4_steps, "noise": spec.__dict__, "seed": seed}))
    np.savez(out_path, **arr)
    print(f"[latents] {split}: {arr['zhat'].shape[0]} patches x {n_pts} points -> {out_path} ({time.time() - t0:.0f} s)",
          flush=True)
    return out_path


# ---------------------------------------------------------------------------------------------------------------
# fit and test
# ---------------------------------------------------------------------------------------------------------------

def load_latents(path, device):
    z = np.load(path)
    t = lambda k, dt=torch.float32: torch.from_numpy(np.asarray(z[k])).to(device=device, dtype=dt)
    return dict(lat=t("lat"), lon=t("lon"), zhat=t("zhat"), cond=t("cond"), static=t("static"),
                day=torch.from_numpy(np.asarray(z["day"])).to(device), radius=t("radius"),
                meta=json.loads(str(z["meta"])))


def mean_loglik(prior, L, idx, bs=256):
    out = []
    for a in range(0, len(idx), bs):
        i = idx[a:a + bs]
        out.append(prior.patch_loglik(L["zhat"][i], L["lat"][i], L["lon"][i], L["cond"][i], L["static"][i]))
    return torch.cat(out)


def fit_prior(prior, L, train_idx, val_idx, epochs=30, lr=1e-2, bs=128, log=print):
    params = [p for p in prior.parameters() if p.requires_grad]
    if not params:
        return {"epochs": 0}
    opt = torch.optim.Adam(params, lr=lr)
    best, best_state, hist = -1e9, None, []
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        perm = train_idx[torch.randperm(len(train_idx), generator=g).to(train_idx.device)]
        tr = 0.0
        for a in range(0, len(perm), bs):
            i = perm[a:a + bs]
            ll = prior.patch_loglik(L["zhat"][i], L["lat"][i], L["lon"][i], L["cond"][i], L["static"][i]).mean()
            opt.zero_grad()
            (-ll).backward()
            opt.step()
            tr += float(ll) * len(i)
        with torch.no_grad():
            va = float(mean_loglik(prior, L, val_idx).mean())
        hist.append((tr / len(perm), va))
        if va > best:
            best, best_state = va, {k: v.detach().clone() for k, v in prior.state_dict().items()}
        log(f"    epoch {ep}: train {tr / len(perm):.4f} val {va:.4f} nats/pt")
    prior.load_state_dict(best_state)
    return {"epochs": epochs, "best_val": best, "history": hist}


def block_bootstrap(d_patch, day_patch, n_boot=2000, seed=0):
    """mean of a per-patch quantity with a day-block bootstrap interval (days resampled with replacement)"""
    days = np.unique(day_patch)
    per_day = np.array([d_patch[day_patch == d].mean() for d in days])
    rng = np.random.default_rng(seed)
    bs = np.array([per_day[rng.integers(0, len(days), len(days))].mean() for _ in range(n_boot)])
    return float(per_day.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run_fit(dirp, device, epochs=30, n_features=1024, fit_file="latents_train.npz"):
    dirp = Path(dirp)
    L = load_latents(dirp / fit_file, device)
    beta = float(L["meta"]["noise"]["beta"])
    base_spec = NoiseSpec(**L["meta"]["noise"])
    masses = baseline_band_masses(base_spec)
    print(f"[fit] {L['zhat'].shape[0]} training patches; baseline band masses {np.round(masses, 4).tolist()} "
          f"({BAND_NAMES})", flush=True)
    days = L["day"].cpu().numpy()
    ud = np.unique(days)
    val_days = ud[::5]                                   # every 5th training day for early stopping
    tr_idx = torch.from_numpy(np.nonzero(~np.isin(days, val_days))[0]).to(device)
    va_idx = torch.from_numpy(np.nonzero(np.isin(days, val_days))[0]).to(device)
    priors, info = {}, {}
    for name, kind in (("P0", "iid"), ("P1", "none"), ("P1s", "fixed"), ("P1r", "static"), ("P3", "cond")):
        torch.manual_seed(0)
        pr = CopulaPrior(kind, beta, cond_dim=L["cond"].shape[-1], static_dim=L["static"].shape[-1],
                         init_masses=masses, n_features=n_features).to(device)
        if kind == "none":
            pr.logits.requires_grad_(False)
        t0 = time.time()
        print(f"  {name} ({kind})", flush=True)
        r = fit_prior(pr, L, tr_idx, va_idx, epochs=epochs if kind in ("fixed", "static", "cond") else 0,
                      lr=3e-2 if kind == "fixed" else 1e-2)
        with torch.no_grad():
            ll_tr = float(mean_loglik(pr, L, tr_idx).mean())
            ll_va = float(mean_loglik(pr, L, va_idx).mean())
            wmean = pr.weights(L["cond"][va_idx[:200]], L["static"][va_idx[:200]]).pow(2).mean((0, 1)).cpu().numpy()
        info[name] = dict(kind=kind, train_ll=ll_tr, val_ll=ll_va, mean_band_variance=wmean.round(4).tolist(),
                          seconds=time.time() - t0, **{k: v for k, v in r.items() if k != "history"})
        print(f"  {name}: train {ll_tr:.4f} val {ll_va:.4f} nats/pt; mean band variance {np.round(wmean, 3).tolist()}",
              flush=True)
        priors[name] = pr
    torch.save({n: {"kind": p.kind, "beta": p.beta, "cond_dim": L["cond"].shape[-1], "static_dim": L["static"].shape[-1],
                    "state": p.state_dict(), "bands": p.bands, "n_features": n_features, "init_masses": masses}
                for n, p in priors.items()}, dirp / "priors.pt")
    (dirp / "fit.json").write_text(json.dumps({"band_names": BAND_NAMES, "baseline_masses": masses.tolist(),
                                               "priors": info}, indent=1))
    return priors


def run_test(dirp, device, test_file="latents_hold.npz"):
    dirp = Path(dirp)
    L = load_latents(dirp / test_file, device)
    saved = torch.load(dirp / "priors.pt", map_location=device, weights_only=False)
    priors = {}
    for n, s in saved.items():
        pr = CopulaPrior(s["kind"], s["beta"], cond_dim=s["cond_dim"], static_dim=s["static_dim"],
                         init_masses=s["init_masses"], bands=s["bands"], n_features=s["n_features"]).to(device)
        pr.load_state_dict(s["state"])
        priors[n] = pr.eval()
    idx = torch.arange(L["zhat"].shape[0], device=device)
    with torch.no_grad():
        ll = {n: mean_loglik(p, L, idx).cpu().numpy() for n, p in priors.items()}
    day = L["day"].cpu().numpy()
    rad = L["radius"].cpu().numpy()
    res = {"n_patches": int(len(idx)), "days": int(len(np.unique(day))), "loglik_nats_per_point": {},
           "delta_vs_P1": {}, "delta_vs_P1_by_radius": {}}
    for n in priors:
        m, lo, hi = block_bootstrap(ll[n], day)
        res["loglik_nats_per_point"][n] = {"mean": m, "ci95": [lo, hi]}
        if n != "P1":
            d = ll[n] - ll["P1"]
            m, lo, hi = block_bootstrap(d, day)
            res["delta_vs_P1"][n] = {"mean": m, "ci95": [lo, hi], "frac_days_positive":
                                     float(np.mean([d[day == t].mean() > 0 for t in np.unique(day)]))}
            res["delta_vs_P1_by_radius"][n] = {}
            for R in np.unique(rad):
                s = rad == R
                m, lo, hi = block_bootstrap(d[s], day[s])
                res["delta_vs_P1_by_radius"][n][f"{R:g}km"] = {"mean": m, "ci95": [lo, hi]}
    if "P3" in ll and "P1r" in ll:                       # A1(a): P3 replaces P1r only if this interval excludes 0
        m, lo, hi = block_bootstrap(ll["P3"] - ll["P1r"], day)
        res["delta_P3_vs_P1r"] = {"mean": m, "ci95": [lo, hi]}
    # C0-type diagnostic: the latents' empirical correlation by distance band, by radius, vs P1's
    with torch.no_grad():
        p1 = priors["P1"]
        diag = {}
        for R in np.unique(rad):
            s = torch.from_numpy(np.nonzero(rad == R)[0]).to(device)[:400]
            lat, lon, z = L["lat"][s], L["lon"][s], L["zhat"][s].double()
            d = torch.cdist(sphere_xyz_km(lat, lon), sphere_xyz_km(lat, lon))
            w = p1.weights(L["cond"][s], L["static"][s])
            Rm = p1.corr_matrix(lat, lon, w)
            emp = z[..., :, None] * z[..., None, :]
            edges = [0.5, 2, 5, 10, 20, 50, 100, 200, 400, 700]
            row = {}
            for a, b in zip(edges[:-1], edges[1:]):
                m = (d > a) & (d <= b)
                if m.any():
                    row[f"{a:g}-{b:g}km"] = {"empirical": float(emp[m].mean()), "P1": float(Rm[m].mean()),
                                            "n_pairs": int(m.sum())}
            diag[f"{R:g}km"] = row
        res["latent_correlation_by_distance"] = diag
        res["latent_std"] = float(L["zhat"].std())
        res["latent_mean"] = float(L["zhat"].mean())
    (dirp / "test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "latent_correlation_by_distance"}, indent=1), flush=True)
    for R, row in diag.items():
        print(f"  latent correlation ({R} patches): " + ", ".join(f"{k}: emp {v['empirical']:+.3f} / P1 {v['P1']:+.3f}"
                                                               for k, v in row.items()), flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------
# sampling with a fitted prior (eval_x --prior-file priors.pt:NAME)
# ---------------------------------------------------------------------------------------------------------------

class SampledPrior:
    """draws, per day, M x K band fields; field(lat, lon, cond, static) -> (M, n, 1) = sum_m w_m xi_m"""

    def __init__(self, prior, M, device):
        self.prior, self.M, self.device = prior.eval(), M, device
        self.params = None

    def start_day(self, gen):
        self.params = [draw_sphere_noise(self.M, 1, s, self.device, gen) for s in self.prior.specs]

    @torch.no_grad()
    def field(self, lat, lon, cond, static):
        M, n = self.M, lat.shape[0]
        la, lo = lat[None].expand(M, -1), lon[None].expand(M, -1)
        xi = torch.stack([eval_sphere_noise(p, la, lo, chunk=noise_chunk(M, s))[..., 0]
                          for p, s in zip(self.params, self.prior.specs)], -1)          # (M, n, K)
        w = self.prior.weights(cond, static[None].expand(M, -1, -1))                   # (M, n, K)
        return (w * xi).sum(-1, keepdim=True)


def load_prior(spec_str, model, spec, M, device):
    path, name = spec_str.split(":")
    s = torch.load(path, map_location=device, weights_only=False)[name]
    pr = CopulaPrior(s["kind"], s["beta"], cond_dim=s["cond_dim"], static_dim=s["static_dim"],
                     init_masses=s["init_masses"], bands=s["bands"], n_features=s["n_features"]).to(device)
    pr.load_state_dict(s["state"])
    return SampledPrior(pr, M, device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["latents", "fit", "test"])
    ap.add_argument("--run", default="s2_europe")
    ap.add_argument("--runs-root", default=str(RUNS_ROOT))
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--dir", default=str(LOGS_X / "copula"))
    ap.add_argument("--splits", default="train,hold")
    ap.add_argument("--n-per-radius", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--day-idx", default=None, help="latents: comma-separated indices into the split's dates")
    ap.add_argument("--seed", type=int, default=0, help="latents: patch-centre / dequantisation seed (per rank when "
                    "the days are spread over several processes, runs_global/p1r_K0.sh)")
    ap.add_argument("--fit-file", default="latents_train.npz")
    ap.add_argument("--test-file", default="latents_hold.npz")
    a = ap.parse_args(argv)
    if a.device != "cpu" and torch.cuda.is_available():       # one process per GCD under srun (runs_global/p1r_K0.sh)
        dev = torch.device(f"cuda:{int(os.environ.get('SLURM_LOCALID', 0))}")
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")
    Path(a.dir).mkdir(parents=True, exist_ok=True)
    if a.cmd == "latents":
        for sp in a.splits.split(","):
            make_latents(a.run, a.runs_root, a.ckpt, sp, Path(a.dir) / f"latents_{sp}.npz", dev,
                         n_per_radius=a.n_per_radius, max_days=a.max_days, seed=a.seed,
                         day_idx=None if a.day_idx is None else [int(x) for x in a.day_idx.split(",")])
    elif a.cmd == "fit":
        run_fit(a.dir, dev, epochs=a.epochs, fit_file=a.fit_file)
    else:
        run_test(a.dir, dev, test_file=a.test_file)


if __name__ == "__main__":
    main()
