"""calib.core -- shared pieces of PROTOCOL.md A6 (see the package docstring).

The members of every A6 score are drawn as cdffm_aq_x.model_x.Sampler + copula_x.SampledPrior draw them for pmG_final
(eval_x, the real-time adapter): per day, generator gp seeded seed * 1_000_003 + t * 1009 + 41 draws the five band
fields of the prior (SampledPrior.start_day); z0 = sum_k w_k(c(p)) xi_k(p); |z0| <= z0_trunc; model_aq.integrate with
bound = z_bound; z_to_ug with pmax. Here the band fields xi are kept apart from the weights so that they can be cached
(they depend on the day, the member and the point only), and a temperature tau can multiply z0 before the truncation.
"""
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cdffm_aq.data_aq import (Scores, assemble_inputs, doy_of, fair_crps, raw_to_device, target_z, z_to_ug)
from cdffm_aq.model_aq import draw_sphere_noise, eval_sphere_noise, integrate, noise_chunk

W = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
RG = W / "runs_ghap"
CAL = RG / "pmG_cal"
FINAL = RG / "pmG_final"
BASE_EVAL = RG / "pmG_base" / "eval_sel2021_M8_s10_P3_guard_T3"
CACHE = CAL / "cache"
RESULTS = CAL / "results"
SEL_IDX = tuple(range(0, 346, 15))                                            # the 24 SEL days (hold-out indices)
V3ODD_IDX = tuple(3 * k for k in range(120) if k % 2 == 1 and (3 * k) % 15)   # the 48 P3 fit days (p1r_G.sh)
REGIONS = (("Europe", 35, 60, -10, 35), ("East Asia", 20, 50, 100, 135), ("South Asia", 5, 35, 60, 95),
           ("North America", 15, 50, -130, -60), ("Africa", -35, 35, -18, 52), ("South America", -40, 12, -82, -34))
REGION_NAMES = tuple(r[0] for r in REGIONS) + ("North", "other")             # first match; North = lat > 60 otherwise


def guard():
    """the guard of the frozen system: {"z_bound": 10.0, "pmax_ug": 1000.0, "z0_trunc": 3.0}"""
    s = json.loads((FINAL / "SAMPLER.json").read_text())
    return {"z_bound": float(s["z_bound"]), "pmax_ug": float(s["pmax_ug"]), "z0_trunc": float(s["z0_trunc"])}


def load_prior_module(device, path=None, name=None):
    """the fitted prior of pmG_final (copula_x.CopulaPrior, frozen, eval mode); as copula_x.load_prior without the sampler"""
    from cdffm_aq_x.copula_x import CopulaPrior
    pj = json.loads((FINAL / "PRIOR.json").read_text())
    path = Path(path) if path else FINAL / pj["file"]
    name = name or pj["name"]
    s = torch.load(path, map_location=device, weights_only=False)[name]
    pr = CopulaPrior(s["kind"], s["beta"], cond_dim=s["cond_dim"], static_dim=s["static_dim"],
                     init_masses=s["init_masses"], bands=s["bands"], n_features=s["n_features"]).to(device)
    pr.load_state_dict(s["state"])
    for p in pr.parameters():
        p.requires_grad_(False)
    return pr.eval()


@torch.no_grad()
def draw_prior_params(prior, M, device, gen):
    """SampledPrior.start_day: one set of sphere-noise parameters per band, drawn in band order from gen"""
    return [draw_sphere_noise(M, 1, s, device, gen) for s in prior.specs]


@torch.no_grad()
def band_fields(prior, params, lat, lon, M):
    """the K band fields of M members at the points lat, lon (n,) -> (M, n, K), as SampledPrior.field computes them"""
    la, lo = lat[None].expand(M, -1), lon[None].expand(M, -1)
    return torch.stack([eval_sphere_noise(p, la, lo, chunk=noise_chunk(M, s))[..., 0]
                        for p, s in zip(params, prior.specs)], -1)


@torch.no_grad()
def prior_z0(prior, xi, cond, static=None):
    """z0 (M, n, 1) = sum_k w_k xi_k with the weights of the pointwise condition cond (M, n, c) (SampledPrior.field)"""
    w = prior.weights(cond, static)
    return (w * xi).sum(-1, keepdim=True)


def temper(z0, tau, knee=None, trunc=3.0):
    """temperature on the prior field z0 (N(0, 1) marginals), applied BEFORE the sampler's unchanged truncation |z0| <= trunc.
    knee None: the plain temperature of PROTOCOL.md A6(e), tau z0 (the truncation then cuts the larger share beyond trunc).
    knee a: the tapered temperature of A6_ADDENDUM_tapered_temperature.md (post hoc): sign(z0) g(|z0|) with g(u) = tau u for
    u <= a and the straight line from (a, tau a) to (trunc, trunc) beyond: the centre is inflated, the values at the truncation
    bound stay where they are (beyond trunc the line continues above trunc and the truncation clamps it, as today)."""
    if tau == 1.0:
        return z0
    if knee is None:
        return z0 * tau
    ta = tau * knee
    if not (0.0 < knee < trunc and ta <= trunc + 1e-9):
        raise ValueError(f"tapered temperature: need 0 < knee < {trunc} and tau * knee <= {trunc} (tau {tau}, knee {knee})")
    u = z0.abs()
    g = torch.where(u <= knee, tau * u, ta + (trunc - ta) * (u - knee) / (trunc - knee))
    return torch.sign(z0) * g


def euler_members(model, z0, cond, n_steps, bound):
    """model_aq.integrate (Euler from t = 0 to 1, state kept in [-bound, bound] before every velocity evaluation and at
    the end) WITH the gradient: z0 (B, N, 1), cond (B, N, c) -> (B, N, 1)"""
    B, N = z0.shape[:2]
    h = 1.0 / n_steps
    x = z0.clamp(-bound, bound) if bound is not None else z0
    for s in range(n_steps):
        t = torch.full((B, N), s * h, device=x.device)
        x = x + model.velocity(x, cond, t) * h
        if bound is not None:
            x = x.clamp(-bound, bound)
    return x


def fair_crps_members(xs, y):
    """xs (B, M, N), y (B, N) -> scalar: mean over days and points of mean_m |x_m - y| - sum_{i<j} |x_i - x_j| / (M (M - 1))
    (Ferro 2014; for M = 2: (|x1 - y| + |x2 - y|) / 2 - |x1 - x2| / 2)"""
    M = xs.shape[1]
    e = (xs - y[:, None]).abs().mean(1)
    pair = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            pair = pair + (xs[:, i] - xs[:, j]).abs()
    return (e - pair / (M * (M - 1))).mean()


# ---------------------------------------------------------------------------------------------------------------
# validation sets
# ---------------------------------------------------------------------------------------------------------------

def region_index(lat, lon):
    """(n,) numpy degrees -> (n,) int index into REGION_NAMES (first match, then North, then other)"""
    out = np.full(lat.shape, len(REGION_NAMES) - 1, np.int64)
    done = np.zeros(lat.shape, bool)
    for i, (_, la0, la1, lo0, lo1) in enumerate(REGIONS):
        m = ~done & (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        out[m] = i
        done |= m
    m = ~done & (lat > 60)
    out[m] = len(REGIONS)
    return out


class ValSet:
    """a fixed set of (day, pixel) pairs with its truth, cached in cache/valset_<name>.npz.
    kind 'hold': hold-out days of 2021 (indices into store.hold_dates), one pixel list for all days, truth = the stored
                 integer k (-1 = hole, zero weight), weights cos(lat) (as eval_x).
    kind 'train': training days (indices into store.train_dates), per-day pixel lists taken from the stored points
                 (cos(lat)-weighted draws: unit weights), truth = the stored integer k."""

    def __init__(self, name, kind, days, dates, pix, truth):
        self.name, self.kind = name, kind
        self.days = [int(t) for t in days]
        self.dates = [dt.date.fromisoformat(str(d)) for d in dates]
        self.pix = np.asarray(pix, np.int64)            # (n,) hold | (n_days, n) train
        self.truth = np.asarray(truth, np.int16)        # (n_days, n)

    def save(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE / f"valset_{self.name}.tmp.npz"
        np.savez(tmp, kind=np.array(self.kind), days=np.array(self.days), dates=np.array([str(d) for d in self.dates]),
                 pix=self.pix, truth=self.truth)
        tmp.replace(CACHE / f"valset_{self.name}.npz")

    @staticmethod
    def load(name):
        z = np.load(CACHE / f"valset_{name}.npz")
        return ValSet(name, str(z["kind"]), z["days"], z["dates"], z["pix"], z["truth"])

    def seed_index(self, i):
        """the day index entering the noise seed (eval_x: the hold-out index; training days: 100000 + train index)"""
        return self.days[i] if self.kind == "hold" else 100000 + self.days[i]


def build_valsets(store, n_pix=100000, seed=0, log=print):
    """writes the five sets of A6: sel (V100k), fit (V100k pixels on the 48 V3-odd days), probe (hot-spot pixels on the
    SEL days), ins2020 / ins2018 (the SEL calendar days of 2020 / 2018 that are training days, 100,000 stored points)"""
    from cdffm_aq_x.data_x import CellGeometry
    from cdffm_aq.train_aq import model_grid
    t0 = time.time()
    g = np.random.default_rng(seed)
    v100k = np.sort(g.choice(store.n_pix, size=min(n_pix, store.n_pix), replace=False))     # = train_aq.ValData(seed 0)
    geom = CellGeometry(store, model_grid(store.name, "025"))
    ext = np.load(BASE_EVAL / "extremes.npz")["cells_pix_gt_440"]
    cells = np.nonzero(ext.reshape(-1) > 0)[0]
    probe = np.nonzero(np.isin(geom.cell_id, cells))[0]
    log(f"[valsets] V100k {v100k.size} pixels; hot-spot probe: {cells.size} cells, {probe.size} pixels, frozen-system "
        f"pixel-days above T_G in these cells {int(ext.sum())}")
    sel_truth, probe_truth = [], []
    for t in SEL_IDX:
        hv = store.holdout(t)
        sel_truth.append(hv[v100k])
        probe_truth.append(hv[probe])
    ValSet("sel", "hold", SEL_IDX, [store.hold_dates[t] for t in SEL_IDX], v100k, np.stack(sel_truth)).save()
    ValSet("probe", "hold", SEL_IDX, [store.hold_dates[t] for t in SEL_IDX], probe, np.stack(probe_truth)).save()
    log(f"[valsets] sel, probe written ({time.time() - t0:.0f} s)")
    fit_truth = [store.holdout(t)[v100k] for t in V3ODD_IDX]
    ValSet("fit", "hold", V3ODD_IDX, [store.hold_dates[t] for t in V3ODD_IDX], v100k, np.stack(fit_truth)).save()
    log(f"[valsets] fit written: {len(V3ODD_IDX)} days ({time.time() - t0:.0f} s)")
    tr_index = {d: i for i, d in enumerate(store.train_dates)}
    for year in (2020, 2018):
        days, pixs, vals = [], [], []
        for t in SEL_IDX:
            d0 = store.hold_dates[t]
            d = dt.date(year, d0.month, d0.day)
            if d not in tr_index:
                continue
            i = tr_index[d]
            idx, val = store.points(i)
            r = np.random.default_rng([seed, year, i])
            m = np.sort(r.choice(idx.shape[0], size=n_pix, replace=False))
            days.append(i)
            pixs.append(idx[m].astype(np.int64))
            vals.append(val[m])
        ValSet(f"ins{year}", "train", days, [store.train_dates[i] for i in days], np.stack(pixs), np.stack(vals)).save()
        log(f"[valsets] ins{year} written: {len(days)} training days ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------------------------------------------------------
# validation with the inference sampler: per-day sums
# ---------------------------------------------------------------------------------------------------------------

def stat_layout(M):
    """names of the per-day sums (one row per day)"""
    names = []
    for sp in ("ug", "log"):
        names += [f"{sp}/n", f"{sp}/seM", f"{sp}/var", f"{sp}/crps", f"{sp}/bias", f"{sp}/se1"]
        names += [f"{sp}/rank{i}" for i in range(M + 1)]
    names += ["z/n", "z/seM", "z/var", "z/crps", "z/bias"]
    names += ["san/n_pix", "san/n_pix_gt_TG", "san/n_member_at_pmax", "san/n_member_at_bound", "san/n_member_z0_trunc",
              "san/max_member_ug", "san/done"]
    for r in range(len(REGION_NAMES)):
        names += [f"reg{r}/n", f"reg{r}/seM", f"reg{r}/var", f"reg{r}/crps", f"reg{r}/bias"]
    return names


@torch.no_grad()
def validate_set(model, store, pix, vs, prior, device, rank=0, world=1, M=8, ode_steps=10, tau=1.0, seed=0,
                 cache=None, cond_frozen=False, grid="025", transport="t6", chunk=50000, g=None, tmax=None, knee=None):
    """per-day sums of the set's scores for the days of this rank (days[rank::world]; zeros elsewhere), members drawn as
    the inference sampler draws them (module docstring). cache: dict kept by the caller: band fields per (set, day),
    and the condition per (set, day) when cond_frozen (encoder and LIIF frozen: the condition does not change).
    Returns a (n_days, K) float64 tensor on the device (K = len(stat_layout(M)))."""
    model.eval()
    g = g or guard()
    zb, pmax, T = g["z_bound"], g["pmax_ug"], g["z0_trunc"]
    tmax = float(getattr(store, "train_max_ug", 986.0)) if tmax is None else tmax
    names = stat_layout(M)
    col = {n: i for i, n in enumerate(names)}
    out = torch.zeros(len(vs.days), len(names), dtype=torch.float64, device=device)
    cache = cache if cache is not None else {}
    for i in range(rank, len(vs.days), world):
        d = vs.dates[i]
        doy = torch.tensor([doy_of(d)], device=device)
        kk = torch.from_numpy(vs.pix if vs.kind == "hold" else vs.pix[i]).to(device)
        n = kk.numel()
        lat, lon, stat = pix.lat[kk], pix.lon[kk], pix.static[kk]
        mu, sig = pix.mu_sigma(kk[None], doy)                                        # (1, n)
        ck = (vs.name, i)
        cond = cache.get(("cond",) + ck) if cond_frozen else None
        if cond is None:
            raw = raw_to_device(store.raw_inputs(d), device)
            feat = model.encode(assemble_inputs(raw, pix, grid, transport))
            cond = torch.cat([model.condition(feat, lat[None, a:a + chunk], lon[None, a:a + chunk],
                                              stat[None, a:a + chunk], doy) for a in range(0, n, chunk)], 1)   # (1, n, c)
            del feat, raw
            if cond_frozen:
                cache[("cond",) + ck] = cond
        xi = cache.get(("xi",) + ck)
        if xi is None:
            gp = torch.Generator(device=device)
            gp.manual_seed(seed * 1_000_003 + vs.seed_index(i) * 1009 + 41)          # eval_x's prior generator
            params = draw_prior_params(prior, M, device, gp)
            xi = torch.cat([band_fields(prior, params, lat[a:a + chunk], lon[a:a + chunk], M)
                            for a in range(0, n, chunk)], 1)                         # (M, n, K)
            cache[("xi",) + ck] = xi
        condM = cond.expand(M, -1, -1)
        z0 = prior_z0(prior, xi, condM, stat[None].expand(M, -1, -1))
        z0 = temper(z0, tau, knee, T)
        n_trunc = int((z0.abs() > T).sum())
        z0 = z0.clamp(-T, T)
        z = torch.cat([integrate(model, z0[:, a:a + chunk], condM[:, a:a + chunk], ode_steps, bound=zb)
                       for a in range(0, n, chunk)], 1)[..., 0]                      # (M, n)
        n_bound = int((z.abs() >= zb).sum())
        ens = z_to_ug(z, mu, sig, pmax=pmax)
        tv = torch.from_numpy(vs.truth[i].astype(np.float32)).to(device)
        y = tv.clamp(min=1.0)
        if vs.kind == "hold":
            w = pix.w_area[kk] * (tv >= 0).float()
        else:
            w = torch.ones(n, device=device)
        sc = Scores(M, device)
        sc.add(ens, y, w)
        row = out[i]
        for sp in ("ug", "log"):
            for k in ("n", "seM", "var", "crps", "bias", "se1"):
                row[col[f"{sp}/{k}"]] = sc.s[f"{sp}/{k}"]
            row[col[f"{sp}/rank0"]:col[f"{sp}/rank0"] + M + 1] = sc.s[f"{sp}/rank"]
        wd = w.double()
        zt = target_z(y, mu[0], sig[0])                                              # evaluation truth in z (no dequantisation)
        zm = z.mean(0)
        row[col["z/n"]] = wd.sum()
        row[col["z/seM"]] = (wd * (zm - zt).double() ** 2).sum()
        row[col["z/var"]] = (wd * z.var(0, unbiased=True).double()).sum()
        row[col["z/crps"]] = (wd * fair_crps(z, zt).double()).sum()
        row[col["z/bias"]] = (wd * (zm - zt).double()).sum()
        emax = ens.max(0).values
        row[col["san/n_pix"]] = n
        row[col["san/n_pix_gt_TG"]] = (emax > tmax).sum()
        row[col["san/n_member_at_pmax"]] = (ens >= pmax).sum()
        row[col["san/n_member_at_bound"]] = n_bound
        row[col["san/n_member_z0_trunc"]] = n_trunc
        row[col["san/max_member_ug"]] = emax.max()
        row[col["san/done"]] = 1.0
        key = ("reg", vs.name) if vs.kind == "hold" else ("reg", vs.name, i)
        reg = cache.get(key)
        if reg is None:
            reg = torch.from_numpy(region_index(lat.cpu().numpy().astype(np.float64),
                                                lon.cpu().numpy().astype(np.float64))).to(device)
            cache[key] = reg
        m = ens.mean(0)
        per = {"n": wd, "seM": wd * (m - y).double() ** 2, "var": wd * ens.var(0, unbiased=True).double(),
               "crps": wd * fair_crps(ens, y).double(), "bias": wd * (m - y).double()}
        R = len(REGION_NAMES)
        for k, v in per.items():
            acc = torch.zeros(R, dtype=torch.float64, device=device).index_add_(0, reg, v)
            for r in range(R):
                row[col[f"reg{r}/{k}"]] = acc[r]
    return out


def reduce_rows(rows, world):
    """sum over the ranks (each day is filled by one rank only)"""
    if world > 1:
        import torch.distributed as dist
        dist.all_reduce(rows)
    return rows


# ---------------------------------------------------------------------------------------------------------------
# summaries (numpy; also used by the CPU scripts through plain lists)
# ---------------------------------------------------------------------------------------------------------------

def _pooled(S, M, sp):
    n = max(S[f"{sp}/n"], 1e-300)
    rmse = math.sqrt(S[f"{sp}/seM"] / n)
    V = S[f"{sp}/var"] / n
    r = {"n": S[f"{sp}/n"], "crps_fair": S[f"{sp}/crps"] / n, "rmse_ensmean": rmse, "bias": S[f"{sp}/bias"] / n,
         "member_variance": V, "spread_skill_fortin": math.sqrt((M + 1) / M * V) / max(rmse, 1e-12),
         "D_predictive_mean_sq_error": S[f"{sp}/seM"] / n - V / M}
    if f"{sp}/se1" in S:
        r["rmse_1member"] = math.sqrt(S[f"{sp}/se1"] / n)
    if f"{sp}/rank0" in S:
        rk = np.array([S[f"{sp}/rank{i}"] for i in range(M + 1)])
        r["rank_hist"] = (rk / max(rk.sum(), 1e-300)).tolist()
    return r


def summarise(rows, M, days=None):
    """rows: (n_days, K) array of per-day sums -> pooled summary dict (ug, log, z, sanity, regions) of the given days"""
    rows = np.asarray(rows, np.float64)
    names = stat_layout(M)
    if days is not None:
        rows = rows[np.asarray(days)]
    tot = rows.sum(0)
    S = dict(zip(names, tot.tolist()))
    out = {"n_days": int(rows.shape[0]), "ug": _pooled(S, M, "ug"), "log": _pooled(S, M, "log"), "z": _pooled(S, M, "z")}
    npix = max(S["san/n_pix"], 1.0)
    out["sanity"] = {"n_pix": S["san/n_pix"], "n_pix_gt_TG": S["san/n_pix_gt_TG"],
                     "n_member_at_pmax": S["san/n_member_at_pmax"], "n_member_at_bound": S["san/n_member_at_bound"],
                     "n_member_z0_trunc": S["san/n_member_z0_trunc"],
                     "max_member_ug": float(rows[:, names.index("san/max_member_ug")].max()) if rows.size else 0.0,
                     "frac_pix_days_any_member_gt_TG": S["san/n_pix_gt_TG"] / npix,
                     "frac_member_pixel_days_guard_bound": (S["san/n_member_at_bound"] + S["san/n_member_at_pmax"]) / (npix * M),
                     "days_done": int(S["san/done"])}
    out["regions"] = {}
    for r, rn in enumerate(REGION_NAMES):
        Sr = {f"ug/{k}": S[f"reg{r}/{k}"] for k in ("n", "seM", "var", "crps", "bias")}
        if Sr["ug/n"] > 0:
            p = _pooled(Sr, M, "ug")
            out["regions"][rn] = {k: p[k] for k in ("n", "crps_fair", "rmse_ensmean", "bias", "spread_skill_fortin")}
    return out


def daily_table(rows, M, vs):
    """per-day ug scores (for the logs and the day bootstrap)"""
    names = stat_layout(M)
    out = []
    for i in range(len(vs.days)):
        S = dict(zip(names, np.asarray(rows[i], np.float64).tolist()))
        if S["san/done"] < 0.5:
            continue
        p = _pooled(S, M, "ug")
        out.append({"date": str(vs.dates[i]), "crps_fair": p["crps_fair"], "rmse_ensmean": p["rmse_ensmean"],
                    "bias": p["bias"], "spread_skill_fortin": p["spread_skill_fortin"],
                    "n_pix_gt_TG": S["san/n_pix_gt_TG"], "max_member_ug": S["san/max_member_ug"]})
    return out


def boot_pooled(rows, M, sp="ug", n_boot=5000, seed=0, ref=None):
    """day bootstrap (days resampled with replacement) of the pooled crps / rmse / spread-skill of rows, and, with ref
    (rows of the reference system on the same days), of the paired differences rows - ref -> dict of [lo, hi]"""
    names = stat_layout(M)
    ix = {k: names.index(f"{sp}/{k}") for k in ("n", "seM", "var", "crps")}

    def stats(a):
        n = a[..., ix["n"]]
        rm = np.sqrt(a[..., ix["seM"]] / n)
        return {"crps_fair": a[..., ix["crps"]] / n, "rmse_ensmean": rm,
                "spread_skill_fortin": np.sqrt((M + 1) / M * a[..., ix["var"]] / n) / rm}
    rows = np.asarray(rows, np.float64)
    nd = rows.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nd, size=(n_boot, nd))
    bs = stats(rows[idx].sum(1))
    out = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    if ref is not None:
        ref = np.asarray(ref, np.float64)
        br = stats(ref[idx].sum(1))
        out["diff"] = {k: [float(np.percentile(bs[k] - br[k], 2.5)), float(np.percentile(bs[k] - br[k], 97.5))] for k in bs}
        out["rel_diff"] = {k: [float(np.percentile(bs[k] / br[k] - 1, 2.5)), float(np.percentile(bs[k] / br[k] - 1, 97.5))]
                           for k in bs}
    return out
