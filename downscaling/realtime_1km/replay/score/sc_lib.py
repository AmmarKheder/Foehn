#!/usr/bin/env python3
"""WP1b scoring library: station-day table, block bootstrap, scores. Evaluation only; nothing is fitted here.

BLOCK BOOTSTRAP (class Sample), used for every interval unless a table says otherwise:
  * space blocks = 2-degree cells, cell = (floor(lat/2)+90)*1000 + floor(lon/2)+180 (as rc/live_baselines.py);
  * time blocks = ISO calendar weeks (Monday-Sunday) of the VALID day; a week with fewer than 4 valid days in the sample
    is merged with the neighbouring week of the same run of consecutive weeks (function time_blocks);
  * one resample draws C cells with replacement among the C cells of the sample (multiplicities m_c) and, independently,
    W weeks with replacement among its W weeks (multiplicities m_w); every station-day of cell c and week w then enters
    the statistic with weight m_c * m_w (two-way "pigeonhole" block bootstrap: whole cells and whole weeks are resampled,
    so the dependence between days at a monitor, between monitors of a cell, between lead days of one valid day and
    between monitors on the same day is kept inside the blocks);
  * the statistic is recomputed from the weighted sums (ratio of sums of squared errors, of CRPS, counts of a contingency
    table, ...); the interval is the 2.5th-97.5th percentile of B resamples (B = 2000 unless stated); the same draws are
    used for every arm and every group of a table, so contrasts are paired;
  * "mean over regions" = unweighted mean over the catalogue's regions of the regional statistic, recomputed inside each
    resample with the same draws.
  With mode "c" only cells are resampled (the convention of the first live comparison), with mode "w" only weeks."""
import datetime as dt

import numpy as np

D0 = dt.date(2026, 1, 1)
REGIONS = ["Europe", "East Asia", "South Asia", "North America", "Africa", "South America", "North", "Other"]


def doy0(s):
    return (dt.date.fromisoformat(s) - D0).days


def day_of(k):
    return (D0 + dt.timedelta(int(k))).isoformat()


class Data:
    """the station-day table written by build_pairs.py"""

    def __init__(self, path):
        z = np.load(path)
        self.issue_days = [str(x) for x in z["issue_days"]]
        self.arms = [str(x) for x in z["arms"]]
        self.kinds = dict(zip(self.arms, [str(x) for x in z["kinds"]]))
        self.id, self.lat, self.lon = z["id"].astype(str), z["lat"].astype(np.float64), z["lon"].astype(np.float64)
        self.region_name = z["region"].astype(str)
        self.reg_of_mon = np.array([REGIONS.index(r) for r in self.region_name], np.int64)
        self.cell_of_mon = z["cell"].astype(np.int64)
        self.issue = z["issue"].astype(np.int64)
        self.lead = z["lead"].astype(np.int64)
        self.mon = z["mon"].astype(np.int64)
        self.vday = z["vday"].astype(np.int64)
        self.obs = z["obs"].astype(np.float64)
        self.src = z["src"].astype(np.int64)
        self.iday = np.array([doy0(s) for s in self.issue_days], np.int64)[self.issue]       # issue day (days since 1 Jan) per row
        self.m = {a: z[f"m_{a}"].astype(np.float64) for a in self.arms}
        self.e = {a: z[f"e_{a}"] for a in self.arms if self.kinds[a] == "ens"}
        self.N = self.obs.size
        self.reg = self.reg_of_mon[self.mon]
        self._ens = {}

    def finite(self, arms):
        ok = np.isfinite(self.obs)
        for a in arms:
            ok &= np.isfinite(self.m[a])
        return ok

    def issues(self, first, last):
        return (self.iday >= doy0(first)) & (self.iday <= doy0(last))

    def ens_stats(self, a):
        """per-row ensemble statistics of arm a (cached): fair CRPS, unbiased member variance, p10/p90 (linear quantiles,
        as the production products), rank of the observation, exceedance fractions"""
        if a in self._ens:
            return self._ens[a]
        x = np.sort(self.e[a].astype(np.float64), axis=1)
        y = self.obs
        M = x.shape[1]
        k = np.arange(1, M + 1)
        crps = np.abs(x - y[:, None]).mean(1) - (x * (2 * k - M - 1)).sum(1) / (M * (M - 1))
        var = x.var(1, ddof=1)

        def q(p):
            pos = p * (M - 1)
            lo = int(np.floor(pos))
            fr = pos - lo
            return x[:, lo] * (1 - fr) + x[:, min(lo + 1, M - 1)] * fr
        p10, p90 = q(0.1), q(0.9)
        rank = (x < y[:, None]).sum(1)
        out = dict(crps=crps, var=var, p10=p10, p90=p90, inside=((y >= p10) & (y <= p90)).astype(np.float64),
                   below=(y < p10).astype(np.float64), above=(y > p90).astype(np.float64), rank=rank, M=M,
                   width=p90 - p10)
        self._ens[a] = out
        return out

    def pexc(self, a, t):
        if self.kinds[a] == "ens":
            return (self.e[a] > t).mean(1).astype(np.float64)
        return (self.m[a] > t).astype(np.float64)

    def crps(self, a):
        """fair CRPS of an ensemble arm, absolute error of a deterministic arm"""
        if self.kinds[a] == "ens":
            return self.ens_stats(a)["crps"]
        return np.abs(self.m[a] - self.obs)


def time_blocks(vdays):
    """block index of each valid day (array of days since 1 Jan) and the labels of the blocks"""
    u = np.unique(vdays)
    dates = [D0 + dt.timedelta(int(x)) for x in u]
    wk = [tuple(d.isocalendar())[:2] for d in dates]
    weeks = sorted(set(wk))
    cnt = {w: wk.count(w) for w in weeks}
    monday = {w: dt.date.fromisocalendar(w[0], w[1], 1) for w in weeks}
    runs = []
    for w in weeks:
        if runs and (monday[w] - monday[runs[-1][-1]]).days == 7:
            runs[-1].append(w)
        else:
            runs.append([w])
    block_of_week, labels = {}, []
    for run in runs:
        groups = [[w] for w in run]
        changed = True
        while changed and len(groups) > 1:
            changed = False
            for i, g in enumerate(groups):
                if sum(cnt[w] for w in g) < 4:
                    if i == 0:
                        groups[1] = g + groups[1]
                    else:
                        groups[i - 1] = groups[i - 1] + g
                    del groups[i]
                    changed = True
                    break
        for g in groups:
            b = len(labels)
            for w in g:
                block_of_week[w] = b
            inside = [d for d, ww in zip(dates, wk) if ww in g]
            labels.append(f"{min(inside)}..{max(inside)} ({len(inside)} valid days)")
    lut = np.full(int(u.max()) + 1, -1, np.int64)
    lut[u] = [block_of_week[w] for w in wk]
    return lut[vdays], labels


class Sample:
    """a set of station-days with its bootstrap draws; groups default to the catalogue's regions"""

    def __init__(self, D, mask, group=None, gnames=None, B=2000, seed=20261119):
        self.D = D
        self.rows = np.flatnonzero(mask)
        self.n = self.rows.size
        mon = D.mon[self.rows]
        self.lead = D.lead[self.rows] - 1
        self.g = D.reg[self.rows] if group is None else group[self.rows]
        self.gnames = REGIONS if gnames is None else gnames
        self.G = len(self.gnames)
        ucell, self.ci = np.unique(D.cell_of_mon[mon], return_inverse=True)
        self.C = ucell.size
        ukey, self.u = np.unique(self.g * self.C + self.ci, return_inverse=True)
        self.U = ukey.size
        self.unit_cell, self.unit_g = ukey % self.C, ukey // self.C
        self.w, self.blocks = time_blocks(D.vday[self.rows])
        self.W = len(self.blocks)
        rng = np.random.default_rng(seed)
        self.B = B
        self.Mc = rng.multinomial(self.C, np.full(self.C, 1.0 / self.C), size=B).astype(np.float64)
        self.Mw = rng.multinomial(self.W, np.full(self.W, 1.0 / self.W), size=B).astype(np.float64)
        self.Ind = np.zeros((self.U, self.G))
        self.Ind[np.arange(self.U), self.unit_g] = 1.0
        self.key = (self.lead * self.U + self.u) * self.W + self.w
        self.n_issue = int(np.unique(D.issue[self.rows]).size)
        self.n_mon = int(np.unique(mon).size)
        iss = np.unique(D.issue[self.rows])
        self.issue_list = [D.issue_days[i] for i in iss]
        self._cache = {}

    def describe(self):
        d = self.issue_list
        runs, start, prev = [], d[0], d[0]
        for x in d[1:]:
            if (dt.date.fromisoformat(x) - dt.date.fromisoformat(prev)).days != 1:
                runs.append(f"{start}..{prev}")
                start = x
            prev = x
        runs.append(f"{start}..{prev}")
        return (f"{self.n_issue} issue days ({', '.join(runs)}), {self.n:,} station-days, {self.n_mon:,} monitors, "
                f"{self.C:,} 2-degree cells, {self.W} week blocks")

    def col(self, x):
        return x[self.rows]

    def sums(self, q):
        """q: one value per row of the sample -> (5 leads, U units, W weeks)"""
        return np.bincount(self.key, weights=q, minlength=5 * self.U * self.W).reshape(5, self.U, self.W)

    def boot(self, S, mode="cw"):
        """S (5, U, W) -> point sums (5, G) and resampled sums (B, 5, G)"""
        point = np.einsum("luw,ug->lg", S, self.Ind)
        if mode in ("cw", "w"):
            Z = (S.reshape(5 * self.U, self.W) @ self.Mw.T).T.reshape(self.B, 5, self.U)
        else:
            Z = np.broadcast_to(S.sum(2)[None], (self.B, 5, self.U))
        if mode in ("cw", "c"):
            Z = Z * self.Mc[:, self.unit_cell][:, None, :]
        return point, Z @ self.Ind

    def q(self, name, values_fn, mode="cw"):
        """cached (point, resampled) sums of a per-row quantity; values_fn() returns the quantity on the sample rows"""
        k = (name, mode)
        if k not in self._cache:
            self._cache[k] = self.boot(self.sums(values_fn()), mode)
        return self._cache[k]


# ---- aggregation of (.., 5 leads, G groups) sums ------------------------------------------------------------------------
def views(P):
    """dict of aggregations of an array whose last two axes are (lead, group): key (lead_key, group_key) with
    lead_key in 'all', 1..5 and group_key in 'all', 0..G-1"""
    out = {}
    G = P.shape[-1]
    tot_l = P.sum(-1)
    tot_g = P.sum(-2)
    out[("all", "all")] = tot_g.sum(-1)
    for g in range(G):
        out[("all", g)] = tot_g[..., g]
    for L in range(5):
        out[(L + 1, "all")] = tot_l[..., L]
        for g in range(G):
            out[(L + 1, g)] = P[..., L, g]
    return out


def rel_rmse(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (np.sqrt(a / b) - 1.0)


def rel_mean(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (a / b - 1.0)


def pct(x):
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nan)
    if not np.isfinite(x).any():
        return [float("nan"), float("nan")]
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def ratio_contrast(sample, qa, qb, cnt, f=rel_rmse, mode="cw", min_n=100):
    """relative change of arm a against arm b from cached sums; returns {lead_key: {group_key: dict(est, ci, n)}} with the
    group keys 'all' (pooled), 'regmean' (mean over groups of the group statistic) and each group name"""
    (pa, Ra), (pb, Rb), (pn, _) = qa, qb, cnt
    va, vb, vn, wa, wb = views(pa), views(pb), views(pn), views(Ra), views(Rb)
    out = {}
    for L in ["all", 1, 2, 3, 4, 5]:
        o = {}
        o["all"] = dict(est=float(f(va[(L, "all")], vb[(L, "all")])), ci=pct(f(wa[(L, "all")], wb[(L, "all")])), n=int(vn[(L, "all")]))
        gs = [g for g in range(sample.G) if vn[(L, g)] >= min_n]
        for g in gs:
            o[sample.gnames[g]] = dict(est=float(f(va[(L, g)], vb[(L, g)])), ci=pct(f(wa[(L, g)], wb[(L, g)])), n=int(vn[(L, g)]))
        if gs:
            est = float(np.mean([o[sample.gnames[g]]["est"] for g in gs]))
            bs = np.nanmean(np.stack([f(wa[(L, g)], wb[(L, g)]) for g in gs]), axis=0)
            o["regmean"] = dict(est=est, ci=pct(bs), n=int(sum(vn[(L, g)] for g in gs)), n_groups=len(gs),
                                n_negative=int(sum(o[sample.gnames[g]]["est"] < 0 for g in gs)),
                                n_ci_below_zero=int(sum(o[sample.gnames[g]]["ci"][1] < 0 for g in gs)),
                                n_ci_above_zero=int(sum(o[sample.gnames[g]]["ci"][0] > 0 for g in gs)))
        out[str(L)] = o
    return out


def basic_scores(D, rows, a, group, G):
    """n, RMSE, MAE, bias, Pearson r, mean obs, mean forecast of arm a by (lead, group) and aggregated"""
    x, y = D.m[a][rows], D.obs[rows]
    key = (D.lead[rows] - 1) * G + group
    def s(q):
        return np.bincount(key, weights=q, minlength=5 * G).reshape(5, G)
    e = x - y
    P = dict(n=s(np.ones(x.size)), e=s(e), ae=s(np.abs(e)), se=s(e * e), x=s(x), y=s(y), xx=s(x * x), yy=s(y * y), xy=s(x * y))
    V = {k: views(v) for k, v in P.items()}
    out = {}
    for key2 in V["n"]:
        n = float(V["n"][key2])
        if n < 1:
            continue
        mx, my = V["x"][key2] / n, V["y"][key2] / n
        vx, vy = V["xx"][key2] / n - mx * mx, V["yy"][key2] / n - my * my
        r = (V["xy"][key2] / n - mx * my) / np.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
        out[key2] = dict(n=int(n), rmse=float(np.sqrt(V["se"][key2] / n)), mae=float(V["ae"][key2] / n), bias=float(V["e"][key2] / n),
                         r=float(r), obs_mean=float(my), fc_mean=float(mx))
    return out


def sedi(a, b, c, d):
    with np.errstate(divide="ignore", invalid="ignore"):
        H, F = a / (a + c), b / (b + d)
        num = np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)
        den = np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H)
        s = num / den
    return np.where(np.isfinite(s), s, np.nan)


def group_percentile(v, s, ns, p):
    """p-th percentile (linear interpolation) of v inside each group s (0..ns-1); NaN for empty groups"""
    order = np.lexsort((v, s))
    vs = v[order]
    n = np.bincount(s, minlength=ns)
    start = np.cumsum(n) - n
    pos = np.maximum(n - 1, 0) * p / 100.0
    lo = np.floor(pos).astype(np.int64)
    fr = pos - lo
    hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
    ok = n > 0
    q = np.full(ns, np.nan)
    q[ok] = vs[start[ok] + lo[ok]] * (1 - fr[ok]) + vs[start[ok] + hi[ok]] * fr[ok]
    return q, n


def wmedian(V, Wt):
    """weighted median along axis 1 of V (nb, n) with weights Wt (nb, n); NaN values ignored"""
    bad = ~np.isfinite(V)
    Wt = np.where(bad, 0.0, Wt)
    Vs = np.where(bad, np.inf, V)
    order = np.argsort(Vs, axis=1)
    cw = np.cumsum(np.take_along_axis(Wt, order, 1), axis=1)
    tot = cw[:, -1]
    idx = (cw >= (tot / 2.0)[:, None]).argmax(1)
    med = np.take_along_axis(Vs, order, 1)[np.arange(V.shape[0]), idx]
    return np.where(tot > 0, med, np.nan)


def fmt(x, nd=1, sign=False):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def fci(d, nd=1):
    """'-12.3 [-15.0, -9.1]' from dict(est, ci)"""
    if d is None or not np.isfinite(d["est"]):
        return "n/a"
    return f"{d['est']:+.{nd}f} [{d['ci'][0]:+.{nd}f}, {d['ci'][1]:+.{nd}f}]"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"
