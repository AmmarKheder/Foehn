"""cdffm_aq_x.data_x -- data-side additions for the variants.

CellGeometry  nearest encoder-grid point of every mask pixel (grid/cell_id.npy of the smoke store; computed for the
              toy store), the sorted list of mask cells, the cell mask on the grid and its bounding box (+ margin).
cell means    per training day (from the stored 2 M points) and per hold-out day (exact, from the full fields): the
              0.25 deg cell-mean field of z, the target of the hierarchical coarse flow (variant hcf). Built once by
              build_cellmeans() (CPU) into one .npz; CellMeans serves them on the device as (B, 1, H, W) grids.
BlockScores   fair CRPS / RMSE / spread-skill of BLOCK MEANS of the members at 1, 2 and 4 cells (0.25, 0.5, 1 deg):
              joint-structure scores that pointwise scores cannot see (a pointwise flow has a fixed copula, so its
              block-mean spread is a fixed function of its pixel spread, whatever the day). Blocks = (r // s, c // s)
              of the pixel's nearest encoder cell; chunk sums are accumulated per block, the day is closed by end_day.
"""
import math

import numpy as np
import torch

from cdffm_aq.data_aq import PixelArrays, doy_of, fair_crps, target_z


class CellGeometry:
    def __init__(self, store, grid, margin=8):
        H, W = grid.H, grid.W
        cid = None
        root = getattr(store, "root", None)
        if root is not None and (root / "grid" / "cell_id.npy").exists():
            cid = np.load(root / "grid" / "cell_id.npy").astype(np.int64)
        if cid is None or cid.shape[0] != store.n_pix:
            r = np.rint((grid.lat0 - store.lat) / grid.dlat).astype(np.int64).clip(0, H - 1)
            c = np.rint((store.lon - grid.lon0) / grid.dlon).astype(np.int64) % W
            cid = r * W + c
        self.H, self.W = H, W
        self.cell_id = cid                                    # (n_pix,) flat r * W + c
        self.cells = np.unique(cid)                           # sorted flat ids of the mask cells
        self.n_cells = self.cells.size
        self.cell_pos = np.full(H * W, -1, np.int64)          # flat id -> position in self.cells (or -1)
        self.cell_pos[self.cells] = np.arange(self.n_cells)
        mg = np.zeros(H * W, bool)
        mg[self.cells] = True
        self.mask_grid = mg.reshape(H, W)
        r, c = self.cells // W, self.cells % W
        self.box = (max(0, int(r.min()) - margin), min(H, int(r.max()) + margin + 1),
                    max(0, int(c.min()) - margin), min(W, int(c.max()) + margin + 1))   # r0, r1, c0, c1 (c may wrap? no)
        self.grid = grid

    def box_points(self):
        """flat indices, lat and lon (degrees) of the grid points of the bounding box"""
        r0, r1, c0, c1 = self.box
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        flat = (rr * self.W + cc).ravel()
        g = self.grid
        return flat, g.lat0 - g.dlat * rr.ravel(), g.lon0 + g.dlon * cc.ravel()


def _fill_missing(grid_vals, mask_grid):
    """NaN cells inside the mask -> mean of the finite 3 x 3 neighbours (periodic in longitude), else 0"""
    v = grid_vals.copy()
    miss = mask_grid & ~np.isfinite(v)
    if miss.any():
        s = np.zeros_like(v)
        n = np.zeros_like(v)
        fin = np.isfinite(v)
        vz = np.where(fin, v, 0.0)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                s += np.roll(np.roll(vz, dr, 0), dc, 1)
                n += np.roll(np.roll(fin.astype(v.dtype), dr, 0), dc, 1)
        v[miss] = np.where(n[miss] > 0, s[miss] / np.maximum(n[miss], 1), 0.0)
    v[~mask_grid] = 0.0
    v[~np.isfinite(v)] = 0.0
    return v


def build_cellmeans(store, geom, out_path, chunk=2_000_000, verbose=True):
    """cell means of z on the training days (stored points) and hold-out days (full fields) -> .npz"""
    import time
    t0 = time.time()
    pix = PixelArrays(store, torch.device("cpu"))
    H, W, nc = geom.H, geom.W, geom.n_cells
    cp_all = torch.from_numpy(geom.cell_pos[geom.cell_id])            # (n_pix,) position of each pixel's cell

    def means_of(k, v, doy):
        z = torch.empty(k.shape[0], dtype=torch.float64)
        for a in range(0, k.shape[0], chunk):
            e = min(k.shape[0], a + chunk)
            kk = k[a:e]
            mu, sig = pix.mu_sigma(kk[None], torch.tensor([doy]))
            z[a:e] = target_z(v[a:e][None], mu, sig)[0].double()
        cp = cp_all[k]
        s = torch.zeros(nc, dtype=torch.float64).index_add_(0, cp, z)
        n = torch.zeros(nc, dtype=torch.float64).index_add_(0, cp, torch.ones_like(z))
        return s.numpy(), n.numpy()

    train = np.zeros((len(store.train_dates), nc), np.float32)
    npts = np.zeros((len(store.train_dates), nc), np.int32)
    for t, d in enumerate(store.train_dates):
        idx, val = store.points(t)
        s, n = means_of(torch.from_numpy(idx.astype(np.int64)), torch.from_numpy(val), doy_of(d))
        g = np.full(H * W, np.nan)
        g[geom.cells] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        g = _fill_missing(g.reshape(H, W), geom.mask_grid).ravel()
        train[t] = g[geom.cells]
        npts[t] = n
        if verbose and t % 25 == 0:
            print(f"  cell means: training day {t}/{len(store.train_dates)} ({time.time() - t0:.0f} s)", flush=True)
    hold = np.zeros((len(store.hold_dates), nc), np.float32)
    k_all = torch.arange(store.n_pix)
    for t, d in enumerate(store.hold_dates):
        s, n = means_of(k_all, torch.from_numpy(store.holdout(t)), doy_of(d))
        hold[t] = s / np.maximum(n, 1)
    np.savez(out_path, cells=geom.cells, train=train, hold=hold, npts=npts, H=H, W=W,
             train_dates=np.array([str(d) for d in store.train_dates]),
             hold_dates=np.array([str(d) for d in store.hold_dates]))
    if verbose:
        print(f"  cell means written: {out_path} ({nc} cells, {train.shape[0]} + {hold.shape[0]} days, "
              f"min pts/cell/day {npts.min()}, median {np.median(npts):.0f}; {time.time() - t0:.0f} s)", flush=True)
    return out_path


class CellMeans:
    """cell-mean grids on the device; mask (1, 1, H, W) and the sphere-noise points of the bounding box"""

    def __init__(self, geom, device, path=None, table=None, load_hold=True):
        self.geom, self.device = geom, device
        H, W = geom.H, geom.W
        self.cells = torch.from_numpy(geom.cells).to(device)
        m = torch.zeros(H * W, device=device)
        m[self.cells] = 1.0
        self.mask = m.view(1, 1, H, W)
        flat, lat, lon = geom.box_points()
        self.box_flat = torch.from_numpy(flat).to(device)
        self.box_lat = torch.from_numpy(lat).float().to(device)
        self.box_lon = torch.from_numpy(lon).float().to(device)
        self.train = self.hold = None
        if table is not None or path is not None:
            tb = table if table is not None else np.load(path)
            assert np.array_equal(tb["cells"], geom.cells) and int(tb["H"]) == H and int(tb["W"]) == W
            self.train = torch.from_numpy(np.asarray(tb["train"])).to(device)
            if load_hold:                 # hold-out cell means: diagnostics only, never a training input
                self.hold = torch.from_numpy(np.asarray(tb["hold"])).to(device)

    def _grid(self, rows):
        B = rows.shape[0]
        H, W = self.geom.H, self.geom.W
        g = torch.zeros(B, H * W, device=self.device)
        g[:, self.cells] = rows
        return g.view(B, 1, H, W)

    def train_grid(self, days):
        return self._grid(self.train[torch.as_tensor(days, device=self.device).long()])

    def hold_grid(self, days):
        return self._grid(self.hold[torch.as_tensor(days, device=self.device).long()])

    def grid_noise(self, params, chunk=None):
        """sphere-noise parameters (B fields, n_out = 1) evaluated at the bounding-box grid points -> (B, 1, H, W),
        zero outside the box"""
        from cdffm_aq.model_aq import eval_sphere_noise
        B = params["a"].shape[0]
        H, W = self.geom.H, self.geom.W
        nz = eval_sphere_noise(params, self.box_lat[None].expand(B, -1), self.box_lon[None].expand(B, -1),
                               chunk=chunk)[..., 0]                                             # (B, nb)
        g = torch.zeros(B, H * W, device=self.device)
        g[:, self.box_flat] = nz
        return g.view(B, 1, H, W)


class BlockScores:
    SPACES = ("ug", "log")

    def __init__(self, M, geom, device, scales=(1, 2, 4)):
        self.M, self.scales, self.device = M, tuple(scales), device
        H, W = geom.H, geom.W
        cid = torch.from_numpy(geom.cell_id).to(device)
        r, c = cid // W, cid % W
        self.bidx, self.nb = {}, {}
        for s in self.scales:
            self.nb[s] = (H // s) * (W // s)
            self.bidx[s] = (torch.div(r, s, rounding_mode="floor") * (W // s) + torch.div(c, s, rounding_mode="floor"))
        self.tot = {s: {sp: {k: 0.0 for k in ("n", "crps", "seM", "ae", "bias", "var", "spread")}
                        for sp in self.SPACES} for s in self.scales}
        self.daily = []
        self._reset_day()

    def _reset_day(self):
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float64, device=self.device)
        self.day = {s: {"w": z(self.nb[s]), "ug": {"wy": z(self.nb[s]), "wx": z(self.nb[s], self.M)},
                        "log": {"wy": z(self.nb[s]), "wx": z(self.nb[s], self.M)}} for s in self.scales}

    def add(self, kk, ens, y, w):
        """kk (n,) pixel indices, ens (M, n) ug m-3, y (n,) ug m-3, w (n,) weights"""
        w = w.double()
        for s in self.scales:
            b = self.bidx[s][kk]
            d = self.day[s]
            d["w"].index_add_(0, b, w)
            for sp in self.SPACES:
                X, Y = (ens, y) if sp == "ug" else (torch.log1p(ens), torch.log1p(y))
                d[sp]["wy"].index_add_(0, b, w * Y.double())
                d[sp]["wx"].index_add_(0, b, (w[None] * X.double()).T.contiguous())

    def end_day(self, date=None):
        row = {"date": str(date)}
        for s in self.scales:
            d = self.day[s]
            sel = d["w"] > 0
            w = d["w"][sel]
            row[s] = {}
            for sp in self.SPACES:
                ybar = d[sp]["wy"][sel] / w
                X = (d[sp]["wx"][sel] / w[:, None]).T                           # (M, nblocks)
                m = X.mean(0)
                crps = (w * fair_crps(X, ybar)).sum()
                seM = (w * (m - ybar) ** 2).sum()
                ae = (w * (m - ybar).abs()).sum()
                bias = (w * (m - ybar)).sum()
                var = (w * X.var(0, unbiased=True)).sum() if self.M > 1 else torch.zeros((), dtype=w.dtype)
                spread = (w * X.std(0, unbiased=False)).sum() if self.M > 1 else torch.zeros((), dtype=w.dtype)
                n = w.sum()
                t = self.tot[s][sp]
                for k, v in (("n", n), ("crps", crps), ("seM", seM), ("ae", ae), ("bias", bias), ("var", var),
                             ("spread", spread)):
                    t[k] += float(v)
                r = {"n_blocks": int(sel.sum()), "crps_fair": float(crps / n), "rmse_ensmean": math.sqrt(float(seM / n)),
                     "bias": float(bias / n)}
                if self.M > 1:
                    r["spread_skill_fortin"] = math.sqrt((self.M + 1) / self.M * float(var / n)) / max(r["rmse_ensmean"], 1e-12)
                row[s][sp] = r
        self.daily.append(row)
        self._reset_day()
        return row

    def all_reduce(self):
        import torch.distributed as dist
        keys = [(s, sp, k) for s in self.scales for sp in self.SPACES for k in self.tot[s][sp]]
        v = torch.tensor([self.tot[s][sp][k] for s, sp, k in keys], dtype=torch.float64, device=self.device)
        dist.all_reduce(v)
        for (s, sp, k), x in zip(keys, v.tolist()):
            self.tot[s][sp][k] = x

    def summary(self):
        out = {}
        for s in self.scales:
            out[str(s)] = {}
            for sp in self.SPACES:
                t = self.tot[s][sp]
                n = max(t["n"], 1e-300)
                rmse = math.sqrt(t["seM"] / n)
                r = {"n": t["n"], "crps_fair": t["crps"] / n, "rmse_ensmean": rmse, "mae_ensmean": t["ae"] / n,
                     "bias": t["bias"] / n}
                if self.M > 1:
                    r["spread"] = t["spread"] / n
                    r["spread_skill_fortin"] = math.sqrt((self.M + 1) / self.M * t["var"] / n) / max(rmse, 1e-12)
                out[str(s)][sp] = r
        return out
