"""cdffm_aq.store_ghap -- GhapStore: the GHAP V1 store of the CAMS -> GHAP 1 km downscaler (PROTOCOL.md Amendment A2;
data/processed/ghap, built by scripts/build_ghap.py, see its manifest.json), with the SmokeStore / GlobalStore
interface, so BatchStream, PixelArrays, assemble_inputs, train_aq, eval_x, copula_x and realtime_1km/infer.py run on it.

Target: k = max(round(GHAP), 1) int16 (the LGHAP integer convention, dequantised in training as before).
Domain: the GHAP lattice pixels valid on >= 95 % of the days 2017-2021 (grid/ghap_mask.npy, flat id r * 36000 + c,
row r at 83.995 - 0.01 r). Holes: training points are drawn among the valid pixels of each day only; the hold-out
value is -1 at a hole (has_holes = True): every score gives it zero weight (train_aq.validate, eval_x, copula_x).
Split G (A2(b)): training 2017-01-01 .. 2020-12-31 minus the incomplete GHAP days and the days with non-finite inputs
(manifest split_G.train_days, A2(a)); hold-out 2021-01-08 .. 12-31 (358 days, rows of
ghap_eval/full_2021.npy = doy); SEL = hold indices 0, 15, .., 345; 2022 sealed.
Tile (maps, spectra, copula disks): ascending latitude as the LGHAP stores, tile row i = 14399 - r (lat -59.995 + 0.01 i).
Firewall: as GlobalStore, plus the raw GHAP files (ghap1km_*) and data/processed/ghap/sealed (2022 test truth):
training, selection and prediction processes cannot read GHAP 2022 (or any raw GHAP) nor a station file.
Inputs of day d (raw_inputs): exactly GlobalStore's (CAMS 0.25 deg day-0, Foehn met.npy to 2022 / met_2023_2024 after,
fires injh (mami before 2018-07-02), T6) with the normalisation of the GHAP training years (manifest 'normalisation');
days >= 2025-01-01 from the backfill stores through store_global.Backfill2025 (coverage-checked, KeyError beyond)."""
import datetime as dt
import json
from pathlib import Path

import numpy as np

from . import PROJECT
from .data_aq import FOEHN_GLOBAL1KM, MET_FIRST, Norm, NpyRows, parse_date
from .store_global import (D2023, D2025, FIRES_FILES, GLOBAL_ROOT, SEL_HOLD_IDX, T6_YEARS, Backfill2025, injh_status,
                           install_firewall)

GHAP_ROOT = PROJECT / "data" / "processed" / "ghap"
GHAP_NR, GHAP_NC = 14400, 36000
GHAP_LAT0, GHAP_LON0, GHAP_RES = 83.995, -179.995, 0.01
GHAP_MARKERS = ["ghap1km_", "/data/processed/ghap/sealed"]
HOLE = -1


class GhapStore:
    name = "ghap"
    has_holes = True

    def __init__(self, root=GHAP_ROOT, split="G", fires="injh", firewall=True, sigma_k=0, coarse_root=GLOBAL_ROOT):
        from .model_aq import GRIDS
        root, coarse_root = Path(root), Path(coarse_root)
        self.root = root
        if split != "G":
            raise ValueError(f"GhapStore: split {split!r} (only G, PROTOCOL.md A2(b))")
        if int(sigma_k) not in (0, 2):
            raise ValueError(f"sigma_k={sigma_k!r}")
        self.sigma_k = int(sigma_k)
        if firewall:
            install_firewall(GHAP_MARKERS)
        man = json.loads((root / "manifest.json").read_text())
        self.manifest = man
        sp = man["split_G"]
        self.split, self.base_split = "G", "G"
        t0, t1, _ = sp["train"]
        # A2(a), 15:20: the 2017-2020 days with a complete GHAP field and finite inputs (manifest list)
        self.train_dates = [parse_date(x) for x in sp["train_days"]]
        assert len(self.train_dates) == int(sp["n_train_days"]) and self.train_dates == sorted(set(self.train_dates))
        assert parse_date(t0) <= self.train_dates[0] and self.train_dates[-1] <= parse_date(t1)
        v0, v1 = sp["val"]
        self.hold_dates = [parse_date(v0) + dt.timedelta(k) for k in range((parse_date(v1) - parse_date(v0)).days + 1)]
        assert all(d.year <= 2020 for d in self.train_dates) and all(d.year == 2021 for d in self.hold_dates)
        assert len(self.hold_dates) == 358
        self.sel_dates = [self.hold_dates[t] for t in SEL_HOLD_IDX]
        self.grid_native = GRIDS["025"]
        mask = np.load(root / "grid/ghap_mask.npy").astype(np.int64)
        r, c = mask // GHAP_NC, mask % GHAP_NC
        self.lat = GHAP_LAT0 - GHAP_RES * r
        self.lon = GHAP_LON0 + GHAP_RES * c
        self.n_pix = mask.size
        self.clim_mean = np.load(root / "ghap_clim/clim_mean_G.npy")
        self.clim_var = np.load(root / "ghap_clim/clim_var_G.npy")
        self.static = np.load(root / "ghap_clim/static_G.npy")
        self.static_dim = self.static.shape[1]
        self.sigma_floor = float(man["static"]["sigma_floor"])
        self.train_max_ug = float(man["T_G_train_max_points"])        # G0a threshold T_G (A2(c))
        self.pmax_ug = float(man["P_max_ug"])
        self._pts = []
        for y, f in sorted(man["points"]["files"].items()):
            self._pts.append((parse_date(f["first"]), parse_date(f["last"]), NpyRows(root / f["idx"]), NpyRows(root / f["val"])))
        self.n_points = self._pts[0][2].shape[1]
        assert all(idx.shape[1] == val.shape[1] == self.n_points for _, _, idx, val in self._pts)
        self.hold = NpyRows(root / man["eval"]["file"])
        self.cams = NpyRows(coarse_root / "coarse/cams_pm25_025.npy")
        self.met = (NpyRows(FOEHN_GLOBAL1KM / "met.npy"), NpyRows(coarse_root / "coarse/met_2023_2024.npy"))
        if fires not in FIRES_FILES:
            raise ValueError(f"fires={fires!r}")
        if fires == "injh":
            ok, why = injh_status(coarse_root)
            if not ok:
                raise SystemExit(f"GhapStore(fires='injh'): {why}")
        self.fires_name = fires
        self.fires = tuple(NpyRows(coarse_root / f) for f in FIRES_FILES[fires])
        self.t6 = {y: NpyRows(coarse_root / f"transport/t6_{y}.npy") for y in T6_YEARS
                   if (coarse_root / f"transport/t6_{y}.npy").exists()}
        self.bf = Backfill2025(coarse_root, fires, self.t6)
        nc = man["normalisation"]
        self.norm = Norm(nc["cams_log1p_mean_std"][0], nc["cams_log1p_mean_std"][1], nc["t6_channel_mean"], nc["t6_channel_std"])
        self.tile_shape = (GHAP_NR, GHAP_NC)
        self.tile_row = (GHAP_NR - 1 - r).astype(np.int64)
        self.tile_col = c.astype(np.int64)
        self.tile_row_lat = (GHAP_LAT0 - GHAP_RES * (GHAP_NR - 1)) + GHAP_RES * np.arange(GHAP_NR)
        self.tile_col_lon0 = GHAP_LON0
        self.res_deg = GHAP_RES
        assert self.clim_mean.shape == (self.n_pix, 5) and self.static.shape[0] == self.n_pix
        assert self.hold.shape == (365, self.n_pix)
        assert self.cams.row_shape == (self.grid_native.H, self.grid_native.W)
        assert self.met[0].row_shape == (37, self.grid_native.H, self.grid_native.W)
        assert self.fires[0].row_shape == (3, self.grid_native.H, self.grid_native.W)
        for k, r in self.bf.rows.items():
            assert r.row_shape == self.raw_shapes()[k] and len(r) == 730, (k, r.shape)
        self._train_set = set(self.train_dates)

    def raw_shapes(self):
        return {"cams": self.cams.row_shape, "met": self.met[0].row_shape, "fires": self.fires[0].row_shape,
                "t6": (6, self.grid_native.H // 2, self.grid_native.W // 2)}

    def raw_inputs(self, d, out=None):
        out = out or {}
        if d >= D2025:                                  # backfill 2025-2026 (coverage-checked)
            rows = self.bf.lookup(d)
        else:
            new = d >= D2023
            k = (d - D2023).days if new else (d - MET_FIRST).days
            rows = {"cams": (self.cams, (d - MET_FIRST).days), "met": (self.met[new], k), "fires": (self.fires[new], k),
                    "t6": (self.t6[d.year], (d - dt.date(d.year, 1, 1)).days)}
        return {key: r.read(i, out.get(key)) for key, (r, i) in rows.items()}

    def points(self, t):
        """training day t: (positions k (n_points,) uint32, k = max(round(GHAP), 1) int16), valid pixels only"""
        d = self.train_dates[t]
        assert d in self._train_set
        for f0, f1, idx, val in self._pts:
            if f0 <= d <= f1:
                r = (d - f0).days
                return idx.read(r), val.read(r)
        raise KeyError(f"{d}: not a training day")

    def holdout(self, t):
        """hold-out day t (index into hold_dates, 2021): int16 k at every mask pixel, -1 at a hole"""
        d = self.hold_dates[t]
        return self.hold.read((d - dt.date(2021, 1, 1)).days)
