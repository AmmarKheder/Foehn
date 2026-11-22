"""cdffm_aq_m.stations -- station conditioning of the A3 system (PROTOCOL.md Amendment A3(b), A3(c), A3(d')) and, at the
end of the file, its LAG option (Amendment A11, Addendum 2 point 1, experiment M; a11/stageM/README.md). Everything down to
open_stations() is the original cdffm_aq/stations.py, unchanged; the lag code is additive and is reached only when a run
is started with --stn-lag (default off = the A3 code path of pmS_base).

Assimilation set: data/processed/assim/assim_<year>.npz, written by scripts/build_stations_a3.py from the frozen split
data/processed/stations/split_v1.json. It holds ONLY the assimilation monitors (id, network, lat, lon, pos = index into
the GHAP mask order or -1 outside the mask, daily (N, n_days) float32 with NaN where the monitor has no valid UTC day).
The held-out monitors live under data/processed/stations/ (values) -- a path the GhapStore firewall blocks in every
model process -- and their ids are listed in data/processed/assim/split_v1_public.json, so StationSet can refuse any
assimilation file that contains one (the assert of A3(a)).

Feature vector of a query point (A3(b)), STN_DIM = 41 floats: for each of the K = 8 nearest assimilation monitors with
a valid value on the day (ascending distance),
    [log1p(value ug m-3), log1p(distance km), sin(bearing), cos(bearing), valid]
where valid = 1 if the monitor is within R_MAX = 300 km (otherwise the whole slot is 0: the model sees local monitors
only), followed by the day-point flag "no assimilation monitor within 300 km" (1 - any valid). The bearing is from the
query to the monitor (0 = north, clockwise). The neighbours are found by brute force on the unit sphere: float64 dot
products of the query and monitor unit vectors (precise to ~0.1 km near zero distance), chunked over the queries, with
the monitors pre-filtered to the chunk's latitude band +- 2.75 deg (the full-field inference reads the mask in flat
order, so a chunk of 32,768 pixels spans ~0.05 deg of latitude and ~10 % of the monitors are candidates: a few seconds
per 147 M-pixel day; the training batch of 2 x 16,384 scattered points uses every monitor of the day).
none_features(n) is the vector of a point without any monitor (all slots 0, flag 1): the day-level counterfactual
"pmS_nostations" of A3(c) and what the model sees under 100 % dropout.
Station dropout (training only, A3(c)): on each training day a fraction f ~ U(0, 1) of the day's valid assimilation
monitors is hidden monitor by monitor (keep if u >= f), from a torch generator seeded by (args.seed, step, day index)
(train_aq.compute_losses). Validation, selection, prior fit and inference use every valid monitor of the day.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from . import PROJECT

ASSIM_ROOT = PROJECT / "data" / "processed" / "assim"
K_NEAR = 8
R_MAX_KM = 300.0
STN_DIM = 5 * K_NEAR + 1
R_EARTH_KM = 6371.0
DLAT_BAND = R_MAX_KM / 111.0 + 0.05           # latitude pre-filter half-width (deg): 300 km + margin


def n_days(year):
    return 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365


def unit_xyz(lat, lon):
    """(...,) degrees -> (..., 3) float64 unit vectors"""
    phi, lam = torch.deg2rad(lat.double()), torch.deg2rad(lon.double())
    return torch.stack([torch.cos(phi) * torch.cos(lam), torch.cos(phi) * torch.sin(lam), torch.sin(phi)], dim=-1)


def none_features(n, device, dtype=torch.float32):
    """(n, STN_DIM): the vector of a point with no assimilation monitor (flag 'none within 300 km' = 1)"""
    out = torch.zeros(n, STN_DIM, device=device, dtype=dtype)
    out[:, -1] = 1.0
    return out


@torch.no_grad()
def station_features(qlat, qlon, mlat, mlon, mval, K=K_NEAR, rmax_km=R_MAX_KM, chunk=32768, mxyz=None):
    """qlat, qlon (n,) degrees (any dtype); mlat, mlon, mval (m,) the day's valid monitors ->
    (features (n, STN_DIM) float32, dmin_km (n,) float32 = distance to the nearest monitor, inf if none in the band)"""
    n, m = qlat.numel(), mlat.numel()
    dev = qlat.device
    out = none_features(n, dev)
    dmin = torch.full((n,), float("inf"), device=dev)
    if m == 0:
        return out, dmin
    if mxyz is None:
        mxyz = unit_xyz(mlat, mlon)
    qxyz = unit_xyz(qlat, qlon)
    lmval = torch.log1p(mval.double().clamp(min=0.0))
    phi_m, lam_m = torch.deg2rad(mlat.double()), torch.deg2rad(mlon.double())
    mlat_d = mlat.double()
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        ql = qlat[s:e].double()
        band = (mlat_d >= ql.min() - DLAT_BAND) & (mlat_d <= ql.max() + DLAT_BAND)
        idx = band.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        cosang = (qxyz[s:e] @ mxyz[idx].T).clamp(-1.0, 1.0)                     # (nc, mb) float64
        kk = min(K, idx.numel())
        top = cosang.topk(kk, dim=1)                                             # largest cos = nearest, ascending d
        j = idx[top.indices]                                                     # (nc, kk) monitor indices
        d = R_EARTH_KM * torch.acos(top.values)                                  # km
        valid = d <= rmax_km
        phi1, lam1 = torch.deg2rad(ql)[:, None], torch.deg2rad(qlon[s:e].double())[:, None]
        phi2, lam2 = phi_m[j], lam_m[j]
        dl = lam2 - lam1
        theta = torch.atan2(torch.sin(dl) * torch.cos(phi2),
                            torch.cos(phi1) * torch.sin(phi2) - torch.sin(phi1) * torch.cos(phi2) * torch.cos(dl))
        f = torch.stack([lmval[j], torch.log1p(d), torch.sin(theta), torch.cos(theta), valid.double()], dim=-1)
        f = f * valid[..., None]
        out[s:e, :kk * 5] = f.reshape(e - s, kk * 5).float()
        out[s:e, -1] = (~valid.any(1)).float()
        dmin[s:e] = d[:, 0].float()
    return out, dmin


class StationSet:
    """The assimilation monitors of the given years on one device (A3(b)); refuses held-out ids (A3(a))."""

    def __init__(self, root=ASSIM_ROOT, years=range(2017, 2022), device="cpu"):
        root = Path(root)
        self.root, self.device = root, torch.device(device)
        pub = json.loads((root / "split_v1_public.json").read_text())
        self.heldout_ids = set(pub["heldout_ids"])
        self.split_sha256 = pub.get("split_v1_sha256")
        self.years = {}
        for y in years:
            f = root / f"assim_{y}.npz"
            if not f.exists():
                raise FileNotFoundError(f"{f}: assimilation file of {y} not built (scripts/build_stations_a3.py)")
            z = np.load(f)
            ids = z["id"].astype(str).tolist()
            bad = self.heldout_ids.intersection(ids)
            assert not bad, f"{f}: {len(bad)} held-out monitors in the assimilation set, e.g. {sorted(bad)[:3]}"
            daily = np.asarray(z["daily"], np.float32)
            assert daily.shape == (len(ids), n_days(y)), (f, daily.shape)
            lat = torch.from_numpy(np.asarray(z["lat"], np.float32)).to(self.device)
            lon = torch.from_numpy(np.asarray(z["lon"], np.float32)).to(self.device)
            self.years[y] = dict(n=len(ids), ids=ids, network=z["network"].astype(str).tolist(), lat=lat, lon=lon,
                                 xyz=unit_xyz(lat, lon), daily=torch.from_numpy(daily).to(self.device),
                                 pos=np.asarray(z["pos"], np.int64))
        self.n_total = sum(v["n"] for v in self.years.values())

    def day(self, d, hide=None, gen=None):
        """valid monitors of day d -> (lat, lon, value, xyz); hide: fraction in [0, 1] of them hidden at random
        (keep where u >= hide, u ~ U(0, 1) from gen): the training dropout of A3(c)"""
        y = self.years[d.year]
        v = y["daily"][:, (d - dt.date(d.year, 1, 1)).days]
        ok = torch.isfinite(v)
        if hide is not None and hide > 0.0:
            u = torch.rand(ok.numel(), device=self.device, generator=gen)
            ok = ok & (u >= float(hide))
        i = ok.nonzero(as_tuple=True)[0]
        return y["lat"][i], y["lon"][i], v[i], y["xyz"][i]

    def features(self, d, qlat, qlon, hide=None, gen=None):
        """(features (n, STN_DIM), dmin_km (n,)) at the query points for day d (station_features)"""
        mlat, mlon, mval, mxyz = self.day(d, hide=hide, gen=gen)
        return station_features(qlat, qlon, mlat, mlon, mval, mxyz=mxyz)

    def n_valid(self, d):
        y = self.years[d.year]
        return int(torch.isfinite(y["daily"][:, (d - dt.date(d.year, 1, 1)).days]).sum())


def dropout_generator(seed, step, day_index, device):
    """torch generator of the station dropout of one training day (A3(c): seeded per day and step) and its hidden
    fraction f ~ U(0, 1)"""
    g = torch.Generator(device=device)
    g.manual_seed(int(np.random.default_rng([int(seed), int(step), int(day_index), 11]).integers(2 ** 62)))
    f = float(torch.rand(1, device=device, generator=g))
    return g, f


def open_stations(args, years, device, store=None):
    """StationSet of a run (args.stations = assimilation root, None/'' = no stations) or None. A11 experiment M: a run
    with --stn-lag gets a LagStationSet (it needs the store: the climatology at the monitors' pixels)"""
    root = getattr(args, "stations", None)
    if not root:
        return None
    from .store_global import install_firewall
    install_firewall(HELDOUT_VALUE_MARKERS)          # this copy never opens a held-out value file (lag on or off)
    if lag_spec(args) is not None:
        assert store is not None, "--stn-lag: open_stations needs the store (climatology of the monitors' pixels)"
        return LagStationSet(root, years=years, device=device, clim_mean=store.clim_mean, pix_lat=store.lat,
                             pix_lon=store.lon)
    return StationSet(root, years=years, device=device)


# ---------------------------------------------------------------------------------------------------------------
# A11 experiment M (Addendum 2, point 1): the station conditioning of A3 with a LAG, for forecasts.
#
# For a valid day t and a lag l (= the lead day: the valid day is l days after the last monitor day, the day before the
# issue), the network receives the assimilation monitors' values of day t - l, never a later one, and the lag.
# Feature vector of a query point, STN_DIM_LAG = 63 floats:
#   [0:40]   the A3 slots of the K = 8 nearest assimilation monitors that have a valid UTC-day value on day t - l:
#            [log1p(value of day t - l), log1p(distance km), sin(bearing), cos(bearing), valid (<= 300 km)]
#   [40]     the A3 flag "no assimilation monitor within 300 km" (of the monitors valid on day t - l)
#   [41:49]  anomaly of each slot: log1p(max(value, 1)) - mu_p(doy(t - l)), mu_p = the harmonic climatology 2017-2020 of
#            the 1 km estimates at the monitor's OWN pixel (store.clim_mean[pos], the pixel descriptor the package already
#            uses; data_aq.PixelArrays.mu_sigma). max(., 1) is the convention of the target (k = max(round(.), 1)): a
#            monitor below 1 ug m-3 is compared with a climatology that cannot go below log1p(1). 0 where not available.
#   [49:57]  1 where the slot's anomaly is available (slot valid AND the monitor's pixel is inside the store's mask,
#            pos >= 0); about 1 % of the monitors are outside the mask: they keep their A3 slot, without an anomaly
#   [57:63]  one-hot of the lag (0..5), zero at a point without any valid slot
# The first 41 entries are exactly cdffm_aq.stations.station_features of the monitors of day t - l (tested), so the
# same-day model of A3 (pmS_final) can be fed with features[:, :41]: the naive baseline of val_lag.py.
# A point without any monitor within 300 km gets none_features_lag: every entry 0 except [40] = 1, WHATEVER the lag (the
# lag one-hot is zeroed), so the no-monitor state of the network cannot depend on the lag: it is the single station-
# free state, the one reached under 100 % dropout.
# Missing data: a monitor without a valid UTC-day mean on day t - l is absent from the features of (t, l) (no look-back
# to an older value: the lag input stays exact); a monitor that reports on day t - l but not on day t is used (nothing
# of day t is known at the issue); a day t - l before the first assimilation file (2017-01-01) has no monitor at all.
# Training (train_aq.compute_losses): the lag of a training day is drawn by draw_lag (0 with probability
# --stn-lag0-prob, otherwise uniform over --stn-lag), then the A3 dropout is applied to the monitors of day t - l with
# the A3 generator. Validation and inference give every valid monitor of day t - l.
# Guarantees asserted here (and tested by a11/stageM/tests/test_m.py):
#   * no held-out identifier: StationSet refuses an assimilation file that holds one; LagStationSet also keeps the
#     held-out flag of every row and asserts it on every call;
#   * the values served for (t, l) are column (t - l) of the daily array: asserted on every call, with l >= 1 => the
#     source day is strictly before the valid day;
#   * a process that opens a station set through this copy (open_stations, lag on or off) or builds a LagStationSet
#     cannot open a held-out VALUE: the audit-hook firewall of store_global gets the markers of the held-out SEL file
#     (data/processed/assim/heldout_sel2021.npz, the scoring file of A3(d')-4) and of data/processed/stations (the
#     daily_<year>.npz files).
# ---------------------------------------------------------------------------------------------------------------

LAG_MAX = 5
N_LAG = LAG_MAX + 1                                  # one-hot over the lags 0..5
STN_DIM_LAG = STN_DIM + 2 * K_NEAR + N_LAG           # 41 + 8 anomalies + 8 anomaly flags + 6 = 63
LAG_A0, LAG_F0, LAG_L0 = STN_DIM, STN_DIM + K_NEAR, STN_DIM + 2 * K_NEAR      # offsets: anomalies, flags, lag one-hot
ARCHIVE_FIRST_YEAR = 2017                            # first assimilation file: an earlier source day has no monitor
HELDOUT_VALUE_MARKERS = ("heldout_sel", "/data/processed/stations")


def lag_spec(args):
    """None (lag option off: the A3 code path) or dict(lags=(..), p0=float) from args.stn_lag ('1,2,3,4,5') and
    args.stn_lag0_prob (probability of the same-day lag 0 at training)"""
    s = getattr(args, "stn_lag", None)
    p0 = float(getattr(args, "stn_lag0_prob", 0.0) or 0.0)
    if not s:
        if p0 != 0.0:
            raise SystemExit("--stn-lag0-prob needs --stn-lag")
        return None
    if not getattr(args, "stations", None):
        raise SystemExit("--stn-lag needs --stations (the assimilation root)")
    lags = tuple(sorted({int(x) for x in str(s).split(",") if str(x).strip()}))
    if not lags or lags[0] < 1 or lags[-1] > LAG_MAX:
        raise SystemExit(f"--stn-lag {s!r}: lags must be in 1..{LAG_MAX} (the same-day lag 0 is --stn-lag0-prob)")
    if not 0.0 <= p0 < 1.0:
        raise SystemExit(f"--stn-lag0-prob {p0}: a probability in [0, 1)")
    return dict(lags=lags, p0=p0)


def stn_dim_of(args):
    """input size of the station MLP of a run: 0 without --stations, 41 (A3) or 63 (A3 + lag option)"""
    if not getattr(args, "stations", None):
        return 0
    return STN_DIM_LAG if lag_spec(args) is not None else STN_DIM


def draw_lag(seed, step, day_index, spec):
    """the lag of one training day at one step: 0 with probability spec['p0'], otherwise uniform over spec['lags'].
    numpy generator seeded by (seed, step, day index, 13): independent of the batch stream (the days and points of a run
    are the same with the option on or off), of the dropout draw (tag 11) and of the device"""
    g = np.random.default_rng([int(seed), int(step), int(day_index), 13])
    u, j = float(g.random()), int(g.integers(len(spec["lags"])))
    return 0 if u < spec["p0"] else int(spec["lags"][j])


def harmonic_basis(doy):
    """(5,) float32 basis of the K = 2 harmonic climatology at a 0-based day of year, as PixelArrays.mu_sigma"""
    ang = 2.0 * np.pi * float(doy) / 365.0
    return torch.tensor([1.0, np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)], dtype=torch.float64).float()


def none_features_lag(n, device, dtype=torch.float32):
    """(n, STN_DIM_LAG): a point with no assimilation monitor; the same vector for every lag (see the block comment)"""
    out = torch.zeros(n, STN_DIM_LAG, device=device, dtype=dtype)
    out[:, STN_DIM - 1] = 1.0
    return out


class LagStationSet(StationSet):
    """StationSet + the lagged access of experiment M. clim_mean: (n_pix, 5) harmonic coefficients of the store
    (store.clim_mean; numpy array or memory map, only the monitors' rows are read); pix_lat / pix_lon: the store's pixel
    coordinates, to check that pos is the monitor's own pixel (None = not checked: the CPU test does it itself)."""

    def __init__(self, root=ASSIM_ROOT, years=range(2017, 2022), device="cpu", clim_mean=None, pix_lat=None, pix_lon=None,
                 pix_res=0.01):
        from .store_global import install_firewall
        install_firewall(HELDOUT_VALUE_MARKERS)      # before any file of the set is opened
        super().__init__(root, years=years, device=device)
        assert clim_mean is not None and clim_mean.ndim == 2 and clim_mean.shape[1] == 5, "clim_mean (n_pix, 5) needed"
        assert len(self.heldout_ids) > 0, "split_v1_public.json lists no held-out monitor"
        n_pix = clim_mean.shape[0]
        for y, v in self.years.items():
            pos = v["pos"]
            ok = pos >= 0
            assert pos.max(initial=-1) < n_pix, (y, int(pos.max()), n_pix)
            c = np.zeros((v["n"], 5), np.float32)
            order = np.argsort(pos[ok], kind="stable")          # sorted rows: a memory map is read front to back
            rows = np.asarray(clim_mean[pos[ok][order]], np.float32)
            c[np.nonzero(ok)[0][order]] = rows
            assert np.isfinite(c).all(), (y, "non-finite climatology at a monitor pixel")
            if pix_lat is not None:
                la, lo = v["lat"].cpu().numpy().astype(np.float64), v["lon"].cpu().numpy().astype(np.float64)
                dla = np.abs(np.asarray(pix_lat[pos[ok]], np.float64) - la[ok])
                dlo = np.abs(np.asarray(pix_lon[pos[ok]], np.float64) - lo[ok])
                tol = 0.5 * pix_res + 1e-4                       # half a pixel + float32 rounding of the coordinates
                assert dla.size == 0 or (dla.max() <= tol and dlo.max() <= tol), \
                    (y, "pos is not the monitor's own pixel", float(dla.max()), float(dlo.max()))
            v["clim"] = torch.from_numpy(c).to(self.device)
            v["clim_ok"] = torch.from_numpy(ok).to(self.device)
            v["held"] = torch.from_numpy(np.isin(np.asarray(v["ids"]), sorted(self.heldout_ids))).to(self.device)
            assert not bool(v["held"].any())
        self.n_calls = 0

    def lagged(self, d, lag, hide=None, gen=None):
        """the monitors given to the network for the valid day d at lag `lag`: those with a valid value on day d - lag.
        hide / gen: the A3 training dropout (StationSet.day), applied to the monitors of day d - lag.
        -> dict(lat, lon, val, xyz, anom, anom_ok (m,), idx (m,) rows of the source year, src (date), year, lag)"""
        lag = int(lag)
        assert 0 <= lag <= LAG_MAX, lag
        src = d - dt.timedelta(days=lag)
        assert (d - src).days == lag and (lag == 0 or src < d)      # lag >= 1: never a value of the valid day
        self.n_calls += 1
        if src.year < ARCHIVE_FIRST_YEAR:                           # before the archive: no monitor at all
            e = torch.zeros(0, device=self.device)
            return dict(lat=e, lon=e, val=e, xyz=torch.zeros(0, 3, device=self.device, dtype=torch.float64), anom=e,
                        anom_ok=torch.zeros(0, dtype=torch.bool, device=self.device),
                        idx=torch.zeros(0, dtype=torch.long, device=self.device), src=src, year=None, lag=lag)
        if src.year not in self.years:
            raise KeyError(f"lag {lag} of {d}: the assimilation year {src.year} is not loaded (years {sorted(self.years)})")
        y = self.years[src.year]
        col = (src - dt.date(src.year, 1, 1)).days
        assert dt.date(src.year, 1, 1) + dt.timedelta(days=col) == src and 0 <= col < y["daily"].shape[1]
        v = y["daily"][:, col]
        ok = torch.isfinite(v)
        if hide is not None and hide > 0.0:                         # exactly StationSet.day's dropout
            u = torch.rand(ok.numel(), device=self.device, generator=gen)
            ok = ok & (u >= float(hide))
        i = ok.nonzero(as_tuple=True)[0]
        assert not bool(y["held"][i].any()), "held-out monitor in the assimilation set"
        val = v[i]
        mu = (y["clim"][i] * harmonic_basis(col).to(self.device)).sum(-1)
        anom = torch.log1p(val.clamp(min=1.0)) - mu
        return dict(lat=y["lat"][i], lon=y["lon"][i], val=val, xyz=y["xyz"][i], anom=anom, anom_ok=y["clim_ok"][i], idx=i,
                    src=src, year=src.year, lag=lag)

    def ids_of(self, mon):
        """identifiers of the monitors of a lagged() result (tests)"""
        if mon["year"] is None:
            return []
        ids = self.years[mon["year"]]["ids"]
        return [ids[int(k)] for k in mon["idx"].tolist()]


@torch.no_grad()
def lag_features(qlat, qlon, mon, K=K_NEAR, rmax_km=R_MAX_KM, chunk=32768):
    """qlat, qlon (n,) degrees; mon: LagStationSet.lagged(d, lag) ->
    (features (n, STN_DIM_LAG) float32, dmin_km (n,) float32 = distance to the nearest monitor of day d - lag, inf if none
    in the latitude band). The loop is station_features' (same neighbours, same first 41 entries), plus the anomaly of
    each slot, its availability and the lag."""
    lag = int(mon["lag"])
    assert 0 <= lag <= LAG_MAX
    mlat, mlon, mval, mxyz = mon["lat"], mon["lon"], mon["val"], mon["xyz"]
    n, m = qlat.numel(), mlat.numel()
    dev = qlat.device
    out = none_features_lag(n, dev)
    dmin = torch.full((n,), float("inf"), device=dev)
    if m == 0:
        return out, dmin
    qxyz = unit_xyz(qlat, qlon)
    lmval = torch.log1p(mval.double().clamp(min=0.0))
    manom, maok = mon["anom"].double(), mon["anom_ok"]
    phi_m, lam_m = torch.deg2rad(mlat.double()), torch.deg2rad(mlon.double())
    mlat_d = mlat.double()
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        ql = qlat[s:e].double()
        band = (mlat_d >= ql.min() - DLAT_BAND) & (mlat_d <= ql.max() + DLAT_BAND)
        idx = band.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        cosang = (qxyz[s:e] @ mxyz[idx].T).clamp(-1.0, 1.0)                     # (nc, mb) float64
        kk = min(K, idx.numel())
        top = cosang.topk(kk, dim=1)                                             # largest cos = nearest, ascending d
        j = idx[top.indices]                                                     # (nc, kk) monitor indices
        d = R_EARTH_KM * torch.acos(top.values)                                  # km
        valid = d <= rmax_km
        phi1, lam1 = torch.deg2rad(ql)[:, None], torch.deg2rad(qlon[s:e].double())[:, None]
        phi2, lam2 = phi_m[j], lam_m[j]
        dl = lam2 - lam1
        theta = torch.atan2(torch.sin(dl) * torch.cos(phi2),
                            torch.cos(phi1) * torch.sin(phi2) - torch.sin(phi1) * torch.cos(phi2) * torch.cos(dl))
        f = torch.stack([lmval[j], torch.log1p(d), torch.sin(theta), torch.cos(theta), valid.double()], dim=-1)
        f = f * valid[..., None]
        out[s:e, :kk * 5] = f.reshape(e - s, kk * 5).float()
        anyv = valid.any(1)
        out[s:e, STN_DIM - 1] = (~anyv).float()
        aok = valid & maok[j]                                                    # anomaly available in this slot
        out[s:e, LAG_A0:LAG_A0 + kk] = (manom[j] * aok).float()
        out[s:e, LAG_F0:LAG_F0 + kk] = aok.float()
        out[s:e, LAG_L0 + lag] = anyv.float()                                    # the lag, only where a monitor is seen
        dmin[s:e] = d[:, 0].float()
    return out, dmin
