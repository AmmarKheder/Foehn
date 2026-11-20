"""cdffm_aq.spectra -- numpy-only spectral tools (no torch, so the world venv can import it too).

Used by scripts/measure_noise_beta.py (fit of the noise prior on LGHAP z-fields, PLAN.md 4.2 "Noise-prior fitting"),
by test_s0.py (slope of the sphere noise) and by eval_aq.py (spectra of members, ensemble mean, truth, CAMS).

Boxes on the 0.01 deg LGHAP grid are cut about square in km (n_rows x n_cols with n_cols = n_rows / cos(lat_c)), so the
Fourier grid is isotropic in cycles/km; the PSD is the radially averaged per-mode power (as CD-FFM's radial_psd),
binned in log-spaced |k| bins, after mean removal and a separable Hann window.
"""
import numpy as np

R_EARTH_KM = 6371.0
KM_PER_DEG = 2.0 * np.pi * R_EARTH_KM / 360.0          # 111.195 km per degree of latitude


def box_shape(lat_c, size_km=512.0, res_deg=0.01):
    """(n_rows, n_cols, dy_km, dx_km) of a ~size_km x size_km box centred at latitude lat_c on a res_deg lat-lon grid."""
    dy = res_deg * KM_PER_DEG
    dx = dy * np.cos(np.deg2rad(lat_c))
    return int(round(size_km / dy)), int(round(size_km / dx)), float(dy), float(dx)


def default_kbins(size_km=512.0, finest_km=2.0, n=40):
    """log-spaced |k| bin edges in cycles/km, from 1/size_km to 1/finest_km"""
    return np.geomspace(1.0 / size_km, 1.0 / finest_km, n + 1)


def radial_psd(field, dy_km, dx_km, kbins=None, window=True):
    """Radially averaged per-mode power spectrum of a 2-D field (ny, nx) with no NaN (fill invalid pixels before).
    Returns k (geometric bin centres, cycles/km), psd (mean |FFT|^2 per Fourier mode in the bin; NaN if empty), count.
    Only modes inside the inscribed Nyquist circle are used (min(1/2dy, 1/2dx))."""
    f = np.asarray(field, np.float64)
    f = f - f.mean()
    ny, nx = f.shape
    if window:
        w = np.outer(np.hanning(ny), np.hanning(nx))
        f = f * w
        norm = (w ** 2).sum()
    else:
        norm = float(ny * nx)
    P = np.abs(np.fft.fft2(f)) ** 2 / norm
    ky = np.fft.fftfreq(ny, d=dy_km)
    kx = np.fft.fftfreq(nx, d=dx_km)
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    kmax = min(0.5 / dy_km, 0.5 / dx_km)
    if kbins is None:
        kbins = default_kbins(size_km=max(ny * dy_km, nx * dx_km), finest_km=1.0 / kmax)
    sel = (kr > 0) & (kr <= kmax)
    b = np.digitize(kr[sel], kbins) - 1
    ok = (b >= 0) & (b < len(kbins) - 1)
    s = np.bincount(b[ok], weights=P[sel][ok], minlength=len(kbins) - 1)
    c = np.bincount(b[ok], minlength=len(kbins) - 1)
    psd = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    k = np.sqrt(kbins[:-1] * kbins[1:])
    return k, psd, c


def fit_slope(k, psd, lam_lo_km, lam_hi_km):
    """least-squares slope of log psd against log k over wavelengths [lam_lo_km, lam_hi_km]:
    dict(slope, intercept, r2, n, rms) (slope ~ -beta for a k^-beta per-mode spectrum)"""
    k, psd = np.asarray(k), np.asarray(psd)
    m = (k >= 1.0 / lam_hi_km) & (k <= 1.0 / lam_lo_km) & np.isfinite(psd) & (psd > 0)
    if m.sum() < 3:
        return dict(slope=float("nan"), intercept=float("nan"), r2=float("nan"), n=int(m.sum()), rms=float("nan"))
    x, y = np.log(k[m]), np.log(psd[m])
    A = np.stack([x, np.ones_like(x)], 1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - (a * x + b)
    r2 = 1.0 - (res ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return dict(slope=float(a), intercept=float(b), r2=float(r2), n=int(m.sum()), rms=float(np.sqrt((res ** 2).mean())))


def band_slopes(k, psd, edges_km):
    """local slopes over consecutive wavelength bands, edges_km descending or ascending, e.g. (2, 4, 8, 16, ...)"""
    e = sorted(edges_km)
    return {f"{lo:g}-{hi:g}km": fit_slope(k, psd, lo, hi)["slope"] for lo, hi in zip(e[:-1], e[1:])}


def find_boxes(valid, row_lat, size_km=512.0, res_deg=0.01, min_frac=0.995, stride=50, max_boxes=10):
    """Greedy choice of up to max_boxes non-overlapping ~size_km boxes on a (ny, nx) validity tile whose row r has
    latitude row_lat[r], ranked by valid fraction (ties in scan order); candidates on a stride-pixel lattice.
    Returns [(r0, c0, n_rows, n_cols, frac, lat_c)]."""
    valid = np.asarray(valid, bool)
    ny, nx = valid.shape
    ii = np.zeros((ny + 1, nx + 1), np.int64)
    ii[1:, 1:] = valid.astype(np.int64).cumsum(0).cumsum(1)
    cands = []
    for r0 in range(0, ny, stride):
        nr_guess = box_shape(float(row_lat[min(r0, ny - 1)]), size_km, res_deg)[0]
        if r0 + nr_guess > ny:
            break
        lat_c = float(np.mean(row_lat[r0:r0 + nr_guess]))
        nr, nc, _, _ = box_shape(lat_c, size_km, res_deg)
        if r0 + nr > ny:
            continue
        for c0 in range(0, nx - nc + 1, stride):
            s = ii[r0 + nr, c0 + nc] - ii[r0, c0 + nc] - ii[r0 + nr, c0] + ii[r0, c0]
            frac = s / float(nr * nc)
            if frac >= min_frac:
                cands.append((frac, r0, c0, nr, nc, lat_c))
    cands.sort(key=lambda t: -t[0])
    taken = np.zeros_like(valid)
    out = []
    for frac, r0, c0, nr, nc, lat_c in cands:
        if taken[r0:r0 + nr, c0:c0 + nc].any():
            continue
        taken[r0:r0 + nr, c0:c0 + nc] = True
        out.append((r0, c0, nr, nc, float(frac), lat_c))
        if len(out) >= max_boxes:
            break
    return out


def box_psd(tile, valid, box, kbins, res_deg=0.01):
    """PSD of one box (r0, c0, n_rows, n_cols, frac, lat_c) of a tile; invalid pixels are filled with the box mean"""
    r0, c0, nr, nc, _, lat_c = box
    f = np.asarray(tile[r0:r0 + nr, c0:c0 + nc], np.float64)
    v = np.asarray(valid[r0:r0 + nr, c0:c0 + nc], bool) & np.isfinite(f)
    f = np.where(v, f, f[v].mean())
    _, _, dy, dx = box_shape(lat_c, res_deg=res_deg)
    return radial_psd(f, dy, dx, kbins)
