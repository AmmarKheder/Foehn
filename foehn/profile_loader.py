"""Time each read of one training sample separately (numpy only)."""
import time, numpy as np
from pathlib import Path
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
FH, FW, CH_, CW_, M, TILE, NTR, NTC = 4192, 6992, 168, 280, 96, 512, 29, 71
def header(p):
    with open(p, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); return f.tell()
def tile_window(path, hdr, k, r0, c0, h=FH, w=FW):
    tr0, tr1, tc0, tc1 = r0 // TILE, (r0 + h - 1) // TILE, c0 // TILE, (c0 + w - 1) // TILE
    m = np.memmap(path, dtype=np.float16, mode="r", offset=hdr + k * NTR * NTC * TILE * TILE * 2, shape=(NTR, NTC, TILE, TILE))
    blk = np.asarray(m[tr0:tr1 + 1, tc0:tc1 + 1])
    blk = blk.transpose(0, 2, 1, 3).reshape((tr1 - tr0 + 1) * TILE, (tc1 - tc0 + 1) * TILE)
    return blk[r0 - tr0 * TILE:r0 - tr0 * TILE + h, c0 - tc0 * TILE:c0 - tc0 * TILE + w]
def ctx_slice(a, k0, j0, scale=1):
    Hs, Ws = a.shape[-2], a.shape[-1]
    r = np.clip(np.arange(k0 - M, k0 + CH_ + M, scale) // scale, 0, Hs - 1)
    c = (np.arange(j0 - M, j0 + CW_ + M, scale) // scale) % Ws
    r0, r1 = int(r.min()), int(r.max()) + 1
    blk = np.asarray(a[..., r0:r1, :])
    return blk[..., r - r0, :][..., c]
rng = np.random.default_rng(int(time.time()))
i, h, k0, j0 = int(rng.integers(10, 1400)), 3, 4 * int(rng.integers(20, 100)), 4 * int(rng.integers(0, 290))
r0, c0 = 25 * k0, 25 * j0
T = {}
t = time.time(); ft = header(G / "ghap1km_tiles.npy")
for d in (i, i - 1, i + h): tile_window(G / "ghap1km_tiles.npy", ft, d, r0, c0)
T["fine 3 days (tiles)"] = time.time() - t
t = time.time(); st = header(G / "static1km_tiles.npy")
for s in (0, 1): tile_window(G / "static1km_tiles.npy", st, s, r0, c0)
T["static 2 maps (tiles)"] = time.time() - t
met, pm = np.load(G / "met.npy", mmap_mode="r"), np.load(G / "pm.npy", mmap_mode="r")
t = time.time(); np.asarray(met[i][..., k0:k0 + CH_, :]); np.asarray(met[i + h][..., k0:k0 + CH_, :]); np.asarray(pm[i][k0:k0 + CH_, :])
T["coarse window rows (met x2, pm)"] = time.time() - t
wind, ant, fire = np.load(G / "wind05.npy", mmap_mode="r"), np.load(G / "emis_ant.npy", mmap_mode="r"), np.load(G / "fires.npy", mmap_mode="r")
t = time.time(); ctx_slice(wind[i:i + 8], k0, j0, 2); T["wind 8 days (ctx)"] = time.time() - t
t = time.time(); ctx_slice(met[i:i + 8, 33], k0, j0); ctx_slice(met[i:i + 8, 5:8], k0, j0); T["rain+temp 8 days (ctx)"] = time.time() - t
t = time.time(); ctx_slice(pm[i], k0, j0); ctx_slice(ant[3], k0, j0); ctx_slice(fire[i], k0, j0); T["pm/emis/fire (ctx)"] = time.time() - t
for k, v in T.items(): print(f"{k:34s} {v:6.2f} s")
print(f"{'TOTAL':34s} {sum(T.values()):6.2f} s   (sample i={i}, window {k0},{j0})")
