#!/usr/bin/env python3
"""A20 step 0 (CPU, no member, no truth): the pixel sample of A20 -> a20/data/pix20.npz.
  pixel sample  NP_SAMPLE = 1,000,000 of the 8,111,991 populated pixels with at least min_pop (100) residents
                (pop_pixels.npz), numpy default_rng(PIX_SEED).choice(n, NP, replace=False), sorted: positions in the
                populated-pixel list (the row index of G2's and G11's pop files)
  stations      the position in the populated-pixel list of each of the 8,110 stations of the paper's station score
                (score19.stations_in_pop arithmetic, copied: 1 km pixel row * 36000 + column of lat / lon), -1 for the 57
                without a populated pixel; asserted equal to a14/classU/data_ar/st16_run16.npz 'stpos' (the positions of
                the A14 / A19 station scorer). Only the 'lat' and 'lon' arrays of stations_2022.npz are read: no measurement.
  null subset   every NULL_STRIDE-th row of the pixel sample (100,000) and every station row: the rows where the noise
                features of members 4-7 (the null 'another member') are collected
Row layout of every A20 array: rows 0 .. NP - 1 = the pixel sample, rows NP .. NP + NST - 1 = the NST = 8,053 stations
with a populated pixel, in station order (several stations may share a pixel; each keeps its own row).
    python stpos20.py"""
import common20 as C  # noqa: I001  (first: hooks)

C.install("positions")
import time  # noqa: E402

import numpy as np  # noqa: E402


def station_pixels(lat, lon):
    """1 km pixel of each station, as eval_global_green.py (l. 336-338): row * 36000 + column (score19 copy)"""
    sr = np.clip(np.floor((84.0 - lat) / 0.01).astype(int), 0, C.GH - 1)
    sc = np.clip(np.floor((lon + 180.0) / 0.01).astype(int), 0, C.GW - 1)
    return sr.astype(np.int64) * C.GW + sc


def main():
    t0 = time.time()
    pp = np.load(C.G / "pop_pixels.npz")
    idx, pop, min_pop = pp["idx"], pp["pop"].astype(np.float64), float(pp["min_pop"])
    assert len(idx) == C.NPOP and np.all(np.diff(idx) > 0)
    with np.load(C.G / "stations_2022.npz") as st:                  # lazy: only lat and lon are read
        lat, lon = st["lat"], st["lon"]                            # float32, as score19 (no cast: the pixel of a station)
    pix = station_pixels(lat, lon)
    pos = np.clip(np.searchsorted(idx, pix), 0, len(idx) - 1)
    stpos = np.where(idx[pos] == pix, pos, -1).astype(np.int64)
    a14 = np.load(C.ROOT / "a14" / "classU" / "data_ar" / "st16_run16.npz")["stpos"]
    same_a14 = bool(np.array_equal(a14, stpos))
    assert same_a14, "station positions differ from a14 st16_run16.npz"
    cand = np.flatnonzero(pop >= min_pop)
    sel = np.sort(np.random.default_rng(C.PIX_SEED).choice(len(cand), C.NP_SAMPLE, replace=False))
    pixpos = cand[sel].astype(np.int64)
    stcol = np.full(len(stpos), -1, np.int64)
    has = np.flatnonzero(stpos >= 0)
    stcol[has] = np.arange(len(has))
    rows = np.concatenate([pixpos, stpos[has]])
    flat = idx[rows]
    r, c = (flat // C.GW).astype(np.int64), (flat % C.GW).astype(np.int64)
    nullrows = np.concatenate([np.arange(0, C.NP_SAMPLE, C.NULL_STRIDE), C.NP_SAMPLE + np.arange(len(has))]).astype(np.int64)
    out = dict(pixpos=pixpos, stpos=stpos, stcol=stcol, rows=rows, r=r, c=c, pop=pop[rows], nullrows=nullrows,
               NP=np.int64(C.NP_SAMPLE), NST=np.int64(len(has)), seed=np.int64(C.PIX_SEED), min_pop=np.float64(min_pop),
               n_popok=np.int64(len(cand)), stations=np.int64(len(stpos)),
               numpy=np.array(np.__version__))
    C.save_npz(C.DATA / "pix20.npz", **out)
    C.log(f"pix20.npz: {C.NP_SAMPLE} sampled of {len(cand)} pixels with >= {min_pop:g} residents; stations {len(stpos)}, "
          f"{len(has)} with a populated pixel ({len(np.unique(stpos[has]))} distinct pixels); null rows {len(nullrows)}; "
          f"station positions equal a14 st16_run16: {same_a14}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
