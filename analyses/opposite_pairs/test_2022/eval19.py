#!/usr/bin/env python3
"""A19 evaluator (a19/A19_draft.md sections 4-5): A16's populated-pixel evaluation of 2022 (run G2: eval_ar22.run_ar22,
--pop, 179 start days x leads 1 and 3, 64 GCDs) with the member ids of the plan, ids 4..7 being the exact negatives of the
noise of members 0..3 (A17's construction, noise19.py).

run19(a) is a verbatim copy of inspect.getsource(a16/eval_ar22.run_ar22) (itself eval_global_green.run with the marked
hunks A16-H1..H5); every changed or added line ends with '# A19-H<n>' and every line of run_ar22 that does not survive
verbatim is listed in REMOVED_LINES (check19.py DIFF):
  H1  the function name
  H2  WindowsAR19 (noise19.py) replaces WindowsAR22: x0 stacks the members of the plan's member ids a.mids; member id m =
      4..7 is -z(m - 4, d, h) of the A16 evolving noise of 2022 (rho* = 0.59), member id 0..3 is z(m, d, h)
  H3  the particles of the k-th member of the run are member_particles(i, h, a.mids[k]) (an eight-member run's particles)
  H4  no station file is opened (the audit hook of the populated-pixel samplers is installed by noise19): the station
      pixel list is empty, as in a15/eval_ar.py (A15-H3), and the part file stores the station columns with 0 stations
      (their reshape uses the forecast count, as A15-H3)
  removed, no marked line: the 5 km maps (map5km_d*_h*.npy of every seventh start day; no A19 scorer reads them; 52
      files fewer, as a15/eval_ar.py)
Lines kept verbatim call module-level helpers of THIS file: start() (the A19 plan size and layout, the folder claim with
the A19 manifest a19_run.json, the resume state of this rank; its returned dict, named a16f in the verbatim lines, holds
the part-file fields a19 (the manifest), a19_jobs and member_ids) and asave / asavez (noise19: atomic, read back, inside
the claimed folder only).
Everything else is A16's G2 evaluation, unchanged: forecast list and its split over ranks (every 2, leads 1 and 3,
todo[rank::64]: part_<r> holds the forecasts of G2's part_<r>, in the same order), model load, sampler (Euler, 4 steps, bf16
autocast), float32 accumulation and blend, region sums (over this run's members), seams, keys, map25 / map25_sd (over this
run's members), map25_m0 (its first member id), map25_mk, the populated pixels pop_<r>.npy (forecasts, 8,112,762 pixels,
members in the order of the member ids), pop_ghap_<r>.npy, cell40_<r>.npy, pop_idx.npy.
    srun ... python eval19.py --plan main <run> --ckpt <run>/ckpts/epoch=01.ckpt --psub 16 --steps 4 --refine-tail 1
             --members 4 --pop --every 2 --leads 1 3 --out <a19>/data/pop_neg --rho 0.59 --base philox"""
import noise19 as nz  # noqa: I001  (first: audit hooks, numpy 1.26.3, no bytecode)
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from noise19 import WindowsAR19, asave, asavez
from eval_global_green import (FH, FW, MU, RAMP, REC_COLS, REGIONS, SD, THR, THR_ANN, G, GH, GW, FullModuleGreen, Npy,  # noqa: F401
                               TestGreen, block_mean, cell_mean, crop_px, full_map, member_particles, ramp, region_row,
                               region_slices, sample, windows)

an = nz.an
RESUME = ("recs", "seams", "keys", "st_ens", "st_ghap", "map25", "map25_ghap", "map25_sd", "map25_m0", "map25_mk", "keys_rho")


def start(a, n_all, world, rank, info):
    """A19 gates of a run: plan size and layout, folder claim (a19_run.json), resume state of this rank"""
    plan = nz.PLANS[a.plan]
    assert n_all == plan["n"], f"A19 plan {a.plan}: {n_all} forecasts, the plan has {plan['n']}"
    assert world == plan["world"], f"A19 plan {a.plan}: {world} tasks, the plan has {plan['world']}"
    meta = nz.manifest(a.plan, nz.args_of(a), a.rho, a.base, a.year, world)
    out = nz.claim(Path(a.run) / a.out, meta, fresh=a.plan in nz.SMOKE)
    meta_txt = json.dumps(nz.jsonable(meta), sort_keys=True)
    prev = {k: [] for k in RESUME + ("pop_ens", "pop_tgt", "pop_40")}
    prev["done"], jobs = set(), []
    p = out / f"part_{rank}.npz"
    if p.exists():
        with np.load(p) as z:
            assert [str(x) for x in z["info"]] == info, (p, z["info"], info)
            assert str(z["a19"]) == meta_txt, f"A19: {p} was written by another run or code version"
            assert [int(m) for m in z["member_ids"]] == [int(m) for m in a.mids], (p, z["member_ids"])
            assert str(z["base"]) == a.base and all(float(r) == a.rho for r in z["keys_rho"]), p
            for k in RESUME:
                prev[k] = list(z[k])
            jobs = [str(x) for x in z["a19_jobs"]]
            for d, h, _ in z["keys"].tolist():
                prev["done"].add((int(d), int(h)))
        n = len(prev["keys"])
        if a.pop:                                                  # written before the part file: may hold more rows
            for k, f in (("pop_ens", f"pop_{rank}.npy"), ("pop_tgt", f"pop_ghap_{rank}.npy"), ("pop_40", f"cell40_{rank}.npy")):
                arr = np.load(out / f)
                assert len(arr) >= n, (f, len(arr), n)
                prev[k] = list(arr[:n])
        print(f"rank {rank}: resume {p.name}: {n} forecasts already done", flush=True)
    job = os.environ.get("SLURM_JOB_ID", "none")
    fields = dict(a19=np.array(meta_txt), a19_jobs=np.array(jobs + ([job] if job not in jobs else [])),
                  member_ids=np.array([int(m) for m in a.mids], np.int64))
    return out, prev, fields


def run19(a):  # A19-H1
    rank, world, local = (int(os.environ.get(k, 0)) for k in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"))
    world = max(world, 1)
    name = Path(a.run).name
    no_emis = "_noemis" in name
    no_ant = "_noant" in name
    m_ = re.search(r"_windshift(\d+)", name); wind_shift = int(m_.group(1)) if m_ else 0
    ck = Path(a.ckpt) if a.ckpt else Path(a.run) / "final.ckpt"
    state = torch.load(ck, map_location="cpu", weights_only=False)
    fp = any(k.startswith("net.fp.") for k in state["state_dict"])
    green = any(k.startswith("net.enc.") for k in state["state_dict"]) and not fp
    assert green, "eval_global_green.py evaluates Foehn-Green checkpoints (use eval_global_fp.py otherwise)"
    assert state["hyper_parameters"]["max_lead"] == a.max_lead, state["hyper_parameters"]
    hp = state["hyper_parameters"]
    store, emis = a.store or hp.get("store", "green"), bool(hp.get("emis", False))   # --store: e.g. more particles
    P = int(Npy(G / store / "hres" / f"origins_{a.year}.npy").shape[-2])
    assert a.psub <= P, (a.psub, P)
    ds = TestGreen(a.max_lead, P, store, emis, a.year, hp.get("emis_store"))
    win = win_all = windows(np.load(G / "meta.npz")["land"])
    if a.crop_only:                              # case studies: only the windows that overlap the crop boxes (same noise)
        boxes = [crop_px(*c) for c in a.crop]
        win = [(k0, j0) for k0, j0 in win_all if any(25 * k0 < r1 and 25 * k0 + FH > r0 and 25 * j0 < c1 and 25 * j0 + FW > c0
                                                     for r0, r1, c0, c1 in boxes)]
    starts = sorted({i for i, h in ds.samples})
    starts = [i for i in starts if int(ds.doy[i]) in set(a.doys)] if a.doys else starts[::a.every]
    have, leads = set(ds.samples), a.leads or list(range(1, a.max_lead + 1))
    todo = [(i, h) for i in starts for h in leads if (i, h) in have]
    n_all, todo = len(todo), todo[rank::world]
    out, prev, a16f = start(a, n_all, world, rank, [name, ck.name, str(fp), str(no_emis), str(wind_shift), str(a.members), str(a.steps)])  # A16-H3
    todo = [t for t in todo if (int(ds.doy[t[0]]), t[1]) not in prev["done"]]  # A16-H3
    lags = a.lags or [0] * a.members
    assert len(lags) == a.members, (lags, a.members)
    for l in sorted(set(lags) - {0}):
        ds.add_lag(l, a.year)
    loader = DataLoader(WindowsAR19(ds, [(i, h, w) for i, h in todo for w in range(len(win))], win, a.mids, a.rho), batch_size=None,  # A19-H2
                        num_workers=a.workers, prefetch_factor=2 if a.workers else None)
    batches = iter(loader)                                           # workers start reading before CUDA is initialised
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    cls = FullModuleGreen
    model = cls.load_from_checkpoint(ck, map_location="cpu").to(dev).eval()
    if rank == 0:
        print(f"{name} | {ck.name} (epoch {state.get('epoch')}) | FLEXGREEN store {store}: {P} particles, {a.psub} per member"
              f"{' + emission term' if emis else ''}{' | NO TRANSPORT (ablation)' if hp.get('no_transport') else ''}"
              f"{' | no emissions' if no_emis else ''}{' | fires only (no anthropogenic inventory)' if no_ant else ''}"
              f"{f' | path weather shifted {wind_shift} d' if wind_shift else ''}"
              f"{' | DIAGNOSTIC: ERA5 weather after day d' if a.era5 else ''} | "
              f"{len(win)} windows, {a.members} members, {a.steps} steps | {n_all} forecasts on {world} GCDs",
              flush=True)
    del state

    FT, STT = Npy(G / "ghap1km_tiles.npy"), Npy(G / "static1km_tiles.npy")
    land = full_map(STT, 1) > 0
    wgt = np.outer(ramp(FH, RAMP), ramp(FW, RAMP))
    wgt_d = torch.from_numpy(wgt).to(dev)
    wsum = np.zeros((GH, GW), np.float32)
    for k0, j0 in win_all:
        wsum[25 * k0:25 * k0 + FH, 25 * j0:25 * j0 + FW] += wgt
    covered = wsum > 0
    inv_w = np.where(covered, 1.0 / np.maximum(wsum, 1e-12), 0.0).astype(np.float32)
    if a.crop_only:                              # keep the pixels whose every window was run: there the blend is the global one
        for k0, j0 in win:
            wsum[25 * k0:25 * k0 + FH, 25 * j0:25 * j0 + FW] -= wgt
        covered &= wsum <= 1e-4
    del wsum
    lc = land & covered
    regs = {} if a.crop_only else region_slices()
    sr = sc = np.zeros(0, int)  # A19-H4
    s_ok = np.zeros(0, bool)  # A19-H4
    if a.year != 2022:                              # stations only evaluate 2022: nothing stored for another year
        s_ok[:] = False
    edges_r = sorted({k * 25 for k, _ in win} | {k * 25 + FH for k, _ in win})
    edges_c = sorted({j * 25 for _, j in win} | {j * 25 + FW for _, j in win})
    edges_r = [e for e in edges_r if 0 < e < GH - 1]; edges_c = [e for e in edges_c if 0 < e < GW - 1]
    pop_idx = np.load(G / "pop_pixels.npz")["idx"] if a.pop else None           # populated pixels + station pixels
    if a.pop and rank == 0:
        asave(out / "pop_idx.npy", pop_idx)  # A16-H4
    pop_ens, pop_tgt, pop_40 = prev["pop_ens"], prev["pop_tgt"], prev["pop_40"]  # A16-H3
    ann_idx = np.flatnonzero(lc) if a.annual else None                 # --annual: sums per lead over the land pixels
    if a.annual and rank == 0:
        np.save(out / "annual_idx.npy", ann_idx)
    ann = {h: {k: np.zeros(len(ann_idx), np.float32 if k in ("f", "g") else np.uint16) for k in ("f", "g", "n", "xf", "xg")}
           for h in sorted({h for _, h in todo})} if a.annual else {}
    acc = np.zeros((a.members, GH, GW), np.float16 if a.members > 8 else np.float32)   # 16 members: 17 GB instead of 33 GB per GCD
    recs, seams, keys, st_ens, st_ghap, m25, g25, sd25, m025 = [], [], [], [], [], [], [], [], []
    m25k, krho, mk_m = [], [], (a.members if a.members <= 4 else 0)  # A16-H5
    for lst, k_ in zip((recs, seams, keys, st_ens, st_ghap, m25, g25, sd25, m025, m25k, krho), RESUME):  # A16-H3
        lst.extend(prev[k_])  # A16-H3
    t_start = time.time()

    def save():
        if a.pop:  # A16-H4 (first: the part file is the record of the forecasts done)
            asave(out / f"pop_{rank}.npy", np.stack(pop_ens))  # A16-H4
            asave(out / f"pop_ghap_{rank}.npy", np.stack(pop_tgt)); asave(out / f"cell40_{rank}.npy", np.stack(pop_40))  # A16-H4
        asavez(out / f"part_{rank}.npz", recs=np.array(recs, np.float64).reshape(-1, len(REC_COLS)), rec_cols=np.array(REC_COLS),  # A16-H4
                 seams=np.array(seams, np.float64).reshape(-1, 3), keys=np.array(keys, np.int32).reshape(-1, 3),
                 st_ens=np.array(st_ens, np.float16).reshape(len(keys), len(sr), a.members),  # A19-H4
                 st_ghap=np.array(st_ghap, np.float16).reshape(len(keys), len(sr)),  # A19-H4
                 map25=np.array(m25, np.float16).reshape(-1, GH // 25, GW // 25),
                 map25_ghap=np.array(g25, np.float16).reshape(-1, GH // 25, GW // 25),
                 map25_sd=np.array(sd25, np.float16).reshape(-1, GH // 25, GW // 25),
                 map25_m0=np.array(m025, np.float16).reshape(-1, GH // 25, GW // 25),
                 map25_mk=np.array(m25k, np.float16).reshape(len(keys), mk_m, GH // 25, GW // 25),  # A16-H5
                 keys_rho=np.array(krho, np.float64).reshape(-1), base=np.array(a.base), **a16f,  # A16-H5
                 windows=np.array(win), regions=np.array(list(REGIONS)), thresholds=np.array(THR),
                 info=np.array([name, ck.name, str(fp), str(no_emis), str(wind_shift), str(a.members), str(a.steps)]))
        if a.annual:                                                  # the forecasts summed are those of keys
            np.savez(out / f"annual_{rank}.npz", leads=np.array(sorted(ann)), thr=np.array(THR_ANN),
                     **{f"{k}_{h}": v for h, A in ann.items() for k, v in A.items()})

    for n_done, (i, h) in enumerate(todo):
        d0, dt = int(ds.doy[i]), int(ds.doy[i + h])
        acc.fill(0.0)
        t_io = t_gpu = 0.0
        for (k0, j0) in win:
            t = time.time(); b = next(batches); t_io += time.time() - t
            t = time.time()
            r0, c0 = 25 * k0, 25 * j0
            bt = {k: (v[None] if v.dim() > 0 else v.reshape(1)).to(dev, non_blocking=True) for k, v in b.items() if k != "x0"}
            base = {k: v for k, v in bt.items() if not k.startswith("lag")}
            for m in range(a.members):
                bm = dict(base)
                if lags[m] > 0:                                          # this member's weather run and its transport
                    l = lags[m]
                    bm["coarse"] = base["coarse"].clone(); bm["coarse"][:, -31:] = bt[f"lag{l}_coarse_fc"]
                    for k in ("org", "fsens", "esens"):
                        if f"lag{l}_{k}" in bt:
                            bm[k] = bt[f"lag{l}_{k}"]
                org_m = bm["org"]
                bm["org"] = org_m[..., torch.from_numpy(member_particles(i, h, a.mids[m], org_m.shape[-2], a.psub)).to(dev), :]  # A19-H3
                x0 = b["x0"][m].to(dev).float()[None, None]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = sample(model, bm, x0, a.steps, a.refine_tail, a.solver)
                acc[m, r0:r0 + FH, c0:c0 + FW] += (wgt_d * (x[0, 0, :FH, :FW].float() * SD + MU)).cpu().numpy()
            t_gpu += time.time() - t
        t = time.time()
        acc *= inv_w                                                  # blended members (0 outside the windows)
        mean = acc.mean(0, dtype=np.float32)                          # float32 even when acc is float16 (16 members)
        tgt = full_map(FT, i + h).astype(np.float32)
        per = full_map(FT, i).astype(np.float32)
        ok = lc & (tgt > 0)
        if a.annual:
            A = ann[h]; v = mean.ravel()[ann_idx]; g = tgt.ravel()[ann_idx]; k = g > 0
            A["f"] += np.where(k, v, 0); A["g"] += np.where(k, g, 0); A["n"] += k
            A["xf"] += k & (v > THR_ANN); A["xg"] += k & (g > THR_ANN)
            del v, g, k
        rows = {}
        for rid, (rname, (rs, cs)) in enumerate(regs.items()):
            mk = ok[rs, cs]
            if not mk.any():
                continue
            rows[rname] = [d0, h, rid] + region_row(acc[:, rs, cs][:, mk], tgt[rs, cs][mk], per[rs, cs][mk], a.members)
            recs.append(rows[rname])
        mean[~covered] = np.nan
        jr = [np.nanmean(np.abs(mean[e] - mean[e - 1])[lc[e] & lc[e - 1]]) for e in edges_r]
        jc = [np.nanmean(np.abs(mean[:, e] - mean[:, e - 1])[lc[:, e] & lc[:, e - 1]]) for e in edges_c]
        dr = np.abs(np.diff(mean, axis=0))[lc[1:] & lc[:-1]]; dc = np.abs(np.diff(mean, axis=1))[lc[:, 1:] & lc[:, :-1]]
        seam = 0.5 * (np.nanmean(jr) / np.nanmean(dr) + np.nanmean(jc) / np.nanmean(dc))
        seams.append((d0, h, seam))
        keys.append((d0, h, dt))
        st_ens.append(np.where(s_ok[:, None], acc[:, sr, sc].T, np.nan))
        st_ghap.append(np.where(s_ok & (tgt[sr, sc] > 0), tgt[sr, sc], np.nan))
        m25.append(cell_mean(mean, ok)); g25.append(cell_mean(tgt, ok))
        sd = np.zeros((GH, GW), np.float32)                          # ensemble spread (1 km), member by member: acc.std
        if a.members > 1:                                             # would make a full float32 copy (33 GB at 16 members)
            for m_ in range(a.members):
                dm = acc[m_].astype(np.float32); dm -= mean; dm *= dm; sd += dm
            del dm
            np.sqrt(sd / a.members, out=sd)
        sd25.append(cell_mean(sd, ok)); del sd                                                     # cell means of the spread
        m025.append(cell_mean(acc[0].astype(np.float32), ok))                                       # one member (variance)
        m25k.append(np.stack([cell_mean(acc[k].astype(np.float32), ok) for k in range(mk_m)]) if mk_m else np.zeros((0, GH // 25, GW // 25), np.float16))  # A16-H5
        krho.append(a.rho)  # A16-H5
        if a.pop:
            pop_ens.append(acc.reshape(a.members, -1)[:, pop_idx].T.astype(np.float16))
            pop_tgt.append(np.where(tgt.ravel()[pop_idx] > 0, tgt.ravel()[pop_idx], np.nan).astype(np.float16))
            pop_40.append(np.stack([block_mean(mean, ok, 40), block_mean(tgt, ok, 40)]))
        for b_, (cn, cs_, cw, ce) in enumerate(a.crop or []):           # every member at 1 km over the crop boxes
            if d0 in a.crop_days:
                cr0, cr1, cc0, cc1 = crop_px(cn, cs_, cw, ce)
                asavez(out / f"crop{b_}_d{d0}_h{h}.npz", members=acc[:, cr0:cr1, cc0:cc1].astype(np.float16),  # A16-H4
                       box=np.array([cn, cs_, cw, ce]), rows=np.array([cr0, cr1]), cols=np.array([cc0, cc1]))  # A16-H4
        if d0 in a.save_1km:
            asave(out / f"map1km_d{d0}_h{h}.npy", mean.astype(np.float16))  # A16-H4
            asave(out / f"map1km_d{d0}_h{h}_member0.npy", np.where(covered, acc[0], np.nan).astype(np.float16))  # A16-H4
        t_post = time.time() - t
        rm = lambda r, c: np.sqrt(rows[r][REC_COLS.index(c)] / rows[r][3]) if r in rows else np.nan
        print(f"rank {rank}: start {d0} lead {h}: io wait {t_io:.0f}s gpu {t_gpu:.0f}s post {t_post:.0f}s | seam {seam:.3f} | RMSE "
              + " ".join(f"{r.split()[0]} {rm(r, 'sse'):.1f}/{rm(r, 'sse_persist'):.1f}" for r in ("Europe", "East Asia", "South Asia", "North America", "Africa"))
              + f" (model/persistence) | {n_done + 1}/{len(todo)} {time.time() - t_start:.0f}s", flush=True)
        if (n_done + 1) % 10 == 0 or n_done + 1 == len(todo):
            save()
    print("rank", rank, "done", len(todo), "forecasts", flush=True)


REMOVED_LINES = [
    'def run_ar22(a):  # A16-H1',
    'loader = DataLoader(WindowsAR22(ds, [(i, h, w) for i, h in todo for w in range(len(win))], win, a.members, lags, a.year, a.rho, a.base), batch_size=None,  # A16-H2',
    'bm["org"] = org_m[..., torch.from_numpy(member_particles(i, h, m, org_m.shape[-2], a.psub)).to(dev), :]',
    'stz = np.load(G / "stations_2022.npz")',
    'inside = (stz["lat"] < 84.0) & (stz["lat"] > -60.0)',
    'sr = np.clip(np.floor((84.0 - stz["lat"]) / 0.01).astype(int), 0, GH - 1)',
    'sc = np.clip(np.floor((stz["lon"] + 180.0) / 0.01).astype(int), 0, GW - 1)',
    's_ok = inside & covered[sr, sc]',
    'st_ens=np.array(st_ens, np.float16).reshape(-1, len(sr), a.members),',
    'st_ghap=np.array(st_ghap, np.float16).reshape(-1, len(sr)),',
    'if d0 % 7 == 0:',
    'asave(out / f"map5km_d{d0}_h{h}.npy", block_mean(mean, lc, 5))  # A16-H4',
]
HUNKS = (1, 2, 3, 4)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()                                  # the options of eval_ar22.py (A16 G2)
    ap.add_argument("run")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--refine-tail", type=int, default=1)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--doys", type=int, nargs="*", default=None)
    ap.add_argument("--leads", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-1km", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="eval_global")
    ap.add_argument("--crop", type=float, nargs=4, action="append", metavar=("N", "S", "W", "E"))
    ap.add_argument("--crop-days", type=int, nargs="*", default=[])
    ap.add_argument("--crop-only", action="store_true")
    ap.add_argument("--pop", action="store_true")
    ap.add_argument("--psub", type=int, default=16)
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--plan", required=True, choices=sorted(nz.PLANS))  # A19: the run (noise19.PLANS)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--base", required=True, choices=["philox", "stored"])
    a = ap.parse_args(argv)
    for k, v in dict(max_lead=7, era5=False, annual=False, lags=None, store=None, solver="euler").items():
        setattr(a, k, v)
    a.mids = tuple(int(m) for m in nz.PLANS[a.plan]["mids"])
    assert a.members == len(a.mids), (a.members, a.mids)
    nz.plan_match(a.plan, nz.args_of(a))
    return a


if __name__ == "__main__":
    args = parse_args()
    nz.gate(args.plan, args.rho, args.base)
    print("A19 eval19", json.dumps({"plan": args.plan, "mids": list(args.mids), "out": str(Path(args.run) / args.out)}), flush=True)
    run19(args)
