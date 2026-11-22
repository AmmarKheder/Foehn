#!/usr/bin/env python3
"""A16 evaluator (protocol_amendments/A16_test_year_evolving_noise.md, runs G1-G9 and smoke steps a, b4, c):
eval_global_green.run with ONLY the initial noise replaced by the evolving noise of A15 (arnoise22.py).

run_ar22(a) is a verbatim copy of inspect.getsource(eval_global_green.run); every changed or added line ends with
'# A16-H<n>' and every original line that does not survive verbatim is listed in REMOVED_LINES (check_cpu.py checks both):
  H1  the function name
  H2  WindowsAR22 replaces Windows: x0[m] = tile_window(ARNoise(year, ds.doy[i], h, rho, base, members), m, ...)
  H3  start(): the plan's forecast count and GCD count, the output-folder claim (a16_run.json: plan, arguments, rho, base,
      year, code hashes; a folder holding a file A16 does not write, or another run's manifest, is refused) and the resume
      state of this rank (a part file that exists is reloaded and its (start day, lead) are skipped)
  H4  every file is written atomically (temporary file, fsync, read-back, rename); the population files before the part
      file, which is the record of the forecasts done
  H5  part files also hold keys_rho, base, a16 (the manifest), a16_jobs and, for 4 members, map25_mk (n, 4, 576, 1440)
      float16, the 0.25 deg cell means of every member (member 0 = map25_m0), as A15's runs
Everything else is the stored evaluation, unchanged: forecast list and its split over ranks (--every / --doys / --leads,
todo[rank::world]), stations_2022.npz read for pixel coordinates exactly as before, --pop, --crop / --crop-only, --save-1km,
map5km files, model load, sampler (Euler, 4 steps, bf16 autocast), particles, blend, region sums, seams, maps.
Fixed (no stored command of F1-F9 sets them): --max-lead 7, no --era5, --annual, --lags, --store, --solver euler.
Every argument must equal the plan (arnoise22.EVAL_PLANS); gate(): rho* of a15/results/rho.json with base philox (rho 1
with base stored for smoke (a) only), check_cpu.json and, except in the smoke, smoke_check.json all_passed.
    srun ... python eval_ar22.py --plan G1 <run> --ckpt <run>/ckpts/epoch=01.ckpt --psub 16 --steps 4 --refine-tail 1
             --members 4 --every 2 --out eval_final_crps_e01_ar --rho 0.59 --base philox"""
import os  # FOEHN_WORK, FOEHN_ENVS: roots of the working trees and of the environments
# SHOW (natcomms/rebuild/figs_show, 2026-11-21): a marked copy of a16/eval_ar22.py for ONE forecast of run G1, start day
# index 38 (8 Feb 2022), lead 1, with its 1 km map saved. Changed or added lines end with "# SHOW-<n>":
#   SHOW-1  A16's arnoise22 is imported UNCHANGED from a16/ (this folder holds no copy): noise, gate(), claim(), writes
#   SHOW-2  one added plan S38 = G3 (layout 8 GCDs, every argument, checkpoint, sampler, noise) with doys [38],
#           save_1km [38] and a new output folder under figs_show/run; nothing else differs from eval_ar22.py
import sys  # SHOW-1
sys.path.insert(0, (os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution/a16"))  # SHOW-1
import arnoise22 as an  # noqa: I001  (first: numpy 1.26.3)
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from arnoise22 import WindowsAR22, asave, asavez
from eval_global_green import (FH, FW, MU, RAMP, REC_COLS, REGIONS, SD, THR, THR_ANN, G, GH, GW, FullModuleGreen, Npy,  # noqa: F401
                               TestGreen, block_mean, cell_mean, crop_px, full_map, member_particles, ramp, region_row,
                               region_slices, sample, windows)

an.EVAL_PLANS["S38"] = an._p(out=(os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/figs_show/run/eval_show_d38_crps_e01_ar"),  # SHOW-2
                             doys=[38], leads=[1], save_1km=[38], n=1, world=8, stored="eval_final_crps_e01")  # SHOW-2
RESUME = ("recs", "seams", "keys", "st_ens", "st_ghap", "map25", "map25_ghap", "map25_sd", "map25_m0", "map25_mk", "keys_rho")


def start(a, n_all, world, rank, info):
    """A16 gates of a run: plan size and layout, folder claim, resume state of this rank"""
    plan = an.EVAL_PLANS[a.plan]
    assert n_all == plan["n"], f"A16 plan {a.plan}: {n_all} forecasts, the plan has {plan['n']}"
    assert world == plan["world"], f"A16 plan {a.plan}: {world} tasks, the plan has {plan['world']}"
    meta = an.manifest(a.plan, an.eval_args(a), a.rho, a.base, a.year, world, [Path(__file__).resolve()])
    out = an.claim(Path(a.run) / a.out, meta, fresh=a.plan in an.SMOKE)
    meta_txt = json.dumps(an.jsonable(meta), sort_keys=True)
    prev = {k: [] for k in RESUME + ("pop_ens", "pop_tgt", "pop_40")}
    prev["done"], jobs = set(), []
    p = out / f"part_{rank}.npz"
    if p.exists():
        with np.load(p) as z:
            assert [str(x) for x in z["info"]] == info, (p, z["info"], info)
            assert str(z["a16"]) == meta_txt, f"A16: {p} was written by another run or code version"
            assert str(z["base"]) == a.base and all(float(r) == a.rho for r in z["keys_rho"]), p
            for k in RESUME:
                prev[k] = list(z[k])
            jobs = [str(x) for x in z["a16_jobs"]]
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
    a16f = dict(a16=np.array(meta_txt), a16_jobs=np.array(jobs + ([job] if job not in jobs else [])))
    return out, prev, a16f


def run_ar22(a):  # A16-H1
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
    loader = DataLoader(WindowsAR22(ds, [(i, h, w) for i, h in todo for w in range(len(win))], win, a.members, lags, a.year, a.rho, a.base), batch_size=None,  # A16-H2
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
    stz = np.load(G / "stations_2022.npz")
    inside = (stz["lat"] < 84.0) & (stz["lat"] > -60.0)
    sr = np.clip(np.floor((84.0 - stz["lat"]) / 0.01).astype(int), 0, GH - 1)
    sc = np.clip(np.floor((stz["lon"] + 180.0) / 0.01).astype(int), 0, GW - 1)
    s_ok = inside & covered[sr, sc]
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
                 st_ens=np.array(st_ens, np.float16).reshape(-1, len(sr), a.members),
                 st_ghap=np.array(st_ghap, np.float16).reshape(-1, len(sr)),
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
                bm["org"] = org_m[..., torch.from_numpy(member_particles(i, h, m, org_m.shape[-2], a.psub)).to(dev), :]
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
        if d0 % 7 == 0:
            asave(out / f"map5km_d{d0}_h{h}.npy", block_mean(mean, lc, 5))  # A16-H4
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
    'def run(a):',
    'loader = DataLoader(Windows(ds, [(i, h, w) for i, h in todo for w in range(len(win))], win, a.members, lags), batch_size=None,',
    'out = Path(a.run) / a.out; out.mkdir(exist_ok=True)',
    'np.save(out / "pop_idx.npy", pop_idx)',
    'pop_ens, pop_tgt, pop_40 = [], [], []',
    'np.savez(out / f"part_{rank}.npz", recs=np.array(recs, np.float64).reshape(-1, len(REC_COLS)), rec_cols=np.array(REC_COLS),',
    'if a.pop:                                                     # (forecasts in the order of keys, pixels, members)',
    'np.save(out / f"pop_{rank}.npy", np.stack(pop_ens))',
    'np.save(out / f"pop_ghap_{rank}.npy", np.stack(pop_tgt)); np.save(out / f"cell40_{rank}.npy", np.stack(pop_40))',
    'np.save(out / f"map5km_d{d0}_h{h}.npy", block_mean(mean, lc, 5))',
    'np.savez(out / f"crop{b_}_d{d0}_h{h}.npz", members=acc[:, cr0:cr1, cc0:cc1].astype(np.float16),',
    'box=np.array([cn, cs_, cw, ce]), rows=np.array([cr0, cr1]), cols=np.array([cc0, cc1]))',
    'np.save(out / f"map1km_d{d0}_h{h}.npy", mean.astype(np.float16))',
    'np.save(out / f"map1km_d{d0}_h{h}_member0.npy", np.where(covered, acc[0], np.nan).astype(np.float16))',
]
HUNKS = (1, 2, 3, 4, 5)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()                                  # the options of eval_global_green.py used by F1-F9
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
    ap.add_argument("--plan", required=True, choices=sorted(an.EVAL_PLANS))  # A16: the run (arnoise22.EVAL_PLANS)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--base", required=True, choices=["philox", "stored"])
    a = ap.parse_args(argv)
    for k, v in dict(max_lead=7, era5=False, annual=False, lags=None, store=None, solver="euler").items():
        setattr(a, k, v)
    an.plan_match(a.plan, an.eval_args(a))
    return a


if __name__ == "__main__":
    args = parse_args()
    an.gate(args.plan, args.rho, args.base)
    run_ar22(args)
