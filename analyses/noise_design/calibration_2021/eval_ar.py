#!/usr/bin/env python3
"""A15 evaluator (a15/A15_draft.md sections 4-5): eval_global_green.run with the evolving noise of arnoise.py.

run_ar(a) is a verbatim copy of inspect.getsource(eval_global_green.run); every changed or added line ends with '# A15-H<n>'
and every original line that does not survive verbatim is listed in REMOVED_LINES (check_cpu.py checks both):
  H1  the forecasts come from a fixed plan (--plan, constants below), items (start index, lead, rho) sorted by
      (lead, start day, rho), then todo[rank::world]
  H2  WindowsAR (arnoise.py) replaces Windows: x0[m] = tile_window(ARNoise(2021, doy, lead, rho, base), m, ...); no lags
  H3  no station file is opened: sr = sc = empty, s_ok = empty (st_ens / st_ghap have 0 columns)
  H4  output folder A15/data/<plan dir> (smoke_* -> data/smoke), part_<tag or rank>.npz
  H5  removed: --pop, --annual, --crop / --crop-only, --save-1km, the map5km_d* files and the per-crop files
  H6  per-member 0.25 deg cell means map25_mk (n, M, 576, 1440) float16 (member 0 = map25_m0), keys_rho, base
  H7  atomic part file (tmp, fsync, os.replace, fsync of the folder); resume: a part that exists is reloaded and its
      (start day, lead, rho) are skipped; its info (checkpoint, plan, base, tag) must be unchanged
  H8  the progress line shows the plan and rho
Everything else is the paper's evaluation, imported unchanged: model load, sample (Euler, 4 steps, bf16 autocast),
member_particles, the float32 accumulation and inverse blend weight, the region sums, seams, keys, map25, map25_ghap,
map25_sd, map25_m0.
    srun ... python eval_ar.py --plan smoke_stored|smoke_cost|calib|main [--tag T] [--workers 4|6]"""
import arnoise  # noqa: F401,I001  (first: audit hook against station / monitor files, numpy 1.26.3)
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from arnoise import A15, CAL, CONF, RUN, YEAR, WindowsAR, fsync_dir
from eval_global_green import (FH, FW, MU, RAMP, REC_COLS, REGIONS, SD, THR, G, GH, GW, FullModuleGreen, Npy, TestGreen,  # noqa: F401
                               cell_mean, full_map, member_particles, ramp, region_row, region_slices, sample, windows)

PLANS = {"smoke_stored": ("stored", "smoke"), "smoke_cost": ("philox", "smoke"), "calib": ("philox", "calib"),
         "main": ("philox", "main")}
RHO_GRID = (0.50, 0.70, 0.90)
RESUME = ("recs", "seams", "keys", "st_ens", "st_ghap", "map25", "map25_ghap", "map25_sd", "map25_m0", "map25_mk", "keys_rho")


def passed(name):
    p = A15 / "results" / name
    return p.exists() and json.loads(p.read_text()).get("all_passed") is True


def plan_items(plan):
    """(start day of year, lead, rho) of a plan, from fixed constants (draft sections 4.4, 4.5, 5)"""
    if plan == "smoke_stored":
        return [(9, 1, 1.0), (9, 2, 1.0)]
    if plan == "smoke_cost":
        return [(1, 7, 0.70)]
    assert passed("check_cpu.json") and passed("smoke_check.json"), "check_cpu.json and smoke_check.json must have passed"
    if plan == "calib":
        return [(d, 1, 0.70) for d in CAL] + [(d, 2, r) for d in CAL for r in RHO_GRID]
    assert plan == "main", plan
    rj = json.loads((A15 / "results" / "rho.json").read_text())
    assert rj["clipped"] is False, "rho* clipped: A15 stops before run M (UNDECIDED)"
    r = float(rj["rho_star"])
    assert 0.50 <= r <= 0.90 and abs(round(r, 2) - r) < 1e-12, r
    return [(d, h, r) for d in CONF for h in range(1, 8)]


def plan_todo(a, ds):
    assert a.year == YEAR == 2021 and a.max_lead == 7
    assert a.tag is None or int(os.environ.get("SLURM_NTASKS", 1)) == 1, "--tag is for one-task runs"
    items = plan_items(a.plan)
    assert len(set(items)) == len(items), "an item is repeated"
    starts = sorted({i for i, h in ds.samples})
    idx_of = {int(ds.doy[i]): i for i in starts}
    assert len(idx_of) == len(starts), "two start indices share a day of year"
    have = set(ds.samples)
    todo = []
    for d, h, r in items:
        i = idx_of[d]
        assert (i, h) in have and int(ds.doy[i + h]) == int(ds.doy[i]) + h <= 364, (d, h)
        todo.append((i, h, float(r)))
    todo.sort(key=lambda t: (t[1], int(ds.doy[t[0]]), t[2]))
    return todo


def plan_out(a):
    data = (A15 / "data").resolve()
    out = (data / PLANS[a.plan][1]).resolve()
    assert str(out).startswith(str(data) + os.sep), out
    return out


def load_part(path, info, a15):
    """resume: the lists of an existing part file and its done (start day, lead, rho); its info must be unchanged"""
    prev = {k: [] for k in RESUME}
    prev["done"], prev["a15"] = set(), a15
    if not path.exists():
        return prev
    with np.load(path) as z:
        assert [str(x) for x in z["info"]] == info, (path, z["info"], info)
        assert [str(x) for x in z["a15"]] == a15, (path, z["a15"], a15)
        for k in RESUME:
            prev[k] = list(z[k])
        for (d, h, _), r in zip(z["keys"].tolist(), z["keys_rho"].tolist()):
            prev["done"].add((int(d), int(h), float(r)))
    print(f"resume {path.name}: {len(prev['done'])} forecasts already done", flush=True)
    return prev


def run_ar(a):  # A15-H1
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
    todo = plan_todo(a, ds)  # A15-H1
    n_all, todo = len(todo), todo[rank::world]
    out = plan_out(a)  # A15-H4
    tag = str(rank) if a.tag is None else a.tag  # A15-H4
    prev = load_part(out / f"part_{tag}.npz", [name, ck.name, str(fp), str(no_emis), str(wind_shift), str(a.members), str(a.steps)], [str(ck), a.plan, a.base, tag])  # A15-H7
    todo = [t for t in todo if (int(ds.doy[t[0]]), t[1], t[2]) not in prev["done"]]  # A15-H7
    lags = [0] * a.members  # A15-H2
    assert len(lags) == a.members, (lags, a.members)
    for l in sorted(set(lags) - {0}):
        ds.add_lag(l, a.year)
    loader = DataLoader(WindowsAR(ds, [(i, h, w) for i, h, _ in todo for w in range(len(win))], win, a.members, [r for _, _, r in todo for _ in win], a.base), batch_size=None,  # A15-H2
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
    del wsum
    lc = land & covered
    regs = region_slices()  # A15-H5
    sr = sc = np.zeros(0, int)  # A15-H3
    s_ok = np.zeros(0, bool)  # A15-H3
    edges_r = sorted({k * 25 for k, _ in win} | {k * 25 + FH for k, _ in win})
    edges_c = sorted({j * 25 for _, j in win} | {j * 25 + FW for _, j in win})
    edges_r = [e for e in edges_r if 0 < e < GH - 1]; edges_c = [e for e in edges_c if 0 < e < GW - 1]
    out.mkdir(parents=True, exist_ok=True)  # A15-H4
    acc = np.zeros((a.members, GH, GW), np.float16 if a.members > 8 else np.float32)   # 16 members: 17 GB instead of 33 GB per GCD
    recs, seams, keys, st_ens, st_ghap, m25, g25, sd25, m025 = [], [], [], [], [], [], [], [], []
    m25k, krho = [], []  # A15-H6
    for lst, k_ in zip((recs, seams, keys, st_ens, st_ghap, m25, g25, sd25, m025, m25k, krho), RESUME):  # A15-H7
        lst.extend(prev[k_])  # A15-H7
    t_start = time.time()

    def save():
        tmp = out / f"part_{tag}.tmp.npz"  # A15-H7
        with open(tmp, "wb") as f_:  # A15-H7
            np.savez(f_, recs=np.array(recs, np.float64).reshape(-1, len(REC_COLS)), rec_cols=np.array(REC_COLS),  # A15-H7
                 seams=np.array(seams, np.float64).reshape(-1, 3), keys=np.array(keys, np.int32).reshape(-1, 3),
                 st_ens=np.array(st_ens, np.float16).reshape(len(keys), len(sr), a.members),  # A15-H3
                 st_ghap=np.array(st_ghap, np.float16).reshape(len(keys), len(sr)),  # A15-H3
                 map25=np.array(m25, np.float16).reshape(-1, GH // 25, GW // 25),
                 map25_ghap=np.array(g25, np.float16).reshape(-1, GH // 25, GW // 25),
                 map25_sd=np.array(sd25, np.float16).reshape(-1, GH // 25, GW // 25),
                 map25_m0=np.array(m025, np.float16).reshape(-1, GH // 25, GW // 25),
                 map25_mk=np.array(m25k, np.float16).reshape(-1, a.members, GH // 25, GW // 25),  # A15-H6
                 keys_rho=np.array(krho, np.float64).reshape(-1), base=np.array(a.base), a15=np.array(prev["a15"]),  # A15-H6
                 windows=np.array(win), regions=np.array(list(REGIONS)), thresholds=np.array(THR),
                 info=np.array([name, ck.name, str(fp), str(no_emis), str(wind_shift), str(a.members), str(a.steps)]))
            f_.flush(); os.fsync(f_.fileno())  # A15-H7
        os.replace(tmp, out / f"part_{tag}.npz"); fsync_dir(out)  # A15-H7

    for n_done, (i, h, rho) in enumerate(todo):  # A15-H1
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
        m25k.append(np.stack([cell_mean(acc[k].astype(np.float32), ok) for k in range(a.members)]))  # A15-H6
        krho.append(rho)  # A15-H6
        t_post = time.time() - t
        rm = lambda r, c: np.sqrt(rows[r][REC_COLS.index(c)] / rows[r][3]) if r in rows else np.nan
        print(f"rank {rank}: {a.plan} start {d0} lead {h} rho {rho:.2f}: io wait {t_io:.0f}s gpu {t_gpu:.0f}s post {t_post:.0f}s | seam {seam:.3f} | RMSE "  # A15-H8
              + " ".join(f"{r.split()[0]} {rm(r, 'sse'):.1f}/{rm(r, 'sse_persist'):.1f}" for r in ("Europe", "East Asia", "South Asia", "North America", "Africa"))
              + f" (model/persistence) | {n_done + 1}/{len(todo)} {time.time() - t_start:.0f}s", flush=True)
        if (n_done + 1) % 10 == 0 or n_done + 1 == len(todo):
            save()
    print("rank", rank, "done", len(todo), "forecasts", flush=True)


REMOVED_LINES = [
    'def run(a):',
    'if a.crop_only:                              # case studies: only the windows that overlap the crop boxes (same noise)',
    'boxes = [crop_px(*c) for c in a.crop]',
    'win = [(k0, j0) for k0, j0 in win_all if any(25 * k0 < r1 and 25 * k0 + FH > r0 and 25 * j0 < c1 and 25 * j0 + FW > c0',
    'for r0, r1, c0, c1 in boxes)]',
    'starts = sorted({i for i, h in ds.samples})',
    'starts = [i for i in starts if int(ds.doy[i]) in set(a.doys)] if a.doys else starts[::a.every]',
    'have, leads = set(ds.samples), a.leads or list(range(1, a.max_lead + 1))',
    'todo = [(i, h) for i in starts for h in leads if (i, h) in have]',
    'lags = a.lags or [0] * a.members',
    'loader = DataLoader(Windows(ds, [(i, h, w) for i, h in todo for w in range(len(win))], win, a.members, lags), batch_size=None,',
    'if a.crop_only:                              # keep the pixels whose every window was run: there the blend is the global one',
    'for k0, j0 in win:',
    'wsum[25 * k0:25 * k0 + FH, 25 * j0:25 * j0 + FW] -= wgt',
    'covered &= wsum <= 1e-4',
    'regs = {} if a.crop_only else region_slices()',
    'stz = np.load(G / "stations_2022.npz")',
    'inside = (stz["lat"] < 84.0) & (stz["lat"] > -60.0)',
    'sr = np.clip(np.floor((84.0 - stz["lat"]) / 0.01).astype(int), 0, GH - 1)',
    'sc = np.clip(np.floor((stz["lon"] + 180.0) / 0.01).astype(int), 0, GW - 1)',
    's_ok = inside & covered[sr, sc]',
    'if a.year != 2022:                              # stations only evaluate 2022: nothing stored for another year',
    's_ok[:] = False',
    'out = Path(a.run) / a.out; out.mkdir(exist_ok=True)',
    'pop_idx = np.load(G / "pop_pixels.npz")["idx"] if a.pop else None           # populated pixels + station pixels',
    'if a.pop and rank == 0:',
    'np.save(out / "pop_idx.npy", pop_idx)',
    'pop_ens, pop_tgt, pop_40 = [], [], []',
    'ann_idx = np.flatnonzero(lc) if a.annual else None                 # --annual: sums per lead over the land pixels',
    'if a.annual and rank == 0:',
    'np.save(out / "annual_idx.npy", ann_idx)',
    'ann = {h: {k: np.zeros(len(ann_idx), np.float32 if k in ("f", "g") else np.uint16) for k in ("f", "g", "n", "xf", "xg")}',
    'for h in sorted({h for _, h in todo})} if a.annual else {}',
    'np.savez(out / f"part_{rank}.npz", recs=np.array(recs, np.float64).reshape(-1, len(REC_COLS)), rec_cols=np.array(REC_COLS),',
    'st_ens=np.array(st_ens, np.float16).reshape(-1, len(sr), a.members),',
    'st_ghap=np.array(st_ghap, np.float16).reshape(-1, len(sr)),',
    'if a.pop:                                                     # (forecasts in the order of keys, pixels, members)',
    'np.save(out / f"pop_{rank}.npy", np.stack(pop_ens))',
    'np.save(out / f"pop_ghap_{rank}.npy", np.stack(pop_tgt)); np.save(out / f"cell40_{rank}.npy", np.stack(pop_40))',
    'if a.annual:                                                  # the forecasts summed are those of keys',
    'np.savez(out / f"annual_{rank}.npz", leads=np.array(sorted(ann)), thr=np.array(THR_ANN),',
    '**{f"{k}_{h}": v for h, A in ann.items() for k, v in A.items()})',
    'for n_done, (i, h) in enumerate(todo):',
    'if a.annual:',
    'A = ann[h]; v = mean.ravel()[ann_idx]; g = tgt.ravel()[ann_idx]; k = g > 0',
    'A["f"] += np.where(k, v, 0); A["g"] += np.where(k, g, 0); A["n"] += k',
    'A["xf"] += k & (v > THR_ANN); A["xg"] += k & (g > THR_ANN)',
    'del v, g, k',
    'if a.pop:',
    'pop_ens.append(acc.reshape(a.members, -1)[:, pop_idx].T.astype(np.float16))',
    'pop_tgt.append(np.where(tgt.ravel()[pop_idx] > 0, tgt.ravel()[pop_idx], np.nan).astype(np.float16))',
    'pop_40.append(np.stack([block_mean(mean, ok, 40), block_mean(tgt, ok, 40)]))',
    'if d0 % 7 == 0:',
    'np.save(out / f"map5km_d{d0}_h{h}.npy", block_mean(mean, lc, 5))',
    'for b_, (cn, cs_, cw, ce) in enumerate(a.crop or []):           # every member at 1 km over the crop boxes',
    'if d0 in a.crop_days:',
    'cr0, cr1, cc0, cc1 = crop_px(cn, cs_, cw, ce)',
    'np.savez(out / f"crop{b_}_d{d0}_h{h}.npz", members=acc[:, cr0:cr1, cc0:cc1].astype(np.float16),',
    'box=np.array([cn, cs_, cw, ce]), rows=np.array([cr0, cr1]), cols=np.array([cc0, cc1]))',
    'if d0 in a.save_1km:',
    'np.save(out / f"map1km_d{d0}_h{h}.npy", mean.astype(np.float16))',
    'np.save(out / f"map1km_d{d0}_h{h}_member0.npy", np.where(covered, acc[0], np.nan).astype(np.float16))',
    'print(f"rank {rank}: start {d0} lead {h}: io wait {t_io:.0f}s gpu {t_gpu:.0f}s post {t_post:.0f}s | seam {seam:.3f} | RMSE "',
]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=list(PLANS))
    ap.add_argument("--tag", default=None, help="part file suffix instead of the rank (one-task runs)")
    ap.add_argument("--workers", type=int, default=4, choices=[4, 6])
    a = ap.parse_args(argv)
    fixed = dict(run=str(RUN), ckpt=str(RUN / "ckpts" / "epoch=01.ckpt"), members=4, steps=4, refine_tail=1, psub=16,
                 solver="euler", max_lead=7, year=2021, store=None, era5=False, lags=None)
    for k, v in fixed.items():
        setattr(a, k, v)
    a.base = PLANS[a.plan][0]
    for k in ("pop", "annual", "crop", "crop_only", "crop_days", "save_1km", "doys", "leads", "every", "out"):
        assert not hasattr(a, k), k                              # H5: these options do not exist here
    return a


if __name__ == "__main__":
    run_ar(parse_args())
