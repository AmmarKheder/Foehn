#!/usr/bin/env python3
"""A17 evaluator (a17/A17_draft.md sections 4-5): the A15 evaluator a15/eval_ar.run_ar with the opposite-noise members.

run17(a) is a verbatim copy of inspect.getsource(a15/eval_ar.run_ar) (itself eval_global_green.run with the marked hunks
A15-H1..H8: 2021 only, no station file, plan items, per-member 0.25 deg maps map25_mk, atomic resumable part files);
every changed or added line ends with '# A17-H<n>' and every line of run_ar that does not survive verbatim is listed in
REMOVED_LINES (check17.py DIFF):
  H1  the function name
  H2  WindowsAR17 (noise17.py) replaces WindowsAR: x0 stacks the members of the plan's member ids a.mids; member id m
      = 4..7 is -z(m - 4, d, h) of the A15/A16 evolving noise (rho* = 0.59), member id 0..3 is z(m, d, h)
  H3  the particles of the k-th member of the run are member_particles(i, h, a.mids[k]) (an 8-member run's particles)
  H4  the run identity stored in the part file also holds the member ids and 'A17'
  H5  part files hold a17 (that identity) and member_ids instead of a15
  H6  a CRC read-back of the temporary part file before it replaces the previous one
Lines kept verbatim call module-level helpers of THIS file: plan_todo (the A17 plans of noise17.PLANS and the gates:
check17 passed with the same code; for main also the smoke passed with the same code; rho* = 0.59), plan_out
(a17/data/<plan>, a smoke folder must start empty), load_part (resume: a part file that exists is reloaded and its
(start day, lead, rho) are skipped; its info and identity must be unchanged).
Everything else is the paper's evaluation, unchanged: model load, sample (Euler, 4 steps, bf16 autocast), the float32
accumulation and inverse blend weight, region sums (over the members of this run), seams, keys, map25 (mean of this run's
members), map25_ghap, map25_sd, map25_m0 (= its first member id), map25_mk (n, members, 576, 1440).
    srun ... python eval17.py --plan smoke_alone|smoke_eight|main [--tag T] [--workers 4|6]"""
import noise17 as nz  # noqa: I001  (first: audit hooks, numpy 1.26.3, no bytecode)
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from noise17 import A17, RHO_A17, YEAR, WindowsAR17, fsync_dir, npz_readback
from eval_global_green import (FH, FW, MU, RAMP, REC_COLS, REGIONS, SD, THR, G, GH, GW, FullModuleGreen, Npy, TestGreen,  # noqa: F401
                               cell_mean, full_map, member_particles, ramp, region_row, region_slices, sample, windows)

RUN = nz.an.RUN
RESUME = ("recs", "seams", "keys", "st_ens", "st_ghap", "map25", "map25_ghap", "map25_sd", "map25_m0", "map25_mk", "keys_rho")
PART_KEYS = ("recs", "rec_cols", "seams", "keys", "st_ens", "st_ghap", "map25", "map25_ghap", "map25_sd", "map25_m0", "map25_mk",
             "keys_rho", "base", "a17", "member_ids", "windows", "regions", "thresholds", "info")
PART_NAME = re.compile(r"part_\d+\.npz(\.tmp_a17_\d+)?|part_\d+\.tmp\.npz")


def plan_items(plan):
    """(start day of year, lead, rho*) of a plan (noise17.PLANS), after the gates"""
    rho = nz.gate(plan)
    return [(d, h, rho) for d, h in nz.PLANS[plan]["items"]]


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
    data = nz.DATA.resolve()
    out = (data / a.plan).resolve()
    assert str(out).startswith(str(data) + os.sep), out
    if out.exists():
        names = sorted(os.listdir(out))
        bad = [n for n in names if not PART_NAME.fullmatch(n)]
        assert not bad, f"A17: {out} holds files that A17 does not write: {bad[:10]}"
        assert a.plan == "main" or not names, f"A17: the smoke folder {out} must be empty: {names[:10]}"
    return out


def load_part(path, info, ident):
    """resume: the lists of an existing part file and its done (start day, lead, rho); info and identity unchanged"""
    prev = {k: [] for k in RESUME}
    prev["done"], prev["a17"] = set(), ident
    if not path.exists():
        return prev
    with np.load(path) as z:
        assert [str(x) for x in z["info"]] == info, (path, z["info"], info)
        assert [str(x) for x in z["a17"]] == ident, (path, z["a17"], ident)
        assert [int(m) for m in z["member_ids"]] == [int(m) for m in ident[4].split()], (path, z["member_ids"])
        for k in RESUME:
            prev[k] = list(z[k])
        for (d, h, _), r in zip(z["keys"].tolist(), z["keys_rho"].tolist()):
            prev["done"].add((int(d), int(h), float(r)))
    print(f"resume {path.name}: {len(prev['done'])} forecasts already done", flush=True)
    return prev


def rho_one(todo):
    rs = {float(r) for _, _, r in todo} or {RHO_A17}
    assert rs == {RHO_A17}, rs
    return RHO_A17


def run17(a):  # A17-H1
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
    prev = load_part(out / f"part_{tag}.npz", [name, ck.name, str(fp), str(no_emis), str(wind_shift), str(a.members), str(a.steps)], [str(ck), a.plan, a.base, tag, " ".join(str(m) for m in a.mids), "A17"])  # A17-H4
    todo = [t for t in todo if (int(ds.doy[t[0]]), t[1], t[2]) not in prev["done"]]  # A15-H7
    lags = [0] * a.members  # A15-H2
    assert len(lags) == a.members, (lags, a.members)
    for l in sorted(set(lags) - {0}):
        ds.add_lag(l, a.year)
    loader = DataLoader(WindowsAR17(ds, [(i, h, w) for i, h, _ in todo for w in range(len(win))], win, a.mids, rho_one(todo)), batch_size=None,  # A17-H2
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
                 keys_rho=np.array(krho, np.float64).reshape(-1), base=np.array(a.base), a17=np.array(prev["a17"]), member_ids=np.array(a.mids, np.int64),  # A17-H5
                 windows=np.array(win), regions=np.array(list(REGIONS)), thresholds=np.array(THR),
                 info=np.array([name, ck.name, str(fp), str(no_emis), str(wind_shift), str(a.members), str(a.steps)]))
            f_.flush(); os.fsync(f_.fileno())  # A15-H7
        npz_readback(tmp, PART_KEYS)  # A17-H6
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
                bm["org"] = org_m[..., torch.from_numpy(member_particles(i, h, a.mids[m], org_m.shape[-2], a.psub)).to(dev), :]  # A17-H3
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
    'def run_ar(a):  # A15-H1',
    'prev = load_part(out / f"part_{tag}.npz", [name, ck.name, str(fp), str(no_emis), str(wind_shift), str(a.members), str(a.steps)], [str(ck), a.plan, a.base, tag])  # A15-H7',
    'loader = DataLoader(WindowsAR(ds, [(i, h, w) for i, h, _ in todo for w in range(len(win))], win, a.members, [r for _, _, r in todo for _ in win], a.base), batch_size=None,  # A15-H2',
    'bm["org"] = org_m[..., torch.from_numpy(member_particles(i, h, m, org_m.shape[-2], a.psub)).to(dev), :]',
    'keys_rho=np.array(krho, np.float64).reshape(-1), base=np.array(a.base), a15=np.array(prev["a15"]),  # A15-H6',
]
HUNKS = (1, 2, 3, 4, 5, 6)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=list(nz.PLANS))
    ap.add_argument("--tag", default=None, help="part file suffix instead of the rank (one-task runs)")
    ap.add_argument("--workers", type=int, default=4, choices=[4, 6])
    a = ap.parse_args(argv)
    fixed = dict(run=str(RUN), ckpt=str(RUN / "ckpts" / "epoch=01.ckpt"), steps=4, refine_tail=1, psub=16,
                 solver="euler", max_lead=7, year=2021, store=None, era5=False, lags=None, base="philox")
    for k, v in fixed.items():
        setattr(a, k, v)
    a.mids = tuple(nz.PLANS[a.plan]["mids"])
    a.members = len(a.mids)
    for k in ("pop", "annual", "crop", "crop_only", "crop_days", "save_1km", "doys", "leads", "every", "out"):
        assert not hasattr(a, k), k
    return a


if __name__ == "__main__":
    args = parse_args()
    print("A17 eval17", json.dumps({"plan": args.plan, "mids": args.mids, "workers": args.workers, "a17": str(A17)}), flush=True)
    run17(args)
