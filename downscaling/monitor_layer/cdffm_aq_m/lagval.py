"""cdffm_aq_m.lagval -- A11 experiment M: validation of the lagged-monitor conditioning against the 1 km estimates.

One function, validate_lag, for the in-training validation of a --stn-lag run (train_aq, rank 0) and for
a11/stageM/val_lag.py. It scores several ARMS on the validation days and pixels of train_aq.validate (ValData: hold-out
days of 2021, pixels drawn uniformly from the mask with seed 0; truth = the store's 1 km estimate k, a hole gets zero
weight), with the same noise for every arm and lag: the sphere-noise parameters of a day come from the generator seed of
train_aq.validate, (seed, hold index), and are evaluated at the pixels (common random numbers; M members x Euler steps,
no prior, no guard: the sampler of the in-training validation of A2/A3).
  kind "lag"   a model with the lag option: the 63 features of the monitors of day t - lag (stations.lag_features)
  kind "a3"    a same-day model of A3 fed with the monitors of day t - lag (the first 41 features): the naive baseline
  kind "free"  a model without station input (its members do not depend on the lag)
and, for the "lag" and "a3" arms, the label "<name>|nostations": every monitor hidden (the single no-monitor state of the
network, lag-independent by construction).
A monitor value is never a truth here: monitors enter as inputs only, through LagStationSet (assimilation set, held-out
ids asserted, firewall on the held-out value files). The held-out SEL file of A3(d')-4 is never opened.
Scores (data_aq.Scores: fair CRPS, RMSE of the ensemble mean, spread/skill, ...; cos(lat) weights) by lag and by distance
class of the pixel to the nearest assimilation monitor with a valid value on day t - lag (the monitors the network is
given, no dropout): "all", "0-10km", "10-50km", "50-150km", ">150km" (lower bound exclusive, upper bound inclusive, so
"0-10km" is A3's near10km). A lag-independent arm is scored once per lag with that lag's classes.
"""
import numpy as np
import torch

from .data_aq import Scores, assemble_inputs, bilinear_log1p_at, doy_of, raw_to_device, z_to_ug
from .model_aq import draw_sphere_noise, eval_sphere_noise, integrate, noise_chunk
from .stations import LAG_MAX, STN_DIM, STN_DIM_LAG, lag_features

DIST_CLASSES = (("0-10km", 0.0, 10.0), ("10-50km", 10.0, 50.0), ("50-150km", 50.0, 150.0), (">150km", 150.0, float("inf")))
CLASS_NAMES = ("all",) + tuple(c[0] for c in DIST_CLASSES)
NOSTN = "|nostations"


def arm_kind(model):
    """lag | a3 | free from the model itself"""
    if not getattr(model, "stations", False):
        return "free"
    return "lag" if int(model.stn_dim) == STN_DIM_LAG else "a3"


def class_masks(dmin):
    """dmin (P,) km -> {class name: (P,) float mask}; a pixel with no monitor in its latitude band (inf) is '>150km'"""
    out = {"all": torch.ones_like(dmin)}
    for name, lo, hi in DIST_CLASSES:
        out[name] = ((dmin > lo) & (dmin <= hi)).float() if lo > 0 else (dmin <= hi).float()
    return out


def compact(sc):
    """the numbers of a Scores object that the lag tables keep (None when the class is empty)"""
    r = sc.summary()
    u, lg = r["ug"], r["log"]
    if not u["n"] > 0:
        return None
    out = {k: u[k] for k in ("n", "crps_fair", "rmse_ensmean", "rmse_1member", "mae_ensmean", "bias")}
    for k in ("spread", "spread_skill", "spread_skill_fortin"):
        if k in u:
            out[k] = u[k]
    out["log_crps_fair"], out["log_rmse_ensmean"] = lg["crps_fair"], lg["rmse_ensmean"]
    return out


@torch.no_grad()
def validate_lag(arms, store, pix, val, stations, lags, M, ode_steps, spec, seed, device, chunk=50000, baselines=False,
                 nostations=True, members_out=None, log=None):
    """arms: list of dict(name, model (eval mode, on device), kind lag|a3|free, grid, transport, flowmap=False);
    val: train_aq.ValData built WITHOUT a held-out file; stations: LagStationSet holding the years of the days t - lag;
    lags: the lags to score (0..5; 0 = the same-day analysis, a reference, not a forecast).
    members_out: a dict to receive {(label, lag, date): (M, P) float32 ug m-3} (small runs only).
    -> dict(lags, classes, scores {label: {lag: {class: compact}}}, pooled {label: Scores.summary() over the lags, class
    'all'}, baselines {name: Scores.summary()} when asked, info)"""
    assert val.held is None, "validate_lag: the validation set must be built without the held-out monitor file"
    lags = [int(x) for x in lags]
    assert lags and all(0 <= x <= LAG_MAX for x in lags), lags
    names = [a["name"] for a in arms]
    assert len(set(names)) == len(names) and not any(NOSTN in n for n in names), names
    for a in arms:
        assert a["kind"] == arm_kind(a["model"]), (a["name"], a["kind"], arm_kind(a["model"]))
        a["model"].eval()
    k = torch.from_numpy(val.pix).to(device)
    lat, lon, stat = pix.lat[k], pix.lon[k], pix.static[k]
    P = k.numel()
    chunks = [(a, min(P, a + chunk)) for a in range(0, P, chunk)]
    labels = {}
    for a in arms:
        labels[a["name"]] = M
        if nostations and a["kind"] != "free":
            labels[a["name"] + NOSTN] = M
    if baselines:
        labels["climatology_median"] = 1
        labels["cams_bilinear"] = 1
    S = {lb: {lg: {c: Scores(m_, device) for c in CLASS_NAMES} for lg in lags} for lb, m_ in labels.items()}
    pool = {lb: Scores(m_, device) for lb, m_ in labels.items()}
    n_assim = {lg: [] for lg in lags}
    cw_sum = {lg: {c: 0.0 for c in CLASS_NAMES} for lg in lags}
    finite = True
    for i, (t, d) in enumerate(zip(val.days, val.dates)):
        doy = torch.tensor([doy_of(d)], device=device)
        raw = raw_to_device(store.raw_inputs(d), device)
        mu, sig = pix.mu_sigma(k[None], doy)
        truth = torch.from_numpy(val.truth[i].astype(np.float32)).to(device).clamp(min=1.0)
        w = pix.w_area[k]
        if getattr(store, "has_holes", False):      # GHAP (PROTOCOL.md A2(a)): a hole (-1) gets zero weight
            w = w * torch.from_numpy(val.truth[i] >= 0).to(device)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) * 1_000_003 + int(t) * 1009 + 17)          # the seed of train_aq.validate
        params = draw_sphere_noise(M, 1, spec, device, gen)
        z0 = [eval_sphere_noise(params, lat[None, a:e].expand(M, -1), lon[None, a:e].expand(M, -1),
                                chunk=noise_chunk(M, spec)) for a, e in chunks]
        feat_of, cls_of = {}, {}
        for lg in lags:
            mon = stations.lagged(d, lg)                     # every valid assimilation monitor of day d - lg (no dropout)
            assert (d - mon["src"]).days == lg and mon["lag"] == lg
            f, dmin = lag_features(lat, lon, mon)
            feat_of[lg], cls_of[lg] = f, class_masks(dmin)
            n_assim[lg].append(int(mon["val"].numel()))
            for c, m_ in cls_of[lg].items():
                cw_sum[lg][c] += float((w * m_).sum())

        def add(label, lg, X):
            for c, m_ in cls_of[lg].items():
                S[label][lg][c].add(X, truth, w * m_)
            pool[label].add(X, truth, w)
            if members_out is not None:
                members_out[(label, lg, str(d))] = X.float().cpu().numpy()

        for a in arms:
            model = a["model"]
            feat = model.encode(assemble_inputs(raw, pix, a["grid"], a["transport"]))

            def sample(stn_all):
                ens = torch.empty(M, P, device=device)
                for ci, (s0, e0) in enumerate(chunks):
                    cond = model.condition(feat, lat[None, s0:e0], lon[None, s0:e0], stat[None, s0:e0], doy,
                                           stn=None if stn_all is None else stn_all[None, s0:e0])
                    z = integrate(model, z0[ci], cond.expand(M, -1, -1), ode_steps, flowmap=bool(a.get("flowmap", False)))
                    ens[:, s0:e0] = z[..., 0]
                return z_to_ug(ens, mu, sig)

            if a["kind"] == "free" or nostations:            # the lag-independent members: once per day
                X0 = sample(None)
                finite &= bool(torch.isfinite(X0).all())
                for lg in lags:
                    add(a["name"] if a["kind"] == "free" else a["name"] + NOSTN, lg, X0)
            if a["kind"] != "free":
                for lg in lags:
                    f = feat_of[lg]
                    X = sample(f if a["kind"] == "lag" else f[:, :STN_DIM])
                    finite &= bool(torch.isfinite(X).all())
                    add(a["name"], lg, X)
            del feat
        if baselines:
            Xc = z_to_ug(torch.zeros(1, P, device=device), mu, sig)
            Xb = bilinear_log1p_at(raw["cams"][0], lat, lon, store.grid_native)[None]
            for lg in lags:
                add("climatology_median", lg, Xc)
                add("cams_bilinear", lg, Xb)
        if log is not None:
            log(f"[validate_lag] {d} ({i + 1}/{len(val.days)}): monitors of day t - lag " +
                ", ".join(f"lag {lg}: {n_assim[lg][-1]}" for lg in lags))
    scores = {lb: {str(lg): {c: compact(S[lb][lg][c]) for c in CLASS_NAMES} for lg in lags} for lb in labels}
    nums = [v for lb in scores.values() for lg in lb.values() for c in lg.values() if c is not None for v in c.values()]
    out = dict(lags=lags, classes=list(CLASS_NAMES), scores=scores,
               pooled={lb: pool[lb].summary() for lb in labels if lb not in ("climatology_median", "cams_bilinear")},
               baselines={lb: pool[lb].summary() for lb in ("climatology_median", "cams_bilinear") if lb in labels},
               info=dict(days=[str(d) for d in val.dates], hold_idx=[int(t) for t in val.days], n_pixels=int(P), members=int(M),
                         ode_steps=int(ode_steps), seed=int(seed), n_assim_valid={str(lg): n_assim[lg] for lg in lags},
                         class_weight={str(lg): cw_sum[lg] for lg in lags}, arms={a["name"]: a["kind"] for a in arms},
                         members_finite=bool(finite), scores_finite=bool(np.isfinite(np.asarray(nums, np.float64)).all()),
                         pooled_note="pooled = the lags above added up, class 'all'; a lag-independent label enters once per lag",
                         truth="the store's 1 km estimate k (GHAP V1), holes excluded; no monitor value is a truth"))
    return out
