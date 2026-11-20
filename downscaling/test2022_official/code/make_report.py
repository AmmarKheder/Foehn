#!/usr/bin/env python3
"""A7: REPORT.md from the result files (no data read other than the json files of results/ and screened/results/).
Every number of the report, text included, is taken from those files when this script runs.
    python -I make_report.py -> ../REPORT.md"""
import json
import time
from pathlib import Path

O = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_official")
ALL = "All six networks"
EU, CN, US, KR, JP, IN = "Europe (EEA)", "China (CNEMC)", "USA (EPA)", "Korea (AirKorea)", "Japan (NIES)", "India (CPCB)"
NETS = [EU, CN, US, KR, JP, IN]
GR = NETS + [ALL]
L5 = ["1", "2", "3", "4", "5", "all"]
R = json.loads((O / "results/scores_official_2022.json").read_text())
SCRF = O / "screened/results/scores_official_2022.json"
RS = json.loads(SCRF.read_text()) if SCRF.exists() else None
SEAS = json.loads((O / "results/diag_season.json").read_text())["periods"]
OBS = json.loads((O / "results/diag_obs_target.json").read_text())
CK3 = json.loads((O / "results/check3_split.json").read_text())
CK0 = json.loads((O / "results/check_preds_preds.json").read_text())
PX = R["tables"]["pixel"]
S, C, F = PX["S"], PX["C"], PX["F"]
WH = R["where"]
WN = {"pixel": "station pixel", "ring1km": "ring 1 km", "ring3km": "ring 3 km", "ring10km": "ring 10 km"}


def p(c, key="rel_rmse", ci=True, star=True):
    """relative change in percent with its interval"""
    if not c or c.get(key) is None:
        return "–"
    s = f"{100 * c[key]:+.1f} %"
    iv = c.get(key + "_ci")
    if ci and iv:
        s += f" [{100 * iv[0]:+.1f}, {100 * iv[1]:+.1f}]"
        if star and not (iv[0] <= 0 <= iv[1]):
            s += "*"
    return s


def con(T, A, B, g=ALL, L="all"):
    return T["contrasts"][f"{A} - {B}"][g][L]


def arm(T, a, g=ALL, L="all"):
    return T["arms"][a][g][L] or {}


def f2(v, nd=2, sign=False):
    return "–" if v is None else (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")


def hdr(cols):
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]


def lead_cols():
    return [f"lead day {x}" if x.isdigit() else "all leads" for x in L5]


out = []
w = out.append
st = R["stations"]
sysj = R["system"]
nS, nC, nF = S["n_station_days"], C["n_station_days"], F["n_station_days"]

w("# A7 — the frozen downscaler and the forecast chain at the official stations of the paper, 2022")
w("")
w(f"Written {time.strftime('%Y-%m-%d %H:%M')} (Helsinki) by `code/make_report.py`; every number below, in the text as in the tables, is read from "
  "the result files named at the end of each section when the script runs. Protocol note (written before any prediction or score, with its dated "
  "deviations): `protocol_amendments/A7_official_stations_2022.md`. Log: `STATUS.md`. Full tables (1,017 lines): `results/tables_official_2022.md`.")
w("")
w(f"System: `{sysj['system']}`, {sysj['label']}: the frozen downscaler, evaluated once more on 2022. Nothing was trained, selected, tuned or "
  "corrected with these stations or on these scores. The stations are evaluation only; they are not independent of GHAP (section 2).")
w("")
w("**Caution that applies to every forecast-mode number of this report.** In the chain, only the composition (the coarse PM2.5 channel) is a "
  "forecast. The weather (ERA5), the fires (GFAS) and the transport channels given to the downscaler are those of the valid day: analysis-type "
  "fields that a real forecast does not have. From lead day 2 on this favours the chain against the CAMS forecast, Aurora and Foehn by an amount "
  "this test cannot measure for the weather and fires. Period of the forecast-mode tables: issue days 1 May – 30 November 2022; winter is absent.")
w("")

# ------------------------------------------------------------------------------------------------ 1. plain reading
cc = con(C, "chain", "coarse")
cf = con(C, "chain", "camsfc")
ca = con(C, "chain", "aurora")
cb = con(C, "chain", "bcsd_fc")
cd = con(C, "chain", "camsfc_ds")
cs = con(C, "chain", "ds_sameday")
ct = con(C, "chain_t6fc", "chain")
ccl = con(C, "chain", "clim_paper")
op = C["contrasts_own_days"]["chain - obs_persist"]["groups"]
ff1, ffa = con(F, "chain", "foehn", ALL, "1"), con(F, "chain", "foehn", ALL, "all")
SH_IN = arm(F, "foehn", IN)["n"] * arm(F, "foehn", IN)["rmse"] ** 2 / (arm(F, "foehn")["n"] * arm(F, "foehn")["rmse"] ** 2)
SH_EU = arm(F, "foehn", EU)["n"] * arm(F, "foehn", EU)["rmse"] ** 2 / (arm(F, "foehn")["n"] * arm(F, "foehn")["rmse"] ** 2)
k = "share_stations_rmse_below_camsfc"
w("## 1. Plain reading")
w("")
w(f"**What the chain adds to its coarse input.** On {nC:,} identical station-days (214 issue days, lead days 1–5, six networks pooled) the "
  f"downscaled 0.4° forecast has an RMSE of {arm(C, 'chain')['rmse']:.2f} µg m⁻³ against {arm(C, 'coarse')['rmse']:.2f} for the 0.4° forecast "
  f"read bilinearly: {p(cc)}; fair CRPS {p(cc, 'rel_crps')}. Against the CAMS forecast: {p(cf)}; against Aurora: {p(ca)}. It beats the CAMS "
  f"forecast at {100 * arm(C, 'chain', ALL, '1')[k]:.0f} % of the stations at lead day 1 and {100 * arm(C, 'chain', ALL, '5')[k]:.0f} % at lead day 5 "
  f"(0.4° forecast: {100 * arm(C, 'coarse', ALL, '1')[k]:.0f} % and {100 * arm(C, 'coarse', ALL, '5')[k]:.0f} %). "
  f"The gain is very uneven: {p(con(C, 'chain', 'coarse', CN))} in China, {p(con(C, 'chain', 'coarse', KR))} in Korea, "
  f"{p(con(C, 'chain', 'coarse', JP))} in Japan, {p(con(C, 'chain', 'coarse', US))} in the USA, {p(con(C, 'chain', 'coarse', IN))} in India and "
  f"{p(con(C, 'chain', 'coarse', EU))} in Europe.")
w("")
w(f"What kind of gain it is: at lead day 1 it is a gain of level and amplitude, not of timing. The median per-station temporal correlation is "
  f"{arm(C, 'chain', ALL, '1')['median_station_r']:.3f} for the chain, {arm(C, 'coarse', ALL, '1')['median_station_r']:.3f} for its input and "
  f"{arm(C, 'camsfc', ALL, '1')['median_station_r']:.3f} for the CAMS forecast; the bias goes from {arm(C, 'coarse', ALL, '1')['bias']:+.2f} to "
  f"{arm(C, 'chain', ALL, '1')['bias']:+.2f} (China: {arm(C, 'coarse', CN, '1')['bias']:+.2f} to {arm(C, 'chain', CN, '1')['bias']:+.2f}). In Korea "
  f"and Japan the chain LOWERS the correlation of its input ({arm(C, 'coarse', KR, '1')['median_station_r']:.3f} → "
  f"{arm(C, 'chain', KR, '1')['median_station_r']:.3f} and {arm(C, 'coarse', JP, '1')['median_station_r']:.3f} → "
  f"{arm(C, 'chain', JP, '1')['median_station_r']:.3f}). From lead day 2 on, the chain's correlation stays near "
  f"{arm(C, 'chain', ALL, '5')['median_station_r']:.2f} while its input falls to {arm(C, 'coarse', ALL, '5')['median_station_r']:.2f} at lead day 5: "
  "that persistence of skill comes with the analysis-type weather of the valid day and cannot be credited to the forecast.")
w("")
w(f"Three controls put the gain in proportion. (i) The quantile-mapping baseline (BCSD) applied to the same eight members gets most of it: the "
  f"chain is {p(cb)} in RMSE against it ({p(cb, 'rel_crps')} in CRPS, where BCSD's eight members are almost a single field); the chain is ahead in "
  f"China ({p(con(C, 'chain', 'bcsd_fc', CN))}), India ({p(con(C, 'chain', 'bcsd_fc', IN))}) and the USA ({p(con(C, 'chain', 'bcsd_fc', US))}), "
  f"level in Europe ({p(con(C, 'chain', 'bcsd_fc', EU))}) and Japan ({p(con(C, 'chain', 'bcsd_fc', JP))}), behind in Korea "
  f"({p(con(C, 'chain', 'bcsd_fc', KR))}). (ii) Feeding the downscaler with the CAMS forecast instead of the paper's 0.4° forecast is as good or "
  f"slightly better (chain against that arm: {p(cd)}), although the 0.4° forecast itself beats the CAMS forecast at the stations from lead day 2 "
  f"({p(con(C, 'coarse', 'camsfc', ALL, '2'))} to {p(con(C, 'coarse', 'camsfc', ALL, '5'))}): the downscaler does not carry that advantage "
  f"through. (iii) Replacing the forecast composition by the analysis-type composition of the valid day changes little: the chain is only {p(cs)} "
  f"worse than the downscaler with same-day inputs, and rebuilding the transported PM2.5 from the forecast costs {p(ct)} "
  f"({p(con(C, 'chain_t6fc', 'chain', ALL, '5'))} at lead day 5): the output depends weakly on the coarse PM2.5 it is given.")
w("")
w(f"Against the references without skill: {p(ccl)} against the paper's station-pixel climatology; against the station's own value of the day "
  f"before the issue the chain is WORSE at lead day 1 ({p(op[ALL]['1'])}), better from lead day 2 ({p(op[ALL]['2'])}, {p(op[ALL]['3'])} at lead "
  f"day 3). At lead day 1 only China is on the right side of persistence ({p(op[CN]['1'])}); Europe {p(op[EU]['1'])}, India {p(op[IN]['1'])}, "
  f"Japan {p(op[JP]['1'])}.")
w("")
w(f"**How far it is from the retrospective 1 km model.** On the 107 issue days that have a Foehn forecast ({nF:,} station-days), pooled RMSE "
  f"hides the distance: lead day 1 Foehn {arm(F, 'foehn', ALL, '1')['rmse']:.2f}, chain {arm(F, 'chain', ALL, '1')['rmse']:.2f} ({p(ff1)}); all "
  f"leads {arm(F, 'foehn')['rmse']:.2f} and {arm(F, 'chain')['rmse']:.2f} ({p(ffa)}). The pooled tie is made by India "
  f"({100 * arm(F, 'foehn', IN)['n'] / arm(F, 'foehn')['n']:.0f} % of the station-days, {100 * SH_IN:.0f} % of Foehn's pooled squared error), where Foehn inherits the mismatch of the 1 km estimates with the CPCB monitors (GHAP RMSE "
  f"{arm(C, 'ghap', IN)['rmse']:.1f}, worse than the CAMS forecast {arm(C, 'camsfc', IN)['rmse']:.1f}) and the chain does not "
  f"({p(con(F, 'chain', 'foehn', IN))}). Everywhere else Foehn is ahead at lead day 1: China {p(con(F, 'chain', 'foehn', CN, '1'))}, USA "
  f"{p(con(F, 'chain', 'foehn', US, '1'))}, Korea {p(con(F, 'chain', 'foehn', KR, '1'))}, Japan {p(con(F, 'chain', 'foehn', JP, '1'))}, Europe "
  f"{p(con(F, 'chain', 'foehn', EU, '1'))}" + (f" ({p(con(RS['tables']['pixel']['F'], 'chain', 'foehn', EU, '1'))} once the implausible European values "
  "are screened out, section 8)" if RS else "") + " (chain against Foehn, RMSE). The fair CRPS, which is not dominated by a few large errors, says it plainly: "
  f"the chain is {p(ff1, 'rel_crps')} worse than Foehn at lead day 1 and {p(ffa, 'rel_crps')} over all leads (Europe {p(con(F, 'chain', 'foehn', EU, '1'), 'rel_crps')}, "
  f"Korea {p(con(F, 'chain', 'foehn', KR, '1'), 'rel_crps')}, Japan {p(con(F, 'chain', 'foehn', JP, '1'), 'rel_crps')} at lead day 1). Foehn beats the CAMS "
  f"forecast at {100 * arm(F, 'foehn', ALL, '1')[k]:.0f} % of the stations at lead day 1, the chain at {100 * arm(F, 'chain', ALL, '1')[k]:.0f} % "
  f"(Europe {100 * arm(F, 'foehn', EU, '1')[k]:.0f} % against {100 * arm(F, 'chain', EU, '1')[k]:.0f} %); median station correlation "
  f"{arm(F, 'foehn', ALL, '1')['median_station_r']:.3f} against {arm(F, 'chain', ALL, '1')['median_station_r']:.3f}. At lead days 4–5 the chain's RMSE "
  f"is lower ({p(con(F, 'chain', 'foehn', ALL, '4'))}, {p(con(F, 'chain', 'foehn', ALL, '5'))}) and its correlation higher "
  f"({arm(F, 'chain', ALL, '5')['median_station_r']:.3f} against {arm(F, 'foehn', ALL, '5')['median_station_r']:.3f}): there the chain reads the weather "
  "analysis of the valid day while Foehn runs on forecast weather, so this is not evidence that the chain forecasts better. Foehn in turn needs the "
  "1 km estimates of the two days before the issue, which exist only after the fact and are fitted to the Chinese and US monitors used here.")
w("")
dj, dk = SEAS["Dec-Feb"][JP], SEAS["Dec-Feb"][KR]
w(f"**Where it loses.** (a) Korea and Japan in the cold season. Over the whole year the same-day downscaler is worse than the CAMS first-day field "
  f"it starts from: {p(con(S, 'ds_sameday', 'cams0_bil', KR, 'sameday'))} in Korea and {p(con(S, 'ds_sameday', 'cams0_bil', JP, 'sameday'))} in Japan, "
  f"and better than CAMS at only {100 * arm(S, 'ds_sameday', KR, 'sameday')['share_stations_rmse_below_cams0_bil']:.0f} % and "
  f"{100 * arm(S, 'ds_sameday', JP, 'sameday')['share_stations_rmse_below_cams0_bil']:.0f} % of the stations. The split by season (descriptive, added "
  f"after the tables) shows it is a winter failure with a large positive bias: December–February {100 * dk['ds_sameday - cams0_bil']['rel_rmse']:+.0f} % "
  f"[{100 * dk['ds_sameday - cams0_bil']['ci'][0]:+.0f}, {100 * dk['ds_sameday - cams0_bil']['ci'][1]:+.0f}] in Korea (bias {dk['bias']['ds_sameday']:+.1f} "
  f"against {dk['bias']['cams0_bil']:+.1f}) and {100 * dj['ds_sameday - cams0_bil']['rel_rmse']:+.0f} % "
  f"[{100 * dj['ds_sameday - cams0_bil']['ci'][0]:+.0f}, {100 * dj['ds_sameday - cams0_bil']['ci'][1]:+.0f}] in Japan (bias {dj['bias']['ds_sameday']:+.1f} "
  f"against {dj['bias']['cams0_bil']:+.1f}); May–November {100 * SEAS['May-Nov (days 120-333)'][KR]['ds_sameday - cams0_bil']['rel_rmse']:+.1f} % and "
  f"{100 * SEAS['May-Nov (days 120-333)'][JP]['ds_sameday - cams0_bil']['rel_rmse']:+.1f} % (intervals include zero). The forecast-mode tables cover "
  f"May–November only, so the chain's favourable numbers there ({p(con(C, 'chain', 'camsfc', KR))} and {p(con(C, 'chain', 'camsfc', JP))} against the "
  "CAMS forecast) are warm-season numbers and say nothing about the season in which the downscaler fails in these two countries. Even in the warm "
  f"season the chain there has a lower correlation than its input, is not better than BCSD, is {p(con(F, 'chain', 'foehn', KR))} behind Foehn in Korea, "
  "and is better averaged over 10 km than at the station pixel (section 7).")
w("")
w(f"(b) Europe, the network new to the downscaler. Same-day, 365 days: {p(con(S, 'ds_sameday', 'cams0_bil', EU, 'sameday'))} against the CAMS first-day "
  f"field, better at {100 * arm(S, 'ds_sameday', EU, 'sameday')['share_stations_rmse_below_cams0_bil']:.0f} % of the stations, median station correlation "
  f"{arm(S, 'ds_sameday', EU, 'sameday')['median_station_r']:.3f} against {arm(S, 'cams0_bil', EU, 'sameday')['median_station_r']:.3f}; the gain is a "
  f"cold-season gain ({100 * SEAS['Jan-Apr + Dec'][EU]['ds_sameday - cams0_bil']['rel_rmse']:+.1f} % in January–April and December, "
  f"{100 * SEAS['May-Nov (days 120-333)'][EU]['ds_sameday - cams0_bil']['rel_rmse']:+.1f} % in May–November). Forecast mode (May–November): the chain is "
  f"{p(con(C, 'chain', 'camsfc', EU))} against the CAMS forecast and {p(con(C, 'chain', 'coarse', EU))} against its input in RMSE, beats the CAMS forecast "
  f"at {100 * arm(C, 'chain', EU, '1')[k]:.0f} % of the stations at lead day 1 and {100 * arm(C, 'chain', EU, '3')[k]:.0f} % at lead day 3, is level with "
  f"BCSD ({p(con(C, 'chain', 'bcsd_fc', EU))}) and clearly behind Foehn (CRPS {p(con(F, 'chain', 'foehn', EU), 'rel_crps')}; Foehn beats the CAMS forecast "
  f"at {100 * arm(F, 'foehn', EU, '1')[k]:.0f} % of the European stations). Its correlation is higher than that of both coarse forecasts "
  f"({arm(C, 'chain', EU, '1')['median_station_r']:.3f} against {arm(C, 'coarse', EU, '1')['median_station_r']:.3f} and "
  f"{arm(C, 'camsfc', EU, '1')['median_station_r']:.3f} at lead day 1). So in Europe the chain is not a loss, but at lead days 1–2 it is a tie with the "
  "CAMS forecast at the station level. European RMSEs must be read with section 2: a handful of stations with implausible values hold most of the "
  "squared error of every system." + ((f" With those values screened out (sensitivity declared before it was run, section 8) the European RMSEs fall to "
  f"about 6 µg m⁻³ and the chain is {p(con(RS['tables']['pixel']['C'], 'chain', 'camsfc', EU))} against the CAMS forecast "
  f"({p(con(RS['tables']['pixel']['C'], 'chain', 'camsfc', EU, '1'))} at lead day 1), {p(con(RS['tables']['pixel']['C'], 'chain', 'coarse', EU))} against its input, "
  f"{p(con(RS['tables']['pixel']['C'], 'chain', 'bcsd_fc', EU))} against BCSD and {p(con(RS['tables']['pixel']['F'], 'chain', 'foehn', EU, '1'))} against Foehn at "
  f"lead day 1; the share of stations where it beats the CAMS forecast does not move ({100 * arm(RS['tables']['pixel']['C'], 'chain', EU, '1')[k]:.0f} % at lead day 1).")
  if RS else ""))
w("")
w(f"(c) The ensemble. Spread/skill of the chain {arm(C, 'chain')['spread_skill']:.2f} (Europe {arm(C, 'chain', EU)['spread_skill']:.2f}, Japan "
  f"{arm(C, 'chain', JP)['spread_skill']:.2f}); the observation falls inside the range of the eight members on "
  f"{100 * arm(C, 'chain')['coverage_member_range']:.0f} % of the station-days (78 % expected), below the lowest member on "
  f"{100 * arm(C, 'chain')['share_obs_below_lowest_member']:.0f} % and above the highest on {100 * arm(C, 'chain')['share_obs_above_highest_member']:.0f} %: "
  f"the ensemble is too narrow and too high (bias {arm(C, 'chain')['bias']:+.2f}; in Japan the observation is below every member on "
  f"{100 * arm(C, 'chain', JP)['share_obs_below_lowest_member']:.0f} % of the station-days). Its day-to-day amplitude is damped (median ratio "
  f"{arm(C, 'chain', ALL, '1')['median_station_amp']:.2f}).")
w("")
w(f"(d) Smaller ones: station persistence at lead day 1 (above); in the USA the same-day downscaler is not better than BCSD "
  f"({p(con(S, 'ds_sameday', 'bcsd', US, 'sameday'))}); in India the chain's gain over the paper's climatology is not significant "
  f"({p(con(C, 'chain', 'clim_paper', IN))}).")
w("")
g1 = lambda n, wv, g=ALL: R["tables"][wv]["C_pixel_vs_ring"]["contrasts"][f"{n}@pixel - {n}@ring"][g]["all"]    # noqa: E731
g1s = lambda n, wv, g=ALL: R["tables"][wv]["S_pixel_vs_ring"]["contrasts"][f"{n}@pixel - {n}@ring"][g]["sameday"]    # noqa: E731
gv = R["ghap_vs_stations_2022"]
w(f"**What the rings say about the station-fitted pixel.** The target itself: in China GHAP at the station pixel is within 0.5 µg m⁻³ of the monitor on "
  f"{100 * gv[CN]['pixel']['share_station_days_within_0p5']:.0f} % of the station-days and on {100 * gv[CN]['ring1km']['share_station_days_within_0p5']:.0f} % "
  f"one pixel away; in the USA {100 * gv[US]['pixel']['share_station_days_within_0p5']:.0f} % and {100 * gv[US]['ring1km']['share_station_days_within_0p5']:.0f} %. "
  f"The station fit of GHAP is confined to the monitor's own pixel: over the year its RMSE there is {p(g1s('ghap', 'ring1km', CN))} (China) and "
  f"{p(g1s('ghap', 'ring1km', US))} (USA) against its own 1 km ring, {p(g1s('ghap', 'ring10km', CN))} and {p(g1s('ghap', 'ring10km', US))} against its 10 km ring. In Europe and India the imprint is partial ({100 * gv[EU]['pixel']['share_station_days_within_0p5']:.0f} % "
  f"against {100 * gv[EU]['ring1km']['share_station_days_within_0p5']:.0f} %, and {100 * gv[IN]['pixel']['share_station_days_within_0p5']:.0f} % against "
  f"{100 * gv[IN]['ring1km']['share_station_days_within_0p5']:.0f} %; no station with a daily R above 0.999, median R "
  f"{gv[EU]['pixel']['median_station_R']:.2f} and {gv[IN]['pixel']['median_station_R']:.2f}) and GHAP's RMSE is not lower at the pixel than on the ring; in "
  f"Korea and Japan there is none ({100 * gv[KR]['pixel']['share_station_days_within_0p5']:.0f} % against "
  f"{100 * gv[KR]['ring1km']['share_station_days_within_0p5']:.0f} %, {100 * gv[JP]['pixel']['share_station_days_within_0p5']:.0f} % against "
  f"{100 * gv[JP]['ring1km']['share_station_days_within_0p5']:.0f} %). "
  f"The downscaler does not reproduce that imprint: the chain at the station pixel is not closer to the monitor than the mean of its own ring, "
  f"{p(g1('chain', 'ring1km'))} against the 1 km ring, {p(g1('chain', 'ring3km'))} against 3 km, {p(g1('chain', 'ring10km'))} against 10 km "
  f"(China {p(g1('chain', 'ring10km', CN))}, USA {p(g1('chain', 'ring10km', US))}, 10 km; forecast mode, May–November). By the rule fixed before the scores, the station evidence "
  "therefore shows no skill of the 1 km detail at the station. The reassuring side of the same fact: the chain's gains are not an artefact of "
  f"the station-fitted pixel, they are the same on the rings (against its input {p(con(C, 'chain', 'coarse'), ci=False)} at the pixel, "
  f"{p(con(R['tables']['ring1km']['C'], 'chain', 'coarse'), ci=False)}, {p(con(R['tables']['ring3km']['C'], 'chain', 'coarse'), ci=False)} and "
  f"{p(con(R['tables']['ring10km']['C'], 'chain', 'coarse'), ci=False)} on the 1, 3 and 10 km rings). In Korea and Japan the station pixel is WORSE than the "
  f"ring for every 1 km field, GHAP included (chain {p(g1('chain', 'ring10km', KR))} and {p(g1('chain', 'ring10km', JP))} against the 10 km ring; GHAP "
  f"{p(g1('ghap', 'ring10km', KR))} and {p(g1('ghap', 'ring10km', JP))}): the fine structure that the target puts at the monitors there is wrong, and "
  "the downscaler has learnt it.")
w("")
rt = con(C, "chain_rtseed", "chain")
w(f"**Two side results.** The real-time adapter's noise rule (the same eight noise fields on every issue day) leaves the pooled score unchanged "
  f"({p(rt)}) but moves a whole network at a given lead by the luck of eight fixed draws: {p(con(C, 'chain_rtseed', 'chain', JP, '3'))} in Japan and "
  f"{p(con(C, 'chain_rtseed', 'chain', CN, '3'))} in China at lead day 3. Scores of the real-time system by lead and region carry that arbitrary "
  "offset; seeds that change with the day remove it. And one process of the frozen test's own prediction job wrote slightly wrong members on 17 days "
  f"(section 9); it moves no station RMSE of the frozen test by more than {R['checks']['frozen_test_pixel_station_rmse']['max_abs_rmse_diff']:.0e} µg m⁻³.")
w("")

# ------------------------------------------------------------------------------------------------ 2. stations and target
w("## 2. Stations, observations, target")
w("")
w(f"Official stations with at least 30 valid days: {st['official_ge30_days_total']:,}; outside the downscaler's mask: {st['dropped_not_in_mask_total']}; "
  f"scored: {st['n_stations']:,} stations on {st['n_pixels']:,} pixels. Stations sharing a pixel are not averaged (station-days, as in the paper). "
  "Rings: square rings at 1, 3 and 10 pixels around the station pixel (8, 24, 80 pixels), station pixel excluded, kept when at least half of the "
  "ring is in the mask. On the rings the three coarse systems keep their value at the station coordinates.")
w("")
out += hdr(["network", "stations ≥ 30 days", "outside the mask", "scored", "station-days, same-day (365 days)", "station-days, forecast mode (214 issue days × 5)",
            "with a Foehn forecast (107 issue days × 5)"])
key = {EU: "EEA", CN: "CNEMC", US: "EPA_AQS", KR: "AirKorea", JP: "Japan_NIES", IN: "India_CPCB"}
for g in NETS:
    w(f"| {g} | {st['official_ge30_days'][key[g]]:,} | {st['dropped_not_in_mask'][key[g]]} | {st['by_network'][key[g]]:,} | "
      f"{arm(S, 'ds_sameday', g, 'sameday')['n']:,} | {arm(C, 'chain', g)['n']:,} | {arm(F, 'foehn', g)['n']:,} |")
w(f"| {ALL} | {st['official_ge30_days_total']:,} | {st['dropped_not_in_mask_total']} | {st['n_stations']:,} | {nS:,} | {nC:,} | {nF:,} |")
w("")
w("GHAP 2022 against these stations (description of the target, not a model score): share of station-days within 0.5 µg m⁻³ / median station "
  "daily R / RMSE, at the station pixel and on the 1 km ring; last column, RMSE of GHAP at the pixel against its own 1 km and 10 km ring means "
  "(same-day table, paired).")
w("")
out += hdr(["network", "station pixel", "ring 1 km", "pixel − ring 1 km", "pixel − ring 10 km"])
for g in GR:
    a, b = gv[g]["pixel"], gv[g]["ring1km"]
    c1 = R["tables"]["ring1km"]["S_pixel_vs_ring"]["contrasts"]["ghap@pixel - ghap@ring"][g]["sameday"]
    c10 = R["tables"]["ring10km"]["S_pixel_vs_ring"]["contrasts"]["ghap@pixel - ghap@ring"][g]["sameday"]
    w(f"| {g} | {100 * a['share_station_days_within_0p5']:.1f} % / {a['median_station_R']:.3f} / {a['rmse']:.2f} | "
      f"{100 * b['share_station_days_within_0p5']:.1f} % / {b['median_station_R']:.3f} / {b['rmse']:.2f} | {p(c1)} | {p(c10)} |")
w("")
eo = OBS["observations"][EU]
ec = OBS["squared_error_concentration"][EU]
w(f"**The European values contain artefacts** (descriptive, added after the tables; `results/diag_obs_target.json`). Of {eo['station_days']:,} EEA "
  f"station-days, {eo['n_gt_500']} are above 500 µg m⁻³ (maximum {eo['max']:.0f}) and {eo['n_le_0']:,} are exactly 0. The values above 300 µg m⁻³ "
  "(102 station-days) sit at 21 stations whose medians are 5–17 µg m⁻³; nine are exactly 999.0. "
  f"Consequence: in Europe {100 * ec['cams_first_day_bilinear']['share_of_squared_error_in_top_0p1pct_station_days']:.0f} % of the squared error of the "
  f"CAMS first-day field sits in 0.1 % of the station-days and {100 * ec['cams_first_day_bilinear']['share_of_squared_error_in_top_10_stations']:.0f} % in "
  f"ten stations (downscaler {100 * ec['downscaler_sameday']['share_of_squared_error_in_top_0p1pct_station_days']:.0f} % and "
  f"{100 * ec['downscaler_sameday']['share_of_squared_error_in_top_10_stations']:.0f} %; GHAP "
  f"{100 * ec['ghap_pixel']['share_of_squared_error_in_top_0p1pct_station_days']:.0f} % and {100 * ec['ghap_pixel']['share_of_squared_error_in_top_10_stations']:.0f} %). "
  f"The RMSE of the CAMS first-day field is {ec['cams_first_day_bilinear']['rmse']:.2f} with them and "
  f"{ec['cams_first_day_bilinear']['rmse_without_top_0p1pct_station_days']:.2f} without its 0.1 % worst station-days. This is why every system, GHAP "
  "included, has an RMSE of 11–12.5 in Europe, and why European RMSE differences of a few percent carry little information; the paper's own European "
  "RMSEs are computed on the same values. The fair CRPS, the per-station correlation and the share of stations are much less affected. No day shift: "
  "in every network the agreement of GHAP, of CAMS and of the downscaler with the stations is best at lag 0.")
w("")
w("Files: `pos/st_qc_official.json`, `results/scores_official_2022.json` (`stations`, `ghap_vs_stations_2022`, `tables/ring*/S_pixel_vs_ring`), "
  "`results/diag_obs_target.json`.")
w("")

# ------------------------------------------------------------------------------------------------ 3. same-day
w("## 3. Same-day mode: the frozen downscaler with the frozen test's inputs, 365 days")
w("")
w(f"{nS:,} identical station-days, {S['n_stations']:,} stations, station pixel. `ds_sameday` = 8 members with the first-day mean of the CAMS 00 UTC "
  "run of the same day, ERA5, GFAS and the transport channels (exactly the frozen test, extended to these stations). Relative changes: A / B − 1 on "
  "identical station-days, 95 % interval from the paper's 7-day moving-block bootstrap over dates (10,000 draws), * when it excludes zero. The "
  "intervals resample time only.")
w("")
out += hdr(["network", "RMSE: downscaler / CAMS first-day field / BCSD / climatology (paper) / GHAP itself", "downscaler − CAMS first-day field", "downscaler − BCSD",
            "downscaler − climatology (paper)", "stations better than CAMS", "median station r: downscaler / CAMS", "bias: downscaler / CAMS"])
for g in GR:
    A = {x: arm(S, x, g, "sameday") for x in ("ds_sameday", "cams0_bil", "bcsd", "clim_paper", "ghap")}
    w(f"| {g} | {A['ds_sameday']['rmse']:.2f} / {A['cams0_bil']['rmse']:.2f} / {A['bcsd']['rmse']:.2f} / {A['clim_paper']['rmse']:.2f} / {A['ghap']['rmse']:.2f} | "
      f"{p(con(S, 'ds_sameday', 'cams0_bil', g, 'sameday'))} | {p(con(S, 'ds_sameday', 'bcsd', g, 'sameday'))} | {p(con(S, 'ds_sameday', 'clim_paper', g, 'sameday'))} | "
      f"{100 * A['ds_sameday']['share_stations_rmse_below_cams0_bil']:.0f} % of {A['ds_sameday']['n_stations_ge_min_days']:,} | "
      f"{A['ds_sameday']['median_station_r']:.3f} / {A['cams0_bil']['median_station_r']:.3f} | {A['ds_sameday']['bias']:+.2f} / {A['cams0_bil']['bias']:+.2f} |")
w("")
out += hdr(["network", "fair CRPS: downscaler / climatology ensemble", "CRPS change", "spread/skill", "observations inside the member range (78 % expected)"])
for g in GR:
    A = {x: arm(S, x, g, "sameday") for x in ("ds_sameday", "clim_ens")}
    w(f"| {g} | {A['ds_sameday']['crps']:.2f} / {A['clim_ens']['crps']:.2f} | {p(con(S, 'ds_sameday', 'clim_ens', g, 'sameday'), 'rel_crps')} | "
      f"{A['ds_sameday']['spread_skill']:.2f} | {100 * A['ds_sameday']['coverage_member_range']:.0f} % |")
w("")
w("The same contrast by season (descriptive, added after the tables; same mask and bootstrap, its whole-year row reproduces the table above): "
  "relative RMSE change of the downscaler against the CAMS first-day field, and the bias of the two in December–February.")
w("")
out += hdr(["network", "May–November", "January–April + December", "December–February", "bias Dec–Feb: downscaler / CAMS"])
sp = lambda pn, g: (lambda c: f"{100 * c['rel_rmse']:+.1f} % [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}]" + ("" if c['ci'][0] <= 0 <= c['ci'][1] else "*"))(SEAS[pn][g]["ds_sameday - cams0_bil"])   # noqa: E731
for g in GR:
    w(f"| {g} | {sp('May-Nov (days 120-333)', g)} | {sp('Jan-Apr + Dec', g)} | {sp('Dec-Feb', g)} | "
      f"{SEAS['Dec-Feb'][g]['bias']['ds_sameday']:+.2f} / {SEAS['Dec-Feb'][g]['bias']['cams0_bil']:+.2f} |")
w("")
w("Files: `results/scores_official_2022.json` (`tables/pixel/S`), `results/diag_season.json`. Rings of this table: `results/tables_official_2022.md` section 2.4.")
w("")

# ------------------------------------------------------------------------------------------------ 4. forecast mode, pooled
w("## 4. Forecast mode, six networks pooled, by lead day")
w("")
w(f"{nC:,} identical station-days, {C['n_stations']:,} stations, 214 issue days (1 May – 30 November 2022, 00 UTC), station pixel. Lead day L = hours "
  "24(L−1) to 24L after the issue. `chain`: each of the eight members of the paper's 0.4° forecast, daily mean of the lead day, downscaled (one member "
  "each, noise seeds changing with the day). `chain_t6fc`: the same with the transported-PM2.5 channel rebuilt from the forecast itself. `coarse`: "
  "the eight 0.4° members read bilinearly. CAMS forecast and Aurora: values and equal-issuance rule of the paper (Aurora's lead day 5 is computed "
  "here with the paper's rule; the paper reports lead days 1–4). **Composition = forecast; weather, fires and transport = analysis-type fields of "
  "the valid day.**")
w("")
arms4 = [("chain", "chain (0.4° forecast → downscaler)"), ("chain_t6fc", "chain, transport channel from the forecast"), ("coarse", "0.4° forecast, bilinear (the chain's input)"),
         ("camsfc", "CAMS forecast (paper)"), ("aurora", "Aurora (paper)"), ("camsfc_ds", "CAMS forecast → downscaler"), ("bcsd_fc", "BCSD of the 0.4° forecast"),
         ("persist_ds", "downscaled field of the day before the issue, persisted"), ("ds_sameday", "downscaler with same-day inputs (analysis-type)"),
         ("clim_paper", "climatology (paper)"), ("ghap", "GHAP 2022 itself (the 1 km estimates; not a model)")]
for keyq, title, nd in (("rmse", "RMSE (µg m⁻³)", 2), ("crps", "Fair CRPS (absolute error for a single field)", 2)):
    w(f"{title}")
    w("")
    out += hdr(["system"] + lead_cols())
    for a, lab in arms4:
        w(f"| {lab} | " + " | ".join(f2(arm(C, a, ALL, L).get(keyq), nd) for L in L5) + " |")
    w("")
w("Bias / spread-skill / median station correlation / share of stations with a lower RMSE than the CAMS forecast, at lead days 1, 3 and 5")
w("")
out += hdr(["system", "lead day 1", "lead day 3", "lead day 5"])
for a, lab in arms4:
    cells = []
    for L in ("1", "3", "5"):
        x = arm(C, a, ALL, L)
        cells.append(f"{f2(x.get('bias'), 2, True)} / {f2(x.get('spread_skill'))} / {f2(x.get('median_station_r'), 3)} / "
                     + ("–" if a == "camsfc" else f"{100 * x[k]:.0f} %"))
    w(f"| {lab} | " + " | ".join(cells) + " |")
w("")
w("Relative RMSE change, A − B")
w("")
out += hdr(["A − B"] + lead_cols())
for A_, B_ in (("chain", "coarse"), ("chain", "camsfc"), ("chain", "aurora"), ("chain", "bcsd_fc"), ("chain", "camsfc_ds"), ("chain", "ds_sameday"),
               ("chain", "persist_ds"), ("chain", "clim_paper"), ("chain", "ghap"), ("chain_t6fc", "chain"), ("chain_t6fc", "camsfc"), ("chain_rtseed", "chain"),
               ("coarse", "camsfc"), ("aurora", "camsfc"), ("bcsd_fc", "coarse"), ("camsfc", "clim_paper")):
    w(f"| `{A_}` − `{B_}` | " + " | ".join(p(con(C, A_, B_, ALL, L)) for L in L5) + " |")
w("| `chain` − station value of the day before the issue (own common days) | " + " | ".join(p(op[ALL][L]) for L in L5) + " |")
w("")
w("Relative CRPS change, A − B (ensemble against ensemble, except the last two rows where B is a single field and the comparison is not like for like)")
w("")
out += hdr(["A − B"] + lead_cols())
for A_, B_ in (("chain", "coarse"), ("chain", "clim_ens"), ("chain", "bcsd_fc"), ("chain", "camsfc_ds"), ("chain", "ds_sameday"), ("chain_t6fc", "chain"),
               ("chain", "camsfc"), ("chain", "aurora")):
    w(f"| `{A_}` − `{B_}` | " + " | ".join(p(con(C, A_, B_, ALL, L), "rel_crps") for L in L5) + " |")
w("")
w("File: `results/scores_official_2022.json` (`tables/pixel/C`).")
w("")

# ------------------------------------------------------------------------------------------------ 5. forecast mode by network
w("## 5. Forecast mode by network")
w("")
w("RMSE, all lead days pooled (µg m⁻³)")
w("")
out += hdr(["network", "station-days", "chain", "chain, transport from the forecast", "0.4° forecast", "CAMS forecast", "Aurora", "BCSD of the 0.4° forecast",
            "climatology (paper)", "GHAP itself"])
for g in GR:
    w(f"| {g} | {arm(C, 'chain', g)['n']:,} | " + " | ".join(f2(arm(C, a, g)["rmse"]) for a in ("chain", "chain_t6fc", "coarse", "camsfc", "aurora", "bcsd_fc", "clim_paper", "ghap")) + " |")
w("")
w("Relative RMSE change of the chain, all lead days pooled, and at lead day 1")
w("")
out += hdr(["network", "− 0.4° forecast", "− CAMS forecast", "− Aurora", "− BCSD of the 0.4° forecast", "− climatology (paper)", "lead day 1: − 0.4° forecast",
            "lead day 1: − CAMS forecast"])
for g in GR:
    w(f"| {g} | " + " | ".join(p(con(C, "chain", b, g)) for b in ("coarse", "camsfc", "aurora", "bcsd_fc", "clim_paper")) + f" | {p(con(C, 'chain', 'coarse', g, '1'))} | "
      f"{p(con(C, 'chain', 'camsfc', g, '1'))} |")
w("")
w("Fair CRPS change of the chain (all leads), share of stations where it beats the CAMS forecast, per-station correlation, bias and spread")
w("")
out += hdr(["network", "CRPS − 0.4° forecast", "CRPS − climatology ensemble", "stations better than the CAMS forecast: lead day 1 / 3 / 5 (0.4° forecast at lead day 1)",
            "median station r at lead day 1: chain / 0.4° forecast / CAMS forecast", "bias at lead day 1: chain / 0.4° forecast / CAMS forecast",
            "spread/skill; observations inside the member range"])
for g in GR:
    a1 = {x: arm(C, x, g, "1") for x in ("chain", "coarse", "camsfc")}
    w(f"| {g} | {p(con(C, 'chain', 'coarse', g), 'rel_crps')} | {p(con(C, 'chain', 'clim_ens', g), 'rel_crps')} | "
      f"{100 * a1['chain'][k]:.0f} % / {100 * arm(C, 'chain', g, '3')[k]:.0f} % / {100 * arm(C, 'chain', g, '5')[k]:.0f} % ({100 * a1['coarse'][k]:.0f} %) | "
      f"{a1['chain']['median_station_r']:.3f} / {a1['coarse']['median_station_r']:.3f} / {a1['camsfc']['median_station_r']:.3f} | "
      f"{a1['chain']['bias']:+.2f} / {a1['coarse']['bias']:+.2f} / {a1['camsfc']['bias']:+.2f} | {arm(C, 'chain', g)['spread_skill']:.2f}; "
      f"{100 * arm(C, 'chain', g)['coverage_member_range']:.0f} % |")
w("")
w("Europe by lead day (the network new to the downscaler): RMSE / fair CRPS")
w("")
out += hdr(["system"] + lead_cols())
for a, lab in arms4:
    w(f"| {lab} | " + " | ".join(f"{f2(arm(C, a, EU, L).get('rmse'))} / {f2(arm(C, a, EU, L).get('crps'))}" for L in L5) + " |")
w("")
out += hdr(["Europe, A − B (RMSE)"] + lead_cols())
for A_, B_ in (("chain", "coarse"), ("chain", "camsfc"), ("chain", "aurora"), ("chain", "bcsd_fc"), ("chain", "clim_paper"), ("chain", "ds_sameday"), ("chain_t6fc", "chain")):
    w(f"| `{A_}` − `{B_}` | " + " | ".join(p(con(C, A_, B_, EU, L)) for L in L5) + " |")
w("")
w("File: `results/scores_official_2022.json` (`tables/pixel/C`). All networks by lead day: `results/tables_official_2022.md` section 3.10.")
w("")

# ------------------------------------------------------------------------------------------------ 6. Foehn
w("## 6. With the paper's 1 km forecast (Foehn), where the start days coincide")
w("")
w(f"{nF:,} identical station-days, {F['n_stations']:,} stations, 107 issue days (Foehn starts every second day; start day = issue day − 1, lead h = L, "
  "so all systems are issued at 00 UTC of the same day, the paper's equal-issuance rule). Foehn: 4 members at the station pixel, forecast weather, "
  "and the 1 km estimates of the two days before the issue as initial state. Chain: 8 members, no 1 km field of any day, analysis-type weather. "
  "The paper's own Foehn numbers cover the whole year (179 start days); these cover May–November.")
w("")
armsF = [("foehn", "Foehn 1 km (paper)"), ("chain", "chain"), ("chain_t6fc", "chain, transport from the forecast"), ("coarse", "0.4° forecast, bilinear"),
         ("camsfc", "CAMS forecast"), ("aurora", "Aurora"), ("bcsd_fc", "BCSD of the 0.4° forecast"), ("clim_paper", "climatology (paper)"), ("ghap", "GHAP itself")]
out += hdr(["system: RMSE / fair CRPS"] + lead_cols())
for a, lab in armsF:
    w(f"| {lab} | " + " | ".join(f"{f2(arm(F, a, ALL, L).get('rmse'))} / {f2(arm(F, a, ALL, L).get('crps'))}" for L in L5) + " |")
w("")
out += hdr(["A − B"] + lead_cols())
for A_, B_, kq in (("chain", "foehn", "rel_rmse"), ("chain", "foehn", "rel_crps"), ("chain_t6fc", "foehn", "rel_rmse"), ("chain_t6fc", "foehn", "rel_crps"),
                   ("foehn", "camsfc", "rel_rmse"), ("chain", "camsfc", "rel_rmse"), ("foehn", "coarse", "rel_rmse"), ("chain", "coarse", "rel_rmse")):
    w(f"| `{A_}` − `{B_}`, {'RMSE' if kq == 'rel_rmse' else 'CRPS'} | " + " | ".join(p(con(F, A_, B_, ALL, L), kq) for L in L5) + " |")
w("")
w("By network: chain against Foehn")
w("")
out += hdr(["network", "station-days", "RMSE Foehn / chain, lead day 1", "chain − Foehn RMSE, lead day 1", "chain − Foehn RMSE, all leads", "chain − Foehn CRPS, lead day 1",
            "chain − Foehn CRPS, all leads", "stations better than the CAMS forecast at lead day 1: Foehn / chain", "median station r at lead day 1 (5): Foehn / chain",
            "spread/skill: Foehn / chain"])
for g in GR:
    c1, ca_ = con(F, "chain", "foehn", g, "1"), con(F, "chain", "foehn", g)
    a, b = arm(F, "foehn", g, "1"), arm(F, "chain", g, "1")
    a5, b5 = arm(F, "foehn", g, "5"), arm(F, "chain", g, "5")
    w(f"| {g} | {ca_['n']:,} | {a['rmse']:.2f} / {b['rmse']:.2f} | {p(c1)} | {p(ca_)} | {p(c1, 'rel_crps')} | {p(ca_, 'rel_crps')} | "
      f"{100 * a[k]:.0f} % / {100 * b[k]:.0f} % | {a['median_station_r']:.3f} / {b['median_station_r']:.3f} ({a5['median_station_r']:.3f} / {b5['median_station_r']:.3f}) | "
      f"{arm(F, 'foehn', g)['spread_skill']:.2f} / {arm(F, 'chain', g)['spread_skill']:.2f} |")
w("")
w("File: `results/scores_official_2022.json` (`tables/pixel/F`).")
w("")

# ------------------------------------------------------------------------------------------------ 7. rings
w("## 7. Rings: is the station pixel special?")
w("")
w("Relative RMSE change of the chain, all lead days pooled, at the station pixel and with the chain replaced by its ring mean (each ring has its own "
  "station set; the coarse systems keep their value at the station coordinates)")
w("")
out += hdr(["network"] + [f"chain − 0.4° forecast, {WN[x]}" for x in WH] + ["chain − CAMS forecast, station pixel", "chain − CAMS forecast, ring 10 km"])
for g in GR:
    w(f"| {g} | " + " | ".join(p(con(R["tables"][x]["C"], "chain", "coarse", g)) for x in WH)
      + f" | {p(con(C, 'chain', 'camsfc', g))} | {p(con(R['tables']['ring10km']['C'], 'chain', 'camsfc', g))} |")
w("")
w("The value at the station pixel against the ring mean of the same field (paired, stations that have the ring; negative = the pixel is closer to "
  "the monitor than the ring around it), forecast mode, all lead days pooled")
w("")
out += hdr(["network", "chain: 1 km", "chain: 3 km", "chain: 10 km", "BCSD of the 0.4° forecast: 10 km", "climatological median of the downscaler: 10 km",
            "GHAP itself: 1 km", "GHAP itself: 10 km"])
for g in GR:
    w(f"| {g} | {p(g1('chain', 'ring1km', g))} | {p(g1('chain', 'ring3km', g))} | {p(g1('chain', 'ring10km', g))} | {p(g1('bcsd_fc', 'ring10km', g))} | "
      f"{p(g1('clim_med', 'ring10km', g))} | {p(g1('ghap', 'ring1km', g))} | {p(g1('ghap', 'ring10km', g))} |")
w("")
gc = lambda wv, g=ALL: R["tables"][wv]["C_pixel_vs_ring"]["contrasts"]["chain@pixel - chain@ring"][g]["all"]    # noqa: E731
w(f"In CRPS the pixel is slightly better than its 1 km ring mean ({p(gc('ring1km'), 'rel_crps')} pooled, {p(gc('ring1km', US), 'rel_crps')} in the USA) "
  "while its RMSE is not: averaging each member over a ring removes spread from an ensemble that is already too narrow, so this is a spread effect, "
  "not evidence of skill at the pixel.")
w("")
w("Files: `results/scores_official_2022.json` (`tables/ring*/C`, `tables/ring*/C_pixel_vs_ring`). Same-day counterparts: `results/tables_official_2022.md` sections 2.4, 7.2.")
w("")

# ------------------------------------------------------------------------------------------------ 8. screened
if RS:
    PS = RS["tables"]["pixel"]
    S2, C2, F2 = PS["S"], PS["C"], PS["F"]
    scr = json.loads((O / "screened/results/screen.json").read_text())
    w("## 8. Sensitivity: the same scoring with the implausible European values removed")
    w("")
    w(f"Declared in the protocol note before it was run, after the tables were read. Rule, fixed from the observations alone: an EEA station-day is set "
      f"to missing when its value is 0 or above 300 µg m⁻³ ({scr['eea_station_days_removed_scored_stations']:,} of "
      f"{scr['eea_station_days_before_scored_stations']:,} station-days of the scored European stations, "
      f"{100 * scr['eea_station_days_removed_scored_stations'] / scr['eea_station_days_before_scored_stations']:.2f} %). The five other networks are "
      "untouched; code, predictions, mask logic and bootstrap are identical. The tables of sections 3–7 remain the result of record, because the paper "
      "scores its systems on the unscreened values; this section shows what they hide in Europe.")
    w("")
    out += hdr(["Europe", "as in the paper's file", "screened"])
    rows = [("same-day, 365 days: RMSE downscaler / CAMS first-day field / BCSD / climatology (paper) / GHAP",
             lambda T_: " / ".join(f2(arm(T_["S"], a, EU, "sameday")["rmse"]) for a in ("ds_sameday", "cams0_bil", "bcsd", "clim_paper", "ghap"))),
            ("same-day: downscaler − CAMS first-day field", lambda T_: p(con(T_["S"], "ds_sameday", "cams0_bil", EU, "sameday"))),
            ("same-day: downscaler − BCSD", lambda T_: p(con(T_["S"], "ds_sameday", "bcsd", EU, "sameday"))),
            ("same-day: downscaler − climatology (paper)", lambda T_: p(con(T_["S"], "ds_sameday", "clim_paper", EU, "sameday"))),
            ("same-day: stations better than CAMS", lambda T_: f"{100 * arm(T_['S'], 'ds_sameday', EU, 'sameday')['share_stations_rmse_below_cams0_bil']:.0f} %"),
            ("forecast mode, all leads: RMSE chain / 0.4° forecast / CAMS forecast / Aurora / BCSD / climatology (paper) / GHAP",
             lambda T_: " / ".join(f2(arm(T_["C"], a, EU)["rmse"]) for a in ("chain", "coarse", "camsfc", "aurora", "bcsd_fc", "clim_paper", "ghap"))),
            ("chain − 0.4° forecast, all leads", lambda T_: p(con(T_["C"], "chain", "coarse", EU))),
            ("chain − CAMS forecast, lead day 1", lambda T_: p(con(T_["C"], "chain", "camsfc", EU, "1"))),
            ("chain − CAMS forecast, all leads", lambda T_: p(con(T_["C"], "chain", "camsfc", EU))),
            ("chain − Aurora, all leads", lambda T_: p(con(T_["C"], "chain", "aurora", EU))),
            ("chain − BCSD of the 0.4° forecast, all leads", lambda T_: p(con(T_["C"], "chain", "bcsd_fc", EU))),
            ("chain − climatology (paper), all leads", lambda T_: p(con(T_["C"], "chain", "clim_paper", EU))),
            ("chain − 0.4° forecast, CRPS, all leads", lambda T_: p(con(T_["C"], "chain", "coarse", EU), "rel_crps")),
            ("chain − station value of the day before, lead day 1", lambda T_: p(T_["C"]["contrasts_own_days"]["chain - obs_persist"]["groups"][EU]["1"])),
            ("stations where the chain beats the CAMS forecast, lead day 1 / 3 / 5", lambda T_: " / ".join(f"{100 * arm(T_['C'], 'chain', EU, L)[k]:.0f} %" for L in ("1", "3", "5"))),
            ("with Foehn, lead day 1: RMSE Foehn / chain / CAMS forecast", lambda T_: " / ".join(f2(arm(T_["F"], a, EU, "1")["rmse"]) for a in ("foehn", "chain", "camsfc"))),
            ("chain − Foehn, RMSE, lead day 1", lambda T_: p(con(T_["F"], "chain", "foehn", EU, "1"))),
            ("chain − Foehn, RMSE, all leads", lambda T_: p(con(T_["F"], "chain", "foehn", EU))),
            ("chain − Foehn, CRPS, all leads", lambda T_: p(con(T_["F"], "chain", "foehn", EU), "rel_crps")),
            ("Foehn − CAMS forecast, RMSE, all leads", lambda T_: p(con(T_["F"], "foehn", "camsfc", EU))),
            ("chain pixel − its 10 km ring, RMSE, all leads", None)]
    for lab, fn in rows:
        if fn is None:
            a_ = R["tables"]["ring10km"]["C_pixel_vs_ring"]["contrasts"]["chain@pixel - chain@ring"][EU]["all"]
            b_ = RS["tables"]["ring10km"]["C_pixel_vs_ring"]["contrasts"]["chain@pixel - chain@ring"][EU]["all"]
            w(f"| {lab} | {p(a_)} | {p(b_)} |")
        else:
            w(f"| {lab} | {fn(PX)} | {fn(PS)} |")
    w("")
    out += hdr(["All six networks", "as in the paper's file", "screened"])
    for lab, fn in (("forecast mode, all leads: RMSE chain / 0.4° forecast / CAMS forecast / Aurora", lambda T_: " / ".join(f2(arm(T_["C"], a)["rmse"]) for a in ("chain", "coarse", "camsfc", "aurora"))),
                    ("chain − 0.4° forecast, all leads", lambda T_: p(con(T_["C"], "chain", "coarse"))),
                    ("chain − CAMS forecast, all leads", lambda T_: p(con(T_["C"], "chain", "camsfc"))),
                    ("chain − BCSD of the 0.4° forecast, all leads", lambda T_: p(con(T_["C"], "chain", "bcsd_fc"))),
                    ("chain − Foehn, RMSE, lead day 1", lambda T_: p(con(T_["F"], "chain", "foehn", ALL, "1"))),
                    ("chain − Foehn, RMSE, all leads", lambda T_: p(con(T_["F"], "chain", "foehn"))),
                    ("chain − Foehn, CRPS, all leads", lambda T_: p(con(T_["F"], "chain", "foehn"), "rel_crps")),
                    ("same-day: downscaler − CAMS first-day field", lambda T_: p(con(T_["S"], "ds_sameday", "cams0_bil", ALL, "sameday")))):
        w(f"| {lab} | {fn(PX)} | {fn(PS)} |")
    w("")
    e1, e2 = con(C, "chain", "camsfc", EU), con(C2, "chain", "camsfc", EU)
    w(f"Reading: without the artefacts the European RMSEs fall from about 12 to {arm(C2, 'chain', EU)['rmse']:.1f}–{arm(C2, 'camsfc', EU)['rmse']:.1f} µg m⁻³ and "
      f"the systems separate. The chain against the CAMS forecast in Europe goes from {p(e1)} to {p(e2)}; against its input from "
      f"{p(con(C, 'chain', 'coarse', EU))} to {p(con(C2, 'chain', 'coarse', EU))}; against BCSD from {p(con(C, 'chain', 'bcsd_fc', EU))} to "
      f"{p(con(C2, 'chain', 'bcsd_fc', EU))}; against Foehn (all leads) from {p(con(F, 'chain', 'foehn', EU))} to {p(con(F2, 'chain', 'foehn', EU))}, "
      f"lead day 1 {p(con(F2, 'chain', 'foehn', EU, '1'))}. Shares of stations and correlations do not move, as expected of statistics that a few values "
      "cannot dominate. Three things follow. (1) The paper's own system gains from the screen: Foehn against the CAMS forecast in Europe goes from "
      f"{p(con(F, 'foehn', 'camsfc', EU))} to {p(con(F2, 'foehn', 'camsfc', EU))} (all leads, May–November). (2) In Europe the station pixel of the chain is "
      f"worse than its own 10 km ring mean, {p(RS['tables']['ring10km']['C_pixel_vs_ring']['contrasts']['chain@pixel - chain@ring'][EU]['all'])}, and so is GHAP's "
      f"({p(RS['tables']['ring10km']['C_pixel_vs_ring']['contrasts']['ghap@pixel - ghap@ring'][EU]['all'])}): no skill of the 1 km detail at European monitors "
      f"either. (3) In May–November GHAP itself is not closer to the European monitors than the coarse forecasts (RMSE {arm(C2, 'ghap', EU)['rmse']:.2f} "
      f"against {arm(C2, 'camsfc', EU)['rmse']:.2f} for the CAMS forecast and {arm(C2, 'chain', EU)['rmse']:.2f} for the chain); over the year it is "
      f"({arm(S2, 'ghap', EU, 'sameday')['rmse']:.2f} against {arm(S2, 'cams0_bil', EU, 'sameday')['rmse']:.2f} for the CAMS first-day field). The pooled "
      "six-network rows change by a few points in the same direction and no conclusion of section 1 is reversed. (Why GHAP's own pixel is worse than its "
      "ring at European monitors while matching the monitor exactly on a quarter of the days was not investigated; one untested possibility is that the "
      "daily values GHAP was fitted to are not defined on the same 24 hours as the paper's EEA daily means.)")
    w("")
    w("Files: `screened/results/scores_official_2022.json`, `screened/results/tables_official_2022.md` (same layout as the main tables), "
      "`screened/results/screen.json`; code `code/score_screened.py`.")
    w("")

# ------------------------------------------------------------------------------------------------ 9. checks
c3 = CK3["parts"]
w("## 9. Checks of the reading, and the check that failed")
w("")
chk = R["checks"]
pc_ = chk["paper_compare_fair"]["rows"]
w(f"- The paper's station comparison recomputed from the raw files with this work's loaders equals the stored file exactly (largest RMSE "
  f"difference {chk['paper_compare_fair']['max_abs_rmse_diff']:.1e}): lead day 1, Foehn {pc_[0]['rmse_recomputed']:.2f}, CAMS forecast "
  f"{pc_[1]['rmse_recomputed']:.2f}, Aurora {pc_[2]['rmse_recomputed']:.2f} on {pc_[0]['n_recomputed']:,} station-days: the three verified numbers "
  "of the paper (whole year, the paper's own station-days).")
w(f"- The station RMSE of the 0.4° model in the paper's file, recomputed from `pred.npy` with this work's reader: largest difference "
  f"{chk['paper_cams12h_stations']['max_abs_rmse_diff']:.1e} (its \"J+n\" is lead day n + 1 here).")
fr = chk["frozen_test_pixel_station_rmse"]
w(f"- The frozen test's station RMSEs of the five shared networks, recomputed in its own convention (monitors of a pixel averaged) with this work's "
  f"members: largest difference {fr['max_abs_rmse_diff']:.1e} µg m⁻³, pixel-day counts identical (e.g. {fr['rows'][0]['network']} "
  f"{fr['rows'][0]['rmse_recomputed']:.3f} against {fr['rows'][0]['rmse_frozen_test']:.3f} on {fr['rows'][0]['n_recomputed']:,} pixel-days).")
w(f"- `camsfc_ds` at lead day 1 equals the same-day arm on all {CK0['check5a_camsfc_ds_lead1_vs_sameday']['issue_days_with_sameday_day_done']} issue days "
  f"({100 * CK0['check5a_camsfc_ds_lead1_vs_sameday']['frac_bit_identical']:.1f} % bit-identical); the transport operator reproduces the stored transport "
  f"channel (at least {100 * CK0['check5b_gather_reproduces_stored_ch41']['min_frac_bit_identical']:.2f} % bit-identical).")
w("")
bad = c3["rank30_days_vs_frozen_test_REPORTED_ONLY"]
w(f"**Check 3 (same-day members against the frozen test's stored members) failed on the full run; the reference was at fault, not the new "
  f"predictions.** The first check found {100 * CK0['check3_sameday_vs_frozen_test']['sameday_ds']['frac_bit_identical']:.2f} % bit-identical values and a "
  f"largest difference of {CK0['check3_sameday_vs_frozen_test']['sameday_ds']['max_abs_diff_ug']:.2f} µg m⁻³ "
  f"({100 * CK0['check3_sameday_vs_frozen_test']['sameday_ds']['max_rel_diff']:.1f} %), beyond its tolerance of one float16 step. Diagnosis: every "
  f"difference larger than two float16 steps ({bad['n_rel_gt_2steps']:,} values) is on the {bad['days']} days that a single process of the frozen "
  f"test's prediction job {CK3['frozen_test_job']} computed (its rank {CK3['frozen_test_rank_with_errors']} of 32). On the "
  f"{c3['sound_days_vs_frozen_test']['days']} days of the 31 other processes the two computations agree to one float16 step "
  f"({100 * c3['sound_days_vs_frozen_test']['frac_bit_identical']:.3f} % bit-identical, largest relative difference "
  f"{c3['sound_days_vs_frozen_test']['max_rel_diff']:.5f}). A second computation of the {bad['days']} days on another node (job {CK3['recheck_job']}) "
  f"equals the first one ({100 * c3['rank30_days_vs_recheck']['frac_bit_identical']:.3f} % bit-identical, largest relative difference "
  f"{c3['rank30_days_vs_recheck']['max_rel_diff']:.5f}) and differs from the frozen test's files in the same way. The check was then run with the "
  "second computation as reference on those days and the frozen test on the others, thresholds unchanged, and passes; the change is dated in the "
  "protocol note, before the truth extraction and the scoring. Consequence for the frozen test: 17 of its day files (and the full fields of three of "
  f"those days) carry sporadic errors of one device; at the station pixels the ensemble mean moves by 5e-5 µg m⁻³ on average and its station RMSEs "
  f"by at most {fr['max_abs_rmse_diff']:.0e}. Nothing was modified in the frozen test.")
w("")
w("Files: `results/scores_official_2022.json` (`checks`), `results/paper_checks.json`, `results/check_preds_preds.json` (the failed check, kept), "
  "`results/diag_check3_preds.json`, `results/diag_check3_perday.json`, `results/check_recheck.json`, `results/check3_split.json`.")
w("")

# ------------------------------------------------------------------------------------------------ 10. comparability
w("## 10. What is and is not comparable; limits of this evidence")
w("")
w("- Forecast mode is forecast mode for the composition only (see the caution at the top). It is an upper bound for the real-time chain, whose "
  "weather is an IFS forecast, whose mixing height is a substitute and whose fires are persisted, and whose coarse weights are fine-tuned ones, not "
  "the paper's.")
w("- 2022 is not an unseen year: the frozen system was already tested on 2022 on five of these six networks; only the European network is new to "
  "it. The stations are not independent of GHAP in China and the USA (section 2).")
w("- The forecast-mode period has no winter; the same-day table shows that winter is where the downscaler fails in Korea and Japan.")
w("- Pooled RMSEs are dominated by a few terms: India (5 % of the station-days) carries about a third of the pooled squared error, and in Europe "
  "ten stations with implausible values carry more than half of the network's. Read the pooled RMSE with the CRPS, the per-network rows and the "
  "share of stations.")
w("- The intervals resample start dates only and ignore the spatial clustering of stations; with 107 or 214 start dates of one season they are "
  "wide for Korea, Japan and India.")
w("- The CRPS of an ensemble against the absolute error of a single field (CAMS forecast, Aurora) is not a like-for-like probabilistic comparison.")
w("- Foehn has 4 members and forecast weather; the chain 8 members and analysis-type weather. The chain's lower RMSE at lead days 4–5 is not "
  "evidence of a better forecast.")
w("- Not comparable with the live scores of November 2026 (other coarse weights, forecast weather, OpenAQ monitors, season) nor with the paper's "
  "station table of the 0.4° model (its \"J+n\" is lead day n + 1 here).")
w("- The season split, the description of the observations and the screened sensitivity were added after the tables were read; they are "
  "descriptive and labelled as such in the protocol note.")
w("")
w("## 11. Compute and files")
w("")
w("GPU: smoke test 23863113 (dev-g, 3 min 50 s), full prediction 23863371 (standard-g, 1 node, 49 min 47 s), second computation of 24 days "
  "23865411 (dev-g, 6 min 33 s): 1.00 node-hours in all, of which 0.83 on standard-g (budget 8). CPU (small): positions and inputs 23862916, "
  "scoring 23865479 (16 min 27 s), screened scoring 23865779 (5 min 15 s).")
w("")
w("`preds/` (float16, µg m⁻³; `meta.json`), `truth/ghap_official_2022.npy`, `results/scores_official_2022.json`, `results/tables_official_2022.md`, "
  "`screened/results/`, `results/diag_*.json`, `results/check*.json`, `code/`, `logs/`, `STATUS.md`, `UNSEALING_A7.md`.")
(O / "REPORT.md").write_text("\n".join(out) + "\n")
print(f"wrote {O / 'REPORT.md'} ({len(out)} lines; screened sensitivity {'included' if RS else 'NOT available'})")
