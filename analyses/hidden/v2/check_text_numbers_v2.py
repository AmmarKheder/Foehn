#!/usr/bin/env python3
"""Every number of the version 2 texts against results/numbers_v2.json.
Each entry: the file, a fragment of the text with {} in place of the numbers, and for each {} the key of the value in
numbers_v2.json ("path|index|decimals|options"; index 0 = value, 1 = lower, 2 = upper bound of the 95 % interval; options:
a = absolute value, n = sign reversed, t = thousands separator). The fragment filled with the formatted values must occur
in the file. The script also counts the words of the legend.
    python check_text_numbers_v2.py"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
N = json.loads((HERE / f"results{SFX}" / "numbers_v2.json").read_text())   # the hand-written texts below stay the stored ones
TXT = {"si": (HERE / "results" / "text_methods_si_v2.tex").read_text(), "main": (HERE / "results" / "replacement_text_v2.tex").read_text()}


def S(t, h, s, m):
    return f"systems/thr{t}/lead{h}/World/{s}/{m}"


def D(t, h, pair, m):
    return f"systems/thr{t}/lead{h}/World/differences/{pair}/{m}"


def EN(t, h, s, m):
    return f"systems/thr{t}/lead{h}/World/equal_number_warned/{s}/{m}"


def EC(t, h, k, m):
    return f"systems/thr{t}/lead{h}/World/equal_correct_share/{k}/{m}"


def X(t, k, m):
    return f"foehn_3_days_against_references_1_day/thr{t}/{k}/{m}"


def E(h, s, m):
    return f"checks_of_the_bias_corrected_references/lead{h}/errors_population_weighted/{s}/{m}"


def get(path):
    v = N
    for k in path.split("/"):
        v = v[k]
    return v


def fmt(spec):
    path, idx, dec, *opt = spec.split("|")
    opt = opt[0] if opt else ""
    v = get(path)
    v = v[int(idx)] if isinstance(v, list) else v
    if "a" in opt:
        v = abs(v)
    if "n" in opt:
        v = -v
    return f"{v:,.{int(dec)}f}" if "t" in opt else f"{v:.{int(dec)}f}"


def tri(path, dec, opt=""):
    """value [lower, upper]"""
    return [f"{path}|0|{dec}|{opt}", f"{path}|1|{dec}|{opt}", f"{path}|2|{dec}|{opt}"]


def trin(path, dec):
    """a difference written with its sign reversed: value [lower, upper] of minus the stored difference"""
    return [f"{path}|0|{dec}|n", f"{path}|2|{dec}|n", f"{path}|1|{dec}|n"]


F, F4, C, A, Q, B, K = "foehn_p50", "foehn_40km", "cams", "aurora", "cams_qm", "cams_bcsd", "clim_1km"
CHECKS = [
    # ------------------------------------------------------------------ legend
    ("si", r"\model{} reaches {}\% of the people exposed with {}\% of correct warnings, against {}\% and {}\% for the CAMS forecast and {}\% and {}\% for Aurora",
     [S(75, 1, F, "reach_pct") + "|0|1", S(75, 1, F, "correct_pct") + "|0|1", S(75, 1, C, "reach_pct") + "|0|1", S(75, 1, C, "correct_pct") + "|0|1",
      S(75, 1, A, "reach_pct") + "|0|1", S(75, 1, A, "correct_pct") + "|0|1"]),
    ("si", r"at 37.5 ({}\% and {}\%, against {}\% and {}\%, and {}\% and {}\%) and at 75\,\textmu g\,m$^{-3}$ ({}\% and {}\%)",
     [S(37.5, 3, F, "reach_pct") + "|0|1", S(37.5, 3, F, "correct_pct") + "|0|1", S(37.5, 1, C, "reach_pct") + "|0|1", S(37.5, 1, C, "correct_pct") + "|0|1",
      S(37.5, 1, A, "reach_pct") + "|0|1", S(37.5, 1, A, "correct_pct") + "|0|1", S(75, 3, F, "reach_pct") + "|0|1", S(75, 3, F, "correct_pct") + "|0|1"]),
    ("si", r"{}\% against {}\% one day ahead, and {}\% against {}\% of those hidden with a margin",
     [S(37.5, 1, F, "hidden_warned_pct") + "|0|1", S(37.5, 1, F4, "hidden_warned_pct") + "|0|1", S(37.5, 1, F, "hidden_margin_warned_pct") + "|0|1",
      S(37.5, 1, F4, "hidden_margin_warned_pct") + "|0|1"]),
    ("si", r"the CAMS forecast warns {} million of them and {} million people wrongly, and Aurora {} and {} million, against {} and {} million for \model{}",
     [S(37.5, 1, C, "hidden_warned_M") + "|0|0", S(37.5, 1, C, "false_in_silent40_M") + "|0|0", S(37.5, 1, A, "hidden_warned_M") + "|0|0",
      S(37.5, 1, A, "false_in_silent40_M") + "|0|0", S(37.5, 1, F, "hidden_warned_M") + "|0|0", S(37.5, 1, F, "false_in_silent40_M") + "|0|0"]),
    # ------------------------------------------------------------------ Methods
    ("si", r"from {} to {}\,\textmu g\,m$^{-3}$, and to {}\,\textmu g\,m$^{-3}$ with the pattern (\model{}'s ensemble mean, {}\,\textmu g\,m$^{-3}$)",
     [E(1, C, "rmse") + "|0|1", E(1, Q, "rmse") + "|0|1", E(1, B, "rmse") + "|0|1", E(1, "foehn_mean", "rmse") + "|0|1"]),
    ("si", r"(bias $-{}$ and $-{}$\,\textmu g\,m$^{-3}$, against $-{}$\,\textmu g\,m$^{-3}$ before correction)",
     [E(1, Q, "bias") + "|0|1|a", E(1, B, "bias") + "|0|1|a", E(1, C, "bias") + "|0|1|a"]),
    # ------------------------------------------------------------------ Supplementary note: as issued
    ("si", r"the bias-corrected CAMS forecast warns {} million people a day, {} million of them rightly: it reaches {}\% of the people exposed [{}, {}], and {}\% of its warnings are correct [{}, {}]",
     [S(37.5, 1, Q, "warned_M") + "|0|0", S(37.5, 1, Q, "right_M") + "|0|0"] + tri(S(37.5, 1, Q, "reach_pct"), 1) + tri(S(37.5, 1, Q, "correct_pct"), 1)),
    ("si", r"With the climatological 1\,km pattern it warns {} million people, {} million of them rightly ({}\% [{}, {}]; {}\% [{}, {}])",
     [S(37.5, 1, B, "warned_M") + "|0|0|t", S(37.5, 1, B, "right_M") + "|0|0"] + tri(S(37.5, 1, B, "reach_pct"), 1) + tri(S(37.5, 1, B, "correct_pct"), 1)),
    ("si", r"\model{} warns {} million people, {} million of them rightly ({}\%; {}\%): it rightly warns {} million more people a day than the corrected forecast with the pattern [{}, {}] and wrongly warns {} million more [{}, {}]; its critical success index is {}, against {} (difference {} [{}, {}])",
     [S(37.5, 1, F, "warned_M") + "|0|0|t", S(37.5, 1, F, "right_M") + "|0|0|t", S(37.5, 1, F, "reach_pct") + "|0|1", S(37.5, 1, F, "correct_pct") + "|0|1"]
     + tri(D(37.5, 1, f"{F}-{B}", "right_M"), 0) + tri(D(37.5, 1, f"{F}-{B}", "false_M"), 0) + [S(37.5, 1, F, "csi") + "|0|2", S(37.5, 1, B, "csi") + "|0|2"]
     + tri(D(37.5, 1, f"{F}-{B}", "csi"), 2)),
    # ------------------------------------------------------------------ Supplementary note: equal number of people warned
    ("si", r"lowered to {} times the threshold, it warns as many people as \model{}, reaches {}\% of the people exposed [{}, {}] and {}\% of its warnings are correct [{}, {}]: {} points [{}, {}] and {} points [{}, {}] below \model{}, which rightly warns {} million more people a day [{}, {}]",
     [EN(37.5, 1, B, "decision_threshold_factor") + "|0|2"] + tri(EN(37.5, 1, B, "reach_pct"), 1) + tri(EN(37.5, 1, B, "correct_pct"), 1)
     + tri(EN(37.5, 1, B, "foehn_p50_minus_reach_pct"), 1) + tri(EN(37.5, 1, B, "foehn_p50_minus_correct_pct"), 1) + tri(EN(37.5, 1, B, "foehn_p50_minus_right_M"), 0)),
    ("si", r"Without the pattern the differences are {} points [{}, {}] and {} points [{}, {}]; for the CAMS forecast and Aurora as issued, which already warn about as many people as \model{}, they are {} and {} points, and {} and {} points",
     tri(EN(37.5, 1, Q, "foehn_p50_minus_reach_pct"), 1) + tri(EN(37.5, 1, Q, "foehn_p50_minus_correct_pct"), 1)
     + [EN(37.5, 1, C, "foehn_p50_minus_reach_pct") + "|0|1", EN(37.5, 1, C, "foehn_p50_minus_correct_pct") + "|0|1",
        EN(37.5, 1, A, "foehn_p50_minus_reach_pct") + "|0|1", EN(37.5, 1, A, "foehn_p50_minus_correct_pct") + "|0|1"]),
    ("si", r"above 75\,\textmu g\,m$^{-3}$ it closes two fifths ({} against {} points of reach)",
     [EN(75, 1, C, "foehn_p50_minus_reach_pct") + "|0|1", EN(75, 1, B, "foehn_p50_minus_reach_pct") + "|0|1"]),
    ("si", r"by {} points of reach [{}, {}] and {} points of correct warnings [{}, {}], and above 15\,\textmu g\,m$^{-3}$ by {} points [{}, {}] and {} points [{}, {}]",
     tri(EN(75, 1, B, "foehn_p50_minus_reach_pct"), 1) + tri(EN(75, 1, B, "foehn_p50_minus_correct_pct"), 1)
     + tri(EN(15, 1, B, "foehn_p50_minus_reach_pct"), 1) + tri(EN(15, 1, B, "foehn_p50_minus_correct_pct"), 1)),
    ("si", r"Three days ahead the differences are {} points [{}, {}] and {} points [{}, {}] above 37.5\,\textmu g\,m$^{-3}$; above 75\,\textmu g\,m$^{-3}$ they are within their intervals ({} points [$-{}$, {}] and {} points [$-{}$, {}]), and above 15\,\textmu g\,m$^{-3}$ the corrected forecast is ahead by {} points of reach [{}, {}] and {} points of correct warnings [{}, {}]",
     tri(EN(37.5, 3, B, "foehn_p50_minus_reach_pct"), 1) + tri(EN(37.5, 3, B, "foehn_p50_minus_correct_pct"), 1)
     + [EN(75, 3, B, "foehn_p50_minus_reach_pct") + "|0|1", EN(75, 3, B, "foehn_p50_minus_reach_pct") + "|1|1|a", EN(75, 3, B, "foehn_p50_minus_reach_pct") + "|2|1",
        EN(75, 3, B, "foehn_p50_minus_correct_pct") + "|0|1", EN(75, 3, B, "foehn_p50_minus_correct_pct") + "|1|1|a", EN(75, 3, B, "foehn_p50_minus_correct_pct") + "|2|1"]
     + trin(EN(15, 3, B, "foehn_p50_minus_reach_pct"), 1) + trin(EN(15, 3, B, "foehn_p50_minus_correct_pct"), 1)),
    # ------------------------------------------------------------------ Supplementary note: hidden people
    ("si", r"With the climatological pattern the corrected forecast warns {} million of the hidden people a day ({}\% [{}, {}]), against {} million without it ({}\% [{}, {}]), a difference of {} points [{}, {}]; of the people hidden with a margin of 10\% it warns {}\% against {}\%",
     [S(37.5, 1, B, "hidden_warned_M") + "|0|1"] + tri(S(37.5, 1, B, "hidden_warned_pct"), 1) + [S(37.5, 1, Q, "hidden_warned_M") + "|0|1"]
     + tri(S(37.5, 1, Q, "hidden_warned_pct"), 1) + tri(D(37.5, 1, f"{B}-{Q}", "hidden_warned_pct"), 1)
     + [S(37.5, 1, B, "hidden_margin_warned_pct") + "|0|1", S(37.5, 1, Q, "hidden_margin_warned_pct") + "|0|1"]),
    ("si", r"\model{} as issued warns {} million ({}\%), {} million more than the corrected forecast with the pattern [{}, {}]",
     [S(37.5, 1, F, "hidden_warned_M") + "|0|1", S(37.5, 1, F, "hidden_warned_pct") + "|0|1"] + tri(D(37.5, 1, f"{F}-{B}", "hidden_warned_M"), 1)),
    ("si", r"the corrected forecast with the pattern warns {} million of the hidden people [{}, {}], {} million more than \model{} [{}, {}], and without the pattern {} million, as many as \model{} (difference {} million [$-{}$, {}])",
     tri(EN(37.5, 1, B, "hidden_warned_M"), 1) + trin(EN(37.5, 1, B, "foehn_p50_minus_hidden_warned_M"), 1) + [EN(37.5, 1, Q, "hidden_warned_M") + "|0|1",
     EN(37.5, 1, Q, "foehn_p50_minus_hidden_warned_M") + "|0|1", EN(37.5, 1, Q, "foehn_p50_minus_hidden_warned_M") + "|1|1|a", EN(37.5, 1, Q, "foehn_p50_minus_hidden_warned_M") + "|2|1"]),
    ("si", r"At the share of correct warnings of \model{} ({}\%), the corrected forecast with the pattern reaches {}\% of the people exposed, {} points fewer [{}, {}], and warns {} million of the hidden people, as many as \model{} (ratio {} [{}, {}])",
     [S(37.5, 1, F, "correct_pct") + "|0|1", EC(37.5, 1, f"{B}_at_correct_share_of_{F}", "reach_pct") + "|0|1"]
     + trin(EC(37.5, 1, f"{B}_at_correct_share_of_{F}", "moved_minus_fixed_reach_pct"), 1) + [EC(37.5, 1, f"{B}_at_correct_share_of_{F}", "hidden_warned_M") + "|0|1"]
     + tri(EC(37.5, 1, f"{B}_at_correct_share_of_{F}", "moved_over_fixed_hidden_warned_M"), 2)),
    ("si", r"the corrected forecasts warn {} and {} million people a day, {} and {} million of them wrongly, that is {} and {} people wrongly warned for each hidden person warned, against {} for \model{}, {} for \model{} on 40\,km cells, {} for the CAMS forecast and {} for Aurora",
     [S(37.5, 1, Q, "warned_in_silent40_M") + "|0|0", S(37.5, 1, B, "warned_in_silent40_M") + "|0|0", S(37.5, 1, Q, "false_in_silent40_M") + "|0|0",
      S(37.5, 1, B, "false_in_silent40_M") + "|0|0", S(37.5, 1, Q, "false_per_hidden_warned") + "|0|1", S(37.5, 1, B, "false_per_hidden_warned") + "|0|1",
      S(37.5, 1, F, "false_per_hidden_warned") + "|0|1", S(37.5, 1, F4, "false_per_hidden_warned") + "|0|1", S(37.5, 1, C, "false_per_hidden_warned") + "|0|1",
      S(37.5, 1, A, "false_per_hidden_warned") + "|0|1"]),
    # ------------------------------------------------------------------ Supplementary note: the climatology alone
    ("si", r"warns {} million people a day above 37.5\,\textmu g\,m$^{-3}$: it reaches {}\% of the people exposed, {}\% of its warnings are correct, and it warns {} million of the hidden people ({}\%), {} million more than \model{} [{}, {}], with {} million wrong warnings a day against {} million",
     [S(37.5, 1, K, "warned_M") + "|0|0|t", S(37.5, 1, K, "reach_pct") + "|0|1", S(37.5, 1, K, "correct_pct") + "|0|1", S(37.5, 1, K, "hidden_warned_M") + "|0|1",
      S(37.5, 1, K, "hidden_warned_pct") + "|0|1"] + trin(D(37.5, 1, f"{F}-{K}", "hidden_warned_M"), 1) + [S(37.5, 1, K, "false_M") + "|0|0", S(37.5, 1, F, "false_M") + "|0|0"]),
    ("si", r"the ensemble mean of \model{} reaches {} points more of the people exposed [{}, {}] with {} points more correct warnings [{}, {}], and warns as many of the hidden people ({} against {} million)",
     tri(f"systems/thr37.5/lead1/World/foehn_mean_at_the_number_warned_by/{K}/foehn_mean_minus_{K}_reach_pct", 1)
     + tri(f"systems/thr37.5/lead1/World/foehn_mean_at_the_number_warned_by/{K}/foehn_mean_minus_{K}_correct_pct", 1)
     + [f"systems/thr37.5/lead1/World/foehn_mean_at_the_number_warned_by/{K}/hidden_warned_M|0|1", S(37.5, 1, K, "hidden_warned_M") + "|0|1"]),
    # ------------------------------------------------------------------ Supplementary note: three days against one day
    ("si", r"On the {} target days", ["foehn_3_days_against_references_1_day/n_common_target_days|0|0"]),
    ("si", r"reaches {} points more of the people exposed above 37.5\,\textmu g\,m$^{-3}$ than the CAMS forecast issued one day ahead [{}, {}], with {} points more correct warnings [{}, {}]; against Aurora the differences are {} points [{}, {}] and {} points [{}, {}]",
     tri(X(37.5, f"{F}_lead3-{C}_lead1", "reach_pct"), 1) + tri(X(37.5, f"{F}_lead3-{C}_lead1", "correct_pct"), 1)
     + tri(X(37.5, f"{F}_lead3-{A}_lead1", "reach_pct"), 1) + tri(X(37.5, f"{F}_lead3-{A}_lead1", "correct_pct"), 1)),
    ("si", r"they are {} points [{}, {}] and {} points [{}, {}] against the CAMS forecast, and {} points [$-{}$, {}] and {} points [{}, {}] against Aurora",
     tri(X(75, f"{F}_lead3-{C}_lead1", "reach_pct"), 1) + tri(X(75, f"{F}_lead3-{C}_lead1", "correct_pct"), 1)
     + [X(75, f"{F}_lead3-{A}_lead1", "reach_pct") + "|0|1", X(75, f"{F}_lead3-{A}_lead1", "reach_pct") + "|1|1|a", X(75, f"{F}_lead3-{A}_lead1", "reach_pct") + "|2|1"]
     + tri(X(75, f"{F}_lead3-{A}_lead1", "correct_pct"), 1)),
    ("si", r"they are {} points [{}, {}] and {} point [{}, {}] against the CAMS forecast, and {} points [{}, {}] and {} points [{}, {}] against Aurora",
     tri(X(15, f"{F}_lead3-{C}_lead1", "reach_pct"), 1) + tri(X(15, f"{F}_lead3-{C}_lead1", "correct_pct"), 1)
     + tri(X(15, f"{F}_lead3-{A}_lead1", "reach_pct"), 1) + tri(X(15, f"{F}_lead3-{A}_lead1", "correct_pct"), 1)),
    ("si", r"({} points more than the corrected forecast with the pattern above 37.5\,\textmu g\,m$^{-3}$ [{}, {}]) with fewer correct warnings ({} points fewer [{}, {}])",
     tri(X(37.5, f"{F}_lead3-{B}_lead1", "reach_pct"), 1) + trin(X(37.5, f"{F}_lead3-{B}_lead1", "correct_pct"), 1)),
    # ------------------------------------------------------------------ main text, full version
    ("main", r"reaches {}\% of the 1.51 billion people exposed a day, and {}\% of the warnings are correct, against {}\% and {}\% for the CAMS forecast issued at the same time and {}\% and {}\% for Aurora",
     [S(37.5, 1, F, "reach_pct") + "|0|1", S(37.5, 1, F, "correct_pct") + "|0|1", S(37.5, 1, C, "reach_pct") + "|0|1", S(37.5, 1, C, "correct_pct") + "|0|1",
      S(37.5, 1, A, "reach_pct") + "|0|1", S(37.5, 1, A, "correct_pct") + "|0|1"]),
    ("main", r"\model{} reaches {}\% of the people exposed with {}\% of correct warnings, against {}\% and {}\%, and {}\% and {}\%",
     [S(75, 1, F, "reach_pct") + "|0|1", S(75, 1, F, "correct_pct") + "|0|1", S(75, 1, C, "reach_pct") + "|0|1", S(75, 1, C, "correct_pct") + "|0|1",
      S(75, 1, A, "reach_pct") + "|0|1", S(75, 1, A, "correct_pct") + "|0|1"]),
    ("main", r"{}\% and {}\% above 37.5\,\textmu g\,m$^{-3}$, and {}\% and {}\% above 75\,\textmu g\,m$^{-3}$",
     [S(37.5, 3, F, "reach_pct") + "|0|1", S(37.5, 3, F, "correct_pct") + "|0|1", S(75, 3, F, "reach_pct") + "|0|1", S(75, 3, F, "correct_pct") + "|0|1"]),
    ("main", r"where the difference in reach from Aurora ({} points) is within its interval", [X(75, f"{F}_lead3-{A}_lead1", "reach_pct") + "|0|1"]),
    ("main", r"\model{} warns {}\% of them (95\% interval, {}--{}\%), {} of {} million a day, against {}\% ({}--{}\%) for its ensemble mean averaged over 40\,km cells ({}\% for the ensemble mean at 1\,km), and {}\% against {}\% of the people hidden with a margin",
     tri(S(37.5, 1, F, "hidden_warned_pct"), 1) + [S(37.5, 1, F, "hidden_warned_M") + "|0|0", S(37.5, 1, F, "hidden_M") + "|0|0"] + tri(S(37.5, 1, F4, "hidden_warned_pct"), 1)
     + [S(37.5, 1, "foehn_mean", "hidden_warned_pct") + "|0|1", S(37.5, 1, F, "hidden_margin_warned_pct") + "|0|1", S(37.5, 1, F4, "hidden_margin_warned_pct") + "|0|1"]),
    ("si", r"({}\% against {}\%); with the ensemble mean at 1\,km the share is {}\%, so that {} points [{}, {}] come from the resolution alone, and {} points [{}, {}] for the people hidden with a margin of 10\% ({}\% against {}\%)",
     [S(37.5, 1, F, "hidden_warned_pct") + "|0|1", S(37.5, 1, F4, "hidden_warned_pct") + "|0|1", S(37.5, 1, "foehn_mean", "hidden_warned_pct") + "|0|1"]
     + tri(D(37.5, 1, "foehn_mean-foehn_40km", "hidden_warned_pct"), 1) + tri(D(37.5, 1, "foehn_mean-foehn_40km", "hidden_margin_warned_pct"), 1)
     + [S(37.5, 1, "foehn_mean", "hidden_margin_warned_pct") + "|0|1", S(37.5, 1, F4, "hidden_margin_warned_pct") + "|0|1"]),
    ("main", r"The CAMS forecast and Aurora warn {}\% and {}\% of the hidden people, but by warning widely (Fig.~\ref{fig:warned}c): where the mean of the 40\,km cell is below the threshold they warn {} and {} million people a day, {} and {} million of them wrongly, against {} and {} million for \model{}",
     [S(37.5, 1, C, "hidden_warned_pct") + "|0|0", S(37.5, 1, A, "hidden_warned_pct") + "|0|0", S(37.5, 1, C, "warned_in_silent40_M") + "|0|0", S(37.5, 1, A, "warned_in_silent40_M") + "|0|0",
      S(37.5, 1, C, "false_in_silent40_M") + "|0|0", S(37.5, 1, A, "false_in_silent40_M") + "|0|0", S(37.5, 1, F, "warned_in_silent40_M") + "|0|0",
      S(37.5, 1, F, "false_in_silent40_M") + "|0|0"]),
    ("main", r"it warns {} million people a day, {}\% of them rightly, and reaches {}\% of the people exposed and {}\% of the hidden",
     [S(37.5, 1, B, "warned_M") + "|0|0|t", S(37.5, 1, B, "correct_pct") + "|0|1", S(37.5, 1, B, "reach_pct") + "|0|1", S(37.5, 1, B, "hidden_warned_pct") + "|0|1"]),
    ("main", r"it reaches {}\% of the people exposed with {}\% of correct warnings, {} points ({}--{}) and {} points ({}--{}) below \model{}, and it then warns {} million of the hidden people, more than \model{}",
     [EN(37.5, 1, B, "reach_pct") + "|0|1", EN(37.5, 1, B, "correct_pct") + "|0|1"] + tri(EN(37.5, 1, B, "foehn_p50_minus_reach_pct"), 1)
     + tri(EN(37.5, 1, B, "foehn_p50_minus_correct_pct"), 1) + [EN(37.5, 1, B, "hidden_warned_M") + "|0|0"]),
    # ------------------------------------------------------------------ main text, compact version
    ("main", r"reach {}\% of the 1.51 billion people exposed a day and {}\% of them are correct, against {}\% and {}\% for the CAMS forecast issued at the same time; issued three days ahead, \model{} still does better ({}\% and {}\%)",
     [S(37.5, 1, F, "reach_pct") + "|0|1", S(37.5, 1, F, "correct_pct") + "|0|1", S(37.5, 1, C, "reach_pct") + "|0|1", S(37.5, 1, C, "correct_pct") + "|0|1",
      S(37.5, 3, F, "reach_pct") + "|0|1", S(37.5, 3, F, "correct_pct") + "|0|1"]),
    ("main", r"It warns {}\% of the people whom a 40\,km grid hides, against {}\% for its ensemble mean averaged over 40\,km cells (Fig.~\ref{fig:warned}b); the CAMS forecast warns {}\% of them by warning widely, with {} million people wrongly warned a day where the 40\,km mean is below the threshold, against {} million",
     [S(37.5, 1, F, "hidden_warned_pct") + "|0|1", S(37.5, 1, F4, "hidden_warned_pct") + "|0|1", S(37.5, 1, C, "hidden_warned_pct") + "|0|0",
      S(37.5, 1, C, "false_in_silent40_M") + "|0|0", S(37.5, 1, F, "false_in_silent40_M") + "|0|0"]),
    ("main", r"reaches {}\% of the people exposed with {}\% of correct warnings, {} and {} points below \model{}, and warns as many of the hidden people or more",
     [EN(37.5, 1, B, "reach_pct") + "|0|1", EN(37.5, 1, B, "correct_pct") + "|0|1", EN(37.5, 1, B, "foehn_p50_minus_reach_pct") + "|0|1",
      EN(37.5, 1, B, "foehn_p50_minus_correct_pct") + "|0|1"]),
    # ------------------------------------------------------------------ main text, notes on version 1
    ("main", r"the ratio is {} [{}, {}] ({} against {} million at its share of correct warnings,",
     tri(EC(37.5, 1, f"foehn_mean_at_correct_share_of_{B}", "moved_over_fixed_hidden_warned_M"), 2)
     + [EC(37.5, 1, f"foehn_mean_at_correct_share_of_{B}", "hidden_warned_M") + "|0|1", S(37.5, 1, B, "hidden_warned_M") + "|0|1"]),
    ("main", r"({} against {} and {}", [S(37.5, 1, F, "hidden_warned_M") + "|0|1", S(37.5, 1, Q, "hidden_warned_M") + "|0|1", S(37.5, 1, B, "hidden_warned_M") + "|0|1"]),
    ("main", r"+{} million [{}, {}] against the one with the pattern), but not at an equal number of people warned ({}",
     tri(D(37.5, 1, f"{F}-{B}", "hidden_warned_M"), 1) + [S(37.5, 1, F, "hidden_warned_M") + "|0|1"]),
    ("main", r"against {} million) and not above 15 ug m-3 ({} against {} million as issued)",
     [EN(37.5, 1, B, "hidden_warned_M") + "|0|1", S(15, 1, F, "hidden_warned_M") + "|0|1", S(15, 1, B, "hidden_warned_M") + "|0|1"]),
]


def main():
    bad = 0
    for key, tpl, specs in CHECKS:
        tpl = tpl.replace("\\model{}", "\x00")                              # the braces of \model{} are not a placeholder
        assert tpl.count("{}") == len(specs), (tpl, len(specs))
        vals = [fmt(s) for s in specs]
        text = tpl
        for v in vals:
            text = text.replace("{}", v, 1)
        text = text.replace("\x00", "\\model{}")
        flat = " ".join(TXT[key].replace("\n% ", " ").replace("\n%", " ").split())
        ok = " ".join(text.split()) in flat
        bad += not ok
        print(("ok   " if ok else "FAIL ") + key + ": " + text[:210] + ("" if ok else "\n       keys: " + "; ".join(specs)))
    # facts read in the map file (Methods): 1,412 days, 78 to 124 days per month, 21 quantiles
    m = get("checks_of_the_bias_corrected_references/map")
    days = int(re.search(r"days=(\d+)", m).group(1)); per = [int(v) for v in re.search(r"days_per_month=([\d,]+)", m).group(1).split(",")]
    for frag in (f"{days:,} days of 2017--2020", f"({min(per)} to {max(per)} days per month)", "21 quantiles"):
        ok = frag in TXT["si"]; bad += not ok
        print(("ok   " if ok else "FAIL ") + "si: " + frag)
    # words of the legend
    cap = TXT["si"].split("\\caption{", 1)[1].split("\n\n", 1)[0]
    plain = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cap)
    plain = plain.replace("\\model{}", "Foehn").replace("\\,", " ").replace("\\textmu", "µ").replace("$^{-3}$", "-3").replace("$^\\circ$", "°").replace("\\%", "%")
    plain = re.sub(r"[{}$\\]", "", plain)
    words = [w for w in plain.split() if re.search(r"[A-Za-z0-9]", w)]
    print(f"legend: {len(words)} words (limit 300)")
    bad += len(words) > 300
    print("ALL NUMBERS OF THE TEXTS AGREE WITH numbers_v2.json" if not bad else f"{bad} PROBLEM(S)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
