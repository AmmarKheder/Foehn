#!/usr/bin/env python3
"""Who is warned, real forecast against real forecast (2022). Written from a copy of figures/fig_hidden.py (its panel e of
15 October 2026: bars, whiskers, colours, panel letters); the original script is not modified and nothing is written
outside this directory. No field is read here: every number comes from results/numbers.json (tables.py).

Residents of the 8.1 M 1 km pixels with at least 100 residents; a resident is exposed when the 1 km estimate of the target
day at their pixel is above the threshold, hidden when the mean of their 40 km cell is not above it, and warned when the
forecast read at their pixel is above it (Foehn: at least 2 of its 4 members; Foehn on 40 km cells: its ensemble mean
averaged over the cell; the CAMS forecast and Aurora issued at the same time, interpolated to the pixel).
a, people warned a day above 37.5 ug m-3, 1 and 3 days ahead: rightly (system colour), of whom hidden by the 40 km grid
   (pink), and wrongly (hatched); dashed line: the people exposed.
b, share of the hidden people who are warned (bars, with 95 % intervals over days) and the same share for the people hidden
   with a margin of 10 % of the threshold on both sides (diamonds).
c, share of the exposed people whom the warnings reach against the share of the warnings that are correct, for the WHO
   24-h guideline level (15) and interim targets 3 (37.5) and 1 (75 ug m-3); filled: 1 day ahead, open: 3 days ahead.
    python fig_hidden_warned.py            -> figs/fig_hidden_warned.pdf, .png
    LETTERS=fgh python fig_hidden_warned.py   (other panel letters, if the panels are appended to another figure; the files
                                               are then named fig_hidden_warned_fgh)"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures")
sys.path.insert(0, str(FIG))

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from gc_style import AURORA, CAMS, FOEHN, clean, letter_before  # noqa: E402
from nature_style import DOUBLE, MM, plt  # noqa: E402

PINK = "#C2185B"                                                    # hidden people (Fig. 2)
F40 = "#8FB2DD"                                                     # Foehn's own forecast on 40 km cells (light Foehn blue)
INK = "#1B1E23"
SYSTEMS = (("foehn_p50", "Foehn", FOEHN), ("foehn_40km", "Foehn on\n40 km cells", F40), ("cams", "CAMS\nforecast", CAMS),
           ("aurora", "Aurora", AURORA))
THRS = (("15", "o", "15"), ("37.5", "s", "37.5"), ("75", "^", "75"))
BW = 0.36


def xpos(i, j):
    return i + (j - 0.5) * (BW + 0.04)


def system_axis(ax, y_names=-0.105, size=5.6):
    ax.set_xticks([xpos(i, j) for i in range(len(SYSTEMS)) for j in range(2)])
    ax.set_xticklabels(["1 d", "3 d"] * len(SYSTEMS), fontsize=5.2)
    ax.tick_params(axis="x", length=0, pad=1.5)
    for i, (_, name, col) in enumerate(SYSTEMS):
        ax.text(i, y_names, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=size, color=col,
                fontweight="bold", linespacing=0.95)
    ax.set_xlim(-0.55, len(SYSTEMS) - 0.45)


def main():
    N = json.loads((HERE / "results" / "numbers.json").read_text())["systems"]
    plt.rcParams["hatch.linewidth"] = 0.45
    W, H = DOUBLE, 60 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    S = N["thr37.5"]
    # ------------------------------------------------------------------ a: people warned a day
    aa = fig.add_axes([fx(13), fy(12.5), fx(60), fy(40)])
    for i, (s, _, col) in enumerate(SYSTEMS):
        for j, h in enumerate((1, 3)):
            d = S[f"lead{h}"]["World"][s]
            hid, fa = d["hidden_warned_M"][0], d["false_M"][0]; seen = d["right_M"][0] - hid
            x = xpos(i, j)
            aa.bar(x, seen, width=BW, color=col, zorder=3, lw=0)
            aa.bar(x, hid, bottom=seen, width=BW, color=PINK, zorder=3, lw=0)
            aa.bar(x, fa, bottom=seen + hid, width=BW, facecolor="white", edgecolor=col, hatch="//////", lw=0.5, zorder=3)
            print(f"a: {s} {h} d: rightly {d['right_M'][0]:.1f} M (hidden {hid:.1f}), wrongly {fa:.1f} M, warned {d['warned_M'][0]:.1f} M")
    ex = S["lead1"]["World"]["foehn_p50"]["exposed_M"][0]
    aa.axhline(ex, color="#555555", lw=0.6, ls=(0, (3, 2)), zorder=2)
    aa.set_ylim(0, 2330); aa.set_yticks(range(0, 2001, 500))
    aa.set_ylabel("People warned a day (millions)", fontsize=6)
    clean(aa); aa.grid(axis="x", visible=False); system_axis(aa)
    aa.legend(handles=[Patch(facecolor="#8A8A8A", lw=0, label="rightly warned"),
                       Patch(facecolor=PINK, lw=0, label="of whom hidden by the 40 km grid"),
                       Patch(facecolor="white", edgecolor="#8A8A8A", hatch="//////", lw=0.5, label="wrongly warned"),
                       Line2D([], [], color="#555555", lw=0.6, ls=(0, (3, 2)), label="people exposed")],
              fontsize=5.3, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, handlelength=1.3, columnspacing=1.6,
              handletextpad=0.5, borderaxespad=0.2, labelspacing=0.35)
    aa.set_title("People warned a day above 37.5 µg m$^{-3}$", fontsize=6.3, loc="left", pad=5)
    # ------------------------------------------------------------------ b: of the hidden people
    ab = fig.add_axes([fx(88), fy(12.5), fx(47), fy(40)])
    for i, (s, _, col) in enumerate(SYSTEMS):
        for j, h in enumerate((1, 3)):
            d = S[f"lead{h}"]["World"][s]
            v, m = d["hidden_warned_pct"], d["hidden_margin_warned_pct"]
            x = xpos(i, j)
            ab.bar(x, v[0], width=BW, color=PINK, zorder=3, lw=0)
            ab.vlines(x, v[1], v[2], color=INK, lw=0.5, zorder=4)
            ab.plot([x], [m[0]], marker="D", ms=2.6, mfc="white", mec=INK, mew=0.5, ls="none", zorder=5)
            print(f"b: {s} {h} d: hidden warned {v[0]:.1f} % [{v[1]:.1f}, {v[2]:.1f}] ({d['hidden_warned_M'][0]:.1f} M), with margin {m[0]:.1f} %")
    ab.set_ylim(0, 58.25); ab.set_yticks(range(0, 51, 10))
    ab.set_ylabel("Hidden people warned (%)", fontsize=6)
    clean(ab); ab.grid(axis="x", visible=False); system_axis(ab)
    ab.legend(handles=[Line2D([], [], marker="D", ms=2.6, mfc="white", mec=INK, mew=0.5, ls="none", label="hidden with a margin of 10 %")],
              fontsize=5.3, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02), handletextpad=0.2, borderaxespad=0.2)
    ab.set_title("Hidden people warned", fontsize=6.3, loc="left", pad=5)
    # ------------------------------------------------------------------ c: reach against correct warnings, three thresholds
    ac = fig.add_axes([fx(149), fy(12.5), fx(32.5), fy(40)])
    for t, mk, _ in THRS:
        for s, _, col in SYSTEMS:
            p = [(N[f"thr{t}"][f"lead{h}"]["World"][s]["correct_pct"], N[f"thr{t}"][f"lead{h}"]["World"][s]["reach_pct"]) for h in (1, 3)]
            ac.plot([p[0][0][0], p[1][0][0]], [p[0][1][0], p[1][1][0]], color=col, lw=0.5, zorder=3)
            ac.plot([p[1][0][0]], [p[1][1][0]], marker=mk, ms=3.0, mfc="white", mec=col, mew=0.7, ls="none", zorder=4)
            ac.plot([p[0][0][0]], [p[0][1][0]], marker=mk, ms=3.2, mfc=col, mec=col, mew=0.5, ls="none", zorder=5)
            print(f"c: > {t} {s}: 1 d correct {p[0][0][0]:.1f} reach {p[0][1][0]:.1f} | 3 d correct {p[1][0][0]:.1f} reach {p[1][1][0]:.1f}")
    ac.set_xlim(28, 96); ac.set_ylim(34, 96); ac.set_xticks(range(30, 91, 20)); ac.set_yticks(range(40, 91, 10))
    ac.set_xlabel("Warnings that are correct (%)", fontsize=6, labelpad=2); ac.set_ylabel("Exposed people reached (%)", fontsize=6)
    clean(ac)
    hs = [Line2D([], [], marker=mk, ms=3.0, mfc="#8A8A8A", mec="#8A8A8A", ls="none", label=f"> {lab}") for _, mk, lab in THRS]
    hs += [Line2D([], [], marker="s", ms=3.0, mfc="#8A8A8A", mec="#8A8A8A", ls="none", label="1 d ahead"),
           Line2D([], [], marker="s", ms=3.0, mfc="white", mec="#8A8A8A", mew=0.7, ls="none", label="3 d ahead")]
    ac.legend(handles=hs, fontsize=5.2, frameon=False, loc="lower right", bbox_to_anchor=(1.03, -0.02), handletextpad=0.1, borderaxespad=0.1,
              labelspacing=0.3, title="µg m$^{-3}$", title_fontsize=5.2)
    ac.set_title("Reach and correct warnings", fontsize=6.3, loc="left", pad=5)
    fig.canvas.draw()                                                  # letters just before the titles (fig_hidden.py, panel e)
    let = os.environ.get("LETTERS", "abc")
    for a_, s_ in zip((aa, ab, ac), let):
        letter_before(fig, a_._left_title, s_)
    (HERE / "figs").mkdir(exist_ok=True)
    name = "fig_hidden_warned" + ("" if let == "abc" else f"_{let}")
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "figs" / f"{name}.{ext}")
    print("saved", HERE / "figs" / f"{name}.pdf")


if __name__ == "__main__":
    main()
