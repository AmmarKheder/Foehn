#!/usr/bin/env python3
"""Who is warned, version 2. Written from a copy of ../fig_hidden_warned.py (house style,
fonts, colours of the systems, bars, whiskers, panel letters unchanged; that script is not modified). No field is read
here: every number comes from results/numbers_v3.json (tables_v3.py).

Residents of the 8.1 M 1 km grid cells with at least 100 residents; a resident is exposed when the 1 km estimate of the
target day in their grid cell is above the threshold, hidden when the mean of their 40 km cell is not above it, and warned
when the forecast read at their grid cell is above it. Systems (one colour each, in every panel):
  Foehn (at least 2 of its 4 members), Foehn on 40 km cells (its ensemble mean averaged over the cell), the CAMS forecast and
  Aurora issued at the same time, the CAMS forecast bias-corrected towards the 1 km estimates (quantile map per 0.25 deg cell
  and month fitted on 2017-2020; cell value), and the same with the climatological 1 km pattern of the estimates.
a, share of the exposed people whom the warnings reach against the share of the warnings that are correct. Marker shape =
   threshold (15, 37.5, 75 ug m-3), filling = lead time (filled 1 day, open 3 days), colour = system; three separate keys.
   Inset: the cluster of the 15 ug m-3 points, enlarged. With TAILS=1 (default) a thin line leads from each bias-corrected
   reference issued 1 day ahead to the point it reaches when its decision threshold is lowered until it warns as many
   people as Foehn (small dot).
b, the kilometre contrast: share of the hidden people who are warned by Foehn at 1 km and by the same forecast on 40 km
   cells, for all hidden people and for the people hidden with a margin of 10 % of the threshold on both sides (37.5 ug m-3).
c, the wide net: people warned a day where the mean of the 40 km cell is below the threshold (37.5 ug m-3): hidden people
   (pink) and people wrongly warned (hatched), for every system.
    python fig_hidden_warned_v3.py            -> figs/fig_hidden_warned_v2.pdf, .png
    TAILS=0 python fig_hidden_warned_v3.py    (no tails; files named ..._notails)
    CLIM=1  python fig_hidden_warned_v3.py    (adds the 1 km climatology alone as a seventh system; files named ..._clim)
    LETTERS=fgh python fig_hidden_warned_v3.py   (other panel letters)"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))
sys.path.insert(0, str(FIG))

from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from gc_style import AURORA, CAMS, CLIM, FOEHN, clean, letter_before  # noqa: E402
from nature_style import DOUBLE, MM, plt  # noqa: E402

PINK = "#C2185B"                                                    # hidden people (Fig. 2)
F40 = "#8FB2DD"                                                     # Foehn's own forecast on 40 km cells (light Foehn blue)
BCQ, BCP = "#E69F00", "#009E73"                                     # bias-corrected CAMS forecast: cell value, with the 1 km pattern
INK, KEY = "#1B1E23", "#8A8A8A"                                     # (Okabe-Ito; separation from the four house colours checked)
TAILS = os.environ.get("TAILS", "1") == "1"
WITH_CLIM = os.environ.get("CLIM", "0") == "1"
SYSTEMS = [("foehn_p50", "Foehn", "Foehn", FOEHN), ("foehn_40km", "Foehn on 40 km cells", "Foehn on\n40 km cells", F40),
           ("cams", "CAMS forecast", "CAMS\nforecast", CAMS), ("aurora", "Aurora", "Aurora", AURORA),
           ("cams_qm", "CAMS forecast, bias-corrected", "CAMS forecast,\nbias-corrected", BCQ),
           ("cams_bcsd", "CAMS forecast, bias-corrected + 1 km pattern", "bias-corrected\n+ 1 km pattern", BCP)]
if WITH_CLIM:                                                       # diagnostic variant: shorter names under the bars of panel c
    SYSTEMS[4] = SYSTEMS[4][:2] + ("bias-\ncorrected",) + SYSTEMS[4][3:]
    SYSTEMS[5] = SYSTEMS[5][:2] + ("+ 1 km\npattern",) + SYSTEMS[5][3:]
    SYSTEMS.append(("clim_1km", "1 km climatology alone", "1 km clima-\ntology", CLIM))
THRS = (("15", "o"), ("37.5", "s"), ("75", "^"))
MS = {"o": 3.5, "s": 3.2, "^": 3.7}                                 # equal visual weight of the three shapes
BW = 0.36


def main():
    SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
    N = json.loads((HERE / f"results{SFX}" / "numbers_v3.json").read_text())["systems"]
    plt.rcParams["hatch.linewidth"] = 0.45
    W, H = DOUBLE, 97 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)

    def val(t, h, s, m):
        return N[f"thr{t}"][f"lead{h}"]["World"][s][m]
    # ------------------------------------------------------------------ a: reach against correct warnings
    X0, X1, Y0, Y1 = 27.0, 96.0, 25.0, 95.5
    aw = 76.0; ah = aw * (Y1 - Y0) / (X1 - X0)                         # equal scales on both axes
    aa = fig.add_axes([fx(12.5), fy(11.5), fx(aw), fy(ah)])
    zi = aa.inset_axes([30.0, 71.5, 27.5, 22.5], transform=aa.transData, zorder=6)   # the 15 ug m-3 cluster, enlarged
    ZX, ZY = (85.4, 92.6), (80.2, 91.3)

    def mark(ax, x, y, mk, col, filled, z, scale=1.0):
        ms = MS[mk] * scale
        if filled:
            ax.plot([x], [y], marker=mk, ms=ms + 0.5, mfc=col, mec="white", mew=0.45, ls="none", zorder=z, clip_on=True)
        else:                                                       # no face: a filled marker under an open one stays visible
            ax.plot([x], [y], marker=mk, ms=ms, mfc="none", mec=col, mew=0.8, ls="none", zorder=z + 20, clip_on=True)
    order = [s for s in SYSTEMS if s[0] != "foehn_p50"] + [s for s in SYSTEMS if s[0] == "foehn_p50"]     # Foehn drawn last
    for t, mk in THRS:
        for zo, (s, _, _, col) in enumerate(order):
            for h in ((1,) if s == "clim_1km" else (3, 1)):         # the climatology does not depend on the lead time
                x, y = val(t, h, s, "correct_pct")[0], val(t, h, s, "reach_pct")[0]
                for ax, sc in ((aa, 1.0), (zi, 1.0)):
                    mark(ax, x, y, mk, col, h == 1, 4 + 2 * zo, sc)
                print(f"a: > {t} {s} {h} d: correct {x:.1f} reach {y:.1f}")
        if TAILS:                                                   # the bias-corrected references at Foehn's number of people warned
            for s, col in (("cams_qm", BCQ), ("cams_bcsd", BCP)):
                q = N[f"thr{t}"]["lead1"]["World"]["equal_number_warned"][s]
                x0, y0 = val(t, 1, s, "correct_pct")[0], val(t, 1, s, "reach_pct")[0]
                sw = N[f"thr{t}"]["lead1"]["World"]["decision_threshold_sweep"][s]
                mu_eq = q["decision_threshold_factor"]
                pts = [(x0, y0)] + [(v["correct_pct"], v["reach_pct"]) for mu, v in sorted(sw.items(), key=lambda kv: -float(kv[0])) if mu_eq < float(mu) < 1.0]
                pts.append((q["correct_pct"][0], q["reach_pct"][0]))
                for ax in (aa, zi):
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=0.5, zorder=3, solid_capstyle="round")
                    ax.plot([pts[-1][0]], [pts[-1][1]], marker="o", ms=1.9, mfc=col, mec=col, mew=0, ls="none", zorder=3.5)
                print(f"a: > {t} {s} tail to ({pts[-1][0]:.1f}, {pts[-1][1]:.1f}) at {mu_eq:.2f} x threshold")
    aa.set_xlim(X0, X1); aa.set_ylim(Y0, Y1)
    aa.set_xticks(range(30, 91, 10)); aa.set_yticks(range(30, 91, 10))
    aa.set_xlabel("Warnings that are correct (%)", fontsize=6, labelpad=2); aa.set_ylabel("Exposed people reached (%)", fontsize=6, labelpad=2)
    clean(aa)
    zi.set_xlim(*ZX); zi.set_ylim(*ZY); zi.set_xticks([86, 88, 90, 92]); zi.set_yticks([82, 86, 90])
    zi.tick_params(labelsize=5.0, length=1.5, width=0.4, pad=1.2); zi.grid(lw=0.3, color="#E3E3E3", zorder=0); zi.set_axisbelow(True)
    for sp in zi.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_edgecolor("#9A9A9A")
    zi.set_facecolor("white")
    aa.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0], fill=False, ec="#9A9A9A", lw=0.4, zorder=2.5))
    aa.plot([57.5, ZX[0]], [94.0, ZY[1]], color="#9A9A9A", lw=0.35, zorder=2.5); aa.plot([57.5, ZX[0]], [71.5, ZY[0]], color="#9A9A9A", lw=0.35, zorder=2.5)
    # three separate keys: threshold (shape), lead time (filling); the systems (colour) above the figure
    hT = [Line2D([], [], marker=mk, ms=MS[mk] + 0.3, mfc=KEY, mec=KEY, ls="none") for _, mk in THRS]
    l1 = aa.legend(hT, [f"{t}" for t, _ in THRS], title="Threshold (µg m$^{-3}$)", title_fontsize=5.3, fontsize=5.3, frameon=False, loc="upper left",
                   bbox_to_anchor=(65.5, 50.5), bbox_transform=aa.transData, handletextpad=0.2, borderaxespad=0, labelspacing=0.32, alignment="left")
    aa.add_artist(l1)
    f3 = tuple(Line2D([], [], marker=mk, ms=MS[mk] + 0.3, mfc=KEY, mec=KEY, ls="none") for _, mk in THRS)
    o3 = tuple(Line2D([], [], marker=mk, ms=MS[mk], mfc="white", mec=KEY, mew=0.8, ls="none") for _, mk in THRS)
    hl, ll = [f3, o3], ["1 day", "3 days"]
    l2 = aa.legend(hl, ll, title="Issued ahead", title_fontsize=5.3, fontsize=5.3, frameon=False, loc="upper left", bbox_to_anchor=(83.0, 50.5),
                   bbox_transform=aa.transData, handletextpad=0.5, borderaxespad=0, labelspacing=0.32, handlelength=3.4, alignment="left",
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)})
    aa.add_artist(l2)
    if TAILS:                                                       # the line and dot of the bias-corrected references
        class TailHandler(HandlerTuple):
            def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
                y0, y1 = yd + 0.15 * h, yd + 0.85 * h
                out = []
                for k, col in enumerate((BCQ, BCP)):
                    dx = 0.42 * w * k
                    ln = Line2D([xd + dx + 0.5 * w, xd + dx], [y0, y1], color=col, lw=0.5, transform=trans)
                    dot = Line2D([xd + dx], [y1], marker="o", ms=1.9, mfc=col, mec=col, mew=0, ls="none", transform=trans)
                    out += [ln, dot]
                return out
        l3 = aa.legend([(Line2D([], []),)], ["bias-corrected references\nwarning as many people\nas Foehn (1 day ahead)"], fontsize=5.3, frameon=False,
                       loc="upper left", bbox_to_anchor=(70.0, 36.8), bbox_transform=aa.transData, handletextpad=0.5, borderaxespad=0, handlelength=1.5,
                       handleheight=2.2, handler_map={tuple: TailHandler()})
        aa.add_artist(l3)
    aa.set_title("Reach and correct warnings", fontsize=6.3, loc="left", pad=5)
    # ------------------------------------------------------------------ b: the kilometre contrast (hidden people warned)
    S = N["thr37.5"]
    bx0, bw_ = 104.5, 76.5
    ab = fig.add_axes([fx(bx0), fy(60.5), fx(bw_), fy(ah + 11.5 - 60.5)])
    pair = [s for s in SYSTEMS if s[0] in ("foehn_p50", "foehn_40km")]
    xs = {}
    for g, (m, _) in enumerate((("hidden_warned_pct", "All hidden people"), ("hidden_margin_warned_pct", "Hidden with a margin of 10 %"))):
        for i, (s, _, _, col) in enumerate(pair):
            for j, h in enumerate((1, 3)):
                v = S[f"lead{h}"]["World"][s][m]
                x = g * 2.5 + i + (j - 0.5) * (BW + 0.04); xs[(g, i, j)] = x
                ab.bar(x, v[0], width=BW, color=col, zorder=3, lw=0)
                ab.vlines(x, v[1], v[2], color=INK, lw=0.5, zorder=4)
                print(f"b: {m} {s} {h} d: {v[0]:.1f} % [{v[1]:.1f}, {v[2]:.1f}]")
    ab.set_xticks(list(xs.values())); ab.set_xticklabels(["1 d", "3 d"] * 4, fontsize=5.2); ab.tick_params(axis="x", length=0, pad=1.5)
    for g, lab in enumerate(("All hidden people", "Hidden with a margin of 10 %")):
        ab.text(g * 2.5 + 0.5, -0.155, lab, transform=ab.get_xaxis_transform(), ha="center", va="top", fontsize=5.6, color=INK)
    ab.set_xlim(-0.62, 4.12); ab.set_ylim(0, 50); ab.set_yticks(range(0, 51, 10))
    ab.set_ylabel("Hidden people warned (%)", fontsize=6, labelpad=2)
    clean(ab); ab.grid(axis="x", visible=False)
    ab.legend(handles=[Patch(facecolor=FOEHN, lw=0, label="Foehn, 1 km"), Patch(facecolor=F40, lw=0, label="Foehn on 40 km cells")], fontsize=5.3, frameon=False,
              loc="upper right", bbox_to_anchor=(1.01, 1.04), handlelength=1.3, handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3)
    ab.set_title("Hidden people warned above 37.5 µg m$^{-3}$: 1 km against 40 km cells", fontsize=6.3, loc="left", pad=5)
    # ------------------------------------------------------------------ c: the wide net (warnings where the 40 km mean is below)
    ac = fig.add_axes([fx(bx0), fy(11.5 + 8.0), fx(bw_), fy(26.5)])
    for i, (s, _, _, col) in enumerate(SYSTEMS):
        for j, h in enumerate((1, 3)):
            d = S[f"lead{h}"]["World"][s]
            hid, fa = d["hidden_warned_M"][0], d["false_in_silent40_M"][0]
            x = i + (j - 0.5) * (BW + 0.04)
            ac.bar(x, hid, width=BW, color=PINK, zorder=3, lw=0)
            ac.bar(x, fa, bottom=hid, width=BW, facecolor="white", edgecolor=col, hatch="//////", lw=0.5, zorder=3)
            print(f"c: {s} {h} d: warned where the 40 km mean is below {d['warned_in_silent40_M'][0]:.0f} M = hidden {hid:.1f} + wrongly {fa:.0f} "
                  f"({d['false_per_hidden_warned'][0]:.2f} per hidden person warned)")
    ac.set_xticks([i + (j - 0.5) * (BW + 0.04) for i in range(len(SYSTEMS)) for j in range(2)])
    ac.set_xticklabels(["1 d", "3 d"] * len(SYSTEMS), fontsize=5.0); ac.tick_params(axis="x", length=0, pad=1.5)
    for i, (_, _, name, col) in enumerate(SYSTEMS):                    # the two long names are moved apart by 0.07 of a slot
        ac.text(i + ({} if WITH_CLIM else {4: -0.07, 5: 0.07}).get(i, 0.0), -0.17, name, transform=ac.get_xaxis_transform(), ha="center", va="top", fontsize=5.0, color=col,
                fontweight="bold", linespacing=0.95)
    ac.set_xlim(-0.55, len(SYSTEMS) - 0.45); ac.set_ylim(0, 700); ac.set_yticks(range(0, 601, 200))
    ac.set_ylabel("People warned a day\n(millions)", fontsize=6, labelpad=2)
    clean(ac); ac.grid(axis="x", visible=False)
    ac.legend(handles=[Patch(facecolor="white", edgecolor=KEY, hatch="//////", lw=0.5, label="people wrongly warned"),
                       Patch(facecolor=PINK, lw=0, label="hidden people warned")],
              fontsize=5.3, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.06), ncol=1, handlelength=1.3, handletextpad=0.5, borderaxespad=0.2,
              labelspacing=0.3)
    ac.set_title("People warned above 37.5 µg m$^{-3}$ where the 40 km mean is below", fontsize=6.3, loc="left", pad=5)
    # ------------------------------------------------------------------ the systems (colour), one key for the three panels
    hs = [Line2D([], [], marker="s", ms=3.6, mfc=col, mec=col, ls="none", label=name) for _, name, _, col in SYSTEMS]
    fig.legend(handles=hs, fontsize=5.5, frameon=False, loc="lower left", bbox_to_anchor=(fx(9.0), fy(ah + 11.5 + 6.2)), ncol=len(SYSTEMS), handletextpad=0.1,
               columnspacing=1.25, borderaxespad=0)
    fig.canvas.draw()                                                  # letters just before the titles (fig_hidden.py, panel e)
    let = os.environ.get("LETTERS", "abc")
    for a_, s_ in zip((aa, ab, ac), let):
        letter_before(fig, a_._left_title, s_)
    FIGD = HERE.parents[2] / "rebuild" / f"figs{SFX}" if SFX else HERE / "figs"   # renders of a design: natcomms/rebuild/figs_<D>
    FIGD.mkdir(exist_ok=True)
    name = "fig_hidden_warned_v3" + ("" if TAILS else "_notails") + ("_clim" if WITH_CLIM else "") + ("" if let == "abc" else f"_{let}")
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"{name}.{ext}")
    print("saved", FIGD / f"{name}.pdf")


if __name__ == "__main__":
    main()
