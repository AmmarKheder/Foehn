#!/usr/bin/env python3
"""Foehn logo (animated SVG): a disc of kilometre pixels that the generator turns from noise into a field of PM2.5 in the
four steps of Foehn's own sampler (flow times 0.03, 0.17, 0.53, 1; each pixel = (1 - tau) noise + tau forecast); the
plume then drifts with the wind across the disc (the computed transport), holds, and dissolves back into noise. The word
Foehn beside it, as outlines (Liberation Sans Bold converted to paths, so that it looks the same everywhere). SMIL
animation, so it plays wherever the SVG is shown as an image (GitHub README, the site). Colours of the PM2.5 classes.
  docs/assets/logo/foehn_mark.svg         the disc alone (17 x 17), any background
  docs/assets/logo/foehn_mark_small.svg   9 x 9, for the navigation bar
  docs/assets/logo/foehn_logo_light.svg   disc + word for light pages; foehn_logo_dark.svg for dark pages
  docs/assets/logo/foehn_mark.png, favicon.png, apple-touch-icon.png   stills of the formed disc
    python tools/make_logo.py      (world venv: numpy, matplotlib, PIL)"""
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "logo")
PM = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026", "#67001f"]
TAU = [0.0, 0.03, 0.17, 0.53, 1.0]                 # the flow times of the sampler: noise, three steps, forecast
DUR = 10.0                                         # seconds per loop
DRIFT = 0.5                                        # how far the plume travels (disc radii) while it drifts
NDRIFT = 7                                         # key frames of the drift


def hex2rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def rgb2hex(c):
    c = np.clip(np.round(c), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c)


def field(x, y, s=0.0):
    """a plume of haze across the disc with a hot core, carried a distance s along the wind: 0 clean .. 1 very polluted"""
    u, v = 0.82 * x + 0.57 * y - s, -0.57 * x + 0.82 * y
    plume = np.exp(-((v + 0.12 - 0.18 * u ** 2) / 0.34) ** 2) * (0.55 + 0.45 * np.tanh(2.2 * (u + 0.35)))
    core = np.exp(-(((u - 0.31) / 0.24) ** 2 + ((v - 0.03) / 0.2) ** 2))
    return np.clip(0.06 + 0.7 * plume + 0.45 * core, 0, 1)


def colour(x, y, s=0.0):
    return hex2rgb(PM[int(np.clip(field(x, y, s) * len(PM), 0, len(PM) - 1))])


def mark(cx, cy, R, n=17, rng=None, gap=0.16):
    """the pixel disc: SVG elements, each pixel animated noise -> 4 steps -> forecast -> drift -> hold -> noise"""
    rng = rng or np.random.default_rng(7)
    pitch = 2 * R / n; s = pitch * (1 - gap); els = []
    # key times: noise to 0.07; the four steps 0.07-0.31 (each held a beat); the drift 0.31-0.80; hold to 0.90; noise at 1
    for i in range(n):
        for j in range(n):
            x, y = -1 + (j + 0.5) * 2 / n, 1 - (i + 0.5) * 2 / n
            if x * x + y * y > (1 - 0.5 / n) ** 2:
                continue
            fin = colour(x, y); g = rng.uniform(70, 200); noise = np.array([g, g, g * 1.05])
            cols = [rgb2hex((1 - t) * noise + t * fin) for t in TAU]
            d = rng.uniform(0, 0.025)
            kt = [0, 0.07 + d, 0.13 + d, 0.19 + d, 0.25 + d, 0.31 + d]
            vals = [cols[0], cols[0], cols[1], cols[2], cols[3], cols[4]]
            for q in range(1, NDRIFT + 1):
                kt.append(0.31 + d + 0.49 * q / NDRIFT); vals.append(rgb2hex(colour(x, y, DRIFT * q / NDRIFT)))
            kt += [0.90, 1.0]; vals += [vals[-1], cols[0]]
            px, py = cx + x * R - s / 2, cy - y * R - s / 2
            els.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{s:.2f}" height="{s:.2f}" rx="{s * 0.22:.2f}" fill="{cols[4]}">'
                       f'<animate attributeName="fill" dur="{DUR}s" repeatCount="indefinite" calcMode="linear" '
                       f'keyTimes="{";".join(f"{k:.3f}" for k in kt)}" values="{";".join(vals)}"/></rect>')
    ring = (f'<circle cx="{cx}" cy="{cy}" r="{R + pitch * 0.55:.2f}" fill="none" stroke="#58C4DD" stroke-width="{max(1.2, R / 40):.2f}" '
            f'stroke-opacity="0.55"><animate attributeName="stroke-opacity" dur="{DUR}s" repeatCount="indefinite" '
            f'keyTimes="0;0.31;0.4;0.9;1" values="0.25;0.25;0.85;0.55;0.25"/></circle>')
    return "\n".join([ring] + els)


def svg(w, h, body, title="Foehn"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{title}">\n'
            f'<title>{title}</title>\n{body}\n</svg>\n')


def word_paths(x, y, size, ink, spacing=-2.5):
    """the word Foehn as outlines (Liberation Sans Bold, metric-compatible with Arial), one path per letter"""
    from matplotlib import font_manager as fm
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    fp = FontProperties(fname=fm.findfont(FontProperties(family="Liberation Sans", weight="bold")))
    out, cx = [], x
    for ch in "Foehn":
        tp = TextPath((0, 0), ch, size=size, prop=fp)
        d = []
        for verts, code in tp.iter_segments():
            v = [f"{cx + verts[k]:.2f},{y - verts[k + 1]:.2f}" for k in range(0, len(verts), 2)]
            d.append({1: "M ", 2: "L ", 3: "Q ", 4: "C ", 79: "Z"}[code] + " ".join(v))
        out.append(f'<path d="{" ".join(d)}" fill="{ink}" fill-rule="evenodd"/>')
        cx += tp.get_extents().x1 + spacing + (size * 0.08 if ch == "F" else 0)
    return "\n".join(out), cx


def still(n, size):
    """the formed disc as a PNG (favicon, cards)"""
    from PIL import Image, ImageDraw
    S = 4; W = 160 * S; im = Image.new("RGBA", (W, W), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    R, cx, cy = 66 * S, 80 * S, 80 * S; pitch = 2 * R / n; s = pitch * 0.84
    for i in range(n):
        for j in range(n):
            x, y = -1 + (j + 0.5) * 2 / n, 1 - (i + 0.5) * 2 / n
            if x * x + y * y > (1 - 0.5 / n) ** 2:
                continue
            c = tuple(int(v) for v in colour(x, y))
            px, py = cx + x * R - s / 2, cy - y * R - s / 2
            dr.rounded_rectangle([px, py, px + s, py + s], radius=s * 0.22, fill=c + (255,))
    rr = R + pitch * 0.55
    dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(88, 196, 221, 170), width=int(max(1.65, 66 / 40) * S))
    return im.resize((size, size), Image.LANCZOS)


def main():
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/foehn_mark.svg", "w").write(svg(160, 160, mark(80, 80, 66)))
    open(f"{OUT}/foehn_mark_small.svg", "w").write(svg(160, 160, mark(80, 80, 66, n=9)))
    for name, ink in (("light", "#0F1B2A"), ("dark", "#E8EAED")):
        paths, xe = word_paths(178, 116, 100, ink)
        open(f"{OUT}/foehn_logo_{name}.svg", "w").write(svg(int(xe + 10), 160, mark(80, 80, 66) + "\n" + paths))
    still(17, 256).save(f"{OUT}/foehn_mark.png"); still(9, 64).save(f"{OUT}/favicon.png"); still(9, 180).save(f"{OUT}/apple-touch-icon.png")
    print("written:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
