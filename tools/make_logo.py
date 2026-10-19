#!/usr/bin/env python3
"""Foehn logo (animated SVG): a disc of kilometre pixels that a generator turns from noise into a field of PM2.5, in the
four steps of Foehn's own sampler (flow times 0.03, 0.17, 0.53, 1: each pixel = (1 - tau) noise + tau forecast), holds,
then dissolves back into noise; the word Foehn beside it (no tagline). SMIL animation, so it plays wherever the SVG is shown as an
image (GitHub README, the site). Colours of the PM2.5 classes of the paper.
  docs/assets/logo/foehn_mark.svg        the disc alone (any background)
  docs/assets/logo/foehn_logo_light.svg  disc + word, for light pages      foehn_logo_dark.svg  for dark pages
  docs/assets/logo/foehn_mark.png        a still of the formed disc (favicon, social cards)
    python tools/make_logo.py"""
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "logo")
PM = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026", "#67001f"]
TAU = [0.0, 0.03, 0.17, 0.53, 1.0]                 # the flow times of the sampler: noise, three steps, forecast
DUR = 7.0                                          # seconds per loop


def hex2rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def rgb2hex(c):
    c = np.clip(np.round(c), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c)


def field(x, y):
    """a plume of haze across the disc, with a hot core: 0 clean .. 1 very polluted"""
    u, v = 0.82 * x + 0.57 * y, -0.57 * x + 0.82 * y
    plume = np.exp(-((v + 0.12 - 0.18 * u ** 2) / 0.34) ** 2) * (0.55 + 0.45 * np.tanh(2.2 * (u + 0.35)))
    core = np.exp(-(((x - 0.28) / 0.22) ** 2 + ((y - 0.18) / 0.2) ** 2))
    return np.clip(0.06 + 0.7 * plume + 0.45 * core, 0, 1)


def mark(cx, cy, R, n=17, rng=None, gap=0.16):
    """the pixel disc: SVG elements, each pixel animated noise -> 4 steps -> forecast -> hold -> noise"""
    rng = rng or np.random.default_rng(7)
    pitch = 2 * R / n; s = pitch * (1 - gap); els = []
    for i in range(n):
        for j in range(n):
            x, y = -1 + (j + 0.5) * 2 / n, 1 - (i + 0.5) * 2 / n
            if x * x + y * y > (1 - 0.5 / n) ** 2:
                continue
            fin = hex2rgb(PM[int(np.clip(field(x, y) * len(PM), 0, len(PM) - 1))])
            g = rng.uniform(70, 200); noise = np.array([g, g, g * 1.05])
            cols = [rgb2hex((1 - t) * noise + t * fin) for t in TAU]
            # key times: noise until 0.10, the four steps 0.10-0.46 (each held a beat), forecast held to 0.86, back to noise at 1
            d = rng.uniform(0, 0.035)
            kt = [0, 0.10 + d, 0.19 + d, 0.28 + d, 0.37 + d, 0.46 + d, 0.86, 1]
            vals = [cols[0], cols[0], cols[1], cols[2], cols[3], cols[4], cols[4], cols[0]]
            px, py = cx + x * R - s / 2, cy - y * R - s / 2
            els.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{s:.2f}" height="{s:.2f}" rx="{s * 0.22:.2f}" fill="{cols[4]}">'
                       f'<animate attributeName="fill" dur="{DUR}s" repeatCount="indefinite" calcMode="linear" '
                       f'keyTimes="{";".join(f"{k:.3f}" for k in kt)}" values="{";".join(vals)}"/></rect>')
    ring = (f'<circle cx="{cx}" cy="{cy}" r="{R + pitch * 0.55:.2f}" fill="none" stroke="#58C4DD" stroke-width="{max(1.2, R / 40):.2f}" '
            f'stroke-opacity="0.55"><animate attributeName="stroke-opacity" dur="{DUR}s" repeatCount="indefinite" '
            f'keyTimes="0;0.46;0.6;0.86;1" values="0.25;0.25;0.85;0.55;0.25"/></circle>')
    return "\n".join([ring] + els)


def svg(w, h, body, title="Foehn"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{title}">\n'
            f'<title>{title}</title>\n{body}\n</svg>\n')


def word(x, y, ink):
    font = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    return f'<text x="{x}" y="{y}" font-family="{font}" font-size="96" font-weight="800" letter-spacing="-2" fill="{ink}">Foehn</text>'


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
            c = tuple(int(v) for v in hex2rgb(PM[int(np.clip(field(x, y) * len(PM), 0, len(PM) - 1))]))
            px, py = cx + x * R - s / 2, cy - y * R - s / 2
            dr.rounded_rectangle([px, py, px + s, py + s], radius=s * 0.22, fill=c + (255,))
    rr = R + pitch * 0.55
    dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(88, 196, 221, 170), width=int(max(1.65, 66 / 40) * S))
    return im.resize((size, size), Image.LANCZOS)


def main():
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/foehn_mark.svg", "w").write(svg(160, 160, mark(80, 80, 66)))
    for name, ink in (("light", "#0F1B2A"), ("dark", "#E8EAED")):
        open(f"{OUT}/foehn_logo_{name}.svg", "w").write(svg(480, 160, mark(80, 80, 66) + "\n" + word(176, 114, ink)))
    open(f"{OUT}/foehn_mark_small.svg", "w").write(svg(160, 160, mark(80, 80, 66, n=9)))      # navigation bar, favicon
    still(17, 256).save(f"{OUT}/foehn_mark.png"); still(9, 64).save(f"{OUT}/favicon.png"); still(9, 180).save(f"{OUT}/apple-touch-icon.png")
    print("written:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
