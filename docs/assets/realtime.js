/* Foehn in real time: the script of the section #realtime, fetched when the section comes near the screen.
 *   1. the index of the latest forecast and a world view made of four tiles shown as plain images;
 *   2. on request ("Open the street map"): MapLibre GL JS (vendor/, 3-Clause BSD) and the OpenFreeMap basemap;
 *   3. when its block is reached: the daily verification at the monitors.
 * Nothing is kept on the device (no storage, no service worker, no counter, no position request), and no text of the
 * downloaded files is written into the page: only dates, numbers and colours, after a check of their form.
 */
(() => {
  "use strict";

  // ---- The ONE address of the data. Everything this section downloads comes from it:
  //        live1km/latest.json · live1km/<day>/tiles.json · live1km/<day>/mean/<lead>/<z>/<x>/<y>.png · verif/history.json
  //      To serve the data from another place, change this line only (same file names under the new address).
  const DATA = "https://raw.githubusercontent.com/AmmarKheder/Foehn/";

  const TILES = DATA + "live1km/", VERIF = DATA + "verif/history.json";
  const BASEMAP = "https://tiles.openfreemap.org/styles/";  // street basemap (OpenFreeMap), asked for only when the map is opened
  const VENDOR = "vendor/maplibre-gl";  // MapLibre GL JS 4.7.1 (.js and .css), loaded only when the map is opened
  const PROTO = "rt1k";  // tiles of the forecast layer: rt1k://mean/<lead>/{z}/{x}/{y}
  const EQUATOR_KM = 40075.017, WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];

  const $ = (id) => document.getElementById(id);
  const card = $("rt-card"); if (!card || card.dataset.rt) return; card.dataset.rt = "1";
  const S = { latest: null, meta: null, day: 1, bits: {}, dec: new Map(), lut: null, map: null, styleReady: false, live: false, opening: false,
    popup: null, popupAt: null, timer: null, opacity: 1, verif: null, errors: [] };
  if (/[?&]debug=1(&|$)/.test(location.search)) window.__rt = S;  // test handle, on request only

  // ---- small helpers
  const el = (tag, attrs, ...kids) => { const e = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) { if (v == null || v === false) continue; if (k === "class") e.className = v; else e.setAttribute(k, v); }
    for (const k of kids) if (k != null) e.append(k); return e; };
  const isDay = (s) => typeof s === "string" && /^\d{4}-\d{2}-\d{2}$/.test(s);
  const isHex = (s) => typeof s === "string" && /^#[0-9a-fA-F]{6}$/.test(s);
  const dfmt = (iso, o) => new Date(iso + "T12:00:00Z").toLocaleDateString("en-GB", Object.assign({ timeZone: "UTC" }, o));
  const todayIso = () => new Date().toISOString().slice(0, 10);
  const dayDiff = (a, b) => Math.round((Date.parse(a + "T00:00:00Z") - Date.parse(b + "T00:00:00Z")) / 864e5);
  const fmt = (x) => (+x).toLocaleString("en-GB", { minimumFractionDigits: x >= 10 ? 0 : 1, maximumFractionDigits: x >= 10 ? 0 : 1 });
  const int = (x) => Math.round(x).toLocaleString("en-GB");
  const sgn = (x, d) => (x < 0 ? "−" : "+") + Math.abs(x).toFixed(d);
  const word = (n) => (Number.isInteger(n) && n >= 0 && n <= 10 ? WORDS[n] : String(n));
  const coord = (lat, lon) => Math.abs(lat).toFixed(2) + "° " + (lat >= 0 ? "N" : "S") + ", " + Math.abs(lon).toFixed(2) + "° " + (lon >= 0 ? "E" : "W");
  const pixelKm = (z) => (EQUATOR_KM / (256 * Math.pow(2, z))).toFixed(1);
  const isDark = () => { const t = document.documentElement.dataset.theme; return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; };
  const getJson = async (url) => { const r = await fetch(url, { cache: "no-cache" }); if (!r.ok) throw new Error("http " + r.status); return r.json(); };
  const msg = (t) => { const m = $("rt-msg"); m.textContent = t || ""; m.hidden = !t; };
  const members = () => (S.meta && Number.isInteger(S.meta.members) ? word(S.meta.members) + " " : "");

  // ---- stage 1: the index of the latest forecast (latest.json, then the tiles.json it names)
  function checkMeta(m) {  // the fields used below, in the form expected
    const e = m.encoding, lay = m.layers && m.layers.mean;
    const ok = isDay(m.date) && Array.isArray(m.leads) && m.leads.length > 0 && lay &&
      m.leads.every((L) => Number.isInteger(L) && L > 0 && L < 30 && lay[L] && isDay(lay[L].valid) && Number.isInteger(lay[L].zmax)) &&
      e && Array.isArray(e.bounds) && e.bounds.length === 11 && e.bounds.every(Number.isFinite) && Array.isArray(e.colors) && e.colors.length === 10 &&
      e.colors.every(isHex) && isHex(e.over) && Number.isFinite(e.top) && Array.isArray(e.n) && Array.isArray(e.base) && e.n.length === e.base.length &&
      e.n.every(Number.isInteger) && e.base.every(Number.isInteger) && m.tiles && m.tiles.mean;
    if (!ok) throw new Error("index");
  }
  async function loadIndex(tries) {
    try {
      const latest = await getJson(TILES + "latest.json");
      if (typeof latest.tiles_json !== "string" || !/^\d{4}-\d{2}-\d{2}\/tiles\.json$/.test(latest.tiles_json)) throw new Error("latest");
      const meta = await getJson(TILES + latest.tiles_json); checkMeta(meta);
      S.latest = latest; S.meta = meta; S.dec.clear(); S.bits = {};
      for (const [z, b64] of Object.entries(meta.tiles.mean)) S.bits[z] = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      buildLut();
      S.day = meta.leads.find((L) => meta.layers.mean[L].valid === todayIso()) || meta.leads[0];
      issueLine(); days(); legend(); help(); coarseNote(); poster();
      $("rt-open").hidden = S.live || S.opening; if (S.map) addLayers();
    } catch (e) {
      const again = (tries || 0) < 5;
      $("rt-issue").textContent = again ? "The latest forecast is not available at the moment; a new attempt is made every minute." : "The latest forecast could not be loaded. Please come back later.";
      if (again) setTimeout(() => loadIndex((tries || 0) + 1), 60000);
    }
  }
  let lastCheck = 0;
  async function refresh() {  // a tile is missing: the data hold only the latest day, so the forecast on screen may have been replaced by the next one
    if (Date.now() - lastCheck < 60000) return; lastCheck = Date.now();
    try { const l = await getJson(TILES + "latest.json"); if (S.meta && isDay(l.latest) && l.latest !== S.meta.date) loadIndex(0); } catch (e) {}
  }
  function issueLine() {  // date and hour of issue, read from the two files
    const m = S.meta, up = S.latest && typeof S.latest.updated === "string" && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(S.latest.updated) ? S.latest.updated : null;
    let t = "Latest forecast: issued on " + dfmt(m.date, { weekday: "long", day: "numeric", month: "long", year: "numeric" }) + " from the 00 UTC analysis";
    if (up) t += ", published " + (up.slice(0, 10) === m.date ? "" : "on " + dfmt(up.slice(0, 10), { day: "numeric", month: "long" }) + " ") + "at " + up.slice(11, 16) + " UTC";
    $("rt-issue").textContent = t + (Number.isInteger(m.members) ? " · " + m.members + " members." : ".");
    const age = dayDiff(todayIso(), m.date), st = $("rt-stale"); st.hidden = age < 1;
    if (age === 1) st.textContent = "Today's run is not published yet (it usually arrives around midday UTC). This is the forecast issued yesterday: its day 2 is today.";
    else if (age > 1) st.textContent = "This forecast is " + age + " days old: the more recent runs are late or missing.";
  }
  function days() {
    const box = $("rt-days"), m = S.meta; box.textContent = "";
    box.append(el("button", { type: "button", class: "play", id: "rt-play", "aria-label": S.timer ? "Stop" : "Play through the days" }, S.timer ? "❚❚" : "▶"));
    for (const L of m.leads) { const v = m.layers.mean[L].valid;
      box.append(el("button", { type: "button", "data-lead": L, class: L === S.day ? "on" : null, "aria-pressed": L === S.day ? "true" : "false" },
        el("span", null, dfmt(v, { weekday: "short", day: "numeric" })), el("span", { class: "rt-mo" }, " " + dfmt(v, { month: "short" })), el("small", null, "day " + L))); }
  }
  function setDay(L) {
    if (!S.meta || !S.meta.leads.includes(L)) return; S.day = L;
    for (const b of $("rt-days").querySelectorAll("button[data-lead]")) { const on = +b.dataset.lead === L; b.classList.toggle("on", on); b.setAttribute("aria-pressed", on ? "true" : "false"); }
    help(); if (S.map) { addLayers(); if (S.popupAt) readout(S.popupAt.lng, S.popupAt.lat); } if (!S.live) poster();
  }
  function legend() {  // the classes of the tiles: bounds and colours as published with them
    const e = S.meta.encoding, cols = e.colors.concat([e.over]), box = $("rt-legend"); box.textContent = "";
    const unit = el("span", { class: "rt-lunit" }); unit.innerHTML = "PM<sub>2.5</sub>, daily mean (µg&nbsp;m<sup>−3</sup>)";
    const sc = el("div", { class: "rt-lscale" });
    cols.forEach((c, i) => sc.append(el("div", null, el("i", { style: "background:" + c }), el("b", null, String(e.bounds[i])))));
    box.append(unit, sc, el("span", { class: "rt-lhatch" }, el("i"), "no 1\u00a0km value: 0.4° forecast or interpolation"));
  }
  function help() {
    const m = S.meta, lay = m.layers.mean[S.day];
    $("rt-help").textContent = "Day " + S.day + " of " + m.leads.length + ", " + dfmt(lay.valid, { weekday: "long", day: "numeric", month: "long" }) + " (UTC day): mean of the " + members() +
      "members' daily means, published at about " + pixelKm(lay.zmax) + "\u00a0km per pixel at the equator." + (S.live ? " Tap or click the map to read a value." : " The street map is loaded on request, with a basemap from OpenFreeMap.");
  }
  function coarseNote() {  // which days are published below the model's resolution
    const m = S.meta, lay = m.layers.mean, top = Math.max(...m.leads.map((L) => lay[L].zmax)), low = m.leads.filter((L) => lay[L].zmax < top), high = m.leads.filter((L) => lay[L].zmax === top);
    const span = (a) => (a.length === 1 ? "day " + a[0] : a[a.length - 1] - a[0] === a.length - 1 && a.length > 2 ? "days " + a[0] + " to " + a[a.length - 1] : "days " + a.slice(0, -1).join(", ") + " and " + a[a.length - 1]);
    const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
    $("rt-coarse").textContent = low.length ? cap(span(low)) + (low.length > 1 ? " are" : " is") + " computed at 1\u00a0km but published at about " + pixelKm(lay[low[0]].zmax) + "\u00a0km per pixel at the equator (" + span(high) + ": about " + pixelKm(top) + "\u00a0km)."
      : "The " + word(m.leads.length) + " days are published at about " + pixelKm(top) + "\u00a0km per pixel at the equator.";
  }
  // the world view: the four zoom-1 tiles shown as plain images (their palette holds the class colours), one world across the box.
  // Each tile is fetched once and shown through an object URL, so that no element of the page carries the address of the data.
  const tileBlob = new Map(), wanted = new WeakMap();
  function show(im, url) {
    wanted.set(im, url); if (!url) { im.removeAttribute("src"); return; }
    let p = tileBlob.get(url);
    if (!p) { p = fetch(url).then((r) => { if (!r.ok) throw new Error("tile " + r.status); return r.blob(); }).then((b) => URL.createObjectURL(b)); tileBlob.set(url, p); p.catch(() => tileBlob.delete(url)); }
    p.then((u) => { if (wanted.get(im) === url) im.src = u; }, () => { if (wanted.get(im) === url) im.removeAttribute("src"); refresh(); });
  }
  function poster() {
    const p = $("rt-poster"), m = S.meta; if (!m) return;
    let grid = p.querySelector(".rt-tiles");
    if (!grid) { grid = el("div", { class: "rt-tiles" }); for (let i = 0; i < 4; i++) grid.append(el("img", { alt: "", decoding: "async", draggable: "false" })); p.append(el("div", { class: "rt-world" }, grid)); }
    [...grid.children].forEach((im, i) => { const x = i % 2, y = i >> 1; show(im, hasTile(1, x, y) ? tileUrl(S.day, 1, x, y) : null); });
    p.setAttribute("aria-label", "World map of the forecast daily mean PM2.5 for " + dfmt(m.layers.mean[S.day].valid, { weekday: "long", day: "numeric", month: "long", year: "numeric" }));
  }
  $("rt-days").addEventListener("click", (ev) => {
    const b = ev.target.closest("button"); if (!b || !S.meta) return;
    if (b.id === "rt-play") {
      if (S.timer) { clearInterval(S.timer); S.timer = null; } else S.timer = setInterval(() => { const L = S.meta.leads; setDay(L[(L.indexOf(S.day) + 1) % L.length]); }, 1500);
      b.textContent = S.timer ? "❚❚" : "▶"; b.setAttribute("aria-label", S.timer ? "Stop" : "Play through the days"); return; }
    if (b.dataset.lead) setDay(+b.dataset.lead);
  });
  $("rt-days").addEventListener("keydown", (ev) => {
    if (!S.meta || (ev.key !== "ArrowRight" && ev.key !== "ArrowLeft")) return;
    const L = S.meta.leads, i = L.indexOf(S.day) + (ev.key === "ArrowRight" ? 1 : -1); if (i < 0 || i >= L.length) return;
    ev.preventDefault(); setDay(L[i]); const b = $("rt-days").querySelector('button[data-lead="' + L[i] + '"]'); if (b) b.focus();
  });

  // ---- the tiles: 256 x 256 palette PNGs; the palette index gives the class, a sub-bin of the value and the source of the pixel
  const hasTile = (z, x, y) => { const b = S.bits[z], n = 1 << z; if (!b || x < 0 || y < 0 || x >= n || y >= n) return false; const i = y * n + x; return ((b[i >> 3] >> (7 - (i & 7))) & 1) === 1; };
  const tileUrl = (L, z, x, y) => TILES + S.meta.date + "/mean/" + L + "/" + z + "/" + x + "/" + y + ".png";
  const zmaxOf = (L) => (S.meta && S.meta.layers.mean[L] ? S.meta.layers.mean[L].zmax : 6);
  function decodeMean(i) {  // palette index -> value (centre of its bin), class, source
    const e = S.meta.encoding; if (!i) return null;
    let s = 0; for (let q = 0; q < e.base.length; q++) if (i >= e.base[q]) s = q;
    const B = e.bounds.concat([e.top]), off = i - e.base[s], c = Math.floor(off / e.n[s]), sub = off % e.n[s]; if (c > 10) return null;
    const a = B[c], b = B[c + 1], f0 = sub / e.n[s], f1 = (sub + 1) / e.n[s];
    return { v: c === 0 ? a + (b - a) * (f0 + f1) / 2 : a * Math.pow(b / a, (f0 + f1) / 2), c, src: s };
  }
  function buildLut() {  // colour (of the class) and source of every palette index
    const e = S.meta.encoding, cols = e.colors.concat([e.over]).map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)));
    const rgb = new Uint8Array(768), src = new Uint8Array(256).fill(255);
    for (let s = 0; s < e.n.length; s++) for (let c = 0; c < 11; c++) for (let j = 0; j < e.n[s]; j++) { const i = e.base[s] + c * e.n[s] + j; if (i < 1 || i > 255) continue; rgb.set(cols[c], i * 3); src[i] = s; }
    S.lut = { rgb, src };
  }
  const toImage = (data, w, h) => createImageBitmap(new ImageData(data, w, h));
  const blank = () => toImage(new Uint8ClampedArray(4), 1, 1);
  function renderTile(ix, z, x, y, sh, nx, ny) {  // a displayed tile from the native one: class colours, kept sharp when enlarged; light hatching where the value is not from the 1 km model
    const L = S.lut, out = new Uint8ClampedArray(65536 * 4), sc = Math.pow(2, sh), cm = new Int32Array(256);
    for (let u = 0; u < 256; u++) cm[u] = Math.floor(((x - nx * sc) * 256 + u) / sc);
    for (let v = 0; v < 256; v++) {
      const rowN = Math.floor(((y - ny * sc) * 256 + v) / sc) * 256, Y = y * 256 + v;
      for (let u = 0; u < 256; u++) {
        const i = ix[rowN + cm[u]]; if (!i || L.src[i] === 255) continue;
        const o = (v * 256 + u) * 4, p = i * 3, hatch = L.src[i] !== 0 && (x * 256 + u + Y) % 8 === 0;
        out[o] = hatch ? L.rgb[p] * 0.8 + 51 : L.rgb[p]; out[o + 1] = hatch ? L.rgb[p + 1] * 0.8 + 51 : L.rgb[p + 1]; out[o + 2] = hatch ? L.rgb[p + 2] * 0.8 + 51 : L.rgb[p + 2]; out[o + 3] = 255;
      }
    }
    return out;
  }
  async function inflate(bytes) { const ds = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate")); return new Uint8Array(await new Response(ds).arrayBuffer()); }
  async function pngIndices(buf) {  // PNG reader (palette, 8 bits or fewer, not interlaced): the palette index of every pixel
    const dv = new DataView(buf), u8 = new Uint8Array(buf); let p = 8, w = 0, h = 0, bd = 8; const parts = [];
    while (p + 8 <= u8.length) { const len = dv.getUint32(p), type = String.fromCharCode(u8[p + 4], u8[p + 5], u8[p + 6], u8[p + 7]);
      if (type === "IHDR") { w = dv.getUint32(p + 8); h = dv.getUint32(p + 12); bd = u8[p + 16]; } else if (type === "IDAT") parts.push(u8.subarray(p + 8, p + 8 + len)); else if (type === "IEND") break;
      p += 12 + len; }
    if (w !== 256 || h !== 256) throw new Error("tile size");
    const z = new Uint8Array(parts.reduce((a, x) => a + x.length, 0)); let o = 0; for (const x of parts) { z.set(x, o); o += x.length; }
    const d = await inflate(z), bpr = Math.ceil(w * bd / 8), out = new Uint8Array(w * h); let prev = new Uint8Array(bpr), cu = new Uint8Array(bpr), q = 0;
    for (let y = 0; y < h; y++) {
      const f = d[q++];
      for (let i = 0; i < bpr; i++) { const x = d[q++], a = i ? cu[i - 1] : 0, b = prev[i], c = i ? prev[i - 1] : 0; let v;
        if (f === 0) v = x; else if (f === 1) v = x + a; else if (f === 2) v = x + b; else if (f === 3) v = x + ((a + b) >> 1);
        else { const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c); v = x + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c); }
        cu[i] = v & 255; }
      if (bd === 8) out.set(cu.subarray(0, w), y * w);
      else { const per = 8 / bd, mask = (1 << bd) - 1; for (let x = 0; x < w; x++) out[y * w + x] = (cu[(x / per) | 0] >> (8 - bd * ((x % per) + 1))) & mask; }
      const t = prev; prev = cu; cu = t;
    }
    return out;
  }
  function nativeIndices(L, z, x, y) {  // one download per native tile, shared by the map and the readout
    const key = L + "/" + z + "/" + x + "/" + y; let p = S.dec.get(key);
    if (!p) { p = fetch(tileUrl(L, z, x, y)).then((r) => { if (!r.ok) throw new Error("tile " + r.status); return r.arrayBuffer(); }).then(pngIndices);
      S.dec.set(key, p); p.catch(() => { S.dec.delete(key); refresh(); }); if (S.dec.size > 160) S.dec.delete(S.dec.keys().next().value); }
    return p;
  }
  function tileXY(lng, lat, z) {
    const n = 1 << z, x = (lng + 180) / 360 * n, r = lat * Math.PI / 180, y = (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * n, tx = Math.floor(x), ty = Math.floor(y);
    return { tx: ((tx % n) + n) % n, ty, px: Math.min(255, Math.floor((x - tx) * 256)), py: Math.min(255, Math.floor((y - ty) * 256)) };
  }
  async function pixel(L, lng, lat) {  // the palette index of the pixel at a position (0: none)
    if (!S.meta || lat >= 85 || lat <= -85) return 0;
    const z = zmaxOf(L), t = tileXY(lng, lat, z); if (!hasTile(z, t.tx, t.ty)) return 0;
    return (await nativeIndices(L, z, t.tx, t.ty))[t.py * 256 + t.px];
  }

  // ---- stage 2: the street map (MapLibre GL JS and the OpenFreeMap basemap), opened on request
  const addCss = (href) => new Promise((res, rej) => { const l = el("link", { rel: "stylesheet", href }); l.onload = res; l.onerror = rej; document.head.append(l); });
  const addJs = (src) => new Promise((res, rej) => { const s = el("script", { src }); s.onload = res; s.onerror = rej; document.head.append(s); });
  const hasGl = () => { try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); } catch (e) { return false; } };
  const styleUrl = () => BASEMAP + (isDark() ? "dark" : "positron");
  // the view on which the map opens: one world across the box, as the world view it replaces; in a tall box (phones) the band 85° N – 60° S fills the height
  function worldView() {
    const b = $("rt-gl"), w = Math.max(256, b.clientWidth), h = b.clientHeight, W = Math.max(w, h / 0.75), f = h / W, y = Math.min(Math.max(0.4406, f / 2), Math.max(f / 2, 0.75 - f / 2));
    return { center: [W > w ? 10 : 0, Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180 / Math.PI], zoom: Math.log2(W / 512) };
  }
  // the forecast is opaque on the world and lets the map show through at city scale; the legend follows, so that its colours stay those of the map
  const fade = (z) => (z <= 5 ? 1 : z >= 10 ? 0.78 : 1 - 0.22 * (z - 5) / 5);
  const opacityExpr = () => ["interpolate", ["linear"], ["zoom"], 5, S.opacity, 10, +(0.78 * S.opacity).toFixed(3)];
  const syncLegend = () => card.style.setProperty("--rt-op", S.live && S.map ? (S.opacity * fade(S.map.getZoom())).toFixed(3) : "1");
  // a calm basemap under the forecast: no names of countries or regions, no shields, airports, railways, footpaths, piers, minor labels or inner boundaries
  const CLUTTER = /shield|airport|aeroway|railway|highway_path|highway-name-path|road_pier|road_area_pier|label_other|place_other|label_state|place_state|waterway_line_label|boundary_3|boundary_state|boundary_disputed/;
  function calm(map) {
    for (const l of map.getStyle().layers) if ((CLUTTER.test(l.id) || (l.type === "symbol" && /country/i.test(l.id))) && map.getLayer(l.id)) map.removeLayer(l.id);
    if (map.getLayer("building")) map.setPaintProperty("building", "fill-opacity", 0.35);
  }
  const underRoads = (map) => { for (const l of map.getStyle().layers) if (l.type === "symbol" || (l.type === "line" && /tunnel|aeroway|road|highway|railway|bridge|boundary/.test(l.id))) return l.id; };
  function addLayers() {  // the forecast of the selected day, above the land and below the roads and the names
    const map = S.map; if (!map || !S.styleReady || !S.meta) return;
    try { calm(map); } catch (e) {}
    if (map.getLayer("pm")) map.removeLayer("pm"); if (map.getSource("pm")) map.removeSource("pm");
    map.addSource("pm", { type: "raster", tiles: [PROTO + "://mean/" + S.day + "/{z}/{x}/{y}"], tileSize: 256, minzoom: 0, maxzoom: 16, bounds: [-180, -60.2, 180, 84.5],
      attribution: "Forecast: Foehn (research) · land mask: Natural Earth" });
    map.addLayer({ id: "pm", type: "raster", source: "pm", paint: { "raster-opacity": opacityExpr(), "raster-resampling": "linear", "raster-fade-duration": 0 } }, underRoads(map));
  }
  function goLive() {
    if (S.live || !S.map) return; S.live = true; card.classList.add("rt-live"); msg(""); syncLegend(); if (S.meta) help();
    const at = $("rt-gl").querySelector(".maplibregl-ctrl-attrib");  // on a narrow map the credits start folded behind their button
    if (at && $("rt-gl").clientWidth < 600) { at.classList.remove("maplibregl-compact-show"); at.removeAttribute("open"); }
  }
  function closeMap(text, retry) {  // back to the world view, with a plain message
    if (S.map) { try { S.map.remove(); } catch (e) {} }
    S.map = null; S.popup = null; S.popupAt = null; S.live = false; S.opening = false; S.styleReady = false; card.classList.remove("rt-live"); msg(text); syncLegend();
    $("rt-open").hidden = !retry || !S.meta; if (S.meta) { help(); poster(); }
  }
  async function openMap() {
    if (S.opening || S.map) return; S.opening = true; $("rt-open").hidden = true; msg("Loading the street map…");
    try {
      if (!hasGl() || typeof DecompressionStream === "undefined" || typeof createImageBitmap !== "function") throw new Error("unsupported");
      if (typeof maplibregl === "undefined") { await addCss(VENDOR + ".css"); await addJs(VENDOR + ".js"); }
      maplibregl.addProtocol(PROTO, async (params) => {
        const [L, z, x, y] = params.url.split("/").slice(-4).map(Number);
        if (!S.meta) return { data: await blank() };
        const nz = Math.min(z, zmaxOf(L)), sh = z - nz, nx = Math.floor(x / Math.pow(2, sh)), ny = Math.floor(y / Math.pow(2, sh));
        if (!hasTile(nz, nx, ny)) return { data: await blank() };  // sea: no request at all
        try { return { data: await toImage(renderTile(await nativeIndices(L, nz, nx, ny), z, x, y, sh, nx, ny), 256, 256) }; }
        catch (e) { if (e && e.name === "AbortError") throw e; return { data: await blank() }; }
      });
      const map = new maplibregl.Map({ container: "rt-gl", style: styleUrl(), ...worldView(), minZoom: -1, maxZoom: 16,
        attributionControl: { compact: true }, dragRotate: false, pitchWithRotate: false, touchPitch: false, cooperativeGestures: true,
        pixelRatio: Math.min(2, window.devicePixelRatio || 1), maxTileCacheZoomLevels: 3 });
      S.map = map; map.touchZoomRotate.disableRotation(); map.keyboard.disableRotation();
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
      map.addControl(new maplibregl.ScaleControl({ maxWidth: 100 }), "bottom-left");
      map.on("style.load", () => { S.styleReady = true; addLayers(); setTimeout(goLive, 6000); });
      map.once("load", goLive);
      map.on("click", (ev) => readout(ev.lngLat.lng, ev.lngLat.lat)); map.on("zoom", syncLegend);
      map.on("styleimagemissing", (ev) => { if (!map.hasImage(ev.id)) map.addImage(ev.id, { width: 1, height: 1, data: new Uint8Array(4) }); });  // an icon absent from the basemap's sprite: left blank, quietly
      map.on("error", (ev) => { if (S.errors.length < 50) S.errors.push(String((ev && ev.error && ev.error.message) || ev)); });  // kept for the tests; nothing is shown
      setTimeout(() => { if (S.map === map && !S.live) closeMap("The street map could not be loaded (the basemap did not answer). The world view stays available.", true); }, 25000);
    } catch (e) {
      const un = e && e.message === "unsupported";
      closeMap(un ? "The street map cannot be opened in this browser (it needs WebGL and a recent browser). The world view stays available." : "The street map could not be loaded. The world view stays available.", !un);
    }
  }
  $("rt-open").addEventListener("click", openMap);
  $("rt-opacity").addEventListener("input", (ev) => { S.opacity = +ev.target.value; if (S.map && S.map.getLayer("pm")) S.map.setPaintProperty("pm", "raster-opacity", opacityExpr()); syncLegend(); });
  // the box on the map: the value of the pixel for the selected day, its class, and where the value comes from
  const SOURCE = [null, "No 1\u00a0km value here: the 0.4° forecast is shown as it is.", "No 1\u00a0km value here: the 0.4° forecast is shown.", "Gap in the 1\u00a0km grid: interpolated from the 1\u00a0km values around."];
  async function readout(lng, lat) {
    if (!S.map || !S.meta) return; lng = ((lng + 180) % 360 + 360) % 360 - 180; S.popupAt = { lng, lat };
    const m = S.meta, L = S.day, box = el("div", null, el("div", { class: "rt-pn" }, coord(lat, lng)), el("div", { class: "rt-ps" }, dfmt(m.layers.mean[L].valid, { weekday: "long", day: "numeric", month: "long" }) + " (UTC day)"));
    try {
      const im = decodeMean(await pixel(L, lng, lat));
      if (!im) box.append(el("div", { class: "rt-ps" }, "No forecast here (sea, or outside 60° S – 84° N)."));
      else { const e = m.encoding, v = el("div", { class: "rt-pv" }, "≈ " + fmt(im.v)); v.append(el("small", null, "µg\u00a0m⁻³"));
        box.append(v, el("div", { class: "rt-pc" }, el("i", { style: "background:" + (im.c < 10 ? e.colors[im.c] : e.over) }), im.c < 10 ? "class " + e.bounds[im.c] + "–" + e.bounds[im.c + 1] : "class above " + e.bounds[10]));
        if (SOURCE[im.src]) box.append(el("div", { class: "rt-ps" }, SOURCE[im.src])); }
    } catch (e) { box.append(el("div", { class: "rt-ps" }, "The value could not be read.")); }
    if (!S.map || S.day !== L) return;
    if (!S.popup) S.popup = new maplibregl.Popup({ closeOnClick: false, maxWidth: "260px", offset: 10 }).on("close", () => { S.popupAt = null; });
    S.popup.setLngLat([lng, lat]).setDOMContent(box); if (!S.popup.isOpen()) S.popup.addTo(S.map);
  }
  // the page theme (device setting, or the switch of the top bar): the basemap follows it
  let dark = isDark();
  const themeChanged = () => { const d = isDark(); if (d === dark) return; dark = d; if (S.map) { S.styleReady = false; S.map.setStyle(styleUrl(), { diff: false }); } };
  new MutationObserver(themeChanged).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", themeChanged);

  // ---- stage 3: the verification at the monitors (history.json): rows = (verified day, lead day, region) with n, rmse, bias, in1090
  const niceStep = (r) => { const p = Math.pow(10, Math.floor(Math.log10(Math.max(r, 1e-6)))), q = r / p; return (q <= 1 ? 1 : q <= 2 ? 2 : q <= 5 ? 5 : 10) * p; };
  async function loadVerif() {
    try {
      const h = await getJson(VERIF), sys = h.systems || {}, issues = Object.keys(sys).filter(isDay).sort(); if (!issues.length || !Array.isArray(h.rows)) throw new Error("history");
      const tag = sys[issues[issues.length - 1]];  // the 1 km model of the latest verified forecast: only its own forecasts are pooled
      const rows = h.rows.filter((r) => r && r.region === "all" && isDay(r.day) && isDay(r.issue) && sys[r.issue] === tag && Number.isInteger(r.lead) && r.lead > 0 && r.lead < 30 &&
        [r.n, r.rmse, r.bias, r.in1090, r.obs_mean].every(Number.isFinite) && r.n > 0);
      const leads = [...new Set(rows.map((r) => r.lead))].sort((a, b) => a - b).map((L) => { const R = rows.filter((r) => r.lead === L), n = R.reduce((s, r) => s + r.n, 0), w = (k) => R.reduce((s, r) => s + r.n * r[k], 0) / n;
        return { lead: L, days: R.length, n, rmse: Math.sqrt(R.reduce((s, r) => s + r.n * r.rmse * r.rmse, 0) / n), bias: w("bias"), in1090: w("in1090") }; });
      if (!leads.length) throw new Error("no row");
      const n = rows.reduce((s, r) => s + r.n, 0), iss = rows.map((r) => r.issue).sort(), dmin = Math.min(...leads.map((d) => d.days)), dmax = Math.max(...leads.map((d) => d.days));
      S.verif = { leads, n, first: iss[0], last: iss[iss.length - 1], share: rows.reduce((s, r) => s + r.n * r.in1090, 0) / n, obs: rows.reduce((s, r) => s + r.n * r.obs_mean, 0) / n,
        prov: new Set(rows.filter((r) => r.final === false).map((r) => r.day)).size, x: [] };
      const V = S.verif, up = typeof h.updated === "string" && isDay(h.updated.slice(0, 10)) ? h.updated.slice(0, 10) : null;
      $("rt-vnote").textContent = "Forecasts issued from " + dfmt(V.first, { day: "numeric", month: "long" }) + " to " + dfmt(V.last, { day: "numeric", month: "long", year: "numeric" }) + ": " +
        (dmin === dmax ? dmax : dmin + " to " + dmax) + " verified days for each lead day, " + int(n) + " pairs of forecast and measurement, whose measurements average " + fmt(V.obs) + " µg\u00a0m⁻³." +
        (V.prov ? " The last " + (V.prov > 1 ? word(V.prov) + " verified days are" : "verified day is") + " provisional: late measurements may still arrive." : "") + (up ? " Updated on " + dfmt(up, { day: "numeric", month: "long", year: "numeric" }) + "." : "");
      const pc = Math.round(100 * V.share);
      $("rt-vspread").textContent = V.share < 0.5 ? "; and the " + members() + "members are too close to one another: " + pc + " % of the measurements fall inside their 10–90 % range"
        : "; " + pc + " % of the measurements fall inside the 10–90 % range of the " + members() + "members";
      const tb = $("rt-vrows"); tb.textContent = "";
      for (const d of leads) tb.append(el("tr", null, el("td", null, "Day " + d.lead), el("td", null, String(d.days)), el("td", null, int(d.n)), el("td", null, d.rmse.toFixed(1)), el("td", null, sgn(d.bias, 1)), el("td", null, Math.round(100 * d.in1090) + " %")));
      drawChart();
    } catch (e) { $("rt-vnote").textContent = "The verification could not be loaded. Please come back later."; }
  }
  function drawChart() {  // RMSE and bias by lead day, one axis (same unit)
    const V = S.verif, svg = $("rt-chart"); if (!V) return;
    const W = Math.round(Math.min(640, Math.max(280, svg.getBoundingClientRect().width || 560))), H = W < 460 ? Math.round(W * 0.62) : 250, m = { l: 34, r: 50, t: 22, b: 26 };
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    const D = V.leads, n = D.length, pad = 24, x = (i) => (n === 1 ? (m.l + W - m.r) / 2 : m.l + pad + (i / (n - 1)) * (W - m.l - m.r - 2 * pad)), f = (v) => v.toFixed(1);
    let hi = Math.max(...D.map((d) => d.rmse)), lo = Math.min(0, ...D.map((d) => d.bias));
    const st = niceStep((hi - lo) / 4); hi = Math.ceil((hi * 1.06) / st) * st; lo = Math.floor((lo * 1.06) / st) * st;
    const y = (v) => m.t + ((hi - v) / (hi - lo)) * (H - m.t - m.b);
    let s = "";
    for (let v = lo; v <= hi + st / 2; v += st) s += '<line class="' + (Math.abs(v) < st / 1e6 ? "rt-zero" : "rt-grid") + '" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + f(y(v)) + '" y2="' + f(y(v)) + '"/><text x="' + (m.l - 6) + '" y="' + f(y(v) + 4) + '" text-anchor="end">' + (Math.abs(v) < st / 1e6 ? 0 : (v < 0 ? "−" : "") + Math.abs(+v.toFixed(6))) + "</text>";
    s += '<line class="rt-cross" id="rt-cross" x1="0" x2="0" y1="' + m.t + '" y2="' + (H - m.b) + '" visibility="hidden"/>';
    const pts = (k) => D.map((d, i) => f(x(i)) + "," + f(y(d[k]))).join(" ");
    if (n > 1) s += '<polyline class="rt-bias" points="' + pts("bias") + '"/><polyline class="rt-rmse" points="' + pts("rmse") + '"/>';
    D.forEach((d, i) => { const cx = x(i), yb = y(d.bias), yr = y(d.rmse); V.x[i] = cx;
      s += '<rect class="rt-dbias" x="' + f(cx - 4) + '" y="' + f(yb - 4) + '" width="8" height="8" transform="rotate(45 ' + f(cx) + " " + f(yb) + ')"/><circle class="rt-drmse" cx="' + f(cx) + '" cy="' + f(yr) + '" r="4.5"/>';
      s += '<text x="' + f(cx) + '" y="' + (H - 7) + '" text-anchor="middle">Day ' + d.lead + "</text>";
      if (i === 0 || i === n - 1) s += '<text class="rt-val" x="' + f(cx) + '" y="' + f(yr - 10) + '" text-anchor="middle">' + f(d.rmse) + '</text><text class="rt-val" x="' + f(cx) + '" y="' + f(yb + 19) + '" text-anchor="middle">' + sgn(d.bias, 1) + "</text>"; });
    s += '<text x="' + f(x(n - 1) + 11) + '" y="' + f(y(D[n - 1].rmse) + 4) + '">RMSE</text><text x="' + f(x(n - 1) + 11) + '" y="' + f(y(D[n - 1].bias) + 4) + '">bias</text>';
    D.forEach((d, i) => { const x0 = i === 0 ? m.l : (x(i - 1) + x(i)) / 2, x1 = i === n - 1 ? W - m.r : (x(i) + x(i + 1)) / 2;
      s += '<rect class="rt-hit" data-i="' + i + '" x="' + f(x0) + '" y="' + m.t + '" width="' + f(x1 - x0) + '" height="' + (H - m.t - m.b) + '" tabindex="0" role="img" aria-label="Day ' + d.lead + ": RMSE " + f(d.rmse) + ", bias " + sgn(d.bias, 1) +
        " micrograms per cubic metre; " + d.days + " verified days, " + d.n + ' monitor-days"/>'; });
    svg.innerHTML = s;
  }
  function tip(i) {  // the readout of one lead day: every series, then the sample
    const V = S.verif, t = $("rt-tip"), cr = $("rt-cross"), svg = $("rt-chart"), d = V && i != null ? V.leads[i] : null;
    if (!d || !cr) { t.hidden = true; if (cr) cr.setAttribute("visibility", "hidden"); return; }
    const row = (key, val, name) => el("span", { class: "rt-tr" }, el("i", { class: key }), el("strong", null, val), " " + name);
    cr.setAttribute("x1", V.x[i]); cr.setAttribute("x2", V.x[i]); cr.setAttribute("visibility", "visible");
    t.textContent = ""; t.append(el("b", null, "Day " + d.lead), row("rt-k", d.rmse.toFixed(1), "RMSE"), row("rt-k rt-kb", sgn(d.bias, 1), "bias"), el("span", { class: "rt-tn" }, d.days + " verified days · " + int(d.n) + " monitor-days"));
    t.hidden = false; const box = svg.getBoundingClientRect(), k = box.width / svg.viewBox.baseVal.width, w = t.offsetWidth, left = V.x[i] * k + 12;
    t.style.left = Math.max(0, left + w > box.width ? V.x[i] * k - 12 - w : left) + "px";
  }
  const chart = $("rt-chart"), over = (ev) => { const r = ev.target.closest(".rt-hit"); if (r) tip(+r.dataset.i); };
  chart.addEventListener("pointerover", over); chart.addEventListener("pointerdown", over); chart.addEventListener("focusin", over);
  chart.addEventListener("pointerleave", () => tip(null)); chart.addEventListener("focusout", () => tip(null));
  let rz = 0; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { tip(null); drawChart(); }, 200); });

  // ---- start: the index now (the section is near the screen), the verification when its block is reached
  S.api = { pixel, decodeMean, hasTile, openMap };
  loadIndex(0);
  if ("IntersectionObserver" in window) { const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); loadVerif(); } }, { rootMargin: "100px 0px" }); io.observe($("rt-verif")); }
  else loadVerif();
})();
