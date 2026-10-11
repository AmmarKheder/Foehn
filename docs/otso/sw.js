/* Otso 1.1 service worker.
   - the page itself (navigations) is fetched from the network first, so an online visitor always gets the latest version;
     the cached copy is the offline fallback
   - static assets (style, gazetteer, coastlines, icons) are served from the cache and refreshed in the background
   - the live forecast (raw.githubusercontent.com) is network first; the last successful copy is kept for offline use */
const VERSION = "otso-1.2-202610110941";
const SHELL = ["./", "./index.html", "../assets/style.css?v=20261011a", "./lines.json", "./gazetteer.json", "./manifest.webmanifest",
               "./icon-192.png", "./icon-512.png", "../assets/logo_mark.svg", "../assets/favicon.png", "./qr.svg", "./labels.json", "../assets/amc_logo_full.png", "../assets/lumi_logo.png", "../assets/csc_logo.svg"];
self.addEventListener("install", (ev) => { ev.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener("activate", (ev) => { ev.waitUntil(caches.keys().then(async (ks) => {
  const old = ks.filter((k) => k !== VERSION && k !== VERSION + "-data");
  await Promise.all(old.map((k) => caches.delete(k)));
  await self.clients.claim();
  if (old.length) {                                                                       // a new version replaced an old one: reload the open pages once
    const cs = await self.clients.matchAll({ type: "window" });
    for (const c of cs) { try { c.navigate(c.url); } catch (e) {} }
  }
})); });
self.addEventListener("fetch", (ev) => {
  const u = new URL(ev.request.url);
  if (ev.request.method !== "GET") return;
  const networkFirst = (cacheName) => fetch(ev.request).then((r) => { if (r.ok) { const cp = r.clone(); caches.open(cacheName).then((c) => c.put(ev.request, cp)); } return r; })
    .catch(() => caches.match(ev.request, { ignoreSearch: ev.request.mode === "navigate" }).then((r) => r || (ev.request.mode === "navigate" ? caches.match("./index.html") : null) || new Response("", { status: 503, statusText: "offline" })));
  if (u.hostname === "raw.githubusercontent.com") { ev.respondWith(networkFirst(VERSION + "-data")); return; }
  if (u.origin !== location.origin) return;                                                  // geocoders, counters: straight to the network
  if (ev.request.mode === "navigate" || u.pathname.endsWith("/") || u.pathname.endsWith("index.html")) { ev.respondWith(networkFirst(VERSION)); return; }
  ev.respondWith(caches.match(ev.request).then((hit) => {                                   // static assets: cache first, refreshed in the background
    const net = fetch(ev.request).then((r) => { if (r.ok) { const cp = r.clone(); caches.open(VERSION).then((c) => c.put(ev.request, cp)); } return r; }).catch(() => hit);
    return hit || net;
  }));
});
