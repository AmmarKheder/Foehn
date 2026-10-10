/* Otso 1.1 service worker: the app shell is cached (so the page opens offline and installs as a PWA); the live forecast
   (raw.githubusercontent.com) is fetched from the network first and the last successful copy is kept as a fallback, so the
   last forecast is still shown without a connection. */
const VERSION = "otso-1.1-20261010g";
const SHELL = ["./", "./index.html", "../assets/style.css?v=20261010g", "./lines.json", "./gazetteer.json", "./manifest.webmanifest",
               "./icon-192.png", "./icon-512.png", "../assets/logo_mark.svg", "../assets/favicon.png", "./qr.svg"];
self.addEventListener("install", (ev) => { ev.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener("activate", (ev) => { ev.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== VERSION && k !== VERSION + "-data").map((k) => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener("fetch", (ev) => {
  const u = new URL(ev.request.url);
  if (ev.request.method !== "GET") return;
  if (u.hostname === "raw.githubusercontent.com") {                               // live data: network first, cache fallback
    ev.respondWith(fetch(ev.request).then((r) => { if (r.ok) { const cp = r.clone(); caches.open(VERSION + "-data").then((c) => c.put(ev.request, cp)); } return r; })
      .catch(() => caches.match(ev.request).then((r) => r || new Response("", { status: 503, statusText: "offline" }))));
    return;
  }
  if (u.origin === location.origin) {                                              // shell: cache first, refreshed in the background
    ev.respondWith(caches.match(ev.request, { ignoreSearch: false }).then((hit) => {
      const net = fetch(ev.request).then((r) => { if (r.ok) { const cp = r.clone(); caches.open(VERSION).then((c) => c.put(ev.request, cp)); } return r; }).catch(() => hit);
      return hit || net;
    }));
  }
});
