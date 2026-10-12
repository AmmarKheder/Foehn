/* Otso moved to https://ammarkheder.com/Otso/ : this service worker replaces the old one, clears its caches, unregisters
   itself and sends every open page or installed app to the new address. */
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (ev) => { ev.waitUntil((async () => {
  for (const k of await caches.keys()) await caches.delete(k);
  await self.registration.unregister();
  for (const c of await self.clients.matchAll({ type: "window" })) { try { await c.navigate("https://ammarkheder.com/Otso/?source=pwa"); } catch (e) {} }
})()); });
self.addEventListener("fetch", (ev) => { ev.respondWith(Response.redirect("https://ammarkheder.com/Otso/", 302)); });
