const METRIC_NAMES = { accuracy: "Accuracy", f1: "F1 macro", auc: "AUC (OvR)", mcc: "MCC" };
const COLORS = { a: "#3d3a8c", b: "#a23b72" };
const state = { config: null, models: [], meta: {}, overlays: {}, points: {}, marker: {}, chart: null };

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
/* ---------------- mapas ---------------- */
function makeMap(id) {
  const map = L.map(id, { zoomControl: true, attributionControl: true, preferCanvas: true });
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
  }).addTo(map);
  map.fitBounds(state.config.bounds);
  return map;
}

let syncing = false;
function sync(src, dst) {
  src.on("move", () => {
    if (syncing) return;
    syncing = true;
    dst.setView(src.getCenter(), src.getZoom(), { animate: false });
    syncing = false;
  });
}

async function pointsLayer(dataset) {
  if (!state.points[dataset]) {
    const pts = await getJSON(`/api/points/${dataset}`);
    state.points[dataset] = pts;
  }
  return L.layerGroup(state.points[dataset].map((p) =>
    L.circleMarker([p.lat, p.lon], { radius: 3, weight: 1, color: "#1e2a33", fillColor: p.color, fillOpacity: 1 })
      .bindTooltip(`Observado: ${p.value.toFixed(1)}`)));
}

async function showModel(side) {
  const id = $(side === "a" ? "selA" : "selB").value;
  const map = side === "a" ? state.mapA : state.mapB;
  const meta = await getJSON(`/api/models/${encodeURIComponent(id)}`);
  state.meta[side] = meta;
  const old = state.overlays[side];
  if (old) { map.removeLayer(old.img); map.removeLayer(old.pts); old.ctrl.remove(); }
  const img = L.imageOverlay(`/api/models/${encodeURIComponent(id)}/map.png`, state.config.bounds, { opacity: 0.85 }).addTo(map);
  const pts = await pointsLayer(meta.dataset);
  pts.addTo(map);
  const ctrl = L.control.layers(null, { "Zonas predichas": img, [`Observaciones ${meta.dataset.toUpperCase()}`]: pts }, { collapsed: true }).addTo(map);
  state.overlays[side] = { img, pts, ctrl };
  $(side === "a" ? "titleA" : "titleB").textContent = meta.label;
}

/* ---------------- regla solar ---------------- */
function renderRuler() {
  const { vmin, vmax, stops } = state.config.palette;
  const pct = (v) => ((v - vmin) / (vmax - vmin)) * 100;
  let html = `<div class="ruler-bar" style="background:linear-gradient(90deg, ${stops.join(",")})"></div>`;
  for (const side of ["a", "b"]) {
    const meta = state.meta[side];
    if (!meta) continue;
    for (const z of meta.zones) {
      const left = pct(Math.max(z.low, vmin));
      const width = Math.max(0.5, pct(Math.min(z.high, vmax)) - left);
      html += `<div class="ruler-zone ${side}" style="left:${left}%;width:${width}%" title="${esc(z.name)}: ${z.low.toFixed(1)}–${z.high.toFixed(1)}">${esc(z.name)}</div>`;
    }
  }
  $("ruler").innerHTML = html;
  const ticks = Array.from({ length: 7 }, (_, i) => (vmin + (i * (vmax - vmin)) / 6).toFixed(0));
  $("rulerTicks").innerHTML = ticks.map((t) => `<span>${t}</span>`).join("");
}

/* ---------------- métricas ---------------- */
function renderMetrics() {
  const A = state.meta.a, B = state.meta.b;
  if (!A || !B) return;
  let rows = `<tr><th>Métrica</th><th class="num"><span class="dot a"></span>A · CV</th><th class="num">A · holdout</th>
              <th class="num"><span class="dot b"></span>B · CV</th><th class="num">B · holdout</th></tr>`;
  for (const [k, name] of Object.entries(METRIC_NAMES)) {
    const ta = A.test[k], tb = B.test[k];
    rows += `<tr><td>${name}</td>
      <td class="num">${fmt(A.cv[k])} <small>±${fmt(A.cv_std[k], 2)}</small></td>
      <td class="num ${ta > tb ? "win" : ""}">${fmt(ta)}</td>
      <td class="num">${fmt(B.cv[k])} <small>±${fmt(B.cv_std[k], 2)}</small></td>
      <td class="num ${tb > ta ? "win" : ""}">${fmt(tb)}</td></tr>`;
  }
  rows += `<tr><td>Tiempo de ajuste (s)</td><td class="num" colspan="2">${fmt(A.fit_time, 2)}</td><td class="num" colspan="2">${fmt(B.fit_time, 2)}</td></tr>`;
  $("metricsTable").innerHTML = rows;

  const labels = Object.values(METRIC_NAMES);
  const keys = Object.keys(METRIC_NAMES);
  const data = {
    labels,
    datasets: [
      { label: "A · CV", data: keys.map((k) => A.cv[k]), backgroundColor: COLORS.a + "66" },
      { label: "A · holdout", data: keys.map((k) => A.test[k]), backgroundColor: COLORS.a },
      { label: "B · CV", data: keys.map((k) => B.cv[k]), backgroundColor: COLORS.b + "66" },
      { label: "B · holdout", data: keys.map((k) => B.test[k]), backgroundColor: COLORS.b },
    ],
  };
  if (state.chart) state.chart.destroy();
  state.chart = new Chart($("metricsChart"), {
    type: "bar", data,
    options: { scales: { y: { min: 0, max: 1 } }, plugins: { legend: { position: "bottom" } }, animation: false },
  });
}

function confusionHTML(meta) {
  const cm = meta.confusion;
  const max = Math.max(...cm.flat(), 1);
  const names = meta.zones.map((z) => z.name);
  let h = `<table class="cm"><tr><th>Real \\ Pred.</th>${names.map((n) => `<th>${esc(n)}</th>`).join("")}</tr>`;
  cm.forEach((row, i) => {
    h += `<tr><th>${esc(names[i])}</th>`;
    row.forEach((v, j) => {
      const a = v / max;
      const base = i === j ? "47,125,79" : "162,59,114";
      h += `<td style="background:rgba(${base},${(0.08 + 0.8 * a).toFixed(2)});color:${a > 0.55 ? "#fff" : "inherit"}">${v}</td>`;
    });
    h += "</tr>";
  });
  const total = cm.flat().reduce((s, v) => s + v, 0);
  return h + `</table><p class="caption">Holdout de ${total} observaciones. Diagonal en verde: aciertos.</p>`;
}

async function renderComparison() {
  const a = $("selA").value, b = $("selB").value;
  $("cmA").innerHTML = confusionHTML(state.meta.a);
  $("cmB").innerHTML = confusionHTML(state.meta.b);
  if (state.diffLayer) state.diffMap.removeLayer(state.diffLayer);
  state.diffLayer = L.imageOverlay(`/api/diff.png?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`,
    state.config.bounds, { opacity: 0.9 }).addTo(state.diffMap);
  const ag = await getJSON(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
  let h = `<div class="stat"><div class="bignum">${fmt(ag.range_overlap_pct, 1)} %</div>
           <span>del área en la que los rangos de irradiancia de ambas zonas se solapan</span></div>`;
  if (ag.same_scheme) {
    h += `<div class="stat"><div class="bignum">${fmt(ag.same_zone_pct, 1)} %</div><span>del área con la misma zona (mismos cortes)</span></div>`;
  }
  h += `<div class="stat"><div class="bignum">${fmt(ag.mean_abs_mid_diff, 1)}</div>
        <span>diferencia media entre los puntos medios de zona (máx. ${fmt(ag.diff_limit, 1)})</span></div>`;
  $("agreement").innerHTML = h;
}

/* ---------------- clic en el mapa ---------------- */
async function onMapClick(e) {
  const { lat, lng } = e.latlng;
  const ids = [$("selA").value, $("selB").value];
  for (const [side, map] of [["a", state.mapA], ["b", state.mapB]]) {
    if (state.marker[side]) map.removeLayer(state.marker[side]);
    state.marker[side] = L.marker([lat, lng]).addTo(map);
  }
  $("pointInfo").innerHTML = `<p class="empty">Clasificando…</p>`;
  const res = await getJSON(`/api/predict?lat=${lat}&lon=${lng}&models=${ids.map(encodeURIComponent).join(",")}`);
  let h = `<p><small>Lat ${lat.toFixed(4)}, Lon ${lng.toFixed(4)} · EPSG:3857 X ${res.x.toFixed(0)}, Y ${res.y.toFixed(0)}</small></p>`;
  res.models.forEach((m, i) => {
    const side = i === 0 ? "a" : "b";
    const z = m.zone;
    h += `<div class="pointcard ${side}">
      <div class="zone"><span class="swatch" style="background:${z.color}"></span>${esc(z.name)}</div>
      <div>Rango ${z.low.toFixed(1)} – ${z.high.toFixed(1)} <small>(${esc(m.label)})</small></div>
      <small>Observación más cercana a ${m.distance_km.toFixed(1)} km: ${m.nearest.value.toFixed(1)} (zona real ${esc(m.nearest.zone)})</small>
      ${m.inside ? "" : `<div class="warnline">Fuera del área con observaciones: la predicción es una extrapolación.</div>`}
    </div>`;
    const popup = `<strong>${side.toUpperCase()}: ${esc(z.name)}</strong><br>${z.low.toFixed(1)} – ${z.high.toFixed(1)}`;
    state.marker[side].bindPopup(popup, { autoPan: false }).openPopup();
  });
  if (res.models.length) {
    const f = res.models[0].features;
    h += `<details><summary>Bandas interpoladas (IDW) usadas como entrada</summary><table>` +
      res.models.map((m) => `<tr><th colspan="2">${m.dataset.toUpperCase()}</th></tr>` +
        Object.entries(m.features).filter(([k]) => k.startsWith("band"))
          .map(([k, v]) => `<tr><td>${k}</td><td class="num">${fmt(v, 4)}</td></tr>`).join("")).join("") +
      `</table></details>`;
  }
  $("pointInfo").innerHTML = h;
}

/* ---------------- arranque ---------------- */
const rank = (m) => (m.cv_score ?? (m.cv.mcc + m.cv.f1) / 2);  // mismo criterio que el notebook

function fillSelects() {
  const byDataset = {};
  state.models.forEach((m) => (byDataset[m.dataset] ??= []).push(m));
  const html = Object.entries(byDataset).map(([ds, list]) =>
    `<optgroup label="${ds.toUpperCase()}">` +
    list.sort((x, y) => rank(y) - rank(x))
      .map((m) => `<option value="${esc(m.id)}">${esc(m.label)} · ${m.zones.length} zonas · CV ${fmt(rank(m), 2)}${m.source === "web" ? " (web)" : ""}</option>`).join("") +
    `</optgroup>`).join("");
  $("selA").innerHTML = html;
  $("selB").innerHTML = html;
  const params = new URLSearchParams(location.search);
  const best = (ds) => state.models.filter((m) => m.dataset === ds).sort((x, y) => rank(y) - rank(x))[0];
  const a = params.get("a") || best("landsat")?.id || state.models[0].id;
  const b = params.get("b") || best("modis")?.id || state.models[Math.min(1, state.models.length - 1)].id;
  $("selA").value = a;
  $("selB").value = b;
}

async function refresh(sides) {
  for (const s of sides) await showModel(s);
  const url = new URL(location);
  url.searchParams.set("a", $("selA").value);
  url.searchParams.set("b", $("selB").value);
  history.replaceState(null, "", url);
  renderRuler();
  renderMetrics();
  await renderComparison();
}

(async function init() {
  state.config = await getJSON("/api/config");
  state.models = await getJSON("/api/models");
  if (!state.models.length) {
    document.querySelector("main").innerHTML =
      `<section class="panel"><h2>Aún no hay modelos guardados</h2><p>Ejecuta el notebook para exportar los mejores pipelines
       o <a href="/modelos">entrena uno desde la página de modelos</a>.</p></section>`;
    return;
  }
  state.mapA = makeMap("mapA");
  state.mapB = makeMap("mapB");
  state.diffMap = makeMap("diffmap");
  sync(state.mapA, state.mapB);
  sync(state.mapB, state.mapA);
  state.mapA.on("click", onMapClick);
  state.mapB.on("click", onMapClick);
  fillSelects();
  $("selA").addEventListener("change", () => refresh(["a"]));
  $("selB").addEventListener("change", () => refresh(["b"]));
  await refresh(["a", "b"]);
})();
