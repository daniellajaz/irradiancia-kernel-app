const $ = (id) => document.getElementById(id);
const fmt = (v, d = 3) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function loadModels() {
  const models = await (await fetch("/api/models")).json();
  if (!models.length) {
    $("modelsTable").innerHTML = `<tr><td class="empty">No hay modelos. Entrena uno arriba o ejecuta el notebook.</td></tr>`;
    return;
  }
  let h = `<tr><th>Satélite</th><th>Pipeline</th><th class="num">Accuracy</th><th class="num">F1</th>
           <th class="num">AUC</th><th class="num">MCC</th><th>Origen</th><th></th></tr>`;
  for (const m of models) {
    const s = m.spec;
    h += `<tr>
      <td>${m.dataset.toUpperCase()}</td>
      <td><strong>${esc(s.model.toUpperCase())} · ${esc(s.kernel)}</strong><br>
          <small>${esc(s.scaler)} / ${esc(s.discretizer)} / ${esc(s.reducer)}</small></td>
      <td class="num">${fmt(m.test.accuracy)}</td><td class="num">${fmt(m.test.f1)}</td>
      <td class="num">${fmt(m.test.auc)}</td><td class="num">${fmt(m.test.mcc)}</td>
      <td><span class="badge ${m.source === "web" ? "web" : ""}">${m.source === "web" ? "Web" : "Notebook"}</span><br>
          <small>${esc(m.created_at || "")}</small></td>
      <td><a href="/?a=${encodeURIComponent(m.id)}">Ver en el mapa</a>
          <button class="danger" data-id="${esc(m.id)}">Eliminar</button></td></tr>`;
  }
  $("modelsTable").innerHTML = h;
  document.querySelectorAll("button[data-id]").forEach((btn) =>
    btn.addEventListener("click", async () => {
      if (!confirm("¿Eliminar este modelo del registro y del disco?")) return;
      await fetch(`/api/models/${encodeURIComponent(btn.dataset.id)}`, { method: "DELETE" });
      loadModels();
    }));
}

$("trainForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = Object.fromEntries(new FormData(e.target).entries());
  const status = $("trainStatus");
  $("trainBtn").disabled = true;
  status.className = "status";
  status.textContent = "Entrenando y evaluando (validación cruzada 5×2 + holdout)…";
  try {
    const r = await fetch("/api/models/train", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const res = await r.json();
    if (!r.ok) throw new Error(res.error || r.statusText);
    status.className = "status ok";
    status.textContent = `Guardado: ${res.label}. MCC holdout ${fmt(res.test.mcc)}, accuracy ${fmt(res.test.accuracy)}.`;
    loadModels();
  } catch (err) {
    status.className = "status error";
    status.textContent = `No se guardó: ${err.message}`;
  } finally {
    $("trainBtn").disabled = false;
  }
});

loadModels();
