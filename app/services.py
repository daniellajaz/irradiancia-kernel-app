"""Servicios de la aplicación: registro de modelos (SQLite + joblib) y
generación de mapas/predicciones."""
import io
import json
import os
import sqlite3
import threading

import joblib
import numpy as np
from PIL import Image

from fkernel_lab.experiment import DATASETS, load_dataset
from fkernel_lab.geo import (SpectralField, StudyGrid, irradiance_rgb, lonlat_to_merc,
                             merc_to_lonlat, rgb_hex)


# ---------------------------------------------------------------------------
# Registro de modelos
# ---------------------------------------------------------------------------
class ModelRegistry:
    """Los modelos se guardan como archivos ``.joblib`` en ``models/`` y sus
    metadatos (métricas, configuración, zonas) en ``models/registry.db``.
    Al arrancar se registran automáticamente los .joblib exportados por el
    notebook que aún no estén en la base de datos."""

    def __init__(self, models_dir):
        self.dir = models_dir
        os.makedirs(self.dir, exist_ok=True)
        self.db_path = os.path.join(self.dir, "registry.db")
        self._cache = {}
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY, dataset TEXT, label TEXT, spec_id TEXT,
                file TEXT, source TEXT, created_at TEXT, meta TEXT)""")
        self.sync_folder()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def sync_folder(self):
        known = {r[0] for r in self._conn().execute("SELECT file FROM models")}
        for fname in sorted(os.listdir(self.dir)):
            if fname.endswith(".joblib") and fname not in known:
                try:
                    art = joblib.load(os.path.join(self.dir, fname))
                    self._insert(art["meta"], fname)
                except Exception as exc:  # archivo corrupto o ajeno
                    print(f"[registro] se omite {fname}: {exc}")
        # quita entradas cuyo archivo ya no existe
        with self._conn() as c:
            for mid, fname in c.execute("SELECT id, file FROM models").fetchall():
                if not os.path.exists(os.path.join(self.dir, fname)):
                    c.execute("DELETE FROM models WHERE id=?", (mid,))

    def _insert(self, meta, fname):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?,?,?)",
                      (meta["id"], meta["dataset"], meta["label"], meta["spec_id"], fname,
                       meta.get("source", "notebook"), meta.get("created_at", ""),
                       json.dumps(meta)))

    def add(self, artifact):
        import re
        import time
        from fkernel_lab.experiment import save_artifact
        meta = artifact["meta"]
        name = None
        if meta.get("source") == "web":  # nunca sobrescribe los del notebook
            base = re.sub(r"[^a-z0-9]+", "-", f"{meta['dataset']}-{meta['spec_id']}".lower()).strip("-")
            name = f"{base}-web-{int(time.time())}"
        path = save_artifact(artifact, self.dir, name=name)
        self._insert(artifact["meta"], os.path.basename(path))
        return artifact["meta"]

    def list(self):
        rows = self._conn().execute(
            "SELECT meta FROM models ORDER BY dataset, source DESC, created_at").fetchall()
        return [json.loads(r[0]) for r in rows]

    def meta(self, mid):
        row = self._conn().execute("SELECT meta FROM models WHERE id=?", (mid,)).fetchone()
        return json.loads(row[0]) if row else None

    def model(self, mid):
        with self._lock:
            if mid not in self._cache:
                row = self._conn().execute("SELECT file FROM models WHERE id=?", (mid,)).fetchone()
                if not row:
                    raise KeyError(mid)
                self._cache[mid] = joblib.load(os.path.join(self.dir, row[0]))["model"]
            return self._cache[mid]

    def delete(self, mid):
        row = self._conn().execute("SELECT file FROM models WHERE id=?", (mid,)).fetchone()
        if not row:
            return False
        with self._conn() as c:
            c.execute("DELETE FROM models WHERE id=?", (mid,))
        path = os.path.join(self.dir, row[0])
        if os.path.exists(path):
            os.remove(path)
        self._cache.pop(mid, None)
        return True


# ---------------------------------------------------------------------------
# Mapas y predicción puntual
# ---------------------------------------------------------------------------
class MapService:
    def __init__(self, data_dir, registry, step=2000.0, max_dist=10000.0):
        self.registry = registry
        self.frames = {n: load_dataset(n, data_dir) for n in DATASETS}
        self.fields = {n: SpectralField(f) for n, f in self.frames.items()}
        self.grid = StudyGrid(list(self.frames.values()), step=step, max_dist=max_dist)
        self.max_dist = max_dist
        values = np.concatenate([f["value"].to_numpy() for f in self.frames.values()])
        self.vmin, self.vmax = float(values.min()), float(values.max())
        self._grid_features = {n: self.fields[n].features_at(self.grid.x, self.grid.y)[0]
                               for n in DATASETS}
        self._zone_cache = {}
        self._lock = threading.Lock()

    # --- zonas con color ------------------------------------------------
    def zones(self, mid):
        meta = self.registry.meta(mid)
        out = []
        for z in meta["zones"]:
            z = dict(z)
            z["color"] = rgb_hex(irradiance_rgb([z["mid"]], self.vmin, self.vmax)[0])
            out.append(z)
        return out

    def palette(self):
        stops = np.linspace(self.vmin, self.vmax, 7)
        return {"vmin": self.vmin, "vmax": self.vmax,
                "stops": [rgb_hex(c) for c in irradiance_rgb(stops, self.vmin, self.vmax)]}

    # --- predicción sobre la malla ------------------------------------------
    def grid_zones(self, mid):
        with self._lock:
            if mid not in self._zone_cache:
                meta = self.registry.meta(mid)
                model = self.registry.model(mid)
                self._zone_cache[mid] = model.predict(self._grid_features[meta["dataset"]])
            return self._zone_cache[mid]

    def grid_midpoints(self, mid):
        mids = np.array([z["mid"] for z in self.registry.meta(mid)["zones"]])
        return mids[self.grid_zones(mid)]

    def _png(self, rgba):
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
        return buf.getvalue()

    def zone_png(self, mid):
        img = self.grid.to_image(self.grid_midpoints(mid))
        valid = ~np.isnan(img)
        rgba = np.zeros(img.shape + (4,), np.uint8)
        rgba[valid, :3] = irradiance_rgb(img[valid], self.vmin, self.vmax)
        rgba[valid, 3] = 205
        return self._png(rgba)

    def diff_png(self, a, b):
        """Diferencia de punto medio de zona (B - A): rojo = B estima más
        irradiancia, azul = B estima menos, blanco = coinciden."""
        d = self.grid.to_image(self.grid_midpoints(b) - self.grid_midpoints(a))
        valid = ~np.isnan(d)
        lim = max(1.0, np.nanmax(np.abs(d)))
        t = np.clip(d[valid] / lim, -1, 1)
        rgba = np.zeros(d.shape + (4,), np.uint8)
        pos, neg = np.clip(t, 0, 1), np.clip(-t, 0, 1)
        rgba[valid, 0] = (255 * (1 - neg)).astype(np.uint8)
        rgba[valid, 1] = (255 * (1 - np.abs(t))).astype(np.uint8)
        rgba[valid, 2] = (255 * (1 - pos)).astype(np.uint8)
        rgba[valid, 3] = 215
        return self._png(rgba), lim

    def agreement(self, a, b):
        za, zb = self.registry.meta(a)["zones"], self.registry.meta(b)["zones"]
        pa, pb = self.grid_zones(a), self.grid_zones(b)
        lo_a = np.array([z["low"] for z in za])[pa]
        hi_a = np.array([z["high"] for z in za])[pa]
        lo_b = np.array([z["low"] for z in zb])[pb]
        hi_b = np.array([z["high"] for z in zb])[pb]
        overlap = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b) > 0
        diff = self.grid_midpoints(b) - self.grid_midpoints(a)
        same_scheme = [round(z["low"], 6) for z in za] == [round(z["low"], 6) for z in zb] \
            and len(za) == len(zb)
        res = {
            "cells": int(len(pa)),
            "range_overlap_pct": float(100 * overlap.mean()),
            "mean_abs_mid_diff": float(np.abs(diff).mean()),
            "same_scheme": bool(same_scheme),
            "area_by_zone": {
                "a": [float(100 * np.mean(pa == i)) for i in range(len(za))],
                "b": [float(100 * np.mean(pb == i)) for i in range(len(zb))],
            },
        }
        if same_scheme:
            res["same_zone_pct"] = float(100 * np.mean(pa == pb))
        return res

    # --- predicción en un punto -----------------------------------------------
    def predict_point(self, lat, lon, model_ids):
        x, y = lonlat_to_merc(lon, lat)
        out = {"lat": lat, "lon": lon, "x": float(x), "y": float(y), "models": []}
        for mid in model_ids:
            meta = self.registry.meta(mid)
            if meta is None:
                continue
            ds = meta["dataset"]
            feats, dist = self.fields[ds].features_at(x, y)
            model = self.registry.model(mid)
            zone = int(model.predict(feats)[0])
            scores = model.scores(feats)[0]
            zones = self.zones(mid)
            # observación real más cercana (para contrastar)
            frame = self.frames[ds]
            _, j = self.fields[ds].tree.query([float(x), float(y)])
            observed = float(frame["value"].iloc[j])
            obs_zone = int(model.labels([observed])[0])
            olon, olat = merc_to_lonlat(frame["latitude"].iloc[j], frame["longitude"].iloc[j])
            out["models"].append({
                "id": mid, "label": meta["label"], "dataset": ds,
                "zone": zones[zone], "scores": [float(s) for s in scores],
                "distance_km": float(dist[0] / 1000),
                "inside": bool(dist[0] <= self.max_dist),
                "features": {k: float(v) for k, v in zip(model.features, feats[0])},
                "nearest": {"value": observed, "zone": zones[obs_zone]["name"],
                            "lat": float(olat), "lon": float(olon)},
            })
        return out

    def points(self, dataset):
        f = self.frames[dataset]
        lon, lat = merc_to_lonlat(f["latitude"].to_numpy(), f["longitude"].to_numpy())
        colors = [rgb_hex(c) for c in irradiance_rgb(f["value"].to_numpy(), self.vmin, self.vmax)]
        return [{"lat": float(a), "lon": float(b), "value": float(v), "color": c}
                for a, b, v, c in zip(lat, lon, f["value"], colors)]
