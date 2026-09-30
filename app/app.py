"""Aplicación web: zonificación de irradiancia con pipelines kernel.

Ejecutar desde la raíz del repositorio:
    python app/app.py                      (desarrollo, http://127.0.0.1:5000)
    gunicorn -w 1 -b 0.0.0.0:8000 app.app:app   (producción)
"""
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from flask import Flask, abort, jsonify, render_template, request, send_file  # noqa: E402
import io  # noqa: E402

from fkernel_lab.experiment import (DATASETS, MODEL_LABELS, MODELS, REDUCERS, SCALERS,  # noqa: E402
                                    Spec, build_artifact, evaluate_spec, holdout_split)
from fkernel_lab.kernels import KERNEL_CATALOG  # noqa: E402
from fkernel_lab.preprocessing import DISCRETIZER_LABELS  # noqa: E402

from app.services import MapService, ModelRegistry  # noqa: E402

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(ROOT, "models"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))

app = Flask(__name__)
registry = ModelRegistry(MODELS_DIR)
maps = MapService(DATA_DIR, registry)

CHOICES = {
    "dataset": {d: d.upper() for d in DATASETS},
    "scaler": {"standard": "StandardScaler", "minmax": "MinMaxScaler", "normalizer": "Normalizer"},
    "discretizer": DISCRETIZER_LABELS,
    "reducer": {"pca": "PCA", "lda": "LDA"},
    "model": MODEL_LABELS,
    "kernel": KERNEL_CATALOG,
}
assert set(CHOICES["scaler"]) == set(SCALERS) and set(CHOICES["model"]) == set(MODELS)
assert set(CHOICES["reducer"]) == set(REDUCERS)


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------
@app.route("/")
def compare_page():
    return render_template("compare.html", page="compare")


@app.route("/modelos")
def models_page():
    return render_template("models.html", page="models", choices=CHOICES)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/config")
def api_config():
    return jsonify({"bounds": maps.grid.bounds_latlon(), "palette": maps.palette(),
                    "choices": CHOICES})


@app.get("/api/models")
def api_models():
    return jsonify(registry.list())


@app.get("/api/models/<mid>")
def api_model(mid):
    meta = registry.meta(mid)
    if meta is None:
        abort(404)
    meta = dict(meta)
    meta["zones"] = maps.zones(mid)
    return jsonify(meta)


@app.delete("/api/models/<mid>")
def api_delete(mid):
    if not registry.delete(mid):
        abort(404)
    maps._zone_cache.pop(mid, None)
    return jsonify({"deleted": mid})


@app.get("/api/models/<mid>/map.png")
def api_map(mid):
    if registry.meta(mid) is None:
        abort(404)
    resp = send_file(io.BytesIO(maps.zone_png(mid)), mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


@app.get("/api/diff.png")
def api_diff():
    a, b = request.args.get("a"), request.args.get("b")
    if registry.meta(a) is None or registry.meta(b) is None:
        abort(404)
    png, _ = maps.diff_png(a, b)
    return send_file(io.BytesIO(png), mimetype="image/png")


@app.get("/api/compare")
def api_compare():
    a, b = request.args.get("a"), request.args.get("b")
    if registry.meta(a) is None or registry.meta(b) is None:
        abort(404)
    _, lim = maps.diff_png(a, b)
    res = maps.agreement(a, b)
    res["diff_limit"] = lim
    return jsonify(res)


@app.get("/api/predict")
def api_predict():
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
    except (KeyError, ValueError):
        abort(400, "lat y lon son obligatorios")
    ids = [m for m in request.args.get("models", "").split(",") if m]
    return jsonify(maps.predict_point(lat, lon, ids))


@app.get("/api/points/<dataset>")
def api_points(dataset):
    if dataset not in DATASETS:
        abort(404)
    return jsonify(maps.points(dataset))


@app.post("/api/models/train")
def api_train():
    """Entrena, evalúa (misma partición y protocolo que el notebook) y
    ALMACENA un pipeline nuevo en el registro."""
    body = request.get_json(force=True)
    try:
        spec = Spec(*(body[k] for k in ("scaler", "discretizer", "reducer", "model", "kernel")))
        dataset = body["dataset"]
        for field, value in zip(("dataset",) + Spec._fields, (dataset,) + tuple(spec)):
            if value not in CHOICES[field]:
                raise ValueError(f"{field} inválido: {value}")
        params = {}
        for key in ("C", "alpha", "bandwidth"):
            if body.get(key) not in (None, ""):
                params[key] = float(body[key])
        if "C" in params and spec.model != "ksvc":
            params.pop("C")
        if "alpha" in params and spec.model == "ksvc":
            params.pop("alpha")
        if spec.kernel == "linear":
            params.pop("bandwidth", None)
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    df = maps.frames[dataset]
    tr, te = holdout_split(df)
    try:
        result = evaluate_spec(spec, df, tr, te, model_params=params, keep_predictions=True)
    except Exception as exc:
        return jsonify({"error": f"No se pudo entrenar: {exc}"}), 422
    artifact = build_artifact(result, source="web")
    meta = registry.add(artifact)
    return jsonify(meta)


@app.errorhandler(400)
def bad_request(err):
    return jsonify({"error": str(err.description)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
