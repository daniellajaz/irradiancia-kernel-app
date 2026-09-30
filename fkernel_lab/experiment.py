"""Motor experimental: espacio de pipelines, evaluación justa y modelo
desplegable (discretizador + pipeline)."""
import os
import time
import urllib.request
from itertools import product
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from .estimators import KANNCVec, KSVCVec
from .kridge import KRidgeClassifier
from .kernels import KERNELS
from .preprocessing import DISCRETIZERS, DimReducer, IrradianceDiscretizer

SEED = 2021
N_ZONES = 4
FEATURES = ["latitude", "longitude", "band1", "band2", "band3", "band4", "band5", "band6", "band7"]
TARGET = "value"
DATA_URL = "https://raw.githubusercontent.com/magohector/fkernel/master/Experimentos/{}_model.csv"
DATASETS = ("landsat", "modis")

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "normalizer": Normalizer}
REDUCERS = ("pca", "lda")
MODELS = ("ksvc", "kann", "kridge")
MODEL_LABELS = {"ksvc": "KSVC", "kann": "KANN", "kridge": "KRidgeClassifier"}

# Hiperparámetros por defecto de la ETAPA 1 (iguales para todos los kernels;
# la escala de cada kernel la fija la heurística de la mediana)
DEFAULT_PARAMS = {
    "ksvc": dict(C=10.0),
    "kann": dict(hidden_layer_sizes=(50,), activation="relu", alpha=1e-3,
                 learning_rate_init=0.01, max_iter=300, early_stopping=False,
                 n_components=80),
    "kridge": dict(alpha=0.1),
}


class Spec(NamedTuple):
    """Una combinación Escalador × Discretizador × Reductor × Modelo × Kernel."""
    scaler: str
    discretizer: str
    reducer: str
    model: str
    kernel: str

    @property
    def id(self):
        return "|".join(self)

    @classmethod
    def from_id(cls, text):
        return cls(*text.split("|"))


def all_specs(kernels=KERNELS):
    return [Spec(*c) for c in product(SCALERS, DISCRETIZERS, REDUCERS, MODELS, kernels)]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def load_dataset(name, data_dir="data"):
    path = os.path.join(data_dir, f"{name}_model.csv")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL.format(name), path)
    df = pd.read_csv(path)
    df.attrs["name"] = name
    return df


def holdout_split(df, test_size=0.2, seed=SEED):
    """80/20 estratificado por quintiles de irradiancia (independiente del
    discretizador, para que TODOS los pipelines usen la misma partición)."""
    strata = pd.qcut(df[TARGET], 5, labels=False, duplicates="drop")
    return train_test_split(np.arange(len(df)), test_size=test_size,
                            random_state=seed, stratify=strata)


# ---------------------------------------------------------------------------
# Construcción de pipelines
# ---------------------------------------------------------------------------
def make_estimator(model, kernel, **params):
    p = {**DEFAULT_PARAMS[model], **params}
    if model == "ksvc":
        return KSVCVec(kernel=kernel, random_state=SEED, **p)
    if model == "kann":
        return KANNCVec(kernel=kernel, random_state=SEED, **p)
    if model == "kridge":
        return KRidgeClassifier(kernel=kernel, **p)
    raise ValueError(f"Modelo desconocido: {model!r}")


def make_pipeline(spec, **model_params):
    """Pipeline de sklearn sobre X. El discretizador actúa sobre y, por eso
    vive en ``IrradianceZoneModel`` y no dentro del Pipeline."""
    return Pipeline([
        ("scaler", SCALERS[spec.scaler]()),
        ("reducer", DimReducer(spec.reducer, random_state=SEED)),
        ("model", make_estimator(spec.model, spec.kernel, **model_params)),
    ])


def decision_scores(pipe, X):
    """Puntuaciones por clase (decision_function o predict_proba)."""
    if hasattr(pipe, "decision_function"):
        s = pipe.decision_function(X)
    else:
        s = pipe.predict_proba(X)
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = np.column_stack([-s, s])
    return s


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def ovr_auc(y_true, scores, classes):
    """AUC macro uno-contra-resto. Se calcula clase a clase con las
    puntuaciones crudas, así sirve para SVM/Ridge (no probabilísticos)."""
    vals = []
    for j, c in enumerate(classes):
        b = (np.asarray(y_true) == c).astype(int)
        if 0 < b.sum() < len(b):
            vals.append(roc_auc_score(b, scores[:, j]))
    return float(np.mean(vals)) if vals else np.nan


def compute_metrics(y_true, y_pred, scores, classes, n_classes):
    labels = np.arange(n_classes)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "auc": ovr_auc(y_true, scores, classes),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


METRICS = ("accuracy", "f1", "auc", "mcc")


def selection_score(mcc, f1):
    """Criterio de selección: media de MCC y F1 macro.

    MCC resume toda la matriz de confusión, pero con zonas muy desbalanceadas
    (p. ej. DBSCAN crea zonas intermedias diminutas) puede ser alto aunque el
    modelo ignore las zonas pequeñas. F1 macro da el mismo peso a cada zona y
    penaliza ese comportamiento. Promediar ambos evita premiar esa trampa."""
    return (np.asarray(mcc) + np.asarray(f1)) / 2


def add_selection_score(df):
    df = df.copy()
    df["cv_score"] = selection_score(df["cv_mcc"], df["cv_f1"])
    df["test_score"] = selection_score(df["test_mcc"], df["test_f1"])
    return df


# ---------------------------------------------------------------------------
# Modelo desplegable
# ---------------------------------------------------------------------------
class IrradianceZoneModel:
    """Discretizador de irradiancia + pipeline (escalador, reductor, modelo)."""

    def __init__(self, spec, dataset, model_params=None, n_zones=N_ZONES):
        self.spec = spec if isinstance(spec, Spec) else Spec(*spec)
        self.dataset = dataset
        self.model_params = dict(model_params or {})
        self.n_zones = n_zones
        self.features = list(FEATURES)

    def fit(self, X, y_continuous):
        self.discretizer_ = IrradianceDiscretizer(self.spec.discretizer, self.n_zones).fit(y_continuous)
        labels = self.discretizer_.transform(y_continuous)
        self.pipeline_ = make_pipeline(self.spec, **self.model_params).fit(X, labels)
        self.classes_ = self.pipeline_.classes_
        return self

    def labels(self, y_continuous):
        return self.discretizer_.transform(y_continuous)

    def predict(self, X):
        return self.pipeline_.predict(np.asarray(X, dtype=float))

    def scores(self, X):
        return decision_scores(self.pipeline_, np.asarray(X, dtype=float))

    def zones(self):
        return self.discretizer_.zones()


# ---------------------------------------------------------------------------
# Evaluación justa
# ---------------------------------------------------------------------------
def make_cv(n_splits=5, n_repeats=2):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)


def evaluate_spec(spec, df, train_idx, test_idx, model_params=None, cv=None,
                  keep_predictions=False):
    """Evalúa una configuración con:
    1) discretizador ajustado SOLO con la irradiancia de entrenamiento,
    2) validación cruzada repetida y estratificada (mismas particiones para
       todos los pipelines con el mismo discretizador),
    3) ajuste final en el 80 % y medición en el holdout 20 % intacto."""
    cv = cv or make_cv()
    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    Xtr, ytr, Xte, yte = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    model = IrradianceZoneModel(spec, df.attrs.get("name", ""), model_params)
    disc = IrradianceDiscretizer(spec.discretizer, model.n_zones).fit(ytr)
    ltr = disc.transform(ytr)
    k = disc.n_classes_
    template = make_pipeline(spec, **(model_params or {}))

    rows = []
    for fit_i, val_i in cv.split(Xtr, ltr):
        t0 = time.perf_counter()
        pipe = clone(template).fit(Xtr[fit_i], ltr[fit_i])
        t1 = time.perf_counter()
        pred = pipe.predict(Xtr[val_i])
        sc = decision_scores(pipe, Xtr[val_i])
        t2 = time.perf_counter()
        m = compute_metrics(ltr[val_i], pred, sc, pipe.classes_, k)
        m.update(fit_time=t1 - t0, score_time=t2 - t1)
        rows.append(m)
    cvdf = pd.DataFrame(rows)

    model.fit(Xtr, ytr)
    lte = model.labels(yte)
    pred = model.predict(Xte)
    sc = model.scores(Xte)
    hold = compute_metrics(lte, pred, sc, model.classes_, k)

    out = {"dataset": df.attrs.get("name", ""), **spec._asdict(), "spec_id": spec.id,
           "n_zones": k}
    for m in METRICS + ("fit_time", "score_time"):
        out[f"cv_{m}"] = float(cvdf[m].mean())
        out[f"cv_{m}_std"] = float(cvdf[m].std(ddof=1))
    for m in METRICS:
        out[f"test_{m}"] = float(hold[m])
    if keep_predictions:
        out["_model"] = model
        out["_y_true"] = lte
        out["_y_pred"] = pred
        out["_scores"] = sc
        out["_confusion"] = confusion_matrix(lte, pred, labels=np.arange(k))
    return out


def run_grid(datasets, specs, out_csv, split=None, model_params=None, verbose=True):
    """Evalúa todas las configuraciones y guarda cada fila en ``out_csv``.
    Es reanudable: si el CSV ya contiene una configuración, no se repite."""
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["dataset"], prev["spec_id"]))
    total = len(datasets) * len(specs)
    t0 = time.time()
    for name, df in datasets.items():
        tr, te = split[name] if split else holdout_split(df)
        for spec in specs:
            if (name, spec.id) in done:
                continue
            try:
                row = evaluate_spec(spec, df, tr, te, model_params)
                target = out_csv
            except Exception as exc:  # se registra aparte y se sigue
                row = {"dataset": name, "spec_id": spec.id, "error": repr(exc)}
                target = out_csv.replace(".csv", "_errors.csv")
            pd.DataFrame([row]).to_csv(target, mode="a", index=False,
                                       header=not os.path.exists(target))
            done.add((name, spec.id))
            if verbose and len(done) % 50 == 0:
                print(f"{len(done)}/{total} configuraciones · {time.time() - t0:.0f}s", flush=True)
    return pd.read_csv(out_csv)


# ---------------------------------------------------------------------------
# Etapa 2: sintonización de hiperparámetros de los mejores pipelines
# ---------------------------------------------------------------------------
BANDWIDTHS = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
PARAM_SPACE = {
    "ksvc": {"model__C": list(np.logspace(0, 5, 11)), "model__bandwidth": BANDWIDTHS,
             "model__class_weight": [None, "balanced"]},
    "kridge": {"model__alpha": list(np.logspace(-4, 2, 13)), "model__bandwidth": BANDWIDTHS,
               "model__class_weight": [None, "balanced"]},
    "kann": {"model__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], "model__bandwidth": BANDWIDTHS,
             "model__hidden_layer_sizes": [(25,), (50,), (100,), (50, 25)],
             "model__activation": ["relu", "tanh", "identity"]},
}


def tune_spec(spec, df, train_idx, n_iter=20, n_splits=5):
    """Búsqueda aleatoria (criterio: media de MCC y F1 macro) sobre el 80 % de entrenamiento.
    Devuelve los mejores parámetros del modelo (sin el prefijo model__)."""
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    X = df[FEATURES].to_numpy(float)[train_idx]
    y = df[TARGET].to_numpy(float)[train_idx]
    labels = IrradianceDiscretizer(spec.discretizer, N_ZONES).fit_transform(y)
    space = dict(PARAM_SPACE[spec.model])
    if spec.kernel == "linear":
        space.pop("model__bandwidth", None)
    if spec.kernel == "hyperbolic":
        space["model__coef0"] = [-1.0, -0.5, 0.0, 0.5]
    def _score(y_true, y_pred):
        return float(selection_score(matthews_corrcoef(y_true, y_pred),
                                     f1_score(y_true, y_pred, average="macro", zero_division=0)))

    search = RandomizedSearchCV(
        make_pipeline(spec), space, n_iter=n_iter, random_state=SEED,
        scoring=make_scorer(_score),
        cv=StratifiedKFold(n_splits, shuffle=True, random_state=SEED), error_score=np.nan)
    search.fit(X, labels)
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return best, float(search.best_score_)


# ---------------------------------------------------------------------------
# Exportación de modelos (artefacto que usa la aplicación web)
# ---------------------------------------------------------------------------
def _clean(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def build_artifact(result, source="notebook"):
    """Empaqueta un resultado de ``evaluate_spec(keep_predictions=True)``."""
    import datetime
    model = result["_model"]
    spec = model.spec
    meta = {
        "dataset": result["dataset"],
        "spec": spec._asdict(),
        "spec_id": spec.id,
        "label": f"{result['dataset'].upper()} · {MODEL_LABELS[spec.model]} · {spec.kernel} "
                 f"({spec.scaler}/{spec.discretizer}/{spec.reducer})",
        "params": {k: _clean(v) for k, v in model.model_params.items()},
        "kernel_params": {k: _clean(v) for k, v in
                          getattr(model.pipeline_.named_steps["model"], "kernel_params_", {}).items()},
        "cv": {m: result[f"cv_{m}"] for m in METRICS},
        "cv_std": {m: result[f"cv_{m}_std"] for m in METRICS},
        "test": {m: result[f"test_{m}"] for m in METRICS},
        "cv_score": float(selection_score(result["cv_mcc"], result["cv_f1"])),
        "fit_time": result["cv_fit_time"],
        "confusion": result["_confusion"].tolist(),
        "zones": model.zones(),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": source,
    }
    return {"model": model, "meta": meta}


def save_artifact(artifact, models_dir="models", name=None):
    import joblib
    import re
    os.makedirs(models_dir, exist_ok=True)
    meta = artifact["meta"]
    slug = name or re.sub(r"[^a-z0-9]+", "-", f"{meta['dataset']}-{meta['spec_id']}".lower()).strip("-")
    path = os.path.join(models_dir, f"{slug}.joblib")
    meta["id"] = slug
    joblib.dump(artifact, path, compress=3)
    return path
