"""Discretización de la irradiancia (variable objetivo) y reducción de
dimensionalidad.

Discretizador por UMBRALES
--------------------------
Cada método (uniforme, K-Means, DBSCAN, jerárquico) agrupa los valores de
irradiancia de ENTRENAMIENTO. Como la irradiancia es unidimensional, cada grupo
es un intervalo contiguo; se ordenan de menor a mayor y el umbral entre dos
zonas consecutivas es el punto medio entre el máximo de una y el mínimo de la
siguiente. El resultado es una lista de cortes y cada clase es un rango de
irradiancia interpretable (p. ej. "Zona 2: 203-218 W/m²"), que es lo que
necesita el mapa de la aplicación. Valores nuevos (o puntos de ruido de DBSCAN)
se clasifican con ``np.digitize`` sobre esos cortes.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

DISCRETIZERS = ("uniform", "kmeans", "dbscan", "hierarchical")
DISCRETIZER_LABELS = {
    "uniform": "Uniforme (rangos iguales)",
    "kmeans": "Clustering K-Means",
    "dbscan": "Clustering DBSCAN",
    "hierarchical": "Clustering jerárquico (Ward)",
}


class IrradianceDiscretizer:
    """Convierte irradiancia continua en zonas 0..k-1 ordenadas."""

    def __init__(self, method="uniform", n_zones=4, min_samples=8, random_state=2021):
        if method not in DISCRETIZERS:
            raise ValueError(f"Discretizador desconocido: {method!r}")
        self.method = method
        self.n_zones = n_zones
        self.min_samples = min_samples
        self.random_state = random_state

    # --- agrupamientos 1-D ------------------------------------------------
    def _groups(self, v):
        col = v[:, None]
        if self.method == "kmeans":
            return KMeans(self.n_zones, n_init=20, random_state=self.random_state).fit_predict(col)
        if self.method == "hierarchical":
            return AgglomerativeClustering(self.n_zones, linkage="ward").fit_predict(col)
        if self.method == "dbscan":
            return self._dbscan(v)
        raise AssertionError

    def _dbscan(self, v):
        """DBSCAN no recibe k: se barre eps y se elige la partición que
        (1) deja como máximo un 10 % de ruido, (2) tiene un número de grupos
        lo más cercano posible a ``n_zones`` y (3) tiene el menor ruido."""
        z = ((v - v.mean()) / (v.std() or 1.0))[:, None]
        best = None
        for eps in np.linspace(0.02, 0.6, 59):
            labels = DBSCAN(eps=eps, min_samples=self.min_samples).fit_predict(z)
            groups = set(labels) - {-1}
            if len(groups) < 2:
                continue
            noise = float(np.mean(labels == -1))
            key = (noise > 0.10, abs(len(groups) - self.n_zones), noise)
            if best is None or key < best[0]:
                best = (key, labels, eps)
        if best is None:
            raise ValueError("DBSCAN no encontró al menos dos zonas")
        self.eps_ = float(best[2])
        self.noise_fraction_ = float(np.mean(best[1] == -1))
        return best[1]

    # --- API --------------------------------------------------------------
    def fit(self, y):
        v = np.asarray(y, dtype=float).ravel()
        if self.method == "uniform":
            cuts = np.linspace(v.min(), v.max(), self.n_zones + 1)[1:-1]
        else:
            labels = self._groups(v)
            ranges = sorted((v[labels == g].min(), v[labels == g].max())
                            for g in set(labels) - {-1})
            cuts = [(hi + lo_next) / 2 for (_, hi), (lo_next, _) in zip(ranges[:-1], ranges[1:])]
        self.cuts_ = np.asarray(cuts, dtype=float)
        self.n_classes_ = len(self.cuts_) + 1
        self.y_min_, self.y_max_ = float(v.min()), float(v.max())
        return self

    def transform(self, y):
        return np.digitize(np.asarray(y, dtype=float).ravel(), self.cuts_)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def zones(self):
        """Tabla de zonas: índice, rango y punto medio (para mapas/leyendas)."""
        edges = np.r_[self.y_min_, self.cuts_, self.y_max_]
        names = zone_names(self.n_classes_)
        return [{"zone": i, "name": names[i], "low": float(edges[i]), "high": float(edges[i + 1]),
                 "mid": float((edges[i] + edges[i + 1]) / 2)} for i in range(self.n_classes_)]


def zone_names(k):
    presets = {2: ["Baja", "Alta"], 3: ["Baja", "Media", "Alta"],
               4: ["Baja", "Media-baja", "Media-alta", "Alta"],
               5: ["Muy baja", "Baja", "Media", "Alta", "Muy alta"]}
    return presets.get(k, [f"Zona {i + 1}" for i in range(k)])


class DimReducer(TransformerMixin, BaseEstimator):
    """PCA o LDA reduciendo SIEMPRE al mismo número de dimensiones.

    LDA puede producir como máximo (n_clases - 1) componentes. Para que PCA y
    LDA se comparen en igualdad de condiciones, ambos proyectan a
    min(n_clases - 1, n_variables) dimensiones; la diferencia está solo en el
    criterio (varianza vs. separabilidad entre clases)."""

    def __init__(self, method="pca", random_state=2021):
        self.method = method
        self.random_state = random_state

    def fit(self, X, y):
        k = max(1, min(len(np.unique(y)) - 1, X.shape[1]))
        if self.method == "pca":
            self.model_ = PCA(n_components=k, random_state=self.random_state)
        elif self.method == "lda":
            self.model_ = LinearDiscriminantAnalysis(n_components=k)
        else:
            raise ValueError(f"Reductor desconocido: {self.method!r}")
        self.model_.fit(X, y)
        self.n_components_ = k
        return self

    def transform(self, X):
        return self.model_.transform(X)
