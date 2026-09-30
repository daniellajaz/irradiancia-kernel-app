"""Funciones kernel de Belanche en forma matricial (vectorizada).

Las funciones de ``KernelUtilities.py`` del repositorio magohector/fkernel
calculan k(x, y) para UN par de vectores y construyen la matriz de Gram con un
doble bucle de Python. Aquí se reescribe la MISMA matemática para matrices
completas X (n x d) e Y (m x d), de modo que la matriz de Gram K(X, Y) se
obtiene con operaciones de NumPy.

Además, ``KernelFunction`` es una clase (no una clausura) y por tanto se puede
serializar con joblib/pickle: esto es lo que permite guardar los modelos y
desplegarlos en la aplicación web.

Calibración por la mediana
--------------------------
Cuando el parámetro de escala vale ``'median'`` se fija a partir de los datos
de entrenamiento para que el kernel valga 0.5 a la distancia mediana entre
observaciones. Así todos los kernels arrancan en una escala comparable (la
comparación es justa) y ``bandwidth`` actúa como multiplicador común que se
puede sintonizar después.
"""
import numpy as np

LN2 = np.log(2.0)

# Nombre corto -> descripción (se usa en el notebook y en la app web)
KERNEL_CATALOG = {
    "linear": "Lineal  k = x·y",
    "poly": "Polinómico  k = (a x·y + 1)^m",
    "rbf": "Gaussiano / MRBF  k = exp(-g Σ(x-y)^β)",
    "hyperbolic": "Tangente hiperbólica  k = tanh(a x·y + b)",
    "triangle": "Triangular  k = max(0, 1 - ||x-y||/a)",
    "radial_basic": "Radial básico (ANOVA)  k = (Σ exp(-g (x_k-y_k)^2))^m",
    "rquadratic": "Cuadrático racional  k = 1 - ||x-y||²/(||x-y||² + c)",
    "canberra": "Canberra  k = 1 - (1/d) Σ g|x_k-y_k|/(|x_k|+|y_k|)",
    "truncated": "Truncado  k = (1/d) Σ max(0, 1 - |x_k-y_k|/g)",
}
KERNELS = tuple(KERNEL_CATALOG)


# ---------------------------------------------------------------------------
# Piezas básicas de distancia
# ---------------------------------------------------------------------------
def _sq_dists(X, Y):
    """||x_i - y_j||² sin crear el tensor n x m x d."""
    xx = np.einsum("ij,ij->i", X, X)[:, None]
    yy = np.einsum("ij,ij->i", Y, Y)[None, :]
    return np.maximum(xx + yy - 2.0 * X @ Y.T, 0.0)


def _coord_diffs(X, Y):
    """Tensor de diferencias coordenada a coordenada (n x m x d)."""
    return X[:, None, :] - Y[None, :, :]


def _chunked(func, X, Y, max_cells=4_000_000):
    """Evalúa func(Xc, Y) por bloques para acotar memoria en kernels que
    necesitan el tensor n x m x d (radial básico, canberra, truncado, mrbf)."""
    per_row = max(1, Y.shape[0] * max(1, X.shape[1]))
    step = max(1, max_cells // per_row)
    if X.shape[0] <= step:
        return func(X, Y)
    return np.vstack([func(X[i:i + step], Y) for i in range(0, X.shape[0], step)])


# ---------------------------------------------------------------------------
# Kernels en forma matricial (misma definición que KernelUtilities.py)
# ---------------------------------------------------------------------------
def k_linear(X, Y, **_):
    return X @ Y.T


def k_poly(X, Y, degree=2, gamma=0.01, **_):
    return (gamma * (X @ Y.T) + 1.0) ** degree


def k_mrbf(X, Y, degree=2, gamma=0.01, **_):
    if degree == 2:  # caso gaussiano: no hace falta el tensor
        return np.exp(-gamma * _sq_dists(X, Y))

    def f(A, B):
        return np.exp(-np.sum(gamma * _coord_diffs(A, B) ** degree, axis=2))
    return _chunked(f, X, Y)


def k_hyperbolic(X, Y, gamma=0.01, coef0=0.0, **_):
    return np.tanh(gamma * (X @ Y.T) + coef0)


def k_triangle(X, Y, gamma=0.01, **_):
    dist = np.sqrt(_sq_dists(X, Y))
    return np.where(dist <= gamma, 1.0 - dist / gamma, 0.0)


def k_radial_basic(X, Y, degree=2, gamma=0.01, **_):
    def f(A, B):
        return np.sum(np.exp(-gamma * _coord_diffs(A, B) ** 2), axis=2) ** degree
    return _chunked(f, X, Y)


def k_rquadratic(X, Y, coef0=0.1, **_):
    d2 = _sq_dists(X, Y)
    return 1.0 - d2 / (d2 + coef0)


def k_canberra(X, Y, gamma=1.0, **_):
    d = X.shape[1]

    def f(A, B):
        num = gamma * np.abs(_coord_diffs(A, B))
        den = np.abs(A)[:, None, :] + np.abs(B)[None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = num / den
        r[~np.isfinite(r)] = 0.0  # el original descarta los NaN (0/0)
        return 1.0 - r.sum(axis=2) / d
    return _chunked(f, X, Y)


def k_truncated(X, Y, gamma=0.01, **_):
    d = X.shape[1]

    def f(A, B):
        val = 1.0 - np.abs(_coord_diffs(A, B)) / gamma
        return np.where(val > 0, val, 0.0).sum(axis=2) / d
    return _chunked(f, X, Y)


_MATRIX_KERNELS = {
    "linear": k_linear, "poly": k_poly, "rbf": k_mrbf, "mrbf": k_mrbf,
    "hyperbolic": k_hyperbolic, "triangle": k_triangle,
    "radial_basic": k_radial_basic, "rquadratic": k_rquadratic,
    "canberra": k_canberra, "can": k_canberra,
    "truncated": k_truncated, "tru": k_truncated,
}
# Alias usados por las clases originales KSVC / KANNC
ORIGINAL_NAMES = {"canberra": "can", "truncated": "tru"}


# ---------------------------------------------------------------------------
# Heurística de la mediana
# ---------------------------------------------------------------------------
def median_statistics(X, max_points=300, random_state=0):
    """Distancia euclídea mediana, diferencia absoluta mediana por coordenada
    y 'escala' tipo sklearn (1 / (d · var(X)))."""
    X = np.asarray(X, dtype=float)
    if X.shape[0] > max_points:
        idx = np.random.default_rng(random_state).choice(X.shape[0], max_points, replace=False)
        X = X[idx]
    iu = np.triu_indices(X.shape[0], k=1)
    dist = np.sqrt(_sq_dists(X, X))[iu]
    dist = dist[dist > 0]
    coord = np.abs(_coord_diffs(X, X))[iu].ravel()
    coord = coord[coord > 0]
    var = X.var()
    return {
        "median_dist": float(np.median(dist)) if dist.size else 1.0,
        "median_coord": float(np.median(coord)) if coord.size else 1.0,
        "scale": float(1.0 / (X.shape[1] * var)) if var > 0 else 1.0,
    }


def calibrate(name, stats, bandwidth=1.0):
    """Devuelve (gamma, coef0) calibrados para que k ≈ 0.5 a la distancia
    mediana. ``bandwidth`` > 1 ensancha el kernel (más suave)."""
    s, c, sc = stats["median_dist"], stats["median_coord"], stats["scale"]
    b = float(bandwidth)
    if name in ("rbf", "mrbf"):
        return LN2 / (b * s) ** 2, None
    if name == "radial_basic":
        return LN2 / (b * c) ** 2, None
    if name == "triangle":
        return 2.0 * b * s, None
    if name in ("truncated", "tru"):
        return 2.0 * b * c, None
    if name == "rquadratic":
        return None, (b * s) ** 2
    if name in ("poly", "hyperbolic"):
        return sc / b, None
    if name in ("canberra", "can"):
        return min(1.0, 1.0 / b), None  # gamma ∈ (0, 1]
    return None, None


class KernelFunction:
    """Kernel serializable que devuelve la matriz de Gram K(X, Y).

    Parámetros
    ----------
    name : nombre del kernel (ver ``KERNEL_CATALOG``)
    degree, gamma, coef0 : mismos significados que en KernelUtilities.py.
        ``gamma='median'`` (o coef0 en rquadratic) activa la calibración.
    bandwidth : multiplicador de la escala calibrada.
    """

    def __init__(self, name="rbf", degree=2, gamma="median", coef0=0.0, bandwidth=1.0):
        if name not in _MATRIX_KERNELS:
            raise ValueError(f"Kernel desconocido: {name!r}")
        self.name = name
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.bandwidth = bandwidth

    # -- ajuste de la escala ------------------------------------------------
    def fit(self, X):
        self.gamma_, self.coef0_ = self.gamma, self.coef0
        auto = isinstance(self.gamma, str) or (self.name == "rquadratic" and isinstance(self.coef0, str))
        if auto:
            g, c = calibrate(self.name, median_statistics(X), self.bandwidth)
            if g is not None and isinstance(self.gamma, str):
                self.gamma_ = g
            if c is not None:
                self.coef0_ = c
        if isinstance(self.gamma_, str):  # kernels sin gamma (lineal, rq)
            self.gamma_ = 0.01
        if isinstance(self.coef0_, str):
            self.coef0_ = 0.1 if self.name == "rquadratic" else 0.0
        return self

    def params_(self):
        return {"degree": self.degree, "gamma": float(self.gamma_), "coef0": float(self.coef0_)}

    # -- evaluación ---------------------------------------------------------
    def __call__(self, X, Y):
        if not hasattr(self, "gamma_"):
            self.fit(X)
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        K = _MATRIX_KERNELS[self.name](X, Y, **self.params_())
        return np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

    def __repr__(self):
        extra = ""
        if hasattr(self, "gamma_"):
            extra = f", gamma_={self.gamma_:.4g}, coef0_={self.coef0_:.4g}"
        return f"KernelFunction({self.name!r}, bandwidth={self.bandwidth}{extra})"
