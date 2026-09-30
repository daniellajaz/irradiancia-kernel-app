"""Estimadores kernel para clasificación de irradiancia.

* ``KSVCVec``  -> hereda de ``KSVC`` (KSVM.py original).
* ``KANNCVec`` -> hereda de ``KANNC`` (KANN.py original).
(``KRidgeClassifier`` está en ``kridge.py``.)

Las dos primeras conservan el constructor, ``set_params``, ``predict`` y
``mtransform`` de las clases originales y reutilizan su ``fit`` cuando el kernel
es nativo de sklearn. Para los kernels alternativos sustituyen las clausuras
de ``KernelUtilities`` (doble bucle de Python y no serializables) por
``KernelFunction``, que calcula la misma matriz de Gram de forma vectorizada y
se puede guardar con joblib. En el notebook se comprueba numéricamente que las
predicciones coinciden con las de las clases originales.
"""
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .kernels import KernelFunction
from .original import KANNC, KSVC

NATIVE_SVC = ("linear", "poly", "rbf")  # los que KSVC delega en sklearn


def _kernel_name(kernel):
    return kernel.name if isinstance(kernel, KernelFunction) else kernel


# ---------------------------------------------------------------------------
# KSVC vectorizado
# ---------------------------------------------------------------------------
class KSVCVec(KSVC):
    """KSVC del repositorio + kernels vectorizados/serializables."""

    def __init__(self, C=1.0, kernel="rbf", degree=2, gamma="median", coef0=0.0,
                 bandwidth=1.0, shrinking=True, probability=False, tol=1e-3,
                 cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape="ovr", random_state=None):
        super().__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                         shrinking=shrinking, probability=probability, tol=tol,
                         cache_size=cache_size, class_weight=class_weight,
                         verbose=verbose, max_iter=max_iter,
                         decision_function_shape=decision_function_shape,
                         random_state=random_state)
        self.bandwidth = bandwidth

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        name = _kernel_name(self.kernel)
        self.kernel_name_ = name
        user_gamma = self.gamma
        kf = KernelFunction(name, self.degree, self.gamma, self.coef0, self.bandwidth).fit(X)
        try:
            if name in NATIVE_SVC:
                # Camino ORIGINAL: KSVC.fit delega en SVC (gamma ya numérico)
                self.kernel = name
                self.gamma = kf.gamma_ if name != "linear" else "scale"
                KSVC.fit(self, X, y)
            else:
                # Kernel alternativo: callable Gram vectorizado y serializable
                self.kernel = kf
                self.gamma = "scale"  # sklearn lo ignora con kernels callable
                SVC.fit(self, X, y, sample_weight=sample_weight)
        finally:
            self.gamma = user_gamma
        self.kernel_params_ = kf.params_()
        return self


# ---------------------------------------------------------------------------
# Aproximación de Nyström con kernel vectorizado
# ---------------------------------------------------------------------------
class KernelNystroem(TransformerMixin, BaseEstimator):
    """Equivalente a ``sklearn.kernel_approximation.Nystroem`` (misma
    normalización U S^-1/2 V) pero usando ``KernelFunction`` matricial."""

    def __init__(self, kernel=None, n_components=100, random_state=None):
        self.kernel = kernel
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.kernel_ = self.kernel if self.kernel is not None else KernelFunction("rbf")
        self.kernel_.fit(X)
        n = min(self.n_components, X.shape[0])
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(X.shape[0])[:n]
        self.components_ = X[idx]
        K = self.kernel_(self.components_, self.components_)
        U, S, V = linalg.svd(K)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        return self

    def transform(self, X):
        K = self.kernel_(np.asarray(X, dtype=float), self.components_)
        return K @ self.normalization_.T


# ---------------------------------------------------------------------------
# KANN vectorizado
# ---------------------------------------------------------------------------
class KANNCVec(KANNC):
    """KANNC del repositorio (MLP sobre un mapa de Nyström) con kernels
    vectorizados, ``predict_proba`` coherente y serialización."""

    def __init__(self, hidden_layer_sizes=(100,), activation="identity", solver="adam",
                 alpha=0.0001, learning_rate_init=0.001, max_iter=200, random_state=None,
                 tol=1e-4, early_stopping=True, validation_fraction=0.1,
                 n_iter_no_change=10, kernel="rbf", degree=2, gamma="median",
                 coef0=0.0, bandwidth=1.0, n_components=100):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                         solver=solver, alpha=alpha, learning_rate_init=learning_rate_init,
                         max_iter=max_iter, random_state=random_state, tol=tol,
                         early_stopping=early_stopping,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change, kernel=kernel,
                         degree=degree, gamma=gamma, coef0=coef0)
        self.bandwidth = bandwidth
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.kernel_name_ = self.kernel
        if self.kernel == "linear":
            return KANNC.fit(self, X, y)  # camino original sin mapa kernel
        kf = KernelFunction(self.kernel, self.degree, self.gamma, self.coef0, self.bandwidth)
        self.feature_map_nystroem = KernelNystroem(kf, self.n_components, self.random_state)
        Z = self.feature_map_nystroem.fit_transform(X)
        MLPClassifier.fit(self, Z, y)
        self.isfit = True
        self.kernel_params_ = kf.params_()
        return self

    # predict() y mtransform() se heredan tal cual de KANNC
    def predict_proba(self, X):
        return MLPClassifier.predict_proba(self, self.mtransform(np.asarray(X, dtype=float)))
