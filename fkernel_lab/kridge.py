"""KRidgeClassifier: RidgeClassifier de scikit-learn extendido con el truco
kernel (formulación dual de la regresión Ridge)."""
import numpy as np
from scipy import linalg
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .kernels import KernelFunction


class KRidgeClassifier(RidgeClassifier):
    """Clasificador Ridge kernelizado.

    RidgeClassifier codifica las clases como +1/-1 (una columna por clase) y
    resuelve una regresión Ridge multisalida. Con el truco kernel el vector de
    pesos vive en el espacio de Hilbert, w = Σ α_i φ(x_i), y el problema

        min_α  Σ_i s_i ||y_i - f(x_i)||² + alpha · αᵀ K α

    tiene solución cerrada (K̃ + alpha · S⁻¹) α = Ỹ, donde K̃ es la matriz de
    Gram centrada (el intercepto no se penaliza), S son los pesos de muestra
    (``class_weight``) e Ỹ los objetivos centrados. La predicción es
    f(x) = k̃(x)ᵀ α + ȳ y la clase es argmax f(x), exactamente como en
    RidgeClassifier: por eso ``predict`` se HEREDA sin cambios.
    """

    def __init__(self, alpha=1.0, kernel="rbf", degree=2, gamma="median", coef0=1.0,
                 bandwidth=1.0, fit_intercept=True, class_weight=None):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, class_weight=class_weight)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.bandwidth = bandwidth

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, dtype=float)
        self.n_features_in_ = X.shape[1]
        # Misma codificación que RidgeClassifier: +1 clase correcta, -1 resto.
        # Debe llamarse _label_binarizer: RidgeClassifier lo usa en predict().
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y).astype(float)
        classes = self._label_binarizer.classes_
        try:
            self.classes_ = classes  # sklearn >= 1.7: atributo normal
        except AttributeError:
            pass  # sklearn <= 1.6: classes_ es una propiedad de solo lectura
            #       que ya devuelve self._label_binarizer.classes_
        if len(classes) < 2:
            raise ValueError("KRidgeClassifier necesita al menos dos clases")

        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if self.class_weight is not None:
            w = w * compute_sample_weight(self.class_weight, y)
        w = w / w.mean()

        # Truco kernel: solo se necesita la matriz de Gram K(X, X)
        self.kernel_ = KernelFunction(self.kernel, self.degree, self.gamma, self.coef0,
                                      self.bandwidth).fit(X)
        K = self.kernel_(X, X)
        self.X_fit_ = X

        if self.fit_intercept:  # centrado (ponderado) en el espacio de Hilbert
            p = w / w.sum()
            self._p = p
            self._k_col_mean = K @ p                 # E_p[k(x_i, ·)]
            self._k_all_mean = float(p @ K @ p)
            self._y_offset = p @ Y
            Kc = K - self._k_col_mean[:, None] - self._k_col_mean[None, :] + self._k_all_mean
            Yc = Y - self._y_offset
        else:
            self._y_offset = np.zeros(Y.shape[1])
            Kc, Yc = K, Y

        # Sistema simétrico equivalente: (S½ K̃ S½ + alpha I) β = S½ Ỹ,  α = S½ β
        sq = np.sqrt(w)
        A = sq[:, None] * Kc * sq[None, :]
        A[np.diag_indices_from(A)] += self.alpha
        B = sq[:, None] * Yc
        try:
            beta = linalg.solve(A, B, assume_a="sym")
        except (linalg.LinAlgError, ValueError):
            beta = linalg.lstsq(A, B)[0]  # kernels no definidos positivos
        self.dual_coef_ = sq[:, None] * beta
        self.kernel_params_ = self.kernel_.params_()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["dual_coef_", "X_fit_"])
        X = check_array(X, dtype=float)
        K = self.kernel_(X, self.X_fit_)
        if self.fit_intercept:  # mismo centrado que en fit
            K = K - (K @ self._p)[:, None] - self._k_col_mean[None, :] + self._k_all_mean
        scores = K @ self.dual_coef_ + self._y_offset
        return scores.ravel() if scores.shape[1] == 1 else scores

    # predict() se hereda de RidgeClassifier: argmax de decision_function
