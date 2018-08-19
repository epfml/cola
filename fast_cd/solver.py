import warnings
import numpy as np

from scipy import sparse
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_random_state
from . import svm
from . import elasticnet


class SVMCoordSolver(object):
    def __init__(self, C, max_iter=1.0, tol=1e-4, warm_start=True, random_state=0):
        self.C = C
        self.max_iter = float(max_iter)
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.rng = check_random_state(self.random_state)

    def fit(self, X, y, b, check_input=True):
        assert y.ndim == 1
        n_samples, n_features = X.shape
        n_targets = 1

        if check_input:
            X, y = check_X_y(X, y, accept_sparse='csr', order='F', dtype=[np.float32, np.float64],
                             copy=True, multi_output=True, y_numeric=True)
            y = check_array(y, order='F', copy=True, dtype=X.dtype.type, ensure_2d=False)
            b = check_array(b, order='F', copy=True, dtype=X.dtype.type, ensure_2d=False)

        if not self.warm_start or not hasattr(self, "dual_coef_"):
            dual_coef_ = np.zeros(n_samples, dtype=X.dtype, order='F')
        else:
            dual_coef_ = self.dual_coef_

        # Computation
        dual_coef_ = np.asfortranarray(dual_coef_, dtype=X.dtype)
        if sparse.isspmatrix(X):
            Xt = X.T
            model = svm.sparse_cd(dual_coef_, y, b, Xt.data, Xt.indices, Xt.indptr, Xt.shape[0],
                                  self.max_iter, self.C, self.tol, self.rng)
        else:
            Xt = np.asfortranarray(X.T, dtype=X.dtype)
            model = svm.dense_cd(dual_coef_, y, b, Xt, self.max_iter, self.C, self.tol, self.rng)
        dual_coef_, dual_gap_, eps_, n_iter_ = model

        self.dual_coef_ = np.asarray(dual_coef_, dtype=X.dtype)
        self.coef_ = self.C * dual_coef_ @ X
        self.dual_gap_ = dual_gap_
        self.n_iter_ = n_iter_
        return self


class ElasticNetCoordSolver(object):
    """
            min 1/ (2 * n_samples) * || Ax - y ||^2_2
            + l1_ratio * lambda_ * || x ||_1
            + (1 - l1_ratio) * lambda_ * 0.5 * || x ||_2^2
    """
    # The duality gap computed in the pyx file use the dual variable
    #       w = ratio * grad f(v)
    # where the ratio is chosen such that the dual of elastic net regularizer is 0.

    def __init__(self, lambda_=1.0, l1_ratio=0.5, max_iter=1.0, tol=1e-4, warm_start=True, random_state=0):
        self.lambda_ = lambda_
        self.l1_ratio = l1_ratio
        self.max_iter = float(max_iter)
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.rng = check_random_state(self.random_state)

    def fit(self, X, y, check_input=True):
        assert y.ndim == 1

        n_samples, n_features = X.shape
        n_targets = 1

        if check_input:
            X, y = check_X_y(X, y, accept_sparse='csc', order='F', dtype=[np.float32, np.float64],
                             copy=True, multi_output=True, y_numeric=True)
            y = check_array(y, order='F', copy=True, dtype=X.dtype.type, ensure_2d=False)

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_ = np.zeros(n_features, dtype=X.dtype, order='F')
        else:
            coef_ = self.coef_

        alpha = self.lambda_ * self.l1_ratio * len(y)
        beta = self.lambda_ * (1 - self.l1_ratio) * len(y)

        # Computation
        coef_ = np.asfortranarray(coef_, dtype=X.dtype)
        y = y.astype(X.dtype)
        if sparse.isspmatrix(X):
            model = elasticnet.sparse_cd(coef_, alpha, beta, X.data, X.indices, X.indptr, y,
                                         self.max_iter, self.tol, self.rng)
        else:
            model = elasticnet.dense_cd(coef_, alpha, beta, X, y, self.max_iter, self.tol, self.rng)
        coef_, gap_, eps_, n_iter_ = model

        self.coef_ = np.asarray(coef_, dtype=X.dtype)
        self.gap_ = gap_
        self.n_iter_ = n_iter_
        return self


class DADMMElasticNetCoordSolver(object):
    """
    This solver is used by ADMM local solver. It assumes the input problem has the following form.

            min 1/ (2 * n_samples) * || Ax - y ||^2_2
            + l1_ratio * lambda_ * || x ||_1
            + (1 - l1_ratio) * lambda_ * 0.5 * || x ||_2^2
    """
    # The duality gap computed in the pyx file use the dual variable
    #       w = ratio * grad f(v)
    # where the ratio is chosen such that the dual of elastic net regularizer is 0.

    def __init__(self, lambda_=1.0, l1_ratio=0.5, max_iter=1.0, tol=1e-4, warm_start=True,
                 rho=None, n_neighbor=None, random_state=0):
        self.lambda_ = lambda_
        self.l1_ratio = l1_ratio
        self.max_iter = float(max_iter)
        self.tol = tol
        self.rho = rho
        self.n_neighbor = n_neighbor
        self.warm_start = warm_start
        self.random_state = random_state
        self.rng = check_random_state(self.random_state)

    def fit(self, X, y, c, check_input=True):
        assert y.ndim == 1

        n_samples, n_features = X.shape
        n_targets = 1

        if check_input:
            X, y = check_X_y(X, y, accept_sparse='csc', order='F', dtype=[np.float32, np.float64],
                             copy=True, multi_output=True, y_numeric=True)
            y = check_array(y, order='F', copy=True, dtype=X.dtype.type, ensure_2d=False)

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_ = np.zeros(n_features, dtype=X.dtype, order='F')
        else:
            coef_ = self.coef_

        alpha = self.lambda_ * self.l1_ratio * len(y)
        beta = self.lambda_ * (1 - self.l1_ratio) * len(y) + 2 * self.rho * self.n_neighbor

        # Computation
        coef_ = np.asfortranarray(coef_, dtype=X.dtype)
        if sparse.isspmatrix(X):
            model = elasticnet.sparse_cd(coef_, alpha, beta, X.data, X.indices, X.indptr, y, c,
                                         self.max_iter, self.tol, self.rng)
        else:
            model = elasticnet.dense_cd(coef_, alpha, beta, X, y, c, self.max_iter, self.tol, self.rng)
        coef_, gap_, eps_, n_iter_ = model

        self.coef_ = np.asarray(coef_, dtype=X.dtype)
        self.gap_ = gap_
        self.n_iter_ = n_iter_
        return self
