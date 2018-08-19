"""Given a centralized optimization objective, solve the cocoa-style subproblem."""
import numpy as np
from abc import ABCMeta
from abc import abstractmethod

from fast_cd.solver import SVMCoordSolver


class CoCoASubproblemSolver(metaclass=ABCMeta):
    """Solve cocoa subproblem.

    Assume the centralized objective has the following form:
            min f(Ax) + g(x)
    The CoCoA subproblem at node k is
            min_dx_[k] <grad_f(v), A_[k] dx_[k]> + sigma/2tau ||A_[k] dx_[k]||_2^2
                        + g_[k](x_[k])
    This solver will transform the subproblem to some standard forms so that they can
    by solved by existing solvers.
    """

    @property
    def tau(self):
        """Smoothness of `f`"""
        return self._tau

    @property
    def sigma(self):
        """Safe bount"""
        return self._sigma

    @abstractmethod
    def grad_f(self, v):
        """Gradient of f"""
        pass

    @abstractmethod
    def f(self, v):
        """Value of f(v)"""
        pass

    @abstractmethod
    def gk(self, xk):
        """Sum of gi(x_i) for i in node k"""
        pass

    @abstractmethod
    def f_conj(self, w):
        """Convex conjugate of f"""
        pass

    @abstractmethod
    def gk_conj(self, yk):
        """Sum of gi_conj(x_i) for i in node k where gi_conj is the conjugate of gi"""
        pass

    def dist_init(self, Ak, y, theta, local_iters, sigma):
        """Initialize for distributed environment."""
        self.Ak = Ak
        self.y = y
        self.theta = theta
        self._sigma = sigma
        self.local_iters = local_iters
        self.load_approximate_solver(sigma, local_iters, theta)

    @property
    @abstractmethod
    def solver_coef(self):
        pass

    @abstractmethod
    def load_approximate_solver(self, sigma, local_iters, theta):
        """Load approximate solver to solve the standized problem."""
        pass

    @abstractmethod
    def standize_subproblem(self, v, w):
        """Convert subproblem to a standard form so that local solver can solve."""
        pass

    @abstractmethod
    def recover_solution(self, xk):
        """From the standardized solution to original solution."""
        pass

    def solve(self, v, Akxk, xk):
        # Standarize data
        v = np.asarray(v)
        Akxk = np.asarray(Akxk)
        xk = np.asarray(xk)

        # Compute gradient
        w = self.grad_f(v)

        # Transform to the standardized subproblem
        self.subproblem_y = self.standize_subproblem(Akxk, w)

        self.solver.coef_ = xk.copy()

        # Solve subproblem
        self.solver.fit(self.Ak, self.subproblem_y, check_input=False)

        # Compute the new xk
        xk_new = self.recover_solution()

        # Compute delta
        delta_xk = xk_new - xk
        delta_Akxk = self.Ak @ delta_xk

        return delta_xk, delta_Akxk


class ElasticNet(CoCoASubproblemSolver):
    """
    Assume the original problem is
        min 1/2 * || Ax - y ||^2_2
            + n_samples * l1_ratio * lambda_ * || x ||_1
            + n_samples * (1 - l1_ratio) * lambda_ * 0.5 * || x ||_2^2

    Split the dataset by features.
    """

    def __init__(self, lambda_, l1_ratio, random_state):
        super(ElasticNet, self).__init__()
        self.lambda_ = lambda_
        self.l1_ratio = l1_ratio
        self.random_state = random_state

    def grad_f(self, v):
        v = np.asarray(v)
        return (v - self.y)

    def f(self, v):
        v = np.asarray(v)
        return np.linalg.norm(self.y - v) ** 2 / 2

    def gk(self, xk):
        xk = np.asarray(xk)
        L1 = np.linalg.norm(xk, 1) * len(self.y)
        L2 = 0.5 * np.linalg.norm(xk, 2) ** 2 * len(self.y)
        return self.lambda_ * (self.l1_ratio * L1 + (1 - self.l1_ratio) * L2)

    def f_conj(self, w):
        w = np.asarray(w)
        return np.linalg.norm(w, 2) ** 2 / 2 + w @ self.y

    def gk_conj(self, w):
        """
        Conjugate of Regularizer use Lemma 6 and Lemma 7 in 
            L1-Regularized Distributed Optimization- A Communication-Efficient Primal-Dual Framework
        """
        w = np.asarray(w)
        x = - w @ self.Ak
        if self.l1_ratio < 1.0:
            # Conjugate of ElasticNet (0 <= l1_ratio < 1)
            def conjugate(x):
                return np.sum(np.clip(np.abs(x) - self.l1_ratio, 0, np.inf) ** 2 / 2 / (1 - self.l1_ratio))
        else:
            # Lasso
            def conjugate(x):
                return self.B * np.sum(np.clip(np.abs(x) - 1, 0, np.inf))

        c = self.lambda_ * len(self.y)
        return c * conjugate(x / c)

    @property
    def solver_coef(self):
        return self.solver.coef_

    def dist_init(self, Ak, y, theta, local_iters, sigma):
        super(ElasticNet, self).dist_init(Ak, y, theta, local_iters, sigma)

        # This constant is used to compute conjugate of modified L1 norm
        self.B = self.f(np.zeros(len(self.y))) / self.lambda_

    def load_approximate_solver(self, sigma, local_iters, theta):
        """Load approximate solver to solve the standized problem."""
        self._tau = 1

        from fast_cd.solver import ElasticNetCoordSolver
        self.solver = ElasticNetCoordSolver(
            # lambda_=self.lambda_,
            lambda_=self.tau / sigma * self.lambda_,
            l1_ratio=self.l1_ratio, max_iter=local_iters,
            tol=theta, warm_start=True, random_state=self.random_state)

    def standize_subproblem(self, v, w):
        """Convert subproblem to a standard form so that local solver can solve."""
        import torch.distributed as dist
        return v - self.tau / self.sigma * w

    def recover_solution(self):
        """From the standardized solution to original solution."""
        return self.solver.coef_


class LinearSVM(CoCoASubproblemSolver):
    """
    Assume the original problem is
        min_w C/2 sum_i max{0, 1 - y_i*(w @ A_i)} + 0.5 || w ||_2^2
    where y in {-1, +1}.

    Consider the dual problem:
        min_x       C/2 ||Ax||_2^2 - <y, x>
        subject to  y_i x_i in [0, 1] for all i

    Note that we split the dataset by data points.
    """

    def __init__(self, C, random_state):
        super(LinearSVM, self).__init__()
        self.C = C
        self.random_state = random_state

    def grad_f(self, v):
        v = np.asarray(v)
        return self.C * v

    def f(self, v):
        v = np.asarray(v)
        return np.linalg.norm(v) ** 2 / 2 * self.C

    def gk(self, xk):
        xk = np.asarray(xk)
        # TODO: Add inf when xi*yi not in [0, 1]
        return - np.sum(xk * self.y)

    def f_conj(self, w):
        w = np.asarray(w)
        return np.linalg.norm(w, 2) ** 2 / (2 * self.C)

    def gk_conj(self, w):
        w = np.asarray(w)
        return np.sum(np.clip(1 - self.y * (w @ self.Ak), 0, np.inf))

    @property
    def solver_coef(self):
        return self.solver.dual_coef_

    def load_approximate_solver(self, sigma, local_iters, theta):
        """Load approximate solver to solve the standized problem."""
        self._tau = 1 / self.C
        self.solver = SVMCoordSolver(
            C=sigma / self.tau,
            max_iter=local_iters,
            tol=theta,
            warm_start=True,
            random_state=self.random_state)

    def standize_subproblem(self, v, w, Akxk):
        """Convert subproblem to a standard form so that local solver can solve."""
        return self.Ak.T @ (self.sigma / self.tau * Akxk - w)

    def recover_solution(self):
        """From the standardized solution to original solution."""
        return self.solver.dual_coef_

    def solve(self, v, Akxk, xk):
        # Standarize data
        v = np.asarray(v)
        Akxk = np.asarray(Akxk)
        xk = np.asarray(xk)

        # Compute gradient
        w = self.grad_f(v)

        # Transform to the standardized subproblem
        self.subproblem_y = self.standize_subproblem(Akxk, w, Akxk)

        self.solver.coef_ = xk.copy()

        # Solve subproblem
        self.solver.fit(self.Ak.T, self.y, self.subproblem_y)

        # Compute the new xk
        xk_new = self.recover_solution()

        # Compute delta
        delta_xk = xk_new - xk
        delta_Akxk = self.Ak @ delta_xk

        return delta_xk, delta_Akxk


def configure_solver(name, random_state, split_by, **params):
    if name == 'ElasticNet':
        assert split_by == 'features', 'This solver only works for splitting by features'
        solver = ElasticNet(
            lambda_=params['lambda_'], l1_ratio=params['l1_ratio'], random_state=random_state)
    elif name == 'LinearSVM':
        assert split_by == 'samples', 'This solver only works for splitting by samples'
        solver = LinearSVM(C=params['C'], random_state=random_state)
    else:
        raise NotImplementedError
    return solver
