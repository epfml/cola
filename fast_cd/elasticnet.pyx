cimport numpy as np
cimport cython

from libc.math cimport fabs
import numpy as np
import warnings

import time

from libc.stdio cimport printf
from cython cimport floating
from scipy.linalg.cython_blas cimport sdot, ddot
from scipy.linalg.cython_blas cimport saxpy, daxpy
from scipy.linalg.cython_blas cimport sasum, dasum

ctypedef np.uint32_t UINT32_t
ctypedef floating(*DOT)(int * N, floating * X, int * incX, floating * Y, int * incY) nogil
ctypedef void(*AXPY)(int * N, floating * alpha, floating * X, int * incX, floating * Y, int * incY) nogil
ctypedef floating(*ASUM)(int * N, floating * X, int * incX) nogil

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

cdef inline UINT32_t our_rand_r(UINT32_t * seed) nogil:
    seed[0] ^= <UINT32_t > (seed[0] << 13)
    seed[0] ^= <UINT32_t > (seed[0] >> 17)
    seed[0] ^= <UINT32_t > (seed[0] << 5)

    return seed[0] % ( < UINT32_t > RAND_R_MAX + 1)

cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t * random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end

cdef inline floating fmax(floating x, floating y) nogil:
    return x if x > y else y

cdef inline floating fmin(floating x, floating y) nogil:
    return y if x > y else x

cdef inline floating fsign(floating f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0

cdef floating abs_max(int n, floating * a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef floating m = fabs(a[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dense_cd(np.ndarray[floating, ndim=1, mode='c'] w,
             floating alpha, floating beta,
             np.ndarray[floating, ndim=2, mode='fortran'] X,
             np.ndarray[floating, ndim=1, mode='c'] y,
             floating max_iter, floating tol,
             object rng):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression
        We minimize
        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) norm(w, 2)^2
    """

    # fused types version of BLAS functions
    cdef DOT dot
    cdef AXPY axpy
    cdef ASUM asum

    if floating is float:
        dtype = np.float32
        dot = sdot
        axpy = saxpy
        asum = sasum
    else:
        dtype = np.float64
        dot = ddot
        axpy = daxpy
        asum = dasum

    # get the data information into easy vars
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]

    # compute norms of the columns of X
    cdef np.ndarray[floating, ndim = 1] norm_cols_X = (X**2).sum(axis=0)

    # initial value of the residuals
    cdef np.ndarray[floating, ndim = 1] R = np.zeros(n_samples, dtype=dtype)
    cdef np.ndarray[floating, ndim = 1] XtA = np.zeros(n_features, dtype=dtype)

    cdef floating tmp
    cdef floating w_ii
    cdef floating mw_ii
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating dual_norm_XtA
    cdef floating R_norm2
    cdef floating w_norm2
    cdef floating l1_norm
    cdef floating const
    cdef floating A_norm2
    cdef int ii
    cdef int i
    cdef int n_iter = int(max_iter * n_features)
    cdef int f_iter
    cdef int one = 1
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t * rand_r_state = &rand_r_state_seed

    cdef floating * X_data = <floating*> X.data
    cdef floating * y_data = <floating*> y.data
    cdef floating * w_data = <floating*> w.data
    cdef floating * R_data = <floating*> R.data
    cdef floating * XtA_data = <floating*> XtA.data

    if alpha == 0 and beta == 0:
        warnings.warn("Coordinate descent with no regularization may lead to unexpected"
                      " results and is discouraged.")

    cdef floating t = time.time()
    with nogil:
        # R = y - np.dot(X, w)
        for i in range(n_samples):
            R[i] = y[i] - dot(& n_features, X_data + i, & n_samples, w_data, & one)

        while n_iter > 0:
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(min(n_features, n_iter)):
                ii = rand_int(n_features, rand_r_state)

                if norm_cols_X[ii] == 0.0:
                    continue

                w_ii = w[ii]  # Store previous value

                if w_ii != 0.0:
                    # R += w_ii * X[:,ii]
                    axpy(& n_samples, & w_ii, X_data + ii * n_samples, & one,
                          R_data, & one)

                # tmp = (X[:,ii]*R).sum()
                tmp = dot( & n_samples, X_data + ii * n_samples, & one, R_data, & one)

                w[ii] = (fsign(tmp) * fmax(fabs(tmp) - alpha, 0)
                         / (norm_cols_X[ii] + beta))

                if w[ii] != 0.0:
                    # R -=  w[ii] * X[:,ii] # Update residual
                    mw_ii = -w[ii]
                    axpy(& n_samples, & mw_ii, X_data + ii * n_samples, & one,
                          R_data, & one)

                # update the maximum absolute coefficient update
                d_w_ii = fabs(w[ii] - w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                if fabs(w[ii]) > w_max:
                    w_max = fabs(w[ii])

            if (w_max == 0.0 or
                d_w_max / w_max < d_w_tol or
                    n_iter <= n_features):
                # the biggest coordinate update of this iteration was smaller
                # than the tolerance: check the duality gap as ultimate
                # stopping criterion

                # XtA = np.dot(X.T, R) - beta * w
                for i in range(n_features):
                    XtA[i] = dot( & n_samples, X_data + i * n_samples,
                                 & one, R_data, & one) - beta * w[i]

                dual_norm_XtA = abs_max(n_features, XtA_data)

                # R_norm2 = np.dot(R, R)
                R_norm2 = dot( & n_samples, R_data, & one, R_data, & one)

                # w_norm2 = np.dot(w, w)
                w_norm2 = dot( & n_features, w_data, & one, w_data, & one)

                if (dual_norm_XtA > alpha):
                    const = alpha / dual_norm_XtA
                    A_norm2 = R_norm2 * (const ** 2)
                    gap = 0.5 * (R_norm2 + A_norm2)
                else:
                    const = 1.0
                    gap = R_norm2

                l1_norm = asum( & n_features, w_data, & one)

                # np.dot(R.T, y)
                gap += (alpha * l1_norm
                        - const * dot( & n_samples, R_data, & one, y_data, & one)
                        + 0.5 * beta * (1 + const ** 2) * (w_norm2))

                gap /= n_samples
                if gap < tol:
                    # return if we reached desired tolerance
                    break

            n_iter -= min(n_features, n_iter)

    return w, gap, tol, (int(max_iter * n_features) - n_iter) + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparse_cd(floating[:] w,
              floating alpha, floating beta,
              np.ndarray[floating, ndim=1, mode='c'] X_data,
              np.ndarray[int, ndim=1, mode='c'] X_indices,
              np.ndarray[int, ndim=1, mode='c'] X_indptr,
              np.ndarray[floating, ndim=1] y,
              floating max_iter, floating tol, object rng):
    """Cython version of the coordinate descent algorithm for Elastic-Net
    We minimize:
        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2

    Note that we assume the X_data is csc_matrix which means the 2D-matrix is stacked by
    its columns.
    """

    # get the data information into easy vars
    cdef int n_samples = y.shape[0]
    cdef int n_features = w.shape[0]

    # compute norms of the columns of X
    cdef int ii
    cdef int one = 1
    cdef floating[:] norm_cols_X

    cdef int startptr = X_indptr[0]
    cdef int endptr

    # initial value of the residuals
    cdef floating[:] R = y.copy()

    cdef floating[:] X_T_R
    cdef floating[:] XtA

    # fused types version of BLAS functions
    cdef DOT dot
    cdef ASUM asum

    if floating is float:
        dtype = np.float32
        dot = sdot
        asum = sasum
    else:
        dtype = np.float64
        dot = ddot
        asum = dasum

    norm_cols_X = np.zeros(n_features, dtype=dtype)
    X_T_R = np.zeros(n_features, dtype=dtype)
    XtA = np.zeros(n_features, dtype=dtype)

    cdef floating tmp
    cdef floating w_ii
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating R_sum = 0.0
    cdef floating R_norm2
    cdef floating w_norm2
    cdef floating A_norm2
    cdef floating l1_norm
    cdef floating normalize_sum
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating dual_norm_XtA
    cdef int jj
    cdef int n_iter = int(max_iter * n_features)
    cdef int f_iter
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t * rand_r_state = &rand_r_state_seed

    for ii in range(n_features):
        endptr = X_indptr[ii + 1]
        normalize_sum = 0.0
        w_ii = w[ii]

        for jj in range(startptr, endptr):
            normalize_sum += X_data[jj] ** 2
            R[X_indices[jj]] -= X_data[jj] * w_ii
        norm_cols_X[ii] = normalize_sum

        startptr = endptr

    with nogil:
        while n_iter > 0:
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(min(n_features, n_iter)):
                ii = rand_int(n_features, rand_r_state)
                if norm_cols_X[ii] == 0.0:
                    continue

                startptr = X_indptr[ii]
                endptr = X_indptr[ii + 1]
                w_ii = w[ii]  # Store previous value

                if w_ii != 0.0:
                    # R += w_ii * X[:,ii]
                    for jj in range(startptr, endptr):
                        R[X_indices[jj]] += X_data[jj] * w_ii

                # tmp = (X[:,ii] * R).sum()
                tmp = 0.0
                for jj in range(startptr, endptr):
                    tmp += R[X_indices[jj]] * X_data[jj]

                w[ii] = fsign(tmp) * fmax(fabs(tmp) - alpha, 0) \
                    / (norm_cols_X[ii] + beta)

                if w[ii] != 0.0:
                    # R -=  w[ii] * X[:,ii] # Update residual
                    for jj in range(startptr, endptr):
                        R[X_indices[jj]] -= X_data[jj] * w[ii]

                # update the maximum absolute coefficient update
                d_w_ii = fabs(w[ii] - w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                if fabs(w[ii]) > w_max:
                    w_max = fabs(w[ii])

            if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter <= n_features:
                # the biggest coordinate update of this iteration was smaller than
                # the tolerance: check the duality gap as ultimate stopping
                # criterion

                for ii in range(n_features):
                    X_T_R[ii] = 0.0
                    for jj in range(X_indptr[ii], X_indptr[ii + 1]):
                        X_T_R[ii] += X_data[jj] * R[X_indices[jj]]
                    XtA[ii] = X_T_R[ii] - beta * w[ii]

                dual_norm_XtA = abs_max(n_features, & XtA[0])

                # R_norm2 = np.dot(R, R)
                R_norm2 = dot(& n_samples, & R[0], & one, & R[0], & one)

                # w_norm2 = np.dot(w, w)
                w_norm2 = dot(& n_features, & w[0], & one, & w[0], & one)
                if (dual_norm_XtA > alpha):
                    const = alpha / dual_norm_XtA
                    A_norm2 = R_norm2 * const**2
                    gap = 0.5 * (R_norm2 + A_norm2)
                else:
                    const = 1.0
                    gap = R_norm2

                l1_norm = asum(& n_features, & w[0],  & one)

                gap += (alpha * l1_norm - const * dot(
                    & n_samples,
                    & R[0], & one,
                    & y[0], & one
                )
                    + 0.5 * beta * (1 + const ** 2) * w_norm2)

                gap /= n_samples
                if gap < tol:
                    # return if we reached desired tolerance
                    break

            n_iter -= min(n_features, n_iter)
    return w, gap, tol, (int(max_iter * n_features) - n_iter) + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def admm_dense_cd(np.ndarray[floating, ndim=1] w,
                  floating alpha, floating beta,
                  np.ndarray[floating, ndim=2, mode='fortran'] X,
                  np.ndarray[floating, ndim=1, mode='c'] y,
                  np.ndarray[floating, ndim=1, mode='c'] c,
                  float max_iter, floating tol,
                  object rng):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression
        We minimize
        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) norm(w - c, 2)^2
            + <c, w>
    """

    # fused types version of BLAS functions
    cdef DOT dot
    cdef AXPY axpy
    cdef ASUM asum

    if floating is float:
        dtype = np.float32
        dot = sdot
        axpy = saxpy
        asum = sasum
    else:
        dtype = np.float64
        dot = ddot
        axpy = daxpy
        asum = dasum

    # get the data information into easy vars
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]

    # get the number of tasks indirectly, using strides
    cdef int n_tasks = y.strides[0] / sizeof(floating)

    # compute norms of the columns of X
    cdef np.ndarray[floating, ndim = 1] norm_cols_X = (X**2).sum(axis=0)

    # initial value of the residuals
    cdef np.ndarray[floating, ndim = 1] R = np.empty(n_samples, dtype=dtype)
    cdef np.ndarray[floating, ndim = 1] XtA = np.empty(n_features, dtype=dtype)

    cdef floating tmp
    cdef floating w_ii
    cdef floating mw_ii
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating dual_norm_XtA
    cdef floating R_norm2
    cdef floating w_norm2
    cdef floating l1_norm
    cdef floating const
    cdef floating A_norm2
    cdef int ii
    cdef int i
    cdef int n_iter = int(max_iter * n_features)
    cdef int f_iter
    cdef int one = 1

    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t * rand_r_state = &rand_r_state_seed

    cdef floating * X_data = <floating*> X.data
    cdef floating * y_data = <floating*> y.data
    cdef floating * w_data = <floating*> w.data
    cdef floating * R_data = <floating*> R.data
    cdef floating * XtA_data = <floating*> XtA.data

    if alpha == 0 and beta == 0:
        warnings.warn("Coordinate descent with no regularization may lead to unexpected"
                      " results and is discouraged.")

    with nogil:
        # R = y - np.dot(X, w)
        for i in range(n_samples):
            R[i] = y[i] - dot(& n_features, X_data + i, & n_samples, w_data, & one)

        # tol *= np.dot(y, y)
        tol *= dot(& n_samples, y_data, & n_tasks, y_data, & n_tasks)

        while n_iter > 0:
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(min(n_features, n_iter)):
                ii = rand_int(n_features, rand_r_state)

                if norm_cols_X[ii] == 0.0:
                    continue

                w_ii = w[ii]  # Store previous value

                if w_ii != 0.0:
                    # R += w_ii * X[:,ii]
                    axpy(& n_samples, & w_ii, X_data + ii * n_samples,  & one,
                          R_data, & one)

                # tmp = (X[:,ii]*R).sum()
                tmp = dot(& n_samples, X_data + ii * n_samples, & one, R_data, & one)

                # tmp += beta * c[ii]
                tmp -= c[ii]

                w[ii] = (fsign(tmp) * fmax(fabs(tmp) - alpha, 0)
                         / (norm_cols_X[ii] + beta))

                if w[ii] != 0.0:
                    # R -=  w[ii] * X[:,ii] # Update residual
                    mw_ii = -w[ii]
                    axpy( & n_samples, & mw_ii, X_data + ii * n_samples, & one,
                         R_data, & one)

                # update the maximum absolute coefficient update
                d_w_ii = fabs(w[ii] - w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                if fabs(w[ii]) > w_max:
                    w_max = fabs(w[ii])

            if (w_max == 0.0 or
                d_w_max / w_max < d_w_tol or
                    n_iter == max_iter - 1):
                # the biggest coordinate update of this iteration was smaller
                # than the tolerance: check the duality gap as ultimate
                # stopping criterion

                # XtA = np.dot(X.T, R) - beta * w
                for i in range(n_features):
                    XtA[i] = dot(& n_samples, X_data + i * n_samples,
                                  & one, R_data, & one) - beta * w[i]

                dual_norm_XtA = abs_max(n_features, XtA_data)

                # R_norm2 = np.dot(R, R)
                R_norm2 = dot( & n_samples, R_data, & one, R_data, & one)

                # w_norm2 = np.dot(w, w)
                w_norm2 = dot( & n_features, w_data, & one, w_data, & one)

                if (dual_norm_XtA > alpha):
                    const = alpha / dual_norm_XtA
                    A_norm2 = R_norm2 * (const ** 2)
                    gap = 0.5 * (R_norm2 + A_norm2)
                else:
                    const = 1.0
                    gap = R_norm2

                l1_norm = asum(& n_features, w_data, & one)

                # np.dot(R.T, y)
                gap += (alpha * l1_norm
                        - const * dot(& n_samples, R_data, & one, y_data, & n_tasks)
                        + 0.5 * beta * (1 + const ** 2) * (w_norm2))

                gap /= n_samples
                if gap < tol:
                    # return if we reached desired tolerance
                    break
            n_iter -= min(n_features, n_iter)
    return w, gap, tol, (int(max_iter * n_features) - n_iter) + 1
