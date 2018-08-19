cimport numpy as np
cimport cython

from libc.math cimport fabs
import numpy as np
from libc.stdio cimport printf
from cython cimport floating
from scipy.linalg.cython_blas cimport sdot, ddot
from scipy.linalg.cython_blas cimport saxpy, daxpy

ctypedef np.uint32_t UINT32_t
ctypedef floating(*DOT)(int * N, floating * X, int * incX, floating * Y, int * incY) nogil
ctypedef void(*AXPY)(int * N, floating * alpha, floating * X, int * incX, floating * Y, int * incY) nogil

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

cdef inline UINT32_t our_rand_r(UINT32_t * seed) nogil:
    seed[0] ^= <UINT32_t > (seed[0] << 13)
    seed[0] ^= <UINT32_t > (seed[0] >> 17)
    seed[0] ^= <UINT32_t > (seed[0] << 5)

    return seed[0] % (< UINT32_t > RAND_R_MAX + 1)

cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t * random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end

cdef inline floating fmax(floating x, floating y) nogil:
    return x if x > y else y

cdef inline floating fmin(floating x, floating y) nogil:
    return y if x > y else x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparse_cd(np.ndarray[floating, ndim=1] alpha,
              np.ndarray[floating, ndim=1] y,
              np.ndarray[floating, ndim=1] b,
              np.ndarray[floating, ndim=1, mode='c'] A_data,  # Matrix A should be stored in csc format
              np.ndarray[int, ndim=1, mode='c'] A_indices,
              np.ndarray[int, ndim=1, mode='c'] A_indptr,
              int n_rows,
              floating max_iter, floating C, floating tol, object rng):
    """
    Minimize
        D(alpha) := C/2 norm(A @ alpha, 2) ** 2 - <y + b, alpha>
    where (y * alpha) between 0 and 1.
    """
    cdef DOT dot
    cdef AXPY axpy
    if floating is float:
        dtype = np.float32
        dot = sdot
        axpy = saxpy
    else:
        dtype = np.float64
        dot = ddot
        axpy = daxpy

    cdef int n_cols = len(y)

    # compute norms of the columns of A
    cdef int ii
    cdef int jj
    cdef int n_iter = int(max_iter * n_cols)
    cdef int f_iter
    cdef floating tmp
    cdef floating delta
    cdef floating alpha_jj
    cdef floating alpha_max
    cdef floating d_alpha_jj
    cdef floating d_alpha_max
    cdef floating d_alpha_tol = tol
    cdef floating gap = tol + 1.0
    cdef floating[:] norm_cols_A

    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t * rand_r_state = &rand_r_state_seed

    cdef int startptr = A_indptr[0]
    cdef int endptr

    cdef floating normalize_sum

    norm_cols_A = np.zeros(n_cols, dtype=dtype)
    with nogil:
        for jj in range(n_cols):
            endptr = A_indptr[jj + 1]
            normalize_sum = 0.0
            for ii in range(startptr, endptr):
                normalize_sum += A_data[ii] ** 2
            norm_cols_A[jj] = normalize_sum
            startptr = endptr

    # V = A @ alpha
    cdef np.ndarray[floating, ndim= 1] V = np.zeros(n_rows, dtype=dtype)

    cdef floating * alpha_data = <floating*> alpha.data
    cdef floating * b_data = <floating*> b.data
    cdef floating * y_data = <floating*> y.data
    cdef floating * V_data = <floating*> V.data
    cdef floating * A_data_ = <floating*> A_data.data
    with nogil:
        # V = np.dot(A, alpha)
        startptr = A_indptr[0]
        for jj in range(n_cols):
            endptr = A_indptr[jj + 1]
            alpha_jj = alpha[jj]
            for ii in range(startptr, endptr):
                V[A_indices[ii]] += A_data[ii] * alpha_jj

        while n_iter > 0:
            alpha_max = 0.0
            d_alpha_max = 0.0
            for f_iter in range(min(n_cols, n_iter)):
                jj = rand_int(n_cols, rand_r_state)

                if norm_cols_A[jj] == 0.0:
                    continue

                alpha_jj = alpha[jj]  # Store previous value

                # tmp = (A[:,jj]*V).sum()
                tmp = 0.0
                for ii in range(A_indptr[jj], A_indptr[jj + 1]):
                    tmp += A_data[ii] * V[A_indices[ii]]

                delta = (y[jj] + b[jj] - C * tmp) / (C * norm_cols_A[jj])
                alpha[jj] = y[jj] * fmax(0, fmin(1, y[jj] * (alpha_jj + delta)))
                delta = alpha[jj] - alpha_jj

                if delta != 0.0:
                    # V +=  (alpha[jj] - alpha_jj) * A[:,jj]
                    for ii in range(A_indptr[jj], A_indptr[jj + 1]):
                        V[A_indices[ii]] += (alpha[jj] - alpha_jj) * A_data[ii]

                # update the maximum absolute coefficient update
                d_alpha_jj = fabs(alpha[jj] - alpha_jj)
                if d_alpha_jj > d_alpha_max:
                    d_alpha_max = d_alpha_jj

                if fabs(alpha[jj]) > alpha_max:
                    alpha_max = fabs(alpha[jj])

            if (alpha_max == 0.0 or d_alpha_max / alpha_max < d_alpha_tol or n_iter <= n_cols):
                # Compute gap
                gap = 0
                for jj in range(n_cols):
                    # dot(A[:, jj], V)
                    tmp = 0.0
                    for ii in range(A_indptr[jj], A_indptr[jj + 1]):
                        tmp += V[A_indices[ii]] * A_data[ii]
                    gap += tmp * alpha[jj] + fmax(0, 1 - y[jj] * (tmp - b[jj])) - (y[jj] + b[jj]) * alpha[jj]

                gap /= n_cols

                if gap < tol:
                    # return if we reached desired tolerance
                    break
            n_iter -= min(n_cols, n_iter)

    return alpha, gap, tol, (int(max_iter * n_cols) - n_iter) + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dense_cd(np.ndarray[floating, ndim=1] alpha,
             np.ndarray[floating, ndim=1] y, np.ndarray[floating, ndim=1] b,
             np.ndarray[floating, ndim=2, mode='fortran'] A,
             floating max_iter, floating C, floating tol, object rng):
    """
    Minimize
        D(alpha) := C/2 norm(A * alpha, 2) ** 2 - <y + b, alpha>
    where (y * alpha) between 0 and 1.
    """
    cdef DOT dot
    cdef AXPY axpy
    if floating is float:
        dtype = np.float32
        dot = sdot
        axpy = saxpy
    else:
        dtype = np.float64
        dot = ddot
        axpy = daxpy

    cdef int n_rows = A.shape[0]
    cdef int n_cols = A.shape[1]

    # Norm of columns of A
    cdef np.ndarray[floating, ndim= 1] norm_cols_A = (A**2).sum(axis=0)

    # V = A @ alpha
    cdef np.ndarray[floating, ndim= 1] V = np.empty(n_rows, dtype=dtype)

    # Declare variables
    cdef int i
    cdef int one = 1
    cdef int n_iter = int(max_iter * n_cols)
    cdef int ii
    cdef int f_iter
    cdef floating tmp
    cdef floating delta
    cdef floating alpha_ii
    cdef floating alpha_max
    cdef floating d_alpha_ii
    cdef floating d_alpha_max
    cdef floating d_alpha_tol = tol
    cdef floating gap = tol + 1.0
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t * rand_r_state = &rand_r_state_seed

    # Declare pointers
    cdef floating * A_data = <floating*> A.data
    cdef floating * alpha_data = <floating*> alpha.data
    cdef floating * b_data = <floating*> b.data
    cdef floating * y_data = <floating*> y.data
    cdef floating * V_data = <floating*> V.data

    with nogil:
        # V = np.dot(A, alpha)
        for i in range(n_rows):
            V[i] = dot( & n_cols, A_data + i, & n_rows, alpha_data, & one)

        while n_iter > 0:
            alpha_max = 0.0
            d_alpha_max = 0.0
            for f_iter in range(min(n_cols, n_iter)):
                ii = rand_int(n_cols, rand_r_state)

                if norm_cols_A[ii] == 0.0:
                    continue

                alpha_ii = alpha[ii]  # Store previous value

                # tmp = (A[:,ii]*V).sum()
                tmp = dot(& n_rows, A_data + ii * n_rows, & one, V_data, & one)

                delta = (y[ii] + b[ii] - C * tmp) / (C * norm_cols_A[ii])
                alpha[ii] = y[ii] * fmax(0, fmin(1, y[ii] * (alpha_ii + delta)))
                delta = alpha[ii] - alpha_ii

                if delta != 0.0:
                    # V +=  (alpha[ii] - alpha_ii) * A[:,ii]
                    axpy( & n_rows, & delta, A_data + ii * n_rows, & one, V_data, & one)

                # update the maximum absolute coefficient update
                d_alpha_ii = fabs(alpha[ii] - alpha_ii)
                if d_alpha_ii > d_alpha_max:
                    d_alpha_max = d_alpha_ii

                if fabs(alpha[ii]) > alpha_max:
                    alpha_max = fabs(alpha[ii])

            if (alpha_max == 0.0 or d_alpha_max / alpha_max < d_alpha_tol or n_iter <= n_cols):
                # Compute gap
                gap = 0
                for i in range(n_cols):
                    # dot(A[:, i], V)
                    tmp = dot(& n_rows, A_data + i * n_rows, & one, V_data, & one) * C
                    gap += tmp * alpha[i] + fmax(0, 1 - y[i] * (tmp - b[i])) - (y[i] + b[i]) * alpha[i]

                gap /= n_cols

                if gap < tol:
                    # return if we reached desired tolerance
                    break
            n_iter -= min(n_cols, n_iter)

    return alpha, gap, tol, (int(max_iter * n_cols) - n_iter) + 1
