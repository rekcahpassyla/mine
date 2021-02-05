
cimport numpy as np
import numpy as np
cpdef float foo(np.ndarray[float] x):
    return np.sum(x)
import cython
from cython.parallel import prange
cimport libc.math
from scipy.spatial.distance import cdist

# for some reason without including this, cinit doesn't happen properly
TEST = "test"

# prediction for onenn
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void onenn(
        # nsims by m: for output
        np.ndarray[np.int8_t, ndim=2] ypred,
        # nsims by m by 2^n: for doing cdist of each sum
        # pre-allocate in Python because easier than messing with malloc
        np.ndarray[np.int8_t, ndim=3] working,
        # m by 2^n: same for each nsim
        np.ndarray[np.int8_t, ndim=2] testxs,
        # nsims by m by n:
        np.ndarray[np.int8_t, ndim=3] trainxs,
        # nsims by m
        np.ndarray[np.int8_t, ndim=2] trainys,
        int nsims,
        int m,   # number of input samples
        int n,    # dimension
        int t     # number of test items
):
    onenn_(ypred, working, testxs, trainxs, trainys, nsims, m, n, t)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void onenn_(
    char[:, :] ypred,
    char[:, :, :] working,
    char[:, :] testxs,
    char[:, :, :] trainxs,
    char[:, :] trainys,
    int nsims,
    int m,
    int n,
    int t
):
    cdef:
        int nsim = 0
        int yp
        int mini = -1
        Py_ssize_t i = 0
        Py_ssize_t j = 0
        Py_ssize_t k = 0
        Py_ssize_t testi = 0

    for nsim in prange(nsims, nogil=True):
        # first calculate cdist
        for i in prange(m):
            for j in prange(t):
                working[nsim][i][j] = 0
                for k in prange(n):
                    working[nsim][i][j] += (
                        #libc.math.fabs(trainxs[nsim][i][k] - testxs[nsim][j][k])
                        (trainxs[nsim][i][k] - testxs[j][k])**2
                    )
        # need argmin over each column of working[nsim]
        for j in prange(t):
            mini = -1
            for i in range(m):
                if mini == -1:
                    mini = i
                if working[nsim][i][j] < working[nsim][mini][j]:
                    mini = i
            ypred[nsim][j] = trainys[nsim][mini]



# prediction for onenn
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void onenn_float(
        # nsims by Ntest: for output
        np.ndarray[np.float64_t, ndim=2] ypred,
        # nsims by Ntrain by Ntest: for doing cdist of each sum
        # pre-allocate in Python because easier than messing with malloc
        np.ndarray[np.float64_t, ndim=3] working,
        # Ntrain by Ntest: same for each nsim
        np.ndarray[np.float64_t, ndim=2] testxs,
        # nsims by Ntrain by d:
        np.ndarray[np.float64_t, ndim=3] trainxs,
        # nsims by Ntrain
        np.ndarray[np.float64_t, ndim=2] trainys,
        int nsims,
        int m,   # number of input samples
        int n,    # dimension
        int t     # number of test items
):
    onenn_f(ypred, working, testxs, trainxs, trainys, nsims, m, n, t)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void onenn_f(
    double[:, :] ypred,
    double[:, :, :] working,
    double[:, :] testxs,
    double[:, :, :] trainxs,
    double[:, :] trainys,
    int nsims,
    int m,
    int n,
    int t
):
    cdef:
        int nsim = 0
        int yp
        int mini = -1
        Py_ssize_t i = 0
        Py_ssize_t j = 0
        Py_ssize_t k = 0
        Py_ssize_t testi = 0

    for nsim in prange(nsims, nogil=True):
        # first calculate cdist
        for i in prange(m):
            for j in prange(t):
                working[nsim][i][j] = 0
                for k in prange(n):
                    working[nsim][i][j] += (trainxs[nsim][i][k] - testxs[j][k])**2
                working[nsim][i][j] = libc.math.sqrt(working[nsim][i][j])
        # need argmin over each column of working[nsim]
        for j in prange(t):
            mini = -1
            for i in range(m):
                if mini == -1:
                    mini = i
                if working[nsim][i][j] < working[nsim][mini][j]:
                    mini = i
            ypred[nsim][j] = trainys[nsim][mini]

