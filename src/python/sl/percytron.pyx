cimport numpy as np
import numpy as np
cpdef float foo(np.ndarray[float] x):
    return np.sum(x)
import cython
from cython.parallel import prange
from scipy.spatial.distance import cdist
cimport libc.math


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void train(
    np.ndarray[np.float64_t, ndim=1] y_,
    np.ndarray[np.float64_t, ndim=1] predictions,
    np.ndarray[np.float64_t, ndim=1] alpha,
    np.ndarray[np.float64_t, ndim=2] K,
    int N,
    int epochs
):
    train_(y_, predictions, alpha, K, N, epochs)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void train_(
    double[:] y_,
    double[:] predictions,
    double[:] alpha,
    double[:, :] K,
    int N,
    int epochs
):
    cdef:
        int i = 0
        Py_ssize_t idx
        bint mistake
        int ypred
        Py_ssize_t j = 0
    while i < epochs:
        for idx in prange(N, nogil=True):
            ypred = 1 if predictions[idx] > 0 else -1
            mistake = (ypred != y_[idx])
            if mistake:
                alpha[idx] += mistake
                for j in range(N):
                    predictions[j] += y_[idx] * K[idx][j]
        i += 1


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void train_all(
        np.ndarray[np.float64_t, ndim=2] y_,
        np.ndarray[np.float64_t, ndim=2] predictions,
        np.ndarray[np.float64_t, ndim=2] alpha,
        np.ndarray[np.float64_t, ndim=2] K,
        int N,
        int d,
        int epochs
):
    train_all_(y_, predictions, alpha, K, N, d, epochs)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void train_all_(
    # columns: each perceptron
    double[:, :] y_,
    double[:, :] predictions,
    double[:, :] alpha,
    double[:, :] K,
    int N,
    int d,
    int epochs
):
    cdef:
        int counter = 0
        int i = 0
        int j = 0
        int dim = 0
        bint mistake
        int ypred

    while counter < epochs:
        for dim in prange(d, nogil=True):
            for i in prange(N):
                ypred = 1 if predictions[i][dim] > 0 else -1
                mistake = (ypred != y_[i][dim])
                if mistake:
                    alpha[i][dim] += mistake
                    for j in range(N):
                        predictions[j][dim] += y_[i][dim] * K[i][j]
        counter += 1

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void eval_all(
        np.ndarray[np.float64_t, ndim=2] yraw,
        np.ndarray[np.float64_t, ndim=2] y_,
        np.ndarray[np.float64_t, ndim=2] alpha,
        np.ndarray[np.float64_t, ndim=2] K,
        int N,
        int d,
        int ntest
):
    eval_all_(yraw, y_, alpha, K, N, d, ntest)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void eval_all_(
    # columns: each perceptron
    double[:, :] yraw,    # for output
    double[:, :] y_,
    double[:, :] alpha,
    double[:, :] K,
    int N,    # number of inputs
    int d,    # number of perceptrons
    int ntest # number of test points
):
    cdef:
        int i = 0
        int j = 0
        int dim = 0

    for dim in prange(d, nogil=True):
        for i in prange(N):
            for j in range(ntest):
                yraw[j][dim] += alpha[i][dim] * y_[i][dim] * K[i][j]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void histogram(
        np.ndarray[np.float64_t, ndim=2] inputs,
        np.ndarray[np.float64_t, ndim=1] binvalues,
        np.ndarray[np.float64_t, ndim=2] out
):
    # inputs: data points, in rows
    # binvalues: right boundaries of bins
    N = inputs.shape[0]
    d = inputs.shape[1]
    nbins = len(binvalues)
    histogram_(inputs, binvalues, out, N, d, nbins)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void histogram_(
    double[:, :] inputs,
    double[:] binvalues,
    double[:, :] out,
    int N,    # number of inputs
    int d,    # dimension
    int nbins # number of bins
):
    cdef:
        int i = 0
        int j = 0
        int k = 0
        int dim = 0
        float item
    for i in prange(N, nogil=True):
        for j in range(d):
            item = inputs[i][j]
            for k in range(nbins):
                if item <= binvalues[k]:
                    out[i][k] += 1
                    break



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void histdiff(
        np.ndarray[np.float64_t, ndim=2] Xis,
        np.ndarray[np.float64_t, ndim=2] Xjs,
        np.ndarray[np.float64_t, ndim=2] out
):
    NXis = Xis.shape[0]
    NXjs = Xjs.shape[0]
    nbins = Xis.shape[1]
    histdiff_(Xis, Xjs, out, NXis, NXjs, nbins)


# Given 2 arrays of histograms, calculate the difference
# defined as the sum of (minimum value for each corresponding bin)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void histdiff_(
    double[:, :] Xis, # Nxis by Nbins
    double[:, :] Xjs, # Nxjs by Nbins
    double[:, :] out, # Nxis by Nxjs
    int NXis,
    int NXjs,
    int nbins # number of bins
):
    cdef:
        int i = 0
        int j = 0
        int k = 0

    for i in prange(NXis, nogil=True):
        for j in prange(NXjs):
            for k in range(nbins):
                out[i][j] += libc.math.fmin(Xis[i][k], Xjs[j][k])


# Given 2 arrays of histograms, calculate the difference
# defined as the sum of (minimum value for each corresponding bin)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void min_(
    double[:, :] Xis, # Nxis by Nbins
    double[:, :] Xjs, # Nxjs by Nbins
    double[:, :] out, # Nxis by Nxjs
    int NXis,
    int NXjs,
    int dim
):
    cdef:
        int i = 0
        int j = 0
        int k = 0
    for i in prange(NXis, nogil=True):
        for j in prange(NXjs):
            for k in range(dim):
                out[i][j] = libc.math.fmin(Xis[i][k], Xjs[j][k])


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gaussian(Xis, x, sigma):
    # Xis: float[N, d]
    # The LHS of the kernel functions. Each row represents one data point
    # Xjs: float[M, d]
    # The RHS of the kernel functions. Each row represents one data point
    # sigma: float
    # The parameter controlling the kernel width
    # Returns: float[N, M]
    # kernel matrix for gaussian kernel with parameter sigma
    # of the Xis, Xjs.
    # Call this with Xjs = Xis to get the Gram matrix
    # Call this with Xis = training values, Xjs = test values
    # to get the evaluation matrix for regression
    sigma2 = 2*sigma**2
    Xjs = np.atleast_2d(x)
    diffnorm = cdist(Xis, Xjs, 'sqeuclidean')
    gaussian_kernel = np.exp(-diffnorm/sigma2)
    return gaussian_kernel.squeeze()


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gaussian_c(Xis, x, c):
    # Xis: float[N, d]
    # The LHS of the kernel functions. Each row represents one data point
    # Xjs: float[M, d]
    # The RHS of the kernel functions. Each row represents one data point
    # sigma: float
    # The parameter controlling the kernel width
    # Returns: float[N, M]
    # kernel matrix for gaussian kernel with parameter sigma
    # of the Xis, Xjs.
    # Call this with Xjs = Xis to get the Gram matrix
    # Call this with Xis = training values, Xjs = test values
    # to get the evaluation matrix for regression
    Xjs = np.atleast_2d(x)
    diffnorm = cdist(Xis, Xjs, 'sqeuclidean')
    gaussian_kernel = np.exp(-c*diffnorm)
    return gaussian_kernel.squeeze()


