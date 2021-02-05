import numpy as np
from scipy.spatial.distance import cdist
import percytron


def gaussian_(Xis, Xjs, sigma):
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
    # The below shows the unvectorised calculation
    # ||x_i - x_j||
    #[40]: out = []
    #...: for xi in a:
    #...:row = []
    #...:for xj in a:
    #...:    xdiff = xi-xj
    #...:    sqnorm = np.sum(xdiff*xdiff))
    #...:    row.append(sqnorm)
    #...:    out.append(row)
    #...: out = np.array(out)
    # vectorise x_i - x_j
    diff = Xis[:, None] - Xjs[None, :]
    # redundant, should just take sum of squares rather than norm then square
    diffnorm = np.square(np.linalg.norm(diff, axis=2))
    gaussian_kernel = np.exp(-sigma2*diffnorm)
    return gaussian_kernel


def gaussian(Xis, x, sigma):
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


def gaussian_c(Xis, x, c):
    # expressed in different form
    Xjs = np.atleast_2d(x)
    diffnorm = cdist(Xis, Xjs, 'sqeuclidean')
    gaussian_kernel = np.exp(-c*diffnorm)
    return gaussian_kernel.squeeze()


def polynomial(Xis, Xjs, d, c=0):
    # Xis: float[N, d]
    # The LHS of the kernel functions. Each row represents one data point
    # Xjs: float[M, d]
    # The RHS of the kernel functions. Each row represents one data point
    # sigma: float
    # The parameter controlling the kernel width
    # Returns: float[N, M]
    # kernel matrix for polynomial kernel to power d
    # Call this with Xjs = Xis to get the Gram matrix
    # Call this with Xis = training values, Xjs = test values
    # to get the evaluation matrix for regression
    Xjs = np.atleast_2d(Xjs)
    out = np.einsum('ij,jk->ik', Xis, Xjs.T)
    out = np.power(out + c, d)
    return out


def solve_alpha(K, y):
    # K: float[n, n]
    #   The pre-calculated kernel matrix
    # y: float[n]
    #   The observed outputs
    mod_gram_inv = np.linalg.inv(K)
    # each row of first argument times each element of y
    # then sum over that row
    alpha = np.einsum('ij,j->i', mod_gram_inv, y)
    return alpha

# c to sigma
def c2s(c):
    return np.sqrt(1/2*c)


def makeblocks(inputs, side, blocksize):
    subs = np.linspace(0, side, num=side//blocksize+1, dtype=int)
    splits = []
    inputs = inputs.reshape(-1, side, side)
    s = blocksize**2
    # split according to blocksize, then flatten the blocks
    for starti, stopi in zip(subs[:-1], subs[1:]):
        for startj, stopj in zip(subs[:-1], subs[1:]):
            splits.append(inputs[:, starti:stopi, startj:stopj].reshape(-1, s))
    return splits



def localcorr_kernel(Xis, x, localdeg=2, deg=2):
    # take dot product at different levels
    Xjs = np.atleast_2d(x)
    side = 16
    n = np.array([3, 2])
    blocksizes = np.power(2, n)
    levels = []
    NXis = Xis.shape[0]
    NXjs = Xjs.shape[0]
    for blocksize in blocksizes:
        splitsxi = makeblocks(Xis, side, blocksize)
        splitsxj = makeblocks(Xjs, side, blocksize)

        results = np.zeros((NXis, NXjs), dtype=float)
        for spliti, splitj in zip(splitsxi, splitsxj):
            # polynomial kernel at localdeg
            out = polynomial(spliti, splitj, d=localdeg)
            results += out
        results /= len(splitsxi)
        if deg > 1:
            results = np.power(results, deg)
        levels.append(results)
    levels = np.array(levels).mean(axis=0)
    return levels



def subhists(inputs, side, blocksize, nbins=0, localdeg=1, deg=1):
    splits = makeblocks(inputs, side, blocksize)
    # adaptive bins, twice the number of the block size
    # if nbins = 0 then use adaptive
    if nbins == 0:
        nbins = blocksize*2
    bins = np.linspace(-1, 1, num=nbins)
    bins = bins[1:]
    binwidth = bins[1] - bins[0]

    results = []
    for split in splits:
        out = np.zeros((inputs.shape[0], len(bins)))
        percytron.histogram(split, bins, out)
        factor = binwidth * out[0]
        out /= factor.sum()
        if localdeg > 1:
            out = np.power(out, localdeg)
        results.append(out)
    if deg > 1:
        results = np.power(results, deg)
    return results


def histogram_kernel(Xis, x, nbins=0):
    Xjs = np.atleast_2d(x)
    side = 16
    n = np.array([4, 3, 2, 1])
    blocksizes = np.power(2, n)
    levels = []
    NXis = Xis.shape[0]
    NXjs = Xjs.shape[0]
    for blocksize in blocksizes:
        subxi = subhists(Xis, side=side, blocksize=blocksize,
                         nbins=nbins, localdeg=1, deg=1)
        subxj = subhists(Xjs, side=side, blocksize=blocksize,
                         nbins=nbins, localdeg=1, deg=1)
        # take minimum and sum
        out = np.zeros((NXis, NXjs), dtype=float)
        # use the same output array for all sub-histograms
        for subi, subj in zip(subxi, subxj):
            percytron.histdiff(subi, subj, out)
        levels.append(out)
    levels = np.array(levels)
    # now we have to process and weight the levels
    # last level is weighted the highest
    result = levels / blocksizes[:, None, None]
    out = np.einsum('ij,jk->ik', Xis, Xjs.T)
    result += 0.5 * out
    result = result.sum(axis=0)
    result /= result.max()
    return result
