import concurrent
from functools import partial

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

import loadsplit
from alcygos import onenn_float
from kernels import polynomial, histogram_kernel, localcorr_kernel
from percytron import gaussian_c, histogram


def histogram_kernel_(Xis, x, bins):
    nbins = len(bins)
    N, d = Xis.shape
    outXi = np.zeros((N, nbins), dtype=float)
    histogram(Xis, bins, outXi)

    Xjs = np.atleast_2d(x)
    N, d = Xjs.shape
    outXj = np.zeros((N, nbins), dtype=float)
    histogram(Xjs, bins, outXj)

    diffnorm = cdist(outXi, outXj, 'sqeuclidean')
    return diffnorm.squeeze()


def kpca(kf, xin, k=None):
    # kf: kernel function
    # xin: inputs
    # k: number of components to keep
    K = kf(xin, xin)
    N = K.shape[0]
    # centering
    IN = np.eye(N, dtype=K.dtype)
    H = IN - np.ones((N, N)) / N
    K = np.einsum('ij,jk,kl->il', H, K, H, optimize=True)
    e, v = eigh(K, b=IN, turbo=True, check_finite=False)
    # these are in descending order: reverse them
    e = e[::-1]
    # keep v in the columns
    v = v.T[::-1].T
    if k is None:
        return K, e, v
    else:
        alphas = e[:k]
        components = v[:, :k]
        trainx_proj = project(K.T, components, alphas)
        return trainx_proj, alphas, components


def project(K, components, e):
    # K: precomputed kernel matrix
    proj = np.einsum('ij,jk->ik', K, components, optimize=True)
    return proj / e


def kpca_1nn(kf, trainxs, trainys, testxs, testys,
             keep=range(15, 36), eval_train=True):
    # get the components
    K, e, v = kpca(kf, trainxs)
    Ktest = kf(trainxs, testxs)
    correct_train = {}
    correct_test = {}
    for k in keep:
        print(f"Evaluating {k} components")
        alphas = e[:k]
        components = v[:, :k]
        proj_train = project(K.T, components, alphas)
        proj_test = project(Ktest.T, components, alphas)
        # onenn the projections with the K
        Ntrain, d = trainxs.shape[0], k
        Ntest = testys.size
        if eval_train:
            working = np.zeros((1, Ntrain, Ntrain), dtype=float)
            ypred_train = np.zeros((1, Ntrain), dtype=float)
            onenn_float(ypred_train, working, proj_train,
                        np.array([proj_train]), np.atleast_2d(trainys),
                        1, Ntrain, d, Ntrain)
            ypred_train = ypred_train.squeeze()
            correct_train[k] = (ypred_train == trainys)
        else:
            correct_train = None

        working = np.zeros((1, Ntrain, Ntest), dtype=float)
        ypred = np.zeros((1, Ntest), dtype=float)
        onenn_float(ypred, working, proj_test,
                    np.array([proj_train]), np.atleast_2d(trainys),
                    1, Ntrain, d, Ntest)
        ypred = ypred.squeeze()
        correct_test[k] = (ypred == testys)
    correct_test = pd.DataFrame(correct_test)
    if correct_train is not None:
        correct_train = pd.DataFrame(correct_train)
    return correct_test, correct_train


def kpca_multi(args):
    trainxs, trainys, testxs, testys, param, ncomponents, eval_train = args
    kf = partial(localcorr_kernel, localdeg=param, deg=param)
    print("-------------------")
    print(f"Scheduling {ncomponents} components, param {param}")
    res = kpca_1nn(
        kf, trainxs, trainys, testxs, testys, ncomponents, eval_train
    )
    return param, res


results = {}
results_train = {}
Nsplits = 20

Ncomponents = list(range(70, 101))

store = pd.HDFStore('kpca_1nn_localcorr_70_101_deg_localdeg.hdf5', 'a')

Nparams = [2, 3, 4]

Nworkers = 3

for split in range(0, Nsplits):
    np.random.seed(split+1)
    results[split] = {}
    results_train[split] = {}
    train, test, _, _ = loadsplit.randomsample(loadsplit.data, 0.2)

    trainys = train[:, 0]
    trainxs = train[:, 1:]

    testys = test[:, 0]
    testxs = test[:, 1:]
    argslist = []
    for param in Nparams:
        argslist.append((
            trainxs, trainys, testxs, testys, param, Ncomponents, True
        ))

    if Nworkers:
        with concurrent.futures.ProcessPoolExecutor(max_workers=Nworkers) as executor:
            for (param, (correct, correct_train)) in executor.map(
                    kpca_multi, argslist):

                results[split][param] = correct.mean(axis=0)
                results_train[split][param] = correct_train.mean(axis=0)
                print(
                    f"split: {split}: components={Ncomponents}: "
                    f"training acc={correct_train.mean(axis=0)}, "
                    f"test acc={correct.mean(axis=0)}"
                )
    else:
        for args in argslist:
            (param, (correct, correct_train)) = kpca_multi(args)
            results[split][param] = correct.mean(axis=0)
            results_train[split][param] = correct_train.mean(axis=0)
            print(
                f"split: {split}: components={Ncomponents}: "
                f"training acc={correct_train.mean(axis=0)}, "
                f"test acc={correct.mean(axis=0)}"
            )

for split in results:
    res = results[split]
    res_train = results_train[split]
    results[split] = pd.DataFrame(res).unstack()
    results_train[split] = pd.DataFrame(res_train).unstack()
results = pd.DataFrame(results)
results_train = pd.DataFrame(results_train)
store['results'] = results
store['results_train'] = results_train

store.flush()

""""""



Nsplits = 20
seeds = np.arange(1, Nsplits+1)

all_split_results = {}
all_params = {}
all_params_results = {}

Nworkers = 4

for idx in range(Nsplits):
    seed = seeds[idx]
    np.random.seed(seed)
    print(f'========== Split {idx} ==========')
    trainset, testset, train_indexes, test_indexes = loadsplit.randomsample(loadsplit.data, 0.2)
    # Cross-validation: Perform 20 runs :
    # when using the 80% training data split from within to perform
    # 5-fold cross-validation to select the “best” parameter k∗
    # then retrain on full 80% training set using k∗
    # and then record the test errors on the remaining 20%.
    # Thus you will find 20 d∗ and 20 test errors.
    # Your final result will consist of a mean test error±std and a mean d∗ with std.
    traindf = pd.DataFrame(trainset)
    folds = loadsplit.split_folds(traindf, n=5)
    d_test_results = []
    for param in Nparams:
        fold_test_results = []

        argslist = []
        for f, (trainfold, testfold) in enumerate(folds):
            print(f"param={param}, fold={f}")
            trainfoldys, trainfoldxs = loadsplit.labelsvalues(trainfold.values)
            testfoldys, testfoldxs = loadsplit.labelsvalues(testfold.values)

            argslist.append(
                (trainfoldxs, trainfoldys, testfoldxs, testfoldys,
                 param, Ncomponents, False)
            )
        if Nworkers:
            with concurrent.futures.ProcessPoolExecutor(max_workers=Nworkers) as executor:
                for (param, (correct_test_fold, _)) in executor.map(
                        kpca_multi, argslist):
                    test_fold_err = 1 - correct_test_fold.mean(axis=0)
                    print(f"param={param}, fold={f}: Test err: {test_fold_err}")
                    fold_test_results.append(test_fold_err.values)
        else:
            for args in argslist:
                _, (correct_test_fold, _) = kpca_multi(args)
                test_fold_err = 1 - correct_test_fold.mean(axis=0)
                print(f"param={param}, fold={f}: Test err: {test_fold_err}")
                fold_test_results.append(test_fold_err.values)
        fold_test_results = np.array(fold_test_results)
        mean_fold_error = fold_test_results.mean(axis=0)
        # now pick the d with the lowest error
        d_test_results.append(mean_fold_error)
    d_test_results = pd.DataFrame(
        np.array(d_test_results), index=Nparams, columns=Ncomponents).unstack()
    all_params_results[idx] = d_test_results
    argmin = np.argmin(d_test_results)
    best_params = d_test_results.index[argmin]
    best_value = d_test_results.loc[best_params]
    print("--------------------")
    print(f"Split {idx}: Best params: {best_params} with value {best_value}")
    all_params[idx] = best_params
    # now run the best params against the full 80% training set
    k, d = best_params
    trainys, trainxs = loadsplit.labelsvalues(trainset)
    testys, testxs = loadsplit.labelsvalues(testset)

    _, (correct_test, correct_train) = kpca_multi(
        (trainxs, trainys, testxs, testys,
         d, [k], False)
    )
    test_err = 1 - correct_test.mean()
    print(f"Split {idx}: Test err: {test_err}")
    all_split_results[idx] = test_err
all_split_results = pd.Series(all_split_results)
all_params = pd.Series(all_params)
all_params_results = pd.DataFrame(all_params_results)

store['all_split_results'] = all_split_results
store['all_params'] = all_params
store['all_params_results'] = all_params_results

store.close()

