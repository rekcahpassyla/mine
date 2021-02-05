# Cross validation script to find best d.
import pandas as pd
from functools import partial

import numpy as np
import loadsplit

import generic
import kernels

MultiKernelPerceptron = generic.MultiKernelPerceptron
gaussian = kernels.gaussian
polynomial = kernels.polynomial


train_results = []
test_results = []
confusions = []

Nsplits = 20
epochs = 20


store = pd.HDFStore('polynomial_crossvalidation.hdf5', 'a')

all_split_results = {}
all_ds = {}
all_ds_results = {}
# keys: split
# values: unravelled confusion matrix
all_confusions = {}
# keys: split
# values: dataframe of alphas
all_alphas = {}
seeds = np.arange(1, Nsplits+1)

for idx in range(Nsplits):
    seed = seeds[idx]
    np.random.seed(seed)
    print(f'========== Split {idx} ==========')
    trainset, testset, train_indexes, test_indexes = loadsplit.randomsample(loadsplit.data, 0.2)
    # Cross-validation: Perform 20 runs :
    # when using the 80% training data split from within to perform
    # 5-fold cross-validation to select the “best” parameter d∗
    # then retrain on full 80% training set using d∗
    # and then record the test errors on the remaining 20%.
    # Thus you will find 20 d∗ and 20 test errors.
    # Your final result will consist of a mean test error±std and a mean d∗ with std.
    traindf = pd.DataFrame(trainset)
    folds = loadsplit.split_folds(traindf, n=5)
    d_test_results = []
    ds = np.arange(1, 8, dtype=int)
    for d in ds:
        fold_test_results = []
        for f, (trainfold, testfold) in enumerate(folds):
            print(f"d={d}, fold={f}: Training for {epochs} epochs")
            trainfoldys, trainfoldxs = loadsplit.labelsvalues(trainfold.values)
            testfoldys, testfoldxs = loadsplit.labelsvalues(testfold.values)
            kf = partial(polynomial, d=d)
            mp = MultiKernelPerceptron(
                kf, trainfoldxs, trainfoldys, reflabels=list(range(0, 10)),
                parallel=10
            )
            mp.train(epochs)
            correct_test_fold, _ = mp.evaluate(testfoldxs, testfoldys)
            test_fold_err = 1 - correct_test_fold.mean()
            print(f"d={d}, fold={f}: Test err: {test_fold_err}")
            fold_test_results.append(test_fold_err)
        fold_test_results = np.array(fold_test_results)
        mean_fold_error = fold_test_results.mean()
        # now pick the d with the lowest error
        d_test_results.append(mean_fold_error)
    d_test_results = np.array(d_test_results)
    all_ds_results[idx] = d_test_results
    best_d = ds[np.argmin(d_test_results)]
    print("--------------------")
    print(f"Split {idx}: Best d: {best_d}")
    all_ds[idx] = best_d
    # now run the best d against the full 80% training set
    trainys, trainxs = loadsplit.labelsvalues(trainset)
    testys, testxs = loadsplit.labelsvalues(testset)
    kf = partial(polynomial, d=best_d)
    mp = MultiKernelPerceptron(
        kf, trainxs, trainys, reflabels=list(range(0, 10)), parallel=10
    )
    alpha = mp.train(epochs)
    alpha = pd.DataFrame(alpha, index=train_indexes)

    correct_test, confusion = mp.evaluate(testxs, testys)
    test_err = 1 - correct_test.mean()
    print(f"Split {idx}: Test err: {test_err}")
    all_split_results[idx] = test_err
    all_confusions[idx] = confusion.values.ravel()
    all_alphas[idx] = alpha.unstack()

all_split_results = pd.Series(all_split_results)
all_ds = pd.Series(all_ds)
all_ds_results = pd.DataFrame(all_ds_results)
all_alphas = pd.DataFrame(all_alphas)
all_confusions = pd.DataFrame(all_confusions)

store['all_confusions'] = all_confusions
store['all_split_results'] = all_split_results
store['all_ds'] = all_ds
store['all_ds_results'] = all_ds_results
store['all_alphas'] = all_alphas

store.flush()
store.close()
