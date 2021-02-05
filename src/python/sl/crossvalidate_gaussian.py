# Cross validation script to find best d.
import pandas as pd
from functools import partial

import numpy as np
import loadsplit

import generic
import kernels
import percytron

MultiKernelPerceptron = generic.MultiKernelPerceptron
gaussian = kernels.gaussian_c
polynomial = kernels.polynomial


train_results = []
test_results = []
confusions = []

Nsplits = 20
epochs = 10

store = pd.HDFStore('gaussian_crossvalidation_final_fine.hdf5', 'a')

all_split_results = {}
all_cs = {}
all_cs_results = {}
# keys: split
# values: unravelled confusion matrix
all_confusions = {}

cs = [0.0014, 0.0015, 0.0016,  0.017, 0.018, 0.019, 0.02, 0.021, 0.022]
for idx in range(Nsplits):
    print(f'========== Split {idx} ==========')
    trainset, testset = loadsplit.randomsample(loadsplit.data, 0.2)
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
    for c in cs:
        fold_test_results = []
        for f, (trainfold, testfold) in enumerate(folds):
            print(f"c={c}, fold={f}: Training for {epochs} epochs")
            trainfoldys, trainfoldxs = loadsplit.labelsvalues(trainfold.values)
            testfoldys, testfoldxs = loadsplit.labelsvalues(testfold.values)
            kf = partial(gaussian, c=c)
            mp = MultiKernelPerceptron(
                kf, trainfoldxs, trainfoldys, reflabels=list(range(0, 10)), parallel=10
            )
            mp.train(epochs)
            correct_test_fold, _ = mp.evaluate(testfoldxs, testfoldys)
            test_fold_err = 1 - correct_test_fold.mean()
            print(f"c={c}, fold={f}: Test err: {test_fold_err}")
            fold_test_results.append(test_fold_err)
        fold_test_results = np.array(fold_test_results)
        mean_fold_error = fold_test_results.mean()
        # now pick the d with the lowest error
        d_test_results.append(mean_fold_error)
    d_test_results = np.array(d_test_results)
    all_cs_results[idx] = d_test_results
    best_c = cs[np.argmin(d_test_results)]
    print("--------------------")
    print(f"Split {idx}: Best c: {best_c}")
    all_cs[idx] = best_c
    # now run the best c against the full 80% training set
    trainys, trainxs = loadsplit.labelsvalues(trainset)
    testys, testxs = loadsplit.labelsvalues(testset)
    kf = partial(gaussian, c=best_c)
    mp = MultiKernelPerceptron(
        kf, trainxs, trainys, reflabels=list(range(0, 10)), parallel=10
    )
    mp.train(epochs)
    correct_test, confusion = mp.evaluate(testxs, testys)
    test_err = 1 - correct_test.mean()
    print(f"Split {idx}: Test err: {test_err}")
    all_split_results[idx] = test_err
    all_confusions[idx] = confusion.values.ravel()

all_split_results = pd.Series(all_split_results)
all_cs = pd.Series(all_cs)
all_cs_results = pd.DataFrame(all_cs_results)
all_cs_results.index = cs

store['all_split_results'] = all_split_results
store['all_cs'] = all_cs
store['all_cs_results'] = all_cs_results

store.flush()
store.close()
