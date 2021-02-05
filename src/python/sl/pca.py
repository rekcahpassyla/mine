import numpy as np
import pandas as pd

import loadsplit


# PCA all the individual categories


def pca(xin, k):
    # rows are data items
    X = xin - xin.mean(axis=0)
    U, S, V = np.linalg.svd(X, full_matrices=False)
    components = V
    keepv = components[:k]
    projected = X @ keepv.T
    return keepv, projected


def pca_predict(trainxs, trainys, testxs, testys, keep=10):
    data = {
        label: trainxs[trainys == label]
        for label in range(0, 10)
    }
    pca_components = {}
    pca_projections = {}

    for label, ldata in data.items():
        components, projected = pca(ldata, keep)
        pca_components[label] = components
        pca_projections[label] = projected

    # On test set, transform the digits using each set of PCAs
    # Find the closest match
    testdata = {
        label: testxs[testys == label]
        for label in range(0, 10)
    }
    # Project each test item against the components for each class
    scores = np.zeros((testys.size, len(testdata)), dtype=float)

    testpcas = {}
    testxs_centered = testxs - testxs.mean(axis=0)
    txs_centered_norm = np.linalg.norm(testxs_centered, axis=1)
    for label, components in pca_components.items():
        test_projected = testxs_centered @ components.T
        testpcas[label] = test_projected
        # use cosine distance
        this_scores = np.linalg.norm(test_projected, axis=1) #/ txs_centered_norm
        scores[:, label] = np.abs(this_scores)

    ypred = np.argmax(scores, axis=1)
    return ypred, ypred == testys

store = pd.HDFStore('pca.hdf5', 'a')

results = {}
results_train = {}
for split in range(0, 20):
    results[split] = {}
    results_train[split] = {}
    for keep in range(5, 41):
        train, test, _, _ = loadsplit.randomsample(loadsplit.data, 0.2)

        trainys = train[:, 0]
        trainxs = train[:, 1:]

        testys = test[:, 0]
        testxs = test[:, 1:]

        _ , correct_train = pca_predict(
            trainxs, trainys, trainxs, trainys, keep=keep)

        ypred, correct = pca_predict(
            trainxs, trainys, testxs, testys, keep=keep)
        results[split][keep] = correct.mean()
        results_train[split][keep] = correct_train.mean()
        print(
            f"split: {split}: keep={keep}: "
            f"training acc={correct_train.mean()}, "
            f"test acc={correct.mean()}"
        )


results = pd.DataFrame(results)
results_train = pd.DataFrame(results_train)
store['results'] = results
store['results_train'] = results_train
store.flush()
"""

Nsplits = 20
seeds = np.arange(1, Nsplits+1)


all_split_results = {}
all_ks = {}
all_ks_results = {}
ks = np.arange(10, 31, dtype=int)

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
    k_test_results = []
    for k in ks:
        fold_test_results = []
        for f, (trainfold, testfold) in enumerate(folds):
            print(f"k={k}, fold={f}")
            trainfoldys, trainfoldxs = loadsplit.labelsvalues(trainfold.values)
            testfoldys, testfoldxs = loadsplit.labelsvalues(testfold.values)
            _, correct_test_fold = pca_predict(
                trainfoldxs, trainfoldys, testfoldxs, testfoldys, keep=k)
            test_fold_err = 1 - correct_test_fold.mean()
            print(f"k={k}, fold={f}: Test err: {test_fold_err}")
            fold_test_results.append(test_fold_err)
        fold_test_results = np.array(fold_test_results)
        mean_fold_error = fold_test_results.mean()
        # now pick the d with the lowest error
        k_test_results.append(mean_fold_error)
    k_test_results = np.array(k_test_results)
    all_ks_results[idx] = k_test_results
    best_k = ks[np.argmin(k_test_results)]
    print("--------------------")
    print(f"Split {idx}: Best k: {best_k}")
    all_ks[idx] = best_k
    # now run the best k against the full 80% training set
    trainys, trainxs = loadsplit.labelsvalues(trainset)
    testys, testxs = loadsplit.labelsvalues(testset)
    _, correct_test = pca_predict(
        trainxs, trainys, testxs, testys, keep=best_k)
    test_err = 1 - correct_test.mean()
    print(f"Split {idx}: Test err: {test_err}")
    all_split_results[idx] = test_err

all_split_results = pd.Series(all_split_results)
all_ks = pd.Series(all_ks)
all_ks_results = pd.DataFrame(all_ks_results, index=list(ks))

store['all_split_results'] = all_split_results
store['all_ks'] = all_ks
store['all_ks_results'] = all_ks_results

store.close()

"""
