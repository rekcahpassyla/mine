# Base script, run different degrees of polynomial kernel

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
epochs = 3

outfile = 'polynomial_gridsearch.csv'
fh = open(outfile, 'w')

fh.write('i,d,train_err,test_err\n')

for d in range(1, 8):
    d_train_results = []
    d_test_results = []
    print('-------------------')
    for idx in range(Nsplits):

        trainset, testset, _, _ = loadsplit.randomsample(loadsplit.data, 0.2)
        trainys, trainxs = loadsplit.labelsvalues(trainset)
        testys, testxs = loadsplit.labelsvalues(testset)
        kf = partial(polynomial, d=d)
        mp = MultiKernelPerceptron(
            kf, trainxs, trainys, reflabels=list(range(0, 10)), parallel=10
        )
        mp.train(epochs)
        correct_train, _ = mp.evaluate(trainxs, trainys)
        correct_test, _ = mp.evaluate(testxs, testys)
        train_err = 1 - correct_train.mean()
        test_err = 1 - correct_test.mean()
        d_train_results.append(train_err)
        d_test_results.append(test_err)
        msg = f"{idx},{d},{train_err},{test_err}"
        fh.write(msg)
        fh.write("\n")
        fh.flush()
        print(f"i={idx} d={d}: Train error: {train_err:.4f}, Test error: {test_err:.4f}")

fh.close()
