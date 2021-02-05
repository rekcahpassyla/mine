# Base script, run different degrees of polynomial kernel

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

outfile = 'gaussian_gridsearch_final.csv'
fh = open(outfile, 'w')

fh.write('i,c,train_err,test_err\n')
# cs to try
cs = [0.001, 0.01, 0.014, 0.017, 0.018, 0.019, 0.02, 0.022, 0.024, 0.03, 0.1]
# convert to sigma. c = 1/(2*sigma**2), sigma = sqrt(1/2c)

for c in cs:
    d_train_results = []
    d_test_results = []
    print('-------------------')
    for idx in range(Nsplits):
        trainset, testset = loadsplit.randomsample(loadsplit.data, 0.2)
        trainys, trainxs = loadsplit.labelsvalues(trainset)
        testys, testxs = loadsplit.labelsvalues(testset)
        kf = partial(gaussian, c=c)
        mp = MultiKernelPerceptron(
            kf, trainxs, trainys, reflabels=list(range(0, 10)), parallel=5
        )
        mp.train(epochs)
        correct_train, _ = mp.evaluate(trainxs, trainys)
        correct_test, _ = mp.evaluate(testxs, testys)
        train_err = 1 - correct_train.mean()
        test_err = 1 - correct_test.mean()
        d_train_results.append(train_err)
        d_test_results.append(test_err)
        msg = f"{idx},{c},{train_err},{test_err}"
        fh.write(msg)
        fh.write("\n")
        fh.flush()
        print(f"i={idx} c={c}: Train error: {train_err:.4f}, Test error: {test_err:.4f}")

fh.close()
