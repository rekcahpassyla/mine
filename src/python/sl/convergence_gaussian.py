from functools import partial

import numpy as np
import loadsplit

import generic
import kernels
import percytron

MultiKernelPerceptron = generic.MultiKernelPerceptron
gaussian = percytron.gaussian_c
polynomial = kernels.polynomial
c2s = kernels.c2s


train_results = []
test_results = []
confusions = []

epochs = 50

outfile = 'gaussian_convergence_0.006_0.04.csv'
fh = open(outfile, 'w')

fh.write('i,c,train_err,test_err\n')

# cs to try
cs = [0.006, 0.008, 0.01, 0.02, 0.04]

for c in cs:
    d_train_results = []
    d_test_results = []
    print('-------------------')
    trainset, testset = loadsplit.randomsample(loadsplit.data, 0.2)
    trainys, trainxs = loadsplit.labelsvalues(trainset)
    testys, testxs = loadsplit.labelsvalues(testset)
    kf = partial(gaussian, c=c)
    mp = MultiKernelPerceptron(
        kf, trainxs, trainys, reflabels=list(range(0, 10)), parallel=5
    )
    for i in range(epochs):
        mp.train(1)
        correct_train, _ = mp.evaluate(trainxs, trainys)
        correct_test, _ = mp.evaluate(testxs, testys)
        train_err = 1 - correct_train.mean()
        test_err = 1 - correct_test.mean()
        d_train_results.append(train_err)
        d_test_results.append(test_err)
        msg = f"{i},{c},{train_err},{test_err}"
        fh.write(msg)
        fh.write("\n")
        fh.flush()
        print(f"i={i} c={c}: Train error: {train_err:.4f}, Test error: {test_err:.4f}")


fh.close()
