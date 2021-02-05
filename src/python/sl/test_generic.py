import pandas as pd

import unittest
import scipy.linalg

from functools import partial

import numpy as np
from pandas._testing import assert_frame_equal

from alcygos import onenn_float
from .generic import KernelPerceptron, MultiKernelPerceptron, MultiKernelPerceptron1v1
from .kernels import polynomial
from .percytron import gaussian_c as gaussian


def generate_positive(size, reflabel=1, neglabel=-1):
    # return reflabel if x > 0, -1 otherwise
    xs = np.random.uniform(low=-2, high=2, size=(size, 2))
    ys = np.where(
        xs[:, 0] > 0,
        reflabel,
        neglabel
    )
    return xs, ys


def generate_diagonal(size, reflabel=1):
    # return reflabel if x > y, -1 otherwise
    xs = np.random.uniform(low=-2, high=2, size=(size, 2))
    ys = np.where(
        xs[:, 0] > xs[:, -1] + 0.5,
        reflabel,
        -1
    )
    return xs, ys


def generate_radial(size, reflabel=1, neglabel=-1):
    half = size//2
    xplus = np.random.uniform(low=-1, high=1, size=(half, 2))
    xminus = np.random.uniform(low=-1, high=1, size=(half, 2))
    xplus /= 3
    xminus *= 3
    x = np.vstack([xplus, xminus])
    np.random.shuffle(x)

    xpow = np.sum(np.power(x, 2), axis=1)
    y = np.where(xpow <= 1, reflabel, neglabel)
    return x, y


def generate_radial_3(size):
    # generate labels 0, 1, 2
    # 0 is for r <= 1
    # 1 is for 1 < r <= 2
    # 2 is for 2 < r <= 3

    x = np.random.uniform(low=-1.5, high=1.5, size=(size, 2))
    xpow = np.sum(np.power(x, 2), axis=1)
    y = []
    for xi in xpow:
        if xi <= 1:
            y.append(0)
        elif xi <= 2:
            y.append(1)
        else:
            y.append(2)
    y = np.array(y)
    return x, y


def ktest(Xi, x, c=0.1):
    # Xi: data in rows
    # x: test point
    # Test function that acts as normal inner product
    # but it has to have a bias term
    return np.einsum(
        'ij,j->i',
        Xi,
        x
    ) + c


def ktest_single(x, y, c=0.1):
    return np.dot(x, y) + c


def radial_kernel(X, x):
    return np.square(
        np.einsum('ij,j->i', X, x) + 0.1
    )


def radial_single(x, y):
    return np.square(np.dot(x, y) + 0.1)


class TestGenericKP(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)

    def test_generic(self):
        # test generic kernel perceptron
        # that it can correctly separate a trivial test set
        trainxs, trainys = generate_positive(100, reflabel=0)
        testxs, testys = generate_positive(10, reflabel=0)
        kf = partial(polynomial, d=1, c=0.1)
        kp = KernelPerceptron(kf, trainxs, trainys, reflabel=0)

        kp.train(50)

        correct = kp.evaluate(testxs, testys)
        acc = correct.mean()
        self.assertEquals(acc, 1.0)


    def test_generic_diagonal(self):
        # test a slightly less trivial test set
        trainxs, trainys = generate_diagonal(200)
        testxs, testys = generate_diagonal(10)
        kf = partial(polynomial, d=1, c=0.1)
        kp = KernelPerceptron(kf, trainxs, trainys)

        kp.train(50)

        correct = kp.evaluate(testxs, testys)
        acc = correct.mean()
        self.assertEquals(acc, 1.0)

    def test_multiclass_single_binary(self):
        # test the usage within a multi class.
        # generate data that has labels 0, 1, 2
        # only 0 should be triggered as the valid label

        def generate(size):
            # generate labels 0, 1, 2
            # 0 is for x <=-1
            # 1 is for -1 < x <= 1
            # 2 is for x > 1
            x = np.random.uniform(low=-2, high=2, size=(size, 2))
            y = []
            for xi in x:
                if xi[1] <= -1.0:
                    y.append(0)
                elif xi[1] < 1.0:
                   y.append(1)
                else:
                   y.append(2)
            y = np.array(y)
            return x, y

        trainxs, trainys = generate(2000)
        testxs, testys = generate(20)
        k = partial(ktest, c=0.1)
        k = partial(polynomial, d=1, c=0.1)
        kp = KernelPerceptron(k, trainxs, trainys, reflabel=0)

        kp.train(100)

        correct = kp.evaluate(testxs, testys)
        acc = correct.mean()
        self.assertEquals(acc, 1.0)

    def test_radial(self):
        # test radial data
        reflabel = 47

        trainxs, trainys = generate_radial(1000, reflabel)
        testxs, testys = generate_radial(10, reflabel)

        kf = partial(polynomial, d=2, c=0.1)
        kp = KernelPerceptron(kf, trainxs, trainys, reflabel)
        kp.train(100)
        correct = kp.evaluate(testxs, testys)
        acc = correct.mean()
        self.assertEquals(acc, 1.0)

    def test_justalittlebit(self):
        # input is 1 or -1, output is first dimension
        trainxs = np.random.random(size=(1000, 750)) - 0.5
        trainxs = np.where(trainxs > 0, 1.0, -1.0)
        trainys = trainxs[:, 0]

        testxs = np.random.random(size=(10, 750)) - 0.5
        testxs = np.where(testxs > 0, 1.0, -1.0)
        testys = testxs[:, 0]

        kf = partial(polynomial, d=1, c=0)
        kp = KernelPerceptron(kf, trainxs, trainys)

        kp.train(100)

        correct = kp.evaluate(testxs, testys)
        acc = correct.mean()
        self.assertEquals(acc, 1.0)


class TestMulti(unittest.TestCase):

    def setUp(self):
        np.random.seed(12321312)
        self.testcls = MultiKernelPerceptron

    def test_linear(self):
        # multi class had better act like single class
        trainxs, trainys = generate_positive(
            size=100, reflabel=0, neglabel=1)
        testxs, testys = generate_positive(
            size=10, reflabel=0, neglabel=1)
        kf = partial(polynomial, d=1, c=0.1)
        mp = self.testcls(
            kf, trainxs, trainys, reflabels=[0, 1])
        mp.train(50)
        correct, confusion = mp.evaluate(testxs, testys)
        acc = correct.mean()
        self.assertEquals(acc, 1.0)

    def test_radial(self):
        # multi class had better act like single class
        trainxs, trainys = generate_radial(
            size=1000, reflabel=0, neglabel=1)
        testxs, testys = generate_radial(
            size=10, reflabel=0, neglabel=1)
        kf = partial(polynomial, d=2, c=0.1)
        mp = self.testcls(
            kf, trainxs, trainys, reflabels=[0, 1])
        mp.train(100)
        correct, confusion = mp.evaluate(testxs, testys)
        acc = correct.mean()
        self.assertEquals(acc, 1.0)

    def test_3(self):

        def generate(size):
            # generate labels 0, 1, 2
            # 0 is for x <=-1
            # 1 is for -1 < x <= 1
            # 2 is for x > 1
            x = np.random.uniform(low=-2, high=2, size=(size, 2))
            y = []
            for xi in x:
                if xi[0] <= -1:
                    y.append(0)
                elif xi[0] < 1:
                    y.append(1)
                else:
                    y.append(2)
            y = np.array(y)
            return x, y

        trainxs, trainys = generate(size=1000)
        testxs, testys = generate(size=100)
        kf = partial(polynomial, d=1, c=0.1)
        mp = self.testcls(
            kf, trainxs, trainys, reflabels=[0, 1, 2], parallel=4)
        mp.train(100)
        correct, confusion = mp.evaluate(testxs, testys)
        acc = correct.mean()
        # data are not actually linearly separable, oops
        self.assertTrue(acc > 0.95)

    def test_3_radial(self):


        trainxs, trainys = generate_radial_3(size=1000)
        testxs, testys = generate_radial_3(size=100)
        gauss = partial(gaussian, c=20)
        # separate using Gaussian kernel
        mp = self.testcls(
            gauss,
            trainxs, trainys, reflabels=[0, 1, 2], parallel=4)
        mp.train(100)
        correct, confusion = mp.evaluate(testxs, testys)
        acc = correct.mean()
        # data are not actually linearly separable, oops
        assert acc >= 0.95, acc

    def test_confusion_matrix(self):
        gauss = partial(gaussian, c=20)
        # Dummy, just for testing confusion matrix
        trainxs = np.random.random((10, 5))
        trainys = np.random.random(10)
        mp = self.testcls(
            gauss,
            trainxs, trainys, reflabels=[0, 1, 2, 3, 4])
        # actual
        testys = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        ypred =  np.array([1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2])
        expected = pd.DataFrame(
            [[0, 0, 1, 1, 1],
             [1, 0, 0, 1, 1],
             [1, 1, 0, 0, 1],
             [1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0]],
            index=mp.reflabels, columns=mp.reflabels,
            dtype=float
        )
        cm = mp.confusion(ypred, testys)
        assert_frame_equal(cm, expected)


class TestMulti1v1(TestMulti):

    def setUp(self):
        np.random.seed(12321312)
        self.testcls = MultiKernelPerceptron1v1


class TestPCA(unittest.TestCase):

    def test_3(self):

        def generate(size):
            # generate labels 0, 1, 2
            # 0 is for x <=-1
            # 1 is for -1 < x <= 1
            # 2 is for x > 1
            x = np.random.uniform(low=-2, high=2, size=(size, 2))
            y = []
            for xi in x:
                if xi[0] <= -1:
                    y.append(0)
                elif xi[0] < 1:
                    y.append(1)
                else:
                    y.append(2)
            y = np.array(y)
            return x, y

        trainxs, trainys = generate(size=1000)
        testxs, testys = generate(size=100)
        kf = partial(polynomial, d=1, c=0.1)
        mp = self.testcls(
            kf, trainxs, trainys, reflabels=[0, 1, 2], parallel=4)
        mp.train(100)
        correct, confusion = mp.evaluate(testxs, testys)
        acc = correct.mean()
        # data are not actually linearly separable, oops
        self.assertTrue(acc > 0.95)


    def test_3_radial(self):
        trainxs, trainys = generate_radial_3(size=1000)
        testxs, testys = generate_radial_3(size=100)
        X = trainxs - trainxs.mean(axis=0)
        U, S, V = np.linalg.svd(X, full_matrices=False)
        # flip svd?
        components = V
        keep = 2
        keepv = components[:keep]
        projected = X @ keepv.T
        testX = testxs - testxs.mean(axis=0)
        proj_test = testX @ keepv.T
        ypred = np.zeros((1, testys.size), dtype=float)
        shape = (1, X.shape[0], testX.shape[0])
        working = np.zeros(shape, dtype=float)
        onenn_float(
            ypred, working, proj_test, np.atleast_3d(projected),
            np.atleast_2d(trainys.astype(float)),
            1, X.shape[0], X.shape[1], testX.shape[0]
        )
        print("")


class TestKernelCCA(unittest.TestCase):

    def test_3_radial(self):
        trainxs, trainys = generate_radial_3(size=1000)
        testxs, testys = generate_radial_3(size=100)
        labels = np.array([0, 1, 2])
        import rcca
        cca = rcca.CCA(reg=0.1, numCC=3, kernelcca=True, ktype='poly')
        cca.degree = 8
        cca.train([trainxs, np.atleast_2d(trainys).T])
        print("")
