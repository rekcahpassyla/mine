from unittest import TestCase

import numpy as np

from algos import Perceptron, Winnow, LR, OneNN, generate_random, generate_n


class _TestAlgos:

    algo_cls = None

    def setUp(self):
        np.random.seed(12321312)

    def test(self):
        trainxs, trainys = generate_random(100, 200, 3)
        testxs, testys = generate_random(100, 10, 3)
        algo = self.algo_cls(trainxs, trainys)

        algo.train(50)
        ypred, correct = algo.predict(testxs, testys)
        check = np.where(testys < 0, 0, 1)
        self.assertTrue(np.all(correct))

    def test_single(self):
        # this is the test where there is only one testx / testy
        # for each set of simulations.
        trainxs, trainys = generate_random(100, 200, 15)
        algo = self.algo_cls(trainxs, trainys)
        algo.train(1)
        testxs = generate_n(15)
        testys = testxs[:, 0]
        ypred, allcorrect = algo.predict(testxs, testys)
        check = np.where(testys < 0, 0, 1)
        self.assertTrue(np.all(allcorrect))


class TestPerceptron(_TestAlgos, TestCase):
    algo_cls = Perceptron


class TestWinnow(_TestAlgos, TestCase):
    algo_cls = Winnow


class TestLR(_TestAlgos, TestCase):
    algo_cls = LR


class TestOneNN(_TestAlgos, TestCase):
    algo_cls = OneNN

    def test(self):
        trainxs, trainys = generate_random(100, 200, 5)
        testxs, testys = generate_random(1, 10, 5)
        algo = self.algo_cls(trainxs, trainys)

        algo.train(50)
        # different for oneNN
        ypred, correct = algo.predict(testxs.squeeze(), testys.squeeze())
        check = np.where(testys < 0, 0, 1)
        self.assertTrue(np.all(correct))
