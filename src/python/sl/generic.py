import itertools

import percytron

import concurrent.futures
from copy import copy, deepcopy

import pandas as pd
import numpy as np


# needed for multiprocessing
def trainp(args):
    k, p, epochs = args
    p.train(epochs)
    return k, p


def evalp(args):
    k, p = args
    import data
    yraw = p.evaluate(data.testxs)
    return (k, p, yraw)




# generic kernel perceptron that for now will act like vanilla
class KernelPerceptron:
    # Runs a batch kernel perceptron
    # given the training batch.
    # Will return 1 or -1
    # iF reflabel is passed, will compare against that as the correct value
    def __init__(self, kernel_function, trainxs, trainys, reflabel=None,
                 parallel=False, testxs=None, testys=None):
        # must be callable and vectorised
        self.kernel_function = kernel_function
        self.N, self.D = trainxs.shape
        self.indices = np.arange(self.N)
        self.alpha = np.zeros(self.N, dtype=float)
        self.parallel = parallel
        if not self.parallel:
            self.X = trainxs
            self.y = trainys
            self.testxs = testxs
            self.testys = testys
        else:
            import data
            self.X = data.X
            self.y = data.y
            self.testxs = data.testxs
            self.testys = data.testys
        self.predictions = np.zeros_like(self.y, dtype=float)
        self.reflabel = reflabel
        if self.reflabel is not None:
            # have to change self.y so it is 1 and -1
            self.y_ = np.where(
                self.y == self.reflabel,
                1,
                -1
            )
        else:
            self.y_ = self.y
        self.y_ = self.y_.astype(float)

    def train(self, epochs):
        K = self.kernel_function(self.X, self.X)
        # simulate everything being 0 at the start
        # should hold the raw scores
        percytron.train(self.y_, self.predictions,
                        self.alpha, K, self.N, epochs)
        return self

    def evaluate(self, testxs, testys=None):
        K = self.kernel_function(self.X, testxs)
        yraw = np.einsum(
            'i,i,ij->j', self.alpha, self.y_, K
        )
        if testys is None:
            return yraw

        ypred = np.where(yraw > 0, 1, -1)

        if self.reflabel is not None:
            ref = np.where(testys == self.reflabel, 1, -1)
        else:
            ref = testys
        correct = ypred == ref
        return correct


class MultiKernelPerceptron:
    # composes several classes of kernel perceptron,
    # each detecting a single label
    def __init__(self, kernel_function, trainxs, trainys, reflabels,
                 parallel=False, testxs=None, testys=None):
        # must be callable and vectorised
        self.kernel_function = kernel_function
        self.N, self.D = trainxs.shape
        self.X = trainxs
        self.y = trainys
        self.testxs = testxs
        self.testys = testys
        self.reflabels = np.array(reflabels)
        self.parallel = parallel
        # same order as reflabels
        if not self.parallel:
            self.perceptrons = [
                KernelPerceptron(kernel_function, trainxs, trainys, label)
                for label in self.reflabels
            ]
        else:
            import data
            data.X = trainxs
            data.y = trainys
            data.testxs = testxs
            data.testys = testys
            self.perceptrons = [
                KernelPerceptron(kernel_function, trainxs, trainys, label,
                                 testxs, testys)
                for label in self.reflabels
            ]

    def train(self, epochs):
        N = self.y.size
        d = len(self.reflabels)
        y_, predictions, alpha = [], [], []
        K = self.kernel_function(self.X, self.X)
        # TODO: orientation?
        # can't be bothered to refactor everything to not include the
        # individual perceptrons, at this point they are just data structures
        ps = self.perceptrons
        for p in ps:
            y_.append(p.y_)
            predictions.append(p.predictions)
            alpha.append(p.alpha)
        y_ = np.vstack(y_).T
        predictions = np.vstack(predictions).T
        alpha = np.vstack(alpha).T
        percytron.train_all(y_, predictions, alpha, K, N, d, epochs)
        # now assign back
        for i, p in enumerate(ps):
            p.y_ = y_[:, i]
            p.predictions = predictions[:, i]
            p.alpha = alpha[:, i]
        # return alphas so we can track which items were misclassified
        return alpha

    def evaluate(self, testxs, testys):
        self.all_preds = np.zeros(
            (testys.size, len(self.reflabels)), dtype=float)
        N = self.y.size
        d = len(self.reflabels)
        ntest = testys.size
        y_, alpha = [], []
        K = self.kernel_function(self.X, testxs)
        ps = self.perceptrons
        for p in ps:
            y_.append(p.y_)
            alpha.append(p.alpha)
        y_ = np.vstack(y_).T
        alpha = np.vstack(alpha).T
        percytron.eval_all(self.all_preds, y_, alpha, K, N, d, ntest)
        ypred = self.reflabels[np.argmax(self.all_preds, axis=1)]
        correct = ypred == testys
        confusion = self.confusion(ypred, testys)
        return correct, confusion

    def confusion(self, ypred, testys):
        # generate confusion matrix
        confusion = {}
        for label in self.reflabels:
            expected = testys == label
            wrong = testys[expected] != ypred[expected]
            wronglabels = ypred[expected][wrong]
            predicted, counts = np.unique(wronglabels, return_counts=True)
            confusion[label] = pd.Series(data=counts, index=predicted)
        confusion = pd.DataFrame(confusion)
        confusion = confusion.reindex(
            self.reflabels).T.reindex(self.reflabels).T.fillna(0)
        return confusion


class MultiKernelPerceptron1v1(MultiKernelPerceptron):
    # 1 vs 1 classification
    # For each pair of classes, set up specific classifier
    # We will still be passed all the data at one shot
    # and we will have to split it here.
    def __init__(self, kernel_function, trainxs, trainys, reflabels,
                 parallel=False, testxs=None, testys=None):
        # must be callable and vectorised
        self.kernel_function = kernel_function
        self.reflabels = np.array(reflabels)

        # make the splits
        splits = list(itertools.combinations(self.reflabels, 2))
        data = np.hstack([np.atleast_2d(trainys).T, trainxs])
        data_by_category = {
            label: data[data[:, 0] == label]
            for label in self.reflabels
        }

        self.perceptrons = {}
        self.parallel = parallel

        for label, other in splits:
            # create one kernel perceptron for each split
            # the data will be a reduced data set
            # the perceptron will return 1 if the item is "label" and -1
            # if it is the other in the split
            splitx = np.vstack([
                data_by_category[label],
                data_by_category[other],
            ])
            # now shuffle it
            np.random.shuffle(splitx)
            splity = splitx[:, 0].squeeze()
            splitx = splitx[:, 1:]
            p = KernelPerceptron(kernel_function, splitx, splity, label)
            self.perceptrons[(label, other)] = p

    def train(self, epochs):
        if not self.parallel:
            for k, p in self.perceptrons.items():
                p.train(epochs)
        else:
            args = [(k, p, epochs) for k, p in self.perceptrons.items()]
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallel) as executor:
                for key, p in executor.map(trainp, args):
                    self.perceptrons[key] = p

    def evaluate(self, testxs, testys):
        ncombinations = len(self.perceptrons.keys())
        self.all_preds = np.zeros((testys.size, ncombinations), dtype=float)
        # get the prediction for each single binary classifier
        # the final result will be the class with the greatest number of votes
        if not self.parallel:
            for idx, ((label, other), p) in enumerate(self.perceptrons.items()):
                yraw = p.evaluate(testxs)
                yvotes = np.where(yraw > 0, label, other)
                self.all_preds[:, idx] = yvotes
        else:
            args = self.perceptrons.items()
            all_pairs = list(self.perceptrons.keys())
            import data
            data.testxs = testxs
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallel) as executor:
                for (key, p, yraw) in executor.map(evalp, args):
                    label, other = key
                    idx = all_pairs.index(key)
                    yvotes = np.where(yraw > 0, label, other)
                    self.all_preds[:, idx] = yvotes

        counts = (self.all_preds[:, :, None] == self.reflabels)
        argmax = np.argmax(counts.sum(axis=1), axis=1)
        ypred = self.reflabels[argmax]
        correct = ypred == testys
        confusion = self.confusion(ypred, testys)
        return correct, confusion
