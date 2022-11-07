# Holds all the leetcode for part 2
# perceptron, winnow,, least squares, 1-nn
# format of all is that they will take a sequence of points
# and return a callable that can evaluate other points
import concurrent.futures
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from alcygos import onenn



class Algo(metaclass=ABCMeta):

    # these classes are all vectorised
    # they will compute on a block of x and y inputs

    def __init__(self, trainxs, trainys):
        # trainxs is 3d, trainys is 2d
        # first dimension is number of simulations
        self.trainxs = trainxs
        self.trainys = trainys
        self.nsims, self.N, self.d = self.trainxs.shape

    @abstractmethod
    def train(self, epochs):
        # do whatever is needed to update internal state
        pass

    @abstractmethod
    def predict(self, trainxs, trainys):
        # run prediction on all trainxs
        # compare with trainys
        # return vector of number correct
        pass


class Perceptron(Algo):

    def __init__(self, trainxs, trainys):
        # non OneNN must convert to float or weird things happen with einsum
        super(Perceptron, self).__init__(
            trainxs.astype(float), trainys.astype(float))
        self.weights = np.zeros((self.nsims, self.d), dtype=float)
        # no bias since we know our inputs are +/- 1?

    def train(self, epochs):
        counter = 0
        x_ = np.transpose(self.trainxs, (1, 0, 2))
        y_ = self.trainys.T
        while counter < epochs:
            for x, y in zip(x_, y_):
                yraw = np.einsum('...i,...i->...',
                                 self.weights, x)
                ypred = np.where(yraw > 0, 1, -1)
                delta = y - ypred
                self.weights += delta[:, None] * x
            counter += 1

    def predict(self, testxs, testys):
        # compute all at once
        yraw = np.einsum('...j,...ij->...i',
                         self.weights, testxs.astype(float))
        ypred = np.where(yraw > 0, 1, -1)
        return ypred, ypred == testys


class Winnow(Algo):

    def __init__(self, trainxs, trainys):
        super(Winnow, self).__init__(
            trainxs.astype(float), trainys.astype(float))
        # Winnow needs 1, 0 values so convert
        self.y_ = np.where(self.trainys < 0, 0, 1)
        self.x_ = np.where(self.trainxs < 0, 0, 1)
        self.weights = np.ones((self.nsims, self.d))
        self.two = np.ones(self.d) * 2

    def train(self, epochs):
        counter = 0
        # transpose so that iterate over x and process each sim
        x_ = np.transpose(self.x_, (1, 0, 2))
        y_ = self.y_.T
        self.weights2 = self.weights.copy()
        while counter < epochs:
            # use the transformed values: 0 instead of -1
            for i, (x, y) in enumerate(zip(x_, y_)):
                yraw = np.einsum('...i,...i->...',
                                 self.weights, x)
                ypred = np.where(yraw < self.d, 0, 1)
                loss = (y - ypred)
                mistake = loss != 0
                new_wts = (
                    self.weights * np.power(self.two, loss[:, None] * x)
                )
                # if there was a mistake, take new_wts
                # otherwise take current weights
                self.weights = np.where(mistake[:, None], new_wts, self.weights)
                self.weights2 = new_wts
                if not np.allclose(self.weights, self.weights2):
                    raise AssertionError()

            counter += 1

    def predict(self, testxs, testys):
        # convert to 0, 1
        x_ = np.where(testxs < 0, 0, 1)
        y_ = np.where(testys < 0, 0, 1)
        yraw = np.einsum('...j,...ij->...i',
                         self.weights, x_.astype(float))
        ypred = np.where(yraw < self.d, 0, 1)
        return ypred, ypred == y_


class LR(Algo):

    def __init__(self, trainxs, trainys):
        super(LR, self).__init__(
            trainxs.astype(float), trainys.astype(float))
        # XTX^-1 XTY
        xtx = np.einsum('...ji,...jk->...ik', trainxs, trainxs)
        xty = np.einsum('...ji,...j->...i',  trainxs, trainys)
        xtxinv = np.linalg.pinv(xtx)
        self.weights = np.einsum('...ij,...j->...i', xtxinv, xty)

    def train(self, epochs):
        # this class has no training, everything needed is in the init
        pass

    def predict(self, testxs, testys):
        yraw = np.einsum('...j,...ij->...i', self.weights, testxs.astype(float))
        ypred = np.where(yraw > 0, 1, -1)
        return ypred, ypred == testys


def onenn_(ypred, working, testxs, trainxs, trainys, nsims, m, n, t):
    i = 0
    j = 0
    mini = -1
    for nsim in range(nsims):
        # first calculate cdist
        for i in range(m):
            for j in range(t):
                working[nsim][i][j] = 0
                # add coordinate
                for k in range(n):
                    working[nsim][i][j] += (
                        abs(trainxs[nsim][i][k] - testxs[nsim][j][k])
                    )
        test = abs(trainxs[nsim][:, :, None] - testxs[nsim].T).sum(axis=1)
        try:
            assert np.allclose(test, working[nsim])
        except:
            raise
        # need argmin over each column of working[nsim]
        # use ypred to store
        for j in range(t):
            mini = -1
            for i in range(m):
                if mini == -1:
                    mini = i
                if working[nsim][i][j] < working[nsim][mini][j]:
                    mini = i
            ypred[nsim][j] = trainys[nsim][mini]
            test = trainys[nsim][working[nsim][:, j].argmin()]
            try:
                assert np.allclose(test, ypred[nsim][j])
            except:
                raise


class OneNN(Algo):

    def train(self, epochs):
        # also no training.
        pass

    def predict(self, testxs, testys, batchsize=5):
        # find nearest.
        # Use cdist for now, don't know if can optimise by checking for equality
        # slow, cythonise next
        nsims, m, n = self.trainxs.shape
        ypred = np.zeros((nsims, testys.size), dtype=np.int8)
        t = testys.size
        # do it in batches.
        if not batchsize:
            working = np.zeros((nsims, m, t), dtype=np.int8)
            onenn(ypred, working, testxs, self.trainxs, self.trainys,
                    nsims, m, n, t)
        else:
            working = np.zeros((batchsize, m, t), dtype=np.int8)
            for idx in range(0, nsims, batchsize):
                onenn(ypred[idx:idx+batchsize, :], working, testxs, self.trainxs, self.trainys,
                      batchsize, m, n, t)

        return ypred, ypred == testys


def gen_err(wrong, n):
    # return generalisation error which is
    # sum(wrong) / (2**n)
    return wrong / (2**n)


def generate_random(nsims, m, n):
    # nsims = number of simulations
    # m = dataset size
    # n = dimension
    data = np.random.random((nsims, m, n)) - 0.5
    x = np.where(data > 0, 1, -1).astype(np.int8)
    y = x[:, :, 0]
    return x, y

def generate_n(n):
    # generate all combinations of {-1, 1}^n
    data = []
    inp = np.array([-1, 1])
    for i in range(2**n):
        bs = '{i:0>{n}b}'.format(i=i, n=n)
        indexes = [int(s) for s in bs]
        data.append(inp[indexes])
    return np.array(data, dtype=np.int8)
"""
def generate_n(n, split=None):
    # generate all combinations of {-1, 1}^n
    inp = np.array([-1, 1])
    num = 2**n
    if split is None or num < split:
        data = []
        for i in range(2**n):
            bs = '{i:0>{n}b}'.format(i=i, n=n)
            indexes = [int(s) for s in bs]
            data.append(inp[indexes])
        yield np.array(data, dtype=np.int8)
    else:
        splits = num // split
        for split_idx in range(splits):
            start, end = split_idx*split, (split_idx+1)*split
            data = []
            for i in range(start, end):
                bs = '{i:0>{n}b}'.format(i=i, n=n)
                indexes = [int(s) for s in bs]
                data.append(inp[indexes])
            yield np.array(data, dtype=np.int8)
"""
#def run_sim(clsname, nsims=1000, maxm=500, maxn=10):
def run_sim(args):
    algo_cls, nsims, maxm, maxn = args
    values, errors, stds = simulate(algo_cls, nsims, maxm, maxn)
    return args, values, errors, stds


def predictalg(args):
    alg, tx, ty = args
    ypred, correct = alg.predict(tx, ty)
    return correct


def simulate(alg_cls, nsims=50, maxm=500, maxn=10, max_data=128):
    # simulations for sample complexity
    # C(A) = min_{m} [average generalisation error(A(S_m)) < 0.1]
    values = np.zeros(maxn, dtype=float)
    errs = {}
    stds = {}
    ns = list(range(1, maxn+1))
    #ns = list(range(15, maxn+1))
    n_to_m = {}
    np.random.seed(12321312)
    ONENN = alg_cls is OneNN

    if not ONENN:
        ms = list(range(1, maxm+1))
        for n in ns:
            n_to_m[n] = ms
    else:
        # we know it's exponential and we have a bound on it
        slope = 0.40
        intercept = 0.98

        def testm(n):
            return np.exp(slope*n + intercept)

        ms_ = testm(np.array(ns))
        for n, m in zip(ns, ms_):
            m_ = max(0, int(m))
            if n < 12:
                lo = max(0, m_-5)
            elif n < 23:
                lo = max(0, m_-20)
            else:
                lo = max(0, m_-100)
            n_to_m[n] = list(range(lo, m_+50))

    for n in ns:
        print("-------------------")
        thisn = np.zeros(maxm, dtype=float)
        if alg_cls is not OneNN and n > ns[0]:
            prev = max(0, values[n-2]-2)
        else:
            prev = 0
        for m in n_to_m[n]:
            if m < prev:
            #    print(f"Skipping {m}")
                continue
            # each simulation will find err(A(S_m))
            trainx, trainy = generate_random(nsims, m, n)
            alg = alg_cls(trainx, trainy)
            # everything trains for 1 epoch
            alg.train(1)
            # keep track of, for each sim, how many wrong predictions there were
            wrong = np.zeros(nsims, dtype=float)
            num = 2**n
            lg_maxdata = np.log2(max_data)
            if num < max_data:
                testx = generate_n(n)
                testy = testx[:, 0]
                ndata = n
            else:
                # generate random data
                testx, testy = generate_random(1, max_data, n)
                testx = testx.squeeze()
                testy = testy.squeeze()
                ndata = lg_maxdata
            _, correct = alg.predict(testx, testy)
            # count how many are wrong in each simulation
            wrong += (~correct).sum(axis=1)
            err = gen_err(wrong, ndata)
            mean_err = err.mean()
            std_err = err.std()
            # estimate stdev of error.
            # error = 2^{-n} * sum(wrong) which is linear
            thisn[m-1] = mean_err
            errs[(n, m)] = mean_err
            stds[(n, m)] = std_err
            msg = (
                f"{alg_cls.__name__}:{nsims}: n={n}, m={m}, "
                f"mean err={mean_err}, std={std_err}"
            )
            print(msg)
            if mean_err < 0.10:
                print(f"Found generalisation error < 0.10: {msg}")
                values[n-1] = m
                break
    values = pd.Series(values, ns)
    errs = pd.Series(errs)
    stds = pd.Series(stds)
    return values, errs, stds


if __name__ == '__main__':
    np.random.seed(12321312)
    ONENN = False
    if ONENN:
        clsnames = ["OneNN"]
        parallel = 3
        nsims = [50]
        store = pd.HDFStore('algo_convergence_n25_50sims_1nn.hdf5', 'a')
        maxn = 25
    else:
        clsnames = ["Perceptron", "Winnow", "LR"]
        clsnames = ['Winnow']
        nsims = [100]
        parallel = 0
        store = pd.HDFStore('algo_complexity_n500_100sims_winnow.hdf5', 'a')
        maxn = 500

    values = {N: {} for N in nsims}
    errors = {N: {} for N in nsims}
    stds = {N: {} for N in nsims}
    maxm = 100000
    args = []
    for N in nsims:
        for clsname in clsnames:
            args.append((eval(clsname), N, maxm, maxn))
    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
            for (alg_cls, N, maxm, maxn), v, e, s in executor.map(run_sim, args):
                clsname = alg_cls.__name__
                values[N].setdefault(clsname, {})
                errors[N].setdefault(clsname, {})
                stds[N].setdefault(clsname, {})
                values[N][clsname] = v
                errors[N][clsname] = e
                stds[N][clsname] = s
    else:
        for argset in args:
            (alg_cls, N, _, _), v, e, s = run_sim(argset)
            clsname = alg_cls.__name__
            values[N].setdefault(clsname, {})
            errors[N].setdefault(clsname, {})
            stds[N].setdefault(clsname, {})
            values[N][clsname] = v
            errors[N][clsname] = e
            stds[N][clsname] = s
    valuesdf = pd.DataFrame({
        nsims: pd.DataFrame(values[nsims]).unstack() for nsims in values})

    store['convergence'] = valuesdf

    errorsdict = {}
    stdsdict = {}
    for nsims in errors:
        df = pd.DataFrame({
            clsname: errors[nsims][clsname]
            for clsname in errors[nsims]
        })
        errorsdict[nsims] = df
        df = pd.DataFrame({
            clsname: stds[nsims][clsname]
            for clsname in stds[nsims]
        })
        stdsdict[nsims] = df

    for nsims in errorsdict:
        store[f"errors_{nsims}"] = errorsdict[nsims]
        store[f"stds_{nsims}"] = stdsdict[nsims]

    store.close()
