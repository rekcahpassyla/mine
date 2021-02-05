import numpy as np
data = np.loadtxt('zipcombo.dat')


def randomsample(data, testpct):
    # return (trainset, testset)
    # number of samples in trainset is trainpct
    # testpct: float from 0 to 1 defining what fraction will be used as test set
    # returns (trainset, testset) where each is an array of the data
    N = data.shape[0]
    indexes = np.arange(N)
    test = np.random.choice(N, size=int(testpct*N), replace=False)
    train = set(indexes).difference(set(test))
    trainset = data[np.array(list(train))]
    testset = data[np.array(list(test))]
    # also return indexes
    return trainset, testset, train, test


def labelsvalues(dataset):
    return dataset[:, 0], dataset[:, 1:]


def split_folds(data, n=5):
    # data: pd.DataFrame[float](nrows, ncols)
    # Input data as a frame, each row represents one data point
    # n: number of folds
    # returns: ((pd.DataFrame, pd.DataFrame), ...)
    # Returns n folds, each is a pair of dataframes representing training set
    # and test set respectively.
    numdata = data.shape[0]
    samplesize = numdata//n
    out = []
    choices = data.index.copy()
    for idx in range(n):
        this_rows = np.random.choice(choices, size=samplesize, replace=False)
        this_rows = sorted(this_rows)
        this_test = data.loc[this_rows]
        # this will result in available rows getting smaller and smaller
        choices = choices.drop(this_rows)
        # training set is all the rest of the data except the current fold
        this_train = data.loc[data.index.difference(this_rows)]
        out.append((this_train, this_test))
    return out


def mles(trainset):

    labels, values = labelsvalues(trainset)

    splits = {
        item: values[labels==item]
        for item in np.unique(labels)
    }

    mles = np.array([
        splits[item].mean(axis=0)
        for item in splits
    ])
    return mles

# test sample for distance against all means
# report the one with the lowest distance
def test(sample, mles_):
    # sample contains the data in rows
    # mles have the digits in rows
    scores = np.dot(mles_, sample.T)
    return np.argmax(scores, axis=0)
