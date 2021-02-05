import scipy.stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


store = pd.HDFStore('data/algo_convergence_n20_allsims_except_1nn.hdf5', mode='r')
store.keys()

algos = ['Perceptron', 'Winnow', 'LR']

names = {'LR': 'Linear regression'}

sims = [50, 100, 200, 500, 1000, 2000, 5000]

all_errors = {}
# TODO: errorbars
for algo in algos:
    plt.figure();
    errors = {}
    convergence = store['convergence'].unstack().T[algo].unstack().T
    for n in sims:
        err = store[f'errors_{n}'][algo].unstack().max(axis=1)
        errors[n] = err
        plt.errorbar(convergence[n].index, convergence[n].values,
                     err.values, label=n)
    errors = pd.DataFrame(errors)
    all_errors[algo] = errors
    plt.grid()
    name = names.get(algo, algo)
    plt.title(f'{name} convergence: different numbers of simulations')
    plt.xlabel('n')
    plt.ylabel('m')
    plt.legend()
    plt.savefig(f'{algo}_convergence.png')

nnsims = [50, 100, 200, 500, 1000]
storenn = pd.HDFStore('data/algo_convergence_n20_allsims_1nn.hdf5', mode='r')
algo = 'OneNN'
plt.figure()
convergence = storenn['convergence'].unstack().T[algo].unstack().T
for n in nnsims:
    err = storenn[f'errors_{n}'][algo].unstack().max(axis=1)
    plt.errorbar(convergence[n].index, convergence[n].values,
                 err.values, label=n)
plt.grid()
name = names.get(algo, algo)
plt.title(f'{name} convergence: different numbers of simulations')
plt.xlabel('n')
plt.ylabel('m')
plt.legend()
plt.savefig(f'{algo}_convergence.png')

cstore = pd.HDFStore('data/algo_complexity_n100_100sims_except_1nn.hdf5', 'r')
cstorenn = pd.HDFStore('data/algo_complexity_n25_50sims_1nn.hdf5', 'r')

complexity = cstore['convergence'].unstack().T.loc[100]
complexity_1nn = cstorenn['convergence'].unstack().T.loc[50]

# fit various shapes
perc = complexity['Perceptron']
win = complexity['Winnow']
lr = complexity['LR']
nn = complexity_1nn['OneNN']

p = scipy.stats.linregress(perc.index, perc.values)
l = scipy.stats.linregress(lr.index, lr.values)
nwin = np.log(win.index)
w = scipy.stats.linregress(nwin.values, win.values)
lnn = np.log2(nn)
n = scipy.stats.linregress(nn.index, lnn.values)

fitp = pd.Series(p.slope*perc.index + p.intercept, perc.index)
fitl = pd.Series(l.slope*lr.index + l.intercept, lr.index)


def testwin(n, ww, factor=1):
    return ww.slope*np.log(n)*factor + ww.intercept

fitw = pd.Series(testwin(win.index, w), win.index)



def plot_with_fit(actual, fitted, algo, tag, r, filetag=None, additional=None,
                  additional_labels=None):
    plt.figure()
    plt.plot(actual, label=algo)
    plt.grid()
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('m')
    plt.title(f'Sample complexity for {algo}')
    if filetag:
        plt.savefig(f'{algo}_{filetag}.png')
    else:
        plt.savefig(f'{algo}.png')
    plt.plot(fitted, color='k', linestyle='-', label=tag, alpha=0.5)
    if additional:
        linestyles = ['--', ':']
        for line, label, style in zip(additional, additional_labels, linestyles):
            plt.plot(line, color='k', linestyle=style, alpha=0.4, label=label)
    plt.title(f'Sample complexity for {algo} with best fit line (R={r:.4f})')
    plt.legend()
    if filetag:
        plt.savefig(f'{algo}_with_fit_{filetag}.png')
    else:
        plt.savefig(f'{algo}_with_fit.png')


ptag = f'm = {p.slope:.3f}n {"+" if p.intercept >= 0 else "-"}{abs(p.intercept):.3f}'

pu = pd.Series(p.slope*perc.index*1.1 + p.intercept, perc.index)
pl = pd.Series(p.slope*perc.index*0.9 + p.intercept, perc.index)

ptags = [
    f'm = {factor*p.slope:.3f}n {"+" if p.intercept >= 0 else "-"}{abs(p.intercept):.3f}'
    for factor in (1.1, 0.9)
]

plot_with_fit(perc, fitp, 'Perceptron', ptag, p.rvalue)


plot_with_fit(perc, fitp, 'Perceptron', ptag, p.rvalue, 'bounds',
              additional=[pu, pl], additional_labels=ptags)

ltag = f'm = {l.slope:.3f}n {"+" if l.intercept >= 0 else "-"} {abs(l.intercept):.3f}'
plot_with_fit(lr, fitl, 'Linear Regression', ltag, l.rvalue)
lu = pd.Series(l.slope*lr.index*1.1 + l.intercept, lr.index)
ll = pd.Series(l.slope*lr.index*0.9 + l.intercept, lr.index)

ltags = [
    f'm = {factor*l.slope:.3f}n {"+" if l.intercept >= 0 else "-"}{abs(l.intercept):.3f}'
    for factor in (1.1, 0.9)
]
plot_with_fit(lr, fitl, 'Linear Regression', ltag, l.rvalue, 'bounds',
              additional=[lu, ll], additional_labels=ltags)


wtag = (
    f'm = ${w.slope:.3f} \log(n) {"+" if w.intercept >= 0 else "-"} {abs(w.intercept):.3f}$'
)


plot_with_fit(win, fitw, 'Winnow', wtag, w.rvalue)


def testn(n_, factor=1):
    pow = n.slope*n_+ n.intercept
    return factor*np.power(2.0, pow)  #np.exp(pow)

fitn = pd.Series(testn(nn.index), nn.index)

ntag = f'm = 2**({n.slope:.3f}n {"+" if n.intercept >= 0 else "-"} {abs(n.intercept):.3f})'
plot_with_fit(nn, fitn, 'OneNN', ntag, n.rvalue)

nu = testn(nn.index, factor=1.1)
nl = testn(nn.index, factor=0.9)

nu = pd.Series(nu, nn.index)
nl = pd.Series(nl, nn.index)

ntags = [
    f'm = {factor}*2**({n.slope:.3f}n {"+" if n.intercept >= 0 else "-"} ' 
           f'{abs(n.intercept):.3f})'
    for factor in (1.1, 0.9)
]

plot_with_fit(nn, fitn, 'OneNN', ntag, n.rvalue, 'bounds',
              additional=[nu, nl], additional_labels=ntags)


wstore = pd.HDFStore('algo_complexity_n500_100sims_winnow.hdf5', 'r')
wc = wstore['convergence']
win2 = wc[100]['Winnow']

nwin2 = np.log(win2.index)
nwin2 = pd.Series(nwin2, win2.index)

w2 = scipy.stats.linregress(nwin2.values, win2.values)


fitw2 = testwin(win2.index, w2)

wtag2 = (
   f'm = ${w2.slope:.3f} \log(n) {"+" if w.intercept >= 0 else "-"}{abs(w2.intercept):.3f}$'
)

plot_with_fit(win2, fitw2, 'Winnow', wtag2, w2.rvalue, '500')



wu = testwin(win2.index, w2, factor=1.1)
wu = pd.Series(wu, win2.index)

wl = testwin(win2.index, w2, factor=0.9)
wl = pd.Series(wl, win2.index)

wtags = [(
    f'm = ${w2.slope*factor:.3f} \log(n) {"+" if w.intercept >= 0 else "-"}{abs(w2.intercept):.3f}$'
)
for factor in (1.1, 0.9)
]


plot_with_fit(win2, fitw2, 'Winnow', wtag2, w2.rvalue, '500_bounds',
              additional=[wu, wl], additional_labels=wtags)
