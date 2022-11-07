import loadsplit
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_stats(file):
    # loading stats for leetcode
    store = pd.HDFStore(file, 'r')
    convergence = store['convergence']
    errors = {}
    stds = {}

    for key in store.keys():
        if key.startswith('/errors'):
            sims = int(key.split("_")[1])
            errors[sims] = store[key]
            stds[sims] = store[f'/stds_{sims}']
    store.close()

    # reformat the errors and stds to only have the maximum
    e = {}
    for sims, errs in errors.items():
        e[sims] = {}
        for alg in errs.columns:
            algdict = {}
            df = errs[alg].dropna().unstack().T
            dfs = stds[sims][alg].dropna().unstack().T
            for n in df.columns:
                argmin = df[n].dropna().argmin()
                algdict[n] = (
                    df[n].dropna().iloc[argmin],
                    dfs[n].dropna().iloc[argmin]
                )

            e[sims][alg] = pd.Series(algdict)
            e[sims][alg] = pd.DataFrame(e[sims][alg])

    return convergence, e


#c, e = get_stats('data/algo_convergence_n20_allsims_except_1nn.hdf5')


def meanstd_table(means, stds, fmt='.6f'):
    # convert to text df with +/- in the cells
    # mean, std are 2d arrays
    vals = {}
    fmtstr = f"{{thismean:{fmt}}} \pm {{thisstd:{fmt}}}"
    for col in means:
        thiscol = []
        for name in means.index:
            thismean = means[col][name]
            thisstd = stds[col][name]
            thisval = fmtstr.format(thismean=thismean, thisstd=thisstd)
            thiscol.append(thisval)
            vals[col] = thiscol
    vals = pd.DataFrame(vals)
    vals.columns = means.columns
    vals.index = means.index
    out = vals.to_latex()
    # convert the \pm to proper formatting
    out = out.replace("\\textbackslash pm", "$\pm$")
    return out


def plot3dparams(resultsdf, xlabel, ylabel, zlabel, title, filename):
    # xindex: values for x index eg. polynomial degree from 1-4
    # yindex: values for y index eg. number of components
    # Z: variable
    xindex = resultsdf.columns
    yindex = resultsdf.index
    Z = resultsdf.values
    X, Y = np.meshgrid(xindex, yindex) # X: gamma, Y: sigma
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha = 0.6)
    totalmin = resultsdf.unstack().argmin()
    # these are values not indices
    xminval, yminval = resultsdf.unstack().index[totalmin]
    ax.scatter3D(
        xminval, yminval,
        resultsdf[xminval].loc[yminval],
        color='red'
    )
    ax.set_xlabel(xlabel)#, fontsize= 15, labelpad = 20)
    ax.set_ylabel(ylabel)#, fontsize= 15, labelpad = 20)
    ax.set_zlabel(zlabel)#, fontsize= 15, labelpad = 20)
    ax.set_title(title)#, fontsize=20)
    ax.tick_params(axis='z')#, labelsize= 15)
    plt.show()
    fig.savefig(filename)


def parse_gridsearch(filename, tag, title, drop=None):
    # drop = index values not to keep

    df = pd.read_csv(filename, index_col=(0, 1))
    # print table of means / stds for training and test err
    means = df.unstack().mean(axis=0).unstack().T
    if drop is not None:
        kept = means.index.difference(drop)
    else:
        kept = means.index
    means = means.loc[kept]
    stds = df.unstack().std(axis=0).unstack().T.loc[kept]
    texout = meanstd_table(means, stds)
    # plot errorbars of same
    for col, full  in [('train_err', "Training error"),
                       ('test_err', "Test error")]:
        plt.figure();
        plt.errorbar(means.index, means[col].values, stds[col].values)
        plt.grid()
        plt.title(f"{title}: {full}")
        plt.savefig(f"{col}_{tag}.png")
    print(texout)

# Gaussian
#parse_gridsearch("data/gaussian_gridsearch_final.csv", "gaussian", "Gaussian kernel", drop=[0.001])
#parse_crossvalidation('data/gaussian_crossvalidation_final_fine.hdf5', 'gaussian', 'Gaussian cross-validation results: Best test error per split')

# Polynomial
#parse_gridsearch("data/polynomial_gridsearch.csv", "poly", "Polynomial kernel")
#parse_crossvalidation('data/polynomial_crossvalidation.hdf5', 'poly', 'Polynomial cross-validation results: Best test error per split')


# Polynomial
#parse_gridsearch("data/polynomial_gridsearch_1v1.csv", "poly_1v1", "Polynomial kernel, 1 vs 1")
#parse_crossvalidation('data/polynomial_crossvalidation_1v1.hdf5', 'poly_1v1', 'Polynomial (1 vs 1 strategy) cross-validation results: Best test error per split')

def confusions():
    store = pd.HDFStore('data/polynomial_crossvalidation.hdf5', 'r')
    c = store['all_confusions']
    store.close()
    confusions = c.T.values.reshape((-1, 10, 10))
    counts = np.unique(loadsplit.data[:, 0], return_counts=True)
    counts = pd.Series(counts[1], counts[0].astype(int))
    cnorm = confusions / counts.values[None, :, None]
    cnormmeans = pd.DataFrame(cnorm.mean(axis=0))
    cnormstds = pd.DataFrame(cnorm.std(axis=0))
    return cnormmeans, cnormstds


# Taken from:
# https://matplotlib.org/3.3.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    imdata = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        #threshold = im.norm(data.max())/2.
        threshold = im.norm(imdata.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            #kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            #text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            kw.update(color=textcolors[int(im.norm(imdata[i, j]) < threshold)])
            text = im.axes.text(j, i, data[i][j], **kw)
            texts.append(text)

    return texts

def plot_confusion():

    cnormmeans, cnormstds = confusions()

    fig, ax = plt.subplots()

    im, cbar = heatmap(cnormmeans.values, cnormmeans.index, cnormmeans.index, ax=ax,
                       cbarlabel="mean confusion\n(misclassifications / counts in category)")

    data = []

    for i in range(10):
        thisrow = []
        for j in range(10):
            if i == j:
                text = "0"
            else:
                text = '%.4f\n\u00B1%.4f' % (cnormmeans.values[i][j], cnormstds[i][j])
            thisrow.append(text)
        data.append(thisrow)

    data = np.array(data)

    texts = annotate_heatmap(im, data=data, valfmt=None)#"{x:.1f} t")

    fig.tight_layout()
    plt.show()
    plt.title('Confusion matrix\nColumns indicate actual label, rows indicate predicted label')



def parse_crossvalidation(filename, tag, title, paramname='d'):
    store = pd.HDFStore(filename, 'r')
    best_param = store[f'all_{paramname}s']
    best_err = store['all_split_results']
    print(f'Best param: {best_param.mean(), best_param.std()}')
    print(f'Best err: {best_err.mean(), best_err.std()}')
    plt.figure();
    plt.scatter(best_param.values, best_err.values)
    plt.xlabel(paramname)
    plt.ylabel("Test error")
    plt.grid()
    plt.title(title)
    plt.savefig(f'xv_{tag}.png')
    store.close()



def parse_gridsearch_2d(filename, tag, title, paramname='d'):
    store = pd.HDFStore(filename, 'r')
    means = (1-store['results']).mean(axis=1).unstack()
    stds = (1-store['results']).std(axis=1).unstack()
    # number of components in columns
    plt.figure()
    for p in means.index:
        thismean = means.loc[p]
        thisstd = stds.loc[p]
        plt.errorbar(thismean.index, thismean.values, thisstd.values, label=p)
    plt.xlabel(paramname)
    plt.ylabel('Test error')
    plt.grid()
    plt.legend()
    plt.title(f"{title}: Test error mean over 20 splits")
    plt.savefig(f"test_err_{tag}.png")

    best_param = store[f'all_{paramname}s']
    best_err = store['all_split_results']
    print(f'Best param: {best_param.mean(), best_param.std()}')
    print(f'Best err: {best_err.mean(), best_err.std()}')
    plt.figure()
    plt.scatter(best_param.values, best_err.values)
    plt.xlabel(paramname)
    plt.ylabel("Test error")
    plt.grid()
    plt.title(title)
    plt.savefig(f'xv_{tag}.png')
    store.close()



def parse_crossvalidation_2d(filename, tag, title, break_at=None, paramname='d'):
    # for handling crossvalidation over 2 dimensions
    store = pd.HDFStore(filename, 'r')
    means = (1-store['results']).mean(axis=1).unstack()
    stds = (1-store['results']).std(axis=1).unstack()
    # number of components in columns
    plt.figure()
    for p in means.index:
        thismean = means.loc[p]
        thisstd = stds.loc[p]
        plt.errorbar(thismean.index, thismean.values, thisstd.values, label=f"{paramname}={p}")
    plt.xlabel('Number of components')
    plt.ylabel('Test error')
    plt.grid()
    plt.legend()
    plt.title(f"{title}: Test error mean over 20 splits")
    plt.savefig(f"test_err_{tag}.png")


    # results by params in columns, rows = splits
    all_params_results = store['all_params_results']
    # best param for each split
    all_params = store['all_params']
    split_results = store['split_results']
    print(f"Best parameters: {(all_params.mean(axis=0), all_params.std(axis=0))}")
    print(f"Best error: {(split_results.mean(axis=0), split_results.std(axis=0))}")
    store.close()

    # get minimum value for each parameter combination, over all splits
    mins = np.argmin(all_params_results.values, axis=1)
    minvalues = []
    for idx, item in enumerate(all_params_results.index):
        minvalues.append(all_params_results[mins[idx]][item])
    minvalues = pd.Series(minvalues, all_params_results.index)
    allmins = minvalues.unstack()

    means = all_params_results.mean(axis=1).unstack()
    stds = all_params_results.std(axis=1).unstack()

    plot3dparams(
        allmins, paramname, 'components', 'Test error',
        f'Test error surface of minimum test error over all splits:\n{title}', f'test_err_surf_{tag}.png')



#parse_crossvalidation_2d('data/kpca_1nn_poly_40_81.hdf5', 'kpca_poly', 'Kernel PCA + 1NN / Polynomial kernel', paramname='d')
#parse_crossvalidation_2d('data/kpca_1nn_gaussian_40_81_2.hdf5', 'kpca_gaussian', 'Kernel PCA + 1NN / Gaussian kernel', paramname='c')
#parse_crossvalidation_2d('data/kpca_1nn_histogram.hdf5', 'kpca_histogram', 'Kernel PCA + 1NN / Histogram kernel', paramname='bins')
#parse_crossvalidation_2d('data/kpca_1nn_localcorr_40_81_deg_localdeg.hdf5', 'kpca_localcorr', 'Kernel PCA + 1NN / Local correlation kernel', paramname='d')

def parse_cnn_crossvalidation_2d(results, all_params_results, all_params, split_results, tag, title):
    # for handling crossvalidation over 2 dimensions

    means = results.mean(axis=1).unstack().T
    stds = results.std(axis=1).unstack().T
    # batch size in rows
    plt.figure()
    for bs in means.index:
        thismean = means.loc[bs]
        thisstd = stds.loc[bs]
        plt.errorbar(thismean.index, thismean.values, thisstd.values, label=f"batch size={bs}")
    plt.xlabel('learning rate')
    plt.ylabel('Test error')
    plt.grid()
    plt.legend()
    plt.title(f"{title}: Test error mean over 20 splits")
    plt.savefig(f"test_err_{tag}.png")


    # results by params in columns, rows = splits
    #all_params_results = store['all_params_results']
    # best param for each split
    #all_params = store['all_params']
    #split_results = store['split_results']
    print(f"Best parameters: {(all_params.mean(axis=0), all_params.std(axis=0))}")
    print(f"Best error: {(split_results.mean(axis=0), split_results.std(axis=0))}")
    #store.close()

    # get minimum value for each parameter combination, over all splits
    mins = np.argmin(all_params_results.values, axis=1)
    minvalues = []
    for idx, item in enumerate(all_params_results.index):
        minvalues.append(all_params_results[mins[idx]][item])
    minvalues = pd.Series(minvalues, all_params_results.index)
    allmins = minvalues.unstack()

    means = all_params_results.mean(axis=1).unstack()
    stds = all_params_results.std(axis=1).unstack()

    plot3dparams(
        allmins, 'learning rate', 'batch size', 'Test error',
        f'Test error surface of minimum test error over all splits:\n{title}', f'test_err_surf_{tag}.png')

def parse_cnn_crossvalidation(all_params, all_split_results, tag, title):
    best_err = all_split_results
    #print(f'Best param: {best_param.mean(), best_param.std()}')
    #print(f'Best err: {best_err.mean(), best_err.std()}')
    plt.figure();
    plt.scatter(best_param.values, best_err.values)
    plt.xlabel('learning rate')
    plt.ylabel("Test error")
    plt.grid()
    plt.title(title)
    plt.savefig(f'xv_{tag}.png')
    store.close()
