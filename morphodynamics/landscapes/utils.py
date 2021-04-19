import numpy as np
import pickle
import os
from sklearn.neighbors import KernelDensity


def make_dir_weights(dir_weights):
    if os.path.isdir(dir_weights):
        raise OSError('Directory already exists; exiting.')
    else:
        os.mkdir(dir_weights)

def dump_pickle(obj, path):
    file = open(path, 'wb')
    pickle.dump(obj, file)


def unpack_lims(xlims, ylims, tlims):

    all = xlims + ylims + tlims
    return all

def get_meshgrid(xlims, ylims, dims, flatBool):
    xs = np.linspace(xlims[0], xlims[1], dims)
    ys = np.linspace(ylims[0], ylims[1], dims)
    xg, yg = np.meshgrid(xs, ys)

    if flatBool:
        xg = xg.flatten()[:, None]
        yg = yg.flatten()[:, None]
    return xg, yg



def get_mean_std_from_datasets(datasets_list):

    targets_removed = []
    for i in datasets_list:
        targets_removed.append(i[:, :3])

    concat = np.concatenate(targets_removed, axis = 0)
    mean = np.mean(concat, axis = 0)
    std = np.std(concat, axis = 0)
    return mean, std

def kde(snapshot, lims_list, dims, kde_bw):

    snapshot = np.array(snapshot)
    snapshot = snapshot

    xs = np.linspace(lims_list[0], lims_list[1], dims)
    ys = np.linspace(lims_list[2], lims_list[3], dims)
    xx, yy = np.meshgrid(xs, ys)
    positions = np.vstack([xx.ravel(), yy.ravel()]).T

    kernel = KernelDensity(bandwidth = kde_bw)
    kernel.fit(snapshot)
    pdf_array = np.exp(kernel.score_samples(positions))
    pdf_array = np.reshape(pdf_array, xx.shape)

    return pdf_array
