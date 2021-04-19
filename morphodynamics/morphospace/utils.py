import numpy as np
from sklearn.neighbors import KernelDensity
import glob
import sys
import os



def get_lims_from_points(points):
    xs, ys = [], []
    for item in points:
        xs.append(item[0][0])
        ys.append(item[0][1])
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    return min_x, max_x, min_y, max_y



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



def _standardize_coords(coords, orig_lims):
    [xmin, xmax, ymin, ymax] = orig_lims
    x_new = -10 + 20*(coords[0]-xmin)/(xmax-xmin)
    y_new = -10 + 20*(coords[1]-ymin)/(ymax-ymin)
    return [x_new, y_new]
