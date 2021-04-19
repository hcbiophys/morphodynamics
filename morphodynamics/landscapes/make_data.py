from scipy.stats import skewnorm
import numpy as np
import time
import sys
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import random


from fokker_planck.utils import get_meshgrid, unpack_lims


def make_PDE_dataset(num_collocation, xlims, ylims, tlims, dims):
    xmin, xmax, ymin, ymax, tmin, tmax = unpack_lims(xlims, ylims, tlims)

    x_eqns = np.random.uniform(xmin, xmax, size = num_collocation)[:, None]
    y_eqns = np.random.uniform(ymin, ymax, size = num_collocation)[:, None]
    t_eqns = np.random.uniform(tmin, tmax, size = num_collocation)[:, None]

    return np.concatenate([x_eqns, y_eqns, t_eqns], axis = 1)


def get_random_norm_slice(xlims, ylims, tlims, dims):

    x_norm, y_norm = get_meshgrid(xlims, ylims, dims, flatBool = True)

    t = random.choice(np.linspace(tlims[0], tlims[1], 100))
    t_norm = np.tile(np.array([t]), (dims**2, 1))

    xyts = np.concatenate([x_norm, y_norm, t_norm], axis = 1)

    return xyts


def make_BC_dataset(num_BC, xlims, ylims, tlims, dims):
    xmin, xmax, ymin, ymax, tmin, tmax = unpack_lims(xlims, ylims, tlims)

    t_BC = np.random.uniform(tmin, tmax, size = num_BC)[:, None]

    left_x = np.full(int(num_BC/4), xmin)
    left_y = np.random.uniform(ymin, ymax, size = int(num_BC/4))

    right_x = np.full(int(num_BC/4), xmax)
    right_y = np.random.uniform(ymin, ymax, size = int(num_BC/4))

    bottom_x = np.random.uniform(xmin, xmax, size = int(num_BC/4))
    bottom_y = np.full(int(num_BC/4), ymin)

    top_x = np.random.uniform(xmin, xmax, size = int(num_BC/4))
    top_y = np.full(int(num_BC/4), ymax)

    x_BC = np.concatenate([left_x, right_x, bottom_x, top_x])[:, None]
    y_BC = np.concatenate([left_y, right_y, bottom_y, top_y])[:, None]

    return np.concatenate([x_BC, y_BC, t_BC], axis = 1)




def make_pdf_datasets(pdfArrays_list, xlims, ylims, tlims, dims, num_snaps):

    pdf_datasets = []

    tmin, tmax = tlims[0], tlims[1]

    x_data, y_data = get_meshgrid(xlims, ylims, dims, flatBool = True)

    t = np.linspace(tmin, tmax, num_snaps)
    xs, ys, ts, ps = [], [], [], []
    for idx, pdf in enumerate(pdfArrays_list):

        t_data = np.tile(t[idx], (dims**2, 1))
        p_data = pdf.flatten()[:, None]

        pdf_datasets.append(np.concatenate([x_data, y_data, t_data, p_data], axis = 1))

    return pdf_datasets
