from scipy.stats import skewnorm
import tensorflow as tf
import numpy as np
import time
import sys
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from morphodynamics.landscapes.utils import *



def get_section_flatten(xg, yg, row_sector, row_sec_size):
    xg_partial = xg[(row_sector*row_sec_size):(row_sector*row_sec_size)+row_sec_size, :]
    yg_partial = yg[(row_sector*row_sec_size):(row_sector*row_sec_size)+row_sec_size, :]
    xg_partial = xg_partial.flatten()[:, None]
    yg_partial = yg_partial.flatten()[:, None]
    return xg_partial, yg_partial


def get_p(net_p,  xs, ys, ts):
    xyt = np.concatenate([xs, ys, ts], axis = 1)
    xyt = tf.Variable(xyt, dtype = tf.float64)
    p = net_p(xyt)
    return p

def get_D(net_D,  xs, ys, ts):
    xyt = np.concatenate([xs, ys, ts], axis = 1)
    xyt = tf.Variable(xyt, dtype = tf.float64)
    D = net_D(xyt)
    return D


def _rebin(arr, new_shape):
    """
    For 2D arrays only.
    """
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)



def get_listsOf_pArrays(model, num_snaps, dims0, dimsN, save_path = None):
    """
    Run inference on the model to get a list of pdfs at different time points
    """
    row_sec_size = 10

    t0 = np.tile(np.array([model.tlims[0]]), (dims0*row_sec_size, 1))
    t_list = [t0]
    for idx_snap in range(1, num_snaps):
        t = model.tlims[0] + idx_snap*(model.tlims[1] - model.tlims[0])/(num_snaps-1)
        t_list.append(np.tile(np.array([t]), (dimsN**2, 1)))

    xg0, yg0 = get_meshgrid(model.xlims, model.ylims, dims0, False)
    xgN, ygN = get_meshgrid(model.xlims, model.ylims, dimsN, True)

    p_lists = []

    p_list = []

    p_0 = np.zeros((dims0, dims0))

    for row_sector in range(dims0 // row_sec_size):
        ts = t0

        xg_partial, yg_partial = get_section_flatten(xg0, yg0, row_sector, row_sec_size)

        p = get_p(model.net_p, xg_partial, yg_partial, t0)
        p = p.numpy().reshape(row_sec_size, dims0)
        p_0[(row_sector*row_sec_size):(row_sector*row_sec_size)+row_sec_size, :] = p

    p_list.append(p_0)

    for idx_time in range(1, num_snaps):
        ts = t_list[idx_time]

        p = get_p(model.net_p, xgN, ygN, ts)

        p = p.numpy().reshape(dimsN, dimsN)
        p_list.append(p)

    p_list_200 = [_rebin(i, (200, 200)) for i in p_list]

    dump_pickle(p_list_200, save_path.format(0))

    p_lists.append(p_list)

    return p_lists

def get_listsOf_DArrays(model, num_Ds, dims, save_path):
    """
    Run inference on the model to get a list of diffusivities (arrays on the spatial grid) at different time points
    """

    listsOf_DArrays = []

    t_list = []
    for idx_snap in range(num_Ds):
        t = model.tlims[0] + idx_snap*(model.tlims[1] - model.tlims[0])/(num_Ds-1)
        t_list.append(np.tile(np.array([t]), (dims**2, 1)))

    D_list = []
    for ts in t_list:
        xs, ys = get_meshgrid(model.xlims, model.ylims, dims, True)
        D = get_D(model.net_D, xs, ys, ts)
        D = D.numpy().reshape(dims, dims)
        D_list.append(D)

    #dump_pickle(D_list, save_path.format(0))
    listsOf_DArrays.append(D_list)

    return listsOf_DArrays



def get_F_field(model, dims,  save_path):
    """
    Run inference on the model to get the force field (array on the spatial grid)
    """
    row_sec_size = 10

    grid = np.zeros((dims, dims, 2))

    xg, yg = get_meshgrid(model.xlims, model.ylims, dims, False)


    for row_sector in range(dims // row_sec_size):
        xg_partial, yg_partial = get_section_flatten(xg, yg, row_sector, row_sec_size)

        xy = np.concatenate([xg_partial, yg_partial], axis = 1)
        xy = tf.Variable(xy, dtype = tf.float64)

        with tf.GradientTape() as tape:
            U = model.net_U(xy)

        first_derivs = tape.gradient(U, xy)
        Fx = -first_derivs[:, 0:1]
        Fy = -first_derivs[:, 1:2]

        out_x = Fx.numpy().reshape((row_sec_size, dims))
        out_y = Fy.numpy().reshape((row_sec_size, dims))

        grid[(row_sector*row_sec_size):(row_sector*row_sec_size)+row_sec_size, :, 0] = out_x
        grid[(row_sector*row_sec_size):(row_sector*row_sec_size)+row_sec_size, :, 1] = out_y


    grid_200 = np.zeros((200, 200, 2))
    grid_200[:, :, 0] = _rebin(grid[:, :, 0], (200, 200))
    grid_200[:, :, 1] = _rebin(grid[:, :, 1], (200, 200))

    #dump_pickle(grid_200, save_path)

    return grid
