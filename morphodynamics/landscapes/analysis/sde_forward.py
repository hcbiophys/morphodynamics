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
import copy

from morphodynamics.landscapes.utils import get_meshgrid, dump_pickle, kde

class Run_SDE():

    def __init__(self, kdes_data, p_nn_list, F_array, D_arrays, xlims, ylims, kde_bw):

        self.kdes_data = kdes_data
        self.kdes_data_2 = None

        self.p_nn_list = p_nn_list
        self.F_array = F_array
        self.D_arrays = D_arrays

        self.scatters_nn = [[], [], [], [], [], [], [], [], []]
        self.kdes_nn = []
        self.paths = []

        self.xlims = xlims
        self.ylims = ylims

        self.kde_bw = kde_bw

    def _randomSample_xys(self, array, dims, num):
        rowcols = []
        idxs = np.random.choice(array.flatten().shape[0], num, p = array.flatten()/np.sum(array))
        for idx in idxs:
            row = idx // dims
            col = idx % dims
            rowcols.append((row, col))
        xys = []
        for (row, col) in rowcols:
            x = self.xlims[0] + (col/dims)*(self.xlims[1]-self.xlims[0])
            y = self.ylims[0] + (row/dims)*(self.ylims[1]-self.ylims[0])
            xys.append((x, y))
        return xys

    def _xy_to_rowcol(self, x, y, dims):
        row = int(dims*(y-self.ylims[0])/(self.ylims[1]-self.ylims[0]))
        col = int(dims*(x-self.xlims[0])/(self.xlims[1]-self.xlims[0]))
        return row, col

    def _force_at_position(self, x, y):
        row, col = self._xy_to_rowcol(x, y, dims = 5000)
        F_x = self.F_array[row, col, 0]
        F_y = self.F_array[row, col, 1]
        return F_x, F_y

    def get_D(self, x, y, t, dims, num_Ds):
        idx_time = int(t//(120/num_Ds))
        row, col = self._xy_to_rowcol(x, y, dims = dims)
        D = self.D_arrays[idx_time][row, col]
        return D

    def run(self, num_particles, dt, T):
        thresh_p_nn_0 = copy.deepcopy(self.p_nn_list[0])
        starts = self._randomSample_xys(thresh_p_nn_0, 5000, num_particles)
        starts = [(i,j) for (i,j) in starts if i > self.xlims[0] and i < self.xlims[1] and j > self.ylims[0] and j < self.ylims[1]]

        temp_snap_times = np.linspace(0, T, 9)
        snap_times = []
        for time in temp_snap_times:
            idx = np.abs(np.arange(0, T, dt)-time).argmin()
            snap_times.append(np.arange(0, T, dt)[idx])

        for idx_start, position in enumerate(starts):
            print(idx_start)
            path = []
            for time_step in np.arange(0, T, dt):
                for idx_snap, snap_time in enumerate(snap_times):
                    if time_step == snap_time:
                        self.scatters_nn[idx_snap].append(position)
                if position[0] > self.xlims[0] and position[0] < self.xlims[1] and position[1] > self.ylims[0] and position[1] < self.xlims[1]:
                    F_x, F_y = self._force_at_position(position[0], position[1])
                    F = np.array([F_x, F_y])
                    D = self.get_D(position[0], position[1], time_step, dims = 200, num_Ds = 18)
                    sigma = np.sqrt(2*D)
                    position = position + F*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1, size = 2)
                    path.append(position)
                else:
                    print('Exit time step:', time_step)
                    break

            self.paths.append(path)

    def set_kdes_data_2(self, error_bw):

        self.kdes_data_2 = []

        fig = plt.figure(figsize = (10, 10))

        for idx, kde_orig in enumerate(self.kdes_data):
            ax = fig.add_subplot(2, 9, idx+1)
            ax.imshow(np.log10(kde_orig), vmin = -5, vmax = 0.19, cmap = 'jet')
            xys = self._randomSample_xys(kde_orig, dims=200, num=10000)

            snapshot = np.zeros((len(xys), 2))
            for i in range(len(xys)):
                snapshot[i, 0] = xys[i][0]
                snapshot[i, 1] = xys[i][1]

            kde_2 = kde(snapshot, self.xlims+self.ylims, dims = 200, kde_bw = error_bw)
            ax = fig.add_subplot(2, 9, 8+idx+1)
            ax.imshow(np.log10(kde_2), vmin = -5, vmax = 0.19, cmap = 'jet')
            self.kdes_data_2.append(kde_2)
        plt.savefig('/end/home/hc2814/Desktop/code_direct_outputs/comparison.png')

    def set_kdes_nn(self, error_bw):

        for idx_snap, scatter in enumerate(self.scatters_nn):

            snapshot = np.zeros((len(scatter), 2))
            for i in range(len(scatter)):
                snapshot[i, 0] = scatter[i][0]
                snapshot[i, 1] = scatter[i][1]

            p_array = kde(snapshot, self.xlims+self.ylims, dims = 200, kde_bw = error_bw)

            self.kdes_nn.append(p_array)


    def pickle_trajectories(self, keep_every, save_path):

        paths_subsampled = [i[::keep_every] for i in self.paths]

        dump_pickle(paths_subsampled, save_path)


    def pickle_for_errors(self, save_path1, save_path2, save_path3):

        dump_pickle(self.kdes_data_2, save_path1)
        #dump_pickle(self.p_nn_list, save_path2)
        dump_pickle(self.kdes_nn, save_path3)
