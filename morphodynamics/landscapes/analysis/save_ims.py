import tensorflow as tf
import numpy as np
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os
import copy

from morphodynamics.landscapes.utils import get_meshgrid
from morphodynamics.landscapes.analysis.get_fields import *
from morphodynamics.landscapes.analysis.sde_forward import *



class Save_Ims():
    """
    Save losses during training and the potential as an array
    """

    def __init__(self, model, save_dir):
        """
        - model: this is the physics-informed neural network (PINN)
        - save_dir: where to save the plot and potential to
        """

        self.model = model
        self.save_dir = save_dir

        x_test, y_test = get_meshgrid(model.xlims, model.ylims, model.dims, flatBool = True)
        self.x_test , self.y_test = tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test)

        self.fig = plt.figure(figsize = (30, 20))
        self.gs = gridspec.GridSpec(nrows = 15, ncols = 17)


    def __call__(self):
        self._plot_losses()
        self._plot_pdfs_getUandD()
        self._plot_and_save_U()
        #plt.savefig(self.save_dir + 'View_{}_{}.png'.format(self.model.save_append, self.model.idx_save))
        #plt.close()

    def _setup_ax(self, ax):
        ax.set_aspect('equal', adjustable = 'box')
        ax.set_xlim(self.model.xlims)
        ax.set_ylim(self.model.ylims)


    def _plot_losses(self):
        """
        Plot how each of the loss terms changes in time
        """

        ax = self.fig.add_subplot(self.gs[2:5, :7])
        losses = [self.model.data_losses, self.model.BC_losses,  self.model.pde_losses, self.model.total_losses, self.model.norm_losses]
        labels = ['pdf', 'BC', 'pde', 'total', 'norm']
        zipped = zip(losses, labels)
        for loss_list, label in zipped:
            ax.plot(np.log10(loss_list), label = label)
        ax.legend()

    def _plot_pdfs_getUandD(self):
        """
        Run inference to get the pdf, potential (U) and diffusivity (D)
        """

        p_max = 0
        D_max = 0
        for idx_t, test_time in enumerate(np.linspace(self.model.tlims[0], self.model.tlims[1], 7)): # test for a range of unseen times
            t_test = np.tile(np.array([test_time]), (self.x_test.shape[0], 1))
            t_test = tf.convert_to_tensor(t_test)
            xyt_test = tf.concat((self.x_test, self.y_test, t_test), axis = 1)
            p_out, D_out, U_out = self.model.predict(xyt_test)
            D_out = D_out.numpy()
            p_max = max(p_max, np.max(p_out))
            D_max = max(D_max, np.max(D_out))


        for idx_t, test_time in enumerate(np.linspace(self.model.tlims[0], self.model.tlims[1], 7)): # test for a range of unseen times

            t_test = np.tile(np.array([test_time]), (self.x_test.shape[0], 1))
            t_test = tf.convert_to_tensor(t_test)
            xyt_test = tf.concat((self.x_test, self.y_test, t_test), axis = 1)
            p_out, D_out, U_out = self.model.predict(xyt_test)
            p_out = p_out.numpy()
            D_out = D_out.numpy()
            U_out = U_out.numpy()

            ax_p = self.fig.add_subplot(self.gs[6, idx_t])

            p_out[p_out<1e-7] = np.nan
            ax_p.scatter(self.x_test, self.y_test, c = np.log10(p_out), vmin = -7, vmax = max(np.log10(p_max), -7))
            self._setup_ax(ax_p)

            ax_D = self.fig.add_subplot(self.gs[6, 8+idx_t])
            ax_D.scatter(self.x_test, self.y_test, c = D_out, vmin = 0, vmax = D_max)
            self._setup_ax(ax_D)

        for idx_t, arr in enumerate(self.model.pdf_list):
            ax = self.fig.add_subplot(self.gs[14, idx_t])
            to_log = copy.deepcopy(arr)
            to_log[to_log<1e-7] = np.nan
            ax.imshow(np.log10(to_log.reshape((200, 200))[::-1, :]))

        self.U_out = U_out


    def _plot_and_save_U(self):
        """
        Plot and save the potential as an array
        """

        U = np.reshape(self.U_out, (self.model.dims, self.model.dims))

        path = self.save_dir + 'potential.pickle'
        dump_pickle(U, path)

        ax = self.fig.add_subplot(self.gs[:4, 10:14])
        gx, gy = np.gradient(U)
        ax.imshow(np.log10(np.sqrt(gx**2 + gy**2))[::-1, :])
        ax.set_aspect('equal', adjustable = 'box')
