import tensorflow as tf

from scipy.stats import skewnorm
import numpy as np
import scipy.io
import time
import sys
import scipy.stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import random

from morphodynamics.landscapes.networks import Residual_Net, FP_2D
from morphodynamics.landscapes.make_data import *
from morphodynamics.landscapes.utils import get_mean_std_from_datasets

from morphodynamics.landscapes.analysis.save_ims import Save_Ims
from morphodynamics.landscapes.analysis.save_sde_objs import SDE_Mixin




class Landscape_Model(SDE_Mixin):
    """
    The physics-informed neural network (PINN) architecture
    """


    def __init__(self, pdf_list, kde_bw,
                    xlims, ylims, tlims, dims, layers_p, layers_U, layers_D,
                    data_weight, pde_weight, BC_weight, norm_weight,
                    num_collocation, num_BC,
                    batch_size, learning_rate, save_append):
        """
        - pdf_list: data pdf list
        - kde_bw: bandwidth for the kernel density estimate (kde)
        - xlims, ylims, tlims: limits for x, y (spatial) and t (time)
        - dims: dimensions of the spatial grid
        - layers_p, layers_U, layers_D: layers for the three networks
        - data_weight, pde_weight, BC_weight, norm_weight: hyperparameters weighting the total loss, L_total
        - num_collocation: number of points for the PDE loss
        - num_BC: number of points for the boundary condition loss
        - batch_size, learning_rate
        - save_append: string with information when naming output files
        """

        self.pdf_list, self.kde_bw = pdf_list, kde_bw

        self.idx_save = None
        self.num_systems = 8

        self.xlims, self.ylims, self.tlims, self.dims = xlims, ylims, tlims, dims
        self.data_weight, self.pde_weight, self.BC_weight, self.norm_weight = data_weight, pde_weight, BC_weight, norm_weight
        self.num_collocation, self.num_BC = num_collocation, num_BC
        self.learning_rate, self.batch_size = learning_rate, batch_size
        self.save_append = save_append

        self.pdf_datasets = None
        self.BC_dataset = None
        self.PDE_dataset = None


        self._make_data()
        p_net_means, p_net_stds, D_net_means, D_net_stds, U_net_means, U_net_stds = self._get_normalisation_stats() # for normalizing the network inputs

        self.net_p = Residual_Net(layers_p, means = p_net_means, stds = p_net_stds, final_act = 'softplus', sigMult = None)
        self.net_U = Residual_Net(layers_U, means = U_net_means, stds = U_net_stds, final_act = 'sigmoid', sigMult = 3)
        self.net_D = Residual_Net(layer_dims = layers_D, means = D_net_means, stds = D_net_stds, final_act = 'sigmoid', sigMult = 1)

        self.net_p.build((None, 3))
        self.net_D.build((None, 3))
        self.net_U.build((None, 2))

        self.MSE = tf.keras.losses.MeanSquaredError()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        self.total_losses, self.data_losses, self.pde_losses, self.BC_losses,  self.norm_losses = [], [], [], [], []

        self.iterations = 0

    def _make_data(self):
        """
        Creat datasets so random points can be chosen for each loss component
        """
        pdf_datasets_all = make_pdf_datasets(self.pdf_list, self.xlims, self.ylims, self.tlims, self.dims, 9)
        self.pdf_dataset = np.concatenate(pdf_datasets_all, axis = 0)
        self.PDE_dataset = make_PDE_dataset(self.num_collocation, self.xlims, self.ylims, self.tlims, self.dims)
        self.BC_dataset = make_BC_dataset(self.num_BC, self.xlims, self.ylims, self.tlims, self.dims)


    def _get_normalisation_stats(self):
        """
        For normalizing the network inputs
        """
        p_net_datasets =  [self.pdf_dataset] + [self.PDE_dataset] + [self.BC_dataset]
        p_net_means, p_net_stds = get_mean_std_from_datasets(p_net_datasets)

        D_net_datasets = [self.PDE_dataset]
        D_net_means, D_net_stds = get_mean_std_from_datasets(D_net_datasets)

        U_net_datasets = [self.PDE_dataset]
        U_net_means, U_net_stds = get_mean_std_from_datasets(U_net_datasets)

        return p_net_means, p_net_stds, D_net_means, D_net_stds, U_net_means, U_net_stds


    def train(self, total_time):
        """
        Train the PINN for a specified amount of time
        """

        start_time = time.time()
        time_so_far = 0

        while time_so_far < total_time:

            with tf.GradientTape(persistent=True) as tape:

                # DATA LOSS
                data_idxs_batch = np.random.choice(self.pdf_dataset.shape[0], self.batch_size)
                data_batch = self.pdf_dataset[data_idxs_batch, :]
                data_batch = tf.Variable(data_batch, dtype = tf.float32)
                data_xyts = data_batch[:, :3]
                data_target = data_batch[:, 3:4]
                data_p_out = self.net_p(data_xyts)

                data_loss = self.data_weight*self.MSE(data_p_out, data_target)


                # PDE LOSS
                idxs_batch = np.random.choice(self.PDE_dataset.shape[0], self.batch_size)
                pde_batch = self.PDE_dataset[idxs_batch, :]
                pde_batch = tf.Variable(pde_batch, dtype = tf.float32)
                residual = FP_2D(self.net_p, self.net_D, self.net_U, pde_batch, tape)

                target = tf.zeros(residual.shape, dtype = tf.float32)
                pde_loss = self.pde_weight*self.MSE(residual, target)


                # BC LOSS
                idxs_batch = np.random.choice(self.BC_dataset.shape[0], self.batch_size)
                BC_batch = self.BC_dataset[idxs_batch, :]
                BC_batch = tf.Variable(BC_batch, dtype = tf.float32)
                p_out = self.net_p(BC_batch)
                target = tf.fill(p_out[:, 0:1].shape, np.float32(0))
                BC_loss = self.BC_weight*self.MSE(p_out, target)


                # NORMALIZING LOSS
                segment_area = (self.xlims[1] - self.xlims[0])*(self.ylims[1] - self.ylims[0])/(self.dims**2)
                xyts = get_random_norm_slice(self.xlims, self.ylims, self.tlims, self.dims)
                xyts = tf.Variable(xyts, dtype = tf.float32)
                p_out = self.net_p(xyts)
                pdf_integral = segment_area*tf.math.reduce_sum(p_out)
                norm_loss = self.norm_weight*(pdf_integral - 1.)**2


                total_loss = data_loss + pde_loss + BC_loss + norm_loss # note the weightings are applied before this


            trainables = self.net_p.trainable_variables + self.net_D.trainable_variables + self.net_U.trainable_variables

            grads = tape.gradient(total_loss, trainables)

            del tape

            self.optimizer.apply_gradients(zip(grads, trainables))


            time_so_far_prev = time_so_far
            time_so_far = (time.time() - start_time)/3600.

            if self.iterations % 10 == 0:

                self.total_losses.append(total_loss.numpy())
                self.data_losses.append(data_loss.numpy())
                self.pde_losses.append(pde_loss.numpy())
                self.BC_losses.append(BC_loss.numpy())
                self.norm_losses.append(norm_loss)

                tf.print('It: %d, Total loss: %.3e, Data loss: %.3e, PDE loss: %.3e, BC loss: %.3e,  norm_loss: %.3e, Time: %.2fh' \
                    % (self.iterations, total_loss, data_loss, pde_loss, BC_loss,  norm_loss, time_so_far))
                sys.stdout.flush()


            self.iterations += 1


    def predict(self, xyts_test):
        """
        Run inference on a set of points, xyts_test
        """

        p_out = self.net_p(xyts_test, training = False)
        D_out = self.net_D(xyts_test, training = False)
        U_out = self.net_U(xyts_test[:, :2], training = False)

        return p_out, D_out, U_out


    def save_networks(self, dir_weights):
        """
        Save the network weights and biases
        """

        path_p = dir_weights + 'p_{}.ckpt'
        path_D = dir_weights + 'D_{}.ckpt'
        path_U = dir_weights + 'U_{}.ckpt'

        self.net_p.save_weights(path_p.format(self.idx_save))
        self.net_D.save_weights(path_D.format(self.idx_save))
        self.net_U.save_weights(path_U.format(self.idx_save))



    def load_networks(self, dir_weights, idx_load):
        """
        Load the network weights and biases
        """

        path_p = dir_weights + 'p_{}.ckpt'
        path_D = dir_weights + 'D_{}.ckpt'
        path_U = dir_weights + 'U_{}.ckpt'

        self.net_p.load_weights(path_p.format(idx_load))
        self.net_D.load_weights(path_D.format(idx_load))
        self.net_U.load_weights(path_U.format(idx_load))


    def save_ims(self, save_dir):
        """
        Save graph of losses changing during training
        """
        save_ims = Save_Ims(model = self, save_dir = save_dir)
        save_ims()
