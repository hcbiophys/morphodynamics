import os
import sys
import time
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import cv2
import itertools
import scipy.stats
import scipy.stats as stats

from morphodynamics.morphospace.visualizations import Visualizations_Mixin
from morphodynamics.morphospace.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAE(torch.nn.Module, Visualizations_Mixin):
    """
    Variational Autoencoder (with beta = 0 this becomes a regular autoencoder, as used in the paper)
    """

    def __init__(self, code_size, beta, learning_rate, batch_size, lims_list):
        """
        - code_size
        - beta: KLD weighting in loss
        - learning_rate
        - batch_size
        - lims_list: minimum and maximum along each morphospace dimension for visualizations; [xmin, xmax, ymin, ymax]
        """
        super(VAE, self).__init__()

        self.learning_rate = learning_rate
        self.code_size = code_size
        self.batch_size = batch_size
        self.beta = beta

        self.lims_list = lims_list
        self.data_limits = None

        self.num_epochs_trained = 0


        self.input_dims = [200, 200]

        act_func = torch.nn.ReLU()
        final_act = torch.nn.Sigmoid()

        conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        bn1 = torch.nn.BatchNorm2d(16)
        conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        bn2 = torch.nn.BatchNorm2d(32)
        conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        bn3 = torch.nn.BatchNorm2d(64)
        conv4 = torch.nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1)
        bn4 = torch.nn.BatchNorm2d(16)


        self.encoder = torch.nn.Sequential(conv1, bn1, act_func, conv2, bn2, act_func, conv3, bn3, act_func, conv4, bn4, act_func, Flatten())
        self.encoder.apply(self.init_weights)

        conv5 = torch.nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        bn5 = torch.nn.BatchNorm2d(64)
        conv6 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        bn6 = torch.nn.BatchNorm2d(32)
        conv7 = torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        bn7 = torch.nn.BatchNorm2d(16)
        conv8 = torch.nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)

        self.decoder = torch.nn.Sequential(UnFlatten(16, 50, 50), conv5, bn5, act_func, conv6, bn6, act_func, conv7, bn7, act_func, conv8, final_act)
        self.decoder.apply(self.init_weights)

        self.fc11 = torch.nn.Linear(40000, self.code_size)
        self.fc12 = torch.nn.Linear(40000, self.code_size)
        self.fc2 = torch.nn.Linear(self.code_size, 40000)
        for i in [self.fc11, self.fc12, self.fc2]:
            i.apply(self.init_weights)

        self.points = None


    def init_weights(self, m):
        """
        Xavier initialization
        """
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def set_data_limits(self, dataloader):
        """
        Find and set as attribute the morphospace limits, for visualizations
        """

        self.eval()

        xs, ys = [], []
        for i, tup in enumerate(dataloader):
            if i % 100 == 0:
                print('batch {}'.format(i))
                sys.stdout.flush()
            paths, batch, labels = tup[0], tup[1], tup[2]
            batch = batch.type('torch.FloatTensor').to(device)
            labels = labels.to(device)

            intermediate = batch

            intermediate = self.encoder(intermediate)
            mu_ = self.fc11(intermediate)
            mu_ = mu_.data.cpu().numpy()

            labels_ = labels.data.cpu().numpy()
            for idx in range(mu_.shape[0]):
                xs.append(mu_[idx, :][0])
                ys.append(mu_[idx, :][1])

        self.data_limits = (min(xs), max(xs), min(ys), max(ys))

    def _reparametrize(self, mu, logvar):
        """
        Central reparametrization
        Args:
        - mu: mean of stochastic embedding.
        - logvar: logarithm variable.
        """

        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(device) # default mean 0, std 1
        return eps.mul(std).add_(mu)


    def forward(self, x):
        """
        Propagate batch
        Args:
        - x: batch to flow through.
        """
        x = self.encoder(x)

        mu = self.fc11(x)
        logvar = self.fc12(x)
        x = self._reparametrize(mu, logvar)

        x = self.fc2(x)

        recon_x = self.decoder(x)

        return mu, logvar, recon_x

    def _custom_loss(self, recon_x, x, mu, logvar):
        """
        Loss is reconstruction and weighted KLD. The latter is difference between unit Gaussian
        and each point's stochastic (and also Gaussian) embedding.
        Args:
        - recon_x: reconstuction
        - x: input data
        - mu: mean of stochastic embedding.
        - logvar: log variable of stochastic embedding.
        """

        #bce = F.binary_cross_entropy(torch.add(recon_x, 1e-10), x, size_average=False)
        bce = F.binary_cross_entropy(recon_x, x, size_average=False)
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        loss = bce + self.beta*kld.sum()

        bce /= x.shape[0]
        kld_loss = self.beta*kld.sum()/x.shape[0]
        loss /= x.shape[0]

        return bce, kld_loss, loss


    def train_oneEpoch(self, dataloader, save_recons_path = None):
        """
        Train the model for one epoch.
        Args
        - dataloader: PyTorch dataloader object
        - save_recons_path: where to save reconstructions each epoch
        """

        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)


        MSE_losses = []
        KLD_losses = []

        for i, tup in enumerate(dataloader):
            start_time = time.time()

            if i % 100 == 0:
                print('batch {}'.format(i))
                sys.stdout.flush()

            _, batch, _ = tup[0], tup[1], tup[2]


            batch = batch.type('torch.FloatTensor').to(device)

            # forward
            mu, logvar, x = self(batch)

            if save_recons_path is not None:
                # visualise some reconstructions
                if i == 10:

                    fig, axarr = plt.subplots(4, 4, figsize=(70, 40))
                    count = 0
                    for row_start in [0, 2]:
                        for col in range(4):
                            axarr[row_start, col].imshow(batch.detach().cpu().view(batch.size(0), self.input_dims[0], self.input_dims[1])[count + col, :], cmap = 'gray')
                            axarr[row_start, col].set_title('{}_{}'.format(torch.min(batch).item(), torch.max(batch).item()))
                            axarr[row_start + 1, col].imshow(x.detach().cpu().view(x.size(0), self.input_dims[0], self.input_dims[1])[count + col, :]) # detach as can't call numpy() on requires_grad var
                            axarr[row_start + 1, col].set_title('{}_{}'.format(torch.min(x).item(), torch.max(x).item()))
                        count += 4
                    plt.axis('off')
                    plt.savefig(save_recons_path.format(self.num_epochs_trained))
                    plt.close(fig)


            MSE, beta_KLD, loss = self._custom_loss(x, batch, mu, logvar)

            MSE_losses.append(MSE.item())
            KLD_losses.append(beta_KLD.item())

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {}, MSE_loss: {}, KLD_loss: {}, Loss: {}'.format(self.num_epochs_trained, MSE, beta_KLD, loss.item()))
        self.num_epochs_trained += 1


    def harvest_points(self, dataloader):
        """
        Save embeddings as a class attribute (self.points).
        Args
        - dataloader: PyTorch dataloader object to be modelled.
        """

        self.eval()

        points = []
        for i, tup in enumerate(dataloader):
            if i % 100 == 0:
                print('batch {}'.format(i))
                sys.stdout.flush()

            paths, batch, labels = tup[0], tup[1], tup[2]
            batch = batch.type('torch.FloatTensor').to(device)
            labels = labels.to(device)

            batch = self.encoder(batch)
            mu_ = self.fc11(batch)
            mu_ = mu_.data.cpu().numpy()

            labels_ = labels.data.cpu().numpy()
            for idx in range(mu_.shape[0]):
                points.append([mu_[idx, :], labels_[idx], paths[idx]])

        self.points = points




    def save(self, save_path):
        """
        Save the model's state_dict under arg: save_path.
        """
        torch.save(self.state_dict(), save_path)


    def load(self, path):
        """
        Load the model's state_dict from arg: path.
        """
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))
