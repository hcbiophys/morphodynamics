import argparse
import os
import sys
import glob
import time
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2


from morphodynamics.morphospace.autoencoder import VAE
from morphodynamics.morphospace.dataset import Dataset
from morphodynamics.morphospace.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Implement_Autoencoder():
    """
    Class for operations using the VAE class.
    """


    def __init__(self, code_size, beta, lr, batch_size, lims_list, save_path):
        """
        Args
        - code_size: dimensionality of the morphospace
        - beta: weighting of the Kullback-Leibler divergence in the VAE loss (this is zero for a plain autoencoder)
        - lr: learning rate
        - batch_size
        - save_path: path to save visualizations to. Must include '{}' to be formatted with visualization-specific string
        """

        self.code_size = code_size
        self.beta = beta
        self.lr = lr
        self.batch_size = batch_size
        self.save_path = save_path

        self.VAE = None

        self.img_transforms = torchvision.transforms.Compose(  [torchvision.transforms.ToTensor()] ) # convert images to PyTorch tensor objects

        self.lims_list = lims_list


    def load_model(self, path_weights):
        """
        Load the neural network state_dict from disk.
        """

        self.VAE = VAE(self.code_size, self.beta, self.lr, self.batch_size, self.lims_list).to(device)
        self.VAE.load(path_weights)


    def graphic_1snap_drug(self, drug_name):
        """
        Graphic of 1 drug colored by time.
        """

        dataset = Dataset(self.img_transforms, RGB = False)
        dataset.keep_oneDrug_timeLabelled(drug_name)
        drug_dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)


        self.VAE.harvest_points(drug_dataloader)

        ims_array= self.VAE.graphic_drugOrtime_1snap(which_colors = 'drug_colors', save_path = self.save_path)




    def graphic_1snap_time(self, time0to8):
        """
        Graphic of 1 time colored by drug.
        """

        dataset = Dataset(self.img_transforms, RGB = False)
        dataset.keep_oneTime_drugLabelled(time0to8)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        self.VAE.harvest_points(dataloader)
        ims_array = self.VAE.graphic_drugOrtime_1snap(which_colors = 'drug_colors', save_path = self.save_path)



    def scatter_1drug_allSnaps(self, drug, idx_drug):
        """
        Scatter over multiple axes a drug's embeddings evolving.
        """

        dataset = Dataset(self.img_transforms, RGB = False)
        dataset.keep_oneDrug_timeLabelled(drug)
        drug_dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        self.VAE.harvest_points(drug_dataloader)
        self.VAE.scatter_1drug_allSnaps(drug, idx_drug, save_path = self.save_path)



    def set_data_limits_all(self):
        """
        Find the minimum and maximum along each morphospace dimension (for plotting visualizations)
        """
        dataset = Dataset(self.img_transforms, RGB = False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        self.VAE.set_data_limits(dataloader)



if __name__ == '__main__':

    """
    # set morphospace limits (for morphospace visualizations) by finding the minimum and maximum embeddings along each axis; otherwise, just use manually-set limits, as below
    lims_dataset = Dataset(implement_autoencoder.img_transforms, RGB = False)
    lims_dataloader = torch.utils.data.DataLoader(lims_dataset, batch_size = 32, shuffle = True)
    implement_autoencoder.VAE.set_data_limits(lims_dataloader)
    print(implement_autoencoder.VAE.data_limits)
    sys.exit()
    """

    lims_list = (-10, 70, -50, 55) # limits are normalized for the landscape model so each morphospace axis has limits [-10, 10]

    # to train ... (pre-trained weights are provided, loaded below)
    #implement_autoencoder.train_and_save_model(num_epochs_train=10, add_videos = True, add_synths = True, remove_90percent_spores=True)

    # load autoecoder weights
    path_to_here = os.path.dirname(os.path.realpath(__file__))
    path_weights = os.path.join(path_to_here, '../data/network_weights/autoencoder/epoch5.pth.tar')

    implement_autoencoder = Implement_Autoencoder(code_size = 2, beta = 0, lr = 1e-4, batch_size = 50, lims_list = lims_list, save_path = path_to_here+'/../outputs/{}.png')
    implement_autoencoder.load_model(path_weights = path_weights)

    # visualizations
    implement_autoencoder.VAE.code_projections_2D((200, 200), save_path = implement_autoencoder.save_path, proj_sampling_rate = 15)
    implement_autoencoder.graphic_1snap_time(8)
    #implement_autoencoder.scatter_1drug_allSnaps(drug = 'compound_X', idx_drug = 2) # idxs are order in drugnames list attribute of Dataset
