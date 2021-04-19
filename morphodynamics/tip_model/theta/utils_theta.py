import os
import sys
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
import torch
import torchvision
sys.path.append('/end/home/hc2814/Desktop/code/custom_dataset_classes')
from custom_dataset_classes.datasets_y3 import Custom_Dataset_Y3
sys.path.append('/end/home/hc2814/Desktop/code/NN_scripts/unsupervised/VAE')
from unsupervised.VAE.y3.VAEs_y3 import VAE_Y3_Data
import pickle
from sklearn.neighbors import KernelDensity



def to_fung_im(points, pixels_width, isSpore):
    """
    Convert simulation points to image
    Args
    - points: list of points
    - r: radius in pixels (?)
    """
    im = np.zeros((200, 200))
    if not isSpore:
        cv2.polylines(im, [points], False, (1, 1, 1), thickness = pixels_width) # because torchvision totensor() puts real data in [0,1] range then dataloader into just over [0, 0.5] range
        #cv2.polylines(im, [points], False, (0.5098, 0.5098, 0.5098), thickness = int(2*r/(1.3e-6)))


    cv2.circle(im, (int(im.shape[1]/5)-5, int(im.shape[0]/2)), 8, (1, 1, 1), -1)
    #cv2.circle(im, (int(im.shape[1]/5)-5, int(im.shape[0]/2)), 8, (0.5098, 0.5098, 0.5098), -1)
    return im

def distance_abs(x_dict, y_dict):

    bins = 20

    x = x_dict['X_2']
    y = y_dict['X_2']

    data_hist, _, _ = np.histogram2d([i[0] for i in x], [i[1] for i in x], bins = bins, range = [[-10, 10], [-10, 10]], density = True)
    sim_hist, _, _ = np.histogram2d([i[0] for i in y], [i[1] for i in y], bins = bins, range = [[-10, 10], [-10, 10]], density = True)

    dist = 0
    for row in range(data_hist.shape[0]):
        for col in range(data_hist.shape[1]):
            a = data_hist[row, col]
            b = sim_hist[row, col]
            dist += abs(a - b)

    print('Distance: ', dist)
    sys.stdout.flush()
    return dist
