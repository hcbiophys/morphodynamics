import os
import sys
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import lognorm

from morphodynamics.morphospace.autoencoder import VAE
from morphodynamics.tip_model.theta.utils_theta import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# L0 ~ 15e-6 if using


class Morph_Sim():
    """
    Simulation used in ABC
    """

    def __init__(self, num_runs, L_params, data_points = None):
        """
        - num_runs: number of fungi simulated per comparison
        - L_params to use (MAP values)
        - data_points: data embeddings to compare simulations with
        """

        self.data_points = data_points

        self.num_runs = num_runs
        [self.t_germ_s, self.t_germ_loc, self.t_germ_scale, self.grad_s, self.grad_loc, self.grad_scale] = L_params

        self.VAE = None
        self.sim_ims = None
        self.sim_points = None

        self.ims_array = None

        self.img_transforms = torchvision.transforms.Compose(  [torchvision.transforms.ToTensor()] )


    def load_VAE(self):
        """
        Load the autoencoder
        """
        path_to_here = os.path.dirname(os.path.realpath(__file__))
        VAE_path = os.path.join(path_to_here, '../../data/network_weights/autoencoder/plusSynths_epoch7.pth.tar')
        VAE = VAE_Y3_Data(code_size=2, beta=0, learning_rate=1e-3, batch_size=32, lims_list=(-10, 70, -50, 55)).to(device)
        VAE.load(VAE_path)
        VAE.eval()
        self.VAE = VAE


    def _get_tgerm_grad(self):
        """
        Sample from the probability distributions associated with lengthening parameters to get a germination time & growth rate
        """
        t_germ = lognorm.rvs(s = self.t_germ_s, loc = self.t_germ_loc, scale = self.t_germ_scale, size = 1)

        grad = -1
        count = 0
        while grad < 0:
            grad = -lognorm.rvs(s = self.grad_s, loc = 0, scale = self.grad_scale, size = 1) + self.grad_loc
            count += 1

        return t_germ, grad


    def _xys_to_im(self, xs, ys, dLs):
        """
        Convert simulation trajectory to a fungus image
        """
        if ys[-1] > ys[0]:
            ys = [-i for i in ys]

        points = np.array([[int(200/5) +4+ int(i/(1.3e-6)), 100-int(j/(1.3e-6))] for i,j in zip(xs, ys)])

        if dLs[-1] == 0:
            im = to_fung_im(points, pixels_width = 5, isSpore = True) # pixels_width=5 gives actual pixel width of 7 (which is width of videos and snapshot ims) - accounts for bug in polylines
        else:
            im = to_fung_im(points, pixels_width = 5, isSpore = False) # pixels_width=5 gives actual pixel width of 7 (which is width of videos and snapshot ims) - accounts for bug in polylines

        return im



    def M0_set_sim_ims(self, sigma):
        """
        Run model 0 simulation forward to get images
        """
        print('M0;', 'sigma',  sigma)
        sys.stdout.flush()

        self.sim_ims = []

        for idx_run in range(self.num_runs):
            t_germ, grad = self._get_tgerm_grad()

            ts = [0]
            dLs = [0]
            thetas = [0]

            dt = 1
            for idx in range(210):
                if ts[idx] < t_germ:
                    dLs.append(0)
                    dtheta = 0
                else:
                    dLs.append(grad*1e-6*dt)
                    dtheta = sigma*np.sqrt(dt)*np.random.normal(0, 1, 1)[0]

                theta = thetas[idx] + dtheta
                thetas.append(theta)

                ts.append(ts[idx] + dt)

            xs = [0]
            ys = [0]

            x0, y0 = 0, 0
            for idx_step in range(len(dLs)):
                x1 = x0 + dLs[idx_step]*np.cos(dt*thetas[idx_step])
                y1 = y0 + dLs[idx_step]*np.sin(dt*thetas[idx_step])

                xs.append(x1)
                ys.append(y1)

                x0, y0 = x1, y1

            im = self._xys_to_im(xs, ys, dLs)



            self.sim_ims.append(im)


    def M1_set_sim_ims(self, sigma):
        """
        Run model 1 simulation forward to get images
        """
        print('M1;', 'sigma',  sigma)
        sys.stdout.flush()

        self.sim_ims = []

        for idx_run in range(self.num_runs):
            t_germ, grad = self._get_tgerm_grad()

            ts = [0]
            dLs = [0]
            thetas = [0]

            dt = 1
            for idx in range(210):
                if ts[idx] < t_germ:
                    dLs.append(0)
                    dtheta = 0
                else:
                    dLs.append(grad*1e-6*dt)
                    dtheta = sigma*np.sqrt(dt)*np.random.normal(0, 1, 1)[0]

                theta = thetas[idx] + dtheta
                thetas.append(theta)

                ts.append(ts[idx] + dt)

            xs = [0]
            ys = [0]

            x0, y0 = 0, 0
            for idx_step in range(len(dLs)):
                x1 = x0 + dLs[idx_step]*np.cos(grad*dt*np.cumsum(thetas[:idx_step+1])[-1])
                y1 = y0 + dLs[idx_step]*np.sin(grad*dt*np.cumsum(thetas[:idx_step+1])[-1])

                xs.append(x1)
                ys.append(y1)

                x0, y0 = x1, y1

            im = self._xys_to_im(xs, ys, dLs)


            self.sim_ims.append(im)



    def M2_set_sim_ims(self, sigma, inv_tau):
        """
        Run model 2 simulation forward to get images
        """
        print('M2;', 'sigma, inv_tau:',  sigma, inv_tau)
        sys.stdout.flush()

        self.sim_ims = []

        for idx_run in range(self.num_runs):

            t_germ, grad = self._get_tgerm_grad()

            ts = [0]
            dLs = [0]
            thetas = [0]

            dt = 1
            for idx in range(210):
                if ts[idx] < t_germ:
                    dLs.append(0)
                    dtheta = 0
                else:
                    dLs.append(grad*1e-6*dt)
                    dtheta = sigma*np.sqrt(dt)*np.random.normal(0, 1, 1)[0] -inv_tau*thetas[idx]*dt

                theta = thetas[idx] + dtheta
                thetas.append(theta)

                ts.append(ts[idx] + dt)

            xs = [0]
            ys = [0]

            x0, y0 = 0, 0
            for idx_step in range(len(dLs)):
                x1 = x0 + dLs[idx_step]*np.cos(grad*dt*np.cumsum(thetas[:idx_step+1])[-1])
                y1 = y0 + dLs[idx_step]*np.sin(grad*dt*np.cumsum(thetas[:idx_step+1])[-1])

                xs.append(x1)
                ys.append(y1)

                x0, y0 = x1, y1

            im = self._xys_to_im(xs, ys, dLs)

            self.sim_ims.append(im)



    def _standardize_coords(self, coords, orig_lims = (-10, 70, -50, 55)):
        """
        Convert embedding coordinates to the [-10, 10, -10, 10] range
        """

        [xmin, xmax, ymin, ymax] = orig_lims
        #x_new = -9 + 18*(coords[0]-xmin)/(xmax-xmin)
        #y_new = -9 + 18*(coords[1]-ymin)/(ymax-ymin)
        x_new = -10 + 20*(coords[0]-xmin)/(xmax-xmin)
        y_new = -10 + 20*(coords[1]-ymin)/(ymax-ymin)
        return [x_new, y_new]




    def set_sim_points(self):
        """
        Find the embeddings of the simulated images
        """
        self.sim_points = []
        xs = []
        ys = []

        batch_size = 50
        for idx_batch in range(int(len(self.sim_ims)/50)):
            ims_batch = self.sim_ims[idx_batch*50: idx_batch*50 + 50]
            ims_batch = [Image.fromarray(im) for im in ims_batch]
            ims_batch = [self.img_transforms(im) for im in ims_batch]
            #ims_batch = [torch.unsqueeze(im, 0) for im in ims_batch]
            batch = torch.stack(ims_batch, dim = 0).to(device)
            mu, logvar, recon_x = self.VAE(batch)
            mu = mu.squeeze().detach().cpu().numpy()
            mus = [mu[idx, :] for idx in range(mu.shape[0])]


            mus = [self._standardize_coords(mu) for mu in mus]
            self.sim_points += mus


        #plt.scatter([i[0] for i in self.sim_points], [i[1] for i in self.sim_points])
        #plt.savefig('/end/home/hc2814/Desktop/code_direct_outputs/sim_mus.png')
        #plt.close()


    def _set_data_ims_array(self, drug_name):
        """
        Set up the array with data images, ready to add simulations.
        """
        path_to_here = os.path.dirname(os.path.realpath(__file__))
        file = path_to_here+'/vis_arrays/{}.png'.format(drug_name)
        self.ims_array = cv2.imread(file)


    def add_simIms_toimsArray(self, drug_name, idx_model, idx_pop, lims_list = [-10, 10, -10, 10]):
        """
        Add simulation images to the array of data images.
        """
        array_dim = 2200
        [xmin, xmax, ymin, ymax] = lims_list
        xrange = xmax-xmin
        yrange = ymax-ymin
        self._set_data_ims_array(drug_name)

        for idx in range(len(self.sim_ims)):
            if idx < 300:
                im = self.sim_ims[idx]
                point = self.sim_points[idx]

                im*=255
                im = im.astype(np.uint8)
                contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



                x_start = int(((array_dim-200)/xrange)*(point[0] + abs(xmin)))
                y_start = int(((array_dim-200)/yrange)*(point[1] + abs(ymin)))

                contour = contours[0]
                contour[:, :, 0] += x_start
                #contour[:, :, 0] = np.mean(contour[:, :, 0]) - contour[:, :, 0] + np.mean(contour[:, :, 0]) # to flip
                contour[:, :, 1] += array_dim - y_start
                cv2.drawContours(self.ims_array, [contour], -1, (0, 0, 0), -1)

                if len(contours) > 1:
                    contour = contours[1]
                    contour[:, :, 0] += x_start
                    #contour[:, :, 0] = np.mean(contour[:, :, 0]) - contour[:, :, 0] + np.mean(contour[:, :, 0]) # to flip
                    contour[:, :, 1] += array_dim - y_start
                    cv2.drawContours(self.ims_array, [contour], -1, (255, 255, 255), -1)


        figMorphs = plt.figure(figsize = (70, 40))
        ax = figMorphs.add_subplot(1, 1, 1)
        ax.imshow(self.ims_array)

        path_to_here = os.path.dirname(os.path.realpath(__file__))
        plt.savefig(path_to_here+'/../../outputs/morphs_{}_model{}_pop{}.png'.format(drug_name, idx_model, idx_pop))




    def scatter(self, save_extra = '_'):
        """
        Scatter the data and simulation points
        """

        bins = 30
        vmin = 0
        vmax = 1

        figScatt = plt.figure(figsize=(70, 40))
        for time in range(9):
            ax = figScatt.add_subplot(4, 9, time+1)
            xs = [i[0] for i in self.data_points[time]]
            ys = [i[1] for i in self.data_points[time]]
            ax.scatter(xs, ys, s = 2, c = 'white', marker = 'o')
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_aspect('equal')
            ax.set_facecolor((0, 0, 0))
            ax = figScatt.add_subplot(4, 9, 9+time+1)
            h_data, _, _ = np.histogram2d(xs, ys, bins = bins, range = [[-10, 10], [-10, 10]], normed = True)
            plot_h_data = np.rot90(h_data)
            im = ax.imshow(np.sqrt(plot_h_data), vmin = vmin, vmax = vmax)
            ax = figScatt.add_subplot(4, 9, 18+time+1)

            xs = [i[0] for i in self.data_points[time]]
            ys = [i[1] for i in self.data_points[time]]
            ax.scatter(xs, ys, s = 2, c = 'white', marker = 'o')
            xs = [i[0] for i in self.sim_points[time]]
            ys = [i[1] for i in self.sim_points[time]]
            ax.scatter(xs, ys, s = 2, marker = 'o', c = 'red')
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_aspect('equal')
            ax.set_facecolor((0, 0, 0))

            ax = figScatt.add_subplot(4, 9, 27+time+1)
            h_sim, _, _ = np.histogram2d(xs, ys, bins = bins, range = [[-10, 10], [-10, 10]], normed = True)
            plot_h_sim = np.rot90(h_sim)
            ax.imshow(np.sqrt(plot_h_sim), vmin = vmin, vmax = vmax)

        #figScatt.colorbar(im)
        for ax in figScatt.axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.subplots_adjust(wspace = 0.0, hspace = 0.0)
        path_to_here = os.path.dirname(os.path.realpath(__file__))
        plt.savefig(path_to_here+'../../outputs/scatter_' + save_extra + '.png')
        plt.close()
