import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from mayavi import mlab
#mlab.options.offscreen=True
import cv2
from matplotlib import colors
import matplotlib.gridspec as gridspec
import os
import copy
from scipy.ndimage import gaussian_filter
import random


drug_name = sys.argv[1]
to_plot = sys.argv[2]


xlims = [-13, 13]
ylims = [-13, 13]

path_to_here = os.path.dirname(os.path.realpath(__file__))
path_vis_data = os.path.join(path_to_here, '../data/landscape_visualizations/')
pList_data = os.path.join(path_to_here, '../data/embeddings/lims13/lims13_{}.pickle'.format(drug_name))


path_SDE_objects = path_vis_data + '{}/original/'.format(drug_name)
if drug_name == 'DMSO':
    path_SDE_objects = path_vis_data + '{}/original/30_hours/'.format(drug_name) # path with all the objects for plotting, after running inference on the PINN


UArray = path_SDE_objects + 'U.pickle' # potential array
trajectories_subsampled = path_SDE_objects + 'subsampled_paths_p0.pickle' # subsampled paths of eq. 1
pList_NN = path_SDE_objects +  'p_list_0.pickle' # pdf generated by PINN at timepoints
DMSO_pList_NN = path_vis_data + 'DMSO/original/30_hours/p_list_0.pickle' # same as above but for DMSO


div_bys = {'DMSO': 4, 'compound_A': 4, 'compound_B': 1, 'compound_C_0_041': 1, 'compound_C_10': 1, 'compound_X': 1} # how much to scale the outer regions, for visualising (1 is not at all)




class Visualisations():
    """
    A class for visualizations of landscapes and errors through comparing data with eq. 1 simulations
    """

    def __init__(self, xlims, ylims,
                    pList_data, pList_NN, DMSO_pList_NN,
                    UArray,
                    trajectories_subsampled):
        """
        - xlims, ylims: limits for plotting
        - pList_data: data pdfs
        - pList_NN: PINN pdfs
        - DMSO_pList_NN: PINN pdfs for DMSO
        - UArray: potential as an array
        - trajectories_subsampled: subsampled simulations of eq.1 (subsampled to reduce memory)
        """

        self.resize_size = (1000, 1000)

        self.xlims, self.ylims = xlims, ylims

        self.pList_data = [np.reshape(i, (200, 200)) for i in pickle.load(open(pList_data, 'rb'))]
        self.pList_data = [cv2.resize(i, self.resize_size, interpolation = cv2.INTER_LINEAR) for i in self.pList_data]

        self.pList_NN = pickle.load(open(pList_NN, 'rb'))
        self.pList_NN = [cv2.resize(i, self.resize_size, interpolation = cv2.INTER_LINEAR) for i in self.pList_NN]

        self.DMSO_pList_NN = pickle.load(open(DMSO_pList_NN, 'rb'))
        self.DMSO_pList_NN = [cv2.resize(i, self.resize_size, interpolation = cv2.INTER_LINEAR) for i in self.DMSO_pList_NN]

        self.UArray = pickle.load(open(UArray, 'rb')).reshape((200, 200))
        self.UArray = cv2.resize(self.UArray, self.resize_size, interpolation = cv2.INTER_LINEAR)

        self.trajectories_subsampled = pickle.load(open(trajectories_subsampled, 'rb'))
        random.shuffle(self.trajectories_subsampled)

    def _get_and_squeeze_largest_contour(self, im):
        """
        Removes a dimension of a contour so it's more usable.
        """
        contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cont) for cont in contours]
        contour = contours[areas.index(max(areas))]
        contour = np.squeeze(contour, axis = 1)
        return contour

    def _get_DMSO_and_drug_outlines(self):
        """
        Get outlines for DMSO and other drug where PDF falls below 1e-3
        """
        thresh = 1e-3
        minimum, divBy = -13, self.resize_size[0]/26

        DMSO_im = self.DMSO_pList_NN[-1]
        DMSO_im[DMSO_im<thresh] = 0
        DMSO_im[DMSO_im>thresh] = 255
        DMSO_im = DMSO_im.astype(np.uint8)
        DMSO_contour = self._get_and_squeeze_largest_contour(DMSO_im)
        DMSO_contour = [minimum+np.array(i)/divBy for i in DMSO_contour] # image size is 1000 therefore divide
        DMSO_contour.append(DMSO_contour[0])

        drug_im = self.pList_NN[-1]
        drug_im[drug_im<thresh] = 0
        drug_im[drug_im>thresh] = 255
        drug_im = drug_im.astype(np.uint8)
        drug_contour = self._get_and_squeeze_largest_contour(drug_im)
        drug_contour = [minimum+np.array(i)/divBy for i in drug_contour]
        drug_contour.append(drug_contour[0])

        return DMSO_contour, drug_contour


    def _points_to_xyzs(self, points, array):
        """
        Converts 2D points into lists of x, y & z values (z being height of the potential)
        """
        xs = [i[0] for i in points]
        ys = [i[1] for i in points]

        cols = [self.resize_size[0]*(x-self.xlims[0])/(self.xlims[1]-self.xlims[0]) for x in xs]
        rows = [self.resize_size[0]*(y-self.ylims[0])/(self.ylims[1]-self.ylims[0]) for y in ys]
        zs = []
        for idx in range(len(xs)):
            zs.append(0.1 + 10*array[int(rows[idx]), int(cols[idx])])


        return xs, ys, zs


    def _outer_split(self, div_by = 4):
        """
        Split U into and inner (pdf > 1e-3) and outer (pdf < 1e-3) section, for transparency in low pdf regions
        """

        mask = np.zeros_like(self.pList_NN[-1])
        mask[self.pList_NN[-1] > 1e-3] = 1

        outer = copy.deepcopy(self.UArray)

        mask2 = np.zeros_like(mask)

        self.UArray[mask == 0] = np.nan
        outer[mask == 1] = np.nan

        for col in range(outer.shape[1]):
            for row in range(outer.shape[0]):
                if mask[row, col] == 1:
                    outer[:row, col] = self.UArray[row, col] + (outer[:row, col] - self.UArray[row, col])/div_by
                    for row2 in range(row, outer.shape[1]):
                        if mask[row2, col] == 0:
                            outer[row2:, col] = self.UArray[row2-1, col] + (outer[row2:, col] - self.UArray[row2-1, col])/div_by
                            break
                    break

        for col in range(outer.shape[1]):
            if np.nansum(self.UArray[:, col]) != 0:
                outer[:, :col] = np.nanmean(self.UArray[:, col]) + (outer[:, :col] - np.nanmean(self.UArray[:, col]))/div_by
                break

        for col in reversed(range(outer.shape[1])):
            if np.nansum(self.UArray[:, col]) != 0:
                outer[:, col:] = np.nanmean(self.UArray[:, col]) + (outer[:, col:] - np.nanmean(self.UArray[:, col]))/div_by
                break

        return outer



    def _pad_mask(self, mask):
        """
        Add a single pixel thick layer to a mask
        """
        mask2 = np.zeros_like(mask)
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if row != 0 and col != 0 and row != mask.shape[0]-1 and col != mask.shape[1]-1:
                    if mask[row, col]==1 or mask[row-1, col]==1 or mask[row+1, col]==1 or mask[row, col-1]==1 or mask[row, col+1]==1:
                        mask2[row, col] = 1
        return mask2

    def _get_G(self, array, log):
        """
        find the gradient magnitude of an array
        """
        Gx, Gy = np.gradient(array, 26./self.resize_size[0], 26./self.resize_size[0]) # gradients with respect to x and y
        G = (Gx**2+Gy**2)**.5  # gradient magnitude
        if log:
            G = np.log(G)
        G[np.isinf(G)] = np.nan
        return G

    def _from_13to10_border(self, array):
        """
        make the border region from 10->13 and -10>-13 np.nan, so still consistent with the x,y grid.
        """
        resolution = int(self.resize_size[0]/26)
        array = array[3*resolution:23*resolution, :]
        array = array[:, 3*resolution:23*resolution]

        return array



    def gradient_colored(self):
        """
        plot the landscape, coloured by gradient.
        """


        figU = mlab.figure(size=(5000, 5000), bgcolor = (1, 1, 1))

        x = np.linspace(self.xlims[0], self.xlims[1], self.resize_size[0])
        y = np.linspace(self.ylims[0], self.ylims[1], self.resize_size[0])
        xg, yg = np.meshgrid(x, y)



        outer_init = self._outer_split(div_by = div_bys[drug_name])
        U_init = self.UArray
        self.combined = np.nan_to_num(outer_init) + np.nan_to_num(U_init)
        self.combined = gaussian_filter(self.combined, sigma=10)
        mask = np.zeros_like(outer_init)
        mask[~np.isnan(outer_init)] = 1
        mask2 = self._pad_mask(mask)

        outer, self.UArray = copy.deepcopy(self.combined), copy.deepcopy(self.combined)
        outer[np.isnan(mask2)] = np.nan

        self.UArray = self._from_13to10_border(self.UArray)
        outer = self._from_13to10_border(outer)
        xg = self._from_13to10_border(xg)
        yg = self._from_13to10_border(yg)
        U_init = self._from_13to10_border(U_init)

        vmin = -5
        vmax = -2


        mask_plot = np.zeros_like(self.UArray).astype(bool)
        mask_plot[np.isnan(U_init)] = True



        mesh = mlab.mesh(xg, yg, 10*self.UArray, scalars = self._get_G(self.UArray, log=True), mask = mask_plot, figure = figU, colormap = 'jet', opacity = 1, vmin = vmin, vmax = vmax)
        mesh.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
        mesh = mlab.mesh(xg, yg, 10*outer, scalars = self._get_G(outer, log=True), figure = figU, colormap = 'jet', opacity = 0.3, vmin = vmin, vmax = vmax)


        mesh.actor.property.lighting = False

        if self.trajectories_subsampled is not None:

            DMSO_contour, drug_contour = self._get_DMSO_and_drug_outlines()

            xs, ys, zs = self._points_to_xyzs(drug_contour, array = self.combined)
            mlab.plot3d(xs, ys, zs, line_width = 8, figure = figU,  tube_radius=None, color = (1, 1, 1))

            xs, ys, zs = self._points_to_xyzs(DMSO_contour, array = self.combined)
            mlab.plot3d(xs, ys, zs, line_width = 8, figure = figU,  tube_radius=None, color = (0, 0, 0))

            count = 0
            for idx_traj, trajectory in enumerate(self.trajectories_subsampled):
                xs, ys, zs = self._points_to_xyzs(trajectory, array = self.combined)
                #mlab.plot3d(xs[::60], ys[::60], zs[::60], line_width = 3, figure = figU,  tube_radius=None, color = (1, 1, 1))
                mlab.plot3d(xs, ys, zs, line_width = 3, figure = figU,  tube_radius=None, color = (1, 1, 1))
                mlab.points3d(0.3+xs[0], ys[0], 0.3+zs[0], scale_factor = 0.7, color = (1, 0.4, 1), figure = figU)
                mlab.points3d(0.3+xs[-1], ys[-1], 0.3+zs[-1], scale_factor = 0.7, color = (0, 0.9, 0.9), figure = figU)
                count += 1
                if count > 7: # plot 7 random trajectories
                    break


        mlab.colorbar(orientation = 'vertical', nb_labels = 4)
        mlab.view(azimuth = 0, elevation = 30, distance = 70)

        mlab.savefig(path_to_here+'/../outputs/'+'quasipotential_{}.png'.format(drug_name), size = (100, 100))
        #mlab.show()
        #plt.show()


    def plot_sde_errors(self):
        """
        Plot the errors by comparing data with eq. 1 simulations
        """

        kdes_data_2 = open(path_SDE_objects + 'kdes_data_2.pickle', 'rb')
        kdes_data_2 = [np.reshape(i, (200, 200)) for i in pickle.load(kdes_data_2)]

        kdes_nn = open(path_SDE_objects + 'kdes_nn.pickle', 'rb')
        kdes_nn = [np.reshape(i, (200, 200)) for i in pickle.load(kdes_nn)]

        errors = [np.subtract(kdes_nn[idx], kdes_data_2[idx]) for idx in range(len(kdes_nn))]
        errors_max_abs = max([np.max(np.abs(i)) for i in errors])

        kde_max = max([np.max(i) for i in kdes_data_2+kdes_nn])
        log_kde_max = np.log10(kde_max)
        log_clip = -5


        num_cols = len(kdes_data_2)+1
        fig = plt.figure(figsize = (7, 2))
        gs = gridspec.GridSpec(3, num_cols)
        gs.update(wspace=0, hspace=0)

        for i in range(len(kdes_nn)):
            kdes_data_2[i] = np.clip(kdes_data_2[i], 10**log_clip, 100)
            kdes_nn[i] = np.clip(kdes_nn[i], 10**log_clip, 100)

        for idx in range(len(kdes_data_2)):
            ax1 = fig.add_subplot(gs[1+idx])
            ax2 = fig.add_subplot(gs[1+num_cols+idx])
            ax3 = fig.add_subplot(gs[1+2*num_cols+idx], aspect = 1)

            c1 = ax1.imshow(np.log10(kdes_data_2[idx])[::-1, :],  vmin = log_clip, vmax = log_kde_max, cmap = 'jet')
            ax2.imshow(np.log10(kdes_nn[idx])[::-1, :], vmin = log_clip, vmax = log_kde_max, cmap = 'jet')
            x = np.linspace(self.xlims[0], self.xlims[1], 200)
            y = np.linspace(self.ylims[0], self.ylims[1], 200)
            xg, yg = np.meshgrid(x, y)

            ax3.pcolormesh(x, y, errors[idx],
                                   norm=colors.SymLogNorm(linthresh=10**-2.5, linscale=0.000001, # 10**-2.72 is single particle
                                                          vmin=-0.2, vmax=0.2, base = 10),
                                   cmap='PiYG', shading = 'auto')


        ax = fig.add_subplot(gs[0])
        ax.set_visible(False)
        c2 = ax.imshow(np.array([[-1, 0, 1]]), cmap = 'PiYG')
        for ax in fig.axes:
            ax.tick_params(axis='both', which='both', bottom=False, left = False, labelleft = False, labelbottom=False)
        plt.subplots_adjust(hspace = 0, wspace = 0, right = 0.8)

        c1_loc = fig.add_axes([0.82, 0.50, 0.02, 0.25]) # left, bottom, width, height
        c1_bar = fig.colorbar(c1, cax=c1_loc, ticks=[log_clip, log_kde_max])
        c1_bar.ax.set_yticklabels([r'<$10^{{{}}}$'.format(log_clip), r'$10^{{{}}}$'.format(np.round(log_kde_max, 2))], fontsize = 6)

        c2_loc = fig.add_axes([0.82, 0.11, 0.02, 0.25]) # left, bottom, width, height
        c2_bar = fig.colorbar(c2, cax=c2_loc, ticks=[-1, 0, 1])
        c2_bar.ax.set_yticklabels([r'-$10^{{{}}}$'.format(np.round(np.log10(errors_max_abs), 2)), r'$\pm 10^{{{}}}$'.format(-2.5), r'$10^{{{}}}$'.format(np.round(np.log10(errors_max_abs), 2))], fontsize = 6)

        plt.savefig(path_to_here+'/../outputs/errors_{}.png'.format(drug_name), dpi = 900)


    def morphospace_connection(self):
        """
        Plot the morphospace coloured by the gradient magnitude
        """

        path = path_to_here+'/../data/landscape_visualizations/vis_embeddings.png'

        morphospace = cv2.imread(path, 0)
        morphospace = cv2.resize(morphospace, (5000,5000), interpolation = cv2.INTER_LINEAR)
        morphospace = 255-morphospace
        morphospace[morphospace>1] = 1

        morphospace = morphospace[600:4451, 638:4489]
        self.UArray[self.pList_NN[-1] < 1e-3] = np.nan
        resolution = int(self.resize_size[0]/26)
        self.UArray = self.UArray[3*resolution:23*resolution, 3*resolution:23*resolution]


        G = self._get_G(self.UArray, log = True)
        G = cv2.resize(G, morphospace.shape)

        #self.G[morphospace<170] = np.nan


        fig = plt.figure(figsize = (30, 30))
        ax = fig.add_subplot(111)
        ax.imshow(G[::-1, :], alpha = 1-morphospace,  cmap = 'jet', vmin = -5, vmax = -2)
        plt.savefig(path_to_here+'/../outputs/'+'morphospace_connection_{}.png'.format(drug_name))


    def entropy(self):

        colors_dict = {'DMSO':'lightgrey', 'compound_A':'magenta', 'compound_X':'deepskyblue', 'compound_C_0_041':'springgreen', 'compound_B':'orangered'}
        kdes_nn = open(path_SDE_objects + 'kdes_nn.pickle', 'rb')
        kdes_nn = [np.reshape(i, (200, 200)) for i in pickle.load(kdes_nn)]

        entropies = []
        for i in kdes_nn:

            i[i<1e-5] = 0
            p_int = np.sum(i*(26**2)/(200**2))
            S = 0
            for row in range(i.shape[0]):
                for col in range(i.shape[1]):
                    if i[row, col] != 0:
                        S += i[row, col]*np.log2(i[row, col])*(26**2)/(200**2)
            S = -S
            entropies.append(S)
        plt.plot([90, 105, 120, 135, 150, 165, 180, 195, 210], entropies, c = colors_dict[drug_name])
        plt.xlim([90, 210])


        plt.savefig(path_to_here+'/../outputs/'+'entropy_{}.png'.format(drug_name))



if __name__ == '__main__':

    vis = Visualisations(xlims, ylims,
                    pList_data, pList_NN, DMSO_pList_NN,
                    UArray,
                    trajectories_subsampled)

    if to_plot == 'landscape':
        vis.gradient_colored()
    elif to_plot == 'errors':
        vis.plot_sde_errors()
    elif to_plot == 'morphospace_connection':
        vis.morphospace_connection()
    elif to_plot == 'entropy':
        vis.entropy()
