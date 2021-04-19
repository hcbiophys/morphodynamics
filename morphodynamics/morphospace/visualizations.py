import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import torch

from unsupervised.utils_unsupervised import get_lims_from_points, kde, _standardize_coords

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Visualizations_Mixin:

    def code_projections_2D(self, im_size, save_path, proj_sampling_rate):
        """
        Sample code space on grid and visualize the features the network
        is capturing.
        Args:
        - im_size: tuple of dimensions of each image on the grid.
        - save_as: filename to save image as.
        - proj_sampling_rate: number of samples along each axis.
        """

        self.eval()

        # set the limits of the grid to project from to be either data_lims (set by running the function that
        # finds the min & max of data embeddings), or lims_list which is specified when initializing the VAE class
        if self.data_limits is None:
            min_x, max_x = self.lims_list[0], self.lims_list[1]
            min_y, max_y = self.lims_list[2], self.lims_list[3]
        else:
            min_x, max_x = self.data_limits[0], self.data_limits[1]
            min_y, max_y = self.data_limits[2], self.data_limits[3]

        im_array = np.zeros((int(max_y) - int(min_y), int(max_x) - int(min_x)))

        # split the code space into grid
        grid_xs = np.linspace(min_x, max_x, proj_sampling_rate)
        grid_ys = np.linspace(min_y, max_y, proj_sampling_rate)
        x_im_width = grid_xs[1]-grid_xs[0]
        y_im_width = grid_ys[1]-grid_ys[0]

        # dimension of each image to be plottted
        im_dimension = 200

        # empty array to fill with the images
        array = np.zeros( ( im_dimension * len(grid_ys), im_dimension * len(grid_xs) ) )

        fig = plt.figure(figsize = (50, 50))
        # i, j for placement in empty array; x, y initial codes
        for i, x in enumerate(grid_xs):
            x += x_im_width/2
            for j, y in enumerate(grid_ys):
                y += y_im_width/2
                code = torch.tensor([x, y], dtype = torch.float32)
                code = torch.unsqueeze(code, 0)
                code = code.to(device)
                #batch = F.relu(batch)
                code = self.fc2(code)
                batch = self.decoder(code)

                batch = batch.view(*im_size)
                im = batch.detach().cpu().numpy() # need to detach from graph before it can be turned into array

                im = cv2.resize(im, (im_dimension, im_dimension))

                row1 = im_dimension * j
                row2 =  im_dimension * j + im_dimension
                col1 = im_dimension * i
                col2 = im_dimension * i + im_dimension
                array[row1:row2, col1:col2] = im[::-1, :] # filling the empty array with the images

        array = array[::-1, :]

        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
        plt.imshow(array, cmap='Blues')
        plt.savefig(save_path.format('code_projections'))
        plt.close()

        self.proj_array = array
        self.proj_array_im_dim = im_dimension
        self.proj_sampling_rate = proj_sampling_rate




    def scatter_1drug_allSnaps(self, drug, idx_drug, save_path):
        """
        ( After dataset.keep_one_drug_timeLabelled() )
        Scatter for each temporal snapshot, for a single drug.
        Args
        - save_as: save path
        """

        # set the limits of the grid to project from to be either data_lims (set by running the function that
        # finds the min & max of data embeddings), or lims_list which is specified when initializing the VAE class
        if self.data_limits is None:
            xmin, xmax = self.lims_list[0], self.lims_list[1]
            ymin, ymax = self.lims_list[2], self.lims_list[3]
        else:
            xmin, xmax = self.data_limits[0], self.data_limits[1]
            ymin, ymax = self.data_limits[2], self.data_limits[3]

        xs, ys, labels = [], [], []
        for item in self.points:
            xs.append(item[0][0])
            ys.append(item[0][1])
            labels.append(item[1])


        data_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}

        for i in range(len(xs)):
            data_dict[labels[i]].append((xs[i], ys[i]))

        labels_to_colours = {0 :colors.to_rgba('lightgrey'), 1 : colors.to_rgba('magenta'), 2: colors.to_rgba('deepskyblue'), 3:colors.to_rgba('orangered') , 4:colors.to_rgba('springgreen') , 5: colors.to_rgba('darkgreen')}

        fig = plt.figure(figsize = (70, 40))
        for idx_time, coords in data_dict.items():
            ax = fig.add_subplot(1, 9, idx_time+1)
            coords = [_standardize_coords(c, orig_lims = [xmin, xmax, ymin, ymax]) for c in coords]
            ax.scatter([c[0] for c in coords], [c[1] for c in coords], s = 5, c = labels_to_colours[idx_drug])
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
        plt.savefig(save_path.format(drug))
        plt.close()



    def graphic_drugOrtime_1snap(self, which_colors, save_path):
        """
        Plots a single 2D graphic, either
        a) Snapshot of all drugs at a specific time point.
        b) Single drug with different time snaps.
        Args
        - save_as
        - which_colors: colors depending if executing (a) or (b)
        """

        if which_colors == 'drug_colors':
            labels_to_colours = {0 :colors.to_rgba('lightgrey'), 1 : colors.to_rgba('magenta'), 2: colors.to_rgba('deepskyblue'), 3:colors.to_rgba('orangered') , 4:colors.to_rgba('springgreen') , 5: colors.to_rgba('darkgreen')}
        elif which_colors == 'time_colors':
            labels_to_colours = {0 : (1, 0.9, 0.9), 1 : (1, 0.8, 0.8), 2 : (1, 0.7, 0.7), 3 : (1, 0.6, 0.6), 4: (1, 0.5, 0.5), 5: (1, 0.4, 0.4), 6: (1, 0.3, 0.3), 7: (1, 0.2, 0.2), 8: (1, 0.1, 0.1)}


        xs, ys, colours = [], [], []

        for item in self.points:
            xs.append(item[0][0])
            ys.append(item[0][1])
            colours.append(labels_to_colours[item[1]])

        fig = plt.figure(figsize = (20, 20))
        ax = fig.add_subplot(111)
        ax.scatter(xs, ys, c = colours)
        ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))

        plt.savefig(save_path.format('dots'))
        plt.close()


        # set the limits of the grid to project from to be either data_lims (set by running the function that
        # finds the min & max of data embeddings), or lims_list which is specified when initializing the VAE class
        if self.data_limits is None:
            min_x, max_x = self.lims_list[0], self.lims_list[1]
            min_y, max_y = self.lims_list[2], self.lims_list[3]
        else:
            min_x, max_x = self.data_limits[0], self.data_limits[1]
            min_y, max_y = self.data_limits[2], self.data_limits[3]


        x_range = max_x - min_x
        y_range = max_y - min_y

        array_dim = 2200
        ims_array = np.zeros((array_dim, array_dim, 3), dtype = np.uint8)
        ims_array+=255
        random.shuffle(self.points)

        for item in self.points:
            label = item[1]

            path = item[2]
            im = cv2.imread(path, 0)

            _, thresh = cv2.threshold(im, 30, 255, cv2.THRESH_BINARY)

            cv2.circle(thresh, (int(thresh.shape[1]/5)-5, int(thresh.shape[0]/2)), 8, 255, -1)

            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                for contour in contours:
                    x_start = int(((array_dim-200)/x_range)*(item[0][0] + abs(min_x)))
                    y_start = int(((array_dim-200)/y_range)*(item[0][1] + abs(min_y)))

                    contour[:, :, 0] += x_start
                    #contour[:, :, 0] = np.mean(contour[:, :, 0]) - contour[:, :, 0] + np.mean(contour[:, :, 0])
                    contour[:, :, 1] += array_dim - y_start

                    colour_increased = np.array(labels_to_colours[label])*255
                    cv2.drawContours(ims_array, [contour], -1, colour_increased, -1)

        fig, ax = plt.subplots(1, figsize=(100, 100))
        ax.imshow(ims_array)
        plt.savefig(save_path.format('morphologies'))
        plt.close()

        return ims_array
