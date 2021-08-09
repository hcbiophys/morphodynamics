import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys
import os
import glob





drug_name = sys.argv[1]
to_plot = sys.argv[2]


cmap = 'jet'
resize_size = (1000, 1000)

path_to_here = os.path.dirname(os.path.realpath(__file__))


def _load_and_resize_list(path):
    array_list = pickle.load(open(path, 'rb'))
    array_list = [cv2.resize(i, (1000, 1000), interpolation = cv2.INTER_LINEAR) for i in array_list]
    return array_list


def plot_ablation_losses():
    """
    Plot the ablation analysis showing diffusion is best with dependence on both position and time.
    """

    path1 = os.path.join(path_to_here, '../data/landscape_visualizations/DMSO/original/losses00-54.pickle')
    file = open(path1, 'rb')
    losses_array_1 = pickle.load(file)
    losses_array_1 = losses_array_1[1:, :]

    path2 = os.path.join(path_to_here, '../data/landscape_visualizations/DMSO/D_t/losses00-44.pickle')
    file = open(path2, 'rb')
    losses_array_2 = pickle.load(file)
    losses_array_2 = losses_array_2

    path3 = os.path.join(path_to_here, '../data/landscape_visualizations/DMSO/D_x_y/losses00-44.pickle')
    file = open(path3, 'rb')
    losses_array_3 = pickle.load(file)
    losses_array_3 = losses_array_3

    path4 = os.path.join(path_to_here, '../data/landscape_visualizations/DMSO/homog/losses00-76.pickle')
    file = open(path4, 'rb')
    losses_array_4 = pickle.load(file)
    losses_array_4 = losses_array_4[1:, :]

    paths = [path1, path2, path3, path4]




    # 'pdf', 'BC', 'pde', 'total', 'norm'
    fig = plt.figure(figsize = (2.1, 1.5))


    for path, losses_array, label in zip(paths, [losses_array_1, losses_array_2, losses_array_3, losses_array_4], ['D(x, y, t)', 'D(t)', 'D(x, y)', 'D']):

        print('shape', losses_array.shape)

        losses_base = os.path.basename(path)
        total_losses = losses_array[3, :] # total losses are idx 3 for these runs
        idxs = []
        means = []
        for idx in range(len(total_losses)-200):
            num_half_hrs = int(losses_base[6:8]) + idx*(int(losses_base[9:11])-int(losses_base[6:8]))/len(total_losses)
            if num_half_hrs < 55:
                idxs.append(num_half_hrs)
                means.append(np.mean(total_losses[idx:idx+200]))


        plt.scatter([i/2 for i in idxs], np.log10(means), s = 0.1, label = label)

    plt.ylabel(r'$log_{10}L_{total}$', fontsize = 6, labelpad = 1)
    plt.xlabel('Hours trained', fontsize = 6, labelpad = 1)
    plt.tick_params(axis = 'both', labelsize = 6)
    plt.tight_layout()
    plt.legend(fontsize = 6)



    plt.savefig(path_to_here+'/../outputs/ablation.png', dpi = 1200)




def losses_repeats():
    """
    Plot training loss over the 3 repeats for DMSO
    """

    paths = [os.path.join(path_to_here, '../data/landscape_visualizations/DMSO/original/*'),
            os.path.join(path_to_here, '../data/landscape_visualizations/DMSO/repeat_a/*'),
            os.path.join(path_to_here, '../data/landscape_visualizations/DMSO/repeat_b/*')]

    fig = plt.figure(figsize = (2.1, 1.5))
    for path, color, label in zip(paths, ['blue', 'green', 'orange'], ['original', 'repet_a', 'repeat_b']):
        losses_paths = [file for file in glob.glob(path) if os.path.basename(file)[:6] == 'losses']
        all_idxs, all_losses = [], []
        for losses_path in losses_paths:
            losses_base = os.path.basename(losses_path)
            losses = pickle.load(open(losses_path, 'rb'))
            losses = losses[4, :] # total losses are idx 4 for these runs

            for idx in range(len(losses)-200):
                all_idxs.append( int(losses_base[6:8]) + idx*(int(losses_base[9:11])-int(losses_base[6:8]))/len(losses) )
                all_losses.append(np.mean(losses[idx:idx+200]))

        plt.scatter([i/2 for i in all_idxs], np.log10(all_losses), s = 0.1, color = color, label = label)
    plt.legend(fontsize = 6)
    plt.ylabel(r'$log_{10}L_{total}$', fontsize = 6, labelpad = 1)
    plt.xlabel('Hours trained', fontsize = 6, labelpad = 1)
    plt.tick_params(axis = 'both', labelsize = 6)
    plt.tight_layout()
    plt.savefig(path_to_here+'/../outputs/losses_repeats.png', dpi = 1200)





def spore_integrals():
    """
    Plot how the (proxy for the) spore region integral changes with time, to see when the PINN begins
    to fit to individual snapshots
    """

    path_tops = os.path.join(path_to_here, '../data/landscape_visualizations/DMSO/{}/{}_hours')
    idx_epochs = [25, 30]
    repeat_names = ['original', 'repeat_a', 'repeat_b']
    pList_data = os.path.join(path_to_here, '../data/embeddings/lims13/lims13_DMSO.pickle')
    pList_data = _load_and_resize_list(pList_data)


    def _plot_from_idx_repeat(ax, repeat_name, idx_epoch, color, linewidth = 0.6, linestyle = '-'):
        if os.path.isfile(path_tops.format(repeat_name, idx_epoch) + '/p_list_0.pickle'):
            p_list_NN = _load_and_resize_list(path_tops.format(repeat_name, idx_epoch) + '/p_list_0.pickle')
            NN_spore_integrals = [np.sum(i[427:442, 182:196])*(26**2)/(1000**2) for i in p_list_NN]
            ax.plot([1, 2, 3, 4, 5, 6, 7, 8], NN_spore_integrals[1:],  c = color, linewidth = linewidth)


    data_spore_integrals = [np.sum(i[427:442, 182:196])*(26**2)/(1000**2) for i in pList_data]

    fig = plt.figure(figsize = (2.3, 1.5))
    for i, idx_epoch in enumerate(idx_epochs):
        ax = fig.add_subplot(len(idx_epochs), 1, i+1)
        ax.plot([1, 2, 3, 4, 5, 6, 7, 8], data_spore_integrals[1:], linewidth = 1, c = 'red')
        if idx_epoch is not idx_epochs[-1]:
            ax.set_xticks([])
        for repeat_name, color in zip(repeat_names, ['blue', 'green', 'orange']):
            _plot_from_idx_repeat(ax = ax, repeat_name = repeat_name, idx_epoch = idx_epoch, color = color)
        if idx_epoch == idx_epochs[-1]:
            _plot_from_idx_repeat(ax = ax, repeat_name = 'original', idx_epoch = 60, color = 'black', linewidth = 1, linestyle = '--')

    ax.set_ylabel(r'$\approx \int_{spore} \hat{p}(\mathbf{x},t)$', fontsize = 6, labelpad = 1)
    ax.set_xlabel('Snapshot', fontsize = 6, labelpad = 1)
    for ax in fig.axes:
        ax.tick_params(axis = 'both', labelsize = 6)

    plt.subplots_adjust(hspace = 0)
    plt.tight_layout()
    plt.savefig(path_to_here+'/../outputs/'+'spore_integrals.png', dpi = 1200)






def change_array_lims(array):
    res = int(resize_size[0]/26)
    array = array[3*res:-3*res, 3*res:-3*res]
    return array


def plot_field_uncertainties():
    """
    Plot the mean sigma and sigma & F uncertainties
    """

    resize_size = (1000, 1000)


    dirs = [os.path.join(path_to_here, '../data/landscape_visualizations/{}/{}/'.format(drug_name, j)) for j in ['original', 'repeat_a', 'repeat_b']]
    if drug_name == 'DMSO':
        dirs = [os.path.join(path_to_here, '../data/landscape_visualizations/{}/{}/30_hours/'.format(drug_name, j)) for j in ['original', 'repeat_a', 'repeat_b']]

    def transform(x):
        if type(x) is np.ndarray:
            x = change_array_lims(x)
        x = np.log(x)
        return x

    F_unc_vmin = -7
    F_unc_vmax = -4
    sigma_vmin = -5
    sigma_vmax = 0 #0.4
    sigma_unc_vmin = -6
    sigma_unc_vmax = -2

    fig_Fs = [plt.figure() for _ in range(3)]
    fig_uncertainty = plt.figure()
    sigma_lists, F_arrays = [], []
    for idx_fig, dir in enumerate(dirs):

        p_list = _load_and_resize_list(dir+'p_list_0.pickle')
        D_list = _load_and_resize_list(dir+'D_list_0.pickle')
        U_array = pickle.load(open(dir+'U.pickle', 'rb'))
        U_array = cv2.resize(U_array, resize_size, interpolation = cv2.INTER_LINEAR)
        Gx, Gy = np.gradient(U_array, 26./resize_size[0], 26./resize_size[0]) # gradients with respect to x and y
        F_array = (Gx**2+Gy**2)**.5  # gradient magnitude
        F_array[np.isinf(F_array)] = np.nan
        F_array[p_list[-1]<1e-3]=np.nan # final PDF
        sigma_list = []
        for j in range(9):
            arr = D_list[2*j] # current PDF
            arr[p_list[j]<1e-3]=np.nan
            sigma_list.append(np.sqrt(2*arr))


        sigma_lists.append(sigma_list)
        F_arrays.append(F_array)

        ax = fig_Fs[idx_fig].add_subplot(111)
        ax.imshow(transform(F_array)[::-1, :], cmap = cmap, vmin = -4.6, vmax = -2)
        ax.set_title(dir)

    all_axes = [i for j in fig_Fs for i in j.axes]
    for ax in all_axes:
        ax.axis('off')

    # uncertainties

    std = np.std(F_arrays, axis = 0)
    ax = fig_uncertainty.add_subplot(121)
    ax.imshow(transform(std)[::-1, :], cmap = cmap, vmin = F_unc_vmin, vmax = F_unc_vmax)
    ax.set_title('F_uncertainty')

    fig_sigma = plt.figure()
    ax = fig_sigma.add_subplot(111)
    ax.imshow(transform(np.nanmean(sigma_lists[0], axis = 0))[::-1, :], cmap = cmap, vmin = sigma_vmin, vmax = sigma_vmax) # index 0 (i.3 'original' is corresponds to the landscapes in other figures)
    ax.set_title('sigma_mean')

    sigma_means = [np.nanmean(sigma_list, axis = 0) for sigma_list in sigma_lists]
    std_array = np.nanstd(sigma_means, axis = 0)
    ax = fig_uncertainty.add_subplot(122)
    ax.imshow(transform(std_array)[::-1, :], cmap = cmap, vmin = sigma_unc_vmin, vmax = sigma_unc_vmax)
    ax.set_title('sigma_uncertainty')

    fig_sigma.savefig(path_to_here+'/../outputs/{}_mean_sigma.png'.format(drug_name), dpi = 1200)
    fig_uncertainty.savefig(path_to_here+'/../outputs/{}_uncertainties.png'.format(drug_name), dpi = 1200)






if to_plot == 'ablation':
    plot_ablation_losses()
elif to_plot == 'losses_repeats':
    losses_repeats()
elif to_plot == 'spore_integrals':
    spore_integrals()
elif to_plot == 'plot_field_uncertainties':
    plot_field_uncertainties()
