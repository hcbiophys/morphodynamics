import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os


def find_MSDs(paths):
    SDs_all = []
    for path in paths:
        x0, y0 = path[0][0], path[0][1]
        SDs = [(i[0]-x0)**2 + (i[1]-y0)**2 for i in path]
        SDs_all.append(SDs)
    MSDs = []
    #min_len = min([len(i) for i in SDs_all])
    for step in range(35):
        sum = 0
        num = 0
        for SDs in SDs_all:
            if len(SDs) > step:
                sum += SDs[step]
                num += 1
        MSDs.append(sum/num)
    return MSDs




def add_sims_to_paths():
    fig1 =  plt.figure(figsize = (10, 10))
    ax = fig1.add_subplot(111)
    im = cv2.imread(path_to_here+'/../data/embeddings/video_trajs/paths.png', 8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    ax.imshow(im)
    for i in sims:
        xs = [j[0] for j in i]
        ys = [j[1] for j in i]
        if min(xs) > -10 and max(xs) < 10 and min(ys) > -10 and max(ys) < 10:
            xs = [((x+10)/20)*im_dims for x in xs]
            ys = [im_dims - ((y+10)/20)*im_dims for y in ys]
            ax.plot(xs, ys, c = 'black', linewidth = 1)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(path_to_here+'/../outputs/video_embeddings.png', dpi = 300)






def plot_MSDs():
    colors_dict = {'DMSO':'lightgrey', 'compound_A':'magenta', 'compound_X':'deepskyblue', 'compound_C_0_041':'springgreen', 'compound_B':'orangered'}


    fig2 = plt.figure(figsize = (1.6, 2))
    ax = fig2.add_subplot(211)
    video_MSDs = find_MSDs(video_trajs)
    ax.plot([60+i*3 for i in range(len(video_MSDs))], np.log10(video_MSDs), label = 'videos', c = colors_dict[drug_name])
    sims_MSDs = find_MSDs(sims)
    ax.plot([90+i*3 for i in range(len(sims_MSDs))], np.log10(sims_MSDs), label = 'sims', c = 'black')
    #ax.set_xlabel('Time (min) after mixing with solution', fontsize = 6)
    #ax.set_ylabel(r'log(MSD) ($mu^{2}$)', fontsize = 6)
    ax.tick_params(labelleft=False)
    ax.set_ylim([-1, 2.2])
    ax.set_xlim([0, 210])
    ax.tick_params(axis="x", direction="in", labelsize = 6)
    ax.tick_params(axis="y", direction="in", labelsize = 6)
    plt.tight_layout()
    plt.savefig(path_to_here+'/../outputs/MSDs_{}.png'.format(drug_name), dpi = 1200)




def cross_correlation_matrix():

    def _cross_corr(p, q):
        corr = np.sum(np.abs(np.subtract(p, q)))/len(p)
        return corr

    matrix = np.zeros(shape = (5, 5))
    for i, drug_video in enumerate(['DMSO', 'Carb', 'Pik', 'Sol41', 'Tou']):
        for j, drug_sim in enumerate(['DMSO', 'Carb', 'Pik', 'Sol41', 'Tou']):
            a = find_MSDs(video_trajs_all[drug_video])
            b = find_MSDs(sims_all[drug_sim])

            a, b = [i for i in a if i >= 10**(-1)], [i for i in b if i >= 10**(-1)]
            a, b = np.log10(a), np.log10(b)

            corrs = []
            for rep in [1, 2]:
                for shift in range(5):
                    if rep == 1:
                        a_new = a[shift:]
                        b_new = b[:len(a[shift:])]
                    elif rep == 2:
                        b_new = b[shift:]
                        a_new = a[:len(b[shift:])]
                    min_len = min([len(a_new), len(b_new)])
                    a_new, b_new = a_new[:min_len], b_new[:min_len]
                    corr = _cross_corr(a_new, b_new)
                    corrs.append(corr)

            matrix[i, j] = min(corrs)

    print(matrix)
    fig = plt.figure(figsize = (1.7, 1.7))
    ax = fig.add_subplot(111)

    pos = ax.imshow(np.log10(matrix), vmin = -1.2, vmax = -0.25)
    fig.colorbar(pos, ax=ax, ticks = [])
    plt.tight_layout()

    plt.savefig('/Users/harry/Desktop/confusion_matrix.png'.format(drug), dpi = 1200)








for drug_name in ['compound_A', 'compound_B', 'compound_C_0_041', 'compound_X', 'DMSO']:

    path_to_here = os.path.dirname(os.path.realpath(__file__))

    video_trajs_ = open(path_to_here+'/../data/embeddings/video_trajs/{}.pickle'.format(drug_name), 'rb')
    video_trajs_ = pickle.load(video_trajs_)

    path = path_to_here+'/../data/landscape_visualizations/{}/original/subsampled_paths_p0.pickle'.format(drug_name)
    if drug_name == 'DMSO':
        path = path_to_here+'/../data/landscape_visualizations/{}/original/60_hours/subsampled_paths_p0.pickle'.format(drug_name)

    sims = open(path, 'rb')
    sims = pickle.load(sims)


    lims_list = (-10, 70, -50, 55)
    im_dims = 4000

    def _standardize_coords(coords, orig_lims):
        [xmin, xmax, ymin, ymax] = orig_lims
        x_new = -10 + 20*(coords[0]-xmin)/(xmax-xmin)
        y_new = -10 + 20*(coords[1]-ymin)/(ymax-ymin)
        return [x_new, y_new]


    video_trajs = []
    for idx, i in enumerate(video_trajs_):
        video_trajs.append([_standardize_coords(j, lims_list) for j in i])

    video_trajs = [i[:35] for i in video_trajs]
    sims = [i[::6] for i in sims] # so it's 3 min intervals like the videos
    sims = [i[:35] for i in sims]












    plot_MSDs()
add_sims_to_paths()
cross_correlation_matrix()
