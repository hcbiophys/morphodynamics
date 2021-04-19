import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import skewnorm, norm, uniform, lognorm, gamma
import random
import pickle
import matplotlib.ticker as ticker
import os



@ticker.FuncFormatter
def major_formatter(x, pos):
    """
    For formatting plots
    """
    label = -x if x < 0 else x
    if label == 0:
        return '0'
    label = np.round(label, 2)
    return str(label)


def get_sim_hists(times, t_germ_s, t_germ_loc, t_germ_scale, grad_s, grad_loc, grad_scale, num_runs, bins):
    """
    Run simulations and generate a histogram to be compared with the data
    - times: the snapshot times
    - t_germ_s, t_germ_loc, t_germ_scale, grad_s, grad_loc, grad_scale: the parameters for the simulation
    - num_runs: number of simulations to run
    - bins: resolution of the histogram
    """

    sim_nestedLists_lengths = [[] for i in range(len(times))]
    for idx_run in range(num_runs): # for each run

        t_germ = lognorm.rvs(s = t_germ_s, loc = t_germ_loc, scale = t_germ_scale, size = 1)

        count = 0
        grad = -1
        while grad < 0:
            grad = -lognorm.rvs(s = grad_s, loc = 0, scale = grad_scale, size = 1) + grad_loc
            count += 1
            if count > 5: # if computation will continue indefinitely, set gradient to unrealistic value so these parameters are rejected
                grad = 50


        for idx_snap, snap in enumerate(times):
            if t_germ < snap:
                length = grad*(snap-t_germ)
                sim_nestedLists_lengths[idx_snap].append(length)
            else:
                sim_nestedLists_lengths[idx_snap].append(0) # add length zero if it is yet to germinate

    # generate the histogram for comparison with data
    hists_sim = []
    for idx_time in range(9):
        hist, _ = np.histogram(sim_nestedLists_lengths[idx_time], bins = bins, range = [0, 150], density = True)
        hists_sim.append(hist)


    return hists_sim


def plot_MAP_comparison(times, data_hists, num_runs, bins, drug_name, param_dict):
    """
    Plot comparisons of model simulations with data
    - data_hists: data histograms
    - num_runs: number of runs used in ABC-SMC
    - bins: histogram resolution
    - drug_name
    - param_dict: simulation parameters to compare with data
    """



    colors_dict = {'DMSO':'lightgrey', 'compound_A':'magenta', 'compound_X':'deepskyblue', 'compound_C_0_041':'springgreen'}

    hists_sim = get_sim_hists(times, param_dict[drug_name][0], param_dict[drug_name][1], param_dict[drug_name][2], param_dict[drug_name][3], param_dict[drug_name][4], param_dict[drug_name][5], num_runs, bins)

    ## ALL 9 HISTOGRAMS
    fig = plt.figure(figsize = (7, 1.5))
    plotted = 0
    for idx_time in range(9):
        data_hist = data_hists[idx_time]
        ax = fig.add_subplot(1, 9, plotted+1)
        ax.barh(np.linspace(0, 150, bins), width = -data_hist, height = 1+150./bins, color = 'black')
        ax.barh(np.linspace(0, 150, bins), width = hists_sim[idx_time], height = 1+150./bins, color = colors_dict[drug_name])
        ax.set_xlim([-0.12, 0.12])
        ax.set_xticks([-0.1, 0, 0.1])
        ax.xaxis.set_major_formatter(major_formatter)
        ax.set_ylim([0, 150])

        if idx_time == 0:
            ax.set_ylabel(r'L($\mu m$)', fontsize = 6, fontname="Arial", labelpad = 1)
        else:
            ax.tick_params(labelleft=False)
        if idx_time == 4:
            ax.set_xlabel('PDF', fontsize = 6, fontname="Arial", labelpad = 1)
        plotted += 1

    for ax in fig.axes:
        ax.tick_params(direction='out', pad=1)
        ax.tick_params(axis="x", direction="in", labelsize = 6)
        ax.tick_params(axis="y", direction="in", labelsize = 6)
    plt.subplots_adjust(bottom = 0.2)
    #plt.tight_layout()
    path_to_here = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(path_to_here+'/../../outputs/MAP_simulation_{}.png'.format(drug_name), dpi = 1200)
    plt.close()


    ## CDFs & PDFs
    fig = plt.figure(figsize = (4.1, 1))

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    xs1 = np.arange(50, 210, 1)
    xs2 = np.arange(0, 5, 0.01)
    for drug in ['DMSO', 'compound_A', 'compound_X', 'compound_C_0_041']:
        ax1.plot(xs1, lognorm.cdf(xs1, s = param_dict[drug][0], loc = param_dict[drug][1], scale = param_dict[drug][2]), c = colors_dict[drug], linewidth = 1)
        ax2.plot(-xs2+param_dict[drug][4], lognorm.pdf(xs2, s = param_dict[drug][3], loc = 0, scale = param_dict[drug][5]), c = colors_dict[drug], linewidth = 1)

    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1.2])
    ax2.set_ylim([0, 5])
    ax1.set_ylabel('CDF', fontsize = 6, labelpad = 1)
    ax1.set_xlabel('Germination time (min)', fontsize = 6, labelpad = 1)
    ax2.set_ylabel('PDF', fontsize = 6, labelpad = 1)
    ax2.set_xlabel(r'Growth rate ($\mu m \; min^{-1})$', fontsize = 6, labelpad = 1)

    for ax in fig.axes:
        ax.tick_params(direction='out', pad=1)
        ax.tick_params(axis="x", direction="in", labelsize = 6)
        ax.tick_params(axis="y", direction="in", labelsize = 6)
    plt.subplots_adjust(bottom = 0.4,  hspace = 0)
    path_to_here = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(path_to_here+'/../../outputs/MAP_distributions.png', dpi = 1200)
    plt.close()




def distance_abs(x_dict, y_dict):
    """
    Distance function for use in pyABC package, which takes dictionary arguments
    """
    x = x_dict['X_2']
    y = y_dict['X_2']

    dist = 0
    for idx_time in range(9):
            sim_hist = x[idx_time]
            data_hist = y[idx_time]
            for idx_el in range(len(data_hist)):
                a = data_hist[idx_el]
                b = sim_hist[idx_el]
                dist += abs(a - b)

    return dist


def distance_abs_histograms(x_hists, y_hists):
    """
    Distance function for comparing directly with histogram inputs x_hists and y_hists
    """

    dist = 0
    for idx_time in range(9):
            sim_hist = x_hists[idx_time]
            data_hist = y_hists[idx_time]
            for idx_el in range(len(data_hist)):
                a = data_hist[idx_el]
                b = sim_hist[idx_el]
                dist += abs(a - b)

    return dist
