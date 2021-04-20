import sys
import torch
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import pyabc
import tempfile
import pandas as pd

from morphodynamics.tip_model.theta.morph_sim_class import *
from morphodynamics.tip_model.theta.utils_theta import *
from morphodynamics.tip_model.theta.accepted_params_kappa import *

to_run = sys.argv[1] # 'model_probabilities', 'MAP_vis', 'M2_posterior' or 'full_inference'
drug_name = sys.argv[2]
idx_model = int(sys.argv[3])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db"))





class theta_ABC():

    def __init__(self, drug_name, L_params, num_runs, pop_size):
        """
        - drug_name
        - L_params: parameters to use for the length equation
        - num_runs: number of fungi simulated per comparison
        - pop_size: number of comparisons in each population
        """

        self.drug_name = drug_name
        self.L_params = L_params

        self.num_runs = num_runs
        self.pop_size = pop_size

        path_to_here = os.path.dirname(os.path.realpath(__file__))
        file = os.path.join(path_to_here, '../data/embeddings/lims10/lims10_{}.pickle'.format(drug_name))
        file = open(file, 'rb')
        self.data_points = pickle.load(file)
        self.data_points = list(self.data_points[8])

        self.M0_sigma_min = 0
        self.M0_sigma_max = 0.2

        self.M1_sigma_min = 0
        self.M1_sigma_max = 0.01

        self.M2_sigma_min = 0
        self.M2_sigma_max = 0.05
        self.M2_inv_tau_min = 0
        self.M2_inv_tau_max = 0.2

        self.M0_priors = pyabc.Distribution(sigma = pyabc.RV("uniform", self.M0_sigma_min, self.M0_sigma_max))

        self.M1_priors = pyabc.Distribution(sigma = pyabc.RV("uniform", self.M1_sigma_min, self.M1_sigma_max))

        self.M2_priors = pyabc.Distribution(sigma = pyabc.RV("uniform", self.M2_sigma_min, self.M2_sigma_max),
                                            inv_tau = pyabc.RV("uniform", self.M2_inv_tau_min, self.M2_inv_tau_max))


        self.morph_sim  = Morph_Sim(self.num_runs, [self.L_params[0], self.L_params[1], self.L_params[2], self.L_params[3], self.L_params[4], self.L_params[5]],  data_points = self.data_points)
        self.morph_sim.load_VAE()



    def M0_simulation(self, parameters):
        """
        Model 0 simluation
        - parameters: sigma and inv_tau
        """
        sigma = parameters['sigma']
        self.morph_sim.M0_set_sim_ims(sigma = sigma)
        self.morph_sim.set_sim_points()
        return {'X_2' : self.morph_sim.sim_points}

    def M1_simulation(self, parameters):
        """
        Model 1 simluation
        - parameters: sigma and inv_tau
        """
        sigma = parameters['sigma']
        self.morph_sim.M1_set_sim_ims(sigma = sigma)
        self.morph_sim.set_sim_points()
        return {'X_2' : self.morph_sim.sim_points}

    def M2_simulation(self, parameters):
        """
        Model 2 simluation
        - parameters: sigma and inv_tau
        """
        sigma = parameters['sigma']
        inv_tau = parameters['inv_tau']
        self.morph_sim.M2_set_sim_ims(sigma, inv_tau)
        self.morph_sim.set_sim_points()
        return {'X_2' : self.morph_sim.sim_points}


    def run(self, idxs_models, num_populations, numPops_save_every):
        """
        Run the whole ABC-SMC routine
        - idxs_models: models to run with. If more than 1 model, a model selection is run (with model index as an extra parameter)
        - num_populations: number of populations to run for
        - numPops_save_every: how often to save viualizations
        """

        all_model_sims = [self.M0_simulation, self.M1_simulation, self.M2_simulation]
        all_priors = [self.M0_priors, self.M1_priors, self.M2_priors]


        count_pop = 0

        while count_pop < num_populations:
            abc_object = pyabc.ABCSMC(models = [j for i,j in enumerate(all_model_sims) if i in idxs_models],
                         parameter_priors = [j for i,j in enumerate(all_priors) if i in idxs_models],
                         distance_function = distance_abs,
                         population_size = self.pop_size,
                         sampler = pyabc.sampler.SingleCoreSampler())

            if count_pop == 0:
                abc_object.new(db_path, {"X_2": self.data_points})
            else:
                abc_object.load(db_path, load_id)


            history = abc_object.run(minimum_epsilon = 1e-20, max_nr_populations = numPops_save_every) # each population is a t (with unique epsilon)
            load_id = history.id


            model_probabilities = history.get_model_probabilities()
            print('model probabilities:', model_probabilities)

            df, w = history.get_distribution(m=0)
            print('df')
            for column in df:
                print(column)
                print(df[column].tolist())
            print('w',  list(w))

            self.morph_sim.add_simIms_toimsArray(drug_name = self.drug_name, idx_model = idxs_models[0], idx_pop = count_pop)
            count_pop += numPops_save_every



    def MAP_visualization(self, idx_model, *args):
        """
        Plot the comparison of data and simulated morphospace embeddings
        Args are the parameters of the model
        """
        all_models_set_sim_ims = [self.morph_sim.M0_set_sim_ims, self.morph_sim.M1_set_sim_ims, self.morph_sim.M2_set_sim_ims]
        all_models_set_sim_ims[idx_model](*args)
        self.morph_sim.set_sim_points()
        self.morph_sim.add_simIms_toimsArray(drug_name = self.drug_name, idx_model = idx_model[0], idx_pop = idx_model)
        x_dict = {"X_2": self.data_points}
        y_dict = {'X_2' : self.morph_sim.sim_points}
        distance_abs(x_dict, y_dict)


def save_M2_posterior(drug):
    """
    Save the posterior distribution plot
    """

    if drug == 'DMSO':
        sigmas, inv_taus, weights = DMSO_sigmas, DMSO_inv_taus, DMSO_weights
    elif drug == 'compound_A':
        sigmas, inv_taus, weights = compound_A_sigmas, compound_A_inv_taus, compound_A_weights
    elif drug == 'compound_X':
        sigmas, inv_taus, weights = compound_X_sigmas, compound_X_inv_taus, compound_X_weights
    elif drug == 'compound_C_0_041':
        sigmas, inv_taus, weights = compound_C_0_041_sigmas, compound_C_0_041_inv_taus, compound_C_0_041_weights

    # make dataframe
    df =  pd.DataFrame({'sigma': sigmas, 'inv_tau': inv_taus})


    X, Y, PDF = pyabc.visualization.kde.kde_2d(df, w = np.array(weights), x = 'sigma', y = 'inv_tau', numx = 200, numy = 200,
                                        xmin = 0, xmax = 0.05, ymin = 0, ymax = 0.2)

    fig = plt.figure(figsize = (1.2, 0.8))
    ax = fig.add_subplot(111)
    ax.pcolormesh(X, Y, PDF, cmap = 'jet', vmin = 0, vmax = 2000)
    ax.scatter(sigmas, inv_taus, c = 'white', s = 3, marker = 'x', linewidth=0.5)

    # find MAP estimate
    post_values = []
    for sigma, inv_tau in zip(sigmas, inv_taus):
        col = int((sigma/0.05)*200)
        row = int((inv_tau/0.2)*200)
        post_values.append(PDF[row, col])
    idx_MAP = post_values.index(max(post_values))
    print('{} MAP'.format(drug), sigmas[idx_MAP], inv_taus[idx_MAP])
    im = ax.scatter(sigmas[idx_MAP], inv_taus[idx_MAP], c = 'red', s = 6, marker = 'x', linewidth=1)


    ax.set_xticks([0, 0.05])
    ax.set_yticks([0, 0.2])
    ax.set_xticklabels(['0', '0.05'])
    ax.set_yticklabels(['0', '0.2'])
    ax.tick_params(axis="both", direction="in", labelsize = 6, pad = 1)
    #ax.tick_params(labelleft=False)
    ax.set_ylabel(r'$\tau^{-1} \; (min^{-1})$', fontsize = 6)
    ax.set_xlabel(r'$\sigma \; (rad \; min^{-\frac{1}{2}})$', fontsize = 6)
    ticklabels = ax.get_xticklabels()
    ticklabels[0].set_ha("left")
    ticklabels[-1].set_ha("right")
    ticklabels = ax.get_yticklabels()
    ticklabels[0].set_va("bottom")
    ticklabels[-1].set_va("top")

    cb_loc = fig.add_axes([0.82, 0.50, 0.02, 0.25]) # left, bottom, width, height
    cb = fig.colorbar(im, cax=cb_loc, ticks=[0, 2000])
    #cb.ax.set_yticklabels([], fontsize = 6)

    plt.subplots_adjust(left = 0.2, bottom = 0.2, right = 0.8)
    path_to_here = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(path_to_here+'/../outputs/posterior_{}.png'.format(drug), dpi = 1200)
    plt.close()






def model_selection_plot():
    """
    Plot how model probabilities evolve through the ABC-SMC populations
    """
    fig = plt.figure(figsize = (6.2, 2))
    bars = [[0.3, 0.5, 0.2],
            [0.094, 0.47, 0.43],
            [0.224149, 0.41, 0.37],
            [0.17, 0.52, 0.31],
            [0.054, 0.49, 0.45],
            [0, 0.53, 0.47],
            [0, 0.55, 0.45],
            [0, 0.49, 0.5],
            [0, 0.29, 0.71]]
    for idx, bar in enumerate(bars):
        ax = fig.add_subplot(3, 3, idx+1)
        ax.bar([0, 1, 2], bar, color = 'red')
        ax.set_xticks([])
        ax.set_ylim([0, 1.1])
        ax.tick_params(axis="both", direction="in", labelsize = 6)
        path_to_here = os.path.dirname(os.path.realpath(__file__))
        plt.savefig(path_to_here+'/../outputs/model_selection.png', dpi = 1200)
    sys.exit()







if __name__ == '__main__':



    drug_names = ['DMSO', 'compound_A', 'compound_X', 'compound_C_0_041']
    idx_drug = drug_names.index(drug_name)
    L_params = [[4.55, 78, 1.58, 0.34, 1.24, 0.55],
                [4.84, 75, 4.12, 0.18, 1.42, 0.94],
                [2.45, 66.2, 61.4, 0.14, 1.09, 0.64],
                [1.85, 68.4, 36.6, 0.62, 0.95, 0.30]]

    ABC = theta_ABC(drug_names[idx_drug], L_params[idx_drug], num_runs = 1000, pop_size = 40)

    model0 = {'compound_A': [0.126]}
    model1 = {'compound_A': [0.0065]}
    model2 = {'DMSO': [0.0116, 0.068], 'compound_A': [0.040, 0.138], 'compound_X': [0.0029, 0.107], 'compound_C_0_041': [0.0074, 0.048]}
    models_and_drugs = [model0, model1, model2]



    if to_run == 'model_probabilities':
        model_selection_plot()
    elif to_run == 'MAP_vis':
        ABC.MAP_visualization(idx_model, *models_and_drugs[idx_model][drug_name])
    elif to_run == 'M2_posterior':
        save_M2_posterior(drug_name)
    elif to_run == 'full_inference':
        if idx_model == -1:
            ABC.run(idxs_models = [0, 1, 2], num_populations = 200, numPops_save_every = 3)
        else:
            ABC.run(idxs_models = [idx_model], num_populations = 200, numPops_save_every = 3)
