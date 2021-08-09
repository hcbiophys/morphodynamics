import pickle
import numpy as np
import sys
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import time
import pyabc

from morphodynamics.tip_model.L import utils


drug_name = sys.argv[1]
to_run = sys.argv[2] # 'MAP_simulations' or 'full_inference'
db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db"))

class L_ABC():
    """
    Class for approximate Bayesian computation with sequential Monte Carlo (ABC-SMC) on fungus lengthening
    """

    def __init__(self, drug_name, num_runs, pop_size):
        """
        drug_name
        num_runs: number of single fungus simulations per histogram comparison
        pop_size: number of histograms comparisons for each ABC-SMC iteration (sequential iterations have lower acceptance thresholds)
        """

        self.bins = 20 # histogram dimensions for comparison

        # load the length data for comparison
        path_to_here = os.path.dirname(os.path.realpath(__file__))
        path_lengths = os.path.join(path_to_here, '../data/fungus_lengths/{}.pickle'.format(drug_name))
        file = open(path_lengths, 'rb')
        data_nestedLists_lengths = pickle.load(file)
        self.data_hists = []
        for idx_time in range(9):
            hist, _ = np.histogram(data_nestedLists_lengths[idx_time], bins = self.bins, range = [0, 150], density = True)
            self.data_hists.append(hist)

        self.drug_name = drug_name


        self.times = [90, 105, 120, 135, 150, 165, 180, 195, 210]


        self.num_runs = num_runs
        self.pop_size = pop_size



        #data_nestedLists_lengths = removeXfrac_spores(data_nestedLists_lengths, frac_stay)


        # prior distribution limits, note the notation here is same as in scipy, not same as in paper notation
        self.t_germ_s_min = 0.01
        self.t_germ_s_max = 5
        self.t_germ_loc_min = 20
        self.t_germ_loc_max = 100
        self.t_germ_scale_min= 0.01
        self.t_germ_scale_max = 150

        self.grad_s_min = 0.01
        self.grad_s_max = 1
        self.grad_loc_min = 0.3
        self.grad_loc_max = 1.5
        self.grad_scale_min = 0.01
        self.grad_scale_max = 2

        # Use uniform distributions as priors
        self.priors = pyabc.Distribution(t_germ_s = pyabc.RV("uniform", self.t_germ_s_min, self.t_germ_s_max-self.t_germ_s_min),
                                        t_germ_loc = pyabc.RV("uniform", self.t_germ_loc_min, self.t_germ_loc_max-self.t_germ_loc_min),
                                        t_germ_scale = pyabc.RV("uniform", self.t_germ_scale_min, self.t_germ_scale_max-self.t_germ_scale_min),
                                        grad_s = pyabc.RV("uniform", self.grad_s_min, self.grad_s_max-self.grad_s_min),
                                        grad_loc = pyabc.RV("uniform", self.grad_loc_min, self.grad_loc_max-self.grad_loc_min),
                                        grad_scale = pyabc.RV("uniform", self.grad_scale_min, self.grad_scale_max-self.grad_scale_min))



    def simulation(self, parameters):
        """
        Generate histogram over simulation morphospace embeddings
        """


        t_germ_s = parameters['t_germ_s']
        t_germ_loc = parameters['t_germ_loc']
        t_germ_scale = parameters['t_germ_scale']

        grad_s = parameters['grad_s']
        grad_loc = parameters['grad_loc']
        grad_scale = parameters['grad_scale']

        hists_sim = utils.get_sim_hists(self.times,  t_germ_s, t_germ_loc, t_germ_scale, grad_s, grad_loc, grad_scale, self.num_runs, self.bins)


        return {'X_2' : hists_sim}




    def run(self, end_time, numPops_save_every):
        """
        Run ABC-SMC
        - end_time: time to stop at
        - numPops_save_every: save image outputs every [this number] of populations

        """

        start_time = time.time()

        count_pop = 0
        time_taken = time.time() - start_time
        while time_taken < end_time:


            abc_object = pyabc.ABCSMC(models = self.simulation,
                         parameter_priors = self.priors,
                         distance_function = utils.distance_abs,
                         population_size = self.pop_size) # each t (with uniques epsilon continues until 10 have succeeded)
                         #sampler = pyabc.sampler.SingleCoreSampler())


            if count_pop == 0:
                abc_object.new(db_path, {"X_2": self.data_hists})
            else:
                abc_object.load(db_path, load_id)

            history = abc_object.run(minimum_epsilon = 1e-20, max_nr_populations = numPops_save_every) # each population is a t (with unique epsilon)
            load_id = history.id

            dfw = history.get_distribution(m=0) # m is model index (there is only one model used for the lengthening, hence index 0)


            grid = pyabc.visualization.plot_kde_matrix(*dfw,
            limits={"t_germ_s": (self.t_germ_s_min, self.t_germ_s_max),
                    "t_germ_loc": (self.t_germ_loc_min, self.t_germ_loc_max),
                        "t_germ_scale": (self.t_germ_scale_min, self.t_germ_scale_max),
                        "grad_s": (self.grad_s_min, self.grad_s_max),
                        "grad_loc": (self.grad_loc_min, self.grad_loc_max),
                        "grad_scale": (self.grad_scale_min, self.grad_scale_max)})
            #plt.gcf().set_size_inches(8, 8)

            path_to_here = os.path.dirname(os.path.realpath(__file__))
            plt.savefig(path_to_here+'/../outputs/{}_{}kde.png'.format(self.drug_name, count_pop))
            plt.close()


            count_pop += numPops_save_every
            time_taken = (time.time() - start_time)/60
            print('time_taken :', time_taken)





    def MAPs(self, drug_name):


        # order: t_germ_s, t_germ_loc, t_germ_scale, grad_s, grad_loc, grad_scale
        param_dict = {'DMSO':[4.55, 78, 1.58, 0.34, 1.24, 0.55],
        'compound_A': [4.84, 75, 4.12, 0.18, 1.42, 0.94],
        'compound_C_0_041': [1.85, 68.4, 36.6, 0.62, 0.95, 0.30],
        'compound_X': [2.45, 66.2, 61.4, 0.14, 1.09, 0.64]}


        utils.plot_MAP_comparison(self.times, self.data_hists, self.num_runs, self.bins, drug_name, param_dict)




if __name__ == "__main__":

    ABC = L_ABC(drug_name, num_runs = 5000, pop_size = 100)
    if to_run == 'MAP_simulations':
        ABC.MAPs(drug_name)
    elif to_run == 'full_inference':
        ABC.run(end_time = 50, numPops_save_every = 3)
