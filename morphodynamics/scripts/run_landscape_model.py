import sys
import os
import tensorflow as tf
import sys

tf.keras.backend.set_floatx('float32')

from morphodynamics.landscapes.model import Landscape_Model
from morphodynamics.landscapes.analysis.save_sde_objs import SDE_Mixin
from morphodynamics.landscapes.utils import make_dir_weights
from morphodynamics.data.embeddings.lims13.load_embeddings import load_func

drug_name = sys.argv[1]
train_load = sys.argv[2]
train_time_hrs = float(sys.argv[3])
save_num = int(sys.argv[4])


xlims = [-13, 13] #limits beyond [-10, 10] to aid fulfilling zero boundary condition
ylims = [-13, 13]
tlims = [0, 120] # [90, 210]

dims = 200

batch_size = 10000

learning_rate = 5e-4

num_collocation = 1000000 # number of points (uniformly distributed) to get PDE residual at
num_BC = 1000000 # number of boundary condition points

# hyperparameters in the total loss, L_total. These are a, b, c & d in the paper
data_weight = 1
BC_weight = 1
pde_weight = 500
norm_weight = 0.01

# number of neurons and layers used in the networks for the pdf and landscape
layers_p = [3] + [50, 50, 50, 50, 50] + [1]
layers_U = [2] + [50, 50, 50, 50, 50] + [1]
layers_D = [3] + [50, 50, 50, 50, 50] + [1]

kde_bw = 0.2



pdf_list = load_func('lims13_{}.pickle'.format(drug_name)) # load the PDF data


save_append = '{}_P{}_PDE{}_BC{}_n{}'.format(drug_name, data_weight, pde_weight, BC_weight, norm_weight) # save outputs with info on the drug_name and hyperparameters a, b, c & d
model = Landscape_Model(pdf_list, kde_bw,
                xlims, ylims, tlims, dims, layers_p, layers_U, layers_D,
                data_weight, pde_weight, BC_weight,  norm_weight,
                num_collocation, num_BC,
                batch_size, learning_rate, save_append)

# load the PINN weights
path_to_here = os.path.dirname(os.path.realpath(__file__))
dir_weights = path_to_here+'/../data/network_weights/PINN/{}/'.format(drug_name)


dir_save_vis = path_to_here+'/../data/landscape_visualizations/' # where to save visualizations and data for reconstructing landscapes



# -------------------

# landscape training
if train_load == 'train':

    model.idx_save = 0
    for _ in range(save_num):

        model.train(total_time = train_time_hrs / save_num)
        model.save_networks(dir_weights = path_to_here+'/../outputs/')
        model.save_ims(save_dir = path_to_here+'/../outputs/')
        model.idx_save += 1

# Outputting objects for visualization (single particle simulations (equation 1 in the paper) & landscape as an array)
elif train_load == 'load':
    idxs_load = {'DMSO':34, 'compound_A':40, 'compound_X':40, 'compound_C_0_041':40, 'compound_C_10':32, 'compound_B':40}
    model.idx_save = idxs_load[drug_name] # this is part of the path name loaded

    model.load_networks(dir_weights = dir_weights, idx_load = model.idx_save)
    sde_objs_path = path_to_here+'/../outputs/new_sde_objs/{}/'.format(drug_name)
    if not os.path.exists(path_to_here+'/../outputs/new_sde_objs/'):
        os.mkdir(path_to_here+'/../outputs/new_sde_objs/')
    os.mkdir(sde_objs_path)
    model.save_ims(save_dir = sde_objs_path + '/')
    model.save_sde_objs(sde_num_particles = 200, sde_objs_path = sde_objs_path)
