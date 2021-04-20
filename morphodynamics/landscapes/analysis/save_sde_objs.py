from morphodynamics.landscapes.analysis.get_fields import *
from morphodynamics.landscapes.analysis.sde_forward import Run_SDE


class SDE_Mixin():

    def save_sde_objs(self, sde_num_particles, sde_objs_path):
        """
        Run and save simulations of eq. 1
        - sde_num_particles: number of particles to simulate
        - sde_objs_path: directory to save outputs into
        """


        p_lists = get_listsOf_pArrays(model = self, num_snaps = 9, dims0 = 5000, dimsN = 200, save_path = sde_objs_path + 'p_list_{}.pickle')
        F_array = get_F_field(model = self, dims = 5000, save_path = sde_objs_path + 'F_array.pickle')
        D_lists = get_listsOf_DArrays(model = self, num_Ds = 18, dims = 200, save_path = sde_objs_path + 'D_list_{}.pickle')

        sde_fwd = Run_SDE([i.reshape((200, 200)) for i in self.pdf_list], p_lists[0], F_array, D_lists[0], self.xlims, self.ylims, kde_bw = self.kde_bw)
        sde_fwd.run(num_particles = sde_num_particles, dt = 0.01, T = 120)
        sde_fwd.set_kdes_data_2(error_bw = 0.2)
        sde_fwd.set_kdes_nn(error_bw = 0.2)
        sde_fwd.pickle_trajectories(keep_every = 50, save_path = sde_objs_path + 'subsampled_paths_p{}.pickle'.format(0))
        sde_fwd.pickle_for_errors(save_path1 = sde_objs_path + 'kdes_data_2.pickle', save_path2 = sde_objs_path + 'p_nn.pickle', save_path3 = sde_objs_path + 'kdes_nn.pickle')
