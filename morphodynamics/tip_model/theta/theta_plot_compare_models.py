import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

idx_model = int(sys.argv[1])


path_to_here = os.path.dirname(os.path.realpath(__file__))

def M0(sigma):
    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_subplot(111)

    for _ in range(5):

        ts = [0]
        thetas = [0]

        dt = 1
        for idx in range(210):
            dtheta = sigma*np.sqrt(dt)*np.random.normal(0, 1, 1)[0]
            theta = thetas[idx] + dtheta
            thetas.append(theta)

            ts.append(ts[idx] + dt)

        ax.plot(ts, thetas, linewidth = 1)
    ax.set_ylim([-3.14, 3.14])
    ax.tick_params(axis="both", direction="in", labelsize = 6)
    ax.set_ylabel(r'$\theta_{global} \; (rad)$', fontsize = 6, labelpad = 0.2)
    ax.set_xlabel('Time (min)', fontsize = 6, labelpad = 0.2)
    plt.tight_layout()
    plt.savefig(path_to_here+'/../../outputs/M0.png', dpi = 1200)




def M1_M2(idx_model, *args):
    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_subplot(111)

    for _ in range(5):
        thetas = [0]
        ts = [0]
        dt = 1
        for idx in range(210):
            if idx_model == 1:
                dtheta = args[0]*np.sqrt(dt)*np.random.normal(0, 1, 1)[0]
            elif idx_model == 2:
                dtheta = args[0]*np.sqrt(dt)*np.random.normal(0, 1, 1)[0] - args[1]*thetas[idx]*dt

            theta = thetas[idx] + dtheta
            thetas.append(theta)

            ts.append(ts[idx] + dt)

        global_thetas = []
        tip_thetas = []
        for idx_step in range(len(thetas)):
            tip_theta = thetas[idx_step]
            tip_thetas.append(tip_theta)
            global_theta = 0.75*dt*np.cumsum(thetas[:idx_step+1])[-1]
            global_thetas.append(global_theta)

        ax.plot(ts, global_thetas, linewidth = 1)
    ax.set_ylim([-3.14, 3.14])
    ax.tick_params(axis="both", direction="in", labelsize = 6)
    ax.set_xlabel('Time (min)', fontsize = 6, labelpad = 0.2)
    plt.tight_layout()
    plt.savefig(path_to_here+'/../../outputs/M{}.png'.format(idx_model), dpi = 1200)




if __name__ == '__main__':

    if idx_model == 0:
        M0(sigma = 0.126)
    elif idx_model == 1:
        M1_M2(1, 0.0065)
    elif idx_model == 2:
        M1_M2(2, 0.040, 0.1386)
