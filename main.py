import numpy as np
import matplotlib.pyplot as plt
import sys
from control.utils import Keypoints3D as K

if __name__=="__main__":
    print(sys.argv)
    data = np.load('./inference/output_dir/{}.npy'.format(sys.argv[1]),
        allow_pickle=True)
    # with 
    # print(data)
    # fig,ax = plt.subplots(3)
    # n_plots = 17
    print(data.shape)
    plot_ax = 1 # either 1 (DoF) or 2 (dimension)
    n_plots = data.shape[plot_ax]
    fig,ax = plt.subplots(n_plots,figsize=(6,6*(3-plot_ax)))
    for i in range(n_plots):
        # ax[i].plot(data[...,i])
        if plot_ax==1:
            ax[i].plot(data[:,i,:])
            ax[i].set_ylabel(K(i).name,fontsize=8)
        elif plot_ax==2:
            ax[i].plot(data[...,i])
    plt.savefig('./inference/output_dir/{}.png'.format(sys.argv[1]))
    plt.show(block=False)
    input('Press Enter to quit and exit')
    # import pdb; pdb.set_trace();

# arms = 13, 16