import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def vio_mc_plotter(
    x_mc:NDArray,
    x_error_mc: NDArray,
    x_mc_mean: NDArray,
    x_mc_std: NDArray,
    x_error_mc_mean: NDArray,
    x_error_mc_std: NDArray,
    t: list | NDArray,
    sigma_bound_fator: int = 1,
    interactive:bool=False,
    error_only:bool=False,
):
    if interactive:
        import matplotlib
        matplotlib.use('ipympl')
    
    # extract raw monte carlo states and errors
    N_mc = x_mc[0].transpose()
    E_mc = x_mc[1].transpose()
    yaw_mc = x_mc[2].transpose()
    N_error_mc = x_error_mc[0].transpose()
    E_error_mc = x_error_mc[1].transpose()
    yaw_error_mc = x_error_mc[2].transpose()

    # extract means/stds for each state and state error
    # state means
    N_mean = x_mc_mean[0]
    E_mean = x_mc_mean[1]
    yaw_mean = x_mc_mean[2]
    # state mc stds
    N_std = x_mc_std[0]
    E_std = x_mc_std[1]
    yaw_std = x_mc_std[2]

    # state error means
    N_error_mean = x_error_mc_mean[0]
    E_error_mean = x_error_mc_mean[1]
    yaw_error_mean = x_error_mc_mean[2]
    # state error stds
    N_error_std = x_error_mc_std[0]
    E_error_std = x_error_mc_std[1]
    yaw_error_std = x_error_mc_std[2]

    # ---- state error plots ----
    # North error
    plt.figure()
    plt.plot(t, N_error_mc, linewidth=1, alpha=0.4, color='gray')
    plt.plot(t, N_error_mean + sigma_bound_fator * N_error_std, 'r', label='Experimental bounds')
    plt.plot(t, N_error_mean - sigma_bound_fator * N_error_std, 'r')
    plt.legend()
    plt.title('North Position Error')
    plt.xlabel('Time [s]')
    plt.ylabel('[m]')
    plt.show()

    # East
    plt.figure()
    plt.plot(t, E_error_mc, linewidth=1, alpha=0.4, color='gray')
    plt.plot(t, E_error_mean + sigma_bound_fator * E_error_std, 'r', label='Experimental bounds')
    plt.plot(t, E_error_mean - sigma_bound_fator * E_error_std, 'r')
    plt.legend()
    plt.title('East Position Error')
    plt.xlabel('Time [s]')
    plt.ylabel('[m]')
    plt.show()

    # yaw
    plt.figure()
    plt.plot(t, np.rad2deg(yaw_error_mc), linewidth=1, alpha=0.4, color='gray')
    plt.plot(t, np.rad2deg(yaw_error_mean + sigma_bound_fator * yaw_error_std), 'r', label='Experimental bounds')
    plt.plot(t, np.rad2deg(yaw_error_mean - sigma_bound_fator * yaw_error_std), 'r')
    plt.legend()
    plt.title('Yaw Error')
    plt.xlabel('Time [s]')
    plt.ylabel('[deg]')
    plt.show()

    # ---- state plots ----
    if not error_only:
        # North
        plt.figure()
        plt.plot(t, N_mc, linewidth=1, alpha=0.4, color='gray')
        plt.plot(t, N_mean + sigma_bound_fator * N_std, 'r', label='Experimental bounds')
        plt.plot(t, N_mean - sigma_bound_fator * N_std, 'r')
        plt.legend()
        plt.title('North Position')
        plt.xlabel('Time [s]')
        plt.ylabel('[m]')
        plt.show()

        # East
        plt.figure()
        plt.plot(t, E_mc, linewidth=1, alpha=0.4, color='gray')
        plt.plot(t, E_mean + sigma_bound_fator * E_std, 'r', label='Experimental bounds')
        plt.plot(t, E_mean - sigma_bound_fator * E_std, 'r')
        plt.legend()
        plt.title('East Position')
        plt.xlabel('Time [s]')
        plt.ylabel('[m]')
        plt.show()

        # yaw
        plt.figure()
        plt.plot(t, np.rad2deg(yaw_mc), linewidth=1, alpha=0.4, color='gray')
        plt.plot(t, np.rad2deg(yaw_mean + sigma_bound_fator * yaw_std), 'r', label='Experimental bounds')
        plt.plot(t, np.rad2deg(yaw_mean - sigma_bound_fator * yaw_std), 'r')
        plt.legend()
        plt.title('Yaw')
        plt.xlabel('Time [s]')
        plt.ylabel('[deg]')
        plt.show()