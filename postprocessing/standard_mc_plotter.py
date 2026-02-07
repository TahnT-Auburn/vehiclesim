import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def standard_mc_plotter(x_mc: NDArray,
                        x_error_mc: NDArray,
                        x_mc_mean: NDArray,
                        x_mc_std: NDArray,
                        x_error_mc_mean: NDArray,
                        x_error_mc_std: NDArray,
                        theo_std: NDArray,
                        t: list | NDArray,
                        sigma_bound_fator: int = 1,
                        interactive:bool=False,
                        error_only:bool=False,):
    """
    Generates plots for the states and errors of the monte carlo simulation

    Args:
        x_mc (NDArray): MC states (n,mc,L).
        x_error_mc (NDArray): MC error states (n,mc,L).
        x_mc_mean (NDArray): MC mean of states (n,L).
        x_mc_std (NDArray): MC std of states (n,L).
        x_error_mc_mean (NDArray): MC mean of state errors (n,L).
        x_error_mc_std (NDArray): MC std of state errors (n,L).
        theo_std (NDArray): Theoretical std for each state according to the filter (n,L)
        t (list | NDArray): Time vector (L).
        sigma_bound_factor (int): The desired sigma bound to be plotted. Tpyically 1-3.
        interactive (bool, optional). Switch between interactive or non-interactive plots. Deafult is False.
        error_only (bool, optional). Plots the error statistics only. Default is False.
    """
    if interactive:
        import matplotlib
        matplotlib.use('ipympl')

    # extract raw monte carlo states and errors
    N_mc = x_mc[0].transpose()
    E_mc = x_mc[1].transpose()
    vx_mc = x_mc[2].transpose()
    vy_mc = x_mc[3].transpose()
    yaw_rate_mc = x_mc[4].transpose()
    yaw_mc = x_mc[5].transpose()
    hitch_rate_mc = x_mc[6].transpose()
    hitch_mc = x_mc[7].transpose()
    bias_yr_mc = x_mc[8].transpose()

    N_error_mc = x_error_mc[0].transpose()
    E_error_mc = x_error_mc[1].transpose()
    vx_error_mc = x_error_mc[2].transpose()
    vy_error_mc = x_error_mc[3].transpose()
    yaw_rate_error_mc = x_error_mc[4].transpose()
    yaw_error_mc = x_error_mc[5].transpose()
    hitch_rate_error_mc = x_error_mc[6].transpose()
    hitch_error_mc = x_error_mc[7].transpose()
    bias_yr_error_mc = x_error_mc[8].transpose()

    # extract means/stds for each state and state error
    # state means
    N_mean = x_mc_mean[0]
    E_mean = x_mc_mean[1]
    vx_mean = x_mc_mean[2]
    vy_mean = x_mc_mean[3]
    yaw_rate_mean = x_mc_mean[4]
    yaw_mean = x_mc_mean[5]
    hitch_rate_mean = x_mc_mean[6]
    hitch_mean = x_mc_mean[7]
    bias_yr_mean = x_mc_mean[8]

    # state mc stds
    N_std = x_mc_std[0]
    E_std = x_mc_std[1]
    vx_std = x_mc_std[2]
    vy_std = x_mc_std[3]
    yaw_rate_std = x_mc_std[4]
    yaw_std = x_mc_std[5]
    hitch_rate_std = x_mc_std[6]
    hitch_std = x_mc_std[7]
    bias_yr_std = x_mc_std[8]

    # state error means
    N_error_mean = x_error_mc_mean[0]
    E_error_mean = x_error_mc_mean[1]
    vx_error_mean = x_error_mc_mean[2]
    vy_error_mean = x_error_mc_mean[3]
    yaw_rate_error_mean = x_error_mc_mean[4]
    yaw_error_mean = x_error_mc_mean[5]
    hitch_rate_error_mean = x_error_mc_mean[6]
    hitch_error_mean = x_error_mc_mean[7]
    bias_yr_error_mean = x_error_mc_mean[8]

    # state error stds
    N_error_std = x_error_mc_std[0]
    E_error_std = x_error_mc_std[1]
    vx_error_std = x_error_mc_std[2]
    vy_error_std = x_error_mc_std[3]
    yaw_rate_error_std = x_error_mc_std[4]
    yaw_error_std = x_error_mc_std[5]
    hitch_rate_error_std = x_error_mc_std[6]
    hitch_error_std = x_error_mc_std[7]
    bias_yr_error_std = x_error_mc_std[8]

    # theo stds
    N_theo_std = theo_std[0]
    E_theo_std = theo_std[1]
    vx_theo_std = theo_std[2]
    vy_theo_std = theo_std[3]
    yaw_rate_theo_std = theo_std[4]
    yaw_theo_std = theo_std[5]
    hitch_rate_theo_std = theo_std[6]
    hitch_theo_std = theo_std[7]
    bias_yr_theo_std = theo_std[8]

    # ---- state error plots ----
    # North error
    plt.figure()
    plt.plot(t, N_error_mc, linewidth=1, alpha=0.4, color='gray')
    plt.plot(t, N_error_mean + sigma_bound_fator * N_error_std, 'r', label='Experimental bounds')
    plt.plot(t, 0 + sigma_bound_fator * N_theo_std, '--k', label='Theoretical bounds')
    plt.plot(t, N_error_mean - sigma_bound_fator * N_error_std, 'r')
    plt.plot(t, 0 - sigma_bound_fator * N_theo_std, '--k')
    plt.legend()
    plt.title('North Position Error')
    plt.xlabel('Time [s]')
    plt.ylabel('[m]')
    plt.show()

    # East
    plt.figure()
    plt.plot(t, E_error_mc, linewidth=1, alpha=0.4, color='gray')
    plt.plot(t, E_error_mean + sigma_bound_fator * E_error_std, 'r', label='Experimental bounds')
    plt.plot(t, 0 + sigma_bound_fator * E_theo_std, '--k', label='Theoretical bounds')
    plt.plot(t, E_error_mean - sigma_bound_fator * E_error_std, 'r')
    plt.plot(t, 0 - sigma_bound_fator * E_theo_std, '--k')
    plt.legend()
    plt.title('East Position Error')
    plt.xlabel('Time [s]')
    plt.ylabel('[m]')
    plt.show()

    # vx
    plt.figure()
    plt.plot(t, vx_error_mc, linewidth=1, alpha=0.4, color='gray')
    plt.plot(t, vx_error_mean + sigma_bound_fator * vx_error_std, 'r', label='Experimental bounds')
    plt.plot(t, 0 + sigma_bound_fator * vx_theo_std, '--k', label='Theoretical bounds')
    plt.plot(t, vx_error_mean - sigma_bound_fator * vx_error_std, 'r')
    plt.plot(t, 0 - sigma_bound_fator * vx_theo_std, '--k')
    plt.legend()
    plt.title('Longitudinal Velocity Error')
    plt.xlabel('Time [s]')
    plt.ylabel('[m/s]')
    plt.show()

    # vy
    plt.figure()
    plt.plot(t, vy_error_mc, linewidth=1, alpha=0.4, color='gray')
    plt.plot(t, vy_error_mean + sigma_bound_fator * vy_error_std, 'r', label='Experimental bounds')
    plt.plot(t, 0 + sigma_bound_fator * vy_theo_std, '--k', label='Theoretical bounds')
    plt.plot(t, vy_error_mean - sigma_bound_fator * vy_error_std, 'r')
    plt.plot(t, 0 - sigma_bound_fator * vy_theo_std, '--k')
    plt.legend()
    plt.title('Lateral Velocity Error')
    plt.xlabel('Time [s]')
    plt.ylabel('[m/s]')
    plt.show()

    # yaw rate
    plt.figure()
    plt.plot(t, np.rad2deg(yaw_rate_error_mc), linewidth=1, alpha=0.4, color='gray')
    plt.plot(t, np.rad2deg(yaw_rate_error_mean + sigma_bound_fator * yaw_rate_error_std), 'r', label='Experimental bounds')
    plt.plot(t, np.rad2deg(0 + sigma_bound_fator * yaw_rate_theo_std), '--k', label='Theoretical bounds')
    plt.plot(t, np.rad2deg(yaw_rate_error_mean - sigma_bound_fator * yaw_rate_error_std), 'r')
    plt.plot(t, np.rad2deg(0 - sigma_bound_fator * yaw_rate_theo_std), '--k')
    plt.legend()
    plt.title('Yaw Rate Error')
    plt.xlabel('Time [s]')
    plt.ylabel('[deg/s]')
    plt.show()

    # yaw
    plt.figure()
    plt.plot(t, np.rad2deg(yaw_error_mc), linewidth=1, alpha=0.4, color='gray')
    plt.plot(t, np.rad2deg(yaw_error_mean + sigma_bound_fator * yaw_error_std), 'r', label='Experimental bounds')
    plt.plot(t, np.rad2deg(0 + sigma_bound_fator * yaw_theo_std), '--k', label='Theoretical bounds')
    plt.plot(t, np.rad2deg(yaw_error_mean - sigma_bound_fator * yaw_error_std), 'r')
    plt.plot(t, np.rad2deg(0 - sigma_bound_fator * yaw_theo_std), '--k')
    plt.legend()
    plt.title('Yaw Error')
    plt.xlabel('Time [s]')
    plt.ylabel('[deg]')
    plt.show()

    # hitch rate
    plt.figure()
    plt.plot(t, np.rad2deg(hitch_rate_error_mc), linewidth=1, alpha=0.4, color='gray')
    plt.plot(t, np.rad2deg(hitch_rate_error_mean + sigma_bound_fator * hitch_rate_error_std), 'r', label='Experimental bounds')
    plt.plot(t, np.rad2deg(0 + sigma_bound_fator * hitch_rate_theo_std), '--k', label='Theoretical bounds')
    plt.plot(t, np.rad2deg(hitch_rate_error_mean - sigma_bound_fator * hitch_rate_error_std), 'r')
    plt.plot(t, np.rad2deg(0 - sigma_bound_fator * hitch_rate_theo_std), '--k')
    plt.legend()
    plt.title('Hitch Rate Error')
    plt.xlabel('Time [s]')
    plt.ylabel('[deg/s]')
    plt.show()

    # hitch
    plt.figure()
    plt.plot(t, np.rad2deg(hitch_error_mc), linewidth=1, alpha=0.4, color='gray')
    plt.plot(t, np.rad2deg(hitch_error_mean + sigma_bound_fator * hitch_error_std), 'r', label='Experimental bounds')
    plt.plot(t, np.rad2deg(0 + sigma_bound_fator * hitch_theo_std), '--k', label='Theoretical bounds')
    plt.plot(t, np.rad2deg(hitch_error_mean - sigma_bound_fator * hitch_error_std), 'r')
    plt.plot(t, np.rad2deg(0 - sigma_bound_fator * hitch_theo_std), '--k')
    plt.legend()
    plt.title('Hitch Error')
    plt.xlabel('Time [s]')
    plt.ylabel('[deg]')
    plt.show()

    # ---- state plots ----
    if not error_only:
        # North
        plt.figure()
        plt.plot(t, N_mc, linewidth=1, alpha=0.4, color='gray')
        plt.plot(t, N_mean + sigma_bound_fator * N_std, 'r', label='Experimental bounds')
        plt.plot(t, N_mean + sigma_bound_fator * N_theo_std, '--k', label='Theoretical bounds')
        plt.plot(t, N_mean - sigma_bound_fator * N_std, 'r')
        plt.plot(t, N_mean - sigma_bound_fator * N_theo_std, '--k')
        plt.legend()
        plt.title('North Position')
        plt.xlabel('Time [s]')
        plt.ylabel('[m]')
        plt.show()

        # East
        plt.figure()
        plt.plot(t, E_mc, linewidth=1, alpha=0.4, color='gray')
        plt.plot(t, E_mean + sigma_bound_fator * E_std, 'r', label='Experimental bounds')
        plt.plot(t, E_mean + sigma_bound_fator * E_theo_std, '--k', label='Theoretical bounds')
        plt.plot(t, E_mean - sigma_bound_fator * E_std, 'r')
        plt.plot(t, E_mean - sigma_bound_fator * E_theo_std, '--k')
        plt.legend()
        plt.title('East Position')
        plt.xlabel('Time [s]')
        plt.ylabel('[m]')
        plt.show()

        # vx
        plt.figure()
        plt.plot(t, vx_mc, linewidth=1, alpha=0.4, color='gray')
        plt.plot(t, vx_mean + sigma_bound_fator * vx_std, 'r', label='Experimental bounds')
        plt.plot(t, vx_mean + sigma_bound_fator * vx_theo_std, '--k', label='Theoretical bounds')
        plt.plot(t, vx_mean - sigma_bound_fator * vx_std, 'r')
        plt.plot(t, vx_mean - sigma_bound_fator * vx_theo_std, '--k')
        plt.legend()
        plt.title('Longitudinal Velocity')
        plt.xlabel('Time [s]')
        plt.ylabel('[m/s]')
        plt.show()

        # vy
        plt.figure()
        plt.plot(t, vy_mc, linewidth=1, alpha=0.4, color='gray')
        plt.plot(t, vy_mean + sigma_bound_fator * vy_std, 'r', label='Experimental bounds')
        plt.plot(t, vy_mean + sigma_bound_fator * vy_theo_std, '--k', label='Theoretical bounds')
        plt.plot(t, vy_mean - sigma_bound_fator * vy_std, 'r')
        plt.plot(t, vy_mean - sigma_bound_fator * vy_theo_std, '--k')
        plt.legend()
        plt.title('Lateral Velocity')
        plt.xlabel('Time [s]')
        plt.ylabel('[m/s]')
        plt.show()

        # yaw rate
        plt.figure()
        plt.plot(t, np.rad2deg(yaw_rate_mc), linewidth=1, alpha=0.4, color='gray')
        plt.plot(t, np.rad2deg(yaw_rate_mean + sigma_bound_fator * yaw_rate_std), 'r', label='Experimental bounds')
        plt.plot(t, np.rad2deg(yaw_rate_mean + sigma_bound_fator * yaw_rate_theo_std), '--k', label='Theoretical bounds')
        plt.plot(t, np.rad2deg(yaw_rate_mean - sigma_bound_fator * yaw_rate_std), 'r')
        plt.plot(t, np.rad2deg(yaw_rate_mean - sigma_bound_fator * yaw_rate_theo_std), '--k')
        plt.legend()
        plt.title('Yaw Rate')
        plt.xlabel('Time [s]')
        plt.ylabel('[deg/s]')
        plt.show()

        # yaw
        plt.figure()
        plt.plot(t, np.rad2deg(yaw_mc), linewidth=1, alpha=0.4, color='gray')
        plt.plot(t, np.rad2deg(yaw_mean + sigma_bound_fator * yaw_std), 'r', label='Experimental bounds')
        plt.plot(t, np.rad2deg(yaw_mean + sigma_bound_fator * yaw_theo_std), '--k', label='Theoretical bounds')
        plt.plot(t, np.rad2deg(yaw_mean - sigma_bound_fator * yaw_std), 'r')
        plt.plot(t, np.rad2deg(yaw_mean - sigma_bound_fator * yaw_theo_std), '--k')
        plt.legend()
        plt.title('Yaw')
        plt.xlabel('Time [s]')
        plt.ylabel('[deg]')
        plt.show()

        # hitch rate
        plt.figure()
        plt.plot(t, np.rad2deg(hitch_rate_mc), linewidth=1, alpha=0.4, color='gray')
        plt.plot(t, np.rad2deg(hitch_rate_mean + sigma_bound_fator * hitch_rate_std), 'r', label='Experimental bounds')
        plt.plot(t, np.rad2deg(hitch_rate_mean + sigma_bound_fator * hitch_rate_theo_std), '--k', label='Theoretical bounds')
        plt.plot(t, np.rad2deg(hitch_rate_mean - sigma_bound_fator * hitch_rate_std), 'r')
        plt.plot(t, np.rad2deg(hitch_rate_mean - sigma_bound_fator * hitch_rate_theo_std), '--k')
        plt.legend()
        plt.title('Hitch')
        plt.xlabel('Time [s]')
        plt.ylabel('[deg]')
        plt.show()

        # hitch
        plt.figure()
        plt.plot(t, np.rad2deg(hitch_mc), linewidth=1, alpha=0.4, color='gray')
        plt.plot(t, np.rad2deg(hitch_mean + sigma_bound_fator * hitch_std), 'r', label='Experimental bounds')
        plt.plot(t, np.rad2deg(hitch_mean + sigma_bound_fator * hitch_theo_std), '--k', label='Theoretical bounds')
        plt.plot(t, np.rad2deg(hitch_mean - sigma_bound_fator * hitch_std), 'r')
        plt.plot(t, np.rad2deg(hitch_mean - sigma_bound_fator * hitch_theo_std), '--k')
        plt.legend()
        plt.title('Hitch')
        plt.xlabel('Time [s]')
        plt.ylabel('[deg]')
        plt.show()