import numpy as np
import matplotlib.pyplot as plt

def standard_state_est_plotter(x, x_truth, t):
    """
    Generates plots for the standard 9-state estimates vs truth and their errors.

    Args:
        x (list): List of state estimates.
        x_truth (list): List of state truths.
        t (list): Time.
    """
    N = x[0]
    E = x[1]
    vx = x[2]
    vy = x[3]
    yaw_rate = x[4]
    yaw = x[5]
    hitch_rate = x[6]
    hitch = x[7]
    bias_yr = x[8]
    
    N_truth = x_truth[0]
    E_truth = x_truth[1]
    vx_truth = x_truth[2]
    vy_truth = x_truth[3]
    yaw_rate_truth = x_truth[4]
    yaw_truth = x_truth[5]
    hitch_rate_truth = x_truth[6]
    hitch_truth = x_truth[7]
    
    # absolute position
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Local NE position')
    ax1.plot(E_truth,N_truth)
    ax1.plot(E,N)
    ax1.axis('equal')
    ax1.set_ylabel('Northing [m]')
    ax1.set_xlabel('Easting [m]')
    abs_error = compute_abs_pos_error((E_truth, N_truth), (E,N))
    ax2.plot(t, abs_error)
    ax2.set_ylabel('Abs position error [m]')
    ax2.set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()
    
    # north east positions
    fig, axs = plt.subplots(2,2)
    fig.suptitle('North and East Positions')
    axs[0,0].plot(t, N_truth)
    axs[0,0].plot(t, N)
    axs[0,0].set_ylabel('Northing [m]')
    axs[0,0].set_xlabel('Time [s]')
    axs[0,1].plot(t, N_truth - N)
    axs[0,1].set_ylabel('Northing Error [m]')
    axs[0,1].set_xlabel('Time [s]')
    axs[1,0].plot(t, E_truth)
    axs[1,0].plot(t, E)
    axs[1,0].set_ylabel('Easting [m]')
    axs[1,0].set_xlabel('Time [s]')
    axs[1,1].plot(t, E_truth - E)
    axs[1,1].set_ylabel('Easting Error [m]')
    axs[1,1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()
    
    # velocities
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Longitudinal and Lateral Velocities')
    axs[0,0].plot(t, vx_truth)
    axs[0,0].plot(t, vx)
    axs[0,0].set_ylabel('Vx [m/s]')
    axs[0,0].set_xlabel('Time [s]')
    axs[0,1].plot(t, vx_truth - vx)
    axs[0,1].set_ylabel('Vx Error [m/s]')
    axs[0,1].set_xlabel('Time [s]')
    axs[1,0].plot(t, vy_truth)
    axs[1,0].plot(t, vy)
    axs[1,0].set_ylabel('Vy [m/s]')
    axs[1,0].set_xlabel('Time [s]')
    axs[1,1].plot(t, vy_truth - vy)
    axs[1,1].set_ylabel('Vy Error [m/s]')
    axs[1,1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()
    
    # yaw rate and yaw
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Yaw Rate and Yaw')
    axs[0,0].plot(t, np.rad2deg(yaw_rate_truth))
    axs[0,0].plot(t, np.rad2deg(yaw_rate))
    axs[0,0].set_ylabel('Yaw Rate [deg/s]')
    axs[0,0].set_xlabel('Time [s]')
    axs[0,1].plot(t, np.rad2deg(yaw_rate_truth - yaw_rate))
    axs[0,1].set_ylabel('Yaw Rate Error [deg/s]')
    axs[0,1].set_xlabel('Time [s]')
    axs[1,0].plot(t, np.rad2deg(yaw_truth))
    axs[1,0].plot(t, np.rad2deg(yaw))
    axs[1,0].set_ylabel('Yaw [deg]')
    axs[1,0].set_xlabel('Time [s]')
    axs[1,1].plot(t, np.rad2deg(yaw_truth - yaw))
    axs[1,1].set_ylabel('Yaw Error [deg]')
    axs[1,1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()
    
    # hitch rate and hitch
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Hitch Rate and Hitch')
    axs[0,0].plot(t, np.rad2deg(hitch_rate_truth))
    axs[0,0].plot(t, np.rad2deg(hitch_rate))
    axs[0,0].set_ylabel('Hitch Rate [deg/s]')
    axs[0,0].set_xlabel('Time [s]')
    axs[0,1].plot(t, np.rad2deg(hitch_rate_truth - hitch_rate))
    axs[0,1].set_ylabel('Hitch Rate Error [deg/s]')
    axs[0,1].set_xlabel('Time [s]')
    axs[1,0].plot(t, np.rad2deg(hitch_truth))
    axs[1,0].plot(t, np.rad2deg(hitch))
    axs[1,0].set_ylabel('Hitch [deg]')
    axs[1,0].set_xlabel('Time [s]')
    axs[1,1].plot(t, np.rad2deg(hitch_truth - hitch))
    axs[1,1].set_ylabel('Hitch Error [deg]')
    axs[1,1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()
    
    # yaw rate bias
    plt.plot(t, bias_yr)
    plt.title('Yaw rate bias')
    plt.ylabel('rad/s')
    plt.xlabel('Time [s]')
    plt.show()
    
def compute_abs_pos_error(coords1, coords2):
    X1,Y1 = coords1
    X2,Y2 = coords2
    
    error = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)
    return error