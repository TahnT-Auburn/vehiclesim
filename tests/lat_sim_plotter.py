'''
plotter function for tract_trail_lat_sim.py
'''
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def plot_states(t,truth_states, ol_states, cl_states, nav_states=None, t_factor=1):

    #parse inputs
    vy_truth = truth_states[0]
    yaw_rate_truth = truth_states[1]
    yaw_truth = truth_states[2]
    hitch_rate_truth = truth_states[3]
    hitch_truth = truth_states[4]

    vy_ol = ol_states[0]
    yaw_rate_ol = ol_states[1]
    yaw_ol = ol_states[2]
    hitch_rate_ol = ol_states[3]
    hitch_ol = ol_states[4]

    vy_cl = cl_states[0]
    yaw_rate_cl = cl_states[1]
    yaw_cl = cl_states[2]
    hitch_rate_cl = cl_states[3]
    hitch_cl = cl_states[4]
    
    vy_nav = nav_states[0]
    yaw_rate_nav = nav_states[1]
    yaw_nav = nav_states[2]
    hitch_rate_nav = nav_states[3]
    hitch_nav = nav_states[4]

    #### lateral velocity ####
    ax1 = plt.subplot(211)
    ax1.plot(t/t_factor,vy_truth, c='cyan', label='Truth')
    ax1.plot(t/t_factor,vy_ol, c='red', label='Model')
    ax1.plot(t/t_factor,vy_cl, c='k', label='KF')
    ax1.plot(t/t_factor,vy_nav, c='m', label='Nav KF')
    ax1.legend()
    # ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Lateral Velocity [m/s]')
    ax1.tick_params(axis='x',labelsize=13)
    ax1.tick_params(axis='y',labelsize=13)

    ax2 = plt.subplot(212)
    ax2.plot(t/t_factor,vy_ol - vy_truth, c='red', label='Model')
    ax2.plot(t/t_factor,vy_cl - vy_truth, c='k', label='KF')
    ax2.plot(t/t_factor,vy_nav - vy_truth, c='m', label='Nav KF')
    ax2.legend()
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Error [m/s]')
    ax2.tick_params(axis='x',labelsize=13)
    ax2.tick_params(axis='y',labelsize=13)
    plt.tight_layout()
    plt.show()

    #### yaw ####
    ax1 = plt.subplot(211)
    ax1.plot(t/t_factor,yaw_truth, c='cyan', label='Truth')
    ax1.plot(t/t_factor,np.rad2deg(yaw_ol), c='red', label='Model')
    ax1.plot(t/t_factor,np.rad2deg(yaw_cl), c='k', label='KF')
    ax1.plot(t/t_factor,np.rad2deg(yaw_nav), c='m', label='Nav KF')
    ax1.legend()
    # ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Yaw [deg]')
    ax1.tick_params(axis='x',labelsize=13)
    ax1.tick_params(axis='y',labelsize=13)

    ax2 = plt.subplot(212)
    ax2.plot(t/t_factor,np.rad2deg(yaw_ol) - yaw_truth, c='red', label='Model')
    ax2.plot(t/t_factor,np.rad2deg(yaw_cl) - yaw_truth, c='k', label='KF')
    ax2.plot(t/t_factor,np.rad2deg(yaw_nav) - yaw_truth, c='m', label='Nav KF')
    ax2.legend()
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Error [deg]')
    ax2.tick_params(axis='x',labelsize=13)
    ax2.tick_params(axis='y',labelsize=13)
    plt.tight_layout()
    plt.show()

    #### yaw rate ####
    ax1 = plt.subplot(211)
    ax1.plot(t/t_factor,yaw_rate_truth, c='cyan', label='Truth')
    ax1.plot(t/t_factor,np.rad2deg(yaw_rate_ol), c='red', label='Model')
    ax1.plot(t/t_factor,np.rad2deg(yaw_rate_cl), c='k', label='KF')
    ax1.plot(t/t_factor,np.rad2deg(yaw_rate_nav), c='m', label='Nav KF')
    ax1.legend()
    # ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Yaw Rate [deg/s]')
    ax1.tick_params(axis='x',labelsize=13)
    ax1.tick_params(axis='y',labelsize=13)

    ax2 = plt.subplot(212)
    ax2.plot(t/t_factor, np.rad2deg(yaw_rate_ol) - yaw_rate_truth, c='red', label='Model')
    ax2.plot(t/t_factor, np.rad2deg(yaw_rate_cl) - yaw_rate_truth, c='k', label='KF')
    ax2.plot(t/t_factor, np.rad2deg(yaw_rate_nav) - yaw_rate_truth, c='m', label='Nav KF')
    ax2.legend()
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Error [deg/s]')
    ax2.tick_params(axis='x',labelsize=13)
    ax2.tick_params(axis='y',labelsize=13)
    plt.tight_layout()
    plt.show()

    #### hitch ####
    ax1 = plt.subplot(211)
    ax1.plot(t/t_factor,hitch_truth, c='cyan', label='Truth')
    ax1.plot(t/t_factor,np.rad2deg(hitch_ol), c='red', label='Model')
    ax1.plot(t/t_factor,np.rad2deg(hitch_cl), c='k', label='KF')
    ax1.plot(t/t_factor,np.rad2deg(hitch_nav), c='m', label='Nav KF')
    ax1.legend()
    # ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Hitch Angle [deg]')
    ax1.tick_params(axis='x',labelsize=13)
    ax1.tick_params(axis='y',labelsize=13)

    ax2 = plt.subplot(212)
    ax2.plot(t/t_factor,np.rad2deg(hitch_ol) - hitch_truth, c='red', label='Model')
    ax2.plot(t/t_factor,np.rad2deg(hitch_cl) - hitch_truth, c='k', label='KF')
    ax2.plot(t/t_factor,np.rad2deg(hitch_nav) - hitch_truth, c='m', label='Nav KF')
    ax2.legend()
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Error [deg]')
    ax2.tick_params(axis='x',labelsize=13)
    ax2.tick_params(axis='y',labelsize=13)
    plt.tight_layout()
    plt.show()

    #### hitch rate #### 
    ax1 = plt.subplot(211)
    ax1.plot(t/t_factor,hitch_rate_truth, c='cyan', label='Truth')
    ax1.plot(t/t_factor,np.rad2deg(hitch_rate_ol), c='red', label='Model')
    ax1.plot(t/t_factor,np.rad2deg(hitch_rate_cl), c='k', label='KF')
    ax1.plot(t/t_factor,np.rad2deg(hitch_rate_nav), c='m', label='Nav KF')
    ax1.legend()
    # ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Hitch Rate [deg/s]')
    ax1.tick_params(axis='x',labelsize=13)
    ax1.tick_params(axis='y',labelsize=13)

    ax2 = plt.subplot(212)
    ax2.plot(t/t_factor,np.rad2deg(hitch_rate_ol) - hitch_rate_truth , c='red', label='Model')
    ax2.plot(t/t_factor,np.rad2deg(hitch_rate_cl) - hitch_rate_truth, c='k', label='KF')
    ax2.plot(t/t_factor,np.rad2deg(hitch_rate_nav) - hitch_rate_truth, c='m', label='Nav KF')
    ax2.legend()
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Error [deg/s]')
    ax2.tick_params(axis='x',labelsize=13)
    ax2.tick_params(axis='y',labelsize=13)
    plt.tight_layout
    plt.show()

def plot_pos(t, truth_pos, model_pos, kf_pos, mod_pos_error, kf_pos_error):

    #### position propagation ####
    fig, ax1 = plt.subplots()
    ax1.plot(truth_pos[0], truth_pos[1], c='cyan', label='TruckSim')
    ax1.plot(model_pos[0], model_pos[1], c='red', label='Model')
    ax1.plot(kf_pos[0], kf_pos[1], c='k', label='KF')
    ax1.legend()
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.tick_params(axis='x',labelsize=13)
    ax1.tick_params(axis='y',labelsize=13)
    plt.tight_layout
    plt.show()

    fig, ax2 = plt.subplots()
    ax2.plot(t, mod_pos_error, c='red', label='Model')
    ax2.plot(t, kf_pos_error, c='k', label='KF')
    # ax2.legend()
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Position Error [m]')
    ax2.tick_params(axis='x',labelsize=13)
    ax2.tick_params(axis='y',labelsize=13)
    plt.tight_layout
    plt.show()