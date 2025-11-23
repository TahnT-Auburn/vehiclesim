import numpy as np
import matplotlib.pyplot as plt

def body_frame_displacements_plotter(x, x_truth, interactive:bool=False):
    """
    Generates plots for the standard 9-state estimates vs truth and their errors.

    Args:
        x (list): List of state estimates.
        x_truth (list): List of state truths.
        t (list): Time.
        interactive (bool, optional). Switch between interactive or non-interactive plots. Deafult is False. 
    """
    if interactive:
        import matplotlib
        matplotlib.use('ipympl')
        
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
    
    def tangent_to_body_frame_translation(pose1, pose2):
        """
        Convert from tangent plane poses to body frame relative translation
        
        Args:
            pose1: (X1, Y1, yaw1) - starting pose
            pose2: (X2, Y2, yaw2) - ending pose
            
        Returns:
            (dx_body, dy_body) - translation in body frame of pose1
        """
        X1, Y1, yaw1 = pose1
        X2, Y2, yaw2 = pose2
        
        # World frame displacement
        dx_world = X2 - X1
        dy_world = Y2 - Y1
        
        # Rotate into body frame of pose1
        cos_yaw = np.cos(-yaw1)
        sin_yaw = np.sin(-yaw1)
        
        dx_body = cos_yaw * dx_world - sin_yaw * dy_world
        dy_body = sin_yaw * dx_world + cos_yaw * dy_world
        
        # Relative yaw change
        dyaw = yaw2 - yaw1
        
        # Normalize yaw to [-pi, pi]
        dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
        
        return dx_body, dy_body, dyaw
    
    # compute displacements
    dx_body_ = []
    dy_body_ = []
    dyaw_ = []
    dx_body_truth_ = []
    dy_body_truth_ = []
    dyaw_truth_ = []
    for i in range(1,len(N)):
        pose_prev = (E[i-1], N[i-1], yaw[i-1])
        pose_current = (E[i], N[i], yaw[i])
        dx_body, dy_body, dyaw = tangent_to_body_frame_translation(pose_current, pose_prev)
        dx_body_.append(dx_body)
        dy_body_.append(dy_body)
        dyaw_.append(dyaw)
        
        pose_prev_truth = (E_truth[i-1], N_truth[i-1], yaw_truth[i-1])
        pose_current_truth = (E_truth[i], N_truth[i], yaw_truth[i])
        dx_body_truth, dy_body_truth, dyaw_truth = tangent_to_body_frame_translation(pose_current_truth, pose_prev_truth)
        dx_body_truth_.append(dx_body_truth)
        dy_body_truth_.append(dy_body_truth)
        dyaw_truth_.append(dyaw_truth)  
        
    dx_body_ = np.array(dx_body_)
    dy_body_ = np.array(dy_body_)
    dyaw_ = np.array(dyaw_)
    dx_body_truth_ = np.array(dx_body_truth_)
    dy_body_truth_ = np.array(dy_body_truth_)
    dyaw_truth_ = np.array(dyaw_truth_)
    
    # displacements plot
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Body Frame Displacements')
    ax1.plot(dx_body_truth_)
    ax1.plot(dx_body_)
    ax1.legend(['Truth','Est'])
    ax1.set_ylabel('X Displacement [m]')
    ax2.plot(dy_body_truth_)
    ax2.plot(dy_body_)
    ax2.set_ylabel('Y Displacement [m]')
    ax3.plot(np.rad2deg(dyaw_truth_))
    ax3.plot(np.rad2deg(dyaw_))
    ax3.set_ylabel('Yaw Displacement [deg]')
    plt.tight_layout()
    plt.show()
    
    # displacements error plot
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Body Frame Displacement Errors')
    ax1.plot(dx_body_truth_ - dx_body_)
    ax1.set_ylabel('X Displacement Error [m]')
    ax2.plot(dy_body_truth_ - dy_body_)
    ax2.set_ylabel('Y Displacement Error [m]')
    ax3.plot(np.rad2deg(dyaw_truth_) - np.rad2deg(dyaw_))
    ax3.set_ylabel('Yaw Displacement Error [deg]')
    plt.tight_layout()
    plt.show()