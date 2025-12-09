import numpy as np

def body_disp_meas_sim(N,E,yaw,sigmas=np.array([0,0,0]),biases=np.array([0,0,0])):
    """
    Generates simulated planar body frame displacement measurements which includes
    a delta x, delta y translations and delta yaw rotation in the body frame.

    Args:
        N (array-like): North positions.
        E (array-like): East positions.
        yaw (array-like): Yaw angles in radians.
        sigmas (array-like): 3 element sigmas to apply white noise to odom measurements.
        biases (array-like): 3 element biases to apply to odom measurements.
    Returns:
        dx_body (array-like): Delta x translation.
        dy_body (array-like): Delta y translation.
        dyaw (array-like): Delta yaw rotation in radians.
    """
    L = len(N)
    dx_body = []
    dy_body = []
    dyaw = []
    for i in range(1,L):
        pose_prev = (E[i-1], N[i-1], yaw[i-1])
        pose_current = (E[i], N[i], yaw[i])
        dx_body_, dy_body_, dyaw_ = tangent_to_body_frame_translation(pose_prev, pose_current)
        dx_body.append(dx_body_)
        dy_body.append(dy_body_)
        dyaw.append(dyaw_)
    # apply white noise
    L_disp = len(dx_body)
    dx_body = np.array(dx_body) + sigmas[0]*np.random.randn(L_disp) + biases[0]
    dy_body = np.array(dy_body) + sigmas[1]*np.random.randn(L_disp) + biases[1]
    dyaw = np.array(dyaw) + sigmas[2]*np.random.randn(L_disp) + biases[2]
    
    return dx_body, dy_body, dyaw


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