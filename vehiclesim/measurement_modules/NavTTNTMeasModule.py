import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

from trailer_pose_network.models.spacetime.async_st_ca_rn_trailer import AsyncSpaceTimeCrossAttentionResNet

#%%

class NavTTNTMeasModule():
    """
    Generates a measurement model for the standard 9-state navigation model.
    This measurement module is designed for direct nav outputs (N,E,yaw) and trailer
    hitch angle from the delta translations, delta rotation, and hitch angle predictions
    from the neural network.
    """
    def __init__(self, network_model:nn.Module, init_states:list, error_model:NDArray):
        """Nav NN measurement module for 9-state navigation model.

        Args:
            network_model (nn.Module): The nueral network model.
            init_states (list): List of inertial states [N,E,yaw] to propagate with deltas.
            error_model (NDArray): Measurement error model. Equivalent to measurement noise matrix, R.
        """
        self.network_model = network_model
        self.N_prev = init_states[0]
        self.E_prev = init_states[1]
        self.yaw_prev = init_states[2]
        self.error_model = error_model
        
    def generate_meas_model(self, x:NDArray, network_inputs:list[torch.Tensor]):
        """Generates the measurement model for the direct nav output from the neural network.

        Args:
            x (NDArray): The currect state vector.
            network_inputs (list): A list of tensor inputs [images, inerts] to the network.
        Returns:
            z (NDArray): Measurements vector.
            H (NDArray): Measurement observation matrix.
            h_x (NDArray | float): Predicted measurement(s), h(x).
            R (NDArray): Measurement noise matrix.
        """
        north_meas, east_meas, yaw_meas, hitch_meas = self._generate_measurement_from_network(network_inputs)
        z = np.array([
            [north_meas],
            [east_meas],
            [yaw_meas],
            [hitch_meas],
        ])
        H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0]
        ])
        h_x = H @ x
        R = self.error_model
        
        return z, H, h_x, R
    
    def _generate_measurement_from_network(self, network_inputs:list[torch.Tensor]):
        """Utility function to create a pseudo position and yaw estimate from odometry predictions
        from the neural network.
        
        Args:
            network_inputs (list): A list of tensor inputs [images, inerts] to the network.
        """
        # predict from model
        with torch.no_grad():
            self.network_model.eval()
            trans_est, rot_est, hitch_est = self.network_model(network_inputs)
        dx = trans_est.squeeze().cpu().numpy()[0]
        dy = trans_est.squeeze().cpu().numpy()[1]
        dyaw = rot_est.squeeze().cpu().numpy()
        # translate to tangent frame
        pose_prev = (self.E_prev, self.N_prev, self.yaw_prev)
        east_meas, north_meas = self._body_to_tangent_frame_translation(pose_prev, dx, dy)
        yaw_meas = self.yaw_prev + dyaw
        hitch_meas = hitch_est.squeeze().cpu().numpy()
        
        # update previous states
        self.N_prev = north_meas
        self.E_prev = east_meas
        self.yaw_prev = yaw_meas
        
        return north_meas, east_meas, yaw_meas, hitch_meas
    
    def _body_to_tangent_frame_translation(self, pose1, dx_body, dy_body):
        """
        Convert from body frame translation to tangent plane displacement
        
        Args:
            pose1: (X1, Y1, yaw1) - starting pose
            dx_body, dy_body: translation in body frame
            
        Returns:
            (dx_world, dy_world) - translation in world/tangent frame
        """
        X1, Y1, yaw1 = pose1
        
        # Rotate from body frame to world frame
        cos_yaw = np.cos(yaw1)  # Note: positive yaw1
        sin_yaw = np.sin(yaw1)  # Note: positive yaw1
        
        dx_world = cos_yaw * dx_body - sin_yaw * dy_body
        dy_world = sin_yaw * dx_body + cos_yaw * dy_body
        
        X2 = X1 + dx_world
        Y2 = Y1 + dy_world
        
        return X2, Y2