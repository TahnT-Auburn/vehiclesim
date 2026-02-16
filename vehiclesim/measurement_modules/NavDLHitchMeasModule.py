import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn

class NavDLHitchMeasModule():
    """
    Generates a measurement model for hitch angle measurements from a NN that
    map to the standard 9-state tractor-trialer navigation model.
    """
    def __init__(
        self,
        network_model:nn.Module,
        hitch_init:float,
        error_model:NDArray,
    ):
        """
        Args:
            network_model (nn.Module): The nueral network model.
            error_model (NDArray): Measurement error model. Equivalent to measurement noise matrix, R.
        """
        self.network_model = network_model
        self.hitch_prev = hitch_init
        self.error_model = error_model

    def generate_meas_model(self, x:NDArray, network_inputs:torch.Tensor):
        """Generates the measurement model for hitch angle measurements computed from a NN.
        
        Args:
            x (NDArray): The currect state vector.
            network_inputs (torch.Tensor): Image tensor input to the network.
    Returns:
            z (NDArray): Measurements vector.
            H (NDArray): Measurement observation matrix.
            h_x (NDArray | float): Predicted measurement(s), h(x).
            R (NDArray): Measurement noise matrix.
        """
        hitch_meas = self._generate_measurement_from_network(network_inputs)
        # hitch_rate
        hitch_rate_meas = (hitch_meas - self.hitch_prev) / 0.1
        self.hitch_prev = hitch_meas
        z = np.array([
            [hitch_meas],
            [hitch_rate_meas]
        ])
        H = np.array([
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0]
        ])
        h_x = H @ x
        R = self.error_model
        
        return z, H, h_x, R


    def _generate_measurement_from_network(self, network_inputs:torch.Tensor):
        """Utility function to generate a hitch prediction from the given model
        
        Args:
            network_inputs (torch.Tensor): Image tensor input to the network.
        Return:
            hitch_est (float): Hitch estimate 
        """
        with torch.no_grad():
            self.network_model.eval()
            hitch_est = self.network_model(network_inputs)
            hitch_est = hitch_est.squeeze().cpu().numpy().item()

        return hitch_est