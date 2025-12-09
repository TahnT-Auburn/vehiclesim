import numpy as np
from numpy.typing import NDArray

class NavInertialOdomMeasModule():
    """
    Generates a measurement model for the 12-state odom navigation model defined
    in NavOdomStateModule. This measurement module is designed around a longitudinal
    velocity and a yaw rate measurements.
    """
    def __init__(self, error_model:NDArray):
        """
        Inertial measurement module for 9-state navigation model.

        Args:
            error_model (NDArray): Measurement error model. Equivalent to measurement noise matrix, R.
        """
        self.error_model = error_model
        
    def generate_meas_model(self, vx, yaw_rate):
        """
        Generates the measurement model for the longitudal vel and yaw rate gyro inertial measurements.
        
        Args:
            vx (float): Longitudinal velocity measurement as reported from sensor.
            yaw_rate (float): Yaw rate measurement as reported from sensor.
        Returns:
            z (NDArray): Measurements vector.
            H (NDArray): Measurement observation matrix.
            R (NDArray): Measurement noise matrix.
        """
        z = np.array([
            [vx],
            [yaw_rate]
        ])
        H = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        ])
        R = self.error_model
        
        return z, H, R
        