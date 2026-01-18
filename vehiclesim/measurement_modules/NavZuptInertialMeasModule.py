import numpy as np
from numpy.typing import NDArray

class NavZuptInertialMeasModule():
    """
    Generates a zero velocity update (ZUPT) measurement model for the standard 9-state navigation model defined
    in NavFullStateModule. This measurement module is designed to zero out velocity measurements
    in zero-velocity updates.
    """
    def __init__(self, error_model:NDArray):
        """
        ZUPT measurement module for 9-state navigation model.

        Args:
            error_model (NDArray): Measurement error model. Equivalent to measurement noise matrix, R.
        """
        self.error_model = error_model
        
    def generate_meas_model(self, x:NDArray, yaw_rate:float):
        """
        Generates the measurement model for the longitudal vel and yaw rate gyro inertial measurements.
        
        Args:
            x (NDArray): The current state vector.
            yaw_rate (float): Yaw rate measurement from gyro
        Returns:
            z (NDArray): Measurements vector.
            H (NDArray): Measurement observation matrix.
            h_x (NDarray | float): Predicted measurement, h(x).
            R (NDArray): Measurement noise matrix.
        """
        z = np.array([
            [0.0],
            [yaw_rate]
        ])
        H = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1]
        ])
        h_x = H @ x
        R = self.error_model
        
        return z, H, h_x, R
        