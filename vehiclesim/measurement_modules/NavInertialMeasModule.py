import numpy as np
from numpy.typing import NDArray

class NavInertialMeasModule():
    """
    Generates a measurement model for the standard 9-state navigation model defined
    in NavFullStateModule. This measurement module is designed for yaw rate measurements.
    """
    def __init__(self, error_model:NDArray):
        """
        Inertial measurement module for 9-state navigation model.

        Args:
            error_model (NDArray): Measurement error model. Equivalent to measurement noise matrix, R.
        """
        self.error_model = error_model
        
    def generate_meas_model(self, x:NDArray, yaw_rate:float):
        """
        Generates the measurement model for yaw rate gyro inertial measurements.
        
        Args:
            x (NDArray): Current state vector.
            vx (float): Longitudinal velocity measurement as reported from sensor.
            yaw_rate (float): Yaw rate measurement as reported from sensor.
        Returns:
            z (NDArray): Measurements vector.
            H (NDArray): Measurement observation matrix.
            h_x (NDArray | float): Predicted measurement(s), h(x).
            R (NDArray): Measurement noise matrix.
        """
        z = np.array([
            [yaw_rate]
        ])
        H = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0, 1]
        ])
        h_x = H @ x
        R = self.error_model
        
        return z, H, h_x, R
        