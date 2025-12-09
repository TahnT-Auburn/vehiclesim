import numpy as np
from numpy.typing import NDArray

class NavOdomMeasModule():
    """
    Generates a measurement model that uses planar body frame odometry measurements.
    Specifically, it expects to make corrections given a delta x, delta y translations
    and delta yaw rotations in the body frame. The measurements are mapped to the standard
    9-state navigation model defined in NavFullStateModule.
    """
    def __init__(
        self,
        error_model:NDArray,
        N_init:float,
        E_init:float,
        yaw_init:float,
    ):
        """
        Body frame odometry measurement module

        Args:
            error_model (NDArray): Measurement error model. Equivalent to mesaurement noise matrix, R.
            N_init (float): Initial North position in meters.
            E_init (float): Initial East position in meters.
            yaw_init (float): Initial yaw position in meters.
        """
        self.error_model = error_model
        self.pose_prev = (E_init, N_init, yaw_init)
        
    def _body_to_tangent_frame_translation(self, pose1, dx_body, dy_body, dyaw):
        """
        Convert from body frame translation to tangent plane displacement
        
        Args:
            pose1: (X1, Y1, yaw1) - starting pose
            dx_body, dy_body: translation in body frame
            dyaw: delta yaw rotation.
        Returns:
            X2, Y2, yaw2: Resultant X, Y position and yaw after translation and rotation.
        """
        X1, Y1, yaw1 = pose1
        
        # Rotate from body frame to world frame
        cos_yaw = np.cos(yaw1)  # Note: positive yaw1
        sin_yaw = np.sin(yaw1)  # Note: positive yaw1
        
        dx_world = cos_yaw * dx_body - sin_yaw * dy_body
        dy_world = sin_yaw * dx_body + cos_yaw * dy_body
        
        X2 = X1 + dx_world
        Y2 = Y1 + dy_world
        yaw2 = yaw1 + dyaw
        
        return X2, Y2, yaw2

    def generate_meas_model(
        self,
        x: NDArray,
        x_prev: NDArray,
        dx_body:float,
        dy_body:float,
        dyaw:float,
    ):
        """
        Generates the measurement model for planar body frame odometry measurements.

        Args:
            x (NDArray): Most current state estimtes.
            dx_body (float): Delta x translation.
            dy_body (float): Delta y translation.
            dyaw (float): Delta yaw rotation in radians.
        Returns:
            z (NDArray): Measurements vector.
            H (NDArray): Measurement observation matrix.
            R (NDArray): Measurement noise matrix.
        """
        N_prev = x_prev[0,0]
        E_prev = x_prev[1,0]
        yaw_prev = x_prev[5,0]
        yaw_current = x[5,0]
        bias_x = x[9,0]
        bias_y = x[10,0]
        bias_yaw = x[11,0]
        
        # correct deltas with estimated biases
        dx_body_corr = dx_body - bias_x
        dy_body_corr = dy_body - bias_y
        dyaw_corr = dyaw - bias_yaw
        
        import math
        if math.isnan(bias_x) or math.isnan(bias_y) or math.isnan(bias_yaw):
            print('NaN detected in bias estimates')
            print(f'Bias dx: {bias_x}')
            print(f'Bias dy: {bias_y}')
            print(f'Bias dyaw: {bias_yaw}')
            
        # self.pose_prev = (E_prev, N_prev, yaw_prev)
        E, N, yaw = self._body_to_tangent_frame_translation(self.pose_prev, dx_body_corr, dy_body_corr, dyaw_corr)
        self.pose_prev = (E,N,yaw)
        z = np.array([
            [N],[E],[yaw]
        ])
        # print(yaw_state)
        H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, -np.cos(yaw), np.sin(yaw), 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, -np.sin(yaw), -np.cos(yaw), 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1]
        ])
        # H = np.array([
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, np.cos(yaw_state), -np.sin(yaw_state), 0],
        #     [0, 1, 0, 0, 0, 0, 0, 0, 0, np.sin(yaw_state), np.cos(yaw_state), 0],
        #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        # ])
        R = self.error_model
        
        return z, H, R