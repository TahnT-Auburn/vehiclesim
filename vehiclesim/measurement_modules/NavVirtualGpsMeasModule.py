import numpy as np
from numpy.typing import NDArray

class NavVirtualGpsMeasModule():
    """Generates a virtial GPS measurement update for North, East, and yaw measurements"""
    
    def __init__(self, error_model:NDArray):
        self.error_model = error_model
        
    def generate_meas_model(self, x:NDArray, N:float, E:float, yaw:float):
        z = np.array([
            [N],
            [E],
            [yaw]
        ])
        H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])
        h_x = H @ x
        R = self.error_model
        
        return z, H, h_x, R