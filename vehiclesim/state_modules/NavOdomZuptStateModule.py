import numpy as np
from numpy.typing import NDArray

class NavOdomZuptStateModule():
    """
    Generates a zero-velocity update (ZUPT) for the 12-state odom
    (see NavOdomStateModule for detailed list) navigation model.  
    """
    def __init__(self, error_model: NDArray):
        """
        Zupt 12-state odom navigation state module.

        Args:
            error_model (NDArray): State error model. Equivalent to process noise matrix, Q. 
        """
        self.error_model = error_model
        
    def generate_state_model(self,):
        """
        Generates zupt model for the 12-state odom navigation model.

        Returns:
            Phi (NDArray): State transition matrix.
            G (NDArray): State input matrix.
            Q (NDArray): State process noise matrix. 
        """
        PHI = np.diag([1,1,0,0,0,1,0,1,1,1,1,1])
        G = np.zeros((12,1))
        Q = self.error_model
        
        return PHI, G, Q 