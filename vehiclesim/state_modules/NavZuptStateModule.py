import numpy as np
from numpy.typing import NDArray

class NavZuptStateModule():
    """
    Generates a zero-velocity update (ZUPT) for the standard 9-state
    (see NavFullStateModule for detailed list) navigation model.  
    """
    def __init__(self, error_model: NDArray):
        """
        Zupt 9-state navigation state module.

        Args:
            error_model (NDArray): State error model. Equivalent to process noise matrix, Q. 
        """
        self.error_model = error_model
        
    def generate_state_model(self,):
        """
        Generates zupt model for the standard 9-state navigation model.

        Returns:
            Phi (NDArray): State transition matrix.
            G (NDArray): State input matrix.
            Q (NDArray): State process noise matrix. 
        """
        PHI = np.diag([1,1,0,0,0,1,0,1,1])
        F = PHI
        G = np.zeros((9,1))
        Q = self.error_model
        
        return PHI, F, G, Q 