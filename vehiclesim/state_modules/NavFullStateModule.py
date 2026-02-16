import numpy as np
from vehiclesim.tractor_trailer import TractorTrailer

class NavFullStateModule():
    """
    Generates the standard 9-state navigation model using tractor-trailer dynamics and
    additional navigation states.
    Assumes the following states in order:
        (1) North position (m)
        (2) East position (m)
        (3) Tractor longitudinal velocity (m/s)
        (4) Tractor lateral velocity (m/s)
        (5) Tractor yaw rate (rad/s)
        (6) Tractor yaw angle wrt to local frame (rad)
        (7) Trailer hitch rate (rad/s)
        (8) Trailer hitch angle (rad)
        (9) Yaw rate bias (rad/s)
    """
    def __init__(self, error_model, vehicle_config):
        """
        Standard 9-state navigation state module.

        Args:
            error_model (NDArray): State error model. Equivalent to process noise matrix, Q.
            vehicle_config (str): String path to vehicle configuration file.
        """
        self.tract_trail_model = TractorTrailer(vehicle_config)
        self.error_model = error_model
        
    def generate_state_model(self, steer_ang, x, dt):
        """
        Generates state model for the standard 9-state navigation model using tractor-trailer dynamics
        and additional navigation states. 

        Args:
            steer_ang (float): Tractor-trailer steer angle at road.
            x (array-like): Current state estimates. Assumes [N, E, vx, vy, yaw rate, yaw, hitch rate, hitch, yaw rate bias]
            dt (float): Delta time.
        
        Returns:
            PHI (NDArray): State transition matrix.
            G (NDArray): State input matrix.
            Q (NDArray): State process noise matrix. 
        """
        vx = x[2, 0]
        vy = x[3, 0]
        yaw = x[5, 0]
        # generate tractor-trailer dynamics model
        _, sysd = self.tract_trail_model.latModel(steer_ang, vx, dt)
        A = sysd.A
        B = sysd.B
        
        # generate full navigation state model
        pos_matrix = np.array([[1, 0, np.sin(yaw)*dt, np.cos(yaw)*dt],
                               [0, 1, np.cos(yaw)*dt, -np.sin(yaw)*dt],
                               [0, 0, 1, 0]])
        
        bias_relation_matrix = np.array([[0],
                                        [0],
                                        [dt], # dt
                                        [0],
                                        [0]])
        
        bias_prop_matrix = np.array([[1]])
        
        PHI = np.block([[pos_matrix, np.zeros((3,5))],
                [np.zeros((5,3)), A, bias_relation_matrix],
                [np.zeros((1,8)), bias_prop_matrix]])
        
        G = np.vstack([np.zeros((3,1)), B, np.zeros((1,1))])
        
        # process noise
        Q = self.error_model
        
        # Generate jacobian for PHI
        F = np.eye(9)
    
        # North position derivatives (row 0)
        F[0, 0] = 1
        F[0, 2] = np.sin(yaw) * dt      # ∂N/∂vx
        F[0, 3] = np.cos(yaw) * dt      # ∂N/∂vy
        F[0, 5] = (vx*np.cos(yaw) - vy*np.sin(yaw)) * dt  # ∂N/∂yaw
        
        # East position derivatives (row 1)
        F[1, 1] = 1
        F[1, 2] = np.cos(yaw) * dt      # ∂E/∂vx
        F[1, 3] = -np.sin(yaw) * dt     # ∂E/∂vy
        F[1, 5] = (-vx*np.sin(yaw) - vy*np.cos(yaw)) * dt  # ∂E/∂yaw
        
        # Vehicle dynamics (rows 2-6, cols 2-6)
        F[3:8, 3:8] = A
        
        # dyaw_rate/dyaw_rate_bias
        # Yaw affected by yaw rate bias (row 4, col 8)
        F[5, 8] = dt  # ∂(yaw)/∂(yaw_rate_bias)

        # enforce ndarray on PHI and G since A, B from latmodel method is np.matrix
        PHI = np.array(PHI)
        G = np.array(G)
        
        return PHI, F, G, Q