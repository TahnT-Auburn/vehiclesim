import numpy as np
from vehiclesim.tractor_trailer import TractorTrailer

VEH_CONFIG = 'C:\\Users\\Tahn\\SoftDevel\\vehiclesim\\vehiclesim\\vehicle_configs\\5a_config.yaml'
# TODO: Should move veh config as a config to class. Will change for MC implementations.
class NavOdomStateModule():
    """
    Generates the 12-state navigation model using tractor-trailer dynamics and
    additional navigation states and odometry biases.
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
        (10) North position bias (m)
        (11) East position bias (m)
        (12) Yaw angle bias (rad)
    """
    def __init__(self, error_model, vehicle_config):
        """
        12-state navigation state module with aditional odom biases.

        Args:
            error_model (NDArray): State error model. Equivalent to process noise matrix, Q.
            vehicle_config (str): String path to vehicle configuration file.
        """
        self.tract_trail_model = TractorTrailer(VEH_CONFIG)
        self.error_model = error_model
        
    def generate_state_model(self, steer_ang, x, dt):
        """
        Generates state model for the 12-state odom navigation model using tractor-trailer dynamics,
        additional navigation states, and odom biases. 

        Args:
            steer_ang (float): Tractor-trailer steer angle at road.
            x (array-like): Current state estimates.
            dt (float): Delta time.
        
        Returns:
            PHI (NDArray): State transition matrix.
            G (NDArray): State input matrix.
            Q (NDArray): State process noise matrix. 
        """
        vx = x[2, 0]
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
        
        bias_prop_matrix = np.eye(4)
        
        PHI = np.block([[pos_matrix, np.zeros((3,8))],
                [np.zeros((5,3)), A, bias_relation_matrix, np.zeros((5,3))],
                [np.zeros((4,8)), bias_prop_matrix]])
        
        G = np.vstack([np.zeros((3,1)), B, np.zeros((4,1))])
        
        # process noise
        Q = self.error_model
        
        # enforce ndarray on PHI and G since A, B from latmodel method is np.matrix
        PHI = np.array(PHI)
        G = np.array(G)
         
        return PHI, G, Q