'''
#################### Function to generate full navigation matrices ####################

    Author: 
        Tahn Thawainin, AU GAVLAB
        email: pzt0029@auburn.edu
        github: https://github.com/TahnT-Auburn
    
    Description: Function to generate full navigation matrices for GPS/GNSS denied tractor trailer nav
#####################################################################################################
'''
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def genNavMatrices(A_veh, B_veh, vx, yaw, dt, use_cams:bool=False):
    '''
    Parameters:
        A_veh : np.ndarray (5x5).
            Discrete time state transition matrix.
        B_veh : np.ndarray (5x1).
            Discrete 
        vx : float
            Longitudinal velocity in m/s (Q: state or measurement?).
        yaw : float
            Tractor yaw angle in rad.
        dt : float
            Delta time (1/sampling rate).
        use_cams (bool, optional):
            Returns modified H matrix for adding hitch measurements from cameras
    
    Returns:
        A : np.matrix (nxn)
            Full naviation state transition matrix
        B : np.matrix (nx1)
            Full navigation input matrix
        H : np.matrix (mxn)
            Full navigation observation matrix
    '''
    # generate intermittent
    pos_matrix = np.array([[1, np.cos(yaw)*dt, 0, -np.sin(yaw)*dt],
                           [0, 1, 0, 0],
                           [0, np.sin(yaw)*dt, 1, np.cos(yaw)*dt]])
    
    # bias_relation_matrix = np.array([[dt, 0],
    #                                  [0, 1],
    #                                  [0, dt],
    #                                  [0, 0],
    #                                  [0, 0]])

    # bias_relation_matrix = np.array([[dt, 0],
    #                                  [0, 1],
    #                                  [0, 0],
    #                                  [0, 0],
    #                                  [0, 0]])

    bias_relation_matrix = np.array([[0],
                                     [0],
                                     [dt],
                                     [0],
                                     [0]])
    
    # bias_matrix = np.array([[1, 0],
    #                         [0, 1]])

    bias_matrix = np.array([[1]])

    # A = np.block([[pos_matrix, np.zeros((3,6))],
    #               [np.zeros((5,3)), A_veh, bias_relation_matrix],
    #               [np.zeros((2,8)), bias_matrix]])
    
    A = np.block([[pos_matrix, np.zeros((3,5))],
                  [np.zeros((5,3)), A_veh, bias_relation_matrix],
                  [np.zeros((1,8)), bias_matrix]])
    
    # B = np.vstack([np.zeros((3,1)), B_veh, np.zeros((2,1))])

    B = np.vstack([np.zeros((3,1)), B_veh, np.zeros((1,1))])

    if use_cams:
        H = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, vx, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    else:
        # H = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, vx, 0, 0, 0, 1, 0],
        #               [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])

        H = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 1]])

        # H = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
       
    A = np.asmatrix(A)
    B = np.asmatrix(B)
    H = np.asmatrix(H)

    return A, B, H