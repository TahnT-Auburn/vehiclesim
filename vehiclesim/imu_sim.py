import numpy as np
from box import Box

def imu_sim(grade,Ax,Ay,Az,AVx,AVy,AVz,L):
    """
    #################### IMU Sim ####################

        Author: 
            Tahn Thawainin, AU GAVLAB

        Description:
            Simple IMU "simulator". Takes pre-generated 3DOF linear 
            acceleration and 3DOF angular velocity signals and adds guassian noise 
            and bias corresponding to specified IMU grade (source: Groves Table 4.1)

        Input(s):
            grade: Desired IMU grade
            Options include (low-high):
                * consumer      (1)
                * tactical      (2)
                * intermediate  (3)
                * aviation      (4)
                * marine        (5)

            Ax: Acceleration in x-axis [m/s^2]
            Ay: Acceleration in y-axis [m/s^2]
            Az: Acceleration in z-axis [m/s^2]

            AVx: Angular velocity in x-axis [rad/s]
            AVy: Angular velocity in y-axis [rad/s]
            AVz: Angular velocity in z-axis [rad/s]

            L: Length of signals (All signals must have the same length)

        Output(s):

            imu: imu data struct

    ################################################################
    """
    #----- Consumer/Automotive -----#
    if (grade == 1):

        # Noise STD
        sigma_Ax = 0.15
        sigma_Ay = 0.15
        sigma_Az = 0.15

        sigma_AVx = 0.005
        sigma_AVy = 0.005
        sigma_AVz = 0.005

        # Biases
        bias_Ax = 0.15
        bias_Ay = 0.15
        bias_Az = 0.15

        bias_AVx = 5e-4
        bias_AVy = 5e-4
        bias_AVz = 5e-4

    #------ Tactical -----%
    elif (grade == 2):
        
        # Noise STD
        sigma_Ax = 0.085
        sigma_Ay = 0.085
        sigma_Az = 0.085

        # Biases
        sigma_AVx = 0.0025
        sigma_AVy = 0.0025
        sigma_AVz = 0.0025

        # Biases
        bias_Ax = 0.085
        bias_Ay = 0.085
        bias_Az = 0.085

        bias_AVx = 5e-5
        bias_AVy = 5e-5
        bias_AVz = 5e-5

    #Add gravity to Az signal
    Az = Az + (-9.81)

    #Generate noise vector
    n_Ax  = sigma_Ax*np.random.randn(L,1)
    n_Ay  = sigma_Ay*np.random.randn(L,1)
    n_Az  = sigma_Az*np.random.randn(L,1)

    n_AVx  = sigma_AVx*np.random.randn(L,1)
    n_AVy  = sigma_AVy*np.random.randn(L,1)
    n_AVz  = sigma_AVz*np.random.randn(L,1)

    #Generate IMU measurements
    Ax = Ax + n_Ax + bias_Ax
    Ay = Ay + n_Ay + bias_Ay
    Az = Az + n_Az + bias_Az

    AVx = AVx + n_AVx + bias_AVx
    AVy = AVy + n_AVy + bias_AVy
    AVz = AVz + n_AVz + bias_AVz

    #Generate IMU data struct
    imu = Box({'grade': 'NaN',\
                'linaccel': 'NaN',\
                'angvel': 'NaN',\
                'covar': {'linaccel': 'NaN', 'angvel': 'NaN'},\
                'bias': {'linaccel': 'NaN', 'angvel': 'NaN'}})
    
    if (grade == 1):
        imu.grade = 'Consumer/Automotive'
    elif (grade == 2):
        imu.grade = 'Tactical'
    elif (grade == 3):
        imu.grade = 'Intermediate'
    elif (grade == 4):
        imu.grade = 'Aviation'
    elif (grade == 5):
        imu.grade = 'Marine'

    imu.linaccel = [Ax, Ay, Az]
    imu.angvel = [AVz, AVy, AVz]
    
    imu.covar.linaccel = [sigma_Ax^2, sigma_Ay^2, sigma_Az^2]
    imu.covar.angvel = [sigma_AVx^2, sigma_AVy^2, sigma_AVz^2]

    imu.bias.linaccel = [bias_Ax, bias_Ay, bias_Az]
    imu.bias.angvel = [bias_AVx, bias_AVy, bias_AVz]
    
    return imu
