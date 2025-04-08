import numpy as np
from box import Box

def imu_sim(grade,accel,gyro,L):
    """
    #################### IMU Sim ####################

        Author: 
            Tahn Thawainin, AU GAVLAB

        Description:
            Simple IMU "simulator". Takes pre-generated 3DOF linear 
            acceleration and 3DOF angular velocity signals and adds guassian noise 
            and bias corresponding to specified IMU grade (source: Groves Table 4.1)

        Parameters:
            * grade: Desired IMU grade
            Options include (low-high):
                consumer      (1)
                tactical      (2)
                intermediate  (3)
                aviation      (4)
                marine        (5)

            * accel: body frame acceleration (not normalized)

                Ax: Acceleration in x-axis [m/s^2]
                Ay: Acceleration in y-axis [m/s^2]
                Az: Acceleration in z-axis [m/s^2]

            * gyro: body frame angular rate

                AVx: Angular velocity in x-axis [rad/s]
                AVy: Angular velocity in y-axis [rad/s]
                AVz: Angular velocity in z-axis [rad/s]

            * L: Length of signals (All signals must have the same length)

        Returns:
            * imu: imu data struct

    ################################################################
    """
    # declare 3dof signals
    Ax = accel[0]
    Ay = accel[1]
    Az = accel[2]
    AVx = gyro[0]
    AVy = gyro[1]
    AVz = gyro[2]

    #----- Consumer/Automotive -----#
    if (grade == 1):

        # Noise STD [x, y, z]
        sigma_accel = [0.15, 0.15, 0.15]
        sigma_gyro = [0.005, 0.005, 0.005]

        # Biases [x, y, z]
        bias_accel = [0.10, 0.10, 0.10] #[m/s^2]
        bias_gyro = [5e-4, 5e-4, 5e-4]  #[rad/s]

    #------ Tactical -----%
    elif (grade == 2):
        
        # Noise STD [x, y, z]
        sigma_accel = [0.075, 0.075, 0.075]
        sigma_gyro = [0.0025, 0.0025, 0.0025]

        # Biases
        bias_accel = [0.05, 0.05, 0.05] #[m/s^2]
        bias_gyro = [5e-5, 5e-5, 5e-5]  #[rad/s]

    #------ Intermediate -----%
    elif (grade == 3):
        
        # Noise STD [x, y, z]
        sigma_accel = [0.0375, 0.0375, 0.0375]
        sigma_gyro = [0.00125, 0.00125, 0.00125]

        # Biases
        bias_accel = [0.005, 0.005, 0.005] #[m/s^2]
        bias_gyro = [5e-7, 5e-7, 5e-7]  #[rad/s]

    #------ Aviation -----%
    elif (grade == 4):
        
        # Noise STD [x, y, z]
        sigma_accel = [0.01875, 0.01875, 0.01875]
        
        sigma_gyro = [6.25e-4, 6.25e-4, 6.25e-4]

        # Biases
        bias_accel = [5e-4, 5e-4, 5e-4] #[m/s^2]
        bias_gyro = [5e-8, 5e-8, 5e-8]  #[rad/s]

    #------ Marine -----%
    elif (grade == 5):
        
        # Noise STD [x, y, z]
        sigma_accel = [0.009375, 0.009375, 0.009375]
        sigma_gyro = [3.125e-4, 3.125e-4, 3.125e-4]

        # Biases
        bias_accel = [1e-4, 1e-4, 1e-4] #[m/s^2]
        bias_gyro = [5e-9, 5e-9, 5e-9]  #[rad/s]
        # bias_accel = [0,0,0]
        # bias_gyro = [0,0,0]

    #Add gravity to Az signal
    Az = Az + (-9.81)

    #Generate noise vector
    n_Ax  = sigma_accel[0]*np.random.randn(L)
    n_Ay  = sigma_accel[1]*np.random.randn(L)
    n_Az  = sigma_accel[2]*np.random.randn(L)

    n_AVx  = sigma_gyro[0]*np.random.randn(L)
    n_AVy  = sigma_gyro[1]*np.random.randn(L)
    n_AVz  = sigma_gyro[2]*np.random.randn(L)

    #Generate IMU measurements
    Ax = Ax + n_Ax + bias_accel[0]
    Ay = Ay + n_Ay + bias_accel[1]
    Az = Az + n_Az + bias_accel[2]

    AVx = AVx + n_AVx + bias_gyro[0]
    AVy = AVy + n_AVy + bias_gyro[1]
    AVz = AVz + n_AVz + bias_gyro[2]

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

    imu.accel = np.array([[Ax], [Ay], [Az]]).reshape(3,L)
    imu.gyro = np.array([[AVx], [AVy], [AVz]]).reshape(3,L)
    
    imu.covar.accel = [sigma_accel[0]**2], [sigma_accel[1]**2], [sigma_accel[2]**2]
    imu.covar.gyro = [sigma_gyro[0]**2], [sigma_gyro[1]**2], [sigma_gyro[2]**2]

    imu.bias.accel = [bias_accel[0], bias_accel[1], bias_accel[2]]
    imu.bias.gyro = [bias_gyro[0], bias_gyro[1], bias_gyro[2]]
    
    return imu
