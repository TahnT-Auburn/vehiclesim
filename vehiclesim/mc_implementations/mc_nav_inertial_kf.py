#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io
import random

from vehiclesim.state_modules.NavFullStateModule import NavFullStateModule
from vehiclesim.state_modules.NavZuptStateModule import NavZuptStateModule
from vehiclesim.measurement_modules.NavInertialMeasModule import NavInertialMeasModule
from vehiclesim.measurement_modules.NavZuptInertialMeasModule import NavZuptInertialMeasModule
from vehiclesim.measurement_simulations.imu_sim import simulate_imu

from filter_tools.estimators import Estimators

from postprocessing.standard_mc_plotter import standard_mc_plotter

VEH_CONFIG = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\vehiclesim\\vehicle_configs\\5a_config.yaml'
#%%
# load csv data file
CSV = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\simulation\\processed\\FF\\FF2\\FF2.csv'
df = pd.read_csv(CSV, dtype={'SUBSET':str}, header='infer')
# sensor variables
steer_can = df['steer_ang']
vx_can = df['vx']
# imu_gyro_z = df['imu_gyro_z']
# truth variables
N_truth = df['Y']
E_truth = df['X']
vy_truth = df['vy']
yaw_truth = df['yaw']
yaw_rate_truth = df['yaw_rate']
hitch_truth = df['hitch']
hitch_rate_truth = df['hitch_rate']
# other variables
vx_thresh = 0.1
t = df['t']
dt = round(np.mean(np.diff(t)),3)
L = len(t)
N = 9 # number of filter states
M = 2 # number of measurements

# load trucksim mat file (for custom imu simulation)
TS_MAT = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\simulation\\raw\\FF\\FF2\\FF2_TS.mat'
ts_mat = scipy.io.loadmat(TS_MAT)
L_ts = len(ts_mat['T_Event'].squeeze())

# generate true linear accelerations and angular rates
lin_accel = [
    ts_mat['Ax'].squeeze()*9.81,
    ts_mat['Ay'].squeeze()*9.81,
    ts_mat['Az'].squeeze()*9.81
]
ang_vel = [
    np.deg2rad(ts_mat['AVx'].squeeze()),
    np.deg2rad(ts_mat['AVy'].squeeze()),
    np.deg2rad(ts_mat['AVz'].squeeze()),
]


#%%
# set up monte carlo loop variables and filter modules 
L_MC = 500

# storage variables
x_mc = np.zeros((N, L_MC, L)) # state
x_error_mc = np.zeros((N, L_MC, L)) # state errors
P_mc = np.zeros((N, N, L_MC, L))
# instantiate modules
standard_state_module = NavFullStateModule(
    error_model=np.diag([
        0.1,# N
        0.1,# E
        0.0001,# vx    
        0.05,# vy
        0.0001,# yaw rate
        0.001,# yaw
        0.0001,# hitch_rate
        0.005,# hitch
        1e-6 # bias ar
    ]),
    # error_model=1e-5*np.eye(N),
    vehicle_config=VEH_CONFIG,
)
zupt_state_module = NavZuptStateModule(
    error_model=np.diag([
        1e-3,# N
        1e-3,# E
        1e-3,# vx    
        1e-3,# vy
        1e-4,# yaw rate
        1e-3,# yaw
        1e-4,# hitch_rate
        1e-3,# hitch
        1e-6 # bias ar
    ]),
)
inertial_measurement_module = NavInertialMeasModule(
    error_model=np.diag([
        1e-3,# vx_can
        0.005**2 # imu_gyro_z
    ]),
)
zupt_measurement_module = NavZuptInertialMeasModule(
    error_model=np.diag([
        1e-3,
        1e-3
    ])
)
estimators = Estimators(n=N ,m=M)

#%%
# monte carlo loop
for m in tqdm(range(0,L_MC)):

    # grade = random.randint(1,5)
    # setup variance variables (IMU for now)
    # TODO: Vary grade. Testing consumer grade only for now
    imu = simulate_imu(1, lin_accel, ang_vel, L_ts)
    steer_can = steer_can + np.deg2rad(0.5)*np.random.randn(L)
    vx_can = vx_can + 0.01*np.random.randn(L)

    x_ = [] # state
    x_error_ = [] # state error

    P_ = []
    innov_ = []
    K_ = []

    # initialize (using truth)
    x = np.array([
        [N_truth[0]],
        [E_truth[0]],
        [vx_can[0]],
        [vy_truth[0]],
        [yaw_rate_truth[0]],
        [yaw_truth[0]],
        [hitch_rate_truth[0]],
        [hitch_truth[0]],
        [0]
    ])
    x_truth = x
    x_truth[-1] = imu.bias.gyro[2]
    x_error = x - x_truth
    # P = np.diag([1.37791859e+01, 1.10708923e+01, 9.90195136e-03, 1.76029376e+01,
    #          1.36849423e-03, 1.01000014e+02, 6.32907841e-01, 3.12263142e-01, 1.37416405e-03])
    P = np.diag([
            0.05,# N
            0.05,# E
            0.001,# vx    
            0.01,# vy
            0.0001,# yaw rate
            0.001,# yaw
            0.0001,# hitch_rate
            0.001,# hitch
            1e-6 # bias ar
    ])
    # x_.append(x)
    # x_error_.append(x_error)
    P_.append(P)
    x_mc[:,m,0] = x.squeeze()
    x_error_mc[:,m,0] = x_error.squeeze()
    # P_mc[:,:,m,0] = P

    # ---- filter loop ----
    for k in range(0,L-1):
        # zupt condition
        if vx_can[k+1] <= vx_thresh:
            PHI, G, Q = zupt_state_module.generate_state_model()
            # model input
            u = np.array([0])
            # zupt measurement model
            z, H, R = zupt_measurement_module.generate_meas_model()
        # standard state model
        else:
            PHI, G, Q = standard_state_module.generate_state_model(steer_can[k+1], x, dt)
            # model input
            u = np.array([steer_can[k+1]]) # single element array for matrix operation
            # measurement model
            z, H, R = inertial_measurement_module.generate_meas_model(vx_can[k+1], imu.gyro[2,k+1])
        # kalman filter core
        x, P, K, innov = estimators.kf(
            T=dt,
            num_inputs=1,
            F=PHI,
            B=G,
            u=u,
            Q=Q,
            z=z,
            H=H,
            R=R,
            P=P,
            x=x
        )
        # get truth state for error
        x_truth = np.array([
            [N_truth[k+1]],
            [E_truth[k+1]],
            [vx_can[k+1]],
            [vy_truth[k+1]],
            [yaw_rate_truth[k+1]],
            [yaw_truth[k+1]],
            [hitch_rate_truth[k+1]],
            [hitch_truth[k+1]],
            [imu.bias.gyro[2]]
        ])
        x_error = x - x_truth
        x_.append(x)
        x_error_.append(x_error)
        P_.append(P)
        K_.append(K)
        innov_.append(innov)
        
        # populate mc variables
        x_mc[:,m,k+1] = x.squeeze()
        x_error_mc[:,m,k+1] = x_error.squeeze()
        # P_mc[:,:,m,k+1] = P

    # ---- end of filter loop (single MC) ----
    
    # populate mc variables
    # x_array = np.array(x_).squeeze().transpose()
    # x_error_array = np.array(x_error_).squeeze().transpose()
    # x_mc[:,m,:] = x_array
    # x_error_mc[:,m,:] = x_error_array

# ---- end of mc loop ----

# extract statistics (mean/stds along mc dimension)
x_mc_mean = np.mean(x_mc, axis=1)
x_mc_std = np.std(x_mc, axis=1)
x_error_mc_mean = np.mean(x_error_mc, axis=1)
x_error_mc_std = np.std(x_error_mc, axis=1)

# extract theorethical std from filter covariance
P_array = np.array(P_)
theo_std = np.sqrt(np.diagonal(P_array, axis1=1, axis2=2).transpose())
# P_mean = np.mean(P_mc, axis=2) # mean across all mc runs
# theo_std = np.sqrt(np.diagonal(P_mean, axis1=0, axis2=1).transpose())

#%%
# call postprocessing plotting function
standard_mc_plotter(x_mc=x_mc,
                    x_error_mc=x_error_mc,
                    x_mc_mean=x_mc_mean,
                    x_mc_std=x_mc_std,
                    x_error_mc_mean=x_error_mc_mean,
                    x_error_mc_std=x_error_mc_std,
                    theo_std=theo_std,
                    t=t,
                    sigma_bound_fator=1,
                    error_only=True,
                    interactive=True)
#TODO:
# - Create MC loop (DONE)
# - Integrate filter (DONE)
# - Postprocess (DONE)
# - Create MC plotting helper functions
# - Figure out how to inject variance into model parameters