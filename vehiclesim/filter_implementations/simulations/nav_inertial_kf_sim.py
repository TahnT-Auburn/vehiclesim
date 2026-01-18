#!/usr/bin/env python3
#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io

from vehiclesim.state_modules.NavFullStateModule import NavFullStateModule
from vehiclesim.state_modules.NavZuptStateModule import NavZuptStateModule
from vehiclesim.measurement_modules.NavLonVelMeasModule import NavLonVelMeasModule
from vehiclesim.measurement_modules.NavInertialMeasModule import NavInertialMeasModule
from vehiclesim.measurement_modules.NavZuptInertialMeasModule import NavZuptInertialMeasModule
from vehiclesim.measurement_simulations.imu_sim import simulate_imu
from vehiclesim.measurement_simulations.body_displacement_meas_sim import body_disp_meas_sim

from filter_tools.estimators import Estimators

from postprocessing.standard_state_est_plotter import standard_state_est_plotter
from postprocessing.bodyframe_displacements_plotter import body_frame_displacements_plotter

VEH_CONFIG = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\vehiclesim\\vehicle_configs\\5a_config.yaml'
#%%
# load csv data file
CSV = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\simulation\\processed\\FF\\FF2\\FF2.csv'
# CSV = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TrainingData\\experimental\\40Hz\\original\\6_19_25\\01\\01.csv'

df = pd.read_csv(CSV, dtype={'SUBSET':str}, header='infer')
L = len(df)
# sensor variables
steer_truth = df['steer_ang'] + np.deg2rad(0.5)*np.random.randn(L)
vx_truth = df['vx'] + 0.01*np.random.randn(L)
imu_gyro_z = df['imu_gyro_z']
# etalin variables for truth
N_truth = df['Y']
E_truth = df['X']
vy_truth = df['vy']
yaw_truth = df['yaw']
yaw_rate_truth = df['yaw_rate']
hitch_truth = df['hitch']
hitch_rate_truth = df['hitch_rate']
# other variables
vx_thresh = 0.5
t = df['t']
dt = round(np.mean(np.diff(t)),3)
N = 9
M = 3
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
# filter implementation

# storage list
x_ = []
P_ = []
innov_ = []
K_ = []

# initialize
x = np.array([
    [N_truth[0]],
    [E_truth[0]],
    [vx_truth[0]],
    [vy_truth[0]],
    [yaw_rate_truth[0]],
    [yaw_truth[0]],
    [hitch_rate_truth[0]],
    [hitch_truth[0]],
    [0]
])
# P = np.diag([1.37791859e+01, 1.10708923e+01, 9.90195136e-03, 1.76029376e+01,
#              1.36849423e-03, 1.01000014e+02, 6.32907841e-01, 3.12263142e-01, 1.37416405e-03])
P = np.diag([
        0.05,# N
        0.05,# E
        0.001,# vx    
        0.01,# vy
        0.0001,# yaw rate
        0.001,# yaw
        0.0001,# hitch_rate
        0.001,# hitch
        1e-5 # bias ar
])
x_.append(x)
P_.append(P)

# instantiate modules
standard_state_module = NavFullStateModule(
    error_model=np.diag([
        0.05,# N
        0.05,# E
        0.0001,# vx    
        0.05,# vy
        0.00001,# yaw rate
        0.0001,# yaw
        0.00001,# hitch_rate
        0.0001,# hitch
        1e-6 # bias ar
    ]),
    vehicle_config=VEH_CONFIG,
)
zupt_state_module = NavZuptStateModule(
    error_model=np.diag([
        1e-6,# N
        1e-6,# E
        1e-6,# vx    
        1e-6,# vy
        1e-6,# yaw rate
        1e-6,# yaw
        1e-6,# hitch_rate
        1e-6,# hitch
        1e-6 # bias ar
    ]),
)
inertial_measurement_module = NavInertialMeasModule(
    error_model=np.diag([
        5e-3 # imu_gyro_z
    ]),
)
vx_measurement_module = NavLonVelMeasModule(
    error_model=np.diag([
        1e-3
    ])
)
zupt_measurement_module = NavZuptInertialMeasModule(
    error_model=np.diag([
        1e-3,
        1e-3
    ])
)
kf_estimator = Estimators(n=9 ,m=2)

# simulate imu
imu = simulate_imu(1, lin_accel, ang_vel, L_ts)

dx_body_meas, dy_body_meas, dyaw_meas = body_disp_meas_sim(
    N_truth.to_numpy(),
    E_truth.to_numpy(),
    yaw_truth.to_numpy(),
    sigmas=np.array([0.01,0.01,0.001]),
    biases=np.array([0.01, 0.001, 0.001])
)
# sythetic yaw rate
r = dyaw_meas / dt

# filter loop
vx_last_set = False  
for k in tqdm(range(0,L-1)):
    # ---- ZUPT ----
    if vx_truth[k+1] <= vx_thresh:
        # time update
        PHI, G, Q = zupt_state_module.generate_state_model()
        u = np.array([[0]])
        x, P = kf_estimator.kf_predict(x, P, PHI, G, u, Q)

        # measurement update
        z, H, h_x, R = zupt_measurement_module.generate_meas_model(x, imu.gyro[2,k+1])
        x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)

    # ---- STANDARD NAV STATE/MEASUREMENT MODEL ----
    else:
        # time update
        PHI, G, Q = standard_state_module.generate_state_model(steer_truth[k+1], x, dt)
        u = np.array([[steer_truth[k+1]]]) # single element array for matrix operation
        x, P = kf_estimator.kf_predict(x, P, PHI, G, u, Q)

        # measurement update
        z, H, h_x, R = vx_measurement_module.generate_meas_model(x, vx_truth[k+1])
        x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)
        z, H, h_x, R = inertial_measurement_module.generate_meas_model(x, imu_gyro_z[k+1])
        x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)

    # x, P, K, innov = estimators.kf(
    #     T=dt,
    #     num_inputs=1,
    #     F=PHI,
    #     B=G,
    #     u=u,
    #     Q=Q,
    #     z=z,
    #     H=H,
    #     R=R,
    #     P=P,
    #     x=x
    # )
    x_.append(x)
    P_.append(P)
    # K_.append(K)
    # innov_.append(innov)

# postprocessing
x_plot = np.array(x_).squeeze().transpose().tolist()
x_truth_plot = [N_truth, E_truth, vx_truth, vy_truth, yaw_rate_truth, yaw_truth, hitch_rate_truth, hitch_truth]
P_array = np.array(P_)
std = np.sqrt(np.diagonal(P_array, axis1=1, axis2=2).transpose())
standard_state_est_plotter(x_plot, x_truth_plot, std, t, interactive=True)
body_frame_displacements_plotter(x_plot, x_truth_plot, interactive=True)