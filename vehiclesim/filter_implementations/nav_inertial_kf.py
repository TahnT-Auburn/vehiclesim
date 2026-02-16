#!/usr/bin/env python3
#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
from decimal import Decimal

from vehiclesim.state_modules.NavFullStateModule import NavFullStateModule
from vehiclesim.state_modules.NavZuptStateModule import NavZuptStateModule
from vehiclesim.measurement_modules.NavInertialMeasModule import NavInertialMeasModule
from vehiclesim.measurement_modules.NavZuptInertialMeasModule import NavZuptInertialMeasModule
from vehiclesim.vehicle_configs.veh_params import vp as vp_dict

from filter_tools.estimators import Estimators

from postprocessing.standard_state_est_plotter import standard_state_est_plotter
from postprocessing.bodyframe_displacements_plotter import body_frame_displacements_plotter

VEH_CONFIG = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\vehiclesim\\vehicle_configs\\5a_config.yaml'
#%%
# load csv data file
CSV = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TrainingData\\experimental\\40Hz\\original\\6_19_25\\01\\01.csv'
df = pd.read_csv(CSV, dtype={'SUBSET':str}, header='infer')
L = len(df)
# sensor variables
steer_can = df['steer_ang']
vx_can = df['vx']
imu_gyro_z = df['imu_gyro_z']
# etalin variables for truth
N_etal = df['Y']
E_etal = df['X']
vy_etal = df['vy']
yaw_etal = df['yaw']
yaw_rate_etal = df['yaw_rate']
hitch_etal = df['hitch']
hitch_rate_etal = df['hitch_rate']
# other variables
vx_thresh = 0.1
t = df['t']
dt = round(np.mean(np.diff(t)),3)

#%%
# filter implementation

# storage list
x_ = []
P_ = []
innov_ = []
K_ = []

# initialize
x = np.array([
    [N_etal[0]],
    [E_etal[1]],
    [vx_can[0]],
    [vy_etal[0]],
    [yaw_rate_etal[0]],
    [yaw_etal[0]],
    [hitch_rate_etal[0]],
    [hitch_etal[0]],
    [0]
])
# P = np.diag([1.37791859e+01, 1.10708923e+01, 9.90195136e-03, 1.76029376e+01,
#              1.36849423e-03, 1.01000014e+02, 6.32907841e-01, 3.12263142e-01, 1.37416405e-03])
P = np.diag([
    0.05,    # N
    0.05,    # E  
    0.001,   # vx
    0.5,    # vy
    0.001,  # yaw_rate
    0.01,    # yaw  <-- Changed from 101 to 0.01
    0.0001,  # hitch_rate
    0.001,   # hitch
    1e-6     # bias
])
x_.append(x)
P_.append(P)

# instantiate modules
standard_state_module = NavFullStateModule(
    error_model=np.diag([
        0.5,# N
        0.5,# E
        0.001,# vx    
        0.01,# vy
        0.0001,# yaw rate
        0.001,# yaw
        0.0001,# hitch_rate
        0.001,# hitch
        1e-6 # bias ar
    ]),
    vehicle_config=vp_dict,
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
        1e-3,# vx_can
        5e-3 # imu_gyro_z
    ]),
)
zupt_measurement_module = NavZuptInertialMeasModule(
    error_model=np.diag([
        1e-3,
        1e-3
    ])
)
kf_estimator = Estimators(n=9 ,m=2)

# filter loop
vx_last_set = False  
for k in tqdm(range(0,L-1)):
    # use last vel if nan
    if np.isnan(vx_can[k+1]):
        if not vx_last_set:
            vx_last = float(vx_can[k])
            vx_last_set = True  
        vx_can.iloc[k+1] = vx_last 
    # ---- ZUPT ----
    if vx_can[k+1] <= vx_thresh:
        # time update
        PHI, G, Q = zupt_state_module.generate_state_model()
        u = np.array([[0]])
        x, P = kf_estimator.kf_predict(x, P, PHI, G, u, Q)

        # measurement update
        z, H, h_x, R = zupt_measurement_module.generate_meas_model(x, imu_gyro_z[k+1])
        x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)

    # ---- STANDARD NAV STATE/MEASUREMENT MODEL ----
    else:
        # time update
        PHI, G, Q = standard_state_module.generate_state_model(steer_can[k+1], x, dt)
        u = np.array([[steer_can[k+1]]]) # single element array for matrix operation
        x, P = kf_estimator.kf_predict(x, P, PHI, G, u, Q)

        # measurement update
        z, H, h_x, R = inertial_measurement_module.generate_meas_model(x, vx_can[k+1], imu_gyro_z[k+1])
        x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)
    # # kalman filter core
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
    if np.isnan(x).any():
        print(f"Nan(s) detected in state vector x: {x}")
    # store variables
    # if Decimal(str(t[k+1])) % Decimal(str(0.1)) == 0:
    x_.append(x)
    P_.append(P)
    K_.append(K)
    innov_.append(innov)
    
# postprocessing
x_plot = np.array(x_).squeeze().transpose().tolist()
x_truth_plot = [N_etal, E_etal, vx_can, vy_etal, yaw_rate_etal, yaw_etal, hitch_rate_etal, hitch_etal]
P_array = np.array(P_)
# std = np.sqrt(np.diagonal(P_array, axis1=1, axis2=2).transpose())
std=None
# x_truth_plot = [N_truth, E_truth, vx_truth, vy_truth, yaw_rate_truth, yaw_truth, hitch_rate_truth, hitch_truth]
standard_state_est_plotter(x_plot, x_truth_plot, std, t, interactive=True)
body_frame_displacements_plotter(x_plot, x_truth_plot, interactive=True)