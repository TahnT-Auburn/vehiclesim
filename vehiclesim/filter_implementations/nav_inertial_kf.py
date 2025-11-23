#!/usr/bin/env python3
#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
from decimal import Decimal

from vehiclesim.state_modules.NavFullStateModule import NavFullStateModule
from vehiclesim.state_modules.NavZuptStateModule import NavZuptStateModule
from vehiclesim.measurement_modules.NavInertialMeasModule import NavInertialMeasModule
from filter_tools.estimators import Estimators

from postprocessing.standard_state_est_plotter import standard_state_est_plotter
from postprocessing.bodyframe_displacements_plotter import body_frame_displacements_plotter

VEH_CONFIG = 'C:\\Users\\Tahn\\SoftDevel\\vehiclesim\\vehiclesim\\vehicle_configs\\5a_config.yaml'
#%%
# load csv data file
CSV = 'D:\\Tahn\\6_19_25\\csv\\raw\\original\\02\\02.csv'
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

t_ = [] # DELETE LATER
t_.append(t[0])
N_truth = []
E_truth = []
vx_truth = []
vy_truth = []
yaw_truth = []
yaw_rate_truth = []
hitch_truth = []
hitch_rate_truth = []
N_truth.append(N_etal[0])
E_truth.append(E_etal[0])
vx_truth.append(vx_can[0])
vy_truth.append(vy_etal[0])
yaw_truth.append(yaw_etal[0])
yaw_rate_truth.append(yaw_rate_etal[0])
hitch_truth.append(hitch_etal[0])
hitch_rate_truth.append(hitch_rate_etal[0])

# initialize
x = np.array([
    [0],
    [0],
    [vx_can[0]],
    [vy_etal[0]],
    [yaw_rate_etal[0]],
    [yaw_etal[0]],
    [hitch_rate_etal[0]],
    [hitch_etal[0]],
    [0]
])
P = np.diag([1.37791859e+01, 1.10708923e+01, 9.90195136e-03, 1.76029376e+01,
             1.36849423e-03, 1.01000014e+02, 6.32907841e-01, 3.12263142e-01, 1.37416405e-03])
x_.append(x)
P_.append(P)

# instantiate modules
standard_state_module = NavFullStateModule(
    error_model=np.diag([
        2,# N
        2,# E
        0.001,# vx    
        0.01,# vy
        0.001,# yaw rate
        0.01,# yaw
        0.001,# hitch_rate
        0.01,# hitch
        1e-6 # bias ar
    ]),
    vehicle_config=VEH_CONFIG,
)
zupt_state_module = NavZuptStateModule(
    error_model=np.diag([
        2,# N
        2,# E
        0.001,# vx    
        0.01,# vy
        0.001,# yaw rate
        0.01,# yaw
        0.001,# hitch_rate
        0.01,# hitch
        1e-6 # bias ar
    ]),
)
inertial_measurement_module = NavInertialMeasModule(
    error_model=np.diag([
        1e-3,# vx_can
        1e-2 # imu_gyro_z
    ]),
)
estimators = Estimators(n=9 ,m=2)

# filter loop
for k in tqdm(range(0,L-1)):
    # zupt condition
    if vx_can[k+1] <= vx_thresh:
        PHI, G, Q = zupt_state_module.generate_state_model()
    # standard state model
    else:
        PHI, G, Q = standard_state_module.generate_state_model(steer_can[k+1], x, dt)
    # model input
    u = np.array([steer_can[k+1]]) # single element array for matrix operation
    # measurement model
    z, H, R = inertial_measurement_module.generate_meas_model(vx_can[k+1], imu_gyro_z[k+1])
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
    # store variables
    if Decimal(str(t[k+1])) % Decimal(str(0.1)) == 0:
        x_.append(x)
        P_.append(P)
        K_.append(K)
        innov_.append(innov)
        t_.append(t[k+1])
        
        # truths DELETE LATER
        N_truth.append(N_etal[k+1])
        E_truth.append(E_etal[k+1])
        vx_truth.append(vx_can[k+1])
        vy_truth.append(vy_etal[k+1])
        yaw_truth.append(yaw_etal[k+1])
        yaw_rate_truth.append(yaw_rate_etal[k+1])
        hitch_truth.append(hitch_etal[k+1])
        hitch_rate_truth.append(hitch_rate_etal[k+1])
        
N_truth = np.array(N_truth)
E_truth = np.array(E_truth)
vx_truth = np.array(vx_truth)
vy_truth = np.array(vy_truth)
yaw_truth = np.array(yaw_truth)
yaw_rate_truth = np.array(yaw_rate_truth)
hitch_truth = np.array(hitch_truth)
hitch_rate_truth = np.array(hitch_rate_truth)

# postprocessing
x_plot = np.array(x_).squeeze().transpose().tolist()
# x_truth_plot = [N_etal, E_etal, vx_can, vy_etal, yaw_rate_etal, yaw_etal, hitch_rate_etal, hitch_etal]
x_truth_plot = [N_truth, E_truth, vx_truth, vy_truth, yaw_rate_truth, yaw_truth, hitch_rate_truth, hitch_truth]
standard_state_est_plotter(x_plot, x_truth_plot, t_, interactive=True)
body_frame_displacements_plotter(x_plot, x_truth_plot, interactive=True)