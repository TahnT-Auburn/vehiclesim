#!/usr/bin/env python3
#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('ipympl')

from vehiclesim.state_modules.NavOdomStateModule import NavOdomStateModule
from vehiclesim.state_modules.NavOdomZuptStateModule import NavOdomZuptStateModule
from vehiclesim.measurement_modules.NavInertialOdomMeasModule import NavInertialOdomMeasModule
from vehiclesim.measurement_modules.NavOdomZuptMeasModule import NavOdomZuptMeasModule
from vehiclesim.measurement_modules.NavOdomMeasModule import NavOdomMeasModule
from vehiclesim.measurement_simulations.body_displacement_meas_sim import body_disp_meas_sim

from filter_tools.estimators import Estimators

from postprocessing.odom_state_est_plotter import odom_state_est_plotter
from postprocessing.bodyframe_displacements_plotter import body_frame_displacements_plotter

VEH_CONFIG = 'C:\\Users\\Tahn\\SoftDevel\\vehiclesim\\vehiclesim\\vehicle_configs\\5a_config.yaml'
MODEL_OUTPUT = 'C:\\Users\\Tahn\\SoftDevel\\vehiclesim\\data\\experimental\\csv\\misc\\model_odom_outputs.csv'

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
# load model output of odom measurements
model_odom_meas = pd.read_csv(MODEL_OUTPUT)
dx_body_meas = model_odom_meas['dx_body'].to_list()
dy_body_meas = model_odom_meas['dy_body'].to_list()
dyaw_meas = model_odom_meas['dyaw'].to_list()
stop=1
# generate a 10hz mask using time vector
mask_10hz = [Decimal(str(t_)) % Decimal('0.1') == 0 for t_ in t]
N_etal_10hz = N_etal[mask_10hz].reset_index(drop=True)
E_etal_10hz = E_etal[mask_10hz].reset_index(drop=True)
yaw_etal_10hz = yaw_etal[mask_10hz].reset_index(drop=True)
# N_etal_10hz = N_etal[mask_10hz].reset_index(drop=True)[0:12066]
# E_etal_10hz = E_etal[mask_10hz].reset_index(drop=True)[0:12066]
# yaw_etal_10hz = yaw_etal[mask_10hz].reset_index(drop=True)[0:12066]

# pregenerate simulated odom measurements
dx_body_meas, dy_body_meas, dyaw_meas = body_disp_meas_sim(
    N_etal_10hz.to_numpy(),
    E_etal_10hz.to_numpy(),
    yaw_etal_10hz.to_numpy(),
    sigmas=np.array([0.01,0.01,0.001]),
    # biases=np.array([0.0, 0.0, 0.0])
    biases=np.array([0.01, 0.001, 0.0001])
)

# visualize measurements
dx_body_truth, dy_body_truth, dyaw_truth = body_disp_meas_sim(
    N_etal_10hz.to_numpy(),
    E_etal_10hz.to_numpy(),
    yaw_etal_10hz.to_numpy(),
)

# dx_body_meas = model_odom_meas['dx_body'].to_list()
# dy_body_meas = model_odom_meas['dy_body'].to_list()
# dyaw_meas = model_odom_meas['dyaw'].to_list()

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Body Frame Displacements')
ax1.plot(dx_body_truth)
ax1.plot(dx_body_meas)
ax1.legend(['Truth','Est'])
ax1.set_ylabel('X Displacement [m]')
ax2.plot(dy_body_truth)
ax2.plot(dy_body_meas)
ax2.set_ylabel('Y Displacement [m]')
ax3.plot(np.rad2deg(dyaw_truth))
ax3.plot(np.rad2deg(dyaw_meas))
ax3.set_ylabel('Yaw Displacement [deg]')
plt.tight_layout()
plt.show()

# displacements error plot
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Body Frame Displacement Errors')
ax1.plot(dx_body_truth - dx_body_meas)
ax1.set_ylabel('X Displacement Error [m]')
ax2.plot(dy_body_truth - dy_body_meas)
ax2.set_ylabel('Y Displacement Error [m]')
ax3.plot(np.rad2deg(dyaw_truth) - np.rad2deg(dyaw_meas))
ax3.set_ylabel('Yaw Displacement Error [deg]')
plt.tight_layout()
plt.show()
    
#%%
# filter implementation

# storage list
x_ = []
P_ = []
innov_ = []
K_ = []

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
    [0],
    [0],
    [0],
    [0]
])
P = np.diag([1.37791859e+01, 1.10708923e+01, 9.90195136e-03, 1.76029376e+01,
             1.36849423e-03, 1.01000014e+02, 6.32907841e-01, 3.12263142e-01,
             1.37416405e-03, 1.37416405e-03, 1.37416405e-03, 1.37416405e-03])
x_.append(x)
P_.append(P)

# instantiate modules
odom_state_module = NavOdomStateModule(
    error_model=np.diag([
        2,# N
        2,# E
        0.001,# vx    
        0.01,# vy
        0.001,# yaw rate
        0.01,# yaw
        0.001,# hitch_rate
        0.01,# hitch
        1e-6, # bias ar
        1e-9, # bias dx
        1e-9, # bias dy
        1e-9, # bias dyaw
    ]),
    vehicle_config=VEH_CONFIG,
)
odom_zupt_state_module = NavOdomZuptStateModule(
    error_model=np.diag([
        1e-3,# N
        1e-3,# E
        1e-3,# vx    
        1e-3,# vy
        1e-4,# yaw rate
        1e-3,# yaw
        1e-4,# hitch_rate
        1e-3,# hitch
        1e-6, # bias ar
        1e-9, # bias dx
        1e-9, # bias dy
        1e-9, # bias dyaw
    ]),
)
inertial_odom_measurement_module = NavInertialOdomMeasModule(
    error_model=np.diag([
        1e-3,# vx_can
        5e-3 # imu_gyro_z
    ]),
)
odom_measurement_module = NavOdomMeasModule(
    error_model=np.diag([
        1e-3, # dx
        1e-3, # dy
        1e-3 # dyaw
    ]),
    N_init=N_etal[0],
    E_init=E_etal[0],
    yaw_init=yaw_etal[0],
)
odom_zupt_measurement_module = NavOdomZuptMeasModule(
    error_model=np.diag([
        1e-3,
        1e-3
    ])
)
estimators = Estimators(n=12 ,m=2)

# filter loop
j = 0
for k in tqdm(range(0,L-1)):
    if k == 0:
        x_prev = x
    # zupt condition
    if vx_can[k+1] <= vx_thresh:
        PHI, G, Q = odom_zupt_state_module.generate_state_model()
        # model input
        u = np.array([0])
        # zupt measurement model
        z, H, R = odom_zupt_measurement_module.generate_meas_model()
    # standard state model
    else:
        PHI, G, Q = odom_state_module.generate_state_model(steer_can[k+1], x, dt)
        # model input
        u = np.array([steer_can[k+1]]) # single element array for matrix operation
        # measurement model
        z, H, R = inertial_odom_measurement_module.generate_meas_model(vx_can[k+1], imu_gyro_z[k+1])
    if (Decimal(str(t[k+1])) % Decimal('0.1') == 0): #and j < len(model_odom_meas): # Every 10Hz use odom
        # z, H, R = inertial_measurement_module.generate_meas_model(vx_can[k+1], imu_gyro_z[k+1])
        z, H, R = odom_measurement_module.generate_meas_model(x, x_prev, dx_body_meas[j],dy_body_meas[j],dyaw_meas[j])
        x_prev = x
        j+=1
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
    # if Decimal(str(t[k+1])) % Decimal(str(0.1)) == 0:
    x_.append(x)
    P_.append(P)
    K_.append(K)
    innov_.append(innov)

# postprocessing
x_plot = np.array(x_).squeeze().transpose().tolist()
x_truth_plot = [N_etal, E_etal, vx_can, vy_etal, yaw_rate_etal, yaw_etal, hitch_rate_etal, hitch_etal]
# x_truth_plot = [N_truth, E_truth, vx_truth, vy_truth, yaw_rate_truth, yaw_truth, hitch_rate_truth, hitch_truth]
odom_state_est_plotter(x_plot, x_truth_plot, t, interactive=True)
body_frame_displacements_plotter(x_plot, x_truth_plot, interactive=True)