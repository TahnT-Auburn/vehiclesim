#!/usr/bin/env python3
#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2

from vehiclesim.state_modules.NavFullStateModule import NavFullStateModule
from vehiclesim.state_modules.NavZuptStateModule import NavZuptStateModule
from vehiclesim.measurement_modules.NavInertialMeasModule import NavInertialMeasModule
from vehiclesim.measurement_modules.NavInertialDlVioMeasModule import NavInertialDlVioMeasModule

from filter_tools.estimators import Estimators
from trailer_pose_network.models.spacetime.async_space_time_ca_yaw_hist import AsyncSpaceTimeYawHist

from postprocessing.standard_state_est_plotter import standard_state_est_plotter

#%%
# load csv data files
# CSV = 'D:\\Tahn\\6_19_25\\csv\\raw\\original\\04\\04.csv'
# IMG_CSV = 'D:\\Tahn\\6_19_25\\csv\\synced_v2\\04\\04.csv'
IMG_CSV = "D:\\TestingData\\experimental\\10Hz\\original\\6_19_25\\02\\02.csv"
CSV = "D:\\TestingData\\experimental\\40Hz\\original\\6_19_25\\02\\02.csv"
df = pd.read_csv(CSV, dtype={'SUBSET':str}, header='infer')
img_df = pd.read_csv(IMG_CSV, dtype={'SUBSET':str}, header='infer')
L = len(df)
# sensor variables
steer_can = df['steer_ang']
vx_can = df['vx']
imu_accel_x = df['imu_accel_x']
imu_accel_y = df['imu_accel_y']
imu_accel_z = df['imu_accel_z']
imu_gyro_x = df['imu_gyro_x']
imu_gyro_y = df['imu_gyro_y']
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
# image variables
left_img_paths = img_df['LRMC']
right_img_paths = img_df['RRMC']
img_t = img_df['t']

# === MODEL PARAMETERS ===
NUM_FRAMES = 2
IMG_SIZE = (224,448)
NUM_IMU_SAMPLES = 5
EMBED_DIM = 384
NUM_HEADS = 8
DEPTH = 8
PATCH_SIZE = 16
IN_CHANNELS = 3
IMU_CHANNELS = 8
DROPOUT = 0.
NUM_OUTPUTS = 5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_WEIGHTS = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\async_space_time_yaw_hist\\async_space_time_yaw_hist_v3.pth"

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
        0.001,# yaw
        0.001,# hitch_rate
        0.01,# hitch
        1e-6 # bias ar
    ])
)
zupt_state_module = NavZuptStateModule(
    error_model=np.diag([
        2,# N
        2,# E
        0.001,# vx    
        0.01,# vy
        0.001,# yaw rate
        0.001,# yaw
        0.001,# hitch_rate
        0.01,# hitch
        1e-6 # bias ar
    ])
)
inertial_measurement_module = NavInertialMeasModule(
    error_model=np.diag([
        1e-3,# vx_can
        1e-2 # imu_gyro_z
    ])
)
inertial_dlvio_measurement_module = NavInertialDlVioMeasModule(
    error_model=np.diag([
       1e-3,# vx_can
       #1e-2,# imu_gyro_z
       3e-2,# yaw est from model 
    ]),
    model=AsyncSpaceTimeYawHist(
        IMG_SIZE,
        PATCH_SIZE,
        IN_CHANNELS,
        EMBED_DIM,
        NUM_FRAMES,
        NUM_IMU_SAMPLES,
        IMU_CHANNELS,
        NUM_HEADS,
        DEPTH,
        DROPOUT,
        NUM_OUTPUTS
    ),
    device=DEVICE,
    model_weights=MODEL_WEIGHTS,
    img_transforms=v2.Compose([
        v2.ToPILImage(),
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
    ]),
)
estimators = Estimators(n=9, m=3)

# filter loop
model_yaw_est = []
for k in tqdm(range(0,L-1)):
    # zupt condition
    if vx_can[k+1] <= vx_thresh:
        PHI, G, Q = zupt_state_module.generate_state_model()
    # standard state model
    else:
        PHI, G, Q = standard_state_module.generate_state_model(steer_can[k+1], x, dt)
    # model input
    u = np.array([steer_can[k+1]])
    # measurement model 
    # Checks to see if current time matches with a time image data is recieved.
    # If so use the mixed measurement module, otherwise default to inertial measurement module.
    index_mask = (img_t == t[k+1]) # data is preprocessed so that we can simply find matching timestamps.
    matched_index = img_t.index[index_mask].tolist()
    if matched_index: # found matching time
        assert len(matched_index) == 1,\
            f'Length of match indices is invalid. Must be 1.'
        # if img_t[matched_index].item() % 1.0 == 0.0: # update every 5 seconds
        matched_index = matched_index[0]
        current_index = k+1
        start_index = max(0, current_index - NUM_IMU_SAMPLES + 1)
        # prepare inputs to mixed measurement module
        imu_bank = np.array([
            [imu_accel_x[start_index:current_index+1]],
            [imu_accel_y[start_index:current_index+1]],
            [imu_accel_z[start_index:current_index+1]],
            [imu_gyro_x[start_index:current_index+1]],
            [imu_gyro_y[start_index:current_index+1]],
            [imu_gyro_z[start_index:current_index+1]],
        ]).squeeze().transpose()
        # imu_bank = np.random.randn(5,6)
        can_bank = np.array([
            [steer_can[start_index:current_index+1]],
            [vx_can[start_index:current_index+1]]
        ]).squeeze().transpose()
        # can_bank = np.random.randn(5,2)
        # yaw_hist_bank = np.array(x_).squeeze()[-NUM_IMU_SAMPLES+1:][:,5] # creates array of yaw estimate histories from the most current entry back to NUM_IMU_SAMPLES-1.
        # yaw_hist_bank = yaw_etal[start_index:current_index].to_numpy()
        yaw_hist_bank = np.array(x_).squeeze()[-1:][:,5]
        # yaw_hist_bank = np.array(yaw_etal[current_index])
        image_paths = [left_img_paths.tolist()[max(0,matched_index-NUM_FRAMES+1):matched_index+1], right_img_paths.tolist()[max(0,matched_index-NUM_FRAMES+1):matched_index+1]]
        # populate mixed measurement module
        z, H, R = inertial_dlvio_measurement_module.generate_meas_model(vx_can[k+1],
                                                                        imu_gyro_z[k+1],
                                                                        imu_bank,
                                                                        can_bank,
                                                                        yaw_hist_bank,
                                                                        image_paths)
        model_yaw_est.append(z[1,0])
        # else:
        #     z, H, R = inertial_measurement_module.generate_meas_model(vx_can[k+1],
        #                                                             imu_gyro_z[k+1])
    # if images not recieved (no matching times) then default to standard inertial update
    else:
        z, H, R = inertial_measurement_module.generate_meas_model(vx_can[k+1],
                                                                  imu_gyro_z[k+1])
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
    x_.append(x)
    P_.append(P)
    K_.append(K)
    innov_.append(innov)
    
# postprocessing
x_plot = np.array(x_).squeeze().transpose().tolist()
x_truth_plot = [N_etal, E_etal, vx_can, vy_etal, yaw_rate_etal, yaw_etal, hitch_rate_etal, hitch_etal]
standard_state_est_plotter(x_plot, x_truth_plot, t, interactive=True)

plt.plot(model_yaw_est)
plt.show()