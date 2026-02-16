#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io
import random
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import cv2
from concurrent.futures import ThreadPoolExecutor

from vehiclesim.state_modules.NavFullStateModule import NavFullStateModule
from vehiclesim.state_modules.NavZuptStateModule import NavZuptStateModule
from vehiclesim.measurement_modules.NavLonVelMeasModule import NavLonVelMeasModule
from vehiclesim.measurement_modules.NavInertialMeasModule import NavInertialMeasModule
from vehiclesim.measurement_modules.NavZuptInertialMeasModule import NavZuptInertialMeasModule
from vehiclesim.measurement_modules.NavHitchMeasModule import NavHitchMeasModule

from vehiclesim.measurement_simulations.imu_sim_advanced import simulate_imu_advanced
from vehiclesim.measurement_simulations.imu_sim import simulate_imu

from vehiclesim.vehicle_configs.veh_params import vp as vp_dict
from vehiclesim.mc_tools.mc_veh_config import perturb_parameters
from vehiclesim.mc_tools.multi_dim_interp import multi_dim_interp

from filter_tools.estimators import Estimators

from trailer_pose_network.models.spacetime.finalized.async_space_time_cross_attention import AsyncSpaceTimeCrossAttention

from postprocessing.standard_mc_plotter import standard_mc_plotter
from postprocessing.vio_mc_plotter import vio_mc_plotter
from postprocessing.joint_kf_vio_mc_plotter import joint_kf_vio_mc_plotter

#%%
# load csv data file
SET = 'FF'
SUBSET = 'FF2'
CSV = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\simulation\\processed\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'.csv'
df = pd.read_csv(CSV, dtype={'SUBSET':str}, header='infer')
# sensor variables
steer_truth = df['steer_ang']
vx_truth = df['vx']
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
L = len(df)
N = 9 # number of filter states
M = 2 # number of measurements

#%%
# load 10Hz csv for vio mc
VIO_CSV = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\simulation\\10Hz\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'.csv'
vio_df = pd.read_csv(VIO_CSV, dtype={'SUBSET':str}, header='infer')
t_vio = vio_df['t']
L_vio = len(vio_df)
N_VIO = 3

# truth variables
N_truth_vio = vio_df['Y']
E_truth_vio = vio_df['X']
yaw_truth_vio = vio_df['yaw']

#%% 
# load trucksim mat file (for custom imu simulation)
TS_MAT = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\simulation\\raw\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'_TS.mat'
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
# utility functions
def _body_to_tangent_frame_translation(pose1, dx_body, dy_body):
    """
    Convert from body frame translation to tangent plane displacement
    
    Args:
        pose1: (X1, Y1, yaw1) - starting pose
        dx_body, dy_body: translation in body frame
        
    Returns:
        (dx_world, dy_world) - translation in world/tangent frame
    """
    X1, Y1, yaw1 = pose1
    
    # Rotate from body frame to world frame
    cos_yaw = np.cos(yaw1)  # Note: positive yaw1
    sin_yaw = np.sin(yaw1)  # Note: positive yaw1
    
    dx_world = cos_yaw * dx_body - sin_yaw * dy_body
    dy_world = sin_yaw * dx_body + cos_yaw * dy_body
    
    X2 = X1 + dx_world
    Y2 = Y1 + dy_world
    
    return X2, Y2

def load_image(image_path):
    """Loads an image using OpenCV."""
    try:
            img = cv2.imread(image_path)
            if img is None:
                    print(f"Error: Could not read image at {image_path}")
                    return None
            return img
    except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

#%%
# set up monte carlo loop variables and filter modules 
L_MC = 50

# storage variables (filter)
x_mc = np.zeros((N, L_MC, L)) # state
x_error_mc = np.zeros((N, L_MC, L)) # state errors
P_mc = np.zeros((N, N, L_MC, L))

# storage variables (vio)
x_mc_vio = np.zeros((N_VIO, L_MC, L_vio))
x_error_mc_vio =np.zeros((N_VIO, L_MC, L_vio))

# create network model and dataloader
# === DATALOADER PARAMETERS ===
SEQ_LOOKBACK = 2
IMG_SIZE = (224,224)
BATCH_SIZE = 1

# === VIO MODEL PARAMETERS ===
VIO_WEIGHTS= "C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\vehiclesim\\weights\\sim_vio_v0.pth"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 16
NUM_FRAMES = 2
NUM_IMU_SAMPLES = 5
VIO_EMBED_DIM = 384
NUM_HEADS = 8
DEPTH = 8
IN_CHANNELS = 3
IMU_CHANNELS = 8
DROPOUT = 0.
NUM_OUTPUTS = 3

transform_img=v2.Compose([
    v2.ToPILImage(),
    v2.Resize(IMG_SIZE),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

vio_network_model = AsyncSpaceTimeCrossAttention(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=IN_CHANNELS,
    embed_dim=VIO_EMBED_DIM,
    num_frames=NUM_FRAMES,
    num_imu_samples=NUM_IMU_SAMPLES,
    imu_channels=IMU_CHANNELS,
    num_heads=NUM_HEADS,
    depth=DEPTH,
    dropout=DROPOUT,
)
vio_network_model = vio_network_model.to(DEVICE)
# Load weights
vio_state_dict = torch.load(VIO_WEIGHTS)
vio_network_model.load_state_dict(vio_state_dict)

# instantiate modules
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
vx_measurement_module = NavLonVelMeasModule(
    error_model=np.diag([
        1e-3
    ])
)
inertial_measurement_module = NavInertialMeasModule(
    error_model=np.diag([
        5e-2 # imu_gyro_z
    ]),
)
zupt_measurement_module = NavZuptInertialMeasModule(
    error_model=np.diag([
        1e-3,
        1e-3
    ])
)
hitch_measurement_module = NavHitchMeasModule(
    error_model=np.diag([
        1e-3
    ])
)
kf_estimator = Estimators(n=N ,m=M)

#%%
# monte carlo loop
for m in tqdm(range(0,L_MC)):
    # Generate new vehicle configs for every MC
    perturbed_vp = perturb_parameters(
        nominal_params=vp_dict,
        percentage=0.5,
        distribution='uniform'
    )
    standard_state_module = NavFullStateModule(
        error_model=np.diag([
            0.1,# N
            0.1,# E
            0.01,# vx    
            0.01,# vy
            0.001,# yaw rate
            0.0001,# yaw
            0.001,# hitch_rate
            0.0001,# hitch
            1e-6 # bias ar
        ]),
        vehicle_config=perturbed_vp,
    )
    # resimulate IMU for every MC
    imu = simulate_imu_advanced(
        lin_accel,
        ang_vel,
        accel_bias_sigma=(0.05, 0.05, 0.05),
        accel_bias_tau = (300.0, 300.0, 300.0),  # seconds (5 minutes)
        accel_rw_sigma = (0.002, 0.002, 0.002),  # m/s^2 (white noise)
        gyro_bias_sigma = (0.005, 0.005, 0.005),  # rad/s (about 0.1 deg/s or 360 deg/hr)
        gyro_bias_tau = (300.0, 300.0, 300.0),  # seconds (5 minutes)
        gyro_rw_sigma = (0.0007, 0.0007, 0.0007),  # rad/s (about 0.02 deg/s white noise)
        dt=dt,
        L=L_ts,
    )
    # imu = simulate_imu(
    #     grade=1,
    #     accel=lin_accel,
    #     gyro=ang_vel,
    #     L=L
    # )
    # initialize (using truth)
    
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
    x_truth = x
    x_error = x - x_truth
    P_ = []
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
    P_.append(P)
    x_mc[:,m,0] = x.squeeze()
    x_error_mc[:,m,0] = x_error.squeeze()

    # initialize vio
    x_vio = np.array([
        [N_truth_vio[0]],
        [E_truth_vio[0]],
        [yaw_truth_vio[0]]
    ])
    x_truth_vio = x_vio
    x_error_vio = x_vio - x_truth_vio

    x_mc_vio[:,m,0] = x_vio.squeeze()
    x_error_mc_vio[:,m,0] = x_error_vio.squeeze()
    
    # ==== FILTER LOOP ====
    for k in range(0,L-1):
        # ---- ZUPT ----
        if vx_truth[k+1] <= vx_thresh:
            # time update
            PHI, F, G, Q = zupt_state_module.generate_state_model()
            u = np.array([[0]])
            x, P = kf_estimator.kf_predict(x, P, PHI, F, G, u, Q)

            # measurement update
            z, H, h_x, R = zupt_measurement_module.generate_meas_model(x, imu.gyro[2,k+1])
            x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)

        # ---- STANDARD NAV STATE/MEASUREMENT MODEL ----
        else:
            # time update
            PHI, F, G, Q = standard_state_module.generate_state_model(steer_truth[k+1], x, dt)
            u = np.array([[steer_truth[k+1]]]) # single element array for matrix operation
            x, P = kf_estimator.kf_predict(x, P, PHI, F, G, u, Q)

            # measurement update
            z, H, h_x, R = vx_measurement_module.generate_meas_model(x, vx_truth[k+1])
            x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)
            z, H, h_x, R = inertial_measurement_module.generate_meas_model(x, imu.gyro[2,k+1])
            x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)

            # z, H, h_x, R = hitch_measurement_module.generate_meas_model(x, hitch_rate_truth[k+1])
            # x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)

        # get truth state for error
        x_truth = np.array([
            [N_truth[k+1]],
            [E_truth[k+1]],
            [vx_truth[k+1]],
            [vy_truth[k+1]],
            [yaw_rate_truth[k+1]],
            [yaw_truth[k+1]],
            [hitch_rate_truth[k+1]],
            [hitch_truth[k+1]],
            [0]
        ])
        x_error = x - x_truth
        P_.append(P)
        # populate mc variables
        x_mc[:,m,k+1] = x.squeeze()
        x_error_mc[:,m,k+1] = x_error.squeeze()
    # ---- end of filter loop (single MC) ----

    # ==== VIO LOOP ====
    for j in range(0,L_vio-1):
        # extract the inertial measurements by matching times
        t_to_find = [t_vio.iloc[j], t_vio.iloc[j+1]]
        mask = t.isin(t_to_find)
        indices = t[mask].index.tolist()
        input_inert = np.array([
            steer_truth[indices[0]:indices[-1]+1],
            vx_truth[indices[0]:indices[-1]+1],
            imu.accel[0, indices[0]:indices[-1]+1],
            imu.accel[1, indices[0]:indices[-1]+1],
            imu.accel[2, indices[0]:indices[-1]+1],
            imu.gyro[0, indices[0]:indices[-1]+1],
            imu.gyro[1, indices[0]:indices[-1]+1],
            imu.gyro[2, indices[0]:indices[-1]+1]
        ]).transpose()
        input_inert = torch.tensor(input_inert).unsqueeze(0)

        # extract images
        left_paths = [vio_df['LRMC'].iloc[j], vio_df['LRMC'].iloc[j+1]]
        right_paths = [vio_df['RRMC'].iloc[j], vio_df['RRMC'].iloc[j+1]]

        with ThreadPoolExecutor(max_workers=32) as executor:
            left_images = list(executor.map(load_image, left_paths))
            right_images = list(executor.map(load_image, right_paths))

            image_pairs = list(zip(right_images, left_images))
            concat_images = list(executor.map(cv2.hconcat, image_pairs))

            concat_images = list(executor.map(transform_img, concat_images))

        input_cam = torch.stack(concat_images).unsqueeze(0)

        # cast inputs to device
        input_cam = input_cam.to(device=DEVICE, dtype=torch.float32)
        input_inert = input_inert.to(device=DEVICE, dtype=torch.float32)
        network_inputs = [input_cam, input_inert]

        # predict from model
        with torch.no_grad():
            vio_network_model.eval()
            trans_est, rot_est = vio_network_model(network_inputs)
            dx = trans_est.squeeze().cpu().numpy()[0]
            dy = trans_est.squeeze().cpu().numpy()[1]
            dyaw = rot_est.squeeze().cpu().numpy()
            
            # propagate position and orientation from vio output
            pose_prev = (x_vio[1,0], x_vio[0,0], x_vio[2,0])
            east_meas, north_meas = _body_to_tangent_frame_translation(pose_prev, dx, dy)
            yaw_meas = x_vio[2,0] + dyaw

            # state
            x_vio = np.array([
                [north_meas],
                [east_meas],
                [yaw_meas]
            ])
            # truth
            x_truth_vio = np.array([
                [N_truth_vio[j+1]],
                [E_truth_vio[j+1]],
                [yaw_truth_vio[j+1]]
            ])
            x_error_vio = x_vio - x_truth_vio

            # populate mc variables
            x_mc_vio[:, m, j+1] = x_vio.squeeze()
            x_error_mc_vio[:, m, j+1] = x_error_vio.squeeze()

    # ---- end of vio loop (single MC) ----
# ---- end of mc loop ----

# extract statistics (mean/stds along mc dimension) for filter
x_mc_mean = np.mean(x_mc, axis=1)
x_mc_std = np.std(x_mc, axis=1)
x_error_mc_mean = np.mean(x_error_mc, axis=1)
x_error_mc_std = np.std(x_error_mc, axis=1)

# extract theorethical std from filter covariance
P_array = np.array(P_)
theo_std = np.sqrt(np.diagonal(P_array, axis1=1, axis2=2).transpose())

# extract statistics (mean/stds along mc dimension) for vio
# interpolate vio estimates to match filter
x_mc_vio = multi_dim_interp(x_mc_vio, L)
x_error_mc_vio = multi_dim_interp(x_error_mc_vio, L)
x_mc_mean_vio = np.mean(x_mc_vio, axis=1)
x_mc_std_vio = np.std(x_mc_vio, axis=1)
x_error_mc_mean_vio = np.mean(x_error_mc_vio, axis=1)
x_error_mc_std_vio = np.std(x_error_mc_vio, axis=1)

#%%
# call postprocessing plotting function
joint_kf_vio_mc_plotter(
    x_mc=x_mc,
    x_error_mc=x_error_mc,
    x_mc_mean=x_mc_mean,
    x_mc_std=x_mc_std,
    x_error_mc_mean=x_error_mc_mean,
    x_error_mc_std=x_error_mc_std,
    theo_std=theo_std,
    x_mc_vio=x_mc_vio,
    x_error_mc_vio=x_error_mc_vio,
    x_mc_mean_vio=x_mc_mean_vio,
    x_mc_std_vio=x_mc_std_vio,
    x_error_mc_mean_vio=x_error_mc_mean_vio,
    x_error_mc_std_vio=x_error_mc_std_vio,
    t=t,
    sigma_bound_fator=1,
    error_only=False,
    interactive=True
)
# standard_mc_plotter(
#     x_mc=x_mc,
#     x_error_mc=x_error_mc,
#     x_mc_mean=x_mc_mean,
#     x_mc_std=x_mc_std,
#     x_error_mc_mean=x_error_mc_mean,
#     x_error_mc_std=x_error_mc_std,
#     theo_std=theo_std,
#     t=t,
#     sigma_bound_fator=1,
#     error_only=True,
#     interactive=True
# )

# vio_mc_plotter(
#     x_mc_vio, 
#     x_error_mc_vio, 
#     x_mc_mean_vio, 
#     x_mc_std_vio, 
#     x_error_mc_mean_vio, 
#     x_error_mc_std_vio, 
#     t, 
#     sigma_bound_fator=1, 
#     interactive=True, 
#     error_only=True
# )