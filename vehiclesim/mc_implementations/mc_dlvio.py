#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import cv2
from concurrent.futures import ThreadPoolExecutor

from vehiclesim.measurement_simulations.imu_sim_advanced import simulate_imu_advanced
from vehiclesim.measurement_simulations.imu_sim import simulate_imu

from trailer_pose_network.models.spacetime.finalized.async_space_time_cross_attention import AsyncSpaceTimeCrossAttention
from trailer_pose_network.models.spacetime.finalized.trailer_hitch_model import HitchModel

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader
from trailer_pose_network.dataloaders.trailer_hitch_dataloader import HitchDataloader

from postprocessing.vio_mc_plotter import vio_mc_plotter

SET = 'FF'
SUBSET = 'FF2'

#%%
# load csv data file
BASE_CSV = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\simulation\\10Hz\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'.csv'
IMU_CSV = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\simulation\\processed\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'.csv'
df = pd.read_csv(BASE_CSV, dtype={'SUBSET':str}, header='infer')
imu_df = pd.read_csv(IMU_CSV, dtype={'SUBSET':str}, header='infer')
t_imu = imu_df['t']
L = len(df)
# sensor variables
steer_truth = imu_df['steer_ang']
vx_truth = imu_df['vx']

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

# ---- load roots for dataloader ----
SEQ_ROOT_PROCESSED = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\simulation\\10Hz\\'+SET+'\\'+SUBSET+'\\'
SEQ_ROOT_RAW = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\simulation\\processed\\'+SET+'\\'+SUBSET+'\\'

# ---- load trucksim mat file (for custom imu simulation) ----
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

vio_dataset = AsyncTemporalDataLoader(
    sequence_root_processed=SEQ_ROOT_PROCESSED,
    sequence_root_raw=SEQ_ROOT_RAW,
    sequential_lookback=SEQ_LOOKBACK,
    inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':False},
    transform_img=v2.Compose([
        v2.ToPILImage(),
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]),
)

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

#%%
# set up monte carlo loop
L_MC = 50
N = 3 # N, E, yaw from vio
x_mc = np.zeros((N, L_MC, L))
x_error_mc = np.zeros((N, L_MC, L))

#%%
# monte carlo loop
for m in tqdm(range(0, L_MC)):
    # simulate imu
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
    x_ = [] # state
    x_error_ = [] # state error

    # initialize states
    x = np.array([
        [N_truth[0]],
        [E_truth[0]],
        [yaw_truth[0]],
    ])
    x_truth = x
    x_error = x - x_truth
    x_.append(x)
    x_error_.append(x_)
    x_mc[:,m,0] = x.squeeze()
    x_error_mc[:,m,0] = x_error.squeeze()

    # ---- vio loop ----
    for k in range(0,L-1):
        # extract the inertial measurements by matching times
        t_to_find = [t.iloc[k], t.iloc[k+1]]
        mask = t_imu.isin(t_to_find)
        indices = t_imu[mask].index.tolist()
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
        left_paths = [df['LRMC'].iloc[k], df['LRMC'].iloc[k+1]]
        right_paths = [df['RRMC'].iloc[k], df['RRMC'].iloc[k+1]]

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
            pose_prev = (x[1,0], x[0,0], x[2,0])
            east_meas, north_meas = _body_to_tangent_frame_translation(pose_prev, dx, dy)
            yaw_meas = x[2,0] + dyaw

            # state
            x = np.array([
                [north_meas],
                [east_meas],
                [yaw_meas]
            ])
            # truth
            x_truth = np.array([
                [N_truth[k+1]],
                [E_truth[k+1]],
                [yaw_truth[k+1]]
            ])
            x_error = x - x_truth
            x_.append(x)
            x_error_.append(x_error)

            # populate mc variables
            x_mc[:, m, k+1] = x.squeeze()
            x_error_mc[:, m, k+1] = x_error.squeeze()
    # ---- end of vio loop (single MC) ----
# ---- end of mc loop ----

# extract statistics (mean/stds along mc dimension)
x_mc_mean = np.mean(x_mc, axis=1)
x_mc_std = np.std(x_mc, axis=1)
x_error_mc_mean = np.mean(x_error_mc, axis=1)
x_error_mc_std = np.std(x_error_mc, axis=1)

vio_mc_plotter(
    x_mc, 
    x_error_mc, 
    x_mc_mean, 
    x_mc_std, 
    x_error_mc_mean, 
    x_error_mc_std, 
    t, 
    sigma_bound_fator=1, 
    interactive=True, 
    error_only=False
)