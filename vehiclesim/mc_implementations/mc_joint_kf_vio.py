#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io
import random
from pathlib import Path
import pickle

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
from vehiclesim.measurement_modules.NavDLHitchMeasModule import NavDLHitchMeasModule

from vehiclesim.measurement_simulations.imu_sim_advanced import simulate_imu_advanced
from vehiclesim.measurement_simulations.imu_sim import simulate_imu

from vehiclesim.vehicle_configs.veh_params import vp as vp_dict
from vehiclesim.mc_tools.mc_veh_config import perturb_parameters
from vehiclesim.mc_tools.multi_dim_interp import multi_dim_interp

from filter_tools.estimators import Estimators

from trailer_pose_network.models.spacetime.finalized.async_space_time_cross_attention import AsyncSpaceTimeCrossAttention
from trailer_pose_network.models.spacetime.finalized.trailer_hitch_model import HitchModel

from nav_tools.imu_mechanization import ImuMech
from nav_tools.nav_utilities import body2rotm, rotm2eul
from postprocessing.joint_kf_vio_mc_plotter import joint_kf_vio_mc_plotter

from postprocessing.standard_mc_plotter import standard_mc_plotter
from postprocessing.vio_mc_plotter import vio_mc_plotter
from postprocessing.joint_ins_kf_vio_mc_plotter import joint_ins_kf_vio_mc_plotter

#%%
# Logging variables
# SAVE_RESULTS_PATH = 'C:\\Users\\Tahn\\SoftDevel\\vehiclesim\\evaluations\\sim_mcs\\FF2\\mc_results_low_grade_50_vp_lq_img_100.pkl'
SAVE_RESULTS_PATH = None

#%%
# load csv data file
SET = 'FF'
SUBSET = 'FF1'
CSV = 'D:\\TestingData\\simulation\\processed\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'.csv'
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

# velocities for ins
vz_truth = np.zeros(L)
ve_truth = np.cos(yaw_truth) * vx_truth - np.cos(yaw_truth) * vy_truth
vn_truth = np.sin(yaw_truth) * vx_truth + np.sin(yaw_truth) * vy_truth

#%%
# load 10Hz csv for vio mc
VIO_CSV = 'D:\\TestingData\\simulation\\10Hz\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'.csv'
vio_df = pd.read_csv(VIO_CSV, dtype={'SUBSET':str}, header='infer')
t_vio = vio_df['t']
L_vio = len(vio_df)
N_VIO = 7

# truth variables
N_truth_vio = vio_df['Y']
E_truth_vio = vio_df['X']
yaw_truth_vio = vio_df['yaw']
hitch_truth_vio = vio_df['hitch']
vy_truth_vio = vio_df['vy']
yaw_rate_truth_vio = vio_df['yaw_rate']
hitch_rate_truth_vio = vio_df['hitch_rate']

#%% 
# load trucksim mat file (for custom imu simulation)
TS_MAT = 'D:\\TestingData\\simulation\\raw\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'_TS.mat'
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

def add_noise_to_image(image:torch.Tensor, std=0.01, mean=0):
    """Adds noise to image for MC simulations"""
    # create guassian noise
    noise = torch.normal(mean, std, size=(image.shape[1], image.shape[2]))
    noisy_image = image + noise.unsqueeze(0)
    return noisy_image
    
#%%
# set up monte carlo loop variables and filter modules 
L_MC = 100

# storage variables (filter)
x_mc = np.zeros((N, L_MC, L)) # state
x_error_mc = np.zeros((N, L_MC, L)) # state errors
P_mc = np.zeros((N, N, L_MC, L))

# storage variables (vio)
x_mc_vio = np.zeros((N_VIO, L_MC, L_vio))
x_error_mc_vio =np.zeros((N_VIO, L_MC, L_vio))

# storage variables (ins)
N_INS = 3 # N, E, yaw
x_mc_ins = np.zeros((N_INS, L_MC, L))
x_error_mc_ins = np.zeros((N_INS, L_MC, L))

# create network model and dataloader
# === DATALOADER PARAMETERS ===
SEQ_LOOKBACK = 2
IMG_SIZE = (224,224)
BATCH_SIZE = 1

# === VIO MODEL PARAMETERS ===
VIO_WEIGHTS= "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\async_space_time_official\\sim_v0.pth"
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

# === HITCH MODEL PARAMETERS ===
HITCH_WEIGHTS = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\trailer_hitch\\sim_v1.pth"
HITCH_EMBED_DIM = 784

hitch_network_model = HitchModel(
    encoder=torchvision.models.mobilenet_v2(weights=None),
    embed_dim=HITCH_EMBED_DIM,
    dropout=0.
)
hitch_network_model = hitch_network_model.to(DEVICE)
# Load weights
hitch_state_dict = torch.load(HITCH_WEIGHTS)
hitch_network_model.load_state_dict(hitch_state_dict)

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
kf_estimator = Estimators(n=N ,m=M)

#%%
# monte carlo loop
for m in tqdm(range(0,L_MC)):
    # Generate new vehicle configs for every MC
    perturbed_vp = perturb_parameters(
        nominal_params=vp_dict,
        percentage=0.05,
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
    imu = simulate_imu_advanced( # MID GRADE
        lin_accel,
        ang_vel,
        accel_bias_sigma = (0.005, 0.005, 0.005),   # m/s² — 10x smaller, ~0.5 mg
        accel_bias_tau   = (500.0, 500.0, 500.0),   # seconds — slower decorrelation
        accel_rw_sigma   = (0.01, 0.01, 0.01), # m/s²/√s — velocity random walk
        gyro_bias_sigma = (0.000175, 0.000175, 0.000175),  # rad/s (about 0.1 deg/s or 360 deg/hr)
        gyro_bias_tau = (500.0, 500.0, 500.0),  # seconds (5 minutes)
        gyro_rw_sigma = (0.00035, 0.00035, 0.00035),  # rad/s (about 0.02 deg/s white noise)
        dt=dt,
        L=L,
    )
    # imu = simulate_imu_advanced( # LOW GRADE
    #     lin_accel,
    #     ang_vel,
    #     accel_bias_sigma = (0.005, 0.005, 0.005),   # m/s² — 10x smaller, ~0.5 mg
    #     accel_bias_tau   = (300.0, 300.0, 300.0),   # seconds — slower decorrelation
    #     accel_rw_sigma   = (0.05, 0.05, 0.05), # m/s²/√s — velocity random walk
    #     gyro_bias_sigma = (0.00175, 0.00175, 0.00175),  # rad/s (about 0.1 deg/s or 360 deg/hr)
    #     gyro_bias_tau = (300.0, 300.0, 300.0),  # seconds (5 minutes)
    #     gyro_rw_sigma = (0.0007, 0.0007, 0.0007),  # rad/s (about 0.02 deg/s white noise)
    #     dt=dt,
    #     L=L,
    # )
    
    # imu = simulate_imu(
    #     grade=2,
    #     accel=lin_accel,
    #     gyro=ang_vel,
    #     L=L
    # )
    # initialize (using truth)
    # simulated noisy lon vel measurement
    vx_meas = vx_truth + 0.01*np.random.randn(len(df))
    
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
        [yaw_truth_vio[0]],
        [hitch_truth_vio[0]],
        [vy_truth_vio[0]],
        [yaw_rate_truth_vio[0]],
        [hitch_rate_truth_vio[0]]
    ])
    x_truth_vio = x_vio
    x_error_vio = x_vio - x_truth_vio

    x_mc_vio[:,m,0] = x_vio.squeeze()
    x_error_mc_vio[:,m,0] = x_error_vio.squeeze()
    
    # initialize ins
    x_ins = np.array([
        [N_truth[0]],
        [E_truth[0]],
        [yaw_truth[0]]
    ])
    x_truth_ins = x_ins
    x_error_ins = x_ins - x_truth_ins
    
    x_mc_ins[:,m,0] = x_ins.squeeze()
    x_error_mc_ins[:,m,0] = x_error_ins.squeeze()
    
    
    # ==== INS LOOP ===
    imu_mech = ImuMech()
    C_ = body2rotm(yaw_truth[0], 0, 0, order='XYZ')
    v_ = np.array([[ve_truth[0]],
                        [vn_truth[0]],
                        [vz_truth[0]]])
    r_ = np.array([[E_truth[0]],
                        [N_truth[0]],
                        [0]])
    for i in range(0,L-1):
        # mechanize
        C_, v_, r_ = imu_mech.tanMech(C_, v_, r_, ref_lla=[0,0,0],
                                lin_accel=imu.accel[:,i+1],
                                ang_vel=imu.gyro[:,i+1],
                                T=dt,
                                simplified=True)
        att_ = rotm2eul(C_, order='XYZ')
        x_ins = np.array([
            [r_[1,0]],
            [r_[0,0]],
            [att_[2]]
        ])
        x_truth_ins = np.array([
            [N_truth[i+1]],
            [E_truth[i+1]],
            [yaw_truth[i+1]]
        ])
        x_error_ins = x_ins - x_truth_ins
        
        x_mc_ins[:,m,i+1] = x_ins.squeeze()
        x_error_mc_ins[:,m,i+1] = x_error_ins.squeeze()
        
        
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
            z, H, h_x, R = vx_measurement_module.generate_meas_model(x, vx_meas[k+1])
            x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)
            z, H, h_x, R = inertial_measurement_module.generate_meas_model(x, imu.gyro[2,k+1])
            x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)

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
            vx_meas[indices[0]:indices[-1]+1],
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

            # concatentate left and right images
            image_pairs = list(zip(right_images, left_images))
            concat_images = list(executor.map(cv2.hconcat, image_pairs))

            # apply torch transforms
            concat_images = list(executor.map(transform_img, concat_images))
            
            # add noise to images
            concat_images = list(executor.map(add_noise_to_image, concat_images))
            
        input_cam = torch.stack(concat_images).unsqueeze(0)
        curr_img = input_cam[:,-1] # grab current image for hitch model
        
        # visualize 
        # image = curr_img.squeeze(0).permute(1,2,0).numpy()
        # cv2.imshow("test", image)
        # cv2.waitKey(0)

        # cast inputs to device
        input_cam = input_cam.to(device=DEVICE, dtype=torch.float32)
        input_inert = input_inert.to(device=DEVICE, dtype=torch.float32)
        curr_img = curr_img.to(device=DEVICE, dtype=torch.float32)
        
        network_inputs = [input_cam, input_inert]

        # predict from model
        with torch.no_grad():
            vio_network_model.eval()
            hitch_network_model.eval()
            
            # call vio model
            trans_est, rot_est = vio_network_model(network_inputs)
            dx = trans_est.squeeze().cpu().numpy()[0]
            dy = trans_est.squeeze().cpu().numpy()[1]
            dyaw = rot_est.squeeze().cpu().numpy()
            
            # call hitch model
            hitch_est = hitch_network_model(curr_img)
            hitch_meas = hitch_est.squeeze().cpu().numpy()
            
            # propagate position and orientation from vio output
            pose_prev = (x_vio[1,0], x_vio[0,0], x_vio[2,0])
            east_meas, north_meas = _body_to_tangent_frame_translation(pose_prev, dx, dy)
            yaw_meas = x_vio[2,0] + dyaw

            # compute velocity measurements
            vy_meas = dy / 0.1
            yaw_rate_meas = dyaw / 0.1
            hitch_rate_meas = (hitch_meas - x_vio[3,0]) / 0.1
            
            # state
            x_vio = np.array([
                [north_meas],
                [east_meas],
                [yaw_meas],
                [hitch_meas],
                [vy_meas],
                [yaw_rate_meas],
                [hitch_rate_meas]
            ])
            # truth
            x_truth_vio = np.array([
                [N_truth_vio[j+1]],
                [E_truth_vio[j+1]],
                [yaw_truth_vio[j+1]],
                [hitch_truth_vio[j+1]],
                [vy_truth_vio[j+1]],
                [yaw_rate_truth_vio[j+1]],
                [hitch_rate_truth_vio[j+1]]
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

# extract statistics (mean/stds along mc dimension) for ins
x_mc_mean_ins = np.mean(x_mc_ins, axis=1)
x_mc_std_ins = np.std(x_mc_ins, axis=1)
x_error_mc_mean_ins = np.mean(x_error_mc_ins, axis=1)
x_error_mc_std_ins = np.std(x_error_mc_ins, axis=1)

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
    sigma_bound_fator=3,
    error_only=False,
    interactive=True
)

joint_ins_kf_vio_mc_plotter(
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
    x_mc_ins=x_mc_ins,
    x_error_mc_ins=x_error_mc_ins,
    x_mc_mean_ins=x_mc_mean_ins,
    x_mc_std_ins=x_mc_std_ins,
    x_error_mc_mean_ins=x_error_mc_mean_ins,
    x_error_mc_std_ins=x_error_mc_std_ins,
    t=t,
    sigma_bound_fator=3,
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

#%%
# save results
results = {
    'x_mc': x_mc,
    'x_error_mc':x_error_mc,
    'x_mc_mean':x_mc_mean,
    'x_mc_std':x_mc_std,
    'x_error_mc_mean':x_error_mc_mean,
    'x_error_mc_std':x_error_mc_std,
    'theo_std':theo_std,
    'x_mc_vio':x_mc_vio,
    'x_error_mc_vio':x_error_mc_vio,
    'x_mc_mean_vio':x_mc_mean_vio,
    'x_mc_std_vio':x_mc_std_vio,
    'x_error_mc_mean_vio':x_error_mc_mean_vio,
    'x_error_mc_std_vio':x_error_mc_std_vio,
    'x_mc_ins':x_mc_ins,
    'x_error_mc_ins':x_error_mc_ins,
    'x_mc_mean_ins':x_mc_mean_ins,
    'x_mc_std_ins':x_mc_std_ins,
    'x_error_mc_mean_ins':x_error_mc_mean_ins,
    'x_error_mc_std_ins':x_error_mc_std_ins,
}
if SAVE_RESULTS_PATH is not None:
    with open(SAVE_RESULTS_PATH, 'wb') as file:
        pickle.dump(results, file)
    print(f'MC Results saved to {SAVE_RESULTS_PATH}')
