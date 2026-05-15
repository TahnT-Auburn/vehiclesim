#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io
import random
from pathlib import Path
import pickle
from box import Box

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import cv2
from concurrent.futures import ThreadPoolExecutor
from vehiclesim.state_modules.NavFullStateModule import NavFullStateModule
from vehiclesim.state_modules.NavZuptStateModule import NavZuptStateModule
from vehiclesim.measurement_modules.NavLonVelMeasModule import NavLonVelMeasModule
from vehiclesim.measurement_modules.NavInertialMeasModule import NavInertialMeasModule
from vehiclesim.measurement_modules.NavZuptInertialMeasModule import NavZuptInertialMeasModule
from vehiclesim.measurement_modules.NavDLVIOMeasModule import NavDLVIOMeasModule
from vehiclesim.measurement_modules.NavDLHitchMeasModule import NavDLHitchMeasModule
from vehiclesim.measurement_modules.NavHitchMeasModule import NavHitchMeasModule
from vehiclesim.measurement_modules.NavVirtualGpsMeasModule import NavVirtualGpsMeasModule

from vehiclesim.measurement_simulations.imu_sim_advanced import simulate_imu_advanced
from vehiclesim.measurement_simulations.imu_sim import simulate_imu

from vehiclesim.vehicle_configs.veh_params import vp as vp_dict
from vehiclesim.mc_tools.mc_veh_config import perturb_parameters

from filter_tools.estimators import Estimators

from trailer_pose_network.models.spacetime.finalized.async_space_time_cross_attention import AsyncSpaceTimeCrossAttention
from trailer_pose_network.models.spacetime.async_st_ca_rn import AsyncSpaceTimeCrossAttentionResNet
from trailer_pose_network.models.spacetime.finalized.trailer_hitch_model import HitchModel
from trailer_pose_network.dataloaders.trailer_hitch_dataloader import HitchDataloader

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader

from nav_tools.imu_mechanization import ImuMech
from nav_tools.nav_utilities import body2rotm, rotm2eul

import matplotlib.pyplot as plt
import matplotlib as mpl

def run():
    # load data file 
    SET = '6_19_25'
    SUBSET = '05'
    CSV = 'D:\\TrainingData\\experimental\\40Hz\\original\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'.csv'
    df = pd.read_csv(CSV, dtype={'SUBSET':str}, header='infer')
    t_correction = df['t'].iloc[-1] // 2

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

    # construct IMU
    imu = Box({
        'accel': 'NaN',\
        'gyro': 'NaN',\
    })
    imu.accel = np.array([
        [df['imu_accel_x']],
        [df['imu_accel_y']],
        [df['imu_accel_z']]
    ]).squeeze()
    imu.gyro = np.array([
        [df['imu_gyro_x']],
        [df['imu_gyro_y']],
        [df['imu_gyro_z']]
    ]).squeeze()

    # velocities for ins
    vz_truth = np.zeros(L)
    ve_truth = np.cos(yaw_truth) * vx_truth - np.cos(yaw_truth) * vy_truth
    vn_truth = np.sin(yaw_truth) * vx_truth + np.sin(yaw_truth) * vy_truth

    # load 10Hz csv for vio mc
    VIO_CSV = 'D:\\TrainingData\\experimental\\10Hz\\original\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'.csv'
    vio_df = pd.read_csv(VIO_CSV, dtype={'SUBSET':str}, header='infer')
    L_vio = len(vio_df)
    t_vio = vio_df['t']

    ds_df = df.iloc[::4].reset_index(drop=True)
    N_truth_vio = ds_df['Y']
    E_truth_vio = ds_df['X']
    yaw_truth_vio = ds_df['yaw']
    hitch_truth_vio = ds_df['hitch']

    
    # utility functions
    def body_to_tangent_frame_translation(pose1, dx_body, dy_body):
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

    
    # === VIO/HITCH DATALOADER PARAMETERS ===
    SEQ_ROOT_PROCESSED = 'D:\\TrainingData\\experimental\\10Hz\\original\\'+SET+'\\'+SUBSET
    SEQ_ROOT_RAW = 'D:\\TrainingData\\experimental\\40Hz\\original\\'+SET+'\\'+SUBSET
    SEQ_LOOKBACK = 2
    IMG_SIZE = (224,224)
    BATCH_SIZE = 6
    NUM_WORKERS = 4

    # VIO DATALOADER
    test_set = AsyncTemporalDataLoader(sequence_root_processed=SEQ_ROOT_PROCESSED,
                                sequence_root_raw=SEQ_ROOT_RAW,
                                single_test=True,
                                sequential_lookback=SEQ_LOOKBACK,
                                inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':False},
                                transform_img=v2.Compose([
                                    v2.ToPILImage(),
                                    v2.Resize(IMG_SIZE),
                                    v2.ToTensor(),
                                ]),
                                # transform_img=v2.Compose([
                                #     v2.ToPILImage(),
                                #     v2.Resize(IMG_SIZE),
                                #     v2.ToImage(),
                                #     v2.ToDtype(torch.float32, scale=True),
                                #     v2.Normalize(
                                #         mean=[0.485, 0.456, 0.406],
                                #         std=[0.229, 0.224, 0.225]
                                #     ),
                                # ]),
                                preprocess_data=None,
                            )
    
    # HITCH DATALOADER
    HITCH_IMG_SIZE = (224,224)
    hitch_set = HitchDataloader(
        csv_root=SEQ_ROOT_PROCESSED,
        transforms=v2.Compose([
            v2.ToPILImage(),
            v2.Resize(HITCH_IMG_SIZE),
            v2.ToTensor(),
        ]),
    )
    
    # === VIO MODEL PARAMETERS ===
    VIO_WEIGHTS= "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\async_space_time_official\\exp_v1.pth"
    # VIO_WEIGHTS = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\async_st_ca_rn_acc_yaw\\async_st_ca_rn_acc_yaw_v4.pth"
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
    # vio_network_model = AsyncSpaceTimeCrossAttentionResNet(
    #     resnet_model=torchvision.models.resnet34(weights=None),
    #     num_deltas=1,
    #     img_size=IMG_SIZE,
    #     seqential_lookback=SEQ_LOOKBACK,
    #     in_channels=IN_CHANNELS,
    #     embed_dim=VIO_EMBED_DIM,
    #     num_frames=NUM_FRAMES,
    #     num_imu_samples=NUM_IMU_SAMPLES,
    #     imu_channels=IMU_CHANNELS,
    #     num_heads=NUM_HEADS,
    #     depth=DEPTH,
    #     dropout=DROPOUT,
    # )
    vio_network_model = vio_network_model.to(DEVICE)
    # Load weights
    vio_state_dict = torch.load(VIO_WEIGHTS)
    vio_network_model.load_state_dict(vio_state_dict)


    
    # === HITCH MODEL PARAMETERS ===
    HITCH_WEIGHTS = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\trailer_hitch\\exp_v0.pth"
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

    # === MODEL EKF === 

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
    virtual_gps_measurement_module = NavVirtualGpsMeasModule(
        error_model=np.diag([
            1e-3,
            1e-3,
            1e-5,
        ])
    )
    hitch_measurement_module = NavHitchMeasModule(
        error_model=np.diag([
            1e-3
        ])
    )
    # for aided ekf
    # dlvio_measurement_module = NavDLVIOMeasModule(
    #     network_model=vio_network_model,
    #     init_states=[N_truth[0], E_truth[0], yaw_truth[0]],
    #     error_model = np.diag([
    #         1e-3,
    #         5e-3,

    #     ])
    # )
    # dlhitch_measurement_module = NavDLHitchMeasModule(
    #     network_model=hitch_network_model,
    #     hitch_init=hitch_truth[0],
    #     error_model=np.diag([
    #         1e-3,
    #     ])
    # )
    
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
        vehicle_config=vp_dict,
    )
    kf_estimator = Estimators(n=N ,m=M)

    # === MODEL EKF ===

    # initialize filter
    x_ = []
    x_error_ = []
    P_ = []

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
    x_.append(x)
    x_error_.append(x_error)

    # ==== FILTER LOOP ====
    vx_last_set = False
    for k in tqdm(range(0,L-1)):
        # ---- ZUPT ----
        if np.isnan(vx_truth[k+1]):
            if not vx_last_set:
                vx_last = float(vx_truth[k])
                vx_last_set = True  
            vx_truth.iloc[k+1] = vx_last
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

        if df['t'].iloc[k] == t_correction:
            z, H, h_x, R = virtual_gps_measurement_module.generate_meas_model(x, N_truth[k+1], E_truth[k+1], yaw_truth[k+1])
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

        x_.append(x)
        x_error_.append(x_error)
        P_.append(P)
    x_filter_out = np.array(x_).squeeze()

    print("EKF Evaluation Complete")
    
    
    
    
    # === VIO NETWORK ===
    # intialize
    x_vio_ = []
    x_error_vio_ = []
    # state
    # initialize vio
    x_vio = np.array([
        [N_truth_vio[0]],
        [E_truth_vio[0]],
        [yaw_truth_vio[0]],
        [hitch_truth_vio[0]],
    ])
    x_truth_vio = x_vio
    x_error_vio = x_vio - x_truth_vio
    x_vio_.append(x_vio)
    x_error_vio_.append(x_error_vio)

    # === VIO LOOP === 

    # Generate loaders
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    hitch_loader = DataLoader(hitch_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # evalulate single model
    est_array = []
    truth_array = []
    print("Evaluating VIO Model ...")
    with torch.no_grad():
        vio_network_model.eval()
        # hitch_network_model.eval()

        for t, (x,y) in enumerate(tqdm(test_loader)):
            
            x[0] = x[0].to(device=DEVICE, dtype=torch.float32)
            x[1] = x[1].to(device=DEVICE, dtype=torch.float32)
            # curr_images = x[0][:,-1]
            # curr_images = curr_images.to(device=DEVICE, dtype=torch.float32)

            y = y.to(device=DEVICE, dtype=torch.float32)

            trans_est, rot_est = vio_network_model(x)
            # hitch_est = hitch_network_model(curr_images)

            est = torch.cat((trans_est, rot_est), dim=1)
            est_array.append(est)
            truth_array.append(y)
            
    est_array = torch.cat(est_array).cpu().numpy()
    truth_array = torch.cat(truth_array).cpu().numpy()

    print("VIO Evaluation Complete")

    # evalulate single model
    hitch_est_array = []
    hitch_truth_array = []
    print("Evaluating Hitch Model ...")
    with torch.no_grad():
        hitch_network_model.eval()

        for t, (x,y) in enumerate(tqdm(hitch_loader)):
            
            x = x.to(device=DEVICE, dtype=torch.float32)
            y = y.to(device=DEVICE, dtype=torch.float32)

            hitch_est = hitch_network_model(x)

            hitch_est_array.append(hitch_est)
            hitch_truth_array.append(y)
            
    hitch_est_array = torch.cat(hitch_est_array).cpu().numpy()
    hitch_truth_array = torch.cat(hitch_truth_array).cpu().numpy()
    
    print("Hitch Evaluation Complete")
    
    # compute position and yaw from deltas
    dx_body = est_array[:,0]
    dy_body = est_array[:,1]
    dyaw = est_array[:,2]
    hitch_est = hitch_est_array

    dx_body_truth = truth_array[:,0]
    dy_body_truth = truth_array[:,1]
    dyaw_truth = truth_array[:,2]
    # hitch_truth_ = truth_array[:,3]

    # df = pd.read_csv(TEST_CSV)
    X_est_array = []
    Y_est_array = []
    yaw_est_array = []
    X_est_array.insert(0,vio_df.iloc[0]["X"])
    Y_est_array.insert(0,vio_df.iloc[0]["Y"])
    yaw_est_array.insert(0,vio_df.iloc[0]["yaw"])

    for i in range(1,len(vio_df)):
        if vio_df['t'].iloc[i] == t_correction: # psuedo GPS correction
            X_est_array[i-1] = vio_df['X'].iloc[i-1]
            Y_est_array[i-1] = vio_df['Y'].iloc[i-1]
            yaw_est_array[i-1] = vio_df['yaw'].iloc[i-1]

        pose_prev = (X_est_array[i-1], Y_est_array[i-1], yaw_est_array[i-1])
        X_est, Y_est = body_to_tangent_frame_translation(pose_prev, dx_body=dx_body[i-1], dy_body=dy_body[i-1])
        yaw_est = yaw_est_array[i-1] + dyaw[i-1]
        
        X_est_array.append(float(X_est))
        Y_est_array.append(float(Y_est))
        yaw_est_array.append(float(yaw_est))

    stop=1
    x_vio_ = [Y_est_array, X_est_array, yaw_est_array, hitch_est.squeeze().tolist()]
    x_vio_out = np.array(x_vio_).squeeze().transpose()

    # # === AIDED EKF ===
    # CSV_10HZ = 'D:\\TrainingData\\experimental\\10Hz\\original\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'.csv'
    # df_10hz = pd.read_csv(CSV_10HZ, dtype={'SUBSET':str}, header='infer')
    
    # times_10hz = df_10hz['t']
    # mask  = df['t'].isin(times_10hz)
    # df_filt = df[mask]
    # L_filt = len(df_filt)
    # # sensor variables
    # steer_meas = df_filt['steer_ang'].to_numpy()
    # vx_truth = df_filt['vx'].to_numpy() + 0.01*np.random.randn(L_filt)
    # imu_gyro_z = df_filt['imu_gyro_z'].to_numpy()
    # # etalin variables for truth
    # N_truth = df_filt['Y'].to_numpy()
    # E_truth = df_filt['X'].to_numpy()
    # vy_truth = df_filt['vy'].to_numpy()
    # yaw_truth = df_filt['yaw'].to_numpy()
    # yaw_rate_truth = df_filt['yaw_rate'].to_numpy()
    # hitch_truth = df_filt['hitch'].to_numpy()
    # hitch_rate_truth = df_filt['hitch_rate'].to_numpy()
    # # other variables
    # vx_thresh = 0.5
    # t = df_filt['t']
    # t_correction = df_filt['t'].iloc[-1] // 2
    # dt = round(np.mean(np.diff(t)),3)
    
    # # yaw rate from VIO
    # yaw_rate_vio = dyaw.squeeze() / 0.1
    # yaw_rate_vio = np.insert(yaw_rate_vio, 0, df['yaw_rate'].iloc[0])
    # hitch_meas = hitch_est.squeeze()
    # # yaw_rate_vio = np.random.randn(L_filt)
    # # yaw_rate_vio = np.random.randn(L_filt)
    # # hitch_meas = np.random.randn(L_filt)
    
    # # storage list
    # x_ = []
    # P_ = []
    # innov_ = []
    # K_ = []

    # # initialize
    # x = np.array([
    #     [N_truth[0]],
    #     [E_truth[0]],
    #     [vx_truth[0]],
    #     [vy_truth[0]],
    #     [yaw_rate_truth[0]],
    #     [yaw_truth[0]],
    #     [hitch_rate_truth[0]],
    #     [hitch_truth[0]],
    #     [0]
    # ])
    # P = np.diag([
    #     0.05,# N
    #     0.05,# E
    #     0.001,# vx    
    #     0.01,# vy
    #     0.0001,# yaw rate
    #     0.001,# yaw
    #     0.0001,# hitch_rate
    #     0.001,# hitch
    #     1e-5 # bias ar
    # ])
    # x_.append(x)
    # P_.append(P)
    
    #     # filter loop
    # j = 0
    # inertial_inputs = []
    # model_preds = []
    # vx_last_set = False
    # yaw_prev = df['yaw'].iloc[0]
    # for k in tqdm(range(0,L_filt-1)):
    #         # use last vel if nan
    #     if np.isnan(vx_truth[k+1]):
    #         if not vx_last_set:
    #             vx_last = float(vx_truth[k])
    #             vx_last_set = True  
    #         vx_truth.iloc[k+1] = vx_last 
    #     # ---- ZUPT ----
    #     if vx_truth[k+1] <= vx_thresh:
    #         # time update
    #         PHI, F, G, Q = zupt_state_module.generate_state_model()
    #         u = np.array([[0]])
    #         x, P = kf_estimator.kf_predict(x, P, PHI, F, G, u, Q)

    #         # measurement update
    #         z, H, h_x, R = zupt_measurement_module.generate_meas_model(x, yaw_rate_vio[k])
    #         x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)

    #     # ---- STANDARD NAV STATE/MEASUREMENT MODEL ----
    #     else:
    #         # time update
    #         PHI, F, G, Q = standard_state_module.generate_state_model(steer_meas[k+1], x, dt)
    #         u = np.array([[steer_meas[k+1]]]) # single element array for matrix operation
    #         x, P = kf_estimator.kf_predict(x, P, PHI, F, G, u, Q)

    #         # measurement update
    #         z, H, h_x, R = vx_measurement_module.generate_meas_model(x, vx_truth[k+1])
    #         x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)
    #         z, H, h_x, R = inertial_measurement_module.generate_meas_model(x, yaw_rate_vio[k])
    #         x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)
        
    #         z, H, h_x, R = hitch_measurement_module.generate_meas_model(x, hitch_meas[k+1])
    #         x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)
        
        
    #     if df_filt['t'].iloc[k] == t_correction:
    #         z, H, h_x, R = virtual_gps_measurement_module.generate_meas_model(x, N_truth[k+1], E_truth[k+1], yaw_truth[k+1])
    #         x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)
    #         # yaw_prev = x[5,0] # update yaw previous as posteriori yaw state
            
    #     x_.append(x)
    #     P_.append(P)
    #     K_.append(K)
    #     innov_.append(innov)
    # x_aided_out = np.array(x_).squeeze()

    return x_filter_out, x_vio_out


if __name__ == "__main__":
    x_filter_out, x_vio_out = run()
    
    x_vio = np.linspace(0, 1, len(x_vio_out))
    # x_aided = np.linspace(0, 1, len(x_aided_out))
    x_ekf = np.linspace(0, 1, len(x_filter_out))

    N_vio_interp = np.interp(x_ekf, x_vio, x_vio_out[:,0])
    E_vio_interp = np.interp(x_ekf, x_vio, x_vio_out[:,1])
    yaw_vio_interp = np.interp(x_ekf, x_vio, x_vio_out[:,2])
    hitch_vio_interp = np.interp(x_ekf, x_vio, x_vio_out[:,3])

    # N_aided_interp = np.interp(x_ekf, x_aided, x_aided_out[:,0])
    # E_aided_interp = np.interp(x_ekf, x_aided, x_aided_out[:,1])
    # yaw_aided_interp = np.interp(x_ekf, x_aided, x_aided_out[:,2])
    # hitch_aided_interp = np.interp(x_ekf, x_aided, x_aided_out[:,3])

    N_filter = x_filter_out[:,0]
    E_filter = x_filter_out[:,1]
    yaw_filter = x_filter_out[:,5]
    hitch_filter = x_filter_out[:,7]

    # truth set 
    SET = '6_19_25'
    SUBSET = '05'
    CSV = 'D:\\TrainingData\\experimental\\40Hz\\original\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'.csv'
    df = pd.read_csv(CSV, dtype={'SUBSET':str}, header='infer')
    t = df['t']
    t_correction = df['t'].iloc[-1] // 2
    correction_index = df['t'][df['t'] == t_correction].index

    # compute absolute position errors
    def compute_abs_pos_error(coords1, coords2):
        X1,Y1 = coords1
        X2,Y2 = coords2
        
        error = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)
        return error
    abs_pos_error_filter = compute_abs_pos_error((df['X'], df['Y']), (E_filter, N_filter))
    abs_pos_error_vio = compute_abs_pos_error((df['X'], df['Y']), (E_vio_interp, N_vio_interp))
    # abs_pos_error_aided = compute_abs_pos_error((df['X'], df['Y']), (E_aided_interp, N_aided_interp))
    
    mpl.rcParams['font.size'] = 12          # General font size for all text elements
    mpl.rcParams['axes.labelsize'] = 12     # Font size for axis labels
    mpl.rcParams['xtick.labelsize'] = 12    # Font size for x-axis tick labels
    mpl.rcParams['ytick.labelsize'] = 12    # Font size for y-axis tick labels

    plt.figure()
    plt.subplot(211)
    plt.plot(df['X'], df['Y'], '--', linewidth=1.5, label='Truth')
    plt.plot(E_filter, N_filter, 'r', linewidth=1.5, label='EKF')
    # plt.plot(E_aided_interp, N_aided_interp, 'cyan', linewidth=1.5, label='EKF Aided')
    plt.plot(E_vio_interp, N_vio_interp, 'k', linewidth=1.5, label='DL-VIO')
    plt.plot(df['X'].iloc[correction_index], df['Y'].iloc[correction_index], 'o', color='cyan', markersize=5)
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.subplot(212)
    plt.plot(t, abs_pos_error_filter, 'r', linewidth=1.5, label='EKF')
    # plt.plot(t, abs_pos_error_aided, 'cyan', linewidth=1.5, label='EKF Aided')
    plt.plot(t, abs_pos_error_vio, 'k', linewidth=1.5, label='DL-VIO')
    plt.plot(t[correction_index], 0, 'o', color='cyan', markersize=5, label='GNSS')
    plt.xlabel('Time [s]')
    plt.ylabel('Absolute Position Error [m]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.legend()

    plt.figure()
    plt.subplot(211)
    plt.plot(t, np.rad2deg(df['yaw']), '--', linewidth=1.5, label='Truth')
    plt.plot(t, np.rad2deg(yaw_filter), 'r', linewidth=1.5, label='EKF')
    plt.plot(t, np.rad2deg(yaw_vio_interp), 'k', linewidth=1.5, label='DL-VIO')
    plt.plot(t[correction_index], np.rad2deg(df['yaw'].iloc[correction_index]), 'o', color='cyan', markersize=5)
    plt.xlabel('Time [s]')
    plt.ylabel('Yaw [deg]')
    plt.grid(True)
    plt.legend()
    plt.subplot(212)
    plt.plot(t, np.rad2deg(yaw_filter) - np.rad2deg(df['yaw']), 'r', linewidth=1.5, label='EKF')
    plt.plot(t, np.rad2deg(yaw_vio_interp) - np.rad2deg(df['yaw']) , 'k', linewidth=1.5, label='DL-VIO')
    plt.plot(t[correction_index], 0, 'o', color='cyan', markersize=5, label='GNSS')
    plt.xlabel('Time [s]')
    plt.ylabel('Yaw Error [deg]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.subplot(211)
    plt.plot(t, np.rad2deg(df['hitch']), '--', linewidth=1.5, label='Truth')
    plt.plot(t, np.rad2deg(hitch_filter), 'r', linewidth=1.5, label='EKF')
    plt.plot(t, np.rad2deg(hitch_vio_interp), 'k', linewidth=1.5, label='DL-VIO')
    plt.plot(t[correction_index], np.rad2deg(df['hitch'].iloc[correction_index]), 'o', color='cyan', markersize=5)
    plt.xlabel('Time [s]')
    plt.ylabel('Hitch [deg]')
    plt.grid(True)
    plt.legend()
    plt.subplot(212)
    plt.plot(t, np.rad2deg(hitch_filter) - np.rad2deg(df['hitch']), 'r', linewidth=1.5, label='EKF')
    plt.plot(t, np.rad2deg(hitch_vio_interp) - np.rad2deg(df['hitch']) , 'k', linewidth=1.5, label='DL-VIO')
    plt.plot(t[correction_index], 0, 'o', color='cyan', markersize=5, label='GNSS')
    plt.xlabel('Time [s]')
    plt.ylabel('Hitch Error [deg]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Compute error statistics
    def rmse(x_true, x_pred):
        '''
        Calculates root mean squared error (RMSE)
        '''
        return np.sqrt(np.mean((x_true - x_pred)**2))
    
    def abs_pos_rmse(abs_pos_error):
        return np.sqrt(np.mean(abs_pos_error**2))
    
    def find_time_before_error_exceeds(val, time, error):
        indices = [i for i in range(len(error) - 1) if error[i] < val <= error[i + 1]]
        # return [time[i] for i in indices]
        return time[indices[0]]

    # EKF stats
    abs_pos_rmse_filter = abs_pos_rmse(abs_pos_error_filter)
    north_rmse_filter = rmse(N_filter, df['Y'])
    east_rmse_filter = rmse(E_filter, df['X'])
    yaw_rmse_filter = np.rad2deg(rmse(yaw_filter, df['yaw']))
    hitch_rmse_filter = np.rad2deg(rmse(hitch_filter, df['hitch']))
    
    abs_pos_std_filter = np.std(abs_pos_error_filter)
    yaw_std_filter = np.std(np.rad2deg(yaw_filter - df['yaw']))
    hitch_std_filter = np.std(np.rad2deg(hitch_filter - df['hitch']))
    
    abs_pos_max_filter = np.max(abs_pos_error_filter)
    yaw_max_filter = np.max(np.rad2deg(yaw_filter - df['yaw']))
    hitch_max_filter = np.max(np.rad2deg(hitch_filter - df['hitch']))
    
    # vio stats
    abs_pos_rmse_vio = abs_pos_rmse(abs_pos_error_vio)
    north_rmse_vio = rmse(N_vio_interp, df['Y'])
    east_rmse_vio = rmse(E_vio_interp, df['X'])
    yaw_rmse_vio = np.rad2deg(rmse(yaw_vio_interp, df['yaw']))
    hitch_rmse_vio = np.rad2deg(rmse(hitch_vio_interp, df['hitch']))
    
    abs_pos_std_vio = np.std(abs_pos_error_vio)
    yaw_std_vio = np.std(np.rad2deg(yaw_vio_interp - df['yaw']))
    hitch_std_vio = np.std(np.rad2deg(hitch_vio_interp - df['hitch']))
    
    abs_pos_max_vio = np.max(abs_pos_error_vio)
    yaw_max_vio = np.max(np.rad2deg(yaw_vio_interp - df['yaw']))
    hitch_max_vio = np.max(np.rad2deg(hitch_vio_interp - df['hitch']))
    
    # time before difts
    filter_t_1m = find_time_before_error_exceeds(1, t, abs_pos_error_filter)
    filter_t_5m = find_time_before_error_exceeds(5, t, abs_pos_error_filter)
    filter_t_10m = find_time_before_error_exceeds(10, t, abs_pos_error_filter)
    
    vio_t_1m = find_time_before_error_exceeds(1, t, abs_pos_error_vio)
    vio_t_5m = find_time_before_error_exceeds(5, t, abs_pos_error_vio)
    vio_t_10m = find_time_before_error_exceeds(10, t, abs_pos_error_vio)
    
    print(f'Filter Error Statistics')
    print('=='*10)
    print(f'RMSES: Position: {abs_pos_rmse_filter}, Yaw: {yaw_rmse_filter}, Hitch: {hitch_rmse_filter}')
    print(f'STD: Position: {abs_pos_std_filter}, Yaw: {yaw_std_filter}, Hitch: {hitch_std_filter}')
    print(f'Max Error: Position: {abs_pos_max_filter}, Yaw: {yaw_max_filter}, Hitch: {hitch_max_filter}')
    
    print()
    
    print(f'VIO Error Statistics')
    print('=='*10)
    print(f'RMSES: Position: {abs_pos_rmse_vio}, Yaw: {yaw_rmse_vio}, Hitch: {hitch_rmse_vio}')
    print(f'STD: Position: {abs_pos_std_vio}, Yaw: {yaw_std_vio}, Hitch: {hitch_std_vio}')
    print(f'Max Error: Position: {abs_pos_max_vio}, Yaw: {yaw_max_vio}, Hitch: {hitch_max_vio}')
    
    print()
    print(f'Drift Time Statistics')
    print('=='*10)
    print(f'Filter Drift')
    print('--'*10)
    print(f'Times until: 1 meter: {filter_t_1m}, 5 meters: {filter_t_5m}, 10 meters: {filter_t_10m}')
    print(f'VIO Drift')
    print('--'*10)
    print(f'Times until: 1 meter: {vio_t_1m}, 5 meters: {vio_t_5m}, 10 meters: {vio_t_10m}')
    
    # save results log
    results = {
        'N_filter':N_filter,
        'E_filter':E_filter,
        'yaw_filter':yaw_filter, 
        'hitch_filter':hitch_filter, 
        'N_vio_interp':N_vio_interp, 
        'E_vio_interp':E_vio_interp, 
        'yaw_vio_interp':yaw_vio_interp, 
        'hitch_vio_interp':hitch_vio_interp,
    }
    SAVE_RESULTS_PATH = "C:\\Users\\Tahn\\SoftDevel\\vehiclesim\\evaluations\\exp_evals\\S5\\exp_results_S5.pkl"
    # SAVE_RESULTS_PATH = None
    if SAVE_RESULTS_PATH is not None:
        with open(SAVE_RESULTS_PATH, 'wb') as file:
            pickle.dump(results, file)
        print(f'Exp Results saved to {SAVE_RESULTS_PATH}')
    
    
    
# # ==== VIO LOOP ====
# for j in tqdm(range(0,L_vio-1)):
#     # extract the inertial measurements by matching times
#     t_to_find = [t_vio.iloc[j], t_vio.iloc[j+1]]
#     mask = t.isin(t_to_find)
#     indices = t[mask].index.tolist()
#     input_inert = np.array([
#         steer_truth[indices[0]:indices[-1]+1],
#         vx_truth[indices[0]:indices[-1]+1],
#         imu.accel[0, indices[0]:indices[-1]+1],
#         imu.accel[1, indices[0]:indices[-1]+1],
#         imu.accel[2, indices[0]:indices[-1]+1],
#         imu.gyro[0, indices[0]:indices[-1]+1],
#         imu.gyro[1, indices[0]:indices[-1]+1],
#         imu.gyro[2, indices[0]:indices[-1]+1]
#     ]).transpose()
#     input_inert = torch.tensor(input_inert).unsqueeze(0)

#     # extract images
#     left_paths = [vio_df['LRMC'].iloc[j], vio_df['LRMC'].iloc[j+1]]
#     right_paths = [vio_df['RRMC'].iloc[j], vio_df['RRMC'].iloc[j+1]]

#     with ThreadPoolExecutor(max_workers=32) as executor:
#         left_images = list(executor.map(load_image, left_paths))
#         right_images = list(executor.map(load_image, right_paths))

#         # concatentate left and right images
#         image_pairs = list(zip(right_images, left_images))
#         concat_images = list(executor.map(cv2.hconcat, image_pairs))

#         # apply torch transforms
#         concat_images = list(executor.map(transform_img, concat_images))
        
#         # add noise to images
#         concat_images = list(executor.map(add_noise_to_image, concat_images))
        
#     input_cam = torch.stack(concat_images).unsqueeze(0)
#     curr_img = input_cam[:,-1] # grab current image for hitch model
    
#     # visualize 
#     # image = curr_img.squeeze(0).permute(1,2,0).numpy()
#     # cv2.imshow("test", image)
#     # cv2.waitKey(0)

#     # cast inputs to device
#     input_cam = input_cam.to(device=DEVICE, dtype=torch.float32)
#     input_inert = input_inert.to(device=DEVICE, dtype=torch.float32)
#     curr_img = curr_img.to(device=DEVICE, dtype=torch.float32)
    
#     network_inputs = [input_cam, input_inert]

#     # predict from model
#     with torch.no_grad():
#         vio_network_model.eval()
#         hitch_network_model.eval()
        
#         # call vio model
#         trans_est, rot_est = vio_network_model(network_inputs)
#         dx = trans_est.squeeze().cpu().numpy()[0]
#         dy = trans_est.squeeze().cpu().numpy()[1]
#         dyaw = rot_est.squeeze().cpu().numpy()
        
#         # call hitch model
#         hitch_est = hitch_network_model(curr_img)
#         hitch_meas = hitch_est.squeeze().cpu().numpy()
        
#         # propagate position and orientation from vio output
#         pose_prev = (x_vio[1,0], x_vio[0,0], x_vio[2,0])
#         east_meas, north_meas = _body_to_tangent_frame_translation(pose_prev, dx, dy)
#         yaw_meas = x_vio[2,0] + dyaw

#         # compute velocity measurements
#         vy_meas = dy / 0.1
#         yaw_rate_meas = dyaw / 0.1
#         hitch_rate_meas = (hitch_meas - x_vio[3,0]) / 0.1
        
#         # state
#         x_vio = np.array([
#             [north_meas],
#             [east_meas],
#             [yaw_meas],
#             [hitch_meas],
#         ])
#         # truth
#         x_truth_vio = np.array([
#             [N_truth_vio[j+1]],
#             [E_truth_vio[j+1]],
#             [yaw_truth_vio[j+1]],
#             [hitch_truth_vio[j+1]],
#         ])
#         x_error_vio = x_vio - x_truth_vio
#         x_vio_.append(x_vio)
#         x_error_vio_.append(x_error_vio)

# %%
