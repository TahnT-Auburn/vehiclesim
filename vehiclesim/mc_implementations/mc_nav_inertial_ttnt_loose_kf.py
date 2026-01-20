#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

from vehiclesim.state_modules.NavFullStateModule import NavFullStateModule
from vehiclesim.state_modules.NavZuptStateModule import NavZuptStateModule
from vehiclesim.measurement_modules.NavLonVelMeasModule import NavLonVelMeasModule
from vehiclesim.measurement_modules.NavInertialMeasModule import NavInertialMeasModule
from vehiclesim.measurement_modules.NavZuptInertialMeasModule import NavZuptInertialMeasModule
from vehiclesim.measurement_modules.NavTTNTMeasModule import NavTTNTMeasModule

from vehiclesim.measurement_simulations.imu_sim_advanced import simulate_imu_advanced

from trailer_pose_network.models.spacetime.async_st_ca_rn_trailer import AsyncSpaceTimeCrossAttentionResNet
from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader

from filter_tools.estimators import Estimators

from postprocessing.standard_mc_plotter import standard_mc_plotter

VEH_CONFIG = 'C:\\Users\\Tahn\\SoftDevel\\vehiclesim\\vehiclesim\\vehicle_configs\\5a_config.yaml'
SET = 'FF'
SUBSET = 'FF2'

#%%
# load csv data file
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
L = len(t)
N = 9 # number of filter states
M = 2 # number of measurements

# ---- load roots for dataloader ----
SEQ_ROOT_PROCESSED = 'D:\\TestingData\\simulation\\10Hz\\'+SET+'\\'+SUBSET+'\\'
SEQ_ROOT_RAW = 'D:\\TestingData\\simulation\\processed\\'+SET+'\\'+SUBSET+'\\'

# ---- load trucksim mat file (for custom imu simulation) ----
TS_MAT = 'D:\\TestingData\\simulation\\raw\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'_TS.mat'

TS_MAT = 'D:\\TestingData\\simulation\\raw\\FF\\FF2\\FF2_TS.mat'
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
# create network model and dataloader
# === DATALOADER PARAMETERS ===
SEQ_LOOKBACK = 2
IMG_SIZE = (224,224)
BATCH_SIZE = 1

# === MODEL PARAMETERS ===
WEIGHTS= "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\async_st_ca_rn_acc_yaw_trailer\\async_st_ca_rn_acc_yaw_trailer_v1.pth"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_DELTAS = 1
NUM_FRAMES = 2
NUM_IMU_SAMPLES = 5
EMBED_DIM = 384
NUM_HEADS = 8
DEPTH = 8
IN_CHANNELS = 3
IMU_CHANNELS = 8
DROPOUT = 0.
NUM_OUTPUTS = 3

dataset = AsyncTemporalDataLoader(
    sequence_root_processed=SEQ_ROOT_PROCESSED,
    sequence_root_raw=SEQ_ROOT_RAW,
    sequential_lookback=SEQ_LOOKBACK,
    inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':False},
    transform_img=v2.Compose([
        v2.ToPILImage(),
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]),
)

network_model = AsyncSpaceTimeCrossAttentionResNet(
    resnet_model=torchvision.models.resnet34(weights=None),
    resnet_model_hitch=torchvision.models.resnet34(weights=None),
    num_deltas=NUM_DELTAS,
    img_size=IMG_SIZE,
    seqential_lookback=SEQ_LOOKBACK,
    in_channels=IN_CHANNELS,
    embed_dim=EMBED_DIM,
    num_frames=NUM_FRAMES,
    num_imu_samples=NUM_IMU_SAMPLES,
    imu_channels=IMU_CHANNELS,
    num_heads=NUM_HEADS,
    depth=DEPTH,
    dropout=DROPOUT,
)
network_model = network_model.to(DEVICE)
# Load weights
state_dict = torch.load(WEIGHTS)
network_model.load_state_dict(state_dict)
    
# freeze batch norm layers
for module in network_model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        module.weight.requires_grad = False
        module.bias.requires_grad = False
            
#%%
# set up monte carlo loop variables and filter modules 
L_MC = 100

# storage variables
x_mc = np.zeros((N, L_MC, L)) # state
x_error_mc = np.zeros((N, L_MC, L)) # state errors
P_mc = np.zeros((N, N, L_MC, L))
# instantiate modules
standard_state_module = NavFullStateModule(
    error_model=np.diag([
        0.1,# N
        0.1,# E
        0.01,# vx    
        0.01,# vy
        0.00001,# yaw rate
        0.0001,# yaw
        0.00001,# hitch_rate
        0.0001,# hitch
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
vx_measurement_module = NavLonVelMeasModule(
    error_model=np.diag([
        1e-3
    ])
)
inertial_measurement_module = NavInertialMeasModule(
    error_model=np.diag([
        5e-3 # imu_gyro_z
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
    # refresh measurement module to reset initial states
    ttnt_measurement_module = NavTTNTMeasModule(
        network_model=network_model,
        init_states=[N_truth[0], E_truth[0], yaw_truth[0]],
        error_model = np.diag([
            1e0,
            1e0,
            1e-3,
            1e-3
        ])
    )
    # grade = random.randint(1,5)
    # setup variance variables (IMU for now)
    # TODO: Vary grade. Testing consumer grade only for now
    # simulate imu
    imu = simulate_imu_advanced(
        lin_accel,
        ang_vel,
        accel_bias_sigma=(0.05, 0.05, 0.05),
        accel_bias_tau = (300.0, 300.0, 300.0),  # seconds (5 minutes)
        accel_rw_sigma = (0.002, 0.002, 0.002),  # m/s^2 (white noise)
        gyro_bias_sigma = (0.005, 0.005, 0.005),  # rad/s (about 0.1 deg/s or 360 deg/hr)
        gyro_bias_tau = (100.0, 100.0, 100.0),  # seconds (5 minutes)
        gyro_rw_sigma = (0.005, 0.005, 0.005),  # rad/s (about 0.02 deg/s white noise)
        dt=dt,
        L=L,
    )

    steer_truth = steer_truth + np.deg2rad(0.5)*np.random.randn(L)
    vx_truth = vx_truth + 0.01*np.random.randn(L)

    x_ = [] # state
    x_error_ = [] # state error

    P_ = []
    innov_ = []
    K_ = []

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
    j = 0
    for k in range(0,L-1):
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
            z, H, h_x, R = inertial_measurement_module.generate_meas_model(x, imu.gyro[2,k+1])
            x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)
        # naive fusion for now - always take corrections when available to make timing easier
        if k !=0 and k % 4 == 0:
            idx = j
            # get inputs from dataloader
            inputs, _ = dataset.__getitem__(idx)
            # cast to device
            inputs[0] = inputs[0].to(device=DEVICE, dtype=torch.float32).unsqueeze(dim=0) # emulates a batchsize of 1
            inputs[1] = inputs[1].to(device=DEVICE, dtype=torch.float32).unsqueeze(dim=0) # emulates a batchsize of 1
            # call TTNT measurement module
            z, H, h_x, R = ttnt_measurement_module.generate_meas_model(x, inputs)
            x, P, innov, K = kf_estimator.kf_update(x, P, z, H, h_x, R)
            j += 1
            
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
        K_.append(K)
        innov_.append(innov)
        
        # populate mc variables
        x_mc[:,m,k+1] = x.squeeze()
        x_error_mc[:,m,k+1] = x_error.squeeze()

    # ---- end of filter loop (single MC) ----
# ---- end of mc loop ----

# extract statistics (mean/stds along mc dimension)
x_mc_mean = np.mean(x_mc, axis=1)
x_mc_std = np.std(x_mc, axis=1)
x_error_mc_mean = np.mean(x_error_mc, axis=1)
x_error_mc_std = np.std(x_error_mc, axis=1)

# extract theorethical std from filter covariance
P_array = np.array(P_)
theo_std = np.sqrt(np.diagonal(P_array, axis1=1, axis2=2).transpose())


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
