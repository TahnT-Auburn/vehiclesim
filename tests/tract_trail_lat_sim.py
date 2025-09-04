"""
#################### Tractor Trailer Lateral Simulation ####################

    Author: 
        Tahn Thawainin, AU GAVLAB
        pzt0029@auburn.edu

    Description: 
        A script to simulate a tractor-trailer's lateral states and propagate
        global position.
        Architecture includes a bicycle model time update and IMU/Camera measurement
        correction stage. ***Maybe link paper here***

############################################################################
"""
#%%
import numpy as np
import matplotlib
# matplotlib.use('ipympl')
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import DataLoader, TensorDataset, Sampler

# import matlab.engine

from vehiclesim.tractor_trailer import TractorTrailer
from filter_tools.estimators import Estimators
from nav_tools.imu_mechanization import *
from vehiclesim.imu_sim import *
from postprocessing.lat_sim_plotter import *
from postprocessing.calc_error_statics import *
from genNavMatrices import *

from trailer_pose_network.data_setup import TractorTrailerData

#%% simulation setup

# call instances
# double lane change
veh_config_file = 'C:\\Users\\Tahn\\SoftDevel\\vehiclesim\\tests\\veh_config\\tractor_trailer\\5a_config.yaml'
# ts_data_file = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\tests\\data\\30_mph_step_180.csv'
ts_data_file = 'D:\\TestingData\\simulation\\raw\\FF\\FF2\\FF2_TS.mat'
camera_file = 'D:\\TestingData\\simulation\\processed\\FF\\FF2\\FF2.csv'
tract_trail = TractorTrailer(veh_config_file=veh_config_file, config_type='5a', ts_data_file=ts_data_file)

# load vehicle parameters
vp = tract_trail.vp

# load trucksim data
ts_data = tract_trail.ts_data

# simulation specs
t = ts_data.T_Event
L = len(t)
# dt = 1/40   # trucksim sampling rate
dt = np.mean(np.diff(ts_data.T_Event))

# system inputs
# steer_ang = np.deg2rad((ts_data.Steer_L1 + ts_data.Steer_R1)/2)
axle_steer = np.deg2rad((ts_data.Steer_L1 + ts_data.Steer_R1)/2)
hand_steer = np.deg2rad(ts_data.Steer_SW*(1/25))
steer_ang = hand_steer
steer_thresh = np.deg2rad(0.5)

vx = ts_data.Vx*(1e3/3600)
vy = ts_data.Vy*(1e3/3600)
vz = ts_data.Vz*(1e3/3600)

# generate tractor imu measurements
accel = [ts_data.Ax, ts_data.Ay, ts_data.Az] #g's
accel = np.array([9.81*x for x in accel]) #.transpose(1,0) # convert to m/s^2
gyro = np.deg2rad(np.array([ts_data.AVx, ts_data.AVy, ts_data.AVz])) #.transpose(1,0)
# tract_imu = simulate_imu(accel, gyro, dt)
tract_imu = simulate_imu(1, accel, gyro,L)

#%%
# visualize IMU
ax1 = plt.subplot(311)
ax1.plot(t, tract_imu.accel[0])
ax1.plot(t, 9.81*ts_data.Ax)
ax1.set_ylabel('X Accel')
ax2 = plt.subplot(312)
ax2.plot(t, tract_imu.accel[1])
ax2.plot(t, 9.81*ts_data.Ay)
ax2.set_ylabel('Y Accel')
ax3 = plt.subplot(313)
ax3.plot(t, tract_imu.accel[2])
ax3.plot(t, 9.81*ts_data.Az)
ax3.set_ylabel('Z Accel')
plt.tight_layout()
plt.show()

ax1 = plt.subplot(311)
ax1.plot(t, np.rad2deg(tract_imu.gyro[0]))
ax1.plot(t, ts_data.AVx)
ax1.set_ylabel('X Gyro')
ax2 = plt.subplot(312)
ax2.plot(t, np.rad2deg(tract_imu.gyro[1]))
ax2.plot(t, ts_data.AVy)
ax2.set_ylabel('Y Gyro')
ax3 = plt.subplot(313)
ax3.plot(t, np.rad2deg(tract_imu.gyro[2]))
ax3.plot(t, ts_data.AVz)
ax3.set_ylabel('Z Gyro')
plt.tight_layout()
plt.show()

#%%
# Set up network configurations
# NET = 'mobilenetv2'
WEIGHTS = 'C:\\Users\\Tahn\\SoftDevel\\vehiclesim\\tests\weights\\mobilenetv2_weights.pth'
USECAMS = False

if USECAMS:

    # call model
    model = torchvision.models.mobilenet_v2(weights=False)
    num_features = model.classifier[1].in_features

    # modify network head
    model.classifier = nn.Sequential(             
    nn.Linear(num_features, 500),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(500,300),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(300,1)
    )

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print("Device in Use: %s" % device)

    # load model to device and set weights
    model = model.to(device)
    state_dict = torch.load(WEIGHTS)
    model.load_state_dict(state_dict)

    # setup dataloader
    dataset = TractorTrailerData(csv_file=camera_file,
                          output_states='hitch',
                          transform=transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((512,512)),
                            transforms.ToTensor()
                          ]),)

    class SkipFirstSampler(Sampler):
        def __init__(self, data_source):
            self.data_source = data_source

        def __iter__(self):
            return iter(range(1, len(self.data_source)))

        def __len__(self):
            return len(self.data_source) - 1
    
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # prepredict model estimates
    start_time = time.time()
    hitch_est_array = []
    with torch.no_grad():
        model.eval()
        for k, (img,y) in enumerate(tqdm(loader)):
            # start_time = time.time()
            # call neural network for TAA measurement
            img = img.to(device=device, dtype=torch.float32)
            hitch_est = model(img)
            hitch_est_array.append(hitch_est)
            # print(f"Model pass time: {time.time() - start_time},  Iter: {k}")
            # stop=1
    print(f"Model call total time: {time.time() - start_time}")

    start_time = time.time()
    nn_hitch = torch.stack([hitch.detach().cpu() for hitch in hitch_est_array]).numpy().squeeze()
    print(f"conversion time: {time.time() - start_time}")

#%%
# Mechanize IMU
imuMech = ImuMech()

# storage lists
C = []
att = []
v = []
r = []

# initialize
C_init = body2rotm(np.deg2rad(ts_data.Yaw[0]), np.deg2rad(ts_data.Pitch[0]), np.deg2rad(ts_data.Roll[0]), order='XYZ')
att_init = [np.deg2rad(ts_data.Roll[0]), np.deg2rad(ts_data.Pitch[0]), np.deg2rad(ts_data.Yaw[0])]
v_init = np.array([[vx[0]],
                    [vy[0]],
                    [vz[0]]])
r_init = np.array([[ts_data.XCG_SM[0]],
                    [ts_data.YCG_SM[0]],
                    [ts_data.ZCG_SM[0]]])
C_ = C_init
v_ = v_init
r_ = r_init
att_ = att_init

C.append(C_)
v.append(v_)
r.append(r_)
att.append(att_init)

for i in range(0,L-1):
    # mechanize
    C_, v_, r_ = imuMech.tanMech(C_, v_, r_, ref_lla=[0,0,0],
                                lin_accel=tract_imu.accel[:,i+1],
                                ang_vel=tract_imu.gyro[:,i+1],
                                T=dt,
                                simplified=True)
    # convert rotation matrices to euler angles
    att_ = rotm2eul(C_, order='XYZ')
    C.append(C_)
    att.append(att_)
    v.append(v_)
    r.append(r_)

mech_att = np.array(att).transpose()
mech_vel = np.squeeze(np.array(v)).transpose()
mech_pos = np.squeeze(np.array(r)).transpose()

# plot mechanization outputs
ax1 = plt.subplot(311)
ax1.plot(t, np.rad2deg(mech_att[0,:]), label='Mechanized')
ax1.plot(t, ts_data.Roll, label='TruckSim')
ax1.set_ylabel('Roll')
ax1.legend()
ax2 = plt.subplot(312)
ax2.plot(t, np.rad2deg(mech_att[1,:]), label='Mechanized')
ax2.plot(t, ts_data.Pitch, label='TruckSim')
ax2.set_ylabel('Pitch')
ax3 = plt.subplot(313)
ax3.plot(t, np.rad2deg(mech_att[2,:]), label='Mechanized')
ax3.plot(t, wrap_to(ts_data.Yaw, '180'), label='TruckSim')
ax3.set_ylabel('Yaw')
plt.tight_layout()
plt.show()

#%% simulate bicycle model

# storage lists
sysc_ol = []
x_ol = []
xdot_ol = []

# initialize
vy_ol = np.zeros(L)
yaw_rate_ol = np.zeros(L)
yaw_ol = np.zeros(L)
hitch_rate_ol = np.zeros(L)
hitch_ol = np.zeros(L)

x_ = np.array([[0],[0],[0],[0],[0]])
x_ol.append(x_)

vy_ol[0] = x_[0].item()
yaw_rate_ol[0] = x_[1].item()
yaw_ol[0] = x_[2].item()
hitch_rate_ol[0] = x_[3].item()
hitch_ol[0] = x_[4].item()

for i in range(0,L-1):

    # if abs(axle_steer[i]) <= steer_thresh:
    #     steer_ang[i] = axle_steer[i]
        
    sysc_, _ = tract_trail.latModel(steer_ang=steer_ang[i+1], Vx=vx[i+1], dt=dt)
    # sysc_, sysd_ = tract_trail.latModel(steer_ang=np.deg2rad(45), Vx=6.7, dt=dt)
    sysc_ol.append(sysc_)

    u = steer_ang[i]
    xdot_ = sysc_.A*x_ + sysc_.B*u
    xdot_ol.append(xdot_)

    x_ = x_ + xdot_*dt
    x_ol.append(x_)

    vy_ol[i+1] = x_[0].item()
    yaw_rate_ol[i+1] = x_[1].item()
    yaw_ol[i+1] = x_[2].item()
    hitch_rate_ol[i+1] = x_[3].item()
    hitch_ol[i+1] = x_[4].item()

ol_states = [vy_ol, yaw_rate_ol, yaw_ol, hitch_rate_ol, hitch_ol]

#%% kalman filter

# storage lists
sysc_cl = []
sysd_cl = []
x_cl = []
xdot_cl = []
P_list = []
innov = []
K = []

# preallocate states 
vy_cl = np.zeros(L)
yaw_rate_cl = np.zeros(L)
yaw_cl = np.zeros(L)
hitch_rate_cl = np.zeros(L)
hitch_cl = np.zeros(L)

# initialize
x_ = np.array([[0.0],[0.0],[0.0],[0.0],[0.0]])
x_cl.append(x_)

vy_cl[0] = x_[0].item()
yaw_rate_cl[0] = x_[1].item()
yaw_cl[0] = x_[2].item()
hitch_rate_cl[0] = x_[3].item()
hitch_cl[0] = x_[4].item()

P_ = np.diag(np.ones(5))
P_list.append(P_)

# call kalman filter from estimators class
start_time = time.time()
for k in range(0,L-1):   
    if USECAMS:
        # Call KF
        kf_inst = Estimators(n=5,m=3)
        # process noise
        Q = np.array([[1, 0, 0, 0, 0],
                    [0, 0.01, 0, 0, 0],
                    [0, 0, 0.1, 0, 0],
                    [0, 0, 0, 0.001, 0],
                    [0, 0, 0, 0, 0.001]])

        # measurement noise
        R = np.diag([1e2, 1e-3, 1e-2])

        # time update
        _, sysd_ = tract_trail.latModel(steer_ang=steer_ang[k+1], Vx=vx[k+1], dt=dt)
        A = sysd_.A
        B = sysd_.B

        # model input
        u = np.array([[steer_ang[k+1]]])    

        # start_time = time.time()
        z = np.array([[tract_imu.accel[1][k+1]],
                        [tract_imu.gyro[2][k+1]],
                        [nn_hitch[k+1]]])
        # print(f"CPU conversion2 time: {time.time()-start_time}")
        # stop=1

        # measurement map
        H = np.array([[0, vx[k+1], 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1]])
        
        # start_time = time.time()
        # kalman filter
        x_, P_, K_, innov_ = kf_inst.kf(T=dt,
                                        num_inputs=1,
                                        F=A,
                                        B=B,
                                        u=u,
                                        Q=Q,
                                        z=z,
                                        H=H,
                                        R=R,
                                        P=P_,
                                        x=x_)
        # print(f"KF time: {time.time()-start_time}")
        # stop=1

    else:
        kf_inst = Estimators(n=5,m=2)
        # process noise
        Q = np.array([[1, 0, 0, 0, 0],
                    [0, 0.01, 0, 0, 0],
                    [0, 0, 0.1, 0, 0],
                    [0, 0, 0, 0.001, 0],
                    [0, 0, 0, 0, 0.001]])

        # measurement noise
        R = np.array([[1e2, 0],
                [0, 1e-3]])
        
        # time update
        _, sysd_ = tract_trail.latModel(steer_ang=steer_ang[k+1], Vx=vx[k+1], dt=dt)
        sysc_cl.append(sysc_)
        sysd_cl.append(sysd_)

        # model input
        u = np.array([steer_ang[k+1]])    

        # imu measurements
        z = np.array([[tract_imu.accel[1][k+1]],
                    [tract_imu.gyro[2][k+1]]])

        # measurement map
        H = np.array([[0, vx[k+1], 0, 0, 0],
                    [0, 1, 0, 0, 0]])
    
        # kalman filter
        x_, P_, K_, innov_ = kf_inst.kf(T=dt,
                                        num_inputs=1,
                                        F=sysd_.A,
                                        B=sysd_.B,
                                        u=u,
                                        Q=Q,
                                        z=z,
                                        H=H,
                                        R=R,
                                        P=P_,
                                        x=x_)
    # populate states
    x_cl.append(x_)
    P_list.append(P_)
    K.append(K_)
    innov.append(innov_)

    vy_cl[k+1] = x_[0].item()
    yaw_rate_cl[k+1] = x_[1].item()
    yaw_cl[k+1] = x_[2].item()
    hitch_rate_cl[k+1] = x_[3].item()
    hitch_cl[k+1] = x_[4].item()

    # print(f"Total loop time: {time.time()-start_time}")
    # stop=1

# end = time.time() - start_time
# print(f"Time: {end}")
# stop=1

cl_states = [vy_cl, yaw_rate_cl, yaw_cl, hitch_rate_cl, hitch_cl]

#%%
##### position propagation #####

# absolute distance error helper function
def calc_abs_pos_error(truth, est):
    x_truth = truth[0]
    y_truth = truth[1]
    x_est = est[0]
    y_est = est[1]
    pos_error = np.sqrt((x_truth - x_est)**2 + (y_truth - y_est)**2)
    return pos_error

# initialize
X_mod = np.zeros(L)
Y_mod = np.zeros(L)
X_mod[0] = ts_data.XCG_SM[0]
Y_mod[0] = ts_data.YCG_SM[0]

X_kf = np.zeros(L)
Y_kf = np.zeros(L)
X_kf[0] = ts_data.XCG_SM[0]
Y_kf[0] = ts_data.YCG_SM[0]

for j in range(0,L-1):

    X_mod[j+1] = X_mod[j] + (vx[j]*np.cos(yaw_ol[j]) - vy_ol[j]*np.sin(yaw_ol[j]))*dt
    Y_mod[j+1] = Y_mod[j] + (vx[j]*np.sin(yaw_ol[j]) + vy_ol[j]*np.cos(yaw_ol[j]))*dt

    X_kf[j+1] = X_kf[j] + (vx[j]*np.cos(yaw_cl[j]) - vy_cl[j]*np.sin(yaw_cl[j]))*dt
    Y_kf[j+1] = Y_kf[j] + (vx[j]*np.sin(yaw_cl[j]) + vy_cl[j]*np.cos(yaw_cl[j]))*dt

truth_pos = [ts_data.XCG_SM, ts_data.YCG_SM]
imu_pos = [mech_pos[0], mech_pos[1]]
mod_pos = [X_mod, Y_mod]
kf_pos = [X_kf, Y_kf]

# calculate pos error
mod_pos_error = calc_abs_pos_error(truth_pos, mod_pos)
imu_pos_error = calc_abs_pos_error(truth_pos, imu_pos)
kf_pos_error = calc_abs_pos_error(truth_pos, kf_pos)

# trailer position propagation (geometric)
yaw_ol_2 = hitch_ol + yaw_ol    # kinematic trailer yaw
yaw_cl_2 = hitch_cl + yaw_cl

X_mod_2 = (X_mod - vp.c*np.cos(yaw_ol)) - vp.d*np.cos(yaw_ol_2)
Y_mod_2 = (Y_mod - vp.c*np.sin(yaw_ol)) - vp.d*np.sin(yaw_ol_2)

X_kf_2 = (X_kf - vp.c*np.cos(yaw_cl)) - vp.d*np.cos(yaw_cl_2)
Y_kf_2 = (Y_kf - vp.c*np.sin(yaw_cl)) - vp.d*np.sin(yaw_cl_2)

truth_pos_2 = [ts_data.XCG_SM2, ts_data.YCG_SM2]
mod_pos_2 = [X_mod_2, Y_mod_2]
kf_pos_2 = [X_kf_2, Y_kf_2]

mod_pos_error_2 = calc_abs_pos_error(truth_pos_2, mod_pos_2)
kf_pos_error_2 = calc_abs_pos_error(truth_pos_2, kf_pos_2)

#%%
##### Implement full navigation kalman filter #####

# storage list
sysd_nav = []
x_nav = []
xdot_nav = []
P_nav = []
innov_nav = []
K_nav = []

# preallocate states
X_nav = np.zeros(L)
vx_nav = np.zeros(L)
Y_nav = np.zeros(L)
vy_nav = np.zeros(L)
yaw_rate_nav = np.zeros(L)
yaw_nav = np.zeros(L)
hitch_rate_nav = np.zeros(L)
hitch_nav = np.zeros(L)
bias_ay_nav = np.zeros(L)
bias_ar_nav = np.zeros(L)

# initialize
x_ = np.array([[ts_data.XCG_SM[0]],[vx[0]],[ts_data.YCG_SM[0]],[0],[0],[0],[0],[0],[0],[0]])
x_nav.append(x_)

X_nav[0] = x_[0].item()
vx_nav[0] = x_[1].item()
Y_nav[0] = x_[2].item()
vy_nav[0] = x_[3].item()
yaw_rate_nav[0] = x_[4].item()
yaw_nav[0] = x_[5].item()
hitch_rate_nav[0] = x_[6].item()
hitch_nav[0] = x_[7].item()
bias_ay_nav[0] = x_[8].item()
bias_ar_nav[0] = x_[9].item()

P_ = np.diag(np.ones(10))
P_nav.append(P_)

for k in range(0,L-1):
    if USECAMS:
        # generate a KF instance
        kfnav = Estimators(n=10,m=4)
            # process noise
        Q = np.diag([0.1,                     # X
                    0.5,                    # vx    
                    0.1,                      # Y
                    1,                    # vy
                    0.01,                 # yaw rate
                    0.1,                  # yaw
                    0.001,                  # hitch_rate
                    0.001,                  # hitch
                    1e-10,                  # bias ay
                    1e-10])                 # bias ar

        # measurement noise
        R = np.diag([1e-3, 1e2, 1e-3, 1e-3])

        # time uppdate
        # call vehicle state model
        _, sysd_ = tract_trail.latModel(steer_ang=steer_ang[k+1], Vx=vx[k+1], dt=dt)

        # generate full navigation matrices
        A, B, H = genNavMatrices(A_veh=sysd_.A, B_veh=sysd_.B, vx=x_[1].item(), yaw=x_[5][0].item(), dt=dt, use_cams=USECAMS)

        # model input
        u = np.array([steer_ang[k+1]])

        # imu measurements
        z = np.array([[vx[k+1]],
                    [tract_imu.accel[1][k+1]],
                    [tract_imu.gyro[2][k+1]],
                    [nn_hitch[k+1]]])
        
        # warm up for initial P
        if k == 0:          
            for _ in range(100):
                _, P_, K_, innov_ = kfnav.kf(T=dt,
                                    num_inputs=1,
                                    F=A,
                                    B=B,
                                    u=u,
                                    Q=Q,
                                    z=z,
                                    H=H,
                                    R=R,
                                    P=P_,
                                    x=x_)
        # call KF    
        x_, P_, K_, innov_ = kfnav.kf(T=dt,
                                    num_inputs=1,
                                    F=A,
                                    B=B,
                                    u=u,
                                    Q=Q,
                                    z=z,
                                    H=H,
                                    R=R,
                                    P=P_,
                                    x=x_)

    else:
        # generate a KF instance
        kfnav = Estimators(n=10,m=3)

        # process noise
        Q = np.diag([0.1,                     # X
                    0.5,                    # vx    
                    0.1,                      # Y
                    1,                    # vy
                    0.01,                 # yaw rate
                    0.1,                  # yaw
                    0.001,                  # hitch_rate
                    0.001,                  # hitch
                    1e-5,                   # bias ay
                    1e-9])                 # bias ar

        # measurement noise
        R = np.diag([1e-3, 2e3, 2.5e-3])

        # time uppdate
        # call vehicle state model
        _, sysd_ = tract_trail.latModel(steer_ang=steer_ang[k+1], Vx=vx[k+1], dt=dt)

        # generate full navigation matrices
        A, B, H = genNavMatrices(A_veh=sysd_.A, B_veh=sysd_.B, vx=x_[1].item(), yaw=x_[5][0].item(), dt=dt)

        # model input
        u = np.array([steer_ang[k+1]])

        # imu measurements
        z = np.array([[vx[k+1]],
                    [tract_imu.accel[1][k+1]],
                    [tract_imu.gyro[2][k+1]]])
        
        # warm up for initial P
        if k == 0:          
            for _ in range(100):
                _, P_, K_, innov_ = kfnav.kf(T=dt,
                                    num_inputs=1,
                                    F=A,
                                    B=B,
                                    u=u,
                                    Q=Q,
                                    z=z,
                                    H=H,
                                    R=R,
                                    P=P_,
                                    x=x_)
        # call KF    
        x_, P_, K_, innov_ = kfnav.kf(T=dt,
                                    num_inputs=1,
                                    F=A,
                                    B=B,
                                    u=u,
                                    Q=Q,
                                    z=z,
                                    H=H,
                                    R=R,
                                    P=P_,
                                    x=x_)

    x_nav.append(x_)
    P_nav.append(P_)
    K_nav.append(K_)
    innov_nav.append(innov)

    X_nav[k+1] = x_[0].item()
    vx_nav[k+1] = x_[1].item()
    Y_nav[k+1] = x_[2].item()
    vy_nav[k+1] = x_[3].item()
    yaw_rate_nav[k+1] = x_[4].item()
    yaw_nav[k+1] = x_[5].item()
    hitch_rate_nav[k+1] = x_[6].item()
    hitch_nav[k+1] = x_[7].item()
    bias_ay_nav[k+1] = x_[8].item()
    bias_ar_nav[k+1] = x_[9].item()

nav_states = [X_nav, vx_nav, Y_nav, vy_nav, yaw_rate_nav, yaw_nav, hitch_rate_nav, hitch_nav, bias_ay_nav, bias_ar_nav]
nav_states_veh = [vy_nav, yaw_rate_nav, yaw_nav, hitch_rate_nav, hitch_nav]
nav_pos = [X_nav, Y_nav]
nav_pos_error = calc_abs_pos_error(truth_pos, nav_pos)


#%% Plots
# call plotter functions
truth_states = [ts_data.VyBf_SM*(1e3/3600),
                ts_data.AVz,
                ts_data.Yaw,
                ts_data.ArtR_H,
                ts_data.Art_H,
                ]

plot_states(t, truth_states, ol_states, cl_states, nav_states_veh, t_factor=60)

ax1 = plt.subplot(211)
ax1.plot(bias_ay_nav)
ax1.plot(tract_imu.bias.accel)
ax1.set_ylabel('Ay Bias')
ax2 = plt.subplot(212)
ax2.plot(bias_ar_nav)
ax2.set_ylabel('Yaw Rate Bias')
plt.tight_layout()
plt.show()

# plot_pos(t, truth_pos, 
#         imu_pos=imu_pos, imu_pos_error =imu_pos_error,
#         model_pos=mod_pos, mod_pos_error=mod_pos_error,
#         kf_pos=kf_pos,  kf_pos_error=kf_pos_error,
#         navkf_pos=nav_pos, navkf_pos_error=nav_pos_error)
# plot_pos(t, truth_pos_2, model_pos=mod_pos_2, kf_pos=kf_pos_2, mod_pos_error=mod_pos_error_2, kf_pos_error=kf_pos_error_2)

#%%
# prepare list of states for plots

truth_states = [
    [ts_data.XCG_SM, 'X Position', '[m]'],
    [ts_data.Vx*(1e3/3600), 'Body Frame Vx', '[m/s]'],
    [ts_data.YCG_SM, 'Y Position', '[m]'],
    [ts_data.Vy*(1e3/3600), 'Body Frame Vy', '[m/s]'],
    [ts_data.AVz, 'Yaw Rate', '[deg/s]'],
    [ts_data.Yaw, 'Yaw Angle', '[deg]'],
    [ts_data.ArtR_H, 'Hitch Rate', '[deg/s]'],
    [ts_data.Art_H, 'Hitch Angle', '[deg]'],
    
]

filter_states = [
    [X_nav, 'X Position', '[m]'],
    [vx_nav, 'Body Frame Vx', '[m/s]'],
    [Y_nav, 'Y Position', '[m]'],
    [vy_nav, 'Body Frame Vy', '[m/s]'],
    [np.rad2deg(yaw_rate_nav), 'Yaw Rate', '[deg/s]'],
    [np.rad2deg(yaw_nav), 'Yaw Angle', '[deg]'],
    [np.rad2deg(hitch_rate_nav), 'Hitch Rate', '[deg/s]'],
    [np.rad2deg(hitch_nav), 'Hitch Angle', '[deg]'],
]

# Get state errors
state_errors = calc_error_statics(truth_states, filter_states)
plot_error_and_sigma_bounds(t, state_errors, P_nav, std_factor=1)