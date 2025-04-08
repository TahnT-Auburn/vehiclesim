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
matplotlib.use('ipympl')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
# import matlab.engine

from vehiclesim.tractor_trailer import TractorTrailer
from filter_tools.estimators import Estimators
from vehiclesim.imu_sim import *
from lat_sim_plotter import *
from genNavMatrices import *

from trailer_pose_network.data_setup import TrailerData

#%% simulation setup

# call instances
# double lane change
veh_config_file = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\tests\\veh_config\\tractor_trailer\\5a_config.yaml'
ts_data_file = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\tests\\data\\camera_sets\\FF2_1_TS.csv'
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

for i in range(0,L):

    # if abs(axle_steer[i]) <= steer_thresh:
    #     steer_ang[i] = axle_steer[i]
        
    sysc_, _ = tract_trail.latModel(steer_ang=steer_ang[i], Vx=vx[i], dt=dt)
    # sysc_, sysd_ = tract_trail.latModel(steer_ang=np.deg2rad(45), Vx=6.7, dt=dt)
    sysc_ol.append(sysc_)

    u = steer_ang[i]
    xdot_ = sysc_.A*x_ + sysc_.B*u
    xdot_ol.append(xdot_)

    x_ = x_ + xdot_*dt
    x_ol.append(x_)

    vy_ol[i] = x_[0]
    yaw_rate_ol[i] = x_[1]
    yaw_ol[i] = x_[2]
    hitch_rate_ol[i] = x_[3]
    hitch_ol[i] = x_[4]

ol_states = [vy_ol, yaw_rate_ol, yaw_ol, hitch_rate_ol, hitch_ol]

#%% kalman filter

# generate tractor imu measurements
accel = [ts_data.Ax, ts_data.Ay, ts_data.Az] #g's
accel = [9.81*x for x in accel] # convert to m/s^2
gyro = np.deg2rad([ts_data.AVx, ts_data.AVy, ts_data.AVz])
tract_imu = imu_sim(1,accel,gyro,L)

# storage lists
sysc_cl = []
sysd_cl = []
x_cl = []
xdot_cl = []
P = []
innov = []
K = []

# preallocate states 
vy_cl = np.zeros(L)
yaw_rate_cl = np.zeros(L)
yaw_cl = np.zeros(L)
hitch_rate_cl = np.zeros(L)
hitch_cl = np.zeros(L)

# initialize
x_ = np.array([[0],[0],[0],[0],[0]])

P_ = np.diag(np.ones(5))

# process noise
Q = np.array([[1, 0, 0, 0, 0],
              [0, 0.01, 0, 0, 0],
              [0, 0, 0.1, 0, 0],
              [0, 0, 0, 0.001, 0],
              [0, 0, 0, 0, 0.001]])

# measurement noise
R = np.array([[1e2, 0],
              [0, 1e-3]])

# call kalman filter from estimators class
kf_inst = Estimators(n=5,m=2)
for k in range(0,L):

    # if abs(axle_steer[k]) <= steer_thresh:
    #     steer_ang[k] = axle_steer[k]

    # time update
    _, sysd_ = tract_trail.latModel(steer_ang=steer_ang[k], Vx=vx[k], dt=dt)
    sysc_cl.append(sysc_)
    sysd_cl.append(sysd_)

    # model input
    u = np.array([steer_ang[k]])    

    # imu measurements
    z = np.array([[tract_imu.accel[1][k]],
                  [tract_imu.gyro[2][k]]])
    # z = np.array([[vx[k]*tract_imu.gyro[2][k]],
    #               [tract_imu.gyro[2][k]]])
    
    # measurement map
    H = np.array([[0, vx[k], 0, 0, 0],
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
    
    x_cl.append(x_)
    P.append(P_)
    K.append(K_)
    innov.append(innov_)

    vy_cl[k] = x_[0]
    yaw_rate_cl[k] = x_[1]
    yaw_cl[k] = x_[2]
    hitch_rate_cl[k] = x_[3]
    hitch_cl[k] = x_[4]

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
mod_pos = [X_mod, Y_mod]
kf_pos = [X_kf, Y_kf]

# calculate pos error
mod_pos_error = calc_abs_pos_error(truth_pos, mod_pos)
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
x_ = np.array([[0],[vx[0]],[0],[0],[0],[0],[0],[0],[0],[0]])

P_ = np.diag(np.ones(10))

# process noise
Q = np.diag([3, 1, 3, 1, 0.01, 0.1, 0.001, 0.001, 0.05, 0.001])

# measurement noise
R = np.diag([1e-3, 1e2, 1e-3])

# generate a KF instance
kfnav = Estimators(n=10,m=3)

for k in range(0,L):
    # time uppdate
    # call vehicle state model
    _, sysd_ = tract_trail.latModel(steer_ang=steer_ang[k], Vx=vx[k], dt=dt)

    # generate full navigation matrices
    A, B, H = genNavMatrices(A_veh=sysd_.A, B_veh=sysd_.B, vx=vx[k], yaw=float(x_[5][0]), dt=dt)

    # model input
    u = np.array([steer_ang[k]])

    # imu measurements
    z = np.array([[vx[k]],
                  [tract_imu.accel[1][k]],
                  [tract_imu.gyro[2][k]]])
    
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

    X_nav[k] = x_[0]
    vx_nav[k] = x_[1]
    Y_nav[k] = x_[2]
    vy_nav[k] = x_[3]
    yaw_rate_nav[k] = x_[4]
    yaw_nav[k] = x_[5]
    hitch_rate_nav[k] = x_[6]
    hitch_nav[k] = x_[7]
    bias_ay_nav[k] = x_[8]
    bias_ar_nav[k] = x_[9]

nav_states = [X_nav, vx_nav, Y_nav, vy_nav, yaw_rate_nav, yaw_nav, hitch_rate_nav, hitch_nav, bias_ay_nav, bias_ar_nav]
nav_states_veh = [vy_nav, yaw_rate_nav, yaw_nav, hitch_rate_nav, hitch_nav]

#%%
# Kalman filter with hitch measurements
SET = "FF"
SUBSET = "FF2_1"
single_instance = {"cond": False,
                   "paths":["C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\testing\\processed\\"+SET+"\\"+SUBSET+"\\images\\LRMC\\",
                            "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\testing\\processed\\"+SET+"\\"+SUBSET+"\\images\\RRMC\\"]}

#%% Plots
# call plotter functions
truth_states = [ts_data.VyBf_SM*(1e3/3600),
                ts_data.AVz,
                ts_data.Yaw,
                ts_data.ArtR_H,
                ts_data.Art_H,
                ]

plot_states(t, truth_states, ol_states, cl_states, nav_states_veh, t_factor=60)
plot_pos(t, truth_pos, model_pos=mod_pos, kf_pos=kf_pos, mod_pos_error=mod_pos_error, kf_pos_error=kf_pos_error)
plot_pos(t, truth_pos_2, model_pos=mod_pos_2, kf_pos=kf_pos_2, mod_pos_error=mod_pos_error_2, kf_pos_error=kf_pos_error_2)

#%%
