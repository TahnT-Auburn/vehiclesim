#!/usr/bin/env python3
#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt
from scipy import signal

from vehiclesim.vehicle_configs.veh_params import vp as vp_dict
from vehiclesim.tractor_trailer import TractorTrailer

from nav_tools.imu_mechanization import *
from nav_tools.nav_utilities import *

from vehiclesim.measurement_simulations.imu_sim import simulate_imu
from vehiclesim.measurement_simulations.imu_sim_advanced import simulate_imu_advanced

SET = 'FF'
SUBSET = 'FF1'
#%%
# load csv data file
# CSV = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\simulation\\processed\\'+SET+'\\'+SUBSET+'\\'+SUBSET+'.csv'
CSV = 'C:\\Users\\pzt0029\\Documents\\Data\\Thesis\\TestingData\\experimental\\40Hz\\original\\6_19_25\\04\\04.csv'

df = pd.read_csv(CSV, dtype={'SUBSET':str}, header='infer')
L = len(df)
# sensor variables
steer_truth = df['steer_ang']
vx_truth = df['vx']
imu_gyro_z = df['imu_gyro_z']
# etalin variables for truth
N_truth = df['Y']
E_truth = df['X']
vy_truth = df['vy']
yaw_truth = df['yaw']
yaw_rate_truth = df['yaw_rate']
hitch_truth = df['hitch']
hitch_rate_truth = df['hitch_rate']
t = df['t']
dt = round(np.mean(np.diff(t)),3)

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
vz_truth = np.zeros(L)
ve_truth = np.cos(yaw_truth) * vx_truth - np.cos(yaw_truth) * vy_truth
vn_truth = np.sin(yaw_truth) * vx_truth + np.sin(yaw_truth) * vy_truth

#%%
#Simulate IMU

# imu = simulate_imu(1, lin_accel, ang_vel, L_ts)
# imu = simulate_imu_advanced(
#     lin_accel,
#     ang_vel,
#     accel_bias_sigma=(0.05, 0.05, 0.05),
#     accel_bias_tau = (300.0, 300.0, 300.0),  # seconds (5 minutes)
#     accel_rw_sigma = (0.002, 0.002, 0.002),  # m/s^2 (white noise)
#     gyro_bias_sigma = (0.005, 0.005, 0.005),  # rad/s (about 0.1 deg/s or 360 deg/hr)
#     gyro_bias_tau = (300.0, 300.0, 300.0),  # seconds (5 minutes)
#     gyro_rw_sigma = (0.0007, 0.0007, 0.0007),  # rad/s (about 0.02 deg/s white noise)
#     dt=dt,
#     L=L,
# )
# accel = imu.accel
# gyro = imu.gyro
# generate from experimental data
accel = np.array([[df['imu_accel_x']], [df['imu_accel_y']], [df['imu_accel_z']]]).squeeze()
gyro = np.array([[df['imu_gyro_x']], [df['imu_gyro_y']], [df['imu_gyro_z']]]).squeeze()

accel_filt = np.zeros_like(accel)
gyro_filt = np.zeros_like(gyro)

# filter signal
cutoff_freq = 10 # 10 Hz is still high for tractor-trailer dynamics
order = 3

sos = signal.butter(order, cutoff_freq, btype='low', analog=False, output='sos', fs=1/dt)
accel_filt[0,:] = signal.sosfiltfilt(sos, accel[0,:])
accel_filt[1,:] = signal.sosfiltfilt(sos, accel[1,:])
accel_filt[2,:] = signal.sosfiltfilt(sos, accel[2,:])
gyro_filt[0,:] = signal.sosfiltfilt(sos, gyro[0,:])
gyro_filt[1,:] = signal.sosfiltfilt(sos, gyro[1,:])
gyro_filt[2,:] = signal.sosfiltfilt(sos, gyro[2,:])

#%%
# Mechanize IMU
imu_mech = ImuMech()

# storage lists
C = []
att = []
v = []
r = []

# initialize
# C_init = body2rotm(np.deg2rad(ts_mat['Yaw'][0,0]), np.deg2rad(ts_mat['Pitch'][0,0]), np.deg2rad(ts_mat['Roll'][0,0]), order='XYZ')
# att_init = [np.deg2rad(ts_mat['Roll'][0,0]), np.deg2rad(ts_mat['Pitch'][0,0]), np.deg2rad(ts_mat['Yaw'][0,0])]
C_init = body2rotm(yaw_truth[0], 0, 0, order='XYZ')
att_init = [0, 0, yaw_truth[0]]
v_init = np.array([[ve_truth[0]],
                    [vn_truth[0]],
                    [vz_truth[0]]])
r_init = np.array([[N_truth[0]],
                    [E_truth[0]],
                    [0]])
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
    C_, v_, r_ = imu_mech.tanMech(C_, v_, r_, ref_lla=[0,0,0],
                                lin_accel=accel_filt[:,i+1],
                                ang_vel=gyro_filt[:,i+1],
                                T=dt,
                                simplified=True)
    # convert rotation matrices to euler angles
    # reset v to truth to test
    
    # v_ = np.array([[ve_truth[i+1]],
    #                 [vn_truth[i+1]],
    #                 [vz_truth[0]]])
    att_ = rotm2eul(C_, order='XYZ')
    C.append(C_)
    att.append(att_)
    v.append(v_)
    r.append(r_)

mech_att = np.array(att).transpose()
mech_vel = np.squeeze(np.array(v)).transpose()
mech_pos = np.squeeze(np.array(r)).transpose()

plt.figure('Att')
plt.suptitle('Attitude')
ax1 = plt.subplot(311)
ax1.plot(t, np.rad2deg(mech_att[0,:]), label='Mechanized')
ax1.plot(t, ts_mat['Roll'], label='TruckSim')
ax1.set_ylabel('Roll')
ax1.legend()
ax2 = plt.subplot(312)
ax2.plot(t, np.rad2deg(mech_att[1,:]), label='Mechanized')
ax2.plot(t, ts_mat['Pitch'], label='TruckSim')
ax2.set_ylabel('Pitch')
ax3 = plt.subplot(313)
ax3.plot(t, np.rad2deg(mech_att[2,:]), label='Mechanized')
ax3.plot(t, wrap_to(ts_mat['Yaw'], '180'), label='TruckSim')
ax3.set_ylabel('Yaw')
plt.tight_layout()
plt.show()

plt.figure('Vel')
plt.suptitle('Velocity')
ax1 = plt.subplot(311)
ax1.plot(t, mech_vel[0,:], label='Mechanized')
ax1.plot(t, ve_truth, label='TruckSim')
ax1.set_ylabel('VE')
ax1.legend()
ax2 = plt.subplot(312)
ax2.plot(t, mech_vel[1,:], label='Mechanized')
ax2.plot(t, vn_truth, label='TruckSim')
ax2.set_ylabel('VN')
ax3 = plt.subplot(313)
ax3.plot(t, mech_vel[2,:], label='Mechanized')
ax3.plot(t, vz_truth, label='TruckSim')
ax3.set_ylabel('Vz')
plt.tight_layout()
plt.show()

plt.figure('Pos')
plt.suptitle('Position')
ax1 = plt.subplot(311)
ax1.plot(t, mech_pos[0,:], label='Mechanized')
ax1.plot(t, E_truth, label='TruckSim')
ax1.set_ylabel('N')
ax1.legend()
ax2 = plt.subplot(312)
ax2.plot(t, mech_pos[1,:], label='Mechanized')
ax2.plot(t, N_truth, label='TruckSim')
ax2.set_ylabel('E')
ax3 = plt.subplot(313)
ax3.plot(t, mech_pos[2,:], label='Mechanized')
ax3.plot(t, ts_mat['Zo'], label='TruckSim')
ax3.set_ylabel('U')
plt.tight_layout()
plt.show()

plt.figure('Trajectory')
plt.suptitle('Trajectory')
plt.plot(mech_pos[0,:], mech_pos[1,:])
plt.plot(E_truth, N_truth)
plt.axis('equal')
plt.tight_layout()
plt.show()

# %%
