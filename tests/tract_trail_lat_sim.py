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
import matplotlib.pyplot as plt
# import matlab.engine

from vehiclesim.tractor_trailer import TractorTrailer
from filter_tools.estimators import Estimators
from vehiclesim.imu_sim import *

#%% simulation setup

# call instances
# double lane change
veh_config_file = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\tests\\veh_config\\tractor_trailer\\5a_config.yaml'
ts_data_file = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\tests\\data\\Run103.csv'
dblc_inst = TractorTrailer(veh_config_file=veh_config_file, config_type='5a', ts_data_file=ts_data_file)

# load vehicle parameters
vp = dblc_inst.vp

# load trucksim data
ts_data = dblc_inst.ts_data

# simulation specs
t = ts_data.T_Event
L = len(t)
dt = 1/40   # trucksim sampling rate

# system inputs
steer_ang = np.deg2rad((ts_data.Steer_L1 + ts_data.Steer_R1)/2)
vx = ts_data.Vx*(1e3/3600)

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

    sysc_, sysd_ = dblc_inst.latModel(steer_ang=steer_ang[i], Vx=vx[i], dt=dt)
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
P_ = 10*np.eye(5)

# process noise
Q = np.array([[0.01, 0, 0, 0, 0],
              [0, 10, 0, 0, 0],
              [0, 0, 10, 0, 0],
              [0, 0, 0, 0.001, 0],
              [0, 0, 0, 0, 0.001]])

# measurement noise
R = np.array([[1e5, 0],
              [0, 1e-5]])

# call kalman filter from estimators class
kf_inst = Estimators(n=5,m=2)
for k in range(0,L):

    # time update
    sysc_, sysd_ = dblc_inst.latModel(steer_ang=steer_ang[k], Vx=vx[k], dt=dt)
    sysc_cl.append(sysc_)
    sysd_cl.append(sysd_)

    # model input
    u = np.array([steer_ang[k]])

    # imu measurements
    z = np.array([[tract_imu.accel[1][k]],
                  [tract_imu.gyro[2][k]]])
    
    # measurement map
    H = np.array([[0, vx[k], 0, 0, 0],
                  [0, 1, 0, 0, 0]])
    
    # kalman filter
    x_, P_, K_, innov_ = kf_inst.kf(T=dt,
                                    a=1,
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