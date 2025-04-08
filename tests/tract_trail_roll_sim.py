'''
Tractor trailer roll simulation 
'''
#%%
import numpy as np
import matplotlib
matplotlib.use('ipympl')
import matplotlib.pyplot as plt
from scipy.linalg import expm

from vehiclesim.tractor_trailer import TractorTrailer
from vehiclesim.imu_sim import *

#%%
# simulation setup
veh_config_file = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\tests\\veh_config\\tractor_trailer\\5a_config.yaml'
ts_data_file = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\tests\\data\\30_mph_dblc.csv'
tract_trail = TractorTrailer(veh_config_file=veh_config_file,
                             ts_data_file=ts_data_file)

# load vehicle parameters
vp = tract_trail.vp

# load trucksim data
ts_data = tract_trail.ts_data

# simulation specs
t = ts_data.T_Event
L = len(t)
# dt = 1/40   # trucksim sampling rate
dt = np.mean(np.diff(ts_data.T_Event))

# generate tractor imu measurements
accel = [ts_data.Ax, ts_data.Ay, ts_data.Az] #g's
accel = [9.81*x for x in accel] # convert to m/s^2
gyro = np.deg2rad([ts_data.AVx, ts_data.AVy, ts_data.AVz])

accel2 = [ts_data.Ax_2, ts_data.Ay_2, ts_data.Az_2] #g's
accel2 = [9.81*x for x in accel2] # convert to m/s^2
gyro2 = np.deg2rad([ts_data.AVx_2, ts_data.AVy_2, ts_data.AVz_2])

tract_imu = imu_sim(2,accel,gyro,L)
trail_imu = imu_sim(2,accel2,gyro2,L)

g=9.81

#%%
# simulate nonlinear roll model

# initialize
phidd = 0
phid = 0
phi = 0
roll_rate = np.zeros(L)
roll = np.zeros(L)

for i in range(0,L-1):
    # call roll model
    phidd = tract_trail.rollModel('Trailer', phi, phid, ts_data.Ay_2[i]*9.81, hr=0.5)

    # euler integrate
    phid = phid + phidd*dt
    phi = phi + phid*dt

    roll_rate[i+1] = phid
    roll[i+1] = phi

#%%
# simulate linearized roll model
# initialize

lin_roll = np.zeros(L)
lin_roll_rate = np.zeros(L)
sysc, _ = tract_trail.linRollModel('Trailer', hr=0.91, dt=dt)
x_ = np.array([[0],[0]])
for j in range(0,L):
    u = ts_data.Ay_2[j]*9.81
    x_dot = sysc.A*x_ + sysc.B*u
    x_ = x_ + x_dot*dt
    
    lin_roll_rate[j] = x_[0]
    lin_roll[j] = x_[1]

#%%
# Roll Extended Kalman filter

# storage list
x = []
P = []
innov = []
K = []

# preallocate states
roll_cl = np.zeros(L)
roll_rate_cl = np.zeros(L)
hr_cl = np.zeros(L)

# initialize
phid = 0
phi = 0
x_ = np.array([[0],[0],[0]])

P_ = np.diag([3, 3, 3])

# process noise
Q = np.diag([0.00002**2, 0.0002**2, 0.08**2])

# measurement noise
R = np.array([[0.1**2, 0],
              [0, 0.05**2]])
# R = np.array([0.1**2])

roll_meas = 0.005*np.random.randn(L) + np.deg2rad(ts_data.Roll_2)
# roll_rate_meas = 0.005**2*np.random.randn(L) + np.deg2rad(ts_data.AVx_2)

unit = 'Trailer'
if unit == 'Tractor' or unit == 1:
    j_xx = vp.j_xx1
    m_s = vp.m_s1
    ks = vp.ks1*3
    c = vp.c1*3
elif unit == 'Trailer' or unit ==2:
    j_xx = vp.j_xx2
    m_s = vp.m_s2
    ks = vp.ks2*4
    c = vp.c2*4

for k in range(0,L):
    # time update
    phid = float(x_[0])
    phi = float(x_[1])
    hr = float(x_[2])

    Ay = trail_imu.accel[1,k]
    # Ay = 9.81*ts_data.AyBf_SM[k]
    phidd = tract_trail.rollModel(unit, phi=phi, phid=phid, Ay=Ay, hr=hr)
    phid = phid + phidd*dt
    phi = phi + phid*dt

    x_ = np.array([[phid],[phi],[hr]])

    # jacobian of nonlinear model
    F11 = (-1/2)*c*vp.ts**2*np.cos(phi) / (j_xx + m_s*hr**2)

    F12 = ((1/2)*c*vp.ts**2*np.sin(phi)*phid - m_s*Ay*hr*np.sin(phi) \
            + (m_s*g*hr - (1/2)*ks*vp.ts**2)*np.cos(phi)) / (j_xx + m_s*hr**2)

    F13 = -(2*m_s*hr*((-1/2)*c*vp.ts**2*np.cos(phi)*phid + m_s*Ay*hr*np.cos(phi) \
            + (m_s*g*hr - (1/2)*ks*vp.ts**2)*np.sin(phi)) / (j_xx + m_s*hr**2)**2) \
            + ((m_s*g*np.sin(phi) + m_s*Ay*np.cos(phi)) / (j_xx + m_s*hr**2))

    F = np.array([[F11, F12, F13],
                  [1, 0, 0],
                  [0, 0, 0]])
    
    # discretize
    Ad = expm(F*dt)

    # priori covariance prop
    P_ = Ad @ P_ @ Ad.transpose() + Q

    # measurement update
    # z = tract_imu.gyro[0,k]
    # z = np.deg2rad(ts_data.AVx[k])
    z = np.array([[trail_imu.gyro[0,k]],
                  [roll_meas[k]]])
    # z = np.array([roll_rate_meas[k]])

    # H = np.array([1, 0, 0])
    H = np.array([[1, 0, 0],
                  [0, 1, 0]])

    # kalman gain
    K_ = P_ @ H.transpose() @ np.linalg.inv(H @ P_ @ H.transpose() + R)    #Kalman gain
    # K_ = P_ @ H.reshape(-1,1) @ (H @ P_ @ H.reshape(-1,1) + R)**-1    #Kalman gain

    innov_ = (z - H @ x_)
    x_ = x_ + K_ @ (innov_)
    # x_ = x_ + (K_ * (innov_)).reshape(-1,1)
    P_ = (np.identity(3) - K_ @ H) @ P_ 

    x.append(x_)
    P.append(P_)
    innov.append(innov_)
    K.append(K_)

    roll_rate_cl[k] = x_[0]
    roll_cl[k] = x_[1]
    hr_cl[k] = x_[2]

#%%
# plots
ax1 = plt.subplot(211)
ax1.plot(t,ts_data.Roll_2, c='cyan', label='TruckSim')
ax1.plot(t, np.rad2deg(roll), c='k', label='Model')
# ax1.plot(t, np.rad2deg(lin_roll), label='Linearized Model')
# ax1.plot(t,np.rad2deg(roll_cl), label='KF')
ax1.legend()
ax1.set_ylabel('Roll Angle [deg]')
ax1.tick_params(axis='x',labelsize=13)
ax1.tick_params(axis='y',labelsize=13)

ax2 = plt.subplot(212)
ax2.plot(t,np.rad2deg(roll) - ts_data.Roll_2, c='k', label='Model')
# ax2.plot(t,np.rad2deg(lin_roll) - ts_data.Roll_2, label='Linearized Model')
# ax2.plot(t,np.rad2deg(roll_cl) - ts_data.Roll_2, label='KF')
ax2.legend()
ax2.set_ylabel('Error [deg]')
ax2.tick_params(axis='x',labelsize=13)
ax2.tick_params(axis='y',labelsize=13)

plt.tight_layout()
plt.show()

ax1 = plt.subplot(211)
ax1.plot(t,ts_data.AVx_2, c='cyan', label='TruckSim')
ax1.plot(t,np.rad2deg(roll_rate), c='k', label='Model')
# ax1.plot(t,np.rad2deg(roll_rate_cl), label='KF')
ax1.legend()
ax1.set_ylabel('Roll Rate [deg/s]')
ax1.tick_params(axis='x',labelsize=13)
ax1.tick_params(axis='y',labelsize=13)

ax2 = plt.subplot(212)
ax2.plot(t,np.rad2deg(roll_rate) - ts_data.AVx_2, c='k', label='Model')
# ax2.plot(t,np.rad2deg(roll_rate_cl) - ts_data.AVx_2, label='KF')
ax2.legend()
ax2.set_ylabel('Error [deg/s]')
ax2.tick_params(axis='x',labelsize=13)
ax2.tick_params(axis='y',labelsize=13)

plt.tight_layout()
plt.show()

ax1 = plt.subplot()
ax1.plot(t,hr_cl, label='KF')
ax1.legend()
ax1.set_ylabel('Roll Height [deg/s]')
ax1.tick_params(axis='x',labelsize=13)
ax1.tick_params(axis='y',labelsize=13)

plt.tight_layout()
plt.show()
# %%
