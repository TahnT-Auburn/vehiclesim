'''
##########################################################
Vehicle Roll Simulation and EKF State Estimation

Author: Tahn Thawainin, AU GAVLAB
        pzt0029@auburn.edu

#########################################################
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
import scipy as sp

from filter_tools.estimators import Estimators

#%% Simulate Roll EOM

# simulation specs
dt = 0.001
t = np.arange(0, 10, dt)
L = len(t)

# model parameters
# ms = 2205       # sprung mass [kg]
# Ixx = 5512.5    # roll moment [kg-m**2]
# ts = 1.5        # suspension track width [m]
# hr = 0.7        # roll height [m]
# k = 2e4         # suspension stiffness [N/m]
# c = 3.9e2       # suspension damping
ms = 6493       # sprung mass [kg]
Ixx = 6875    # roll moment [kg-m**2]
ts = 1        # suspension track width [m]
hr = 0.1        # roll height [m]
k = 550e3         # suspension stiffness [N/m]
c = 25e3       # suspension damping
g = 9.81

# initialize
phi_ = []    # state arrays
phid_ = []

phi = 0     # state variables
phid = 0

# lateral acceleration input
Ay = 3*np.sin(1*np.pi*t)

for i in range(L):

    # nonlinear EOM
    phidd = (1 / (Ixx + ms*hr**2))*(ms*Ay[i]*np.cos(phi) + ms*g*hr*np.sin(phi) \
            - (1/2)*k*ts**2*np.sin(phi) - (1/2)*c*ts**2*np.cos(phi)*phid)
    
    phid = phid + phidd*dt

    phi = phi + phid*dt

    phid_.append(phid)
    phi_.append(phi)

#%% Roll State EKF

# initialize
x_ = []     # states
phi_hat = []
phid_hat = []
hr_hat = []

x = np.array([[0],[0],[0.1]])
x_.append(x)

P_ = []     # state covariance
P = np.diag((10,10,10))
P_.append(P)

# process noise
Q = np.diag((0.0002**2,0.0002**2,0.0001**2))

# measurement noise
R = np.diag((0.005**2,0.025**2))

# measurements
phid_meas = phid_ + 0.005*np.random.randn(len(t))
phi_meas = phi_ +  0.025*np.random.randn(len(t))
z = np.array([phid_meas,
              phi_meas])

Ay = Ay + 0.01*np.random.randn(len(t))

for k in range(L):

    # time update
    phid = float(x_[k][0])
    phi = float(x_[k][1])
    hr = float(x_[k][2])

    # propagate states using nonlinear EOM
    phidd = (1 / (Ixx + ms*hr**2))*(ms*Ay[k]*np.cos(phi) + ms*g*hr*np.sin(phi) \
            - (1/2)*k*ts**2*np.sin(phi) - (1/2)*c*ts**2*np.cos(phi)*phid)
    
    phid = phid + phidd*dt

    phi = phi + phid*dt

    x = np.array([phid,phi,hr]).reshape(-1,1)

    # jacobian of nonlinear model
    F11 = (-1/2)*c*ts**2*np.cos(phi) / (Ixx + ms*hr**2)
    F12 = ((1/2)*c*ts**2*np.sin(phi)*phid - ms*Ay[k]*hr*np.sin(phi) \
            + (ms*g*hr - (1/2)*k*ts**2)*np.cos(phi)) / (Ixx + ms*hr**2)
    F13 = (-2*ms*hr*((-1/2)*c*ts**2*np.cos(phi)*phid + ms*Ay[k]*hr*np.cos(phi) \
            + (ms*g*hr - (1/2)*k*ts**2)*np.sin(phi)) / (Ixx + ms*hr**2)**2) \
            + ((ms*g*np.sin(phi) + ms*Ay[k]*np.cos(phi)) / (Ixx + ms*hr**2))
    
    F = np.array([[float(F11), float(F12), float(F13)],
                  [1, 0, 0],
                  [0, 0, 0]])
    
    # discretiz
    Ad = sp.linalg.expm(F*dt)

    # priori covariance propagation
    P = Ad @ P @ np.transpose(Ad) + Q

    # measurement update
    H = np.array([[1, 0, 0],
                  [0, 1, 0]])
    
    # H = np.array([0, 1, 0])
    
    # print(f"Rank: {np.linalg.matrix_rank(obsv(Ad,H))}")

    # kalman gain
    K = P @ np.transpose(H) @ np.linalg.inv(H @ P @ np.transpose(H) + R)

    # covariance update
    P = (np.eye(3) - K @ H) @ P

    # innovation
    innov = (z[:,k].reshape(-1,1) - H @ x)

    # state update
    x = x + K @ innov

    # append
    x_.append(x)
    P_.append(P)
    phid_hat.append(float(x[0]))
    phi_hat.append(float(x[1]))
    hr_hat.append(float(x[2]))